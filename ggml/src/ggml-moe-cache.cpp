// SPDX-License-Identifier: MIT
// MoE per-expert GPU slot cache — Phase 1 implementation.
// See ggml-moe-cache.h for the public API + design notes.
//
// Layout: each (layer, bucket) gets its own backing ggml_backend_buffer.
// Required because models like Qwen3.6 MXFP4_MOE use mixed quantization
// across layers (q5_K, q6_K, mxfp4 mixed), so expert_size_bytes varies
// per layer. A single contiguous buffer with one size-per-bucket fails
// bind for every layer after the first that has a different quant.
// Trade-off: O(n_layers * n_buckets) buffer objects instead of 1 —
// at 40*4 = 160 buffers on Qwen3.6 the overhead is negligible.

#include "ggml-moe-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdarg>

// -----------------------------------------------------------------------------
// Internal types
// -----------------------------------------------------------------------------

struct ggml_moe_cache {
    ggml_backend_t backend     = nullptr;
    size_t total_bytes         = 0;          // sum of all allocated cell buffers

    int n_layers           = 0;
    int slots_per_bucket   = 0;
    size_t max_bytes_cap   = 0;

    // Per-(layer, bucket) cell. Indexed by [layer * GGML_MOE_BUCKET_COUNT + bucket].
    // Allocated lazily on first bind for that cell.
    //
    // Eviction policy: LFU (Least Frequently Used).
    //
    //   - Each slot keeps a `freq` counter that counts how many lookups have
    //     hit that slot since the slot was last (re-)populated.
    //   - On lookup hit: cell.freq[s]++
    //   - On miss: select the slot with the smallest freq. Tiebreak by lowest
    //     slot index (deterministic, has no perf consequence).
    //   - On populate (record_slot): reset cell.freq[s] = 1, since a freshly
    //     loaded expert hasn't been "used" yet beyond this miss-recovery.
    //
    // Why LFU here (vs LRU / SLRU): the S3/S4 page-in mechanism is most
    // dependent on cache adaptivity during the warmup phase. LFU concentrates
    // the slot pool on the genuinely-frequently-used experts (the routing's
    // long-tail Zipfian hot set), whereas pure LRU can thrash if a brief
    // sequence touches a string of one-off experts and evicts hot ones. SLRU
    // partially mitigates this with a probationary list, but its protection
    // is binary (in M or not); LFU integrates over the full access history.
    //
    // O(slots) eviction (linear scan of freq) is fine at slots <= 128 — a
    // single cache-line walk. A bucket-indexed structure could give O(1)
    // but would be more code and harder to validate; revisit if profiles
    // flag this loop.
    struct cell_state {
        bool   bound              = false;
        size_t expert_size_bytes  = 0;        // size of one expert weight in this cell
        size_t slot_stride        = 0;        // padded expert size, slot-aligned
        ggml_backend_buffer_t buf = nullptr;  // own backing buffer for this cell's slots
        uint8_t * base            = nullptr;  // buf base pointer
        std::vector<int32_t>  slot_map;       // [slots]: expert_id resident in each slot
        std::vector<uint32_t> freq;           // [slots]: hit count since populate
        int next_unused = 0;                  // slots [0, next_unused) have been touched
    };
    std::vector<cell_state> cells;            // n_layers * GGML_MOE_BUCKET_COUNT

    // Diagnostic env-var flags
    bool force_noop      = false;   // GGML_MOE_CACHE_FORCE_NOOP
    bool force_skip_even = false;   // GGML_MOE_CACHE_FORCE_SKIP_EVEN

    // Stats
    ggml_moe_cache_stats stats{};
};

// -----------------------------------------------------------------------------
// Logging helper
// -----------------------------------------------------------------------------

static void moe_cache_log(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "moe-cache: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// -----------------------------------------------------------------------------
// Bucket / layer identification from tensor name
// -----------------------------------------------------------------------------

static ggml_moe_bucket bucket_from_name(const char * name) {
    if (!name) return GGML_MOE_BUCKET_INVALID;
    // Order matters: "gate_up_exps" matches before "gate_exps" or "up_exps".
    if (strstr(name, "gate_up_exps")) return GGML_MOE_BUCKET_GATE_UP;
    if (strstr(name, "down_exps"))    return GGML_MOE_BUCKET_DOWN;
    if (strstr(name, "gate_exps"))    return GGML_MOE_BUCKET_GATE;
    if (strstr(name, "up_exps"))      return GGML_MOE_BUCKET_UP;
    return GGML_MOE_BUCKET_INVALID;
}

static int layer_from_name(const char * name) {
    if (!name) return -1;
    int layer = -1;
    if (sscanf(name, "blk.%d.", &layer) == 1 && layer >= 0) {
        return layer;
    }
    return -1;
}

static inline int cell_idx(int layer_idx, ggml_moe_bucket bucket) {
    return layer_idx * GGML_MOE_BUCKET_COUNT + (int) bucket;
}

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------

ggml_moe_cache_t ggml_moe_cache_init(
        ggml_backend_t backend,
        int            n_layers,
        int            slots_per_bucket,
        size_t         max_bytes) {
    if (!backend || n_layers <= 0 || slots_per_bucket <= 0) {
        return nullptr;
    }

    auto * c = new ggml_moe_cache();
    c->backend          = backend;
    c->n_layers         = n_layers;
    c->slots_per_bucket = slots_per_bucket;
    c->max_bytes_cap    = max_bytes;

    c->force_noop      = getenv("GGML_MOE_CACHE_FORCE_NOOP")      != nullptr;
    c->force_skip_even = getenv("GGML_MOE_CACHE_FORCE_SKIP_EVEN") != nullptr;

    // One cell per (layer, bucket). All per-slot vectors are sized here so
    // the LFU counters are valid before the first bind_bucket call (the
    // buffer itself stays unallocated until bind, but the bookkeeping is
    // cheap and lets bind_bucket be branch-free w.r.t. counter init).
    c->cells.resize((size_t) n_layers * GGML_MOE_BUCKET_COUNT);
    for (auto & cell : c->cells) {
        cell.slot_map.assign(slots_per_bucket, -1);
        cell.freq.assign(slots_per_bucket, 0);
    }

    if (c->force_noop) {
        moe_cache_log("GGML_MOE_CACHE_FORCE_NOOP set; cache will return miss on every lookup (diagnostic)");
    }
    if (c->force_skip_even) {
        moe_cache_log("GGML_MOE_CACHE_FORCE_SKIP_EVEN set; faking hits on even expert_ids (output GARBAGE; diagnostic only)");
    }

    moe_cache_log("init: %d layers x %d buckets, %d slots/bucket (LFU eviction); buffers allocated lazily per (layer, bucket)",
                  n_layers, (int) GGML_MOE_BUCKET_COUNT, slots_per_bucket);

    return c;
}

void ggml_moe_cache_free(ggml_moe_cache_t c) {
    if (!c) return;

    // Dump final stats so we can see whether the cache was actually being used.
    if (c->stats.total_lookups > 0) {
        const double hit_rate = (double) c->stats.total_hits / (double) c->stats.total_lookups;
        int bound_cells = 0;
        for (const auto & cell : c->cells) if (cell.bound) ++bound_cells;
        moe_cache_log("final stats: %lld lookups, %lld hits, %lld misses (%.1f%% hit rate); %d / %d cells bound; %.2f MiB total",
                      (long long) c->stats.total_lookups,
                      (long long) c->stats.total_hits,
                      (long long) c->stats.total_misses,
                      hit_rate * 100.0,
                      bound_cells, (int) c->cells.size(),
                      c->total_bytes / (1024.0 * 1024.0));
    } else {
        moe_cache_log("final stats: no lookups recorded — cache code path never executed");
    }

    for (auto & cell : c->cells) {
        if (cell.buf) {
            ggml_backend_buffer_free(cell.buf);
            cell.buf = nullptr;
        }
    }
    delete c;
}

// -----------------------------------------------------------------------------
// Lazy per-cell buffer allocation
// -----------------------------------------------------------------------------

static bool ensure_cell_allocated(ggml_moe_cache * c, ggml_moe_cache::cell_state & cell, size_t expert_size_bytes) {
    if (cell.bound) return true;

    // Pad each slot to 512 alignment for MMQ kernel safety (matches the existing
    // copy_experts padding in ggml-backend.cpp).
    const size_t slot_stride = ((expert_size_bytes + 511) / 512) * 512 + 512;
    const size_t total       = (size_t) c->slots_per_bucket * slot_stride;

    if (c->max_bytes_cap > 0 && c->total_bytes + total > c->max_bytes_cap) {
        moe_cache_log("cell alloc would push total to %zu bytes, exceeds cap of %zu",
                      c->total_bytes + total, c->max_bytes_cap);
        return false;
    }

    auto * buft = ggml_backend_get_default_buffer_type(c->backend);
    cell.buf = ggml_backend_buft_alloc_buffer(buft, total);
    if (!cell.buf) {
        moe_cache_log("failed to allocate %zu bytes for cell on backend", total);
        return false;
    }
    cell.base              = (uint8_t *) ggml_backend_buffer_get_base(cell.buf);
    cell.expert_size_bytes = expert_size_bytes;
    cell.slot_stride       = slot_stride;
    cell.bound             = true;
    c->total_bytes        += total;

    return true;
}

bool ggml_moe_cache_bind_bucket(
        ggml_moe_cache_t      c,
        int                    layer_idx,
        ggml_moe_bucket       bucket,
        size_t                 expert_size_bytes,
        int64_t                /*n_experts_total*/) {
    if (!c || bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return false;
    if (layer_idx < 0 || layer_idx >= c->n_layers)            return false;
    if (expert_size_bytes == 0)                                return false;

    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (cell.bound) {
        if (cell.expert_size_bytes != expert_size_bytes) {
            // Same (layer, bucket) cell observed at two different sizes —
            // shouldn't happen because tensor shape is fixed at load time.
            moe_cache_log("cell (layer=%d, bucket=%d) expert size changed: bound=%zu, got=%zu",
                          layer_idx, (int) bucket, cell.expert_size_bytes, expert_size_bytes);
            return false;
        }
        return true;
    }
    bool ok = ensure_cell_allocated(c, cell, expert_size_bytes);
    if (ok) {
        // Diagnostic: log the first few cell binds so we can verify the path is hit.
        static int n_bound = 0;
        if (n_bound < 8) {
            moe_cache_log("cell bound (layer=%d, bucket=%d, expert_size=%zu, slot_stride=%zu)",
                          layer_idx, (int) bucket, cell.expert_size_bytes, cell.slot_stride);
            ++n_bound;
        }
    }
    return ok;
}

// -----------------------------------------------------------------------------
// Tensor identification
// -----------------------------------------------------------------------------

bool ggml_moe_cache_identify_tensor(
        ggml_moe_cache_t        c,
        const struct ggml_tensor * input,
        int                       n_layers,
        int *                     out_layer_idx,
        ggml_moe_bucket *        out_bucket) {
    if (!c || !input || !out_layer_idx || !out_bucket) return false;
    if (c->slots_per_bucket <= 0)                       return false;

    const auto bucket = bucket_from_name(input->name);
    if (bucket == GGML_MOE_BUCKET_INVALID) return false;

    const int layer = layer_from_name(input->name);
    if (layer < 0 || layer >= n_layers) return false;

    *out_layer_idx = layer;
    *out_bucket    = bucket;
    return true;
}

// -----------------------------------------------------------------------------
// Lookup + slot selection — SLRU policy (Phase 1.3).
//
// All operations are O(1) via the SLRU helpers above. The linear scan in
// lookup() is O(slots) but is unavoidable without a per-cell hashmap; for
// slots_per_bucket <= 128 (the practical range) it's a single cache-line
// walk and faster than a hashmap probe.
// -----------------------------------------------------------------------------

int ggml_moe_cache_lookup(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int32_t expert_id) {
    if (!c) return -1;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.bound) return -1;

    c->stats.total_lookups++;

    // Periodic hit-rate progress so we can see the cache warming up even if
    // the process is killed before the destructor stats fire.
    if ((c->stats.total_lookups % 10000) == 0) {
        const double hit_rate = (double) c->stats.total_hits / (double) c->stats.total_lookups;
        moe_cache_log("progress: %lld lookups, %.1f%% hit rate, %.1f MiB allocated",
                      (long long) c->stats.total_lookups,
                      hit_rate * 100.0,
                      c->total_bytes / (1024.0 * 1024.0));
    }

    if (c->force_noop) {
        c->stats.total_misses++;
        return -1;
    }
    if (c->force_skip_even && (expert_id % 2 == 0)) {
        c->stats.total_hits++;
        return 0;
    }

    // Linear scan to locate the expert. With slots_per_bucket <= 128 this is
    // ~1 cache line of data per scan and dwarfs any hashmap overhead.
    for (int s = 0; s < c->slots_per_bucket; s++) {
        if (cell.slot_map[s] == expert_id) {
            c->stats.total_hits++;
            cell.freq[s]++;  // LFU: bump this slot's usage count
            return s;
        }
    }
    c->stats.total_misses++;
    return -1;
}

int ggml_moe_cache_select_slot_for_miss(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int32_t /*expert_id*/) {
    if (!c) return -1;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.bound) return -1;

    // Cold-start: take the next never-used slot.
    if (cell.next_unused < c->slots_per_bucket) {
        return cell.next_unused++;
    }

    // LFU: evict the slot with the smallest hit-count. Tiebreak by lowest
    // slot index (i.e., insertion order — irrelevant for hit-rate but
    // makes the policy deterministic). Linear scan; slots <= 128 in practice.
    int      victim_slot = 0;
    uint32_t victim_freq = cell.freq[0];
    for (int s = 1; s < c->slots_per_bucket; s++) {
        if (cell.freq[s] < victim_freq) {
            victim_freq = cell.freq[s];
            victim_slot = s;
        }
    }
    return victim_slot;
}

void ggml_moe_cache_record_slot(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int slot_idx, int32_t expert_id) {
    if (!c) return;
    if (slot_idx < 0 || slot_idx >= c->slots_per_bucket) return;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    cell.slot_map[slot_idx] = expert_id;
    // Reset freq for the newly-loaded expert. Counting starts from this
    // populate; freq is "hits since installation in this slot".
    cell.freq[slot_idx] = 1;
}

// -----------------------------------------------------------------------------
// Slot data pointer — layout is just (cell.base + slot * stride).
// -----------------------------------------------------------------------------

void * ggml_moe_cache_slot_data(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int slot_idx) {
    if (!c) return nullptr;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.bound || !cell.base) return nullptr;
    return cell.base + (size_t) slot_idx * cell.slot_stride;
}

// -----------------------------------------------------------------------------
// D2D copy primitive — actual CUDA call lives in ggml-cuda.cu and is looked up
// at runtime via the backend reg's proc address table. We can't link directly
// against ggml_cuda_moe_cache_d2d_copy_async because ggml-base and ggml-cuda
// are separate shared libraries: a strong reference would force ggml-base to
// link against ggml-cuda (breaking non-CUDA builds) and even with both built
// the cross-DSO strong symbol may not resolve. The reg proc address table is
// the standard ggml mechanism for backend-specific entry points.
// -----------------------------------------------------------------------------

typedef bool (*moe_d2d_copy_fn_t)(ggml_backend_t, void *, const void *, size_t);
typedef bool (*moe_wait_fn_t)(ggml_backend_t);

// Generic one-shot resolver; templated on the symbol name. The function
// pointer is cached after first lookup so we pay reg traversal cost once
// per process. Returns nullptr if the symbol isn't exported by this backend.
template <typename F>
static F resolve_backend_fn(ggml_backend_t backend, const char * symbol_name, F & cache_slot, bool & looked_up_slot) {
    if (looked_up_slot) return cache_slot;
    auto * dev = ggml_backend_get_device(backend);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        cache_slot = (F) ggml_backend_reg_get_proc_address(reg, symbol_name);
    }
    looked_up_slot = true;
    return cache_slot;
}

bool ggml_moe_cache_copy_d2d_async(
        ggml_backend_t backend, void * dst, const void * src, size_t size) {
    static moe_d2d_copy_fn_t cached_fn = nullptr;
    static bool              looked_up = false;
    auto fn = resolve_backend_fn(backend, "ggml_cuda_moe_cache_d2d_copy_async",
                                 cached_fn, looked_up);
    if (!fn) {
        // First-call warning only (looked_up_slot guards repeat).
        static bool warned = false;
        if (!warned) {
            warned = true;
            moe_cache_log("backend reg does not expose ggml_cuda_moe_cache_d2d_copy_async — "
                          "all copies will fall back to the existing H2D path (cache disabled effectively)");
        }
        return false;
    }
    return fn(backend, dst, src, size);
}

bool ggml_moe_cache_copy_async_on_copy_stream(
        ggml_backend_t backend, void * dst, const void * src, size_t size) {
    static moe_d2d_copy_fn_t cached_fn = nullptr;
    static bool              looked_up = false;
    auto fn = resolve_backend_fn(backend, "ggml_cuda_moe_cache_copy_async_on_copy_stream",
                                 cached_fn, looked_up);
    if (!fn) return false;
    return fn(backend, dst, src, size);
}

bool ggml_moe_cache_compute_wait_for_copies(ggml_backend_t backend) {
    static moe_wait_fn_t cached_fn = nullptr;
    static bool          looked_up = false;
    auto fn = resolve_backend_fn(backend, "ggml_cuda_moe_cache_compute_wait_for_copies",
                                 cached_fn, looked_up);
    if (!fn) return false;
    return fn(backend);
}

bool ggml_moe_cache_copy_stream_wait_for_compute(ggml_backend_t backend) {
    static moe_wait_fn_t cached_fn = nullptr;
    static bool          looked_up = false;
    auto fn = resolve_backend_fn(backend, "ggml_cuda_moe_cache_copy_stream_wait_for_compute",
                                 cached_fn, looked_up);
    if (!fn) return false;
    return fn(backend);
}

int ggml_moe_cache_n_layers(ggml_moe_cache_t c) {
    return c ? c->n_layers : 0;
}

// -----------------------------------------------------------------------------
// Stats accessors
// -----------------------------------------------------------------------------

size_t ggml_moe_cache_total_bytes(ggml_moe_cache_t c) {
    return c ? c->total_bytes : 0;
}

void ggml_moe_cache_get_stats(ggml_moe_cache_t c, ggml_moe_cache_stats * out) {
    if (!c || !out) return;
    *out = c->stats;
}

void ggml_moe_cache_reset_stats(ggml_moe_cache_t c) {
    if (!c) return;
    c->stats = {};
}
