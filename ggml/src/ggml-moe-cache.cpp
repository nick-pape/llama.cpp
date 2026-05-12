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
    // Eviction policy is Segmented-LRU (SLRU). Each cell maintains two
    // intrusive doubly-linked lists of slot indices:
    //
    //   - P (probationary): freshly-populated slots; capacity = ~20% of slots
    //   - M (protected):    slots that have been hit at least once after
    //                       entering P; capacity = remainder (~80%)
    //
    // All operations are O(1):
    //
    //   - On miss + select_slot_for_miss():
    //       * If a never-used slot exists, take it and add at P-head
    //       * Otherwise evict P-tail and re-add at P-head
    //   - On lookup hit, the hit slot is unlinked from its current list
    //     (P or M) and re-inserted at M-head. If M now exceeds its capacity,
    //     M-tail is demoted to P-head.
    //
    // Why SLRU over plain LRU: pure LRU evicts a slot on the next "stranger"
    // miss even if it was just used (one-hit wonder pollution). SLRU requires
    // two hits before a slot moves to the protected pool, so a single-use
    // expert can't displace something hot. On Qwen3.6-A3B's long-tail Zipfian
    // routing, this materially raised hit rate over pure LRU in the original
    // tinyserve / vLLM PR #37190 LFRU policy that inspired this design.
    //
    // O(slots) eviction (linear scan of last_tick) was the previous Phase 1.2
    // approach; with 16-128 slots it didn't show up in profiles, but rather
    // than keep the inefficiency we got the proper structure right while
    // rewriting for SLRU.
    struct cell_state {
        bool   bound              = false;
        size_t expert_size_bytes  = 0;        // size of one expert weight in this cell
        size_t slot_stride        = 0;        // padded expert size, slot-aligned
        ggml_backend_buffer_t buf = nullptr;  // own backing buffer for this cell's slots
        uint8_t * base            = nullptr;  // buf base pointer
        std::vector<int32_t>  slot_map;       // [slots]: expert_id resident in each slot

        // SLRU list bookkeeping. All vectors are sized to slots_per_bucket
        // when the cell is allocated. slot_list[s] is the list this slot
        // currently lives in (LIST_NONE / LIST_P / LIST_M).
        std::vector<int32_t> prev_in_list;    // [slot] -> prev slot in same list, -1 if head
        std::vector<int32_t> next_in_list;    // [slot] -> next slot in same list, -1 if tail
        std::vector<uint8_t> slot_list;       // [slot] -> LIST_*
        int p_head = -1, p_tail = -1, p_count = 0, p_capacity = 0;
        int m_head = -1, m_tail = -1, m_count = 0, m_capacity = 0;
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
// SLRU list ops (O(1) intrusive doubly-linked-list manipulation).
//
// Each cell owns two lists, P (probationary) and M (protected). Slots are
// identified by index in [0, slots_per_bucket). The slot's current list is
// recorded in slot_list[s]; prev/next pointers (-1 = end) live in
// prev_in_list[s] / next_in_list[s]. Head/tail/count/capacity are per-list
// fields on the cell.
// -----------------------------------------------------------------------------

static constexpr uint8_t SLRU_LIST_NONE = 0;
static constexpr uint8_t SLRU_LIST_P    = 1;
static constexpr uint8_t SLRU_LIST_M    = 2;

static inline void slru_unlink(ggml_moe_cache::cell_state & cell, int s) {
    const int p = cell.prev_in_list[s];
    const int n = cell.next_in_list[s];
    if (p >= 0) {
        cell.next_in_list[p] = n;
    } else if (cell.slot_list[s] == SLRU_LIST_P) {
        cell.p_head = n;
    } else if (cell.slot_list[s] == SLRU_LIST_M) {
        cell.m_head = n;
    }
    if (n >= 0) {
        cell.prev_in_list[n] = p;
    } else if (cell.slot_list[s] == SLRU_LIST_P) {
        cell.p_tail = p;
    } else if (cell.slot_list[s] == SLRU_LIST_M) {
        cell.m_tail = p;
    }
    if (cell.slot_list[s] == SLRU_LIST_P) {
        cell.p_count--;
    } else if (cell.slot_list[s] == SLRU_LIST_M) {
        cell.m_count--;
    }
    cell.prev_in_list[s] = -1;
    cell.next_in_list[s] = -1;
    cell.slot_list[s]    = SLRU_LIST_NONE;
}

static inline void slru_insert_head_p(ggml_moe_cache::cell_state & cell, int s) {
    cell.slot_list[s]    = SLRU_LIST_P;
    cell.prev_in_list[s] = -1;
    cell.next_in_list[s] = cell.p_head;
    if (cell.p_head >= 0) cell.prev_in_list[cell.p_head] = s;
    cell.p_head = s;
    if (cell.p_tail < 0) cell.p_tail = s;
    cell.p_count++;
}

static inline void slru_insert_head_m(ggml_moe_cache::cell_state & cell, int s) {
    cell.slot_list[s]    = SLRU_LIST_M;
    cell.prev_in_list[s] = -1;
    cell.next_in_list[s] = cell.m_head;
    if (cell.m_head >= 0) cell.prev_in_list[cell.m_head] = s;
    cell.m_head = s;
    if (cell.m_tail < 0) cell.m_tail = s;
    cell.m_count++;
}

// On hit: bring slot s to M-head. If s was in P, this is a promotion (may
// require demoting M-tail to P-head to keep M within capacity). If s was
// already in M, this is just a move-to-head within M.
static inline void slru_promote_to_m(ggml_moe_cache::cell_state & cell, int s) {
    const bool was_in_p = (cell.slot_list[s] == SLRU_LIST_P);
    slru_unlink(cell, s);
    slru_insert_head_m(cell, s);
    if (was_in_p && cell.m_count > cell.m_capacity) {
        // Demote M-tail back to P to keep M within its budget.
        const int demoted = cell.m_tail;
        if (demoted >= 0) {
            slru_unlink(cell, demoted);
            slru_insert_head_p(cell, demoted);
        }
    }
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
    // the SLRU pointers are valid before the first bind_bucket call (the
    // buffer itself stays unallocated until bind, but the bookkeeping is cheap
    // and lets bind_bucket be branch-free w.r.t. list initialization).
    c->cells.resize((size_t) n_layers * GGML_MOE_BUCKET_COUNT);
    // SLRU capacity split: 20% probationary, 80% protected, with a minimum
    // of 1 each so the structure makes sense at very small cache sizes.
    const int p_cap = std::max(1, slots_per_bucket / 5);
    const int m_cap = std::max(1, slots_per_bucket - p_cap);
    for (auto & cell : c->cells) {
        cell.slot_map.assign(slots_per_bucket, -1);
        cell.prev_in_list.assign(slots_per_bucket, -1);
        cell.next_in_list.assign(slots_per_bucket, -1);
        cell.slot_list.assign(slots_per_bucket, SLRU_LIST_NONE);
        cell.p_capacity = p_cap;
        cell.m_capacity = m_cap;
    }

    if (c->force_noop) {
        moe_cache_log("GGML_MOE_CACHE_FORCE_NOOP set; cache will return miss on every lookup (diagnostic)");
    }
    if (c->force_skip_even) {
        moe_cache_log("GGML_MOE_CACHE_FORCE_SKIP_EVEN set; faking hits on even expert_ids (output GARBAGE; diagnostic only)");
    }

    moe_cache_log("init: %d layers x %d buckets, %d slots/bucket (SLRU: %d protected + %d probationary); buffers allocated lazily per (layer, bucket)",
                  n_layers, (int) GGML_MOE_BUCKET_COUNT, slots_per_bucket, m_cap, p_cap);

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
            // Promotion to M-head: first hit moves the slot from P to M; a
            // second-or-later hit moves within M; either way the slot ends up
            // at M-head and (if necessary) one entry demotes from M to P.
            slru_promote_to_m(cell, s);
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

    int victim;
    if (cell.next_unused < c->slots_per_bucket) {
        // Cold-start: take the next never-used slot. No list cleanup needed.
        victim = cell.next_unused++;
    } else if (cell.p_tail >= 0) {
        // Normal eviction: P-tail (one-hit-wonder protection — these slots
        // never escaped probation, so they're the right things to evict).
        victim = cell.p_tail;
        slru_unlink(cell, victim);
    } else {
        // Defensive: if P is empty (which shouldn't happen with sensible
        // capacities), fall back to M-tail.
        victim = cell.m_tail;
        if (victim < 0) return -1;
        slru_unlink(cell, victim);
    }

    // The newly-selected victim slot enters P-head with its (about-to-be-
    // written) expert id. record_slot just sets slot_map; no further list
    // manipulation needed.
    slru_insert_head_p(cell, victim);
    return victim;
}

void ggml_moe_cache_record_slot(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int slot_idx, int32_t expert_id) {
    if (!c) return;
    if (slot_idx < 0 || slot_idx >= c->slots_per_bucket) return;
    c->cells[cell_idx(layer_idx, bucket)].slot_map[slot_idx] = expert_id;
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
