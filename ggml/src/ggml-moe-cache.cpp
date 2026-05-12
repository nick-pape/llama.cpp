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
    ggml_moe_cache_policy policy = GGML_MOE_CACHE_POLICY_LFRU;

    // Per-(layer, bucket) cell. Indexed by [layer * GGML_MOE_BUCKET_COUNT + bucket].
    // Allocated lazily on first bind for that cell.
    //
    // Eviction policy: LFRU (LFU with LRU tiebreak).
    //
    //   - Each slot keeps a `freq` counter (lookups that hit this slot) and
    //     a `last_tick` (monotonically-increasing access stamp).
    //   - On lookup hit: cell.freq[s]++; cell.last_tick[s] = ++c->lfru_tick
    //   - On populate (record_slot): freq=1, last_tick = ++c->lfru_tick.
    //     Newly-installed entries enter at the highest tick so they aren't
    //     immediately evicted when several slots share freq=1.
    //   - On miss-evict: scan for slot with smallest freq. If multiple slots
    //     tie at the same freq (common during warmup), tiebreak by smallest
    //     last_tick (i.e., LRU among the ties).
    //
    // Pure LFU without the LRU tiebreak is degenerate in practice: every
    // newly-populated slot has freq=1, every fresh miss sees a cell full of
    // freq=1 slots, and the tiebreaker decides everything. Vanilla LFU
    // tiebreaks by slot index → effectively evicts the just-populated slot
    // every time → hit rate collapses (measured 14% at cache=16 vs LRU's 48%).
    // The LRU tiebreak rescues this without giving up LFU's long-tail
    // protection of genuinely-frequently-used experts.
    //
    // O(slots) eviction (linear scan) is fine at slots <= 128 — a single
    // cache-line walk. A bucket-indexed structure could give O(1) but would
    // be more code and harder to validate; revisit if profiles flag this loop.
    struct cell_state {
        bool   bound              = false;
        size_t expert_size_bytes  = 0;        // size of one expert weight in this cell
        size_t slot_stride        = 0;        // padded expert size, slot-aligned
        ggml_backend_buffer_t buf = nullptr;  // own backing buffer for this cell's slots
        uint8_t * base            = nullptr;  // buf base pointer
        std::vector<int32_t>  slot_map;       // [slots]: expert_id resident in each slot
        std::vector<uint32_t> freq;           // [slots]: hit count since populate
        std::vector<uint64_t> last_tick;      // [slots]: tick of last access (LRU tiebreak)
        int next_unused = 0;                  // slots [0, next_unused) have been touched

        // S3 telemetry + adaptive offload routing.
        //   - n_lookups / n_hits          : lookup-level statistics for this cell
        //   - n_dispatches                : how many MUL_MAT_ID ops touched this cell
        //                                   (one per layer+bucket per token)
        //   - n_dispatches_to_cpu         : of those, how many were routed to CPU
        //                                   (the GPU offload was declined for cold cells)
        //   - misses_in_dispatch_sum      : sum of miss_count across all dispatches;
        //                                   avg-misses-per-op = .../n_dispatches
        // These are 64-bit so they don't wrap during a long decode.
        uint64_t n_lookups               = 0;
        uint64_t n_hits                  = 0;
        uint64_t n_dispatches            = 0;
        uint64_t n_dispatches_to_cpu     = 0;
        uint64_t misses_in_dispatch_sum  = 0;
    };
    std::vector<cell_state> cells;            // n_layers * GGML_MOE_BUCKET_COUNT
    uint64_t lfru_tick = 0;                   // monotonic stamp for LRU tiebreak

    // Pool-manager state (Phase 1+).
    //
    // Demand counters are per (layer, bucket, expert_id). They track how
    // many times the scheduler has missed-and-routed-to-CPU for each
    // expert, integrated over time with periodic decay. maintain() reads
    // them to decide which experts to admit into the GPU pool.
    //
    // n_experts_per_cell is set on first bind_bucket from the input
    // tensor's n_expert dimension. Until then we don't size the counters.
    //
    // Counter type: uint32_t. Saturates at 4G hints which is way more
    // than any realistic decode workload.
    int n_experts_per_cell = 0;
    std::vector<uint32_t> demand;             // [layers * buckets * n_experts]
    uint64_t maintain_calls = 0;              // cadence / decay tracking

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
        size_t         max_bytes,
        ggml_moe_cache_policy policy) {
    if (!backend || n_layers <= 0 || slots_per_bucket <= 0) {
        return nullptr;
    }

    auto * c = new ggml_moe_cache();
    c->backend          = backend;
    c->n_layers         = n_layers;
    c->slots_per_bucket = slots_per_bucket;
    c->max_bytes_cap    = max_bytes;
    c->policy           = (policy == GGML_MOE_CACHE_POLICY_DEFAULT)
                            ? GGML_MOE_CACHE_POLICY_LFRU : policy;

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
        cell.last_tick.assign(slots_per_bucket, 0);
    }

    if (c->force_noop) {
        moe_cache_log("GGML_MOE_CACHE_FORCE_NOOP set; cache will return miss on every lookup (diagnostic)");
    }
    if (c->force_skip_even) {
        moe_cache_log("GGML_MOE_CACHE_FORCE_SKIP_EVEN set; faking hits on even expert_ids (output GARBAGE; diagnostic only)");
    }

    const char * policy_name =
        c->policy == GGML_MOE_CACHE_POLICY_LRU  ? "LRU" :
        c->policy == GGML_MOE_CACHE_POLICY_LFRU ? "LFRU" : "unknown";
    moe_cache_log("init: %d layers x %d buckets, %d slots/bucket (%s eviction); buffers allocated lazily per (layer, bucket)",
                  n_layers, (int) GGML_MOE_BUCKET_COUNT, slots_per_bucket, policy_name);

    return c;
}

void ggml_moe_cache_free(ggml_moe_cache_t c) {
    if (!c) return;

    // Dump final stats so we can see whether the cache was actually being used.
    if (c->stats.total_lookups > 0) {
        const double hit_rate = (double) c->stats.total_hits / (double) c->stats.total_lookups;
        int bound_cells = 0;
        uint64_t total_dispatches      = 0;
        uint64_t total_dispatches_cpu  = 0;
        uint64_t total_misses_in_disp  = 0;
        for (const auto & cell : c->cells) {
            if (cell.bound) ++bound_cells;
            total_dispatches      += cell.n_dispatches;
            total_dispatches_cpu  += cell.n_dispatches_to_cpu;
            total_misses_in_disp  += cell.misses_in_dispatch_sum;
        }
        moe_cache_log("final stats: %lld lookups, %lld hits, %lld misses (%.1f%% hit rate); %d / %d cells bound; %.2f MiB total",
                      (long long) c->stats.total_lookups,
                      (long long) c->stats.total_hits,
                      (long long) c->stats.total_misses,
                      hit_rate * 100.0,
                      bound_cells, (int) c->cells.size(),
                      c->total_bytes / (1024.0 * 1024.0));
        if (total_dispatches > 0) {
            const uint64_t gpu_dispatches = total_dispatches - total_dispatches_cpu;
            const double cpu_pct = 100.0 * (double) total_dispatches_cpu / (double) total_dispatches;
            const double avg_misses_when_gpu = gpu_dispatches > 0
                ? (double) total_misses_in_disp / (double) gpu_dispatches
                : 0.0;
            moe_cache_log("dispatch breakdown: %lld total ops, %lld on CPU (%.1f%%), %lld on GPU (%.1f%%); avg misses-per-op when on GPU = %.2f",
                          (long long) total_dispatches,
                          (long long) total_dispatches_cpu, cpu_pct,
                          (long long) gpu_dispatches, 100.0 - cpu_pct,
                          avg_misses_when_gpu);
        }
        if (c->n_experts_per_cell > 0 && !c->demand.empty()) {
            // Summarize demand counters: how many distinct experts saw any
            // demand, and the top-N counts (handy when verifying that hint()
            // is firing and the distribution is Zipfian-ish).
            uint64_t total_hints = 0;
            size_t   nonzero     = 0;
            for (uint32_t d : c->demand) {
                total_hints += d;
                if (d > 0) ++nonzero;
            }
            moe_cache_log("pool-manager demand: %lld total hints across %zu / %zu (expert, cell) entries; %llu maintain() calls",
                          (long long) total_hints, nonzero, c->demand.size(),
                          (unsigned long long) c->maintain_calls);
        }
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
        int64_t                n_experts_total) {
    if (!c || bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return false;
    if (layer_idx < 0 || layer_idx >= c->n_layers)            return false;
    if (expert_size_bytes == 0)                                return false;

    // Lazy-init demand counters on first bind (we need n_experts to size them).
    // All cells in the model share the same n_experts dimension, so we only
    // need to do this once.
    if (c->n_experts_per_cell == 0 && n_experts_total > 0) {
        c->n_experts_per_cell = (int) n_experts_total;
        c->demand.assign((size_t) c->n_layers * GGML_MOE_BUCKET_COUNT * c->n_experts_per_cell, 0);
        moe_cache_log("pool manager: demand counters sized for %d layers x %d buckets x %d experts = %.1f KiB",
                      c->n_layers, (int) GGML_MOE_BUCKET_COUNT, c->n_experts_per_cell,
                      c->demand.size() * sizeof(uint32_t) / 1024.0);
    }

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
    cell.n_lookups++;
    for (int s = 0; s < c->slots_per_bucket; s++) {
        if (cell.slot_map[s] == expert_id) {
            c->stats.total_hits++;
            cell.n_hits++;                       // S3 telemetry
            cell.freq[s]++;                      // LFRU: bump usage count
            cell.last_tick[s] = ++c->lfru_tick;  // and recency stamp (LRU tiebreak)
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

    // Eviction depends on the cache's configured policy.
    int      victim_slot = 0;
    uint64_t victim_tick = cell.last_tick[0];
    uint32_t victim_freq = cell.freq[0];
    if (c->policy == GGML_MOE_CACHE_POLICY_LRU) {
        // LRU: evict slot with smallest last_tick.
        for (int s = 1; s < c->slots_per_bucket; s++) {
            if (cell.last_tick[s] < victim_tick) {
                victim_tick = cell.last_tick[s];
                victim_slot = s;
            }
        }
    } else {
        // LFRU: evict slot with smallest freq; tiebreak by smallest last_tick.
        for (int s = 1; s < c->slots_per_bucket; s++) {
            if (cell.freq[s] < victim_freq ||
                (cell.freq[s] == victim_freq && cell.last_tick[s] < victim_tick)) {
                victim_freq = cell.freq[s];
                victim_tick = cell.last_tick[s];
                victim_slot = s;
            }
        }
    }
    return victim_slot;
}

void ggml_moe_cache_record_slot(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int slot_idx, int32_t expert_id) {
    if (!c) return;
    if (slot_idx < 0 || slot_idx >= c->slots_per_bucket) return;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    cell.slot_map[slot_idx]  = expert_id;
    cell.freq[slot_idx]      = 1;                    // counts the access that caused this populate
    cell.last_tick[slot_idx] = ++c->lfru_tick;       // newest -> escapes immediate LRU eviction
}

// -----------------------------------------------------------------------------
// Pool manager (Phase 1+): hint + maintain
// -----------------------------------------------------------------------------

static inline size_t demand_idx(const ggml_moe_cache * c,
                                int layer_idx, ggml_moe_bucket bucket, int32_t expert_id) {
    return ((size_t) layer_idx * GGML_MOE_BUCKET_COUNT + (size_t) bucket)
           * (size_t) c->n_experts_per_cell + (size_t) expert_id;
}

void ggml_moe_cache_hint(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int32_t expert_id) {
    if (!c || c->n_experts_per_cell <= 0) return;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return;
    if (expert_id < 0 || expert_id >= c->n_experts_per_cell) return;

    uint32_t & d = c->demand[demand_idx(c, layer_idx, bucket, expert_id)];
    if (d != UINT32_MAX) d++;  // saturate at uint32 max
}

void ggml_moe_cache_maintain(ggml_moe_cache_t c) {
    if (!c) return;
    c->maintain_calls++;
    // Phase 1 stub: pool manager policy not yet implemented. Phase 4 will:
    //   1. Scan demand counters per cell
    //   2. Identify experts above admission threshold not already resident
    //   3. Issue async H2D on the copy stream (S1's stream)
    //   4. Update slot_map atomically after H2D completes
    //   5. Periodically decay counters (halve every K maintain() calls)
    //
    // Until then, the existing select_slot_for_miss + record_slot path from
    // compute_splits handles populate synchronously on the miss path. The
    // hint() data is being accumulated and is available for inspection.
}

// -----------------------------------------------------------------------------
// Phase 3: CPU dispatch for missed experts (scaffolding)
// -----------------------------------------------------------------------------

bool ggml_moe_cache_dispatch_cpu(
        ggml_moe_cache_t          c,
        ggml_backend_t            cpu_backend,
        const ggml_tensor *       expert_weights_host,
        const ggml_tensor *       src1_dev,
        const ggml_tensor *       src2_dev,
        int                       layer_idx,
        ggml_moe_bucket           bucket) {
    if (!c || !cpu_backend || !expert_weights_host || !src1_dev || !src2_dev) {
        return false;
    }
    (void) layer_idx; (void) bucket;  // reserved for per-cell timing in Phase 4

    // First-call debug: log what we're about to dispatch.
    static int dbg = 0;
    if (dbg < 3) {
        moe_cache_log("cpu dispatch entry [%d]: src0 type=%d ne=[%lld %lld %lld %lld] data=%p, "
                      "src1 type=%d ne=[%lld %lld %lld %lld] data=%p, "
                      "src2 type=%d ne=[%lld %lld %lld %lld] data=%p",
                      dbg,
                      (int) expert_weights_host->type,
                      (long long) expert_weights_host->ne[0], (long long) expert_weights_host->ne[1],
                      (long long) expert_weights_host->ne[2], (long long) expert_weights_host->ne[3],
                      expert_weights_host->data,
                      (int) src1_dev->type,
                      (long long) src1_dev->ne[0], (long long) src1_dev->ne[1],
                      (long long) src1_dev->ne[2], (long long) src1_dev->ne[3],
                      src1_dev->data,
                      (int) src2_dev->type,
                      (long long) src2_dev->ne[0], (long long) src2_dev->ne[1],
                      (long long) src2_dev->ne[2], (long long) src2_dev->ne[3],
                      src2_dev->data);
        ++dbg;
    }

    // Stepwise log to bisect crash. First-call only.
    #define DBG_STEP(s) do { if (dbg <= 1) moe_cache_log("cpu dispatch step: %s", s); } while(0)
    DBG_STEP("entering body");

    // We construct a mini graph on a dedicated context. Memory needed:
    //   ~6 * tensor headers (= ~6 * 400 bytes) plus the cgraph's internal
    //   arrays. Default cgraph size is 2048 nodes which would need >50 KiB
    //   of arrays alone; we use ggml_new_graph_custom with size=8 below to
    //   keep this tiny. 32 KiB is conservative.
    const size_t ctx_size = 32 * 1024;
    void * ctx_buf = malloc(ctx_size);
    if (!ctx_buf) {
        moe_cache_log("cpu dispatch: ctx alloc failed");
        return false;
    }
    DBG_STEP("ctx_buf allocated");
    ggml_init_params ip = { /*.mem_size=*/ctx_size, /*.mem_buffer=*/ctx_buf, /*.no_alloc=*/true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        free(ctx_buf);
        return false;
    }
    DBG_STEP("ggml_init done");

    const size_t src1_bytes = ggml_nbytes(src1_dev);
    const size_t src2_bytes = ggml_nbytes(src2_dev);
    DBG_STEP("nbytes computed");

    const int64_t ne01    = expert_weights_host->ne[1];
    const int64_t top_k   = src2_dev->ne[0];
    const int64_t n_toks  = src2_dev->ne[1];

    void * src1_host = aligned_alloc(64, (src1_bytes + 63) & ~63ULL);
    void * src2_host = aligned_alloc(64, (src2_bytes + 63) & ~63ULL);
    if (!src1_host || !src2_host) {
        moe_cache_log("cpu dispatch: aligned host alloc failed");
        free(src1_host); free(src2_host);
        ggml_free(ctx); free(ctx_buf);
        return false;
    }
    DBG_STEP("aligned alloc src1/src2");

    ggml_backend_tensor_get(src1_dev, src1_host, 0, src1_bytes);
    DBG_STEP("d2h src1");
    ggml_backend_tensor_get(src2_dev, src2_host, 0, src2_bytes);
    DBG_STEP("d2h src2");

    ggml_tensor * t_src0 = ggml_new_tensor(ctx, expert_weights_host->type, GGML_MAX_DIMS,
                                           expert_weights_host->ne);
    memcpy(t_src0->nb, expert_weights_host->nb, sizeof(t_src0->nb));
    DBG_STEP("new_tensor src0");
    t_src0->data = expert_weights_host->data;

    ggml_tensor * t_src1 = ggml_new_tensor(ctx, src1_dev->type, GGML_MAX_DIMS, src1_dev->ne);
    memcpy(t_src1->nb, src1_dev->nb, sizeof(t_src1->nb));
    DBG_STEP("new_tensor src1");
    t_src1->data = src1_host;

    ggml_tensor * t_src2 = ggml_new_tensor(ctx, src2_dev->type, GGML_MAX_DIMS, src2_dev->ne);
    memcpy(t_src2->nb, src2_dev->nb, sizeof(t_src2->nb));
    DBG_STEP("new_tensor src2");
    t_src2->data = src2_host;

    const size_t dst_bytes = ne01 * top_k * n_toks * sizeof(float);
    void * dst_host = aligned_alloc(64, (dst_bytes + 63) & ~63ULL);
    if (!dst_host) {
        moe_cache_log("cpu dispatch: dst alloc failed");
        free(src1_host); free(src2_host);
        ggml_free(ctx); free(ctx_buf);
        return false;
    }
    DBG_STEP("aligned alloc dst");

    int64_t dst_ne[GGML_MAX_DIMS] = { ne01, top_k, n_toks, 1 };
    ggml_tensor * t_dst = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, dst_ne);
    DBG_STEP("new_tensor dst");
    t_dst->data = dst_host;
    t_dst->op   = GGML_OP_MUL_MAT_ID;
    t_dst->src[0] = t_src0;
    t_dst->src[1] = t_src1;
    t_dst->src[2] = t_src2;
    DBG_STEP("set dst op + srcs");

    // Custom-size graph: tiny (8 nodes, no grad) instead of the default
    // 2048-node graph, which would blow our ctx budget.
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, /*size=*/8, /*grads=*/false);
    DBG_STEP("new_graph");
    ggml_build_forward_expand(gf, t_dst);
    DBG_STEP("build_forward_expand");

    // Dispatch on CPU backend. The strides-copy above is what makes this
    // work — ggml_new_tensor() computes default contiguous nb[] which
    // doesn't always match the original tensor's actual layout in the
    // backing buffer (mxfp4 / q5_K block strides aren't quite what the
    // default-stride formula produces in all cases). Without that
    // memcpy of nb[], graph_compute segfaults inside MUL_MAT_ID's row
    // walker reading garbage offsets.
    DBG_STEP("about to graph_compute");
    const int64_t t0 = ggml_time_us();
    enum ggml_status st = ggml_backend_graph_compute(cpu_backend, gf);
    DBG_STEP("graph_compute returned");
    const int64_t t1 = ggml_time_us();

    // Phase 3 scaffolding: the CPU result is in dst_host but NOT yet
    // integrated into the GPU output. Next session adds the H2D+merge
    // path. For now we just measure dispatch latency.
    static int n_calls   = 0;
    static int64_t total = 0;
    n_calls++;
    total += (t1 - t0);
    if (n_calls <= 8 || (n_calls % 200) == 0) {
        moe_cache_log("cpu dispatch [%d]: status=%d, %lldµs (running avg %.0fµs); op=[%lld x %lld x %lld]",
                      n_calls, (int) st, (long long)(t1 - t0), (double) total / n_calls,
                      (long long) ne01, (long long) top_k, (long long) n_toks);
    }

    free(dst_host);
    free(src1_host);
    free(src2_host);
    ggml_free(ctx);
    free(ctx_buf);

    return st == GGML_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------------
// S3 adaptive routing
// -----------------------------------------------------------------------------

bool ggml_moe_cache_should_offload_to_gpu(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket) {
    if (!c) return true;  // no cache => normal scheduler behavior
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return true;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return true;

    const auto & cell = c->cells[cell_idx(layer_idx, bucket)];

    // Not yet bound: let GPU run so the cache code path allocates this cell's
    // backing buffer on first touch. After that, normal warmup applies.
    if (!cell.bound) return true;

    // Warming up: slots are still being populated from never-touched. Force
    // GPU so the cache fills. Once full, fall through to the hit-rate check.
    if (cell.next_unused < c->slots_per_bucket) return true;

    // Need enough recent data to make a decision. With 8 lookups per dispatch
    // (top_k=8 routing), a few dispatches' worth of samples is plenty.
    if (cell.n_lookups < 64) return true;  // default GPU early on

    // Hit-rate threshold (env-tunable for experiments).
    // Default 0.5: if at least half of expert lookups in this cell are hits,
    // the cache D2D path beats the CPU path. Below that, CPU wins because it
    // avoids the per-miss PCIe stall.
    static const double threshold = []() {
        const char * env = getenv("GGML_MOE_CACHE_GPU_HIT_THRESHOLD");
        return env ? std::atof(env) : 0.5;
    }();

    const double hit_rate = (double) cell.n_hits / (double) cell.n_lookups;
    return hit_rate >= threshold;
}

void ggml_moe_cache_record_dispatch(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket,
        int miss_count, bool on_cpu) {
    if (!c) return;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return;

    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    cell.n_dispatches++;
    if (on_cpu) {
        cell.n_dispatches_to_cpu++;
        // miss_count is N/A for CPU dispatches (we never ran a lookup), so we
        // don't accumulate it.
    } else if (miss_count > 0) {
        cell.misses_in_dispatch_sum += (uint64_t) miss_count;
    }
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
