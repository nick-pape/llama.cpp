// SPDX-License-Identifier: MIT
// MoE expert-weight cache — v2 implementation (issue #20757).
//
// Architecture: persistent backend-owned slot pool per (layer, bucket).
// Routing ids are remapped from expert-ids to slot-ids; the scheduler
// repoints the MoE op's src[0] (expert weights) and src[2] (ids) to
// the cache's persistent tensors. Kernel reads slot pool directly,
// no D2D-into-input_cpy per hit.
//
// Buffers are allocated via ggml_backend_buft_alloc_buffer (NOT
// ggml_gallocr), so their device addresses survive graph_compute
// calls. Tensors set `buffer` so gallocr respects external ownership.
//
// See MOE-EXPERT-CACHE-V2-PLAN.md for the architectural plan and
// reference architectures (vLLM PR #37190, tinyserve, FATE, HOBBIT,
// martinalderson's moe-profile PoC, e1n00r's PR #21609).

#include "ggml-moe-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------------
// Internal types
// -----------------------------------------------------------------------------

struct ggml_moe_cache {
    ggml_backend_t backend = nullptr;
    int            n_layers = 0;
    int            slots_per_bucket = 0;  // user param; capped to n_experts at bind time
    size_t         max_bytes_cap = 0;
    size_t         total_bytes = 0;
    ggml_moe_cache_policy policy = GGML_MOE_CACHE_POLICY_RR;
    // For LFRU_DECAY: how often to halve frequencies. Measured in tick
    // increments (hits + inserts). Tuned to "couple of full sweeps
    // through cache" so older hot entries fade gradually.
    uint64_t       decay_interval = 4096;
    uint64_t       last_decay_tick = 0;

    // Long-lived ggml_context for the slot-pool + slot-ids tensor wrappers.
    // no_alloc=true: we attach our own backend buffers manually.
    void *         tensor_ctx_mem = nullptr;
    ggml_context * tensor_ctx     = nullptr;

    // One cell per (layer, bucket). Indexed [layer * GGML_MOE_BUCKET_COUNT + bucket].
    struct cell {
        bool   bound = false;
        size_t expert_size = 0;       // bytes per expert in this (layer, bucket)
        size_t slot_stride = 0;       // padded expert_size, multiple of 512 for MMQ safety
        int    n_slots = 0;           // min(slots_per_bucket, n_experts)
        int64_t n_experts = 0;        // weight tensor ne[2]
        int    top_k = 0;             // routing top-k; sizes slot_ids buffer
        int    max_n_tokens = 0;      // upper bound on n_tokens, sizes slot_ids buffer

        // Slot pool: persistent device buffer + ggml_tensor wrapper.
        ggml_backend_buffer_t pool_buf    = nullptr;
        uint8_t *             pool_base   = nullptr;
        ggml_tensor *         pool_tensor = nullptr;

        // Slot ids: persistent device buffer + ggml_tensor wrapper.
        // Sized for top_k * max_n_tokens int32 entries.
        ggml_backend_buffer_t ids_buf     = nullptr;
        uint8_t *             ids_base    = nullptr;
        ggml_tensor *         ids_tensor  = nullptr;

        // Original src[2] (e.g. selected_experts) recorded by the
        // scheduler. compute_splits patches node->src[2] to ids_tensor
        // (above) so the kernel reads remapped slot indices; across
        // graph reuse that patch persists, so on the next call we
        // need this original to D2H the actual expert ids.
        ggml_tensor *         original_ids_tensor = nullptr;

        // Persistent ggml_tensor wrapper for the overflow scratch. Built
        // once per cell on first overflow; reused for subsequent
        // overflows (only ONE op per cell per compute_splits). The data
        // pointer is reassigned each acquire to point at the latest
        // cudaMallocAsync'd scratch.
        ggml_tensor *         overflow_wrapper = nullptr;

        // Residency.
        std::vector<int32_t>             slot_to_expert;  // [n_slots], -1 if empty
        std::unordered_map<int32_t, int> expert_to_slot;  // O(1) lookup
        int                              next_unused = 0;

        // Eviction state (shared across policies; updates always done,
        // policies read only what they need).
        std::vector<uint32_t> freq;       // [n_slots]: hits + inserts (LFRU_DECAY)
        std::vector<uint64_t> last_tick;  // [n_slots]: last access tick (LRU / SLRU / tiebreak)
        std::vector<uint8_t>  tier;       // [n_slots]: 0=probationary, 1=protected (SLRU only)

        // GPU-resident expert->slot mapping (allocated via the regular
        // ggml backend buffer path so the kernel can address it).
        // mapping[expert_id] = slot_id if resident, -1 if not.
        ggml_backend_buffer_t mapping_buf = nullptr;
        void *                mapping_dev = nullptr;
        // Expert ids whose mapping[] entry needs a 4-byte H2D before
        // the next remap kernel (set in record_slot when a slot is
        // reassigned). Flushed by ggml_moe_cache_remap_ids_on_device.
        std::vector<int32_t> dirty_mapping_experts;
        // Host shadow of the GPU mapping; lets us compute the value
        // we need to flush without re-querying the device.
        std::vector<int32_t> mapping_host;

        // Per-cell stats.
        uint64_t n_lookups = 0;
        uint64_t n_hits    = 0;
    };
    std::vector<cell> cells;
    uint64_t          tick = 0;  // monotonic, for LFRU LRU-tiebreak

    // Shared overflow scratch buffer. Allocated lazily on first overflow,
    // grown to fit the largest expert tensor seen. A real
    // ggml_backend_buffer_t — needed because ggml_cuda_mul_mat_id
    // dereferences src0->buffer->buft, so cudaMallocAsync'd raw pointers
    // would segfault the kernel.
    //
    // Safe to share across cells because the scheduler issues one MoE
    // op per split (each expert weight is host-resident and creates a
    // split boundary); successive splits are serial within
    // compute_splits, so the scratch's data is overwritten between
    // splits but consumed by a kernel before the next overwrite.
    ggml_backend_buffer_t scratch_buf       = nullptr;
    size_t                scratch_capacity  = 0;
    uint64_t              total_overflow_ops = 0;

    // Aggregate stats.
    ggml_moe_cache_stats stats{};
};

static_assert(GGML_MOE_BUCKET_COUNT == 4, "ggml_moe_cache assumes 4 buckets");

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static void moe_cache_log(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "moe-cache: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

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
        ggml_backend_t              backend,
        int                         n_layers,
        int                         slots_per_bucket,
        size_t                      max_bytes,
        ggml_moe_cache_policy       policy) {
    if (!backend || n_layers <= 0 || slots_per_bucket <= 0) {
        return nullptr;
    }

    auto * c = new ggml_moe_cache();
    c->backend          = backend;
    c->n_layers         = n_layers;
    c->slots_per_bucket = slots_per_bucket;
    c->max_bytes_cap    = max_bytes;
    c->policy           = policy;

    // Long-lived ggml_context for tensor wrappers. We allocate space for
    // 2 wrappers per cell (slot pool + slot_ids) plus headroom. Each
    // ggml_tensor is ~400 bytes; 4 buckets * n_layers * 2 * 400 ≈ 128 KiB
    // for a 40-layer model. Round up generously.
    const size_t ctx_mem_size = 512 * 1024;
    c->tensor_ctx_mem = malloc(ctx_mem_size);
    if (!c->tensor_ctx_mem) {
        delete c;
        return nullptr;
    }
    ggml_init_params ip = {
        /*.mem_size   =*/ ctx_mem_size,
        /*.mem_buffer =*/ c->tensor_ctx_mem,
        /*.no_alloc   =*/ true,
    };
    c->tensor_ctx = ggml_init(ip);
    if (!c->tensor_ctx) {
        free(c->tensor_ctx_mem);
        delete c;
        return nullptr;
    }

    c->cells.resize((size_t) n_layers * GGML_MOE_BUCKET_COUNT);

    const char * policy_name = "rr";
    switch (policy) {
        case GGML_MOE_CACHE_POLICY_RR:         policy_name = "rr";         break;
        case GGML_MOE_CACHE_POLICY_LRU:        policy_name = "lru";        break;
        case GGML_MOE_CACHE_POLICY_SLRU:       policy_name = "slru";       break;
        case GGML_MOE_CACHE_POLICY_LFRU_DECAY: policy_name = "lfru-decay"; break;
    }
    moe_cache_log("v2 init: %d layers x %d buckets, %d slots/bucket, policy=%s; buffers allocated lazily per (layer, bucket)",
                  n_layers, (int) GGML_MOE_BUCKET_COUNT, slots_per_bucket, policy_name);

    return c;
}

void ggml_moe_cache_free(ggml_moe_cache_t c) {
    if (!c) return;

    // Final stats.
    if (c->stats.total_lookups > 0) {
        const double hit_rate = (double) c->stats.total_hits / (double) c->stats.total_lookups;
        const double saved_mib = c->stats.total_bytes_saved / (1024.0 * 1024.0);
        int bound_cells = 0;
        for (const auto & cell : c->cells) if (cell.bound) ++bound_cells;
        moe_cache_log("final: %lld lookups, %lld hits (%.1f%%), %lld misses; %.1f MiB H2D saved; %d / %d cells bound; %.2f MiB allocated",
                      (long long) c->stats.total_lookups,
                      (long long) c->stats.total_hits,
                      hit_rate * 100.0,
                      (long long) c->stats.total_misses,
                      saved_mib,
                      bound_cells, (int) c->cells.size(),
                      c->total_bytes / (1024.0 * 1024.0));
    } else {
        moe_cache_log("final: cache code path never executed");
    }

    for (auto & cell : c->cells) {
        if (cell.pool_buf)    ggml_backend_buffer_free(cell.pool_buf);
        if (cell.ids_buf)     ggml_backend_buffer_free(cell.ids_buf);
        if (cell.mapping_buf) ggml_backend_buffer_free(cell.mapping_buf);
    }
    if (c->scratch_buf)    ggml_backend_buffer_free(c->scratch_buf);
    if (c->tensor_ctx)     ggml_free(c->tensor_ctx);
    if (c->tensor_ctx_mem) free(c->tensor_ctx_mem);
    delete c;
}

// -----------------------------------------------------------------------------
// Identification
// -----------------------------------------------------------------------------

bool ggml_moe_cache_identify_tensor(
        ggml_moe_cache_t           c,
        const struct ggml_tensor * input,
        int                        n_layers,
        int *                      out_layer_idx,
        ggml_moe_bucket *          out_bucket) {
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
// Lazy allocation: slot pool + slot_ids buffer + tensor wrappers
// -----------------------------------------------------------------------------

bool ggml_moe_cache_bind_bucket(
        ggml_moe_cache_t           c,
        int                        layer_idx,
        ggml_moe_bucket            bucket,
        const struct ggml_tensor * weight,
        int                        top_k,
        int                        max_n_tokens) {
    if (!c || !weight || bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return false;
    if (layer_idx < 0 || layer_idx >= c->n_layers)                       return false;
    if (top_k <= 0 || max_n_tokens <= 0)                                 return false;

    auto & cell = c->cells[cell_idx(layer_idx, bucket)];

    // Idempotent: subsequent binds re-validate but don't reallocate.
    const size_t expert_size = ggml_nbytes(weight) / weight->ne[2];
    if (cell.bound) {
        if (cell.expert_size != expert_size || cell.n_experts != weight->ne[2]
            || cell.top_k < top_k || cell.max_n_tokens < max_n_tokens) {
            moe_cache_log("cell (layer=%d, bucket=%d) shape changed (expert_size %zu->%zu, n_experts %lld->%lld, top_k %d->%d, max_n_tokens %d->%d)",
                          layer_idx, (int) bucket, cell.expert_size, expert_size,
                          (long long) cell.n_experts, (long long) weight->ne[2],
                          cell.top_k, top_k, cell.max_n_tokens, max_n_tokens);
            return false;
        }
        return true;
    }

    const int64_t n_experts = weight->ne[2];
    cell.n_experts   = n_experts;
    cell.n_slots     = std::min<int>(c->slots_per_bucket, (int) n_experts);
    cell.expert_size = expert_size;
    cell.slot_stride = ((expert_size + 511) / 512) * 512;   // 512-byte align for MMQ
    cell.top_k       = top_k;
    cell.max_n_tokens = max_n_tokens;

    // +512 bytes of MMQ-safety padding past the last slot. The CUDA MMQ
    // kernel reads slightly past expert boundaries; the original
    // copy_experts H2D path adds the same padding to input_cpy. Without
    // it, the last slot's kernel read crashes on illegal memory access.
    const size_t pool_bytes = (size_t) cell.n_slots * cell.slot_stride + 512;
    const size_t ids_bytes  = (size_t) top_k * max_n_tokens * sizeof(int32_t);

    if (c->max_bytes_cap > 0 && c->total_bytes + pool_bytes + ids_bytes > c->max_bytes_cap) {
        moe_cache_log("cell (layer=%d, bucket=%d) alloc would exceed cap (%zu + %zu > %zu)",
                      layer_idx, (int) bucket, c->total_bytes, pool_bytes + ids_bytes, c->max_bytes_cap);
        return false;
    }

    auto * buft = ggml_backend_get_default_buffer_type(c->backend);

    // Allocate the slot pool buffer.
    cell.pool_buf = ggml_backend_buft_alloc_buffer(buft, pool_bytes);
    if (!cell.pool_buf) {
        moe_cache_log("cell (layer=%d, bucket=%d) failed to allocate %zu bytes for slot pool",
                      layer_idx, (int) bucket, pool_bytes);
        return false;
    }
    cell.pool_base = (uint8_t *) ggml_backend_buffer_get_base(cell.pool_buf);

    // Allocate the slot_ids buffer.
    cell.ids_buf = ggml_backend_buft_alloc_buffer(buft, ids_bytes);
    if (!cell.ids_buf) {
        moe_cache_log("cell (layer=%d, bucket=%d) failed to allocate %zu bytes for slot_ids",
                      layer_idx, (int) bucket, ids_bytes);
        ggml_backend_buffer_free(cell.pool_buf);
        cell.pool_buf = nullptr;
        return false;
    }
    cell.ids_base = (uint8_t *) ggml_backend_buffer_get_base(cell.ids_buf);

    // Build slot pool tensor wrapper: shape [K, N, n_slots] with stride
    // nb[2] = slot_stride (padded). Manually attach our buffer so
    // ggml_gallocr leaves it alone.
    cell.pool_tensor = ggml_new_tensor_3d(
        c->tensor_ctx, weight->type,
        weight->ne[0], weight->ne[1], cell.n_slots);
    if (!cell.pool_tensor) {
        moe_cache_log("cell (layer=%d, bucket=%d) failed to construct pool tensor wrapper", layer_idx, (int) bucket);
        ggml_backend_buffer_free(cell.pool_buf); cell.pool_buf = nullptr;
        ggml_backend_buffer_free(cell.ids_buf);  cell.ids_buf  = nullptr;
        return false;
    }
    cell.pool_tensor->nb[0] = weight->nb[0];
    cell.pool_tensor->nb[1] = weight->nb[1];
    cell.pool_tensor->nb[2] = cell.slot_stride;
    cell.pool_tensor->nb[3] = cell.slot_stride * cell.n_slots;
    cell.pool_tensor->data   = cell.pool_base;
    cell.pool_tensor->buffer = cell.pool_buf;
    snprintf(cell.pool_tensor->name, sizeof(cell.pool_tensor->name),
             "moe-cache-pool-L%d-B%d", layer_idx, (int) bucket);

    // Build slot_ids tensor wrapper: shape [top_k, max_n_tokens] int32.
    // ne[1] is rewritten to actual n_tokens in set_ids before each op.
    cell.ids_tensor = ggml_new_tensor_2d(
        c->tensor_ctx, GGML_TYPE_I32, top_k, max_n_tokens);
    if (!cell.ids_tensor) {
        moe_cache_log("cell (layer=%d, bucket=%d) failed to construct ids tensor wrapper", layer_idx, (int) bucket);
        ggml_backend_buffer_free(cell.pool_buf); cell.pool_buf = nullptr;
        ggml_backend_buffer_free(cell.ids_buf);  cell.ids_buf  = nullptr;
        return false;
    }
    cell.ids_tensor->data   = cell.ids_base;
    cell.ids_tensor->buffer = cell.ids_buf;
    snprintf(cell.ids_tensor->name, sizeof(cell.ids_tensor->name),
             "moe-cache-ids-L%d-B%d", layer_idx, (int) bucket);

    cell.slot_to_expert.assign(cell.n_slots, -1);
    cell.expert_to_slot.reserve((size_t) cell.n_slots);
    cell.freq.assign(cell.n_slots, 0);
    cell.last_tick.assign(cell.n_slots, 0);
    cell.tier.assign(cell.n_slots, 0);   // probationary
    cell.next_unused = 0;
    cell.bound       = true;
    c->total_bytes  += pool_bytes + ids_bytes;

    static int n_bound = 0;
    if (n_bound < 8) {
        moe_cache_log("cell bound (layer=%d, bucket=%d) expert_size=%zu slot_stride=%zu n_slots=%d top_k=%d max_n_tokens=%d pool=%.2f MiB ids=%zu B",
                      layer_idx, (int) bucket, cell.expert_size, cell.slot_stride,
                      cell.n_slots, top_k, max_n_tokens,
                      pool_bytes / (1024.0 * 1024.0), ids_bytes);
        ++n_bound;
    }

    return true;
}

// -----------------------------------------------------------------------------
// Lookup / select / record
// -----------------------------------------------------------------------------

int ggml_moe_cache_lookup(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int32_t expert_id) {
    if (!c) return -1;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return -1;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return -1;

    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.bound) return -1;

    ++cell.n_lookups;
    ++c->stats.total_lookups;

    auto it = cell.expert_to_slot.find(expert_id);
    if (it == cell.expert_to_slot.end()) {
        ++c->stats.total_misses;
        return -1;
    }

    const int s = it->second;
    ++cell.freq[s];
    cell.last_tick[s] = ++c->tick;
    cell.tier[s] = 1;   // SLRU promotion on hit (no-op for other policies)
    ++cell.n_hits;
    ++c->stats.total_hits;
    c->stats.total_bytes_saved += (int64_t) cell.expert_size;
    return s;
}

// LRU: evict slot with smallest last_tick.
static int evict_lru(const ggml_moe_cache::cell & cell) {
    int best = -1; uint64_t best_t = UINT64_MAX;
    for (int s = 0; s < cell.n_slots; ++s) {
        if (cell.last_tick[s] < best_t) { best_t = cell.last_tick[s]; best = s; }
    }
    return best;
}

// SLRU: evict LRU among probationary (tier=0); fall back to LRU among
// protected (tier=1) only if no probationary slots exist. New entries
// always enter probationary (set in record_slot).
static int evict_slru(const ggml_moe_cache::cell & cell) {
    int best = -1; uint64_t best_t = UINT64_MAX;
    for (int s = 0; s < cell.n_slots; ++s) {
        if (cell.tier[s] == 0 && cell.last_tick[s] < best_t) {
            best_t = cell.last_tick[s]; best = s;
        }
    }
    if (best >= 0) return best;
    return evict_lru(cell);
}

// LFRU with periodic frequency halving: classic LFU pathology fix.
// Halving once every decay_interval ticks lets stale "once-hot"
// entries age out instead of clogging the cache forever.
static int evict_lfru_decay(ggml_moe_cache & c, ggml_moe_cache::cell & cell) {
    if (c.tick - c.last_decay_tick >= c.decay_interval) {
        for (auto & f : cell.freq) f = f / 2;
        c.last_decay_tick = c.tick;
    }
    int best = -1; uint32_t best_f = UINT32_MAX; uint64_t best_t = UINT64_MAX;
    for (int s = 0; s < cell.n_slots; ++s) {
        if (cell.freq[s] < best_f || (cell.freq[s] == best_f && cell.last_tick[s] < best_t)) {
            best_f = cell.freq[s]; best_t = cell.last_tick[s]; best = s;
        }
    }
    return best;
}

int ggml_moe_cache_select_slot_for_miss(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int32_t /*expert_id*/) {
    if (!c) return -1;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return -1;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return -1;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.bound || cell.n_slots <= 0) return -1;

    switch (c->policy) {
        case GGML_MOE_CACHE_POLICY_RR: {
            // Round-robin: empirically beats classic LFRU on both
            // decode t/s and hit rate (probably because LFU's "new
            // admission" pathology hurts more than RR's no-smarts).
            const int slot = cell.next_unused;
            cell.next_unused = (cell.next_unused + 1) % cell.n_slots;
            return slot;
        }
        case GGML_MOE_CACHE_POLICY_LRU:        return evict_lru(cell);
        case GGML_MOE_CACHE_POLICY_SLRU:       return evict_slru(cell);
        case GGML_MOE_CACHE_POLICY_LFRU_DECAY: return evict_lfru_decay(*c, cell);
    }
    return -1;
}

void ggml_moe_cache_record_slot(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int slot_idx, int32_t expert_id) {
    if (!c) return;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.bound || slot_idx < 0 || slot_idx >= cell.n_slots) return;

    // If we're overwriting an existing resident expert, drop its mapping.
    const int32_t prev = cell.slot_to_expert[slot_idx];
    if (prev >= 0) {
        cell.expert_to_slot.erase(prev);
        // Mark prev as no-longer-resident in the GPU mapping.
        if (!cell.mapping_host.empty()) {
            cell.mapping_host[prev] = -1;
            cell.dirty_mapping_experts.push_back(prev);
        }
    }
    cell.slot_to_expert[slot_idx] = expert_id;
    cell.expert_to_slot[expert_id] = slot_idx;
    cell.freq[slot_idx]      = 1;
    cell.last_tick[slot_idx] = ++c->tick;
    cell.tier[slot_idx]      = 0;   // new entry starts probationary (SLRU)

    // Update GPU mapping shadow + dirty list so the next remap kernel
    // sees the new slot for this expert.
    if (!cell.mapping_host.empty() && expert_id < (int32_t) cell.mapping_host.size()) {
        cell.mapping_host[expert_id] = slot_idx;
        cell.dirty_mapping_experts.push_back(expert_id);
    }

    c->stats.total_h2d_bytes += (int64_t) cell.expert_size;
}

// -----------------------------------------------------------------------------
// Slot data + tensor accessors
// -----------------------------------------------------------------------------

void * ggml_moe_cache_slot_data(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int slot_idx) {
    if (!c) return nullptr;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return nullptr;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return nullptr;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.bound || slot_idx < 0 || slot_idx >= cell.n_slots) return nullptr;
    return cell.pool_base + (size_t) slot_idx * cell.slot_stride;
}

ggml_tensor * ggml_moe_cache_pool_tensor(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket) {
    if (!c) return nullptr;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return nullptr;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return nullptr;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    return cell.bound ? cell.pool_tensor : nullptr;
}

ggml_tensor * ggml_moe_cache_ids_tensor(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket) {
    if (!c) return nullptr;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return nullptr;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return nullptr;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    return cell.bound ? cell.ids_tensor : nullptr;
}

// -----------------------------------------------------------------------------
// Per-op overflow scratch (small-cache prefill fallback)
// -----------------------------------------------------------------------------
//
// One scratch buffer per cache, sized to the largest expert tensor we've
// seen. Each cell has its own ggml_tensor wrapper pointing into the
// shared buffer's base. Safe to share because the scheduler emits one
// MoE op per split (each expert input is host-resident and forces a
// split boundary), and splits run serially within compute_splits.

ggml_tensor * ggml_moe_cache_acquire_overflow_scratch(
        ggml_moe_cache_t c, ggml_backend_t backend,
        int layer_idx, ggml_moe_bucket bucket,
        const ggml_tensor * weight) {
    if (!c || !backend || !weight) return nullptr;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return nullptr;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return nullptr;

    const size_t needed = ggml_nbytes(weight) + 512;   // +512 MMQ-safety pad
    if (!c->scratch_buf || c->scratch_capacity < needed) {
        if (c->scratch_buf) {
            ggml_backend_buffer_free(c->scratch_buf);
            c->scratch_buf = nullptr;
        }
        auto * buft = ggml_backend_get_default_buffer_type(c->backend);
        c->scratch_buf = ggml_backend_buft_alloc_buffer(buft, needed);
        if (!c->scratch_buf) {
            moe_cache_log("acquire_overflow_scratch: failed to allocate %zu-byte scratch buffer", needed);
            c->scratch_capacity = 0;
            return nullptr;
        }
        c->scratch_capacity = needed;
    }

    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.overflow_wrapper) {
        cell.overflow_wrapper = ggml_new_tensor(c->tensor_ctx, weight->type,
            GGML_MAX_DIMS, weight->ne);
        if (!cell.overflow_wrapper) {
            moe_cache_log("acquire_overflow_scratch: ggml_new_tensor failed");
            return nullptr;
        }
        memcpy(cell.overflow_wrapper->nb, weight->nb, sizeof(cell.overflow_wrapper->nb));
        snprintf(cell.overflow_wrapper->name, sizeof(cell.overflow_wrapper->name),
                 "moe-cache-overflow-L%d-B%d", layer_idx, (int) bucket);
    }
    // Rebind each call: wrappers share data but the buffer pointer must
    // be valid every acquire so the kernel's src0->buffer->buft check
    // succeeds.
    cell.overflow_wrapper->data   = ggml_backend_buffer_get_base(c->scratch_buf);
    cell.overflow_wrapper->buffer = c->scratch_buf;

    ++c->total_overflow_ops;
    return cell.overflow_wrapper;
}

void ggml_moe_cache_release_overflow_scratches(
        ggml_moe_cache_t c, ggml_backend_t backend) {
    // No-op: scratch buffer is shared and persistent. Kept across
    // splits within one compute_splits and across compute_splits calls.
    (void) c; (void) backend;
}

void ggml_moe_cache_record_original_ids(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket,
        ggml_tensor * original_ids) {
    if (!c || !original_ids) return;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    cell.original_ids_tensor = original_ids;
}

ggml_tensor * ggml_moe_cache_resolve_original_ids(
        ggml_moe_cache_t c, ggml_tensor * maybe_ids) {
    if (!c) return maybe_ids;
    if (!maybe_ids) return nullptr;
    // Linear scan: 4 * n_layers cells. With n_layers=40 that's 160
    // pointer comparisons — fine, runs once per MoE op's D2H setup.
    for (const auto & cell : c->cells) {
        if (cell.bound && cell.ids_tensor == maybe_ids) {
            return cell.original_ids_tensor ? cell.original_ids_tensor : maybe_ids;
        }
    }
    return maybe_ids;
}

bool ggml_moe_cache_set_ids(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket,
        ggml_backend_t backend, const int32_t * slot_ids_host, int top_k, int n_tokens) {
    if (!c || !backend || !slot_ids_host || top_k <= 0 || n_tokens <= 0) return false;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return false;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return false;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.bound || !cell.ids_tensor) return false;
    if (top_k > cell.top_k || n_tokens > cell.max_n_tokens) {
        moe_cache_log("set_ids: out-of-bounds (layer=%d bucket=%d top_k=%d/%d n_tokens=%d/%d)",
                      layer_idx, (int) bucket, top_k, cell.top_k, n_tokens, cell.max_n_tokens);
        return false;
    }

    // Update the tensor's effective shape so the kernel reads only the
    // relevant portion. nb[0..1] are I32 standard (sizeof(int32)=4).
    cell.ids_tensor->ne[0] = top_k;
    cell.ids_tensor->ne[1] = n_tokens;
    cell.ids_tensor->ne[2] = 1;
    cell.ids_tensor->ne[3] = 1;
    cell.ids_tensor->nb[0] = sizeof(int32_t);
    cell.ids_tensor->nb[1] = (size_t) top_k * sizeof(int32_t);
    cell.ids_tensor->nb[2] = cell.ids_tensor->nb[1] * n_tokens;
    cell.ids_tensor->nb[3] = cell.ids_tensor->nb[2];

    const size_t bytes = (size_t) top_k * (size_t) n_tokens * sizeof(int32_t);
    // Sync H2D for now. Async tested (probe #1 follow-up) and was
    // slightly *slower* across all cache sizes — the host wait being
    // eliminated by async doesn't shorten the critical path because
    // the GPU stream-waits for the H2D before the kernel anyway.
    (void) backend;
    ggml_backend_tensor_set(cell.ids_tensor, slot_ids_host, 0, bytes);
    return true;
}

// -----------------------------------------------------------------------------
// Stats / introspection
// -----------------------------------------------------------------------------

int ggml_moe_cache_n_layers(ggml_moe_cache_t c) {
    return c ? c->n_layers : 0;
}

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

// -----------------------------------------------------------------------------
// GPU-side expert -> slot remap (per-op gather kernel)
// -----------------------------------------------------------------------------

typedef void * (*moe_mapping_alloc_fn_t)(ggml_backend_t, size_t);
typedef void   (*moe_mapping_free_fn_t)(ggml_backend_t, void *);
typedef bool   (*moe_mapping_set_fn_t)(ggml_backend_t, void *, int32_t, int32_t);
typedef bool   (*moe_remap_fn_t)(ggml_backend_t, const void *, const void *, void *, int);

static moe_mapping_alloc_fn_t resolve_mapping_alloc(ggml_backend_t backend) {
    static moe_mapping_alloc_fn_t cached = nullptr;
    static bool looked_up = false;
    if (looked_up) return cached;
    auto * dev = ggml_backend_get_device(backend);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) cached = (moe_mapping_alloc_fn_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_cuda_moe_cache_mapping_alloc");
    looked_up = true;
    return cached;
}

static moe_mapping_free_fn_t resolve_mapping_free(ggml_backend_t backend) {
    static moe_mapping_free_fn_t cached = nullptr;
    static bool looked_up = false;
    if (looked_up) return cached;
    auto * dev = ggml_backend_get_device(backend);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) cached = (moe_mapping_free_fn_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_cuda_moe_cache_mapping_free");
    looked_up = true;
    return cached;
}

static moe_mapping_set_fn_t resolve_mapping_set(ggml_backend_t backend) {
    static moe_mapping_set_fn_t cached = nullptr;
    static bool looked_up = false;
    if (looked_up) return cached;
    auto * dev = ggml_backend_get_device(backend);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) cached = (moe_mapping_set_fn_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_cuda_moe_cache_mapping_set");
    looked_up = true;
    return cached;
}

static moe_remap_fn_t resolve_remap(ggml_backend_t backend) {
    static moe_remap_fn_t cached = nullptr;
    static bool looked_up = false;
    if (looked_up) return cached;
    auto * dev = ggml_backend_get_device(backend);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) cached = (moe_remap_fn_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_cuda_moe_cache_remap_ids");
    looked_up = true;
    return cached;
}

bool ggml_moe_cache_remap_ids_on_device(
        ggml_moe_cache_t c, ggml_backend_t backend,
        int layer_idx, ggml_moe_bucket bucket,
        const void * expert_ids_data, void * slot_ids_data, int n_elements) {
    if (!c || !backend || !expert_ids_data || !slot_ids_data || n_elements <= 0) return false;
    if (bucket < 0 || bucket >= GGML_MOE_BUCKET_COUNT) return false;
    if (layer_idx < 0 || layer_idx >= c->n_layers)     return false;
    auto & cell = c->cells[cell_idx(layer_idx, bucket)];
    if (!cell.bound) return false;

    // Lazy-init mapping table + host shadow.
    if (!cell.mapping_dev) {
        // Use the ggml backend's regular buffer allocator, same path
        // as the slot pool and ids tensor. cudaMalloc / cudaMallocAsync
        // returned device pointers that the gather kernel couldn't
        // read from in our context (verified via staged-probe kernel).
        auto * buft = ggml_backend_get_default_buffer_type(c->backend);
        const size_t bytes = (size_t) cell.n_experts * sizeof(int32_t);
        cell.mapping_buf = ggml_backend_buft_alloc_buffer(buft, bytes);
        if (!cell.mapping_buf) {
            moe_cache_log("remap_ids: mapping buft_alloc(%zu) failed", bytes);
            return false;
        }
        cell.mapping_dev = ggml_backend_buffer_get_base(cell.mapping_buf);
        cell.mapping_host.assign((size_t) cell.n_experts, -1);
        // Initialize device memory to -1. ggml_backend_tensor_memset
        // isn't readily available for raw buffers; do a small H2D of
        // the host shadow instead.
        // Use a temporary ggml_tensor view of the buffer for the H2D.
        ggml_init_params ip = { /*.mem_size=*/ 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
        ggml_context * tmp_ctx = ggml_init(ip);
        if (tmp_ctx) {
            ggml_tensor * t = ggml_new_tensor_1d(tmp_ctx, GGML_TYPE_I32, cell.n_experts);
            if (t) {
                t->buffer = cell.mapping_buf;
                t->data = cell.mapping_dev;
                ggml_backend_tensor_set(t, cell.mapping_host.data(), 0, bytes);
            }
            ggml_free(tmp_ctx);
        }
        // Re-stage already-resident entries (cache may have been
        // warmed before the first remap call).
        for (int s = 0; s < cell.n_slots; ++s) {
            const int32_t e = cell.slot_to_expert[s];
            if (e >= 0 && e < (int32_t) cell.mapping_host.size()) {
                cell.mapping_host[e] = s;
                cell.dirty_mapping_experts.push_back(e);
            }
        }
    }

    // Flush dirty entries to device.
    if (!cell.dirty_mapping_experts.empty()) {
        auto set_fn = resolve_mapping_set(backend);
        if (!set_fn) return false;
        // Dedupe so we issue at most one H2D per expert per call.
        std::sort(cell.dirty_mapping_experts.begin(), cell.dirty_mapping_experts.end());
        auto last = std::unique(cell.dirty_mapping_experts.begin(), cell.dirty_mapping_experts.end());
        cell.dirty_mapping_experts.erase(last, cell.dirty_mapping_experts.end());
        static int dbg_flush = 0;
        if (dbg_flush < 3) {
            fprintf(stderr, "remap_flush#%d L=%d B=%d dirty_n=%zu (first few: ", dbg_flush, layer_idx, (int)bucket, cell.dirty_mapping_experts.size());
            for (size_t i = 0; i < cell.dirty_mapping_experts.size() && i < 8; ++i) {
                int32_t e = cell.dirty_mapping_experts[i];
                fprintf(stderr, "[e=%d s=%d] ", e, cell.mapping_host[e]);
            }
            fprintf(stderr, ")\n"); fflush(stderr);
            ++dbg_flush;
        }
        for (int32_t e : cell.dirty_mapping_experts) {
            if (e < 0 || e >= (int32_t) cell.mapping_host.size()) continue;
            set_fn(backend, cell.mapping_dev, e, cell.mapping_host[e]);
        }
        cell.dirty_mapping_experts.clear();
    }

    auto remap_fn = resolve_remap(backend);
    if (!remap_fn) return false;
    return remap_fn(backend, expert_ids_data, cell.mapping_dev, slot_ids_data, n_elements);
}
