// SPDX-License-Identifier: MIT
// MoE per-expert GPU slot cache — Phase 1 implementation.
// See ggml-moe-cache.h for the public API + design notes.

#include "ggml-moe-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

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
    ggml_backend_buffer_t buf  = nullptr;   // single contiguous backing buffer
    uint8_t * base             = nullptr;   // buf base pointer; offsets derived from layout
    size_t total_bytes         = 0;

    int n_layers           = 0;
    int slots_per_bucket   = 0;
    size_t max_bytes_cap   = 0;

    // Per-bucket state. Filled lazily on first bind_bucket() call for each bucket.
    struct bucket_state {
        bool   bound              = false;
        size_t expert_size_bytes  = 0;
        size_t per_layer_bytes    = 0;       // slots_per_bucket * expert_size_bytes
        size_t buf_offset_base    = 0;       // base offset within `buf` for layer 0 of this bucket
        // slot_map[layer * slots_per_bucket + slot] = expert_id (or -1)
        std::vector<int32_t> slot_map;
        // round-robin counter per layer (Phase 1; Phase 2 = LRU)
        std::vector<int>     next_evict;
    } buckets[GGML_MOE_BUCKET_COUNT];

    // Diagnostic env-var flags
    bool force_noop      = false;   // GGML_MOE_CACHE_FORCE_NOOP
    bool force_skip_even = false;   // GGML_MOE_CACHE_FORCE_SKIP_EVEN

    // Stats
    ggml_moe_cache_stats stats{};
};

// -----------------------------------------------------------------------------
// Logging helper — matches the ggml log conventions; lightweight stderr fprintf.
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
// Bucket identification from tensor name
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
    // Pattern: "blk.<N>.ffn_..." — sscanf returns 1 on success
    int layer = -1;
    if (sscanf(name, "blk.%d.", &layer) == 1 && layer >= 0) {
        return layer;
    }
    return -1;
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

    // Initialize per-bucket slot maps & evict counters; buffer alloc deferred to bind_bucket.
    for (int b = 0; b < GGML_MOE_BUCKET_COUNT; b++) {
        c->buckets[b].slot_map.assign(n_layers * slots_per_bucket, -1);
        c->buckets[b].next_evict.assign(n_layers, 0);
    }

    if (c->force_noop) {
        moe_cache_log("GGML_MOE_CACHE_FORCE_NOOP set; cache will return miss on every lookup (diagnostic)");
    }
    if (c->force_skip_even) {
        moe_cache_log("GGML_MOE_CACHE_FORCE_SKIP_EVEN set; faking hits on even expert_ids (output GARBAGE; diagnostic only)");
    }

    return c;
}

void ggml_moe_cache_free(ggml_moe_cache_t c) {
    if (!c) return;
    if (c->buf) {
        ggml_backend_buffer_free(c->buf);
        c->buf = nullptr;
    }
    delete c;
}

// -----------------------------------------------------------------------------
// Buffer allocation — eager, once we know expert_size_bytes from the first bound bucket.
// All buckets share a single backing buffer; offsets are computed from layout.
// -----------------------------------------------------------------------------

static bool ensure_buffer_allocated(ggml_moe_cache * c, size_t expert_size_bytes) {
    if (c->buf) return true;

    // Pad each slot to 512 alignment for MMQ kernel safety (matches existing copy_experts padding).
    const size_t slot_stride = ((expert_size_bytes + 511) / 512) * 512 + 512;
    const size_t per_layer   = c->slots_per_bucket * slot_stride;
    const size_t per_bucket  = c->n_layers * per_layer;
    const size_t total       = GGML_MOE_BUCKET_COUNT * per_bucket;

    if (c->max_bytes_cap > 0 && total > c->max_bytes_cap) {
        moe_cache_log("buffer would need %zu bytes, exceeds cap of %zu — failing init",
                      total, c->max_bytes_cap);
        return false;
    }

    auto * buft = ggml_backend_get_default_buffer_type(c->backend);
    c->buf = ggml_backend_buft_alloc_buffer(buft, total);
    if (!c->buf) {
        moe_cache_log("failed to allocate %zu bytes on backend", total);
        return false;
    }
    c->base = (uint8_t *) ggml_backend_buffer_get_base(c->buf);
    c->total_bytes = total;

    // Lay out bucket base offsets within the single buffer
    for (int b = 0; b < GGML_MOE_BUCKET_COUNT; b++) {
        c->buckets[b].buf_offset_base = b * per_bucket;
        c->buckets[b].per_layer_bytes = per_layer;
        // expert_size_bytes filled later in bind_bucket for this bucket
    }

    moe_cache_log("allocated cache buffer: %.2f GB (%d layers x %d buckets x %d slots x ~%.1f MB/slot)",
                  total / (1024.0 * 1024.0 * 1024.0),
                  c->n_layers, GGML_MOE_BUCKET_COUNT, c->slots_per_bucket,
                  slot_stride / (1024.0 * 1024.0));
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

    if (!ensure_buffer_allocated(c, expert_size_bytes)) {
        return false;
    }

    auto & bs = c->buckets[bucket];
    if (!bs.bound) {
        bs.expert_size_bytes = expert_size_bytes;
        bs.bound = true;
    } else if (bs.expert_size_bytes != expert_size_bytes) {
        moe_cache_log("expert size mismatch for bucket %d: bound=%zu, got=%zu",
                      (int) bucket, bs.expert_size_bytes, expert_size_bytes);
        return false;
    }
    return true;
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
// Lookup + slot selection — Phase 1: linear scan, round-robin eviction.
// All callers are inside compute_splits which is single-threaded, so no locking.
// -----------------------------------------------------------------------------

int ggml_moe_cache_lookup(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int32_t expert_id) {
    if (!c) return -1;
    auto & bs = c->buckets[bucket];
    if (!bs.bound) return -1;

    c->stats.total_lookups++;

    if (c->force_noop) {
        c->stats.total_misses++;
        return -1;
    }
    if (c->force_skip_even && (expert_id % 2 == 0)) {
        // Fake a hit on slot 0 of layer 0 — output WILL be garbage, this is timing diagnostic only
        c->stats.total_hits++;
        return 0;
    }

    const int * row = &bs.slot_map[layer_idx * c->slots_per_bucket];
    for (int s = 0; s < c->slots_per_bucket; s++) {
        if (row[s] == expert_id) {
            c->stats.total_hits++;
            return s;
        }
    }
    c->stats.total_misses++;
    return -1;
}

int ggml_moe_cache_select_slot_for_miss(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int32_t /*expert_id*/) {
    if (!c) return -1;
    auto & bs = c->buckets[bucket];
    if (!bs.bound) return -1;

    // Phase 1: round-robin per (layer, bucket). Phase 2: LRU.
    int slot = bs.next_evict[layer_idx];
    bs.next_evict[layer_idx] = (slot + 1) % c->slots_per_bucket;
    return slot;
}

void ggml_moe_cache_record_slot(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int slot_idx, int32_t expert_id) {
    if (!c) return;
    if (slot_idx < 0 || slot_idx >= c->slots_per_bucket) return;
    c->buckets[bucket].slot_map[layer_idx * c->slots_per_bucket + slot_idx] = expert_id;
}

// -----------------------------------------------------------------------------
// Slot data pointer — used by scheduler to issue copies to/from this address.
// Layout: buf_offset_base + layer_idx * per_layer_bytes + slot_idx * slot_stride
// where slot_stride is implicit in per_layer_bytes / slots_per_bucket.
// -----------------------------------------------------------------------------

void * ggml_moe_cache_slot_data(
        ggml_moe_cache_t c, int layer_idx, ggml_moe_bucket bucket, int slot_idx) {
    if (!c || !c->base) return nullptr;
    auto & bs = c->buckets[bucket];
    if (!bs.bound) return nullptr;

    const size_t slot_stride = bs.per_layer_bytes / c->slots_per_bucket;
    const size_t offset = bs.buf_offset_base
                        + (size_t) layer_idx * bs.per_layer_bytes
                        + (size_t) slot_idx  * slot_stride;
    return c->base + offset;
}

// -----------------------------------------------------------------------------
// D2D copy primitive — actual CUDA call lives in ggml-cuda.cu so ggml-base
// doesn't need to include cuda_runtime.h. When CUDA isn't built, the extern
// symbol is provided as a weak no-op below; non-CUDA returns false and the
// scheduler falls back to the existing contiguous-batch H2D path.
// -----------------------------------------------------------------------------

extern "C" {
// Provided by ggml-cuda.cu when GGML_USE_CUDA is set; weak no-op otherwise.
bool ggml_cuda_moe_cache_d2d_copy_async(
    ggml_backend_t backend, void * dst, const void * src, size_t size);
}

#ifndef GGML_USE_CUDA
// Weak fallback when CUDA isn't built — keeps ggml-base linking when ggml-cuda
// isn't part of the build.
extern "C" __attribute__((weak)) bool ggml_cuda_moe_cache_d2d_copy_async(
        ggml_backend_t /*backend*/, void * /*dst*/, const void * /*src*/, size_t /*size*/) {
    return false;
}
#endif

bool ggml_moe_cache_copy_d2d_async(
        ggml_backend_t backend, void * dst, const void * src, size_t size) {
    if (!ggml_cuda_moe_cache_d2d_copy_async) return false; // weak symbol may be null
    return ggml_cuda_moe_cache_d2d_copy_async(backend, dst, src, size);
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
