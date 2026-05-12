// SPDX-License-Identifier: MIT
// MoE per-expert GPU slot cache — Phase 1 (static slots, no eviction)
//
// Hooks into ggml_backend_sched_compute_splits's MoE expert offload path
// (ggml/src/ggml-backend.cpp:1576). For each used expert this token:
//   - HIT  → D2D copy from cache slot to input_cpy at expert offset
//   - MISS → existing H2D from host (preserves current behavior) + populate cache slot
//
// Phase 1 scope:
//   - Static slot pool, no LRU (Phase 2 adds eviction)
//   - Direct cudaMemcpyAsync for D2D hits (Phase 2 may add a proper backend interface)
//   - Three buckets per layer for Qwen3.5MoE: down_exps, gate_exps, up_exps
//     (also supports the combined gate_up_exps variant via separate bucket)
//   - Linear search through slots for lookup (slot count is small, ≤ 32)
//
// See nick-pape/ai-pape-house:llama-cpp-moe-cache-plan.md for the full design + upstream
// PR strategy. See DAY1-READING-NOTES.md (in this repo) for architectural deep-dive.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare; opaque to consumers.
struct ggml_moe_cache;
typedef struct ggml_moe_cache * ggml_moe_cache_t;

// Per-bucket layout. For Qwen3.5MoE, each layer has up to four expert
// weight tensors (down, gate, up, gate_up combined). Each gets its own
// slot pool. Identification is by tensor-name substring match.
enum ggml_moe_bucket {
    GGML_MOE_BUCKET_DOWN     = 0,   // ffn_down_exps
    GGML_MOE_BUCKET_GATE     = 1,   // ffn_gate_exps
    GGML_MOE_BUCKET_UP       = 2,   // ffn_up_exps
    GGML_MOE_BUCKET_GATE_UP  = 3,   // ffn_gate_up_exps (combined variant)
    GGML_MOE_BUCKET_COUNT    = 4,
    GGML_MOE_BUCKET_INVALID  = -1,
};

// Create a cache for the given backend. Buffer is pre-allocated up front
// (NOT lazy — avoids any future CUDA graph-capture interaction). Each slot
// holds one expert's worth of bytes. The buffer is N_LAYERS × N_BUCKETS ×
// slots_per_bucket × expert_size_bytes, capped by max_bytes if non-zero.
//
// Returns NULL on allocation failure.
ggml_moe_cache_t ggml_moe_cache_init(
    ggml_backend_t backend,
    int            n_layers,
    int            slots_per_bucket,    // user param --moe-expert-cache-size
    size_t         max_bytes);          // 0 = no cap (allocate everything)

void ggml_moe_cache_free(ggml_moe_cache_t cache);

// Identify which (layer_idx, bucket) a tensor belongs to from its name.
// Returns true if the tensor is a recognized MoE expert weight tensor
// AND the cache has slots configured for that bucket.
// Tensor name pattern: "blk.{layer}.ffn_{down|gate|up|gate_up}_exps.weight"
bool ggml_moe_cache_identify_tensor(
    ggml_moe_cache_t        cache,
    const struct ggml_tensor * input,
    int                       n_layers,    // for bounds check
    int *                     out_layer_idx,
    enum ggml_moe_bucket *   out_bucket);

// Bind the cache to a specific (layer, bucket) for a sequence of operations
// in compute_splits. Internally records the expert_size_bytes (from input
// tensor metadata) on first call for this bucket; subsequent calls assert
// consistency. Returns false if buffer allocation fails or sizes are inconsistent.
bool ggml_moe_cache_bind_bucket(
    ggml_moe_cache_t        cache,
    int                      layer_idx,
    enum ggml_moe_bucket    bucket,
    size_t                   expert_size_bytes,
    int64_t                  n_experts_total);

// Look up an expert in the slot map. Returns slot index (≥ 0) if cached,
// or -1 if not cached. Phase 1: linear search through slots_per_bucket
// entries. Phase 2: hash map.
int ggml_moe_cache_lookup(
    ggml_moe_cache_t        cache,
    int                      layer_idx,
    enum ggml_moe_bucket    bucket,
    int32_t                  expert_id);

// Phase 1 eviction: round-robin slot selection. Phase 2: LRU.
// Returns the slot index to write the new expert into. May overwrite a
// previously-cached expert; caller is responsible for issuing the copy.
int ggml_moe_cache_select_slot_for_miss(
    ggml_moe_cache_t        cache,
    int                      layer_idx,
    enum ggml_moe_bucket    bucket,
    int32_t                  expert_id);

// Record that an expert now occupies a slot. Called after the H2D copy
// (or D2D-from-input_cpy) is issued. Updates the slot map atomically
// for the scheduler (single-threaded at this point, no lock needed).
void ggml_moe_cache_record_slot(
    ggml_moe_cache_t        cache,
    int                      layer_idx,
    enum ggml_moe_bucket    bucket,
    int                      slot_idx,
    int32_t                  expert_id);

// Get the device pointer for a slot in (layer, bucket, slot_idx). Used
// by the scheduler to issue cudaMemcpyAsync directly from this address.
// Phase 2 may replace this with a proper ggml_tensor wrapper.
void * ggml_moe_cache_slot_data(
    ggml_moe_cache_t        cache,
    int                      layer_idx,
    enum ggml_moe_bucket    bucket,
    int                      slot_idx);

// Issue an async D2D copy on the given backend's primary stream. The copy
// is ordered with subsequent compute ops on that backend. Returns false if
// the backend doesn't support direct D2D (currently CUDA-only — non-CUDA
// backends should fall back to the contiguous-batch H2D path).
// Free function (no cache state needed) but lives here because it
// encapsulates the CUDA-specific cudaMemcpyAsync call.
bool ggml_moe_cache_copy_d2d_async(
    ggml_backend_t backend,
    void *         dst,
    const void *   src,
    size_t         size);

// Page-in infrastructure (S1+): an async copy issued on a DEDICATED stream
// separate from the backend's primary compute stream. The compute stream is
// only ordered behind these copies when ggml_moe_cache_compute_wait_for_copies()
// is called. Lets the scheduler overlap H2D / D2D moves with concurrent
// compute work on the primary stream. Returns false on non-CUDA backends
// or copy failure — caller should treat as if no copy happened.
bool ggml_moe_cache_copy_async_on_copy_stream(
    ggml_backend_t backend,
    void *         dst,
    const void *   src,
    size_t         size);

// Make the backend's compute stream wait for all copies issued via
// ggml_moe_cache_copy_async_on_copy_stream() up to this point. Returns
// false if the backend doesn't expose the copy stream (non-CUDA build).
// On non-CUDA backends the cache currently falls back to the existing
// H2D path on the primary stream, where ordering is already implicit
// and this call is a no-op.
bool ggml_moe_cache_compute_wait_for_copies(ggml_backend_t backend);

// Inverse: make the copy stream wait for the compute stream's pending work.
// Use before issuing a copy-stream op that reads memory the compute stream
// is currently writing (e.g., S2 cache-populate reads from input_cpy that
// compute's H2D just filled). Returns false on non-CUDA.
bool ggml_moe_cache_copy_stream_wait_for_compute(ggml_backend_t backend);

// Number of layers the cache was sized for. Used by the scheduler to validate
// that layer_idx parsed from tensor names is in range before dispatch.
int ggml_moe_cache_n_layers(ggml_moe_cache_t cache);

// Total bytes allocated for the cache buffer. Useful for VRAM accounting
// reported to the user.
size_t ggml_moe_cache_total_bytes(ggml_moe_cache_t cache);

// Per-cell stats, for the diagnostic harness (analogous to wsl2-staging-pool's
// step-0 and skip-50% ablation commits — see DAY1-READING-NOTES.md).
struct ggml_moe_cache_stats {
    int64_t total_hits;
    int64_t total_misses;
    int64_t total_lookups;
    int64_t total_h2d_bytes;
    int64_t total_d2d_bytes;
};
void ggml_moe_cache_get_stats(
    ggml_moe_cache_t              cache,
    struct ggml_moe_cache_stats * out);

void ggml_moe_cache_reset_stats(ggml_moe_cache_t cache);

// Diagnostic mode flags (env var override): set GGML_MOE_CACHE_FORCE_NOOP=1 in
// the environment to make every lookup return "miss" (preserves vanilla
// H2D path; measures whether our new code path adds overhead). Set
// GGML_MOE_CACHE_FORCE_SKIP_EVEN=1 to fake hits on even expert_ids (output
// will be garbage but timing isolates D2D bandwidth from compute).
// Both are picked up at cache init.

#ifdef __cplusplus
}
#endif
