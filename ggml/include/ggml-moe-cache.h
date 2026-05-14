// SPDX-License-Identifier: MIT
// MoE expert-weight cache for the CUDA backend (issue #20757).
//
// Architecture: persistent backend-owned slot pool per (layer, bucket).
// The kernel reads expert weights directly from the slot pool — no
// D2D-into-input_cpy per cache hit. Routing ids are remapped from
// expert-ids in [0, n_experts) to slot-ids in [0, n_slots); the MoE
// op's src[0] and src[2] are repointed in compute_splits to the
// cache's persistent tensors.
//
// Reference architectures: vLLM PR #37190, tinyserve, martinalderson's
// moe-profile PoC, e1n00r's PR #21609. All converge on this pattern.
//
// See MOE-EXPERT-CACHE-V2-PLAN.md for the architectural plan.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_moe_cache;
typedef struct ggml_moe_cache * ggml_moe_cache_t;

// Each (layer, bucket) has its own slot pool. Buckets correspond to the
// MoE expert weight tensors in Qwen3.5-style architectures.
enum ggml_moe_bucket {
    GGML_MOE_BUCKET_DOWN     = 0,   // ffn_down_exps
    GGML_MOE_BUCKET_GATE     = 1,   // ffn_gate_exps
    GGML_MOE_BUCKET_UP       = 2,   // ffn_up_exps
    GGML_MOE_BUCKET_GATE_UP  = 3,   // ffn_gate_up_exps (combined variant)
    GGML_MOE_BUCKET_COUNT    = 4,
    GGML_MOE_BUCKET_INVALID  = -1,
};

// Eviction policy selector.
enum ggml_moe_cache_policy {
    GGML_MOE_CACHE_POLICY_RR         = 0,  // Round-robin (current default; cyclic next_unused)
    GGML_MOE_CACHE_POLICY_LRU        = 1,  // Evict slot with oldest last_tick (updated on hit)
    GGML_MOE_CACHE_POLICY_SLRU       = 2,  // Segmented LRU: probationary -> protected on hit
    GGML_MOE_CACHE_POLICY_LFRU_DECAY = 3,  // LFU + LRU tiebreak, with periodic frequency halving
};

// Create a cache. Slot pools and id buffers are allocated lazily on
// first bind for each (layer, bucket). Returns NULL on failure.
ggml_moe_cache_t ggml_moe_cache_init(
    ggml_backend_t              backend,
    int                         n_layers,
    int                         slots_per_bucket,    // user param --moe-expert-cache-size
    size_t                      max_bytes,           // 0 = no cap; otherwise per-cache VRAM budget
    enum ggml_moe_cache_policy  policy);             // eviction policy

void ggml_moe_cache_free(ggml_moe_cache_t cache);

// Identify which (layer_idx, bucket) a tensor belongs to from its name.
// Tensor name pattern: "blk.{layer}.ffn_{down|gate|up|gate_up}_exps.weight"
bool ggml_moe_cache_identify_tensor(
    ggml_moe_cache_t           cache,
    const struct ggml_tensor * input,
    int                        n_layers,
    int *                      out_layer_idx,
    enum ggml_moe_bucket *     out_bucket);

// Bind (lazy-allocate) the slot pool + ids buffer for (layer, bucket).
// `weight` is the host-resident expert weight tensor (input to the MoE
// op); the cache reads its dtype/strides + ne[2]=n_experts from it to
// size the slot pool. `top_k` is the routing top-k (ids[0]); the cache
// sizes the slot_ids buffer for top_k * max_n_tokens entries.
// Returns false on allocation failure or shape mismatch.
bool ggml_moe_cache_bind_bucket(
    ggml_moe_cache_t           cache,
    int                        layer_idx,
    enum ggml_moe_bucket       bucket,
    const struct ggml_tensor * weight,
    int                        top_k,
    int                        max_n_tokens);

// Returns slot index >= 0 if expert is currently resident, else -1.
int ggml_moe_cache_lookup(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket,
    int32_t                 expert_id);

// Select a slot for a missed expert. Picks an empty slot if available,
// otherwise evicts the LFRU resident expert. Returns the slot index;
// caller is responsible for H2D'ing the expert's bytes into the slot's
// memory and then calling record_slot to commit. Returns -1 only on
// cache misconfiguration.
int ggml_moe_cache_select_slot_for_miss(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket,
    int32_t                 expert_id);

// Commit: mark `expert_id` as occupying `slot_idx`. Updates the LFRU
// freq/last_tick counters for the slot. Called after the caller has
// completed the H2D into the slot's memory (or fire-and-forget on the
// compute stream — record_slot is bookkeeping-only).
void ggml_moe_cache_record_slot(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket,
    int                     slot_idx,
    int32_t                 expert_id);

// Get the device pointer for a slot in (layer, bucket, slot_idx).
// Caller uses this as the H2D destination on a cache miss. Returns
// NULL on misconfiguration.
void * ggml_moe_cache_slot_data(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket,
    int                     slot_idx);

// Returns the persistent ggml_tensor wrapping the slot pool for
// (layer, bucket). The tensor has shape [K, N, n_slots] (n_slots
// instead of n_experts) and is backed by the cache's own
// ggml_backend_buffer (outside ggml_gallocr). The scheduler patches
// MoE op's src[0] to this tensor before kernel launch.
struct ggml_tensor * ggml_moe_cache_pool_tensor(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket);

// Returns the persistent ggml_tensor wrapping the slot_ids buffer for
// (layer, bucket). Has int32 dtype and shape [top_k, max_n_tokens];
// the effective shape for any given op is set by ggml_moe_cache_set_ids
// before kernel launch.
struct ggml_tensor * ggml_moe_cache_ids_tensor(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket);

// Write the remapped slot ids to the cache's ids tensor for this
// (layer, bucket) and update ne[0] = top_k, ne[1] = n_tokens to
// match the current op. The data is H2D'd on the given backend's
// compute stream (typically the GPU backend doing the MoE op).
// Returns true on success.
bool ggml_moe_cache_set_ids(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket,
    ggml_backend_t          backend,
    const int32_t *         slot_ids_host,
    int                     top_k,
    int                     n_tokens);

// Returns the persistent expert_id -> slot_idx mapping tensor for
// (layer, bucket). Shape [n_experts] int32, GPU-resident. Used in
// build_moe_ffn as the source of ggml_get_rows(mapping, selected_experts)
// so slot_ids are computed on the GPU rather than built on host +
// H2D'd via set_ids. The device data is kept current by record_slot
// (shadow update) + ggml_moe_cache_flush_mapping_to_device (H2D flush)
// from the compute_splits cache hook.
//
// Entries default to 0 (slot 0). After eviction of expert E from slot S,
// mapping[E] is reset to 0. mul_mat_id's assertion (expert<ne02) holds
// for all reachable mapping values. Reading a "stale" mapping entry
// returns slot 0's resident expert — wrong expert, no crash. Sticky
// routing makes this rare in practice (~1% mismatch at cache>=top_k*2).
//
// Returns NULL if cell not bound.
struct ggml_tensor * ggml_moe_cache_mapping_tensor(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket);

// Flush dirty mapping entries to device. Called from compute_splits
// cache hook after miss handling + record_slot updates. Issues one
// async 4-byte H2D per dirty entry (typically <= top_k entries per
// op) on the given backend's compute stream so writes are ordered
// before downstream get_rows reads.
void ggml_moe_cache_flush_mapping_to_device(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket,
    ggml_backend_t          backend);

// --- Phase 2: per-op in-graph miss handling ------------------------------
//
// The compute_splits node-walk intercepts each MoE mul_mat_id node right
// before it computes. At that point all upstream nodes (incl. argsort)
// have run, so node->src[2] holds the CURRENT-pass expert ids. The walk
// reads top_k/n_tokens straight from that live node, D2Hs the ids,
// classifies + H2Ds misses, calls set_ids (which reshapes the persistent
// per-cell ids tensor to exactly this op), then patches node->src[2] to
// that ids tensor. Everything comes from the live node — robust against
// graph reserve/reuse and varying ubatch sizes.

// If `node` is a MoE mul_mat_id whose src[0] is one of the cache's slot
// pool tensors (named "moe-cache-pool-L{layer}-B{bucket}"), fill
// out_layer_idx + out_bucket and return true. Otherwise return false.
// Called per graph node in the compute_splits node-walk; O(1) (parses
// the pool tensor's name).
bool ggml_moe_cache_node_cache_op(
    ggml_moe_cache_t           cache,
    const struct ggml_tensor * node,
    int *                      out_layer_idx,
    enum ggml_moe_bucket *     out_bucket);

// If `node` is a per-layer ids_c tensor — ggml_cont(selected_experts),
// named "ffn_moe_slot_ids-{layer}" — return its layer index, else -1.
// This is the node-walk's per-layer interception point: it is the LAST
// node of the prefix chunk, so when the walk syncs after computing the
// prefix, ids_c holds the CURRENT-pass expert ids. Structural match
// (op == GGML_OP_CONT + name) — robust across graph reserve/reuse.
int ggml_moe_cache_node_topk_layer(
    ggml_moe_cache_t           cache,
    const struct ggml_tensor * node);

// Handle misses for ONE (layer, bucket) using the current-pass expert
// ids D2H'd from the live mul_mat_id node's src[2]. Classifies used
// experts, evicts+H2Ds missing experts into the slot pool from the
// bound host weight, records slots, then builds slot_ids = mapping[ids]
// and pushes them via set_ids (which also reshapes the per-cell ids
// tensor to [top_k, n_tokens]). The caller then patches the node's
// src[2] to ggml_moe_cache_ids_tensor(layer, bucket).
//
// `ids` is top_k*n_tokens int32 values in [0, n_experts). `backend` is
// the compute backend running the MoE op (H2Ds go on its stream).
void ggml_moe_cache_handle_op_miss(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket,
    ggml_backend_t          backend,
    const int32_t *         ids,
    int                     top_k,
    int                     n_tokens);

// Record the "original" ids tensor (e.g. selected_experts) for this
// (layer, bucket) so that the scheduler can recover it after our
// node->src[2] patch from a previous call has persisted via graph
// reuse. Pass the tensor pointer the caller observed BEFORE patching.
void ggml_moe_cache_record_original_ids(
    ggml_moe_cache_t        cache,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket,
    struct ggml_tensor *    original_ids);

// If `maybe_ids` matches any cell's ids tensor (i.e. it's our cache
// tensor from a prior patch), return the cell's stored original.
// Otherwise return `maybe_ids` unchanged. O(n_cells) linear scan;
// n_cells = 4 * n_layers, fine on every host. Returns NULL on bad
// cache pointer.
struct ggml_tensor * ggml_moe_cache_resolve_original_ids(
    ggml_moe_cache_t        cache,
    struct ggml_tensor *    maybe_ids);

// Per-op prefill-overflow fallback: when an op uses more unique
// experts than the cache pool's n_slots, allocate a scratch buffer
// of the full expert tensor's size, return a ggml_tensor wrapper
// suitable for use as MUL_MAT_ID's src[0]. The wrapper is
// constructed in the cache's long-lived ggml_context and reuses
// nb[]/type from `weight`. Caller patches node->src[0] = returned
// tensor, runs copy_experts to fill it, and the scratch is freed
// via ggml_moe_cache_release_overflow_scratch (called by the
// scheduler at end of compute_splits, after graph_compute_async
// returns so the kernel has finished reading it).
//
// CUDA memory pool reuses the same VRAM across stream-ordered
// allocs/frees, so peak live scratch across all overflow ops in
// one graph_compute is bounded by one cell's worth (~140-220 MiB
// on Qwen3.6).
//
// Returns NULL on failure (non-CUDA backend, OOM, etc).
struct ggml_tensor * ggml_moe_cache_acquire_overflow_scratch(
    ggml_moe_cache_t        cache,
    ggml_backend_t          backend,
    int                     layer_idx,
    enum ggml_moe_bucket    bucket,
    const struct ggml_tensor * weight);

// Release every scratch buffer acquired since the last call. Stream-
// ordered: the cudaFreeAsync runs after the kernels that consumed
// the scratches, so it's safe to release immediately after
// graph_compute_async returns.
void ggml_moe_cache_release_overflow_scratches(
    ggml_moe_cache_t        cache,
    ggml_backend_t          backend);

// Number of layers the cache was sized for.
int ggml_moe_cache_n_layers(ggml_moe_cache_t cache);

// Total VRAM allocated across all slot pools + id buffers.
size_t ggml_moe_cache_total_bytes(ggml_moe_cache_t cache);

// Per-cache stats (aggregated across cells).
struct ggml_moe_cache_stats {
    int64_t total_hits;
    int64_t total_misses;
    int64_t total_lookups;
    int64_t total_h2d_bytes;     // bytes H2D'd to slot pools (misses)
    int64_t total_bytes_saved;   // hits * expert_size: H2D bytes avoided
};
void ggml_moe_cache_get_stats(ggml_moe_cache_t cache, struct ggml_moe_cache_stats * out);
void ggml_moe_cache_reset_stats(ggml_moe_cache_t cache);

#ifdef __cplusplus
}
#endif
