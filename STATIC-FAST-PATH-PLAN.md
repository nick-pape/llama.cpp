# Static / Fast-Path MoE Cache — Design Plan

## Goal

Close the cache=256 → pure-GPU gap (115 → 147 t/s, ~22%) and have the
fix accrue smoothly to cache=255, 192, 128, ... not just at the corner.

## Bottleneck breakdown (measured, from ncu + chrono profiling)

| Source | Approx contribution | Path |
|---|---|---|
| `set_ids` sync H2D per op | ~7% of decode | Phase 1 |
| compute_splits per-op machinery (classify, slot_ids build) | ~3% of decode | Phase 1 |
| **Split-per-op scheduler overhead** (every MoE op forces a split boundary because expert weights are host-resident) | ~12% of decode | Phase 2/3 |

Phase 1 by itself closes ~10%, leaving cache=256 around 127 t/s.
Phase 2 (cache=256 only) closes the rest at the corner.
Phase 3 extends Phase 2 to cache=255..1 via a hybrid GPU mapping.

## Phase 1 — GPU-side `ggml_get_rows` for slot_id remap

### Design

Replace the per-op host-side path:

```
classify (host) → build slot_ids vector (host) → set_ids H2D (sync)
```

with a graph-time gather node:

```
selected_experts ─┐
                  ├─→ ggml_get_rows ─→ slot_ids ─→ mul_mat_id
mapping_tensor ───┘
```

`mapping_tensor` is GPU-resident, owned by the cache, sized
n_experts × int32, semantics `mapping[expert_id] = slot_id` (or -1
unmapped). Updated incrementally via small (4-byte) H2D on miss.

For cache=N == n_experts, mapping is identity and never changes after
warmup → zero per-op host work beyond the classify/miss-handling
needed to *populate* the cache pool. For cache<n_experts, mapping
updates trickle in on miss but the gather kernel does all the per-op
indexing on-device.

### Required changes (~250 LOC)

1. **`ggml-moe-cache`**: per-cell `mapping_buf` (n_experts int32),
   `mapping_tensor` wrapper. `record_slot` records to a dirty list.
   New API: `ggml_moe_cache_mapping_tensor(cache, layer, bucket)`
   returns the wrapper.
2. **`llama-graph`** (`build_moe_ffn`): when cache enabled, after
   `selected_experts = argsort`, insert
   `slot_ids = ggml_get_rows(ctx0, mapping_tensor, selected_experts)`
   and pass `slot_ids` to mul_mat_id instead of `selected_experts`.
3. **`compute_splits`**: remove slot_ids build + set_ids. Flush dirty
   mapping entries via cudaMemcpyAsync(mapping_dev+expert_id, slot_id, 4)
   on the compute stream. The H2D is stream-ordered before the gather
   kernel reads it (gather is in the graph after this prep).
4. **Plumbing**: `llm_graph_context` gets a `moe_cache` pointer
   plumbed from llama_context.

### Risks

- The mapping H2D must complete on the stream before the gather
  reads. ggml's `tensor_set` on a 4-byte pageable source is implicitly
  sync — should be fine — but worth a smoke test under graph reuse.
- `cparams.moe_expert_cache_size > 0` must be the gating signal in
  build_moe_ffn. If cache is enabled but bind fails for some cell,
  fall back to selected_experts directly.

## Phase 2 — Model tensor hijack at n_slots == n_experts

### Design

At cache init when `slots_per_bucket >= n_experts`, after the cache
pool is allocated for a cell, pre-load all experts from the host
weight tensor into the pool, then override the **model's** expert
weight tensor:

```c
model_tensor->buffer = cell.pool_buf;
model_tensor->data   = cell.pool_base;
```

The scheduler now sees a GPU-resident input to mul_mat_id → no split
boundary → no compute_splits cache prep needed for this op → the
full per-op machinery cost disappears.

Combined with Phase 1, cache=256 should land at pure-GPU speed
(modulo a one-time ~700 ms init H2D for the full preload).

### Required changes (~50 LOC)

1. **`ggml-moe-cache`**: new API
   `ggml_moe_cache_hijack_model_tensor(cache, layer, bucket, model_tensor)`
   that allocates pool, copies all experts, overrides the model tensor's
   buffer/data.
2. **`llama-context`**: when cache_size >= n_experts, call hijack for
   each MoE bucket of each layer.
3. **Bypass detection** in `compute_splits`: if hijacked, the cache
   code path shouldn't run for that node at all (achieved naturally
   because no split is created).

## Phase 3 — Hybrid hijack for n_slots < n_experts

### Design

Pool buffer is the model tensor (hijacked). Has n_experts slots,
but only n_slots are "live" — the rest hold stale or zeroed data.
mapping_tensor (from Phase 1) routes expert_id → live_slot for
resident experts; for non-resident, mapping returns the slot of
the LRU expert that gets evicted on miss.

On miss: copy the missed expert into the pool slot of the LRU
victim, update mapping_tensor to point new_expert → that slot,
update old_expert → invalid (or -1, kernel handles gracefully).

Same architecture as Phase 2 but with dynamic slot rotation.

### Risks

- Model tensor with n_experts physical slots but only n_slots live
  is weird — kernel might still read past live slots if mapping is
  wrong. Need careful validation.
- Inserting the gather in the graph means the kernel reads slot_ids
  in [0, n_experts), and the pool has n_experts slot positions —
  consistent. But cache=255 means slot 255 is the "dynamic" slot
  that cycles; mapping must always point to a slot < n_experts that
  holds the requested expert.

## Sequencing

1. Phase 1 first (generalizable, ~10% gain everywhere).
2. Measure. If gain is as predicted, the remaining gap confirms split overhead.
3. Phase 2 next (cache=256 jumps to ceiling). Small commit.
4. Phase 3 if the user wants cache=255 to follow. Larger commit.
