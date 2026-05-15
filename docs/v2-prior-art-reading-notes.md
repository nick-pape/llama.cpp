# Day-1 reading notes — llama.cpp H5 expert cache CUDA

Reading work that preceded the v2 implementation. The original v1 plan in `nick-pape/ai-pape-house:llama-cpp-moe-cache-plan.md` referenced by this note has been retired; the v2 implementation plan now lives at [`docs/v2-implementation-plan.md`](./v2-implementation-plan.md), and the canonical architecture + port-to-vLLM guide at [`docs/moe-expert-cache.md`](./moe-expert-cache.md). This file is preserved as historical context — the exploration that informed the v2 design.

## Environment

- **Working directory:** `~/work/llama-cpp-moe-cache` (this clone)
- **Base:** `ggml-org/llama.cpp:master` at commit `1ec7ba0` (2026-05-12, includes `spec : parallel drafting support #22838` which merged earlier today)
- **Decision: forking from master, NOT building on top of PR #21067.** Verified 2026-05-12: PR #21067 is **OPEN, draft, CONFLICTING merge with master, stale since 2026-04-25**. Per the plan's Option B, we accept reimplementing ~50 LOC of copy_stream plumbing if needed.

## The architectural location — `ggml/src/ggml-backend.cpp` lines 1576–1660

**`ggml_backend_sched_compute_splits()` already implements MoE expert offload selectively** — this is what `-ot 'blk\.\d+\.ffn.*exps=CPU'` rides on in prod today. The existing logic:

```cpp
// lines 1576-1583: MoE optimization gate
if (split->graph.n_nodes > 0 &&
    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
    ggml_backend_buffer_is_host(input->buffer) && (
    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
    )) {
    // ... we're inside this block ...
}
```

When MoE-with-offload is active:
1. **Lines 1585-1602:** identify the `ids_tensor` (the routing decisions, `node->src[2]`)
2. **Lines 1604-1620:** read ids to host, build a `used_ids` bitset
3. **Lines 1624-1636:** `copy_experts(first_id, last_id)` lambda — `ggml_backend_tensor_set_async` copies a contiguous expert range from CPU `input->data` to GPU `input_cpy` at the right offset
4. **Lines 1638-1660:** outer loop walks `used_ids`, finds consecutive runs of set bits, calls `copy_experts` for each run

Per token, only top_k experts × num_layers get copied. Routing decisions and bitsets are rebuilt every step.

**`input_cpy` is allocated per-split via `tensor_copy()`** at line 1558. It's the destination tensor on the GPU backend, full size of one layer's expert weight stack (all 256 experts worth of bytes, even though only top_k get filled).

## What H5 changes

**Smaller than I expected — no kernel changes, no op signature changes.** The downstream kernel reads `input_cpy[expert_id * expert_size]` no matter where the data came from. The H5 surgery changes only **where the bytes come from** for each expert:

- **Cache hit:** D2D copy from cache slot → `input_cpy[expert_id * expert_size]` (skip the H2D entirely)
- **Cache miss:** existing H2D copy from CPU → `input_cpy[expert_id * expert_size]`, AND record this in a free cache slot for future hits (via D2D `input_cpy → cache[slot]` or H2D `CPU → cache[slot]`)

The kernel sees `input_cpy` exactly as before — full tensor layout, expert `i` at offset `i * expert_size`. **The slot remap happens entirely inside the scheduler. Zero kernel-side changes.**

## Inserted-code sketch

```cpp
// NEW: per-context state, allocated at model load
struct moe_expert_cache {
    int n_slots_per_layer;  // from --moe-expert-cache-size
    ggml_backend_buffer_t cache_buffer;  // pre-allocated, NOT lazy (avoids CUDA graph capture issue)
    // slot_map[layer][bucket][slot] -> expert_id (or -1)
    // bucket: 0=gate_up combined, 1=down (per martinalderson's two-bucket split)
    // OR: keyed on input tensor pointer for backend-agnostic identification
    std::vector<int32_t> slot_map;
    std::vector<int64_t> lru_timestamp;  // for LRU eviction
};

// EXISTING (lines 1623-1636): the copy_experts lambda
// REPLACED with: a per-expert decision (hit/miss) instead of contiguous-range batching

// For each used expert (in the for-loop at line 1645):
for (int32_t id : used_experts) {
    int slot = cache.lookup(layer_idx, bucket, id);
    if (slot >= 0) {
        // HIT: D2D from cache to input_cpy
        ggml_backend_tensor_copy_async(
            split_backend, split_backend,  // both on same device
            cache.tensor_for_slot(layer_idx, bucket, slot),
            input_cpy,  // with offset = id * expert_size
            id * expert_size, expert_size);
        cache.touch(layer_idx, bucket, id);  // LRU update
    } else {
        // MISS: existing H2D + record in slot
        int evict_slot = cache.evict_lru(layer_idx, bucket);
        ggml_backend_tensor_set_async(split_backend, input_cpy,
            (const uint8_t *)input->data + id * expert_size,
            id * expert_size, expert_size);
        // After H2D completes, D2D copy to cache slot for future
        ggml_backend_tensor_copy_async(
            split_backend, split_backend,
            input_cpy, cache.tensor_for_slot(layer_idx, bucket, evict_slot),
            id * expert_size, expert_size);
        cache.set_slot(layer_idx, bucket, evict_slot, id);
    }
}
```

**Estimated total addition: ~250 LOC**, broken across:
- ~80 LOC: `moe_expert_cache` struct + allocation/init/destruction
- ~50 LOC: slot lookup + LRU eviction logic (`std::unordered_map<int32_t,int32_t>` + `std::list` for LRU is fine, scheduler is single-threaded here)
- ~50 LOC: tensor name parsing to identify layer_idx + bucket (carry forward martinalderson's `sscanf("blk.%d.", ...)` + `strstr(..., "down")`)
- ~50 LOC: replace existing contiguous-range copy_experts with per-expert hit/miss decision
- ~20 LOC: CLI flag wiring + context plumbing

## CUDA-specific concerns — REVISED after reading `ggml-cuda.cu`

1. **CUDA graph capture is a NON-ISSUE.** ~~The calibration scout's "one real gotcha"~~ — turned out to be wrong. Reading `ggml_backend_cuda_graph_compute` at lines 4498-4510: `cudaStreamBeginCapture` is called *inside* `ggml_backend_graph_compute_async()`, which is called from `compute_splits` AT LINE 1678 — AFTER all input copying (including our H5 logic) finishes. **Our cache code runs pre-capture, exempt from capture-mode restrictions.** Lazy or eager allocation both work; same stream is used throughout.

2. **`cpy_tensor_async` (D2D) copies the WHOLE tensor, no offset/size variant.** Read `ggml_backend_cuda_cpy_tensor_async` (line 3177-3230): signature is `(backend_src, backend_dst, src, dst)` and the body does `cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream())`. **This means the cache HIT path can't just call `cpy_tensor_async` — we need a partial-copy primitive.**
   - **Option A (chosen for Phase 1):** call `cudaMemcpyAsync(...)` directly from the scheduler for the hit path, using the split_backend's stream (extract via casting to `ggml_backend_cuda_context`). Breaks backend-agnostic abstraction at scheduler level — only CUDA backend supported initially. Compatible with martinalderson's pattern (he made his Vulkan PoC CUDA-agnostic by using ggml backend APIs but had to use the `GGML_TYPE_I8` hack).
   - **Option B (cleaner, for upstream PR):** add new backend interface `cpy_tensor_range_async(backend, src, src_offset, dst, dst_offset, size)`. Bigger surface (every backend gets a new function), but clean for an upstream PR.
   - **Decision:** start with Option A for Phase 1 dev velocity. Refactor to Option B before upstreaming.

3. **Same-stream guarantee.** Both `set_tensor_async` (H2D, line 3143) and the same-backend branch of `cpy_tensor_async` (line 3227) use `cuda_ctx->stream()` — the primary stream of the backend's context. This means H2D and D2D issued sequentially are naturally ordered. **No explicit cudaEvent fence needed** for our miss path's "H2D then D2D-to-cache" sequence.

4. **Padding alignment for MMQ.** The existing `copy_experts` lambda adds `+ padding_end` (line 1635, std::min(expert_size, 512) bytes) to avoid NaNs in MMQ kernel's read past the last expert. Per-slot cache layout needs the same end-padding treatment. Allocate slot size = `expert_size + 512` rounded up.

5. **`tensor_copy()` is a macro** (line 833), not a function. Returns a pre-allocated tensor from `sched->hv_tensor_copies` (hash-indexed array). The dst tensor (`input_cpy`) is allocated at schedule-time, sized identically to the source host tensor (full expert weight stack).

## Function call graph (compute_splits → CUDA backend)

```
ggml_backend_sched_compute_splits  (ggml/src/ggml-backend.cpp:1541)
 └─ for each split:
     ├─ for each input tensor:
     │   ├─ MoE optimization branch (line 1576):
     │   │   ├─ build used_ids bitset
     │   │   └─ copy_experts() lambda  ← [H5 INSERTS HERE]
     │   │       └─ ggml_backend_tensor_set_async(...)  (line 1630)
     │   │           └─ split_backend->iface.set_tensor_async(...)
     │   │               └─ [CUDA backend] ggml_backend_cuda_buffer_set_tensor (ggml-cuda.cu)
     │   │                   └─ cudaMemcpyAsync(...)
     │   └─ ... non-MoE branches
     └─ ggml_backend_graph_compute_async(split_backend, &split->graph)  (line 1678)
         └─ [CUDA backend] dispatches to per-op kernels
             └─ MUL_MAT_ID kernel reads input_cpy[expert_id * expert_size]
```

**Key insight: H5 insertion is in `compute_splits` *only*. The CUDA backend itself (`ggml-cuda.cu`) needs no changes** — we use its existing `set_tensor_async` and `cpy_tensor_async` interfaces. The only backend-specific concern is the graph-capture interaction, and that's handled by pre-allocating the buffer outside capture mode.

This is **significantly less code** than martinalderson's Vulkan PoC (which also touched `ggml-vulkan.cpp` for the SAM/ReBAR optimization). For our CUDA H5, the SAM/ReBAR-equivalent (`cudaHostRegister` with WriteCombined or similar) is the out-of-scope side quest the plan already noted.

## Validation strategy

The plan calls for cherry-picking `step-0` (always-false hook) and `skip-50%` (alternate skip by parity) ablation commits from `nick-pape/llama-moe-cache:wsl2-staging-pool`. Those commits target the HOOK-based architecture, but the diagnostic idea translates cleanly:

- **`step-0` equivalent for H5:** add an env var `MOE_CACHE_FORCE_NOOP=1` that makes the cache always return "miss" (forces the existing H2D path). Measures whether our new code path introduces any overhead even when not used.
- **`skip-50%` equivalent for H5:** add `MOE_CACHE_FORCE_SKIP_EVEN=1` that skips half the experts entirely (returns "hit" from a zeroed slot for even expert_ids). Measures H2D bandwidth saturation. Output will be garbage but timing is the diagnostic.

These are 5-line patches each. Add as last-step PRs OR keep in our internal branch and don't upstream them.

## Open questions remaining for Phase 1 implementation

1. **Where does the cache state live?** Options: (a) static globals in `ggml-backend.cpp` (martinalderson's approach — simple, single-context-per-process), (b) field on `ggml_backend_sched`, (c) field on `ggml_context`. Likely (b) for proper multi-context support.

2. **How to identify (layer_idx, bucket) for the input tensor?** Two approaches: (a) parse the tensor name string (martinalderson's `sscanf("blk.%d.", ...)`), (b) add a new tensor flag `GGML_TENSOR_FLAG_MOE_EXPERT` that the model loader sets, plus layer_idx as metadata. (b) is cleaner for upstream but a bigger surface change. Start with (a) for Phase 1, refactor to (b) if review demands.

3. **Slot pool sizing:** plan says `2*top_k` to `4*top_k` per layer per bucket. For Qwen3.6 (top_k=8), that's 16-32 slots per layer per bucket × 40 layers × 2 buckets. Each slot is one expert (~110 MB at MXFP4). At 16 slots: 40 × 2 × 16 × 110 MB = ~140 GB. **That's not right.**
   - Wait — each slot is one expert's weight tensor, NOT 110 MB. Let me re-check: at MXFP4, Qwen3.6 expert tensor (gate_up combined) is ~2.6 MB per martinalderson's measurement on Q4_K_M (256 experts × 2.6 MB = 670 MB per layer). So slot size = 2.6 MB.
   - 16 slots × 40 layers × 2 buckets × 2.6 MB = 3.3 GB. Reasonable.
   - **Confirm at Phase 1 by reading the actual tensor dimensions from a loaded Qwen3.6 model.**

4. **Two buckets or one?** Resolved — Qwen3.5MoE (= Qwen3.6) uses **separate tensors per bucket**, naming:
   - `blk.{i}.ffn_down_exps.weight` — shape `[n_ff_exp, n_embd, n_expert]`
   - `blk.{i}.ffn_gate_exps.weight` and `blk.{i}.ffn_up_exps.weight` (created by `create_tensor_gate_up_exps` helper at `src/models/qwen35moe.cpp:90`)
   - **Three buckets total: down / gate / up.** Each gets its own slot pool.
   - String-match identification: `strstr(input->name, "down_exps")`, `strstr(input->name, "gate_exps")`, `strstr(input->name, "up_exps")`.

## Next reading session targets

Before writing code:
- **Read `ggml-cuda.cu` lines around the `set_tensor_async` and `cpy_tensor_async` implementations.** Find where streams are selected, where graph capture is entered/exited.
- **Read llama-model.cpp's MoE model loading** to understand how expert tensors are named and bucketed for Qwen3.6 specifically. Identify the actual tensor names (`blk.X.ffn_gate_inps`, `blk.X.ffn_up_exps`, etc.).
- **Read `tensor_copy()` definition** (referenced at line 1558) to understand `input_cpy` allocation semantics.
- **Read the prefetch infrastructure in PR #21067's branch** — even though we're not building on top of it, the copy_stream design may inform our async ordering.

## Status

Day 1 reading complete + corrections from the CUDA backend deep-dive. **The work is genuinely tractable — biased toward the LOWER end of the 1.5-3 week estimate.** Updated understanding:

- ✅ Architectural location pinned: `compute_splits` lines 1576-1660
- ✅ Existing infrastructure (expert selection, ids bitset, contiguous-range H2D) handles ~80% of the work
- ✅ CUDA graph capture: not a concern (runs pre-capture)
- ⚠️ D2D partial-copy: needs direct `cudaMemcpyAsync` for Phase 1 (no public API exists with offset/size for D2D), refactor to new backend API before upstreaming
- ✅ Three buckets per layer: `down`, `gate`, `up` — string-match identification works
- ✅ Same-stream-throughout: no explicit sync fences needed for miss path

**Phase 1 starting code structure** (next session):
```
src/llama-moe-cache.h         (~60 LOC) — public API + struct definitions
src/llama-moe-cache.cpp       (~150 LOC) — alloc/lookup/eviction/destroy
ggml/src/ggml-backend.cpp     (~50 LOC modifications) — insert call into compute_splits
common/arg.cpp                (~10 LOC) — --moe-expert-cache-size flag
src/llama-context.cpp         (~10 LOC) — wire cache into context
```

Total ~280 LOC for Phase 1 (correctness-neutral cache plumbing, no eviction, slots = top_k). Beyond original ~250 estimate because we need `llama-moe-cache.{h,cpp}` as proper files (cleaner than martinalderson's static globals).

## Notes on the existing code that surprised me

- The MoE offload code (lines 1576-1660) is already remarkably efficient — contiguous-range batching of CPU→GPU copies minimizes PCIe transactions. Our cache addition only beats this when hit rate is high enough that the saved H2D bytes outweigh the per-expert D2D overhead. At low hit rate (cold cache, first few hundred tokens), the new code path is SLOWER than the existing code. Need to handle warmup carefully — perhaps a `cache_warmup_tokens` parameter that disables the cache until N tokens have been seen.
- The `prev_ids_tensor` optimization (line 1604, 1620) caches the bitset build between calls when the ids tensor doesn't change. Our cache should preserve this — only do the per-expert hit/miss decision when used_ids has changed.

## Reading sessions remaining before Phase 1 code

- [ ] Read PR #21067 copy_stream design (skipped tonight — PR is stale and we're not basing on it; brief read at most for reference)
- [ ] Read `src/models/qwen35moe.cpp` graph build to confirm where MUL_MAT_ID with expert tensors gets emitted
- [ ] Quick read of `common/arg.cpp` for CLI flag pattern (look at how `--prefetch-weights` was added in PR #21067 for the closest precedent)

After those: write Phase 1 code.
