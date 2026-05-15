# MoE Expert Weight Cache — Architecture & Port Guide

This document is the algorithmic and historical record of the MoE
expert-weight cache implemented in this llama.cpp fork (branch
`moe-expert-cache-v2`, merge `cd9a2bd60`). It is written to enable
re-implementation of the same ideas on top of vLLM PR
[#37190](https://github.com/vllm-project/vllm/pull/37190), which uses
the same architectural pattern in a different codebase. **The
contributions are algorithmic; the llama.cpp code is reference, not
target.**

If you are about to port this work to vLLM and only have time to read
one section, read [§5](#5-the-four-portable-contributions). Those are
the deltas this work adds on top of the canonical slot-pool design;
everything else in this doc is context.

---

## 1. Why a cache, what it solves

**Setup.** Modern MoE LLMs (Qwen3-A3B, DeepSeek-MoE, Mixtral-style)
fit the *compute* of a small dense model into the *storage* of a large
one — only `top_k` of `n_experts` activate per token, so decode FLOPs
are 3-5B-class even at 35-300B total parameters. Inference servers
that fit such a model on a single consumer GPU use a per-tensor
override (llama.cpp: `-ot 'blk\.\d+\.ffn.*exps=CPU'`; vLLM: similar
expert-offload mechanism) to keep the giant expert weight tensors in
system RAM and stream them to GPU per forward pass.

**Bottleneck.** Per token, the routing selects `top_k` experts; each
expert is one row across `n_experts` in the expert weight tensor. With
CPU-resident experts, every token triggers `top_k × 3` (down / gate /
up) host→device transfers, each ~140-220 MiB on Qwen3.6-A3B. PCIe
saturates and decode collapses from the GPU-only ceiling (~150 t/s
measured) to ~33-40 t/s.

**Solution.** A GPU-resident cache of recently-used experts.
Experts hit in cache: zero H2D. Experts miss: H2D once, into the
cache, then served from there until evicted. This is the same problem
as KV cache, applied to the *weight* tensor instead of the activation
tensor.

The canonical design pattern that works for this problem — confirmed
by **four independent implementations** (this fork, vLLM #37190,
[tinyserve](https://github.com/lentil/tinyserve),
martinalderson's `moe-profile` PoC, and e1n00r's closed llama.cpp PR
#21609) — is the **slot pool + ids remap** pattern described in §2.

---

## 2. The pattern (algorithmic blueprint)

Allocate a **persistent GPU buffer of N slots** per `(layer, bucket)`,
sized `n_slots × expert_size`. Slots are not tied to expert IDs; they
hold whatever experts the eviction policy decided to keep.

Maintain three pieces of bookkeeping per `(layer, bucket)` cell:

| State | Type | Purpose |
|---|---|---|
| `slot_of[expert_id]` | dict / int32[n_experts] | Where each cached expert lives. -1 = not resident. |
| `eviction_state` | per-slot counters | LFRU / LRU / SLRU state for the next victim choice. |
| `mapping_tensor` | GPU int32[n_experts] | Same data as `slot_of`, but on-device. |

On each MoE forward op for that cell:

1. **For each routed expert** (token's top-k entries):
   - Lookup in `slot_of`. Hit → existing slot index.
   - Miss → evict per policy, mark slot, H2D the expert's bytes from
     host into the slot, update `slot_of` and `eviction_state`.
2. **Remap routing ids from expert-space to slot-space.** The MoE
   kernel takes a routing tensor `ids[n_tokens, top_k]` with
   `int32` values in `[0, n_experts)`. After remapping, the same
   tensor has values in `[0, n_slots)`. Two ways to do this:
   - Build the remapped array on the host and H2D over the kernel's
     `ids` input. Simple but adds a small CPU pass.
   - Keep `slot_of` mirrored in GPU memory as `mapping_tensor`, then
     `slot_ids = mapping_tensor[expert_ids]` (a `get_rows` / gather
     on-device). See [§5.3](#53-gpu-side-expert-slot-remap).
3. **Point the MoE kernel's weight input at the slot pool.** The
   pool tensor has shape `[K, N, n_slots]` instead of
   `[K, N, n_experts]`. With remapped ids in `[0, n_slots)`, the
   existing `MUL_MAT_ID` kernel works unmodified — it iterates
   `0..ne[2]-1` and the upper bound is now `n_slots`, not
   `n_experts`.

**Why this beats the obvious alternatives:**

- ❌ **D2D-into-input_cpy on hit.** Maintain a slot pool but also a
  full-size `input_cpy` buffer that the kernel reads from; on every
  hit, copy expert bytes from slot to input_cpy. Pays a structural
  D2D-per-hit tax. This was v1 of this fork — measured cap at ~53%
  of GPU ceiling, abandoned.
- ❌ **Per-expert GPU buffers with bound checks in the kernel.**
  Requires kernel modification, breaks compatibility with the
  existing `MUL_MAT_ID` family, and the buffer-per-expert pattern
  doesn't compose with `ggml_gallocr` / vLLM's allocator
  expectations.
- ✅ **Slot pool, kernel reads directly.** Zero copy on hit, no
  kernel changes, allocator-friendly. This is what all four
  references converged on.

---

## 3. Where it lives in each codebase

### 3.1 llama.cpp (this fork) — reference

| Component | File | Purpose |
|---|---|---|
| Public API + types | `ggml/include/ggml-moe-cache.h` | `init/free`, `bind_bucket`, `lookup`, `select_slot_for_miss`, `record_slot`, `pool_tensor`, `ids_tensor`, `set_ids`, `acquire_overflow_scratch`, `get_stats` |
| Slot pool + eviction logic | `ggml/src/ggml-moe-cache.cpp` | LFRU/LRU/SLRU/RR eviction; persistent `ggml_backend_buffer_t` allocation outside `ggml_gallocr`; per-cell slot tensor wrapper construction. |
| Scheduler integration | `ggml/src/ggml-backend.cpp` (`split_graph`) | Substitutes the slot pool as the MoE op's `tensor_id_copy` entry, before any copy gets allocated. Patches `node->src[2]` (routing ids) to the cache's persistent ids tensor. |
| Per-op ids dup | `src/llama-graph.cpp` (`build_moe_ffn`) | Inserts `ggml_dup(ctx0, selected_experts)` after argsort when cache enabled — gives each MoE op (up/gate/down) its own ids tensor so per-op slot_ids writes don't collide. |
| Lifecycle | `src/llama-context.cpp` | Creates the cache when `--moe-expert-cache-size > 0`; binds buckets as MoE tensors are encountered; frees on context destruction. |
| CLI | `common/arg.cpp` | `--moe-expert-cache-size N` (slots per bucket), `--moe-cache-policy {rr,lru,slru,lfru-decay}` |

### 3.2 vLLM PR #37190 (e1n00r) — target

vLLM expresses the same primitives with Python/PyTorch idioms:

| Component | vLLM analog | Notes |
|---|---|---|
| Slot pool | `torch.empty(n_slots, K, N, device='cuda')` as a module attribute per `(layer, bucket)` | Allocated at layer init, lifetime = model lifetime. Equivalent to llama.cpp's `ggml_backend_buffer_t` outside `ggml_gallocr`. |
| `slot_of` | `_mapping: torch.Tensor` (int32, length `n_experts`) | On-device. Same role as `mapping_tensor` in our header. |
| Eviction state | Python dict / counters held on the layer module | LFU/LRU per slot. |
| Per-op `prepare()` | Layer module method returning `(topk_ids_remapped, w13_view, w2_view)` | Replaces the kernel's normal weight-fetch path. |
| Kernel | Existing vLLM fused-MoE kernel, unmodified | Same insight as ours: with remapped ids, the kernel doesn't know it's reading a slot pool. |

**Key absence in vLLM #37190 as currently drafted:** no
overflow-scratch fallback for prefill, no CPU intercept floor, and
the eviction policy is settled by initial choice rather than swept.
Those are the three deltas worth carrying in. See §5.

---

## 4. v1 → v2: what we tried first and why we abandoned it

This fork's v1 implementation (preserved on branch
`moe-expert-cache-pagein`) used a "parallel pool + D2D-on-hit"
design: maintain the slot pool, **but** also let the scheduler
allocate the kernel's normal `input_cpy` buffer; on each cache hit,
issue a device-to-device copy from the slot to `input_cpy`.

Two structural problems killed it:

1. **D2D-per-hit tax.** Every cache hit cost a per-expert D2D copy of
   the full expert bytes. At hit rate 97%, the saved H2D is huge
   but the introduced D2D on the compute stream stalls the kernel.
   Measured cap: 53% of the no-`-ot` GPU ceiling. Cannot be tuned
   away; it's a fixed cost of the architecture.

2. **Graph-reuse incompatible.** Patching `node->src[0]` at
   `compute_splits` time (after graph capture) breaks
   CUDA-graph replay: the captured graph holds the wrong src[0]
   pointer, forcing a per-token rebuild. Measured decode at 0.66
   t/s with reuse off, 72+ t/s with reuse on. Where to patch — at
   `split_graph` time, before any copies are allocated — is the
   subtle but critical detail.

v2 fixed both by moving the substitution into `split_graph` (kernel
reads slot pool directly; no input_cpy; reuse preserved). **For a
vLLM port: this lesson translates as "do the slot remap in
`prepare()` before the kernel's allocator sees the request, not in a
post-capture hook."**

---

## 5. The four portable contributions

These are the deltas this work adds on top of the canonical slot-pool
design. The first three are independent improvements; the fourth is
an empirical finding that informs initial policy choice.

### 5.1 Per-op `cudaMallocAsync` scratch for prefill overflow

**Problem.** Prefill processes many tokens per op (ubatch up to
4096+). The set of unique experts touched in one op,
`n_unique_experts`, can easily exceed `n_slots` even for reasonable
cache sizes. The reference designs handle this badly:

- e1n00r #21609: maps overflow experts to slot 0 — **silent
  corruption** (all overflow experts read the same weights).
- Other variants: fall back to "no cache for this op, do full
  `n_experts`-size H2D" — but this collides with the allocator's
  bookkeeping and can deadlock under VRAM pressure.

**Our solution** (commits `59ed8fb4d`, `7a38c4a1d`, `af05ee97f`,
`89dda8026`; API: `ggml_moe_cache_acquire_overflow_scratch` /
`release_overflow_scratches`):

For an op where `n_unique > n_slots`, the cache allocates a per-op
scratch tensor sized for the **full** expert tensor
(`n_experts × expert_size`), via the CUDA stream-ordered allocator
(`cudaMallocAsync`). The op uses this scratch as its `src[0]`,
filled with the unique experts H2D'd directly (no slot pool for this
one op). `cudaFreeAsync` runs after `graph_compute_async` returns;
stream ordering guarantees the kernel has finished reading. The CUDA
memory pool reuses the same VRAM across overflow ops in one graph,
so peak live scratch is bounded by one cell's full-tensor footprint
(~140-220 MiB on Qwen3.6).

**Why it matters.** This is what makes `cache=1` ≥ baseline possible
in principle (no cache means every expert is "missed"; the op uses
scratch identical to the no-cache case). It also makes large prefill
ubatches safe.

**vLLM port sketch.** vLLM uses `torch.empty` for the slot pool; the
analogous scratch allocation is `torch.empty(...,
device='cuda')` inside `prepare()` when `n_unique > n_slots`. PyTorch's
caching allocator already reuses VRAM across allocations on the same
stream, so the bound holds. The check is: pre-compute `n_unique`
from `topk_ids`, branch to a scratch-fill path when it exceeds
`n_slots`, free the scratch tensor by letting it go out of scope
after the kernel call.

### 5.2 CPU intercept floor for high-miss ops

**Problem.** For small caches (`n_slots` significantly less than the
working set), the cache *loses* perf vs. no-cache:
- The miss-handling cost (slot selection + per-expert H2D + mapping
  tensor update + ids remap) is non-trivial host overhead.
- If miss rate is high enough, that overhead exceeds the H2D savings.
- The cache is doing thrash work for no benefit.

**Our solution** (commits `836d8dabf`, `fbf40a0f2`; constant
`MIN_USEFUL_TOPK`):

Add a runtime check: if `n_slots < top_k × thrash_threshold` (default
`top_k × 4` — i.e., the cache can't even hold the working set of one
token's experts), skip cache substitution entirely for that op. The
op runs the regular no-cache code path. The cache continues to
service ops that DO have useful coverage.

**Why it matters.** Without this floor, `--moe-expert-cache-size 1`
isn't just "no benefit" — it's actively a regression vs. no cache.
The floor makes the cache a Pareto improvement at every size: at
size 1 you get baseline; at size large you get the speedup; nothing
in between is worse than baseline.

**vLLM port sketch.** A method on the layer module:
`should_use_cache(self, topk_ids) -> bool` that returns False when
`n_slots < top_k × MIN_USEFUL_FACTOR`. Guard the `prepare()` slot-pool
path on this check; otherwise return the unmodified weight tensor.

### 5.3 GPU-side expert→slot remap

**Problem.** The simple version of "remap routing ids from
expert-space to slot-space" builds the slot_ids array on the host
(iterate over topk_ids, look up in `slot_of[]`, write to a CPU
buffer), then H2D over the kernel's ids input. This adds CPU work
proportional to `top_k × n_tokens` per op, and an H2D on every op.
For small models / high token counts it becomes visible.

**Our solution** (Phase 1 commits referenced by tasks P1a/P1b/P1c —
note: not all in this branch's top log; squashed into the merge):

Mirror `slot_of` as a GPU tensor (`mapping_tensor`, int32
`[n_experts]`), updated via H2D *only on miss* (small payload, one
int per miss). Then in the graph, replace the host-side build with a
GPU `get_rows` op: `slot_ids = mapping_tensor[expert_ids]`. This is
a few-microsecond GPU op vs. a host pass + H2D.

**Why it matters.** Removes a per-op host bottleneck that gets
visible at high decode throughput. Modest win individually, but
combined with the other contributions it closes the gap to the GPU
ceiling.

**vLLM port sketch.** Likely already what vLLM does. PyTorch's
`topk_ids` is already a GPU tensor; `_mapping[topk_ids]` is the
natural way to express the remap. Verify when porting that vLLM
isn't doing this on the host accidentally.

### 5.4 Eviction policy — empirical finding

**Problem.** Initial choice between LFRU (LFU with LRU tiebreak +
periodic decay), SLRU (segmented LRU), RR (round-robin), and plain
LRU. The references don't agree:
- tinyserve / vLLM #37190: LFU-style with frequency tracking.
- e1n00r #21609: round-robin (FIFO).
- Some papers (FATE, HOBBIT): segmented LRU.

**Our finding** (sweep on `realistic-sweep.sh`, plot in
`aa873fd0f`):

For the Qwen3-A3B routing pattern (8 experts per token from 256,
working set ~96-128 experts per layer), **plain LRU wins**:

| Policy | Hit rate (cache=96) | Notes |
|---|---|---|
| RR | 78% | Evicts hot experts under load — known FIFO failure mode |
| LRU | **94%** | Adapts to working-set shifts cleanly |
| SLRU | 93% | Slightly worse — protected segment doesn't help here |
| LFRU-decay | 92% | Decay tuning is hard; frequency counters lag working-set shifts |

**Why.** The routing distribution over experts has fast drift but a
clear "recently used = likely used again" pattern. Frequency
tracking helps less than recency.

**Important caveat for vLLM port:** this finding is workload-specific
to Qwen3-A3B. The 8-of-256 routing distribution may differ
materially for DeepSeek-MoE-V2/V3 (256/12 active, different
distribution), Mixtral (8/2), etc. **Re-sweep on the target
deployment's workload before settling.** The infrastructure for
selecting policies at runtime should be in place; the *default*
should be empirical.

---

## 6. Measured results (llama.cpp / Qwen3.6-A3B-MXFP4_MOE / RTX PRO 4500)

Source: [`docs/v2-results.md`](./v2-results.md). Workload: `realistic-sweep.sh`,
6814-token prompt, 499 decode tokens.

| Metric | v0 baseline (cache=0) | v1 (cache=256 LRU) | **v2 cache=256** | GPU ceiling |
|---|---|---|---|---|
| Decode t/s | 33.3 | 80.7 | **118.27** | 152.4 |
| Prefill t/s | 988.8 | 2,152 | **2,425.5** | n/a |
| Hit rate | n/a | 97.9% | **94.9%** | n/a |
| H2D bytes saved | 0 | ~327 GB | **329.96 GB** | n/a |

- v2 = **3.55× cache=0 baseline** decode
- v2 = **1.47× v1** decode (the kernel-reads-from-slot pattern beats v1's D2D-on-hit by avoiding the structural per-hit copy)
- v2 = **78% of GPU ceiling**

The remaining ~22% gap to GPU ceiling is per-op host overhead in the
cache hook + warmup. Likely closable with cross-layer prefetch
(FATE-style; see §7).

---

## 7. What we didn't ship and why

- **FATE-style cross-layer prefetch.** Use layer N's routing decision
  to start the H2D for layer N+1's experts on a copy stream during
  layer N's compute. Out of scope for phase 1; needs upstream
  infrastructure (am17an's PR
  [#21067](https://github.com/ggml-org/llama.cpp/pull/21067) in
  llama.cpp adds `caps.copy_stream`, which is the right primitive).
  vLLM has the equivalent infrastructure natively.
- **Mixed-precision miss fallback.** When VRAM is tight, evicted
  experts could go to a CPU mirror at lower precision rather than
  full eviction. Future work.
- **Dynamic resize at runtime.** Shrink the cache to free VRAM for
  other workloads (image generation, KV burst). Discussed in the
  homelab notes; not implemented.
- **`--moe-expert-cache-size auto`.** Auto-size to free VRAM at
  startup. Trivial once the rest is stable.

---

## 8. Port checklist (for future-me on vLLM #37190)

1. **Wait for #37190 to merge.** Building on the open PR risks
   re-doing the work when the API shape changes pre-merge.
2. **Identify the equivalents** of the API in our header:
   `bind_bucket`, `lookup`, `select_slot_for_miss`, `record_slot`,
   `acquire_overflow_scratch`. Most map to methods on the layer
   module; the scratch path is the new contribution.
3. **Add §5.1 (overflow scratch).** Branch in `prepare()` on
   `n_unique > n_slots`; allocate a full-size tensor; fill from
   host weights; return it as the kernel input for this op.
   Verify PyTorch's allocator coalesces across calls so peak live
   stays bounded.
4. **Add §5.2 (CPU intercept floor).** Guard the cache substitution
   on `n_slots >= top_k × MIN_USEFUL_FACTOR`. Default factor 4 (the
   llama.cpp default after sweep tuning).
5. **Verify §5.3 (GPU-side remap)** is already PyTorch-native. If
   vLLM is doing CPU iteration, replace with `_mapping[topk_ids]`.
6. **Don't blindly carry §5.4 (LRU policy)** — re-sweep on the
   target deployment's routing distribution. Keep the policy
   pluggable.
7. **Reproduce the §6 numbers** on the target hardware. The 78%
   of GPU ceiling figure is the baseline to beat — vLLM's
   continuous batching + paged attention should let you push higher
   on multi-request workloads where llama.cpp can't.

The work this fork did on top of the canonical pattern is small in
LoC but specific. Once #37190 lands, the port is ~1-2 weeks of
careful translation; the meaningful intellectual content is the four
items in §5 plus the v1 → v2 lessons in §4.
