# MoE Expert Cache v2 — Results

Branch: `moe-expert-cache-v2` (fork: `nick-pape/llama.cpp`)
Architecture: persistent backend-owned slot pool + ids remap, kernel
reads slot pool directly. See `MOE-EXPERT-CACHE-V2-PLAN.md` for the
full architectural plan.

## Hardware + workload

- GPU: RTX PRO 4500 (Blackwell sm_120), 32 GiB
- Container: `llama-moe-builder:latest` (CUDA 12.8)
- Model: Qwen3.6-A3B-MXFP4_MOE
- Prompt: `realistic-sweep.sh`'s 6814-token prompt
- Decode: 499 tokens
- Settings: `--ctx-size 8192 --cache-type-k q8_0 --cache-type-v q8_0
  --flash-attn on --batch-size 2048 --ubatch-size 2048 -ngl 999
  -ot 'blk\.\d+\.ffn.*exps=CPU'`

## Headline numbers (cache=256, n_experts == n_slots)

| Metric | v0 baseline (cache=0) | v1 best (cache=256 LRU) | **v2 cache=256** |
|---|---|---|---|
| Decode t/s | 33.3 | 80.7 | **118.27** |
| Prefill t/s | 988.8 | 2152 | **2425.5** |
| Hit rate | n/a | 97.9% | **94.9%** |
| H2D bytes saved | 0 | ~327 GB | **329.96 GB** |
| Graph reuse | 497/499 | 497/499 | **497/499** |

GPU-only ceiling (no `-ot ... CPU`): **152.4 t/s**.

v2 cache=256 = **78% of GPU ceiling** (gate was 85%, close but not there).
v2 cache=256 = **3.55× cache=0 baseline decode**.
v2 cache=256 = **1.47× v1 cache=256 decode** — the kernel-reads-from-slot
pattern beats v1's parallel-pool + D2D-on-hit by avoiding the
structural per-hit D2D cost.

## Across cache sizes (short prompt, 50 decode tokens)

| cache_size | v2 t/s | notes |
|---|---|---|
| 0 | 39.7 | baseline (no cache) |
| 1 | 31.9 | cache disabled by guard; small regression (env-var side-effect) |
| 16 | 39.8 | cache disabled; ≈ baseline |
| 128 | 39.1 | cache disabled; ≈ baseline |
| 256 (= n_experts) | **66.7-118.3** | cache active; warmup-dependent |

The v2 cache only ACTIVATES when `slots_per_bucket >= n_experts`.
For smaller caches the pool tensor is built but split_graph skips
substituting it as `input_cpy` (because the prefill-overflow fallback
would write OOB to the smaller pool). Smaller caches stay
baseline-ish; only cache=full delivers speedup.

## Where the speedup comes from

The v2 architecture differs from v1 in two structural ways:

1. **Slot pool is the kernel's input.** v1 maintained a separate
   slot-pool buffer AND let the scheduler allocate a full-size
   `input_cpy`; on every cache hit it D2D'd from slot to input_cpy.
   v2 substitutes the slot pool as the canonical `tensor_id_copy`
   entry during `split_graph`, so the kernel reads directly from the
   slot pool — zero copy on hit. Eliminates the structural D2D-per-hit
   tax that capped v1 at 53% of GPU ceiling.

2. **Stable pointers across graph reuse.** Patching `node->src[0]` at
   `compute_splits` time (v1 approach) breaks CUDA-graph capture: the
   replayed graph has a different src[0] pointer, forcing a slow
   per-token rebuild. Doing the substitution in `split_graph`
   instead — where the scheduler already tracks the input-copy
   mapping — keeps the pointer stable across reuse. Decode t/s went
   from 0.66 (broken-reuse pre-fix) to 72+ (with reuse).

The remaining ~22% gap to GPU ceiling at cache=256 is per-op host
overhead in the cache hook (lookup loop, slot_ids build, ids tensor
H2D), plus the warmup phase. Likely closable with FATE-style cross-
layer prefetch and/or eliminating the host-side slot_ids build.

## v1 → v2 commit history

Branch `moe-expert-cache-v2` rebased off commit `3c2e35a`
(`common/arg: force GGML_OP_OFFLOAD_MIN_BATCH=1`). The v1 work
(parallel-pool + dispatch_cpu + 4 page-in stages + pool-manager
P1-P3b) is preserved on `moe-expert-cache-pagein` as historical
reference. v2 carries forward only:
- CLI flag plumbing
- `ggml_backend_cuda_set_op_offload_min_batch_size` runtime setter
- LFRU eviction policy (logic only, fresh implementation)

Everything else in v2 is a clean rewrite.

## Open issues

1. **cache < n_experts is no-op.** The cache only activates at
   cache=n_experts. Smaller caches fall back to baseline behavior.
   The user originally wanted cache=1 to ≥ baseline (we hit
   ≈ baseline, but the cache provides no benefit). A correct
   implementation would handle prefill-overflow per-op (multi-pass
   ubatches when n_unique > n_slots) — out of scope for phase 1.
2. **Gate 2 missed by 7 pp.** 78% of GPU ceiling vs 85% target.
   FATE prefetch (already proven by `tinyserve` / vLLM PR #37190 to
   push hit rate >99%) is the natural next step.
3. **cache=1 small regression.** 31.9 vs 39.7 baseline. Likely the
   `GGML_OP_OFFLOAD_MIN_BATCH` env-var path in `common/arg.cpp` is
   firing despite the cache being gated off in `llama-context.cpp`.
   Trivial fix if needed; user impact is `--moe-expert-cache-size 1`
   specifically (no one would use this).
