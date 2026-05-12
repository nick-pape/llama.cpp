# MoE Expert Cache for llama.cpp CUDA — Results

Empirical results from the `moe-expert-cache-cuda` (stable) and
`moe-expert-cache-pagein` (experimental) branches. Bench harness, hardware,
and model details below.

## Hardware / model / config

| | |
|---|---|
| **GPU** | NVIDIA Blackwell RTX PRO 4500 (sm_120a, 32 GiB) |
| **CPU** | Ryzen 9 5900XT (12 cores) |
| **Driver** | 570.153.02 (CUDA 12.8) |
| **Host** | Proxmox LXC (privileged), CUDA 12.8.0-devel-ubuntu24.04 build container |
| **Model** | `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (40 layers × 256 experts × top-8 routing, mixed mxfp4 / q5_K / q6_K per layer) |
| **Decode config** | ctx=4096, batch=ubatch=2048, KV q8_0, flash-attn on, threads 4/12 |
| **Override** | `-ot 'blk\.\d+\.ffn.*exps=CPU'` (pins expert weights to CUDA_Host) |
| **Prompt** | 200-word essay request (~12 tokens), 200 decode tokens predicted |

## Headline results (stable branch — `moe-expert-cache-cuda`)

`--moe-expert-cache-size N` slots per (layer, bucket) cell.
Lookup hit rate is per-expert. Generation tok/s measured end-to-end on
200-token decode.

### Eviction policy comparison

| N (slots) | Cache VRAM | LRU hit% | LRU t/s | SLRU hit% | SLRU t/s |
|---|---|---|---|---|---|
| 0 (baseline, CPU MoE) | 0 | — | **36.0** | — | **36.0** |
| 16 | 1.1 GiB | 48.0 | 27.0 | 46.5 | 26.3 |
| 32 | 2.3 GiB | 62.3 | 33.8 | 62.9 | 34.0 |
| 64 | 4.5 GiB | 78.1 | 46.8 | 78.4 | 47.6 |
| 128 | 9.1 GiB | 88.4 | **64.6** | 88.1 | 63.7 |

**Verdict:** SLRU is within measurement noise of LRU on Qwen3.6-A3B routing.
The one-hit-wonder protection that SLRU defends against isn't a real concern
for this routing distribution. Default policy is LRU; SLRU available for
A/B on other models via `--moe-cache-policy` (planned).

**Headline number:** **+79% decode throughput at cache=128 (88% hit rate)**
relative to the CPU-MoE baseline, using 9.1 GiB of cache VRAM.

### Cache-too-small regime

At cache=16 and cache=32 the cache is *net negative* — slower than the
CPU-MoE baseline. Cause: every miss costs a full PCIe stall (~700 KiB ×
top_k × layers per token), and at low hit rates the misses dominate.
Mitigation is the open work on the page-in branch (see below).

## Page-in branch (`moe-expert-cache-pagein`) — experimental

Branched from the stable SLRU state. Each step is an independent commit.

### S1: copy-stream + event-ordering infrastructure
Adds a dedicated CUDA stream + event in `ggml_backend_cuda_context` for
moving cache traffic off the primary compute stream. No semantic change
on its own — pure plumbing for the steps below. Two new sync primitives:
`compute_wait_for_copies` and `copy_stream_wait_for_compute`. Exposed via
the reg proc-address table so non-CUDA builds resolve to no-ops.

### S2: async cache populate
The cache-populate D2D copy was blocking the kernel on the compute stream
even though the populate is fire-and-forget for the current token (it only
benefits next token's hit lookup). S2 moves the populate to the copy
stream. Cross-token correctness preserved by `compute_wait_for_copies()`
at the top of every cache-aware MoE branch entry.

| N | LRU baseline | S2 (LRU + async populate) | Δ |
|---|---|---|---|
| 16 | 27.0 | 27.1 | +0.1 |
| 32 | 33.8 | 34.6 | +0.6 |
| 64 | 46.8 | 47.3 | +0.5 |
| 128 | 64.6 | 62.0 | -2.6 (noise + sync overhead at high cache) |

S2 is a small win at small caches, neutral at high cache. Sets up the
infrastructure for S4 (where async fill becomes the load-bearing piece).

### S3: whole-op CPU fallback on heavy miss *(stopgap, being reverted)*
Routes cold cells (hit rate < 50%) to the CPU backend, avoiding the
per-miss PCIe stall. Achieves "cache never net-negative" but **defeats
the dynamic-cache intent** — cells routed to CPU stop receiving populates
and their slot composition is frozen until re-warmed.

| N | LRU baseline | S3 | Δ |
|---|---|---|---|
| 16 | 27.0 | 38.8 | **+11.8 (above baseline)** |
| 32 | 33.8 | ~38.9 | +5.1 |
| 64 | 46.8 | TBD | |
| 128 | 64.6 | TBD | |

S3 commit will be parked on `moe-expert-cache-pagein-s3-stopgap` for the
record. The path forward is S4 (below) plus an LFU eviction policy.

### LFRU + S3 sweep (final, page-in branch HEAD)

LFRU eviction + S3 adaptive CPU/GPU dispatch, all S* infrastructure
landed:

| Cache | VRAM | Hit% | Gen t/s | CPU dispatch % | Avg miss/op (GPU) |
|---|---|---|---|---|---|
| 0 (baseline) | 0 | — | 36.0 | — | — |
| 16 | 1.1 GiB | 12.9 | 37.8 | 21.8 | 14.56 |
| 32 | 2.3 GiB | 13.5 | 38.4 | 21.8 | 13.74 |
| 64 | 4.5 GiB | 66.7 | 40.6 | 4.1  | 3.12 |
| 128 | 9.1 GiB | 88.6 | 62.2 | 0.0  | 0.93 |

**The S3-routing trade-off, in one table:**

| Cache | LRU only | LRU + S3 | LFRU + S3 | Best choice |
|---|---|---|---|---|
| 16 | 27.0 (below baseline) | **38.8** | 37.8 | S3 (+11.8) |
| 32 | 33.8 (below baseline) | **38.9** | 38.4 | S3 (+5.1) |
| 64 | **46.8** | 41.5 | 40.6 | LRU only (-6 with S3) |
| 128 | **64.6** | 62.3 | 62.2 | LRU only (-2.4 with S3) |

S3 is a clear win at low cache sizes (the cache-too-small regime where
PCIe stalls dominate). At cache ≥ 64, S3 hurts because it permanently
freezes some cells on CPU (locked out of further hit-rate evolution),
dragging the global hit rate down (78.1% → 70.2% at cache=64). S3's
re-warming task (Task #30) is the structural fix.

**LFRU vs SLRU comparison is within measurement noise** (0.1-1.0 t/s)
across all cache sizes when combined with S3. The eviction policy
matters less than expected because S3 routes the cells where eviction
would matter most (cold cells, miss-heavy) to CPU entirely, bypassing
the cache. LFRU should still be retained as the default policy because:
(1) it's the principled choice for warmup, (2) it costs nothing to
keep, and (3) on a different workload (less long-tail routing) the gap
could open up.

**The S4 motivation is the cache=128 result:** even at 88.6% hit rate
with 0% CPU dispatches, every op averages 0.93 misses — every op pays
a small PCIe stall. S4 (per-op hybrid: GPU does hits, CPU does misses
on host weights, merge kernel sums) eliminates that stall while keeping
the cache dynamic.

## Implementation journey notes

### The LFU → LFRU pivot

Pure LFU was the wrong instinct. With cells starting empty, every
freshly-populated slot enters at `freq=1`. After all slots fill (one
populate each), every slot has `freq=1`, and the next miss tiebreaks
by slot index — effectively evicting the just-populated slot every
time. Measured: cache=16 with naive LFU got 14% hit rate / 18.7 t/s
vs LRU's 48% / 27.0 t/s. Classic "instant eviction" pathology.

Fix: **LFRU = LFU with LRU tiebreak.** Each slot carries both `freq`
(hit count since populate) and `last_tick` (monotonic access stamp).
Eviction picks `min(freq)`; ties resolved by `min(last_tick)` — i.e.,
LRU among the freq-equals. Newly-populated slots get the highest tick
so they survive the immediate next miss.

This is the design the vLLM PR #37190 author calls "LFRU" and what
the ExpertFlow paper recommends as the upgrade over plain LFU.

Combined with S3 (cold-cell CPU fallback, restored from the stopgap
branch on top of LFRU), the page-in branch HEAD is now:

```
LFRU eviction
+ S3 adaptive CPU/GPU dispatch
+ S2 async cache populate
+ S1 copy stream infrastructure
+ SLRU base (replaced by LFRU)
+ Per-(layer, bucket) buffers
+ Runtime CUDA op_offload_min_batch_size override
```

Results sweep pending.

### Planned: S4 hybrid + merge kernel

The principled fix for the "even at cache=128 every op has 1-2 misses
costing PCIe stalls" problem:

For any MoE op with cache misses:
- GPU computes the cached experts' contributions (existing path, with
  zero-fill for missed slots so the GPU kernel still runs over the
  full top_k and contributes 0 for missed positions)
- CPU computes the missed experts' contributions in parallel (reading
  host-pinned weights directly, zero PCIe traffic)
- A small CUDA merge kernel sums them into the final output
- The async populate from S2 *still happens* on the copy stream so
  LFRU keeps adapting every cell continuously

Implementation cost is real (~400 LOC across CPU compute path, copy
stream sync, merge kernel, dispatch logic) so this is a follow-up
work item.

## Architectural findings (what we learned)

### 1. The CUDA `op_offload_min_batch_size` gate was the unblock

`ggml_backend_cuda_device_offload_op` returns true only when
`get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size`, and for
`MUL_MAT_ID` `get_op_batch_size` returns `op->ne[2]` which is 1 during
single-token decode. Default `min_batch_size` is 32 (env-tunable
`GGML_OP_OFFLOAD_MIN_BATCH`). So at decode time MoE was always going to
CPU and the entire H2D-expert-offload code path in
`ggml_backend_sched_compute_splits` (≈ line 1576) — the very branch this
cache hooks into — was dead code.

Setting `min_batch_size=1` is required for the cache to engage during
decode. Runtime setter is plumbed via `ggml_backend_cuda_set_op_offload_min_batch_size()`
because env-var-set-from-`--moe-expert-cache-size`-handler races with
`-ot`'s lazy `ggml_backend_load_all()`.

### 2. Per-(layer, bucket) buffers for mixed-quant models

Qwen3.6-A3B-MXFP4 ships with mixed quantization across layers:

```
blk.0.ffn_down_exps.weight (176 MiB q5_K)
blk.0.ffn_gate_exps.weight (136 MiB mxfp4)
blk.0.ffn_up_exps.weight   (136 MiB mxfp4)
blk.1.ffn_down_exps.weight (210 MiB q6_K)
...
```

Per-expert weight sizes vary per layer (557K / 720K / 860K bytes for
mxfp4 / q5_K / q6_K respectively). A single contiguous backing buffer
with one expert-size-per-bucket fails `bind_bucket` for every layer
after the first that has a different quant.

Solution: one `ggml_backend_buffer` per (layer, bucket) cell, sized to
that cell's observed expert weight. 40 layers × 3 buckets = 120 buffers
on Qwen3.6 — slot pointers are still O(1) (`cell.base + slot * stride`),
so no perf cost.

### 3. `cudaMemcpyDefault` required, not `cudaMemcpyDeviceToDevice`

The miss-populate path copies from `input_cpy` (which sits on
CUDA_Host — pinned host memory) to a device cache slot.
`cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)` with a host source
returns `cudaErrorInvalidValue` silently. `record_slot` only fires when
the copy succeeds, so every slot stayed at -1 forever and the
cache had a measured 0.0% hit rate. Diagnosed via a populate-success
counter that showed `cum ok=0 fail=20710` over one decode.

`cudaMemcpyDefault` uses UVA (enabled on every supported CUDA device) to
auto-detect direction at the same throughput as explicit direction.
Solves both the miss-populate (HtoD) and hit-fetch (DtoD) paths through
one function.

### 4. Cross-DSO function pointer resolution via reg proc-address

`ggml-base` and `ggml-cuda` link as separate shared libraries. A strong
reference from `ggml-base` to a `ggml-cuda` symbol doesn't resolve
cross-DSO at runtime — the weak fallback in `ggml-base` wins, returning
false. The ggml convention for backend-specific entry points is
`ggml_backend_reg_get_proc_address(reg, "symbol_name")`. We use this for
all four S* hooks (and the cache D2D copy, and the active-cache setter,
and the min-batch-size setter).

### 5. Eviction policy matters less than expected

We benched round-robin, LRU, and SLRU at four cache sizes. SLRU was
within measurement noise of LRU. Conclusion: Qwen3.6-A3B routing has a
broad "warm pool" rather than a tight hot set; both policies converge to
similar steady states. We expose `--moe-cache-policy` so users can
explore other policies on other models, but the choice doesn't move our
headline number.

## Reference: prior research

- vLLM PR #37190 (e1n00r) — the closest prior art. Uses LFRU. Hit
  ~91.96% on different MoE models. Limited to BF16 / FP8 quants, won't
  fit on 32 GiB hardware with Qwen3.6 → why we built this in llama.cpp.
- `e1n00r/tinyserve` — standalone implementation behind PR #37190.
  Validated GGUF-compatible (Q4_K, Q5_K, Q6_K, Q8_0) but not MXFP4.
- `martinalderson/llama.cpp:moe-profile` — Vulkan-only prior attempt.
  Closed without CUDA implementation. Issue #20757 left open for CUDA
  backend; this branch is that implementation.
- `nick-pape/llama-moe-cache:wsl2-staging-pool` — prior 3090 / WSL2
  research (the H5 expert cache prototype).
- ExpertFlow paper — frequently-cited cache-policy paper, claims 91.96%
  hit rate vs LRU + 27.65pp on MoE workloads. Likely informs the LFU
  exploration above.

## Reproducing

Build the moe-cache builder image (CUDA 12.8 devel + cmake + ninja):

```bash
docker build -t llama-moe-builder:latest /root/llama-moe-builder
```

Build llama-cli:

```bash
docker run --rm --gpus all \
  -v $(pwd):/work -v /root/.ccache:/ccache -w /work \
  llama-moe-builder:latest \
  bash -c 'cmake -S . -B build -G Ninja \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_CURL=OFF \
   && cmake --build build -j 12 --target llama-cli'
```

Run a single-size bench (cache=128):

```bash
docker run --rm --gpus all \
  -v $(pwd):/work -v /path/to/gguf-cache:/cache -w /work \
  llama-moe-builder:latest \
  ./build/bin/llama-cli \
    --model /cache/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
    -ngl 999 \
    -ot 'blk\.\d+\.ffn.*exps=CPU' \
    --ctx-size 4096 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on \
    --batch-size 2048 --ubatch-size 2048 \
    --threads 4 --threads-batch 12 \
    --moe-expert-cache-size 128 \
    --prompt 'Write a 200-word essay on Renaissance painters.' \
    --n-predict 200 \
    --no-warmup -no-cnv --simple-io < /dev/null
```

Look for `moe-cache: final stats:` and `[ Prompt: ... | Generation: ... ]`
in stderr.
