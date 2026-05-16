# MTP Speculative Decoding — Bench Results (2026-05-15)

Companion to [`docs/moe-expert-cache.md`](./moe-expert-cache.md) and the
plan at `~/.claude/plans/rustling-kindling-wadler.md`. Bench was run to
evaluate whether the MTP support added by upstream PR
[#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (am17an,
**OPEN as of 2026-05-15**) delivers usable decode speedup on top of our
existing MoE-expert-weight cache, on the homelab's specific workload.

**TL;DR — MTP works, but it's a regression on this card under SDXL
coexistence constraints.** At isolated test conditions
(parallel=1 ctx=8192 cache=96), MTP delivers +1.8% to +28% depending
on workload shape. But fitting MTP into a budget that *also* allows
SDXL on the same card requires dropping parallel 4→3 and cache 96→64,
which together cost MORE decode than MTP recovers (head-to-head loses
−10% to −31% across all workloads tested). The MTP economics need
cache=96 + parallel=4, which is exactly what we have to give up.

**Decision: stay on current prod (`cd9a2bd60`, no MTP, cache=96,
parallel=4).** Branch `mtp-experiment` (HEAD `7ae14a77a`) is
preserved on `nick-pape/llama.cpp` for the day the second GPU lands
(see "Pro 2000 unblocks this" below).

**Path A (self-convert MXFP4+MTP) is unnecessary** because Unsloth
already publishes the same MXFP4_MOE quant we use in prod with MTP
heads pre-baked in `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`.

**Pro 2000 unblocks this.** Once a second GPU dedicates ~16 GiB to
SDXL + embed-gpu + rerank-gpu + whisper-gpu, the 4500 gets all 32 GiB
for Qwen+MTP. At full cache=96 parallel=4 ctx=1M MTP-on, the +28% on
factual workloads + +14% on short prompts comes back for free with no
trade-offs. MTP becomes a clear prod default *only* in that hardware
configuration. The work captured here ports forward unchanged.

## Setup

- **Hardware:** RTX PRO 4500 Blackwell (sm_120, 32 GiB), via Proxmox LXC
- **Build:** `nick-pape/llama.cpp:mtp-experiment` @ `c5905e14e` —
  fork master + merge of PR #22673 (which subsumes #22400). CUDA 12.8.1.
- **Model:** `unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-MXFP4_MOE.gguf`
  (20.66 GiB; same MXFP4_MOE quant as our prod model, with MTP heads
  baked in — verified `blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`
  tensors + `qwen35moe.nextn_predict_layers` metadata key).
  - **Key finding from sourcing:** the unsloth MTP repo publishes the
    same MXFP4_MOE quant as the regular repo. Path A from the plan
    (self-convert + llama-quantize MXFP4_MOE) is unnecessary.
- **Flags (constant across runs):** `-ngl 999 -ot 'blk\.\d+\.ffn.*exps=CPU'
  --moe-expert-cache-size 96 --moe-cache-policy lru --ctx-size 8192
  --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
  --batch-size 2048 --ubatch-size 2048 --threads 4 --threads-batch 12
  --jinja`
- **Prompt:** `/root/sweep-prompt.txt` on ai.service — 26 KB (~6800
  tokens), asks the model to review the v2 MoE-cache C++ source.
  Generation cap: 1500 tokens.
- Prod `llama-qwen36-apex` was stopped during the bench so the
  experiment had the full GPU.

## Results

| Run | Prompt t/s | Decode t/s | Δ vs baseline | Main cache hit | MTP cache hit |
|---|---|---|---|---|---|
| **baseline** (no `--spec-type`, MTP-bearing GGUF) | 982.1 | **71.2** | 1.00× | 89.5% | n/a |
| `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75` | 760.4 | 70.1 | 0.98× | 86.5% | 91.1% |
| `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75` | 748.3 | 65.5 | 0.92× | 85.4% | 91.7% |
| `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75` | 757.6 | 63.2 | 0.89× | 86.8% | 91.8% |
| **`--spec-draft-n-max 2 --spec-draft-p-min 0.9` + MTP block kept on GPU** (`-ot blk\.([0-3]?[0-9])\.ffn.*exps=CPU`) | 772.7 | **72.5** | **1.018×** | 85.3% | (n/a on this run) |

A separate short-prompt smoke test (single-line "Write a short paragraph
about Paris") showed MTP@2 hitting 82.3 t/s vs 72.2 t/s baseline
(**+14%**) — so MTP *can* help, but the long code-generation workload
is the wrong shape for it.

### Workload-shape sensitivity (run after the main sweep)

To stress the workload hypothesis, ran the *same* MTP@2 +
p-min=0.9 + blk.40-on-GPU config on a predictable prompt: "List all 50
US states in alphabetical order, with the year each joined the union
in parentheses."

| Run (states list, n_predict=1500) | Prompt t/s | Decode t/s | Δ |
|---|---|---|---|
| Baseline (no MTP) | 62.8 | 86.9 | 1.00× |
| MTP@2 + p-min=0.9 + blk.40 on GPU | 58.1 | **111.3** | **1.28×** |

**MTP delivers a real +28% speedup on factual / list / structured
output**, while it ties or regresses on creative / code generation.
Same flags, same model, only the prompt's predictability differs.

Cache hit rate on the states prompt was higher across the board (91.1%
main vs 85.3% on code-review), reflecting that a tighter expert
distribution helps both the trunk cache AND MTP draft acceptance.
Same flags, only the prompt changes.

This is consistent with am17an's reference 0.72 accept rate on his
unspecified bench — predictable next-token distribution → high accept
→ MTP wins. Code generation has high entropy → low accept → MTP
overhead dominates.

## Analysis

The MoE-cache infrastructure survived the merge intact and even
performed slightly better at the no-MTP cache=96 baseline (71.2 t/s
here vs 68 t/s in `docs/v2-results.md` — likely upstream improvements
landed in the merge window). So the regression check passed cleanly.

MTP's economics on this hardware/workload don't pencil out:

1. **MTP draft head is one MoE block** at `blk.40` (`nextn_predict_layers
   == 1`). Under `-ot blk\.\d+\.ffn.*exps=CPU` it inherits expert
   offload; even keeping it on GPU (the bottom-row test) only buys
   +1.4 t/s over the offloaded-MTP variant.
2. **Accept rate is presumably low** for code-generation. am17an's
   reference 2.4× speedup was at accept rate 0.72 on different
   prompts; on creative/code-generation workloads the next token is
   high-entropy and drafts get rejected often. We couldn't measure
   accept rate directly — `llama-cli` doesn't print it in this build
   without an explicit verbose flag.
3. **The trunk's per-token cost is already low** thanks to the MoE
   cache (89% hit rate, only ~85 MiB H2D per miss). The denominator
   in the speedup formula `(T) / ((T + N·M) / (1 + N·A))` is small,
   so even modest draft overhead matters.
4. **n_max scales the wrong way.** Speedup *decreases* as we increase
   draft horizon (0.98× → 0.92× → 0.89×) — consistent with low accept
   rate where larger drafts are mostly wasted work.

## What we *did* learn

- The fork's MoE cache is fully compatible with PR #22673's MTP path —
  the slot-pool substitution fires for both the trunk MoE blocks and
  the MTP head's MoE block automatically (no fork-side changes needed
  beyond carrying our v2 patches forward through the merge).
- The merge of PR #22673 onto our fork was clean (single hand-merge in
  `common/arg.cpp` and `src/llama-context.cpp`; auto-resolved by git).
- The MTP cache hit rate is high (~91%) — the slot pool serves the MTP
  head's experts effectively when MTP runs.
- Prefill throughput regresses moderately under MTP (982 → 760 t/s) —
  expected, since prefill doesn't benefit from speculative decoding
  but pays the GDN-rollback memory bookkeeping.

## Head-to-head at prod-shape budget (2026-05-15, evening)

After the workload-shape finding, the real question became: can we
**enable MTP as the prod default** given the VRAM cost?

### Measured cost at full prod settings (parallel=4 ctx=1048576 q8/q8 KV)

| Config | Process VRAM | Δ vs no-MTP baseline |
|---|---|---|
| Baseline (MTP off, mmproj on) | 24,926 MiB | — |
| MTP on + blk.40 on GPU | 30,212 MiB | **+5,286 MiB (~5.2 GiB)** |

Breakdown of the 5.2 GiB MTP cost at prod shape (empirically probed
with `-ctkd q8_0 -ctvd q8_0` and `--spec-draft-cpu-moe`):
- MTP head MoE expert weights (offloadable to CPU): ~0.3 GiB (measured)
- MTP draft KV (quant change had **zero** effect, so likely tiny —
  scales with `--spec-draft-n-max` tokens, not full ctx): negligible
- MTP non-MoE weights (attention norms, no flag): ~few hundred MiB
- **~4.5 GiB hides in compute-buffer expansion + spec-decode
  scheduler bookkeeping + extra context allocation — no flag exposes
  it.** Would require patching the spec-decode allocator to reduce.

### Budget against ComfyUI/SDXL coexistence

SDXL measured cost on the same card (image-gen via litellm → comfy shim → comfyui):
- Total GPU before: 437 MiB (driver only)
- Total GPU during/post gen: 7,061 MiB
- **SDXL exact footprint: 6,624 MiB (~6.5 GiB)** — and `comfyui /free` releases it cleanly back to 437 MiB.

Budget for Qwen+MTP to coexist with SDXL: 32,134 − 437 − 6,624 = **25,073 MiB**.

### Knob sweep to fit budget (parallel=3 + cache shrink, MTP on)

| cache | proc VRAM (MiB) | fits 25,073 MiB? |
|---|---|---|
| 96 | 27,030 | ❌ over by 2.0 GiB |
| 88 | 26,466 | ❌ over by 1.4 GiB |
| 80 | 25,896 | ❌ over by 0.8 GiB |
| 72 | 25,332 | ❌ over by 0.3 GiB |
| **64** | **24,846** | ✅ fits with 227 MiB margin |
| 56 | 24,120 | ✅ fits with 953 MiB margin |

To fit MTP alongside SDXL on this card requires **parallel 4→3 + cache 96→64** (plus other smaller compromises).

### Head-to-head: original prod vs MTP-tuned at budget

| workload | A: prod (parallel=4 cache=96 no MTP) | B: MTP-tuned (parallel=3 cache=64 MTP-on) | Δ decode | Δ prompt |
|---|---|---|---|---|
| code (long, creative) | 68.5 t/s | 47.4 t/s | **−31%** | −30% |
| list (factual) | 84.5 t/s | 76.3 t/s | **−10%** | −20% |
| summ (structured prose) | 66.3 t/s | 56.2 t/s | **−15%** | −22% |

**B regresses on every workload.** The cache shrink + parallel drop required to make room for MTP costs more decode than MTP recovers — even on the "list" workload where MTP previously gave +28% (at cache=96 parallel=1). The MTP win requires holding cache=96, which we can't afford alongside SDXL on this card.

## Decision

Per the plan's acceptance gate (1.5× decode) and decision point #2:

- **Path A (self-convert MXFP4_MOE+MTP)** — unnecessary regardless,
  since `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` already publishes the same
  MXFP4_MOE quant we use in prod with MTP heads pre-baked.
- **Production deployment as a default** — no. The code-generation /
  agentic-tool-call workloads we run most don't benefit (or regress).
  Prod stack continues to run the v2-merge `cd9a2bd60` build with the
  existing MoE cache.
- **MTP as a *conditional* path** — worth considering. If a workload
  is known to be high-accept-rate (factual recall, structured output,
  retrieval summarization), enabling MTP is a +28% throughput win
  with the right config (`--spec-draft-n-max 2 --spec-draft-p-min 0.9`,
  MTP block kept on GPU via `-ot 'blk\.([0-3]?[0-9])\.ffn.*exps=CPU'`).
  Could be implemented as a separate LiteLLM model name pointing at a
  parallel `llama-server` instance with MTP enabled.
- **Branch lifecycle** — `mtp-experiment` stays on `nick-pape/llama.cpp`.
  Re-test triggers:
  1. PR #22673 merges upstream and the API stabilizes (we'd rebase).
  2. We add a workload-aware routing layer that picks MTP-on for
     high-predictability requests. (Symmetric with the
     embed-cpu/embed-gpu routing pattern in
     [[project_prod_ai_stack]].)
  3. The mmproj SIGSEGV bug is fixed (would let us bench with vision).

## Open observations worth following up

- A shorter-prompt smoke (Paris paragraph) showed +14%. Bench across
  more workload types — fact recall, summarization, structured output —
  would clarify where MTP wins vs loses on this stack.
- `--spec-draft-stats` or equivalent to print accept rate per run. This
  build's `llama-cli` output didn't include it; would need to wire it
  up or use a server with the `/metrics` endpoint.
- am17an's PR is still open and iterating (HEAD `a957b7747` adds
  review fixes); a future revision may behave differently.

## Reproducing

Sweep script + per-run logs preserved at `/root/mtp-sweep.sh` +
`/root/mtp-results/` on `ai.service.pape.house`. Image
`llama-mtp:c5905e14e` exists locally on that host. To re-bench from
scratch: `bash /root/mtp-sweep.sh`.

---

# SHIPPED 2026-05-16 — MTP live in prod

The earlier sections above were from a cli-only investigation where MTP
turned out to be **silently inactive** (cli has no `common_speculative_init`
wiring — only `tools/server` and `examples/speculative-simple` do).
Re-bench through `llama-server` showed MTP is a real win. Then a small
patch to the MTP draft ctx's `n_ubatch` reclaimed enough VRAM to fit
MTP-on alongside SDXL at full prod shape (par=3, cache=120). Now live.

## Final config (live in prod)

`nick-pape/llama.cpp:mtp-experiment` HEAD `003d36897`. Two commits on
top of master's MoE cache v2 work:

- `cbf6ba9ae` — merge of upstream PR #22673 (am17an MTP)
- `003d36897` — single-hunk patch capping MTP draft ctx's `n_ubatch`
  to `max(64, n_seq_max * (1 + n_max) + 4)` in `tools/server/server-context.cpp`.
  Reduces MTP overhead from +5.3 GiB to +2.1 GiB on Qwen3.6-35B-A3B-MTP
  at our prod shape. Proposed upstream as
  [PR #22673 comment](https://github.com/ggml-org/llama.cpp/pull/22673#issuecomment-4465608286).

Prod flags:

```
--model        Qwen3.6-35B-A3B-MTP-MXFP4_MOE.gguf
--mmproj       mmproj-F16.gguf
-ngl 999 -ot 'blk\.\d+\.ffn.*exps=CPU'
--moe-expert-cache-size 120 --moe-cache-policy lru
--ctx-size 786432 --parallel 3
--cache-type-k q8_0 --cache-type-v q8_0
--flash-attn on --batch-size 2048 --ubatch-size 2048 --kv-offload
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75
--threads 4 --threads-batch 12 --jinja
```

## Final A vs B++ measurement (server-based, SDXL-coexistence verified)

Hardware: single RTX PRO 4500 32 GiB (Blackwell sm_120). Same prompts
across both configs; A baseline measured with SDXL coresident, B++ with
SDXL recently unloaded (separately verified to coexist at peak 30,966 MiB
under live image-gen — see "SDXL coexistence" below).

| Workload | A: prior prod (par=4 c96 no-MTP) | B++: new prod (par=3 c120 MTP-on) | Δ decode | Accept | pp_tps Δ |
|---|---|---|---|---|---|
| code (800t)      | 62.2 t/s @ 30,566 | **83.7 t/s @ 25,080** | **+34.6%** | 64.5% | 967 → 697 (-28%) |
| list (800t)      | 81.8 t/s          | **123.9 t/s**          | **+51.6%** | 88.5% | 70 → 85 (+22%) |
| summ (800t)      | 61.8 t/s          | **91.8 t/s**           | **+48.6%** | 80.8% | 76 → 89 (+17%) |
| realistic (1500t)| 65.4 t/s          | **88.0 t/s**           | **+34.6%** | 70.0% | 972 → 822 (-15%) |

## SDXL coexistence (the binding budget)

Single-card budget for Qwen alongside SDXL is `32134 - 437 driver - 6293 SDXL = 25,404 MiB`.

Measured under realistic conditions (image-gen via litellm → ComfyUI
loaded SDXL; then started patched MTP container with cache=120; then
triggered second image-gen while MTP was held resident):

| Stage | VRAM used | Free | Notes |
|---|---|---|---|
| SDXL alone (resident) | 6,293 MiB | 25,841 | Baseline cost of holding SDXL warm |
| + MTP c120 booted     | 30,476 MiB | 1,658 | Both loaded — boots in 5s |
| During code workload  | **30,966 peak** | 1,168 | Inference completes cleanly |
| + 2nd image-gen request | 31,190 MiB | 944 | Both functional through real traffic |

Both inference paths return HTTP 200 with valid content. The +35% on
code held even under SDXL pressure (86.96 t/s during the coexistence run).

## Cache size selection (par=3 with patched MTP-on)

Code workload at par=3, ctx=786K, ub=2048, MTP-on patched:

| Cache | dec_tps | pp_tps | peak | vs c96 | fits ≤25,073? |
|---|---|---|---|---|---|
| 96  | 70.9 | 668 | 23,450 | — | yes (-1,623 under) |
| 104 | 74.8 | 669 | 23,948 | +5.5% | yes (-1,125) |
| 112 | 77.3 | 668 | 24,526 | +9.0% | yes (-547) |
| **120** | **82.6** | **689** | **25,110** | **+16.5%** | yes (verified live alongside SDXL) |
| 128 | 88.1 | 690 | 25,848 | +24.3% | no (+775) |
| 144 | 96.1 | 670 | 26,752 | +35.5% | no |
| 160 | 101.2 | 709 | 27,908 | +42.7% | no |

Decode scales linearly with cache; PP stays flat. c120 is the largest
that fits alongside SDXL under live coexistence.

## Ubatch sweep at c120

Target `--ubatch-size` higher than 2048 doesn't help:

| ubatch | dec_tps | pp_tps | peak | notes |
|---|---|---|---|---|
| **2048** | **81.7** | 690 | 24,496 | optimum |
| 3072 | 83.2 | 740 | 26,182 | +7% PP costs +1.7 GiB, over budget |
| 4096 | 76.8 | 635 | 27,830 | WORSE on both metrics, way over |

## mmproj + MTP

The known upstream SIGSEGV in `handle_mtp_for_ubatch` for vision
requests **does not reproduce on our build**. Verified end-to-end: image
request to `/v1/chat/completions` with `image_url` data: URL returned
HTTP 200 with the model correctly reading the test image (recognized
blue background + "HELLO" text), MTP fired at 84% accept rate during
vision decode, server stayed up. Likely the bug was fixed in the commit
range we merged.

## Patches not pursued

Two other VRAM-reduction candidates from the same investigation were
rejected at our shape:

- **Cap `n_rs_seq`** (`common/common.h::need_n_rs_seq`): a no-op at our
  `--spec-draft-n-max 2`. The cap to 4 would only help users with the
  default `n_max=16`.
- **Quantize RS cache to bf16** (`src/llama-model.cpp` recurrent cache
  type): would silently corrupt GDN recurrent state because
  `ggml_cuda_op_gated_delta_net` hardcodes `(const float*)` casts on
  state tensors. Would require rewriting the GDN CUDA kernel for bf16.
  Not pursued.
