# MTP Speculative Decoding — Bench Results (2026-05-15)

Companion to [`docs/moe-expert-cache.md`](./moe-expert-cache.md) and the
plan at `~/.claude/plans/rustling-kindling-wadler.md`. Bench was run to
evaluate whether the MTP support added by upstream PR
[#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (am17an,
**OPEN as of 2026-05-15**) delivers usable decode speedup on top of our
existing MoE-expert-weight cache, on the homelab's specific workload.

**TL;DR — strongly workload-dependent.** On the long code-review
prompt that's our default sweep workload, MTP regresses or barely
ties baseline (best: 72.5 vs 71.2 t/s = 1.018×). On a predictable
factual workload (US states list), the *same* MTP configuration
delivers **+28% (111.3 vs 86.9 t/s)**. Plan's acceptance gate of 1.5×
not met on either, but the gap is meaningful enough that MTP is worth
keeping in the toolbox for high-accept-rate workloads.

**Path A (self-convert MXFP4+MTP) is unnecessary** because Unsloth
already publishes the same MXFP4_MOE quant we use in prod with MTP
heads pre-baked in `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`. Branch
`mtp-experiment` (HEAD `7ae14a77a`) is preserved on
`nick-pape/llama.cpp`.

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
