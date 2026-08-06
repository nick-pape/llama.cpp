#!/usr/bin/env python3
"""Plot v2 sweep: t/s vs cache size with horizontal reference lines."""
import matplotlib.pyplot as plt

# Sweep results 2026-05-13 — model: Qwen3.6-A3B-MXFP4_MOE,
# prompt: 6814-token, decode: 500 tokens, --batch 2048 --ctx 8192,
# RTX PRO 4500 32GiB, llama-cpp-moe-cache commit 89dda80.
sizes  = [0,   8,   16,   32,   64,   96,   128,  160,  192,  224,   256]
decode = [37.4, 17.4, 20.4, 23.4, 34.7, 55.7, 73.7, 87.9, 90.6, 104.5, 115.6]
prefill= [989, 967, 979, 979, 979, 973, 977, 979, 979, 1060, 2467]
hit_pct= [None, 5.0, 21.1, 33.7, 60.7, 81.8, 90.2, 94.2, None, 94.6, 95.0]
vram   = [0,  0.58, 1.14, 2.24, 4.46, 6.67, 8.88, 10.83, 13.0, 15.53, 17.74]
GPU_CEILING = 147.4
CPU_BASELINE = 37.4

fig, (ax_decode, ax_pre) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- decode plot ---
ax_decode.plot(sizes, decode, 'o-', color='#1f77b4', linewidth=2, markersize=7, label='v2 cache decode t/s')
ax_decode.axhline(GPU_CEILING, color='#2ca02c', linestyle='--', linewidth=1.5,
                  label=f'GPU-only ceiling ({GPU_CEILING} t/s)')
ax_decode.axhline(CPU_BASELINE, color='#d62728', linestyle='--', linewidth=1.5,
                  label=f'CPU-offload baseline ({CPU_BASELINE} t/s)')

for x, y, h in zip(sizes, decode, hit_pct):
    label = f'{y}'
    if h is not None:
        label += f'\n{h:.0f}% hit'
    ax_decode.annotate(label, xy=(x, y), xytext=(0, 8), textcoords='offset points',
                       fontsize=8, ha='center')

ax_decode.set_ylabel('Decode tokens/sec', fontsize=11)
ax_decode.set_title('MoE Expert Cache v2 — Qwen3.6-A3B-MXFP4 — RTX PRO 4500 32 GiB\n'
                    '6814-token prompt, 500 decode tokens', fontsize=12)
ax_decode.legend(loc='lower right', fontsize=9)
ax_decode.grid(True, alpha=0.3)
ax_decode.set_ylim(0, 160)

# --- prefill plot ---
ax_pre.plot(sizes, prefill, 's-', color='#ff7f0e', linewidth=2, markersize=7, label='v2 cache prefill t/s')
for x, y in zip(sizes, prefill):
    ax_pre.annotate(f'{int(y)}', xy=(x, y), xytext=(0, 8), textcoords='offset points',
                    fontsize=8, ha='center')
ax_pre.set_ylabel('Prefill tokens/sec', fontsize=11)
ax_pre.set_xlabel('Cache size (slots per layer-bucket)\nVRAM cost (GiB) shown below', fontsize=11)
ax_pre.legend(loc='lower right', fontsize=9)
ax_pre.grid(True, alpha=0.3)

# Add second x-axis with VRAM annotations
secax = ax_pre.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
secax.set_xticks(sizes)
secax.set_xticklabels([f'{v:.1f}' if v else '0' for v in vram])
secax.spines['bottom'].set_position(('outward', 40))
secax.set_xlabel('VRAM cost (GiB)', fontsize=10)

ax_pre.set_xticks(sizes)
ax_pre.set_xticklabels([str(s) for s in sizes])

plt.tight_layout()
plt.savefig('v2-sweep-plot.png', dpi=120, bbox_inches='tight')
print('Saved: v2-sweep-plot.png')
