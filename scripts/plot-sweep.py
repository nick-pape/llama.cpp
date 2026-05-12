#!/usr/bin/env python3
"""Parse the full-sweep results and generate two line charts.

Run after /root/full-sweep.sh completes. Reads /root/sweep-results/*.log,
extracts (cache_size, vram_mib, hit_pct, gen_tps, cpu_dispatch_pct) for
each config, and writes:
  - sweep-cache-size.png
  - sweep-cache-vram.png
  - sweep.csv

Usage: scp /root/sweep-results -> local, then run this against the dir.
"""

import os, re, csv, sys
from pathlib import Path

RE_GEN     = re.compile(r"\[ Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s ]")
RE_PROG    = re.compile(r"progress: (\d+) lookups, ([\d.]+)% hit rate, ([\d.]+) MiB allocated")
RE_DISP    = re.compile(r"dispatch breakdown: (\d+) total ops, (\d+) on CPU \(([\d.]+)%\), (\d+) on GPU \(([\d.]+)%\); avg misses-per-op when on GPU = ([\d.]+)")

def parse_log(path):
    text = path.read_text(errors='replace')
    row = {'log': path.name}
    if m := RE_GEN.search(text):
        row['prompt_tps'] = float(m.group(1))
        row['gen_tps']    = float(m.group(2))
    # take the LAST progress line (most representative of final state)
    progs = list(RE_PROG.finditer(text))
    if progs:
        row['lookups']  = int(progs[-1].group(1))
        row['hit_pct']  = float(progs[-1].group(2))
        row['vram_mib'] = float(progs[-1].group(3))
    if m := RE_DISP.search(text):
        row['total_ops']      = int(m.group(1))
        row['cpu_ops']        = int(m.group(2))
        row['cpu_dispatch_pct'] = float(m.group(3))
        row['gpu_ops']        = int(m.group(4))
        row['avg_miss_per_op_gpu'] = float(m.group(6))
    return row

def main():
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sweep-results")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

    rows = []
    for path in sorted(in_dir.glob("*.log")):
        name = path.stem            # e.g. "lru-16" / "lru-s3-128" / "baseline" / "gpu-only"
        if name == "gpu-only":
            config = "GPU only (no -ot)"
            cache = 0
        elif name.startswith("baseline"):
            config = "baseline"
            cache = 0
        elif name.startswith("lru-s3-"):
            config = "LRU+S3"
            cache = int(name.split("-")[-1])
        elif name.startswith("lru-"):
            config = "LRU only"
            cache = int(name.split("-")[-1])
        else:
            continue
        row = parse_log(path)
        row['config'] = config
        row['cache_size'] = cache
        rows.append(row)

    # CSV
    csv_path = out_dir / "sweep.csv"
    keys = ['config', 'cache_size', 'vram_mib', 'hit_pct', 'prompt_tps',
            'gen_tps', 'total_ops', 'cpu_dispatch_pct', 'avg_miss_per_op_gpu']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r['config'], r['cache_size'])):
            w.writerow(r)
    print(f"wrote {csv_path}")

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots. CSV is written.")
        return

    baseline_tps = next((r['gen_tps'] for r in rows
                        if r['config']=='baseline' and 'gen_tps' in r), None)
    gpu_only_tps = next((r['gen_tps'] for r in rows
                        if r['config']=='GPU only (no -ot)' and 'gen_tps' in r), None)
    by_config = {}
    for r in rows:
        if r['config'] in ('baseline', 'GPU only (no -ot)'):
            continue
        if 'gen_tps' not in r:
            continue
        by_config.setdefault(r['config'], []).append(r)

    for cfg, items in by_config.items():
        items.sort(key=lambda r: r['cache_size'])

    # Plot 1: x = cache_size (slots)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'LRU only': '#1f77b4', 'LRU+S3': '#d62728'}
    for cfg, items in by_config.items():
        xs = [r['cache_size'] for r in items]
        ys = [r['gen_tps']    for r in items]
        ax.plot(xs, ys, marker='o', label=cfg, color=colors.get(cfg))
    if baseline_tps:
        ax.axhline(baseline_tps, color='gray', linestyle='--',
                   label=f'CPU-MoE baseline = {baseline_tps:.1f} t/s')
    if gpu_only_tps:
        ax.axhline(gpu_only_tps, color='green', linestyle=':',
                   label=f'GPU-only ceiling = {gpu_only_tps:.1f} t/s')
    ax.set_xlabel('Cache size (slots per layer per bucket)')
    ax.set_ylabel('Decode tok/s')
    ax.set_title('MoE expert cache: decode throughput vs cache size\n'
                 'Qwen3.6-35B-A3B-MXFP4, Blackwell RTX PRO 4500, ctx=4096')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_dir / 'sweep-cache-size.png'
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    # Plot 2: x = VRAM (MiB)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg, items in by_config.items():
        xs = [r.get('vram_mib', 0) for r in items]
        ys = [r['gen_tps']         for r in items]
        ax.plot(xs, ys, marker='o', label=cfg, color=colors.get(cfg))
    if baseline_tps:
        ax.axhline(baseline_tps, color='gray', linestyle='--',
                   label=f'CPU-MoE baseline = {baseline_tps:.1f} t/s')
    if gpu_only_tps:
        ax.axhline(gpu_only_tps, color='green', linestyle=':',
                   label=f'GPU-only ceiling = {gpu_only_tps:.1f} t/s')
    ax.set_xlabel('Cache VRAM allocated (MiB)')
    ax.set_ylabel('Decode tok/s')
    ax.set_title('MoE expert cache: decode throughput vs cache VRAM\n'
                 'Qwen3.6-35B-A3B-MXFP4, Blackwell RTX PRO 4500, ctx=4096')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_dir / 'sweep-cache-vram.png'
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

if __name__ == '__main__':
    main()
