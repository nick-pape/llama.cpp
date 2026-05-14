#!/usr/bin/env python3
"""Compare ncu CSV outputs from cache=256 vs pure-GPU mul_mat kernel metrics."""
import csv
import sys
from collections import defaultdict
from statistics import mean, median

def parse(path):
    """Return list of (kernel_name, {metric_name: value}) tuples."""
    invocations = defaultdict(dict)
    kernel_names = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('"'):
                continue
            reader = csv.reader([line])
            row = next(reader)
            if len(row) < 15 or row[0] == 'ID':
                continue
            kid = row[0]
            kname_full = row[4]
            metric = row[12]
            value = row[14]
            kname = ('mul_mat_q' if 'mul_mat_q<' in kname_full and 'fixup' not in kname_full
                     else 'stream_k_fixup' if 'fixup' in kname_full
                     else kname_full[:30])
            kernel_names[kid] = kname
            try:
                v = float(value.replace(',', ''))
                invocations[kid][metric] = v
            except ValueError:
                pass
    return [(kernel_names[kid], invocations[kid]) for kid in sorted(invocations.keys(), key=int)]

def summarize(data, label):
    by_kernel = defaultdict(list)
    for kname, metrics in data:
        by_kernel[kname].append(metrics)
    print(f"\n=== {label} ===")
    for kname, invs in by_kernel.items():
        print(f"  {kname}: n={len(invs)}")
        for metric in ('gpu__time_duration.sum', 'sm__cycles_active.avg.pct_of_peak_sustained_elapsed',
                       'dram__bytes.sum', 'lts__t_sector_hit_rate.pct', 'l1tex__t_sector_hit_rate.pct'):
            vals = [m.get(metric) for m in invs if metric in m]
            if not vals:
                continue
            mn = min(vals); mx = max(vals); avg = mean(vals); med = median(vals)
            unit = {'gpu__time_duration.sum': 'ns', 'dram__bytes.sum': 'B'}.get(metric, '%')
            print(f"    {metric:55s}  avg={avg:>12.1f} {unit}  median={med:>12.1f}  min={mn:>10.1f}  max={mx:>10.1f}")
    return by_kernel

c = parse('cache256.csv')
g = parse('puregpu.csv')

cb = summarize(c, 'cache=256 (cache active, kernel reads pool_tensor)')
gb = summarize(g, 'pure-GPU (no offload, kernel reads model native tensor)')

# Side-by-side comparison of matched kernels
print("\n=== SIDE-BY-SIDE (cache=256 vs pure-GPU) ===")
for kname in cb:
    if kname not in gb:
        continue
    print(f"\n  Kernel: {kname}  (cache=256 n={len(cb[kname])}, pureGPU n={len(gb[kname])})")
    for metric in ('gpu__time_duration.sum', 'sm__cycles_active.avg.pct_of_peak_sustained_elapsed',
                   'dram__bytes.sum', 'lts__t_sector_hit_rate.pct', 'l1tex__t_sector_hit_rate.pct'):
        cv = [m.get(metric) for m in cb[kname] if metric in m]
        gv = [m.get(metric) for m in gb[kname] if metric in m]
        if not cv or not gv:
            continue
        cm = median(cv); gm = median(gv)
        delta = (cm - gm) / gm * 100 if gm else 0
        unit = {'gpu__time_duration.sum': 'ns', 'dram__bytes.sum': 'B'}.get(metric, '%')
        print(f"    {metric:55s}  cache256={cm:>12.1f}  pureGPU={gm:>12.1f}  delta={delta:+.1f}%  ({unit})")
