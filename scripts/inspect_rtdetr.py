"""
Dump RT-DETR-R18 architecture & state_dict for WGSL implementation planning.

Usage:
    pip install ultralytics
    python scripts/inspect_rtdetr.py
"""

from ultralytics import RTDETR
import torch

# Downloads on first run (~50 MB). Other variants: rtdetr-l.pt, rtdetr-x.pt.
model = RTDETR('rtdetr-l.pt')  # R18 variant is not officially released by Baidu/Ultralytics;
                                # 'l' is the smallest Ultralytics-exported option.
m = model.model

print("=" * 70)
print("Module tree:")
print("=" * 70)
print(m)

print("\n" + "=" * 70)
print("state_dict summary (name, shape, dtype):")
print("=" * 70)
sd = m.state_dict()
total = 0
for k, v in sd.items():
    print(f"  {k:65s}  {tuple(v.shape)}  {v.dtype}")
    total += v.numel()
print(f"\nTotal params: {total/1e6:.2f} M")

print("\n" + "=" * 70)
print("Unique module types:")
print("=" * 70)
types = set()
for name, mod in m.named_modules():
    types.add(type(mod).__name__)
for t in sorted(types):
    print(f"  {t}")
