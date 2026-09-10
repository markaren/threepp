"""Compare two dumped frames: python diff_frames.py a.npy b.npy [out.png]"""
import sys
import numpy as np

a = np.load(sys.argv[1]).astype(np.int16)
b = np.load(sys.argv[2]).astype(np.int16)
print("shape", a.shape, b.shape, "dtype in file", np.load(sys.argv[1]).dtype)
d = np.abs(a - b)
if d.ndim == 3:
    dm = d.max(axis=2)
else:
    dm = d
ys, xs = np.nonzero(dm)
n = len(ys)
print(f"differing pixels: {n} of {dm.size} ({100.0 * n / dm.size:.3f}%)")
if n:
    print(f"max abs diff {int(dm.max())}, mean over differing {dm[ys, xs].mean():.2f}")
    print(f"bbox x {xs.min()}..{xs.max()}  y {ys.min()}..{ys.max()}")
    hist = np.bincount(dm[ys, xs].astype(np.int64), minlength=9)
    print("diff histogram (1..8+):", hist[1:9].tolist(), "over 8:", int(hist[9:].sum()) if len(hist) > 9 else 0)
    # coarse map: 16x9 grid of counts
    H, W = dm.shape
    g = np.zeros((9, 16), np.int64)
    for y, x in zip(ys, xs):
        g[min(y * 9 // H, 8), min(x * 16 // W, 15)] += 1
    print("grid of differing pixel counts (rows top->bottom):")
    for row in g:
        print("  " + " ".join(f"{v:6d}" for v in row))
    if len(sys.argv) > 3:
        try:
            from PIL import Image
            img = np.zeros((H, W, 3), np.uint8)
            img[..., 0] = np.clip(dm * 32, 0, 255)
            img[..., 1] = (a.max(axis=2) // 3).astype(np.uint8) if a.ndim == 3 else (a // 3).astype(np.uint8)
            Image.fromarray(img).save(sys.argv[3])
            print("wrote", sys.argv[3])
        except Exception as e:  # noqa: BLE001
            print("no png:", e)
