"""Element-wise comparison of two isaac_audit.py --sample dumps (one frame's rgb, depth, lidar and the imu log).

    python compare_samples.py a_frame60.npz b_frame60.npz

Prints, per array: how many elements differ, and the median and maximum absolute difference over
the differing elements, in the array's own units (levels of 255 for rgb, metres for depth and
lidar ranges). Point clouds of unequal length are reported by their counts and compared over the
common prefix. The 4.5 numbers in the paper (camera pixels differing on 15.6%, median 1 level,
max 8; 2.4% of 3,600 flat-scan beams, median 1.5 mm, max 4.4 m) were computed this way.
"""
import sys

import numpy as np


def diff_stats(name, a, b, scale=1.0, unit=""):
    a, b = np.asarray(a), np.asarray(b)
    n = min(a.shape[0], b.shape[0])
    note = "" if a.shape == b.shape else f" (shapes {a.shape} vs {b.shape}; compared over the first {n})"
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    if a.ndim > 1 and name.startswith("rgb"):
        d = np.abs(a - b).max(-1)                # per pixel: the largest channel difference
        label = "pixels"
    else:
        d = np.abs(a - b).reshape(len(a), -1).max(-1) if a.ndim > 1 else np.abs(a - b)
        label = "elements"
    m = d > 0
    if not m.any():
        print(f"  {name:24s} identical over {d.size} {label}{note}")
        return
    print(f"  {name:24s} {m.sum()} of {d.size} {label} differ ({100.0 * m.mean():.1f}%), median {scale * np.median(d[m]):.3g}{unit}, "
          f"max {scale * d[m].max():.3g}{unit}{note}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    A, B = np.load(sys.argv[1]), np.load(sys.argv[2])
    print(f"{sys.argv[1]} vs {sys.argv[2]}")
    for k in sorted(set(A.files) | set(B.files)):
        if k not in A.files or k not in B.files:
            print(f"  {k:24s} only in {'A' if k in A.files else 'B'}")
            continue
        if k == "rgb":
            diff_stats("rgb (levels)", A[k], B[k])
        elif k == "depth":
            diff_stats("depth (m)", A[k], B[k])
        elif k == "lidar.point_cloud_data":
            print(f"  {'lidar points':24s} {len(A[k])} and {len(B[k])}")
            diff_stats("point cloud (m)", A[k], B[k])
        elif k in ("lidar.range", "lidar.linear_depth_data", "lidar.intensities_data"):
            diff_stats(k + (" (m)" if "intens" not in k else ""), A[k], B[k])
        elif k == "imu":
            diff_stats("imu (all columns)", A[k], B[k])
        else:
            diff_stats(k, A[k], B[k])


if __name__ == "__main__":
    main()
