"""Smaller copies of a trained 3DGS .ply without retraining: SH cut to a lower degree, splats under an opacity
dropped, or both. What a lighter delivery format would save before any quantisation (score.py scores the copies).

    py -3.14 python/examples/splat_mapping/slim.py --splats RUN/seed0/splats.ply --sh-degree 1 --min-opacity 0.02 --out X.ply
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--splats", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sh-degree", type=int, default=3)
ap.add_argument("--min-opacity", type=float, default=0.0, help="drop splats whose sigmoid(opacity) is below this")
args = ap.parse_args()


def read_ply(path):
    with open(path, "rb") as f:
        names, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        arr = np.fromfile(f, "<f4", count=n * len(names)).reshape(n, len(names))
    return names, arr


names, arr = read_ply(args.splats)
col = {k: i for i, k in enumerate(names)}
rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
per_channel = len(rest) // 3                              # INRIA layout: channel-major
keep_per_channel = (args.sh_degree + 1) ** 2 - 1
keep_rest = [f"f_rest_{c * per_channel + j}" for c in range(3) for j in range(keep_per_channel)]
out_names = [k for k in names if not k.startswith("f_rest_")]
out_names = out_names[:out_names.index("f_dc_2") + 1] + [f"f_rest_{i}" for i in range(len(keep_rest))] + \
    out_names[out_names.index("f_dc_2") + 1:]
src = [col[k] for k in names if not k.startswith("f_rest_")]
src = src[:src.index(col["f_dc_2"]) + 1] + [col[k] for k in keep_rest] + src[src.index(col["f_dc_2"]) + 1:]
opa = 1.0 / (1.0 + np.exp(-arr[:, col["opacity"]]))
rows = opa >= args.min_opacity
out = np.ascontiguousarray(arr[rows][:, src])
with open(args.out, "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(out)}\n"
             + "".join(f"property float {k}\n" for k in out_names) + "end_header\n").encode("ascii"))
    out.astype("<f4").tofile(f)
info = {"in": os.path.basename(args.splats), "splats_in": int(len(arr)), "splats_out": int(len(out)),
        "sh_degree": args.sh_degree, "min_opacity": args.min_opacity, "floats_per_splat": len(out_names),
        "bytes_in": os.path.getsize(args.splats), "bytes_out": os.path.getsize(args.out)}
print(json.dumps(info))
with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=1)
