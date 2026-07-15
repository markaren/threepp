#!/usr/bin/env python3
"""Cross-check a region pack: NVDB road y-heights vs DTM sampled at (x,z).

This is the load-bearing orientation proof.  If row order or the z sign is
wrong, sampled DTM heights won't match the road's own elevations and the
median |dh| will be large.  Usage: python verify_pack.py <pack_dir>
"""
import json
import os
import sys

import numpy as np


def load(pack_dir):
    with open(os.path.join(pack_dir, "region.json"), encoding="utf-8") as f:
        region = json.load(f)
    dim = region["dim"]
    h = np.fromfile(os.path.join(pack_dir, "heights.f32"), dtype="<f4")
    assert h.size == dim * dim, f"heights size {h.size} != {dim*dim}"
    h = h.reshape(dim, dim)  # [iz, ix], row 0 = north
    with open(os.path.join(pack_dir, "roads.json"), encoding="utf-8") as f:
        roads = json.load(f)["roads"]
    return region, h, roads


def sample_bilinear(h, region, x, z):
    """Sample the DTM at local world (x,z). ix=(x+S/2)/res, iz=(z+S/2)/res."""
    dim = region["dim"]
    size = region["worldSize"]
    res = size / (dim - 1)
    fx = (x + size / 2.0) / res
    fz = (z + size / 2.0) / res
    if fx < 0 or fz < 0 or fx > dim - 1 or fz > dim - 1:
        return None
    ix0 = int(np.floor(fx)); iz0 = int(np.floor(fz))
    ix1 = min(ix0 + 1, dim - 1); iz1 = min(iz0 + 1, dim - 1)
    tx = fx - ix0; tz = fz - iz0
    a = h[iz0, ix0] * (1 - tx) + h[iz0, ix1] * tx
    b = h[iz1, ix0] * (1 - tx) + h[iz1, ix1] * tx
    return float(a * (1 - tz) + b * tz)


def main(pack_dir):
    region, h, roads = load(pack_dir)
    print(f"pack {region['name']}: dim={region['dim']} "
          f"hMin={region['heightMin']} hMax={region['heightMax']} "
          f"roads={len(roads)}")

    def stats(zsign, label):
        diffs = []
        for rd in roads:
            for x, y, z in rd["points"]:
                dtm = sample_bilinear(h, region, x, zsign * z)
                if dtm is not None:
                    diffs.append(abs(y - dtm))
        d = np.array(diffs)
        print(f"  [{label}] n={len(d)} median|dh|={np.median(d):.2f} "
              f"mean={np.mean(d):.2f} p90={np.percentile(d,90):.2f} "
              f"max={np.max(d):.2f}")
        return np.median(d)

    # Correct mapping uses z as-is. A wrong z sign should blow up the error.
    m_ok = stats(+1, "z as-is (correct)")
    m_flip = stats(-1, "z flipped (sanity)")
    print(f"  -> correct/flipped median ratio = {m_ok/max(m_flip,1e-6):.3f}")

    npts = sum(len(r["points"]) for r in roads)
    cats = {}
    for r in roads:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"  total road points={npts}  categories={cats}")
    for name in ("region.json", "heights.f32", "roads.json"):
        p = os.path.join(pack_dir, name)
        print(f"  {name}: {os.path.getsize(p)} bytes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\dev\threepp\geodata\trollstigen")
