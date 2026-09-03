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

# The packs live in the repo's (gitignored) geodata/ root -- the same place
# examples/extras/terrain/norway_terrain.cpp reads as PROJECT_FOLDER/geodata.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(pack_dir):
    with open(os.path.join(pack_dir, "region.json"), encoding="utf-8") as f:
        region = json.load(f)
    dim = region["dim"]
    h = np.fromfile(os.path.join(pack_dir, "heights.f32"), dtype="<f4")
    assert h.size == dim * dim, f"heights size {h.size} != {dim*dim}"
    h = h.reshape(dim, dim)  # [iz, ix], row 0 = north
    with open(os.path.join(pack_dir, "roads.json"), encoding="utf-8") as f:
        roads = json.load(f)["roads"]
    buildings = []
    bpath = os.path.join(pack_dir, "buildings.json")
    if os.path.exists(bpath):
        with open(bpath, encoding="utf-8") as f:
            buildings = json.load(f)["buildings"]
    return region, h, roads, buildings


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


def verify_buildings(region, h, buildings):
    """Building checks: stored ground vs DTM sampled at the footprint centroid
    (the z-orientation proof for buildings), bounds, height stats."""
    half = region["worldSize"] / 2.0

    def stats(zsign, label):
        diffs = []
        for b in buildings:
            ring = b["outer"]
            cx = sum(p[0] for p in ring) / len(ring)
            cz = sum(p[1] for p in ring) / len(ring)
            dtm = sample_bilinear(h, region, cx, zsign * cz)
            if dtm is None:
                continue
            mid = 0.5 * (b["groundMin"] + b["groundMax"])
            diffs.append(abs(mid - dtm))
        d = np.array(diffs)
        print(f"  [{label}] n={len(d)} median|dg|={np.median(d):.2f} "
              f"mean={np.mean(d):.2f} p90={np.percentile(d,90):.2f} "
              f"max={np.max(d):.2f}")
        return np.median(d)

    m_ok = stats(+1, "bld ground z as-is (correct)")
    m_flip = stats(-1, "bld ground z flipped (sanity)")
    print(f"  -> correct/flipped median ratio = {m_ok/max(m_flip,1e-6):.3f}")

    hts = np.array([b["height"] for b in buildings])
    srcs = {}
    oob = bad_ground = 0
    for b in buildings:
        srcs[b["heightSource"]] = srcs.get(b["heightSource"], 0) + 1
        if b["groundMin"] > b["groundMax"]:
            bad_ground += 1
        for x, z in b["outer"]:
            if not (-half - 100 <= x <= half + 100 and
                    -half - 100 <= z <= half + 100):
                oob += 1
                break
    print(f"  heights: min={hts.min():.1f} median={np.median(hts):.1f} "
          f"p90={np.percentile(hts,90):.1f} max={hts.max():.1f} m")
    print(f"  sources={srcs}  groundMin>groundMax: {bad_ground}  "
          f"far-out-of-bounds: {oob}")


def main(pack_dir):
    region, h, roads, buildings = load(pack_dir)
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

    if buildings:
        print(f"buildings: {len(buildings)}")
        verify_buildings(region, h, buildings)

    if "texture" in region:
        tpath = os.path.join(pack_dir, region["texture"])
        try:
            from PIL import Image
            with Image.open(tpath) as im:
                w, hh = im.size
            mpp = region["worldSize"] / w
            square = "square" if w == hh else f"NOT SQUARE ({w}x{hh})"
            print(f"texture: {region['texture']} {w}x{hh} px, {square}, "
                  f"{mpp:.2f} m/px, layer={region.get('textureLayer', '?')}")
        except ImportError:
            print(f"texture: {region['texture']} (pillow not installed; "
                  f"dimensions not checked)")

    for name in ("region.json", "heights.f32", "roads.json", "buildings.json",
                 "texture.png"):
        p = os.path.join(pack_dir, name)
        if os.path.exists(p):
            print(f"  {name}: {os.path.getsize(p)} bytes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(REPO_ROOT, "geodata", "trollstigen"))
