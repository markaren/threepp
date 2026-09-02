"""Fetch the Calico Tanks trail scan the Calico examples walk on.

"Calico Tanks Trail, Red Rock Canyon, Las Vegas NV (XGRIDS PortalCam)" by tosolini,
published on SuperSplat under CC BY 4.0: https://superspl.at/scene/19312f07

SuperSplat's Download button needs an account. The viewer itself streams the very
same asset, a splat-transform "streamed SOG" (lod-meta.json + one directory per
chunk + env/), from a public CDN, and that is what this script pulls: about 430 MB,
72 chunk directories, 360 WebP planes. threepp's SogLoader reads the directory as
is (SogLoader.is_sog(dir) is true), so nothing is converted.

    py -3.14 fetch_calico_asset.py                # -> C:/dev/splats/calico_tanks (or $THREEPP_CALICO_ASSET)
    py -3.14 fetch_calico_asset.py --out D:/scans/calico_tanks

Re-running skips files already present. An ATTRIBUTION.txt is written beside the
data; keep it with any redistribution, the licence asks for the credit.
"""
import argparse
import concurrent.futures as cf
import json
import os
import urllib.request

SCENE_ID = "19312f07"
BASE = f"https://d28zzqy0iyovbz.cloudfront.net/{SCENE_ID}/v1/"
DEFAULT_OUT = os.environ.get("THREEPP_CALICO_ASSET", "C:/dev/splats/calico_tanks")

ATTRIBUTION = (
    "Calico Tanks Trail, Red Rock Canyon, Las Vegas NV (XGRIDS PortalCam)\n"
    f"by tosolini - https://superspl.at/scene/{SCENE_ID}\n"
    "Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)\n"
    "Fetched as the SuperSplat streamed SOG (splat-transform v3.1.7) the viewer reads.\n"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"target directory (default {DEFAULT_OUT})")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    def get(rel):
        dst = os.path.join(out, rel.replace("/", os.sep))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return rel, os.path.getsize(dst), "cached"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with urllib.request.urlopen(BASE + rel, timeout=120) as r, open(dst + ".part", "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(dst + ".part", dst)
        return rel, os.path.getsize(dst), "ok"

    get("lod-meta.json")
    lod = json.load(open(os.path.join(out, "lod-meta.json")))
    metas = list(lod["filenames"]) + [lod["environment"]]
    for m in metas:
        get(m)
    files = []
    for m in metas:
        d = os.path.dirname(m)
        j = json.load(open(os.path.join(out, m.replace("/", os.sep))))
        for v in j.values():
            if isinstance(v, dict) and "files" in v:
                files += [f"{d}/{fn}" for fn in v["files"]]
    print(f"{len(metas)} chunks, {len(files)} files -> {out}", flush=True)
    total = 0
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for i, (rel, n, st) in enumerate(ex.map(get, files), 1):
            total += n
            if i % 40 == 0 or i == len(files):
                print(f"{i}/{len(files)} {total / 1e6:.1f} MB", flush=True)
    with open(os.path.join(out, "ATTRIBUTION.txt"), "w", encoding="utf-8") as f:
        f.write(ATTRIBUTION)
    print(f"done, {total / 1e6:.1f} MB on disk")


if __name__ == "__main__":
    main()
