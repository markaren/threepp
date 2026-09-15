"""Compare isaac_audit.py manifests across fresh processes, aligned by the renderer's own delivery stamps.

    python compare_runs.py a.json b.json [c.json ...]

Isaac Sim 6.x delivers camera and lidar frames at their own cadence (the camera on about two of
three world steps, the Example_Rotary lidar every sixth), so a step-by-step comparison would
compare a fresh frame with a repeated one. This tool keeps, per row, the frames whose delivery
stamp (the camera's or lidar's ``rendering_time``, recorded by isaac_audit.py under ``stamps``)
advanced, and compares processes at equal stamps. Rows without stamps (imu, poses) are compared
step by step. Manifests without ``stamps`` (4.5, or the first 6.1 batch) fall back to the order
of distinct frames.
"""
import json
import sys


def deliveries(m, row):
    """[(stamp, hash)] for the frames whose stamp advanced; falls back to distinct-frame order."""
    pf = m["per_frame"][row]
    st = m.get("stamps", {})
    key = {"rgb": "cam_time", "rgb.noaa": "cam_time", "depth": "cam_time", "lidar": "lidar_time"}.get(row)
    times = st.get(key) if key else None
    if times and len(times) == len(pf):
        out, last = [], None
        for t, h in zip(times, pf):
            if t != last:
                out.append((round(float(t), 6), h))
                last = t
        return out, "stamp"
    out = []
    for i, h in enumerate(pf):
        if not out or out[-1][1] != h:
            out.append((len(out), h))
    return out, "order"


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        sys.exit(__doc__)
    ms = [json.load(open(p)) for p in paths]
    names = [p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".json", "") for p in paths]
    meta = ms[0]["meta"]
    print(f"{len(ms)} processes: {', '.join(names)}; isaac {meta.get('version')} kit {meta.get('kit')} {meta.get('graphics_api')} "
          f"{meta.get('rendermode')} aa={meta.get('aa')} {meta['frames']} steps")
    for row in ms[0]["rows"]:
        if row in ("imu", "poses"):
            seqs = [m["per_frame"][row] for m in ms]
            n = min(map(len, seqs))
            diff = [i for i in range(n) if len({s[i] for s in seqs}) > 1]
            full = len({(m["rows"][row]["fnv"] if isinstance(m["rows"][row], dict) else m["rows"][row]) for m in ms})
            print(f"  {row:8s} step by step: {n} steps, {len(diff)} differ (first {diff[:3]}); full-run hashes distinct {full} of {len(ms)}")
            continue
        dels = [deliveries(m, row) for m in ms]
        mode = dels[0][1]
        maps = [dict(d) for d, _ in dels]
        common = set(maps[0]).intersection(*maps[1:])
        counts = [len(d) for d, _ in dels]
        diff = sorted(t for t in common if len({mp[t] for mp in maps}) > 1)
        full = len({(m["rows"][row]["fnv"] if isinstance(m["rows"][row], dict) else m["rows"][row]) for m in ms})
        first = f"{diff[0]:.4f} s" if diff and mode == "stamp" else (str(diff[0]) if diff else "-")
        print(f"  {row:8s} delivered frames {counts} (aligned by {mode}); {len(common)} common; {len(diff)} differ at common "
              f"{'stamps' if mode == 'stamp' else 'positions'} (first {first}); full-run hashes distinct {full} of {len(ms)}")


if __name__ == "__main__":
    main()
