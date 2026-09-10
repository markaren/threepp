"""Group the diagnostic runs per variant: row hashes, per-frame agreement, and the LOD chain timeline."""
import collections, glob, json, os, sys

d = sys.argv[1]
runs = {}
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    if os.path.basename(p) == "diag_meta.json":
        continue
    runs[os.path.basename(p)[:-5]] = json.load(open(p))
mp = os.path.join(d, "diag_meta.json")
meta = json.load(open(mp))["runs"] if os.path.exists(mp) else {}
byvar = collections.defaultdict(list)
for n in runs:
    byvar[n.rsplit("_", 1)[0]].append(n)
for var, names in byvar.items():
    print(f"== {var}: {len(names)} runs ==")
    for row in ["rgb", "tip.rgb", "aov.depth", "aov.ids", "events.sorted", "traj", "imu"]:
        g = collections.defaultdict(list)
        for n in names:
            g[runs[n]["rows"][row]["fnv"]].append(n.rsplit("_", 1)[1])
        print(f"  {row:14s} {len(g)} group(s): " + "  ".join("".join(v) for v in sorted(g.values(), key=len, reverse=True)))
    for stream in ["rgb", "tip.rgb"]:
        pf = {n: runs[n]["per_frame"][stream] for n in names}
        nf = min(len(v) for v in pf.values())
        agree = [len({pf[n][i] for n in names}) == 1 for i in range(nf)]
        bad = [i for i, a in enumerate(agree) if not a]
        print(f"  per-frame {stream:8s}: {sum(agree)}/{nf} frames agree; first diff {bad[0] if bad else None}, last diff {bad[-1] if bad else None}")
    for n in names:
        L = runs[n].get("lod")
        if not L:
            continue
        w, s, f = L["warmup"], L["settle"], L["frames"]
        allrows = [L["first"]] + w + s + f
        settled = next((i for i in range(len(allrows)) if all(r[1] == 0 for r in allrows[i:])), None)
        ready_steps = [(i, r[0]) for i, r in enumerate(allrows) if i == 0 or r[0] != allrows[i - 1][0]]
        print(f"  {n:14s} wall={meta.get(n, {}).get('wall_s', '?'):>6} rgb={runs[n]['rows']['rgb']['fnv'][:6]} "
              f"ready(at)={ready_steps[-1]} queued0_from={settled} bytes@cap0={f[0][2] if f else None} "
              f"levels@cap0={f[0][3:] if f else None} levels@end={f[-1][3:] if f else None}")
        print(f"                 ready steps (index, chains): {ready_steps[:12]}{' ...' if len(ready_steps) > 12 else ''}")
    print()
