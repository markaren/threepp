"""Regression gate for the lesson films: does a refactor of the lesson toolkit change a film?

    python lesson_gate.py --save lesson_out/_api/base1 [--only intro,rocket]
    python lesson_gate.py --compare base1 base2                    # the null: two runs of one tree
    python lesson_gate.py --compare base1 after --null base1 base2  # exit 0 = within the null

--save runs every lesson in ../lesson with `--gate <dir>/<lesson>` (one setup(), narration on),
which writes the captions on the film clock (.srt), every caption's spoken text and every hold
(.voice.json), a contact sheet and a full-size still at the middle of every caption.

--compare needs the .srt and .voice.json to match exactly. Without --null a frame must be
identical. With --null, the two null runs mark where a frame is live and changes from run to run
(16-pixel cells that differ by more than 4 levels, grown by a cell: warp_threepp's streamed hero
shot). Outside that mask, a pixel may differ by no more than the null runs did there (1 level
for the GL lessons, up to 4 for warp), except in specks of at most 3x3 pixels (200 pixels in
all: the streamed shot's noise lands anew each run), and at most 5000 pixels (or twice the
null's count) may differ.

Each lesson runs from its film folder (under lesson_out/ in the current directory, the lessons'
own default --outdir), so it finds the voice cache (and warp_threepp its clips) the film was
made with. A voice line missing from the cache is synthesized and reported; a missing Warp clip
fails the run instead of being rendered again.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
LESSON_DIR = os.path.join(os.path.dirname(HERE), "lesson")
FILMS = os.path.abspath("lesson_out")

# name: (script, cwd, --outdir relative to cwd)
LESSONS = {
    "intro": ("threepp_intro.py", f"{FILMS}/intro", "."),
    "first_app": ("first_app.py", f"{FILMS}/first_app", "."),
    "imu": ("imu_tilt.py", f"{FILMS}/imu", "."),
    "rocket": ("rocket_pid.py", f"{FILMS}/rocket", "."),
    "ik_fr3": ("ik_fr3.py", f"{FILMS}/_api/voice/ik_fr3", "."),
    "depth_map": ("depth_map.py", f"{FILMS}/_api/voice/depth_map", "."),
    "warp": ("warp_threepp.py", f"{FILMS}/warp", "lesson_out"),
}
WARP_CLIPS = ("warp_fluid.mp4", "blast.mp4", "gummy.mp4", "jelly.mp4")


def _hashes(paths):
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            out[p] = hashlib.sha1(f.read()).hexdigest()
    return out


def save(dest, names):
    os.makedirs(dest, exist_ok=True)
    maths = glob.glob(os.path.join(LESSON_DIR, "*.math.json"))
    maths_before = _hashes(maths)
    record = os.path.join(dest, "save.json")
    summary = {}
    if os.path.isfile(record):                  # a second --only run into the same folder adds to it
        with open(record, encoding="utf-8") as fh:
            summary = json.load(fh)
    for name in names:
        script, cwd, outdir = LESSONS[name]
        os.makedirs(cwd, exist_ok=True)
        cache = os.path.join(cwd, outdir, "voice_cache")
        if name == "warp":
            missing = [c for c in WARP_CLIPS if not os.path.isfile(os.path.join(cwd, outdir, "warp_clips", c))]
            if missing:
                print(f"[{name}] FAIL: clips missing from {cwd}/{outdir}/warp_clips: {missing}")
                summary[name] = {"rc": None, "error": f"missing clips {missing}"}
                continue
        before = set(os.listdir(cache)) if os.path.isdir(cache) else set()
        gate = os.path.abspath(os.path.join(dest, name))
        log = os.path.join(dest, f"{name}.log")
        cmd = [sys.executable, os.path.join(LESSON_DIR, script), "--gate", gate, "--outdir", outdir]
        print(f"[{name}] {' '.join(cmd)}  (cwd {cwd})", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode
        after = set(os.listdir(cache)) if os.path.isdir(cache) else set()
        new = sorted(after - before)
        summary[name] = {"rc": rc, "seconds": round(time.time() - t0, 1), "new_voice_files": new}
        print(f"[{name}] exit {rc} in {time.time() - t0:.0f}s; {len(new)} new voice-cache files; log {log}",
              flush=True)
    changed = [os.path.basename(p) for p, h in _hashes(maths).items() if maths_before.get(p) != h]
    summary["_math_json_changed"] = sorted(set(summary.get("_math_json_changed", [])) | set(changed))
    if changed:
        print("[gate] WARNING: these tracked formula caches changed:", changed)
    with open(record, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return all(summary[n].get("rc") == 0 for n in names)


def _images(d):
    """{relative path: file} for a lesson's gate folder: the sheet and the stills."""
    out = {}
    for p in glob.glob(os.path.join(d, "*_sheet.png")) + glob.glob(os.path.join(d, "stills", "*.png")):
        out[os.path.relpath(p, d).replace("\\", "/")] = p
    return out


TILE = 16          # the null mask's cell, in pixels
MASK_LEVEL = 4     # a cell is live when two null runs differ there by more than this
NOISE_PIXELS = 5000
SPECK = 9          # outside the mask, a streamed shot's noise lands anew each run: specks up to 3x3 px pass
SPECK_PIXELS = 200 # ... up to this many pixels of them per frame


def _load(p):
    return np.asarray(Image.open(p).convert("RGB"), np.int16)


def _absdiff(pa, pb):
    a, b = _load(pa), _load(pb)
    return None if a.shape != b.shape else np.abs(a - b).max(axis=2)


def _stats(d):
    ys, xs = np.nonzero(d)
    r = {"max": int(d.max()), "pixels": int(len(ys))}
    if len(ys):
        r["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        r["p99"] = int(np.percentile(d[d > 0], 99))
    return r


def _null_mask(d):
    """Where two runs of one tree differ by more than MASK_LEVEL: TILE-pixel cells, grown by one
    cell. Warp's live hero stream (event camera, sparks) lands here. At a lower level, the hero's
    faint whole-frame shimmer (2-3 levels) would mask the overlays too."""
    h, w = d.shape
    th, tw = -(-h // TILE), -(-w // TILE)
    pad = np.zeros((th * TILE, tw * TILE), bool)
    pad[:h, :w] = d > MASK_LEVEL
    cells = pad.reshape(th, TILE, tw, TILE).any(axis=(1, 3))
    grown = cells.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown |= np.roll(np.roll(cells, dy, 0), dx, 1)
    return np.repeat(np.repeat(grown, TILE, 0), TILE, 1)[:h, :w]


def _specks(over):
    """(pixels, largest 8-connected blob) of a boolean image."""
    n = int(over.sum())
    if not n:
        return 0, 0
    from scipy import ndimage
    lab, _ = ndimage.label(over, structure=np.ones((3, 3)))
    return n, int(np.bincount(lab.ravel())[1:].max())


def _read_text(d, suffix):
    f = glob.glob(os.path.join(d, "*" + suffix))
    if not f:
        return None
    with open(f[0], encoding="utf-8") as fh:
        t = fh.read()
    return json.loads(t) if suffix.endswith(".json") else t


def compare_lesson(da, db, null=None):
    """Text must match exactly. With a null, a frame passes when, outside the null mask, the
    pixels that differ by more than the null did there (at least 1) are only specks (blobs of
    at most SPECK pixels, SPECK_PIXELS in all), and no more pixels differ at all than
    NOISE_PIXELS or twice the null's count. Without a null, a frame must be identical."""
    rep = {"text": {}, "frames": {}}
    for suffix in (".srt", ".voice.json"):
        ta, tb = _read_text(da, suffix), _read_text(db, suffix)
        rep["text"][suffix] = "missing" if ta is None or tb is None else ("same" if ta == tb else "DIFFERENT")
    ia, ib = _images(da), _images(db)
    inull = _images(null[0]) if null else {}
    for k in sorted(set(ia) | set(ib)):
        if k not in ia or k not in ib:
            rep["frames"][k] = {"missing": "A" if k not in ia else "B", "pass": False}
            continue
        d = _absdiff(ia[k], ib[k])
        if d is None:
            rep["frames"][k] = {"shape_differs": True, "pass": False}
            continue
        r = _stats(d)
        if null:
            nd = _absdiff(inull[k], os.path.join(null[1], k)) if k in inull else None
            if nd is None or nd.shape != d.shape:
                nd = np.zeros(d.shape, np.int16)
            mask = _null_mask(nd)
            nout, out = nd[~mask], d[~mask]
            level = max(1, int(nout.max()) if nout.size else 0)
            allowed = max(NOISE_PIXELS, 2 * int((nout > 0).sum()))
            specks, blob = _specks((d > level) & ~mask)
            r["masked_fraction"] = round(float(mask.mean()), 4)
            r["outside_max"], r["outside_max_allowed"] = (int(out.max()) if out.size else 0), level
            r["outside_pixels"], r["outside_pixels_allowed"] = int((out > 0).sum()), allowed
            r["specks"], r["largest_speck"] = specks, blob
            r["pass"] = blob <= SPECK and specks <= SPECK_PIXELS and r["outside_pixels"] <= allowed
        else:
            r["pass"] = r["pixels"] == 0
        rep["frames"][k] = r
    return rep


def compare(a, b, null=None):
    names = sorted(n for n in LESSONS if os.path.isdir(os.path.join(a, n)) and os.path.isdir(os.path.join(b, n)))
    report, ok = {}, True
    for n in names:
        nl = (os.path.join(null[0], n), os.path.join(null[1], n)) if null else None
        rep = compare_lesson(os.path.join(a, n), os.path.join(b, n), nl)
        text_ok = all(v == "same" for v in rep["text"].values())
        bad = [k for k, r in rep["frames"].items() if not r["pass"]]
        rep["frames_failing"] = bad
        rep["pass"] = text_ok and not bad
        ok &= rep["pass"]
        report[n] = rep
        diffs = [r for r in rep["frames"].values() if r.get("pixels")]
        worst = max(diffs, key=lambda r: r["pixels"], default=None)
        masked = max((r.get("masked_fraction", 0) for r in rep["frames"].values()), default=0)
        print(f"{n:10s} srt {rep['text']['.srt']}, voice {rep['text']['.voice.json']}; "
              f"{len(rep['frames'])} frames, {len(diffs)} differ"
              + (f" (worst: max {worst['max']}, {worst['pixels']} px, bbox {worst.get('bbox')})" if worst else "")
              + (f"; null mask up to {100 * masked:.1f}% of a frame" if null else "")
              + (f"; {len(bad)} fail" if bad else "")
              + ("  PASS" if rep["pass"] else "  FAIL"))
    out = os.path.join(b, f"compare_{os.path.basename(os.path.normpath(a))}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("report:", out)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", metavar="DIR")
    ap.add_argument("--only", default=None, help="comma-separated lessons: " + ",".join(LESSONS))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--null", nargs=2, metavar=("N1", "N2"), help="two runs of one tree: the allowed difference")
    args = ap.parse_args()
    if args.save:
        names = args.only.split(",") if args.only else list(LESSONS)
        unknown = [n for n in names if n not in LESSONS]
        if unknown:
            ap.error(f"unknown lessons {unknown}")
        sys.exit(0 if save(args.save, names) else 1)
    if args.compare:
        sys.exit(0 if compare(*args.compare, null=args.null) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
