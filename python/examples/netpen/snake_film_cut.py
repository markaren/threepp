"""Pass 2 of the snake net-pen film: the edit.

Pass 1 (`snake_netpen.py --snake-film`) rendered the mission every frame from DOCKED to DOCKED and
wrote the 3D frames, the head-camera view + sonar image stream, the run's telemetry and a per-frame
film log (mission time, shot, cuts). This script cuts them into the film, from the run's OWN
telemetry: every time-lapse segment, inset emphasis and label hangs off the run's phase events
and logged distances, never off a hard-coded time.

    python snake_film_cut.py                                  # D:/dev/snake_out/film/raw_seed0*
    python snake_film_cut.py --raw D:/dev/snake_out/film/raw_seed0 --workers 4
    python snake_film_cut.py --plan                           # print the schedule and exit

Writes D:/dev/snake_out/film/snake_film.mp4 (1920x1080, 60 fps, libx264 crf 18, yuv420p),
snake_film_contact.png (16 frames) and snake_film_poster.png.

The edit: a title card; the mission with speed-ramped time-lapse on the long transits (labelled by
the panels' TIME-LAPSE badge; real time for the undock, the tear, the first pass, the U-turn and
the funnel capture and latch); a hard cut at every camera cut logged by pass 1 (DISSOLVE > 0 holds
the outgoing frame under a dissolve instead, so story time never repeats); dissolves only from the
title and into the end card; the head-camera view and the sonar
composited here (the head camera grows while the head passes the tear, the sonar grows when the
tear fires); snake_panels' HUD on top; the end card with the three-seed mission numbers.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.dirname(_HERE)
_PY = os.path.dirname(_EX)                                 # demo_common imports threepp from here
for _p in (_HERE, _EX, _PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from PIL import Image

from demo_common import Encoder, find_ffmpeg
import snake_panels as SP

FPS = 60
OUT_W, OUT_H = 1920, 1080
DISSOLVE = 0.0                                             # s at a logged cut: 0 = a hard cut (a dissolve between two
                                                           # angles on the same moving snake reads as a double exposure)
TITLE_S, END_S = 3.4, 9.0
MARGIN = 24
INS_SMALL, INS_BIG = (544, 306), (736, 414)                # head camera: normal, passing the tear
SON_SCALE_SMALL, SON_SCALE_BIG = 0.92, 1.18                # sonar panel scale: normal, when the tear fires
SON_EMPH_S = 3.2                                           # s the sonar is emphasised after the fire; the head camera after it


# ---- inputs ---------------------------------------------------------------------------------------
class Reader:
    """Sequential ffmpeg decode of an mp4 to RGB frames; get(i) moves forward only (keeps `keep`)."""

    def __init__(self, path, w, h, keep=()):
        self.w, self.h, self.n = w, h, w * h * 3
        self.p = subprocess.Popen([find_ffmpeg(), "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                  stdout=subprocess.PIPE, bufsize=self.n * 4)
        self.i, self.cur, self.keep, self.kept = -1, None, set(keep), {}

    def get(self, i):
        if i in self.kept:
            return self.kept[i]
        if i < self.i:
            raise RuntimeError(f"Reader: frame {i} is behind {self.i}")
        while self.i < i:
            buf = self.p.stdout.read(self.n)
            if len(buf) < self.n:
                return self.cur                            # past the end: hold the last frame
            self.i += 1
            self.cur = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            if self.i in self.keep:
                self.kept[self.i] = self.cur
        return self.cur

    def close(self):
        self.p.stdout.close()
        self.p.kill()


def events_of(tel_npz):
    d = np.load(tel_npz, allow_pickle=False)
    ev = {}
    for s in d["events"]:
        t, name = str(s).split(" ", 1)
        ev.setdefault(name, []).append(float(t))
    return ev


# ---- the schedule ----------------------------------------------------------------------------------
def plan_segments(tel, ev):
    """[(t0, t1, 1)] mission-time windows played in real time, cut hard from one to the next (a jump
    in the clock, never a fast-forward: a sped-up swimmer reads as a wiggle), from the run's own
    events and closest approaches. Windows closer than a second merge."""
    t = tel.t
    first = lambda k, dflt=None: ev[k][0] if k in ev and ev[k] else dflt   # noqa: E731
    t_und, t_piv, t_dep = first("UNDOCK"), first("PIVOT"), first("DEPART")
    t_tear = first("TEAR", first("INSPECT"))
    t_insp, t_ut, t_i2 = first("INSPECT"), first("UTURN"), first("INSPECT_2")
    t_ret = first("RETURN", first("APPROACH"))
    t_fin, t_cap = first("FINAL"), first("CAPTURE")
    t_dock = ev["DOCKED"][-1] if len(ev.get("DOCKED", [])) > 1 else float(t[-1]) - 2.0
    dt_ = tel.col("d_head_tear")
    ph = tel.phase_i

    def closest(name, t0, t1):
        m = (t >= t0) & (t <= t1)
        if name in tel.phases:
            m &= ph == tel.phases.index(name)
        if not m.any():
            return 0.5 * (t0 + t1)
        k = np.nonzero(m)[0]
        return float(t[k[np.argmin(dt_[k])]])
    c1 = closest("INSPECT", t_insp, t_ut)
    c2 = closest("INSPECT_2", t_i2, t_ret)
    wins = [
        (max(t_und - 2.5, 0.3), t_und + 5.0),            # latched, the lamps come on, it starts backing out
        # one stretch, no jump: the tail clears the funnel, the pivot, away under the sun, along the net,
        # the tear fires, the first pass with the neck on the hole
        (t_piv - 2.5, c1 + 3.0),
        (t_ut + 1.0, t_ut + 8.0),                        # the U-turn
        (c2 - 3.0, c2 + 3.0),                            # the second pass
        (t_cap - 6.0, t_cap + 4.0),                      # into the funnel, the capture
        (t_dock - 2.5, min(t_dock + 2.8, float(t[-1]))),  # latched
    ]
    wins = sorted((a, b) for a, b in wins if b > a)
    out = [list(wins[0])]
    for a, b in wins[1:]:
        if a <= out[-1][1] + 1.0:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    segs = [(a, b, 1.0) for a, b in out]
    info = dict(undock=t_und, pivot=t_piv, depart=t_dep, tear=t_tear, closest1=c1, uturn=t_ut, inspect2=t_i2,
                closest2=c2, ret=t_ret, final=t_fin, capture=t_cap, docked=t_dock)
    return segs, info


def build_schedule(tel, flog, segs, info):
    """Per output frame of the mission section: source index, mission time, the jump badge (seconds
    skipped at the last jump, for 1.6 s after it), dissolve source + weight (none: DISSOLVE is 0),
    head-camera and sonar emphasis (eased, so every worker sees the same values)."""
    ft = flog[:, 0]
    rows = []
    jump, t_jump = 0.0, -1e9
    for w, (a, b, _) in enumerate(segs):
        if w > 0:
            jump, t_jump = a - segs[w - 1][1], a
        x = a
        while x < b:
            src = int(np.clip(np.searchsorted(ft, x - 0.5 / FPS), 0, len(ft) - 1))
            rows.append([src, float(ft[src]), 1.0, jump if x - t_jump < 1.6 else 0.0])
            x += 1.0 / FPS
    n = len(rows)
    src = np.array([r[0] for r in rows])
    diss_src, diss_w = np.full(n, -1), np.zeros(n)
    # emphasis: the head camera while the head passes the tear; the sonar when the tear fires
    tt = tel.t
    dht = tel.col("d_head_tear")
    ph_names = [tel.phases[i] for i in tel.phase_i]
    t_fire = tel.t_fire if tel.t_fire is not None else info["tear"]
    e_ins, e_son = np.zeros(n), np.zeros(n)
    a, b = 0.0, 0.0
    for k in range(n):
        tm = rows[k][1]
        if k and tm - rows[k - 1][1] > 0.5:               # a jump: the emphasis restarts from rest
            a = b = 0.0
        i = int(np.clip(np.searchsorted(tt, tm), 0, len(tt) - 1))
        after = t_fire is None or tm - t_fire > SON_EMPH_S + 0.4          # one inset grows at a time
        want_i = 1.0 if (ph_names[i] in ("TEAR", "INSPECT", "INSPECT_2") and dht[i] < 2.3 and after) else 0.0
        want_s = 1.0 if (t_fire is not None and 0.0 <= tm - t_fire <= SON_EMPH_S) else 0.0
        a += float(np.clip(want_i - a, -1.0 / (0.8 * FPS), 1.0 / (0.8 * FPS)))
        b += float(np.clip(want_s - b, -1.0 / (0.6 * FPS), 1.0 / (0.6 * FPS)))
        e_ins[k], e_son[k] = a, b
    return dict(src=src, t=np.array([r[1] for r in rows]), speed=np.array([r[2] for r in rows]),
                badge=np.array([r[3] for r in rows]), diss_src=diss_src, diss_w=diss_w, e_ins=e_ins, e_son=e_son)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _plate(w, h, r=8, fill=(5, 15, 19, 172), edge=(255, 255, 255, 40)):
    L = SP.Layer(w, h)
    L.plate(r, fill=fill, edge=edge)
    return L


def inset_rect(e):
    k = ease(e)
    w = int(round(INS_SMALL[0] + (INS_BIG[0] - INS_SMALL[0]) * k))
    h = int(round(INS_SMALL[1] + (INS_BIG[1] - INS_SMALL[1]) * k))
    return MARGIN, OUT_H - MARGIN - h, w, h


def sonar_rect(e, sw, sh):
    k = ease(e)
    s = SON_SCALE_SMALL + (SON_SCALE_BIG - SON_SCALE_SMALL) * k
    w, h = int(round(sw * s)), int(round(sh * s))
    return OUT_W - MARGIN - w, OUT_H - MARGIN - h, w, h


def label(img, x, y, text, sub=None, accent=False):
    px = 15
    w = SP.text_width(text, px, "sb") + 20 + (SP.text_width("   " + sub, 13) if sub else 0)
    L = _plate(w, 26, r=5, edge=(242, 194, 48, 200) if accent else (255, 255, 255, 40))
    L.text(10, 13, text, px, "sb", SP.ACCENT if accent else SP.WHITE, anchor="lm")
    if sub:
        L.text(10 + SP.text_width(text + "   ", px, "sb"), 13, sub, 13, "r", SP.GREY, anchor="lm")
    im = L.done()
    img.alpha_composite(im, (int(x), int(max(y, 0))))


def frame3d(raw, aux, t, e_i, e_s, vw, vh, sw, sh, d_tear, phase):
    """The rendered frame with the head camera (bottom left) and the sonar (bottom right) on it."""
    img = Image.fromarray(raw, "RGB").convert("RGBA")
    x, y, w, h = inset_rect(e_i)
    ins = Image.fromarray(np.ascontiguousarray(aux[:vh, :vw]), "RGB").resize((w, h), Image.LANCZOS)
    img.paste(ins, (x, y))
    edge = SP.Layer(w + 4, h + 4)
    edge.rect(1, 1, w + 3, h + 3, outline=(242, 194, 48, int(110 + 120 * ease(e_i))) if e_i > 0.02 else (255, 255, 255, 70),
              width=1.6, r=3)
    img.alpha_composite(edge.done(), (x - 2, y - 2))
    sub = f"nose to tear centre {d_tear:.2f} m" if (e_i > 0.05 and d_tear is not None) else None
    label(img, x, y - 32, "HEAD CAMERA", sub, accent=e_i > 0.5)
    # sonar: premultiplied-ish: the dark fan background at 80 %, the echoes opaque
    son = np.ascontiguousarray(aux[:sh, vw:vw + sw])
    sx, sy, sw2, sh2 = sonar_rect(e_s, sw, sh)
    s_im = Image.fromarray(son, "RGB").resize((sw2, sh2), Image.LANCZOS)
    lum = np.asarray(s_im, np.float32).max(-1)
    alpha = (np.clip(0.80 + 0.2 * lum / 40.0, 0.0, 1.0) * 255).astype(np.uint8)
    s_rgba = Image.fromarray(np.dstack([np.asarray(s_im), alpha]), "RGBA")
    img.alpha_composite(s_rgba, (sx, sy))
    label(img, sx + 8, sy + 8, "HEAD SONAR", "130 deg fan, 10 m shown", accent=e_s > 0.5)
    return img, (sx, sy, sw2, sh2)


def gap_xy(tel, i, rect, sw, sh, view_m=10.0):
    """Frame pixel of the tear in the sonar panel at log row i (the true tear, seen from the head's
    logged pose: bearing positive to starboard = right in the fan), or None outside the fan."""
    c = tel.c
    if not all(k in c for k in ("head_x", "head_z", "psi_head", "tear_x", "tear_z")):
        return None
    L = tel.L[i]
    psi = L[c["psi_head"]]
    d = np.array([L[c["tear_x"]] - L[c["head_x"]], L[c["tear_z"]] - L[c["head_z"]]])
    b = math.atan2(d @ [math.sin(psi), math.cos(psi)], d @ [math.cos(psi), -math.sin(psi)])
    r = float(np.hypot(*d))
    if abs(b) > math.radians(62.0) or r > 0.95 * view_m:
        return None
    rpx = r / view_m * (sh - 24)
    x, y, w, h = rect
    k = w / sw
    return x + (0.5 * sw + rpx * math.sin(b)) * k, y + ((sh - 12) - rpx * math.cos(b)) * k


# ---- title and end card ---------------------------------------------------------------------------
def title_card():
    bg = np.array((9, 22, 26), float)
    g = 0.35 * np.linspace(0, 1, OUT_H)[:, None, None]
    base = bg * (1.0 - g) + np.array((4, 10, 12), float) * g
    img = Image.fromarray(np.repeat(base, OUT_W, axis=1).clip(0, 255).astype(np.uint8), "RGB").convert("RGBA")
    L = SP.Layer(OUT_W, OUT_H)
    x = 160
    L.rect(x, 380, x + 6, 470, fill=SP._rgba(SP.ACCENT))
    L.text(x + 30, 372, "A snake robot that lives in a fish farm", 58, "sb", SP.WHITE)
    L.text(x + 30, 446, "Resident net-pen inspection, simulated", 26, "r", SP.ACCENT)
    body = ("An underwater snake robot (9 modules, 1.62 m, side and tunnel thrusters like NTNU's Eelume) lives in "
            "a dock inside a salmon net pen. It backs out, follows the net on its own sonar, finds a tear and turns "
            "its head to look into it, then docks into the current. One closed-loop run of the physics; the camera "
            "follows what it does.")
    y = 540
    for ln in SP.Panels._lines(body, 24, "r", 1500):
        L.text(x + 30, y, ln, 24, "r", SP.GREY)
        y += 36
    L.text(x + 30, y + 30, "threepp: PhysX at 240 Hz with the Kelasidi et al. (2015) fluid model and thrusters, Warp net "
                           "and salmon, Vulkan renderer", 18, "r", SP.DIM)
    img.alpha_composite(L.done())
    return np.asarray(img.convert("RGB"))


def blend(a, b, w):
    return (a.astype(np.float32) * (1.0 - w) + b.astype(np.float32) * w).clip(0, 255).astype(np.uint8)


# ---- a worker: one contiguous chunk of the mission section ------------------------------------------
def work(args):
    (k, lo, hi, raw, sched, tel_npz, summ_json, sweep, chunk_path, picks, poster_k) = args
    S = {kk: v for kk, v in np.load(sched, allow_pickle=False).items()}
    fl = np.load(raw + "_film_log.npz", allow_pickle=False)
    vw, vh = [int(v) for v in fl["view_wh"]]
    sw, sh = [int(v) for v in fl["sonar_wh"]]
    P = SP.Panels(tel_npz, summ_json, sweep)
    tel = P.tel
    dht = tel.col("d_head_tear")
    held_src = set(int(v) for v in S["diss_src"][lo:hi] if v >= 0)     # decoded on the way, kept for the dissolve
    rr = Reader(raw + ".mp4", OUT_W, OUT_H, keep=held_src)
    ra = Reader(raw + "_aux.mp4", vw + sw, max(vh, sh), keep=held_src)
    enc = Encoder(chunk_path, OUT_W, OUT_H, FPS, crf=18, preset="medium")
    held = {}
    shots = {}
    t0 = time.perf_counter()
    for j in range(lo, hi):
        src = int(S["src"][j])
        tm = float(S["t"][j])
        i = tel.idx(tm)
        ph = tel.phases[tel.phase_i[i]]
        d_tear = float(dht[i]) if np.isfinite(dht[i]) else None
        img, srect = frame3d(rr.get(src), ra.get(src), tm, S["e_ins"][j], S["e_son"][j], vw, vh, sw, sh, d_tear, ph)
        ds = int(S["diss_src"][j])
        if ds >= 0:
            if ds not in held:
                i2 = tel.idx(float(fl["log"][ds, 0]))
                d2 = float(dht[i2]) if np.isfinite(dht[i2]) else None
                held[ds] = frame3d(rr.get(ds), ra.get(ds), tm, S["e_ins"][j], S["e_son"][j], vw, vh, sw, sh, d2, ph)[0]
            w = float(S["diss_w"][j])
            img = Image.fromarray(blend(np.asarray(img), np.asarray(held[ds]), w), "RGBA")
        base = np.asarray(img.convert("RGB"))
        out = P.compose(base, tm, jump=float(S["badge"][j]), sonar_rect=srect,
                        sonar_gap_xy=gap_xy(tel, i, srect, sw, sh))
        enc.send(out)
        if j in picks:
            shots[j] = out
        if j == poster_k:                                  # the poster: this frame with both insets at rest size
            clean = np.asarray(frame3d(rr.get(src), ra.get(src), tm, 0.0, 0.0, vw, vh, sw, sh, d_tear, ph)[0].convert("RGB"))
            Image.fromarray(P.compose(clean, tm, speedup=1.0)).save(chunk_path.replace(".mp4", "_poster.png"))
        if (j - lo) % 600 == 599:
            print(f"  worker {k}: {j - lo + 1}/{hi - lo} frames, {(time.perf_counter() - t0) / (j - lo + 1) * 1e3:.0f} ms/f",
                  flush=True)
    enc.close()
    rr.close()
    ra.close()
    for j, im in shots.items():
        Image.fromarray(im).save(chunk_path.replace(".mp4", f"_pick{j:06d}.png"))
    return k, hi - lo, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="D:/dev/snake_out/film/raw_seed0")
    ap.add_argument("--seeds", default="D:/dev/snake_out/mission/seed0.json,D:/dev/snake_out/mission/seed1.json,"
                                       "D:/dev/snake_out/mission/seed2.json")
    ap.add_argument("--seeds-note", default="", help="the end card's seed-block note (e.g. which tree they ran on)")
    ap.add_argument("--sweep", default="D:/dev/snake_out/sweep")
    ap.add_argument("--out", default="D:/dev/snake_out/film/snake_film.mp4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--limit", type=float, default=0.0, help="only the first N s of the mission section (a test)")
    a = ap.parse_args()
    raw = a.raw
    tel_npz, summ_json = raw + "_telemetry.npz", raw + ".json"
    P = SP.Panels(tel_npz, summ_json, a.sweep)
    tel = P.tel
    ev = events_of(tel_npz)
    fl = np.load(raw + "_film_log.npz", allow_pickle=False)
    segs, info = plan_segments(tel, ev)
    S = build_schedule(tel, fl["log"], segs, info)
    n = len(S["src"])
    if a.limit > 0:
        n = min(n, int(a.limit * FPS))
        S = {k: v[:n] for k, v in S.items()}
    total = TITLE_S + n / FPS + END_S
    print("windows (mission t0 -> t1, real time, cut hard between):")
    for s0, s1, sp in segs:
        print(f"  {s0:6.1f} -> {s1:6.1f}  {(s1 - s0) / sp:5.1f} s")
    print(f"events used: " + ", ".join(f"{k} {v:.1f}" for k, v in info.items() if v is not None))
    print(f"mission section {n / FPS:.1f} s, film {total:.1f} s at {FPS} fps; cuts dissolved: "
          f"{int((S['diss_w'] > 0).sum() // max(int(DISSOLVE * FPS), 1))}")
    if a.plan:
        return
    out_dir = os.path.dirname(os.path.abspath(a.out))
    stem = os.path.splitext(os.path.abspath(a.out))[0]
    work_dir = os.path.join(out_dir, "_cut_" + os.path.basename(stem))
    os.makedirs(work_dir, exist_ok=True)
    for p in glob.glob(os.path.join(work_dir, "*")):
        os.remove(p)
    sched = os.path.join(work_dir, "schedule.npz")
    np.savez(sched, **S)
    # the end card: three seeds when their summaries exist
    seeds = [s for s in a.seeds.split(",") if s and os.path.exists(s)]
    summs = [SP.load_summary(s) for s in seeds]
    print(f"end card: {len(summs)} seed summaries ({', '.join(seeds)})")
    card = P.end_card((OUT_W, OUT_H), seeds=summs or None, seeds_note=a.seeds_note)
    Image.fromarray(card).save(stem + "_end_card.png")
    title = title_card()
    Image.fromarray(title).save(stem + "_title.png")
    ntot = int(round(TITLE_S * FPS)) + n + int(round(END_S * FPS))
    t_off = int(round(TITLE_S * FPS))
    picks_glob = set(np.linspace(0, ntot - 1, 16).round().astype(int).tolist())
    picks = {j - t_off for j in picks_glob if t_off <= j < t_off + n}
    kc = int(np.argmin(np.abs(S["t"] - info["closest1"])))
    poster_k = kc
    # workers over the mission section
    nw = max(1, min(a.workers, n // 300 or 1))
    bounds = np.linspace(0, n, nw + 1).round().astype(int)
    jobs = [(k, int(bounds[k]), int(bounds[k + 1]), raw, sched, tel_npz, summ_json, a.sweep,
             os.path.join(work_dir, f"chunk{k:02d}.mp4"), picks, poster_k) for k in range(nw)]
    t0 = time.perf_counter()
    if nw == 1:
        res = [work(jobs[0])]
    else:
        from multiprocessing import Pool
        with Pool(nw) as pool:
            res = pool.map(work, jobs)
    for k, m, s in res:
        print(f"  chunk {k}: {m} frames in {s:.0f} s")
    # title + end: the first mission frame and the last (with panels) come back from the picks
    first = os.path.join(work_dir, "first.png")
    last = os.path.join(work_dir, "last.png")
    # re-render the two joins cheaply from the chunks' first/last frames via ffmpeg
    ff = find_ffmpeg()
    subprocess.run([ff, "-v", "error", "-y", "-i", jobs[0][8], "-frames:v", "1", first], check=True)
    subprocess.run([ff, "-v", "error", "-y", "-sseof", "-0.05", "-i", jobs[-1][8], "-update", "1", "-frames:v", "1", last],
                   check=True)
    f_im = np.asarray(Image.open(first).convert("RGB"))
    l_im = np.asarray(Image.open(last).convert("RGB"))
    black = np.zeros_like(title)
    head = os.path.join(work_dir, "head.mp4")
    enc = Encoder(head, OUT_W, OUT_H, FPS, crf=18, preset="medium")
    nt = int(round(TITLE_S * FPS))
    for j in range(nt):
        x = j / FPS
        if x < 0.5:
            fr = blend(black, title, ease(x / 0.5))
        elif x > TITLE_S - 0.7:
            fr = blend(title, f_im, ease((x - (TITLE_S - 0.7)) / 0.7))
        else:
            fr = title
        enc.send(fr)
    enc.close()
    tail = os.path.join(work_dir, "tail.mp4")
    enc = Encoder(tail, OUT_W, OUT_H, FPS, crf=18, preset="medium")
    ne = int(round(END_S * FPS))
    for j in range(ne):
        x = j / FPS
        if x < 0.9:
            fr = blend(l_im, card, ease(x / 0.9))
        elif x > END_S - 0.6:
            fr = blend(card, black, ease((x - (END_S - 0.6)) / 0.6))
        else:
            fr = card
        enc.send(fr)
    enc.close()
    lst = os.path.join(work_dir, "list.txt")
    with open(lst, "w") as fh:
        for p in [head] + [j[8] for j in jobs] + [tail]:
            fh.write(f"file '{p}'\n")
    subprocess.run([ff, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", a.out], check=True)
    # contact sheet + poster
    stem = os.path.splitext(a.out)[0]
    tiles = []
    for jg in sorted(picks_glob):
        if jg < t_off:
            im = title
        elif jg >= t_off + n:
            im = card
        else:
            k = [jj for jj in jobs if jj[1] <= jg - t_off < jj[2]][0]
            im = np.asarray(Image.open(k[8].replace(".mp4", f"_pick{jg - t_off:06d}.png")))
        tiles.append(Image.fromarray(im).resize((480, 270), Image.LANCZOS))
    cs = Image.new("RGB", (480 * 4, 270 * 4), (0, 0, 0))
    for i, im in enumerate(tiles[:16]):
        cs.paste(im, (480 * (i % 4), 270 * (i // 4)))
    cs.save(stem + "_contact.png")
    k = [jj for jj in jobs if jj[1] <= poster_k < jj[2]][0]
    Image.open(k[8].replace(".mp4", "_poster.png")).save(stem + "_poster.png")
    dur = (nt + n + ne) / FPS
    print(f"film -> {a.out} ({dur:.1f} s, {nt + n + ne} frames at {FPS} fps) in {(time.perf_counter() - t0) / 60:.1f} min; "
          f"{stem}_contact.png, {stem}_poster.png")


if __name__ == "__main__":
    main()
