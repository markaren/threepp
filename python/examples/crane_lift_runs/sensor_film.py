"""The crane lift's sensor film: the hero frames of a `--op-film` run with the tip range fan,
the event camera and the hull IMU / load cell / swing traces drawn over them.

CPU only, after the run, from the run's own files: the film mp4 (the rendered frame with the
tip camera inset the renderer composites), the `--op-panels` npz (the events binned 4x4 per film
frame, every 32x32 fan scan), the op npz log (the IMU samples, the load cell, the contact
switch, the fan's swing estimate) and the manifest (the phase boundaries, the loop's state).
A layout change is a re-run of this script, not of the GPU.

    python sensor_film.py <run dir> --tag s0c --op op_s0_film3            # film_s0c_sensors.mp4, the whole run
    python sensor_film.py <run dir> --tag s0c --op op_s0_film3 --t0 36 --t1 60 --out seg1.mp4
    python sensor_film.py <run dir> --tag s0c --op op_s0_film3 --still 45 still_45.png

Output is 1920x1080 (the 1280x720 hero upscaled, so make_video.py's normalisation is a no-op)
at the film's own rate, H.264. Every number on screen is read from the run's files.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_W, OUT_H = 1920, 1080
G = 9.81                                  # PhysX gravity, subtracted from the accelerometer's y for the trace

# ---- panel geometry on the 1920x1080 canvas (chosen against the hero framing: the sky above the
# vessel at the left, the sky between the boom tip and the frame's top in the middle, the sky right
# of the turbine tower; the tip camera inset the renderer draws sits at the lower right) ----
TRACES = (24, 24, 600, 360)
FAN = (648, 24, 240, 262)                 # title row, 32 x 6 px cells, two note rows; ends above the boom at arrival
EVENTS = (1256, 24, 640, 360)

BG = (12, 16, 22, 196)
BORDER = (150, 160, 175, 230)
TEXT = (235, 238, 242, 255)
DIM = (150, 158, 170, 255)
AXIS = (70, 78, 90, 255)
C3 = ((255, 120, 80, 255), (120, 230, 120, 255), (110, 170, 255, 255))     # x, y, z
C_TENSION = (255, 196, 60, 255)
C_SWING = (90, 220, 255, 255)
C_CONTACT = (140, 40, 40, 120)
C_LOOP = (90, 220, 255, 140)
C_MARK = (255, 255, 255, 120)
EV_BG = (16, 18, 24)
EV_ON = (235, 235, 245)
EV_OFF = (255, 120, 60)


def font(size):
    try:
        import matplotlib
        p = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default(size=size)


F_TITLE, F_SMALL, F_TINY = font(17), font(14), font(12)


def panel(w, h):
    im = Image.new("RGBA", (w, h), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=BORDER, width=1)
    return im, d


def polyline(d, xs, ys, colour, width=2):
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if len(pts) >= 2:
        d.line(pts, fill=colour, width=width, joint="curve")


class Run:
    def __init__(self, run, tag, op, film, panels_path):
        self.dir = run
        self.film = film or os.path.join(run, f"film_{tag}.mp4")
        self.panels = np.load(panels_path or os.path.join(run, f"panels_{tag}.npz"))
        z = np.load(os.path.join(run, op + ".npz"))
        self.log, self.imu = z["log"], z["imu"]
        with open(os.path.join(run, op + ".json")) as fh:
            self.meta = json.load(fh)["meta"]
        p = self.panels
        self.t = p["t"]
        self.log_row = p["log_row"]
        self.film_frame = p["film_frame"]
        self.fan = p["fan"]
        self.fan_frame = p["fan_frame"]
        self.fan_n = int(p["fan_n"])
        self.load_id = int(p["load_id"])
        self.ev_pos, self.ev_neg = p["ev_pos"], p["ev_neg"]
        self.tip_rect = p["tip_rect"]
        self.hero_size = p["hero_size"]
        self.T1, self.T2, self.T3, self.T4 = [float(v) for v in p["boundaries"]]
        self.t_land = float(p["t_land"])
        self.fps = float(p["film_fps"])
        self.loop = bool(self.meta.get("antiswing", False))
        self.engage = self.meta.get("engage", "ondemand")
        assert len(self.t) == len(self.ev_pos) == len(self.log_row), "panels npz: inconsistent lengths"

    def engaged(self, row):
        if not self.loop or self.log[row, 23] > 0.5:
            return False
        return self.engage == "always" or self.log[row, 0] >= self.T2


# ---- the traces panel -------------------------------------------------------------------------
def draw_traces(run, i, win, gyro_range, acc_range, tension_range, swing_range):
    x0, y0, w, h = TRACES
    im, d = panel(w, h)
    row = int(run.log_row[i])
    t_now = float(run.log[row, 0])
    r0 = max(0, row - int(round(win * 60)))
    L = run.log[r0:row + 1]
    M = run.imu[r0:row + 1]
    t_lo = t_now - win
    left, right = 40, w - 10
    def tx(t):
        return left + (np.asarray(t, np.float64) - t_lo) / win * (right - left)
    d.text((10, 6), f"t = {t_now:6.2f} s    last {win:.0f} s", font=F_TITLE, fill=TEXT)
    strips = [
        ("hull IMU gyro, deg/s", [np.degrees(M[:, 1 + k]) for k in range(3)], C3, (-gyro_range, gyro_range), M[:, 0]),
        ("hull IMU accel, m/s² (y minus g)", [M[:, 4], M[:, 5] - G, M[:, 6]], C3, (-acc_range, acc_range), M[:, 0]),
        ("hoist load cell, kN", [L[:, 26] * 1e-3], (C_TENSION,), (0.0, tension_range), L[:, 0]),
        ("swing from the fan, mm", [np.hypot(L[:, 17], L[:, 18]) * 1e3], (C_SWING,), (0.0, swing_range), L[:, 0]),
    ]
    top, gap = 34, 6
    sh = (h - top - 8 - gap * (len(strips) - 1)) // len(strips)
    for s, (title, series, colours, (lo, hi), ts) in enumerate(strips):
        sy0 = top + s * (sh + gap)
        sy1 = sy0 + sh
        plot0, plot1 = sy0 + 16, sy1 - 2
        def ty(v):
            return plot1 - (np.clip(np.asarray(v, np.float64), lo, hi) - lo) / (hi - lo) * (plot1 - plot0)
        # the contact band and the loop state, drawn under the lines
        if s == 2:
            on = L[:, 22] > 0.5
            if on.any():
                xs = tx(L[on, 0])
                d.rectangle([float(xs.min()), plot0, float(xs.max()) + 1, plot1], fill=C_CONTACT)
        if s == 3:
            eng = np.array([run.engaged(r) for r in range(r0, row + 1)])
            if eng.any():
                xs = tx(L[eng, 0])
                d.rectangle([float(xs.min()), plot0, float(xs.max()) + 1, plot0 + 3], fill=C_LOOP)
        # the title row: the title, then the current value of each series in its own colour, right-aligned
        d.text((10, sy0 - 1), title, font=F_SMALL, fill=TEXT)
        names = ("x", "y", "z") if len(series) == 3 else ("",)
        parts = [(f"{nm} {float(v[-1]):+.2f}" if lo < 0 else f"{nm}{float(v[-1]):.0f}").strip()
                 for nm, v in zip(names, series)]
        xr = right
        for k in range(len(parts) - 1, -1, -1):
            tw = d.textlength(parts[k], font=F_SMALL)
            xr -= tw
            d.text((xr, sy0 - 1), parts[k], font=F_SMALL, fill=colours[k % len(colours)])
            xr -= 12
        d.line([(left, plot0), (left, plot1), (right, plot1)], fill=AXIS, width=1)
        if lo < 0.0 < hi:
            d.line([(left, float(ty(0.0))), (right, float(ty(0.0)))], fill=AXIS, width=1)
        d.text((4, plot0), f"{hi:+g}" if lo < 0 else f"{hi:g}", font=F_TINY, fill=DIM)
        d.text((4, plot1 - 12), f"{lo:+g}" if lo < 0 else "0", font=F_TINY, fill=DIM)
        for k, v in enumerate(series):
            polyline(d, tx(ts), ty(v), colours[k % len(colours)])
        # the phase marks
        for tm, label in ((run.T2, "arrival" + (", loop on" if run.loop else "")), (run.T3, "pay-out"),
                          (run.t_land, "touchdown")):
            if tm < 0 or not (t_lo <= tm <= t_now):
                continue
            xm = float(tx(tm))
            d.line([(xm, plot0), (xm, plot1)], fill=C_MARK, width=1)
            if s == 0:
                lw = d.textlength(label, font=F_TINY)
                lx = xm + 3 if xm + 3 + lw < right else xm - 3 - lw
                d.text((lx, plot0), label, font=F_TINY, fill=TEXT)
    return im


# ---- the fan panel ----------------------------------------------------------------------------
def draw_fan(run, i, near, far):
    x0, y0, w, h = FAN
    im, d = panel(w, h)
    n = run.fan_n
    k = int(np.searchsorted(run.fan_frame, run.film_frame[i], side="right")) - 1
    cell = 6
    ox, oy = (w - cell * n) // 2, 26
    img = np.zeros((n, n, 3), np.uint8)
    img[:] = (18, 22, 30)
    note = "no scan yet"
    if k >= 0:
        scan = run.fan[k]
        dist = scan[:, 0].reshape(n, n)
        ids = scan[:, 1].reshape(n, n).astype(np.int64)
        hit = dist > 0.5
        v = 1.0 - np.clip((dist - near) / max(far - near, 1e-6), 0.0, 1.0)
        load = hit & (ids == run.load_id)
        other = hit & ~load
        grey = (60 + 150 * v).astype(np.uint8)
        img[other] = np.stack([grey, grey, grey], -1)[other]
        b = 0.3 + 0.7 * v
        img[load] = np.stack([255 * b, 150 * b, 40 * b], -1).astype(np.uint8)[load]
        big = Image.fromarray(img).resize((cell * n, cell * n), Image.NEAREST)
        im.paste(big, (ox, oy))
        d = ImageDraw.Draw(im)
        if load.any():
            iz, ix = np.nonzero(load)
            cx, cz = ox + (ix.mean() + 0.5) * cell, oy + (iz.mean() + 0.5) * cell
            d.line([(cx - 8, cz), (cx + 8, cz)], fill=(255, 255, 255, 255), width=2)
            d.line([(cx, cz - 8), (cx, cz + 8)], fill=(255, 255, 255, 255), width=2)
            note = f"load: {int(load.sum())} returns, {float(np.median(dist[load])):.1f} m"
        else:
            note = "load: not in the fan"
    # the sensor axis
    ax, az = ox + cell * n / 2, oy + cell * n / 2
    d.line([(ax - 5, az), (ax + 5, az)], fill=DIM, width=1)
    d.line([(ax, az - 5), (ax, az + 5)], fill=DIM, width=1)
    row = int(run.log_row[i])
    sx, sz = run.log[row, 17] * 1e3, run.log[row, 18] * 1e3
    d.text((6, 3), f"tip range fan {n}x{n}, 20 Hz", font=F_SMALL, fill=TEXT)
    d.text((6, h - 36), note, font=F_TINY, fill=TEXT)
    d.text((6, h - 20), f"swing estimate x {sx:+.0f}, z {sz:+.0f} mm", font=F_TINY, fill=TEXT)
    return im


# ---- the events panel -------------------------------------------------------------------------
def draw_events(run, i, sat):
    x0, y0, w, h = EVENTS
    pos = run.ev_pos[i].astype(np.float32)
    neg = run.ev_neg[i].astype(np.float32)
    a = np.clip(pos / sat, 0.0, 1.0)[:, :, None]
    b = np.clip(neg / sat, 0.0, 1.0)[:, :, None]
    bg = np.array(EV_BG, np.float32)
    img = bg + a * (np.array(EV_ON, np.float32) - bg) + b * (np.array(EV_OFF, np.float32) - bg)
    img = np.clip(img, 0, 255).astype(np.uint8)
    eh, ew = img.shape[:2]
    inner_w, inner_h = w - 2, h - 24
    pic = Image.fromarray(img).resize((inner_w, inner_h), Image.NEAREST).convert("RGBA")
    im, d = panel(w, h)
    im.paste(pic, (1, 22))
    d = ImageDraw.Draw(im)
    n_ev = int(pos.sum() + neg.sum())
    head = f"event camera, hero view: {n_ev:,} events this frame   "
    d.text((6, 3), head, font=F_SMALL, fill=TEXT)
    tw = d.textlength(head, font=F_SMALL)
    d.rectangle([6 + tw, 7, 6 + tw + 10, 17], fill=EV_ON + (255,))
    d.text((6 + tw + 14, 3), "+", font=F_SMALL, fill=TEXT)
    d.rectangle([6 + tw + 30, 7, 6 + tw + 40, 17], fill=EV_OFF + (255,))
    d.text((6 + tw + 44, 3), "−", font=F_SMALL, fill=TEXT)
    return im


def tip_label(base, run):
    sx, sy = OUT_W / float(run.hero_size[0]), OUT_H / float(run.hero_size[1])
    x, y, w, h = run.tip_rect
    lx, ly = int(x * sx), int(y * sy) - 22
    lab = Image.new("RGBA", (int(w * sx), 22), (0, 0, 0, 0))
    d = ImageDraw.Draw(lab)
    d.rectangle([0, 0, 150, 21], fill=(12, 16, 22, 196))
    d.text((6, 3), f"tip camera {int(w)}x{int(h)}", font=F_SMALL, fill=TEXT)
    base.alpha_composite(lab, dest=(lx, ly))


def compose(run, i, frame, a):
    base = Image.fromarray(frame).resize((OUT_W, OUT_H), Image.LANCZOS).convert("RGBA")
    base.alpha_composite(draw_traces(run, i, a.window, a.gyro_range, a.acc_range, a.tension_range, a.swing_range),
                         dest=(TRACES[0], TRACES[1]))
    base.alpha_composite(draw_fan(run, i, a.fan_near, a.fan_far), dest=(FAN[0], FAN[1]))
    base.alpha_composite(draw_events(run, i, a.ev_sat), dest=(EVENTS[0], EVENTS[1]))
    tip_label(base, run)
    return np.asarray(base.convert("RGB"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the run folder (a crane_lift_runs round)")
    ap.add_argument("--tag", default="s0c", help="film_<tag>.mp4 and panels_<tag>.npz in the run folder")
    ap.add_argument("--op", default="op_s0_film3", help="the op manifest's base name (json + npz)")
    ap.add_argument("--film", help="the film mp4 (default film_<tag>.mp4)")
    ap.add_argument("--panels", help="the panels npz (default panels_<tag>.npz)")
    ap.add_argument("--out", help="output mp4 (default film_<tag>_sensors.mp4 in the run folder)")
    ap.add_argument("--t0", type=float, default=None, help="first sim second to keep")
    ap.add_argument("--t1", type=float, default=None, help="last sim second to keep")
    ap.add_argument("--still", nargs=2, metavar=("T", "PNG"), help="write the frame nearest sim time T as a PNG and stop")
    ap.add_argument("--crf", default="16")
    ap.add_argument("--window", type=float, default=10.0, help="the traces' history, s")
    ap.add_argument("--gyro-range", type=float, default=1.5, help="deg/s, symmetric (the calm-sea run peaks at 1.1)")
    ap.add_argument("--acc-range", type=float, default=0.5, help="m/s^2, symmetric (the first second's transient clips)")
    ap.add_argument("--tension-range", type=float, default=60.0, help="kN (49 kN carries the load; the start transient clips)")
    ap.add_argument("--swing-range", type=float, default=1000.0, help="mm (the open-loop transfer peaks near 1 m)")
    ap.add_argument("--fan-near", type=float, default=6.0, help="the range drawn brightest, m")
    ap.add_argument("--fan-far", type=float, default=32.0, help="the range drawn darkest, m")
    ap.add_argument("--ev-sat", type=float, default=3.0, help="events per 4x4 bin per film frame that saturate the colour")
    ap.add_argument("--stats", action="store_true", help="print the ranges the traces would need and stop")
    a = ap.parse_args()

    run = Run(a.run, a.tag, a.op, a.film, a.panels)
    if a.stats:
        M, L = run.imu, run.log
        print(f"gyro max |deg/s| per axis: {np.degrees(np.abs(M[:, 1:4]).max(0))}")
        print(f"accel max |m/s^2| (y minus g): {np.abs(M[:, 4:7] - [0, G, 0]).max(0)}")
        print(f"tension max kN: {L[:, 26].max() * 1e-3:.1f}; swing est max mm: {np.hypot(L[:, 17], L[:, 18]).max() * 1e3:.0f}")
        print(f"events per film frame: mean {float(run.ev_pos.sum((1, 2)).mean() + run.ev_neg.sum((1, 2)).mean()):.0f}, "
              f"max bin count {int(max(run.ev_pos.max(), run.ev_neg.max()))}")
        fan = run.fan
        load = fan[:, :, 1] == run.load_id
        d = np.where(load, fan[:, :, 0], np.nan)
        print(f"fan: {len(fan)} scans, load range {np.nanmin(d):.1f} .. {np.nanmax(d):.1f} m, "
              f"other hits {np.nanmin(np.where(~load & (fan[:, :, 0] > 0.5), fan[:, :, 0], np.nan)):.1f} .. "
              f"{np.nanmax(np.where(~load & (fan[:, :, 0] > 0.5), fan[:, :, 0], np.nan)):.1f} m")
        return

    import imageio.v2 as imageio
    reader = imageio.get_reader(run.film)
    n_film = reader.count_frames()
    if n_film != len(run.t):
        print(f"warning: {n_film} frames in {run.film} against {len(run.t)} in the panels npz; the shorter count is used")
    n = min(n_film, len(run.t))

    if a.still:
        T, png = float(a.still[0]), a.still[1]
        i = int(np.argmin(np.abs(run.t[:n] - T)))
        frame = reader.get_data(i)
        Image.fromarray(compose(run, i, frame, a)).save(png)
        print(f"still: film frame {i} at t = {run.t[i]:.3f} s -> {png}")
        return

    out = a.out or os.path.join(a.run, f"film_{a.tag}_sensors.mp4")
    keep = np.ones(n, bool)
    if a.t0 is not None:
        keep &= run.t[:n] >= a.t0
    if a.t1 is not None:
        keep &= run.t[:n] <= a.t1
    idx = np.nonzero(keep)[0]
    writer = imageio.get_writer(out, fps=run.fps, codec="libx264", quality=None, macro_block_size=None,
                                ffmpeg_params=["-crf", a.crf, "-pix_fmt", "yuv420p", "-preset", "medium"])
    import time
    w0 = time.perf_counter()
    for j, i in enumerate(idx):
        writer.append_data(compose(run, int(i), reader.get_data(int(i)), a))
        if j % 300 == 0:
            print(f"  {j}/{len(idx)} frames, t = {run.t[i]:.2f} s, {time.perf_counter() - w0:.0f} s", flush=True)
    writer.close()
    print(f"{len(idx)} frames ({run.t[idx[0]]:.2f} to {run.t[idx[-1]]:.2f} s) -> {out}, "
          f"{os.path.getsize(out) / 1e6:.1f} MB, {time.perf_counter() - w0:.0f} s")


if __name__ == "__main__":
    main()
