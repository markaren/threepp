"""Offline 2D overlays and the end card for the snake net-pen film, from the logged telemetry.

Pure code: numpy + PIL (+ matplotlib for the end card's plots). No threepp, no GPU. The film renders
the 3D frames (with the head-camera inset and the sonar panel already in them); this module draws the
HUD on top of each frame from the mission log, so the layout can change without a re-render.

    python snake_panels.py --preview                      # PNGs in D:/dev/snake_out/film

API:
    P = Panels("seed0_telemetry.npz", "seed0.json", "D:/dev/snake_out/sweep")
    out = P.compose(frame_rgb, t_mission, speedup=1.0, sonar_rect=None, sonar_gap_xy=None)
    card = P.end_card((1920, 1080))

Inputs (written by snake_netpen.py --mission; the loader checks every key it uses and fails loudly):
  telemetry npz: cols, log (N x len(cols)), links (N, 9, 3) world link centres (link 0 = head),
      quats (N, 9, 4) xyzw, phases, gaits (index tables for the phase / gait columns), dock_c, dock_u,
      tear_true (pen bearing, rad); optional net_y3 (net nodes at the mission depth, x/z), events.
  summary json: completed, mission_time_s, path_length_m, per_gait, follow_standoff_err_rms_m,
      min_link_to_net_m, tear{mode, t_s, bearing_err_deg}, dock_error, energy_abs_J, events.
  sweep dir: sweep_complex.json and sweep_control.json (snake_sweep.py: rows, verdicts, reference_gait).

Top view convention (same as snake_mission.plot_top): x to the right, z down the page, looking down.
Yaw psi: heading (x, z) = (cos psi, -sin psi).
"""

import argparse
import json
import math
import os
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SS = 2                                     # panels are drawn at 2x and downsampled (anti-aliasing)

ACCENT = (242, 194, 48)                    # the snake's yellow
EEL_C = (150, 222, 236)                    # eel-like trace (pale cyan)
WHITE = (238, 242, 243)
GREY = (168, 182, 186)
DIM = (112, 128, 132)
PLATE = (5, 15, 19, 172)
PLATE_EDGE = (255, 255, 255, 30)

PEN_R = 7.0
N_LINKS = 9

# gait name (telemetry 'gaits' table) -> swimming pattern; keep in step with snake_mission.GAITS
GAIT_PATTERN = {"cruise": "lateral", "creep": "lateral", "dock": "lateral", "undock": "lateral",
                "inspect": "eel", "stop": None}
PATTERN_LABEL = {"lateral": "lateral undulation", "eel": "eel-like", None: "joints straight"}
PATTERN_COLOR = {"lateral": ACCENT, "eel": EEL_C, None: DIM}

PHASE_LABEL = {
    "DOCKED": "DOCKED", "UNDOCK": "UNDOCKING", "ACQUIRE": "LOOKING FOR THE NET ON SONAR",
    "FOLLOW_WALL": "FOLLOWING THE NET ON SONAR", "TEAR": "TEAR DETECTED",
    "INSPECT": "INSPECTING THE TEAR, PASS 1", "UTURN": "U-TURN", "INSPECT_2": "INSPECTING THE TEAR, PASS 2",
    "RETURN": "RETURNING TO THE DOCK", "APPROACH": "APPROACHING THE DOCK", "TURN_IN": "TURNING IN TO THE DOCK",
    "FINAL": "DOCKING", "DOCK": "DOCKING", "CAPTURE": "DOCKING", "LATCHED": "DOCKED",
    "RETREAT": "BACKING OUT FOR ANOTHER TRY", "ABORT": "DOCKING ABORTED",
}

TEAR_NOTE = {                               # summary tear.mode -> the sonar annotation
    "sonar": "gap in the wall return = the tear",
    "sector+sonar": "reported tear sector, gap confirmed on sonar",
    "sector": "reported tear sector (no sonar gap fire)",
    "sector (sonar did not confirm)": "reported tear sector (sonar did not confirm the gap)",
}

REQ_ARRAYS = ("cols", "log", "links", "quats", "phases", "gaits", "dock_c", "dock_u", "tear_true")
REQ_COLS = ("t", "phase", "gait", "speed", "meas", "d_head_true", "dpsi_head", "tear_fired", "com_x", "com_z",
            "wall_seen")
REQ_SUMMARY = ("completed", "mission_time_s", "path_length_m", "per_gait", "follow_standoff_err_rms_m",
               "min_link_to_net_m", "tear", "dock_error", "energy_abs_J", "events")

CITATION = ("E. Kelasidi, P. Liljeback, K. Y. Pettersen, J. T. Gravdahl (2015). Experimental investigation of "
            "efficient locomotion of underwater snake robots for lateral undulation and eel-like motion "
            "patterns. Robotics and Biomimetics 2:8.")
HONESTY = ("Trend-level check: the paper tabulates no absolute speeds. Coefficients are the ones the paper prints "
           "(from its generic simulation study, not identified on Mamba). The speed drop past ~30 deg amplitude "
           "is reproduced only for lateral undulation with the paper's full coefficients.")
CREDITS = "threepp (PhysX, Warp, Vulkan) - simulated, not filmed"


class TelemetryError(RuntimeError):
    pass


@lru_cache(maxsize=64)
def font(px, weight="r"):
    """Segoe UI at `px` pixels (weight r / sb / b / l), falling back to DejaVu / PIL's default."""
    names = {"r": ("segoeui.ttf",), "sb": ("seguisb.ttf", "segoeuib.ttf"), "b": ("segoeuib.ttf",),
             "l": ("segoeuil.ttf", "segoeui.ttf")}[weight]
    for name in names + ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, max(int(round(px)), 6))
        except Exception:                  # noqa: BLE001 - try the next face
            continue
    return ImageFont.load_default()


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _rgba(c, a=255):
    return tuple(c[:3]) + (int(a),)


class Layer:
    """One RGBA panel in output pixels; drawn at SS x, downsampled by done()."""

    def __init__(self, w, h):
        self.w, self.h = int(math.ceil(w)), int(math.ceil(h))
        self.im = Image.new("RGBA", (self.w * SS, self.h * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def plate(self, r=8.0, fill=PLATE, edge=PLATE_EDGE):
        self.d.rounded_rectangle([0, 0, self.w * SS - 1, self.h * SS - 1], radius=r * SS, fill=fill,
                                 outline=edge, width=SS)

    def line(self, pts, fill, width=1.0):
        pts = [(x * SS, y * SS) for x, y in pts]
        if len(pts) >= 2:
            self.d.line(pts, fill=fill, width=max(int(round(width * SS)), 1), joint="curve")

    def circle(self, x, y, r, fill=None, outline=None, width=1.0):
        self.d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], fill=fill, outline=outline,
                       width=max(int(round(width * SS)), 1))

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0, r=0.0):
        box = [min(x0, x1) * SS, min(y0, y1) * SS, max(x0, x1) * SS, max(y0, y1) * SS]
        if r > 0:
            self.d.rounded_rectangle(box, radius=r * SS, fill=fill, outline=outline, width=max(int(round(width * SS)), 1))
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=max(int(round(width * SS)), 1))

    def polygon(self, pts, fill):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=fill)

    def text(self, x, y, s, px, weight="r", fill=WHITE, anchor="la"):
        self.d.text((x * SS, y * SS), s, font=font(px * SS, weight), fill=fill, anchor=anchor)

    @staticmethod
    def tw(s, px, weight="r"):
        return font(px * SS, weight).getlength(s) / SS

    def done(self):
        return self.im.resize((self.w, self.h), Image.LANCZOS)


def text_width(s, px, weight="r"):
    return Layer.tw(s, px, weight)


# ---- inputs ------------------------------------------------------------------------------------------
class Telemetry:
    """The mission log (snake_netpen.run_mission's npz), checked key by key."""

    def __init__(self, path):
        if not os.path.exists(path):
            raise TelemetryError(f"telemetry not found: {path}")
        d = np.load(path, allow_pickle=False)
        hint = "snake_netpen.run_mission's writer changed; update snake_panels (REQ_ARRAYS / REQ_COLS / tables)."
        miss = [k for k in REQ_ARRAYS if k not in d.files]
        if miss:
            raise TelemetryError(f"{path}: missing arrays {miss} (has {d.files}). {hint}")
        self.cols = [str(c) for c in d["cols"]]
        miss = [k for k in REQ_COLS if k not in self.cols]
        if miss:
            raise TelemetryError(f"{path}: missing log columns {miss} (has {self.cols}). {hint}")
        self.L = np.asarray(d["log"], np.float64)
        if self.L.ndim != 2 or self.L.shape[1] != len(self.cols):
            raise TelemetryError(f"{path}: log shape {self.L.shape} does not match {len(self.cols)} cols. {hint}")
        n = len(self.L)
        self.links = np.asarray(d["links"], np.float64)
        self.quats = np.asarray(d["quats"], np.float64)
        if self.links.shape != (n, N_LINKS, 3) or self.quats.shape != (n, N_LINKS, 4):
            raise TelemetryError(f"{path}: links {self.links.shape} / quats {self.quats.shape}, expected "
                                 f"({n}, {N_LINKS}, 3/4). {hint}")
        self.c = {k: i for i, k in enumerate(self.cols)}
        self.t = self.col("t")
        if n < 2 or np.any(np.diff(self.t) <= 0):
            raise TelemetryError(f"{path}: the time column is not strictly increasing. {hint}")
        self.phases = [str(p) for p in d["phases"]]
        self.gaits = [str(g) for g in d["gaits"]]
        unknown = [g for g in self.gaits if g not in GAIT_PATTERN]
        if unknown:
            raise TelemetryError(f"{path}: gaits {unknown} have no pattern in snake_panels.GAIT_PATTERN "
                                 f"(lateral or eel?). {hint}")
        self.phase_i = self.col("phase").astype(int)
        self.gait_i = self.col("gait").astype(int)
        if self.phase_i.min() < 0 or self.phase_i.max() >= len(self.phases):
            raise TelemetryError(f"{path}: phase index outside the phases table. {hint}")
        gi = np.clip(self.gait_i, 0, len(self.gaits) - 1)
        self.pattern = [GAIT_PATTERN[self.gaits[k]] if self.gait_i[j] >= 0 else None for j, k in enumerate(gi)]
        self.dock_c = np.asarray(d["dock_c"], np.float64)
        self.dock_u = np.asarray(d["dock_u"], np.float64)
        self.tear_true = float(d["tear_true"])
        self.net = np.asarray(d["net_y3"], np.float64) if "net_y3" in d.files else None
        self.seed = int(d["seed"]) if "seed" in d.files else None
        self.has_current = "cur_x" in self.c and "cur_z" in self.c
        # link yaw from the quaternion's forward axis; joint i = yaw(link i) - yaw(link i+1), head first
        q = self.quats
        qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        yaw = np.arctan2(2 * (qw * qy - qx * qz), 1 - 2 * (qy * qy + qz * qz))
        self.joints = wrap(yaw[:, :-1] - yaw[:, 1:])
        self.dpsi = wrap(self.col("dpsi_head"))            # head yaw about the mean heading
        fired = self.col("tear_fired") > 0.5
        self.i_fire = int(np.argmax(fired)) if fired.any() else None
        self.t_fire = float(self.t[self.i_fire]) if self.i_fire is not None else None

    def col(self, name):
        return self.L[:, self.c[name]]

    def idx(self, t):
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))

    def phase_start(self, name):
        if name not in self.phases:
            return None
        m = np.nonzero(self.phase_i == self.phases.index(name))[0]
        return float(self.t[m[0]]) if len(m) else None

    def window(self, i, seconds):
        dt = float(np.median(np.diff(self.t)))
        return max(0, i - int(round(seconds / dt)) + 1), i + 1


def load_summary(path):
    with open(path) as fh:
        s = json.load(fh)
    miss = [k for k in REQ_SUMMARY if k not in s]
    if miss:
        raise TelemetryError(f"{path}: summary keys {miss} missing (has {sorted(s)}); snake_mission.summarize "
                             f"changed, update snake_panels.REQ_SUMMARY / end_card.")
    for k in ("mode", "t_s", "bearing_err_deg", "detected"):
        if k not in s["tear"]:
            raise TelemetryError(f"{path}: tear.{k} missing (has {sorted(s['tear'])}).")
    return s


def load_sweep(sweep_dir):
    out = {}
    for coef in ("complex", "control"):
        p = os.path.join(sweep_dir, f"sweep_{coef}.json")
        with open(p) as fh:
            d = json.load(fh)
        for k in ("rows", "verdicts", "reference_gait"):
            if k not in d:
                raise TelemetryError(f"{p}: key {k} missing; snake_sweep.py's output changed.")
        out[coef] = d
    return out


def sweep_series(rows, pattern, vary, y="speed"):
    """(x, y) along one axis of the sweep around the reference gait alpha 30, omega 120, delta 30."""
    ref = {"alpha": 30, "omega": 120, "delta": 30}
    sel = [r for r in rows if r["pattern"] == pattern and all(r[k] == v for k, v in ref.items() if k != vary)]
    sel.sort(key=lambda r: r[vary])
    return np.array([r[vary] for r in sel], float), np.array([r[y] for r in sel], float)


# ---- the HUD ------------------------------------------------------------------------------------------
class Panels:
    """compose(frame, t) draws the mission HUD over a rendered frame; end_card() draws the closing card.

    layout (1080p units, scaled with the frame height; x < 0 counts from the right edge):
      title_xy (24, 24), gait_w 470, map_size 300, map_xy (-24, 24), margin 14.
    """

    def __init__(self, telemetry_npz, summary_json, sweep_dir, layout=None):
        self.tel = Telemetry(telemetry_npz)
        self.summ = load_summary(summary_json)
        self.sweep = load_sweep(sweep_dir)
        self.lay = dict(title_xy=(24, 24), gait_w=470, map_size=300, map_xy=(-24, 24), margin=14)
        self.lay.update(layout or {})
        ref = self.sweep["complex"]["reference_gait"]
        self.ref_yaw = (ref["lateral"]["head_yaw_rms_deg"], ref["eel"]["head_yaw_rms_deg"])
        self.ref_speed = (ref["lateral"]["speed"], ref["eel"]["speed"])
        self.seed = self.summ.get("seed", self.tel.seed)
        net = self.tel.net
        if net is not None and len(net) >= 8:
            b = np.arctan2(net[:, 1], net[:, 0])
            self.net_ring = net[np.argsort(b)]
        else:
            self.net_ring = None
        # the tear on the net: the ring node nearest the tear bearing (else the rest circle)
        tb = self.tel.tear_true
        if self.net_ring is not None:
            b = np.arctan2(self.net_ring[:, 1], self.net_ring[:, 0])
            k = int(np.argmin(np.abs(wrap(b - tb))))
            r = float(np.hypot(*self.net_ring[k]))
        else:
            r = PEN_R
        self.tear_xz = np.array([r * math.cos(tb), r * math.sin(tb)])

    # -- per frame
    def compose(self, frame_rgb, t_mission, speedup=1.0, sonar_rect=None, sonar_gap_xy=None):
        """frame_rgb (H, W, 3) uint8 -> uint8 with the HUD. speedup > 1 shows a TIME-LAPSE badge;
        sonar_rect (x, y, w, h) in frame pixels = the sonar panel, annotated for 6 s after the tear fires;
        sonar_gap_xy optionally points the annotation's arrow at the gap."""
        fr = np.asarray(frame_rgb)
        if fr.ndim != 3 or fr.shape[2] != 3:
            raise ValueError(f"frame must be (H, W, 3), got {fr.shape}")
        H, W = fr.shape[:2]
        s = H / 1080.0
        img = Image.fromarray(fr.astype(np.uint8), "RGB").convert("RGBA")
        i = self.tel.idx(t_mission)
        m = self.lay["margin"] * s
        tx, ty = self.lay["title_xy"]
        title = self._title(i, t_mission, s)
        x0, y0 = self._xy(tx, ty, title.width, W, s)
        img.alpha_composite(title, (int(x0), int(y0)))
        if speedup and speedup > 1.01:
            badge = self._badge(speedup, s)
            img.alpha_composite(badge, (int(x0 + title.width + m), int(y0)))
        gait = self._gait(i, s)
        img.alpha_composite(gait, (int(x0), int(y0 + title.height + m)))
        mp = self._minimap(i, t_mission, s)
        mx, my = self.lay["map_xy"]
        x1, y1 = self._xy(mx, my, mp.width, W, s)
        img.alpha_composite(mp, (int(x1), int(y1)))
        if sonar_rect is not None and self.tel.t_fire is not None and 0.0 <= t_mission - self.tel.t_fire <= 6.0:
            self._sonar_note(img, sonar_rect, sonar_gap_xy, t_mission - self.tel.t_fire, s)
        return np.asarray(img.convert("RGB"))

    @staticmethod
    def _xy(x, y, w, W, s):
        return (W + x * s - w if x < 0 else x * s), y * s

    def _title(self, i, t, s):
        tel = self.tel
        ph = tel.phases[tel.phase_i[i]]
        label = PHASE_LABEL.get(ph, ph.replace("_", " "))
        a, b = tel.window(i, 1.0)
        spd = float(np.nanmean(tel.col("speed")[a:b]))
        a2, b2 = tel.window(i, 0.5)
        d_true = float(np.nanmean(tel.col("d_head_true")[a2:b2]))
        seen = tel.col("wall_seen")[i] > 0.5
        meas = tel.col("meas")[a2:b2]
        meas = float(np.nanmean(meas)) if seen and np.isfinite(meas).any() else None
        tt = max(float(t), 0.0)
        clock = f"T+ {int(tt // 60)}:{tt % 60:04.1f}"
        fields = [("", clock), ("speed", f"{spd:.2f} m/s"), ("to net", f"{d_true:.2f} m"),
                  ("sonar", f"{meas:.2f} m" if meas is not None else "no wall")]
        pat = tel.pattern[i]
        fields.append(("gait", PATTERN_LABEL[pat]))
        T1, T2, T3, T4 = 25, 14.5, 23, 16.5
        head = "Underwater snake robot on net-pen patrol"
        sub = "Mamba-style, 9 links, 1.62 m.  Kelasidi et al. (2015) fluid model in PhysX"
        gap = 16
        # fixed slots (widest value each field can take), so the plate and the columns never jitter
        tmpl = [["T+ 00:00.0"], ["0.00 m/s"], ["00.00 m"], ["00.00 m", "no wall"], list(PATTERN_LABEL.values())]
        slots = [(text_width(lb + "  ", T4 - 2) if lb else 0.0) + max(text_width(v, T4, "sb") for v in tv)
                 for (lb, _), tv in zip(fields, tmpl)]
        labels = [PHASE_LABEL.get(p, p.replace("_", " ")) for p in tel.phases]
        w = max(text_width(head, T1, "sb"), text_width(sub, T2), max(text_width(lb, T3, "sb") for lb in labels),
                sum(slots) + gap * (len(slots) - 1)) + 36
        w, h = w * s, 156 * s
        L = Layer(w, h)
        L.plate(8 * s)
        x = 18 * s
        L.text(x, 12 * s, head, T1 * s, "sb", WHITE)
        L.text(x, 44 * s, sub, T2 * s, "r", GREY)
        L.line([(x, 72 * s), (w - 18 * s, 72 * s)], _rgba(WHITE, 40), 1 * s)
        L.rect(x, 84 * s, x + 4 * s, 108 * s, fill=_rgba(ACCENT))
        L.text(x + 12 * s, 82 * s, label, T3 * s, "sb", ACCENT)
        y = 120 * s
        for (lb, v), sw in zip(fields, slots):
            x1 = x
            if lb:
                L.text(x1, y + 1.5 * s, lb, (T4 - 2) * s, "r", GREY)
                x1 += L.tw(lb + "  ", (T4 - 2) * s)
            col = PATTERN_COLOR[pat] if lb == "gait" else WHITE
            L.text(x1, y, v, T4 * s, "sb", col)
            x += (sw + gap) * s
        return L.done()

    def _badge(self, speedup, s):
        txt = f"x{speedup:.0f}" if abs(speedup - round(speedup)) < 0.05 else f"x{speedup:.1f}"
        w = (text_width("TIME-LAPSE", 15, "sb") + text_width(txt, 26, "b") + 44) * s
        L = Layer(w, 50 * s)
        L.plate(8 * s, edge=_rgba(ACCENT, 200))
        L.text(16 * s, 25 * s, "TIME-LAPSE", 15 * s, "sb", GREY, anchor="lm")
        L.text(w - 16 * s, 25 * s, txt, 26 * s, "b", ACCENT, anchor="rm")
        return L.done()

    def _gait(self, i, s):
        tel = self.tel
        W_, H_ = self.lay["gait_w"], 176
        L = Layer(W_ * s, H_ * s)
        L.plate(8 * s)
        pat = tel.pattern[i]
        col = PATTERN_COLOR[pat]
        x = 18 * s
        L.text(x, 12 * s, "GAIT", 14 * s, "sb", GREY)
        L.text(x + 46 * s, 9 * s, PATTERN_LABEL[pat], 19 * s, "sb", col)
        a, b = tel.window(i, 2.0)
        rms = math.degrees(float(np.sqrt(np.nanmean(tel.dpsi[a:b] ** 2))))
        L.text((W_ - 18) * s, 13 * s, f"head yaw RMS, last 2 s:  {rms:.0f} deg", 13.5 * s, "r", GREY, anchor="ra")
        # head yaw relative to the mean heading, last 8 s, coloured by gait
        px0, px1, py0, py1 = 18, 300, 44, 128
        ymid, yk = (py0 + py1) / 2, (py1 - py0) / 2 / math.radians(90)
        jk = (py1 - py0) / 2 / math.radians(60)          # joint bars: full height = the 60 deg joint limit
        for dv in (-90, -45, 0, 45, 90):
            yy = ymid - math.radians(dv) * yk
            L.line([(px0 * s, yy * s), (px1 * s, yy * s)], _rgba(WHITE, 55 if dv == 0 else 20), 1 * s)
            if dv % 90 == 0:
                L.text((px1 + 4) * s, yy * s, f"{dv:+d}" if dv else "0", 11 * s, "r", DIM, anchor="lm")
        a, b = tel.window(i, 8.0)
        ts, ys = tel.t[a:b], tel.dpsi[a:b]
        span = 8.0
        xs = px1 - (tel.t[i] - ts) / span * (px1 - px0)
        pats = tel.pattern[a:b]
        k0 = 0
        for k in range(1, len(xs) + 1):
            if k == len(xs) or pats[k] != pats[k0]:
                seg = [(xs[j] * s, (ymid - np.clip(ys[j], -1.57, 1.57) * yk) * s) for j in range(k0, min(k + 1, len(xs)))
                       if np.isfinite(ys[j])]
                L.line(seg, _rgba(PATTERN_COLOR[pats[k0]], 235), 1.8 * s)
                k0 = k
        if np.isfinite(ys[-1]):
            L.circle(px1 * s, (ymid - np.clip(ys[-1], -1.57, 1.57) * yk) * s, 3.2 * s, fill=_rgba(col))
        L.text(px0 * s, (py1 + 5) * s, "head yaw vs heading, last 8 s (deg)", 12 * s, "r", DIM)
        # the 8 joint angles now, head to tail
        bx0, bx1 = 338, W_ - 18
        bw = (bx1 - bx0) / 8.0
        L.line([(bx0 * s, ymid * s), (bx1 * s, ymid * s)], _rgba(WHITE, 55), 1 * s)
        for j, ph in enumerate(tel.joints[i]):
            cx = bx0 + (j + 0.5) * bw
            hh = float(np.clip(ph, -1.05, 1.05)) * jk
            L.rect((cx - bw * 0.3) * s, ymid * s, (cx + bw * 0.3) * s, (ymid - hh) * s, fill=_rgba(col, 225))
        L.text(bx0 * s, (py1 + 5) * s, "joints, head to tail", 12 * s, "r", DIM)
        L.line([(18 * s, 150 * s), ((W_ - 18) * s, 150 * s)], _rgba(WHITE, 30), 1 * s)
        L.text(18 * s, 155 * s, f"Sweep, the paper's reference gait: head yaw RMS {self.ref_yaw[0]:.0f} deg lateral, "
                                f"{self.ref_yaw[1]:.0f} deg eel-like", 12.5 * s, "r", GREY)
        return L.done()

    def _map_view(self):
        """(centre x, centre z, half extent m) of the top view: 'pen' = the whole pen, 'mission' = the
        square around everything the mission touches (track, dock, tear), padded."""
        if self.lay.get("map_view", "mission") == "pen":
            return 0.0, 0.0, PEN_R + 0.6
        tel = self.tel
        pts = [tel.links[::30, :, :].reshape(-1, 3)[:, [0, 2]], np.stack([tel.col("com_x"), tel.col("com_z")], 1),
               self._dock_corners(), self.tear_xz[None, :]]
        q = np.concatenate(pts, 0)
        q = q[np.all(np.isfinite(q), 1)]
        lo, hi = q.min(0), q.max(0)
        c = (lo + hi) / 2
        half = max(float((hi - lo).max()) / 2 + 1.4, 3.0)
        return float(c[0]), float(c[1]), half

    def _dock_corners(self, hw=0.22, hl=1.0):
        c, u = self.tel.dock_c, self.tel.dock_u
        n = np.array([-u[2], u[0]])
        ux = np.array([u[0], u[2]])
        return np.array([c[[0, 2]] + sx * hl * ux + sy * hw * n for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def _minimap(self, i, t, s):
        tel = self.tel
        M = self.lay["map_size"]
        head_h, foot_h = 34, 62
        W_, H_ = M + 20, M + head_h + foot_h
        L = Layer(W_ * s, H_ * s)
        L.plate(8 * s)
        L.text(14 * s, 11 * s, "TOP VIEW", 14 * s, "sb", GREY)
        L.text((W_ - 14) * s, 12 * s, f"net pen r {PEN_R:.0f} m, depth 3 m", 12.5 * s, "r", DIM, anchor="ra")
        if not hasattr(self, "_view"):
            self._view = self._map_view()
        vx, vz, ext = self._view
        k = M * s / (2 * ext)                          # px per m
        G = Layer(M * s, M * s)                        # the map itself, clipped to its square
        G.rect(0, 0, M * s, M * s, fill=(0, 0, 0, 40), r=4 * s)
        cx, cy = M * s / 2, M * s / 2

        def P(x, z):
            return cx + (x - vx) * k, cy + (z - vz) * k

        # the rest circle and the bowed net (a snapshot at the end of the run)
        G.circle(*P(0.0, 0.0), PEN_R * k, outline=_rgba(WHITE, 45), width=1 * s)
        if self.net_ring is not None:
            pts = [P(x, z) for x, z in self.net_ring]
            G.line(pts + pts[:1], _rgba(WHITE, 150), 1.5 * s)
        # the dock (cradle), 2 m along its axis
        cor = self._dock_corners()
        G.polygon([P(*q) for q in cor], _rgba(GREY, 120))
        G.line([P(*q) for q in np.vstack([cor, cor[:1]])], _rgba(WHITE, 190), 1 * s)
        c, u = tel.dock_c, tel.dock_u
        n = np.array([-u[2], u[0]])
        side = n if float(n @ c[[0, 2]]) < 0 else -n          # label on the pen-centre side of the dock
        G.text(*P(*(c[[0, 2]] + 0.75 * side)), "dock", 12 * s, "r", GREY, anchor="mm")
        # the COM track so far, 10 Hz, coloured by gait pattern
        step = max(1, int(round(0.1 / float(np.median(np.diff(tel.t))))))
        idx = np.arange(0, i + 1, step)
        if len(idx) and idx[-1] != i:
            idx = np.append(idx, i)
        cxz = np.stack([tel.col("com_x")[idx], tel.col("com_z")[idx]], 1)
        pats = [tel.pattern[j] for j in idx]
        k0 = 0
        for q in range(1, len(idx) + 1):
            if q == len(idx) or pats[q] != pats[k0]:
                seg = [P(*cxz[j]) for j in range(k0, min(q + 1, len(idx)))]
                G.line(seg, _rgba(PATTERN_COLOR[pats[k0]], 150), 1.6 * s)
                k0 = q
        # the tear: hidden until detected, then a pulsing ring; label outside the net
        if tel.t_fire is not None and t >= tel.t_fire:
            tx, ty = P(*self.tear_xz)
            fr = ((t - tel.t_fire) / 1.4) % 1.0
            G.circle(tx, ty, (7 + 14 * fr) * s, outline=_rgba(ACCENT, 230 * (1 - fr)), width=1.8 * s)
            G.circle(tx, ty, 5.5 * s, outline=_rgba(ACCENT), width=2.2 * s)
            out = self.tear_xz / max(float(np.linalg.norm(self.tear_xz)), 1e-6)
            lx, ly = P(*(self.tear_xz + 0.55 * out))
            G.text(lx, ly, "tear", 13 * s, "sb", ACCENT, anchor="rm" if out[0] < 0 else "lm")
        # the snake: 9 link centres, head bigger
        pts = [P(*q) for q in tel.links[i][:, [0, 2]]]
        G.line(pts, _rgba((6, 10, 12), 235), 6.4 * s)
        G.line(pts, _rgba(ACCENT), 3.4 * s)
        for j, (x, y) in enumerate(pts):
            G.circle(x, y, (4.6 if j == 0 else 2.3) * s, fill=_rgba(ACCENT if j else WHITE),
                     outline=_rgba((6, 10, 12), 235), width=1.0 * s)
        L.im.alpha_composite(G.im, (10 * SS, int(round(head_h * s)) * SS))
        # footer: legend, current, scale bar
        fy = (head_h + M + 16) * s
        x = 14 * s
        for pat in ("lateral", "eel"):
            L.line([(x, fy), (x + 18 * s, fy)], _rgba(PATTERN_COLOR[pat]), 2.4 * s)
            L.text(x + 24 * s, fy, PATTERN_LABEL[pat], 12 * s, "r", GREY, anchor="lm")
            x += (24 + text_width(PATTERN_LABEL[pat], 12) + 16) * s
        fy2 = (head_h + M + 40) * s
        if tel.has_current:
            cu = np.array([tel.col("cur_x")[i], tel.col("cur_z")[i]])
            sp = float(np.hypot(*cu))
            if sp > 1e-3:
                d = cu / sp
                ax, ay = 24 * s, fy2
                ln = 10 * s
                sx_, sy_, ex, ey = ax - d[0] * ln, ay - d[1] * ln, ax + d[0] * ln, ay + d[1] * ln
                G_ = L
                G_.line([(sx_, sy_), (ex, ey)], _rgba(WHITE, 190), 1.5 * s)
                pn = np.array([-d[1], d[0]])
                G_.polygon([(ex, ey), (ex - d[0] * 7 * s + pn[0] * 4 * s, ey - d[1] * 7 * s + pn[1] * 4 * s),
                            (ex - d[0] * 7 * s - pn[0] * 4 * s, ey - d[1] * 7 * s - pn[1] * 4 * s)], _rgba(WHITE, 190))
                L.text(40 * s, fy2, f"current {sp:.2f} m/s", 12 * s, "r", GREY, anchor="lm")
        sb = 1.0 * k
        x1 = (W_ - 14) * s
        L.line([(x1 - sb, fy2), (x1, fy2)], _rgba(WHITE, 200), 1.4 * s)
        for xx in (x1 - sb, x1):
            L.line([(xx, fy2 - 4 * s), (xx, fy2 + 4 * s)], _rgba(WHITE, 200), 1.4 * s)
        L.text(x1 - sb - 8 * s, fy2, "1 m", 12 * s, "r", GREY, anchor="rm")
        return L.done()

    def _sonar_note(self, img, rect, gap_xy, dt, s):
        x, y, w, h = rect
        mode = self.summ["tear"]["mode"]
        note = TEAR_NOTE.get(mode, f"tear ({mode})")
        fade = min(1.0, dt / 0.3) * min(1.0, (6.0 - dt) / 0.5)
        W, H = img.size
        L = Layer(W, H)
        L.rect(x, y, x + w, y + h, outline=_rgba(ACCENT, 230 * fade), width=2.2 * s, r=4 * s)
        tw = text_width(note, 17 * s, "sb") + 28 * s
        bx, by = x, (y - 44 * s if y - 44 * s >= 0 else y + h + 10 * s)   # above the panel, else below
        L.rect(bx, by, bx + tw, by + 34 * s, fill=(5, 15, 19, int(190 * fade)), outline=_rgba(ACCENT, 200 * fade),
               width=1.2 * s, r=6 * s)
        L.text(bx + 14 * s, by + 17 * s, note, 17 * s, "sb", _rgba(ACCENT, 255 * fade), anchor="lm")
        if gap_xy is not None:
            gx, gy = gap_xy
            sx, sy = bx + tw * 0.5, by + 34 * s
            L.line([(sx, sy), (gx, gy)], _rgba(ACCENT, 230 * fade), 2 * s)
            d = np.array([gx - sx, gy - sy])
            d = d / max(np.linalg.norm(d), 1e-6)
            pn = np.array([-d[1], d[0]])
            L.polygon([(gx, gy), (gx - d[0] * 12 * s + pn[0] * 6 * s, gy - d[1] * 12 * s + pn[1] * 6 * s),
                       (gx - d[0] * 12 * s - pn[0] * 6 * s, gy - d[1] * 12 * s - pn[1] * 6 * s)],
                      _rgba(ACCENT, 230 * fade))
        img.alpha_composite(L.done())

    # -- the end card
    def end_card(self, size=(1920, 1080), seeds=None, seeds_note=""):
        """The closing card. seeds: optional list of summary dicts (one per seed); the mission block
        then shows the result across them (n/n docked, ranges) instead of this run's alone;
        seeds_note replaces the block header's parenthesis (e.g. which tree the seeds ran on)."""
        W, H = size
        s = H / 1080.0
        bg = (9, 22, 26)
        g = 0.35 * np.linspace(0, 1, H)[:, None, None]                 # a soft vertical gradient
        base = np.array(bg, float) * (1.0 - g) + np.array((4, 10, 12), float) * g
        img = Image.fromarray(np.repeat(base, W, axis=1).clip(0, 255).astype(np.uint8), "RGB").convert("RGBA")
        fig_w, fig_h = int(1060 * s), int(760 * s)
        fig = self._validation_fig(fig_w, fig_h, s, bg)
        img.alpha_composite(fig, (int(40 * s), int(120 * s)))
        L = Layer(W, H)
        x0 = 56 * s
        L.text(x0, 40 * s, "The swimming, checked against the paper", 32 * s, "sb", WHITE)
        L.text(x0, 84 * s, "Kelasidi et al. (2015) fluid model in PhysX at 240 Hz, 9 links x 0.18 m. "
                           "Speed and joint power, mean over 10-30 s of 30 s runs.", 16 * s, "r", GREY)
        rx = 1150 * s
        rw = W - rx - 56 * s
        y = 40 * s
        L.text(rx, y, "The numbers", 32 * s, "sb", WHITE)
        y = 100 * s
        L.text(rx, y, "At the paper's reference gait", 19 * s, "sb", ACCENT)
        L.text(rx, y + 27 * s, "alpha 30 deg, omega 120 deg/s, delta 30 deg, full coefficients", 14 * s, "r", GREY)
        ref = self.sweep["complex"]["reference_gait"]
        c1, c2 = rx + 300 * s, rx + 470 * s
        y += 60 * s
        L.text(c1, y, "lateral", 14.5 * s, "sb", ACCENT)
        L.text(c2, y, "eel-like", 14.5 * s, "sb", EEL_C)
        rows = [("speed", f"{ref['lateral']['speed']:.2f} m/s", f"{ref['eel']['speed']:.2f} m/s"),
                ("head yaw RMS", f"{ref['lateral']['head_yaw_rms_deg']:.0f} deg", f"{ref['eel']['head_yaw_rms_deg']:.0f} deg"),
                ("mean |joint power|", f"{ref['lateral']['power_abs']:.2f} W", f"{ref['eel']['power_abs']:.2f} W")]
        y += 26 * s
        for lb, a, b in rows:
            L.text(rx, y, lb, 17 * s, "r", GREY)
            L.text(c1, y, a, 19 * s, "sb", WHITE)
            L.text(c2, y, b, 19 * s, "sb", WHITE)
            y += 32 * s
        ig = self.inspect_gait()
        if ig and GAIT_PATTERN[ig] == "eel":
            why = "Eel-like is the inspection gait: slower, but the head (camera, sonar) is steadier."
        else:
            why = ("Eel-like holds the head (camera, sonar) steadier; this run inspected with "
                   f"{PATTERN_LABEL[GAIT_PATTERN[ig]] if ig else 'no inspection pass'}" + (f" ({ig})." if ig else "."))
        L.text(rx, y + 2 * s, why, 14 * s, "r", GREY)
        y += 50 * s
        L.line([(rx, y), (rx + rw, y)], _rgba(WHITE, 40), 1 * s)
        y += 18 * s
        sm = self.summ
        seed = self.seed
        if seeds:
            L.text(rx, y, f"The mission, {len(seeds)} seeds ({seeds_note or 'current direction jitter'})", 19 * s,
                   "sb", ACCENT)
            rows = self._mission_rows_multi(seeds)
        else:
            L.text(rx, y, f"The mission{'' if seed is None else f', seed {seed}'}", 19 * s, "sb", ACCENT)
            rows = self._mission_rows(sm)
        y += 36 * s
        for lb, v in rows:
            L.text(rx, y, lb, 16 * s, "r", GREY)
            self._wrap_text(L, rx + 250 * s, y, v, 17 * s, "sb", WHITE, rw - 250 * s, 23 * s)
            y += max(1, self._n_lines(v, 17 * s, "sb", rw - 250 * s)) * 23 * s + 8 * s
        # honesty + citation + credits
        fy = H - 150 * s
        L.line([(x0, fy - 16 * s), (W - 56 * s, fy - 16 * s)], _rgba(WHITE, 40), 1 * s)
        yy = self._wrap_text(L, x0, fy, HONESTY, 15.5 * s, "r", GREY, W - 112 * s, 22 * s)
        yy = self._wrap_text(L, x0, yy + 6 * s, CITATION, 14 * s, "r", DIM, W - 112 * s, 20 * s)
        L.text(W - 56 * s, H - 36 * s, CREDITS, 17 * s, "sb", WHITE, anchor="rs")
        img.alpha_composite(L.done())
        return np.asarray(img.convert("RGB"))

    def _mission_rows(self, sm):
        ev = sm.get("events") or []
        last = ev[-1][1] if ev else "?"
        out = []
        if sm["completed"]:
            att = sm.get("dock_attempts")
            out.append(("outcome", "docked" + (f" (attempt {att})" if att and att > 1 else "")))
        else:
            out.append(("outcome", f"not docked (ended in {last})"))
        out.append(("time, path", f"{sm['mission_time_s']:.1f} s, {sm['path_length_m']:.1f} m"))
        fm = sm.get("follow_true_mean_m")
        rms = sm["follow_standoff_err_rms_m"]
        v = (f"true distance mean {fm:.2f} m" if fm is not None else "") + \
            (f"; sonar vs truth RMS {rms:.2f} m" if rms is not None else "")
        out.append(("following the net", v.strip("; ")))
        out.append(("closest link to the net", f"{sm['min_link_to_net_m']:.2f} m"))
        tr = sm["tear"]
        if tr["detected"]:
            how = {"sonar": "sonar gap detector", "sector+sonar": "reported sector +-20 deg, confirmed on sonar",
                   "sector": "reported sector +-20 deg, no sonar fire"}.get(tr["mode"], tr["mode"])
            out.append(("tear", f"at t {tr['t_s']:.1f} s, bearing error {tr['bearing_err_deg']:.1f} deg ({how})"))
        else:
            out.append(("tear", "not detected"))
        pg = sm["per_gait"]
        mass = self._mass(pg)
        e = sm["energy_abs_J"]
        v = f"{e:.0f} J"
        if mass and sm["path_length_m"] > 0:
            v += f"; cost of transport {e / (mass * sm['path_length_m']):.1f} J/(kg m) over the mission"
        out.append(("energy (joint work)", v))
        ig = self.inspect_gait()
        hy = []
        if "cruise" in pg:
            hy.append(f"cruising {pg['cruise']['head_yaw_rms_deg']:.0f} deg")
        if ig and ig in pg and ig != "cruise":
            hy.append(f"inspecting {pg[ig]['head_yaw_rms_deg']:.0f} deg ({PATTERN_LABEL[GAIT_PATTERN[ig]]})")
        if hy:
            out.append(("head yaw RMS, mission", ", ".join(hy) + "; steering included"))
        de = sm["dock_error"]
        out.append(("docking error", "n/a" if not de else f"nose {de['nose_m']:.2f} m, heading {de['heading_deg']:.1f} deg"))
        return out

    def _mission_rows_multi(self, summs):
        """The mission block over several seeds: n/n docked, then min-max ranges; the last row is this
        film's own run (its telemetry/summary), which is a re-run and not bit-identical to its seed."""
        def rng(vals, fmt, unit=""):
            v = [x for x in vals if x is not None]
            if not v:
                return "n/a"
            a, b = fmt.format(min(v)), fmt.format(max(v))
            return (a if a == b else f"{a}-{b}") + unit
        how_of = {"sonar": "sonar gap detector", "sector+sonar": "reported sector +-20 deg, confirmed on sonar",
                  "sector": "reported sector +-20 deg, no sonar fire"}

        def how(mode):
            return how_of.get(mode, mode.replace("sector (", "reported sector +-20 deg (") if mode else "?")
        n = len(summs)
        nd = sum(bool(s["completed"]) for s in summs)
        att = [s.get("dock_attempts") for s in summs]
        out = [("outcome", f"{nd}/{n} docked" + (", each on the first attempt" if all(a == 1 for a in att) else ""))]
        out.append(("time, path", rng([s["mission_time_s"] for s in summs], "{:.0f}", " s") + ", "
                    + rng([s["path_length_m"] for s in summs], "{:.1f}", " m")))
        out.append(("following the net", "true distance mean " + rng([s.get("follow_true_mean_m") for s in summs], "{:.2f}", " m")
                    + "; sonar vs truth RMS " + rng([s["follow_standoff_err_rms_m"] for s in summs], "{:.2f}", " m")))
        out.append(("closest link to the net", rng([s["min_link_to_net_m"] for s in summs], "{:.2f}", " m")))
        modes = [s["tear"]["mode"] for s in summs if s["tear"]["detected"]]
        cnt = ", ".join(f"{how(m)} {modes.count(m)}/{n}" for m in sorted(set(modes), key=lambda m: -modes.count(m)))
        err = [abs(s["tear"]["bearing_err_deg"]) for s in summs if s["tear"].get("bearing_err_deg") is not None]
        out.append(("tear found", f"t {rng([s['tear']['t_s'] for s in summs], '{:.1f}', ' s')}: {cnt}"
                    + (f"; bearing error {rng(err, '{:.1f}', ' deg')}" if err else "")))
        out.append(("closest pass, nose to tear", "pass 1 " + rng([s.get("inspect_closest_head_to_tear_m") for s in summs], "{:.2f}", " m")
                    + ", pass 2 " + rng([s.get("inspect_2_closest_head_to_tear_m") for s in summs], "{:.2f}", " m")))
        out.append(("energy (joint work)", rng([s["energy_abs_J"] for s in summs], "{:.0f}", " J")))
        de = [s["dock_error"] for s in summs if s.get("dock_error")]
        out.append(("latched at", "nose " + rng([d["nose_m"] for d in de], "{:.2f}", " m") + ", heading "
                    + rng([d["heading_deg"] for d in de], "{:.1f}", " deg")))
        sm = self.summ
        tr = sm["tear"]
        film = (("docked" if sm["completed"] else "not docked") + f", {sm['mission_time_s']:.0f} s; tear: "
                + (how(tr["mode"]) if tr["detected"] else "not detected"))
        out.append((f"this film (seed {self.seed}, re-run)", film))
        return out

    @staticmethod
    def _mass(pg):
        """Robot mass from the per-gait rows (energy / (distance x CoT)); None if they disagree."""
        m = [v["energy_J"] / (v["dist_m"] * v["cot_J_per_kg_m"]) for v in pg.values()
             if v.get("dist_m", 0) > 0.2 and v.get("cot_J_per_kg_m", 0) > 0]
        if not m or (max(m) - min(m)) > 0.05 * np.mean(m):
            return None
        return float(np.mean(m))

    def inspect_gait(self):
        """The gait the mission actually used in the INSPECT phases (from the log), or None."""
        tel = self.tel
        ids = [tel.phases.index(p) for p in ("INSPECT", "INSPECT_2") if p in tel.phases]
        m = np.isin(tel.phase_i, ids) & (tel.gait_i >= 0)
        if not m.any():
            return None
        return tel.gaits[int(np.bincount(tel.gait_i[m]).argmax())]

    @staticmethod
    def _lines(s_, px, weight, width):
        words, lines, cur = s_.split(), [], ""
        for wd in words:
            t = (cur + " " + wd).strip()
            if text_width(t, px, weight) <= width or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        return lines

    def _n_lines(self, s_, px, weight, width):
        return len(self._lines(s_, px, weight, width))

    def _wrap_text(self, L, x, y, s_, px, weight, fill, width, lh):
        for ln in self._lines(s_, px, weight, width):
            L.text(x, y, ln, px, weight, fill)
            y += lh
        return y

    def _validation_fig(self, w, h, s, bg):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        fam = ["Segoe UI", "DejaVu Sans"]
        have = {f.name for f in font_manager.fontManager.ttflist}
        fam = [f for f in fam if f in have] or ["DejaVu Sans"]
        hexc = lambda c: "#%02x%02x%02x" % tuple(c[:3])      # noqa: E731
        fg, grid = hexc(GREY), "#2a3d42"
        rc = {"font.family": fam, "font.size": 10.5 * s, "axes.edgecolor": "#3c5358", "axes.labelcolor": fg,
              "xtick.color": fg, "ytick.color": fg, "axes.facecolor": hexc((13, 29, 34)),
              "figure.facecolor": "none", "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6}
        cx, ct = self.sweep["complex"], self.sweep["control"]
        vall = [self.sweep[k]["verdicts"][p] for k in ("complex", "control") for p in ("lateral", "eel")]

        def npass(T):
            return sum(bool(v[T]["passed"]) for v in vall)

        dec = [f"{p}, {'full' if k == 'complex' else 'control'}" for k in ("complex", "control")
               for p in ("lateral", "eel") if self.sweep[k]["verdicts"][p]["T1"].get("declines_35_40")]
        r2 = min(v["T2"]["r2_60_90"] for v in vall)
        am = sorted({int(v["T3"]["argmax_delta_deg"]) for v in vall})
        panels = [
            ("alpha", "speed", "amplitude alpha (deg)", "speed (m/s)",
             "paper: speed rises with alpha, falls past ~30 deg",
             f"ours: rises in {npass('T1')}/4; falls past 35 deg only in " + (", ".join(dec) if dec else "none")),
            ("omega", "speed", "frequency omega (deg/s)", "speed (m/s)",
             "paper: ~linear in omega up to 90 deg/s",
             f"ours: {npass('T2')}/4 pass, R² (60-90 deg/s) >= {r2:.3f}"),
            ("delta", "speed", "phase shift delta (deg)", "speed (m/s)",
             "paper: an interior delta maximises speed",
             f"ours: {npass('T3')}/4 pass, optimum at " + " and ".join(f"{a}" for a in am) + " deg"),
            ("alpha", "power_abs", "amplitude alpha (deg)", "mean |joint power| (W)",
             "paper: power rises with alpha and omega, falls with delta",
             f"ours: {npass('T4')}/4 pass (rank correlation +1, +1, -1)"),
        ]
        with plt.rc_context(rc):
            fig, axs = plt.subplots(2, 2, figsize=(w / 100, h / 100), dpi=100)
            for ax, (vary, yk, xl, yl, t1, t2) in zip(axs.flat, panels):
                for pat, c in (("lateral", ACCENT), ("eel", EEL_C)):
                    for coef, d, ls, mk in (("complex", cx, "-", "o"), ("control", ct, "--", None)):
                        xs, ys = sweep_series(d["rows"], pat, vary, yk)
                        ax.plot(xs, ys, ls, color=hexc(c), lw=(2.2 if coef == "complex" else 1.3) * s,
                                marker=mk, ms=4 * s, alpha=1.0 if coef == "complex" else 0.75)
                ax.set_xlabel(xl, fontsize=10 * s)
                ax.set_ylabel(yl, fontsize=10 * s)
                ax.tick_params(labelsize=9.5 * s)
                ax.set_title(t1 + "\n" + t2, loc="left", fontsize=10.5 * s, color=hexc(WHITE), linespacing=1.4)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            from matplotlib.lines import Line2D
            hs = [Line2D([], [], color=hexc(ACCENT), lw=2.2 * s, marker="o", ms=4 * s),
                  Line2D([], [], color=hexc(EEL_C), lw=2.2 * s, marker="o", ms=4 * s),
                  Line2D([], [], color=fg, lw=2.2 * s), Line2D([], [], color=fg, lw=1.3 * s, ls="--")]
            fig.legend(hs, ["lateral undulation", "eel-like", "paper's full coefficients",
                            "paper's control-oriented set"], loc="lower center", ncol=4, frameon=False,
                       fontsize=10.5 * s, labelcolor=fg)
            fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.2 * s, w_pad=2.0 * s)
            fig.canvas.draw()
            arr = np.asarray(fig.canvas.buffer_rgba()).copy()
            plt.close(fig)
        return Image.fromarray(arr, "RGBA")


# ---- preview --------------------------------------------------------------------------------------------
def _test_frame(W=1920, H=1080):
    """A neutral dark-teal stand-in for a rendered frame, with dummy inset / sonar rects."""
    y = np.linspace(0, 1, H)[:, None, None]
    top, bot = np.array([34, 78, 82], float), np.array([12, 40, 46], float)
    fr = (top * (1 - y) + bot * y).repeat(W, axis=1)
    rng = np.random.default_rng(0)
    fr += rng.normal(0, 2.0, fr.shape)
    fr = fr.clip(0, 255).astype(np.uint8)
    im = Image.fromarray(fr, "RGB")
    d = ImageDraw.Draw(im)
    s = H / 1080
    cam = (int(20 * s), int(H - 20 * s - 320 * s), int(20 * s + 570 * s), int(H - 20 * s))
    son = (int(W - 20 * s - 610 * s), int(H - 20 * s - 460 * s), int(W - 20 * s), int(H - 20 * s))
    for r, lb in ((cam, "head camera inset (3D render)"), (son, "sonar panel (3D render)")):
        d.rectangle(r, fill=(16, 26, 30), outline=(120, 110, 80), width=2)
        d.text((r[0] + 12, r[1] + 10), lb, font=font(18 * s), fill=(150, 150, 150))
    return np.asarray(im), (son[0], son[1], son[2] - son[0], son[3] - son[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--telemetry", default="D:/dev/snake_out/mission/seed0_telemetry.npz")
    ap.add_argument("--summary", default="D:/dev/snake_out/mission/seed0.json")
    ap.add_argument("--sweep", default="D:/dev/snake_out/sweep")
    ap.add_argument("--still", default="D:/dev/snake_out/mission/seed0_follow_wall.png")
    ap.add_argument("--out", default="D:/dev/snake_out/film")
    a = ap.parse_args()
    if not a.preview:
        ap.print_help()
        return
    os.makedirs(a.out, exist_ok=True)
    P = Panels(a.telemetry, a.summary, a.sweep)
    tel = P.tel

    def at(name, off, fallback):
        t0 = tel.phase_start(name)
        return t0 + off if t0 is not None else fallback

    times = [at("UNDOCK", 6.0, 8.0), at("FOLLOW_WALL", 1.5, 25.0), at("TEAR", 0.6, 36.0),
             at("INSPECT_2", 5.0, 50.0), at("CAPTURE", 3.0, float(tel.t[-1]) - 1.0)]
    frame, son = _test_frame()
    paths = []
    for k, t in enumerate(times):
        out = P.compose(frame, t, speedup=3.0 if k in (1, 3) else 1.0, sonar_rect=son)
        p = os.path.join(a.out, f"panels_preview_t{t:05.1f}.png")
        Image.fromarray(out).save(p)
        paths.append(p)
    if os.path.exists(a.still):
        st = np.asarray(Image.open(a.still).convert("RGB"))
        t = at("FOLLOW_WALL", 4.0, 25.0)
        out = P.compose(st, t, speedup=1.0)
        p = os.path.join(a.out, f"panels_preview_still_t{t:05.1f}.png")
        Image.fromarray(out).save(p)
        paths.append(p)
    card = P.end_card((1920, 1080))
    p = os.path.join(a.out, "end_card.png")
    Image.fromarray(card).save(p)
    paths.append(p)
    for p in paths:
        print(p)


if __name__ == "__main__":
    sys.exit(main())
