"""Warp x threepp: GPU simulation in Python, shared with the renderer. About a minute.

    python warp_threepp.py                               # the film (1080p60), lesson_out/warp_threepp.mp4
    python warp_threepp.py --stills 5,20,40              # individual frames
    python warp_threepp.py --sheet                       # contact sheet, one frame per 3 s
    python warp_threepp.py --preview --out preview.mp4   # 960x540 @ 30 fps

Every moving picture is a real program's output. The cloth, the sparks and the event camera come
from ../warp_round_trip.py, run by the film in a subprocess (`--stream`) that steps its
simulation at 60 Hz on the film's clock and pipes the frames in; its code on screen is read from
that file (the `# [name]` markers). The other shots are clips the Warp examples render themselves
(`ensure_clips`, cached in lesson_out/warp_clips): the fluid (the opening), the blast, the gummy
candies and the jelly wall. The closing reel cuts on the narrator's words (Kokoro's own alignment),
and the film ends on a wall of every shot at once, under the title. Every example shown shares its simulation's buffers with the renderer
(vertex, particle-field or frame interop) rather than copying through the CPU.
"""
from __future__ import annotations

import atexit
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import lesson  # noqa: E402
from lesson import (DIM, TEXT, Hud, Stage, Timeline, envelope, fade_in_out, remap, run, smooth, tp)  # noqa: E402
import threepp_intro as intro  # noqa: E402  (code cards and syntax colours)
from threepp_intro import card, code_block  # noqa: E402

FPS = 60
EXAMPLES = os.path.dirname(_HERE)
HERO = os.path.join(EXAMPLES, "warp_round_trip.py")
CLIPS = os.path.join("lesson_out", "warp_clips")

C_HOT = 0xffb347             # the sparks
C_COOL = 0x4cc9f0
C_RED = 0xff5a5f
BG = Stage.BG

intro.KW["py"] |= {"not", "and", "or", "elif", "while", "self"}
lesson.SPOKEN_WORDS.update({"CPU": "C P U"})

# ── the script ────────────────────────────────────────────────────────────────
CAPTION_TEXT = {
    "o1": "Simulated in Python by NVIDIA Warp. Drawn by threepp. And the data never leaves the GPU.",
    "k1": "Warp turns a Python function into a GPU kernel: one thread for every particle, all running at once.",
    "m1": "No trip to the CPU and back: Warp writes the cloth and the sparks straight into buffers it shares "
          "with the renderer.",
    "m2": "So the ray tracer sees every fold in the same frame.",
    "n1": "With no copies in the way, it scales. This blast is millions of particles.",
    "f1": "And back the other way: the finished frame goes straight to a Warp kernel, which turns it into an "
          "event camera.",
    "e1": "Cloth, sparks, jelly, water, fire. Python on the GPU, drawn by threepp.",
}
SPOKEN_ONLY = {"o1"}                         # read over the cold open and the title, not drawn
SPEECH = {"o1": 7.30, "k1": 7.05, "m1": 7.22, "m2": 3.45, "n1": 5.17, "f1": 6.72,
          "e1": 6.20}                        # seconds, af_heart, measured
# when each word of the last line is said, seconds into its clip (af_heart, measured); setup()
# reads Kokoro's own alignment instead when it can, so the reel cuts land on the words
E1_WORDS = {"Cloth": 0.28, "sparks": 0.71, "jelly": 1.20, "water": 1.57, "fire": 1.95, "Python": 2.79,
            "drawn": 4.25}
PLAN = [("open", 0.4, ["o1"], 0.4), ("kernel", 0.2, ["k1"], 0.6), ("share", 0.2, ["m1", "m2"], 0.7),
        ("scale", 0.1, ["n1"], 0.4), ("out", 0.2, ["f1"], 0.9), ("reel", 0.1, ["e1"], 0.1), ("end", 0.0, [], 2.6)]
SLACK = lesson.VOICE_LEAD + lesson.VOICE_TAIL + 0.1


def _layout():
    tl, cap, t = Timeline(), {}, 0.0
    for name, lead, keys, tail in PLAN:
        c = t + lead
        for k in keys:
            cap[k] = (c, c + SPEECH[k] + SLACK)
            c = cap[k][1] + 0.1
        end = (c - 0.1 if keys else t) + tail
        tl.add(name, t, end)
        t = end
    return tl, cap


TL, CAP = _layout()


def cap0(k):
    return CAP[k][0]


def said(word, words=None):
    """Film time at which a word of the last line is heard."""
    return cap0("e1") + lesson.VOICE_LEAD + (words or E1_WORDS)[word]


def e1_words():
    """The last line's word onsets, from Kokoro's alignment (the table above without it)."""
    try:
        got = {}
        for w, a, _ in lesson.Narration().word_times(CAPTION_TEXT["e1"]):
            got.setdefault(w, a)
        out = {w: got[w] for w in E1_WORDS}
    except (RuntimeError, KeyError) as e:
        print(f"[reel] no word alignment ({e}); cutting on the measured table")
        return dict(E1_WORDS)
    off = max(abs(out[w] - E1_WORDS[w]) for w in E1_WORDS)
    if off > 0.05:
        print(f"[reel] the spoken words moved by up to {off:.2f} s from the table: {out}")
    return out


# ── the clips the examples render themselves ──────────────────────────────────
# (name, file, fps, the example's command line; run from CLIPS). Rendered once, then cached.
CLIP_SPECS = {
    "jelly": ("jelly.mp4", 60, ["warp_jelly_wreck.py", "--video", "8", "--no-sensors", "--clean",
                                "--out-dir", ".", "--tag", "jelly"]),
    "gummy": ("gummy.mp4", 60, ["warp_gummy_rain.py", "--video", "11", "--clean", "--out", ".", "--tag", "gummy"]),
    "fluid": ("warp_fluid.mp4", 60, ["warp_fluid.py", "--vulkan", "--video", "11", "--size", "1920x1080"]),
    "blast": ("blast.mp4", 60, ["warp_explosion.py", "--video", "11", "--size", "1920x1080", "--out", "blast.mp4"]),
}


def ensure_clips():
    os.makedirs(CLIPS, exist_ok=True)
    for name, (fname, _, cmd) in CLIP_SPECS.items():
        if os.path.isfile(os.path.join(CLIPS, fname)):
            continue
        print(f"[clips] rendering {name}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run([sys.executable, os.path.join(EXAMPLES, cmd[0])] + cmd[1:], cwd=CLIPS).returncode
        if not os.path.isfile(os.path.join(CLIPS, fname)):
            raise RuntimeError(f"{cmd[0]} exited {rc} without writing {fname}")


class Clip:
    """A clip decoded by ffmpeg at the film's size. Frames come in order; asking for an
    earlier one restarts the decoder there."""

    def __init__(self, name, w, h):
        fname, self.fps, _ = CLIP_SPECS[name]
        self.path = os.path.join(CLIPS, fname)
        self.w, self.h = w, h
        self.proc, self.k, self.cur, self.k0 = None, -1, None, 0
        atexit.register(self.close)

    def _start(self, k):
        self.close()
        self.proc = subprocess.Popen([lesson.ffmpeg_exe(), "-loglevel", "error", "-ss", f"{k / self.fps:.4f}",
                                      "-i", self.path, "-f", "rawvideo", "-pix_fmt", "rgb24",
                                      "-s", f"{self.w}x{self.h}", "-"], stdout=subprocess.PIPE, bufsize=0)
        self.k0, self.k = k, k - 1

    def at(self, ts):
        """The frame at ts seconds into the clip (held at the last frame past its end)."""
        k = max(int(math.floor(ts * self.fps + 1e-6)), 0)
        if self.proc is None or k <= self.k - 1 or k > self.k + 3 * self.fps:
            if self.cur is None or k != self.k:
                self._start(k)
        size = self.w * self.h * 3
        while self.k < k:
            data = bytearray()
            while len(data) < size:
                chunk = self.proc.stdout.read(size - len(data))
                if not chunk:
                    return self.cur                         # past the end: hold
                data += chunk
            self.k += 1
            self.cur = np.frombuffer(bytes(data), np.uint8).reshape(self.h, self.w, 3)
        return self.cur

    def close(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self.proc = None


class Hero:
    """warp_round_trip.py in a subprocess, from film time t0 to t1 at 60 Hz: the render and the
    events per frame. `preroll` seconds of simulation run before the first frame."""

    def __init__(self, w, h, t0, t1, preroll, dots=False):
        self.w, self.h, self.t0, self.t1, self.preroll, self.dots = w, h, t0, t1, preroll, dots
        self.proc, self.k, self.cur, self.n = None, -1, None, 0
        atexit.register(self.close)

    def _start(self):
        self.close()
        cmd = [sys.executable, HERO, "--stream", f"{self.w}x{self.h}", "--seconds", f"{self.t1 - self.t0:.3f}",
               "--preroll", f"{self.preroll:.3f}"] + (["--dots"] if self.dots else [])
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        line = b""
        while not line.endswith(b"\n"):
            ch = self.proc.stdout.read(1)
            if not ch:
                raise RuntimeError("warp_round_trip.py ended before its first frame")
            line += ch
        self.n = int(line.split()[1])
        self.k = -1

    def at(self, t):
        """(render, events) at film time t."""
        k = min(max(int(round((t - self.t0) * FPS)), 0), max(self.n - 1, 0))
        if self.proc is None or k < self.k:
            self._start()
            k = min(k, self.n - 1)
        size = self.w * self.h * 3
        while self.k < k:
            data = bytearray()
            while len(data) < 2 * size:
                chunk = self.proc.stdout.read(2 * size - len(data))
                if not chunk:
                    raise RuntimeError(f"warp_round_trip.py ended at frame {self.k + 1}")
                data += chunk
            self.k += 1
            a = np.frombuffer(bytes(data), np.uint8)
            self.cur = (a[:size].reshape(self.h, self.w, 3), a[size:].reshape(self.h, self.w, 3))
        return self.cur

    def close(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self.proc = None


def quoted(name):
    """The lines of warp_round_trip.py between `# [name]` and `# [/name]`, dedented."""
    with open(HERO, encoding="utf-8") as f:
        src = f.read().splitlines()
    i = next(k for k, l in enumerate(src) if l.strip() == f"# [{name}]")
    j = next(k for k, l in enumerate(src) if l.strip() == f"# [/{name}]")
    lines = src[i + 1:j]
    pad = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
    return [l[pad:].rstrip() for l in lines]


def strip_comment(line):
    """A code line without its trailing comment (a card's width is the scarce thing)."""
    return re.sub(r"\s+#.*$", "", line) if not line.lstrip().startswith("#") else line


# ── the picture ───────────────────────────────────────────────────────────────
FULL = (0, 0, 1920, 1080)


class Slot:
    """A texture that shows a new numpy image every frame."""

    def __init__(self):
        self.tex, self.shape = None, None

    def texture(self, img):
        img = np.ascontiguousarray(img[::-1])            # textures take their rows bottom first
        if self.tex is None or self.shape != img.shape:
            self.tex, self.shape = tp.data_texture(img, True), img.shape
        else:
            self.tex.update_data(img)
        return self.tex


def dim(img, k):
    return (img.astype(np.float32) * k).astype(np.uint8) if k < 0.999 else img


class Painter:
    def __init__(self, ov, clips, heroes, code, sw, sh, words):
        self.ov, self.clips, self.heroes, self.code = ov, clips, heroes, code
        self.sw, self.sh = sw, sh
        # the reel cuts a hair before each word is heard; the ending starts on "Python"
        self.cuts = [TL.start("reel")] + [said(w, words) - 0.06 for *_, w in self.REEL[1:]]
        self.end0 = said("Python", words) - 0.06
        self.title0 = said("drawn", words)
        self.slots = {}
        self.captions = [(a, b, CAPTION_TEXT[k]) + ((False,) if k in SPOKEN_ONLY else ())
                         for k, (a, b) in CAP.items()]

    # helpers ------------------------------------------------------------------
    def show(self, slot, img, a=1.0, rect=FULL):
        if a <= 0.003 or img is None:
            return
        self.ov.image(*rect, self.slots.setdefault(slot, Slot()).texture(img), alpha=a)

    def pill(self, text, a, x=60, y=60, color=TEXT, sub=None, anchor="l", size=24):
        if a <= 0.003:
            return
        ov = self.ov
        tw = ov.text_width(text, size, "semibold") + 44
        if sub:
            tw = max(tw, ov.text_width(sub, 19) + 44)
        ph = 50 if not sub else 82
        px = x if anchor == "l" else x - tw
        ov.panel(px, y, tw, ph, radius=14, fill=0x0d131e, alpha=0.8 * a)
        ov.text(px + 22, y + 25, text, size=size, color=color, alpha=a, kind="semibold", anchor="lm")
        if sub:
            ov.text(px + 22, y + 60, sub, size=19, color=DIM, alpha=a, anchor="lm")

    def code_card(self, t, t_in, t_out, x, y, lines, title, size=19, lh=26, max_w=None):
        a = envelope(t, t_in, t_out, 0.4, 0.4)
        if a <= 0.003:
            return
        ov = self.ov
        cw = ov.text_width("0", size, "mono")
        w = max(len(l) for l in lines) * cw + 60
        if max_w and w > max_w:
            return self.code_card(t, t_in, t_out, x, y, lines, title, size - 1, lh - 1, max_w)
        card(ov, x, y, w, 60 + lh * len(lines), a, title)
        code_block(ov, x + 30, y + 46, lines, "py", size, lh, a,
                   reveal=lambda j: remap(t, t_in + 0.15 + 0.12 * j, t_in + 0.45 + 0.12 * j))

    def flash(self, t, t_cut, k=0.55, dur=0.22):
        u = (t - t_cut) / dur
        if 0.0 <= u < 1.0:
            self.ov.fade(k * (1 - u) ** 2, color=0xffffff)

    # beats ----------------------------------------------------------------------
    def draw(self, t):
        for beat in (self.opening, self.kernel, self.share, self.scale, self.out, self.reel, self.end):
            beat(t)
        self.ov.captions(t, self.captions)

    def opening(self, t):
        t1 = TL.end("open")
        if t > t1 + 0.3:
            return
        self.show("fluid", self.clips["fluid"].at(t), envelope(t, -1.0, t1 + 0.3, 0.1, 0.3))
        # the title comes in once the first wave has broken over the block
        ov = self.ov
        ti = 2.4
        a = envelope(t, ti, t1 - 0.1, 0.25, 0.4)
        if a <= 0.003:
            return
        y0 = 700                                            # below the pool, over the loungers
        ov.panel(-20, y0 - 50, 1960, 330, radius=0, fill=0x05070b, alpha=0.78 * a)
        rise = 22 * (1 - lesson.ease_out(remap(t, ti, ti + 0.5)))
        x0 = 150
        ov.text(x0 + 4, y0 + rise, "GPU SIMULATION IN PYTHON", size=24, color=C_HOT, alpha=a, kind="semibold",
                tracking=5)
        w = ov.rich(x0, y0 + 150 + rise, [("Warp", TEXT, "bold"), (" × ", C_HOT, "bold"), ("threepp", TEXT, "bold")],
                    size=128, alpha=a)
        bar = smooth(remap(t, ti + 0.15, ti + 0.6))
        ov.panel(x0 + 4, y0 + 178 + rise, w * bar, 7, radius=3, fill=C_HOT, alpha=a)
        sa = a * smooth(remap(t, ti + 0.5, ti + 1.0))
        ov.text(x0 + 4, y0 + 230 + rise, "Simulated, shared and drawn on the GPU, with no copy in between", size=32,
                color=DIM, alpha=sa, kind="light")

    def kernel(self, t):
        s, e = TL.start("kernel"), TL.end("kernel")
        if not s - 0.3 <= t <= e + 0.9:
            return
        dots = self.heroes["dots"].at(t)[0] if t <= e + 0.8 else None
        self.show("dots", dots, envelope(t, s - 0.3, e + 0.8, 0.3, 0.8))
        self.flash(t, s)
        lines = self.code["kernel"]
        self.code_card(t, s + 0.3, e - 0.2, 56, 60, lines, "THE CLOTH'S FIRST KERNEL, IN PYTHON", max_w=760)
        self.pill("One dot, one particle, one thread", envelope(t, cap0("k1") + 3.0, e - 0.2, 0.4, 0.4),
                  x=1864, y=60, anchor="r", color=C_HOT)

    def share(self, t):
        s, e = TL.start("share"), TL.end("share")
        if not TL.end("kernel") - 0.9 <= t <= e + 0.3:
            return
        rgb = self.heroes["a"].at(t)[0]
        m2 = cap0("m2")
        k = 1.0 - 0.62 * envelope(t, s + 0.2, m2 - 0.1, 0.5, 0.5)            # dimmed under the diagram
        self.show("hero_a", dim(rgb, k), envelope(t, TL.end("kernel") - 0.9, e + 0.3, 0.8, 0.3))
        self.diagram(t, s + 0.2, m2 - 0.1)
        self.code_card(t, s + 3.2, m2 - 0.1, 56, 652, [strip_comment(l) for l in self.code["in"]],
                       "THE CLOTH AND THE SPARKS, SHARED", size=18, lh=25)
        self.pill("Ray traced: the reflections and shadows follow every fold", envelope(t, m2 + 0.2, e, 0.4, 0.4),
                  x=60, y=60, color=C_COOL)

    def diagram(self, t, t0, t1):
        """The usual trip through the CPU, struck out, then one buffer the two share."""
        a = envelope(t, t0, t1, 0.4, 0.4)
        if a <= 0.003:
            return
        ov = self.ov
        gx, gy, gw, gh = 360, 250, 1200, 380
        ov.panel(gx, gy, gw, gh, radius=26, fill=0x0d131e, alpha=0.72 * a, outline=0x8aa0c0, outline_alpha=0.25)
        ov.text(gx + 30, gy + 30, "GPU", size=20, color=DIM, alpha=a, kind="semibold", tracking=4)
        bw, bh, by = 330, 130, gy + 190
        lx, rx = gx + 80, gx + gw - 80 - bw
        for x, title, sub, c in ((lx, "Warp", "kernels", C_HOT), (rx, "threepp", "Vulkan renderer", C_COOL)):
            ov.panel(x, by, bw, bh, radius=18, fill=0x16202f, alpha=0.95 * a, outline=c, outline_alpha=0.8)
            ov.text(x + bw / 2, by + 50, title, size=40, color=c, alpha=a, kind="bold", anchor="mm")
            ov.text(x + bw / 2, by + 96, sub, size=22, color=DIM, alpha=a, anchor="mm")
        # the usual way: up to the CPU and back down
        cx, cy, cw_, ch_ = gx + gw / 2 - 130, 60, 260, 110
        u1 = smooth(remap(t, t0 + 0.2, t0 + 0.8))
        strike = smooth(remap(t, t0 + 2.4, t0 + 2.9))
        old = a * u1 * (1 - 0.65 * strike)
        ov.panel(cx, cy, cw_, ch_, radius=18, fill=0x1b1f27, alpha=0.95 * old, outline=0x8aa0c0, outline_alpha=0.5)
        ov.text(cx + cw_ / 2, cy + ch_ / 2, "CPU", size=40, color=TEXT, alpha=old, kind="bold", anchor="mm")
        up = [(lx + bw / 2, by), (lx + bw / 2, cy + ch_ / 2), (cx, cy + ch_ / 2)]
        down = [(cx + cw_, cy + ch_ / 2), (rx + bw / 2, cy + ch_ / 2), (rx + bw / 2, by)]
        col = C_RED if strike > 0.5 else DIM
        for path in (up, down):
            for p0, p1 in zip(path[:-1], path[1:]):
                ov.dashed(p0, p1, col, 4, old, dash=14, gap=10, phase=-60 * (t - t0))
        ov.arrow2d(down[-2], down[-1], col, 4, 16, old)
        if strike > 0.01:
            m = (cx + cw_ / 2, cy + ch_ / 2)
            r = 90 * strike
            ov.line([(m[0] - r, m[1] - r * 0.6), (m[0] + r, m[1] + r * 0.6)], C_RED, 9, a * strike)
            ov.line([(m[0] - r, m[1] + r * 0.6), (m[0] + r, m[1] - r * 0.6)], C_RED, 9, a * strike)
        # instead: one buffer, both sides
        u2 = smooth(remap(t, t0 + 2.8, t0 + 3.5))
        if u2 > 0.003:
            x0, x1 = lx + bw, rx
            yb = by + bh / 2
            fill = x0 + (x1 - x0) * u2
            ov.panel(x0, yb - 26, fill - x0, 52, radius=12, fill=C_HOT, alpha=0.9 * a)
            ov.text((x0 + x1) / 2, yb, "shared buffers", size=24, color=0x10131a, alpha=a * u2, kind="bold",
                    anchor="mm")
            pulse = (t * 1.6) % 1.0
            ov.circle(x0 + (x1 - x0) * pulse, yb - 40, 7, fill=C_HOT, alpha=a * u2 * math.sin(math.pi * pulse))
            ov.text((x0 + x1) / 2, yb + 56, "cloth vertices · spark particles", size=20, color=DIM,
                    alpha=a * u2, anchor="mm")

    def scale(self, t):
        s, e = TL.start("scale"), TL.end("scale")
        if not s - 0.1 <= t <= e + 0.3:
            return
        self.show("blast", self.clips["blast"].at(t - s + 0.6), envelope(t, s - 0.05, e + 0.3, 0.05, 0.3))
        self.flash(t, s, 0.7)
        self.pill("Warp gas and fire, PhysX bricks, one renderer", envelope(t, s + 0.6, e, 0.3, 0.3),
                  x=60, y=60, color=C_HOT)

    def out(self, t):
        s, e = TL.start("out"), TL.end("out")
        if not s - 0.3 <= t <= e + 0.5:
            return
        rgb, ev = self.heroes["b"].at(t)
        # a wipe from the render to what the Warp kernel made of it
        u = 1.0 - 0.45 * smooth(remap(t, s + 1.2, s + 2.4)) - 0.4 * smooth(remap(t, s + 4.6, s + 5.8))
        x = int(round(u * self.sw))
        img = rgb.copy()
        if x < self.sw:
            img[:, x:] = ev[:, x:]
        self.show("hero_b", img, envelope(t, s - 0.3, e + 0.5, 0.3, 0.5))
        a = envelope(t, s + 1.2, e, 0.4, 0.4)
        if 0.0 < u < 1.0 and a > 0.003:
            px = u * 1920
            self.ov.line([(px, 0), (px, 1080)], 0xffffff, 3.0, 0.9 * a)
        self.pill("Drawn by threepp", a * smooth(remap(u, 0.2, 0.35)), x=40, y=60)
        self.pill("Read in place by a Warp kernel", a, x=1880, y=60, anchor="r", color=C_COOL,
                  sub="an event camera: pixels fire when the brightness changes")
        lines = self.code["out"] + [strip_comment(l) for l in self.code["see"] if "launch(events" in l]
        self.code_card(t, s + 2.4, e - 0.1, 40, 134, lines, "THE FRAME, BACK TO WARP", size=17, lh=24)

    # (source, seconds into the clip at the cut, the word on screen, the word as it is said)
    REEL = [("hero", None, "CLOTH", "Cloth"), ("hero", None, "SPARKS", "sparks"), ("gummy", 1.0, "JELLY", "jelly"),
            ("fluid", 6.0, "WATER", "water"), ("blast", 1.2, "FIRE", "fire")]

    def clip_at(self, k, t):
        """Reel shot k at film time t: the clip running on from its cut."""
        src, off, *_ = self.REEL[k]
        return self.clips[src].at(off + t - self.cuts[k])

    def reel(self, t):
        s = TL.start("reel")
        if not s - 0.1 <= t < self.end0:
            return
        k = max(i for i, c in enumerate(self.cuts) if c <= t) if t >= s else 0
        img = self.heroes["c"].at(t)[0] if self.REEL[k][0] == "hero" else self.clip_at(k, t)
        self.show("reel", img, envelope(t, s - 0.05, self.end0 + 1, 0.05, 0.1))
        if k > 0:
            self.flash(t, self.cuts[k], 0.25, 0.12)
        nxt = self.cuts[k + 1] if k + 1 < len(self.cuts) else self.end0
        word = self.REEL[k][2]
        a = envelope(t, self.cuts[k], nxt, 0.06, 0.06)
        self.ov.text(84, 155, word, size=110, color=0x000000, alpha=0.55 * a, kind="bold", tracking=6)
        self.ov.text(80, 150, word, size=110, color=C_HOT if word == "SPARKS" else TEXT, alpha=a, kind="bold",
                     tracking=6)

    # the ending: every shot at once, the fire shrinking into its place, then the title over them
    TW, TH, GAP = 600, 338, 24
    TILE_X0, TILE_Y0 = (1920 - 3 * 600 - 2 * 24) // 2, (1080 - 2 * 338 - 24) // 2

    def tile(self, i):
        c, r = i % 3, i // 3
        return (self.TILE_X0 + c * (self.TW + self.GAP), self.TILE_Y0 + r * (self.TH + self.GAP), self.TW, self.TH)

    def end(self, t):
        s = self.end0
        if t < s:
            return
        ov = self.ov
        ov.panel(-20, -20, 1960, 1120, radius=0, fill=0x05070b, alpha=smooth(remap(t, s, s + 0.4)))
        rgb, ev = self.heroes["c"].at(t)
        # (the fire runs on from the reel; the others start afresh, where each clip has enough left)
        shots = [("t_cloth", rgb), ("t_events", ev), ("t_jelly", self.clips["jelly"].at(0.5 + t - s)),
                 ("t_gummy", self.clips["gummy"].at(2.0 + t - s)), ("t_water", self.clips["fluid"].at(1.0 + t - s))]
        for i, (slot, img) in enumerate(shots):
            u = smooth(remap(t, s + 0.15 + 0.08 * i, s + 0.55 + 0.08 * i))
            x, y, w, h = self.tile(i)
            g = 0.9 + 0.1 * u                                   # each settles into place as it appears
            self.show(slot, img, u, (x + w * (1 - g) / 2, y + h * (1 - g) / 2, w * g, h * g))
        u = smooth(remap(t, s, s + 0.7))
        rect = tuple(a + (b - a) * u for a, b in zip(FULL, self.tile(5)))
        self.show("reel", self.clip_at(4, t), 1.0, rect)
        # the title, over the wall of shots, as "drawn by threepp" is said
        ti = self.title0
        a = smooth(remap(t, ti, ti + 0.5))
        if a <= 0.003:
            return
        ov.panel(-20, -20, 1960, 1120, radius=0, fill=0x05070b, alpha=0.45 * a)
        ov.panel(960 - 480, 385, 960, 310, radius=26, fill=0x0d131e, alpha=0.9 * a, outline=0x8aa0c0,
                 outline_alpha=0.25 * a)
        rise = 16 * (1 - lesson.ease_out(remap(t, ti, ti + 0.6)))
        w = ov.rich(960, 500 + rise, [("Warp", TEXT, "bold"), (" × ", C_HOT, "bold"), ("threepp", TEXT, "bold")],
                    size=112, alpha=a, anchor="ms")
        bar = smooth(remap(t, ti + 0.2, ti + 0.7))
        ov.panel(960 - w * bar / 2, 526 + rise, w * bar, 6, radius=3, fill=C_HOT, alpha=a)
        la = a * smooth(remap(t, ti + 0.6, ti + 1.2))
        ov.text(960, 590, "python/examples/warp_round_trip.py", size=30, color=DIM, alpha=la, kind="mono",
                anchor="mm")
        ov.text(960, 650, "github.com/markaren/threepp", size=30, color=C_COOL, alpha=la, anchor="mm")


def setup(width, height):
    t0 = time.time()
    ensure_clips()
    sw, sh = width, height
    clips = {name: Clip(name, sw, sh) for name in CLIP_SPECS}
    ks, ke = TL.start("kernel"), TL.end("kernel")
    heroes = {
        # the dotted cloth and the plain one start together, so they cross-fade on the same motion
        "dots": Hero(sw, sh, ks - 0.3, ke + 0.8, preroll=2.0, dots=True),
        "a": Hero(sw, sh, ke - 0.9, TL.end("share") + 0.3, preroll=2.0 + (ke - 0.9) - (ks - 0.3)),
        "b": Hero(sw, sh, TL.start("out") - 0.3, TL.end("out") + 0.5, preroll=9.0),
        "c": Hero(sw, sh, TL.start("reel") - 0.1, TL.duration + 0.1, preroll=5.0),     # the reel and the ending
    }
    code = {name: quoted(name) for name in ("kernel", "in", "out", "see")}
    st = Stage(width, height, renderer="gl", floor=False, fog=None)
    ov = Hud(1920, 1080)
    painter = Painter(ov, clips, heroes, code, sw, sh, e1_words())
    print(f"[setup] {time.time() - t0:.1f}s, film {TL.duration:.1f} s on the script clock, "
          + ", ".join(f"{n} {TL.start(n):.1f}-{TL.end(n):.1f}" for n, *_ in PLAN))

    def render(t):
        ov.begin()
        painter.draw(t)
        ov.fade(fade_in_out(t, TL.duration, fade_in=0.15, fade_out=0.6))
        ov.end()
        for hero in heroes.values():                      # a stream whose part of the film is over can go
            if hero.proc is not None and t > hero.t1 + 0.5:
                hero.close()
        return st.frame(t, hud=ov)
    return render, painter.captions


if __name__ == "__main__":
    run("warp_threepp", TL.duration, setup, fps=FPS)
