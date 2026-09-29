"""Your first real app.  Part 0b: hello.cpp grows into an app in eight steps, about 3 min.

    python first_app.py                               # the film (1080p60), lesson_out/first_app.mp4
    python first_app.py --stills 30,60,95             # individual frames
    python first_app.py --sheet                       # contact sheet, one frame per 3 s
    python first_app.py --preview --out preview.mp4   # 960x540 @ 30 fps

The program is examples/lesson/app.cpp. It starts as Part 0's hello.cpp and grows one
step at a time (orbit controls, a glTF model, an HDR sky, animation, shadows, instancing,
picking, a Dear ImGui panel), each step marked `#if STEP >= n` in the source.

What is on screen was checked against the program:

* the code shown for a step is read out of app.cpp: the lines a step adds (and removes)
  are those that appear (and disappear) when the preprocessor runs with STEP = n instead
  of n - 1, so every snippet is part of a program that compiles;
* the window is that program's own output. Each step is built on its own
  (lesson_app_capture_<n>, CMake targets beside lesson_app) and run while the film is
  made: headless, on a fixed 1/60 s clock, with the mouse replayed from a script through
  the canvas's own event path (examples/lesson/capture.hpp). The pointer drawn over the
  window marks where that script puts the mouse; the program itself draws none;
* the draw calls and triangles are what the renderer counted for that frame
  (GLRenderer::info()).
"""
from __future__ import annotations

import atexit
import glob
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import lesson  # noqa: E402
from lesson import (DIM, TEXT, Hud, Stage, clamp01, ease_out, envelope, fade_in_out, remap,  # noqa: E402
                    run, smooth, tp)
import threepp_intro as intro  # noqa: E402  (the code cards and the syntax colours)
from threepp_intro import card, tokenize  # noqa: E402

FPS = 60
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
APP_CPP = os.path.join(REPO, "examples", "lesson", "app.cpp")

C_ACCENT = 0x4cc9f0
C_WARM = 0xffb347
C_OK = 0x5ee27a
C_DEL = 0xef476f
C_OUT = 0xa6e3a1

intro.KW["cpp"] |= {"for", "if", "else", "bool", "true", "false", "nullptr", "static", "void"}

# how the narrator says the names on screen: Part 0's table, then this film's own
WORDS = {**intro.WORDS, "glTF": "G L T F", "HDR": "H D R", "C++": "C plus plus", "ImGui": "Im Gooey",
         "OrbitControls": "Orbit Controls", "IOCapture": "I O Capture",
         "examples/lesson/app.cpp": "examples, lesson, app dot C P P", "hello.cpp": "hello dot C P P",
         "STEP": "step"}

# ── the script ────────────────────────────────────────────────────────────────
# As in Part 0, the narration sets the pace: each caption lasts as long as its spoken line
# (Kokoro, af_heart, measured in first_app.speech.json) plus the voice's lead-in and tail and
# a small margin.
CAPTION_TEXT = {
    "r1": "Part 0 ended with hello.cpp: a box that turns. Here it grows into a real app, one step at a time.",
    "r2": "Each step is a program that builds and runs, and the window shows what that program draws.",
    "o1": "Step one: let the user move the camera. OrbitControls listens to the mouse. Drag to turn, scroll to zoom.",
    "o2": "And when the window changes size, the camera's aspect and the renderer's size follow, so nothing stretches.",
    "m1": "Step two: a real model instead of the box. The glTF loader turns the file into the same kind of tree: "
          "meshes, materials and bones.",
    "m2": "Lit by one light from one side, the rest of him stays black. Real light comes from all around.",
    "k1": "Step three: light from a photo. One HDR image of a park becomes the background, and what every material "
          "reflects.",
    "k2": "Tone mapping then fits the bright sky into what a screen can show.",
    "a1": "Step four: motion. The file also holds animations, and a mixer plays the walk on the model's skeleton.",
    "a2": "Each frame, it moves on by the time since the last frame, so he walks at the same pace on any computer.",
    "h1": "Step five: shadows. They are off until you ask: turn them on in the renderer, then choose which light and "
          "objects cast them, and what receives them.",
    "h2": "The light sees the scene through a camera of its own. Only what is inside it gets a shadow, so size it to "
          "your scene.",
    "n1": "Step six: two thousand stones. As one mesh each, that is about {sep} draw calls every frame, one for every stone "
          "in view.",
    "n2": "An instanced mesh draws them together: one geometry, one material, and a matrix and a colour per stone. "
          "Now the frame takes {inst} draw calls.",
    "p1": "Step seven: point at things. The raycaster sends a ray from the camera, through the mouse, into the scene.",
    "p2": "It returns what the ray hits, nearest first. On an instanced mesh it also says which instance, so one stone "
          "can light up.",
    "u1": "Step eight: knobs. Dear ImGui builds a panel in a few lines, and its sliders write straight into your "
          "variables.",
    "u2": "IOCapture keeps the camera still while the mouse is busy with the panel.",
    "c1": "That is the whole app, {lines} lines of C++. It is in the repository as examples/lesson/app.cpp.",
    "c2": "Build it with STEP set to any number from zero to eight, and you get the program as it stood after that "
          "step.",
}
SPEECH = lesson.Speech(CAPTION_TEXT, os.path.join(_HERE, "first_app.speech.json"), words=WORDS)
PLAN = [("open", 0.0, [], 6.0), ("recap", 0.4, ["r1", "r2"], 0.8),
        ("s1", 0.4, ["o1", "o2"], 0.8), ("s2", 0.4, ["m1", "m2"], 0.8), ("s3", 0.4, ["k1", "k2"], 1.4),
        ("s4", 0.4, ["a1", "a2"], 0.8), ("s5", 0.4, ["h1", "h2"], 0.8), ("s6", 0.4, ["n1", "n2"], 1.0),
        ("s7", 0.4, ["p1", "p2"], 1.0), ("s8", 0.4, ["u1", "u2"], 1.6), ("end", 0.4, ["c1", "c2"], 0.6),
        ("outro", 0.0, [], 8.0)]
TL, CAP = SPEECH.layout(PLAN)


def cap0(k):
    return CAP[k][0]


STEP_NAMES = ["hello", "Orbit", "Model", "Sky", "Motion", "Shadows", "Many", "Pick", "UI"]
STEP_TITLES = ["hello.cpp, from Part 0", "a camera you can move", "a real model", "light from a photo", "motion",
               "shadows", "many things, one draw call", "point at things", "knobs"]
BEAT = {0: "recap", 1: "s1", 2: "s2", 3: "s3", 4: "s4", 5: "s5", 6: "s6", 7: "s7", 8: "s8"}

SUMMARY = [(r"$\mathrm{input}$", "OrbitControls and a Raycaster turn the mouse into actions."),
           (r"$\mathrm{assets}$", "glTF models and HDR skies load into the same scene tree."),
           (r"$\mathrm{time}$", "Animations advance by each frame's delta, not by frames."),
           (r"$\mathrm{instancing}$", "Thousands of copies, one draw call.")]

BUILD_CMD = ["cmake -B build -DLESSON_APP_STEP=3", "cmake --build build --target lesson_app"]


# ── app.cpp, step by step ─────────────────────────────────────────────────────
def _cond(directive, step, defines):
    """The value of one #if / #ifdef / #ifndef line (STEP >= k and STEP < k are all app.cpp uses)."""
    d = directive.strip()
    m = re.match(r"#(?:el)?if\s+STEP\s*(>=|<)\s*(\d+)$", d)
    if m:
        k = int(m.group(2))
        return step >= k if m.group(1) == ">=" else step < k
    m = re.match(r"#if(n?)def\s+(\w+)$", d)
    if m:
        on = m.group(2) in defines
        return not on if m.group(1) else on
    raise ValueError(f"app.cpp: a directive the film cannot read: {d}")


def preprocess(lines, step, defines=("STEP",)):
    """Indices of the source lines the preprocessor keeps for STEP = step (directives dropped)."""
    keep, stack, active = [], [], True
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("#if ", "#ifdef ", "#ifndef ")):
            c = _cond(s, step, defines)
            stack.append([active, c])
            active = active and c
        elif s.startswith("#elif"):
            parent, taken = stack[-1]
            c = _cond(s, step, defines)
            active = parent and not taken and c
            stack[-1][1] = taken or c
        elif s.startswith("#else"):
            parent, taken = stack[-1]
            active = parent and not taken
            stack[-1][1] = True
        elif s.startswith("#endif"):
            active = stack.pop()[0]
        elif active:
            keep.append(i)
    return keep


def _indent(s):
    return len(s) - len(s.lstrip(" "))


def step_rows(lines, step, defines=("STEP",), base=None):
    """What step `step` changes, as display rows [(kind, text)]: 'add' and 'del' lines, and
    dim 'ctx' lines (the block they sit in) and 'gap's between separate places."""
    now = set(preprocess(lines, step, defines))
    before = set(preprocess(lines, step - 1, ("STEP",))) if base is None else set(base)
    changed = sorted((now - before) | (before - now))
    rows, last, ctx_open = [], None, None
    for i in changed:
        if not lines[i].strip() and (last is None or i != last + 1):
            continue
        if last is not None and any(lines[j].strip() and (j in now or j in before) for j in range(last + 1, i)):
            # an unchanged line sits between: a new place in the file
            ind = _indent(lines[i])
            ctx = None
            if ind > 4:
                for j in range(i - 1, -1, -1):
                    if j in now and lines[j].strip() and _indent(lines[j]) < ind:
                        ctx = j
                        break
            if ctx is not None and ctx == ctx_open:
                rows.append(("gap", ""))
            else:
                if ctx_open is not None:
                    rows.append(("ctx", "});"))
                rows.append(("gap", ""))
                if ctx is not None:
                    rows.append(("ctx", lines[ctx][4:]))
                ctx_open = ctx
        elif last is None:
            ind = _indent(lines[i])
            if ind > 4:
                for j in range(i - 1, -1, -1):
                    if j in now and lines[j].strip() and _indent(lines[j]) < ind:
                        rows.append(("ctx", lines[j][4:]))
                        ctx_open = j
                        break
        kind = "add" if i in now else "del"
        rows.append((kind, lines[i][4:] if lines[i].startswith("    ") else lines[i]))
        last = i
    while rows and not rows[-1][1].strip() and rows[-1][0] != "gap":
        rows.pop()
    if ctx_open is not None:
        rows.append(("ctx", "});"))
    # blank rows inside a run stay (they are part of the code); strip them at the ends
    while rows and rows[0][0] == "gap":
        rows.pop(0)
    return rows


def app_source():
    with open(APP_CPP, encoding="utf-8") as f:
        return f.read().split("\n")


# ── the program's own window, streamed from its capture build ─────────────────
def capture_exe(tag):
    name = f"lesson_app_capture_{tag}.exe" if os.name == "nt" else f"lesson_app_capture_{tag}"
    roots = [os.environ.get("LESSON_APP_BIN", "")] + sorted(glob.glob(os.path.join(REPO, "cmake-build-*", "bin")))
    for r in roots:
        p = os.path.join(r, name)
        if r and os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{name}: build the CMake targets lesson_app_capture_0..8 and "
                            f"lesson_app_capture_6_separate (or point LESSON_APP_BIN at them)")


class Script:
    """A mouse script for one capture: events at film times, turned into capture frames."""

    def __init__(self, t0, t1):
        self.t0, self.t1 = t0, t1
        self.events = []        # (frame, text)
        self.pointer = []       # (frame, x, y, pressed)
        self.free = t0          # when the previous gesture is over

    def frame(self, t):
        return max(0, int(round((t - self.t0) * FPS)))

    @property
    def frames(self):
        return self.frame(self.t1) + 1

    def move(self, t, x, y, pressed=False):
        k = self.frame(t)
        self.events.append((k, f"move {x:.1f} {y:.1f}"))
        self.pointer.append((k, x, y, pressed))

    def glide(self, t0, t1, p0, p1, pressed=False):
        k0, k1 = self.frame(t0), self.frame(t1)
        for k in range(k0, k1 + 1):
            u = lesson.smoother(remap(k, k0, k1))
            self.move(self.t0 + k / FPS, p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u, pressed)

    def drag(self, t0, t1, p0, p1, button=0):
        assert t0 - 0.3 > self.free, "a drag must start after the previous one has let go"
        self.glide(max(t0 - 0.6, self.free + 0.1), t0 - 0.05, (p0[0] + 60, p0[1] + 90), p0)
        k = self.frame(t0)
        self.events.append((k, f"down {button} {p0[0]:.1f} {p0[1]:.1f}"))
        self.glide(t0, t1, p0, p1, pressed=True)
        self.events.append((self.frame(t1) + 1, f"up {button} {p1[0]:.1f} {p1[1]:.1f}"))
        self.pointer.append((self.frame(t1) + 1, p1[0], p1[1], False))
        self.free = t1 + 2.0 / FPS

    def wheel(self, t, dy):
        self.events.append((self.frame(t), f"wheel {dy}"))

    def text(self):
        return f"frames {self.frames}\n" + "\n".join(f"{k} {e}" for k, e in sorted(self.events, key=lambda e: e[0]))

    def pointer_at(self, t):
        """(x, y, pressed, alpha) of the pointer drawn over the window, or None."""
        if not self.pointer:
            return None
        k = (t - self.t0) * FPS
        first, last = self.pointer[0][0], max(p[0] for p in self.pointer)
        a = envelope(k / FPS, first / FPS - 0.3, last / FPS + 1.0, 0.3, 0.5)
        if a <= 0.003:
            return None
        cur = self.pointer[0]
        for p in self.pointer:
            if p[0] <= k:
                cur = p
            else:
                break
        return cur[1], cur[2], cur[3], a


class Stream:
    """One capture build, running headless and streaming its frames into a texture. Frames
    come in order; asking for an earlier one restarts the program (it is deterministic)."""
    W, H = 1280, 720

    def __init__(self, tag, script, stats=False):
        self.tag, self.script = tag, script
        self.exe = capture_exe(tag)
        fd, self.script_path = tempfile.mkstemp(prefix=f"lesson_app_{tag}_", suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(script.text() + "\n")
        self.stats_path = self.script_path[:-4] + ".stats"
        self.proc, self.k, self.cur = None, -1, None
        self.texture = None
        self.calls = None
        atexit.register(self.close)
        if stats:
            self._read_stats()

    def _start(self):
        self.close_proc()
        env = dict(os.environ, LESSON_SCRIPT=self.script_path, LESSON_OUT="-", LESSON_STATS=self.stats_path)
        self.proc = subprocess.Popen([self.exe], env=env, cwd=os.path.dirname(self.exe), stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        # anything the program printed before its first frame comes first; the marker line ends it
        buf = b""
        while b"LESSON_FRAMES" not in buf or not buf[buf.index(b"LESSON_FRAMES"):].count(b"\n"):
            ch = self.proc.stdout.read(1)
            if not ch:
                raise RuntimeError(f"{self.exe} ended before its first frame")
            buf += ch
        self.k = -1

    def _read_one(self):
        n = self.W * self.H * 3
        data = bytearray()
        while len(data) < n:
            chunk = self.proc.stdout.read(n - len(data))
            if not chunk:
                raise RuntimeError(f"{self.exe} ended at frame {self.k + 1}")
            data += chunk
        self.k += 1
        return data

    def frame(self, k):
        """Frame k (0-based, clamped to the script) as (H, W, 3) uint8."""
        k = min(max(int(k), 0), self.script.frames - 1)
        if self.proc is None or k < self.k:
            self._start()
        data = None
        while self.k < k:
            data = self._read_one()
        if data is not None:
            self.cur = np.frombuffer(bytes(data), np.uint8).reshape(self.H, self.W, 3)
        return self.cur

    def at(self, t):
        return self.frame(self.script.frame(t))

    def tex(self, t):
        img = self.at(t)[::-1]            # textures take their rows bottom first
        if self.texture is None:
            self.texture = tp.data_texture(np.ascontiguousarray(img), True)
            self._shown = self.k
        elif self._shown != self.k:
            self.texture.update_data(np.ascontiguousarray(img))
            self._shown = self.k
        return self.texture

    def _read_stats(self):
        """Run the program once through, for the renderer's counts on every frame."""
        self.frame(self.script.frames - 1)
        self.close_proc()
        self.calls = np.loadtxt(self.stats_path).reshape(-1, 2)

    def stat(self, t):
        k = min(self.script.frame(t), len(self.calls) - 1)
        return int(self.calls[k, 0]), int(self.calls[k, 1])

    def close_proc(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self.proc = None

    def close(self):
        self.close_proc()
        for p in (self.script_path, self.stats_path):
            try:
                os.remove(p)
            except OSError:
                pass


# ── when each program runs, and what the mouse does ───────────────────────────
TYPE_S = 0.09                                     # seconds per code row as it types in


def w_start(n, rows):
    """When the window switches to step n: once its code has typed in."""
    b = TL.start(BEAT[n])
    return b + 0.8 + min(TYPE_S * rows, 2.6) + 0.5


def build_streams(rows_of):
    ws = {n: w_start(n, len(rows_of[n])) for n in range(9)}
    ws[0] = TL.start("recap") + 0.5
    ends = {n: ws[n + 1] + 0.6 for n in range(8)}
    ends[8] = TL.duration
    sc = {}
    # the opening: the finished app, untouched
    sc["open"] = Script(0.0, TL.end("open") + 0.6)
    for n in range(9):
        sc[n] = Script(ws[n], ends[n])
    # 1: drag to turn, then scroll to zoom
    s, c = sc[1], cap0("o1")
    s.drag(c + 3.2, c + 5.0, (660, 420), (560, 380))
    for j in range(4):
        s.wheel(c + 6.0 + 0.18 * j, 1.0)
    s.drag(cap0("o2") + 1.0, cap0("o2") + 3.0, (560, 380), (700, 440))
    # 2 and 3: the same turn round to his back, once under one light, once under the sky
    for n, key in ((2, "m2"), (3, "k1")):
        s, c = sc[n], cap0(key) + (1.2 if n == 2 else 4.0)
        s.drag(c, c + 1.6, (520, 430), (780, 430))
        s.drag(c + 2.8, c + 4.2, (780, 430), (520, 430))
    # 7: the pointer sweeps over the stones
    # (stone centres picked from the program's own frame, so each pause lands on a stone)
    s, c = sc[7], ws[7] + 0.8
    stops = [(216, 576), (307, 599), (325, 663), (441, 645), (751, 601), (966, 589), (1154, 676)]
    prev = (120, 700)
    for p in stops:
        s.glide(c, c + 0.9, prev, p)
        c += 1.6
        prev = p
    # 8: the sun slider, the speed slider, then a drag in the scene and one on the panel
    s, c = sc[8], ws[8] + 1.0
    s.drag(c, c + 2.2, (58, 55), (175, 55))
    s.drag(c + 3.0, c + 4.0, (131, 80), (195, 80))
    c = max(cap0("u2") + 0.3, c + 5.0)
    s.drag(c, c + 1.3, (700, 470), (610, 470))
    s.drag(c + 2.2, c + 3.6, (175, 55), (105, 55))
    return ws, sc


# ── the picture ───────────────────────────────────────────────────────────────
WX0, WY0, WX1 = 960, 112, 1880                       # the program's window on screen
WBAR = 34
WY1 = WY0 + WBAR + (WX1 - WX0) * 9 / 16
SX = (WX1 - WX0 - 4) / Stream.W                        # window pixels to screen


def pointer_icon(ov, x, y, pressed, alpha):
    """An arrow pointer with its tip at (x, y)."""
    pts = [(0, 0), (0, 26), (6.5, 20), (11, 30), (15, 28.5), (10.5, 18.8), (19, 18.8), (0, 0)]
    s = 1.0 if not pressed else 0.92
    ov.line([(x + px * s, y + py * s) for px, py in pts], 0x000000, 5.0, 0.55 * alpha)
    ov.line([(x + px * s, y + py * s) for px, py in pts], 0xffffff, 2.2, alpha)
    if pressed:
        ov.circle(x, y, 14, outline=C_WARM, width=2.5, alpha=0.9 * alpha)


class Painter:
    def __init__(self, st, ov, rec):
        self.st, self.ov, self.rec = st, ov, rec
        self.lines = rec["lines"]
        self.rows = rec["rows"]
        self.ws, self.sc = rec["ws"], rec["sc"]
        self.streams = rec["streams"]
        self.captions = SPEECH.captions(CAP, fields=rec["numbers"])

    # which program the window shows at t, and how strongly (crossfades at each step)
    def window_layers(self, t):
        if t < TL.end("open") + 0.1:
            return [("open", 1.0)]
        order = [n for n in range(9)]
        cur = 0
        for n in order:
            if t >= self.ws[n]:
                cur = n
        layers = []
        if cur > 0 and t < self.ws[cur] + 0.4:
            layers.append((cur - 1, 1.0))
            layers.append((cur, smooth(remap(t, self.ws[cur], self.ws[cur] + 0.4))))
        else:
            layers.append((cur, 1.0))
        out = []
        for n, a in layers:
            if n == 6 and t < cap0("n2") + 0.6:
                out.append(("6sep", a))
                if t > cap0("n2") + 0.2:
                    out.append((6, a * smooth(remap(t, cap0("n2") + 0.2, cap0("n2") + 0.6))))
            else:
                out.append((n, a))
        return out

    def stream_script(self, key):
        return self.sc[6] if key == "6sep" else self.sc[key]

    def draw_window(self, t, alpha):
        ov = self.ov
        if alpha <= 0.003:
            return
        x0, y0, x1 = self.window_zoom(t)
        y1 = y0 + WBAR + (x1 - x0) * 9 / 16
        sx = (x1 - x0 - 4) / Stream.W
        ov.panel(x0, y0, x1 - x0, y1 - y0, radius=10, fill=0x000000, alpha=alpha)
        for key, a in self.window_layers(t):
            s = self.streams[key]
            ov.image(x0 + 2, y0 + WBAR + 2, x1 - x0 - 4, y1 - y0 - WBAR - 4, s.tex(t), alpha=alpha * a)
        ov.outline(x0, y0, x1 - x0, y1 - y0, 0x8aa0c0, 0.55 * alpha, width=2.0, radius=10)
        ov.panel(x0 + 2, y0 + 2, x1 - x0 - 4, WBAR, radius=8, fill=0x1a2233, alpha=0.95 * alpha)
        n = self.step_at(t)
        label = "app" if n is None else f"app  ·  built with STEP={n}"
        ov.text(x0 + 20, y0 + 19, label, size=17, color=TEXT, alpha=alpha, anchor="lm")
        for k, c in enumerate((0x5ee27a, 0xf4d35e, 0xef476f)):
            ov.circle(x1 - 26 - 22 * k, y0 + 19, 6, fill=c, alpha=0.8 * alpha)
        # the scripted mouse
        key = self.window_layers(t)[-1][0]
        if key != "open":
            p = self.stream_script(key).pointer_at(t)
            if p is not None:
                px, py, pressed, pa = p
                pointer_icon(ov, x0 + 2 + px * sx, y0 + WBAR + 2 + py * sx, pressed, alpha * pa)
        # the switch to a new step
        for n in range(1, 9):
            fa = alpha * envelope(t, self.ws[n] - 0.05, self.ws[n] + 1.4, 0.15, 0.5)
            if fa > 0.003:
                txt = f"STEP {n}  ·  built and running"
                w = ov.text_width(txt, 22, "semibold") + 48
                cx, cy = (x0 + x1) / 2, y0 + WBAR + 40
                ov.panel(cx - w / 2, cy - 22, w, 44, radius=22, fill=0x0d131e, alpha=0.85 * fa)
                ov.text(cx, cy, txt, size=22, color=C_ACCENT, alpha=fa, kind="semibold", anchor="mm")
        return x0, y0, x1, y1, sx

    def window_zoom(self, t):
        """The window's rectangle (x0, y0, x1): large for the opening, then beside the code."""
        u = smooth(remap(t, TL.end("open") - 1.0, TL.end("open") + 0.4))
        big = (1010, 250, 1850)
        return tuple(b + (s - b) * u for b, s in zip(big, (WX0, WY0, WX1)))

    def step_at(self, t):
        if t < TL.start("recap"):
            return None
        n = 0
        for k in range(9):
            if t >= self.ws[k]:
                n = k
        return n

    # 2D ----------------------------------------------------------------------
    def draw2d(self, t):
        ov = self.ov
        wa = envelope(t, 0.3, TL.end("end") + 0.6, 0.9, 0.8)
        ov.title_card(t, "A THREEPP LESSON  ·  PART 0b", "Your first real app",
                      "hello.cpp grows into an app, in eight steps", C_ACCENT, t_in=0.6, t_out=5.6, size=84)
        geo = self.draw_window(t, wa)
        self.pills(t)
        for n in range(9):
            self.code_step(t, n)
        if geo is not None:
            self.step_extras(t, geo)
        self.end_card(t)
        ov.captions(t, self.captions)
        ov.summary(t, TL.start("outro"), TL.end("outro"), SUMMARY,
                   "Every window in this film is app.cpp's own output, one build per step", C_ACCENT, C_ACCENT,
                   text_dx=330)

    def pills(self, t):
        ov = self.ov
        a = envelope(t, TL.start("recap") + 0.3, TL.end("end"), 0.6, 0.6)
        if a <= 0.003:
            return
        cur = self.step_at(t) or 0
        px = WX0
        for k, name in enumerate(STEP_NAMES):
            seen = t >= self.ws[k] - 0.1
            on = 1.0 if k == cur else 0.0
            wpx = ov.text_width(name, 17, "semibold") + 30
            ov.panel(px, 58, wpx, 34, radius=17, fill=0x2b6f8f if on else 0x223452,
                     alpha=a * (0.95 if on else 0.55 if seen else 0.3))
            ov.text(px + wpx / 2, 75, name, size=17, color=TEXT if (seen or on) else DIM,
                    alpha=a * (1.0 if seen else 0.45), kind="semibold", anchor="mm")
            px += wpx + 8

    def code_step(self, t, n):
        """Step n's code: what it adds (and takes away), typing in from its beat's start."""
        ov = self.ov
        b = BEAT[n]
        t_in = TL.start(b) + 0.3
        t_out = TL.end(b) - 0.1
        a = envelope(t, t_in, t_out, 0.5, 0.5)
        if a <= 0.003:
            return
        rows = self.rows[n]
        size, lh = 18, 24
        cw = ov.text_width("0", size, "mono")
        x, y = 40, 56
        w = max(max(len(r[1]) for r in rows) * cw + 64, 600)
        h = 64 + lh * len(rows)
        title = f"STEP {n}  ·  {STEP_TITLES[n].upper()}" if n else "HELLO.CPP, FROM PART 0"
        card(ov, x, y, w, h, a, title)
        t_type = t_in + 0.5
        for j, (kind, text) in enumerate(rows):
            if not text.strip():
                continue
            yy = y + 48 + j * lh
            ra = a * smooth(remap(t, t_type + TYPE_S * j, t_type + TYPE_S * j + 0.35))
            if ra <= 0.003:
                continue
            if kind == "add" and n > 0:
                ov.panel(x + 12, yy - 1, 4, lh, radius=2, fill=C_OK, alpha=0.9 * ra)
            if kind == "del":
                ra *= 0.55
                ov.panel(x + 12, yy - 1, 4, lh, radius=2, fill=C_DEL, alpha=0.9 * ra)
            if kind == "ctx":
                ra *= 0.45
            toks = tokenize(text, "cpp")
            for col, s, c in toks:
                ov.text(x + 30 + col * cw, yy + lh / 2, s, size=size, color=c, alpha=ra, kind="mono", anchor="lm")
            if kind == "del":
                ov.line([(x + 30, yy + lh / 2), (x + 30 + len(text.rstrip()) * cw, yy + lh / 2)], C_DEL, 2.0, ra)

    def step_extras(self, t, geo):
        ov = self.ov
        x0, y0, x1, y1, sx = geo
        # 6: what the renderer counted
        a = envelope(t, self.ws[6] + 0.2, TL.end("s6") - 0.1, 0.5, 0.5)
        if a > 0.003:
            sep = t < cap0("n2") + 0.4
            s = self.streams["6sep" if sep else 6]
            calls, tris = s.stat(t)
            ov.readout(x1 - 16 - 400, y0 + WBAR + 16, 400, "RENDERER.INFO(), THIS FRAME",
                       [("draw calls", f"{calls:,}", C_WARM if sep else C_OK), ("triangles", f"{tris:,}", None),
                        ("", "one Mesh per stone" if sep else "one InstancedMesh", C_WARM if sep else C_OK)],
                       alpha=a, value_size=24)
            self.slow_card(t, x0, y1 + 24, a)
        # 7: the ray, seen from the side
        a = envelope(t, cap0("p1") + 0.8, TL.end("s7") - 0.1, 0.5, 0.5)
        if a > 0.003:
            self.ray_diagram(t, x0, y1 + 20, x1 - x0, 180, a)
        # 5: the light's camera, seen from the side
        a = envelope(t, cap0("h2") + 0.3, TL.end("s5") - 0.1, 0.5, 0.5)
        if a > 0.003:
            self.shadow_diagram(t, x0, y1 + 20, x1 - x0, 180, a)
        # 8: the panel, up close
        a = envelope(t, self.ws[8] + 0.8, TL.end("s8") - 0.1, 0.5, 0.5)
        if a > 0.003:
            self.panel_zoom(t, geo, a)

    def slow_card(self, t, x, y, a):
        """The Mesh-per-stone lines (app.cpp's SEPARATE_MESHES variant), shown while it runs."""
        ov = self.ov
        rows = self.rec["slow_rows"]
        size, lh = 17, 22
        cw = ov.text_width("0", size, "mono")
        w = max(len(r) for r in rows) * cw + 64
        ca = a * smooth(remap(t, cap0("n1") + 1.0, cap0("n1") + 1.6)) * (1 - smooth(remap(t, cap0("n2"), cap0("n2") + 0.5)))
        if ca <= 0.003:
            return
        card(ov, x, y, w, 60 + lh * len(rows), ca, "IN THE LOOP, THE SLOW WAY  ·  -DSEPARATE_MESHES")
        for j, text in enumerate(rows):
            for col, s, c in tokenize(text, "cpp"):
                ov.text(x + 30 + col * cw, y + 46 + j * lh + lh / 2, s, size=size, color=c, alpha=ca, kind="mono",
                        anchor="lm")

    def ray_diagram(self, t, x, y, w, h, a):
        """Camera, the screen with the pointer on it, and a ray that hits stones in order."""
        ov = self.ov
        ov.panel(x, y, w, h, radius=14, alpha=0.66 * a, outline=0x8aa0c0, outline_alpha=0.16)
        ov.text(x + 24, y + 22, "SEEN FROM THE SIDE", size=16, color=DIM, alpha=a, kind="semibold", tracking=2.2)
        gy = y + h - 34
        ov.line([(x + 30, gy), (x + w - 30, gy)], 0x55627a, 2.0, a)
        cx, cy = x + 120, y + 100
        ov.circle(cx, cy, 11, fill=C_ACCENT, alpha=a)
        ov.text(cx, cy + 34, "camera", size=17, color=DIM, alpha=a, anchor="mm")
        sx = cx + 120
        ov.line([(sx, cy - 36), (sx, cy + 36)], 0xdfe8ff, 2.0, 0.7 * a)
        ov.text(sx + 12, cy - 30, "screen", size=17, color=DIM, alpha=a, anchor="lm")
        u = smooth(remap(t, cap0("p1") + 1.2, cap0("p1") + 3.0))
        py = cy + 18
        ov.circle(sx, py, 6, fill=0xffffff, alpha=a)
        # the ray: through the pointer, down to the ground
        d = np.array([sx - cx, py - cy], float)
        d /= np.linalg.norm(d)
        L = (gy - 10 - cy) / d[1]
        ex, ey = cx + d[0] * L * u, cy + d[1] * L * u
        ov.arrow2d((cx, cy), (ex, ey), C_WARM, 2.6, 12, a * (u > 0.02))
        stones = [(0.55, 1), (0.78, 2), (0.93, 3)]
        order = smooth(remap(t, cap0("p2") + 0.5, cap0("p2") + 1.5))
        for f, k in stones:
            qx, qy = cx + d[0] * L * f, cy + d[1] * L * f
            hit = u >= f
            ov.circle(qx, qy + 4, 12, fill=C_WARM if (hit and k == 1) else 0x8a8f99, alpha=a * (0.5 + 0.5 * hit))
            if order > 0.01 and hit:
                ov.text(qx + 22, qy - 14, f"{k}", size=18, color=C_WARM if k == 1 else DIM, alpha=a * order,
                        kind="semibold", anchor="mm")
        ov.text(x + w - 30, y + 60, "hits[0] is the nearest", size=19, color=TEXT, alpha=a * order, anchor="ra")
        ov.text(x + w - 30, y + 92, "hits[0].instanceId says which stone", size=19, color=TEXT, alpha=a * order,
                anchor="ra")

    def shadow_diagram(self, t, x, y, w, h, a):
        """The light's orthographic camera from the side: a band 14 m across along its rays.
        Ground and stones inside the band can be shadowed; nothing outside it is."""
        ov = self.ov
        ov.panel(x, y, w, h, radius=14, alpha=0.66 * a, outline=0x8aa0c0, outline_alpha=0.16)
        ov.text(x + 24, y + 22, "THE LIGHT'S CAMERA, FROM THE SIDE", size=16, color=DIM, alpha=a,
                kind="semibold", tracking=2.2)
        S, gy, cx = 6.0, y + h - 28, x + w * 0.38            # pixels per metre, the ground, the soldier
        el = math.atan2(6, 5)                                 # the light's elevation, from (3, 6, 4)
        d = np.array([math.cos(el), math.sin(el)])            # towards the light
        p = np.array([-d[1], d[0]])                           # across its rays

        def P(v):
            return (cx + v[0] * S, gy - v[1] * S)
        u = smooth(remap(t, cap0("h2") + 0.6, cap0("h2") + 1.6))
        far = 15.0
        ov.panel(*P(np.array([-8.0, 0.4])), 16 * S, 0.4 * S, radius=0, fill=0x55627a, alpha=0.5 * a)
        for k in np.linspace(-6.5, 6.5, 12):
            if abs(k) > 1.0:
                ov.circle(*P(np.array([k, 0.2])), 3.5, fill=0x8a8f99, alpha=a)
        ov.line([P(np.array([0.0, 0.0])), P(np.array([0.0, 1.8]))], 0xdfe8ff, 5.0, a)
        # the band: its two sides, from where they meet the ground up to the light
        feet = []
        for sgn in (-1, 1):
            s0 = -7 * sgn * p[1] / d[1]                      # where this side meets the ground
            e0, e1 = p * 7 * sgn + d * s0, p * 7 * sgn + d * max(s0, far * u)
            feet.append(e0)
            ov.dashed(P(e0), P(e1), C_WARM, 2.0, 0.9 * a, dash=9, gap=6)
        ov.line([P(feet[0]), P(feet[1])], C_WARM, 4.0, 0.7 * a * u)
        if u > 0.98:
            ov.line([P(p * 7 + d * far), P(-p * 7 + d * far)], C_WARM, 2.5, a)
            q = P(d * far)
            ov.circle(q[0], q[1], 9, fill=0xfff1c4, alpha=a)
            ov.text(q[0] + 18, q[1] + 2, "light", size=17, color=DIM, alpha=a, anchor="lm")
        q = P(p * 7 + d * 6)
        ov.text(q[0] - 14, q[1], "7 m", size=18, color=C_WARM, alpha=a * u, anchor="rm")
        ov.text(x + w - 30, y + 60, "left, right, top, bottom = ±7 m", size=19, color=TEXT, alpha=a, anchor="ra")
        ov.text(x + w - 30, y + 92, "inside the band: shadows", size=19, color=DIM, alpha=a, anchor="ra")

    def panel_zoom(self, t, geo, a):
        """The ImGui panel, magnified: the same pixels, cropped from the frame."""
        ov = self.ov
        x0, y0, x1, y1, sx = geo
        img = self.streams[8].at(t)
        crop = np.ascontiguousarray(img[8:110, 8:388][::-1])
        if getattr(self, "_zoom_tex", None) is None:
            self._zoom_tex = tp.data_texture(crop, True)
        else:
            self._zoom_tex.update_data(crop)
        zw, zh = crop.shape[1] * 1.8, crop.shape[0] * 1.8
        zx, zy = x0 + 40, y1 + 22
        ov.panel(zx - 4, zy - 4, zw + 8, zh + 8, radius=8, fill=0x000000, alpha=a)
        ov.image(zx, zy, zw, zh, self._zoom_tex, alpha=a)
        ov.outline(zx - 4, zy - 4, zw + 8, zh + 8, C_ACCENT, 0.6 * a, width=2.0, radius=8)
        ov.text(zx + zw + 30, zy + 24, "the panel, 1.8 times", size=20, color=DIM, alpha=a, anchor="lm")
        # its pointer, when it is over the panel
        p = self.sc[8].pointer_at(t)
        if p is not None and p[0] < 388 and p[1] < 110:
            pointer_icon(ov, zx + (p[0] - 8) * 1.8, zy + (p[1] - 8) * 1.8, p[2], a * p[3])

    def end_card(self, t):
        ov = self.ov
        a = envelope(t, TL.start("end") + 0.2, TL.end("end") + 0.3, 0.6, 0.6)
        if a <= 0.003:
            return
        # the whole file as a strip: every line, coloured by the step that brought it
        lines, owner = self.lines, self.rec["owner"]
        x, y = 70, 70
        card(ov, x - 30, y - 14, 860, 790, a, "app.cpp  ·  EVERY LINE, BY THE STEP THAT ADDED IT")
        shown = [i for i in range(len(lines)) if i in owner]
        lh = min(3.4, 700 / max(len(shown), 1))
        pal = [0x8aa0c0, 0x4cc9f0, 0xf4a259, 0x5ee27a, 0xc792ea, 0xffcb6b, 0xef476f, 0x82aaff, 0xa6e3a1]
        for j, i in enumerate(shown):
            ra = a * smooth(remap(t, TL.start("end") + 0.4 + 0.006 * j, TL.start("end") + 0.7 + 0.006 * j))
            s = lines[i].rstrip()
            ind = _indent(s)
            if not s.strip():
                continue
            ov.panel(x + 20 + ind * 4.2, y + 52 + j * lh, (len(s) - ind) * 4.2, max(lh - 1.0, 1.5), radius=0,
                     fill=pal[owner[i]], alpha=0.9 * ra)
        for n in range(9):
            la = a * smooth(remap(t, TL.start("end") + 1.2 + 0.15 * n, TL.start("end") + 1.6 + 0.15 * n))
            yy = y + 52 + 60 * n + 10
            ov.circle(x + 520, yy, 7, fill=pal[n], alpha=la)
            ov.text(x + 540, yy, f"{n}  {STEP_NAMES[n]}", size=21, color=TEXT, alpha=la, anchor="lm")
        ca = a * smooth(remap(t, cap0("c2") - 0.2, cap0("c2") + 0.5))
        card(ov, 960, WY1 + 24, 920, 64 + 34 * len(BUILD_CMD), ca, "ANY STEP, ON ITS OWN")
        for j, line in enumerate(BUILD_CMD):
            ov.text(990, WY1 + 24 + 50 + 34 * j + 17, line, size=21, color=C_ACCENT, alpha=ca, kind="mono",
                    anchor="lm")


# ── the stage ─────────────────────────────────────────────────────────────────
def setup(width, height):
    t0 = time.time()
    st = Stage(width, height, renderer="gl", fog=(4.0, 14.0))
    st.look((0.0, 1.6, 6.5), (0.0, 0.4, 0.0), 32)
    lines = app_source()
    rows = {n: step_rows(lines, n) for n in range(1, 9)}
    rows[0] = []
    for i in preprocess(lines, 0):
        if _body(lines, i, blank=True) and not (rows[0] and not rows[0][-1][1] and not lines[i].strip()):
            rows[0].append(("add", lines[i][4:] if lines[i].startswith("    ") else lines[i]))
    slow = preprocess(lines, 6, ("STEP", "SEPARATE_MESHES"))
    fast = set(preprocess(lines, 6))
    slow_rows = [lines[i][8:] for i in slow if i not in fast and lines[i].strip()]
    # which step brought each line of the whole app
    owner = {}
    for n in range(9):
        for i in preprocess(lines, n):
            owner.setdefault(i, n)
    for i in list(owner):
        if i not in set(preprocess(lines, 8)) or not _body(lines, i):
            del owner[i]
    n_lines = sum(1 for i in owner if lines[i].strip())
    ws, sc = build_streams(rows)
    streams = {"open": Stream(8, sc["open"])}
    for n in range(9):
        streams[n] = Stream(n, sc[n], stats=(n == 6))
    streams["6sep"] = Stream("6_separate", sc[6], stats=True)
    k2 = cap0("n2") + 2.0
    sep = streams["6sep"].calls[:, 0]
    k0, k1 = sc[6].frame(cap0("n2")), sc[6].frame(CAP["n2"][1])
    inst = streams[6].calls[k0:k1 + 1, 0]          # while the line about it is on screen
    lo, hi = int(inst.min()), int(inst.max())
    numbers = {"sep": f"{int(round(np.median(sep), -1)):,}", "inst": f"{lo}" if lo == hi else f"{lo} or {hi}",
               "lines": f"{n_lines}"}
    print(f"[app] {n_lines} lines in the finished app; step rows {[len(rows[n]) for n in range(9)]}")
    print(f"[draw calls] one Mesh per stone {sep.min():.0f}..{sep.max():.0f}, "
          f"one InstancedMesh {inst.min():.0f}..{inst.max():.0f}")
    rec = dict(lines=lines, rows=rows, slow_rows=slow_rows, owner=owner, ws=ws, sc=sc, streams=streams,
               numbers=numbers)
    ov = Hud(1920, 1080, math_cache=os.path.join(_HERE, "first_app.math.json"))
    painter = Painter(st, ov, rec)
    print(f"[setup] {time.time() - t0:.1f}s")

    def render(t):
        ov.begin()
        painter.draw2d(t)
        ov.fade(fade_in_out(t, TL.duration))
        ov.end()
        # a program whose part of the film is over can go
        for key, s in streams.items():
            if s.proc is not None and t > s.script.t1 + 1.0:
                s.close_proc()
        return st.frame(t, hud=ov)
    return render, painter.captions


def _body(lines, i, blank=False):
    """A line inside main() (blank ones only when asked)."""
    start = next(k for k, l in enumerate(lines) if l.startswith("int main"))
    end = max(k for k, l in enumerate(lines) if l.rstrip() == "}")
    return start < i < end and (blank or lines[i].strip() != "")


if __name__ == "__main__":
    run("first_app", TL.duration, setup, fps=FPS, speech=SPEECH)
