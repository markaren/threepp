"""threepp is the view.  Part 0c: Snake, from one main() to a model, views and players, about 4 min.

    python snake_oo.py                               # the film (1080p60), lesson_out/snake_oo.mp4
    python snake_oo.py --stills 30,60,95             # individual frames
    python snake_oo.py --sheet                       # contact sheet, one frame per 3 s
    python snake_oo.py --preview --out preview.mp4   # 960x540 @ 30 fps

The programs are in examples/lesson/snake/: tangle/ (Snake written the Part 0b way, all in
main()), trap/ (class Snake : public Group), and the finished app, whose rules (model/) are a
library that doesn't link threepp, drawn by views (view/) and played by controllers
(control/). main.cpp grows in six steps marked `#if STEP >= n`.

What is on screen was checked against the programs:

* every snippet is read out of those files; a step's rows are the lines that appear (and
  disappear) when the preprocessor runs with STEP = n instead of n - 1;
* every window is that program's own output. Each program (and each step) is built as a
  capture target (lesson_snake_*_capture, beside the programs) and run while the film is
  made: headless, on a fixed 1/60 s clock, with the keys pressed by a script through the
  canvas's own event path (examples/lesson/capture.hpp). The keys drawn under the window are
  the ones that script presses;
* the text board is what lesson_snake_capture_5 --ascii printed, frame for frame; the test
  run is test_snake's own output, run while the film is made;
* the compile error is what MSVC printed when a threepp #include was added to
  model/Game.cpp and snake_model was built (2026-09-29; the lines are shortened only where
  they carried the checkout's absolute path);
* "same game, frame for frame": the tangle and trap captures are compared frame by frame
  while the film is made, and the note is only shown if they match.
"""
from __future__ import annotations


import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import lesson  # noqa: E402
from lesson import (DIM, TEXT, Hud, Stage, Stream, card, envelope, fade_in_out, preprocess, remap,  # noqa: E402
                    run, smooth, step_rows, tokenize)

FPS = 60
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
SNAKE = os.path.join(REPO, "examples", "lesson", "snake")

C_ACCENT = 0x9be564
C_BLUE = 0x4cc9f0
C_WARM = 0xffb347
C_OK = 0x5ee27a
C_DEL = 0xef476f
C_OUT = 0xa6e3a1

CPP_KEYWORDS = lesson.KEYWORDS["cpp"] | {"for", "if", "else", "bool", "true", "false", "nullptr", "static", "void",
                                         "override", "virtual", "public", "private", "class", "struct", "const",
                                         "return", "while", "using"}

WORDS = {"AI": "A I", "AIs": "A Is", "main": "main", "STEP": "step", "C++": "C plus plus"}

# ── the script ────────────────────────────────────────────────────────────────
# The narration sets the pace: each caption lasts as long as its spoken line (Kokoro, af_heart,
# measured in snake_oo.speech.json) plus the voice's lead-in and tail and a small margin.
CAPTION_TEXT = {
    "title": "threepp is the view. Where the code goes, when a program has to grow.",
    "t1": "Here is Snake, written like our first app: everything lives in main, and it plays.",
    "t2": "But look at how it decides the snake has eaten: two float positions, less than half a unit apart. "
          "The truth of the game lives in the meshes.",
    "t3": "And a key press turns the snake at once. Press two keys between moves, and it turns back into its "
          "own neck.",
    "t4": "Now three requests arrive: test the rules, run them without a window, add a second player. "
          "Here, each one means untangling main.",
    "p1": "The first try at objects often looks like this: a Snake class that inherits from threepp's Group. "
          "Same game, frame for frame.",
    "p2": "It's tidier, but a snake is now a scene node, and everything a Group has is public: position, "
          "scale, add, remove.",
    "p3": "So one line anywhere can move it, and the snake you see and the snake the rules check part ways. "
          "It eats an apple it never touched, and dies off the board.",
    "p4": "A snake isn't a scene node. It has a look. So the rules move out, into classes of their own.",
    "m1": "The model: whole cells on a board, a snake, an apple, and one tick of the rules. No floats, no "
          "meshes, no window.",
    "m2": "It's a library of its own that doesn't link threepp. Include a threepp header in it, and the build "
          "stops.",
    "e1": "Rules without a window can be tested. A test sets up a game by hand, ticks it, and checks what "
          "happened.",
    "e2": "The whole suite runs in the blink of an eye, and it pins down the double key press from before.",
    "v1": "Now threepp comes back, as a view. It holds a group of meshes, and every frame it reads the game "
          "and places them.",
    "v2": "At first the view jumps from tick to tick. Then it slides between them. The rules keep their own "
          "clock, and never changed.",
    "c1": "Who plays is an interface: one question, asked once per tick. Which way now?",
    "c2": "The keyboard answers with the keys it has queued. The AI looks at the game and picks. Press tab, "
          "and the loop can't tell the difference.",
    "c3": "This is where inheritance belongs: a contract. The keyboard inherits two, ours and threepp's key "
          "listener, and no code.",
    "ev1": "When the snake eats, the game tells whoever is listening. The view pops a ring. The game never "
           "knows it's there.",
    "w1": "With the rules on their own, more views are cheap: a minimap in the corner, and the same game as "
          "text, with no threepp at all.",
    "w2": "Three views, one truth. They can't disagree, because none of them owns the game.",
    "r1": "And a second game is just more objects: another game, another view, another player.",
    "r2": "No globals to untangle, nothing to copy. Two AIs, side by side.",
    "z1": "The model owns the truth. Views only read it. Players only say what they want.",
    "z2": "threepp draws. It doesn't decide. Keep it at the edge, and the rest of your program stays yours to "
          "test, change and reuse.",
}
SPEECH = lesson.Speech(CAPTION_TEXT, os.path.join(_HERE, "snake_oo.speech.json"), words=WORDS,
                       voice_only={"title"})
PLAN = [("open", 0.4, ["title"], 2.4),
        ("tangle", 0.4, ["t1", "t2", "t3", "t4"], 0.8),
        ("trap", 0.4, ["p1", "p2", "p3", "p4"], 0.8),
        ("model", 0.4, ["m1", "m2"], 0.8),
        ("tests", 0.4, ["e1", "e2"], 1.0),
        ("view", 0.4, ["v1", "v2"], 0.8),
        ("players", 0.4, ["c1", "c2", "c3"], 0.8),
        ("events", 0.4, ["ev1"], 1.2),
        ("views", 0.4, ["w1", "w2"], 1.0),
        ("rival", 0.4, ["r1", "r2"], 1.4),
        ("end", 0.4, ["z1", "z2"], 0.8),
        ("outro", 0.0, [], 8.0)]
TL, CAP = SPEECH.layout(PLAN)


def cap0(k):
    return CAP[k][0]


def cap1(k):
    return CAP[k][1]


BEATS = ["tangle", "trap", "model", "tests", "view", "players", "events", "views", "rival"]
BEAT_PILLS = ["tangle", "trap", "model", "tests", "view", "players", "events", "views", "rival"]

SUMMARY = [(r"$\mathrm{model}$", "Rules in plain classes: no threepp, tested without a window."),
           (r"$\mathrm{view}$", "threepp reads the model and draws it. It never changes it."),
           (r"$\mathrm{player}$", "An interface: the keys, an AI or a test can play."),
           (r"$\mathrm{has\ a}$", "A class has a look. It doesn't inherit one.")]

# What MSVC printed when `#include "threepp/math/Vector3.hpp"` was added to model/Game.cpp and
# snake_model was built (2026-09-29), the checkout's absolute path left out.
COMPILE_TRY = ['#include "Game.hpp"', '#include "threepp/math/Vector3.hpp"']
COMPILE_OUT = ["> cmake --build build --target snake_model",
               "FAILED: examples/lesson/snake/CMakeFiles/snake_model.dir/model/Game.cpp.obj",
               "model\\Game.cpp(2): fatal error C1083: Cannot open include file:",
               "  'threepp/math/Vector3.hpp': No such file or directory",
               "ninja: build stopped: subcommand failed."]

GROUP_MEMBERS = ["position", "rotation", "scale", "add()", "remove()", "clear()", "visible", "castShadow",
                 "children", "lookAt()", "traverse()", "parent"]


# ── the sources ───────────────────────────────────────────────────────────────
def source(*parts):
    with open(os.path.join(SNAKE, *parts), encoding="utf-8") as f:
        return f.read().split("\n")


def excerpt(lines, first, last, drop=()):
    """The lines from the first containing `first` to the next containing `last`, dedented,
    without lines containing any of `drop`."""
    i = next(k for k, l in enumerate(lines) if first in l)
    j = next(k for k in range(i, len(lines)) if last in lines[k])
    out = [l for l in lines[i:j + 1] if not any(d in l for d in drop)]
    ind = min(len(l) - len(l.lstrip(" ")) for l in out if l.strip())
    return [l[ind:] for l in out]


def uses_threepp(*parts):
    return any('#include "threepp/' in l for l in source(*parts))


# ── keys, pressed by a script, at the moments the program moves ───────────────
def move_frames(kind, n):
    """The capture frames (0-based) on which the program makes its moves 1..n.
    tangle and trap move when their timer passes 0.125 s and start it again from 0: every 8th
    frame. The finished app keeps the remainder (a fixed time step), as float32 does."""
    if kind == "tangle":
        return [8 * k - 1 for k in range(1, n + 1)]
    out, since = [], np.float32(0)
    dt, tick = np.float32(1) / np.float32(60), np.float32(0.125)
    f = 0
    while len(out) < n:
        since = np.float32(since + dt)
        while since >= tick and len(out) < n:
            since = np.float32(since - tick)
            out.append(f)
        f += 1
    return out


TURN = {"R": ("UP", "U"), "U": ("LEFT", "L"), "L": ("DOWN", "D"), "D": ("RIGHT", "R")}
PERP = {"R": "UP", "L": "DOWN", "U": "LEFT", "D": "RIGHT"}
BACK = {"R": "LEFT", "L": "RIGHT", "U": "DOWN", "D": "UP"}


class KeyScript:
    """The keys one capture presses: events at capture frames, from film time t0 to t1."""

    def __init__(self, t0, t1):
        self.t0, self.t1 = t0, t1
        self.keys = []          # (frame, name)

    def frame(self, t):
        return max(0, int(round((t - self.t0) * FPS)))

    @property
    def frames(self):
        return self.frame(self.t1) + 1

    def press(self, k, name):
        self.keys.append((int(k), name))

    def press_at(self, t, name):
        self.press(self.frame(t), name)

    def text(self):
        return f"frames {self.frames}\n" + "\n".join(f"{k} key {n}" for k, n in sorted(self.keys))

    def recent(self, t, span=0.8):
        """[(name, age)] of the keys pressed in the last `span` seconds before t."""
        k = (t - self.t0) * FPS
        return [(n, (k - f) / FPS) for f, n in self.keys if 0 <= k - f < span * FPS]

    def loop(self, kind, first, side=6, until=None, bug_after=None):
        """Drive round a square: `first` moves on, then a turn left every `side` moves. With
        `bug_after` (a film time), two keys go down between the next two moves: one to the
        side, then straight back, which a program that turns on the key press can't survive."""
        n = self.frames // 7 + 2
        frames = move_frames(kind, n)
        heading, turn_at = "R", first + 1
        until = self.frames if until is None else self.frame(until)
        for m in range(1, n + 1):
            f = frames[m - 1]
            if f >= until:
                break
            if bug_after is not None and f - 6 >= self.frame(bug_after) and m != turn_at:
                self.press(f - 5, PERP[heading])
                self.press(f - 3, BACK[heading])
                return
            if m == turn_at:
                key, heading = TURN[heading]
                self.press(f - 3, key)
                turn_at += side


# ── the picture ───────────────────────────────────────────────────────────────
WX0, WY0, WX1 = 960, 112, 1880                       # the program's window on screen
WBAR = 34
WY1 = WY0 + WBAR + (WX1 - WX0) * 9 / 16
DY0 = WY1 + 22                                       # the diagram, under the window


def code_rows(lines):
    return [("code", l) for l in lines]


class Painter:
    def __init__(self, st, ov, rec):
        self.st, self.ov, self.rec = st, ov, rec
        self.streams, self.win = rec["streams"], rec["windows"]
        self.captions = SPEECH.captions(CAP)

    # which picture the window shows at t
    def window_layers(self, t):
        cur = None
        for key, a, b, _ in self.win:
            if a <= t:
                cur = (key, a)
        if cur is None:
            return []
        idx = [w[0] for w in self.win].index(cur[0])
        if idx > 0 and t < cur[1] + 0.4:
            return [(self.win[idx - 1][0], 1.0), (cur[0], smooth(remap(t, cur[1], cur[1] + 0.4)))]
        return [(cur[0], 1.0)]

    def label(self, key):
        return next(w[3] for w in self.win if w[0] == key)

    def window_zoom(self, t):
        u = smooth(remap(t, TL.end("open") - 1.0, TL.end("open") + 0.4))
        big = (1010, 250, 1850)
        return tuple(b + (s - b) * u for b, s in zip(big, (WX0, WY0, WX1)))

    def draw_window(self, t):
        ov = self.ov
        alpha = envelope(t, 0.3, TL.end("end") + 0.6, 0.9, 0.8)
        if alpha <= 0.003:
            return None
        x0, y0, x1 = self.window_zoom(t)
        y1 = y0 + WBAR + (x1 - x0) * 9 / 16
        ov.panel(x0, y0, x1 - x0, y1 - y0, radius=10, fill=0x000000, alpha=alpha)
        layers = self.window_layers(t)
        for key, a in layers:
            if key == "term":
                self.terminal(t, x0 + 2, y0 + WBAR + 2, x1 - x0 - 4, y1 - y0 - WBAR - 4, alpha * a)
            else:
                ov.image(x0 + 2, y0 + WBAR + 2, x1 - x0 - 4, y1 - y0 - WBAR - 4, self.streams[key].tex(t),
                         alpha=alpha * a)
        ov.outline(x0, y0, x1 - x0, y1 - y0, 0x8aa0c0, 0.55 * alpha, width=2.0, radius=10)
        ov.panel(x0 + 2, y0 + 2, x1 - x0 - 4, WBAR, radius=8, fill=0x1a2233, alpha=0.95 * alpha)
        if layers:
            ov.text(x0 + 20, y0 + 19, self.label(layers[-1][0]), size=17, color=TEXT, alpha=alpha, anchor="lm")
        for k, c in enumerate((0x5ee27a, 0xf4d35e, 0xef476f)):
            ov.circle(x1 - 26 - 22 * k, y0 + 19, 6, fill=c, alpha=0.8 * alpha)
        # the keys the script presses
        if layers and layers[-1][0] != "term":
            s = self.streams[layers[-1][0]].script
            px = x0 + 16
            for name, age in s.recent(t):
                ka = alpha * (1 - smooth(remap(age, 0.5, 0.8)))
                w = ov.text_width(name, 17, "semibold") + 26
                py = y1 - 44 + 10 * (1 - smooth(remap(age, 0, 0.15)))
                ov.panel(px, py, w, 32, radius=7, fill=0x0d131e, alpha=0.85 * ka, outline=C_WARM, outline_alpha=0.8)
                ov.text(px + w / 2, py + 16, name, size=17, color=C_WARM, alpha=ka, kind="semibold", anchor="mm")
                px += w + 8
        # the switch to a new step
        for key, a, _, lab in self.win:
            if not key.startswith("s") or key == "s6b":
                continue
            fa = alpha * envelope(t, a - 0.05, a + 1.4, 0.15, 0.5)
            if fa > 0.003:
                txt = f"STEP {key[1]}  ·  built and running"
                w = ov.text_width(txt, 22, "semibold") + 48
                cx, cy = (x0 + x1) / 2, y0 + WBAR + 40
                ov.panel(cx - w / 2, cy - 22, w, 44, radius=22, fill=0x0d131e, alpha=0.85 * fa)
                ov.text(cx, cy, txt, size=22, color=C_ACCENT, alpha=fa, kind="semibold", anchor="mm")
        return x0, y0, x1, y1

    def terminal(self, t, x, y, w, h, a):
        """The window as a terminal: the compile error, then the test run."""
        ov = self.ov
        ov.panel(x, y, w, h, radius=0, fill=0x0a0e14, alpha=a)
        size, lh = 17, 25
        if t < TL.start("tests"):
            ca = a * smooth(remap(t, cap0("m2") + 2.2, cap0("m2") + 2.6))
            lines = [(l, C_BLUE if l.startswith(">") else (C_DEL if "error" in l or "FAILED" in l else TEXT))
                     for l in COMPILE_OUT]
            for j, (l, c) in enumerate(lines):
                la = ca * smooth(remap(t, cap0("m2") + 2.4 + 0.25 * j, cap0("m2") + 2.6 + 0.25 * j))
                ov.text(x + 24, y + 30 + j * lh, l, size=size, color=c, alpha=la, kind="mono", anchor="lm")
            if ca < 0.01:
                ov.text(x + 24, y + 30, ">", size=size, color=C_BLUE, alpha=a, kind="mono", anchor="lm")
            return
        names, summary = self.rec["tests"]
        t0 = cap0("e1") + 1.0
        ov.text(x + 24, y + 30, "> test_snake", size=size, color=C_BLUE, alpha=a * smooth(remap(t, t0, t0 + 0.3)),
                kind="mono", anchor="lm")
        for j, (name, ok) in enumerate(names):
            la = a * smooth(remap(t, t0 + 0.6 + 0.05 * j, t0 + 0.7 + 0.05 * j))
            yy = y + 30 + (j + 1) * 24
            ov.text(x + 24, yy, "passed" if ok else "FAILED", size=16, color=C_OK if ok else C_DEL, alpha=la,
                    kind="mono", anchor="lm")
            ov.text(x + 110, yy, name, size=16, color=TEXT, alpha=la, kind="mono", anchor="lm")
        la = a * smooth(remap(t, t0 + 0.8 + 0.05 * len(names), t0 + 1.0 + 0.05 * len(names)))
        ov.text(x + 24, y + 30 + (len(names) + 1.4) * 24, summary, size=16, color=C_OUT, alpha=la, kind="mono",
                anchor="lm")
        # the test that covers the double key press
        hl = smooth(remap(t, cap0("e2") + 2.6, cap0("e2") + 3.2)) * (1 - smooth(remap(t, TL.end("tests") - 0.5,
                                                                                       TL.end("tests"))))
        for j, (name, _) in enumerate(names):
            if "turn back onto itself" in name and hl > 0.01:
                yy = y + 30 + (j + 1) * 24
                ov.panel(x + 12, yy - 13, w - 24, 26, radius=6, fill=0x223452, alpha=0.8 * hl * a)
                ov.panel(x + 12, yy - 13, 4, 26, radius=2, fill=C_WARM, alpha=hl * a)
                ov.text(x + 24, yy, "passed", size=16, color=C_OK, alpha=a, kind="mono", anchor="lm")
                ov.text(x + 110, yy, name, size=16, color=C_WARM, alpha=a, kind="mono", anchor="lm")

    # 2D ----------------------------------------------------------------------
    def draw2d(self, t):
        ov = self.ov
        ov.title_card(t, "A THREEPP LESSON  ·  PART 0c", "threepp is the view",
                      "Snake: from one main() to model, views and players", C_ACCENT, t_in=0.6,
                      t_out=TL.end("open") - 0.4, size=84)
        geo = self.draw_window(t)
        self.pills(t)
        for spec in self.rec["cards"]:
            self.rows_card(t, *spec)
        if geo is not None:
            self.under_window(t, geo)
        self.requests(t)
        self.is_has(t)
        self.ascii_panel(t)
        self.file_tree(t)
        ov.captions(t, self.captions)
        ov.summary(t, TL.start("outro"), TL.end("outro"), SUMMARY,
                   "Every window in this film is the program's own output, one build per step", C_ACCENT, C_ACCENT,
                   text_dx=300)

    def pills(self, t):
        ov = self.ov
        a = envelope(t, TL.start("tangle") + 0.3, TL.end("end"), 0.6, 0.6)
        if a <= 0.003:
            return
        cur = None
        for b in BEATS:
            if t >= TL.start(b):
                cur = b
        px = WX0
        for b in BEAT_PILLS:
            seen = t >= TL.start(b) - 0.1
            on = b == cur and t < TL.start("end")
            wpx = ov.text_width(b, 17, "semibold") + 26
            ov.panel(px, 58, wpx, 34, radius=17, fill=0x3d6b2e if on else 0x223452,
                     alpha=a * (0.95 if on else 0.55 if seen else 0.3))
            ov.text(px + wpx / 2, 75, b, size=17, color=TEXT if seen else DIM, alpha=a * (1.0 if seen else 0.45),
                    kind="semibold", anchor="mm")
            px += wpx + 7

    def rows_card(self, t, t_in, t_out, x, y, title, rows, hl=None, size=18, lh=24):
        """A card of source rows typing in from t_in: ('code', text) plain, ('add', text) with a
        green bar, ('del', text) struck out, ('ctx', text) dim, ('gap', ''). hl(j, t) -> 0..1
        highlights row j."""
        ov = self.ov
        a = envelope(t, t_in, t_out, 0.5, 0.5)
        if a <= 0.003:
            return
        longest = max(len(r[1]) for r in rows)
        while size > 13 and longest * ov.text_width("0", size, "mono") + 64 > WX0 - 60 - x:
            size, lh = size - 1, lh - 1
        cw = ov.text_width("0", size, "mono")
        w = max(longest * cw + 64, 560)
        h = 74 + lh * len(rows)
        card(ov, x, y, w, h, a, title)
        t_type = t_in + 0.4
        for j, (kind, text) in enumerate(rows):
            if not text.strip():
                continue
            yy = y + 58 + j * lh
            ra = a * smooth(remap(t, t_type + 0.06 * j, t_type + 0.06 * j + 0.3))
            if ra <= 0.003:
                continue
            h_j = hl(j, t) if hl else 0.0
            if h_j > 0.003:
                ov.panel(x + 12, yy - 1, w - 24, lh, radius=5, fill=0x223452, alpha=0.8 * h_j * ra)
                ov.panel(x + 12, yy - 1, 4, lh, radius=2, fill=C_WARM, alpha=h_j * ra)
            if kind == "add":
                ov.panel(x + 12, yy - 1, 4, lh, radius=2, fill=C_OK, alpha=0.9 * ra)
            if kind == "bad":
                ov.panel(x + 12, yy - 1, 4, lh, radius=2, fill=C_DEL, alpha=0.9 * ra)
            if kind == "del":
                ra *= 0.55
                ov.panel(x + 12, yy - 1, 4, lh, radius=2, fill=C_DEL, alpha=0.9 * ra)
            if kind == "ctx":
                ra *= 0.45
            for col, s, c in tokenize(text, "cpp", CPP_KEYWORDS):
                ov.text(x + 30 + col * cw, yy + lh / 2, s, size=size, color=c, alpha=ra, kind="mono", anchor="lm")
            if kind == "del":
                ov.line([(x + 30, yy + lh / 2), (x + 30 + len(text.rstrip()) * cw, yy + lh / 2)], C_DEL, 2.0, ra)

    # under the window: the Group's members, the architecture ----------------
    def under_window(self, t, geo):
        x0, y0, x1, y1 = geo
        ov = self.ov
        # p2: everything a Group has, public on Snake
        a = envelope(t, cap0("p2") + 1.2, cap1("p3") - 0.2, 0.5, 0.5)
        if a > 0.003:
            card(ov, WX0, DY0, WX1 - WX0, 150, a, "INHERITED FROM GROUP, PUBLIC ON EVERY SNAKE")
            px, py = WX0 + 28, DY0 + 62
            for k, m in enumerate(GROUP_MEMBERS):
                ma = a * smooth(remap(t, cap0("p2") + 1.5 + 0.12 * k, cap0("p2") + 1.8 + 0.12 * k))
                w = ov.text_width(m, 19, "mono") + 26
                if px + w > WX1 - 20:
                    px, py = WX0 + 28, py + 44
                ov.panel(px, py - 16, w, 34, radius=8, fill=0x223452, alpha=0.8 * ma)
                ov.text(px + w / 2, py + 1, m, size=19, color=TEXT, alpha=ma, kind="mono", anchor="mm")
                px += w + 10
        self.diagram(t)
        # p1: same game, frame for frame
        a = envelope(t, cap0("p1") + 3.0, cap0("p3"), 0.5, 0.5)
        if a > 0.003 and self.rec["same_pixels"]:
            txt = "same keys, same frames as the tangle"
            w = ov.text_width(txt, 19, "semibold") + 40
            ov.panel(x1 - w - 16, y0 + WBAR + 16, w, 38, radius=19, fill=0x0d131e, alpha=0.85 * a)
            ov.text(x1 - w / 2 - 16, y0 + WBAR + 35, txt, size=19, color=C_ACCENT, alpha=a, kind="semibold",
                    anchor="mm")

    def diagram(self, t):
        """Who depends on whom, built up beat by beat: the model in the middle, threepp at the
        edge. The arrows are #includes (checked in setup against the files)."""
        ov = self.ov
        a = envelope(t, TL.start("model") + 0.3, TL.end("end") + 0.3, 0.6, 0.6)
        a *= 1 - smooth(remap(t, TL.start("views") - 0.3, TL.start("views"))) + \
            smooth(remap(t, TL.end("views") - 0.3, TL.end("views")))
        if a <= 0.003:
            return
        y = DY0
        card(ov, WX0, y, WX1 - WX0, 196, a, "WHO INCLUDES WHOM")
        boxes = {"tests": (990, y + 100, 150, "tests", TL.start("tests")),
                 "model": (1210, y + 100, 170, "model", TL.start("model")),
                 "views": (1450, y + 58, 170, "views", TL.start("view")),
                 "players": (1450, y + 152, 170, "players", TL.start("players")),
                 "threepp": (1700, y + 100, 150, "threepp", TL.start("model"))}
        cur = None
        for b in BEATS + ["end"]:
            if t >= TL.start(b):
                cur = b
        lit = {"model": "model", "tests": "tests", "view": "views", "players": "players", "events": "views",
               "views": "views", "rival": "model", "end": None}.get(cur)

        def box(k):
            bx, by, bw, name, t_on = boxes[k]
            ba = a * smooth(remap(t, t_on + 0.2, t_on + 0.8))
            if ba <= 0.003:
                return 0.0
            on = k == lit
            ov.panel(bx, by - 22, bw, 44, radius=10, fill=0x3d6b2e if on else (0x1d3354 if k == "threepp" else 0x223452),
                     alpha=0.95 * ba, outline=C_ACCENT if on else 0x8aa0c0, outline_alpha=0.6 if on else 0.2)
            ov.text(bx + bw / 2, by, name, size=20, color=TEXT, alpha=ba, kind="semibold", anchor="mm")
            return ba

        def arrow(k0, k1, label, t_on, color=0x8aa0c0, cut=False, below=False):
            ea = a * smooth(remap(t, t_on, t_on + 0.6))
            if ea <= 0.003:
                return
            bx0, by0, bw0, _, _ = boxes[k0]
            bx1, by1, bw1, _, _ = boxes[k1]
            p0 = (bx0 + bw0 + 4, by0) if bx1 > bx0 else (bx0 - 4, by0)
            p1 = (bx1 - 6, by1) if bx1 > bx0 else (bx1 + bw1 + 6, by1)
            if cut:
                ov.dashed(p0, p1, C_DEL, 2.2, 0.8 * ea, dash=8, gap=6)
                mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                ov.line([(mx - 9, my - 9), (mx + 9, my + 9)], C_DEL, 3.0, ea)
                ov.line([(mx - 9, my + 9), (mx + 9, my - 9)], C_DEL, 3.0, ea)
                ov.text(mx, my + 18, label, size=15, color=C_DEL, alpha=ea, anchor="mm")
            else:
                ov.arrow2d(p0, p1, color, 2.2, 10, ea)
                mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                ov.text(mx, my + (16 if below else -14), label, size=15, color=DIM, alpha=ea, anchor="mm")

        for k in boxes:
            box(k)
        arrow("model", "threepp", "can't include", cap0("m2") + 3.0, cut=True)
        arrow("tests", "model", "checks", cap0("e1") + 0.5)
        arrow("views", "model", "reads", cap0("v1") + 1.5, C_ACCENT)
        arrow("views", "threepp", "draws with", cap0("v1") + 2.2, C_BLUE)
        arrow("players", "model", "steers", cap0("c1") + 1.0, C_WARM, below=True)
        arrow("players", "threepp", "keys only", cap0("c3") + 2.0, 0x55627a, below=True)

    # left-hand cards that aren't code ----------------------------------------
    def requests(self, t):
        ov = self.ov
        a = envelope(t, cap0("t4"), TL.end("tangle") - 0.1, 0.5, 0.5)
        if a <= 0.003:
            return
        x, y = 40, 56
        card(ov, x, y, 860, 360, a, "THREE REQUESTS")
        rows = [("test the rules", "they run inside main's render loop"),
                ("run without a window", "the game's state is the meshes"),
                ("add a second player", "one of everything, in one function")]
        for k, (req, why) in enumerate(rows):
            ra = a * smooth(remap(t, cap0("t4") + 1.4 + 0.9 * k, cap0("t4") + 1.9 + 0.9 * k))
            yy = y + 90 + 84 * k
            ov.text(x + 40, yy, req, size=30, color=TEXT, alpha=ra, kind="semibold", anchor="lm")
            ov.text(x + 40, yy + 34, why, size=21, color=C_DEL, alpha=ra, anchor="lm")

    def is_has(self, t):
        ov = self.ov
        a = envelope(t, cap0("p4"), TL.end("trap") - 0.1, 0.5, 0.5)
        if a <= 0.003:
            return
        x, y = 40, 56
        card(ov, x, y, 860, 300, a, "IS A, OR HAS A")
        cw = ov.text_width("0", 20, "mono")
        is_a, has_a = self.rec["is_has"]
        rows = [(is_a, "trap/main.cpp: a snake is a scene node", C_DEL, 0.3),
                (has_a, "view/SnakeView.hpp: a view has one, and the snake has none", C_OK, 2.2)]
        yy = y + 80
        for k, (code, why, c, dt) in enumerate(rows):
            ra = a * smooth(remap(t, cap0("p4") + dt, cap0("p4") + dt + 0.5))
            for j, line in enumerate(code):
                for col, s, cc in tokenize(line, "cpp", CPP_KEYWORDS):
                    ov.text(x + 40 + col * cw, yy + 28 * j, s, size=20, color=cc, alpha=ra, kind="mono", anchor="lm")
                if k == 0:
                    ov.line([(x + 34, yy + 28 * j), (x + 46 + len(line) * cw, yy + 28 * j)], C_DEL, 2.5, ra)
            yy += 28 * len(code) + 12
            ov.text(x + 40, yy, why, size=21, color=c, alpha=ra, anchor="lm")
            yy += 64

    def ascii_panel(self, t):
        """What lesson_snake_capture_5 --ascii printed: its latest board."""
        ov = self.ov
        a = envelope(t, cap0("w1") + 3.0, TL.end("views") - 0.1, 0.5, 0.5)
        if a <= 0.003:
            return
        s = self.streams["s5"]
        board = [l for l in s.printed(t) if l.startswith("#")][-22:]
        x, y = 40, 470
        size, lh = 14, 15
        cw = ov.text_width("0", size, "mono")
        w = 23 * cw + 150
        card(ov, x, y, max(w, 420), 60 + lh * 22, a, "STDOUT  ·  lesson_snake --ascii")
        for j, line in enumerate(board):
            ov.text(x + 30, y + 52 + j * lh, line, size=size, color=C_OUT, alpha=a, kind="mono", anchor="lm")

    def file_tree(self, t):
        ov = self.ov
        a = envelope(t, TL.start("end") + 0.2, TL.end("end") + 0.4, 0.6, 0.6)
        if a <= 0.003:
            return
        x, y = 40, 56
        rows = self.rec["tree"]
        card(ov, x, y, 860, 90 + 34 * len(rows), a, "examples/lesson/snake")
        for k, (name, desc, tp_) in enumerate(rows):
            ra = a * smooth(remap(t, TL.start("end") + 0.5 + 0.12 * k, TL.start("end") + 0.9 + 0.12 * k))
            yy = y + 66 + 34 * k
            indent = 0 if name.endswith("/") or name == "main.cpp" else 34
            if tp_ is not None:
                ov.circle(x + 44 + indent, yy, 6, fill=C_BLUE if tp_ else C_OK, alpha=ra)
            ov.text(x + 60 + indent, yy, name, size=20, color=TEXT, alpha=ra, kind="mono", anchor="lm")
            ov.text(x + 440, yy, desc, size=19, color=DIM, alpha=ra, anchor="lm")
        la = a * smooth(remap(t, TL.start("end") + 2.2, TL.start("end") + 2.8))
        yy = y + 66 + 34 * len(rows) + 4
        ov.circle(x + 44, yy, 6, fill=C_OK, alpha=la)
        ov.text(x + 60, yy, "no threepp", size=19, color=DIM, alpha=la, anchor="lm")
        ov.circle(x + 230, yy, 6, fill=C_BLUE, alpha=la)
        ov.text(x + 246, yy, "includes threepp", size=19, color=DIM, alpha=la, anchor="lm")


# ── the stage ─────────────────────────────────────────────────────────────────
def word_time(key, word, default):
    """Film time at which `word` of caption `key` is heard (Kokoro's alignment), else `default`
    seconds into the caption."""
    try:
        for w, s, _ in lesson.Narration(words=SPEECH.words, cache_dir="voice_cache").word_times(CAPTION_TEXT[key]):
            if w.lower().strip(".,") == word:
                return cap0(key) + lesson.VOICE_LEAD + s
    except Exception as e:  # noqa: BLE001  (no Kokoro: the film still builds)
        print(f"[words] no alignment for {key} ({e})")
    return cap0(key) + default


def run_tests():
    """test_snake's own run: every test case, passed or not, and its summary line."""
    exe = lesson.capture_exe("test_snake")
    xml = subprocess.run([exe, "-r", "junit"], capture_output=True, text=True).stdout
    names, seen = [], set()
    for tc in ET.fromstring(xml).iter("testcase"):
        name = tc.get("name").split("/")[0]
        ok = tc.find("failure") is None and tc.find("error") is None
        if name in seen:
            names = [(n, o and ok) if n == name else (n, o) for n, o in names]
            continue
        seen.add(name)
        names.append((name, ok))
    summary = subprocess.run([exe], capture_output=True, text=True).stdout.strip().split("\n")[-1]
    return names, summary


def setup(width, height):
    t0 = time.time()
    st = Stage(width, height, renderer="gl", fog=(4.0, 14.0))
    st.look((0.0, 1.6, 6.5), (0.0, 0.4, 0.0), 32)

    # the windows, in order: (key, from, to, title bar)
    t_slide = word_time("v2", "then", 2.4) - 0.1
    t_tab = word_time("c2", "tab", 5.0)
    t_moved = word_time("p3", "eats", 5.0) - 71 / FPS         # the moved trap eats on its frame 71
    win = [("open", 0.0, TL.end("open") + 0.6, "snake  ·  the finished app"),
           ("tangle", TL.start("tangle") + 0.2, TL.end("tangle") + 0.6, "tangle  ·  everything in main()"),
           ("trap", TL.start("trap") + 0.2, t_moved, "trap  ·  class Snake: public Group"),
           ("moved", t_moved, TL.end("trap") + 0.6, "trap  ·  and one line moves the Group"),
           ("term", TL.start("model") + 0.2, TL.end("tests") + 0.6, "terminal"),
           ("s1", TL.start("view") + 0.2, t_slide, "snake  ·  built with STEP=1"),
           ("s2", t_slide, TL.end("view") + 0.6, "snake  ·  built with STEP=2"),
           ("s3", TL.start("players") + 0.2, TL.end("players") + 0.6, "snake  ·  built with STEP=3"),
           ("s4", TL.start("events") + 0.2, TL.end("events") + 0.6, "snake  ·  built with STEP=4"),
           ("s5", TL.start("views") + 0.2, TL.end("views") + 0.6, "snake --ascii  ·  built with STEP=5"),
           ("s6", TL.start("rival") + 0.2, TL.duration, "snake  ·  built with STEP=6")]
    scripts = {}
    for key, a, b, _ in win:
        if key != "term":
            scripts[key] = KeyScript(a, b + 0.6)
    scripts["open"].press(1, "TAB")
    scripts["tangle"].loop("tangle", 5, bug_after=cap0("t3") + 2.6)
    scripts["trap"].loop("tangle", 5)
    moved = scripts["moved"]
    for k, n in ((42, "UP"), (74, "RIGHT")):        # to the apple, then along the row to the wall
        moved.press(k, n)
    scripts["s1"].loop("app", 3)
    scripts["s2"].loop("app", 3)
    scripts["s3"].loop("app", 3, until=t_tab - 0.2)
    scripts["s3"].press_at(t_tab, "TAB")
    for k in ("s4", "s5", "s6"):
        scripts[k].press(1, "TAB")
    names = {"open": "lesson_snake_capture_6", "tangle": "lesson_snake_tangle_capture",
             "trap": "lesson_snake_trap_capture", "moved": "lesson_snake_trap_moved_capture"}
    streams = {}
    for key, s in scripts.items():
        name = names.get(key, f"lesson_snake_capture_{key[1:]}")
        streams[key] = Stream(name, s, log=(key == "s5"), args=["--ascii"] if key == "s5" else ())

    # the tangle and the trap, frame for frame, while the trap is on screen (and before the
    # tangle's double key press, which the trap's script doesn't make)
    k1 = min(scripts["trap"].frames, min(k for k, _ in scripts["tangle"].keys[-2:])) - 1
    same = all(np.array_equal(streams["tangle"].frame(k), streams["trap"].frame(k)) for k in range(0, k1, 30))
    streams["tangle"].close_proc()
    streams["trap"].close_proc()
    print(f"[same game] tangle and trap frames {'match' if same else 'DIFFER'} (every 30th of {k1})")

    # the cards: (t_in, t_out, x, y, title, rows, hl)
    main = source("main.cpp")
    tangle, trap = source("tangle", "main.cpp"), source("trap", "main.cpp")
    game_h, view_h = source("model", "Game.hpp"), source("view", "SnakeView.hpp")
    ctrl_h, key_h = source("control", "Controller.hpp"), source("control", "KeyboardController.hpp")
    test_c = source("tests", "test_game.cpp")
    cmake = source("CMakeLists.txt")

    def rows_of(step):
        return step_rows(main, step)

    def hl_on(rows, needle, t_a, t_b):
        idx = {j for j, (_, l) in enumerate(rows) if needle in l}
        return lambda j, t: envelope(t, t_a, t_b, 0.3, 0.3) if j in idx else 0.0

    loop_rows = code_rows(excerpt(tangle, "Vector3 next = snake[0]", "apple->position.z ="))
    key_rows = code_rows(excerpt(tangle, "KeyAdapter keys(", "canvas.addKeyListener(keys);"))
    trap_rows = code_rows(excerpt(trap, "class Snake: public Group {", "        return false;") + ["    }", "    ..."])
    moved_rows = code_rows(["auto snake = std::make_shared<Snake>();", "scene.add(snake);", "",
                            "// one line, anywhere else in the program:"] +
                           [l.split("//")[0].strip() for l in trap if "snake->position.x = 3" in l])
    game_rows = code_rows(excerpt(game_h, "class Game {", "bool isFree(Vec2i cell) const;", drop=("Listener",)))
    cm_rows = code_rows(excerpt(cmake, "add_library(snake_model", "target_compile_features(snake_model"))
    try_rows = [("code", COMPILE_TRY[0]), ("bad", COMPILE_TRY[1])]
    test_rows = code_rows(excerpt(test_c, 'TEST_CASE("eating grows', "CHECK(game.isFree(game.apple()));") + ["}"])
    view_rows = code_rows(excerpt(view_h, "class SnakeView {", "void pop(Vec2i cell);"))
    ctrl_rows = code_rows(excerpt(ctrl_h, "class Controller {", "virtual Direction next(const Game& game) = 0;") + ["};"])
    key_h_rows = code_rows(excerpt(key_h, "class KeyboardController:", "std::deque<Direction> presses_;") + ["};"])
    ev_rows = code_rows(excerpt(game_h, "// anyone may listen", "void onDie(Listener f);"))
    # step 1: the game and its view, and the loop that ticks one and draws the other
    kept = preprocess(main, 1)
    g0 = next(i for i in kept if "Game game(20" in main[i])
    a0 = next(i for i in kept if "canvas.animate([&] {" in main[i])
    a1 = max(i for i in kept if main[i].rstrip() == "    });")
    step1 = [("code", main[i][4:]) for i in kept if g0 <= i <= g0 + 2] + [("gap", "")]
    step1 += [("code", main[i][4:]) for i in kept if a0 <= i <= a1 and main[i].strip()]
    E = TL.end
    cards = [
        (TL.start("tangle") + 0.3, cap0("t3") - 0.1, 40, 56, "tangle/main.cpp  ·  EVERY TICK, INSIDE THE RENDER LOOP",
         loop_rows, hl_on(loop_rows, "distanceTo", cap0("t2") + 0.8, cap1("t2"))),
        (cap0("t3"), cap0("t4") - 0.1, 40, 56, "tangle/main.cpp  ·  THE KEYS", key_rows,
         hl_on(key_rows, "direction.set", cap0("t3") + 0.6, cap1("t3"))),
        (TL.start("trap") + 0.3, cap0("p3") - 0.1, 40, 56, "trap/main.cpp  ·  THE FIRST TRY AT OBJECTS", trap_rows,
         lambda j, t: envelope(t, cap0("p1") + 1.0, cap1("p2"), 0.3, 0.3) if j == 0 else
         (envelope(t, cap0("p1") + 3.0, cap1("p1"), 0.3, 0.3) if "distanceTo" in trap_rows[j][1] else 0.0)),
        (cap0("p3"), cap0("p4") - 0.1, 40, 56, "trap/main.cpp  ·  -DLESSON_MOVE_GROUP", moved_rows,
         hl_on(moved_rows, "position.x", cap0("p3") + 0.8, cap1("p3"))),
        (TL.start("model") + 0.3, cap0("m2") - 0.1, 40, 56, "model/Game.hpp  ·  THE RULES", game_rows,
         hl_on(game_rows, "void tick();", cap0("m1") + 3.5, cap1("m1"))),
        (cap0("m2"), E("model") - 0.1, 40, 56, "CMakeLists.txt  ·  A LIBRARY OF ITS OWN", cm_rows, None),
        (cap0("m2") + 1.6, E("model") - 0.1, 40, 300, "model/Game.cpp  ·  TRY IT", try_rows, None),
        (TL.start("tests") + 0.3, E("tests") - 0.1, 40, 56, "tests/test_game.cpp  ·  NO WINDOW", test_rows,
         hl_on(test_rows, "game.tick();", cap0("e1") + 3.0, cap1("e1"))),
        (TL.start("view") + 0.3, cap0("v2") - 0.1, 40, 56, "view/SnakeView.hpp", view_rows,
         hl_on(view_rows, "void update(", cap0("v1") + 3.0, cap1("v1"))),
        (cap0("v2"), t_slide + 0.2, 40, 56, "main.cpp  ·  STEP 1", step1, None),
        (t_slide + 0.2, E("view") - 0.1, 40, 56, "main.cpp  ·  STEP 2  ·  WHAT CHANGES", rows_of(2), None),
        (TL.start("players") + 0.3, cap0("c2") - 0.1, 40, 56, "control/Controller.hpp", ctrl_rows,
         hl_on(ctrl_rows, "virtual Direction next", cap0("c1") + 2.0, cap1("c1"))),
        (cap0("c2"), cap0("c3") - 0.1, 40, 56, "main.cpp  ·  STEP 3  ·  WHAT CHANGES", rows_of(3), None),
        (cap0("c3"), E("players") - 0.1, 40, 56, "control/KeyboardController.hpp", key_h_rows,
         hl_on(key_h_rows, "class KeyboardController:", cap0("c3") + 1.5, cap1("c3"))),
        (TL.start("events") + 0.3, E("events") - 0.1, 40, 56, "model/Game.hpp", ev_rows, None),
        (TL.start("events") + 0.9, E("events") - 0.1, 40, 250, "main.cpp  ·  STEP 4  ·  WHAT CHANGES", rows_of(4), None),
        (TL.start("views") + 0.3, E("views") - 0.1, 40, 56, "main.cpp  ·  STEP 5  ·  WHAT CHANGES", rows_of(5), None),
        (TL.start("rival") + 0.3, E("rival") - 0.1, 40, 56, "main.cpp  ·  STEP 6  ·  WHAT CHANGES", rows_of(6), None),
    ]
    for spec in cards:
        print(f"[card] {spec[4]}: {len(spec[5])} rows")

    tree = [("model/", "the rules", None),
            ("Vec2i  Snake  Game", "cells, the snake, one tick", uses_threepp("model", "Game.hpp")),
            ("view/", "how it looks", None),
            ("SnakeView", "3D, a group of meshes", uses_threepp("view", "SnakeView.hpp")),
            ("MinimapView", "one pixel per cell", uses_threepp("view", "MinimapView.hpp")),
            ("AsciiView", "text", uses_threepp("view", "AsciiView.hpp")),
            ("control/", "who plays", None),
            ("Controller", "the interface", uses_threepp("control", "Controller.hpp")),
            ("KeyboardController", "the arrow keys", uses_threepp("control", "KeyboardController.hpp")),
            ("GreedyAI", "heads for the apple", uses_threepp("control", "GreedyAI.hpp")),
            ("tests/", "the rules, the text view, the AI", uses_threepp("tests", "test_game.cpp")),
            ("main.cpp", "wires them together", True)]
    tests = run_tests()
    print(f"[tests] {len(tests[0])} test cases, {tests[1]}")
    view_cls = excerpt(view_h, "class SnakeView {", "std::shared_ptr<threepp::Object3D> root_;")
    is_has = ([next(l for l in trap if l.startswith("class Snake: public Group"))],
              [view_cls[0], "    ...", view_cls[-1]])
    rec = dict(streams=streams, windows=win, cards=cards, tree=tree, tests=tests, same_pixels=same, is_has=is_has)
    ov = Hud(1920, 1080, math_cache=os.path.join(_HERE, "snake_oo.math.json"))
    painter = Painter(st, ov, rec)
    print(f"[setup] {time.time() - t0:.1f}s")

    def render(t):
        ov.begin()
        painter.draw2d(t)
        ov.fade(fade_in_out(t, TL.duration))
        ov.end()
        for key, s in streams.items():         # a program whose part of the film is over can go
            if s.proc is not None and t > s.script.t1 + 1.0:
                s.close_proc()
        return st.frame(t, hud=ov)
    return render, painter.captions


if __name__ == "__main__":
    run("snake_oo", TL.duration, setup, fps=FPS, speech=SPEECH)
