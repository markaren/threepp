"""Snake, written three times: two films about where the code goes, about 4 min each.

    python snake_oo.py                               # Part 0c, "threepp is the view" (1080p60), lesson_out/snake_oo.mp4
    python snake_oo.py --part d                      # Part 0d, "Who plays?", lesson_out/snake_oo_d.mp4
    python snake_oo.py --stills 30,60,95             # individual frames (and the other flags every lesson takes)

Part 0c opens on the game itself (what Snake is), says what the film will do, then writes Snake
three times: all in main() (examples/lesson/snake/tangle), as a class that inherits from
threepp's Group (trap), and with the rules in a model of their own (model/, a library that
doesn't link threepp), tested without a window and drawn by a view. Part 0d carries on with the
finished app's later steps: the view sliding between ticks, players behind an interface, events,
more views of one game, and a second game.

What is on screen was checked against the programs:

* every snippet is read out of those files; a step's rows are the lines that appear (and
  disappear) when the preprocessor runs with STEP = n instead of n - 1;
* every window is that program's own output. Each program (and each step of main.cpp) is built
  as a capture target (lesson_snake_*_capture) and run while the film is made: headless, on a
  fixed 1/60 s clock, with the keys pressed by a script through the canvas's own event path
  (examples/lesson/capture.hpp). The keys drawn under the window are the ones the script presses,
  and the labels in the opening point at the squares the script's route puts things on;
* the text board is what lesson_snake_capture_5 --ascii printed, frame for frame; the test run
  is test_snake's own output, run while the film is made;
* the compile error is what MSVC printed when a threepp #include was added to model/Game.cpp
  and snake_model was built (2026-09-29; shortened only where it carried the checkout's path);
* "same frames": the tangle and trap captures are compared frame by frame while the film is
  made, and the note is only shown if they match.
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


def _part():
    """--part c|d, taken off the command line before run() reads the rest."""
    if "--part" in sys.argv:
        i = sys.argv.index("--part")
        p = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        if p not in ("c", "d"):
            raise SystemExit("--part is c or d")
        return p
    return "c"


PART = _part()
NAME = "snake_oo" if PART == "c" else "snake_oo_d"

FPS = 60
TICK = 0.2                                           # the programs move five squares a second
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

# ── the scripts ───────────────────────────────────────────────────────────────
# The narration sets the pace: each caption lasts as long as its spoken line (Kokoro, af_heart,
# measured in <name>.speech.json), and the beats leave the picture room between lines.
TEXT_C = {
    "title": "threepp is the view. Where the code goes, when a program has to grow.",
    "i1": "This is Snake. The snake moves on its own, one square at a time, and the arrow keys turn it.",
    "i2": "Eat the apple, and it grows. Hit a wall, or its own tail, and the game is over.",
    "i3": "It's a small game, which makes it a good place to ask a big question: where should each part of "
          "the code live?",
    "i4": "We'll write it three times. First, everything in one function. Then, a class that inherits from "
          "threepp. And last, the rules on their own, with threepp only drawing them.",
    "t1": "Here is the first version, written like our first app. Everything lives in main, and it plays.",
    "t2": "Look at how it decides that the snake has eaten: two float positions, less than half a unit apart. "
          "The game's truth lives in the meshes.",
    "t3": "And a key press turns the snake at once. Press two keys between moves, and it turns back into its "
          "own neck.",
    "t4": "Now three requests arrive: test the rules, run them without a window, and add a second player. "
          "Here, every one of them means untangling main.",
    "p1": "The second version is the usual first try at objects: a Snake class that inherits from threepp's "
          "Group. It plays the same game, frame for frame.",
    "p2": "It's tidier. But a snake is now a scene node, and everything a Group has is public: position, "
          "scale, add, remove.",
    "p3": "So one line, anywhere, can move it. Now the snake you see and the snake the rules check have parted "
          "ways. It eats an apple it never touched, and dies off the board.",
    "p4": "A snake isn't a scene node. It has a look. So in the third version, the rules move out, into classes "
          "of their own.",
    "m1": "This is the model: whole squares on a board, a snake, an apple, and one tick of the rules. No floats, "
          "no meshes, no window.",
    "m2": "It's a library of its own, and it doesn't link threepp. Include a threepp header in it, and the build "
          "stops.",
    "e1": "Rules without a window can be tested. A test sets up a game by hand, ticks it once, and checks what "
          "happened.",
    "e2": "The whole suite runs in the blink of an eye. And it pins down the double key press from the first "
          "version.",
    "v1": "Now threepp comes back, as a view. The view holds a group of meshes. Every frame, it reads the game "
          "and puts them in place.",
    "v2": "It never changes the game. The game is the truth, and the view only shows it.",
    "z1": "The model owns the truth. The view only reads it. And threepp stays at the edge, where it draws, but "
          "never decides.",
    "z2": "Next time: who plays the game, how the game tells others what happened, and more than one view of it.",
}
PLAN_C = [("open", 0.4, ["title"], 2.6),
          ("intro", 1.0, ["i1", "i2", "i3", "i4"], 2.2),
          ("tangle", 1.2, ["t1", "t2", "t3", "t4"], 2.0),
          ("trap", 1.2, ["p1", "p2", "p3", "p4"], 2.0),
          ("model", 1.2, ["m1", "m2"], 2.0),
          ("tests", 1.2, ["e1", "e2"], 2.0),
          ("view", 1.2, ["v1", "v2"], 2.4),
          ("end", 1.2, ["z1", "z2"], 1.6),
          ("outro", 0.0, [], 9.0)]

TEXT_D = {
    "title": "Who plays? Players, events, and more than one view of a game.",
    "a1": "Last time, Snake's rules moved into a model of their own, and threepp came back as a view.",
    "a2": "This time, the same program grows three more pieces: who plays, how the game tells others what "
          "happened, and more views of one game.",
    "k1": "First, a detail of the view. The rules tick at their own pace, and the screen draws much faster.",
    "k2": "So the view slides the snake between ticks. Only the view changed. The rules didn't.",
    "c1": "Who plays is an interface: one question, asked once per tick. Which way now?",
    "c2": "The keyboard answers with the keys it has queued. The AI looks at the game and picks. Press tab, and "
          "the loop can't tell the difference.",
    "c3": "This is where inheritance belongs: a contract. The keyboard inherits two, ours and threepp's key "
          "listener, and no code.",
    "ev1": "When the snake eats, the game tells whoever is listening. The view pops a ring.",
    "ev2": "The game never knows who listens, or what they do.",
    "w1": "With the rules on their own, more views are cheap: a minimap in the corner, and the same game as "
          "text, with no threepp at all.",
    "w2": "Three views, one truth. They can't disagree, because none of them owns the game.",
    "r1": "And a second game is just more objects: another game, another view, another player.",
    "r2": "No globals to untangle, nothing to copy. Two AIs, side by side.",
    "z1": "The model owns the truth. Views only read it. Players only say what they want.",
    "z2": "threepp draws. It doesn't decide. Keep it at the edge, and the rest of your program stays yours to "
          "test, change and reuse.",
}
PLAN_D = [("open", 0.4, ["title"], 2.6),
          ("recap", 1.0, ["a1", "a2"], 2.0),
          ("clocks", 1.2, ["k1", "k2"], 2.0),
          ("players", 1.2, ["c1", "c2", "c3"], 2.0),
          ("events", 1.2, ["ev1", "ev2"], 2.0),
          ("views", 1.2, ["w1", "w2"], 2.2),
          ("rival", 1.2, ["r1", "r2"], 2.4),
          ("end", 1.2, ["z1", "z2"], 1.6),
          ("outro", 0.0, [], 9.0)]

CAPTION_TEXT, PLAN = (TEXT_C, PLAN_C) if PART == "c" else (TEXT_D, PLAN_D)
SPEECH = lesson.Speech(CAPTION_TEXT, os.path.join(_HERE, f"{NAME}.speech.json"), words=WORDS,
                       voice_only={"title"})
TL, CAP = SPEECH.layout(PLAN, gap=0.9)


def cap0(k):
    return CAP[k][0]


def cap1(k):
    return CAP[k][1]


if PART == "c":
    TITLE = ("A THREEPP LESSON  ·  PART 0c", "threepp is the view", "Snake, written three times")
    PILLS = [("1  one function", ["tangle"]), ("2  inherits a Group", ["trap"]),
             ("3  rules on their own", ["model", "tests", "view"])]
    SUMMARY = [(r"$\mathrm{model}$", "Rules in plain classes: no threepp, tested without a window."),
               (r"$\mathrm{view}$", "threepp reads the model and draws it. It never changes it."),
               (r"$\mathrm{has\ a}$", "A class has a look. It doesn't inherit one.")]
    FOOTER = "Every window in this film is the program's own output"
else:
    TITLE = ("A THREEPP LESSON  ·  PART 0d", "Who plays?", "Snake: players, events and more views")
    PILLS = [("slide", ["clocks"]), ("players", ["players"]), ("events", ["events"]), ("views", ["views"]),
             ("a second game", ["rival"])]
    SUMMARY = [(r"$\mathrm{player}$", "An interface: the keys, an AI or a test can play."),
               (r"$\mathrm{events}$", "The game says what happened; it never knows who listens."),
               (r"$\mathrm{views}$", "As many as you like, and none of them owns the game."),
               (r"$\mathrm{objects}$", "A second game is one more object, not a copy.")]
    FOOTER = "Every window in this film is the program's own output, one build per step"

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


def code_rows(lines):
    return [("code", l) for l in lines]


# ── keys, pressed by a script, at the moments the program moves ───────────────
def move_frames(kind, n):
    """The capture frames (0-based) on which the program makes its moves 1..n, as its float32
    clock does it. tangle and trap move once their timer is past TICK and start it again from 0;
    the finished app keeps the remainder (a fixed time step)."""
    out, acc = [], np.float32(0)
    dt, tick = np.float32(1) / np.float32(60), np.float32(TICK)
    f = 0
    while len(out) < n:
        acc = np.float32(acc + dt)
        if kind == "tangle":
            if acc > tick:
                out.append(f)
                acc = np.float32(0)
        else:
            while acc >= tick and len(out) < n:
                acc = np.float32(acc - tick)
                out.append(f)
        f += 1
    return out


KEY = {"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT"}
STEP = {"U": (0, 1), "D": (0, -1), "L": (-1, 0), "R": (1, 0)}
LEFT_OF = {"R": "U", "U": "L", "L": "D", "D": "R"}
BACK = {"R": "L", "L": "R", "U": "D", "D": "U"}


def square(first, side, laps):
    """Legs [(heading, moves)]: `first` moves right, then left turns every `side` moves."""
    legs, h = [("R", first)], "R"
    for _ in range(4 * laps):
        h = LEFT_OF[h]
        legs.append((h, side))
    return legs


class KeyScript:
    """The keys one capture presses: events at capture frames, from film time t0 to t1."""

    def __init__(self, t0, t1, kind, step=None):
        self.t0, self.t1, self.kind, self.step = t0, t1, kind, step
        self.keys = []          # (frame, name)
        self.mf = move_frames(kind, self.frames // 11 + 4)
        self.cells = []         # (frame, cell) of the head after each move, when the route knows it

    def frame(self, t):
        return max(0, int(round((t - self.t0) * FPS)))

    def time(self, k):
        return self.t0 + k / FPS

    @property
    def frames(self):
        return self.frame(self.t1) + 1

    def press(self, k, name):
        self.keys.append((int(k), name))

    def press_at(self, t, name):
        self.press(self.frame(t), name)

    def text(self):
        return f"frames {self.frames}\n" + "\n".join(f"{k} key {n}" for k, n in sorted(self.keys))

    def recent(self, t, span=1.0):
        """[(name, age)] of the keys pressed in the last `span` seconds before t."""
        k = (t - self.t0) * FPS
        return [(n, (k - f) / FPS) for f, n in self.keys if 0 <= k - f < span * FPS]

    def drive(self, legs, start="R", head=None, until=None, bug_after=None):
        """Press the keys for a route of legs [(heading, moves)]: each turn goes down a few
        frames before the move it applies to. With `bug_after` (a film time), two keys go down
        between the next two moves: one to the side, then straight back. `head` (a cell) records
        where the head is after each move."""
        h, m = start, 0
        cell = head
        until = self.frames if until is None else self.frame(until)
        for heading, n in legs:
            for j in range(n):
                f = self.mf[m]
                if f >= until:
                    return
                if bug_after is not None and f - 7 >= self.frame(bug_after) and j > 0:
                    self.press(f - 6, KEY[LEFT_OF[h]])
                    self.press(f - 3, KEY[BACK[h]])
                    return
                if heading != h:
                    self.press(f - 3, KEY[heading])
                    h = heading
                if cell is not None:
                    cell = (cell[0] + STEP[h][0], cell[1] + STEP[h][1])
                    self.cells.append((f, cell))
                m += 1

    def head_at(self, t):
        """The head at t as a STEP=1 build draws it (snapped to its square)."""
        k = (t - self.t0) * FPS
        cur = None
        for f, c in self.cells:
            if f <= k:
                cur = c
        return cur


def board_to_capture(cell, y=0.45):
    """A square of the finished app's 20 x 20 board to pixels in its 1280 x 720 capture, through
    the camera main.cpp sets up (50 degrees, from (0, 20, 10) towards (0, 0, 1.5))."""
    p = np.array([cell[0] - 9.5, y, 9.5 - cell[1]], float)
    c, target = np.array([0.0, 20.0, 10.0]), np.array([0.0, 0.0, 1.5])
    f = target - c
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = p - c
    z = d @ f
    th = np.tan(np.radians(50) / 2)
    x, yy = (d @ r) / z / (th * 16 / 9), (d @ u) / z / th
    return (x + 1) / 2 * 1280, (1 - yy) / 2 * 720


# ── the picture ───────────────────────────────────────────────────────────────
WX0, WY0, WX1 = 960, 112, 1880                       # the program's window beside the code
WBAR = 34
WY1 = WY0 + WBAR + (WX1 - WX0) * 9 / 16
DY0 = WY1 + 22                                       # the diagram, under the window
TYPE_S = 0.12                                        # seconds per code row as it types in
OPEN_RECT, BIG_RECT, STD_RECT = (1010, 250, 1850), (300, 70, 1620), (WX0, WY0, WX1)


class Painter:
    def __init__(self, st, ov, rec):
        self.st, self.ov, self.rec = st, ov, rec
        self.streams, self.win = rec["streams"], rec["windows"]
        self.captions = SPEECH.captions(CAP)

    # the window: where it is, and which picture it shows ------------------------
    def window_rect(self, t):
        """The window's (x0, y0, x1), moving between the keyed rectangles over the 1.2 s before each key."""
        keys = self.rec["rect_keys"]
        cur = keys[0][1]
        for (ta, ra), (tb, rb) in zip(keys, keys[1:]):
            if t >= tb:
                cur = rb
            elif t > ta:
                u = smooth(remap(t, tb - 1.2, tb))
                cur = tuple(a + (b - a) * u for a, b in zip(ra, rb))
                break
        return cur

    def window_layers(self, t):
        cur = None
        for key, a, b, _ in self.win:
            if a <= t:
                cur = (key, a)
        if cur is None:
            return []
        idx = [w[0] for w in self.win].index(cur[0])
        if idx > 0 and t < cur[1] + 0.5:
            return [(self.win[idx - 1][0], 1.0), (cur[0], smooth(remap(t, cur[1], cur[1] + 0.5)))]
        return [(cur[0], 1.0)]

    def label(self, key):
        return next(w[3] for w in self.win if w[0] == key)

    def draw_window(self, t):
        ov = self.ov
        alpha = envelope(t, 0.3, TL.end("end") + 0.6, 0.9, 0.8)
        if alpha <= 0.003:
            return None
        x0, y0, x1 = self.window_rect(t)
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
        big = (x1 - x0) > 1100
        # the keys the script presses
        if layers and layers[-1][0] != "term":
            s = self.streams[layers[-1][0]].script
            px = x0 + 16
            size, hh = (24, 40) if big else (17, 32)
            for name, age in s.recent(t):
                ka = alpha * (1 - smooth(remap(age, 0.6, 1.0)))
                w = ov.text_width(name, size, "semibold") + 30
                py = y1 - hh - 12 + 10 * (1 - smooth(remap(age, 0, 0.15)))
                ov.panel(px, py, w, hh, radius=8, fill=0x0d131e, alpha=0.85 * ka, outline=C_WARM, outline_alpha=0.8)
                ov.text(px + w / 2, py + hh / 2, name, size=size, color=C_WARM, alpha=ka, kind="semibold",
                        anchor="mm")
                px += w + 8
        # the switch to a new step
        for key, a, _, lab in self.win:
            if not (len(key) == 2 and key[0] == "s" and key[1].isdigit()) or a < TL.end("open"):
                continue
            fa = alpha * envelope(t, a - 0.05, a + 1.8, 0.2, 0.6)
            if fa > 0.003:
                txt = f"STEP {key[1]}  ·  built and running"
                w = ov.text_width(txt, 22, "semibold") + 48
                cx, cy = (x0 + x1) / 2, y0 + WBAR + 40
                ov.panel(cx - w / 2, cy - 22, w, 44, radius=22, fill=0x0d131e, alpha=0.85 * fa)
                ov.text(cx, cy, txt, size=22, color=C_ACCENT, alpha=fa, kind="semibold", anchor="mm")
        return x0, y0, x1, y1

    def to_screen(self, t, cell, y=0.45):
        x0, y0, x1 = self.window_rect(t)
        px, py = board_to_capture(cell, y)
        s = (x1 - x0 - 4) / 1280
        return x0 + 2 + px * s, y0 + WBAR + 2 + py * s

    def terminal(self, t, x, y, w, h, a):
        """The window as a terminal: the compile error, then the test run."""
        ov = self.ov
        ov.panel(x, y, w, h, radius=0, fill=0x0a0e14, alpha=a)
        size, lh = 17, 25
        if t < TL.start("tests"):
            ca = a * smooth(remap(t, cap0("m2") + 3.0, cap0("m2") + 3.4))
            lines = [(l, C_BLUE if l.startswith(">") else (C_DEL if "error" in l or "FAILED" in l else TEXT))
                     for l in COMPILE_OUT]
            for j, (l, c) in enumerate(lines):
                la = ca * smooth(remap(t, cap0("m2") + 3.2 + 0.35 * j, cap0("m2") + 3.4 + 0.35 * j))
                ov.text(x + 24, y + 30 + j * lh, l, size=size, color=c, alpha=la, kind="mono", anchor="lm")
            if ca < 0.01:
                ov.text(x + 24, y + 30, ">", size=size, color=C_BLUE, alpha=a, kind="mono", anchor="lm")
            return
        names, summary = self.rec["tests"]
        t0 = cap0("e1") + 1.5
        ov.text(x + 24, y + 30, "> test_snake", size=size, color=C_BLUE, alpha=a * smooth(remap(t, t0, t0 + 0.3)),
                kind="mono", anchor="lm")
        for j, (name, ok) in enumerate(names):
            la = a * smooth(remap(t, t0 + 0.8 + 0.05 * j, t0 + 0.9 + 0.05 * j))
            yy = y + 30 + (j + 1) * 24
            ov.text(x + 24, yy, "passed" if ok else "FAILED", size=16, color=C_OK if ok else C_DEL, alpha=la,
                    kind="mono", anchor="lm")
            ov.text(x + 110, yy, name, size=16, color=TEXT, alpha=la, kind="mono", anchor="lm")
        la = a * smooth(remap(t, t0 + 1.0 + 0.05 * len(names), t0 + 1.2 + 0.05 * len(names)))
        ov.text(x + 24, y + 30 + (len(names) + 1.4) * 24, summary, size=16, color=C_OUT, alpha=la, kind="mono",
                anchor="lm")
        # the test that covers the double key press (its sections are listed under the test case)
        hl = smooth(remap(t, cap0("e2") + 3.0, cap0("e2") + 3.6)) * (1 - smooth(remap(t, TL.end("tests") - 0.5,
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
        ov.title_card(t, *TITLE, C_ACCENT, t_in=0.6, t_out=TL.end("open") - 0.4, size=84)
        geo = self.draw_window(t)
        self.pills(t)
        for spec in self.rec["cards"]:
            self.rows_card(t, *spec)
        if geo is not None:
            self.under_window(t, geo)
        for f in self.rec["widgets"]:
            f(self, t)
        ov.captions(t, self.captions)
        ov.summary(t, TL.start("outro"), TL.end("outro"), SUMMARY, FOOTER, C_ACCENT, C_ACCENT, text_dx=300)

    def pills(self, t):
        ov = self.ov
        a = envelope(t, TL.start(PILLS[0][1][0]) - 0.6, TL.end("end"), 0.6, 0.6)
        if a <= 0.003:
            return
        px = WX0
        for label, beats in PILLS:
            seen = t >= TL.start(beats[0]) - 0.1
            on = any(TL.start(b) <= t < TL.end(b) for b in beats)
            wpx = ov.text_width(label, 17, "semibold") + 28
            ov.panel(px, 58, wpx, 34, radius=17, fill=0x3d6b2e if on else 0x223452,
                     alpha=a * (0.95 if on else 0.55 if seen else 0.3))
            ov.text(px + wpx / 2, 75, label, size=17, color=TEXT if seen else DIM, alpha=a * (1.0 if seen else 0.45),
                    kind="semibold", anchor="mm")
            px += wpx + 8

    def rows_card(self, t, t_in, t_out, x, y, title, rows, hl=None, size=18, lh=24):
        """A card of source rows typing in from t_in: ('code', text) plain, ('add', text) with a
        green bar, ('bad', text) with a red one, ('del', text) struck out, ('ctx', text) dim,
        ('gap', ''). hl(j, t) -> 0..1 highlights row j."""
        ov = self.ov
        a = envelope(t, t_in, t_out, 0.8, 0.6)
        if a <= 0.003:
            return
        longest = max(len(r[1]) for r in rows)
        while size > 13 and longest * ov.text_width("0", size, "mono") + 64 > WX0 - 60 - x:
            size, lh = size - 1, lh - 1
        cw = ov.text_width("0", size, "mono")
        w = max(longest * cw + 64, 560)
        h = 74 + lh * len(rows)
        card(ov, x, y, w, h, a, title)
        t_type = t_in + 0.6
        for j, (kind, text) in enumerate(rows):
            if not text.strip():
                continue
            yy = y + 58 + j * lh
            ra = a * smooth(remap(t, t_type + TYPE_S * j, t_type + TYPE_S * j + 0.4))
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

    # under the window ----------------------------------------------------------
    def under_window(self, t, geo):
        x0, y0, x1, y1 = geo
        ov = self.ov
        if "p2" in CAP:
            # everything a Group has, public on Snake
            a = envelope(t, cap0("p2") + 1.5, cap1("p3") - 0.2, 0.6, 0.6)
            if a > 0.003:
                card(ov, WX0, DY0, WX1 - WX0, 150, a, "INHERITED FROM GROUP, PUBLIC ON EVERY SNAKE")
                px, py = WX0 + 28, DY0 + 62
                for k, m in enumerate(GROUP_MEMBERS):
                    ma = a * smooth(remap(t, cap0("p2") + 1.8 + 0.15 * k, cap0("p2") + 2.1 + 0.15 * k))
                    w = ov.text_width(m, 19, "mono") + 26
                    if px + w > WX1 - 20:
                        px, py = WX0 + 28, py + 44
                    ov.panel(px, py - 16, w, 34, radius=8, fill=0x223452, alpha=0.8 * ma)
                    ov.text(px + w / 2, py + 1, m, size=19, color=TEXT, alpha=ma, kind="mono", anchor="mm")
                    px += w + 10
            # same game, frame for frame
            a = envelope(t, cap0("p1") + 4.0, cap0("p3"), 0.6, 0.6)
            if a > 0.003 and self.rec["same_pixels"]:
                txt = "same keys, same frames as the first version"
                w = ov.text_width(txt, 19, "semibold") + 40
                ov.panel(x1 - w - 16, y0 + WBAR + 16, w, 38, radius=19, fill=0x0d131e, alpha=0.85 * a)
                ov.text(x1 - w / 2 - 16, y0 + WBAR + 35, txt, size=19, color=C_ACCENT, alpha=a, kind="semibold",
                        anchor="mm")
        self.diagram(t)

    def diagram(self, t):
        """Who includes whom, built up beat by beat: the model in the middle, threepp at the edge.
        The arrows are #includes (checked in setup against the files)."""
        spec = self.rec["diagram"]
        ov = self.ov
        a = envelope(t, spec["t0"], TL.end("end") + 0.3, 0.6, 0.6)
        if a <= 0.003:
            return
        y = DY0
        card(ov, WX0, y, WX1 - WX0, 196, a, "WHO INCLUDES WHOM")
        boxes = {k: (bx, y + dy, bw, name, t_on) for k, (bx, dy, bw, name, t_on) in spec["boxes"].items()}
        lit = spec["lit"](t)
        for k, (bx, by, bw, name, t_on) in boxes.items():
            ba = a * smooth(remap(t, t_on + 0.2, t_on + 0.8))
            if ba <= 0.003:
                continue
            on = k == lit
            ov.panel(bx, by - 22, bw, 44, radius=10,
                     fill=0x3d6b2e if on else (0x1d3354 if k == "threepp" else 0x223452),
                     alpha=0.95 * ba, outline=C_ACCENT if on else 0x8aa0c0, outline_alpha=0.6 if on else 0.2)
            ov.text(bx + bw / 2, by, name, size=20, color=TEXT, alpha=ba, kind="semibold", anchor="mm")
        for k0, k1, label, t_on, color, cut, below in spec["arrows"]:
            ea = a * smooth(remap(t, t_on, t_on + 0.6))
            if ea <= 0.003:
                continue
            bx0, by0, bw0, _, _ = boxes[k0]
            bx1, by1, bw1, _, _ = boxes[k1]
            p0 = (bx0 + bw0 + 4, by0) if bx1 > bx0 else (bx0 - 4, by0)
            p1 = (bx1 - 6, by1) if bx1 > bx0 else (bx1 + bw1 + 6, by1)
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            if cut:
                ov.dashed(p0, p1, C_DEL, 2.2, 0.8 * ea, dash=8, gap=6)
                ov.line([(mx - 9, my - 9), (mx + 9, my + 9)], C_DEL, 3.0, ea)
                ov.line([(mx - 9, my + 9), (mx + 9, my - 9)], C_DEL, 3.0, ea)
                ov.text(mx, my + 18, label, size=15, color=C_DEL, alpha=ea, anchor="mm")
            else:
                ov.arrow2d(p0, p1, color, 2.2, 10, ea)
                ov.text(mx, my + (16 if below else -14), label, size=15, color=DIM, alpha=ea, anchor="mm")


# ── widgets that belong to one film ───────────────────────────────────────────
def w_intro(p, t):
    """The opening: what is on the board, then what the film will do."""
    ov = p.ov
    sc = p.streams["intro"].script
    # the snake, the apple, a wall: labels that follow the words
    head = sc.head_at(t)
    a = envelope(t, p.rec["t_snake"], cap1("i1") - 0.2, 0.4, 0.5)
    if a > 0.003 and head is not None:
        hx, hy = p.to_screen(t, head, 0.9)
        ov.callout((hx, hy), (hx - 160, hy - 80), "the snake", C_ACCENT, a, size=30)
    a = envelope(t, p.rec["t_apple"] - 0.1, p.rec["t_eat"], 0.3, 0.3)
    if a > 0.003:
        ax, ay = p.to_screen(t, p.rec["apple"], 0.8)
        ov.callout((ax, ay), (ax + 160, ay - 70), "the apple", 0xff6b6b, a, size=30)
    a = envelope(t, p.rec["t_wall"] - 0.1, cap1("i2") - 0.2, 0.3, 0.5)
    if a > 0.003:
        wx, wy = p.to_screen(t, (16.5, 19.5), 0.0)
        ov.callout((wx, wy), (wx + 180, wy - 40), "a wall", C_WARM, a, size=30)
    # the plan
    a = envelope(t, cap0("i4"), TL.end("intro") + 0.4, 0.6, 0.6)
    if a > 0.003:
        x, y = 40, 120
        card(ov, x, y, 860, 380, a, "SNAKE, THREE TIMES")
        rows = [("1", "everything in one function", "all of it in main()"),
                ("2", "a class that inherits from threepp", "class Snake : public Group"),
                ("3", "the rules on their own", "and threepp only draws them")]
        for k, (n, what, how) in enumerate(rows):
            ra = a * smooth(remap(t, p.rec["t_plan"][k], p.rec["t_plan"][k] + 0.5))
            yy = y + 110 + 92 * k
            ov.circle(x + 58, yy, 22, fill=0x3d6b2e, alpha=ra)
            ov.text(x + 58, yy, n, size=24, color=TEXT, alpha=ra, kind="semibold", anchor="mm")
            ov.text(x + 100, yy - 12, what, size=30, color=TEXT, alpha=ra, kind="semibold", anchor="lm")
            ov.text(x + 100, yy + 22, how, size=20, color=DIM, alpha=ra, kind="mono" if k == 1 else "regular",
                    anchor="lm")


def w_recap(p, t):
    """Part 0d's start: what Part 0c did, then what this film adds."""
    ov = p.ov
    x = 40
    a = envelope(t, cap0("a1") + 0.3, TL.end("recap") - 0.1, 0.6, 0.6)
    if a > 0.003:
        card(ov, x, 56, 860, 250, a, "LAST TIME, IN PART 0c")
        rows = ["the rules in a model of their own, with no threepp", "tested without a window",
                "threepp back as a view, which only reads the game"]
        for k, row in enumerate(rows):
            ra = a * smooth(remap(t, cap0("a1") + 1.2 + 1.2 * k, cap0("a1") + 1.7 + 1.2 * k))
            ov.circle(x + 50, 56 + 100 + 52 * k, 6, fill=C_OK, alpha=ra)
            ov.text(x + 72, 56 + 100 + 52 * k, row, size=26, color=TEXT, alpha=ra, anchor="lm")
    a = envelope(t, cap0("a2") + 0.3, TL.end("recap") - 0.1, 0.6, 0.6)
    if a > 0.003:
        card(ov, x, 340, 860, 250, a, "THIS TIME")
        for k, (row, t_on) in enumerate(zip(["who plays: players behind an interface",
                                             "events: the game says what happened",
                                             "more views of one game, and a second game"], p.rec["t_this"])):
            ra = a * smooth(remap(t, t_on, t_on + 0.5))
            ov.circle(x + 50, 340 + 100 + 52 * k, 6, fill=C_WARM, alpha=ra)
            ov.text(x + 72, 340 + 100 + 52 * k, row, size=26, color=TEXT, alpha=ra, anchor="lm")


def w_requests(p, t):
    ov = p.ov
    a = envelope(t, cap0("t4"), TL.end("tangle") - 0.1, 0.6, 0.6)
    if a <= 0.003:
        return
    x, y = 40, 56
    card(ov, x, y, 860, 360, a, "THREE REQUESTS")
    rows = [("test the rules", "they run inside main's render loop"),
            ("run without a window", "the game's state is the meshes"),
            ("add a second player", "one of everything, in one function")]
    for k, (req, why) in enumerate(rows):
        ra = a * smooth(remap(t, cap0("t4") + 1.6 + 1.1 * k, cap0("t4") + 2.1 + 1.1 * k))
        yy = y + 90 + 84 * k
        ov.text(x + 40, yy, req, size=30, color=TEXT, alpha=ra, kind="semibold", anchor="lm")
        ov.text(x + 40, yy + 34, why, size=21, color=C_DEL, alpha=ra, anchor="lm")


def w_is_has(p, t):
    ov = p.ov
    a = envelope(t, cap0("p4"), TL.end("trap") - 0.1, 0.6, 0.6)
    if a <= 0.003:
        return
    x, y = 40, 56
    card(ov, x, y, 860, 300, a, "IS A, OR HAS A")
    cw = ov.text_width("0", 20, "mono")
    is_a, has_a = p.rec["is_has"]
    rows = [(is_a, "trap/main.cpp: a snake is a scene node", C_DEL, 0.4),
            (has_a, "view/SnakeView.hpp: a view has one, and the snake has none", C_OK, 2.6)]
    yy = y + 80
    for k, (code, why, c, dt) in enumerate(rows):
        ra = a * smooth(remap(t, cap0("p4") + dt, cap0("p4") + dt + 0.6))
        for j, line in enumerate(code):
            for col, s, cc in tokenize(line, "cpp", CPP_KEYWORDS):
                ov.text(x + 40 + col * cw, yy + 28 * j, s, size=20, color=cc, alpha=ra, kind="mono", anchor="lm")
            if k == 0:
                ov.line([(x + 34, yy + 28 * j), (x + 46 + len(line) * cw, yy + 28 * j)], C_DEL, 2.5, ra)
        yy += 28 * len(code) + 12
        ov.text(x + 40, yy, why, size=21, color=c, alpha=ra, anchor="lm")
        yy += 64


def w_ascii(p, t):
    """What lesson_snake_capture_5 --ascii printed: its latest board."""
    ov = p.ov
    a = envelope(t, cap0("w1") + 3.5, TL.end("views") - 0.1, 0.6, 0.6)
    if a <= 0.003:
        return
    board = [l for l in p.streams["s5"].printed(t) if l.startswith("#")][-22:]
    x, y = 40, 470
    size, lh = 14, 15
    cw = ov.text_width("0", size, "mono")
    card(ov, x, y, max(23 * cw + 150, 420), 60 + lh * 22, a, "STDOUT  ·  lesson_snake --ascii")
    for j, line in enumerate(board):
        ov.text(x + 30, y + 52 + j * lh, line, size=size, color=C_OUT, alpha=a, kind="mono", anchor="lm")


def w_tree(p, t):
    ov = p.ov
    a = envelope(t, TL.start("end") + 0.2, TL.end("end") + 0.4, 0.6, 0.6)
    if a <= 0.003:
        return
    x, y = 40, 56
    rows = p.rec["tree"]
    card(ov, x, y, 860, 90 + 34 * len(rows), a, "examples/lesson/snake")
    for k, (name, desc, tp_) in enumerate(rows):
        ra = a * smooth(remap(t, TL.start("end") + 0.5 + 0.15 * k, TL.start("end") + 0.9 + 0.15 * k))
        yy = y + 66 + 34 * k
        indent = 0 if name.endswith("/") or name == "main.cpp" else 34
        if tp_ is not None:
            ov.circle(x + 44 + indent, yy, 6, fill=C_BLUE if tp_ else C_OK, alpha=ra)
        ov.text(x + 60 + indent, yy, name, size=20, color=TEXT, alpha=ra, kind="mono", anchor="lm")
        ov.text(x + 440, yy, desc, size=19, color=DIM, alpha=ra, anchor="lm")
    la = a * smooth(remap(t, TL.start("end") + 2.4, TL.start("end") + 3.0))
    yy = y + 66 + 34 * len(rows) + 4
    ov.circle(x + 44, yy, 6, fill=C_OK, alpha=la)
    ov.text(x + 60, yy, "no threepp", size=19, color=DIM, alpha=la, anchor="lm")
    ov.circle(x + 230, yy, 6, fill=C_BLUE, alpha=la)
    ov.text(x + 246, yy, "includes threepp", size=19, color=DIM, alpha=la, anchor="lm")


# ── the stage ─────────────────────────────────────────────────────────────────
def word_time(key, word, default, nth=1):
    """Film time at which the nth `word` of caption `key` is heard (Kokoro's alignment), else
    `default` seconds into the caption."""
    try:
        seen = 0
        for w, s, _ in lesson.Narration(words=SPEECH.words, cache_dir="voice_cache").word_times(CAPTION_TEXT[key]):
            if w.lower().strip(".,:?") == word:
                seen += 1
                if seen == nth:
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


STREAM_OF = {"tangle": "lesson_snake_tangle_capture", "trap": "lesson_snake_trap_capture",
             "moved": "lesson_snake_trap_moved_capture"}


def make_streams(scripts, logged=()):
    streams = {}
    for key, s in scripts.items():
        name = STREAM_OF.get(key) or f"lesson_snake_capture_{s.step}"
        streams[key] = Stream(name, s, log=key in logged, args=["--ascii"] if key in logged else ())
    return streams


def hl_on(rows, needle, t_a, t_b):
    idx = {j for j, (_, l) in enumerate(rows) if needle in l}
    return lambda j, t: envelope(t, t_a, t_b, 0.4, 0.4) if j in idx else 0.0


def setup_c():
    rec = {}
    main = source("main.cpp")
    tangle, trap = source("tangle", "main.cpp"), source("trap", "main.cpp")
    game_h, view_h = source("model", "Game.hpp"), source("view", "SnakeView.hpp")
    test_c, cmake = source("tests", "test_game.cpp"), source("CMakeLists.txt")

    # The opening game: circle near the start, eat the apple as it is named, then run into a wall
    # as that is named. The finished app starts its head at (10, 10), heading right, and (seed 7)
    # its first apple at (13, 12): three right and two up.
    t_snake = word_time("i1", "snake", 1.0, nth=2)
    t_apple = word_time("i2", "apple", 0.6)
    t_wall = word_time("i2", "wall", 2.0)
    lap = [("D", 3), ("L", 3), ("U", 3), ("R", 3)]              # round and back to (10, 10), heading right
    # start the program early enough that, after whole laps, it eats just as "apple" is said
    t_eat = t_apple + 1.0
    probe = move_frames("app", 400)
    k_eat = next(k for k in range(30) if probe[12 * k + 4] / FPS >= t_eat - TL.start("intro"))
    eat_move = 12 * k_eat + 5
    intro = KeyScript(t_eat - probe[eat_move - 1] / FPS, cap0("i3") + 1.0, "app", step=1)
    after = [("U", 3), ("R", 3), ("D", 3), ("L", 3)]            # round and back to the apple's square
    m_after = max(0, int(round((t_wall + 0.6 - t_eat - 8 * TICK) / (12 * TICK))))
    intro.drive(lap * k_eat + [("R", 3), ("U", 2)] + after * m_after + [("U", 8)], head=(10, 10))
    t_die = intro.time(intro.mf[eat_move + 12 * m_after + 8 - 1])
    print(f"[intro] eats at {t_eat:.1f} s ('apple' at {t_apple:.1f}), hits the wall at {t_die:.1f} s "
          f"('wall' at {t_wall:.1f})")
    calm = KeyScript(cap0("i3") + 0.3, TL.end("intro") + 1.2, "app", step=1)
    calm.drive(lap * 30)
    rec.update(t_snake=t_snake, t_apple=t_apple, t_wall=t_wall, t_eat=t_eat, apple=(13, 12),
               t_plan=[word_time("i4", w, d) for w, d in (("first", 2.0), ("then", 4.5), ("last", 7.5))])

    tangle_s = KeyScript(TL.start("tangle"), TL.end("tangle") + 1.2, "tangle")
    tangle_s.drive(square(5, 6, 20), bug_after=word_time("t3", "press", 2.0) + 0.6)
    trap_s = KeyScript(TL.start("trap"), TL.end("trap") + 1.2, "tangle")
    trap_s.drive(square(5, 6, 20))
    # the moved trap eats on its ninth move (UP from the sixth): start it so that lands on "eats"
    eat_f = move_frames("tangle", 9)[8]
    t_moved = word_time("p3", "eats", 6.0) - eat_f / FPS
    moved = KeyScript(t_moved, TL.end("trap") + 1.2, "tangle")
    moved.drive([("R", 5), ("U", 4), ("R", 12)])
    view_s = KeyScript(TL.start("view"), TL.duration + 0.6, "app", step=1)
    view_s.drive(square(3, 6, 40))
    open_s = KeyScript(0.0, TL.end("open") + 1.2, "app", step=1)
    open_s.drive(lap * 5)
    win = [("open", 0.0, TL.end("open") + 0.6, "snake"),
           ("intro", TL.start("intro"), cap0("i3") + 0.3, "snake"),
           ("calm", cap0("i3") + 0.3, TL.end("intro") + 0.6, "snake"),
           ("tangle", TL.start("tangle"), TL.end("tangle") + 0.6, "version 1  ·  everything in main()"),
           ("trap", TL.start("trap"), t_moved, "version 2  ·  class Snake : public Group"),
           ("moved", t_moved, TL.end("trap") + 0.6, "version 2  ·  and one line moves the Group"),
           ("term", TL.start("model"), TL.end("tests") + 0.6, "terminal"),
           ("s1", TL.start("view"), TL.duration, "version 3  ·  snake, built with STEP=1")]
    scripts = {"open": open_s, "intro": intro, "calm": calm, "tangle": tangle_s, "trap": trap_s, "moved": moved,
               "s1": view_s}
    streams = make_streams(scripts)

    # the tangle and the trap, frame for frame (before the tangle's double key press)
    k1 = min(trap_s.frames, min(k for k, _ in tangle_s.keys[-2:])) - 1
    same = all(np.array_equal(streams["tangle"].frame(k), streams["trap"].frame(k)) for k in range(0, k1, 30))
    streams["tangle"].close_proc()
    streams["trap"].close_proc()
    print(f"[same game] tangle and trap frames {'match' if same else 'DIFFER'} (every 30th of {k1})")

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
    view_rows = code_rows(excerpt(view_h, "class SnakeView {", "void pop(Vec2i cell);", drop=("pop", "a ring")))
    kept = preprocess(main, 1)
    g0 = next(i for i in kept if "Game game(20" in main[i])
    a0 = next(i for i in kept if "canvas.animate([&] {" in main[i])
    a1 = max(i for i in kept if main[i].rstrip() == "    });")
    step1 = [("code", main[i][4:]) for i in kept if g0 <= i <= g0 + 2] + [("gap", "")]
    step1 += [("code", main[i][4:]) for i in kept if a0 <= i <= a1 and main[i].strip()]
    E = TL.end
    rec["cards"] = [
        (TL.start("tangle") + 0.3, cap0("t3") - 0.1, 40, 56, "tangle/main.cpp  ·  EVERY MOVE, INSIDE THE RENDER LOOP",
         loop_rows, hl_on(loop_rows, "distanceTo", cap0("t2") + 1.0, cap1("t2"))),
        (cap0("t3"), cap0("t4") - 0.1, 40, 56, "tangle/main.cpp  ·  THE KEYS", key_rows,
         hl_on(key_rows, "direction.set", cap0("t3") + 0.8, cap1("t3"))),
        (TL.start("trap") + 0.3, cap0("p3") - 0.1, 40, 56, "trap/main.cpp  ·  THE FIRST TRY AT OBJECTS", trap_rows,
         lambda j, t: envelope(t, cap0("p1") + 1.5, cap1("p2"), 0.4, 0.4) if j == 0 else 0.0),
        (cap0("p3"), cap0("p4") - 0.1, 40, 56, "trap/main.cpp  ·  -DLESSON_MOVE_GROUP", moved_rows,
         hl_on(moved_rows, "position.x", cap0("p3") + 1.0, cap1("p3"))),
        (TL.start("model") + 0.3, cap0("m2") - 0.1, 40, 56, "model/Game.hpp  ·  THE RULES", game_rows,
         hl_on(game_rows, "void tick();", cap0("m1") + 4.0, cap1("m1"))),
        (cap0("m2"), E("model") - 0.1, 40, 56, "CMakeLists.txt  ·  A LIBRARY OF ITS OWN", cm_rows, None),
        (cap0("m2") + 2.2, E("model") - 0.1, 40, 300, "model/Game.cpp  ·  TRY IT", try_rows, None),
        (TL.start("tests") + 0.3, E("tests") - 0.1, 40, 56, "tests/test_game.cpp  ·  NO WINDOW", test_rows,
         hl_on(test_rows, "game.tick();", cap0("e1") + 3.5, cap1("e1"))),
        (TL.start("view") + 0.3, cap0("v2") + 1.0, 40, 56, "view/SnakeView.hpp", view_rows,
         hl_on(view_rows, "void update(", cap0("v1") + 4.0, cap1("v1"))),
        (cap0("v2") + 1.0, E("view") - 0.1, 40, 56, "main.cpp  ·  STEP 1  ·  THE GAME AND ITS VIEW", step1,
         lambda j, t: envelope(t, cap0("v2") + 2.5, E("view"), 0.4, 0.4)
         if ("view.update" in step1[j][1] or "game.tick" in step1[j][1]) else 0.0),
    ]
    view_cls = excerpt(view_h, "class SnakeView {", "std::shared_ptr<threepp::Object3D> root_;")
    rec["is_has"] = ([next(l for l in trap if l.startswith("class Snake: public Group"))],
                     [view_cls[0], "    ...", view_cls[-1]])
    rec["tree"] = [("model/", "the rules", None),
                   ("Vec2i  Snake  Game", "squares, the snake, one tick", uses_threepp("model", "Game.hpp")),
                   ("view/", "how it looks", None),
                   ("SnakeView", "3D, a group of meshes", uses_threepp("view", "SnakeView.hpp")),
                   ("tests/", "the rules, without a window", uses_threepp("tests", "test_game.cpp")),
                   ("main.cpp", "wires them together", True)]
    rec["tests"] = run_tests()
    print(f"[tests] {len(rec['tests'][0])} test cases, {rec['tests'][1]}")
    beats = {"model": "model", "tests": "tests", "view": "view"}
    rec["diagram"] = dict(
        t0=TL.start("model") + 0.5,
        boxes={"tests": (990, 100, 150, "tests", TL.start("tests")),
               "model": (1210, 100, 170, "model", TL.start("model")),
               "view": (1450, 58, 170, "view", TL.start("view")),
               "threepp": (1700, 100, 150, "threepp", TL.start("model"))},
        arrows=[("model", "threepp", "can't include", cap0("m2") + 3.5, None, True, False),
                ("tests", "model", "checks", cap0("e1") + 0.8, 0x8aa0c0, False, False),
                ("view", "model", "reads", cap0("v1") + 2.0, C_ACCENT, False, False),
                ("view", "threepp", "draws with", cap0("v1") + 3.0, C_BLUE, False, False)],
        lit=lambda t: next((v for b, v in beats.items() if TL.start(b) <= t < TL.end(b)), None))
    rec["widgets"] = [w_intro, w_requests, w_is_has, w_tree]
    rec["rect_keys"] = [(0.0, OPEN_RECT), (TL.end("open") - 0.2, OPEN_RECT), (TL.start("intro") + 1.0, BIG_RECT),
                        (cap0("i4") - 0.4, BIG_RECT), (cap0("i4") + 0.8, STD_RECT)]
    rec.update(streams=streams, windows=win, same_pixels=same)
    return rec


def setup_d():
    rec = {}
    main = source("main.cpp")
    game_h = source("model", "Game.hpp")
    ctrl_h, key_h = source("control", "Controller.hpp"), source("control", "KeyboardController.hpp")

    t_tab = word_time("c2", "tab", 5.0)
    t_slide = word_time("k2", "slides", 1.0) - 0.2
    win = [("open", 0.0, TL.end("open") + 0.6, "snake"),
           ("s1", TL.start("recap"), t_slide, "snake, built with STEP=1"),
           ("s2", t_slide, TL.end("clocks") + 0.6, "snake, built with STEP=2"),
           ("s3", TL.start("players"), TL.end("players") + 0.6, "snake, built with STEP=3"),
           ("s4", TL.start("events"), TL.end("events") + 0.6, "snake, built with STEP=4"),
           ("s5", TL.start("views"), TL.end("views") + 0.6, "snake --ascii, built with STEP=5"),
           ("s6", TL.start("rival"), TL.duration, "snake, built with STEP=6")]
    scripts = {key: KeyScript(a, b + 0.6, "app", step=6 if key == "open" else int(key[1])) for key, a, b, _ in win}
    scripts["open"].press(1, "TAB")
    for k in ("s1", "s2"):
        scripts[k].drive(square(3, 6, 40))
    scripts["s3"].drive(square(3, 6, 40), until=t_tab - 0.2)
    scripts["s3"].press_at(t_tab, "TAB")
    for k in ("s4", "s5", "s6"):
        scripts[k].press(1, "TAB")
    streams = make_streams(scripts, logged=("s5",))

    ctrl_rows = code_rows(excerpt(ctrl_h, "class Controller {", "virtual Direction next(const Game& game) = 0;") + ["};"])
    key_h_rows = code_rows(excerpt(key_h, "class KeyboardController:", "std::deque<Direction> presses_;") + ["};"])
    ev_rows = code_rows(excerpt(game_h, "// anyone may listen", "void onDie(Listener f);"))
    E = TL.end
    rec["cards"] = [
        (TL.start("clocks") + 0.3, E("clocks") - 0.1, 40, 56, "main.cpp  ·  STEP 2  ·  WHAT CHANGES", step_rows(main, 2),
         None),
        (TL.start("players") + 0.3, cap0("c2") - 0.1, 40, 56, "control/Controller.hpp", ctrl_rows,
         hl_on(ctrl_rows, "virtual Direction next", cap0("c1") + 2.5, cap1("c1"))),
        (cap0("c2"), cap0("c3") - 0.1, 40, 56, "main.cpp  ·  STEP 3  ·  WHAT CHANGES", step_rows(main, 3), None),
        (cap0("c3"), E("players") - 0.1, 40, 56, "control/KeyboardController.hpp", key_h_rows,
         hl_on(key_h_rows, "class KeyboardController:", cap0("c3") + 2.0, cap1("c3"))),
        (TL.start("events") + 0.3, E("events") - 0.1, 40, 56, "model/Game.hpp", ev_rows, None),
        (TL.start("events") + 1.2, E("events") - 0.1, 40, 250, "main.cpp  ·  STEP 4  ·  WHAT CHANGES",
         step_rows(main, 4), None),
        (TL.start("views") + 0.3, E("views") - 0.1, 40, 56, "main.cpp  ·  STEP 5  ·  WHAT CHANGES", step_rows(main, 5),
         None),
        (TL.start("rival") + 0.3, E("rival") - 0.1, 40, 56, "main.cpp  ·  STEP 6  ·  WHAT CHANGES", step_rows(main, 6),
         None),
    ]
    rec["tree"] = [("model/", "the rules", None),
                   ("Vec2i  Snake  Game", "squares, the snake, one tick", uses_threepp("model", "Game.hpp")),
                   ("view/", "how it looks", None),
                   ("SnakeView", "3D, a group of meshes", uses_threepp("view", "SnakeView.hpp")),
                   ("MinimapView", "one pixel per square", uses_threepp("view", "MinimapView.hpp")),
                   ("AsciiView", "text", uses_threepp("view", "AsciiView.hpp")),
                   ("control/", "who plays", None),
                   ("Controller", "the interface", uses_threepp("control", "Controller.hpp")),
                   ("KeyboardController", "the arrow keys", uses_threepp("control", "KeyboardController.hpp")),
                   ("GreedyAI", "heads for the apple", uses_threepp("control", "GreedyAI.hpp")),
                   ("tests/", "the rules, the text view, the AI", uses_threepp("tests", "test_game.cpp")),
                   ("main.cpp", "wires them together", True)]
    beats = {"recap": "model", "clocks": "views", "players": "players", "events": "views", "views": "views",
             "rival": "model"}
    t_rec = TL.start("recap") + 0.8
    rec["diagram"] = dict(
        t0=TL.start("recap") + 0.5,
        boxes={"tests": (990, 100, 150, "tests", t_rec),
               "model": (1210, 100, 170, "model", t_rec),
               "views": (1450, 58, 170, "views", t_rec),
               "players": (1450, 152, 170, "players", TL.start("players")),
               "threepp": (1700, 100, 150, "threepp", t_rec)},
        arrows=[("model", "threepp", "can't include", t_rec, None, True, False),
                ("tests", "model", "checks", t_rec, 0x8aa0c0, False, False),
                ("views", "model", "reads", t_rec, C_ACCENT, False, False),
                ("views", "threepp", "draws with", t_rec, C_BLUE, False, False),
                ("players", "model", "steers", cap0("c1") + 1.5, C_WARM, False, True),
                ("players", "threepp", "keys only", cap0("c3") + 2.5, 0x55627a, False, True)],
        lit=lambda t: next((v for b, v in beats.items() if TL.start(b) <= t < TL.end(b)), None))
    rec["t_this"] = [word_time("a2", w, d) for w, d in (("plays", 3.0), ("tells", 4.5), ("views", 7.0))]
    rec["widgets"] = [w_recap, w_ascii, w_tree]
    rec["rect_keys"] = [(0.0, OPEN_RECT), (TL.end("open") - 0.2, OPEN_RECT), (TL.end("open") + 1.0, STD_RECT)]
    rec.update(streams=streams, windows=win, same_pixels=True)
    return rec


def setup(width, height):
    t0 = time.time()
    st = Stage(width, height, renderer="gl", fog=(4.0, 14.0))
    st.look((0.0, 1.6, 6.5), (0.0, 0.4, 0.0), 32)
    rec = setup_c() if PART == "c" else setup_d()
    for spec in rec["cards"]:
        print(f"[card] {spec[4]}: {len(spec[5])} rows")
    ov = Hud(1920, 1080, math_cache=os.path.join(_HERE, f"{NAME}.math.json"))
    painter = Painter(st, ov, rec)
    streams = rec["streams"]
    print(f"[setup] {time.time() - t0:.1f}s")

    def render(t):
        ov.begin()
        painter.draw2d(t)
        ov.fade(fade_in_out(t, TL.duration))
        ov.end()
        for s in streams.values():             # a program whose part of the film is over can go
            if s.proc is not None and t > s.script.t1 + 1.0:
                s.close_proc()
        return st.frame(t, hud=ov)
    return render, painter.captions


if __name__ == "__main__":
    run(NAME, TL.duration, setup, fps=FPS, speech=SPEECH)
