"""Snake, written three times: two films about where the code goes, for complete beginners.

    python snake_oo.py                               # Part 0c, "threepp is the view" (1080p60), lesson_out/snake_oo.mp4
    python snake_oo.py --part d                      # Part 0d, "Who plays?", lesson_out/snake_oo_d.mp4
    python snake_oo.py --stills 30,60,95             # individual frames (and the other flags every lesson takes)

Part 0c shows one thing at a time. It opens on the game (what Snake is), says what it will do,
then writes Snake three times: all in main() (examples/lesson/snake/tangle), whose lines it
colours by their job (the rules in green, the drawing in blue, both where a line does both); a
class that inherits from threepp's Group (trap), as "is a" against "has a"; and the rules in a
Game class of their own (model/) with the drawing in SnakeView (view/). Then the game as text
beside the 3D picture of it, the rules' own build (CMake), and one test. Part 0d carries on with
the finished app's later steps: the view sliding between ticks, players behind an interface,
events, more views of one game, and a second game.

What is on screen was checked against the programs:

* every snippet is read out of those files, and the stripes are their lines (indent and length);
  the colours are this script's reading of each line (what it touches: a mesh, the renderer and
  scene, or the game's state), not something the compiler says;
* every window is that program's own output. Each program (and each step of main.cpp) is built
  as a capture target (lesson_snake_*_capture) and run while the film is made: headless, on a
  fixed 1/60 s clock, with the keys pressed by a script through the canvas's own event path
  (examples/lesson/capture.hpp). The keys drawn under the window are the ones the script presses,
  and the labels in the opening point at the squares the script's route puts things on. The 3D
  board beside the text board is lesson_snake_capture_5's frame, cropped to the board;
* the text board is what lesson_snake_capture_5 --ascii printed, frame for frame; the test run is
  test_snake's own output (its first words), run while the film is made;
* the compile error is what MSVC printed when a threepp #include was added to model/Game.cpp and
  snake_model was built (2026-09-29; shortened only where it carried the checkout's path).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import lesson  # noqa: E402
from lesson import (DIM, TEXT, Hud, Stage, Stream, card, envelope, fade_in_out, preprocess, remap, run,  # noqa: E402
                    smooth, step_rows, tokenize, tp)


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
C_RULE = 0x7bd389                                    # the rules
C_DRAW = 0x4cc9f0                                    # the drawing
C_BLUE = 0x4cc9f0
C_WARM = 0xffb347
C_OK = 0x5ee27a
C_DEL = 0xef476f
C_OUT = 0xa6e3a1
C_GREY = 0x55627a

CPP_KEYWORDS = lesson.KEYWORDS["cpp"] | {"for", "if", "else", "bool", "true", "false", "nullptr", "static", "void",
                                         "override", "virtual", "public", "private", "class", "struct", "const",
                                         "return", "while", "using"}

WORDS = {"AI": "A I", "AIs": "A Is", "main": "main", "STEP": "step", "C++": "C plus plus", "SnakeView": "Snake View",
         "CMake": "C Make"}

# ── the scripts ───────────────────────────────────────────────────────────────
# The narration sets the pace: each caption lasts as long as its spoken line (Kokoro, af_heart,
# measured in <name>.speech.json), and the beats leave the picture room between lines.
TEXT_C = {
    "title": "threepp is the view. Where does the code go?",
    "i1": "This is Snake. The snake moves on its own, one square at a time. The arrow keys turn it.",
    "i2": "Eat the apple, and it grows. Hit a wall, and the game is over.",
    "i3": "We'll write this game three times, and each time, ask where the code should live.",
    "i4": "First, everything in one function. Then, a class that inherits from threepp. And last, the rules on "
          "their own.",
    "t1": "Version one. Everything is in main, like in our first app. And it works.",
    "t2": "Let's color each line by its job. Green lines are the rules of the game.",
    "t3": "Blue lines draw the picture.",
    "t4": "In version one, they are all mixed together.",
    "t5": "Look at this line. To know if the snake has eaten, a rule asks a 3D shape where it is.",
    "t6": "So to test the rules, you need a window. And to change the picture, you have to touch the rules.",
    "p1": "Version two. A common first try is a Snake class that inherits from threepp's Group.",
    "p2": "A Group is a thing in threepp's 3D scene. So this says: the snake is a 3D scene object.",
    "p3": "But a snake is a piece of the game, not a scene object.",
    "p4": "It has a 3D look. And another class, SnakeView, can own that look and draw it.",
    "m1": "Version three. The rules move into a class of their own, called Game.",
    "m2": "The snake's 3D look moves into another class, called SnakeView.",
    "m3": "What's left stays in main. It sets up the window and the camera.",
    "m4": "And every step, main moves the game on, then asks the view to draw it.",
    "g1": "Game only knows squares on a grid. A view turns those squares into a picture.",
    "g2": "This view prints them as text, with no threepp at all. This one draws them in 3D, with threepp.",
    "g3": "One game, two pictures. The rules don't care who is watching.",
    "b1": "Game even gets a build of its own. Its CMake file doesn't mention threepp at all.",
    "b2": "Try to use threepp in it, and the build stops. The rules can't draw, not even by accident.",
    "e1": "And rules without a window are easy to test.",
    "e2": "Set up a snake next to an apple. Take one step. Check that the snake grew.",
    "e3": "It passes, without opening a single window.",
    "z1": "The rules live in Game, with no threepp. The 3D look lives in SnakeView, and only the view uses threepp.",
    "z2": "Next time: who plays the game.",
}
PLAN_C = [("open", 0.4, ["title"], 2.6),
          ("intro", 1.0, ["i1", "i2", "i3", "i4"], 2.0),
          ("v1", 1.2, ["t1", "t2", "t3", "t4", "t5", "t6"], 2.0),
          ("v2", 1.2, ["p1", "p2", "p3", "p4"], 2.0),
          ("v3", 1.2, ["m1", "m2", "m3", "m4"], 2.0),
          ("grid", 1.2, ["g1", "g2", "g3"], 2.4),
          ("build", 1.2, ["b1", "b2"], 2.0),
          ("test", 1.2, ["e1", "e2", "e3"], 2.0),
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

CAPTION_TEXT, PLAN, GAP = (TEXT_C, PLAN_C, 1.2) if PART == "c" else (TEXT_D, PLAN_D, 0.9)
SPEECH = lesson.Speech(CAPTION_TEXT, os.path.join(_HERE, f"{NAME}.speech.json"), words=WORDS,
                       voice_only={"title"})
TL, CAP = SPEECH.layout(PLAN, gap=GAP)


def cap0(k):
    return CAP[k][0]


def cap1(k):
    return CAP[k][1]


if PART == "c":
    TITLE = ("A THREEPP LESSON  ·  PART 0c", "threepp is the view", "Snake, written three times")
    PILLS = []
    SUMMARY = [(r"$\mathrm{rules}$", "In a class of their own, with no threepp in it."),
               (r"$\mathrm{view}$", "Only the view uses threepp, to draw what the rules say."),
               (r"$\mathrm{has\ a}$", "A snake has a 3D look. It isn't a scene object.")]
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
COMPILE_TRY = '#include "threepp/math/Vector3.hpp"'
COMPILE_ERR = ["model\\Game.cpp(2): fatal error C1083: Cannot open include file:",
               "  'threepp/math/Vector3.hpp': No such file or directory"]


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


# what a line of tangle/main.cpp does: this film's reading of it, by what it touches
_DRAW = ("Canvas", "GLRenderer", "renderer", "Scene", "scene", "camera", "Light", "light", "Material", "material",
         "Geometry", "Mesh::create", "board", "->color", "rotation", "Color(")
_STATE_IN_A_MESH = ("->position", "shared_ptr<Mesh>", "snake.push_back(segment)")
_RULE = ("dead", "timer", "direction", "next", "score", "rng", "Key::", "KeyAdapter", "snake[", "snake.", "clock",
         "std::abs", "for (int i", ": snake)")


def job(line):
    """'rule', 'draw', 'both' (a rule that keeps or reads its state in a mesh) or 'other'."""
    s = line.strip()
    if not s or s.startswith("//") or s in ("}", "});", "} else {") or s.startswith("canvas.animate"):
        return "other"
    if any(k in s for k in _STATE_IN_A_MESH) and not s.startswith("light->"):
        return "both"
    rule, draw = any(k in s for k in _RULE), any(k in s for k in _DRAW)
    if rule and draw:
        return "both"
    return "draw" if draw else ("rule" if rule else "other")


_SETUP = ("Canvas", "GLRenderer", "renderer", "Scene scene", "PerspectiveCamera", "camera", "AmbientLight",
          "DirectionalLight", "light")


def dest(line, kind):
    """Where a line of version one goes in version three: 'game' (the rules), 'view' (the snake's
    3D look), both, or 'main' (the window, the camera, the lights, the loop: they stay)."""
    if kind == "rule":
        return "game"
    if kind == "both":
        return "both"
    if kind == "draw" and not any(k in line for k in _SETUP):
        return "view"
    return "main"


def main_job(line):
    """The colour of a line of version three's main: its calls into the game and the view, the
    window it sets up, and the rest."""
    s = line.strip()
    if "game" in s.lower() and "view" not in s:
        return "rule"
    if "view" in s or any(k in s for k in _SETUP):
        return "draw"
    return "other"


def main_body(lines):
    i = next(k for k, l in enumerate(lines) if l.startswith("int main"))
    j = max(k for k, l in enumerate(lines) if l.rstrip() == "}")
    return lines[i + 1:j]


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

    def drive(self, legs, start="R", head=None, until=None):
        """Press the keys for a route of legs [(heading, moves)]: each turn goes down a few
        frames before the move it applies to. `head` (a square) records where the head is
        after each move."""
        h, m = start, 0
        cell = head
        until = self.frames if until is None else self.frame(until)
        for heading, n in legs:
            for _ in range(n):
                f = self.mf[m]
                if f >= until:
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
        self._tex = {}

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
        alpha = envelope(t, 0.3, TL.end("end") + 0.6, 0.9, 0.8) * self.rec["win_alpha"](t)
        if alpha <= 0.003:
            return None
        x0, y0, x1 = self.window_rect(t)
        y1 = y0 + WBAR + (x1 - x0) * 9 / 16
        ov.panel(x0, y0, x1 - x0, y1 - y0, radius=10, fill=0x000000, alpha=alpha)
        layers = self.window_layers(t)
        for key, a in layers:
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
        if layers:
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

    def crop_tex(self, key, img):
        """A texture of an (H, W, 3) image, updated in place from frame to frame."""
        img = np.ascontiguousarray(img[::-1])            # textures take their rows bottom first
        if key not in self._tex:
            self._tex[key] = tp.data_texture(img, True)
        else:
            self._tex[key].update_data(img)
        return self._tex[key]

    # 2D ----------------------------------------------------------------------
    def draw2d(self, t):
        ov = self.ov
        ov.title_card(t, *TITLE, C_ACCENT, t_in=0.6, t_out=TL.end("open") - 0.4, size=84)
        self.draw_window(t)
        self.pills(t)
        for spec in self.rec["cards"]:
            self.rows_card(t, *spec)
        if self.rec.get("diagram"):
            self.diagram(t)
        for f in self.rec["widgets"]:
            f(self, t)
        ov.captions(t, self.captions)
        ov.summary(t, TL.start("outro"), TL.end("outro"), SUMMARY, FOOTER, C_ACCENT, C_ACCENT, text_dx=300)

    def pills(self, t):
        if not PILLS:
            return
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
            if kind == "del":
                ra *= 0.55
                ov.panel(x + 12, yy - 1, 4, lh, radius=2, fill=C_DEL, alpha=0.9 * ra)
            if kind == "ctx":
                ra *= 0.45
            for col, s, c in tokenize(text, "cpp", CPP_KEYWORDS):
                ov.text(x + 30 + col * cw, yy + lh / 2, s, size=size, color=c, alpha=ra, kind="mono", anchor="lm")
            if kind == "del":
                ov.line([(x + 30, yy + lh / 2), (x + 30 + len(text.rstrip()) * cw, yy + lh / 2)], C_DEL, 2.0, ra)

    def diagram(self, t):
        """Who includes whom, built up beat by beat: the model in the middle, threepp at the edge."""
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


# ── Part 0c: one picture at a time ────────────────────────────────────────────
def big_code(ov, x, y, lines, size, alpha, colors=None, lh=None, lang="cpp"):
    """Lines of code in a large monospace font; `colors` [(line, col0, col1, color)] recolours
    spans (and underlines them)."""
    cw = ov.text_width("0", size, "mono")
    lh = lh or size * 1.45
    for j, line in enumerate(lines):
        for col, s, c in tokenize(line, lang, CPP_KEYWORDS if lang == "cpp" else None):
            for (lj, c0, c1, cc) in colors or ():
                if lj == j and c0 <= col < c1:
                    c = cc
            ov.text(x + col * cw, y + j * lh, s, size=size, color=c, alpha=alpha, kind="mono", anchor="lm")
    for (lj, c0, c1, cc) in colors or ():
        ov.line([(x + c0 * cw, y + lj * lh + size * 0.72), (x + c1 * cw, y + lj * lh + size * 0.72)], cc, 3.0, alpha)
    return cw


def box(ov, x, y, w, h, title, sub, color, alpha, fill=0x0d131e):
    ov.panel(x, y, w, h, radius=16, fill=fill, alpha=0.9 * alpha, outline=color, outline_alpha=0.9)
    ov.text(x + w / 2, y + 36, title, size=30, color=color, alpha=alpha, kind="semibold", anchor="mm")
    if sub:
        ov.text(x + w / 2, y + 70, sub, size=20, color=DIM, alpha=alpha, anchor="mm")


def stripe(ov, x, y, w, h, kind, alpha, split=0.5):
    """One line of code as a bar, coloured by its job; a line that does both is half and half."""
    if alpha <= 0.003 or w <= 0:
        return
    if kind == "rule":
        ov.panel(x, y, w, h, radius=0, fill=C_RULE, alpha=alpha)
    elif kind == "draw":
        ov.panel(x, y, w, h, radius=0, fill=C_DRAW, alpha=alpha)
    elif kind == "both":
        ov.panel(x, y, w * split, h, radius=0, fill=C_RULE, alpha=alpha)
        ov.panel(x + w * split, y, w * (1 - split), h, radius=0, fill=C_DRAW, alpha=alpha)
    else:
        ov.panel(x, y, w, h, radius=0, fill=C_GREY, alpha=0.6 * alpha)


def w_intro(p, t):
    """The opening: what is on the board, then what the film will do."""
    ov = p.ov
    sc = p.streams["intro"].script
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
    a = envelope(t, cap0("i4") - 0.2, TL.end("intro") + 0.4, 0.7, 0.7)
    if a > 0.003:
        x, y = 460, 170
        card(ov, x, y, 1000, 420, a, "SNAKE, THREE TIMES")
        rows = [("1", "everything in one function", "all of it in main()"),
                ("2", "a class that inherits from threepp", "class Snake : public Group"),
                ("3", "the rules on their own", "and threepp only draws them")]
        for k, (n, what, how) in enumerate(rows):
            ra = a * smooth(remap(t, p.rec["t_plan"][k], p.rec["t_plan"][k] + 0.5))
            yy = y + 120 + 100 * k
            ov.circle(x + 70, yy, 26, fill=0x3d6b2e, alpha=ra)
            ov.text(x + 70, yy, n, size=28, color=TEXT, alpha=ra, kind="semibold", anchor="mm")
            ov.text(x + 120, yy - 14, what, size=34, color=TEXT, alpha=ra, kind="semibold", anchor="lm")
            ov.text(x + 120, yy + 24, how, size=22, color=DIM, alpha=ra, kind="mono" if k == 1 else "regular",
                    anchor="lm")


# where the stripes of version one sit, and the two boxes of version three
SX0, SY0, SCW = 150, 140, 5.6                      # where the stripes end up (left), for the zoom and version three
SX_MID, SCW_MID = 560, 8.4                         # where they are first shown: centred, larger
GAME_BOX, VIEW_BOX = (1010, 150, 400, 620), (1460, 150, 400, 620)


def t_main():
    return cap0("m3") + 0.2                        # what is left of version one becomes version three's main


def w_version1(p, t):
    """Version one's main(), one bar per line, coloured by job; then the line where a rule asks a
    mesh; then (version three) the rules and the snake's look leave for their boxes, and what is
    left stays: it becomes version three's main."""
    ov, R = p.ov, p.rec
    lines, jobs, dests = R["tangle_lines"], R["tangle_jobs"], R["tangle_dest"]
    n = len(lines)
    lh = min(7.5, 640 / n)
    a = envelope(t, cap0("t2") - 0.2, TL.end("v1") - 0.2, 0.8, 0.6) + envelope(t, TL.start("v3"), t_main() + 0.6,
                                                                                0.6, 0.6)
    if a <= 0.003:
        return
    green = smooth(remap(t, R["t_green"], R["t_green"] + 0.6)) if t < TL.start("v3") else 1.0
    blue = smooth(remap(t, R["t_blue"], R["t_blue"] + 0.6)) if t < TL.start("v3") else 1.0
    zoom = envelope(t, R["t_zoom"], TL.end("v1") - 0.2, 0.6, 0.6)
    in_v3 = t >= TL.start("v3") - 0.3
    u = 1.0 if in_v3 else smooth(remap(t, R["t_zoom"] - 0.8, R["t_zoom"] + 0.2))
    sx0, scw = SX_MID + (SX0 - SX_MID) * u, SCW_MID + (SCW - SCW_MID) * u
    card(ov, sx0 - 40, SY0 - 70, 100 * scw + 80, 60 + n * lh + 50,
         a * (1 - smooth(remap(t, t_main(), t_main() + 0.6))) if in_v3 else a, "version 1  ·  main()")
    legend(ov, sx0 + 100 * scw - 280, SY0 - 50, a * green, a * blue)
    for i, (line, kind) in enumerate(zip(lines, jobs)):
        s = line.rstrip()
        if not s.strip():
            continue
        ind = len(s) - len(s.lstrip(" "))
        x, y, w = sx0 + ind * scw, SY0 + i * lh, (len(s) - ind) * scw
        if in_v3:
            fly(p, t, i, n, x, y, w, lh, kind, dests[i], a)
            continue
        shown = kind
        if kind == "rule" and green < 0.5 or kind == "draw" and blue < 0.5:
            shown = "other"
        if kind == "both" and (green < 0.5 or blue < 0.5):
            shown = "rule" if green >= 0.5 else "other"
        la = a * (0.35 + 0.65 * (1 - zoom)) if i != R["eat_line"] else a
        stripe(ov, x, y, w, max(lh - 1.5, 2), shown, la)
    # the zoom: the line where a rule asks a mesh where it is
    if zoom > 0.003 and not in_v3:
        i = R["eat_line"]
        s = lines[i].strip()
        zx, zy = 900, 380
        ov.panel(zx - 40, zy - 70, 1000, 240, radius=16, alpha=0.9 * zoom, outline=0x8aa0c0, outline_alpha=0.3)
        ov.line([(sx0 + (len(lines[i].rstrip()) + 1) * scw, SY0 + i * lh), (zx - 42, zy)], C_WARM, 2.0, zoom)
        c_rule0, c_rule1 = s.index("next"), s.index("(apple") + 1
        c_mesh0 = s.index("apple->position")
        c_mesh1 = c_mesh0 + len("apple->position")
        cw = big_code(ov, zx, zy, [s], 34, zoom, colors=[(0, c_rule0, c_rule1, C_RULE), (0, c_mesh0, c_mesh1, C_DRAW)])
        ov.text(zx + (c_rule0 + c_rule1) / 2 * cw, zy + 62, "a rule...", size=24, color=C_RULE, alpha=zoom,
                anchor="mm")
        ov.text(zx + (c_mesh0 + c_mesh1) / 2 * cw, zy + 62, "...asks a 3D shape", size=24, color=C_DRAW,
                alpha=zoom, anchor="mm")


def legend(ov, x, y, a_rule, a_draw):
    ov.panel(x, y - 8, 16, 16, radius=3, fill=C_RULE, alpha=a_rule)
    ov.text(x + 24, y, "the rules", size=18, color=TEXT, alpha=a_rule, anchor="lm")
    ov.panel(x + 140, y - 8, 16, 16, radius=3, fill=C_DRAW, alpha=a_draw)
    ov.text(x + 164, y, "the drawing", size=18, color=TEXT, alpha=a_draw, anchor="lm")


def fly(p, t, i, n, x, y, w, lh, kind, where, a):
    """Version three: bar i of version one flies into its box (a bar that does both splits), or
    stays in main until version three's main takes its place."""
    ov = p.ov
    h = max(lh - 1.5, 2)
    if where == "game":
        parts = [("rule", GAME_BOX, TL.start("v3"), x, w)]
    elif where == "view":
        parts = [("draw", VIEW_BOX, cap0("m2"), x, w)]
    elif where == "both":
        parts = [("rule", GAME_BOX, TL.start("v3"), x, w / 2), ("draw", VIEW_BOX, cap0("m2"), x + w / 2, w / 2)]
    else:
        stripe(ov, x, y, w, h, kind, a * (1 - smooth(remap(t, t_main(), t_main() + 0.6))))
        return
    for piece, bx, t_go, px, pw in parts:
        t_i = t_go + 1.2 + 2.0 * i / n
        u = smooth(remap(t, t_i, t_i + 1.0))
        tx, ty = bx[0] + 30 + (px - SX0) * 0.55, bx[1] + 110 + (y - SY0) * 0.75
        stripe(ov, px + (tx - px) * u, y + (ty - y) * u, pw * (1 - 0.45 * u), h, piece, a * (1 - u) * (1 - 0.3 * u))


def w_version3(p, t):
    """The two boxes, filling with their files' own lines; version three's main in place of what
    was left; then main's loop, up close."""
    ov, R = p.ov, p.rec
    a = envelope(t, TL.start("v3") + 0.2, TL.end("v3") - 0.1, 0.7, 0.6)
    if a <= 0.003:
        return
    for (bx, by, bw, bh), title, sub, color, lines, t_go in (
            (GAME_BOX, "Game", "model/  ·  the rules  ·  no threepp", C_RULE, R["game_lines"], TL.start("v3")),
            (VIEW_BOX, "SnakeView", "view/  ·  the 3D look  ·  threepp", C_DRAW, R["view_lines"], cap0("m2"))):
        ba = a * smooth(remap(t, t_go + 0.6, t_go + 1.2))
        if ba <= 0.003:
            continue
        box(ov, bx, by, bw, bh, title, sub, color, ba)
        n = len(lines)
        lh = min(6.0, (bh - 130) / n)
        for j, line in enumerate(lines):
            s = line.rstrip()
            if not s.strip():
                continue
            ind = len(s) - len(s.lstrip(" "))
            ja = ba * smooth(remap(t, t_go + 1.6 + 2.0 * j / n, t_go + 2.2 + 2.0 * j / n))
            stripe(ov, bx + 30 + ind * 3.2, by + 110 + j * lh, min((len(s) - ind) * 3.2, bw - 60 - ind * 3.2),
                   max(lh - 1.2, 1.5), "rule" if color == C_RULE else "draw", ja * 0.9)
    # version three's main: what was left, now calling the game and the view
    lines, jobs = R["main_lines"], R["main_jobs"]
    n = len(lines)
    lh = min(7.5, 640 / len(R["tangle_lines"]))
    va = a * envelope(t, t_main(), cap0("m4") + 0.2, 0.6, 0.6)
    if va > 0.003:
        card(ov, SX0 - 40, SY0 - 70, 100 * SCW + 80, 60 + len(R["tangle_lines"]) * lh + 50, va, "version 3  ·  main()")
        legend(ov, SX0 + 100 * SCW - 280, SY0 - 50, va, va)
        for i, (line, kind) in enumerate(zip(lines, jobs)):
            s = line.rstrip()
            if not s.strip():
                continue
            ind = len(s) - len(s.lstrip(" "))
            stripe(ov, SX0 + ind * SCW, SY0 + i * lh, (len(s) - ind) * SCW, max(lh - 1.5, 2), kind, va)
            if "game.tick" in s or "view.update" in s:
                ov.text(SX0 + len(s) * SCW + 14, SY0 + i * lh + lh / 2, s.strip().split("(")[0] + "()", size=18,
                        color=C_RULE if kind == "rule" else C_DRAW, alpha=va, kind="mono", anchor="lm")
    # main's loop, up close
    ma = a * smooth(remap(t, cap0("m4") + 0.2, cap0("m4") + 0.9))
    if ma > 0.003:
        lines = R["main_loop"]
        card(ov, 110, 250, 820, 120 + 52 * len(lines), ma, "version 3  ·  main()  ·  EVERY FRAME")
        rows = [(j, len(l.rstrip()), C_RULE if "game.tick" in l else C_DRAW)
                for j, l in enumerate(lines) if "game.tick" in l or "view.update" in l]
        big_code(ov, 150, 340, lines, 30, ma, lh=52,
                 colors=[(j, len(lines[j]) - len(lines[j].lstrip(" ")), c1, c) for j, c1, c in rows])


def w_version2(p, t):
    """A snake is a Group (crossed out), or has a 3D look (a SnakeView that holds a Group)."""
    ov, R = p.ov, p.rec
    a = envelope(t, TL.start("v2") + 0.2, TL.end("v2") - 0.1, 0.7, 0.6)
    if a <= 0.003:
        return
    line = R["trap_line"]
    size = 40
    cw = ov.text_width("0", size, "mono")
    x = 960 - len(line) * cw / 2
    big_code(ov, x, 170, [line], size, a)
    no = smooth(remap(t, R["t_not"], R["t_not"] + 0.5))
    if no > 0.003:
        ov.line([(x - 10, 170), (x + len(line) * cw + 10, 170)], C_DEL, 4.0, a * no)
    # is a
    ia = a * (1 - smooth(remap(t, cap0("p4") - 0.2, cap0("p4") + 0.6)))
    if ia > 0.003:
        box(ov, 460, 380, 360, 110, "Snake", "a piece of the game", C_RULE, ia)
        gb = ia * smooth(remap(t, R["t_group"], R["t_group"] + 0.5))
        box(ov, 1100, 380, 360, 110, "Group", "a 3D scene object", C_DRAW, max(gb, 0.0) if gb > 0.003 else 0.0)
        ov.arrow2d((830, 435), (1090, 435), TEXT, 3.0, 16, ia)
        ov.text(960, 405, "is a", size=30, color=TEXT, alpha=ia, kind="semibold", anchor="mm")
        if no > 0.003:
            ov.line([(930, 400), (990, 470)], C_DEL, 6.0, ia * no)
            ov.line([(930, 470), (990, 400)], C_DEL, 6.0, ia * no)
            ov.text(960, 540, "a snake is not a scene object", size=28, color=C_DEL, alpha=ia * no, anchor="mm")
    # has a
    ha = a * smooth(remap(t, cap0("p4") + 0.3, cap0("p4") + 1.0))
    if ha > 0.003:
        box(ov, 400, 400, 360, 110, "Snake", "a piece of the game", C_RULE, ha)
        ov.panel(1000, 330, 520, 250, radius=18, fill=0x0d131e, alpha=0.9 * ha, outline=C_DRAW, outline_alpha=0.9)
        ov.text(1260, 370, "SnakeView", size=30, color=C_DRAW, alpha=ha, kind="semibold", anchor="mm")
        box(ov, 1110, 430, 300, 110, "Group", "the snake's 3D look", C_DRAW, ha, fill=0x1d3354)
        ov.text(1260, 610, "SnakeView has the 3D look, and draws the snake", size=24, color=DIM, alpha=ha,
                anchor="mm")
        da = ha * smooth(remap(t, R["t_draw"], R["t_draw"] + 0.5))
        ov.arrow2d((990, 455), (770, 455), DIM, 2.5, 12, da)
        ov.text(880, 430, "draws", size=22, color=DIM, alpha=da, anchor="mm")


def w_grid(p, t):
    """One game, two views: Game on top, and what lesson_snake --ascii prints (AsciiView, no
    threepp) beside its window (SnakeView, threepp), cropped to the board."""
    ov, R = p.ov, p.rec
    a = envelope(t, TL.start("grid") + 0.2, TL.end("grid") - 0.1, 0.7, 0.6)
    if a <= 0.003:
        return
    box(ov, 740, 80, 440, 110, "Game", "squares on a grid, and no picture", C_RULE, a)
    s = p.streams["grid"]
    # the text view
    ta = a * smooth(remap(t, R["t_text"], R["t_text"] + 0.6))
    size, lh = 20, 22
    cw = ov.text_width("0", size, "mono")
    legend_text = "O  the head      o  the body      @  the apple"
    cw_w = max(22 * cw, ov.text_width(legend_text, 18)) + 80
    x, y = 170, 310
    if ta > 0.003:
        card(ov, x - 40, y - 70, cw_w, 22 * lh + 140, ta, "AsciiView  ·  TEXT, NO THREEPP")
        board = [l for l in s.printed(t) if l.startswith("#")][-22:]
        for j, line in enumerate(board):
            ov.text(x, y + j * lh, line.split("  score")[0], size=size, color=C_OUT, alpha=ta, kind="mono",
                    anchor="lm")
        ov.text(x, y + 22 * lh + 16, legend_text, size=18, color=DIM, alpha=ta, anchor="lm")
        ov.arrow2d((850, 196), (x - 40 + cw_w / 2, y - 76), C_RULE, 2.5, 12, ta)
        ov.text(640, 210, "reads", size=20, color=DIM, alpha=ta, anchor="mm")
    # the 3D view, cropped from the same program's frame
    da = a * smooth(remap(t, R["t_3d"], R["t_3d"] + 0.6))
    if da > 0.003:
        x0, y0, x1, y1 = R["crop"]
        img = s.at(t)[y0:y1, x0:x1]
        w3 = 700
        h3 = w3 * (y1 - y0) / (x1 - x0)
        X, Y = 1060, 270
        card(ov, X - 30, Y - 70, w3 + 60, h3 + 100, da, "SnakeView  ·  3D, WITH THREEPP")
        ov.image(X, Y, w3, h3, p.crop_tex("grid", img), alpha=da)
        ov.arrow2d((1070, 196), (X + w3 / 2, Y - 76), C_RULE, 2.5, 12, da)
        ov.text(1280, 210, "reads", size=20, color=DIM, alpha=da, anchor="mm")
    else:
        s.at(t)                                   # keep the program moving while its window is not shown yet


def w_build(p, t):
    """The rules' own build: CMake, then what happens when they try to use threepp."""
    ov, R = p.ov, p.rec
    a = envelope(t, TL.start("build") + 0.2, TL.end("build") - 0.1, 0.7, 0.6)
    if a <= 0.003:
        return
    lines = R["cmake_lines"]
    card(ov, 260, 130, 1400, 90 + 46 * len(lines), a, "CMakeLists.txt  ·  THE RULES' OWN BUILD")
    big_code(ov, 300, 210, lines, 26, a, lh=46, lang="cmake")
    ea = a * smooth(remap(t, cap0("b2") + 0.3, cap0("b2") + 1.0))
    if ea > 0.003:
        card(ov, 260, 460, 1400, 290, ea, "model/Game.cpp  ·  TRY IT")
        big_code(ov, 300, 540, [COMPILE_TRY], 26, ea, colors=[(0, 0, len(COMPILE_TRY), C_DEL)])
        er = a * smooth(remap(t, R["t_stops"], R["t_stops"] + 0.5))
        for j, l in enumerate(COMPILE_ERR):
            ov.text(300, 620 + j * 40, l, size=24, color=C_DEL, alpha=er, kind="mono", anchor="lm")
        ov.text(300, 710, "the build stops", size=26, color=TEXT, alpha=er, kind="semibold", anchor="lm")


def w_test(p, t):
    """One test, three parts, and its run."""
    ov, R = p.ov, p.rec
    a = envelope(t, TL.start("test") + 0.2, TL.end("test") - 0.1, 0.7, 0.6)
    if a <= 0.003:
        return
    card(ov, 200, 130, 1520, 400, a, "tests/test_game.cpp  ·  NO WINDOW")
    y = 230
    for k, (label, lines, t_on) in enumerate(R["test_parts"]):
        pa = a * smooth(remap(t, t_on, t_on + 0.6))
        ov.text(260, y, label, size=26, color=C_WARM, alpha=pa, kind="semibold", anchor="lm")
        big_code(ov, 480, y, lines, 26, pa, lh=42)
        y += 42 * len(lines) + 40
    ra = a * smooth(remap(t, cap0("e3") + 0.2, cap0("e3") + 0.8))
    if ra > 0.003:
        card(ov, 200, 570, 1520, 150, ra, "TERMINAL")
        ov.text(240, 640, f'> test_snake "{R["test_name"]}"', size=24, color=C_BLUE, alpha=ra, kind="mono",
                anchor="lm")
        ov.text(240, 682, R["test_out"], size=26, color=C_OK, alpha=a * smooth(remap(t, cap0("e3") + 0.9,
                                                                                         cap0("e3") + 1.3)),
                kind="mono", anchor="lm")


def w_end(p, t):
    """The rules on one side, with no threepp; the view and threepp on the other."""
    ov = p.ov
    a = envelope(t, TL.start("end") + 0.2, TL.end("end") + 0.4, 0.7, 0.6)
    if a <= 0.003:
        return
    box(ov, 200, 380, 440, 140, "Game", "the rules  ·  no threepp", C_RULE, a)
    ra = a * smooth(remap(t, TL.start("end") + 1.0, TL.start("end") + 1.6))
    x0, y0, x1, y1 = 820, 300, 1740, 600
    for q0, q1 in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        ov.dashed(q0, q1, C_DRAW, 2.0, 0.7 * ra, dash=12, gap=8)
    ov.text(x0 + 24, y0 + 30, "uses threepp", size=22, color=C_DRAW, alpha=ra, kind="semibold", anchor="lm")
    box(ov, 870, 380, 400, 140, "SnakeView", "the 3D look", C_DRAW, a)
    box(ov, 1310, 380, 380, 140, "threepp", "draws", TEXT, ra, fill=0x1d3354)
    ov.arrow2d((860, 450), (650, 450), DIM, 2.5, 12, a)
    ov.text(755, 425, "reads", size=20, color=DIM, alpha=a, anchor="mm")
    ov.arrow2d((1275, 450), (1305, 450), DIM, 2.5, 10, ra)


# ── Part 0d's widgets ─────────────────────────────────────────────────────────
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


STREAM_OF = {"tangle": "lesson_snake_tangle_capture"}


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
    test_c, cmake = source("tests", "test_game.cpp"), source("CMakeLists.txt")

    # The opening game: circle near the start, eat the apple a second after it is named, then run
    # into a wall. The finished app starts its head at (10, 10), heading right, and (seed 7) its
    # first apple at (13, 12): three right and two up.
    t_snake = word_time("i1", "snake", 1.0, nth=2)
    t_apple = word_time("i2", "apple", 0.6)
    t_wall = word_time("i2", "wall", 2.0)
    lap = [("D", 3), ("L", 3), ("U", 3), ("R", 3)]              # round and back to (10, 10), heading right
    t_eat = t_apple + 1.0
    probe = move_frames("app", 400)
    k_eat = next(k for k in range(30) if probe[12 * k + 4] / FPS >= t_eat - TL.start("intro"))
    eat_move = 12 * k_eat + 5
    intro = KeyScript(t_eat - probe[eat_move - 1] / FPS, cap0("i4") + 1.0, "app", step=1)
    after = [("U", 3), ("R", 3), ("D", 3), ("L", 3)]            # round and back to the apple's square
    m_after = max(0, int(round((t_wall + 0.6 - t_eat - 8 * TICK) / (12 * TICK))))
    intro.drive(lap * k_eat + [("R", 3), ("U", 2)] + after * m_after + [("U", 8)], head=(10, 10))
    t_die = intro.time(intro.mf[eat_move + 12 * m_after + 8 - 1])
    print(f"[intro] eats at {t_eat:.1f} s ('apple' at {t_apple:.1f}), hits the wall at {t_die:.1f} s "
          f"('wall' at {t_wall:.1f})")
    rec.update(t_snake=t_snake, t_apple=t_apple, t_wall=t_wall, t_eat=t_eat, apple=(13, 12),
               t_plan=[word_time("i4", w, d) for w, d in (("first", 0.8), ("then", 3.0), ("last", 6.0))])

    tangle_s = KeyScript(TL.start("v1"), cap0("t2") + 1.0, "tangle")
    tangle_s.drive(square(5, 6, 20))
    open_s = KeyScript(0.0, TL.end("open") + 1.2, "app", step=1)
    open_s.drive(lap * 5)
    grid_s = KeyScript(TL.start("grid"), TL.end("grid") + 1.0, "app", step=5)
    grid_s.press(1, "TAB")                                       # the AI plays
    win = [("open", 0.0, TL.end("open") + 0.6, "snake"),
           ("intro", TL.start("intro"), cap0("i4") + 1.0, "snake"),
           ("tangle", TL.start("v1"), cap0("t2") + 1.0, "version 1  ·  everything in main()")]
    scripts = {"open": open_s, "intro": intro, "tangle": tangle_s, "grid": grid_s}
    streams = make_streams(scripts, logged=("grid",))

    def win_alpha(t):
        # the game, then out of the way: back for version one's first line, then gone
        return (1 - smooth(remap(t, cap0("i4") - 0.6, cap0("i4") + 0.2))) + \
            envelope(t, TL.start("v1") + 0.1, cap0("t2") + 0.6, 0.6, 0.6)

    body = main_body(tangle)
    rec["tangle_lines"] = body
    rec["tangle_jobs"] = [job(l) for l in body]
    rec["tangle_dest"] = [dest(l, k) for l, k in zip(body, rec["tangle_jobs"])]
    mbody = main_body(main)
    first = next(i for i, l in enumerate(main) if l.startswith("int main"))
    kept = set(preprocess(main, 1))
    rec["main_lines"] = [l for k, l in enumerate(mbody) if first + 1 + k in kept]
    rec["main_jobs"] = [main_job(l) for l in rec["main_lines"]]
    rec["eat_line"] = next(i for i, l in enumerate(body) if "next.distanceTo(apple->position)" in l)
    counts = {k: rec["tangle_jobs"].count(k) for k in ("rule", "draw", "both", "other")}
    stays = sum(1 for d in rec["tangle_dest"] if d == "main")
    print(f"[jobs] tangle main(): {counts}; {stays} lines stay in main")
    rec["t_green"] = word_time("t2", "green", 3.0)
    rec["t_blue"] = word_time("t3", "blue", 0.2)
    rec["t_zoom"] = cap0("t5")
    rec["game_lines"] = source("model", "Snake.cpp") + source("model", "Game.cpp")
    rec["view_lines"] = source("view", "SnakeView.cpp")
    # main's loop as it stands at step 1 (the alpha line left out: it is 1 until Part 0d)
    kept = preprocess(main, 1)
    w0 = next(i for i in kept if "while (sinceTick >= tickTime)" in main[i])
    w1 = next(i for i in kept if "renderer.render(scene, camera);" in main[i] and i > w0)
    rec["main_loop"] = [main[i][8:] for i in kept if w0 <= i <= w1 and main[i].strip() and "alpha = " not in main[i]]
    rec["trap_line"] = next(l for l in trap if l.startswith("class Snake: public Group")).rstrip(" {") + " { ... };"
    rec["t_group"] = cap0("p2") + 0.3
    rec["t_not"] = word_time("p3", "piece", 1.2)
    rec["t_draw"] = word_time("p4", "draw", 4.0)

    # the grid: the finished app at step 5, the AI playing, its board as text beside its 3D frame
    xs, ys = zip(*[board_to_capture(c, 0.0) for c in ((-0.5, -0.5), (19.5, -0.5), (-0.5, 19.5), (19.5, 19.5))])
    rec["crop"] = (int(min(xs)) - 20, max(0, int(min(ys)) - 40), int(max(xs)) + 20, min(720, int(max(ys)) + 20))
    assert rec["crop"][2] < 1280 - 160 - 16, "the crop would reach the minimap"
    rec["t_text"] = word_time("g2", "text", 1.4)
    rec["t_3d"] = word_time("g2", "this", 3.2, nth=2)

    i = next(k for k, l in enumerate(cmake) if l.startswith("# The rules are a library"))
    rec["cmake_lines"] = [cmake[i], cmake[i + 1], "", next(l for l in cmake if l.startswith("add_library(snake_model"))]
    rec["t_stops"] = word_time("b2", "stops", 2.5)

    tc = excerpt(test_c, 'TEST_CASE("eating grows', "CHECK(game.isFree(game.apple()));")
    setup_lines = [l.strip() for l in tc if l.strip().startswith(("Snake snake", "Game game"))]
    step_line = [l.strip() for l in tc if l.strip() == "game.tick();"]
    check_line = [l.strip() for l in tc if "length() == 4" in l]
    rec["test_parts"] = [("set up", setup_lines, word_time("e2", "set", 0.3)),
                         ("one step", step_line, word_time("e2", "take", 2.5)),
                         ("check", check_line, word_time("e2", "check", 4.0))]
    rec["test_name"] = tc[0].split('"')[1]
    out = subprocess.run([lesson.capture_exe("test_snake"), rec["test_name"]], capture_output=True, text=True)
    last = out.stdout.strip().split("\n")[-1]
    rec["test_out"] = last.split(" (")[0]                        # its first words: no counts in the picture
    print(f"[test] {rec['test_name']}: {last}")
    assert out.returncode == 0 and last.startswith("All tests passed")

    rec["cards"] = []
    rec["widgets"] = [w_intro, w_version1, w_version2, w_version3, w_grid, w_build, w_test, w_end]
    rec["rect_keys"] = [(0.0, OPEN_RECT), (TL.end("open") - 0.2, OPEN_RECT), (TL.start("intro") + 1.0, BIG_RECT)]
    rec.update(streams=streams, windows=win, win_alpha=win_alpha)
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
    rec.update(streams=streams, windows=win, win_alpha=lambda t: 1.0)
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
