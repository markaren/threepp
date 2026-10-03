"""The 2D layer of a lesson, drawn by threepp over the 3D scene (`Hud`): panels, text, maths,
lines, plots, readouts, captions and source code (`tokenize`, `Hud.code`)."""
from __future__ import annotations

import math
import os
import re
from types import MappingProxyType

import numpy as np

import threepp as tp

from .scene import to_hex
from .timing import clamp01, envelope

__all__ = ["TEXT", "DIM", "FONT_DIRS", "FONT_FILES", "KEYWORDS", "CODE_COLOURS", "CODE_ACCENT", "tokenize",
           "Hud"]


TEXT = 0xf2f5fa     # body text and maths
DIM = 0x9fb2cc      # labels, notes, secondary text

# Where system fonts are looked for, and the files that serve each kind, in order of
# preference (read-only: a lesson with fonts of its own passes `Hud(fonts=)`).
FONT_DIRS = (os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"),
             "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts")
FONT_FILES = MappingProxyType({
    "regular": ("segoeui.ttf", "DejaVuSans.ttf"),
    "light": ("segoeuil.ttf", "segoeuisl.ttf", "DejaVuSans.ttf"),
    "semibold": ("seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"),
    "bold": ("segoeuib.ttf", "DejaVuSans-Bold.ttf"),
    "italic": ("segoeuii.ttf", "DejaVuSans-Oblique.ttf"),
    "numeric": ("bahnschrift.ttf", "DejaVuSansMono.ttf"),
    "mono": ("consola.ttf", "DejaVuSansMono.ttf"),
})


def _find_font(kind):
    for name in FONT_FILES[kind]:
        for d in FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


# Keywords per language (read-only: a lesson that wants more passes its own set, e.g.
# KEYWORDS["cpp"] | {"for"}), and the colours of each kind of token (read-only too: `tokenize(palette=)`).
KEYWORDS = MappingProxyType({
    "py": frozenset({"import", "as", "def", "return", "for", "in", "if", "else", "True", "False", "None", "from"}),
    "cpp": frozenset({"using", "namespace", "int", "auto", "return", "const", "new", "float"}),
    "js": frozenset({"const", "new", "let", "function"}), "cmake": frozenset(), "sh": frozenset({"pip", "python"})})
CODE_COLOURS = MappingProxyType({"kw": 0xc792ea, "str": 0xc3e88d, "num": 0xf78c6c, "fn": 0x82aaff, "type": 0xffcb6b,
                                 "mod": 0x89ddff, "com": 0x6b7a90, "punc": 0x8fa3bf, "id": 0xe6ecf5, "pre": 0xc792ea})
CODE_ACCENT = 0x4cc9f0       # the bar beside a highlighted line
_TOK = re.compile(r'(//.*|#.*)|("[^"]*")|(\b0x[0-9a-fA-F]+\b|\b\d+\.?\d*f?\b)|([A-Za-z_][A-Za-z_0-9]*)|(\s+)|(.)')


def tokenize(line, lang, keywords=None, palette=None):
    """(column, text, colour) per token of one source line; whitespace is skipped. `keywords`
    replaces KEYWORDS[lang], `palette` replaces CODE_COLOURS."""
    kw = KEYWORDS.get(lang, ()) if keywords is None else keywords
    col = CODE_COLOURS if palette is None else palette
    out = []
    ms = list(_TOK.finditer(line))
    for k, m in enumerate(ms):
        s = m.group(0)
        if m.group(5):
            continue
        if m.group(1):
            if lang == "cpp" and s.startswith("#"):          # #include "..."
                word = s.split()[0]
                out.append((m.start(), word, col["pre"]))
                rest = s[len(word):]
                if rest.strip():
                    out.append((m.start() + len(word) + (len(rest) - len(rest.lstrip())), rest.strip(), col["str"]))
                continue
            if (lang == "cpp") != s.startswith("//"):         # a '#' in C++ or '//' elsewhere is not a comment
                out.append((m.start(), s, col["punc"]))
                continue
            out.append((m.start(), s, col["com"]))
        elif m.group(2):
            out.append((m.start(), s, col["str"]))
        elif m.group(3):
            out.append((m.start(), s, col["num"]))
        elif m.group(4):
            nxt = line[m.end():].lstrip()[:1]
            if s in kw:
                c = col["kw"]
            elif s in ("tp", "THREE", "threepp"):
                c = col["mod"]
            elif s[0].isupper():
                c = col["type"]
            elif nxt == "(":
                c = col["fn"]
            else:
                c = col["id"]
            out.append((m.start(), s, c))
        else:
            out.append((m.start(), s, col["punc"]))
    return out


def _dispose(o):
    """Free what a pooled HUD object owns, its geometry and material; the renderer's caches
    listen for it and drop their copies."""
    g = getattr(o, "geometry", None)
    if g is not None:
        g.dispose()
    m = getattr(o, "material", None)
    if m is not None:
        m.dispose()


class Hud:
    """The 2D layer, drawn by threepp: an orthographic scene rendered over the 3D one.

    Immediate-mode API: call `begin()`, then draw calls (`panel`, `text`, `math`,
    `line`, ...), then `end()`. Behind it every call reuses a pooled threepp mesh:
    glyphs are `Text2D` meshes built from the TTF, panels are `ShapeGeometry`
    rounded rectangles, lines are triangle strips, and equations are matplotlib's
    glyph outlines turned into an SVG path and loaded with `SVGLoader`, so they are
    real geometry too. Objects a frame does not touch are hidden, and each call is
    drawn on top of the previous one (render order = call order, no depth test).

    Coordinates are pixels in a design space of `width` x `height`, origin top-left,
    whatever the actual render size; `Stage(design=)` names the same space, so that
    `project()` lands on these pixels. Text is measured with the same threepp Font
    that draws it (`Font.advance`, `ascender`, `descender`), so layout and glyphs agree.
    """

    def __init__(self, width=1920, height=1080, math_cache=None, fonts=None, pool_limit=2000):
        """`math_cache`: a JSON file of typeset equations. With it, a lesson whose
        equations are all cached renders without matplotlib; a new or edited
        equation is typeset once and appended. `fonts`: {kind: TTF path} to use instead
        of the search in FONT_DIRS (kinds as in FONT_FILES); a kind with no font found
        falls back to threepp's default font. `pool_limit`: how many drawn things (a
        string at a size, a panel of a size, an equation, ...) are kept ready between
        frames; past it, `end()` releases the least recently drawn."""
        self.W, self.H = int(width), int(height)
        self._font_paths = dict(fonts or {})
        self._math_file = math_cache
        self._math_disk = {}
        if math_cache and os.path.isfile(math_cache):
            import json
            with open(math_cache, encoding="utf-8") as f:
                self._math_disk = json.load(f)
        self.scene = tp.Scene()
        self.camera = tp.OrthographicCamera(0, self.W, self.H, 0, -10, 10)
        self._fonts_tp = {}
        self._pool = {}          # key -> the pooled objects, in the scene, hidden when unused
        self._used = {}          # key -> how many of them this frame has drawn
        self._last = {}          # key -> the frame that last drew one
        self._held = {}          # key -> the texture an image mesh shows (its id is in the key)
        self._frame = 0
        self._order = 0
        self.pool_limit = int(pool_limit)
        self._math_cache = {}

    # fonts -----------------------------------------------------------------
    def _tp_font(self, kind):
        f = self._fonts_tp.get(kind)
        if f is None:
            path = self._font_paths.get(kind) or _find_font(kind)
            f = tp.FontLoader().load(path) if path else None
            if f is None:
                f = tp.FontLoader().default_font()
            self._fonts_tp[kind] = f
        return f

    def text_width(self, s, size=28, kind="regular"):
        return self._tp_font(kind).advance(s, size)

    # pooling ---------------------------------------------------------------
    def begin(self):
        self._used = {k: 0 for k in self._pool}
        self._order = 0
        self._frame += 1
        return self

    def end(self):
        for k, objs in self._pool.items():
            n = self._used.get(k, 0)
            for o in objs[n:]:
                o.visible = False
        self._evict()

    def _evict(self):
        """Release the least recently drawn pooled objects, never this frame's, until the pool
        is within `pool_limit`; what they own is disposed."""
        total = sum(len(objs) for objs in self._pool.values())
        for key in sorted(self._pool, key=self._last.__getitem__):
            if total <= self.pool_limit:
                break
            if self._last[key] == self._frame:
                continue
            for o in self._pool.pop(key):
                self.scene.remove(o)
                o.traverse(_dispose)
                total -= 1
            self._used.pop(key, None)
            self._last.pop(key, None)
            self._held.pop(key, None)

    def _acquire(self, key, factory):
        objs = self._pool.setdefault(key, [])
        n = self._used.get(key, 0)
        if n < len(objs):
            o = objs[n]
        else:
            o = factory()
            self.scene.add(o)
            objs.append(o)
        self._used[key] = n + 1
        self._last[key] = self._frame
        o.visible = True
        self._order += 1
        o.render_order = self._order
        return o

    @staticmethod
    def _material(color=0xffffff):
        m = tp.MeshBasicMaterial()
        m.color = to_hex(color)
        m.transparent = True
        m.depth_test = False
        m.depth_write = False
        m.tone_mapped = False
        m.side = tp.Side.Double
        return m

    def _paint(self, mesh, color, alpha):
        m = mesh.material
        m.color = to_hex(color)
        m.opacity = clamp01(alpha)

    def _Y(self, y):
        return self.H - y

    # primitives --------------------------------------------------------------
    @staticmethod
    def _rounded_shape(x, y, w, h, r):
        """Rounded rectangle, lower-left (x, y), y up."""
        r = max(0.0, min(r, w / 2, h / 2))
        s = tp.Shape()
        s.move_to(x + r, y)
        s.line_to(x + w - r, y)
        s.absarc(x + w - r, y + r, r, -math.pi / 2, 0, False)
        s.line_to(x + w, y + h - r)
        s.absarc(x + w - r, y + h - r, r, 0, math.pi / 2, False)
        s.line_to(x + r, y + h)
        s.absarc(x + r, y + h - r, r, math.pi / 2, math.pi, False)
        s.line_to(x, y + r)
        s.absarc(x + r, y + r, r, math.pi, 1.5 * math.pi, False)
        return s

    def _rrect(self, x, y, w, h, radius, color, alpha, outline_w=0.0):
        """A filled rounded rect (or just its outline ring) in design pixels."""
        key = ("rrect", round(w, 1), round(h, 1), round(radius, 1), round(outline_w, 2))

        def make():
            if outline_w > 0:
                outer = self._rounded_shape(0, 0, w, h, radius)
                inner = self._rounded_shape(outline_w, outline_w, w - 2 * outline_w, h - 2 * outline_w,
                                            max(radius - outline_w, 0.0))
                hole = tp.Path()
                pts = inner.get_points(12)
                hole.set_from_points(list(reversed(pts)))
                outer.holes = list(outer.holes) + [hole]
                geom = tp.ShapeGeometry([outer], 12)
            else:
                geom = tp.ShapeGeometry([self._rounded_shape(0, 0, w, h, radius)], 12)
            return tp.Mesh(geom, self._material())
        m = self._acquire(key, make)
        m.position.set(x, self._Y(y + h), 0)
        self._paint(m, color, alpha)
        return m

    def panel(self, x, y, w, h, radius=14, fill=0x0d131e, alpha=0.72, outline=None, outline_alpha=0.25,
              width=1.5):
        if alpha <= 0.003:
            return
        self._rrect(x, y, w, h, radius, fill, alpha)
        if outline is not None:
            self.outline(x, y, w, h, outline, outline_alpha * alpha / 0.72, width=width, radius=radius)

    def outline(self, x, y, w, h, color=0xffffff, alpha=1.0, width=1.5, radius=0):
        """A rounded-rectangle outline with no fill."""
        if alpha <= 0.003:
            return
        self._rrect(x, y, w, h, radius, color, alpha, outline_w=width)

    def _glyphs(self, s, size, kind):
        key = ("text", s, round(size, 1), kind)

        # tessellate by size: a 16 px label needs few segments per curve, a 96 px title many
        segs = int(min(32, max(8, size / 3)))

        def make():
            return tp.Text2D(self._tp_font(kind), s, size=size, curve_segments=segs, material=self._material())
        return self._acquire(key, make)

    def text(self, x, y, s, size=28, color=0xffffff, alpha=1.0, kind="regular", anchor="la", tracking=0.0,
             live=False):
        """anchor = horizontal (l/m/r) + vertical (a ascender, m middle, s baseline, d descender).
        A string is one pooled mesh; `live` draws it one glyph at a time from a per-character
        pool instead, for text that changes every frame (a readout). The font has no kerning,
        so both lay the string out alike."""
        if alpha <= 0.003 or not s:
            return
        f = self._tp_font(kind)
        asc, desc = f.ascender(size), -f.descender(size)
        v = anchor[1]
        base = y + (asc if v == "a" else (asc - desc) / 2 if v == "m" else -desc if v == "d" else 0.0)
        glyphs = bool(tracking) or live
        if glyphs:
            widths = [f.advance(ch, size) for ch in s]
            total = sum(widths) + tracking * (len(s) - 1)
        else:
            total = f.advance(s, size)
        h = anchor[0]
        x0 = x - (total if h == "r" else total / 2 if h == "m" else 0.0)
        if glyphs:
            cx = x0
            for ch, w in zip(s, widths):
                if ch != " ":
                    g = self._glyphs(ch, size, kind)
                    g.position.set(cx, self._Y(base), 0)
                    self._paint(g, color, alpha)
                cx += w + tracking
            return
        g = self._glyphs(s, size, kind)
        g.position.set(x0, self._Y(base), 0)
        self._paint(g, color, alpha)

    def rich(self, x, y, parts, size=28, alpha=1.0, anchor="ls"):
        widths = [self.text_width(t, size, k) for t, _, k in parts]
        total = sum(widths)
        cx = x - (total if anchor[0] == "r" else total / 2 if anchor[0] == "m" else 0)
        for (t, c, k), w in zip(parts, widths):
            self.text(cx, y, t, size=size, color=c, alpha=alpha, kind=k, anchor="l" + anchor[1])
            cx += w
        return total

    def _strip(self, pts, width, round_ends=True):
        """Triangles for a thick polyline (design pixels, y already flipped)."""
        P = np.asarray(pts, np.float64)
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
        P = P[keep]
        if len(P) < 2:
            return None, None
        hw = width / 2.0
        pos, idx = [], []
        for a, b in zip(P[:-1], P[1:]):
            d = b - a
            n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * hw
            k = len(pos)
            pos += [a + n, a - n, b + n, b - n]
            idx += [k, k + 1, k + 2, k + 1, k + 3, k + 2]
        # round joins and caps: small fans
        joins = P if round_ends else P[1:-1]
        seg = 10
        for c in joins:
            k = len(pos)
            pos.append(c)
            for s_ in range(seg + 1):
                ang = 2 * math.pi * s_ / seg
                pos.append(c + hw * np.array([math.cos(ang), math.sin(ang)]))
            for s_ in range(seg):
                idx += [k, k + 1 + s_, k + 2 + s_]
        pos = np.c_[np.asarray(pos), np.zeros(len(pos))].astype(np.float32)
        return pos, np.asarray(idx, np.uint32)

    def _dyn_mesh(self, pos, idx, color, alpha):
        m = self._acquire(("dyn",), lambda: tp.Mesh(tp.BufferGeometry(), self._material()))
        g = tp.BufferGeometry()
        g.set_attribute("position", pos)
        g.set_index(idx)
        m.set_geometry(g)
        m.frustum_culled = False
        m.position.set(0, 0, 0)
        self._paint(m, color, alpha)
        return m

    def line(self, pts, color=0xffffff, width=2.0, alpha=1.0):
        if alpha <= 0.003:
            return
        P = [(float(px), self._Y(float(py))) for px, py in pts]
        pos, idx = self._strip(P, width)
        if pos is not None:
            self._dyn_mesh(pos, idx, color, alpha)

    def dashed(self, p0, p1, color=0xffffff, width=2.0, alpha=1.0, dash=10, gap=7, phase=0.0):
        if alpha <= 0.003:
            return
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 1e-3:
            return
        u = (p1 - p0) / L
        s = -phase % (dash + gap) - (dash + gap)
        allpos, allidx = [], []
        while s < L:
            a, b = max(s, 0.0), min(s + dash, L)
            if b > a:
                A, B = p0 + u * a, p0 + u * b
                pos, idx = self._strip([(A[0], self._Y(A[1])), (B[0], self._Y(B[1]))], width, round_ends=False)
                if pos is not None:
                    allidx.append(idx + sum(len(q) for q in allpos))
                    allpos.append(pos)
            s += dash + gap
        if allpos:
            self._dyn_mesh(np.concatenate(allpos), np.concatenate(allidx), color, alpha)

    def circle(self, x, y, r, fill=None, outline=None, width=2.0, alpha=1.0):
        if alpha <= 0.003:
            return
        if fill is not None:
            m = self._acquire(("disc",), lambda: tp.Mesh(tp.CircleGeometry(1.0, 48), self._material()))
            m.position.set(x, self._Y(y), 0)
            m.scale.set(r, r, 1)
            self._paint(m, fill, alpha)
        if outline is not None:
            pts = [(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 64)]
            self.line(pts, outline, width, alpha)

    def arrow2d(self, p0, p1, color=0xffffff, width=2.5, head=12, alpha=1.0):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1e-3 or alpha <= 0.003:
            return
        u = d / L
        n = np.array([-u[1], u[0]])
        base = p1 - u * head
        self.line([p0, base + u * 1.0], color, width, alpha)
        tri = np.array([p1, base + n * head * 0.5, base - n * head * 0.5])
        pos = np.c_[tri[:, 0], self.H - tri[:, 1], np.zeros(3)].astype(np.float32)
        self._dyn_mesh(pos, np.array([0, 1, 2], np.uint32), color, alpha)

    def bar(self, x, y, w, h, frac, color, alpha=1.0, track=0x2a3342, radius=None):
        radius = h / 2 if radius is None else radius
        self._rrect(x, y, w, h, radius, track, alpha)
        fw = max(h, w * clamp01(frac))
        self._rrect(x, y, fw, h, radius, color, alpha)

    # maths ---------------------------------------------------------------------
    def _math_group(self, tex, size):
        """matplotlib mathtext -> glyph outlines -> one SVG path -> threepp meshes (cached per size)."""
        key = (tex, round(size, 1))
        hit = self._math_cache.get(key)
        if hit is not None:
            return hit
        dkey = f"{round(size, 1)}|{tex}"
        disk = self._math_disk.get(dkey)
        if disk is not None:
            info = (disk["svg"], disk["x0"], disk["y0"], disk["w"], disk["h"])
            self._math_cache[key] = info
            return info
        import matplotlib
        from matplotlib.font_manager import FontProperties
        from matplotlib.path import Path as MPath
        from matplotlib.textpath import TextPath
        with matplotlib.rc_context({"mathtext.fontset": "cm"}):
            tpath = TextPath((0, 0), tex, size=size, prop=FontProperties(size=size))
        V, C = tpath.vertices, tpath.codes
        d = []
        i = 0
        while i < len(V):
            c = C[i]
            x, y = V[i]
            if c == MPath.MOVETO:
                d.append(f"M{x:.3f},{-y:.3f}")
                i += 1
            elif c == MPath.LINETO:
                d.append(f"L{x:.3f},{-y:.3f}")
                i += 1
            elif c == MPath.CURVE3:
                (x1, y1), (x2, y2) = V[i], V[i + 1]
                d.append(f"Q{x1:.3f},{-y1:.3f} {x2:.3f},{-y2:.3f}")
                i += 2
            elif c == MPath.CURVE4:
                (x1, y1), (x2, y2), (x3, y3) = V[i], V[i + 1], V[i + 2]
                d.append(f"C{x1:.3f},{-y1:.3f} {x2:.3f},{-y2:.3f} {x3:.3f},{-y3:.3f}")
                i += 3
            elif c == MPath.CLOSEPOLY:
                d.append("Z")
                i += 1
            else:
                i += 1
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg"><path fill="#ffffff" fill-rule="nonzero" '
               f'd="{" ".join(d)}"/></svg>')
        ext = tpath.get_extents()
        info = (svg, float(ext.x0), float(ext.y0), float(ext.width), float(ext.height))
        self._math_cache[key] = info
        if self._math_file:
            import json
            self._math_disk[dkey] = dict(zip(("svg", "x0", "y0", "w", "h"), info))
            with open(self._math_file, "w", encoding="utf-8") as f:
                json.dump(self._math_disk, f, indent=0, sort_keys=True)
        return info

    def math_size(self, tex, size=34):
        _, _, _, w, h = self._math_group(tex, size)
        return w, h

    def math(self, x, y, tex, size=34, color=0xffffff, alpha=1.0, anchor="lm"):
        """Place maths; (x, y) is the anchor point ('l'/'m'/'r' + 't'/'m'/'b') of its ink box."""
        if alpha <= 0.003:
            return (0, 0)
        svg, x0, y0, w, h = self._math_group(tex, size)

        def make():
            g = tp.SVGLoader().parse(svg)          # y-flipped by the loader: back to y-up
            mat = self._material()
            for ch in g.children:
                ch.set_material(mat)
            return g
        g = self._acquire(("math", tex, round(size, 1)), make)
        for ch in g.children:
            ch.render_order = g.render_order
            self._paint(ch, color, alpha)
        left = x - (w if anchor[0] == "r" else w / 2 if anchor[0] == "m" else 0)
        top = y - (h if anchor[1] == "b" else h / 2 if anchor[1] == "m" else 0)
        # ink box: x0..x0+w, y0..y0+h in y-up TextPath units
        g.position.set(left - x0, self._Y(top) - (y0 + h), 0)
        return (w, h)

    # compound widgets ----------------------------------------------------------
    def caption(self, text, alpha, y=None, size=34, sub=None, width=1640, color=0xf2f5fa):
        """Lower-third caption: one or two balanced lines, centred, on a soft panel."""
        if alpha <= 0.003:
            return
        y = self.H - 118 if y is None else y
        lines = self.wrap_balanced(text, size, width)
        lh = size * 1.28
        widths = [self.text_width(l, size) for l in lines]
        w = max(widths) + 64
        h = lh * len(lines) + 34
        x = (self.W - w) / 2
        top = y - h / 2
        self.panel(x, top, w, h, radius=18, alpha=0.62 * alpha)
        for i, l in enumerate(lines):
            self.text(self.W / 2, top + 17 + i * lh + size * 0.02, l, size=size, color=color, alpha=alpha,
                      anchor="ma")

    def captions(self, t, captions):
        """All captions of a film, [(start, end, text)]: each fades in and out over its span.
        An entry (start, end, text, False) is spoken but not drawn (a line over a title card)."""
        for a0, b0, txt, *shown in captions:
            if shown and not shown[0]:
                continue
            al = envelope(t, a0, b0, 0.45, 0.4)
            if al > 0:
                self.caption(txt, al)

    def equation_card(self, t, rows, x=70, y=64, size=40, gap=12, min_width=470):
        """A card of equations, rows [(t_in, t_out, tex, note or None)]. Each row fades in and
        out over its span, with its note above it in small capitals, and the card is sized
        to the rows showing at t."""
        rows = [e for e in rows if e[0] - 0.1 <= t <= e[1] + 0.6]
        if not rows:
            return
        sizes = [self.math_size(tex, size) for _, _, tex, _ in rows]
        w = max(max(s[0] for s in sizes) + 60, min_width)
        h = sum(s[1] + (30 if note else gap - 4) for s, (_, _, _, note) in zip(sizes, rows)) + 40
        pa = max(envelope(t, a, b, 0.6, 0.6) for a, b, _, _ in rows)
        self.panel(x, y, w, h, radius=16, alpha=0.66 * pa, outline=0x8aa0c0, outline_alpha=0.16)
        yy = y + 22
        for (a, b, tex, note), (sw, sh) in zip(rows, sizes):
            al = envelope(t, a, b, 0.6, 0.6)
            if note:
                self.text(x + 30, yy, note.upper(), size=16, color=DIM, alpha=al, kind="semibold", tracking=2.2)
                yy += 24
            self.math(x + 30, yy + sh / 2, tex, size=size, color=TEXT, alpha=al, anchor="lm")
            yy += sh + gap

    def wrap_balanced(self, text, size, width, kind="regular"):
        """Wrap into the fewest lines, then even out their lengths (no orphans)."""
        n = len(self.wrap(text, size, width, kind))
        if n <= 1:
            return [text]
        lo, hi = min(200.0, width / 2), float(width)      # lo < hi: narrow widths too
        best = self.wrap(text, size, width, kind)
        for _ in range(18):
            mid = (lo + hi) / 2
            w = self.wrap(text, size, mid, kind)
            if len(w) <= n and all(self.text_width(l, size, kind) <= mid + 1 for l in w):
                best, hi = w, mid
            else:
                lo = mid
        return best

    def wrap(self, text, size, width, kind="regular"):
        words = [w for w in text.split(" ") if w]     # a no-break space (U+00A0) keeps its words together
        lines, cur = [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if self.text_width(cand, size, kind) <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def callout(self, anchor, label_xy, text, color=0xffffff, alpha=1.0, size=24, kind="semibold",
                dot=5, grow=1.0):
        """Leader line from a projected 3D point to a label."""
        if alpha <= 0.003:
            return
        ax, ay = anchor
        lx, ly = label_xy
        ex = ax + (lx - ax) * grow
        ey = ay + (ly - ay) * grow
        self.circle(ax, ay, dot, fill=color, alpha=alpha)
        self.line([(ax, ay), (ex, ey)], color, 1.6, alpha * 0.9)
        if grow > 0.98:
            left = lx < ax
            self.text(lx + (-10 if left else 10), ly, text, size=size, color=color, alpha=alpha, kind=kind,
                      anchor="rm" if left else "lm")

    def logplot(self, x, y, w, h, values, alpha=1.0, color=0xffb347, ymin=1e-4, ymax=1.0, xmax=None,
                title=None, threshold=None, unit_fmt=None, marker_last=True):
        """Small log-scale plot of a positive series (error vs iteration)."""
        if alpha <= 0.003:
            return
        self.panel(x, y, w, h, alpha=0.70 * alpha, outline=0x8aa0c0, outline_alpha=0.18)
        px, py, pw, ph = x + 70, y + 50, w - 92, h - 80
        if title:
            self.text(x + 22, y + 16, title, size=17, color=0x9fb2cc, alpha=alpha, kind="semibold", tracking=2.0)
        lo, hi = math.log10(ymin), math.log10(ymax)

        def Y(v):
            v = max(float(v), ymin)
            return py + ph * (1 - (math.log10(v) - lo) / (hi - lo))

        # only decades inside [ymin, ymax]: Y() clamps, so one below ymin would sit at ymin's height
        for dec in range(math.ceil(lo - 1e-9), math.floor(hi + 1e-9) + 1):
            yy = Y(10.0 ** dec)
            self.line([(px, yy), (px + pw, yy)], 0x3a4558, 1.0, alpha * 0.8)
            lab = unit_fmt(10.0 ** dec) if unit_fmt else f"1e{dec}"
            self.text(px - 10, yy, lab, size=15, color=0x7f8ea6, alpha=alpha, kind="numeric", anchor="rm")
        if threshold is not None:
            yy = Y(threshold)
            self.dashed((px, yy), (px + pw, yy), 0x5ee27a, 1.4, alpha * 0.85, dash=6, gap=5)
        n = len(values)
        xm = max(xmax or n, 2)

        def X(i):
            return px + pw * i / (xm - 1)

        if n >= 2:
            self.line([(X(i), Y(v)) for i, v in enumerate(values)], color, 2.6, alpha)
        for i, v in enumerate(values):
            self.circle(X(i), Y(v), 4.2 if (i == n - 1 and marker_last) else 3.0, fill=color, alpha=alpha)

    def plot(self, x, y, w, h, series, xlim, ylim, alpha=1.0, title=None, xlog=False, ylog=False,
             xticks=(), yticks=(), xfmt=str, yfmt=str, legend=True, markers=(), width=2.4):
        """Line plot on linear or log axes. series: [(xs, ys, colour, label or None)];
        points outside xlim/ylim are clamped to the frame. xticks/yticks: values that get a
        label (xfmt/yfmt) and, for y, a grid line. markers: [(x, y, colour)] dots."""
        if alpha <= 0.003:
            return
        self.panel(x, y, w, h, alpha=0.70 * alpha, outline=0x8aa0c0, outline_alpha=0.18)
        top = 52 if title else 24
        px, py, pw, ph = x + 70, y + top, w - 94, h - top - 46
        if title:
            self.text(x + 22, y + 16, title, size=17, color=DIM, alpha=alpha, kind="semibold", tracking=2.0)
        fx = math.log10 if xlog else float
        fy = math.log10 if ylog else float
        x0, x1 = fx(xlim[0]), fx(xlim[1])
        y0, y1 = fy(ylim[0]), fy(ylim[1])

        def X(v):
            return px + pw * clamp01((fx(max(v, xlim[0]) if xlog else v) - x0) / (x1 - x0))

        def Y(v):
            return py + ph * (1 - clamp01((fy(max(v, ylim[0]) if ylog else v) - y0) / (y1 - y0)))

        for v in yticks:
            yy = Y(v)
            self.line([(px, yy), (px + pw, yy)], 0x3a4558, 1.0, alpha * 0.8)
            self.text(px - 10, yy, yfmt(v), size=15, color=0x7f8ea6, alpha=alpha, kind="numeric", anchor="rm")
        for v in xticks:
            self.text(X(v), py + ph + 12, xfmt(v), size=15, color=0x7f8ea6, alpha=alpha, kind="numeric", anchor="ma")
        for xs, ys, color, _ in series:
            if len(xs) >= 2:
                self.line([(X(a), Y(b)) for a, b in zip(xs, ys)], color, width, alpha)
        for mx, my, color in markers:
            self.circle(X(mx), Y(my), 5.0, fill=color, alpha=alpha)
        if legend:
            ly = py + 4
            for _, _, color, label in series:
                if label:
                    self.line([(px + pw - 150, ly + 9), (px + pw - 126, ly + 9)], color, 3.0, alpha)
                    self.text(px + pw - 118, ly + 9, label, size=15, color=TEXT, alpha=alpha, anchor="lm")
                    ly += 22

    def readout(self, x, y, w, title, rows, alpha=1.0, row_h=40, value_size=26):
        """A panel of live values: rows [(label, value text, value colour or None)]."""
        if alpha <= 0.003:
            return
        self.panel(x, y, w, 58 + row_h * len(rows), radius=16, alpha=0.66 * alpha, outline=0x8aa0c0,
                   outline_alpha=0.16)
        self.text(x + 24, y + 22, title, size=16, color=DIM, alpha=alpha, kind="semibold", tracking=2.2)
        for j, (label, value, color) in enumerate(rows):
            yy = y + 58 + j * row_h + row_h / 2
            self.text(x + 24, yy, label, size=19, color=DIM, alpha=alpha, anchor="lm")
            self.text(x + w - 24, yy, value, size=value_size, color=TEXT if color is None else color, alpha=alpha,
                      kind="numeric", anchor="rm")

    def fade(self, amount, color=0x000000):
        """Full-frame veil, drawn last: 1 = solid colour."""
        if amount <= 0.003:
            return
        self._rrect(-4, -4, self.W + 8, self.H + 8, 0, color, amount)

    def image(self, x, y, w, h, texture, alpha=1.0):
        """A texture as a w x h rectangle, top-left at (x, y)."""
        if alpha <= 0.003:
            return

        def make():
            mat = self._material()
            mat.map = texture
            return tp.Mesh(tp.PlaneGeometry(1, 1), mat)
        key = ("image", id(texture))
        m = self._acquire(key, make)
        self._held[key] = texture       # alive while pooled, so no other texture can take its id
        m.position.set(x + w / 2, self._Y(y + h / 2), 0)
        m.scale.set(w, h, 1)
        m.material.opacity = clamp01(alpha)

    def colorbar(self, x, y, w, h, cmap, alpha=1.0, steps=48):
        """Horizontal gradient bar: cmap(t) -> 0xRRGGBB for t in [0, 1]."""
        if alpha <= 0.003:
            return
        sw = w / steps
        for k in range(steps):
            self._rrect(x + k * sw, y, sw + 0.6, h, 0, cmap((k + 0.5) / steps), alpha)

    def code(self, x, y, lines, lang, size=20, lh=27, alpha=1.0, line_alpha=None, hl=None, reveal=None,
             keywords=None, palette=None, accent=CODE_ACCENT):
        """Source lines in a monospace font with syntax colours (see `tokenize`). line_alpha(j),
        hl(j) (a highlight bar, 0..1) and reveal(j) (0..1 of the line's tokens shown) are
        optional per-line functions."""
        if alpha <= 0.003:
            return
        cw = self.text_width("0", size, "mono")
        for j, line in enumerate(lines):
            a = alpha * (line_alpha(j) if line_alpha else 1.0)
            if a <= 0.003 or not line.strip():
                continue
            yy = y + j * lh
            h = hl(j) if hl else 0.0
            if h > 0.003:
                self.panel(x - 16, yy - 2, len(max(lines, key=len)) * cw + 30, lh + 2, radius=5, fill=0x223452,
                           alpha=0.75 * h * alpha)
                self.panel(x - 16, yy - 2, 4, lh + 2, radius=2, fill=accent, alpha=h * alpha)
            toks = tokenize(line, lang, keywords, palette)
            n = len(toks) if reveal is None else int(math.ceil(reveal(j) * len(toks) - 1e-9))
            for col, s, c in toks[:n]:
                self.text(x + col * cw, yy + lh / 2, s, size=size, color=c, alpha=a, kind="mono", anchor="lm")
