"""lesson: the house layer of the threepp lesson films, on top of `threepp.lesson`.

The toolkit itself (Timeline, Stage, Hud, Film, Narration, Speech, ...) is the experimental
`threepp.lesson` package; see its docstring for the two-pass model (choreograph once, then
`render(t)` is pure). This module re-exports all of it and adds what makes these films look
and run alike:

`Stage`
    threepp.lesson.Stage with the lessons' defaults: GL, and `env` a file name in
    threepp_data/textures/env (the studio HDR unless told otherwise), found by `data_file`.

`Hud`
    threepp.lesson.Hud plus the house cards: `title_card` (the opening title) and `summary`
    ("IN SHORT" and the repo link). `card` and `code_card` are the house card and a card of
    code that types itself in.

`data_file`
    A threepp_data file from THREEPP_DATA_DIR, a checkout next to the repo, or the copies CMake
    fetches into the build directories.

`run`
    The command line every lesson shares: the film, a preview, one stretch, stills, a contact
    sheet, the captions as an .srt, a regression gate, or remeasuring the narration. A lesson
    provides `setup(width, height) -> (render, captions)`.

`house_rule_warnings`
    The house rules every lesson film follows (a narrated title card, no specific numbers in
    the narration), checked by `run`.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import math
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.dirname(os.path.dirname(_HERE))
if _PY not in sys.path:
    sys.path.insert(0, _PY)
import threepp as tp  # noqa: E402,F401  (the lessons take tp from here)
from threepp.lesson import *  # noqa: E402,F401,F403
from threepp.lesson import Hud as _Hud, Stage as _Stage  # noqa: E402


# ── threepp_data ──────────────────────────────────────────────────────────────
def data_roots():
    """Where threepp_data may be, in search order: THREEPP_DATA_DIR, a checkout next
    to the repo, then the copies CMake fetches into the build directories."""
    repo = os.path.dirname(_PY)
    roots = [os.environ.get("THREEPP_DATA_DIR", "")]
    roots += [os.path.join(os.path.dirname(repo), n) for n in ("threepp-data", "threepp_data")]
    roots += sorted(glob.glob(os.path.join(repo, "cmake-build-*", "_deps", "threepp_data-src")))
    return [r for r in roots if r and os.path.isdir(r)]


def data_file(*parts):
    """Path of a threepp_data file in the first root that has it. Roots are checked
    per file, so an older checkout that lacks it does not hide a newer copy."""
    rel = os.path.join(*parts)
    roots = data_roots()
    for r in roots:
        p = os.path.join(r, rel)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"threepp_data file '{rel}' not found in {roots or 'any data directory'}; "
                            f"set THREEPP_DATA_DIR to a threepp_data checkout that has it")


# ── the house stage and cards ─────────────────────────────────────────────────
class Stage(_Stage):
    """threepp.lesson.Stage with the lessons' defaults: GL, and `env` a file name in
    threepp_data/textures/env, found by `data_file`. A missing one is reported and the stage
    renders without image-based lighting."""

    def __init__(self, width=1920, height=1080, renderer="gl", env="empty_warehouse_01_2k.hdr", **kw):
        if renderer != "gl":
            raise ValueError(f"the lessons render with GL, not {renderer!r}")
        path = None
        if env:
            try:
                path = data_file("textures", "env", env)
            except FileNotFoundError as e:
                print(f"[stage] rendering without image-based lighting: {e}", file=sys.stderr)
        super().__init__(width, height, env=path, **kw)


class Hud(_Hud):
    """threepp.lesson.Hud plus the house cards: the opening title and the closing summary."""

    def title_card(self, t, kicker, title, subtitle, accent, t_in=0.9, t_out=6.4, size=96, sub_dy=118):
        """Opening title, left-aligned: a small tracked kicker, the title rising into place,
        and a subtitle `sub_dy` below it that fades in after it."""
        a = envelope(t, t_in, t_out, 0.9, 0.8)
        if a <= 0:
            return
        y = 360 + 14 * (1 - ease_out(remap(t, t_in, t_in + 1.3)))
        self.text(128, y - 58, kicker, size=22, color=accent, alpha=a, kind="semibold", tracking=4)
        self.text(122, y, title, size=size, color=TEXT, alpha=a, kind="semibold")
        sub = a * smooth(remap(t, t_in + 0.7, t_in + 1.7))
        self.text(128, y + sub_dy, subtitle, size=38, color=DIM, alpha=sub, kind="light")

    def summary(self, t, t0, t1, rows, footer, accent, link_color, text_dx=260, link="github.com/markaren/threepp"):
        """Closing card over the darkened frame, from t0 to past t1: "IN SHORT", rows of
        (tex, sentence) that appear one after another, then a footer and a link."""
        a = envelope(t, t0 + 0.2, t1 + 1, 0.8, 0.1)
        if a <= 0:
            return
        self.panel(-20, -20, self.W + 40, self.H + 40, radius=0, fill=0x070a10, alpha=0.8 * a)
        x = 250
        self.text(x, 250, "IN SHORT", size=22, color=accent, alpha=a, kind="semibold", tracking=4)
        for k, (tex, txt) in enumerate(rows):
            al = a * smooth(remap(t, t0 + 0.6 + 0.5 * k, t0 + 1.3 + 0.5 * k))
            yy = 350 + k * 110
            self.math(x, yy, tex, size=46, color=TEXT, alpha=al, anchor="lm")
            self.text(x + text_dx, yy, txt, size=38, color=TEXT, alpha=al, anchor="lm")
        al = a * smooth(remap(t, t0 + 2.4, t0 + 3.2))
        self.text(x, 780, footer, size=28, color=DIM, alpha=al, kind="semibold")
        self.text(x, 824, link, size=26, color=link_color, alpha=al)


# ── the house rules ───────────────────────────────────────────────────────────
TITLE_VOICE_BY = 2.0                     # the first spoken line starts this early: the title card is narrated
_NUMBER_RE = re.compile(r"\d(?![dD]\b)")   # a digit, except the one in "2D" / "3D"


def house_rule_warnings(captions):
    """The house rules every lesson film follows, as warnings: no specific numbers in the
    narration, and a narrated title card (a caption from the first seconds, spoken-only if the
    title card should stay clean). The picture is the lesson's own to check."""
    out = []
    if not captions or min(c[0] for c in captions) > TITLE_VOICE_BY:
        out.append(f"the title card is not narrated: no caption starts in the first {TITLE_VOICE_BY:.0f} s")
    for a, _, txt, *_ in captions:
        if _NUMBER_RE.search(txt):
            out.append(f"the caption at {a:.1f} s has a number in it: {txt[:70]!r}")
    return out


def card(ov, x, y, w, h, alpha, title=None):
    """The house card: a rounded panel with a small spaced-out title."""
    ov.panel(x, y, w, h, radius=16, alpha=0.74 * alpha, outline=0x8aa0c0, outline_alpha=0.16)
    if title:
        ov.text(x + 26, y + 22, title, size=16, color=DIM, alpha=alpha, kind="semibold", tracking=2.2)


def code_card(ov, t, t_in, t_out, x, y, lines, lang, title, size=20, lh=27, stagger=0.35, min_w=0, out=None,
              out_t=None, keywords=None, out_color=0xa6e3a1):
    """A titled card of code whose lines type in one after another from t_in. `out`: lines
    the program prints, shown under the code from `out_t` (default: once the code is in)."""
    a = envelope(t, t_in, t_out, 0.6, 0.6)
    if a <= 0.003:
        return
    cw = ov.text_width("0", size, "mono")
    w = max(max(len(l) for l in lines + (out or [])) * cw + 60, min_w)
    h = 64 + lh * len(lines) + (lh * len(out) + 26 if out else 0)
    card(ov, x, y, w, h, a, title)
    ov.code(x + 30, y + 50, lines, lang, size, lh, a,
            reveal=lambda j: remap(t, t_in + 0.3 + stagger * j, t_in + 0.9 + stagger * j), keywords=keywords)
    if out:
        oa = a * smooth(remap(t, out_t if out_t is not None else t_in + 0.9 + stagger * len(lines),
                              (out_t if out_t is not None else t_in + 0.9 + stagger * len(lines)) + 0.4))
        yy = y + 50 + lh * len(lines) + 10
        ov.panel(x + 12, yy - 4, w - 24, lh * len(out) + 12, radius=8, fill=0x070a10, alpha=0.8 * oa)
        for j, line in enumerate(out):
            ov.text(x + 30, yy + 2 + lh * j + lh / 2, line, size=size, color=out_color, alpha=oa, kind="mono",
                    anchor="lm")


# ── the command line ──────────────────────────────────────────────────────────
def _contact_sheet(thumbs, cols=6):
    th, tw = thumbs[0].shape[:2]
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, im in enumerate(thumbs):
        y, x = (k // cols) * th, (k % cols) * tw
        sheet[y:y + th, x:x + tw] = im
    return sheet


def _write_gate(out, name, captions, clips, narr, tm, duration, film_duration, frame, W, every):
    """`--gate DIR`: what a refactor of the toolkit must leave alone, from one setup().
    <name>.srt, the captions on the film clock (as --srt writes them); <name>.voice.json,
    every caption's clocks, text and spoken text, and every hold; <name>_sheet.png, one frame
    every `every` s (as --sheet makes it); and stills/, a full-size frame at the middle of
    every caption. Frames are rendered in film order, so a lesson that streams its pictures
    (warp_threepp's subprocess) never has to rewind."""
    os.makedirs(os.path.join(out, "stills"), exist_ok=True)
    write_srt(os.path.join(out, f"{name}.srt"), [(tm.film(a), tm.film(b), txt) for a, b, txt, *_ in captions])
    words = narr.words if narr is not None else None
    record = {
        "voice": narr.voice if narr is not None else None,
        "duration": duration, "film_duration": film_duration,
        "captions": [{"start": a, "end": b, "film_start": tm.film(a), "film_end": tm.film(b),
                      "text": txt, "drawn": bool(rest[0]) if rest else True, "spoken": spoken(txt, words),
                      "samples": None if clips is None else int(len(clips[k]))}
                     for k, (a, b, txt, *rest) in enumerate(captions)],
        "holds": [[a, d, list(tag) if isinstance(tag, tuple) else tag] for a, d, tag in tm.holds],
    }
    with open(os.path.join(out, f"{name}.voice.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, ensure_ascii=False, default=repr)

    f = max(1, W // 480)
    jobs = [(float(t), "sheet", k) for k, t in enumerate(np.arange(0.5, film_duration, every))]
    jobs += [(0.5 * (tm.film(a) + tm.film(b)), "still", k) for k, (a, b, *_) in enumerate(captions)]
    thumbs = {}
    t0 = time.time()
    for n, (t, kind, k) in enumerate(sorted(jobs)):
        img = frame(t)
        if kind == "sheet":
            thumbs[k] = shrink(img, f)
        else:
            write_png(os.path.join(out, "stills", f"{name}_cap{k:02d}_{t:07.2f}.png"), img)
        if n % 10 == 0:
            print(f"[gate] {n}/{len(jobs)} frames, {time.time() - t0:.0f}s", flush=True)
    write_png(os.path.join(out, f"{name}_sheet.png"), _contact_sheet([thumbs[k] for k in sorted(thumbs)]))
    print(f"[gate] wrote {out}: {len(captions)} captions, {len(tm.holds)} holds, "
          f"{len(thumbs)} sheet frames, {len(captions)} stills ({time.time() - t0:.0f}s)")


def run(name, duration, setup, fps=60, speech=None):
    """The command line every lesson shares.

    `setup(width, height)` builds the lesson at that render size and returns
    `(render, captions)`. `render(t)` gives the (H, W, 3) frame at t seconds on the
    lesson's own clock. A lesson that declares holds (`render.holds = [(t, seconds, tag)]`)
    takes `render(t, hold)`, where hold is (tag, 0..1) while the film stands still at t.
    `captions` is the [(start, end, text)] list the film shows, on the lesson's clock.
    An entry (start, end, text, False) is read aloud and written to the .srt but not drawn:
    use it to narrate the title card, so the film has sound from the start.

    The captions are read aloud (Kokoro, voice `--voice`) unless `--no-voice`. Where a
    line runs longer than its caption, the film holds the picture until it has been said.
    A lesson paced by its narration passes its `speech` (a Speech): its pronunciation table
    reaches the narrator, and `--remeasure` measures its final lines and rewrites the
    .speech.json its layout is read from.

        (no flags)            the film, 1920x1080 at `fps`, to <outdir>/<name>.mp4
        --preview             960x540 at 30 fps
        --from S --to S       one stretch of the film
        --stills 5,15,40      single frames, <outdir>/<name>_still_<t>.png
        --sheet [--every S]   contact sheet, one frame every S seconds from 0.5 s
        --srt                 the captions as <outdir>/<name>.srt
        --gate DIR            a regression record for refactors, into DIR (see `_write_gate`)
        --remeasure           say every line (Kokoro, --voice) and rewrite <lesson>.speech.json
    Times on the command line are film times.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help=f"the film (default: <outdir>/{name}.mp4)")
    ap.add_argument("--outdir", default="lesson_out", help="stills, sheets, subtitles and the default film")
    ap.add_argument("--stills", default=None, help="comma-separated times (s)")
    ap.add_argument("--sheet", action="store_true", help="contact sheet, one frame every --every seconds")
    ap.add_argument("--every", type=float, default=3.0)
    ap.add_argument("--srt", action="store_true", help=f"write the captions to <outdir>/{name}.srt")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice for the narration (default af_heart)")
    ap.add_argument("--no-voice", dest="no_voice", action="store_true", help="no narration: a silent film")
    ap.add_argument("--preview", action="store_true", help="960x540 @ 30 fps")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--gate", default=None, metavar="DIR",
                    help="write the captions, the spoken lines and holds, a sheet and a still per caption to DIR")
    ap.add_argument("--remeasure", action="store_true", help="measure every spoken line and rewrite the "
                    "lesson's .speech.json (the layout moves on the next run)")
    args = ap.parse_args()
    if args.remeasure and speech is None:
        ap.error("this lesson is not laid out from measured speech")
    if speech is not None and args.voice != speech.voice and not args.no_voice:
        print(f"[speech] the layout is timed for {speech.voice}; {args.voice} gets holds wherever it runs long")

    W, H = (960, 540) if args.preview else (1920, 1080)
    fps = args.fps or (30 if args.preview else fps)
    render, captions = setup(W, H)
    for msg in house_rule_warnings(captions):
        print("[house rules]", msg)
    os.makedirs(args.outdir, exist_ok=True)
    if speech is not None:
        if args.remeasure:
            speech.remeasure(os.path.join(args.outdir, "voice_cache"), args.voice)
            return
        stale = speech.stale()
        if stale:
            print(f"[speech] these lines changed since they were measured: {stale} (--remeasure)")

    tm = TimeMap(getattr(render, "holds", ()))
    clips = narr = None
    if not args.no_voice:
        narr = Narration(args.voice, cache_dir=os.path.join(args.outdir, "voice_cache"),
                         words=speech.words if speech is not None else None)
        t0 = time.time()
        clips = [narr.clip(c[2]) for c in captions]
        said = sum(len(c) for c in clips) / Narration.RATE
        for _, t, d in overruns([c[:2] for c in captions], [len(c) / Narration.RATE for c in clips],
                                [(a, d) for a, d, _ in tm.holds]):
            tm.add(t, d, ("voice", None))
        held = sum(d for _, d, tag in tm.holds if tag[0] == "voice")
        print(f"[voice] {len(clips)} lines, {said:.0f} s of speech ({args.voice}) in {time.time() - t0:.1f}s; "
              f"the film holds {held:.1f} s for it")
    film_duration = tm.film(duration)
    takes_hold = len(inspect.signature(render).parameters) >= 2

    def frame(t):
        ts, hold = tm.script(t)
        return render(ts, hold) if takes_hold else render(ts)

    if args.gate:
        _write_gate(args.gate, name, captions, clips, narr, tm, duration, film_duration, frame, W, args.every)
        return
    if args.srt:
        p = os.path.join(args.outdir, f"{name}.srt")
        write_srt(p, [(tm.film(a), tm.film(b), txt) for a, b, txt, *_ in captions])
        print("saved", p, f"({len(captions)} captions)")
        return
    if args.stills:
        for tok in args.stills.split(","):
            t = float(tok)
            p = os.path.join(args.outdir, f"{name}_still_{t:06.2f}.png")
            write_png(p, frame(t))
            print("saved", p)
        return
    if args.sheet:
        f = max(1, W // 480)
        thumbs = [shrink(frame(float(t)), f) for t in np.arange(0.5, film_duration, args.every)]
        p = os.path.join(args.outdir, f"{name}_sheet.png")
        write_png(p, _contact_sheet(thumbs))
        print("saved", p, f"({len(thumbs)} frames, every {args.every} s from 0.5 s, row-major)")
        return

    audio = None
    if clips is not None:
        track = np.zeros(int(math.ceil(film_duration * Narration.RATE)) + 1, np.float32)
        for (a, *_), c in zip(captions, clips):
            i0 = int(round((tm.film(a) + VOICE_LEAD) * Narration.RATE))
            n = min(len(c), len(track) - i0)
            track[i0:i0 + n] += c[:n]
        audio = os.path.join(args.outdir, f"{name}.voice.wav")
        write_wav(audio, track, Narration.RATE)
    t_to = args.t_to if args.t_to is not None else film_duration
    out = args.out or os.path.join(args.outdir, f"{name}.mp4")
    film = Film(out, W, H, fps=fps, crf=16 if not args.preview else 22,
                preset="slow" if not args.preview else "veryfast", audio=audio, audio_offset=args.t_from)
    nfr = int(round((t_to - args.t_from) * fps))
    t_start = time.time()
    for f in range(nfr):
        t = args.t_from + f / fps
        film.write(frame(t))
        if f % (fps * 5) == 0:
            el = time.time() - t_start
            print(f"[film] {t:6.1f}s  frame {f}/{nfr}  {el / max(f, 1) * 1000:.0f} ms/frame", flush=True)
    path = film.close()
    print(f"[film] wrote {path} ({nfr} frames, {time.time() - t_start:.0f}s)")
