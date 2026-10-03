"""The command line every lesson shares (`run`): the film, a preview, one stretch, stills, a
contact sheet, the captions as SubRip, a regression record of frames and narration
(`write_gate`), or remeasuring the narration. `contact_sheet` tiles frames."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time

import numpy as np

from .media import Film, shrink, write_png, write_srt, write_wav
from .speech import VOICE_LEAD, Narration, overruns, spoken
from .timing import TimeMap

__all__ = ["contact_sheet", "write_gate", "run"]


def contact_sheet(thumbs, cols=6):
    """Frames of one size tiled row-major, `cols` to a row."""
    th, tw = thumbs[0].shape[:2]
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, im in enumerate(thumbs):
        y, x = (k // cols) * th, (k % cols) * tw
        sheet[y:y + th, x:x + tw] = im
    return sheet


def write_gate(out, name, captions, clips, narr, tm, duration, film_duration, frame, W, every):
    """`--gate DIR`: what a refactor of the toolkit must leave alone, from one setup().
    <name>.srt, the captions on the film clock (as --srt writes them); <name>.voice.json,
    every caption's clocks, text and spoken text, and every hold; <name>_sheet.png, one frame
    every `every` s (as --sheet makes it); and stills/, a full-size frame at the middle of
    every caption. Frames are rendered in film order, so a lesson that streams its pictures
    from a subprocess never has to rewind."""
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
    write_png(os.path.join(out, f"{name}_sheet.png"), contact_sheet([thumbs[k] for k in sorted(thumbs)]))
    print(f"[gate] wrote {out}: {len(captions)} captions, {len(tm.holds)} holds, "
          f"{len(thumbs)} sheet frames, {len(captions)} stills ({time.time() - t0:.0f}s)")


def run(name, duration, setup, fps=60, speech=None, warn=None, argv=None):
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
    .speech.json its layout is read from. `warn(captions)`, if given, returns notes about
    the captions (a house's rules), printed before anything is rendered. `argv` stands in
    for the command line.

        (no flags)            the film, 1920x1080 at `fps`, to <outdir>/<name>.mp4
        --preview             960x540 at 30 fps
        --from S --to S       one stretch of the film
        --stills 5,15,40      single frames, <outdir>/<name>_still_<t>.png
        --sheet [--every S]   contact sheet, one frame every S seconds from 0.5 s
        --srt                 the captions as <outdir>/<name>.srt
        --gate DIR            a regression record for refactors, into DIR (see `write_gate`)
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
    args = ap.parse_args(argv)
    if args.remeasure and speech is None:
        ap.error("this lesson is not laid out from measured speech")
    if speech is not None and args.voice != speech.voice and not args.no_voice:
        print(f"[speech] the layout is timed for {speech.voice}; {args.voice} gets holds wherever it runs long")

    W, H = (960, 540) if args.preview else (1920, 1080)
    fps = args.fps or (30 if args.preview else fps)
    render, captions = setup(W, H)
    if warn is not None:
        for msg in warn(captions):
            print("[captions]", msg)
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
        write_gate(args.gate, name, captions, clips, narr, tm, duration, film_duration, frame, W, args.every)
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
        write_png(p, contact_sheet(thumbs))
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
    nfr = int(round((t_to - args.t_from) * fps))
    t_start = time.time()
    with Film(out, W, H, fps=fps, crf=16 if not args.preview else 22,
              preset="slow" if not args.preview else "veryfast", audio=audio, audio_offset=args.t_from) as film:
        for f in range(nfr):
            t = args.t_from + f / fps
            film.write(frame(t))
            if f % (fps * 5) == 0:
                el = time.time() - t_start
                print(f"[film] {t:6.1f}s  frame {f}/{nfr}  {el / max(f, 1) * 1000:.0f} ms/frame", flush=True)
    print(f"[film] wrote {film.path} ({nfr} frames, {time.time() - t_start:.0f}s)")
