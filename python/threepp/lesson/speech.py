"""Narration: caption text as a TTS model should read it (`spoken`), read aloud by Kokoro
(`Narration`), and a lesson's lines with their measured lengths, from which its captions
are laid out (`Speech`, `overruns`)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from types import MappingProxyType

import numpy as np


from .media import read_wav, write_wav
from .timing import TimeMap, Timeline

__all__ = ["DEFAULT_WORDS", "spoken", "Narration", "VOICE_LEAD", "VOICE_TAIL", "SLACK", "overruns", "Speech"]


# How the narrator says a few names, for every lesson (read-only: a lesson adds its own with
# `words=`). Kokoro's G2P reads "PhysX" as "fize-ex" and runs "three.js" together; a
# [word](/phonemes/) link fixes the pronunciation outright.
DEFAULT_WORDS = MappingProxyType({"IMU": "I M U", "GNSS": "G N S S", "fps": "frames per second",
                                  "threepp": "three p p", "FR3": "F R 3", "PhysX": "[PhysX](/fˈɪzˌɛks/)",
                                  "three.js": "three J S"})
_UNITS = [("m/s²", "metres per second squared"), ("°/s", "degrees per second"),
          ("ms", "milliseconds"), ("mm", "millimetres"), ("cm", "centimetres"), ("s", "seconds"),
          ("m", "metres"), ("%", "percent"), ("°", "degrees")]

_UNIT_RE = re.compile(r"(\d)[  ]?(" + "|".join(re.escape(u) for u, _ in _UNITS) + r")(?![A-Za-z²])")


def spoken(text, words=None):
    """Caption text as it should be read aloud: units after numbers, and a few acronyms, in words.
    `words` {written: spoken} extends DEFAULT_WORDS; substitutions run in that order."""
    t = text.replace("×", " by ")
    units = dict(_UNITS)
    t = _UNIT_RE.sub(lambda m: m.group(1) + " " + units[m.group(2)], t)
    for k, v in {**DEFAULT_WORDS, **(words or {})}.items():
        t = re.sub(r"(?<![\w])" + re.escape(k) + r"(?![\w])", v, t)
    return t.replace(" ", " ")


class Narration:
    """Captions read aloud by Kokoro (offline neural TTS, hexgrad/Kokoro-82M), one clip per
    caption, cached as WAV by (voice, speed, spoken text) so a re-render only synthesises the
    lines that changed."""

    RATE = 24000

    def __init__(self, voice="af_heart", speed=1.0, cache_dir="lesson_out/voice_cache", words=None):
        self.voice, self.speed, self.cache_dir, self.words = voice, speed, cache_dir, words
        self._pipe = None
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, text):
        say = spoken(text, self.words)
        key = hashlib.sha1(f"{self.voice}|{self.speed}|{say}".encode("utf-8")).hexdigest()[:16]
        return say, os.path.join(self.cache_dir, f"{self.voice}_{key}.wav")

    def clip(self, text):
        say, path = self._path(text)
        if os.path.isfile(path):
            return read_wav(path)[0]
        self._synth(say, path)
        return read_wav(path)[0]

    def word_times(self, text):
        """[(word, start, end)] in seconds into the clip, as Kokoro aligned them (punctuation
        dropped), so a picture can cut on a spoken word. Cached beside the clip's WAV."""
        say, path = self._path(text)
        side = path[:-4] + ".words.json"
        if not os.path.isfile(side):
            self._synth(say, path)                  # the WAV again too, so the two always agree
        with open(side, encoding="utf-8") as f:
            return [tuple(w) for w in json.load(f)]

    def _synth(self, say, path):
        if self._pipe is None:
            try:
                from kokoro import KPipeline
            except ImportError as e:
                raise RuntimeError("narration needs Kokoro (pip install kokoro soundfile); "
                                   "or render without it with --no-voice") from e
            # Kokoro voice names start with their language code: a = American, b = British English, ...
            self._pipe = KPipeline(lang_code=self.voice[0], repo_id="hexgrad/Kokoro-82M")
        chunks, words, off = [], [], 0.0
        for r in self._pipe(say, voice=self.voice, speed=self.speed):
            a = np.asarray(r.audio, np.float32)
            for tk in r.tokens or ():
                if tk.start_ts is not None and any(ch.isalnum() for ch in tk.text):
                    words.append((tk.text.strip(".,;:!?"), off + tk.start_ts, off + tk.end_ts))
            chunks.append(a)
            off += len(a) / self.RATE
        write_wav(path, np.concatenate(chunks), self.RATE)
        with open(path[:-4] + ".words.json", "w", encoding="utf-8") as f:
            json.dump([[w, round(a, 3), round(b, 3)] for w, a, b in words], f)


VOICE_LEAD, VOICE_TAIL = 0.15, 0.45     # speech starts this far into a caption, and ends this far before its end
SLACK = VOICE_LEAD + VOICE_TAIL + 0.3    # a caption laid out from its speech outlasts it by this much


def overruns(spans, seconds, holds=(), margin=0.0):
    """[(i, t, extra)]: every caption i whose line needs more time than it gets, and how much
    more, placed at t, the moment its speech should have ended (VOICE_TAIL before its end).
    A line gets the time from VOICE_LEAD into its caption to t, plus any holds (t, d) in that
    window: the picture stands still there, the voice goes on. Each overrun found counts as a
    hold for the captions after it. `spans` [(start, end)] and `seconds` go in step."""
    holds, out = list(holds), []
    for i, ((a, b), need) in enumerate(zip(spans, seconds)):
        q, p = a + VOICE_LEAD, b - VOICE_TAIL
        have = p - q + sum(d for h, d in holds if q <= h < p)
        if need + margin > have:
            out.append((i, p, need + margin - have))
            holds.append((p, need + margin - have))
    return out


class Speech:
    """A lesson's lines by key, and how long each takes to say. A lesson whose narration sets
    its pace lays its timeline out from these lengths, so the lengths must not move by
    themselves: they are measured once (`--remeasure`) and committed next to the lesson as
    <lesson>.speech.json, {voice: {key: {"s": seconds, "said": spoken text}}}. A line with no
    measurement gets an estimate from its word count, reported; a line whose text changed
    since it was measured is reported by `run`. `words` is the lesson's own pronunciation
    table (see `spoken`); `voice_only` lines are read aloud but not drawn."""

    def __init__(self, lines, cache, voice="af_heart", words=None, voice_only=()):
        self.lines, self.cache, self.voice = dict(lines), cache, voice
        self.words, self.voice_only = dict(words or {}), set(voice_only)
        self.measured = {}
        if os.path.isfile(cache):
            with open(cache, encoding="utf-8") as fh:
                self.measured = json.load(fh).get(voice, {})
        self.estimated, self.final = [], {}

    def seconds(self, k):
        m = self.measured.get(str(k))
        if m is not None:
            return m["s"]
        if self.lines[k] is None:
            raise KeyError(f"line {k!r} has no text yet and no measurement in {self.cache}")
        if k not in self.estimated:
            self.estimated.append(k)
            print(f"[speech] line {k!r} is not measured for {self.voice}; estimating from its words "
                  f"(--remeasure measures it)")
        return 0.42 * len(self.lines[k].split()) + 0.4

    def span(self, k, start, slack=SLACK):
        """Caption k from `start`, as long as its line takes to say, plus `slack`."""
        return (start, start + self.seconds(k) + slack)

    def layout(self, plan, gap=0.2, slack=SLACK):
        """(Timeline, {key: (start, end)}) from a plan of beats [(name, lead-in, keys, tail)]:
        each beat's captions follow each other `gap` apart, after its lead-in, and the beat
        ends `tail` after its last one (or after its lead-in's start when it has none)."""
        tl, cap, t = Timeline(), {}, 0.0
        for name, lead, keys, tail in plan:
            c = t + lead
            for k in keys:
                cap[k] = (c, c + self.seconds(k) + slack)
                c = cap[k][1] + gap
            end = (c - gap if keys else t) + tail
            tl.add(name, t, end)
            t = end
        return tl, cap

    def stretch(self, spans, holds=(), margin=0.3):
        """For captions written at fixed times, {key: (start, end)}: a TimeMap whose `.film`
        maps the written clock onto one with more time just before a caption ends, wherever
        its line (plus `margin`) would run over. `holds` [(t, d)] on the written clock count
        as time to speak."""
        keys = list(spans)
        found = overruns([spans[k] for k in keys], [self.seconds(k) for k in keys], holds, margin)
        return TimeMap([(t, d, ("speech", keys[i])) for i, t, d in found])

    def captions(self, spans, fields=None, texts=None):
        """[(start, end, text)] for run(), one per {key: (start, end)}. `texts` holds whole
        lines written after the choreography (a line in `lines` may be None until then);
        `fields` fills the {names} in the rest. Voice-only lines get the not-drawn flag."""
        out = []
        for k, (a, b) in spans.items():
            txt = texts[k] if texts and k in texts else self.lines[k]
            if fields is not None:
                txt = txt.format(**fields)
            self.final[k] = txt
            out.append((a, b, txt) + ((False,) if k in self.voice_only else ()))
        return out

    def stale(self):
        """The lines whose final text is not what was measured."""
        return [k for k, txt in self.final.items()
                if str(k) in self.measured and self.measured[str(k)].get("said") != spoken(txt, self.words)]

    def remeasure(self, cache_dir, voice=None):
        """Say every final line with Kokoro (through the voice cache) and write the lengths."""
        voice = voice or self.voice
        narr = Narration(voice, cache_dir=cache_dir, words=self.words)
        data = {}
        if os.path.isfile(self.cache):
            with open(self.cache, encoding="utf-8") as fh:
                data = json.load(fh)
        old = data.get(voice, {})
        new = {str(k): {"s": round(len(narr.clip(txt)) / Narration.RATE, 2), "said": spoken(txt, self.words)}
               for k, txt in self.final.items()}
        for k, v in new.items():
            if old.get(k, {}).get("s") != v["s"]:
                print(f"[speech] {k}: {old.get(k, {}).get('s')} -> {v['s']} s")
        data[voice] = new
        with open(self.cache, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print(f"[speech] wrote {self.cache}: {len(new)} lines for {voice}")
