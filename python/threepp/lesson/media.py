"""Frames and sound out: PNG without an imaging library, SubRip captions, WAV, and H.264
through an ffmpeg pipe (`Film`)."""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import wave
import zlib

import numpy as np

__all__ = ["write_png", "shrink", "ffmpeg_exe", "rawvideo_args", "FramePipe", "Film", "write_srt", "write_wav",
           "read_wav"]


def write_png(path, rgb):
    """Write an (H, W, 3) uint8 array as a PNG with only the standard library."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))   # filter type 0 per row

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def shrink(rgb, factor):
    """Box-filter downsample by an integer factor."""
    h, w = rgb.shape[0] // factor * factor, rgb.shape[1] // factor * factor
    a = rgb[:h, :w].astype(np.float32).reshape(h // factor, factor, w // factor, factor, -1)
    return a.mean(axis=(1, 3)).round().astype(np.uint8)


def ffmpeg_exe(prefer="bundled"):
    """An ffmpeg binary: imageio-ffmpeg's bundled one, or the one on PATH, whichever is found
    first in the order `prefer` gives ("bundled" or "path"). None if there is neither."""
    def bundled():
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:                      # noqa: BLE001 - an optional dependency
            return None
    for find in ((bundled, lambda: shutil.which("ffmpeg")) if prefer == "bundled"
                 else (lambda: shutil.which("ffmpeg"), bundled)):
        exe = find()
        if exe:
            return exe
    return None


def rawvideo_args(exe, width, height, fps, loglevel="error", hide_banner=False):
    """The start of an ffmpeg command that reads raw RGB frames of width x height from stdin."""
    return [exe, "-y"] + (["-hide_banner"] if hide_banner else []) + [
        "-loglevel", loglevel, "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-"]


class FramePipe:
    """(H, W, 3) uint8 frames written into the stdin of an ffmpeg command (`rawvideo_args` plus
    the output options). Nothing touches the disk but what ffmpeg writes. `log`: an open file
    that takes ffmpeg's stdout and stderr. `close()` returns ffmpeg's exit code."""

    def __init__(self, cmd, log=None):
        self.cmd = cmd
        redirect = {"stdout": log, "stderr": log} if log is not None else {}
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, **redirect)
        self.n = 0

    def write(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        return self.p.wait()

    def abort(self):
        """Stop ffmpeg where it is (the frame loop failed); what it wrote so far stays."""
        try:
            self.p.stdin.close()
        except OSError:
            pass
        self.p.kill()
        self.p.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()


class Film(FramePipe):
    """Frames -> H.264 mp4 through an ffmpeg pipe. Written to <path>.part.mp4 and renamed by
    `close()`; as a context manager, closed when the block ends and aborted (the part file
    removed) when it raises. `audio`: a WAV muxed in as AAC, starting `audio_offset` seconds
    into it."""

    def __init__(self, path, width, height, fps=60, crf=16, preset="slow", threads=0, audio=None, audio_offset=0.0):
        exe = ffmpeg_exe()
        if exe is None:
            raise RuntimeError('Film needs ffmpeg: pip install imageio-ffmpeg (or "threepp[lesson]"), or put an '
                               "ffmpeg executable on PATH")
        self.path = path
        self.tmp = path + ".part.mp4"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cmd = rawvideo_args(exe, width, height, fps)
        if audio:
            cmd += ["-ss", f"{audio_offset:.3f}", "-i", audio, "-map", "0:v", "-map", "1:a",
                    "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-profile:v", "high", "-movflags", "+faststart", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-colorspace", "bt709"]
        if threads:
            cmd += ["-threads", str(threads)]
        super().__init__(cmd + [self.tmp])

    def close(self):
        rc = super().close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")
        os.replace(self.tmp, self.path)
        return self.path

    def abort(self):
        """Stop ffmpeg and remove the part file: a failed loop leaves no film behind."""
        super().abort()
        try:
            os.remove(self.tmp)
        except OSError:
            pass


def write_srt(path, captions):
    """Captions [(start s, end s, text)] as a SubRip (.srt) subtitle file."""
    def stamp(s):
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    with open(path, "w", encoding="utf-8") as f:
        for k, (a, b, txt, *_) in enumerate(captions, 1):
            f.write(f"{k}\n{stamp(a)} --> {stamp(b)}\n{txt}\n\n")


def write_wav(path, samples, rate):
    """Mono float samples in [-1, 1] as a 16-bit WAV, with only the standard library."""
    pcm = (np.clip(np.asarray(samples, np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32767
        return data, w.getframerate()
