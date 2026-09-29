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

__all__ = ["write_png", "shrink", "ffmpeg_exe", "Film", "write_srt", "write_wav", "read_wav"]


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


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class Film:
    """Frames -> H.264 mp4 through an ffmpeg pipe. Written to <path>.part, renamed on close.
    `audio`: a WAV muxed in as AAC, starting `audio_offset` seconds into it."""

    def __init__(self, path, width, height, fps=60, crf=16, preset="slow", threads=0, audio=None, audio_offset=0.0):
        self.path = path
        self.tmp = path + ".part.mp4"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if audio:
            cmd += ["-ss", f"{audio_offset:.3f}", "-i", audio, "-map", "0:v", "-map", "1:a",
                    "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-profile:v", "high", "-movflags", "+faststart", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-colorspace", "bt709"]
        if threads:
            cmd += ["-threads", str(threads)]
        cmd += [self.tmp]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")
        os.replace(self.tmp, self.path)
        return self.path


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
