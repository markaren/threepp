"""What the examples share that needs nothing but numpy and threepp.

`warp_common.py` is the films' shared module and it imports Warp, so anything
put there is a Warp dependency for whoever uses it. The hello-world demos, the
pure-renderer showcases and the renderer-only kits (`fireworks.py`,
`drone_rig.py`) must keep running on `pip install threepp` alone, so the
pieces they share live here instead, and `warp_common` re-exports them for
the films.
"""
import shutil
import subprocess
import sys

import numpy as np

import threepp as tp

# --- command line --------------------------------------------------------------


def cli_arg(flag, default, cast):
    """`--flag value` from sys.argv, or `default`. A bare flag also yields default."""
    if flag in sys.argv:
        k = sys.argv.index(flag)
        if k + 1 < len(sys.argv) and not sys.argv[k + 1].startswith("--"):
            return cast(sys.argv[k + 1])
    return default


def parse_size(text):
    """'1280x720' -> (1280, 720)."""
    w, h = text.lower().split("x")
    return int(w), int(h)


# --- ffmpeg --------------------------------------------------------------------


def find_ffmpeg():
    """The ffmpeg binary on PATH, or imageio-ffmpeg's bundled one, or None."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                      # noqa: BLE001 - optional dependency
        return None


class Encoder:
    """A pipe to x264, fed raw RGB frames.

    The film demos all learned the same lesson: writing a PNG per frame and
    encoding the directory afterwards costs more than the render does (the hull
    film paid 5.4 GB of intermediates and eighteen minutes for eighty seconds of
    picture). So `read_pixels()` off the frame already on the GPU goes straight
    down this pipe and nothing touches the disk but the mp4.

    The keyword flags exist because the films disagree about them and their
    output must not change: `preset`, `faststart`, `an`, `hide_banner`,
    `loglevel`, `vf` (a filter, e.g. the odd-size fixup) and `log` (a path this
    class opens and closes, or an already-open handle it only writes to).
    """

    def __init__(self, path, w, h, fps, crf=18, preset=None, faststart=False,
                 an=True, hide_banner=True, loglevel="warning", vf=None,
                 log=None, ffmpeg=None):
        exe = ffmpeg or find_ffmpeg()
        if exe is None:
            raise RuntimeError("no ffmpeg on PATH and no imageio-ffmpeg")
        cmd = [exe, "-y"]
        if hide_banner:
            cmd += ["-hide_banner"]
        cmd += ["-loglevel", loglevel, "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if an:
            cmd += ["-an"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf)]
        if preset:
            cmd += ["-preset", preset]
        if faststart:
            cmd += ["-movflags", "+faststart"]
        if vf:
            cmd += ["-vf", vf]
        self.cmd = cmd + [path]
        self._own_log = isinstance(log, str)
        self.log = open(log, "w") if self._own_log else log
        redirect = {"stdout": self.log, "stderr": self.log} if self.log is not None else {}
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, **redirect)
        self.n = 0

    def send(self, rgb):
        """One RGB frame, HxWx3 uint8."""
        self.p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        """Close the pipe and wait for the encoder; returns its exit code."""
        self.p.stdin.close()
        rc = self.p.wait()
        if self._own_log:
            self.log.close()
        return rc


def encode_png_sequence(pattern, path, fps, crf=18, preset=None, faststart=False,
                        an=False, vf=None, loglevel="error", ffmpeg=None, check=True):
    """Encode an already-written PNG sequence (`pattern` is an ffmpeg %0Nd path).

    The other half of the film encoders: the shots that render to disk first,
    either because the frames are wanted as stills too or because the run is
    assembled from segments afterwards.
    """
    exe = ffmpeg or find_ffmpeg()
    if exe is None:
        raise RuntimeError("no ffmpeg on PATH and no imageio-ffmpeg")
    cmd = [exe, "-y", "-loglevel", loglevel, "-framerate", str(fps), "-i", pattern]
    if an:
        cmd += ["-an"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf)]
    if preset:
        cmd += ["-preset", preset]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    return subprocess.run(cmd + [path], check=check)


# --- Radiance .hdr -------------------------------------------------------------


def encode_rgbe(rgb):
    """Vectorised linear-RGB float -> Radiance RGBE bytes, shape (H, W, 4)."""
    rgb = np.maximum(np.asarray(rgb, np.float64), 0.0)
    m = rgb.max(axis=2)
    mask = m >= 1e-32
    safe = np.where(mask, m, 1.0)
    mant, exp = np.frexp(safe)             # m = mant * 2**exp,  mant in [0.5, 1)
    scale = np.where(mask, mant * 256.0 / safe, 0.0)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    for c in range(3):
        out[..., c] = np.clip(rgb[..., c] * scale, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(mask, np.clip(exp + 128, 0, 255), 0).astype(np.uint8)
    return out


def write_radiance_hdr(path, rgb):
    """Write an (H, W, 3) linear float array as an uncompressed Radiance .hdr.

    Four demos build their own equirect sky -- each a hand-tuned look, so the
    pixels stay where they are -- and then all four end the same way: encode to
    RGBE, dodge the RLE signature, write the two-line header. That tail is what
    lives here. Returns `path`, to hand straight to RGBELoader.
    """
    rgbe = encode_rgbe(rgb)
    # An RGBE row starting (2, 2, <128) reads as "adaptive RLE" to a .hdr
    # reader; nudge the one pixel that could fake that signature, so stb takes
    # the scanlines uncompressed.
    if rgbe[0, 0, 0] == 2 and rgbe[0, 0, 1] == 2 and rgbe[0, 0, 2] < 128:
        rgbe[0, 0, 0] = 3
    h, w = rgbe.shape[:2]
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (h, w))
        f.write(rgbe.tobytes())
    return path


# --- scene plumbing ------------------------------------------------------------


def resize_handler(camera, renderer):
    """The window-resize callback every example installs."""
    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)
    return on_resize


def standard_material(color, roughness=1.0, metalness=0.0, **props):
    """MeshStandardMaterial with the common knobs set (the defaults are the
    material's own); extra keywords are assigned as attributes (side=,
    emissive=, ...)."""
    m = tp.MeshStandardMaterial()
    m.color = color
    m.roughness = roughness
    m.metalness = metalness
    for k, v in props.items():
        setattr(m, k, v)
    return m
