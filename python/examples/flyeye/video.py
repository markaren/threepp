"""mp4 of a live eye run: one column per eye, composed on the GPU and piped to ffmpeg.

Rows: the render; what the 721 receptor columns receive (each hex cell filled with its
BoxEye value); the circuit's T4/T5 motion field; and, when truth is given, the true flow
field on the same lattice. Fields use one colour code: hue = image direction (x right,
y up; red = right, yellow-green = up, cyan = left, blue-violet = down), brightness =
magnitude, full at field_full (circuit, a.u.) and truth_full (true flow, rad/s).
"""

from __future__ import annotations

import math
import subprocess

import torch

from .lattice import HexLattice


class VideoOut:
    GAP, TEXT = 4, 44

    def __init__(self, path, lattice: HexLattice, n_eyes: int, fps: int = 100, size: int = 403,
                 yaws=(45.0, -45.0), truth: bool = False, field_full: float = 1.0, truth_full: float = 1.0):
        import imageio_ffmpeg
        from PIL import Image, ImageDraw

        self.Image, self.ImageDraw = Image, ImageDraw
        self.size, self.yaws, self.n = size, yaws, n_eyes
        self.field_full, self.truth_full = field_full, truth_full
        self.rows = 4 if truth else 3
        rc = lattice.pixel_rc(size).float().cuda()
        yy, xx = torch.meshgrid(torch.arange(size, device="cuda"), torch.arange(size, device="cuda"), indexing="ij")
        pix = torch.stack([yy.flatten(), xx.flatten()], 1).float()
        d, i = zip(*(torch.cdist(c, rc).min(1) for c in pix.split(16384)))
        self.index = torch.cat(i).view(size, size)  # nearest column per pixel
        self.inside = (torch.cat(d) <= 8.5).view(size, size)  # outside the lattice stays black
        self.W = n_eyes * size + (n_eyes - 1) * self.GAP
        self.H = self.TEXT + self.rows * size + (self.rows - 1) * self.GAP
        self.W, self.H = self.W + self.W % 2, self.H + self.H % 2
        self.last = None  # the last composed frame, (H, W, 3) uint8 numpy
        self.proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.W}x{self.H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    @staticmethod
    def field_rgb(f, full: float = 1.0):
        """(2, 721) image-space flow -> (721, 3) uint8, HSV with S = 1, V = |f| / full."""
        h = (torch.atan2(f[1], f[0]) / (2 * math.pi)) % 1.0
        v = (f.norm(dim=0) / full).clamp(0, 1)
        k = (torch.tensor([5.0, 3.0, 1.0], device=f.device)[:, None] + 6 * h) % 6
        rgb = v * (1 - torch.minimum(k, 4 - k).clamp(0, 1))
        return (rgb.T * 255).to(torch.uint8)

    def write(self, colors, receptors, field, line1, line2, truth=None):
        S, img = self.size, torch.zeros(self.H, self.W, 3, dtype=torch.uint8, device="cuda")
        for e in range(self.n):
            x0, y = e * (S + self.GAP), self.TEXT
            img[y:y + S, x0:x0 + S] = colors[e][..., [2, 1, 0]]
            y += S + self.GAP
            g = (receptors[e].clamp(0, 1) * 255).to(torch.uint8)[self.index] * self.inside
            img[y:y + S, x0:x0 + S] = g[..., None]
            y += S + self.GAP
            img[y:y + S, x0:x0 + S] = self.field_rgb(field[e].float(), self.field_full)[self.index] * self.inside[..., None]
            if truth is not None and self.rows == 4:
                y += S + self.GAP
                img[y:y + S, x0:x0 + S] = self.field_rgb(truth[e].float(), self.truth_full)[self.index] * self.inside[..., None]
        im = self.Image.fromarray(img.cpu().numpy())
        dr = self.ImageDraw.Draw(im)
        dr.text((6, 4), line1, fill=(255, 255, 255))
        dr.text((6, 22), line2, fill=(255, 255, 255))
        labels = ["render", "721 receptor columns (BoxEye input)",
                  f"T4/T5 field: hue = direction, full = {self.field_full:g} a.u."]
        if self.rows == 4:
            labels.append(f"true flow (box mean): full = {self.truth_full:g} rad/s")
        for e, yaw in enumerate(self.yaws[:self.n]):
            x0 = e * (S + self.GAP) + 4
            for row, label in enumerate(labels):
                label = f"eye yaw {yaw:+.0f} deg: {label}" if row == 0 else label
                y = self.TEXT + row * (S + self.GAP) + 4
                dr.rectangle((x0 - 2, y - 2, x0 + 6 * len(label) + 2, y + 12), fill=(0, 0, 0))
                dr.text((x0, y), label, fill=(255, 255, 255))
        buf = im.tobytes()
        self.last = im
        self.proc.stdin.write(buf)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
