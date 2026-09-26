"""Event-camera and lidar tiles for the jelly-wreck sensor mosaic.

EventPanel  the renderer's GPU DVS detector (per-pixel log-intensity crossings)
            on the presented hero frame, drawn as event polarity on black.
LidarPanel  a 32-beam spinning lidar on a post beside the set, traced through
            the renderer's own acceleration structure (scan_lidar), drawn as a
            point splat from a fixed 3/4 viewpoint.

Both are read after the hero render of the same film frame.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import threepp as tp


# --- event camera -----------------------------------------------------------

class EventPanel:
    """DVS tile: brightening events cyan, darkening events magenta, on black.

    The detector runs on the presented frame (event_camera_source 'final'), so
    the hero keeps its TAA jitter; the sensor box-averages that frame down to
    (w, h), which also averages most of the jitter out. Each pixel shows its
    latest event and fades at `decay` per RENDERER frame (a host that renders
    twice per film frame fades twice as fast).

    Latency: the accumulator read is the oldest ring slot (no device wait), two
    renderer frames behind the hero render it is called after.

    Slow motion: the detector keeps its per-pixel reference until a crossing
    fires, so a slow brightness change still integrates up to the threshold and
    the 1/8x impact keeps firing; the hero camera moves on film time, so its
    ego-motion fires at full rate throughout.
    """

    POS = np.float32([70, 205, 255])
    NEG = np.float32([255, 45, 125])
    DEAD = 4                          # |v - 128| at or below this is "no event" (decay residue)

    def __init__(self, renderer, w, h, threshold=0.16, decay=0.80):
        self.r = renderer
        self.w, self.h = int(w), int(h)
        renderer.event_camera_source = "final"
        renderer.set_event_camera_resolution(self.w, self.h)
        renderer.set_event_camera_params(threshold=threshold, decay=decay, min_luma=0.01,
                                         max_events_per_pixel=5)
        renderer.event_camera_enabled = True
        s = (np.arange(256, dtype=np.float32) - 128.0)
        s = np.sign(s) * np.maximum(np.abs(s) - self.DEAD, 0.0) / (127.0 - self.DEAD)
        p = np.sqrt(np.clip(s, 0.0, 1.0))[:, None]        # sqrt: fading trails stay legible
        n = np.sqrt(np.clip(-s, 0.0, 1.0))[:, None]
        self.lut = np.clip(p * self.POS + n * self.NEG, 0.0, 255.0).astype(np.uint8)
        self.ms = 0.0

    def frame(self):
        t0 = time.perf_counter()
        v = self.r.read_event_camera_visualisation()
        if v.size == 0:
            return np.zeros((self.h, self.w, 3), np.uint8)
        img = self.lut[v]
        if img.shape[:2] != (self.h, self.w):             # sensor clamped to a smaller swapchain
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize((self.w, self.h), Image.BILINEAR))
        self.ms = 1e3 * (time.perf_counter() - t0)
        return img


# --- lidar ------------------------------------------------------------------

def _turbo(t):
    """Turbo colormap (polynomial fit), t in [0, 1] -> (N, 3) float in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    c = np.float32([[0.13572138, 0.09140261, 0.10667330],
                    [4.61539260, 2.19418839, 12.64194608],
                    [-42.66032258, 4.84296658, -60.58204836],
                    [132.13108234, -14.18503333, 110.36276771],
                    [-152.94239396, 4.27729857, -89.90310912],
                    [59.28637943, 2.82956604, 27.34824973]])
    out = c[5]
    for k in range(4, -1, -1):
        out = out * t + c[k]
    return np.clip(out, 0.0, 1.0)


def _look_at(eye, target):
    eye = np.asarray(eye, np.float32)
    f = np.asarray(target, np.float32) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    return eye, np.stack([r, np.cross(r, f), f]).astype(np.float32)   # rows: right, up, forward


class LidarPanel:
    """Spinning lidar tile: 32 beams over +/-22.5 deg (1.45 deg apart), 1024
    columns per revolution (0.35 deg), 12 m range; one full revolution per film
    frame, traced at the instant of the hero render (the TLAS it last built).

    Clear jelly returns only its Fresnel reflection, so the detector threshold
    is set low enough (2e-5) that off-normal faces still register; the chrome
    ball returns only the cap facing the sensor. The stage haze is a look, not
    an atmosphere: a LIDAR medium below the floor replaces the scene fog for
    the scan, so the cloud carries no fog back-scatter.

    The cloud is drawn from a fixed viewpoint above the set, points coloured by
    height (turbo, 0-2.6 m) and faded with distance, over a 1 m ground grid.
    `show_sensor` adds the post and sensor head to the scene; the beams start
    inside the head, so min_range (0.25 m) skips its own housing.
    """

    ORIGIN = (-2.7, 1.2, 2.1)        # sensor head (m): off the camera orbit, clear of the ball
    VIEW_EYE = (1.5, 4.6, 8.4)
    VIEW_TARGET = (-0.4, 0.75, 1.2)
    VIEW_FOV = 32.0                  # vertical, deg
    SS = 2                           # supersampling (the 2x2 box-down in frame() assumes 2)
    SPLAT_R = 1.8                    # splat radius in supersampled pixels
    H_MAX = 2.6                      # height at the top of the colormap (m)

    def __init__(self, renderer, scene, w, h, beams=32, vfov=(-22.5, 22.5), columns=1024,
                 max_range=12.0, show_sensor=True):
        self.r = renderer
        self.w, self.h = int(w), int(h)
        el = np.radians(np.linspace(vfov[0], vfov[1], beams))[:, None]
        az = (2.0 * math.pi / columns) * np.arange(columns)[None, :]
        d = np.stack([np.cos(el) * np.sin(az), np.sin(el) * np.ones_like(az),
                      np.cos(el) * np.cos(az)], -1)
        self.dirs = np.ascontiguousarray(d.reshape(-1, 3), np.float32)
        self.origins = np.ascontiguousarray(np.broadcast_to(np.float32(self.ORIGIN), self.dirs.shape))
        self.params = tp.LidarParams()
        self.params.max_range = max_range
        self.params.min_range = 0.25
        self.params.detector_threshold = 2e-5
        self.params.medium_extinction = 1e-4
        self.params.medium_surface_y = -100.0
        self.n_points = 0
        self.ms = {"scan": 0.0, "draw": 0.0}

        self.eye, self.rot = _look_at(self.VIEW_EYE, self.VIEW_TARGET)
        self.W, self.H = self.w * self.SS, self.h * self.SS
        self.fpx = 0.5 * self.H / math.tan(math.radians(0.5 * self.VIEW_FOV))
        rr = int(math.ceil(self.SPLAT_R))
        oy, ox = np.mgrid[-rr:rr + 1, -rr:rr + 1]
        keep = ox ** 2 + oy ** 2 <= self.SPLAT_R ** 2 + 0.5
        self.off = (oy[keep] * self.W + ox[keep]).astype(np.int64)
        self.rr = rr
        self.bg = self._background()
        if show_sensor:
            self._add_sensor(scene)

    def _project(self, p):
        """World points (N, 3) -> pixel x, pixel y, depth (supersampled frame)."""
        c = (np.asarray(p, np.float32) - self.eye) @ self.rot.T
        z = np.maximum(c[:, 2], 1e-3)
        return (0.5 * self.W + self.fpx * c[:, 0] / z,
                0.5 * self.H - self.fpx * c[:, 1] / z, c[:, 2])

    def _background(self):
        W, H = self.W, self.H
        yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
        bg = np.broadcast_to((1.0 - yy) * np.float32([6, 7, 10]) + yy * np.float32([12, 14, 19]),
                             (H, W, 3)).copy()
        flat = bg.reshape(-1, 3)
        # ground grid, 1 m, faded with distance
        ts = np.linspace(-9.0, 9.0, 3601, dtype=np.float32)
        for g in range(-9, 10):
            for pts in (np.stack([np.full_like(ts, g), np.zeros_like(ts), ts], 1),
                        np.stack([ts, np.zeros_like(ts), np.full_like(ts, g)], 1)):
                px, py, z = self._project(pts)
                ok = (z > 0.3) & (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 2)
                a = np.clip(1.3 - 0.075 * z[ok], 0.2, 1.0)[:, None] * np.float32([20, 24, 32])
                i = py[ok].astype(np.int64) * W + px[ok].astype(np.int64)
                flat[i] = flat[i] + a
                flat[i + W] = flat[i + W] + 0.5 * a
        # sensor post and head
        o = np.float32(self.ORIGIN)
        post = np.stack([np.full(400, o[0]), np.linspace(0.0, o[1], 400), np.full(400, o[2])], 1)
        px, py, _ = self._project(post)
        for dx in (-1, 0, 1):
            flat[py.astype(np.int64) * W + px.astype(np.int64) + dx] = np.float32([64, 68, 78])
        hx, hy, _ = self._project(o[None, :])
        yy, xx = np.mgrid[0:H, 0:W]
        d2 = (xx - hx[0]) ** 2 + (yy - hy[0]) ** 2
        bg[d2 <= (3.4 * self.SS) ** 2] = np.float32([235, 238, 245])
        bg[d2 <= (1.8 * self.SS) ** 2] = np.float32([40, 44, 52])
        bg = np.concatenate([np.clip(bg, 0.0, 255.0).astype(np.uint8), np.full((H, W, 1), 255, np.uint8)], 2)
        return np.ascontiguousarray(bg).view(np.uint32)[..., 0]       # (H, W) packed RGBA

    def _add_sensor(self, scene):
        from demo_common import standard_material
        ox, oy, oz = self.ORIGIN
        dark = standard_material(tp.Color(0.05, 0.05, 0.055), 0.35, 0.6)
        housing = standard_material(tp.Color(0.55, 0.56, 0.58), 0.3, 1.0)
        window = standard_material(tp.Color(0.01, 0.01, 0.012), 0.05, 0.0)
        g = tp.Group()
        g.position.set(ox, 0.0, oz)
        base = tp.Mesh(tp.CylinderGeometry(0.15, 0.17, 0.03, 40), dark)
        base.position.y = 0.015
        post = tp.Mesh(tp.CylinderGeometry(0.02, 0.02, oy - 0.04, 16), dark)
        post.position.y = 0.5 * (oy - 0.04)
        head = tp.Mesh(tp.CylinderGeometry(0.052, 0.052, 0.08, 40), housing)
        head.position.y = oy
        band = tp.Mesh(tp.CylinderGeometry(0.0535, 0.0535, 0.03, 40), window)
        band.position.y = oy
        for m in (base, post, head, band):
            m.cast_shadow = True
            g.add(m)
        scene.add(g)
        self.sensor = g

    def frame(self, scene, camera, sim_t):
        t0 = time.perf_counter()
        ret = self.r.scan_lidar(self.origins, self.dirs, self.params)
        t1 = time.perf_counter()
        hit = (ret["return_no"] > 0) & (ret["return_kind"] == 0)
        p = ret["position"][hit]
        self.n_points = int(hit.sum())
        W, H, rr = self.W, self.H, self.rr
        buf = self.bg.copy()
        flat = buf.reshape(-1)
        if len(p):
            px, py, z = self._project(p)
            ok = (z > 0.2) & (px >= rr) & (px < W - rr - 1) & (py >= rr) & (py < H - rr - 1)
            px, py, z, y = px[ok], py[ok], z[ok], p[ok, 1]
            col = _turbo(0.06 + 0.88 * np.clip(y / self.H_MAX, 0.0, 1.0)) * 255.0
            col *= (np.clip(1.3 - 0.04 * z, 0.5, 1.0) * np.where(y < 0.05, 0.75, 1.0))[:, None]   # floor a step down
            order = np.argsort(-z)                                     # far first: near points overwrite
            base = (py.astype(np.int64) * W + px.astype(np.int64))[order]
            idx = (base[:, None] + self.off[None, :]).ravel()
            c8 = np.concatenate([col[order].astype(np.uint8), np.full((len(order), 1), 255, np.uint8)], 1)
            flat[idx] = np.repeat(c8.view(np.uint32)[:, 0], len(self.off))
        a = buf.view(np.uint8).reshape(H, W, 4)[..., :3].astype(np.uint16)
        out = ((a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) >> 2).astype(np.uint8)
        t2 = time.perf_counter()
        self.ms = {"scan": 1e3 * (t1 - t0), "draw": 1e3 * (t2 - t1)}
        return out
