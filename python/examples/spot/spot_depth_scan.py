"""Forward-facing DepthSensor -> local elevation map -> the 45-cell height scan the spotv2 policy reads.

The spotv2 policy was trained on a PRIVILEGED scan: the exact analytic terrain height at 45 heading-
relative grid points (-0.35..1.1 m forward x +/-0.30 m lateral). That is ground truth no real robot has.
This module replaces it with onboard PERCEPTION, the way a real legged robot does it:

  1. A `tp.DepthSensor` is rigidly mounted on the robot body (forward + up a little, pitched DOWN ~40deg),
     mirroring Spot's real forward-looking depth cameras.
  2. Each control tick it renders the scene from that viewpoint and reprojects to a world-space point
     cloud (the robot's own body is hidden during the scan = perfect self-filtering).
  3. The cloud is fused into an accumulating 2.5-D ELEVATION MAP (a world-anchored height grid, per-cell
     EMA). Accumulation is what makes a forward camera usable for the full grid: cells now beside/under/
     behind the robot were *ahead* of it a moment ago, so they are remembered. Unobserved cells read as
     "flat" (delta 0) until the camera sweeps over them -- exactly the gradual reveal a real robot gets.
  4. The 45 heading-relative grid points are sampled from the map and returned as the same
     `ahead = clip(height - height_under_base, -1, 1)` vector the analytic scan produced -> drop-in.

`ForwardDepthScanner` also owns two optional visualizations (both auto-hidden during the scan so the
sensor never sees them): the raw colored point cloud, and the 45-cell grid the policy actually consumes.

Shared by all the spotv2 play_* viewers; see e.g. play_spot_steps.py / play_spot_heightfield.py.
"""
import math

import numpy as np

import threepp as tp
from spot_deploy import _quat_to_R, _quat_from_R
from spot_terrain_env import SCAN_GX, SCAN_GY, N_SCAN, scan_xy_np


def _height_ramp(z, lo=0.0, hi=0.8):
    """Map height z (m) to a blue(low)->green->red(high) ramp -> per-point colors [N,3] for the cloud."""
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(t - 1.0) * 2.0, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(t - 0.5) * 2.5, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(t - 0.0) * 2.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32)


def _delta_color(d):
    """Height-delta d in [-1,1] -> 0xRRGGBB: blue (drop) -> green (level) -> red (rise). For grid markers."""
    if d >= 0.0:
        r, g, b = 0.2 + 0.8 * d, 0.9 - 0.7 * d, 0.2
    else:
        r, g, b = 0.2, 0.9 + 0.6 * d, 0.3 - 0.6 * d
    c = lambda v: int(max(0.0, min(1.0, v)) * 255)
    return (c(r) << 16) | (c(g) << 8) | c(b)


class ForwardDepthScanner:
    """A body-mounted forward depth camera + accumulating elevation map producing the spotv2 height scan.

    bounds = (x0, x1, y0, y1): world extent of the elevation map (cover the demo's terrain; points/queries
    outside read as flat ground, which is what is actually there). Reset the map with clear_map() whenever
    the robot is teleported (spawn / R / level change) so stale terrain under the new pose is not reused.
    """

    def __init__(self, renderer, scene, robot_meshes, bounds, *,
                 cell=0.04, mount_fwd=0.30, mount_up=0.15, pitch_deg=40.0,
                 fov_y=90.0, width=200, height=150, near=0.05, far=10.0,
                 noise=0.0, ema=0.6, query_radius=0.05,
                 show_cloud=True, show_grid=True):
        self.renderer = renderer
        self.scene = scene
        self.cell = float(cell)
        self.ema = float(ema)
        self.qr = max(1, int(round(query_radius / self.cell)))      # query half-window in cells

        self.x0, self.x1, self.y0, self.y1 = (float(v) for v in bounds)
        self.nx = int(math.ceil((self.x1 - self.x0) / self.cell)) + 1
        self.ny = int(math.ceil((self.y1 - self.y0) / self.cell)) + 1
        self.H = np.full((self.nx, self.ny), np.nan, np.float32)     # world elevation map (per-cell EMA)
        self.h_here_last = 0.0

        # body-frame mount: camera-local -> body rotation. The DepthSensor looks down its local -z; this
        # maps that to "body +x forward, pitched DOWN by pitch_deg" with image-up tilted forward.
        p = math.radians(pitch_deg)
        sp, cp = math.sin(p), math.cos(p)
        self.R_mount = np.array([[0.0, sp, -cp],
                                 [-1.0, 0.0, 0.0],
                                 [0.0, cp, sp]], float)
        self.mount_local = np.array([mount_fwd, 0.0, mount_up], float)

        self.sensor = tp.DepthSensor(fov_y=fov_y, width=width, height=height, near=near, far=far)
        self.sensor.range_noise = noise

        # 45 heading-relative query offsets (forward-major), reused from the env so the layout matches.
        self.gx = np.asarray(SCAN_GX, np.float32)
        self.gy = np.asarray(SCAN_GY, np.float32)

        self.hide = list(robot_meshes)                              # hidden during the scan (self-filter)

        # ---- optional viz: raw point cloud (fixed-capacity dynamic Points) ----
        self.show_cloud = show_cloud
        self.cloud = None
        if show_cloud:
            self.cap = int(width) * int(height)
            g = tp.BufferGeometry()
            g.set_attribute("position", np.zeros((self.cap, 3), np.float32))
            g.set_attribute("color", np.zeros((self.cap, 3), np.float32))
            g.set_draw_range(0, 0)
            pm = tp.PointsMaterial(); pm.size = 0.03; pm.size_attenuation = True; pm.vertex_colors = True
            self.cloud = tp.Points(g, pm); self.cloud.frustum_culled = False
            self.cloud_geom = g
            scene.add(self.cloud)
            self.hide.append(self.cloud)

        # ---- optional viz: the 45-cell scan the policy actually consumes ----
        self.show_grid = show_grid
        self.markers = []
        if show_grid:
            grp = tp.Group()
            for _ in range(N_SCAN):
                mk = tp.Mesh(tp.SphereGeometry(0.022, 8, 6), tp.MeshStandardMaterial())
                mk.material.color = 0x22ff88
                self.markers.append(mk); grp.add(mk)
            self.marker_group = grp
            scene.add(grp)
            self.hide.append(grp)

    def clear_map(self):
        """Forget the elevation map (call on any teleport so old terrain is not sampled under a new pose)."""
        self.H[:] = np.nan
        self.h_here_last = 0.0

    def prewarm(self, rs, n=4):
        """Populate the map from the current (stationary) pose so the first control step is not blind."""
        for _ in range(n):
            self.scan(rs)

    # ---------------------------------------------------------------- accumulate / sample
    def _accumulate(self, pts):
        if pts.shape[0] == 0:
            return
        ix = np.floor((pts[:, 0] - self.x0) / self.cell).astype(np.int64)
        iy = np.floor((pts[:, 1] - self.y0) / self.cell).astype(np.int64)
        m = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        if not m.any():
            return
        flat = ix[m] * self.ny + iy[m]
        ncell = self.nx * self.ny
        s = np.bincount(flat, weights=pts[m, 2], minlength=ncell)
        c = np.bincount(flat, minlength=ncell)
        seen = c > 0
        mean = np.zeros(ncell, np.float32)
        mean[seen] = (s[seen] / c[seen]).astype(np.float32)
        Hf = self.H.reshape(-1)
        fresh = seen & np.isnan(Hf)                                  # never-seen cells: take the observation
        Hf[fresh] = mean[fresh]
        upd = seen & ~np.isnan(Hf)                                   # seen before: EMA toward observation
        Hf[upd] = (1.0 - self.ema) * Hf[upd] + self.ema * mean[upd]

    def _sample(self, qx, qy):
        """Highest mapped height within a small window of each (qx,qy); NaN where wholly unobserved."""
        cix = np.floor((qx - self.x0) / self.cell).astype(np.int64)
        ciy = np.floor((qy - self.y0) / self.cell).astype(np.int64)
        best = np.full(qx.shape, np.nan, np.float32)
        for dx in range(-self.qr, self.qr + 1):
            for dy in range(-self.qr, self.qr + 1):
                ix, iy = cix + dx, ciy + dy
                ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
                vals = np.full(qx.shape, np.nan, np.float32)
                vals[ok] = self.H[ix[ok], iy[ok]]
                both = ~np.isnan(best) & ~np.isnan(vals)
                best = np.where(np.isnan(best), vals, best)
                best[both] = np.maximum(best[both], vals[both])
        return best

    # ---------------------------------------------------------------- the scan
    def scan(self, rs):
        """rs = articulation root_state [x,y,z, qx,qy,qz,qw]. Returns (ahead[45] float32, h_here float).

        Renders one depth frame from the body-mounted camera, fuses it into the elevation map, and reads
        the 45-cell heading-relative scan + the height under the base out of the map.
        """
        x, y, z = float(rs[0]), float(rs[1]), float(rs[2])
        R = _quat_to_R(rs[3:7])                                      # body->world

        wp = np.array([x, y, z]) + R @ self.mount_local
        self.sensor.position.set(float(wp[0]), float(wp[1]), float(wp[2]))
        q = _quat_from_R(R @ self.R_mount)                          # camera-local -> world
        self.sensor.quaternion.set(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        saved = [(o, o.visible) for o in self.hide]                 # self-filter: hide robot + own viz
        for o, _ in saved:
            o.visible = False
        pts = self.sensor.scan(self.renderer, self.scene)
        for o, v in saved:
            o.visible = v

        self._accumulate(pts)

        hx, hy = float(R[0, 0]), float(R[1, 0]); nrm = math.hypot(hx, hy) or 1.0
        cyaw, syaw = hx / nrm, hy / nrm
        qpx, qpy = scan_xy_np(x, y, cyaw, syaw)                     # 45 world query points
        qpx = np.asarray(qpx, np.float32); qpy = np.asarray(qpy, np.float32)
        h_grid = self._sample(qpx, qpy)

        h_here = self._sample(np.array([x], np.float32), np.array([y], np.float32))[0]
        if math.isnan(h_here):
            h_here = self.h_here_last
        else:
            self.h_here_last = float(h_here)

        ahead = np.where(np.isnan(h_grid), 0.0, h_grid - h_here).astype(np.float32)
        ahead = np.clip(ahead, -1.0, 1.0)

        self._update_viz(pts, qpx, qpy, h_grid, h_here)
        return ahead, float(h_here)

    def _update_viz(self, pts, qpx, qpy, h_grid, h_here):
        if self.show_cloud and self.cloud is not None:
            self.cloud.visible = True
            n = min(int(pts.shape[0]), self.cap)
            if n:
                self.cloud_geom.update_attribute("position", np.ascontiguousarray(pts[:n], np.float32))
                self.cloud_geom.update_attribute("color", _height_ramp(pts[:n, 2]))
            self.cloud_geom.set_draw_range(0, n)
        elif self.cloud is not None:
            self.cloud.visible = False

        if self.show_grid and self.markers:
            self.marker_group.visible = True
            for i, mk in enumerate(self.markers):
                hi = h_grid[i]
                seen = not math.isnan(hi)
                hz = float(hi) if seen else float(h_here)
                mk.position.set(float(qpx[i]), float(qpy[i]), hz + 0.03)
                if seen:
                    mk.material.color = _delta_color(float(np.clip((hz - h_here) / 0.2, -1.0, 1.0)))
                else:
                    mk.material.color = 0x3a3f47                    # dim grey = not yet observed
        elif self.markers:
            self.marker_group.visible = False
