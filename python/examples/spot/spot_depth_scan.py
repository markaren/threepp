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
    """Map height z (m) to a blue(low)->green->red(high) ramp -> per-point colors [N,3] for the cloud.

    lo/hi are ABSOLUTE world heights, so callers must band them around whatever ground the robot is
    standing on. Leaving the old lo=0/hi=0.8 defaults on terrain whose surface sits at z~7 m clamps
    every point to t=1 and paints the whole cloud flat red — the ramp carries no information at all.
    _update_viz bands them around h_here for exactly this reason.

    The ramp stops are authored as DISPLAY (sRGB) colors — (1, 0.25, 0) is meant
    to READ as deep red. threepp vertex colors are linear and every renderer
    output path applies the sRGB OETF (the Vulkan point overlay included, since
    the overlay color-parity fix), so decode the stops to linear here; encoded
    back at display they reproduce the authored ramp instead of washing the top
    end to orange.
    """
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(t - 1.0) * 2.0, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(t - 0.5) * 2.5, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(t - 0.0) * 2.0, 0.0, 1.0)
    c = np.stack([r, g, b], axis=1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


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

    Backend: tp.DepthSensor.scan is backend-neutral (GL raster / Vulkan path-traced), so this works with a
    GLRenderer or a VulkanRenderer. The robot self-filter here (hiding meshes during the scan) relies on the
    GL sensor RE-rendering the scene; on Vulkan the scan traces the renderer's TLAS from the last render(),
    so a Vulkan deploy must instead render() each frame BEFORE scanning and exclude the robot another way
    (e.g. LidarReturn.hit_instance_id), and the body must not sit in the sensor's field of view.
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
            # Last colour pushed per marker. The Vulkan backend caches derived
            # material state and re-reads it only when the material's version
            # changes, so `mk.material.color = ...` alone is INVISIBLE there —
            # every marker keeps its construction colour for the whole run.
            # tp.flush_material() bumps the version (no-op on GL); tracking the
            # last value keeps that to the handful of markers that actually
            # changed shade this scan instead of all 45 every time.
            self._mk_color = [0x22ff88] * N_SCAN

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
    def _place(self, rs):
        """Put the sensor on its body mount from root state rs = [x,y,z, qx,qy,qz,qw]."""
        x, y, z = float(rs[0]), float(rs[1]), float(rs[2])
        R = _quat_to_R(rs[3:7])                                      # body->world

        wp = np.array([x, y, z]) + R @ self.mount_local
        self.sensor.position.set(float(wp[0]), float(wp[1]), float(wp[2]))
        q = _quat_from_R(R @ self.R_mount)                           # camera-local -> world
        self.sensor.quaternion.set(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _process(self, pts, rs):
        """Fuse a cloud into the map, then read the 45-cell scan at the pose rs. -> (ahead, h_here).

        The map is world-anchored, so `pts` may come from a fire issued several frames ago; the QUERY
        points must be built from the CURRENT rs, which is why the pose is passed in separately.
        """
        x, y = float(rs[0]), float(rs[1])
        R = _quat_to_R(rs[3:7])

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

    def scan(self, rs):
        """rs = articulation root_state [x,y,z, qx,qy,qz,qw]. Returns (ahead[45] float32, h_here float).

        Renders one depth frame from the body-mounted camera, fuses it into the elevation map, and reads
        the 45-cell heading-relative scan + the height under the base out of the map.

        SYNCHRONOUS: on Vulkan this blocks until the GPU trace behind the in-flight frame retires
        (~17-30 ms in the spot_slam scene). Interactive viewers should prefer scan_fire/scan_harvest.
        """
        self._place(rs)

        saved = [(o, o.visible) for o in self.hide]                 # self-filter: hide robot + own viz
        for o, _ in saved:
            o.visible = False
        pts = self.sensor.scan(self.renderer, self.scene)
        for o, v in saved:
            o.visible = v

        return self._process(pts, rs)

    # ---------------------------------------------------------------- pipelined scan (fire / harvest)
    def scan_fire(self, rs):
        """Place the sensor at rs and FIRE an async scan; the beams snapshot this pose now.

        Intended frame pattern (Vulkan):

            harvest = scanner.scan_harvest(rs)   # BEFORE render — free if a render happened since the fire
            ...
            rend.render(scene, camera)
            if frame % SCAN_EVERY == 0:
                scanner.scan_fire(rs)            # AFTER render — traces the TLAS this render just built

        The fire and its harvest are separated by at least one render(), so the collect finds the trace
        already retired and costs ~nothing instead of stalling on the in-flight frame. The elevation map
        is world-anchored, so fusing a cloud that is a few frames old is correct; only the 45 query
        points must be current-pose-relative, and scan_harvest builds them from the rs it is given.

        Returns True if a scan was fired. A fire while one is still outstanding is REFUSED rather
        than issued: scan_begin overwrites the sensor's handle, so the earlier scan's result slot
        would never be collected and would stay checked out until the backend recycled it. Under a
        fixed fire cadence that leaks a slot per missed harvest and eventually starves the sensor —
        the scan simply stops updating. Harvest first, then fire.
        """
        if self.sensor.scan_pending:
            return False
        self._place(rs)
        saved = [(o, o.visible) for o in self.hide]                 # self-filter: hide robot + own viz
        for o, _ in saved:
            o.visible = False
        self.sensor.scan_begin(self.renderer, self.scene)
        for o, v in saved:
            o.visible = v
        return True

    def scan_harvest(self, rs):
        """Collect a pending fire if it has landed. -> (ahead[45] float32, h_here float) or None.

        Returns None (and does nothing) when no fire is outstanding or the trace has not been delivered
        yet — the caller keeps using its cached scan for that frame. See scan_fire for the frame pattern.
        """
        if not self.sensor.scan_pending:
            return None
        if not self.sensor.scan_ready(self.renderer):
            return None
        pts = self.sensor.scan_collect(self.renderer)
        return self._process(pts, rs)

    def _update_viz(self, pts, qpx, qpy, h_grid, h_here):
        if self.show_cloud and self.cloud is not None:
            self.cloud.visible = True
            n = min(int(pts.shape[0]), self.cap)
            if n:
                self.cloud_geom.update_attribute("position", np.ascontiguousarray(pts[:n], np.float32))
                # Band the ramp around the ground the robot is standing on, not around z=0: this
                # reads as "blue = a step down, green = level, red = an obstacle up to 35 cm high"
                # at any terrain elevation, and follows the robot as it climbs.
                self.cloud_geom.update_attribute(
                    "color", _height_ramp(pts[:n, 2], lo=h_here - 0.35, hi=h_here + 0.35))
            self.cloud_geom.set_draw_range(0, n)
        elif self.cloud is not None:
            self.cloud.visible = False

        if self.show_grid and self.markers:
            self.marker_group.visible = True
            dirty = []
            for i, mk in enumerate(self.markers):
                hi = h_grid[i]
                seen = not math.isnan(hi)
                hz = float(hi) if seen else float(h_here)
                mk.position.set(float(qpx[i]), float(qpy[i]), hz + 0.03)
                col = (_delta_color(float(np.clip((hz - h_here) / 0.2, -1.0, 1.0))) if seen
                       else 0x3a3f47)                               # dim grey = not yet observed
                if col != self._mk_color[i]:
                    mk.material.color = col
                    self._mk_color[i] = col
                    dirty.append(mk.material)
            # Push the edits to the GPU (Vulkan caches material state by version; no-op on GL).
            if dirty:
                tp.flush_material(self.renderer, *dirty)
        elif self.markers:
            self.marker_group.visible = False
