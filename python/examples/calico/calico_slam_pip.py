"""WP3 of the Calico demo: the live SLAM surface and the depth picture-in-picture.

Two pieces, both driven from `spot_calico.py`'s frame loop:

`SlamSurface`
    spot_slam.py's marching-cubes reconstruction, brought back and re-skinned.
    The forward scanner's accumulated elevation map feeds a `tp.VoxelGrid`; a
    background thread runs `splat_points_to_field` + `marching_cubes` and hands
    back an iso-surface; a generation counter makes a rebuild that was already
    running when the map was cleared drop its result instead of restoring the
    pre-reset surface. What is NEW here is the look: a SOLID, semi-transparent,
    vertex-coloured mesh (height banded +/-0.35 m around the robot's own ground
    height), not spot_slam's blue wireframe.

    WHY that means a MeshBasicMaterial. On Vulkan an unlit, untextured,
    `transparent` mesh is tagged kSnapUiBlend in VulkanCoreScene::snapMeshFlags
    and routed to the raster overlay pass, where the coloured fill pipeline
    multiplies the per-vertex colour by the material colour and blends with
    material.opacity. That is the ONLY path in this renderer that actually
    blends a solid triangle mesh: a `MeshStandardMaterial` with transparent=True
    goes into the deferred G-buffer, which has no blend, and comes out opaque.
    `--slam-mat standard` renders it that way for comparison.

    The overlay route buys one more thing for free. Overlay entries are excluded
    from the ray-traced scene (`if (e0.isOverlay) continue` in the TLAS build),
    and the Vulkan DepthSensor traces that TLAS -- so the robot cannot map its
    own map, without any per-frame hiding. On the `standard` route it CAN, and
    hiding it at scan_fire time does not help (the trace runs against the TLAS
    the previous render() built), so that route is a diagnostic, not the demo.

    Two consequences of the overlay pass are designed around rather than fixed:
    it ignores BufferGeometry.set_draw_range (so the fixed-capacity buffer is
    padded with degenerate triangles instead), and secondary views skip it
    entirely (so the SLAM mesh never shows up in the depth PiP -- which is what
    we want there anyway).

`DepthPip`
    A secondary view (`renderer.add_view`) parented to the same body mount as
    the depth scanner, with `set_view_sensor_surfaces(handle, True)` so it
    rasterizes the baked splat surface. Splats are off in secondary views by
    default and stay off: this is the DEPTH picture, and the bake is what the
    robot's depth camera measures. Each update reads that view's depth
    attachment (`read_gbuffer_aov_raw("depth", view=handle)`, D32 reverse-Z),
    converts to metres, colours it near = bright, and rewrites a screen-space
    Sprite's texture in place (`Texture.update_data`). The sprite is composited
    by the primary's overlay pass, so it lands in the window and in save_frame.

    `renderer.set_view_display_rect` would composite the view with no readback
    at all -- one image copy inside the frame's own command buffer -- but it
    shows the view's SHADED COLOUR, and this PiP is meant to be depth.
"""
import math
import threading

import numpy as np

import threepp as tp
from spot_deploy import _quat_to_R, _quat_from_R
from spot_depth_scan import _height_ramp


# ── the live SLAM surface ─────────────────────────────────────────────────────
class SlamSurface:
    """Scanner elevation map -> VoxelGrid -> marching cubes -> one solid mesh."""

    VOXEL = 0.12
    CELL  = 0.14
    RAD   = 0.22      # tight: the surface hugs the hits instead of ballooning
    ISO   = 0.55
    MAX_VERTS = 300_000        # 100k triangles of preallocated capacity
    BAND  = 0.35               # +/- metres around the LOCAL floor
    FLOOR_CELL = 1.0           # XY cell the local floor is estimated over
    KEEP_ABOVE = 0.5           # drop drawn geometry higher than this over the floor
    KEEP_SLOPE_DEG = 45.0      # ...and only where the floor itself is walkable
    SLOPE_CELL = 0.4           # the slope test's own, FINER grid (see _worker)

    def __init__(self, scene, mat="basic", opacity=0.62, lift=0.0,
                 keep_above=KEEP_ABOVE, keep_slope_deg=KEEP_SLOPE_DEG):
        self.scene = scene
        self.lift = float(lift)
        self.keep_above = float(keep_above)
        # The height filter alone keeps a cell that is only 0.3 m above its own
        # local floor even when that floor is the near-vertical FLANK of the
        # slab: the drawn shards left at the slab's foot in the lookback frame
        # are exactly those cells. They are legitimate measurements (the sensor
        # did see the flank), so they stay in the VoxelGrid and in the map the
        # robot navigates by -- what they are not is walkable ground, which is
        # what this surface is drawn to show. A cell is drawn only where the
        # smoothed local-floor field's own gradient is under tan(45 deg).
        self.keep_slope = float(math.tan(math.radians(float(keep_slope_deg))))
        self.keep_slope_deg = float(keep_slope_deg)
        self._floor = None         # (F, x0, y0, cell) from the last rebuild
        self.grid = tp.VoxelGrid(self.VOXEL, max_points_per_voxel=3, min_spacing=0.12)
        self._pending = [None]
        self._busy = [False]
        self._visible = [True]
        self._sent = None
        self._lock = threading.Lock()
        self._gen = 0
        self._draw_n = 0
        self._band_c = [0.0]       # ground height the colour ramp is banded around

        g = tp.BufferGeometry()
        g.set_attribute("position", np.zeros((self.MAX_VERTS, 3), np.float32))
        g.set_attribute("normal",   np.zeros((self.MAX_VERTS, 3), np.float32))
        g.set_attribute("color",    np.zeros((self.MAX_VERTS, 3), np.float32))
        self._geo = g

        if mat == "standard":
            m = tp.MeshStandardMaterial()
            m.roughness = 0.85
            m.metalness = 0.0
        else:
            m = tp.MeshBasicMaterial()      # unlit + transparent + no map -> overlay route
        m.color = 0xffffff                  # the vertex ramp carries the colour
        m.vertex_colors = True
        m.transparent = True
        m.opacity = float(opacity)
        m.side = tp.Side.Double
        self.material = m
        self.kind = mat

        self._surf = tp.Mesh(g, m)
        self._surf.frustum_culled = False
        self._surf.visible = False
        self._added = False

    # ---- feed ----------------------------------------------------------------
    def insert_scanner(self, scanner):
        """Push the cells the scanner has newly filled since the last call.

        Only a first-time cell can add a voxel (later EMA nudges move a cell by
        centimetres, far inside VOXEL/min_spacing), so hand over just the delta
        instead of re-offering the whole map every scan."""
        valid = ~np.isnan(scanner.H)
        if self._sent is None:
            self._sent = np.zeros_like(valid)
        new = valid & ~self._sent
        if not new.any():
            return
        self._sent |= new
        ix, iy = np.nonzero(new)
        self.grid.insert_array(np.stack([ix * scanner.cell + scanner.x0,
                                         iy * scanner.cell + scanner.y0,
                                         scanner.H[new]], axis=1).astype(np.float32))

    # ---- rebuild -------------------------------------------------------------
    def trigger_rebuild(self, band_center=None):
        with self._lock:
            if self._busy[0] or self.grid.voxel_count < 30:
                return False
            pts = self.grid.collect()
            self._busy[0] = True
            gen = self._gen
        if band_center is not None:
            self._band_c[0] = float(band_center)
        threading.Thread(target=self._worker, args=(pts, gen), daemon=True).start()
        return True

    # ---- the local floor -----------------------------------------------------
    # Both the drawn extent and the colour ramp are relative to the walkable
    # surface UNDER each cell, never to one plane or to one h_here. The wash
    # itself drops ~0.3 m over the walked 8 m and more over the whole spine, so
    # a band around the robot's current ground height paints half the map one
    # flat colour (which is what the first trackside capture showed), and a
    # single height cut would take the wash floor off at one end and leave a
    # metre of canyon wall standing at the other.
    @staticmethod
    def _floor_grid(pts, cell):
        """Per-cell lowest z over a coarse XY grid -> (F, x0, y0, cell)."""
        x0 = float(pts[:, 0].min()) - cell
        y0 = float(pts[:, 1].min()) - cell
        nx = int((pts[:, 0].max() - x0) / cell) + 2
        ny = int((pts[:, 1].max() - y0) / cell) + 2
        ix = np.clip(((pts[:, 0] - x0) / cell).astype(np.int64), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - y0) / cell).astype(np.int64), 0, ny - 1)
        F = np.full(nx * ny, np.inf, np.float32)
        np.minimum.at(F, ix * ny + iy, pts[:, 2])
        F = F.reshape(nx, ny)
        raw = F.copy()
        # A cell's own minimum is its floor (brush and boulders sit above it and
        # cannot pull it up). Smoothing those minima over a 3x3 m neighbourhood
        # then gives a surface the colour ramp can be signed against: a cell
        # below its neighbourhood reads blue, level reads green, above reads
        # orange -- on a wash that drops a metre end to end just as much as on a
        # flat one. Unscanned cells inherit the neighbourhood so the ramp does
        # not fall off a cliff at the edge of the mapped region.
        F[~np.isfinite(F)] = np.nan
        pad = np.full((nx + 2, ny + 2), np.nan, np.float32)
        pad[1:-1, 1:-1] = F
        stack = np.stack([pad[a:a + nx, b:b + ny]
                          for a in range(3) for b in range(3)])
        with np.errstate(invalid="ignore"):
            F = np.nanmean(stack, axis=0).astype(np.float32)
        F[~np.isfinite(F)] = float(np.median(pts[:, 2]))
        # RAW per-cell minima as well as the smoothed field: the 3x3 mean is what
        # the colour ramp and the height band want (a floor, not a staircase),
        # but it is also what FLATTENS the slab's flank -- differentiating it
        # gives a maximum gradient of ~0.3, so a 45-degree test on it never
        # fires. The slope test differentiates the raw minima instead.
        raw[~np.isfinite(raw)] = F[~np.isfinite(raw)]
        return F, x0, y0, cell, raw

    @staticmethod
    def _floor_at(fl, x, y):
        F, x0, y0, cell = fl[:4]
        ix = np.clip(((x - x0) / cell).astype(np.int64), 0, F.shape[0] - 1)
        iy = np.clip(((y - y0) / cell).astype(np.int64), 0, F.shape[1] - 1)
        return F[ix, iy]

    @staticmethod
    def _slope_at(fl, x, y):
        """|grad| of the smoothed local-floor field, sampled per point.

        The same field the height band is measured against, differentiated over
        its own 1 m cells: 1.0 is a 45-degree floor. np.gradient uses central
        differences inside and one-sided differences at the border, which is
        what we want -- the border of the mapped region is where the wall is.
        """
        F, x0, y0, cell = fl[:4]
        R = fl[4] if len(fl) > 4 else F
        gx, gy = np.gradient(R.astype(np.float32), cell)
        G = np.hypot(gx, gy)
        ix = np.clip(((x - x0) / cell).astype(np.int64), 0, F.shape[0] - 1)
        iy = np.clip(((y - y0) / cell).astype(np.int64), 0, F.shape[1] - 1)
        return G[ix, iy]

    def _worker(self, pts, gen):
        try:
            # The VoxelGrid keeps everything the sensor ever saw -- brush, the
            # canyon wall, the boulder tops. The DRAWN surface keeps only what
            # is within keep_above of the local floor: the rest is not walkable
            # ground, and on the overlay route it composites over the cliff it
            # belongs to instead of behind it.
            fl = self._floor_grid(pts, self.FLOOR_CELL)
            if self.keep_above > 0.0:
                # Two tests, both against the same smoothed local floor: close
                # enough to it (not brush, not the boulder top) AND standing on
                # a piece of it that is walkable (not the slab's flank).
                keep = pts[:, 2] <= self._floor_at(fl, pts[:, 0], pts[:, 1]) + self.keep_above
                n_h = int(keep.sum())
                if self.keep_slope > 0.0:
                    # On the 1 m floor grid the slab's flank is INVISIBLE to a
                    # gradient test: a cell's value is the MINIMUM in it, so a
                    # vertical face inside one cell contributes only its foot and
                    # the neighbouring minima differ by centimetres (measured:
                    # the 45-degree test on the 1 m field dropped 0 of 1836
                    # cells). The flank's rise happens WITHIN a metre, so the
                    # slope test gets its own 0.4 m grid, where a 1 m face steps
                    # a full cell height and reads as the >45-degree ground it is.
                    sl = self._floor_grid(pts, self.SLOPE_CELL)
                    g = self._slope_at(sl, pts[:, 0], pts[:, 1])
                    keep &= g < self.keep_slope
                    print(f"[slam] slope on the {self.SLOPE_CELL} m grid: "
                          f"p50 {np.percentile(g, 50):.2f} p90 {np.percentile(g, 90):.2f} "
                          f"max {g.max():.2f} (limit {self.keep_slope:.2f})")
                if keep.sum() >= 30:
                    print(f"[slam] drawn cells: {len(pts)} -> {n_h} within "
                          f"{self.keep_above} m of the local floor -> {int(keep.sum())} "
                          f"also under {self.keep_slope_deg:.0f} deg of slope")
                    pts = np.ascontiguousarray(pts[keep])
            with self._lock:
                if gen == self._gen:
                    self._floor = fl
            field = tp.splat_points_to_field(pts, self.CELL, self.RAD, max_nodes=6_000_000)
            iso = tp.marching_cubes(field, self.ISO)
            with self._lock:
                if gen != self._gen:
                    return                 # cleared while we were building
                self._pending[0] = iso if not iso.empty else None
        finally:
            with self._lock:
                self._busy[0] = False

    def rebuild_sync(self, band_center=None):
        """Rebuild on THIS thread and publish it. Returns the triangle count.

        The film mode's whole determinism argument rests on this: with the
        background worker, WHICH frame a new surface appears on depends on how
        long marching cubes took, i.e. on the machine's mood, and two runs of the
        same simulation cut the map in at different frames. Run synchronously
        every N frames and the map's content is a function of the frame index
        alone. It costs ~60 ms on the frame it happens (measured on the walked
        map), which the film does not care about.
        """
        with self._lock:
            if self._busy[0] or self.grid.voxel_count < 30:
                return 0
            pts = self.grid.collect()
            self._busy[0] = True
            gen = self._gen
        if band_center is not None:
            self._band_c[0] = float(band_center)
        self._worker(pts, gen)
        return self.apply_pending()

    def set_opacity(self, renderer, a):
        """Ramp the surface in or out. Vulkan caches material state, so the
        change has to be pushed (tp.flush_material) or the first opacity the
        material was drawn with is the only one it ever has."""
        a = float(max(0.0, min(1.0, a)))
        if abs(a - float(self.material.opacity)) < 1e-4:
            return
        self.material.opacity = a
        tp.flush_material(renderer, self.material)

    def apply_pending(self):
        """Publish a finished rebuild. Returns the triangle count, or 0."""
        with self._lock:
            iso = self._pending[0]
            self._pending[0] = None
        if iso is None:
            return 0
        pos, nrm = iso.positions, iso.normals
        n = min(pos.shape[0], self.MAX_VERTS)
        n -= n % 3
        if n <= 0:
            self._draw_n = 0
            self._apply_visibility()
            return 0
        if pos.shape[0] > self.MAX_VERTS:
            print(f"[slam] surface clipped to {self.MAX_VERTS} of {pos.shape[0]} verts")
        p = np.ascontiguousarray(pos[:n], np.float32)
        v = np.ascontiguousarray(nrm[:n], np.float32)
        if self.lift:
            p = p.copy()
            p[:, 2] += self.lift
        # Blue BELOW the local walkable surface, green level with it, orange
        # above -- the same reading at either end of a wash that drops a metre.
        with self._lock:
            fl = self._floor
        if fl is not None:
            dz = p[:, 2] - self._floor_at(fl, p[:, 0], p[:, 1]) - self.lift
            # The floor field is the MEAN of neighbourhood minima and the iso
            # surface sits a little above the samples that made it, so raw dz is
            # biased positive by ~0.2 m and the whole wash reads orange. The
            # median of the drawn surface IS the walkable floor by construction
            # -- centre on it, and green means "level with the trail here".
            dz = dz - float(np.median(dz))
        else:
            dz = p[:, 2] - self._band_c[0]
        c = _height_ramp(dz, lo=-self.BAND, hi=self.BAND)

        # The overlay pass ignores set_draw_range, so the tail of the fixed
        # buffer is not "not drawn" -- it is drawn, as triangles at the origin.
        # Collapse it onto the last real vertex: every padding triangle then has
        # three identical corners, zero area, and rasterizes to nothing.
        P = np.repeat(p[-1:], self.MAX_VERTS, axis=0)
        N = np.zeros((self.MAX_VERTS, 3), np.float32)
        C = np.zeros((self.MAX_VERTS, 3), np.float32)
        P[:n] = p
        N[:n] = v
        C[:n] = c
        self._geo.update_attribute("position", P)
        self._geo.update_attribute("normal", N)
        self._geo.update_attribute("color", C)
        self._geo.set_draw_range(0, n)      # honoured on the deferred route
        self._draw_n = n
        self._apply_visibility()
        if not self._added:
            self.scene.add(self._surf)      # once per session, not per rebuild
            self._added = True
        return n // 3

    def clear(self):
        with self._lock:
            self._gen += 1                  # a running rebuild drops its result
            self._pending[0] = None
        self.grid.clear()
        self._sent = None
        self._geo.set_draw_range(0, 0)
        self._draw_n = 0
        self._apply_visibility()

    # ---- state ---------------------------------------------------------------
    @property
    def busy(self):
        return self._busy[0]

    @property
    def voxels(self):
        return self.grid.voxel_count

    @property
    def triangles(self):
        return self._draw_n // 3

    @property
    def mesh(self):
        return self._surf

    def _apply_visibility(self):
        self._surf.visible = bool(self._visible[0] and self._draw_n > 0)

    @property
    def visible(self):
        return self._visible[0]

    @visible.setter
    def visible(self, v):
        self._visible[0] = bool(v)
        self._apply_visibility()


# ── the depth picture-in-picture ──────────────────────────────────────────────
# Depth ramp stops, near -> far. NEAR IS BRIGHT: white at the toes, through
# yellow and orange to a deep blue at range, black where nothing was hit. The
# stops are spaced over the range the mount actually resolves (RAMP_NEAR ..
# RAMP_FAR); ramping over the full 0.06..8 m instead puts every ground pixel in
# the top eighth of the scale and the picture comes out a flat pale wash, which
# is exactly what the first capture showed.
_RAMP = np.array([[1.00, 1.00, 0.96],
                  [1.00, 0.90, 0.45],
                  [0.98, 0.62, 0.18],
                  [0.72, 0.28, 0.35],
                  [0.24, 0.30, 0.72],
                  [0.06, 0.10, 0.34]], np.float32)


def _depth_rgb(d, near, far, ramp_near=0.3, ramp_far=8.0):
    """Metric depth (H, W) -> an RGB ramp. NEAR = BRIGHT, no-hit = near-black.

    LOG spacing. The mount is 0.85 m off the ground, so a linear ramp spends
    most of its scale on the metre of floor directly under the camera and the
    trail beyond 2 m arrives as one flat colour -- the white blob the first
    capture showed. log(d/near)/log(far/near) gives each octave of range the
    same number of stops, which is what a depth picture is read for."""
    t = np.clip(np.log(np.maximum(d, ramp_near) / ramp_near)
                / math.log(ramp_far / ramp_near), 0.0, 1.0)
    x = t * (len(_RAMP) - 1)
    i = np.clip(x.astype(np.int32), 0, len(_RAMP) - 2)
    f = (x - i)[..., None]
    rgb = _RAMP[i] * (1.0 - f) + _RAMP[i + 1] * f
    rgb[d >= far * 0.999] = 0.02                 # nothing within range
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


class DepthPip:
    """A secondary view on the robot's forward depth camera, drawn as a corner PiP.

    The view asks for sensor surfaces, so it sees the BAKED splat surface -- the
    same triangles PhysX and the lidar use -- where the primary view draws the
    real splats. Splats stay off in the view (the renderer's default), which is
    the point: this is a depth camera, not a second RGB camera.
    """

    def __init__(self, renderer, scene, scanner, width=256, height=192,
                 fov_y=90.0, near=0.06, far=8.0, scale=2, margin=16,
                 anchor=(1.0, 0.0), pitch_deg=18.0):
        self.rend = renderer
        self.scanner = scanner
        self.w, self.h = int(width), int(height)
        self.near, self.far = float(near), float(far)
        self.scale = int(scale)
        self.margin = int(margin)
        self.anchor = anchor
        self.handle = 0
        # The sensor beat of the film grows the panel to a quarter of the frame.
        # It is a SPRITE scale, not the readback size: the depth image stays
        # 224x168 (the measurement must not change because the picture got
        # bigger), and the sprite is drawn `scale * zoom` window pixels wide.
        self.zoom = 1.0
        self.frames = 0
        self.sig = None            # mean/first-moment signature of the last read
        self.updates = 0           # readbacks whose content actually CHANGED
        # Same body mount as the scanner, its OWN pitch. The policy's camera
        # looks 40 deg down because it is measuring the next footfall, and at
        # that angle 90 % of the frame is the floor within a metre -- a correct
        # depth image of a patch of gravel. Pitching the PICTURE up shows the
        # trail receding, which is what the PiP is in the frame to say.
        p = math.radians(float(pitch_deg))
        sp, cp = math.sin(p), math.cos(p)
        self.R_mount = np.array([[0.0, sp, -cp],
                                 [-1.0, 0.0, 0.0],
                                 [0.0, cp, sp]], float)
        self.pitch_deg = float(pitch_deg)

        self.camera = tp.PerspectiveCamera(float(fov_y), self.w / float(self.h),
                                           self.near, self.far)
        self.camera.up.set(0, 0, 1)

        self._tex = None
        self._mat = tp.SpriteMaterial()
        self._sprite = tp.Sprite(self._mat)
        self._sprite.screen_space = True
        self._sprite.screen_anchor.set(float(anchor[0]), float(anchor[1]))
        self._sprite.center.set(0.0, 0.0)
        self._sprite.visible = False
        scene.add(self._sprite)

    def attach(self):
        """Create the view. add_view shares the primary's render pass and
        pipelines, so it returns 0 until a first render() has happened."""
        self.handle = self.rend.add_view(self.camera, self.w, self.h)
        if not self.handle:
            print("[pip] add_view returned 0 -- render() once before attaching")
            return False
        # Rasterize splats::make_sensor_mesh in THIS view only. The primary
        # never draws sensor surfaces (the renderer refuses handle 0), so the
        # picture keeps the real splats.
        ok = self.rend.set_view_sensor_surfaces(self.handle, True)
        # Nothing should be drawn into the frame by the renderer itself: the
        # composited picture is the depth ramp below, not the view's shading.
        self.rend.hide_view(self.handle)
        print(f"[pip] view {self.handle} {self.w}x{self.h} fov {self.camera.fov:.0f} "
              f"near {self.near} far {self.far}  sensor_surfaces={ok}")
        return True

    def place(self, rs):
        """Put the view camera on the scanner's own body mount."""
        s = self.scanner
        R = _quat_to_R(rs[3:7])
        wp = np.array([float(rs[0]), float(rs[1]), float(rs[2])]) + R @ s.mount_local
        self.camera.position.set(float(wp[0]), float(wp[1]), float(wp[2]))
        q = _quat_from_R(R @ self.R_mount)
        self.camera.quaternion.set(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def update(self, size):
        """Read the view's depth attachment from the frame just rendered and
        rewrite the PiP texture. Call AFTER render()."""
        if not self.handle:
            return False
        raw = self.rend.read_gbuffer_aov_raw("depth", self.handle)
        if raw is None or raw.size == 0:
            return False
        z = np.ascontiguousarray(raw).view(np.float32).reshape(raw.shape[0], raw.shape[1])
        n, f = self.near, self.far
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(z <= 0.0, f, (n * f) / (n + z * (f - n)))
        d = np.clip(np.nan_to_num(d, nan=f, posinf=f), n, f)
        self.last_depth = d
        # Is this actually a NEW frame, or the same image read twice? Two
        # captures of two different runs can legitimately produce the same PiP
        # (same policy, same walk, same final pose), so staleness has to be
        # judged INSIDE a run, between consecutive readbacks.
        sig = (float(d.mean()), float((d * np.arange(d.shape[1])).mean()))
        if self.sig is None or abs(sig[0] - self.sig[0]) > 1e-6 or \
                abs(sig[1] - self.sig[1]) > 1e-6:
            self.updates += 1
        self.sig = sig
        rgb = _depth_rgb(d, n, f)
        if self.scale > 1:
            rgb = np.repeat(np.repeat(rgb, self.scale, 0), self.scale, 1)
        # A 3 px frame, so the panel reads as an instrument rather than as a
        # patch of the picture that happens to be a different colour.
        b = 3
        rgb[:b, :] = rgb[-b:, :] = 220
        rgb[:, :b] = rgb[:, -b:] = 220
        rgb = np.ascontiguousarray(rgb[::-1])       # sprite samples v=0 at the bottom
        if self._tex is None:
            self._tex = tp.data_texture(rgb, srgb=True)
            self._tex.mag_filter = tp.Filter.Nearest
            self._tex.min_filter = tp.Filter.Linear
            self._tex.generate_mipmaps = False
            self._tex.wrap_s = tp.TextureWrapping.ClampToEdge
            self._tex.wrap_t = tp.TextureWrapping.ClampToEdge
            self._mat.map = self._tex
            self._mat.needs_update()
        else:
            self._tex.update_data(rgb)
        pw = int(round(rgb.shape[1] * self.zoom))
        ph = int(round(rgb.shape[0] * self.zoom))
        self._sprite.scale.set(float(pw), float(ph), 1.0)
        x = -(pw + self.margin) if self.anchor[0] > 0.5 else self.margin
        y = -(ph + self.margin) if self.anchor[1] > 0.5 else self.margin
        self._sprite.position.set(float(x), float(y), 0.0)
        self._sprite.visible = True
        self.frames += 1
        return True

    @property
    def sprite(self):
        return self._sprite

    @property
    def visible(self):
        return self._sprite.visible

    @visible.setter
    def visible(self, v):
        self._sprite.visible = bool(v)

    def rect(self, size):
        """(x, y, w, h) of the PiP in window pixels, TOP-LEFT origin -- what a
        1:1 crop of the capture has to cut out."""
        w, h = size
        pw = int(round(self.w * self.scale * self.zoom))
        ph = int(round(self.h * self.scale * self.zoom))
        x = w - pw - self.margin if self.anchor[0] > 0.5 else self.margin
        y = self.margin if self.anchor[1] > 0.5 else h - ph - self.margin
        return int(x), int(y), int(pw), int(ph)
