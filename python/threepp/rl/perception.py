"""The height scan a robot could actually have seen, instead of the one it is handed.

Training a legged policy on a terrain height scan normally means handing it the exact ground
truth at every grid cell -- privileged information no robot has. At deploy the same 45 numbers
come from a body-mounted depth camera fused into an elevation map, and they differ from the
privileged version in ways the policy was never shown: a cell behind a riser is *occluded*, a
cell outside the frustum was never *observed*, and a cell under the robot is known only because
it was ahead of it a moment ago. The gap between those two vectors is the sim-to-deploy gap for
everything the policy knows about the terrain.

`PerceivedScan` closes it inside the training loop. Each control step it casts one ray from the
camera to each query point (`TerrainRays.visible`), marks what it saw into a per-env, world-
anchored *seen* map, and answers the scan from that map -- so a cell reads as terrain once it
has been observed, and as flat ground until then. It returns the same `(ahead, h_here)` pair
threepp's deploy-side `spot_depth_scan.ForwardDepthScanner` returns, and takes the same mount
numbers, so the two describe one camera.

    from threepp.rl.raycast import TerrainRays
    from threepp.rl.perception import PerceivedScan

    scan = PerceivedScan(rays, K, origin_b=env.lane_y, bounds=(-1.0, 31.6, 1.5))
    ahead, h_here = scan.read(sim.root_position, sim.root_quat, px, py)
    scan.forget(reset_idx)            # a teleport invalidates the map under the new pose

What it models, and what it does not. Occlusion, the frustum, the finite range and the memory
are simulated exactly, against the same BVH the physics collides with. The map stores a *seen*
bit rather than a fused height, which is equivalent for static terrain -- an EMA over repeated
noise-free observations of a surface that does not move converges to that surface -- and much
cheaper: one bit per cell instead of a float, so the map fits at K=2048. `noise` re-introduces
the sensor error as a fixed per-cell bias drawn once from the cell's own hash, which is what an
elevation map's error actually looks like after it has converged: frozen, not resampled every
tick. What is approximated is the depth image itself: the map fills from a ground-sampled sweep
of the camera footprint (`sweep_fwd`, `sweep_lat`) rather than one ray per pixel, which is two
orders of magnitude cheaper and, because both go through the same frustum and occlusion tests,
puts the same cells in the map -- just at a coarser stride than a 200x150 sensor would.
"""
import math

import numpy as np
import torch

_kernel = None          # (warp module, mark kernel, read kernel) -- built once per process


def _compile():
    global _kernel
    if _kernel is not None:
        return _kernel
    import warp as wp

    @wp.kernel
    def mark(vis: wp.array2d(dtype=wp.uint8),          # [K,N] seen by the camera this tick
             a: wp.array2d(dtype=float),               # [K,N] query point
             b: wp.array2d(dtype=float),
             origin_b: wp.array(dtype=float),          # [K] per-env offset of the map's b axis
             a0: float, b_half: float, cell: float, nx: int, ny: int, radius: int,
             seen: wp.array3d(dtype=wp.uint8)):        # [K,nx,ny] the map
        k, n = wp.tid()
        if vis[k, n] == wp.uint8(0):
            return
        ia = int(wp.floor((a[k, n] - a0) / cell))
        ib = int(wp.floor((b[k, n] - origin_b[k] + b_half) / cell))
        # A depth sample does not inform one map cell, it informs the patch it covers. `radius` is
        # sized so the sweep's own stride tiles: without it the map is a stipple of isolated cells
        # and nearly every query lands in a gap and reads as never-observed.
        for da in range(-radius, radius + 1):
            for db in range(-radius, radius + 1):
                ja = ia + da
                jb = ib + db
                if ja >= 0 and ja < nx and jb >= 0 and jb < ny:
                    seen[k, ja, jb] = wp.uint8(1)

    @wp.kernel
    def read(a: wp.array2d(dtype=float),               # [K,N] query point
             b: wp.array2d(dtype=float),
             h: wp.array2d(dtype=float),               # [K,N] true ground height there
             origin_b: wp.array(dtype=float),
             a0: float, b_half: float, cell: float, nx: int, ny: int,
             noise: float, seed: int,
             seen: wp.array3d(dtype=wp.uint8),
             known: wp.array2d(dtype=wp.uint8),        # [K,N] 1 = the map has an answer here
             out: wp.array2d(dtype=float)):            # [K,N] that answer
        k, n = wp.tid()
        ia = int(wp.floor((a[k, n] - a0) / cell))
        ib = int(wp.floor((b[k, n] - origin_b[k] + b_half) / cell))
        known[k, n] = wp.uint8(0)
        out[k, n] = 0.0
        # A 3x3 window, mirroring the deploy map's query radius: a cell counts as known when
        # anything in its immediate neighbourhood has been observed.
        for da in range(-1, 2):
            for db in range(-1, 2):
                ja = ia + da
                jb = ib + db
                if ja >= 0 and ja < nx and jb >= 0 and jb < ny:
                    if seen[k, ja, jb] != wp.uint8(0):
                        known[k, n] = wp.uint8(1)
        if known[k, n] == wp.uint8(0):
            return
        v = h[k, n]
        if noise > 0.0:
            # A bias fixed to the CELL, not resampled per tick: a converged map is wrong in the
            # same direction every time you read it, and a policy can learn around a fresh error.
            i = wp.clamp(ia, 0, nx - 1) * ny + wp.clamp(ib, 0, ny - 1)
            state = wp.rand_init(seed, k * nx * ny + i)
            v = v + noise * wp.randn(state)
        out[k, n] = v

    _kernel = (wp, mark, read)
    return _kernel


def _body_forward(quat):
    """The body x-axis in world, from a [K,4] (x,y,z,w) quaternion -> [K,3]."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y + z * w),
                        2.0 * (x * z - y * w)], dim=1)


def _quat_from_matrix(m):
    """3x3 rotation -> (x, y, z, w). Shepperd's branch, so no branch is near-degenerate."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)


def mount_rotation(pitch_deg):
    """The camera-local -> body rotation for a forward camera pitched `pitch_deg` DOWN.

    Identical to `spot_depth_scan.ForwardDepthScanner`'s R_mount: image right is body -y, image
    up is body +z tilted forward by the pitch, and the view axis (-z in camera space) points
    forward and down. Returned as a quaternion (x, y, z, w)."""
    p = math.radians(pitch_deg)
    sp, cp = math.sin(p), math.cos(p)
    return _quat_from_matrix(np.array([[0.0, sp, -cp],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, cp, sp]]))


class PerceivedScan:
    """A forward depth camera + a remembered elevation map, batched over K envs.

    `bounds` is (a0, a1, b_half): the map covers a in [a0, a1] and b within +/- b_half of each
    env's `origin_b` (the lane offset), which is where the robot can go. Queries outside read as
    unknown, i.e. flat, which is what is actually out there.
    """

    def __init__(self, rays, num_envs, *, origin_b, bounds, cell=0.10,
                 mount=(0.30, 0.0, 0.15), pitch_deg=40.0, fov_y=90.0, aspect=200.0 / 150.0,
                 near=0.05, far=10.0, bias=0.02, noise=0.0, seed=0,
                 sweep_fwd=(0.05, 3.0, 14), sweep_lat=(1.0, 9), mark_radius=None):
        wp, self._mark, self._read = _compile()
        self._wp = wp
        self.rays = rays
        self.K = int(num_envs)
        self.device = rays.device
        self._wp_device = rays._wp_device
        a0, a1, b_half = (float(v) for v in bounds)
        self.a0, self.b_half, self.cell = a0, b_half, float(cell)
        self.nx = max(1, int(math.ceil((a1 - a0) / self.cell)) + 1)
        self.ny = max(1, int(math.ceil(2.0 * b_half / self.cell)) + 1)
        self.seen = torch.zeros((self.K, self.nx, self.ny), dtype=torch.uint8, device=self.device)
        self.origin_b = origin_b.contiguous().float()
        self.mount = tuple(float(v) for v in mount)
        self.mount_rot = mount_rotation(pitch_deg)
        self.tan_y = math.tan(math.radians(0.5 * fov_y))
        self.tan_x = self.tan_y * float(aspect)
        self.near, self.far, self.bias = float(near), float(far), float(bias)
        self.noise, self.seed = float(noise), int(seed)
        # h_here survives an unobserved tick the way the deploy scanner's does: the last height the
        # map could actually answer with. Zero at the start, which is the ground it spawns on.
        self.h_here_last = torch.zeros(self.K, device=self.device)
        self._buf = None                                             # allocated on the first read
        self.last_visible = None      # [K,N] uint8 — seen by the camera THIS tick
        self.last_known = None        # [K,N] uint8 — answerable from the map (this tick or memory)

        # The camera does not sample the scan grid, it sweeps a whole depth image, so the map fills
        # from a footprint far wider and further ahead than the 45 cells being asked about. This is
        # that footprint as a body-frame GROUND pattern: `sweep_fwd` = (near, far, n) ahead of the
        # body, `sweep_lat` = (half-width, n) across it. Sampling the ground rather than the image
        # plane costs one ray per point instead of one per pixel, and the frustum and occlusion
        # tests are the real ones either way — a cell behind a riser still never enters the map.
        f0, f1, nf = sweep_fwd
        lat, nl = sweep_lat
        fwd = torch.linspace(float(f0), float(f1), int(nf), device=self.device)
        side = torch.linspace(-float(lat), float(lat), int(nl), device=self.device)
        self.sweep_a = fwd.repeat_interleave(int(nl))                # [M] body-forward offset
        self.sweep_b = side.repeat(int(nf))                          # [M] body-lateral offset
        self.M = self.sweep_a.numel()
        # Default the sample footprint to whatever it takes for the sweep to tile the ground with
        # no holes: a patch (2r+1) cells wide must cover the coarser of the two strides.
        stride = max(float(f1 - f0) / max(int(nf) - 1, 1), 2.0 * float(lat) / max(int(nl) - 1, 1))
        self.mark_radius = (int(mark_radius) if mark_radius is not None
                            else max(1, int(math.ceil((stride / self.cell - 1.0) * 0.5))))

    def forget(self, idx=None):
        """Drop the map for these envs (all of them if idx is None). Call on every teleport: the
        terrain remembered under the old pose says nothing about the new one."""
        if idx is None:
            self.seen.zero_()
            self.h_here_last.zero_()
        elif idx.numel():
            self.seen[idx] = 0
            self.h_here_last[idx] = 0.0

    @property
    def bytes(self):
        return self.seen.numel()

    def read(self, root_pos, root_quat, a, b):
        """Observe from the current pose, then answer the scan -> (ahead [K,N], h_here [K]).

        `a`, `b` are the [K, N] query points in the terrain's horizontal basis, exactly as they
        would be handed to `TerrainRays.heights`. Three sets of points ride in one pair of kernel
        launches: the camera's ground sweep, which is what actually fills the map; the N query
        points; and the body's own position, because `h_here` comes out of the same map -- and,
        like the deploy scanner, falls back to the last answerable value when it cannot.
        """
        wp = self._wp
        K, N = a.shape
        M = self.M
        total = M + N + 1
        if self._buf is None or self._buf.shape != (2, K, total):
            self._buf = torch.zeros((2, K, total), device=self.device)
        qa, qb = self._buf[0], self._buf[1]

        # body heading = the two horizontal components of the body x-axis in world
        f = _body_forward(root_quat)
        ca, cb = f @ self.rays.ax_t, f @ self.rays.bx_t
        nrm = torch.sqrt(ca * ca + cb * cb).clamp_min(1e-6)
        ca, cb = ca / nrm, cb / nrm
        ra, rb = root_pos @ self.rays.ax_t, root_pos @ self.rays.bx_t
        qa[:, :M] = ra[:, None] + ca[:, None] * self.sweep_a - cb[:, None] * self.sweep_b
        qb[:, :M] = rb[:, None] + cb[:, None] * self.sweep_a + ca[:, None] * self.sweep_b
        qa[:, M:M + N], qb[:, M:M + N] = a, b
        qa[:, -1], qb[:, -1] = ra, rb

        h = self.rays.heights(qa, qb)
        vis = self.rays.visible(root_pos, root_quat, qa, qb, h,
                                mount=self.mount, mount_rot=self.mount_rot,
                                tan_x=self.tan_x, tan_y=self.tan_y,
                                near=self.near, far=self.far, bias=self.bias)
        wp.launch(self._mark, dim=(K, total),
                  inputs=[wp.from_torch(vis), wp.from_torch(qa), wp.from_torch(qb),
                          wp.from_torch(self.origin_b), self.a0, self.b_half, self.cell,
                          self.nx, self.ny, self.mark_radius],
                  outputs=[wp.from_torch(self.seen)], device=self._wp_device)

        known = torch.empty((K, total), dtype=torch.uint8, device=self.device)
        out = torch.empty((K, total), device=self.device)
        wp.launch(self._read, dim=(K, total),
                  inputs=[wp.from_torch(qa), wp.from_torch(qb), wp.from_torch(h),
                          wp.from_torch(self.origin_b), self.a0, self.b_half, self.cell,
                          self.nx, self.ny, self.noise, self.seed, wp.from_torch(self.seen)],
                  outputs=[wp.from_torch(known), wp.from_torch(out)], device=self._wp_device)

        here_known = known[:, -1].bool()
        h_here = torch.where(here_known, out[:, -1], self.h_here_last)
        self.h_here_last = h_here
        ahead = torch.where(known[:, M:M + N].bool(), out[:, M:M + N] - h_here[:, None],
                            torch.zeros((), device=self.device)).clamp(-1.0, 1.0)
        self.last_visible = vis[:, M:M + N]
        self.last_known = known[:, M:M + N]
        self.last_here_known = here_known
        return ahead, h_here

    def __repr__(self):
        mb = self.bytes / 1e6
        return (f"PerceivedScan({self.K} envs, map {self.nx}x{self.ny} @ {self.cell} m = {mb:.1f} MB, "
                f"sweep {self.M} pts r={self.mark_radius}, fov {2 * math.degrees(math.atan(self.tan_x)):.0f}x"
                f"{2 * math.degrees(math.atan(self.tan_y)):.0f} deg, far {self.far} m, "
                f"noise {self.noise} m)")
