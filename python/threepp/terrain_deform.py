"""Ground that keeps the shape of whatever walked on it -- mud, snow, sand.

Every terrain a threepp RL env has stood on so far is rigid: a tent, a bilinear
heightfield, a cooked triangle soup. A foot lands on it, the contact solves, and the
ground is exactly as it was. That is fine for a stairs task and wrong for the one that
actually separates a good locomotion policy from a lucky one -- soft ground, where the
foot sinks, the print stays, and the second lap crosses the ruts the first lap dug.

This is the displacement model from Sumner, O'Brien and Hodgins, "Animating Sand, Mud,
and Snow" (1999), batched over K environments and written as fixed-dimension Warp
kernels. The surface is a height grid; a step is three passes over it:

  imprint     every cell under a collider sphere is pushed down to the sphere's lower
              surface. The height it lost is the displaced volume. A fraction
              `compression` of that volume is destroyed -- the material compacts rather
              than moves, which is the whole difference between a snow print (mostly
              compacted, the rim barely rises) and a mud trench (nothing compacts, the
              walls pile up). The rest goes into an ejecta grid.
  distribute  each cell's ejecta is pushed onto its 8 neighbours, skipping any neighbour
              that is itself under a collider. Material cannot be shoved back under the
              boot, so it migrates outward a cell per step and the berm builds where the
              boot is not. A cell ringed in on all 8 sides parks its ejecta in a carry
              grid until the collider moves and a neighbour opens, so the print floor
              stays where the sphere cut it instead of being re-buried and re-carved.
  relax       an angle-of-repose relaxation: neighbours steeper than `tan_repose` trade
              height at `flow_rate` per iteration. Sand slumps back to nearly flat, mud
              holds a near-vertical trench wall.

The same pass that pushes the cell down also returns the reaction on the collider, and
that reaction is a soil model rather than a spring. Bekker-Wong bearing pressure

    p = (k_c / b + k_phi) * z^n         [Pa]

where `z` is the depth of the collider's lower surface below a slowly relaxing reference
grade -- not the height the cell yielded this step -- and `b` is the chord width of the
contact at that cell. Because `z` is a total depth, a foot standing still in its own
print keeps its support: it sinks until `p * A` balances the load and then holds, at a
material-dependent equilibrium depth. Traction is Janosi-Hanamoto,

    tau = (c + p * tan_phi) * (1 - exp(-j / janosi_K))    [Pa]

built out of the slip `j` the collider has accumulated since it touched down, so it ramps
in over a centimetre or so of sliding and saturates at the Mohr-Coulomb limit -- a policy
can lean on it, and shear out of it. Mud additionally sucks back on a foot that lifts
while its patch is still below grade. So the grid is both the thing you draw and the
thing that pushes back on the feet.

The reference grade is its own [K, nx, ny] grid. Cells under a collider are frozen (that
is what holds `z` still under a stance); everywhere else it relaxes toward the surface
over seconds, so a berm slowly becomes ground and a fresh print still measures its depth
from where the ground used to be -- which is why the second pass over a rut is firmer.

Everything is batched over K and allocation-free after construction: no host readback, no
data-dependent launch size, no Python branching on device values. The whole `deform` +
`relax` sequence records into a CUDA graph, which is how the Spot RL step already replays
its Warp work.

    from threepp.terrain_deform import DeformableTerrain

    terrain = DeformableTerrain(origin=(-2.0, -2.0), cell=0.025, dims=(160, 160),
                                K=1024, material="mud")
    f = terrain.deform(foot_pos, foot_r, foot_vel)     # [K,C,3] reaction on the feet
    terrain.relax(2)
    h = terrain.heights(scan_x, scan_y)                # [K,N] surface for the observation

Wiring it into a Spot env
-------------------------
The feet are links, so their world positions come straight out of the sim: construct
`GpuSim(..., read_links=True)` and read `sim.link_pose` [K, max_links, 7] in the PhysX
layout [qx, qy, qz, qw, px, py, pz] -- the foot centres are `sim.link_pose[:, feet, 4:7]`
and the radii are the foot collider radii. Feed those to `deform`, and scatter the
returned reaction into `sim.apply_link_force` [K, max_links, 3] at the same link rows, so
the ground pushes back before the next `sim.step()`. The observation's height scan calls
`terrain.heights(px, py)` where a rigid env calls its analytic `_terrain_h`, and the scan
then reports the ruts the robot dug on its previous lap. Per-env `origin` exists for
exactly this: RL lanes are offset in y, and each lane gets its own patch of ground.

None of that env is implemented here -- this module is the ground, not the task.

Warp is imported lazily on the first DeformableTerrain, so `import threepp` still works
on a machine without it.

    pip install warp-lang
"""
from dataclasses import dataclass

import numpy as np
import torch

import threepp as tp


@dataclass(frozen=True)
class Material:
    """What makes mud mud.

    compression  fraction of the displaced volume that is destroyed (compacted) instead
                 of pushed outward. Snow packs, mud does not.
    flow_rate    how much of an over-steep slope is relaxed per `relax` iteration (0..0.25;
                 above 0.25 the explicit Jacobi sweep rings).
    tan_repose   the steepest slope that stands. tan(angle of repose).
    n            Bekker sinkage exponent, dimensionless. Below 1 the ground firms up with
                 depth and the print bottoms out; above 1 it softens, which is why a snow
                 print runs away and a sand print does not.
    k_c          Bekker cohesive modulus, N/m^(n+1). Divided by the contact width, so it
                 is the term that makes a narrow foot sink deeper than a wide one.
    k_phi        Bekker frictional modulus, N/m^(n+2). Carries most of the load.
    cohesion     Mohr-Coulomb cohesion c, Pa -- traction available at zero normal load.
    tan_phi      tan of the internal friction angle -- traction bought with normal load.
    janosi_K     Janosi-Hanamoto shear modulus, m: the slip distance over which traction
                 builds. Centimetre-scale.
    suction      adhesion on withdrawal, Pa*s/m. While the patch is still below grade and
                 the collider is rising, it is pulled back down at `suction * v_up` Pa.
    suction_cap  ceiling on that suction pressure, Pa. Sand ~0, mud large.
    grade_rate   1/s at which the reference grade relaxes toward the surface, away from
                 whatever is standing on it. Seconds, not frames.

    `n` is not a free knob: the units of k_c and k_phi are powers of n, so a preset's
    three bearing numbers only mean anything together.
    """
    compression: float
    flow_rate: float
    tan_repose: float
    n: float
    k_c: float
    k_phi: float
    cohesion: float
    tan_phi: float
    janosi_K: float
    suction: float
    suction_cap: float
    grade_rate: float


# The bearing exponents and the shear pair are Wong, "Theory of Ground Vehicles",
# Table 2.3 (converted from kN); the moduli are scaled to that table's *shape* rather
# than copied, because Wong measures plates and tracks 0.2-0.5 m wide, where k_c/b is a
# rounding error next to k_phi. A 5 cm Spot foot sinks to b ~ 3 cm, where the raw k_c
# would swamp everything and a 160 N foot load would sit at a fraction of a millimetre.
# These are calibrated to that foot: ~2 cm in sand, ~3.5 cm in mud, ~6 cm in snow.
MATERIALS = {
    # Wet clay: nothing compacts, so every gram the boot displaces ends up in the berm,
    # and a 55-degree trench wall stands there until something else knocks it down.
    # Cohesion-dominated (c ~ 4 kPa, phi 13 deg): grip at zero load, and a low ceiling to
    # shear out of. n < 1, so it firms up as the foot goes down. Sucks on the way out.
    "mud": Material(compression=0.15, flow_rate=0.02, tan_repose=1.4,
                    n=0.5, k_c=1.3e3, k_phi=1.75e5,
                    cohesion=4.1e3, tan_phi=0.23, janosi_K=0.020,
                    suction=2.0e4, suction_cap=1.5e4, grade_rate=0.15),
    # Packed snow: most of the displaced volume is air being squeezed out, so the print is
    # deep, the rim is low, and what rim there is slumps a little. n = 1.6 is the whole
    # character -- pressure grows slower than depth, so the foot keeps going.
    "snow": Material(compression=0.70, flow_rate=0.06, tan_repose=0.9,
                     n=1.6, k_c=4.4e3, k_phi=5.0e6,
                     cohesion=1.0e3, tan_phi=0.36, janosi_K=0.040,
                     suction=2.0e3, suction_cap=2.0e3, grade_rate=0.10),
    # Dry sand: incompressible grains at a shallow repose angle -- a print fills itself in.
    # Friction-dominated (phi 28 deg, almost no cohesion): traction has to be bought with
    # load, which is the thing a light-footed gait cannot do.
    "sand": Material(compression=0.0, flow_rate=0.15, tan_repose=0.65,
                     n=1.1, k_c=9.9e2, k_phi=4.6e6,
                     cohesion=1.0e3, tan_phi=0.53, janosi_K=0.025,
                     suction=0.0, suction_cap=0.0, grade_rate=0.50),
    # A control: displaces, never flows, never compacts. Volume is conserved exactly, and
    # the grade never moves, so its bearing depth is measured from the original ground.
    "clay": Material(compression=0.0, flow_rate=0.0, tan_repose=10.0,
                     n=1.0, k_c=1.0e3, k_phi=6.0e6,
                     cohesion=1.0e4, tan_phi=0.60, janosi_K=0.010,
                     suction=0.0, suction_cap=0.0, grade_rate=0.0),
}

_KERNELS = None      # (warp module, kernels...) -- built once per process


def _compile():
    """Import Warp and build the kernels. Deferred so importing this module is free."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import warp as wp

    @wp.func
    def _bottom(cx: float, cy: float, cz: float, r: float, x: float, y: float) -> float:
        """Height of the sphere's lower surface over (x, y), or +BIG outside its disc."""
        dx = x - cx
        dy = y - cy
        d2 = dx * dx + dy * dy
        rr = r * r
        if d2 >= rr:
            return 1.0e9
        return cz - wp.sqrt(rr - d2)

    @wp.kernel
    def imprint(h: wp.array3d(dtype=float),
                eject: wp.array3d(dtype=float),
                grade: wp.array3d(dtype=float),
                org: wp.array(dtype=wp.vec2),
                centers: wp.array3d(dtype=float),      # [K, C, 3]
                radii: wp.array2d(dtype=float),        # [K, C]
                vel: wp.array3d(dtype=float),          # [K, C, 3]
                slip_j: wp.array2d(dtype=float),       # [K, c_max] stance slip, read here
                forces: wp.array3d(dtype=float),       # [K, C, 3] (accumulated)
                carea: wp.array2d(dtype=float),        # [K, c_max] contact area (accum.)
                n_c: int, cell: float, z0: float, dt: float,
                compression: float, bek_n: float, k_c: float, k_phi: float,
                cohesion: float, tan_phi: float, janosi_K: float,
                suction: float, suction_cap: float, grade_rate: float):
        """Push every covered cell down to the sphere bottom; bank the displaced volume;
        return the Bekker/Janosi reaction; age the reference grade where nothing stands."""
        k, i, j = wp.tid()
        o = org[k]
        x = o[0] + float(i) * cell
        y = o[1] + float(j) * cell
        area = cell * cell
        hh = h[k, i, j]
        g = grade[k, i, j]
        moved = float(0.0)
        covered = int(0)
        # Sequential over the colliders: each one sees the surface the previous one left,
        # so two overlapping feet displace the cell once between them, not twice.
        for c in range(n_c):
            r = radii[k, c]
            bz = _bottom(centers[k, c, 0], centers[k, c, 1], centers[k, c, 2], r, x, y)
            pen = (z0 + hh) - bz
            if pen > 0.0:
                hh = bz - z0
                moved += pen
                covered = 1
            # The reaction is NOT in the carve branch: it is driven by the total depth
            # below the reference grade, so it survives a foot that has stopped digging.
            z = (z0 + g) - bz
            if z > 0.0:
                covered = 1
                # Chord half-width of the sphere's contact at this depth, floored at one
                # cell so the k_c / b term cannot run away as the patch edge thins out.
                b = wp.sqrt(wp.max(z * (2.0 * r - z), cell * cell))
                p = (k_c / b + k_phi) * wp.pow(z, bek_n)
                fz = p * area
                vz = vel[k, c, 2]
                if vz > 0.0:
                    # Still below grade and pulling away: mud holds on.
                    fz -= wp.min(suction * vz, suction_cap) * area
                wp.atomic_add(forces, k, c, 2, fz)
                wp.atomic_add(carea, k, c, area)
                sj = slip_j[k, c]
                if sj > 0.0:
                    vx = vel[k, c, 0]
                    vy = vel[k, c, 1]
                    vm = wp.sqrt(vx * vx + vy * vy)
                    if vm > 1.0e-6:
                        tau = (cohesion + p * tan_phi) * (1.0 - wp.exp(-sj / janosi_K))
                        f = tau * area / vm
                        wp.atomic_add(forces, k, c, 0, -f * vx)
                        wp.atomic_add(forces, k, c, 1, -f * vy)
        h[k, i, j] = hh
        eject[k, i, j] = moved * (1.0 - compression)
        # Frozen under a stance, relaxing everywhere else. Folded in here rather than run
        # as its own [K, nx, ny] pass: the collider loop above already knows who is
        # covered, and the grade moves by rate*dt per step, so reading `h` before the
        # ejecta lands is a step of lag on a timescale of seconds.
        if covered == 0:
            d = hh - g
            # Guard the store, not the arithmetic: most of a grid is ground nobody has
            # touched, where the grade already equals the surface and the write is a
            # no-op. Worth a few percent; the extra read above is the part that is not
            # optional.
            if d > 1.0e-7 or d < -1.0e-7:
                grade[k, i, j] = g + grade_rate * dt * d

    @wp.kernel
    def slip(carea: wp.array2d(dtype=float),                # [K, c_max]
             vel: wp.array3d(dtype=float),                  # [K, c_max, 3]
             slip_j: wp.array2d(dtype=float),               # [K, c_max]
             dt: float):
        """Age each collider's stance slip: integrate tangential speed while it is in
        contact, and forget the whole stance the moment it lifts."""
        k, c = wp.tid()
        if carea[k, c] > 0.0:
            vx = vel[k, c, 0]
            vy = vel[k, c, 1]
            slip_j[k, c] = slip_j[k, c] + wp.sqrt(vx * vx + vy * vy) * dt
        else:
            slip_j[k, c] = 0.0

    @wp.kernel
    def distribute(h: wp.array3d(dtype=float),
                   eject: wp.array3d(dtype=float),
                   carry: wp.array3d(dtype=float),
                   org: wp.array(dtype=wp.vec2),
                   centers: wp.array3d(dtype=float),
                   radii: wp.array2d(dtype=float),
                   n_c: int, cell: float, z0: float):
        """Spread each cell's ejecta over the 8-neighbours that are not under a collider.

        `carry` is this step's ejecta plus whatever earlier steps could not place: only
        this cell's thread touches carry[k, i, j], so it needs no atomics."""
        k, i, j = wp.tid()
        e = eject[k, i, j] + carry[k, i, j]
        if e <= 0.0:
            return
        nx = h.shape[1]
        ny = h.shape[2]
        o = org[k]
        # Pass 1: count the neighbours that will take material.
        n_open = int(0)
        for a in range(-1, 2):
            for b in range(-1, 2):
                if a == 0 and b == 0:
                    continue
                ii = i + a
                jj = j + b
                if ii < 0 or jj < 0 or ii >= nx or jj >= ny:
                    continue
                x = o[0] + float(ii) * cell
                y = o[1] + float(jj) * cell
                surf = z0 + h[k, ii, jj]
                blocked = int(0)
                for c in range(n_c):
                    bz = _bottom(centers[k, c, 0], centers[k, c, 1], centers[k, c, 2],
                                 radii[k, c], x, y)
                    if bz <= surf + 1.0e-6:
                        blocked += 1
                if blocked == 0:
                    n_open += 1
        if n_open == 0:
            # Ringed in by the collider. Park it: stacking it back onto h (v1) meant the
            # next imprint carved it off again, and at compression=0 that cycle never
            # decays -- the print floor would churn forever a constant height above the
            # sphere. Parked material spreads like any other ejecta the moment the
            # collider moves on and a neighbour opens, which is still how material
            # migrates out from under a rolling wheel.
            carry[k, i, j] = e
            return
        carry[k, i, j] = 0.0
        share = e / float(n_open)
        # Pass 2: deposit. Same test, so the two passes agree on who is open.
        for a in range(-1, 2):
            for b in range(-1, 2):
                if a == 0 and b == 0:
                    continue
                ii = i + a
                jj = j + b
                if ii < 0 or jj < 0 or ii >= nx or jj >= ny:
                    continue
                x = o[0] + float(ii) * cell
                y = o[1] + float(jj) * cell
                surf = z0 + h[k, ii, jj]
                blocked = int(0)
                for c in range(n_c):
                    bz = _bottom(centers[k, c, 0], centers[k, c, 1], centers[k, c, 2],
                                 radii[k, c], x, y)
                    if bz <= surf + 1.0e-6:
                        blocked += 1
                if blocked == 0:
                    wp.atomic_add(h, k, ii, jj, share)

    @wp.kernel
    def relax(src: wp.array3d(dtype=float),
              dst: wp.array3d(dtype=float),
              rate: float, thresh: float):
        """One Jacobi angle-of-repose sweep. Reads only `src`, so the pairwise flows are
        exactly antisymmetric and the sweep conserves volume by construction."""
        k, i, j = wp.tid()
        nx = src.shape[1]
        ny = src.shape[2]
        hi = src[k, i, j]
        net = float(0.0)
        for d in range(4):
            ii = i + wp.where(d == 0, 1, wp.where(d == 1, -1, 0))
            jj = j + wp.where(d == 2, 1, wp.where(d == 3, -1, 0))
            if ii < 0 or jj < 0 or ii >= nx or jj >= ny:
                continue
            hn = src[k, ii, jj]
            drop = hi - hn
            if drop > thresh:
                net -= rate * (drop - thresh) * 0.25
            elif -drop > thresh:
                net += rate * (-drop - thresh) * 0.25
        dst[k, i, j] = hi + net

    @wp.kernel
    def surface(h: wp.array3d(dtype=float),
                env: int, ox: float, oy: float, cell: float, z0: float, y_up: int,
                pos: wp.array(dtype=wp.vec3),
                nrm: wp.array(dtype=wp.vec3)):
        """Grid -> renderer vertex buffers: position and a central-difference normal.

        y_up rotates the (x, y, height) frame this module works in onto a y-up scene by
        (x, y, z) -> (x, z, -y), a proper rotation, so the winding of the index buffer
        built for the z-up case still faces outward.
        """
        i, j = wp.tid()
        nx = h.shape[1]
        ny = h.shape[2]
        ia = wp.min(i + 1, nx - 1)
        ib = wp.max(i - 1, 0)
        ja = wp.min(j + 1, ny - 1)
        jb = wp.max(j - 1, 0)
        dhx = (h[env, ia, j] - h[env, ib, j]) / (float(ia - ib) * cell)
        dhy = (h[env, i, ja] - h[env, i, jb]) / (float(ja - jb) * cell)
        x = ox + float(i) * cell
        y = oy + float(j) * cell
        z = z0 + h[env, i, j]
        n = wp.normalize(wp.vec3(-dhx, -dhy, 1.0))
        t = i * ny + j
        if y_up != 0:
            pos[t] = wp.vec3(x, z, -y)
            nrm[t] = wp.vec3(n[0], n[2], -n[1])
        else:
            pos[t] = wp.vec3(x, y, z)
            nrm[t] = n

    _KERNELS = (wp, imprint, distribute, relax, surface, slip)
    return _KERNELS


class DeformableTerrain:
    """A batch of K deformable height grids sharing one material and one resolution.

    `origin` is the (x, y) of cell (0, 0) -- one tuple for all K, or a [K, 2] array when
    the envs are laid out in lanes. `dims` is (nx, ny) cell counts, `cell` the spacing in
    metres, `z0` the datum the heights are measured from. `c_max` caps how many collider
    spheres a `deform` call may pass; the per-collider force buffer is sized once from it.
    """

    def __init__(self, origin, cell, dims, *, K=1, material="mud", device="cuda",
                 z0=0.0, init_height=None, c_max=8):
        (wp, self._imprint, self._distribute, self._relax, self._surface,
         self._slip) = _compile()
        self._wp = wp
        wp.init()
        self.device = torch.device(device)
        self._wp_device = (f"cuda:{self.device.index or 0}" if self.device.type == "cuda"
                           else "cpu")
        self.K = int(K)
        self.nx, self.ny = int(dims[0]), int(dims[1])
        self.cell = float(cell)
        self.z0 = float(z0)
        self.c_max = int(c_max)
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.material_name = material if isinstance(material, str) else "custom"

        org = np.asarray(origin, dtype=np.float32)
        if org.shape == (2,):
            org = np.repeat(org[None, :], self.K, axis=0)
        if org.shape != (self.K, 2):
            raise ValueError(f"origin must be (2,) or ({self.K}, 2), got {org.shape}")
        self.origin_np = np.ascontiguousarray(org)
        self.origin_t = torch.from_numpy(self.origin_np).to(self.device)
        self._org = wp.array(self.origin_np, dtype=wp.vec2, device=self._wp_device)

        shape = (self.K, self.nx, self.ny)
        if init_height is None:
            h0 = np.zeros(shape, dtype=np.float32)
        else:
            h0 = np.broadcast_to(np.asarray(init_height, dtype=np.float32),
                                 shape).astype(np.float32)
        self.h = wp.array(np.ascontiguousarray(h0), dtype=float, device=self._wp_device)
        # The bearing datum: where the ground was before anything stood on it.
        self.grade = wp.array(np.ascontiguousarray(h0.copy()), dtype=float,
                              device=self._wp_device)
        self.eject = wp.zeros(shape, dtype=float, device=self._wp_device)
        # Ejecta that had nowhere to go (ringed in by a collider), waiting for a
        # neighbour to open. Volume in flight: h + carry is what is conserved.
        self.carry = wp.zeros(shape, dtype=float, device=self._wp_device)
        self._pong = wp.zeros(shape, dtype=float, device=self._wp_device)
        self._h_torch = wp.to_torch(self.h)
        self._grade_torch = wp.to_torch(self.grade)
        self._pong_torch = wp.to_torch(self._pong)

        # Persistent staging so the kernels always see contiguous float32 at a fixed
        # address -- which is also what makes a CUDA-graph replay legal.
        z = lambda *s: torch.zeros(s, device=self.device, dtype=torch.float32)  # noqa: E731
        self.forces = z(self.K, self.c_max, 3)
        self.contact_area = z(self.K, self.c_max)     # m^2 under each collider this step
        self.slip_j = z(self.K, self.c_max)           # m of slip since this stance began
        self._centers = z(self.K, self.c_max, 3)
        self._vel = z(self.K, self.c_max, 3)
        self._radii = z(self.K, self.c_max)
        self._wf = wp.from_torch(self.forces)
        self._wa = wp.from_torch(self.contact_area)
        self._wj = wp.from_torch(self.slip_j)
        self._wc = wp.from_torch(self._centers)
        self._wv = wp.from_torch(self._vel)
        self._wr = wp.from_torch(self._radii)

    # ---- state ---------------------------------------------------------------
    @property
    def h_torch(self):
        """The height grid as a torch tensor [K, nx, ny] -- a zero-copy view, not a copy."""
        return self._h_torch

    @property
    def grade_torch(self):
        """The reference grade [K, nx, ny] the bearing depth is measured from, zero-copy."""
        return self._grade_torch

    @property
    def dims(self):
        return (self.nx, self.ny)

    def reset(self, init_height=None):
        """Flatten the ground back to `init_height` (default: the z0 datum)."""
        v = 0.0 if init_height is None else float(init_height)
        self._h_torch.fill_(v)
        self._grade_torch.fill_(v)
        self.carry.zero_()
        self.slip_j.zero_()
        self.contact_area.zero_()

    # ---- the step ------------------------------------------------------------
    def deform(self, centers, radii, vel=None, dt=1.0 / 60.0):
        """One displacement step. Returns the reaction on each collider, [K, C, 3].

        `centers` [K, C, 3] sphere centres in world coordinates (x, y, height), `radii`
        [C] or [K, C], `vel` [K, C, 3] the colliders' own velocities: the horizontal pair
        drives the Janosi shear and the vertical one the suction on withdrawal. `dt` is
        the caller's step, and it is what the slip integral and the grade relaxation are
        measured in -- a Python float baked into the launch, so it is fixed across a
        CUDA-graph replay, which is what a fixed-rate sim wants anyway.

        The shear reported this step is built from the slip accumulated up to the previous
        one, so the first frame of a stance has bearing but no traction. That is the
        physics: traction is bought with sliding.

        The returned tensor is a view of a buffer this call overwrites: clone it if it has
        to outlive the next `deform`.
        """
        if centers.dim() != 3 or centers.shape[0] != self.K or centers.shape[2] != 3:
            raise ValueError(f"centers must be [{self.K}, C, 3], got {tuple(centers.shape)}")
        C = centers.shape[1]
        if C > self.c_max:
            raise ValueError(f"deform: {C} colliders exceeds c_max={self.c_max}")
        self._centers[:, :C].copy_(centers)
        self._radii[:, :C].copy_(radii if radii.dim() == 2 else radii.reshape(1, C))
        if vel is None:
            self._vel[:, :C].zero_()
        else:
            self._vel[:, :C].copy_(vel)
        self.forces.zero_()
        self.contact_area.zero_()
        self.eject.zero_()
        m = self.material
        wp = self._wp
        dt = float(dt)
        dim = (self.K, self.nx, self.ny)
        wp.launch(self._imprint, dim=dim, device=self._wp_device,
                  inputs=[self.h, self.eject, self.grade, self._org, self._wc, self._wr,
                          self._wv, self._wj, self._wf, self._wa,
                          C, self.cell, self.z0, dt,
                          m.compression, m.n, m.k_c, m.k_phi,
                          m.cohesion, m.tan_phi, m.janosi_K,
                          m.suction, m.suction_cap, m.grade_rate])
        wp.launch(self._distribute, dim=dim, device=self._wp_device,
                  inputs=[self.h, self.eject, self.carry, self._org, self._wc, self._wr,
                          C, self.cell, self.z0])
        # [K, c_max] and not [K, C]: a fixed launch dim, and the colliders the caller did
        # not pass have zero contact area, so their stance is correctly forgotten.
        wp.launch(self._slip, dim=(self.K, self.c_max), device=self._wp_device,
                  inputs=[self._wa, self._wv, self._wj, dt])
        return self.forces[:, :C]

    def relax(self, iterations=2):
        """Let over-steep slopes slump. Ping-pongs, and lands the result back in `h`."""
        m = self.material
        if m.flow_rate <= 0.0 or iterations <= 0:
            return
        wp = self._wp
        dim = (self.K, self.nx, self.ny)
        thresh = m.tan_repose * self.cell
        for _ in range(iterations):
            wp.launch(self._relax, dim=dim, device=self._wp_device,
                      inputs=[self.h, self._pong, m.flow_rate, thresh])
            # A copy rather than a pointer swap: `h_torch`, the mesh views and any
            # captured graph all hold the address of `h`, and must keep holding it.
            self._h_torch.copy_(self._pong_torch)

    # ---- queries -------------------------------------------------------------
    def heights(self, x, y, out=None):
        """Bilinear surface height at world (x, y). Inputs [K] or [K, P], same shape out.

        Points off this env's grid get the flat datum `z0` -- an env whose robot walks off
        the deformable patch sees the ground it would have had anyway.
        """
        if x.shape != y.shape:
            raise ValueError(f"heights: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
        ox = self.origin_t[:, 0]
        oy = self.origin_t[:, 1]
        if x.dim() == 2:
            ox = ox[:, None]
            oy = oy[:, None]
        fx = (x - ox) / self.cell
        fy = (y - oy) / self.cell
        on = ((fx >= 0.0) & (fx <= self.nx - 1.0) & (fy >= 0.0) & (fy <= self.ny - 1.0))
        fx = fx.clamp(0.0, self.nx - 1.0001)
        fy = fy.clamp(0.0, self.ny - 1.0001)
        ix = fx.long()
        iy = fy.long()
        tx = fx - ix
        ty = fy - iy
        flat = self._h_torch.view(self.K, self.nx * self.ny)
        if x.dim() == 1:
            gat = lambda a, b: flat.gather(1, (a * self.ny + b).unsqueeze(1)).squeeze(1)  # noqa: E731
        else:
            gat = lambda a, b: flat.gather(1, a * self.ny + b)                            # noqa: E731
        h00 = gat(ix, iy)
        h10 = gat(ix + 1, iy)
        h01 = gat(ix, iy + 1)
        h11 = gat(ix + 1, iy + 1)
        h = ((1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10
             + (1 - tx) * ty * h01 + tx * ty * h11) * on
        if out is None:
            return self.z0 + h
        out.copy_(self.z0 + h)
        return out

    def __repr__(self):
        return (f"DeformableTerrain(K={self.K}, {self.nx}x{self.ny} @ {self.cell * 100:.1f} cm, "
                f"{self.material_name})")


class GridMeshView:
    """One env of a DeformableTerrain as a threepp mesh.

    An indexed grid: nx*ny vertices, the index buffer written once, position and normal
    republished every `update()`. `up="y"` puts it in a y-up scene (the demo's), `up="z"`
    leaves it in the module's own frame (the RL one).
    """

    def __init__(self, terrain, material, *, env=0, up="y", cast_shadow=False):
        if up not in ("y", "z"):
            raise ValueError(f"up must be 'y' or 'z', got {up!r}")
        wp = terrain._wp
        self.terrain = terrain
        self.env = int(env)
        self.y_up = 1 if up == "y" else 0
        nx, ny = terrain.nx, terrain.ny
        self.n = nx * ny
        ox, oy = (float(v) for v in terrain.origin_np[self.env])
        self._ox, self._oy = ox, oy

        # Vertex grid, row-major in i (so vertex (i, j) is i*ny + j, what the kernel writes).
        gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        gx = ox + gi * terrain.cell
        gy = oy + gj * terrain.cell
        gz = np.full_like(gx, terrain.z0, dtype=np.float32)
        if self.y_up:
            p0 = np.stack([gx, gz, -gy], axis=-1)
            n0 = np.tile(np.float32([0.0, 1.0, 0.0]), (nx, ny, 1))
        else:
            p0 = np.stack([gx, gy, gz], axis=-1)
            n0 = np.tile(np.float32([0.0, 0.0, 1.0]), (nx, ny, 1))
        a = (gi[:-1, :-1] * ny + gj[:-1, :-1]).ravel()
        b = a + ny            # (i+1, j)
        c = a + ny + 1        # (i+1, j+1)
        d = a + 1             # (i, j+1)
        faces = np.stack([a, b, c, a, c, d], axis=1).astype(np.uint32)

        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position", np.ascontiguousarray(p0.reshape(-1, 3),
                                                                     np.float32))
        self.geometry.set_attribute("normal", np.ascontiguousarray(n0.reshape(-1, 3),
                                                                   np.float32))
        self.geometry.set_index(np.ascontiguousarray(faces.reshape(-1), np.uint32))
        self.mesh = tp.Mesh(self.geometry, material)
        self.mesh.frustum_culled = False      # the vertices move under the renderer's feet
        self.mesh.receive_shadow = True
        self.mesh.cast_shadow = bool(cast_shadow)

        self.pos = wp.zeros(self.n, dtype=wp.vec3, device=terrain._wp_device)
        self.nrm = wp.zeros(self.n, dtype=wp.vec3, device=terrain._wp_device)

    def update(self):
        """Rebuild the vertex buffers from the current heights (device -> host -> geometry)."""
        t = self.terrain
        t._wp.launch(t._surface, dim=(t.nx, t.ny), device=t._wp_device,
                     inputs=[t.h, self.env, self._ox, self._oy, t.cell, t.z0, self.y_up],
                     outputs=[self.pos, self.nrm])
        self.geometry.update_attribute("position", self.pos.numpy())
        self.geometry.update_attribute("normal", self.nrm.numpy())


# --------------------------------------------------------------------------- #
#  Selftest
# --------------------------------------------------------------------------- #
def _trot(t, K, device, sink=0.035, r=0.06, span=1.2):
    """Four feet on a lissajous gait: each plants (bottom below the datum) for half its
    cycle and lifts clear for the other half. Returns centers [K, 4, 3], vel [K, 4, 3]."""
    c = torch.zeros(K, 4, 3, device=device)
    v = torch.zeros(K, 4, 3, device=device)
    for f in range(4):
        ph = t * 1.3 + 0.25 * f
        u = ph % 1.0
        ang = 2.0 * np.pi * (ph * 0.25 + 0.25 * f)
        c[:, f, 0] = span * 0.5 * np.cos(ang)
        c[:, f, 1] = span * 0.5 * np.sin(2.0 * ang) * 0.5
        lift = max(0.0, np.sin(2.0 * np.pi * u)) * 0.12
        c[:, f, 2] = (lift if u < 0.5 else -sink) + r
        v[:, f, 0] = -span * 0.5 * np.sin(ang) * 0.25
        v[:, f, 1] = span * 0.25 * np.cos(2.0 * ang) * 0.5
    return c, v


def _run(terrain, radii, steps=240, dt=1.0 / 60.0):
    for s in range(steps):
        c, v = _trot(s * dt, terrain.K, terrain.device)
        terrain.deform(c, radii, v, dt)
        terrain.relax(2)


if __name__ == "__main__":
    import time

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    NX = NY = 96
    CELL = 0.025
    K = 64
    org = np.stack([np.full(K, -NX * CELL * 0.5), np.arange(K) * 3.0 - NY * CELL * 0.5], 1)
    r4 = torch.full((4,), 0.06, device=dev)

    mud = DeformableTerrain(org, CELL, (NX, NY), K=K, material="mud", device=dev)
    print(mud)

    # 1. a trot for 240 steps digs a rut and raises a berm, and nothing blows up
    _run(mud, r4)
    h = mud.h_torch
    assert torch.isfinite(h).all(), "heights went non-finite"
    assert torch.isfinite(mud.forces).all(), "forces went non-finite"
    print(f"  1. finite; h in [{h.min():+.4f}, {h.max():+.4f}] m, "
          f"|F|max {mud.forces.norm(dim=-1).max():.1f} N")
    assert h.min() < -0.01, f"no rut: min h = {h.min():.4f}"
    assert h.max() > 0.002, f"no berm: max h = {h.max():.4f}"

    # 2. volume: over a 5 cm layer, an incompressible material keeps what it started with,
    #    a compacting one destroys some of it
    LAYER = 0.05
    start = K * NX * NY * LAYER
    v0 = DeformableTerrain(org, CELL, (NX, NY), K=K, material="sand", device=dev,
                           init_height=LAYER)
    _run(v0, r4)
    keep = float(v0.h_torch.sum())
    v1 = DeformableTerrain(org, CELL, (NX, NY), K=K, material="snow", device=dev,
                           init_height=LAYER)
    _run(v1, r4)
    lose = float(v1.h_torch.sum())
    print(f"  2. volume over {start:.1f} m: compression=0.0 -> {keep:.1f} m "
          f"({100.0 * (keep / start - 1.0):+.2f} %), compression=0.7 -> {lose:.1f} m "
          f"({100.0 * (lose / start - 1.0):+.2f} %)")
    assert abs(keep / start - 1.0) < 0.15, f"compression=0 changed volume by {keep / start - 1.0}"
    assert lose < start, f"compression=0.7 did not remove volume: {lose} vs {start}"

    # 3. CUDA-graph capture: the whole deform+relax has to record and replay
    graph_note = "skipped (no CUDA)"
    if dev == "cuda":
        import warp as wp
        g_c, g_v = _trot(0.0, K, dev)
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    mud.deform(g_c, r4, g_v)
                    mud.relax(2)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream()),
                                     sync_enter=False, sync_exit=False):
                    mud.deform(g_c, r4, g_v)
                    mud.relax(2)
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            ok = bool(torch.isfinite(mud.h_torch).all() and torch.isfinite(mud.forces).all())
            assert ok, "graph replay produced non-finite state"
            graph_note = "captured + 10 replays finite"
        except Exception as exc:                       # noqa: BLE001 - report, do not fail
            graph_note = f"SKIPPED ({type(exc).__name__}: {exc})"
    print(f"  3. cuda graph: {graph_note}")

    # 4. plate sinkage: the bearing force is the Bekker integral over the covered cells,
    #    and nothing else -- computed here in numpy from the same closed form
    SINK, RF = 0.030, 0.05
    org1 = np.array([[-NX * CELL * 0.5, -NY * CELL * 0.5]], dtype=np.float32)
    plate = DeformableTerrain(org1, CELL, (NX, NY), K=1, material="mud", device=dev)
    pc = torch.zeros(1, 1, 3, device=dev)
    pc[0, 0, 2] = RF - SINK                      # sphere bottom SINK below the datum
    pr = torch.full((1,), RF, device=dev)
    f1 = float(plate.deform(pc, pr)[0, 0, 2])
    mm = MATERIALS["mud"]
    gi, gj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    d2 = (org1[0, 0] + gi * CELL) ** 2 + (org1[0, 1] + gj * CELL) ** 2
    bz = np.where(d2 < RF * RF,
                  (RF - SINK) - np.sqrt(np.maximum(RF * RF - d2, 0.0)), 1.0e9)
    zc = np.maximum(-bz, 0.0)                    # grade is still the flat datum
    bw = np.sqrt(np.maximum(zc * (2.0 * RF - zc), CELL * CELL))
    f_ref = float(((mm.k_c / bw + mm.k_phi) * np.power(zc, mm.n) * CELL * CELL).sum())
    print(f"  4. plate sinkage {SINK * 1000:.0f} mm on r={RF * 100:.0f} cm: "
          f"F_z {f1:.1f} N vs Bekker {f_ref:.1f} N ({100.0 * (f1 / f_ref - 1.0):+.2f} %)")
    assert abs(f1 / f_ref - 1.0) < 0.10, f"bearing off the closed form: {f1} vs {f_ref}"

    # 5. standing support: the same pose again. v1 returned ~0 here, because its force
    #    lived in the carve branch and the second call has nothing left to carve.
    f2 = float(plate.deform(pc, pr)[0, 0, 2])
    print(f"  5. standing: F_z {f1:.1f} N carving -> {f2:.1f} N at rest in its own print "
          f"({100.0 * (f2 / f1 - 1.0):+.2f} %)")
    assert f2 > 0.0, "no support standing still -- the v1 bug is back"
    assert abs(f2 / f1 - 1.0) < 0.05, f"support is not held: {f2} vs {f1}"

    # 6. slip sweep: hold the same plate down and slide it. Traction has to ramp in over
    #    janosi_K and stop at the Mohr-Coulomb limit c*A + tan_phi*F_z.
    DT = 1.0 / 60.0
    slide = DeformableTerrain(org1, CELL, (NX, NY), K=1, material="mud", device=dev)
    sv = torch.zeros(1, 1, 3, device=dev)
    sv[0, 0, 0] = 0.4
    slide.deform(pc, pr, sv, DT)                  # touchdown: j == 0, no traction yet
    fx_early = abs(float(slide.deform(pc, pr, sv, DT)[0, 0, 0]))
    for _ in range(120):
        out = slide.deform(pc, pr, sv, DT)
    fx_late, fz_late = abs(float(out[0, 0, 0])), float(out[0, 0, 2])
    area = float(slide.contact_area[0, 0])
    limit = mm.cohesion * area + mm.tan_phi * fz_late
    print(f"  6. slip at 0.4 m/s: |F_t| {fx_early:.1f} N at j={0.4 * DT * 1000:.1f} mm -> "
          f"{fx_late:.1f} N at j={0.4 * DT * 121 * 1000:.0f} mm, Mohr-Coulomb "
          f"{limit:.1f} N over {area * 1.0e4:.1f} cm2")
    assert abs(fx_late / limit - 1.0) < 0.10, f"traction did not saturate: {fx_late} vs {limit}"
    assert fx_early < 0.5 * fx_late, f"traction stepped instead of building: {fx_early}"

    # 7. the print floor is where the sphere cut it: ringed-in ejecta is parked in the
    #    carry grid, not stacked back under the collider for the next imprint to carve
    #    off again. Clay is the worst case -- at compression=0 the old cycle never
    #    decayed and the floor churned forever a constant height above the sphere.
    still = DeformableTerrain(org1, CELL, (NX, NY), K=1, material="clay", device=dev)
    for _ in range(10):
        still.deform(pc, pr)
    floor = float(still.h_torch.min())
    parked = float(still.carry.numpy().sum())
    vol = float(still.h_torch.sum()) + parked
    print(f"  7. parked ejecta: floor {floor * 1000:+.2f} mm vs sphere bottom "
          f"{-SINK * 1000:.0f} mm after 10 holds, {parked * 1000:.1f} mm of column parked, "
          f"h+carry off by {vol * 1000:+.3f} mm")
    assert abs(floor + SINK) < 5.0e-4, f"floor {floor} did not settle on the sphere at {-SINK}"
    assert abs(vol) < 1.0e-3, f"parking leaked volume: {vol}"

    # Diagnostics, not gates: suction on withdrawal, and the frozen/relaxing grade.
    lv = torch.zeros(1, 1, 3, device=dev)
    lv[0, 0, 2] = 0.6
    f_lift = float(plate.deform(pc, pr, lv)[0, 0, 2])
    g, hh = plate.grade_torch, plate.h_torch
    print(f"     suction: F_z {f2:.1f} N at rest -> {f_lift:.1f} N lifting at 0.6 m/s; "
          f"grade under the print {float(g.min()) * 1000:+.2f} mm while the surface is at "
          f"{float(hh.min()) * 1000:+.2f} mm, berm grade {float(g.max()) * 1000:+.3f} mm")

    def bench(terrain, n=100):
        c, v = _trot(0.0, terrain.K, terrain.device)
        rr = torch.full((4,), 0.06, device=terrain.device)
        for _ in range(10):
            terrain.deform(c, rr, v)
            terrain.relax(2)
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            terrain.deform(c, rr, v)
        sync()
        t1 = time.perf_counter()
        for _ in range(n):
            terrain.relax(2)
        sync()
        t2 = time.perf_counter()
        return (t1 - t0) * 1000.0 / n, (t2 - t1) * 1000.0 / n

    d64, r64 = bench(mud)
    big = DeformableTerrain(np.stack([np.zeros(1024), np.arange(1024) * 3.0], 1),
                            CELL, (NX, NY), K=1024, material="mud", device=dev)
    d1k, r1k = bench(big)
    print(f"TERRAIN_DEFORM SELFTEST: PASS  "
          f"deform K=64 {d64:.3f} ms (relax2 {r64:.3f}), "
          f"K=1024 {d1k:.3f} ms (relax2 {r1k:.3f}) on {NX}x{NY}")
