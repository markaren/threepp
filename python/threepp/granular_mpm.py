"""MLS-MPM granular core with load-driven rigid colliders (two-way coupling).

The same math as warp_mudsnow_drive.py's gravel lane (gb_p2g / gb_grid /
gb_g2p): quadratic B-splines, APIC transfer, Hencky (log-strain) elasticity and
a Drucker-Prager return map on the log singular values -- but parameterised
(grid origin, cell size, dims, E, nu, rho, friction angle, cohesion, dt) instead
of module constants, so a testbed and a drive demo can share it.

What is new against the demo is the COLLIDER. A body (a cylinder wheel or an
axis-aligned box plate) enforces a velocity boundary condition v_b + omega x r
with Coulomb friction on the grid nodes it covers. The momentum that BC removes
from the grid each substep is an impulse the soil put into the body; divided by
dt it is a FORCE (a grid momentum is a momentum -- unlike a PBD position
correction, which is not) and its moment about the body centre is the torque
the soil puts on the axle. Both are accumulated on the GPU.

A body in mode FREE_Y has a DYNAMIC vertical position: every substep a
one-thread-per-body kernel integrates

    m_eff * y'' = F_soil_y - W - c * y'

with W an external load (a dead weight in a testbed; PhysX's suspension force
in a car). The body sinks until the grains carry W. Horizontal velocity and
spin stay prescribed. Everything runs on the device, so a whole frame of
substeps is one CUDA graph.

Two things the demo's material law does not have, both optional:

  * a compaction cap (Material(cap_p0=..., cap_lambda=...)): elastic volumetric
    strain beyond what the pre-consolidation pressure p_c allows becomes plastic
    compaction, hardening p_c = p0 exp(-e_vp / lambda). Without it the DP cone
    can only flow, never compact, and a loaded wheel's sinkage is mostly
    ELASTIC (measured: 59-64 % rebound at E = 3-10 MPa). Loose soil needs it.
  * the bin floor and walls accumulate the momentum THEY remove too
    (take_wall_force), the far end of the load path: with a body carrying W,
    floor reaction - soil weight = W is the independent closure check.

Stress units: E and cohesion in Pa, rho in kg/m^3, lengths in m. y is up.

    pip install warp-lang
"""
import math

import numpy as np

KINEMATIC = 0
FREE_Y = 1
CYLINDER = 0
BOX = 1

_K = None


def _compile():
    """Import Warp and build the kernels. Deferred so importing this module is free."""
    global _K
    if _K is not None:
        return _K
    import warp as wp

    @wp.struct
    class Params:
        origin: wp.vec3
        h: float
        inv_h: float
        nx: int
        ny: int
        nz: int
        vol: float
        pmass: float
        mu: float
        la: float
        dp_k: float        # alpha * (3 la + 2 mu) / (2 mu): the cone slope on tr(eps)
        coh: float         # cohesion, as a log-strain offset of the cone
        tr_apex: float     # tr(eps) at the cone apex (0 without cohesion)
        apic: float
        grav: float
        vmax: float
        lo: wp.vec3        # bin interior: floor at lo.y, walls at lo/hi x and z
        hi: wp.vec3
        mu_floor: float
        mu_wall: float
        eps: float         # collider skin, m
        bulk: float        # la + 2 mu / 3
        cap_p0: float      # compaction cap: initial pre-consolidation pressure, Pa
        cap_lam: float     # plastic volumetric strain per e-fold of cap pressure (0 = off)

    @wp.func
    def collide(v: wp.vec3, n: wp.vec3, vc: wp.vec3, mu: float) -> wp.vec3:
        """Project a grid velocity out of a moving collider, with Coulomb friction."""
        rel = v - vc
        vn = wp.dot(rel, n)
        if vn >= 0.0:
            return v
        vt = rel - n * vn
        vtl = wp.length(vt)
        if vtl > 1.0e-7:
            vt = vt * wp.max(0.0, 1.0 + mu * vn / vtl)
        else:
            vt = wp.vec3(0.0, 0.0, 0.0)
        return vc + vt

    @wp.func
    def bspline(f: float) -> wp.vec3:
        return wp.vec3(0.5 * (1.5 - f) * (1.5 - f),
                       0.75 - (f - 1.0) * (f - 1.0),
                       0.5 * (f - 0.5) * (f - 0.5))

    @wp.kernel
    def p2g(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
            C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
            gm: wp.array3d(dtype=float), gv: wp.array3d(dtype=wp.vec3),
            P: Params, dt: float):
        p = wp.tid()
        xp = x[p]
        gx = (xp[0] - P.origin[0]) * P.inv_h
        gy = (xp[1] - P.origin[1]) * P.inv_h
        gz = (xp[2] - P.origin[2]) * P.inv_h
        bi = int(wp.floor(gx - 0.5))
        bj = int(wp.floor(gy - 0.5))
        bk = int(wp.floor(gz - 0.5))
        if bi < 0 or bj < 0 or bk < 0 or bi > P.nx - 3 or bj > P.ny - 3 or bk > P.nz - 3:
            return
        fx = wp.vec3(gx - float(bi), gy - float(bj), gz - float(bk))
        wx = bspline(fx[0])
        wy = bspline(fx[1])
        wz = bspline(fx[2])
        U = wp.mat33()
        V = wp.mat33()
        sig = wp.vec3()
        wp.svd3(F[p], U, sig, V)
        e0 = wp.log(wp.max(sig[0], 1.0e-4))
        e1 = wp.log(wp.max(sig[1], 1.0e-4))
        e2 = wp.log(wp.max(sig[2], 1.0e-4))
        tr = e0 + e1 + e2
        t = wp.vec3(2.0 * P.mu * e0 + P.la * tr,
                    2.0 * P.mu * e1 + P.la * tr,
                    2.0 * P.mu * e2 + P.la * tr)
        tau = U * wp.diag(t) * wp.transpose(U)
        affine = tau * (-dt * P.vol * 4.0 * P.inv_h * P.inv_h) + C[p] * P.pmass
        mv = v[p] * P.pmass
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    w = wx[a] * wy[b] * wz[c]
                    dpos = wp.vec3((float(a) - fx[0]) * P.h,
                                   (float(b) - fx[1]) * P.h,
                                   (float(c) - fx[2]) * P.h)
                    wp.atomic_add(gm, bi + a, bj + b, bk + c, w * P.pmass)
                    wp.atomic_add(gv, bi + a, bj + b, bk + c, (mv + affine * dpos) * w)

    @wp.kernel
    def grid(gm: wp.array3d(dtype=float), gv: wp.array3d(dtype=wp.vec3), P: Params,
             nb: int,
             bpos: wp.array(dtype=wp.vec3), bvel: wp.array(dtype=wp.vec3),
             bomega: wp.array(dtype=wp.vec3), baxis: wp.array(dtype=wp.vec3),
             bshape: wp.array(dtype=int), bsize: wp.array(dtype=wp.vec3),
             bmu: wp.array(dtype=float),
             react: wp.array(dtype=wp.vec3), rtorque: wp.array(dtype=wp.vec3),
             wreact: wp.array(dtype=wp.vec3), dt: float):
        """Momentum -> velocity, gravity, the bodies (with reaction), the bin.

        `wreact` collects the impulse the bin floor and walls take out of the
        grid: the independent end of the load path (W + soil weight)."""
        i, j, k = wp.tid()
        m = gm[i, j, k]
        if m <= 1.0e-12:
            gv[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
            return
        v = gv[i, j, k] * (1.0 / m) + wp.vec3(0.0, P.grav * dt, 0.0)
        p = wp.vec3(P.origin[0] + float(i) * P.h, P.origin[1] + float(j) * P.h,
                    P.origin[2] + float(k) * P.h)
        for q in range(nb):
            c = bpos[q]
            d = p - c
            sz = bsize[q]
            hit = int(0)
            n = wp.vec3(0.0, 1.0, 0.0)
            lever = d
            if bshape[q] == 0:
                ax = baxis[q]
                a = wp.dot(d, ax)
                rad = d - ax * a
                rl = wp.length(rad)
                if wp.abs(a) <= sz[1] and rl < sz[0] + P.eps:
                    hit = 1
                    n = rad * (1.0 / wp.max(rl, 1.0e-9))
                    lever = n * sz[0]
            else:
                qx = wp.abs(d[0]) - sz[0]
                qy = wp.abs(d[1]) - sz[1]
                qz = wp.abs(d[2]) - sz[2]
                mq = wp.max(qx, wp.max(qy, qz))
                if mq < P.eps:
                    hit = 1
                    if qy >= qx and qy >= qz:
                        n = wp.vec3(0.0, wp.sign(d[1]), 0.0)
                    elif qx >= qz:
                        n = wp.vec3(wp.sign(d[0]), 0.0, 0.0)
                    else:
                        n = wp.vec3(0.0, 0.0, wp.sign(d[2]))
            if hit == 1:
                before = v
                v = collide(v, n, bvel[q] + wp.cross(bomega[q], lever), bmu[q])
                dp = (before - v) * m        # impulse the soil put INTO the body
                wp.atomic_add(react, q, dp)
                wp.atomic_add(rtorque, q, wp.cross(lever, dp))
        zero = wp.vec3(0.0, 0.0, 0.0)
        vb = v
        if p[1] <= P.lo[1]:
            v = collide(v, wp.vec3(0.0, 1.0, 0.0), zero, P.mu_floor)
        if p[0] <= P.lo[0]:
            v = collide(v, wp.vec3(1.0, 0.0, 0.0), zero, P.mu_wall)
        if p[0] >= P.hi[0]:
            v = collide(v, wp.vec3(-1.0, 0.0, 0.0), zero, P.mu_wall)
        if p[2] <= P.lo[2]:
            v = collide(v, wp.vec3(0.0, 0.0, 1.0), zero, P.mu_wall)
        if p[2] >= P.hi[2]:
            v = collide(v, wp.vec3(0.0, 0.0, -1.0), zero, P.mu_wall)
        dw = vb - v
        if wp.length(dw) > 0.0:
            wp.atomic_add(wreact, 0, dw * m)
        sp = wp.length(v)
        if sp > P.vmax:
            v = v * (P.vmax / sp)
        gv[i, j, k] = v

    @wp.kernel
    def g2p(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
            C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
            evp: wp.array(dtype=float),
            gv: wp.array3d(dtype=wp.vec3), P: Params, dt: float):
        p = wp.tid()
        xp = x[p]
        gx = (xp[0] - P.origin[0]) * P.inv_h
        gy = (xp[1] - P.origin[1]) * P.inv_h
        gz = (xp[2] - P.origin[2]) * P.inv_h
        bi = int(wp.floor(gx - 0.5))
        bj = int(wp.floor(gy - 0.5))
        bk = int(wp.floor(gz - 0.5))
        if bi < 0 or bj < 0 or bk < 0 or bi > P.nx - 3 or bj > P.ny - 3 or bk > P.nz - 3:
            v[p] = wp.vec3(0.0, 0.0, 0.0)
            return
        fx = wp.vec3(gx - float(bi), gy - float(bj), gz - float(bk))
        wx = bspline(fx[0])
        wy = bspline(fx[1])
        wz = bspline(fx[2])
        nv = wp.vec3(0.0, 0.0, 0.0)
        nc = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    w = wx[a] * wy[b] * wz[c]
                    g = gv[bi + a, bj + b, bk + c]
                    nv += g * w
                    nc += wp.outer(g, wp.vec3(float(a) - fx[0], float(b) - fx[1],
                                              float(c) - fx[2])) * (4.0 * P.inv_h * w)
        nc = nc * P.apic
        Fn = (wp.identity(n=3, dtype=float) + nc * dt) * F[p]
        U = wp.mat33()
        V = wp.mat33()
        sig = wp.vec3()
        wp.svd3(Fn, U, sig, V)
        s0 = wp.min(wp.max(sig[0], 0.05), 4.0)
        s1 = wp.min(wp.max(sig[1], 0.05), 4.0)
        s2 = wp.min(wp.max(sig[2], 0.05), 4.0)
        # Drucker-Prager on the log singular values (Klar et al. 2016): past the
        # apex -> stress free; outside the cone -> radial return onto it.
        e0 = wp.log(s0)
        e1 = wp.log(s1)
        e2 = wp.log(s2)
        tr = e0 + e1 + e2
        if P.cap_lam > 0.0:
            # Compaction cap (loose soil): the elastic volumetric strain may not
            # exceed what the current pre-consolidation pressure allows; the
            # excess becomes plastic compaction, which hardens the cap along a
            # normal-compression line  p_c = p0 exp(-evp / lam).
            pc = P.cap_p0 * wp.exp(-evp[p] / P.cap_lam)
            trc = -pc / P.bulk
            if tr < trc:
                dv = tr - trc
                evp[p] = evp[p] + dv
                e0 = e0 - dv / 3.0
                e1 = e1 - dv / 3.0
                e2 = e2 - dv / 3.0
                tr = trc
                s0 = wp.exp(e0)
                s1 = wp.exp(e1)
                s2 = wp.exp(e2)
        if tr >= P.tr_apex:
            kk = wp.exp(P.tr_apex / 3.0)
            s0 = kk
            s1 = kk
            s2 = kk
        else:
            h0 = e0 - tr / 3.0
            h1 = e1 - tr / 3.0
            h2 = e2 - tr / 3.0
            fn = wp.sqrt(h0 * h0 + h1 * h1 + h2 * h2)
            dg = fn - P.coh + P.dp_k * tr
            if dg > 0.0 and fn > 1.0e-9:
                sc = dg / fn
                s0 = wp.exp(e0 - h0 * sc)
                s1 = wp.exp(e1 - h1 * sc)
                s2 = wp.exp(e2 - h2 * sc)
        F[p] = U * wp.diag(wp.vec3(s0, s1, s2)) * wp.transpose(V)
        C[p] = nc
        v[p] = nv
        q = xp + nv * dt
        pad = 0.05 * P.h
        x[p] = wp.vec3(wp.min(wp.max(q[0], P.lo[0] + pad), P.hi[0] - pad),
                       wp.min(wp.max(q[1], P.lo[1] + pad),
                              P.origin[1] + float(P.ny - 3) * P.h),
                       wp.min(wp.max(q[2], P.lo[2] + pad), P.hi[2] - pad))

    @wp.kernel
    def body_step(bpos: wp.array(dtype=wp.vec3), bvel: wp.array(dtype=wp.vec3),
                  mode: wp.array(dtype=int), load: wp.array(dtype=float),
                  mass: wp.array(dtype=float), damp: wp.array(dtype=float),
                  react: wp.array(dtype=wp.vec3), rtorque: wp.array(dtype=wp.vec3),
                  fsum: wp.array(dtype=wp.vec3), tsum: wp.array(dtype=wp.vec3),
                  nsum: wp.array(dtype=float), dt: float):
        """One thread per body: the soil's force this substep, then y'' = (F_y - W)/m."""
        q = wp.tid()
        f = react[q] * (1.0 / dt)
        t = rtorque[q] * (1.0 / dt)
        vel = bvel[q]
        if mode[q] == 1:
            vy = vel[1] + dt * (f[1] - load[q] - damp[q] * vel[1]) / mass[q]
            vel = wp.vec3(vel[0], vy, vel[2])
            bvel[q] = vel
        bpos[q] = bpos[q] + vel * dt
        fsum[q] = fsum[q] + f
        tsum[q] = tsum[q] + t
        nsum[q] = nsum[q] + 1.0
        react[q] = wp.vec3(0.0, 0.0, 0.0)
        rtorque[q] = wp.vec3(0.0, 0.0, 0.0)

    _K = dict(wp=wp, Params=Params, p2g=p2g, grid=grid, g2p=g2p, body_step=body_step)
    return _K


class Material:
    """Soil constants in SI. cohesion in Pa, phi in degrees."""

    def __init__(self, E=3.0e6, nu=0.3, rho=1700.0, phi_deg=36.0, cohesion=0.0, apic=0.94,
                 cap_p0=0.0, cap_lambda=0.0):
        self.E, self.nu, self.rho = float(E), float(nu), float(rho)
        self.cap_p0, self.cap_lambda = float(cap_p0), float(cap_lambda)
        self.phi_deg, self.cohesion, self.apic = float(phi_deg), float(cohesion), float(apic)

    @property
    def mu(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def la(self):
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def p_wave(self):
        """Dilatational wave speed sqrt((la + 2 mu)/rho), m/s: what sets the CFL."""
        return math.sqrt((self.la + 2.0 * self.mu) / self.rho)

    def cfl_dt(self, h, cfl=0.35, vmax=0.0):
        return cfl * h / (self.p_wave + vmax)


class GranularMPM:
    """A soil bin of MLS-MPM grains plus up to `max_bodies` rigid colliders.

    lo, hi: the bin interior (floor at lo[1]; walls at lo/hi in x and z; open
    top). The grid is laid over it with a 3-cell margin and `headroom` metres of
    air above `fill_height`.
    """

    def __init__(self, material, h, lo, hi, fill_height, dt, device="cuda:0",
                 ppc_spacing=0.5, headroom=0.3, max_bodies=4, mu_floor=0.6,
                 mu_wall=0.3, eps_cells=0.25, vmax=8.0, seed=3):
        K = _compile()
        wp = K["wp"]
        self.wp, self.K = wp, K
        self.mat, self.h, self.dt = material, float(h), float(dt)
        self.dev = device
        self.lo = np.asarray(lo, np.float64)
        self.hi = np.asarray(hi, np.float64)
        self.pd = ppc_spacing * self.h
        origin = self.lo - 3.0 * self.h
        top = self.lo[1] + fill_height + headroom
        dims = (int(math.ceil((self.hi[0] + 3 * self.h - origin[0]) / self.h)) + 1,
                int(math.ceil((top - origin[1]) / self.h)) + 1,
                int(math.ceil((self.hi[2] + 3 * self.h - origin[2]) / self.h)) + 1)
        self.origin, self.dims = origin, dims

        # A jittered lattice filling the bin to `fill_height`.
        pd = self.pd
        n = [max(1, int(round((self.hi[0] - self.lo[0]) / pd))),
             max(1, int(round(fill_height / pd))),
             max(1, int(round((self.hi[2] - self.lo[2]) / pd)))]
        ix, iy, iz = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]),
                                 indexing="ij")
        p0 = np.stack([self.lo[0] + (ix.ravel() + 0.5) * pd,
                       self.lo[1] + (iy.ravel() + 0.5) * pd,
                       self.lo[2] + (iz.ravel() + 0.5) * pd], axis=-1)
        p0 += np.random.default_rng(seed).uniform(-0.2 * pd, 0.2 * pd, p0.shape)
        self.p0 = np.ascontiguousarray(p0, np.float32)
        self.n = len(self.p0)
        self.fill_height = fill_height

        m = material
        P = K["Params"]()
        P.origin = wp.vec3(*origin)
        P.h = self.h
        P.inv_h = 1.0 / self.h
        P.nx, P.ny, P.nz = dims
        P.vol = pd ** 3
        P.pmass = m.rho * pd ** 3
        P.mu = m.mu
        P.la = m.la
        sphi = math.sin(math.radians(m.phi_deg))
        alpha = math.sqrt(2.0 / 3.0) * 2.0 * sphi / (3.0 - sphi)
        P.dp_k = alpha * (3.0 * m.la + 2.0 * m.mu) / (2.0 * m.mu)
        # Cohesion c (Pa) on the compression-meridian DP fit of Mohr-Coulomb:
        # sqrt(J2) <= A p + B, B = 6 c cos(phi) / (sqrt(3) (3 - sin(phi))). In the
        # log-strain cone that is ||dev eps|| <= dp_k (-tr eps) + sqrt(2) B / (2 mu).
        B = 6.0 * m.cohesion * math.cos(math.radians(m.phi_deg)) / (math.sqrt(3.0) * (3.0 - sphi))
        P.coh = math.sqrt(2.0) * B / (2.0 * m.mu)
        P.tr_apex = P.coh / P.dp_k if P.dp_k > 0 else 0.0
        P.apic = m.apic
        P.grav = -9.81
        P.vmax = vmax
        P.lo = wp.vec3(*self.lo)
        P.hi = wp.vec3(*self.hi)
        P.mu_floor = mu_floor
        P.mu_wall = mu_wall
        P.eps = eps_cells * self.h
        P.bulk = m.la + 2.0 * m.mu / 3.0
        P.cap_p0 = m.cap_p0
        P.cap_lam = m.cap_lambda
        self.P = P
        self.eps = P.eps

        self.x = wp.array(self.p0, dtype=wp.vec3, device=device)
        self.v = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.C = wp.zeros(self.n, dtype=wp.mat33, device=device)
        self.F = wp.array(np.tile(np.eye(3, dtype=np.float32), (self.n, 1, 1)),
                          dtype=wp.mat33, device=device)
        self.evp = wp.zeros(self.n, dtype=float, device=device)
        self.gm = wp.zeros(dims, dtype=float, device=device)
        self.gv = wp.zeros(dims, dtype=wp.vec3, device=device)

        nb = max_bodies
        self.nb = 0
        self.max_bodies = nb
        z3 = lambda: wp.zeros(nb, dtype=wp.vec3, device=device)   # noqa: E731
        self.bpos, self.bvel, self.bomega = z3(), z3(), z3()
        self.baxis = wp.array(np.tile(np.float32([0, 0, 1]), (nb, 1)), dtype=wp.vec3, device=device)
        self.bsize = z3()
        self.bshape = wp.zeros(nb, dtype=int, device=device)
        self.bmu = wp.zeros(nb, dtype=float, device=device)
        self.bmode = wp.zeros(nb, dtype=int, device=device)
        self.bload = wp.zeros(nb, dtype=float, device=device)
        self.bmass = wp.ones(nb, dtype=float, device=device)
        self.bdamp = wp.zeros(nb, dtype=float, device=device)
        self.react, self.rtorque = z3(), z3()
        self.fsum, self.tsum = z3(), z3()
        self.nsum = wp.zeros(nb, dtype=float, device=device)
        self.wreact = wp.zeros(1, dtype=wp.vec3, device=device)
        self._wsub = 0
        self._graph = None
        self._graph_n = 0

    # -- bodies ------------------------------------------------------------------

    def _set(self, arr, i, val):
        a = arr.numpy()
        a[i] = val
        arr.assign(a)

    def add_body(self, shape, size, pos, vel=(0, 0, 0), omega=(0, 0, 0), axis=(0, 0, 1),
                 mu=0.6, mode=KINEMATIC, load=0.0, mass=1.0, damp=0.0):
        """size: CYLINDER (radius, half-width, -); BOX (half extents)."""
        i = self.nb
        assert i < self.max_bodies
        self.nb += 1
        self._set(self.bshape, i, shape)
        self._set(self.bsize, i, size)
        self._set(self.bpos, i, pos)
        self._set(self.bvel, i, vel)
        self._set(self.bomega, i, omega)
        self._set(self.baxis, i, axis)
        self._set(self.bmu, i, mu)
        self._set(self.bmode, i, mode)
        self._set(self.bload, i, load)
        self._set(self.bmass, i, mass)
        self._set(self.bdamp, i, damp)
        self._graph = None
        return i

    def set_body(self, i, **kw):
        """Host-side update between frames (arrays keep their addresses, so a
        captured graph sees the new values)."""
        table = dict(pos=self.bpos, vel=self.bvel, omega=self.bomega, load=self.bload,
                     mass=self.bmass, mode=self.bmode, damp=self.bdamp, mu=self.bmu)
        for k, val in kw.items():
            self._set(table[k], i, val)

    def body_state(self, i):
        return self.bpos.numpy()[i].astype(np.float64), self.bvel.numpy()[i].astype(np.float64)

    def take_mean_force(self):
        """Mean soil force and torque on each body since the last call (N, N m)."""
        n = np.maximum(self.nsum.numpy(), 1.0)[:, None]
        f = self.fsum.numpy() / n
        t = self.tsum.numpy() / n
        self.fsum.zero_()
        self.tsum.zero_()
        self.nsum.zero_()
        return f[:self.nb].astype(np.float64), t[:self.nb].astype(np.float64)

    # -- stepping ----------------------------------------------------------------

    def _substep(self):
        wp, K, dev = self.wp, self.K, self.dev
        self.gm.zero_()
        self.gv.zero_()
        wp.launch(K["p2g"], dim=self.n, device=dev,
                  inputs=[self.x, self.v, self.C, self.F, self.gm, self.gv, self.P, self.dt])
        wp.launch(K["grid"], dim=self.dims, device=dev,
                  inputs=[self.gm, self.gv, self.P, self.nb, self.bpos, self.bvel,
                          self.bomega, self.baxis, self.bshape, self.bsize, self.bmu,
                          self.react, self.rtorque, self.wreact, self.dt])
        if self.nb:
            wp.launch(K["body_step"], dim=self.nb, device=dev,
                      inputs=[self.bpos, self.bvel, self.bmode, self.bload, self.bmass,
                              self.bdamp, self.react, self.rtorque, self.fsum, self.tsum,
                              self.nsum, self.dt])
        wp.launch(K["g2p"], dim=self.n, device=dev,
                  inputs=[self.x, self.v, self.C, self.F, self.evp, self.gv, self.P, self.dt])

    def take_wall_force(self):
        """Mean force the soil put on the bin floor + walls since the last call, N."""
        f = self.wreact.numpy()[0].astype(np.float64) / max(self._wsub * self.dt, 1e-12)
        self.wreact.zero_()
        self._wsub = 0
        return f

    @property
    def soil_weight(self):
        return self.n * self.P.pmass * 9.81

    def step(self, nsub, graph=True):
        """Advance `nsub` substeps; the first call per nsub captures a CUDA graph."""
        wp = self.wp
        self._wsub += nsub
        if not graph or not str(self.dev).startswith("cuda"):
            for _ in range(nsub):
                self._substep()
            return
        if self._graph is None or self._graph_n != nsub:
            # This frame runs eagerly (compiles the module); the capture only
            # records, so the next call is the first graph launch.
            for _ in range(nsub):
                self._substep()
            wp.synchronize_device(self.dev)
            with wp.ScopedCapture(device=self.dev) as cap:
                for _ in range(nsub):
                    self._substep()
            self._graph, self._graph_n = cap.graph, nsub
            return
        wp.capture_launch(self._graph)

    def surface_height(self, xlo, xhi, zlo, zhi):
        """Free-surface height over a patch: the top layer's mean + half a spacing."""
        x = self.x.numpy()
        m = (x[:, 0] > xlo) & (x[:, 0] < xhi) & (x[:, 2] > zlo) & (x[:, 2] < zhi)
        y = x[m, 1]
        if len(y) == 0:
            return float("nan")
        top = y[y > y.max() - 0.8 * self.pd]
        return float(top.mean() + 0.5 * self.pd)


# =============================================================================
# Phase 2: K moving-window patches in ONE simulation (one wheel per patch).
#
# One particle array (slot p belongs to patch p // cap), one grid of K blocks
# stacked along x, every kernel ONE launch per substep for all patches, one
# CUDA graph per frame. Each block follows its wheel by whole cells. Particles
# left behind by a shift are RETIRED into a far-field heightfield (atomic max
# of their tops over a small footprint); columns a shift uncovers are SEEDED
# from it. Per-patch slot ranges with a device free stack keep capacity fixed
# and need no host round trip.
#
# Per substep: p2g (reads the Kirchhoff stress cached by the previous g2p, so
# there is ONE SVD per particle per substep, not two), grid (clears the other
# half of a ping-pong velocity grid and its own mass, so no memsets), g2p
# (threads 0..K-1 also integrate the wheels' vertical DOF). Three launches.
# =============================================================================

_KP = None


def _compile_patched():
    global _KP
    if _KP is not None:
        return _KP
    import warp as wp

    @wp.struct
    class PP:
        h: float
        inv_h: float
        nx: int            # block grid nodes
        ny: int
        nz: int
        ix: int            # block interior cells (x, z)
        iz: int
        marg: int
        cap: int           # particle slots per patch
        nb: int            # number of patches == wheels
        oy: float          # grid origin y
        floor_y: float
        vol: float
        pmass: float
        mu: float
        la: float
        dp_k: float
        coh: float
        tr_apex: float
        apic: float
        grav: float
        vmax: float
        mu_floor: float
        mu_wall: float
        eps: float
        bulk: float
        cap_p0: float
        cap_lam: float
        pd: float
        hf_ox: float
        hf_oz: float
        hf_dx: float
        hf_nx: int
        hf_nz: int
        splat_r: float
        lead: float
        shift_thr: float
        seed: int
        skx: int           # device re-sort lattice (pd cells) over a block interior
        sky: int
        skz: int
        snk: int           # sort keys per patch (the last one = dead slot)
        dip_thr: float     # write-back: a vacated cell this far below its neighbours' median is a pit

    @wp.func
    def pcollide(v: wp.vec3, n: wp.vec3, vc: wp.vec3, mu: float) -> wp.vec3:
        rel = v - vc
        vn = wp.dot(rel, n)
        if vn >= 0.0:
            return v
        vt = rel - n * vn
        vtl = wp.length(vt)
        if vtl > 1.0e-7:
            vt = vt * wp.max(0.0, 1.0 + mu * vn / vtl)
        else:
            vt = wp.vec3(0.0, 0.0, 0.0)
        return vc + vt

    @wp.func
    def pspline(f: float) -> wp.vec3:
        return wp.vec3(0.5 * (1.5 - f) * (1.5 - f),
                       0.75 - (f - 1.0) * (f - 1.0),
                       0.5 * (f - 0.5) * (f - 0.5))

    @wp.func
    def blo(pc: wp.array2d(dtype=int), k: int, P: PP) -> wp.vec2:
        """Interior lower corner (x, z) of block k."""
        return wp.vec2(float(pc[k, 0] + P.marg) * P.h, float(pc[k, 1] + P.marg) * P.h)

    @wp.func
    def inside(lo: wp.vec2, x: float, z: float, P: PP) -> int:
        if x >= lo[0] and x < lo[0] + float(P.ix) * P.h and z >= lo[1] and z < lo[1] + float(P.iz) * P.h:
            return 1
        return 0

    @wp.func
    def hf_sample(H: wp.array2d(dtype=float), x: float, z: float, P: PP) -> float:
        u = (x - P.hf_ox) / P.hf_dx - 0.5
        w = (z - P.hf_oz) / P.hf_dx - 0.5
        i0 = int(wp.floor(u))
        j0 = int(wp.floor(w))
        fu = u - float(i0)
        fw = w - float(j0)
        i0c = wp.clamp(i0, 0, P.hf_nx - 1)
        i1c = wp.clamp(i0 + 1, 0, P.hf_nx - 1)
        j0c = wp.clamp(j0, 0, P.hf_nz - 1)
        j1c = wp.clamp(j0 + 1, 0, P.hf_nz - 1)
        a = H[i0c, j0c] * (1.0 - fu) + H[i1c, j0c] * fu
        b = H[i0c, j1c] * (1.0 - fu) + H[i1c, j1c] * fu
        return a * (1.0 - fw) + b * fw

    @wp.kernel
    def apply_cmd(cmd: wp.array2d(dtype=float), bpos: wp.array(dtype=wp.vec3),
                  bvel: wp.array(dtype=wp.vec3), bomega: wp.array(dtype=wp.vec3),
                  baxis: wp.array(dtype=wp.vec3), bload: wp.array(dtype=float),
                  bmass: wp.array(dtype=float), bdamp: wp.array(dtype=float),
                  bsize: wp.array(dtype=wp.vec3), bmu: wp.array(dtype=float),
                  bmode: wp.array(dtype=int)):
        """The car's per-frame command -> body state. y / vy stay device-owned
        unless the command's set_y flag is on."""
        k = wp.tid()
        p = bpos[k]
        v = bvel[k]
        y = p[1]
        vy = v[1]
        if cmd[k, 19] > 0.5:
            y = cmd[k, 1]
            vy = cmd[k, 4]
        bpos[k] = wp.vec3(cmd[k, 0], y, cmd[k, 2])
        bvel[k] = wp.vec3(cmd[k, 3], vy, cmd[k, 5])
        bomega[k] = wp.vec3(cmd[k, 6], cmd[k, 7], cmd[k, 8])
        baxis[k] = wp.vec3(cmd[k, 9], cmd[k, 10], cmd[k, 11])
        bload[k] = cmd[k, 12]
        bmass[k] = cmd[k, 13]
        bdamp[k] = cmd[k, 14]
        bsize[k] = wp.vec3(cmd[k, 15], cmd[k, 16], 0.0)
        bmu[k] = cmd[k, 17]
        bmode[k] = int(cmd[k, 18] + 0.5)

    @wp.kernel
    def window(bpos: wp.array(dtype=wp.vec3), bvel: wp.array(dtype=wp.vec3),
               pc: wp.array2d(dtype=int), pco: wp.array2d(dtype=int),
               fresh: wp.array(dtype=int), nshift: wp.array(dtype=int), P: PP):
        """Block k follows wheel k (led by half a frame of travel) in whole cells."""
        k = wp.tid()
        pco[k, 0] = pc[k, 0]
        pco[k, 1] = pc[k, 1]
        c = bpos[k] + bvel[k] * P.lead
        hx = 0.5 * float(P.ix) * P.h
        hz = 0.5 * float(P.iz) * P.h
        if fresh[k] == 1:
            pc[k, 0] = int(wp.round((c[0] - hx) * P.inv_h)) - P.marg
            pc[k, 1] = int(wp.round((c[2] - hz) * P.inv_h)) - P.marg
            return
        lo = blo(pc, k, P)
        dx = c[0] - (lo[0] + hx)
        dz = c[2] - (lo[1] + hz)
        sx = int(0)
        sz = int(0)
        if wp.abs(dx) > P.shift_thr * P.h:
            sx = int(wp.round(dx * P.inv_h))
        if wp.abs(dz) > P.shift_thr * P.h:
            sz = int(wp.round(dz * P.inv_h))
        if sx != 0 or sz != 0:
            pc[k, 0] = pc[k, 0] + sx
            pc[k, 1] = pc[k, 1] + sz
            nshift[k] = nshift[k] + 1

    @wp.kernel
    def hf_clear(H: wp.array2d(dtype=float), pc: wp.array2d(dtype=int),
                 pco: wp.array2d(dtype=int), fresh: wp.array(dtype=int), P: PP):
        """Heightfield cells the block just LEFT start from the floor; the
        retiring particles then raise them to their tops."""
        k, a, b = wp.tid()
        if fresh[k] == 1:
            return
        lo = blo(pco, k, P)
        ln = blo(pc, k, P)
        if lo[0] == ln[0] and lo[1] == ln[1]:
            return
        ci = int(wp.floor((lo[0] - P.hf_ox) / P.hf_dx)) + a
        cj = int(wp.floor((lo[1] - P.hf_oz) / P.hf_dx)) + b
        if ci < 0 or cj < 0 or ci >= P.hf_nx or cj >= P.hf_nz:
            return
        cx = P.hf_ox + (float(ci) + 0.5) * P.hf_dx
        cz = P.hf_oz + (float(cj) + 0.5) * P.hf_dx
        if inside(lo, cx, cz, P) == 1 and inside(ln, cx, cz, P) == 0:
            H[ci, cj] = P.floor_y

    @wp.kernel
    def retire(x: wp.array(dtype=wp.vec3), alive: wp.array(dtype=int),
               fstack: wp.array(dtype=int), ftop: wp.array(dtype=int),
               H: wp.array2d(dtype=float), pc: wp.array2d(dtype=int),
               pco: wp.array2d(dtype=int), fresh: wp.array(dtype=int), P: PP):
        p = wp.tid()
        if alive[p] == 0:
            return
        k = p // P.cap
        xp = x[p]
        ln = blo(pc, k, P)
        if inside(ln, xp[0], xp[2], P) == 0:
            alive[p] = 0
            idx = wp.atomic_add(ftop, k, 1)
            fstack[k * P.cap + idx] = p
        if fresh[k] == 1:
            return
        lo = blo(pco, k, P)
        if lo[0] == ln[0] and lo[1] == ln[1]:
            return
        # EVERY live particle (kept or retiring) splats its top into the cells the
        # block just vacated: a vacated cell next to the new block edge then sees
        # the top layer on both sides of the edge, not a one-sided half-disc
        # (the half-disc often held no top-layer grain: ~25 mm write-back pits).
        top = xp[1] + 0.5 * P.pd
        r = P.splat_r
        i0 = int(wp.floor((xp[0] - r - P.hf_ox) / P.hf_dx))
        j0 = int(wp.floor((xp[2] - r - P.hf_oz) / P.hf_dx))
        n = int(2.0 * r / P.hf_dx) + 2
        for a in range(n):
            for b in range(n):
                ci = i0 + a
                cj = j0 + b
                if ci >= 0 and cj >= 0 and ci < P.hf_nx and cj < P.hf_nz:
                    cx = P.hf_ox + (float(ci) + 0.5) * P.hf_dx
                    cz = P.hf_oz + (float(cj) + 0.5) * P.hf_dx
                    ddx = cx - xp[0]
                    ddz = cz - xp[2]
                    if ddx * ddx + ddz * ddz <= r * r:
                        if inside(lo, cx, cz, P) == 1 and inside(ln, cx, cz, P) == 0:
                            wp.atomic_max(H, ci, cj, top)

    @wp.kernel
    def fill_track(pc: wp.array2d(dtype=int), pco: wp.array2d(dtype=int), plast: wp.array2d(dtype=int),
                   pfill: wp.array2d(dtype=int), fresh: wp.array(dtype=int)):
        """pfill = the block corner before the PREVIOUS shift (pit repair spans two bands)."""
        k = wp.tid()
        if fresh[k] == 1:
            plast[k, 0] = pc[k, 0]
            plast[k, 1] = pc[k, 1]
            pfill[k, 0] = pc[k, 0]
            pfill[k, 1] = pc[k, 1]
            return
        if pco[k, 0] != pc[k, 0] or pco[k, 1] != pc[k, 1]:
            pfill[k, 0] = plast[k, 0]
            pfill[k, 1] = plast[k, 1]
            plast[k, 0] = pco[k, 0]
            plast[k, 1] = pco[k, 1]

    @wp.func
    def fill_nb(H: wp.array2d(dtype=float), ci: int, cj: int, t: int, ln: wp.vec2, P: PP) -> wp.vec2:
        """Neighbour t (0..8, 4 = self) of cell (ci, cj): (value, valid). Valid =
        in range and NOT under the new block (those cells hold stale pre-pass values)."""
        ni = ci + t // 3 - 1
        nj = cj + t - (t // 3) * 3 - 1
        if t == 4 or ni < 0 or nj < 0 or ni >= P.hf_nx or nj >= P.hf_nz:
            return wp.vec2(0.0, 0.0)
        cx = P.hf_ox + (float(ni) + 0.5) * P.hf_dx
        cz = P.hf_oz + (float(nj) + 0.5) * P.hf_dx
        if inside(ln, cx, cz, P) == 1:
            return wp.vec2(0.0, 0.0)
        return wp.vec2(H[ni, nj], 1.0)

    @wp.kernel
    def hf_fill(H: wp.array2d(dtype=float), S: wp.array3d(dtype=float), pc: wp.array2d(dtype=int),
                pco: wp.array2d(dtype=int), pfill: wp.array2d(dtype=int), fresh: wp.array(dtype=int), P: PP):
        """Write-back pit repair: a vacated cell more than dip_thr below the
        median of its valid neighbours (vacated this frame or far field, never
        cells under the new block) takes that median. Result staged in S."""
        k, a, b = wp.tid()
        S[k, a, b] = -1.0e9
        if fresh[k] == 1:
            return
        lp = blo(pco, k, P)
        ln = blo(pc, k, P)
        if lp[0] == ln[0] and lp[1] == ln[1]:
            return
        # The bands vacated by this shift AND the previous one: the previous
        # band now has all 8 neighbours written, so a dip there (e.g. a rut-wall
        # staircase notch) is judged against its full neighbourhood.
        lo = blo(pfill, k, P)
        ci = int(wp.floor((lo[0] - P.hf_ox) / P.hf_dx)) + a
        cj = int(wp.floor((lo[1] - P.hf_oz) / P.hf_dx)) + b
        if ci < 0 or cj < 0 or ci >= P.hf_nx or cj >= P.hf_nz:
            return
        cx = P.hf_ox + (float(ci) + 0.5) * P.hf_dx
        cz = P.hf_oz + (float(cj) + 0.5) * P.hf_dx
        if inside(lo, cx, cz, P) == 0 or inside(ln, cx, cz, P) == 1:
            return
        n = int(0)
        for t in range(9):
            q = fill_nb(H, ci, cj, t, ln, P)
            if q[1] > 0.5:
                n += 1
        if n < 3:
            return
        # The true median (mean of the two middle ranks for an even count): at
        # a rut-wall foot with 4 neighbours on the wall and 4 on the floor it is
        # the wall's midpoint, so a one-cell notch into the wall (the particle
        # lattice aliasing against the heightfield cells) becomes a ramp.
        r_lo = (n - 1) // 2
        r_hi = n // 2
        m_lo = H[ci, cj]
        m_hi = H[ci, cj]
        for t in range(9):
            q = fill_nb(H, ci, cj, t, ln, P)
            if q[1] > 0.5:
                cl = int(0)
                ce = int(0)
                for s in range(9):
                    r = fill_nb(H, ci, cj, s, ln, P)
                    if r[1] > 0.5:
                        if r[0] < q[0]:
                            cl += 1
                        elif r[0] == q[0]:
                            ce += 1
                if cl <= r_lo and r_lo < cl + ce:
                    m_lo = q[0]
                if cl <= r_hi and r_hi < cl + ce:
                    m_hi = q[0]
        med = 0.5 * (m_lo + m_hi)
        if H[ci, cj] < med - P.dip_thr:
            S[k, a, b] = med

    @wp.kernel
    def hf_fill_apply(H: wp.array2d(dtype=float), S: wp.array3d(dtype=float), pfill: wp.array2d(dtype=int),
                      P: PP):
        k, a, b = wp.tid()
        val = S[k, a, b]
        if val < -1.0e8:
            return
        lo = blo(pfill, k, P)
        ci = int(wp.floor((lo[0] - P.hf_ox) / P.hf_dx)) + a
        cj = int(wp.floor((lo[1] - P.hf_oz) / P.hf_dx)) + b
        H[ci, cj] = val

    @wp.kernel
    def sort_key(x: wp.array(dtype=wp.vec3), alive: wp.array(dtype=int), pc: wp.array2d(dtype=int),
                 keys: wp.array(dtype=int), vals: wp.array(dtype=int), P: PP):
        """Slab-order key (y layer, then x, then z on the pd lattice of the block
        interior), offset by patch so a global sort keeps every patch in its own
        slot range; dead slots sort to the end of their patch."""
        p = wp.tid()
        k = p // P.cap
        key = P.snk - 1
        if alive[p] == 1:
            xp = x[p]
            lo = blo(pc, k, P)
            kx = wp.clamp(int(wp.floor((xp[0] - lo[0]) / P.pd)), 0, P.skx - 1)
            ky = wp.clamp(int(wp.floor((xp[1] - P.floor_y) / P.pd)), 0, P.sky - 1)
            kz = wp.clamp(int(wp.floor((xp[2] - lo[1]) / P.pd)), 0, P.skz - 1)
            key = (ky * P.skx + kx) * P.skz + kz
        keys[p] = k * P.snk + key
        vals[p] = p

    @wp.kernel
    def gather(perm: wp.array(dtype=int),
               x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33),
               F: wp.array(dtype=wp.mat33), tau: wp.array(dtype=wp.mat33), evp: wp.array(dtype=float),
               alive: wp.array(dtype=int),
               x2: wp.array(dtype=wp.vec3), v2: wp.array(dtype=wp.vec3), C2: wp.array(dtype=wp.mat33),
               F2: wp.array(dtype=wp.mat33), tau2: wp.array(dtype=wp.mat33), evp2: wp.array(dtype=float),
               alive2: wp.array(dtype=int)):
        q = wp.tid()
        p = perm[q]
        x2[q] = x[p]
        v2[q] = v[p]
        C2[q] = C[p]
        F2[q] = F[p]
        tau2[q] = tau[p]
        evp2[q] = evp[p]
        alive2[q] = alive[p]

    @wp.kernel
    def rebuild_free(ftop: wp.array(dtype=int), fstack: wp.array(dtype=int), P: PP):
        """After the sort patch k's live particles fill its first cap - ftop[k]
        slots; the free stack is the rest."""
        q = wp.tid()
        k = q // P.cap
        i = q - k * P.cap
        na = P.cap - ftop[k]
        if i >= na:
            fstack[k * P.cap + i - na] = q

    @wp.kernel
    def seed(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
             C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
             tau: wp.array(dtype=wp.mat33), evp: wp.array(dtype=float),
             alive: wp.array(dtype=int), fstack: wp.array(dtype=int),
             ftop: wp.array(dtype=int), ovf: wp.array(dtype=int),
             H: wp.array2d(dtype=float), pc: wp.array2d(dtype=int),
             pco: wp.array2d(dtype=int), fresh: wp.array(dtype=int), P: PP):
        """Columns the block just COVERED: a lattice from the floor up to h(x,z)."""
        k, a, b = wp.tid()
        ln = blo(pc, k, P)
        lo = blo(pco, k, P)
        ci = int(wp.floor(ln[0] / P.pd)) + a
        cj = int(wp.floor(ln[1] / P.pd)) + b
        cx = (float(ci) + 0.5) * P.pd
        cz = (float(cj) + 0.5) * P.pd
        if inside(ln, cx, cz, P) == 0:
            return
        if fresh[k] == 0 and inside(lo, cx, cz, P) == 1:
            return
        depth = hf_sample(H, cx, cz, P) - P.floor_y
        n = int(depth / P.pd + 0.5)
        if n <= 0:
            return
        sy = depth / float(n)
        rng = wp.rand_init(P.seed, ci * 7919 + cj * 104729)
        pad = 0.05 * P.h
        hx = ln[0] + float(P.ix) * P.h - pad
        hz = ln[1] + float(P.iz) * P.h - pad
        for j in range(n):
            idx = wp.atomic_sub(ftop, k, 1) - 1
            if idx < 0:
                wp.atomic_add(ftop, k, 1)
                wp.atomic_add(ovf, 0, 1)
                return
            p = fstack[k * P.cap + idx]
            jx = wp.randf(rng, -0.2, 0.2) * P.pd
            jz = wp.randf(rng, -0.2, 0.2) * P.pd
            jy = wp.randf(rng, -0.05, 0.05) * sy
            x[p] = wp.vec3(wp.clamp(cx + jx, ln[0] + pad, hx),
                           P.floor_y + (float(j) + 0.5) * sy + jy,
                           wp.clamp(cz + jz, ln[1] + pad, hz))
            v[p] = wp.vec3(0.0, 0.0, 0.0)
            C[p] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            F[p] = wp.identity(n=3, dtype=float)
            tau[p] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            evp[p] = 0.0
            alive[p] = 1

    @wp.kernel
    def p2g(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
            C: wp.array(dtype=wp.mat33), tau: wp.array(dtype=wp.mat33),
            alive: wp.array(dtype=int), pc: wp.array2d(dtype=int),
            gm: wp.array3d(dtype=float), gv: wp.array3d(dtype=wp.vec3),
            P: PP, dt: float):
        p = wp.tid()
        if alive[p] == 0:
            return
        k = p // P.cap
        xp = x[p]
        gx = (xp[0] - float(pc[k, 0]) * P.h) * P.inv_h
        gy = (xp[1] - P.oy) * P.inv_h
        gz = (xp[2] - float(pc[k, 1]) * P.h) * P.inv_h
        bi = int(wp.floor(gx - 0.5))
        bj = int(wp.floor(gy - 0.5))
        bk = int(wp.floor(gz - 0.5))
        if bi < 0 or bj < 0 or bk < 0 or bi > P.nx - 3 or bj > P.ny - 3 or bk > P.nz - 3:
            return
        fx = wp.vec3(gx - float(bi), gy - float(bj), gz - float(bk))
        wx = pspline(fx[0])
        wy = pspline(fx[1])
        wz = pspline(fx[2])
        affine = tau[p] * (-dt * P.vol * 4.0 * P.inv_h * P.inv_h) + C[p] * P.pmass
        mv = v[p] * P.pmass
        gi = k * P.nx + bi
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    w = wx[a] * wy[b] * wz[c]
                    dpos = wp.vec3((float(a) - fx[0]) * P.h,
                                   (float(b) - fx[1]) * P.h,
                                   (float(c) - fx[2]) * P.h)
                    wp.atomic_add(gm, gi + a, bj + b, bk + c, w * P.pmass)
                    wp.atomic_add(gv, gi + a, bj + b, bk + c, (mv + affine * dpos) * w)

    @wp.kernel
    def p2g_tiled(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                  C: wp.array(dtype=wp.mat33), tau: wp.array(dtype=wp.mat33),
                  alive: wp.array(dtype=int), pc: wp.array2d(dtype=int),
                  gm: wp.array3d(dtype=float), gv: wp.array3d(dtype=wp.vec3),
                  P: PP, dt: float, nslots: int):
        """EXPERIMENTAL block-local P2G: a thread block of 128 consecutive slots
        scatters into a shared-memory node tile over its bounding box, then
        flushes one global atomic per touched node. Falls back to per-particle
        global atomics when the box exceeds the tile (unsorted slots)."""
        blk, lane = wp.tid()
        t = wp.tile_zeros(shape=1024, dtype=wp.vec4, storage="shared")
        p = blk * 128 + lane
        ok = int(0)
        gi = int(0)
        bj = int(0)
        bk = int(0)
        fx = wp.vec3(0.0, 0.0, 0.0)
        mv = wp.vec3(0.0, 0.0, 0.0)
        affine = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if p < nslots:
            if alive[p] == 1:
                k = p // P.cap
                xp = x[p]
                gx = (xp[0] - float(pc[k, 0]) * P.h) * P.inv_h
                gy = (xp[1] - P.oy) * P.inv_h
                gz = (xp[2] - float(pc[k, 1]) * P.h) * P.inv_h
                bi = int(wp.floor(gx - 0.5))
                bj = int(wp.floor(gy - 0.5))
                bk = int(wp.floor(gz - 0.5))
                if bi >= 0 and bj >= 0 and bk >= 0 and bi <= P.nx - 3 and bj <= P.ny - 3 and bk <= P.nz - 3:
                    ok = 1
                    fx = wp.vec3(gx - float(bi), gy - float(bj), gz - float(bk))
                    gi = k * P.nx + bi
                    affine = tau[p] * (-dt * P.vol * 4.0 * P.inv_h * P.inv_h) + C[p] * P.pmass
                    mv = v[p] * P.pmass
        big = 1.0e9
        a0 = big
        a1 = -big
        b0 = big
        b1 = -big
        c0 = big
        c1 = -big
        if ok == 1:
            a0 = float(gi)
            a1 = float(gi)
            b0 = float(bj)
            b1 = float(bj)
            c0 = float(bk)
            c1 = float(bk)
        i0f = wp.tile_extract(wp.tile_min(wp.tile(a0)), 0)
        i1f = wp.tile_extract(wp.tile_max(wp.tile(a1)), 0)
        j0f = wp.tile_extract(wp.tile_min(wp.tile(b0)), 0)
        j1f = wp.tile_extract(wp.tile_max(wp.tile(b1)), 0)
        k0f = wp.tile_extract(wp.tile_min(wp.tile(c0)), 0)
        k1f = wp.tile_extract(wp.tile_max(wp.tile(c1)), 0)
        if i0f > i1f:
            return
        i0 = int(i0f)
        j0 = int(j0f)
        k0 = int(k0f)
        ex = int(i1f) - i0 + 3
        ey = int(j1f) - j0 + 3
        ez = int(k1f) - k0 + 3
        wx = pspline(fx[0])
        wy = pspline(fx[1])
        wz = pspline(fx[2])
        if ex * ey * ez <= 1024:
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        w = wx[a] * wy[b] * wz[c]
                        dpos = wp.vec3((float(a) - fx[0]) * P.h, (float(b) - fx[1]) * P.h,
                                       (float(c) - fx[2]) * P.h)
                        val = (mv + affine * dpos) * w
                        idx = ((gi - i0 + a) * ey + (bj - j0 + b)) * ez + (bk - k0 + c)
                        if ok == 0:
                            idx = 0
                        wp.tile_scatter_add(t, idx, wp.vec4(val[0], val[1], val[2], w * P.pmass), ok == 1)
            n = lane
            tot = ex * ey * ez
            while n < tot:
                q = wp.tile_extract(t, n)
                if q[3] > 0.0:
                    li = n // (ey * ez)
                    r = n - li * ey * ez
                    lj = r // ez
                    lk = r - lj * ez
                    wp.atomic_add(gm, i0 + li, j0 + lj, k0 + lk, q[3])
                    wp.atomic_add(gv, i0 + li, j0 + lj, k0 + lk, wp.vec3(q[0], q[1], q[2]))
                n += 128
        else:
            if ok == 1:
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                            w = wx[a] * wy[b] * wz[c]
                            dpos = wp.vec3((float(a) - fx[0]) * P.h, (float(b) - fx[1]) * P.h,
                                           (float(c) - fx[2]) * P.h)
                            wp.atomic_add(gm, gi + a, bj + b, bk + c, w * P.pmass)
                            wp.atomic_add(gv, gi + a, bj + b, bk + c, (mv + affine * dpos) * w)

    @wp.kernel
    def grid(gm: wp.array3d(dtype=float), gv: wp.array3d(dtype=wp.vec3),
             gv_next: wp.array3d(dtype=wp.vec3), pc: wp.array2d(dtype=int), P: PP,
             bpos: wp.array(dtype=wp.vec3), bvel: wp.array(dtype=wp.vec3),
             bomega: wp.array(dtype=wp.vec3), baxis: wp.array(dtype=wp.vec3),
             bsize: wp.array(dtype=wp.vec3), bmu: wp.array(dtype=float),
             react: wp.array(dtype=wp.vec3), rtorque: wp.array(dtype=wp.vec3), dt: float):
        i, j, kk = wp.tid()
        q = i // P.nx
        li = i - q * P.nx
        zero = wp.vec3(0.0, 0.0, 0.0)
        gv_next[i, j, kk] = zero
        m = gm[i, j, kk]
        gm[i, j, kk] = 0.0
        if m <= 1.0e-12:
            gv[i, j, kk] = zero
            return
        v = gv[i, j, kk] * (1.0 / m) + wp.vec3(0.0, P.grav * dt, 0.0)
        ox = float(pc[q, 0]) * P.h
        oz = float(pc[q, 1]) * P.h
        p = wp.vec3(ox + float(li) * P.h, P.oy + float(j) * P.h, oz + float(kk) * P.h)
        # This block's own wheel (a cylinder of radius sz[0], half-width sz[1]).
        d = p - bpos[q]
        sz = bsize[q]
        ax = baxis[q]
        aa = wp.dot(d, ax)
        rad = d - ax * aa
        rl = wp.length(rad)
        if wp.abs(aa) <= sz[1] and rl < sz[0] + P.eps:
            n = rad * (1.0 / wp.max(rl, 1.0e-9))
            lever = n * sz[0]
            before = v
            v = pcollide(v, n, bvel[q] + wp.cross(bomega[q], lever), bmu[q])
            dp = (before - v) * m
            wp.atomic_add(react, q, dp)
            wp.atomic_add(rtorque, q, wp.cross(lever, dp))
        lox = ox + float(P.marg) * P.h
        loz = oz + float(P.marg) * P.h
        hix = lox + float(P.ix) * P.h
        hiz = loz + float(P.iz) * P.h
        if p[1] <= P.floor_y:
            v = pcollide(v, wp.vec3(0.0, 1.0, 0.0), zero, P.mu_floor)
        if p[0] <= lox:
            v = pcollide(v, wp.vec3(1.0, 0.0, 0.0), zero, P.mu_wall)
        if p[0] >= hix:
            v = pcollide(v, wp.vec3(-1.0, 0.0, 0.0), zero, P.mu_wall)
        if p[2] <= loz:
            v = pcollide(v, wp.vec3(0.0, 0.0, 1.0), zero, P.mu_wall)
        if p[2] >= hiz:
            v = pcollide(v, wp.vec3(0.0, 0.0, -1.0), zero, P.mu_wall)
        sp = wp.length(v)
        if sp > P.vmax:
            v = v * (P.vmax / sp)
        gv[i, j, kk] = v

    @wp.kernel
    def g2p(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
            C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
            tau: wp.array(dtype=wp.mat33), evp: wp.array(dtype=float),
            alive: wp.array(dtype=int), pc: wp.array2d(dtype=int),
            gv: wp.array3d(dtype=wp.vec3), P: PP, dt: float,
            bpos: wp.array(dtype=wp.vec3), bvel: wp.array(dtype=wp.vec3),
            mode: wp.array(dtype=int), load: wp.array(dtype=float),
            mass: wp.array(dtype=float), damp: wp.array(dtype=float),
            react: wp.array(dtype=wp.vec3), rtorque: wp.array(dtype=wp.vec3),
            fsum: wp.array(dtype=wp.vec3), tsum: wp.array(dtype=wp.vec3),
            nsum: wp.array(dtype=float)):
        p = wp.tid()
        if p < P.nb:
            # The wheel's vertical DOF (the grid kernel's reaction is complete).
            f = react[p] * (1.0 / dt)
            t = rtorque[p] * (1.0 / dt)
            vel = bvel[p]
            if mode[p] == 1:
                vy = vel[1] + dt * (f[1] - load[p] - damp[p] * vel[1]) / mass[p]
                vel = wp.vec3(vel[0], vy, vel[2])
                bvel[p] = vel
            bpos[p] = bpos[p] + vel * dt
            fsum[p] = fsum[p] + f
            tsum[p] = tsum[p] + t
            nsum[p] = nsum[p] + 1.0
            react[p] = wp.vec3(0.0, 0.0, 0.0)
            rtorque[p] = wp.vec3(0.0, 0.0, 0.0)
        if alive[p] == 0:
            return
        k = p // P.cap
        xp = x[p]
        ox = float(pc[k, 0]) * P.h
        oz = float(pc[k, 1]) * P.h
        gx = (xp[0] - ox) * P.inv_h
        gy = (xp[1] - P.oy) * P.inv_h
        gz = (xp[2] - oz) * P.inv_h
        bi = int(wp.floor(gx - 0.5))
        bj = int(wp.floor(gy - 0.5))
        bk = int(wp.floor(gz - 0.5))
        if bi < 0 or bj < 0 or bk < 0 or bi > P.nx - 3 or bj > P.ny - 3 or bk > P.nz - 3:
            v[p] = wp.vec3(0.0, 0.0, 0.0)
            return
        fx = wp.vec3(gx - float(bi), gy - float(bj), gz - float(bk))
        wx = pspline(fx[0])
        wy = pspline(fx[1])
        wz = pspline(fx[2])
        nv = wp.vec3(0.0, 0.0, 0.0)
        nc = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        gi = k * P.nx + bi
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    w = wx[a] * wy[b] * wz[c]
                    g = gv[gi + a, bj + b, bk + c]
                    nv += g * w
                    nc += wp.outer(g, wp.vec3(float(a) - fx[0], float(b) - fx[1],
                                              float(c) - fx[2])) * (4.0 * P.inv_h * w)
        nc = nc * P.apic
        Fn = (wp.identity(n=3, dtype=float) + nc * dt) * F[p]
        U = wp.mat33()
        V = wp.mat33()
        sig = wp.vec3()
        wp.svd3(Fn, U, sig, V)
        e0 = wp.log(wp.min(wp.max(sig[0], 0.05), 4.0))
        e1 = wp.log(wp.min(wp.max(sig[1], 0.05), 4.0))
        e2 = wp.log(wp.min(wp.max(sig[2], 0.05), 4.0))
        tr = e0 + e1 + e2
        if P.cap_lam > 0.0:
            pcap = P.cap_p0 * wp.exp(-evp[p] / P.cap_lam)
            trc = -pcap / P.bulk
            if tr < trc:
                dv = tr - trc
                evp[p] = evp[p] + dv
                e0 = e0 - dv / 3.0
                e1 = e1 - dv / 3.0
                e2 = e2 - dv / 3.0
                tr = trc
        if tr >= P.tr_apex:
            e0 = P.tr_apex / 3.0
            e1 = e0
            e2 = e0
        else:
            h0 = e0 - tr / 3.0
            h1 = e1 - tr / 3.0
            h2 = e2 - tr / 3.0
            fn = wp.sqrt(h0 * h0 + h1 * h1 + h2 * h2)
            dg = fn - P.coh + P.dp_k * tr
            if dg > 0.0 and fn > 1.0e-9:
                sc = dg / fn
                e0 = e0 - h0 * sc
                e1 = e1 - h1 * sc
                e2 = e2 - h2 * sc
        F[p] = U * wp.diag(wp.vec3(wp.exp(e0), wp.exp(e1), wp.exp(e2))) * wp.transpose(V)
        trn = e0 + e1 + e2
        # Cache the Kirchhoff stress for the next p2g: one SVD per substep.
        tau[p] = U * wp.diag(wp.vec3(2.0 * P.mu * e0 + P.la * trn,
                                     2.0 * P.mu * e1 + P.la * trn,
                                     2.0 * P.mu * e2 + P.la * trn)) * wp.transpose(U)
        C[p] = nc
        v[p] = nv
        q = xp + nv * dt
        pad = 0.05 * P.h
        lox = ox + float(P.marg) * P.h
        loz = oz + float(P.marg) * P.h
        x[p] = wp.vec3(wp.min(wp.max(q[0], lox + pad), lox + float(P.ix) * P.h - pad),
                       wp.min(wp.max(q[1], P.floor_y + pad), P.oy + float(P.ny - 3) * P.h),
                       wp.min(wp.max(q[2], loz + pad), loz + float(P.iz) * P.h - pad))

    @wp.kernel
    def finalize(bpos: wp.array(dtype=wp.vec3), bvel: wp.array(dtype=wp.vec3),
                 baxis: wp.array(dtype=wp.vec3), bsize: wp.array(dtype=wp.vec3),
                 fsum: wp.array(dtype=wp.vec3), tsum: wp.array(dtype=wp.vec3),
                 nsum: wp.array(dtype=float), ftop: wp.array(dtype=int),
                 ovf: wp.array(dtype=int), pc: wp.array2d(dtype=int),
                 nshift: wp.array(dtype=int), fresh: wp.array(dtype=int),
                 out: wp.array2d(dtype=float), P: PP):
        """Per-wheel frame means + state into one small array (ONE D->H copy)."""
        k = wp.tid()
        n = wp.max(nsum[k], 1.0)
        f = fsum[k] / n
        t = tsum[k] / n
        out[k, 0] = f[0]
        out[k, 1] = f[1]
        out[k, 2] = f[2]
        out[k, 3] = t[0]
        out[k, 4] = t[1]
        out[k, 5] = t[2]
        out[k, 6] = wp.dot(t, baxis[k])
        out[k, 7] = bpos[k][1] - bsize[k][0] - P.eps
        out[k, 8] = bpos[k][0]
        out[k, 9] = bpos[k][1]
        out[k, 10] = bpos[k][2]
        out[k, 11] = bvel[k][1]
        out[k, 12] = float(P.cap - ftop[k])
        out[k, 13] = float(ovf[0])
        out[k, 14] = float(pc[k, 0])
        out[k, 15] = float(pc[k, 1])
        fsum[k] = wp.vec3(0.0, 0.0, 0.0)
        tsum[k] = wp.vec3(0.0, 0.0, 0.0)
        nsum[k] = 0.0
        fresh[k] = 0

    _KP = dict(wp=wp, PP=PP, apply_cmd=apply_cmd, window=window, hf_clear=hf_clear,
               retire=retire, seed=seed, p2g=p2g, p2g_tiled=p2g_tiled, grid=grid, g2p=g2p,
               finalize=finalize, fill_track=fill_track, hf_fill=hf_fill, hf_fill_apply=hf_fill_apply, sort_key=sort_key,
               gather=gather, rebuild_free=rebuild_free)
    return _KP


# Readback columns of GranularPatches.step_frame()
OUT_F = slice(0, 3)          # mean soil force on the wheel over the frame, N (world)
OUT_T = slice(3, 6)          # mean soil moment about the hub, N m (world)
OUT_T_AXLE = 6               # its component along the wheel axis, N m
OUT_Y_BOTTOM = 7             # rim bottom (hub y - r - skin): the road-override height, m
OUT_POS = slice(8, 11)       # hub position at the end of the frame
OUT_VY = 11
OUT_ALIVE = 12               # live particles in the patch
OUT_OVERFLOW = 13            # seeding overflows so far (all patches)
OUT_BLOCK = slice(14, 16)    # block origin in cells (x, z)
NCMD = 20

_SORT_RESERVED = {}


def _reserve_sort_temp(wp, dev, n):
    """Warp's radix sort keeps ONE temp buffer per stream and re-allocates it when
    a bigger sort comes along -- which would leave an earlier captured graph
    pointing at freed memory. Grow it once, generously, before any capture."""
    want = max(int(n), 1 << 19)
    if _SORT_RESERVED.get(str(dev), 0) >= want:
        return
    k = wp.zeros(2 * want, dtype=int, device=dev)
    v = wp.zeros(2 * want, dtype=int, device=dev)
    wp.utils.radix_sort_pairs(k, v, want)
    wp.synchronize_device(dev)
    _SORT_RESERVED[str(dev)] = want


class GranularPatches:
    """K moving-window MLS-MPM soil patches, one wheel each, in ONE simulation.

    Every block is `patch_size` (x, z) metres of soil interior, `depth` deep
    over a flat floor at `floor_y`, with a 3-cell grid margin and `headroom`
    of air. The far field is a heightfield `hf` over `hf_origin`..+`hf_size`
    (x, z) at `hf_dx`: where a block goes it is seeded from it, where it
    leaves the grains are written back into it.

    Per frame (the car interface): set_wheels(...) then step_frame(nsub) -> a
    (K, 16) float32 array (see OUT_*), one graph launch + one small readback.

    Gate iteration 2 (2026-09-22) -- API unchanged, new keyword arguments with
    the SHIPPED configuration as their defaults (pass the old values to get
    iteration 1's behaviour back):

      patch_size=(0.8, 0.44)  was (1.0, 0.52)
      cfl=0.5                 nsub_for()'s default CFL (was a fixed 0.35);
                              nsub_for(speed, cfl=0.35) still works
      sort_every=1            device re-sort of every patch's slots into slab
                              order (radix sort + gather INSIDE the frame graph),
                              every N frames; 0 = off. Every frame is fastest.
      tiled_p2g=True          shared-memory P2G (the attribute still exists)
      fill_dips=True, dip_thr=0.006, splat_r_pd=1.0
                              write-back without pits: every live particle
                              (kept or retiring) splats its top into the vacated
                              cells, and a vacated cell more than dip_thr below
                              its valid neighbours' median (the true median:
                              mean of the two middle ranks) takes that median,
                              judged over the last two shifts' bands, twice per
                              frame (attribute fill_iters = 2). This also turns
                              the one-cell notches a steep rut wall gets from
                              the particle lattice aliasing against the 20 mm
                              heightfield into ramps.

    Recommended for a car (NOT a default, set_wheels' default is still W/g,
    undamped): set_wheels(..., mass=45.0, damp=5000.0) -- the unsprung mass and
    a near-critical damper; at 8 m/s it cuts the frame-to-frame Fy/W spread
    from 0.33 to 0.01 (std).
    """

    def __init__(self, material, h, n_patches, patch_size=(0.8, 0.44), depth=0.3,
                 hf_origin=(-2.0, -2.0), hf_size=(40.0, 4.0), hf_dx=None, floor_y=0.0,
                 ppc_spacing=0.63, headroom=0.2, mu_floor=0.6, mu_wall=0.3, eps_cells=0.25,
                 vmax=40.0, cap_factor=1.35, lead=1.0 / 120.0, shift_thr=1.0, seed=3,
                 device="cuda:0", hf_init=None, sort_every=1, splat_r_pd=1.0, fill_dips=True,
                 dip_thr=0.006, cfl=0.5, tiled_p2g=True):
        K = _compile_patched()
        wp = K["wp"]
        self.wp, self.K, self.dev = wp, K, device
        self.mat, self.h = material, float(h)
        self.nb = int(n_patches)
        self.pd = ppc_spacing * self.h
        self.floor_y, self.depth = float(floor_y), float(depth)
        ix = int(round(patch_size[0] / self.h))
        iz = int(round(patch_size[1] / self.h))
        marg = 3
        nx, nz = ix + 2 * marg + 1, iz + 2 * marg + 1
        oy = self.floor_y - marg * self.h
        ny = int(math.ceil((self.floor_y + depth + headroom - oy) / self.h)) + 1
        self.ix, self.iz, self.marg, self.dims = ix, iz, marg, (nx, ny, nz)
        nominal = int(math.ceil(ix * self.h / self.pd) * math.ceil(iz * self.h / self.pd)
                      * math.ceil(depth / self.pd))
        self.cap = int(cap_factor * nominal) + 64
        self.n_slots = self.cap * self.nb
        self.hf_dx = float(hf_dx if hf_dx is not None else 0.5 * self.h)
        hnx = int(math.ceil(hf_size[0] / self.hf_dx))
        hnz = int(math.ceil(hf_size[1] / self.hf_dx))
        self.hf_origin = (float(hf_origin[0]), float(hf_origin[1]))

        m = material
        P = K["PP"]()
        P.h, P.inv_h = self.h, 1.0 / self.h
        P.nx, P.ny, P.nz = nx, ny, nz
        P.ix, P.iz, P.marg = ix, iz, marg
        P.cap, P.nb = self.cap, self.nb
        P.oy, P.floor_y = oy, self.floor_y
        P.vol = self.pd ** 3
        P.pmass = m.rho * self.pd ** 3
        P.mu, P.la = m.mu, m.la
        sphi = math.sin(math.radians(m.phi_deg))
        alpha = math.sqrt(2.0 / 3.0) * 2.0 * sphi / (3.0 - sphi)
        P.dp_k = alpha * (3.0 * m.la + 2.0 * m.mu) / (2.0 * m.mu)
        B = 6.0 * m.cohesion * math.cos(math.radians(m.phi_deg)) / (math.sqrt(3.0) * (3.0 - sphi))
        P.coh = math.sqrt(2.0) * B / (2.0 * m.mu)
        P.tr_apex = P.coh / P.dp_k if P.dp_k > 0 else 0.0
        P.apic, P.grav, P.vmax = m.apic, -9.81, vmax
        P.mu_floor, P.mu_wall = mu_floor, mu_wall
        P.eps = eps_cells * self.h
        P.bulk = m.la + 2.0 * m.mu / 3.0
        P.cap_p0, P.cap_lam = m.cap_p0, m.cap_lambda
        P.pd = self.pd
        P.hf_ox, P.hf_oz, P.hf_dx = self.hf_origin[0], self.hf_origin[1], self.hf_dx
        P.hf_nx, P.hf_nz = hnx, hnz
        P.splat_r = float(splat_r_pd) * self.pd
        P.lead, P.shift_thr, P.seed = lead, shift_thr, seed
        P.skx = int(math.ceil(ix * self.h / self.pd)) + 1
        P.skz = int(math.ceil(iz * self.h / self.pd)) + 1
        P.sky = int(math.ceil((depth + headroom) / self.pd)) + 1
        P.snk = P.skx * P.sky * P.skz + 1
        P.dip_thr = float(dip_thr)
        self.P, self.eps = P, P.eps
        self.sort_every = int(sort_every)   # device re-sort of the slots every N frames (0 = never)
        self.fill_dips = bool(fill_dips)    # write-back pit repair
        self.fill_iters = 2                 # repair passes per frame (a repaired cell changes its neighbours' median)
        self.cfl = float(cfl)               # nsub_for() default
        self._frame_i = 0
        self._sort_bits = max(1, int(math.ceil(math.log2(self.nb * P.snk + 1))))

        dev = device
        n = self.n_slots
        self.x = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.v = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.C = wp.zeros(n, dtype=wp.mat33, device=dev)
        self.F = wp.array(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), dtype=wp.mat33, device=dev)
        self.tau = wp.zeros(n, dtype=wp.mat33, device=dev)
        self.evp = wp.zeros(n, dtype=float, device=dev)
        self.alive = wp.zeros(n, dtype=int, device=dev)
        self.fstack = wp.array(np.arange(n, dtype=np.int32), dtype=int, device=dev)
        self.ftop = wp.array(np.full(self.nb, self.cap, np.int32), dtype=int, device=dev)
        self.ovf = wp.zeros(1, dtype=int, device=dev)
        gd = (self.nb * nx, ny, nz)
        self.gm = wp.zeros(gd, dtype=float, device=dev)
        self.gv = [wp.zeros(gd, dtype=wp.vec3, device=dev), wp.zeros(gd, dtype=wp.vec3, device=dev)]
        hf0 = np.full((hnx, hnz), self.floor_y + depth, np.float32) if hf_init is None \
            else np.ascontiguousarray(hf_init, np.float32)
        self.hf = wp.array(hf0, dtype=float, device=dev)
        self.pc = wp.zeros((self.nb, 2), dtype=int, device=dev)
        self.pco = wp.zeros((self.nb, 2), dtype=int, device=dev)
        self.plast = wp.zeros((self.nb, 2), dtype=int, device=dev)
        self.pfill = wp.zeros((self.nb, 2), dtype=int, device=dev)
        self.fresh = wp.ones(self.nb, dtype=int, device=dev)
        self.nshift = wp.zeros(self.nb, dtype=int, device=dev)
        K3 = self.nb
        z3 = lambda: wp.zeros(K3, dtype=wp.vec3, device=dev)  # noqa: E731
        self.bpos, self.bvel, self.bomega, self.baxis, self.bsize = z3(), z3(), z3(), z3(), z3()
        self.bload = wp.zeros(K3, dtype=float, device=dev)
        self.bmass = wp.ones(K3, dtype=float, device=dev)
        self.bdamp = wp.zeros(K3, dtype=float, device=dev)
        self.bmu = wp.zeros(K3, dtype=float, device=dev)
        self.bmode = wp.zeros(K3, dtype=int, device=dev)
        self.react, self.rtorque, self.fsum, self.tsum = z3(), z3(), z3(), z3()
        self.nsum = wp.zeros(K3, dtype=float, device=dev)
        pinned = str(dev).startswith("cuda")
        self.cmd_h = wp.zeros((K3, NCMD), dtype=float, device="cpu", pinned=pinned)
        self.cmd_d = wp.zeros((K3, NCMD), dtype=float, device=dev)
        self.out_d = wp.zeros((K3, 16), dtype=float, device=dev)
        self.out_h = wp.zeros((K3, 16), dtype=float, device="cpu", pinned=pinned)
        self.cmd = self.cmd_h.numpy()          # a view: write the command here
        self.cmd[:, 11] = 1.0                  # axis +z (placeholder until set_wheels)
        self.cmd[:, 13] = 1.0                  # mass
        self._graphs = {}
        self.tiled_p2g = bool(tiled_p2g)   # shared-memory P2G (pays off on sorted slots)
        # Device re-sort scratch (radix sort needs 2x storage) + gather targets.
        self.skeys = wp.zeros(2 * n, dtype=int, device=dev)
        self.svals = wp.zeros(2 * n, dtype=int, device=dev)
        self._x2 = wp.zeros(n, dtype=wp.vec3, device=dev)
        self._v2 = wp.zeros(n, dtype=wp.vec3, device=dev)
        self._C2 = wp.zeros(n, dtype=wp.mat33, device=dev)
        self._F2 = wp.zeros(n, dtype=wp.mat33, device=dev)
        self._tau2 = wp.zeros(n, dtype=wp.mat33, device=dev)
        self._evp2 = wp.zeros(n, dtype=float, device=dev)
        self._alive2 = wp.zeros(n, dtype=int, device=dev)
        ncx = int(ix * self.h / self.hf_dx) + 3
        ncz = int(iz * self.h / self.hf_dx) + 3
        self._hfS = wp.zeros((self.nb, ncx, ncz), dtype=float, device=dev)
        _reserve_sort_temp(wp, dev, n)

    # -- the car interface ---------------------------------------------------------

    def set_wheels(self, pos, vel, omega, axis, load, radius, half_width, mu=0.6,
                   mass=None, damp=0.0, free_y=True, set_y=False):
        """Per-wheel hub command for the next frame. pos/vel/omega/axis: (K, 3)
        world; the hub's y and vy are the grains' unless set_y. load: (K,) N,
        pushing the wheel DOWN (suspension force + unsprung weight). mass: the
        vertical DOF's inertia (default W/g; for a car pass the unsprung mass,
        e.g. 45 kg, with damp ~5000 N s/m)."""
        c = self.cmd
        pos, vel = np.asarray(pos, np.float32), np.asarray(vel, np.float32)
        c[:, 0:3] = pos
        c[:, 3:6] = vel
        c[:, 6:9] = np.asarray(omega, np.float32)
        c[:, 9:12] = np.asarray(axis, np.float32)
        ld = np.broadcast_to(np.asarray(load, np.float32), (self.nb,))
        c[:, 12] = ld
        c[:, 13] = np.maximum(ld / 9.81, 1.0) if mass is None else mass
        c[:, 14] = damp
        c[:, 15] = radius
        c[:, 16] = half_width
        c[:, 17] = mu
        c[:, 18] = 1.0 if free_y else 0.0
        c[:, 19] = 1.0 if set_y else 0.0

    def nsub_for(self, speed, cfl=None, even=True):
        """Substeps per 1/60 s frame for a max body/grain speed (the CFL counts it).
        cfl=None uses the constructor's `cfl`."""
        cfl = self.cfl if cfl is None else cfl
        dt0 = self.mat.cfl_dt(self.h, cfl, vmax=max(2.0, float(speed)))
        n = int(math.ceil((1.0 / 60.0) / dt0))
        return n + (n & 1) if even else n

    def _pre(self, do_sort=False):
        wp, K, dev, P = self.wp, self.K, self.dev, self.P
        wp.copy(self.cmd_d, self.cmd_h)
        wp.launch(K["apply_cmd"], dim=self.nb, device=dev,
                  inputs=[self.cmd_d, self.bpos, self.bvel, self.bomega, self.baxis, self.bload,
                          self.bmass, self.bdamp, self.bsize, self.bmu, self.bmode])
        wp.launch(K["window"], dim=self.nb, device=dev,
                  inputs=[self.bpos, self.bvel, self.pc, self.pco, self.fresh, self.nshift, P])
        ncx = int(self.ix * self.h / self.hf_dx) + 3
        ncz = int(self.iz * self.h / self.hf_dx) + 3
        wp.launch(K["hf_clear"], dim=(self.nb, ncx, ncz), device=dev,
                  inputs=[self.hf, self.pc, self.pco, self.fresh, P])
        wp.launch(K["retire"], dim=self.n_slots, device=dev,
                  inputs=[self.x, self.alive, self.fstack, self.ftop, self.hf, self.pc, self.pco,
                          self.fresh, P])
        nqx = int(self.ix * self.h / self.pd) + 3
        nqz = int(self.iz * self.h / self.pd) + 3
        if self.fill_dips:
            wp.launch(K["fill_track"], dim=self.nb, device=dev,
                      inputs=[self.pc, self.pco, self.plast, self.pfill, self.fresh])
            for _ in range(self.fill_iters):
                wp.launch(K["hf_fill"], dim=(self.nb, ncx, ncz), device=dev,
                          inputs=[self.hf, self._hfS, self.pc, self.pco, self.pfill, self.fresh, P])
                wp.launch(K["hf_fill_apply"], dim=(self.nb, ncx, ncz), device=dev,
                          inputs=[self.hf, self._hfS, self.pfill, P])
        wp.launch(K["seed"], dim=(self.nb, nqx, nqz), device=dev,
                  inputs=[self.x, self.v, self.C, self.F, self.tau, self.evp, self.alive,
                          self.fstack, self.ftop, self.ovf, self.hf, self.pc, self.pco,
                          self.fresh, P])
        if do_sort:
            self._sort()

    def _sort(self):
        """Device re-sort of every patch's slots into slab order (inside the graph)."""
        wp, K, dev, P, n = self.wp, self.K, self.dev, self.P, self.n_slots
        wp.launch(K["sort_key"], dim=n, device=dev,
                  inputs=[self.x, self.alive, self.pc, self.skeys, self.svals, P])
        wp.utils.radix_sort_pairs(self.skeys, self.svals, n, 0, self._sort_bits)
        src = [self.x, self.v, self.C, self.F, self.tau, self.evp, self.alive]
        dst = [self._x2, self._v2, self._C2, self._F2, self._tau2, self._evp2, self._alive2]
        wp.launch(K["gather"], dim=n, device=dev, inputs=[self.svals] + src + dst)
        for a, b in zip(src, dst):
            wp.copy(a, b)
        wp.launch(K["rebuild_free"], dim=n, device=dev, inputs=[self.ftop, self.fstack, P])

    def _substeps(self, nsub):
        wp, K, dev, P = self.wp, self.K, self.dev, self.P
        dt = (1.0 / 60.0) / nsub
        for s in range(nsub):
            ga, gb = self.gv[s & 1], self.gv[(s + 1) & 1]
            if self.tiled_p2g:
                wp.launch_tiled(K["p2g_tiled"], dim=[(self.n_slots + 127) // 128], block_dim=128, device=dev,
                                inputs=[self.x, self.v, self.C, self.tau, self.alive, self.pc, self.gm, ga,
                                        P, dt, self.n_slots])
            else:
                wp.launch(K["p2g"], dim=self.n_slots, device=dev,
                          inputs=[self.x, self.v, self.C, self.tau, self.alive, self.pc, self.gm, ga, P, dt])
            wp.launch(K["grid"], dim=(self.nb * self.dims[0], self.dims[1], self.dims[2]), device=dev,
                      inputs=[self.gm, ga, gb, self.pc, P, self.bpos, self.bvel, self.bomega,
                              self.baxis, self.bsize, self.bmu, self.react, self.rtorque, dt])
            wp.launch(K["g2p"], dim=self.n_slots, device=dev,
                      inputs=[self.x, self.v, self.C, self.F, self.tau, self.evp, self.alive, self.pc,
                              ga, P, dt, self.bpos, self.bvel, self.bmode, self.bload, self.bmass,
                              self.bdamp, self.react, self.rtorque, self.fsum, self.tsum, self.nsum])

    def _post(self):
        wp, K, dev = self.wp, self.K, self.dev
        wp.launch(K["finalize"], dim=self.nb, device=dev,
                  inputs=[self.bpos, self.bvel, self.baxis, self.bsize, self.fsum, self.tsum,
                          self.nsum, self.ftop, self.ovf, self.pc, self.nshift, self.fresh,
                          self.out_d, self.P])
        wp.copy(self.out_h, self.out_d)

    def _frame(self, nsub, pre=True, sub=True, post=True, do_sort=False):
        if pre:
            self._pre(do_sort)
        if sub:
            self._substeps(nsub)
        if post:
            self._post()

    def step_frame(self, nsub, graph=True, parts=(True, True, True)):
        """One 1/60 s frame: command upload, window/retire/(pit fill)/seed,
        (device re-sort every `sort_every` frames), nsub substeps, finalize +
        readback. Returns the (K, 16) readback (see OUT_*). nsub must be even
        (ping-pong grid). `parts` exists for cost breakdowns."""
        assert nsub % 2 == 0, "nsub must be even (ping-pong grid)"
        wp = self.wp
        do_sort = bool(parts[0]) and self.sort_every > 0 and self._frame_i % self.sort_every == 0
        self._frame_i += 1
        key = (nsub, self.tiled_p2g, self.fill_dips, do_sort) + tuple(parts)
        if not graph or not str(self.dev).startswith("cuda"):
            self._frame(nsub, *parts, do_sort=do_sort)
        else:
            g = self._graphs.get(key)
            if g is None:
                # This frame runs eagerly (and compiles the module); the capture
                # only records, so the next call is the first graph launch.
                # (The eager frame IS this frame; the capture is not executed.)
                self._frame(nsub, *parts, do_sort=do_sort)
                wp.synchronize_device(self.dev)
                with wp.ScopedCapture(device=self.dev) as cap:
                    self._frame(nsub, *parts, do_sort=do_sort)
                self._graphs[key] = cap.graph
            else:
                wp.capture_launch(g)
        wp.synchronize_device(self.dev)
        return self.out_h.numpy().copy()

    # -- inspection ----------------------------------------------------------------

    def heightfield(self):
        return self.hf.numpy()

    def hf_coords(self):
        hnx, hnz = self.hf.shape
        xs = self.hf_origin[0] + (np.arange(hnx) + 0.5) * self.hf_dx
        zs = self.hf_origin[1] + (np.arange(hnz) + 0.5) * self.hf_dx
        return xs, zs

    def particles(self, k=None):
        x = self.x.numpy()
        a = self.alive.numpy().astype(bool)
        if k is not None:
            sl = np.zeros_like(a)
            sl[k * self.cap:(k + 1) * self.cap] = True
            a &= sl
        return x[a]

    def block_interior(self, k):
        """(xlo, xhi, zlo, zhi) of block k's soil interior, m."""
        pc = self.pc.numpy()[k]
        lo = (pc + self.marg) * self.h
        return lo[0], lo[0] + self.ix * self.h, lo[1], lo[1] + self.iz * self.h
