"""
usv_rig -- a boat on the FFT ocean: her strips, her drive, her mark on the sea.

Four things, and nothing scene-specific:

  * ``StripHull`` -- one boat on her Bonjean strips: <boat>_spec.json and <boat>_hydro.json
                     (schema threepp.usv_hydro/1, what build_<boat>_blender.py writes). The
                     half-section tables are sampled under each station on both sides against
                     ocean.sample_height, and the strip forces and their moments heave, pitch
                     and roll a 6-DOF rigid body. What moves her over the water is one of two
                     things: a soft DP toward a target pose (the default: she rides the waves
                     and holds station, or follows a route), or a drive (``hull.drive``).
  * the drives    -- ``Waterjet`` (the Mariner), ``AzimuthPods`` (the Otter X) and
                     ``FixedPods`` (the Otter): thrust, hull resistance and the lumped
                     manoeuvring terms, the helm and how fast the actuators follow it, the
                     actuator nodes of the .glb, the foam. ``drive_for(hull)`` makes a boat's.
  * ``Wake``      -- the hull footprint and the Kelvin wake trail on the ocean, or on one of
                     its vessels (``ocean.vessel(i)``) where several boats share a sea.
  * ``Wash``      -- what her propulsors and her hull leave ON the water: the races of her
                     propellers or her jet, and the lane astern of her hull, as sources of the
                     ocean's wake field (``ocean.wake_field``, a patch of it for each boat).

Nothing here imports threepp or opens a window, and nothing reads a module global belonging to
a scene: a scene hands in its ocean, its loaded .glb and its own wake figures.

FRAMES
------
Vessel frame (the spec's): X forward, Y up, Z starboard, origin on the baseline amidships. The
rigid body tracks the CoG in world space and a rotation R (vessel -> world); positive yaw turns
to port, positive pitch lifts the bow, positive roll puts the starboard side down. A heading a
points the vessel's +X along world (cos a, 0, -sin a).

ORDER IN A FRAME
----------------
ocean.sample_height reads the last rendered field, so a boat steps AFTER a render::

    hull = StripHull("otter", ocean)
    hull.seat(x, z, heading)
    wake = Wake(hull, ocean)
    wash = Wash(hull, drive_for(hull), ocean, 0)      # ocean.wake_field.resolution set before the first render

    # on a DP, along a route                    # under her own power
    drive = drive_for(hull).bind(glb_root)      hull.drive = drive_for(hull).bind(glb_root)
    hull.set_target(x, z, hd, vx, vz, rate)     hull.drive.throttle_cmd = 1.0
    hull.advance(dt, substeps)                  hull.drive.tick(dt); hull.advance(dt, substeps)
    u = wake.update(dt)                         u = wake.update(dt)
    drive.follow(u, rate, dt)
    ocean.clear_wake_sources()                  # once a frame, before the boats' washes
    wash.update(dt)
    hull.place(glb_root); drive.pose()          hull.place(glb_root); hull.drive.pose()

PER-BOAT NUMBERS
----------------
The spec's ``strip_model`` block holds what the strip model needs beyond the tables: where each
half's water level is read (``probe_z``) and the ocean's hull footprint. A spec without one
gets both from its design waterline. The drives read the spec's ``propulsion`` block; their
lumped resistance and damping figures are tuned per boat and stand in the classes below.

MANOEUVRING (opt-in, ``hull.manoeuvring = True``, a hull under her own power with pods)
--------------------------------------------------------------------------------------
Off by default; with it off nothing above changes. On, the horizontal plane (surge, sway, yaw)
follows the 3-DOF manoeuvring model in water-relative velocity (Fossen, Handbook of Marine
Craft Hydrodynamics and Motion Control, 2011, ch. 6 and 10), and heave, roll and pitch stay
on the strips::

    M nu_r' + C(nu_r) nu_r + D(nu_r) nu_r = tau        nu_r = (u_r, v_r, r), body x ahead, y to starboard
    eta' = R(psi) nu_r + V_c                            eta = (north, east, psi), psi clockwise from north

    M = M_RB + M_A = diag(m + a11, m + a22, I_z + a66)  (the origin is the CoG)
    C = C_RB + C_A, both on nu_r (Kirchhoff):  C nu_r = (-(m + a22) v_r r, (m + a11) u_r r, (a22 - a11) u_r v_r)

The third term of C nu_r is the Munk moment. V_c is ``hull.current``, a uniform steady current
(world x east, y up, z south, m/s; its y is not used). The numbers are the rig's own, none of
them identified: a11, a22, a66 are the added-mass fractions StripHull has always used (0.05 m,
0.60 m, 0.30 I_z, "the usual rough figures for a hull this shape"); D(nu_r) nu_r is the
drive's lumped terms taken on water-relative velocity: surge resistance R(u_r) (linear,
quadratic, the wave-making hump), sway KV1 (|u_r| + 1) v_r + KV2 v_r |v_r| acting at the CoG's
station (so no yaw moment from sway: N_v = 0), yaw KR1 (|u_r| + 1) r + KR2 r |r|. tau is the
pods' thrust (thrust_max(u_r)) along their thrust line, its force and its yaw moment about the
CoG. Heave, roll and pitch: the strips' buoyancy, the heave, roll and pitch damping, the
thrust's and the sway damping's roll and pitch moments, the squat moment, as in loads(). The
heading is psi, integrated from r; the rotation R takes its roll and pitch from the strips and
its heading from psi. The horizontal plane reads nothing from the sea but the share of the
weight it carries and the pods' immersion (both clamped to 1 while she floats), so on a sea
that keeps her afloat her track does not depend on the waves.

Open-loop directional stability (``drive.sway_yaw(u)``, pods amidships, the linear sway-yaw
pair at surge u): with N_v = 0 the Munk moment is the only yaw moment from sideslip, and the
pair is stable only while KV1 (u + 1) KR1 (u + 1) > (m + a11) (a22 - a11) u^2. The Otter X at
her design load (1030 kg) is directionally unstable above 0.98 m/s through the water: its
unstable root is +0.10 /s at 1.5 m/s, +0.20 at 2.0, +0.29 at 2.5 (the stable one -1.31, -1.65,
-1.99). Her heading needs a controller.
"""
import json
import math
import os

import numpy as np

USV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "usv")

BUOY_MASK = 0b011                  # swell + mid band; cascade 2 is chop the hull ignores


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def orthonormal(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def quat_of(R):
    """(x, y, z, w) of a rotation matrix."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def attitude(R):
    """Heading, pitch (bow up +) and roll (starboard down +), radians."""
    return (math.atan2(-R[2, 0], R[0, 0]), math.asin(max(-1.0, min(1.0, R[1, 0]))),
            math.atan2(-R[1, 2], R[1, 1]))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def slew(x, target, rate, dt):
    """x moved toward target at no more than rate: an actuator following its command."""
    return x + max(-rate * dt, min(rate * dt, target - x))


# --------------------------------------------------------------------------- #
#  The hull on her strips
# --------------------------------------------------------------------------- #
class StripHull:
    """One boat on her Bonjean strips.

    boat       the name of <boat>_spec.json and <boat>_hydro.json
    load       a key of the spec's mass conditions (default: its design condition)
    spec_dir   the folder holding the two files (default USV_DIR)
    probe_z    half-breadth where each side's water is read (default: the spec's
               strip_model.probe_z, else 0.314 x the design waterline beam: the strip model
               heels the half-sections by the level change AT probe_z, so probe_z sets its
               roll stiffness, and 0.314 bwl reproduces the sjark's GM_T 1.39 m
               (sjark_hydro.json) under a known moment)
    excl       ocean hull footprint (half length, half beam, waterplane centre x) (default:
               the spec's strip_model.footprint, else half the design waterline length,
               0.42 x its beam, its LCF)
    mask       the ocean cascades she feels (default BUOY_MASK)

    drive      None: a soft DP holds surge, sway and yaw toward set_target() and leaves heave,
               pitch and roll free (assumed: ~0.35 rad/s horizontal, ~0.5 rad/s yaw, near
               critical; wn_xy, wn_yaw, zeta). A drive (drive_for(hull)): its loads move her."""

    def __init__(self, boat, ocean, load=None, *, spec_dir=None, probe_z=None, excl=None, mask=None):
        self.boat, self.ocean = boat, ocean
        spec_dir = spec_dir or USV_DIR
        with open(os.path.join(spec_dir, f"{boat}_spec.json"), encoding="utf-8") as fh:
            spec = json.load(fh)
        with open(os.path.join(spec_dir, f"{boat}_hydro.json"), encoding="utf-8") as fh:
            hydro = json.load(fh)
        self.spec = spec
        self.load = load = load or spec["mass"]["design_condition"]
        self.rho, self.g = hydro["rho"], hydro["g"]
        items = spec["mass"]["budget"] + spec["mass"]["conditions"][load]["extra"]
        m = sum(b["mass"] for b in items)
        self.mass = m
        self.cog = np.array([sum(b["mass"] * b["centroid"][i] for b in items) / m for i in range(3)])
        self.design = D = hydro["conditions"][load]
        gyr = spec["mass"]["gyradius"]
        # Added mass and inertia as fractions (surge, heave, sway) and (roll, yaw, pitch): the
        # usual rough figures for a hull this shape, not a strip-theory result.
        self.m_eff = m * (1.0 + np.array([0.05, 0.80, 0.60]))
        self.i_eff = m * np.array([gyr["roll"], gyr["yaw"], gyr["pitch"]]) ** 2 * (1.0 + np.array([0.25, 0.30, 0.60]))
        # Bonjean strips: every station, both sides. The generator tabulated the starboard
        # half; port is its mirror.
        bj = hydro["bonjean"]
        xs = np.array(bj["station_x"])
        ns = len(xs)
        self.dx = bj["station_dx"]
        hs = np.array(bj["heights"])
        self.h0, self.dh, self.nh = hs[0], hs[1] - hs[0], len(hs)
        self.tab_a = np.vstack([bj["area_half"], bj["area_half"]])
        self.tab_z = np.vstack([bj["zc_half"], bj["zc_half"]])
        self.tab_y = np.vstack([bj["yc_half"], bj["yc_half"]])
        self.sx = np.concatenate([xs, xs])
        self.side = np.concatenate([np.ones(ns), -np.ones(ns)])
        self.row = np.arange(2 * ns)
        rig = spec.get("strip_model", {})
        if probe_z is None:
            probe_z = rig["probe_z"] if "probe_z" in rig else 0.314 * D["bwl"]
        if excl is None:
            fp = rig.get("footprint")
            excl = ((fp["half_length"], fp["half_beam"], fp["x_centre"]) if fp
                    else (0.5 * D["lwl"], 0.42 * D["bwl"], D["lcf"]))
        self.probe_z, self.excl = probe_z, tuple(excl)
        self.mask = BUOY_MASK if mask is None else mask
        self.lowest = spec["principal"]["lowest_point_y"]
        # damping from the design hydrostatics: a fraction of critical on each axis
        rg = self.rho * self.g
        self.k_heave = rg * D["waterplane_area"]
        self.k_roll = rg * D["displacement_m3"] * D["gm_t"]
        self.k_pitch = rg * D["displacement_m3"] * D["gm_l"]
        self.c_heave = 2.0 * 0.35 * math.sqrt(self.k_heave * self.m_eff[1])
        self.c_roll = 2.0 * 0.15 * math.sqrt(self.k_roll * self.i_eff[0])
        self.c_pitch = 2.0 * 0.45 * math.sqrt(self.k_pitch * self.i_eff[2])
        self.drive = None
        self.wn_xy, self.wn_yaw, self.zeta = 0.35, 0.5, 0.9
        self.p = np.zeros(3)                   # CoG, world
        self.v = np.zeros(3)                   # CoG velocity, world
        self.R = np.eye(3)                     # vessel -> world
        self.w = np.zeros(3)                   # angular velocity, vessel frame
        self.water = np.zeros(2 * ns)
        self.pts = None                        # where self.water was read
        self.buoy = 0.0
        self.u = 0.0                           # surge speed going into the last substep
        self.balance = None                    # that substep's (vb, w, buoyancy moment, other moment), vessel frame
        self.target = (0.0, 0.0, 0.0)          # x, z, heading
        self.target_v = (0.0, 0.0, 0.0)        # vx, vz, heading rate
        self.target_a = (0.0, 0.0)             # feed-forward horizontal acceleration
        self.log = []
        # MANOEUVRING (opt-in): the 3-DOF horizontal plane in water-relative velocity
        self.manoeuvring = False
        self.current = np.zeros(3)             # uniform steady current, world (x east, z south), m/s
        self.nu_r = None                       # (u_r, v_r, r): body x ahead, y to starboard, r clockwise from above
        self.psi = 0.0                         # heading, rad clockwise from north (NED)

    # ---- the sea under her
    def probe_points(self):
        """World XZ under each half-section's probe, at the CoG's height."""
        rel = np.stack([self.sx - self.cog[0], np.zeros(len(self.sx)), self.side * self.probe_z - self.cog[2]], axis=1)
        return self.p + rel @ self.R.T

    def sample_water(self):
        self.pts = pts = self.probe_points()
        self.water = np.array([self.ocean.sample_height(float(x), float(z), self.mask) for x, _, z in pts])
        return pts

    def seat(self, x, z, heading):
        """Drop her on her design waterline and trim at (x, z), heading, at rest."""
        self.R = rot_y(heading) @ rot_z(math.radians(self.design["trim_deg"]))
        self.p = np.array([x, 0.0, z], float)          # the CoG; (x, z) is where she holds station
        self.sample_water()
        h0 = float(self.water.mean())
        rel = np.array([0.0, self.design["waterline_y_at_x0"], 0.0]) - self.cog
        self.p[1] = h0 - float(self.R[1] @ rel)
        self.v[:] = 0.0
        self.w[:] = 0.0
        self.nu_r = None                               # MANOEUVRING: taken from v, w and R at the next step
        self.target = (float(self.p[0]), float(self.p[2]), heading)

    def set_target(self, x, z, heading, vx=0.0, vz=0.0, rate=0.0, ax=0.0, az=0.0):
        self.target, self.target_v, self.target_a = (x, z, heading), (vx, vz, rate), (ax, az)

    def dp_loads(self, vb):
        """Vessel-frame force and moment of the DP: a horizontal PD toward the target (world),
        the target's velocity and acceleration fed forward, and the heave, roll and pitch damping."""
        R, v, w = self.R, self.v, self.w
        tx, tz, th = self.target
        tvx, tvz, trate = self.target_v
        wn, ze = self.wn_xy, self.zeta
        acc = np.array([self.target_a[0] + wn * wn * (tx - self.p[0]) + 2 * ze * wn * (tvx - v[0]), 0.0,
                        self.target_a[1] + wn * wn * (tz - self.p[2]) + 2 * ze * wn * (tvz - v[2])])
        F_dp = self.mass * 1.3 * acc
        hd = attitude(R)[0]
        wy = self.wn_yaw
        M_yaw = self.i_eff[1] * (wy * wy * wrap(th - hd) + 2 * ze * wy * (trate - w[1]))
        Fb = np.array([0.0, -self.c_heave * vb[1], 0.0]) + R.T @ F_dp
        Mb = np.array([-self.c_roll * w[0], M_yaw, -self.c_pitch * w[2]])
        return Fb, Mb

    def buoyancy(self):
        """The strips' buoyancy less the weight, world frame: (force, moment about the CoG, each
        half-section's immersed area). Sets self.buoy."""
        R = self.R
        cog, sx, side = self.cog, self.sx, self.side
        # buoyancy: each half-section at its own water level, in vessel y
        rx, rz = sx - cog[0], side * self.probe_z - cog[2]
        yb = cog[1] + (self.water - self.p[1] - R[1, 0] * rx - R[1, 2] * rz) / max(R[1, 1], 0.3)
        t = np.clip((yb - self.h0) / self.dh, 0.0, self.nh - 1.001)
        i = t.astype(np.intp)
        f = t - i
        row = self.row
        area = self.tab_a[row, i] + (self.tab_a[row, i + 1] - self.tab_a[row, i]) * f
        zc = self.tab_z[row, i] + (self.tab_z[row, i + 1] - self.tab_z[row, i]) * f
        yc = self.tab_y[row, i] + (self.tab_y[row, i + 1] - self.tab_y[row, i]) * f
        fb = self.rho * self.g * self.dx * area
        arm = np.stack([sx - cog[0], yc - cog[1], side * zc - cog[2]], axis=1) @ R.T
        F = np.array([0.0, float(fb.sum()) - self.mass * self.g, 0.0])
        tau = np.array([-float(arm[:, 2] @ fb), 0.0, float(arm[:, 0] @ fb)])     # sum r x (0, fb, 0)
        self.buoy = float(fb.sum())
        return F, tau, area

    def step(self, dt):
        """One substep: strip buoyancy + the DP's or the drive's loads, semi-implicit Euler."""
        if self.manoeuvring and self.drive is not None:
            return self.step_manoeuvring(dt)
        R, v, w = self.R, self.v, self.w
        F, tau, area = self.buoyancy()
        # vessel-frame loads from everything but buoyancy and gravity, damping included
        vb = R.T @ v
        self.u = vb[0]
        Fb, Mb = self.dp_loads(vb) if self.drive is None else self.drive.loads(vb, w, area)
        # integrate: forces in the vessel frame against the per-axis added mass
        tb = R.T @ tau
        Fv = R.T @ F + Fb
        tv = tb + Mb
        self.balance = (vb, w, tb, Mb)
        self.v = v + (R @ (Fv / self.m_eff)) * dt
        self.p = self.p + self.v * dt
        Iw = self.i_eff * w
        self.w = w + (tv - np.cross(w, Iw)) / self.i_eff * dt
        self.R = orthonormal(R @ rodrigues(self.w * dt))

    # ---- MANOEUVRING (opt-in)
    def mass_3dof(self):
        """(m11, m22, m66, a11, a22, a66): M = M_RB + M_A of the 3-DOF model, from the rig's added-mass
        fractions (m_eff, i_eff)."""
        return (self.m_eff[0], self.m_eff[2], self.i_eff[1], self.m_eff[0] - self.mass, self.m_eff[2] - self.mass,
                self.i_eff[1] - self.mass * self.spec["mass"]["gyradius"]["yaw"] ** 2)

    def current_ned(self):
        """hull.current as (north, east), m/s."""
        return -float(self.current[2]), float(self.current[0])

    def init_manoeuvring(self):
        """nu_r and psi from the rigid body's v, w and R (and the current)."""
        self.psi = 0.5 * math.pi - attitude(self.R)[0]
        cn, ce = self.current_ned()
        rn, re = -float(self.v[2]) - cn, float(self.v[0]) - ce
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.nu_r = np.array([c * rn + s * re, -s * rn + c * re, -float(self.w[1])])

    def step_manoeuvring(self, dt):
        """One substep with the horizontal plane on the 3-DOF model in water-relative velocity
        (MANOEUVRING), heave, roll and pitch on the strips; semi-implicit Euler like step()."""
        if self.nu_r is None:
            self.init_manoeuvring()
        R, v, w = self.R, self.v, self.w
        F, tau, area = self.buoyancy()
        vb = R.T @ v
        self.u = vb[0]
        (X, Y, N), Fb, Mb = self.drive.loads_3dof(self.nu_r, vb, w, area)
        m11, m22, m66, a11, a22, _ = self.mass_3dof()
        u, vr, r = self.nu_r
        # M nu_r' = tau - C(nu_r) nu_r - D(nu_r) nu_r (D is in the drive's X, Y, N)
        u_n = u + (X + m22 * vr * r) / m11 * dt
        v_n = vr + (Y - m11 * u * r) / m22 * dt
        r_n = r + (N - (a22 - a11) * u * vr) / m66 * dt
        self.nu_r = np.array([u_n, v_n, r_n])
        psi = self.psi + r_n * dt
        # eta' = R(psi) nu_r + V_c
        cn, ce = self.current_ned()
        c, s = math.cos(psi), math.sin(psi)
        vn, ve = c * u_n - s * v_n + cn, s * u_n + c * v_n + ce
        # heave, roll and pitch: the strips and the drive's vertical loads, as step()
        tb = R.T @ tau
        Fv = R.T @ F + Fb
        tv = tb + Mb
        self.balance = (vb, w, tb, Mb)
        vy = float(v[1] + (R @ (Fv / self.m_eff))[1] * dt)
        self.v = np.array([ve, vy, -vn])
        self.p = self.p + self.v * dt
        Iw = self.i_eff * w
        wn = w + (tv - np.cross(w, Iw)) / self.i_eff * dt
        wn[1] = -r_n                                    # the yaw rate is the 3-DOF model's
        self.w = wn
        Rn = orthonormal(R @ rodrigues(wn * dt))
        self.R = rot_y(wrap(0.5 * math.pi - psi - attitude(Rn)[0])) @ Rn      # its heading is psi
        self.psi = psi

    def advance(self, dt, substeps=1):
        """Sample the (already rendered) sea, then step. Logs draft, trim, roll."""
        self.sample_water()
        for _ in range(substeps):
            self.step(dt / substeps)
        r = self.readout()
        self.log.append((r["draft"], r["trim"], r["roll"]))

    # ---- scene
    def place(self, obj):
        o = self.p - self.R @ self.cog
        obj.position.set(float(o[0]), float(o[1]), float(o[2]))
        obj.quaternion.set(*quat_of(self.R))

    def to_world(self, pv):
        """A vessel-frame point to world."""
        return self.p + self.R @ (np.asarray(pv, float) - self.cog)

    def footprint(self, target, on=True):
        """Give her hull footprint to the ocean, or to one of its vessels: the sea's own plane
        under her, least squares over the points the water was last read at. on=False takes the
        footprint away. Returns its centre (world) and the ocean's heading for her."""
        hl, hb, xc = self.excl
        fwd = self.R[:, 0]
        yaw_ex = math.atan2(fwd[0], fwd[2])          # the ocean's heading: forward = (sin, cos)
        s_ex = np.array([math.cos(yaw_ex), -math.sin(yaw_ex)])
        c = self.to_world([xc, 0.0, 0.0])
        pts = self.pts if self.pts is not None else self.probe_points()
        A = np.stack([np.ones(len(pts)), pts[:, 0] - c[0], pts[:, 2] - c[2]], axis=1)
        h0, gx, gz = np.linalg.lstsq(A, self.water, rcond=None)[0]
        target.hull_exclusion.set_pose(float(c[0]), float(c[2]), float(h0), yaw=yaw_ex,
                                       pitch=math.atan(gx * fwd[0] + gz * fwd[2]),
                                       roll=math.atan(gx * s_ex[0] + gz * s_ex[1]),
                                       half_length=hl if on else 0.0, half_beam=hb)
        return c, yaw_ex

    def readout(self):
        hd, pitch, roll = attitude(self.R)
        wl = float(np.mean(self.water))
        low = float(self.to_world([0.0, self.lowest, 0.0])[1])
        return {"heading": math.degrees(hd), "trim": math.degrees(pitch), "roll": math.degrees(roll),
                "draft": wl - low}

    def summary(self, skip=0):
        a = np.array(self.log[skip:]) if len(self.log) > skip else np.zeros((1, 3))
        return (f"{self.boat}: {self.mass:.0f} kg, draft {a[:, 0].mean():.3f} m (range {a[:, 0].min():.3f}"
                f"..{a[:, 0].max():.3f}), trim {a[:, 1].min():+.2f}..{a[:, 1].max():+.2f} deg, "
                f"roll {a[:, 2].min():+.2f}..{a[:, 2].max():+.2f} deg")


# --------------------------------------------------------------------------- #
#  The drives: the boat's own half. Propulsion and the manoeuvring terms, the
#  helm, the actuator nodes, the foam she throws.
#
#  loads() returns the vessel-frame force and the moment about the CoG from
#  everything but buoyancy and gravity, damping included. The hydrostatics are
#  the tables; the manoeuvring is a lumped sketch tuned to the boat's top speed.
# --------------------------------------------------------------------------- #
class _Loads:
    """Point loads summed into a force and a moment about the CoG, vessel frame."""

    def __init__(self, cog):
        self.cog, self.F, self.M = cog, np.zeros(3), np.zeros(3)

    def add(self, force, at):
        self.F = self.F + force
        self.M = self.M + np.cross(np.asarray(at) - self.cog, force)


class Drive:
    """A boat's propulsion. The helm writes throttle_cmd (throttle_min..1) and steer_cmd
    (-steer_max..steer_max, + to starboard); tick() slews the actuators after them.

    Every drive has: u_top (m/s), throttle_min, steer_max, deep_pts (the low points a draft is
    measured to, vessel frame) and solver_draft (the hydrostatic solver's draft to them)."""

    throttle_min = 0.0

    def __init__(self, hull):
        self.hull = hull
        self.throttle = self.throttle_cmd = 0.0
        self.steer = self.steer_cmd = 0.0
        self.nodes = {}
        self.wet = 0.0                         # 0..1, how much of her weight the sea carried at the last loads()

    def _state(self, vb):
        """Surge, heave, sway, and how far she is afloat, upright, trimmed and heeled."""
        h, R = self.hull, self.hull.R
        u, heave, sway = vb
        self.wet = wet = min(max(h.buoy / (0.35 * h.mass * h.g), 0.0), 1.0)
        upright = min(max(R[1, 1] * 2.0, 0.0), 1.0)
        trim_deg = math.degrees(math.asin(max(-1.0, min(1.0, R[1, 0]))))
        roll = math.atan2(-R[1, 2], R[1, 1])
        return u, heave, sway, wet, upright, trim_deg, roll

    def loads_3dof(self, nu_r, vb, w, area):
        raise NotImplementedError(f"{type(self).__name__} has no 3-DOF manoeuvring loads (MANOEUVRING: the pods have)")

    def thrusts(self, u):
        """Her propulsors at surge speed u, each as (name, where it is, the force it puts on the
        boat (N), its disc's or nozzle's radius, is it a jet), the first two in the vessel frame.
        Wash turns these into races on the water. A drive that is the hull's own reports what its
        actuators are doing; one that only follows a hull something else moves (a DP on a route)
        reports the thrust her speed takes."""
        return []

    def bind(self, root):
        """Find the actuator nodes under a loaded .glb's root. Returns self."""
        self.nodes = {k: root.get_object_by_name(n) for k, n in self.NODE_NAMES.items()}
        return self

    def foam(self, ocean, u):
        """Her foam splats on the ocean at surge speed u. They last one frame: the scene calls
        ocean.clear_foam_disturbances() before its boats add theirs."""
        h = self.hull

        def emit(xb, zb, radius, k):
            q = h.to_world([xb, 0.4, zb])
            if k > 0.02:
                ocean.add_foam_disturbance(float(q[0]), float(q[2]), radius, min(k, 1.0))
        self._foam(emit, u)


class Waterjet(Drive):
    """The Mariner: a Hamilton-type jet. Thrust along the steerable nozzle, a reverse bucket that
    turns it round, a bow thruster for slow work; resistance with a planing hump and a planing
    lift, tuned so full throttle makes the brochure's 24 kn at about 4 deg of running trim."""

    NODE_NAMES = {"steer": "jet_steering", "bucket": "jet_reverse_bucket"}

    HUMP_U, HUMP_W, HUMP_R = 4.2, 1.8, 1600.0           # the planing hump
    TRIM_PLANE = math.radians(3.5)                      # running trim on the plane
    R_LIN = 80.0
    # Planing lift grows with speed AND with trim, and its centre of pressure slides
    # aft as the bow comes up; that is what makes a planing hull stable in pitch.
    # A lift fixed ahead of the CoG with a constant hump moment has no such brake:
    # the first version stood her on her transom at 30 deg and rolled her over.
    LIFT_MAX = 0.45                    # x weight, on the plane at LIFT_TRIM
    LIFT_TRIM = 5.0                    # deg of (trim + 1.5) that gives LIFT_MAX
    HUMP_M = 15000.0                   # bow-up moment at the hump (N m), gone by 8 deg of trim
    K_BANK = 1200.0                    # deep-V banks into a turn
    KV1, KV2 = 1500.0, 3000.0          # sway damping
    KR1, KR2 = 575.0, 32000.0          # yaw damping
    SWAY_Y = 0.35                      # height the lateral force acts at

    def __init__(self, hull):
        super().__init__(hull)
        spec = hull.spec
        jet = spec["propulsion"]["waterjet"]
        self.t_bollard = jet["bollard_thrust_n"]
        self.u_top = spec["propulsion"]["top_speed_mps"]
        self.jet_at = np.array(jet["thrust_point"])
        self.nozzle_r = jet["tailpipe"]["radius"][0]
        self.steer_max = math.radians(jet["steering"]["limit_deg"])
        self.bucket_max = math.radians(jet["reverse_bucket"]["limit_deg"][1])
        self.bt = spec["propulsion"]["bow_thruster"]
        self.bt_at = np.array(self.bt["position"])
        self.bucket = self.bucket_cmd = 0.0             # 0 raised (ahead) .. 1 across the jet
        self.thruster = 0.0                             # bow thruster, -1 .. 1 (+ pushes the bow to starboard)
        u_top = self.u_top
        # Full throttle balances resistance plus the weight's pull along a hull trimmed
        # TRIM_PLANE at U_TOP, so she makes the brochure's 24 kn.
        self.r_quad = (self.thrust_max(u_top) - hull.mass * hull.g * math.sin(self.TRIM_PLANE)
                       - self.R_LIN * u_top) / u_top ** 2
        # Up on the plane she rides the narrow bottom of her V, where the waterline is
        # half as wide and the hydrostatic righting moment (~ breadth cubed) a quarter
        # of what it is at rest: measured in the first run, 2 kN m at 20 deg against
        # 8.8. A planing hull's roll stiffness comes from the planing surface instead:
        # the lift's centre moves toward the side that goes down. lift_roll_arm (m per
        # rad of roll) is sized so that dynamic stiffness matches the static one.
        self.lift_roll_arm = hull.k_roll / (hull.mass * hull.g * self.LIFT_MAX)
        self.aft = (hull.sx > -2.2) & (hull.sx < -1.4)  # the sections over the jet's intake
        ap = spec["appendages"]
        # the ends of the two flats that reach the lowest point: keel shoe and gondola
        self.deep_pts = np.array([[x, hull.lowest, 0.0] for x in (ap["keel_shoe"]["x_aft"], 1.66, ap["gondola"]["x_fwd"],
                                                                 ap["gondola"]["x_flat_aft"])])
        self.solver_draft = hull.design["draft_max"]

    def thrust_max(self, u):
        """A jet loses thrust with speed: bollard at rest, ~55 % of it at the top."""
        return self.t_bollard * (1.0 - 0.45 * min(max(u / self.u_top, 0.0), 1.3))

    def resistance(self, u):
        au = abs(u)
        return math.copysign(self.R_LIN * au + self.r_quad * au * au
                             + self.HUMP_R * math.exp(-((au - self.HUMP_U) / self.HUMP_W) ** 2), u)

    def loads(self, vb, w, area):
        """Hull resistance, planing lift, the jet, the bow thruster, damping."""
        h = self.hull
        cog = h.cog
        u, heave, sway, wet, upright, trim_deg, roll = self._state(vb)
        L = _Loads(cog)
        jet_wet = min(max(float(area[self.aft].sum()) / 0.25, 0.0), 1.0) * upright
        L.add(np.array([-self.resistance(u) * wet, -h.c_heave * heave, 0.0]), cog)
        L.add(np.array([0.0, 0.0, -(self.KV1 * (abs(u) + 1.0) * sway + self.KV2 * sway * abs(sway)) * wet]),
              (cog[0], self.SWAY_Y, cog[2]))
        lift = (h.mass * h.g * self.LIFT_MAX * smoothstep(2.5, 9.0, u) * wet * upright
                * min(max((trim_deg + 1.5) / self.LIFT_TRIM, 0.0), 1.4))
        x_cp = cog[0] + min(max(0.8 - 0.22 * (trim_deg - 2.0), -1.2), 1.0)
        z_cp = min(max(self.lift_roll_arm * roll, -0.8), 0.8)
        L.add(np.array([0.0, lift, 0.0]), (x_cp, cog[1], z_cp))
        t_ax = self.throttle * self.thrust_max(u) * (1.0 - 1.6 * self.bucket) * jet_wet
        d = self.steer
        L.add(t_ax * np.array([math.cos(d), 0.0, -math.sin(d)]), self.jet_at)
        L.add(np.array([0.0, 0.0, self.bt["thrust_n"] * self.thruster * max(0.0, 1.0 - abs(u) / 3.0) * wet]), self.bt_at)
        p_, r_, q_ = w                                  # roll, yaw, pitch rates (vessel X, Y, Z)
        Mb = L.M + np.array([-h.c_roll * p_ - self.K_BANK * u * r_ * wet,
                             -(self.KR1 * (abs(u) + 1.0) * r_ + self.KR2 * r_ * abs(r_)) * max(wet, 0.2),
                             -h.c_pitch * q_ + self.HUMP_M * math.exp(-((u - self.HUMP_U) / 2.0) ** 2) * wet
                             * min(max(1.0 - trim_deg / 8.0, 0.0), 1.0)])
        return L.F, Mb

    def thrusts(self, u):
        d = self.steer
        if self.hull.drive is self:
            t = self.throttle * self.thrust_max(u) * (1.0 - 1.6 * self.bucket)
        else:
            t = self.resistance(u)
        out = [("jet", self.jet_at, t * np.array([math.cos(d), 0.0, -math.sin(d)]), self.nozzle_r, True)]
        if abs(self.thruster) > 0.05:
            out.append(("bow_thruster", self.bt_at, np.array([0.0, 0.0, self.bt["thrust_n"] * self.thruster]),
                        self.bt["tunnel_radius"], False))
        return out

    def tick(self, dt):
        self.throttle = slew(self.throttle, self.throttle_cmd, 0.8, dt)
        self.steer = slew(self.steer, self.steer_cmd, self.steer_max / 0.7, dt)
        self.bucket = slew(self.bucket, self.bucket_cmd, 1.0 / 0.8, dt)

    def follow(self, u, yaw_rate, dt, full_rate=0.3, lock=math.radians(22.0)):
        """For a hull that something else moves (a DP on a route): the throttle her speed implies
        and the nozzle the turn implies, hard over at full_rate rad/s. A positive heading rate is
        a turn to port, a negative nozzle."""
        self.throttle = min(max(u / self.u_top, self.throttle_min), 1.0)
        self.steer = float(np.clip(-yaw_rate / full_rate, -1.0, 1.0)) * lock

    def pose(self):
        n = self.nodes
        if n.get("steer") is not None:
            n["steer"].rotation.y = self.steer
        if n.get("bucket") is not None:
            n["bucket"].rotation.z = self.bucket * self.bucket_max

    def status(self):
        return f"thr {self.throttle:.2f}  steer {math.degrees(self.steer):+5.1f}  bucket {self.bucket:.2f}"

    def _foam(self, foam, u):
        # Splats are gaussians whose visible halo reaches ~1.9 radii, so each sits far
        # enough aft that none of it lands ahead of the stem, and at intensities that
        # leave lace at the edges instead of a solid sheet.
        spd = min(abs(u) / self.u_top, 1.0)
        for zs in (-0.75, 0.75):
            foam(2.2, zs, 0.9, 0.10 + 0.45 * spd)                            # bow wave, off the shoulders
        for zs in (-1.15, 1.15):
            foam(0.2, zs, 1.1, 0.45 * spd)                                   # spray off the collar
        foam(-3.4, 0.0, 1.0 + 0.6 * spd, 0.55 * self.throttle + 0.3 * self.bucket * self.throttle)  # jet wash
        if abs(self.thruster) > 0.05:
            for zs in (-0.6, 0.6):
                foam(self.bt_at[0], zs, 0.7, 0.6 * abs(self.thruster))


class _Pods(Drive):
    """Two pods, one under each demihull of a displacement catamaran: friction and form drag,
    a mild wave-making hump at Froude number 0.45 on her waterline length, a squat moment at
    the hump. Full throttle on both pods balances the resistance at u_top. A pod pushes
    DIFF_GAIN more and the other as much less on the differential; astern (throttle below
    zero) gives the spec's fraction of the bollard thrust. A pod draws air when the aft
    sections on its side come out of the water.

    A subclass gives the per-boat figures and _setup()'s arguments."""

    NODE_NAMES = {"pod_port": "thruster_port", "pod_stbd": "thruster_stbd",
                  "rotor_port": "thruster_port_rotor", "rotor_stbd": "thruster_stbd_rotor"}
    throttle_min = -0.5                # the pods run astern
    DIFF_GAIN = 0.6
    K_BANK = 0.0                       # a catamaran heels out of a turn on the sway force alone

    def _setup(self, th, u_top, pontoon_z, pod_x, pod_y):
        h = self.hull
        D = h.design
        self.t_bollard = th["bollard_thrust_n"]            # per pod
        self.astern = th["astern_fraction"]
        self.u_top = u_top
        self.pontoon_z = pontoon_z
        self.pod_at = {side: np.array([pod_x, pod_y, s * pontoon_z]) for side, s in (("port", -1.0), ("stbd", 1.0))}
        self.spin = {"port": 0.0, "stbd": 0.0}             # the drawn rotors' angles
        self.hump_u = 0.45 * math.sqrt(h.g * D["lwl"])
        self.r_quad = (2.0 * self.thrust_max(u_top) - self.R_LIN * u_top
                       - self.HUMP_R * math.exp(-((u_top - self.hump_u) / self.HUMP_W) ** 2)) / u_top ** 2
        self.aft = {"port": (h.sx < self.AFT_X) & (h.side < 0), "stbd": (h.sx < self.AFT_X) & (h.side > 0)}
        self.pod_wet_area = 0.3 * float(h.tab_a[self.aft["stbd"], int(round((D["waterline_y_at_x0"] - h.h0) / h.dh))].sum())

    def thrust_max(self, u):
        """A pod loses thrust with speed: bollard at rest, 60 % of it at the top."""
        return self.t_bollard * (1.0 - 0.4 * min(max(u / self.u_top, 0.0), 1.3))

    def thrusts(self, u):
        line = self._thrust_line()
        out = []
        for side in ("port", "stbd"):
            if self.hull.drive is self:
                c = self.pod_command(side)
                t = c * (self.thrust_max(u) if c >= 0.0 else self.astern * self.t_bollard)
            else:
                t = 0.5 * self.resistance(u)
            out.append(("pod_" + side, self.pod_at[side], t * line, self.disc_r, False))
        return out

    def pod_command(self, side):
        """-1 (full astern) .. 1 (full ahead) for one pod."""
        return min(max(self.throttle + (self.DIFF_GAIN if side == "port" else -self.DIFF_GAIN) * self._diff(), -1.0), 1.0)

    def resistance(self, u):
        au = abs(u)
        return math.copysign(self.R_LIN * au + self.r_quad * au * au
                             + self.HUMP_R * math.exp(-((au - self.hump_u) / self.HUMP_W) ** 2), u)

    def loads(self, vb, w, area):
        """Hull resistance, the two pods, the squat moment, damping."""
        h = self.hull
        cog = h.cog
        u, heave, sway, wet, upright, trim_deg, roll = self._state(vb)
        L = _Loads(cog)
        L.add(np.array([-self.resistance(u) * wet, -h.c_heave * heave, 0.0]), cog)
        L.add(np.array([0.0, 0.0, -(self.KV1 * (abs(u) + 1.0) * sway + self.KV2 * sway * abs(sway)) * wet]),
              (cog[0], self.SWAY_Y, cog[2]))
        line = self._thrust_line()
        for side in ("port", "stbd"):
            c = self.pod_command(side)
            pod_wet = min(max(float(area[self.aft[side]].sum()) / self.pod_wet_area, 0.0), 1.0) * upright
            L.add(c * (self.thrust_max(u) if c >= 0.0 else self.astern * self.t_bollard) * pod_wet * line, self.pod_at[side])
        p_, r_, q_ = w                                  # roll, yaw, pitch rates (vessel X, Y, Z)
        Mb = L.M + np.array([-h.c_roll * p_,
                             -(self.KR1 * (abs(u) + 1.0) * r_ + self.KR2 * r_ * abs(r_)) * max(wet, 0.2),
                             -h.c_pitch * q_ + self.SQUAT_M * math.exp(-((abs(u) - self.hump_u) / self.HUMP_W) ** 2) * wet])
        return L.F, Mb

    def loads_3dof(self, nu_r, vb, w, area):
        """loads() for a hull on the 3-DOF manoeuvring model (MANOEUVRING): the same terms with the
        resistance, the sway and yaw damping and the pods' thrust taken on the water-relative surge,
        sway and yaw rate nu_r = (u_r, v_r, r) (body x ahead, y to starboard, r clockwise from above).
        Returns ((X, Y, N) in that frame, the vessel-frame force (heave), the vessel-frame moment
        (roll and pitch; its yaw is N))."""
        h = self.hull
        cog = h.cog
        u, v, r = (float(a) for a in nu_r)
        _, heave, _, wet, upright, trim_deg, roll = self._state(vb)
        L = _Loads(cog)
        x_res = -self.resistance(u) * wet
        L.add(np.array([0.0, 0.0, -(self.KV1 * (abs(u) + 1.0) * v + self.KV2 * v * abs(v)) * wet]),
              (cog[0], self.SWAY_Y, cog[2]))           # vessel z is starboard: sway v_r
        line = self._thrust_line()
        for side in ("port", "stbd"):
            c = self.pod_command(side)
            pod_wet = min(max(float(area[self.aft[side]].sum()) / self.pod_wet_area, 0.0), 1.0) * upright
            L.add(c * (self.thrust_max(u) if c >= 0.0 else self.astern * self.t_bollard) * pod_wet * line, self.pod_at[side])
        n_damp = -(self.KR1 * (abs(u) + 1.0) * r + self.KR2 * r * abs(r)) * max(wet, 0.2)
        p_, _, q_ = w                                   # roll and pitch rates (vessel X, Z)
        Fb = np.array([0.0, -h.c_heave * heave, 0.0])
        Mb = np.array([L.M[0] - h.c_roll * p_, 0.0,
                       L.M[2] - h.c_pitch * q_ + self.SQUAT_M * math.exp(-((abs(u) - self.hump_u) / self.HUMP_W) ** 2) * wet])
        return (x_res + L.F[0], L.F[2], n_damp - L.M[1]), Fb, Mb

    def sway_yaw(self, u):
        """The open-loop sway-yaw pair of the 3-DOF model at water-relative surge u, linearised about
        straight running with the pods amidships: (A, eigenvalues), x' = A x for x = (v_r, r)."""
        m11, m22, m66, a11, a22, _ = self.hull.mass_3dof()
        A = np.array([[-self.KV1 * (abs(u) + 1.0) / m22, -m11 * u / m22],
                      [-(a22 - a11) * u / m66, -self.KR1 * (abs(u) + 1.0) / m66]])
        return A, np.linalg.eigvals(A)

    def _spin(self, dt):
        for side in ("port", "stbd"):
            self.spin[side] = (self.spin[side] + self.ROTOR_RATE * self.pod_command(side) * dt) % (2.0 * math.pi)

    def follow(self, u, yaw_rate, dt, full_rate=0.3, lock=math.radians(22.0), spin=9.0):
        """For a hull that something else moves (a DP on a route): the throttle her speed implies,
        the rotors turned by the way made (spin rad per metre), and pods that steer turned as the
        turn implies."""
        self.throttle = min(max(u / self.u_top, self.throttle_min), 1.0)
        for side in ("port", "stbd"):
            self.spin[side] = (self.spin[side] + u * dt * spin) % (2.0 * math.pi)

    def pose(self):
        n = self.nodes
        for s in ("port", "stbd"):
            if n.get("rotor_" + s) is not None:
                n["rotor_" + s].rotation.x = self.spin[s]


class AzimuthPods(_Pods):
    """The Otter X: two rim-drive azimuth pods under the keel lines. Both steer to the helm's
    angle; diff_cmd (-1..1) moves thrust from one pod to the other and turns her on the spot.
    Tuned so full throttle makes 8 kn, an assumed figure until the vendor publishes one."""

    ROTOR_RATE = 30.0                  # rad/s of the drawn rotors at full thrust
    HUMP_W, HUMP_R = 1.0, 200.0
    R_LIN = 40.0
    SQUAT_M = 1400.0                   # bow-up moment at the hump (N m), about 1.5 deg of trim
    KV1, KV2 = 600.0, 1800.0           # sway damping: two hulls and their skegs broadside
    KR1, KR2 = 250.0, 7000.0           # yaw damping
    SWAY_Y = 0.2                       # height the lateral force acts at: mid-draft
    AFT_X = -1.5

    def __init__(self, hull):
        super().__init__(hull)
        spec = hull.spec
        th = spec["propulsion"]["thrusters"]
        duct = th["duct"]
        self._setup(th, spec["propulsion"]["demo_top_speed_kn"] * 0.5144, spec["pontoons"]["keel_z"],
                    duct["centre_xy"][0], duct["centre_xy"][1])
        self.steer_max = math.radians(th["helm_deg"])
        self.disc_r = duct["inner_r"]
        self.diff = self.diff_cmd = 0.0
        heel = min(spec["appendages"]["skeg"]["profile_xy"], key=lambda p: p[1])
        # skeg heels and duct bottoms, both sides
        self.deep_pts = np.array([[x, y, s * self.pontoon_z] for s in (-1.0, 1.0)
                                  for (x, y) in (heel, (duct["centre_xy"][0], duct["centre_xy"][1] - duct["outer_r"]))])
        self.solver_draft = hull.design["draft_max"]

    def _diff(self):
        return self.diff

    def _thrust_line(self):
        d = self.steer
        return np.array([math.cos(d), 0.0, -math.sin(d)])        # thrust line of both pods

    def tick(self, dt):
        self.throttle = slew(self.throttle, self.throttle_cmd, 0.8, dt)
        self.steer = slew(self.steer, self.steer_cmd, math.radians(60.0), dt)      # the pods slew at 60 deg/s
        self.diff = slew(self.diff, self.diff_cmd, 2.0, dt)
        self._spin(dt)

    def follow(self, u, yaw_rate, dt, full_rate=0.3, lock=math.radians(22.0), spin=9.0):
        """As _Pods.follow, and both pods hard over at full_rate rad/s. A positive heading rate is
        a turn to port, a negative pod angle."""
        super().follow(u, yaw_rate, dt, full_rate, lock, spin)
        self.steer = float(np.clip(-yaw_rate / full_rate, -1.0, 1.0)) * lock

    def pose(self):
        super().pose()
        n = self.nodes
        for s in ("port", "stbd"):
            if n.get("pod_" + s) is not None:
                n["pod_" + s].rotation.y = self.steer

    def status(self):
        return f"thr {self.throttle:+.2f}  steer {math.degrees(self.steer):+5.1f}  diff {self.diff:+.2f}"

    def _foam(self, foam, u):
        spd = min(abs(u) / self.u_top, 1.0)
        for zs in (-self.pontoon_z, self.pontoon_z):
            foam(1.3, zs, 0.5, 0.06 + 0.35 * spd)                            # bow wave off each pontoon
        line = self._thrust_line()
        for side in ("port", "stbd"):
            c = self.pod_command(side)
            wash = self.pod_at[side] - math.copysign(0.9, c) * line
            foam(wash[0], wash[2], 0.5 + 0.3 * spd, 0.5 * abs(c))            # pod wash, down the thrust line


class FixedPods(_Pods):
    """The Otter: two fixed pods under the pontoons' sterns. The helm is the thrust difference
    (steer is a fraction here, not an angle), so with the throttle at zero she turns on the
    spot. The Otter X's terms at her size, tuned so full throttle makes the vendor's 4.5 kn."""

    ROTOR_RATE = 60.0                  # rad/s of the drawn propellers at full thrust
    HUMP_W, HUMP_R = 0.5, 15.0
    R_LIN = 5.0
    SQUAT_M = 15.0                     # bow-up moment at the hump (N m), about half a degree of trim; the low thrust line adds more
    KV1, KV2 = 40.0, 330.0             # sway damping: two pontoons broadside
    KR1, KR2 = 6.0, 80.0               # yaw damping
    SWAY_Y = 0.1                       # height the lateral force acts at: mid-draft
    AFT_X = -0.6

    def __init__(self, hull):
        super().__init__(hull)
        spec = hull.spec
        th = spec["propulsion"]["thrusters"]
        self._setup(th, spec["propulsion"]["top_speed_kn"] * 0.5144, spec["frame_tubes"]["aft"]["arm_z"],
                    th["strut_x"] + th["rotor"]["x"], th["pod_y"])
        self.steer_max = 1.0
        self.disc_r = th["rotor"]["blade_tip_r"]
        guard = th["guard"]
        low = min(y for (_, y) in guard["path_xy"])
        # the thruster guards' feet, both sides
        self.deep_pts = np.array([[x, y - guard["radius"], s * self.pontoon_z] for s in (-1.0, 1.0)
                                  for (x, y) in guard["path_xy"] if y == low])
        # the tables hold the pontoons alone, so the solver's own draft stops at the keel line;
        # this is its waterline taken down to the guards
        D = hull.design
        self.solver_draft = float((D["waterline_y_at_x0"] - math.tan(math.radians(D["trim_deg"])) * self.deep_pts[:, 0]
                                   - self.deep_pts[:, 1]).max())

    def _diff(self):
        return self.steer

    def _thrust_line(self):
        return np.array([1.0, 0.0, 0.0])

    def tick(self, dt):
        self.throttle = slew(self.throttle, self.throttle_cmd, 0.8, dt)
        self.steer = slew(self.steer, self.steer_cmd, 2.0, dt)
        self._spin(dt)

    def status(self):
        return f"thr {self.throttle:+.2f}  port {self.pod_command('port'):+.2f}  stbd {self.pod_command('stbd'):+.2f}"

    def _foam(self, foam, u):
        spd = min(abs(u) / self.u_top, 1.0)
        for side, zs in (("port", -self.pontoon_z), ("stbd", self.pontoon_z)):
            foam(0.6, zs, 0.25, 0.05 + 0.25 * spd)                           # bow wave off each pontoon
            c = self.pod_command(side)
            foam(self.pod_at[side][0] - math.copysign(0.4, c), zs, 0.25 + 0.1 * spd, 0.4 * abs(c))   # propeller wash


DRIVES = {"mariner": Waterjet, "otterx": AzimuthPods, "otter": FixedPods}


def drive_for(hull):
    """The drive of hull's boat. It moves her once it is hull.drive; left off the hull (a boat
    on a DP) it still binds, follows and poses her actuator nodes."""
    if hull.boat not in DRIVES:
        raise KeyError(f"no drive for '{hull.boat}' (usv_rig.DRIVES has {', '.join(DRIVES)})")
    return DRIVES[hull.boat](hull)


# --------------------------------------------------------------------------- #
#  The sea around her: displacement footprint and Kelvin wake
# --------------------------------------------------------------------------- #
class Wake:
    """A boat's footprint and wake trail on the ocean.

    target             the ocean (its vessel 0), or ocean.vessel(i) where several boats share it
    floor, gain, cap   the speed handed to the ocean's analytic wake: floor + gain |u|, capped.
                       The wake gates its foam trail on smoothstep(0.5, 1.5, speed) and spreads
                       it over 1.5 x the footprint's half-beam (foam_world.comp), so fed a
                       boat's true speed it is a white carpet three beams wide from 3 kn up.
    max_age, max_samples   the trail: seconds a sample lives, and how many are kept"""

    def __init__(self, hull, target, floor=0.6, gain=0.10, cap=1.2, max_age=8.0, max_samples=64):
        self.hull, self.target = hull, target
        self.floor, self.gain, self.cap = floor, gain, cap
        self.max_age, self.max_samples = max_age, max_samples
        self.accum, self.last = 0.0, None
        self.centre = None                     # the footprint's centre at the last update (world)

    def speed(self, u):
        return math.copysign(min(self.floor + self.gain * abs(u), self.cap), u) if abs(u) > 0.2 else 0.0

    def update(self, dt, on=True):
        """After the hull's advance: her footprint, the wake's speed, and a trail sample every
        0.1 s or metre made. on=False takes the footprint and the wake away. Returns her surge speed."""
        h, V = self.hull, self.target
        c, yaw_ex = h.footprint(V, on)
        self.centre = c
        u = float(h.R[:, 0] @ h.v)
        ws = self.speed(u)
        V.wake.forward_speed = ws if on else 0.0
        if on and dt > 0.0:
            V.age_wake(dt, self.max_age, self.max_samples)
            self.accum += dt
            moved = 1e9 if self.last is None else math.hypot(c[0] - self.last[0], c[2] - self.last[1])
            if abs(u) > 0.4 and (self.accum >= 0.1 or moved >= 1.0):
                self.accum, self.last = 0.0, (c[0], c[2])
                V.add_wake_sample(float(c[0]), float(c[2]), math.sin(yaw_ex), math.cos(yaw_ex), float(ws),
                                  self.max_samples)
        return u

    def clear(self):
        self.target.clear_wake()
        self.accum, self.last = 0.0, None


# --------------------------------------------------------------------------- #
#  What she leaves ON the water: her races and her lane, in the ocean's wake field
# --------------------------------------------------------------------------- #
def race(thrust, advance, area, jet=False, rho=1025.0):
    """The speed of a propulsor's race THROUGH the water (m/s) from momentum theory: what a
    thrust (N) takes from a disc or a nozzle of `area` (m2) that the water meets at `advance`
    (m/s). A propeller is an actuator disc, T = 2 rho A (V + v) v, and its race far astern runs
    2 v faster than the water round it; a jet takes water in at V and throws it out at Vj,
    T = rho A Vj (Vj - V), and its race is Vj - V."""
    t, v = abs(thrust), max(advance, 0.0)
    if t <= 0.0 or area <= 0.0:
        return 0.0
    if jet:
        return 0.5 * (v + math.sqrt(v * v + 4.0 * t / (rho * area))) - v
    return math.sqrt(v * v + 2.0 * t / (rho * area)) - v


class Wash:
    """What a boat leaves on the water behind her: a patch of the ocean's wake field that
    trails her, and her sources in it. Two kinds of producer, and neither knows the other:

      her propulsors   drive.thrusts(u). Each one's race is taken from its thrust (race()), run
                       down its own line until it has widened to the surface (it widens by
                       SPREAD of the way it has run; the boat has moved on meanwhile), and put
                       there as a source: the water moving at what is left of the race's speed,
                       turbulence a share of that, and air by how hard the race is for how
                       little water it has over it (w^2 / g d, a Froude number on its
                       submergence): none under about 1, bubbles above, a white film well above,
                       and a race that hard breaks the surface wider than it came up.
                       A second, fainter source lies over the disc itself: the race's air seen
                       through the water it has not yet come up through, so the mark starts at
                       her stern. A race is not steady, and where it surfaces wanders a part of
                       its own width (WANDER).
      her hulls        the turbulent water a hull drags astern: a lane a hull's breadth wide
                       from each stern, with no air in it until she is fast for her length.
                       And her WAVES, where the wake field has ripples
                       (ocean.wake_field.ripple_resolution): she bears down on the water with her
                       weight, laid out as her hulls carry it (the Bonjean strips at her design
                       waterline: a stretch of hull's share of the buoyancy, where its centroid
                       is, as wide as it is at the water, so a fine bow is a narrow load and a
                       catamaran is two), and the field's small linear sea makes of that the
                       waves a weight of that shape makes at her speed. No wave is drawn: at
                       rest she makes none, at speed the fan has the angle and the wavelengths
                       of that speed, and it bends where she turned.

    hull, drive   the boat; the drive need not be the one that moves her (see Drive.thrusts),
                  and None is a hull with no propulsor (a kayak: her paddle is put() by the scene)
    ocean         its wake field must be on (ocean.wake_field.resolution, before the first
                  render); on a sea that has none (GL) a Wash does nothing
    index         the patch of the wake field that is hers
    size          the patch's side (m): how much water astern she keeps. The patch has the
                  field's resolution whatever its size, so this also sets how fine her mark is
                  drawn (default 20 lengths of her, between 30 and 120 m)
    air           scales the air her propulsors take down (1 = as computed)
    ripples       the side (m) of the window her waves are kept in (default 10 lengths of her,
                  at least 16 m and at most her patch): the smaller, the finer they are drawn

    The scene clears the ocean's sources once a frame (ocean.clear_wake_sources()) and then
    updates every boat's Wash."""

    SPREAD = 0.3          # a race's radius grows by this much of the distance it has run
    TURBULENCE = 0.45     # rms turbulent speed of a race where it surfaces, as a share of its speed there
    EDDY = 2.4            # her patch's large eddies, in radii of her widest race at the surface
    WANDER = 0.45         # how far a race's surfacing point strays to either side, in its radii there
    STRIPS = 8            # stretches a side her weight is laid out in along her hulls
    G = 9.81

    def __init__(self, hull, drive, ocean, index, size=None, air=1.0, ripples=None):
        self.hull, self.drive, self.ocean = hull, drive, ocean
        self.patch = ocean.wake_patch(index) if hasattr(ocean, "wake_patch") else None
        self.index = index
        self.size = size or min(max(20.0 * hull.spec["principal"]["loa"], 30.0), 120.0)
        self.air = air
        self.ripple_size = min(self.size, ripples or max(10.0 * hull.spec["principal"]["loa"], 16.0))
        self.ripples = False                   # the wake field has ripples (read at each update)
        self.strips = None                     # her weight along her hulls (weight_strips(), once)
        self.last = {}                         # each producer's source point at the last update (world x, z)
        self.report = {}                       # each producer's last figures, for a readout
        self.t = 0.0
        self.put_names = set()                 # the scene's own producers that put() since the last update

    def clear(self):
        """Empty her patch and forget where her producers were (after she is seated elsewhere)."""
        if self.patch is not None:
            self.patch.size = 0.0
        self.last = {}

    def put(self, name, at, vel, radius, foam=0.0, aeration=0.0, turbulence=0.0, lane=0.0, push=0.0):
        """A source of any other producer of hers (a paddle's blade, an anchor going down): `at` a
        point and `vel` the water velocity it imparts over the ground, both in the vessel frame;
        `push` the force (N) it bears down on the water with, which is what makes waves.
        After update(), every frame the producer is at work; a frame without it ends its trail."""
        if self.patch is None or not self.patch.size > 0.0:
            return
        h = self.hull
        q = h.to_world(at)
        v = h.R @ np.asarray(vel, float)
        texel = self.size / max(self.ocean.wake_field.resolution, 1)
        self._put("put:" + name, q[[0, 2]], v[[0, 2]], max(radius, 1.25 * texel), foam, aeration, turbulence, lane,
                  push)
        self.put_names.add("put:" + name)

    def weight_strips(self):
        """Her weight along her hulls at the design waterline, from the Bonjean strips: a list of
        (x, z, share of her weight, radius, half length) in the vessel frame, starboard and port,
        at most STRIPS stretches a side. A stretch's radius is that of the gaussian with the
        spread of a load as wide as the hull is at the water there."""
        h = self.hull
        t = min(max((h.design["waterline_y_at_x0"] - h.h0) / h.dh, 0.0), h.nh - 1.001)
        i, f = int(t), t - int(t)
        ns = len(h.sx) // 2                                      # the starboard halves come first
        a = h.tab_a[:ns, i] + (h.tab_a[:ns, i + 1] - h.tab_a[:ns, i]) * f
        zc = h.tab_z[:ns, i] + (h.tab_z[:ns, i + 1] - h.tab_z[:ns, i]) * f
        b = (h.tab_a[:ns, i + 1] - h.tab_a[:ns, i]) / h.dh       # dA/dh: the half-section's breadth at the water
        wet = np.flatnonzero(a > 1e-4 * float(a.max()))
        out = []
        for g in np.array_split(wet, min(self.STRIPS, len(wet))):
            w = float(a[g].sum())
            x = 0.5 * float(h.sx[g[0]] + h.sx[g[-1]])            # the stretches tile her length
            z = float(zc[g] @ a[g]) / w
            r = 0.41 * float(b[g] @ a[g]) / w
            for side in (1.0, -1.0):
                out.append((x, side * z, 0.5 * w / float(a[wet].sum()), r, 0.5 * h.dx * len(g)))
        return out

    def _put(self, name, p, vel, radius, foam, aeration, turbulence, lane, push=0.0, axis=(0.0, 0.0)):
        x0, z0 = self.last.get(name, (p[0], p[1]))
        if math.hypot(p[0] - x0, p[1] - z0) > 0.5 * self.size:
            x0, z0 = p[0], p[1]                # she was moved, not sailed
        if push and self.ripples:
            self.ocean.add_wake_source(float(p[0]), float(p[1]), float(vel[0]), float(vel[1]), float(radius),
                                       float(foam), float(aeration), float(turbulence), float(lane), float(x0), float(z0),
                                       push=float(push), ax=float(axis[0]), az=float(axis[1]), patch=self.index)
        else:
            self.ocean.add_wake_source(float(p[0]), float(p[1]), float(vel[0]), float(vel[1]), float(radius),
                                       float(foam), float(aeration), float(turbulence), float(lane), float(x0), float(z0))
        self.last[name] = (float(p[0]), float(p[1]))

    def update(self, dt, on=True):
        """After the hull's advance: move her patch and add this frame's sources."""
        if self.patch is None:
            return
        if not on:
            self.clear()
            return
        h = self.hull
        fwd = np.array([h.R[0, 0], h.R[2, 0]])
        fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
        loa = h.spec["principal"]["loa"]
        # she sits a length inside its leading edge; the rest of it lies astern
        c = h.p[[0, 2]] - fwd * (0.5 * self.size - 0.08 * self.size - loa)
        self.patch.set(float(c[0]), float(c[1]), float(self.size))
        texel = self.size / max(self.ocean.wake_field.resolution, 1)
        self.ripples = getattr(self.ocean.wake_field, "ripple_resolution", 0) > 0
        if self.ripples:                       # her waves' window: she sits a fifth of it inside its leading edge
            cr = h.p[[0, 2]] - fwd * (0.3 * self.ripple_size)
            self.patch.set_ripples(float(cr[0]), float(cr[1]), float(self.ripple_size))
        self.t += dt
        for name in [n for n in self.last if n.startswith("put:") and n not in self.put_names]:
            del self.last[name]                # it was not put last frame: its trail has ended
        self.put_names = set()
        u = float(h.R[:, 0] @ h.v)
        vb = h.v[[0, 2]]
        widest = 0.0

        for k, (name, at, force, r0, jet) in enumerate(self.drive.thrusts(u) if self.drive is not None else ()):
            t = float(np.linalg.norm(force))
            if t < 1e-3:
                self.last.pop(name, None)
                self.last.pop(name + "_disc", None)
                continue
            line = -(h.R @ (np.asarray(force, float) / t))       # the race runs against the force (world)
            d = line[[0, 2]]
            d /= max(float(np.linalg.norm(d)), 1e-9)
            q = h.to_world(at)
            depth = float(self.ocean.sample_height(float(q[0]), float(q[2]), h.mask)) - float(q[1])
            if depth < -r0:                                      # out of the water: it draws air and makes no race
                self.last.pop(name, None)
                self.last.pop(name + "_disc", None)
                continue
            w = race(t, -float(vb @ d), math.pi * r0 * r0, jet)
            if w < 0.02:
                self.last.pop(name, None)
                self.last.pop(name + "_disc", None)
                continue
            cover = max(depth - r0, 0.0)                         # water over the top of the race where it starts
            run = cover / self.SPREAD                            # how far it runs before it has widened to the surface
            rs = r0 + self.SPREAD * run
            ws = w * r0 / rs                                     # momentum kept, spread over the wider race
            ts = run / max(0.5 * (w + ws), 1e-3)
            p = q[[0, 2]] - vb * ts + d * run                    # where that water is when it gets there
            a, b = 1.7 * self.t + 2.4 * k, 3.1 * self.t + 4.1 * k
            p = p + np.array([-d[1], d[0]]) * (self.WANDER * rs * (math.sin(a) + 0.6 * math.sin(b)) / 1.6)
            fr = self.air * w * w / (self.G * max(cover, 0.02))
            aer = 1.0 - math.exp(-(fr / 2.5) ** 2)
            foam = 1.0 - math.exp(-max(fr - 1.5, 0.0) ** 2 / 20.0)
            turb = self.TURBULENCE * ws
            rs *= 1.0 + 0.25 * min(math.sqrt(fr), 6.0)           # a hard race breaks the surface wider than it arrived
            widest = max(widest, rs)
            self._put(name, p, d * ws, max(rs, 1.25 * texel), foam, aer, turb, min(turb / 0.05, 1.0))
            if run > 2.0 * r0:                                   # over the disc: its air, through the water above it
                self._put(name + "_disc", q[[0, 2]], d * (0.4 * w), max(1.3 * r0, 1.25 * texel), 0.0,
                          0.6 * aer * math.exp(-1.2 * cover), 0.0, 0.0)
            else:
                self.last.pop(name + "_disc", None)
            self.report[name] = {"thrust": t, "race": w, "at_surface": ws, "run": run, "froude": fr,
                                 "aeration": aer, "foam": foam}

        # her hulls: the water she drags, a lane from each stern
        hl, hb, xc = h.excl
        zs = getattr(self.drive, "pontoon_z", 0.0)
        rh = max(hb - zs, 0.1)                                   # a hull's half-breadth at the water
        self.patch.eddy = float(max(self.EDDY * max(widest, rh), 4.0 * texel))
        fn = abs(u) / math.sqrt(self.G * 2.0 * hl)
        lane = smoothstep(0.15, 0.6, abs(u))
        if lane > 0.0:
            air = smoothstep(0.35, 0.85, fn)                     # a transom runs dry and white when she is fast for her length
            for k, z in enumerate((-zs, zs) if zs > 0.0 else (0.0,)):
                q = h.to_world([xc - math.copysign(hl, u), h.design["waterline_y_at_x0"], z])
                self._put(f"hull_{k}", q[[0, 2]], 0.15 * vb, max(rh, 1.25 * texel), smoothstep(0.45, 1.05, fn) * air,
                          air, 0.07 * abs(u), lane)
        else:
            for k in range(2):
                self.last.pop(f"hull_{k}", None)

        # her weight on the water, laid out along her hulls: what makes her waves
        if self.ripples:
            if self.strips is None:
                self.strips = self.weight_strips()
            wl = h.design["waterline_y_at_x0"]
            for k, (x, z, share, r, half) in enumerate(self.strips):
                q = h.to_world([x, wl, z])
                self._put(f"weight_{k}", q[[0, 2]], (0.0, 0.0), r, 0.0, 0.0, 0.0, 0.0,
                          push=h.mass * self.G * share, axis=fwd * half)
