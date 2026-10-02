"""
usv_rig -- a boat on the FFT ocean: her strips, her drive, her mark on the sea.

Three things, and nothing scene-specific:

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

    # on a DP, along a route                    # under her own power
    drive = drive_for(hull).bind(glb_root)      hull.drive = drive_for(hull).bind(glb_root)
    hull.set_target(x, z, hd, vx, vz, rate)     hull.drive.throttle_cmd = 1.0
    hull.advance(dt, substeps)                  hull.drive.tick(dt); hull.advance(dt, substeps)
    u = wake.update(dt)                         u = wake.update(dt)
    drive.follow(u, rate, dt)
    hull.place(glb_root); drive.pose()          hull.place(glb_root); hull.drive.pose()

PER-BOAT NUMBERS
----------------
The spec's ``strip_model`` block holds what the strip model needs beyond the tables: where each
half's water level is read (``probe_z``) and the ocean's hull footprint. A spec without one
gets both from its design waterline. The drives read the spec's ``propulsion`` block; their
lumped resistance and damping figures are tuned per boat and stand in the classes below.
"""
import json
import math
import os

import numpy as np

USV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usv")

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

    def step(self, dt):
        """One substep: strip buoyancy + the DP's or the drive's loads, semi-implicit Euler."""
        R, v, w = self.R, self.v, self.w
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
