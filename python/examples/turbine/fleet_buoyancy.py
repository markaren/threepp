"""Strip buoyancy for the two sea drones, one instance per boat, and a soft DP hold.

Lifted from ../usv/usv_ocean.py (module-level there): the Bonjean half-section tables sampled
under each station on both sides against ocean.sample_height, the mass/inertia setup, the
hydrostatic loads and the rigid-body step. Propulsion, planing and wakes are left out.
Station keeping: heave, pitch and roll are free; surge, sway and yaw are held by a PD toward a
target pose (a DP system), so the hull rides the waves but stays put.

Vessel frame (the spec's): X forward, Y up, Z starboard, origin on the baseline amidships.
"""
import json
import math
import os

import numpy as np

USV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "usv")

BUOY_MASK = 0b011                  # swell + mid band; cascade 2 is chop the hull ignores (usv_ocean)
PROBE_Z = {"mariner": 0.55, "otterx": 0.87}         # where each half's water level is read (usv_ocean)
# ocean hull footprint (usv_ocean): half length, half beam, waterplane centre x
EXCL = {"mariner": (2.95, 0.85, -0.1), "otterx": (2.0, 1.0, 0.0)}
# the low points a draft is measured to (usv_ocean DEEP_PTS, simplified to the lowest y)


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
    """(x, y, z, w) of a rotation matrix (usv_ocean)."""
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
    """Heading, pitch (bow up +) and roll (starboard down +), radians (usv_ocean)."""
    return (math.atan2(-R[2, 0], R[0, 0]), math.asin(max(-1.0, min(1.0, R[1, 0]))),
            math.atan2(-R[1, 2], R[1, 1]))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class StripHull:
    """One boat on its Bonjean strips. heading a: vessel +X points along world (cos a, 0, -sin a).

    Keyword-only, for boats other than the two drones (defaults keep the drones as they were):
      spec_dir   folder holding <boat>_spec.json and <boat>_hydro.json (default USV_DIR)
      probe_z    half-breadth where each side's water is read (default PROBE_Z[boat], else
                 0.314 x the design waterline beam: the strip model heels the half-sections by
                 the level change AT probe_z, so probe_z sets its roll stiffness, and 0.314 bwl
                 reproduces the sjark's GM_T 1.39 m (sjark_hydro.json) under a known moment)
      excl       ocean hull footprint (half length, half beam, waterplane centre x) (default
                 EXCL[boat], else half the design waterline length, 0.42 x its beam, its LCF)"""

    def __init__(self, boat, ocean, load=None, *, spec_dir=None, probe_z=None, excl=None):
        self.boat, self.ocean = boat, ocean
        spec_dir = spec_dir or USV_DIR
        with open(os.path.join(spec_dir, f"{boat}_spec.json"), encoding="utf-8") as fh:
            spec = json.load(fh)
        with open(os.path.join(spec_dir, f"{boat}_hydro.json"), encoding="utf-8") as fh:
            hydro = json.load(fh)
        self.spec = spec
        load = load or spec["mass"]["design_condition"]
        self.rho, self.g = hydro["rho"], hydro["g"]
        items = spec["mass"]["budget"] + spec["mass"]["conditions"][load]["extra"]
        m = sum(b["mass"] for b in items)
        self.mass = m
        self.cog = np.array([sum(b["mass"] * b["centroid"][i] for b in items) / m for i in range(3)])
        self.design = D = hydro["conditions"][load]
        gyr = spec["mass"]["gyradius"]
        # added mass / inertia fractions (surge, heave, sway), (roll, yaw, pitch): usv_ocean's rough figures
        self.m_eff = m * (1.0 + np.array([0.05, 0.80, 0.60]))
        self.i_eff = m * np.array([gyr["roll"], gyr["yaw"], gyr["pitch"]]) ** 2 * (1.0 + np.array([0.25, 0.30, 0.60]))
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
        if probe_z is None:
            probe_z = PROBE_Z[boat] if boat in PROBE_Z else 0.314 * D["bwl"]
        if excl is None:
            excl = EXCL[boat] if boat in EXCL else (0.5 * D["lwl"], 0.42 * D["bwl"], D["lcf"])
        self.probe_z, self.excl = probe_z, tuple(excl)
        self.lowest = spec["principal"]["lowest_point_y"]
        # damping from the design hydrostatics, a fraction of critical per axis (usv_ocean)
        rg = self.rho * self.g
        k_heave = rg * D["waterplane_area"]
        k_roll = rg * D["displacement_m3"] * D["gm_t"]
        k_pitch = rg * D["displacement_m3"] * D["gm_l"]
        self.c_heave = 2.0 * 0.35 * math.sqrt(k_heave * self.m_eff[1])
        self.c_roll = 2.0 * 0.15 * math.sqrt(k_roll * self.i_eff[0])
        self.c_pitch = 2.0 * 0.45 * math.sqrt(k_pitch * self.i_eff[2])
        # DP hold (assumed): ~0.35 rad/s horizontal, ~0.5 rad/s yaw, near critical
        self.wn_xy, self.wn_yaw, self.zeta = 0.35, 0.5, 0.9
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.w = np.zeros(3)
        self.water = np.zeros(2 * ns)
        self.buoy = 0.0
        self.target = (0.0, 0.0, 0.0)          # x, z, heading
        self.target_v = (0.0, 0.0, 0.0)        # vx, vz, heading rate
        self.target_a = (0.0, 0.0)             # feed-forward horizontal acceleration
        self.log = []

    # ---- the sea under her
    def probe_points(self):
        rel = np.stack([self.sx - self.cog[0], np.zeros(len(self.sx)), self.side * self.probe_z - self.cog[2]], axis=1)
        return self.p + rel @ self.R.T

    def sample_water(self):
        pts = self.probe_points()
        self.water = np.array([self.ocean.sample_height(float(x), float(z), BUOY_MASK) for x, _, z in pts])
        return pts

    def seat(self, x, z, heading):
        """Drop her on her design waterline and trim at (x, z), heading, at rest (usv_ocean seat_on_water)."""
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

    # ---- one substep: strip buoyancy + damping + DP, semi-implicit Euler (usv_ocean step)
    def step(self, dt):
        R, v, w = self.R, self.v, self.w
        cog, sx, side = self.cog, self.sx, self.side
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
        tau = np.array([-float(arm[:, 2] @ fb), 0.0, float(arm[:, 0] @ fb)])
        self.buoy = float(fb.sum())
        # DP: horizontal PD toward the target (world), with the target's velocity and acceleration fed forward
        tx, tz, th = self.target
        tvx, tvz, trate = self.target_v
        wn, ze = self.wn_xy, self.zeta
        acc = np.array([self.target_a[0] + wn * wn * (tx - self.p[0]) + 2 * ze * wn * (tvx - v[0]), 0.0,
                        self.target_a[1] + wn * wn * (tz - self.p[2]) + 2 * ze * wn * (tvz - v[2])])
        F_dp = self.mass * 1.3 * acc
        hd = attitude(R)[0]
        wy = self.wn_yaw
        M_yaw = self.i_eff[1] * (wy * wy * wrap(th - hd) + 2 * ze * wy * (trate - w[1]))
        # vessel-frame loads: heave/roll/pitch damping (usv_ocean), the DP force and yaw moment
        vb = R.T @ v
        Fb = np.array([0.0, -self.c_heave * vb[1], 0.0]) + R.T @ F_dp
        Mb = np.array([-self.c_roll * w[0], M_yaw, -self.c_pitch * w[2]])
        Fv = R.T @ F + Fb
        tv = R.T @ tau + Mb
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

    def footprint(self, ocean):
        """Give the ocean's one hull footprint to this boat (usv_ocean sea_update)."""
        hl, hb, xc = self.excl
        R = self.R
        fwd = R[:, 0]
        yaw_ex = math.atan2(fwd[0], fwd[2])
        s_ex = np.array([math.cos(yaw_ex), -math.sin(yaw_ex)])
        c = self.to_world([xc, 0.0, 0.0])
        pts = self.probe_points()
        A = np.stack([np.ones(len(pts)), pts[:, 0] - c[0], pts[:, 2] - c[2]], axis=1)
        h0, gx, gz = np.linalg.lstsq(A, self.water, rcond=None)[0]
        ocean.hull_exclusion.set_pose(float(c[0]), float(c[2]), float(h0), yaw=yaw_ex,
                                      pitch=math.atan(gx * fwd[0] + gz * fwd[2]),
                                      roll=math.atan(gx * s_ex[0] + gz * s_ex[1]),
                                      half_length=hl, half_beam=hb)

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
