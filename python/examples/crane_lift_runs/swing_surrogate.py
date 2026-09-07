"""Control-design surrogate for the offshore crane lift (E3 of the ICRA 2027 paper).

No renderer, no assets: the pendulum copied verbatim from crane_lift.py, the tip as a
first-order servo in polar coordinates about the king, and the vessel/IK residual replayed
from the recorded law-off run.  Used to design the replacement anti-swing law.

    python swing_surrogate.py                 # validate, sweep, robustness, PNG
    python swing_surrogate.py --stage validate
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "round3")
STAGE = "all"
for i, a in enumerate(sys.argv):
    if a == "--stage" and i + 1 < len(sys.argv):
        STAGE = sys.argv[i + 1]

# ---- constants copied from crane_lift.py -------------------------------------
FPS = 60
DT = 1.0 / FPS
VESSEL_Y0 = -3.0
DECK_Y = 15.6 + VESSEL_Y0                       # 12.6
CRANE_ON_VESSEL = (6.0, DECK_Y - 8.0, 25.0)
TURBINE_POS = (34.0, 0.0, 27.0)
PLATFORM = np.array([TURBINE_POS[0] - 9.04, 20.3, TURBINE_POS[2] - 3.18])
PICKUP = np.array([-6.0, DECK_Y, 36.0])
SLING_H = 2.6
CONT_H = 3.11
LOAD_DROP = SLING_H + CONT_H                    # 5.71
KING_XZ = np.array([CRANE_ON_VESSEL[0], CRANE_ON_VESSEL[2]])
wire0 = 7.0
PEND_DAMP = 0.08
PEND_SUB = 4
TAU = 0.1
G = 9.81

# legacy law
AS_KP, AS_KD = 0.30, 0.40
AS_CLAMP, AS_CUTOUT = 1.2, 3.0

# the tip servo in polar coordinates about the king: [theta, r, y].
# V/A are the C25 limits as crane_lift.py applies them to the joints.  The azimuth lag is
# NOT the 0.1 s drive lag alone: the recorded tip trails its target twice as far in azimuth
# as a 0.1 s lag can explain (the IK's six damped Gauss-Newton iterations do not converge
# each frame, and the king axis itself translates with the vessel), so S_TAU[0] is a lumped
# azimuth lag fitted to the recorded transfer.  See validate(), V2.
S_VMAX = np.array([0.0838, 2.5, 1.0])
S_AMAX = np.array([0.0838, 2.5, 1.0])
S_TAU = np.array([0.50, 0.10, 0.10])

SENSOR_EVERY = 3                                # 20 Hz on a 60 Hz clock
DT_SENSOR = SENSOR_EVERY * DT
SENSOR_SIGMA = 0.003                            # centroid of ~100 returns at 20 mm each


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def op_target(t):
    """Nominal tip target in the world and the wire length, as a function of simulation time."""
    hover_up = 2.0
    A = PICKUP + np.array([0.0, wire0 + LOAD_DROP, 0.0])            # tip above the pickup, the container on the deck
    B = PLATFORM + np.array([0.0, hover_up + LOAD_DROP, 0.0])       # tip above the platform
    if t < 4.0:
        return A, wire0
    if t < 44.0:                                    # the transfer: a slew ROUND the king at the boom's reach, not a line over it
        u = smooth((t - 4.0) / 40.0)
        ra, rb = A[[0, 2]] - KING_XZ, B[[0, 2]] - KING_XZ
        tha, thb = math.atan2(ra[1], ra[0]), math.atan2(rb[1], rb[0])
        dth = (thb - tha + math.pi) % (2.0 * math.pi) - math.pi
        th = tha + dth * u
        r = float(np.linalg.norm(ra)) * (1.0 - u) + float(np.linalg.norm(rb)) * u
        p = np.array([KING_XZ[0] + r * math.cos(th), A[1] * (1.0 - u) + B[1] * u, KING_XZ[1] + r * math.sin(th)])
        p[1] += 1.0 * math.sin(math.pi * u)         # clear the bulwark
        return p, wire0
    if t < 56.0:                                    # pay out until the container hovers 1 m over the grating
        u = smooth((t - 44.0) / 12.0)
        return B, wire0 + (hover_up - 1.0) * u
    return B, wire0 + hover_up - 1.0


# ---- the payload: the spherical pendulum, copied verbatim ---------------------
class Pendulum:
    def __init__(self, p0, v0=None):
        self.p = np.array(p0, np.float64)
        self.v = np.zeros(3) if v0 is None else np.array(v0, np.float64)

    def step(self, tip, dt, wire_len):
        """Position-based: gravity, then the wire as a DISTANCE constraint (it can go slack, never push)."""
        h = dt / PEND_SUB
        p, v = self.p, self.v
        for _ in range(PEND_SUB):
            v = v * (1.0 - PEND_DAMP * h) + np.array([0.0, -9.81, 0.0]) * h
            p_new = p + v * h
            d = p_new - tip
            n = float(np.linalg.norm(d))
            if n > wire_len:                          # taut: pull the hook back onto the sphere; slack: fall free
                p_new = tip + d * (wire_len / n)
            v = (p_new - p) / h
            p = p_new
        self.p, self.v = p, v
        return p


# ---- the tip as a position servo in polar coordinates about the king ----------
def to_polar(p):
    dx, dz = p[0] - KING_XZ[0], p[2] - KING_XZ[1]
    return np.array([math.atan2(dz, dx), math.hypot(dx, dz), p[1]])


def to_cart(qq):
    return np.array([KING_XZ[0] + qq[1] * math.cos(qq[0]), qq[2], KING_XZ[1] + qq[1] * math.sin(qq[0])])


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Servo:
    """Mirrors actuate(): first-order lag TAU, then the rate limit, then the acceleration limit."""

    def __init__(self, tip0):
        self.q = to_polar(tip0)
        self.qf = self.q.copy()
        self.v = np.zeros(3)
        self.sat = self.sat_rate = self.sat_acc = False

    def step(self, cmd, dt):
        qc = self.q.copy()
        qc[0] = self.q[0] + wrap(to_polar(cmd)[0] - self.q[0])       # unwrapped slew
        qc[1] = math.hypot(cmd[0] - KING_XZ[0], cmd[2] - KING_XZ[1])
        qc[2] = cmd[1]
        self.qf += (qc - self.qf) * np.minimum(1.0, dt / S_TAU)
        v_raw = (self.qf - self.q) / dt
        v_des = np.clip(v_raw, -S_VMAX, S_VMAX)
        a_raw = (v_des - self.v) / dt
        acc = np.clip(a_raw, -S_AMAX, S_AMAX)
        self.sat_rate = bool(np.any(np.abs(v_raw) > S_VMAX * (1.0 + 1e-9)))
        self.sat_acc = bool(np.any(np.abs(a_raw) > S_AMAX * (1.0 + 1e-9)))
        self.sat = self.sat_rate or self.sat_acc
        self.v = self.v + acc * dt
        self.q = self.q + self.v * dt
        return to_cart(self.q)


# ---- the ZV input shaper on the polar target ---------------------------------
class ZVShaper:
    """Two impulses, 0.5 at t and 0.5 at t + T/2, on the target in polar coordinates about the king.

    Fed the target in (theta_unwrapped, r, y); the caller keeps theta continuous.
    """

    def __init__(self, q0, dt):
        self.dt = dt
        self.buf = [np.array(q0, np.float64)]
        self.max_n = int(round(6.0 / dt))

    def shape(self, q_now, wire_len):
        self.buf.append(np.array(q_now, np.float64))
        if len(self.buf) > self.max_n:
            self.buf.pop(0)
        T = 2.0 * math.pi * math.sqrt(max(wire_len, 1e-3) / G)
        k = int(round(0.5 * T / self.dt))
        j = max(0, len(self.buf) - 1 - k)
        return 0.5 * self.buf[-1] + 0.5 * self.buf[j]


# ---- the new law -------------------------------------------------------------
class IntegralAntiswing:
    """u' = k s - gamma u, integrated per sensor update and held between them; |u| <= UMAX.

    aw=True adds conditional-integration anti-windup: while the tip servo is at a rate or
    acceleration limit the correction may shrink but not grow, so u does not run away against
    an actuator that cannot execute it (the transfer saturates the slew for a third of its length).
    """

    UMAX = 1.5

    def __init__(self, k, gamma, aw=False):
        self.k, self.gamma, self.aw = k, gamma, aw
        self.u = np.zeros(2)
        self.sat = False

    def update(self, s_est, have_est, dt_sensor):
        if have_est:
            u_new = self.u + (self.k * np.asarray(s_est, np.float64) - self.gamma * self.u) * dt_sensor
            n = float(np.linalg.norm(u_new))
            if n > self.UMAX:
                u_new = u_new * (self.UMAX / n)
            if not (self.aw and self.sat and n > float(np.linalg.norm(self.u))):
                self.u = u_new
        return self.u


# ---- the surrogate -----------------------------------------------------------
def load(name):
    return np.load(os.path.join(RUNS, name))["log"]


LOG_OFF = load("op_s0_noas.npz")
LOG_ON = load("op_s0_a.npz")
N = len(LOG_OFF)
T_LOG = LOG_OFF[:, 0]
TIP_LOG = LOG_OFF[:, 1:4]
HOOK_LOG = LOG_OFF[:, 4:7]
WIRE_LOG = LOG_OFF[:, 10]
TGT_LOG = LOG_OFF[:, 14:17]
PEND_P0 = TIP_LOG[0] + np.array([0.0, -5.0, 0.0])   # first_frames() seeds the hook with wire_len still 5 m


def servo_only(targets, tip0):
    """The servo fed a target sequence, no residual: the tip it produces."""
    sv = Servo(tip0)
    out = np.zeros((len(targets), 3))
    for i, tg in enumerate(targets):
        out[i] = sv.step(tg, DT)
    return out


def residual():
    """d(t) = tip_logged - servo(target_logged), from the law-off run."""
    return TIP_LOG - servo_only(TGT_LOG, TIP_LOG[0])


D_RES = residual()


def simulate(law="integral", k=0.8, gamma=0.05, shaper=True, seed=0, d=None,
             n=None, sensor_gain=None, sensor_sigma=SENSOR_SIGMA, aw=False, extend_to=None,
             integrate60=False):
    """servo + replayed residual + pendulum + 20 Hz noisy tip sensor + a law."""
    d = D_RES if d is None else d
    n = N if n is None else n
    if extend_to is not None:                       # continue past the log: the sea residual of the last 10 s, tiled
        n = int(round(extend_to * FPS))
        tail = d[-600:]
        d = np.concatenate([d] + [tail] * (1 + (n - len(d)) // len(tail)))[:n]
    rng = np.random.default_rng(seed)
    sv = Servo(TIP_LOG[0])
    pend = Pendulum(PEND_P0)
    q_t = to_polar(TGT_LOG[0])
    sh = ZVShaper(q_t, DT)
    th_un = q_t[0]
    ctrl = IntegralAntiswing(k, gamma, aw)
    u = np.zeros(2)
    s_est = np.zeros(2)
    ds_est = np.zeros(2)
    seen = False
    tip = TIP_LOG[0].copy()
    hook = PEND_P0.copy()
    tip_h = np.zeros((n, 3))
    hook_h = np.zeros((n, 3))
    tgt_h = np.zeros((n, 3))
    sat_h = np.zeros(n, bool)
    satr_h = np.zeros(n, bool)
    u_h = np.zeros((n, 2))
    for i in range(n):
        t = T_LOG[i] if i < N else T_LOG[N - 1] + (i - N + 1) * DT
        target, wl = op_target(t)
        qt = to_polar(target)
        th_un = th_un + wrap(qt[0] - th_un)
        qt[0] = th_un
        cmd = to_cart(sh.shape(qt, wl)) if shaper else target.copy()
        if law == "integral":
            cmd[0] += u[0]
            cmd[2] += u[1]
        elif law == "legacy" and seen:
            corr = AS_KP * s_est + AS_KD * ds_est
            nn = float(np.linalg.norm(corr))
            if nn > AS_CLAMP:
                corr = corr * (AS_CLAMP / nn)
            if float(np.linalg.norm(s_est)) > AS_CUTOUT:
                corr = np.zeros(2)
            cmd[0] += corr[0]
            cmd[2] += corr[1]
        if law == "integral" and integrate60 and seen:
            ctrl.sat = sv.sat_rate                  # integrate the HELD estimate every frame:
            u = ctrl.update(s_est, True, DT)        # no 20 Hz staircase into the acceleration limit
        tip = sv.step(cmd, DT) + d[i]
        hook = pend.step(tip, DT, wl)
        # the tip sensor, 20 Hz, its own clock (frame_i = i + 1 in crane_lift.py)
        if (i + 1) % SENSOR_EVERY == 0 and (i + 1) > 2:
            s_true = np.array([hook[0] - tip[0], hook[2] - tip[2]])
            if sensor_gain is not None:
                s_true = s_true * sensor_gain(float(np.linalg.norm(s_true)))
            s_new = s_true + rng.normal(0.0, sensor_sigma, 2)
            if seen:
                ds_est = (s_new - s_est) / DT_SENSOR
            s_est = s_new
            seen = True
            if law == "integral" and not integrate60:
                ctrl.sat = sv.sat_rate
                u = ctrl.update(s_est, True, DT_SENSOR)
        tip_h[i] = tip
        hook_h[i] = hook
        tgt_h[i] = target
        sat_h[i] = sv.sat
        satr_h[i] = sv.sat_rate
        u_h[i] = u
    return dict(tip=tip_h, hook=hook_h, target=tgt_h, sat=sat_h, sat_rate=satr_h, u=u_h)


# ---- metrics -----------------------------------------------------------------
PHASES = [("hold", 0.0, 4.0), ("transfer", 4.0, 44.0), ("payout", 44.0, 50.5)]


def swing_of(tip, hook):
    return np.hypot(hook[:, 0] - tip[:, 0], hook[:, 2] - tip[:, 2])


def metrics(res, n=None):
    n = len(res["tip"])
    t = T_LOG[0] + np.arange(n) * DT
    s = swing_of(res["tip"], res["hook"])
    err = np.linalg.norm(res["tip"] - res["target"], axis=1)
    m = {"rms": float(np.sqrt((s ** 2).mean())), "max": float(s.max()),
         "err": float(np.sqrt((err ** 2).mean())), "sat": float(res["sat"].mean()),
         "satr": float(res["sat_rate"].mean()), "tail": float(np.sqrt((s[t >= 48.5] ** 2).mean()))}
    for nm, a, b in PHASES:
        sel = (t >= a) & (t < b)
        m[nm] = float(np.sqrt((s[sel] ** 2).mean()))
        m[nm + "_max"] = float(s[sel].max())
    return m


def rms(a):
    return float(np.sqrt((np.asarray(a) ** 2).mean()))


# ---- validation --------------------------------------------------------------
def validate():
    print("=" * 78)
    print("V1  recorded tip path -> the pendulum copy")
    pend = Pendulum(PEND_P0)
    hk = np.zeros((N, 3))
    for i in range(N):
        hk[i] = pend.step(TIP_LOG[i], DT, WIRE_LOG[i])
    s = swing_of(TIP_LOG, hk)
    s_log = swing_of(TIP_LOG, HOOK_LOG)
    print(f"    surrogate pendulum : swing RMS {rms(s):.3f} m, max {s.max():.3f} m")
    print(f"    recorded           : swing RMS {rms(s_log):.3f} m, max {s_log.max():.3f} m")
    print(f"    hook position difference: RMS {rms(np.linalg.norm(hk - HOOK_LOG, axis=1)):.4f} m, "
          f"max {np.linalg.norm(hk - HOOK_LOG, axis=1).max():.4f} m")

    print("V2  recorded nominal target -> the servo (d = 0)")
    tip = servo_only(TGT_LOG, TIP_LOG[0])
    e = np.linalg.norm(tip - TIP_LOG, axis=1)
    sel = (T_LOG >= 4.0) & (T_LOG < 44.0)
    print(f"    servo vs recorded tip: RMS {rms(e):.3f} m overall, {rms(e[sel]):.3f} m over the transfer, max {e.max():.3f} m")
    for nm, a, b in PHASES:
        m = (T_LOG >= a) & (T_LOG < b)
        print(f"      {nm:9s} RMS {rms(e[m]):.3f} m")
    ehz = np.abs(tip[:, 1] - TIP_LOG[:, 1])
    ehr = np.hypot(tip[:, 0] - TIP_LOG[:, 0], tip[:, 2] - TIP_LOG[:, 2])
    print(f"      vertical RMS {rms(ehz):.3f} m, horizontal RMS {rms(ehr):.3f} m")

    print("V3  the legacy law inside the full surrogate")
    r = simulate(law="legacy", shaper=False, seed=0)
    m = metrics(r)
    print(f"    surrogate legacy: swing RMS {m['rms']:.2f} m, max {m['max']:.2f} m, "
          f"per phase {m['hold']:.2f}/{m['transfer']:.2f}/{m['payout']:.2f} m, tip-target RMS {m['err']:.2f} m")
    print(f"    recorded legacy : swing RMS 3.24 m, max 6.51 m")
    r0 = simulate(law="off", shaper=False, seed=0)
    m0 = metrics(r0)
    print(f"    surrogate law off: swing RMS {m0['rms']:.2f} m, max {m0['max']:.2f} m, "
          f"per phase {m0['hold']:.3f}/{m0['transfer']:.2f}/{m0['payout']:.2f} m, tip-target RMS {m0['err']:.2f} m")
    print(f"    recorded law off : swing RMS 0.84 m, max 2.57 m, per phase 0.054/0.885/0.817 m, tip-target RMS 1.10 m")
    print("=" * 78)
    return m0, m


# ---- the sweep ---------------------------------------------------------------
def sweep(aw=False, gammas=(0.0, 0.05, 0.1), shapers=(False, True)):
    rows = []
    for shaper in shapers:
        for k in (0.3, 0.5, 0.8, 1.2):
            for gamma in gammas:
                m = metrics(simulate(law="integral", k=k, gamma=gamma, shaper=shaper, seed=0, aw=aw))
                m.update(k=k, gamma=gamma, shaper=shaper, aw=aw)
                rows.append(m)
    return rows


def ref_rows():
    out = []
    for law in ("off", "legacy"):
        m = metrics(simulate(law=law, shaper=False, seed=0))
        m.update(k=law, gamma="-", shaper=False, aw=False)
        out.append(m)
    return out


def table(rows):
    out = ["| shaper | k | gamma | swing RMS | swing max | hold 0-4 | transfer 4-44 | payout 44-50.5 | "
           "tip-target RMS | rate-sat | any-sat |", "|" + "---|" * 11]
    for r in rows:
        kk = r["k"] if isinstance(r["k"], str) else "%.1f" % r["k"]
        gg = r["gamma"] if isinstance(r["gamma"], str) else "%.2f" % r["gamma"]
        out.append(f"| {'on' if r['shaper'] else 'off'} | {kk} | {gg} | "
                   f"{r['rms']:.3f} | {r['max']:.3f} | {r['hold']:.3f} | {r['transfer']:.3f} | "
                   f"{r['payout']:.3f} | {r['err']:.3f} | {r['satr']:.2f} | {r['sat']:.2f} |")
    return chr(10).join(out)


def settled(cfgs):
    """The 50.5 s log holds barely one pendulum period after the transfer: continue to 80 s."""
    print()
    print("settled hold (the log continued to 80 s, the last 10 s of the sea residual tiled)")
    for lab, c in cfgs:
        r = simulate(seed=0, extend_to=80.0, **c)
        n = len(r["tip"])
        t = T_LOG[0] + np.arange(n) * DT
        s = swing_of(r["tip"], r["hook"])
        m = t >= 56.0
        print(f"  {lab:30s} 56-80 s swing RMS {rms(s[m]):.3f} m, max {s[m].max():.3f} m")


def robustness(best):
    print()
    print("robustness: the sea residual shifted +0.5 s (30 frames, held at the start)")
    d_shift = D_RES[np.maximum(np.arange(N) - 30, 0)]
    for lab, c in best:
        m0 = metrics(simulate(seed=0, **c))
        m1 = metrics(simulate(seed=0, d=d_shift, **c))
        print(f"  {lab:30s} swing RMS {m0['rms']:.3f} -> {m1['rms']:.3f} m, "
              f"max {m0['max']:.3f} -> {m1['max']:.3f} m, payout {m0['payout']:.3f} -> {m1['payout']:.3f} m")
    for law in ("off", "legacy"):
        m0 = metrics(simulate(law=law, shaper=False, seed=0))
        m1 = metrics(simulate(law=law, shaper=False, seed=0, d=d_shift))
        print(f"  {law:30s} swing RMS {m0['rms']:.3f} -> {m1['rms']:.3f} m, "
              f"max {m0['max']:.3f} -> {m1['max']:.3f} m")

    print()
    print("ten sensor seeds 0..9")
    for lab, c in best:
        runs = [simulate(seed=sd, **c) for sd in range(10)]
        r = [metrics(x)["rms"] for x in runs]
        h0 = runs[0]["hook"]
        tan = np.gradient(h0, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        ct_rms, ct_max = [], []
        for x in runs[1:]:
            dv = x["hook"] - h0
            ct = np.linalg.norm(dv - (dv * tan).sum(1)[:, None] * tan, axis=1)
            ct_rms.append(rms(ct))
            ct_max.append(ct.max())
        print(f"  {lab:30s} swing RMS {min(r):.4f}..{max(r):.4f} m "
              f"(spread {1e3 * (max(r) - min(r)):.1f} mm), hook cross-track vs seed 0: "
              f"RMS {1e3 * max(ct_rms):.1f} mm, max {1e3 * max(ct_max):.1f} mm")


def sensor_bias_check(best):
    """The recorded sensor UNDER-reports the swing (deck returns inside the cone): what does it cost?"""
    i = np.arange(N)
    m = ((i + 1) % 3 == 0) & (i > 1)
    print()
    for nm, L in (("law off", LOG_OFF), ("legacy", LOG_ON)):
        nt = np.hypot(*(L[:, [4, 6]] - L[:, [1, 3]])[m].T)
        ne = np.hypot(*L[:, 17:19][m].T)
        sel = nt > 0.2
        print(f"recorded sensor gain |s_est|/|s_true| ({nm} run): median {float(np.median(ne[sel] / nt[sel])):.2f}")
    for lab, c in best:
        for gain in (1.0, 0.6, 0.4):
            m2 = metrics(simulate(seed=0, sensor_gain=(lambda _n, gg=gain: gg), **c))
            print(f"  {lab:30s} sensor gain {gain:.1f}: swing RMS {m2['rms']:.3f} m, "
                  f"payout {m2['payout']:.3f} m")


def png(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab, c = cfg
    r = simulate(seed=0, **c)
    s_off = swing_of(LOG_OFF[:, 1:4], LOG_OFF[:, 4:7])
    s_on = swing_of(LOG_ON[:, 1:4], LOG_ON[:, 4:7])
    s_new = swing_of(r["tip"], r["hook"])
    e_off = np.linalg.norm(LOG_OFF[:, 1:4] - LOG_OFF[:, 14:17], axis=1)
    e_on = np.linalg.norm(LOG_ON[:, 1:4] - LOG_ON[:, 14:17], axis=1)
    e_new = np.linalg.norm(r["tip"] - r["target"], axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(9.6, 6.8), sharex=True)
    for a in ax:
        a.axvspan(0, 4, color="0.93", zorder=0)
        a.axvspan(44, 50.5, color="0.93", zorder=0)
        a.axvspan(17.9, 30.1, color="#fbeacd", zorder=0)
        a.grid(alpha=0.3)
    ax[0].plot(LOG_ON[:, 0], s_on, color="#c0392b", lw=1.0, label="legacy law, recorded (RMS 3.24 m)")
    ax[0].plot(LOG_OFF[:, 0], s_off, color="#7f8c8d", lw=1.1, label="law off, recorded (RMS 0.84 m)")
    ax[0].plot(T_LOG, s_new, color="#1a6fb0", lw=1.5,
               label=f"integral law, surrogate: {lab} (RMS {rms(s_new):.2f} m)")
    ax[0].set_ylabel("horizontal swing |hook - tip|  [m]")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("crane lift, seed 0: payload swing and tip tracking" + chr(10) +
                    "grey: hold and pay-out.  cream: the nominal target demands more slew rate "
                    "than the C25 has, so the tip is rate-saturated", fontsize=9)
    ax[1].plot(LOG_ON[:, 0], e_on, color="#c0392b", lw=1.0)
    ax[1].plot(LOG_OFF[:, 0], e_off, color="#7f8c8d", lw=1.1)
    ax[1].plot(T_LOG, e_new, color="#1a6fb0", lw=1.5)
    ax[1].set_ylabel("tip - nominal target  [m]")
    ax[1].set_xlabel("simulation time [s]")
    ax[1].set_xlim(0, 50.5)
    fig.tight_layout()
    out = os.path.join(HERE, "swing_surrogate.png")
    fig.savefig(out, dpi=140)
    print()
    print(f"wrote {out}")


REC = ("k=0.8, gamma=0.4, shaper off, anti-windup",
       dict(law="integral", k=0.8, gamma=0.4, shaper=False, aw=True))
ALT = ("k=0.5, gamma=0.4, shaper off, anti-windup",
       dict(law="integral", k=0.5, gamma=0.4, shaper=False, aw=True))

# =============================================================================
# DROP-IN FOR crane_lift.py (phase 2).  Paste this block above step(); the call
# sites are marked below.  Nothing here imports from the surrogate.
# =============================================================================
AS_K, AS_GAMMA, AS_UMAX = 0.8, 0.4, 1.5     # integral gain [1/s], leak [1/s], correction clamp [m]
AS_CUTOUT_M = 3.0                           # runaway guard, kept from the legacy law
ZV_ON = False                               # the ZV shaper: measured as a net loss on this profile


def antiswing_state():
    """The controller's whole memory: one integrator, one flag."""
    return {"u": np.zeros(2), "sat": False, "cutout": 0}


def antiswing_update(s_est, have_est, dt_sensor, state):
    """u' = k s - gamma u, integrated once per sensor update and HELD between updates.

    s_est          the tip sensor's horizontal swing estimate (hook - tip) in world x, z [m]
    have_est       False when the scan found no load: the correction is held, not zeroed
    dt_sensor      seconds since the previous estimate (0.05 s at 20 Hz)
    state          the dict from antiswing_state(); state["sat"] must be set True by the
                   caller on any frame the drive is at a RATE limit (see actuate() below)

    Returns the world (x, z) correction to add to the tip target.

    Why an integrator and not the legacy PD: through a position servo the pivot velocity,
    not the pivot position, sets the swing damping.  s'' = -omega^2 s - k s' - p_target'',
    so a positive k is damping and any positive KD on s' is ANTI-damping (see plans/crane-lift-e3.md).
    The leak gamma bounds the correction and unwinds it after the transfer; the conditional
    integration keeps the integrator from winding up against a rate-saturated slew drive.
    """
    u = state["u"]
    if not have_est:
        return u.copy()                     # sensor dropout: hold the last correction
    s = np.asarray(s_est, np.float64)
    if float(np.linalg.norm(s)) > AS_CUTOUT_M:      # runaway guard: bleed out, do not jump to zero
        state["cutout"] += 1
        state["u"] = u * max(0.0, 1.0 - 2.0 * dt_sensor)
        return state["u"].copy()
    u_new = u + (AS_K * s - AS_GAMMA * u) * dt_sensor
    n = float(np.linalg.norm(u_new))
    if n > AS_UMAX:
        u_new = u_new * (AS_UMAX / n)
    if not (state["sat"] and n > float(np.linalg.norm(u))):   # anti-windup: may shrink, not grow
        state["u"] = u_new
    return state["u"].copy()


class ZVShaper:
    """Zero-vibration shaper on the tip target, in polar coordinates about the king.

    Two impulses of 0.5 separated by half the pendulum period T = 2 pi sqrt(L/g); the delay
    follows the wire length frame by frame.  Feed it the NOMINAL target; the anti-swing
    correction is added after shaping, never shaped.
    """

    def __init__(self, king_xz, dt, max_seconds=8.0):
        self.king = np.asarray(king_xz, np.float64)
        self.dt = dt
        self.max_n = int(round(max_seconds / dt))
        self.buf = []
        self.th = None

    def _polar(self, p):
        dx, dz = p[0] - self.king[0], p[2] - self.king[1]
        th = math.atan2(dz, dx)
        if self.th is not None:                     # keep the slew angle continuous
            th = self.th + (th - self.th + math.pi) % (2.0 * math.pi) - math.pi
        self.th = th
        return np.array([th, math.hypot(dx, dz), p[1]])

    def shape(self, target, wire_len):
        q = self._polar(target)
        self.buf.append(q)
        if len(self.buf) > self.max_n:
            self.buf.pop(0)
        T = 2.0 * math.pi * math.sqrt(max(wire_len, 1e-3) / 9.81)
        j = max(0, len(self.buf) - 1 - int(round(0.5 * T / self.dt)))
        qs = 0.5 * self.buf[-1] + 0.5 * self.buf[j]
        return np.array([self.king[0] + qs[1] * math.cos(qs[0]), qs[2], self.king[1] + qs[1] * math.sin(qs[0])])


# --- the two call sites in crane_lift.py -------------------------------------
# 1. next to swing_est, once:
#        as_state = antiswing_state()
#        zv = ZVShaper(KING_XZ, DT)
# 2. in actuate(), after the rate clip, so the law can see a saturated drive:
#        as_state["sat"] = bool(np.any(np.abs((q_filt - q) / dt) > V_MAX * (1.0 + 1e-9)))
# 3. in step(), replacing `cmd = target.copy()` and the AS_KP/AS_KD block:
#        cmd = zv.shape(target, wire_len) if ZV_ON else target.copy()
#        if antiswing_on[0]:
#            corr = antiswing_update(swing_est["s"], swing_est["seen"], sensor_every * dt, as_state)
#            cmd[0] += corr[0]
#            cmd[2] += corr[1]
#    (leave swing_est["ds"] in the log; the new law does not use it.)


if __name__ == "__main__":
    if STAGE in ("validate", "all"):
        validate()
    if STAGE in ("sweep", "all"):
        print()
        print("sweep, the law exactly as specified (no anti-windup)")
        print()
        print(table(ref_rows() + sweep(aw=False)))
        print()
        print("the same grid with conditional-integration anti-windup, gamma extended past the grid")
        print()
        print(table(ref_rows() + sweep(aw=True, gammas=(0.0, 0.05, 0.1, 0.2, 0.4))))
    if STAGE in ("robust", "all"):
        best = [REC, ALT]
        settled([("law off", dict(law="off", shaper=False)),
                 ("legacy", dict(law="legacy", shaper=False))] + best)
        robustness(best)
        sensor_bias_check(best)
    if STAGE in ("png", "all"):
        png(REC)
