"""Spot foot and body kinematics, the tread-local closed form of the authored tents, and the placement and
stability metric definitions — one ruler shared by SpotStepsEnv's device counters, score_e0.py's
per-touchdown records and any CPU harness.

LINK ORDER, MEASURED 2026-09-12 (a K=8 GpuSim stand and one CPU TGS Spot on flat ground, each link's pose
matched to the joint geometry build_spot authors):
  direct-GPU path (GpuSim.link_pose, PhysxGpuBatch eLINK_GLOBAL_POSE) is PhysX BREADTH-FIRST:
      0 base | 1-4 hips fl,fr,hl,hr | 5-8 upper legs fl,fr,hl,hr | 9-12 lower legs fl,fr,hl,hr
      Link 0 equals root_pose exactly. This is NOT add order (threepp/rl/sim.py's read_links comment used to
      say it was).
  CPU articulation (art.links) is ADD order, per leg (hip, upper, lower):
      0 base | fl 1,2,3 | fr 4,5,6 | hl 7,8,9 | hr 10,11,12  ->  lower legs 3,6,9,12, upper legs 2,5,8,11
  Copying one indexing into the other reads hips as feet, and every number comes out plausible and wrong.

FOOT TIP, MEASURED on the same stands: a leg link's frame has its origin at the capsule midpoint with local +Y
along the capsule, pointing away from the parent joint (spot_deploy._capsule maps +Y onto knee->foot). So the
foot-tip hemisphere centre is (0, +0.15, 0) in the lower-leg frame: settled on flat ground, its z minus the
0.028 m foot radius reads 0.0000 m on all four feet, GPU and CPU alike (within 0.1 mm); (0, -0.15, 0) is the
knee end, 0.21-0.25 m up. The upper leg is the same construction (its +0.15 end sits 2.2 cm from the lower
leg's -0.15 end). Settled tips relative to the base on the GPU stand: dx +0.296 (front) / -0.383 (hind),
dy +-0.215 / +-0.208, so spot_steps_env.FOOT_DX/FOOT_DY (+-0.30 / +-0.17) are an approximation.

THE TENT, as _add_steps builds it (t = x - tent start, run 0.30, n_up 3, landing 0.8): the last ascent box is
flush with the landing, so there are n_up - 1 = 2 real treads going up (tops r, 2r), a 1.1 m plateau at 3r
(the flush top step + the landing), n_up - 1 = 2 real treads going down (2r, r), then ground from
t = 0.9 + 0.8 + 0.6. A tread here has a real edge at BOTH ends; the drop edge is the rear one on the way up
and the front one on the way down. (An earlier placement probe counted the flush top step and the ground run
after the last descent box as 0.30 m treads with a virtual edge; this ruler does not.)
"""
import numpy as np
import torch

# ---- measured link layout --------------------------------------------------------------------------------
GPU_LINKS = {"base": 0, "hip": (1, 2, 3, 4), "uleg": (5, 6, 7, 8), "lleg": (9, 10, 11, 12)}
CPU_LINKS = {"base": 0, "hip": (1, 4, 7, 10), "uleg": (2, 5, 8, 11), "lleg": (3, 6, 9, 12)}
FEET = ("fl", "fr", "hl", "hr")
TIP_LOCAL = (0.0, 0.15, 0.0)          # lower-leg frame -> foot-tip hemisphere centre (measured, see above)
FOOT_R = 0.028                        # foot capsule radius (spot_deploy.build_spot)
ULEG_R = 0.045                        # upper-leg capsule radius
ULEG_SAMPLES = (-0.15, 0.0, 0.15)     # along the upper-leg axis: hip end, middle, knee end
BASE_HALF = (0.35, 0.09, 0.095)       # base box half extents
BASE_SAMPLES_X = (-0.35, -0.175, 0.0, 0.175, 0.35)   # 0.175 m apart: a 0.30 m tread's nosing cannot hide
BASE_SAMPLES_Y = (-0.09, 0.09)                       # between two samples under the base's lower face

# ---- honest termination ------------------------------------------------------------------------------------
TILT_UP = 0.35          # up_z below this ...
TILT_TICKS = 10         # ... for 10 control ticks (0.2 s) in a row is a fall
BODY_CONTACT = 0.03     # a base or upper-leg sample this close to the terrain below it is a fall
# ... except the upper leg's KNEE END (sample +0.15), measured 2026-09-12 (spot_steps.pt,
# the score_e0 world, K=128, 820 ticks). With all three samples the rule first fired 107 times, 106 of them on a
# knee end and 89 of those a FRONT knee skimming the landing plateau or its nosing with the robot upright (up
# 0.93-0.99) and walking on; it fired in 44 of the 46 S20 episodes that succeeded (legacy score_e0 run). On stair
# lanes the front knee end is within 0.03 m on ~0.9% of live ticks and touching on 0.03%, while the hip end and
# the middle are within 0.03 m on <= 0.02%. A knee scraping a nosing is not a fall, so the knee end is recorded as
# its own cause (knee_ticks, knee_fired, term_knee) and does not terminate. True restores the literal rule.
KNEE_TERMINATES = False

# ---- touchdowns and placement ------------------------------------------------------------------------------
TD_ON = 0.01            # stance switches ON when the tip clearance drops below this (m) ...
TD_OFF = 0.02           # ... and OFF only above this one, so a chattering contact is one touchdown
TD_ARM_TICKS = 10       # nothing counts as a touchdown in an episode's first 10 ticks: that is the spawn drop
EDGE_NEAR = 0.05        # P(d_edge < 0.05)
DROP_NEAR = FOOT_R      # P(d_drop < 0.028): the contact patch overhangs the drop
NOSE_SCRAPE = 0.02      # nosing clearance below this is a scrape
STALL_DIST = 0.06       # a touchdown this close short of an edge that rises in the travel direction ...
STALL_STRIDE = 0.10     # ... after a swing that moved the tip less than this is a stall at the nosing
# A touchdown whose ground patch stands more than PERCH_RISE risers above the surface under its tip centre is PERCHED on
# the next surface's nosing, centre hanging past that surface's edge. Measured 2026-09-12 (review probe, spot_steps.pt,
# K=64): every ascent touchdown with the centre within FOOT_R of the riser ahead (across 0.272-0.30) stood on the tread
# above, 16-36% of ascent touchdowns on S10/S20/P2, 69-81% of them read as stalls. So it is placed against the surface it
# stands on (negative across / d_edge / d_drop) and is never a stall.
PERCH_RISE = 0.5
# Slip counts the tip centre's travel only over ticks the foot is ON the ground at both ends. Measured (same probe): the
# stance hysteresis (TD_ON..TD_OFF) keeps a landing or lifting foot in stance 1-2 cm up; ticks with either clearance
# >= 5 mm were 7% of stance ticks and carried 45-60% of the slip, while on the remaining ticks the tip-centre speed
# (S10/S20/F/P2/SC20 0.093/0.121/0.083/0.154/0.279 m/s) matched the contact-point sliding speed from the link
# velocities (0.084/0.113/0.079/0.148/0.270).
SLIP_CONTACT = 0.005
PATCH_D = (0.0, FOOT_R, -FOOT_R, 0.0, 0.0)    # the foot's ground patch: centre, +-r in x, +-r in y
PATCH_E = (0.0, 0.0, 0.0, FOOT_R, -FOOT_R)
# Chance on a 0.30 m tread with d_edge ~ U(0, 0.15): median 0.075, P(d_edge < 0.05) = 1/3,
# P(d_drop < 0.028) = 0.028 / 0.30.
# Uniform tip-centre placement along a 0.30 m tread, read through THIS ruler. P(d_drop < 0.028) is not
# FOOT_R/0.30 = 0.093 here: every riser is taller than FOOT_R, so a centre within FOOT_R before an edge that
# rises in the direction of travel rests on that nosing and is read as perched (d_drop < 0). Measured with
# 2M uniform touchdowns through tread_frame(at=...): 0.1871 ascent, 0.1864 descent.
CHANCE = {"median_d_edge": 0.075, "p_edge_lt05": 1.0 / 3.0, "p_drop_lt028": 0.187}

KINDS = ("approach", "ascent", "landing", "descent", "runout", "flat", "offlane")
K_APPROACH, K_ASCENT, K_LANDING, K_DESCENT, K_RUNOUT, K_FLAT, K_OFFLANE = range(len(KINDS))

# ---- stability, recovery and validity gates ---------------------------------------------------------------
SLOPE_DX, SLOPE_DY = 0.30, 0.20       # heading-aligned stance rectangle the local slope is plane-fitted over
REC_UP = 0.02           # recovered: |up - up the tick before the shove| < 0.02 ...
REC_V = 0.15            # ... and |cmd_xy - body v_xy| < 0.15 m/s ...
REC_HOLD = 5            # ... held for 5 ticks; the recovery time is the first tick of that run
TAU_CAP = (45.0, 45.0, 115.0)         # hx, hy, kn effort caps (N·m)
GROUPS = ("hx", "hy", "kn")

# ---- histogram layouts (the accumulator columns below index into these) ------------------------------------
EDGE_BINS, EDGE_W = 30, 0.005         # d_edge 0..0.15 m
ACROSS_BINS, ACROSS_W = 30, 0.01      # across_tread 0..0.30 m (0 = the tread's rear edge, 0.30 = its front)
TAU_BINS, TAU_W = 50, 5.0             # |commanded torque| 0..250 N·m, last bin open
QD_BINS, QD_W = 50, 1.0               # |joint speed| 0..50 rad/s, last bin open
REC_BINS, REC_W = 20, 5               # recovery ticks, 5-tick bins, last bin open
# interior edges -> len + 1 bins (torch.bucketize(right=True)); the outer bins are open
NOSE_EDGES = (-0.04, -0.02, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
              0.40, 0.50, 0.70)
UPREL_EDGES = (0.0, 0.35, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
               0.995, 0.999)


def _tread_cols(dn):
    # the *_stall columns repeat the placement numbers for the stall touchdowns alone, so a stall (set down short of
    # a rising edge on the surface under the foot, after a short swing) can be taken out of the placement distribution.
    # perched_* counts the touchdowns placed on this tread from its neighbour's side (see PERCH_RISE); they read
    # a negative d_edge / d_drop, binned at 0 in the histograms
    return ((f"tread_td_{dn}", 1), (f"edge_lt05_{dn}", 1), (f"drop_lt028_{dn}", 1), (f"edge_sum_{dn}", 1),
            (f"stall_{dn}", 1), (f"edge_hist_{dn}", EDGE_BINS), (f"across_hist_{dn}", ACROSS_BINS),
            (f"edge_lt05_stall_{dn}", 1), (f"drop_lt028_stall_{dn}", 1), (f"edge_sum_stall_{dn}", 1),
            (f"edge_hist_stall_{dn}", EDGE_BINS), (f"perched_{dn}", 1))


# Per-episode accumulator columns, summed every control tick into [K, width] and flushed into the (block, lane
# type, episode bin) rows when the episode ends — the STAT_KEYS idiom, one level down.
TICK_COLS = (("ticks", 1), ("path_m", 1), ("up_sum", 1), ("cos_slope_sum", 1), ("uprel_hist", len(UPREL_EDGES) + 1),
             ("stance_ticks", 1), ("slip_m", 1), ("miss_body", 1), ("miss_feet", 1), ("cf_mismatch", 1),
             ("knee_ticks", 1),
             ("offsurf_ticks", 1),     # slope lanes only: ticks with the base over a lane of another surface (spot_slopes.surface_ids)
             ("recovered", 1), ("recovery_ticks", 1), ("recovery_hist", REC_BINS),
             *[(f"tau_hist_{g}", TAU_BINS) for g in GROUPS], *[(f"qd_hist_{g}", QD_BINS) for g in GROUPS],
             ("td", 1), ("td_kind", len(KINDS)), ("stall", 1), ("perched", 1),
             ("nose_n", 1), ("nose_lt02", 1), ("nose_sum", 1), ("nose_hist", len(NOSE_EDGES) + 1),
             *_tread_cols("asc"), *_tread_cols("desc"))
# Written once, when the episode ends. term_* are the causes AT the terminating tick (both rules evaluated on
# the same state, whichever one terminated); *_fired / *_first latch over the whole episode, so a legacy run
# says how many of its terminations an honest rule would also have called, and when.
EPISODE_COLS = ("term_tilt_sustained", "term_body", "term_body_base", "term_body_uleg", "term_legacy_cond",
                "term_honest_cond", "legacy_fired", "honest_fired", "honest_fired_tilt", "honest_fired_body",
                "legacy_first", "honest_first", "drift_abs_m", "drift_signed_m", "term_knee", "knee_fired",
                "drift_downhill_m")
# drift_downhill_m: the lateral offset at the episode end signed toward gravity's pull on a cross-slope lane (the
# corrugated cross lanes alternate it, so drift_signed_m cancels over a block); 0 on every other lane.


def layout():
    """{name: (offset, width)} over TICK_COLS then EPISODE_COLS, and the total width."""
    out, o = {}, 0
    for name, w in TICK_COLS:
        out[name] = (o, w)
        o += w
    for name in EPISODE_COLS:
        out[name] = (o, 1)
        o += 1
    return out, o


LAYOUT, WIDTH = layout()
TREAD_W = sum(w for _, w in _tread_cols("asc"))      # the descent block sits TREAD_W after the ascent one

# One row per touchdown in score_e0's .npz (float32 on the device; ids are cast on the host).
TD_FIELDS = ("lane_x", "tread_index", "kind", "across_tread", "edge_dist", "drop_dist", "direction", "riser",
             "nose_clear", "stall", "x", "y_lane", "x_rel_base", "stride", "cmd_vx", "perched")


# ---- kinematics -------------------------------------------------------------------------------------------
def quat_axes(q):
    """The body axes in the world frame, each [..., 3], from (qx, qy, qz, qw) body->world."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ex = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
    ey = torch.stack([2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)], dim=-1)
    ez = torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)
    return ex, ey, ez


def quat_axis_y(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)], dim=-1)


class FootKin:
    """Foot tips and body contact samples from a [K, links, 7] link-pose buffer (PhysX layout, quaternion
    first). Every constant is a device tensor built here, so the per-step calls allocate nothing but op
    temporaries and replay from a CUDA graph."""

    def __init__(self, device, links=GPU_LINKS):
        dev = torch.device(device)
        self.feet = torch.tensor(links["lleg"], dtype=torch.long, device=dev)
        self.uleg = torch.tensor(links["uleg"], dtype=torch.long, device=dev)
        gx, gy = np.meshgrid(BASE_SAMPLES_X, BASE_SAMPLES_Y, indexing="ij")
        self.base_gx = torch.tensor(gx.reshape(-1), dtype=torch.float32, device=dev)     # [10]
        self.base_gy = torch.tensor(gy.reshape(-1), dtype=torch.float32, device=dev)
        self.uleg_s = torch.tensor(ULEG_SAMPLES, dtype=torch.float32, device=dev)         # [3]
        self.n_base = self.base_gx.numel()
        self.n_body = self.n_base + 4 * len(ULEG_SAMPLES)
        self.body_r = torch.tensor([0.0] * self.n_base + [ULEG_R] * (4 * len(ULEG_SAMPLES)), device=dev)
        # [12] which upper-leg samples (legs x ULEG_SAMPLES, as body_points orders them) count for a fall
        self.uleg_term = torch.tensor([KNEE_TERMINATES or s < max(ULEG_SAMPLES) for _ in range(4) for s in ULEG_SAMPLES],
                                      dtype=torch.bool, device=dev)

    def tips(self, link_pose):
        """[K, 4, 3] world foot-tip hemisphere centres (fl, fr, hl, hr)."""
        ll = link_pose.index_select(1, self.feet)
        return ll[..., 4:7] + TIP_LOCAL[1] * quat_axis_y(ll[..., 0:4])

    def body_points(self, root_pos, root_quat, up, link_pose):
        """[K, n_body, 3]: 10 samples on the base's DOWNWARD-facing face (the lowest box corner is always among
        them), then 3 along each upper-leg axis. Clearance = z - body_r - terrain."""
        k = root_pos.shape[0]
        ex, ey, ez = quat_axes(root_quat)
        face = torch.where(up >= 0.0, -BASE_HALF[2], BASE_HALF[2])                          # [K]
        base = (root_pos[:, None, :] + self.base_gx[None, :, None] * ex[:, None, :]
                + self.base_gy[None, :, None] * ey[:, None, :] + face[:, None, None] * ez[:, None, :])
        ul = link_pose.index_select(1, self.uleg)                                           # [K,4,7]
        ax = quat_axis_y(ul[..., 0:4])
        legs = ul[..., None, 4:7] + self.uleg_s[None, None, :, None] * ax[:, :, None, :]    # [K,4,3,3]
        return torch.cat([base, legs.reshape(k, -1, 3)], dim=1)


def cpu_foot_tips(art):
    """[4, 3] numpy foot tips of one CPU articulation (add-order links, see the module docstring)."""
    import threepp as tp
    links = art.links
    out = np.empty((4, 3))
    for i, li in enumerate(CPU_LINKS["lleg"]):
        p = links[li].world_point(tp.Vector3(*TIP_LOCAL))
        out[i] = (p.x, p.y, p.z)
    return out


# ---- the tent, in closed form -----------------------------------------------------------------------------
def tent_surfaces(run, n_up, land):
    """The walkable surfaces of one tent in t (x from the tent start), indexed like tread_index:
    (t_lo, t_hi, height in risers) for ascent treads 0..n_up-2, the plateau, descent treads 0..n_up-2."""
    top0, land_end = (n_up - 1) * run, n_up * run + land
    s = [(k * run, (k + 1) * run, k + 1) for k in range(n_up - 1)]
    s.append((top0, land_end, n_up))
    s += [(land_end + k * run, land_end + (k + 1) * run, n_up - 1 - k) for k in range(n_up - 1)]
    return np.array(s, np.float64)


def tread_frame(t, riser, run, n_up, land, at=None):
    """Where t (x from the tent start; any shape) sits on the tent. No boolean indexing, so it replays from a
    graph. `at` (default t) picks the surface and t is measured on it: a perched foot (see PERCH_RISE) passes a
    point on the surface it stands on and reads across < 0 or >= width there, so d_edge / d_drop < 0 say how far
    its centre hangs past the edge. Returns a dict of tensors shaped like t:

      kind       K_APPROACH / K_ASCENT / K_LANDING / K_DESCENT / K_RUNOUT
      index      tread_index (ascent k = k, plateau = n_up-1, descent k = n_up + k), -1 on the ground
      across     from the surface's rear (low-x) edge: [0, run) on a tread, [0, run + land) on the plateau
      d_edge     min(across, width - across); NaN on the ground
      d_drop     distance to the edge past which the surface falls away (rear on the way up, front on the
                 way down, the nearer one on the plateau); NaN on the ground
      height     surface top, in metres
      rise_fwd   distance ahead (+x) to the next edge that rises, inf if none (approach and ascent treads)
      rise_bwd   the same for -x travel (run-out and descent treads)
      direction  +1 on an ascent tread, -1 on a descent tread, 0 elsewhere
    """
    top0 = (n_up - 1) * run
    land_end = n_up * run + land
    desc_end = land_end + (n_up - 1) * run
    kmax = float(max(n_up - 2, 0))
    at = t if at is None else at
    ka = torch.clamp(torch.floor(at / run), 0.0, kmax)
    kd = torch.clamp(torch.floor((at - land_end) / run), 0.0, kmax)
    asc = (at >= 0.0) & (at < top0)
    lnd = (at >= top0) & (at < land_end)
    dsc = (at >= land_end) & (at < desc_end)
    kind = torch.where(at < 0.0, K_APPROACH, torch.where(asc, K_ASCENT, torch.where(
        lnd, K_LANDING, torch.where(dsc, K_DESCENT, K_RUNOUT))))
    nan = float("nan")
    across = torch.where(asc, t - ka * run, torch.where(lnd, t - top0, torch.where(dsc, t - land_end - kd * run, nan)))
    width = torch.where(lnd, land_end - top0, run)
    d_edge = torch.minimum(across, width - across)
    d_drop = torch.where(asc, across, torch.where(dsc, run - across, torch.where(lnd, d_edge, nan)))
    index = torch.where(asc, ka, torch.where(lnd, float(n_up - 1), torch.where(dsc, n_up + kd, -1.0)))
    steps = torch.where(asc, ka + 1.0, torch.where(lnd, float(n_up), torch.where(dsc, (n_up - 1) - kd, 0.0)))
    inf = float("inf")
    rise_fwd = torch.where(at < 0.0, -t, torch.where(asc, run - across, inf))
    rise_bwd = torch.where(at >= desc_end, t - desc_end, torch.where(dsc, across, inf))
    direction = torch.where(asc, 1.0, torch.where(dsc, -1.0, 0.0))
    return {"kind": kind, "index": index, "across": across, "d_edge": d_edge, "d_drop": d_drop,
            "height": steps * riser, "rise_fwd": rise_fwd, "rise_bwd": rise_bwd, "direction": direction}


def nosing_crossing(t_prev, t, riser, run, n_up, land):
    """A tip moving from t_prev to t crossing an edge that RISES in its direction of travel: the ascent nosings
    (t = j*run, top (j+1)*riser, j = 0..n_up-1) forwards, the descent nosings (land_end + j*run, top
    (n_up-j)*riser) backwards. Motion per tick is far below one run, so at most one is crossed.
    -> (crossed bool, frac in [0,1] along the tick, nosing top height)."""
    land_end = n_up * run + land
    j = torch.floor(t / run)
    e = j * run
    fwd = (t_prev < e) & (e <= t) & (j >= 0.0) & (j <= n_up - 1)
    jb = torch.floor((t_prev - land_end) / run)
    eb = land_end + jb * run
    bwd = (t < eb) & (eb <= t_prev) & (jb >= 0.0) & (jb <= n_up - 1)
    ex = torch.where(fwd, e, eb)
    h = torch.where(fwd, (j + 1.0) * riser, (n_up - jb) * riser)
    dt = t - t_prev
    safe = torch.where(dt.abs() > 1e-9, dt, 1.0)
    frac = ((ex - t_prev) / safe).clamp(0.0, 1.0)
    return fwd | bwd, frac, h


def tread_frame_np(t, riser, run, n_up, land, at=None):
    """numpy mirror of tread_frame (one implementation: it runs the torch one on CPU tensors)."""
    tt = torch.as_tensor(np.asarray(t, np.float64))
    rr = torch.as_tensor(np.asarray(riser, np.float64))
    aa = None if at is None else torch.as_tensor(np.asarray(at, np.float64))
    return {k: v.numpy() for k, v in tread_frame(tt, rr, run, n_up, land, at=aa).items()}


# ---- reading the counters back (numpy; score_e0 and the aggregate) ---------------------------------------
def uniform_edges(n, w):
    """full edges of n uniform bins of width w from 0, the last bin open"""
    return np.concatenate([np.arange(n) * w, [np.inf]])


def full_edges(interior):
    return np.concatenate([[-np.inf], np.asarray(interior, np.float64), [np.inf]])


def hist_quantile(counts, edges, q):
    """q-quantile of a histogram with linear interpolation inside a finite bin; an open outer bin returns its
    finite edge. None for an empty histogram."""
    c = np.asarray(counts, np.float64)
    n = c.sum()
    if n <= 0:
        return None
    cum = np.cumsum(c)
    i = int(np.searchsorted(cum, q * n, side="left"))
    i = min(i, len(c) - 1)
    lo, hi = edges[i], edges[i + 1]
    if not np.isfinite(lo):
        return float(hi)
    if not np.isfinite(hi):
        return float(lo)
    prev = cum[i - 1] if i > 0 else 0.0
    f = (q * n - prev) / c[i] if c[i] > 0 else 0.0
    return float(lo + f * (hi - lo))


def col(ins, name):
    v = ins[name]
    return np.asarray(v, np.float64) if isinstance(v, (list, tuple, np.ndarray)) else float(v)


def _div(a, b):
    return (a / b) if b else None


def placement_summary(ins):
    """Per direction ('asc', 'desc') and pooled ('all'): touchdowns on real treads, median / mean d_edge,
    P(d_edge < 0.05), P(d_drop < 0.028), stalls, the across-tread histogram — with the chance values."""
    out = {}
    for dn in ("asc", "desc", "all"):
        parts = ("asc", "desc") if dn == "all" else (dn,)
        s = lambda key: sum(col(ins, f"{key}_{p}") for p in parts)
        n = s("tread_td")
        eh = s("edge_hist")
        ue = uniform_edges(EDGE_BINS, EDGE_W)
        nx = n - s("stall")                                       # the same numbers without the stall touchdowns
        out[dn] = {"touchdowns": int(round(n)), "median_d_edge": hist_quantile(eh, ue, 0.5),
                   "mean_d_edge": _div(s("edge_sum"), n), "p_edge_lt05": _div(s("edge_lt05"), n),
                   "p_drop_lt028": _div(s("drop_lt028"), n), "stalls": int(round(s("stall"))),
                   "touchdowns_excl_stalls": int(round(nx)),
                   "median_d_edge_excl_stalls": hist_quantile(eh - s("edge_hist_stall"), ue, 0.5),
                   "mean_d_edge_excl_stalls": _div(s("edge_sum") - s("edge_sum_stall"), nx),
                   "p_edge_lt05_excl_stalls": _div(s("edge_lt05") - s("edge_lt05_stall"), nx),
                   "p_drop_lt028_excl_stalls": _div(s("drop_lt028") - s("drop_lt028_stall"), nx),
                   "perched": int(round(s("perched"))) if "perched_asc" in ins else None,     # None: an older JSONL
                   "edge_hist": [int(round(v)) for v in eh], "across_hist": [int(round(v)) for v in s("across_hist")]}
    nh = col(ins, "nose_hist")
    out["nosing"] = {"crossings": int(round(col(ins, "nose_n"))),
                     "p10": hist_quantile(nh, full_edges(NOSE_EDGES), 0.10),
                     "median": hist_quantile(nh, full_edges(NOSE_EDGES), 0.50),
                     "p_scrape_lt02": _div(col(ins, "nose_lt02"), col(ins, "nose_n")),
                     "mean": _div(col(ins, "nose_sum"), col(ins, "nose_n"))}
    out["touchdowns"] = int(round(col(ins, "td")))
    out["by_kind"] = {k: int(round(v)) for k, v in zip(KINDS, col(ins, "td_kind"))}
    out["stalls"] = int(round(col(ins, "stall")))
    out["perched"] = int(round(col(ins, "perched"))) if "perched" in ins else None
    out["chance"] = CHANCE
    return out


def stability_summary(ins, counters, dt, termination="legacy"):
    """Falls per 100 m by cause, drift per metre, slip per stance-second, up_z against cos(local slope),
    recovery after the shove. `counters` = the block's STAT_KEYS row (episodes, terminations, pushes, ...);
    `termination` = the rule the counters ran under ('legacy' | 'honest' | anything else for a mixed pool)."""
    path = col(ins, "path_m")
    per100 = lambda n: (100.0 * n / path) if path > 0 else None
    t = counters["terminations"]
    rh = col(ins, "recovery_hist")
    legacy = termination == "legacy"
    return {"path_m": path, "falls": t, "falls_per_100m": per100(t), "termination": termination,
            # STAT_KEYS term_tilt / term_low split the terminations by up < 0.35 AT the terminating tick. Under the
            # legacy rule that IS its cause split (tipped over / base_above < 0.18 while upright); under any other rule
            # it is only tilted vs upright, so legacy_* are None there and legacy_cond_at_term (the legacy test true on
            # the terminating tick) is the reconciliation
            "by_cause_per_100m": {"tilt_sustained": per100(col(ins, "term_tilt_sustained")),
                                  "body_base": per100(col(ins, "term_body_base")),
                                  "body_uleg": per100(col(ins, "term_body_uleg")),
                                  "legacy_tilt": per100(counters["term_tilt"]) if legacy else None,
                                  "legacy_low": per100(counters["term_low"]) if legacy else None,
                                  "tilted_at_term": per100(counters["term_tilt"]),
                                  "upright_at_term": per100(counters["term_low"]),
                                  "legacy_cond_at_term": per100(col(ins, "term_legacy_cond"))},
            "term_counts": {k: int(round(col(ins, k))) for k in EPISODE_COLS if k.startswith("term_")},
            "reconcile": {k: int(round(col(ins, k))) for k in ("legacy_fired", "honest_fired", "honest_fired_tilt",
                                                               "honest_fired_body", "legacy_first", "honest_first")},
            "drift_per_m": _div(col(ins, "drift_abs_m"), path),
            "drift_signed_per_m": _div(col(ins, "drift_signed_m"), path),
            "drift_downhill_per_m": _div(col(ins, "drift_downhill_m"), path) if "drift_downhill_m" in ins else None,
            "slip_m_per_stance_s": _div(col(ins, "slip_m"), col(ins, "stance_ticks") * dt),
            "up_mean": _div(col(ins, "up_sum"), col(ins, "ticks")),
            "cos_slope_mean": _div(col(ins, "cos_slope_sum"), col(ins, "ticks")),
            "uprel_p05": hist_quantile(col(ins, "uprel_hist"), full_edges(UPREL_EDGES), 0.05),
            "uprel_p50": hist_quantile(col(ins, "uprel_hist"), full_edges(UPREL_EDGES), 0.50),
            "pushes": counters["pushes"], "recovered": int(round(col(ins, "recovered"))),
            "recovery_s_mean": _div(col(ins, "recovery_ticks") * dt, col(ins, "recovered")),
            "recovery_s_p95": (None if (q := hist_quantile(rh, uniform_edges(REC_BINS, REC_W), 0.95)) is None
                               else q * dt),
            "ray_misses": {"body": int(round(col(ins, "miss_body"))), "feet": int(round(col(ins, "miss_feet")))},
            "closed_form_mismatch": int(round(col(ins, "cf_mismatch"))),
            "offsurf_ticks": int(round(col(ins, "offsurf_ticks"))) if "offsurf_ticks" in ins else None,
            "offsurf_frac": _div(col(ins, "offsurf_ticks"), col(ins, "ticks")) if "offsurf_ticks" in ins else None,
            "knee": {"ticks": int(round(col(ins, "knee_ticks"))), "fired_episodes": int(round(col(ins, "knee_fired"))),
                     "at_termination": int(round(col(ins, "term_knee"))), "terminates": KNEE_TERMINATES}}


def gates_summary(ins):
    """p95 |commanded torque| and p95 |joint speed| per joint group, against the effort caps."""
    out = {}
    for g, cap in zip(GROUPS, TAU_CAP):
        out[g] = {"tau_p95": hist_quantile(col(ins, f"tau_hist_{g}"), uniform_edges(TAU_BINS, TAU_W), 0.95),
                  "tau_cap": cap,
                  "qd_p95": hist_quantile(col(ins, f"qd_hist_{g}"), uniform_edges(QD_BINS, QD_W), 0.95)}
    return out


# ---- per-touchdown records (score_e0 only, never training) -------------------------------------------------
class TouchdownRecorder:
    """Buffers SpotStepsEnv's per-tick touchdown outputs on the device and moves the touchdown rows to the host
    every `every` ticks (one nonzero + one copy per flush, not per tick). Call after_step() after each
    env.step(); the env's td_event / td_rec / tick / episode buffers are plain tensors, so a partial reset
    inside step() does not wipe the terminal tick's touchdowns before they are read here."""

    def __init__(self, env, every=64):
        self.env, self.every = env, int(every)
        K, dev = env.K, env.device
        self._ev = torch.zeros(self.every, K, 4, dtype=torch.bool, device=dev)
        self._val = torch.zeros(self.every, K, 4, len(TD_FIELDS), device=dev)
        self._tick = torch.zeros(self.every, K, dtype=torch.long, device=dev)
        self._ep = torch.zeros(self.every, K, dtype=torch.long, device=dev)
        self._i, self._chunks, self.flushes = 0, [], 0

    def after_step(self):
        e, i = self.env, self._i
        self._ev[i].copy_(e.td_event)
        self._val[i].copy_(e.td_rec)
        self._tick[i].copy_(e.td_tick)
        self._ep[i].copy_(e.td_episode)
        self._i += 1
        if self._i == self.every:
            self.flush()

    def flush(self):
        n = self._i
        if n:
            nz = torch.nonzero(self._ev[:n])                     # [m, 3] = (slot, env, foot)
            if nz.numel():
                s, k, f = nz.unbind(1)
                self._chunks.append((k.cpu().numpy(), f.cpu().numpy(), self._tick[s, k].cpu().numpy(),
                                     self._ep[s, k].cpu().numpy(), self._val[s, k, f].cpu().numpy()))
            self.flushes += 1
        self._i = 0

    def columns(self, block_id, episodes_done, episodes):
        """Rows of COMPLETED, COUNTED episodes only (episode < min(E, episodes that env finished)), which is
        exactly the set the device counters flush — so the row count must equal the 'td' counter."""
        self.flush()
        if self._chunks:
            k, f, tick, ep, val = (np.concatenate([c[i] for c in self._chunks]) for i in range(5))
        else:
            k = f = tick = ep = np.zeros(0, np.int64)
            val = np.zeros((0, len(TD_FIELDS)), np.float32)
        keep = (ep < episodes) & (ep < np.asarray(episodes_done)[k])
        cols = {"env": k[keep].astype(np.int32), "foot": f[keep].astype(np.int8),
                "tick": tick[keep].astype(np.int32), "episode": ep[keep].astype(np.int16),
                "block": np.asarray(block_id)[k[keep]].astype(np.int16)}
        ints = {"tread_index": np.int8, "kind": np.int8, "direction": np.int8, "stall": np.int8, "perched": np.int8}
        for j, name in enumerate(TD_FIELDS):
            v = val[keep, j]
            cols[name] = v.astype(ints[name]) if name in ints else v.astype(np.float32)
        return cols


def save_touchdowns(path, cols, env, block_names):
    """<json stem>_touchdowns.npz: the per-touchdown columns plus per-lane tread geometry (tent start, riser,
    lane y, block, scan offset) and the shared surface table, so a figure needs nothing else."""
    k = env.K
    band = (env._lane_band if env._lane_band is not None else env.level).cpu().numpy()
    tent_x0 = band * env.band_len + env.flat_approach
    riser = env.risers.cpu().numpy()[band] * env.is_stairs.cpu().numpy()
    surf = tent_surfaces(env.step_run, env.n_up, env.landing)
    scan_dx = np.zeros(k, np.float32) if env._scan_dx is None else env._scan_dx.cpu().numpy()
    np.savez_compressed(
        path, **cols,
        lane_y=env.lane_y.cpu().numpy(), lane_block=env.block_id.cpu().numpy().astype(np.int16),
        lane_is_stairs=env.is_stairs.cpu().numpy(), lane_tent_x0=tent_x0.astype(np.float32),
        lane_riser=riser.astype(np.float32), lane_scan_dx=scan_dx,
        single_band=np.array(env._lane_band is not None),
        surface_t_lo=surf[:, 0], surface_t_hi=surf[:, 1], surface_steps=surf[:, 2].astype(np.int8),
        step_run=np.float32(env.step_run), n_up=np.int8(env.n_up), landing=np.float32(env.landing),
        flat_approach=np.float32(env.flat_approach), band_len=np.float32(env.band_len),
        foot_r=np.float32(FOOT_R), block_names=np.array(block_names), kind_names=np.array(KINDS),
        readme=np.array("One row per touchdown (stance switching on: tip clearance < TD_ON with TD_OFF hysteresis, "
                        "not in an episode's first TD_ARM_TICKS) of a completed, counted episode. lane_x = tip x "
                        "- lane_tent_x0; tread_index indexes surface_t_lo/hi/steps (height = steps * lane_riser); "
                        "-1 = ground. tread_index / kind / across / edge / drop / direction describe the surface the "
                        "foot STANDS on: perched = 1 is a foot on the next surface's nosing with its centre past that "
                        "surface's edge (lane_x lies on the surface below), so across_tread < 0 or >= the width and "
                        "edge_dist / drop_dist < 0 = how far the centre hangs over. across_tread from the surface's "
                        "rear (low-x) edge. direction +1 ascent tread, -1 descent tread. nose_clear = min tip-bottom "
                        "clearance over a rising nosing between this foot's previous touchdown and this one, stance "
                        "ticks included, clamped at 0 (a contact); NaN = none crossed. y_lane = tip y - lane_y. "
                        "tick = control ticks into the episode."))
