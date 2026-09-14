"""spotv2 — DISCRETE STAIRS with an ADAPTIVE per-env difficulty curriculum.

Stairs are the hard case (a flat walker stalled at ~0.17 m because it can't lift its feet over a
tall riser). Two things make this attempt different: (1) warm-start from scratch_flat_best.pt — it has the
clock-augmented flat gait (OBS_DIM=50, normalize_obs=True, stiff gains=90); (2) an ADAPTIVE terrain-level
curriculum so each env only faces a riser it has earned.

Curriculum design (works within GpuSim's static terrain — terrain is built once, only the robot moves):
each STAIR lane is a strip of constant-difficulty BANDS along +x; band j is a flat approach + an up/down
"tent" at riser RISERS[j] + a run-out (so every band starts and ends at ground level). Each env carries a
`level` that selects its spawn band. After an episode the level PROMOTES if the robot cleared its tent
(forward distance > CLEAR_DIST) and DEMOTES only if it never got near it (forward distance < REACH_DIST);
a fall does not demote and does not block a promotion — so envs start at the smallest riser and ramp
only as the policy improves. FLAT_FRAC of lanes are kept flat (no tents) for full-steering replay.

Eval hooks, all opt-in and off by default (the defaults build exactly the training world above): an
instance riser ladder, explicit lane types, per-env start levels with the curriculum frozen, one band
per lane, held commands, one fixed-magnitude shove per episode at a random point on the tent, per-env
foot friction and payload, and device-side episode counters (episode_stats) that count what the old
`last_clear` sample-and-hold could not. Also opt-in: link reads, an honest termination rule, the placement /
stability / validity instrument (spot_feet.py), a scan-offset counterfactual on the observation, and per-lane
slope families (planar up / down / cross ramps from spot_slopes.py, with their own spawn and settle-health gate).

Training course (opt-in, course=True; spot_course.py, 2026-09-13): contiguous lane blocks of flat / stairs / hills /
cross-tilted bands / rough ground, six curriculum levels per family, per-family promotion, the IK stance and a held
settle on every hills / cross / rough spawn (partial resets included), and the ingredients the course trains with, each
behind its own switch: interval shoves on a device-side dv ramp, stand mode (a zero command freezes the clock and shows
the policy a (0,0) clock), a foot-placement reward on stair treads, and an explicit terrain material for friction DR.

New obs layout: [0:48] proprio | [48:50] clock | [50:51] base_above | [51:96] scan (45)  =  OBS_DIM=96.
First 50 dims byte-identical to scratch_flat → warm-start copies cols [0:50], terrain cols zero-init.
Anchor = scratch_flat_best.pt (50-d, norm-aware); stiff gains (90) on both training and deploy robots.
"""
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))   # scratch_clock / scratch_env

import threepp as tp
from threepp.rl import GpuSim, VecTask, load_policy
from spot_deploy import build_spot, default_q, add_to_isaac, isaac_to_add, ACTION_SCALE, LIM
from spot_terrain_env import (quat_rotate_inverse, up_z, heading_cossin, _flat_ground,
                              scan_offsets, scan_xy, N_SCAN,
                              CONTROL_HZ, DT, SUBSTEPS, SPACING, SPAWN_Z, PROBE_DX, ACT_DIM,
                              HIDDEN, HALF_W, VX_LO, VX_HI, VY_HI, WZ_HI, STAND_PROB,
                              FWD_DRIVE_FRAC, CMD_MIN, CMD_MAX, SIG)
from scratch_clock import CLOCK0, CLOCK_DIM, GAIT_PERIOD, advance, clock_obs, reset_phi
from scratch_env import STIFF_GAINS
import spot_feet as sf
import spot_slopes as ss
import spot_course as sc

OBS_DIM = 48 + CLOCK_DIM + 1 + N_SCAN   # = 96: [proprio(48)|clock(2)|base_above(1)|scan(45)]

# --------------------------------------------------------------------------- #
#  Stair bands (the difficulty ladder along +x) + curriculum
# --------------------------------------------------------------------------- #
RISERS = (0.04, 0.07, 0.10, 0.13, 0.16, 0.20)   # per-LEVEL riser height (the difficulty axis)
N_LEVELS = len(RISERS)
N_UP = 3                       # steps up (= steps down) per tent
STEP_RUN = 0.30                # tread depth (m)
FLAT_APPROACH = 1.6            # flat run before each tent (room to spawn + line up)
LAND = 0.8                     # landing at the tent peak
RUNOUT = 0.9                   # flat run-out after each tent (back to ground level)
TENT_LEN = 2.0 * N_UP * STEP_RUN + LAND          # ascend + landing + descend
BAND_LEN = FLAT_APPROACH + TENT_LEN + RUNOUT     # one difficulty band's x-extent
SPAWN_OFF = 0.7               # spawn this far into the band's flat approach
# promote/demote are measured as forward distance FROM SPAWN (spawn is SPAWN_OFF into the approach):
CLEAR_DIST = (FLAT_APPROACH - SPAWN_OFF) + TENT_LEN   # cleared the whole tent -> promote
REACH_DIST = (FLAT_APPROACH - SPAWN_OFF) * 0.6        # never even reached the tent -> demote
STRIP_LEN = N_LEVELS * BAND_LEN

FLAT_FRAC = 0.30               # fraction of FLAT lanes (no tents) -> full-steering replay
W_IMIT = 0.25                  # scan-gated imitation anchor (hold the teacher's gait on flat patches;
                               # 0.2 let BACKWARD steering regress ~2x, so anchor harder — cf. heightfield)
STEPS_EPISODE_S = 16.0         # time to approach + climb a tent at the commanded speed
HALF_W_STEPS = SPACING * 0.5   # on-lane gate = FULL lane half-width (no flat strip to walk around)
HALF_W_BOX = SPACING           # tent box width = full lane -> tents tile with no gap between lanes
# Effective mass the base presents to an external force, MEASURED: settle the robot with gravity
# off, difference a force step against a no-force control, F*dt/dv came out 23.7-25.2 kg on all
# three axes. Spot's total is ~32 kg; the legs lag, so the base sees less. Used to turn a shove
# expressed as a velocity change into the force that delivers it.
PUSH_MASS = 24.0

# ---- the jump on command inside the walking task (opt-in: `jump`, `jump_obs`) ----------------------------------------
# spot_jump_env.py learned a jump as its own policy (spot_jump2, 2026-09-14: 99-100% jumped and landed from standing).
# Here the walking policy learns it: a trigger opens a JUMP_WINDOW-tick window and the observation grows two channels,
# obs[96:98] = (1, ticks since the trigger / JUMP_WINDOW) in the window, else (0, 0). The walking command stays in force,
# so a jump happens standing or walking. In the window the JUMP_OFF_TERMS walking terms are off and the jump terms pay.
JUMP_WINDOW = 60                 # control ticks (1.2 s), = spot_jump_env.WINDOW_TICKS
JUMP_MIN_STEPS = 50              # no trigger in an episode's first second (spawn holds, messy resets)
JUMP_AIR = 0.02                  # every foot this far above the terrain under it (the instrument's feet_clear) = airborne
JUMP_TAKEOFF_VZ = 0.2            # a take-off also needs the base rising (a fast crouch lifts the feet as the body drops)
JUMP_MIN_FLIGHT = 5              # ticks: a shorter flight re-arms the jump
JUMP_LAND_UP = 0.85
JUMP_VZ_CAP = 3.0
J_RISE, J_FLIGHT, J_LAND, J_MISS, J_VZ = 150.0, 3.0, 50.0, 30.0, 40.0
J_REF_KEYS = ((0, (0.0, 0.0, 0.0)), (12, (0.0, 0.3, -0.6)), (16, (0.0, 0.3, -0.6)), (24, (0.0, -0.3, 0.6)),
              (40, (0.0, 0.1, -0.3)), (60, (0.0, 0.0, 0.0)))          # = spot_jump_env.REF_KEYS
J_IMIT_SIGMA2 = 0.1
JUMP_OFF_TERMS = ("imit", "vz", "scrape", "stand")   # the teacher's gait, the vertical-speed and scrape penalties, the stand
JUMP_BUCKETS = ("stand", "walk", "fast")             # the command at the trigger: zero, |vx| < 1 m/s, faster
JUMP_OBS_DIM = OBS_DIM + 2
# blowup_guard (opt-in): a robot whose state is non-finite or runs away terminates and earns nothing that tick, and every
# observation is finite and within +-OBS_LIMIT. spot_unified1 (2026-09-14): one exploding robot (vz, angrate and stand
# terms at -7e22 .. -1e28) wrecked the return scaling in all four runs, and ppo.py skips non-finite losses from then on.
BLOWUP_V, BLOWUP_W, BLOWUP_JV = 30.0, 50.0, 150.0     # m/s base speed, rad/s base spin, rad/s any joint
OBS_LIMIT = 1.0e3
FOOT_DX = (0.30, 0.30, -0.30, -0.30)   # stance foot offsets for spawn clearance
FOOT_DY = (0.17, -0.17, 0.17, -0.17)
# A clear only counts if the robot was in its own lane WHEN IT CROSSED: CLEAR_DIST is x-progress
# alone, so a robot shoved round the tent into the neighbour's lane would otherwise read as having
# climbed it. Read at the crossing, not at the end of the episode — the tent is cleared ~10 s before
# a 16 s episode times out, and drifting over the run-out after that is a steering story, not a
# climbing one (it is still counted, as out_of_lane).
IN_LANE = 1.3
# Episode counters (episode_stats). One row per (block, lane type, episode-index bin). The fall
# location is band-relative: approach / ascent / landing / descent / run-out / past the run-out (for
# the top band that is past the strip). Flat lanes get the same bins against a virtual tent at the
# same distance from their spawn, so a flat cell and a stair cell are read on one ruler.
# term_tilt = up < 0.35 at the terminating tick; term_low = a termination with up >= 0.35 — the two partition
# the terminations. Under the legacy rule that is its cause split (fell over / base_above < 0.18 while still
# upright, the base came down onto the terrain); under termination="honest" term_low is every termination that
# came with the robot upright (a body contact), whatever base_above read — the legacy rule's own verdict is the
# instrument's term_legacy_cond / legacy_fired there. crossed = the x-progress half of a
# clear on its own, so a success reads as climbing (crossed) times holding the lane (cleared/crossed)
# instead of one number that mixes them.
STAT_KEYS = ("episodes", "timeouts", "terminations", "term_tilt", "term_low",
             "crossed", "cleared", "success", "fell_before_clear", "fell_after_clear",
             "fall_approach", "fall_ascent", "fall_landing", "fall_descent", "fall_runout", "fall_past",
             "pushes", "pushed_episodes", "falls_after_push", "falls_pushed_episode", "out_of_lane")
_LOC_EDGES = (0.0, N_UP * STEP_RUN, N_UP * STEP_RUN + LAND, TENT_LEN, TENT_LEN + RUNOUT)
# lane_family names -> the spot_slopes family each builds. Stair and flat lanes build no slope geometry.
LANE_FAMILIES = {"stairs": "flat", "flat": "flat", "slope_up": "up", "slope_down": "down", "cross": "cross",
                 "hills": "flat", "cross_bands": "flat", "rough": "flat"}     # the last three are spot_course lanes
assert sc.SPAWN_OFF == SPAWN_OFF, "spot_course.SPAWN_OFF has drifted from spot_steps_env.SPAWN_OFF"
# Foot placement reward (w_place > 0): per touchdown on a stair tread, min(d_edge, PLACE_SAT) / PLACE_SAT, and
# -PLACE_PERCH for a touchdown perched on a nosing (spot_feet.PERCH_RISE). 0.10 m is two thirds of the way to a 0.30 m
# tread's centre: past it every placement is as good as any other, so the reward cannot pull feet onto the centre line.
PLACE_SAT = 0.10
PLACE_PERCH = 1.0
# Stand mode (stand_mode=True): a stand env pays W_STAND * mean(joint_vel^2) per step, the cheap form of the stand-motion
# penalty in plans/spot-frontier.md "Standing still". Measured 2026-09-13 (course smoke, K=256, 1200 steps, raycast_s4
# parent act_mean, stand mode on): mean joint_vel^2 over 23k stand env-steps 1.22 rad^2/s^2 against 4.1-4.2 reward per
# step, so 0.25 charges the parent's in-place motion ~0.31 per step (~7%).
W_STAND = 0.25
SPAWN_HOLD = 10      # control ticks a course IK spawn holds its joints at the start of every episode (= TD_ARM_TICKS)
EARLY_TICKS = 50     # reset_noise: a fall within this many ticks (1 s) of the spawn is counted as a spawn fall, per family
# reset_noise (0.08, 0.06, 0.3) with nohold 0.3, MEASURED 2026-09-13 (course smoke, K=2048, hard ladder, cmd_switch
# (3, 8, 0.3), raycast_s4 act_mean, 1000 steps): hold-window settle failures flat 7/501 (1.4%), stairs 3/1130, hills
# 1/928, cross 0/731, rough 0/673; IK failures 0 of 2350; falls within 1 s held / unheld flat 5/503 1/202, stairs 5/1141
# 0/473, hills 1/931, cross 7/739, rough 1/680. No family over spot_slopes.SPAWN_FAIL_MAX (2%).
SWITCH_YAW_RATE = 0.5    # reset_noise: initial yaw rate U(-this, this) rad/s

CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "substeps": SUBSTEPS, "spacing": SPACING,
          "terrain": "steps", "risers": list(RISERS), "n_up": N_UP, "step_run": STEP_RUN,
          "band_len": BAND_LEN, "episode_s": STEPS_EPISODE_S, "probe_dx": list(PROBE_DX),
          "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden": list(HIDDEN),
          "vx": [VX_LO, VX_HI], "vy_hi": VY_HI, "wz_hi": WZ_HI, "stand_prob": STAND_PROB,
          "sig": SIG, "w_imit": W_IMIT,
          "stiff_gains": {k: list(v) for k, v in STIFF_GAINS.items()},
          "gait_period": GAIT_PERIOD}


def _tent(t, riser, run=STEP_RUN, n=N_UP, land=LAND):
    """Up/down tent height at LOCAL x `t` (t<0 = before the tent -> 0; ascend n -> landing -> descend n
    -> 0). Vectorized; t and riser broadcast. riser may be a per-element tensor."""
    up_end = n * run
    land_end = up_end + land
    tent_end = land_end + n * run
    asc = torch.clamp(torch.floor(t / run) + 1.0, 0.0, float(n))
    desc = torch.clamp(torch.floor((t - land_end) / run) + 1.0, 0.0, float(n))
    steps = torch.where(t < 0.0, torch.zeros_like(t),
            torch.where(t < up_end, asc,
            torch.where(t < land_end, torch.full_like(t, float(n)),
            torch.where(t < tent_end, float(n) - desc, torch.zeros_like(t)))))
    return steps * riser


def _per_env(v, k, dtype, name):
    """A scalar or a length-k sequence/tensor -> a [k] numpy array of `dtype`."""
    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()
    a = np.asarray(v, dtype=dtype)
    if a.ndim == 0:
        return np.full(k, a.item(), dtype=dtype)
    a = a.reshape(-1)
    if a.shape[0] != k:
        raise ValueError(f"{name}: expected a scalar or {k} values, got {a.shape[0]}")
    return a.copy()


def _add_steps(world, k, spacing, is_stairs, risers_np, w=HALF_W_BOX, bands=None, material=None):
    """Per STAIR lane: the band ladder — for each level j a tent at riser risers_np[j], placed at band
    j's x-offset (flat approach in front, run-out behind). Solid boxes from the ground up to each tread
    top. `bands` [k], when given, builds only that one band per lane: an eval world with frozen levels,
    where each lane meets one tent and then flat ground, as the top rung always has. `material` (a
    world.create_material object) is an added contact; None makes exactly the historical call."""
    add = world.add_static if material is None else (lambda m: world.add_static(m, material))
    for i in range(k):
        if not bool(is_stairs[i]):
            continue
        y = i * spacing
        for j in (range(len(risers_np)) if bands is None else (int(bands[i]),)):
            r = float(risers_np[j])
            x0 = j * BAND_LEN + FLAT_APPROACH                      # tent starts after the flat approach
            for s in range(N_UP):                                  # ascend: tread s top at (s+1)*r
                h = (s + 1) * r
                b = tp.Mesh(tp.BoxGeometry(STEP_RUN, w, h), tp.MeshStandardMaterial())
                b.position.set(x0 + s * STEP_RUN + STEP_RUN * 0.5, y, h * 0.5)
                add(b)
            up_end = x0 + N_UP * STEP_RUN
            top = N_UP * r
            lb = tp.Mesh(tp.BoxGeometry(LAND, w, top), tp.MeshStandardMaterial())   # landing
            lb.position.set(up_end + LAND * 0.5, y, top * 0.5)
            add(lb)
            land_end = up_end + LAND
            for s in range(N_UP - 1):                              # descend: tread s top at (n-1-s)*r
                h = (N_UP - 1 - s) * r
                b = tp.Mesh(tp.BoxGeometry(STEP_RUN, w, h), tp.MeshStandardMaterial())
                b.position.set(land_end + s * STEP_RUN + STEP_RUN * 0.5, y, h * 0.5)
                add(b)


class SpotStepsEnv(VecTask):
    control_hz = CONTROL_HZ
    episode_s = STEPS_EPISODE_S
    act_dim = ACT_DIM
    control = "drive"                # stiff position PD drives -> targets, not forces
    substeps = SUBSTEPS
    settle_steps = 20                # settle to a clean stand (default targets) after a full reset
    clip_actions = None              # the policy emits ~[-8,8]; do NOT clamp

    def __init__(self, num_envs=1024, device="cuda", seed=0, flat_only=False,
                 height_source=None, perceive=False, perceive_noise=0.0, graph=False,
                 cadence_jitter=0.0, push_vel=0.0, push_prob=0.02,
                 risers=None, lane_types=None, init_level=None, freeze_level=False,
                 single_band=False, hold_cmd=None, push_mode="random", push_dv=None,
                 push_lanes=None, push_window=100, foot_mu=None, foot_combine="min",
                 payload_kg=None, block_id=None, count_episodes=None, drive_limits_are_forces=False,
                 read_links=False, termination="legacy", instrument=False, scan_offset=None,
                 lane_family=None, slope_deg=None, slope_material=None, slope_cross_mode="corrugate",
                 course=False, course_shares=None, rough_backend="trimesh", spawn_hold=SPAWN_HOLD,
                 terrain_material=None, push_interval=(3.0, 8.0), push_dv_min=0.3, push_max=0.5,
                 stand_mode=False, w_stand=W_STAND, w_place=0.0, course_ladder=None, cmd_switch=None,
                 reset_noise=None, reset_nohold_frac=0.3, jump=None, jump_obs=False, jump_lanes=None,
                 w_jump_imit=2.0, qd_max=None, self_collision=False, blowup_guard=False):
        """Everything after push_prob is opt-in, and its default reproduces the training world:

        risers         per-level riser ladder (None = RISERS). More levels also lengthen the ground
                       so the top band's spawn plus a full episode at VX_HI stays on it.
        lane_types     [K] bool, True = stair lane; replaces the FLAT_FRAC draw.
        init_level     int or [K] start level (also settable later through env.level before reset()).
        freeze_level   on_done leaves env.level alone; the counters still update.
        single_band    each stair lane builds only its start level's band (needs freeze_level).
        hold_cmd       [K,3] or [3] command re-written every step and after every reset (a row of
                       NaN leaves that env on the sampler).
        push_mode      'random' (U(0, push_vel) at push_prob per step, today's shoves) or
                       'once_on_tent': one shove of exactly push_dv m/s per episode, uniform horizontal
                       direction, fired when the base first passes a trigger x drawn uniformly over the
                       env's tent (stair lanes) or over [spawn+1, spawn+6] m (flat lanes).
        push_lanes     [K] bool, which lanes may be shoved (None = the stair lanes, as training does).
        push_window    control steps after a shove within which a fall is attributed to it.
        foot_mu        scalar or [K] foot friction (static = dynamic = mu, restitution 0, combine
                       `foot_combine`); NaN = the default material. None = default feet everywhere.
        payload_kg     scalar or [K] extra base mass; shoves are sized with PUSH_MASS + payload so a
                       commanded dv stays a dv.
        drive_limits_are_forces  make the joint effort caps real torques (45/45/115 N·m). Off, PhysX
                       treats them as impulses, i.e. max_force/dt: the plant every existing checkpoint
                       was trained on.
        block_id       [K] int cell id for the counters, so one world can host many cells.
        count_episodes E: the counters take only each env's first E episodes (None = all).
        read_links     also read every link's pose and velocity each tick. termination='honest' and
                       instrument=True need the feet and upper legs, so either one turns it on.
        termination    'legacy': up < 0.35 or base_above < 0.18 under the base centre (the training rule).
                       'honest': up < 0.35 held spot_feet.TILT_TICKS ticks, or a base sample or an upper-leg
                       hip-end / middle sample within BODY_CONTACT of the terrain below it (a raycast miss
                       reads as the lane ground, never as "far below", and is counted). The upper leg's knee end
                       is recorded as its own cause and does not terminate (measured: it skims nosings on
                       normal climbs; spot_feet.KNEE_TERMINATES). obs[50] is base_above under both rules.
        instrument     device counters (see spot_feet.TICK_COLS): touchdowns placed against the tread closed
                       form, nosing clearance and stalls; path, falls by honest cause, drift, foot slip, up_z
                       against the local slope, recovery after the shove; p95 commanded torque and joint
                       speed. Both termination rules are evaluated every tick whichever one terminates, so
                       their causes reconcile on the same episodes. Also fills the per-tick touchdown buffers
                       (td_event, td_rec, feet_clear) that score_e0's recorder reads.
        scan_offset    scalar or [K] metres added along the heading to the OBSERVATION's scan points only;
                       reward, termination, spawn and every counter stay on the true terrain. If placement
                       follows the scan, the touchdowns move with it.
        lane_family    [K] of 'stairs' | 'flat' | 'slope_up' | 'slope_down' | 'cross'; replaces lane_types
                       (stairs = a stair lane, everything else a flat-type lane with its own geometry). A slope
                       lane is a spot_slopes.SlopeLanes lane at `slope_deg` built from rotated static boxes; it
                       needs height_source='raycast' (the analytic branch cannot represent a plane), reads link
                       poses, spawns on the IK stance (spot_slopes.spawn_pose) at SPAWN_OFF along the lane, holds
                       those joints through a full reset's settle and is then checked by settle_health (counted
                       per block in episode_stats()['rows'][i]['spawn']; nothing else about the row changes). With
                       termination='honest' or the instrument, body and foot clearances on a slope lane are
                       measured perpendicular to the local plane. Placement counters stay stair-only.
        slope_deg      scalar or [K] slope angle in degrees for the slope lanes (0 = the shared ground).
        slope_material None | a world.create_material 5-tuple | [K] of those, for the slope boxes (ramps, plateaus).
                       An explicit material is an added cell; its feet need foot_mu too (spot_slopes.GRIPPY).
        slope_cross_mode  spot_slopes cross_mode. 'corrugate' (its default) alternates the tilt lane to lane, so a
                       row of cross lanes is a zig-zag with a V gutter at every lane edge; 'offset' makes a contiguous
                       run of cross lanes at one angle ONE plane, every lane pulling toward -y. Measured 2026-09-12
                       (spot_steps.pt, K=256): on corrugated lanes every cross episode ends in the gutter, 1.29-1.41 m
                       toward gravity at 5, 10, 15, 20 and 25 deg alike, so drift and sliding are capped there.
        course         True: the training course (spot_course.py). The lanes are laid out as contiguous family blocks
                       by `course_shares` (replaces lane_family / lane_types / flat_only), height_source defaults to
                       'raycast' (required), block_id defaults to the family code (episode_stats rows per family), and
                       the curriculum is per family: stairs as today; hills / cross / rough promote on walking their
                       band's feature (spot_course.FEATURE_DIST) and demote on a fall short of it or a timeout under
                       spot_course.DEMOTE_PROGRESS of the commanded forward distance. Every non-flat lane gets the stair
                       lanes' commands (forward drive, a forced forward command at spawn) and, by default, the shoves.
                       lane_family 'hills' | 'cross_bands' | 'rough' builds the same lanes without course=True (score_e0
                       cells: with single_band each lane builds only its level's band).
        course_shares  dict or 'flat=0.15,stairs=0.35,...' (spot_course.DEFAULT_SHARES).
        rough_backend  'trimesh' (one cooked trimesh per rough lane; the BVH is that very Mesh) or 'heightfield'.
        spawn_hold     control ticks a hills / cross / rough spawn holds its IK joints at the start of EVERY episode, so a
                       partial reset (which has no settle) gets one too; the policy's action is overridden for those
                       rows and ticks, last_act says what the drives held, and settle health is checked at the end of
                       the window (episode_stats()['rows'][i]['spawn']['hold_*']).
        terrain_material  None | a world.create_material 5-tuple for the ground, the tents and the hills / cross boxes (the
                       rough colliders cannot take one from Python). Friction DR is this plus per-env foot_mu.
        push_mode 'interval' (course training): per env a countdown drawn U(push_interval) s; when it runs out a shove of
                       dv ~ U(push_dv_min, push_max) m/s, uniform horizontal direction, on push_lanes (default: every
                       non-flat lane) and the countdown is redrawn. push_max is a device scalar the trainer ramps
                       (set_push_max), so the value is never baked into anything.
        stand_mode     a command of exactly (0,0,0) freezes the gait clock, shows the policy a (0,0) clock (a value
                       sin/cos never takes), zeroes its imitation weight and charges w_stand * mean(joint_vel^2).
                       The deploy side must send the same sentinel (_common.v2_obs(phi=None)).
        w_place        > 0: the foot-placement reward on stair treads (PLACE_SAT, PLACE_PERCH), from the instrument's
                       touchdowns; needs termination='honest' or instrument=True. Logged as reward term 'place'.
        course_ladder  spot_course.LADDERS key for the hills / cross / rough levels (and, with course=True, the stair
                       risers when `risers` is None). Default 'hard' with course=True, 'pilot' otherwise (score_e0's
                       course cells are named after the pilot ladder's values).
        cmd_switch     (lo, hi, p_standgo): every env's command is resampled every U(lo, hi) s instead of the
                       CMD_MIN..CMD_MAX step timer. With probability p_standgo the switch is abrupt: a standing env
                       (command exactly 0) gets (U(1.0, VX_HI), 0, 0), a walking one exactly (0, 0, 0); otherwise the usual
                       sampler (forward drive on terrain lanes). Counted per family (switch_counts()). With stand_mode the
                       clock freezes and the (0,0) sentinel shows from the tick the zero command is in force.
        reset_noise    (q_std, drop_max, v_max): every reset adds N(0, q_std) rad to the joint preset (default stance or IK
                       stance, clipped to the joint limits; the base is lifted by however far the noise lowered a foot),
                       U(0, drop_max) m of spawn height, a horizontal base velocity U(-v_max, v_max) per axis and a yaw rate
                       U(-SWITCH_YAW_RATE, SWITCH_YAW_RATE). Hills / cross / rough keep their IK hold (the target stays the
                       clean IK stance); flat and stair resets hold the default stance for spawn_hold ticks too, except a
                       share reset_nohold_frac of them, which get no hold and must catch themselves. Spawn falls per
                       family: spawn_families().
        jump           (lo, hi) seconds: on the jump lanes (default the flat lanes) a jump trigger every U(lo, hi) s after the
                       window closes (the first after JUMP_MIN_STEPS plus one draw), and the 98-d observation (module
                       constants JUMP_*). Needs the instrument's foot clearance: termination='honest' or instrument=True.
                       Counters per command bucket: jump_stats().
        jump_obs       the 98-d observation with the jump channels at zero and no triggers (evaluating a jump checkpoint).
        w_jump_imit    the jump reference's weight (J_REF_KEYS, as spot_jump_env's w_imit).
        qd_max         V rad/s: the drive targets move at most V * DT per control tick (spot_recovery_env's limit, in this
                       env's action units), outside the settle.
        self_collision the legs collide with the body and each other (the recovery and jump plant).
        blowup_guard   terminate a robot with a non-finite state, base speed > BLOWUP_V, spin > BLOWUP_W or a joint
                       faster than BLOWUP_JV, zero its reward terms that tick, and keep every observation finite within
                       +-OBS_LIMIT (counted: blowup_count()).
        """
        course_ladder = course_ladder or ("hard" if course else "pilot")
        if course_ladder not in sc.LADDERS:
            raise ValueError(f"course_ladder must be one of {list(sc.LADDERS)}, got {course_ladder!r}")
        if course:
            if lane_family is not None or lane_types is not None or flat_only:
                raise ValueError("course=True lays out its own lanes: pass none of lane_family, lane_types, flat_only")
            lane_family = sc.lane_families(num_envs, course_shares)
            if height_source is None:
                height_source = "raycast"
            if risers is None and sc.LADDERS[course_ladder]["risers"] is not None:
                risers = sc.LADDERS[course_ladder]["risers"]
        self.course_ladder = course_ladder
        if height_source is None:              # perception needs the BVH, so it implies the raycast
            height_source = "raycast" if perceive else "analytic"
        if height_source not in ("analytic", "raycast"):
            raise ValueError(f"height_source must be 'analytic' or 'raycast', got {height_source!r}")
        if perceive and height_source != "raycast":
            raise ValueError("perceive=True needs height_source='raycast': the camera rays and the "
                             "elevation map both come off the same BVH")
        self.height_source = height_source
        self.rays = None                       # set below when height_source == "raycast"
        self.percept = None                    # set below when perceive
        slope_code = None                      # [K] spot_slopes family per lane, when any lane is a slope lane
        course_code = None                     # [K] spot_course family per lane, when any lane is a course terrain lane
        if lane_family is not None:
            if lane_types is not None or flat_only:
                raise ValueError("pass one of lane_family, lane_types or flat_only")
            fam = [str(f) for f in np.asarray(lane_family, dtype=object).reshape(-1)]
            if len(fam) != num_envs:
                raise ValueError(f"lane_family: expected {num_envs} values, got {len(fam)}")
            unknown = sorted(set(fam) - set(LANE_FAMILIES))
            if unknown:
                raise ValueError(f"lane_family: unknown {unknown}, expected one of {list(LANE_FAMILIES)}")
            is_stairs_np = np.array([f == "stairs" for f in fam], bool)
            code = np.array([ss.FAMILIES[LANE_FAMILIES[f]] for f in fam], np.int64)
            if (code != ss.FLAT).any():
                slope_code = code
                if height_source != "raycast":
                    raise ValueError("slope lanes need height_source='raycast': the analytic branch of _terrain_h "
                                     "reads 0 off the tents and cannot represent a plane")
                if perceive:
                    raise ValueError("perceive=True with slope lanes is not validated (the elevation map spans "
                                     "the stair strip)")
                if graph:
                    raise ValueError("slope lanes need the raycast, whose Warp launches do not replay from a CUDA "
                                     "graph (results/spot_seed_array_2026_09/README.md:72-73)")
            cc = sc.family_codes(fam)
            if course or np.isin(cc, sc.IK_SPAWN).any():
                course_code = cc
                if np.isin(cc, sc.IK_SPAWN).any():
                    if height_source != "raycast":
                        raise ValueError("hills / cross_bands / rough lanes need height_source='raycast'")
                    if perceive or graph:
                        raise ValueError("course terrain lanes need the eager raycast: no perceive, no graph")
        elif lane_types is not None:
            if flat_only:
                raise ValueError("pass lane_types or flat_only, not both")
            is_stairs_np = _per_env(lane_types, num_envs, bool, "lane_types")
        else:
            rng = np.random.default_rng(seed)
            is_stairs_np = np.ones(num_envs, bool) if not flat_only else np.zeros(num_envs, bool)
            if not flat_only:
                is_stairs_np[rng.random(num_envs) < FLAT_FRAC] = False     # FLAT lanes -> steering replay
        # The terrain ladder is per instance: everything below reads these, never the module constants.
        self.riser_list = tuple(float(r) for r in (RISERS if risers is None else risers))
        self.n_levels = len(self.riser_list)
        self.band_len = BAND_LEN
        self.strip_len = self.n_levels * BAND_LEN
        self.clear_dist, self.reach_dist = CLEAR_DIST, REACH_DIST
        self.step_run, self.n_up, self.landing, self.flat_approach = STEP_RUN, N_UP, LAND, FLAT_APPROACH
        if termination not in ("legacy", "honest"):
            raise ValueError(f"termination must be 'legacy' or 'honest', got {termination!r}")
        self.termination = termination
        self.instrument = bool(instrument)
        self._feet_on = self.instrument or termination == "honest"     # both need the link reads below
        if jump is not None and not self._feet_on:
            raise ValueError("jump reads the instrument's foot clearance: termination='honest' or instrument=True")
        risers_np = np.array(self.riser_list, np.float32)
        level0 = np.zeros(num_envs, np.int64)
        if init_level is not None:
            level0[:] = _per_env(init_level, num_envs, np.int64, "init_level")
            if level0.min() < 0 or level0.max() >= self.n_levels:
                raise ValueError(f"init_level must lie in 0..{self.n_levels - 1}")
        if single_band and not freeze_level:
            raise ValueError("single_band builds only the start band, so it needs freeze_level=True")
        bands = level0 if single_band else None
        # The legacy 6-band ladder keeps its historical 60 m box (x -10..50). A longer ladder grows the
        # box so the top band's spawn plus a whole episode at VX_HI stays on the ground.
        if self.n_levels <= N_LEVELS:
            ground_len, ground_cx = 60.0, 20.0
        else:
            x_hi = (self.n_levels - 1) * BAND_LEN + SPAWN_OFF + STEPS_EPISODE_S * VX_HI + 2.0
            ground_len, ground_cx = x_hi + 10.0, (x_hi - 10.0) * 0.5
        self.ground_extent = (ground_cx - ground_len * 0.5, ground_cx + ground_len * 0.5)
        # Slope lanes (opt-in): one SlopeLanes over every lane, stair and flat lanes as its 'flat' (builds nothing).
        self._slope_lanes = slope_lanes = None
        if slope_code is not None:
            deg_np = (np.zeros(num_envs) if slope_deg is None else _per_env(slope_deg, num_envs, np.float64, "slope_deg"))
            slope_lanes = self._slope_lanes = ss.SlopeLanes([LANE_FAMILIES[f] for f in fam], deg_np, spacing=SPACING,
                                                            cross_mode=slope_cross_mode)
            ext = slope_lanes.ground_x_extent()      # the up approaches and down run-outs stand on the shared ground
            if ext is not None and (ext[0] < self.ground_extent[0] or ext[1] > self.ground_extent[1]):
                raise ValueError(f"slope lanes need ground over x {ext}, the ground box covers {self.ground_extent}")
        # Course terrain lanes (opt-in): the ground grows to the top band's spawn plus a whole episode at VX_HI, as a
        # longer riser ladder does (hills bands are 7.9 m, so the top one ends 15 m past the stair strip).
        self._course = course_lanes = None
        if course_code is not None and np.isin(course_code, sc.IK_SPAWN).any():
            course_lanes = self._course = sc.CourseLanes(course_code, SPACING, bands=bands, seed=seed,
                                                         rough_backend=rough_backend, ladder=course_ladder)
            x_hi = max(self.ground_extent[1], course_lanes.spawn_x_max() + STEPS_EPISODE_S * VX_HI + 2.0)
            x_lo = min(self.ground_extent[0], -10.0)
            ground_len, ground_cx = x_hi - x_lo, 0.5 * (x_lo + x_hi)
            self.ground_extent = (x_lo, x_hi)
        terrain_mat = []                                   # the explicit terrain material, kept alive with the env
        mu_np = None if foot_mu is None else _per_env(foot_mu, num_envs, np.float64, "foot_mu")
        pay_np = None if payload_kg is None else _per_env(payload_kg, num_envs, np.float64, "payload_kg")
        self.drive_limits_are_forces = dlf = bool(drive_limits_are_forces)
        self.self_collision = sc_on = bool(self_collision)
        foot_mats = {}
        # Stiff gains (90) = same plant the base gait scratch_flat_best.pt was trained on.
        class _SpotStepsRobot:
            def __init__(self_, world, i):
                kw = {}
                if mu_np is not None and np.isfinite(mu_np[i]):
                    # One material per distinct value (the scratch_env pattern), shared by every env
                    # that draws it; restitution 0, as the base gait's feet were.
                    mu = float(mu_np[i])
                    if mu not in foot_mats:
                        foot_mats[mu] = world.create_material(mu, mu, 0.0, foot_combine, foot_combine)
                    kw["foot_material"] = foot_mats[mu]
                if pay_np is not None and pay_np[i] != 0.0:
                    kw["payload_kg"] = float(pay_np[i])
                self_.art, _ = build_spot(world, assets=None, base_xy=(0.0, i * SPACING),
                                          gains=STIFF_GAINS, drive_limits_are_forces=dlf, self_collision=sc_on, **kw)
        # Under "raycast" the builders write into a CollectedWorld, which forwards every call
        # to the real world and keeps the Mesh it was handed — so the BVH is built from the
        # boxes PhysX actually got, not from a second description of them.
        collector = []

        def _build(world):
            if height_source == "raycast":
                from threepp.rl.raycast import CollectedWorld
                world = CollectedWorld(world)
                collector.append(world)
            if terrain_material is None:
                _flat_ground(world, num_envs, SPACING, length=ground_len, x_center=ground_cx)
                mat = None
            else:
                mat = world.create_material(*terrain_material)
                terrain_mat.append(mat)
                g = tp.Mesh(tp.BoxGeometry(ground_len, SPACING * num_envs + 20, 1.0), tp.MeshStandardMaterial())
                g.position.set(float(ground_cx), SPACING * (num_envs - 1) * 0.5, -0.5)    # = _flat_ground's box
                world.add_static(g, mat)
            _add_steps(world, num_envs, SPACING, is_stairs_np, risers_np, bands=bands, material=mat)
            if slope_lanes is not None:
                slope_lanes.build(world, slope_material)    # materials stay alive on slope_lanes
            if course_lanes is not None:
                course_lanes.build(world, material=mat, x_end=self.ground_extent[1],
                                   bvh_sink=world.meshes if height_source == "raycast" else None)

        super().__init__(num_envs, lambda world, i: _SpotStepsRobot(world, i),
                         gravity=(0.0, 0.0, -9.81), spacing=SPACING, device=device, seed=seed,
                         read_root=True, read_links=(bool(read_links) or self._feet_on or slope_lanes is not None
                                                     or course_lanes is not None or reset_noise is not None),
                         build_world=_build, graph=graph)
        self._terrain_mat = terrain_mat
        self.terrain_material = terrain_material
        if height_source == "raycast":
            from threepp.rl.raycast import TerrainRays
            self.rays = TerrainRays.from_objects(collector[0].meshes, device=self.device, up="z")
            collector[0].meshes.clear()      # the BVH owns the triangles now; drop the threepp objects
        dev = self.device
        self.default_q = torch.from_numpy(default_q).to(dev)
        self.i2a = torch.from_numpy(isaac_to_add.astype(np.int64)).to(dev)
        self.a2i = torch.from_numpy(add_to_isaac.astype(np.int64)).to(dev)
        self.stand_q_add = self.default_q[self.a2i].expand(num_envs, -1).contiguous()
        self.grav = torch.tensor([0.0, 0.0, -1.0], device=dev)
        self.risers = torch.from_numpy(risers_np).to(dev)                 # [N_LEVELS]
        self.is_stairs = torch.from_numpy(is_stairs_np).to(dev)           # [K] bool
        # The lanes that get the stair lanes' commands (forward drive, a forced forward command at spawn), are shoved by
        # default and are NOT the steering-replay set the flat gate reads. The stair lanes themselves, unless a course
        # lane exists, when it is every lane that is not flat. The same tensor object by default, so nothing changes.
        self._cc = self._clear_k = self._band_len_k = self._ep_cmd = None
        self._terrain_lane = self.is_stairs
        if course_code is not None:
            self._cc = torch.from_numpy(course_code).to(dev)
            self._terrain_lane = self._cc != sc.FLAT
            bl = np.where(course_code == sc.STAIRS, BAND_LEN, 0.0)
            fd = np.full(num_envs, CLEAR_DIST)                             # flat lanes: the virtual tent, as today
            if self._course is not None:
                bl = np.where(np.isin(course_code, sc.IK_SPAWN), self._course.band_len(), bl)
                fd = np.where(np.isin(course_code, sc.IK_SPAWN), self._course.feature_dist(), fd)
            self._band_len_k = torch.from_numpy(bl).float().to(dev)
            self._clear_k = torch.from_numpy(fd).float().to(dev)          # [K] per-lane 'crossed' distance
            self._ep_cmd = self.env_state(())                             # commanded forward distance this episode
            self._moves = torch.zeros(len(sc.FAMILIES), 3, dtype=torch.float64, device=dev)  # episodes, up, down
        # [K] family code for the per-family counters: the course family, else stairs / flat
        self._fam_k = (self._cc if self._cc is not None else
                       torch.where(self.is_stairs, sc.STAIRS, sc.FLAT).long())
        self._switch = None
        if cmd_switch is not None:
            lo, hi, p = (float(v) for v in cmd_switch)
            if not (0.0 < lo <= hi and 0.0 <= p <= 1.0):
                raise ValueError(f"cmd_switch must be (0 < lo <= hi seconds, 0 <= p_standgo <= 1), got {cmd_switch}")
            self._switch = (lo, hi, p)
            self._sw = torch.zeros(len(sc.FAMILIES), 3, dtype=torch.float64, device=dev)   # switches, stand->go, go->stand
        self.gx, self.gy = scan_offsets(dev)                              # [N_SCAN] heading-relative grid offsets
        # Anchor = the clock-aware base gait (50-d, normalize_obs=True); frozen throughout.
        _scratch = os.path.join(_HERE, "scratch_distillation", "scratch_flat_best.pt")
        self.anchor_ac, self.anchor_norm, _ = load_policy(_scratch, device=dev)
        self.anchor_ac.eval()
        self.lane_y = torch.arange(num_envs, device=dev, dtype=torch.float32) * SPACING
        pos = torch.zeros(num_envs, 3, device=dev); pos[:, 1] = self.lane_y; pos[:, 2] = SPAWN_Z
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)
        # CURRICULUM: per-env difficulty band — persists ACROSS episodes, so NOT env_state
        self.level = torch.from_numpy(level0).to(dev)
        self.freeze_level = bool(freeze_level)
        self._lane_band = self.level.clone() if single_band else None     # the one band each lane has
        self._foot_mats = foot_mats                                        # keep the materials alive
        # per-episode state: registered so the base re-inits it on every reset, full or partial
        self.last_act = self.env_state((ACT_DIM,))
        self.prev_act = self.env_state((ACT_DIM,))
        self.phi = self.env_state(())                                      # phase clock ∈ [0,1)
        self.cmd = self.env_state((3,))
        self.cmd_timer = self.env_state((), init=0, dtype=torch.long)
        self.ep_start_x = self.env_state(())
        self.ep_max_climb = self.env_state(())                             # forward distance this episode
        self.clear_dy = self.env_state((), init=-1.0)                      # |y - lane| when it crossed (-1 = not yet)
        self.jump_obs = bool(jump_obs) or jump is not None
        self.obs_dim = JUMP_OBS_DIM if self.jump_obs else OBS_DIM
        self._last_obs = torch.zeros(num_envs, self.obs_dim, device=dev)
        self.up = torch.zeros(num_envs, device=dev)
        self._resample_cmd(torch.arange(num_envs, device=dev))             # valid cmd before the first reset()
        # Per-env gait period. With one shared constant the policy can only vary stride LENGTH,
        # so it has no way to express a second gait however the terrain changes; jitter > 0 gives
        # each episode its own cadence and makes the clock something to read rather than assume.
        self.cadence_jitter = float(cadence_jitter)
        self.period = self.env_state((), init=GAIT_PERIOD)
        # The imitation weight lives on the device because the trainer anneals it: as a Python
        # float it would be baked into the step graph at capture and silently never change.
        self._imit_w = torch.full((), W_IMIT, device=dev)
        # Random shoves. Nothing ever perturbed this robot in training, so recovery was never a
        # behaviour that got selected for; `push_vel` is the velocity change a shove delivers, which
        # is the interpretable end of it (the force follows from PUSH_MASS and the substep).
        self.push_vel = float(push_vel)
        self.push_prob = float(push_prob)
        self._push_buf = torch.zeros(num_envs, self.sim.batch.max_links, 3, device=dev)
        if push_mode not in ("random", "once_on_tent", "interval"):
            raise ValueError(f"push_mode must be 'random', 'once_on_tent' or 'interval', got {push_mode!r}")
        self.push_mode = push_mode
        self.push_lanes = (self._terrain_lane.clone() if push_lanes is None else
                           torch.from_numpy(_per_env(push_lanes, num_envs, bool, "push_lanes")).to(dev))
        self.push_window = int(push_window)
        # dv -> force needs the mass the base presents; a payload rides on the base, so it adds to it
        self._push_mass = (None if pay_np is None else
                           torch.from_numpy(PUSH_MASS + pay_np).float().to(dev))
        self.push_dv = None
        if push_mode == "once_on_tent":
            dv = push_vel if push_dv is None else push_dv
            self.push_dv = torch.from_numpy(_per_env(dv, num_envs, np.float32, "push_dv")).to(dev)
            # its own stream, so where and which way a lane is shoved does not depend on how many
            # commands the sampler happened to draw before it
            self._push_g = torch.Generator(device=dev).manual_seed(int(seed) + 7919)
            self.push_trig_x = self.env_state(())
        if push_mode == "interval":
            # E1's interval shove curriculum (plans/spot-frontier.md): the countdown and the dv draw on their own stream,
            # push_max on the device so the trainer's ramp is read, not baked in
            lo, hi = (float(v) for v in push_interval)
            if not 0.0 < lo <= hi:
                raise ValueError(f"push_interval must be 0 < lo <= hi seconds, got {push_interval}")
            self.push_interval, self.push_dv_min = (lo, hi), float(push_dv_min)
            self._push_g = torch.Generator(device=dev).manual_seed(int(seed) + 7919)
            self._push_max = torch.full((), float(push_max), device=dev)
            self._push_cd = self.env_state((), init=0, dtype=torch.long)    # control ticks to the next shove
        # per-episode shove bookkeeping (both modes): how many landed, and the step of the last one
        self.n_push = self.env_state((), init=0, dtype=torch.long)
        self.push_step = self.env_state((), init=-1, dtype=torch.long)
        self._st_push = torch.zeros((), device=dev)
        self._st_track = torch.zeros((), device=dev)
        self._st_flat = torch.zeros((), device=dev)
        self._st_level = torch.zeros((), device=dev)
        self._st_fell = torch.zeros((), device=dev)
        self.last_clear = 0.0          # legacy: the clear rate of the LAST done batch only (sample-and-hold)
        # Held commands (the measure_tracking pattern, per env): written after the sampler every step
        # and after the forced stair-lane command on every reset, so the hold always wins.
        self._hold = self._hold_on = None
        if hold_cmd is not None:
            h = torch.as_tensor(hold_cmd, dtype=torch.float32, device=dev)
            h = (h.expand(num_envs, 3) if h.dim() == 1 else h).clone()
            self._hold_on = torch.isfinite(h).all(dim=1)
            self._hold = torch.nan_to_num(h)
        # Episode counters: written in on_done (eager), except the termination cause, which only
        # exists inside the step region — terminated() copies it into this persistent buffer so a
        # replayed graph writes the same address the counters later read.
        self._term_tilt = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        if block_id is None and course:
            block_id = course_code                          # one counter row per family (block names: sc.FAMILIES)
        self.block_names = list(sc.FAMILIES) if (course and block_id is course_code) else None
        self.block_id = (torch.zeros(num_envs, dtype=torch.long, device=dev) if block_id is None else
                         torch.from_numpy(_per_env(block_id, num_envs, np.int64, "block_id")).to(dev))
        self.n_blocks = int(self.block_id.max().item()) + 1
        self.count_episodes = None if count_episodes is None else int(count_episodes)
        self._ep_bins = self.count_episodes if self.count_episodes else 2   # else: first / later
        self.ep_index = torch.zeros(num_envs, dtype=torch.long, device=dev)  # episodes ended, per env
        self._stats = torch.zeros(self.n_blocks * 2 * self._ep_bins, len(STAT_KEYS),
                                  dtype=torch.float64, device=dev)
        self._prog_sum = torch.zeros(self.n_blocks * 2 * self._ep_bins, dtype=torch.float64, device=dev)
        self._loc_edges = torch.tensor(_LOC_EDGES, device=dev)
        cell = self.block_id * 2 + (~self.is_stairs).long()
        self._lanes = torch.bincount(cell, minlength=self.n_blocks * 2).cpu().numpy().reshape(-1, 2)
        self._scan_dx = (None if scan_offset is None else torch.from_numpy(np.nan_to_num(
            _per_env(scan_offset, num_envs, np.float32, "scan_offset"))).to(dev))
        self._hold_row = self._ik_row = self._hold_ep = self._spf = None   # rows that hold their spawn joints
        self._reset_noise = None
        if reset_noise is not None:
            qs, dmax, vmax = (float(v) for v in reset_noise)
            if min(qs, dmax, vmax) < 0.0 or not 0.0 <= float(reset_nohold_frac) <= 1.0:
                raise ValueError(f"reset_noise must be non-negative (q_std, drop_max, v_max) and reset_nohold_frac in "
                                 f"[0, 1], got {reset_noise}, {reset_nohold_frac}")
            self._reset_noise, self.reset_nohold_frac = (qs, dmax, vmax), float(reset_nohold_frac)
            # its own stream, so the noise a reset gets does not depend on how many commands were drawn before it
            self._reset_g = torch.Generator(device=dev).manual_seed(int(seed) + 15485863)
            self._q_lo = torch.tensor([LIM[g][0] for g in sf.GROUPS for _ in range(4)], device=dev)   # isaac order
            self._q_hi = torch.tensor([LIM[g][1] for g in sf.GROUPS for _ in range(4)], device=dev)
        if self._slope_lanes is not None:
            self._init_slopes(slope_code)
        if self._course is not None or reset_noise is not None:
            self._init_course_spawn(course_code if course_code is not None else self._fam_k.cpu().numpy(), spawn_hold)
        self.stand_mode, self.w_stand = bool(stand_mode), float(w_stand)
        self.jump = None if jump is None else tuple(float(v) for v in jump)
        self.qd_max = float(qd_max) if qd_max else None
        self.w_jump_imit = float(w_jump_imit)
        self.blowup_guard = bool(blowup_guard)
        self._blowup = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self._blow_n = torch.zeros((), dtype=torch.float64, device=dev)
        if self.jump is not None:
            if not (len(self.jump) == 2 and 0.0 < self.jump[0] <= self.jump[1]):
                raise ValueError(f"jump must be (0 < lo <= hi seconds), got {jump}")
            lanes = (self._fam_k == sc.FLAT) if jump_lanes is None else torch.as_tensor(np.asarray(jump_lanes, bool))
            self._jump_lane = lanes.to(dev)
            self.jump_t = self.env_state((), init=-1, dtype=torch.long)       # ticks since the trigger (-1: no window)
            self.jump_cd = self.env_state((), init=0, dtype=torch.long)       # ticks to the next trigger
            self.jstate = self.env_state((), init=0, dtype=torch.long)        # 0 pending, 1 flying, 2 landed
            self.jflight = self.env_state((), init=0, dtype=torch.long)
            self.jz0 = self.env_state(())
            self.jzmax = self.env_state(())
            self.jvzmax = self.env_state(())
            self.jclear = self.env_state(())
            self.jbucket = self.env_state((), init=0, dtype=torch.long)
            self.jumped = self.env_state((), init=False, dtype=torch.bool)
            self.jlanded = self.env_state((), init=False, dtype=torch.bool)
            self._uair = self.env_state((), init=0, dtype=torch.long)
            ref = np.zeros((JUMP_WINDOW, 3), np.float32)
            vals = np.array([v for _, v in J_REF_KEYS], np.float32)
            for g in range(3):
                ref[:, g] = np.interp(np.arange(JUMP_WINDOW), [k for k, _ in J_REF_KEYS], vals[:, g])
            self._jref = torch.from_numpy(np.tile(ref, (1, 4))).to(dev)        # [window, 12], add order per leg
            self._j_rise = torch.zeros(num_envs, device=dev)
            self._j_vz = torch.zeros(num_envs, device=dev)
            self._j_fly = torch.zeros(num_envs, dtype=torch.bool, device=dev)
            self._j_land = torch.zeros(num_envs, device=dev)
            self._j_miss = torch.zeros(num_envs, dtype=torch.bool, device=dev)
            # per bucket: windows, jumped, landed upright, missed, fell in the window, flight s, rise, clearance, peak vz
            self._jst = torch.zeros(len(JUMP_BUCKETS), 9, dtype=torch.float64, device=dev)
            self._juncmd = torch.zeros(2, dtype=torch.float64, device=dev)    # uncommanded flights, env ticks outside
        self.w_place = float(w_place)
        self._place_buf = None
        self._st_place = torch.zeros((), device=dev)
        if self.w_place > 0.0 or course:
            if not self._feet_on:
                if self.w_place > 0.0:
                    raise ValueError("w_place needs the touchdowns: termination='honest' or instrument=True")
            else:
                self._place_buf = torch.zeros(num_envs, device=dev)      # this tick's placement signal, unweighted
        self._istats = None
        if self._feet_on:
            self._init_instrument()
        if perceive:
            from threepp.rl.perception import PerceivedScan
            # Same mount, FOV and range as spot_depth_scan.ForwardDepthScanner, so the training
            # camera and the deploy camera are one camera. The map spans the strip and the lane.
            self.percept = PerceivedScan(self.rays, num_envs, origin_b=self.lane_y,
                                         bounds=(-1.0, self.strip_len + 1.0, HALF_W_STEPS),
                                         noise=perceive_noise, seed=seed)

    # The per-step stats live on the device so the reward can be replayed from a graph; these read
    # them back, which the trainers do once per iteration rather than once per step.
    last_track = property(lambda self: self._st_track.item())
    last_flat_track = property(lambda self: self._st_flat.item())
    last_level = property(lambda self: self._st_level.item())
    last_fell = property(lambda self: self._st_fell.item())
    last_push = property(lambda self: self._st_push.item())
    last_place = property(lambda self: self._st_place.item())    # unweighted placement signal per stair-lane step

    def set_push_max(self, v):
        """The interval shoves' dv ceiling (m/s), written through a device scalar like set_imit_weight."""
        self._push_max.fill_(float(v))

    def on_pre_substep(self):
        """Shove a random subset, world-frame and horizontal, for one substep.

        PhysX clears link forces after every step, so this lands as an impulse of
        push_vel * PUSH_MASS N*s and nothing persists. Skipped while the spawn is settling —
        knocking the robot over before the episode starts measures nothing.

        push_mode 'once_on_tent' instead lands exactly one shove of push_dv per episode, where the
        base first passes that episode's trigger x (see on_reset). push_mode 'interval' is _push_interval."""
        if self.push_mode == "interval":
            if not self.settling:
                self._push_interval()
            return
        if self.push_mode == "once_on_tent":
            if not self.settling:
                self._push_once()
            return
        if self.push_vel <= 0.0 or self.settling:
            return
        dev = self.device
        # Shove the STAIR lanes only (push_lanes defaults to them). The flat lanes exist as the
        # steering-replay set, and last_flat_track is what the anchor anneal gates on — shove them
        # and that number stops measuring steering and starts measuring shove recovery, so the anchor
        # lets go on a gate it only cleared because the baseline was shoved too. Measured: steering
        # regressed from a worst ratio of 1.19 to 1.65 exactly that way.
        hit = ((torch.rand(self.K, device=dev) < self.push_prob) & self.push_lanes).float()
        mag = torch.rand(self.K, device=dev) * self.push_vel * hit
        ang = torch.rand(self.K, device=dev) * (2.0 * math.pi)
        # dv -> the force that delivers it in one substep
        scale = PUSH_MASS / (DT / SUBSTEPS) if self._push_mass is None else self._push_mass / (DT / SUBSTEPS)
        self._push_buf.zero_()
        self._push_buf[:, 0, 0] = torch.cos(ang) * mag * scale     # link 0 is the root (measured)
        self._push_buf[:, 0, 1] = torch.sin(ang) * mag * scale
        self.sim.apply_link_force(self._push_buf)
        self._st_push.copy_(hit.mean())
        fired = hit > 0.0
        self.n_push += fired.long()
        self.push_step.copy_(torch.where(fired, self.steps, self.push_step))

    def _push_once(self):
        """One shove of exactly push_dv per episode, uniform horizontal direction, on the first
        control step the base x is past the episode's trigger. Eager (simulate runs outside the step
        graph), so the host-free masks here are all it needs."""
        dev = self.device
        fire = (self.push_lanes & (self.n_push == 0) & (self.push_dv > 0.0)
                & (self.sim.root_position[:, 0] >= self.push_trig_x))
        ang = torch.rand(self.K, device=dev, generator=self._push_g) * (2.0 * math.pi)
        mass = PUSH_MASS if self._push_mass is None else self._push_mass
        mag = self.push_dv * fire.float() * (mass / (DT / SUBSTEPS))
        self._push_buf.zero_()
        self._push_buf[:, 0, 0] = torch.cos(ang) * mag             # link 0 is the root (measured)
        self._push_buf[:, 0, 1] = torch.sin(ang) * mag
        self.sim.apply_link_force(self._push_buf)
        self._st_push.copy_(fire.float().mean())
        self.n_push += fire.long()
        self.push_step.copy_(torch.where(fire, self.steps, self.push_step))

    def _push_interval(self):
        """The interval shove curriculum: every push lane counts down its own U(push_interval) s; at zero it takes one
        shove of dv ~ U(push_dv_min, push_max) m/s in a uniform horizontal direction and redraws the countdown. No
        boolean indexing and no host read; the three draws come off the env's own stream every tick, so where and
        when a lane is shoved does not depend on the command sampler."""
        dev = self.device
        self._push_cd.sub_(1)
        fire = self.push_lanes & (self._push_cd <= 0)
        u = torch.rand(self.K, device=dev, generator=self._push_g)
        ang = torch.rand(self.K, device=dev, generator=self._push_g) * (2.0 * math.pi)
        nxt = torch.rand(self.K, device=dev, generator=self._push_g)
        dv = self.push_dv_min + u * (self._push_max - self.push_dv_min).clamp_min(0.0)
        mass = PUSH_MASS if self._push_mass is None else self._push_mass
        mag = dv * fire.float() * (mass / (DT / SUBSTEPS))
        self._push_buf.zero_()
        self._push_buf[:, 0, 0] = torch.cos(ang) * mag             # link 0 is the root (measured)
        self._push_buf[:, 0, 1] = torch.sin(ang) * mag
        self.sim.apply_link_force(self._push_buf)
        lo, hi = self.push_interval
        self._push_cd.copy_(torch.where(fire, ((lo + (hi - lo) * nxt) / DT).long(), self._push_cd))
        self._st_push.copy_(fire.float().mean())
        self.n_push += fire.long()
        self.push_step.copy_(torch.where(fire, self.steps, self.push_step))

    def set_imit_weight(self, w):
        """Anneal the imitation anchor. Written through a device scalar rather than rebound as a
        Python float, so a captured step graph sees the change instead of the value it was born
        with — the one failure mode verify_graph cannot see."""
        self._imit_w.fill_(float(w))

    def _terrain_h(self, x, y):
        """Ground height at (x,y). x,y [K] or [K,P].

        Under height_source="raycast" this is a ray straight down into the BVH over the boxes the
        world was actually built from, and none of the formula below runs. The analytic branch picks
        the band from x, evaluates that band's tent, and gates to stair lanes + lane width — it only
        answers correctly for terrain this env authored, which is the constraint the raycast lifts."""
        if self.rays is not None:
            return self.rays.heights(x, y)
        lane = self.lane_y if x.dim() == 1 else self.lane_y[:, None]
        stairs = self.is_stairs.float() if x.dim() == 1 else self.is_stairs.float()[:, None]
        band = torch.clamp(torch.floor(x / self.band_len).long(), 0, self.n_levels - 1)
        riser = self.risers[band]                                          # per-(x) riser of its band
        t = (x - band.to(x.dtype) * self.band_len) - FLAT_APPROACH         # local x within the band's tent
        h = _tent(t, riser)
        on = (torch.abs(y - lane) < HALF_W_STEPS) & (x >= 0.0) & (x < self.strip_len)
        if self._lane_band is not None:                                    # single_band: only its own tent
            on = on & (band == (self._lane_band if x.dim() == 1 else self._lane_band[:, None]))
        return h * on.float() * stairs

    def _sample_cmd(self, stairs):
        """Fresh commands for the n envs whose stair flags are `stairs` [n] -> (cmd [n,3], timer [n]).

        Nothing here selects with a boolean mask, so the shapes are fixed by n alone — which is what
        lets the step-path caller sample for every env and pick with `where` instead of `nonzero`."""
        n = stairs.numel()
        dev = self.sim.device
        vx = torch.empty(n, device=dev).uniform_(VX_LO, VX_HI)
        vy = torch.empty(n, device=dev).uniform_(-VY_HI, VY_HI)
        wz = torch.empty(n, device=dev).uniform_(-WZ_HI, WZ_HI)
        drive = (torch.rand(n, device=dev) < FWD_DRIVE_FRAC) & stairs   # stair lanes drive at the tent
        vx = torch.where(drive, torch.empty(n, device=dev).uniform_(0.4, VX_HI), vx)
        vy = torch.where(drive, vy * 0.2, vy)
        wz = torch.where(drive, wz * 0.2, wz)
        cmd = torch.stack([vx, vy, wz], dim=1)
        stand = (torch.rand(n, 1, device=dev) < STAND_PROB)
        cmd = torch.where(stand, torch.zeros_like(cmd), cmd)
        timer = torch.randint(CMD_MIN, CMD_MAX + 1, (n,), device=dev)
        if self._switch is not None:                  # cmd_switch: the next switch in U(lo, hi) seconds
            lo, hi, _ = self._switch
            timer = ((lo + (hi - lo) * torch.rand(n, device=dev)) / DT).long()
        return cmd, timer

    def _resample_cmd(self, idx):
        """Subset form, for the reset path. Runs eagerly — idx already cost a host sync."""
        if idx.numel() == 0:
            return
        cmd, timer = self._sample_cmd(self._terrain_lane[idx])
        self.cmd[idx] = cmd
        self.cmd_timer[idx] = timer

    def _resample_due(self):
        """Step form: sample for every env, keep it only where the timer ran out. K*3 extra randoms
        instead of a nonzero, which is the trade that keeps the step graph-capturable."""
        due = self.cmd_timer <= 0
        cmd, timer = self._sample_cmd(self._terrain_lane)
        if self._switch is not None:
            # the abrupt switches a keyboard makes: stand -> a brisk forward walk, or walking -> an exact stand
            p = self._switch[2]
            K, dev = self.K, self.device
            standing = (self.cmd == 0.0).all(dim=1)
            abrupt = torch.rand(K, device=dev) < p
            zero = torch.zeros(K, device=dev)
            go = torch.stack([torch.empty(K, device=dev).uniform_(1.0, VX_HI), zero, zero], dim=1)
            cmd = torch.where(abrupt[:, None], torch.where(standing[:, None], go, torch.zeros_like(cmd)), cmd)
            self._sw.index_add_(0, self._fam_k, torch.stack([due, due & abrupt & standing, due & abrupt & ~standing],
                                                            dim=1).double())
        self.cmd.copy_(torch.where(due.unsqueeze(1), cmd, self.cmd))
        self.cmd_timer.copy_(torch.where(due, timer, self.cmd_timer))

    def switch_counts(self):
        """cmd_switch: per family, the switches since the last call and how many were stand -> go / go -> stand."""
        if self._switch is None:
            return {}
        s = self._sw.cpu().numpy()
        self._sw.zero_()
        return {n: {"switches": int(s[c, 0]), "stand_to_go": int(s[c, 1]), "go_to_stand": int(s[c, 2])}
                for c, n in enumerate(sc.FAMILIES) if s[c, 0] > 0}

    def _spawn_x(self, idx):
        """Spawn x for the subset: stair lanes at their level's band approach; flat lanes anywhere flat. Course terrain
        lanes spawn SPAWN_OFF into their level's band, whose length is their family's."""
        dev = self.sim.device
        bl = self.band_len if self._band_len_k is None else self._band_len_k[idx]
        sx = self.level[idx].to(torch.float32) * bl + SPAWN_OFF
        flat = ~self._terrain_lane[idx]
        sx = torch.where(flat, torch.rand(idx.numel(), device=dev) * 3.0, sx)
        return sx

    def _foot_clear_z(self, idx, sx):
        """Highest terrain under the stance footprint at the spawn -> drop-settle (no spawn penetration)."""
        dev = self.sim.device
        fdx = torch.tensor(FOOT_DX, device=dev); fdy = torch.tensor(FOOT_DY, device=dev)
        fx = sx[:, None] + fdx[None, :]                                   # [n,4]
        fy = self.lane_y[idx][:, None] + fdy[None, :]                     # [n,4] world-y
        if self.rays is not None:
            return self.rays.heights(fx, fy).max(dim=1).values
        # evaluate terrain at the foot points for these specific lanes (per-row band + riser + gate)
        lane = self.lane_y[idx][:, None]
        stairs = self.is_stairs[idx].float()[:, None]
        band = torch.clamp(torch.floor(fx / self.band_len).long(), 0, self.n_levels - 1)
        riser = self.risers[band]
        t = (fx - band.to(fx.dtype) * self.band_len) - FLAT_APPROACH
        h = _tent(t, riser)
        on = (torch.abs(fy - lane) < HALF_W_STEPS) & (fx >= 0.0) & (fx < self.strip_len)
        if self._lane_band is not None:
            on = on & (band == self._lane_band[idx][:, None])
        return (h * on.float() * stairs).max(dim=1).values

    # ---- slope lanes (opt-in; see lane_family and spot_slopes) -------------------------------------------------
    def _init_slopes(self, code):
        """Buffers for the slope lanes, allocated only when a lane is one."""
        K, dev = self.K, self.device
        self._slope_row = torch.from_numpy(code != ss.FLAT).to(dev)           # [K] this lane is a slope lane
        self._slope_fk = sf.FootKin(dev)                                      # GPU link order (spot_feet, measured)
        # +1 / -1 where gravity pulls a cross lane toward +y / -y (the corrugation alternates it lane to lane), else 0
        self._downhill_y = torch.tensor([self._slope_lanes.downhill(i)[1] for i in range(K)], device=dev)
        self._slope_lanes.normals(self.lane_y[:1], self.lane_y[:1])          # build its device constants now, not in a step
        # the surface each lane is part of, so a robot that drifted over another one (a flat buffer, a guard of another
        # angle, the other face of a corrugation) is counted as off its surface rather than read as slope walking
        self._surf_id = torch.from_numpy(self._slope_lanes.surface_ids()).to(dev)
        self._offsurf_col = sf.LAYOUT["offsurf_ticks"][0]
        self._settle_act = torch.zeros(K, ACT_DIM, device=dev)               # the action that holds each spawn's joints
        self._spawn_nz = torch.ones(K, device=dev)                            # cos of the fitted slope under the spawn
        self._spawn_resets = torch.zeros(K, dtype=torch.long, device=dev)    # resets of a slope lane, full or partial ...
        self._spawn_bad = torch.zeros(K, dtype=torch.long, device=dev)       # ... whose IK clamped a leg or left a foot high
        self._settle_checked = torch.zeros(K, dtype=torch.bool, device=dev)  # the last full reset's settle, latched per env
        self._settle_bad = torch.zeros_like(self._settle_checked)
        self._settle_uprel = torch.ones(K, device=dev)
        self._settle_gap = torch.zeros(K, device=dev)
        self._hold_row = self._slope_row

    def _slope_spawn(self, idx, sx, pose, q0):
        """The slope rows of a reset: spawn at SPAWN_U along the lane on spot_slopes' IK stance (every foot 5 mm above
        the plane along its normal, the body tilted by half the fitted slope, heading +x) and remember the action
        that holds those joints through the settle. Writes pose in place; returns (sx, joint preset, add order).
        Rows that are not slope lanes keep the values they came in with. Eager, like the rest of on_reset."""
        slope = self._slope_row[idx]
        sx = torch.where(slope, torch.full_like(sx, self._slope_lanes.x0 + ss.SPAWN_U), sx)
        sp = ss.spawn_pose(self.rays.heights, sx, self.lane_y[idx], normal_fn=self._slope_lanes.normals)
        pose.copy_(torch.where(slope[:, None], torch.cat([sp.quat, sp.pos], dim=1).to(pose.dtype), pose))
        q0 = torch.where(slope[:, None], sp.joint_q[:, self.a2i].to(q0.dtype), q0)
        self._settle_act[idx] = torch.where(slope[:, None], ss.settle_action(sp.joint_q), 0.0)
        self._spawn_nz[idx] = torch.where(slope, sp.normal[:, 2], 1.0)
        bad = ~sp.reachable.all(dim=1) | (sp.gap.max(dim=1).values > ss.CONTACT_TOL)
        self._spawn_resets[idx] += slope.long()
        self._spawn_bad[idx] += (slope & bad).long()
        return sx, q0

    # ---- course terrain lanes (opt-in; see course and spot_course) -----------------------------------------------
    def _init_course_spawn(self, code, hold):
        """Spawn buffers for the hills / cross / rough rows (the slope lanes' ones, made here when no slope lane made
        them) plus the hold window's own settle-health counters. Also made for reset_noise, where flat and stair resets
        can hold too: `code` [K] is then the per-family code (every row FLAT or STAIRS without course lanes)."""
        K, dev = self.K, self.device
        if self._hold_row is None:
            self._slope_fk = sf.FootKin(dev)
            self._settle_act = torch.zeros(K, ACT_DIM, device=dev)
            self._spawn_nz = torch.ones(K, device=dev)
            self._spawn_resets = torch.zeros(K, dtype=torch.long, device=dev)
            self._spawn_bad = torch.zeros(K, dtype=torch.long, device=dev)
            self._settle_checked = torch.zeros(K, dtype=torch.bool, device=dev)
            self._settle_bad = torch.zeros_like(self._settle_checked)
            self._settle_uprel = torch.ones(K, device=dev)
            self._settle_gap = torch.zeros(K, device=dev)
        self._ik_row = torch.from_numpy(np.isin(code, sc.IK_SPAWN)).to(dev)
        self._hold_row = self._ik_row if self._hold_row is None else (self._hold_row | self._ik_row)
        self.spawn_hold = int(hold)
        self._hold_checks = torch.zeros(K, dtype=torch.long, device=dev)   # hold windows that ended, per env ...
        self._hold_bad = torch.zeros(K, dtype=torch.long, device=dev)      # ... and how many ended in bad settle health
        # whether THIS episode holds (every IK row; with reset_noise also the flat / stair rows that drew a hold)
        self._hold_ep = self.env_state((), init=False, dtype=torch.bool)
        # per family: resets, held resets, hold windows checked, bad, falls within EARLY_TICKS (held, not held)
        self._spf = torch.zeros(len(sc.FAMILIES), 6, dtype=torch.float64, device=dev)

    def _noisy_spawn(self, idx, q0):
        """reset_noise for the rows `idx`: joint preset + N(0, q_std) clipped to the limits, the base lift that keeps the
        lowest noised foot where the clean one was plus U(0, drop_max), base velocity and yaw rate. The body-frame foot
        heights come from spot_slopes.leg_fk (checked against PhysX to 0.03 mm) and ignore the body tilt of an IK spawn
        (<= 12.5 deg, a 2% error on a few cm). -> (q0 add order, lift [n], linvel [n,3], angvel [n,3])"""
        qs, dmax, vmax = self._reset_noise
        n, dev, g = idx.numel(), self.device, self._reset_g
        qi = q0[:, self.i2a]
        qn = torch.minimum(torch.maximum(qi + torch.randn(n, 12, device=dev, generator=g) * qs, self._q_lo), self._q_hi)
        tip_z = lambda q: ss.leg_fk(q.view(n, 3, 4).transpose(1, 2))[..., 2]
        lift = ((tip_z(qi) - tip_z(qn)).clamp_min(0.0).max(dim=1).values
                + torch.rand(n, device=dev, generator=g) * dmax)
        lin = torch.zeros(n, 3, device=dev)
        lin[:, :2] = (torch.rand(n, 2, device=dev, generator=g) * 2.0 - 1.0) * vmax
        ang = torch.zeros(n, 3, device=dev)
        ang[:, 2] = (torch.rand(n, device=dev, generator=g) * 2.0 - 1.0) * SWITCH_YAW_RATE
        return qn[:, self.a2i].contiguous(), lift, lin, ang

    def spawn_families(self):
        """Per family (one host read): IK failures (hills / cross / rough), hold windows and their settle-health failures,
        and falls within EARLY_TICKS of the spawn split by whether the episode held. Cumulative. A family is flagged when
        the IK or the hold failure share exceeds spot_slopes.SPAWN_FAIL_MAX."""
        if self._spf is None:
            return {}
        s = self._spf.cpu().numpy()
        fam = self._fam_k.cpu().numpy()
        res, unr = self._spawn_resets.cpu().numpy(), self._spawn_bad.cpu().numpy()
        div = lambda a, b: (a / b) if b else None
        out = {}
        for c, name in enumerate(sc.FAMILIES):
            if s[c, 0] == 0:
                continue
            m = fam == c
            n_res, held = int(s[c, 0]), int(s[c, 1])
            ik_res, ik_bad = int(res[m].sum()), int(unr[m].sum())
            out[name] = {"resets": n_res, "held": held, "hold_checked": int(s[c, 2]), "hold_bad": int(s[c, 3]),
                         "hold_fail_frac": div(s[c, 3], s[c, 2]), "ik_resets": ik_res, "ik_bad": ik_bad,
                         "early_falls_held": int(s[c, 4]), "early_falls_unheld": int(s[c, 5]),
                         "early_fall_frac_held": div(s[c, 4], held), "early_fall_frac_unheld": div(s[c, 5], n_res - held)}
            out[name]["flagged"] = bool((div(s[c, 3], s[c, 2]) or 0.0) > ss.SPAWN_FAIL_MAX
                                        or (div(ik_bad, ik_res) or 0.0) > ss.SPAWN_FAIL_MAX)
        return out

    def _course_spawn(self, idx, sx, pose, q0):
        """The hills / cross / rough rows of a reset: spot_slopes' IK stance at the spawn x (every foot 5 mm above the
        terrain under it, the body tilted by half the plane fitted under the footprint), the action that holds those
        joints remembered for the settle and the hold window, and IK / foot-gap failures counted. A partial reset has no
        settle, so the policy is also handed the held joints as its last action, which is what a full reset's settle
        leaves behind. Writes pose in place; returns the joint preset (add order). Other rows keep their values."""
        ik = self._ik_row[idx]
        sp = ss.spawn_pose(self.rays.heights, sx, self.lane_y[idx])      # fitted plane: rough ground has no closed form
        pose.copy_(torch.where(ik[:, None], torch.cat([sp.quat, sp.pos], dim=1).to(pose.dtype), pose))
        q0 = torch.where(ik[:, None], sp.joint_q[:, self.a2i].to(q0.dtype), q0)
        hold = torch.where(ik[:, None], ss.settle_action(sp.joint_q).to(self._settle_act.dtype), self._settle_act[idx])
        self._settle_act[idx] = hold
        self._spawn_nz[idx] = torch.where(ik, sp.normal[:, 2].to(self._spawn_nz.dtype), self._spawn_nz[idx])
        bad = ~sp.reachable.all(dim=1) | (sp.gap.max(dim=1).values > ss.CONTACT_TOL)
        self._spawn_resets[idx] += ik.long()
        self._spawn_bad[idx] += (ik & bad).long()
        self.last_act[idx] = torch.where(ik[:, None], hold, self.last_act[idx])
        self.prev_act[idx] = torch.where(ik[:, None], hold, self.prev_act[idx])
        return q0

    def _hold_check(self):
        """settle_health on the rows whose hold window ended this tick (every episode, full or partial reset), normals by
        finite differences of the terrain. Counted per env (episode_stats()) and per family (spawn_families())."""
        due = self._hold_ep & (self.steps == self.spawn_hold)
        tips = self._slope_fk.tips(self.sim.link_pose)
        ok, _, _ = ss.settle_health(self.up, tips, self._terrain_h, self._spawn_nz)
        self._hold_checks += due.long()
        self._hold_bad += (due & ~ok).long()
        self._spf[:, 2:4].index_add_(0, self._fam_k, torch.stack([due, due & ~ok], dim=1).double())

    def _settle_check(self):
        """spot_slopes.settle_health at the end of a full reset's settle: up_z >= 0.98 cos(slope fitted under the
        spawn) and all four feet within CONTACT_TOL of the plane along its normal. Latched per env (the next full
        reset overwrites it; a partial reset has no settle to check), read per block in episode_stats(). Course rows
        take their normals from finite differences of the raycast (spot_slopes.settle_health's default)."""
        tips = self._slope_fk.tips(self.sim.link_pose)
        if self._slope_lanes is not None:
            nz = self._slope_lanes.normals(tips[..., 0], tips[..., 1])[..., 2]
            ok, _, gap = ss.settle_health(self.up, tips, self.rays.heights, self._spawn_nz, normal_z=nz)
        if self._ik_row is not None:
            ok_c, _, gap_c = ss.settle_health(self.up, tips, self._terrain_h, self._spawn_nz)
            if self._slope_lanes is None:
                ok, gap = ok_c, gap_c
            else:
                ok = torch.where(self._ik_row, ok_c, ok)
                gap = torch.where(self._ik_row[:, None], gap_c, gap)
        self._settle_checked.copy_(self._hold_row)
        self._settle_bad.copy_(self._hold_row & ~ok)
        self._settle_uprel.copy_(self.up / self._spawn_nz)
        self._settle_gap.copy_(gap.max(dim=1).values)

    def _spawn_counts(self):
        """Per block (one host read): lanes that are slope lanes, the settle health of the last full reset, and the
        spawn geometry over every reset. A block is flagged when either failure share exceeds
        spot_slopes.SPAWN_FAIL_MAX, or when it has slope lanes and no settle was checked."""
        host = lambda t: t.cpu().numpy()
        bid, row, chk, bad = host(self.block_id), host(self._hold_row), host(self._settle_checked), host(self._settle_bad)
        res, unr, upr, gap = host(self._spawn_resets), host(self._spawn_bad), host(self._settle_uprel), host(self._settle_gap)
        hck = hbd = None
        if self._ik_row is not None:
            hck, hbd = host(self._hold_checks), host(self._hold_bad)
        out = []
        for b in range(self.n_blocks):
            m = (bid == b) & row
            mc = m & chk
            n_chk, n_bad, n_res, n_unr = int(mc.sum()), int((m & bad).sum()), int(res[m].sum()), int(unr[m].sum())
            f_set = (n_bad / n_chk) if n_chk else None
            f_unr = (n_unr / n_res) if n_res else None
            out.append({"lanes": int(m.sum()), "settle_checked": n_chk, "settle_bad": n_bad, "settle_fail_frac": f_set,
                        "resets": n_res, "unreachable": n_unr, "unreachable_frac": f_unr,
                        "min_up_over_cos": float(upr[mc].min()) if n_chk else None,
                        "max_foot_gap_m": float(gap[mc].max()) if n_chk else None,
                        "flagged": bool((m.any() and not n_chk) or (f_set or 0.0) > ss.SPAWN_FAIL_MAX
                                        or (f_unr or 0.0) > ss.SPAWN_FAIL_MAX)})
            if hck is not None:            # course rows: the hold window at the start of every episode
                n_h, n_hb = int(hck[m].sum()), int(hbd[m].sum())
                out[-1].update({"hold_ticks": self.spawn_hold, "hold_checked": n_h, "hold_bad": n_hb,
                                "hold_fail_frac": (n_hb / n_h) if n_h else None})
                out[-1]["flagged"] = bool(out[-1]["flagged"] or (n_h and n_hb / n_h > ss.SPAWN_FAIL_MAX))
        return out

    # ---- the task ---------------------------------------------------------------
    def on_reset(self, idx):
        n = idx.numel()
        dev = self.device
        pose = self.base_pose[idx].clone()
        sx = self._spawn_x(idx)
        sz = self._foot_clear_z(idx, sx) + SPAWN_Z + 0.03                 # drop-settle clearance
        pose[:, 4] = sx; pose[:, 6] = sz
        q0 = self.stand_q_add[idx]
        if self._slope_lanes is not None:                                  # slope rows: the IK stance on the plane
            sx, q0 = self._slope_spawn(idx, sx, pose, q0)
        if self._course is not None:                                       # hills / cross / rough rows: the same stance
            q0 = self._course_spawn(idx, sx, pose, q0)
        lin = ang = None
        if self._reset_noise is not None:                                  # messy resets (the hold target stays clean)
            q0, lift, lin, ang = self._noisy_spawn(idx, q0)
            pose[:, 6] += lift
        if self._hold_ep is not None:
            ik = self._ik_row[idx]
            hold = ik
            if self._reset_noise is not None:
                u = torch.rand(n, device=dev, generator=self._reset_g)
                hold = ik | (u >= self.reset_nohold_frac)
            self._hold_ep[idx] = hold
            self._spf[:, 0:2].index_add_(0, self._fam_k[idx], torch.stack([torch.ones_like(hold), hold], dim=1).double())
        self.sim.set_root_state(idx, pose, lin, ang)
        self.sim.set_joint_state(idx, q0, torch.zeros(n, self.sim.dof, device=dev))
        if self.percept is not None:
            self.percept.forget(idx)         # a teleport invalidates every remembered cell
        self.phi[idx] = reset_phi(n, dev)    # randomise phase (decorrelate batch; don't advance during settle)
        if self.cadence_jitter > 0.0:
            lo, hi = 1.0 - self.cadence_jitter, 1.0 + self.cadence_jitter
            self.period[idx] = GAIT_PERIOD * (torch.rand(n, device=dev) * (hi - lo) + lo)
        self.ep_start_x[idx] = sx
        self._resample_cmd(idx)
        # stair lanes: force a forward command at spawn so the robot ATTEMPTS the tent (level evaluation)
        st = idx[self._terrain_lane[idx]]
        if st.numel() > 0:
            self.cmd[st, 0] = torch.empty(st.numel(), device=dev).uniform_(0.5, VX_HI)
            self.cmd[st, 1:] = 0.0
        if self._hold is not None:           # a held command overrides the forced one too
            self.cmd[idx] = torch.where(self._hold_on[idx, None], self._hold[idx], self.cmd[idx])
        if self.push_mode == "once_on_tent":
            # the trigger: uniform over this episode's tent (stair lanes), or over [spawn+1, spawn+6]
            u = torch.rand(n, device=dev, generator=self._push_g)
            tent0 = sx + (FLAT_APPROACH - SPAWN_OFF)
            self.push_trig_x[idx] = torch.where(self.is_stairs[idx], tent0 + u * TENT_LEN, sx + 1.0 + 5.0 * u)
        if self.push_mode == "interval":
            lo, hi = self.push_interval
            u = torch.rand(n, device=dev, generator=self._push_g)
            self._push_cd[idx] = ((lo + (hi - lo) * u) / DT).long()
        if self.jump is not None:
            lo, hi = self.jump
            u = torch.rand(n, device=dev)
            self.jump_cd[idx] = ((lo + (hi - lo) * u) / DT).long() + JUMP_MIN_STEPS

    def act(self, a):
        # FULL policy action (not a residual): isaac -> add-order drive targets. The settle loop
        # feeds a=0, which lands exactly on the default stand targets.
        if self.qd_max is not None and not self.settling:
            # the target rate limit, in action units (a hold below still sets its joints outright)
            da = self.qd_max * DT / ACTION_SCALE
            a = self.last_act + (a - self.last_act).clamp(-da, da)
        if self._hold_row is not None:
            if self.settling:
                # slope rows hold their spawn joints through the settle (spot_slopes INTEGRATION 5); every other row
                # keeps a = 0. last_act then says what the drives hold, which is what the policy is handed.
                a = torch.where(self._hold_row[:, None], self._settle_act, a)
            elif self._hold_ep is not None and self.spawn_hold > 0:
                # course IK rows hold them for the first spawn_hold ticks of every episode, partial resets included
                # (they have no settle); with reset_noise so do the flat / stair rows that drew a hold (their
                # _settle_act is 0, the default stance). The policy's action for those rows and ticks is not the one
                # executed: ~1.3% of an episode's samples, the price of not stepping part of a batch.
                a = torch.where((self._hold_ep & (self.steps < self.spawn_hold))[:, None], self._settle_act, a)
        self.prev_act.copy_(self.last_act)
        self.last_act.copy_(a)
        return (self.default_q + ACTION_SCALE * a)[:, self.a2i]

    def on_settled(self):
        self.up = up_z(self.sim.root_quat)
        if self._hold_row is not None:
            self._settle_check()

    def on_step(self, s):
        if self.blowup_guard:
            rp, rv, rw = self.sim.root_position, self.sim.root_linvel, self.sim.root_angvel
            jp, jv = self.sim.joint_pos, self.sim.joint_vel
            blow = (~torch.isfinite(rp).all(dim=1) | ~torch.isfinite(rv).all(dim=1) | ~torch.isfinite(rw).all(dim=1)
                    | ~torch.isfinite(jp).all(dim=1) | ~torch.isfinite(jv).all(dim=1) | (rv.norm(dim=1) > BLOWUP_V)
                    | (rw.norm(dim=1) > BLOWUP_W) | (jv.abs().amax(dim=1) > BLOWUP_JV))
            self._blowup.copy_(blow)
            self._blow_n += blow.sum().double()
        if self.stand_mode:
            # a zero command freezes the clock where it is: the command in force over the step just simulated
            stand = (self.cmd == 0.0).all(dim=1)
            self.phi.copy_(torch.where(stand, self.phi, advance(self.phi, period=self.period)))
        else:
            self.phi.copy_(advance(self.phi, period=self.period))            # clock after physics, before next obs
        if self._ep_cmd is not None:
            self._ep_cmd += self.cmd[:, 0].clamp_min(0.0) * DT
        self.cmd_timer -= 1
        self._resample_due()
        if self._hold is not None:
            self.cmd.copy_(torch.where(self._hold_on[:, None], self._hold, self.cmd))

        q = self.sim.root_quat
        self.up = up_z(q)
        x, y, zz = self.sim.root_position[:, 0], self.sim.root_position[:, 1], self.sim.root_position[:, 2]
        self._roll = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        self._ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
        self._lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
        h_here = self._terrain_h(x, y)
        self._base_above = zz - h_here
        cyaw, syaw = heading_cossin(q)
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)
        ahead = self._terrain_h(px, py) - h_here[:, None]
        change = ahead.abs().max(dim=1).values
        self._w_imit = (1.0 - change / 0.10).clamp(0.0, 1.0)
        if self.cadence_jitter > 0.0:
            # The anchor's teacher is a FIXED-cadence walker, so it only has an opinion worth
            # copying about envs running near that cadence. Anchoring a jittered env pulls it
            # toward a gait that does not fit its own clock — an incoherent target, and exactly
            # the pull that would keep the cadence pinned however wide the jitter.
            off = (self.period - GAIT_PERIOD).abs() / (GAIT_PERIOD * self.cadence_jitter)
            self._w_imit = self._w_imit * (1.0 - off.clamp(0.0, 1.0))
        if self.stand_mode:
            # the teacher trots in place at a zero command, so it has no opinion worth copying about a stand
            self._w_imit = self._w_imit * (self.cmd != 0.0).any(dim=1).float()
        self.ep_max_climb.copy_(torch.maximum(self.ep_max_climb, (x - self.ep_start_x).clamp_min(0.0)))
        # Latch how far off its lane centre the robot was the FIRST time it got past the tent, which
        # is where the in-lane half of a clear is read (see IN_LANE). A where, so the step stays
        # graph-capturable and costs no host sync.
        clear = self.clear_dist if self._clear_k is None else self._clear_k
        self.clear_dy.copy_(torch.where((self.ep_max_climb > clear) & (self.clear_dy < 0.0),
                                        (y - self.lane_y).abs(), self.clear_dy))
        if self._feet_on:
            self._instrument_step(x, y)
        if self.jump is not None:
            self._jump_step()
        if self._hold_ep is not None and self.spawn_hold > 0:
            self._hold_check()

    def _jump_step(self):
        """Once per control tick after the instrument (feet_clear is this tick's): the flight state machine and the reward
        pieces for the envs inside a window, the window's close and its counters, then new triggers. The policy sees
        (1, jump_t / JUMP_WINDOW) from the trigger tick on (spot_jump_env's contract, minus the distance command)."""
        K, dev = self.K, self.device
        air = (self.feet_clear > JUMP_AIR).all(dim=1)
        vz = torch.nan_to_num(self.sim.root_linvel[:, 2], nan=0.0)
        ba = self._base_above
        win = self.jump_t >= 0
        st = self.jstate
        active = win & (st <= 1)
        new_z = torch.maximum(self.jzmax, ba)
        self._j_rise.copy_(torch.where(active, new_z - self.jzmax, 0.0))
        self.jzmax.copy_(torch.where(active, new_z, self.jzmax))
        new_v = torch.maximum(self.jvzmax, vz.clamp(0.0, JUMP_VZ_CAP))
        self._j_vz.copy_(torch.where(active, new_v - self.jvzmax, 0.0))
        self.jvzmax.copy_(torch.where(active, new_v, self.jvzmax))
        takeoff = win & (st == 0) & air & (vz > JUMP_TAKEOFF_VZ)
        fly_air = win & ((st == 1) | takeoff) & air
        self._j_fly.copy_(fly_air)
        self.jclear.copy_(torch.where(fly_air, torch.maximum(self.jclear, self.feet_clear.min(dim=1).values), self.jclear))
        touch = win & (st == 1) & ~air
        counted = touch & (self.jflight >= JUMP_MIN_FLIGHT)
        short = touch & ~counted
        self._j_land.copy_((counted & (self.up > JUMP_LAND_UP)).float() * J_LAND)
        self.jumped.copy_(self.jumped | counted)
        self.jlanded.copy_(self.jlanded | (counted & (self.up > JUMP_LAND_UP)))
        self.jflight.copy_(torch.where(fly_air, self.jflight + 1, torch.where(short, 0, self.jflight)))
        self.jstate.copy_(torch.where(counted, 2, torch.where(takeoff, 1, torch.where(short, 0, st))))
        # uncommanded flights, a metric: a run of JUMP_MIN_FLIGHT airborne ticks outside a window, counted once
        self._uair.copy_(torch.where(air & ~win, self._uair + 1, 0))
        self._juncmd[0] += (self._uair == JUMP_MIN_FLIGHT).sum().double()
        self._juncmd[1] += (~win).sum().double()
        # advance the windows; one closes JUMP_WINDOW ticks after its trigger
        t = torch.where(win, self.jump_t + 1, self.jump_t)
        closing = win & (t >= JUMP_WINDOW)
        self._j_miss.copy_(closing & ~self.jumped)
        self._jump_count(closing, torch.zeros_like(closing))
        t = torch.where(closing, -1, t)
        # triggers: the countdown ran out, no window open, past the episode's first second
        cd = self.jump_cd - 1
        trig = self._jump_lane & (t < 0) & (cd <= 0) & (self.steps > JUMP_MIN_STEPS)
        lo, hi = self.jump
        self.jump_cd.copy_(torch.where(trig, ((lo + (hi - lo) * torch.rand(K, device=dev)) / DT).long() + JUMP_WINDOW, cd))
        self.jump_t.copy_(torch.where(trig, 0, t))
        self.jstate.copy_(torch.where(trig, 0, self.jstate))
        self.jflight.copy_(torch.where(trig, 0, self.jflight))
        self.jz0.copy_(torch.where(trig, ba, self.jz0))
        self.jzmax.copy_(torch.where(trig, ba, self.jzmax))
        self.jvzmax.copy_(torch.where(trig, 0.0, self.jvzmax))
        self.jclear.copy_(torch.where(trig, 0.0, self.jclear))
        self.jumped.copy_(self.jumped & ~trig)
        self.jlanded.copy_(self.jlanded & ~trig)
        bucket = torch.where((self.cmd == 0.0).all(dim=1), 0, torch.where(self.cmd[:, 0].abs() < 1.0, 1, 2))
        self.jbucket.copy_(torch.where(trig, bucket, self.jbucket))

    def _jump_count(self, mask, fell):
        """Close the windows in `mask` into the per-bucket counters (fell: those that ended in a fall)."""
        m = mask.double()
        j = (mask & self.jumped).double()
        cols = torch.stack([m, j, (mask & self.jlanded).double(), (mask & ~self.jumped).double(), fell.double(),
                            j * self.jflight.double() * DT, m * (self.jzmax - self.jz0).clamp_min(0.0).double(),
                            j * self.jclear.double(), m * self.jvzmax.double()], dim=1)
        self._jst.index_add_(0, self.jbucket, cols)

    def jump_stats(self, reset=True):
        """Per command bucket at the trigger (and all): windows, shares jumped / landed upright / missed / fell, mean
        flight (of the jumps), rise, clearance (of the jumps) and peak upward speed; uncommanded flights per env-minute."""
        s, u = self._jst.cpu().numpy(), self._juncmd.cpu().numpy()
        div = lambda a, b: (float(a) / float(b)) if b else None

        def row(v):
            return {"windows": int(v[0]), "jumped": div(v[1], v[0]), "landed_up": div(v[2], v[0]), "missed": div(v[3], v[0]),
                    "fell": div(v[4], v[0]), "flight_s": div(v[5], v[1]), "rise_m": div(v[6], v[0]),
                    "clearance_m": div(v[7], v[1]), "vz_max": div(v[8], v[0])}

        out = {name: row(s[i]) for i, name in enumerate(JUMP_BUCKETS)}
        out["all"] = row(s.sum(axis=0))
        out["uncommanded_flights_per_env_min"] = div(u[0], u[1] * DT / 60.0)
        if reset:
            self._jst.zero_(); self._juncmd.zero_()
        return out

    def jump_config(self):
        """What a checkpoint's meta records about the jump, the observation and the plant knobs (all None / False off)."""
        return {"obs_dim": self.obs_dim, "jump": list(self.jump) if self.jump else None, "jump_obs": self.jump_obs,
                "qd_max": self.qd_max, "self_collision": self.self_collision, "blowup_guard": self.blowup_guard,
                **({"w_jump_imit": self.w_jump_imit, "jump_window": JUMP_WINDOW, "jump_min_steps": JUMP_MIN_STEPS,
                    "jump_air": JUMP_AIR, "jump_takeoff_vz": JUMP_TAKEOFF_VZ, "jump_min_flight": JUMP_MIN_FLIGHT,
                    "jump_land_up": JUMP_LAND_UP, "jump_vz_cap": JUMP_VZ_CAP,
                    "jump_weights": {"rise": J_RISE, "flight": J_FLIGHT, "land": J_LAND, "miss": J_MISS, "vz": J_VZ},
                    "jump_ref_keys": [[k, list(v)] for k, v in J_REF_KEYS], "jump_imit_sigma2": J_IMIT_SIGMA2,
                    "jump_off_terms": list(JUMP_OFF_TERMS)} if self.jump is not None else {})}

    def terminated(self, s):
        tilt = self.up < 0.35
        self._term_tilt.copy_(tilt)          # the cause, for the counters (written through, graph-safe)
        if self.termination == "honest":
            out = self._honest_now           # set this tick by _instrument_step
        else:
            out = tilt | (self._base_above < 0.18)
        return (out | self._blowup) if self.blowup_guard else out

    def blowup_count(self, reset=True):
        """blowup_guard: robot-ticks terminated as blow-ups since the last call."""
        n = int(self._blow_n.item())
        if reset:
            self._blow_n.zero_()
        return n

    # ---- the instrument (opt-in; see __init__ and spot_feet) ----------------------------------------------
    def _init_instrument(self):
        """Buffers for the honest termination and the instrument, allocated only when one is asked for, so
        the default world registers no extra state and no extra ops."""
        K, dev = self.K, self.device
        L = sf.LAYOUT
        self._fk = sf.FootKin(dev)                                         # GPU link order, measured
        self._patch_dx = torch.tensor(sf.PATCH_D, device=dev)
        self._patch_dy = torch.tensor(sf.PATCH_E, device=dev)
        self._patch_centre = torch.tensor([True, False, False, False, False], device=dev)
        self._slope_dx = torch.tensor([sf.SLOPE_DX, sf.SLOPE_DX, -sf.SLOPE_DX, -sf.SLOPE_DX], device=dev)
        self._slope_dy = torch.tensor([sf.SLOPE_DY, -sf.SLOPE_DY, sf.SLOPE_DY, -sf.SLOPE_DY], device=dev)
        self._miss_level = (self.rays.miss + 0.5) if self.rays is not None else float("-inf")
        self._uprel_edges = torch.tensor(sf.UPREL_EDGES, device=dev)
        self._nose_edges = torch.tensor(sf.NOSE_EDGES, device=dev)
        # isaac joint order is type-grouped (hx x4, hy x4, kn x4), which is also the gate grouping
        self._kp = torch.tensor([STIFF_GAINS[g][0] for g in sf.GROUPS for _ in range(4)], device=dev)
        self._kd = torch.tensor([STIFF_GAINS[g][1] for g in sf.GROUPS for _ in range(4)], device=dev)
        self._off_tau = torch.tensor([L[f"tau_hist_{g}"][0] for g in sf.GROUPS for _ in range(4)], device=dev)
        self._off_qd = torch.tensor([L[f"qd_hist_{g}"][0] for g in sf.GROUPS for _ in range(4)], device=dev)
        self._sc_idx = torch.tensor([L[n][0] for n in ("ticks", "path_m", "up_sum", "cos_slope_sum", "stance_ticks",
                                                       "slip_m", "miss_body", "miss_feet", "cf_mismatch",
                                                       "recovered", "recovery_ticks", "knee_ticks")], device=dev)
        self._pf_idx = torch.tensor([L[n][0] for n in ("td", "stall", "nose_n", "nose_lt02", "nose_sum", "perched")],
                                    device=dev)
        assert L["tread_td_desc"][0] == L["tread_td_asc"][0] + sf.TREAD_W
        a0 = L["tread_td_asc"][0]                                          # offsets inside one direction's block
        self._tro = {n: L[f"{n}_asc"][0] - a0 for n in ("tread_td", "edge_lt05", "drop_lt028", "edge_sum", "stall",
                                                         "edge_hist", "across_hist", "edge_lt05_stall",
                                                         "drop_lt028_stall", "edge_sum_stall", "edge_hist_stall",
                                                         "perched")}
        self._ones_k = torch.ones(K, dtype=torch.float64, device=dev)
        self._ones_k12 = torch.ones(K, 12, dtype=torch.float64, device=dev)
        # per-episode state, re-initialised by the base on every reset
        self._stance = self.env_state((4,), init=False, dtype=torch.bool)
        self._tilt_ticks = self.env_state((), init=0, dtype=torch.long)
        self._lift_x = self.env_state((4,), init=-1.0e3)
        self._nose_min = self.env_state((4,), init=9.0)                   # 9 = no nosing crossed this swing
        self._up_pre = self.env_state((), init=1.0)
        self._rec_run = self.env_state((), init=0, dtype=torch.long)
        self._rec_ticks = self.env_state((), init=-1, dtype=torch.long)
        self._first_legacy = self.env_state((), init=-1, dtype=torch.long)
        self._first_honest = self.env_state((), init=-1, dtype=torch.long)
        self._honest_cause = self.env_state((), init=0, dtype=torch.long)  # at the first honest fire: 1 tilt | 2 base | 4 upper leg
        self._knee_fired = self.env_state((), init=False, dtype=torch.bool)  # a knee end came within BODY_CONTACT
        self._iacc = self.env_state((sf.WIDTH - len(sf.EPISODE_COLS),), init=0.0, dtype=torch.float64)
        # the previous tick (read only when steps >= 2, i.e. from inside the same episode)
        self._prev_xy = torch.zeros(K, 2, device=dev)
        self._tip_prev = torch.zeros(K, 4, 3, device=dev)
        self._up_prev = torch.ones(K, device=dev)
        # this tick's termination tests: terminated() returns one, _count reads all of them
        self._tilt_sus = torch.zeros(K, dtype=torch.bool, device=dev)
        self._hit_base = torch.zeros_like(self._tilt_sus)
        self._hit_uleg = torch.zeros_like(self._tilt_sus)
        self._legacy_now = torch.zeros_like(self._tilt_sus)
        self._honest_now = torch.zeros_like(self._tilt_sus)
        self._knee_now = torch.zeros_like(self._tilt_sus)
        # this tick's touchdowns. NOT env_state: the partial reset inside step() must not wipe the terminal
        # tick's rows before a recorder reads them after step() returns
        self.td_event = torch.zeros(K, 4, dtype=torch.bool, device=dev)
        self.td_rec = torch.zeros(K, 4, len(sf.TD_FIELDS), device=dev)
        self.td_tick = torch.zeros(K, dtype=torch.long, device=dev)
        self.td_episode = torch.zeros(K, dtype=torch.long, device=dev)
        self.feet_clear = torch.zeros(K, 4, device=dev)
        self._istats = torch.zeros(self.n_blocks * 2 * self._ep_bins, sf.WIDTH, dtype=torch.float64, device=dev)

    def _tent_t(self, tx, ty):
        """World points [K, n] -> (t from the tent start, riser, on a stair lane and inside it). The band is
        the lane's own in a single_band world, else picked from x exactly as _terrain_h picks it."""
        if self._lane_band is not None:
            band = self._lane_band[:, None].expand_as(tx)
        else:
            band = torch.clamp(torch.floor(tx / self.band_len).long(), 0, self.n_levels - 1)
        t = tx - band.to(tx.dtype) * self.band_len - self.flat_approach
        on = self.is_stairs[:, None] & ((ty - self.lane_y[:, None]).abs() < HALF_W_STEPS)
        return t, self.risers[band], on

    def _instrument_step(self, x, y):
        """Once per control tick, after the state refresh: both termination tests, touchdowns against the tread
        closed form, and the stability and validity accumulators. Tensor ops on buffers written through — no
        boolean indexing, no host read — so it replays from the step graph like the rest of on_step."""
        K, L, f64 = self.K, sf.LAYOUT, torch.float64
        q, root, steps = self.sim.root_quat, self.sim.root_position, self.steps
        fk = self._fk
        valid = steps >= 2                                                 # a delta needs a tick of this episode
        tips = fk.tips(self.sim.link_pose)                                 # [K,4,3]
        body = fk.body_points(root, q, self.up, self.sim.link_pose)        # [K,nb,3]
        nb = fk.n_body
        cyaw, syaw = heading_cossin(q)
        # ONE terrain query: a 5-point patch under each foot, the body samples, the stance rectangle
        sx = x[:, None] + self._slope_dx * cyaw[:, None] - self._slope_dy * syaw[:, None]
        sy = y[:, None] + self._slope_dx * syaw[:, None] + self._slope_dy * cyaw[:, None]
        h = self._terrain_h(torch.cat([(tips[:, :, 0:1] + self._patch_dx).reshape(K, -1), body[:, :, 0], sx], dim=1),
                            torch.cat([(tips[:, :, 1:2] + self._patch_dy).reshape(K, -1), body[:, :, 1], sy], dim=1))
        miss = h <= self._miss_level
        h = torch.where(miss, 0.0, h)            # a raycast miss is the lane ground, never "far below"; counted
        hf = h[:, :20].view(K, 4, 5)
        # the highest ground in the foot's patch that is not above the tip centre: a foot resting on a nosing
        # with its centre just past the edge reads ~0; a foot swinging up a riser face does not read as down
        hfoot = torch.where(self._patch_centre | (hf <= tips[:, :, 2:3]), hf, -1.0e3).max(dim=2).values
        clear = tips[:, :, 2] - sf.FOOT_R - hfoot
        bclr = body[:, :, 2] - fk.body_r - h[:, 20:20 + nb]
        if self._slope_lanes is not None:
            # slope lanes: the foot and every body sample measured perpendicular to the local plane (closed-form
            # normal: a sphere of radius r resting on a plane reads 0 at any angle), the foot off its patch centre
            nrm = self._slope_lanes.normals(torch.cat([tips[:, :, 0], body[:, :, 0], x[:, None]], dim=1),
                                            torch.cat([tips[:, :, 1], body[:, :, 1], y[:, None]], dim=1))
            sr = self._slope_row[:, None]
            clear = torch.where(sr, (tips[:, :, 2] - hf[:, :, 0]) * nrm[:, :4, 2] - sf.FOOT_R, clear)
            bclr = torch.where(sr, (body[:, :, 2] - h[:, 20:20 + nb]) * nrm[:, 4:4 + nb, 2] - fk.body_r, bclr)
        hit_base = bclr[:, :fk.n_base].min(dim=1).values < sf.BODY_CONTACT
        ul = bclr[:, fk.n_base:]
        hit_uleg = torch.where(fk.uleg_term, ul, 1.0e3).min(dim=1).values < sf.BODY_CONTACT
        knee = torch.where(fk.uleg_term, 1.0e3, ul).min(dim=1).values < sf.BODY_CONTACT    # recorded, not a fall
        hs = h[:, 20 + nb:]
        gxs = (hs[:, 0] + hs[:, 1] - hs[:, 2] - hs[:, 3]) / (4.0 * sf.SLOPE_DX)
        gys = (hs[:, 0] + hs[:, 2] - hs[:, 1] - hs[:, 3]) / (4.0 * sf.SLOPE_DY)
        cos_slope = torch.rsqrt(1.0 + gxs * gxs + gys * gys)

        # termination: both rules, whichever one terminates
        tilt = self.up < sf.TILT_UP
        self._tilt_ticks.copy_(torch.where(tilt, self._tilt_ticks + 1, 0))
        tilt_sus = self._tilt_ticks >= sf.TILT_TICKS
        honest = tilt_sus | hit_base | hit_uleg
        legacy = tilt | (self._base_above < 0.18)
        self._tilt_sus.copy_(tilt_sus); self._hit_base.copy_(hit_base); self._hit_uleg.copy_(hit_uleg)
        self._legacy_now.copy_(legacy); self._honest_now.copy_(honest)
        self._knee_now.copy_(knee); self._knee_fired.copy_(self._knee_fired | knee)
        self._first_legacy.copy_(torch.where((self._first_legacy < 0) & legacy, steps, self._first_legacy))
        first_h = (self._first_honest < 0) & honest
        self._first_honest.copy_(torch.where(first_h, steps, self._first_honest))
        self._honest_cause.copy_(torch.where(first_h, tilt_sus.long() + 2 * hit_base.long() + 4 * hit_uleg.long(),
                                             self._honest_cause))

        # stance with hysteresis, touchdowns, swings
        st0 = self._stance
        st1 = torch.where(st0, clear <= sf.TD_OFF, clear < sf.TD_ON)
        td = st1 & ~st0 & (steps > sf.TD_ARM_TICKS)[:, None]
        lift = st0 & ~st1
        self._lift_x.copy_(torch.where(lift, tips[:, :, 0], self._lift_x))
        t, riser, on = self._tent_t(tips[:, :, 0], tips[:, :, 1])
        t_prev = t - (tips[:, :, 0] - self._tip_prev[:, :, 0])
        crossed, frac, h_nose = sf.nosing_crossing(t_prev, t, riser, self.step_run, self.n_up, self.landing)
        z_c = self._tip_prev[:, :, 2] + frac * (tips[:, :, 2] - self._tip_prev[:, :, 2])
        # EVERY crossing counts, whatever the stance state: a toe that clips the nosing switches stance on (its +-r patch
        # point sees the tread above), so a swing-only rule drops exactly the scrapes. The minimum runs from one
        # touchdown to the next, stance included; below 0 is the chord between two ticks cutting the corner, a contact
        cross = crossed & on & valid[:, None]
        nose_min = torch.where(cross, torch.minimum(self._nose_min, (z_c - sf.FOOT_R - h_nose).clamp_min(0.0)),
                               self._nose_min)
        self._nose_min.copy_(torch.where(td, 9.0, nose_min))             # restarts once this touchdown has read it
        fr_c = sf.tread_frame(t, riser, self.step_run, self.n_up, self.landing)     # the surface under the tip centre
        # perched: the ground patch stands a riser above the surface under the centre, i.e. the foot is on the next
        # surface's nosing with its centre past that edge. Placed against the surface it STANDS on (the +-r patch point
        # that found it says which side), where across / d_edge / d_drop come out negative; never a stall below it
        perched = on & (hfoot > fr_c["height"] + sf.PERCH_RISE * riser)
        # 1 mm past the patch point: a ray at a nosing's exact float edge reads the tread above, the closed form the one below
        side = torch.where(hf[:, :, 1] >= hf[:, :, 2], sf.FOOT_R + 1.0e-3, -(sf.FOOT_R + 1.0e-3))
        fr = sf.tread_frame(t, riser, self.step_run, self.n_up, self.landing, at=torch.where(perched, t + side, t))
        kind = torch.where(on, fr["kind"], torch.where(self.is_stairs[:, None], sf.K_OFFLANE, sf.K_FLAT))
        tread = td & on & ((fr["kind"] == sf.K_ASCENT) | (fr["kind"] == sf.K_DESCENT))
        d_edge = torch.nan_to_num(fr["d_edge"], nan=0.0)
        d_drop = torch.nan_to_num(fr["d_drop"], nan=0.0)
        across = torch.nan_to_num(fr["across"], nan=0.0)
        if self._place_buf is not None:
            # the placement signal (reward term 'place' when w_place > 0): a tread touchdown pays its distance to the
            # nearer edge up to PLACE_SAT, one perched on a nosing (negative d_edge) costs PLACE_PERCH
            self._place_buf.copy_(((tread & ~perched).float() * (d_edge.clamp(0.0, PLACE_SAT) / PLACE_SAT)
                                   - (tread & perched).float() * PLACE_PERCH).sum(dim=1))
        stride = tips[:, :, 0] - self._lift_x
        d_rise = torch.where(self.cmd[:, 0:1] >= 0.0, fr["rise_fwd"], fr["rise_bwd"])
        stall = td & on & (d_rise < sf.STALL_DIST) & (stride.abs() < sf.STALL_STRIDE)
        has_nose = td & (nose_min < 8.0)
        nose = torch.where(has_nose, nose_min, 0.0)
        cfm = on & ~miss[:, :20].view(K, 4, 5)[:, :, 0] & ((fr_c["height"] - hf[:, :, 0]).abs() > 0.005)

        # stability
        dxy = (root[:, 0:2] - self._prev_xy).norm(dim=1) * valid
        if self._slope_lanes is not None:
            # slope lanes walk the surface: a horizontal step d on the local plane rises by -(n_xy . d) / n_z
            dv = root[:, 0:2] - self._prev_xy
            rise = -(dv * nrm[:, -1, 0:2]).sum(dim=1) / nrm[:, -1, 2]
            dxy = torch.where(self._slope_row, (dv.pow(2).sum(dim=1) + rise * rise).sqrt() * valid, dxy)
            lane = torch.round(y / SPACING).clamp(0.0, float(K - 1)).long()       # the lane under the base, by its y
            offsurf = self._slope_row & (self._surf_id[lane] != self._surf_id)
            self._iacc[:, self._offsurf_col].add_(offsurf.to(f64))              # a column view, written through
        # on the ground at both ends of the tick (sf.SLIP_CONTACT), not merely inside the stance hysteresis; feet_clear
        # still holds the previous tick's clearance here
        slip_ok = st0 & st1 & (clear < sf.SLIP_CONTACT) & (self.feet_clear < sf.SLIP_CONTACT) & valid[:, None]
        slip = ((tips[:, :, 0:2] - self._tip_prev[:, :, 0:2]).norm(dim=2) * slip_ok).sum(dim=1)
        pushed = self.push_step >= 0
        since = steps - self.push_step
        self._up_pre.copy_(torch.where(pushed & (since == 1), self._up_prev, self._up_pre))
        verr = torch.hypot(self.cmd[:, 0] - self._lin_b[:, 0], self.cmd[:, 1] - self._lin_b[:, 1])
        ok = pushed & (since >= 1) & ((self.up - self._up_pre).abs() < sf.REC_UP) & (verr < sf.REC_V)
        self._rec_run.copy_(torch.where(ok, self._rec_run + 1, 0))
        rec_now = (self._rec_ticks < 0) & (self._rec_run >= sf.REC_HOLD)
        self._rec_ticks.copy_(torch.where(rec_now, since - (sf.REC_HOLD - 1), self._rec_ticks))

        # validity gates: |commanded torque| = |kp (q_tgt - q) - kd qd| and |qd|, per joint, isaac order
        qi, qdi = self.sim.joint_pos[:, self.i2a], self.sim.joint_vel[:, self.i2a]
        tau = (self._kp * (self.default_q + ACTION_SCALE * self.last_act - qi) - self._kd * qdi).abs()
        tau_i = self._off_tau + torch.clamp((tau / sf.TAU_W).long(), 0, sf.TAU_BINS - 1)
        qd_i = self._off_qd + torch.clamp((qdi.abs() / sf.QD_W).long(), 0, sf.QD_BINS - 1)

        # one scatter into the per-episode accumulator
        d = lambda v: v.to(f64)
        sc_val = torch.stack([self._ones_k, d(dxy), d(self.up), d(cos_slope), d(slip_ok.sum(dim=1)), d(slip),
                              d(miss[:, 20:20 + nb].sum(dim=1)), d(miss[:, :20].sum(dim=1)), d(cfm.sum(dim=1)),
                              d(rec_now), d(rec_now * self._rec_ticks), d(knee)], dim=1)
        dyn_idx = torch.stack([L["uprel_hist"][0] + torch.bucketize(self.up / cos_slope, self._uprel_edges, right=True),
                               L["recovery_hist"][0] + torch.clamp(self._rec_ticks // sf.REC_W, 0, sf.REC_BINS - 1)],
                              dim=1)
        dyn_val = torch.stack([self._ones_k, d(rec_now)], dim=1)
        pf_val = torch.stack([d(td), d(stall), d(has_nose), d(has_nose & (nose < sf.NOSE_SCRAPE)), d(nose),
                              d(td & perched)], dim=2)
        base = torch.where(fr["kind"] == sf.K_DESCENT, sf.TREAD_W, 0) + L["tread_td_asc"][0]
        e_bin = torch.clamp((d_edge / sf.EDGE_W).long(), 0, sf.EDGE_BINS - 1)
        a_bin = torch.clamp((across / sf.ACROSS_W).long(), 0, sf.ACROSS_BINS - 1)
        o, ts = self._tro, tread & stall
        near_e, near_d = d_edge < sf.EDGE_NEAR, d_drop < sf.DROP_NEAR
        pfd_idx = torch.stack([L["td_kind"][0] + kind,
                               L["nose_hist"][0] + torch.bucketize(nose_min, self._nose_edges, right=True),
                               base + o["tread_td"], base + o["edge_lt05"], base + o["drop_lt028"],
                               base + o["edge_sum"], base + o["stall"], base + o["edge_hist"] + e_bin,
                               base + o["across_hist"] + a_bin, base + o["edge_lt05_stall"],
                               base + o["drop_lt028_stall"], base + o["edge_sum_stall"],
                               base + o["edge_hist_stall"] + e_bin, base + o["perched"]], dim=2)
        pfd_val = torch.stack([d(td), d(has_nose), d(tread), d(tread & near_e), d(tread & near_d), d(tread * d_edge),
                               d(ts), d(tread), d(tread), d(ts & near_e), d(ts & near_d), d(ts * d_edge), d(ts),
                               d(tread & perched)], dim=2)
        self._iacc.scatter_add_(1, torch.cat([self._sc_idx.expand(K, -1), dyn_idx, tau_i, qd_i,
                                              self._pf_idx.expand(K, 4, -1).reshape(K, -1), pfd_idx.reshape(K, -1)], dim=1),
                                torch.cat([sc_val, dyn_val, self._ones_k12, self._ones_k12,
                                           pf_val.reshape(K, -1), pfd_val.reshape(K, -1)], dim=1))

        # this tick's touchdown rows (read by score_e0's recorder), then roll the previous-tick state
        f32, nan = torch.float32, float("nan")
        self.td_event.copy_(td)
        self.feet_clear.copy_(clear)
        self.td_tick.copy_(steps)
        self.td_episode.copy_(self.ep_index)
        self.td_rec.copy_(torch.stack([
            t, torch.where(on, fr["index"], -1.0), kind.to(f32), torch.where(on, fr["across"], nan),
            torch.where(on, fr["d_edge"], nan), torch.where(on, fr["d_drop"], nan),
            torch.where(on, fr["direction"], 0.0), riser * on, torch.where(has_nose, nose_min, nan),
            stall.to(f32), tips[:, :, 0], tips[:, :, 1] - self.lane_y[:, None], tips[:, :, 0] - root[:, 0:1],
            stride, self.cmd[:, 0:1].expand(K, 4), perched.to(f32)], dim=2))
        self._stance.copy_(st1)
        self._prev_xy.copy_(root[:, 0:2])
        self._tip_prev.copy_(tips)
        self._up_prev.copy_(self.up)

    def reward_terms(self, s, a):
        # Anchor: the 50-d clock base gait with its frozen RunningNorm (obs[:,:50] = proprio+clock).
        anchor_a = self.anchor_ac.act_mean(self.anchor_norm.norm(self._last_obs[:, :50]))
        imit = self._w_imit * (a - anchor_a).pow(2).mean(dim=1)
        arate = a - self.prev_act

        e_lin = (self.cmd[:, 0] - self._lin_b[:, 0]).pow(2) + (self.cmd[:, 1] - self._lin_b[:, 1]).pow(2)
        e_ang = (self.cmd[:, 2] - self._ang_b[:, 2]).pow(2)
        track_lin = torch.exp(-e_lin / SIG)
        track_ang = torch.exp(-e_ang / SIG)
        terms = {
            "track_lin": 3.0 * track_lin,
            "track_ang": 1.5 * track_ang,
            "alive": torch.full((self.K,), 0.05, device=self.device),
            "roll": -1.0 * self._roll.pow(2),                # ROLL only — climbing legitimately PITCHES
            "vz": -0.1 * self._lin_b[:, 2].pow(2),
            "angrate": -0.05 * (self._ang_b[:, 0].pow(2) + self._ang_b[:, 2].pow(2)),
            "scrape": -3.0 * torch.relu(0.30 - self._base_above),   # anti-scrape over the steps
            "arate": -0.001 * arate.pow(2).mean(dim=1),
            "imit": -self._imit_w * imit,
            "fell": -5.0 * s.terminated.float(),
        }
        if self.w_place > 0.0:
            terms["place"] = self.w_place * self._place_buf
        if self.stand_mode:
            terms["stand"] = -self.w_stand * (self.cmd == 0.0).all(dim=1).float() * s.joint_vel.pow(2).mean(dim=1)
        if self.jump is not None:
            inw = self.jump_t >= 0
            for name in JUMP_OFF_TERMS:
                if name in terms:
                    terms[name] = torch.where(inw, torch.zeros_like(terms[name]), terms[name])
            k = self.jump_t.clamp(0, JUMP_WINDOW - 1)
            err = (s.joint_pos - self.stand_q_add - self._jref[k]).pow(2).mean(dim=1)
            terms["jrise"] = J_RISE * self._j_rise
            terms["jvz"] = J_VZ * self._j_vz
            terms["jflight"] = J_FLIGHT * self._j_fly.float()
            terms["jland"] = self._j_land
            terms["jmiss"] = -J_MISS * self._j_miss.float()
            terms["jimit"] = self.w_jump_imit * inw.float() * torch.exp(-err / J_IMIT_SIGMA2)
            fell = s.terminated & inw
            self._jump_count(fell, fell)
        # Device-side, and read back only when a trainer asks (see the last_* properties). Masked
        # means rather than boolean indexing: a mask keeps the shapes static, which is what lets the
        # whole reward be replayed from a CUDA graph.
        trk = track_lin + track_ang
        self._st_fell.copy_(s.terminated.float().mean())
        self._st_track.copy_(trk.mean())
        flat = (~self._terrain_lane).float()
        self._st_flat.copy_((trk * flat).sum() / flat.sum().clamp_min(1.0))
        st = self.is_stairs.float()
        self._st_level.copy_((self.level.float() * st).sum() / st.sum().clamp_min(1.0))
        if self._place_buf is not None:
            self._st_place.copy_((self._place_buf * st).sum() / st.sum().clamp_min(1.0))
        if self.blowup_guard:
            # an exploding robot earns nothing, and no term of any robot is non-finite
            terms = {k: torch.where(self._blowup, torch.zeros_like(v), torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
                     for k, v in terms.items()}
        return terms

    def _course_done(self, idx):
        """The course curriculum for the envs ending now. Stair lanes exactly as on_done: promote on clearing the tent,
        demote only if the robot never got near it, a fall does neither. Hills / cross / rough: promote on forward
        progress past the band's feature (spot_course.FEATURE_DIST); progress is the episode's maximum, so a crossing
        always came before any fall and a fall after it does not block the promotion. Demote when the feature was not
        walked and the episode either ended in a fall or timed out under DEMOTE_PROGRESS of the commanded forward
        distance (a robot commanded slower than the feature needs is not demoted for being slow). Flat lanes keep
        level 0."""
        code, prog = self._cc[idx], self.ep_max_climb[idx]
        term = self.state.terminated[idx]
        stair = code == sc.STAIRS
        terr = (code != sc.FLAT) & ~stair
        crossed = prog > self._clear_k[idx]
        up = torch.where(stair, prog > self.clear_dist, terr & crossed)
        slow = prog < sc.DEMOTE_PROGRESS * self._ep_cmd[idx]
        dn = torch.where(stair, prog < self.reach_dist, terr & ~crossed & (term | slow))
        if not self.freeze_level:
            top = torch.where(stair, self.n_levels - 1, sc.N_LEVELS - 1)
            self.level[idx] = torch.minimum((self.level[idx] + up.long() - dn.long()).clamp_min(0), top)
        self._moves.index_add_(0, code, torch.stack([torch.ones_like(prog), up.float(), dn.float()], dim=1).double())
        if bool(stair.any()):
            self.last_clear = (prog[stair] > self.clear_dist).float().mean().item()
        stf = self.is_stairs.float()
        self._st_level.copy_((self.level.float() * stf).sum() / stf.sum().clamp_min(1.0))

    def course_levels(self):
        """Per course family (one host read): lanes, mean level, level histogram, and the episodes / promotions /
        demotions since the last call ({} without course lanes)."""
        if self._cc is None:
            return {}
        code, lvl = self._cc.cpu().numpy(), self.level.cpu().numpy()
        mv = self._moves.cpu().numpy()
        self._moves.zero_()
        out = {}
        for c, name in enumerate(sc.FAMILIES):
            m = code == c
            if c == sc.FLAT or not m.any():
                continue
            nl = self.n_levels if c == sc.STAIRS else sc.N_LEVELS
            out[name] = {"lanes": int(m.sum()), "level": float(lvl[m].mean()),
                         "ladder": list(self.riser_list) if c == sc.STAIRS else
                         (list(self._course.level_value[c]) if self._course is not None else None),
                         "hist": np.bincount(lvl[m], minlength=nl).tolist(),
                         "episodes": int(mv[c, 0]), "promoted": int(mv[c, 1]), "demoted": int(mv[c, 2])}
        return out

    def on_done(self, idx):
        if self._spf is not None:
            # spawn falls: a termination within EARLY_TICKS of the spawn, split by whether the episode held
            early = (self.state.terminated[idx] & (self.steps[idx] <= EARLY_TICKS))
            held = self._hold_ep[idx]
            self._spf[:, 4:6].index_add_(0, self._fam_k[idx], torch.stack([early & held, early & ~held], dim=1).double())
        if self._cc is not None:
            self._course_done(idx)
            self._count(idx)
            return
        # CURRICULUM update for STAIR envs ending now: promote on clearing the tent, demote only if the
        # robot never got near it. A fall does not demote.
        st = idx[self.is_stairs[idx]]
        if st.numel() > 0:
            # DISTANCE-based (not fall-gated): clearing the tent promotes even if it falls later;
            # only a robot that never reached the tent (fell early / stalled) demotes.
            prog = self.ep_max_climb[st]
            if not self.freeze_level:
                self.level[st] = torch.clamp(self.level[st]
                                             + (prog > self.clear_dist).long()
                                             - (prog < self.reach_dist).long(), 0, self.n_levels - 1)
            self.last_clear = (prog > self.clear_dist).float().mean().item()
            # on_done runs outside the step graph, but keep the same device-side accumulator the
            # reward writes so the two never disagree about what "level" means.
            stf = self.is_stairs.float()
            self._st_level.copy_((self.level.float() * stf).sum() / stf.sum().clamp_min(1.0))
        self._count(idx)

    def _count(self, idx):
        """Episode counters for the envs ending now (see STAT_KEYS). Device-side and host-free: one
        index_add_ of a [n, len(STAT_KEYS)] indicator matrix into the (block, lane type, episode bin)
        rows. Runs in on_done, i.e. before the reset wipes the episode's state."""
        term = self.state.terminated[idx]
        to = ~term
        tilt = self._term_tilt[idx]
        pos = self.sim.root_position[idx]
        prog = self.ep_max_climb[idx]
        crossed = prog > (self.clear_dist if self._clear_k is None else self._clear_k[idx])
        cleared = crossed & (self.clear_dy[idx] < IN_LANE)       # in lane AT THE CROSSING (see on_step)
        inlane_end = (pos[:, 1] - self.lane_y[idx]).abs() < IN_LANE
        # band-relative x: 0 at this episode's tent start (a virtual one on flat lanes)
        t = pos[:, 0] - self.ep_start_x[idx] - (FLAT_APPROACH - SPAWN_OFF)
        loc = torch.bucketize(t, self._loc_edges, right=True)             # 0..5
        pushed = self.n_push[idx] > 0
        since = self.steps[idx] - self.push_step[idx]
        cols = [torch.ones_like(term), to, term, term & tilt, term & ~tilt,
                crossed, cleared, cleared & to, term & ~crossed, term & crossed,
                *[term & (loc == b) for b in range(6)],
                self.n_push[idx], pushed, term & pushed & (since <= self.push_window), term & pushed,
                ~inlane_end]
        m = torch.stack([c.to(torch.float64) for c in cols], dim=1)
        ep = self.ep_index[idx]
        if self.count_episodes:
            m = m * (ep < self.count_episodes).to(torch.float64)[:, None]
        row = ((self.block_id[idx] * 2 + (~self.is_stairs[idx]).long()) * self._ep_bins
               + ep.clamp(max=self._ep_bins - 1))
        self._stats.index_add_(0, row, m)
        self._prog_sum.index_add_(0, row, prog.to(torch.float64) * m[:, 0])
        if self._istats is not None:
            self._count_instrument(idx, term, pos, row, ep)
        self.ep_index[idx] += 1

    def _count_instrument(self, idx, term, pos, row, ep):
        """Flush the ending episodes' instrument accumulators plus spot_feet.EPISODE_COLS into the same
        (block, lane type, episode bin) rows, under the same first-E gate as the STAT_KEYS counters."""
        fl, fh, cause = self._first_legacy[idx], self._first_honest[idx], self._honest_cause[idx]
        hb, hu = self._hit_base[idx], self._hit_uleg[idx]
        dy = pos[:, 1] - self.lane_y[idx]
        cols = [term & self._tilt_sus[idx], term & (hb | hu), term & hb, term & hu, term & self._legacy_now[idx],
                term & self._honest_now[idx], fl >= 0, fh >= 0, (fh >= 0) & ((cause & 1) > 0),
                (fh >= 0) & ((cause & 6) > 0), (fl >= 0) & ((fh < 0) | (fl < fh)),
                (fh >= 0) & ((fl < 0) | (fh < fl)), dy.abs(), dy, term & self._knee_now[idx], self._knee_fired[idx],
                dy * self._downhill_y[idx] if self._slope_lanes is not None else torch.zeros_like(dy)]
        m = torch.cat([self._iacc[idx], torch.stack([c.to(torch.float64) for c in cols], dim=1)], dim=1)
        if self.count_episodes:
            m = m * (ep < self.count_episodes).to(torch.float64)[:, None]
        self._istats.index_add_(0, row, m)

    def reset_stats(self, episodes=True):
        """Zero the episode counters; `episodes` also restarts every env's episode index (the first-E
        window), which an eval wants and a trainer logging per window does not. The slope-lane spawn counters
        are left alone: the reset an eval zeroes the counters after is the one whose settle they checked."""
        self._stats.zero_()
        self._prog_sum.zero_()
        if self._istats is not None:
            self._istats.zero_()
        if episodes:
            self.ep_index.zero_()

    def episode_stats(self):
        """The counters as plain Python numbers (one host sync): a row per (block, lane type) that has
        lanes, with the STAT_KEYS totals, the summed forward progress, and the same keys per
        episode-index bin (bin i = the i-th counted episode; without count_episodes, first / later)."""
        c = self._stats.view(self.n_blocks, 2, self._ep_bins, len(STAT_KEYS)).cpu().numpy()
        p = self._prog_sum.view(self.n_blocks, 2, self._ep_bins).cpu().numpy()
        ins = (None if self._istats is None else
               self._istats.view(self.n_blocks, 2, self._ep_bins, sf.WIDTH).sum(dim=2).cpu().numpy())
        spawn = self._spawn_counts() if self._hold_row is not None else None
        rows = []
        for b in range(self.n_blocks):
            for lt, name in ((0, "stair"), (1, "flat")):
                if self._lanes[b, lt] == 0:
                    continue
                tot = c[b, lt].sum(axis=0)
                row = {"block": b, "lane_type": name, "lanes": int(self._lanes[b, lt])}
                if self.block_names is not None:
                    row["block_name"] = self.block_names[b]
                row.update({k: int(round(float(v))) for k, v in zip(STAT_KEYS, tot)})
                row["progress_sum"] = float(p[b, lt].sum())
                row["by_episode"] = [{k: int(round(float(v))) for k, v in zip(STAT_KEYS, c[b, lt, e])}
                                     for e in range(self._ep_bins)]
                if ins is not None:          # summed over the episode bins; spot_feet.*_summary reads it
                    row["instrument"] = {name: (float(ins[b, lt, o]) if w == 1 else ins[b, lt, o:o + w].tolist())
                                         for name, (o, w) in sf.LAYOUT.items()}
                if spawn is not None and name == "flat" and spawn[b]["lanes"]:    # slope lanes are flat-type
                    row["spawn"] = spawn[b]
                rows.append(row)
        return {"stat_keys": list(STAT_KEYS), "count_episodes": self.count_episodes,
                "episode_bins": self._ep_bins, "rows": rows, "instrument": ins is not None}

    def observe(self, s):
        q = s.root_quat
        lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
        ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
        proj_g = quat_rotate_inverse(q, self.grav.expand(self.K, 3))
        qpos = s.joint_pos[:, self.i2a] - self.default_q
        jv_isaac = s.joint_vel[:, self.i2a]
        x, y, zz = s.root_pos[:, 0], s.root_pos[:, 1], s.root_pos[:, 2]
        cyaw, syaw = heading_cossin(q)
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)
        if self._scan_dx is not None:        # the counterfactual moves what the policy SEES, nothing else
            px = px + (self._scan_dx * cyaw)[:, None]
            py = py + (self._scan_dx * syaw)[:, None]
        if self.percept is not None:
            # What the robot could have SEEN: occluded and never-observed cells read flat, exactly
            # as they do at deploy. Reward, termination and the spawn keep reading ground truth —
            # the policy is handicapped, the training signal is not.
            ahead, h_here = self.percept.read(self.sim.root_position, q, px, py)
        else:
            h_here = self._terrain_h(x, y)
            ahead = (self._terrain_h(px, py) - h_here[:, None]).clamp(-1.0, 1.0)
        base_above = (zz - h_here).unsqueeze(-1)
        clk = clock_obs(self.phi)                                          # [K,2] clock after last substep
        if self.stand_mode:                  # the stand sentinel: (0,0) is a value sin/cos never takes
            clk = torch.where((self.cmd == 0.0).all(dim=1, keepdim=True), torch.zeros_like(clk), clk)
        # Layout: [proprio(48)|clock(2)|base_above(1)|scan(45)] = 96-d
        # First 50 (proprio+clock) byte-identical to scratch_flat -> anchor reads obs[:,:50]
        obs = torch.cat([lin_b, ang_b, proj_g, self.cmd, qpos, jv_isaac, self.last_act,
                         clk, base_above, ahead], dim=1)
        if self.jump_obs:                    # [96:98] the jump flag and phase (jump_obs without triggers: zeros)
            if self.jump is not None:
                w = (self.jump_t >= 0).float()
                jch = torch.stack([w, self.jump_t.clamp_min(0).float() / JUMP_WINDOW * w], dim=1)
            else:
                jch = torch.zeros(self.K, 2, device=self.device)
            obs = torch.cat([obs, jch], dim=1)
        if self.blowup_guard:
            obs = torch.nan_to_num(obs, nan=0.0, posinf=OBS_LIMIT, neginf=-OBS_LIMIT).clamp(-OBS_LIMIT, OBS_LIMIT)
        self._last_obs.copy_(obs)     # in place: reward_terms reads this, and a graph captures the
        return obs                    # address, so rebinding would strand it on a stale tensor

    def config(self):
        return {**super().config(), **CONFIG, "risers": list(self.riser_list)}

    @torch.no_grad()
    def measure_tracking(self, act_fn, cmd, steps=160, warm=60):
        dev = self.device
        c = torch.tensor(cmd, device=dev, dtype=torch.float32).expand(self.K, 3).contiguous()
        obs = self.reset()
        errs = []
        for t in range(steps):
            self.cmd.copy_(c)
            self.cmd_timer.fill_(10 ** 9)
            obs, _, _, _, _ = self.step(act_fn(obs))
            if t >= warm:
                q = self.sim.root_quat
                lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
                ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
                e = (lin_b[:, :2] - c[:, :2]).norm(dim=1).mean() + (ang_b[:, 2] - c[:, 2]).abs().mean()
                errs.append(e.item())
        return sum(errs) / max(1, len(errs))


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    K = int(os.environ.get("K", "64"))
    src = os.environ.get("HEIGHT_SOURCE", "analytic")     # HEIGHT_SOURCE=raycast -> Warp BVH scan
    see = os.environ.get("PERCEIVE", "") == "1"           # PERCEIVE=1 -> the camera-limited scan
    gph = os.environ.get("GRAPH", "") == "1"              # GRAPH=1 -> CUDA-graph the step's torch region
    env = SpotStepsEnv(num_envs=K, height_source=None if see else src, perceive=see, graph=gph)
    obs = env.reset()
    assert obs.shape == (K, 96), f"expected obs (K,96), got {tuple(obs.shape)}"
    print(f"obs {tuple(obs.shape)} (OBS_DIM={OBS_DIM}=96) finite={bool(torch.isfinite(obs).all())}  "
          f"levels={N_LEVELS} risers={RISERS}  height_source={env.height_source}"
          + (f" {env.rays}" if env.rays is not None else "")
          + (f"  {env.percept}" if env.percept is not None else ""))
    for _ in range(200):
        obs, rew, done, term, to = env.step(torch.zeros(K, ACT_DIM, device=env.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"zero-action (stand): track={env.last_track:.3f}  flat_track={env.last_flat_track:.3f}  "
          f"level={env.last_level:.2f}  fell/step={env.last_fell:.3f}  rew={rew.mean().item():+.3f}")
    print("per-term:", env.stats_line() or "(no episode finished yet)")
    if gph:
        print(f"graph replay verified: worst |obs - recomputed| = {env.verify_graph(steps=32):.3e}")
    print("SPOTV2-STEPS ENV SELFTEST: PASS")
