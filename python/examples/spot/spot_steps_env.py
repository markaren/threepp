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
from spot_deploy import build_spot, default_q, add_to_isaac, isaac_to_add, ACTION_SCALE
from spot_terrain_env import (quat_rotate_inverse, up_z, heading_cossin, _flat_ground,
                              scan_offsets, scan_xy, N_SCAN,
                              CONTROL_HZ, DT, SUBSTEPS, SPACING, SPAWN_Z, PROBE_DX, ACT_DIM,
                              HIDDEN, HALF_W, VX_LO, VX_HI, VY_HI, WZ_HI, STAND_PROB,
                              FWD_DRIVE_FRAC, CMD_MIN, CMD_MAX, SIG)
from scratch_clock import CLOCK0, CLOCK_DIM, GAIT_PERIOD, advance, clock_obs, reset_phi
from scratch_env import STIFF_GAINS
import spot_feet as sf
import spot_slopes as ss

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
LANE_FAMILIES = {"stairs": "flat", "flat": "flat", "slope_up": "up", "slope_down": "down", "cross": "cross"}

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


def _add_steps(world, k, spacing, is_stairs, risers_np, w=HALF_W_BOX, bands=None):
    """Per STAIR lane: the band ladder — for each level j a tent at riser risers_np[j], placed at band
    j's x-offset (flat approach in front, run-out behind). Solid boxes from the ground up to each tread
    top. `bands` [k], when given, builds only that one band per lane: an eval world with frozen levels,
    where each lane meets one tent and then flat ground, as the top rung always has."""
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
                world.add_static(b)
            up_end = x0 + N_UP * STEP_RUN
            top = N_UP * r
            lb = tp.Mesh(tp.BoxGeometry(LAND, w, top), tp.MeshStandardMaterial())   # landing
            lb.position.set(up_end + LAND * 0.5, y, top * 0.5)
            world.add_static(lb)
            land_end = up_end + LAND
            for s in range(N_UP - 1):                              # descend: tread s top at (n-1-s)*r
                h = (N_UP - 1 - s) * r
                b = tp.Mesh(tp.BoxGeometry(STEP_RUN, w, h), tp.MeshStandardMaterial())
                b.position.set(land_end + s * STEP_RUN + STEP_RUN * 0.5, y, h * 0.5)
                world.add_static(b)


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
                 lane_family=None, slope_deg=None, slope_material=None, slope_cross_mode="corrugate"):
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
        """
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
        mu_np = None if foot_mu is None else _per_env(foot_mu, num_envs, np.float64, "foot_mu")
        pay_np = None if payload_kg is None else _per_env(payload_kg, num_envs, np.float64, "payload_kg")
        self.drive_limits_are_forces = dlf = bool(drive_limits_are_forces)
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
                                          gains=STIFF_GAINS, drive_limits_are_forces=dlf, **kw)
        # Under "raycast" the builders write into a CollectedWorld, which forwards every call
        # to the real world and keeps the Mesh it was handed — so the BVH is built from the
        # boxes PhysX actually got, not from a second description of them.
        collector = []

        def _build(world):
            if height_source == "raycast":
                from threepp.rl.raycast import CollectedWorld
                world = CollectedWorld(world)
                collector.append(world)
            _flat_ground(world, num_envs, SPACING, length=ground_len, x_center=ground_cx)
            _add_steps(world, num_envs, SPACING, is_stairs_np, risers_np, bands=bands)
            if slope_lanes is not None:
                slope_lanes.build(world, slope_material)    # materials stay alive on slope_lanes

        super().__init__(num_envs, lambda world, i: _SpotStepsRobot(world, i),
                         gravity=(0.0, 0.0, -9.81), spacing=SPACING, device=device, seed=seed,
                         read_root=True, read_links=bool(read_links) or self._feet_on or slope_lanes is not None,
                         build_world=_build, graph=graph)
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
        self._last_obs = torch.zeros(num_envs, OBS_DIM, device=dev)
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
        if push_mode not in ("random", "once_on_tent"):
            raise ValueError(f"push_mode must be 'random' or 'once_on_tent', got {push_mode!r}")
        self.push_mode = push_mode
        self.push_lanes = (self.is_stairs.clone() if push_lanes is None else
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
        if self._slope_lanes is not None:
            self._init_slopes(slope_code)
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

    def on_pre_substep(self):
        """Shove a random subset, world-frame and horizontal, for one substep.

        PhysX clears link forces after every step, so this lands as an impulse of
        push_vel * PUSH_MASS N*s and nothing persists. Skipped while the spawn is settling —
        knocking the robot over before the episode starts measures nothing.

        push_mode 'once_on_tent' instead lands exactly one shove of push_dv per episode, where the
        base first passes that episode's trigger x (see on_reset)."""
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
        return cmd, torch.randint(CMD_MIN, CMD_MAX + 1, (n,), device=dev)

    def _resample_cmd(self, idx):
        """Subset form, for the reset path. Runs eagerly — idx already cost a host sync."""
        if idx.numel() == 0:
            return
        cmd, timer = self._sample_cmd(self.is_stairs[idx])
        self.cmd[idx] = cmd
        self.cmd_timer[idx] = timer

    def _resample_due(self):
        """Step form: sample for every env, keep it only where the timer ran out. K*3 extra randoms
        instead of a nonzero, which is the trade that keeps the step graph-capturable."""
        due = self.cmd_timer <= 0
        cmd, timer = self._sample_cmd(self.is_stairs)
        self.cmd.copy_(torch.where(due.unsqueeze(1), cmd, self.cmd))
        self.cmd_timer.copy_(torch.where(due, timer, self.cmd_timer))

    def _spawn_x(self, idx):
        """Spawn x for the subset: stair lanes at their level's band approach; flat lanes anywhere flat."""
        dev = self.sim.device
        sx = self.level[idx].to(torch.float32) * self.band_len + SPAWN_OFF
        flat = ~self.is_stairs[idx]
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

    def _settle_check(self):
        """spot_slopes.settle_health at the end of a full reset's settle: up_z >= 0.98 cos(slope fitted under the
        spawn) and all four feet within CONTACT_TOL of the plane along its normal. Latched per env (the next full
        reset overwrites it; a partial reset has no settle to check), read per block in episode_stats()."""
        tips = self._slope_fk.tips(self.sim.link_pose)
        nz = self._slope_lanes.normals(tips[..., 0], tips[..., 1])[..., 2]
        ok, _, gap = ss.settle_health(self.up, tips, self.rays.heights, self._spawn_nz, normal_z=nz)
        self._settle_checked.copy_(self._slope_row)
        self._settle_bad.copy_(self._slope_row & ~ok)
        self._settle_uprel.copy_(self.up / self._spawn_nz)
        self._settle_gap.copy_(gap.max(dim=1).values)

    def _spawn_counts(self):
        """Per block (one host read): lanes that are slope lanes, the settle health of the last full reset, and the
        spawn geometry over every reset. A block is flagged when either failure share exceeds
        spot_slopes.SPAWN_FAIL_MAX, or when it has slope lanes and no settle was checked."""
        host = lambda t: t.cpu().numpy()
        bid, row, chk, bad = host(self.block_id), host(self._slope_row), host(self._settle_checked), host(self._settle_bad)
        res, unr, upr, gap = host(self._spawn_resets), host(self._spawn_bad), host(self._settle_uprel), host(self._settle_gap)
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
        self.sim.set_root_state(idx, pose)
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
        st = idx[self.is_stairs[idx]]
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

    def act(self, a):
        # FULL policy action (not a residual): isaac -> add-order drive targets. The settle loop
        # feeds a=0, which lands exactly on the default stand targets.
        if self.settling and self._slope_lanes is not None:
            # slope rows hold their spawn joints through the settle (spot_slopes INTEGRATION 5); every other row
            # keeps a = 0. last_act then says what the drives hold, which is what the policy is handed.
            a = torch.where(self._slope_row[:, None], self._settle_act, a)
        self.prev_act.copy_(self.last_act)
        self.last_act.copy_(a)
        return (self.default_q + ACTION_SCALE * a)[:, self.a2i]

    def on_settled(self):
        self.up = up_z(self.sim.root_quat)
        if self._slope_lanes is not None:
            self._settle_check()

    def on_step(self, s):
        self.phi.copy_(advance(self.phi, period=self.period))                # clock after physics, before next obs
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
        self.ep_max_climb.copy_(torch.maximum(self.ep_max_climb, (x - self.ep_start_x).clamp_min(0.0)))
        # Latch how far off its lane centre the robot was the FIRST time it got past the tent, which
        # is where the in-lane half of a clear is read (see IN_LANE). A where, so the step stays
        # graph-capturable and costs no host sync.
        self.clear_dy.copy_(torch.where((self.ep_max_climb > self.clear_dist) & (self.clear_dy < 0.0),
                                        (y - self.lane_y).abs(), self.clear_dy))
        if self._feet_on:
            self._instrument_step(x, y)

    def terminated(self, s):
        tilt = self.up < 0.35
        self._term_tilt.copy_(tilt)          # the cause, for the counters (written through, graph-safe)
        if self.termination == "honest":
            return self._honest_now          # set this tick by _instrument_step
        return tilt | (self._base_above < 0.18)

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
        # Device-side, and read back only when a trainer asks (see the last_* properties). Masked
        # means rather than boolean indexing: a mask keeps the shapes static, which is what lets the
        # whole reward be replayed from a CUDA graph.
        trk = track_lin + track_ang
        self._st_fell.copy_(s.terminated.float().mean())
        self._st_track.copy_(trk.mean())
        flat = (~self.is_stairs).float()
        self._st_flat.copy_((trk * flat).sum() / flat.sum().clamp_min(1.0))
        st = self.is_stairs.float()
        self._st_level.copy_((self.level.float() * st).sum() / st.sum().clamp_min(1.0))
        return terms

    def on_done(self, idx):
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
        crossed = prog > self.clear_dist
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
        spawn = self._spawn_counts() if self._slope_lanes is not None else None
        rows = []
        for b in range(self.n_blocks):
            for lt, name in ((0, "stair"), (1, "flat")):
                if self._lanes[b, lt] == 0:
                    continue
                tot = c[b, lt].sum(axis=0)
                row = {"block": b, "lane_type": name, "lanes": int(self._lanes[b, lt])}
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
        # Layout: [proprio(48)|clock(2)|base_above(1)|scan(45)] = 96-d
        # First 50 (proprio+clock) byte-identical to scratch_flat -> anchor reads obs[:,:50]
        obs = torch.cat([lin_b, ang_b, proj_g, self.cmd, qpos, jv_isaac, self.last_act,
                         clk, base_above, ahead], dim=1)
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
