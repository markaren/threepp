# crane_lift: state of the demo (2026-09-07, branch `crane-demo`)

The candidate replacement for the ROV net-pen hero in the ICRA 2027 paper: an
offshore crane on a heaving vessel lands a container on a wind-turbine
platform. Built in one 120-minute session from the co-author's crane twin
(`crane_turbine_operation_threepp`, Erik Espenakk) and the netpen demo's
audit pattern. Nothing in the paper has been changed.

## What exists

`python/examples/crane_lift.py`, one script, Vulkan only:

- The vessel rides the FFT ocean the renderer draws: heave, pitch and roll
  from four buoyancy points (`ocean.sample_height`), a 1.6 s first-order lag
  for the hull's inertia, the hull exclusion so no wave crosses the deck.
- The crane is the twin's `crane.glb` rig (King slew about its Z, the inner
  `C1` hinge for the luff, `Telescope.position.y` for the extension), mounted
  on the stern deck as the twin mounts it (rotate -90 deg X, -180 deg Z).
  The three wires are our own cylinders aimed between the glb's winch and
  target nodes; the glb's stretched wire meshes are hidden.
- Active motion compensation: the tip is held at a world target by damped
  Gauss-Newton IK on a finite-difference Jacobian through the scene graph
  (the twin's `solveTipIK`), then the twin's first-order drive lag (the crane
  is electric) and
  the Seaonics C25 rate and acceleration limits per joint. The slew range is
  unwrapped to +-360 deg; luff and telescope keep the C25 envelope.
- The payload is the 20 ft container on four slings under the hook, the hook
  on the hoist wire as a spherical pendulum (position-based, four substeps,
  0.08 damping). The wire is a distance constraint: it can go slack, it
  never pushes.
- Perception in the loop: a tip-mounted range fan, 28 x 28 ray-traced beams
  in a +-26 deg cone straight down, one `scan_lidar` dispatch at 20 Hz with
  seeded Gaussian range noise (20 mm). The container is the cluster of
  returns nearer than the deck; its centroid minus the tip is the swing
  estimate, and the anti-swing law moves the tip target after the load
  (0.30 on the offset, 0.40 s on its rate, clamped to 1.2 m, and opened
  when the estimated swing exceeds 3 m as a runaway guard).
- The operation: 4 s hold over the pickup on the port quarter (16 m from the
  king), a 40 s slew ROUND the king at the boom's reach to the platform
  (a straight line would pass over the pedestal and saturate the telescope
  at zero and the luff at its limit), 12 s pay-out to 1 m over the grating,
  hold. No joint touches a limit in the 50 s run. `--op SECONDS` runs it and writes the
  manifest; `--seed` moves only the fan's noise.
- A tip camera as a picture-in-picture secondary view (480 x 270).
- `--audit N` writes the scene's determinism manifest in `sensor_audit.py`'s
  format so two fresh processes are judged by `sensor_audit.py --compare`:
  rows `rgb`, five AOVs, `tip.rgb`, `fan`, `events.raw`, `events.sorted`
  (the co-author's content-versus-order split), `traj`, `vessel`.

## Assets

`crane.glb`, `Vessel_Fixed.glb`, `Wind_Turbine.glb` are Seaonics-derived and
stay outside the repository until their release is cleared; the script reads
them from `CRANE_ASSETS_DIR`. The manifests in this folder were produced with
the copies in the co-author's alignment package.

## Measured (RTX 4060 Laptop GPU, driver as recorded, Windows 11, 1280 x 720)

Every manifest is `sensor_audit.py`'s format; compare two with
`python sensor_audit.py --compare a.json b.json`, or a folder with
`python analyze.py <folder>` (per-row replay, the first differing frame of
the rendered rows from the per-frame hashes, the seed spread, the control).

### Round 1, grey-cube payload, sensor blind (`round1_grey_cube/`)

The first version: the fan's rays started inside the boom-tip mesh, so the
sensor never saw the load and the anti-swing did nothing (the control run
printed the same numbers). Kept because the replay result stands:

| run | rows | result |
|---|---|---|
| audit, 120 frames, 2 fresh processes | 12 | 10 bit-identical; `rgb`, `tip.rgb` differ |
| the same with the clouds off | 12 | 10 bit-identical; `rgb`, `tip.rgb` differ (clouds are not the carrier) |
| lift, 30 s (1800 frames), 2 fresh processes | 12 | 12 bit-identical, rendered frames included |

### Round 2, container payload, sensor live, sea raised (`round2_*/`)

| run | rows | result |
|---|---|---|
| audit, 120 frames, 3 fresh processes | 12 | 12 bit-identical, per-frame hashes equal on every frame |
| lift, 50 s (3000 frames), 3 fresh processes | 12 | 12 bit-identical |

The lift of round 2 was UNSTABLE (swing RMS 3.9 m): the sea at wave_scale
2.8 with the pickup and platform at the edge of the envelope put the
telescope at its minimum and the luff at its maximum for a fifth of the
frames. The replay holds for an unstable controller exactly as for a stable
one, which is the point of the audit; the controller was then fixed (round 3).

So far the rendered-frame rows have agreed in five fresh-process sets and
differed in two pairs, both in round 1, both 120-frame audits; the AOVs, the
fan, the event stream (raw AND sorted), the trajectory and the vessel have
agreed in every set. The frame difference is intermittent and not the
clouds; the per-frame hashes added in round 2 will name its first frame the
next time it appears.

### Round 3, no joint at a limit (`round3/`)

| run | rows | result |
|---|---|---|
| lift, 50 s, seed 0, 2 fresh processes | 12 | 12 bit-identical (rendered frames and both event hashes included) |
| lift, 50 s, seed 0, anti-swing OFF (control) | 12 | one process; the physics comparison below |

| | swing RMS | swing max | hold phase (after 44 s) RMS |
|---|---|---|---|
| anti-swing on (0.30, 0.40 s, clamp 1.2 m) | 3.24 m | 6.51 m | 2.73 m |
| anti-swing off | 0.84 m | 2.57 m | 0.82 m |

The loop as tuned makes the swing WORSE, by four times. The shot-mode
test of the same code gave 1.2 m RMS; the manifest run starts the
operation half a second later against the sea, and that phase difference
is enough to take a marginally damped loop to a large swing. Deterministic,
sensor-driven, not robust: the anti-swing law is the placeholder it was
declared to be, and the next step is either delayed-feedback damping with
the pendulum period, or the twin's velocity feedforward from the vessel
motion, before any seed-spread or ten-process run is worth making.

Event camera at 24 s (pictures in `round3/`): from the tip looking down
the wire, the `final` source fires 385 k events per frame, most of them
the sea's sun glints; the `shaded` (Lambert proxy) source fires 96 k and
shows the container, the slings and the platform edge crisply with the
sea quiet. This is the co-author's ocean finding (specular saturation)
reproduced in one frame, and the reason a tip event camera in the loop
would use the shaded source or an albedo override. From the hero view:
13 k (final) and 14 k (shaded) events per frame. No overflow anywhere.

### Round 5, the feasible transfer, the landing and the integral law (`round5_demo/`)

The scene changed in five places (`crane_lift.py`): defaults `--amc-kp 3 --ff-mode both`, the
in-house TAA selected before the first frame (`aa: taa` in the manifest, no upscaler in the
loop); a 60 s transfer instead of 40 s and a tip target over the platform that finally carries
the WIRE as well as the sling drop (rounds 1-4 put the container 5.5 m UNDER the grating, inside
the turbine base, and nothing in the log said so); the tip sensor gated on the container's
instance id, its head moved 0.45 m outboard of the hoist wire and its centre taken as the
mid-range of the returns in the load's own frame; the phase-1 integral law
(`--law integral|legacy|off`, `--as-k`); and a grating contact so the container lands, sticks and
lets the wire go slack, with a `landed` state that disengages the loop.

The sensor was the round's real finding. On the wire's axis every one of the 784 rays left
through the inside of the 0.05 m hoist-wire cylinder, so rounds 1-4 measured the WIRE, not the
container - the near-vertical rays ran 0.5 to 6 m inside it and passed the range gate. That is
the 0.84 / 0.53 gain of the plan's diagnosis, and it is why the swing estimate looked like half
the swing: a point half way down the wire. Moved outboard and gated on the id, the fan carries
about 170 container returns per scan and the estimate's gain is 1.02 to 1.05 with a mean bias of
7 mm. The centroid of those returns is NOT the container's centre (the wire, hook and slings
shadow the inboard half of the top face; the centroid sat 0.27 m outboard, gain 3.8), so the
estimator takes the mid-range in the load frame instead.

Five runs, seed 0, 76 s (4,560 frames), 1280 x 720, RTX 4070, 93 to 98 ms/frame. Swing is the
horizontal hook-tip distance; the phases are 0-4 s hold, 4-64 s transfer, 64-76 s pay-out.

| run | swing RMS/max | hold | transfer | pay-out | hook dev, pay-out before touchdown | settle | tip-target RMS | slew at rate limit | landed |
|---|---|---|---|---|---|---|---|---|---|
| loop open | **276 / 1041 mm** | 58 / 91 | 310 / 1041 | **51 / 107** | **36 mm** | 0.0 s | **123 mm** | 9.2 % | 73.67 s, 57 mm off |
| integral k 0.5 | 879 / 2961 | 46 / 69 | 992 / 2961 | 66 / 178 | 89 mm | 0.0 s | 615 mm | 18.5 % | 73.67 s, **46 mm** off |
| integral k 0.8 | 1071 / 3157 | 42 / 69 | 1207 / 3157 | 179 / 486 | 166 mm | 3.3 s | 728 mm | 20.7 % | 73.67 s, 104 mm off |
| integral k 1.2 | 1197 / 3348 | **37 / 69** | 1331 / 3348 | 512 / 1164 | 312 mm | 10.0 s | 736 mm | 20.6 % | 73.70 s, 134 mm off |
| legacy PD | 1128 / 2666 | 48 / 71 | 866 / 1896 | 2048 / 2666 | 2906 mm | never | 534 mm | 8.5 % | no |

The loop still loses to the control, and the reason is measured this time: the SLEW HAS NO
AUTHORITY LEFT during the transfer. The 60 s profile peaks at 0.0616 rad/s, 73 percent of the
C25's 0.0838, exactly as phase 1 designed - but the motion compensation against the sea needs
0.020 rad/s RMS and 0.052 rad/s at the 99th percentile of its own, so the slew is already at its
rate limit in 9.2 percent of frames with the loop OPEN (36 percent between 20 and 34 s). Add the
law's correction and it is 18 to 21 percent; the conditional anti-windup cannot hold, because
saturation is intermittent and the integrator grows in the gaps until it reaches its 1.5 m clamp
(measured: |u| max 1.50 m in all three integral runs). Below 16 s, where nothing saturates, the
law does what phase 1 predicted: 55 mm RMS against the control's 83 to 114 mm, |u| under 0.19 m.
The gain ordering is monotone - the smaller k, the less damage - so k = 0.5 is the pick, and it
lands closest of all five runs (46 mm from the platform point). The legacy PD law diverges as it
did in round 3 and never sets the container down.

To make the transfer genuinely feasible the profile's peak must leave the compensation its
0.05 rad/s: peak <= 0.034 rad/s, i.e. a transfer of 105 s or more (`--xfer`), or a shorter slew
sweep than 141 degrees, or a pickup nearer the king. That is the next thing to try, and until it
is tried the honest claim for E3 is the one the control run supports.

Two more limits worth writing down. The landing point PLATFORM is the turbine glb's
`CenterPoint (Should be at Landing Target)` node, and that node sits on the platform deck's
south-west CORNER: the deck runs x 25.0 to 38.7, z 22.3 to 32.6 (13.7 x 10.3 m, walking surface
y = 20.50 over plating at 20.30) away from it, so the container comes to rest with about half its
length over the edge. The crane cannot do better - the deck's nearest point is 19.0 m from the
king and the telescope is at 10.6 of 11 m there - so the fix is the vessel or the turbine three
metres closer, not the controller. And the contact test is a 4.5 m radius about PLATFORM rather
than the measured rectangle, which would reject the landing point itself by 4 cm.

### Round 6, the deck, the operator's transfer, on-demand, and PhysX (`round6_physx/`)

Four changes, each behind a flag with round 5 kept: `--geom new|old`, `--xfer-profile
trapezoid|smooth`, `--engage ondemand|always`, `--payload physx|pbd`.

**Geometry.** The turbine glb was measured rather than guessed: the RAILING encloses x -8.96 to
+4.69, z -4.72 to +5.62 about the turbine's own axis, the grating's walking surface is at
y = 20.50, and the tower is a 3.08 m radius cylinder standing on that deck ON the turbine axis.
So the deck is 13.65 x 10.34 m with a 6.2 m tower in the middle of it and only the -x arm is
usable: 8.96 m from the near railing to the axis, **5.88 m of clear deck** between railing and
tower surface. A 7.0 m container fits there only ACROSS the boom. The turbine moved from
(34, 0, 27) to (30.10, 0, 24.55), which puts the landing point at (24.50, 20.50, 25.00),
**18.50 m from the king** (round 5: 19.0 m, with the telescope at 10.6 of 11 m). The plan asked
for the landing point 4.5 m inside the near railing; that is geometrically impossible here -
4.5 m in is local x = -4.46 and the container's 1.43 m half-width then reaches 3.03 m from the
axis, inside the 3.08 m tower. The compromise taken is 3.36 m inside the railing. Clearances:
container's near face 1.93 m inside the near railing, both ends 1.67 m inside the side railings,
boom tip 5.62 m from the tower axis, container's nearest FACE 1.09 m from the tower surface and
its nearest CORNER 2.09 m. No joint reached a position limit in any of the six runs (0/3930).

**The transfer.** A trapezoid instead of a smoothstep: ramp at the C25's 0.0838 rad/s^2 to 80
percent of its 0.0838 rad/s rate limit, cruise, ramp down. Over the 137.5 deg sweep that is
0.80 + 34.99 + 0.80 = **36.59 s**, and its length is a consequence rather than a choice. Luff and
telescope stay on smoothsteps over the same interval. Phases: hold 0-4.00, transfer 4.00-40.59,
**arrival hold 40.59-48.59**, pay-out 48.59-60.59, then hold (`--op 65`). The pay-out starts at a
fixed time in every variant, so the runs are compared at equal times. The residual swing this
leaves at arrival is **1068 mm** - two orders more than the smoothstep's, which is the point.

**PhysX payload.** An invisible 2.86 x 3.11 x 7.00 m box proxy at 193 kg/m^3 (12.0 t) on ONE
`Joint.Type.DISTANCE` tether (upper 6.0 m, lower 0, no stiffness) from a kinematic sphere at the
boom tip; `add_static_trimesh_tree(turbine)` for the platform, railing and tower (13 colliders,
0.01 s to cook, 0.01 s to build the whole world); `create_material(0.6, 0.5, 0.0)`;
`ContactSensor(proxy)` as the touchdown switch and `Joint.reaction()` as the load cell (both
manifest rows). No deck collider - the container hangs 0.3 m clear at t = 0. The tether reads
118.6 kN carrying the load against mg = 117.7 kN and 0.00 kN once it is down, and stretches at
most **25 mm over 6.0 m** (0.4 percent) with ONE substep per frame.

Two joint-frame details. `Joint`'s constructor derives both local anchors from one world frame,
so the container is created with its hook point AT the tip (baking anchors of (0,0,0) and
(0, 4.155, 0)) and then `set_pose`d down to hang. And `Params.upper` is read once at creation
with no way to reopen it, so rather than re-create the joint every centimetre of pay-out - which
throws away the solver's warm start and restarts `reaction()`, the very signal the landing is
detected with - the tether keeps its length and the KINEMATIC ANCHOR is lowered by the pay-out.
Same motion, one pendulum period throughout, and the visual wire is drawn from the real tip.

The container's heading is now fixed (`LOAD_YAW`): a distance joint transmits no torque, so a
PhysX container keeps whatever heading it was picked up with (its quaternion's y stayed 0.00000
over a whole lift), and the heading is chosen so the 7 m length lies across the boom at the
landing. The tip fan widened to +-42 deg to cover metres of operator swing, and the estimator's
yaw comes from that constant instead of the slew encoder.

Six runs, seed 0, 65 s (3,930 frames), 1280 x 720, RTX 4070, 115 ms/frame. Swing is the
horizontal distance from the tip to the CONTAINER'S ANCHOR.

| run | hold | transfer | arrival hold | pay-out to touchdown | swing at T2 | at pay-out start | settle <0.25 m | touchdown | offset | on deck | tip-target | tower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| open loop | 35 | 837 / 1565 | 516 / 1068 | 333 / 755 | 1068 | **461** | never | 58.92 s | 203 mm | yes | **198** | **struck, -0.07 m** |
| k 0.3 | 35 | 837 / 1565 | 364 / 1068 | 369 / 709 | 1068 | 193 | 7.69 s | 56.60 s | 108 mm | yes | 216 | struck, -0.51 m |
| **k 0.5** | 35 | 837 / 1565 | 324 / 1068 | 327 / 634 | 1068 | 167 | 7.67 s | 56.52 s | **78 mm** | yes | 226 | struck, -0.50 m |
| k 0.8 | 35 | 837 / 1565 | 317 / 1068 | 321 / 595 | 1068 | 204 | 7.59 s | 56.55 s | 133 mm | yes | 237 | struck, -0.48 m |
| k 0.5, engage always | 45 | **579** / 1548 | **320** / 645 | **115** / 222 | **378** | **52** | **5.87 s** | 58.88 s | 220 mm | yes | 681 | **never, +0.75 m** |
| k 0.5, payload pbd | 58 | 638 / 1392 | 363 / 984 | 102 / 168 | 821 | 93 | 3.72 s | 58.88 s | 176 mm | yes | 225 | n/a |

(mm RMS / max; the pay-out column stops at touchdown, after which the metric measures the tip's
own correction offset against a load that is no longer moving.)

The on-demand loop does beat the open loop where it acts: the swing at the moment the pay-out
starts falls 461 to 167 mm at k = 0.5 (2.8x), the container reaches the deck 2.4 s sooner
because it is hanging straighter, and it lands 78 mm from the point against 203 mm. k = 0.5 is
the pick: k = 0.3 damps less and k = 0.8 lands wider with a bigger correction (|u| max 825 mm
against 580).

But the round's real finding is the one the control run made. **The operator's transfer swings
the container into the tower at t = 40.33 s, 0.26 s BEFORE the arrival hold begins**, so the
on-demand loop cannot prevent the strike - it is not yet engaged. Worse, once engaged it makes
the strike deeper (-0.48 to -0.51 m of penetration against the control's -0.07 m), because the
law's whole content is "move the tip toward the load" and the load is against an obstruction:
the loop presses it in. The always-engaged run is the one that clears the tower entirely
(minimum clearance +0.75 m, no contact at all before the pay-out) and settles to 52 mm by the
start of the pay-out, and it pays for that with the tip 681 mm RMS off its nominal target and
the slew on its rate limit in 50 percent of the transfer's frames against 34.9. So on-demand
buys accuracy at the landing and always-on buys clearance from the structure, and neither is
free.

Two more measurements worth keeping. The tip sensor's gain reads **1.49 against the anchor and
0.88 against the container's own centre** (1.04 in the pbd run, where they are the same point):
that is geometry, not sensor error. A rigid container tilts with the rope, so its centre hangs
on an effective pendulum of wire + 4.16 m while the anchor hangs on the wire alone, and the
fan measures the top face. And **more PhysX substeps make this scene worse, not better**: at
`--physx-sub 4` the transfer swing goes from 837/1565 to 965/2528 mm and the tower penetration
from -0.50 to -0.69 m, because `set_kinematic_target` is called once per FRAME - PhysX moves the
kinematic tip over ONE substep and then holds it still for the other three, which is a stair-step
drive at four times the speed. One substep per frame is the right setting until the target is
interpolated in an `on_pre_substep` callback.

Open, for the next round: in the three on-demand runs the container comes to rest with its anchor
0.49 m higher than a level box on the grating (26.70 m against 26.21 m; the control run rests at
26.211 m, level to the millimetre), so it is perched on the deck furniture - the glb has three
1.09 m doors and their plugs standing on the walking surface inside the railing - rather than
bedded flat. Where the load comes down within a few tens of centimetres decides which. That, and
the tower penetration depth, are the two things to look at before this scene is frozen.

## Next

1. A swing law that damps (delayed feedback, or the twin's MRU feedforward),
   verified against the control run, then seeds 1..9 and ten fresh processes.
2. The event camera at the tip as the loop sensor, with the co-author's
   wire-line fit, shaded source.
3. A manifest that hashes on the GPU or subsamples the image rows: the
   50 s lift costs 6 to 11 minutes per process today, 55 ms/frame without
   the readbacks.
4. The intermittent rendered-frame mismatch of round 1 (two of seven sets):
   rerun 120-frame audits until it reappears; the per-frame hashes name the
   frame.
5. Asset clearance from Seaonics before any frame of this scene enters the
   paper or the video.


## Rounds 7 and 8 (2026-09-07 night): the scene as frozen

Lars's rule for the demo: it is a crane, and it uses sensors in the loop; everything else is
simplified. Two findings of round 6 set the scene: the tower is a 3.08 m cylinder standing on the
platform, so a 20 ft container beside it has about a metre of clearance and an operator's stop
swings it into the tower; and the sea of rounds 1 to 7 was too aggressive for a lift.

Round 7 (`round7_small/`, runs killed before completion): the payload is a 10 ft box (half the
20 ft glb's length, 5 t; `--load 20ft` keeps the old one), the landing point is chosen for
clearances (1.5 m to the tower surface, 1.0 m to every railing, clear of the three door leaves,
which are also taken out of the collider set) and offset in +z so the box reads beside the tower
in the hero frame; the turbine moved to (30.10, 0, 24.55) and the landing point is 18.5 m from the
king. `meta.geom` carries `land_off`, `clear_rect`, `rail_margin`, `tower_r`.

Round 8 (`round8_calm/`): the sea calmed (`--sea calm` is the default: wind 9 m/s, wave scale 1.3;
the old sea is `--sea fresh`), the operator's stop softened (`--ramp 1.5` s instead of the drive's
0.8 s limit), k = 0.5 on demand fixed (round 6's pick; no more sweeps). The look pass: the sun
moved from in front of the lens behind the tower to behind the camera's left shoulder
(SUN_DIR (-0.25, 0.50, 0.83)), exposure pinned at 0.90 by default (`--adapt-exposure` restores
adaptation; E1's rendered-frame row is pinned, and one of three round-4 lifts differed in `rgb`
with it adapting), 18 gulls abeam to port (`tp.Flock`, seeded, stepped on the sim clock, off
with `--no-gulls`), the hero camera (-40, 9, 58) to (18, 19, 27), the tip inset lower right.

Seed 0, 65 s, RTX 4070, TAA pinned, PhysX payload, one substep:

| | transfer | arrival hold | swing at pay-out start | pay-out | landed | offset from the mark | tilt | contact |
|---|---|---|---|---|---|---|---|---|
| loop open | 422 mm RMS | 491 mm RMS | 444 mm | 517 mm RMS | 58.15 s | 686 mm | 0 deg | grating only; tower 1.02 m at the closest |
| loop on (k 0.5, on demand) | 422 mm RMS | 240 mm RMS, under 150 mm 7.0 s after arrival | 108 mm | 72 mm RMS | 57.87 s | 20 mm | 0 deg | grating only; tower 0.94 m at the closest |

The transfer is identical in both runs because the loop is off during it (the `vessel` row is
the only manifest row the two runs share, as it should be). The sea in the run: heave span
0.20 m, roll 0.64 deg; hold-phase tip error 13 mm. Tension 49 kN carrying the 5 t box, zero at
rest.

A gotcha found on the way: `renderer.read_aovs_typed(...)["rgb"]` already carries the displayed
secondary view at its display rect, so the film composite that drew the inset itself produced
two insets once the rect moved; the composite is now a no-op, and the `rgb` hash row includes the
inset pixels. The round 8 films show the double inset; the protocol's films do not.

The protocol: `run_protocol.py <folder> --assets <dir> --film` runs the control, ten fresh
processes at seed 0 and seeds 1 to 9 from the committed script (it refuses a dirty
`crane_lift.py`), writes `protocol_meta.json` (git head, GPU, driver, every command line) and the
films from the cited runs, then `analyze.py`.

Late addition, 2026-09-07 (before the protocol): a hull IMU as a manifest row of its own, `imu`,
the motion reference unit of a real crane. `tp.Imu` needs a PhysX body in its ancestry, so it rides a
small kinematic body driven with the vessel's pose every frame; seeded MEMS noise exactly as
`sensor_audit.py` seeds it, one sample per physics step (60 Hz), hashed sample by sample; the samples
are saved as `imu` (N x 7: t, gyro xyz, accel xyz) beside the log in the npz. Checked on a 3 s run:
the gyro's roll-rate peak matches the log's (0.0144 vs 0.0140 rad/s), mean acceleration 9.79 m/s^2.


## The protocol on the frozen scene (2026-09-08, `round9_protocol/`, `round10_protocol/`)

`run_protocol.py <dir> --assets <dir> --film`: the control, ten identical processes at seed 0, an
eleventh that also writes the film, and seeds 1 to 9; 21 runs, about 2.6 h on the RTX 4070 (driver
595.97), one GPU job at a time, the GPU otherwise unshared. Round 9 ran the anti-swing gain at the
script's stale default 0.8 by omission (the runner passed none); round 10 at the chosen 0.5 from
commit 77b38778 is the cited set (`data/e3_crane/` in the paper repository is a copy).

Both rounds, eleven identical processes: eleven of the fifteen rows agree to the bit over 3,900
frames (the five AOVs, the fan, the trajectory, the vessel, the tension, the contact, the IMU).
The rendered frame took three states across the ten and the tip view two (the events follow the
frame), and in both rounds the seven processes slowed by CPU contention from the desktop shared
one state while the three unloaded ones split two to one: the paper's E1 load-case mechanism, a
state selected in the first frames, here with the GPU unshared. A twelfth process (`op_s0_film2`,
the split-screen candidate) took a fourth state. (Round 11 below found the trigger and the omitted repair.)

Round 10, seed 0: the container lands 20 mm from the mark, flat, inside the clear rectangle,
with 72 mm RMS of swing in the pay-out; the control (loop open) lands 686 mm off with 517 mm RMS.
Seeds 1 to 9: landing 16 to 42 mm from the mark; the container's path fans out from seed 0 by
8 mm RMS and 21 mm at most from arrival to touchdown (3 mm RMS, 35 mm max over the whole run).
Round 9 at gain 0.8 landed 261 mm off in every process (223 to 276 mm over the seeds).

Two controls run 20 minutes apart on the unshared GPU before the batch already showed the frame's
state behaviour (all other rows identical); a control overlapped by another session's CUDA and
Vulkan renders differed also in the label images, the fan and the tip view, which the paper's box
and fjord scenes never did under load. That case is open and outside the claim.


## The sensor film (2026-09-08 morning, `sensor_film.py`, `round10_protocol/op_s0_film3`)

The cut of the night showed the rendered frame and the tip camera inset and nothing of the fan,
the events or the IMU. `crane_lift.py --op-panels` now dumps what a compositor needs beside the
film (the events binned 4x4 per film frame, every 32x32 fan scan, the log row of each film frame,
the tip rect, the phase boundaries; 18 MB compressed, gitignored like the films), and
`sensor_film.py` draws the panels over the film on the CPU afterwards, 1920x1080 from the 1280x720
frame: the hull IMU (gyro and accelerometer), the hoist load cell with the contact switch's band,
the fan's swing estimate with the loop's engaged state, the phase marks (arrival, pay-out,
touchdown); the fan as a range image with the container's returns in orange, its centroid and the
estimate in mm; the event camera of the hero view as the classic on/off picture with the count;
and a label on the tip inset. Every number on screen is read from the run's files. A layout
change is a re-run of this script, about two minutes for the whole film, not of the GPU.

The film with the dump is a thirteenth process of the frozen scene, `op_s0_film3` (script at
f58373b4, the dump code only; the simulation and the hash rows are untouched). Compared against
the cited film run `op_s0_film` with `sensor_audit.py --compare`: the eleven sensor rows are
bit-identical (the five AOVs, the fan, the trajectory, the vessel, the tension, the contact, the
IMU), and the rendered frame, the tip view and the events differ, as between any two processes
of this scene (the state mechanism above). So the panels in the film are drawn from the cited
run's own sensor data, to the bit, over a frame that took its own state.


## Round 11 (2026-09-10 evening): the frame's states, diagnosed (`round11_diag*/`, on `dev`)

The branch was merged onto `dev` first (the engine the paper's other measurements use; the
physics log of a dev run is byte-identical to round 10's `op_s0_a` over the 270 rows compared).
Short runs, `--op 4` (30 warm-up frames, 240 captured), one fresh process each, 27 to 39 s per
run, launched by `round11_tools/run_diag.py`; the knobs are in `crane_lift.py`, all off by
default so round 10's configuration is unchanged; `diag_report.py` and `wh_compare.py` group the
manifests, `diff_frames.py` compares the frame dumps.

What round 10 saw reproduces on dev: six baseline processes split 3/3 on the rendered frame and
the tip view, on every captured frame from the first, with the five AOVs, the fan, the
trajectory, the vessel, the tension, the contact and the IMU identical (`round11_diag/base_*`).

- Not the auto-LOD: with it off the frame still splits (1 of 4); with it on, the chain timeline
  (11 chains enqueued at warm-up frame 7 after the engine's quiet window, 8 finalized two per
  frame by warm-up frame 10, 3 failed; 2 of 222 entries above level 0 at capture) is identical
  in every process (`nolod_*`, the `lod` block of every manifest).
- Where it starts: hashing the frame, the AOVs and the tip view on the first render and every
  warm-up frame (`--warmup-hash`, `round11_diag_wh/`), the first render and warm-up frame 0 are
  byte-identical between the groups; the groups part at warm-up frame 1, 61 pixels of 921,600,
  on the railings, the ladder, the hull edges and the left border at the horizon (three of them
  by more than 8 of 255, the rest by 1), and the difference then spreads through the histories
  (125 pixels a frame later, 769 at the first captured frame). Warm-up frame 1 is the fourth
  internal frame of the process: the script's first `render()` drives three internal frames
  (`set_flush_frames` defaults to 3; the script set 1 only afterwards), the tip view is added
  after it and created at internal frame 3, and the primary view parts at frame 4.
- The trigger needs both the secondary view and that three-frame first render: no tip view,
  6 of 6 identical (`round11_diag_r3/wh_notip_*`); `set_flush_frames(1)` before the first
  render, 6 of 6 (`wh_ff1_*`); both, 4 of 4.
- The passes: with any one of ReSTIR DI, ray-traced AO, probe GI or the soft sun (angular
  radius 0) off, 5 of 5 identical (`round11_diag_r4/wh_norestir_*`, `wh_noao_*`,
  `wh_noprobegi_*`, `wh_hardsun_*`); with the denoiser off the split remains (1 of 5,
  `wh_nodenoise_*`); with the G-buffer MSAA off it remains too (1 of 5,
  `round11_diag_r5/wh_msaa1_*`). The E1 box scene's load case needed ReSTIR, probe GI, the denoiser
  or RTAO each; here the denoiser is not involved and the soft sun is.
- The carrier: with 300 extra frames rendered after the warm-up without stepping the world
  (`--settle 300`), the first captured frame agrees, the difference re-emerges for about fifty
  moving frames and dies for three of four processes; the fourth stays different on every frame
  (`settle_*`), the histories being fp16 accumulators.
- The repair the audit prescribes: `renderer.reset_temporal_history()` after the warm-up
  frames, which every E1 capture does and round 10 did not. With it, 4 of 4 identical on every
  row and every frame (`reset_*`). It is on by default from this round (`--no-warmup-reset`
  reproduces round 10).

The engine-side mechanism (why the fourth frame, what the tip view's shade shares with the
primary's under ReSTIR) is not pinned; the E1 box scene's second state under GPU load starts at
frame 4 too and also needed ReSTIR, but the reset did not remove that one and it touched every
lit pixel by 1 ulp, so the two are not shown to be the same defect.

Round 12: the protocol of round 10 on the frozen script with the reset, `round12_protocol/`.
Run 2026-09-10 21:51 to 2026-09-11 00:05 on the RTX 4070 (driver 595.97), commit ded788e4, the GPU
otherwise unshared, 5.9 to 9.5 min per run: the eleven seed-0 processes (the ten and the film run,
which also wrote the panels dump) agree on all fifteen rows over 3,900 frames, and on every frame of
the rendered frame and the tip view. The trajectory is not round 10's: the fan seeds its sub-beam
jitter from the renderer's sample index (`pc.rngSeed = sampleIndex`), which the reset restarts, so the
loop follows a different sample sequence from the first scan after the capture starts (log row 38);
the vessel and IMU rows equal round 10's to the bit. Seed 0 lands 15 mm from the mark (20 mm in round
10), the control 686 mm; seeds 1 to 9 deviate from seed 0 by 3 mm RMS and 30 mm at most over the
whole run (`analyze.py`). The paper's numbers are re-derived from this directory.
