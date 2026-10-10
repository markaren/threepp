# The Skywalker X8

A fixed-wing UAV for the examples: the Skywalker X8 flying wing on the 6-DOF model that
B. Løw-Hansen, R. Hann, K. Gryte, T. A. Johansen and C. Deiler identified from flight tests,
"Modeling and identification of a small fixed-wing UAV using estimated aerodynamic angles",
CEAS Aeronautical Journal 16, 501-523 (2025), <https://doi.org/10.1007/s13272-025-00816-3>
(open access, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). The model's equations
(5)-(29) are implemented as the article writes them, with the numbers of its Tables 1 and 6-9,
unchanged. Built the way the boats are (`../usv/`): one spec file, a headless-Blender generator,
a `.glb` that threepp loads with `GLTFLoader`, and a rig any scene imports.

| file | what it is |
|---|---|
| `x8_spec.json` | every number, each with its source: the article's tables, the product page, or ASSUMED with the reason; the article's trim point, linear matrices and poles for the comparison; the 3D model's planform (fitted to the vendor's underside photo), the materials |
| `../rigs/x8_rig.py` | the aircraft for any scene: `X8` (the article's model), `Autopilot` and `LOS` (ours), `Visual` (the `.glb` posed from the model); no threepp import |
| `build_x8_blender.py` | builds the geometry (numpy) with `../build_common.py`'s mesh and export code and exports `x8.glb` through Blender |
| `x8_flight.py` | the example: the X8 flies a route (a straight leg, a turn, a leg) over a lake in a wind you set; window, stills or telemetry |
| `x8.glb` | generated, not committed |

## Build

From this folder, once (about 3 s; `blender` is your Blender executable):

```
blender --background --factory-startup --python build_x8_blender.py -- --spec x8_spec.json --out x8.glb
```

`python build_x8_blender.py --check` builds the geometry without Blender and prints the
checks: span over the winglets 2.120 m (product page 2120 mm), nose to spinner 0.795 m
(790 mm), planform area to the winglet bend 0.729 m^2 (the article's S = 0.75), elevons 0.0705
m^2 together. The build fails if the span is more than 5 mm off, the length more than 2 cm, or
any closed part is not closed and consistently wound.

## Run

```
python x8_flight.py                              # a window; C cycles the camera (chase, high, ground)
python x8_flight.py --wind 0                     # calm (default: 6 m/s from the west)
python x8_flight.py --wind 8 --from 315          # 8 m/s from the north-west
python x8_flight.py --hold heading               # hold the nose, not the track: the wind carries it off
python x8_flight.py --shot 20 --out x8.png       # headless still at t = 20 s
python x8_flight.py --stills 8,28,38 --out-dir out --cam high
python x8_flight.py --telemetry [--csv x8.csv]   # numpy only: a line a second, a summary, every step to a file
```

The route is 600 m north, a right turn, 450 m east, 60 m up at 18 m/s. Drawn: the route at
the commanded height, the waypoint poles, the aim point the guidance steers at and the line of
sight to it, the flown track, and the wind as an arrow on the ground (1 m per m/s).

## Frames

Inside the model everything is the article's: NED position (N, E, D), body axes x forward,
y right, z down, Euler angles (phi, theta, psi) with psi from north toward east. One mapping
to threepp's world, at the boundary (`x8_rig.ned_to_world`): world = (E, h0 - D, -N), x east,
y up, z south, as the drone rig uses it. Heading psi = 0 points the
nose along -z, psi = 90 deg along +x. Wind is the air's velocity over the ground, NED; the
example's `--from` is where it blows from.

The `.glb` is X forward, Y up, Z right (starboard), metres, origin at the CG (the article's
body origin), 435 mm behind the nose (the product page's CG range 430-440 mm), the chord plane
at Y = 0. A body vector (x, y, z) is (x, -z, y) in it.

- `airframe` (wing, centre body, winglets), `hatches` (three, orange), `fittings` (pitot,
  GNSS puck, camera windows, motor fairing and can): meshes.
- `elevon_left`, `elevon_right`: on the hinge lines, local +Z along the hinge toward
  starboard; a positive turn about local +Z puts the trailing edge down on either side, which
  is the article's positive deflection (eq. 14 with Clda > 0, Cmde < 0).
- `propeller`: at the disc centre, spins about local +X; positive is the model's Omega_p, a
  spin vector along body +x (clockwise seen from behind), the sense the article's rear-propeller
  torque (13) and gyroscopic term (11) imply. Two blades, 14 x 8 in.
- Empties (look down local -Z, local +Y up): `imu`, `gnss`, `pitot_tip`, `camera_nadir`,
  `camera_fpv`.

`x8_rig.Visual(root).pose(x8, h0)` places the root and turns the elevons and the propeller.
Checked on the posed meshes: +0.3 rad on an elevon moves its trailing fifth 13.2 mm down; a
pitch-up command (both -0.3) raises both 13.3 mm; a roll-right command (aileron +0.3) lowers
the left and raises the right; +0.2 rad of propeller moves the upper blade 33 mm toward body
+y; heading 90, pitch 10, roll 20 deg puts the nose along world (0.985, 0.174, 0) and the right
wing down.

## What is the article's, what is assumed, what is ours

The article's, as written (`X8.rigid_rates` numbers each step): air-relative velocity against
the wind (15), (16); lift, drag, side force and moments (17), (18) with the rates normalised by
V_a; the stability-to-body rotation (19), (20); forces and moments (21), (22); the propeller's
advance ratio, thrust and torque polynomials (5)-(9); its gyroscopic moment (10), (11) and the
rear-propeller sign (12), (13); the motor (28) and the propeller's speed (29); the inertia (24),
the rigid body (25), the position (26), the Euler-angle kinematics (27); the elevon mixing (14);
the servos (second order, 100 rad/s, 0.707, 0.07 s delay) and the throttle (first order, 0.2 s,
0.05 s delay) of Table 9. Integration is fourth-order Runge-Kutta at 1/300 s, so both delays are
whole steps (21 and 15) and 60, 50 or 30 frames a second are whole numbers of steps.

Not in the article, ASSUMED (`x8_spec.json` says why each): the air density (the ISA troposphere
at the aircraft's height above sea level: 1.225 kg/m^3 at the sea, 1.2042 at the article's
178 m), gravity 9.82 m/s^2 (what the article's matrices print as g cos(theta)), the battery
voltage 15.8 V (the middle of the flights' 15.2-16.4 V), the elevon throw +-25 deg.

OURS, kept out of the article's equations:
- Guards (`envelope` in the spec). The coefficients are linear in alpha and beta with no stall,
  so alpha and beta enter (17), (18) clamped to the range the article's flight data cover (alpha
  -10..+20 deg in its Fig. 4, beta -15..+15 in its Fig. 11): past it the coefficients hold their
  edge values, a lift plateau, not a stall; the lift and drag directions still follow the true
  alpha. V_a is floored at 1 m/s in the divisions; the advance ratio is held at J = 1.93, where
  Table 6's C_T(J) turns back up; Omega_p is held at or above zero. `X8.guards` counts each
  step a guard acted: none did on any flight below.
- `Autopilot`: successive loop closure (Beard and McLain, ch. 6): roll and pitch on the
  elevons, course or heading on roll, altitude on pitch, airspeed on throttle, each gain
  computed from the article's coefficients at the operating point and the spec's choice of
  frequency, damping and limits. The textbook's pitch PD does not fit this airframe: its own
  pitch stiffness (10 rad/s at 18 m/s) is above what the 0.07 s servo delay lets a pitch loop
  reach, and the formula then subtracts stiffness (with its gains the closed loop diverged within
  8 s). The pitch loop adds the airframe's own stiffness once over (kp = -Cma / Cmde) and an
  integral, without rate feedback. From the 18 m/s trim: +10 m of height in 4.7 s (90 %, 0.55 m
  over), a 90 deg course change in 5.2 s (5.4 deg over, the height within 0.9 m), +2 m/s of
  airspeed in 1.3 s (0.34 m/s over).
- `LOS`: line of sight on a polyline, chi_d = path angle + atan(-e / delta), the leg handed
  over a turn radius before the corner. `Autopilot.command(course=...)` is the hook a film's own
  law replaces; `Autopilot.aim_at(n, e)` is pure pursuit.

## Against the article's linearisation

`X8.jacobian` linearises the implementation numerically (central differences) at the article's
trim (36), (37). The article does not state its air density, its propeller speed or how it
took the motor out of its 9-state model, so those were read off its own matrices:

- Air density: the force rows, which involve no inertia (d(u')/d(de), d(v')/d(da),
  d(w')/d(de)), give 1.2050, 1.2030 and 1.2040: the ISA density at the trim's 178 m is 1.2042.
- Propeller speed: the ratio of the throttle column's u' and w' entries (thrust against the
  thrust coefficient in the drag) fixes Omega_p* = 558 rad/s, their size U_b = 15.92 V. At the
  printed throttle (0.44) the motor model runs at 431 rad/s; 558 rad/s needs 0.583. The article
  added a throttle offset to its flight data to match the measured power (its section 3.6, size
  not given); 0.14 would account for it. At either speed the trim point is not an equilibrium of
  the model (u' = -0.24 or +2.27 m/s^2): it was taken from a flight manoeuvre (its Fig. 10).
- The motor: the article's A matches the propeller speed held at Omega* (d(u')/du = -0.588,
  printed -0.59; with the motor at steady state -0.559), its throttle column the motor at steady
  state (held, it would be zero).

With those, each of the 171 printed entries is compared against a band: half a unit of its last
printed digit plus the change the rounding of every printed input (Tables 1, 6, 7, 8, the trim)
can make. 13 entries fall outside it, with four causes:

| entries | article | this model | cause |
|---|---|---|---|
| A[q',r], A[r',q], A[p',q] | 0, 0, 0 | -1.379, +0.486, +0.043 | the gyroscopic moment (11), I_p Omega_p = 0.19 N m s, is in the model and not in the article's linearisation |
| A_lat[phi',r] | 0 | -0.019 | the article linearises (27) as the identity; exactly, d(phi')/dr = cos(phi) tan(theta) |
| the roll row: A[p',v], A[p',p], B[p',da], B[p',vw] (in A_lat/B_lat and A/B), B[p',dt] | -5.45, -16.81, 92.93, 5.45, -5.76 | -5.592, -17.318, 95.557, 5.592, -6.004 | every entry of the roll row is 2.5-2.9 % larger here, the yaw and pitch rows are not: the article's matrices match Ix = 0.335 against its Table 1's 0.325 |

With the gyroscopic terms out, the kinematics as the identity and Ix = 0.335, no entry is
outside the band. The pitch row is inside it as printed (cbar 0.36 +- 0.005), and is exact with
cbar = S / b = 0.3571: d(q')/d(de), d(q')/dw and d(q')/dq then give 1.2040, 1.2028 and 1.2037 for
the density. The model keeps Table 1's numbers; none of this is fitted into it.

Density sensitivity: entries outside the band, as implemented / with the four causes removed:
13 / 0 at 1.2042, 25 / 13 at 1.225, 7 / 2 at 1.19. The entries scale with rho, 1.7 % between
the ISA values at 178 m and at the sea.

Modes (rad/s, damping) of the decoupled 4 x 4 blocks, against Tables 10 and 11:

| mode | article | this model | with Ix = 0.335 |
|---|---|---|---|
| roll | -17.50 | -17.99 | -17.48 |
| dutch roll | 3.83, 0.22 | 3.84, 0.22 | 3.84, 0.22 |
| spiral | -0.42 | -0.41 | -0.40 |
| short period | 7.92, 0.78 | 7.95, 0.79 | 7.92, 0.78 |
| phugoid | 0.99, 0.22 | 1.00, 0.22 | 1.00, 0.22 |

The article's printed A_lat itself gives a spiral pole of -0.40 (Table 10 lists -0.42). In the
coupled 8-state model the gyroscopic terms move the short period to 8.05 rad/s and the dutch
roll to 3.94 rad/s, 0.20 (without them: 7.93 and 3.82, 0.22; the article's coupled A: 7.90 and
3.80, 0.22).

## Trim, wind, actuators, the model's range

Straight and level at 18 m/s in still air at the sea (`X8.trim`): alpha 7.48 deg, beta +0.89,
bank -0.67 (Cl0, Cn0, CY0 and the propeller's torque need a little aileron, bank and sideslip),
elevator -1.82 deg, aileron -2.35, throttle 0.547, Omega_p 526 rad/s (J 0.605); thrust 7.38 N
along the nose at 7.47 deg of pitch, 7.32 N of it horizontal against 7.33 N of drag; lift 32.06 N
plus the thrust's vertical 0.96 N against 33.03 N of weight; motor 8.64 V and 22.5 A (194 W), battery 12.3 A, L/D 4.4 (the article's CD0
is 0.058).

Across speeds, still air: alpha 19.6 deg at 12 m/s (the edge of the data), 13.9 at 14, 10.1 at
16, 5.6 at 20, 4.2 at 22, 2.6 at 25; throttle 0.41 to 0.76; power 93 W at 12 m/s, 258 W at 20,
492 W at 25. The model was identified at 18 m/s: believe it from about 14 to 22 m/s. Below
13 m/s the trim needs alpha beyond the flight data and the real aircraft stalls, which this model
cannot (at 10 m/s it finds a trim at alpha 73 deg on the guards' lift plateau); above 22 m/s alpha
falls toward the sparsely sampled negative range, where the article reports its fit worst. Rates
far beyond its manoeuvres (p up to about 150 deg/s in its Fig. 11), sideslip past 15 deg and the
motor at full throttle (no current limit: 51 A at 28 m/s) are outside it too.

Wind, course hold due north at 18 m/s (`Autopilot` holding the ground track): the air velocity
points 9.59, 19.47 and 30.00 deg off the track in a 3, 6 and 9 m/s crosswind, asin(W / V_a) to
0.01 deg; the nose adds the airframe's own 0.9-1.0 deg of sideslip. On the example's route, the
settled first leg's cross-track error (rms):

| wind | course hold | heading hold |
|---|---|---|
| calm | 0.00 m | 1.03 m (the still-air sideslip) |
| 6 m/s from the west (across leg 1) | 0.00 m | 22.2 m (delta tan(crab) = 21.2) |
| 8 m/s from the north-west (5.66 m/s across both legs) | 0.00 m | 21.0 m |

The turn, with the hand-over a turn radius before the corner: 0 m past the corner in calm air,
17.3 m with 6 m/s from the west (the ground speed rises from 17 to 24 m/s through it), bank up
to 31 deg, height 59.1-60.7 m, airspeed 17.7-18.2 m/s throughout.

Actuators, stepped from the trim: the elevon first moves one step after its 0.07 s, overshoots
4.30 % at 0.1133 s (second order at 100 rad/s and 0.707: 4.33 % at 0.1144 s; damping read back
0.7077), and differs from the analytic response by at most 0.008 % of the step; the throttle
first moves after 0.05 s and differs from 1 - exp(-t / 0.2) by less than 0.001 %.

The integration: two runs give bit-identical states at every step. A torque-free tumble (no
air, no gravity) keeps its energy to 3e-10 and its angular momentum to 3e-9 over 600 s while
|theta| stays under 55 deg; through theta = 90 deg the Euler angles of (27) are singular and the
momentum drifts 0.5 %. Ten minutes round a 600 m square in a 6 m/s wind: the energy over the
ground stays within 2211..2980 J (the ground speed swinging from 12 to 24 m/s with the wind),
and its change (+380.101 J) equals the work of the aerodynamic and propeller forces and moments
(+380.100 J).

## In another scene

`../rigs/x8_rig.py` imports no renderer and reads no scene's globals:

```python
from x8_rig import X8, Autopilot, LOS, Visual, model_path

x8 = X8(wind=(0.0, -6.0, 0.0), h0=0.0)        # NED wind (here 6 m/s from the east); h0: the NED origin's height
x8.trim(18.0, n=0.0, e=0.0, h=60.0, course=0.0)   # level at 18 m/s, crabbed onto a course north
ap = Autopilot(x8)
los = LOS([(0.0, 0.0), (400.0, 0.0), (400.0, 300.0)])
visual = Visual(tp.GLTFLoader().load(model_path()).scene)
# each step (x8.dt = 1/300 s):
vg = x8.ground_velocity()
ap.command(course=los.update(x8.n, x8.e, math.hypot(vg[0], vg[1])), altitude=60.0, airspeed=18.0)
ap.update()
x8.step()
# each frame:
visual.pose(x8, h0=0.0)
```

`los.aim` is the aim point, `los.e` the cross-track error; `x8.out` holds the last evaluated air
data, forces and coefficients; `x8.world_position(h0)` and `x8.world_rotation()` give the
aircraft in the world for a camera.

# The Matrice 350 RTK

A multirotor for the examples: the DJI Matrice 350 RTK carrying a Zenmuse H20T, as a rigid body on four
rotors in wind. Unlike the X8 above it has no published flight model behind it: DJI publishes dimensions,
masses and limits, and no thrust, torque, drag or inertia. So the model here is a standard multirotor model
with the product page's numbers where there are any, every other number ASSUMED and marked so in
`m350_spec.json`, and two of them CALIBRATED against the two flight figures the product page does give.
Nothing is validated against a flight log.

| file | what it is |
|---|---|
| `m350_spec.json` | every number with its source: DJI's specification pages (read 2026-10-09), or ASSUMED with the reason; the layout, the mass budget, the propulsion, the drag, the controller's gains, the 3D model's parts and materials |
| `../rigs/m350_rig.py` | the aircraft for any scene: `M350` (the rigid body, the rotors, the skids on the ground), `Wind` (a mean wind and Dryden turbulence), `Autopilot`, `Mission` and `Gimbal` (ours, not DJI's), `Visual` (the `.glb` posed from the model); no threepp import |
| `build_m350_blender.py` | builds the geometry (numpy) with `../build_common.py` and exports `m350.glb` through Blender |
| `m350_flight.py` | the checks and a short flight, on the command line; numpy only, no window |
| `m350.glb` | generated, not committed |

```
blender --background --factory-startup --python build_m350_blender.py -- --spec m350_spec.json --out m350.glb
python build_m350_blender.py --check             # the geometry's gates and projected areas, without Blender
python m350_flight.py --checks                   # the eight checks below
python m350_flight.py --telemetry --wind 8 --from 250 [--csv m350.csv]
```

## The model

The product's: 895 mm between diagonal motor axes, 670 mm wide over the motors, 430 mm tall, 21 in
propellers, 3.77 kg without and 6.47 kg with its two batteries, the H20T's 0.828 kg and its size. The
model meets those; it does not meet the product page's length of 810 mm (it is 745 mm: with the rotor axes
the wheelbase and the width give, nothing on the aircraft reaches 810). The motors and propellers hang
UNDER the arm ends and the body stands above the arm plane, as on the aircraft (a first build had them on
top; the owner's photograph of the aircraft corrected it). Shapes are fitted by eye to DJI's photographs
and that photograph; what could not be seen in them is listed in the spec (`model.unverified`: the side,
top and bottom vision windows, the lamps' places, the gimbal's yoke, which window of the camera head is
which). No logos or lettering. Nodes: `airframe`, `prop_fr/fl/rl/rr` (each at its disc centre, spun about
local +Y), `gimbal_pan` > `gimbal_roll` > `gimbal_tilt` (about +Y, +X, +Z), lamps with their own emissive
materials, empties for the cameras and antennas.

## The flight model

`M350.rates` has the equations (its docstring writes them out). A rotor's thrust is kT w^2 less what the
air's speed through the disc takes (blade element and momentum theory), its in-plane force is linear in
the air's speed across the disc, its torque kQ w^2 with the rotor's angular momentum in the rigid-body
equation; the airframe's drag is its projected areas (measured on the model's own meshes) times an ASSUMED
drag coefficient of 1; the rotors follow their commands with a first-order lag. Mass, centre of gravity
and inertia come from a budget of boxes (batteries and payload at the product's masses; the split of the
rest ASSUMED). Fourth-order Runge-Kutta at 1/300 s; two runs are bit-identical.

CALIBRATED, which is not validated: the figure of merit and the drive efficiency (0.72, 0.88, both
ASSUMED) are chosen together so the hover at 6.47 kg draws the 574 W that the product's 55 min from 526 Wh
implies; the rotors' in-plane force coefficient is solved so 23 m/s needs the product's 30 deg of tilt
(with the drag coefficient at 1 the airframe alone is 96 % of the drag there: that coefficient is nearly
calibrated away). The maximum thrust (64 N a rotor) is ASSUMED, set so that point is inside the rotors'
range. Battery power away from the hover is not to be quoted: the torque is the static one and there is no
translational lift, so power only rises with speed.

`Wind`: a logarithmic mean profile (or a scene's own field: `field(n, e, h)`), and on it the low-altitude
Dryden turbulence of MIL-F-8785C / MIL-HDBK-1797 as shaping filters on seeded noise. OURS: a hovering
aircraft has no airspeed to carry it through frozen turbulence, so the turbulence is convected past it at
the speed of the mean air past the aircraft. It is a point model: all four rotors see the same gust.

`Autopilot` (ours, not DJI's): position, velocity, thrust vector with the product's 25 deg tilt limit,
quaternion attitude, body rates, mixer. It does not know the wind and rejects it by feedback. It is fed
the EXACT state: no GNSS, no vision, no estimator, no sensor noise, so what it holds is a floor under what
a real aircraft holds. `Mission` flies straight legs on jerk-limited profiles, takes off and lands;
`Gimbal` holds the camera on a point within the H20T's published ranges (its slew rate ASSUMED).

## The checks (`m350_flight.py --checks`, with the payload unless said)

- 7.298 kg; hover 260 rad/s (2490 rpm), 17.9 N a rotor, 681 W, 46 min. Without the payload 574 W and
  55.0 min: the calibration above, not a result.
- Holding position in a steady 8 m/s: 3.7 deg of tilt nose into it, 5.0 deg beam on.
- Steps from a hover in still air: 5 m sideways in 1.2 s (10 to 90 %) with 3.2 % overshoot, 2 m up in
  1.3 s with none, 90 deg of yaw in 1.1 s with 2.2 %; tilt and rates inside the limits.
- Hovering 30 m up for 120 s in turbulence at the standard's intensities: 0.02 m rms sideways in 4 m/s,
  0.06 m in 8 m/s (0.21 m at the most), 0.44 m in 12 m/s (2.0 m at the most: 12 m/s 10 m up is 15 m/s at
  30 m, and the tilt limit, not the thrust, bounds it). The product states +-0.1 m with RTK and no wind
  for that figure.
- A 5 m/s gust over 2 s on the beam: pushed 0.09 m, back inside 0.02 m after 3.7 s.
- Take-off to 10 m, a hold, a landing: touchdown at 0.24 m/s, no bounce; at rest on a 5 deg slope with
  the rotors at idle.
- A torque-free tumble keeps energy and angular momentum to 1e-10; in a still-air flight the change in
  energy equals the work of the non-conservative forces to 0.001 %.
- 45 to 60 times real time, autopilot and wind included.

Not in the model: ground effect, vortex ring state in a descent, blade flapping, the rotors' wash on the
airframe, any sensor.

`python/examples/trollstigen/trollstigen_inspect.py` (a site project outside this repository) flies it on
a bridge inspection in a terrain-shaped wind.

## Acoustic localization (`m350_acoustic.py`)

Audio as a SENSOR: nine microphones in a field record the M350's four rotors, and from those recordings
alone a standard localizer says where the drone is, every eighth of a second. numpy only (`np.fft`), no
window; matplotlib (Agg) for the figure; outputs under `out/acoustic/` (not committed).

```
python m350_acoustic.py --checks                  # the three checks below
python m350_acoustic.py --flight A                # a flight: a table, .npz, .csv and a figure
python m350_acoustic.py --flight B --snr 10 [--ground-reflection] [--window 0.5] [--hop 0.125] [--seed N]
python m350_acoustic.py --report                  # the table below, also out/acoustic/report_vNN.txt
python m350_acoustic.py --flight B --snr 35 --ambient pink    # the film's run: pink noise, 15 dB quieter
python m350_acoustic.py --shot 18 --flight B [--out shot.png] # one frame of the film (35 dB pink by default)
python m350_acoustic.py --film --flight B                     # the film: out/acoustic/m350_acoustic_vNN.mp4
```

The array: eight microphones on a ring of 15 m radius about the origin, 1.5 m above the ground, every
45 deg with a seeded +-1 m jitter, and one on a 6 m mast at the centre. 16 kHz, c = 343 m/s. Two flights
from a hover, logged every step (the four rotor speeds and the four hubs from `M350.rotor_hubs_ned()`):
A, the TUNING flight, 25 m up from the south-west straight over the array at 5 m/s, 2 s hold, back and up to
35 m, in 4 m/s of Dryden wind; B, the HELD-OUT flight, 40 m up passing BESIDE the array (never over it) at
6 m/s, then down to 30 m, in 8 m/s. Every knob was set on A and B was run once with them frozen.

The pipeline: per rotor a source signal on the 300 Hz log of its speed; to every microphone with retarded
time (an emission at t_e arrives at t_e + r(t_e) / c, which is where the Doppler comes from) and 1/r; the
microphones' own seeded noise at an SNR relative to the four rotors at 50 m; then per window (0.5 s, every 0.125 s) GCC-PHAT on
the 36 pairs (Hann window, 100 to 4000 Hz, upsampled x4) and SRP-PHAT over north and east -100..100 m and
2..80 m up on a 2 m grid. A cell's max-filtered GCC (over the lags the cell spans) bounds its SRP from
above and the sharp SRP at its centre bounds the maximum from below: branch and bound down to 3 cm. (A
single refine box about the coarse grid's argmax does not work: the coarse map is a plateau many metres
long along the range.) The truth a window is scored against is the centroid of the four hubs at the time
the sound left it: the window's centre less the estimate's distance to the array over c.

Two nulls, scored on the same windows against the same truth: Null 1 knows nothing (the array's centre,
30 m up); Null 2 knows the levels and not the timing (the microphones' centroid weighted by each one's
energy in the window, 30 m up).

What is real, what is standard, what is ASSUMED:

- REAL (the rig's): the flight dynamics, the four rotor speeds at every step, the hub positions, the
  wind, the autopilot.
- STANDARD (textbook): spherical spreading 1/r, retarded time, GCC-PHAT, SRP-PHAT.
- ASSUMED, every number of it: the rotor's sound. A harmonic series at the blade-pass frequency (2 blades:
  83 Hz at the 260 rad/s hover) with 12 harmonics falling as 1/k, a broadband part (white noise through a
  one-pole low-pass at 3 kHz) whose level rises as the rotor speed to the 2.5, at half the tonal RMS at
  hover (`BROADBAND` 0.5), and the microphones' noise. No recording of an M350 was used. Units are
  arbitrary: the SNR at the 50 m reference distance is the only level that means anything. The demo is
  about GEOMETRY and TIMING; the timbre is a placeholder.
- Not modelled: air absorption (ranges stay under 150 m), rotor directivity, wind noise on the
  microphones, occlusion. The ground reflection (an image source per rotor, coefficient 0.6, ASSUMED) is a
  flag and one row of the table.

`--report` (the timbre and the units ASSUMED, see above; 0.5 s windows every 0.125 s; B scored once):

| flight | N windows | median 3D (m) | p90 3D (m) | horizontal median (m) | Null 1 median (m) | Null 2 median (m) |
|---|---|---|---|---|---|---|
| A 20 dB (tuning) | 470 | 0.36 | 0.80 | 0.36 | 51.2 | 48.1 |
| B 20 dB | 392 | 0.44 | 0.95 | 0.41 | 56.8 | 54.2 |
| B 10 dB | 392 | 0.45 | 1.04 | 0.43 | 56.8 | 54.5 |
| B 20 dB + ground reflection | 392 | 0.45 | 1.00 | 0.43 | 56.8 | 54.2 |

On the held-out flight at 20 dB the median 3D error is 0.44 m, and the nulls are 130 and 124 times worse:
the timing does the work, not the levels (Null 2 is hardly better than knowing nothing, because a 15 m ring
hears a drone 50 m away at nearly the same level everywhere).

The floor under the median is the source's size, not the array: the estimate lands on a rotor (0.16 m from
the nearest hub at the median on A) and the truth is the four hubs' centroid, 0.45 m from each hub. On A
the height is off by 0.07 m at the median; a non-planar array (every other ring microphone 8 m up, a 12 m
mast) and a 25 m ring gave the same 0.45 m and 0.49 m on A, so the plan's array stayed. The window was
the one knob that moved on A: 0.125 s gave 0.47 m (p90 1.15 m), 0.25 s 0.45 m (p90 0.80 m), 0.5 s 0.36 m
(p90 0.80 m); no estimate jumped by the ~4 m of path one blade-pass period is, so `BROADBAND` stayed 0.5.

The checks: (1) a static source at (20, -10, 25 up) m, one window at 30 dB, found within 0.3 m (Null 1 is
23 m off); (2) two runs of flight A are bit-identical in the signals and the estimates; (3) Doppler at the
mast: one rotor's blade-pass line 3 s before the overhead pass over 3 s after, each over the rotor's own
blade-pass at the emission, against (c + v_r) / (c - v_r) within 20 % of the shift. It is one rotor because
the four together do not resolve: in forward flight the front pair turns near 248 rad/s and the rear pair
near 270, lines 1.5 Hz apart, as far as the Doppler shift.

The film (`--film`, headless, 1920 x 1080 at 30 fps, all of flight B): on the left a GL view of the field
from the south-west, its camera fitted to the flight and the array, with a close-up from 9 m at 1:1 scale;
the nine microphones, the truth track, the estimate with its stalk to the ground, a trail of the last ten
and a line to the truth. The aircraft is drawn where it was when the sound now at the microphones left it.
On the right, per window, the SRP-PHAT over north and east at the estimated height (a 2 m grid of cell
upper bounds, for the picture only) and the 3D error so far against both nulls. The sound is stereo: the
ring's westmost microphone left, its eastmost right, together to -3 dBFS (`flightB_<tag>_stereo.wav`).
`--ambient pink` shapes the same seeded noise by 1/sqrt(f) at the same RMS; the film runs at 35 dB pink
because at 20 dB white the listener hears a cheap microphone's hiss, while a quiet field's floor is
low-frequency and 15 dB lower. The localizer does not care (B at 35 dB pink: median 0.45 m, p90 1.02 m,
nulls 56.8 m and 54.2 m).

### Next step: other sound sources (TODO)

As built, the demo localizes the loudest broadband thing in the field; nothing in it knows what a drone
is. Measured on 2026-10-10 (a scratch script on flight B at 20 dB, the demo's own `propagate` and
`Localizer`): one STATIC broadband source (the low-passed noise plus a 50 Hz hum with harmonics, a generator
or a tractor) on the ground 50 m north-west of the array, at levels relative to the four rotors at the
same 50 m. "On" means within 3 m.

| interferer | on the drone | on the interferer | median error while on the drone | confidence (peak / mean) |
|---|---|---|---|---|
| none | 100 % | 0 % | 0.44 m | 5.1 |
| broadband, 10 dB quieter | 85 % | 15 % | 0.45 m | 4.2 |
| broadband, equal | 0 % | 100 % | - | 9.2 |
| broadband, 10 dB louder | 0 % | 100 % | - | 16.8 |
| 50 Hz hum, 10 dB louder | 100 % | 0 % | 0.44 m | 4.9 |

Three readings. PHAT weighs every bin alike, so the map's peaks are as tall as the share of bins each
source owns: a broadband source of equal level takes the argmax in every window, a loud hum owns a handful
of bins and changes nothing. What counts is the level at the microphones, not at the source: the quieter
generator wins the windows where the drone is farthest. And the confidence RISES when the generator
captures the estimate (a static source on the ground is a cleaner peak than a moving drone), so it is not
the flag; the estimate's own behaviour is (it stops moving and sits at 2 m).

The plan, in order, each step with its gate; every knob still set on A and B run once with it frozen,
every new number a row of `--report`, the timbre still ASSUMED until step 7:

- [ ] **1. `--interferer`** (repeatable: `kind,n,e,h,level_db`): static sources through the same
  `propagate`. Kinds: `broad` (the generator above), `hum` (narrowband), `diffuse` (wind, surf: coloured
  noise independent per microphone, no arrival direction). `--report --interferers` reproduces the table
  above from the script itself. Gate: the table above within a few percent.
- [ ] **2. Peaks, not an argmax.** `Localizer.locate` returns the top K peaks of the map (non-maximum
  suppression over about 5 m) with their heights; the panel of the film draws them. Gate: with the equal
  broadband generator the drone's peak is among the top two in at least 90 % of windows.
- [ ] **3. A tracker.** Constant-velocity Kalman (or alpha-beta) in NED over the peaks with gated
  association; a track starts when a peak persists three windows; the drone is the track that MOVES (over
  1 m/s across 2 s) and is off the ground (over 5 m up); a static track is labelled "ground source" and
  shown as such. Gate: equal broadband generator, at least 90 % of windows on the drone, median error under
  1 m; the nulls unchanged.
- [ ] **4. A drone signature on the bins.** Weight each PHAT bin by how drone-like it is: the blade-pass
  fundamental found per window on the mast microphone (a harmonic sum over 60 to 120 Hz) and its multiples
  within a few Hz get weight 1, the rest 0.1. Gate: the 10 dB louder generator, at least 80 % on the drone
  with the weighting alone (no tracker); both the weighted and the tracked rows in the table.
- [ ] **5. Two drones.** Flight A's log, shifted, as a second moving source; two tracks. Gate: both within
  1 m median, no identity swap across the flight.
- [ ] **6. A building between.** Bind the C++ `AcousticScene.transmission` (BVH occlusion,
  `threepp/audio/Acoustics.hpp`) and apply it per rotor-microphone path; measure the smearing. After 1 to 5.
- [ ] **7. The real timbre.** A bench recording of the lab's own M350 replaces `sources()`; every row above
  re-run. Until then every number here carries "timbre ASSUMED".
- [ ] **8. Film v02.** The generator in the field as an object, its peak and the drone's both on the panel,
  the tracks labelled, the generator audible in the stereo track.

# The Babyshark 260 VTOL

A hybrid for the examples: the Foxtech Babyshark 260 VTOL, a 2.5 m quadplane (four lift rotors on two
booms, a pusher propeller, an inverted V-tail), on the flight model that B. P. Graesdal identified from
flight tests: "Full Nonlinear System Identification for a Vertical-Takeoff-and-Landing Unmanned Aerial
Vehicle", Master's thesis, NTNU, 2021, <https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2981320>, with
the model itself at <https://github.com/bernhardpg/babyshark_vtol_model>. Between the X8 above (a wing with
a published model) and the Matrice (a multirotor with none), this one has a published model for the wing
and none for the hover or the way between them: so the wing is the thesis's, and the rest is built around
it and marked for what it is.

Neither of the thesis's repositories carries a licence. Its README asks that the thesis be cited and says
every equation and parameter is there to be ported, so the equations and numbers are taken and cited; no
code is copied, and its 3D model is not used: `babyshark.glb` is built here from the author's own
measurements of the lifting surfaces (his vortex-lattice input file) and the thesis's figures.

| file | what it is |
|---|---|
| `babyshark_spec.json` | every number with its source: the thesis's Tables 6.2 to 6.5 unchanged, its linear matrices and modes for the comparison, Foxtech's figures, and what is ASSUMED or OURS with the reason (propulsion, the range outside the data, the ground, both autopilots, the transition); the layout and the 3D model's parts |
| `../rigs/babyshark_rig.py` | the aircraft for any scene: `Babyshark` (the thesis's model and what a hover needs), `Autopilot`, `Mission` and `Pilot` (ours), `Visual` (the `.glb` posed from the model); it uses `m350_rig.Wind` and `x8_rig.LOS` as they are; no threepp import |
| `build_babyshark_blender.py` | builds the geometry (numpy) with `../build_common.py` and exports `babyshark.glb` through Blender |
| `babyshark_flight.py` | the checks, a flight's telemetry, and stills or a window of that flight |
| `babyshark_fly.py` | the aircraft in a window, flown with the keyboard, the camera on the mouse |
| `babyshark.glb` | generated, not committed |

```
blender --background --factory-startup --python build_babyshark_blender.py -- --spec babyshark_spec.json --out babyshark.glb
python build_babyshark_blender.py --check          # the geometry's gates, without Blender
python babyshark_flight.py --checks                # the ten checks below
python babyshark_flight.py --telemetry --wind 8 --from 250 [--csv babyshark.csv]
python babyshark_flight.py --stills 12,26,45,150 --out-dir out/babyshark [--cam chase|side|high]
python babyshark_flight.py --window                # the same flight in a window; C cycles the camera
python babyshark_fly.py [--wind 4 --from 250]      # fly it yourself (below)
```

The flight is a take-off to 60 m, a front transition flown level, a circuit of 800 by 250 m there at
20 m/s laid out so its last leg is into the wind, a back transition 140 m short of the pad, and a landing
on it.

## The 3D model

`babyshark.glb`: X forward, Y up, Z right, metres, origin at the centre of gravity (the thesis's body
origin). The wing's six sections, the tail's panels and the rotor axes are met exactly (the author's
vortex-lattice file and the thesis's Table 4.3); the pod, the moulded booms, the motors and the landing
gear are fitted by eye to the thesis's renders of the author's CAD model (Figs. 3-1, 3-2, 4-1), scaled by
the rotor axes. 40 480 triangles. `--check` fails the build unless: the span is 2.500 m, each lift rotor is
on its axis, the tip circles are 0.4064 and 0.381 m, the discs clear the wing, the tail, each other and
the ground (the pusher's tip passes 25 mm above the feet's plane), the tail's ends and the feet are on the
layout, and every part is closed and wound outward. It prints: wing planform 0.575 m^2; wing and tail
0.715 m^2 projected, 0.659 with what the pod covers left out (the thesis's S is 0.6617); length 1.46 m,
height 0.61 m; and the projected areas the flight model's bluff body uses (front 0.130, side 0.223,
top 0.894 m^2).

- `airframe`: pod, wing, tail, booms, motors, gear, fittings.
- `aileron_left`, `aileron_right`, `ruddervator_left`, `ruddervator_right`: on their hinge lines, local +Z
  along the hinge toward starboard; a positive turn puts an aileron's trailing edge down and a
  ruddervator's toward its panel's lower, inner face.
- `lift_fr`, `lift_fl`, `lift_rl`, `lift_rr`: at their disc centres, spun about local +Y, the blades handed
  by the spin sense. `pusher`: spun about local +X.
- Empties: `imu`, `gnss`, `pitot_tip`, `camera_nadir`, `camera_fpv`.

Checked on the posed model through `babyshark_rig.Visual`: a positive aileron (20 deg) lowers the left
trailing edge 17 mm and raises the right one; a positive elevator moves both ruddervators' trailing edges
down and inward; a positive rudder moves both to port; rotors 1 and 2 (front right, rear left) turn
counter-clockwise seen from above; the pusher turns clockwise seen from behind; heading 90, pitch 10,
roll 20 deg puts the nose along world (0.985, 0.174, 0) and the right wing down.

Not seen in the thesis's renders and kept plain (the spec's `model.unverified`): the wing's section (a
stand-in for the Eppler 397), how the wing blends into the pod, the motors' and propellers' shapes, the
pitot tube and the GNSS puck, hatches and seams. Foxtech's product photographs were not consulted. No
logos or lettering.

## What is the thesis's, what is assumed, what is ours

The thesis's, as written (`Babyshark.rates` numbers each step): the rigid body (6.6) with the Gamma
constants of Table 6.2; lift, drag, side force and moments (6.7)-(6.10), the rates made non-dimensional by
the TRIM airspeed (6.11) and the surfaces measured from their trim (6.13); the lift rotors' thrusts,
torques and moments (6.14)-(6.16); the pusher's static thrust (6.15a); the servos (3.41). It was identified
in fixed-wing flight around 21 m/s with the lift rotors stopped, in air taken as still; its data reach
alpha -13 to +16 deg and its propellers were measured with no incoming air. The thesis says itself what
that leaves out: the stall, any flight on the lift rotors, the transition, wind, and the propellers' loss
of thrust with airspeed.

OURS or ASSUMED, each in the spec with its reason, none of it fitted to a flight:

- Wind: the air-relative velocity stands where the thesis has the body velocity in the airspeed, alpha
  and beta (6.12). The air density follows the height (ISA; 1.225 at the sea, the thesis's constant).
- Outside the identified alpha and beta (`envelope`): the coefficients are held at the edge and the forces
  and moments go over, across 10 deg, to a bluff body's (the airframe's projected areas, each at its centre
  of pressure). That is the stall, and what the hover and both transitions fly in. `Babyshark.guards`
  counts the steps spent there: none on the wing.
- The pusher: T = rho D^4 cT n^2 (1 - J / J0), the thesis's law at J = 0, with J0 = 0.80 from the
  propeller's pitch. `Babyshark(thesis_propeller=True)` is the thesis's static law at every speed.
- A lift rotor's thrust loses kz n v_ax to the air through its disc and has a small in-plane force (the
  Matrice's rotor model): in a still hover it is the thesis's law. This is what damps the hover.
- Motors: first-order lags; the pusher's limit is the top of the thesis's thrust-stand run (142 rev/s),
  the lift rotors' a thrust-to-weight ratio of 1.8. Two battery packs, read off the thesis's mass table.
- The landing gear on a ground you hand in (the Matrice's contact model).

Not in the model: the propellers' gyroscopic moment and the pusher's torque (the thesis neglects both),
translational lift and what the lift rotors and the wing do to each other in the transition (a
wind-tunnel study of a quadplane found more drag and less thrust there: Mathur and Atkins,
arXiv:2301.12316), ground effect, any sensor.

`Autopilot` (ours, not PX4, which the thesis's aircraft flew) is two autopilots and the way between them:
the Matrice's cascade on the lift rotors for the hover, with a weathervane that turns the nose into the
wind the velocity loop's integral has learned (the rotors' drag torque cannot hold the tail across a wind);
the X8's successive loop closure for the wing, on three surfaces, its gains computed from the thesis's
coefficients; and a transition laid out as PX4's standard-VTOL one. Front: the pusher ramps up, the lift
rotors hold the height it has and, with a weight that fades between 12 and 18 m/s, the attitude, while
the surfaces fly the same attitude throughout (a mission's cruise altitude is the wing's to reach
afterwards, at the wing's own climb rate). Back: the pusher stops, the lift rotors take over at once and
brake, with the nose kept a degree under the angle of attack at which the wing would carry the whole
weight, so the braking does not balloon it. It is fed the EXACT state: no estimator, no sensor noise.

## Flying it yourself (`babyshark_fly.py`)

```
python babyshark_fly.py                            # on the pad, nose into 4 m/s from 250 deg
python babyshark_fly.py --wind 0                   # still air
python babyshark_fly.py --demo                     # the keys press themselves: a hand flight to watch
python babyshark_fly.py --demo --stills 26,68,160 --out-dir out/babyshark_fly     # the same without a window
```

| key | hovering | on the wing |
|---|---|---|
| `W` `S` (or up, down) | forward (8 m/s), back (4 m/s) | the airspeed setpoint, 18 to 22 m/s |
| `A` `D` (or left, right) | turn, 30 deg/s | bank, 30 deg; let go and it levels and holds the course it has |
| `Q` `E` | sideways (4 m/s) | |
| `SPACE` `SHIFT` | up (3 m/s), down (2 m/s; 0.5 m/s near the ground) | the height setpoint, up at what the pusher has to give (0.6 m/s at 21 m/s), down at 2.5 m/s |
| `T` | to the wing | to the hover |
| `M`, `H` | the autopilot flies the circuit and lands on the pad; flies home and lands | the same |
| `V`, `C`, `R` | the weathervane on and off; the camera; back on the pad | |

On the ground `SPACE` starts the rotors and lifts off, and `SHIFT` held on the ground stops them. Any of
the eight stick keys takes the aircraft back from the autopilot. The mouse looks around the aircraft
(drag) and zooms (wheel); the camera is a chase that turns with the nose, the same without the turning,
or a spectator's 60 m beside the pad. The panel down the left edge has what it is doing, the numbers
against their setpoints, the motors and the batteries, the keys (lit while held) and a chart.

The keys go through `babyshark_rig.Pilot`: four sticks that move the autopilot's setpoints, the way a
VTOL's assisted modes take a pilot's, the same four hovering and on the wing; the transitions stay the
autopilot's. Sticks at rest, the hover brakes and holds its place and its heading, and after five seconds
of standing still the weathervane has the nose (not sooner: after a turn at speed the velocity loop's
integral still leans the aircraft, and the weathervane would follow that). On the wing the airspeed
setpoint stops at the transition's 18 m/s and the bank the stick gets shrinks to nothing between there
and 16.5 m/s: a level turn banked 30 deg needs 16.2 m/s of this model, and the two loops (height on the
elevator, airspeed on the pusher) do not look after each other at that edge. Flown onto the ground on the
wing, every motor stops and `R` is what is left. All of it is ours and ASSUMED (`control.pilot`), and
the two things that will be noticed are the model's: the wing climbs slowly (slower climbs better), and
a hover cannot hold its tail across a fresh wind.

The window itself was not opened for testing here: `--demo --stills` plays the same keys through the same
frame, camera and panel on a hidden canvas, and check 10 flies the sticks without a renderer.

## The checks (`babyshark_flight.py --checks`)

1. 12.14 kg. The eight Gamma constants from Table 6.2's four inertias agree with the numbers the thesis's
   code carries to 6e-5. Hover: 100.1 rev/s on the front rotors (33.3 N each) and 88.9 on the rear (26.3 N),
   since the centre of gravity is nearer the front axes; 1848 W at the shafts by the thesis's torque
   coefficient (figure of merit 0.62), 8.0 min on the lift battery as ASSUMED.
2. Against the thesis's printed linearisation (6.17), (6.18), at its trim with its static propeller law:
   45 of 48 entries agree to the printed digits. The three that do not are the derivatives the drag's change
   with alpha enters (u' by u, u' by w, w' by w); with the drag held at cD0 in the linearisation, 47 of 48
   agree, and the last is 0.689 against 0.686. The thesis's matrices are its model with the drag's alpha
   terms left out; the model here keeps (6.8d) whole. Modes against Table 6.7: short period -3.29 +- 7.77i
   (-3.28 +- 7.79i), roll -8.82 (-8.82), dutch roll -0.942 +- 4.940i (the same), spiral +0.116 (the same),
   phugoid -0.077 +- 0.657i (-0.067 +- 0.657i). The thesis's trim point is not an equilibrium of its model
   (the lift is 9 % over the weight there); it was read off manual flights.
3. Level flight on the wing: alpha 1.8 deg, elevator -1.7 deg, 18.4 N of thrust and L/D 6.5 at 21 m/s.
   Slowest steady flight 15.1 m/s, where the elevator's throw ends (Foxtech: a stall speed of 15 to 16 m/s);
   fastest 22.6 m/s, 81 km/h, with the pusher at 142 rev/s (Foxtech: about 100 km/h). Neither is fitted.
   At 21 m/s the pusher turns 133 rev/s with the advance-ratio law (the thesis's Table 6.5 prints 125) and
   92 rev/s with the thesis's static law (delta_t = 8470; the recorded inputs the thesis's repository ships
   sit near 9750). Past the data the lift peaks at cL 1.64 at 16 deg and falls.
4. The wing's loops, from the 21 m/s trim: +10 m of height in 13.8 s with 0.2 m over (the pusher can hold a
   climb of 0.7 m/s at 21 m/s, and the pitch command stops at 80 % of it, so a climb costs height rate and
   not airspeed); a 90 deg course change in 4.1 s, 1.6 deg over, bank to 33 deg, sideslip to 4.0 deg, the
   height within 3.1 m; -2 m/s of airspeed in 1.1 s.
5. The hover's loops: 5 m sideways in 1.7 s (10 to 90 %) with no overshoot, 2 m up in 1.5 s, 90 deg of
   yaw in 2.4 s.
6. Hovering 30 m up in 8 m/s on the beam (10 m/s at that height): the weathervane has the nose within
   10 deg of the wind after 13 s, the aircraft pushed 0.6 m off its point while it turns; in turbulence at
   the standard's intensities it then holds 0.11 m rms (0.24 m at the most), the nose 7 deg rms off the wind.
   The wing carries part of the weight there (the front rotors turn at 90 against 100 rev/s).
7. Transitions at 60 m. Still air: front in 11.9 s and 115 m, the height within -1.0 / +0.7 m; back from
   21 m/s to a standstill in 10.6 s and 125 m, within -0.3 / +1.0 m. Into 8 m/s (11 m/s up there): front in
   7.4 s and 21 m over the ground; back in 8.9 s and 30 m, the height 3.6 m over at the most.
8. The flight above. Still air: down after 187 s, sinking 0.23 m/s at the touch, on the pad to a
   centimetre; at most 35 m from the route (it cuts each corner by its turn radius), -2.6 / +0.7 m off the
   height; the lift battery gave 18 %, the cruise battery 3.2 %. In 8 m/s from 250 deg with turbulence: 232 s,
   0.30 m/s, 4 cm, 64 m, -5.5 / +2.2 m, 22 % and 4.1 %.
9. A torque-free tumble keeps its energy to 4e-12 and its angular momentum to 3e-10 over 60 s; a minute on
   the wing changes the energy by exactly the work of the aerodynamic and propeller forces (1786.4 J);
   two runs in turbulence are bit-identical; about 20 times real time, autopilot and wind included.
10. The pilot's sticks, in still air. Hovering: up from the ground, the feet leave it after 2.1 s and it
    climbs at 3.0 m/s; let go, it rises 1.5 m more. Forward: 8.2 m/s after 8 s; let go, it stops in 18 m
    and 4.1 s. Turn: 31 deg/s; let go, the nose goes 17 deg further and stops there (the rotors' drag
    torque again). `T`: 11.9 s later the wing has it, the height within -1.0 / +0.7 m. Banked 31 deg it
    turns 15.2 deg/s and is 1.0 m under its height after 8 s; let go, it holds the course it then has
    within 0.7 deg. Up: 0.58 m/s. Slowed to the 18 m/s the setpoint stops at, a descent, the level-off and
    a turn never see less than 16.3 m/s of airspeed. `T` and down: on the ground 17 s after the stick
    went down, sinking 0.30 m/s at the touch, the rotors stopped. Handed to the autopilot for the circuit
    and taken back on the first leg with a touch of a stick; handed back to land, it is on the pad's
    middle 100 s later with the motors stopped.

Believe the wing from about 17 to 22 m/s in gentle manoeuvres. The hover, the stall and the transitions
are a standard model with assumed numbers: they are finite, signed the right way and conserve what they
should, and nothing more is claimed. The battery powers are not to be quoted (the pusher's torque is its
static one at every airspeed).

## In another scene

```python
from babyshark_rig import Babyshark, Wind, Autopilot, Mission, Visual, load_spec, model_path

spec = load_spec()
a = Babyshark(spec, wind=Wind(6.0, from_deg=250.0, spec=spec, seed=3), ground=height_at, h0=0.0)
a.place_on_ground(0.0, 0.0, yaw=math.radians(250.0))          # nose into the wind
ap = Autopilot(a)
mission = Mission([dict(kind="takeoff", height=60.0),
                   dict(kind="cruise", route=[(400.0, -300.0), (400.0, 300.0)], altitude=60.0, airspeed=20.0),
                   dict(kind="land", pos=(0.0, 0.0))])
visual = Visual(tp.GLTFLoader().load(model_path()).scene)
# each step (a.dt = 1/300 s):
mission.update(a, ap)
ap.update()
a.step()
# each frame:
visual.pose(a, h0=0.0)
```

By hand: `pilot = Pilot(a, ap)`, then each step `pilot.keys(canvas)` (or `pilot.sticks(forward=..., turn=...,
side=..., up=...)`) and `pilot.update()` before `ap.update()`; `pilot.toggle()` for the transitions,
`pilot.fly(legs)` to hand it to a `Mission`.

Without a mission or a pilot: `ap.command(position=(n, e, h), yaw=...)` in a hover, `ap.transition("fw", course=...)`,
then `ap.command(course=..., altitude=..., airspeed=...)` on the wing, `ap.transition("mc")` to come back.
`a.out` holds the last evaluated air data, coefficients, thrusts and powers; `a.guards` the steps outside
the identified range; `a.world_position(h0)` and `a.world_rotation()` the aircraft in the world.

# The Bixler 3

A small conventional aircraft for the examples: the HobbyKing Bixler 3, a 1.55 m foam motor glider with
ailerons, elevator and rudder and a pusher propeller on a pylon behind the wing, on the flight model that
B. M. Simmons identified from flight tests of one (his "Bix3", 1.2 kg with its instruments): "System
Identification of a Nonlinear Flight Dynamics Model for a Small, Fixed-Wing UAV", Master's thesis,
Virginia Tech, 2018, <http://hdl.handle.net/10919/95324> (the journal version: Simmons, McClelland and
Woolsey, Journal of Aircraft 56(3), 2019, <https://doi.org/10.2514/1.C035160>, not open access). Where the
X8 above is a flying wing at 18 m/s with a propeller model of its own, this one has three surfaces, flies
at 12 m/s, and has no propeller in its model at all.

The repository page of the thesis states no licence. Its equations and numbers are cited; no text, figure
or code is copied.

| file | what it is |
|---|---|
| `bixler3_spec.json` | every number with its source: the thesis's Tables 3.2, 3.4, 4.5 and 4.7 unchanged, each estimate's printed uncertainty, its modes and its vortex-lattice derivatives for the comparison, the vendor's parts list, and what is ASSUMED or OURS with the reason (the propulsion, the servos, the air, the guards, the autopilot, the guidance) |
| `../rigs/bixler3_rig.py` | the aircraft for any scene: `Bixler3` (the thesis's model, and a throttle that is ours), `Autopilot` and `LOS` (ours), `Visual` (the `.glb` posed from the model); it uses `m350_rig.Wind` and `x8_rig.LOS` as they are; no threepp import |
| `build_bixler3_blender.py` | builds the geometry (numpy) with `../build_common.py` and exports `bixler3.glb` through Blender |
| `bixler3_flight.py` | the checks, a flight's telemetry, and stills or a window of that flight |
| `bixler3.glb` | generated, not committed |

```
blender --background --factory-startup --python build_bixler3_blender.py -- --spec bixler3_spec.json --out bixler3.glb
python build_bixler3_blender.py --check            # the geometry's gates, without Blender
python bixler3_flight.py --checks                  # the ten checks below
python bixler3_flight.py --telemetry [--csv b3.csv] # numpy only: a line a second, a summary
python bixler3_flight.py --wind 5 --from 315 --turbulence 1 --telemetry
python bixler3_flight.py                           # a window; C cycles the camera (chase, high, ground)
python bixler3_flight.py --stills 8,28,38 --out-dir out --cam high
```

The route is 300 m north, a right turn, 250 m east, 60 m up at 12 m/s, by default in 3 m/s from the
west. `--turbulence k` makes the wind an `m350_rig.Wind`: `--wind` is then its speed 10 m up, with a
logarithmic profile and Dryden gusts at k times the standard's intensities.

## The 3D model

`bixler3.glb`: X forward, Y up, Z right, metres, origin at the centre of gravity (the thesis's body
origin). Met and gated: the thesis's projected span 1.540 m, wing area and mean aerodynamic chord
(0.2841 m^2 and 0.1876 m on the mesh against S = 0.285 and cbar = 0.188, each within 0.5 %; the
ailerons' end gaps and the faceted tips are the difference), the centre of gravity 0.078 m aft of the
wing root's leading edge and 0.028 m below the wing's underside, the wing's incidence +0.5 deg and the
tailplane's -3 deg (thesis Table 3.5), the vendor's length 0.948 m (nose to the rudder's trailing edge)
and its 1550 mm span, taken as the span along the wing from tip to tip: 5 deg of dihedral and tips
turned up to 27.9 deg make 1.550 m along the wing over 1.540 m projected. The wing's section is the
Clark Y the thesis names. Everything else is fitted by eye and marked ASSUMED in the spec's `geometry`:
no drawing, CAD model or measured airframe is behind it, only the vendor's two overall dimensions and a
builder's photographs (the spec's `reference.airframe`). The wing's root and tip chord (0.2042 and
0.1497 m) are solved for S and cbar, not measured. 21 240 triangles. `--check` also fails the build
unless the propeller's tip circle is 0.1778 m (7 in), its disc is behind the wing's trailing edge and
the pylon and 12 mm above the tail boom, the raised elevator passes under the rudder at any rudder
angle, and every part is closed and wound outward. It prints: height 0.277 m, tailplane span 0.460 m
(0.049 m^2 with the elevator), pod 0.085 m wide and 0.128 m high.

- `airframe`: fuselage (with the canopy), wing, tailplane, fin, pylon, motor, fittings.
- `aileron_left`, `aileron_right`, `elevator` (one piece): on their hinge lines, local +Z along the
  hinge toward starboard; a positive turn about local +Z puts the trailing edge down.
- `rudder`: on its hinge line, local +Z down the hinge; a positive turn moves the trailing edge to port.
- `propeller`: at the disc centre, spun about local +X (clockwise seen from behind for thrust).
- Empties: `imu`, `gnss`, `pitot_tip`, `camera_nadir`, `camera_fpv` (cameras look down local -Z, +Y up).

A hinge node's rest rotation has no turn about its own +Z, so a scene sets `rotation.z` alone. Checked
on the model as threepp loads it, each node turned by hand: +20 deg lowers both ailerons' and the
elevator's trailing edges and moves the rudder's to port, -20 deg the opposite; the propeller turns
about its own axis; at the actuators' limits (elevator 20 deg up, rudder 25 deg either way) the
rudder's lower edge stays 3 mm over the elevator. `bixler3_rig.Visual` gives the right aileron +da and
the left one -da, the elevator de and the rudder dr: the thesis's signs.

No landing gear: the kit's wheels are optional and the model is a belly lander, as most autopilot
conversions fly. Plain or not drawn (the spec's `geometry.unverified`): flaps (optional on the kit),
the belly skid, servos, horns and pushrods, the wing's joint and any fillet where it meets the pod, the
canopy's real outline, the motor's and propeller's shapes, the thrust line's angle (parallel to body x
here), the trim colours' shapes. It has the Bixler's layout and its published dimensions; it is not a
likeness. No logos or lettering.

## What is the thesis's, what is assumed, what is ours

The thesis's, as written (`Bixler3.rigid_rates` numbers each step): the rigid body (2.7)-(2.12) in body
axes x nose, y right wing, z belly, with Ixz zero; the forces and moments (2.13), X = qbar S CX and the
rest, qbar at the airspeed of the moment; the coefficients (4.2)-(4.7) in the non-dimensional states of
(4.1), which are the TOTAL body velocities and rates over a FIXED 12 m/s (u / Vo, w / Vo, p b / (2 Vo)),
not perturbations and not over the airspeed; the estimates of Table 4.5 (longitudinal, three of them fixed
at vortex-lattice values) and Table 4.7 (lateral-directional). It was identified from hand-flown doublets
about 12 m/s in near-still air, stalls left out, angle of attack and sideslip not measured.

Those estimates are loose, and the thesis prints how loose (`aero.uncertainty` in the spec, 95 % bounds):
several are as large as the estimate. CXu is -0.156 +- 0.157, CXo 0.197 +- 0.195, CZde -0.308 +- 0.351,
CYo 0.0286 +- 0.0382, CXw2 0.960 +- 1.90. Two things the model does follow from such numbers: the one
speed it flies level at on its own (CXu and CXo), and the 3.5 deg of bank it needs to fly straight at
12 m/s (CYo).

The thesis has NO thrust model and no throttle. It did not separate the propeller's thrust from the
airframe's drag: its CX is the two together, at whatever constant throttle each manoeuvre held (it does
not say which). So what makes this an aircraft one can fly is ours, and an assumption:

- The throttle. X = qbar S CX + [T(now) - T(reference)], T a generic propeller law for the vendor's 7x5
  propeller (a static thrust coefficient of 0.11 falling in a straight line to zero at an advance ratio of
  0.80, 207 rev/s at full throttle, a 0.1 s lag: round figures, none measured), and the reference the
  throttle the thesis's flights are TAKEN to have held: 0.83, chosen so that the propeller gives the drag
  of an ASSUMED lift-to-drag ratio of 8 at the speed where the thesis's model flies level hands-off. At
  that throttle the bracket is zero and the model is the thesis's exactly (check 2). The split of CX into
  thrust and drag is an assumption. Nothing about the propulsion is the thesis's, and what follows from it
  is not to be trusted: the glide, the climb rate, the throttle a speed needs, the slowest and fastest
  level speeds. There is no torque, current or battery in it.
- The signs of the deflections. The thesis states none; they are inferred from its coefficients and are
  the NASA (Klein and Morelli) ones: positive elevator is trailing edge down (Cmde < 0), positive aileron
  rolls LEFT (Clda < 0), positive rudder is trailing edge to port and yaws the nose left (Cndr < 0).
- Servos (second order, 60 rad/s, 0.8) and throws (20, 20 and 25 deg); gravity 9.81; the air density
  (ISA at the height; the thesis quotes none, and its own tables point at 1.18 kg/m^3, below).
- Wind: the thesis took the air as still, so its body velocity is the air-relative one; with wind the
  air-relative body velocity stands where it has u, v, w.
- Guards (`envelope`): w / Vo enters the coefficients held to -0.15..+0.25 and v / Vo to +-0.25. The
  model has no stall. Its lift is a quadratic in w / Vo whose slope reaches zero at 0.379 and which
  changes sign near 0.79; the guard stops short of that and holds the lift at its edge value, a plateau.
  Because w is divided by the fixed 12 m/s, the same w / Vo is a larger angle of attack at a lower
  airspeed: at its slow end the model flies level at 8 m/s and 20 deg, which is its arithmetic and not an
  aircraft's. There is no division by the airspeed, so no floor on it. `Bixler3.guards` counts the steps
  on a guard (and those on which the propeller law gives no thrust): none on the flights below.
- `Autopilot`: successive loop closure (Beard and McLain, ch. 6), the X8's with a rudder: roll on the
  ailerons, course or heading on roll, pitch on the elevator, altitude on pitch, airspeed on throttle,
  gains computed from the thesis's coefficients at 12 m/s. The rudder holds the sideslip at zero, through
  the yawing moment, since the rudder's side force (CYdr 0.0157 +- 0.0482) is too small and too uncertain
  for the textbook's loop. It is fed the EXACT state: no estimator, no sensor noise.
- `LOS`: the X8's, with a 40 m look-ahead.

To fly the thesis's model untouched, leave the throttle where `set_state` and `level_speed` put it:
`b = Bixler3(); b.level_speed(apply=True, h=60.0); b.run(10.0)`.

## Against the thesis's modes

The thesis prints the modes of its model at 12 m/s, theta = 0 (Tables 4.2 and 4.8) and no linear matrices.
The implementation is linearised numerically at that point (12 m/s along the nose, everything else zero;
not an equilibrium). The thesis quotes no air density, so the method is first tried on a case with a known
answer: the thesis's vortex-lattice derivatives (Table 3.7) put through the same code must give the modes
XFLR5 printed for them (Table 3.8).

| mode | Table 3.8 (XFLR5) | Table 3.7 here, rho 1.18 | rho 1.225 |
|---|---|---|---|
| short period | -9.31 +- 9.43i | -9.317 +- 9.380i | -9.670 +- 9.535i |
| phugoid | 0.00181 +- 0.860i | +0.002 +- 0.867i | -0.000 +- 0.860i |
| dutch roll | -0.651 +- 4.63i | -0.644 +- 4.609i | -0.676 +- 4.689i |
| roll | -13.3 | -13.30 | -13.79 |
| spiral | 0.0839 | +0.0853 | +0.0855 |

Within 1.6 % at 1.18 kg/m^3 and 3.7 % at 1.225: the non-dimensional states and the linearisation are the
thesis's, and its density was near 1.18. The identified model at that density:

| mode | thesis | this model, rho 1.18 | rho 1.225 |
|---|---|---|---|
| dutch roll (Table 4.8) | -1.89 +- 5.57i | -1.917 +- 5.599i (0.7 % off) | -2.048 +- 5.711i |
| roll (Table 4.8) | -7.77 | -7.814 (0.6 %) | -8.000 |
| spiral (Table 4.8) | -0.118 | -0.1188 (0.7 %) | -0.1190 |
| short period (Table 4.2) | -12.5 +- 4.61i | -6.294 +- 3.844i | -6.532 +- 3.869i |

The lateral-directional modes agree. The short period does NOT, and it is reported as a failed check.
Table 4.2 belongs to the thesis's first longitudinal estimates (Table 4.1, flight data alone, with CZq
+16.7 +- 43.7), not to the set it selects (Table 4.5, the one the model carries); it prints no modes for
Table 4.5. But Table 4.1's numbers give -6.054 +- 5.185i here, no nearer. The printed real part is twice
ours and the imaginary part is not; the cause is not known. The same code reproduces Tables 3.8 and 4.8,
so the short period of this model is taken to be what its coefficients give: 7.4 rad/s, damping 0.85.
The phugoid, which the thesis leaves out, is -0.013 +- 0.764i: barely damped, since the model's x force
hardly changes with speed.

## The checks (`bixler3_flight.py --checks`)

1. The thesis's model untouched flies level at 15.29 m/s: alpha 0.89 deg, bank -5.64 deg, elevator
   +1.36 deg. Hands-off for 30 s it stays there, the thrust difference exactly zero and no guard touched.
   That speed is the difference of two estimates each uncertain by as much as itself.
2. At the reference throttle the six forces and moments are eqs. (4.1)-(4.7) and (2.13), written out a
   second time, to 1e-14 on 500 random states; at full throttle only the x force differs.
3. No air: a thrown, turning body keeps its energy to 3e-11 and its angular momentum to 2e-9 over 20 s.
4. The modes above: the control and the lateral-directional modes pass, the short period fails.
5. Level at 12 m/s with our throttle: alpha 3.41 deg, bank -3.48 deg with no sideslip, elevator -0.14 deg,
   throttle 0.694, 1.14 N of thrust (a lift-to-drag ratio of 10.4 by our split; 6.1 at 10 m/s, 11.2 at
   13, 8.6 at 15).
6. The autopilot, from that trim: +10 m of height in 6.1 s (10 to 90 %), 0.26 m over; a 90 deg course
   change in 1.8 s, 6.2 deg over, bank to 36 deg, the height within -1.4 / +0.6 m; +2 m/s of airspeed in
   1.1 s, 0.17 m/s over. In the roll into that turn the sideslip reaches 10 deg: the model's yaw from roll
   rate (Cnp -0.242) is large, and the rudder loop does not anticipate it.
7. The route in 4 m/s from the west, course hold: on the first leg the cross-track error is 0.00 m and the
   air velocity points 19.47 deg off the track, asin(4 / 12) to 0.01 deg, at 11.31 m/s over the ground; the
   turn ends 0.0 m past the corner; 0.62 m rms on the second leg. Heading hold is 14 m off after 20 s.
8. The same route in an `m350_rig.Wind` (4 m/s 10 m up, 5.6 m/s at 60 m, gusts at the standard's
   intensities): 0.19 m rms on the first leg (0.38 m at the most) with the bank moving 0.9 deg and the
   pitch 1.7 deg (standard deviations), the height within -1.0 / +0.7 m, the throttle between 0.54 and
   0.94. The same seed gives the same flight.
9. No step on a guard on either flight. With the elevator held at its stop and the throttle shut, 1427 of
   1799 steps are on the w / Vo guard: it counts.
10. The slowest steady level flight is 8.04 m/s at alpha 19.6 deg and full throttle, the fastest 17.10 m/s:
    both are where the ASSUMED propeller ends, neither is a stall or the thesis's.

Believe it from about 10 to 15 m/s in gentle manoeuvres, and believe the lateral-directional side more
than the longitudinal. In stronger gusts (`--wind 5 --from 315 --turbulence 2`) the airspeed loop works
the throttle from stop to stop.

## In another scene

```python
from bixler3_rig import Bixler3, Autopilot, LOS, Visual, Wind, model_path

b = Bixler3(wind=(0.0, 4.0, 0.0), h0=0.0)        # NED wind (here 4 m/s from the west), or Wind(4.0, from_deg=270.0, seed=3)
b.trim(12.0, n=0.0, e=0.0, h=60.0, course=0.0)   # level at 12 m/s, crabbed onto a course north
ap = Autopilot(b)
los = LOS([(0.0, 0.0), (300.0, 0.0), (300.0, 250.0)])
visual = Visual(tp.GLTFLoader().load(model_path()).scene)
# each step (b.dt = 1/300 s):
vg = b.ground_velocity()
ap.command(course=los.update(b.n, b.e, math.hypot(vg[0], vg[1])), altitude=60.0, airspeed=12.0)
ap.update()
b.step()
# each frame:
visual.pose(b, h0=0.0)
```

`b.cmd` is (elevator, aileron, rudder, throttle) for a controller of your own; `b.throttle_ref` the
throttle at which the model is the thesis's; `b.out` the last evaluated air data, coefficients and thrust;
`b.guards` the steps on a guard; `b.world_position(h0)` and `b.world_rotation()` the aircraft in the world.

# The Aerosonde

The textbook's aircraft for the examples: the Aerosonde UAV as R. W. Beard and T. W. McLain model it in
*Small Unmanned Aircraft: Theory and Practice* (Princeton University Press, 2012), with the parameters the
authors give for it in the book's repository, <https://github.com/byu-magicc/mavsim_public>
(`mavsim_python/parameters/aerosonde_parameters.py`). The X8 above flies a model identified from one
aircraft's flight tests; this one flies the model a course is taught on, the book's chapters 3 and 4 and
the propeller and motor model its authors later added to chapter 4. It is the textbook's Aerosonde:
nothing here is validated against the real aircraft, which the authors' page says weighs about 25 kg where
the parameter set has 11.

The repository is GPL-3.0. Its numbers are cited as the facts they are and its equations are the book's,
written out here from the book and from the authors' chapter 4 slides; none of its code is copied. The
printed book's Appendix E lists an older parameter set for the same aircraft, with another mass, other
coefficients and the book's first propeller model: the set that flies here is the repository's, and the
printed table differs from it.

| file | what it is |
|---|---|
| `aerosonde_spec.json` | every number with its source: the authors' parameter file unchanged (`physical`, `aero`, `propeller`, `propulsion`), and what is ASSUMED or OURS with the reason (`actuators`, `environment`, `envelope`, `autopilot`, `guidance`); the 3D model's parts |
| `../rigs/aerosonde_rig.py` | the aircraft for any scene: `Aerosonde` (the book's model), `Autopilot` (ours), `LOS` (the X8's, reading this spec), `Visual` (the `.glb` posed from the model); it uses `m350_rig.Wind` as it is; no threepp import |
| `build_aerosonde_blender.py` | builds the geometry and exports `aerosonde.glb` through Blender |
| `aerosonde_flight.py` | the checks, a flight's telemetry, and stills or a window of that flight |
| `aerosonde.glb` | generated, not committed |

```
blender --background --factory-startup --python build_aerosonde_blender.py -- --spec aerosonde_spec.json --out aerosonde.glb
python build_aerosonde_blender.py --check          # the geometry's gates, without Blender
python aerosonde_flight.py --checks                # the ten checks below (numpy only)
python aerosonde_flight.py --checks 1,3            # some of them
python aerosonde_flight.py --telemetry [--csv aerosonde.csv]     # numpy only: a line a second, a summary
python aerosonde_flight.py --telemetry --wind 8 --from 315 --turbulence 1
python aerosonde_flight.py                         # a window; C cycles the camera (chase, high, ground)
python aerosonde_flight.py --hold heading          # hold the nose, not the track: the wind carries it off
python aerosonde_flight.py --shot 30 --out aerosonde.png         # headless still at t = 30 s
python aerosonde_flight.py --stills 10,45,60 --out-dir out --cam high
```

The route is 1000 m north, a right turn, 1000 m east, 100 m up at 25 m/s, in 8 m/s from the west unless
told otherwise. `--turbulence` above 0 makes the wind `m350_rig.Wind`: `--wind` is then the mean 10 m above
the ground on a logarithmic profile (half as much again at the route's height) with Dryden gusts on it.
Drawn: the route at the commanded height, the waypoint poles, the aim point the guidance steers at and
the line of sight to it, the flown track, and the wind as an arrow on the ground.

## The 3D model

`aerosonde.glb`: X forward, Y up, Z right, metres, origin at the centre of gravity. The textbook gives no
drawing and no centre of gravity, so only four numbers are met exactly: the wing's span, planform area and
mean chord (the book's b = 2.8956 m, S = 0.55 m^2, c = 0.18994 m) and the propeller's diameter (0.508 m).
The book's wing is the rectangle b by c; the model keeps the span and the area with a mild taper (constant
chord between the booms, then tapering to the tip about a straight quarter-chord line, 2 deg of dihedral
outboard of the booms). Everything else is ASSUMED and marked so in the spec: fitted to one photograph
(an Aerosonde on its car-roof launch cradle, NOAA Photo Library, on Wikimedia Commons) as described in
words, and to a length of 1.70 m and a height of 0.60 m that are quoted next to the 2.9 m span but are on
no page read. No drawing and nobody's 3D model was used. The origin is ASSUMED on the wing's quarter-chord
line and on the propeller's axis (the book's thrust acts along body x through the centre of mass).
22 160 triangles.

`--check` fails the build unless: the span is 2.8956 m within 2 mm; the wing's area seen from above
(with its ailerons, on a 1 mm grid) is 0.55 m^2 within 0.003 and that area over the span is 0.18994 m
within 1 mm (it measures 0.5493 m^2 and 0.1897 m: the ailerons' end gaps); the blades' tip circle is
0.508 m; the disc clears the booms and the tail by at least 20 mm seen along its axis (75 and 52 mm) and
the blades clear the pod, the wing and the engine ahead of them; the length is 1.70 m and the height
0.60 m within 10 mm; a positive turn of each control node moves its trailing edge the contracted way; and
every part is closed and wound outward. It prints the triangle count, the aspect ratio (15.3), the
length (1.700 m, 1.780 with the pitot tube) and the height (0.600 m from the belly to the tail's top,
0.754 m with a blade straight down).

- `airframe`: fuselage (the pod, orange ahead and white behind, and the pylon the wing stands on), wing,
  tail, booms, engine, fittings.
- `aileron_left`, `aileron_right`, `ruddervator_left`, `ruddervator_right`: on their hinge lines, local +Z
  along the hinge toward starboard; a positive turn puts an aileron's trailing edge down and a
  ruddervator's toward its panel's lower, inner face. The fixed section stops at the hinge, with 3 mm gaps
  at a surface's ends.
- `propeller`: at the disc centre, spun about local +X, the blades pitched for a positive turn (clockwise
  seen from behind); two blades and the spinner.
- Empties: `imu`, `gnss`, `pitot_tip`, `camera_nadir`, `camera_fpv` (cameras look down local -Z, local +Y up).

Checked on the model as threepp loads it, each node turned by hand: +20 deg lowers either aileron's
trailing edge 15 mm; +20 deg on either ruddervator moves its trailing edge down and toward the centreline;
a positive turn of the propeller moves the upper blade to starboard; the empties are where the spec puts
them and the cameras look where it says. `aerosonde_rig.Visual` gives the left aileron +da and the right
one -da (the book's sign), and the ruddervators (de - dr) / 2 on the left and (de + dr) / 2 on the right:
the book's elevator lowers both trailing edges, its rudder moves both to port.

One photograph, described in words, is all the shapes rest on (the spec's `geometry.unverified`): the
pod's length and diameter and where the wing stands on it, the taper, dihedral and section, the ailerons'
extent, the booms' spacing, the tail's angle and chord, the engine (a plain single-cylinder model engine,
its cylinder upright), the propeller's blades and spinner, the pitot tube, the GNSS puck and both cameras
are guesses. The tail is the proportion to trust least: its height follows from the unconfirmed 0.60 m.
The real aircraft burns petrol; the flight model's motor is the textbook's electric one, and the model
shows the engine. Kept plain: no hatches, seams, servo horns, antennas or wing joints. No logos or
lettering.

## What is the book's, what is assumed, what is ours

The book's, as written (`Aerosonde.rigid_rates` names each step's chapter and section):

- Chapter 3: the rigid body, twelve states (position north, east and down, body velocity, the Euler
  angles, the body rates), the inertia's x-z product through the eight Gamma constants.
- Chapter 4: gravity; the air-relative velocity against a wind vector, and from it the airspeed, alpha
  and beta; the lift, C_L(alpha) the linear wing blended into a flat plate past alpha0 = 0.47 rad (the
  book's stall); the drag, a parasitic part plus the induced drag of the linear lift; the pitching moment;
  the side force and the rolling and yawing moments, linear in beta, the rates and the surfaces; lift and
  drag turned from the stability frame to the body by alpha.
- The authors' addendum to chapter 4: the motor's voltage is 44.4 V times the throttle, the propeller's
  speed is the positive root of the quadratic that balances the motor's torque against the propeller's,
  and the thrust and torque are rho n^2 D^4 C_T(J) and rho n^2 D^5 C_Q(J). It is quasi-steady: the rotor
  has no state and no inertia. The thrust acts along body x through the centre of mass; the propeller
  turns clockwise seen from behind and its torque reacts on the airframe as -Q about body x.
- The signs: a positive elevator is trailing edge down and pitches the nose down, a positive aileron rolls
  the right wing down, a positive rudder yaws the nose to port. The model flies the book's three virtual
  surfaces; the real aircraft's inverted V-tail appears only where the 3D model is posed.

`CD0`, `CDalpha` and `S_prop` in the spec belong to the book's linear models and its first propeller
model; they are kept and not used. Integration is fourth-order Runge-Kutta at 1/300 s. The air density is
the parameter file's constant, 1.2682 kg/m^3 at every height, unless `Aerosonde(rho="isa")` or a number
says otherwise.

ASSUMED (`actuators` in the spec): the book's Aerosonde has no actuator dynamics, so a second-order servo
on each surface (50 rad/s, damping 0.707), a throw of +-25 deg on each and a first-order throttle lag of
0.2 s are added, sized for an aircraft of this class and measured on none.

OURS, kept out of the book's equations:

- Guards (`envelope`): the airspeed is floored at 1 m/s where the equations divide by it, and a propeller
  the motor cannot turn (throttle near zero under about 6 m/s) stands still. `Aerosonde.guards` counts
  the steps on each: none on any flight below. There is no clamp on alpha: the book's blend keeps the
  lift finite at every angle. The rest stays the book's and is no stall model (the pitching moment linear
  in alpha, the induced drag growing with the linear lift), so believe it at small angles.
- The wind: a NED vector, a function of time and place, or `m350_rig.Wind` (a mean profile and Dryden
  gusts), sampled once a step and held across the Runge-Kutta stages.
- `Autopilot`: successive loop closure as the book's chapter 6 lays it out (roll on aileron, course or
  heading on roll, pitch on elevator, altitude on pitch, airspeed on throttle), each gain computed from
  the spec's coefficients at the commanded airspeed and the spec's choice of frequency, damping and
  limits. The pitch loop's natural frequency is set as a multiple of the airframe's own (10 rad/s at
  25 m/s), so its proportional gain is -Cmalpha / Cmde at every speed, and it has an integral. The rudder
  is a yaw damper, not a sideslip hold: without any rudder the model's steady sideslip in a 30 deg banked
  turn is under a degree, while its dutch roll has a damping of 0.24, which a rudder proportional to beta
  would stiffen and not damp; the washed-out yaw rate brings the linearised mode to 0.67. It is fed the
  EXACT state: no estimator, no sensor noise.
- `LOS`: the X8's line of sight on a polyline, with a look-ahead of 85 m for 25 m/s.

Not in the model: the propeller's inertia and gyroscopic moment, the battery's sag, the ground, any
sensor.

## The checks (`aerosonde_flight.py --checks`)

1. Level at 25 m/s in still air: residual 3e-15; alpha 2.85 deg, elevator -7.10 deg, throttle 0.774, the
   propeller at 514 rad/s (4912 rev/min, J 0.60); 10.33 N of thrust against 10.32 N of drag, L/D 10.4; the
   propeller's 0.63 N m of torque held by 0.34 deg of aileron, 0.05 of rudder and 0.03 deg of bank; the
   motor at 34.3 V and 11.1 A, 382 W. At 18 m/s: alpha 7.8 deg, elevator -20.8, throttle 0.58, L/D 15.5,
   194 W; at 30 m/s: 1.2 deg, -2.6, 0.93, 7.6, 611 W.
2. Thrown and tumbling where there is no air (no aerodynamics, no thrust), 10 s: the energy is kept to
   5e-13 and the angular momentum to 2e-13, and it falls the 425.7748 m that v t + g t^2 / 2 gives.
3. The linearised modes at that trim (`Aerosonde.jacobian`): short period 11.0 rad/s, damping 0.44;
   phugoid 0.50 rad/s, 0.29; roll 0.045 s; dutch roll 4.79 rad/s, 0.24; and a spiral mode that is
   UNSTABLE, doubling in 7.8 s. That is the parameter set's (Clbeta Cnr - Cnbeta Clr is negative), not the
   implementation's, and the autopilot's roll loop holds it.
4. Open loop from the trim, 60 s and 1500 m with the commands held: the airspeed, the height, the heading
   and the bank stay to 1e-10 or better. Started half a degree of bank off the trim it is banked 7 deg and
   has turned 27 deg after 30 s (the spiral).
5. The autopilot's steps from the trim: +10 m of height in 2.2 s (10 to 90 %), 0.46 m over, the airspeed
   within 24.0 to 25.6 m/s; a 90 deg course change in 5.8 s, 1.2 deg over, bank to 31 deg, sideslip to
   2.2 deg, the height within 1.0 m; +3 m/s of airspeed in 1.4 s, 0.20 m/s over, the throttle at its stop
   on the way.
6. The route in a steady 8 m/s from the west, course hold, 78.8 s. Leg north: on the line to a centimetre,
   the air velocity 18.66 deg off the track, which is the wind triangle's asin(8 / 25), 23.7 m/s over the
   ground. Leg east, downwind at 33.0 m/s: 0.28 m rms off the line over its last 500 m. The turn cuts the
   corner by 39 m and goes 19.7 m past the new leg (the hand-over is computed at the ground speed before
   the turn, and the tailwind adds 9 m/s through it). `--hold heading` flies the first leg 28.7 m downwind
   of the line, the look-ahead times the tangent of the crab angle.
7. The guards on that flight: none. Height 99.0 to 100.7 m, airspeed 24.8 to 25.2 m/s, alpha 2.4 to
   4.1 deg, sideslip within 2.3 deg, throttle 0.76 to 0.79, aileron within 23.7 deg of its 25.
8. The glide with the throttle closed: 3.1 m/s of sink and a glide ratio of 5.8 at 18 m/s, 7.6 m/s and
   3.1 at 25 m/s, 13.0 m/s and 2.1 at 30 m/s, where the airframe's own L/D is 15.6, 10.1 and 6.9. This is
   the propeller model and not a measured glide: with no voltage the motor is a short across its winding
   and holds the propeller near standstill (16 rad/s at 25 m/s), and the thrust polynomial, fitted where a
   propeller drives, makes of that 22.6 N of drag, twice the airframe's.
9. The slowest steady level flight is 16.8 m/s (alpha 9.3 deg, C_L 1.09), where the elevator's ASSUMED
   25 deg ends; the fastest is 32.5 m/s, at full throttle. The stall blend does nothing at the slow end
   (sigma is 2e-7 there). The book's lift curve peaks at C_L 2.42, alpha 23.6 deg, which would carry the
   weight at 11.3 m/s, and trimming there would take 65 deg of elevator: no steady flight of this model
   reaches its stall, and a C_L of 2.4 is no real wing's.
10. The route in `m350_rig.Wind`, 8 m/s from the west 10 m up (12.0 m/s at the route's height) with gusts
    at the standard's intensities: done in 80.4 s, 0.24 m rms off the first leg and 1.4 m off the second,
    height 97.9 to 101.1 m, airspeed 23.9 to 26.2 m/s, sideslip within 5.1 deg, no guard; two runs are
    bit-identical; 11 times real time with the gusts on a busy machine (24 without them), autopilot
    included.

Believe it for what a textbook model is: gentle flight between about 18 and 30 m/s. The stall, the glide
and anything near the ends of the speed range are the formulas' and the assumed actuators', not an
Aerosonde's.

## In another scene

```python
from aerosonde_rig import Aerosonde, Autopilot, LOS, Visual, Wind, model_path

a = Aerosonde(wind=(0.0, 8.0, 0.0), h0=0.0)      # NED wind (here 8 m/s from the west), or Wind(8.0, from_deg=270.0, seed=3)
a.trim(25.0, n=0.0, e=0.0, h=100.0, course=0.0)  # level at 25 m/s, crabbed onto a course north
ap = Autopilot(a)
los = LOS([(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0)])
visual = Visual(tp.GLTFLoader().load(model_path()).scene)
# each step (a.dt = 1/300 s):
vg = a.ground_velocity()
ap.command(course=los.update(a.n, a.e, math.hypot(vg[0], vg[1])), altitude=100.0, airspeed=25.0)
ap.update()
a.step()
# each frame:
visual.pose(a, h0=0.0)
```

`a.cmd` is (elevator, aileron, rudder, throttle) for a controller of your own; `a.out` holds the last
evaluated air data, coefficients, the propeller's speed, thrust and torque and the motor's current;
`a.guards` the steps on a guard; `a.trim(V, throttle=0.0)` the glide; `a.world_position(h0)` and
`a.world_rotation()` the aircraft in the world.
