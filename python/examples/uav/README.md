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
