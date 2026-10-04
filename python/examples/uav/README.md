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
| `../x8_rig.py` | the aircraft for any scene: `X8` (the article's model), `Autopilot` and `LOS` (ours), `Visual` (the `.glb` posed from the model); no threepp import |
| `build_x8_blender.py` | builds the geometry (numpy) with the Mariner generator's mesh and export code and exports `x8.glb` through Blender |
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

`../x8_rig.py` imports no renderer and reads no scene's globals:

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
