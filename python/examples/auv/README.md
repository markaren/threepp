# The REMUS 100

An AUV for the examples: the REMUS vehicle on the 6-DOF model of T. Prestero, "Verification of a
Six-Degree of Freedom Simulation Model for the REMUS Autonomous Underwater Vehicle", S.M. thesis,
MIT / WHOI Joint Program, 2001, <http://hdl.handle.net/1721.1/65068>. The thesis's non-linear
equations of motion are implemented as it writes them, with the coefficients of its Appendix B,
unchanged. MIT's notice on the thesis allows it to be viewed for any purpose and requires
permission to reproduce or distribute it, so nothing of its text, figures or PDF is here: the
spec cites each number to its table and page. Built the way the X8 (`../uav/`) and the boats
(`../usv/`) are: one spec file, a headless-Blender generator, a `.glb` that threepp loads with
`GLTFLoader`, and a rig any scene imports.

| file | what it is |
|---|---|
| `remus100_spec.json` | every number, each with its source: the thesis's tables, or ASSUMED with the reason; the thesis's results (steady conditions, simulated fin steps read off its figures, its linearised depth-plane model) for the comparison; the hull, fins and fittings; the materials |
| `../remus_rig.py` | the vehicle for any scene: `Remus` (the thesis's model), `Autopilot` and `LOS` (ours), `Visual` (the `.glb` posed from the model); no threepp import |
| `build_remus_blender.py` | builds the geometry (numpy) with the Mariner generator's mesh and export code and exports `remus100.glb` through Blender |
| `remus_dive.py` | the example: the REMUS runs a route (a leg at 4 m, a leg down to 12 m, a leg at 12 m) over a seabed in a current you set; window, stills or telemetry |
| `remus100.glb` | generated, not committed |

## Build

From this folder, once (about 5 s; `blender` is your Blender executable):

```
blender --background --factory-startup --python build_remus_blender.py -- --spec remus100_spec.json --out remus100.glb
```

`python build_remus_blender.py --check` builds the geometry without Blender and prints the
checks: hull length 1.3327 m (Table 2.1: 1.33), volume 0.03166 m^3 (Table A.1: 0.0315), frontal
area 0.0287 m^2 (0.0285), projected area 0.2265 m^2 (0.226), wetted area 0.725 m^2 with the end
faces (0.709); the hull's volume centroid 5.2 mm ahead of the measured centre of buoyancy
(Table 2.7's bare-hull estimate: 5.54 mm); the radius at the fin's root leading edge 61.7 mm
(Fig. 2-3: 62). The build fails if the length is more than 1 mm off or the volume or projected
area more than 2 %, or any closed part is not closed and consistently wound.

## Run

```
python remus_dive.py                              # a window; C cycles the camera (chase, side, fixed)
python remus_dive.py --current 0                  # still water (default: 0.3 m/s setting east)
python remus_dive.py --current 0.5 --set 45       # 0.5 m/s setting north-east
python remus_dive.py --hold heading               # hold the nose, not the track: the current carries it off
python remus_dive.py --shot 145 --out remus.png   # headless still at t = 145 s
python remus_dive.py --stills 30,145,160 --out-dir out --cam chase
python remus_dive.py --telemetry [--csv remus.csv]   # numpy only: a line every 5 s, a summary, every step to a file
```

The route is 200 m north at 4 m, 120 m east descending to 12 m, 160 m south at 12 m, over sand
at 20 m, at 1.75 m/s through the water. Drawn: the route at its depths, the poles at its
waypoints, the aim point the guidance steers at and the line of sight to it, the travelled
track, and the current as an arrow on the seabed (4 m per m/s).

## Frames

Inside the model everything is the thesis's: NED position (N, E, D), D the depth; body axes x
forward, y starboard, z down, origin at the centre of buoyancy (0.611 m behind the nose);
Euler angles (phi, theta, psi), psi from north toward east. One mapping to threepp's world, at
the boundary (`remus_rig.ned_to_world`): world = (E, h0 - D, -N), x east, y up, z south, as the X8
rig and the drone rig use it; h0 is the world height of the NED origin (the
sea surface). Heading psi = 0 points the nose along -z, psi = 90 deg along +x. The current is the
water's velocity over the ground, NED; the example's `--set` is where it flows toward.

The `.glb` is X forward, Y up, Z right (starboard), metres, origin at the centre of buoyancy. A
body vector (x, y, z) is (x, -z, y) in it.

- `hull` (the Myring hull, the black nose cap, the section seams), `fittings` (the LBL transducer
  of Fig. 2-2, three nose pockets, two side-scan transducers and eight ADCP transducers sized from
  Table 5.3, the antenna mast): meshes.
- `stern_port`, `stern_starboard`, `rudder_upper`, `rudder_lower`: Fig. 2-3's fins with NACA 0012
  sections, each on its own node at the fin post (x_fin = -0.638 m, Table 2.2). Local +Z is the
  body axis the pair turns on: body +y (model +Z) for the stern planes, body +z (model -Y) for the
  rudders. A positive turn about local +Z is the thesis's positive angle: delta_s puts the
  trailing edge down (Fig. 4-2), delta_r puts it to port (Fig. 4-1).
- `propeller`: three blades on a hub behind the tail, spins about local +X; positive is a spin
  vector along body +x (clockwise seen from behind), the sense in which the hull's reaction is the
  thesis's Kprop < 0. ASSUMED geometry (the thesis gives none): 0.15 m across, 0.12 m pitch.
- Empties (look down local -Z, local +Y up): `imu`, `depth`, `lbl`, `adcp_down`, `adcp_up`,
  `sss_port`, `sss_starboard`, `gnss`.

`remus_rig.Visual(root).pose(auv, h0)` places the root and turns the fins (after their rest
rotations) and the propeller. Checked on the builder's meshes through `Visual`'s rotations: +0.2
rad on a stern plane moves its trailing edge 5.5 mm down, on a rudder 5.5 mm to port; the model
then gives the stern planes' force 4.34 N up at the tail and -2.77 N m of pitch (nose down), the
rudders' 4.34 N to starboard and -2.77 N m of yaw (nose to port); +0.2 rad of propeller moves the
upper blade 14 mm to starboard; heading 90, pitch 10, roll 20 deg puts the nose along world
(0.985, 0.174, 0) and starboard down. The rendered stills at +-13.6 deg show the same.

## What is the thesis's, what is assumed, what is ours

The thesis's, as written (`Remus.forces` and `Remus.rigid_rates` every step): the rigid body
about the centre of buoyancy with the centre of gravity 19.6 mm below it (3.8); the hydrostatics
r_G x f_G - r_B x f_B (4.2); the force and moment sums (4.49), with Appendix B's coefficients for
axial and crossflow drag, body lift and its moment, the added mass and its cross terms, fin lift,
the rolling drag; the propeller's thrust 3.86 N and torque -0.543 N m at 1500 RPM (Table C.9); the
mass matrix (6.8); the kinematics (3.1)-(3.5); the fin limit +-13.6 deg (Table 2.2). The force
sums agree with the thesis's own listing (REMUS.m, its p. 117) to 1.4e-14 over 2000 random states.
Fourth-order Runge-Kutta (6.18) at 1/120 s, the commands held over the step.

Not in the thesis, ASSUMED (`remus100_spec.json` says why each): g = 9.81 (m = W / g = 30.48 kg);
a first-order lag of 0.1 s on each fin pair (the thesis assumes instantaneous fins; `fin_tau=0`
restores that, and the comparison below runs with it); the propeller's thrust and torque away from
1500 RPM, Xprop (n / 1500)^2 and Kprop (n / 1500)^2, and a 0.3 s lag of the propeller speed (the
thesis runs at a constant 1500 RPM); the 3D model's propeller, antenna and sonar placements, the
colours.

OURS, kept out of the thesis's equations:
- The current: the velocity state is the velocity through the water, every term of the thesis's
  equations takes it, and the kinematics add the current back: (N, E, D)' = J1 nu_r + V_c. That is
  the standard form for a current steady and uniform in the earth frame (Fossen 2011, Section
  10.3): the hydrodynamic terms depend on the motion relative to the water by derivation, and the
  rigid-body terms take nu_r exactly, because the current's body-frame components then change only
  by the rotation, so nu' + omega x nu = nu_r' + omega x nu_r. The hydrostatics do not depend on
  velocity. `remus_rig.py`'s THE CURRENT says it in full.
- Guards (`envelope` in the spec), counted in `Remus.guards` each step they act: the fin pairs'
  lift held at its value at an effective angle (delta plus the inflow angle of eq. 4.42) of 12 deg,
  the stall angle the thesis gives for the stern planes (Section 9.5; its model assumes no stall);
  cos(theta) held at or above cos(85 deg) in (3.5). `Remus.flags` counts, without changing
  anything, the steps with the speed outside 1.0-2.0 m/s or the angle of attack or sideslip past
  15 deg. `guards=False` (`--no-guards`) runs the thesis's equations bare.
- `Autopilot`: stern planes on pitch, pitch on depth, rudder on course or heading, propeller on
  speed. The model is open-loop unstable in both planes at speed (at the 1500 RPM trim the
  sway-yaw root is +0.85 /s and the heave-pitch pair +0.37 +- 0.40i /s: the Munk moments Muw, Nuv
  outweigh the fins), so the pitch and heading gains are output feedback on the 3-state models
  (w, q, theta) and (v, r, psi) of the Jacobian at the trim, placing a pair at the spec's
  frequency and damping (`place3`); the third poles come out at -4.3 and -3.6 /s at 1.75 m/s.
  From the 1.75 m/s trim in still water: 10 -> 15 m in 19.3 s (90 %) with 0.15 m over; a 90 deg
  heading change with 1.9 deg over and the depth within 0.08 m; 15 -> 8 m in 17.4 s (90 %) with
  0.24 m over. The pitch command is limited to 12 deg; the stall guard acted 349 steps (2.9 s)
  over the three steps.
- `LOS`: line of sight on a polyline, chi_d = path angle + atan(-e / delta), delta = 8 m, the leg
  handed over R tan(|dchi| / 2) before the corner with R the model's own steady turning radius at
  60 % rudder (`Remus.turn_radius`: 7.0 m at 1.75 m/s). `Autopilot.command(course=...)` is the hook
  a film's own law replaces.

## Against the thesis

### Inconsistencies in the thesis, and the reading taken

| where | what is printed | reading taken, and why |
|---|---|---|
| Table 8.2 vs Appendix B (B.1, B.2, C.2) | 8.2 multiplies Zww by 10 and Mqq by 12.5; B prints Zww unmultiplied (-131) and Mqq x 20 (-188) | Appendix B: the thesis states its simulator ran these (Section 8.1.2, Appendix B's preamble). Its own figures do not settle it (below) |
| eq. (6.7), Y | + m zg q r | - m zg q r: the rigid body (3.8) and the listing |
| eq. (6.7), K | + m (u q - v p) | absent: (3.8) has it only multiplied by y_g = 0, and the listing has none |
| listing, N | (Nwp - m xg) w p + (Nur + m xg) u r | (Nwp + m xg) w p + (Nur - m xg) u r: (3.8) and (6.7); x_g = 0, so no result changes |
| eq. (4.3), K and N | -(y_g W - y_b B) cos theta cos phi in K; both N terms | r_G x f_G - r_B x f_B of (4.2), which flips those terms; all multiply y_g, y_b, x_g, x_b = 0 |
| listing | clips only positive fin angles | both signs clipped at 13.6 deg |
| Table 2.3 vs A.1 | B = 306 N vs 308 N | 306 N: Sections 2.4 and 7.4.2 ballast to about 1.5 lb (6.7 N) of buoyancy |
| eq. (2.2) | (s - l)^2 in the second term | (s - l_f)^2: the Myring form; gives d/2 at l_f, Fig. 2-3's 62 mm, Table A.1's volume to 0.5 % and Table 2.7's centroid offset to 0.3 mm |
| Xuu, Section 4.2.1 / A.1 / 5.4 | -1.62 kg/m is cd = 0.110; the text uses 0.27, A.1 prints 0.30, 5.4 0.267 | -1.62 as printed (with Xprop = 3.86 N it gives the 1.54 m/s the thesis states) |
| Section 4.6.1 | 1.51 m/s and "-2.28 Xuu" | 1.54 m/s (Table 8.1, Section 8.1.1, and Xprop / -Xuu) |
| eq. (4.48) vs B.1 | Yuv = Yuvl + Yuvf = -38.2; B.1 prints -28.6 (Yuvl alone); likewise Zuw | B.1's value (the moment sums do include the fin terms) |
| C.10 vs Table 2.2 | fin lift 9.64 = 0.902 x (1/2) rho cLa Sfin; eq. (4.44) drops the 1/2 of (4.43); Chapter 9 uses rho cLa Sfin U^2 | C.10 / B.1's 9.64 |
| Mrp, Npq | from the unadjusted Kpd (4.86); with Table 8.2's Kpd they are 4.81 | as printed (0.8 % from the Kirchhoff value) |
| Table 2.2 | a_fin = 5.14 m | not used; Fig. 2-3 shows 0.131 m |
| Table 9.3 | last row labelled Z_delta_s | it is M_delta_s |
| eq. (9.36) vs Table 9.3 | (9.30) with Table 9.3 gives the w-row 2.1 times smaller than (9.36) prints | reported below, not used |
| eq. (9.52) vs (9.40) | -3.18 / (s^2 + 1.09 s + 0.52) vs -4.16 / (s^2 + 0.82 s + 0.69) | reported below |
| Section 8.3 vs Figs. 8-2, 8-3 | +4 deg rudder for 25 s; the plotted forces step at 10 s and 40 s | the plotted times |
| Table 8.1 / Figs. 8-3, 8-8, 8-13 | no ballast stated per run; the force sums at t = 0 give W - B = 0, +2.25 / cos 5 deg, -6.6 / cos 5 deg | those per run, for the step comparisons |

### Steady conditions

- Speed at 1500 RPM: 1.5436 m/s (Xprop / -Xuu), the thesis's 1.54 m/s.
- Roll at which the hydrostatic moment balances the propeller torque: -5.316 deg; the thesis
  measured -5.3 deg at sea and derived Kprop from it (eq. 4.47).
- Straight motion at constant depth at 1500 RPM (`Remus.trim`, 306 N of buoyancy): u 1.477 m/s
  (the buoyancy's component along the pitched-down axis costs 0.33 N of thrust), theta -2.69 deg,
  alpha -2.67 deg, phi -5.32 deg, stern planes -9.34 deg (effective -12.01 deg with the inflow),
  rudder -0.93 deg. The thesis prints no trim; its Fig. 7-9 shows the vehicle at sea holding depth
  with +4 deg of pitch fin, which it ascribes to a nose-down ballast that its model does not have.

### Fin-step responses (Chapter 8), from Table 8.1's initial state, fins instantaneous

Rudder +4 deg at 10-40 s, -4 deg at 40-70 s, W = B (Figs. 8-1 to 8-3):

| | thesis (figure) | this model |
|---|---|---|
| yaw rate at 14 s; mean 25-38 s; mean 50-68 s | -9.5; -9.0; +9.3 deg/s | -9.48; -8.98; +8.98 |
| surge in the turns | 1.38-1.40 m/s | 1.371 |
| sway, +4 / -4 deg | +0.07 / -0.04 m/s | +0.058 / -0.058 |
| heading at 10 s (no rudder yet); at 42 s | about +20 (at 11 s); -245 deg | +18.0; -243 |
| north / east extent | -3..+31 / -18..+3 m | -2.7..+30.9 / -18.7..+3.0 |
| pitch, pitch rate in the turns | +-10 deg, +-6 deg/s, period 10-14 s | +-15 deg, +-6.0 deg/s, period 15.9 s |
| roll-rate oscillation | about 1 Hz | 1.15 s |

With the spec's 7 N of buoyancy instead (Table 8.1 states no ballast) the same input gives -5.1
deg/s and 0.67 m/s: the vehicle pitches down and dives through the run. The figures' force
sums at t = 0 show neutral ballast.

Stern planes -4 deg at 2 s, W - B = +2.26 N (pitching up, Figs. 8-6 to 8-10), and +8 deg at 2 s,
W - B = -6.63 N (pitching down, Figs. 8-11 to 8-15), values at 1, 2, 3, 3.8 s:

| | thesis (figure) | this model |
|---|---|---|
| up: pitch (deg) | 1.0, 6.0, 15.5, 24.5 | 0.9, 4.8, 11.8, 18.2 |
| up: pitch rate (deg/s) | 2.7, 6.8, 11.0, 11.0 | 2.5, 5.0, 7.6, 7.7 |
| up at 3.8 s: depth change, u, w, heading, yaw rate | -0.47 m, 1.43, +0.17 m/s, +4.0 deg, +4.5 deg/s | -0.34, 1.46, +0.13, +4.2, +4.3 |
| down: pitch (deg) | -2.3, -12.0, -26.0, -36.5 | -2.3, -9.5, -19.8, -28.1 |
| down: pitch rate (deg/s) | -6.6, -11.8, -13.7, -11.5 | -5.5, -8.0, -10.2, -9.2 |
| down at 3.8 s: depth change, u, w, heading, yaw rate | +0.55 m, 1.19, -0.20 m/s, -7.5 deg, -6.2 deg/s | +0.46, 1.27, -0.21, -8.5, -6.2 |

The yaw and roll coupling (through the -5 deg roll) matches; the pitch response is 20-25 %
weaker than the figures. The force sums the thesis plots (Figs. 8-8, 8-13) locate it: Z and N
agree from t = 0, the pitch moment M falls behind (0.58 against 0.73 N m at 1 s pitching up).
The difference follows Mqq: the rms difference of Z and M against the two figures is 0.47 and
0.48 with Appendix B's -188, 0.24 and 0.20 with Table 8.2's -117.5 (12.5 x -9.40), 0.12 and 0.10
with -94; Zww x 10 (Table 8.2) makes it 1.36 and 1.93. The turn's pitch amplitude points the other
way: +-15 deg with Appendix B, +-21 with Mqq -117.5 alone, +-10.3 with Table 8.2 applied in full.
No one printed coefficient set reproduces all three figures; the model keeps Appendix B. The step
times and the per-run ballast above are read off the thesis's own force plots; nothing of the
model is fitted.

### The linearised depth-plane model (Chapter 9)

Chapter 9 linearises with its own coefficient set (Table 9.3): crossflow drag linearised by
slopes, fin lift for both fins without C.10's factor, body lift 1/2 rho d^2 cyd_beta = 22.4 kg/m.
The nonlinear model's own derivatives at u = 1.54 m/s (no quadratic drag at w = q = 0):

| | Table 9.3 | this model |
|---|---|---|
| Z_w (kg/s) | -66.6 | -44.0 |
| Z_q (kg m/s) | -9.67 | -8.04 |
| M_w (kg m/s) | 30.7 | 36.96 |
| M_q (kg m^2/s) | -6.87 | -3.08 |
| M_theta (kg m^2/s^2) | -5.77 | -5.86 |
| Z_delta_s, M_delta_s | -50.6, -34.6 | -22.9, -14.6 |

The 4-state (w, q, z, theta) poles: (9.36) as printed 0.29 +- 0.53i, -4.07; Table 9.3 through
(9.30) 0.17 +- 0.51i, -2.44; this model's Jacobian 0.43 +- 0.19i, -2.18. All three have the
unstable oscillatory pair of the Munk moment. The 3-state pitch model (q, z, theta): (9.40)
-0.41 +- 0.72i, (9.52) -0.55 +- 0.47i, this model -0.19 +- 0.82i. The thesis's controller (Kp
10.345, tau_d 0.21 s, gamma -0.772, Figs. 9-5, 9-8) was designed on (9.40) and is not used.

## Internal checks

- Weight and buoyancy: W 299 N, B 306 N, m 30.48 kg; at rest K = M = -z_g W sin(angle) exactly
  (5.8604 N m/rad).
- Free decay at rest, W = B, from 10 deg: roll period 1.2778 s against 2 pi sqrt(I_eff / z_g W) =
  1.2768 s (I_eff = Ixx - Kpd less the sway coupling, 0.2420 kg m^2), amplitude 8.9, 7.3, 6.1, 5.3
  deg (Kpp); pitch 7.469 s against 7.460 s, heavily damped by Mqq (1.3 deg after one period).
- The added-mass cross terms of Tables C.6-C.8 equal -C_A(nu) nu of the mass matrix's added mass
  (Kirchhoff, Fossen 2011 eq. 6.43) except Mrp and Npq (0.8 %, the unadjusted Kpd).
- Energy: with only the ideal-fluid terms and W = B, a 600 s tumble keeps T + V to 1.5e-6 J
  (kinetic 1.5-2.5 J). The full model for 120 s with the fins held off the trim: the energy changes
  +123.5757 J, the work of drag, lift, fins, propeller and the non-Kirchhoff remainder is +123.5753 J.
- Two runs give bit-identical states at every step (60 s with the autopilot in a current).
- Actuators: the fin and propeller steps follow 1 - exp(-t / tau) to 1.6e-7 and 1.9e-9 of the step.

## The current

`remus_dive.py --telemetry`, 1.75 m/s through the water, the settled middle of each leg:

| current (toward east) | hold | leg 1 (north): water velocity off the track / asin(-V_c / U) | cross-track rms, leg 1 / leg 3 |
|---|---|---|---|
| 0 | course | 0.00 / 0.00 deg | 0.00 / 0.15 m |
| 0.3 m/s | course | -9.87 / -9.87 deg | 0.00 / 0.13 m |
| 0.3 m/s | heading | -9.86 / -9.86 deg | 1.39 / 1.41 m (delta tan(crab) = 1.39) |
| 0.5 m/s | course | -16.59 / -16.59 deg | 0.00 / 0.27 m |
| 0.5 m/s | heading | -16.59 / -16.59 deg | 2.38 / 2.49 m (2.38) |

The nose sits within 0.06 deg of the water velocity (no steady sideslip). Under heading hold the
line-of-sight law settles where atan(e / delta) equals the crab angle. The corner and descent
onto leg 2: at most 1.4 m off the track and 0.6 m off the depth with 0.3 m/s, 2.4 m and 1.1 m with
0.5 m/s; the stall guard acts for 3.9-4.5 s of each run, while the stern planes level the descent.

## The model's range

The thesis's coefficients are fixed at its trial point: 1500 RPM, 1.54 m/s ("small deviations from
this forward speed", Section 4.6.1), small angles of attack (Sections 4.4, 4.5). Straight trims:

| RPM | u (m/s) | pitch | stern (effective) | roll |
|---|---|---|---|---|
| 1250 | 1.153 | -4.3 deg | -13.9 (-18.2) deg | -3.7 deg |
| 1500 | 1.477 | -2.7 | -9.3 (-12.0) | -5.3 |
| 1750 | 1.760 | -1.9 | -6.8 (-8.7) | -7.3 |
| 2000 | 2.032 | -1.4 | -5.2 (-6.7) | -9.5 |

Below about 1300 RPM (1.2 m/s) no depth can be held: the stern planes would need more than their
13.6 deg against the 7 N of buoyancy, and the vehicle rises; at 1000 RPM there is no trim. At the
thesis's own 1500 RPM the trim's effective stern-plane angle is the 12 deg stall angle the thesis
gives, so with the guard on there is no margin to pitch up: holding 10 m, going to 15 m and back
ends at 12.3 m (the guard acting 3561 steps); without the guard the same commands end at 10.00 m.
The autopilot therefore runs at 1.75 m/s by default (`autopilot.speed`); `--speed 1.54
--no-guards` runs the thesis's point bare. The roll offset grows with the propeller torque, as
n^2 by the assumed law. Past 30 deg of pitch REMUS aborts (Section 9.5), and the hull terms are
small-angle derivations: `Remus.flags` counts those steps; none occurred on the runs above.

## In another scene

`../remus_rig.py` imports no renderer and reads no scene's globals:

```python
from remus_rig import Remus, Autopilot, LOS, Visual, model_path

auv = Remus(current=(0.0, 0.3, 0.0))                 # NED current (here 0.3 m/s setting east)
ap = Autopilot(auv)                                  # 1.75 m/s by the spec
auv.trim(ap.rpm, n=0.0, e=0.0, d=4.0, course=0.0)    # straight at 4 m, crabbed onto a course north
los = LOS([(0.0, 0.0), (200.0, 0.0), (200.0, 120.0)], auv=auv)
visual = Visual(tp.GLTFLoader().load(model_path()).scene)
# each step (auv.dt = 1/120 s):
vg = auv.ground_velocity()
ap.command(course=los.update(auv.n, auv.e, math.hypot(vg[0], vg[1])), depth=4.0)
ap.update()
auv.step()
# each frame:
visual.pose(auv, h0=0.0)                             # h0: the world height of the sea surface
```

`los.aim` is the aim point, `los.e` the cross-track error; `auv.out` holds the last evaluated
forces by origin, angles and guard flags; `auv.world_position(h0)` and `auv.world_rotation()` give
the vehicle in the world for a camera.
