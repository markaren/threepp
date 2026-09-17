# flyeye

A fly optic lobe as a vision sensor for threepp. A rendered frame becomes the input of
721 photoreceptor columns on a hex lattice. The pretrained flyvis network (45,669
neurons, 65 cell types, 1.5 M synapses from the FIB-25/FIB-19 medulla connectomes) is
then stepped on it, and named cell populations (T4/T5 local motion, and later wide-field
and looming readouts) come out as sensor signals. The runtime is our own torch code:
one sparse matrix-vector product and a leaky-integrator update per step, read from
`data/flyeye_model.npz`. flyvis is never imported at runtime.

| File | Role |
|---|---|
| `lattice.py` | `HexLattice`: column (u, v), BoxEye 13x13 box mean at the 721 columns, sRGB frame to luminance; column pixels, rays and image tangents of a pinhole eye |
| `optic_lobe.py` | `OpticLobe(npz, device, dtype)`: `reset`, `fade_in`, `step`, `by_type`, `central`; one eye or a batch of B |
| `eye.py` | `EyeView(renderer, camera, size)`: one `add_view` and its `FrameTensors`, live BGRA to 721 receptors on CUDA; `configure_sensor_renderer` |
| `readouts.py` | `MotionField` (T4/T5 flow per column), `RotationReadout` (wide-field rotation), `Looming` |
| `spike_offline.py` | Phase 0 gate: own path vs the flyvis reference, direction selectivity, the figure |
| `dt_sweep.py` | Euler stability at dt 1/60, 1/100, 1/200 and `step` timing |
| `render_edges.py` | The eight threepp Vulkan moving-edge movies used by the gate |
| `online_gate.py` | Phase 1 gate: the online eye against Phase 0, frames identical and live, flush sweep |
| `timing.py` | Phase 1 timing table |
| `sanity_rig.py` | Phase 1 readout sign check on a live two-eye rig |
| `tools/export_flyvis.py`, `tools/flyvis_reference.py` | The only flyvis users (throwaway venv) |

## Phase 0 results (2026-09-17, RTX 4070, torch 2.12.1+cu126, Python 3.14)

Stimuli: 8 threepp-rendered movies, ON (204) and OFF (53) edges on grey 128, moving
right/left/up/down at 13 columns/s (1.69 px/frame at 100 fps). Each movie is 403x403 px
and 261 frames: 20 pre-roll frames plus 241 frames for the crossing. The reference is
flyvis 1.2.0 (`flow/0000/000`) on the same PNGs at dt = 1/100, using tutorial 07's
`BoxEye`, `fade_in_state(1.0)` and `simulate`.

**Match.** The table gives the max abs difference over all 45,669 nodes, all 261 frames
and all 8 movies. The last column is relative to max|v_ref|, which is 5.05.

| Own path | BoxEye | post-fade-in state | responses | relative |
|---|---|---|---|---|
| CPU float32 | 0 (bit-exact) | 9.5e-7 | 1.7e-6 | 4.0e-7 |
| CUDA float32 | 6.0e-8 | 8.3e-7 | 1.9e-6 | 4.7e-7 |
| CPU float64 vs flyvis float64 | 0 | 1.3e-15 | 2.7e-15 | 6.2e-16 |

The gate required a relative difference of at most 1e-4 in float32, or 1e-9 in float64.
Both pass. The float64 agreement shows that the float32 gap is only round-off.
`box_eye` also matches flyvis on random frames that take the resize branch (300x300
and 305x417): bit-exact on CPU, and within 1.3e-6 on CUDA.

**Direction selectivity.** Central column, rendered edges, own float32. Each peak is
measured above the state at the last pre-roll frame. DSI = (pref - null) / (pref + null).
Image +y is down.

| Type | Edge | Preferred | Pref peak | Null peak | DSI |
|---|---|---|---|---|---|
| T4a | ON | left | 0.857 | 0.126 | 0.74 |
| T4b | ON | right | 0.729 | 0.052 | 0.87 |
| T4c | ON | up | 0.671 | 0.016 | 0.95 |
| T4d | ON | down | 0.111 | 0.000 | 0.99 |
| T5a | OFF | left | 0.474 | 0.014 | 0.94 |
| T5b | OFF | right | 0.336 | 0.008 | 0.96 |
| T5c | OFF | up | 0.480 | 0.114 | 0.62 |
| T5d | OFF | down | 0.982 | 0.218 | 0.64 |

The flyvis reference gives the same preferred directions and DSIs. Two caveats:

- T4d is selective but weak. Its central column rests at v = -0.03 and peaks at +0.08.
- T4c also responds 0.50 to rightward ON edges.

Figure: `figures/phase0_direction_selectivity.png`.

**dt sweep** (`dt_sweep.py`). A procedural numpy edge movie with the same geometry and
speed, run on CUDA float32 plus 3 s holding the last frame:

- Every dt is stable: finite, with max|v| over all nodes of 5.04 at 1/200, 5.05 at
  1/100 and 5.07 at 1/60. The smallest time constant is 19.4 ms, so the tau clamp is
  inactive even at 1/60.
- Max deviation of the central T4a-d/T5a-d traces from the 1/200 run, on the 1/60 time
  grid, relative to the largest T4/T5 swing of 0.94:
  - 1/100: 0.048 (5 %)
  - 1/60: 0.130 (14 %), largest for T4a
- DSI at 1/60 vs 1/200: 0.74/0.75, 0.85/0.87, 0.95/0.95, 0.99/0.99, 0.86/0.99, 0.90/0.96,
  0.59/0.62, 0.61/0.64. The preferred directions are unchanged.
- At 1/100 the procedural movie reproduces the DSIs of the rendered movies to two
  decimals.

**Step timing.** Median of 200 `OpticLobe.step` calls after 50 warm-up steps, float32:

- CUDA: 0.26 ms (p90 0.56), with `torch.cuda.synchronize`
- CPU, 6 threads: 0.65 ms (p90 0.71)

## Phase 1 results (2026-09-17, same machine)

```
py -3.14 python/examples/flyeye/online_gate.py   # gate a + b, flush sweep, two figures
py -3.14 python/examples/flyeye/timing.py
py -3.14 python/examples/flyeye/sanity_rig.py    # add --flush 1 for the motion AOV line
py -3.14 python/examples/flyeye/sanity_rig.py --out C:/dev/_flyeye/sanity_video --figure C:/dev/_flyeye/sanity_video/figure.png --video C:/dev/_flyeye/renders/sanity_rig_flight.mp4
                                                 # the same run as an mp4: renders, receptor columns, T4/T5 field
cd python && py -3.14 -m pytest tests/test_flyeye.py
```

### API

- `HexLattice.pixel_rc(size)`, `column_ndc(size)`, `column_rays(size, fov)` (721, 3) and
  `column_tangents(size, fov)` (721, 2, 3), in the camera frame (-Z forward, +X right,
  +Y up; image +row down). Below 391 px the pixel is in the resized 391 frame.
- `OpticLobe` steps a batch: v (B, N), receptors (B, 721) or (721,), one SpMM for all
  eyes, `fade_in` on (B, 721). Batch vs B single-eye runs: 9.5e-7 (CPU), 8.3e-7 (CUDA).
  The single-eye path is the Phase 0 code; the `spike_offline.py` rerun gives every match
  number unchanged.
- `EyeView(renderer, camera, size=403)`. Order: `render()`, `EyeView`, `render()`,
  `arm()`, `render()`, `receptors()`. Arming and reading at once gives an all-zero frame.
  `receptors()` is (721,) float32 on CUDA with no host transfer. It owns a `HexLattice`
  with CUDA centres; do not share that lattice with the readouts.
- `MotionField(lobe, gains=None)`: (..., 2, 721), x right = b - a, y up = c - d, T4 + T5,
  relu rates minus their rest rate at grey 0.5. Unit gains by default. On the Phase 0
  edges, 75 % of inner columns point within 45 deg of the true direction (median error
  14 deg). With `gains="phase0"` (1/peak) it is 66 % (29 deg), because T4d x9 flips
  off_up. on_right (+59 deg, T4c) and on_down (weak T4d) fail with unit gains; the other
  six movies are within 45 deg for at least 98 % of columns.
- `RotationReadout(size, fov, eye_rotations, method="matched"|"lstsq")`: (3,) = (wx, wy,
  wz) in the vehicle frame (-Z forward, +Y up, +X right). +wy is yaw left, +wx pitch up,
  +wz roll with the right side going up. Templates: -w x d projected on the unit column
  tangents (not the pixel Jacobian).
- `Looming(size, fov, eye_rotations, opponent=True)`: dict(value, centre, values). Radial
  outward component per column, mean per quadrant, half-wave rectified, min over the four
  quadrants, max over a 10 deg grid of centres (63 kept on the +-45 rig).
  `opponent=False` rectifies per column before the mean (see the sanity check).

### Gate

**a. Frames identical.** The 8 Phase 0 PNG movies were uploaded as BGRA uint8 CUDA
tensors and run through `eye.receptors_from_bgra` and `OpticLobe` on CUDA float32, then
compared with the own CPU float32 responses. Luma is bit-identical to PIL. Max abs:
receptors 2.4e-7, post-fade-in state 8.3e-7, responses (all nodes, frames and movies)
1.43e-6, relative 3.3e-7 (limit 1e-4). Pass.

**b. Live.** The same 8 movies were re-rendered by `render_edges.Stage` and seen by a
403 px `add_view` twin camera through `EyeView` and `OpticLobe` on CUDA, with `fade_in` on
live frame 0, at flush 16. Pass:

- Preferred directions are identical for all 8 types.
- DSI T4a-d 0.745, 0.867, 0.954, 0.994; T5a-d 0.943, 0.956, 0.617, 0.637. The largest
  change from Phase 0 is 0.0011 (limit 0.02).
- Central traces deviate by at most 0.002 to 0.006. Max abs: receptors 0.0127, responses 0.0138.

Figure: `figures/phase1_online_vs_offline.png`. The live dashed traces lie on the offline
ones in all 8 panels.

The live colour differs from the PNGs only within -2..+1 px of the edge (max 22 to 37
levels). There are two causes:

1. **RCAS runs on the primary only.** The view has no over/undershoot. Mid-crossing
   on_right: live 204 204 180 128 128, PNG 204 209 153 121 128.
2. **The Halton jitter counter is renderer-wide.** `haltonFrame_` is incremented once
   per view per GPU frame (`VulkanCoreUploads.cpp:1030`, per view at
   `VulkanCoreFrame.cpp:1656`). With 8 phases and V extra views, each camera gets
   8/gcd(V+1, 8) of them:
   - 1 view: 4 phases, a biased mean (+0.06 px shift in x)
   - 2 eyes: all 8
   - 3 views: 2
   - 7 views: 1

   Before any view exists, the primary alone reproduces the PNGs bit-exactly. Once the
   view exists, its edge pixels move by 7 to 22 levels. The mechanism is read from the
   code, not isolated experimentally.

### TAA effect

The live view on the same movies (1.69 px/frame) at 16, 3 and 1 GPU frames per
`render()`. Preferred directions are identical at every flush.

| flush | max DSI change vs Phase 0 | max central-trace deviation | receptors max abs | responses max abs | edge lag (px) | max edge error (px) |
|---|---|---|---|---|---|---|
| 16 | 0.0011 | 0.006 | 0.0127 | 0.014 | 0.02-0.05 | 0.42 |
| 3 | 0.0017 | 0.015 | 0.0175 | 0.042 | 0.17-0.19 | 0.69 |
| 1 | 0.0056 (T5c) | 0.023 (T4a, 2.7 % of peak) | 0.0208 | 0.054 | 0.22-0.26 | 0.85 |

- The Phase 0 primary had a max edge error of 0.24 px.
- At flush 1, 1608 pixels fall outside [region, grey] over 8 movies x 261 frames, a small
  TAA overshoot. There are none at flush 3 and 16.
- Figure: `figures/phase1_taa_flush.png`. Live minus offline is biphasic around each peak,
  the signature of a small time shift. It is largest at flush 1.

**Recommendation.** No C++ switch to turn TAA off on secondary views for now. At flush 1
the edge lags by about a quarter pixel (0.15 frame, 1/50 of a column). Fix the shared
Halton counter instead (a per-view index). Re-check the lag at Phase 2's fast flows: a
0.15 frame lag at 10 px/frame would be about 1.5 px (extrapolated).

### Timing (`timing.py`)

RTX 4070, 128 px primary canvas, textured scene with lit spinning meshes. Eyes are 90 deg
views yawed +-45 deg on a moving vehicle. There are 10 configurations, interleaved over 5
rounds; each value is the median of 300 iterations, with `torch.cuda.synchronize` around
every GPU stage. "render" includes the frame fence. View ms per eye = (render with eyes -
render without) / eyes.

| flush | eyes | view px | render ms | view ms per eye | receptors ms per eye | step ms | loop ms | loop Hz |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | - | 0.43 | - | - | - | 0.45 | 2233 |
| 1 | 1 | 403 | 1.02 | 0.59 | 0.57 | 0.22 | 1.82 | 551 |
| 1 | 2 | 403 | 1.42 | 0.50 | 0.56 | 0.40 (batched) | 3.08 | 324 |
| 1 | 1 | 256 | 0.74 | 0.31 | 0.62 | 0.22 | 1.58 | 632 |
| 1 | 2 | 256 | 1.02 | 0.30 | 0.59 | 0.40 (batched) | 2.61 | 383 |
| 3 | 0 | - | 0.94 | - | - | - | 0.96 | 1040 |
| 3 | 1 | 403 | 2.29 | 1.36 | 0.57 | 0.23 | 3.17 | 315 |
| 3 | 2 | 403 | 3.56 | 1.31 | 0.64 | 0.40 (batched) | 5.26 | 190 |
| 3 | 1 | 256 | 1.68 | 0.74 | 0.60 | 0.23 | 2.50 | 400 |
| 3 | 2 | 256 | 2.45 | 0.76 | 0.60 | 0.40 (batched) | 4.12 | 243 |

- `OpticLobe.step` alone: 0.223 ms for 1 eye, 0.371 ms for 2 eyes batched.
- `receptors()` (luma plus the 13x13 conv) costs about as much as rendering a 403 px view
  at flush 1.
- A 256 px view is no cheaper in `receptors()`, because it is resized to 391 first.
- Per GPU frame, a 403 px view costs about 0.45 ms and a 256 px view about 0.25 ms.

### Readouts on synthetic flow (`tests/test_flyeye.py`)

Two eyes at +-45 deg yaw, 403 px, 90 deg. Fields are -w x d (or translation inside a unit
sphere of points), projected on the column tangents.

**Rotation cross-talk.** Rows are unit rotations about x, y and z; columns are the
estimate.

- Float64, matched and lstsq: identity. The largest off-diagonal is 6.9e-16 (matched) and
  8.5e-16 (lstsq).
- Float32 field (both methods):

  ```
   1        -8.7e-11  -4.2e-8
  -2.2e-9    1        -7.9e-10
  -3.5e-8    4.5e-9    1
  ```

- The templates are orthogonal on this symmetric rig, so matched and lstsq give the same
  numbers, live as well.
- A single eye at +45 deg has pitch/roll cross-talk of -0.42 with matched filters and
  identity with lstsq.
- Forward translation reads zero rotation. Lateral (+X) translation reads wy = -0.69.

**Looming.** Each field is scaled to a mean per-column flow of 1.

| Flow | Value | Winning centre |
|---|---|---|
| Forward expansion | 0.457 | (0, 0, -1) |
| Expansion 30 deg left | 0.470 | yaw +30 |
| Uniform image flow; yaw; pitch; translation up | 0 | |
| Translation +X | 0 (0.034 with `opponent=False`) | |
| Roll | 0 (0.008 with `opponent=False`) | |

### Sanity check on a live rig (`sanity_rig.py`)

**Setup.**

- A vehicle at the centre of a closed 12 m box room. Every wall, the floor and the
  ceiling carry unlit multi-scale noise textures; the eye snapshots show texture
  everywhere.
- Two 403 px, 90 deg eyes at +-45 deg yaw. Flush 3, 100 Hz, sim time pinned.
- Motion: 1 s rest, then 75 deg/s body rotations for 1 s each with 0.5 s rests (yaw left,
  yaw right, pitch up, pitch down, roll +z, roll -z). Then an approach to the -Z wall at
  3 m/s from 6 m, stopping at 1.2 m (1/tau up to 2 /s).
- `--flush 1` also runs the renderer's motion AOV through the same templates.

Figure: `figures/phase1_readout_sanity.png`.

**Rotation.** Values are segment means after the first 0.25 s. The true mean on the
active axis is 1.222 rad/s, including the ramp.

| Manoeuvre | Circuit (wx, wy, wz), matched = lstsq | Sign | Largest off-axis / on-axis | Motion AOV (flush 1), rad/s |
|---|---|---|---|---|
| yaw left | -0.081, +0.235, +0.012 | right | 0.34 | 0, +1.221, 0 |
| yaw right | +0.045, -0.267, -0.015 | right | 0.17 | 0, -1.221, 0 |
| pitch up | +0.180, +0.003, -0.013 | right | 0.07 | +1.221, 0, 0 |
| pitch down | -0.199, -0.010, -0.006 | right | 0.05 | -1.221, 0, 0 |
| roll +z | -0.009, -0.001, +0.210 | right | 0.04 | 0, 0, +1.221 |
| roll -z | +0.021, 0.000, -0.206 | right | 0.10 | 0, 0, -1.221 |

- **Signs.** All 6 are right for the circuit (matched and lstsq) and for the motion AOV.
- **Geometry.** The AOV line reads the true rate to 0.1 %, with off-axis values at most
  3e-4. This confirms the rig geometry, the eye rotations and the template signs
  independently of the circuit.
- **Rest offset.** (+0.017, -0.008, -0.001).
- **Gain.** About 0.19 units per rad/s at 75 deg/s.
- **Flush.** Flush 1 and flush 3 circuit readouts differ by at most 0.0035 (1.3 % of the
  maximum).
- **Yaw leaks into pitch.** The leak flips with the yaw direction (rest-subtracted -0.098
  on yaw left, +0.028 on yaw right). This matches T4c answering rightward ON edges: yaw
  left gives rightward flow ahead, spurious up flow, and that reads as pitch down.
- **Approach.** It reads roll -0.036 on average (-0.08 near the end).

**Looming.**

| | rest | rotation segments, max mean (peak) | approach, first 0.3 s | approach, last 0.3 s before the stop | approach peak | corr with 1/tau |
|---|---|---|---|---|---|---|
| circuit, default (`opponent=True`) | 0.010 | 0.021 (0.059), roll +z | 0.020 | 0.117 | 0.133 | 0.81 |
| circuit, `opponent=False` | 0.054 | 0.097 (0.124), roll +z | 0.066 | 0.152 | 0.162 | 0.83 |
| motion AOV, flush 1, default | 0 | 0.002 (0.003) | 0.104 | 0.464 | 0.585 | 1.00 |

**Pass.**

- The dominant-axis sign is right on all six rotations.
- The default looming rises 6x over the approach, tracks 1/tau (r = 0.81), and stays at or
  below 0.021 on average during rotation.
- With per-column rectification (the first version), the circuit's T4/T5 field in a
  static textured scene is never zero. The rectified noise sets a 0.054 floor that
  rotation raises to 0.097, above the approach's first 0.3 s. Averaging the signed radial
  component per quadrant before rectifying removes that floor. It leaves the synthetic
  expansion values unchanged and brings translation +X and roll to 0.
- The circuit's looming saturates. It peaks at 11.32 s, 0.27 s before 1/tau does. Binned
  by 1/tau (0-0.5, 0.5-1, 1-1.3, 1.3-1.6 and 1.6-2 /s), its means are 0.01, 0.06, 0.12,
  0.13 and 0.11. Near the wall the room texture also runs out of fine detail (a 1024 px
  texture on a 12 m wall). On the true flow it stays linear (r = 1.00).

### Open issues for Phase 2

1. **Motion AOV at flush > 1.** It reads zero: the extra GPU frames repeat the scene
   state. `truth.py` must render ground truth at flush 1.
2. **Shared Halton counter.** A renderer-wide Halton index gives an eye 4 of 8 jitter
   phases when one view exists, and 2 or 1 phases with 3 or 7 views. Needs a per-view
   index in C++.
3. **TAA lag at high image speeds.** Measured only at 1.69 px/frame; re-check on the
   fast-flow scenarios before ruling out a TAA-off switch.
4. **Direction biases.** T4c answers rightward ON edges (yaw leaks into pitch at 0.17 to
   0.34) and T4d is weak. on_right and on_down fail the 45 deg test. Calibrate
   `MotionField` gains, or a per-axis correction, against the scenarios.
5. **Units.** Readout units are arbitrary (about 0.19 per rad/s at 75 deg/s for a
   TF-tuned circuit). The templates use unit tangents, not the pixel Jacobian.
6. **Receptor cost.** `receptors()` is not batched across eyes (0.57 ms per eye); a
   stacked `box_eye` would save about 0.5 ms per extra eye.
7. **Looming saturation.** The circuit's looming saturates above 1/tau of about 1 /s
   on the approach, while the true-flow looming stays linear. Check it against speed and
   texture scale before using it as a time-to-contact signal.
8. **`render_edges.Stage` oddities.** It reports `volumetric_fog` True after setting it
   False, and prints "FSR 3.1 upscaler active" while `renderer.fsr` reads False. Neither
   affected any result here.

## Receptor input convention

This is what the pretrained models saw, recorded in the npz key `input_convention`.

1. **Luminance.** Take the 8-bit display-encoded sRGB frame, compute its PIL `convert('L')`
   luma (ITU-R 601-2, integer formula), and divide by 255. Do not linearise, take a log
   or subtract the mean. Grey 0.5 is the rest value. `lattice.luminance()` reproduces
   PIL bit for bit. threepp `FrameTensors` colour is BGRA, so reorder the channels first.
2. **BoxEye.** Take the 13x13 box mean (zero pad 6, /169), sampled at
   (H//2 + trunc(13(u + v/2)), W//2 + 13v). Frames smaller than 391 px are resized
   first. The same value goes to R1..R8 of that column, with no gain or offset.
3. **Start-up.** Begin at v = bias, then run `fade_in`: int(1/dt) steps that ramp the
   contrast of the first frame up from grey. After that, run one `step` per frame, with
   the frame held for dt.

The model was trained at dt = 1/50 on 24 fps Sintel footage, with contrast, brightness
and noise augmentation.

## Angular scale

- One column is 13 px, and the lattice spans 391 px (31 columns across).
- At 403 px and a 90 deg FOV that averages to about 2.9 deg per column. With the
  perspective projection it is 3.7 deg at the image centre and less toward the edges.
- The Phase 0 speed of 13 columns/s is 169 px/s, about 48 deg/s at the centre.
- flyvis's own HexEye stimuli use 5.8 deg per ommatidium. A 90 deg, 403 px eye therefore
  sees the world at roughly twice the fly's angular resolution, and angular speeds scale
  the same way.

## Regenerating the npz and the reference

Both steps run in a throwaway Python 3.12 venv, because flyvis needs Python below 3.13.
The venv pins flyvis 1.2.0, torch 2.14.0+cpu, datamate 1.0.0 and numpy 2.5.3. Download
the pretrained models with `flyvis download-pretrained` (only
`results_pretrained_models.zip`, 3.4 MB, is needed), and set `FLYVIS_ROOT_DIR` to the
data root.

```
python tools/export_flyvis.py                          # -> data/flyeye_model.npz (0.89 MB)
python tools/flyvis_reference.py <png_dir> 0.01 out.npz [--dtype float64]
```

On Windows, datamate 1.0.0 fails the first time it builds the connectome cache
(WinError 32). Both tools patch this at import. In the threepp interpreter, with no
flyvis installed:

```
py -3.14 python/examples/flyeye/render_edges.py --out C:/dev/_flyeye/stimuli
py -3.14 python/examples/flyeye/spike_offline.py      # reads C:/dev/_flyeye/{stimuli,reference}
py -3.14 python/examples/flyeye/dt_sweep.py
```

## Open questions (plan section 6)

1. **Minimum stable dt.** 1/60, 1/100 and 1/200 are all stable. 1/100 stays within 5 %
   of 1/200.
2. **Lens distortion on `addView` views.** No. Secondary views get no lens or sensor
   stage (`VulkanRenderer.hpp:230-233`, `VulkanCoreFrame.cpp:1683-1686`). The zero-copy
   interop path skips the warp even on the primary view. Any ommatidial warp has to be
   done in torch on pinhole views.
3. **Render cost of a 403 px secondary view.** Not measured; Phase 1.
4. **MaleCNS column coordinates.** Open; Phase 4.
5. **Input normalisation.** Luma/255 of display-encoded sRGB (see above).
6. **Parameter storage and license.** `syn_strength` is stored per edge type (604
   values). `syn_count` is per (type pair, du, dv), stored as log(mean). The npz holds
   both the expanded weights and the shared tables. The download ships no license file;
   the weights are distributed only through the MIT-licensed flyvis project.

Facts about secondary views for Phase 1:

- `FrameTensors(renderer, view=handle)` works on secondary views.
- The Motion AOV is tracked per view. It reads zero on a view's first frame and after
  `setViewCamera`.
- TAA always runs on secondary views and cannot be switched off from Python. Use slow
  stimuli, or check against the pre-TAA AOVs.
- Secondary views read the primary's auto exposure. Set `auto_exposure = False`, a fixed
  exposure, `NoToneMapping` and bloom 0.
- Colour is 8-bit display-encoded sRGB, which is exactly what flyvis expects.

## Attribution and license

- **flyvis.** The optic-lobe network and its pretrained parameters (ensemble member
  `flow/0000/000`) come from flyvis 1.2.0, https://github.com/TuragaLab/flyvis, MIT
  License, Copyright (c) 2023 Janne K. Lappalainen, Fabian D. Tschopp, Mason McGill,
  Jakob H. Macke, Srinivas C. Turaga. They were exported to `flyeye_model.npz` with
  `tools/export_flyvis.py`. Cite: Lappalainen, J. K., Tschopp, F. D., Prakhya, S.,
  McGill, M., Nern, A., Shinomiya, K., Takemura, S., Gruntman, E., Macke, J. H., and
  Turaga, S. C. Connectome-constrained networks predict neural activity across the fly
  visual system. Nature 634, 1132-1140 (2024). doi:10.1038/s41586-024-07939-3.
- **Connectome.** Cell types, synapse counts and signs come from `fib25-fib19_v2.2.json`
  as shipped with flyvis. It derives from the Drosophila medulla FIB-SEM reconstructions:
  - Takemura, S. et al. Synaptic circuits and their variations within different columns
    in the visual system of Drosophila. PNAS 112, 13711-13716 (2015).
    doi:10.1073/pnas.1509820112 (FIB-25).
  - Takemura, S. et al. The comprehensive connectome of a neural substrate for 'ON'
    motion detection in Drosophila. eLife 6, e24394 (2017). doi:10.7554/eLife.24394
    (FIB-19).
  - Shinomiya, K. et al. Comparisons between the ON- and OFF-edge motion pathways in the
    Drosophila brain. eLife 8, e40025 (2019). doi:10.7554/eLife.40025 (FIB-19, T5
    pathway).
  - Related earlier reconstruction: Takemura, S. et al. A visual motion detection circuit
    suggested by Drosophila connectomics. Nature 500, 175-181 (2013).
    doi:10.1038/nature12450.

The flyvis copyright line was checked verbatim against the repository's `license` file,
and the DOIs, volumes and pages against Crossref. I have not checked which of these
connectome papers the Methods of Lappalainen et al. 2024 cite.
