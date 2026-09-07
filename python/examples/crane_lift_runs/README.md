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
