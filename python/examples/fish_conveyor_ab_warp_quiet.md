# Newton VBD vs PhysX 5 deformable volumes, on the same fish conveyor

_2026-08-24 22:41 - 12 runs, 8 s of settled conveying each, headless, one subprocess per run._

Every arm shares the scans, the auto-orientation, the decimated render surface, the barycentric bind, the belt, the roller pitch, the rails, the spawn layout and seed, the GPU skinning path, the headless renderer and the quality metrics. The solver -- and in one arm the tetrahedral mesh -- is what differs.

## Results

Speed is the median over seeds; every quality number is the WORST over seeds, because a solver that inverts a tet on one layout in three has a failure mode rather than a good average. Penetration is measured between cage vertices of different fish at a fixed 30 mm thickness, so it compares across cages.

| arm | fish | ms/frame | fps | realtime | tets | inverted | volume | stretch | droop | pen pairs | worst pen | convey | on belt | VRAM |
|---|---:|---:|---:|:--:|---:|---:|---|---:|---:|---:|---:|---:|:--:|---:|
| warp-corot | 60 | 2.6 | 386 | yes | 5175 | 0 | 1.00-1.00 | 1.13 | 0.38 | 0 | 0.0 mm | +0.479 of 0.50 | 60/60 | 1626 MB |
| warp-corot | 100 | 3.5 | 287 | yes | 8625 | 0 | 1.00-1.00 | 1.21 | 0.37 | 0 | 0.0 mm | +0.478 of 0.50 | 100/100 | 1639 MB |
| physx | 60 | 13.9 | 72 | yes | 20220 | 14 | 0.99-1.00 | 1.15 | 0.64 | 2133 | 28.4 mm | +0.489 of 0.50 | 60/60 | 3048 MB |
| physx | 100 | 18.9 | 53 | no | 33700 | 22 | 0.99-1.00 | 1.22 | 0.31 | 3568 | 29.0 mm | +0.488 of 0.50 | 100/100 | 3061 MB |

## Protocol 1 - equal budget (16.7 ms/frame at 60 Hz)

- **warp-corot** holds 60 Hz to **100 fish** (3.5 ms, 8625 tets); at that count 0 inverted tets, 1.21x worst stretch, 0.0 mm worst penetration.
- **physx** holds 60 Hz to **60 fish** (13.9 ms, 20220 tets); at that count 14 inverted tets, 1.15x worst stretch, 28.4 mm worst penetration.
  - misses the budget at: 100 fish (18.9 ms)

## Protocol 2 - acceptance bar at 60 fish

| arm | no inverted tets | worst tet edge stretch < 2.0x | worst cross-fish penetration < 10 mm | every fish still on the belt | conveying >= 0.8 x belt speed | verdict |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| warp-corot | PASS | PASS | PASS | PASS | PASS | **PASS** |
| physx | **FAIL** | PASS | **FAIL** | PASS | PASS | **FAIL** |

- **warp-corot** clears the bar at 2.6 ms/frame (386 fps).
- **physx** fails on inverted, penetration: inverted 14, stretch 1.15x, worst penetration 28.4 mm, 60/60 on belt, conveying +0.489 m/s.

## Frames (60 fish, same camera, same sim time)

- **warp-corot**: `C:\dev\threepp\python\examples\fish_conveyor_ab_warp-corot.png`, 1:1 crop `C:\dev\threepp\python\examples\fish_conveyor_ab_warp-corot_crop.png`
- **physx**: `C:\dev\threepp\python\examples\fish_conveyor_ab_physx.png`, 1:1 crop `C:\dev\threepp\python\examples\fish_conveyor_ab_physx_crop.png`

## Where this is still not apples-to-apples

- **Different tetrahedral meshes in two of the three arms.** Newton carves a voxel cage; PhysX cooks a conforming collision mesh plus a voxelised simulation mesh. Volume ratio and edge stretch are then the same measurement on different meshes. That is exactly what `newton-physx-tets` removes -- read it against `physx` for a mesh-controlled comparison and against `newton-voxel` for the cost of the mesh alone.
- **PhysX simulates two meshes per fish, Newton one.** The tet counts in the table are the collision meshes. PhysX additionally integrates a voxel simulation mesh and skins the collision mesh to it; that work is in its milliseconds but not in its tet count.
- **Different substep counts, on purpose.** Each engine runs at its own shipped rate: Newton 4 substeps x 4 VBD iterations, PhysX 1 substep x 20 solver iterations. Equalising substeps would favour whichever solver the chosen number happened to suit. Protocol 1 is the honest form of the comparison: fixed wall-clock budget, whatever each engine spends it on.
- **Skinning is zero-copy for Newton and a readback for PhysX.** Newton's cage vertices are already Warp arrays; PhysX's live in GPU memory it owns, with no bridge into a foreign CUDA context, so each frame copies them device->host and back up. It is a real cost of using PhysX from this pipeline, not a measurement artefact, and it is reported separately as `readback_ms` in the raw JSON so it can be subtracted if the question is only about the solver.
- **Rollers are cylinders in Newton and capsules in PhysX** (PhysX has no cylinder primitive). Over the belt the two surfaces are identical; the capsule's hemispherical caps sit outside rails the fish cannot pass. The kinematic drive differs though: PhysX needs a fresh setKinematicTarget per substep, ~300 bound calls, which is inside its frame time.
- **No per-fish scale jitter, on either side.** PhysX's cook cache keys on the source geometry and applies rotation and translation only, so a per-fish scale would mean one cook per fish. Both arms run with it off; yaw and position jitter are unchanged.
- **The same seed does not give the same pile twice.** Both solvers accumulate contacts with atomics, so the summation order varies run to run, and a heap of sixty soft bodies over eight seconds amplifies a last-bit difference into a visibly different arrangement. Measured: newton-voxel at 16 fish, seed 7, reported 3.03x worst stretch on one run and 2.18x on a repeat of the identical command, at 7.26 and 7.27 ms. Milliseconds are stable; WORST-case quality numbers are a sample, which is what the multi-seed worst-case aggregation is for. Do not read a 10% difference in a stretch column as a difference between engines.
- **One machine, one driver, one session.** Every number is an RTX 4070 on the day. The ordering should travel; the absolute milliseconds will not.
- **Matrix reduced.** seeds [7, 11, 13] and counts [60, 100] rather than the full 3 seeds x (16, 40, 60, 100), for time. Quality columns are a worst-case over 3 layout(s), not 3.

## Raw

```json
{"engine": "warp-corot", "count": 60, "seed": 7, "ms_frame": 3.1618818749848288, "fps": 316.26734948939173, "tets": 5175, "cage_verts": 3435, "convey": 0.4786651857978706, "belt": 0.5, "onbelt": 60, "vram": 1626.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.998067598413794, "vol_hi": 1.0023490443663778, "stretch": 1.1139183079469357, "droop": 0.638388637405486, "pen_pairs": 0, "pen_worst_mm": 0.0, "shot": "C:\\dev\\threepp\\python\\examples\\fish_conveyor_ab_warp-corot.png", "wall_s": 6.625428500003181, "exit": 0}
{"engine": "physx", "count": 60, "seed": 7, "ms_frame": 14.61456979165329, "fps": 68.424867393026, "tets": 20220, "cage_verts": 7950, "convey": 0.4887325009833901, "belt": 0.5, "onbelt": 60, "vram": 3048.1171875, "seconds": 8.0, "finite": true, "inverted": 3, "vol_lo": 0.9924494280179922, "vol_hi": 0.9992559770322956, "stretch": 1.0931039270133804, "droop": 0.7093179450116613, "pen_pairs": 2133, "pen_worst_mm": 28.44884619116783, "shot": "C:\\dev\\threepp\\python\\examples\\fish_conveyor_ab_physx.png", "readback_ms": 2.081385087538429, "wall_s": 13.561686300003203, "exit": 0}
{"engine": "warp-corot", "count": 60, "seed": 11, "ms_frame": 2.5799866666905777, "fps": 387.5989023163164, "tets": 5175, "cage_verts": 3435, "convey": 0.47853633438348114, "belt": 0.5, "onbelt": 60, "vram": 1626.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9978112435373895, "vol_hi": 1.0018711215627834, "stretch": 1.1263096829766492, "droop": 0.44065243922462677, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 6.306849299988244, "exit": 0}
{"engine": "physx", "count": 60, "seed": 11, "ms_frame": 12.657306458337795, "fps": 79.00575081211424, "tets": 20220, "cage_verts": 7950, "convey": 0.4903473085126267, "belt": 0.5, "onbelt": 60, "vram": 3048.1171875, "seconds": 8.0, "finite": true, "inverted": 14, "vol_lo": 0.9922674983526827, "vol_hi": 0.999950237479811, "stretch": 1.1540736986397027, "droop": 0.8139497046348207, "pen_pairs": 1927, "pen_worst_mm": 27.833718806505203, "readback_ms": 1.5330982459872438, "wall_s": 13.010761300014565, "exit": 0}
{"engine": "warp-corot", "count": 60, "seed": 13, "ms_frame": 2.5892725000327723, "fps": 386.20886754381513, "tets": 5175, "cage_verts": 3435, "convey": 0.47842273663001933, "belt": 0.5, "onbelt": 60, "vram": 1626.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9970234468624481, "vol_hi": 1.0024591237639728, "stretch": 1.0877493695451579, "droop": 0.3811957123675287, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 6.144243299990194, "exit": 0}
{"engine": "physx", "count": 60, "seed": 13, "ms_frame": 13.875684583338929, "fps": 72.06851625906361, "tets": 20220, "cage_verts": 7950, "convey": 0.4890995114099824, "belt": 0.5, "onbelt": 60, "vram": 3048.1171875, "seconds": 8.0, "finite": true, "inverted": 14, "vol_lo": 0.994112890629539, "vol_hi": 0.9998888396662065, "stretch": 1.0780406036234558, "droop": 0.6389882730427662, "pen_pairs": 2047, "pen_worst_mm": 27.211323380470276, "readback_ms": 1.788795963920277, "wall_s": 13.36720609999611, "exit": 0}
{"engine": "warp-corot", "count": 100, "seed": 7, "ms_frame": 3.4775562500120336, "fps": 287.5582530107283, "tets": 8625, "cage_verts": 5725, "convey": 0.4779504018474339, "belt": 0.5, "onbelt": 100, "vram": 1639.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9976636667742795, "vol_hi": 1.0019746521410393, "stretch": 1.0870581783690079, "droop": 0.37219603842896143, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 6.684963899984723, "exit": 0}
{"engine": "physx", "count": 100, "seed": 7, "ms_frame": 18.882881250040857, "fps": 52.95801984656533, "tets": 33700, "cage_verts": 13250, "convey": 0.4886711402607043, "belt": 0.5, "onbelt": 100, "vram": 3061.0546875, "seconds": 8.0, "finite": true, "inverted": 21, "vol_lo": 0.9925746361418606, "vol_hi": 0.9998767253011657, "stretch": 1.0748363581490905, "droop": 0.7783137372480369, "pen_pairs": 3568, "pen_worst_mm": 28.954077512025833, "readback_ms": 3.4623928075010904, "wall_s": 15.969822600018233, "exit": 0}
{"engine": "warp-corot", "count": 100, "seed": 11, "ms_frame": 3.4783208333465154, "fps": 287.49504370414655, "tets": 8625, "cage_verts": 5725, "convey": 0.477948258706695, "belt": 0.5, "onbelt": 100, "vram": 1639.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9975556125524806, "vol_hi": 1.0012958084667727, "stretch": 1.2072188029112458, "droop": 0.4950793551541826, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 6.605938399996376, "exit": 0}
{"engine": "physx", "count": 100, "seed": 11, "ms_frame": 17.89468312496562, "fps": 55.88252069157113, "tets": 33700, "cage_verts": 13250, "convey": 0.4877103930765982, "belt": 0.5, "onbelt": 100, "vram": 3061.0546875, "seconds": 8.0, "finite": true, "inverted": 10, "vol_lo": 0.9925799785016041, "vol_hi": 0.999989273352093, "stretch": 1.0882873411980194, "droop": 0.30713210710005656, "pen_pairs": 3421, "pen_worst_mm": 28.10015343129635, "readback_ms": 3.4726878942660333, "wall_s": 15.76792949999799, "exit": 0}
{"engine": "warp-corot", "count": 100, "seed": 13, "ms_frame": 3.5349181249936614, "fps": 282.8919835312432, "tets": 8625, "cage_verts": 5725, "convey": 0.4805575511364858, "belt": 0.5, "onbelt": 100, "vram": 1639.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9981332897179116, "vol_hi": 1.0012624427187957, "stretch": 1.110064352409405, "droop": 0.6851196658900819, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 7.0284994000103325, "exit": 0}
{"engine": "physx", "count": 100, "seed": 13, "ms_frame": 18.97835666671502, "fps": 52.691601151844665, "tets": 33700, "cage_verts": 13250, "convey": 0.48777215053863815, "belt": 0.5, "onbelt": 100, "vram": 3061.0546875, "seconds": 8.0, "finite": true, "inverted": 22, "vol_lo": 0.9867695154587925, "vol_hi": 1.0000281797884643, "stretch": 1.2227788116609972, "droop": 0.6409057937139928, "pen_pairs": 3533, "pen_worst_mm": 28.83066236972809, "readback_ms": 3.51317385997901, "wall_s": 15.976423499989323, "exit": 0}
```
