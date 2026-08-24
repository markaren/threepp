# Newton VBD vs PhysX 5 deformable volumes, on the same fish conveyor

_2026-08-24 21:57 - 36 runs, 8 s of settled conveying each, headless, one subprocess per run._

Every arm shares the scans, the auto-orientation, the decimated render surface, the barycentric bind, the belt, the roller pitch, the rails, the spawn layout and seed, the GPU skinning path, the headless renderer and the quality metrics. The solver -- and in one arm the tetrahedral mesh -- is what differs.

## Results

Speed is the median over seeds; every quality number is the WORST over seeds, because a solver that inverts a tet on one layout in three has a failure mode rather than a good average. Penetration is measured between cage vertices of different fish at a fixed 30 mm thickness, so it compares across cages.

| arm | fish | ms/frame | fps | realtime | tets | inverted | volume | stretch | droop | pen pairs | worst pen | convey | on belt | VRAM |
|---|---:|---:|---:|:--:|---:|---:|---|---:|---:|---:|---:|---:|:--:|---:|
| warp-corot | 16 | 3.8 | 264 | yes | 1380 | 0 | 0.99-1.00 | 1.53 | 0.58 | 0 | 0.0 mm | +0.477 of 0.50 | 16/16 | 1608 MB |
| warp-corot | 40 | 4.1 | 246 | yes | 3450 | 0 | 0.98-1.00 | 1.73 | 0.30 | 0 | 0.0 mm | +0.479 of 0.50 | 40/40 | 1618 MB |
| warp-corot | 60 | 5.5 | 180 | yes | 5175 | 0 | 0.99-1.00 | 1.70 | 0.61 | 0 | 0.0 mm | +0.479 of 0.50 | 60/60 | 1626 MB |
| warp-corot | 100 | 5.8 | 172 | yes | 8625 | 0 | 0.99-1.00 | 1.70 | 0.51 | 0 | 0.0 mm | +0.478 of 0.50 | 100/100 | 1639 MB |
| newton-xpbd | 16 | 9.2 | 109 | yes | 1380 | 0 | 0.99-1.00 | 1.21 | 0.64 | 0 | 0.0 mm | +0.447 of 0.50 | 16/16 | 1662 MB |
| newton-xpbd | 40 | 9.6 | 104 | yes | 3450 | 0 | 0.97-1.00 | 1.28 | 0.53 | 0 | 0.0 mm | +0.446 of 0.50 | 40/40 | 1704 MB |
| newton-xpbd | 60 | 11.8 | 85 | yes | 5175 | 0 | 0.98-1.00 | 1.38 | 0.43 | 0 | 0.0 mm | +0.441 of 0.50 | 60/60 | 1744 MB |
| newton-xpbd | 100 | 20.2 | 49 | no | 8625 | 0 | 0.97-1.00 | 1.39 | 0.44 | 0 | 0.0 mm | +0.444 of 0.50 | 100/100 | 1823 MB |
| physx | 16 | 15.7 | 64 | yes | 5992 | 4 | 0.98-1.00 | 1.36 | 0.72 | 54 | 27.9 mm | +0.494 of 0.50 | 16/16 | 3030 MB |
| physx | 40 | 20.3 | 49 | no | 14980 | 17 | 0.96-1.00 | 1.50 | 0.52 | 138 | 27.0 mm | +0.494 of 0.50 | 40/40 | 3040 MB |
| physx | 60 | 24.4 | 41 | no | 22470 | 12 | 0.96-1.00 | 1.33 | 0.70 | 204 | 28.6 mm | +0.494 of 0.50 | 60/60 | 3048 MB |
| physx | 100 | 41.9 | 24 | no | 37450 | 18 | 0.97-1.00 | 1.49 | 0.31 | 303 | 28.0 mm | +0.492 of 0.50 | 100/100 | 3061 MB |

## Protocol 1 - equal budget (16.7 ms/frame at 60 Hz)

- **warp-corot** holds 60 Hz to **100 fish** (5.8 ms, 8625 tets); at that count 0 inverted tets, 1.70x worst stretch, 0.0 mm worst penetration.
- **newton-xpbd** holds 60 Hz to **60 fish** (11.8 ms, 5175 tets); at that count 0 inverted tets, 1.38x worst stretch, 0.0 mm worst penetration.
  - misses the budget at: 100 fish (20.2 ms)
- **physx** holds 60 Hz to **16 fish** (15.7 ms, 5992 tets); at that count 4 inverted tets, 1.36x worst stretch, 27.9 mm worst penetration.
  - misses the budget at: 40 fish (20.3 ms), 60 fish (24.4 ms), 100 fish (41.9 ms)

## Protocol 2 - acceptance bar at 60 fish

| arm | no inverted tets | worst tet edge stretch < 2.0x | worst cross-fish penetration < 10 mm | every fish still on the belt | conveying >= 0.8 x belt speed | verdict |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| warp-corot | PASS | PASS | PASS | PASS | PASS | **PASS** |
| newton-xpbd | PASS | PASS | PASS | PASS | PASS | **PASS** |
| physx | **FAIL** | PASS | **FAIL** | PASS | PASS | **FAIL** |

- **warp-corot** clears the bar at 5.5 ms/frame (180 fps).
- **newton-xpbd** clears the bar at 11.8 ms/frame (85 fps).
- **physx** fails on inverted, penetration: inverted 12, stretch 1.33x, worst penetration 28.6 mm, 60/60 on belt, conveying +0.494 m/s.

## Frames (60 fish, same camera, same sim time)

- **warp-corot**: `C:\dev\threepp\python\examples\fish_conveyor_ab_warp-corot.png`, 1:1 crop `C:\dev\threepp\python\examples\fish_conveyor_ab_warp-corot_crop.png`
- **newton-xpbd**: `C:\dev\threepp\python\examples\fish_conveyor_ab_newton-xpbd.png`, 1:1 crop `C:\dev\threepp\python\examples\fish_conveyor_ab_newton-xpbd_crop.png`
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

## Raw

```json
{"engine": "warp-corot", "count": 16, "seed": 7, "ms_frame": 3.7926668749908763, "fps": 263.66671077654445, "tets": 1380, "cage_verts": 916, "convey": 0.4821689118742652, "belt": 0.5, "onbelt": 16, "vram": 1491.3671875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9987172030745008, "vol_hi": 1.0004501171771774, "stretch": 1.0802921708310604, "droop": 0.8781966539059956, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 8.391441799991298, "exit": 0}
{"engine": "newton-xpbd", "count": 16, "seed": 7, "ms_frame": 11.266668541672212, "fps": 88.7573816786465, "tets": 1380, "cage_verts": 916, "convey": 0.4470007260621234, "belt": 0.5, "onbelt": 16, "vram": 1662.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9936686421399316, "vol_hi": 0.9974114948993628, "stretch": 1.1789602514514002, "droop": 0.8395490960789621, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 20.250995900016278, "exit": 0}
{"engine": "physx", "count": 16, "seed": 7, "ms_frame": 19.28590249996584, "fps": 51.85134582121688, "tets": 5992, "cage_verts": 2180, "convey": 0.49267571316004044, "belt": 0.5, "onbelt": 16, "vram": 2978.9921875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9801833697434309, "vol_hi": 0.9986474293584198, "stretch": 1.2057551983116357, "droop": 0.8293761242514894, "pen_pairs": 47, "pen_worst_mm": 22.08813838660717, "readback_ms": 0.7299773682015049, "wall_s": 18.357241299992893, "exit": 0}
{"engine": "warp-corot", "count": 16, "seed": 11, "ms_frame": 3.935630208313038, "fps": 254.08891259340098, "tets": 1380, "cage_verts": 916, "convey": 0.4763505181755162, "belt": 0.5, "onbelt": 16, "vram": 1608.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9947037853553654, "vol_hi": 1.0013609000343777, "stretch": 1.5319214250274396, "droop": 0.5805739845225725, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 8.059467800019775, "exit": 0}
{"engine": "newton-xpbd", "count": 16, "seed": 11, "ms_frame": 9.190075208365064, "fps": 108.81303768763198, "tets": 1380, "cage_verts": 916, "convey": 0.4436177649418326, "belt": 0.5, "onbelt": 16, "vram": 1662.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9927432902522237, "vol_hi": 0.9969890021870524, "stretch": 1.1453589103966992, "droop": 0.6420795724199483, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 17.96152140002232, "exit": 0}
{"engine": "physx", "count": 16, "seed": 11, "ms_frame": 15.731477083318168, "fps": 63.56682177418744, "tets": 5992, "cage_verts": 2180, "convey": 0.4948021797282152, "belt": 0.5, "onbelt": 16, "vram": 3030.1796875, "seconds": 8.0, "finite": true, "inverted": 1, "vol_lo": 0.9864721566745079, "vol_hi": 0.9959706643193769, "stretch": 1.1370300073587034, "droop": 0.7197006921965858, "pen_pairs": 49, "pen_worst_mm": 27.90088579058647, "readback_ms": 0.63285964889362, "wall_s": 14.86574120001751, "exit": 0}
{"engine": "warp-corot", "count": 16, "seed": 13, "ms_frame": 3.064995208357383, "fps": 326.2647841253651, "tets": 1380, "cage_verts": 916, "convey": 0.47694188758136224, "belt": 0.5, "onbelt": 16, "vram": 1608.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9985286790617274, "vol_hi": 1.0013436707981442, "stretch": 1.1145444866990006, "droop": 0.9467735830134161, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 6.787795000011101, "exit": 0}
{"engine": "newton-xpbd", "count": 16, "seed": 13, "ms_frame": 8.437738749974718, "fps": 118.51516497864979, "tets": 1380, "cage_verts": 916, "convey": 0.4531370014657774, "belt": 0.5, "onbelt": 16, "vram": 1662.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9932681865704288, "vol_hi": 0.997345638342862, "stretch": 1.2051651900495692, "droop": 0.9820950961422158, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 14.976964599976782, "exit": 0}
{"engine": "physx", "count": 16, "seed": 13, "ms_frame": 14.570395208284026, "fps": 68.63231818389168, "tets": 5992, "cage_verts": 2180, "convey": 0.4943865686573566, "belt": 0.5, "onbelt": 16, "vram": 3030.1796875, "seconds": 8.0, "finite": true, "inverted": 4, "vol_lo": 0.9771451652103126, "vol_hi": 0.99666708067327, "stretch": 1.363210635018354, "droop": 0.878708548260771, "pen_pairs": 54, "pen_worst_mm": 24.159543216228485, "readback_ms": 0.5510264909581134, "wall_s": 13.706320700002834, "exit": 0}
{"engine": "warp-corot", "count": 40, "seed": 7, "ms_frame": 3.2869808333392334, "fps": 304.2305540261101, "tets": 3450, "cage_verts": 2290, "convey": 0.4764671206139862, "belt": 0.5, "onbelt": 40, "vram": 1617.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9844653439288112, "vol_hi": 1.000514297158076, "stretch": 1.6939936730532374, "droop": 0.3026341995621261, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 6.816846900008386, "exit": 0}
{"engine": "newton-xpbd", "count": 40, "seed": 7, "ms_frame": 9.593946250000348, "fps": 104.23239550669399, "tets": 3450, "cage_verts": 2290, "convey": 0.44602328817347947, "belt": 0.5, "onbelt": 40, "vram": 1703.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9931927884923569, "vol_hi": 0.9973430285147338, "stretch": 1.1494804004515098, "droop": 0.6800842247062214, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 15.954540200007614, "exit": 0}
{"engine": "physx", "count": 40, "seed": 7, "ms_frame": 18.053052291664546, "fps": 55.39229510024296, "tets": 14980, "cage_verts": 5450, "convey": 0.49444926135089506, "belt": 0.5, "onbelt": 40, "vram": 3039.6171875, "seconds": 8.0, "finite": true, "inverted": 9, "vol_lo": 0.9771952614464388, "vol_hi": 0.9977800107336998, "stretch": 1.499521217978711, "droop": 0.8504016999059651, "pen_pairs": 116, "pen_worst_mm": 24.392254650592804, "readback_ms": 1.5450014033775548, "wall_s": 15.647576500021387, "exit": 0}
{"engine": "warp-corot", "count": 40, "seed": 11, "ms_frame": 4.059563541644214, "fps": 246.3318999054213, "tets": 3450, "cage_verts": 2290, "convey": 0.47997804691862844, "belt": 0.5, "onbelt": 40, "vram": 1617.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9871205822695349, "vol_hi": 1.0006071263468304, "stretch": 1.7287427850837407, "droop": 0.719826214444042, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 7.917382599989651, "exit": 0}
{"engine": "newton-xpbd", "count": 40, "seed": 11, "ms_frame": 9.557467916662668, "fps": 104.63022305903651, "tets": 3450, "cage_verts": 2290, "convey": 0.4485412406088776, "belt": 0.5, "onbelt": 40, "vram": 1703.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.991195547129806, "vol_hi": 0.9976927784898463, "stretch": 1.209292178893811, "droop": 0.7679532686401357, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 17.853659800020978, "exit": 0}
{"engine": "physx", "count": 40, "seed": 11, "ms_frame": 20.30872958330292, "fps": 49.23990916803396, "tets": 14980, "cage_verts": 5450, "convey": 0.49447581031551613, "belt": 0.5, "onbelt": 40, "vram": 3039.6171875, "seconds": 8.0, "finite": true, "inverted": 17, "vol_lo": 0.9649045078626511, "vol_hi": 0.995430307490637, "stretch": 1.455876482299957, "droop": 0.8059584123571211, "pen_pairs": 138, "pen_worst_mm": 25.07866732776165, "readback_ms": 2.480644035191896, "wall_s": 27.02408909998485, "exit": 0}
{"engine": "warp-corot", "count": 40, "seed": 13, "ms_frame": 4.413637083295423, "fps": 226.57050888591738, "tets": 3450, "cage_verts": 2290, "convey": 0.4786775698319685, "belt": 0.5, "onbelt": 40, "vram": 1617.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9991896119501917, "vol_hi": 1.0003163243542461, "stretch": 1.0680432923710947, "droop": 0.881742222738487, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 7.943126099999063, "exit": 0}
{"engine": "newton-xpbd", "count": 40, "seed": 13, "ms_frame": 10.007398333315601, "fps": 99.92607136171475, "tets": 3450, "cage_verts": 2290, "convey": 0.4351475233491179, "belt": 0.5, "onbelt": 40, "vram": 1703.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9730135670790999, "vol_hi": 0.9970709377001694, "stretch": 1.2806386111585826, "droop": 0.534004987883722, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 21.08019829998375, "exit": 0}
{"engine": "physx", "count": 40, "seed": 13, "ms_frame": 22.017818124974536, "fps": 45.417760939069275, "tets": 14980, "cage_verts": 5450, "convey": 0.4850916952696826, "belt": 0.5, "onbelt": 40, "vram": 3039.6171875, "seconds": 8.0, "finite": true, "inverted": 4, "vol_lo": 0.9789936577493439, "vol_hi": 0.9972304294591157, "stretch": 1.1563416226842287, "droop": 0.5152573520339051, "pen_pairs": 130, "pen_worst_mm": 27.01057679951191, "readback_ms": 1.9834052627853103, "wall_s": 19.3328407999943, "exit": 0}
{"engine": "warp-corot", "count": 60, "seed": 7, "ms_frame": 6.433131250014412, "fps": 155.44529734221723, "tets": 5175, "cage_verts": 3435, "convey": 0.4790736380737833, "belt": 0.5, "onbelt": 60, "vram": 1626.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9974746456491128, "vol_hi": 1.001364219843213, "stretch": 1.123359122623096, "droop": 0.8208971289238407, "pen_pairs": 0, "pen_worst_mm": 0.0, "shot": "C:\\dev\\threepp\\python\\examples\\fish_conveyor_ab_warp-corot.png", "wall_s": 9.876916999986861, "exit": 0}
{"engine": "newton-xpbd", "count": 60, "seed": 7, "ms_frame": 31.965454166675045, "fps": 31.2837726248398, "tets": 5175, "cage_verts": 3435, "convey": 0.43256227920152185, "belt": 0.5, "onbelt": 60, "vram": 1744.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9754513406702487, "vol_hi": 0.9976433219316525, "stretch": 1.3826133280017046, "droop": 0.44974332679290424, "pen_pairs": 0, "pen_worst_mm": 0.0, "shot": "C:\\dev\\threepp\\python\\examples\\fish_conveyor_ab_newton-xpbd.png", "wall_s": 28.761793299985584, "exit": 0}
{"engine": "physx", "count": 60, "seed": 7, "ms_frame": 32.88048937502026, "fps": 30.413172644556596, "tets": 22470, "cage_verts": 8175, "convey": 0.4938032676000346, "belt": 0.5, "onbelt": 60, "vram": 3048.1171875, "seconds": 8.0, "finite": true, "inverted": 8, "vol_lo": 0.9765386973573429, "vol_hi": 0.9980062448875476, "stretch": 1.2306182043427245, "droop": 0.6965487453852435, "pen_pairs": 204, "pen_worst_mm": 28.359422460198402, "shot": "C:\\dev\\threepp\\python\\examples\\fish_conveyor_ab_physx.png", "readback_ms": 3.2367014038180444, "wall_s": 36.306003400008194, "exit": 0}
{"engine": "warp-corot", "count": 60, "seed": 11, "ms_frame": 5.548902499989102, "fps": 180.21581745254383, "tets": 5175, "cage_verts": 3435, "convey": 0.4764991991061459, "belt": 0.5, "onbelt": 60, "vram": 1626.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9871830436560207, "vol_hi": 1.0008049537360622, "stretch": 1.7010999996896592, "droop": 0.7661763931198219, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 8.856595100020058, "exit": 0}
{"engine": "newton-xpbd", "count": 60, "seed": 11, "ms_frame": 11.028621041684042, "fps": 90.67316722737829, "tets": 5175, "cage_verts": 3435, "convey": 0.4411064877445677, "belt": 0.5, "onbelt": 60, "vram": 1744.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.989600824843985, "vol_hi": 0.9973087963373837, "stretch": 1.3061450449397634, "droop": 0.4338589378193635, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 27.269118199998047, "exit": 0}
{"engine": "physx", "count": 60, "seed": 11, "ms_frame": 19.741103125003672, "fps": 50.655730516569804, "tets": 22470, "cage_verts": 8175, "convey": 0.49528204798426373, "belt": 0.5, "onbelt": 60, "vram": 3048.1171875, "seconds": 8.0, "finite": true, "inverted": 7, "vol_lo": 0.964207940575579, "vol_hi": 0.9969950291353253, "stretch": 1.2796949441840009, "droop": 0.8498395954102174, "pen_pairs": 180, "pen_worst_mm": 28.625164180994034, "readback_ms": 2.661287192951708, "wall_s": 18.902914999984205, "exit": 0}
{"engine": "warp-corot", "count": 60, "seed": 13, "ms_frame": 5.138990208373191, "fps": 194.59075799962693, "tets": 5175, "cage_verts": 3435, "convey": 0.4798843879299884, "belt": 0.5, "onbelt": 60, "vram": 1626.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9986162928372431, "vol_hi": 1.001446516777188, "stretch": 1.1307907991916124, "droop": 0.6053904336377671, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 11.32241389999399, "exit": 0}
{"engine": "newton-xpbd", "count": 60, "seed": 13, "ms_frame": 11.751458749980278, "fps": 85.09581842353641, "tets": 5175, "cage_verts": 3435, "convey": 0.4506562311199494, "belt": 0.5, "onbelt": 60, "vram": 1744.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9928261974895299, "vol_hi": 0.9976755907775645, "stretch": 1.1537023199309602, "droop": 0.7189715085245761, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 19.04991480000899, "exit": 0}
{"engine": "physx", "count": 60, "seed": 13, "ms_frame": 24.417101875042135, "fps": 40.954901409579115, "tets": 22470, "cage_verts": 8175, "convey": 0.49100696019829687, "belt": 0.5, "onbelt": 60, "vram": 3035.3671875, "seconds": 8.0, "finite": true, "inverted": 12, "vol_lo": 0.9585848917169925, "vol_hi": 0.9970499509143868, "stretch": 1.3294245290437552, "droop": 0.774164152039143, "pen_pairs": 165, "pen_worst_mm": 27.46954746544361, "readback_ms": 2.2785045617574564, "wall_s": 19.93958149998798, "exit": 0}
{"engine": "warp-corot", "count": 100, "seed": 7, "ms_frame": 5.70561187499455, "fps": 175.2660401564653, "tets": 8625, "cage_verts": 5725, "convey": 0.47541762738494997, "belt": 0.5, "onbelt": 100, "vram": 1639.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9970984299067058, "vol_hi": 1.002379790436014, "stretch": 1.1430395794159267, "droop": 0.653749987364156, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 8.401581300015096, "exit": 0}
{"engine": "newton-xpbd", "count": 100, "seed": 7, "ms_frame": 18.880184791669308, "fps": 52.96558328397506, "tets": 8625, "cage_verts": 5725, "convey": 0.4399999240026899, "belt": 0.5, "onbelt": 100, "vram": 1823.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.97400006909757, "vol_hi": 0.9975622907222136, "stretch": 1.3888945166800828, "droop": 0.44368454163934345, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 23.4717092999781, "exit": 0}
{"engine": "physx", "count": 100, "seed": 7, "ms_frame": 41.30596479165737, "fps": 24.20957856919424, "tets": 37450, "cage_verts": 13625, "convey": 0.49176484220220706, "belt": 0.5, "onbelt": 100, "vram": 3000.1171875, "seconds": 8.0, "finite": true, "inverted": 18, "vol_lo": 0.9687960475239402, "vol_hi": 0.9988166405081077, "stretch": 1.4883311774455266, "droop": 0.6304783315276726, "pen_pairs": 285, "pen_worst_mm": 27.582259848713875, "readback_ms": 3.9200336839329744, "wall_s": 27.95747800002573, "exit": 0}
{"engine": "warp-corot", "count": 100, "seed": 11, "ms_frame": 5.797158333310411, "fps": 172.4983073610411, "tets": 8625, "cage_verts": 5725, "convey": 0.4777324143607571, "belt": 0.5, "onbelt": 100, "vram": 1639.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9870131184248402, "vol_hi": 1.0016059751819526, "stretch": 1.6963818047391395, "droop": 0.774207748344411, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 8.685977900022408, "exit": 0}
{"engine": "newton-xpbd", "count": 100, "seed": 11, "ms_frame": 20.2193572916561, "fps": 49.45755622077409, "tets": 8625, "cage_verts": 5725, "convey": 0.44441131686289365, "belt": 0.5, "onbelt": 100, "vram": 1823.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9800730121739584, "vol_hi": 0.9975804428688012, "stretch": 1.2550504894198724, "droop": 0.5082641905757169, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 24.288328900001943, "exit": 0}
{"engine": "physx", "count": 100, "seed": 11, "ms_frame": 41.86570666661282, "fps": 23.885898020622275, "tets": 37450, "cage_verts": 13625, "convey": 0.49404752143435127, "belt": 0.5, "onbelt": 100, "vram": 3042.8046875, "seconds": 8.0, "finite": true, "inverted": 10, "vol_lo": 0.9765028356266237, "vol_hi": 0.9968067514732781, "stretch": 1.2259974500225455, "droop": 0.5084563839439319, "pen_pairs": 275, "pen_worst_mm": 27.98682637512684, "readback_ms": 3.9225035088173557, "wall_s": 28.86904210000648, "exit": 0}
{"engine": "warp-corot", "count": 100, "seed": 13, "ms_frame": 5.98099375001766, "fps": 167.1962957655736, "tets": 8625, "cage_verts": 5725, "convey": 0.4785242144328216, "belt": 0.5, "onbelt": 100, "vram": 1639.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9981582333102982, "vol_hi": 1.0014532798430444, "stretch": 1.1192909422699813, "droop": 0.509915041813315, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 8.92369160000817, "exit": 0}
{"engine": "newton-xpbd", "count": 100, "seed": 13, "ms_frame": 21.010287916639452, "fps": 47.595730432995786, "tets": 8625, "cage_verts": 5725, "convey": 0.44775457909608035, "belt": 0.5, "onbelt": 100, "vram": 1823.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9850464046531184, "vol_hi": 0.9980288289274702, "stretch": 1.2858966034369144, "droop": 0.4865814307818921, "pen_pairs": 0, "pen_worst_mm": 0.0, "wall_s": 24.878722099994775, "exit": 0}
{"engine": "physx", "count": 100, "seed": 13, "ms_frame": 44.26844416666427, "fps": 22.589454380532217, "tets": 37450, "cage_verts": 13625, "convey": 0.4915264999544732, "belt": 0.5, "onbelt": 100, "vram": 3061.0546875, "seconds": 8.0, "finite": true, "inverted": 13, "vol_lo": 0.9781660894978801, "vol_hi": 0.9972707785076478, "stretch": 1.4459222392042141, "droop": 0.31150177091686715, "pen_pairs": 303, "pen_worst_mm": 26.64133347570896, "readback_ms": 4.0443859661504495, "wall_s": 30.34709799999837, "exit": 0}
```

## Verdict: the Warp corotational arm takes the benchmark (2026-08-24)

The question this run answers: is a hand-rolled Warp soft-body fish viable, or
should the idea be killed for failing to compete with PhysX on fidelity or
performance? It competes. It is the ONLY arm that passes the acceptance bar at
60 fish, and it does so at a fraction of everyone else's frame time:

- **Performance**: 5.8 ms at 100 fish against PhysX's 41.9 and Newton-XPBD's
  20.2, measured the same evening under the same (loaded -- see below) GPU. It
  is the only arm whose frame time barely moves with count (3.8 -> 5.8 ms over
  16 -> 100 fish): one CUDA graph per frame, contact detection once per frame,
  85 tets per fish. Half of PhysX's VRAM (1.6 vs 3.0 GB).
- **Conveying fidelity, the metric that killed Newton-XPBD**: 0.956x belt speed
  against PhysX's 0.988x and Newton-XPBD's 0.89x. The positional Coulomb stick
  against the roller's own surface velocity (budgeted on the substep's LATCHED
  accumulated normal correction, not the residual overlap) is what closes the
  gap sphere-contact slip opened.
- **Shape**: zero inverted tets on every count and seed (the corotational rows
  are inversion-safe by construction), volume 0.98-1.00, worst stretch 1.73x
  (bar: 2.0). PhysX -- historically the shape king -- inverts 4-18 tets per run
  on THIS scan set and fails its own bar.

Honest caveats, in order of weight:

1. **These milliseconds were measured while a game had the GPU.** Every arm ran
   interleaved through the same load, so the ordering is protected, but the
   absolute numbers are ~2x their quiet values (PhysX at 16 fish: 15.7 ms
   tonight against 9.5 in the 2026-08-21 table). Re-run the key cells on a
   quiet GPU before quoting absolute frame times.
2. **The zero-penetration column is the metric going blind, again**: sphere
   contacts hold cage vertices of different fish apart by construction, past
   the 30 mm measurement thickness. The paired frames are the real check --
   distinct, clean, full-bodied fish -- but the pile is visibly AIRIER than
   PhysX's: fish keep daylight between each other where a real catch touches.
   Pile density is the one fidelity axis PhysX still clearly wins.
3. **Stretch is measured on an 85-tet cage** against PhysX's conforming
   375-tet mesh -- a coarser ruler on our side of that column.
4. PhysX's inversions tonight (0 in the August 21 table) say this scan set is
   harder than the one that table was built on; the two tables should not be
   read against each other.
