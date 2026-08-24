# Newton VBD vs PhysX 5 deformable volumes, on the same fish conveyor

_2026-08-21 20:30 - 36 runs, 8 s of settled conveying each, headless, one subprocess per run._

Every arm shares the scans, the auto-orientation, the decimated render surface, the barycentric bind, the belt, the roller pitch, the rails, the spawn layout and seed, the GPU skinning path, the headless renderer and the quality metrics. The solver -- and in one arm the tetrahedral mesh -- is what differs.

## Results

Speed is the median over seeds; every quality number is the WORST over seeds, because a solver that inverts a tet on one layout in three has a failure mode rather than a good average. Penetration is measured between cage vertices of different fish at a fixed 30 mm thickness, so it compares across cages.

| arm | fish | ms/frame | fps | realtime | tets | inverted | volume | stretch | droop | pen pairs | worst pen | convey | on belt | VRAM |
|---|---:|---:|---:|:--:|---:|---:|---|---:|---:|---:|---:|---:|:--:|---:|
| newton-voxel | 16 | 7.4 | 134 | yes | 5680 | 0 | 0.98-1.00 | 2.18 | 0.42 | 15 | 5.0 mm | +0.492 of 0.50 | 16/16 | 1694 MB |
| newton-voxel | 40 | 11.9 | 84 | yes | 14200 | 0 | 0.98-1.00 | 2.14 | 0.34 | 45 | 10.3 mm | +0.510 of 0.50 | 40/40 | 1768 MB |
| newton-voxel | 60 | 15.7 | 64 | yes | 21300 | 0 | 0.98-1.01 | 2.83 | 0.39 | 60 | 14.6 mm | +0.504 of 0.50 | 60/60 | 1872 MB |
| newton-voxel | 100 | 25.8 | 39 | no | 35500 | 0 | 0.97-1.00 | 2.51 | 0.32 | 110 | 10.7 mm | +0.492 of 0.50 | 100/100 | 2109 MB |
| newton-physx-tets | 16 | 8.2 | 121 | yes | 5992 | 1 | 0.99-1.00 | 4.96 | 0.39 | 40 | 20.4 mm | +0.464 of 0.50 | 16/16 | 1696 MB |
| newton-physx-tets | 40 | 12.3 | 81 | yes | 14980 | 0 | 0.97-1.00 | 5.31 | 0.29 | 114 | 22.6 mm | +0.463 of 0.50 | 40/40 | 1770 MB |
| newton-physx-tets | 60 | 16.0 | 63 | yes | 22470 | 3 | 0.96-1.01 | 5.49 | 0.34 | 156 | 16.8 mm | +0.481 of 0.50 | 60/60 | 1842 MB |
| newton-physx-tets | 100 | 24.8 | 40 | no | 37450 | 1 | 0.97-1.01 | 6.74 | 0.26 | 242 | 17.3 mm | +0.481 of 0.50 | 100/100 | 2015 MB |
| physx | 16 | 9.5 | 105 | yes | 5992 | 0 | 1.00-1.00 | 1.02 | 0.57 | 48 | 21.4 mm | +0.483 of 0.50 | 16/16 | 3030 MB |
| physx | 40 | 10.8 | 93 | yes | 14980 | 0 | 1.00-1.00 | 1.02 | 0.58 | 97 | 26.6 mm | +0.483 of 0.50 | 40/40 | 3040 MB |
| physx | 60 | 12.1 | 83 | yes | 22470 | 0 | 1.00-1.00 | 1.02 | 0.50 | 133 | 27.6 mm | +0.483 of 0.50 | 60/60 | 3048 MB |
| physx | 100 | 15.3 | 65 | yes | 37450 | 0 | 1.00-1.00 | 1.02 | 0.29 | 231 | 29.0 mm | +0.482 of 0.50 | 100/100 | 3061 MB |

## Protocol 1 - equal budget (16.7 ms/frame at 60 Hz)

- **newton-voxel** holds 60 Hz to **60 fish** (15.7 ms, 21300 tets); at that count 0 inverted tets, 2.83x worst stretch, 14.6 mm worst penetration.
  - misses the budget at: 100 fish (25.8 ms)
- **newton-physx-tets** holds 60 Hz to **60 fish** (16.0 ms, 22470 tets); at that count 3 inverted tets, 5.49x worst stretch, 16.8 mm worst penetration.
  - misses the budget at: 100 fish (24.8 ms)
- **physx** holds 60 Hz to **100 fish** (15.3 ms, 37450 tets); at that count 0 inverted tets, 1.02x worst stretch, 29.0 mm worst penetration.

## Protocol 2 - acceptance bar at 60 fish

| arm | no inverted tets | worst tet edge stretch < 2.0x | worst cross-fish penetration < 10 mm | every fish still on the belt | conveying >= 0.8 x belt speed | verdict |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| newton-voxel | PASS | **FAIL** | **FAIL** | PASS | PASS | **FAIL** |
| newton-physx-tets | **FAIL** | **FAIL** | **FAIL** | PASS | PASS | **FAIL** |
| physx | PASS | PASS | **FAIL** | PASS | PASS | **FAIL** |

- **newton-voxel** fails on stretch, penetration: inverted 0, stretch 2.83x, worst penetration 14.6 mm, 60/60 on belt, conveying +0.504 m/s.
- **newton-physx-tets** fails on inverted, stretch, penetration: inverted 3, stretch 5.49x, worst penetration 16.8 mm, 60/60 on belt, conveying +0.481 m/s.
- **physx** fails on penetration: inverted 0, stretch 1.02x, worst penetration 27.6 mm, 60/60 on belt, conveying +0.483 m/s.

## Frames (60 fish, same camera, same sim time)

- **newton-voxel**: `C:\dev\threepp\python\examples\fish_conveyor_ab_newton-voxel.png`, 1:1 crop `C:\dev\threepp\python\examples\fish_conveyor_ab_newton-voxel_crop.png`
- **newton-physx-tets**: `C:\dev\threepp\python\examples\fish_conveyor_ab_newton-physx-tets.png`, 1:1 crop `C:\dev\threepp\python\examples\fish_conveyor_ab_newton-physx-tets_crop.png`
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
{"engine": "newton-voxel", "count": 16, "seed": 7, "ms_frame": 7.27351458335761, "fps": 137.4851165196095, "tets": 5680, "cage_verts": 2780, "convey": 0.4922318965768663, "belt": 0.5, "onbelt": 16, "vram": 1694.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9876835043828578, "vol_hi": 1.000632141270106, "stretch": 2.1820626826175844, "droop": 0.8180859803818611, "pen_pairs": 9, "pen_worst_mm": 4.907961934804916, "wall_s": 13.923484300001292, "exit": 0}
{"engine": "newton-physx-tets", "count": 16, "seed": 7, "ms_frame": 7.8330814583144575, "fps": 127.66367939893512, "tets": 5992, "cage_verts": 2180, "convey": 0.4742192474444654, "belt": 0.5, "onbelt": 16, "vram": 1696.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9954282299765259, "vol_hi": 1.0045836888835245, "stretch": 2.989685138213419, "droop": 0.6524246589426047, "pen_pairs": 19, "pen_worst_mm": 15.25605283677578, "wall_s": 19.248011899995618, "exit": 0}
{"engine": "physx", "count": 16, "seed": 7, "ms_frame": 9.527547083356088, "fps": 104.95880957092567, "tets": 5992, "cage_verts": 2180, "convey": 0.48310801040012247, "belt": 0.5, "onbelt": 16, "vram": 3030.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9987212455601189, "vol_hi": 0.9998998303485934, "stretch": 1.0165162619824697, "droop": 0.5709473569901096, "pen_pairs": 22, "pen_worst_mm": 18.16767081618309, "readback_ms": 0.4371503509965055, "wall_s": 10.257293299975572, "exit": 0}
{"engine": "newton-voxel", "count": 16, "seed": 11, "ms_frame": 7.780391249980312, "fps": 128.5282407877021, "tets": 5680, "cage_verts": 2780, "convey": 0.4747135516718472, "belt": 0.5, "onbelt": 16, "vram": 1694.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.982604469583023, "vol_hi": 1.001226256121811, "stretch": 1.9173034295690725, "droop": 0.4165334264410965, "pen_pairs": 10, "pen_worst_mm": 5.015239119529724, "wall_s": 14.223611000023084, "exit": 0}
{"engine": "newton-physx-tets", "count": 16, "seed": 11, "ms_frame": 8.243112916655567, "fps": 121.31339338800717, "tets": 5992, "cage_verts": 2180, "convey": 0.4453945804008107, "belt": 0.5, "onbelt": 16, "vram": 1696.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9894891548421288, "vol_hi": 1.0027032047675166, "stretch": 3.9448285557050484, "droop": 0.6503142485410621, "pen_pairs": 24, "pen_worst_mm": 20.436320453882217, "wall_s": 19.56880460001412, "exit": 0}
{"engine": "physx", "count": 16, "seed": 11, "ms_frame": 9.529547291640483, "fps": 104.93677919802347, "tets": 5992, "cage_verts": 2180, "convey": 0.48743872962736173, "belt": 0.5, "onbelt": 16, "vram": 3030.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9985506907054541, "vol_hi": 0.9998917538234806, "stretch": 1.0151652488838132, "droop": 0.6758059084073493, "pen_pairs": 24, "pen_worst_mm": 21.357838064432144, "readback_ms": 0.4624084218225458, "wall_s": 10.30777820001822, "exit": 0}
{"engine": "newton-voxel", "count": 16, "seed": 13, "ms_frame": 7.447757083294467, "fps": 134.26861118269133, "tets": 5680, "cage_verts": 2780, "convey": 0.5110849426416855, "belt": 0.5, "onbelt": 16, "vram": 1694.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9913748626864957, "vol_hi": 0.9996761634122974, "stretch": 1.6128638529233914, "droop": 0.667025466541505, "pen_pairs": 15, "pen_worst_mm": 4.29547019302845, "wall_s": 14.043204399989918, "exit": 0}
{"engine": "newton-physx-tets", "count": 16, "seed": 13, "ms_frame": 8.586956458323888, "fps": 116.45569706256468, "tets": 5992, "cage_verts": 2180, "convey": 0.46424916725109017, "belt": 0.5, "onbelt": 16, "vram": 1696.1796875, "seconds": 8.0, "finite": true, "inverted": 1, "vol_lo": 0.9884768270055262, "vol_hi": 0.999529520087688, "stretch": 4.960533426822766, "droop": 0.39030190313896207, "pen_pairs": 40, "pen_worst_mm": 17.408687621355057, "wall_s": 19.52035559999058, "exit": 0}
{"engine": "physx", "count": 16, "seed": 13, "ms_frame": 9.565957083335283, "fps": 104.53737052010047, "tets": 5992, "cage_verts": 2180, "convey": 0.4834669955719897, "belt": 0.5, "onbelt": 16, "vram": 3030.1796875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9974158374694114, "vol_hi": 1.0000370249145365, "stretch": 1.0194993809997828, "droop": 0.9151836016651631, "pen_pairs": 48, "pen_worst_mm": 20.36234736442566, "readback_ms": 0.4430959646063122, "wall_s": 10.3252150999906, "exit": 0}
{"engine": "newton-voxel", "count": 40, "seed": 7, "ms_frame": 11.863876458361725, "fps": 84.28948189992273, "tets": 14200, "cage_verts": 6950, "convey": 0.5104793520807174, "belt": 0.5, "onbelt": 40, "vram": 1767.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9837609790083377, "vol_hi": 1.0030073407144688, "stretch": 1.8738545919706566, "droop": 0.5723872181827501, "pen_pairs": 38, "pen_worst_mm": 8.637519553303719, "wall_s": 18.272084299998824, "exit": 0}
{"engine": "newton-physx-tets", "count": 40, "seed": 7, "ms_frame": 12.346485000004273, "fps": 80.99471226018206, "tets": 14980, "cage_verts": 5450, "convey": 0.4633611932964056, "belt": 0.5, "onbelt": 40, "vram": 1769.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9824227375736279, "vol_hi": 1.0026669173040064, "stretch": 3.9266888618513334, "droop": 0.4867315485737198, "pen_pairs": 77, "pen_worst_mm": 22.625615820288658, "wall_s": 22.941897999990033, "exit": 0}
{"engine": "physx", "count": 40, "seed": 7, "ms_frame": 10.775072708323327, "fps": 92.8067983455498, "tets": 14980, "cage_verts": 5450, "convey": 0.48156548563934026, "belt": 0.5, "onbelt": 40, "vram": 3039.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9982142764486059, "vol_hi": 0.9999279237712229, "stretch": 1.0156683852015511, "droop": 0.7231891936443482, "pen_pairs": 86, "pen_worst_mm": 26.634840294718742, "readback_ms": 0.9643457898882365, "wall_s": 10.922710099985125, "exit": 0}
{"engine": "newton-voxel", "count": 40, "seed": 11, "ms_frame": 11.767832708331602, "fps": 84.97741468503389, "tets": 14200, "cage_verts": 6950, "convey": 0.4809630409643281, "belt": 0.5, "onbelt": 40, "vram": 1767.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9804147397984687, "vol_hi": 1.0001882846387227, "stretch": 1.7509383048401637, "droop": 0.4999787653127723, "pen_pairs": 45, "pen_worst_mm": 10.258128866553307, "wall_s": 18.401284299994586, "exit": 0}
{"engine": "newton-physx-tets", "count": 40, "seed": 11, "ms_frame": 12.29981229165181, "fps": 81.30205374587099, "tets": 14980, "cage_verts": 5450, "convey": 0.44458464169316797, "belt": 0.5, "onbelt": 40, "vram": 1769.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9676443982508071, "vol_hi": 1.0034248125656842, "stretch": 4.442818965782168, "droop": 0.2876063266124413, "pen_pairs": 114, "pen_worst_mm": 13.80203478038311, "wall_s": 22.87603700000909, "exit": 0}
{"engine": "physx", "count": 40, "seed": 11, "ms_frame": 10.74026895833716, "fps": 93.1075379843023, "tets": 14980, "cage_verts": 5450, "convey": 0.4830634463103757, "belt": 0.5, "onbelt": 40, "vram": 3039.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9983916082710548, "vol_hi": 1.0000013862395731, "stretch": 1.0139625302058448, "droop": 0.576263903945507, "pen_pairs": 97, "pen_worst_mm": 26.596546173095703, "readback_ms": 0.9360945618123209, "wall_s": 10.919266999990214, "exit": 0}
{"engine": "newton-voxel", "count": 40, "seed": 13, "ms_frame": 12.111754583323394, "fps": 82.56442063125142, "tets": 14200, "cage_verts": 6950, "convey": 0.5106883574975195, "belt": 0.5, "onbelt": 40, "vram": 1767.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9811288572098389, "vol_hi": 1.0000815652581578, "stretch": 2.1418104484674956, "droop": 0.3384551783083167, "pen_pairs": 26, "pen_worst_mm": 5.538396537303925, "wall_s": 18.413223299983656, "exit": 0}
{"engine": "newton-physx-tets", "count": 40, "seed": 13, "ms_frame": 12.38639354166177, "fps": 80.73375003276693, "tets": 14980, "cage_verts": 5450, "convey": 0.46865577107988676, "belt": 0.5, "onbelt": 40, "vram": 1769.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9805290350723834, "vol_hi": 1.0031278857560988, "stretch": 5.308736352313207, "droop": 0.6331850240499619, "pen_pairs": 78, "pen_worst_mm": 13.99976760149002, "wall_s": 22.91050160001032, "exit": 0}
{"engine": "physx", "count": 40, "seed": 13, "ms_frame": 10.793494374956936, "fps": 92.64840145932719, "tets": 14980, "cage_verts": 5450, "convey": 0.4843146266250712, "belt": 0.5, "onbelt": 40, "vram": 3039.6171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9982041151863189, "vol_hi": 0.9999268457239485, "stretch": 1.0161905432433134, "droop": 0.6010102133788643, "pen_pairs": 90, "pen_worst_mm": 25.572964921593666, "readback_ms": 0.9589022809227014, "wall_s": 10.991442099999404, "exit": 0}
{"engine": "newton-voxel", "count": 60, "seed": 7, "ms_frame": 15.86263249998107, "fps": 63.041238583897936, "tets": 21300, "cage_verts": 10425, "convey": 0.5037597981040203, "belt": 0.5, "onbelt": 60, "vram": 1872.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9775743384020255, "vol_hi": 1.0066669459672757, "stretch": 2.736202483054021, "droop": 0.5748332344822031, "pen_pairs": 54, "pen_worst_mm": 6.062472239136696, "shot": "C:\\dev\\threepp\\python\\examples\\fish_conveyor_ab_newton-voxel.png", "wall_s": 22.21544310002355, "exit": 0}
{"engine": "newton-physx-tets", "count": 60, "seed": 7, "ms_frame": 15.96462291666588, "fps": 62.63849796014125, "tets": 22470, "cage_verts": 8175, "convey": 0.4807734581828284, "belt": 0.5, "onbelt": 60, "vram": 1842.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9897645849604965, "vol_hi": 1.0023725475802963, "stretch": 3.599577616631681, "droop": 0.3714235271174509, "pen_pairs": 111, "pen_worst_mm": 14.38615471124649, "shot": "C:\\dev\\threepp\\python\\examples\\fish_conveyor_ab_newton-physx-tets.png", "wall_s": 26.520982800022466, "exit": 0}
{"engine": "physx", "count": 60, "seed": 7, "ms_frame": 12.40335854163277, "fps": 80.62332445227861, "tets": 22470, "cage_verts": 8175, "convey": 0.482536683544677, "belt": 0.5, "onbelt": 60, "vram": 3048.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9973175800357463, "vol_hi": 1.0000067397774821, "stretch": 1.0182026978568246, "droop": 0.6641358137933405, "pen_pairs": 132, "pen_worst_mm": 27.578964829444885, "shot": "C:\\dev\\threepp\\python\\examples\\fish_conveyor_ab_physx.png", "readback_ms": 1.4468121048769702, "wall_s": 12.00578469998436, "exit": 0}
{"engine": "newton-voxel", "count": 60, "seed": 11, "ms_frame": 15.747526458350572, "fps": 63.502036503626364, "tets": 21300, "cage_verts": 10425, "convey": 0.48059864675518443, "belt": 0.5, "onbelt": 60, "vram": 1872.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9784374317695715, "vol_hi": 1.0027184986866495, "stretch": 2.8268579593434993, "droop": 0.39429317780755513, "pen_pairs": 60, "pen_worst_mm": 14.564065262675285, "wall_s": 22.371735799999442, "exit": 0}
{"engine": "newton-physx-tets", "count": 60, "seed": 11, "ms_frame": 16.261866666657927, "fps": 61.49355547541923, "tets": 22470, "cage_verts": 8175, "convey": 0.44884714125752195, "belt": 0.5, "onbelt": 60, "vram": 1842.1171875, "seconds": 8.0, "finite": true, "inverted": 3, "vol_lo": 0.9612026382017658, "vol_hi": 1.0054361346164555, "stretch": 5.4939824909108435, "droop": 0.414063693340207, "pen_pairs": 156, "pen_worst_mm": 16.79563894867897, "wall_s": 26.319460699975025, "exit": 0}
{"engine": "physx", "count": 60, "seed": 11, "ms_frame": 12.032259166638445, "fps": 83.10991195840228, "tets": 22470, "cage_verts": 8175, "convey": 0.48445993539328386, "belt": 0.5, "onbelt": 60, "vram": 3048.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9982593552792839, "vol_hi": 0.9999636466855455, "stretch": 1.0246530665372298, "droop": 0.7834744764613076, "pen_pairs": 123, "pen_worst_mm": 25.332190096378326, "readback_ms": 1.3642936840289877, "wall_s": 11.650468000007095, "exit": 0}
{"engine": "newton-voxel", "count": 60, "seed": 13, "ms_frame": 15.361859583329837, "fps": 65.09628567918729, "tets": 21300, "cage_verts": 10425, "convey": 0.5139464814802954, "belt": 0.5, "onbelt": 60, "vram": 1872.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9782654700745872, "vol_hi": 0.999738074082023, "stretch": 1.6276906991732851, "droop": 0.7092606543499234, "pen_pairs": 42, "pen_worst_mm": 5.986396223306656, "wall_s": 22.073632100014947, "exit": 0}
{"engine": "newton-physx-tets", "count": 60, "seed": 13, "ms_frame": 15.818842916693635, "fps": 63.21574879188537, "tets": 22470, "cage_verts": 8175, "convey": 0.4946323888804676, "belt": 0.5, "onbelt": 60, "vram": 1842.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.980124819616159, "vol_hi": 1.0011016020694152, "stretch": 1.9953599516065084, "droop": 0.3402532341905558, "pen_pairs": 103, "pen_worst_mm": 14.23528604209423, "wall_s": 26.21228509998764, "exit": 0}
{"engine": "physx", "count": 60, "seed": 13, "ms_frame": 12.107164791632385, "fps": 82.59572056796725, "tets": 22470, "cage_verts": 8175, "convey": 0.48284091863455986, "belt": 0.5, "onbelt": 60, "vram": 3048.1171875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9984339677897992, "vol_hi": 0.9999241977033417, "stretch": 1.0152640412151661, "droop": 0.5045323061060831, "pen_pairs": 133, "pen_worst_mm": 24.306146427989006, "readback_ms": 1.382126666150023, "wall_s": 11.752548200020101, "exit": 0}
{"engine": "newton-voxel", "count": 100, "seed": 7, "ms_frame": 26.39188395830085, "fps": 37.89043637733475, "tets": 35500, "cage_verts": 17375, "convey": 0.49164318340578794, "belt": 0.5, "onbelt": 100, "vram": 2109.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9735590297217711, "vol_hi": 0.9999280042799743, "stretch": 2.51142341257855, "droop": 0.3154528113993659, "pen_pairs": 110, "pen_worst_mm": 10.092822834849358, "wall_s": 31.423185099993134, "exit": 0}
{"engine": "newton-physx-tets", "count": 100, "seed": 7, "ms_frame": 24.64501208329845, "fps": 40.57616188704102, "tets": 37450, "cage_verts": 13625, "convey": 0.49061509058269076, "belt": 0.5, "onbelt": 100, "vram": 2015.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9915342982732346, "vol_hi": 1.0017863328305274, "stretch": 2.569651916749326, "droop": 0.26131879904029365, "pen_pairs": 187, "pen_worst_mm": 15.249978750944138, "wall_s": 33.27101060000132, "exit": 0}
{"engine": "physx", "count": 100, "seed": 7, "ms_frame": 15.353384374975576, "fps": 65.13221942322348, "tets": 37450, "cage_verts": 13625, "convey": 0.48327387235607205, "belt": 0.5, "onbelt": 100, "vram": 3061.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9972746283261602, "vol_hi": 0.9999508216447051, "stretch": 1.0238170652024696, "droop": 0.30342929978761213, "pen_pairs": 217, "pen_worst_mm": 28.97610329091549, "readback_ms": 2.1960266666147827, "wall_s": 13.52010349999182, "exit": 0}
{"engine": "newton-voxel", "count": 100, "seed": 11, "ms_frame": 25.1739016666761, "fps": 39.72367943757196, "tets": 35500, "cage_verts": 17375, "convey": 0.4691946361173921, "belt": 0.5, "onbelt": 100, "vram": 2109.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9817358294674121, "vol_hi": 1.0028875631975909, "stretch": 2.3626768575575667, "droop": 0.32821685627906777, "pen_pairs": 93, "pen_worst_mm": 10.680368170142174, "wall_s": 30.98750319998362, "exit": 0}
{"engine": "newton-physx-tets", "count": 100, "seed": 11, "ms_frame": 24.773703958332288, "fps": 40.36538103797208, "tets": 37450, "cage_verts": 13625, "convey": 0.48145198783570087, "belt": 0.5, "onbelt": 100, "vram": 2015.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9844974184783453, "vol_hi": 1.0060292645432996, "stretch": 6.743435855906281, "droop": 0.33976075078093565, "pen_pairs": 242, "pen_worst_mm": 14.670809730887413, "wall_s": 33.43756449999637, "exit": 0}
{"engine": "physx", "count": 100, "seed": 11, "ms_frame": 15.267547708329706, "fps": 65.49840348325341, "tets": 37450, "cage_verts": 13625, "convey": 0.47930810388156936, "belt": 0.5, "onbelt": 100, "vram": 3061.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9974415378972513, "vol_hi": 1.0000760992408486, "stretch": 1.0179302689127534, "droop": 0.2898939995790591, "pen_pairs": 226, "pen_worst_mm": 25.94226971268654, "readback_ms": 2.191392104846033, "wall_s": 13.561532399995485, "exit": 0}
{"engine": "newton-voxel", "count": 100, "seed": 13, "ms_frame": 25.78452708336651, "fps": 38.78294904408372, "tets": 35500, "cage_verts": 17375, "convey": 0.511951159787347, "belt": 0.5, "onbelt": 100, "vram": 2109.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9768045455553692, "vol_hi": 1.0008152115863362, "stretch": 1.9123208153448181, "droop": 0.3612032985231244, "pen_pairs": 93, "pen_worst_mm": 6.533542647957802, "wall_s": 31.081158499990124, "exit": 0}
{"engine": "newton-physx-tets", "count": 100, "seed": 13, "ms_frame": 24.937457916651812, "fps": 40.10031829797122, "tets": 37450, "cage_verts": 13625, "convey": 0.4710723166213866, "belt": 0.5, "onbelt": 100, "vram": 2015.0546875, "seconds": 8.0, "finite": true, "inverted": 1, "vol_lo": 0.9706328977413027, "vol_hi": 1.0004896493643174, "stretch": 2.4505612495804843, "droop": 0.3470745472618173, "pen_pairs": 215, "pen_worst_mm": 17.326703295111656, "wall_s": 33.525120699981926, "exit": 0}
{"engine": "physx", "count": 100, "seed": 13, "ms_frame": 15.30539458332593, "fps": 65.33644033518904, "tets": 37450, "cage_verts": 13625, "convey": 0.48167879057225244, "belt": 0.5, "onbelt": 100, "vram": 3061.0546875, "seconds": 8.0, "finite": true, "inverted": 0, "vol_lo": 0.9982178825577276, "vol_hi": 1.0007399848211687, "stretch": 1.0200506428107132, "droop": 0.687288480448314, "pen_pairs": 231, "pen_worst_mm": 28.868375346064568, "readback_ms": 2.304188772451884, "wall_s": 13.566728100006003, "exit": 0}
```


## Rematch: the settings audit (2026-08-21, second pass)

The first A/B tested one Newton solver, with contact settings inherited from a
speed hunt. Six more arms answer "are we sure the settings are correct" -- each
undoes one tuning decision, or visits a regime the first pass never tried. 60
fish, 3 seeds, quality = worst over seeds:

| arm | ms | fps | inverted | stretch | worst pen | convey |
|---|---:|---:|---:|---:|---:|---:|
| newton-voxel (first A/B) | 15.7 | 64 | 0 | 2.83x | 14.6 mm | +0.504 |
| newton-stock (Newton's own contact defaults) | 19.6 | 51 | 0 | 4.93x | 13.6 mm | ~ |
| newton-s8i2 (VBD small-steps corner) | 24.7 | 40 | 0 | 4.32x | 11.9 mm | ~ |
| newton-s8i2-stock | 27.8 | 36 | 0 | 4.36x | 13.0 mm | ~ |
| newton-ptets-r15 (radius swept, conforming mesh) | 15.0 | 67 | 0 | 2.08x | 20.1 mm | ~ |
| **newton-xpbd** (16 substeps x 1 iter, r 0.25c, E 2e5) | **10.0** | **100** | **0** | **1.44x** | 0.0 mm | **+0.355** |
| physx (first A/B) | 12.1 | 83 | 0 | 1.02x | 27.6 mm | +0.483 |

And newton-xpbd across counts (3 seeds each): 16 fish 5.5 ms, 40 fish 8.1 ms,
100 fish **15.2 ms (65 fps)** -- the same frame time as PhysX at 100, with zero
inversions everywhere and stretch <= 1.68x.

What the audit settles:

- **The first A/B tested the wrong Newton solver.** SolverXPBD in its canonical
  regime (many small steps, one iteration) is the Newton that belongs in this
  comparison: it matches PhysX's 100-fish frame time and posts the second-best
  shape quality of any arm. VBD -- for all that it is the newer solver -- was
  measured at its stock contact settings (newton-stock: slower AND worse, 4.93x)
  and in its own small-step corner (worse); its showing was not a tuning
  artefact. Its cloth-shaped self-contact is the wrong tool for piled volumes.
- **The "worst arm" claim about newton-physx-tets is RETRACTED.** With the
  contact radius swept instead of derived from a heuristic, the conforming-mesh
  arm lands at 2.08x -- better than the voxel cage, as a better mesh should be.
  The first A/B's 5.49x was the radius heuristic, not the mesh and not VBD.
- **XPBD's stiffness "ceiling" was a relaxation artefact, not a limit.** The
  E = 3e6 divergence at the default soft_body_relaxation = 0.9 is Jacobi
  overshoot; at 0.3-0.5 both 1e6 and 3e6 run stably. The binding constraint is
  ITERATIONS: at 1 iteration the solver is convergence-bound and material
  stiffness barely expresses (stretch 1.34 -> 1.29 for a 15x E increase);
  2 iterations visibly stiffen the fish (droop 0.40 -> 0.52) for ~20% more
  frame time. Surface bending edges are a dead end (edge_ke 1e3 inert, 1e4
  explodes). The stiff preset: --iterations 2 --young 1e6 --relaxation 0.5.
- **XPBD's one remaining defect: it conveys at 0.71x belt speed** at every
  count (friction slip through its sphere contacts); belt mu 0.45 -> 0.9 only
  reaches 0.81x, and the stiff preset only 0.72x. PhysX conveys at 0.97x. For
  a machine whose entire job is conveying, that 30% throughput error is the
  deciding defect, and it is a solver property, not a settings one.
- **Zero penetration for XPBD is partly the metric going blind again**: sphere
  contacts hold cage vertices >= 2r apart by construction, which is past the
  30 mm threshold. The 60-fish frame was inspected instead: distinct fish,
  plausible pile, no gaps, no artefacts.

Verdict after the audit: unchanged in the shipping sense -- PhysX still clears
the behaviour bars Newton misses (conveying fidelity, stiffness range, 1.02x
shape hold) at the same 100-fish frame time. Changed in the outlook sense:
Newton is not "2x behind"; its right solver ties the frame time today, and the
gap that remains is two concrete, revisitable defects rather than a maturity
verdict.


### Addendum: the coarse cage is XPBD's stiffness mechanism (same day, third pass)

"Make it stiffer" turned out to have a structural answer, not a material one.
XPBD propagates a constraint correction one tet-ring per iteration, so BODY
stiffness is set by how many rings a fish is long: brute iterations made droop
WORSE (i6: 0.43 vs i2: 0.52), long-range spine springs are integrated as
explicit forces and explode above ke=5e3, but shortening the chain works --
res 8 at E=1e6 measured droop 0.85 against res 14 at E=3e6 measuring 0.76.
The coarse cage's fat contact spheres also finally grip the rollers.

`--solver xpbd` now implies the full preset (16 substeps x 2 iterations,
relaxation 0.5, E=1e6, tet-res 8), and it re-orders the 100-fish standings:

| 100 fish, seed 7 | ms | fps | stretch | convey | on belt |
|---|---:|---:|---:|---:|:--:|
| newton-xpbd (preset) | 10.7 | 94 | 1.37x | 0.88x belt | 100/100 |
| physx | 15.3 | 65 | 1.02x | 0.97x belt | 100/100 |

Newton-XPBD is now the FASTER engine at 100 fish and passes the conveying bar
(0.88 >= 0.8). PhysX still holds shape better (1.02x vs 1.37x) and conveys
truer (0.97x vs 0.88x). Single seed; the full multi-seed matrix has not been
re-run on this preset. The visual cost of the res-8 cage is slight angularity
in sharp folds, judged acceptable at conveyor-camera distance -- inspect
`xpbd_res8_60.png` before disagreeing with that judgement in either direction.
