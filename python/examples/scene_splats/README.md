# Scene to splats — Gaussian splats trained on a threepp scene

Renders one of our own scenes (the TP-1 rover from [`../rover/`](../rover) on a sand
square) with the Vulkan deferred renderer, trains Gaussian splats on that camera
output with [gsplat](https://github.com/nerfstudio-project/gsplat) (Apache-2.0),
loads the result back into threepp and holds it against the scene it came from.

A scene of our own is a better training subject than a real capture in one way that
matters: the cameras are exact. There is no structure-from-motion step to get wrong,
the start points come straight from the depth buffer, and every result can be scored
against the real geometry.

| Script | What it is |
| --- | --- |
| `make_dataset.py` | Renders the training and held-out views, writes images, depth, start points and a nerfstudio `transforms.json`. |
| `train.py` | Trains splats with gsplat (runs in its own venv) and writes a 3DGS `.ply`. |
| `compare.py` | Renders the `.ply` in threepp from the held-out cameras, bakes its surface, measures both. |
| `common.py` | The scene, the camera rig and the dataset format the three share. |

```
py -3.14 python/examples/scene_splats/make_dataset.py           # ~90 s, writes ~/.cache/threepp/scene_splats/rover
<venv>/Scripts/python python/examples/scene_splats/train.py     # 7000 steps
py -3.14 python/examples/scene_splats/compare.py
```

## What the camera output needs

- **Hold each pose.** TAA, the SVGF denoiser and probe GI are temporal, so the first
  frames after a camera jump are not the picture yet. `make_dataset.py --convergence`
  measures it on this scene: after 24 frames 7.5 % of pixels are still more than two
  8-bit levels off the settled frame, after 48 frames 0.7 %. The default is 48.
- **No upscaler, no auto exposure, a pinned clock** (`common.setup_renderer`): every
  view is the same picture each run, and every view has the same exposure.
- **A background colour nothing in the scene has**, so the trainer cannot explain a
  dark part of the model as background showing through.
- **The splats are display values.** They were fitted to the tone-mapped frame, so
  `compare.py` renders them with `ToneMapping.NoToneMapping`; a tone curve on top
  would apply it twice.
- **The `.ply` is in the frame every 3DGS `.ply` comes in**, the COLMAP one with +Y
  down. Training runs in the scene's own frame and the export turns the splats half a
  turn about X (centres, rotations and the SH terms that are odd in y and z), so the
  editor's import and the `gaussian_splats` example, which turn every `.ply` back
  with `rotation.x = pi` on the node, show it upright where the rover stood.
  `points3d.ply` stays in the dataset's own frame (Y up), as `transforms.json` needs.

## Results (2026-11-15, RTX 5080, 800×600)

200 training views, 24 held out between them, 150k start points.

| | |
| --- | --- |
| dataset | 224 views in 89 s |
| training, 7000 steps | 64 s, 279k splats |
| held-out PSNR, gsplat's render | 37.46 dB (worst view 34.2), SSIM 0.984 |
| held-out PSNR, threepp's Vulkan render of the same `.ply`, turned upright on the node | 36.80 dB (worst 34.0) |
| threepp vs gsplat, same splats | 44.4 dB |
| expected depth vs the true depth, median | 3.4 mm |
| `bake_surface` at 2 cm voxels, distance to the real surfaces | median 3.1 mm, p90 10 mm, 98.8 % within one voxel |
| … on the rover alone | median 4.9 mm, p90 21 mm, 87.7 % within one voxel |
| the rover's visible surface the bake reaches (within two voxels) | 98.8 % |

Where the bake is wrong, by looking at `compare_sheet.jpg`: wheel spokes and hubs
fill in, the gold-foil sides come out lumpy (the glitter texture makes the splat depth
noisy), and the mast's camera head is rounded off.

## gsplat on Windows

`train.py` handles the first two; the third is a one-line edit in the installed gsplat.

1. **torch's CUDA major can be newer than the toolkit's** (here torch cu130 and
   toolkit 12.8). torch refuses a major mismatch when it builds an extension;
   `train.py` lifts that check. The kernels only need the driver.
2. **The built module needs the toolkit's `cudart64_12.dll`**, and Windows does not
   look for DLLs on `PATH`. `train.py` adds the toolkit's `bin` (from `CUDA_HOME`,
   `CUDA_PATH` or the installer's `CUDA_PATH_V*`). After the first build it also imports
   the module straight from torch's extension cache, so later runs need no Visual
   Studio shell.
3. **gsplat 1.5.3 passes GCC's `-Wno-attributes` to MSVC** (error D8021). In
   `gsplat/cuda/_backend.py`, make the C++ flags MSVC's on Windows:
   `extra_cflags = ["/O2"] if os.name == "nt" else [opt_level, "-Wno-attributes"]`.

The first build takes a few minutes, from a Visual Studio developer shell with
`TORCH_CUDA_ARCH_LIST=12.0` (an RTX 50) set.

## Limits

- The scene must hold still: water, particles, wind and anything on the wall clock
  would be trained as noise.
- View-dependent light is only as good as SH degree 3: glass and water reflections
  will train poorly.
- The bake's cameras are the dataset's own (every fourth training view), which is
  why it covers 98.8 % of what is visible; the 32 % of the rover no camera sees (the
  underside, inside the wheels) has no surface, by construction.
