# Splat mapping — Gaussian-splat maps from a moving vehicle's camera, scored against exact truth

A vehicle moves through a threepp scene with a camera. Its frames, taken where the vehicle actually was, are the
only training data for a Gaussian-splat map. Because the scene is ours, the map can be scored against exact
truth: the picture from views ON the vehicle's path and OFF it, and the geometry against the real meshes.

This folder is the generic harness. A site runner (one per scene, kept with the scene) builds its scene, moves its
vehicle, and writes a dataset in the format below; the harness trains and scores it.

| File | What it is |
| --- | --- |
| `dataset.py` | The dataset format (nerfstudio `transforms.json` plus split, sim time, mask and class map per frame), its writer, the start points from masked depth, the mask overlay and contact sheets, masked PSNR / SSIM. |
| `manifest.py` | `write_manifest(run_dir, ...)`: commit and dirty flag of every repo, Python / numpy / torch / gsplat versions, the CUDA toolkit, the driver and GPU, argv, seed, UTC time. Every script calls it. |
| `score.py` | Renders the trained `.ply` in threepp from every held-out view (masked PSNR / SSIM per split), bakes its surface and measures it against the true surface samples. |
| `../scene_splats/train.py` | THE trainer (gsplat, Apache-2.0), shared with the orbit pilot: masks, named splits, `--seed`, `--max-splats`. |

## Dataset format

Per frame: `file_path` (rgb png), `depth_file_path` (uint16 planar depth in `depth_unit_scale_factor` metres,
0 = none), `mask_path` (uint8, 255 = train on / score this pixel), `class_path` (uint8 raw class: 0 static,
1 water, 2 moving, 3 sky, 4 own vehicle), `transform_matrix` (OpenGL camera-to-world), `split` (`train`,
`on_path`, `off_path:<name>`), `t` (sim time). Global: intrinsics, `near` / `far`, the range cap, the start
points' voxel and cap, and a `threepp` block with every parameter of the capture.

## A run

```
py -3.14 <site runner> --settle                                   # how many frames a keyframe needs (N)
py -3.14 <site runner> --frames N                                 # capture: RUN/dataset, RUN/truth_surface.npz
<venv>/Scripts/python python/examples/scene_splats/train.py --data RUN/dataset --out RUN/seed0 --steps 15000 --seed 0
py -3.14 python/examples/splat_mapping/score.py --data RUN/dataset --splats RUN/seed0/splats.ply --truth RUN/truth_surface.npz --frames N
```

The orbit pilot is the harness's regression check: `scene_splats/make_dataset.py` writes its held-out views and
four off-band splits (top-down, grazing, far, close) into the same format, and `train.py` scores every split.
