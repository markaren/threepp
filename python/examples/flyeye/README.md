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
| `lattice.py` | `HexLattice`: column (u, v), BoxEye 13x13 box mean at the 721 columns, sRGB frame to luminance |
| `optic_lobe.py` | `OpticLobe(npz, device, dtype)`: `reset`, `fade_in`, `step`, `by_type`, `central` |
| `spike_offline.py` | Phase 0 gate: own path vs the flyvis reference, direction selectivity, the figure |
| `dt_sweep.py` | Euler stability at dt 1/60, 1/100, 1/200 and `step` timing |
| `render_edges.py` | The eight threepp Vulkan moving-edge movies used by the gate |
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
