# Gaussian splats on the Vulkan backend

`SplatCloud` renders on both backends. The GL path draws it as an
`InstancedMesh` over a unit quad with a CPU depth sort; the Vulkan path is a
**compute tile rasterizer** — `src/threepp/renderers/vulkan/SplatPass.{hpp,cpp}`
plus nine `splat_*.comp` shaders — that composites into the linear-HDR scene
buffer between the deferred shade and the depth of field.

The GL path is the correctness oracle. `examples/objects/gaussian_splats.cpp`
drives **both** backends from one scene so the comparison is of rasterizers and
not of two demos:

```
gaussian_splats <scene.ply>                    GL
gaussian_splats <scene.ply> --vulkan           Vulkan compute tile rasterizer
gaussian_splats <scene.ply> --vulkan --shot out.png --frames 200
gaussian_splats <scene.ply> --vulkan --occluder     splat-behind-mesh depth test
gaussian_splats <scene.ply> --vulkan --fog          does the cloud sit IN the medium
gaussian_splats <scene.ply> --vulkan --fog --sun    ...with the medium LIT by a sun only
gaussian_splats <scene.ply> --vulkan --bench 200    orbit, timings, GPU breakdown
gaussian_splats <scan.zip>  --vulkan --level 1      a coarser SOG level, same scene
gaussian_splats <scan.zip>  --vulkan --scale 0.5    the pass runs at RENDER resolution
```

Compare captures with tone mapping off (the example's Vulkan default) so the
sRGB decode on the way into `sceneHdr` and the encode on the way out are
inverses. With a tone curve enabled the splats tone-map along with the scene —
correct, and deliberately not what GL does.

## Why compute, and why there

`sceneHdr` carries no `COLOR_ATTACHMENT` usage (`BloomPass` owns it as a
`STORAGE|SAMPLED` image), and the reference 3DGS design is a software tile
rasterizer anyway. Compositing at that point — linear HDR, **pre**-post — means
splats get depth of field, bloom, tone mapping and TAA through the same code the
rest of the scene uses. The post-TAA overlay path, where world `Sprite`s and
particles composite, is display-referred LDR; putting splats there would mean
re-deriving every one of those effects.

Pipeline, all compute, one descriptor set layout shared by every stage:

| stage | what it does |
|---|---|
| `splat_project` | view transform, near/frustum cull, 3D→2D covariance (EWA), conic, tile rect, SH colour, atomic min/max over visible view distances |
| `splat_scan` ×N | exclusive prefix sum over the per-splat tile counts (recursive) |
| `splat_expand` | writes `(key, splatIndex)` for every covered tile at `offset[splat] + k` |
| `splat_radix` ×2 | per 4-bit digit: block histogram, then a stable scatter |
| `splat_range` | tile → `[begin, end)` from the sorted keys |
| `splat_raster` | one 16×16 workgroup per tile, one thread per pixel, front-to-back with a transmittance early-out |

## The blend domain

A 3DGS optimiser minimises `|Σ cᵢ·αᵢ·Tᵢ − I_sRGB|` against sRGB-**encoded**
training images. Its coefficients are therefore display-referred values whose
display-space alpha blend reproduces the photograph — **not radiances**.

So the whole composite (the splats *and* the scene behind them) happens in the
display-referred domain and the result re-enters `sceneHdr` through the sRGB
decode. The background round-trips exactly, so a pixel no splat covered is
bit-unchanged and an HDR sky behind a translucent splat keeps its range; a fully
covered pixel is exactly the GL answer.

Blending in linear light instead is a different operator on different numbers,
and it is not subtle — measured +15 % mean on the procedural cloud, and on the
5.0M Sanctuaire scan the near-invisible sky shell (a halo at ~2 % coverage)
lifted 7× into a visible grey dome above the horizon where GL renders black:

| framing | linear blend | display blend |
|---|---|---|
| procedural (3 translucent shells) | 25.5 dB | **39.2 dB** |
| ATLAS 216k | 39.8 dB | **49.4 dB** |
| ATLAS 216k, sky-heavy | 47.2 dB | **54.4 dB** |
| Sanctuaire 5.0M | 24.3 dB | **33.2 dB** |
| Sanctuaire 5.0M, sky-heavy | 24.7 dB | **35.9 dB** |

The honest cost: partial-coverage compositing is not linear-light. It is the
operator the asset was fitted with, which for a scan — where the ground truth is
what the capture looks like — is the one that matters. Everything downstream
still sees a linear value.

## Determinism

`VulkanSplat_test` gates it. Two frames from the same camera must produce the
same sorted key array, the same payload array and the same composited pixels —
verified bit-identical, and identical across processes as well (the committed
golden differs by `maxDelta 0` between the run that wrote it and the run that
checks it).

The gate is on the tile **expansion**: the obvious design (every splat
`atomicAdd`s a global cursor) produces a different `(splat, tile)` ordering every
run, the stable sort under it faithfully preserves that ordering into the image,
and the result passes every visual check while quietly making sensor goldens
worthless. A prefix sum over the per-splat tile counts makes the ordering a pure
function of the projection instead.

There are **no subgroup operations anywhere** in these shaders. The codebase
never queries `VkPhysicalDeviceSubgroupProperties` and enables
`subgroupSizeControl` only inside `THREEPP_WITH_FSR`, so a subgroup-width
assumption in the one pass that must be bit-reproducible would be an unchecked
portability bet.

## Measured (RTX 4070, 960×600, orbiting, vsync off, 3 interleaved rounds)

Frame time, median of 195 frames after 5 warmup:

| asset | GL | Vulkan | of which the splat pass |
|---|---|---|---|
| ATLAS 216k | 1.93–2.01 ms | 3.47–3.61 ms | 3.2 ms |
| Sanctuaire 5.0M | 53.1–55.4 ms | 25.1–28.9 ms | 24–28 ms |

The tile rasterizer loses at 216k and wins by ~1.9× at 5M. The crossover is the
GL path's per-frame CPU counting sort and five million quads against this path's
fixed cost of eight radix passes.

Morton storage order (`--morton`, `SplatData::reorderMorton`) makes **no
measurable difference** on this path — ATLAS 3.56 vs 3.62 ms median, Sanctuaire
36.2 vs 36.3 ms, all inside run-to-run noise. That refutes the "a tile
rasterizer reads per screen region, so spatial storage locality should align"
hypothesis for this implementation, but it is also not the 60 % regression
Morton is on the GL draw-order path. The likely reason: the raster stages splats
through shared memory in batches taken from the **sorted** payload order, which
is already tile-major, so storage order changes where the data lives without
changing what a tile fetches.

## Out of scope, on purpose

Splats cast no shadows, appear in no reflection, contribute to no probe GI, do
not participate in froxel fog or MSAA, are invisible to the RT sensors, and are
not exposed to Python, the editor or serialization. Secondary views (`addView`)
skip the pass entirely rather than paint splats into a sensor AOV nobody asked
for.

Every fog term that carries light back INTO the camera→splat leg is mirrored:
analytic height fog, the murk below a water surface, and the sun's single-
scattering glow. The sun term is **closed form** rather than the surface path's
16-step march — for a directional light both `L` and `rd` are fixed, so the HG
phase is constant along the leg and the weight the march is left carrying
integrates exactly to `1 - Ta` over the profile `splatHeightFogOd` already
integrates. Checked numerically against that march across six configurations:
worst difference 0.94 %, and that residual is the march's own 16-step truncation,
not a modelling gap.

Still not mirrored, both additive: the froxel LUT's point-light glow (wants the
froxel volume + cluster grid this pass does not bind) and the **shadowing** of
the sun term (wants the TLAS), so a cloud is a little less lit by nearby lamps
than a mesh beside it, and its sun haze is the unoccluded upper bound rather than
a shafted one — too bright by a bounded amount, never too dark.

The sun term is the one that mattered, because its absence was not a dimming. A
sun-only scene zeroes the ambient and env terms that were the only ones here, so
the leg kept its extinction and got nothing back: the cloud sank toward black
while the lit air around it stayed grey. `--fog --sun` is that configuration, and
`--fog`'s density is now scaled to the fit radius — a fixed density tuned for a
metres-scale scan leaves the unit-scale procedural cloud at `T = 0.97`, which
renders a plausible frame while testing nothing.

## Tile size: 16×16 stays, and zooming OUT is the open problem

`kTileW/kTileH = 16` was measured against 8×8 (three sites: `splat_common.glsl`,
`SplatPass.cpp`, and the raster's `local_size`), 5.0M scan under `--lod-dynamic`:

| camera | 16×16 | 8×8 | entries at 8×8 |
|---|---|---|---|
| zoomed 16× OUT | 66.1 ms (raster 66.7) | **42.0 ms** (raster 34.1) | 1.69M → 2.51M |
| framed | 7.29 ms | 7.91 ms | → 5.07M |
| zoomed 8× IN | 7.78 ms | **17.97 ms** (sort 9.9) | → **19.5M** |

Smaller tiles buy 1.6× in the pathological zoomed-out case and lose 2.3× in the
common close-up one, because a splat covers four times as many tiles and the
expansion and sort pay for every pair — 19.5M entries at 8×8 also comes within
reach of the 25M budget, i.e. of truncation. One global constant has to serve
every camera, so 16×16 stands. This is measured now rather than assumed.

**The zoomed-out cliff is real and unfixed**: 66 ms at 15 fps with the LOD policy
already choosing its coarsest level and the sort down at 1.05 ms. The raster is
~66 of those milliseconds. Zoomed far out the whole scan lands in a few dozen
tiles, so ~1.7M entries pile into them and each tile blends its share serially in
one workgroup — the same tile-starvation mechanism that makes `--scale 0.5` cost
73% more, and note 16× out has FEWER entries than 4× out yet runs 3× slower.

LOD cannot reach it: at that zoom the scan covers ~100×60 px, so even a 625k
level is ~100 splats per pixel and the policy's ~1/px target is unreachable with
the levels an asset ships. The candidates are a sub-pixel cull in project (a
splat below a footprint threshold costs a full entry and a full serial blend step
for a sliver of alpha — note `kScreenDilation = 0.3` deliberately keeps such
splats alive, matching GL), or LOD levels coarser than the asset's own. Neither
is measured yet.

## The per-cloud tax (measured 2026-08-06)

`record()` runs the WHOLE pipeline per cloud — the clears, the sizing dispatch,
eight rounds of fill/hist/scan/scatter, and a full-screen `(tilesX, tilesY)` tile
walk — so a second cloud is not a second batch, it is a second pass. Measured
with `--clouds K`, which partitions the same splats across K clouds at constant
total (1.25M, level 2, framed):

| K | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| frame | 7.1 ms | 9.6 ms | 13.2 ms | 16.4 ms |

**~1.3 ms per extra cloud**, and it is nearly all fixed overhead: the per-stage
numbers (which cover the first cloud only) shrink as K rises while the frame
grows. Ten clouds would cost ~12 ms before drawing a splat.

Consequences. Per-chunk LOD cannot be N chunk-clouds — it wants ONE cloud whose
buffer is assembled from the selected per-chunk levels, which also removes the
inter-cloud compositing-order worry. And `kMaxClouds = 8` is not a limit anyone
should want to raise: eight clouds is already 16 ms.

## V3 — the perf checklist

Not implemented. Ordered by measured or estimated value.

1. ~~**Size the sort and range dispatches from the actual entry count.**~~ DONE.
   `splat_indirect.comp` reads `g.entryCount` after the expansion barrier and
   writes two `VkDispatchIndirectCommand`s; histogram, scatter and range are
   `vkCmdDispatchIndirect` off them. Measured, 5.0M Sanctuaire at 960×600:

   | camera | before | after | sort before → after |
   |---|---|---|---|
   | basilica fills the screen | 28.72 ms | **24.18 ms** | 9.71 → 4.89 |
   | zoom 8× in | 12.96 ms | **7.44 ms** | 9.38 → 3.69 |
   | pointed at empty space | 9.81 ms | **1.77 ms** | 8.50 → **0.16** |

   34.6 → 41.5 fps framed, 75.6 → 132.7 zoomed, 94 → 548 with nothing on
   screen. Byte-identical output: `hashKey`/`hashVal`/`hashColor` match the
   worst-case path exactly at 5.0M and 2.5M, `scanBad 0 orderBad 0`.

   The scan chain is still host-sized and that is now the visible residue: at
   full view 8,724,270 entries of a 20M budget means the sort should have fallen
   to ~44 %, and it did (9.71 → 4.89), but the 8 × 625k-word histogram scan and
   its ~64 barriers ride along at worst case regardless. Both remaining targets
   in the plan (full view ≤23 ms, zoom ≤6 ms) were missed by 1–1.5 ms on exactly
   that. Making the scan indirect too is the follow-up, and it is a bigger change
   than this one: `recordScan` computes per-level counts AND scratch offsets on
   the host.

   `THREEPP_VK_SPLAT_NOINDIRECT=1` restores the worst-case dispatches — the A/B
   switch the numbers above came from, and the first thing to try if another
   driver disagrees.

   Follow-up, ablated and REFUTED as written: the 8 per-pass histogram zero-fills
   cost ~0.5 ms of the 3.9 ms zoom sort (ablation: skip them, 3.43 vs 3.92–4.22),
   but double-buffering the histogram so each fill rides the previous pass's
   barrier recovered NOTHING (3.70/3.88 zoom, 4.84/5.20 full — inside noise), and
   was reverted rather than kept as unmotivated complexity in the sort. So the
   cost is the fill's own launch, not its barrier. Removing the fills for real
   needs the scan to stop dirtying the histogram tail — an OUT-OF-PLACE scan, one
   fill per frame — which is the same class of change as making the scan extent
   dynamic and should be evaluated with it. The other ablation, "skip the scan
   chain", is INVALID as a measurement: it leaves garbage offsets, so the scatter
   thrashes and the sort gets *slower* (5.55). What is valid: 4 passes instead of
   8 halves the sort (1.90 vs 3.92), so cost is linear in passes at ~0.49 ms
   each, against ~12 MB of real data per pass at that zoom. The sort is
   dispatch-and-barrier bound, roughly 6 such units per pass, which is why only
   FEWER PASSES moves it much.

   Now measured rather than estimated, and it is worth more than the −12 % this
   line used to claim. With the per-stage timings (item 4) on the 5.0M
   Sanctuaire at 960×600: sort **9.86 ms** of a 28 ms frame with the basilica
   filling the screen, and **7.9 ms with the camera pointed at empty space** —
   i.e. 7.9 ms to sort ZERO entries, because the dispatch, the 16×39,063-word
   histogram and its scan chain are all sized from a 20M-entry budget that is
   `splatCount × 4` and never looks at the data. Only ~2 ms of the full-view
   sort is the sort. The intuition that the tail blocks "read the count and
   exit, so over-dispatch is nearly free" is WRONG by about 8 ms.

   This is also the blocker for dynamic LOD: `maxSplats_` is a high-water mark,
   so once the finest level has been resident the coarse levels keep paying the
   fine level's sort. Rendering 625k splats with 5M resident would cost ~7.9 ms
   of sort against ~0.6 ms earned, capping any dynamic scheme around 110 fps
   however coarse it goes.
2. **Chunk frustum culling before the sort.** ATLAS culls 5 k of 216 k and
   Sanctuaire 134 k of 5.0 M by the per-splat near/offscreen test alone, all of
   it *after* per-splat projection work. A coarse per-chunk AABB reject would
   remove whole runs before projection. Do NOT extend `occl_cull` — its
   records-never-move invariant is documented as GPU-hang-on-violation.
3. **Narrow the sort key.** 8 radix passes cover the full 32 bits. At 960×600 the
   tile id needs 12 and the depth 20; capping depth at 12 would fit 24 bits and
   6 passes (−25 % sort) at the cost of 4096 depth buckets instead of a million.
   Measure the tie-order cost on the Sanctuaire sky first — the tail-band
   doctrine exists because that is where quantisation shows.
4. ~~**Per-stage GpuTimings.**~~ DONE. `TP_SplatProject` / `TP_SplatSort` /
   `TP_SplatRaster` partition `TP_Splat` from inside the pass, for the first
   splat cloud of the frame (one slot pair per stage, so a second writer would
   be a VUID violation and a wrong number). Surfaced as
   `FrameTimings::splat{Project,Sort,Raster}Ms` and printed by the example's
   `--bench`. The 5.0M Sanctuaire at 960×600, median of 195 frames:

   | framing | project | sort | raster | total |
   |---|---|---|---|---|
   | basilica fills the screen | 3.66 ms | 9.86 ms | 13.87 ms | 27.4 ms |
   | camera pointed away | 1.34 ms | 7.90 ms | 0.055 ms | 9.3 ms |
   | level 2 (1.25M), same framing | 1.03 ms | 2.29 ms | 4.42 ms | 7.8 ms |

   Read it as: the raster half is real work on real coverage, the sort is
   mostly not (item 1), and per-splat projection is the smallest of the three
   even at 5M.
5. ~~**Shrink the budget as well as grow it.**~~ DONE, as VRAM reclamation (the
   time cost died with indirect dispatch). The scratch follows the LIVE demand:
   it grows as before, shrinks (with a factor-of-two threshold plus a ~60-sync
   delay) when the submitted demand halves, and is released outright when the
   last resident cloud goes. Hidden clouds are PARKED, not evicted — collected
   via a full traverse and reported to `syncClouds` so their buffers stay and a
   visibility toggle costs nothing; only leaving the scene evicts. The traded
   corner is stated in the code: a fresh import after a full release re-learns
   its entry budget through one truncated frame, exactly like a first-ever
   import.
6. **A tile-local early-out on transmittance.** The raster already breaks when
   every pixel in a tile has saturated; it does not skip splats that arrive after
   the tile is opaque but before the batch boundary.
7. **The `STORAGE` bit on the motion attachment** (added so the splat pass can
   write motion vectors) is paid by every scene, splats or not. UNMEASURED — it
   may cost framebuffer compression on some hardware. Measure with a
   splat-free scene, before/after the bit, interleaved.
8. **THEN** the large-scene numbers with everything above in: Sanctuaire at 5M,
   interleaved against GL, at 1080p and 4K.
