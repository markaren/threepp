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

## V3 — the perf checklist

Not implemented. Ordered by measured or estimated value.

1. **Size the sort and range dispatches from the actual entry count.** The
   expanded count lives on the GPU, so today every pass after the expansion is
   dispatched over the BUDGET. Measured upside, Sanctuaire orbit: 28.8 → 25.2 ms
   (−12 %), and 36.2 → 25.2 ms (−30 %) against the pre-tuning budget.
   `vkCmdDispatchIndirect` off a count the expansion writes; the histogram scan's
   extent has to follow the same count or the offsets do not line up.
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
4. **Per-stage GpuTimings.** `TP_Splat` brackets the whole pass; project / scan /
   expand / sort / raster want their own so items 1–3 can be attributed rather
   than inferred.
5. **Shrink the budget as well as grow it.** The budget never comes back down
   after a close-up frame inflates it, so a scene that pans away keeps paying.
   Same machinery as the growth path (`syncClouds` already reads the entry count
   every frame), plus hysteresis so a pan does not thrash the reallocation.
6. **A tile-local early-out on transmittance.** The raster already breaks when
   every pixel in a tile has saturated; it does not skip splats that arrive after
   the tile is opaque but before the batch boundary.
7. **The `STORAGE` bit on the motion attachment** (added so the splat pass can
   write motion vectors) is paid by every scene, splats or not. UNMEASURED — it
   may cost framebuffer compression on some hardware. Measure with a
   splat-free scene, before/after the bit, interleaved.
8. **THEN** the large-scene numbers with everything above in: Sanctuaire at 5M,
   interleaved against GL, at 1080p and 4K.
