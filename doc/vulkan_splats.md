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

Splats cast no shadows, contribute to no probe GI, do not participate in
froxel fog or MSAA, and are invisible to the RT sensors.

Reflections left this list on purpose: each cloud upload also bakes a small
rgba16f density/radiance volume, and the deferred shade's traced reflection
legs (water, glass, and the glossy path) march it — so a scan appears in a
pond or a chrome sphere, as a soft volumetric image, at ~0.03 ms. Primary
view only; the sensor wall above stands. `THREEPP_VK_SPLATVOL_OFF=1` restores
the pre-feature frame byte-exactly. See `splat_volume.glsl`, the `--water` /
`--metal` flags on the `gaussian_splats` example, and `VulkanSplatVolume_test`
for the asserted A/B.

The RT sensors left the list next, and through a proxy rather than a crack in
the pass: `threepp::splats::bakeSurface` fuses the median-depth AOV below into
a triangle mesh, `splats::makeSensorMesh` marks that mesh
`VulkanRenderer::kSensorOnlyLayer`, and
`VulkanRenderer::setSensorOnlySurfaces(true)` lets the scene's lidar beams hit
it — measured at 3.4 mm of range error against a 5 cm-voxel bake of a synthetic
plane (`VulkanSplatSurface_test`).

**Where the bake's cameras STAND is a choice, and the default is outside.**
`SurfaceBakeOptions::poseSet` is `Orbit` — a Fibonacci sphere around the fit
sphere, or a ring plus a top-down grid for a scan wider than tall — which is
right for an object, a facade or a site and WRONG for a scan of the inside of a
room: orbited, a room reconstructs the outside of its walls, and colliders end up
where nothing can walk. `PoseSet::Interior` stands the cameras in the scan
instead (the fit centre plus a few deterministically jittered stations, each
fanning directions that include straight up and straight down) and derives its
allocation gate from the fit extents rather than from a pose distance that no
longer means anything. Measured on the synthetic shell built at radius 1.000:
the orbit finds 1.0338, the interior finds 0.9818, and their signed volumes have
opposite sign — they are two different surfaces of one cloud, not two estimates
of one. `SurfaceMesh::Stats::beyondCentreSamples` is the report-only tell that an
orbit bake was the wrong call; `plans/splat-surface-bake.md` P6 has the numbers.

**Raster visibility is PER VIEW, and off by default.** "Secondary view" is not
a synonym for "sensor": an RGB camera preview (`CameraSensor`) and an editor
viewport pane are `addView` views too, and an untextured bake shell standing in
front of the cloud it approximates is a defect in both. So a secondary view
rasterizes sensor-only meshes only if it asked —
`setViewSensorSurfaces(handle, true)`, the caller reading depth off that
view — AND the scene master above is on. Both conditions, no exceptions, and no
view is granted it implicitly. The lidar side stays scene-level because a beam
list is already a per-consumer choice: whoever calls `scanLidar` asked for
those beams.

What that does NOT breach, and the reason the wording above is "through a
proxy": no sensor TRACE sees the splats. It sees a mesh baked from them, at
voxel resolution, with no colour. An RGB view that wants the real thing takes
the raster instead — `setViewSplats`, below. The splat pass remains absent from
every acceleration structure. The primary camera never
rasterizes a sensor-only mesh and no radiance trace — reflection, refraction,
shadow, GI, emissive NEE — carries it in its cull mask, so the picture stays
the splats' own. And the opt-in defaults OFF: until it is taken, the mesh's
TLAS instance carries mask 0, nothing renders or senses it, and a scene that
never asks is byte-unchanged.

## Splats in a secondary view (`setViewSplats`, the consumers default it ON)

The splat pass was PRIMARY-ONLY, and the reason was cost rather than
correctness: the radix sort scales with SPLAT COUNT, not view size, so a
640x480 sensor on the 5M-splat town pays the same 8-13 ms the primary does. The
consequence was that an RGB `CameraSensor` — a secondary view — saw EMPTY SPACE
where a scan stood, even after `bakeSurface` gave physics and lidar something to
touch.

That wall is now per view and opt-in:

```cpp
bool setViewSplats(uint32_t handle, bool enabled);// default false
bool viewSplats(uint32_t handle) const;
```

On, the view composites every cloud with the SAME deterministic compute
rasterizer the primary uses, at the same point in its own frame (linear HDR,
before its bloom / tonemap / TAA tail). `SplatPass` grew a small TARGET table
for it: slot 0 is the primary and the rest are handed to views that ask, each
with its own extent, tile grid, descriptor sets and UBO region. They SHARE the
sort scratch, which is safe for the same reason the pass already reuses it
across clouds — the stages are separated by its own compute→compute barriers,
and the targets record sequentially inside one command buffer.

Measured (RTX 4070, `VulkanMultiView_test`, one 256x192 secondary view, idle
GPU):

| cloud | frame, flag off | frame, flag on | per-view splat cost |
|---|---|---|---|
| 60k splats | 2.18 ms | 4.74 ms | 2.57 ms |
| 500k splats | 2.34 ms | 9.50 ms | 7.16 ms |

**The RENDERER defaults it off; the two consumers that are pictures turn it
on.** "Secondary view" covers both a sensor and a viewport, and for anything
whose output a HUMAN or a perception policy looks at, an empty space where a
scan stands is a defect rather than a saving — the more so because the OpenGL
backend draws a `SplatCloud` as an ordinary instanced mesh and has never had
this wall, so off-by-default there would have been a backend divergence, not a
policy. So:

- `CameraSensor` takes the flag for its view. `CameraSensor::renderSplats`
  (public, default true, read on every capture) is the opt-out, for a sensor
  that cannot afford the sort.
- The editor's `VulkanViewPane` takes it unconditionally: the primary viewport
  beside the pane draws splats, and so does the same pane on OpenGL.

Cost is the reason the switch exists, and on a heavy scene it is not small: the
sort scales with SPLAT COUNT, not view size, so a 128x96 wrist camera on the 5M
town pays the same ~8-13 ms the primary does, per opted-in view, per frame.
Three such views at once is the ceiling, and a fourth is refused. A sensor rig
over a town-scale scan should clear `renderSplats` on the views that only need
geometry and read the baked surface instead (above).

The walls that stand. Default OFF at the RENDERER, so a scene that does not ask
is byte-unchanged (asserted: the primary's own splat expansion is identical with
a secondary opted in). The DEPTH AOV stays primary-only — a secondary view's AOV
image is 1x1 by construction — and so does the debug checksum, which has one
readback buffer per frame slot. At most three views may hold the flag at once; a
fourth is refused on stderr. And the town-scale cost above is the CALLER's
choice, which is why this is a per-view switch and not a renderer mode: a
dynamic-LOD app can quarter it by submitting coarser chunks, but the submission
ranges are per-CLOUD state (`SplatCloud::setSubmitRanges`), so today a secondary
view draws whatever the app selected for the PRIMARY this frame. Per-view LOD
selection is the follow-up, recorded in `plans/splat-surface-bake.md`.

## The depth AOV: expected and median

`VulkanRenderer::setSplatDepthAov(true)` — or, since P0 of
`plans/splat-surface-bake.md`, `setSplatDepthAov(SplatDepthMode::Expected |
::Median | ::Off)`, the bool being `Expected` so every pre-mode caller is
untouched — exports what the raster already
computes. The accumulation loop carries `D += viewDist * alpha * T` alongside
the colour — it has to, because a rigid cloud's motion vectors are pure camera
reprojection of a depth and this is that depth — and `expDist = D / (1 - T)`
falls out at the end of the tile loop. The AOV is that number, written to an
`r32f` image and readable through `readGBufferAOV(GBufferAOV::SplatDepth)`.

Three decisions worth stating, because each has a plausible alternative:

**View-space distance in world units, not reversed-Z NDC.** The consumers are
outside the renderer — picking unprojects it, an occupancy build compares it
against a metric range — and neither wants to invert a projection first.

**Only where accumulated coverage passes 0.5; 0 everywhere else.** The same
gate, for the same reason, that the motion write above it uses: below half
coverage the geometry behind the translucent fringe is the better answer, and a
halo reported as a surface is a worse lie than no surface. That makes the AOV's
covered-pixel count much smaller than a "lit pixel" count — measured about 7×
smaller on the outer strip of the test cloud, which is nearly all fringe.

**Nearest cloud wins where two overlap**, via a compare before the store. Each
cloud is its own dispatch with a barrier between, so the read-modify-write
races nothing; without the compare, submission order rather than geometry would
decide what a sensor sees.

What the expected value is *not* is a surface. It sits behind the visible front
of a cloud by roughly its own thickness along the ray, so it localizes a wall
well and a canopy poorly. That is why `Median` exists: the view distance at
which accumulated transmittance crosses 0.5, one compare per contributing splat
in the same loop, LERPED between the two splats that straddle the crossing
(linear in `T`, which is all that is known between two samples — first-crossing
alone quantizes the answer to whichever splat happened to tip the sum, and
always to the far side of the interval). Same coverage gate, same nearest-wins
rule, same `r32f` image; and only the AOV changes — motion vectors and the
per-splat fog keep using the expected value they are defined against, so
switching statistic cannot move a pixel.

Measured on `VulkanSplat_test`'s synthetic cloud, over its 13 987 covered
pixels: the median is in front of the expected value at 80.8 % of them, by
+0.011 world units typically and +0.32 at most. The mean difference is −0.0001,
and that sign flip is worth stating because a fusion consumer will meet it: as
coverage falls toward the 0.5 gate the crossing degenerates to the LAST
contributing splat and lands *behind* the expected value — up to 2.0 units
behind on this cloud. It is a property of the statistic, not a defect: near the
gate half the light is still getting through, and there is no surface there to
find. Weight fused samples by coverage, or gate harder than 0.5, rather than
trusting a near-gate median.

Off by default, and off means the backing image is one texel: full-res `r32f`
per frame in flight is ~25 MB at 1080p that a scene without splats would never
read. Crossing `Off` reallocates the render-extent resources, so it is a setup
knob like `setGbufferMsaa`, not a per-frame one; changing only the statistic is
a UBO flag and reallocates nothing. With it off the frame is unchanged — the
golden in `VulkanSplat_test` matches byte-exact across the change.

### The renderer's own consumer: overlay occlusion

The compositor is compute and owns no depth attachment, so it reads the
G-buffer depth and writes nothing back. That is invisible until something is
drawn AFTER it against a depth buffer: the post-TAA overlay pass — wireframe
meshes, `Line`/`LineSegments`, world `Sprite`s, particle billboards — depth-
tests against the unjittered prepass buffer, which holds scene geometry only.
A gizmo behind a cloud therefore passed that test and drew over the cloud at
full strength, while GL, which draws the cloud last in its transparent pass,
blends it away.

`splat_overlay_depth.frag` closes it: one fullscreen depth-only draw between
the splat composite and the overlay draw, re-expressing the AOV's view-space
distance as the reverse-Z depth the overlay compares against, with the test set
to `GREATER` so a cloud behind a wall leaves the wall's depth alone. It runs on
the same attachment the overlay uses, hardware-MSAA path included.

Two consequences worth naming. The AOV turns itself **on** the first frame a
scene holds both clouds and overlay content — the stamp has nothing to read
otherwise — and stays on (the latch is sticky: turning it off again is a device
idle every time a gizmo is hidden). It reports `Median` when it was enabled for
this reason alone, since the front of the cloud is what an occlusion test wants;
an app that asked for `Expected` keeps `Expected`, one image and one statistic.
And the answer is **binary at the AOV's `coverage > 0.5` gate** — a pixel the
cloud more than half owns hides the overlay, one it owns less does not. GL
attenuates instead, so a fringe a pixel or two wide differs. A depth buffer has
no vocabulary for "60 % hidden", and the alternative is every overlay shader
sampling splat transmittance and blending against it.

**An overlay that hugs the cloud's surface must not be stamped against, and
`VulkanRenderer::kSplatUnoccludedOverlayLayer` is the exemption.** The AOV is
rasterized with the frame's TAA jitter and the stamp compares unjittered
overlay geometry against it, so at a fixed pixel both the stamp's depth and its
coverage gate wobble frame to frame — by the jitter amplitude times the local
depth gradient, which at a grazing view from a distance is metres, over a
region that at that framing is the whole surface. An overlay sitting NEAR the
stamped depth therefore flickers however the comparison is biased: measured in
`SplatOverlayFlicker_probe` (a baked-surface wireframe over its own scan,
grazing, 960×600, static camera, 31 frame pairs), the depth-tested hugging
overlay flips 25 px/frame across 18 % of its pixels, a one-voxel normal push
plus a 3×3-max stamp still flips 12.6, and a 3×3 **gate** makes it 7× worse —
the covered SET churns, not just the values. The exemption is the fix that
measures 0.0: the stamp records as a draw **inside** the overlay pass, overlays
that enable the layer draw before it (depth-tested against scene geometry —
the probe's wall hides 566 of their pixels — never against clouds), and
everything after it is occluded exactly as before. The editor's baked-surface
preview is the intended user: it is a picture OF the splat surface, so the
splat surface must not hide it. A grid should NOT take the layer — behind a
cloud is behind it.

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

## Point mode, and point clouds as splats

`SplatCloud::setPointMix(m)` draws the same cloud as dots: at `m = 1` every
splat is an opaque disc of `setPointSize` pixels (default 2) centred on its
mean, nearest wins where two overlap; at `m = 0` the pass is byte-identical
to one rendered before the knob existed; between, the projected covariance is
lerped toward the disc's and the opacity toward 1, so a slider sweep is a
continuous dissolve from surface to points. Both backends read the same two
numbers (`pointMix`, `pointSigma = (size / 2 + 0.5) / 3` px) and apply the
same two lerps — `splat_project.comp` / the GL vertex shader on the
covariance and opacity, `splat_raster.comp` / the GL fragment shader on the
falloff, where the Gaussian `exp(power)` is mixed with a one-pixel-feathered
disc `clamp((3 - sqrt(-2 power)) * sigma, 0, 1)`.

Nothing else in the pass changes, and that is the point: the sort, the
software depth test against the G-buffer, the depth AOV, the overlay depth
stamp and the reflection volume are all unaware of the mode. A point cloud
is therefore occluded by a mesh, occludes a gizmo and appears in a pond
exactly as the Gaussians do. `VulkanSplat_test` 2d holds the slab from the
occlusion check in place, switches to dots and asserts the same hiding, the
determinism of the dot frame, and that mix 0 restores the Gaussian frame.

Cost: the disc footprint is smaller than nearly every Gaussian's, so the
expansion and the raster shrink; the sort is unchanged.

**A colour-only point cloud loads into the same object.** A PLY whose
vertices carry `x y z` and optionally `red green blue`, `nx ny nz`,
`intensity` and scalar fields — a laser scan, a CloudCompare, MeshLab or
Open3D export — has no `f_dc_0`, so `SplatLoader::isSplatPly` says no and
`SplatLoader::isPointCloudPly` says yes (a mesh PLY with faces says no to
both). `SplatLoader::loadPointCloudPly` turns every point into one degree-0
Gaussian: DC colour from the point's colour (white without one, grey from
`intensity` when that is all there is), opacity 1, and one isotropic sigma
for the whole cloud, `1.0 ×` its median nearest-neighbour distance
(`splats::medianNeighbourSpacing`, a hash-grid estimate over a 20k-point
sample; `PointCloudOptions::sigma` overrides it — a Poisson sampling's
nearest neighbour sits at about half its mean pitch, which is why the factor
is 1 and not the 0.6 a lattice would need). A point with normals
becomes a disc facing them, its sigma along the normal `0.15 ×` the others.
So the cloud renders as a closed surface at mix 0 and as its dots at mix 1.
Binary little- and big-endian and ascii bodies, any numeric property type,
and elements before or after the vertices all parse. The editor's import,
the `gaussian_splats` example and the Python bindings
(`SplatLoader.load_point_cloud_ply`, `SplatCloud.point_mix` /
`.point_size`) take the same path; the example's `--points` flag starts in
point mode, `--point-size` sets the disc, and P dissolves between the looks.

**A point cloud gets its collider straight from its points.**
`splats::buildPointSurface` (`threepp/splats/PointSurface.hpp`, CPU, every
backend) voxel-hashes the transformed means at `2 ×` their median spacing,
evaluates the union-of-balls field `max(0, 1 - d / voxel)` on the nodes
around occupied voxels, runs marching cubes at `0.5` with welded vertices
and outward winding, and drops islands under 64 cells. It returns the same
`SurfaceMesh` the depth-fusion bake does, so PhysX cooking, the sensor mesh
and the editor's preview overlay are shared. The surface is an OFFSET one:
it sits half a voxel outside the samples on both sides of a sheet, which is
the price of a method with no normals and no free-space carving. That is why
it is the wrong tool for a Gaussian scan — every fog splat's centre is a
point to it — and why the editor's Surface section routes by `Method`: Auto
takes the point route for a cloud imported as a point cloud and the fusion
bake otherwise, and the other two force one. `PointSurface_test` pins the
slab-around-a-sheet geometry, a closed manifold shell at its radius,
determinism, the transform, the island filter and the voxel ceiling.

**A scan can be a moving body.** The Surface section's `Body` is Static,
Dynamic or Kinematic (`SplatSurfaceConfig::body`, with `mass` and a V-HACD
`hulls` budget). Static cooks the triangles as they are. PhysX will not
simulate a triangle mesh, so a moving body has its baked surface split into
convex hulls (`decomposeConvex`, one hull of every vertex when V-HACD is not
linked) and welded into one compound actor at the cloud's pose; a Dynamic
cloud is bound to the actor and follows it, parents and all, while a
Kinematic one drives the actor from its own transform each frame. The
sensor mesh of a moving body is parented under the cloud in its local frame
instead of sitting at the scene root, so it rides along. The editor selftest
drops a 1500-point shell scan (direct route, 8 hulls, 2 kg) onto the fused
floor scan: it rests at 4.29 over a floor whose top is 4.045, radius 0.25.

**A cloud is saved by reference.** `ObjectExporter` writes a `SplatCloud` as
a `threeppSplat` block: the source path (relative to the document when it
can be; the editor's `SplatImportConfig` mark is where it comes from), the
import ops to replay (`cull`, `lod`, `pointCloud`), the point-mode look, and
the splat count for the record. Never the splats. Inside a `.tpz` the source
file is copied in as `splats/<uuid>.<ext>` whatever its container, so the
archive is self-contained; a cloud with no file (a procedural one) gets a
`.ply` member, or a sidecar `splats/<uuid>.ply` next to a loose document,
written by `SplatLoader::writePly`. `ObjectLoader` re-imports by content
(SOG, 3DGS `.ply`, or a colour point cloud), replays the cull unless a LOD
table is resident, and re-stamps the source mark with this machine's path or
the archive-and-entry mark. A file that is gone leaves a placed, named
`Object3D` and a warning. The editor's play snapshot is the one document
that carries neither a path nor bytes: it keeps the live clouds by uuid and
hands them back through `ObjectLoader::setSplatCloudResolver`, so Stop
re-places the same object instead of reloading a gigabyte.
`ObjectExporterSplat_test` covers the reference, sidecar, archive, pathless
and missing-file cases.

Still out of scope, unchanged: a point cloud is not in any acceleration
structure either, so the sensor wall above applies to it as written.

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

## Host memory per splat (measured 2026-08-07)

Two independent cuts landed the same day. Together a degree-3 splat went from
**606 B to 423 B** including the GL-side data textures, and from 430 B to 247 B
without them — 1.4 GiB rather than 2.4 GiB to hold a 6M-splat scan on Vulkan.

### The rotation: `std::vector<Quaternion>` → `SplatQuat`

`SplatData::byteSize()`, over a generated cloud at each SH degree:

| SH degree | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| before | 168 B | 204 B | 264 B | 348 B |
| **now** | **56 B** | **92 B** | **152 B** | **236 B** |

The difference is one field. `rotations` was a `std::vector<Quaternion>`, and
`sizeof(Quaternion)` is **128**, not 16: its four components are `float_view`
(a float plus a pointer to the owner's change callback, 16 bytes each) and it
carries a `std::function<void()>` besides. 112 bytes a splat of notification
machinery that the splat path never subscribes to — rotations are written once
by a loader and read by `computeCovariance` — which is 672 MB on a 6M-splat
scan, on **both** backends, before any renderer exists, since the data is loaded
first. That is more than the ~1 GB GL-side copy that lazy GL resources went
after, for a fraction of the change. `SplatQuat` (four plain floats, in
`SplatData.hpp`) is the fix, with a `static_assert` on its size so it stays one.

### The instancing: `InstancedMesh` → `Mesh` + `InstancedBufferGeometry`

`SplatCloud` used
to derive from `InstancedMesh`, whose constructor allocates 16 floats of
`instanceMatrix` per instance. The splat path never wrote anything but identity
into them — a splat's placement is its mean and its covariance, not a transform —
and the splat shader does not even declare the attribute, but `GLObjects::update`
uploads `instanceMatrix` for *every* `InstancedMesh` in the render list
regardless, so the identity matrices cost 64 B/splat of host memory **and** 64 of
VRAM. At SH degree 0 that was more than the entire splat payload.

It now derives from `Mesh` over an `InstancedBufferGeometry`, which carries only
the attributes the shader declares — one of them, the sorted draw order, which
shrank from a vec3 `instanceColor` to a single float on the way since only `.x`
was ever read. `SplatCloud::cpuBytes()`, measured over a 200k cloud with no GL
frame drawn:

| SH degree | 0 | 3 |
|---|---|---|
| as InstancedMesh | 139.3 B | 319.3 B |
| **as Mesh** | **67.3 B** | **247.3 B** |

72 B/splat either way — 432 MB of host memory at 6M splats, plus 384 MB of VRAM
that GL no longer uploads. The GL render is byte-identical across the change.

This needed three renderer stubs filled in, all of them pre-existing `if (false)`
/ commented-out placeholders for the mechanism three.js already uses:
`InstancedBufferGeometry`'s draw call in `GLRenderer`, the per-attribute divisor
in `GLBindingStates`, and `InstancedBufferAttribute` itself. Nothing about the
`InstancedMesh` path changed — the new branches are additive, and
`InstancedMesh` remains the right answer whenever instances really are one mesh
at many transforms.

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
