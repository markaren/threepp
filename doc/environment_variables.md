# Environment variables

The library reads the `THREEPP_*` environment variables below at run time. None of
them is needed: each has a default, and most are diagnostics or A/B hatches kept for
judging a change against the path it replaced. Set them in the shell (or in the
process environment before the renderer is created) to use them.

Most Vulkan knobs are read through `threepp::vulkan::env` (`EnvKnob.hpp`), so a value
means the same thing whichever knob it is given to:

- **flag**: unset or empty gives the default. Any other value turns the knob on,
  unless it starts with `0`. So `1`, `yes` and `on` are on, `0` is off.
- **integer** and **number**: unset, empty or not a number gives the caller's
  default; otherwise the value is parsed as an integer (base 10 unless noted) or as a
  floating-point number.

Knobs that take a string (a directory, a name list, a mode name) read `std::getenv`
directly. A few older switches are parsed by hand; their rule is given in the table.

The **Read** column says when the value is taken: *once* means once per process (the
result is cached), *per renderer* means when a renderer (or one of its passes) is
constructed, and *per call* means each time the code runs, for example every frame.
Changing a variable after it has been read once has no effect.

## Logging

The library's console messages go through one facade (`threepp/utils/Log.hpp`).
Every message has a category and a level (trace, debug, info, warning, error), and
goes to stderr as one line, `[category] warning: text` (info, debug and trace lines
carry no level word). The default threshold is info for every category, so a plain
run prints only warnings, errors and a few facts such as the GPU. An application can
also set thresholds with `threepp::log::setLevel` and capture records with
`threepp::log::setSink`; `THREEPP_LOG` wins over `setLevel`.

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_LOG` | string `<category>:<level>,...` | unset: info everywhere | Sets thresholds. A category covers itself and the categories below it (`vulkan` covers `vulkan.probe`), `*` covers all, and a bare level applies to `*`; a later entry wins for what it covers. Levels: `trace`, `debug`, `info`, `warn`, `error`, `off`. Example: `THREEPP_LOG=vulkan.probe:debug,core:warn`. | once | |
| `THREEPP_LOG_FILE` | string (file path) | unset: no file | Also writes every record that passes its threshold to this file, each line prefixed with the frame number and the seconds since the first record (`f123 4.567s [vulkan.validation] error: ...`). | once | diagnostic |

Categories (`THREEPP_LOG=<category>:debug` shows a category's diagnostics):

| Category | What it reports |
|---|---|
| `core.animation` | animation tracks, property bindings, skeletons |
| `core.canvas` | the window, GLFW errors, the GL context |
| `core.color` | unknown color names |
| `core.controls` | orbit, trackball and transform controls |
| `core.geometry` | bounding volumes, line distances, geometry merges |
| `core.gl` | the GLAD loader |
| `core.material` | material values that are not recognised |
| `core.object` | the scene graph (`Object3D::add`) |
| `core.ocean` | the ocean look switches of `DisplacedMesh` |
| `core.raycaster` | `Raycaster::setFromCamera` |
| `core.string` | number parsing |
| `extras.sitl` | the ArduPilot SITL bridge: dropped packets, SITL resets |
| `gl.program` | shader compile and link logs (error), link warnings, unsupported color spaces and tone mappings |
| `gl.texture` | textures that cannot be uploaded, texture unit overflow, incomplete framebuffers |
| `gl.uniform` | uniform values the renderer cannot set |
| `gl.state` | blend modes it does not know |
| `gl.shadow` | shadow map types and lights that cannot cast shadows |
| `gl.renderer` | unknown draw modes, `writeFramebuffer` folders |
| `gl.postprocessing` | post-processing passes |
| `loader.gltf` | glTF: files that do not load, skipped primitives, unsupported material extensions; each light it loads hidden (debug) |
| `loader.obj` | OBJ: lines it does not understand, dropped faces and uvs |
| `loader.stl` | STL |
| `loader.collada` | Collada: no visual scene, skins whose joints are missing |
| `loader.usd` | USD: loading, conversion, packages and fetched URLs |
| `loader.exr` | OpenEXR: why an image does not decode |
| `loader.dds` | DDS: why a texture does not load, truncated mip levels |
| `loader.rgbe` | Radiance HDR |
| `loader.image` | `TextureLoader`, `CubeTextureLoader` and `ImageLoader` (WebP, TIFF) |
| `loader.font` | fonts |
| `loader.svg` | SVG path commands and fill rules it does not support |
| `loader.urdf` | URDF and xacro: missing geometry loaders, duplicate link elements, xacro warnings and errors |
| `loader.object` | the JSON scene loader and exporter: the messages `warnings()` also returns, files that do not open |
| `loader.model` | `ModelLoader` and background loads (`AsyncGroup`) |
| `vulkan.validation` | validation layer errors and warnings, strict-validation verdicts |
| `vulkan.device` | the GPU, ray tracing, surface, swapchain and present mode (extension notes at debug) |
| `vulkan.pipeline_cache` | pipeline cache load and save (debug) |
| `vulkan.pipeline_stats` | `THREEPP_VULKAN_PIPELINE_STATS` output |
| `vulkan.startup` | the time of each renderer startup step (debug) |
| `vulkan.exit` | the time of each renderer and context teardown step (debug) |
| `vulkan.renderer` | calls the renderer ignores or cannot honour |
| `vulkan.frame` | the headless clock, swapchain recreation and view sizes (debug), each acquire (trace) |
| `vulkan.overlay` | HUD and 2D overlay passes |
| `vulkan.upscale` | DLSS, FSR and NGX (`THREEPP_DLSS_LOG`) |
| `vulkan.shade` | deferred shade variants and their compile times |
| `vulkan.rendergraph` | render graph diagnostics; each graph's passes when it changes (trace) |
| `vulkan.memory` | `THREEPP_VK_MEMDUMP` reports, transient pools |
| `vulkan.geometry` | BLAS and TLAS builds, auto-LOD, skinning, deforming geometry |
| `vulkan.displacement` | displacementMap refinement; each refined geometry's time (debug) |
| `vulkan.interop` | vertex and frame interop |
| `vulkan.texture` | textures and environment maps |
| `vulkan.scene` | scene build: walk checks, morph targets; slot edits and deltas handed to the rebuild (debug) |
| `vulkan.draw_stats` | `THREEPP_VK_DRAW_STATS` output |
| `vulkan.cpuprof` | `THREEPP_CPU_PHASE_PROFILE` output |
| `vulkan.probe` | the probe grid fit and its decisions (debug) |
| `vulkan.shadow` | the shadow mode; static shadow cache generation steps (debug) and per-frame cascade and layer lines (trace) |
| `vulkan.splat` | Gaussian splats, `THREEPP_VK_SPLAT_CHECKSUM` output |
| `vulkan.particles` | particle fields |
| `vulkan.ocean` | the ocean spectrum and FFT |
| `vulkan.pathtracer` | the path tracer |
| `python.renderer` | renderer properties the Python bindings ignore (`renderer.fsr`, `renderer.dlss` without the upscaler) |
| `editor` | the editor's own messages: what it opened, saved, imported, and what failed |
| `editor.physics` | play sessions: rigid bodies, robots, joints, vehicles, soft bodies and triggers that are not simulated as authored |
| `editor.character` | play sessions: character controllers |
| `editor.granular` | play sessions: granular chutes |
| `editor.particles` | play sessions: particle fields |
| `editor.sensors` | play sessions: sensors that do not run, recordings that cannot be written |
| `editor.splat_surface` | play sessions: splat scans baked into colliders |
| `editor.audio` | play sessions: sounds, the audio device, acoustics |
| `editor.robot` | robots revived from a saved document |
| `editor.script` | Python scripts: errors and callbacks that never run |
| `player` | `threepp_player`: the scene it opened, document warnings, the view, what would not start |

The editor shows the records of every category in its Console tab, the `editor` ones
without their category. `threepp_player` writes them to stderr; its per-episode lines and
summary stay on stdout.

These variables were replaced by categories and are no longer read:
`THREEPP_PROBE_LOG` (`vulkan.probe:debug`), `THREEPP_DISP_LOG` (`vulkan.displacement:debug`),
`THREEPP_VULKAN_SLOTS_LOG` (`vulkan.scene:debug`), `THREEPP_VK_SHADOW_CACHE_DEBUG`
(`vulkan.shadow:trace`), `THREEPP_VK_STARTUP_TIMING` (`vulkan.startup:debug`),
`THREEPP_VK_EXIT_TIMING` (`vulkan.exit:debug`), `THREEPP_VK_TRACE_ACQUIRE`
(`vulkan.frame:trace`) and `THREEPP_RG_DUMP` (`vulkan.rendergraph:trace`).

## Window, context and device

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_GLFW_PLATFORM` | string | unset: native platform (Linux headless with no `DISPLAY` or `WAYLAND_DISPLAY` picks the Null platform) | `null` forces GLFW's Null platform anywhere, so the display-free path can run on a machine with a display (needs GLFW 3.4; older GLFW prints a warning and ignores it). | per canvas init | |
| `THREEPP_REQUIRE_HARDWARE_GL` | string | unset: a warning only | When the OpenGL context is a software rasteriser, exactly `1` throws instead of printing a warning. | per GL context | |
| `THREEPP_VULKAN_VALIDATION` | flag | on in debug builds, off in release builds | Turns the Vulkan validation layer (and its debug messenger) on or off. | per renderer | diagnostic |
| `THREEPP_VULKAN_STRICT_VALIDATION` | flag | off | With validation active, any validation error makes the process exit with a non-zero code at termination; without the layer it prints that nothing is checked. | per renderer | diagnostic |
| `THREEPP_VULKAN_NO_RT` | flag | off | Runs the no-ray-tracing mode on any device, so that mode can be exercised on RT hardware. | per renderer | A/B hatch |
| `THREEPP_VULKAN_SUPPRESS_PRESENT` | flag | on for the hidden-window headless fallback, off for a real headless surface | On a headless canvas, skips presents and pins one swapchain image per in-flight slot. Offscreen rendering never presents, whatever the value. | per renderer | |
| `THREEPP_VK_ASYNC_PILOT` | integer | 0 (off) | Selects the async-compute pilot mode 1 to 5 (1 moves compute work to its own queue, 2 to 5 are controls that split its costs); other values are off. | once | A/B hatch |
| `THREEPP_VK_NO_ALIAS` | flag | off | Gives every frame image its own allocation instead of sharing memory through the transient pools (for bisecting aliasing and for scratch-image readbacks). | per renderer | A/B hatch |

## Upscaling and denoising

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_DLSS_JITTER_SIGN_X` | string | negative | `neg` or `pos` forces the sign of the X jitter handed to DLSS; anything else keeps the default. | once | diagnostic |
| `THREEPP_DLSS_JITTER_SIGN_Y` | string | positive | `neg` or `pos` forces the sign of the Y jitter handed to DLSS; anything else keeps the default. | once | diagnostic |
| `THREEPP_DLSS_LOG` | flag | off | Turns on NGX's own verbose log, written to `vulkan.upscale` at info, each line with the time since the previous one. | per NGX init | diagnostic |
| `THREEPP_DENOISE` | flag | on | `0` turns off the deferred SVGF denoiser, to tell a shading artifact from a temporal or denoise one. | per renderer | A/B hatch |
| `THREEPP_DENOISE_ATROUS_PASSES` | integer | 4 | Number of a-trous passes of the deferred GI denoise filter, clamped to 2..4 (fewer passes forces the old global cut). | once | A/B hatch |

## Deferred shading and probe GI

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_VK_SHADE_SPEC_MASK` | integer (decimal or `0x` hex) | 0 | Extra shade push-constant flag bits made compile-time specialisation constants, ORed onto the MSAA bits, so the driver drops the code they turn off. | per renderer | diagnostic |
| `THREEPP_VK_SHADE_SPEC_PROBE` | flag | off | Makes the probe GI gate of the deferred shade a compile-time constant. | per renderer | diagnostic |
| `THREEPP_VK_DIAG_NOGATHER` | flag | off | The probe path never runs its stochastic gather (measures the gather's cost ceiling). | once | diagnostic |
| `THREEPP_VK_DIAG_NOCOAT` | flag | off | Uploads every material without its clearcoat (also drops the water glint's and the path tracer's coat). | once | diagnostic |
| `THREEPP_VK_PROBE_CASCADES` | integer | unset: the renderer's setting (default 0) | Probe cascade count, clamped to 0..4 (0 = scene fit always); when set it wins over `setCascades`. | per renderer | |
| `THREEPP_VK_PROBE_CASCADE_BASE` | number | 1 (metres) | Finest probe spacing of the cascades; values at or below 0 keep the default. | per renderer | |
| `THREEPP_VK_PROBE_SCENE_FIT_MAX` | number | 2 (metres) | Largest probe spacing the scene fit may use before the camera-centred cascades take over; values at or below 0 keep the default. | per renderer | |
| `THREEPP_VK_PROBE_FIT_SHAPE` | number | 2 | Shape factor of the scene fit's grid: the probe budget is split by the scene box's proportions, no axis below 1/v of its fixed count (1 = the fixed 32x16x32 grid). | per renderer | |
| `THREEPP_VK_PROBE_BIAS` | number | 0.25 | Offset of the probe lookup point off the surface, in cells. | per renderer | |
| `THREEPP_VK_PROBE_BIAS_AXIS` | flag | off | Measures that offset in the cell size along the surface normal instead of the smallest spacing. | per renderer | |
| `THREEPP_VK_PROBE_RELOCATE` | number | 0 (off) | Probe relocation: each update moves its probe out of geometry and to at least this many minimum spacings from the nearest surface. | per renderer | |
| `THREEPP_VK_PROBE_FINAL_GATHER` | flag | on | `0` turns off the final gather, so the probe field is no longer the visible surface's indirect light. | per renderer | A/B hatch |
| `THREEPP_VK_PROBE_WARMUP` | flag | on | `0` turns off the boosted probe updates after a full clear (flat budget and the shader's flat history cap). | per renderer | A/B hatch |
| `THREEPP_VK_PROBE_FAR` | flag | off | Beyond the coarsest probe cascade the probe path fills with the ray-free sky term instead of its gather. | once | |
| `THREEPP_VK_FOG_HALF_RES` | flag | off | Starts the fog march (`fog_march.comp`) at half resolution, as `setFogMarchHalfRes(true)` would; the setter wins. | per renderer | A/B hatch |

## Path tracer

These four are parsed by hand: on unless the value is exactly `0`.

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_PT_THIN_GLASS_NEE` | flag | on | An emitter seen through a thin pane or a flat blend is lit by the emitter NEE alone; `0` restores the old way where both counted it (about twice the light). | per call (each dispatch) | A/B hatch |
| `THREEPP_PT_LAST_BOUNCE_MIS` | flag | on | The last path vertex traces the bounce its emitter NEE is weighted against, for the emission alone. | per call (each dispatch) | A/B hatch |
| `THREEPP_PT_AMBIENT_LIGHTS` | flag | on | AmbientLight and HemisphereLight light the path as the radiance an escaping ray meets; `0`: neither lights a surface. | per call (each dispatch) | A/B hatch |
| `THREEPP_PT_BLEND_PASS` | flag | on | Flat alpha-blended and additive surfaces are null events the ray passes through (additive ones add their glow); `0`: the old opaque-card and tint behaviour. | per call (each dispatch) | A/B hatch |

## Shadows

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_VK_SHADOW_MODE` | string | `auto` | Seeds the shadow mode: `auto`, `rt` (ray traced), `sm` (shadow maps) or `off`; anything else prints a warning and uses `auto`. The setter wins. | per renderer | |
| `THREEPP_VK_SHADOW_MAPSIZE` | integer | 0 (no cap) | Cap on the shadow map size. | per renderer | |
| `THREEPP_VK_SHADOW_CASCADES` | integer | 4 | Number of directional shadow map cascades, clamped to 1..4; the setter wins. | per renderer | A/B hatch |
| `THREEPP_VK_SHADOW_LAMPS` | string `<spots>,<points>` | 4 spots, 8 points | Number of shadowed spot and point lights, each clamped to the maximum; the points part is optional. The setters win. | per renderer | A/B hatch |
| `THREEPP_VK_SHADOW_CACHE` | flag | on | `0` turns off the static shadow caches of the cascades and the lamp layers. | per renderer | A/B hatch |
| `THREEPP_VK_SHADOW_CACHE_MARGIN` | number | 1.15 | How much wider than its slice a cached cascade's box is frozen, clamped to 1..2. | per renderer | |
| `THREEPP_VK_SHADOW_DWELL` | flag | on | `0` turns off the moving-caster shadow dwell and top-up machinery. | once | A/B hatch |

## Geometry and acceleration structures

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_NO_PACK` | flag | off | Forces every vertex attribute buffer back to tightly packed float. | once | A/B hatch |
| `THREEPP_NO_FORCE_OPAQUE` | flag | off | Stops marking TLAS instances of materials no ray query can reject as force-opaque. | once | A/B hatch |
| `THREEPP_VK_BLAS_COMPACT` | flag | on | `0` turns off compaction of static BLASes (they stay at their build size). | per renderer | A/B hatch |
| `THREEPP_VK_TLAS_FILL_INPLACE` | flag | on | `0` fills the TLAS instance list from empty every time instead of rewriting only the changed rows. | once | A/B hatch |

## Scene build and draw records

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_VULKAN_NO_VERSIONS` | flag | off | Never uses the version-based scene checks (flat snapshot check, structure version); the replay walk decides. | once | A/B hatch |
| `THREEPP_VULKAN_NO_SLOTS` | flag | off | Every scene content change becomes a structural rebuild in the dense layout instead of a slot edit. | once | A/B hatch |
| `THREEPP_VULKAN_NO_DELTA` | flag | off | Turns off the scene delta path (in-place add and remove of subtrees). | once | A/B hatch |
| `THREEPP_VULKAN_SYNC_ADMIT` | flag | off | Admits hidden meshes synchronously and encodes a slot edit's textures one after another on the render thread. | once | A/B hatch |
| `THREEPP_VULKAN_WALK_CHECK` | flag | off | After a delta or a reusing expansion, also walks the scene in full and compares node for node; mismatches are `vulkan.scene` warnings. | once | diagnostic |
| `THREEPP_VK_SPAN_DRAW_CACHE` | flag | on | `0` sends every instanced span through the per-entry draw record loop instead of the per-span draw cache. | once | A/B hatch |
| `THREEPP_VK_SPAN_RECORDS` | flag | on | `0` rebuilds every cached span's draw records each frame instead of keeping them between frames. | once | A/B hatch |
| `THREEPP_VK_SPAN_RECORDS_CHECK` | flag | off | Checks every kept span record against what the entries say each frame; a stale one ends the process. | once | diagnostic |

## Textures

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_NO_BC` | flag | off | Turns off BC7 transcoding and the DDS block pass-through; textures upload uncompressed. | once | A/B hatch |
| `THREEPP_VK_ANISO` | number | unset: auto policy | Startup texture anisotropy override, as `setTextureAnisotropy`: 0 = auto, 1..16 forces a level (clamped to the device maximum). | per renderer | |

## CPU threading and recording

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_VK_CPU_PAR` | flag | on | `0` runs the chunked CPU walks of the frame prep on one thread. | once | A/B hatch |
| `THREEPP_VK_CPU_PAR_CHUNKS` | integer | hardware threads, at most 16 | Most chunks a parallel pass or a split recording uses (capped at 64; values below 1 keep the default). | once for the walks, per call for split recording | |
| `THREEPP_VK_CPU_PAR_MIN` | integer | 1024 | Fewest items per chunk (values below 1 keep the default). | once | |
| `THREEPP_VK_CPU_TASKS` | flag | on | `0` runs the frame prep's independent chains (cull to indirect build, shadow map prepare) one after another. | once | A/B hatch |
| `THREEPP_VK_CPU_VIEWS` | flag | on | `0` prepares the secondary views one after another instead of at once on the thread pool. | once | A/B hatch |
| `THREEPP_VK_CPU_RECORD` | flag | on | `0` records long split passes inline instead of into secondary command buffers on the pool. | once | A/B hatch |

## Gaussian splats

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_VK_SPLAT_CHECKSUM` | flag | off | Hashes the sorted splat key and payload arrays, prints the entry count and checks the scan and sort invariants. | per call | diagnostic |
| `THREEPP_VK_SPLAT_NOMOTION` | flag | off | Splats skip the motion vector write. | once | A/B hatch |
| `THREEPP_VK_SPLAT_NOFOG` | flag | off | Splats skip the per-splat fog. | once | A/B hatch |
| `THREEPP_VK_SPLAT_NOINDIRECT` | flag | off | Restores the worst-case sort dispatches instead of indirect dispatches sized from the data. | per call | A/B hatch |
| `THREEPP_VK_SPLAT_RANGESPLIT` | integer | unset: the app's submit ranges | Values above 1 submit each cloud as that many contiguous ranges covering all of it (an identity, for proving the range path). | per call (each frame) | diagnostic |
| `THREEPP_VK_SPLATVOL_OFF` | flag | off | Never bakes the splat reflection volume: no image, no dispatch, no VRAM. | per renderer | A/B hatch |
| `THREEPP_VK_SPLATVOL_RES` | integer | 128 | Longest-axis voxel budget of the reflection volume bake, clamped to 16..256. | per renderer | |
| `THREEPP_VK_SPLATVOL_SIGMA` | number | 1 | Multiplier on the reflection volume's sigma at sample time (values at or below 0 keep 1). | once | A/B hatch |

## Ocean and wakes

The ocean wave-model switches marked "tri-state" are parsed by hand: unset
or empty leaves the mesh's own `DisplacedMesh::Params` field in charge, a value
starting with `0` forces the feature off for every ocean, anything else forces it on.

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_OCEAN_CWM` | flag (tri-state) | the mesh's `Params::cwm` (false) | Forces the choppy-wave model (full Lagrangian horizontal displacement with a Jacobian limiter) on or off. | once | |
| `THREEPP_OCEAN_BOUND_WAVES` | flag (tri-state) | the mesh's `Params::boundWaves` (false) | Forces the second-order bound waves (higher, sharper crests and flatter troughs) on or off. | once | |
| `THREEPP_OCEAN_WAVE_REFLECTIONS` | flag (tri-state) | the mesh's `Params::waveReflections` (false) | Forces a reflection ray that meets another wave to shade it as water (instead of whatever lies beyond) on or off. | once | |
| `THREEPP_OCEAN_SHORT_WAVE_SPECTRUM` | flag (tri-state) | the mesh's `Params::shortWaveSpectrum` (false) | Forces the short-wave (gravity-capillary) tail of the spectrum on or off. | once | |
| `THREEPP_OCEAN_HULL_CONTACT` | flag (tri-state) | the mesh's `Params::hullContact` (false) | Forces hull contact (the sea meets a vessel's side instead of easing onto her waterline plane) on or off. | once | |
| `THREEPP_OCEAN_CREST_SHARPNESS` | number (`atof`) | unset or negative: the mesh's `Params::crestSharpness` | Forces the crest sharpness for every ocean, clamped to 0..4; it applies only while the choppy-wave model is on. | once | |
| `THREEPP_OCEAN_FFT` | string (`shared` or `butterfly`) | `shared` | Default IFFT path for every ocean FFT: the one-dispatch shared-memory path or the butterfly passes; other values print a warning and use `shared`. | once | A/B hatch |
| `THREEPP_OCEAN_CREST_FOAM_RES` | integer | 2048 | Cap on the crest foam texture's edge, clamped to 256..4096. | once | |
| `THREEPP_OCEAN_CREST_FOAM_MAX` | number | 1.5 | Most foam a breaking crest deposits (at least 1). | once | |
| `THREEPP_OCEAN_CREST_FOAM_TAU` | number | 2.5 (seconds) | E-folding time of the foam a breaker leaves (at least 0.05). | once | |
| `THREEPP_OCEAN_FOLD_FOAM_J` | number | 0.76 | Jacobian below which the drawn surface takes fold foam. | once | |
| `THREEPP_OCEAN_FOLD_FOAM_RAMP` | number | 0.2 | Jacobian ramp over which fold foam goes from none to full. | once | |
| `THREEPP_OCEAN_FOLD_FOAM_LENGTH` | number | 3 (metres) | Shortest wavelength that counts for fold foam. | once | |
| `THREEPP_OCEAN_CREST_LINE_FOAM_REACH` | number | 1.0 | How far a foam cap reaches along its crest, as a multiple of that shortest wavelength. | once | |
| `THREEPP_WAKE_HEIGHTS` | flag (hand-parsed) | on | Exactly `0` turns off the wake ripples' wave heights (`WakeField::rippleHeights` default), leaving slopes only. | once | A/B hatch |
| `THREEPP_WAKE_BREAKING` | flag (hand-parsed) | off | Exactly `1` turns on breaking of the wake ripples' steep waves (`WakeField::rippleBreaking` default). | once | |
| `THREEPP_WAKE_BREAK_SLOPE0` | number | the field's `rippleBreakSlope[0]` (0.25) | Slope at which a wake ripple crest starts to break. | once | |
| `THREEPP_WAKE_BREAK_SLOPE1` | number | the field's `rippleBreakSlope[1]` (0.45) | Slope at which a wake ripple crest is fully broken. | once | |
| `THREEPP_WAKE_BREAK_FOAM` | number | the field's `rippleBreakFoam` (0.30) | Foam film a fully broken wake crest lays in the field. | once | |
| `THREEPP_WAKE_BREAK_AIR` | number | the field's `rippleBreakAir` (0.20) | Bubbles a fully broken wake crest lays in the field. | once | |

## Diagnostics, timings and dumps

| Variable | Kind | Default | Effect | Read | Notes |
|---|---|---|---|---|---|
| `THREEPP_CPU_PHASE_PROFILE` | flag | off | Turns on the CPU phase profiler, which logs each named phase's mean and worst milliseconds per window of frames to `vulkan.cpuprof`. | once | diagnostic |
| `THREEPP_CPU_PHASE_WINDOW` | integer | 300 | Frames per printed profiler block. | once | diagnostic |
| `THREEPP_CPU_PHASE_SPIKE` | integer | 0 (off) | A frame whose scene total passes this many milliseconds prints its own phases. | once | diagnostic |
| `THREEPP_VK_MEMDUMP` | flag | off | Dumps GPU memory statistics to `vulkan.memory` after init, after each structural scene build and at frame 8, and keeps the live-buffer registry for them. | per renderer and per scene build; once for frame 8 and the registry | diagnostic |
| `THREEPP_VK_DRAW_STATS` | integer | 0 (off) | Every N primary frames logs what the frame prep walked to `vulkan.draw_stats`; when above 0 also logs which geometry edits drain the device. | once | diagnostic |
| `THREEPP_VULKAN_PIPELINE_STATS` | flag | off | Captures pipeline executable statistics (changes how pipelines are compiled). | per renderer | diagnostic |
| `THREEPP_SHADE_DUMP_DIR` | string (directory) | unset: no dump | `debugHashShadeImages` also writes each shade image's bytes to this directory as `<name>_f<serial>.raw`. | per call | diagnostic |
| `THREEPP_SHADE_DUMP_MIN` | integer | 0 | First frame serial of the shade image dump window. | per call | diagnostic |
| `THREEPP_SHADE_DUMP_MAX` | integer | 8 | Last frame serial of the shade image dump window. | per call | diagnostic |
| `THREEPP_SHADE_DUMP_NAMES` | string (comma-separated) | unset: all images | Substring filter on the image names the shade dump writes. | per call | diagnostic |
