# WgpuRenderer Feature Parity Matrix

Comparison of WgpuRenderer (WebGPU/wgpu-native) vs GLRenderer (OpenGL 3.3) feature coverage.

Last updated: 2026-04-19


## Legend


| Symbol | Meaning                                |
| ------ | -------------------------------------- |
| Y      | Fully implemented and tested           |
| P      | Partial — implemented with limitations |
| N      | Not implemented                        |
| -      | Not applicable                         |


---

## Materials


| Feature              | GL  | Wgpu | Notes                                                     |
| -------------------- | --- | ---- | --------------------------------------------------------- |
| MeshBasicMaterial    | Y   | Y    |                                                           |
| MeshLambertMaterial  | Y   | Y    |                                                           |
| MeshPhongMaterial    | Y   | Y    | Blinn-Phong specular + all maps                          |
| MeshStandardMaterial | Y   | Y    | PBR metalness/roughness with all texture maps             |
| MeshToonMaterial     | Y   | Y    | Gradient map support                                      |
| MeshNormalMaterial   | Y   | Y    | Maps world-space normals to RGB (N*0.5+0.5)               |
| MeshDepthMaterial    | Y   | Y    | Maps clip-space depth to brightness (1-depth)             |
| MeshMatcapMaterial   | Y   | P    | Renders as unlit white; no matcap texture lookup          |
| LineBasicMaterial    | Y   | Y    |                                                           |
| LineDashedMaterial   | Y   | Y    | Dash params packed into shader uniforms                   |
| PointsMaterial       | Y   | P    | WebGPU lacks gl_PointSize; renders at 1px                 |
| SpriteMaterial       | Y   | Y    | Billboard quad with camera-facing transform               |
| ShaderMaterial       | Y   | Y    | WGSL native; GLSL with `-DTHREEPP_WGPU_GLSL_COMPAT=ON`   |
| RawShaderMaterial    | Y   | N    |                                                           |
| ShadowMaterial       | Y   | Y    | Renders shadow attenuation on transparent surface         |


## Texture Maps


| Feature                | GL  | Wgpu | Notes                |
| ---------------------- | --- | ---- | -------------------- |
| Diffuse/color map      | Y   | Y    |                      |
| Normal map             | Y   | Y    | With normal scale    |
| Emissive map           | Y   | Y    |                      |
| Roughness map          | Y   | Y    |                      |
| Metalness map          | Y   | Y    |                      |
| AO map                 | Y   | Y    | With intensity       |
| Alpha map              | Y   | Y    |                      |
| Specular map           | Y   | Y    |                      |
| Light map              | Y   | Y    |                      |
| Bump map               | Y   | Y    | With bump scale      |
| Environment map (cube) | Y   | Y    | With envMapIntensity |
| Displacement map       | Y   | Y    | With scale           |
| Gradient map           | Y   | Y    | For toon shading     |
| Matcap texture         | Y   | N    | MeshMatcapMaterial not fully supported |


## Lighting


| Feature                | GL  | Wgpu | Notes                                          |
| ---------------------- | --- | ---- | ---------------------------------------------- |
| AmbientLight           | Y   | Y    |                                                |
| DirectionalLight       | Y   | Y    |                                                |
| PointLight             | Y   | Y    |                                                |
| SpotLight              | Y   | Y    |                                                |
| HemisphereLight        | Y   | Y    |                                                |
| Max directional lights | -   | 8    | Default; configurable via `setMaxLights()`     |
| Max point lights       | -   | 8    | Default; configurable via `setMaxLights()`     |
| Max spot lights        | -   | 8    | Default; configurable via `setMaxLights()`     |
| Max hemisphere lights  | -   | 8    | Default; configurable via `setMaxLights()`     |


## Shadows


| Feature                  | GL  | Wgpu | Notes                                              |
| ------------------------ | --- | ---- | -------------------------------------------------- |
| Basic shadow map         | Y   | Y    |                                                    |
| PCF filtering            | Y   | Y    | 3x3 kernel with comparison sampler                 |
| Per-light shadow         | Y   | Y    | Shadow applied per directional/spot light           |
| Multi-light shadows      | Y   | Y    | Shadow-casting lights sorted first in light buffer  |
| VSM (variance)           | Y   | N    |                                                    |
| Shadow bias              | Y   | Y    |                                                    |
| Normal bias              | Y   | Y    |                                                    |
| Shadow map size          | -   | 1024 | Default; configurable via `setShadowConfig()`      |
| Max shadow lights        | -   | 4    | Default; configurable via `setShadowConfig()`      |
| Max shadow point lights  | -   | 2    | Default; configurable via `setShadowConfig()`      |


## Geometry & Objects


| Feature         | GL  | Wgpu | Notes                                          |
| --------------- | --- | ---- | ---------------------------------------------- |
| Mesh            | Y   | Y    |                                                |
| InstancedMesh   | Y   | Y    | With per-instance color                        |
| SkinnedMesh     | Y   | Y    | 4 bones per vertex                             |
| Morph targets   | Y   | Y    | Per-target influences                          |
| Line            | Y   | Y    |                                                |
| LineSegments    | Y   | Y    |                                                |
| LineLoop        | Y   | Y    |                                                |
| Points          | Y   | P    | Renders at 1px; WebGPU has no variable ptSize  |
| Sprite          | Y   | Y    | CPU billboard matrix; shared quad geometry     |
| Group           | Y   | Y    |                                                |
| LOD             | Y   | Y    |                                                |
| Geometry groups | Y   | P    | Basic support, limited testing                 |
| Wireframe mode  | Y   | Y    |                                                |
| Vertex colors   | Y   | Y    |                                                |


## Rendering Features


| Feature                          | GL  | Wgpu | Notes                                 |
| -------------------------------- | --- | ---- | ------------------------------------- |
| Fog (linear)                     | Y   | Y    |                                       |
| Fog (exponential)                | Y   | Y    |                                       |
| Tone mapping (Linear)            | Y   | Y    |                                       |
| Tone mapping (Reinhard)          | Y   | Y    |                                       |
| Tone mapping (Cineon)            | Y   | Y    |                                       |
| Tone mapping (ACESFilmic)        | Y   | Y    | Full matrix ACES (matches GL); post-process blit per RT |
| sRGB output encoding             | Y   | Y    | Applied explicitly in blit; linear surfaceFormat used  |
| MSAA                             | Y   | P    | Wgpu supports 1x or 4x only          |
| Render targets (RTT)             | Y   | Y    |                                       |
| Pixel readback                   | Y   | Y    | BGRA8 → RGB8 conversion              |
| Face culling                     | Y   | Y    | Front/Back/None                       |
| Blending modes                   | Y   | Y    | Normal/Additive/Subtractive/Multiply  |
| Depth write control              | Y   | Y    |                                       |
| Depth test control               | Y   | Y    |                                       |
| Scissor test                     | Y   | Y    |                                       |
| Viewport control                 | Y   | Y    |                                       |
| Clear color/alpha                | Y   | Y    |                                       |
| Color background                 | Y   | Y    |                                       |
| Cube texture skybox              | Y   | N    | GLBackground equivalent missing       |
| Clipping planes                  | Y   | P    | Wgpu: single plane only               |
| Logarithmic depth buffer         | Y   | N    |                                       |
| Frustum culling                  | Y   | Y    |                                       |
| Object sort (opaque/transparent) | Y   | Y    |                                       |
| Headless rendering               | Y   | Y    | `Canvas::Parameters().headless(true)` |


## Post-Processing


| Feature                     | GL  | Wgpu | Notes                                               |
| --------------------------- | --- | ---- | --------------------------------------------------- |
| EffectComposer              | -   | Y    | Scene → shader pass chain → readback                |
| ShaderPass                  | -   | Y    | WGSL full-screen triangle, lazy pipeline creation   |
| Ping-pong render targets    | -   | Y    | Automatic internal texture management               |
| Multi-pass chaining         | -   | Y    | Tested with double-invert round-trip                |
| Identity pass               | -   | Y    | Pixel-accurate (±5 per channel)                     |
| Grayscale effect            | -   | Y    | Luminance-weighted desaturation                     |
| Invert effect               | -   | Y    | Per-channel color inversion                         |
| Brightness effect           | -   | Y    | Scalar multiply with clamp                          |
| Post-process + RenderTarget | -   | Y    | Works with user-supplied render targets             |
| Post-processing via RTT     | Y   | Y    | GL uses manual RTT; Wgpu has EffectComposer API    |


## GPU Compute & Abstractions


| Feature               | GL  | Wgpu | Notes                                              |
| --------------------- | --- | ---- | -------------------------------------------------- |
| WgpuComputePipeline   | -   | Y    | WGSL compute shader dispatch                       |
| WgpuTexture           | -   | Y    | GPU texture (RGBA32F, RG32F, RGBA8); 2D and cube   |
| WgpuBuffer            | -   | Y    | GPU uniform/storage buffer with CPU upload          |
| Storage texture write | -   | Y    | `setStorageTexture()` for compute output            |
| Texture sampling      | -   | Y    | `setTexture()` for compute input                   |
| Uniform buffers       | -   | Y    | `setUniformBuffer()` for compute params             |
| Workgroup dispatch    | -   | Y    | `dispatch(x, y, z)` with auto bind group creation  |


## Custom Shaders


| Feature                        | GL  | Wgpu | Notes                                                    |
| ------------------------------ | --- | ---- | -------------------------------------------------------- |
| Native WGSL ShaderMaterial     | -   | Y    | Direct WGSL vertex/fragment; no translation overhead     |
| GLSL compat ShaderMaterial     | Y   | P    | Requires `-DTHREEPP_WGPU_GLSL_COMPAT=ON`; GLSL→SPIR-V   |
| `#include <chunk>` expansion   | Y   | P    | GLSL compat only; expands three.js ShaderChunks          |
| Built-in uniform injection     | Y   | P    | GLSL compat: modelMatrix, viewMatrix, projectionMatrix   |
| Custom uniform extraction      | Y   | Y    | Non-texture uniforms packed into std140 UBO              |
| Sampler2D/TextureCube binding  | Y   | P    | GLSL compat: auto binding assignment from binding 3      |
| Shader caching (feature key)   | -   | Y    | 64-bit feature bitmask → pipeline cache                  |
| Shader caching (material ID)   | -   | Y    | Custom shader hash → separate cache                      |


## Integration


| Feature                  | GL  | Wgpu | Notes                                            |
| ------------------------ | --- | ---- | ------------------------------------------------ |
| ImGui overlay            | Y   | Y    | Via `imgui_impl_wgpu` + overlay callback         |
| Overlay callback API     | -   | Y    | `setOverlayCallback()` injects into render pass  |
| Native device access     | -   | Y    | `nativeDevice()` → `WGPUDevice`                  |
| Native queue access      | -   | Y    | `nativeQueue()` → `WGPUQueue`                    |
| Native instance access   | -   | Y    | `nativeInstance()` → `WGPUInstance`               |
| Native RT texture access | -   | Y    | `nativeRenderTargetTexture()` → `WGPUTexture`    |
| Surface format query     | -   | Y    | `nativeSurfaceFormat()` for backend init          |
| Render info statistics   | Y   | Y    | Frame, calls, triangles, lines, points, memory   |


## Architecture


| Feature                    | GL  | Wgpu | Notes                                          |
| -------------------------- | --- | ---- | ---------------------------------------------- |
| Material dispose listeners | Y   | Y    | Evicts stale pipeline cache                    |
| Per-frame buffer pooling   | N   | Y    | Avoids per-draw alloc/free                     |
| Nested render support      | Y   | N    | GL has renderListStack/renderStateStack        |
| Dynamic light counts       | Y   | Y    | `setMaxLights()` reconfigures at runtime       |
| Dynamic shadow config      | -   | Y    | `setShadowConfig()` for map size and limits    |
| Pipeline invalidation      | -   | Y    | On MSAA change, material dispose, light reconf |
| Geometry version tracking  | Y   | Y    | Incremental buffer updates                     |
| Texture version tracking   | Y   | Y    | Lazy upload on change                          |


## Platform Support


| Platform                | GL | Wgpu | Notes                                          |
| ----------------------- |----| ---- | ---------------------------------------------- |
| Windows                 | Y  | Y    | Vulkan backend via wgpu-native                 |
| macOS                   | Y  | Y    | Metal backend via wgpu-native                  |
| Linux                   | Y  | Y    | Vulkan backend via wgpu-native                 |
| Emscripten (WebAssembly)| Y  | Y    | Direct browser WebGPU; no wgpu-native needed   |


## Shader Feature Flags (Dynamic WGSL Generation)

The Wgpu shader generator uses a 64-bit feature bitmask to select shader variants.
42 flags control material model, topology, maps, effects, and rendering modes:

| Category        | Flags                                                              |
| --------------- | ------------------------------------------------------------------ |
| Material model  | Texture, Lighting, Specular, PBR                                  |
| Topology        | Triangle, LineList, LineStrip, PointList                           |
| Texture maps    | NormalMap, EmissiveMap, RoughnessMap, MetalnessMap, AOMap, AlphaMap, SpecularMap, LightMap, BumpMap, GradientMap, EnvMap, DisplacementMap |
| Instance/morph  | Instanced, InstanceColor, MorphTargets, Skinning                  |
| Visualisation   | NormalVis, DepthVis                                               |
| Effects         | FogLinear, FogExp2, Shadow, ShadowMat, LineDashed                 |
| Rendering       | SRGBOutput, VertexColors, DepthWriteOff, DepthTestOff, Wireframe  |
| Config          | CullMode (2 bits), BlendMode (3 bits), ToneMapping (3 bits)       |


## Test Coverage

- **105 Wgpu-specific test cases** in `CrossRenderer_wgpu_test` (324+ assertions)
- **42 cross-renderer parity tests** in `CrossRenderer_parity_test` (182 assertions) validating GL↔Wgpu consistency
- **Performance tests** in `WgpuRenderer_perf_test`
- **23 total CTest targets** across the project
