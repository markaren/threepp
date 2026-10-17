// The four analytic light records of GpuLightsUbo (VulkanGpuLayouts.hpp),
// scalar layout. One declaration for every shader that reads the lights
// block (lights_ubo.glsl) and for splat_common.glsl's shortened view of it.

#ifndef LIGHT_STRUCTS_GLSL
#define LIGHT_STRUCTS_GLSL

struct DirLight   { vec3 direction; vec3 color; };
// shadowSlot: the light's layer in the shadow-map arrays (shadow_maps.glsl),
// -1 = none; GpuPointLight / GpuSpotLight in VulkanGpuLayouts.hpp.
struct PointLight { vec3 position; float range; vec3 color; float decay; float radius; float shadowSlot; };
struct SpotLight  { vec3 position; float range; vec3 color; float decay;
                    vec3 direction; float cosAngleOuter; float cosAngleInner; float radius;
                    float shadowSlot; };
struct RectLight  { vec3 position; vec3 halfU; vec3 halfV; vec3 normal; vec3 color; };

#endif// LIGHT_STRUCTS_GLSL
