// The scene-fog block (GpuFogUbo, VulkanGpuLayouts.hpp) as the froxel passes,
// the lit particles and the splats read it. The host struct grows only at its
// tail, so a reader may stop early, and these readers stop at two places:
//   FOG_UBO_HEAD_ONLY  the homogeneous medium, the waterline and world up
//                      (froxel_inject.comp, froxel_integrate.comp)
//   default            the head plus the height-fog mirror and the murk
//                      density and colour (particle_light.comp,
//                      splat_common.glsl)
// The deferred shade (deferred_shade_00_common.glsl) and the deferred filters
// (deferred_filter_common.glsl) declare their own, longer prefixes; lidar.rgen's
// FogUbo is a different block (std140, binding 5).
//
// Binding 33 of the deferred set; splat_common.glsl defines FOG_UBO_BINDING 18
// before the include.

#ifndef FOG_UBO_GLSL
#define FOG_UBO_GLSL

#ifndef FOG_UBO_BINDING
#define FOG_UBO_BINDING 33
#endif

layout(set = 0, binding = FOG_UBO_BINDING, scalar) uniform FogUbo {
    vec3  sigmaT;        // per-channel extinction (1/world unit)
    float enabled;       // 1.0 = fog active
    vec3  color;         // inscatter tint = single-scattering albedo
    float anisotropy;    // HG g
    float waterSurfaceY; // world-Y of the water surface; 1e30 = unbounded
    vec3  worldUp;       // world up axis (= camera.up)
#ifndef FOG_UBO_HEAD_ONLY
    float hfDensity;     // air-medium σ_t at baseY, mirrored from the CloudUbo
    float hfBaseY;       // air-medium base world Y
    float hfFalloff;     // air-medium exponential height scale (m)
    float murkDensity;   // underwater murk σ_t (1/m; 0 = off)
    vec3  murkColor;     // murk tint
#endif
} fog;

#endif// FOG_UBO_GLSL
