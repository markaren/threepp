// The cloud + height-fog block (GpuCloudUbo, VulkanGpuLayouts.hpp): the far
// cloud layer (VulkanRenderer::setClouds) and the near-field heterogeneous
// height fog (setHeightFog) ride one scalar UBO, since they share wind and
// timeSec. The deferred shade, cloud_shadow.comp, froxel_inject.comp,
// froxel_integrate.comp and particle_light.comp bind it at 58 of the deferred
// set; splat_common.glsl defines CLOUD_UBO_BINDING 19 before the include.
//
// cloud_march.comp, the one reader of the host's trailing `epoch`, declares
// its own copy with that member.

#ifndef CLOUD_UBO_GLSL
#define CLOUD_UBO_GLSL

#ifndef CLOUD_UBO_BINDING
#define CLOUD_UBO_BINDING 58
#endif

layout(set = 0, binding = CLOUD_UBO_BINDING, scalar) uniform CloudUbo {
    float enabled;      // 1.0 = far cloud march active
    float coverage;     // 0 = clear sky .. 1 = overcast
    float density;      // density multiplier
    float bottomY;      // shell base (world Y, m)
    float topY;         // shell top (world Y, m)
    float evolveSpeed;  // shape churn rate
    float timeSec;      // wall-clock seconds (wind scroll + evolution)
    float heteroActive; // 1.0 = heterogeneous near-field froxels (height fog on)
    vec3  wind;         // m/s drift (xz used, y ignored)
    float hfDensity;    // height-fog σ_t at baseY (0 = height fog off)
    float hfBaseY;      // height-fog base world Y
    float hfFalloff;    // height-fog exponential height scale (m)
    float hfNoiseAmount;// 0 = smooth analytic .. 1 = fully noise-modulated
    float shadowActive; // 1.0 = cloud shadow map valid this frame
} clouds;

#endif// CLOUD_UBO_GLSL
