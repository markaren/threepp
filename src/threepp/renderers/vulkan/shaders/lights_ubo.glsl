// The scene's lights block: GpuLightsUbo (VulkanGpuLayouts.hpp) as the
// shaders see it, scalar layout. This is the ONE declaration; a shader
// that reads the lights includes this file. Until 2026-10-10 seven shaders
// carried their own copies, two of them cut short after the rect lights
// and one with the structs spelled out, each right only because it was a
// prefix of the host layout.
//
// The block sits at set 0, binding LIGHTS_UBO_BINDING (default 1); a
// shader that binds it elsewhere defines the macro before the include
// (event_shade.comp: 3). Needs GL_EXT_scalar_block_layout from the
// includer. The host buffer is always the full GpuLightsUbo, so declaring
// the whole block is right even where only `ambient` and the sun are read.

#ifndef LIGHTS_UBO_GLSL
#define LIGHTS_UBO_GLSL

#include "light_structs.glsl"

#ifndef LIGHTS_UBO_BINDING
#define LIGHTS_UBO_BINDING 1
#endif

layout(set = 0, binding = LIGHTS_UBO_BINDING, scalar) uniform LightsUbo {
    vec3       ambient;
    uint       dirCount;
    DirLight   dirLights[8];
    uint       pointCount;
    uint       spotCount;
    uint       rectCount;
    PointLight pointLights[8];
    SpotLight  spotLights[8];
    RectLight  rectLights[4];
    // HemisphereLight zero-mean rows (GpuLightsUbo, VulkanGpuLayouts.hpp):
    // the hemi's angular mean is folded into `ambient` above; the surface
    // term is ambient + rows·N with rows = 0.5*(sky-ground)⊗up — see
    // hemiAmbient() in deferred_shade_10_lighting_utils.glsl. All-zero
    // (no HemisphereLight in the scene) makes it inert.
    vec3       hemiDeltaR;
    vec3       hemiDeltaG;
    vec3       hemiDeltaB;
    // scene.environmentRotation / environmentIntensity and the background
    // knobs (GpuLightsUbo tail, VulkanGpuLayouts.hpp; SceneEnvKnobs.hpp).
    // Rotations are world -> env mat3 columns; envKnobFlags bit 0 / bit 1 =
    // env / background rotation active (identity skipped: defaults exact).
    vec3       envRotC0;
    vec3       envRotC1;
    vec3       envRotC2;
    float      envIntensity;
    vec3       bgRotC0;
    vec3       bgRotC1;
    vec3       bgRotC2;
    float      bgIntensity;
    float      bgBlurriness;
    uint       envKnobFlags;
} lights;

#endif// LIGHTS_UBO_GLSL
