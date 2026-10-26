// Cloud shadow transmittance at a world point (bilinear from the top-down
// cloud shadow map, binding 65). 1.0 (full sun) when clouds/shadows are off or
// the point is outside the 8 km camera-centred square. KEEP the extent + centre
// in sync with cloud_shadow.comp.
//
// The deferred shade (deferred_shade_10_lighting_utils.glsl), froxel_inject.comp
// and particle_light.comp include it after declaring what it reads: `clouds`
// (cloud_ubo.glsl), `cam` (camera_ubo.glsl) and cloudShadowTex.

#ifndef CLOUD_SHADOW_SAMPLE_GLSL
#define CLOUD_SHADOW_SAMPLE_GLSL

float cloudShadowSample(vec3 worldPos) {
    if (clouds.shadowActive < 0.5) return 1.0;
    const float halfExt = 4000.0;// == kCloudShadowHalfExt
    const vec2  cenXZ   = cam.viewInverse[3].xz;
    const vec2  suv     = (worldPos.xz - cenXZ) / (2.0 * halfExt) + 0.5;
    if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) return 1.0;
    return texture(cloudShadowTex, suv).r;
}

#endif// CLOUD_SHADOW_SAMPLE_GLSL
