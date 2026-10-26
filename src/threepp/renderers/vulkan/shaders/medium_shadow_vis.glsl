// Binary occlusion toward a light from a point in the air: a froxel sample
// (froxel_inject.comp) or a particle centre (particle_light.comp). The
// opaque-only mask matches the surface shadow rays: blend/glass never blocks a
// shaft, or the sun on a puff. The includer declares topAS (ray_query_stub.glsl
// in the no-ray-query build).
//
// The deferred shade's and probe_update.comp's shadowVis are different
// functions (they alpha-test cutout candidates) and keep their own.

#ifndef MEDIUM_SHADOW_VIS_GLSL
#define MEDIUM_SHADOW_VIS_GLSL

float shadowVis(vec3 origin, vec3 dir, float tMax) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
                          0x01u /*kRayMaskOpaque*/, origin, 1e-3, dir, tMax);
    while (rayQueryProceedEXT(rq)) {}
    return (rayQueryGetIntersectionTypeEXT(rq, true) ==
            gl_RayQueryCommittedIntersectionNoneEXT) ? 1.0 : 0.0;
}

#endif// MEDIUM_SHADOW_VIS_GLSL
