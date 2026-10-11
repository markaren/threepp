// Ray query stand-ins for a device without VK_KHR_ray_query.
//
// Included instead of `#extension GL_EXT_ray_query` and the topAS binding by
// the no-ray-query variants of deferred_shade.comp, froxel_inject.comp and
// particle_light.comp (compiled with -DTHREEPP_NO_RAY_QUERY). Every ray
// misses: proceed is false, the committed type is "none", so each traced
// helper lands in the miss path it already has (environment lookup, sky
// radiance, visibility 1). The hit-side getters return neutral values that a
// miss never reads. The SPIR-V then needs neither the RayQueryKHR capability
// nor an acceleration-structure descriptor.

#ifndef THREEPP_RAY_QUERY_STUB_GLSL
#define THREEPP_RAY_QUERY_STUB_GLSL

#define rayQueryEXT uint
#define topAS 0u

#define rayQueryInitializeEXT(q, as, flags, mask, origin, tMin, dir, tMax) ((q) = 0u)
#define rayQueryProceedEXT(q) false
#define rayQueryConfirmIntersectionEXT(q) (q)
#define rayQueryGetIntersectionTypeEXT(q, committed) 0u
#define rayQueryGetIntersectionTEXT(q, committed) 1e30
#define rayQueryGetIntersectionInstanceCustomIndexEXT(q, committed) 0
#define rayQueryGetIntersectionPrimitiveIndexEXT(q, committed) 0
#define rayQueryGetIntersectionBarycentricsEXT(q, committed) vec2(0.0)
#define rayQueryGetIntersectionWorldToObjectEXT(q, committed) mat4x3(1.0)
#define rayQueryGetIntersectionObjectToWorldEXT(q, committed) mat4x3(1.0)

#define gl_RayFlagsOpaqueEXT 1u
#define gl_RayFlagsTerminateOnFirstHitEXT 4u
#define gl_RayQueryCommittedIntersectionNoneEXT 0u
#define gl_RayQueryCandidateIntersectionTriangleEXT 0u

#endif
