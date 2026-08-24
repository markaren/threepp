// Primary-ray geometry for the camera in the CameraUbo at set 0, binding 0.
// Include AFTER that block is declared, and declare the block WITH its trailing
// `vec4 camAux` member — these helpers read it.
//
// camAux.x is 1 for an ORTHOGRAPHIC camera. What differs there is not the
// projection matrix alone: a perspective camera has ONE ray origin (the eye)
// and a direction per pixel, a parallel one has ONE direction (the camera's
// world forward, camAux.yzw) and an ORIGIN per pixel. Every view vector, fog
// leg, sky ray and volumetric march start in the shading passes is built from
// those two, so getting them right here is what makes the two projections shade
// the same rather than one of them falling apart into a single wrong ray.
//
// The perspective branches are exactly the expressions the call sites used
// before the split, so nothing about a perspective frame changes. camAux.x is
// uniform across the dispatch, so the branch costs nothing.

#ifndef CAMERA_RAY_GLSL
#define CAMERA_RAY_GLSL

// This pixel's own point on the camera's NEAR PLANE, in world space. Reverse-Z,
// so NDC z = 1 IS the near plane, and unprojecting there is the near-plane point
// under BOTH projections — for a parallel camera it is the whole ray origin, for
// a perspective one it is where that pixel's ray pierces the glass: a patch
// spanning ±sqrt(halfW² + halfH² + near²) ≈ a few centimetres around the eye.
//
// That patch is the camera's physical PORT, which is why this exists separately
// from camRayOrigin: a lens straddling a water surface is half in each medium,
// and only a per-pixel point can say which side a given pixel looks out of.
// Use it for MEDIUM decisions only — ray origins, fog legs and view vectors all
// stay on camRayOrigin, or a perspective frame would fan out from a disc.
vec3 camNearPoint(vec2 ndc) {
    const vec4 nh = cam.projInverse * vec4(ndc, 1.0, 1.0);
    return (cam.viewInverse * vec4(nh.xyz / nh.w, 1.0)).xyz;
}

// World-space origin of the primary ray through `ndc`.
// Perspective: the eye. Ortho: this pixel's own point on the near plane.
vec3 camRayOrigin(vec2 ndc) {
    if (cam.camAux.x > 0.5) return camNearPoint(ndc);
    return cam.viewInverse[3].xyz;
}

// World-space direction of the primary ray through `ndc`, normalized.
vec3 camRayDir(vec2 ndc) {
    if (cam.camAux.x > 0.5) return cam.camAux.yzw;
    const vec4 tVS = cam.projInverse * vec4(ndc, 1.0, 1.0);
    return normalize((cam.viewInverse * vec4(normalize(tVS.xyz / tVS.w), 0.0)).xyz);
}

// A VIEW-space point on the ray through `ndc` whose view distance (-z, GL-style
// forward) is `d`. Perspective: scale the unprojected direction. Ortho: the
// lateral offset is the same at every depth, so only z moves. The unproject
// depth is arbitrary in both cases — only the ray matters.
vec3 camViewPointAtDist(vec2 ndc, float d) {
    const vec4 v = cam.projInverse * vec4(ndc, 0.5, 1.0);
    const vec3 p = v.xyz / v.w;
    if (cam.camAux.x > 0.5) return vec3(p.xy, -d);
    return p * (d / max(-p.z, 1e-6));
}

#endif// CAMERA_RAY_GLSL
