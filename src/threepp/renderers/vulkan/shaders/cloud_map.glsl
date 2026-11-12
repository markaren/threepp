// The baked cloud maps (cloud_shadow.comp, binding 65), one 512x1024 R8 image:
//   rows    0-511: NEAR, the sun transmittance of the ray that leaves the deck's base at each texel, over a
//                  16 km square centred on the camera (31 m texels);
//   rows 512-1023: FAR, the vertical transmittance of each deck column, exp(-tau_v), over a 160 km square
//                  (312 m texels), from the eroded density's expectation (cloudDensityMean).
// Every reader of the deck that is not the visible sky's own march reads these, so the shadows, the glitter,
// the water's reflected deck and the march's sun light see one field.
//
// REQUIRES in scope: clouds (binding 58; it brings ocean_extended.glsl), lights, cam, cloudShadowTex (binding 65).
// With kOceanExt off cloudShadowSample is upstream's (cloud_shadow_sample.glsl before the extended ocean).

#include "ocean_extended.glsl"// kOceanExt (cloud_march.comp declares its own CloudUbo)
const float kCloudNearHalfExt = 8000.0;  // KEEP IN SYNC with cloud_shadow.comp
const float kCloudFarHalfExt  = 80000.0;

// uv of a world xz in one half of the image; false off that map. Clamped half a texel inside its half so the
// bilinear tap never reads across the seam into the other map.
bool cloudMapUv(vec2 xz, bool far, out vec2 uv) {
    const float ext = far ? kCloudFarHalfExt : kCloudNearHalfExt;
    vec2 u = (xz - cam.viewInverse[3].xz) / (2.0 * ext) + 0.5;
    if (any(lessThan(u, vec2(0.0))) || any(greaterThan(u, vec2(1.0)))) return false;
    u.y = clamp(u.y, 0.5 / 512.0, 1.0 - 0.5 / 512.0) * 0.5 + (far ? 0.5 : 0.0);
    uv = u;
    return true;
}
// Vertical transmittance of the deck column over xz (the far map); the coverage mean off the map.
float cloudColumnT(vec2 xz) {
    vec2 uv;
    if (!cloudMapUv(xz, true, uv)) return 1.0 - 0.8 * clamp(clouds.coverage, 0.0, 1.0);
    return texture(cloudShadowTex, uv).r;
}
// Transmittance of the deck along a slant segment, from the far map's columns along its ground track:
// optical depth = mean column depth x (path length / deck thickness). ro is the ray origin, t0/t1 the
// shell's entry and exit along rd.
float cloudSlantT(vec3 ro, vec3 rd, float t0, float t1) {
    const float th = max(clouds.topY - clouds.bottomY, 1.0);
    float tau = 0.0;
    for (int k = 0; k < 3; ++k) {
        const vec2 xz = ro.xz + rd.xz * mix(t0, t1, (float(k) + 0.5) / 3.0);
        tau -= log(max(cloudColumnT(xz), 1e-4));
    }
    return exp(-tau / 3.0 * (t1 - t0) / th);
}
// Sun transmittance at a world point under the deck. The near map is indexed where the point's own sun ray
// crosses the deck's base (before, the shadows sat straight under the clouds); past the near map
// the far map's columns along the sun's ground track take over.
float cloudShadowSample(vec3 worldPos) {
    if (clouds.shadowActive < 0.5) return 1.0;
    if (!kOceanExt) {
        // upstream: the 512² map over an 8 km camera-centred square straight under the point
        const float halfExt = 4000.0;// == kCloudShadowHalfExt
        const vec2  cenXZ   = cam.viewInverse[3].xz;
        const vec2  suv     = (worldPos.xz - cenXZ) / (2.0 * halfExt) + 0.5;
        if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) return 1.0;
        return texture(cloudShadowTex, suv).r;
    }
    if (lights.dirCount == 0u) return 1.0;
    const vec3 L = normalize(lights.dirLights[0].direction);
    if (L.y < 0.05) return 1.0;  // as cloud_shadow.comp
    const float tB = max(clouds.bottomY - worldPos.y, 0.0) / L.y;
    vec2 uv;
    if (cloudMapUv(worldPos.xz + L.xz * tB, false, uv)) return texture(cloudShadowTex, uv).r;
    return cloudSlantT(worldPos, L, tB, max(clouds.topY - worldPos.y, 0.0) / L.y);
}
// Sun transmittance from a point INSIDE the deck over the next `reach` metres toward the sun (the cloud
// march's light term). The near map holds base-to-top along this point's own sun ray; the share of it is
// the share of the column's height that `reach` climbs. The reach stands in for the multiple scattering a
// single-scatter march lacks: over the whole column to the top a thick deck at a low sun goes as dark as
// the sky's ambient and reads as clear sky, so the march keeps the old taps' 440 m.
float cloudSunTFrom(vec3 p, vec3 L, float reach) {
    const float th = max(clouds.topY - clouds.bottomY, 1.0);
    const float frac = clamp(min(clouds.topY - p.y, reach * max(L.y, 0.05)) / th, 0.0, 1.0);
    vec2 uv;
    if (L.y > 0.05 && cloudMapUv(p.xz - L.xz * (p.y - clouds.bottomY) / L.y, false, uv))
        return pow(max(texture(cloudShadowTex, uv).r, 1e-4), frac);
    return pow(max(cloudColumnT(p.xz), 1e-4), frac / max(L.y, 0.05));
}
