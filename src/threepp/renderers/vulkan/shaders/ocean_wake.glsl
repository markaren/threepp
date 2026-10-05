// The ocean's WAKE FIELD as its GLSL consumers read it (wake_field.comp writes
// it, the deferred water shade samples it). See DisplacedMesh::WakeField for
// what it is; vulkan::WakeFieldPipeline mirrors the two structs field for field.
//
// The table lives in one device buffer per ocean whose address is stable: the
// water shade reaches it through the ocean's GeometryDesc::foamAddress (1 = an
// ocean with no wake field, anything larger = this table). It is rewritten in
// the frame's own command stream (vkCmdUpdateBuffer), so a frame still in
// flight reads the origins its own images were written with.

#ifndef THREEPP_OCEAN_WAKE_GLSL
#define THREEPP_OCEAN_WAKE_GLSL

#define OCEAN_WAKE_MAX_PATCHES 8

// One patch: a square window of water, world-anchored, `res` texels a side.
// 32 bytes.
struct WakePatch {
    float originX;      // world min corner, snapped to whole texels
    float originZ;
    float size;         // side (m)
    float texel;        // size / res
    float prevOriginX;  // the origin its images were last written with
    float prevOriginZ;
    float live;         // 0 = off; 1 = on, its images empty until this step; 2 = on, with last step's water
    float eddy;         // size of its large turbulent eddies (m)
};

// What makes a mark. 48 bytes (= DisplacedMesh::WakeSource).
struct WakeSource {
    vec2  p0;           // where the producer was at the last step (world xz)
    vec2  p1;           // where it is now
    vec2  vel;          // water velocity it imparts (m/s, over the ground)
    float radius;
    float foam;
    float aeration;
    float turbulence;   // m/s
    float lane;
    float _pad;
};

// 48-byte header, then the patches, then the sources of this step.
layout(buffer_reference, scalar) readonly buffer WakeTable {
    uint  patchCount;
    uint  res;
    float timeSec;
    float dt;
    float foamLife;
    float foamThinning;
    float aerationLife;
    float turbulenceLife;
    float laneLife;
    float velocityLife;
    float eddyTexels;
    float spread;
    WakePatch  patches[OCEAN_WAKE_MAX_PATCHES];
    WakeSource sources[];
};

#ifdef OCEAN_WAKE_SAMPLER
// Value noise over (x, y, time): (value in -1..1, d/dx, d/dy), quintic fade.
// The surface's boil where the water is turbulent takes its slope from this.
float oceanWakeHash(ivec3 p) {
    uint h = uint(p.x) * 0x8da6b343u ^ uint(p.y) * 0xd8163841u ^ uint(p.z) * 0xcb1ab31fu;
    h ^= h >> 15; h *= 0x2c1b3c6du;
    h ^= h >> 12; h *= 0x297a2d39u;
    h ^= h >> 15;
    return float(h) * (1.0 / 4294967296.0);
}
vec3 oceanWakeNoised(vec3 p) {
    const ivec3 i  = ivec3(floor(p));
    const vec3  f  = p - vec3(i);
    const vec3  u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    const vec2  du = 30.0 * f.xy * f.xy * (f.xy * (f.xy - 2.0) + 1.0);
    const float A = mix(oceanWakeHash(i),                  oceanWakeHash(i + ivec3(0, 0, 1)), u.z);
    const float B = mix(oceanWakeHash(i + ivec3(1, 0, 0)), oceanWakeHash(i + ivec3(1, 0, 1)), u.z);
    const float C = mix(oceanWakeHash(i + ivec3(0, 1, 0)), oceanWakeHash(i + ivec3(0, 1, 1)), u.z);
    const float D = mix(oceanWakeHash(i + ivec3(1, 1, 0)), oceanWakeHash(i + ivec3(1, 1, 1)), u.z);
    const float k1 = B - A, k2 = C - A, k4 = A - B - C + D;
    const float v  = A + k1 * u.x + k2 * u.y + k4 * u.x * u.y;
    return vec3(2.0 * v - 1.0, 2.0 * du * vec2(k1 + k4 * u.y, k2 + k4 * u.x));
}

// (foam, aeration, turbulence, lane) at a world XZ: the largest of the patches
// that cover it, each faded out over its outer margin so a patch never ends on
// a line. `footprint` is the pixel's size on the water (m): above a texel the
// read is a box of taps over the footprint, at or under it a B-spline (four
// bilinear taps), whose derivative is continuous where bilinear's is not.
vec4 oceanWakeBicubic(vec2 uv, float layer, float res) {
    const vec2 st = uv * res - 0.5;
    const vec2 i  = floor(st);
    const vec2 f  = st - i;
    const vec2 f2 = f * f;
    const vec2 f3 = f2 * f;
    const vec2 w0 = (-f3 + 3.0 * f2 - 3.0 * f + 1.0) * (1.0 / 6.0);
    const vec2 w1 = (3.0 * f3 - 6.0 * f2 + 4.0) * (1.0 / 6.0);
    const vec2 w2 = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) * (1.0 / 6.0);
    const vec2 w3 = f3 * (1.0 / 6.0);
    const vec2 g0 = w0 + w1;
    const vec2 g1 = w2 + w3;
    const vec2 c0 = (i - 0.5 + w1 / g0) / res;
    const vec2 c1 = (i + 1.5 + w3 / g1) / res;
    return g0.y * (g0.x * textureLod(OCEAN_WAKE_SAMPLER, vec3(c0.x, c0.y, layer), 0.0) +
                   g1.x * textureLod(OCEAN_WAKE_SAMPLER, vec3(c1.x, c0.y, layer), 0.0)) +
           g1.y * (g0.x * textureLod(OCEAN_WAKE_SAMPLER, vec3(c0.x, c1.y, layer), 0.0) +
                   g1.x * textureLod(OCEAN_WAKE_SAMPLER, vec3(c1.x, c1.y, layer), 0.0));
}

vec4 oceanWakeSample(uint64_t tableAddr, vec2 xz, float footprint) {
    WakeTable tb = WakeTable(tableAddr);
    vec4 acc = vec4(0.0);
    const uint n = min(tb.patchCount, uint(OCEAN_WAKE_MAX_PATCHES));
    const float res = float(tb.res);
    for (uint k = 0u; k < n; ++k) {
        const WakePatch p = tb.patches[k];
        if (p.live < 0.5) continue;
        const vec2 uv = (xz - vec2(p.originX, p.originZ)) / p.size;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) continue;
        const vec2  e    = min(uv, 1.0 - uv);
        const float edge = smoothstep(0.0, 0.06, min(e.x, e.y));
        vec4 s;
        if (footprint <= p.texel) {
            s = oceanWakeBicubic(uv, float(k), res);
        } else {
            // n x n bilinear taps over the footprint (a tap spans two texels)
            const int   m    = int(clamp(ceil(0.5 * footprint / p.texel), 1.0, 4.0));
            const float step = footprint / (float(m) * p.size);
            s = vec4(0.0);
            for (int a = 0; a < m; ++a)
                for (int b = 0; b < m; ++b)
                    s += textureLod(OCEAN_WAKE_SAMPLER,
                                    vec3(uv + (vec2(float(a), float(b)) - 0.5 * float(m - 1)) * step, float(k)), 0.0);
            s /= float(m * m);
        }
        acc = max(acc, s * edge);
    }
    return acc;
}
#endif// OCEAN_WAKE_SAMPLER

#endif// THREEPP_OCEAN_WAKE_GLSL
