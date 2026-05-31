// Shared DDGI (dynamic diffuse global illumination) math — octahedral
// direction<->uv mapping and probe-grid addressing. Pure functions only; no
// bindings, no globals. Included by ddgi_update.rgen (probe trace) and by the
// sampling path in closest_hit.rchit / the raygen gbuf-shade path once the
// irradiance/visibility atlases are wired (P1+). Keeping the math here means
// the trace side and the sample side address probes identically by
// construction — a layout disagreement between the two is the classic DDGI
// "everything's subtly wrong" bug, so there is a single source of truth.
//
// Reference: Majercik et al. 2019, "Dynamic Diffuse Global Illumination with
// Ray-Traced Irradiance Fields" (the RTXGI formulation).

#ifndef THREEPP_VULKAN_DDGI_COMMON_GLSL
#define THREEPP_VULKAN_DDGI_COMMON_GLSL

// ── Octahedral encoding ─────────────────────────────────────────────────────
// Maps a unit direction to a point on the [-1,1]² octahedron and back. This is
// how each probe packs a full sphere of irradiance/visibility into a square
// tile. Cusick/Engelhardt signed-octahedron form (matches RTXGI).
vec2 ddgiSignNotZero(vec2 v) {
    return vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

// dir (unit) -> oct coords in [-1, 1].
vec2 ddgiOctEncode(vec3 dir) {
    const float l1 = abs(dir.x) + abs(dir.y) + abs(dir.z);
    vec2 oct = dir.xy * (1.0 / max(l1, 1e-8));
    if (dir.z < 0.0) {
        oct = (vec2(1.0) - abs(oct.yx)) * ddgiSignNotZero(oct);
    }
    return oct;
}

// oct coords in [-1, 1] -> unit direction.
vec3 ddgiOctDecode(vec2 oct) {
    vec3 v = vec3(oct.xy, 1.0 - abs(oct.x) - abs(oct.y));
    if (v.z < 0.0) {
        v.xy = (vec2(1.0) - abs(v.yx)) * ddgiSignNotZero(v.xy);
    }
    return normalize(v);
}

// Convenience wrappers in [0, 1] tile space (what the atlas uv math wants).
vec2 ddgiDirToOctUv(vec3 dir) { return ddgiOctEncode(dir) * 0.5 + 0.5; }
vec3 ddgiOctUvToDir(vec2 uv)  { return ddgiOctDecode(uv * 2.0 - 1.0); }

// ── Probe-grid addressing ───────────────────────────────────────────────────
// Probes form a regular 3-D lattice. `counts` = probes per axis, `origin` =
// world position of probe (0,0,0), `spacing` = world distance between adjacent
// probes per axis. Linear index layout is x-fastest then y then z, matching the
// host's atlas tile-row layout.
int ddgiProbeCoordToIndex(ivec3 c, ivec3 counts) {
    return c.x + c.y * counts.x + c.z * counts.x * counts.y;
}

ivec3 ddgiProbeIndexToCoord(int idx, ivec3 counts) {
    const int x = idx % counts.x;
    const int y = (idx / counts.x) % counts.y;
    const int z = idx / (counts.x * counts.y);
    return ivec3(x, y, z);
}

vec3 ddgiProbeWorldPosition(ivec3 c, vec3 origin, vec3 spacing) {
    return origin + vec3(c) * spacing;
}

// Lowest-corner probe of the cell containing `worldPos`, clamped so the 2×2×2
// trilinear neighbourhood stays in range.
ivec3 ddgiBaseProbeCoord(vec3 worldPos, vec3 origin, vec3 spacing, ivec3 counts) {
    const vec3 g = (worldPos - origin) / spacing;
    return clamp(ivec3(floor(g)), ivec3(0), counts - ivec3(2));
}

// Trilinear blend weights within the cell (per-axis fraction in [0,1]).
vec3 ddgiTrilinearAlpha(vec3 worldPos, vec3 origin, vec3 spacing, ivec3 baseCoord) {
    const vec3 baseWp = origin + vec3(baseCoord) * spacing;
    return clamp((worldPos - baseWp) / spacing, vec3(0.0), vec3(1.0));
}

// ── Atlas addressing ────────────────────────────────────────────────────────
// The atlas packs one octahedral tile per probe in a 2-D grid of tiles, laid
// out `tilesPerRow` tiles wide (probe index increasing left-to-right then
// top-to-bottom). Each tile is `interiorRes` texels plus a `border`-texel
// gutter on all sides (the gutter is written by the blend pass with mirrored
// edge texels so hardware bilinear sampling is seam-correct).
//
// Returns the normalized atlas uv to sample for direction-tile-uv `octUv01`
// of probe `probeIndex`. `atlasSize` is the full atlas resolution in texels.
vec2 ddgiProbeAtlasUv(int probeIndex, vec2 octUv01,
                      int interiorRes, int border, int tilesPerRow,
                      vec2 atlasSize) {
    const int tileSide = interiorRes + 2 * border;
    const ivec2 tile   = ivec2(probeIndex % tilesPerRow, probeIndex / tilesPerRow);
    // Interior texel-space position within the tile, offset past the border.
    const vec2 interior = vec2(border) + clamp(octUv01, 0.0, 1.0) * float(interiorRes);
    const vec2 texel    = vec2(tile * tileSide) + interior;
    // Sample at the texel's centre-relative position; the +0.0 here (vs +0.5)
    // is intentional — `interior` already lands inside the gutter-padded tile,
    // and the blend pass duplicates edge texels into the gutter so a bilinear
    // tap straddling the seam reads the wrapped neighbour correctly.
    return texel / atlasSize;
}

// ── Probe ray directions ────────────────────────────────────────────────────
// Spherical-Fibonacci point set — near-uniform sphere coverage for the probe
// update rays. Both ddgi_update.rgen (casting rays) and ddgi_blend.comp
// (weighting each octahedral texel against the rays that contributed to it)
// derive directions from this same function plus a shared per-frame rotation
// (passed in the DDGI uniform), so the two sides agree by construction — a
// mismatch here is a silent, hard-to-spot energy error.
vec3 ddgiSphericalFibonacci(float i, float n) {
    const float kDdgiTwoPi = 6.28318530717958;
    const float phi  = kDdgiTwoPi * fract(i * 0.61803398875); // golden-ratio increment
    const float cosT = 1.0 - (2.0 * i + 1.0) / n;
    const float sinT = sqrt(clamp(1.0 - cosT * cosT, 0.0, 1.0));
    return vec3(cos(phi) * sinT, sin(phi) * sinT, cosT);
}

#endif // THREEPP_VULKAN_DDGI_COMMON_GLSL
