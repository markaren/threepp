// The site's wave climate (DisplacedMesh::WaveClimate), read by
// water_displace.comp per vertex and by the deferred shade per pixel.
//
// The renderer uploads the map as an RGBA32F texture one row taller than the
// map: row 0 is a header, texel (0, 0) = (originX, originZ, 1 / sizeX,
// 1 / sizeZ), so the map carries its own placement and no push block or
// uniform has to; texel (1, 0).x is the slope variance of the short waves no
// cascade holds that the shade banks into the roughness (the chop moments'
// bankedVar), which the map does not scale; rows 1..H are the map, texel (i, j + 1) the value at the
// centre of cell (i, j). .x, .y, .z scale cascades 0, 1 and 2; .w is the
// caller's own. Read bilinearly at level 0 between texel centres and clamped
// to the edge texels outside the map (so the header row is never blended
// in), as DisplacedMesh::waveClimateGain reads it on the CPU.

#ifndef THREEPP_OCEAN_CLIMATE_GLSL
#define THREEPP_OCEAN_CLIMATE_GLSL

vec4 oceanClimate(sampler2D map, vec2 xz) {
    const ivec2 sz = textureSize(map, 0);
    const vec4  hd = texelFetch(map, ivec2(0, 0), 0);
    const float w  = float(sz.x);
    const float h  = float(sz.y - 1);
    const vec2  f  = (xz - hd.xy) * hd.zw;
    const float tx = clamp(f.x * w, 0.5, w - 0.5);
    const float tz = clamp(f.y * h, 0.5, h - 0.5) + 1.0;
    return textureLod(map, vec2(tx / w, tz / float(sz.y)), 0.0);
}

// The banked short-wave slope variance the header carries.
float oceanClimateBanked(sampler2D map) { return texelFetch(map, ivec2(1, 0), 0).x; }

#endif
