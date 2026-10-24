#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable
// The material index derives from a flat per-instance input, not a
// dynamically uniform value (gbuffer.frag explains why that needs this).
#extension GL_EXT_nonuniform_qualifier : require

#include "vulkan_shared.h"

// Shadow-map caster: depth only. An alpha-cutout material (alphaCutoff > 0)
// casts where its albedo alpha reaches the cutoff and nowhere else, the
// rule shadowOccludes applies to a shadow ray (deferred_shade_10). Blend,
// transmissive and strongly emissive materials never reach this pass
// (buildShadowDrawData leaves them out, as the rays pass them by).

layout(set = 0, binding = 2, scalar) readonly buffer GbufMatBuf {
    MaterialDesc gbufMats[];
};
layout(set = 0, binding = 3) uniform sampler2D gbufAlbedoMaps[kMaxMaterialTextures];
// The per-entry GeometryDesc array as words (binding 5 of the raster set):
// gbufMats is the material table, one record per material, and an entry's
// record index sits in its GeometryDesc.
layout(set = 0, binding = 5, scalar) readonly buffer GbufGeomWordBuf {
    uint gbufGeomWords[];
};

layout(location = 0) in vec2 vUv;
layout(location = 1) flat in uint vInstanceIdx;

void main() {
    const MaterialDesc m = gbufMats[gbufGeomWords[vInstanceIdx * kGeomDescWords + kGeomDescFlagsWord] >> kGeomFlagMaterialShift];
    if (m.alphaCutoff > 0.0 && m.albedoTexIndex >= 0) {
        const int   ai = clamp(m.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
        const float a  = texture(gbufAlbedoMaps[nonuniformEXT(ai)], (m.uvTransform * vec3(vUv, 1.0)).xy).a;
        if (a < m.alphaCutoff) discard;
    }
}
