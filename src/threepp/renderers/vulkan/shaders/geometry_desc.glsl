// GeometryDesc as the shaders see it: one record per BLAS entry, scalar
// layout, mirroring the host struct in VulkanGeometryState.hpp. This is the
// ONE declaration (until 2026-10-10 four shaders each mirrored it by hand).
// A shader that only needs the record includes this; one that fetches
// vertices through it includes geometry_fetch.glsl, which includes this.

#ifndef GEOMETRY_DESC_GLSL
#define GEOMETRY_DESC_GLSL

struct GeometryDesc {
    uint64_t vertexAddress;
    uint64_t normalAddress;
    uint64_t indexAddress;
    uint64_t uvAddress;
    uint64_t foamAddress;
    uint64_t prevVertexAddress;
    vec3     instanceColor;// the entry's InstancedMesh colour, (1,1,1) without one: multiplies its
                           // material's albedo (entryMat), which the table record leaves bare
    uint     flags;// the kGeomFlag* bits of vulkan_shared.h, which the shaders that read this word include;
                   // bits 9..31 the entry's material record, flags >> kGeomFlagMaterialShift
};

#endif// GEOMETRY_DESC_GLSL
