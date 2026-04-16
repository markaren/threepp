#ifndef THREEPP_WGPUPATHTRACERTYPES_HPP
#define THREEPP_WGPUPATHTRACERTYPES_HPP

// Private header — shared POD types, GPU uniform layouts and constants used by
// the path tracer translation units. Not part of the public API.

#include "threepp/renderers/wgpu/WgpuTexture.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace threepp::wgpu_pt {

    // -----------------------------------------------------------------------
    // Limits & atlas constants
    // -----------------------------------------------------------------------
    constexpr int MAX_TEX_SLOTS    = 1024;
    constexpr int DEFAULT_TILE_SIZE = 512;
    constexpr int ATLAS_WIDTH      = 8192;  // fixed atlas width; cols = ATLAS_WIDTH / tileSize
    constexpr int TRI_TEX_HEIGHT   = 8;     // rows per tri: pos(3)+nrm(3)+uv0(2)
    constexpr int MAT_TEX_HEIGHT   = 19;
    constexpr int TEX_PAGE_WIDTH   = 8192;

    // Initial placeholder capacities — grown dynamically as scenes demand more.
    constexpr int INIT_TRI_CAP  = 1;
    constexpr int INIT_MAT_CAP  = 1;
    constexpr int INIT_MESH_CAP = 1;

    // objTriBuf is 32 floats (128 bytes) per tri (8 fields × vec4).
    // Max tri count is computed at runtime from the device's maxStorageBufferBindingSize.
    constexpr std::size_t BYTES_PER_TRI = 32 * sizeof(float);  // 128

    // BVH4 packing constants
    constexpr int MAX_LEAF_TRIS    = 8;
    constexpr int BVH4_GPU_U32S    = 28; // 112 bytes per node (f32 AABBs: 6*vec4 + cIdx)
    constexpr int BVH4_REFIT_INTS  = 4;  // per-node refit metadata

    // -----------------------------------------------------------------------
    // Small inline helpers
    // -----------------------------------------------------------------------
    inline int nextPow2(int v) {
        if (v <= 0) return 1;
        v--;
        v |= v >> 1; v |= v >> 2; v |= v >> 4;
        v |= v >> 8; v |= v >> 16;
        return v + 1;
    }

    inline int triTexPages(int triCap) {
        return (triCap + TEX_PAGE_WIDTH - 1) / TEX_PAGE_WIDTH;
    }

    // -----------------------------------------------------------------------
    // CPU-side uniform structs (must match WGSL layouts)
    // -----------------------------------------------------------------------
    struct alignas(16) RtGpuUniforms {
        float camOri[4];
        float camFwd[4];
        float camRgt[4];
        float camUp[4];
        float prevCamOri[4];
        float prevCamFwd[4];
        float prevCamRgt[4];
        float prevCamUp[4];
        float iRes[4];
        float tanHalfFov[4];
        float frameCount[4];
        float triCount[4];
        float mode[4];
        float lightCount[4];
        float lightPos[4][4];
        float lightCol[4][4];
        float lightType[4][4];      // [i][0]=type (0=pt,1=dir,2=spot), [i][1]=cosAngle, [i][2]=cosOuter, [i][3]=distance
        float lightDir[4][4];       // [i] = spotlight direction xyz, [i][3]=decay
        float    spp[4];
        std::uint32_t movedMeshBits[4];  // bit i = mesh i moved; 4 words cover meshes 0–127
        float    envColor[4];       // xyz = color, w = mode (0=none, 1=solid color, 2=equirect tex)
        float    envIntensity[4];   // x = intensity, y = envWidth, z = envHeight, w = totalLumSum (0 = no CDF)
        float    bgColor[4];        // xyz = color, w = mode (0=sky gradient, 1=solid color, 2=equirect bgTex)
        float    params[4];         // x = maxBounces
        float    emissiveInfo[4];   // x = emissive tri count, y = total emissive power, z = fireflyCap
        float    restirParams[4];   // x = enabled, y = M_clamp, z = reserved, w = reserved
        std::uint32_t bvhAux[4];    // [0] = bvhRootIdx (0 = normal root, >0 = overlay combined root)
    };
    static_assert(sizeof(RtGpuUniforms) == 624, "RtGpuUniforms must be 624 bytes");

    struct alignas(16) VtGpuUniforms {
        std::uint32_t triCount, groupsX, splitAt, _p1;
    };
    struct alignas(16) RefitGpuUniforms {
        std::uint32_t leafCount, groupsX, _p[2];
    };
    struct alignas(16) AtrousGpuUniforms {
        std::uint32_t stepSize, mode;  // mode: 0=diffuse, 1=specular
        float         frameCount, _p1;
    };
    struct alignas(16) PreFilterGpuUniforms {
        std::uint32_t stepSize, _p0;
        float         _p1, _p2;
    };
    struct alignas(16) DepthFillUniforms {
        float projView[16];   // NDC-remapped proj * view  (64 bytes)
        float camOri[4];      // camera position            (16 bytes)
        float camFwd[4];      // camera forward             (16 bytes)
        float camRgt[4];      // camera right               (16 bytes)
        float camUp[4];       // camera up                  (16 bytes)
        float iRes[4];        // resolution xy              (16 bytes)
        float tanHalfFov[4];  // x = tanHalfFov             (16 bytes)
        // total = 160 bytes (16-byte aligned, no padding needed)
    };
    struct alignas(16) TaaGpuUniforms {
        float prevCamOri[4], prevCamFwd[4], prevCamRgt[4], prevCamUp[4];
        float curCamOri[4],  curCamFwd[4],  curCamRgt[4],  curCamUp[4];
        float iRes[4];          // [0]=w [1]=h [2]=prevJx [3]=prevJy
        float tanHalfFov[4];
        float frameCount[4];   // [0]=FC [1]=mode(0=diff,1=spec) [2]=curJx [3]=curJy
        std::uint32_t movedMeshBits[4];
    };
    static_assert(sizeof(TaaGpuUniforms) == 192, "TaaGpuUniforms must be 192 bytes");

    struct alignas(16) UpscaleGpuUniforms {
        float prevCamOri[4], prevCamFwd[4], prevCamRgt[4], prevCamUp[4]; // 64 bytes
        float curCamOri[4],  curCamFwd[4],  curCamRgt[4],  curCamUp[4]; // 64 bytes
        float iRes[4];        // [0]=fullW [1]=fullH [2]=pixelScale [3]=0
        float tanHalfFov[4];
        float frameCount[4];  // [0]=FC
    };
    static_assert(sizeof(UpscaleGpuUniforms) == 176, "UpscaleGpuUniforms must be 176 bytes");

    // -----------------------------------------------------------------------
    // BVH structs
    // -----------------------------------------------------------------------
    // Binary BVH node (used during build, then collapsed to BVH4)
    struct BvhNode {
        float minX, minY, minZ;
        int   left;
        float maxX, maxY, maxZ;
        int   right;
        int   parent;
        int   _pad[3];
    };

    // BVH4 node: up to 4 children, stored in SoA layout for SIMD AABB testing.
    // Leaf encoding: childIdx = -(triStart * MAX_LEAF_TRIS + triCount), where triCount is 1-based.
    // Decode: raw = -childIdx; triStart = (raw - 1) / MAX_LEAF_TRIS; triCount = ((raw - 1) % MAX_LEAF_TRIS) + 1
    struct Bvh4Node {
        float childMinX[4], childMinY[4], childMinZ[4];
        float childMaxX[4], childMaxY[4], childMaxZ[4];
        int   childIdx[4];           // >= 0: internal node index, < 0: leaf
        int   childCount;            // 1..4 valid children
        int   numInternalChildren;   // count of children with childIdx >= 0
        int   parent;
    };

    // -----------------------------------------------------------------------
    // PingPong — ping-pong texture pair with read/write pointer aliases
    // -----------------------------------------------------------------------
    struct PingPong {
        WgpuTexture a, b;
        WgpuTexture* read  = &a;
        WgpuTexture* write = &b;
        void swap() { std::swap(read, write); }
        // Must be called after a/b are recreated (e.g. on viewport resize).
        void resetPtrs() { read = &a; write = &b; }
    };

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERTYPES_HPP
