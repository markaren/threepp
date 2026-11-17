// The ocean's patch quadtree (DisplacedMesh::Lod): which patches a frame lays,
// and the per-vertex morph water_displace.comp mirrors (its lodNorm / lodRest).
//
// A level of cell spacing s is drawn out to a distance of kCameraK * s from the
// eye (3D: the eye's height counts), so a cell spans about the same angle
// everywhere; a posed hull is a second focus whose levels reach kFocusK * s
// past its radius, down to its own finest spacing. The normalised distance of
// a point at a level is the least, over the foci, of its distance over that
// level's reach; a node splits while its box comes under 1 for its children.
// A vertex past kMorphStart of its level's reach slides its odd coordinates
// onto the coarser lattice, all the way at 1: a coarser neighbour exists only
// where the finer patch's border lies past 1, and (for these reaches) a patch
// never morphs at its border with a finer one, so levels meet without cracks.
// The lattice is fixed in the mesh's frame: the root is snapped to whole root
// patches, so every patch has the same place whichever frame lays it.

#ifndef THREEPP_VULKAN_WATER_OCEANLOD_HPP
#define THREEPP_VULKAN_WATER_OCEANLOD_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace threepp::water::lod {

    inline constexpr uint32_t kCells = 16, kSide = kCells + 1u, kPatchVertices = kSide * kSide;
    inline constexpr float kCameraK = 128.f;// a level of spacing s reaches kCameraK * s from the eye
    inline constexpr float kFocusK = 64.f;  // ... and kFocusK * s past a hull's radius
    inline constexpr float kMorphStart = 0.9f;
    inline constexpr float kHorizonSpacing = 16384.f;// the open sea's root cell: a 2^18 m root patch
    inline constexpr uint32_t kMaxFoci = 8;
    inline constexpr uint32_t kRtStride = 2;// the ray-tracing structure keeps every kRtStride-th line of a patch

    struct Focus {
        float x = 0.f, z = 0.f;
        float radius = 0.f;    // m, held at the focus's finest spacing
        float minSpacing = 0.f;// the finest spacing it asks for
    };

    struct View {
        float eyeX = 0.f, eyeZ = 0.f, height = 0.f;// the eye in the mesh's frame, height over the rest plane
        float halfX = 0.f, halfZ = 0.f;              // the water's half extents; 0 = none
        float finest = 0.25f;                        // m
        const Focus* foci = nullptr;
        uint32_t focusCount = 0;
    };

    struct Patch {
        float x = 0.f, z = 0.f;// the patch's corner (least x, least z)
        float spacing = 0.f;
    };

    // The root: its corner and cell spacing.
    inline void root(const View& v, float& x0, float& z0, float& spacing) {
        if (v.halfX > 0.f && v.halfZ > 0.f) {
            const float half = std::max(v.halfX, v.halfZ);
            spacing = std::max(v.finest, std::exp2(std::ceil(std::log2(2.f * half / float(kCells)))));
            x0 = z0 = -0.5f * float(kCells) * spacing;
        } else {
            spacing = kHorizonSpacing;
            const float width = float(kCells) * spacing;
            x0 = std::floor(v.eyeX / kHorizonSpacing) * kHorizonSpacing - 0.5f * width;
            z0 = std::floor(v.eyeZ / kHorizonSpacing) * kHorizonSpacing - 0.5f * width;
        }
    }

    // The normalised distance of the box [x, x + w] x [z, z + w] at a level of
    // spacing s (w = 0: a point).
    inline float norm(const View& v, float x, float z, float w, float s) {
        const float cx = std::clamp(v.eyeX, x, x + w) - v.eyeX;
        const float cz = std::clamp(v.eyeZ, z, z + w) - v.eyeZ;
        float n = std::sqrt(cx * cx + cz * cz + v.height * v.height) / (kCameraK * s);
        for (uint32_t f = 0; f < v.focusCount; ++f) {
            const Focus& fo = v.foci[f];
            if (s < fo.minSpacing * 0.999f) continue;
            const float fx = std::clamp(fo.x, x, x + w) - fo.x;
            const float fz = std::clamp(fo.z, z, z + w) - fo.z;
            n = std::min(n, std::max(std::sqrt(fx * fx + fz * fz) - fo.radius, 0.f) / (kFocusK * s));
        }
        return n;
    }

    // The patches to lay: the leaves of the quadtree within the water. When
    // they outnumber `slots`, the finest level goes until they fit.
    inline std::vector<Patch> select(View v, uint32_t slots) {
        std::vector<Patch> out;
        float x0, z0, s0;
        root(v, x0, z0, s0);
        const bool bounded = v.halfX > 0.f && v.halfZ > 0.f;
        auto visit = [&](auto&& self, float x, float z, float s) -> void {
            const float w = float(kCells) * s;
            if (bounded && (x > v.halfX || x + w < -v.halfX || z > v.halfZ || z + w < -v.halfZ)) return;
            const float c = 0.5f * s;
            if (c >= v.finest * 0.999f && norm(v, x, z, w, c) < 1.f) {
                const float h = 0.5f * w;
                self(self, x, z, c);
                self(self, x + h, z, c);
                self(self, x, z + h, c);
                self(self, x + h, z + h, c);
            } else {
                out.push_back({x, z, s});
            }
        };
        for (int tries = 0; tries < 12; ++tries) {
            out.clear();
            visit(visit, x0, z0, s0);
            if (out.size() <= slots) break;
            v.finest *= 2.f;
        }
        if (out.size() > slots) out.resize(slots);// (never with the slot counts Ocean::create makes)
        return out;
    }

}// namespace threepp::water::lod

#endif//THREEPP_VULKAN_WATER_OCEANLOD_HPP
