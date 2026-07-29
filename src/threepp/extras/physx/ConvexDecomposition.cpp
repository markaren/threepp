#include "threepp/extras/physx/ConvexDecomposition.hpp"

// The single translation unit that carries V-HACD's implementation. VHACD.h is
// header-only with the classic "define the macro in one .cpp" pattern; this is
// that .cpp. Nothing else in the build defines it, so the (large) implementation
// is emitted exactly once.
#define ENABLE_VHACD_IMPLEMENTATION 1
#include <VHACD.h>

#include <algorithm>

namespace threepp {

    std::vector<std::vector<float>> decomposeConvex(
            const float* positions, std::size_t vertexCount,
            const std::uint32_t* indices, std::size_t indexCount,
            const ConvexDecompositionParams& params) {

        std::vector<std::vector<float>> hulls;
        if (!positions || vertexCount < 4 || !indices || indexCount < 3) return hulls;

        VHACD::IVHACD* vhacd = VHACD::CreateVHACD();
        if (!vhacd) return hulls;

        VHACD::IVHACD::Parameters p;
        p.m_maxConvexHulls = std::max<std::uint32_t>(params.maxHulls, 1);
        p.m_resolution = std::max<std::uint32_t>(params.voxelResolution, 10000);
        // Clamp to PhysX's GPU-compatible convex ceiling (64); a hull with more
        // planes than that buys precision no rigid contact needs, and would be
        // rejected by the cook target anyway.
        p.m_maxNumVerticesPerCH = std::clamp<std::uint32_t>(params.maxVertsPerHull, 8, 64);
        // Synchronous: the caller (a Play press) needs the hulls before it can
        // build actors, so there is nothing to gain from a background thread and
        // everything to lose from returning before it finished.
        p.m_asyncACD = false;

        const bool ok = vhacd->Compute(
                positions, static_cast<std::uint32_t>(vertexCount),
                indices, static_cast<std::uint32_t>(indexCount / 3), p);
        if (!ok) {
            vhacd->Clean();
            vhacd->Release();
            return hulls;
        }

        const std::uint32_t n = vhacd->GetNConvexHulls();
        hulls.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            VHACD::IVHACD::ConvexHull ch;
            if (!vhacd->GetConvexHull(i, ch)) continue;
            std::vector<float> flat;
            flat.reserve(ch.m_points.size() * 3);
            for (const auto& v : ch.m_points) {
                // V-HACD works in double; PhysX cooks floats.
                flat.push_back(static_cast<float>(v.mX));
                flat.push_back(static_cast<float>(v.mY));
                flat.push_back(static_cast<float>(v.mZ));
            }
            if (flat.size() >= 12) hulls.push_back(std::move(flat));// >= 4 points
        }

        vhacd->Clean();
        vhacd->Release();
        return hulls;
    }

}// namespace threepp
