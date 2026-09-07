// V-HACD convex decomposition: split a (possibly concave) triangle mesh into a
// small set of convex hulls that PhysX can cook and simulate as one rigid body.
//
// A single convex hull roofs over every concavity — a mug holds no water, a
// chair seat and back fuse into a wedge. Decomposition is what makes the
// collider match the shape: the ball settles INSIDE the mug, and a body can
// fall through the gap between a chair's legs.
//
// The interface here is deliberately free of both PhysX and VHACD.h: it takes
// raw geometry (the position/index arrays a BufferGeometry already holds) and
// returns hull vertex sets as flat x,y,z float arrays, ready to hand straight to
// PhysxWorld::cookConvexHull. VHACD.h is a heavy single header whose
// implementation must live in exactly one translation unit — keeping it behind
// this boundary means only ConvexDecomposition.cpp pays that cost, and the many
// header-only PhysX consumers (PhysicsPlaySession) include none of it.

#ifndef THREEPP_PHYSX_CONVEXDECOMPOSITION_HPP
#define THREEPP_PHYSX_CONVEXDECOMPOSITION_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace threepp {

    struct ConvexDecompositionParams {
        // The V-HACD knobs the editor exposes. Names map onto VHACD::Parameters:
        // maxHulls -> m_maxConvexHulls, voxelResolution -> m_resolution,
        // maxVertsPerHull -> m_maxNumVerticesPerCH (also the PhysX cook target).
        std::uint32_t maxHulls = 16;
        std::uint32_t maxVertsPerHull = 64;
        std::uint32_t voxelResolution = 100000;
    };

    // Run V-HACD on an indexed triangle mesh (positions are tightly packed
    // x,y,z floats; indices are triangle vertex indices). Returns one flat
    // x,y,z float array per convex hull. Empty on a degenerate input (< 4
    // vertices, no triangles) or a decomposition failure — the caller decides
    // what to do with nothing (the editor falls back to a single hull).
    //
    // Runs synchronously: V-HACD's own async mode would hand control back before
    // the hulls exist, which is wrong for a Play button that must have colliders
    // ready before the first step. On a dense mesh this is seconds — the caller
    // is expected to log the cost.
    std::vector<std::vector<float>> decomposeConvex(
            const float* positions, std::size_t vertexCount,
            const std::uint32_t* indices, std::size_t indexCount,
            const ConvexDecompositionParams& params);

}// namespace threepp

#endif// THREEPP_PHYSX_CONVEXDECOMPOSITION_HPP
