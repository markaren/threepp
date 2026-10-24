// EdgeRefine: crack-free refinement of a triangle mesh, edge by edge.
//
// The material displacementMap path (VulkanCoreImpl.hpp, DisplacementSpec)
// can only move the vertices a mesh has. A coarse mesh under a detailed map
// needs more of them, where they show: near the camera, finer than a few
// pixels. This splits the edges that are too long, pass after pass, at their
// midpoints, and re-triangulates each triangle by how many of its edges split
// (one: two triangles, two: three, three: four). An edge's verdict is a
// function of its two end positions alone, so the two triangles that share an
// edge always split it together, at one point: the refined mesh has no cracks
// and no T-junctions, also where a uv seam (or an unindexed mesh) duplicates
// the vertices along the edge.
//
// No device, no renderer state: unit-tested in tests/renderers/EdgeRefine_test.

#ifndef THREEPP_VULKAN_EDGEREFINE_HPP
#define THREEPP_VULKAN_EDGEREFINE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace threepp::vulkan {

    // A per-vertex float attribute carried through the refinement. A new
    // vertex gets the mean of its edge's two ends, renormalised when `unit`
    // (a normal). attrs[0] must be the positions (itemSize 3).
    struct RefineAttribute {
        std::vector<float> data;
        int itemSize = 3;
        bool unit = false;
    };

    struct RefineLimits {
        int maxPasses = 16;// each pass at most halves an edge
        std::size_t maxTriangles = 4'000'000;// a pass that would exceed it is not applied
    };

    struct RefineStats {
        int passes = 0;          // passes applied
        std::size_t triangles = 0;// in the result
        bool hitBudget = false;   // stopped by maxTriangles with edges still too long
    };

    // `split(pa, pb)`: whether the edge between the positions pa and pb
    // (3 floats each) is split. It must be symmetric in its ends and read
    // nothing but them (see the header note). `indices` holds triangles
    // (3 per), refined in place, with the attributes grown to match.
    RefineStats refineByEdges(std::vector<uint32_t>& indices, std::vector<RefineAttribute>& attrs,
                              const std::function<bool(const float* pa, const float* pb)>& split,
                              const RefineLimits& limits = {});

    // The displacementMap refinement's edge test (VulkanRenderer::
    // setDisplacementDetail), one implementation for the renderer and the
    // tests. An edge (object space) splits while it is longer, in the world,
    // than pixelEdge output pixels at its midpoint's distance from the eye (and
    // than minEdge), unless the map's relief there is under half a pixel. With
    // `frustum`, an edge whose bounding sphere, grown by the relief, lies
    // outside the eye's view (widened by frustumMargin) aims for an
    // outsideFactor times longer edge instead: a coarse ring behind the
    // camera, and a large triangle over the view still splits until its edges
    // reach it. Every term reads the edge's midpoint and length only, so the
    // test is symmetric in its ends (EdgeRefine's contract).
    struct ScreenEdgeTest {
        std::array<float, 16> world{};// object → world, column-major
        std::array<float, 16> view{}; // world → eye (the camera's matrixWorldInverse)
        std::array<float, 3> eye{};
        float pixelAngle = 0.f;// radians per output pixel (perspective)
        float orthoPixel = 0.f;// metres per output pixel (orthographic); 0 = perspective
        float halfX = 0.f, halfY = 0.f;// the view's half extents: tangents (perspective) or metres (ortho)
        float pixelEdge = 0.f, minEdge = 0.01f;
        float reach = 0.f;// the map's largest displacement, world metres
        bool frustum = true;
        float frustumMargin = 0.15f;// the view widened by this share of its half extents
        float outsideFactor = 8.f;

        bool operator()(const float* pa, const float* pb) const;
        // Whether a sphere (world centre, radius) reaches into the widened view.
        [[nodiscard]] bool inView(const float* c, float r) const;
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_EDGEREFINE_HPP
