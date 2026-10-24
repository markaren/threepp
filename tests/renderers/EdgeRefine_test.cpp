// EdgeRefine: the displacementMap refinement must leave a conforming mesh
// (every interior edge shared by exactly two triangles, in opposite
// directions, so no crack and no T-junction), cover the same area with the
// same winding, stop at its triangle budget and carry the attributes. CPU only.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/renderers/vulkan/EdgeRefine.hpp"

#include <cmath>
#include <map>
#include <tuple>
#include <utility>

using namespace threepp::vulkan;
using Catch::Matchers::WithinAbs;

namespace {

    // The unit square in z = 0 as two counter-clockwise triangles, with uv =
    // xy and normals +z; unindexed = six vertices, each triangle its own.
    void unitSquare(std::vector<uint32_t>& idx, std::vector<RefineAttribute>& attrs, bool unindexed) {
        const float corners[4][2] = {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
        const uint32_t tris[6] = {0, 1, 2, 0, 2, 3};
        attrs.assign(3, {});
        attrs[0].itemSize = 3;
        attrs[1].itemSize = 3;
        attrs[1].unit = true;
        attrs[2].itemSize = 2;
        idx.clear();
        const int n = unindexed ? 6 : 4;
        for (int i = 0; i < n; ++i) {
            const float* c = corners[unindexed ? tris[i] : i];
            attrs[0].data.insert(attrs[0].data.end(), {c[0], c[1], 0.f});
            attrs[1].data.insert(attrs[1].data.end(), {0.f, 0.f, 1.f});
            attrs[2].data.insert(attrs[2].data.end(), {c[0], c[1]});
        }
        for (uint32_t i = 0; i < 6; ++i) idx.push_back(unindexed ? i : tris[i]);
    }

    float len(const float* a, const float* b) {
        return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
    }

    // Welds by position (the refinement's midpoints are exact means, so equal
    // edges give bit-equal points), then checks the square stays conforming.
    void checkConforming(const std::vector<uint32_t>& idx, const std::vector<float>& pos) {
        std::map<std::tuple<float, float, float>, uint32_t> weld;
        std::vector<uint32_t> id(pos.size() / 3);
        std::vector<std::pair<float, float>> xy;
        for (size_t v = 0; v < id.size(); ++v) {
            const auto key = std::make_tuple(pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]);
            auto [it, fresh] = weld.try_emplace(key, static_cast<uint32_t>(xy.size()));
            if (fresh) xy.emplace_back(pos[3 * v], pos[3 * v + 1]);
            id[v] = it->second;
        }
        std::map<std::pair<uint32_t, uint32_t>, int> directed;
        double area = 0.0;
        for (size_t t = 0; t < idx.size(); t += 3) {
            const uint32_t a = id[idx[t]], b = id[idx[t + 1]], c = id[idx[t + 2]];
            const double cross = (xy[b].first - xy[a].first) * (xy[c].second - xy[a].second) -
                                 (xy[b].second - xy[a].second) * (xy[c].first - xy[a].first);
            REQUIRE(cross > 0.0);// counter-clockwise, as the input: winding kept, no sliver flipped
            area += 0.5 * cross;
            ++directed[{a, b}];
            ++directed[{b, c}];
            ++directed[{c, a}];
        }
        CHECK_THAT(area, WithinAbs(1.0, 1e-5));
        const auto onBorder = [&](uint32_t a, uint32_t b) {
            for (int axis = 0; axis < 2; ++axis) {
                const float pa = axis ? xy[a].second : xy[a].first, pb = axis ? xy[b].second : xy[b].first;
                if (pa == pb && (pa == 0.f || pa == 1.f)) return true;
            }
            return false;
        };
        for (const auto& [e, n] : directed) {
            REQUIRE(n == 1);// no edge twice in one direction
            const bool twin = directed.count({e.second, e.first}) != 0;
            // An edge with no opposite twin is a border edge; anywhere else it
            // is a crack or a T-junction.
            if (!twin) REQUIRE(onBorder(e.first, e.second));
        }
    }

}// namespace

TEST_CASE("a uniform target refines the square into a conforming mesh") {
    std::vector<uint32_t> idx;
    std::vector<RefineAttribute> attrs;
    unitSquare(idx, attrs, false);
    const auto st = refineByEdges(idx, attrs, [](const float* a, const float* b) { return len(a, b) > 0.1f; });
    CHECK(st.passes > 0);
    CHECK_FALSE(st.hitBudget);
    CHECK(st.triangles == idx.size() / 3);
    checkConforming(idx, attrs[0].data);
    // No edge left longer than the target.
    const auto& p = attrs[0].data;
    for (size_t t = 0; t < idx.size(); t += 3)
        for (int k = 0; k < 3; ++k)
            CHECK(len(&p[3 * idx[t + k]], &p[3 * idx[t + (k + 1) % 3]]) <= 0.1f);
}

TEST_CASE("a target graded by distance leaves no crack between fine and coarse") {
    std::vector<uint32_t> idx;
    std::vector<RefineAttribute> attrs;
    unitSquare(idx, attrs, false);
    // Fine at the origin, coarse toward (1, 1): the shape the camera focus makes.
    const auto split = [](const float* a, const float* b) {
        const float mx = 0.5f * (a[0] + b[0]), my = 0.5f * (a[1] + b[1]);
        return len(a, b) > 0.01f + 0.25f * std::sqrt(mx * mx + my * my);
    };
    const auto st = refineByEdges(idx, attrs, split);
    CHECK(st.passes >= 5);
    checkConforming(idx, attrs[0].data);
}

TEST_CASE("an unindexed mesh splits its duplicated edges at one point") {
    std::vector<uint32_t> idx;
    std::vector<RefineAttribute> attrs;
    unitSquare(idx, attrs, true);
    const auto split = [](const float* a, const float* b) {
        const float mx = 0.5f * (a[0] + b[0]);
        return len(a, b) > 0.02f + 0.3f * mx;
    };
    refineByEdges(idx, attrs, split);
    checkConforming(idx, attrs[0].data);
}

TEST_CASE("the refinement stops before a pass would pass its triangle budget") {
    std::vector<uint32_t> idx;
    std::vector<RefineAttribute> attrs;
    unitSquare(idx, attrs, false);
    RefineLimits lim;
    lim.maxTriangles = 500;
    const auto st = refineByEdges(idx, attrs, [](const float* a, const float* b) { return len(a, b) > 0.001f; }, lim);
    CHECK(st.hitBudget);
    CHECK(idx.size() / 3 <= 500);
    CHECK(st.triangles == idx.size() / 3);
    checkConforming(idx, attrs[0].data);
}

TEST_CASE("new vertices carry the mean of their edge, unit attributes renormalised") {
    std::vector<uint32_t> idx;
    std::vector<RefineAttribute> attrs;
    unitSquare(idx, attrs, false);
    // A normal that varies, so the mean is shorter than one.
    attrs[1].data = {0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f};
    refineByEdges(idx, attrs, [](const float* a, const float* b) { return len(a, b) > 0.3f; });
    const auto& p = attrs[0].data;
    const auto& n = attrs[1].data;
    const auto& uv = attrs[2].data;
    for (size_t v = 0; v < p.size() / 3; ++v) {
        CHECK_THAT(uv[2 * v], WithinAbs(p[3 * v], 1e-6));// uv = xy on the square
        CHECK_THAT(uv[2 * v + 1], WithinAbs(p[3 * v + 1], 1e-6));
        CHECK_THAT(std::sqrt(n[3 * v] * n[3 * v] + n[3 * v + 1] * n[3 * v + 1] + n[3 * v + 2] * n[3 * v + 2]),
                   WithinAbs(1.0, 1e-5));
    }
}

namespace {

    // A camera at (0, 0, 1.6) looking along +y over the z = 0 plane: the
    // view matrix (world → eye, column-major) of that pose, by hand.
    ScreenEdgeTest groundCamera() {
        ScreenEdgeTest t;
        t.view = {1.f, 0.f, 0.f, 0.f,
                  0.f, 0.f, -1.f, 0.f,
                  0.f, 1.f, 0.f, 0.f,
                  0.f, -1.6f, 0.f, 1.f};
        t.eye = {0.f, 0.f, 1.6f};
        t.halfY = std::tan(0.5f * 50.f * 3.14159265f / 180.f);
        t.halfX = t.halfY * 16.f / 9.f;
        t.pixelAngle = 2.f * t.halfY / 720.f;
        t.pixelEdge = 6.f;
        t.minEdge = 0.01f;
        t.reach = 0.05f;
        // object = world
        t.world = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        return t;
    }

}// namespace

TEST_CASE("the screen edge test is symmetric in its ends") {
    const ScreenEdgeTest t = groundCamera();
    uint32_t s = 12345u;
    const auto rnd = [&s] {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) / static_cast<float>(1u << 24) * 40.f - 20.f;
    };
    for (int i = 0; i < 2000; ++i) {
        const float a[3] = {rnd(), rnd(), 0.f}, b[3] = {a[0] + rnd() * 0.01f, a[1] + rnd() * 0.01f, 0.f};
        REQUIRE(t(a, b) == t(b, a));
    }
}

TEST_CASE("the screen edge test aims coarser outside the view, unless the limit is off") {
    ScreenEdgeTest t = groundCamera();
    // 0.1 m edges 5 m ahead and 5 m behind: a pixel there is ~6.5 mm, the
    // target 6 px ~3.9 cm in view and 8x that behind.
    const float fa[3] = {-0.05f, 5.f, 0.f}, fb[3] = {0.05f, 5.f, 0.f};
    const float ba[3] = {-0.05f, -5.f, 0.f}, bb[3] = {0.05f, -5.f, 0.f};
    CHECK(t(fa, fb));
    CHECK_FALSE(t(ba, bb));
    t.frustum = false;
    CHECK(t(ba, bb));
    // Relief under half a pixel: no split, however long the edge.
    t.reach = 0.f;
    CHECK_FALSE(t(fa, fb));
}

TEST_CASE("a ground refined around a camera stays conforming and spends its triangles in view") {
    std::vector<uint32_t> idx;
    std::vector<RefineAttribute> attrs;
    unitSquare(idx, attrs, false);
    // The unit square scaled to 20 x 20 m around the camera (object → world).
    ScreenEdgeTest t = groundCamera();
    t.world = {20.f, 0.f, 0.f, 0.f, 0.f, 20.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, -10.f, -10.f, 0.f, 1.f};
    const auto st = refineByEdges(idx, attrs, t);
    CHECK_FALSE(st.hitBudget);
    checkConforming(idx, attrs[0].data);
    // Triangles ahead of the camera (world y > 0) against those behind.
    size_t ahead = 0, behind = 0;
    const auto& p = attrs[0].data;
    for (size_t k = 0; k < idx.size(); k += 3) {
        const float cy = (p[3 * idx[k] + 1] + p[3 * idx[k + 1] + 1] + p[3 * idx[k + 2] + 1]) / 3.f * 20.f - 10.f;
        (cy > 0.f ? ahead : behind) += 1;
    }
    INFO("ahead " << ahead << ", behind " << behind);
    CHECK(ahead > 10 * behind);
}
