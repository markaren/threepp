// Structural invariants for the geometry generators.
//
// These were entirely untested: ~2k lines of index/UV/normal arithmetic across
// 20-odd generators with no coverage at all. Rather than 20 bespoke tests, this
// asserts the properties that must hold for EVERY generated geometry, so a new
// generator gets covered by adding one line to the table. Off-by-one index math,
// desynced attribute arrays, NaN from degenerate parameters and non-unit normals
// are all caught here.

#include <catch2/catch_test_macros.hpp>

#include "threepp/geometries/geometries.hpp"

#include "threepp/geometries/ConvexGeometry.hpp"
#include "threepp/geometries/EdgesGeometry.hpp"
#include "threepp/geometries/ExtrudeGeometry.hpp"
#include "threepp/geometries/LatheGeometry.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"

#include "threepp/extras/core/Shape.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    struct Case {
        std::string name;
        std::function<std::shared_ptr<BufferGeometry>()> make;
    };

    Shape makeSquareShape() {
        Shape s;
        s.moveTo(0, 0);
        s.lineTo(1, 0);
        s.lineTo(1, 1);
        s.lineTo(0, 1);
        s.lineTo(0, 0);
        return s;
    }

    std::shared_ptr<Curve3> makeCurve() {
        std::vector<Vector3> pts{
                Vector3(0.f, 0.f, 0.f), Vector3(1.f, 1.f, 0.f),
                Vector3(2.f, 0.f, 1.f), Vector3(3.f, 1.f, 1.f)};
        return std::make_shared<CatmullRomCurve3>(std::move(pts));
    }

    // Every generator that can be built without external assets (no fonts, no
    // loaded meshes). TextGeometry / ExtrudeTextGeometry need a Font and are
    // covered by the font-loader tests instead.
    std::vector<Case> allCases() {
        std::vector<Case> cases;

        cases.push_back({"Box", [] { return BoxGeometry::create(1, 2, 3, 2, 3, 4); }});
        cases.push_back({"Box(1 segment)", [] { return BoxGeometry::create(); }});
        cases.push_back({"Plane", [] { return PlaneGeometry::create(2, 3, 4, 5); }});
        cases.push_back({"Plane(1 segment)", [] { return PlaneGeometry::create(); }});
        cases.push_back({"Sphere", [] { return SphereGeometry::create(1.5f, 16, 12); }});
        cases.push_back({"Sphere(min segments)", [] { return SphereGeometry::create(1, 3, 2); }});
        cases.push_back({"Circle", [] { return CircleGeometry::create(2, 16); }});
        cases.push_back({"Ring", [] { return RingGeometry::create(0.5f, 1.5f, 16, 4); }});
        cases.push_back({"Cylinder", [] { return CylinderGeometry::create(1, 2, 3, 16, 4); }});
        cases.push_back({"Cylinder(cone, top=0)", [] { return CylinderGeometry::create(0, 1, 2, 12, 1); }});
        cases.push_back({"Cone", [] { return ConeGeometry::create(1, 2, 16, 4); }});
        cases.push_back({"Capsule", [] { return CapsuleGeometry::create(0.5f, 2); }});
        cases.push_back({"Torus", [] { return TorusGeometry::create(2, 0.5f, 12, 20); }});
        cases.push_back({"TorusKnot", [] { return TorusKnotGeometry::create(1, 0.3f, 32, 8); }});
        cases.push_back({"Icosahedron", [] { return IcosahedronGeometry::create(1, 0); }});
        cases.push_back({"Icosahedron(detail 2)", [] { return IcosahedronGeometry::create(1, 2); }});
        cases.push_back({"Octahedron", [] { return OctahedronGeometry::create(1, 0); }});
        cases.push_back({"Octahedron(detail 1)", [] { return OctahedronGeometry::create(1, 1); }});
        cases.push_back({"Lathe", [] {
                             std::vector<Vector2> pts{
                                     Vector2(0.5f, -1.f), Vector2(0.8f, 0.f), Vector2(0.4f, 1.f)};
                             return LatheGeometry::create(pts, 12);
                         }});
        cases.push_back({"Tube", [] { return TubeGeometry::create(makeCurve(), 16, 0.3f, 8); }});
        cases.push_back({"Shape", [] { return ShapeGeometry::create(makeSquareShape()); }});
        cases.push_back({"Extrude", [] {
                             ExtrudeGeometry::Options opts;
                             opts.depth = 2;
                             std::vector<Shape> shapes{makeSquareShape()};
                             return ExtrudeGeometry::create(shapes, opts);
                         }});
        cases.push_back({"Convex", [] {
                             std::vector<Vector3> pts{
                                     {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}, {-1, 0.5f, 0.25f}};
                             return ConvexGeometry::create(pts);
                         }});

        return cases;
    }

    bool allFinite(const std::vector<float>& v) {
        for (const float f : v) {
            if (!std::isfinite(f)) return false;
        }
        return true;
    }

}// namespace

TEST_CASE("Geometry generators satisfy structural invariants") {

    for (const auto& c : allCases()) {

        // Catch2 reports the whole loop as one case, so name every failure.
        INFO("geometry: " << c.name);

        auto geometry = c.make();
        REQUIRE(geometry);

        // --- position -------------------------------------------------------
        auto* position = geometry->getAttribute<float>("position");
        REQUIRE(position);
        CHECK(position->itemSize() == 3);
        CHECK(position->count() > 0);
        CHECK(static_cast<int>(position->array().size()) == position->count() * 3);
        CHECK(allFinite(position->array()));

        // --- index ----------------------------------------------------------
        if (geometry->hasIndex()) {
            const auto* index = geometry->getIndex();
            REQUIRE(index);
            CHECK(index->count() > 0);
            // Triangle soup: a generator emitting a partial face is a bug.
            CHECK(index->array().size() % 3 == 0);

            // The invariant that matters most: no index may address outside the
            // position array. This is what a renderer would read out of bounds on.
            const auto vertexCount = static_cast<unsigned int>(position->count());
            unsigned int maxIndex = 0;
            for (const unsigned int i : index->array()) {
                maxIndex = std::max(maxIndex, i);
            }
            INFO("max index " << maxIndex << " vs " << vertexCount << " vertices");
            CHECK(maxIndex < vertexCount);

        } else {

            // Non-indexed geometries are drawn as a triangle soup, so a vertex
            // count that is not a multiple of 3 means a trailing partial face.
            CHECK(position->count() % 3 == 0);
        }

        // --- normal ---------------------------------------------------------
        if (auto* normal = geometry->getAttribute<float>("normal")) {
            CHECK(normal->itemSize() == 3);
            CHECK(normal->count() == position->count());
            CHECK(allFinite(normal->array()));

            // Every normal must be unit length, or lighting is wrong.
            double worstError = 0;
            for (int i = 0; i < normal->count(); ++i) {
                const double x = normal->getX(i), y = normal->getY(i), z = normal->getZ(i);
                worstError = std::max(worstError, std::abs(std::sqrt(x * x + y * y + z * z) - 1.0));
            }
            INFO("worst |normal| error " << worstError);
            CHECK(worstError < 1e-3);
        }

        // --- uv -------------------------------------------------------------
        if (auto* uv = geometry->getAttribute<float>("uv")) {
            CHECK(uv->itemSize() == 2);
            CHECK(uv->count() == position->count());
            CHECK(allFinite(uv->array()));
        }

        // --- bounds ---------------------------------------------------------
        geometry->computeBoundingBox();
        REQUIRE(geometry->boundingBox.has_value());
        const auto& bb = *geometry->boundingBox;
        CHECK(std::isfinite(bb.min().x));
        CHECK(std::isfinite(bb.max().x));
        CHECK(bb.min().x <= bb.max().x);
        CHECK(bb.min().y <= bb.max().y);
        CHECK(bb.min().z <= bb.max().z);

        geometry->computeBoundingSphere();
        REQUIRE(geometry->boundingSphere.has_value());
        CHECK(std::isfinite(geometry->boundingSphere->radius));
        CHECK(geometry->boundingSphere->radius > 0);
    }
}

// The derived geometries take another geometry as input, so they get their own
// pass over the same table.
TEST_CASE("Edges and wireframe geometries derive valid line lists") {

    for (const auto& c : allCases()) {

        INFO("source geometry: " << c.name);
        auto source = c.make();
        REQUIRE(source);

        SECTION("wireframe of " + c.name) {
            auto wire = WireframeGeometry::create(*source);
            REQUIRE(wire);
            auto* position = wire->getAttribute<float>("position");
            REQUIRE(position);
            CHECK(position->itemSize() == 3);
            // Line segments: two endpoints per line.
            CHECK(position->count() % 2 == 0);
            CHECK(position->count() > 0);
            CHECK(allFinite(position->array()));
        }

        SECTION("edges of " + c.name) {
            auto edges = EdgesGeometry::create(*source);
            REQUIRE(edges);
            auto* position = edges->getAttribute<float>("position");
            REQUIRE(position);
            CHECK(position->itemSize() == 3);
            CHECK(position->count() % 2 == 0);
            CHECK(allFinite(position->array()));
        }
    }
}

// Vertex/index counts for the parametric generators, pinned against the
// documented three.js formulas so a segment-loop off-by-one is caught rather
// than silently changing tessellation.
TEST_CASE("Parametric generators produce the documented vertex counts") {

    SECTION("PlaneGeometry") {
        const unsigned ws = 4, hs = 5;
        auto g = PlaneGeometry::create(2, 3, ws, hs);
        CHECK(g->getAttribute<float>("position")->count() == static_cast<int>((ws + 1) * (hs + 1)));
        REQUIRE(g->hasIndex());
        CHECK(g->getIndex()->array().size() == ws * hs * 6);
    }

    SECTION("BoxGeometry") {
        const unsigned ws = 2, hs = 3, ds = 4;
        auto g = BoxGeometry::create(1, 1, 1, ws, hs, ds);
        // Six planes: two each of w*h, w*d and d*h.
        const auto expected = 2 * ((ws + 1) * (hs + 1) + (ws + 1) * (ds + 1) + (ds + 1) * (hs + 1));
        CHECK(g->getAttribute<float>("position")->count() == static_cast<int>(expected));
        REQUIRE(g->hasIndex());
        CHECK(g->getIndex()->array().size() == 6ull * 2 * (ws * hs + ws * ds + ds * hs));
    }

    SECTION("SphereGeometry") {
        const unsigned ws = 16, hs = 12;
        auto g = SphereGeometry::create(1, ws, hs);
        CHECK(g->getAttribute<float>("position")->count() == static_cast<int>((ws + 1) * (hs + 1)));
    }

    SECTION("TorusGeometry") {
        const unsigned radial = 12, tubular = 20;
        auto g = TorusGeometry::create(2, 0.5f, radial, tubular);
        CHECK(g->getAttribute<float>("position")->count() == static_cast<int>((radial + 1) * (tubular + 1)));
        REQUIRE(g->hasIndex());
        CHECK(g->getIndex()->array().size() == radial * tubular * 6);
    }

    SECTION("RingGeometry") {
        const unsigned thetaSeg = 16, phiSeg = 4;
        auto g = RingGeometry::create(0.5f, 1.f, thetaSeg, phiSeg);
        CHECK(g->getAttribute<float>("position")->count() == static_cast<int>((thetaSeg + 1) * (phiSeg + 1)));
    }
}

TEST_CASE("Geometry groups stay within the index range") {

    // BoxGeometry emits one group per face for multi-material use. A group whose
    // start+count runs past the index buffer is an out-of-bounds draw call.
    auto g = BoxGeometry::create(1, 1, 1, 2, 2, 2);
    REQUIRE(g->hasIndex());
    const auto indexCount = g->getIndex()->array().size();

    REQUIRE(!g->groups.empty());
    for (const auto& group : g->groups) {
        INFO("group start " << group.start << " count " << group.count
                            << " vs index count " << indexCount);
        CHECK(group.start >= 0);
        CHECK(group.count >= 0);
        CHECK(static_cast<size_t>(group.start) + static_cast<size_t>(group.count) <= indexCount);
    }
}
