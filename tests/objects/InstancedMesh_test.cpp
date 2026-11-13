// InstancedMesh per-instance accessor contracts.
//
// setColorAt/setMatrixAt/getColorAt/getMatrixAt address raw offsets into the
// instance buffers, so an index past the buffer capacity (most easily: any
// index at all on a count-0 mesh) used to be a silent out-of-bounds write.
// A count-0 InstancedMesh carrying an instanceColor array is reachable
// straight from a scene JSON document, so the loader path is pinned here too.

#include <catch2/catch_test_macros.hpp>

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"

#include <stdexcept>

using namespace threepp;

TEST_CASE("per-instance accessors reject out-of-range indices") {

    auto mesh = InstancedMesh::create(BoxGeometry::create(), MeshBasicMaterial::create(), 2);

    Matrix4 m;
    Color c;

    CHECK_THROWS_AS(mesh->setMatrixAt(2, m), std::out_of_range);
    CHECK_THROWS_AS(mesh->getMatrixAt(2, m), std::out_of_range);
    CHECK_THROWS_AS(mesh->setColorAt(2, c), std::out_of_range);

    // getColorAt on a mesh that never had colors set is a contract violation
    // of its own, distinct from an index problem.
    CHECK_THROWS_AS(mesh->getColorAt(0, c), std::runtime_error);

    // In-range still works, including the color-buffer lazy allocation.
    CHECK_NOTHROW(mesh->setMatrixAt(1, m));
    CHECK_NOTHROW(mesh->setColorAt(1, Color(0x00ff00)));
    CHECK_NOTHROW(mesh->getMatrixAt(1, m));
    CHECK_NOTHROW(mesh->getColorAt(1, c));
    CHECK(c.getHex() == 0x00ff00);
}

TEST_CASE("count-0 mesh rejects every per-instance access") {

    auto mesh = InstancedMesh::create(BoxGeometry::create(), MeshBasicMaterial::create(), 0);

    Matrix4 m;
    Color c;

    // setColorAt(0) on a count-0 mesh was the heap overflow: it allocated an
    // empty buffer and then wrote three floats into it.
    CHECK_THROWS_AS(mesh->setColorAt(0, c), std::out_of_range);
    CHECK_THROWS_AS(mesh->setMatrixAt(0, m), std::out_of_range);
    CHECK_THROWS_AS(mesh->getMatrixAt(0, m), std::out_of_range);
}

TEST_CASE("capacity, not the live count, bounds the writes") {

    auto mesh = InstancedMesh::create(BoxGeometry::create(), MeshBasicMaterial::create(), 4);

    mesh->setCount(1);
    CHECK(mesh->count() == 1);

    // Staging data for instances beyond the shrunken draw count is legal —
    // the buffer still holds capacity for them.
    Matrix4 m;
    CHECK_NOTHROW(mesh->setMatrixAt(3, m));
    CHECK_THROWS_AS(mesh->setMatrixAt(4, m), std::out_of_range);
}

TEST_CASE("ObjectLoader: count-0 InstancedMesh with instanceColor loads safely") {

    const std::string doc = R"({
        "metadata": {"version": 4.5, "type": "Object", "generator": "test"},
        "geometries": [{"uuid": "geo", "type": "BoxGeometry", "width": 1, "height": 1, "depth": 1}],
        "materials": [{"uuid": "mat", "type": "MeshBasicMaterial"}],
        "object": {
            "uuid": "obj",
            "type": "InstancedMesh",
            "geometry": "geo",
            "material": "mat",
            "count": 0,
            "instanceMatrix": {"itemSize": 16, "type": "Float32Array", "array": []},
            "instanceColor": {"itemSize": 3, "type": "Float32Array", "array": [1, 0, 0]},
            "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        }
    })";

    ObjectLoader loader;
    const auto object = loader.parse(doc);
    REQUIRE(object);

    auto* mesh = object->as<InstancedMesh>();
    REQUIRE(mesh);
    CHECK(mesh->count() == 0);
    // No instances means no per-instance color buffer to fill.
    CHECK(mesh->instanceColor() == nullptr);
}


TEST_CASE("recorded row edits: the rows since a version, or unknown") {
    auto mesh = InstancedMesh::create(BoxGeometry::create(), MeshBasicMaterial::create(), 100);
    const float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const unsigned v0 = mesh->instanceMatrix()->version;
    InstancedMesh::RowEdits ed;

    mesh->setMatrices(10, 1, m);
    mesh->setMatrices(40, 1, m, true);
    REQUIRE(mesh->matrixRowsSince(v0, ed));
    CHECK(ed.lo == 10);
    CHECK(ed.hi == 41);
    CHECK(ed.movedLo == 10);
    CHECK(ed.movedHi == 11);
    CHECK(!ed.placedOnly(10));
    CHECK(ed.placedOnly(40));

    // Since the newer version: the placed edit alone.
    REQUIRE(mesh->matrixRowsSince(v0 + 1, ed));
    CHECK(ed.lo == 40);
    CHECK(ed.hi == 41);
    CHECK(ed.movedHi <= ed.movedLo);

    // A bare needsUpdate is an edit with no rows: unknown from before it.
    const unsigned v1 = mesh->instanceMatrix()->version;
    mesh->instanceMatrix()->needsUpdate();
    CHECK(!mesh->matrixRowsSince(v1, ed));
    mesh->markMatrixRows(3, 2);
    CHECK(!mesh->matrixRowsSince(v1, ed));
    REQUIRE(mesh->matrixRowsSince(v1 + 1, ed));
    CHECK(ed.lo == 3);
    CHECK(ed.hi == 5);

    // More edits than the log keeps: unknown.
    const unsigned v2 = mesh->instanceMatrix()->version;
    for (int k = 0; k < 200; ++k) mesh->setMatrices(k % 100, 1, m);
    CHECK(!mesh->matrixRowsSince(v2, ed));

    // Fades: none until the first write, then recorded the same way.
    CHECK(!mesh->fadeRowsSince(0, ed));
    const float f[2] = {0.5f, 0.25f};
    mesh->setFades(7, 2, f);
    const unsigned fv = mesh->instanceFade()->version;
    mesh->setFadeAt(50, 0.5f);
    REQUIRE(mesh->fadeRowsSince(fv, ed));
    CHECK(ed.lo == 50);
    CHECK(ed.hi == 51);
}
