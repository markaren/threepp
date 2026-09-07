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
