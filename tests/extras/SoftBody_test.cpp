
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/physx/PhysxSoftBody.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace threepp;

namespace {

    // World-space positions of a mesh's vertices, before anything touches it.
    std::vector<Vector3> worldVertices(Mesh& mesh) {

        mesh.updateMatrixWorld(true);
        const auto* positions = mesh.geometry()->getAttribute<float>("position");
        REQUIRE(positions);
        std::vector<Vector3> out;
        out.reserve(positions->count());
        for (unsigned i = 0; i < positions->count(); ++i) {
            Vector3 v(positions->getX(i), positions->getY(i), positions->getZ(i));
            v.applyMatrix4(*mesh.matrixWorld);
            out.push_back(v);
        }
        return out;
    }

    // Creating a soft body immediately re-skins the visual mesh from the tet
    // mesh, so this is the reconstruction error of the rest pose itself.
    float restPoseError(Mesh& mesh, const std::vector<Vector3>& expected) {

        const auto* positions = mesh.geometry()->getAttribute<float>("position");
        REQUIRE(positions);
        REQUIRE(positions->count() == expected.size());
        float worst = 0.f;
        unsigned worstIndex = 0;
        Vector3 worstActual;
        for (unsigned i = 0; i < positions->count(); ++i) {
            const Vector3 actual(positions->getX(i), positions->getY(i), positions->getZ(i));
            if (expected[i].distanceTo(actual) > worst) {
                worst = expected[i].distanceTo(actual);
                worstIndex = i;
                worstActual = actual;
            }
        }
        if (worst > 1e-3f) {
            std::cout << "   worst v" << worstIndex << " of " << positions->count()
                      << " expected (" << expected[worstIndex].x << "," << expected[worstIndex].y
                      << "," << expected[worstIndex].z << ") got (" << worstActual.x << ","
                      << worstActual.y << "," << worstActual.z << ")" << std::endl;
        }
        return worst;
    }

    std::unique_ptr<PhysxWorld> gpuWorld() {

        PhysxWorld::Settings settings;
        settings.enableGpuDynamics = true;
        try {
            return std::make_unique<PhysxWorld>(settings);
        } catch (const std::exception&) {
            return nullptr;// no CUDA device here
        }
    }

}// namespace


// A soft body's visual mesh is rebuilt from the tet mesh every step, including
// the very first one. If the binding is wrong, the mesh is wrong the instant
// Play is pressed — before any simulation has had a chance to deform it.
//
// The failure this pins: the tet hull cooked from a remeshed surface does not
// quite reach a sharp corner of a ROTATED box, and every vertex that lands
// outside the binder's AABB pre-filter used to be bound to an arbitrary tet.
// Those vertices — and the faces they belong to — collapsed into the middle of
// the body, which reads on screen as the mesh being cut open.
TEST_CASE("a soft body reproduces the mesh it was built from", "[physx][gpu]") {

    auto world = gpuWorld();
    if (!world) {
        std::cout << "[skip] no CUDA device - soft body binding not exercised" << std::endl;
        return;
    }

    const auto check = [&](const char* what, const std::shared_ptr<BufferGeometry>& geometry,
                           const Euler& rotation, const Vector3& scale) {
        INFO(what);
        auto mesh = Mesh::create(geometry, MeshBasicMaterial::create());
        mesh->position.set(0.f, 3.f, 0.f);
        mesh->rotation.copy(rotation);
        mesh->scale.copy(scale);

        const auto expected = worldVertices(*mesh);
        auto* body = world->addSoftBody(*mesh, nullptr, 6, 15, false, "", 1.f);
        REQUIRE(body);

        // Sub-millimetre on a 1 m box: the binding must be exact, not "close
        // enough that it looks like a box".
        CHECK(restPoseError(*mesh, expected) < 1e-3f);
        world->removeSoftBody(body);
    };

    const Vector3 unit(1.f, 1.f, 1.f);
    check("axis-aligned box", BoxGeometry::create(), Euler(), unit);
    check("box yawed 45 deg", BoxGeometry::create(), Euler(0.f, 0.7853982f, 0.f), unit);
    check("box yawed 10 deg", BoxGeometry::create(), Euler(0.f, 0.1745329f, 0.f), unit);
    check("box yawed + pitched", BoxGeometry::create(), Euler(0.4f, 0.7853982f, 0.f), unit);
    check("scaled, rotated box", BoxGeometry::create(), Euler(0.f, 0.7853982f, 0.f),
          Vector3(1.5f, 0.75f, 1.f));
    check("sphere", SphereGeometry::create(0.5f, 16, 12), Euler(), unit);
    // A cone's apex is the sharpest corner a stock primitive has, so it is the
    // first thing the cooked hull cuts off — reported from the editor.
    check("cone", ConeGeometry::create(0.5f, 1.f), Euler(), unit);
    check("rotated cone", ConeGeometry::create(0.5f, 1.f), Euler(0.f, 0.6f, 0.35f), unit);
    // This one cooks to exactly as many tet vertices as the sphere has (221),
    // which used to be taken as proof that the two correspond index-for-index.
    check("rotated sphere", SphereGeometry::create(0.5f, 16, 12), Euler(0.3f, 0.9f, 0.f), unit);
}
