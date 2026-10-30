
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/physx/PhysxSoftBody.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#ifdef THREEPP_PHYSX_CUDA_GL_INTEROP
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#endif

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace threepp;

// Where the SDK has no CUDA backend (macOS) PhysxSoftBody.hpp is empty; one case
// says so rather than leaving Catch2 with nothing to run.
#if !THREEPP_PHYSX_GPU

TEST_CASE("soft bodies need a PhysX build with a CUDA backend", "[physx][gpu]") {
    SUCCEED("PX_SUPPORT_GPU_PHYSX is 0 on this platform - soft bodies are compiled out");
}

#else

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
        SKIP("no CUDA device - soft body binding not exercised");
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

#ifdef THREEPP_PHYSX_CUDA_GL_INTEROP

namespace {

    constexpr int kSize = 128;
    constexpr float kHalfExtent = 4.f;// the ortho camera frames [-4, 4] m on both axes

    // World-space height of the centre of everything that is not background.
    // readRGBPixels returns rows bottom-up, so row 0 is the bottom of the frame.
    float litCentroidY(GLRenderer& renderer) {

        const auto px = renderer.readRGBPixels();
        REQUIRE(px.size() == static_cast<std::size_t>(kSize * kSize * 3));
        double sumRow = 0;
        long lit = 0;
        for (int row = 0; row < kSize; ++row) {
            for (int col = 0; col < kSize; ++col) {
                if (px[(row * kSize + col) * 3] > 128) {
                    sumRow += row + 0.5;
                    ++lit;
                }
            }
        }
        REQUIRE(lit > 0);
        return static_cast<float>(sumRow / lit) * (2.f * kHalfExtent / kSize) - kHalfExtent;
    }

}// namespace


// On GL a GPU-skinned body is drawn from a small texture of its tet positions.
// Once that texture is registered with CUDA, PhysX's positions are copied into
// it on the device and the host upload stops. So after registration the only
// way the drawn body can follow the simulation is through that device copy:
// a body that falls in PhysX must fall on screen by the same amount.
TEST_CASE("a GPU-skinned soft body is drawn from PhysX's device positions on GL", "[physx][gpu]") {

    auto world = gpuWorld();
    if (!world) {
        SKIP("no CUDA device - GL tet interop not exercised");
    }

    std::shared_ptr<Canvas> canvas;
    try {
        canvas = std::make_shared<Canvas>(Canvas::Parameters().size(kSize, kSize).headless(true));
    } catch (const std::exception& e) {
        SKIP("no GL context: " << e.what());
    }

    GLRenderer renderer(*canvas);
    renderer.setClearColor(Color::black);

    Scene scene;
    OrthographicCamera camera(-kHalfExtent, kHalfExtent, kHalfExtent, -kHalfExtent, 0.1f, 20.f);
    camera.position.set(0.f, 0.f, 10.f);
    camera.updateMatrixWorld();

    // Unlit, and the Phong material the soft-body example draws with; the lit
    // one glows so it needs no light to be counted.
    std::shared_ptr<Material> material;
    SECTION("unlit") {
        material = MeshBasicMaterial::create();
    }
    SECTION("Phong") {
        auto phong = MeshPhongMaterial::create();
        phong->emissive = Color::white;
        material = phong;
    }

    // Nothing to land on: the body is in free fall for the whole test.
    auto mesh = Mesh::create(SphereGeometry::create(0.5f, 16, 12), material);
    mesh->position.set(0.f, 3.f, 0.f);
    scene.add(mesh);
    auto* body = world->addSoftBody(*mesh, nullptr, 6, 15, false, "", 1.f);
    REQUIRE(body);
    body->enableGpuSkinning();

    // The tet texture has no GL id until the body has been drawn once.
    renderer.render(scene, camera);
    const float before = litCentroidY(renderer);
    CHECK(std::fabs(before - 3.f) < 0.1f);

    REQUIRE(body->needsInteropRegister());
    const auto texId = renderer.getGlTextureId(*body->interopTexture());
    REQUIRE(texId);
    REQUIRE(body->registerGlTexture(*texId));

    for (int i = 0; i < 30; ++i) world->step(1.f / 60.f);

    renderer.render(scene, camera);
    const float after = litCentroidY(renderer);
    const float simulated = body->actor()->getWorldBounds().getCenter().y;

    CHECK(before - simulated > 1.f);// half a second of free fall
    CHECK(std::fabs(after - simulated) < 0.1f);// a pixel is 1/16 m

    // Unregister while the GL context is still alive.
    world->removeSoftBody(body);
}

#endif// THREEPP_PHYSX_CUDA_GL_INTEROP

#endif// THREEPP_PHYSX_GPU
