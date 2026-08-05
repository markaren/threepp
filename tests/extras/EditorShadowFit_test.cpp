// Fitting a directional light's shadow camera to the scene.
//
// DirectionalLightShadow defaults to Ortho(-5,5,5,-5, 0.5, 500) — three.js
// expects the author to replace it, and until now nothing in the editor did.
// The editor's own template scene outgrows it: a 20x20 ground against a 10x10
// camera, so three quarters of the floor silently cast and received nothing,
// while the quarter that worked sat in 2% of the depth range at 205 shadow
// texels per world unit.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/ShadowFit.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cmath>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // EditorApp::buildTemplateScene, as far as shadows are concerned.
    std::shared_ptr<Scene> templateScene(std::shared_ptr<DirectionalLight>& sunOut) {

        auto scene = Scene::create();

        auto sun = DirectionalLight::create(0xffffff, 2.5f);
        sun->position.set(6, 10, 5);
        sun->castShadow = true;
        scene->add(sun);
        sunOut = sun;

        auto ground = Mesh::create(PlaneGeometry::create(20, 20), MeshBasicMaterial::create());
        ground->rotation.x = -math::PI / 2;
        ground->receiveShadow = true;
        scene->add(ground);

        auto box = Mesh::create(BoxGeometry::create(1, 1, 1), MeshBasicMaterial::create());
        box->position.set(0, 0.5f, 0);
        box->castShadow = true;
        box->receiveShadow = true;
        scene->add(box);

        return scene;
    }

    OrthographicCamera& shadowCameraOf(DirectionalLight& light) {

        auto* camera = dynamic_cast<OrthographicCamera*>(light.shadow->camera.get());
        REQUIRE(camera != nullptr);
        return *camera;
    }

}// namespace

TEST_CASE("ShadowFit covers a scene the default camera does not") {

    std::shared_ptr<DirectionalLight> sun;
    auto scene = templateScene(sun);

    auto& camera = shadowCameraOf(*sun);

    // The default, and the reason this exists.
    REQUIRE(camera.right == 5.f);

    REQUIRE(ShadowFit::fitAll(*scene) == 1);

    // The ground's own half-diagonal is sqrt(10^2 + 10^2) = 14.14; a camera
    // that does not reach it leaves the corners unshadowed.
    INFO("extent " << camera.right);
    CHECK(camera.right >= 14.14f);
    CHECK(camera.left == -camera.right);
    CHECK(camera.top == camera.right);
    CHECK(camera.bottom == -camera.right);

    // Square, so the extent does not change as the light swings around and the
    // shadow does not swim.
    CHECK(camera.right - camera.top == 0.f);
}

TEST_CASE("ShadowFit puts the scene inside the depth range, not in 2% of it") {

    std::shared_ptr<DirectionalLight> sun;
    auto scene = templateScene(sun);
    auto& camera = shadowCameraOf(*sun);

    ShadowFit::fitAll(*scene);

    const float range = camera.farPlane - camera.nearPlane;
    INFO("near " << camera.nearPlane << " far " << camera.farPlane);

    // The light stands ~12.85 away with a scene radius of ~14.2, so the whole
    // of it has to fall between the planes...
    CHECK(camera.nearPlane > 0.f);
    CHECK(camera.farPlane > camera.nearPlane);
    CHECK(camera.farPlane >= 12.85f);

    // ...and the range must be the scene's size, not the default's 499.5.
    CHECK(range < 40.f);
}

TEST_CASE("ShadowFit reports nothing to do when already fitted") {

    std::shared_ptr<DirectionalLight> sun;
    auto scene = templateScene(sun);

    CHECK(ShadowFit::fitAll(*scene) == 1);

    // So pressing the button twice does not stack an empty undo entry.
    CHECK(ShadowFit::fitAll(*scene) == 0);
}

// The reason this is a button and not a mode.
//
// A ground plane large enough dominates the bounds, and the fit then spreads
// the map so thin that whatever stands on it casts a shadow a couple of texels
// wide. No rule fixes that in general — the extent has to become the author's
// to set — so what is pinned here is that the trade-off is real and visible in
// the numbers, not that some heuristic hides it.
TEST_CASE("ShadowFit on a huge ground spreads the map thin") {

    std::shared_ptr<DirectionalLight> sun;
    auto scene = templateScene(sun);
    auto& camera = shadowCameraOf(*sun);

    auto huge = Mesh::create(PlaneGeometry::create(1000, 1000), MeshBasicMaterial::create());
    huge->rotation.x = -math::PI / 2;
    huge->receiveShadow = true;
    scene->add(huge);

    ShadowFit::fitAll(*scene);

    // sqrt(500^2 + 500^2) = 707.
    INFO("extent " << camera.right);
    CHECK(camera.right > 700.f);

    // Which is the point: a 2048 map over that is under 1.5 texels per world
    // unit, so the 1x1 box's shadow lands in about one texel.
    const float texelsPerUnit = 2048.f / (2.f * camera.right);
    INFO("texels per world unit " << texelsPerUnit);
    CHECK(texelsPerUnit < 2.f);

    // And the author's recourse is the field the inspector now exposes.
    camera.left = -12.f;
    camera.right = 12.f;
    camera.top = 12.f;
    camera.bottom = -12.f;
    camera.updateProjectionMatrix();
    CHECK(2048.f / (2.f * camera.right) > 80.f);
}

TEST_CASE("ShadowFit follows the scene as it grows") {

    std::shared_ptr<DirectionalLight> sun;
    auto scene = templateScene(sun);
    auto& camera = shadowCameraOf(*sun);

    ShadowFit::fitAll(*scene);
    const float before = camera.right;

    auto far = Mesh::create(BoxGeometry::create(2, 2, 2), MeshBasicMaterial::create());
    far->position.set(60, 1, 0);
    far->castShadow = true;
    scene->add(far);

    CHECK(ShadowFit::fitAll(*scene) == 1);
    INFO("extent " << before << " -> " << camera.right);
    CHECK(camera.right > before);
    CHECK(camera.right >= 30.f);
}

TEST_CASE("ShadowFit ignores objects that neither cast nor receive") {

    std::shared_ptr<DirectionalLight> sun;
    auto scene = templateScene(sun);
    auto& camera = shadowCameraOf(*sun);

    ShadowFit::fitAll(*scene);
    const float before = camera.right;

    // Editor furniture and background props should not stretch the map over
    // ground that will never show a shadow.
    auto decor = Mesh::create(BoxGeometry::create(1, 1, 1), MeshBasicMaterial::create());
    decor->position.set(500, 0, 0);
    scene->add(decor);

    CHECK(ShadowFit::fitAll(*scene) == 0);
    CHECK(camera.right == before);
}
