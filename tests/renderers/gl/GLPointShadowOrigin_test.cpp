// A point light's shadow must not depend on where the world origin is.
//
// The depth variant used to draw a point light's shadow map is a
// MeshDistanceMaterial, which encodes distance from a referencePosition over a
// [nearDistance, farDistance] range. GLShadowMap set those three on the wrong
// object: it cast `material` — the scene object's OWN material, a
// MeshStandardMaterial in any ordinary scene — to MeshDistanceMaterial, instead
// of `result`, the distance variant it had just prepared and was about to render
// with. The cast therefore always failed and nothing was set, leaving
// referencePosition at the origin and near/far at their defaults, whatever the
// light and its shadow camera actually were. r129 tests
// `result.isMeshDistanceMaterial` (WebGLShadowMap.js:360).
//
// A fixed referencePosition is invisible near the origin and wrong everywhere
// else, so this renders one rig twice: once around the origin and once with the
// entire rig — floor, caster, light and camera — translated far away. The two
// images are the same scene, so they must agree.

#include "gl_test_helpers.hpp"

#include "threepp/lights/PointLight.hpp"

namespace {

    // Fraction of the frame that is in shadow: floor pixels noticeably darker
    // than the lit floor, but not background.
    double shadowFraction(const Vector3& origin) {
        auto scene = Scene::create();

        auto floorMat = MeshStandardMaterial::create();
        floorMat->color = Color(1, 1, 1);
        floorMat->roughness = 1.f;
        floorMat->metalness = 0.f;
        auto floor = Mesh::create(PlaneGeometry::create(20, 20), floorMat);
        floor->rotation.x = -math::PI / 2.f;
        floor->position.copy(origin);
        floor->receiveShadow = true;
        scene->add(floor);

        auto casterMat = MeshStandardMaterial::create();
        casterMat->color = Color(1, 1, 1);
        auto caster = Mesh::create(SphereGeometry::create(1.f, 32, 24), casterMat);
        caster->position.copy(origin).add(Vector3(0, 2.f, 0));
        caster->castShadow = true;
        scene->add(caster);

        auto light = PointLight::create(0xffffff, 30.f);
        light->position.copy(origin).add(Vector3(0, 6.f, 0));
        light->castShadow = true;
        scene->add(light);

        scene->add(AmbientLight::create(0x303030));

        auto camera = PerspectiveCamera::create(50, 1.f, 0.1f, 2000.f);
        camera->position.copy(origin).add(Vector3(0, 5.f, 8.f));
        camera->lookAt(origin);

        GLRenderer renderer(glCanvas());
        renderer.shadowMap().enabled = true;
        renderer.setClearColor(Color(0x000000));
        renderer.render(*scene, *camera);

        const auto px = renderer.readRGBPixels();

        // Count mid-dark pixels: darker than the lit floor, brighter than the
        // black background.
        int shadowed = 0;
        for (int i = 0; i < PIXEL_COUNT; ++i) {
            const int b = px[i * 3] + px[i * 3 + 1] + px[i * 3 + 2];
            if (b > 12 && b < 150) ++shadowed;
        }
        return static_cast<double>(shadowed) / PIXEL_COUNT;
    }

}// namespace

TEST_CASE("a point light's shadow is the same wherever the rig sits in the world") {

    const double atOrigin = shadowFraction({0, 0, 0});
    const double faraway = shadowFraction({400.f, 0, 400.f});

    INFO("shadowed fraction at the origin " << atOrigin << ", translated " << faraway);

    // The rig casts a shadow at all — otherwise the comparison below is vacuous.
    REQUIRE(atOrigin > 0.02);

    // And translating the whole scene does not change it.
    CHECK(std::abs(faraway - atOrigin) < 0.02);
}
