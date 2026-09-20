// Adding fog to a scene that had none, and taking it away again.
//
// USE_FOG is a compile-time define, so the program has to be rebuilt when a
// scene's fog appears or disappears. The guard that decides this required BOTH
// the scene's current fog AND the fog the program was built with to be present:
//
//   else if (fog && material->fog && materialProperties->fog && !(fog == ...))
//
// which is exactly the two transitions it cannot see. Starting fogless and
// assigning Scene::fog left USE_FOG undefined, so the fog never appeared; starting
// with fog and clearing it left the old program in place, so the fog never went
// away. r129 compares `materialProperties.fog !== fog`, where either side may be
// null, and so catches both.
//
// The same renderer and the same material instance are reused across the renders
// on purpose: these are cached per material, and building a fresh renderer for
// each frame is what hides the bug.

#include "gl_test_helpers.hpp"

#include "threepp/scenes/Fog.hpp"

namespace {

    // A plane filling the frame, well beyond the fog's far distance, so a fogged
    // render reads essentially the fog colour and an unfogged one reads the
    // material colour.
    struct Fixture {
        std::shared_ptr<Scene> scene;
        std::shared_ptr<MeshBasicMaterial> material;
        std::shared_ptr<PerspectiveCamera> camera;
    };

    Fixture makeFixture() {
        Fixture f;
        f.scene = Scene::create();
        f.material = MeshBasicMaterial::create();
        f.material->color = Color(0x000000);// black, so fog can only brighten
        f.scene->add(Mesh::create(PlaneGeometry::create(400.f, 400.f), f.material));

        f.camera = PerspectiveCamera::create(50, 1.f, 0.1f, 1000.f);
        f.camera->position.set(0, 0, 60.f);
        f.camera->lookAt(Vector3{0, 0, 0});
        return f;
    }

    // Fog colour is white and the plane is black, so "is it fogged" is just "is it
    // bright" — no colour-space subtlety needed.
    Fog whiteFog() {
        return Fog(Color(0xffffff), 1.f, 20.f);
    }

}// namespace

TEST_CASE("fog assigned after the first render takes effect") {

    auto f = makeFixture();
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0x000000));

    renderer.render(*f.scene, *f.camera);
    const double unfogged = avgBrightness(renderer.readRGBPixels());

    f.scene->fog = whiteFog();
    renderer.render(*f.scene, *f.camera);
    const double fogged = avgBrightness(renderer.readRGBPixels());

    INFO("no fog " << unfogged << " -> fog assigned " << fogged);
    REQUIRE(unfogged < 20.0);// the black plane really is black to start with
    CHECK(fogged > 150.0);   // and white fog really does cover it
}

TEST_CASE("fog cleared after the first render stops being applied") {

    auto f = makeFixture();
    f.scene->fog = whiteFog();

    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0x000000));

    renderer.render(*f.scene, *f.camera);
    const double fogged = avgBrightness(renderer.readRGBPixels());

    f.scene->fog = std::nullopt;
    renderer.render(*f.scene, *f.camera);
    const double cleared = avgBrightness(renderer.readRGBPixels());

    INFO("fog " << fogged << " -> fog cleared " << cleared);
    REQUIRE(fogged > 150.0);
    CHECK(cleared < 20.0);
}
