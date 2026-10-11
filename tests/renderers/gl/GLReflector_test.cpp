// The Reflector draws its mirror image with a camera of its own per viewing
// camera: a clone of it, made the first time that camera looks at the mirror
// (three.js' getReflectionCamera). So the mirror shows what the viewer could
// see, layers included, and a scene that wants something kept out of the
// mirror takes it off the clone's layers instead of hiding it around every
// render.

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Reflector.hpp"

#include <functional>

namespace {

    constexpr unsigned int kPlateLayer = 3;

    struct MirrorScene {

        std::shared_ptr<Scene> scene = Scene::create();
        std::shared_ptr<Reflector> mirror;
        PerspectiveCamera camera{50, 1, 0.1f, 100};

        // A grey plate behind the camera on kPlateLayer, which the camera looks
        // at, and a neutral mirror in front of it.
        MirrorScene() {

            scene->background = Color(0, 0, 0);

            auto plateMaterial = MeshBasicMaterial::create();
            plateMaterial->color = Color(0.5f, 0.5f, 0.5f);
            plateMaterial->side = Side::Double;
            auto plate = Mesh::create(PlaneGeometry::create(80, 80), plateMaterial);
            plate->position.z = 5;
            plate->layers.set(kPlateLayer);
            scene->add(plate);

            Reflector::Options options;
            options.color = Color(0.5f, 0.5f, 0.5f);// blendOverlay with 0.5 is the identity
            options.textureWidth = RT_WIDTH;
            options.textureHeight = RT_HEIGHT;
            mirror = Reflector::create(PlaneGeometry::create(60, 60), options);
            mirror->position.z = -5;
            scene->add(mirror);

            camera.layers.enable(kPlateLayer);
            camera.lookAt(Vector3(0, 0, -1));
        }

        AvgColor mirrorCentre(GLRenderer& renderer) {

            renderer.render(*scene, camera);
            renderer.render(*scene, camera);
            return centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);
        }
    };

    bool isBlack(const AvgColor& px) { return px.r < 3 && px.g < 3 && px.b < 3; }
    bool isGrey(const AvgColor& px) { return px.r > 100 && px.g > 100 && px.b > 100; }// 0.5 linear is ~188 sRGB

}// namespace

TEST_CASE("Reflector: the mirror shows what the viewing camera can see", "[objects][layers]") {

    MirrorScene s;
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0, 0, 0));

    // Without a word to the Reflector, the plate on the viewer's extra layer is in the mirror.
    const auto inherited = s.mirrorCentre(renderer);
    INFO("inherited " << inherited.r << "," << inherited.g << "," << inherited.b);
    CHECK(isGrey(inherited));

    // The reflection camera is the viewer's own: taking the layer off it empties the
    // mirror while the viewer still sees the plate directly.
    auto& reflectionCamera = s.mirror->getReflectionCamera(s.camera);
    CHECK(&reflectionCamera == &s.mirror->getReflectionCamera(s.camera));
    CHECK(&reflectionCamera != static_cast<Camera*>(&s.camera));
    reflectionCamera.layers.disable(kPlateLayer);

    const auto hidden = s.mirrorCentre(renderer);
    INFO("hidden " << hidden.r << "," << hidden.g << "," << hidden.b);
    CHECK(isBlack(hidden));

    s.camera.lookAt(Vector3(0, 0, 1));
    renderer.render(*s.scene, s.camera);
    const auto direct = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);
    INFO("direct " << direct.r << "," << direct.g << "," << direct.b);
    CHECK(isGrey(direct));
}

TEST_CASE("Reflector: a second viewing camera gets a reflection camera of its own", "[objects][layers]") {

    MirrorScene s;
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0, 0, 0));

    // A viewer that does not look at the plate's layer: its mirror image has no plate,
    // and the first viewer's mirror image keeps it.
    PerspectiveCamera other(50, 1, 0.1f, 100);
    other.lookAt(Vector3(0, 0, -1));
    renderer.render(*s.scene, other);
    renderer.render(*s.scene, other);
    const auto otherSees = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);
    INFO("other viewer " << otherSees.r << "," << otherSees.g << "," << otherSees.b);
    CHECK(isBlack(otherSees));

    const auto firstSees = s.mirrorCentre(renderer);
    INFO("first viewer " << firstSees.r << "," << firstSees.g << "," << firstSees.b);
    CHECK(isGrey(firstSees));
    CHECK(&s.mirror->getReflectionCamera(other) != &s.mirror->getReflectionCamera(s.camera));
}
