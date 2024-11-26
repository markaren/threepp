
#include "threepp/cameras/PerspectiveCamera.hpp"


#include <catch2/catch_test_macros.hpp>

#include <threepp/canvas/Canvas.hpp>
#include <threepp/renderers/GLRenderer.hpp>
#include <threepp/scenes/Scene.hpp>

using namespace threepp;

TEST_CASE("Headless Tests", "[canvas]") {

    Canvas canvas("Headless window", {{"headless", true}, {"size", WindowSize(800, 800)}});
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::aliceblue;
    PerspectiveCamera camera(60, renderer.size().aspect(), 0.1f, 100.0f);

    renderer.render(scene, camera);

    std::string testFile = "test.jpg";
    renderer.writeFramebuffer(testFile);

    REQUIRE(std::filesystem::exists(testFile));
}
