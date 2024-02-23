
#include "threepp/loaders/CubeTextureLoader.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto loadWalt() {
        OBJLoader loader;
        auto walt = loader.load("data/models/obj/walt/WaltHead.obj");
        walt->scale *= 15;
        walt->position.y = -500;

        return walt;
    }

}// namespace

int main() {

    Canvas canvas("Cubemap");
    GLRenderer renderer(canvas.size());

    PerspectiveCamera camera(50, canvas.aspect(), 1, 50000);
    camera.position.z = 2000;

    std::filesystem::path path("data/textures/cube/SwedishRoyalCastle");
    std::array<std::filesystem::path, 6> urls{
            // clang-format off
            path / "px.jpg", path / "nx.jpg",
            path / "py.jpg", path / "ny.jpg",
            path / "pz.jpg", path / "nz.jpg"
            // clang-format on
    };

    CubeTextureLoader loader{};
    auto reflectionCube = loader.load(urls);

    Scene scene;
    scene.background = reflectionCube;

    auto walt = loadWalt();
    scene.add(walt);

    OrbitControls controls{camera, canvas};

    //lights
    const auto ambient = AmbientLight::create(0xffffff);
    scene.add(ambient);

    const auto pointLight = PointLight::create(0xffffff, 2.f);
    scene.add(pointLight);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}