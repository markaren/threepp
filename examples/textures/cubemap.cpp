
#include "threepp/loaders/CubeTextureLoader.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"
#include "threepp/threepp.hpp"
#include <iostream>

using namespace threepp;

int main() {

    Canvas canvas("Cubemap");
    GLRenderer renderer(canvas.size());
    renderer.checkShaderErrors = true;

    PerspectiveCamera camera(50, canvas.aspect(), 0.1, 1000);
    camera.position.z = 10;

    std::filesystem::path path("data/textures/cube/Bridge2");
    std::array<std::filesystem::path, 6> urls{
            // clang-format off
            path / "posx.jpg", path / "negx.jpg",
            path / "posy.jpg", path / "negy.jpg",
            path / "posz.jpg", path / "negz.jpg"
            // clang-format on
    };

    CubeTextureLoader loader{};
    auto reflectionCube = loader.load(urls);

    Scene scene;
    scene.background = reflectionCube;

    auto material = MeshLambertMaterial::create();
    material->envMap = reflectionCube;
    auto mesh = Mesh::create(SphereGeometry::create(0.5), material);
    scene.add(mesh);

    OrbitControls controls{camera, canvas};

    //lights
    const auto ambient = AmbientLight::create(0xffffff);
    scene.add(ambient);

    const auto directionalLight = DirectionalLight::create(0xffffff);
    directionalLight->position.set(1, 1, 1);
    scene.add(directionalLight);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}
