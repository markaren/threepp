
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("HDR Environment Map");
    GLRenderer renderer(canvas.size());
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;

    auto scene = Scene::create();

    PerspectiveCamera camera(45, canvas.aspect(), 0.1f, 1000);
    camera.position.set(0, 0, 5);

    RGBELoader hdrLoader;

    if (auto hdrTexture = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr")) {
        scene->background = hdrTexture;
        scene->environment = hdrTexture;
    }

    // Metallic sphere — IBL from the HDR environment
    auto sphereGeo = SphereGeometry::create(1.0f, 64, 32);
    auto sphereMat = MeshStandardMaterial::create();
    sphereMat->metalness = 1.0f;
    sphereMat->roughness = 0.0f;
    auto sphere = Mesh::create(sphereGeo, sphereMat);
    scene->add(sphere);

    OrbitControls controls{camera, canvas};

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(*scene, camera);
    });
}
