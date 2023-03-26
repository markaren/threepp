
#include "threepp/textures/DataTexture3D.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(8));
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;

    OrbitControls controls{camera, canvas};

    std::vector<unsigned char> data(64 * 64 * 64);
    int i = 0;
    for (unsigned z = 0; z < 64; z++) {
        for (unsigned y = 0; y < 64; y++) {
            for (unsigned x = 0; x < 64; x++) {
                data[i] = i % 256;
                ++i;
            }
        }
    }

    auto texture = std::make_shared<DataTexture3D>(data, 64, 64, 64);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {

        renderer.render(scene, camera);
    });
}
