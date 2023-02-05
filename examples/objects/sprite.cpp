
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas{Canvas::Parameters().antialiasing(8)};
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 8;

    OrbitControls controls{camera, canvas};

    TextureLoader loader;

    auto material = SpriteMaterial::create();
    material->map = loader.loadTexture("favicon.png");
    material->map->offset.set(0.5, 0.5);

    for (int x = -4; x <= 4; x++) {
        for (int y = -4; y <= 4; y++) {
            auto sprite = Sprite::create(material);
            sprite->position.x = static_cast<float>(x);
            sprite->position.y = static_cast<float>(y);
            scene->add(sprite);
        }
    }

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {

        material->rotation += 1 * dt;

        renderer.render(scene, camera);
    });
}