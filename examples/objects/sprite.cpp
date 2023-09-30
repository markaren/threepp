
#include "threepp/threepp.hpp"

using namespace threepp;
using namespace std::string_literals;

int main() {

    Canvas canvas{"Sprite", {{"aa", 4}, {"favicon", "data/textures/three.png"s}}};
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 8;

    OrbitControls controls{*camera, canvas};

    TextureLoader loader;
    auto material = SpriteMaterial::create();
    material->map = loader.load("data/textures/three.png");
    material->map->offset.set(0.5, 0.5);

    auto sprites = Group::create();
    for (int x = -4; x <= 4; x++) {
        for (int y = -4; y <= 4; y++) {
            auto sprite = Sprite::create(material);
            sprite->position.x = static_cast<float>(x);
            sprite->position.y = static_cast<float>(y);
            sprites->add(sprite);
        }
    }
    scene->add(sprites);

    auto helper = Mesh::create(SphereGeometry::create(0.1));
    scene->add(helper);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Vector2 mouse{-Infinity<float>, -Infinity<float>};
    MouseMoveListener l([&](auto &pos) {
        auto size = canvas.size();
        mouse.x = (pos.x / static_cast<float>(size.width)) * 2 - 1;
        mouse.y = -(pos.y / static_cast<float>(size.height)) * 2 + 1;
    });
    canvas.addMouseListener(&l);

    Clock clock;
    Raycaster raycaster;
    canvas.animate([&]() {
        helper->visible = false;
        material->rotation += 1 * clock.getDelta();

        raycaster.setFromCamera(mouse, *camera);
        auto intersects = raycaster.intersectObject(*sprites, true);
        if (!intersects.empty()) {
            auto &i = intersects.front();
            helper->position.copy(i.point);
            helper->visible = true;
        }

        renderer.render(*scene, *camera);
    });
}
