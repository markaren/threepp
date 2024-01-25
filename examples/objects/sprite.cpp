
#include "threepp/threepp.hpp"

using namespace threepp;
using namespace std::string_literals;

namespace {

    auto createHudSprites() {
        TextureLoader tl;
        auto hudMaterial = SpriteMaterial::create();
        hudMaterial->map = tl.load("data/textures/sprite0.png");
        hudMaterial->map->offset.set(0.5, 0.5);

        auto hudSprite1 = Sprite::create(hudMaterial);
        hudSprite1->center.set(0, 1);
        hudSprite1->scale.set(0.1, 0.1, 1);

        auto hudSprite2 = Sprite::create(hudMaterial);
        hudSprite2->center.set(1, 1);
        hudSprite2->scale.set(0.1, 0.1, 1);

        auto hudSprite3 = Sprite::create(hudMaterial);
        hudSprite3->center.set(0, 0);
        hudSprite3->scale.set(0.1, 0.1, 1);

        auto hudSprite4 = Sprite::create(hudMaterial);
        hudSprite4->center.set(1, 0);
        hudSprite4->scale.set(0.1, 0.1, 1);

        hudSprite1->position.set(0, 1, 0);// top left
        hudSprite2->position.set(1, 1, 0);// top right
        hudSprite3->position.set(0, 0, 0);// bottom left
        hudSprite4->position.set(1, 0, 0);// bottom right

        auto group = Group::create();
        group->add(hudSprite1);
        group->add(hudSprite2);
        group->add(hudSprite3);
        group->add(hudSprite4);

        return group;
    }

    auto createSprites(const std::shared_ptr<SpriteMaterial>& material) {
        auto sprites = Group::create();
        for (int x = -4; x <= 4; x++) {
            for (int y = -4; y <= 4; y++) {
                auto sprite = Sprite::create(material);
                sprite->position.x = static_cast<float>(x);
                sprite->position.y = static_cast<float>(y);
                sprites->add(sprite);
            }
        }
        return sprites;
    }

}// namespace

int main() {

    Canvas canvas{"Sprite", {{"aa", 4}, {"favicon", "data/textures/three.png"s}}};
    auto size = canvas.size();
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false;
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, size.aspect(), 0.1f, 1000);
    camera->position.z = 8;

    OrbitControls controls{*camera, canvas};

    TextureLoader loader;
    auto material = SpriteMaterial::create();
    material->map = loader.load("data/textures/three.png");
    material->map->offset.set(0.5, 0.5);

    auto pickMaterial = material->clone()->as<SpriteMaterial>();

    auto sprites = createSprites(material);
    scene->add(sprites);

    auto helper = Mesh::create(SphereGeometry::create(0.1));
    scene->add(helper);

    HUD hud;
    hud.add(createHudSprites());

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Vector2 mouse{-Infinity<float>, -Infinity<float>};
    MouseMoveListener l([&](auto& pos) {
        auto size = canvas.size();
        mouse.x = (pos.x / static_cast<float>(size.width)) * 2 - 1;
        mouse.y = -(pos.y / static_cast<float>(size.height)) * 2 + 1;
    });
    canvas.addMouseListener(l);

    Clock clock;
    Raycaster raycaster;
    Sprite* lastPicked = nullptr;
    canvas.animate([&]() {
        if (lastPicked) {
            lastPicked->material = material;
            lastPicked->scale.set(1, 1, 1);
        }

        helper->visible = false;
        material->rotation += 1 * clock.getDelta();

        raycaster.setFromCamera(mouse, *camera);
        auto intersects = raycaster.intersectObjects(sprites->children, true);
        if (!intersects.empty()) {
            auto& intersection = intersects.front();
            helper->position.copy(intersection.point);
            helper->visible = true;

            lastPicked = intersection.object->as<Sprite>();
            lastPicked->material = pickMaterial;
            lastPicked->scale.set(1.2, 1.2, 1.2);
        }

        renderer.clear();
        renderer.render(*scene, *camera);
        hud.apply(renderer);
    });
}
