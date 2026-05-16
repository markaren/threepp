
#include "threepp/extras/SpriteInteractor.hpp"
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;
using namespace std::string_literals;

namespace {

    // Helper: configure a corner-anchored 75×75 screen-space sprite that
    // logs to stdout when clicked. `anchor` and `pivot` should be the same
    // corner — e.g. (0, 0) for the bottom-left, (1, 1) for the top-right —
    // so the sprite's anchor point sits flush against the viewport corner.
    std::shared_ptr<Sprite> makeHudSprite(
            const std::shared_ptr<SpriteMaterial>& mat,
            Vector2 anchor, Vector2 pivot, int id) {
        auto sprite = Sprite::create(mat);
        sprite->center.copy(pivot);
        sprite->scale.set(75, 75, 1);
        sprite->screenSpace = true;
        sprite->screenAnchor.copy(anchor);
        sprite->position.set(0, 0, 0);
        sprite->onMouseUp = [id](int) {
            std::cout << "Clicked on sprite " << id << std::endl;
        };
        return sprite;
    }

    void addHudSprites(Scene& scene) {
        TextureLoader tl;
        auto hudMaterial = SpriteMaterial::create();
        hudMaterial->map = tl.load(std::string(DATA_FOLDER) + "/textures/sprite0.png", ColorSpace::sRGB);

        scene.add(makeHudSprite(hudMaterial, {0.f, 1.f}, {0.f, 1.f}, 1));// top-left
        scene.add(makeHudSprite(hudMaterial, {1.f, 1.f}, {1.f, 1.f}, 2));// top-right
        scene.add(makeHudSprite(hudMaterial, {0.f, 0.f}, {0.f, 0.f}, 3));// bottom-left
        scene.add(makeHudSprite(hudMaterial, {1.f, 0.f}, {1.f, 0.f}, 4));// bottom-right
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

    Canvas canvas{"Sprite", {{"aa", 4}, {"favicon", std::string(DATA_FOLDER) + "/textures/three.png"s}}};
    auto size = canvas.size();
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(75, size.aspect(), 0.1f, 1000);
    camera->position.z = 8;

    OrbitControls controls{*camera, canvas};

    TextureLoader loader;
    auto material = SpriteMaterial::create();
    material->map = loader.load(std::string(DATA_FOLDER) + "/textures/three.png", ColorSpace::sRGB);

    auto pickMaterial = material->clone<SpriteMaterial>();

    auto sprites = createSprites(material);
    scene->add(sprites);

    auto helper = Mesh::create(SphereGeometry::create(0.1));
    scene->add(helper);

    addHudSprites(*scene);
    SpriteInteractor spriteInteractor(canvas, *scene);

    canvas.onWindowResize([&](WindowSize newSize) {
        camera->aspect = newSize.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(newSize);
    });

    Vector2 mouse{-Infinity<float>, -Infinity<float>};
    MouseMoveListener l([&](auto& pos) {
        const auto size = canvas.size();
        mouse.x = (pos.x / static_cast<float>(size.width())) * 2 - 1;
        mouse.y = -(pos.y / static_cast<float>(size.height())) * 2 + 1;
    });
    canvas.addMouseListener(l);

    Clock clock;
    Raycaster raycaster;
    Sprite* lastPicked = nullptr;
    canvas.animate([&] {
        if (lastPicked) {
            lastPicked->setMaterial(material);
            lastPicked->scale.set(1, 1, 1);
        }

        helper->visible = false;
        material->rotation += 1 * clock.getDelta();

        raycaster.setFromCamera(mouse, *camera);
        const auto intersects = raycaster.intersectObjects(sprites->children, true);
        if (!intersects.empty()) {
            const auto& intersection = intersects.front();
            helper->position.copy(intersection.point);
            helper->visible = true;

            lastPicked = intersection.object->as<Sprite>();
            lastPicked->setMaterial(pickMaterial);
            lastPicked->scale.set(1.2, 1.2, 1.2);
        }

        renderer->render(*scene, *camera);
    });
}
