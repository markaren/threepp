
#include "threepp/controls/DragControls.hpp"
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

int main() {

    Canvas canvas("Drag controls");
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    PerspectiveCamera camera(60, canvas.aspect());
    camera.position.z = 25;

    Scene scene;
    scene.background = Color(0xf0f0f0);

    scene.add(AmbientLight::create(0xaaaaaa));

    auto light = SpotLight::create(0xffffff, 1.f);
    light->position.set(0, 25, 50);
    light->angle = math::PI / 9;

    light->castShadow = true;
    light->shadow->camera->nearPlane = 10;
    light->shadow->camera->farPlane = 100;
    light->shadow->mapSize.x = 1024;
    light->shadow->mapSize.y = 1024;

    scene.add(light);

    auto group = Group::create();
    scene.add(group);

    auto geometry = BoxGeometry::create();

    std::vector<Object3D*> objects;
    for (unsigned i = 0; i < 200; i++) {

        auto object = Mesh::create(geometry, MeshLambertMaterial::create({{"color", Color().randomize()}}));

        object->position.x = math::randFloat() * 30 - 15;
        object->position.y = math::randFloat() * 15 - 7.5f;
        object->position.z = math::randFloat() * 20 - 10;

        object->rotation.x = math::randFloat() * 2 * math::PI;
        object->rotation.y = math::randFloat() * 2 * math::PI;
        object->rotation.z = math::randFloat() * 2 * math::PI;

        object->scale.x = math::randFloat() * 2 + 1;
        object->scale.y = math::randFloat() * 2 + 1;
        object->scale.z = math::randFloat() * 2 + 1;

        object->castShadow = true;
        object->receiveShadow = true;

        scene.add(object);

        objects.emplace_back(object.get());
    }

    DragControls controls(objects, camera, canvas);
    controls.rotateSpeed = 2;

    struct HoverListener: public EventListener {
        void onEvent(Event& event) override {

            auto target = static_cast<Object3D*>(event.target);
            auto& color = target->material()->as<MaterialWithColor>()->color;

            if (event.type == "hoveron") {
                prevColor = color;
                color = Color::white;
            } else if (event.type == "hoveroff") {
                color = prevColor;
            }
        }

    private:
        Color prevColor;

    } hoverListener;

    controls.addEventListener("hoveron", hoverListener);
    controls.addEventListener("hoveroff", hoverListener);

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
        if (evt.key == Key::M) {
            if (controls.mode == DragControls::Mode::Translate) {
                controls.mode = DragControls::Mode::Rotate;
            } else {
                controls.mode = DragControls::Mode::Translate;
            }
        }
    });
    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();

        renderer.setSize(size);
    });

    std::cout << "Press 'm' to switch between translate and rotate mode" << std::endl;

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}
