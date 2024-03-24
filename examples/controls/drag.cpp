
#include "threepp/controls/DragControls.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Drag controls");
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = threepp::ShadowMap::PFC;

    PerspectiveCamera camera(60, canvas.aspect());
    camera.position.z = 25;

    Scene scene;
    scene.background = Color(0xf0f0f0);

    scene.add(AmbientLight::create(0xaaaaaa));

    auto light = SpotLight::create(0xffffff, 1.f);
    light->position.set(0, 25, 50);
    light->angle = math::PI / 9;

    light->castShadow = true;
    light->shadow->camera->near = 10;
    light->shadow->camera->far = 100;
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

    struct HoverListener {
        void hoverOn(Event& event) {

            auto target = static_cast<Object3D*>(event.target);
            auto& color = target->material()->as<MaterialWithColor>()->color;

            prevColor = color;
            color = Color::white;
        }

        void hoverOff(Event& event) {
            auto target = static_cast<Object3D*>(event.target);
            auto& color = target->material()->as<MaterialWithColor>()->color;

            color = prevColor;
        }

    private:
        Color prevColor;

    } hoverListener;

    controls.HoverOn.subscribeForever([&hoverListener](auto& evt) {
        hoverListener.hoverOn(evt);
    });
    controls.HoverOff.subscribeForever([&hoverListener](auto& evt) {
        hoverListener.hoverOff(evt);
    });


    canvas.keys.Pressed.subscribeForever([&](KeyEvent evt) {
        if (evt.key == Key::M) {
            if (controls.mode == DragControls::Mode::Translate) {
                controls.mode = DragControls::Mode::Rotate;
            } else {
                controls.mode = DragControls::Mode::Translate;
            }
        }
    });


    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();

        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}
