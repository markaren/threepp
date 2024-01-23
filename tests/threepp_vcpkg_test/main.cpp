
#include "threepp/threepp.hpp"

#ifdef EMSCRIPTEN
#include <emscripten.h>
#endif

using namespace threepp;

namespace {

    auto createBox(const Vector3& pos, const Color& color) {
        auto geometry = BoxGeometry::create();
        auto material = MeshBasicMaterial::create();
        material->color.copy(color);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.copy(pos);

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("threepp demo", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create();
    camera->position.z = 5;

    OrbitControls controls{*camera, canvas};

    auto scene = Scene::create();

    auto group = Group::create();
    group->add(createBox({-1, 0, 0}, Color::green));
    group->add(createBox({1, 0, 0}, Color::blue));
    scene->add(group);

    TextRenderer textRenderer;
    auto& textHandle = textRenderer.createHandle("Hello World");
    textHandle.verticalAlignment = threepp::TextHandle::VerticalAlignment::BOTTOM;
    textHandle.setPosition(0, canvas.size().height);
    textHandle.scale = 2;

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        textHandle.setPosition(0, size.height);
    });

    Clock clock;
    float rotationSpeed = 1;
    auto loop = [&] {
        auto dt = clock.getDelta();
        group->rotation.y += rotationSpeed * dt;

        renderer.render(*scene, *camera);

        renderer.resetState();// needed when using TextRenderer
        textRenderer.render();

    };

#ifdef EMSCRIPTEN
    emscripten_set_main_loop(&loop, 0, 1);
#else
    canvas.animateOnce(loop);
#endif
}
