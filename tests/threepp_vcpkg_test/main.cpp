
#include "threepp/threepp.hpp"
#include "threepp/extras/imgui/imgui_context.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create();
    camera->position.z = 5;

    OrbitControls controls{camera, canvas};

    auto scene = Scene::create();

    auto group = Group::create();
    scene->add(group);

    {
        auto geometry = BoxGeometry::create();
        auto material = MeshBasicMaterial::create();
        material->color = Color::green;
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = -1;
        group->add(mesh);
    }

    {
        auto geometry = BoxGeometry::create();
        auto material = MeshBasicMaterial::create();
        material->color = Color::blue;
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = 1;
        group->add(mesh);
    }

    renderer.enableTextRendering();
    auto& textHandle = renderer.textHandle("Hello World");
    textHandle.setPosition(0, canvas.getSize().height-30);
    textHandle.scale = 2;


    std::array<float, 3> posBuf{};
    imgui_functional_context ui(canvas.window_ptr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("Demo");
        ImGui::SliderFloat3("position", posBuf.data(), -1.f, 1.f);
        controls.enabled = !ImGui::IsWindowHovered();
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size){
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        textHandle.setPosition(0, size.height-30);
    });

    canvas.animate([&] {

        renderer.render(scene, camera);

        ui.render();
        group->position.fromArray(posBuf);

    });

}
