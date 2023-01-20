
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

    auto geometry = BoxGeometry::create();
    auto material = MeshBasicMaterial::create();
    material->color = Color::green;
    auto mesh = Mesh::create(geometry, material);

    scene->add(mesh);

#ifdef HAS_IMGUI
    std::array<float, 3> posBuf{};
    imgui_functional_context ui(canvas.window_ptr(), [&]{
        ImGui::SetNextWindowPos({0, 0}, 0, {0,0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("Demo");
        ImGui::SliderFloat3("position", posBuf.data(), -1.f, 1.f);
        controls.enabled = !ImGui::IsWindowHovered();
        ImGui::End();
    });
#endif

    canvas.animate([&]{

        renderer.render(scene, camera);
        ui.render();

#ifdef HAS_IMGUI
        mesh->position.fromArray(posBuf);
#endif

    });

}
