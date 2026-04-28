
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("glTF Material Variants", {{"antialiasing", 4}});
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(45, canvas.aspect(), 0.01f, 100.f);
    camera->position.set(0, 0.5f, 1.f);

    auto ambientLight = AmbientLight::create(0xffffff, 0.6f);
    scene->add(ambientLight);

    auto dirLight = DirectionalLight::create(0xffffff, 1.5f);
    dirLight->position.set(1, 2, 1);
    scene->add(dirLight);

    GLTFLoader loader;
    auto result = loader.load(std::string(DATA_FOLDER) +
                              "/models/gltf/MaterialsVariantsShoe/MaterialsVariantsShoe.glb");
    if (!result) return 1;

    scene->add(result->scene);


    OrbitControls controls{*camera, canvas};
    controls.target.copy(result->scene->position);
    controls.update();

    // Grab variant info (names are empty if extension absent)
    auto& variants = result->variants;
    int selectedVariant = -1;// -1 = default (no variant applied)

    // -------------------------------------------------------------------------
    //  ImGui panel
    // -------------------------------------------------------------------------
    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({220 * ui.dpiScale(), 0}, ImGuiCond_Always);
        ImGui::Begin("Material Variants", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        if (variants.empty()) {
            ImGui::TextDisabled("No variants in this file.");
        } else {
            ImGui::Text("Variants:");
            ImGui::Spacing();

            // "Default" button
            bool isDefault = selectedVariant == -1;
            if (isDefault) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
            if (ImGui::Button("Default", {-1, 0})) {
                if (!isDefault) {
                    variants.reset(*result->scene);
                    selectedVariant = -1;
                }
            }
            if (isDefault) ImGui::PopStyleColor();

            ImGui::Spacing();

            for (int i = 0; i < static_cast<int>(variants.names.size()); ++i) {
                bool isSelected = selectedVariant == i;
                if (isSelected) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
                if (ImGui::Button(variants.names[i].c_str(), {-1, 0})) {
                    if (!isSelected) {
                        variants.apply(variants.names[i], *result->scene);
                        selectedVariant = i;
                    }
                }
                if (isSelected) ImGui::PopStyleColor();
            }
        }

        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&capture);

    canvas.onWindowResize([&](WindowSize size) {
        renderer->setSize(size);
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
    });

    canvas.animate([&] {
        renderer->render(*scene, *camera);
        ui.render();
    });
}
