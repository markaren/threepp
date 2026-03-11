
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("TextSprite");
    GLRenderer renderer(canvas.size());

    Scene scene;
    PerspectiveCamera camera(75, canvas.aspect(), 0.1f, 100);
    camera.position.set(0, 0, 10);

    FontLoader fontLoader;
    auto font = fontLoader.load(std::string(DATA_FOLDER) + "/fonts/typeface/gentilis_regular.typeface.json");

    auto text = TextSprite::create(*font);
    text->setColor(Color::white);
    text->setText("Hello world");
    text->position.set(0, 0, 0);

    scene.add(text);

    auto grid = AxesHelper::create(1);
    scene.add(grid);

    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({});
        ImGui::SetNextWindowSize({});

        ImGui::Begin("Controls");
        //color
        if (ImGui::Button("Random color")) {
            text->setColor(Color().randomize());
        }

        //input text
        ImGui::SameLine();
        static char buf[128] = "Hello world!";
        if (ImGui::InputText("Text", buf, 128)) {
            text->setText(buf);
        }

        //horizontal alignment
        ImGui::Text("Horizontal alignment:");
        ImGui::SameLine();
        if (ImGui::Button("Align left")) {
            text->setHorizontalAlignment(TextSprite::HorizontalAlignment::Left);
        }
        ImGui::SameLine();
        if (ImGui::Button("Align center##h")) {
            text->setHorizontalAlignment(TextSprite::HorizontalAlignment::Center);
        }
        ImGui::SameLine();
        if (ImGui::Button("Align right")) {
            text->setHorizontalAlignment(TextSprite::HorizontalAlignment::Right);
        }

        // vertical alignment
        ImGui::Text("Vertical alignment:");
        ImGui::SameLine();
        if (ImGui::Button("Align above")) {
            text->setVerticalAlignment(TextSprite::VerticalAlignment::Above);
        }
        ImGui::SameLine();
        if (ImGui::Button("Align center##v")) {
            text->setVerticalAlignment(TextSprite::VerticalAlignment::Center);
        }
        ImGui::SameLine();
        if (ImGui::Button("Align below")) {
            text->setVerticalAlignment(TextSprite::VerticalAlignment::Below);
        }

        // scale
        static float scale = 1.f;
        if (ImGui::SliderFloat("Scale", &scale, 0.1f, 5.f)) {
            text->setWorldScale(scale);
        }

        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize newSize) {
        renderer.setSize(newSize);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    canvas.animate([&] {

        renderer.render(scene, camera);
        ui.render();
    });
}
