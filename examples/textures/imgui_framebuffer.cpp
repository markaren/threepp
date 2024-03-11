
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#pragma warning(disable : 4312) // cast from unsigned int to void*

using namespace threepp;

int main() {

    Canvas canvas("Imgui framebuffer");
    GLRenderer renderer(canvas.size());

    Scene scene;
    PerspectiveCamera camera(60, canvas.aspect());
    camera.position.z = 10;

    auto material = MeshBasicMaterial::create();
    material->color = Color::red;
    material->wireframe = true;
    auto sphere = Mesh::create(SphereGeometry::create(1), material);
    scene.add(sphere);

    unsigned int textureSizeXY = 256;
    auto texture = DataTexture::create(std::vector<unsigned char>(textureSizeXY * textureSizeXY * 3), textureSizeXY, textureSizeXY);
    texture->format = Format::RGB;
    texture->minFilter = Filter::Nearest;
    texture->magFilter = Filter::Nearest;

    OrbitControls controls{camera, canvas};

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    bool imguiOnly = true;
    ImguiFunctionalContext ui(canvas.windowPtr(), [&] {

        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({static_cast<float>(textureSizeXY), static_cast<float>(50 + textureSizeXY)}, 0);

        ImGui::Begin("Imgui frame");
        ImGui::Checkbox("Imgui only", &imguiOnly);

        if (auto textureId = renderer.getGlTextureId(*texture)) {
            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddImage(
                    reinterpret_cast<ImTextureID>(textureId.value()),
                    ImVec2(pos.x, pos.y),
                    ImVec2(pos.x + static_cast<float>(textureSizeXY), pos.y + static_cast<float>(textureSizeXY)),
                    ImVec2(0, 1),
                    ImVec2(1, 0));
        }

        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    Vector2 coords;
    canvas.animate([&] {
        renderer.render(scene, camera);

        const auto size = canvas.size();
        coords.x = (float(size.width) / 2) - (float(textureSizeXY) / 2);
        coords.y = (float(size.height) / 2) - (float(textureSizeXY) / 2);

        renderer.copyFramebufferToTexture({coords.x, coords.y}, *texture);

        if (imguiOnly) renderer.clear();

        ui.render();
    });
}
