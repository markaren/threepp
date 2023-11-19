
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/threepp.hpp"

#include "audio/Audio.hpp"

#include <array>

using namespace threepp;

int main() {
    Canvas canvas("Audio demo");
    GLRenderer renderer(canvas.size());

    Scene scene;

    PerspectiveCamera camera(50, canvas.aspect());
    camera.position.z = -5;

    AudioListener listener;
    PositionalAudio audio(listener, "data/sounds/376737_Skullbeatz___Bad_Cat_Maste.mp3");
    audio.setLooping(true);
    audio.play();

    auto audioNode = Mesh::create(SphereGeometry::create());
    audioNode->add(audio);
    scene.add(audioNode);

    camera.add(listener);

    OrbitControls controls{camera, canvas};

    std::array<float, 3> audioPos{};
    bool play = audio.isPlaying();
    float volume = listener.getMasterVolume();
    ImguiFunctionalContext ui(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("Audio settings");
        ImGui::SliderFloat("Volume", &volume, 0.f, 1.f);
        if (ImGui::IsItemEdited()) {
            listener.setMasterVolume(volume);
        }
        ImGui::SliderFloat3("Position", audioPos.data(), -5.f, 5.f);
        if (ImGui::IsItemEdited()) {
            audioNode->position.set(audioPos[0], audioPos[1], audioPos[2]);
        }
        ImGui::Checkbox("Play", &play);
        if (ImGui::IsItemEdited()) {
            audio.togglePlay();
        }
        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    canvas.animate([&] {
        renderer.render(scene, camera);

        ui.render();
    });
}
