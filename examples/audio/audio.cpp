
#include "threepp/canvas/Canvas.hpp"
#include "audio/Audio.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

using namespace threepp;

int main()
{
    Canvas canvas;

    Audio audio("data/sounds/376737_Skullbeatz___Bad_Cat_Maste.mp3");
    audio.setLooping(true);
    audio.play();

    bool play = audio.isPlaying();
    float volume = audio.getMasterVolume();
    ImguiFunctionalContext ui(canvas.windowPtr(), [&]{
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("Audio settings");
        ImGui::SliderFloat("Volume", &volume, 0.f, 1.f);
        if (ImGui::IsItemEdited()) {
            audio.setMasterVolume(volume);
        }
        ImGui::Checkbox("Play", &play);
        if (ImGui::IsItemEdited()) {
            audio.togglePlay();
        }
        ImGui::End();
    });

    canvas.animate([&] {

        ui.render();
    });

}
