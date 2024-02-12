
#include "threepp/threepp.hpp"

#include "threepp/audio/Audio.hpp"

#if HAS_IMGUI
#include "threepp/extras/imgui/ImguiContext.hpp"
#include <array>
#endif


using namespace threepp;

namespace {

    auto createSmiley() {

        Shape smileyShape;
        smileyShape.moveTo(80, 40)
                .absarc(40, 40, 40, 0, math::PI * 2, false);

        auto smileyEye1Path = std::make_shared<Path>();
        smileyEye1Path->moveTo(35, 20)
                .absellipse(25, 20, 10, 10, 0, math::PI * 2, true);

        auto smileyEye2Path = std::make_shared<Path>();
        smileyEye2Path->moveTo(65, 20)
                .absarc(55, 20, 10, 0, math::PI * 2, true);

        auto smileyMouthPath = std::make_shared<Path>();
        smileyMouthPath->moveTo(20, 40)
                .quadraticCurveTo(40, 60, 60, 40)
                .bezierCurveTo(70, 45, 70, 50, 60, 60)
                .quadraticCurveTo(40, 80, 20, 60)
                .quadraticCurveTo(5, 50, 20, 40);

        smileyShape.holes.emplace_back(smileyEye1Path);
        smileyShape.holes.emplace_back(smileyEye2Path);
        smileyShape.holes.emplace_back(smileyMouthPath);

        auto shapeGeometry = ShapeGeometry::create(smileyShape);
        shapeGeometry->center();
        shapeGeometry->scale(0.02f, 0.02f, 0.02f);

        auto material = MeshBasicMaterial::create({{"color", Color::orange},
                                                   {"side", Side::Double}});

        auto mesh = Mesh::create(shapeGeometry, material);
        mesh->rotateZ(math::PI);

        return mesh;
    }

}// namespace

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

    auto audioNode = createSmiley();
    audioNode->add(audio);
    scene.add(audioNode);

    camera.add(listener);

    OrbitControls controls{camera, canvas};

#if HAS_IMGUI
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
#endif

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);

#if HAS_IMGUI
        ui.render();
#endif
    });
}
