
#include "threepp/threepp.hpp"

#include <threepp/extras/imgui/ImguiContext.hpp>

using namespace threepp;

int main() {

    Canvas canvas("Points", {{"aa", 8}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    scene->background = 0x050505;
    scene->fog = Fog(0x050505, 2000, 3500);
    auto camera = PerspectiveCamera::create(27, canvas.aspect(), 5, 3500);
    camera->position.z = 2750;

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    constexpr std::pair minMaxParticles = {1000, 500000};

    int numParticles = minMaxParticles.second;
    std::vector<float> positions(numParticles * 3);
    std::vector<float> colors(numParticles * 3);

    constexpr float n = 1000;
    constexpr float n2 = n / 2;

    for (int i = 0; i < numParticles; i += 3) {
        positions[i] = (math::randFloat() * n - n2);
        positions[i + 1] = (math::randFloat() * n - n2);
        positions[i + 2] = (math::randFloat() * n - n2);

        colors[i] = ((positions[i] / n) + 0.5f);
        colors[i + 1] = ((positions[i + 1] / n) + 0.5f);
        colors[i + 2] = ((positions[i + 2] / n) + 0.5f);
    }

    std::vector<float> positions_clone = positions;
    std::vector<float> colors_clone = colors;

    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    geometry->computeBoundingSphere();

    const auto material = PointsMaterial::create();
    material->size = 2;
    material->vertexColors = true;

    const auto points = Points::create(geometry, material);
    scene->add(points);

    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({});
        ImGui::SetNextWindowSize({}, {});

        ImGui::Begin("Settings");
        if (ImGui::SliderInt("Num points", &numParticles, minMaxParticles.first, minMaxParticles.second)) {
            auto& pos = geometry->getAttribute<float>("position")->array();
            for (int i = 0; i < numParticles; i += 3) {
                pos[i] = positions_clone[i];
                pos[i + 1] = positions_clone[i + 1];
                pos[i + 2] = positions_clone[i + 2];
            }
            for (int i = numParticles; i < minMaxParticles.second; i += 3) {
                pos[i] = positions_clone[0];
                pos[i + 1] = positions_clone[0];
                pos[i + 2] = positions_clone[0];
            }

            geometry->getAttribute<float>("position")->needsUpdate();
        }
        ImGui::End();
    });

    Clock clock;
    canvas.animate([&]() {
        const auto t = clock.getElapsedTime();

        points->rotation.x = t * 0.25f;
        points->rotation.y = t * 0.5f;

        renderer.render(*scene, *camera);

        ui.render();
    });
}
