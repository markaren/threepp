// Vulkan deferred — valley fog with a noise-displaced top.
//
// A procedural ridged valley under a low sun, filled with a height-fog layer
// whose top is displaced by a drifting 3D noise (HeightFogSettings::topNoise),
// so the fog line climbs and falls along the slopes. The displacement reaches
// the surface extinction and the sun's march at any distance; the older
// noiseAmount only reaches the froxels' clustered-light glow.
//
//   vulkan_ridge_fog [--shot name.png] [--frames N] [--time seconds]
//                    [--top-noise metres] [--top-scale metres] [--density sigma]
//                    [--base y] [--falloff metres] [--cam x,y,z] [--look x,y,z]
//
// --time pins the renderer clock, so two captures at the same time are the same
// fog field and two at different times show its motion.

#include "capture_util.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    float hash2(int x, int z) {
        auto h = static_cast<unsigned int>(x) * 374761393u + static_cast<unsigned int>(z) * 668265263u;
        h = (h ^ (h >> 13)) * 1274126177u;
        return static_cast<float>((h ^ (h >> 16)) & 0xffffffu) / static_cast<float>(0xffffff);
    }

    float valueNoise(float x, float z) {
        const float fx = std::floor(x), fz = std::floor(z);
        const int ix = static_cast<int>(fx), iz = static_cast<int>(fz);
        float tx = x - fx, tz = z - fz;
        tx = tx * tx * (3.f - 2.f * tx);
        tz = tz * tz * (3.f - 2.f * tz);
        const float a = hash2(ix, iz), b = hash2(ix + 1, iz);
        const float c = hash2(ix, iz + 1), d = hash2(ix + 1, iz + 1);
        return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
    }

    // Ridged fBm: sharp crests, rounded valleys. A broad trough along z keeps a
    // valley floor in front of the camera for the fog to pool in.
    float terrainHeight(float x, float z) {
        float sum = 0.f, amp = 1.f, freq = 1.f / 260.f, norm = 0.f;
        for (int o = 0; o < 6; ++o) {
            const float n = 1.f - std::abs(valueNoise(x * freq + 13.7f * o, z * freq - 7.1f * o) * 2.f - 1.f);
            sum += n * n * amp;
            norm += amp;
            amp *= 0.5f;
            freq *= 2.03f;
        }
        const float ridge  = sum / norm;
        const float trough = std::clamp(std::abs(x) / 320.f, 0.f, 1.f);
        return -20.f + 170.f * ridge * (0.25f + 0.75f * trough * trough);
    }

    std::shared_ptr<Mesh> makeTerrain() {
        constexpr int   N    = 384;
        constexpr float size = 1800.f;
        std::vector<float> pos, col;
        std::vector<unsigned int> idx;
        pos.reserve((N + 1) * (N + 1) * 3);
        col.reserve((N + 1) * (N + 1) * 3);
        for (int j = 0; j <= N; ++j) {
            for (int i = 0; i <= N; ++i) {
                const float x = (static_cast<float>(i) / N - 0.5f) * size;
                const float z = (static_cast<float>(j) / N - 0.5f) * size;
                const float y = terrainHeight(x, z);
                pos.insert(pos.end(), {x, y, z});
                const float t = std::clamp((y + 20.f) / 150.f, 0.f, 1.f);
                Color c(0.10f, 0.17f, 0.08f);
                c.lerp(Color(0.30f, 0.28f, 0.25f), std::clamp((t - 0.35f) / 0.4f, 0.f, 1.f));
                col.insert(col.end(), {c.r, c.g, c.b});
            }
        }
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                const auto a = static_cast<unsigned int>(j * (N + 1) + i);
                const auto b = a + 1;
                const auto c = a + (N + 1);
                const auto d = c + 1;
                idx.insert(idx.end(), {a, c, b, b, c, d});
            }
        }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
        geo->computeVertexNormals();
        auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                        .roughness(0.95f)
                                                        .metalness(0.f)
                                                        .vertexColors(true));
        return Mesh::create(geo, mat);
    }

}// namespace

int main(int argc, char** argv) {

    const auto args = capture::parseArgs(argc, argv);
    std::string shotPath;
    double pinnedTime = -1.0;
    VulkanRenderer::HeightFogSettings hf;
    // The displacement is kept below the falloff: the noise then modulates a
    // soft gradient. Several falloffs of displacement make the layer nearly
    // binary, and since the noise is read at the surface point the fog then
    // looks painted on the terrain.
    hf.density       = 0.025f;
    hf.baseY         = 12.f;
    hf.falloff       = 11.f;
    hf.noiseAmount   = 0.f;
    hf.topNoise      = 9.f;
    hf.topNoiseScale = 200.f;
    hf.topNoiseSpeed = 4.f;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shot") && i + 1 < argc) shotPath = argv[++i];
        else if (!std::strcmp(argv[i], "--time") && i + 1 < argc) pinnedTime = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--top-noise") && i + 1 < argc) hf.topNoise = static_cast<float>(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--top-scale") && i + 1 < argc) hf.topNoiseScale = static_cast<float>(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--density") && i + 1 < argc) hf.density = static_cast<float>(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--base") && i + 1 < argc) hf.baseY = static_cast<float>(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--falloff") && i + 1 < argc) hf.falloff = static_cast<float>(std::atof(argv[++i]));
    }
    const int shotFrames = args.frames.value_or(90);
    int shotFrame = 0;

    Canvas canvas("Vulkan Deferred - Ridge fog", {{"vsync", false}});
    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 0.75f;
    renderer.setAutoExposure(false);

    Scene scene;
    const Color sky(0.62f, 0.72f, 0.82f);
    renderer.setClearColor(sky);
    scene.background = sky;
    scene.fog = FogExp2(Color(0.82f, 0.87f, 0.91f), 0.03f);// the medium's colour; setHeightFog's density wins

    scene.add(makeTerrain());

    // A low sun and little fill: the slopes facing away stay dark, which is
    // what the lit fog reads against.
    auto sun = DirectionalLight::create(Color(1.0f, 0.82f, 0.62f), 3.6f);
    sun->position.set(-600.f, 150.f, -250.f);
    scene.add(sun);
    scene.add(AmbientLight::create(Color(0.62f, 0.72f, 0.82f), 0.18f));

    renderer.setHeightFog(hf);
    if (pinnedTime >= 0.0) renderer.setSimTime(pinnedTime);

    PerspectiveCamera camera(45.f, canvas.aspect(), 1.f, 6000.f);
    camera.position.copy(args.camPos.value_or(Vector3(-60.f, 85.f, 400.f)));
    OrbitControls controls{camera, canvas};
    controls.target.copy(args.camTarget.value_or(Vector3(0.f, 25.f, 0.f)));
    controls.update();

    std::unique_ptr<RendererSettingsUi> ui;
    if (shotPath.empty()) {
        ui = std::make_unique<RendererSettingsUi>(canvas, renderer, [&] {
            ImGui::TextWrapped("Valley fog. \"Top noise\" displaces the top of the layer with a "
                               "drifting noise; at 0 the layer is flat.");
            ImGui::Separator();
            bool ch = false;
            ch |= ImGui::SliderFloat("Density", &hf.density, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ch |= ImGui::SliderFloat("Base Y (m)", &hf.baseY, -20.f, 120.f, "%.0f");
            ch |= ImGui::SliderFloat("Falloff (m)", &hf.falloff, 1.f, 80.f, "%.1f");
            ch |= ImGui::SliderFloat("Top noise (m)", &hf.topNoise, 0.f, 60.f, "%.1f");
            ch |= ImGui::SliderFloat("Top noise scale (m)", &hf.topNoiseScale, 20.f, 600.f, "%.0f");
            ch |= ImGui::SliderFloat("Top noise speed (m/s)", &hf.topNoiseSpeed, 0.f, 30.f, "%.1f");
            if (ch) renderer.setHeightFog(hf);
        }, "Vulkan Deferred - Ridge fog");
    }

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    canvas.animate([&] {
        controls.update();
        renderer.render(scene, camera);
        if (shotPath.empty()) {
            ui->render();
        } else if (++shotFrame >= shotFrames) {
            capture::finishShot(renderer, shotPath);
        }
    });

    return 0;
}
