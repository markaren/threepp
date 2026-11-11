// A flag on a pole, flown by threepp::Cloth (include/threepp/extras/cloth/Cloth.hpp).
//
// The whole use of the solver is the block marked "The flag" below: a
// PlaneGeometry, a ClothMesh over it, the hoist edge pinned, the pole as a
// capsule collider, a wind vector, and one update() per frame.
//
//   cloth_flag                          interactive; orbit, and a panel for the wind
//   cloth_flag --gl | --vulkan          pick the backend without the prompt
//   cloth_flag --shot <file.png>        render --frames frames at a fixed step, write the last, exit
//   cloth_flag --frames N               (default 360)

#include "renderer_factory.hpp"

#include "threepp/extras/cloth/Cloth.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/threepp.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr float kFixedDt = 1.f / 60.f;

    template<class F>
    float millisecondsOf(F&& f) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // Flag proportions 22 : 16 (the Norwegian flag).
    constexpr float kFlagW = 2.2f, kFlagH = 1.6f;
    constexpr int kFlagNx = 33, kFlagNy = 24;

    constexpr float kPoleHeight = 7.f, kPoleRadius = 0.045f;

    // The Norwegian flag: units 6-1-2-1-12 along the fly, 6-1-2-1-6 up the hoist.
    std::shared_ptr<Texture> makeFlagTexture() {
        const int scale = 16, w = 22 * scale, h = 16 * scale;
        const unsigned char red[3] = {186, 12, 47}, white[3] = {255, 255, 255}, blue[3] = {0, 32, 91};
        std::vector<unsigned char> px(static_cast<size_t>(w) * h * 4);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float u = static_cast<float>(x) / scale, v = static_cast<float>(y) / scale;
                const bool blueX = u >= 7.f && u < 9.f, blueY = v >= 7.f && v < 9.f;
                const bool whiteX = u >= 6.f && u < 10.f, whiteY = v >= 6.f && v < 10.f;
                const unsigned char* c = (blueX || blueY) ? blue : (whiteX || whiteY) ? white
                                                                                         : red;
                unsigned char* o = &px[(static_cast<size_t>(y) * w + x) * 4];
                o[0] = c[0];
                o[1] = c[1];
                o[2] = c[2];
                o[3] = 255;
            }
        }
        auto tex = DataTexture::create(ImageData{std::move(px)}, w, h);
        tex->colorSpace = ColorSpace::sRGB;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Linear;
        tex->needsUpdate();
        return tex;
    }

    struct Options {
        std::optional<GraphicsAPI> api;
        std::string shot;
        int frames = 360;
    };

    Options parseOptions(int argc, char** argv) {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--gl") o.api = GraphicsAPI::OpenGL;
            else if (a == "--vulkan")
                o.api = GraphicsAPI::Vulkan;
            else if (a == "--shot" && i + 1 < argc)
                o.shot = argv[++i];
            else if (a == "--frames" && i + 1 < argc)
                o.frames = std::atoi(argv[++i]);
        }
        return o;
    }

}// namespace

int main(int argc, char** argv) {

    const Options options = parseOptions(argc, argv);
    const bool capturing = !options.shot.empty();

    // A capture must not wait on the backend prompt.
    std::optional<GraphicsAPI> api = options.api;
    if (capturing && !api) api = GraphicsAPI::OpenGL;

    Canvas canvas("Cloth flag", {{"vsync", !capturing}, {"aa", 4}});
    auto renderer = createRenderer(canvas, api);
    const Color sky(0.55f, 0.70f, 0.90f);
    renderer->setClearColor(sky);
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 200.f);
    camera.position.set(3.4f, 5.1f, 5.4f);
    const Vector3 target{1.f, 5.9f, 0.f};
    camera.lookAt(target);

    Scene scene;
    scene.background = sky;

    scene.add(HemisphereLight::create(Color(0.75f, 0.85f, 1.f), Color(0.35f, 0.33f, 0.28f), 0.9f));

    auto sun = DirectionalLight::create(Color(1.f, 0.96f, 0.88f), 2.4f);
    sun->position.set(6.f, 14.f, 9.f);
    sun->castShadow = true;
    {
        auto* shadowCam = sun->shadow->camera->as<OrthographicCamera>();
        shadowCam->left = shadowCam->bottom = -8.f;
        shadowCam->right = shadowCam->top = 8.f;
        shadowCam->nearPlane = 1.f;
        shadowCam->farPlane = 40.f;
        sun->shadow->mapSize.set(2048, 2048);
        sun->shadow->bias = -0.0005f;
    }
    scene.add(sun);

    auto groundMaterial = MeshStandardMaterial::create();
    groundMaterial->color = Color(0.33f, 0.45f, 0.26f);
    groundMaterial->roughness = 1.f;
    auto ground = Mesh::create(PlaneGeometry::create(60.f, 60.f), groundMaterial);
    ground->rotation.x = -math::PI / 2.f;
    ground->receiveShadow = true;
    scene.add(ground);

    auto poleMaterial = MeshStandardMaterial::create();
    poleMaterial->color = Color(0.85f, 0.85f, 0.86f);
    poleMaterial->roughness = 0.45f;
    poleMaterial->metalness = 0.6f;
    auto pole = Mesh::create(CylinderGeometry::create(kPoleRadius, kPoleRadius * 1.4f, kPoleHeight, 16), poleMaterial);
    pole->position.y = 0.5f * kPoleHeight;
    pole->castShadow = true;
    scene.add(pole);
    auto knob = Mesh::create(SphereGeometry::create(kPoleRadius * 2.f, 16, 12), poleMaterial);
    knob->position.y = kPoleHeight;
    knob->castShadow = true;
    scene.add(knob);

    // ── The flag ─────────────────────────────────────────────────────────
    //
    // The cloth lives in the geometry's own space, so the geometry is moved
    // (hoist edge just clear of the pole) and the mesh only translated.
    const float hoistX = kPoleRadius + 0.02f;
    auto flagGeometry = PlaneGeometry::create(kFlagW, kFlagH, kFlagNx, kFlagNy);
    flagGeometry->translate(0.5f * kFlagW + hoistX, 0.f, 0.f);

    auto flagMaterial = MeshStandardMaterial::create();
    flagMaterial->map = makeFlagTexture();
    flagMaterial->roughness = 0.85f;
    flagMaterial->side = Side::Double;

    auto flagMesh = Mesh::create(flagGeometry, flagMaterial);
    flagMesh->position.set(0.f, kPoleHeight - 0.5f * kFlagH - 0.15f, 0.f);
    flagMesh->castShadow = true;
    flagMesh->receiveShadow = true;
    scene.add(flagMesh);

    ClothMesh flag(*flagGeometry);
    Cloth& cloth = flag.cloth();
    for (const size_t i : flag.particlesWhere([&](const Vector3& p) { return p.x < hoistX + 1e-3f; })) {
        cloth.pin(i);
    }
    // The pole, in the flag mesh's frame.
    cloth.addCapsule({0.f, -kPoleHeight, 0.f}, {0.f, 0.5f * kFlagH + 0.15f, 0.f}, kPoleRadius);

    float windSpeed = 7.f;     // m/s
    float windHeading = 20.f;  // degrees from +x towards +z
    const auto applyWind = [&] {
        const float a = windHeading * math::DEG2RAD;
        cloth.setWind({windSpeed * std::cos(a), 0.f, windSpeed * std::sin(a)});
    };
    applyWind();

    // ─────────────────────────────────────────────────────────────────────

    OrbitControls controls{camera, canvas};
    controls.target.copy(target);
    controls.enabled = !capturing;
    controls.update();

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    std::unique_ptr<RendererSettingsUi> ui;
    float stepMs = 0.f;
    bool paused = false;
    if (!capturing) {
        ui = std::make_unique<RendererSettingsUi>(canvas, *renderer, [&] {
            ImGui::Text("%zu particles   %zu + %zu constraints", cloth.particleCount(),
                        cloth.stretchConstraintCount(), cloth.bendConstraintCount());
            ImGui::Text("cloth step %.2f ms", stepMs);
            ImGui::Checkbox("Paused", &paused);

            ImGui::SeparatorText("Wind");
            ImGui::SliderFloat("Speed", &windSpeed, 0.f, 20.f, "%.1f m/s");
            ImGui::SliderFloat("Heading", &windHeading, -180.f, 180.f, "%.0f deg");
            ImGui::SliderFloat("Gustiness", &cloth.params.gustiness, 0.f, 1.f);

            ImGui::SeparatorText("Cloth");
            ImGui::SliderInt("Substeps", &cloth.params.substeps, 1, 32);
            ImGui::SliderFloat("Bend compliance", &cloth.params.bendCompliance, 0.f, 5.f, "%.2f m/N");
            ImGui::SliderFloat("Stretch compliance", &cloth.params.stretchCompliance, 0.f, 0.01f, "%.4f m/N");
            ImGui::SliderFloat("Damping", &cloth.params.damping, 0.f, 2.f, "%.2f /s");
        });
    }

    Clock clock;
    int frame = 0;

    canvas.animate([&] {
        if (capturing) {
            stepMs += millisecondsOf([&] { flag.update(kFixedDt); });
            renderer->render(scene, camera);
            if (++frame >= options.frames) {
                const std::filesystem::path path(options.shot);
                if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
                renderer->writeFramebuffer(path);
                std::cout << "wrote " << std::filesystem::absolute(path).string() << std::endl;
                std::cout << "cloth: " << cloth.particleCount() << " particles, mean step "
                          << stepMs / static_cast<float>(frame) << " ms" << std::endl;
                std::exit(0);
            }
            return;
        }

        controls.update();
        applyWind();

        const float dt = clock.getDelta();
        if (!paused) {
            stepMs = 0.95f * stepMs + 0.05f * millisecondsOf([&] { flag.update(dt); });
        }

        renderer->render(scene, camera);
        if (ui) ui->render();
    });
}
