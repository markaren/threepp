// Procedural log-cabin demo.
//
// Renders the cabin from threepp/extras/architecture/LogCabin.hpp with a live
// ImGui panel that rebuilds it when a parameter changes. Defaults to the
// OpenGL backend; pass --vulkan to force the other path.
//
// Headless capture (this is how the geometry is actually verified — a log
// cabin either reads correctly to the eye or it does not, and no unit test
// says which):
//
//   log_cabin_demo --shot out.png [--frames N] [--cam x y z] [--target x y z]
//                  [--size w h] [--orbit deg] [--dist m]
//
// --orbit/--dist place the camera on a circle around the building at a fixed
// eye height, which makes a four-shot turntable a one-liner.

#include "renderer_factory.hpp"

#include "threepp/extras/architecture/LogCabin.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

using namespace threepp;
using namespace threepp::architecture;

int main(int argc, char** argv) {

    std::string shotPath;
    int shotFrames = 4;
    int shotFrame = 0;
    int width = 1600, height = 900;
    float camArg[3] = {17.f, 6.5f, 17.f};
    float tgtArg[3] = {0.f, 3.2f, 0.f};
    float orbitDeg = -1.f;// <0 → use --cam
    float orbitDist = 26.f;
    float orbitEye = 6.0f;
    std::optional<GraphicsAPI> api = GraphicsAPI::OpenGL;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotPath = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--size") == 0 && i + 2 < argc) {
            width = std::atoi(argv[++i]);
            height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cam") == 0 && i + 3 < argc) {
            for (float& c : camArg) c = std::stof(argv[++i]);
        } else if (std::strcmp(argv[i], "--target") == 0 && i + 3 < argc) {
            for (float& c : tgtArg) c = std::stof(argv[++i]);
        } else if (std::strcmp(argv[i], "--orbit") == 0 && i + 1 < argc) orbitDeg = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--dist") == 0 && i + 1 < argc) orbitDist = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--eye") == 0 && i + 1 < argc) orbitEye = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--vulkan") == 0) api = GraphicsAPI::Vulkan;
        else if (std::strcmp(argv[i], "--gl") == 0) api = GraphicsAPI::OpenGL;
    }
    const bool capturing = !shotPath.empty();

    if (orbitDeg >= 0.f) {
        const float a = orbitDeg * math::DEG2RAD;
        camArg[0] = std::sin(a) * orbitDist;
        camArg[1] = orbitEye;
        camArg[2] = std::cos(a) * orbitDist;
    }

    Canvas canvas("Procedural Log Cabin",
                  {{"vsync", !capturing}, {"aa", 4}, {"size", WindowSize{width, height}}});
    auto renderer = createRenderer(canvas, api);
    renderer->setClearColor(Color(0.55f, 0.66f, 0.78f));
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 0.90f;
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    Scene scene;

    RGBELoader hdrLoader;
    if (auto hdr = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = hdr;
        scene.environment = hdr;
    }

    // Deliberately weak fill. With a bright sky HDR already driving the IBL, a
    // strong ambient term on top flattens the sun's shadows into nothing —
    // the covered porch then renders as brightly as the sunlit facade and the
    // building loses all its depth.
    scene.add(AmbientLight::create(Color(0.72f, 0.80f, 0.95f), 0.08f));

    // Low-ish afternoon sun raking along the facade from the left, NOT square
    // onto it: head-on it floods straight under the porch roof and the deep
    // shade that gives a covered porch its depth never appears.
    auto sun = DirectionalLight::create(Color(1.0f, 0.955f, 0.885f), 4.4f);
    sun->position.set(-26.f, 21.f, 11.f);
    sun->castShadow = true;
    {
        auto* cam = sun->shadow->camera->as<OrthographicCamera>();
        cam->left = cam->bottom = -24.f;
        cam->right = cam->top = 24.f;
        cam->nearPlane = 1.f;
        cam->farPlane = 90.f;
        sun->shadow->mapSize.set(4096, 4096);
        sun->shadow->bias = -0.0006f;
        sun->shadow->normalBias = 0.02f;
    }
    scene.add(sun);

    // Ground — matte grass so the building silhouette and its shadow read
    // clearly. The FJORD host will supply real terrain instead.
    auto ground = Mesh::create(
            PlaneGeometry::create(220.f, 220.f),
            MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                 .color(Color(0.198f, 0.240f, 0.128f))
                                                 .roughness(1.0f)
                                                 .metalness(0.0f)));
    ground->rotation.x = -math::PI / 2.f;
    ground->receiveShadow = true;
    scene.add(ground);

    // ── Cabin ────────────────────────────────────────────────────────────
    CabinParams params;
    auto materials = makeCabinMaterials(params);
    auto cabin = createLogCabin(params, materials);
    scene.add(cabin);

    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 500.f);
    camera.position.set(camArg[0], camArg[1], camArg[2]);
    OrbitControls controls{camera, canvas};
    controls.target.set(tgtArg[0], tgtArg[1], tgtArg[2]);
    controls.update();

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    bool regen = false;
    bool regenMaterials = false;

    auto rebuild = [&] {
        scene.remove(*cabin);
        if (regenMaterials) {
            materials = makeCabinMaterials(params);
            regenMaterials = false;
        }
        cabin = createLogCabin(params, materials);
        scene.add(cabin);
    };

    std::unique_ptr<RendererSettingsUi> ui;
    if (!capturing) {
        ui = std::make_unique<RendererSettingsUi>(canvas, *renderer, [&] {
            const auto m = cabinMetrics(params);
            ImGui::Text("courses %d   eave %.2f   ridge %.2f", m.wallCourses, m.eaveY, m.ridgeY);
            ImGui::Separator();

            auto f = [&](const char* label, float* v, float lo, float hi, const char* fmt = "%.2f") {
                if (ImGui::SliderFloat(label, v, lo, hi, fmt)) regen = true;
            };

            ImGui::SeparatorText("Footprint");
            f("Length", &params.length, 6.f, 30.f, "%.1f");
            f("Depth", &params.depth, 5.f, 16.f, "%.1f");
            f("Wall height", &params.wallHeight, 2.0f, 6.0f);
            f("Floor height", &params.floorHeight, 0.2f, 1.6f);

            ImGui::SeparatorText("Logs");
            f("Log radius", &params.logRadius, 0.08f, 0.35f, "%.3f");
            f("Course overlap", &params.courseOverlap, 0.f, 0.40f, "%.3f");
            f("Corner overhang", &params.cornerOverhang, 0.f, 1.0f);
            f("Radius jitter", &params.logRadiusJitter, 0.f, 0.3f, "%.3f");
            f("Log sag", &params.logSag, 0.f, 0.05f, "%.3f");

            ImGui::SeparatorText("Roof");
            f("Pitch", &params.roofPitchDeg, 15.f, 55.f, "%.1f");
            f("Eave overhang", &params.eaveOverhang, 0.f, 1.5f);
            f("Rake overhang", &params.rakeOverhang, 0.f, 1.5f);
            f("Thickness", &params.roofThickness, 0.05f, 0.4f, "%.3f");

            ImGui::SeparatorText("Dormer");
            if (ImGui::Checkbox("Dormer", &params.dormer)) regen = true;
            f("Dormer x", &params.dormerCenterX, -10.f, 10.f);
            f("Dormer width", &params.dormerWidth, 1.5f, 8.f);
            f("Dormer rise", &params.dormerRiseFrac, 0.3f, 0.95f);

            ImGui::SeparatorText("Porch");
            if (ImGui::Checkbox("Porch", &params.porch)) regen = true;
            f("Porch depth", &params.porchDepth, 1.2f, 5.f);
            f("Porch x0", &params.porchStartX, -16.f, 0.f);
            f("Porch x1", &params.porchEndX, 0.f, 16.f);
            f("Porch pitch", &params.porchRoofPitchDeg, 4.f, 30.f, "%.1f");
            f("Rail height", &params.railHeight, 0.6f, 1.3f);
            f("Post spacing", &params.postSpacing, 1.0f, 4.f);
            f("Steps x", &params.stepsCenterX, -10.f, 10.f);

            ImGui::SeparatorText("Colours (rebuilds textures)");
            if (ImGui::ColorEdit3("Log", params.logColor.data())) { regen = regenMaterials = true; }
            if (ImGui::ColorEdit3("Shingle", params.shingleColor.data())) { regen = regenMaterials = true; }
            if (ImGui::ColorEdit3("Timber", params.timberColor.data())) { regen = regenMaterials = true; }
            if (ImGui::ColorEdit3("Trim", params.trimColor.data())) { regen = regenMaterials = true; }
            if (ImGui::ColorEdit3("Stone", params.stoneColor.data())) { regen = regenMaterials = true; }
            {
                int s = static_cast<int>(params.seed);
                if (ImGui::InputInt("Seed", &s)) {
                    params.seed = static_cast<unsigned int>(std::max(s, 0));
                    regen = regenMaterials = true;
                }
            }
            if (ImGui::Button("Rebuild", ImVec2(-1, 0))) regen = true;
        }, "Log Cabin");
    }

    canvas.animate([&] {
        controls.update();

        if (regen && !ImGui::IsAnyItemActive()) {
            rebuild();
            regen = false;
        }

        renderer->render(scene, camera);
        if (!capturing) {
            ui->render();
        } else if (++shotFrame >= shotFrames) {
            const std::filesystem::path out{shotPath};
            renderer->writeFramebuffer(out);
            std::cout << "wrote " << std::filesystem::absolute(out).string() << std::endl;
            std::exit(0);
        }
    });
}
