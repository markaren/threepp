// Procedural mountain configurator.
//
// A single hero mountain massif rendered through the deferred VulkanRenderer, with a
// live ImGui panel that re-rolls and re-shapes the terrain. The heightfield is
// generated on the CPU by threepp::terrain::TerrainGenerator (fBm / ridged /
// hybrid multifractal + domain warp), optionally carved by droplet-hydraulic +
// thermal/talus EROSION, and baked into a horizontal PlaneGeometry; the renderer
// rebuilds the mesh BLAS automatically when the displaced vertex positions
// change (the plain dynamic-geometry path — no special mesh type, no renderer
// surgery). Lighting is HDRI image-based + a directional sun.
//
// Noise/shape edits re-roll a fast RAW preview on release; the (slower) erosion
// pass runs on demand via the Generate button. Presets bake fully eroded.
//

#include "capture_util.hpp"
#include "renderer_factory.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/extras/terrain/TerrainSplat.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::terrain;

namespace {

    constexpr float kDeg2Rad = 3.14159265358979323846f / 180.f;

    // Full path for a named config slot under the repo's terrain_configs/. The
    // save/load + JSON themselves live in the generator
    // (threepp::terrain::saveConfig / loadConfig).
    std::string configPath(const std::string& name) {
        const std::string n = name.empty() ? "terrain" : name;
        return (std::filesystem::path(PROJECT_FOLDER) / "terrain_configs" / (n + ".json")).string();
    }

    // Minimal, dependency-free ImGui file selector (owned — no native-dialog
    // lib). Browses directories and *.json files under a start folder; used for
    // both Save (editable filename) and Load (pick/double-click a file). Call
    // draw() every frame inside the ImGui frame; it returns true once when the
    // user confirms, with `result` holding the full path.
    struct FileDialog {
        bool active = false;     // popup is open
        bool justOpened = false; // fire ImGui::OpenPopup once
        bool saveMode = false;
        std::filesystem::path dir;
        char filename[80] = "";
        std::string result;

        void open(const std::filesystem::path& startDir, bool save, const char* initialName = "") {
            std::error_code ec;
            dir = std::filesystem::exists(startDir, ec) ? startDir : std::filesystem::current_path(ec);
            saveMode = save;
            std::snprintf(filename, sizeof(filename), "%s", initialName ? initialName : "");
            active = true;
            justOpened = true;
        }

        bool draw(const char* title) {
            if (justOpened) {
                ImGui::OpenPopup(title);
                justOpened = false;
            }
            bool chosen = false;
            ImGui::SetNextWindowSize({480, 360}, ImGuiCond_Appearing);
            if (ImGui::BeginPopupModal(title, &active)) {
                ImGui::TextDisabled("%s", dir.string().c_str());
                ImGui::Separator();

                ImGui::BeginChild("##list", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 2.1f), true);
                std::error_code ec;
                if (dir.has_parent_path() && dir != dir.root_path()) {
                    if (ImGui::Selectable("../")) dir = dir.parent_path();
                }
                std::vector<std::filesystem::path> dirs, files;
                if (std::filesystem::is_directory(dir, ec)) {
                    for (const auto& e : std::filesystem::directory_iterator(
                                 dir, std::filesystem::directory_options::skip_permission_denied, ec)) {
                        if (e.is_directory(ec)) dirs.push_back(e.path());
                        else if (e.path().extension() == ".json") files.push_back(e.path());
                    }
                }
                std::ranges::sort(dirs);
                std::ranges::sort(files);
                for (const auto& d : dirs) {
                    if (ImGui::Selectable(("[D] " + d.filename().string()).c_str())) dir = d;
                }
                for (const auto& f : files) {
                    const std::string stem = f.stem().string();
                    if (ImGui::Selectable(f.filename().string().c_str(), stem == filename))
                        std::snprintf(filename, sizeof(filename), "%s", stem.c_str());
                    if (!saveMode && ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                        result = f.string();
                        chosen = true;
                        ImGui::CloseCurrentPopup();
                    }
                }
                ImGui::EndChild();

                ImGui::SetNextItemWidth(-260);
                ImGui::InputText("##fname", filename, sizeof(filename),
                                 saveMode ? 0 : ImGuiInputTextFlags_ReadOnly);
                ImGui::SameLine();
                ImGui::TextDisabled(".json");

                ImGui::BeginDisabled(filename[0] == '\0');
                if (ImGui::Button(saveMode ? "Save" : "Load", ImVec2(110, 0))) {
                    result = (dir / (std::string(filename) + ".json")).string();
                    chosen = true;
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::Button("Cancel", ImVec2(110, 0))) ImGui::CloseCurrentPopup();

                ImGui::EndPopup();
            }
            if (chosen) active = false;
            return chosen;
        }
    };

    // Build data-driven splat rules from the configurator params. The ImGui
    // sliders drive `params`; this maps them onto ordered SplatLayers (grass →
    // scree → rock → snow) plus curvature response and macro variation. Heights
    // are in world metres (evaluate's `h`, the layer windows, and the curvature
    // height fn all share that unit); slope is 0 flat .. 1 vertical.
    terrain::SplatRules makeMountainRules(const TerrainParams& p, const std::vector<float>& field, int dim) {
        using namespace terrain;
        const float amp = p.amplitude;
        const float cellWorld = p.worldSize / static_cast<float>(std::max(dim - 1, 1));

        SplatRules r;
        // Curvature height field: the eroded [0,1] field scaled to metres,
        // wrapped in a HeightGrid so curvature reads the SAME surface the mesh
        // shows (gullies, ridges). Fixed eps → LOD-agnostic (mountains is a
        // single mesh, but keeps the helper honest for tiled users).
        std::vector<float> hm(field.size());
        for (size_t i = 0; i < field.size(); ++i) hm[i] = field[i] * amp;
        HeightGrid grid(std::move(hm), dim, p.worldSize);
        r.height = [grid = std::move(grid)](float x, float z) { return grid.sampleBilinear(x, z); };
        r.curvEps = std::max(cellWorld * 1.5f, 2.5f);
        r.curvScale = 60.f;
        r.aoStrength = 0.45f;// gentle occlusion in concave folds
        r.aoMax = std::clamp(p.aoMax, 0.f, 0.6f);

        const float snowH = p.snowLine * amp;
        const float e = std::max(p.bandEdge, 0.02f);

        SplatLayer grass;
        grass.color = p.grassColor;
        grass.slopeLo = 0.f;
        grass.slopeHi = p.slopeGrassMax;
        grass.slopeFeather = e;
        grass.heightHi = snowH;
        grass.heightFeather = amp * 0.05f;
        grass.concaveBias = 0.25f;// grass/soil catches in hollows
        grass.noiseAmpSlope = 0.03f;

        SplatLayer scree;
        scree.color = p.screeColor;
        scree.slopeLo = p.slopeGrassMax;
        scree.slopeHi = p.slopeRockMin;
        scree.slopeFeather = e;
        scree.concaveBias = 0.9f;// talus/scree collects in gullies & benches
        scree.noiseAmpSlope = 0.03f;

        SplatLayer rock;
        rock.color = p.rockColor;
        rock.slopeLo = p.slopeRockMin;
        rock.slopeHi = 1.f;
        rock.slopeFeather = 0.08f;
        rock.convexBias = 0.8f;// bare rock on convex ridge crests
        rock.weightFloor = 0.02f;// fallback so no texel resolves to grey

        SplatLayer snow;
        snow.color = p.snowColor;
        snow.slopeLo = 0.f;
        snow.slopeHi = p.snowSlopeMax;
        snow.slopeFeather = 0.1f;
        snow.heightLo = snowH;
        snow.heightFeather = amp * 0.05f;
        snow.noiseAmpHeight = p.snowNoiseAmp * amp;// noisy snowline
        snow.noiseFreq = 0.06f;

        r.layers = {grass, scree, rock, snow};
        return r;
    }

    // Bake the slope/curvature/altitude splat into an sRGB RGBA8 albedo image
    // (dim×dim, one texel per mesh vertex). Replaces TerrainGenerator's built-in
    // bakeSplatColors with the reusable, curvature-aware SplatRules helper.
    std::vector<unsigned char> bakeMountainSplat(const TerrainGenerator& gen, const TerrainParams& p) {
        const int dim = gen.dim();
        std::vector<unsigned char> out(static_cast<size_t>(std::max(dim, 1)) * std::max(dim, 1) * 4, 255u);
        const auto& field = gen.getField();
        if (dim < 2 || static_cast<size_t>(dim) * dim != field.size()) return out;

        const terrain::SplatRules rules = makeMountainRules(p, field, dim);
        const float amp = p.amplitude;
        const float cellWorld = p.worldSize / static_cast<float>(dim - 1);
        const float half = p.worldSize * 0.5f;
        const auto at = [dim](int x, int y) { return static_cast<size_t>(y) * dim + x; };

        std::vector<int> rows(static_cast<size_t>(dim));
        std::iota(rows.begin(), rows.end(), 0);
        parallelForEach(rows.begin(), rows.end(), [&](int z) {
            const float wz = -half + static_cast<float>(z) * cellWorld;
            for (int x = 0; x < dim; ++x) {
                const float wx = -half + static_cast<float>(x) * cellWorld;
                const int xm = std::max(x - 1, 0), xp = std::min(x + 1, dim - 1);
                const int zm = std::max(z - 1, 0), zp = std::min(z + 1, dim - 1);
                const float hC = field[at(x, z)];
                const float dHdx = (field[at(xp, z)] - field[at(xm, z)]) * amp / (2.f * cellWorld);
                const float dHdz = (field[at(x, zp)] - field[at(x, zm)]) * amp / (2.f * cellWorld);
                const float ny = 1.f / std::sqrt(dHdx * dHdx + dHdz * dHdz + 1.f);
                const float slope = 1.f - ny;
                const terrain::Rgb col = rules.evaluate(wx, wz, hC * amp, slope);

                // Z-flip: the albedo map's rows run opposite to world Z vs the
                // PlaneGeometry UV v-axis (matches TerrainGenerator::bakeSplatColors).
                const size_t o = at(x, dim - 1 - z) * 4;
                out[o + 0] = static_cast<unsigned char>(std::clamp(col[0], 0.f, 1.f) * 255.f + 0.5f);
                out[o + 1] = static_cast<unsigned char>(std::clamp(col[1], 0.f, 1.f) * 255.f + 0.5f);
                out[o + 2] = static_cast<unsigned char>(std::clamp(col[2], 0.f, 1.f) * 255.f + 0.5f);
                out[o + 3] = 255u;
            }
        });
        return out;
    }

}// namespace

int main(int argc, char** argv) {

    bool stress = false;// re-roll repeatedly to exercise the runtime regen / BLAS-rebuild path
    bool rerollOnce = false;// single in-place re-roll at frame 20, then settle (motion-vector test)
    bool noErode = false;// force erosion off (capture the pre-erosion baseline)
    bool topDown = false;// overhead camera (inspect drainage channels)
    bool retex = false;// mid-run live texturing change (verify live recolor reaches the GPU)
    bool cfgTest = false;// headless config save/load round-trip self-test
    int startPreset = 0;// 0 Alpine, 1 Rolling, 2 Mesa, 3 Volcanic
    float sceneScale = 1.f;     // debug: scale world/amplitude/feature (precision A/B)
    float ovSunAz = -1, ovSunEl = -1;// debug: override sun azimuth/elevation

    // Headless capture: --shot <name.png> [--frames N] [--preset P]
    // [--topdown] [--closeup]. Renders a settled frame (TAA/auto-exposure
    // converge) then writes PROJECT_FOLDER/aaa_caps/<name> and exits.
    bool closeUp = false;// low camera near a rock face (inspect detail relief)
    std::string shotPath;
    int shotFrames = 140;
    // --cam px py pz tx ty tz: fixed camera (close-up rock-face / grazing-terrain
    // shots for the RTAO validation gate). Overrides the orbit controls.
    bool camSet = false;
    float camV[6] = {};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--preset" && i + 1 < argc) startPreset = std::atoi(argv[++i]);
        else if (a == "--topdown") topDown = true;
        else if (a == "--noerode") noErode = true;
        else if (a == "--closeup") closeUp = true;
        else if (a == "--cam" && i + 6 < argc) {
            for (int k = 0; k < 6; ++k) camV[k] = static_cast<float>(std::atof(argv[++i]));
            camSet = true;
        }
    }

    Canvas canvas("Vulkan Deferred - Mountains", {{"vsync", false}});
    // Headless capture forces the Vulkan deferred renderer (the detail
    // normal/roughness + triplanar layer is Vulkan-only); interactive runs keep
    // the renderer-select menu.
    auto renderer = shotPath.empty() ? createRenderer(canvas)
                                     : createRenderer(canvas, GraphicsAPI::Vulkan);
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 1.0f;

    Scene scene;
    RGBELoader rgbe;
    if (auto env = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = env;
        scene.environment = env;// image-based lighting
    } else {
        scene.background = Color(0.55f, 0.70f, 0.92f);
        std::cerr << "[mountains] HDRI not found - falling back to flat sky background\n";
    }

    // Directional sun. Driven each frame from azimuth/elevation sliders; the
    // HDRI already carries a sun, so this mostly rakes the ridges for clear
    // relief and drives the shadow direction.
    auto sun = DirectionalLight::create(Color(1.0f, 0.96f, 0.88f), 2.6f);
    Object3D sunTarget;
    sunTarget.position.set(0.f, 0.f, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    // Terrain.
    TerrainParams params;
    startPreset = std::clamp(startPreset, 0, 3);
    applyPreset(startPreset, params);
    params.resolution = 1024;// default detail; 256/512 for snappier re-rolls, 2048/4096 for a final bake
    if (sceneScale != 1.f) {// keep terrain shape identical, just change world-coordinate magnitude
        params.worldSize *= sceneScale;
        params.amplitude *= sceneScale;
        params.featureScale *= sceneScale;
    }
    if (noErode) params.erosion = ErosionType::None;

    if (cfgTest) {// save → load into fresh params → compare (no renderer needed)
        const TerrainParams orig = params;
        const std::string path = configPath("cfgtest");
        TerrainParams loaded;// defaults
        const bool io = saveConfig(path, orig) && loadConfig(path, loaded);
        const bool ok = io && loaded.seed == orig.seed && loaded.noiseType == orig.noiseType &&
                        loaded.erosion == orig.erosion && loaded.droplets == orig.droplets &&
                        loaded.octaves == orig.octaves && loaded.terraces == orig.terraces &&
                        std::abs(loaded.worldSize - orig.worldSize) < 0.5f &&
                        std::abs(loaded.amplitude - orig.amplitude) < 0.5f &&
                        std::abs(loaded.snowLine - orig.snowLine) < 1e-3f &&
                        std::abs(loaded.rockColor[0] - orig.rockColor[0]) < 1e-3f &&
                        std::abs(loaded.snowColor[2] - orig.snowColor[2]) < 1e-3f;
        std::cout << "[cfgtest] round-trip " << (ok ? "PASS" : "FAIL") << std::endl;
        std::exit(ok ? 0 : 1);
    }

    TerrainGenerator gen(params.seed);
    auto terrainMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                           .color(Color(0.46f, 0.43f, 0.40f))
                                                           .roughness(0.93f)
                                                           .metalness(0.0f));
    // Initial bake runs erosion if the preset calls for it (a ~1s one-off).
    auto terrain = Mesh::create(gen.createGeometry(params, params.erosion != ErosionType::None), terrainMat);

    // Slope/altitude/snow splat baked into an sRGB albedo map — one texel per
    // mesh vertex, so the PlaneGeometry UVs map it 1:1. Re-baked whenever the
    // field or the texturing params change (see rebakeColors below).
    auto terrainTex = DataTexture::create(ImageData{bakeMountainSplat(gen, params)},
                                          static_cast<unsigned int>(gen.dim()),
                                          static_cast<unsigned int>(gen.dim()));
    terrainTex->colorSpace = ColorSpace::sRGB;
    terrainTex->magFilter = Filter::Linear;
    terrainTex->minFilter = Filter::Linear;
    terrainMat->map = terrainTex;
    terrainMat->color = Color::white;// albedo comes from the map now
    // Cm-scale tiled detail (Vulkan deferred only): albedo breakup + normal
    // relief + roughness, world-XZ anchored and distance-faded. The per-vertex
    // splat is ~sub-metre/texel — mush up close; this sharpens the near ground.
    {
        const terrain::DetailMaps dm = terrain::makeDetailMaps({});
        terrainMat->detailMap = dm.albedo;
        terrainMat->detailNormalMap = dm.normalRough;
        terrainMat->detailRepeat = 0.5f;// one repeat per 2 m
        terrainMat->detailStrength = 0.5f;
        terrainMat->detailNormalScale = 1.2f;
        terrainMat->detailRoughStrength = 0.5f;
    }
    terrainMat->needsUpdate();
    int builtTexDim = gen.dim();
    scene.add(terrain);

    auto rebakeColors = [&] {
        if (gen.dim() == builtTexDim) {
            terrainTex->setData(ImageData{bakeMountainSplat(gen, params)});
            terrainTex->needsUpdate();
        } else {// resolution changed → new texture dimensions
            terrainTex = DataTexture::create(ImageData{bakeMountainSplat(gen, params)},
                                             static_cast<unsigned int>(gen.dim()),
                                             static_cast<unsigned int>(gen.dim()));
            terrainTex->colorSpace = ColorSpace::sRGB;
            terrainTex->magFilter = Filter::Linear;
            terrainTex->minFilter = Filter::Linear;
            terrainMat->map = terrainTex;
            terrainMat->needsUpdate();
            builtTexDim = gen.dim();
        }
    };

    // Surrounding plain — a large flat ground a touch below the base so the
    // EdgeFade'd massif rises out of it instead of floating. Sits just under
    // y=0 (terrain valley floors) to avoid a coplanar z-fight at the seam.
    auto groundMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                          .color(Color(0.34f, 0.35f, 0.30f))
                                                          .roughness(1.0f)
                                                          .metalness(0.0f));
    auto ground = Mesh::create(PlaneGeometry::create(params.worldSize * 12.f, params.worldSize * 12.f), groundMat);
    ground->rotation.x = -math::PI / 2.f;
    ground->position.y = -2.0f;// terrain rim sinks below this and is occluded by the plain
    scene.add(ground);

    int builtResolution = params.resolution;
    float builtWorldSize = params.worldSize;

    // Side-on 3/4 view: ridges read against the sky rather than as a top-down
    // spike field, and the finite-patch edges sit closer to the horizon.
    PerspectiveCamera camera(50.f, canvas.aspect(), 1.f, 10000.f);
    camera.position.set(params.worldSize * 0.95f, params.amplitude * 0.85f, params.worldSize * 0.95f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, params.amplitude * 0.45f, 0.f);
    if (topDown) {// overhead — reveals the drainage network on the surface
        camera.position.set(0.f, params.worldSize * 1.3f, 0.1f);
        controls.target.set(0.f, 0.f, 0.f);
    }
    if (closeUp) {// low, near a mid-slope face — inspect detail relief / roughness
        camera.position.set(params.worldSize * 0.18f, params.amplitude * 0.42f, params.worldSize * 0.18f);
        controls.target.set(0.f, params.amplitude * 0.30f, 0.f);
    }
    controls.update();

    // ---- Configurator state ----
    int preset = startPreset;// 0 Alpine, 1 Rolling, 2 Mesa, 3 Volcanic, 4 Custom
    int noiseTypeIdx = static_cast<int>(params.noiseType);
    int falloffIdx = static_cast<int>(params.falloff);
    int erosionIdx = static_cast<int>(params.erosion);
    float sunAzimuth = ovSunAz >= 0 ? ovSunAz : 135.f;
    float sunElevation = ovSunEl >= 0 ? ovSunEl : 32.f;
    FileDialog fileDlg;// config save/load file selector
    const std::filesystem::path configDir = std::filesystem::path(PROJECT_FOLDER) / "terrain_configs";

    bool regenRequested = false;
    bool regenErode = false;// when servicing a regen, also run the (slow) erosion pass
    bool recolorRequested = false;// texturing params changed; re-bake the splat map only
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    // Noise/shape edits re-roll a fast RAW preview (no erosion); the eroded
    // result is produced on demand by the Generate button (erosion is the
    // expensive pass, ~1s, so it must not run on every slider release).
    auto markCustom = [&](bool changed) { if (changed) { preset = 4; regenRequested = true; } };

    // Generic renderer settings (tone map, exposure, upscaler, ...) come from
    // the shared panel; the terrain configurator widgets are the extra lambda.
    RendererSettingsUi ui(canvas, *renderer, [&] {
        ImGui::Text("verts: %d",
                    terrain->geometry() && terrain->geometry()->getAttribute<float>("position")
                            ? terrain->geometry()->getAttribute<float>("position")->count()
                            : 0);
        ImGui::Separator();

        // Preset + seed.
        if (ImGui::Combo("Preset", &preset, "Alpine\0Rolling Hills\0Desert Mesa\0Volcanic\0Custom\0")) {
            if (preset < 4) {
                applyPreset(preset, params);
                noiseTypeIdx = static_cast<int>(params.noiseType);
                falloffIdx = static_cast<int>(params.falloff);
                erosionIdx = static_cast<int>(params.erosion);
                regenRequested = true;
                regenErode = (params.erosion != ErosionType::None);// presets define a complete eroded look
            }
        }
        {
            int seedI = static_cast<int>(params.seed);
            if (ImGui::InputInt("Seed", &seedI)) {
                params.seed = static_cast<unsigned int>(std::max(seedI, 0));
                regenRequested = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Randomize")) {
                params.seed = std::random_device{}();
                regenRequested = true;
            }
        }

        // Config save/load — writes/reads terrain_configs/<name>.json. Because
        // generation is deterministic, a loaded config reproduces the exact
        // terrain (erosion + texturing included).
        ImGui::SeparatorText("Config");
        if (ImGui::Button("Save...##cfg")) fileDlg.open(configDir, true, "my_terrain");
        ImGui::SameLine();
        if (ImGui::Button("Load...##cfg")) fileDlg.open(configDir, false);
        ImGui::SameLine();
        ImGui::TextDisabled("(.json)");

        ImGui::SeparatorText("Grid");
        {
            static const int resVals[] = {128, 256, 512, 1024, 2048, 4096};
            constexpr int resCount = static_cast<int>(std::size(resVals));
            int resIdx = 3;// fallback highlight = 1024
            for (int i = 0; i < resCount; ++i)
                if (resVals[i] == params.resolution) resIdx = i;
            if (ImGui::Combo("Resolution", &resIdx, "128\0" "256\0" "512\0" "1024\0" "2048\0" "4096\0")) {
                params.resolution = resVals[resIdx];
                regenRequested = true;
            }
            markCustom(ImGui::SliderFloat("World size (m)", &params.worldSize, 200.f, 4000.f, "%.0f"));
        }

        ImGui::SeparatorText("Base noise");
        if (ImGui::Combo("Type", &noiseTypeIdx, "fBm\0Ridged\0Hybrid\0")) {
            params.noiseType = static_cast<NoiseType>(noiseTypeIdx);
            preset = 4;
            regenRequested = true;
        }
        markCustom(ImGui::SliderFloat("Feature scale (m)", &params.featureScale, 40.f, 2000.f, "%.0f"));
        markCustom(ImGui::SliderInt("Octaves", &params.octaves, 1, 11));
        markCustom(ImGui::SliderFloat("Lacunarity", &params.lacunarity, 1.5f, 3.0f, "%.2f"));
        markCustom(ImGui::SliderFloat("Gain", &params.gain, 0.2f, 0.8f, "%.2f"));
        markCustom(ImGui::SliderFloat("Amplitude (m)", &params.amplitude, 0.f, 1500.f, "%.0f"));
        markCustom(ImGui::SliderFloat("Domain warp", &params.warp, 0.f, 1.f, "%.2f"));

        ImGui::SeparatorText("Shape");
        if (params.noiseType != NoiseType::fBm)
            markCustom(ImGui::SliderFloat("Ridge sharpness", &params.ridgeSharpness, 0.f, 1.f, "%.2f"));
        markCustom(ImGui::SliderFloat("Height exponent", &params.heightExponent, 0.6f, 2.0f, "%.2f"));
        markCustom(ImGui::SliderInt("Terraces", &params.terraces, 0, 16));
        if (ImGui::Combo("Falloff", &falloffIdx, "None\0Radial\0")) {
            params.falloff = static_cast<Falloff>(falloffIdx);
            preset = 4;
            regenRequested = true;
        }

        ImGui::SeparatorText("Erosion");
        // Erosion knobs only take effect on the next Generate (it's the slow
        // pass). Noise/shape edits above show the raw, un-eroded shape live.
        if (ImGui::Combo("Erosion type", &erosionIdx, "None\0Hydraulic\0Thermal\0Both\0")) {
            params.erosion = static_cast<ErosionType>(erosionIdx);
            preset = 4;
        }
        ImGui::SliderInt("Droplets", &params.droplets, 0, 400000);
        ImGui::SliderFloat("Erode rate", &params.erodeSpeed, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Deposit rate", &params.depositSpeed, 0.f, 1.f, "%.2f");
        ImGui::SliderInt("Erosion radius", &params.erosionRadius, 1, 6);
        ImGui::SliderFloat("Talus angle", &params.talusAngle, 20.f, 60.f, "%.0f");
        ImGui::SliderInt("Thermal iters", &params.thermalIterations, 0, 200);
        if (ImGui::Button("Generate (erode)")) {
            preset = 4;
            regenRequested = true;
            regenErode = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("~1s pass");

        ImGui::SeparatorText("Texturing");
        // Cheap — re-bakes only the albedo map (no geometry rebuild), on release.
        recolorRequested |= ImGui::SliderFloat("Snow line", &params.snowLine, 0.f, 1.f, "%.2f");
        recolorRequested |= ImGui::SliderFloat("Snow wiggle", &params.snowNoiseAmp, 0.f, 0.2f, "%.2f");
        recolorRequested |= ImGui::SliderFloat("Snow slope max", &params.snowSlopeMax, 0.f, 1.f, "%.2f");
        recolorRequested |= ImGui::SliderFloat("Grass/scree slope", &params.slopeGrassMax, 0.f, 1.f, "%.2f");
        recolorRequested |= ImGui::SliderFloat("Scree/rock slope", &params.slopeRockMin, 0.f, 1.f, "%.2f");
        recolorRequested |= ImGui::SliderFloat("Band softness", &params.bandEdge, 0.01f, 0.2f, "%.2f");
        recolorRequested |= ImGui::ColorEdit3("Rock", params.rockColor.data());
        recolorRequested |= ImGui::ColorEdit3("Grass/base", params.grassColor.data());
        recolorRequested |= ImGui::ColorEdit3("Scree", params.screeColor.data());
        recolorRequested |= ImGui::ColorEdit3("Snow", params.snowColor.data());

        ImGui::SeparatorText("Sun");
        ImGui::SliderFloat("Sun azimuth", &sunAzimuth, 0.f, 360.f, "%.0f");
        ImGui::SliderFloat("Sun elevation", &sunElevation, 1.f, 89.f, "%.0f");
        ImGui::SliderFloat("Sun intensity", &sun->intensity, 0.f, 8.f, "%.2f");

        ImGui::Separator();
        ImGui::TextDisabled("Drag = orbit, scroll = zoom");
        ImGui::TextDisabled("Noise edits preview raw on release;");
        ImGui::TextDisabled("press Generate to erode.");

        // File selector (modal). Confirms into fileDlg.result; save or load
        // accordingly.
        if (fileDlg.draw("Terrain config")) {
            if (fileDlg.saveMode) {
                if (saveConfig(fileDlg.result, params))
                    std::cout << "[config] saved " << fileDlg.result << std::endl;
                else
                    std::cerr << "[config] save failed: " << fileDlg.result << std::endl;
            } else if (loadConfig(fileDlg.result, params)) {
                noiseTypeIdx = static_cast<int>(params.noiseType);
                falloffIdx = static_cast<int>(params.falloff);
                erosionIdx = static_cast<int>(params.erosion);
                preset = 4;// custom
                regenRequested = true;
                regenErode = (params.erosion != ErosionType::None);
                std::cout << "[config] loaded " << fileDlg.result << std::endl;
            } else {
                std::cerr << "[config] load failed: " << fileDlg.result << std::endl;
            }
        }
    }, "Mountain Configurator");

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer->setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (camSet) {// fixed close-up camera overrides the orbit controls
            camera.position.set(camV[0], camV[1], camV[2]);
            controls.target.set(camV[3], camV[4], camV[5]);
            camera.lookAt(controls.target);
        } else {
            controls.update();
        }

        // Single in-place re-roll at frame 20 (then settle): exercises the
        // applyTo path and lets the motion vector settle for the prevVertex
        // re-sync test. With the bug, the re-rolled mesh shows constant nonzero
        // motion forever; fixed, it returns to zero a frame after the re-roll.
        {
            static int loopFrame = 0;
            ++loopFrame;
            if (rerollOnce && loopFrame == 20) {
                params.seed += 24601u;
                regenRequested = true;
                rerollOnce = false;
                std::cout << "[reroll] in-place re-roll, seed=" << params.seed << std::endl;
            }
        }

        // Live texturing test: at frame 30, drop the snow line via the recolor
        // path only (no geometry change). With the renderer's material-texture
        // version refresh, the terrain should turn much snowier; without it the
        // colour would never update.
        {
            static int rtFrame = 0;
            if (retex && ++rtFrame == 30) {
                params.snowLine = 0.1f;
                recolorRequested = true;
                std::cout << "[retex] snowLine -> 0.1 (live recolor only)" << std::endl;
            }
        }

        // Stress mode: drive the same regen path the UI uses, alternating
        // resolution so both the same-topology refit and the topology-change
        // rebuild get exercised at runtime. Headless validation of "re-roll".
        if (stress) {
            static int sTick = 0;
            if (++sTick % 25 == 0) {
                params.seed += 7919u;
                params.resolution = (params.resolution == 512 ? 256 : 512);
                regenRequested = true;
                std::cout << "[stress] re-roll seed=" << params.seed
                          << " res=" << params.resolution << std::endl;
            }
        }

        // Sun direction from azimuth/elevation. position = direction * range
        // (DirectionalLight points from position toward its target at origin).
        const float az = sunAzimuth * kDeg2Rad;
        const float el = sunElevation * kDeg2Rad;
        sun->position.set(std::cos(el) * std::sin(az),
                          std::sin(el),
                          std::cos(el) * std::cos(az));
        sun->position.multiplyScalar(std::max(params.worldSize, 1000.f));

        // Regenerate the terrain once the user releases the control (avoids a
        // per-frame BLAS rebuild while a slider is being dragged). In headless
        // capture/stress mode there is no live UI, so regenerate immediately.
        const bool uiBusy = ImGui::IsAnyItemActive();
        if (regenRequested && !uiBusy) {
            if (params.seed != gen.seed()) gen.reseed(params.seed);
            gen.buildField(params);              // noise heightfield (fast)
            if (regenErode) gen.erode(params);   // droplet + thermal (the ~1s pass)
            if (params.resolution != builtResolution || params.worldSize != builtWorldSize) {
                terrain->setGeometry(gen.makeGeometry(params));
                builtResolution = params.resolution;
                builtWorldSize = params.worldSize;
            } else {
                gen.displaceTo(*terrain->geometry(), params);
            }
            rebakeColors();// slope/altitude changed → re-bake the splat albedo
            regenRequested = false;
            regenErode = false;
            recolorRequested = false;
        } else if (recolorRequested && !uiBusy) {
            rebakeColors();// texturing params changed; geometry is unchanged
            recolorRequested = false;
        }

        renderer->render(scene, camera);

        if (!shotPath.empty()) {
            static int shotFrame = 0;
            if (++shotFrame >= shotFrames) {
                std::ostringstream stats;
                stats << " (" << fps << " fps)";
                capture::finishShot(*renderer, shotPath, stats.str());
            }
        } else {
            ui.render();
        }
    });

    return 0;
}
