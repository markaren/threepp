// Fish soft body demo — loads USDZ fish models from an external database
// and drops them as PhysX deformable volumes. Requires a CUDA-capable GPU.

#include "threepp/threepp.hpp"

#include "threepp/renderers/GLRenderer.hpp"
#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxDebugRenderer.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/loaders/USDLoader.hpp"

#include "ConveyorSystem.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"

#include <nlohmann/json.hpp>

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>

using namespace threepp;
using namespace ::physx;

namespace fs = std::filesystem;

namespace {

    // Default data locations; override positionally on the command line (see main).
    const fs::path kDefaultFishDb = "C:/dev/FHF_FishProcessing/Database-Fish/Fish_database";
    const fs::path kDefaultParamsFile = "C:/dev/FHF_FishProcessing/Database-Fish/Physical_Parameters/parameters.json";

    std::vector<std::string> discoverFishIds(const fs::path& fishDb) {
        std::vector<std::string> ids;
        if (!fs::is_directory(fishDb)) return ids;
        for (const auto& entry : fs::directory_iterator(fishDb)) {
            if (entry.path().extension() == ".usdz") {
                ids.push_back(entry.path().stem().string());
            }
        }
        std::ranges::sort(ids);
        return ids;
    }

    // Per-specimen physical parameters parsed from parameters.json (keyed by the
    // same ID as the .usdz filename). Length drives the real-world scale, weight
    // the soft-body mass, friction the contact response.
    struct FishParams {
        std::string species;
        float lengthM = 0.4f;// total_length_mm / 1000
        float weightKg = 0.f;// 0 => keep PhysX default unit-density mass
        // Measured static friction (head-first) on each conveyor surface. A real
        // conveyor is the plastic module belt — much grippier than bare steel, so
        // that's the right value for fish riding the belt (esp. on slopes).
        float frictionSteel = 0.5f;
        float frictionPerfSteel = 0.5f;
        float frictionBelt = 0.5f;// plastic_module_belt
    };

    std::unordered_map<std::string, FishParams> discoverFishParams(const fs::path& paramsFile) {
        std::unordered_map<std::string, FishParams> out;
        std::ifstream f(paramsFile);
        if (!f) {
            std::cerr << "No parameters file at " << paramsFile << " (using defaults)" << std::endl;
            return out;
        }
        nlohmann::json j;
        try {
            f >> j;
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse " << paramsFile << ": " << e.what() << std::endl;
            return out;
        }
        for (const auto& [id, v] : j.items()) {
            FishParams p;
            if (v.contains("species")) p.species = v["species"].get<std::string>();
            if (v.contains("dimensions") && v["dimensions"].contains("total_length_mm")) {
                p.lengthM = v["dimensions"]["total_length_mm"].get<float>() / 1000.f;
            }
            if (v.contains("weight_kg")) p.weightKg = v["weight_kg"].get<float>();
            if (auto sf = v.find("static_friction"); sf != v.end()) {
                if (sf->contains("steel")) p.frictionSteel = (*sf)["steel"].value("head_first", p.frictionSteel);
                if (sf->contains("perforated_steel")) p.frictionPerfSteel = (*sf)["perforated_steel"].value("head_first", p.frictionPerfSteel);
                if (sf->contains("plastic_module_belt")) p.frictionBelt = (*sf)["plastic_module_belt"].value("head_first", p.frictionBelt);
            }
            out[id] = p;
        }
        return out;
    }

    struct FishCache {
        std::shared_ptr<BufferGeometry> geometry;
        std::shared_ptr<Material> material;
    };

    FishCache loadFish(USDLoader& loader, const fs::path& fishDb, const std::string& fishId, float targetLengthM) {
        auto group = loader.load(fishDb / (fishId + ".usdz"));
        if (!group) throw std::runtime_error("Failed to load " + fishId);

        std::shared_ptr<BufferGeometry> geom;
        std::shared_ptr<Material> mat;
        group->traverseType<Mesh>([&](Mesh& m) {
            if (geom) return;
            geom = m.geometry();
            mat = m.material();
        });
        if (!geom) throw std::runtime_error("No mesh found in " + fishId);

        // Make the fish diffuse-dominant so the Vulkan path tracer albedo-demodulates
        // them (chit gate: specFrac < 0.3). Demod lets the denoiser smooth only the
        // lighting while keeping the skin/scale albedo crisp; otherwise it filters in
        // radiance space and washes the texture out. The usual culprit is authored
        // metalness (F0 -> albedo -> high spec fraction), so drop it; wet cod read
        // matte anyway. MeshPhysicalMaterial also derives from MeshStandardMaterial.
        if (auto stdMat = std::dynamic_pointer_cast<MeshStandardMaterial>(mat)) {
            stdMat->metalness = 0.f;
            stdMat->roughness = std::max(stdMat->roughness, 0.7f);
        }

        // Keep the visual geometry at full resolution. The physics mesh is
        // simplified internally by addSoftBody (remesh + voxelise into a coarse
        // tet cage), and these full-res vertices are skinned barycentrically
        // onto that cage every frame — so only the simulation is decimated, the
        // render model retains its original detail.
        geom = simplifyGeometry(*geom, 0.8f);

        Box3 bbox;
        bbox.setFromObject(*group, true);
        Vector3 size;
        bbox.getSize(size);
        const float maxDim = std::max({size.x, size.y, size.z});
        if (maxDim > 0.f) {
            const float s = targetLengthM / maxDim;
            geom->scale(s, s, s);
            geom->center();
        }
        
        return {geom, mat};
    }

    struct SpawnedFish {
        std::shared_ptr<Mesh> mesh;
        SoftBody* body;
    };

    SpawnedFish spawnFish(PhysxWorld& world, Scene& scene,
                          const FishCache& fish,
                          const Vector3& position,
                          PxDeformableVolumeMaterial* sbMat,
                          float yaw, int voxelRes, int solverIters,
                          const std::string& cacheKey, float mass) {
        auto geomClone = fish.geometry->clone();
        auto mesh = Mesh::create(geomClone, fish.material);
        mesh->position.copy(position);
        mesh->rotation.y = yaw;
        scene.add(mesh);
        auto* body = world.addSoftBody(*mesh, sbMat, voxelRes,
                                       static_cast<unsigned>(solverIters), false, cacheKey, mass);
        return {mesh, body};
    }

}// namespace

int main(int argc, char** argv) {

    // Data locations from the command line (all optional, positional):
    //   physx_fish [conveyorDir] [fishDir] [parametersFile]
    const fs::path conveyorDir = argc > 1 ? fs::path(argv[1])
                                          : fs::path(std::string(PROJECT_FOLDER) + "/data/models/conveyor");
    const fs::path fishDir = argc > 2 ? fs::path(argv[2]) : kDefaultFishDb;
    const fs::path paramsFile = argc > 3 ? fs::path(argv[3]) : kDefaultParamsFile;

    auto fishIds = discoverFishIds(fishDir);
    if (fishIds.empty()) {
        std::cerr << "No .usdz files found in " << fishDir << std::endl;
        return 1;
    }
    std::cout << "Found " << fishIds.size() << " fish models" << std::endl;

    auto fishParams = discoverFishParams(paramsFile);

    Canvas canvas("PhysX Fish Softbody", {{"aa", 4}, {"vsync", true}});
    auto renderer = createRenderer(canvas);

    if (auto pt = dynamic_cast<VulkanRenderer*>(renderer.get())) {
        pt->setSamplesPerPixel(4);
        pt->setDenoise(true);
        pt->setRestirDIEnabled(true);
        pt->setSilhouetteMsaaExtra(4);
    }

    Scene scene;
    RGBELoader hdrLoader;

    if (auto hdrTexture = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr")) {
        scene.background = hdrTexture;
        scene.environment = hdrTexture;
    }

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.01f, 1000);
    camera->position.set(2.f, 5.f, 5.0f);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 0.3f, 0);

    auto sun = DirectionalLight::create(0xffffff, 2.5f);
    sun->position.set(5, 10, 7);
    scene.add(sun);

    PhysxWorld::Settings settings;
    settings.enableGpuDynamics = true;
    PhysxWorld world(settings);

    PhysxDebugRenderer debugRenderer(world);
    debugRenderer.visible = false;
    scene.addRef(debugRenderer);

    // --- Conveyor system: rebuild the layout authored in the designer ----------
    // Belts, rollers, cleats and separators — their PhysX colliders + visuals — all live in
    // ConveyorSystem (see ConveyorSystem.hpp/.cpp). The fish sim just reads its inlets for
    // spawn points and ticks it each frame; the conveyor registers its own substep hooks.
    fishsim::ConveyorSystem conveyor(world, scene, conveyorDir);

    USDLoader usdLoader;// loads the fish models below

    std::unordered_map<std::string, FishCache> fishCache;
    std::unordered_map<std::string, PxDeformableVolumeMaterial*> fishMats;
    std::vector<SpawnedFish> spawnedFish;

    std::mt19937 rng{42};
    int selectedFish = 0;
    int voxelRes = 8;
    int solverIters = 15;// soft-body solver iterations (lower = faster, less stiff)
    // GPU tet-skinning is supported on the GL forward renderer (vertex-shader tet
    // blend from a tet texture) and on the Vulkan path tracer (tet_skinning.comp
    // writes the deformed verts straight into the BLAS). Other renderers (WGPU/Cross)
    // fall back to CPU skin. The toggle stays enabled on supported renderers so it can
    // be turned off to A/B against the CPU path.
    const bool tetSkinSupported =
            dynamic_cast<GLRenderer*>(renderer.get()) != nullptr
#ifdef THREEPP_WITH_VULKAN
            || dynamic_cast<VulkanRenderer*>(renderer.get()) != nullptr
#endif
            ;
    bool gpuSkin = tetSkinSupported;// skin the visual on the GPU (tet texture) instead of CPU
    int beltSurface = 2;// 0=steel, 1=perforated steel, 2=plastic module belt (the belt)

    auto paramFor = [&](const std::string& id) -> const FishParams& {
        static const FishParams def{};
        auto it = fishParams.find(id);
        return it != fishParams.end() ? it->second : def;
    };

    // Measured contact friction for the currently-selected conveyor surface.
    auto frictionFor = [&](const std::string& id) -> float {
        const auto& p = paramFor(id);
        switch (beltSurface) {
            case 0: return p.frictionSteel;
            case 1: return p.frictionPerfSteel;
            default: return p.frictionBelt;
        }
    };

    // One soft-body material per (species, surface): the measured friction differs
    // per fish and per surface. Stiffness/damping aren't in the dataset (constant).
    auto materialFor = [&](const std::string& id) -> PxDeformableVolumeMaterial* {
        const std::string key = id + "#" + std::to_string(beltSurface);
        if (auto it = fishMats.find(key); it != fishMats.end()) return it->second;
        auto* m = world.createSoftBodyMaterial(1.0e4f, 0.45f, frictionFor(id));
        m->setDamping(1.0f);
        m->setElasticityDamping(0.025f);
        fishMats[key] = m;
        return m;
    };

    auto getFish = [&](int idx) -> const FishCache& {
        const auto& id = fishIds[idx];
        if (!fishCache.contains(id)) {
            std::cout << "Loading " << id << "..." << std::endl;
            fishCache[id] = loadFish(usdLoader, fishDir, id, paramFor(id).lengthM);
            std::cout << "  vertices: "
                      << fishCache[id].geometry->getAttribute<float>("position")->count()
                      << std::endl;
        }
        return fishCache.at(id);
    };

    auto dropFish = [&](const Vector3& pos) {
        const int idx = selectedFish;
        const auto& id = fishIds[idx];
        std::uniform_real_distribution<float> yawDist(-math::PI, math::PI);
        try {
            auto fish = spawnFish(world, scene, getFish(idx), pos,
                                  materialFor(id), yawDist(rng), voxelRes, solverIters, id,
                                  paramFor(id).weightKg);
            if (gpuSkin && tetSkinSupported && fish.body) fish.body->enableGpuSkinning();
            spawnedFish.push_back(fish);
        } catch (const std::exception& e) {
            std::cerr << "Spawn failed: " << e.what() << std::endl;
        }
    };

    // Drop one fish on each belt inlet to start; more stream in via auto-spawn.
    for (const auto& in : conveyor.inlets()) dropFish(in);

    bool autoSpawn = true;
    float spawnInterval = 2.5f;
    float spawnAccum = 0.f;
    int spawnInletIdx = 0;// round-robin over the belt inlets
    bool spawnPending = false;
    KeyAdapter spaceKey(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::SPACE) spawnPending = true;
    });
    canvas.addKeyListener(spaceKey);

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&capture);

    bool showDebug = false;
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float width = 300 * ui.dpiScale();
        ImGui::SetNextWindowPos({float(canvas.size().width()) - width, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({width, 0}, 0);
        ImGui::Begin("Fish Softbody");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Fish: %d", (int) spawnedFish.size());
        ImGui::Separator();
        if (ImGui::BeginCombo("Species", fishIds[selectedFish].c_str())) {
            for (int i = 0; i < static_cast<int>(fishIds.size()); ++i) {
                const bool isSelected = (i == selectedFish);
                if (ImGui::Selectable(fishIds[i].c_str(), isSelected)) {
                    selectedFish = i;
                }
                if (isSelected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        const FishParams& sel = paramFor(fishIds[selectedFish]);
        ImGui::Text("Species: %s", sel.species.empty() ? "(unknown)" : sel.species.c_str());
        ImGui::Text("Length: %.0f mm   Weight: %.2f kg", sel.lengthM * 1000.f, sel.weightKg);
        {
            const char* surfaces[] = {"Steel", "Perforated steel", "Plastic module belt"};
            ImGui::Combo("Belt surface", &beltSurface, surfaces, 3);
            const float fr = beltSurface == 0 ? sel.frictionSteel
                             : beltSurface == 1 ? sel.frictionPerfSteel
                                                : sel.frictionBelt;
            ImGui::Text("Friction (new fish): %.3f", fr);
        }
        ImGui::Separator();
        ImGui::Text("Belts: %d", conveyor.beltCount());
        ImGui::SliderFloat("Belt speed x", &conveyor.beltSpeedScale, 0.f, 3.f);
        ImGui::Checkbox("Auto-spawn fish", &autoSpawn);
        ImGui::SliderFloat("Spawn interval (s)", &spawnInterval, 0.3f, 6.f);
        ImGui::SliderInt("Voxel resolution", &voxelRes, 5, 30);
        ImGui::SliderInt("Solver iterations", &solverIters, 4, 30);
        if (!tetSkinSupported) ImGui::BeginDisabled();
        ImGui::Checkbox("GPU skinning", &gpuSkin);
        if (!tetSkinSupported) ImGui::EndDisabled();
        if (!tetSkinSupported) ImGui::TextDisabled("(GPU skinning unsupported on this renderer; using CPU skin)");
        ImGui::TextDisabled("(voxel/solver/GPU-skin apply to NEW fish)");
        ImGui::Separator();
        ImGui::Text("SPACE drops a fish where you aim");
        if (ImGui::Checkbox("Show PhysX debug", &showDebug)) {
            debugRenderer.visible = showDebug;
            if (showDebug) debugRenderer.enableDefaults();
        }
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;

    canvas.animate([&] {
        const float realDt = clock.getDelta();
        fpsAccum += realDt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (spawnPending) {
            spawnPending = false;
            Vector3 dir;
            dir.subVectors(controls.target, camera->position).normalize();
            Vector3 spawn;
            spawn.copy(camera->position).addScaledVector(dir, 3.f);
            spawn.y = std::max(spawn.y, 0.8f);
            dropFish(spawn);
        }

        // Stream fish onto the conveyor inlet.
        if (autoSpawn && !conveyor.inlets().empty()) {
            spawnAccum += realDt;
            if (spawnAccum >= spawnInterval) {
                spawnAccum = 0.f;
                std::uniform_real_distribution<float> j(-0.1f, 0.1f);
                Vector3 sp = conveyor.inlets()[spawnInletIdx % conveyor.inlets().size()];
                ++spawnInletIdx;
                sp.x += j(rng);
                sp.z += j(rng);
                dropFish(sp);
            }
        }

        world.step(realDt);
        debugRenderer.update();

        // Despawn fish that have ridden off the end of the system.
        for (auto it = spawnedFish.begin(); it != spawnedFish.end();) {
            auto* geom = it->body->visualGeometry().get();
            if (geom->boundingSphere && geom->boundingSphere->center.y < -2.f) {
                world.removeSoftBody(it->body);
                it = spawnedFish.erase(it);
            } else {
                ++it;
            }
        }

        // Tick the conveyor visuals (belt texture scroll, roller spin, cleat meshes).
        conveyor.update(realDt);

        renderer->render(scene, *camera);
#ifdef THREEPP_PHYSX_CUDA_GL_INTEROP
        // Once the renderer has uploaded a fish's tet texture, hand its GL handle to
        // the soft body so subsequent frames feed it via CUDA-GL interop
        // (device->device) instead of the GPU->CPU->GPU bridge.
        if (auto* glr = dynamic_cast<GLRenderer*>(renderer.get())) {
            for (auto& f : spawnedFish) {
                if (f.body && f.body->needsInteropRegister()) {
                    if (auto* tex = f.body->interopTexture()) {
                        if (auto id = glr->getGlTextureId(*tex)) {
                            f.body->registerGlTexture(*id);
                        }
                    }
                }
            }
        }
#endif
        ui.render();
    });
}
