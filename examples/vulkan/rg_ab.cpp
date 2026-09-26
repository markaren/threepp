// rg_ab — A/B capture harness for the Vulkan render graph (plans/render-graph.md,
// "Verification method").
//
// Renders one of a fixed set of scenes headless, 640x400, 48 frames, with
// setSimTime pinned to frame/60 and auto-LOD off (its background simplification
// jobs land at a timing-dependent frame), then writes raw readbacks into
// --out <dir>. A change that only moves barriers, passes or memory placement
// must leave every byte equal to a build without it. The scene definitions
// below are the baseline contract: changing one invalidates every baseline
// captured with the old definition.
//
//   rg_ab --scene room --probe 0 --out aaa_caps/rg/new/room
//   rg_ab --compare aaa_caps/rg/base/room aaa_caps/rg/new/room
//   rg_ab --selfcheck --scene fields_views     (ctest rg_ab_<scene>)
//
// Scenes
//   room             closed box, six spheres (diffuse, gold, mirror, glass,
//                    clearcoat, emissive), rotating metal knot, alpha-tested
//                    textured cutout, point + spot light, emissive ceiling panel
//   room_fx          room + 4x G-buffer MSAA, depth of field, motion blur,
//                    render scale 0.75, lens distortion, sensor noise, a
//                    256x256 secondary view, moving camera; DLSS/FSR left at
//                    the renderer defaults (on where built)
//   room_dlss        room at render scale 0.67 with DLSS
//   room_fsr         room at render scale 0.67 with FSR
//   room_debug       room in the hybrid debug view (normals) with a composited
//                    128x128 secondary view
//   room_tail        room + scene capture, event camera (Shaded source) and a
//                    composited secondary view, moving camera
//   room_tail_final  room_tail with the event camera on the Final source
//   room_events      room_tail in events-only mode
//   overlays         room + line, points, wireframe box and a transparent unlit
//                    box (the overlay pass)
//   outdoor          ocean, clouds, height fog, distance fog, fire, a 20k splat
//                    cloud, rocks
//   deform           skinned glTF (GPU skinning + BLAS refit), morph targets,
//                    GPU grass wind, rotating knot (TLAS refit)
//   fields           fire embers with glow, a density volume and sun-occlusion
//                    rays, an 8k splat cloud (splat depth stamp), lens
//                    distortion, sensor noise, a secondary view
//   fields_views     fields scene with two secondary views: one with splats (a
//                    second sort), one composited into the frame
// Unless noted, DLSS and FSR are off. Assets: DATA_FOLDER/textures/three.png,
// models/gltf/Soldier.glb, models/gltf/AnimatedMorphSphere.
//
// Outputs (only those the scene produces): final.rgb, scene_hdr.bin,
// aov_{depth,normal,motion,ids,albedo}.bin, view2.rgb, view3.rgb,
// scene_capture.rgb, event_vis.rgba, event_stream.bin. Printed: a MEM line
// (gpuAllocatedBytes, transientImageBytes), an EVENTS line, the validation
// error/warning counts and the render-graph diagnostic count.
//
// Flags
//   --probe 0        probe GI off. Required for A/B: probe GI is
//                    nondeterministic at baseline (the probe depth store, i.e.
//                    the probe ray hit distances, differs run to run; it
//                    persists with full barriers and an idle device every
//                    frame; root cause open). The room_dlss and room/overlays
//                    modes below were bisected to it. Measured 2026-09-26 on
//                    room: 1 run in 119 on an RTX 4070 (driver 595.97), about
//                    1 in 2 on an RTX 4060 laptop; the divergent run's
//                    final.rgb and AOVs were byte-identical, only the
//                    RG_AUDIT_END probeSh/probeDepth hashes differed, so
//                    --compare cannot see it. To catch one: RG_AUDIT_END +
//                    RG_PROBE_DUMP per run where it is frequent, then diff two
//                    dumps' stores probe by probe.
//   --occl 1         occlusion culling          --msaa N   G-buffer MSAA
//   --aa N           overlay pass sample count (the Canvas antialiasing; default 4)
//   --noviews 1      skip addView (the primary alone)
//   --static 1       knot does not rotate
//   --frames N, --w W, --h H, --autolod 1, --autoexp 0,
//   --restir 0|1, --ao 0|1, --denoise 0|1
//   --time N         after the run, N more frames; prints the median GPU frame
//                    time and CPU record time (first 8 dropped); no files
//   --audit <file>   per-frame debugHashShadeImages() hashes
//   --selfcheck      no --out; probe GI off unless --probe is given; renders
//                    and exits with the render-graph verdict
// Environment
//   RG_AUDIT_END=<file>    the shade-image hashes after the last frame
//   RG_PROBE_DUMP=<dir>    probe_sh.bin and probe_depth.bin (the probe GI
//                          stores) after the last frame
//   RG_SERIALIZE=1         idle the device after every frame (a readback)
//   THREEPP_RG_DUMP=1      print the frame graph (passes, barriers) to stderr
//                          whenever its shape changes
//   THREEPP_VK_NO_ALIAS=1  every transient image gets its own allocation.
//                          readSceneHdrDebug returns false for a pooled
//                          sceneHdr, so scene_hdr.bin is written only with it.
//
// Exit codes: 0 ok; 1 --compare found a difference; 2 usage; 3 the render
// graph reported a diagnostic (VulkanRenderer::renderGraphDiagnosticCount():
// a pass declaring one image in two layouts, or aliased images in use at the
// same time); 42 no Vulkan/RT device (ctest SKIP_RETURN_CODE).
//
// Baseline workflow
//   1. Build rg_ab at the parent commit (git stash, or a worktree), copy
//      bin/rg_ab.exe to bin/rg_ab_base.exe (threepp links statically, so the
//      copy is self-contained).
//   2. Build rg_ab with the change.
//   3. Per scene: rg_ab_base --scene S --probe 0 --out aaa_caps/rg/base/S,
//      rg_ab --scene S --probe 0 --out aaa_caps/rg/new/S,
//      rg_ab --compare aaa_caps/rg/base/S aaa_caps/rg/new/S.
//   Repeat a scene that differs, alternating the two exes, before treating a
//   difference as caused by the change: see the modes below.
//
// Known run-to-run modes (present at baseline, both exes)
//   room_dlss   two outputs, about 50/50, with probe GI on.
//   room, overlays  with probe GI on, about half the runs land on one of two
//               other modes: 419 or 1816 sceneHdr bytes at 1 fp16 ULP, <= 1/255
//               in final.rgb.
//   fields      about 1 run in 5: 186 px over the splat cloud differ by 1/255
//               in final.rgb (last-frame sceneHdr identical, so through TAA
//               history). Suspected: the SplatPass tile-expansion budget grows
//               from a GPU readback, so which early frame is truncated depends
//               on timing.
//   deform      diverges run to run under CPU load (other processes busy);
//               unloaded runs are bit-exact.
//
// Sync validation: THREEPP_VULKAN_VALIDATION=1 and VK_LAYER_SETTINGS_PATH
// pointing at a vk_layer_settings.txt (or its directory) that contains
//   khronos_validation.validate_sync = true
// The SYNC-HAZARD messages count as validation errors in the printed total;
// the count must not rise from baseline to change.

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/extras/effects/FireEffect.hpp"
#include "threepp/extras/vegetation/GrassTiles.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/objects/Ocean.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/splats/SplatData.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace threepp;
namespace fs = std::filesystem;

namespace {

    void writeFile(const fs::path& p, const void* data, size_t n) {
        std::ofstream f(p, std::ios::binary);
        f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    }

    std::vector<uint8_t> readFile(const fs::path& p) {
        std::ifstream f(p, std::ios::binary);
        return {std::istreambuf_iterator<char>(f), {}};
    }

    int compareDirs(const fs::path& a, const fs::path& b) {
        int bad = 0, files = 0;
        for (const auto& e : fs::recursive_directory_iterator(a)) {
            if (!e.is_regular_file()) continue;
            const auto rel = fs::relative(e.path(), a);
            const auto pb  = b / rel;
            ++files;
            if (!fs::exists(pb)) {
                std::printf("  MISSING  %s\n", rel.string().c_str());
                ++bad;
                continue;
            }
            const auto da = readFile(e.path()), db = readFile(pb);
            if (da.size() != db.size()) {
                std::printf("  SIZE     %s (%zu vs %zu)\n", rel.string().c_str(), da.size(), db.size());
                ++bad;
                continue;
            }
            size_t diff = 0;
            int maxD = 0;
            for (size_t i = 0; i < da.size(); ++i) {
                if (da[i] != db[i]) {
                    ++diff;
                    maxD = std::max(maxD, std::abs(int(da[i]) - int(db[i])));
                }
            }
            if (diff) {
                std::printf("  DIFF     %s: %zu of %zu bytes, max byte delta %d\n", rel.string().c_str(),
                            diff, da.size(), maxD);
                ++bad;
            } else {
                std::printf("  same     %s\n", rel.string().c_str());
            }
        }
        std::printf("%d of %d files differ\n", bad, files);
        return bad == 0 ? 0 : 1;
    }

    auto stdMat(const Color& c, float rough, float metal = 0.f) {
        return MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(c).roughness(rough).metalness(metal));
    }

    void buildRoom(Scene& scene) {
        scene.background = Color::black;
        constexpr float S = 10.f;
        auto white = stdMat(Color(0.73f, 0.73f, 0.73f), 0.95f);
        auto floor = Mesh::create(PlaneGeometry::create(S, S), white);
        floor->rotation.x = -math::PI / 2.f;
        scene.add(floor);
        auto back = Mesh::create(PlaneGeometry::create(S, S), white);
        back->position.set(0.f, S / 2.f, -S / 2.f);
        scene.add(back);
        auto left = Mesh::create(PlaneGeometry::create(S, S), stdMat(Color(0.65f, 0.05f, 0.05f), 0.95f));
        left->rotation.y = math::PI / 2.f;
        left->position.set(-S / 2.f, S / 2.f, 0.f);
        scene.add(left);
        auto right = Mesh::create(PlaneGeometry::create(S, S), stdMat(Color(0.12f, 0.45f, 0.15f), 0.95f));
        right->rotation.y = -math::PI / 2.f;
        right->position.set(S / 2.f, S / 2.f, 0.f);
        scene.add(right);
        auto ceiling = Mesh::create(PlaneGeometry::create(S, S), white);
        ceiling->rotation.x = math::PI / 2.f;
        ceiling->position.y = S;
        scene.add(ceiling);

        auto panelMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::white).emissive(Color::white).emissiveIntensity(18.f).roughness(1.f));
        auto panel = Mesh::create(PlaneGeometry::create(3.f, 3.f), panelMat);
        panel->rotation.x = math::PI / 2.f;
        panel->position.set(0.f, 9.99f, 0.f);
        scene.add(panel);

        auto sphere = SphereGeometry::create(0.55f, 48, 32);
        const float dx = 1.4f, x0 = -dx * 2.5f;
        std::vector<std::shared_ptr<Material>> mats;
        mats.push_back(stdMat(Color(0.85f, 0.85f, 0.85f), 0.95f));
        mats.push_back(stdMat(Color(1.f, 0.78f, 0.32f), 0.25f, 1.f));
        mats.push_back(stdMat(Color(0.97f, 0.97f, 0.97f), 0.02f, 1.f));
        {
            auto glass = MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}.color(Color::white).transmission(1.f).roughness(0.f).metalness(0.f));
            glass->setIor(1.5f);
            mats.push_back(glass);
        }
        mats.push_back(MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}.color(Color(0.65f, 0.05f, 0.05f)).roughness(0.4f).clearcoat(1.f).clearcoatRoughness(0.05f)));
        mats.push_back(MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::black).emissive(Color(0.1f, 0.85f, 1.f)).emissiveIntensity(4.f).roughness(1.f)));
        for (size_t i = 0; i < mats.size(); ++i) {
            auto m = Mesh::create(sphere, mats[i]);
            m->position.set(x0 + static_cast<float>(i) * dx, 0.55f, 2.5f);
            scene.add(m);
        }

        auto knot = Mesh::create(TorusKnotGeometry::create(0.55f, 0.18f, 128, 24),
                                 MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}.color(Color(0.95f, 0.93f, 0.88f)).roughness(0.15f).metalness(1.f)));
        knot->name = "knot";
        knot->position.set(0.f, 4.5f, -1.f);
        scene.add(knot);

        TextureLoader tl;
        auto tex = tl.load(std::string(DATA_FOLDER) + "/textures/three.png", SRGBColorSpace);
        auto cutMat = stdMat(Color::white, 0.6f);
        cutMat->map = tex;
        cutMat->alphaTest = 0.5f;
        cutMat->side = Side::Double;
        auto cut = Mesh::create(PlaneGeometry::create(2.f, 2.f), cutMat);
        cut->position.set(2.5f, 2.f, -2.f);
        scene.add(cut);

        auto point = PointLight::create(Color(1.f, 0.6f, 0.3f), 8.f, 12.f);
        point->position.set(-3.f, 3.f, 1.f);
        scene.add(point);
        auto spot = SpotLight::create(Color(0.4f, 0.6f, 1.f), 30.f, 20.f, math::PI / 7.f, 0.3f);
        spot->position.set(3.f, 7.f, 3.f);
        scene.add(spot);
    }

    void buildOutdoor(Scene& scene, VulkanRenderer& renderer) {
        scene.background = Color(0.45f, 0.6f, 0.85f);
        scene.fog = Fog(Color(0.6f, 0.7f, 0.85f), 30.f, 600.f);
        auto sun = DirectionalLight::create(Color::white, 3.f);
        sun->position.set(-40.f, 30.f, 20.f);
        scene.add(sun);
        scene.add(HemisphereLight::create(0xbfd4ff, 0x404030, 0.6f));

        Ocean::Options opts;
        opts.size = 400.f;
        auto ocean = Ocean::create(opts);
        scene.add(ocean);

        auto rockMat = stdMat(Color(0.35f, 0.33f, 0.3f), 0.9f);
        for (int i = 0; i < 5; ++i) {
            auto rock = Mesh::create(SphereGeometry::create(1.2f + 0.3f * i, 24, 18), rockMat);
            rock->position.set(-8.f + 4.f * i, 0.2f, -10.f - 3.f * i);
            scene.add(rock);
        }

        FireEffect::Params fp;
        auto fire = FireEffect::create(fp);
        fire->name = "fire";
        fire->position.set(0.f, 1.5f, -6.f);
        fire->ignite();
        scene.add(fire);

        SplatGenerator::Options o;
        o.count = 20000;
        o.seed = 909u;
        o.extent.set(3.f, 2.f, 3.f);
        auto splats = SplatCloud::create(SplatGenerator::generate(o));
        splats->position.set(6.f, 2.f, -8.f);
        scene.add(splats);

        VulkanRenderer::CloudSettings cs;
        renderer.setClouds(cs);
        VulkanRenderer::HeightFogSettings hf;
        renderer.setHeightFog(hf);
    }

    // Skinned (GPU skinning + BLAS refit), morph targets (CPU blend + BLAS
    // rebuild), GPU grass wind (deform + refit) and a rotating knot (TLAS
    // refit): the deformer head of the frame.
    std::vector<std::unique_ptr<AnimationMixer>> buildDeform(Scene& scene) {
        std::vector<std::unique_ptr<AnimationMixer>> mixers;
        scene.background = Color(0.35f, 0.45f, 0.6f);
        auto sun = DirectionalLight::create(Color::white, 3.f);
        sun->position.set(3.f, 10.f, 6.f);
        scene.add(sun);
        scene.add(HemisphereLight::create(0xffffff, 0x404040, 0.8f));
        auto ground = Mesh::create(PlaneGeometry::create(20.f, 20.f), stdMat(Color(0.4f, 0.38f, 0.33f), 0.9f));
        ground->rotation.x = -math::PI / 2.f;
        scene.add(ground);

        GLTFLoader loader;
        for (const char* path : {"/models/gltf/Soldier.glb", "/models/gltf/AnimatedMorphSphere/AnimatedMorphSphere.gltf"}) {
            auto res = loader.load(std::string(DATA_FOLDER) + path);
            if (!res) continue;
            auto model = res->scene;
            if (std::string(path).find("Morph") != std::string::npos) model->position.set(2.f, 1.f, 0.f);
            scene.add(model);
            if (!res->animations.empty()) {
                auto mixer = std::make_unique<AnimationMixer>(*model);
                mixer->clipAction(res->animations.front())->play();
                mixers.push_back(std::move(mixer));
            }
        }

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        std::vector<vegetation::GrassBlade> blades;
        const Vector3 up{0.f, 1.f, 0.f};
        for (int i = 0; i < 4000; ++i) {
            vegetation::GrassBlade bl;
            bl.position.set(-4.f + 3.f * u01(rng), 0.f, -1.f + 3.f * u01(rng));
            bl.scale.set(0.6f, 0.3f + 0.3f * u01(rng), 0.6f);
            bl.yaw.setFromAxisAngle(up, u01(rng) * math::TWO_PI);
            blades.push_back(bl);
        }
        auto grassMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.42f, 0.5f, 0.26f)).roughness(1.f));
        grassMat->vertexColors = true;
        grassMat->side = Side::Double;
        auto grass = GrassMesh::create(vegetation::buildGrassGeometry(blades), grassMat);
        grass->params.windStrength = 0.3f;
        grass->params.maxAnimDistance = 0.f;
        scene.add(grass);

        auto knot = Mesh::create(TorusKnotGeometry::create(0.4f, 0.12f, 96, 16), stdMat(Color(0.9f, 0.9f, 0.9f), 0.2f, 1.f));
        knot->name = "knot";
        knot->position.set(-2.f, 2.5f, 1.f);
        scene.add(knot);
        return mixers;
    }

    // The tail: field billboards with glow, the volumetric transmittance
    // prepass with its sun-occlusion ray queries, a secondary view that draws
    // them for its own eye, and a rotating knot so the TLAS refits every frame.
    // main() adds lens distortion + sensor noise after the overlay.
    void buildFields(Scene& scene) {
        scene.background = Color(0.05f, 0.06f, 0.1f);
        auto sun = DirectionalLight::create(Color::white, 2.f);
        sun->position.set(-4.f, 8.f, 3.f);
        scene.add(sun);
        scene.add(HemisphereLight::create(0x8090b0, 0x202020, 0.4f));
        auto ground = Mesh::create(PlaneGeometry::create(30.f, 30.f), stdMat(Color(0.3f, 0.3f, 0.3f), 0.9f));
        ground->rotation.x = -math::PI / 2.f;
        scene.add(ground);
        auto wall = Mesh::create(BoxGeometry::create(4.f, 3.f, 0.3f), stdMat(Color(0.5f, 0.45f, 0.4f), 0.8f));
        wall->position.set(-2.f, 1.5f, -2.f);
        scene.add(wall);
        auto knot = Mesh::create(TorusKnotGeometry::create(0.4f, 0.12f, 96, 16), stdMat(Color(0.9f, 0.9f, 0.9f), 0.2f, 1.f));
        knot->name = "knot";
        knot->position.set(-1.5f, 2.5f, 0.5f);
        scene.add(knot);

        FireEffect::Params fp;
        auto fire = FireEffect::create(fp);
        fire->name = "fire";
        if (const auto& embers = fire->emberField()) {
            // A density volume makes the embers march it (the transmittance
            // prepass); sunGeometryShadow adds its ray queries.
            embers->setDensityRepr(Vector3(0.f, 1.5f, 0.f), Vector3(1.5f, 2.f, 1.5f), 0.02f, 32);
            auto& br = embers->billboardRepr();
            br.volumeExtinction  = 1.f;
            br.volumeShadow      = 0.8f;
            br.sunGeometryShadow = true;
        }
        fire->ignite();
        scene.add(fire);

        // A cloud plus overlay content (the embers) latches the splat depth AOV
        // on, so the overlay pass records the splat depth stamp.
        SplatGenerator::Options o;
        o.count = 8000;
        o.seed = 404u;
        o.extent.set(1.f, 1.f, 1.f);
        auto splats = SplatCloud::create(SplatGenerator::generate(o));
        splats->position.set(2.f, 1.f, -1.f);
        scene.add(splats);
    }

    void buildOverlays(Scene& scene) {
        buildRoom(scene);
        // Lines, points, wireframe and a transparent unlit mesh: the overlay pass.
        std::vector<float> pts;
        for (int i = 0; i < 64; ++i) {
            const float t = static_cast<float>(i) / 64.f * math::TWO_PI;
            pts.insert(pts.end(), {3.f * std::cos(t), 2.f + 0.5f * std::sin(3.f * t), 3.f * std::sin(t)});
        }
        auto lineGeom = BufferGeometry::create();
        lineGeom->setAttribute("position", FloatBufferAttribute::create(pts, 3));
        scene.add(Line::create(lineGeom, LineBasicMaterial::create({{"color", Color::yellow}})));
        auto pm = PointsMaterial::create();
        pm->size = 6.f;
        pm->sizeAttenuation = false;
        pm->color = Color::magenta;
        scene.add(Points::create(lineGeom, pm));
        auto wire = MeshBasicMaterial::create();
        wire->wireframe = true;
        wire->color = Color::cyan;
        auto wm = Mesh::create(BoxGeometry::create(1.5f, 1.5f, 1.5f), wire);
        wm->position.set(-2.5f, 2.5f, 0.f);
        scene.add(wm);
        auto glassy = MeshBasicMaterial::create();
        glassy->transparent = true;
        glassy->opacity = 0.4f;
        glassy->color = Color::orange;
        auto gm = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f), glassy);
        gm->position.set(2.5f, 1.f, 1.f);
        scene.add(gm);
    }

    constexpr int kGraphDiagnosticsExit = 3;
    constexpr int kSkipCode = 42;

    // Prints the render-graph diagnostic count (and the distinct messages);
    // returns the exit code: an A/B run with a diagnostic does not pass.
    int graphVerdict(const VulkanRenderer& renderer, const std::string& sceneName) {
        const auto n = renderer.renderGraphDiagnosticCount();
        std::printf("GRAPH %s diagnostics %u\n", sceneName.c_str(), n);
        for (const auto& m : renderer.renderGraphDiagnostics()) std::printf("  %s\n", m.c_str());
        return n == 0 ? 0 : kGraphDiagnosticsExit;
    }

}// namespace

int main(int argc, char** argv) {
    std::string sceneName = "room", outDir, cmpA, cmpB;
    int autoLod = 0, autoExp = 1, timeFrames = 0, aa = -1;
    bool noViews = false;// --noviews 1: skip addView (the primary alone)
    bool selfCheck = false;
    int probe = -1, restir = -1, ao = -1, denoise = -1;// -1 = renderer default
    int occl = 0, msaa = 0;
    std::string auditPath;
    int staticScene = 0;
    int frames = 48, W = 640, H = 400;
    for (int i = 1; i < argc; ++i) {
        auto val = [&](const char* f) -> const char* { return (std::strcmp(argv[i], f) == 0 && i + 1 < argc) ? argv[++i] : nullptr; };
        if (const char* s = val("--scene")) sceneName = s;
        else if (const char* s = val("--out")) outDir = s;
        else if (const char* s = val("--frames")) frames = std::atoi(s);
        else if (const char* s = val("--w")) W = std::atoi(s);
        else if (const char* s = val("--h")) H = std::atoi(s);
        else if (const char* s = val("--autolod")) autoLod = std::atoi(s);
        else if (const char* s = val("--autoexp")) autoExp = std::atoi(s);
        else if (const char* s = val("--probe")) probe = std::atoi(s);
        else if (const char* s = val("--restir")) restir = std::atoi(s);
        else if (const char* s = val("--ao")) ao = std::atoi(s);
        else if (const char* s = val("--denoise")) denoise = std::atoi(s);
        else if (const char* s = val("--occl")) occl = std::atoi(s);
        else if (const char* s = val("--audit")) auditPath = s;
        else if (const char* s = val("--static")) staticScene = std::atoi(s);
        else if (const char* s = val("--msaa")) msaa = std::atoi(s);
        else if (const char* s = val("--time")) timeFrames = std::atoi(s);
        else if (const char* s = val("--aa")) aa = std::atoi(s);
        else if (const char* s = val("--noviews")) noViews = std::atoi(s) != 0;
        else if (std::strcmp(argv[i], "--compare") == 0 && i + 2 < argc) { cmpA = argv[++i]; cmpB = argv[++i]; }
        else if (std::strcmp(argv[i], "--selfcheck") == 0) selfCheck = true;
    }
    if (!cmpA.empty()) return compareDirs(cmpA, cmpB);
    if (selfCheck && probe < 0) probe = 0;
    if (outDir.empty() && !selfCheck) { std::printf("--out <dir> required\n"); return 2; }

    std::unordered_map<std::string, Canvas::ParameterValue> canvasParams{
            {"vsync", false}, {"size", WindowSize{W, H}}, {"headless", true}};
    // The Canvas's antialiasing is the overlay pass's sample count; -1 keeps
    // its default (4).
    if (aa >= 0) canvasParams["antialiasing"] = aa;
    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr   = std::make_unique<Canvas>("rg_ab", canvasParams);
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    Scene scene;
    PerspectiveCamera camera(55.f, static_cast<float>(W) / H, 0.1f, 2000.f);
    camera.position.set(0.f, 3.5f, 11.f);
    camera.lookAt(Vector3{0.f, 2.5f, 0.f});
    std::unique_ptr<PerspectiveCamera> camera2;
    uint32_t view2 = 0;
    std::unique_ptr<PerspectiveCamera> camera3;
    uint32_t view3 = 0;

    const bool room = sceneName.rfind("room", 0) == 0;
    std::vector<std::unique_ptr<AnimationMixer>> mixers;
    if (room) buildRoom(scene);
    else if (sceneName == "deform") {
        mixers = buildDeform(scene);
        camera.position.set(0.f, 2.f, 6.f);
        camera.lookAt(Vector3{0.f, 1.f, 0.f});
    }
    else if (sceneName == "outdoor") {
        buildOutdoor(scene, renderer);
        camera.position.set(0.f, 6.f, 14.f);
        camera.lookAt(Vector3{0.f, 1.f, -10.f});
    } else if (sceneName == "overlays") buildOverlays(scene);
    else if (sceneName == "fields" || sceneName == "fields_views") {
        buildFields(scene);
        camera.position.set(0.f, 2.f, 6.f);
        camera.lookAt(Vector3{0.f, 1.2f, 0.f});
    }
    else { std::printf("unknown scene %s\n", sceneName.c_str()); return 2; }

    if (sceneName == "room_fx") {
        renderer.setGbufferMsaa(4);
        renderer.setDepthOfField(true);
        renderer.setFocusDistance(8.f);
        renderer.setMotionBlur(0.5f);
        renderer.setRenderScale(0.75f);
        LensDistortion barrel;
        barrel.model = LensModel::BrownConrady;
        barrel.k1 = -0.2f;
        renderer.setLensDistortion(barrel);
        VulkanRenderer::SensorNoise sn;
        sn.enabled = true;
        sn.seed = 7u;
        renderer.setSensorNoise(sn);
        camera2 = std::make_unique<PerspectiveCamera>(60.f, 1.f, 0.1f, 100.f);
        camera2->position.set(-4.f, 2.f, 6.f);
        camera2->lookAt(Vector3{0.f, 1.f, 0.f});
        view2 = noViews ? 0u : renderer.addView(*camera2, 256, 256);
    } else if (sceneName == "fields_views") {
        // Two secondary views: one with splats (a second sort over the shared
        // scratch), one composited into the frame.
        renderer.setDlss(false);
        renderer.setFsr(false);
        camera2 = std::make_unique<PerspectiveCamera>(60.f, 1.f, 0.1f, 100.f);
        camera2->position.set(-3.f, 2.f, 4.f);
        camera2->lookAt(Vector3{0.f, 1.f, 0.f});
        view2 = noViews ? 0u : renderer.addView(*camera2, 256, 256);
        renderer.setViewSplats(view2, true);
        camera3 = std::make_unique<PerspectiveCamera>(50.f, 1.5f, 0.1f, 100.f);
        camera3->position.set(3.f, 1.5f, 5.f);
        camera3->lookAt(Vector3{0.f, 1.f, 0.f});
        view3 = noViews ? 0u : renderer.addView(*camera3, 192, 128);
        renderer.setViewDisplayRect(view3, 400, 240, 192, 128);
    } else if (sceneName == "room_debug") {
        // The hybrid debug view (normals) with a secondary view behind it.
        renderer.setDlss(false);
        renderer.setFsr(false);
        renderer.setHybridDebugView(1);
        camera2 = std::make_unique<PerspectiveCamera>(60.f, 1.f, 0.1f, 100.f);
        camera2->position.set(-4.f, 2.f, 6.f);
        camera2->lookAt(Vector3{0.f, 1.f, 0.f});
        view2 = noViews ? 0u : renderer.addView(*camera2, 128, 128);
        renderer.setViewDisplayRect(view2, 24, 16, 128, 128);
    } else if (sceneName == "fields") {
        renderer.setDlss(false);
        renderer.setFsr(false);
        LensDistortion barrel;
        barrel.model = LensModel::BrownConrady;
        barrel.k1 = -0.15f;
        renderer.setLensDistortion(barrel);
        VulkanRenderer::SensorNoise sn;
        sn.enabled = true;
        sn.seed = 11u;
        renderer.setSensorNoise(sn);
        camera2 = std::make_unique<PerspectiveCamera>(60.f, 1.f, 0.1f, 100.f);
        camera2->position.set(-3.f, 2.f, 4.f);
        camera2->lookAt(Vector3{0.f, 1.f, 0.f});
        view2 = noViews ? 0u : renderer.addView(*camera2, 256, 256);
    } else if (sceneName == "room_tail" || sceneName == "room_tail_final" || sceneName == "room_events") {
        // The post-view tail: scene capture, the event camera (Shaded or Final
        // source; events-only mode), a secondary view composited into the frame.
        renderer.setDlss(false);
        renderer.setFsr(false);
        renderer.setSceneCaptureEnabled(true);
        renderer.setEventCameraEnabled(true);
        renderer.setEventCameraResolution(160, 100);
        if (sceneName == "room_tail_final") renderer.setEventCameraSource(VulkanRenderer::EventCameraSource::Final);
        if (sceneName == "room_events") renderer.setEventsOnlyMode(true);
        camera2 = std::make_unique<PerspectiveCamera>(60.f, 1.f, 0.1f, 100.f);
        camera2->position.set(-4.f, 2.f, 6.f);
        camera2->lookAt(Vector3{0.f, 1.f, 0.f});
        view2 = noViews ? 0u : renderer.addView(*camera2, 128, 128);
        renderer.setViewDisplayRect(view2, 24, 16, 128, 128);
    } else if (sceneName == "room_dlss") {
        renderer.setRenderScale(0.67f);
        renderer.setDlss(true);
    } else if (sceneName == "room_fsr") {
        renderer.setRenderScale(0.67f);
        renderer.setDlss(false);
        renderer.setFsr(true);
    } else {
        renderer.setDlss(false);
        renderer.setFsr(false);
    }
    renderer.setAutoLod(autoLod != 0);
    if (denoise >= 0) renderer.setDenoise(denoise != 0);
    if (occl) renderer.setOcclusionCulling(true);
    if (msaa > 1) renderer.setGbufferMsaa(static_cast<uint32_t>(msaa));
    if (ao >= 0) renderer.setDeferredAO(ao != 0);
    if (probe >= 0) renderer.setProbeGI(probe != 0);
    if (restir >= 0) renderer.setRestirDIEnabled(restir != 0);
    renderer.setAutoExposure(autoExp != 0);
    renderer.setBloomIntensity(0.3f);

    auto knot = scene.getObjectByName("knot");
    auto fireObj = scene.getObjectByName("fire");
    auto fire = fireObj ? std::dynamic_pointer_cast<FireEffect>(fireObj->shared_from_this()) : nullptr;

    for (int i = 0; i < frames; ++i) {
        const double t = i / 60.0;
        renderer.setSimTime(t);
        if (knot && !staticScene) knot->rotation.y = static_cast<float>(t * 1.3);
        if (fire) fire->update(static_cast<float>(t));
        for (auto& m : mixers) m->update(1.f / 60.f);
        if (sceneName == "room_fx" || sceneName.rfind("room_tail", 0) == 0 || sceneName == "room_events") {
            camera.position.x = static_cast<float>(std::sin(t * 2.0) * 1.5);
            camera.lookAt(Vector3{0.f, 2.5f, 0.f});
        }
        canvas.animateOnce([&] { renderer.render(scene, camera); });
        if (std::getenv("RG_SERIALIZE")) (void) renderer.readRGBPixels();// idles the device
        if (!auditPath.empty()) {
            static std::ofstream audit(auditPath);
            for (const auto& [name, h] : renderer.debugHashShadeImages())
                audit << i << ' ' << name << ' ' << std::hex << h << std::dec << '\n';
        }
    }

    if (const char* e = std::getenv("RG_AUDIT_END")) {
        std::ofstream audit(e);
        for (const auto& [name, h] : renderer.debugHashShadeImages())
            audit << name << ' ' << std::hex << h << std::dec << '\n';
    }
    if (const char* e = std::getenv("RG_PROBE_DUMP")) {
        std::vector<uint8_t> sh, depth;
        if (renderer.readProbeShDebug(sh, &depth)) {
            const fs::path d(e);
            fs::create_directories(d);
            writeFile(d / "probe_sh.bin", sh.data(), sh.size());
            writeFile(d / "probe_depth.bin", depth.data(), depth.size());
        }
    }

    if (timeFrames > 0) {
        std::vector<float> gpu, rec;
        for (int i = 0; i < timeFrames; ++i) {
            renderer.setSimTime((frames + i) / 60.0);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
            const auto t = renderer.lastFrameTimings();
            if (i >= 8) { gpu.push_back(t.gpuTotalMs); rec.push_back(t.cpuRecordMs); }
        }
        std::sort(gpu.begin(), gpu.end());
        std::sort(rec.begin(), rec.end());
        std::printf("TIME %s gpu_median_ms %.3f record_median_ms %.3f\n", sceneName.c_str(),
                    gpu[gpu.size() / 2], rec[rec.size() / 2]);
        return graphVerdict(renderer, sceneName);
    }

    if (selfCheck) {
        std::printf("%s: %d frames (%dx%d), validation errors %u warnings %u (active %d)\n",
                    sceneName.c_str(), frames, W, H, vulkan::validationErrorCount(),
                    vulkan::validationWarningCount(), vulkan::validationActive() ? 1 : 0);
        return graphVerdict(renderer, sceneName);
    }

    fs::create_directories(outDir);
    const fs::path out(outDir);
    const auto rgb = renderer.readRGBPixels();
    writeFile(out / "final.rgb", rgb.data(), rgb.size());
    std::vector<uint8_t> hdr;
    int hw = 0, hh = 0;
    if (renderer.readSceneHdrDebug(hdr, hw, hh)) writeFile(out / "scene_hdr.bin", hdr.data(), hdr.size());
    std::vector<VulkanRenderer::AOVReadback> aovs;
    using A = VulkanRenderer::GBufferAOV;
    const char* names[] = {"depth", "normal", "motion", "ids", "albedo"};
    if (renderer.readGBufferAOVs({A::Depth, A::Normal, A::Motion, A::Ids, A::Albedo}, aovs)) {
        for (const auto& r : aovs) writeFile(out / (std::string("aov_") + names[static_cast<int>(r.aov)] + ".bin"), r.data.data(), r.data.size());
    }
    if (view2) {
        const auto v = renderer.readViewRGBPixels(view2);
        writeFile(out / "view2.rgb", v.data(), v.size());
    }
    if (view3) {
        const auto v = renderer.readViewRGBPixels(view3);
        writeFile(out / "view3.rgb", v.data(), v.size());
    }
    if (renderer.sceneCaptureEnabled()) {
        const auto s = renderer.readSceneRGBPixels();
        writeFile(out / "scene_capture.rgb", s.data(), s.size());
    }
    if (renderer.eventCameraEnabled()) {
        const auto vis = renderer.readEventCameraVisualisation();
        writeFile(out / "event_vis.rgba", vis.data(), vis.size());
        std::vector<VulkanRenderer::Event> ev(1u << 20);
        bool over = false;
        const size_t n = renderer.readEventStreamInto(ev.data(), ev.size(), &over);
        writeFile(out / "event_stream.bin", ev.data(), n * sizeof(VulkanRenderer::Event));
        std::printf("EVENTS %zu overflow %d\n", n, over ? 1 : 0);
    }
    {
        std::uint64_t shared = 0, unshared = 0;
        renderer.transientImageBytes(shared, unshared);
        std::printf("MEM %s gpu_allocated_mib %.1f transient_shared_mib %.1f transient_unshared_mib %.1f\n",
                    sceneName.c_str(), renderer.gpuAllocatedBytes() / 1048576.0, shared / 1048576.0,
                    unshared / 1048576.0);
    }
    std::printf("%s: wrote %zu B final (%dx%d), validation errors %u warnings %u (active %d)\n",
                sceneName.c_str(), rgb.size(), W, H, vulkan::validationErrorCount(),
                vulkan::validationWarningCount(), vulkan::validationActive() ? 1 : 0);
    return graphVerdict(renderer, sceneName);
}
