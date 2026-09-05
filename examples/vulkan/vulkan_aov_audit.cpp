// AOV replay audit — part 2 of the determinism go/no-go (part 1: the
// proprioceptive replay_audit under examples/extras/sensors).
//
// Renders a fixed scene with scripted motion for N frames on the Vulkan
// deferred renderer (headless canvas, auto-exposure pinned, vsync off) and
// folds every byte of every G-buffer AOV readback into one FNV-1a chain per
// AOV. Two fresh processes must produce identical manifests for the
// ground-truth label claim ("lossless depth, stable ids, normals, motion")
// to be *measured* rather than assumed:
//
//     vulkan_aov_audit --frames 120 --out a.txt
//     vulkan_aov_audit --frames 120 --out b.txt
//     vulkan_aov_audit --compare a.txt b.txt      # exit 0 = bit-identical
//     vulkan_aov_audit --fsr --out a_fsr.txt       # the rgb.fsr row instead of rgb
//     vulkan_aov_audit --events --out a_ev.txt     # + the GPU event-camera stream row
//     vulkan_aov_audit --scene fjord --terrain geodata/norddal --out a_fj.txt
//                                                  # the capstone environment: GeoScene
//                                                  # terrain + FFT ocean + fog + clouds,
//                                                  # camera descending through the surface
//
// The G-buffer AOVs are raster-prepass products, so they are expected to be
// bit-exact per device — unlike the RT-fed beauty frame. The `rgb` row hashes
// the post-composite colour output too, deliberately: GI/ReSTIR are stochastic
// per frame *index* but seeded, so whether the full frame also replays is a
// question worth an answer per commit, and a DIFF on that row alone (AOVs OK)
// is itself a publishable data point, not a failure of this gate.

#include "threepp/threepp.hpp"

#include "threepp/extras/terrain/GeoScene.hpp"
#include "threepp/helpers/LidarModel.hpp"
#include "threepp/helpers/PathTracedLidarSensor.hpp"
#include "threepp/objects/Ocean.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // FNV-1a 64, chained: order is part of the contract.
    class Fnv {
    public:
        void bytes(const void* p, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) {
                h_ ^= b[i];
                h_ *= 0x100000001B3ULL;
            }
        }
        [[nodiscard]] std::uint64_t value() const { return h_; }

    private:
        std::uint64_t h_ = 0xCBF29CE484222325ULL;
    };

    struct Stream {
        Fnv hash;
        std::uint64_t frames = 0;
        std::uint64_t bytes = 0;
    };

    int compare(const std::string& pathA, const std::string& pathB) {
        auto load = [](const std::string& path) {
            std::map<std::string, std::string> rows;
            std::ifstream f(path);
            if (!f) throw std::runtime_error("cannot open " + path);
            std::string name, rest;
            while (f >> name && std::getline(f, rest)) rows[name] = rest;
            return rows;
        };
        const auto a = load(pathA);
        const auto b = load(pathB);
        int bad = 0;
        for (const auto& [name, row] : a) {
            const auto it = b.find(name);
            if (it == b.end()) {
                std::cout << "MISSING  " << name << " (only in " << pathA << ")\n";
                ++bad;
            } else if (it->second != row) {
                std::cout << "DIFF     " << name << "\n  " << pathA << ":" << row
                          << "\n  " << pathB << ":" << it->second << "\n";
                ++bad;
            } else {
                std::cout << "OK       " << name << row << "\n";
            }
        }
        for (const auto& [name, row] : b) {
            if (!a.count(name)) {
                std::cout << "MISSING  " << name << " (only in " << pathB << ")\n";
                ++bad;
            }
        }
        std::cout << (bad ? "NO-GO: " : "GO: ") << (a.size() - bad) << "/" << a.size()
                  << " streams bit-identical\n";
        return bad ? 1 : 0;
    }

}// namespace

int main(int argc, char** argv) {

    int frames = 120;
    std::string outPath;
    // Per-frame rgb hash trace. Diagnostic for a DIFF on the rgb row: if run
    // B's frame k hashes equal run A's frame k±1, the divergence is capture
    // timing (a stale-frame readback), not pixel content; if individual frames
    // differ at the same index, the renderer itself diverged there.
    std::string rgbTracePath;
    // --dumprgb <prefix>: write raw rgb bytes of frames 2..9 to
    // <prefix>_f<N>.raw so two runs can be diffed spatially — WHERE pixels
    // differ says what class of pass diverged (edges = reprojection, scattered
    // singles = ray order, whole-frame LSB = a blend/exposure factor).
    std::string dumpPrefix;
    // --taasplit: additionally hash the TAA INPUT image (the shade→bloom→post
    // product) and the written HISTORY slot per frame, as manifest rows
    // taa.input / taa.history with per-frame traces on stdout. Splits "the
    // shading diverged" from "the temporal resolve diverged".
    bool taaSplit = false;
    // --hdrsplit: additionally hash the linear-HDR scene image (bloom's
    // sceneHdr — what the shade/denoise chain wrote, BEFORE bloom and post
    // touch it) as manifest row shade.hdr with a per-frame trace. Pairs with
    // --no-denoise to split the shade dispatch from the denoiser chain.
    bool hdrSplit = false;
    // --shadesplit: per-frame hash of each deferred-shade temporal image
    // (indirect / momentsSq / reflect / reflAux / shadowVis / directU) via
    // debugHashShadeImages. The first name to differ between two runs is the
    // pass the divergence enters at; directU (no rays, no history) is the
    // control that indicts the dispatch itself if it moves.
    bool shadeSplit = false;
    // Divergence-bisection toggles: each turns off one pass group suspected of
    // carrying run-varying state into the frame. The AOV rows are already
    // proven exact, so whatever breaks rgb replay enters downstream of the
    // G-buffer — find the minimal configuration that replays, then re-enable
    // one group at a time.
    bool noDenoise = false; // setDenoise(false): the inline deterministic shading path
    bool noRestir = false;  // setRestirDIEnabled(false): no reservoir feedback
    bool noOccl = false;    // setOcclusionCulling(false)
    bool noLod = false;     // setAutoLod(false)
    bool hardSun = false;   // setSunAngularRadius(0): no shadow-ray cone jitter
    bool staticScene = false;// --static: no scripted motion → no BLAS refit /
                             // TLAS update. Discriminates acceleration-structure
                             // rebuild nondeterminism from everything else.
    bool noProbes = false;   // --no-probes: setProbeGI(false). The falsification
                             // test for "probe_update is the carrier": with the
                             // atlas out of the chain, rgb must replay bit-exact.
    // --dumpprobes <prefix>: raw probeSh dump per frame (frames 0..5) for
    // byte-level forensics — which probe index, which SH band, what magnitude.
    std::string probeDumpPrefix;
    // --scene-edit: the lidar audit's entry-list churn, applied to the AOV
    // rows. A mesh is added at frame 45 and removed at frame 81, and at those
    // same frames an existing MID-LIST object (the spinner) is pulled out and
    // later re-added at the tail, so the post-revert scene holds the same
    // objects in a different order. The Ids AOV must not care: it carries the
    // stable per-object id, not the entry index.
    bool sceneEdit = false;
    int editAddFrame = 45, editRemoveFrame = 81;
    // --fsr: lift exactly one pin, the FSR 3.1 upscaler, in place of the
    // in-house TAA. The frame row is then emitted as `rgb.fsr`, its own
    // finding: the pip wheel turns FSR on by default wherever it has it, so
    // whether the shipped default replays is a question the matrix must
    // answer, but never by folding it into the TAA row's claim.
    bool fsr = false;
    // --events: the GPU event camera (DVS) rides along and its stream is
    // hashed per frame as row `events`. The detector's sub-frame clock is
    // driven from the scripted dt (frameTimeUs = 1e6/60); --events-default-clock
    // leaves it at the shipped default of 0 so the default's time source is
    // itself measured. --events-final looks at the presented frame instead of
    // the deterministic Lambert proxy.
    bool events = false, eventsFinal = false, eventsDefaultClock = false;
    // --scene fjord --terrain <pack>: the capstone environment in place of the
    // box scene. GeoScene terrain (tiles, cliff shell, canopy forest), the FFT
    // ocean in its fjord look, volumetric + height fog, clouds, and underwater
    // murk, with the camera descending from 3 m above the sea to 4 m below it
    // over the run so both the air and the water-column passes are hashed.
    // Terrain tiles bake ASYNCHRONOUSLY: a fixed settle phase runs first, and
    // the `geo` row reports the tile/bake counters so a divergence here is
    // attributable to streaming rather than to shading.
    std::string sceneName = "default";
    std::string terrainDir;
    // --hold: fjord camera stays at its start pose (no descent, so no LOD
    // churn from camera motion). --settle-idle: after the fixed settle, keep
    // rendering until the terrain reports no bake in flight, then 60 frames
    // more, before hashing. Together they separate "streaming landed at
    // different frames" from "the environment passes themselves diverge".
    bool hold = false, settleIdle = false;
    // --fjord-off a,b,c: leave named environment features out of the fjord
    // scene, for bisecting a frame-row divergence to a pass. Names: ocean,
    // fog (scene.fog + volumetrics + height fog), clouds, murk, forest,
    // shell, bands, objects (the box-scene movers).
    std::string fjordOff;
    // --settle N: a FIXED settle length in place of the 240-frame default and
    // the idle heuristic, so two runs of a bisection pair enter the hash loop
    // after exactly the same number of renders whatever the baker did.
    int settleFixed = 0;
    // --lidar: a VLP-16 fired from the camera pose every 6th frame, its
    // returns (range, instance id, return flag) hashed as row `lidar`. First
    // hit, no shading, no history: if the shaded rows differ while this one
    // is exact, the rays agree on the geometry and the divergence is in the
    // shading; if this one differs too, the ray tracer sees different
    // geometry from run to run while the rasterizer (the AOV rows) does not.
    bool lidar = false;
    // --no-ao: setDeferredAO(false). The ray-traced AO/GI gather is a
    // first-hit-terminate query whose hit DISTANCE is consumed; where surfaces
    // overlap along a short ray (terrain tile skirts under a cliff shell) the
    // first hit found is not unique, so this is the bisection toggle for it.
    bool noAo = false;
    // --lidar-grazing (fjord only): a narrow dense-grid lidar sitting 2 cm above
    // the first shore point found ahead of the camera, aimed along the sun
    // direction, i.e. the geometry of a shadow ray leaving a terrain point.
    // Camera rays meet tile faces head-on; shadow rays graze the neighbouring
    // tile skirts and the cliff shell, where two surfaces sit within float
    // precision of each other and "closest hit" is a tie broken by traversal
    // order. If THIS row differs across fresh processes while the camera lidar
    // is exact, the acceleration structure resolves ties differently run to
    // run, and every boolean shadow query over overlapping geometry inherits
    // that.
    bool lidarGrazing = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (arg == "--scene-edit") sceneEdit = true;
        else if (arg == "--fsr") fsr = true;
        else if (arg == "--events") events = true;
        else if (arg == "--events-final") events = eventsFinal = true;
        else if (arg == "--events-default-clock") events = eventsDefaultClock = true;
        else if (arg == "--scene" && i + 1 < argc) sceneName = argv[++i];
        else if (arg == "--terrain" && i + 1 < argc) terrainDir = argv[++i];
        else if (arg == "--hold") hold = true;
        else if (arg == "--settle-idle") settleIdle = true;
        else if (arg == "--fjord-off" && i + 1 < argc) fjordOff = argv[++i];
        else if (arg == "--settle" && i + 1 < argc) settleFixed = std::atoi(argv[++i]);
        else if (arg == "--lidar") lidar = true;
        else if (arg == "--no-ao") noAo = true;
        else if (arg == "--lidar-grazing") lidarGrazing = true;
        else if (arg == "--edit-add" && i + 1 < argc) editAddFrame = std::atoi(argv[++i]);
        else if (arg == "--edit-remove" && i + 1 < argc) editRemoveFrame = std::atoi(argv[++i]);
        else if (arg == "--rgbtrace" && i + 1 < argc) rgbTracePath = argv[++i];
        else if (arg == "--dumprgb" && i + 1 < argc) dumpPrefix = argv[++i];
        else if (arg == "--taasplit") taaSplit = true;
        else if (arg == "--hdrsplit") hdrSplit = true;
        else if (arg == "--shadesplit") shadeSplit = true;
        else if (arg == "--no-denoise") noDenoise = true;
        else if (arg == "--no-restir") noRestir = true;
        else if (arg == "--no-occl") noOccl = true;
        else if (arg == "--no-lod") noLod = true;
        else if (arg == "--hard-sun") hardSun = true;
        else if (arg == "--static") staticScene = true;
        else if (arg == "--no-probes") noProbes = true;
        else if (arg == "--dumpprobes" && i + 1 < argc) probeDumpPrefix = argv[++i];
        else if (arg == "--compare" && i + 2 < argc) return compare(argv[i + 1], argv[i + 2]);
    }

    Canvas canvas("vulkan_aov_audit",
                  {{"vsync", false}, {"size", WindowSize{800, 600}}, {"headless", true}});
    VulkanRenderer renderer(canvas);
    // Pinned: auto-exposure adapts across frames from image statistics, which
    // makes the colour row depend on history length — exactly the kind of
    // confound this harness exists to exclude.
    renderer.setAutoExposure(false);
    // The rgb row reads the scene-only capture path (post-TAA, pre-overlay) —
    // the same pixels a CameraSensor consumes. Pin the temporal resolve to the
    // in-house TAA: DLSS (and FSR) are third-party black boxes we cannot make
    // determinism claims about, and DLSS measurably breaks fresh-process
    // rgb replay while the AOV rows stay bit-exact. The goldens pin TAA for
    // the same reason. --fsr is the one deliberate exception, reported under
    // its own row name.
    renderer.setSceneCaptureEnabled(true);
    renderer.setDlss(false);
    renderer.setFsr(fsr);
    // fsr() reports the ACTIVE upscaler, which is decided at the first frame;
    // availability (compiled in, context created) is what can be checked here.
    if (fsr && !renderer.fsrAvailable()) {
        std::cout << "FSR requested but unavailable on this build/GPU\n";
        return 2;
    }
    if (noDenoise) renderer.setDenoise(false);
    if (noRestir) renderer.setRestirDIEnabled(false);
    if (noOccl) renderer.setOcclusionCulling(false);
    if (noLod) renderer.setAutoLod(false);
    if (hardSun) renderer.setSunAngularRadius(0.f);
    if (noProbes) renderer.setProbeGI(false);
    if (noAo) renderer.setDeferredAO(false);

    const bool fjord = sceneName == "fjord";
    if (sceneName != "default" && !fjord) {
        std::cout << "unknown --scene " << sceneName << " (default | fjord)\n";
        return 1;
    }
    if (fjord && terrainDir.empty()) {
        std::cout << "--scene fjord needs --terrain <pack dir> (e.g. geodata/norddal)\n";
        return 1;
    }

    constexpr std::uint32_t kEvW = 320, kEvH = 240;
    if (events) {
        renderer.setEventCameraEnabled(true);
        renderer.setEventCameraResolution(kEvW, kEvH);
        renderer.setEventCameraSource(eventsFinal ? VulkanRenderer::EventCameraSource::Final
                                                  : VulkanRenderer::EventCameraSource::Shaded);
        VulkanRenderer::EventCameraParams ep;
        if (!eventsDefaultClock) ep.frameTimeUs = 16667u;// the scripted 60 Hz dt
        renderer.setEventCameraParams(ep);
    }

    Scene scene;
    scene.background = Color(0x304050);

    auto sun = DirectionalLight::create(Color(0xffffff), 3.f);
    sun->position.set(20.f, 30.f, 15.f);
    scene.add(sun);

    // Everything the box scene places is relative to `base`: the origin in the
    // default scene, a point 2.5 m above the sea at the pack centre in the fjord.
    Vector3 base(0.f, 0.f, 0.f);
    std::shared_ptr<terrain::GeoScene> geo;
    float seaLevel = 0.f;
    auto off = [&](const char* name) {
        const std::string n = name;
        std::size_t pos = 0;
        while (pos <= fjordOff.size()) {
            const std::size_t comma = fjordOff.find(',', pos);
            const std::string item = fjordOff.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (item == n) return true;
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        return false;
    };
    if (fjord) {
        terrain::GeoSceneOptions go;
        go.packDir = terrainDir;
        go.focus.set(0.f, 0.f, 0.f);
        go.scatter = false;// the camera lives over and under water
        go.forest = !off("forest");
        go.cliffShell = !off("shell");
        go.bands = !off("bands");
        geo = terrain::GeoScene::create(go);
        scene.add(geo);
        seaLevel = geo->seaLevel();
        base.set(0.f, seaLevel + 2.5f, 0.f);

        Ocean::Options oo;
        oo.size = geo->packWorldSize() * 1.2f;
        oo.resolution = 384;
        oo.look = Ocean::Look::Fjord;
        oo.windSpeed = 5.5f;
        oo.windTheta = 215.f * math::DEG2RAD;
        oo.choppiness = 0.45f;
        oo.tileSize1 = 90.f;
        oo.tileSize2 = 7.f;
        oo.fftSize = 512;
        if (!off("ocean")) {
            auto ocean = Ocean::create(oo);
            ocean->position.y = seaLevel + 0.15f;
            scene.add(ocean);
        }

        // Air: haze with volumetrics and a height profile. Sky: a cloud shell.
        // Water column: murk clipped to below the surface.
        if (!off("fog")) {
            scene.fog = FogExp2(Color(0.62f, 0.70f, 0.78f), 0.0025f);
            renderer.setVolumetricFog(true);
            VulkanRenderer::HeightFogSettings hf;
            hf.density = 0.f;// profile-only: scene.fog supplies the density
            hf.baseY = seaLevel;
            hf.falloff = 120.f;
            renderer.setHeightFog(hf);
        }
        if (!off("clouds")) {
            VulkanRenderer::CloudSettings cs;
            cs.coverage = 0.5f;
            cs.bottomY = seaLevel + 700.f;
            cs.topY = seaLevel + 1500.f;
            renderer.setClouds(cs);
        }
        if (!off("murk")) {
            renderer.setFogWaterSurfaceY(seaLevel);
            renderer.setUnderwaterMurk(0.06f, Color(0.02f, 0.08f, 0.10f));
        }
        const auto st = geo->stats();
        std::cout << "fjord: pack " << geo->packWorldSize() << " m, sea " << seaLevel
                  << " m, loaded in " << st.loadSeconds << " s, shell tris " << st.shellTris
                  << ", forest cells " << st.forestCells << "\n";
    }

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, fjord ? 6000.f : 100.f);
    camera->position.set(base.x + 0.f, base.y + 3.f, base.z + 9.f);
    camera->lookAt(Vector3(base.x, base.y + 1.f, base.z));

    std::shared_ptr<Mesh> ground;
    if (!fjord) {
        ground = Mesh::create(BoxGeometry::create(30.f, 0.5f, 30.f),
                              MeshStandardMaterial::create(
                                      MeshStandardMaterial::Params{}.color(Color(0x556b45))));
        ground->position.y = -0.25f;
        scene.add(ground);
        renderer.setObjectInstanceId(*ground, 1);
        renderer.setObjectClassId(*ground, 1);
    }

    // One mover (translates + rotates on the frame clock: exercises Motion and
    // per-frame TLAS/G-buffer updates), one spinner, one static occluder.
    auto mover = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f),
                              MeshStandardMaterial::create(
                                      MeshStandardMaterial::Params{}.color(Color(0xc8783c))));
    scene.add(mover);
    renderer.setObjectInstanceId(*mover, 2);
    renderer.setObjectClassId(*mover, 2);

    auto spinner = Mesh::create(SphereGeometry::create(0.7f, 32, 16),
                                MeshStandardMaterial::create(
                                        MeshStandardMaterial::Params{}.color(Color(0x3c78c8))));
    spinner->position.set(base.x - 2.5f, base.y + 1.f, base.z);
    scene.add(spinner);
    renderer.setObjectInstanceId(*spinner, 3);
    renderer.setObjectClassId(*spinner, 2);

    auto pillar = Mesh::create(BoxGeometry::create(0.8f, 4.f, 0.8f),
                               MeshStandardMaterial::create(
                                       MeshStandardMaterial::Params{}.color(Color(0x808890))));
    pillar->position.set(base.x + 2.5f, base.y + 2.f, base.z - 1.f);
    scene.add(pillar);
    renderer.setObjectInstanceId(*pillar, 4);
    renderer.setObjectClassId(*pillar, 3);

    // A CPU deformer: a grid whose position+normal attributes are rewritten
    // and needsUpdate()ed EVERY frame, so it graduates to the per-frame
    // dynamic-geometry path (staging upload + frame-cb BLAS refit + the
    // vertex→prevVertex motion snapshot). Without this row the certificate
    // silently excludes the refit machinery the flock and CPU trails run on;
    // the dyn.geom manifest row below asserts the path actually engaged.
    // The wave is a pure function of the scripted clock — never wall time.
    auto waver = Mesh::create(PlaneGeometry::create(4.f, 4.f, 20, 20),
                              MeshStandardMaterial::create(
                                      MeshStandardMaterial::Params{}.color(Color(0xb03a48))));
    waver->rotation.x = -math::PI / 2;
    waver->position.set(base.x - 0.5f, base.y + 0.6f, base.z + 3.0f);
    scene.add(waver);
    renderer.setObjectInstanceId(*waver, 5);
    renderer.setObjectClassId(*waver, 4);
    auto* wavPos = waver->geometry()->getAttribute<float>("position");
    auto* wavNrm = waver->geometry()->getAttribute<float>("normal");
    wavPos->setUsage(DrawUsage::Dynamic);
    wavNrm->setUsage(DrawUsage::Dynamic);
    const std::vector<float> wavRest = wavPos->array();// rest-pose x,y copy
    auto deform = [&](double t) {
        auto& p = wavPos->array();
        auto& n = wavNrm->array();
        for (std::size_t v = 0; v < p.size() / 3; ++v) {
            const float x = wavRest[3 * v + 0];
            const float y = wavRest[3 * v + 1];
            // Local z wave with analytic partials, so the normals are exact
            // rather than re-derived from the mesh (cheaper, and one fewer
            // spot for cross-run arithmetic to hide in).
            const float ax = 2.0f * x + static_cast<float>(t) * 3.0f;
            const float ay = 2.0f * y + static_cast<float>(t) * 2.0f;
            const float z = 0.25f * std::sin(ax) * std::cos(ay);
            const float dzdx = 0.50f * std::cos(ax) * std::cos(ay);
            const float dzdy = -0.50f * std::sin(ax) * std::sin(ay);
            p[3 * v + 2] = z;
            const float inv = 1.f / std::sqrt(dzdx * dzdx + dzdy * dzdy + 1.f);
            n[3 * v + 0] = -dzdx * inv;
            n[3 * v + 1] = -dzdy * inv;
            n[3 * v + 2] = inv;
        }
        wavPos->needsUpdate();
        wavNrm->needsUpdate();
    };

    // The edit subject (see --scene-edit above). Built up front so its
    // geometry upload is not itself the event.
    auto editBox = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f),
                                MeshStandardMaterial::create(
                                        MeshStandardMaterial::Params{}.color(Color(0xd0c060))));
    editBox->position.set(base.x + 3.5f, base.y + 1.0f, base.z + 3.5f);

    // ---- fjord settle: let the asynchronous tile bakes land BEFORE hashing ---
    // A fixed frame count (not "until idle"), so the two runs enter the hash
    // loop after the same number of renders; whether the bakes had all landed
    // by then is reported in the `geo` row rather than assumed.
    constexpr int kSettleFrames = 240;
    int settleFrames = 0;
    if (fjord) {
        const int fixed = settleFixed > 0 ? settleFixed : kSettleFrames;
        for (int f = 0; f < fixed; ++f) {
            renderer.setSimTime(0.0);
            geo->update(camera->position);
            renderer.render(scene, *camera);
            ++settleFrames;
        }
        if (settleIdle && settleFixed <= 0) {
            // Until the tile baker is idle, then a fixed tail so the last landed
            // tile has been through the temporal passes. Bounded: a baker that
            // never idles is itself a finding, printed below.
            // "Idle" = no bake in flight, OR the in-flight count has not moved
            // for 300 frames (the norddal pack holds four bakes in flight
            // indefinitely; a baker that never drains is reported, not waited
            // on forever).
            int idleTail = -1, lastBaking = -1, stableFor = 0;
            for (int f = 0; f < 3600 && idleTail != 0; ++f) {
                renderer.setSimTime(0.0);
                geo->update(camera->position);
                renderer.render(scene, *camera);
                ++settleFrames;
                const int baking = geo->stats().baking;
                stableFor = (baking == lastBaking) ? stableFor + 1 : 0;
                lastBaking = baking;
                if (idleTail < 0 && (baking == 0 || stableFor >= 300)) idleTail = 60;
                else if (idleTail > 0) --idleTail;
            }
        }
        const auto st = geo->stats();
        std::cout << "fjord settle: " << settleFrames << " frames, tiles " << st.tiles
                  << ", bakes in flight " << st.baking << "\n";
    }

    // ---- render + hash ------------------------------------------------------

    struct AovRow {
        VulkanRenderer::GBufferAOV aov;
        const char* name;
        Stream stream;
    };
    AovRow rows[] = {
            {VulkanRenderer::GBufferAOV::Depth, "aov.depth", {}},
            {VulkanRenderer::GBufferAOV::Normal, "aov.normal", {}},
            {VulkanRenderer::GBufferAOV::Motion, "aov.motion", {}},
            {VulkanRenderer::GBufferAOV::Ids, "aov.ids", {}},
            {VulkanRenderer::GBufferAOV::Albedo, "aov.albedo", {}},
    };
    Stream rgb;

    constexpr double kDt = 1.0 / 60.0;// scripted clock — never wall time
    std::vector<std::uint8_t> buf;
    std::ostringstream rgbTrace;
    Stream taaIn, taaHist, shadeHdr;
    std::vector<std::uint8_t> taaInBuf, taaHistBuf, hdrBuf;
    int failures = 0;
    // Two hashes of the same stream: `events` in the order the GPU appended
    // them, `events.sorted` in a canonical order (t, y, x, polarity). A DVS
    // stream is a set with timestamps, and consumers sort by time; if only the
    // raw row differs, the SET replays and the append order is what scheduling
    // moves.
    Stream evRow, evSorted;
    std::vector<VulkanRenderer::Event> evBuf(events ? std::size_t(kEvW) * kEvH * 5 : 0);
    std::uint64_t evCount = 0, evOverflows = 0;
    std::ostringstream evTrace;
    // The first-hit instrument (see --lidar). Not added to the scene: it
    // carries no geometry and updates its own matrix when parentless.
    PathTracedLidarSensor lidarSensor(LidarModel::VLP16(), fjord ? 3000.f : 25.f);
    lidarSensor.params.detectorThreshold = 0.f;
    Stream lidarRow;
    std::vector<LidarReturn> lidarReturns;
    std::uint64_t lidarHits = 0;
    double rtaoMsSum = 0.0, shadeMsSum = 0.0;
    // The grazing probe: 8 deg x 8 deg, 160 x 160 beams, from the shore point.
    PathTracedLidarSensor grazing(8.f, 160u, 160u, 3000.f);
    grazing.params.detectorThreshold = 0.f;
    Stream grazingRow;
    std::uint64_t grazingHits = 0;
    if (lidarGrazing && fjord) {
        // First point ahead of the camera (toward -Z) that is at least 5 m above
        // the sea: the near shore of the far wall.
        float zs = -50.f;
        while (zs > -3500.f && geo->heightAt(0.f, zs) < seaLevel + 5.f) zs -= 10.f;
        const float ys = geo->heightAt(0.f, zs) + 0.02f;
        grazing.position.set(0.f, ys, zs);
        const Vector3 sunDir = Vector3(20.f, 30.f, 15.f).normalize();
        grazing.lookAt(Vector3(0.f + sunDir.x * 100.f, ys + sunDir.y * 100.f, zs + sunDir.z * 100.f));
        std::cout << "grazing probe at (0, " << ys << ", " << zs << ") aimed along the sun\n";
    }

    for (int f = 0; f < frames; ++f) {
        const double t = f * kDt;
        // The deterministic frame clock: with it, the TAA blend weights and
        // every other formerly-wall-clock input advance on this scripted time.
        renderer.setSimTime(t);
        if (!staticScene) {
            mover->position.set(base.x + static_cast<float>(2.0 * std::sin(t * 1.3)), base.y + 1.f,
                                base.z + static_cast<float>(1.5 * std::cos(t * 0.9)));
            mover->rotation.y = static_cast<float>(t * 1.7);
            spinner->rotation.x = static_cast<float>(t * 2.3);
            deform(t);
        }
        if (fjord) {
            // Descend from 3 m above the sea to 4 m below it: the surface is
            // crossed around 60% of the run, so both media get hashed frames.
            // --hold keeps the start pose.
            const float y0 = base.y + 3.f, y1 = seaLevel - 4.f;
            const float k = hold ? 0.f : static_cast<float>(f) / static_cast<float>(std::max(1, frames - 1));
            camera->position.set(base.x, y0 + (y1 - y0) * k, base.z + 9.f);
            camera->lookAt(Vector3(base.x, base.y + 1.f, base.z));
            geo->update(camera->position);
        }
        if (sceneEdit) {
            if (f == editAddFrame) {
                scene.add(editBox);
                renderer.setObjectInstanceId(*editBox, 6);
                renderer.setObjectClassId(*editBox, 2);
                scene.remove(*spinner);
            } else if (f == editRemoveFrame) {
                scene.remove(*editBox);
                scene.add(spinner);
                renderer.setObjectInstanceId(*spinner, 3);
                renderer.setObjectClassId(*spinner, 2);
            }
        }

        renderer.render(scene, *camera);
        {
            // Pass cost on record (stdout only, never the manifest: timings are
            // not a determinism claim). Used for the A/B of a shader fix.
            const auto ft = renderer.lastFrameTimings();
            rtaoMsSum += ft.rtaoMs;
            shadeMsSum += ft.pathTraceMs;
        }

        if (lidarGrazing && fjord && f % 6 == 0) {
            grazing.setSimTime(t);
            std::vector<LidarReturn> gr;
            grazing.scan(renderer, gr);
            std::vector<float> packed;
            packed.reserve(gr.size() * 2);
            for (const auto& r : gr) {
                packed.push_back(r.distance);
                packed.push_back(static_cast<float>(r.hitInstanceId));
                if (r.returnNo > 0) ++grazingHits;
            }
            grazingRow.hash.bytes(packed.data(), packed.size() * sizeof(float));
            grazingRow.bytes += packed.size() * sizeof(float);
            ++grazingRow.frames;
            if (!rgbTracePath.empty()) {
                Fnv one;
                one.bytes(packed.data(), packed.size() * sizeof(float));
                rgbTrace << "f" << f << " grazing=" << std::hex << one.value() << std::dec << "\n";
            }
        }

        if (lidar && f % 6 == 0) {
            lidarSensor.position.copy(camera->position);
            lidarSensor.rotation.copy(camera->rotation);
            lidarSensor.setSimTime(t);
            lidarSensor.scan(renderer, lidarReturns);
            std::vector<float> packed;
            packed.reserve(lidarReturns.size() * 3);
            for (const auto& r : lidarReturns) {
                packed.push_back(r.distance);
                packed.push_back(static_cast<float>(r.hitInstanceId));
                packed.push_back(static_cast<float>(r.returnNo));
                if (r.returnNo > 0) ++lidarHits;
            }
            lidarRow.hash.bytes(packed.data(), packed.size() * sizeof(float));
            lidarRow.bytes += packed.size() * sizeof(float);
            ++lidarRow.frames;
            if (!rgbTracePath.empty()) {
                Fnv one;
                one.bytes(packed.data(), packed.size() * sizeof(float));
                rgbTrace << "f" << f << " lidar=" << std::hex << one.value() << std::dec << "\n";
            }
        }

        if (events) {
            bool overflowed = false;
            const std::size_t n = renderer.readEventStreamInto(evBuf.data(), evBuf.size(), &overflowed);
            const std::uint64_t n64 = n;
            evRow.hash.bytes(&n64, sizeof(n64));
            evRow.hash.bytes(evBuf.data(), n * sizeof(VulkanRenderer::Event));
            evRow.bytes += n * sizeof(VulkanRenderer::Event);
            ++evRow.frames;
            evCount += n;
            if (overflowed) ++evOverflows;
            std::sort(evBuf.begin(), evBuf.begin() + static_cast<std::ptrdiff_t>(n),
                      [](const VulkanRenderer::Event& a, const VulkanRenderer::Event& b) {
                          if (a.t_us != b.t_us) return a.t_us < b.t_us;
                          if (a.y != b.y) return a.y < b.y;
                          if (a.x != b.x) return a.x < b.x;
                          return a.polarity < b.polarity;
                      });
            evSorted.hash.bytes(&n64, sizeof(n64));
            evSorted.hash.bytes(evBuf.data(), n * sizeof(VulkanRenderer::Event));
            evSorted.bytes += n * sizeof(VulkanRenderer::Event);
            ++evSorted.frames;
            if (!rgbTracePath.empty()) {
                Fnv one;
                one.bytes(evBuf.data(), n * sizeof(VulkanRenderer::Event));
                evTrace << "f" << f << " events=" << n << (overflowed ? " OVERFLOW" : "")
                        << " sorted=" << std::hex << one.value() << std::dec << "\n";
            }
        }

        for (auto& row : rows) {
            int w = 0, h = 0, bpp = 0;
            if (renderer.readGBufferAOV(row.aov, buf, w, h, bpp)) {
                row.stream.hash.bytes(buf.data(), buf.size());
                row.stream.bytes += buf.size();
                ++row.stream.frames;
                if (!rgbTracePath.empty() && row.aov == VulkanRenderer::GBufferAOV::Depth) {
                    // Depth is the cleanest onset marker: it has no temporal
                    // history, so the first frame it differs on is the frame
                    // the SCENE differed on.
                    Fnv one;
                    one.bytes(buf.data(), buf.size());
                    rgbTrace << "f" << f << " depth=" << std::hex << one.value() << std::dec << "\n";
                }
            } else if (f > 0) {
                // The first frame legitimately has nothing to read yet
                // (readGBufferAOV documents it); anything later is a failure.
                ++failures;
            }
        }
        const auto pixels = renderer.readSceneRGBPixels();
        if (!pixels.empty()) {
            rgb.hash.bytes(pixels.data(), pixels.size());
            rgb.bytes += pixels.size();
            ++rgb.frames;
            if (!rgbTracePath.empty()) {
                Fnv one;
                one.bytes(pixels.data(), pixels.size());
                rgbTrace << "f" << f << " " << std::hex << one.value() << std::dec << "\n";
            }
            if (taaSplit) {
                int iw = 0, ih = 0, hw = 0, hh = 0;
                if (renderer.readTaaDebugImages(taaInBuf, iw, ih, taaHistBuf, hw, hh)) {
                    Fnv i1, h1;
                    i1.bytes(taaInBuf.data(), taaInBuf.size());
                    h1.bytes(taaHistBuf.data(), taaHistBuf.size());
                    std::cout << "taasplit f" << f << " in=" << std::hex << i1.value()
                              << " hist=" << h1.value() << std::dec << "\n";
                    taaIn.hash.bytes(taaInBuf.data(), taaInBuf.size());
                    taaIn.bytes += taaInBuf.size();
                    ++taaIn.frames;
                    taaHist.hash.bytes(taaHistBuf.data(), taaHistBuf.size());
                    taaHist.bytes += taaHistBuf.size();
                    ++taaHist.frames;
                }
            }
            if (hdrSplit) {
                int hw = 0, hh = 0;
                if (renderer.readSceneHdrDebug(hdrBuf, hw, hh)) {
                    Fnv h1;
                    h1.bytes(hdrBuf.data(), hdrBuf.size());
                    std::cout << "hdrsplit f" << f << " hdr=" << std::hex << h1.value()
                              << std::dec << "\n";
                    shadeHdr.hash.bytes(hdrBuf.data(), hdrBuf.size());
                    shadeHdr.bytes += hdrBuf.size();
                    ++shadeHdr.frames;
                }
            }
            if (shadeSplit) {
                for (const auto& [nm, hsh] : renderer.debugHashShadeImages()) {
                    std::cout << "shadesplit f" << f << " " << nm << "=" << std::hex << hsh
                              << std::dec << "\n";
                }
            }
            if (!probeDumpPrefix.empty() && f <= 5) {
                std::vector<std::uint8_t> shRaw;
                if (renderer.readProbeShDebug(shRaw)) {
                    std::ofstream pf(probeDumpPrefix + "_f" + std::to_string(f) + ".raw",
                                     std::ios::binary);
                    pf.write(reinterpret_cast<const char*>(shRaw.data()),
                             static_cast<std::streamsize>(shRaw.size()));
                }
            }
            if (!dumpPrefix.empty() && f >= 2 && f <= 9) {
                std::ofstream df(dumpPrefix + "_f" + std::to_string(f) + ".raw",
                                 std::ios::binary);
                df.write(reinterpret_cast<const char*>(pixels.data()),
                         static_cast<std::streamsize>(pixels.size()));
            }
        }
    }
    if (!rgbTracePath.empty()) {
        std::ofstream tf(rgbTracePath);
        tf << rgbTrace.str() << evTrace.str();
    }

    std::ostringstream manifest;
    auto emit = [&](const char* name, const Stream& s) {
        manifest << name << " frames=" << s.frames << " bytes=" << s.bytes
                 << " fnv=" << std::hex << s.hash.value() << std::dec << "\n";
    };
    for (auto& row : rows) emit(row.name, row.stream);
    emit(fsr ? "rgb.fsr" : "rgb", rgb);
    if (taaSplit) {
        emit("taa.input", taaIn);
        emit("taa.history", taaHist);
    }
    if (hdrSplit) emit("shade.hdr", shadeHdr);
    if (lidarGrazing && fjord) {
        emit("lidar.grazing", grazingRow);
        manifest << "lidar.grazing.meta beams=" << grazing.beamCount() << " hits=" << grazingHits << "\n";
    }
    if (lidar) {
        emit("lidar", lidarRow);
        manifest << "lidar.meta beams=" << lidarSensor.beamCount() << " hits=" << lidarHits
                 << " maxRange=" << lidarSensor.params.maxRange << "\n";
    }
    if (events) {
        emit("events", evRow);
        emit("events.sorted", evSorted);
        manifest << "events.meta count=" << evCount << " overflows=" << evOverflows
                 << " source=" << (eventsFinal ? "final" : "shaded")
                 << " clock=" << (eventsDefaultClock ? "default" : "scripted")
                 << " res=" << kEvW << "x" << kEvH << "\n";
    }
    if (fjord) {
        const auto st = geo->stats();
        manifest << "geo tiles=" << st.tiles << " baking=" << st.baking
                 << " shellTris=" << st.shellTris << " forestCells=" << st.forestCells
                 << " settle=" << settleFrames << " hold=" << (hold ? 1 : 0)
                 << " settleIdle=" << (settleIdle ? 1 : 0)
                 << " off=" << (fjordOff.empty() ? "none" : fjordOff) << "\n";
    }

    // The graduated-path proof: a manifest row both runs must agree on, and a
    // hard failure if the deformer never graduated — a certificate that reads
    // "deterministic" while the refit path sat idle would be a false claim.
    const auto dyn = renderer.dynamicGeomStats();
    manifest << "dyn.geom graduated=" << dyn.graduated
             << " refits=" << dyn.refitsRecorded
             << " rebuilds=" << dyn.fullRebuilds << "\n";
    // Which experiment this was: two runs that disagree here were not running
    // the same script, and the compare must say so rather than diff pixels.
    const auto tl = renderer.tlasStats();
    manifest << "script frames=" << frames << " sceneEdit=" << (sceneEdit ? 1 : 0)
             << " editAdd=" << (sceneEdit ? editAddFrame : -1)
             << " editRemove=" << (sceneEdit ? editRemoveFrame : -1)
             << " static=" << (staticScene ? 1 : 0)
             << " resolve=" << (fsr ? "fsr" : "taa")
             << " scene=" << sceneName << " ao=" << (noAo ? 0 : 1)
             << " events=" << (events ? (eventsFinal ? "final" : "shaded") : "off")
             << " tlasRebuilds=" << tl.fullRebuilds << " tlasInstances=" << tl.instances << "\n";
    if (!staticScene && dyn.graduated == 0) {
        std::cout << "DYNAMIC-GEOM PATH NEVER ENGAGED (deformer failed to graduate)\n";
        ++failures;
    }

    std::cout << "timings: rtao " << (frames ? rtaoMsSum / frames : 0.0) << " ms, shade "
              << (frames ? shadeMsSum / frames : 0.0) << " ms (means over hashed frames)\n";
    std::cout << "vulkan_aov_audit: " << frames << " frames";
    if (failures) {
        std::cout << ", READBACK FAILURES: " << failures;
    } else {
        std::cout << ", readbacks clean";
    }
    std::cout << "\n" << manifest.str();
    if (!outPath.empty()) {
        std::ofstream out(outPath);
        out << manifest.str();
        std::cout << "wrote " << outPath << "\n";
    }
    return failures ? 2 : 0;
}
