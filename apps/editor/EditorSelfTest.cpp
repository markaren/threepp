
// The editor driving itself: `--selftest` walks every seam the app has (play,
// undo across a scene swap, splines, sensors, scripts, the play-mode lock) and
// prints a PASS/FAIL line per assertion; `--screenshot` builds the spline-tube
// scenario, plays it and writes PNGs to look at. Both run on whichever backend
// the binary was started with, so `--selftest --vulkan` is a second full pass.
//
// The suite is cut into named sections (kSelfTestSections below), each timed
// and each behind `--selftest=<filter>`, so a single failing area can be rerun
// alone and a failure names its section and file:line without a rerun. The
// sections run in declared order and each starts from the state it can find —
// most open with newScene() or look their subjects up fresh — so a filtered
// run is the same code the full run executes, minus the sections skipped.
//
// Its own translation unit because it is half the code EditorApp used to be and
// none of the behaviour: the app and its acceptance harness now recompile
// independently, and a change to one cannot conflict with a change to the
// other. It is still built unconditionally and on purpose — an acceptance suite
// behind an off-by-default flag is one nobody runs, and this one has caught
// every regression in the editor so far.

#include "EditorApp.hpp"

#include "ImportFormats.hpp"

#include "threepp/extras/editor/AcousticSurfaceConfig.hpp"
#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/CharacterConfig.hpp"
#include "threepp/extras/editor/FlockConfig.hpp"
#include "threepp/extras/editor/GranularConfig.hpp"
#include "threepp/extras/editor/JointConfig.hpp"
#include "threepp/extras/editor/ParticleFieldConfig.hpp"
#include "threepp/extras/editor/ParticleFieldPlaySession.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/TerrainSculpt.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/ScriptWorkspace.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SoundConfig.hpp"
#include "threepp/extras/editor/SplatImportConfig.hpp"
#include "threepp/extras/editor/SplatSurfaceCache.hpp"
#include "threepp/extras/editor/SplatSurfaceConfig.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"
#include "threepp/extras/editor/TextConfig.hpp"
#include "threepp/extras/editor/TreeConfig.hpp"
#include "threepp/extras/editor/VehicleConfig.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

#ifdef THREEPP_WITH_AUDIO
#include "threepp/extras/editor/AudioPlaySession.hpp"
#endif

#include "threepp/extras/editor/SensorPlaySession.hpp"
#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/CharacterPlaySession.hpp"
#include "threepp/extras/editor/ConveyorPlaySession.hpp"
#include "threepp/extras/editor/GranularPlaySession.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#ifdef THREEPP_WITH_VULKAN
#include "threepp/extras/editor/SplatSurfacePlaySession.hpp"
#endif
#endif

#include "threepp/extras/editor/ConveyorConfig.hpp"

#include "threepp/core/Clock.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/editor/GeneratorConfig.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/loaders/AssetSource.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatData.hpp"
#include "threepp/scenes/Scene.hpp"
#ifdef THREEPP_WITH_VULKAN
// The screenshot passes ask the renderer which way up its pixels come back.
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>// std::getenv (THREEPP_BENCH_DISABLE)
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Every section of the selftest, in the order the run visits them. This
    // table is the one place a section is registered: `--selftest=<filter>`
    // matches against these names, the unmatched-filter error prints them, and
    // section() in runSelfTest refuses a name that is not here — so the table
    // cannot rot out from under the filter.
    constexpr const char* kSelfTestSections[] = {
            "boot",
            "undo",
            "play-lock",
            "play-gizmo",
            "play-overlays",
            "follow-selection",
            "follow-heading",
            "texture-dialog",
            "soft-body",
            "script-rigid-body",
            "script-fixed-update",
            "script-collision",
            "script-trigger",
            "script-raycast",
            "compound-collider",
            "camera-dock",
            "light-icons",
            "python-scripting",
            "two-scripts",
            "external-edit",
            "splines",
            "conveyors",
            "particle-fields",
            "flocks",
            "granular",
            "import",
            "tpz-archive",
            "prefabs",
            "renderer-look",
            "urdf",
            "urdf-scale",
            "texture-drop",
            "texture-settings",
            "shadow-side",
            "material-edit",
            "ortho-views",
            "view-gizmo",
            "terrain",
            "text",
            "trees",
            "sound",
            "acoustics",
            "joints",
            "vehicle",
            "character",
            "ortho-shading",
            "ortho-shadows",
            "collider-overlay",
            "sensors",
            "camera-sensor",
            "pinhole-migration",
            "scene-userdata",
            "generator",
            "instancing",
            "open-example",
            "splats",
            "splat-surface",
            "splat-churn",
            "sog-scan",
    };

    std::string lowered(std::string text) {
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    bool isRegisteredSection(const std::string& name) {
        for (const char* candidate : kSelfTestSections) {
            if (name == candidate) return true;
        }
        return false;
    }

    // `--selftest=terrain,splat` — comma-separated, case-insensitive terms. A
    // term that names a section exactly selects that section alone ("text" is
    // not the texture blocks); any other term is a substring ("splat" is every
    // splat section, "script-" the scripting ones). An empty filter matches
    // everything: the no-arg run is the full run it has always been.
    bool sectionMatches(const std::string& filter, const char* name) {
        if (filter.empty()) return true;
        const std::string haystack = lowered(name);
        std::istringstream terms(lowered(filter));
        std::string term;
        while (std::getline(terms, term, ',')) {
            if (term.empty()) continue;
            if (isRegisteredSection(term)) {
                if (term == haystack) return true;
            } else if (haystack.find(term) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    // The radius a generated tube actually came out at: the FARTHEST any of its
    // vertices sits from the curve it was swept along. A tube is a ring of
    // vertices per sample, so the maximum is the radius wherever the sweep is
    // honest and larger wherever it is not. Zero when there is nothing to
    // measure. Selftest only.
    float tubeRadius(const Mesh& mesh, const Object3D& spline) {

        const auto geometry = mesh.geometry();
        if (!geometry) return 0.f;
        const auto* position = geometry->getAttribute<float>("position");
        if (!position || position->count() < 4) return 0.f;

        const auto config = editor::SplineConfig::read(spline);
        const auto curve = config ? config->curve(spline) : nullptr;
        if (!curve) return 0.f;
        const auto spine = curve->getPoints(256);

        float farthest = 0.f;
        for (int i = 0; i < position->count(); ++i) {
            const Vector3 vertex(position->getX(i), position->getY(i), position->getZ(i));
            float nearest = std::numeric_limits<float>::max();
            for (const auto& on : spine) nearest = std::min(nearest, vertex.distanceTo(on));
            farthest = std::max(farthest, nearest);
        }
        return farthest;
    }

    // A short 16-bit mono PCM WAV, written from scratch.
    //
    // The audio block needs a file miniaudio can actually decode, and generating
    // one is better than reaching for threepp-data: it makes the block run on a
    // machine that never fetched the assets, and WAV is the one format the
    // vendored miniaudio decodes with no third-party code at all. A quiet sine,
    // long enough that a handful of frames cannot run it out.
    bool writeTestWav(const std::filesystem::path& path, float seconds = 2.f) {

        constexpr int rate = 22050;
        constexpr int bits = 16;
        constexpr int channels = 1;
        const auto frames = static_cast<std::uint32_t>(static_cast<float>(rate) * seconds);
        const std::uint32_t dataBytes = frames * channels * (bits / 8);

        std::ofstream out(path, std::ios::binary);
        if (!out) return false;

        const auto u32 = [&out](std::uint32_t v) {
            const unsigned char bytes[4]{static_cast<unsigned char>(v & 0xff),
                                         static_cast<unsigned char>((v >> 8) & 0xff),
                                         static_cast<unsigned char>((v >> 16) & 0xff),
                                         static_cast<unsigned char>((v >> 24) & 0xff)};
            out.write(reinterpret_cast<const char*>(bytes), 4);
        };
        const auto u16 = [&out](std::uint16_t v) {
            const unsigned char bytes[2]{static_cast<unsigned char>(v & 0xff),
                                         static_cast<unsigned char>((v >> 8) & 0xff)};
            out.write(reinterpret_cast<const char*>(bytes), 2);
        };

        out.write("RIFF", 4);
        u32(36 + dataBytes);
        out.write("WAVE", 4);
        out.write("fmt ", 4);
        u32(16);
        u16(1);// PCM
        u16(channels);
        u32(rate);
        u32(rate * channels * (bits / 8));
        u16(static_cast<std::uint16_t>(channels * (bits / 8)));
        u16(bits);
        out.write("data", 4);
        u32(dataBytes);

        for (std::uint32_t i = 0; i < frames; ++i) {
            const auto t = static_cast<float>(i) / rate;
            const auto sample = static_cast<std::int16_t>(
                    6000.f * std::sin(math::TWO_PI * 440.f * t));
            u16(static_cast<std::uint16_t>(sample));
        }

        return out.good();
    }

}// namespace


// stb_image_write is already compiled into threepp (utils/StbImageWrite.cpp);
// these mirror its two entry points rather than adding an include path for the
// one function this file wants.
extern "C" {
    int stbi_write_png(char const* filename, int w, int h, int comp, const void* data, int stride_in_bytes);
    void stbi_flip_vertically_on_write(int flag);
    unsigned char* stbi_load(char const* filename, int* x, int* y, int* channels_in_file, int desired_channels);
    void stbi_image_free(void* retval_from_stbi_load);
}

// The pixels the renderer just produced, written as a PNG. Shared by both
// screenshot passes; the row order is the BACKEND's, not a constant (GL hands
// back a bottom-up framebuffer and needs the flip, the Vulkan swapchain
// readback is already top-down).
bool EditorApp::shootTo(const std::filesystem::path& path) {

    const auto size = canvas_.size();
    bool bottomUpPixels = true;
#ifdef THREEPP_WITH_VULKAN
    if (dynamic_cast<VulkanRenderer*>(renderer_.get())) bottomUpPixels = false;
#endif
    stbi_flip_vertically_on_write(bottomUpPixels ? 1 : 0);

    const auto pixels = renderer_->readRGBPixels();
    const bool wrote =
            pixels.size() >= static_cast<std::size_t>(size.width()) * size.height() * 3 &&
            stbi_write_png(path.string().c_str(), size.width(), size.height(), 3,
                           pixels.data(), size.width() * 3) != 0;
    // The flip flag is process-global and every other stb writer in the tree
    // (CameraSensor::writeImage, GLRenderer::writeImage, sensor recordings)
    // pre-flips its own buffer and assumes the global is 0 — leave it that way.
    stbi_flip_vertically_on_write(0);
    std::cout << "[screenshot] " << (wrote ? "wrote " : "FAILED to write ") << path.string()
              << std::endl;
    return wrote;
}

// --screenshot over WHATEVER IS OPEN: a scene file, or one of the shipped
// examples. The pass below it builds a fixed spline scenario and is what
// `--screenshot` alone still means; this one exists because a scene you cannot
// photograph is a scene nobody will look at, and adding a bespoke code path per
// scene is how a review harness stops being used.
//
// Everything about it is the command line's: --play decides whether it plays,
// --seconds how long it settles, --shot where the camera stands. With no shots
// it frames the document, which is at least an honest establishing view.
int EditorApp::runSceneScreenshot() {

    Clock clock;
    const auto playFor = [&](float seconds) {
        float elapsed = 0.f;
        for (int i = 0; i < 20000 && elapsed < seconds; ++i) {
            const float dt = clock.getDelta();
            elapsed += std::max(dt, 0.f);
            canvas_.animateOnce([&] { frame(dt); });
        }
    };

    sensorCloudVisible_ = true;
    // Editor furniture off: the grid and the origin axes are drawn ON the
    // scene's own floor, and a shot meant to answer "does this arena read" must
    // not be answered through a wireframe lying across it.
    if (grid_) grid_->visible = false;
    if (axes_) axes_->visible = false;
    // And the whole authoring layer with them — the marker icons, the selection
    // outline, the frustum helper, the handles. A hemisphere light and an
    // ambient light both sit at the origin, so their billboards hang in mid-air
    // in the middle of the arena; a document that opens with something selected
    // (userData["editorFollow"]) would be photographed through a yellow box.
    // Useful when you are authoring, noise when you are judging a picture.
    //
    // Through the same flag Play reads (authoringVisible), so this pass hides
    // what a play session hides and cannot drift from it — and unlike the four
    // hand-hidden nodes it replaces, it also covers whatever the pass selects
    // later. The point cloud stays: it is what the scene is DOING, not
    // furniture.
    hideAuthoring_ = true;
    // One frame before Play so the scene's own materials and shadow maps exist.
    playFor(0.05f);
    if (options_.play) startPlay();

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // Hold the requested keys for the settle, then let go. The provider the app
    // installed reads ImGui, and nothing is going to press a key here.
    if (!options_.keys.empty()) {
        const auto held = options_.keys;
        scripting::keyStateProvider() = [held](const std::string& key) {
            return std::find(held.begin(), held.end(), key) != held.end();
        };
    }
#endif
    playFor(std::max(options_.settle, 0.05f));
#ifdef THREEPP_EDITOR_WITH_PYTHON
    if (!options_.keys.empty() && !options_.holdKeys) {
        scripting::keyStateProvider() = [](const std::string&) { return false; };
        // Long enough for whatever was commanded to settle back to a hover.
        playFor(1.2f);
    }
#endif

    auto shots = options_.shots;
    if (shots.empty()) {
        // Nothing asked for, so take what the session already has: a document
        // that authored its own editorView has ANSWERED the framing question,
        // and with Follow on the camera has been chasing the subject through
        // the settle and IS the shot. Framing the document would throw both
        // away. Anything else has nothing to keep, and gets the automatic view.
        if (!documentView_ && !followSelection_) frameDocument();
        shots.push_back({camera_.position, orbit_->target, ""});
    }

    const auto sibling = [&](const std::string& suffix) {
        if (suffix.empty()) return options_.screenshot;
        auto path = options_.screenshot;
        path.replace_filename(options_.screenshot.stem().string() + "_" + suffix +
                              options_.screenshot.extension().string());
        return path;
    };

    bool wrote = true;
    for (const auto& shot : shots) {
        camera_.position.copy(shot.position);
        orbit_->target.copy(shot.target);
        // Long enough for the point cloud to refill from the new pose and for
        // any script-driven emissive to reach its current value.
        playFor(0.35f);
        wrote = shootTo(sibling(shot.label)) && wrote;
    }

    if (sensorCloud_ && sensorCloud_->geometry()) {
        std::cout << "[screenshot] sensor cloud: "
                  << sensorCloud_->geometry()->drawRange.count << " points" << std::endl;
    }
    if (isPlaying()) stopPlay();
    return wrote ? 0 : 1;
}

// --bench: how long a frame of whatever is open actually takes.
//
// Exists because the numbers a person can read off the running editor are both
// wrong for the question. The status bar's fps is ImGui's smoothed average over
// a moving window, and the window is FIFO-presented, so a renderer that needs
// 12 ms and one that needs 16 both read "60". The constructor turns vsync off
// for this pass (see the Canvas parameters) so what is timed is the renderer.
//
// CPU frame time is the wall clock around one animateOnce — everything the
// editor does in a frame, present included. On Vulkan the per-pass GPU medians
// come from the backend's timestamp queries (VulkanRenderer::lastFrameTimings),
// which is the only way to say WHICH pass a frame is spent in; those are GPU
// time for that pass, so they do not sum to the CPU frame time and are not
// meant to.
//
// THREEPP_BENCH_DISABLE=a,b,c strips pieces of the frame for attribution. It is
// a bench-only ablation switch, deliberately an environment variable rather
// than a command line flag: it exists to answer "what costs what" in a session,
// not to be a supported way to run the editor.
int EditorApp::runBench() {

    // --- ablations ---------------------------------------------------------
    bool noCloud = false, noSensors = false, noUi = false, noOverlay = false, noPlay = false;
    if (const char* raw = std::getenv("THREEPP_BENCH_DISABLE"); raw && *raw) {
        std::string list(raw);
        std::cout << "[bench] disabled: " << list << std::endl;
        const auto has = [&](const char* token) { return list.find(token) != std::string::npos; };
        noCloud    = has("cloud");
        noSensors  = has("sensors");
        noUi       = has("ui");
        noOverlay  = has("overlay");
        noPlay     = has("play");
        if (has("follow")) followSelection_ = false;

#ifdef THREEPP_WITH_VULKAN
        // The deferred pipeline's own knobs, so "what is the floor" can be
        // answered per stage instead of guessed at.
        if (auto* vulkan = dynamic_cast<VulkanRenderer*>(renderer_.get())) {
            if (has("ao")) vulkan->setDeferredAO(false);
            if (has("probegi")) vulkan->setProbeGI(false);
            if (has("restir")) vulkan->setRestirDIEnabled(false);
            if (has("denoise")) vulkan->setDenoise(false);
            // Compute vs bandwidth: half the pixels. A cost that halves is
            // per-pixel work; one that does not is fixed or bandwidth-bound.
            if (has("halfres")) vulkan->setRenderScale(0.5f);
        }
#endif
    }

    if (noSensors) {
        // Strip the authoring, not the objects: the scene keeps its geometry and
        // its mass, and Play simply builds no sensors from it.
        int stripped = 0;
        document_.scene().traverse([&](Object3D& object) {
            if (const auto config = SensorConfig::read(object); config && config->enabled) {
                object.userData.erase("sensor");
                ++stripped;
            }
        });
        std::cout << "[bench] stripped " << stripped << " sensor(s)" << std::endl;
    }
    if (noCloud) sensorCloudVisible_ = false;
    if (noOverlay && overlay_) overlay_->visible = false;
    benchSkipUi_ = noUi;

    if (options_.play && !noPlay) startPlay();

#ifdef THREEPP_EDITOR_WITH_PYTHON
    if (!options_.keys.empty()) {
        const auto held = options_.keys;
        scripting::keyStateProvider() = [held](const std::string& key) {
            return std::find(held.begin(), held.end(), key) != held.end();
        };
    }
#endif

    Clock clock;
    const auto playFor = [&](float seconds) {
        float elapsed = 0.f;
        for (int i = 0; i < 100000 && elapsed < seconds; ++i) {
            const float dt = clock.getDelta();
            elapsed += std::max(dt, 0.f);
            if (!canvas_.animateOnce([&] { frame(dt); })) break;
        }
    };

    // Warm up: shader/pipeline compiles, the first BLAS builds, TAA history,
    // the probe grid's first round-robin sweep and — for a played scene — the
    // controller settling into its hover all land in here rather than in the
    // measurement.
    playFor(std::max(options_.settle, 0.5f));

#ifdef THREEPP_WITH_VULKAN
    auto* vk = dynamic_cast<VulkanRenderer*>(renderer_.get());
    std::vector<VulkanRenderer::FrameTimings> gpu;
    if (vk) gpu.reserve(static_cast<std::size_t>(options_.bench));
#endif

    std::vector<double> cpuMs;
    cpuMs.reserve(static_cast<std::size_t>(options_.bench));

    for (int i = 0; i < options_.bench; ++i) {
        const auto begin = std::chrono::high_resolution_clock::now();
        const float dt = clock.getDelta();
        if (!canvas_.animateOnce([&] { frame(dt); })) break;
        const auto end = std::chrono::high_resolution_clock::now();
        cpuMs.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
#ifdef THREEPP_WITH_VULKAN
        if (vk) gpu.push_back(vk->lastFrameTimings());
#endif
    }

    if (cpuMs.empty()) {
        std::cout << "[bench] no frames measured" << std::endl;
        return 1;
    }

    // Median and p95 rather than a mean: one 40 ms hitch (a pipeline compile
    // that escaped the warmup, the OS taking the core away) moves a mean over
    // 600 frames by enough to hide a real regression, and the p95 is where a
    // periodic stall shows up as itself instead of smearing into the average.
    const auto pct = [](std::vector<double> v, double q) {
        std::sort(v.begin(), v.end());
        const auto at = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5);
        return v[at];
    };
    const double median = pct(cpuMs, 0.5);
    const double p95 = pct(cpuMs, 0.95);
    double sum = 0.;
    for (const double ms : cpuMs) sum += ms;

    std::cout << "[bench] frames=" << cpuMs.size()
              << " cpu_median=" << median << " ms"
              << " cpu_p95=" << p95 << " ms"
              << " cpu_max=" << *std::max_element(cpuMs.begin(), cpuMs.end()) << " ms"
              << " mean_fps=" << (1000. * static_cast<double>(cpuMs.size()) / sum)
              << " median_fps=" << (1000. / median) << std::endl;

#ifdef THREEPP_WITH_VULKAN
    if (vk && !gpu.empty()) {
        const auto medianOf = [&](float VulkanRenderer::FrameTimings::* field) {
            std::vector<double> v;
            v.reserve(gpu.size());
            for (const auto& t : gpu) v.push_back(static_cast<double>(t.*field));
            return pct(std::move(v), 0.5);
        };
        std::cout << "[bench] gpu medians (ms):"
                  << " raster=" << medianOf(&VulkanRenderer::FrameTimings::rasterGbufMs)
                  << " shade=" << medianOf(&VulkanRenderer::FrameTimings::pathTraceMs)
                  << " denoise=" << medianOf(&VulkanRenderer::FrameTimings::denoiseMs)
                  << " taa=" << medianOf(&VulkanRenderer::FrameTimings::taaMs)
                  << " overlay=" << medianOf(&VulkanRenderer::FrameTimings::overlayMs)
                  << " froxel=" << medianOf(&VulkanRenderer::FrameTimings::froxelMs)
                  << " dof=" << medianOf(&VulkanRenderer::FrameTimings::dofMs)
                  << " gbufResolve=" << medianOf(&VulkanRenderer::FrameTimings::gbufResolveMs)
                  << " shadeB=" << medianOf(&VulkanRenderer::FrameTimings::shadeBMs)
                  << std::endl;
        std::cout << "[bench] cpu medians (ms):"
                  << " ensureScene=" << medianOf(&VulkanRenderer::FrameTimings::cpuEnsureSceneMs)
                  << " record=" << medianOf(&VulkanRenderer::FrameTimings::cpuRecordMs)
                  << " render=" << medianOf(&VulkanRenderer::FrameTimings::cpuFrameMs)
                  << std::endl;
    }
#endif

    if (isPlaying()) stopPlay();
    return 0;
}

int EditorApp::runScreenshot() {

    // A document of its own to photograph beats the built-in scenario. The
    // scenario is the DEFAULT, not the contract: `--screenshot=x.png` with
    // nothing else on the line still builds the tubes and writes every sibling
    // view, which is what the road/spline acceptance passes ask for.
    //
    // ANY file on the command line counts, not only a .json. The startup path
    // sends everything else through the import dispatch, so `threepp_editor
    // scan.ply --screenshot=x.png` has a scene to photograph too — and
    // photographing the spline tubes instead, because the file was a model
    // rather than a document, is not a distinction anyone typing that meant to
    // draw. The settle (--seconds) is what the async import finishes in.
    if (!options_.example.empty() || !options_.openOnStart.empty()) {
        return runSceneScreenshot();
    }

    Clock clock;
    const auto playFor = [&](float seconds) {
        float elapsed = 0.f;
        // Wall-clock, not frame-count: with vsync off a frame's dt is tiny and
        // a fixed frame budget would capture the balls still in the air.
        for (int i = 0; i < 20000 && elapsed < seconds; ++i) {
            const float dt = clock.getDelta();
            elapsed += std::max(dt, 0.f);
            canvas_.animateOnce([&] { frame(dt); });
        }
    };

    auto& scene = document_.scene();

    // The tubes this feature is judged on: the factory default, an S-curve, and
    // a GRADED one that climbs into a crest inside a bend — a tube's closed
    // cross-section has to survive all three.
    auto plain = ObjectFactory::createSpline(scene);
    {
        auto config = SplineConfig::read(*plain).value_or(SplineConfig{});
        config.mesh = SplineConfig::MeshKind::Tube;
        config.radius = 0.5f;
        config.write(*plain);
    }
    plain->position.z = -6.f;
    addObject(plain, scene, "Screenshot Tube");

    auto s = ObjectFactory::createSpline(scene);
    {
        static constexpr float points[][3] = {
                {-9.f, 0.5f, -3.f}, {-3.f, 0.5f, 3.f}, {3.f, 0.5f, -3.f}, {9.f, 0.5f, 3.f}};
        const auto nodes = SplineConfig::controlPointNodes(*s);
        for (std::size_t i = 0; i < nodes.size() && i < 4; ++i) {
            nodes[i]->position.set(points[i][0], points[i][1], points[i][2]);
        }
        auto config = SplineConfig::read(*s).value_or(SplineConfig{});
        config.mesh = SplineConfig::MeshKind::Tube;
        config.radius = 0.6f;
        config.write(*s);
    }
    s->position.z = 4.f;
    addObject(s, scene, "Screenshot S Tube");

    auto hill = ObjectFactory::createSpline(scene);
    {
        static constexpr float points[][3] = {
                {-11.f, 0.f, 0.f}, {-4.f, 1.5f, 3.f}, {1.f, 3.f, 0.f}, {6.f, 1.5f, -3.f}, {12.f, 0.f, 0.f}};
        auto config = SplineConfig::read(*hill).value_or(SplineConfig{});
        for (std::size_t i = SplineConfig::controlPointNodes(*hill).size(); i < 5; ++i) {
            hill->add(ObjectFactory::createSplinePoint(*hill));
        }
        const auto nodes = SplineConfig::controlPointNodes(*hill);
        for (std::size_t i = 0; i < nodes.size() && i < 5; ++i) {
            nodes[i]->position.set(points[i][0], points[i][1], points[i][2]);
        }
        config.mesh = SplineConfig::MeshKind::Tube;
        config.radius = 0.4f;
        config.write(*hill);
    }
    hill->position.z = 12.f;
    addObject(hill, scene, "Screenshot Hill Tube");
    playFor(0.1f);// the sync pass derives the tube meshes

    for (auto* spline : {plain.get(), s.get(), hill.get()}) {
        if (auto* mesh = SplineConfig::derivedMesh(*spline)) {
            PhysicsConfig config;
            config.enabled = true;
            config.body = PhysicsConfig::Body::Static;
            config.write(*mesh);
        }
    }
    // Returns the body it dropped, so a caller can instrument it. `label` is the
    // undo label AND the name, which makes the hierarchy in these shots readable.
    const auto drop = [&](Primitive kind, const Vector3& from, const char* label) {
        auto object = ObjectFactory::createPrimitive(kind, scene);
        object->name = label;
        object->position.copy(from);
        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Dynamic;
        config.friction = 0.8f;
        config.write(*object);
        addObject(object, scene, label);
    };
    drop(Primitive::Sphere, {-6.f, 3.f, 4.f}, "Ball on S");
    drop(Primitive::Sphere, {0.f, 3.f, -7.f}, "Ball on default");
    // An IMU on one of the falling bodies, so the Sensors tab has a signal with
    // shape in it: free fall reads ~0, the landing impact spikes, and rest
    // settles on +g. A flat trace proves the plot draws; that one proves it is
    // plotting physics.
    if (auto* ball = document_.scene().getObjectByName("Ball on S")) {
        SensorConfig imu;
        imu.enabled = true;
        imu.type = SensorConfig::Type::Imu;
        imu.rateHz = 60.f;
        imu.write(*ball);
    }
    // A box, not a ball: a sphere on a hill rolls off it, and what wants
    // showing here is a body sitting still on a graded surface — the case a
    // trimmed offset broke, since it invented a height at every corner it cut
    // and left a step to catch on.
    drop(Primitive::Box, {-1.f, 4.5f, 13.5f}, "Box near the crest");

    // A LIDAR on a mast, authored exactly as the inspector authors one. Added
    // BEFORE play, because the pre-play snapshot is what Stop restores — an
    // object added while playing is not in the document at all.
    //
    // This is the shot the sensor feature is judged on. A count in a panel says
    // a scan happened; only the cloud says the beams, the sensor pose and the
    // unprojection agree, and only a picture shows a cloud hugging the geometry
    // rather than floating beside it.
    {
        auto mast = ObjectFactory::createPrimitive(Primitive::Box, scene);
        mast->name = "Lidar Mast";
        mast->position.set(0.f, 2.2f, 6.f);
        mast->scale.set(0.3f, 0.3f, 0.3f);

        SensorConfig sensor;
        sensor.enabled = true;
        sensor.type = SensorConfig::Type::Lidar;
        sensor.beams = SensorConfig::Beams::OS1_64;
        sensor.faceSize = 192;
        sensor.rateHz = 8.f;
        sensor.nearPlane = 0.4f;
        sensor.farPlane = 40.f;
        sensor.rangeStddev = 0.01f;
        sensor.write(*mast);
        addObject(mast, scene, "Add Lidar Mast");
    }

    // The text feature's own acceptance: solid type standing over the tubes.
    // Judged the way the tubes are — by looking. Outlines that fail to
    // triangulate, holes that fill in (the counters of e and p), or a centring
    // bug all show here and in no vertex count.
    {
        auto title = ObjectFactory::createText(scene);
        auto config = TextConfig::read(*title).value_or(TextConfig{});
        config.text = "threepp";
        config.size = 1.4f;
        config.depth = 0.35f;
        config.apply(*title);
        title->name = "Screenshot Title";
        title->position.set(0.f, 4.5f, -2.f);
        addObject(title, scene, "Add Screenshot Title");
    }
    sensorCloudVisible_ = true;

    startPlay();
    playFor(2.5f);// balls settle, the collider overlay's line buffer fills

    camera_.position.set(-1.f, 27.f, 27.f);
    orbit_->target.set(0.f, 0.f, 5.f);
    playFor(0.2f);

    const auto shoot = [&](const std::filesystem::path& path) { return shootTo(path); };

    // Two shots, because one cannot answer both questions. A road drawn under
    // its own collider overlay is a wall of lines — good for asking whether the
    // shapes hug the surface, useless for asking whether the surface is faceted.
    const auto sibling = [&](const char* suffix) {
        auto path = options_.screenshot;
        path.replace_filename(options_.screenshot.stem().string() + suffix +
                              options_.screenshot.extension().string());
        return path;
    };

    bool wrote = shoot(options_.screenshot);

    // And a third, low and near, along the graded road. A crease in a grade is
    // invisible from overhead — it is a shading break, and shading breaks want
    // a glancing angle and the surface filling the frame.
    camera_.position.set(1.f, 3.2f, 34.f);
    orbit_->target.set(0.f, 1.f, 12.f);
    playFor(0.2f);
    wrote = shoot(sibling("_graded")) && wrote;

    camera_.position.set(-1.f, 27.f, 27.f);
    orbit_->target.set(0.f, 0.f, 5.f);
    physicsDebug_ = true;
    playFor(0.4f);// the overlay's line buffer fills
    wrote = shoot(sibling("_colliders")) && wrote;

    // And the sensor cloud, still inside the same play. Collider lines off:
    // a cloud read through a wall of them says nothing about either.
    physicsDebug_ = false;
    {
        const auto cloudPoints = [this] {
            const auto geometry = sensorCloud_ ? sensorCloud_->geometry() : nullptr;
            return geometry ? geometry->drawRange.count : 0;
        };
        // Wall-clock, and waiting on the CLOUD rather than on a frame count: a
        // scan is rate-gated off the physics accumulator, so how many frames it
        // takes depends on the machine. Vision sensors scan in every build —
        // the session no longer needs PhysX — so the wait is unconditional.
        for (int i = 0; i < 400 && sensors_ && cloudPoints() == 0; ++i) playFor(0.05f);
        camera_.position.set(-11.f, 9.f, 20.f);
        orbit_->target.set(0.f, 1.5f, 6.f);
        playFor(0.4f);
        std::cout << "[screenshot] sensor cloud: " << cloudPoints() << " points" << std::endl;
        wrote = shoot(sibling("_sensor_cloud")) && wrote;

        // Low and close, along the beams: a cloud that looks fine from above can
        // still be sitting a metre off the surface it is meant to be measuring,
        // and only a grazing angle shows that.
        camera_.position.set(-2.f, 3.2f, 15.f);
        orbit_->target.set(0.f, 1.2f, 6.f);
        playFor(0.4f);
        wrote = shoot(sibling("_sensor_cloud_near")) && wrote;

        // And the readout, with the Sensors tab brought forward. The plots are
        // the other half of "see them live", and a plot nobody has looked at is
        // a plot nobody knows is drawing the right thing.
        selectSensorsTab_ = true;
        playFor(0.4f);
        wrote = shoot(sibling("_sensor_panel")) && wrote;
    }
    camera_.position.set(-1.f, 27.f, 27.f);
    orbit_->target.set(0.f, 0.f, 5.f);

    // And the axis views, for the same reason the rest of this function exists:
    // a projection is a claim about what the image does to parallel lines, and
    // the only way to check it is to look. Top wants the grid flat under the
    // road; Front wants it stood up behind it.
    physicsDebug_ = false;
    // Stopped and with something selected: the gizmo is rebuilt against the
    // ortho camera when the projection changes, and a gizmo that draws at the
    // wrong size or points the wrong way is the failure mode to look for.
    stopPlay();
    playFor(0.2f);
    if (auto* subject = document_.scene().getObjectByName("Spline 3")) selectObject(subject);

    setOrthographic(true);
    setViewPreset(ViewPreset::Top);
    playFor(0.2f);
    wrote = shoot(sibling("_ortho_top")) && wrote;

    setViewPreset(ViewPreset::Front);
    playFor(0.2f);
    wrote = shoot(sibling("_ortho_front")) && wrote;

    // And the pair that answers the OTHER question about a projection toggle:
    // does the scene still SHADE the same. The axis views above can't — nothing
    // to compare them against — so shoot one user viewpoint twice, perspective
    // then orthographic, and let the toggle's own framing preservation line them
    // up. Lights, shadows, ambient occlusion and fog should read the same in
    // both; only the perspective convergence should differ. This is the shot
    // that caught the Vulkan backend routing an ortho camera into the flat
    // unlit HUD path (see setOrthographicSceneRendering).
    setOrthographic(false);
    camera_.position.set(-1.f, 14.f, 22.f);
    orbit_->target.set(0.f, 0.f, 8.f);
    playFor(0.3f);
    wrote = shoot(sibling("_persp_user")) && wrote;

    setOrthographic(true);
    playFor(0.3f);
    wrote = shoot(sibling("_ortho_user")) && wrote;

    // And the camera-preview dock. A scene camera, selected, renders through
    // the renderer's secondary-pane path — the one that drew flat unlit fills
    // over the editor view on Vulkan. The shot is what says the dock now shows
    // a cleared background and lit, depth-tested geometry.
    setOrthographic(false);
    {
        auto previewCamera = ObjectFactory::createCamera(document_.scene());
        previewCamera->position.set(-8.f, 6.f, 4.f);
        previewCamera->lookAt(Vector3(0.f, 1.f, 6.f));
        addObject(previewCamera, document_.scene(), "Add Camera");
        selectObject(previewCamera.get());
    }
    playFor(0.3f);
    wrote = shoot(sibling("_campreview")) && wrote;

    // And the dock as SENSOR preview: a camera-hosted colour sensor, selected,
    // still in edit mode — no Play anywhere near. This is the shot the sensor
    // authoring flow is judged on now: aiming a sensor IS aiming a camera, so
    // the frustum helper and the sensor glyph stand on the same node in the
    // viewport while the dock shows what the sensor will record, before the
    // first Play ever runs.
    {
        auto eye = ObjectFactory::createCamera(document_.scene());
        eye->name = "Sensor Cam";
        eye->position.set(-2.2f, 3.4f, 8.5f);
        // -Z is the viewing direction; pitched down the slope the tubes run on.
        eye->rotation.set(-0.55f, 0.35f, 0.f);
        eye->fov = 62.f;
        eye->nearPlane = 0.1f;
        eye->farPlane = 60.f;
        eye->updateProjectionMatrix();

        SensorConfig colour;
        colour.enabled = true;
        colour.type = SensorConfig::Type::Camera;
        colour.width = 192;
        colour.height = 128;
        colour.rateHz = 8.f;
        colour.write(*eye);
        addObject(eye, document_.scene(), "Add Sensor Cam");

        // Framed so the sensor camera, its frustum and the slope it looks at
        // are all in shot together with the dock.
        camera_.position.set(-6.f, 8.f, 18.f);
        orbit_->target.set(-1.f, 2.f, 6.f);
    }
    playFor(0.3f);
    wrote = shoot(sibling("_sensor_dock")) && wrote;

    // And instancing. The question a screenshot answers here is the one the
    // assertions cannot: a grid of instances drawn from one object, with the
    // outline sitting around ONE of them. A box around the whole cloud and a box
    // around the right instance both satisfy "an outline exists" — only the
    // picture distinguishes them.
    {
        constexpr int kCols = 6;
        constexpr int kRows = 4;
        auto instanced = InstancedMesh::create(
                BoxGeometry::create(0.8f, 0.8f, 0.8f),
                MeshStandardMaterial::create({{"color", Color(0x66aaff)}}),
                kCols * kRows);
        instanced->name = "Instanced Grid";
        for (int r = 0; r < kRows; ++r) {
            for (int c = 0; c < kCols; ++c) {
                Matrix4 m;
                // A little height variation, so the shot reads as many objects
                // rather than as one flat wall that could be a single mesh.
                const float y = 0.6f + 0.35f * static_cast<float>((r + c) % 3);
                m.setPosition(-5.f + 2.f * static_cast<float>(c), y,
                              -4.f + 2.f * static_cast<float>(r));
                instanced->setMatrixAt(static_cast<std::size_t>(r * kCols + c), m);
            }
        }
        instanced->instanceMatrix()->needsUpdate();
        addObject(instanced, document_.scene(), "Add Instanced Grid");
        playFor(0.2f);

        // A middle instance, so the outline has neighbours on every side: an
        // outline one cell off is obvious here and invisible at a corner.
        const int picked = 2 * kCols + 3;
        selectObject(instanced.get(), picked);
        camera_.position.set(-2.f, 7.f, 11.f);
        orbit_->target.set(0.f, 1.f, -1.f);
        playFor(0.3f);
        std::cout << "[screenshot] instanced grid: " << instanced->count()
                  << " instances, outlining " << picked << std::endl;
        wrote = shoot(sibling("_instancing")) && wrote;
    }

    // And the conveyors. What the picture answers that the assertions cannot:
    // does a generated conveyor read as a MACHINE — frame, legs, drums, belt
    // texture, roller bed, cleat bars, a true circular bend — and does the
    // playing one carry its cargo mid-belt. Everything in the shot is
    // first-party procedural geometry.
    {
        selectObject(nullptr);

        // A long straight belt on the frame, cargo dropped at its upstream end.
        auto straight = ObjectFactory::createConveyor(document_.scene());
        straight->name = "Conveyor Straight";
        straight->position.set(30.f, 0.f, -6.f);
        {
            const auto nodes = ConveyorConfig::waypointNodes(*straight);
            nodes[0]->position.set(-3.5f, 0.75f, 0.f);
            nodes[1]->position.set(0.f, 0.75f, 0.f);
            nodes[2]->position.set(3.5f, 0.75f, 0.f);
            auto config = ConveyorConfig::read(*straight).value_or(ConveyorConfig{});
            config.speed = 1.f;
            config.width = 0.9f;
            config.write(*straight);

            // A diverter plowed across the belt: by the play shot the cargo is
            // being fed toward the edge — the picture that says walls WORK.
            auto wall = ObjectFactory::createConveyorWall(*straight);
            const auto wallPoints = ConveyorWallConfig::pointNodes(*wall);
            if (wallPoints.size() >= 2) {
                wallPoints[0]->position.set(-0.9f, 0.75f, 0.5f);
                wallPoints[1]->position.set(0.7f, 0.75f, -0.2f);
            }
            straight->add(wall);
        }
        addObject(straight, document_.scene(), "Add Conveyor");

        // A guide wall riding beside it — the separator form.
        auto rail = ObjectFactory::createConveyor(document_.scene());
        rail->name = "Conveyor Rail";
        rail->position.set(30.f, 0.75f, -5.35f);
        {
            const auto nodes = ConveyorConfig::waypointNodes(*rail);
            nodes[0]->position.set(-3.5f, 0.f, 0.f);
            nodes[1]->position.set(0.f, 0.f, 0.f);
            nodes[2]->position.set(3.5f, 0.f, 0.f);
            auto config = ConveyorConfig::read(*rail).value_or(ConveyorConfig{});
            config.separator = true;
            config.wallHeight = 0.35f;
            config.write(*rail);
        }
        addObject(rail, document_.scene(), "Add Conveyor Rail");

        // A roller bed running into an exact right-angle bend (a rounded
        // corner waypoint) — the two per-segment surfaces the flat shot can't
        // show, and the tangent fillet the corner model guarantees.
        auto bend = ObjectFactory::createConveyor(document_.scene());
        bend->name = "Conveyor Bend";
        bend->position.set(28.f, 0.f, 2.f);
        {
            const auto nodes = ConveyorConfig::waypointNodes(*bend);
            nodes[0]->position.set(-3.f, 0.75f, 0.f);
            nodes[1]->position.set(0.f, 0.75f, 0.f);
            nodes[2]->position.set(0.f, 0.75f, 3.f);
            ConveyorWaypointConfig rollers;
            rollers.segKind = conveyor::SegKind::Rollers;
            rollers.write(*nodes[0]);
            ConveyorWaypointConfig corner;
            corner.cornerRadius = 2.f;
            corner.write(*nodes[1]);
            auto config = ConveyorConfig::read(*bend).value_or(ConveyorConfig{});
            config.speed = 0.8f;
            config.smooth = false;
            config.width = 0.9f;
            config.write(*bend);
        }
        addObject(bend, document_.scene(), "Add Conveyor Bend");

        // A climb with cleats: the flight bars are why cargo does not slide
        // back down the incline.
        auto climb = ObjectFactory::createConveyor(document_.scene());
        climb->name = "Conveyor Climb";
        climb->position.set(24.f, 0.f, 8.f);
        {
            const auto nodes = ConveyorConfig::waypointNodes(*climb);
            nodes[0]->position.set(-3.f, 0.5f, 0.f);
            nodes[1]->position.set(-0.5f, 0.55f, 0.f);
            nodes[2]->position.set(3.f, 2.f, 0.f);
            ConveyorWaypointConfig cleats;
            cleats.segKind = conveyor::SegKind::Cleats;
            cleats.write(*nodes[1]);
            auto config = ConveyorConfig::read(*climb).value_or(ConveyorConfig{});
            config.speed = 0.7f;
            config.width = 0.9f;
            config.cleatHeight = 0.2f;
            config.write(*climb);
        }
        addObject(climb, document_.scene(), "Add Conveyor Climb");
        playFor(0.2f);// the sync pass derives the parts
        {
            std::size_t parts = 0;
            for (auto* owner : {straight.get(), rail.get(), bend.get(), climb.get()}) {
                if (auto* group = ConveyorConfig::derivedGroup(*owner)) {
                    parts += group->children.size();
                }
            }
            std::cout << "[screenshot] conveyors: 4 authored, " << parts
                      << " generated parts" << std::endl;
        }

        // NOT the `drop` lambda from the tube section: it closed over a Scene
        // reference the play/stop cycles since then have replaced twice.
        const auto cargo = [&](const Vector3& from, const char* label) {
            auto object = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
            object->name = label;
            object->position.copy(from);
            PhysicsConfig config;
            config.enabled = true;
            config.friction = 0.8f;
            config.write(*object);
            addObject(object, document_.scene(), label);
        };
        cargo({26.8f, 1.4f, -6.f}, "Cargo on straight");
        cargo({25.3f, 1.3f, 2.f}, "Cargo on rollers");
        cargo({21.5f, 1.2f, 8.f}, "Cargo on climb");
        playFor(0.2f);

        // The bend's rounded corner, selected: the still shot then carries the
        // design aids — the derived arc centre, its tangent spokes and the
        // flow chevrons — exactly as an author sees them.
        if (auto* liveBend = document_.scene().getObjectByName("Conveyor Bend")) {
            const auto nodes = ConveyorConfig::waypointNodes(*liveBend);
            if (nodes.size() >= 2) selectObject(nodes[1]);
        }

        camera_.position.set(38.f, 8.f, 15.f);
        orbit_->target.set(26.5f, 0.8f, 2.5f);
        playFor(0.3f);
        wrote = shoot(sibling("_conveyors")) && wrote;

        // Plan view: the one projection that shows whether the bend is the
        // exact quarter circle its arc-centre waypoint asked for, and whether
        // the rails run parallel along it.
        setOrthographic(true);
        setViewPreset(ViewPreset::Top);
        playFor(0.2f);
        wrote = shoot(sibling("_conveyors_top")) && wrote;
        setOrthographic(false);
        camera_.position.set(38.f, 8.f, 15.f);
        orbit_->target.set(26.5f, 0.8f, 2.5f);
        playFor(0.2f);

        // The same machines running: cargo mid-belt, cleat bars risen with it.
        startPlay();
        {
            Clock beltClock;
            float elapsed = 0.f;
            for (int i = 0; i < 20000 && elapsed < 2.2f; ++i) {
                const float dt = beltClock.getDelta();
                elapsed += std::max(dt, 0.f);
                canvas_.animateOnce([&] { frame(dt); });
            }
        }
        wrote = shoot(sibling("_conveyors_play")) && wrote;

        // Close on the diverter, still playing: the cargo mid-plow, sliding
        // along the low-friction wall while the belt keeps pushing — the
        // picture that says an attached wall FEEDS rather than decorates.
        camera_.position.set(32.5f, 3.5f, -2.f);
        orbit_->target.set(29.7f, 0.9f, -6.f);
        {
            Clock divertClock;
            float elapsed = 0.f;
            for (int i = 0; i < 20000 && elapsed < 0.8f; ++i) {
                const float dt = divertClock.getDelta();
                elapsed += std::max(dt, 0.f);
                canvas_.animateOnce([&] { frame(dt); });
            }
        }
        wrote = shoot(sibling("_conveyor_divert")) && wrote;

        // Close on the climb, still playing: the travelling cleat bars behind
        // the cargo are the reason it is not sliding back down the incline —
        // and the shot that shows the bars folding flat at the pulleys.
        camera_.position.set(28.5f, 3.f, 12.5f);
        orbit_->target.set(24.f, 1.2f, 8.f);
        playFor(0.4f);
        wrote = shoot(sibling("_conveyor_climb")) && wrote;

        stopPlay();
        playFor(0.2f);
    }

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // And the generator, running the SAME template the Add-generator-script
    // button hands the user. If that template does not produce a field, the first
    // thing anyone tries is broken — so run exactly it, and look at the result.
    {
        GeneratorConfig config;
        config.source = generatorTemplate();
        config.write(document_.scene());
        const bool ran = regenerate(document_.scene());
        auto* output = GeneratorConfig::generatedChild(document_.scene());
        std::cout << "[screenshot] generator: " << (ran ? "ran" : "FAILED") << ", output "
                  << (output ? output->children.size() : 0) << " node(s)" << std::endl;
        if (!ran) {
            for (const auto& line : console_) {
                if (line.find("regenerate") != std::string::npos) {
                    std::cout << "[screenshot] " << line.substr(0, 1200) << std::endl;
                }
            }
        }

        selectObject(output);
        camera_.position.set(-3.f, 8.f, 14.f);
        orbit_->target.set(0.f, 0.5f, 0.f);
        playFor(0.3f);
        wrote = shoot(sibling("_generator")) && wrote;
    }
#endif

    // And the Script Editor, docked. It is the shot that shows what it stopped
    // being: a floating window parked over the viewport, covering the object
    // the script was being written about. Dragged tall, because the height is
    // a preference now and this is the one tab that wants it.
    if (auto* subject = document_.scene().getObjectByName("Box near the crest")) {
        // Two of them, because one script open and several open do not look the
        // same and both are worth having a picture of.
        setInlineScript(*subject, inlineScriptTemplate(), "New Inline Script");
        selectObject(subject);
        openScriptEditor(*subject);

        const float userHeight = settings_.bottomPanelHeight;
        settings_.bottomPanelHeight = std::min(340.f, bottomHeightLimit());
        camera_.position.set(-4.f, 9.f, 20.f);
        orbit_->target.set(0.f, 2.f, 10.f);
        playFor(0.3f);
        wrote = shoot(sibling("_script_editor")) && wrote;

        if (auto* other = document_.scene().getObjectByName("Lidar Mast")) {
            setInlineScript(*other,
                            "# A second script, open at the same time as the first.\n"
                            "class Sweep:\n"
                            "\n"
                            "    rate = 0.5\n"
                            "\n"
                            "    def update(self, dt):\n"
                            "        self.obj.rotation.y += self.rate * dt\n",
                            "New Inline Script");
            openScriptEditor(*other);
            playFor(0.3f);
            wrote = shoot(sibling("_script_editor_two")) && wrote;
        }
        settings_.bottomPanelHeight = userHeight;
    }

    return wrote ? 0 : 1;
}

int EditorApp::runSelfTest() {

    Clock clock;
    int failed = 0;
    int checksRun = 0;

    // ---- the harness ------------------------------------------------------
    //
    // Every top-level block below sits behind `if (section("name"))`, with the
    // names registered once in kSelfTestSections. With no filter every section
    // runs and the suite is what it always was; `--selftest=<filter>` runs only
    // the matching ones. An unmatchable filter is answered with the list rather
    // than a silent zero-section pass.
    //
    // The header line printed as a section is entered is deliberate crash
    // forensics: when a run dies before the summary — the intermittent Vulkan
    // exit -1 does exactly that — the last header in the log names the section
    // that took it down.
    struct SectionRun {
        const char* name;
        double seconds = 0.0;
        int checks = 0;
        int failed = 0;
        bool closed = false;
    };
    std::vector<SectionRun> sections;
    std::vector<std::string> failures;
    const std::string& filter = options_.selfTestFilter;

    if (!filter.empty()) {
        bool matchesAny = false;
        for (const char* name : kSelfTestSections) {
            if (sectionMatches(filter, name)) matchesAny = true;
        }
        if (!matchesAny) {
            std::cout << "[selftest] no section matches --selftest=" << filter
                      << "\n[selftest] sections are:";
            for (const char* name : kSelfTestSections) std::cout << ' ' << name;
            std::cout << std::endl;
            return 2;
        }
    }

    using SectionClock = std::chrono::steady_clock;
    const auto runBegan = SectionClock::now();
    auto sectionBegan = runBegan;
    int checksAtSectionStart = 0;
    int failedAtSectionStart = 0;
    int gatedSectionsRun = 0;

    const auto closeSection = [&] {
        if (sections.empty() || sections.back().closed) return;
        auto& current = sections.back();
        current.seconds = std::chrono::duration<double>(SectionClock::now() - sectionBegan).count();
        current.checks = checksRun - checksAtSectionStart;
        current.failed = failed - failedAtSectionStart;
        current.closed = true;
    };
    // Only the file's name: source_location::file_name() is the full
    // compile-time path, and every failure is in this file anyway.
    const auto baseName = [](const char* path) {
        const std::string_view text(path);
        const auto slash = text.find_last_of("/\\");
        return slash == std::string_view::npos ? text : text.substr(slash + 1);
    };
    const auto section = [&](const char* name,
                             const std::source_location where = std::source_location::current()) {
        closeSection();
        if (!isRegisteredSection(name)) {
            // A name missing from the table is a suite bug: the filter and the
            // unmatched-filter listing would both lie about it. Refuse to run
            // the block, and fail the run loudly enough to get fixed.
            std::ostringstream line;
            line << "[selftest] FAIL section '" << name << "' is not in kSelfTestSections ("
                 << baseName(where.file_name()) << ':' << where.line() << ')';
            std::cout << line.str() << std::endl;
            failures.push_back(line.str());
            ++failed;
            return false;
        }
        if (!sectionMatches(filter, name)) return false;
        sections.push_back({name});
        sectionBegan = SectionClock::now();
        checksAtSectionStart = checksRun;
        failedAtSectionStart = failed;
        ++gatedSectionsRun;
        std::cout << "[selftest] --- " << name << " ---" << std::endl;
        return true;
    };
    // The summary is the localization: per-section wall time (the suite's cost,
    // stated so the sinks can be attacked with data), then every failure again
    // with its section and file:line, so one run answers WHERE without a rerun.
    const auto summary = [&] {
        closeSection();
        const double total =
                std::chrono::duration<double>(SectionClock::now() - runBegan).count();
        std::cout << "[selftest] --- summary ---" << std::endl;
        for (const auto& ran : sections) {
            std::ostringstream line;
            line << "[selftest] " << std::setw(7) << std::fixed << std::setprecision(2)
                 << ran.seconds << " s  " << ran.name;
            if (ran.failed > 0) line << "  " << ran.failed << " FAILED";
            std::cout << line.str() << '\n';
        }
        if (!filter.empty()) {
            std::cout << "[selftest] filter '" << filter << "' ran " << sections.size()
                      << " of " << std::size(kSelfTestSections) << " sections\n";
        }
        for (const auto& line : failures) std::cout << line << '\n';
        std::cout << "[selftest] " << checksRun << " checks, " << failed << " failed, "
                  << std::fixed << std::setprecision(1) << total << " s total" << std::endl;
    };

    // Two steppers, and choosing between them is always the same question: does
    // the assertion measure how much SIMULATED time passed?
    //
    // `step()` advances by the real frame delta — the app's own timing, and the
    // right one for everything that just needs frames to happen: a rebuild, a
    // repaint, a poll, a temporal history settling.
    //
    // `stepFixed()` advances by a constant dt that is also the physics session's
    // substep, so one frame is exactly one substep. Anything asserting on
    // accumulated motion has to use it. Under `step()` the same frame count
    // covers a machine- and load-dependent slice of sim time, which turns the
    // assertion into a frame-rate measurement: the soft body's impact squash
    // below passed only 6-7 runs out of 8 that way, because how deeply it
    // squashed depended on how fast 120 frames were drawn.
    constexpr float kFixedDt = 1.f / 60.f;
    const auto step = [&](int frames = 1) {
        for (int i = 0; i < frames; ++i) {
            canvas_.animateOnce([&] { frame(clock.getDelta()); });
        }
    };
    [[maybe_unused]] const auto stepFixed = [&](int frames = 1) {
        for (int i = 0; i < frames; ++i) {
            canvas_.animateOnce([&] { frame(kFixedDt); });
        }
    };
    const auto check = [&](bool ok, const char* what,
                           const std::source_location where = std::source_location::current()) {
        ++checksRun;
        if (ok) {
            std::cout << "[selftest] PASS " << what << std::endl;
            return;
        }
        ++failed;
        // The failure line carries everything a diagnosis needs — section and
        // file:line — and the summary repeats it, so one run localizes.
        std::ostringstream line;
        line << "[selftest] FAIL " << what << "  ("
             << (sections.empty() ? "?" : sections.back().name) << ", "
             << baseName(where.file_name()) << ':' << where.line() << ')';
        std::cout << line.str() << std::endl;
        failures.push_back(line.str());
    };
    const auto inScene = [&](const Object3D* object) {
        bool found = false;
        document_.scene().traverse([&](Object3D& o) {
            if (&o == object) found = true;
        });
        return found;
    };
    // Overlay children that are not the fixed furniture — i.e. selection
    // outlines. Exactly one while a bounded object is selected, zero otherwise.
    const auto outlineCount = [&] {
        int n = 0;
        for (auto* child : overlay_->children) {
            if (child == grid_.get() || child == axes_.get() ||
                child == markers_.get() || child == splines_.get() ||
                child == conveyors_.get() || child == particles_.get() ||
                child == static_cast<Object3D*>(physicsDebugLines_.get()) ||
                child == static_cast<Object3D*>(cameraHelper_.get()) ||
                child == static_cast<Object3D*>(soundRings_.get()) ||
                child == static_cast<Object3D*>(particleHelper_.get()) ||
                child == static_cast<Object3D*>(brushRing_.get()) ||
                child == static_cast<Object3D*>(gizmo_.get())) continue;
            ++n;
        }
        return n;
    };

    // "boot" is not gated: the template lookups below are the floor every other
    // section stands on, so it runs whatever the filter says (and "boot" as the
    // filter is the quickest possible smoke run).
    sections.push_back({"boot"});
    std::cout << "[selftest] --- boot ---" << std::endl;

    step(2);

    auto* box = document_.scene().getObjectByName("Box");
    auto* ground = document_.scene().getObjectByName("Ground");
    check(box != nullptr, "template Box exists");
    check(ground != nullptr, "template Ground exists");
    if (!box || !ground) {
        summary();
        return 1;
    }

    if (section("undo")) {
        // Outline-leak regression: re-selecting must not accumulate helpers.
        selectObject(box);
        step();
        selectObject(ground);
        step();
        selectObject(box);
        step();
        check(selection_.get() == box, "selection is Box");
        check(outlineCount() == 1, "one outline after repeated re-select");

        // Wireframe + resize regression: flipping a material to wireframe moves the
        // mesh out of the Vulkan TLAS (the overlay pass draws it instead). That
        // flip must classify as STRUCTURAL — before it did, the next TLAS refit
        // after a window resize was a MODE_UPDATE with the wrong instance count,
        // which corrupts traversal (ray-query hang → 2 s TDR → device lost at the
        // next submit). GL has no TLAS; there this pass just exercises the resize.
        if (auto* wireMesh = box->as<Mesh>()) {
            if (auto* mw = dynamic_cast<MaterialWithWireframe*>(wireMesh->material().get())) {
                const auto sizeBefore = canvas_.size();
                mw->wireframe = true;
                wireMesh->material()->needsUpdate();
                step(3);
                canvas_.setSize({sizeBefore.width() - 160, sizeBefore.height() - 90});
                step(3);
                canvas_.setSize({sizeBefore.width(), sizeBefore.height()});
                step(3);
                mw->wireframe = false;
                wireMesh->material()->needsUpdate();
                step();
                // Reaching this line is the assertion: a corrupted TLAS never
                // returns from the resize's device-idle.
                check(true, "wireframe Box survives a window resize");
            }
        }

        // Delete through the same member the Del key and menus call.
        deleteSelected();
        step();
        check(!inScene(box), "delete removes Box from the scene");
        check(selection_.get() == nullptr, "delete clears the selection");
        check(outlineCount() == 0, "no outline after delete");
        check(commands_.canUndo(), "delete is undoable");

        commands_.undo();
        step();
        check(inScene(box), "undo restores Box");
        commands_.redo();
        step();
        check(!inScene(box), "redo removes Box again");

        // Undo across play: stop replaces the whole scene from the snapshot, so a
        // pre-play edit must re-resolve by uuid instead of writing through a
        // pointer into the destroyed graph. `box`/`ground` are stale after this
        // block — every lookup below goes through the current scene.
        commands_.undo();// Box back in the scene for the drive
        step();
        if (auto* target = document_.scene().getObjectByName("Box")) {
            const auto before = SetTransformCommand::read(*target);
            auto after = before;
            after.position.x += 2.f;
            commands_.execute(std::make_unique<SetTransformCommand>(*target, before, after, "Move"));
            startPlay();
            check(isPlaying(), "play starts for the undo-across-play drive");
            step(3);
            stopPlay();
            step();
            check(!isPlaying(), "play stops and restores the scene");
            check(commands_.canUndo(), "pre-play edit survives the scene swap");
            commands_.undo();
            step();
            auto* restored = document_.scene().getObjectByName("Box");
            check(restored != nullptr, "Box exists after stop");
            check(restored && std::abs(restored->position.x - before.position.x) < 1e-4f,
                  "undo after stop restores the pre-edit position");
        } else {
            check(false, "Box available for the undo-across-play drive");
        }
    }

    // Play is a viewer, not an editor. Every mutating entry point is driven
    // here directly rather than through its menu item, because greying a menu
    // is not the guarantee — the refusal in the operation is, and the shortcut
    // handler and file-drop path reach these with no menu in between.
    if (section("play-lock")) {
        const auto countObjects = [&] {
            std::size_t n = 0;
            document_.scene().traverse([&](Object3D&) { ++n; });
            return n;
        };

        auto* target = document_.scene().getObjectByName("Box");
        check(target != nullptr, "Box available for the play-mode lock drive");
        selectObject(target);
        document_.setDirty(false);
        step();

        const auto objectsBefore = countObjects();
        const auto undoBefore = commands_.undoCount();
        const auto redoBefore = commands_.redoCount();
        const auto nameBefore = target ? target->name : std::string();

        startPlay();
        step(2);
        check(isPlaying(), "play starts for the play-mode lock drive");

        deleteSelected();
        duplicateSelected();
        undo();
        redo();
        reparent(document_.scene(), document_.scene());
        clearEnvironment();
        addObject(ObjectFactory::createPrimitive(Primitive::Sphere, document_.scene()),
                  document_.scene(), "Add Sphere");
        saveScene();
        newScene();
        step();

        check(isPlaying(), "and none of them stopped play");
        check(commands_.undoCount() == undoBefore && commands_.redoCount() == redoBefore,
              "no edit during play reaches the command stack");
        check(!document_.dirty(), "and none of them dirties the document");
        check(document_.scene().getObjectByName("Box") != nullptr,
              "the object a Delete was aimed at is still there");

        stopPlay();
        step();

        check(countObjects() == objectsBefore, "stop restores the same object count");
        auto* after = document_.scene().getObjectByName("Box");
        check(after != nullptr && after->name == nameBefore, "and the object itself");
        check(commands_.undoCount() == undoBefore && commands_.redoCount() == redoBefore,
              "and the history is where it was before Play");
        check(!document_.dirty(), "and the document is still clean");

        selectObject(nullptr);
        step();
    }

    // The gizmo is parked for the duration of Play. Not merely hidden: attached
    // handles on a body the solver is moving invite an edit that the gate would
    // refuse, so the attachment itself is what goes — and comes back on Stop
    // with the mode and the space it had, on the selection re-resolved by uuid.
    if (section("play-gizmo")) {
        auto* target = document_.scene().getObjectByName("Box");
        auto* other = document_.scene().getObjectByName("Ground");
        check(target != nullptr && other != nullptr, "Box and Ground available for the gizmo drive");

        gizmoMode_ = "rotate";
        gizmoWorldSpace_ = true;
        selectObject(target);
        step();
        check(gizmo_->attachedObject() == target, "the gizmo is attached to the selection");
        check(gizmo_->visible && gizmo_->enabled, "and on screen while editing");

        startPlay();
        step(2);
        check(gizmo_->attachedObject() == nullptr, "play detaches the gizmo");
        check(!gizmo_->visible && !gizmo_->enabled, "and leaves it hidden and inert");

        // W/E/R and the toolbar buttons both land here. They may record a mode
        // — that is editor state — but they must not put the gizmo back.
        gizmoMode_ = "translate";
        applyGizmoMode();
        step();
        check(!gizmo_->visible && gizmo_->attachedObject() == nullptr,
              "a mode change while playing does not resurrect it");

        // A projection switch builds NEW controls (both hold a Camera& for
        // life), and the replacement has to arrive parked like the one it
        // replaced rather than re-attaching itself on the way in.
        setOrthographic(true);
        step();
        check(gizmo_->attachedObject() == nullptr,
              "nor does switching projection while playing");
        setOrthographic(false);
        step();

        // Picking still works while playing, and still outlines what it picked.
        selectObject(other);
        step();
        check(selection_.get() == other, "selecting during play still selects");
        check(outlineCount() == 1, "and still outlines the selection");
        check(gizmo_->attachedObject() == nullptr, "without attaching the gizmo");

        stopPlay();
        step(2);
        auto* restored = document_.scene().getObjectByName("Ground");
        check(selection_.get() == restored, "stop re-resolves the selection by uuid");
        check(gizmo_->attachedObject() == restored, "and the gizmo comes back on it");
        check(gizmo_->visible && gizmo_->enabled, "on screen again");
        check(gizmoMode_ == "translate" && gizmo_->getSpace() == "world",
              "with the mode and space it had while playing");

        gizmoMode_ = "translate";
        gizmoWorldSpace_ = false;
        selectObject(nullptr);
        step();
    }

    // And the rest of the authoring layer goes with the handles. The outline,
    // the outlined instance, the marker icons and the frustum helper all say
    // what you are EDITING, and Play is when the viewport should be showing the
    // scene. Picking stays live, so the interesting case is the one below: a
    // selection made mid-play builds new nodes, and they must arrive hidden.
    if (section("play-overlays")) {
        auto* target = document_.scene().getObjectByName("Box");
        auto* other = document_.scene().getObjectByName("Ground");
        check(target != nullptr && other != nullptr, "Box and Ground available for the furniture drive");

        // A camera of its own, so the frustum helper has something to be about,
        // and an InstancedMesh, so the instance outline does.
        auto sceneCamera = ObjectFactory::createCamera(document_.scene());
        sceneCamera->name = "Furniture Camera";
        sceneCamera->position.set(3.f, 2.f, 3.f);
        addObject(sceneCamera, document_.scene(), "Add Furniture Camera");
        auto instanced = InstancedMesh::create(BoxGeometry::create(0.4f, 0.4f, 0.4f),
                                               MeshStandardMaterial::create(), 4);
        instanced->name = "Furniture Instances";
        for (std::size_t i = 0; i < 4; ++i) {
            Matrix4 m;
            m.setPosition(-3.f + 1.5f * static_cast<float>(i), 0.3f, -4.f);
            instanced->setMatrixAt(i, m);
        }
        instanced->instanceMatrix()->needsUpdate();
        addObject(instanced, document_.scene(), "Add Furniture Instances");
        const auto cameraUuid = sceneCamera->uuid;
        const auto instancedUuid = instanced->uuid;

        selectObject(sceneCamera.get());
        step(2);
        check(markers_ && markers_->visible, "the marker icons are on screen while editing");
        check(cameraHelper_ && cameraHelper_->visible, "and a selected camera's frustum");

        selectObject(instanced.get(), 2);
        step();
        check(instanceOutline_ && instanceOutline_->visible, "and the outlined instance");

        selectObject(target);
        step();
        check(selectionBox_ && selectionBox_->visible, "and the selection outline");

        // Whatever the View menu was left on — the axes default to off — has to
        // come back out of the session unchanged.
        const bool gridWas = grid_ && grid_->visible;
        const bool axesWas = axes_ && axes_->visible;

        startPlay();
        step(2);
        check(markers_ && !markers_->visible, "play hides the marker icons");
        check(selectionBox_ && !selectionBox_->visible, "and the selection outline");

        // The whole point of applying it every frame: picking stays live while
        // playing, so these are NEW nodes built during the session.
        selectObject(other);
        step();
        check(selection_.get() == other, "selecting during play still selects");
        check(selectionBox_ && !selectionBox_->visible,
              "and the outline it builds arrives hidden");
        check(markers_ && !markers_->visible, "with the icons still gone");

        if (auto* playingCamera = findByUuid(document_.scene(), cameraUuid)) {
            selectObject(playingCamera);
            step();
            check(cameraHelper_ && !cameraHelper_->visible,
                  "a camera selected mid-play gets a hidden frustum");
        } else {
            check(false, "the camera survives into the furniture drive's play session");
        }
        if (auto* playingInstanced = findByUuid(document_.scene(), instancedUuid)) {
            selectObject(playingInstanced, 1);
            step();
            check(instanceOutline_ && !instanceOutline_->visible,
                  "and an instance picked mid-play a hidden outline");
        } else {
            check(false, "the instances survive into the furniture drive's play session");
        }

        // Deselecting and reselecting mid-play is the other way in: the nodes
        // are torn down and rebuilt, and rebuilt is where a default-visible
        // helper would slip through.
        selectObject(nullptr);
        step();
        check(outlineCount() == 0, "deselecting during play leaves no outline behind");
        selectObject(document_.scene().getObjectByName("Box"));
        step();
        check(outlineCount() == 1 && selectionBox_ && !selectionBox_->visible,
              "and reselecting builds one, hidden");

        stopPlay();
        step(2);
        check(markers_ && markers_->visible, "stop puts the marker icons back");
        check(selectionBox_ && selectionBox_->visible, "and the selection outline");

        // Deliberately NOT hidden: the point cloud is what the scene is DOING,
        // and the grid and the axes are a View-menu preference nobody revoked.
        check(grid_ && grid_->visible == gridWas && axes_ && axes_->visible == axesWas,
              "while the grid and the origin axes are the user's own preference");

        // Put the scene back the way the drives that follow expect it.
        for (const auto& uuid : {instancedUuid, cameraUuid}) {
            if (auto* stillThere = findByUuid(document_.scene(), uuid)) {
                selectObject(stillThere);
                deleteSelected();
            }
        }
        check(document_.scene().getObjectByName("Furniture Camera") == nullptr &&
                      document_.scene().getObjectByName("Furniture Instances") == nullptr,
              "and the drive leaves the scene as it found it");
        document_.setDirty(false);
        selectObject(nullptr);
        step();
    }

    // Follow Selection: the orbit target chases what is selected, the camera
    // rides along keeping the user's angle and distance, and it keeps doing it
    // while a Play session owns the transform. Driven with a hand-moved object
    // rather than a simulated one so the assertion is about the CAMERA and not
    // about whatever a solver did this frame.
    if (section("follow-selection")) {
        auto* subject = document_.scene().getObjectByName("Box");
        check(subject != nullptr, "Box available for the follow drive");

        // The approach is exponential and the frame rate is the machine's, so
        // convergence is waited for rather than assumed after a frame count —
        // and waited for in WALL CLOCK, because that is the currency the filter
        // spends: at an unthrottled frame rate a frame budget is only a few
        // hundred milliseconds, and the loop runs out of frames before the
        // filter runs out of error.
        Vector3 world;
        const auto settle = [&](Object3D& node, float tolerance) {
            Clock waited;
            for (int i = 0; i < 100000 && waited.getElapsedTime() < 4.f; ++i) {
                node.getWorldPosition(world);
                if (orbit_->target.distanceTo(world) <= tolerance) break;
                step();
            }
            node.getWorldPosition(world);
        };

        selectObject(subject);
        setFollowSelection(true);
        check(followSelection(), "Follow Selection is on");
        if (subject) settle(*subject, 0.02f);
        check(orbit_->target.distanceTo(world) < 0.05f,
              "follow brings the orbit target onto the selection");

        Vector3 offset;
        offset.subVectors(camera_.position, orbit_->target);

        startPlay();
        step(2);
        // The play session runs on this same graph, so this is the node the
        // chase is watching — a script or a solver moving it looks like this.
        if (auto* playing = document_.scene().getObjectByName("Box")) {
            playing->position.x += 6.f;
            settle(*playing, 0.02f);
            check(orbit_->target.distanceTo(world) < 0.1f, "and keeps chasing it while playing");

            Vector3 nowOffset;
            nowOffset.subVectors(camera_.position, orbit_->target);
            check(nowOffset.distanceTo(offset) < 1e-2f,
                  "translating the camera by the same delta, so the orbit is untouched");

            // Deselect pauses it: there is nothing to chase, and the view stays
            // where the chase left it rather than snapping anywhere.
            selectObject(nullptr);
            const Vector3 parked = orbit_->target;
            playing->position.x += 6.f;
            step(30);
            check(orbit_->target.distanceTo(parked) < 1e-3f, "deselect pauses the chase");
        } else {
            check(false, "Box survives into the follow drive's play session");
        }

        stopPlay();
        step(2);
        setFollowSelection(false);
        check(!followSelection(), "and Follow Selection switches off again");
    }

    // And the chase is a CHASE: the offset lives in the subject's heading frame,
    // so a subject that turns is followed round the corner rather than watched
    // flying sideways out of frame. Yaw only, and world up throughout — a body
    // that banks must not roll the horizon.
    //
    // Driven with stepFixed throughout: every assertion here is about where the
    // dt-based heading filter ARRIVES, which is a question about simulated time.
    // Under step() the same frame counts cover a machine-dependent slice of real
    // time — at an unthrottled few hundred fps the 120-frame settles are a
    // fraction of the filter's 150 ms time constant, and the block becomes a
    // frame-rate measurement (it failed on GL and passed on Vulkan for exactly
    // that reason). At the fixed dt, 120 frames is two seconds: thirteen time
    // constants, on every machine.
    if (section("follow-heading")) {
        auto* subject = document_.scene().getObjectByName("Box");
        check(subject != nullptr, "Box available for the heading drive");
        if (subject) {

            const Vector3 up(0.f, 1.f, 0.f);
            // The camera's own X axis in world. Level means horizontal, and it
            // is the only thing that says the picture is not rolled: a rolled
            // camera still looks at its target.
            const auto cameraRoll = [&] {
                Vector3 right(1.f, 0.f, 0.f);
                right.applyQuaternion(camera_.quaternion);
                return right.y;
            };

            subject->position.set(0.f, 0.5f, 0.f);
            subject->rotation.set(0.f, 0.f, 0.f);
            selectObject(subject);
            setFollowSelection(true);
            stepFixed(60);

            // Directly astern and a little above — the vantage a chase camera is
            // named after, and the one the Hover Arena document authors.
            camera_.position.copy(orbit_->target).add(Vector3(0.f, 2.f, 8.f));
            stepFixed(2);
            Vector3 before;
            before.subVectors(camera_.position, orbit_->target);
            check(std::abs(before.length() - std::sqrt(68.f)) < 0.5f,
                  "the chase starts from the offset the camera was put at");

            // A 90-degree yaw, flown rather than teleported: 90 fixed-dt frames
            // of one degree — a 60 deg/s turn — is a turn the smoothing has to
            // follow, not a step it could snap through.
            for (int i = 0; i < 90; ++i) {
                subject->rotateY(math::degToRad(1.f));
                stepFixed();
            }
            // Then let it settle: the heading follow is exponential, so what is
            // asserted is where it ARRIVES once the subject stops turning.
            stepFixed(120);

            Vector3 after;
            after.subVectors(camera_.position, orbit_->target);
            Vector3 want(before);
            want.applyAxisAngle(up, math::degToRad(90.f));
            check(after.distanceTo(want) < 0.35f,
                  "a 90-degree yaw carries the camera round behind the new heading");
            check(std::abs(after.length() - before.length()) < 0.05f,
                  "at the distance it had - a turn is not a dolly");
            check(std::abs(after.y - before.y) < 0.05f,
                  "and at the height it had - the rotation is about world up alone");
            check(std::abs(cameraRoll()) < 1e-3f, "with the horizon still level");

            // Bank is ATTITUDE, and attitude is exactly what a chase camera must
            // not inherit: a drone banks to move, constantly, and a camera that
            // rolled with it would make the picture unwatchable. A roll is about
            // the nose itself, so it moves the heading by nothing at all.
            subject->rotateZ(math::degToRad(35.f));
            stepFixed(120);
            Vector3 banked;
            banked.subVectors(camera_.position, orbit_->target);
            check(banked.distanceTo(after) < 0.05f, "a bank does not move the camera");
            check(std::abs(cameraRoll()) < 1e-3f, "nor tilt the horizon");

            // Nor does a pitch on a level airframe: the nose dips, and where it
            // points on the ground plane is where it pointed.
            subject->rotateZ(math::degToRad(-35.f));
            subject->rotateX(math::degToRad(20.f));
            stepFixed(120);
            Vector3 pitched;
            pitched.subVectors(camera_.position, orbit_->target);
            check(pitched.distanceTo(after) < 0.05f, "and a pitch does not either");

            // Nose straight up: the forward axis has no ground-plane projection
            // left, so there is no heading to read. Keep the last one — a body
            // tumbling through vertical must not whip the camera round with it.
            subject->rotation.set(0.f, 0.f, 0.f);
            subject->rotateY(math::degToRad(90.f));
            stepFixed(60);
            Vector3 upright;
            upright.subVectors(camera_.position, orbit_->target);
            subject->rotateX(math::degToRad(-90.f));// nose to the sky
            stepFixed(120);
            Vector3 tumbled;
            tumbled.subVectors(camera_.position, orbit_->target);
            check(tumbled.distanceTo(upright) < 0.15f,
                  "a nose pointed at the sky keeps the heading it had");

            // Orbiting composes in the subject's frame: a camera dragged over
            // the subject's shoulder stays over that shoulder through the turn.
            subject->rotation.set(0.f, 0.f, 0.f);
            stepFixed(60);
            camera_.position.copy(orbit_->target).add(Vector3(6.f, 2.f, 6.f));
            stepFixed(30);
            Vector3 shoulder;
            shoulder.subVectors(camera_.position, orbit_->target);
            for (int i = 0; i < 90; ++i) {
                subject->rotateY(math::degToRad(1.f));
                stepFixed();
            }
            stepFixed(120);
            Vector3 carried;
            carried.subVectors(camera_.position, orbit_->target);
            Vector3 wantShoulder(shoulder);
            wantShoulder.applyAxisAngle(up, math::degToRad(90.f));
            check(carried.distanceTo(wantShoulder) < 0.35f,
                  "an orbited vantage is carried round in the subject's frame");

            // A parallel projection translates and does not rotate: an axis view
            // IS a direction of view, and turning it is not following.
            subject->rotation.set(0.f, 0.f, 0.f);
            stepFixed(60);
            setOrthographic(true);
            stepFixed(2);
            Vector3 orthoBefore;
            orthoBefore.subVectors(ortho_.position, orbit_->target);
            for (int i = 0; i < 90; ++i) {
                subject->rotateY(math::degToRad(1.f));
                stepFixed();
            }
            stepFixed(60);
            Vector3 orthoAfter;
            orthoAfter.subVectors(ortho_.position, orbit_->target);
            check(orthoAfter.distanceTo(orthoBefore) < 0.05f,
                  "and an ortho view follows by translation alone");
            setOrthographic(false);

            subject->rotation.set(0.f, 0.f, 0.f);
            subject->position.set(0.f, 0.5f, 0.f);
            setFollowSelection(false);
            selectObject(nullptr);
            stepFixed(2);
        }
    }

    // A texture dialog outliving its target. The picker records (uuid, slot)
    // rather than a Material*, so confirming after the graph has been rebuilt
    // resolves against the new one — or, when the object is gone, does nothing
    // at all. Before this it dereferenced a pointer into the destroyed scene.
    if (section("texture-dialog")) {
        auto* target = document_.scene().getObjectByName("Box");
        check(target != nullptr, "Box available for the stale-texture-dialog drive");

        const auto undoBefore = commands_.undoCount();

        // What the inspector's "Load..." button records.
        pendingTextureSlot_ = {target ? target->uuid : std::string(), "map"};

        // The scene the dialog was opened against is destroyed and rebuilt: same
        // uuids, all-new objects. A Material* recorded above would now dangle.
        startPlay();
        step(2);
        stopPlay();
        step();

        assignTextureToSlot("does-not-exist.png");
        step();
        check(commands_.undoCount() == undoBefore,
              "a texture confirmed on a missing file adds nothing");

        // Now the target itself is gone. Resolution has to fail cleanly rather
        // than write through the uuid's former address.
        pendingTextureSlot_ = {"a-uuid-that-is-not-in-the-scene", "map"};
        assignTextureToSlot("does-not-exist.png");
        step();
        check(commands_.undoCount() == undoBefore,
              "a texture confirmed on a deleted object is dropped, not dereferenced");
        check(pendingTextureSlot_.objectUuid.empty(), "and the pending slot is consumed either way");
    }

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // Soft bodies. Authored the way the inspector authors them (userData), and
    // then actually played: a soft body's whole point is that the VERTICES move
    // and the renderer sees them move, which a transform check cannot observe.
    // The soft body is added and removed inside this block, so the passes below
    // still see the template scene they expect.
    if (section("soft-body")) {
        auto ball = ObjectFactory::createPrimitive(Primitive::Sphere, document_.scene());
        ball->name = "Jelly";
        // Beside the template Box, not on top of it: the Box has no collider,
        // so a body dropped at the origin sinks through it and the drive proves
        // nothing about landing. ROTATED, because a soft body is skinned from a
        // tet mesh that does not quite reach the corners of a rotated shape —
        // the case that used to tear the visual mesh open the moment Play was
        // pressed.
        ball->position.set(2.f, 3.f, 0.f);
        ball->rotation.set(0.4f, 0.7853982f, 0.f);
        auto* jelly = ball.get();

        PhysicsConfig soft;
        soft.enabled = true;
        soft.body = PhysicsConfig::Body::Soft;
        soft.mass = 2.f;
        soft.youngsModulus = 5e4f;
        soft.voxelResolution = 6;
        soft.solverIterations = 15;
        soft.write(*jelly);

        // The template Ground is a plain mesh; a soft body needs something to
        // land on, so give it a static collider for the duration.
        PhysicsConfig floorConfig;
        floorConfig.enabled = true;
        floorConfig.body = PhysicsConfig::Body::Static;
        floorConfig.shape = PhysicsConfig::Shape::Box;
        auto* floor = document_.scene().getObjectByName("Ground");
        if (floor) floorConfig.write(*floor);

        addObject(ball, document_.scene(), "Add Jelly");
        step();

        const auto uuid = jelly->uuid;
        const auto height = [&](Object3D* object) {
            auto* mesh = object ? object->as<Mesh>() : nullptr;
            if (!mesh || !mesh->geometry()) return 0.f;
            const auto* positions = mesh->geometry()->getAttribute<float>("position");
            if (!positions) return 0.f;
            float lo = 1e30f, hi = -1e30f;
            for (unsigned i = 0; i < positions->count(); ++i) {
                lo = std::min(lo, positions->getY(i));
                hi = std::max(hi, positions->getY(i));
            }
            return hi - lo;
        };
        const auto lowest = [&](Object3D* object) {
            auto* mesh = object ? object->as<Mesh>() : nullptr;
            if (!mesh || !mesh->geometry()) return 0.f;
            const auto* positions = mesh->geometry()->getAttribute<float>("position");
            if (!positions) return 0.f;
            float lo = 1e30f;
            for (unsigned i = 0; i < positions->count(); ++i) lo = std::min(lo, positions->getY(i));
            return lo;
        };
        const auto findJelly = [&] { return findByUuid(document_.scene(), uuid); };

        // The authored mesh in world space. Play re-skins the visual from the
        // tet mesh immediately, so this is what it must still look like on the
        // very first frame — before gravity has done anything.
        jelly->updateMatrixWorld(true);
        std::vector<Vector3> authored;
        if (const auto* positions = jelly->geometry()->getAttribute<float>("position")) {
            for (unsigned i = 0; i < positions->count(); ++i) {
                Vector3 v(positions->getX(i), positions->getY(i), positions->getZ(i));
                authored.push_back(v.applyMatrix4(*jelly->matrixWorld));
            }
        }

        const float restHeight = height(jelly);
        startPlay();

        // Read before the first update(): the bodies are built during
        // startPlay, so at this point the mesh has been re-skinned from the tet
        // mesh but no simulated time has passed. Any deviation here is a
        // binding error, with no free fall mixed in — stepping first would put
        // centimetres of honest falling into the same number.
        auto* playing = findJelly();
        // Creating the soft body bakes the world matrix into the geometry, so
        // world-space vertices (around the spawn height) are the signal that
        // PhysX took it. Untouched local-space geometry means no CUDA device:
        // the fallback must keep the editor playing rather than failing Play.
        const bool simulated = playing && lowest(playing) > 2.f;
        check(isPlaying(), "play starts with a soft body in the scene");

        if (simulated) {
            // Vertex-for-vertex, still the authored shape. A torn mesh shows up
            // here as a vertex flung across the body, long before it is visible
            // as "the mesh is cut" on screen.
            float drift = 0.f;
            if (const auto* positions = playing->geometry()->getAttribute<float>("position");
                positions && positions->count() == authored.size()) {
                for (unsigned i = 0; i < positions->count(); ++i) {
                    const Vector3 now(positions->getX(i), positions->getY(i), positions->getZ(i));
                    drift = std::max(drift, authored[i].distanceTo(now));
                }
            } else {
                drift = 1e30f;
            }
            // Sub-millimetre. The tear this pins moved vertices further than
            // the body is wide.
            check(drift < 1e-3f, "a rotated soft body plays the mesh that was authored");

            // Sample every frame: the deepest squash is at the impact instant,
            // and by the time the body has settled it is nearly round again.
            // Checking only the final frame would read the settled shape.
            //
            // stepFixed, so 120 frames are 120 substeps and exactly 2 s of fall
            // and impact every run. It also puts one substep in each frame, so
            // sampling per frame sees every substep the solver took — under
            // `step()` a slow frame ran several substeps behind one sync and
            // the deepest squash could pass between two samples entirely.
            float flattest = restHeight;
            for (int i = 0; i < 120; ++i) {
                stepFixed();
                if (auto* live = findJelly()) flattest = std::min(flattest, height(live));
            }
            playing = findJelly();
            check(playing && lowest(playing) < 0.4f, "the soft body falls to the ground");
            check(playing && lowest(playing) > -0.5f, "and does not fall through it");
            // Measured 0.8597 of the rest height, to six figures, on both
            // backends and every run — the fixed dt is what makes that a
            // property of the sim rather than of the frame rate.
            check(flattest < restHeight * 0.9f,
                  "and squashes on impact - the vertices themselves deform");
        } else {
            check(true, "no CUDA device - the soft body is skipped and play continues");
        }

        stopPlay();
        step();

        auto* restored = findJelly();
        check(restored && std::abs(restored->position.y - 3.f) < 1e-4f &&
                      std::abs(restored->position.x - 2.f) < 1e-4f,
              "stop restores the soft body's authored transform");
        check(restored && std::abs(height(restored) - restHeight) < 1e-4f,
              "and its rest-pose geometry");

        // Put the template scene back for the passes that follow.
        if (restored) {
            selectObject(restored);
            deleteSelected();
        }
        if (auto* groundNow = document_.scene().getObjectByName("Ground")) {
            PhysicsConfig::erase(*groundNow);
        }
        selectObject(nullptr);
        step();
    }

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // threepp.editor.rigid_body_from_object: a script reaching the body the
    // physics session is simulating. The two sessions know nothing about each
    // other, so this is the only pass that proves the seam between them holds
    // in the real app rather than in a test harness that starts both by hand.
    if (section("script-rigid-body")) {
        auto thruster = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        thruster->name = "Thruster";
        thruster->position.set(-2.f, 2.f, 0.f);
        auto* raw = thruster.get();

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Dynamic;
        config.mass = 2.f;
        config.write(*raw);

        addObject(thruster, document_.scene(), "Add Thruster");
        // 2 kg needs ~19.6 N to hover, so 60 N climbs. A script that could not
        // reach its body would leave the box falling instead.
        setInlineScript(*raw,
                        "import threepp\n"
                        "\n"
                        "class Thruster:\n"
                        "    def start(self, obj):\n"
                        "        self.body = threepp.editor.rigid_body_from_object(obj)\n"
                        "\n"
                        "    def update(self, dt):\n"
                        "        if self.body:\n"
                        "            self.body.apply_force(threepp.Vector3(0, 60, 0))\n",
                        "Thruster Script");
        step();

        const auto uuid = raw->uuid;
        startPlay();
        // Fixed dt: 90 frames are 1.5 s, and the 20 m/s^2 of net climb clears
        // the metre this asks for by a wide margin. Wall-clock frames would
        // need only ~0.31 s to get there, so a machine drawing faster than
        // ~290 fps would read the box still on its way up.
        stepFixed(90);
        auto* live = findByUuid(document_.scene(), uuid);
        check(live && live->position.y > 3.f,
              "a script drives the rigid body the physics session is simulating");
        check(scripts_ && scripts_->errorFor(uuid).empty(),
              "and reaching it raised nothing");
        stopPlay();
        step();

        if (auto* done = findByUuid(document_.scene(), uuid)) {
            selectObject(done);
            deleteSelected();
        }
        selectObject(nullptr);
        step();
    }

    // fixed_update: the script session hooked onto the physics world's substep
    // loop. The unit tests drive the sessions by hand; this is the pass that
    // proves the registration finds the world through the real PlayController,
    // where the two sessions are started in registration order and know nothing
    // about each other. stepFixed makes one frame exactly one substep, so the
    // two counters must agree AND agree with the frame count.
    if (section("script-fixed-update")) {
        auto ticker = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        ticker->name = "Ticker";
        ticker->position.set(3.f, 2.f, 0.f);
        auto* raw = ticker.get();
        addObject(ticker, document_.scene(), "Add Ticker");

        // No physics of its own: the fixed clock belongs to the world the
        // session builds, not to any body in it. Publishes into its own
        // transform, which is all a script can hand back to this test.
        setInlineScript(*raw,
                        "class Ticker:\n"
                        "    def start(self, obj):\n"
                        "        self.obj = obj\n"
                        "        self.n = 0\n"
                        "        self.frames = 0\n"
                        "        self.odd = 0\n"
                        "        self.dt0 = 0.0\n"
                        "\n"
                        "    def fixed_update(self, dt):\n"
                        "        if self.n == 0:\n"
                        "            self.dt0 = dt\n"
                        "        elif dt != self.dt0:\n"
                        "            self.odd += 1\n"
                        "        self.n += 1\n"
                        "        self.obj.position.set(float(self.n), float(self.odd), self.dt0)\n"
                        "\n"
                        "    def update(self, dt):\n"
                        "        self.frames += 1\n"
                        "        # fixed_update runs inside the physics step, so it has\n"
                        "        # already had its turn for this frame.\n"
                        "        self.obj.scale.set(float(self.frames), float(self.n - self.frames), 1.0)\n",
                        "Ticker Script");
        step();

        const auto tickerUuid = raw->uuid;
        startPlay();
        stepFixed(60);
        auto* live = findByUuid(document_.scene(), tickerUuid);
        check(live && static_cast<int>(live->position.x) == 60,
              "fixed_update runs once per physics substep");
        check(live && static_cast<int>(live->position.y) == 0 &&
                      std::abs(live->position.z - kFixedDt) < 1e-6f,
              "and always with the world's fixed timestep");
        check(live && static_cast<int>(live->scale.x) == 60 &&
                      static_cast<int>(live->scale.y) == 0,
              "before that frame's update, one for one");
        check(scripts_ && scripts_->errorFor(tickerUuid).empty(),
              "and the substep sweep raised nothing");
        stopPlay();
        step();

        if (auto* done = findByUuid(document_.scene(), tickerUuid)) {
            selectObject(done);
            deleteSelected();
        }
        selectObject(nullptr);
        step();
    }

    // on_collision_enter / on_collision_exit: the same registration story one
    // step further out. Here nothing is authored beyond two physics bodies and a
    // script — the contact-report opt-in, the actor lookup and the queue are all
    // the session's doing — so this is the pass that proves Play alone is enough
    // to make the callbacks fire, through the real PlayController.
    if (section("script-collision")) {
        // The template Ground is a plain mesh; give it a static collider for the
        // duration, exactly as the soft-body pass above does.
        PhysicsConfig floorConfig;
        floorConfig.enabled = true;
        floorConfig.body = PhysicsConfig::Body::Static;
        floorConfig.shape = PhysicsConfig::Shape::Box;
        if (auto* floor = document_.scene().getObjectByName("Ground")) floorConfig.write(*floor);

        auto bumper = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        bumper->name = "Bumper";
        // Beside the template Box, which has no collider of its own.
        bumper->position.set(3.f, 2.f, 0.f);
        auto* raw = bumper.get();

        PhysicsConfig falling;
        falling.enabled = true;
        falling.body = PhysicsConfig::Body::Dynamic;
        falling.shape = PhysicsConfig::Shape::Box;
        falling.mass = 2.f;
        falling.restitution = 0.05f;
        falling.write(*raw);
        addObject(bumper, document_.scene(), "Add Bumper");

        // Publishes into its own SCALE: position and orientation belong to the
        // simulation for as long as this is playing, and scale does not.
        // `ok` folds the whole payload contract into one number — the other
        // object resolved, as its concrete type, under the right name, with a
        // normal pointing up out of the ground and a real impulse behind it.
        setInlineScript(*raw,
                        "class Bumper:\n"
                        "    def start(self, obj):\n"
                        "        self.obj = obj\n"
                        "        self.enters = 0\n"
                        "        self.exits = 0\n"
                        "        self.ok = 0\n"
                        "\n"
                        "    def on_collision_enter(self, contact):\n"
                        "        self.enters += 1\n"
                        "        i = contact.impulse\n"
                        "        strength = (i.x * i.x + i.y * i.y + i.z * i.z) ** 0.5\n"
                        "        other = contact.other\n"
                        "        if (other is not None and other.name == \"Ground\" and\n"
                        "                type(other).__name__ == \"Mesh\" and\n"
                        "                contact.normal.y > 0.9 and strength > 0.0):\n"
                        "            self.ok += 1\n"
                        "\n"
                        "    def on_collision_exit(self, contact):\n"
                        "        self.exits += 1\n"
                        "\n"
                        "    def update(self, dt):\n"
                        "        self.obj.scale.set(float(self.enters), float(self.ok),\n"
                        "                           float(self.exits))\n",
                        "Bumper Script");
        step();

        const auto bumperUuid = raw->uuid;
        startPlay();
        // 1.5 m of fall at one substep per frame, then a while resting on it:
        // PhysX re-reports the manifold every substep until the pair sleeps, and
        // not one of those may read as a second touch.
        stepFixed(150);

        auto* live = findByUuid(document_.scene(), bumperUuid);
        check(live && static_cast<int>(live->scale.x) == 1,
              "a landing body's script gets exactly one on_collision_enter");
        check(live && static_cast<int>(live->scale.y) == 1,
              "and the contact names the other object, with a normal and an impulse");
        check(live && static_cast<int>(live->scale.z) == 0,
              "and no exit while it is still standing on it");
        check(scripts_ && scripts_->errorFor(bumperUuid).empty(),
              "and the collision sweep raised nothing");

        stopPlay();
        step();

        if (auto* done = findByUuid(document_.scene(), bumperUuid)) {
            selectObject(done);
            deleteSelected();
        }
        if (auto* groundNow = document_.scene().getObjectByName("Ground")) {
            PhysicsConfig::erase(*groundNow);
        }
        selectObject(nullptr);
        step();
    }

    // on_trigger_enter / on_trigger_exit, one door over. Nothing is enabled on
    // any actor for these: the volume reports because the DOCUMENT says it is a
    // trigger, and the pass proves the whole chain from that tick to the
    // callback — plus the two things that make it a trigger rather than a
    // collider, that the body PASSED THROUGH and that the script hearing about
    // it is the one on the body that walked in, which is not itself a trigger.
    if (section("script-trigger")) {
        auto gate = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        gate->name = "Gate";
        // Off to one side, with nothing underneath: what the volume does NOT do
        // is half the assertion.
        gate->position.set(6.f, 1.f, 0.f);
        gate->scale.set(2.f, 0.4f, 2.f);
        auto* gateRaw = gate.get();

        PhysicsConfig volume;
        volume.enabled = true;
        volume.body = PhysicsConfig::Body::Static;
        volume.shape = PhysicsConfig::Shape::Box;
        volume.trigger = true;
        volume.write(*gateRaw);
        addObject(gate, document_.scene(), "Add Gate");

        auto dropper = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        dropper->name = "Dropper";
        dropper->position.set(6.f, 4.f, 0.f);
        auto* raw = dropper.get();

        PhysicsConfig falling;
        falling.enabled = true;
        falling.body = PhysicsConfig::Body::Dynamic;
        falling.shape = PhysicsConfig::Shape::Box;
        falling.mass = 2.f;
        falling.write(*raw);
        addObject(dropper, document_.scene(), "Add Dropper");

        // Publishes into its own SCALE, like the collision pass: position
        // belongs to the simulation while this plays, and scale does not. `ok`
        // folds the payload contract into one number — the volume resolved, as
        // its concrete type, under the right name.
        setInlineScript(*raw,
                        "class Dropper:\n"
                        "    def start(self, obj):\n"
                        "        self.obj = obj\n"
                        "        self.enters = 0\n"
                        "        self.exits = 0\n"
                        "        self.ok = 0\n"
                        "\n"
                        "    def on_trigger_enter(self, other):\n"
                        "        self.enters += 1\n"
                        "        if (other is not None and other.name == \"Gate\" and\n"
                        "                type(other).__name__ == \"Mesh\"):\n"
                        "            self.ok += 1\n"
                        "\n"
                        "    def on_trigger_exit(self, other):\n"
                        "        self.exits += 1\n"
                        "\n"
                        "    def update(self, dt):\n"
                        "        self.obj.scale.set(float(self.enters), float(self.ok),\n"
                        "                           float(self.exits))\n",
                        "Dropper Script");
        step();

        const auto dropperUuid = raw->uuid;
        startPlay();
        // Long enough to fall the 2.4 m to the gate, cross it, and keep going.
        stepFixed(150);

        auto* live = findByUuid(document_.scene(), dropperUuid);
        check(live && static_cast<int>(live->scale.x) == 1,
              "a body crossing a trigger volume gets exactly one on_trigger_enter");
        check(live && static_cast<int>(live->scale.y) == 1,
              "and the callback names the volume it entered");
        check(live && static_cast<int>(live->scale.z) == 1,
              "and one on_trigger_exit when it leaves the far side");
        check(live && live->position.y < 0.f,
              "and it fell straight through - a trigger collides with nothing");
        check(scripts_ && scripts_->errorFor(dropperUuid).empty(),
              "and the trigger sweep raised nothing");

        stopPlay();
        step();

        for (const char* name : {"Dropper", "Gate"}) {
            if (auto* done = document_.scene().getObjectByName(name)) {
                selectObject(done);
                deleteSelected();
            }
        }
        selectObject(nullptr);
        step();
    }

    // threepp.editor.raycast: a query put to the same world, from a script that
    // owns a body of its own — so the pass covers the case the API exists for,
    // a ground check cast from INSIDE the caller's own collider. Nothing here is
    // authored beyond a static floor and a falling box.
    if (section("script-raycast")) {
        PhysicsConfig floorConfig;
        floorConfig.enabled = true;
        floorConfig.body = PhysicsConfig::Body::Static;
        floorConfig.shape = PhysicsConfig::Shape::Box;
        if (auto* floor = document_.scene().getObjectByName("Ground")) floorConfig.write(*floor);

        auto prober = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        prober->name = "Prober";
        prober->position.set(-3.f, 2.f, 0.f);
        auto* raw = prober.get();

        PhysicsConfig falling;
        falling.enabled = true;
        falling.body = PhysicsConfig::Body::Dynamic;
        falling.shape = PhysicsConfig::Shape::Box;
        falling.mass = 2.f;
        falling.restitution = 0.f;
        falling.write(*raw);
        addObject(prober, document_.scene(), "Add Prober");

        // Publishes into its own SCALE, like the collision pass: position is the
        // simulation's for as long as this plays, and scale is not.
        //   x  casts that named the floor with ignore
        //   y  casts that named THIS body without it — the footgun, demonstrated
        //   z  worst disagreement (x1000) between the ray's distance and the
        //      body's own height, over the whole fall
        setInlineScript(*raw,
                        "import threepp\n"
                        "\n"
                        "class Prober:\n"
                        "    def start(self, obj):\n"
                        "        self.obj = obj\n"
                        "        self.body = threepp.editor.rigid_body_from_object(obj)\n"
                        "        self.found = 0\n"
                        "        self.itself = 0\n"
                        "        self.worst = 0.0\n"
                        "\n"
                        "    def fixed_update(self, dt):\n"
                        "        if self.body is None:\n"
                        "            return\n"
                        "        p = self.body.position\n"
                        "        down = threepp.Vector3(0.0, -1.0, 0.0)\n"
                        "        hit = threepp.editor.raycast(p, down, 50.0, ignore=self.obj)\n"
                        "        if hit is not None and hit.object is not None and \\\n"
                        "                hit.object.name == \"Ground\" and hit.normal.y > 0.9:\n"
                        "            self.found += 1\n"
                        "            err = abs(hit.distance - p.y)\n"
                        "            if err > self.worst:\n"
                        "                self.worst = err\n"
                        "        naive = threepp.editor.raycast(p, down, 50.0)\n"
                        "        if naive is not None and naive.object is not None and \\\n"
                        "                naive.object.name == \"Prober\":\n"
                        "            self.itself += 1\n"
                        "\n"
                        "    def update(self, dt):\n"
                        "        self.obj.scale.set(float(self.found), float(self.itself),\n"
                        "                           self.worst * 1000.0)\n",
                        "Prober Script");
        step();

        const auto proberUuid = raw->uuid;
        startPlay();
        stepFixed(120);

        auto* live = findByUuid(document_.scene(), proberUuid);
        check(live && static_cast<int>(live->scale.x) == 120,
              "a raycast from fixed_update finds the floor on every substep");
        check(live && static_cast<int>(live->scale.y) == 120,
              "and without ignore the same ray finds the caller's own collider");
        check(live && live->scale.z < 10.f,
              "and the distance it reports agrees with the body's own pose");
        check(scripts_ && scripts_->errorFor(proberUuid).empty(),
              "and the queries raised nothing");

        stopPlay();
        step();

        if (auto* done = findByUuid(document_.scene(), proberUuid)) {
            selectObject(done);
            deleteSelected();
        }
        if (auto* groundNow = document_.scene().getObjectByName("Ground")) {
            PhysicsConfig::erase(*groundNow);
        }
        selectObject(nullptr);
        step();
    }
#endif

    // Compound / decomposed convex colliders: an imported model is a Group with
    // sub-meshes, and before this it fell back to a 1 m unit box. Authored the
    // way the inspector authors it (physics on the GROUP), then played headlessly
    // with stepFixed for a determinstic drop. Added and removed inside this block
    // so the later passes still see the template scene.
    if (section("compound-collider")) {
        // The template Ground is a plain mesh; give it a static box collider to
        // land on for the duration.
        PhysicsConfig floorConfig;
        floorConfig.enabled = true;
        floorConfig.body = PhysicsConfig::Body::Static;
        floorConfig.shape = PhysicsConfig::Shape::Box;
        if (auto* floor = document_.scene().getObjectByName("Ground")) floorConfig.write(*floor);

        // A "model": one Group, two box sub-meshes with a gap between them. A
        // unit-box fallback would fill that gap; a real compound leaves it open.
        auto model = Group::create();
        model->name = "Compound Model";
        model->position.set(0.f, 3.f, 6.f);
        auto* modelRaw = model.get();
        for (int i = 0; i < 2; ++i) {
            auto part = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
            part->scale.set(0.6f, 0.6f, 0.6f);
            part->position.set(i == 0 ? -1.2f : 1.2f, 0.f, 0.f);
            model->add(part);
        }
        PhysicsConfig modelConfig;
        modelConfig.enabled = true;
        modelConfig.body = PhysicsConfig::Body::Dynamic;
        modelConfig.shape = PhysicsConfig::Shape::Auto;// -> per-sub-mesh compound
        modelConfig.mass = 2.f;
        modelConfig.restitution = 0.f;
        modelConfig.write(*modelRaw);
        addObject(model, document_.scene(), "Add Compound Model");

        // A probe dropped through the gap at the model's centre line.
        auto probe = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        probe->name = "Gap Probe";
        probe->scale.set(0.3f, 0.3f, 0.3f);
        probe->position.set(0.f, 6.f, 6.f);
        auto* probeRaw = probe.get();
        PhysicsConfig probeConfig;
        probeConfig.enabled = true;
        probeConfig.body = PhysicsConfig::Body::Dynamic;
        probeConfig.shape = PhysicsConfig::Shape::Box;
        probeConfig.mass = 1.f;
        probeConfig.restitution = 0.f;
        probeConfig.write(*probeRaw);
        addObject(probe, document_.scene(), "Add Gap Probe");

        const auto modelUuid = modelRaw->uuid;
        const auto probeUuid = probeRaw->uuid;
        startPlay();
        // 3 s of fixed-step fall: both bodies settle. Fixed dt so the resting
        // heights are a property of the sim, not the frame rate.
        stepFixed(180);

        auto* liveModel = findByUuid(document_.scene(), modelUuid);
        auto* liveProbe = findByUuid(document_.scene(), probeUuid);
        // The model landed low on its two boxes (~0.3), not at the ~0.7 a unit
        // box would give — the compound is real.
        check(liveModel && liveModel->position.y < 0.6f,
              "a Group of sub-meshes lands as a compound, not a unit box");
        // The probe fell through the gap to the ground rather than onto the model.
        check(liveProbe && liveProbe->position.y < 0.5f,
              "a body falls through the gap between the compound's hulls");
        stopPlay();
        step();

        // Restore the template scene for the passes that follow.
        if (auto* done = findByUuid(document_.scene(), modelUuid)) {
            selectObject(done);
            deleteSelected();
        }
        if (auto* done = findByUuid(document_.scene(), probeUuid)) {
            selectObject(done);
            deleteSelected();
        }
        if (auto* groundNow = document_.scene().getObjectByName("Ground")) {
            PhysicsConfig::erase(*groundNow);
        }
        selectObject(nullptr);
        step();
    }
#endif

    // Viewport markers and the camera frustum. The marker count is also the
    // check that the embedded SVG parses — a failure there is silent by
    // design (the object stays selectable from the hierarchy).
    if (section("camera-dock")) {
        step();
        const auto lightMarkers = viewportMarkers_.size();
        check(lightMarkers > 0, "template lights have marker icons");

        auto addedCamera = ObjectFactory::createCamera(document_.scene());
        auto* cameraRaw = addedCamera.get();
        addObject(addedCamera, document_.scene(), "Add Camera");
        step();
        check(viewportMarkers_.size() == lightMarkers + 1, "adding a camera adds a marker");
        check(selection_.get() == cameraRaw, "the new camera is selected");
        check(cameraHelper_ != nullptr, "selecting a camera shows the frustum helper");
        check(!selectionBox_, "a camera gets no degenerate bounding box");

        selectObject(nullptr);
        step();
        check(cameraHelper_ == nullptr, "deselecting drops the frustum helper");
        check(viewportMarkers_.size() == lightMarkers + 1, "the marker outlives deselection");
        // The dock is not a property of the selection. Adding a camera aimed it
        // (adding selects), and letting go of the selection does not un-aim it.
        check(dockCamera() == cameraRaw, "the dock holds its camera with nothing selected");

        selectObject(cameraRaw);
        step();

        // --- the bottom panel is a size, not a switch --------------------------
        // The height is a dragged preference, so everything laid out against the
        // panel has to read it rather than a constant — the camera dock above all,
        // since it shares the band and is drawn by the renderer, not by ImGui.
        {
            const float s = contentScale_;
            // Saved and put back: this is a persisted preference the user has
            // dragged, and a test run is not a reason to move it. Every height
            // below is derived from the minimum rather than from theirs — someone
            // who has already dragged the panel to the limit leaves no headroom to
            // grow into, and that is a valid preference, not a failing editor.
            const float userHeight = settings_.bottomPanelHeight;
            float dockX = 0, dockY = 0, dockW = 0, dockH = 0;

            settings_.bottomPanelHeight = EditorSettings::minBottomHeight;
            step();

            check(cameraDockRect(dockX, dockY, dockW, dockH) &&
                          std::abs(dockH - bottomPanelPx()) < 0.5f,
                  "the camera dock is exactly as tall as the bottom panel");
            check(std::abs(bottomBandPx() - bottomPanelPx()) < 0.5f,
                  "the side panels come down to the bottom panel with no seam between");

            // Against the limit, not a round number: a test window on a 200% display
            // has little room to spare, and the point here is that the panel takes
            // the height it is given, not that any particular height fits.
            const float taller = std::min(EditorSettings::minBottomHeight + 60.f, bottomHeightLimit());
            settings_.bottomPanelHeight = taller;
            step();
            check(taller > EditorSettings::minBottomHeight &&
                          std::abs(bottomPanelPx() - taller * s) < 0.5f,
                  "a taller panel is a taller panel");
            check(cameraDockRect(dockX, dockY, dockW, dockH) &&
                          std::abs(dockH - bottomPanelPx()) < 0.5f,
                  "the dock grows with it");

            // A settings file (or a shrunken window) must not be able to push the
            // viewport off the screen.
            settings_.bottomPanelHeight = 100000.f;
            step();
            check(std::abs(bottomPanelPx() - bottomHeightLimit() * s) < 0.5f,
                  "an absurd height clamps to what the window can spare");
            check(bottomPanelPx() < static_cast<float>(renderer_->size().height()) -
                                            menuHeight_ - statusHeight_,
                  "the clamp leaves a viewport to look at");
            check(settings_.bottomPanelHeight == 100000.f,
                  "clamping the layout does not rewrite the preference");

            settings_.bottomPanelHeight = userHeight;
            bottomPanelOpen_ = false;
            step();
            check(std::abs(bottomBandPx() - collapsedBottomPx()) < 0.5f,
                  "collapsed, the band is just the tab strip");
            check(!cameraDockRect(dockX, dockY, dockW, dockH),
                  "and the camera dock collapses with it");

            bottomPanelOpen_ = true;
            step();
        }

        // --- the preview dock SHADES ------------------------------------------
        // A selected camera renders its view through the renderer's secondary
        // scissored pane. On Vulkan that path drew flat unlit fills with no clear
        // — every mesh a silhouette in its base colour over whatever the frame
        // already held. Flatness is measurable: point the camera at two faces of
        // the template Box, one toward the sun and one away, and compare. A lit
        // pane separates them (the away face has only the 0.35 ambient); a flat
        // fill answers the same number twice.
        {
            cameraRaw->position.set(-5.f, 4.f, 0.f);
            cameraRaw->lookAt(Vector3(0.f, 1.5f, 0.f));
            // Wide enough that one frame holds all three probes: the box's two
            // faces below the view axis and, above it, rays that clear the 20 m
            // ground plane entirely (at the default 50° the ground fills the
            // frustum from this close and no empty pixel exists to probe).
            cameraRaw->fov = 70.f;
            cameraRaw->updateProjectionMatrix();
            step(8);

            float dockX = 0, dockY = 0, dockW = 0, dockH = 0;
            const bool dockVisible = cameraDockRect(dockX, dockY, dockW, dockH);
            check(dockVisible, "the camera dock has a rect to probe");

            bool bottomUp = true;
#ifdef THREEPP_WITH_VULKAN
            if (dynamic_cast<VulkanRenderer*>(renderer_.get())) bottomUp = false;
#endif
            // Mean luma of a 5x5 patch around where `world` lands in the DOCK —
            // projected through the preview camera at the aspect the preview
            // renders with (renderCameraPreview overrides it per frame and
            // restores it, so it must be re-derived here).
            const auto dockLuma = [&](const Vector3& world) -> double {
                auto* cam = cameraRaw->as<PerspectiveCamera>();
                if (!cam || dockW < 1.f || dockH < 1.f) return -1.0;
                const float aspectBefore = cam->aspect;
                cam->aspect = dockW / dockH;
                cam->updateProjectionMatrix();
                Vector3 ndc = world;
                ndc.project(*cam);
                cam->aspect = aspectBefore;
                cam->updateProjectionMatrix();
                if (std::abs(ndc.x) > 0.9f || std::abs(ndc.y) > 0.9f || ndc.z > 1.f) return -1.0;

                const auto pixels = renderer_->readRGBPixels();
                const auto size = renderer_->size();
                const int w = size.width(), h = size.height();
                if (pixels.size() < static_cast<std::size_t>(w) * h * 3) return -1.0;
                const int cx = static_cast<int>(dockX + (ndc.x * 0.5f + 0.5f) * dockW);
                const int cyTop = static_cast<int>(dockY + (0.5f - ndc.y * 0.5f) * dockH);
                double sum = 0;
                int n = 0;
                for (int y = cyTop - 2; y <= cyTop + 2; ++y) {
                    for (int x = cx - 2; x <= cx + 2; ++x) {
                        if (x < 0 || y < 0 || x >= w || y >= h) continue;
                        const int row = bottomUp ? (h - 1 - y) : y;
                        const std::size_t i = (static_cast<std::size_t>(row) * w + x) * 3;
                        sum += 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
                        ++n;
                    }
                }
                return n > 0 ? sum / n : -1.0;
            };

            const double litFace  = dockLuma(Vector3(0.f, 1.f, 0.f));  // box top — faces the sun
            const double darkFace = dockLuma(Vector3(-0.5f, 0.5f, 0.f));// -x face — ambient only
            check(litFace > 0.0 && darkFace > 0.0, "both probe faces land inside the dock");
            check(litFace > 0.0 && darkFace > 0.0 && litFace > darkFace * 1.3,
                  "the preview shades: the sunlit face outshines the unlit one");

            // Empty preview pixels are the scene background, not a stale frame:
            // an upward ray from a camera above the ground plane hits nothing
            // and must read dark.
            const double emptyPatch = dockLuma(Vector3(7.f, 4.4f, 0.f));
            check(emptyPatch >= 0.0 && emptyPatch < 60.0,
                  "and its empty pixels read as background");

            // With a texture environment, that same empty ray becomes SKY. The
            // Vulkan pane samples the very equirect the deferred miss shows;
            // a bright constant env turns the probe from ~30 (background clear)
            // to near-white, so one threshold answers "is the sky there at all"
            // on both backends without caring how each maps the texture.
            {
                constexpr int W = 8, H = 4;// 2:1 equirect aspect
                std::vector<float> data(static_cast<size_t>(W) * H * 4, 2.5f);
                Image envImg{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
                auto envTex = Texture::create(envImg);
                envTex->format = Format::RGBA;
                envTex->type = Type::Float;
                envTex->colorSpace = ColorSpace::Linear;
                envTex->mapping = Mapping::EquirectangularReflection;
                envTex->needsUpdate();
                document_.scene().background = envTex;
                step(8);

                const double skyPatch = dockLuma(Vector3(7.f, 4.4f, 0.f));
                check(skyPatch > 120.0, "a texture environment shows as the preview's sky");

                document_.scene().background = Color(0x1c1f24);// as the template had it
                step(2);
            }

            // --- the dock is the REAL render, not an impression of one ---------
            // On Vulkan the dock used to be an OverlayPass "lit pane": first
            // directional light plus ambient, and by construction no shadows, no
            // GI, no reflections. It is now a full secondary deferred view, and a
            // cast shadow is the cheapest thing that only the real pipeline can
            // produce. The template scene has what it takes already — a 1 m box
            // that casts, a ground that receives, and a sun at (6,10,5), so the
            // shadow falls toward -x-z and the sunward ground stays lit.
            //
            // Both backends are held to this: GL's dock has always been a real
            // scissored render, so a shadow there is parity, not a new claim.
            {
                cameraRaw->position.set(-3.2f, 2.6f, -3.0f);
                cameraRaw->lookAt(Vector3(0.f, 0.2f, 0.f));
                cameraRaw->fov = 60.f;
                cameraRaw->updateProjectionMatrix();
                step(8);

                // Just off the box's -x-z corner, well inside the umbra, versus the
                // sunward ground the box cannot reach. Both are ground pixels of
                // one material, so the only thing that differs is the shadow.
                //
                // Both probes must survive ANY dock proportions: the dock rect is
                // carved from whatever window and panel layout this machine has,
                // and dockLuma rejects points outside NDC +-0.9. The shadowed
                // probe projects to x = 0.01/aspect (dead centre); the sunlit one
                // sits at x = 0.38/aspect, inside the margin down to an aspect of
                // ~0.45. Its previous home at (-2.6, 1.9) was x = 1.20/aspect —
                // outside the frame on any dock squarer than 4:3, which read as
                // "no shadow" on machines whose dock band is narrower than the
                // one this check was written against. Sunlit is still sunlit:
                // the box's shadow sweeps -x-z from the sun at (6,10,5) and ends
                // at z = 0.5, a couple of metres short of the probe, and the view
                // ray to it clears the box entirely.
                const double shadowed = dockLuma(Vector3(-0.85f, 0.01f, -0.75f));
                const double sunlit = dockLuma(Vector3(0.4f, 0.01f, 2.4f));
                check(shadowed >= 0.0 && sunlit >= 0.0, "both shadow probes land inside the dock");
                check(shadowed >= 0.0 && sunlit >= 0.0 && sunlit > shadowed * 1.25,
                      "the dock casts a shadow the lit pane could not");
            }

            // --- one view, re-pointed --------------------------------------------
            // Switching which camera the dock shows must be setViewCamera on the
            // SAME view, not remove + add: a view is a whole deferred chain and
            // ~46 MB, and churning one per selection click is what the pane class
            // exists to prevent. The handle is the evidence.
            {
                const auto dockMeanLuma = [&]() -> double {
                    float dx = 0, dy = 0, dw = 0, dh = 0;
                    if (!cameraDockRect(dx, dy, dw, dh)) return -1.0;
                    const auto pixels = renderer_->readRGBPixels();
                    const auto size = renderer_->size();
                    const int w = size.width(), h = size.height();
                    if (pixels.size() < static_cast<std::size_t>(w) * h * 3) return -1.0;
                    double sum = 0;
                    int n = 0;
                    // Inset so the dock's 1 px border never enters the average.
                    for (int y = static_cast<int>(dy) + 3; y < static_cast<int>(dy + dh) - 3; y += 2) {
                        for (int x = static_cast<int>(dx) + 3; x < static_cast<int>(dx + dw) - 3; x += 2) {
                            if (x < 0 || y < 0 || x >= w || y >= h) continue;
                            const int row = bottomUp ? (h - 1 - y) : y;
                            const std::size_t i = (static_cast<std::size_t>(row) * w + x) * 3;
                            sum += 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
                            ++n;
                        }
                    }
                    return n > 0 ? sum / n : -1.0;
                };

                const std::uint32_t handleBefore = dockPane_.handle();
                const double lumaFirst = dockMeanLuma();

                // A second camera pointed somewhere completely different — straight
                // up at the empty background, which no framing of the lit scene can
                // be confused with.
                auto addedSecond = ObjectFactory::createCamera(document_.scene());
                auto* secondRaw = addedSecond.get();
                addObject(addedSecond, document_.scene(), "Add Second Camera");
                secondRaw->position.set(0.f, 2.f, 0.f);
                secondRaw->lookAt(Vector3(0.f, 12.f, 0.f));
                secondRaw->updateMatrixWorld();
                selectObject(secondRaw);
                step(8);

                const double lumaSecond = dockMeanLuma();
                check(lumaFirst >= 0.0 && lumaSecond >= 0.0 && std::abs(lumaFirst - lumaSecond) > 4.0,
                      "selecting another camera re-aims the dock at it");
                if (dockPane_.supported()) {
                    check(dockPane_.handle() == handleBefore && handleBefore != 0,
                          "and does it by re-pointing ONE view, not by churning a new one");
                }

                // Play and Stop replace the scene, and with it every camera object
                // in it — behind the SAME uuid. A pane that cached the Camera* (or
                // skipped the update when the uuid matched) would hand the renderer
                // a pointer into freed memory, and it would survive the uuid check
                // precisely because the uuid is what stayed the same.
                // Every raw pointer into the scene dies here, so hold the uuids and
                // re-resolve after — the same discipline the pane itself follows.
                const std::string secondUuid = secondRaw->uuid;
                const std::string firstUuid = cameraRaw->uuid;
                startPlay();
                step(6);
                check(isPlaying(), "play starts with a camera docked");
                stopPlay();
                step(8);
                check(!isPlaying(), "and stops again");
                check(dockMeanLuma() >= 0.0, "the dock still renders across a scene replace");
                if (dockPane_.supported()) {
                    check(dockPane_.handle() != 0, "the pane still holds its view after play/stop");
                }

                auto* secondLive = dynamic_cast<Camera*>(findByUuid(document_.scene(), secondUuid));
                check(secondLive != nullptr, "the docked camera comes back from play by uuid");
                // Nothing after this point is meaningful without it, and every
                // later block navigates from these cameras.
                if (!secondLive) {
                    summary();
                    return 1;
                }

                // --- the dock is NOT the selection ----------------------------
                // The whole point of the dock is watching what a camera sees while
                // you work on what it is pointed at — which means selecting that
                // something. Deselecting, or selecting anything else, used to
                // blank the dock; nothing about either is a statement about which
                // camera you are framing with.
                {
                    const double docked = dockMeanLuma();

                    selectObject(nullptr);
                    step(4);
                    check(dockCamera() == secondLive, "deselecting leaves the dock camera alone");
                    check(docked >= 0.0 && std::abs(dockMeanLuma() - docked) < 2.0,
                          "and the dock keeps rendering it");

                    if (auto* box = document_.scene().getObjectByName("Box")) {
                        selectObject(box);
                        step(4);
                        check(dockCamera() == secondLive,
                              "selecting the object being framed does not take the camera away");
                        check(docked >= 0.0 && std::abs(dockMeanLuma() - docked) < 2.0,
                              "and the dock still shows what that camera sees");
                    }
                    selectObject(nullptr);
                    step(2);
                }

                // --- the picker aims it, without touching the selection --------
                {
                    auto* firstLive = dynamic_cast<PerspectiveCamera*>(
                            findByUuid(document_.scene(), firstUuid));
                    check(firstLive != nullptr, "the first camera survives to be picked");
                    if (firstLive) {
                        setDockCamera(firstLive);
                        step(6);
                        check(dockCamera() == firstLive, "the picker re-aims the dock");
                        check(selection_.get() == nullptr,
                              "and does not select anything to do it");

                        // "None" means none: the fallback that covers a deleted
                        // camera must not undo a deliberate choice.
                        setDockCamera(nullptr);
                        step(4);
                        check(dockCamera() == nullptr, "None empties the dock and stays empty");
                        if (dockPane_.supported()) {
                            check(dockPane_.handle() == 0, "and hands the secondary view back");
                        }

                        setDockCamera(secondLive);
                        step(6);
                        check(dockCamera() == secondLive, "and the dock takes a camera again");
                        if (dockPane_.supported()) {
                            check(dockPane_.handle() != 0, "with a view to render it into");
                        }
                    }
                }

                // The crash a user found, at the churn rate the picker can now
                // produce: None and back is release + create of a real allocation,
                // every time.
                for (int i = 0; i < 8; ++i) {
                    setDockCamera(nullptr);
                    step(2);
                    setDockCamera(secondLive);
                    step(2);
                }
                check(dockMeanLuma() >= 0.0, "the dock survives repeated release/retake");

                // A collapsed dock has no business holding a deferred chain.
                if (dockPane_.supported()) {
                    bottomPanelOpen_ = false;
                    step(3);
                    check(dockPane_.handle() == 0, "collapsing the dock frees its view");
                    bottomPanelOpen_ = true;
                    step(6);
                    check(dockPane_.handle() != 0, "and reopening takes one back");
                }

                // An orthographic scene camera is a legal dock subject: secondary
                // views claim ortho support, and the projection reaches the shaders
                // through camAux rather than the eye-point idiom a perspective view
                // can assume.
                {
                    auto ortho = OrthographicCamera::create(-4.f, 4.f, 3.f, -3.f, 0.1f, 100.f);
                    ortho->name = "Ortho Camera";
                    ortho->position.set(-4.f, 3.5f, 4.f);
                    ortho->lookAt(Vector3(0.f, 0.5f, 0.f));
                    ortho->updateMatrixWorld();
                    auto* orthoRaw = ortho.get();
                    const std::string orthoUuid = orthoRaw->uuid;
                    addObject(ortho, document_.scene(), "Add Ortho Camera");
                    selectObject(orthoRaw);
                    step(10);
                    check(dockMeanLuma() > 8.0, "an orthographic camera renders in the dock");
                    selectObject(orthoRaw);
                    deleteSelected();
                    step(6);
                    // A camera that leaves the scene must not take the dock down
                    // with it: the dock falls back to another camera rather than
                    // going dark on a delete (or on the undo of an Add).
                    check(dockCamera() != nullptr && dockCamera()->uuid != orthoUuid,
                          "deleting the docked camera falls the dock back to another");
                    check(dockMeanLuma() >= 0.0, "and it keeps rendering");
                }

                selectObject(secondLive);
                deleteSelected();
                step(2);
                // cameraRaw died with the scene replace above; the rest of this
                // suite still expects to be holding the first camera.
                cameraRaw = dynamic_cast<PerspectiveCamera*>(findByUuid(document_.scene(), firstUuid));
                check(cameraRaw != nullptr, "and so does the camera the suite started with");
                if (!cameraRaw) {
                    summary();
                    return 1;
                }
                selectObject(cameraRaw);
                step(2);
            }
        }

        deleteSelected();
        step();
        check(viewportMarkers_.size() == lightMarkers, "deleting the camera drops its marker");
        check(cameraHelper_ == nullptr, "deleting the camera drops the frustum helper");
        commands_.undo();// leave the scene as we found it
        step();
    }

    // Every light kind carries its own icon, and each is a separate SVG. One
    // marker per added light is what proves all of them parse — a broken path
    // set would silently produce no marker for just that kind.
    if (section("light-icons")) {
        const LightKind kinds[] = {LightKind::Directional, LightKind::Point, LightKind::Spot,
                                   LightKind::Ambient, LightKind::Hemisphere};
        const auto baseline = viewportMarkers_.size();
        for (auto kind : kinds) {
            addObject(ObjectFactory::createLight(kind, document_.scene()),
                      document_.scene(), "Add Light");
        }
        step();
        check(viewportMarkers_.size() == baseline + std::size(kinds),
              "every light kind gets its own marker icon");
        for (std::size_t i = 0; i < std::size(kinds); ++i) commands_.undo();
        step();
        check(viewportMarkers_.size() == baseline, "removing the lights drops their markers");
    }

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // Python scripting, through the paths a user actually takes: attach a file,
    // let the inspector discover its fields, play, stop. Plus a script that
    // raises every frame, because the one thing scripting must never do is take
    // the editor down with it.
    if (section("python-scripting")) {
        const auto dir = std::filesystem::current_path();
        const auto spinnerPath = dir / "spinner.py";
        const auto throwerPath = dir / "thrower.py";
        {
            std::ofstream out(spinnerPath, std::ios::trunc);
            out << "class Spinner:\n"
                << "    speed = 1.5\n"
                << "\n"
                << "    def start(self, obj):\n"
                << "        self.obj = obj\n"
                << "\n"
                << "    def update(self, dt):\n"
                << "        self.obj.rotation.y += self.speed * dt\n"
                << "\n"
                << "    def stop(self):\n"
                << "        pass\n";
        }
        {
            std::ofstream out(throwerPath, std::ios::trunc);
            out << "class Thrower:\n"
                << "    def update(self, dt):\n"
                << "        raise RuntimeError('selftest: this must not be fatal')\n";
        }

        auto* scripted = document_.scene().getObjectByName("Box");
        check(scripted != nullptr, "Box available for the scripting drive");

        if (scripted) {
            assignScript(*scripted, spinnerPath);
            // Draw the section: this is what discovers the class and its fields.
            selectObject(scripted);
            step(2);

            const auto stored = ScriptConfig::read(*scripted);
            check(stored && !stored->path.empty(), "the script is recorded in userData");

            const auto inspection = scripting::inspect(spinnerPath);
            check(inspection.error.empty() && inspection.className == "Spinner",
                  "the script class is discovered");
            check(inspection.fields.size() == 1 && inspection.fields.front().name == "speed",
                  "the exposed parameter is discovered");

            const float restY = scripted->rotation.y;
            const auto undosBefore = commands_.undoCount();

            startPlay();
            check(isPlaying(), "play starts with a script attached");
            // Fixed dt here and in the script drives below: every one of them
            // asserts on `something += rate * dt` having accumulated, so the
            // quantity under test is simulated seconds, not frames.
            stepFixed(30);

            auto* live = document_.scene().getObjectByName("Box");
            check(live && live->rotation.y > restY + 1e-3f, "the script drives the object");
            // Measured before the stop: a scene replace drops commands that
            // cannot rebind, so afterwards the count is not comparable.
            check(commands_.undoCount() == undosBefore, "a running script pushes no undo entries");

            stopPlay();
            step();

            auto* restored = document_.scene().getObjectByName("Box");
            check(restored != nullptr, "the object survives the round trip");
            check(restored && std::abs(restored->rotation.y - restY) < 1e-4f,
                  "stop restores the pose the script changed");
            check(restored && ScriptConfig::read(*restored).has_value(),
                  "the script reference survives the round trip");

            // A script that raises every single frame: reported once, disabled,
            // and the editor plays on.
            if (restored) {
                assignScript(*restored, throwerPath);
                const auto uuid = restored->uuid;
                startPlay();
                step(10);
                check(isPlaying(), "a raising script does not abort play");
                stopPlay();
                step();
                check(scripts_ && !scripts_->errorFor(uuid).empty(),
                      "the failure is recorded against the object");

                if (auto* clear = document_.scene().getObjectByName("Box")) {
                    assignScript(*clear, {});
                    check(!ScriptConfig::read(*clear).has_value(), "clearing removes the script");
                }
            }
        }

        std::error_code ec;
        std::filesystem::remove(spinnerPath, ec);
        std::filesystem::remove(throwerPath, ec);

        // The same drive again with no file anywhere: source authored in the
        // Script Editor and stored in the scene. Through the tab's own apply
        // path, which is what the user's Ctrl+Enter runs.
        if (auto* box = document_.scene().getObjectByName("Box")) {

            const auto undosBefore = commands_.undoCount();

            const auto boxUuid = box->uuid;
            openScriptEditor(*box);
            check(scriptEditorFor(boxUuid) != nullptr, "the Script Editor opens on the object");
            check(bottomPanelOpen_, "opening it brings the bottom panel up with it");
            check(activeScriptEditor() && activeScriptEditor()->uuid == boxUuid,
                  "and that script is the one Apply acts on");
            // Indented with TABS on purpose: Apply has to normalize them, or
            // this source is a TabError waiting for the first space-indented
            // line somebody adds later.
            scriptEditorFor(boxUuid)->buffer =
                    "class Inline:\n"
                    "\tspeed = 2.0\n"
                    "\n"
                    "\tdef start(self, obj):\n"
                    "\t\tself.obj = obj\n"
                    "\n"
                    "\tdef update(self, dt):\n"
                    "\t\tself.obj.rotation.y += self.speed * dt\n";
            applyScriptEditor();
            step();

            auto stored = ScriptConfig::read(*box).value_or(ScriptConfig{});
            check(stored.isInline(), "the inline source is recorded in userData");
            check(stored.path.empty(), "an inline script carries no file path");
            check(stored.source.find('\t') == std::string::npos, "Apply normalizes tabs to spaces");
            check(scriptEditorFor(boxUuid)->status.empty(), "valid source passes the syntax check");
            check(commands_.undoCount() == undosBefore + 1, "Apply is one undo entry");

            commands_.undo();
            step();
            check(!ScriptConfig::read(*box).has_value(), "undo takes the inline script back off");
            check(scriptEditorFor(boxUuid) == nullptr,
                  "the script's tab closes when its script is undone away");
            commands_.redo();
            step();
            check(ScriptConfig::read(*box).has_value(), "redo puts it back");

            const auto inspection = scripting::inspectSource(stored.source, box->uuid, "Box");
            check(inspection.error.empty() && inspection.className == "Inline",
                  "the inline class is discovered with no file name to match");
            check(inspection.fields.size() == 1 && inspection.fields.front().name == "speed",
                  "the inline script's parameter is discovered");

            // And through the inspector's own cache, which has no file write
            // time to key on and uses a hash of the text instead.
            const auto& cached = inspectScriptSource(box->uuid, "Box", stored.source);
            check(cached.className == "Inline" && cached.fields.size() == 1,
                  "the inspector discovers inline parameters through its cache");
            const auto& reinspected = inspectScriptSource(
                    box->uuid, "Box", "class Edited:\n    def update(self, dt):\n        pass\n");
            check(reinspected.className == "Edited", "changed source is inspected again");

            const float restY = box->rotation.y;
            startPlay();
            stepFixed(30);
            auto* live = document_.scene().getObjectByName("Box");
            check(live && live->rotation.y > restY + 1e-3f, "inline source drives the object");
            stopPlay();
            step();

            auto* restored = document_.scene().getObjectByName("Box");
            check(restored && std::abs(restored->rotation.y - restY) < 1e-4f,
                  "stop restores the pose the inline script changed");

            const auto after = restored ? ScriptConfig::read(*restored).value_or(ScriptConfig{})
                                        : ScriptConfig{};
            check(after.isInline() && after.source == stored.source,
                  "the inline source survives the play/stop round trip");

            // Source that does not parse: reported by Apply, saved anyway (a
            // half-written script is a normal thing to want to keep), and Play
            // survives it.
            if (restored) {
                const auto uuid = restored->uuid;
                openScriptEditor(*restored);
                scriptEditorFor(uuid)->buffer =
                        "class Broken:\n    def update(self dt):\n        pass\n";
                applyScriptEditor();
                step();

                check(!scriptEditorFor(uuid)->status.empty(), "Apply reports the syntax error");
                check(scriptEditorFor(uuid)->status.find("line 2") != std::string::npos,
                      "the syntax error carries its line number");
                check(ScriptConfig::read(*restored).value_or(ScriptConfig{}).source ==
                              scriptEditorFor(uuid)->buffer,
                      "a syntax error does not block Apply");

                startPlay();
                step(5);
                check(isPlaying(), "a broken inline script does not abort play");
                stopPlay();
                step();
                check(scripts_ && !scripts_->errorFor(uuid).empty(),
                      "the inline failure is recorded against the object");

                // The New Inline Script template must itself run: it carries
                // `import threepp` and an Object3D annotation for IDE
                // completion, and this is what proves that import resolves
                // inside a script module against the embedded interpreter.
                // Placed here, on a re-resolved Box, with nothing after it
                // reusing pre-play pointers — the crash the first placement
                // caused is exactly the scene-replace staleness this file
                // keeps having to respect.
                if (auto* target = document_.scene().getObjectByName("Box")) {
                    setInlineScript(*target, inlineScriptTemplate(), "New Inline Script");
                    step();
                    const float beforeTemplate = target->rotation.y;
                    startPlay();
                    stepFixed(30);
                    auto* driven = document_.scene().getObjectByName("Box");
                    check(driven && driven->rotation.y > beforeTemplate + 1e-3f,
                          "the template script runs as generated (import threepp resolves)");
                    stopPlay();
                    step();
                }

                // threepp.editor.scene(): a real handle on the played scene —
                // this script ignores its own object and drives Ground through a
                // lookup. Called from update() rather than start(), which is the
                // half the old scene-as-a-start-argument could not do at all.
                if (auto* target = document_.scene().getObjectByName("Box")) {
                    setInlineScript(*target,
                                    "import threepp\n"
                                    "\n"
                                    "class Reacher:\n"
                                    "    def update(self, dt):\n"
                                    "        scene = threepp.editor.scene()\n"
                                    "        scene.get_object_by_name('Ground').position.y += dt\n",
                                    "Scene-Reaching Script");
                    step();
                    const float groundBefore =
                            document_.scene().getObjectByName("Ground")->position.y;
                    startPlay();
                    stepFixed(30);
                    auto* ground = document_.scene().getObjectByName("Ground");
                    check(ground && ground->position.y > groundBefore + 1e-3f,
                          "threepp.editor.scene() answers during Play and reaches other objects");
                    stopPlay();
                    step();
                    auto* groundRestored = document_.scene().getObjectByName("Ground");
                    check(groundRestored &&
                                  std::abs(groundRestored->position.y - groundBefore) < 1e-4f,
                          "stop restores what a scene-reaching script moved");
                }

                // threepp.editor.script_from_object: reaching another object's
                // live INSTANCE rather than its node. No event bus and no
                // message type — the Button calls a method on the Door's own
                // Python object, and that is the whole signal.
                //
                // The roles are this way round deliberately. The Button sits on
                // Ground, which the template scene adds BEFORE Box, so the Door
                // it resolves in start() has not been started — or, before the
                // two-phase split, even constructed — when it asks. Swap them
                // and the test passes for the wrong reason.
                if (auto* button = document_.scene().getObjectByName("Ground")) {
                    setInlineScript(*button,
                                    "import threepp\n"
                                    "\n"
                                    "class Button:\n"
                                    "    def start(self, obj):\n"
                                    "        self.door = threepp.editor.script_from_object(\n"
                                    "            threepp.editor.scene().get_object_by_name('Box'))\n"
                                    "\n"
                                    "    def update(self, dt):\n"
                                    "        if self.door is not None:\n"
                                    "            self.door.on_opened()\n",
                                    "Button Script");
                    if (auto* door = document_.scene().getObjectByName("Box")) {
                        setInlineScript(*door,
                                        "class Door:\n"
                                        "    def start(self, obj):\n"
                                        "        self.obj = obj\n"
                                        "\n"
                                        "    def on_opened(self):\n"
                                        "        self.obj.position.y += 0.01\n",
                                        "Door Script");
                    }
                    const auto buttonUuid = button->uuid;
                    step();
                    const float doorBefore = document_.scene().getObjectByName("Box")->position.y;
                    startPlay();
                    stepFixed(30);
                    auto* opened = document_.scene().getObjectByName("Box");
                    check(opened && opened->position.y > doorBefore + 1e-3f,
                          "one script drives another's live instance through script_from_object");
                    check(scripts_ && opened && scripts_->errorFor(opened->uuid).empty() &&
                                  scripts_->errorFor(buttonUuid).empty(),
                          "and neither side of the handshake raised");
                    stopPlay();
                    step();
                    // The Button was only ever for this pass; the Door on Box is
                    // cleared by the block below.
                    if (auto* restored = document_.scene().getObjectByName("Ground")) {
                        assignScript(*restored, {});
                    }
                    step();
                }

                if (auto* clear = document_.scene().getObjectByName("Box")) {
                    const auto uuid = clear->uuid;
                    assignScript(*clear, {});
                    step();
                    check(!ScriptConfig::read(*clear).has_value(), "clearing removes the inline script");
                    check(scriptEditorFor(uuid) == nullptr, "clearing the script closes its tab");
                }
            }
        }
    }
#endif

    // --- two scripts open at once ------------------------------------------
    // The editor used to be ONE buffer that retargeted itself onto whatever was
    // selected, which is unusable for the thing people actually do: write two
    // scripts that talk to each other. Each open script now keeps its own tab,
    // its own buffer and its own syntax error, and the only thing the selection
    // does is raise a tab that already exists.
    if (section("two-scripts")) {
        auto* first = document_.scene().getObjectByName("Box");
        auto* second = document_.scene().getObjectByName("Ground");
        if (first && second) {

            const auto firstUuid = first->uuid;
            const auto secondUuid = second->uuid;

            setInlineScript(*first, "class A:\n    def update(self, dt):\n        pass\n", "Script A");
            setInlineScript(*second, "class B:\n    def update(self, dt):\n        pass\n", "Script B");
            step();

            openScriptEditor(*first);
            openScriptEditor(*second);
            step();

            check(scriptEditors_.size() == 2, "two scripts open as two tabs");
            check(scriptEditorFor(firstUuid) && scriptEditorFor(secondUuid),
                  "each object keeps its own");
            check(activeScriptEditor() && activeScriptEditor()->uuid == secondUuid,
                  "the one opened last is the one in front");

            // With `first` selected the whole time: an explicit open outranks
            // the raise-on-selection rule, which otherwise pulls the bar back to
            // the selected object on the next frame and makes opening a script
            // for anything else impossible.
            selectObject(first);
            step();
            openScriptEditor(*second);
            step(3);
            check(activeScriptEditor() && activeScriptEditor()->uuid == secondUuid,
                  "opening a script for an object that is not selected stays open");

            // The whole point: an edit in one is not an edit in the other, and
            // neither is silently retargeted by clicking around the scene.
            const std::string edited = "class A:\n    speed = 3.0\n"
                                       "    def update(self, dt):\n        pass\n";
            scriptEditorFor(firstUuid)->buffer = edited;
            selectObject(second);
            step();
            check(scriptEditorFor(firstUuid) && scriptEditorFor(firstUuid)->buffer == edited,
                  "an unsaved buffer survives the selection moving off it");
            check(scriptEditorFor(secondUuid) &&
                          scriptEditorFor(secondUuid)->buffer.find("class B") != std::string::npos,
                  "and the other tab still holds its own object's source");

            // Selecting an object that has a tab raises it; selecting one that
            // does not opens nothing (that is what Edit… is for).
            selectObject(first);
            step();
            check(activeScriptEditor() && activeScriptEditor()->uuid == firstUuid,
                  "selecting an object with a tab open raises that tab");
            selectObject(&document_.scene());
            step();
            check(scriptEditors_.size() == 2,
                  "selecting something without one opens nothing");

            // Reopening a tab that is already there keeps what was typed into
            // it rather than reloading over the top of it.
            openScriptEditor(*first);
            step();
            check(scriptEditors_.size() == 2, "reopening an object reuses its tab");
            check(scriptEditorFor(firstUuid)->buffer == edited, "and keeps the unsaved text in it");

            // Apply acts on the visible one, and only on it.
            const auto sourceOf = [this](const char* name) {
                auto* object = document_.scene().getObjectByName(name);
                return object ? ScriptConfig::read(*object).value_or(ScriptConfig{}).source
                              : std::string{};
            };
            applyScriptEditor();
            step();
            check(sourceOf("Box").find("speed = 3.0") != std::string::npos,
                  "Apply commits the visible script");
            check(sourceOf("Ground").find("class B") != std::string::npos,
                  "and leaves the other object's alone");

            // Both survive a play/stop: the graph is replaced wholesale and the
            // tabs are keyed by uuid precisely so they do not go with it.
            startPlay();
            step(3);
            stopPlay();
            step();
            check(scriptEditors_.size() == 2, "both tabs survive play/stop");
            check(scriptEditorFor(firstUuid) && !scriptEditorFor(firstUuid)->missing,
                  "and are still pointing at live objects");

            // Closing one leaves the other, and closing the last takes the
            // Scripts tab with it.
            scriptEditorFor(firstUuid)->open = false;
            step();
            check(scriptEditors_.size() == 1 && scriptEditorFor(secondUuid),
                  "closing one tab leaves the other");
            scriptEditorFor(secondUuid)->open = false;
            step();
            check(scriptEditors_.empty(), "closing the last leaves no Scripts tab at all");

            if (auto* a = document_.scene().getObjectByName("Box")) assignScript(*a, {});
            if (auto* b = document_.scene().getObjectByName("Ground")) assignScript(*b, {});
            step();
        }
    }

    // "Edit in VS Code", end to end and without VS Code: the export, the
    // workspace, the poll, the sync back through the Script Editor's Apply, and
    // every way a session ends. The launch itself is the one thing gated out —
    // a test run must not open windows on the machine running it.
    //
    // Outside the Python guard on purpose: none of this needs an interpreter.
    // The compile check is the only part that does, and it is not what a file
    // watcher is for.
    if (section("external-edit")) {
        const auto pump = [&](float seconds) {
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::milliseconds(static_cast<int>(seconds * 1000));
            while (std::chrono::steady_clock::now() < deadline) step();
        };
        // Frame rate is whatever the machine gives us, so wait on the event
        // rather than on a frame count — with a bound, so a failure is a
        // failure rather than a hang.
        const auto pumpUntilSync = [&](int target, float seconds = 10.f) {
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::milliseconds(static_cast<int>(seconds * 1000));
            while (externalEdit_.syncs < target && std::chrono::steady_clock::now() < deadline) step();
            return externalEdit_.syncs >= target;
        };

        std::error_code ec;
        // The scratch directory outlives the process, so start from nothing —
        // otherwise "was the settings file created" is answered by a previous
        // run.
        std::filesystem::remove_all(ScriptWorkspace::scratchDir(), ec);

        // Tier 1: a .py already on disk. Nothing is watched (every Play
        // recompiles it), but the workspace still has to appear beside it.
        {
            const auto dir = std::filesystem::temp_directory_path() / "threepp-editor-selftest";
            std::filesystem::remove_all(dir, ec);
            std::filesystem::create_directories(dir, ec);
            const auto file = dir / "external.py";
            ScriptWorkspace::writeSource(file, "class External:\n    pass\n");

            openScriptFileExternally(file);
            check(std::filesystem::exists(dir / ".vscode" / "settings.json"),
                  "opening a file script generates a workspace beside it");
            std::filesystem::remove_all(dir, ec);
        }

        if (auto* box = document_.scene().getObjectByName("Box")) {

            const auto uuid = box->uuid;
            const std::string source = "class External:\n"
                                       "    speed = 1.0\n"
                                       "\n"
                                       "    def update(self, dt):\n"
                                       "        pass\n";
            setInlineScript(*box, source, "Inline Script");
            step();

            startExternalEdit(*box);
            check(externalEdit_.active, "an external session starts on an inline script");
            const auto scratch = externalEdit_.file;
            check(std::filesystem::exists(scratch), "the source is exported to a scratch file");
            check(ScriptWorkspace::readSource(scratch) == source,
                  "the export is the committed source, byte for byte");
            check(scriptEditorFor(uuid) != nullptr, "the Script Editor comes along to show the session");

            // The workspace: written once, carrying this build's stub path, and
            // never written over the top of what the user made of it.
            const auto settings = scratch.parent_path() / ".vscode" / "settings.json";
            check(std::filesystem::exists(settings), "a workspace is generated for the scratch folder");
            const auto generated = ScriptWorkspace::readSource(settings);
            check(generated.find("python.analysis.stubPath") != std::string::npos,
                  "the generated settings carry a stub path");
            check(generated.find(pythonStubDir().generic_string()) != std::string::npos,
                  "and it is this build's own stub directory");

            // In a directory of its own, so the generated file above survives
            // the run — it is what a user's scratch folder ends up holding, and
            // clobbering it with a sentinel would make that unreadable.
            {
                const auto dir = std::filesystem::temp_directory_path() / "threepp-editor-selftest-ws";
                std::filesystem::remove_all(dir, ec);
                std::filesystem::create_directories(dir, ec);
                const auto file = dir / ".vscode" / "settings.json";

                ensureScriptWorkspace(dir);
                check(std::filesystem::exists(file), "a workspace is generated where there was none");

                const std::string sentinel = "{ \"mine\": true }\n";
                ScriptWorkspace::writeSource(file, sentinel);
                ensureScriptWorkspace(dir);
                check(ScriptWorkspace::readSource(file) == sentinel,
                      "regenerating never overwrites an existing settings.json");

                std::filesystem::remove_all(dir, ec);
            }

            // A save from another program: CRLF endings and a tab-indented
            // line, which is what an editor that is not this one produces.
            const auto undosBefore = commands_.undoCount();
            ScriptWorkspace::writeSource(scratch, "class External:\r\n"
                                                  "    speed = 2.5\r\n"
                                                  "\r\n"
                                                  "    def update(self, dt):\r\n"
                                                  "\t\tpass\r\n");
            check(pumpUntilSync(1), "a save is picked up by the poll");

            const auto stored = [&] {
                auto* live = findByUuid(document_.scene(), uuid);
                return live ? ScriptConfig::read(*live).value_or(ScriptConfig{}) : ScriptConfig{};
            };
            check(stored().source.find("speed = 2.5") != std::string::npos,
                  "the external edit reaches userData");
            check(stored().source.find('\r') == std::string::npos, "CRLF is normalized away");
            check(stored().source.find('\t') == std::string::npos, "tabs are normalized away");
            check(commands_.undoCount() == undosBefore + 1, "the first sync is its own undo entry");

            commands_.undo();
            step();
            check(stored().source == source, "an external sync is undoable");
            commands_.redo();
            step();
            check(stored().source.find("speed = 2.5") != std::string::npos, "and redoable");

            // Ten saves must not be ten undo steps: the session holds one
            // transaction open, so everything after the first merges into it.
            const auto undosAfterFirst = commands_.undoCount();
            ScriptWorkspace::writeSource(scratch, "class External:\n    speed = 3.5\n");
            check(pumpUntilSync(2), "a second save syncs too");
            check(commands_.undoCount() == undosAfterFirst,
                  "successive saves coalesce into one undo step");
            check(stored().source.find("speed = 3.5") != std::string::npos,
                  "and the document holds the newest text");

            // The same text saved again, CRLF-terminated: a new write time,
            // nothing new to say. Without normalizing before comparing, every
            // save in VS Code would land as an edit.
            const auto syncsBefore = externalEdit_.syncs;
            const auto undosBeforeIdle = commands_.undoCount();
            ScriptWorkspace::writeSource(scratch, "class External:\r\n    speed = 3.5\r\n");
            pump(ExternalEditState::pollSeconds * 2.5f);
            check(externalEdit_.syncs == syncsBefore, "a save that changed nothing is not a sync");
            check(commands_.undoCount() == undosBeforeIdle, "and pushes no undo entry");

            // Play, and a save while playing: parked, because Stop puts the
            // snapshot back and would swallow the commit with it.
            startPlay();
            step(5);
            ScriptWorkspace::writeSource(scratch, "class External:\n    speed = 4.5\n");
            pump(ExternalEditState::pollSeconds * 1.5f);
            check(externalEdit_.active, "the session survives Play");
            check(externalEdit_.waiting, "a save during Play is parked rather than committed");
            check(externalEdit_.syncs == syncsBefore, "and nothing is committed while playing");

            stopPlay();
            step();
            check(pumpUntilSync(syncsBefore + 1), "the parked save applies after Stop");
            check(stored().source.find("speed = 4.5") != std::string::npos,
                  "on the object the snapshot restore rebuilt, found by uuid");
            check(externalEdit_.active, "and the session is still live after the round trip");

            // Clearing the script is the end of it: nothing to watch, and the
            // scratch file goes with the session.
            if (auto* clear = findByUuid(document_.scene(), uuid)) {
                assignScript(*clear, {});
                step();
                pump(ExternalEditState::pollSeconds * 1.5f);
                check(!externalEdit_.active, "clearing the script ends the session");
                check(!std::filesystem::exists(scratch), "and takes the scratch file with it");
            }
        }
    }

    // Splines. A spline is a Group whose children are its control points, so
    // everything here runs against ordinary scene nodes and the commands that
    // already move them — which is the whole claim the design rests on.
    if (section("splines")) {
        const auto splineNow = [&](const std::string& uuid) {
            return findByUuid(document_.scene(), uuid);
        };

        const auto markersBefore = viewportMarkers_.size();

        auto created = ObjectFactory::createSpline(document_.scene());
        const auto splineUuid = created->uuid;
        addObject(created, document_.scene(), "Add Spline");
        created.reset();// the command owns it now; nothing here may outlive a replace
        step();

        auto* spline = splineNow(splineUuid);
        check(spline && SplineConfig::isSpline(*spline), "the factory creates a spline");
        check(spline && spline->children.size() == 4, "with control points as its children");
        check(splineOverlays_.size() == 1, "the spline gets a curve overlay");
        check(viewportMarkers_.size() == markersBefore + 4,
              "every control point gets a marker icon");

        // The live-update contract (what examples/.../spline_editor.cpp always
        // did): dragging a point rewrites the SAME position attribute in place
        // and bumps its version. A fresh attribute per move is the bug this
        // pins down — the renderer keys GPU buffers on attribute identity, so a
        // recycled pointer read as already uploaded and the drawn curve froze
        // until play/stop rebuilt the whole line.
        if (spline && splineOverlays_.size() == 1) {
            const auto geometry = splineOverlays_.front().line->geometry();
            auto* attribute = geometry->getAttribute<float>("position");
            const auto version = attribute->version;
            auto* dragged = spline->children.back();
            dragged->position.y += 5;
            step();
            check(splineOverlays_.front().line->geometry() == geometry &&
                          geometry->getAttribute<float>("position") == attribute,
                  "moving a point keeps the same curve buffer");
            check(attribute->version > version,
                  "and re-uploads it, so the drawn curve follows the drag");
            const auto count = geometry->drawRange.count;
            const Vector3 end(attribute->getX(count - 1), attribute->getY(count - 1),
                              attribute->getZ(count - 1));
            check(count > 0 && end.distanceTo(dragged->position) < 1e-3f,
                  "with the curve's end sitting on the moved point");
            dragged->position.y -= 5;
            step();
        }

        // Add Point: appended, undoable, and the curve is longer for it.
        if (spline) {
            const auto config = SplineConfig::read(*spline).value_or(SplineConfig{});
            const float lengthBefore = config.curve(*spline)->getLength();
            const auto overlayGeometry = splineOverlays_.empty()
                                                 ? std::shared_ptr<BufferGeometry>{}
                                                 : splineOverlays_.front().line->geometry();

            addSplinePoint(*spline, AddObjectCommand::atEnd, "Add Spline Point");
            step();
            spline = splineNow(splineUuid);
            check(spline && spline->children.size() == 5, "Add Point appends a control point");
            check(spline && config.curve(*spline)->getLength() > lengthBefore + 1e-3f,
                  "and the point lands past the end, so the curve extends");
            check(!splineOverlays_.empty() &&
                          splineOverlays_.front().line->geometry() != overlayGeometry,
                  "a longer curve swaps in a bigger buffer instead of writing past the old one");
            check(selection_.get() == (spline ? spline->children.back() : nullptr),
                  "the new point is selected, ready to drag");

            commands_.undo();
            step();
            spline = splineNow(splineUuid);
            check(spline && spline->children.size() == 4, "undo takes the point back off");
            commands_.redo();
            step();
            spline = splineNow(splineUuid);
            check(spline && spline->children.size() == 5, "redo puts it back");
            commands_.undo();
            step();
            spline = splineNow(splineUuid);
        }

        // Insert: at a specific index, midway to the neighbour it went in next
        // to — an insert that leaves the curve unchanged is not one.
        if (spline) {
            const auto before = SplineConfig::controlPoints(*spline);
            addSplinePoint(*spline, 1, "Insert Spline Point");
            step();
            spline = splineNow(splineUuid);
            const auto after = SplineConfig::controlPoints(*spline);
            check(after.size() == before.size() + 1, "Insert Before adds a control point");
            Vector3 midpoint;
            midpoint.copy(before[0]).add(before[1]).multiplyScalar(0.5f);
            check(after.size() > 1 && after[1].distanceTo(midpoint) < 1e-4f,
                  "the inserted point sits midway between its neighbours");
            check(after[0].distanceTo(before[0]) < 1e-4f &&
                          after[2].distanceTo(before[1]) < 1e-4f,
                  "and the existing points keep their order");

            commands_.undo();
            step();
            spline = splineNow(splineUuid);
            check(spline && spline->children.size() == before.size(), "the insert is undoable");
        }

        // Removing a point is the ordinary delete — no spline-specific path,
        // which is the claim the whole design makes.
        if (spline && !spline->children.empty()) {
            const auto count = spline->children.size();
            const auto markers = viewportMarkers_.size();

            selectObject(spline->children.back());
            deleteSelected();
            step();
            spline = splineNow(splineUuid);
            check(spline && spline->children.size() == count - 1,
                  "Del on a control point takes it out of the curve");
            check(viewportMarkers_.size() == markers - 1, "and its marker icon with it");
            check(splineOverlays_.size() == 1, "while the spline keeps its curve");

            commands_.undo();
            step();
            spline = splineNow(splineUuid);
            check(spline && spline->children.size() == count, "and the delete is undoable");
        }

        // Generated geometry. The mesh is a REAL child of the spline — saved,
        // materialled, physics-configurable — and the only thing separating it
        // from a control point is its tag, which makes every count and index
        // above the part of this feature most likely to be off by one.
        const auto derivedCount = [](const Object3D& owner) {
            std::size_t n = 0;
            for (const auto* child : owner.children) {
                if (SplineConfig::isDerived(*child)) ++n;
            }
            return n;
        };
        // The inspector's edit path: a PropertyCommand carrying the encoded
        // config, so undo below goes through the same stack the UI uses.
        const auto setConfig = [&](const SplineConfig& after, const char* label) {
            auto* target = splineNow(splineUuid);
            if (!target) return;
            const auto before = SplineConfig::read(*target).value_or(SplineConfig{});
            commands_.execute(makeProperty<SplineConfig>(
                    label, "spline:" + target->uuid,
                    [target](const SplineConfig& value) { value.write(*target); },
                    before, after));
        };

        std::string derivedUuid;
        if (spline) {
            // Selected throughout, so the inspector's Geometry block is drawn
            // for every mesh kind this exercises rather than for none of them.
            selectObject(spline);
            step();

            const auto pointsBefore = SplineConfig::controlPoints(*spline).size();
            const auto markersBefore2 = viewportMarkers_.size();

            auto config = SplineConfig::read(*spline).value_or(SplineConfig{});
            config.mesh = SplineConfig::MeshKind::Tube;
            config.radius = 0.4f;
            setConfig(config, "Spline Mesh");
            step();
            spline = splineNow(splineUuid);

            check(spline && derivedCount(*spline) == 1,
                  "enabling a tube adds exactly one derived child");
            auto* derived = spline ? SplineConfig::derivedMesh(*spline) : nullptr;
            auto* mesh = derived ? derived->as<Mesh>() : nullptr;
            check(mesh != nullptr, "which is a Mesh, and part of the document");
            check(spline && SplineConfig::controlPoints(*spline).size() == pointsBefore,
                  "while the control point count is what it was");
            check(viewportMarkers_.size() == markersBefore2,
                  "and no control-point marker appears for it");
            check(derived && SplineConfig::splineOf(*derived) == nullptr,
                  "the tag, not the parent link, is what says it is not a point");

            if (mesh) {
                derivedUuid = mesh->uuid;

                // Everything a user may have configured on the mesh, set here
                // so the regeneration below has something to lose.
                mesh->userData["physics"] = std::string("enabled=1;type=static;shape=trimesh");
                if (auto* standard = mesh->materialAs<MeshStandardMaterial>()) {
                    standard->color.setHex(0x3366aa);
                }

                const auto geometry = mesh->geometry();
                const auto material = mesh->material();
                bool disposed = false;
                LambdaEventListener watcher{[&disposed](Event&) { disposed = true; }};
                geometry->addEventListener("dispose", watcher);

                auto* dragged = SplineConfig::controlPointNodes(*spline).front();
                dragged->position.y += 2.f;
                step();
                spline = splineNow(splineUuid);

                check(spline && SplineConfig::derivedMesh(*spline) == mesh &&
                              mesh->uuid == derivedUuid,
                      "dragging a point regenerates through the SAME mesh node");
                check(mesh->geometry() != geometry, "swapping in a new geometry");
                check(disposed, "and disposing the one it orphaned");
                check(mesh->material() == material, "the material it carries is untouched");
                check(mesh->materialAs<MeshStandardMaterial>() &&
                              mesh->materialAs<MeshStandardMaterial>()->color.getHex() == 0x3366aa,
                      "down to what was edited on it");
                check(mesh->userData.contains("physics"),
                      "and the physics a user attached survives the rebuild");

                geometry->removeEventListener("dispose", watcher);
                dragged->position.y -= 2.f;
                step();
            }

            // A radius change rebuilds the geometry on the SAME node, which is
            // what keeps a material and a physics setup attached through an
            // edit.
            if (mesh) {
                const auto geometry = mesh->geometry();
                config.radius = 0.75f;
                setConfig(config, "Spline Mesh");
                step();
                spline = splineNow(splineUuid);

                check(spline && SplineConfig::derivedMesh(*spline) == mesh,
                      "a radius change keeps the mesh node");
                check(mesh->geometry() != geometry, "and rebuilds its geometry");
                check(std::abs(tubeRadius(*mesh, *spline) - 0.75f) < 0.02f,
                      "at the radius the config asks for");
            }

            // mesh=none removes it — and undoing that config edit brings it
            // back on the next sync, without an undo entry of its own.
            config.mesh = SplineConfig::MeshKind::None;
            setConfig(config, "Spline Mesh");
            step();
            spline = splineNow(splineUuid);
            check(spline && derivedCount(*spline) == 0, "mesh=none removes the derived child");
            check(spline && SplineConfig::controlPoints(*spline).size() == pointsBefore,
                  "taking no control point with it");

            commands_.undo();
            step();
            spline = splineNow(splineUuid);
            check(spline && derivedCount(*spline) == 1,
                  "undoing the config edit brings the mesh back on the next sync");

            // As a NEW node, though: mesh=none destroys the old one, and the
            // sync re-derives rather than restores. Whatever was configured on
            // the removed mesh is gone with it — undo covers the config edit,
            // which is all it ever claimed to. Re-applied here because the
            // round trip below is about surviving REGENERATION, not deletion.
            if (auto* back = spline ? SplineConfig::derivedMesh(*spline) : nullptr) {
                check(!derivedUuid.empty() && back->uuid != derivedUuid,
                      "though as a fresh node - removal destroys, and undo re-derives");
                derivedUuid = back->uuid;
                back->userData["physics"] = std::string("enabled=1;type=static;shape=trimesh");
                if (auto* standard = back->materialAs<MeshStandardMaterial>()) {
                    standard->color.setHex(0x3366aa);
                }
            }
        }

        // Insert and delete with a derived child present: the point index and
        // the child index are different numbers now, and this is where mixing
        // them up shows.
        if (spline) {
            const auto before = SplineConfig::controlPoints(*spline);
            addSplinePoint(*spline, 1, "Insert Spline Point");
            step();
            spline = splineNow(splineUuid);
            const auto after = SplineConfig::controlPoints(*spline);
            check(after.size() == before.size() + 1,
                  "Insert Before still adds one control point with a mesh in the way");
            Vector3 midpoint;
            if (before.size() > 1) midpoint.copy(before[0]).add(before[1]).multiplyScalar(0.5f);
            check(after.size() > 2 && before.size() > 1 &&
                          after[1].distanceTo(midpoint) < 1e-4f &&
                          after[0].distanceTo(before[0]) < 1e-4f &&
                          after[2].distanceTo(before[1]) < 1e-4f,
                  "at the point index asked for, not the child index");
            check(spline && derivedCount(*spline) == 1, "and the mesh is still the only derived child");

            addSplinePoint(*spline, AddObjectCommand::atEnd, "Add Spline Point");
            step();
            spline = splineNow(splineUuid);
            check(spline && !spline->children.empty() &&
                          SplineConfig::isDerived(*spline->children.back()),
                  "appending a point puts it before the mesh, which stays last");

            // The ordinary delete on "the last point" — the trap being that
            // the last CHILD is the mesh.
            const auto points = SplineConfig::controlPointNodes(*spline);
            if (!points.empty()) {
                selectObject(points.back());
                deleteSelected();
                step();
                spline = splineNow(splineUuid);
                check(spline && SplineConfig::controlPoints(*spline).size() == points.size() - 1,
                      "Del on the last control point takes the point");
                check(spline && derivedCount(*spline) == 1, "and never the generated mesh");
                commands_.undo();
                step();
                spline = splineNow(splineUuid);
            }
            commands_.undo();// the append
            step();
            commands_.undo();// the insert
            step();
            spline = splineNow(splineUuid);
            check(spline && SplineConfig::controlPoints(*spline).size() == before.size(),
                  "and both edits undo back to the point count they started from");
        }

        // Save and reload: the config and every control point have to survive
        // the document round trip, since neither is anything but userData and
        // ordinary child transforms.
        const auto splinePath = std::filesystem::temp_directory_path() /
                                "threepp-editor-selftest-spline.json";
        SplineConfig authored;
        authored.type = SplineConfig::Type::CatmullRom;
        authored.closed = true;
        authored.tension = 0.25f;
        authored.samples = 12;
        // With a generated mesh, so the round trip proves the document carries
        // the geometry too — and that reloading ADOPTS it rather than adding a
        // second one beside it.
        authored.mesh = SplineConfig::MeshKind::Tube;
        authored.radius = 0.45f;
        authored.radialSegments = 10;
        std::vector<Vector3> authoredPoints;

        if (spline) {
            authored.write(*spline);
            // A moved point, so the round trip is proving positions and not
            // just the factory's defaults twice.
            SplineConfig::controlPointNodes(*spline).front()->position.y += 1.5f;
            authoredPoints = SplineConfig::controlPoints(*spline);
            saveSceneAs(splinePath);
            openScene(splinePath);
            step();

            auto* reloaded = splineNow(splineUuid);
            check(reloaded != nullptr, "the spline survives save and reload");
            check(reloaded && SplineConfig::read(*reloaded) == authored,
                  "its config round-trips through the document");
            check(reloaded && SplineConfig::controlPoints(*reloaded).size() == authoredPoints.size(),
                  "and so does the control point count");
            bool positionsMatch = reloaded != nullptr;
            if (reloaded) {
                const auto points = SplineConfig::controlPoints(*reloaded);
                for (std::size_t i = 0; i < points.size() && i < authoredPoints.size(); ++i) {
                    if (points[i].distanceTo(authoredPoints[i]) > 1e-4f) positionsMatch = false;
                }
            }
            check(positionsMatch, "with every control point where it was left");
            check(splineOverlays_.size() == 1, "and a curve overlay rebuilt on the new graph");

            // The document already contains the generated mesh, so the first
            // sync has to ADOPT it. Adding one of its own would leave two.
            check(reloaded && derivedCount(*reloaded) == 1,
                  "the reloaded document has exactly one derived child, adopted not duplicated");
            auto* reloadedMesh = reloaded ? SplineConfig::derivedMesh(*reloaded) : nullptr;
            check(reloadedMesh && reloadedMesh->uuid == derivedUuid,
                  "the same mesh node, by uuid, as the one that was saved");
            check(reloadedMesh && reloadedMesh->userData.contains("physics"),
                  "with the physics config a user put on it");
            check(reloadedMesh && reloadedMesh->materialAs<MeshStandardMaterial>() &&
                          reloadedMesh->materialAs<MeshStandardMaterial>()->color.getHex() == 0x3366aa,
                  "and the material they edited, both surviving the regeneration");
            bool sized = false;
            if (auto* asMesh = reloadedMesh ? reloadedMesh->as<Mesh>() : nullptr) {
                if (auto* live = splineNow(splineUuid)) {
                    sized = std::abs(tubeRadius(*asMesh, *live) - authored.radius) < 0.02f;
                }
            }
            check(sized, "rebuilt to the radius the reloaded config asks for");
        }

#ifdef THREEPP_EDITOR_WITH_PYTHON
        // The documented FollowSpline script: spline_from_object hands it a
        // SplinePath honoring the authored config — no child filtering, no
        // userData parsing in the script. If this stops working, doc/editor.md
        // is handing users a script that does not run. One deviation from the
        // doc: u advances a fixed 0.03 per update instead of speed * dt, so
        // after step(30) it sits at a deterministic u ~= 0.9 — deep in the
        // closing span that exists only when the authored closed=1 reaches the
        // script.
        if (auto* follower = document_.scene().getObjectByName("Box")) {

            const auto followerUuid = follower->uuid;
            setInlineScript(*follower,
                            "import threepp\n"
                            "\n"
                            "\n"
                            "class FollowSpline:\n"
                            "    spline_name = \"Spline\"\n"
                            "\n"
                            "    def start(self, obj):\n"
                            "        self.obj = obj\n"
                            "        self.u = 0.0\n"
                            "        self.path = threepp.editor.spline_from_object(\n"
                            "            threepp.editor.scene().get_object_by_name(self.spline_name))\n"
                            "\n"
                            "    def update(self, dt):\n"
                            "        if self.path is None:\n"
                            "            return\n"
                            "        self.u = (self.u + 0.03) % 1.0\n"
                            "        self.obj.position.copy(self.path.get_point_at(self.u))\n",
                            "Follow Spline");
            step();

            Vector3 rest;
            rest.copy(follower->position);

            startPlay();
            step(30);
            auto* driven = findByUuid(document_.scene(), followerUuid);
            check(driven && driven->position.distanceTo(rest) > 1e-3f,
                  "the FollowSpline script drives the object along the curve");
            // On the AUTHORED curve, not the defaults: the document says
            // closed=1, catmullrom, tension=0.25, and spline_from_object is
            // how that reaches the script. At u ~= 0.9 the follower is inside
            // the closing span, so the same parameters with closed=0 must NOT
            // contain the position — if they do, the config never arrived.
            bool onAuthored = false;
            bool offOpen = true;
            if (driven) {
                if (auto* live = splineNow(splineUuid)) {
                    live->updateMatrixWorld();
                    const auto config = SplineConfig::read(*live).value_or(SplineConfig{});
                    if (auto curve = config.curve(*live)) {
                        for (auto& point : curve->getPoints(512)) {
                            point.applyMatrix4(*live->matrixWorld);
                            if (point.distanceTo(driven->position) < 0.1f) onAuthored = true;
                        }
                    }
                    SplineConfig open = config;
                    open.closed = false;
                    if (auto curve = open.curve(*live)) {
                        for (auto& point : curve->getPoints(512)) {
                            point.applyMatrix4(*live->matrixWorld);
                            if (point.distanceTo(driven->position) < 0.1f) offOpen = false;
                        }
                    }
                }
            }
            check(onAuthored, "and it lands on the authored closed curve");
            check(offOpen, "in the closing span an open curve does not have");
            // The generated mesh is a child too, and at the spline's origin.
            // Counting it as a control point would bend the curve the script
            // builds away from the one the editor draws, so the two checks
            // above are also the test that the documented list comprehension
            // skips it.
            check(splineNow(splineUuid) && derivedCount(*splineNow(splineUuid)) == 1,
                  "with the generated mesh present, which the script has to skip");

            stopPlay();
            step();
            auto* restored = findByUuid(document_.scene(), followerUuid);
            check(restored && restored->position.distanceTo(rest) < 1e-4f,
                  "stop restores the pose FollowSpline changed");
        }
#endif

        // The scene-replace hazard, with a control point selected: the overlay
        // and its markers both hold raw pointers into the graph about to be
        // freed. Open first (replace with an equivalent graph), then New
        // (replace with a different one).
        if (auto* live = splineNow(splineUuid)) {
            selectObject(live);
            step();
            check(splineOverlays_.size() == 1, "the selected spline still has its curve");
            const auto points = SplineConfig::controlPointNodes(*live);
            if (!points.empty()) selectObject(points.front());
            step();
            check(selection_.get() != nullptr, "a control point is selectable");
            // The generated mesh is picked and inspected like any other mesh —
            // that is the whole point of it being a document node.
            if (auto* generated = SplineConfig::derivedMesh(*live)) {
                selectObject(generated);
                step();
                check(selection_.get() == generated, "and so is the generated mesh");
                check(resolveSelectable(generated) == generated,
                      "a click on it drills down to it while its spline is the selection");
            }
        }

        openScene(splinePath);
        step(2);
        check(splineOverlays_.size() == 1, "reopening rebuilds the overlay on the new graph");

        if (auto* live = splineNow(splineUuid)) {
            if (!live->children.empty()) selectObject(live->children.front());
        }
        step();
        newScene();
        step(2);
        check(splineOverlays_.empty(), "a scene replace drops every spline overlay");

        std::error_code ec;
        std::filesystem::remove(splinePath, ec);
    }

    // Conveyors. Same authoring model as splines — a Group whose children are
    // its waypoints — plus generated content (belt, frame, rollers, cleats)
    // and, under Play, kinematic belt physics that CONVEYS bodies.
    if (section("conveyors")) {
        const auto conveyorNow = [&](const std::string& uuid) {
            return findByUuid(document_.scene(), uuid);
        };

        auto created = ObjectFactory::createConveyor(document_.scene());
        const auto conveyorUuid = created->uuid;
        addObject(created, document_.scene(), "Add Conveyor");
        created.reset();// the command owns it now
        step();

        auto* conveyor = conveyorNow(conveyorUuid);
        check(conveyor && ConveyorConfig::isConveyor(*conveyor), "the factory creates a conveyor");
        check(conveyor && ConveyorConfig::waypointNodes(*conveyor).size() == 3,
              "with waypoints as its children");
        check(conveyorOverlays_.size() == 1, "the conveyor gets a path overlay");

        auto* derived = conveyor ? ConveyorConfig::derivedGroup(*conveyor) : nullptr;
        check(derived != nullptr, "and generates its parts group");

        // Parts by role: the generated content is the conveyor's LOOK, all of
        // it first-party procedural geometry.
        const auto roleCount = [&](const char* role) {
            std::size_t n = 0;
            auto* live = conveyorNow(conveyorUuid);
            auto* group = live ? ConveyorConfig::derivedGroup(*live) : nullptr;
            if (group) {
                group->traverse([&](Object3D& o) {
                    if (ConveyorConfig::roleOf(o) == role) ++n;
                });
            }
            return n;
        };
        check(roleCount("belt") == 1, "a straight run generates one belt ribbon");
        check(roleCount("drum") == 2, "an end drum (pulley) at each open end");
        check(roleCount("frame") >= 4, "side rails and legs make up the frame");

        // Per-segment surfaces live on the WAYPOINT nodes, so flipping one
        // regenerates the parts without touching the conveyor's own config.
        if (conveyor) {
            auto nodes = ConveyorConfig::waypointNodes(*conveyor);
            ConveyorWaypointConfig wp;
            wp.segKind = conveyor::SegKind::Rollers;
            wp.write(*nodes.front());
            step();
            check(roleCount("roller") >= 3, "a rollers segment grows a roller bed");
            check(roleCount("belt") == 1, "while the other segment keeps its ribbon");

            wp.segKind = conveyor::SegKind::Cleats;
            wp.write(*nodes.front());
            step();
            check(roleCount("cleat") >= 1, "a cleats segment grows preview flight bars");
            check(roleCount("roller") == 0, "and the rollers it replaced are gone");

            ConveyorWaypointConfig::erase(*nodes.front());
            step();
            check(roleCount("cleat") == 0 && roleCount("belt") == 1,
                  "clearing the waypoint entry returns the segment to a flat belt");
        }

        // A separator is a wall, not a belt: no frame, no drums, one wall.
        if (conveyor) {
            auto config = ConveyorConfig::read(*conveyor).value_or(ConveyorConfig{});
            auto separator = config;
            separator.separator = true;
            separator.write(*conveyor);
            step();
            check(roleCount("wall") == 1 && roleCount("belt") == 0 && roleCount("drum") == 0,
                  "a separator generates one wall and nothing else");
            config.write(*conveyor);
            step();
            check(roleCount("belt") == 1, "and switching back restores the belt");
        }

        // Waypoint editing is the ordinary machinery: add is undoable, delete
        // is the ordinary delete.
        if (conveyor) {
            addConveyorPoint(*conveyor, AddObjectCommand::atEnd, "Add Waypoint");
            step();
            conveyor = conveyorNow(conveyorUuid);
            check(conveyor && ConveyorConfig::waypointNodes(*conveyor).size() == 4,
                  "Add Waypoint appends a waypoint");
            commands_.undo();
            step();
            conveyor = conveyorNow(conveyorUuid);
            check(conveyor && ConveyorConfig::waypointNodes(*conveyor).size() == 3,
                  "and the append is undoable");
        }

        // Attached walls: a child of the conveyor, NOT a waypoint — adding one
        // must not bend the path, and its ribbon appears among the parts.
        if (conveyor) {
            addConveyorWall(*conveyor, "Add Wall");
            step();
            conveyor = conveyorNow(conveyorUuid);
            check(conveyor && ConveyorConfig::wallNodes(*conveyor).size() == 1,
                  "Add Wall attaches a wall to the conveyor");
            check(conveyor && ConveyorConfig::waypointNodes(*conveyor).size() == 3,
                  "without becoming a waypoint");
            check(roleCount("wall") == 1, "and its ribbon joins the generated parts");
            auto* selectedWall = selection_.get();
            check(selectedWall && ConveyorWallConfig::isWall(*selectedWall),
                  "the new wall is selected, ready to drag");

            // A dragged wall point regenerates the ribbon like any other edit.
            if (conveyor) {
                auto walls = ConveyorConfig::wallNodes(*conveyor);
                const auto points = ConveyorWallConfig::pointNodes(*walls.front());
                if (!points.empty()) {
                    points.front()->position.z += 0.4f;
                    step();
                    check(roleCount("wall") == 1, "a dragged wall point keeps one ribbon");
                }
            }

            // The grow verb: Insert After marches the wall along, one point at
            // a time, each landing selected and past the previous end.
            if (conveyor && !ConveyorConfig::wallNodes(*conveyor).empty()) {
                auto* wallNode = ConveyorConfig::wallNodes(*conveyor).front();
                const auto endBefore =
                        ConveyorWallConfig::pointNodes(*wallNode).back()->position;
                addConveyorWallPoint(*wallNode, AddObjectCommand::atEnd, "Insert Wall Point");
                step();
                const auto points = ConveyorWallConfig::pointNodes(*wallNode);
                check(points.size() == 3, "Insert After grows the wall by one point");
                check(selection_.get() == points.back(),
                      "which lands selected, ready to drag");
                check(points.back()->position.distanceTo(endBefore) > 0.05f,
                      "past the previous end, so the wall visibly extends");
                commands_.undo();
                step();
                check(ConveyorWallConfig::pointNodes(*wallNode).size() == 2,
                      "and the growth is undoable");
            }

            commands_.undo();
            step();
            conveyor = conveyorNow(conveyorUuid);
            check(conveyor && ConveyorConfig::wallNodes(*conveyor).empty() &&
                          roleCount("wall") == 0,
                  "undo removes the wall and its ribbon together");
            selectObject(nullptr);
            step();
        }

#ifdef THREEPP_EDITOR_WITH_PHYSX
        // Play: the belt CONVEYS. A dynamic box dropped onto the moving surface
        // must travel along it — that is the feature, everything else is décor.
        // The first segment is a roller bed, so the same run also proves the
        // rollers are REAL colliders (a rollers span builds no drag box — the
        // capsules carry the cargo) and that the rollers→belt handoff works.
        {
            if (auto* live = conveyorNow(conveyorUuid)) {
                auto nodes = ConveyorConfig::waypointNodes(*live);
                ConveyorWaypointConfig rollers;
                rollers.segKind = conveyor::SegKind::Rollers;
                rollers.write(*nodes.front());
                step();
            }

            // An attached wall rides along too (its colliders are counted
            // below). The default is a PASSIVE edge guide, so the centre-line
            // convey measurement below is untouched by it — that passivity is
            // itself part of the contract (the physics test pins it harder).
            if (auto* live = conveyorNow(conveyorUuid)) {
                addConveyorWall(*live, "Add Wall");
                selectObject(nullptr);
                step();
            }

            auto boxMesh = ObjectFactory::createPrimitive(Primitive::Box,
                                                          document_.scene());
            boxMesh->name = "ConveyorCargo";
            // Over the upstream end of the default belt (surface y=0.75),
            // dropped a hair so it lands rather than spawns intersecting.
            boxMesh->position.set(-1.2f, 1.32f, 0.f);
            boxMesh->scale.set(0.4f, 0.4f, 0.4f);
            PhysicsConfig cargo;
            cargo.enabled = true;
            cargo.write(*boxMesh);
            const auto cargoUuid = boxMesh->uuid;
            addObject(boxMesh, document_.scene(), "Add Cargo");
            boxMesh.reset();
            step();

            startPlay();
            step(2);
            check(conveyorSession_ && conveyorSession_->conveyorCount() == 1,
                  "the play session picks the conveyor up");
            check(conveyorSession_ && conveyorSession_->beltCount() >= 2,
                  "and builds belt colliders for it");
            check(conveyorSession_ && conveyorSession_->rollerCount() >= 3,
                  "and real roller colliders for the roller bed");
            check(conveyorSession_ && conveyorSession_->wallCount() >= 1,
                  "and wall colliders for the attached diverter");

            auto* cargo1 = findByUuid(document_.scene(), cargoUuid);
            float startX = cargo1 ? cargo1->position.x : 0.f;
            // Bounded by PLAY TIME, not frames: the selftest runs unthrottled,
            // so a frame is however little wall clock the machine needs, and
            // the belt (0.6 m/s) moves cargo by SIMULATED seconds. Half a
            // metre needs ~1 s of belt time; 8 s is comfortable on any box.
            bool conveyed = false, onBelt = true;
            for (int i = 0; i < 200000 && !conveyed && play_.elapsed() < 8.f; ++i) {
                step();
                cargo1 = findByUuid(document_.scene(), cargoUuid);
                if (!cargo1) break;
                if (cargo1->position.x - startX > 0.5f) conveyed = true;
                if (cargo1->position.y < 0.4f) {
                    onBelt = false;// fell through / off the side
                    break;
                }
            }
            check(conveyed, "the belt conveys a rigid box along its travel direction");
            check(onBelt, "which rides ON the belt the whole way");
            check(cargo1 && std::abs(cargo1->position.z) < 0.4f,
                  "without drifting off the side");

            stopPlay();
            step();
            auto* restored = findByUuid(document_.scene(), cargoUuid);
            check(restored && std::abs(restored->position.x - (-1.2f)) < 1e-3f,
                  "and Stop puts the cargo back where it was authored");

            // Back to a bare flat belt for the rounds that follow (the soft
            // cargo lands on this conveyor too): the cargo and the diverter
            // leave through undo, the roller segment the way it came.
            commands_.undo();// the cargo
            commands_.undo();// the wall
            if (auto* live = conveyorNow(conveyorUuid)) {
                auto nodes = ConveyorConfig::waypointNodes(*live);
                ConveyorWaypointConfig::erase(*nodes.front());
                step();
            }
            check(conveyorNow(conveyorUuid) &&
                          ConveyorConfig::wallNodes(*conveyorNow(conveyorUuid)).empty(),
                  "undo clears the diverter after the run");
        }

        // A rounded corner is ONE rotating bend body, tangent by construction,
        // and it really turns cargo. A right-angle corner, radius 2.
        {
            auto arcConveyor = ObjectFactory::createConveyor(document_.scene());
            const auto arcUuid = arcConveyor->uuid;
            addObject(arcConveyor, document_.scene(), "Add Bend Conveyor");
            arcConveyor.reset();
            step();

            // The fillet the corner resolves to, read from the same helper the
            // preview and the physics build from — the test asserts against
            // the DERIVED centre, not a hand-computed one.
            conveyor::CornerFillet fillet;
            auto* arc = conveyorNow(arcUuid);
            if (arc) {
                auto nodes = ConveyorConfig::waypointNodes(*arc);
                nodes[0]->position.set(-3.f, 0.75f, 0.f);
                nodes[1]->position.set(0.f, 0.75f, 0.f);
                nodes[2]->position.set(0.f, 0.75f, 3.f);
                ConveyorWaypointConfig wp;
                wp.cornerRadius = 2.f;
                wp.write(*nodes[1]);
                step();

                const auto spec = ConveyorConfig::read(*arc)
                                          .value_or(ConveyorConfig{})
                                          .spec(*arc);
                fillet = conveyor::cornerFillet(spec.waypoints, 1);
                check(fillet.valid, "the corner resolves to a tangent fillet");
                check(std::abs(fillet.radius - 2.f) < 1e-3f,
                      "at the authored radius, which these segments allow");
                // Tangency in one number: the spokes to the tangent points are
                // perpendicular to their segments.
                const float inDot = (fillet.t1.x - fillet.centre.x) * 1.f;// incoming dir +x
                const float outDot = (fillet.t2.z - fillet.centre.z) * 1.f;// outgoing dir +z
                check(std::abs(inDot) < 1e-3f && std::abs(outDot) < 1e-3f,
                      "with its spokes perpendicular to both segments");

                // The radius handle, driven through the same core the viewport
                // drag runs — no mouse, real commands. The conveyor sits at the
                // origin, so local and world agree.
                selectObject(nodes[1]);
                step();
                check(conveyorRadiusHandle_ && conveyorRadiusHandle_->visible,
                      "selecting a corner shows the radius handle");
                const float midAngle = fillet.a0 + fillet.sweep * 0.5f;
                const Vector3 expectedMid(fillet.centre.x + fillet.radius * std::cos(midAngle),
                                          (fillet.t1.y + fillet.t2.y) * 0.5f,
                                          fillet.centre.z + fillet.radius * std::sin(midAngle));
                check(conveyorRadiusHandle_ &&
                              conveyorRadiusHandle_->position.distanceTo(expectedMid) < 1e-2f,
                      "sitting on the arc midpoint");

                if (conveyorRadiusHandle_) {
                    const auto handle = conveyor::cornerHandle(spec.waypoints, 1);
                    const auto undos = commands_.undoCount();
                    // Grab the ball with a vertical ray (the bisector is
                    // horizontal, so the ray reads exactly the ball's
                    // distance), then drop the same ray over the point a
                    // radius of 2.5 would put the ball at.
                    Vector3 over = conveyorRadiusHandle_->position;
                    over.y += 5.f;
                    beginConveyorRadiusDrag(over, Vector3(0.f, -1.f, 0.f));
                    Vector3 target = handle.origin;
                    target.addScaledVector(handle.direction, 2.5f * handle.secMinusOne);
                    target.y += 5.f;
                    applyConveyorRadiusDrag(target, Vector3(0.f, -1.f, 0.f));
                    endConveyorRadiusDrag();
                    step();

                    auto liveNodes = ConveyorConfig::waypointNodes(*conveyorNow(arcUuid));
                    const float dragged = ConveyorWaypointConfig::read(*liveNodes[1]).cornerRadius;
                    check(std::abs(dragged - 2.5f) < 0.05f,
                          "dragging the handle outward widens the radius");
                    check(commands_.undoCount() == undos + 1,
                          "and the whole drag is one undo entry");
                    commands_.undo();
                    step();
                    liveNodes = ConveyorConfig::waypointNodes(*conveyorNow(arcUuid));
                    check(std::abs(ConveyorWaypointConfig::read(*liveNodes[1]).cornerRadius - 2.f) < 1e-4f,
                          "which undoes back to the authored radius");
                }
                selectObject(nullptr);
                step();
            }

            auto cargo = ObjectFactory::createPrimitive(Primitive::Box,
                                                        document_.scene());
            cargo->name = "ArcCargo";
            // On the arc's midpoint, halfway around the sweep.
            const float midAngle = fillet.a0 + fillet.sweep * 0.5f;
            cargo->position.set(fillet.centre.x + fillet.radius * std::cos(midAngle), 1.32f,
                                fillet.centre.z + fillet.radius * std::sin(midAngle));
            cargo->scale.set(0.4f, 0.4f, 0.4f);
            PhysicsConfig cargoConfig;
            cargoConfig.enabled = true;
            cargoConfig.write(*cargo);
            const auto cargoUuid = cargo->uuid;
            addObject(cargo, document_.scene(), "Add Arc Cargo");
            cargo.reset();
            step();

            startPlay();
            step(2);
            check(conveyorSession_ && conveyorSession_->conveyorCount() == 2,
                  "both conveyors play at once");

            const Vector3 centre(fillet.centre.x, 0.f, fillet.centre.z);
            auto* riding = findByUuid(document_.scene(), cargoUuid);
            const float angle0 = riding ? std::atan2(riding->position.z - centre.z,
                                                     riding->position.x - centre.x)
                                        : 0.f;
            const float direction = fillet.sweep >= 0.f ? 1.f : -1.f;
            bool turned = false, onArc = true;
            for (int i = 0; i < 200000 && !turned && play_.elapsed() < 8.f; ++i) {
                step();
                riding = findByUuid(document_.scene(), cargoUuid);
                if (!riding) break;
                const float angle = std::atan2(riding->position.z - centre.z,
                                               riding->position.x - centre.x);
                if (direction * (angle - angle0) > 0.15f) turned = true;// toward t2
                const float radius = std::hypot(riding->position.x - centre.x,
                                                riding->position.z - centre.z);
                if (riding->position.y < 0.4f || radius < fillet.radius - 0.7f ||
                    radius > fillet.radius + 0.7f) {
                    onArc = false;
                    break;
                }
            }
            check(turned, "the bend carries cargo around the corner's arc");
            check(onArc, "keeping it on the annulus at belt height");

            stopPlay();
            step();
        }

        // Soft bodies convey too — the reason the belts are kinematic dynamics.
        // Only asserted when the machine could actually cook one (CUDA); the
        // rigid checks above carry the feature elsewhere.
        {
            auto ball = ObjectFactory::createPrimitive(Primitive::Sphere,
                                                       document_.scene());
            ball->name = "SoftCargo";
            ball->position.set(-1.2f, 1.25f, 0.f);
            ball->scale.set(0.6f, 0.6f, 0.6f);
            PhysicsConfig soft;
            soft.enabled = true;
            soft.body = PhysicsConfig::Body::Soft;
            soft.write(*ball);
            const auto ballUuid = ball->uuid;
            addObject(ball, document_.scene(), "Add Soft Cargo");
            ball.reset();
            step();

            startPlay();
            step(2);
            if (physics_ && physics_->softBodyCount() == 1) {
                auto* riding = findByUuid(document_.scene(), ballUuid);
                float startX = riding ? riding->position.x : 0.f;
                bool conveyed = false;
                for (int i = 0; i < 200000 && !conveyed && play_.elapsed() < 10.f; ++i) {
                    step();
                    riding = findByUuid(document_.scene(), ballUuid);
                    if (!riding) break;
                    // A soft body's node stays put; the SIMULATION rewrites its
                    // geometry in world space. Read the surface, not the node.
                    const auto geometry = riding->geometry();
                    const auto* position = geometry ? geometry->getAttribute<float>("position")
                                                    : nullptr;
                    if (position && position->count() > 0) {
                        float meanX = 0.f;
                        for (std::size_t v = 0; v < position->count(); ++v) {
                            meanX += position->getX(v);
                        }
                        meanX /= static_cast<float>(position->count());
                        if (i == 0) startX = meanX;
                        if (meanX - startX > 0.4f) conveyed = true;
                    }
                }
                check(conveyed, "a soft body rides the belt (GPU dynamics)");
            } else {
                // No CUDA: the physics session already logged why. Not a
                // failure of the conveyor.
                check(true, "soft cargo skipped - no GPU soft body this run");
            }
            stopPlay();
            step();
        }
#endif

        // Save and reload: config, per-waypoint entries and the generated group
        // are all plain document content.
        const auto conveyorPath = std::filesystem::temp_directory_path() /
                                  "threepp-editor-selftest-conveyor.json";
        if (auto* live = conveyorNow(conveyorUuid)) {
            ConveyorConfig authored;
            authored.width = 0.8f;
            authored.speed = 1.1f;
            authored.smooth = false;
            authored.cleatSpacing = 0.5f;
            authored.write(*live);
            auto nodes = ConveyorConfig::waypointNodes(*live);
            ConveyorWaypointConfig wp;
            wp.segKind = conveyor::SegKind::Cleats;
            wp.write(*nodes[1]);
            const auto derivedUuid = ConveyorConfig::derivedGroup(*live)
                                             ? ConveyorConfig::derivedGroup(*live)->uuid
                                             : std::string{};
            step();

            saveSceneAs(conveyorPath);
            openScene(conveyorPath);
            step();

            auto* reloaded = conveyorNow(conveyorUuid);
            check(reloaded != nullptr, "the conveyor survives save and reload");
            check(reloaded && ConveyorConfig::read(*reloaded) == authored,
                  "its config round-trips through the document");
            bool wpKept = false;
            if (reloaded) {
                auto reloadedNodes = ConveyorConfig::waypointNodes(*reloaded);
                wpKept = reloadedNodes.size() == 3 &&
                         ConveyorWaypointConfig::read(*reloadedNodes[1]).segKind ==
                                 conveyor::SegKind::Cleats;
            }
            check(wpKept, "and so does the per-waypoint surface choice");
            check(reloaded && ConveyorConfig::derivedGroup(*reloaded) &&
                          ConveyorConfig::derivedGroup(*reloaded)->uuid == derivedUuid,
                  "the parts group is adopted by uuid, not duplicated");
            check(reloaded && roleCount("cleat") >= 1,
                  "and regenerates the cleat bars the reloaded config asks for");
        }

        newScene();
        step(2);
        check(conveyorOverlays_.empty(), "a scene replace drops every conveyor overlay");

        std::error_code conveyorEc;
        std::filesystem::remove(conveyorPath, conveyorEc);
    }

    // Particle fields, EDIT MODE. The authored node is a plain Group carrying
    // one userData entry; the ParticleField itself is never a document node,
    // which is half of what is asserted here (the other half is that the
    // preview follows the config, and only on the backend that can draw it).
    if (section("particle-fields")) {
        const auto particlesNow = [&](const std::string& uuid) {
            return findByUuid(document_.scene(), uuid);
        };

        auto created = ObjectFactory::createParticleField(document_.scene());
        const auto particlesUuid = created->uuid;
        addObject(created, document_.scene(), "Add Particle Field");
        created.reset();// the command owns it now
        step();

        auto* particles = particlesNow(particlesUuid);
        check(particles && ParticleFieldConfig::isParticleField(*particles),
              "the factory creates a particle field");
        check(particles && ParticleFieldConfig::read(*particles) == ParticleFieldConfig::snow(),
              "carrying the snow preset, read back intact");
        // The node IS the emitter frame, and a field at the floor pours its
        // snow through it.
        check(particles && particles->position.y > 1.f,
              "lifted to the top of one lifetime of fall");

        // The Vulkan gate, asserted in BOTH directions: a GL session must build
        // no field at all (and must not crash for not having one), a Vulkan
        // session must build exactly one.
        const bool previewable = particlePreviewAvailable();
        check(particlePreviews_.size() == (previewable ? 1u : 0u),
              previewable ? "a preview field is built on the Vulkan backend"
                          : "no preview field is built on the OpenGL backend");
        check(particleDensityCount() == 1,
              "and its density volume is counted against the scene-wide budget");

        // Selecting it shows the spawn slab — the only picture there is on GL,
        // and the marker icon is the selection path since a field is unpickable.
        selectObject(particlesNow(particlesUuid));
        step();
        check(particleHelper_ && particleHelper_->visible,
              "selecting it draws the spawn slab and the flight arrow");
        selectObject(nullptr);
        step();

        // The preset buttons commit through exactly this path, so applying one
        // here is the button. Rain is structural against snow (different
        // capacity, radius and proxy), which is what makes the rebuild leg run.
        std::string structuralBefore;
        if (!particlePreviews_.empty()) {
            structuralBefore = particlePreviews_.begin()->second.structuralKey;
        }
        if (auto* live = particlesNow(particlesUuid)) {
            auto* node = live;
            const auto before = ParticleFieldConfig::read(*live).value_or(ParticleFieldConfig{});
            commands_.execute(makeProperty<ParticleFieldConfig>(
                    "Particles Rain", "particles:" + live->uuid,
                    [node](const ParticleFieldConfig& value) { value.write(*node); },
                    before, ParticleFieldConfig::rain()));
            step();
        }
        check(particlesNow(particlesUuid) &&
                      ParticleFieldConfig::read(*particlesNow(particlesUuid)) ==
                              ParticleFieldConfig::rain(),
              "a preset button replaces every field in one undo step");
        if (previewable && !particlePreviews_.empty()) {
            check(particlePreviews_.begin()->second.structuralKey != structuralBefore,
                  "and a structural change rebuilds the preview field");
        } else {
            check(particlePreviews_.empty(), "with still no preview field on OpenGL");
        }

        commands_.undo();
        step();
        check(particlesNow(particlesUuid) &&
                      ParticleFieldConfig::read(*particlesNow(particlesUuid)) ==
                              ParticleFieldConfig::snow(),
              "which undoes back to the preset it replaced");

        // A mutable edit must NOT rebuild: that is the whole point of the
        // two-tier split (a rebuild is a vkDeviceWaitIdle and a cleared TAA
        // history, and a slider drag must not pay it).
        if (previewable && !particlePreviews_.empty()) {
            const auto key = particlePreviews_.begin()->second.structuralKey;
            if (auto* live = particlesNow(particlesUuid)) {
                auto config = ParticleFieldConfig::read(*live).value_or(ParticleFieldConfig{});
                config.wind.set(3.f, 0.f, 0.f);
                config.write(*live);
                step();
            }
            check(!particlePreviews_.empty() &&
                          particlePreviews_.begin()->second.structuralKey == key,
                  "a mutable edit is pushed in place, not rebuilt");
        }

        // Undo of the add takes the config and the preview with it.
        const auto undosBefore = commands_.undoCount();
        commands_.undo();
        step();
        check(particlesNow(particlesUuid) == nullptr && commands_.undoCount() < undosBefore,
              "undo of the add removes the particle node");
        check(particlePreviews_.empty() && particleDensityCount() == 0,
              "and the preview entry retires with it");
        commands_.redo();
        step();
        check(particlesNow(particlesUuid) != nullptr, "redo brings it back");

        // PLAY MODE. The session owns its own fields and its own clock: the
        // previews are parked out of its way for the duration, and t starts at
        // 0 every episode, so two plays of one document are the same weather.
        // Driven with stepFixed because the assertion below is about how much
        // of that clock passed.
        {
            const auto fieldsInDocument = [&] {
                std::size_t n = 0;
                document_.scene().traverse([&](Object3D& o) {
                    if (document_.isEditorOnly(o)) return;
                    if (o.type() == "ParticleField") ++n;
                });
                return n;
            };

            startPlay();
            stepFixed(30);

            check(particleSession_ && particleSession_->fieldNodeCount() == 1,
                  "the play session counts the authored node on every backend");
            check(particleSession_ && particleSession_->emitterTime() > 0.4f,
                  "and runs its own deterministic clock from zero");

            if (previewable) {
                const auto* played = particleSession_->fieldFor(particlesUuid);
                check(played != nullptr && fieldsInDocument() == 1,
                      "a session field is built and added to the played scene");
                check(played && played->emitterTime() > 0.4f,
                      "with its emitter advanced by the session's clock");
                // Placement is pushed in from the authored node, which is what
                // makes the node's transform the emitter frame. The field's own
                // matrix IS its world matrix (matrixAutoUpdate is off).
                Vector3 nodeAt, fieldAt;
                if (auto* live = particlesNow(particlesUuid)) live->getWorldPosition(nodeAt);
                if (played) fieldAt.setFromMatrixPosition(*played->matrix);
                check(played && fieldAt.distanceTo(nodeAt) < 1e-4f,
                      "placed on the authored node it was built from");
                check(!particlePreviews_.empty() &&
                              particlePreviews_.begin()->second.field->liveCount() == 0,
                      "and the edit-mode preview is parked while it plays");
            } else {
                check(particleSession_ && particleSession_->liveFieldCount() == 0 &&
                              fieldsInDocument() == 0,
                      "no session field is built on the OpenGL backend");
                check(particlePreviews_.empty() && particlesNow(particlesUuid) != nullptr,
                      "and the scene it declined to draw plays on regardless");
            }

            stopPlay();
            step();
            check(fieldsInDocument() == 0 &&
                          (!particleSession_ || particleSession_->liveFieldCount() == 0),
                  "Stop takes the session's fields with the snapshot restore");
            if (previewable) {
                check(!particlePreviews_.empty() &&
                              particlePreviews_.begin()->second.field->liveCount() > 0,
                      "and the preview un-parks where it left off");
            }
        }

        // Save and reload: what the document carries is one userData string,
        // and NO ParticleField node — the type has no loader case and would
        // export as its zero-area placeholder (ObjectExporter::isUnexportable).
        const auto particlesPath = std::filesystem::temp_directory_path() /
                                   "threepp-editor-selftest-particles.json";
        if (auto* live = particlesNow(particlesUuid)) {
            ParticleFieldConfig authored = ParticleFieldConfig::embers();
            authored.seed = 4242;
            authored.wind.set(0.5f, 0.f, -0.25f);
            authored.write(*live);
            step();

            saveSceneAs(particlesPath);
            openScene(particlesPath);
            step();

            auto* reloaded = particlesNow(particlesUuid);
            check(reloaded != nullptr, "the particle field survives save and reload");
            check(reloaded && ParticleFieldConfig::read(*reloaded) == authored,
                  "its config round-trips through the document");
            std::size_t fieldNodes = 0;
            document_.scene().traverse([&](Object3D& o) {
                if (document_.isEditorOnly(o)) return;
                if (o.type() == "ParticleField") ++fieldNodes;
            });
            check(fieldNodes == 0, "and no ParticleField node leaked into the document");
        }

        newScene();
        step(2);
        check(particlePreviews_.empty(), "a scene replace drops every particle preview");

        std::error_code particlesEc;
        std::filesystem::remove(particlesPath, particlesEc);
    }

    // Flocks. Authoring is a userData entry and the birds are renderer- and
    // PhysX-free, so BOTH halves — the edit-mode round trip and the play-mode
    // birds — run on every build.
    if (section("flocks")) {
        auto created = ObjectFactory::createFlock(document_.scene());
        const auto flockUuid = created->uuid;
        addObject(created, document_.scene(), "Add Flock");
        created.reset();// the command owns it now
        step();

        auto* flock = findByUuid(document_.scene(), flockUuid);
        check(flock && FlockConfig::isFlock(*flock),
              "the factory creates a flock node");
        check(flock && FlockConfig::read(*flock) == FlockConfig{},
              "carrying the default flock config");
        check(flock && flock->position.y > 1.f,
              "lifted to cruising height - home is a loiter volume, not a floor mark");

        selectObject(findByUuid(document_.scene(), flockUuid));
        step();
        check(particleHelper_ && particleHelper_->visible,
              "selecting it draws the territory rings and the wind");
        selectObject(nullptr);
        step();

        // An inspector edit round-trips through the entry, undoably — and the
        // helper follows it (its rebuild key carries the radius).
        if (auto* live = findByUuid(document_.scene(), flockUuid)) {
            auto* node = live;
            const auto before = FlockConfig::read(*live).value_or(FlockConfig{});
            auto after = before;
            after.roamRadius = 12.f;
            after.birdCount = 24;
            commands_.execute(makeProperty<FlockConfig>(
                    "Flock Roam Radius", "flock:" + live->uuid,
                    [node](const FlockConfig& value) { value.write(*node); },
                    before, after));
            step();
            check(FlockConfig::read(*findByUuid(document_.scene(), flockUuid)) == after,
                  "an inspector edit round-trips through the entry");
            commands_.undo();
            step();
            check(FlockConfig::read(*findByUuid(document_.scene(), flockUuid)) == before,
                  "and undoes back to what it replaced");
        }

        const auto liveBirdMeshes = [&] {
            std::size_t n = 0;
            document_.scene().traverse([&](Object3D& o) {
                if (o.type() == "Flock") ++n;
            });
            return n;
        };

        check(liveBirdMeshes() == 0, "no birds exist in edit mode");
        startPlay();
        check(liveBirdMeshes() == 1, "play builds one Flock per authored node");
        stepFixed(240);
        check(liveBirdMeshes() == 1, "and it survives four seconds of flight");
        // The phantom-floor trap. The editor scene carries overlay meshes
        // (gizmo handles, light markers) in the graph the perch bake walks;
        // without the session's editor-only filter, a marker at altitude
        // becomes the highest surface in its column and the whole flock
        // climbs to it (measured: y=430 over a flat template). Home is at
        // 12 and the default cruise band tops out near 31, so 60 is not a
        // tuning number — only a poisoned bake reaches it.
        {
            Flock* liveFlock = nullptr;
            document_.scene().traverse([&](Object3D& o) {
                if (auto* f = dynamic_cast<Flock*>(&o)) liveFlock = f;
            });
            float maxY = -1e9f, minY = 1e9f;
            if (liveFlock) {
                for (int i = 0; i < liveFlock->birdCount(); ++i) {
                    maxY = std::max(maxY, liveFlock->birdPosition(i).y);
                    minY = std::min(minY, liveFlock->birdPosition(i).y);
                }
            }
            check(liveFlock && maxY < 60.f,
                  "the birds hold the authored cruise band, not a helper-mesh ceiling");
            check(liveFlock && minY > -5.f,
                  "and none of them sank through the floor");
        }
        stopPlay();
        step();
        check(liveBirdMeshes() == 0,
              "stop restores a document that never saw the birds");

        // Delete the node so later blocks meet the scene they always did.
        if (auto* live = findByUuid(document_.scene(), flockUuid)) {
            commands_.execute(std::make_unique<RemoveObjectCommand>(*live));
            step();
        }
    }

    // Granular chutes, EDIT MODE. Authoring is PhysX-free — the config is just
    // strings — so this runs on every build; whether grains actually pour is a
    // play-time question the physics blocks answer.
    if (section("granular")) {
        auto created = ObjectFactory::createGranular(document_.scene());
        const auto granularUuid = created->uuid;
        addObject(created, document_.scene(), "Add Granular Particles");
        created.reset();
        step();

        auto* granular = findByUuid(document_.scene(), granularUuid);
        check(granular && GranularConfig::isGranular(*granular),
              "the factory creates a granular chute");
        check(granular && GranularConfig::read(*granular) == GranularConfig{},
              "carrying the default chute config");
        check(granular && granular->position.y > 1.f,
              "standing clear of the floor, so grains do not spawn inside it");

        selectObject(findByUuid(document_.scene(), granularUuid));
        step();
        check(particleHelper_ && particleHelper_->visible,
              "selecting it draws the pour mouth and its direction");
        selectObject(nullptr);
        step();

        if (auto* live = findByUuid(document_.scene(), granularUuid)) {
            auto* node = live;
            const auto before = GranularConfig::read(*live).value_or(GranularConfig{});
            auto after = before;
            after.spacing = 0.04f;
            after.rate = 12000.f;
            after.cohesion = 0.3f;
            commands_.execute(makeProperty<GranularConfig>(
                    "Granular Spacing", "granular:" + live->uuid,
                    [node](const GranularConfig& value) { value.write(*node); },
                    before, after));
            step();
            check(GranularConfig::read(*findByUuid(document_.scene(), granularUuid)) == after,
                  "an inspector edit round-trips through the entry");
            commands_.undo();
            step();
            check(GranularConfig::read(*findByUuid(document_.scene(), granularUuid)) == before,
                  "and undoes back to what it replaced");
        }

#if 0
        // PLAY MODE. Grains are a CUDA-only PhysX feature, so this machine
        // either has a device — in which case a pour has to arrive and be drawn
        // — or it has not, in which case the session must decline with a line
        // and let the rest of the scene play. Both are assertions; "did nothing
        // and said nothing" is the failure.
        {
            // Modest and short: the assertion is that grains flow, not how many
            // a laptop can hold.
            if (auto* live = findByUuid(document_.scene(), granularUuid)) {
                GranularConfig pour;
                pour.capacity = 20000;
                pour.rate = 4000.f;
                pour.write(*live);
                step();
            }

            const auto grainVisuals = [&] {
                std::size_t n = 0;
                document_.scene().traverse([&](Object3D& o) {
                    if (document_.isEditorOnly(o)) return;
                    if (o.name == "Grains (play)") ++n;
                });
                return n;
            };

            startPlay();
            stepFixed(120);

            // Asked of the WORLD rather than of the session, so the branch below
            // is not the session grading its own homework.
            const bool cuda = physics_ && physics_->world() &&
                              physics_->world()->cudaContextManager() != nullptr;
            check(granularSession_ && granularSession_->granularNodeCount() == 1,
                  "the play session counts the authored chute");
            if (cuda) {
                check(granularSession_ && !granularSession_->declined() &&
                              granularSession_->groupCount() == 1,
                      "and pours it into one PBD group on a CUDA machine");
                check(granularSession_ && granularSession_->activeGrainCount() > 0,
                      "grains are emitted through the chute frame");
                check(grainVisuals() == 1, "and drawn by a visual in the played scene");
            } else {
                check(granularSession_ && granularSession_->declined() &&
                              granularSession_->groupCount() == 0,
                      "and declines with a line where there is no CUDA device");
                check(grainVisuals() == 0, "drawing nothing it cannot simulate");
            }

            stopPlay();
            step();
            check(grainVisuals() == 0, "Stop takes the grain visual with it");
            check(granularSession_ && granularSession_->activeGrainCount() == 0,
                  "and the solver with it");
            check(GranularConfig::read(*findByUuid(document_.scene(), granularUuid)).has_value(),
                  "leaving the authored chute exactly as it was");
        }
#endif

        commands_.undo();
        step();
        check(findByUuid(document_.scene(), granularUuid) == nullptr,
              "undo of the add removes the granular node");

        newScene();
        step(2);
    }

    // A description reports its failure to the console and raises no modal, so asking
    // importError_ alone would pass no matter what happened to it.
    const auto importFailed = [this] {
        if (!importError_.empty()) return true;
        return std::any_of(console_.begin(), console_.end(), [](const std::string& line) {
            return line.find("import failed:") != std::string::npos;
        });
    };

    // With a model path on the command line, exercise the async import path
    // end to end: queue -> worker -> finalize -> selected group in the scene.
    if (!options_.openOnStart.empty() && !formats::isScene(options_.openOnStart) &&
        section("import")) {
        const auto childrenBefore = document_.scene().children.size();
        importModel(options_.openOnStart);
        int budget = 3000;// frames; the worker is genuinely asynchronous
        while ((activeImport_ || !importQueue_.empty()) && budget-- > 0) step();
        check(budget > 0, "import completed in time");
        check(!importFailed(), "import reported no error");
        check(document_.scene().children.size() == childrenBefore + 1,
              "import added one group to the scene");
        check(selection_.get() != nullptr, "import selected the new group");

        // Animation preview lifecycle, on the model just imported. Frames run
        // in both states so the inspector draws each branch.
        auto* imported = selection_.get();
        if (imported && !imported->animations.empty()) {

            const auto pose = [imported] {
                std::vector<Vector3> out;
                imported->traverse([&out](Object3D& node) { out.push_back(node.position); });
                return out;
            };
            const auto rest = pose();
            const auto undosBefore = commands_.undoCount();

            startAnimationPreview(*imported, "", true, 1.f);
            check(isPreviewing(*imported), "preview started");
            step(20);
            const auto moved = pose();
            check(moved != rest, "preview actually animates the subtree");

            stopAnimationPreview();
            check(!isPreviewing(*imported), "preview stopped");
            check(pose() == rest, "stopping preview restores the authored pose");
            check(commands_.undoCount() == undosBefore, "preview pushes no undo entries");
            step(5);
        }

        // Save by reference, through the same path File ▸ Save takes. The
        // imported subtree is the expensive part of any real scene, so this is
        // where the setting has to actually pay off — and where a reopened
        // document has to come back identical.
        if (imported) {

            const auto linked = imported->uuid;
            const auto nodesBefore = [&] {
                std::size_t n = 0;
                imported->traverse([&n](Object3D&) { ++n; });
                return n;
            }();
            check(!assetSource(*imported).empty(), "an imported subtree records its source file");

            const auto embeddedPath = std::filesystem::temp_directory_path() / "threepp_editor_embed.json";
            const auto referencedPath = std::filesystem::temp_directory_path() / "threepp_editor_ref.json";

            setImageStorage(ImageStorage::Embed);
            setModelStorage(ModelStorage::Embed);
            saveSceneAs(embeddedPath);
            step();

            setImageStorage(ImageStorage::Reference);
            setModelStorage(ModelStorage::Reference);
            saveSceneAs(referencedPath);
            step();

            std::error_code ec;
            const auto embeddedSize = std::filesystem::file_size(embeddedPath, ec);
            const auto referencedSize = std::filesystem::file_size(referencedPath, ec);
            check(!ec && embeddedSize > 0 && referencedSize > 0, "both documents were written");
            check(referencedSize * 10 < embeddedSize,
                  "referencing the model makes the document at least 10x smaller");

            openScene(referencedPath);
            step(2);
            auto* reopened = findByUuid(document_.scene(), linked);
            check(reopened != nullptr, "the linked subtree is back after reopening");
            if (reopened) {
                std::size_t n = 0;
                reopened->traverse([&n](Object3D&) { ++n; });
                check(n == nodesBefore, "and it came back with every node");
                check(!assetSource(*reopened).empty(), "still linked, so a re-save references it again");

                // Unlink is the escape hatch: the same scene, written in full.
                selectObject(reopened);
                step();
                unlinkSelectedAsset();
                step();
                check(assetSource(*reopened).empty(), "unlink breaks the link");
                saveSceneAs(referencedPath);
                step();
                const auto unlinkedSize = std::filesystem::file_size(referencedPath, ec);
                check(!ec && unlinkedSize > referencedSize * 10,
                      "an unlinked subtree is written out in full again");
            }

            std::filesystem::remove(embeddedPath, ec);
            std::filesystem::remove(referencedPath, ec);

            setImageStorage(ImageStorage::Embed);
            setModelStorage(ModelStorage::Embed);
            newScene();
            step(2);
        }
    }

    // --- One file: the .tpz archive -------------------------------------------
    // Nothing in the editor decides this: DocumentFormat::Auto reads the name it
    // is given, so Save As with an archive name has to write an archive and Open
    // has to take it back without being told which of the two it is.
    if (section("tpz-archive")) {
        newScene();
        step(2);

        auto* subject = document_.scene().children.empty() ? nullptr : document_.scene().children.front();
        const auto uuid = subject ? subject->uuid : std::string{};

        const auto archivePath = std::filesystem::temp_directory_path() / "threepp_editor_archive.tpz";
        std::error_code ec;
        std::filesystem::remove(archivePath, ec);

        saveSceneAs(archivePath);
        step();
        check(std::filesystem::exists(archivePath, ec) && std::filesystem::file_size(archivePath, ec) > 0,
              "Save As with a .tpz name writes an archive");

        openScene(archivePath);
        step(2);
        check(document_.path() == archivePath, "and Open takes it back as the document");
        check(!uuid.empty() && findByUuid(document_.scene(), uuid) != nullptr,
              "with the scene that went into it");

        std::filesystem::remove(archivePath, ec);
        newScene();
        step(2);
    }

    // --- prefabs: a subtree saved as a document, and brought back -------------
    // The whole feature rests on one non-obvious thing: a loaded prefab must be
    // given fresh uuids. ObjectLoader adopts what it reads and ObjectExporter
    // dedupes by uuid, so without that step two instances of the same file are
    // one set of materials as far as the NEXT save is concerned — and the trap
    // only springs on reload, which is why the third block below exists.
    if (section("prefabs")) {
        newScene();
        step(2);

        const auto prefabPath = std::filesystem::temp_directory_path() / "threepp_editor_prefab.json";
        const auto reloadPath = std::filesystem::temp_directory_path() / "threepp_editor_prefab_scene.json";
        std::error_code ec;
        std::filesystem::remove(prefabPath, ec);
        std::filesystem::remove(reloadPath, ec);

        auto source = Group::create();
        source->name = "PrefabSource";
        auto sourceMaterial = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(0x3366cc));
        sourceMaterial->name = "PrefabPaint";
        auto sourceMesh = Mesh::create(BoxGeometry::create(1, 1, 1), sourceMaterial);
        sourceMesh->name = "PrefabBody";
        source->add(sourceMesh);
        auto sourceTip = Group::create();
        sourceTip->name = "PrefabTip";
        sourceTip->position.y = 1.f;
        source->add(sourceTip);

        addObject(source, document_.scene(), "Add PrefabSource");
        step();

        const auto sourceUuid = source->uuid;
        const auto sourceMeshUuid = sourceMesh->uuid;

        savePrefab(*source, prefabPath);
        step();
        check(std::filesystem::exists(prefabPath, ec) && std::filesystem::file_size(prefabPath, ec) > 0,
              "a subtree saves as a document of its own");

        addPrefab(prefabPath, document_.scene());
        step();
        auto* first = selection_.get();
        check(first != nullptr && first != source.get(), "loading it back adds a second subtree");
        if (!first) {
            check(false, "prefab instantiation produced a root");
        } else {
            check(first->children.size() == 2, "with the children it was saved with");
            check(first->getObjectByName("PrefabBody") != nullptr &&
                          first->getObjectByName("PrefabTip") != nullptr,
                  "under the names it was saved with");
            auto* firstMesh = first->getObjectByName("PrefabBody");
            check(first->uuid != sourceUuid, "the instance is not the subtree it came from");
            check(firstMesh && firstMesh->uuid != sourceMeshUuid,
                  "and neither is anything under it");

            // Second instantiation of the SAME file. Two objects that came from
            // one document have to be two objects.
            addPrefab(prefabPath, document_.scene());
            step();
            auto* second = selection_.get();
            check(second != nullptr && second != first, "the same file instantiates twice");
            auto* secondMesh = second ? second->getObjectByName("PrefabBody") : nullptr;
            if (second) {
                check(first->name != second->name, "each instance gets its own name");
                check(first->uuid != second->uuid, "and its own identity");
            }
            Material* firstPaint = nullptr;
            Material* secondPaint = nullptr;
            if (auto* m = firstMesh ? firstMesh->as<Mesh>() : nullptr) firstPaint = m->material().get();
            if (auto* m = secondMesh ? secondMesh->as<Mesh>() : nullptr) secondPaint = m->material().get();
            check(firstPaint != nullptr && secondPaint != nullptr && firstPaint != secondPaint,
                  "and its own material object");
            check(firstPaint && secondPaint && firstPaint->uuid() != secondPaint->uuid(),
                  "with an identity the exporter will not merge");

            // The regression the uuid regeneration exists for. Save the scene
            // holding both instances, read it back with the plain loader, and
            // recolour one: shared uuids would have collapsed the two material
            // entries into one on the way out, and the other instance would
            // change colour with it.
            const auto firstName = first->name;
            const auto secondName = second ? second->name : std::string{};
            saveSceneAs(reloadPath);
            step();

            ObjectLoader reloader;
            auto reloaded = reloader.load(reloadPath);
            check(reloaded != nullptr, "a scene holding two instances reloads");
            auto* reloadedFirst = reloaded ? reloaded->getObjectByName(firstName) : nullptr;
            auto* reloadedSecond = reloaded && !secondName.empty()
                                           ? reloaded->getObjectByName(secondName)
                                           : nullptr;
            // Each search starts at the instance, so the two "PrefabBody"
            // children cannot be confused for each other.
            auto* reloadedFirstMesh = reloadedFirst ? reloadedFirst->getObjectByName("PrefabBody") : nullptr;
            auto* reloadedSecondMesh = reloadedSecond ? reloadedSecond->getObjectByName("PrefabBody") : nullptr;
            auto* a = reloadedFirstMesh ? reloadedFirstMesh->as<Mesh>() : nullptr;
            auto* b = reloadedSecondMesh ? reloadedSecondMesh->as<Mesh>() : nullptr;
            check(a != nullptr && b != nullptr && a->material() != b->material(),
                  "and each instance still owns its material after the round trip");

            auto* colourA = a ? dynamic_cast<MaterialWithColor*>(a->material().get()) : nullptr;
            auto* colourB = b ? dynamic_cast<MaterialWithColor*>(b->material().get()) : nullptr;
            if (colourA && colourB) {
                const Color before = colourB->color;
                colourA->color = Color(0xff0000);
                check(colourB->color.equals(before),
                      "so recolouring one instance leaves the other alone");
            } else {
                check(false, "reloaded prefab instances expose a colour to drive");
            }
        }

        std::filesystem::remove(prefabPath, ec);
        std::filesystem::remove(reloadPath, ec);
        newScene();
        step(2);
    }

    // --- The look rides with the document -------------------------------------
    // What the Renderer Settings panel writes to is the renderer itself, so this
    // drives the renderer directly and then goes through Save / New / Open — the
    // three app paths that have to carry, clear and restore it.
    if (section("renderer-look")) {
        newScene();
        step(2);

        const auto renderPath = std::filesystem::temp_directory_path() / "threepp_editor_render.json";
        const auto entry = [this] {
            const auto& userData = document_.scene().userData;
            const auto it = userData.find(RenderConfig::userDataKey);
            return it != userData.end() && it->second.type() == typeid(std::string)
                           ? std::any_cast<const std::string&>(it->second)
                           : std::string();
        };

        saveSceneAs(renderPath);
        step();
        check(entry().empty(), "a document nobody re-lit saves no render block");

        constexpr float kExposure = 1.75f;
        renderer_->toneMappingExposure = kExposure;
#ifdef THREEPP_WITH_VULKAN
        float wantScale = 0.f;
        if (auto* vk = dynamic_cast<VulkanRenderer*>(renderer_.get())) {
            vk->setBloomIntensity(0.4f);
            // Deliberately not the editor's 0.8 default: this is the field that
            // has to come back from the file rather than from the constructor.
            wantScale = 0.5f;
            vk->setRenderScale(wantScale);
        }
#endif
        saveSceneAs(renderPath);
        step();
        check(entry().find("exposure=1.75") != std::string::npos,
              "an adjusted look is written into the document");

        newScene();
        step(2);
        check(std::abs(renderer_->toneMappingExposure - renderDefaults_.exposure) < 1e-4f,
              "and a new document goes back to the editor's defaults");

        openScene(renderPath);
        step(2);
        check(std::abs(renderer_->toneMappingExposure - kExposure) < 1e-4f,
              "reopening restores the look the document was saved with");
#ifdef THREEPP_WITH_VULKAN
        if (auto* vk = dynamic_cast<VulkanRenderer*>(renderer_.get())) {
            check(std::abs(vk->bloomIntensity() - 0.4f) < 1e-4f, "bloom and all");
            check(std::abs(vk->renderScale() - wantScale) < 1e-4f,
                  "including a render scale that is not the editor default");
        }
#endif

        std::error_code renderEc;
        std::filesystem::remove(renderPath, renderEc);
        newScene();
        step(2);
    }

    // URDF: import, drive a joint, and prove the pose survives a full document
    // round trip. Play/Stop is that round trip — it serialises the scene and
    // rebuilds it — so this covers re-articulation as the user meets it.
    if (!options_.urdf.empty() && section("urdf")) {

        importModel(options_.urdf);
        int budget = 6000;
        while ((activeImport_ || !importQueue_.empty()) && budget-- > 0) step();
        check(budget > 0, "urdf import completed in time");
        check(!importFailed(), "urdf import reported no error");

        auto* robot = selection_.get() ? selection_.get()->as<Robot>() : nullptr;
        check(robot != nullptr, "urdf import yields a Robot");

        if (robot) {
            check(robot->numDOF() > 0, "the robot has articulated joints");
            const auto uuid = robot->uuid;
            const auto config = RobotConfig::read(*robot);
            check(config && !config->urdf.empty(), "the robot records its source file");

            setJointValue(*robot, 0, 0.3f);
            step();
            check(std::abs(robot->getJointValue(0) - 0.3f) < 1e-3f, "a joint value applies");
            const auto posed = RobotConfig::read(*robot);
            check(posed && !posed->joints.empty() && std::abs(posed->joints[0] - 0.3f) < 1e-3f,
                  "the pose is recorded for the document");

            startPlay();
            step(3);
            stopPlay();
            step();

            Object3D* found = nullptr;
            document_.scene().traverse([&](Object3D& o) {
                if (!found && o.uuid == uuid) found = &o;
            });
            check(found != nullptr, "the robot survives the round trip");
            auto* live = found ? found->as<Robot>() : nullptr;
            check(live != nullptr, "it comes back articulated, not frozen");
            check(live && std::abs(live->getJointValue(0) - 0.3f) < 1e-3f,
                  "the joint pose survives the round trip");

            // A sensor authored INSIDE the robot rather than on its root has to
            // survive the same rebuild. The node a viewport click drills down to
            // is a mesh under a link's visual group, and a URDF leaves those
            // unnamed — which is why the carry-over cannot key on the name. This
            // is only about the authored entry outliving Stop, so the sensor is
            // never asked to measure anything.
            if (live) {

                Object3D* anyMesh = nullptr;
                Object3D* unnamedMesh = nullptr;
                live->traverse([&](Object3D& node) {
                    if (&node == live || node.type() != "Mesh") return;
                    if (!anyMesh) anyMesh = &node;
                    if (!unnamedMesh && node.name.empty()) unnamedMesh = &node;
                });
                // The unnamed one is the interesting case; a named mesh still
                // exercises the carry-over if this URDF's loader names them.
                Object3D* deep = unnamedMesh ? unnamedMesh : anyMesh;
                check(deep != nullptr, "the robot has a mesh to author a sensor on");

                if (deep) {
                    SensorConfig deepSensor;
                    deepSensor.enabled = true;
                    deepSensor.type = SensorConfig::Type::Imu;
                    deepSensor.rateHz = 0.f;
                    deepSensor.write(*deep);

                    startPlay();
                    step(3);
                    stopPlay();
                    step();

                    Object3D* rebuiltRobot = nullptr;
                    document_.scene().traverse([&](Object3D& o) {
                        if (!rebuiltRobot && o.uuid == uuid) rebuiltRobot = &o;
                    });
                    std::size_t authored = 0;
                    bool onRoot = false;
                    if (rebuiltRobot) {
                        rebuiltRobot->traverse([&](Object3D& node) {
                            if (!SensorConfig::read(node)) return;
                            ++authored;
                            if (&node == rebuiltRobot) onRoot = true;
                        });
                    }
                    check(authored == 1 && !onRoot,
                          "a sensor authored on a mesh inside the robot survives play/stop");

                    // Leave the robot as this section found it: the PhysX pass
                    // below authors its own encoder and counts live sensors.
                    if (rebuiltRobot) {
                        rebuiltRobot->traverse([](Object3D& node) { SensorConfig::erase(node); });
                    }

                    // Stop replaced the scene, so everything above points into a
                    // graph that no longer exists. Re-seat the handles the rest
                    // of this section uses.
                    found = rebuiltRobot;
                    live = rebuiltRobot ? rebuiltRobot->as<Robot>() : nullptr;
                }
            }

            // A node authored INTO the robot — the way a robot cell is built:
            // drop a camera on the wrist, a marker on a link. Play/Stop used to
            // rebuild the robot's subtree from its URDF, which deleted every one
            // of them; the document carries the joint table now, so Stop restores
            // the subtree it saved instead of re-reading the file.
            if (live) {

                Object3D* mount = nullptr;
                live->traverse([&](Object3D& node) {
                    // A LINK, not the root and not a mesh: the node an authored
                    // camera would actually be parented to.
                    if (mount || &node == live || node.name.empty()) return;
                    if (node.type() == "Mesh") return;
                    mount = &node;
                });
                check(mount != nullptr, "the robot has a named node to author onto");

                if (mount) {
                    const auto mountUuid = mount->uuid;

                    auto bolted = ObjectFactory::createCamera(document_.scene());
                    bolted->name = "Bolted Sensor Cam";
                    bolted->position.set(0.f, 0.f, 0.1f);
                    const auto boltedUuid = bolted->uuid;
                    addObject(bolted, *mount, "Bolt Camera To Link");
                    step();

                    startPlay();
                    step(3);
                    stopPlay();
                    step();

                    Object3D* survivor = nullptr;
                    document_.scene().traverse([&](Object3D& o) {
                        if (!survivor && o.uuid == boltedUuid) survivor = &o;
                    });
                    check(survivor != nullptr,
                          "a node authored under a robot link survives play/stop");
                    check(survivor && survivor->parent && survivor->parent->uuid == mountUuid,
                          "and it is still parented to the link it was bolted to");

                    // Put the robot back the way the rest of this section expects
                    // to find it.
                    if (survivor) survivor->removeFromParent();
                    step();

                    Object3D* rebuiltRobot = nullptr;
                    document_.scene().traverse([&](Object3D& o) {
                        if (!rebuiltRobot && o.uuid == uuid) rebuiltRobot = &o;
                    });
                    found = rebuiltRobot;
                    live = rebuiltRobot ? rebuiltRobot->as<Robot>() : nullptr;
                }
            }

            // Collision hulls: hidden on import, and the opt-in has to outlive
            // the rebuild or it would reset every time play is pressed. A URDF
            // with no <collision> elements has nothing to toggle, so the checks
            // only run when collider nodes exist.
            const auto hasColliders = [](Object3D& root) {
                bool any = false;
                root.traverse([&any](Object3D& node) {
                    if (node.userData.contains("collider")) any = true;
                });
                return any;
            };
            const auto colliderVisible = [](Object3D& root) {
                bool visible = false;
                root.traverse([&visible](Object3D& node) {
                    if (node.userData.contains("collider") && node.visible) visible = true;
                });
                return visible;
            };
            if (live && !hasColliders(*live)) {
                check(true, "no collision geometry in this urdf - the collider toggle is skipped");
            } else if (live) {
                check(!colliderVisible(*live), "collision geometry is hidden by default");

                live->showColliders(true);
                auto shown = RobotConfig::read(*live).value_or(RobotConfig{});
                shown.showColliders = true;
                shown.write(*live);
                step();
                check(colliderVisible(*live), "collision geometry can be shown");

                startPlay();
                step(3);
                stopPlay();
                step();
                Object3D* again = nullptr;
                document_.scene().traverse([&](Object3D& o) {
                    if (!again && o.uuid == uuid) again = &o;
                });
                check(again && colliderVisible(*again),
                      "the collider toggle survives the round trip");
            }

#ifdef THREEPP_EDITOR_WITH_PHYSX
            // Simulate the robot as a PhysX articulation and read a joint of it
            // with a live encoder, played headlessly the way the rest of this
            // section drives play. The ifdef is not optional: without PhysX the
            // session types are forward declarations, so touching their members
            // is a compile error, not a false null check — CI builds the editor
            // with Python and no PhysX.
            Object3D* current = nullptr;
            document_.scene().traverse([&](Object3D& o) {
                if (!current && o.uuid == uuid) current = &o;
            });
            auto* sim = current ? current->as<Robot>() : nullptr;
            if (sim && physics_ && sensors_) {

                const auto joints = sim->getArticulatedJointInfo();
                check(!joints.empty(), "the simulated robot exposes a joint to read");
                const std::string jointName = joints.empty() ? "" : joints.front().name;

                // Author "Simulate" on the robot and an encoder on it, naming the
                // joint. The encoder sits on the root, which findArticulation
                // resolves by walking up from any node in the robot.
                ArticulationConfig art;
                art.enabled = true;
                art.fixedBase = true;
                art.write(*sim);

                SensorConfig enc;
                enc.enabled = true;
                enc.type = SensorConfig::Type::Encoder;
                enc.rateHz = 0.f;// every substep
                enc.joint = jointName;
                enc.write(*sim);

                startPlay();
                check(physics_->articulationCount() == 1,
                      "the robot plays as one articulation");
                check(sensors_->liveCount() >= 1, "the joint encoder comes up live");
                // A handful of frames so the substep loop feeds the encoder.
                step(20);

                const SensorPlaySession::Entry* encEntry = nullptr;
                for (const auto& e : sensors_->entries()) {
                    if (e->encoder) encEntry = e.get();
                }
                check(encEntry && encEntry->samples > 0,
                      "the encoder produced samples during play");
                // The drive held the authored pose (0.3 rad on joint 0) rather
                // than letting gravity collapse it.
                if (encEntry && encEntry->encoder) {
                    const auto sample = encEntry->encoder->latest();
                    check(sample.has_value() && std::abs(sample->position - 0.3f) < 0.25f,
                          "the encoder reads the pose the drive is holding");
                }

                stopPlay();
                step();
                // A second Play/Stop with a Force/Torque sensor exercises the
                // teardown-order path (physics stops first, the FT cache must not
                // be released against a dead world).
                Object3D* rebuilt = nullptr;
                document_.scene().traverse([&](Object3D& o) {
                    if (!rebuilt && o.uuid == uuid) rebuilt = &o;
                });
                if (auto* sim2 = rebuilt ? rebuilt->as<Robot>() : nullptr) {
                    SensorConfig ft;
                    ft.enabled = true;
                    ft.type = SensorConfig::Type::ForceTorque;
                    ft.rateHz = 0.f;
                    ft.joint = jointName;
                    ft.write(*sim2);
                    startPlay();
                    step(10);
                    stopPlay();
                    step();
                    check(true, "a Force/Torque sensor plays and stops without a crash");
                }

#ifdef THREEPP_EDITOR_WITH_PYTHON
                // threepp.editor.articulation_from_object: a script commanding
                // the articulation the physics session is simulating — the seam
                // a policy-playback script stands on. The script raises if the
                // handle does not arrive, so a broken lookup fails the error
                // check rather than reading as "the drive was slow".
                Object3D* scripted = nullptr;
                document_.scene().traverse([&](Object3D& o) {
                    if (!scripted && o.uuid == uuid) scripted = &o;
                });
                if (auto* sim3 = scripted ? scripted->as<Robot>() : nullptr) {
                    setInlineScript(*sim3,
                                    "import threepp\n"
                                    "\n"
                                    "class Waver:\n"
                                    "    def start(self, obj):\n"
                                    "        self.art = threepp.editor.articulation_from_object(obj)\n"
                                    "        assert self.art is not None, 'no articulation handle'\n"
                                    "        assert self.art.num_dof == len(self.art.joint_names)\n"
                                    "        assert len(self.art.joint_positions) == self.art.num_dof\n"
                                    "\n"
                                    "    def update(self, dt):\n"
                                    "        self.art.set_drive_target('" + jointName + "', 0.8)\n",
                                    "Waver Script");
                    step();

                    startPlay();
                    // Fixed dt: 90 frames are 1.5 s, plenty for the default
                    // 500/50 drive to swing the joint from its authored 0.3 rad
                    // most of the way to the scripted 0.8 setpoint.
                    stepFixed(90);
                    Object3D* live = nullptr;
                    document_.scene().traverse([&](Object3D& o) {
                        if (!live && o.uuid == uuid) live = &o;
                    });
                    auto* liveRobot = live ? live->as<Robot>() : nullptr;
                    check(liveRobot && liveRobot->numDOF() > 0 &&
                                  liveRobot->getJointValue(0) > 0.55f,
                          "a script drives the articulation the physics session is simulating");
                    // The error text in the message, because "it raised" without
                    // WHAT it raised is a failure you cannot act on.
                    const std::string scriptError =
                            scripts_ ? scripts_->errorFor(uuid) : "no script host";
                    const std::string raisedMsg =
                            "and reaching it raised nothing" +
                            (scriptError.empty() ? "" : " - " + scriptError);
                    check(scripts_ && scriptError.empty(), raisedMsg.c_str());
                    stopPlay();
                    step();
                }
#endif
            }
#endif
        }
    }

    // --- a millimetre robot lands in a metre scene --------------------------
    // The CAD-export story end to end: import a URDF drawn in millimetres, set
    // the scale the way you would for any other import, and have the thing
    // SIMULATE at the size it renders. A PhysX link has no scale of its own, so
    // the factor is folded into the URDF description at build time; before that
    // a scaled robot was refused outright and just sat there.
    if (section("urdf-scale")) {
        newScene();
        step(2);

        const auto dir = std::filesystem::temp_directory_path() / "threepp-editor-units";
        std::filesystem::create_directories(dir);
        const auto path = dir / "mm_arm.urdf";
        {
            std::ofstream out(path, std::ios::trunc);
            out << R"(
        <robot name="mm_arm">
          <link name="base_link">
            <visual><geometry><box size="200 200 200"/></geometry></visual>
            <collision><geometry><box size="200 200 200"/></geometry></collision>
          </link>
          <link name="upper_link">
            <visual><geometry><box size="100 400 100"/></geometry></visual>
            <collision><geometry><box size="100 400 100"/></geometry></collision>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base_link"/><child link="upper_link"/>
            <origin xyz="0 0 300" rpy="0 0 0"/><axis xyz="0 0 1"/>
            <limit lower="-2.0" upper="2.0"/>
          </joint>
        </robot>)";
        }

        importModel(path);
        int budget = 6000;
        while ((activeImport_ || !importQueue_.empty()) && budget-- > 0) step();
        check(budget > 0 && !importFailed(), "the millimetre urdf imported");

        auto* robot = selection_.get() ? selection_.get()->as<Robot>() : nullptr;
        check(robot != nullptr, "and came in as a Robot");

        // Scaling it is the user's call and the user's job — the same scale
        // field as any other import, through the command stack because that is
        // the path the inspector takes.
        if (robot) {
            const auto before = SetTransformCommand::read(*robot);
            auto after = before;
            after.scale.set(0.001f, 0.001f, 0.001f);
            commands_.execute(std::make_unique<SetTransformCommand>(*robot, before, after, "Scale"));
            step();
        }
        check(robot && std::abs(robot->scale.x - 0.001f) < 1e-6f, "scaled into metres");

#ifdef THREEPP_EDITOR_WITH_PHYSX
        if (robot && physics_) {

            // Play replaces the scene, so the robot has to be found again after.
            const std::string uuid = robot->uuid;

            ArticulationConfig art;
            art.enabled = true;
            art.fixedBase = true;
            art.write(*robot);
            // Authored well off zero so a robot that fails to simulate (or one
            // whose drive collapses it) is visibly different from one that holds.
            setJointValue(*robot, 0, 0.5f);
            step();

            startPlay();
            check(physics_->articulationCount() == 1,
                  "a uniformly scaled robot now plays as an articulation");
            step(30);

            Object3D* live = nullptr;
            document_.scene().traverse([&](Object3D& o) {
                if (!live && o.uuid == uuid) live = &o;
            });
            auto* liveRobot = live ? live->as<Robot>() : nullptr;
            check(liveRobot && liveRobot->numDOF() > 0 &&
                          std::abs(liveRobot->getJointValue(0) - 0.5f) < 0.2f,
                  "and holds its authored pose at millimetre scale");

            stopPlay();
            step();
        }
#endif

        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }

    // --- dropped textures find the right slot -------------------------------
    // A file dropped from the OS carries no ImGui payload, so the slot has to
    // be worked out: from the row the cursor is over, else from the file name.
    // Until this existed every drop went to `map`, as sRGB, whatever it was.
    if (section("texture-drop")) {
        newScene();
        step(2);

        auto* box = document_.scene().getObjectByName("Box");
        auto* mesh = box ? box->as<Mesh>() : nullptr;
        auto material = mesh ? mesh->materialAs<MeshStandardMaterial>() : nullptr;
        check(material != nullptr, "the template Box has a standard material");

        // Real files, because the drop path actually decodes them — but written
        // here rather than taken from the asset repo, so the check runs in an
        // editor-only build too. A 2x2 24-bit BMP is the smallest thing every
        // image decoder agrees on.
        const auto writeBmp = [](const std::filesystem::path& path) {
            const unsigned char bytes[70]{
                    'B', 'M', 70, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0,
                    40, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 24, 0,
                    0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    // two rows of two pixels, each padded to a 4-byte boundary
                    255, 255, 255, 128, 128, 128, 0, 0,
                    128, 128, 128, 255, 255, 255, 0, 0};
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
            return out.good();
        };

        const auto temp = std::filesystem::temp_directory_path();
        const auto plain = temp / "threepp_drop_checker.bmp";
        const auto named = temp / "threepp_drop_normal.bmp";
        std::error_code ec;
        const bool wrote = writeBmp(plain) && writeBmp(named);
        check(wrote, "the drop test could stage its images");

        if (material && wrote) {

            selectObject(mesh);
            step(2);

            // 1. An unrecognised name, dropped nowhere in particular: the base
            //    colour map, exactly as before.
            handleFileDrop({plain.string()});
            resolveTextureDrops(-1.f, -1.f);
            check(material->map != nullptr, "a plain texture drop still fills map");
            check(material->map && material->map->colorSpace == ColorSpace::sRGB,
                  "and a colour map decodes from sRGB");
            check(material->normalMap == nullptr, "and nothing else");

            // 2. Named "..._normal": inferred, and NOT tagged sRGB. That tag is
            //    the part that used to be silently wrong.
            handleFileDrop({named.string()});
            resolveTextureDrops(-1.f, -1.f);
            check(material->normalMap != nullptr, "a *_normal drop is inferred to normalMap");
            check(material->normalMap && material->normalMap->colorSpace == ColorSpace::NoColorSpace,
                  "and a data map is not decoded as sRGB");

            // 3. Dropped onto a specific row, which outranks the name: the
            //    *_normal file aimed at roughnessMap lands in roughnessMap.
            //    The section is collapsed until asked for, exactly as a user
            //    would have to expand it before dropping on a row.
            openTextureSectionOnce_ = true;
            step(2);// let the inspector lay the rows out
            check(!frameTextureSlots_.empty(), "the inspector publishes its texture rows");

            const auto row = std::find_if(frameTextureSlots_.begin(), frameTextureSlots_.end(),
                                          [](const FrameTextureSlot& slot) {
                                              return slot.target.slot == "roughnessMap";
                                          });
            check(row != frameTextureSlots_.end(), "including one for roughnessMap");

            if (row != frameTextureSlots_.end()) {
                const float x = (row->minX + row->maxX) * 0.5f;
                const float y = (row->minY + row->maxY) * 0.5f;
                handleFileDrop({named.string()});
                resolveTextureDrops(x, y);
                check(material->roughnessMap != nullptr,
                      "a drop on a slot row goes to that slot, not the one its name suggests");
                check(material->roughnessMap && material->roughnessMap->colorSpace == ColorSpace::NoColorSpace,
                      "in that slot's colour space");
            }

            // 4. Undoable like every other material edit.
            const auto before = material->roughnessMap;
            commands_.undo();
            check(material->roughnessMap != before, "a dropped texture can be undone");
        }

        std::filesystem::remove(plain, ec);
        std::filesystem::remove(named, ec);
    }

    // --- texture settings ---------------------------------------------------
    // Tiling and the per-slot sampling settings, driven through the very
    // commands the inspector's widgets push. GL shares ONE uv transform between
    // a material's maps, so the tiling command writes every assigned map at
    // once; wrap, filtering and anisotropy belong to the texture and write it
    // alone.
    //
    // The two documents this leaves behind are as much the point as the
    // assertions: `--screenshot` over them is how a person LOOKS at what a
    // repeat of 4 did, without a bespoke scene having to be authored for it.
    if (section("texture-settings")) {
        newScene();
        step(2);

        auto* box = document_.scene().getObjectByName("Box");
        auto* mesh = box ? box->as<Mesh>() : nullptr;
        auto material = mesh ? mesh->materialAs<MeshStandardMaterial>() : nullptr;
        check(material != nullptr, "the template Box has a standard material");

        if (material) {

            // An 8x8 checker built here rather than loaded: what the settings do
            // to it is the subject, and a checker is the one pattern where a
            // repeat of 4 is unmistakable in a photograph.
            std::vector<unsigned char> pixels;
            pixels.reserve(8 * 8 * 4);
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    const unsigned char v = ((x + y) % 2) ? 235 : 30;
                    pixels.insert(pixels.end(), {v, v, v, 255});
                }
            }
            auto texture = Texture::create(Image(std::move(pixels), 8, 8));
            texture->name = "checker";
            texture->colorSpace = ColorSpace::sRGB;
            texture->needsUpdate();

            const auto setMap = [material](const std::shared_ptr<Texture>& t) { material->map = t; };
            commands_.execute(std::make_unique<SetMaterialMapCommand>(
                    *material, "map", setMap, material->map, texture));
            step(2);
            check(material->map == texture, "the checker lands in the map slot");

            const auto plainDoc = std::filesystem::temp_directory_path() / "threepp_editor_tiling_1x1.json";
            const auto tiledDoc = std::filesystem::temp_directory_path() / "threepp_editor_tiling_4x4.json";
            saveSceneAs(plainDoc);
            std::cout << "[tiling] wrote " << plainDoc.string() << std::endl;

            const std::vector<std::shared_ptr<Texture>> maps{texture};

            auto tiling = uvTransformOf(*texture);
            tiling.repeat.set(4, 4);
            applyUvTransform(*material, maps, tiling, "Tiling");
            check(texture->repeat.x == 4.f && texture->repeat.y == 4.f,
                  "the UV transform block tiles every assigned map");

            // A repeat above 1 only tiles where the map WRAPS, and a threepp
            // texture arrives clamped - so the picture below tests the wrap path
            // as much as the tiling one.
            auto sampling = samplingOf(*texture);
            sampling.wrapS = sampling.wrapT = TextureWrapping::Repeat;
            sampling.minFilter = Filter::NearestMipmapLinear;
            sampling.magFilter = Filter::Nearest;
            sampling.anisotropy = 4;
            applyTextureSampling(*material, texture, sampling, "Texture Wrap");
            check(texture->wrapS == TextureWrapping::Repeat && texture->wrapT == TextureWrapping::Repeat,
                  "the slot popup writes both wrap axes");
            check(texture->magFilter == Filter::Nearest, "and the filtering");
            check(texture->anisotropy == 4, "and the anisotropy");
            step(2);

            commands_.undo();
            check(texture->wrapS == TextureWrapping::ClampToEdge && texture->magFilter == Filter::Linear,
                  "undo puts the whole sampling state back as one entry");
            commands_.redo();
            check(texture->wrapS == TextureWrapping::Repeat, "and redo re-applies it");

            commands_.undo();// the sampling
            commands_.undo();// the tiling
            check(texture->repeat.x == 1.f && texture->repeat.y == 1.f,
                  "undo restores each map's own tiling");
            commands_.redo();
            commands_.redo();
            check(texture->repeat.x == 4.f && texture->wrapS == TextureWrapping::Repeat,
                  "and redo brings both back");

            // And a picture of the panel that drove all of it: a PASS line says
            // the command reached the texture, never what the section looks
            // like. The bottom panel steps aside for it - the Textures tree is
            // the last thing in a long inspector and sits below the fold with
            // the console open.
            const bool bottomPanelWas = bottomPanelOpen_;
            bottomPanelOpen_ = false;
            selectObject(mesh);
            openTextureSectionOnce_ = true;
            step(3);
            shootTo(std::filesystem::temp_directory_path() / "threepp_editor_tiling_inspector.png");
            bottomPanelOpen_ = bottomPanelWas;

            saveSceneAs(tiledDoc);
            std::cout << "[tiling] wrote " << tiledDoc.string() << std::endl;
            openScene(tiledDoc);
            step(2);

            auto* reopened = document_.scene().getObjectByName("Box");
            auto* reopenedMesh = reopened ? reopened->as<Mesh>() : nullptr;
            auto reloaded = reopenedMesh ? reopenedMesh->materialAs<MeshStandardMaterial>() : nullptr;
            check(reloaded && reloaded->map != nullptr, "the map survives a save and a reopen");
            if (reloaded && reloaded->map) {
                const auto& map = *reloaded->map;
                check(map.repeat.x == 4.f && map.repeat.y == 4.f, "with its tiling");
                check(map.wrapS == TextureWrapping::Repeat && map.wrapT == TextureWrapping::Repeat,
                      "its wrap");
                check(map.magFilter == Filter::Nearest && map.minFilter == Filter::NearestMipmapLinear,
                      "its filtering");
                check(map.anisotropy == 4, "and its anisotropy");
            }
        }
    }

    // --- material shadowSide ----------------------------------------------
    // The inspector's answer to a Double-sided material self-shadowing into a
    // moire. It is only worth exposing if it survives being written down, and
    // Play is the harshest test of that: the snapshot goes through the same
    // exporter/loader pair a saved scene does.
    if (section("shadow-side")) {
        newScene();
        step(2);

        auto* box = document_.scene().getObjectByName("Box");
        auto* mesh = box ? box->as<Mesh>() : nullptr;
        auto material = mesh ? mesh->materialAs<MeshStandardMaterial>() : nullptr;
        check(material != nullptr, "the template Box has a standard material");

        if (material) {
            const auto uuid = box->uuid;
            check(!material->shadowSide.has_value(), "shadowSide starts unset (renderer's rule)");

            material->side = Side::Double;
            material->shadowSide = Side::Back;
            startPlay();
            step(3);
            stopPlay();
            step();

            Object3D* restored = findByUuid(document_.scene(), uuid);
            auto* restoredMesh = restored ? restored->as<Mesh>() : nullptr;
            auto restoredMaterial = restoredMesh ? restoredMesh->materialAs<MeshStandardMaterial>() : nullptr;
            check(restoredMaterial && restoredMaterial->side == Side::Double,
                  "side survives the play round trip");
            check(restoredMaterial && restoredMaterial->shadowSide == Side::Back,
                  "and so does shadowSide");
        }
    }

    // --- a material edit reaches the frame ---------------------------------
    // The inspector writes plain fields: colour, roughness, opacity. Under
    // Vulkan those reach the GPU only when Material::version() moves — the
    // scene diff refreshes an entry's MaterialDesc on that version and never
    // memcmps the live floats — so a bump-less edit sat invisible until some
    // unrelated rebuild happened to re-derive it. Clicking the viewport did
    // exactly that (the selection outline enters or leaves the overlay), which
    // is why the edit appeared to need a click. Nothing but pixels can answer
    // this, so the check reads the frame rather than the material.
    if (section("material-edit")) {
        newScene();
        selectObject(nullptr);// no outline: nothing else can force a rebuild
        step(4);              // and let the temporal history settle first

        // Count of pixels that moved a long way in green, not a frame mean: a
        // box is a small part of the frame, and an average would bury it under
        // the temporal jitter of everything that did not change.
        std::vector<unsigned char> before = renderer_->readRGBPixels();
        check(!before.empty(), "the frame can be read back at all");

        auto* box = document_.scene().getObjectByName("Box");
        auto* mesh = box ? box->as<Mesh>() : nullptr;
        auto material = mesh ? mesh->materialAs<MeshStandardMaterial>() : nullptr;

        if (material && !before.empty()) {
            material->color = Color(0x00ff00);
            material->needsUpdate();// what every inspector setter now does
            step(2);

            const auto after = renderer_->readRGBPixels();
            std::size_t moved = 0;
            const std::size_t n = std::min(before.size(), after.size());
            for (std::size_t i = 1; i < n; i += 3) {
                if (std::abs(static_cast<int>(after[i]) - static_cast<int>(before[i])) > 24) ++moved;
            }
            std::cout << "[probe] material edit: moved " << moved << " of " << (n / 3)
                      << " need " << ((n / 3) / 500) << ", size " << canvas_.size().width() << "x"
                      << canvas_.size().height() << std::endl;
            check(after.size() == before.size() && moved > (n / 3) / 500,
                  "a material edit shows up without a selection change");
        }
    }

    // --- orthographic axis views ------------------------------------------
    // The screenshots are what say the views look right; these say the state
    // machine behind them holds — that the projection swaps without moving what
    // is framed, that picking still works through the ortho unprojection, and
    // that the label follows the camera rather than the last button pressed.
    if (section("ortho-views")) {
        newScene();
        step(2);

        const Vector3 target = orbit_->target;
        const float perspDistance = camera_.position.distanceTo(target);

        check(!orthographic_ && viewPreset_ == ViewPreset::User,
              "the editor starts in a free perspective view");

        setOrthographic(true);
        step(2);
        check(viewCamera().is<OrthographicCamera>(), "the ortho toggle swaps the projection");
        // What the perspective camera covered at the orbit distance.
        const float expected = 2.f * perspDistance * std::tan(math::degToRad(camera_.fov) * 0.5f);
        check(std::abs((ortho_.top - ortho_.bottom) - expected) < 1e-2f,
              "the ortho frustum is sized to the perspective framing");
        check(std::abs(orbit_->target.distanceTo(target)) < 1e-3f,
              "and the toggle leaves the orbit target alone");

        setViewPreset(ViewPreset::Top);
        step(2);
        Vector3 direction;
        direction.subVectors(viewCamera().position, orbit_->target).normalize();
        check(direction.dot(Vector3(0, 1, 0)) > 0.9999f, "Top looks straight down");
        check(viewPreset_ == ViewPreset::Top, "and keeps its label while it does");
        check(std::abs(grid_->rotation.x) < 1e-4f, "the ground grid stays flat under a Top view");
        // The template Ground is a plane at y=0, exactly where the grid is, and
        // the depth buffer decides coplanar surfaces — the grid has to stand
        // off towards whichever side is being looked from.
        check(grid_->position.y > 0.f, "and stands off towards a camera above it");

        setViewPreset(ViewPreset::Bottom);
        step(2);
        check(grid_->position.y < 0.f, "the stand-off flips for a camera below");

        setViewPreset(ViewPreset::Front);
        step(2);
        direction.subVectors(viewCamera().position, orbit_->target).normalize();
        check(direction.dot(Vector3(0, 0, 1)) > 0.9999f, "Front looks down -Z");
        check(std::abs(grid_->rotation.x - math::PI * 0.5f) < 1e-4f,
              "and stands the grid up into the plane being looked at");

        // Picking goes through the ortho unprojection, which is a different
        // Raycaster path entirely — a click in the middle of the viewport has
        // to land on the template Box in front of the camera.
        selectObject(nullptr);
        step();
        const auto* viewport = ImGui::GetMainViewport();
        pickAt(viewport->Pos.x + viewport->Size.x * 0.5f,
               viewport->Pos.y + viewport->Size.y * 0.5f);
        step();
        check(selection_.get() != nullptr, "a click picks through the orthographic projection");

        // Orbiting off the axis retires the label; the projection is not a
        // thing the orbit gets to change.
        viewCamera().position.copy(orbit_->target).add(Vector3(4.f, 3.f, 5.f));
        step(2);
        check(viewPreset_ == ViewPreset::User, "orbiting off the axis drops back to User");
        check(orthographic_, "without leaving orthographic");
        check(std::abs(grid_->rotation.x) < 1e-4f, "and lays the grid back down");

        setOrthographic(false);
        step(2);
        check(!viewCamera().is<OrthographicCamera>(), "the toggle comes back to perspective");
        const float height = 2.f * camera_.position.distanceTo(orbit_->target) *
                             std::tan(math::degToRad(camera_.fov) * 0.5f);
        check(std::abs(height - expected) < 1e-1f,
              "framing the same height it was showing in ortho");
    }

    // --- the view gizmo's animated snap ------------------------------------
    // The corner gizmo goes through startViewTween rather than setViewPreset:
    // the camera swings over a third of a second and lands exactly on the
    // axis, with the label following only at the end. Fixed steps, because
    // the flight is a function of time and the assertions of frame counts.
    if (section("view-gizmo")) {
        setViewPreset(ViewPreset::Front);
        stepFixed(2);

        startViewTween(ViewPreset::Right);
        stepFixed(5);
        Vector3 direction;
        direction.subVectors(viewCamera().position, orbit_->target).normalize();
        check(viewTween_.active, "the view tween is still in flight after five frames");
        check(direction.dot(viewPresetDirection(ViewPreset::Right)) < 0.999f,
              "and has not teleported to the target axis");

        stepFixed(30);
        direction.subVectors(viewCamera().position, orbit_->target).normalize();
        check(!viewTween_.active, "the tween retires when it lands");
        check(direction.dot(viewPresetDirection(ViewPreset::Right)) > 0.9999f,
              "landing exactly on the Right axis");
        check(viewPreset_ == ViewPreset::Right, "with the label following");

        // Asking for the axis the camera already stands on heads for the far
        // side - the gizmo's click-again idiom, same as Ctrl on the numpad.
        startViewTween(ViewPreset::Right);
        check(viewTween_.active && viewTween_.preset == ViewPreset::Left,
              "clicking the active axis heads for the opposite one");
        stepFixed(40);
        direction.subVectors(viewCamera().position, orbit_->target).normalize();
        check(direction.dot(viewPresetDirection(ViewPreset::Left)) > 0.9999f,
              "and lands there");

        setViewPreset(ViewPreset::User);
    }

    // --- terrain -----------------------------------------------------------
    // A terrain is a Mesh whose triangles ARE the document (TerrainConfig): the
    // config seeds them and makes them re-editable, but opening a scene never
    // regenerates. What has to hold is that generation is deterministic, that a
    // parameter edit carries the SCULPT layer across, and that the whole thing
    // survives the file.
    if (section("terrain")) {
        const auto terrainNow = [&](const std::string& uuid) {
            return findByUuid(document_.scene(), uuid);
        };

        auto created = ObjectFactory::createTerrain(document_.scene());
        const auto terrainUuid = created->uuid;
        check(TerrainConfig::isTerrain(*created), "the factory's terrain carries the terrain entry");
        {
            const auto* position = created->geometry()
                                           ? created->geometry()->getAttribute<float>("position")
                                           : nullptr;
            check(position && position->count() > 1000, "and a baked heightfield to show");
            check(TerrainConfig::albedoTexture(*created) != nullptr,
                  "with the splat albedo baked onto its material");
        }
        addObject(created, document_.scene(), "Add Terrain");
        step(2);

        // The factory rolls a fresh seed per terrain (createTree's reason), which
        // is right for authoring and wrong for a photograph: the shot at the end
        // of this block would be a different landscape every run and nobody could
        // tell a regression from a reroll. Pin it here, through the ordinary
        // rebuild path.
        {
            const auto rolled = TerrainConfig::read(*created).value_or(TerrainConfig::makeDefault());
            auto pinned = rolled;
            pinned.params.seed = 20260816u;
            TerrainConfig::rebuild(*created, rolled, pinned);
        }
        const auto config = TerrainConfig::read(*created).value_or(TerrainConfig::makeDefault());
        check(config.params.seed == 20260816u, "a terrain's seed is the config's to set");

        // Determinism is what makes delta recovery possible at all: base(config)
        // has to be the same lattice every time it is asked for, or the "sculpt"
        // recovered from it is noise.
        {
            const auto a = config.bake();
            const auto b = config.bake();
            check(a.heights.size() == b.heights.size() && !a.heights.empty(),
                  "two bakes of one config produce the same lattice size");
            check(a.heights == b.heights, "and byte-identical heights");
        }

        // The brush kernels, headless. A stroke has to move what is under it,
        // leave what is not, and undo has to put the heights back BIT-exactly —
        // an undo that recomputes drifts a little further every cycle.
        {
            const auto lattice = TerrainLattice::of(*created->geometry(), config.dim());
            check(lattice.valid(), "the height lattice reads off the geometry");

            const auto before = TerrainConfig::heightsOf(*created->geometry());
            auto heights = before;

            TerrainBrush brush;
            brush.kind = TerrainBrush::Kind::Raise;
            brush.radius = 10.f;
            brush.strength = 20.f;
            // Straight over the middle of the patch, one tick of a tenth of a
            // second — a real stroke is a few dozen of these.
            const auto rect = TerrainSculpt::apply(heights, lattice, brush, 0.f, 0.f, 0.1f, 0.f);
            check(!rect.empty(), "a raise stroke touches a rect of the lattice");

            const int dim = config.dim();
            const int centre = dim / 2;
            const size_t inside = static_cast<size_t>(centre) * dim + centre;
            check(heights[inside] > before[inside] + 1e-4f, "and lifts the ground inside the brush");
            // A corner: the brush is 10 m across on a 160 m patch, so this is
            // nowhere near it.
            check(heights[0] == before[0] && heights.back() == before.back(),
                  "while everything outside its radius is untouched");

            // Shift digs with the same shape.
            auto inverted = before;
            brush.invert = true;
            TerrainSculpt::apply(inverted, lattice, brush, 0.f, 0.f, 0.1f, 0.f);
            check(inverted[inside] < before[inside] - 1e-4f, "and Shift inverts it into a dig");

            // The stroke's undo entry: the tight rect plus both sides of it.
            const auto patch = TerrainSculpt::diff(before, heights, dim);
            check(!patch.empty() && patch.w < dim && patch.h < dim,
                  "the stroke's undo patch is the rect that moved, not the whole mesh");
            auto restored = heights;
            TerrainSculpt::applyPatch(restored, dim, patch, true);
            check(restored == before, "and undo restores byte-identical heights");

            // The ray hit is a heightfield march, not a triangle raycast: a ray
            // straight down over the middle has to land ON the surface there.
            Vector3 hit;
            const bool landed = TerrainSculpt::raycast(before, lattice, {0.f, 500.f, 0.f},
                                                       {0.f, -1.f, 0.f}, 2000.f, hit);
            check(landed, "a ray marched down onto the patch finds the surface");
            check(landed && std::abs(hit.y - before[inside]) < lattice.cellSize(),
                  "at the height the lattice says is there");
        }

        // Delta preservation: sculpt a bump by hand, move a noise parameter, and
        // the bump has to still be there — the user does not lose an hour of
        // sculpting because they touched a slider.
        {
            auto heights = TerrainConfig::heightsOf(*created->geometry());
            const int dim = config.dim();
            const size_t peak = static_cast<size_t>(dim / 2) * dim + dim / 2;
            const float baseHeight = heights[peak];
            constexpr float kBump = 7.5f;
            heights[peak] += kBump;
            TerrainConfig::setHeights(*created->geometry(), heights);

            auto after = config;
            after.params.warp = config.params.warp + 0.2f;
            TerrainConfig::rebuild(*created, config, after);

            const auto rebuilt = TerrainConfig::heightsOf(*created->geometry());
            const auto plain = after.bake();
            check(std::abs((rebuilt[peak] - plain.heights[peak]) - kBump) < 1e-3f,
                  "a sculpted bump survives a parameter change intact");
            check(std::abs(rebuilt[peak] - baseHeight - kBump) > 1e-4f ||
                          std::abs(plain.heights[peak] - baseHeight) < 1e-6f,
                  "and the parameter change itself still moved the ground under it");

            // Resolution change resamples the delta rather than dropping it.
            auto coarser = after;
            coarser.params.resolution = after.params.resolution / 2;
            TerrainConfig::rebuild(*created, after, coarser);
            const auto resampled = TerrainConfig::heightsOf(*created->geometry());
            const auto coarseBase = coarser.bake();
            const int cdim = coarser.dim();
            const size_t cpeak = static_cast<size_t>(cdim / 2) * cdim + cdim / 2;
            check(resampled.size() == coarseBase.heights.size(),
                  "a resolution change relattices the mesh");
            check(resampled[cpeak] - coarseBase.heights[cpeak] > kBump * 0.4f,
                  "and the bump survives the resample");

            // Back to what the document should carry for the round trip.
            TerrainConfig::rebuild(*created, coarser, config);
        }

        // Save and reload: the triangles are the truth, so they have to come
        // back byte-identical — a terrain that regenerated on open would drop
        // every sculpt in the file.
        {
            const auto terrainPath = std::filesystem::temp_directory_path() /
                                     "threepp-editor-selftest-terrain.json";
            auto sculpted = TerrainConfig::heightsOf(*created->geometry());
            sculpted[100] += 3.25f;
            TerrainConfig::setHeights(*created->geometry(), sculpted);

            saveSceneAs(terrainPath);
            openScene(terrainPath);
            step();

            auto* reloaded = terrainNow(terrainUuid);
            check(reloaded && TerrainConfig::isTerrain(*reloaded),
                  "the terrain survives save and reload");
            check(reloaded && TerrainConfig::read(*reloaded) == config,
                  "its config round-trips through the document");
            const auto* asMesh = reloaded ? reloaded->as<Mesh>() : nullptr;
            check(asMesh && asMesh->geometry() &&
                          TerrainConfig::heightsOf(*asMesh->geometry()) == sculpted,
                  "and its heights come back byte-identical, sculpt and all");
            check(reloaded && TerrainConfig::albedoTexture(*reloaded) != nullptr,
                  "with the baked albedo back on the material");

            // A picture of what the PASS lines cannot describe: whether Add
            // Terrain gives you usable GROUND, and whether a brush stroke on it
            // reads as a mound. The heights are all right in arithmetic long
            // before they are right to look at.
            if (auto* mesh = reloaded ? reloaded->as<Mesh>() : nullptr) {
                auto heights = TerrainConfig::heightsOf(*mesh->geometry());
                const int dim = config.dim();
                // The round-trip check above spiked ONE vertex to prove the
                // document carries sculpts; leave it in and the picture has a
                // needle through it. Put it back before drawing something meant
                // to be looked at.
                heights[100] -= 3.25f;

                // A real stroke, through the real kernels — not a hand-poked
                // dome. This is the brush the plan is about, so it is the brush
                // the acceptance picture has to show.
                const auto lattice = TerrainLattice::of(*mesh->geometry(), dim);
                TerrainBrush brush;
                brush.kind = TerrainBrush::Kind::Raise;
                brush.radius = 9.f;
                brush.strength = 26.f;
                TerrainSculpt::Rect stroke;
                // Dragged along a short arc, the way a hand would: a single
                // stamp is a cone, a stroke is a landform.
                for (int i = 0; i <= 10; ++i) {
                    const float t = static_cast<float>(i) / 10.f;
                    const float sx = -14.f + 26.f * t;
                    const float sz = 6.f - 12.f * t * t;
                    const auto touched =
                            TerrainSculpt::apply(heights, lattice, brush, sx, sz, 0.1f, 0.f);
                    if (touched.empty()) continue;
                    stroke.add(touched.x0, touched.z0);
                    stroke.add(touched.x1, touched.z1);
                }
                // Then the smooth brush over the same ground, so the picture
                // shows both that a stroke piles material up and that the second
                // brush settles it.
                brush.kind = TerrainBrush::Kind::Smooth;
                brush.radius = 14.f;
                brush.strength = 6.f;
                for (int i = 0; i < 4; ++i) {
                    TerrainSculpt::apply(heights, lattice, brush, 0.f, 0.f, 0.1f, 0.f);
                }
                stroke.grow(2, dim);
                TerrainSculpt::refresh(*mesh->geometry(), heights, lattice, stroke);
                mesh->geometry()->computeBoundingBox();
                mesh->geometry()->computeBoundingSphere();
                // Stroke release: the splat follows the surface it now has.
                {
                    const auto field = config.fieldOf(*mesh->geometry(), heights);
                    TerrainConfig::applyAlbedo(*reloaded, config.bakeAlbedo(field), dim);
                }

                const bool bottomPanelWas = bottomPanelOpen_;
                bottomPanelOpen_ = false;
                selectObject(reloaded);

                // Arm the tool so the palette shows its cell lit and the brush
                // ring is in the picture: a brush with no cursor is a guess, and
                // the ring is the half of the feature a PASS line cannot check.
                const bool sculptWas = sculptTool_;
                sculptTool_ = true;
                brush_.radius = 11.f;
                sculptHover_ = true;
                sculptHoverLocal_.set(6.f, 0.f, -4.f);
                TerrainSculpt::sample(heights, lattice, sculptHoverLocal_.x,
                                      sculptHoverLocal_.z, sculptHoverLocal_.y);
                syncBrushRing();
                check(brushRing_ && brushRing_->visible,
                      "the brush ring shows while the sculpt tool is armed over a terrain");

                camera_.position.set(21.f, 13.f, 27.f);
                orbit_->target.set(-1.f, 1.5f, -1.f);
                step(4);
                shootTo(std::filesystem::temp_directory_path() / "threepp_editor_terrain.png");

                // Selection off the terrain drops the tool back rather than
                // leaving a brush armed over nothing.
                selectObject(nullptr);
                updateSculpt();
                check(!sculptTool_, "and the tool falls back when the selection leaves the terrain");
                check(!brushRing_->visible, "taking its ring with it");

                sculptTool_ = sculptWas;
                bottomPanelOpen_ = bottomPanelWas;
                step();
            }
        }

#ifdef THREEPP_EDITOR_WITH_PHYSX
        // The collider tracks the SCULPT. Static + Shape::Auto cooks a trimesh
        // from the geometry as it stands, and the whole "the mesh is the truth"
        // claim rests on that being the sculpted geometry rather than whatever
        // the config would regenerate. Nothing here is terrain-specific code —
        // that is the point, and it is why it has to be checked rather than
        // assumed.
        {
            auto* terrain = findByUuid(document_.scene(), terrainUuid);
            auto* mesh = terrain ? terrain->as<Mesh>() : nullptr;
            if (mesh && mesh->geometry()) {
                const auto lattice = TerrainLattice::of(*mesh->geometry(), config.dim());
                auto heights = TerrainConfig::heightsOf(*mesh->geometry());

                // A broad raised PLATEAU well clear of the ridge the shot
                // carved, built through the real kernels: raise wide, then
                // flatten the crown level. A cone would be a bad instrument —
                // a sphere dropped on a 45 degree peak rolls off it over four
                // seconds of sim and the test would be measuring gravity, not
                // the collider. Level ground gives the probe somewhere to rest.
                // ~8 m of lift dwarfs the +-2.6 m of base undulation, so the
                // differential below cannot be base noise.
                TerrainBrush brush;
                brush.kind = TerrainBrush::Kind::Raise;
                brush.radius = 26.f;
                brush.strength = 20.f;
                TerrainSculpt::Rect rect;
                const auto note = [&rect](const TerrainSculpt::Rect& touched) {
                    if (touched.empty()) return;
                    rect.add(touched.x0, touched.z0);
                    rect.add(touched.x1, touched.z1);
                };
                for (int i = 0; i < 4; ++i) {
                    note(TerrainSculpt::apply(heights, lattice, brush, 22.f, 22.f, 0.1f, 0.f));
                }
                float crown = 0.f;
                TerrainSculpt::sample(heights, lattice, 22.f, 22.f, crown);
                brush.kind = TerrainBrush::Kind::Flatten;
                brush.radius = 14.f;
                brush.strength = 9.f;
                for (int i = 0; i < 8; ++i) {
                    note(TerrainSculpt::apply(heights, lattice, brush, 22.f, 22.f, 0.1f, crown));
                }
                TerrainSculpt::refresh(*mesh->geometry(), heights, lattice, rect);
                mesh->geometry()->computeBoundingBox();
                mesh->geometry()->computeBoundingSphere();

                PhysicsConfig ground;
                ground.enabled = true;
                ground.body = PhysicsConfig::Body::Static;
                ground.shape = PhysicsConfig::Shape::Auto;
                ground.write(*terrain);

                // Two spheres, same drop height: one over the mound, one over
                // ground the brush never touched. A differential, not an
                // absolute, so it cannot be satisfied by the terrain's own
                // transform offset or by the base noise.
                const auto dropSphere = [&](const char* name, float x, float z) {
                    auto sphere = ObjectFactory::createPrimitive(Primitive::Sphere,
                                                                 document_.scene());
                    sphere->name = name;
                    sphere->position.set(x, 30.f, z);
                    PhysicsConfig body;
                    body.enabled = true;
                    body.body = PhysicsConfig::Body::Dynamic;
                    body.shape = PhysicsConfig::Shape::Sphere;
                    body.mass = 1.f;
                    body.restitution = 0.f;
                    body.write(*sphere);
                    const auto uuid = sphere->uuid;
                    addObject(sphere, document_.scene(), "Add Drop Probe");
                    return uuid;
                };
                const auto moundUuid = dropSphere("Mound Probe", 22.f, 22.f);
                const auto flatUuid = dropSphere("Flat Probe", -22.f, -22.f);

                startPlay();
                // Fixed dt: a resting height is a property of the sim, not of
                // the frame rate.
                stepFixed(240);

                auto* moundProbe = findByUuid(document_.scene(), moundUuid);
                auto* flatProbe = findByUuid(document_.scene(), flatUuid);
                // Stronger than the differential: the probe has to rest where
                // the LATTICE says the sculpted surface is, to within a
                // fraction of a cell. That pins the cook to the actual heights
                // rather than to some hull or slab that merely happens to be
                // taller over there.
                float crownHeight = 0.f;
                TerrainSculpt::sample(heights, lattice, 22.f, 22.f, crownHeight);
                const float expected = crownHeight + terrain->position.y + 0.5f;// + probe radius
                check(moundProbe && std::abs(moundProbe->position.y - expected) < 0.1f,
                      "the probe rests exactly where the sculpted lattice says the ground is");
                check(moundProbe && flatProbe && moundProbe->position.y > -5.f &&
                              flatProbe->position.y > -5.f,
                      "both probes land on the terrain rather than falling through it");
                check(moundProbe && flatProbe &&
                              moundProbe->position.y > flatProbe->position.y + 6.f,
                      "and the one over a sculpted mound rests higher - the collider "
                      "cooks from the sculpted geometry");
                stopPlay();
                step();

                // Leave the scene as the passes that follow expect it.
                for (const auto& uuid : {moundUuid, flatUuid}) {
                    if (auto* done = findByUuid(document_.scene(), uuid)) {
                        selectObject(done);
                        deleteSelected();
                    }
                }
                if (auto* terrainNow = findByUuid(document_.scene(), terrainUuid)) {
                    PhysicsConfig::erase(*terrainNow);
                }
                selectObject(nullptr);
                step();
            }
        }
#endif
    }

    // --- text objects ------------------------------------------------------
    // A text mesh is an ordinary Mesh whose geometry is built from its
    // userData (TextConfig): created by the factory, edited through the same
    // property command the inspector's Text section issues, and rebuilt on
    // undo exactly as on execute.
    if (section("text")) {
        auto text = ObjectFactory::createText(document_.scene());
        check(TextConfig::isText(*text), "the factory's text mesh carries the text entry");
        addObject(text, document_.scene(), "Add Text");
        step(2);

        const auto vertexCount = [](const Object3D& mesh) {
            const auto* position = mesh.geometry()
                                           ? mesh.geometry()->getAttribute<float>("position")
                                           : nullptr;
            return position ? position->count() : 0;
        };
        const int defaultVertices = vertexCount(*text);
        check(defaultVertices > 0, "and its geometry has triangles to show");

        // The inspector's edit, verbatim: apply() through a property command.
        const auto before = TextConfig::read(*text).value_or(TextConfig{});
        auto after = before;
        after.text = "Hi";
        after.depth = 0.f;
        auto* target = text.get();
        commands_.execute(makeProperty<TextConfig>(
                "Edit Text", "text:" + text->uuid,
                [target](const TextConfig& value) { value.apply(*target); },
                before, after));

        check(TextConfig::read(*text)->text == "Hi", "an edit rewrites the entry");
        const int editedVertices = vertexCount(*text);
        check(editedVertices > 0 && editedVertices != defaultVertices,
              "and rebuilds the geometry from it");

        commands_.undo();
        check(TextConfig::read(*text)->text == before.text,
              "undo restores the config");
        check(vertexCount(*text) == defaultVertices,
              "and rebuilds the geometry it described");

        // Empty content is a document state, not an error: the mesh stays,
        // draws nothing, and the guard keeps the centring math off an empty
        // bounding box.
        auto emptied = before;
        emptied.text = "";
        emptied.apply(*text);
        check(vertexCount(*text) == 0, "empty text builds an empty geometry");

        // The reported defect, pinned: the selection outline and the raycast
        // both read the geometry's CACHED bounds, and applyMatrix4 used to
        // leave them where the glyphs stood before the anchoring translate —
        // a box beside the text, and a text nobody could click.
        auto anchored = before;
        anchored.text = "Aim";
        const auto built = anchored.buildGeometry();
        {
            const auto cached = *built->boundingBox;
            built->computeBoundingBox();
            const auto& fresh = *built->boundingBox;
            check(cached.min().distanceTo(fresh.min()) < 1e-5f &&
                          cached.max().distanceTo(fresh.max()) < 1e-5f,
                  "the built geometry's cached bounds sit on the glyphs");
        }
        check(std::abs(built->boundingBox->min().x + built->boundingBox->max().x) < 1e-3f,
              "Center anchors the origin mid-block");
        anchored.align = TextConfig::Align::Left;
        check(std::abs(anchored.buildGeometry()->boundingBox->min().x) < 1e-4f,
              "Left anchors it on the left edge");
        anchored.align = TextConfig::Align::Right;
        check(std::abs(anchored.buildGeometry()->boundingBox->max().x) < 1e-4f,
              "Right anchors it on the right edge");

        commands_.execute(std::make_unique<RemoveObjectCommand>(*text));
        step();
    }

    // --- procedural trees ---------------------------------------------------
    // A tree is a Group carrying TreeConfig with its trunk and foliage
    // generated under it. What is pinned here is the split the design rests
    // on: the config is the authored state and the two meshes are derived, so
    // an edit regrows them, undo regrows them back, a deleted half returns —
    // and a document that already carries them is ADOPTED rather than regrown,
    // which is what keeps opening a forest from costing seconds.
    if (section("trees")) {
        const auto treeNow = [&](const std::string& uuid) {
            return findByUuid(document_.scene(), uuid);
        };
        const auto partGeometry = [](const Object3D* tree, TreeConfig::Part part) {
            auto* node = tree ? TreeConfig::derivedPart(*tree, part) : nullptr;
            return node ? node->geometry() : nullptr;
        };
        const auto partVertices = [&](const Object3D* tree, TreeConfig::Part part) {
            const auto geometry = partGeometry(tree, part);
            const auto* position = geometry ? geometry->getAttribute<float>("position") : nullptr;
            return position ? position->count() : 0;
        };
        // How tall the trunk actually came out — a vertex count can collide
        // between two different trees, a silhouette cannot.
        const auto trunkTop = [&](const Object3D* tree) {
            const auto geometry = partGeometry(tree, TreeConfig::Part::Trunk);
            if (!geometry) return 0.f;
            geometry->computeBoundingBox();
            return geometry->boundingBox->max().y;
        };
        const auto leafMap = [](const Object3D* tree) {
            auto* node = tree ? TreeConfig::derivedPart(*tree, TreeConfig::Part::Leaves) : nullptr;
            auto* material = node ? node->materialAs<MeshStandardMaterial>() : nullptr;
            return material ? material->map : nullptr;
        };
        const auto derivedChildren = [](const Object3D& tree) {
            int found = 0;
            for (const auto* child : tree.children) {
                if (TreeConfig::isDerived(*child)) ++found;
            }
            return found;
        };

        auto created = ObjectFactory::createTree(document_.scene());
        const auto treeUuid = created->uuid;
        check(TreeConfig::isTree(*created), "the factory's tree carries the tree entry");
        addObject(created, document_.scene(), "Add Tree");
        created.reset();// the command owns it now; nothing here may outlive a replace
        step(2);

        auto* tree = treeNow(treeUuid);
        check(tree && derivedChildren(*tree) == 2, "with a trunk and a foliage mesh under it");
        check(partVertices(tree, TreeConfig::Part::Trunk) > 0 &&
                      partVertices(tree, TreeConfig::Part::Leaves) > 0,
              "both of which have triangles to show");
        check(leafMap(tree) != nullptr, "and the foliage carries its procedural atlas");

        // Adoption. The factory already grew these from the config the group
        // carries, so the sync has to leave them alone — a regrow would swap
        // the geometry object, which is exactly what this compares.
        const auto adoptedTrunk = partGeometry(tree, TreeConfig::Part::Trunk);
        step(3);
        check(partGeometry(treeNow(treeUuid), TreeConfig::Part::Trunk) == adoptedTrunk,
              "a tree the document already carries is adopted, not regrown every frame");

        // The inspector's edit, verbatim: the property command writes the
        // CONFIG only, and the sync pass is what turns that into geometry.
        const auto before = TreeConfig::read(*tree).value_or(TreeConfig{});
        const float topBefore = trunkTop(tree);
        {
            auto after = before;
            after.params.trunkHeight = before.params.trunkHeight * 2.f;
            auto* target = tree;
            commands_.execute(makeProperty<TreeConfig>(
                    "Tree Height", "tree:" + tree->uuid,
                    [target](const TreeConfig& value) { value.write(*target); },
                    before, after));
        }
        step(2);
        tree = treeNow(treeUuid);
        check(tree && TreeConfig::read(*tree)->params.trunkHeight == before.params.trunkHeight * 2.f,
              "an edit rewrites the entry");
        check(partGeometry(tree, TreeConfig::Part::Trunk) != adoptedTrunk,
              "and the sync regrows the trunk from it");
        check(trunkTop(tree) > topBefore + 1e-3f, "into the taller tree the config now describes");

        commands_.undo();
        step(2);
        tree = treeNow(treeUuid);
        check(tree && TreeConfig::read(*tree) == before, "undo restores the config");
        check(std::abs(trunkTop(tree) - topBefore) < 1e-3f,
              "and the sync regrows the tree it described");

        // Half a tree is not a document state: the config is the source of
        // truth, so a derived mesh the user deleted comes back.
        if (auto* leaves = TreeConfig::derivedPart(*tree, TreeConfig::Part::Leaves)) {
            leaves->removeFromParent();
        }
        step(2);
        tree = treeNow(treeUuid);
        check(tree && derivedChildren(*tree) == 2, "a deleted half regrows on the next sync");

        // The atlases run on their own, slower clock (they cost 30-80 ms where
        // the geometry costs 4), so a colour change has to be seen to reach
        // them and not just the mesh.
        const auto mapBefore = leafMap(tree);
        {
            const auto now = TreeConfig::read(*tree).value_or(TreeConfig{});
            auto recolored = now;
            recolored.params.leafColor = {0.82f, 0.34f, 0.08f};
            auto* target = tree;
            commands_.execute(makeProperty<TreeConfig>(
                    "Tree Leaf Color", "tree:" + tree->uuid,
                    [target](const TreeConfig& value) { value.write(*target); },
                    now, recolored));
        }
        step(2);
        tree = treeNow(treeUuid);
        check(mapBefore && leafMap(tree) && leafMap(tree) != mapBefore,
              "a leaf colour change repaints the foliage atlas");

        // Save and reload. Everything a tree is — the config, both meshes and
        // the procedural atlases — is ordinary document content, which is the
        // claim that a saved scene shows a tree with no editor and no
        // generator present.
        const auto treePath = std::filesystem::temp_directory_path() /
                              "threepp-editor-selftest-tree.json";
        auto* trunkNode = tree ? TreeConfig::derivedPart(*tree, TreeConfig::Part::Trunk) : nullptr;
        const auto trunkUuid = trunkNode ? trunkNode->uuid : std::string{};
        // A hand-edit on the derived node, so the round trip proves what a
        // user put there survives the regeneration too.
        if (auto* material = trunkNode ? trunkNode->materialAs<MeshStandardMaterial>() : nullptr) {
            material->color.setHex(0x3366aa);
        }
        const auto authored = TreeConfig::read(*tree).value_or(TreeConfig{});
        const float authoredTop = trunkTop(tree);

        saveSceneAs(treePath);
        openScene(treePath);
        step(2);

        auto* reloaded = treeNow(treeUuid);
        check(reloaded != nullptr, "the tree survives save and reload");
        check(reloaded && TreeConfig::read(*reloaded) == authored,
              "its config round-trips through the document");
        check(reloaded && derivedChildren(*reloaded) == 2,
              "with both halves adopted, not duplicated");
        auto* reloadedTrunk = reloaded ? TreeConfig::derivedPart(*reloaded, TreeConfig::Part::Trunk)
                                       : nullptr;
        check(reloadedTrunk && reloadedTrunk->uuid == trunkUuid,
              "the same trunk node, by uuid, as the one that was saved");
        check(reloadedTrunk && reloadedTrunk->materialAs<MeshStandardMaterial>() &&
                      reloadedTrunk->materialAs<MeshStandardMaterial>()->color.getHex() == 0x3366aa,
              "carrying the material the user edited");
        check(reloaded && std::abs(trunkTop(reloaded) - authoredTop) < 1e-3f,
              "and the geometry it was saved with rather than a freshly grown one");
        // Without this the reloaded foliage would be an untextured green slab,
        // and nothing short of an edit would bring the atlas back.
        check(leafMap(reloaded) != nullptr,
              "the procedural leaf atlas round-trips with the material");

        if (auto* live = treeNow(treeUuid)) {
            commands_.execute(std::make_unique<RemoveObjectCommand>(*live));
        }
        step();
        check(treeNow(treeUuid) == nullptr, "and the tree can be removed again");
        check(treeOverlays_.empty(), "retiring its sync record with it");
    }

    // --- sound -------------------------------------------------------------
    // The whole authoring path for a sound node: Add ▸ Sound, the marker and
    // the distance rings it earns, an edit through the same property command
    // the inspector issues, a document round trip, and then Play — once with a
    // file that is NOT there (which must not stop the session) and once with a
    // real one (which must actually be playing).
    if (section("sound")) {
        newScene();
        step(2);

        auto sound = ObjectFactory::createSound(document_.scene());
        const auto soundUuid = sound->uuid;
        check(SoundConfig::read(*sound).has_value(), "the factory's sound node carries the sound entry");
        check(SoundConfig::read(*sound) == SoundConfig{}, "at the documented defaults");

        addObject(sound, document_.scene(), "Add Sound");
        step(2);
        check(inScene(sound.get()), "Add Sound puts it in the scene");

        selectObject(sound.get());
        step();
        check(std::any_of(viewportMarkers_.begin(), viewportMarkers_.end(),
                          [&](const ViewportMarker& marker) { return marker.owner == sound.get(); }),
              "an authored sound gets a viewport marker");
        check(soundRings_ != nullptr && soundRings_->visible,
              "and a selected positional sound draws its distance rings");

        commands_.undo();
        step();
        check(!inScene(sound.get()), "undo removes it");
        commands_.redo();
        step();
        check(inScene(sound.get()), "redo puts it back");
        // Redo re-parents the SAME object, so the shared_ptr above is still the
        // node in the scene — nothing to re-resolve until the document reloads.

        const auto soundDir = std::filesystem::temp_directory_path() / "threepp-editor-selftest-sound";
        std::error_code ec;
        std::filesystem::create_directories(soundDir, ec);
        const auto wav = soundDir / "tone.wav";
        const bool haveWav = writeTestWav(wav);
        check(haveWav, "a decodable test WAV was written");

        // The inspector's edit, verbatim: one SoundAuthoring property command
        // carrying both userData entries.
        SoundConfig authored;
        authored.positional = true;
        authored.autoplay = true;
        authored.loop = false;
        authored.volume = 0.4f;
        authored.rate = 1.25f;
        authored.minDistance = 2.5f;
        authored.maxDistance = 12.f;
        authored.rolloff = 1.75f;
        authored.model = SoundConfig::DistanceModel::Linear;

        {
            auto* target = sound.get();
            const auto before = SoundAuthoring::read(*sound);
            commands_.execute(makeProperty<SoundAuthoring>(
                    "Edit Sound", "sound:" + soundUuid,
                    [target](const SoundAuthoring& value) { value.write(*target); },
                    before, SoundAuthoring{authored, wav.generic_string()}));
            step();
        }
        check(SoundConfig::read(*sound) == authored, "an edit rewrites the entry");
        check(SoundConfig::file(*sound) == wav.generic_string(), "and records the file beside it");

        const auto document = std::filesystem::temp_directory_path() / "threepp_editor_sound.json";
        saveSceneAs(document);
        openScene(document);
        step();

        // Everything past the reload has to go through the uuid: openScene
        // replaced the scene, and `sound` names a node in the outgoing one.
        auto* reloaded = findByUuid(document_.scene(), soundUuid);
        check(reloaded != nullptr, "the sound survives save and reload");
        check(reloaded && SoundConfig::read(*reloaded) == authored,
              "its config round-trips through the document");
        check(reloaded && SoundConfig::file(*reloaded) == wav.generic_string(),
              "and so does the soundFile key");

        // The audition, which Play never goes near: its own listener, its own
        // Audio, and a uuid rather than a pointer holding the two together.
        if (haveWav && reloaded) {
            startAudition(*reloaded);
#ifdef THREEPP_WITH_AUDIO
            // On a machine with no device the button disables itself, and so
            // does this: the failure is already logged once.
            if (!auditionUnavailable_) {
                check(isAuditioning(*reloaded), "Audition starts on the authored file");
            }
#endif
            stopAudition();
            check(!isAuditioning(*reloaded), "and the audition stops");
            // Left RUNNING on purpose: the Play below has to take it down. The
            // global "not while playing" gate does not cover the audition — it
            // is not a document mutation — so nothing but startPlay() would.
            startAudition(*reloaded);
        }

        // A picture of what the PASS lines above cannot describe: the speaker
        // marker, the populated Sound section, and the min/max rings on the
        // ground. The bottom panel steps aside the way the texture-settings
        // shot does, so the section is not below the fold.
        {
            const bool bottomPanelWas = bottomPanelOpen_;
            const auto gizmoWas = gizmoMode_;
            bottomPanelOpen_ = false;
            // Select mode, and off the template's Box: the marker is the point
            // of the shot and both would sit on top of it.
            gizmoMode_ = "select";
            applyGizmoMode();
            if (reloaded) reloaded->position.set(2.4f, 1.4f, 1.2f);
            selectObject(reloaded);
            // Far enough back that the 12 m max ring is in frame whole; the
            // marker keeps a constant screen size, so pulling back costs it
            // nothing.
            camera_.position.set(8.f, 12.f, 22.f);
            orbit_->target.set(2.4f, 0.5f, 1.2f);
            step(4);
            shootTo(std::filesystem::temp_directory_path() / "threepp_editor_sound.png");
            bottomPanelOpen_ = bottomPanelWas;
            gizmoMode_ = gizmoWas;
            applyGizmoMode();
            step();
        }

        // A file that is not there is an authoring mistake, not a crash and not
        // a refused Play. It has to reach the console, though — silence here is
        // the failure everyone spends an afternoon on.
        {
            const auto missing = soundDir / "not-here.wav";
            std::filesystem::remove(missing, ec);
            if (auto* node = findByUuid(document_.scene(), soundUuid)) {
                SoundConfig::setFile(*node, missing.generic_string());
            }
            const auto logged = console_.size();
            startPlay();
            step(3);
            check(isPlaying(), "Play starts even though the sound file is missing");
            check(auditionUuid_.empty(), "and entering Play took the audition down");
            bool reported = false;
            for (std::size_t i = logged; i < console_.size(); ++i) {
                if (console_[i].find("not-here.wav") != std::string::npos) reported = true;
            }
            check(reported, "and the failure reaches the console");
            stopPlay();
            step(2);
        }

        // And with a file that decodes. Skipped, not failed, on a machine with
        // no audio device — but this one has audio, so a SKIP here is news.
        if (haveWav) {
            if (auto* node = findByUuid(document_.scene(), soundUuid)) {
                SoundConfig::setFile(*node, wav.generic_string());
            }
            startPlay();
            step(3);
            check(isPlaying(), "Play starts with a real sound file");
#ifdef THREEPP_WITH_AUDIO
            // The audibility bound is a pure curve — pin it here, no device
            // needed: authored volume inside, silence at and past max, easing
            // through the band just before it.
            check(AudioPlaySession::distanceGate(5.f, 12.f) == 1.f,
                  "the distance gate is unity well inside max");
            check(AudioPlaySession::distanceGate(12.f, 12.f) == 0.f, "zero at max");
            check(AudioPlaySession::distanceGate(40.f, 12.f) == 0.f, "and zero beyond it");
            const float inBand = AudioPlaySession::distanceGate(11.5f, 12.f);
            check(inBand > 0.f && inBand < 1.f, "and eases through the band before max");
            if (audio_ && audio_->listenerReady()) {
                check(audio_->soundCount() == 1, "the session built one sound from it");
                check(audio_->isPlaying(soundUuid), "and it is playing");
            } else {
                std::cout << "[selftest] SKIP no audio device - the audible sound block "
                             "did not run" << std::endl;
            }
#else
            std::cout << "[selftest] SKIP built without audio - the sound play block "
                         "did not run" << std::endl;
#endif
            stopPlay();
            step(2);
        }

        std::filesystem::remove(document, ec);
        std::filesystem::remove_all(soundDir, ec);
    }

    // --- acoustics ----------------------------------------------------------
    // A wall between the listener and a sound. The config is authored on the
    // MESH, undoes to nothing at all (an unflagged mesh leaves no entry),
    // round-trips through the document like every sibling, and at Play turns
    // the same audio session's positional sound into a ray-traced one.
    if (section("acoustics")) {
        newScene();
        step(2);

        auto wall = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        wall->name = "Wall";
        wall->position.set(0.f, 1.f, 0.f);
        wall->scale.set(8.f, 8.f, 0.2f);
        const auto wallUuid = wall->uuid;
        check(!AcousticSurfaceConfig::read(*wall).has_value(),
              "a fresh mesh carries no acoustic entry");

        addObject(wall, document_.scene(), "Add Wall");
        step(2);

        AcousticSurfaceConfig authored;
        authored.enabled = true;
        authored.transmission = 0.15f;
        authored.absorption = 0.08f;

        // The inspector's edit, verbatim.
        {
            auto* target = wall.get();
            const auto before = AcousticSurfaceConfig::read(*wall).value_or(AcousticSurfaceConfig{});
            commands_.execute(makeProperty<AcousticSurfaceConfig>(
                    "Enable Acoustic Surface", "acoustics:" + wallUuid,
                    [target](const AcousticSurfaceConfig& value) { value.write(*target); },
                    before, authored));
            step();
        }
        check(AcousticSurfaceConfig::read(*wall) == authored, "the edit writes the acoustic entry");

        commands_.undo();
        step();
        check(!AcousticSurfaceConfig::read(*wall).has_value(),
              "undo leaves the mesh with no entry at all, not a disabled one");
        commands_.redo();
        step();

        const auto document = std::filesystem::temp_directory_path() / "threepp_editor_acoustics.json";
        saveSceneAs(document);
        openScene(document);
        step();

        auto* reloaded = findByUuid(document_.scene(), wallUuid);
        check(reloaded != nullptr, "the wall survives save and reload");
        check(reloaded && AcousticSurfaceConfig::read(*reloaded) == authored,
              "and its acoustic surface round-trips through the document");

        // Now put a sound behind it and press Play. The listener rides the
        // perspective camera, so the three positions are what decide whether
        // the wall is in the way.
        const auto acousticDir = std::filesystem::temp_directory_path() / "threepp-editor-selftest-acoustics";
        std::error_code ec2;
        std::filesystem::create_directories(acousticDir, ec2);
        const auto wav = acousticDir / "tone.wav";
        const bool haveWav = writeTestWav(wav);

        if (haveWav) {
            auto source = ObjectFactory::createSound(document_.scene());
            source->name = "Behind the wall";
            source->position.set(0.f, 1.f, -6.f);
            SoundConfig::setFile(*source, wav.generic_string());
            const auto sourceUuid = source->uuid;
            addObject(source, document_.scene(), "Add Sound");
            step(2);

            camera_.position.set(0.f, 1.f, 8.f);
            orbit_->target.set(0.f, 1.f, 0.f);
            step();

            startPlay();
            // Fixed steps: the assertion below is on a smoothed value, i.e. on
            // how much SIMULATED time passed (tau is 60 ms).
            stepFixed(30);
            check(isPlaying(), "Play starts with an acoustic surface in the scene");
#ifdef THREEPP_WITH_AUDIO
            if (audio_ && audio_->listenerReady()) {
                check(audio_->acousticsActive(), "the audio session brought up acoustics");
                check(audio_->acousticSurfaceCount() == 1, "over the one flagged mesh");
                // transmission 0.15 through a closed box is 0.15 of the energy,
                // so the smoothed occlusion settles near 0.85.
                check(audio_->occlusionOf(sourceUuid) > 0.5f,
                      "and the wall muffles the sound behind it");
            } else {
                std::cout << "[selftest] SKIP no audio device - the acoustics play block "
                             "did not run" << std::endl;
            }
#else
            std::cout << "[selftest] SKIP built without audio - the acoustics play block "
                         "did not run" << std::endl;
#endif
            stopPlay();
            step(2);

            // The other half of the contract: an unflagged scene must not build
            // any of this, which is what keeps a document authored before
            // acoustics existed playing exactly as it did.
            if (auto* node = findByUuid(document_.scene(), wallUuid)) {
                AcousticSurfaceConfig::erase(*node);
            }
            startPlay();
            step(3);
#ifdef THREEPP_WITH_AUDIO
            check(audio_ && !audio_->acousticsActive(),
                  "and a scene with no flagged mesh builds no acoustics at all");
#endif
            stopPlay();
            step(2);
        }

        std::filesystem::remove(document, ec2);
        std::filesystem::remove_all(acousticDir, ec2);
    }

    // --- joint authoring ----------------------------------------------------
    // The whole authoring path for a joint node: Add ▸ Joint under a body, the
    // marker and the axis helper it earns, an edit through the same property
    // command the inspector issues, a document round trip, and Play — where
    // the authored hinge must actually constrain the two bodies it names.
    if (section("joints")) {
        newScene();
        step(2);

        // Two bodies to join: a static post and a dynamic gate beside it.
        auto post = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        post->name = "Post";
        post->position.set(0.f, 3.f, 0.f);
        {
            PhysicsConfig config;
            config.enabled = true;
            config.body = PhysicsConfig::Body::Static;
            config.shape = PhysicsConfig::Shape::Box;
            config.write(*post);
        }
        addObject(post, document_.scene(), "Add Post");

        auto gate = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        gate->name = "Gate";
        gate->position.set(1.f, 3.f, 0.f);
        {
            PhysicsConfig config;
            config.enabled = true;
            config.body = PhysicsConfig::Body::Dynamic;
            config.shape = PhysicsConfig::Shape::Box;
            config.write(*gate);
        }
        addObject(gate, document_.scene(), "Add Gate");
        step();

        auto joint = ObjectFactory::createJoint(document_.scene());
        const auto jointUuid = joint->uuid;
        check(JointConfig::isJoint(*joint), "the factory's joint node carries the joint entry");
        check(JointConfig::read(*joint) == JointConfig{}, "at the documented defaults");

        // Under the GATE — the parent chain is body A. Anchored at the post
        // (1 m towards it), hinging about world Z (the node's X rotated onto
        // it), which is the plane the play assertions below read.
        addObject(joint, *gate, "Add Joint");
        joint->position.set(-1.f, 0.f, 0.f);
        joint->rotation.y = -math::PI / 2;
        step();

        selectObject(joint.get());
        step();
        check(std::any_of(viewportMarkers_.begin(), viewportMarkers_.end(),
                          [&](const ViewportMarker& marker) { return marker.owner == joint.get(); }),
              "an authored joint gets a viewport marker");
        check(jointHelper_ != nullptr && jointHelper_->visible,
              "and a selected joint draws its axis helper");

        // The inspector's edit, verbatim: one JointConfig property command
        // carrying both userData entries (the flat string and the body name).
        JointConfig authored;
        authored.type = JointConfig::Type::Revolute;
        authored.body = "Post";
        authored.limited = true;
        // ±30°, written as the codec-exact 4-decimal radians: the flat format
        // stores 6 decimals, so a value that needs more (degToRad(30) does)
        // would fail the exact struct compares below by a quantization step.
        authored.lower = -0.5236f;
        authored.upper = 0.5236f;
        {
            auto* target = joint.get();
            const auto before = JointConfig::read(*joint).value_or(JointConfig{});
            commands_.execute(makeProperty<JointConfig>(
                    "Edit Joint", "joint:" + jointUuid + ":selftest",
                    [target](const JointConfig& value) { value.write(*target); },
                    before, authored));
            step();
        }
        check(JointConfig::read(*joint) == authored, "an edit rewrites the entry");

        const auto document = std::filesystem::temp_directory_path() / "threepp_editor_joint.json";
        saveSceneAs(document);
        openScene(document);
        step();

        // Everything past the reload goes through the uuid: openScene replaced
        // the scene, and `joint` names a node in the outgoing one.
        auto* reloaded = findByUuid(document_.scene(), jointUuid);
        check(reloaded != nullptr, "the joint survives save and reload");
        check(reloaded && JointConfig::read(*reloaded) == authored,
              "its config (body name included) round-trips through the document");

        // A picture of what the PASS lines cannot describe: the hinge marker,
        // the axis helper pointing down world Z, the populated Joint section.
        {
            const bool bottomPanelWas = bottomPanelOpen_;
            const auto gizmoWas = gizmoMode_;
            bottomPanelOpen_ = false;
            // Select mode: the transform gizmo would sit exactly on the
            // anchor the helper is the picture of.
            gizmoMode_ = "select";
            applyGizmoMode();
            selectObject(reloaded);
            camera_.position.set(4.f, 5.5f, 7.f);
            orbit_->target.set(0.5f, 3.f, 0.f);
            step(4);
            shootTo(std::filesystem::temp_directory_path() / "threepp_editor_joint.png");
            bottomPanelOpen_ = bottomPanelWas;
            gizmoMode_ = gizmoWas;
            applyGizmoMode();
            step();
        }

#ifdef THREEPP_EDITOR_WITH_PHYSX
        // Play: the hinge holds the gate on the post and the limit holds the
        // swing. stepFixed throughout — the lowest point of a swing is a
        // question about simulated time, and step() would make it a frame-rate
        // measurement (the doctrine written above kFixedDt).
        {
            startPlay();
            stepFixed(2);
            check(isPlaying(), "Play starts with an authored joint in the scene");
            check(physics_ && physics_->jointCount() == 1, "the session built the joint");

            auto* playedGate = document_.scene().getObjectByName("Gate");
            check(playedGate != nullptr, "the played scene still has the gate");

            // The debug view, mid-play: PhysX draws the LIVE joint (frame
            // triad at the anchor, the ±30° limit geometry) into the same
            // buffer as the colliders. A picture, because the lines' shapes
            // and colours are the payload and no counter reads a picture.
            physicsDebug_ = true;
            stepFixed(6);// a substep fills the render buffer
            check(physicsDebugLines_ != nullptr && physicsDebugLines_->visible &&
                          physicsDebugLines_->geometry()->drawRange.count > 0,
                  "the physics debug view draws lines with a joint in the scene");
            {
                const bool bottomPanelWas = bottomPanelOpen_;
                bottomPanelOpen_ = false;
                camera_.position.set(3.f, 4.5f, 5.f);
                orbit_->target.set(0.5f, 2.8f, 0.f);
                step(4);
                shootTo(std::filesystem::temp_directory_path() / "threepp_editor_joint_debug.png");
                bottomPanelOpen_ = bottomPanelWas;
            }
            physicsDebug_ = false;

            float lowestY = 1e30f;
            for (int i = 0; i < 180 && playedGate; ++i) {// 3 s
                stepFixed(1);
                lowestY = std::min(lowestY, playedGate->position.y);
            }
            // ±30° on the 1 m arm puts the bottom of the swing at
            // y = 3 - sin(30°) = 2.5; an unjointed gate free-falls to the
            // template floor and an unlimited hinge passes 2.1 at the bottom.
            check(lowestY > 2.3f && lowestY < 2.95f, "the limited hinge held the gate");
            stopPlay();
            step(2);
        }
#else
        std::cout << "[selftest] SKIP built without PhysX - the joint play block did not run"
                  << std::endl;
#endif

        std::error_code ec;
        std::filesystem::remove(document, ec);
    }

    // --- vehicle authoring --------------------------------------------------
    // The whole authoring path for a drivable car: a primitives car with four
    // named wheels, the marker and the wheel-ring helper it earns, a config
    // edit through the same property command the inspector issues, a document
    // round trip, and Play — where the teleop entry point must actually drive
    // it several metres.
    if (section("vehicle")) {
        newScene();
        step(2);

        // A road to drive on: the template floor draws but has no collider.
        auto road = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        road->name = "Road";
        road->scale.set(200.f, 1.f, 200.f);
        road->position.set(0.f, -0.5f, 0.f);
        {
            PhysicsConfig config;
            config.enabled = true;
            config.body = PhysicsConfig::Body::Static;
            config.shape = PhysicsConfig::Shape::Box;
            config.write(*road);
        }
        addObject(road, document_.scene(), "Add Road");

        // The car: box body over four named cylinder wheels, hubs at
        // (±0.8, 0.4, ±1.4), radius 0.4 — resting exactly on the road.
        auto car = Group::create();
        car->name = "Car";
        const auto carUuid = car->uuid;
        auto body = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        body->name = "CarBody";
        body->scale.set(1.6f, 0.8f, 4.2f);
        body->position.set(0.f, 1.f, 0.f);
        car->add(body);
        const char* wheelNames[4] = {"FR", "FL", "RR", "RL"};
        const Vector3 hubs[4] = {{0.8f, 0.4f, 1.4f},
                                 {-0.8f, 0.4f, 1.4f},
                                 {0.8f, 0.4f, -1.4f},
                                 {-0.8f, 0.4f, -1.4f}};
        for (int i = 0; i < 4; ++i) {
            auto wheel = ObjectFactory::createPrimitive(Primitive::Cylinder, document_.scene());
            wheel->name = wheelNames[i];
            wheel->scale.set(0.8f, 0.3f, 0.8f);// unit cylinder r=0.5 -> r=0.4, width 0.3
            wheel->rotation.z = math::PI / 2;  // cylinder height (Y) onto the axle (X)
            wheel->position.copy(hubs[i]);
            car->add(wheel);
        }
        {
            VehicleConfig config;
            config.wheels = {"FR", "FL", "RR", "RL"};
            config.write(*car);
        }
        addObject(car, document_.scene(), "Add Car");
        step();

        check(VehicleConfig::isVehicle(*car), "the car carries the vehicle entry");
        {
            const auto geo = VehicleConfig::read(*car)->derived(*car);
            check(geo.valid, "the geometry derives from the four picked wheels");
            check(geo.valid && std::abs(geo.wheelbase - 2.8f) < 0.02f &&
                          std::abs(geo.trackWidth - 1.6f) < 0.02f &&
                          std::abs(geo.wheelRadius - 0.4f) < 0.02f,
                  "and reads the authored wheelbase, track and radius off them");
        }

        selectObject(car.get());
        step();
        check(std::any_of(viewportMarkers_.begin(), viewportMarkers_.end(),
                          [&](const ViewportMarker& marker) { return marker.owner == car.get(); }),
              "an authored vehicle gets a viewport marker");
        check(vehicleHelper_ != nullptr && vehicleHelper_->visible,
              "and a selected vehicle draws its wheel rings");

        // The inspector's edit, verbatim: one VehicleConfig property command
        // carrying the flat string and the four wheel keys.
        VehicleConfig authored = VehicleConfig::read(*car).value_or(VehicleConfig{});
        authored.mass = 1200.f;
        authored.driven = VehicleConfig::Driven::Rear;
        {
            auto* target = car.get();
            const auto before = VehicleConfig::read(*car).value_or(VehicleConfig{});
            commands_.execute(makeProperty<VehicleConfig>(
                    "Edit Vehicle", "vehicle:" + carUuid + ":selftest",
                    [target](const VehicleConfig& value) { value.write(*target); },
                    before, authored));
            step();
        }
        check(VehicleConfig::read(*car) == authored, "an edit rewrites the entry");

        const auto document = std::filesystem::temp_directory_path() / "threepp_editor_vehicle.json";
        saveSceneAs(document);
        openScene(document);
        step();

        // Everything past the reload goes through the uuid: openScene replaced
        // the scene, and `car` names a node in the outgoing one.
        auto* reloaded = findByUuid(document_.scene(), carUuid);
        check(reloaded != nullptr, "the vehicle survives save and reload");
        check(reloaded && VehicleConfig::read(*reloaded) == authored,
              "its config (wheel names included) round-trips through the document");

        // A picture of what the PASS lines cannot describe: the wheel rings on
        // the wheels, the forward chevron, the populated Vehicle section.
        {
            const bool bottomPanelWas = bottomPanelOpen_;
            const auto gizmoWas = gizmoMode_;
            bottomPanelOpen_ = false;
            // Select mode: the transform gizmo would sit on the chassis centre
            // the chevron is the picture of.
            gizmoMode_ = "select";
            applyGizmoMode();
            selectObject(reloaded);
            camera_.position.set(6.f, 4.5f, 8.f);
            orbit_->target.set(0.f, 0.7f, 0.f);
            step(4);
            shootTo(std::filesystem::temp_directory_path() / "threepp_editor_vehicle.png");
            bottomPanelOpen_ = bottomPanelWas;
            gizmoMode_ = gizmoWas;
            applyGizmoMode();
            step();
        }

#ifdef THREEPP_EDITOR_WITH_PHYSX
        // Play: the car drives. stepFixed throughout — "how far did it get" is
        // a question about simulated time (the doctrine written above
        // kFixedDt), and the controls go through the same session entry point
        // the W key does.
        {
            startPlay();
            stepFixed(2);
            check(isPlaying(), "Play starts with an authored vehicle in the scene");
            check(physics_ && physics_->vehicleCount() == 1, "the session built the vehicle");

            auto* playedCar = document_.scene().getObjectByName("Car");
            check(playedCar != nullptr, "the played scene still has the car");

            float maxZ = 0.f;
            for (int i = 0; i < 240 && playedCar; ++i) {// 4 s of full throttle
                physics_->driveVehicles(true, false, 0.f, false, kFixedDt);
                stepFixed(1);
                maxZ = std::max(maxZ, playedCar->position.z);
            }
            check(maxZ > 3.f, "full throttle drove the car several metres");

            stopPlay();
            step(2);
            auto* restored = findByUuid(document_.scene(), carUuid);
            check(restored != nullptr && std::abs(restored->position.z) < 0.01f,
                  "and Stop puts the car back where it was authored");
        }
#else
        std::cout << "[selftest] SKIP built without PhysX - the vehicle play block did not run"
                  << std::endl;
#endif

        std::error_code ec;
        std::filesystem::remove(document, ec);
    }

    // --- a character WALKS, and plays the clip that matches how it moves ----
    //
    // The whole point of CharacterConfig is that nothing has to be typed: the
    // capsule is measured off the model and every clip's ROLE is read off its
    // own root motion. So this section builds a rig whose clips are known
    // quantities — a 1.4 m/s walk, a 4 m/s run, a backpedal, two strafes, an
    // idle, a jump, and one deliberately RAMPING "start walking" that must be
    // rejected — and then asserts the matcher put each one where it belongs
    // before ever pressing Play.
    //
    // With --character=PATH the same section runs against a real asset
    // instead (the xbot.glb the mixamo bake produces), which is the pass that
    // says the measurements survive a Mixamo rig's 0.01 armature scale and its
    // rotated bone frames. The --urdf pattern, for the same reason: the editor
    // configures standalone and must not learn where threepp_data lives.
    if (section("character")) {
        newScene();
        // The chase camera yields to Follow Selection, and an earlier section
        // may have left it on. Off, so this section exercises the chase.
        setFollowSelection(false);
        step(2);

        // Ground to walk on: the template floor draws but has no collider.
        auto ground = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
        ground->name = "Ground";
        ground->scale.set(200.f, 1.f, 200.f);
        ground->position.set(0.f, -0.5f, 0.f);
        {
            PhysicsConfig config;
            config.enabled = true;
            config.body = PhysicsConfig::Body::Static;
            config.shape = PhysicsConfig::Shape::Box;
            config.write(*ground);
        }
        addObject(ground, document_.scene(), "Add Ground");
        step();

        Object3D* hero = nullptr;
        std::string heroUuid;
        const bool realAsset = !options_.character.empty();

        if (realAsset) {
            importModel(options_.character);
            int budget = 12000;
            while ((activeImport_ || !importQueue_.empty()) && budget-- > 0) step();
            check(budget > 0 && !importFailed(), "the character model imported");
            hero = selection_.get();
            check(hero != nullptr, "and came in selected");
        } else {
            // --- the synthetic rig ------------------------------------------
            // A box body 1.8 m tall and 0.3 m deep over one "Hips" bone, which
            // is all the matcher needs: it measures the mesh for the capsule
            // and the bone's position tracks for the clips.
            auto rig = Group::create();
            rig->name = "Hero";
            auto body = ObjectFactory::createPrimitive(Primitive::Box, document_.scene());
            body->name = "HeroBody";
            body->scale.set(0.4f, 1.8f, 0.3f);
            body->position.set(0.f, 0.9f, 0.f);
            rig->add(body);
            auto hips = Bone::create();
            hips->name = "Hips";
            hips->position.set(0.f, 0.9f, 0.f);
            rig->add(hips);

            // One clip per role. `travel` is (x, z) metres over `duration`, so
            // the authored speed is |travel| / duration and the DIRECTION is
            // what the matcher classifies on: +Z forward, +X left.
            const auto clip = [](const std::string& name, float duration,
                                 float dx, float dz, bool ramp = false) {
                std::vector<float> times;
                std::vector<float> values;
                const int steps = ramp ? 4 : 1;
                for (int i = 0; i <= steps; ++i) {
                    const float u = static_cast<float>(i) / static_cast<float>(steps);
                    // A ramp starts from a standstill and ends at speed, which
                    // is what a "start walking" transition does — and what the
                    // early-vs-late test must throw out.
                    const float travelled = ramp ? u * u * u : u;
                    times.push_back(u * duration);
                    values.push_back(dx * travelled);
                    values.push_back(0.9f);
                    values.push_back(dz * travelled);
                }
                std::vector<std::shared_ptr<KeyframeTrack>> tracks{
                        std::make_shared<VectorKeyframeTrack>("Hips.position", times, values)};
                return std::make_shared<AnimationClip>(name, duration, tracks);
            };
            rig->animations = {
                    clip("idle", 4.f, 0.f, 0.f),
                    clip("walk", 1.f, 0.f, 1.4f),
                    clip("run", 0.5f, 0.f, 2.f),
                    clip("walkback", 1.f, 0.f, -1.1f),
                    clip("jogback", 0.5f, 0.f, -1.3f),
                    clip("strafeleft", 1.f, 1.2f, 0.f),
                    clip("strafeleftfast", 0.5f, 1.6f, 0.f),
                    clip("straferight", 1.f, -1.2f, 0.f),
                    clip("straferightfast", 0.5f, -1.6f, 0.f),
                    clip("jump", 1.f, 0.f, 0.f),
                    clip("start walking", 2.f, 0.f, 2.4f, true),
            };
            addObject(rig, document_.scene(), "Add Hero");
            hero = rig.get();
        }
        step();

        if (!hero) {
            check(false, "the character scene has a model to author");
        } else {
            heroUuid = hero->uuid;
            CharacterConfig{}.write(*hero);
            step();
            check(CharacterConfig::isCharacter(*hero), "the model carries the character entry");

            // --- what the matcher decided, before anything is simulated -----
            const auto config = CharacterConfig::read(*hero).value_or(CharacterConfig{});
            const auto geo = config.derived(*hero);
            check(geo.valid, "the character geometry derives from the model");
            check(geo.rootBone != nullptr, "and it found the rig's root bone");

            const auto named = [&](Gait gait) {
                const auto& slot = geo.slot(gait);
                return slot.clip ? slot.clip->name() : std::string();
            };
            const auto speedOf = [&](Gait gait) { return geo.slot(gait).speed; };

            if (realAsset) {
                // The Mixamo pack's own numbers, measured at bake time: walking
                // 1.67 m/s, running 4.38, walking backward 1.20, jog backward
                // 2.38, the strafes 1.73 slow and ~4.5 fast. Asserted as BANDS,
                // because what is being tested is that the 0.01 armature scale
                // and the rotated bone frames come out in metres per second —
                // not the animator's exact keyframes.
                // A rigged glTF character measures through its BONES, never
                // through the skinned mesh's own node — whose transform the
                // spec says to ignore, and which on a Mixamo rig carries the
                // armature's 0.01 unit scale. Get that wrong and a 1.8 m
                // person gets an 18 mm capsule that walks under the furniture.
                check(geo.height > 1.6f && geo.height < 2.f,
                      "the capsule is a person's height, not the armature's unit scale");
                check(geo.radius > 0.2f && geo.radius < 0.5f,
                      "with a body-sized radius rather than the T-pose's arm span");
                check(speedOf(Gait::Walk) > 1.f && speedOf(Gait::Walk) < 2.5f,
                      "the walk clip measures a walking pace in m/s");
                check(speedOf(Gait::Run) > speedOf(Gait::Walk) * 1.5f,
                      "and the run clip is clearly faster than the walk");
                check(speedOf(Gait::WalkBack) > 0.5f && speedOf(Gait::StrafeLeft) > 0.5f &&
                              speedOf(Gait::StrafeRight) > 0.5f,
                      "backward and both strafes were identified by direction");
                check(!named(Gait::Idle).empty() && named(Gait::Idle).find("idle") != std::string::npos,
                      "and the idle clip was found by name");
                std::cout << "[selftest] character clips:";
                for (std::size_t i = 0; i < kGaitCount; ++i) {
                    std::cout << ' ' << CharacterConfig::gaitLabels[i] << '='
                              << (geo.gaits[i].clip ? geo.gaits[i].clip->name() : "-");
                }
                std::cout << std::endl;
            } else {
                check(std::abs(geo.height - 1.8f) < 0.02f, "the capsule height is the model's");
                check(std::abs(geo.radius - 0.306f) < 0.02f,
                      "and its radius comes from the model's depth, not its arm span");

                check(named(Gait::Walk) == "walk" && named(Gait::Run) == "run",
                      "the slowest and fastest forward clips became walk and run");
                check(std::abs(speedOf(Gait::Walk) - 1.4f) < 0.05f &&
                              std::abs(speedOf(Gait::Run) - 4.f) < 0.05f,
                      "measured at the speeds they were authored at");
                check(named(Gait::WalkBack) == "walkback" && named(Gait::RunBack) == "jogback",
                      "the backward pair was identified by direction alone");
                check(named(Gait::StrafeLeft) == "strafeleft" &&
                              named(Gait::StrafeLeftFast) == "strafeleftfast",
                      "and so was the left strafe pair");
                check(named(Gait::StrafeRight) == "straferight" &&
                              named(Gait::StrafeRightFast) == "straferightfast",
                      "and the right");
                check(named(Gait::Idle) == "idle" && named(Gait::Jump) == "jump",
                      "idle and jump were picked by name, since a ruler cannot tell them apart");

                bool rampUsed = false;
                for (const auto& slot : geo.gaits) {
                    if (slot.clip && slot.clip->name() == "start walking") rampUsed = true;
                }
                check(!rampUsed,
                      "the ramping transition clip was rejected - its average speed is one "
                      "no controller ever holds");
            }

#ifdef THREEPP_EDITOR_WITH_PHYSX
            // --- Play: it walks, and it walks where the view points ----------
            // stepFixed throughout: "how far did it get" is a question about
            // simulated time, and the controls go through the same entry point
            // the W key does.
            // --- it can be CLICKED ------------------------------------------
            // A rigged character is a SkinnedMesh, and glTF says a skinned
            // mesh's node transform is ignored — so the raycaster's bind-pose
            // reject volume, pushed through a Mixamo armature's 0.01 scale, is
            // a centimetre-wide bubble at the origin that no click ever hits.
            // Selecting the model in the viewport is how a user reaches the
            // Character section in the first place, so it gets a check.
            {
                Vector3 chest;
                hero->getWorldPosition(chest);
                chest.y += 1.f;
                orbit_->target.copy(chest);
                viewCamera().position.set(chest.x, chest.y + 0.2f, chest.z + 4.f);
                selectObject(nullptr);
                step(2);

                const auto* viewport = ImGui::GetMainViewport();
                auto* picked = pickAt(viewport->Pos.x + viewport->Size.x * 0.5f,
                                      viewport->Pos.y + viewport->Size.y * 0.5f, false);
                bool underHero = picked == hero;
                for (const Object3D* o = picked; o && !underHero; o = o->parent) {
                    if (o == hero) underHero = true;
                }
                check(picked != nullptr && underHero,
                      "clicking the character in the viewport selects it");
                selectObject(hero);
                step();
            }

            // Where the view stands before Play. The chase camera takes it
            // over by itself, so Stop owes it back — see the check at the end.
            const Vector3 cameraBefore = viewCamera().position;
            const Vector3 targetBefore = orbit_->target;

            startPlay();
            stepFixed(2);
            check(isPlaying(), "Play starts with an authored character in the scene");
            check(characterSession_ && characterSession_->characterCount() == 1,
                  "the session built one capsule controller");

            auto* playedHero = findByUuid(document_.scene(), heroUuid);
            check(playedHero != nullptr, "the played scene still has the character");

            const auto drive = [&](float forward, float strafe, bool run, bool jump,
                                   float viewYawRad, int frames) {
                CharacterPlaySession::Input input;
                input.forward = forward;
                input.strafe = strafe;
                input.run = run;
                input.jump = jump;
                input.viewYaw = viewYawRad;
                for (int i = 0; i < frames; ++i) {
                    characterSession_->drive(input);
                    stepFixed(1);
                }
            };

            // Two seconds of walking with the view looking along +Z.
            const float startZ = playedHero ? playedHero->position.z : 0.f;
            drive(1.f, 0.f, false, false, 0.f, 120);
            const float walkedZ = playedHero ? playedHero->position.z - startZ : 0.f;
            check(walkedZ > 1.5f, "holding forward walked the character several metres");
            {
                const auto* played = characterSession_->player();
                check(played != nullptr && played->grounded,
                      "and it stayed on the ground the whole way");
                // A clip is actually playing, and the walk role resolved — the
                // demanded speed IS the walk clip's own, so that is the one
                // the picker had to reach for.
                check(played != nullptr && played->current != nullptr &&
                              played->geo.slot(Gait::Walk).clip != nullptr,
                      "with a locomotion clip playing");
                // The anti-foot-slide invariant, stated as a number: the
                // character travels at the WALK CLIP'S OWN speed, so the clip
                // plays at ~1x and each footfall lands where the ground is
                // moving. A still photograph cannot show sliding; this can.
                check(played != nullptr && played->walkSpeed > 0.f &&
                              std::abs(played->speed() - played->walkSpeed) <
                                      0.15f * played->walkSpeed,
                      "travelling at the walk clip's own authored speed");
            }

            // Now run, and cover more ground in the same time.
            const float runFrom = playedHero ? playedHero->position.z : 0.f;
            drive(1.f, 0.f, true, false, 0.f, 120);
            const float ranZ = playedHero ? playedHero->position.z - runFrom : 0.f;
            check(ranZ > walkedZ * 1.4f, "and Shift runs, covering noticeably more ground");

            // Strafing goes SIDEWAYS in the view's frame while the body keeps
            // facing the view — which is the whole reason the strafe clips
            // exist.
            const float strafeFromX = playedHero ? playedHero->position.x : 0.f;
            drive(0.f, 1.f, false, false, 0.f, 90);
            const float strafedX = playedHero ? playedHero->position.x - strafeFromX : 0.f;
            check(strafedX > 0.8f, "A strafes to the character's left, across the view");
            {
                const auto* played = characterSession_->player();
                check(played != nullptr &&
                              std::abs(std::remainder(played->yaw, math::TWO_PI)) < 0.2f,
                      "and the body still faces the way the camera looks");
            }

            // Turn the view a quarter turn: forward now means world +X.
            const float turnFromX = playedHero ? playedHero->position.x : 0.f;
            drive(1.f, 0.f, false, false, math::PI * 0.5f, 120);
            check(playedHero && playedHero->position.x - turnFromX > 1.f,
                  "turning the view turns what forward means");

            // Photographs, because a state machine passing says nothing about
            // whether the thing on screen looks like a person walking. One per
            // gait, all from the same side-on vantage so the four are
            // comparable: what should differ between them is the POSE, not the
            // framing.
            if (realAsset) {
                const auto pose = [&](const char* tag, float forward, float strafe, bool run) {
                    // Settle into the gait first, then frame and shoot: the
                    // crossfade is 0.18 s and a photograph taken during it is
                    // of neither clip.
                    drive(forward, strafe, run, false, 0.f, 45);
                    const Vector3 at = playedHero ? playedHero->position : Vector3();
                    orbit_->target.set(at.x, 0.9f, at.z);
                    viewCamera().position.set(at.x + 3.6f, 1.5f, at.z + 0.6f);
                    drive(forward, strafe, run, false, 0.f, 2);
                    shootTo(std::filesystem::temp_directory_path() /
                            (std::string("threepp_editor_character_") + tag + ".png"));
                };
                pose("walk", 1.f, 0.f, false);
                pose("run", 1.f, 0.f, true);
                pose("strafe", 0.f, 1.f, false);
                pose("back", -1.f, 0.f, false);
                // And the one the reader looks at first: three-quarter view,
                // walking.
                drive(1.f, 0.f, false, false, 0.f, 45);
                const Vector3 at = playedHero ? playedHero->position : Vector3();
                orbit_->target.set(at.x, 1.f, at.z);
                viewCamera().position.set(at.x + 2.6f, 1.8f, at.z + 3.2f);
                drive(1.f, 0.f, false, false, 0.f, 2);
                shootTo(std::filesystem::temp_directory_path() / "threepp_editor_character.png");
            }

            stopPlay();
            step(2);
            auto* restored = findByUuid(document_.scene(), heroUuid);
            check(restored != nullptr && restored->position.length() < 0.01f,
                  "and Stop puts the character back where it was authored");
            // The chase moved the view several metres following the character
            // (and the photo passes moved it further still). Stop restores the
            // scene, so it restores the view with it — otherwise the camera is
            // left staring at empty ground.
            check(viewCamera().position.distanceTo(cameraBefore) < 0.01f &&
                          orbit_->target.distanceTo(targetBefore) < 0.01f,
                  "and puts the camera back where it stood before Play");
#else
            std::cout << "[selftest] SKIP built without PhysX - the character play block "
                         "did not run"
                      << std::endl;
#endif
        }
    }

    // --- the ortho view SHADES like the perspective one --------------------
    // The checks above are about the projection; this one is about the light.
    // The Vulkan backend reads an OrthographicCamera as a 2D HUD unless told
    // otherwise (setOrthographicSceneRendering) and would otherwise draw every
    // mesh as a flat unlit fill — no sun, no shadows, no fog, no tone mapping.
    // That failure is invisible to every state-machine check ever written and
    // obvious the moment two frames of the same scene sit side by side, so
    // measure it: same viewpoint, same pivot, one toggle between them.
    //
    // Mean luminance of a CENTRED crop, which is the viewport (the panels sit
    // outside it) and is symmetric about the middle row — so it reads the same
    // whichever row order the backend hands back. The toggle preserves framing,
    // so the two crops see the same scene; a tolerance of a quarter absorbs the
    // parallel-vs-converging difference in what a pixel covers while still
    // being a fraction of the gap a flat unlit fill opens up.
    if (section("ortho-shading")) {
        newScene();
        selectObject(nullptr);
        camera_.position.set(-1.f, 14.f, 22.f);
        orbit_->target.set(0.f, 0.f, 8.f);
        setOrthographic(false);
        step(8);// TAA/DLSS history settles

        const auto size = canvas_.size();
        const auto cropMean = [&](const std::vector<unsigned char>& px) {
            const int w = size.width(), h = size.height();
            if (px.size() < static_cast<std::size_t>(w) * h * 3) return -1.0;
            const int x0 = w * 35 / 100, x1 = w * 65 / 100;
            const int y0 = h * 35 / 100, y1 = h * 65 / 100;
            double sum = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const std::size_t i = (static_cast<std::size_t>(y) * w + x) * 3;
                    sum += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
                }
            }
            return sum / static_cast<double>((x1 - x0) * (y1 - y0));
        };

        const double perspMean = cropMean(renderer_->readRGBPixels());
        setOrthographic(true);
        step(8);
        const double orthoMean = cropMean(renderer_->readRGBPixels());

        check(perspMean > 8.0, "the perspective viewport renders a lit scene");
        check(orthoMean > 8.0, "and so does the orthographic one");
        check(perspMean > 0.0 && orthoMean > 0.0 &&
                      std::abs(orthoMean - perspMean) < 0.25 * perspMean,
              "the two projections shade the same scene to the same brightness");

        setOrthographic(false);
        step();
    }

    // --- and it casts the same shadows -------------------------------------
    // A mean is a weak witness: a flat unlit fill of a light grey ground
    // averages out close to a lit one, which is exactly how a projection that
    // dropped every light could pass the check above. A SHADOW cannot be
    // faked — it is the one thing the unlit path has no way to produce. So
    // float the template's Box, work out where the sun must put its shadow on
    // the ground, project THAT world point through whichever camera is live,
    // and read the pixel. Lit ground beside it is the control.
    if (section("ortho-shadows")) {
        newScene();
        selectObject(nullptr);

        auto* box = document_.scene().getObjectByName("Box");
        auto* sun = document_.scene().getObjectByName("Sun");
        if (box && sun) {
            box->position.set(0.f, 3.f, 0.f);
            // One light, so the shadow is a real hole rather than a dent in the
            // ambient fill. The deferred path's denoised shadow plus its GI
            // bounce close most of a 0.35 ambient back up, which leaves the
            // measurement riding on a difference too small to state a claim on.
            if (auto* ambient = document_.scene().getObjectByName("Ambient Light")) {
                if (auto* light = ambient->as<Light>()) light->intensity = 0.f;
            }

            // The template's sun points at the origin, so its direction is the
            // way its position falls. Where that direction drops the box centre
            // onto y = 0 is where the shadow's middle lands.
            Vector3 toGround = Vector3(0, 0, 0).sub(sun->position).normalize();
            const float drop = (std::abs(toGround.y) > 1e-3f) ? 3.f / -toGround.y : 0.f;
            const Vector3 shadowed = box->position.clone().addScaledVector(toGround, drop);
            // The control: ground the box cannot reach, but still near enough to
            // the pivot to stay in the middle of the window. The panels are drawn
            // OVER the viewport, so a probe that wanders out to the edge reads
            // ImGui, not the scene — lumaAt refuses anything past the centre.
            const Vector3 lit(3.f, 0.f, 3.f);

            const auto size = renderer_->size();
            bool bottomUp = true;
#ifdef THREEPP_WITH_VULKAN
            if (dynamic_cast<VulkanRenderer*>(renderer_.get())) bottomUp = false;
#endif
            // Mean luminance of a small patch around a WORLD point, or -1 when
            // it projects off-screen. Small enough to sit inside the shadow, big
            // enough that one stray denoised pixel can't decide the answer.
            const auto lumaAt = [&](const Vector3& world) {
                const auto pixels = renderer_->readRGBPixels();
                const int  w = size.width(), h = size.height();
                if (pixels.size() < static_cast<std::size_t>(w) * h * 3) return -1.0;
                Vector3 ndc = world;
                ndc.project(viewCamera());
                // Centre only: the hierarchy, inspector and console panels sit
                // on top of the rendered frame, so a probe outside this box
                // would be measuring UI.
                if (std::abs(ndc.x) > 0.45f || ndc.y < -0.4f || ndc.y > 0.6f) return -1.0;
                const int cx = static_cast<int>((ndc.x * 0.5f + 0.5f) * static_cast<float>(w));
                const float v = bottomUp ? (ndc.y * 0.5f + 0.5f) : (0.5f - ndc.y * 0.5f);
                const int cy = static_cast<int>(v * static_cast<float>(h));
                double sum = 0;
                int    n   = 0;
                for (int y = cy - 2; y <= cy + 2; ++y) {
                    for (int x = cx - 2; x <= cx + 2; ++x) {
                        if (x < 0 || y < 0 || x >= w || y >= h) continue;
                        const std::size_t i = (static_cast<std::size_t>(y) * w + x) * 3;
                        sum += 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
                        ++n;
                    }
                }
                return n > 0 ? sum / n : -1.0;
            };

            // The deferred path resolves this shadow by TEMPORAL ACCUMULATION,
            // and this scene is its worst case: the ambient was just zeroed, so
            // the shadowed ground is lit only by the GI bounce — the
            // slowest-converging term there is. Measured on the perspective
            // view, the shadow reads 23.3 of 33 at 24 frames, 14.0 at 48, 7.6 at
            // 96 and 2.9 once settled, so a 24-frame budget states the claim
            // against a half-drawn image.
            //
            // That, not the projection, is what used to fail here. Whichever
            // view was measured FIRST read an unconverged shadow and the other
            // inherited its history and passed: perspective first failed at
            // delta 9.7, and reversing the order failed perspective's way round
            // instead — ortho first scraped 13.8 against a threshold of 12.
            // Settle both properly and the order stops deciding the answer.
            // 96 leaves the measurement at roughly twice the threshold. GL is
            // unaffected either way; a shadow map is not temporal and reads 0
            // on the first frame.
            constexpr int kShadowSettleFrames = 96;

            // Perspective first — the reference for what "the sun casts a
            // shadow here" looks like in this scene at all.
            setOrthographic(false);
            camera_.position.set(1.f, 16.f, 14.f);
            orbit_->target.set(0.f, 0.f, 0.f);
            step(kShadowSettleFrames);
            const double perspLit = lumaAt(lit);
            const double perspShadow = lumaAt(shadowed);
            // >= 0 is the on-screen test (lumaAt answers -1 when a probe would
            // land under a panel); a shadow that reads exactly 0 is the GL
            // shadow map doing its job, not a failure.
            std::cout << "[probe] shadow persp: lit " << perspLit << " shadow " << perspShadow
                      << " size " << size.width() << "x" << size.height() << std::endl;
            shootTo(std::filesystem::temp_directory_path() / "threepp_editor_shadow_probe.png");
            check(perspLit >= 0.0 && perspShadow >= 0.0 && perspLit - perspShadow > 12.0,
                  "the sun casts a visible shadow in the perspective view");

            // Same claim, same points, parallel projection — and the same
            // settle, so this one is not just reading the frames the
            // perspective pass paid for.
            setOrthographic(true);
            step(kShadowSettleFrames);
            const double orthoLit = lumaAt(lit);
            const double orthoShadow = lumaAt(shadowed);
            check(orthoLit >= 0.0 && orthoShadow >= 0.0 && orthoLit - orthoShadow > 12.0,
                  "and the same shadow through the orthographic one");

            setOrthographic(false);
            step();
        }
    }

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // The physics collider overlay. It exists because a body resting on nothing
    // and a body whose collider is somewhere else look identical, so the checks
    // are about the buffer being there, being emptied, and surviving the scene
    // replace that Stop performs under it.
    if (section("collider-overlay")) {
        newScene();
        step(2);

        for (const char* name : {"Ground", "Box"}) {
            auto* object = document_.scene().getObjectByName(name);
            if (!object) continue;
            PhysicsConfig config;
            config.enabled = true;
            config.body = std::string(name) == "Box" ? PhysicsConfig::Body::Dynamic
                                                     : PhysicsConfig::Body::Static;
            config.write(*object);
        }

        // Vertices the overlay is actually drawing. drawRange starts at "all of
        // them" — a sentinel, not a count — so a geometry that has never been
        // filled has to answer zero rather than half a billion.
        const auto colliderVertices = [this] {
            const auto geometry = physicsDebugLines_ ? physicsDebugLines_->geometry() : nullptr;
            const auto* position = geometry ? geometry->getAttribute<float>("position") : nullptr;
            if (!position) return 0;
            return std::min(geometry->drawRange.count, position->count());
        };

        // PhysX fills the debug buffer during simulate(), and the session steps
        // on a FIXED timestep out of a real-time clock: on a machine drawing
        // frames faster than the substep, several frames pass before the sim
        // advances at all. Waiting on the buffer rather than on a frame count
        // is what keeps this about the overlay instead of about the frame rate.
        const auto stepUntilColliders = [&](int budget) {
            for (int i = 0; i < budget && colliderVertices() == 0; ++i) step();
        };

        physicsDebug_ = true;
        step(2);
        check(physicsDebugLines_ == nullptr, "the collider overlay stays off while stopped");

        startPlay();
        // The lines arrive one substep after the visualization parameter is
        // set, which is one substep after play started.
        stepUntilColliders(600);
        check(physicsDebugLines_ != nullptr, "toggling Physics Debug on while playing draws an overlay");
        check(physicsDebugLines_ && physicsDebugLines_->visible, "which is visible");
        check(colliderVertices() > 0, "with a non-empty line buffer");

        physicsDebug_ = false;
        step(2);
        check(physicsDebugLines_ == nullptr, "toggling it off clears the overlay");

        // Back on, then Stop: that is a full scene replace with live world
        // pointers in the overlay, which is the shape of every scar this file
        // carries.
        physicsDebug_ = true;
        stepUntilColliders(600);
        check(colliderVertices() > 0, "toggling it back on refills the buffer");
        stopPlay();
        step(2);
        check(physicsDebugLines_ == nullptr, "stopping play clears it, scene replace and all");

        // And once more across an explicit document replace while playing.
        startPlay();
        stepUntilColliders(600);
        check(colliderVertices() > 0, "the overlay comes back on the next play");
        stopPlay();
        newScene();
        step(2);
        check(physicsDebugLines_ == nullptr, "a new scene with the toggle on leaves nothing behind");

        // A generated tube collides as its own triangles — a closed
        // cross-section is exactly what a triangle mesh handles well — and the
        // collider overlay draws them. Physics goes on the SPLINE here, not on
        // the mesh: that is where a user puts it, and it used to produce a
        // phantom 1 m box at the spline's origin instead of the tube.
        {
            auto created = ObjectFactory::createSpline(document_.scene());
            const auto splineUuid = created->uuid;
            addObject(created, document_.scene(), "Add Spline");
            step();

            auto* spline = findByUuid(document_.scene(), splineUuid);
            if (spline) {
                const Vector3 shape[]{{-6, 0, 0}, {-2, 0, 2.5f}, {2, 0, -2.5f}, {6, 0, 0}};
                const auto points = SplineConfig::controlPointNodes(*spline);
                for (std::size_t i = 0; i < points.size() && i < 4; ++i) {
                    points[i]->position.copy(shape[i]);
                }
                auto config = SplineConfig::read(*spline).value_or(SplineConfig{});
                config.mesh = SplineConfig::MeshKind::Tube;
                config.radius = 0.5f;
                config.write(*spline);
                step();

                PhysicsConfig physics;
                physics.enabled = true;
                physics.body = PhysicsConfig::Body::Static;
                physics.shape = PhysicsConfig::Shape::Auto;
                physics.write(*spline);
            }

            startPlay();
            stepUntilColliders(600);
            check(colliderVertices() > 0, "an S-curve tube draws collider lines of its own");

            int shapes = -1;
            if (auto* world = physics_ ? physics_->world() : nullptr) {
                using namespace ::physx;
                auto& pxScene = world->scene();
                const PxU32 count = pxScene.getNbActors(PxActorTypeFlag::eRIGID_STATIC);
                std::vector<PxActor*> actors(count);
                if (count) pxScene.getActors(PxActorTypeFlag::eRIGID_STATIC, actors.data(), count);
                shapes = 0;
                for (auto* actor : actors) {
                    shapes += static_cast<int>(static_cast<PxRigidActor*>(actor)->getNbShapes());
                }
            }
            // One shape, cooked from the tube the spline generated — not the
            // unit-box placeholder a geometry-less group used to fall back on.
            check(shapes == 1, "and physics on the SPLINE collides as the tube under it");
            stopPlay();
            step(2);
        }
        physicsDebug_ = false;
    }

    // --- sensors -----------------------------------------------------------
    // Drives the whole authoring-to-measurement path through the same members
    // the UI calls: userData written the way the inspector writes it, Play, and
    // then assertions about what came BACK. A sensor that authors cleanly and
    // measures nothing is the failure mode here, and nothing short of reading
    // the samples catches it.
    if (section("sensors")) {
        newScene();
        step(2);

        auto* ground = document_.scene().getObjectByName("Ground");
        auto* box = document_.scene().getObjectByName("Box");
        check(box != nullptr && ground != nullptr, "template scene for the sensor drive");

        std::string imuUuid, lidarUuid, depthUuid;

        if (box && ground) {
            // KINEMATIC, not dynamic. A kinematic body never moves and gravity
            // does not act on it, so the accelerometer's specific force is
            // exactly -g in world space — i.e. +g on the sensor's up axis, with
            // no settling, no contact and no bounce mixed into the number. Same
            // fact as a body with eDISABLE_GRAVITY reading +g.
            PhysicsConfig physics;
            physics.enabled = true;
            physics.body = PhysicsConfig::Body::Kinematic;
            physics.shape = PhysicsConfig::Shape::Box;
            physics.write(*box);

            SensorConfig imu;
            imu.enabled = true;
            imu.type = SensorConfig::Type::Imu;
            imu.rateHz = 0.f;// every substep
            // A zero noise model is a bit-exact passthrough, which is what makes
            // a physics-truth assertion possible at all.
            imu.gyroNoiseDensity = 0.f;
            imu.gyroRandomWalk = 0.f;
            imu.accelNoiseDensity = 0.f;
            imu.accelRandomWalk = 0.f;
            imu.write(*box);
            imuUuid = box->uuid;

            // The LIDAR goes on a node of its own, above the ground and clear of
            // it. No physics: a vision sensor is pulled from the frame loop and
            // needs no body — which is also what makes its cloud reproducible,
            // since a static pose cannot drift between plays.
            auto mast = ObjectFactory::createPrimitive(Primitive::Sphere, document_.scene());
            mast->name = "Lidar";
            mast->position.set(1.5f, 1.2f, 1.5f);
            mast->scale.set(0.15f, 0.15f, 0.15f);

            SensorConfig lidar;
            lidar.enabled = true;
            lidar.type = SensorConfig::Type::Lidar;
            lidar.beams = SensorConfig::Beams::VLP16;
            lidar.faceSize = 64;
            lidar.rateHz = 0.f;// scan on every frame, so one step = one scan
            lidar.seed = 12345;
            lidar.rangeStddev = 0.01f;// noise on: the seed has to matter
            lidar.nearPlane = 0.2f;
            lidar.farPlane = 30.f;
            lidar.write(*mast);
            lidarUuid = mast->uuid;
            addObject(mast, document_.scene(), "Add Lidar");

            // A depth camera too: it is a different GL path from the LIDAR (one
            // pinhole pass and a readback, not six cube faces), so the LIDAR
            // passing says nothing about it. Aimed straight down at the ground,
            // which is the one thing every template scene has.
            //
            // Hosted on a CAMERA, the way the inspector now authors pinholes,
            // and the camera's frustum disagrees with the flat string ON
            // PURPOSE: the config's numbers (fov 60, far 3.5) would put the
            // ground — 4 m straight down — entirely beyond the range shell and
            // return an empty cloud. Every depth assertion below therefore
            // doubles as proof that Play read the camera, not the stale copy.
            auto eye = ObjectFactory::createCamera(document_.scene());
            eye->name = "Depth Cam";
            eye->position.set(-2.f, 4.f, 2.f);
            // -Z is the viewing direction; -90 degrees about X points it at -Y.
            eye->rotation.x = -1.5707963f;
            eye->fov = 110.f;
            eye->nearPlane = 0.2f;
            eye->farPlane = 20.f;
            eye->updateProjectionMatrix();

            SensorConfig depth;
            depth.enabled = true;
            depth.type = SensorConfig::Type::Depth;
            depth.width = 96;
            depth.height = 72;
            depth.fovY = 60.f; // stale on purpose — the camera's 110 is the truth
            depth.rateHz = 0.f;
            depth.seed = 777;
            depth.rangeStddev = 0.005f;
            depth.nearPlane = 0.2f;
            depth.farPlane = 3.5f;// stale too: shorter than anything in view
            depth.write(*eye);
            depthUuid = eye->uuid;
            addObject(eye, document_.scene(), "Add Depth Cam");
            step();
        }

        // Authored, before any Play: the marker pass has to have picked the
        // sensor glyph for all three, or an instrumented object is invisible.
        check(viewportMarkers_.size() >= 3, "an authored sensor gets a viewport marker");

        const auto entryFor = [this](const std::string& uuid) -> const SensorPlaySession::Entry* {
            if (!sensors_) return nullptr;
            for (const auto& entry : sensors_->entries()) {
                if (entry->uuid == uuid) return entry.get();
            }
            return nullptr;
        };
        // Order-independent so a cloud is compared by CONTENT: the same returns
        // in a different order would be a different cloud to a hash over the
        // sequence, and the beam order is the thing the seed reproduces, so it
        // must not change either. Hash the sequence.
        const auto hashCloud = [](const std::vector<LidarReturn>& cloud) {
            std::size_t h = 1469598103934665603ull;
            const auto mix = [&h](float v) {
                std::uint32_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                h = (h ^ bits) * 1099511628211ull;
            };
            for (const auto& r : cloud) {
                mix(r.position.x);
                mix(r.position.y);
                mix(r.position.z);
                mix(r.distance);
            }
            return h;
        };
        const auto hashPoints = [](const std::vector<Vector3>& cloud) {
            std::size_t h = 1469598103934665603ull;
            const auto mix = [&h](float v) {
                std::uint32_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                h = (h ^ bits) * 1099511628211ull;
            };
            for (const auto& p : cloud) {
                mix(p.x);
                mix(p.y);
                mix(p.z);
            }
            return h;
        };

        const auto recordDir = std::filesystem::temp_directory_path() / "threepp-editor-selftest-sensors";
        std::error_code ec;
        std::filesystem::remove_all(recordDir, ec);
        if (sensors_) {
            sensors_->setRecordDirectory(recordDir);
            // Armed while stopped: the files open on the first measurement of the
            // play that follows, so "Record then Play" captures from t = 0.
            sensors_->setRecording(true);
        }

        // The raster backends read depth back through a 16-bit packed texture,
        // so a replayed scan is bit-identical and the hash is the right test.
        // The ray-traced (Vulkan) path is not: play/stop rebuilds the entry
        // list, so the TLAS is rebuilt with a different instance ordering, and
        // the same ray against the same static surface resolves 1-2 ULP apart
        // (measured: 4.62550545 vs 4.62550735 m, and the ground's instance id
        // moves too). What the authored seed guarantees is the noise draw, not
        // the last bit of a hardware intersection — so there, assert the clouds
        // agree far inside the noise sigma instead. Measured across plays,
        // scan 1 vs scan 1 agrees to 1.5e-5 m, 70x inside this bound; a seed
        // that failed to replay would deviate by ~sigma (0.02 m), 20x outside
        // it. Both failure modes stay far from the line.
        constexpr float kReplayTolerance = 1e-3f;
        const bool bitExactReplay = !options_.vulkan;

        // Replay needs a deterministic sim advance, so everything below steps
        // through the shared `stepFixed()`. Under `step()` — wall clock — the
        // two plays landed their scans at different sim times (measured:
        // 0.033 s vs 0.000 s) and every dynamic object was somewhere else when
        // the beam arrived. That is a property of the machine, not of the seed.

        std::size_t firstHash = 0;
        std::size_t firstPoints = 0;
        std::size_t firstDepthHash = 0;
        std::size_t firstDepthPoints = 0;
        std::vector<LidarReturn> firstReturns;
        std::vector<Vector3> firstDepthCloud;

        // Largest per-point deviation between two clouds of equal length.
        const auto maxDeviation = [](const auto& a, const auto& b, auto position) {
            float worst = 0.f;
            const std::size_t n = std::min(a.size(), b.size());
            for (std::size_t i = 0; i < n; ++i) {
                const Vector3& p = position(a[i]);
                const Vector3& q = position(b[i]);
                worst = std::max(worst, std::abs(p.x - q.x));
                worst = std::max(worst, std::abs(p.y - q.y));
                worst = std::max(worst, std::abs(p.z - q.z));
            }
            return worst;
        };
        const auto returnPos = [](const LidarReturn& r) -> const Vector3& { return r.position; };
        const auto pointPos = [](const Vector3& p) -> const Vector3& { return p; };

        for (int pass = 0; pass < 2; ++pass) {

            startPlay();
            check(isPlaying(), "play starts with sensors in the scene");
            check(sensors_ && sensors_->sensorCount() == 3, "all three authored sensors are built");
            check(sensors_ && sensors_->liveCount() == 3, "and all three come up live");

            // The first scan of each vision sensor, LATCHED the moment it
            // lands and nothing later: every scan draws from the range-noise
            // stream, so scan N is only comparable with scan N of another run.
            //
            // Both sensors are ungated, so both FIRE on the first frame. Which
            // frame they are DELIVERED on is a property of the backend and of
            // the machine: a raster scan lands on the frame it fired (six
            // blocking framebuffer reads), while on Vulkan the beams are traced
            // on the GPU and collected by a later frame's fence poll, because
            // blocking for a readback costs every frame already queued (see
            // SensorPlaySession::scanAll). Worse, the two sensors' deliveries
            // skew INDEPENDENTLY: a fence not ready on the polled frame delays
            // that sensor's delivery by a frame — and the entry's cloud is
            // rewritten in place per scan, so by the time both sensors have
            // landed one, the faster one may already hold scan 2. Reading the
            // live entries at any fixed point in the pass therefore compares
            // scan N of one pass against scan M of the other — measured as
            // pass 1 holding its 11th scan where pass 2 held its 12th, a
            // fresh noise draw on every beam and a ~4 sigma max deviation,
            // where scan 1 vs scan 1 of the same runs agreed to 1.4e-5 m. So:
            // latch each sensor's cloud at the first moment its scan counter
            // is 1, and compare only the latches. What the seed guarantees —
            // the same cloud from the same pose — is untouched by which frame
            // carried it.
            const auto* lidarEntry = entryFor(lidarUuid);
            const auto* depthEntry = entryFor(depthUuid);
            std::vector<LidarReturn> scanReturns;// lidar scan 1, as delivered
            std::vector<Vector3> scanCloud;      // depth scan 1, as delivered
            bool lidarLatched = false;
            bool depthLatched = false;
            for (int i = 0; i < 600; ++i) {
                lidarEntry = entryFor(lidarUuid);
                depthEntry = entryFor(depthUuid);
                // Deliveries land at most one per frame per sensor, and the
                // loop looks after every frame, so an unlatched sensor with
                // scans > 0 is at exactly scan 1.
                if (!lidarLatched && lidarEntry && lidarEntry->scans > 0) {
                    scanReturns = lidarEntry->returns;
                    lidarLatched = true;
                }
                if (!depthLatched && depthEntry && depthEntry->scans > 0) {
                    scanCloud = depthEntry->cloud;
                    depthLatched = true;
                }
                if (lidarLatched && depthLatched) break;
                stepFixed();
            }
            check(lidarEntry != nullptr && lidarEntry->scans >= 1,
                  "the first frame of play fires a scan, and it is delivered");
            const std::size_t scanHash = hashCloud(scanReturns);
            const std::size_t scanPoints = scanReturns.size();

            check(depthEntry != nullptr && depthEntry->scans >= 1,
                  "the depth camera's scan is delivered too");
            const std::size_t depthHash = hashPoints(scanCloud);
            const std::size_t depthPoints = scanCloud.size();
            // Not empty ALSO proves the far plane is the camera's 20 m: the
            // config's 3.5 m shell ends short of the 4 m ground below.
            check(depthPoints > 0, "and its cloud is not empty");
            // And the fov is the camera's 110 degrees: the config's 60-degree
            // cone tops out at 5.55 m of slant range on this geometry, while
            // the wide cone reads past 8 m near the image's x edges.
            {
                float longest = 0.f;
                const Vector3 eyeAt(-2.f, 4.f, 2.f);
                for (const auto& p : scanCloud) {
                    longest = std::max(longest, p.distanceTo(eyeAt));
                }
                check(longest > 6.f,
                      "the scan reads the camera's frustum, not the config's stale copy");
            }

            // The IMU needs a substep to have run, which on a fast machine is not
            // the first frame.
            for (int i = 0; i < 600; ++i) {
                const auto* imuEntry = entryFor(imuUuid);
                if (imuEntry && imuEntry->samples > 0) break;
                stepFixed();
            }

            const auto* imuEntry = entryFor(imuUuid);
            check(imuEntry != nullptr && imuEntry->samples > 0, "IMU samples arrive during play");

            if (imuEntry && imuEntry->imu) {
                const auto sample = imuEntry->imu->latest();
                check(sample.has_value(), "and the IMU has a latest reading");
                if (sample) {
                    // The one assertion that says the sensor is measuring PHYSICS
                    // and not zeros: a level accelerometer at rest reads +g up.
                    check(std::abs(sample->linearAcceleration.y - 9.81f) < 1e-2f,
                          "an IMU at rest reads +g on its up axis");
                    check(std::abs(sample->linearAcceleration.x) < 1e-2f &&
                                  std::abs(sample->linearAcceleration.z) < 1e-2f,
                          "and nothing on the horizontal axes");
                }
                // Sim time, not wall time, and it has to move.
                const double before = imuEntry->lastTime;
                check(before > 0.0, "IMU samples are stamped with a non-zero sim time");
                stepFixed(20);
                check(entryFor(imuUuid) && entryFor(imuUuid)->lastTime > before,
                      "and the timestamps advance with the simulation");
            }

            lidarEntry = entryFor(lidarUuid);
            check(lidarEntry != nullptr && lidarEntry->scans > 0, "the LIDAR scans during play");
            check(lidarEntry != nullptr && !lidarEntry->returns.empty(),
                  "and its cloud is not empty");

            // The overlay is the visible half. It must be drawing the same points.
            const auto cloudGeometry = sensorCloud_ ? sensorCloud_->geometry() : nullptr;
            check(cloudGeometry != nullptr, "the sensor cloud overlay exists while playing");
            check(cloudGeometry && cloudGeometry->drawRange.count > 0, "and is drawing points");
            check(sensorCloud_ && sensorCloud_->visible, "and is visible");

            if (pass == 0) {
                firstHash = scanHash;
                firstPoints = scanPoints;
                firstDepthHash = depthHash;
                firstDepthPoints = depthPoints;
                firstReturns = std::move(scanReturns);
                firstDepthCloud = std::move(scanCloud);
                check(firstPoints > 0, "the first scan has returns to compare");
            } else {
                // Sensors are rebuilt from the authored seed on every Play. Two
                // plays of the same scene are therefore the same dataset — the
                // whole reason the seed is authored rather than drawn from the OS.
                check(scanPoints == firstPoints,
                      "the second play scans the same number of returns");
                check(depthPoints == firstDepthPoints,
                      "the depth camera returns the same number of points");

                if (bitExactReplay) {
                    check(scanHash == firstHash, "and an identical cloud - the seed replays");
                    check(depthHash == firstDepthHash, "and an identical depth cloud");
                } else {
                    const float lidarDev = maxDeviation(scanReturns, firstReturns, returnPos);
                    const float depthDev = maxDeviation(scanCloud, firstDepthCloud, pointPos);
                    // The failure line alone cannot say how far off the clouds
                    // were, and that number is the whole diagnosis: ~1e-5 is
                    // the TLAS rebuild, ~sigma is a seed that did not replay.
                    if (lidarDev >= kReplayTolerance || depthDev >= kReplayTolerance) {
                        std::cout << "[selftest] replay deviation: lidar " << lidarDev
                                  << " m, depth " << depthDev << " m (tolerance "
                                  << kReplayTolerance << " m)" << std::endl;
                    }
                    check(lidarDev < kReplayTolerance,
                          "and the same cloud within the traced path's tolerance - the seed replays");
                    check(depthDev < kReplayTolerance,
                          "and the same depth cloud within that tolerance");
                }
            }

            check(sensors_ && sensors_->recordedRows() > 0, "recording accumulates rows");

            stopPlay();
            stepFixed(2);
            check(sensors_ && sensors_->sensorCount() == 0, "stop drops every sensor");
            check(sensorCloud_ == nullptr, "and clears the cloud overlay");
            check(sensorRig_ && sensorRig_->children.empty(),
                  "and leaves nothing parented to the sensor rig");
        }

        // The CSVs are flushed on Stop, one per sensor, header plus rows.
        int csvFiles = 0;
        int csvRows = 0;
        if (std::filesystem::exists(recordDir)) {
            for (const auto& file : std::filesystem::directory_iterator(recordDir)) {
                if (file.path().extension() != ".csv") continue;
                ++csvFiles;
                std::ifstream in(file.path());
                std::string line;
                int lines = 0;
                while (std::getline(in, line)) ++lines;
                csvRows += std::max(lines - 1, 0);
            }
        }
        // One per sensor, plus the final-cloud dump each vision sensor writes.
        check(csvFiles >= 3, "recording wrote a CSV per sensor");
        check(csvRows > 0, "with rows in them");
        if (sensors_) sensors_->setRecording(false);
        std::filesystem::remove_all(recordDir, ec);

        // Two scene replaces back to back with the cloud toggle still on. Every
        // scar this file carries is a pointer into a graph that was swapped out
        // from under an overlay, and doing it twice is what catches the ones that
        // survive the first.
        newScene();
        step(2);
        newScene();
        step(2);
        check(sensorCloud_ == nullptr, "a double scene replace leaves no sensor overlay");
        check(sensors_ && sensors_->sensorCount() == 0, "and no sensors");
        // The sensor glyphs went with the objects that carried them; the fresh
        // template scene's lights still get theirs, which is the marker pass
        // having survived two graph swaps rather than gone quiet.
        check(!viewportMarkers_.empty(), "and a marker pass that still runs");
    }
#endif

    // --- the colour camera sensor -------------------------------------------
    //
    // Asserted HERE and not in a unit test because the output is PIXELS: a
    // frame needs a live GL context, and every failure mode worth catching is
    // invisible to a test that can only ask whether a sensor was built. A
    // black frame, a frame of one flat colour, a frame the wrong size and a
    // frame full of the editor's own grid all pass "the camera came up".
    //
    // Outside the PhysX block above on purpose: a picture needs a renderer and
    // a scene, so this runs in every build.
    if (section("camera-sensor")) {
        newScene();
        step(2);

        std::string cameraUuid;
        {
            // Deliberately the LEGACY shape — a pinhole on a plain object,
            // which the inspector no longer authors but a saved scene may
            // still carry. It playing here is the soft-enforcement contract:
            // the gate is authoring-side, never the session's.
            auto eye = ObjectFactory::createPrimitive(Primitive::Sphere, document_.scene());
            eye->name = "Wrist Cam";
            // In front of the template scene's contents, looking down its own
            // -Z at them — the same convention every threepp camera has.
            eye->position.set(0.f, 1.2f, 5.f);
            eye->scale.set(0.1f, 0.1f, 0.1f);

            SensorConfig camera;
            camera.enabled = true;
            camera.type = SensorConfig::Type::Camera;
            camera.width = 128;
            camera.height = 96;
            camera.fovY = 55.f;
            camera.rateHz = 0.f;// every frame
            camera.nearPlane = 0.05f;
            camera.farPlane = 60.f;
            camera.write(*eye);
            cameraUuid = eye->uuid;
            addObject(eye, document_.scene(), "Add Wrist Cam");
            step();
        }

        startPlay();
        stepFixed(20);

        const SensorPlaySession::Entry* cameraEntry = nullptr;
        if (sensors_) {
            for (const auto& entry : sensors_->entries()) {
                if (entry->uuid == cameraUuid) cameraEntry = entry.get();
            }
        }
        check(cameraEntry != nullptr && cameraEntry->camera != nullptr,
              "the authored colour camera came up");

        if (cameraEntry && cameraEntry->camera) {
            const auto& sensor = *cameraEntry->camera;
            check(sensor.frames() > 0, "and captured frames during play");

            const auto& image = sensor.image();
            check(image.size() == static_cast<std::size_t>(128 * 96 * 3),
                  "the frame is the authored size, tightly packed RGB8");

            // A frame that is black, or one flat colour, is what a broken
            // render target, a missed clear or a camera inside geometry all
            // look like — and all three would sail past a size check.
            double sum = 0.0;
            unsigned char lo = 255, hi = 0;
            for (std::size_t i = 0; i + 2 < image.size(); i += 3) {
                const auto luma = static_cast<unsigned char>(
                        (299 * image[i] + 587 * image[i + 1] + 114 * image[i + 2]) / 1000);
                sum += luma;
                lo = std::min(lo, luma);
                hi = std::max(hi, luma);
            }
            const double mean = image.empty() ? 0.0 : sum / (static_cast<double>(image.size()) / 3.0);
            check(mean > 4.0, "the frame is not black");
            check(hi - lo > 24, "and has a scene in it rather than one flat fill");

            // Row ORDER, not just row statistics: a vertically mirrored frame
            // keeps every mean and range intact, so only an up-versus-down
            // comparison can see one. image() promises top-left origin, and
            // from this viewpoint the template scene is dark sky over bright
            // ground.
            const auto quarterMeans = [](const unsigned char* rgb, int w, int h) {
                const auto meanOf = [&](int firstRow, int lastRow) {
                    double lumaSum = 0.0;
                    for (int y = firstRow; y < lastRow; ++y)
                        for (int x = 0; x < w; ++x) {
                            const auto* p = rgb + (static_cast<std::size_t>(y) * w + x) * 3;
                            lumaSum += (299 * p[0] + 587 * p[1] + 114 * p[2]) / 1000.0;
                        }
                    return lumaSum / (static_cast<double>(lastRow - firstRow) * w);
                };
                return std::pair{meanOf(0, h / 4), meanOf(h - h / 4, h)};
            };
            const auto [skyLuma, groundLuma] = quarterMeans(image.data(), 128, 96);
            std::cout << "[selftest] camera frame top-quarter mean " << skyLuma
                      << ", bottom-quarter mean " << groundLuma << std::endl;
            check(groundLuma > skyLuma + 8.0,
                  "and is the right way up (dark sky above bright ground)");

            // Round-trip through a file: this is the path a dataset dump and a
            // script's save() both take, and an encode that silently fails
            // would leave a user with an empty directory and no error.
            std::error_code ec;
            const auto shot = std::filesystem::temp_directory_path() /
                              "threepp-selftest-camera" / "frame.png";
            std::filesystem::remove_all(shot.parent_path(), ec);
            check(sensor.writeImage(shot), "the frame writes to a PNG");
            check(std::filesystem::exists(shot) && std::filesystem::file_size(shot, ec) > 0,
                  "and the file has bytes in it");

            // Decoded BACK, not just present: stb's flip-on-write flag is
            // process-global, and the screenshot passes earlier in this run
            // exercise the one caller that sets it. Left set, it inverts every
            // PNG a sensor writes while the in-memory frame stays correct —
            // only the file itself can witness that.
            int decodedW = 0, decodedH = 0;
            unsigned char* decoded = stbi_load(shot.string().c_str(), &decodedW, &decodedH, nullptr, 3);
            if (decoded) {
                const auto [fileSky, fileGround] = quarterMeans(decoded, decodedW, decodedH);
                check(decodedW == 128 && decodedH == 96 && fileGround > fileSky + 8.0,
                      "and the PNG decodes back the right way up (no leaked write-flip)");
                stbi_image_free(decoded);
            } else {
                check(false, "and the PNG decodes back the right way up (no leaked write-flip)");
            }
            std::filesystem::remove_all(shot.parent_path(), ec);
        }

        stopPlay();
        check(sensors_ && sensors_->sensorCount() == 0, "stop drops the camera with everything else");

        // --- and its recording is FRAMES, not a summary --------------------
        // recordCamera is its own path (a PNG per capture plus an index CSV,
        // where the ranging sensors write one row per scan), so the recording
        // pass above proves nothing about it. Armed before Play, like a user
        // pressing Record then Play, which is the capture-from-t=0 contract.
        {
            std::error_code ec;
            const auto recordDir = std::filesystem::temp_directory_path() /
                                   "threepp-selftest-camera-rec";
            std::filesystem::remove_all(recordDir, ec);
            if (sensors_) {
                sensors_->setRecordDirectory(recordDir);
                sensors_->setRecording(true);
            }

            startPlay();
            stepFixed(12);
            stopPlay();
            if (sensors_) sensors_->setRecording(false);

            std::size_t pngs = 0, csvs = 0, rows = 0;
            if (std::filesystem::exists(recordDir, ec)) {
                for (const auto& file : std::filesystem::directory_iterator(recordDir)) {
                    const auto ext = file.path().extension();
                    if (ext == ".png") ++pngs;
                    if (ext != ".csv") continue;
                    ++csvs;
                    std::ifstream in(file.path());
                    std::string line;
                    while (std::getline(in, line)) ++rows;
                }
            }
            check(pngs > 1, "recording a camera writes a PNG per frame");
            check(csvs == 1, "and one index CSV");
            // Header plus one row per PNG: the index and the frames agree.
            check(rows == pngs + 1, "whose rows match the frames on disk");
            std::filesystem::remove_all(recordDir, ec);
        }
    }

    // --- moving a legacy pinhole onto a camera child ------------------------
    //
    // The migration behind the inspector's "Move To Camera Child" hint, driven
    // through the same member the button defers to. Document machinery only —
    // no Play needed — so it runs in every build. The interesting properties:
    // the config crosses whole, the camera is stamped from its numbers, the
    // child's identity transform preserves the aim the host's transform
    // encoded, and ONE undo restores the entire legacy shape.
    if (section("pinhole-migration")) {
        newScene();
        step(2);

        auto host = ObjectFactory::createPrimitive(Primitive::Sphere, document_.scene());
        host->name = "Legacy Eye";
        host->position.set(3.f, 2.f, 0.f);
        host->rotation.x = -0.7f;

        SensorConfig legacy;
        legacy.enabled = true;
        legacy.type = SensorConfig::Type::Depth;
        legacy.fovY = 71.f;
        legacy.nearPlane = 0.25f;
        legacy.farPlane = 17.f;
        legacy.seed = 31;
        legacy.write(*host);
        addObject(host, document_.scene(), "Add Legacy Eye");
        step();

        const auto undoBefore = commands_.undoCount();
        moveSensorToCameraChild(*host);
        step();

        PerspectiveCamera* child = nullptr;
        for (const auto& candidate : host->children) {
            if (auto* camera = candidate->as<PerspectiveCamera>()) child = camera;
        }
        check(child != nullptr, "migration hangs a camera child under the legacy host");
        check(!SensorConfig::read(*host).has_value(), "and the host keeps no sensor entry");
        if (child) {
            const auto moved = SensorConfig::read(*child);
            check(moved && moved->enabled && moved->type == SensorConfig::Type::Depth &&
                          moved->seed == 31,
                  "the config crossed whole");
            check(std::abs(child->fov - 71.f) < 1e-4f &&
                          std::abs(child->nearPlane - 0.25f) < 1e-4f &&
                          std::abs(child->farPlane - 17.f) < 1e-4f,
                  "and the camera frustum is stamped from the authored numbers");
            check(child->position.length() < 1e-6f,
                  "the child sits at the host's origin, keeping the authored aim");
        }
        check(commands_.undoCount() == undoBefore + 1,
              "the move is ONE undo step, not three");

        commands_.undo();
        step();
        check(host->children.empty(), "undo removes the camera child");
        const auto restored = SensorConfig::read(*host);
        check(restored && restored->enabled && restored->type == SensorConfig::Type::Depth &&
                      std::abs(restored->fovY - 71.f) < 1e-4f,
              "and puts the sensor back on the host, numbers intact");
    }

    // ------------------------------------------------ scene-root userData
    //
    // Whether an authoring rule can live ON the scene rather than on a node
    // depends entirely on this: the root is serialised as an object of type
    // "Scene", so its userData should round-trip like any other node's. Play
    // serialises and Stop reloads, which is the same path a save and open take,
    // so this answers the question for both. Multi-line values are the case
    // that matters — an inline script is newlines and quotes, not a scalar.
    if (section("scene-userdata")) {
        newScene();
        step(2);

        const std::string source = "count = 12\nfor i in range(count):\n    pass\n";
        document_.scene().userData["generatorSource"] = source;
        document_.scene().userData["generatorFields"] = std::string("count=12");

        startPlay();
        step(4);
        stopPlay();
        step(4);

        const auto readBack = [this](const char* key) {
            const auto it = document_.scene().userData.find(key);
            if (it == document_.scene().userData.end()) return std::string{};
            if (it->second.type() != typeid(std::string)) return std::string{};
            return std::any_cast<const std::string&>(it->second);
        };
        check(readBack("generatorSource") == source,
              "multi-line userData on the scene root survives a save and reload");
        check(readBack("generatorFields") == "count=12", "and so do its fields");

        // And the same thing through GeneratorConfig, which is what will actually
        // carry it: read back after the reload, since that is the case that
        // decides whether a generator can live on the scene at all.
        const auto restored = GeneratorConfig::read(document_.scene());
        check(restored.has_value(), "GeneratorConfig reads a reloaded scene generator");
        check(restored && restored->source == source, "with its source verbatim");
        check(restored && restored->field("count") == "12", "and its exposed parameter");
        check(GeneratorConfig::isGenerator(document_.scene()), "and reports as a generator");

        // A generator on a plain Group, since the config is object-agnostic.
        auto scoped = Group::create();
        scoped->name = "Scoped Generator";
        GeneratorConfig scopedConfig;
        scopedConfig.source = "pass\n";
        scopedConfig.setField("n", "3");
        scopedConfig.write(*scoped);
        document_.scene().add(scoped);
        step();
        check(GeneratorConfig::isGenerator(*scoped), "a Group can carry one too");
        check(GeneratorConfig::generatedChild(*scoped) == nullptr,
              "with no output before it has run");

        // The tagged-output contract: whichever child carries the mark is the one
        // a regenerate replaces.
        auto output = Group::create();
        output->userData[GeneratorConfig::generatedKey] = std::string("1");
        scoped->add(output);
        step();
        check(GeneratorConfig::generatedChild(*scoped) == output.get(),
              "and finds its tagged output once there is one");

        // Clearing leaves no trace, so a document does not carry a dead key.
        GeneratorConfig::erase(*scoped);
        check(!GeneratorConfig::isGenerator(*scoped), "erasing a generator clears it");
        check(scoped->userData.find(GeneratorConfig::fieldsKey) == scoped->userData.end(),
              "fields included");

        GeneratorConfig::erase(document_.scene());
    }

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // ------------------------------------------------ generator, actually run
    //
    // The whole point, driven through the real button path: author a script on
    // the scene, regenerate, and check that what the script built is in the
    // document as ordinary content.
    if (section("generator")) {
        newScene();
        step(2);

        GeneratorConfig config;
        config.source =
                "import threepp\n"
                "from threepp import editor\n"
                "for i in range(4):\n"
                "    m = threepp.Mesh(threepp.BoxGeometry(1, 1, 1),\n"
                "                     threepp.MeshBasicMaterial())\n"
                "    m.name = \"Gen %d\" % i\n"
                "    m.position.set(i * 2.0, 0.5, 0.0)\n"
                "    editor.add(m)\n";
        config.write(document_.scene());

        const int before = static_cast<int>(document_.scene().children.size());
        check(regenerate(document_.scene()), "a generator script runs");

        auto* output = GeneratorConfig::generatedChild(document_.scene());
        check(output != nullptr, "and leaves a tagged output node");
        check(output && output->children.size() == 4, "holding what the script built");
        check(document_.scene().getObjectByName("Gen 2") != nullptr,
              "as findable scene content");
        check(static_cast<int>(document_.scene().children.size()) == before + 1,
              "under one new child, not four loose ones");

        // Re-running REPLACES rather than accumulates — the failure here would be
        // 8 boxes, which is what makes a generator unusable.
        check(regenerate(document_.scene()), "it runs again");
        auto* second = GeneratorConfig::generatedChild(document_.scene());
        check(second && second->children.size() == 4, "replacing its output, not adding to it");
        check(static_cast<int>(document_.scene().children.size()) == before + 1,
              "still one output node");

        // One undo step, and it puts the PREVIOUS generation back rather than
        // leaving the scene empty.
        check(commands_.canUndo(), "a regenerate is undoable");
        commands_.undo();
        step();
        auto* restored = GeneratorConfig::generatedChild(document_.scene());
        check(restored != nullptr && restored != second,
              "undo restores the previous generation");
        check(restored && restored->children.size() == 4, "with its content");

        // A script that raises must commit NOTHING. The scene keeps the output it
        // had, and the console carries the reason.
        const auto* keep = GeneratorConfig::generatedChild(document_.scene());
        GeneratorConfig broken;
        broken.source = "import threepp\nraise ValueError(\"nope\")\n";
        broken.write(document_.scene());
        check(!regenerate(document_.scene()), "a raising script fails");
        check(GeneratorConfig::generatedChild(document_.scene()) == keep,
              "and changes nothing in the document");

        // A half-built script is the case the detached-sink design exists for:
        // it adds, THEN raises.
        GeneratorConfig halfway;
        halfway.source =
                "import threepp\n"
                "from threepp import editor\n"
                "editor.add(threepp.Mesh(threepp.BoxGeometry(1, 1, 1),\n"
                "                        threepp.MeshBasicMaterial()))\n"
                "raise RuntimeError(\"halfway\")\n";
        halfway.write(document_.scene());
        check(!regenerate(document_.scene()), "a script that adds then raises fails");
        check(GeneratorConfig::generatedChild(document_.scene()) == keep,
              "and commits none of what it had added");

        // Clearing takes the OUTPUT with it. Leaving it behind was a real defect:
        // orphaned content still carrying the generated tag, which the next
        // generator on this object would silently adopt and replace.
        commands_.redo();// back to the second generation
        step();
        check(GeneratorConfig::generatedChild(document_.scene()) != nullptr,
              "there is output to clear");
        const int childrenWithOutput = static_cast<int>(document_.scene().children.size());
        clearGenerator(document_.scene());
        step();
        check(GeneratorConfig::generatedChild(document_.scene()) == nullptr,
              "clearing a generator removes its output");
        check(!GeneratorConfig::isGenerator(document_.scene()), "and the script with it");
        check(static_cast<int>(document_.scene().children.size()) == childrenWithOutput - 1,
              "leaving nothing behind in the scene");

        // ONE undo step brings back both halves, not just one of them.
        commands_.undo();
        step();
        check(GeneratorConfig::isGenerator(document_.scene()), "one undo restores the script");
        check(GeneratorConfig::generatedChild(document_.scene()) != nullptr, "and its output");

        // Refused while playing, like every other document edit.
        GeneratorConfig::erase(document_.scene());
        config.write(document_.scene());
        startPlay();
        step(2);
        check(!regenerate(document_.scene()), "regenerate is refused while playing");
        stopPlay();
        step(2);

        // The VS Code loop: a save in the external editor syncs the source AND
        // re-runs it, so the scene follows the file without a button press. The
        // launch itself is suppressed in the self-test; everything after it is the
        // real path.
        {
            GeneratorConfig::erase(document_.scene());
            GeneratorConfig two;
            two.source =
                    "import threepp\n"
                    "from threepp import editor\n"
                    "for i in range(2):\n"
                    "    editor.add(threepp.Mesh(threepp.BoxGeometry(1, 1, 1),\n"
                    "                            threepp.MeshBasicMaterial()))\n";
            two.write(document_.scene());
            check(regenerate(document_.scene()), "a two-object generator runs");

            startExternalEdit(document_.scene(), ExternalEditKind::Generator);
            check(externalEdit_.active, "an external session starts on a generator");
            check(externalEditActive(document_.scene()), "and reports as this object's");
            const auto scratch = externalEdit_.file;
            check(ScriptWorkspace::readSource(scratch) == two.source,
                  "exporting the generator source byte for byte");

            // Five objects instead of two: enough that "did the file win" cannot
            // be confused with "did anything happen".
            const auto edited = std::string(
                    "import threepp\n"
                    "from threepp import editor\n"
                    "for i in range(5):\n"
                    "    editor.add(threepp.Mesh(threepp.BoxGeometry(1, 1, 1),\n"
                    "                            threepp.MeshBasicMaterial()))\n");
            const int before = externalEdit_.syncs;
            ScriptWorkspace::writeSource(scratch, edited);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (externalEdit_.syncs <= before && std::chrono::steady_clock::now() < deadline) {
                step();
            }
            check(externalEdit_.syncs > before, "saving the file syncs it back");
            const auto synced = GeneratorConfig::read(document_.scene());
            check(synced && synced->source.find("range(5)") != std::string::npos,
                  "the document takes the edited source");

            // The re-run is deferred a frame, so give it one.
            step(3);
            auto* regrown = GeneratorConfig::generatedChild(document_.scene());
            check(regrown && regrown->children.size() == 5,
                  "and the scene re-runs it without a button press");

            // A file that does not parse syncs but must NOT run — the previous
            // output stands rather than being replaced by nothing.
            ScriptWorkspace::writeSource(scratch, "def broken(\n");
            const int beforeBad = externalEdit_.syncs;
            const auto badDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (externalEdit_.syncs <= beforeBad &&
                   std::chrono::steady_clock::now() < badDeadline) {
                step();
            }
            step(3);
            auto* kept = GeneratorConfig::generatedChild(document_.scene());
            check(kept && kept->children.size() == 5,
                  "a file that does not parse leaves the last good output alone");

            stopExternalEdit("selftest done");
            check(!externalEdit_.active, "the session stops");

            GeneratorConfig::erase(document_.scene());
            config.write(document_.scene());
            check(regenerate(document_.scene()), "and the scene regenerates afterwards");
        }

        // And the generated content survives the round trip, because it is
        // ordinary scene content and nothing re-runs on load.
        auto* afterStop = GeneratorConfig::generatedChild(document_.scene());
        check(afterStop != nullptr, "generated output survives play/stop");
        check(afterStop && afterStop->children.size() == 4, "with its objects");
        check(GeneratorConfig::isGenerator(document_.scene()),
              "and the script is still on the scene");
    }
#endif

    // ---------------------------------------------------------- instancing
    //
    // An InstancedMesh is one object standing in for N, so the two things worth
    // pinning are that a pick can tell WHICH of the N it landed on, and that the
    // selection outline then boxes that one instead of the whole cloud. No
    // PhysX: this block sits outside the gate above so it runs in every build.
    if (section("instancing")) {
        newScene();
        step(2);

        constexpr int kInstances = 3;
        constexpr float kSpacing = 4.f;
        constexpr float kHeight = 10.f;// clear of the template Ground

        auto instanced = InstancedMesh::create(
                BoxGeometry::create(1, 1, 1), MeshBasicMaterial::create(), kInstances);
        instanced->name = "Instances";
        for (int i = 0; i < kInstances; ++i) {
            Matrix4 m;
            m.setPosition((static_cast<float>(i) - 1.f) * kSpacing, kHeight, 0.f);
            instanced->setMatrixAt(static_cast<std::size_t>(i), m);
        }
        instanced->instanceMatrix()->needsUpdate();
        document_.scene().add(instanced);
        step(2);

        check(instanced->count() == kInstances, "an InstancedMesh joins the scene");

        // Per-instance hit reporting: straight down onto the LAST instance. The
        // instances are 4 apart and 1 wide, so nothing else is under this ray.
        raycaster_.set({kSpacing, kHeight + 10.f, 0.f}, {0.f, -1.f, 0.f});
        auto hits = raycaster_.intersectObject(*instanced);
        const bool hitLast = !hits.empty() && hits.front().instanceId &&
                             *hits.front().instanceId == kInstances - 1;
        check(hitLast, "a ray reports which instance it hit");

        // The whole point: the outline follows the picked instance, and there is
        // exactly ONE of them (the cloud box must not also be standing there).
        selectObject(instanced.get(), kInstances - 1);
        step();
        check(selection_.get() == instanced.get(), "selecting an instance selects the mesh");
        check(selectedInstance_ && *selectedInstance_ == kInstances - 1,
              "and remembers which instance was picked");
        check(outlineCount() == 1, "with exactly one outline");
        const auto lastCentre = instanceBox_.getCenter();
        check(std::abs(lastCentre.x - kSpacing) < 0.1f &&
                      std::abs(lastCentre.y - kHeight) < 0.1f,
              "boxing that instance, not the cloud");
        // A 3-instance row spans 2*kSpacing+1; one instance is 1 across. Boxing
        // the cloud instead of the instance is exactly the bug this catches.
        check(instanceBox_.getSize().x < kSpacing, "so the box is instance-sized");

        // Picking a different instance moves the box rather than adding one.
        selectObject(instanced.get(), 0);
        step();
        const auto firstCentre = instanceBox_.getCenter();
        check(std::abs(firstCentre.x + kSpacing) < 0.1f, "picking another instance moves the box");
        check(outlineCount() == 1, "and still leaves one outline");

        // Selected from the hierarchy there is no instance to speak of, so the
        // honest answer is the whole-object outline.
        selectObject(instanced.get());
        step();
        check(!selectedInstance_, "selecting the mesh alone picks no instance");
        check(outlineCount() == 1, "and outlines the whole cloud once");

        // An out-of-range index is a caller bug, not a crash or a stale box.
        selectObject(instanced.get(), kInstances + 5);
        step();
        check(!selectedInstance_, "an out-of-range instance falls back to the whole mesh");

        // A plain Mesh must never come back carrying an instance index.
        if (auto* templateBox = document_.scene().getObjectByName("Box")) {
            selectObject(templateBox, 1);
            step();
            check(!selectedInstance_, "a plain Mesh reports no instance");
        }

        // Instances have to SURVIVE. Play serialises the document and Stop
        // reloads it, so a play/stop round trip is the editor driving its own
        // save-and-open path — if an InstancedMesh comes back as N lost objects
        // or as one box, an imported instanced asset would not survive a save
        // either. Assert the count AND a specific instance's translation: a
        // count that round-trips over a zeroed instanceMatrix is the failure
        // this catches.
        selectObject(nullptr);
        startPlay();
        step(4);
        stopPlay();
        step(4);
        auto* reloaded = document_.scene().getObjectByName("Instances");
        auto* reloadedInstanced = reloaded ? reloaded->as<InstancedMesh>() : nullptr;
        check(reloadedInstanced != nullptr, "an InstancedMesh survives play/stop as one");
        if (reloadedInstanced) {
            check(reloadedInstanced->count() == kInstances, "with every instance still there");
            Matrix4 restored;
            reloadedInstanced->getMatrixAt(kInstances - 1, restored);
            const auto p = Vector3().setFromMatrixPosition(restored);
            check(std::abs(p.x - kSpacing) < 1e-3f && std::abs(p.y - kHeight) < 1e-3f,
                  "and the instance transforms intact");
        }

        // Scene replace while an instance is outlined: the index pointed into a
        // mesh that is being freed, and the overlay outlives the swap.
        if (reloadedInstanced) selectObject(reloadedInstanced, 1);
        step();
        newScene();
        step(2);
        check(!selectedInstance_, "a scene replace drops the instance index");
        check(instanceOutline_ == nullptr, "and the instance outline with it");
    }

    // --- File ▸ Open Example --------------------------------------------------
    // The shipped scene, opened through the SAME call the menu makes, and then
    // flown. tests/extras/EditorExampleScene_test.cpp asserts the document and
    // the controller headlessly; what only this pass can see is the app half —
    // the open path, the untitled document, and a play session that actually
    // runs the sensors, because a vision sensor needs a renderer to scan with.
    if (section("open-example")) {
        newScene();
        step(2);
        check(!followSelection(), "a new document follows nothing");
        openExample("hover-arena");

        // Asserted before a single frame is drawn: the open path places the
        // camera, and after that the chase below is entitled to move it.
        {
            const auto& userData = document_.scene().userData;
            const auto it = userData.find("editorView");
            const auto spec = it != userData.end() && it->second.type() == typeid(std::string)
                                      ? std::any_cast<const std::string&>(it->second)
                                      : std::string();
            Vector3 wantPosition;
            Vector3 wantTarget;
            check(parseViewSpec(spec, wantPosition, wantTarget),
                  "the example authors an editorView");
            check(camera_.position.distanceTo(wantPosition) < 1e-3f,
                  "and opening puts the camera where it says");
            check(orbit_->target.distanceTo(wantTarget) < 1e-3f, "aimed where it says");
            check(!orthographic(), "in the projection an authored vantage is written for");
        }
        step(2);

        auto* drone = document_.scene().getObjectByName("Drone");
        check(drone != nullptr, "Open Example loads Hover Arena");
        check(!document_.hasPath(), "and leaves the document untitled");
        check(!document_.dirty(), "and clean, so Save prompts for a path");
        check(document_.scene().getObjectByName("Arena Floor") != nullptr,
              "with the generator's committed output in it");
        // Ready to fly: the document names what the viewport chases, so Play is
        // a chase cam without anyone reaching for the View menu first.
        check(selection_.get() == drone, "and opens with the Drone selected");
        check(followSelection(), "and Follow Selection on");

        openExample("no-such-example");
        step();
        check(document_.scene().getObjectByName("Drone") != nullptr,
              "an unknown example is a console line, not a lost scene");

        if (drone) {
            const float before = drone->position.y;
            startPlay();
            // stepFixed, not step: this asserts on where a controller SETTLES,
            // which is accumulated simulated time (see the note above).
            stepFixed(120);

            auto* playing = document_.scene().getObjectByName("Drone");
            check(playing != nullptr, "the drone survives Play");
            if (playing) {
                Vector3 world;
                playing->getWorldPosition(world);
                // It is holding a 2.2 m hover over a floor whose top is y = 0,
                // hands off, on the noisy IMU. Anything outside this band is a
                // controller that has stopped controlling. The controller is one
                // of the example's Python scripts, so a Python-less build has
                // nothing flying the drone and the band is not a promise it can
                // make — same guard as the script checks just below.
#ifdef THREEPP_EDITOR_WITH_PYTHON
                check(world.y > 1.4f && world.y < 3.2f, "and holds its hover band");
                check(std::abs(world.y - before) < 1.5f, "without drifting off its start height");
#else
                (void) world;
                (void) before;
#endif
            }
#ifdef THREEPP_EDITOR_WITH_PYTHON
            check(scripts_ && scripts_->errorCount() == 0, "with no script errors");
            check(scripts_ && scripts_->instanceCount() == 7, "and all seven scripts live");
#endif
            // The sensors are the other half of the example, and the only half
            // the headless test cannot reach: a LIDAR scan needs a renderer.
            std::size_t samples = 0;
            std::size_t scans = 0;
            if (sensors_) {
                for (const auto& entry : sensors_->entries()) {
                    samples += entry->samples;
                    scans += entry->scans;
                }
            }
            check(sensors_ && sensors_->sensorCount() == 2, "two sensors authored on the drone");
            check(samples > 0, "the IMU is measuring");
            // Rate-gated at 10 Hz off the physics accumulator, so wait rather
            // than assume 120 frames covered a scan on this machine.
            for (int i = 0; i < 200 && scans == 0; ++i) {
                stepFixed(4);
                scans = 0;
                for (const auto& entry : sensors_->entries()) scans += entry->scans;
            }
            check(scans > 0, "and the LIDAR has scanned");

            // Stop restores the scene, NOT the camera. An authored editorView
            // is applied when a document is opened and never again — a Stop
            // that teleported the view back to the document's idea of a good
            // vantage would throw away wherever the user had flown to look.
            // Follow is switched off first so the only thing that could move
            // the camera here is the thing being tested.
            setFollowSelection(false);
            const Vector3 drovePosition(-14.f, 11.f, -19.f);
            const Vector3 droveTarget(-3.f, 1.5f, -6.f);
            camera_.position.copy(drovePosition);
            orbit_->target.copy(droveTarget);
            step(2);

            stopPlay();
            step(2);
            check(document_.scene().getObjectByName("Drone") != nullptr,
                  "and Stop puts the authored scene back");
            check(camera_.position.distanceTo(drovePosition) < 1e-2f &&
                          orbit_->target.distanceTo(droveTarget) < 1e-2f,
                  "leaving the camera where the user drove it");
        }
    }

    // --- Gaussian splat clouds ---------------------------------------------
    // A procedural cloud, so this needs no asset on disk. The three things an
    // editor owes an imported scan: it draws, it can be clicked, and the fact
    // that it is not saved yet is said rather than discovered.
    //
    // stepFixed throughout. Nothing here accumulates simulated time, but Play
    // and Stop do run a session, and the doctrine is cheaper to keep than to
    // decide about per call site.
    if (section("splats")) {
        newScene();
        selectObject(nullptr);

        // Away from the template Box and Ground, and looked at head on, so the
        // before/after difference below is the cloud and only the cloud.
        const Vector3 where(0.f, 4.f, 0.f);
        camera_.position.set(0.f, 4.f, 5.f);
        orbit_->target.copy(where);
        stepFixed(4);

        const auto before = renderer_->readRGBPixels();
        check(!before.empty(), "the frame can be read back at all");

        SplatGenerator::Options options;
        options.count = 3000;
        options.shDegree = 1;
        auto cloud = SplatCloud::create(SplatGenerator::generate(options));
        cloud->name = "Procedural Splats";
        cloud->position.copy(where);

        // Through the same command the import uses, so this exercises the undo
        // entry and the hierarchy insert rather than a bare scene->add.
        addObject(cloud, document_.scene(), "Add Splats");
        // The add selects what it added; drop the selection so the outline is
        // not counted as splat pixels.
        selectObject(nullptr);
        stepFixed(4);

        check(document_.scene().getObjectByName("Procedural Splats") != nullptr,
              "a splat cloud goes into the scene like any other object");

        // Pixels, because nothing else answers "does it draw". Count of pixels
        // that MOVED, not a frame mean: 3000 splats two metres across are a
        // small part of the frame and an average would bury them.
        const auto after = renderer_->readRGBPixels();
        std::size_t moved = 0;
        const std::size_t n = std::min(before.size(), after.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (std::abs(static_cast<int>(after[i]) - static_cast<int>(before[i])) > 16) ++moved;
        }
        check(after.size() == before.size() && moved > n / 500,
              "and it renders (nonzero splat pixels)");
        std::cout << "[selftest] splat pixels moved: " << moved << " of " << n << std::endl;

        // Clickable. Without SplatCloud::raycast this picks nothing: the
        // inherited InstancedMesh version tests the unit quad against 3000
        // identity matrices, none of which is where a splat is.
        const auto* viewport = ImGui::GetMainViewport();
        pickAt(viewport->Pos.x + viewport->Size.x * 0.5f,
               viewport->Pos.y + viewport->Size.y * 0.5f);
        stepFixed();
        check(selection_.get() == cloud.get(), "a click in the viewport selects the cloud");

        // The gizmo needs a bound to sit on, and the cloud's must be its own
        // rather than the unit quad InstancedMesh would compute.
        Box3 bounds;
        bounds.setFromObject(*cloud);
        check(!bounds.isEmpty() && bounds.getSize().x > 1.f,
              "with a bounding box the size of the cloud, not of the quad");

        // The import mark round-trips through userData.
        editor::SplatImportConfig config;
        config.source = "C:/scans/procedural.ply";
        config.culled = true;
        config.removed = 7;
        config.flippedX = true;
        config.write(*cloud);
        const auto read = editor::SplatImportConfig::read(*cloud);
        check(read && read->source == config.source && read->culled && read->removed == 7 &&
                      read->flippedX,
              "the import mark round-trips through userData");
        check(read && read->lod == -1,
              "and a source with no detail levels reads back as lod -1, not lod 0");

        // A SOG asset records which level it came from, and -1 is reserved for
        // "this source has no levels" so a re-import can tell the two apart.
        editor::SplatImportConfig sog;
        sog.source = "C:/scans/scan.zip";
        sog.flippedX = true;
        sog.lod = 0;
        sog.write(*cloud);
        const auto readSog = editor::SplatImportConfig::read(*cloud);
        check(readSog && readSog->flippedX && readSog->lod == 0,
              "a SOG import mark records which detail level it read");
        config.write(*cloud);

        // Re-selected, and with the console out of the way, so the picture
        // shows the whole Splats section — the source path and the
        // saved-by-reference note sit below the fold otherwise, and those are
        // the two lines this pass exists to put on screen.
        selectObject(cloud.get());
        const bool consoleWasOpen = bottomPanelOpen_;
        bottomPanelOpen_ = false;
        stepFixed(2);
        shootTo(std::filesystem::temp_directory_path() / "threepp_editor_splats.png");
        bottomPanelOpen_ = consoleWasOpen;
        stepFixed();

        // Play/Stop restores from a snapshot in the same format the save file
        // uses. The snapshot writes the cloud as a reference and keeps the
        // live object (SceneSnapshot + ObjectLoader::setSplatCloudResolver),
        // so Stop hands back the SAME cloud, re-placed, with no reload.
        const Vector3 authoredPosition = cloud->position;
        cloud->setPointMix(0.6f);
        startPlay();
        stepFixed(4);
        cloud->position.x += 3.f;// what a session might do to it
        cloud->setPointMix(0.f);
        stopPlay();
        stepFixed(4);

        auto* survivor = document_.scene().getObjectByName("Procedural Splats");
        check(survivor != nullptr && survivor->as<SplatCloud>() != nullptr,
              "the cloud survives Play/Stop");
        check(survivor == cloud.get(),
              "as the same live object, not a reload");
        check(survivor && survivor->position.equals(authoredPosition),
              "with its authored placement restored");
        check(survivor && survivor->as<SplatCloud>()->pointMix() == 0.6f,
              "and its point-mode look restored");
        check(survivor && editor::SplatImportConfig::read(*survivor).has_value(),
              "and its import mark still on it");
    }

    // A scan becomes a FLOOR. The cloud carries a surface-bake config, Play
    // fuses it into a triangle mesh (plans/splat-surface-bake.md), cooks that
    // into the PhysX world and hangs the sensor-only twin off the scene root —
    // and a dropped ball lands on the scan rather than through it. The scan is
    // a flat slab of splats, so "landed on it" is a NUMBER: the ball's rest
    // height against the fused plane's own mean, within a voxel, which is the
    // accuracy the whole bake is quoted at.
    //
    // Vulkan-only, asserted in both directions like the particle-field gate: the
    // depth AOV the fusion reads is a Vulkan G-buffer attachment, so a GL
    // session must author the config and bake NOTHING.
#if defined(THREEPP_EDITOR_WITH_PHYSX) && defined(THREEPP_WITH_VULKAN)
    if (section("splat-surface")) {
        newScene();
        selectObject(nullptr);

        // The synthetic plane VulkanSplatSurface_test bakes, at the height its
        // ball is dropped from — 4 m of clear air under it, so nothing in the
        // template scene is what the ball is resting on.
        SplatGenerator::Options options;
        options.count = 40000;
        options.seed = 7u;
        options.shDegree = 0;
        options.extent.set(4.f, 0.02f, 4.f);
        options.minScale = 0.02f;
        options.maxScale = 0.035f;
        options.anisotropy = 1.2f;
        options.minOpacity = 0.85f;
        options.maxOpacity = 1.f;
        auto scanData = SplatGenerator::generate(options);
        for (std::size_t i = 0; i < scanData.count(); ++i) {
            scanData.setDcColor(i, Vector3{0.8f, 0.8f, 0.8f});
        }
        auto scan = SplatCloud::create(std::move(scanData));
        scan->name = "Scan Floor";
        scan->position.set(0.f, 4.f, 0.f);
        addObject(scan, document_.scene(), "Add Splats");
        selectObject(nullptr);

        editor::SplatSurfaceConfig surfaceConfig;
        surfaceConfig.enabled = true;
        surfaceConfig.voxelSize = 0.05f;
        surfaceConfig.poseCount = 16;// the pose count the P1 measurements used
        surfaceConfig.write(*scan);
        const auto readBack = editor::SplatSurfaceConfig::read(*scan);
        check(readBack && *readBack == surfaceConfig,
              "the surface-bake config round-trips on the splat node");

        // The pose-set flag, which the plane below must NOT carry: this scan is
        // a floor seen from outside, and an interior bake of it would stand the
        // cameras in the slab. Round-trip only.
        auto interiorConfig = surfaceConfig;
        interiorConfig.interior = true;
        const auto interiorBack = editor::SplatSurfaceConfig::decode(interiorConfig.encode());
        check(interiorBack && *interiorBack == interiorConfig && interiorConfig != surfaceConfig,
              "the Interior pose-set flag round-trips through the config codec");

        // The direct point route (splats::buildPointSurface): a Gaussian scan
        // resolves Auto to the fusion bake, and forcing Points meshes its splat
        // centres on the CPU with no renderer involved.
        {
            auto pointsConfig = surfaceConfig;
            pointsConfig.method = editor::SplatSurfaceConfig::Points;
            const auto pointsBack = editor::SplatSurfaceConfig::decode(pointsConfig.encode());
            check(pointsBack && *pointsBack == pointsConfig,
                  "the surface Method round-trips through the config codec");
            check(!editor::SplatSurfaceCache::usesPointRoute(*scan, surfaceConfig) &&
                          editor::SplatSurfaceCache::usesPointRoute(*scan, pointsConfig),
                  "Auto takes the fusion bake for a Gaussian scan; Points forces the direct route");
            check(editor::SplatSurfaceCache::availableFor(nullptr, *scan, pointsConfig),
                  "the direct route needs no renderer");

            const auto bakesBefore = splatSurfaces_->bakeCount();
            std::string problem;
            const auto* direct = splatSurfaces_->bake(nullptr, *scan, pointsConfig, &problem);
            if (!direct) std::cout << "[selftest] direct point surface failed: " << problem << std::endl;
            check(direct != nullptr && !direct->empty() && direct->stats.poses == 0,
                  "the direct route meshes the scan's points without a renderer");
            check(splatSurfaces_->bakeCount() == bakesBefore + 1 &&
                          splatSurfaces_->find(*scan, pointsConfig) == direct,
                  "and the memo caches it under its own config key");
            if (direct) {
                std::cout << "[selftest] direct point surface: " << direct->triangleCount()
                          << " triangles at " << direct->stats.voxelSize << " m from "
                          << direct->stats.observedVoxels << " voxels" << std::endl;
                // The scan is a floor slab at y = 4: the surface must sit there.
                check(direct->stats.aabbMin.y > 3.f && direct->stats.aabbMax.y < 5.f,
                      "and the surface sits where the scan is");
            }
        }

        auto ball = Mesh::create(SphereGeometry::create(0.15f, 16, 12),
                                 MeshStandardMaterial::create());
        ball->name = "Scan Ball";
        ball->position.set(0.1f, 5.4f, -0.1f);
        PhysicsConfig ballPhysics;
        ballPhysics.enabled = true;
        ballPhysics.body = PhysicsConfig::Body::Dynamic;
        ballPhysics.shape = PhysicsConfig::Shape::Sphere;
        ballPhysics.restitution = 0.f;// it has 4 s to settle, not to bounce
        ballPhysics.write(*ball);
        addObject(ball, document_.scene(), "Add Ball");
        selectObject(nullptr);
        step(2);

        const bool bakeable = editor::SplatSurfaceCache::available(renderer_.get());

        // The edit-mode preview: the inspector's "Show surface" puts the memo's
        // triangles in the viewport as editor chrome (SplatSurfaceOverlay.cpp).
        // Driven by the same two steps the checkbox takes — set the view flag,
        // bake once through the memo Play is about to hit.
        if (bakeable) {
            splatSurfacePreview_ = true;
            bakeSplatSurface(*scan);
            step(2);

            const auto drawn = splatSurfacePreviews_.find(scan->uuid);
            std::size_t previewTris = 0;
            std::size_t segments = 0;
            if (drawn != splatSurfacePreviews_.end() && drawn->second.mesh) {
                previewTris = drawn->second.triangles;
                if (const auto* position =
                            drawn->second.mesh->geometry()->getAttribute<float>("position")) {
                    segments = static_cast<std::size_t>(position->count()) / 2;
                }
            }
            std::size_t memoTris = 0;
            if (const auto* mesh = splatSurfaces_->find(*scan, surfaceConfig)) {
                memoTris = mesh->triangleCount();
            }
            std::cout << "[selftest] surface preview: " << previewTris << " triangles as "
                      << segments << " edges, memo " << memoTris << ", bakes "
                      << splatSurfaces_->bakeCount() << std::endl;
            check(memoTris > 0 && previewTris == memoTris && segments > memoTris,
                  "Show surface draws the memo's baked triangles over the scan");

            const auto bakes = splatSurfaces_->bakeCount();
            step(2);
            check(splatSurfaces_->bakeCount() == bakes,
                  "and the preview asks the memo every frame rather than baking again");

            // The picture the feature exists for: the wireframe standing on the
            // scan it was fused from. shootTo photographs overlay children (it
            // is how the physics debug lines are photographed above), and the
            // OFF shot at the same pose is its control — a scan is a mass of
            // white splats, so "the wireframe is visible" is a claim that has to
            // be checkable against the same frame without it.
            {
                const bool bottomPanelWas = bottomPanelOpen_;
                bottomPanelOpen_ = false;
                camera_.position.set(5.f, 7.5f, 7.f);
                orbit_->target.set(0.f, 4.f, 0.f);
                step(4);
                shootTo(std::filesystem::temp_directory_path() / "threepp_editor_splat_surface.png");
                if (const auto shot = splatSurfacePreviews_.find(scan->uuid);
                    shot != splatSurfacePreviews_.end() && shot->second.mesh) {
                    const auto geometry = shot->second.mesh->geometry();
                    geometry->computeBoundingSphere();
                    const auto sphere = *geometry->boundingSphere;
                    std::cout << "[selftest] preview node: visible " << shot->second.mesh->visible
                              << ", parent " << (shot->second.mesh->parent
                                                         ? shot->second.mesh->parent->name
                                                         : std::string("none"))
                              << ", bounds c(" << sphere.center.x << "," << sphere.center.y << ","
                              << sphere.center.z << ") r " << sphere.radius << std::endl;
                }
                splatSurfacePreview_ = false;
                step(2);
                shootTo(std::filesystem::temp_directory_path() / "threepp_editor_splat_surface_off.png");
                splatSurfacePreview_ = true;
                step(2);
                bottomPanelOpen_ = bottomPanelWas;
            }

            splatSurfacePreview_ = false;
            step(1);
            check(splatSurfacePreviews_.empty(),
                  "and toggling it off takes the wireframe back out of the viewport");
            splatSurfacePreview_ = true;// left on, so the Play below has one to hide
            step(1);
        }

        // A second scan as a MOVING body: a shell of points, meshed by the
        // direct route (no renderer), split into convex hulls at Play and
        // dropped onto the floor scan. The cloud must follow its actor down.
        std::shared_ptr<SplatCloud> rock;
        {
            SplatData rockData;
            constexpr int kRockPoints = 1500;
            rockData.resize(kRockPoints, 0);
            const float golden = math::PI * (3.f - std::sqrt(5.f));
            for (int i = 0; i < kRockPoints; ++i) {
                const float y = 1.f - 2.f * (static_cast<float>(i) + 0.5f) / kRockPoints;
                const float r = std::sqrt(std::max(0.f, 1.f - y * y));
                const float a = golden * static_cast<float>(i);
                rockData.means[i].set(0.25f * r * std::cos(a), 0.25f * y, 0.25f * r * std::sin(a));
                rockData.scales[i].set(0.01f, 0.01f, 0.01f);
                rockData.opacities[i] = 1.f;
                rockData.setDcColor(static_cast<std::size_t>(i), Vector3{0.6f, 0.4f, 0.3f});
            }
            rock = SplatCloud::create(std::move(rockData));
            rock->name = "Scan Rock";
            rock->position.set(-1.f, 5.2f, 0.4f);
            addObject(rock, document_.scene(), "Add Rock Scan");
            selectObject(nullptr);

            editor::SplatSurfaceConfig rockConfig;
            rockConfig.enabled = true;
            rockConfig.method = editor::SplatSurfaceConfig::Points;
            rockConfig.body = editor::SplatSurfaceConfig::Dynamic;
            rockConfig.mass = 2.f;
            rockConfig.hulls = 8;
            rockConfig.write(*rock);
            const auto rockBack = editor::SplatSurfaceConfig::read(*rock);
            check(rockBack && *rockBack == rockConfig,
                  "the moving-body fields round-trip on the rock scan");
        }
        const double rockStartY = rock->position.y;
        const auto bakesBeforePlay = splatSurfaces_ ? splatSurfaces_->bakeCount() : 0;

        startPlay();
        stepFixed(240);

        {
            const double rockY = rock->position.y;
            std::cout << "[selftest] rock scan: " << (splatSurfaceSession_ ? splatSurfaceSession_->movingCount() : 0)
                      << " moving body, fell from y " << rockStartY << " to " << rockY << std::endl;
            check(splatSurfaceSession_ && splatSurfaceSession_->movingCount() == 1,
                  "Play cooks the rock scan into a dynamic compound of convex hulls");
            check(rockY < rockStartY - 0.3,
                  "and the cloud follows its body down");
            if (bakeable) {
                // Floor slab top at ~4.05, rock radius 0.25 plus its own offset
                // skin: it rests around 4.35 and must not have gone through.
                check(rockY > 3.8 && rockY < 4.7,
                      "and comes to rest on the floor scan rather than through it");
            }
        }

        if (bakeable) {
            check(splatSurfaceSession_ && splatSurfaceSession_->surfaceCount() == 2 &&
                          splatSurfaceSession_->colliderCount() == 2,
                  "Play bakes both scans: one static collider, one dynamic");

            // The plane the ball is graded against is the very mesh the session
            // used — read back out of the memo, not re-derived.
            double meanY = 0.0;
            std::size_t verts = 0;
            if (const auto* mesh = splatSurfaces_->find(*scan, surfaceConfig)) {
                for (std::size_t i = 1; i < mesh->positions.size(); i += 3) {
                    meanY += mesh->positions[i];
                    ++verts;
                }
                if (verts) meanY /= static_cast<double>(verts);
                std::cout << "[selftest] baked scan: " << mesh->triangleCount()
                          << " tris, render " << mesh->stats.renderMs << " ms, fuse "
                          << mesh->stats.fuseMs << " ms, mesh " << mesh->stats.meshMs << " ms"
                          << std::endl;
            }
            const double rest = ball->position.y;
            std::cout << "[selftest] ball rests at y " << rest << " (plane " << meanY
                      << " + r 0.15 -> err " << std::abs(rest - (meanY + 0.15))
                      << ", voxel 0.05)" << std::endl;
            check(verts > 0 && std::abs(rest - (meanY + 0.15)) < 0.05,
                  "and a dropped ball rests on the baked scan, within a voxel");
            check(splatSurfaceSession_ && splatSurfaceSession_->sensorSurfaces(),
                  "the sensor-only master is on while the surface plays");
            check(splatSurfaces_ && splatSurfaces_->bakeCount() == bakesBeforePlay + 1,
                  "the floor came from the memo and only the rock baked at Play");
            check(splatSurfacePreviews_.empty(),
                  "and Play hides the edit-mode preview - its sensor twin is the surface now");
        } else {
            // The floor needs the depth AOV and stays unbaked here; the rock
            // took the direct route and is the one surface this play has.
            check(splatSurfaceSession_ && splatSurfaceSession_->surfaceCount() == 1 &&
                          splatSurfaceSession_->colliderCount() == 1,
                  "without a depth AOV only the direct-route scan is baked");
        }

        stopPlay();
        step(2);

        std::size_t sensorMeshes = 0;
        document_.scene().traverse([&](Object3D& o) {
            if (o.layers.isEnabled(VulkanRenderer::kSensorOnlyLayer)) ++sensorMeshes;
        });
        bool masterOff = true;
        if (auto* vk = dynamic_cast<VulkanRenderer*>(renderer_.get())) {
            masterOff = !vk->sensorOnlySurfaces();
        }
        check(sensorMeshes == 0 && masterOff,
              "and Stop takes the sensor surface out of the scene and the master back off");

        // A view flag survives a scene replace; the cases below author no
        // surfaces, but leaving it on would leave this case's state in theirs.
        splatSurfacePreview_ = false;
        step(1);
    }
#endif

    // Delete a splat cloud, then bring in another one — a user-reported crash.
    // The second cloud is bigger than the first so the import forces the
    // structural resize path (waitIdle + scratch reallocation + descriptor
    // rewrite for every resident cloud), which is the path with the most
    // machinery to get wrong. The checks are that the editor is still alive,
    // that the SECOND cloud is what draws, and that repeating the cycle does
    // not degrade — a residency cache that never evicts fails that last one
    // by running out of slots, silently or otherwise.
    if (section("splat-churn")) {
        newScene();
        selectObject(nullptr);

        const Vector3 where(0.f, 4.f, 0.f);
        camera_.position.set(0.f, 4.f, 5.f);
        orbit_->target.copy(where);
        stepFixed(4);

        const auto empty = renderer_->readRGBPixels();

        auto countMoved = [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
            std::size_t moved = 0;
            const std::size_t n = std::min(a.size(), b.size());
            for (std::size_t i = 0; i < n; ++i)
                if (std::abs(static_cast<int>(b[i]) - static_cast<int>(a[i])) > 16) ++moved;
            return moved;
        };

        for (int cycle = 0; cycle < 3; ++cycle) {

            SplatGenerator::Options options;
            options.count = 3000 + 5000 * (cycle + 1);// each bigger: structural every time
            options.shDegree = 1;
            auto cloud = SplatCloud::create(SplatGenerator::generate(options));
            cloud->name = "Splats " + std::to_string(cycle);
            cloud->position.copy(where);
            addObject(cloud, document_.scene(), "Add Splats");
            selectObject(nullptr);
            stepFixed(4);

            const auto drawn = renderer_->readRGBPixels();
            const auto moved = countMoved(empty, drawn);
            check(moved > empty.size() / 500,
                  ("delete/import cycle " + std::to_string(cycle) + ": the new cloud draws").c_str());

            selectObject(cloud.get());
            deleteSelected();
            selectObject(nullptr);
            stepFixed(4);

            const auto gone = renderer_->readRGBPixels();
            check(countMoved(empty, gone) < empty.size() / 500,
                  ("delete/import cycle " + std::to_string(cycle) +
                   ": and deleting it really clears the frame — a ghost here means "
                   "the pass is drawing a resident copy of a cloud the scene no longer has")
                          .c_str());
        }
    }

    // A real SOG scan, dropped the way a user drops one. Gated on the asset
    // because it is 90 MB at its smallest and nothing in the repo ships it; the
    // generative coverage lives in SogLoader_test, and what this adds is the
    // EDITOR path — the drop handler, the format dispatch and the import mark.
    if (const char* scan = std::getenv("THREEPP_SOG_SCAN"); scan && *scan &&
        section("sog-scan")) {

        // getenv hands back bytes in the ANSI code page on Windows, NOT UTF-8,
        // so the narrow path constructor is the correct one here — decoding
        // them as UTF-8 throws on the first accented character, which is how
        // this was found.
        const std::filesystem::path asset(scan);

        std::error_code ec;
        if (std::filesystem::exists(asset, ec)) {

            const std::size_t before = document_.scene().children.size();

            // The drop handler decodes UTF-8, which is what GLFW hands it — and
            // what a path like ".../Sainte-Anne-de-Beaupré/ssog" needs.
            const auto u8 = asset.u8string();
            handleFileDrop({std::string(u8.begin(), u8.end())});

            // The import runs on a worker; give it room to finish.
            for (int i = 0; i < 4000 && document_.scene().children.size() == before; ++i) stepFixed();

            SplatCloud* imported = nullptr;
            for (const auto& child : document_.scene().children) {
                if (auto* c = child->as<SplatCloud>()) imported = c;
            }

            check(imported != nullptr, "a dropped SOG scan imports as a splat cloud");

            if (imported) {

                std::cout << "[selftest] SOG import: " << imported->splatCount()
                          << " splats, SH degree " << imported->data().shDegree << std::endl;
                check(imported->splatCount() > 0, "with splats in it");

                const auto mark = editor::SplatImportConfig::read(*imported);
                check(mark.has_value(), "and an import mark naming the source");
                // SOG v2 declares +Y up, which tempts you to skip the flip. The
                // container's convention is not the capture's: this scan is a
                // re-encoded COLMAP .ply and comes in upside down without it.
                check(mark && mark->flippedX,
                      "and the same half-turn about X a COLMAP .ply gets");
                check(std::abs(imported->rotation.x - math::PI) < 1e-5f,
                      "which puts the node a half-turn about X");

                // Multi-level import is BACKEND-CONDITIONAL, and each backend
                // has its own contract to assert. Vulkan: several levels
                // resident plus the wiring — after frames run, the policy must
                // have written a non-empty range list (an empty one means the
                // table exists and nothing consumes it, the gap that shipped
                // once already). GL: the guard must have refused — ONE level,
                // no table, no ranges — because the GL path ignores ranges and
                // a multi-level cloud there draws every level stacked.
#ifdef THREEPP_WITH_VULKAN
                const bool vulkanBackend =
                        dynamic_cast<VulkanRenderer*>(renderer_.get()) != nullptr;
#else
                // Without the backend compiled in, VulkanRenderer is a name and
                // not a type — there is nothing to cast to, and no editor built
                // this way can be running it.
                const bool vulkanBackend = false;
#endif
                if (vulkanBackend) {
                    check(!imported->glResourcesBuilt(),
                          "a Vulkan editor never builds the scan's GL-side textures");
                    check(imported->lodTable().levels.size() >= 2,
                          "a multi-level scan keeps several levels resident (Vulkan)");
                    stepFixed(4);
                    check(!imported->submitRanges().empty(),
                          "and the per-frame policy is actually driving its ranges");
                    std::cout << "[selftest] SOG dynamic LOD: level "
                              << imported->lodTable().heldLevel << " held, "
                              << imported->submitRanges().size() << " range(s)" << std::endl;
                } else {
                    check(imported->lodTable().empty(),
                          "a GL editor imports ONE level, no LOD table");
                    stepFixed(4);
                    check(imported->submitRanges().empty(),
                          "and nothing writes ranges the GL path would ignore");
                }

                // The user-reported sequence, verbatim: import a scan, DELETE
                // it, import a DIFFERENT one (THREEPP_SOG_SCAN2 — ideally a
                // bigger scan from another producer, which is what the report
                // used). The first fix for this aimed at the residency leak;
                // this exists so the claim "fixed" rests on the actual
                // sequence at actual scale rather than on a toy repro.
                if (const char* scan2 = std::getenv("THREEPP_SOG_SCAN2"); scan2 && *scan2) {

                    selectObject(imported);
                    deleteSelected();
                    selectObject(nullptr);
                    stepFixed(8);// past framesInFlight+1, so eviction has run

                    const std::filesystem::path asset2(scan2);
                    const auto u8b = asset2.u8string();
                    const std::size_t befor2 = document_.scene().children.size();
                    handleFileDrop({std::string(u8b.begin(), u8b.end())});
                    for (int i = 0; i < 4000 && document_.scene().children.size() == befor2; ++i)
                        stepFixed();

                    SplatCloud* second = nullptr;
                    for (const auto& child : document_.scene().children) {
                        if (auto* c = child->as<SplatCloud>()) second = c;
                    }
                    check(second != nullptr && second != imported,
                          "delete-then-import-another at scan scale survives");
                    if (second) {
                        stepFixed(4);
                        std::cout << "[selftest] second scan: " << second->splatCount()
                                  << " splats, " << second->lodTable().levels.size()
                                  << " level(s), " << second->submitRanges().size()
                                  << " range(s)" << std::endl;
                        check(second->splatCount() > 0, "and it drew in with splats");
                    }
                }
            }
        }
    }

    summary();

    // A filter that matched only sections this build compiled out, or that this
    // invocation did not arm (import, urdf, sog-scan), ran nothing at all. That
    // is not a pass — it is the silent no-op the filter must never turn into.
    if (!filter.empty() && gatedSectionsRun == 0 && !sectionMatches(filter, "boot")) {
        std::cout << "[selftest] the filter ran no section this build/invocation provides"
                  << std::endl;
        return 2;
    }

    std::cout << "[selftest] " << (failed == 0 ? "ALL PASS" : "FAILED") << std::endl;
    return failed == 0 ? 0 : 1;
}
