
#include "PlayerApp.hpp"

#include "DebugDrawOverlay.hpp"
#include "SensorCloudOverlay.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/controls/OrbitControls.hpp"
#include "threepp/core/Clock.hpp"
#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "Scripting.hpp"// keyStateProvider - the seam behind threepp.editor.is_key_down
#include "threepp/input/KeyFromName.hpp"
#endif
#include "threepp/extras/editor/RenderConfig.hpp"
#include "threepp/extras/editor/ViewSpec.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/renderers/RendererFactory.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <any>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

using namespace threepp;
using namespace threepp::player;

namespace {

    // A wall-clock delta big enough to be a stall rather than a frame. Letting a
    // 900 ms hitch through would step the solver 900 ms in one go and explode
    // whatever it is simulating; the sim simply loses the time instead.
    constexpr float kMaxWallDt = 0.1f;

    // Always names a backend — the editor's rule, for the editor's reason:
    // handing createRenderer no preference makes it print a console menu and
    // block on std::cin, which would hang every piped or scripted run this tool
    // exists for.
    GraphicsAPI requestedApi(bool vulkan) {

#ifdef THREEPP_WITH_VULKAN
        if (vulkan) return GraphicsAPI::Vulkan;
#else
        if (vulkan) {
            std::cerr << "threepp player: built without Vulkan support, using OpenGL"
                      << std::endl;
        }
#endif
        return GraphicsAPI::OpenGL;
    }

    // A flat userData string on the scene root, or "" when the document does not
    // carry one. Same two keys the editor reads.
    std::string userDataString(const Object3D& object, const char* key) {

        const auto it = object.userData.find(key);
        if (it == object.userData.end() || it->second.type() != typeid(std::string)) {
            return {};
        }
        return std::any_cast<const std::string&>(it->second);
    }

}// namespace


PlayerApp::PlayerApp(PlayerOptions options)
    : options_(std::move(options)) {

    core_.setLogger([](const std::string& message) {
        // Everything the sessions say goes to stdout, one line each, unadorned:
        // this is a program whose output somebody greps.
        std::cout << "  " << message << std::endl;
    });
}

PlayerApp::~PlayerApp() {

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // The provider closes over `this` and the canvas; neither survives the
    // destructor, so the seam must not either. One player per process, so no
    // ownership token needed — there is nobody to take it back from.
    editor::scripting::keyStateProvider() = nullptr;
#endif
    // The overlays parent nodes into the core's overlay group. Drop them while
    // that group is still alive — member order would otherwise take the core
    // (and its groups) down first.
    sensorCloud_.reset();
    debugDraw_.reset();
    orbit_.reset();
}

int PlayerApp::run() {

    std::string error;
    if (!core_.open(options_.scene, &error)) {
        std::cerr << "threepp player: cannot open " << options_.scene.string() << " - " << error
                  << std::endl;
        // Deliberately not exitCode(): no episode ran, and the caller needs a
        // nonzero code for a document that would not even load.
        return 1;
    }
    log("opened " + options_.scene.filename().string());

    if (!options_.record.empty()) {
        // Per-episode subdirectories only when there is more than one episode:
        // a single run should put its CSVs exactly where it was told, and a
        // multi-episode run must not have every episode overwrite the last (see
        // PlayerCore::beginEpisode).
        core_.setRecordDirectory(options_.record, options_.episodes > 1);
        log("recording sensors to " + options_.record.string() +
            (options_.episodes > 1 ? "/episode_NNN" : ""));
    }

    buildView();

    for (int index = 0; index < options_.episodes; ++index) {
        if (!playEpisode(index)) break;
    }

    const auto failed = core_.failedEpisodes();
    std::cout << "threepp player: " << core_.results().size() << " episode(s), " << failed
              << " failed, " << core_.totalScriptErrors() << " script error(s)" << std::endl;

    return core_.exitCode();
}

void PlayerApp::buildView() {

    // The window comes up hidden in headless mode (GLFW_VISIBLE false) rather
    // than not at all. A depth or lidar sensor scans with the RENDERER, so it
    // needs a live GL context; an invisible window is the cheapest honest way to
    // have one. If even that fails — no display, no driver — the run continues
    // with no renderer at all, which the sensor session already tolerates: the
    // vision sensors are built, say so once and never scan, and every
    // proprioceptive sensor records exactly as it would have.
    try {
        canvas_ = std::make_unique<Canvas>(
                Canvas::Parameters()
                        .title("threepp player - " + options_.scene.filename().string())
                        .size(options_.width, options_.height)
                        .antialiasing(4)
                        // Headless is a batch run, and a FIFO swapchain would
                        // pin it to the monitor's refresh rate for no reason.
                        .vsync(!options_.headless)
                        .headless(options_.headless)
                        .exitOnKeyEscape(true));
        renderer_ = createRenderer(*canvas_, requestedApi(options_.vulkan));
    } catch (const std::exception& e) {
        std::cerr << "threepp player: no renderer (" << e.what()
                  << ") - vision sensors will not scan" << std::endl;
        renderer_.reset();
        canvas_.reset();
    }

    if (renderer_) {
        renderer_->shadowMap().enabled = true;
        renderer_->shadowMap().type = ShadowMap::PFC;
        renderer_->toneMapping = ToneMapping::ACESFilmic;
        renderer_->toneMappingExposure = 1.0f;
        core_.setRenderer(renderer_.get());
        applyDocumentRender();
    }

    if (canvas_) camera_.aspect = canvas_->aspect();
    camera_.updateProjectionMatrix();

    // Orbit and the drawn lines are for a person watching, so neither is stood
    // up for a run nobody can see.
    if (canvas_ && !options_.headless) {
#ifdef THREEPP_EDITOR_WITH_PYTHON
        // threepp.editor.is_key_down, answered from the window's own key state —
        // a scene authored to be DRIVEN (the hover arena's W/A/S/D drone) is
        // drivable here too, not only in the editor. The editor answers this
        // from ImGui because it must ignore keys typed into its panels; the
        // player has no panels, so the Canvas state IS the answer. Headless
        // installs nothing, and the provider then answers False — a script that
        // steers still runs, just uncommanded, which is what an unattended
        // evaluation wants (and what keeps episode CSVs reproducible).
        editor::scripting::keyStateProvider() = [this](const std::string& name) {
            return canvas_ && canvas_->isKeyDown(keyFromName(name));
        };
#endif
        orbit_ = std::make_unique<OrbitControls>(camera_, *canvas_);
        orbit_->enableDamping = true;

        // Somebody is watching, so somebody can hear: the listener rides the
        // camera, as it rides the editor's viewport camera. A headless run
        // leaves this null and the listener stays at the origin — the sounds
        // still load and still play, they are just not spatialized around
        // anyone.
        core_.setAudioListenerHost(&camera_);

        canvas_->onWindowResize([this](WindowSize size) {
            camera_.aspect = size.aspect();
            camera_.updateProjectionMatrix();
            if (renderer_) renderer_->setSize(size);
        });

        if (auto* overlay = core_.overlay()) {
            debugDraw_ = std::make_unique<DebugDrawOverlay>(*overlay);
            debugDraw_->setLogger([this](const std::string& message) { log(message); });
            // Drawing the lines is how they get drained. Without this the core
            // clears the list itself each step, which is what the headless path
            // does — either way it never reaches its cap.
            core_.setDebugDrawDrain([this] { debugDraw_->sync(); });

            // What the vision sensors measured, as the editor draws it: one
            // range-coloured cloud. Under the same scan-hidden overlay group, so
            // a lidar never ranges against its own returns.
            if (auto* sensors = core_.sensors()) {
                sensorCloud_ = std::make_unique<SensorCloudOverlay>(*overlay, *sensors);
            }
        }
    }

    applyDocumentView();
}

void PlayerApp::applyDocumentRender() {

    // A document's own exposure, fog and tone map, layered over what the player
    // just set as its baseline — the same read the editor does on open, so a
    // scene looks the same in both.
    const auto base = editor::RenderConfig::capture(*renderer_);
    editor::RenderConfig::read(core_.scene(), base).value_or(base).apply(*renderer_);
}

void PlayerApp::applyDocumentView() {

    followName_ = userDataString(core_.scene(), "editorFollow");
    follow_ = nullptr;
    followSeeded_ = false;

    const auto view = userDataString(core_.scene(), "editorView");
    Vector3 position;
    Vector3 target;
    // The parse is the library's, shared with the editor and with --shot, so a
    // vantage copied between them means the same thing (extras/editor/ViewSpec).
    if (!view.empty() && editor::parseViewSpec(view, position, target)) {
        camera_.position.copy(position);
        camera_.lookAt(target);
        if (orbit_) orbit_->target.copy(target);
        log("using the document's authored view");
        return;
    }
    if (!view.empty()) {
        log("editorView is \"" + view + "\" - want px,py,pz@tx,ty,tz; ignored");
    }
    frameSceneBounds();
}

void PlayerApp::frameSceneBounds() {

    // Nothing authored: back off far enough to see everything, from the
    // three-quarter angle an editor opens at.
    Box3 bounds;
    bounds.setFromObject(core_.scene());
    if (bounds.isEmpty()) {
        camera_.position.set(6.f, 5.f, 8.f);
        camera_.lookAt({0.f, 0.f, 0.f});
        return;
    }

    Vector3 center;
    Vector3 size;
    bounds.getCenter(center);
    bounds.getSize(size);

    const float extent = std::max({size.x, size.y, size.z, 1.f});
    const float distance = extent * 1.6f;

    camera_.position.set(center.x + distance * 0.6f,
                         center.y + distance * 0.5f,
                         center.z + distance * 0.8f);
    camera_.lookAt(center);
    if (orbit_) orbit_->target.copy(center);
}

bool PlayerApp::playEpisode(int index) {

    std::string error;
    if (!core_.beginEpisode(index, &error)) {
        std::cerr << "threepp player: episode " << index << " would not play - " << error
                  << std::endl;
        // A document that refuses once refuses every time; nothing is learned by
        // trying it another ninety-nine times. The refusal is already recorded
        // as a failed episode, so the exit code is nonzero.
        return false;
    }

    // The scene is rebuilt by each stop(), so the followed object is a different
    // pointer every episode. Re-resolve here, against the graph play() just
    // started on.
    follow_ = followName_.empty() ? nullptr : core_.scene().getObjectByName(followName_);
    followSeeded_ = false;
    if (!followName_.empty() && !follow_) {
        log("editorFollow names \"" + followName_ + "\", which is not in this scene - ignored");
        followName_.clear();
    }

    // Zero means unbounded. Headless has to have SOME bound or it never returns,
    // so it takes a default rather than hanging a CI job.
    float budgetSeconds = options_.seconds;
    if (options_.headless && options_.frames <= 0 && budgetSeconds <= 0.f) {
        budgetSeconds = PlayerOptions::headlessDefaultSeconds;
        if (index == 0) {
            log("headless with no --frames/--seconds: using --seconds=" +
                std::to_string(static_cast<int>(budgetSeconds)));
        }
    }

    // A fixed step is reproducible and is what a headless evaluation wants; the
    // wall clock is what a person watching a window wants. --dt forces the
    // former either way.
    const bool fixed = options_.dt > 0.f || options_.headless;
    const float fixedDt = options_.dt > 0.f ? options_.dt : PlayerCore::defaultDt;

    Clock clock;
    int frames = 0;
    bool windowOpen = true;

    while (true) {
        if (options_.frames > 0 && frames >= options_.frames) break;
        // Against the sim clock the core keeps, NOT a sum of the dts stepped:
        // --seconds promises simulated seconds, and a dt bigger than the physics
        // catch-up cap simulates less than it steps. Summing dt here would end a
        // --dt=0.5 --seconds=1 run after two frames of which the sim saw 0.13 s.
        if (budgetSeconds > 0.f && core_.episodeSeconds() >= budgetSeconds) break;

        float dt = fixedDt;
        if (!fixed) dt = std::min(clock.getDelta(), kMaxWallDt);

        if (canvas_) {
            if (!canvas_->animateOnce([&] { frame(dt); })) {
                windowOpen = false;
                break;
            }
        } else {
            frame(dt);
        }

        ++frames;
    }

    const auto result = core_.endEpisode();
    // stop() replaced the scene the subject lived in. Let go before the pointer
    // is a pointer into a freed graph rather than after.
    follow_ = nullptr;
    followSeeded_ = false;
    // The cloud's "keep the last frame" rule must not cross an episode
    // boundary: episode N+1 opens clean, not wearing episode N's returns.
    if (sensorCloud_) sensorCloud_->clear();

    report(result);
    return windowOpen;
}

void PlayerApp::frame(float dt) {

    core_.step(dt);

    // After the step (the sensors have drained this frame's returns), before
    // the render (this frame should show them).
    if (sensorCloud_) sensorCloud_->sync();

    updateFollow();
    if (orbit_) orbit_->update();

    if (renderer_) renderer_->render(core_.scene(), camera_);
}

void PlayerApp::updateFollow() {

    if (!follow_) return;

    Vector3 position;
    follow_->getWorldPosition(position);

    // A rigid translation of camera AND target by the subject's own movement:
    // the view keeps whatever angle and distance somebody orbited to, and the
    // damping the orbit is carrying still resolves against the same offset.
    if (followSeeded_) {
        const Vector3 delta = position.clone().sub(followLast_);
        camera_.position.add(delta);
        if (orbit_) orbit_->target.add(delta);
    }
    followLast_.copy(position);
    followSeeded_ = true;
}

void PlayerApp::report(const EpisodeResult& result) {

    std::cout << "episode " << result.index << ": " << result.frames << " frames, "
              << result.seconds << " s sim, " << result.scriptInstances << " script(s), "
              << result.scriptErrors << " error(s), " << result.bodyCount << " bodies, "
              << result.sensorCount << " sensor(s)";
    // Only when the document has them: a line that says "0 conveyors" on every
    // scene teaches nobody anything, but a scene that HAD one and now reports
    // none is exactly what somebody reading a CI log needs to see.
    if (result.conveyorCount > 0) std::cout << ", " << result.conveyorCount << " conveyor(s)";
    if (core_.recording()) std::cout << ", " << result.sensorRows << " row(s) recorded";
    if (!result.error.empty()) std::cout << " [" << result.error << "]";
    std::cout << std::endl;
}

void PlayerApp::log(const std::string& message) {

    std::cout << "  " << message << std::endl;
}
