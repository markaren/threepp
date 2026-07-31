
#include "EditorApp.hpp"

#include "EditorTheme.hpp"
#include "ExampleScenes.hpp"
#include "ImportFormats.hpp"
#include "PanelLayout.hpp"

#include "threepp/extras/editor/AnimationPlaySession.hpp"
#include "threepp/extras/editor/GeneratorConfig.hpp"
#include "threepp/extras/editor/MaterialTextureSlots.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/ScriptWorkspace.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"
#include "threepp/extras/editor/ViewSpec.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "ScriptHost.hpp"// runAuthoringSource, for the Generator's Regenerate
#endif

#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"

#include "threepp/extras/editor/SensorPlaySession.hpp"
#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#endif

#include "threepp/canvas/Monitor.hpp"
#include "threepp/core/Clock.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/helpers/AxesHelper.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/helpers/GridHelper.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/AssetSource.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/renderers/RendererFactory.hpp"
#include "threepp/scenes/Scene.hpp"
#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>// std::getenv (THREEPP_BENCH_VSYNC)
#include <cstring>
#include <iostream>
#include <sstream>

// GLFW's window-title setter. Declared rather than included: the GLFW headers
// are private to the threepp target, but the symbol is linked into it and the
// signature is a plain C entry point taking an opaque window handle.
extern "C" void glfwSetWindowTitle(void* window, const char* title);

using namespace threepp;
using namespace threepp::editor;

namespace {

    constexpr int kDefaultWidth = 1600;
    constexpr int kDefaultHeight = 900;

    constexpr std::size_t kConsoleLimit = 400;

#ifdef THREEPP_EDITOR_WITH_PYTHON
    // A key NAME (as a script writes it) -> ImGuiKey. The vocabulary is deliberately the one
    // python/src/bind_render.cpp's keyFromName accepts, so a script that polls 'UP' or 'KP8'
    // reads the same in the editor as it does against a Canvas in the wheel. The enums differ
    // (ImGuiKey here, threepp::Key there), so this is a parallel mapping rather than shared
    // code; keep the accepted spellings in step.
    ImGuiKey imguiKeyFromName(std::string name) {

        for (auto& ch : name) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (name.size() == 1 && name[0] >= 'A' && name[0] <= 'Z') {
            return static_cast<ImGuiKey>(ImGuiKey_A + (name[0] - 'A'));
        }
        if (name.size() == 1 && name[0] >= '0' && name[0] <= '9') {
            return static_cast<ImGuiKey>(ImGuiKey_0 + (name[0] - '0'));
        }
        // Numpad: "KP8" / "NUM8" / "NUMPAD8" -> keypad 8, distinct from the top-row digit.
        for (const std::string& prefix : {std::string("KP"), std::string("NUMPAD"), std::string("NUM")}) {
            if (name.size() == prefix.size() + 1 && name.compare(0, prefix.size(), prefix) == 0 &&
                name.back() >= '0' && name.back() <= '9') {
                return static_cast<ImGuiKey>(ImGuiKey_Keypad0 + (name.back() - '0'));
            }
        }
        if (name == "SPACE") return ImGuiKey_Space;
        if (name == "UP") return ImGuiKey_UpArrow;
        if (name == "DOWN") return ImGuiKey_DownArrow;
        if (name == "LEFT") return ImGuiKey_LeftArrow;
        if (name == "RIGHT") return ImGuiKey_RightArrow;
        if (name == "ESCAPE" || name == "ESC") return ImGuiKey_Escape;
        if (name == "ENTER") return ImGuiKey_Enter;
        if (name == "TAB") return ImGuiKey_Tab;
        if (name == "SHIFT") return ImGuiKey_LeftShift;
        if (name == "CTRL" || name == "CONTROL") return ImGuiKey_LeftCtrl;
        return ImGuiKey_None;
    }
#endif

    // Always names a backend. Handing createRenderer no preference makes it
    // print a console menu and block on std::cin, which a windowed app must
    // never do — it stalls the editor behind a prompt nobody sees and hangs
    // any piped or scripted run.
    GraphicsAPI requestedApi(bool vulkan) {

#ifdef THREEPP_WITH_VULKAN
        if (vulkan) return GraphicsAPI::Vulkan;
#else
        if (vulkan) {
            std::cerr << "threepp editor: built without Vulkan support, using OpenGL\n";
        }
#endif
        return GraphicsAPI::OpenGL;
    }

    // Which way `object` is FACING, as an angle about world up: its forward axis
    // projected onto the ground plane. False when there is nothing to project —
    // a nose within a few degrees of straight up or down, where the projection
    // is numerical noise and the caller must keep the heading it had, or a body
    // tumbling through vertical whips the camera round with it.
    //
    // Forward is -Z. Not a convention picked for one scene: it is threepp's own,
    // the direction lookAt() aims and the direction every camera in the engine
    // looks, so an object modelled to face the way it moves faces -Z. Read off
    // the WORLD quaternion, so a subject parented under something rotated heads
    // where it actually points.
    bool headingOf(Object3D& object, float& radians) {

        Quaternion world;
        object.getWorldQuaternion(world);

        Vector3 forward(0.f, 0.f, -1.f);
        forward.applyQuaternion(world);

        // sin(4 degrees). Below it the ground-plane projection is shorter than
        // the attitude noise of a hovering body.
        constexpr float kFlat = 0.07f;
        if (forward.x * forward.x + forward.z * forward.z < kFlat * kFlat) return false;

        // Zero for an unrotated object, and +theta for a yaw of +theta about +Y,
        // which is exactly what Vector3::applyAxisAngle(worldUp, theta) undoes.
        radians = std::atan2(-forward.x, -forward.z);
        return true;
    }

    // --bench turns vsync off, because a present-capped frame time measures the
    // display rather than the renderer. THREEPP_BENCH_VSYNC=1 puts it back, for
    // the one question the uncapped number cannot answer: does the editor AS
    // SHIPPED hold the refresh rate.
    bool benchWantsVsync(const EditorApp::Options& options) {

        if (options.bench <= 0) return true;
        const char* keep = std::getenv("THREEPP_BENCH_VSYNC");
        return keep && *keep && *keep != '0';
    }

}// namespace


EditorApp::EditorApp(const Options& options)
    : options_(options),
      canvas_(Canvas::Parameters()
                      .title("threepp editor")
                      .size(kDefaultWidth, kDefaultHeight)
                      .antialiasing(4)
                      // A timed pass measures the RENDERER. With vsync on, the
                      // swapchain is FIFO and every frame time is quantized to
                      // the refresh interval, which measures the monitor.
                      .vsync(benchWantsVsync(options))
                      .exitOnKeyEscape(false)),
      renderer_(createRenderer(canvas_, requestedApi(options.vulkan))),
      camera_(55.f, canvas_.aspect(), 0.05f, 5000.f),
      ortho_(-1.f, 1.f, 1.f, -1.f, 0.05f, 10000.f) {

    contentScale_ = monitor::contentScale().first;
    // The fonts already follow the window between monitors (ImguiContext has
    // its own onMonitorChange subscription); this keeps the editor's layout
    // math — panel widths, button sizes, marker pixels — on the same scale,
    // so a HiDPI laptop screen and a 100% external monitor both look right.
    canvas_.onMonitorChange([this](int idx) {
        contentScale_ = monitor::contentScale(idx).first;
    });

    renderer_->shadowMap().enabled = true;
    renderer_->shadowMap().type = ShadowMap::PFC;
    renderer_->toneMapping = ToneMapping::ACESFilmic;
    renderer_->toneMappingExposure = 1.0f;

#ifdef THREEPP_WITH_VULKAN
    // The axis views are a 3D camera that happens to project in parallel, not a
    // 2D overlay. Without this the Vulkan backend reads an OrthographicCamera as
    // a HUD and draws the scene as flat unlit fills — no lights, no shadows, no
    // fog — so Numpad 5 would change how the viewport SHADES, not just how it
    // projects. Ignored by the OpenGL backend, which never had the ambiguity.
    if (auto* vk = dynamic_cast<VulkanRenderer*>(renderer_.get())) {
        vk->setOrthographicSceneRendering(true);

        // The editor is an authoring tool running a deferred path-traced
        // backend on whatever laptop it was opened on, and the shade is the
        // frame's dominant cost — it scales with pixels. 0.8 is 64% of them for
        // a difference TAA reconstructs most of the way back, which is the
        // trade a viewport should default to; a final frame is what --screenshot
        // and the Render scale slider are for. Deliberately editor-only: the
        // renderer itself still defaults to 1.0 for every example and test.
        vk->setRenderScale(0.8f);
    }
#endif

    // After every renderer knob above: this is the baseline a document is
    // opened against and saved as a difference from.
    renderDefaults_ = RenderConfig::capture(*renderer_);

    camera_.position.set(6, 5, 8);

    // --- editor-only overlay ------------------------------------------------
    // One node holds everything the editor draws but never saves. SceneDocument
    // detaches it for every export, so no helper can leak into a document.
    overlay_ = Group::create();
    overlay_->name = "__editor_overlay";
    document_.addEditorOnly(*overlay_);

    grid_ = GridHelper::create(40, 40, 0x4a4a4a, 0x2c2c2c);
    overlay_->add(grid_);

    axes_ = AxesHelper::create(1.5f);
    overlay_->add(axes_);

    markers_ = Group::create();
    markers_->name = "__editor_markers";
    overlay_->add(markers_);

    splines_ = Group::create();
    splines_->name = "__editor_splines";
    overlay_->add(splines_);

    // Editor-only like the overlay, but a SIBLING of it rather than a child: the
    // overlay is hidden for the duration of every sensor scan (a depth camera
    // pointed at the grid otherwise measures the grid), and a sensor must not be
    // hidden from itself. See SensorPlaySession.
    sensorRig_ = Group::create();
    sensorRig_->name = "__editor_sensor_rig";
    document_.addEditorOnly(*sensorRig_);

    // Orbiting while dragging a handle fights the gizmo; and a drag is exactly
    // the span an undo entry should cover.
    gizmoDragListener_ = std::make_unique<LambdaEventListener>([this](Event& event) {
        const bool dragging = std::any_cast<bool>(event.target);
        orbit_->enabled = !dragging;
        auto* selected = selection_.get();
        if (dragging) {
            if (selected) {
                gizmoBefore_ = SetTransformCommand::read(*selected);
                gizmoDragging_ = true;
                commands_.beginTransaction();
            }
            return;
        }
        if (gizmoDragging_ && selected) {
            commands_.push(std::make_unique<SetTransformCommand>(
                    *selected, gizmoBefore_, SetTransformCommand::read(*selected),
                    gizmoMode_ == "translate" ? "Move" : (gizmoMode_ == "rotate" ? "Rotate" : "Scale")));
            commands_.endTransaction();
            document_.setDirty(true);
        }
        gizmoDragging_ = false;
    });

    // Builds orbit_ and gizmo_ against the perspective camera. Called again
    // whenever the projection changes.
    bindViewportControls();
    orbit_->target.set(0, 0.5f, 0);

    // --- ImGui --------------------------------------------------------------
    ui_ = std::make_unique<ImguiFunctionalContext>(canvas_, *renderer_, [this] { drawUi(); });

    ioCapture_.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture_.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture_.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas_.setIOCapture(&ioCapture_);

    canvas_.onWindowResize([this](WindowSize size) {
        camera_.aspect = size.aspect();
        camera_.updateProjectionMatrix();
        // The ortho frustum keeps its height and re-derives its width, so a
        // resize widens the view rather than rescaling what is in it.
        setOrthoHeight((ortho_.top - ortho_.bottom) / std::max(ortho_.zoom, 1e-4f));
        renderer_->setSize(size);
    });

    canvas_.onDrop([this](std::vector<std::string> paths) { handleFileDrop(paths); });

    // A restored or reloaded scene is a different Scene object; everything that
    // pointed into the old one has to be re-resolved.
    document_.onSceneReplaced([this](Scene& scene) {
        // The previewed nodes lived in the old scene; restoring would write
        // through dangling pointers. Discard only.
        stopAnimationPreview(false);
        // Markers, spline curves and the frustum helper point into the outgoing
        // graph too. Drop them before anything can dereference an owner that is
        // gone.
        clearViewportMarkers();
        clearSplineOverlays();
        // The collider lines are world-space and belong to a world that stop()
        // has already destroyed; the node itself is parented to the surviving
        // overlay, so it has to be taken down explicitly.
        clearPhysicsDebug();
        // Same for the sensor cloud: world-space points from sensors the play
        // session has already dropped, hanging off an overlay that survives.
        clearSensorOverlay();
        // Thumbnails are keyed by texture uuid, and the restored scene rebuilds
        // its textures — same uuid, different object. Nothing stale survives.
        clearThumbnailCache();
        if (cameraHelper_) {
            cameraHelper_->removeFromParent();
            cameraHelper_.reset();
            cameraHelperFor_ = nullptr;
        }
        // Recorded commands point into the old scene too. Re-resolve their
        // targets by uuid against the new graph; commands that cannot (raw
        // captures in property setters) are dropped rather than left dangling.
        // Everything below points into the outgoing graph, so note what has to
        // come back before letting go of it.
        const auto uuid = selection_.uuid();
        selection_.set(nullptr);
        gizmo_->detach();
        // The overlay group survives the scene swap, so the outlines must be
        // detached from it, not just dropped. The instance index goes with them:
        // it indexed into a mesh that is about to be freed, and the reselect
        // below re-derives whatever the restored graph actually has.
        if (selectionBox_) {
            selectionBox_->removeFromParent();
            selectionBox_.reset();
        }
        if (instanceOutline_) {
            instanceOutline_->removeFromParent();
            instanceOutline_.reset();
        }
        selectedInstance_.reset();
        instanceBox_.makeEmpty();

        // Before rebinding: this replaces nodes, and the command stack should
        // resolve its targets against the final graph.
        rearticulateRobots(scene);

        commands_.rebind(scene);
        if (!uuid.empty()) {
            Object3D* found = nullptr;
            scene.traverse([&](Object3D& o) {
                if (!found && o.uuid == uuid) found = &o;
            });
            if (found) selectObject(found);
        }
    });

    // Undoing an "Add" deletes the object it created, and undoing a paste or a
    // reparent can move it out from under the selection. Anything still
    // pointing at a node that left the scene has to let go — otherwise the
    // gizmo drives a detached object and TransformControls rightly complains.
    commands_.onChange([this] {
        auto* selected = selection_.get();
        if (!selected) return;
        bool present = false;
        document_.scene().traverse([&](Object3D& object) {
            if (&object == selected) present = true;
        });
        if (!present) selectObject(nullptr);
    });

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // Kept as a member too: the collider overlay reads the world it builds.
    physics_ = std::make_shared<PhysicsPlaySession>();
    // Soft bodies can decline to cook, and the GPU world can decline to come
    // up; both are worth a line in the log rather than silence.
    physics_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(physics_);
#endif
    play_.addSession(std::make_shared<AnimationPlaySession>());
    // After physics (whose world the pushed sensors register with) and after the
    // animation player, so a scan sees the pose the frame ended on. Before the
    // script session, which stays last by rule — a script that moves a sensor's
    // object is therefore read one frame later, which is the price of "scripts
    // are the frame's final word".
#ifdef THREEPP_EDITOR_WITH_PHYSX
    {
        auto sensors = std::make_shared<PhysxSensorPlaySession>();
        sensors->setPhysics(physics_.get());
        sensors_ = std::move(sensors);
    }
#else
    // The base session runs the vision sensors: a depth or lidar scan needs a
    // renderer, not a physics world. Body and joint sensors author, and say at
    // Play which build they are waiting for.
    sensors_ = std::make_shared<SensorPlaySession>();
#endif
    sensors_->setRenderer(renderer_.get());
    sensors_->setRig(sensorRig_.get());
    sensors_->setHiddenDuringScan(overlay_.get());
    sensors_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(sensors_);
#ifdef THREEPP_EDITOR_WITH_PYTHON
    // Last, so a script's transform edits are the final word for the frame —
    // physics and the animation player have already had their say.
    scripts_ = std::make_shared<ScriptPlaySession>();
    scripts_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(scripts_);

    // threepp.editor.is_key_down: let a playing script be DRIVEN. Answered from ImGui's key
    // state, not the canvas's — `ioCapture_.preventKeyboardEvent` gates on
    // WantCaptureKeyboard, which stays true for as long as a panel keeps focus after a click,
    // so the canvas's held-key set would be stale exactly when somebody is watching the
    // viewport. ImGui's own state is fed by the backend regardless of who is capturing.
    //
    // Suppressed on the same rule handleShortcuts() uses (real text entry, an open popup, the
    // file browser) rather than on WantCaptureKeyboard, for the same reason: gating on that
    // made every shortcut dead until the viewport was clicked again.
    scripting::keyStateProvider() = [this](const std::string& name) {
        // Asking at all is the signal: a script that polls the keyboard takes the plain keys
        // off the editor for the rest of the session (see handleShortcuts). Recorded on the
        // first poll, which happens on the script's first update() — before anyone has had
        // time to press anything.
        if (isPlaying() && !scriptsPolledKeys_) {
            scriptsPolledKeys_ = true;
            log("a script is reading the keyboard - editor key shortcuts yield until Stop");
        }
        const ImGuiIO& io = ImGui::GetIO();
        if (io.WantTextInput) return false;
        if (ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) return false;
        if (fileBrowser_.isOpen()) return false;
        const ImGuiKey key = imguiKeyFromName(name);
        return key != ImGuiKey_None && ImGui::IsKeyDown(key);
    };
#endif

    loadSettings();

    assetDir_ = settings_.sceneDir.empty() ? std::filesystem::current_path()
                                           : std::filesystem::path(settings_.sceneDir);

    if (!options_.example.empty()) {
        openExample(options_.example);
    } else if (!options_.openOnStart.empty() && options_.openOnStart.extension() == ".json") {
        openScene(options_.openOnStart);
    } else {
        buildTemplateScene();
        // Any other startup path goes through the same dispatch as a file
        // drop, so `threepp_editor model.glb` imports into the template scene.
        // The self-test drives its own import instead (with assertions).
        if (!options_.openOnStart.empty() && !options_.selfTest) {
            handleFileDrop({options_.openOnStart.string()});
        }
    }

    log("threepp editor ready");
#ifndef THREEPP_EDITOR_WITH_PHYSX
    log("built without PhysX - no physics session; vision sensors still scan");
#endif
#ifndef THREEPP_EDITOR_WITH_PYTHON
    log("built without Python scripting - scripts are authored and saved, not run");
#endif
}

EditorApp::~EditorApp() {

    canvas_.setIOCapture(nullptr);

    // Stops the poll and takes the scratch .py with it. Whatever was saved is
    // already in the document; what is left on disk is a copy nobody will read.
    stopExternalEdit("the editor is closing");

    // The sensor session parents nodes into sensorRig_, which is a member of this
    // same object: member destruction order would take the rig down first and
    // leave the session's destructor unlinking from freed memory. Drop the
    // sessions here, while everything they point into is still alive.
    play_.clearSessions();
    sensors_.reset();
    physics_.reset();
    if (sensorRig_) {
        sensorRig_->clear();
        document_.removeEditorOnly(*sensorRig_);
    }

    // Tear the overlay down before the members it points at (the gizmo owns a
    // pimpl that unregisters canvas listeners in its destructor).
    if (gizmo_) {
        gizmo_->removeEventListener("dragging-changed", *gizmoDragListener_);
        gizmo_->detach();
        gizmo_->removeFromParent();
    }
    if (overlay_) {
        overlay_->clear();
        document_.removeEditorOnly(*overlay_);
    }
}

int EditorApp::run() {

    Clock clock;

    if (options_.selfTest) return runSelfTest();
    if (options_.bench > 0) return runBench();
    if (!options_.screenshot.empty()) return runScreenshot();

    if (options_.play) startPlay();

    if (options_.maxFrames > 0) {
        for (int i = 0; i < options_.maxFrames; ++i) {
            if (!canvas_.animateOnce([&] { frame(clock.getDelta()); })) break;
        }
    } else {
        canvas_.animate([&] {
            frame(clock.getDelta());
        });
    }

    persistSettings();
    return 0;
}

void EditorApp::frame(float dt) {

    play_.update(dt);
    // Before anything reads the graph: the Generator section asked for this last
    // frame, and it replaces a node the panel was drawing from.
    const auto resolveCarrier = [this](const std::string& uuid) -> Object3D* {
        if (document_.scene().uuid == uuid) return &document_.scene();
        Object3D* found = nullptr;
        document_.scene().traverse([&](Object3D& o) {
            if (!found && o.uuid == uuid) found = &o;
        });
        return found;
    };
    if (!pendingRegenerate_.empty()) {
        if (auto* carrier = resolveCarrier(std::exchange(pendingRegenerate_, {}))) {
            regenerate(*carrier);
        }
    }
    if (!pendingGeneratorClear_.empty()) {
        if (auto* carrier = resolveCarrier(std::exchange(pendingGeneratorClear_, {}))) {
            clearGenerator(*carrier);
        }
    }
    pollImports(dt);
    pollExternalEdit(dt);
    if (animPreview_) animPreview_->mixer->update(dt);
    // Before the orbit update, and a rigid translation of target AND camera, so
    // the damping the orbit is carrying still resolves against the same offset.
    updateFollow(dt);
    orbit_->update();
    updateViewPreset();
    // The nudge that keeps the grid off a coplanar ground is measured against
    // the camera, so it is re-derived every frame rather than on view changes.
    updateGridPlacement();

    refreshSelectionHelpers();

    renderer_->render(document_.scene(), viewCamera());
    renderCameraPreview();
    if (!benchSkipUi_) ui_->render();

    updateWindowTitle();
}

void EditorApp::updateWindowTitle() {

    auto title = "threepp editor - " + document_.title();
    if (title == lastWindowTitle_) return;
    lastWindowTitle_ = title;
    glfwSetWindowTitle(canvas_.windowPtr(), title.c_str());
}

void EditorApp::drawUi() {

    theme::apply(contentScale_);

    const ImGuiIO& io = ImGui::GetIO();
    fps_ = io.Framerate;

    // The side panels size themselves against the status bar, which is drawn
    // after them. Seeding the height here (the status bar recomputes the same
    // value) keeps the very first frame from being laid out against zero.
    statusHeight_ = ImGui::GetFrameHeight() + 4 * contentScale_;

    objectCount_ = 0;
    document_.scene().traverse([&](Object3D& o) {
        if (!document_.isEditorOnly(o) && &o != &document_.scene()) ++objectCount_;
    });

    // Rebuilt by drawInspector() below; anything left from last frame points at
    // a layout that no longer exists.
    frameTextureSlots_.clear();

    drawMenuBar();
    drawToolbar();
    // Before the panels: the Scripts tab lives in the bottom panel, and which
    // scripts are in it — if any — is decided here.
    updateScriptEditors();
    drawHierarchy();
    drawInspector();
    drawBottomPanel();
    drawStatusBar();
    drawPlayBanner();
    drawImportToast();

    if (preview_.visible) {
        // Background list, not foreground: the camera image is drawn by the
        // renderer before any ImGui at all, so this layer sits over it while
        // still passing under dialogs and menus.
        auto* draw = ImGui::GetBackgroundDrawList();
        const auto* viewport = ImGui::GetMainViewport();
        const ImVec2 min(viewport->Pos.x + preview_.x, viewport->Pos.y + preview_.y);
        const ImVec2 max(min.x + preview_.w, min.y + preview_.h);
        const float s = contentScale_;

        if (!preview_.active) {
            // Nothing selected: paint the dock so the corner reads as panel
            // rather than as a scrap of viewport nobody can reach.
            draw->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_WindowBg));
            const char* hint = "No camera selected";
            const auto textSize = ImGui::CalcTextSize(hint);
            draw->AddText({min.x + (preview_.w - textSize.x) * 0.5f,
                           min.y + (preview_.h - textSize.y) * 0.5f},
                          ImGui::GetColorU32(theme::muted()), hint);
        }

        draw->AddRect(min, max, ImGui::GetColorU32(ImGuiCol_Border));

        if (preview_.active) {
            const ImVec2 pad(6.f * s, 3.f * s);
            const auto textSize = ImGui::CalcTextSize(preview_.label.c_str());
            draw->AddRectFilled(min, {min.x + textSize.x + 2 * pad.x, min.y + textSize.y + 2 * pad.y},
                                ImGui::GetColorU32(ImGuiCol_WindowBg, 0.85f));
            draw->AddText({min.x + pad.x, min.y + pad.y},
                          ImGui::GetColorU32(ImGuiCol_Text), preview_.label.c_str());
        }
    }

    // Dialogs and modals last so they sit above the panels.
    if (fileBrowser_.draw(contentScale_)) {
        const auto path = fileBrowser_.result();
        switch (pendingDialog_) {
            case PendingDialog::Open:
                settings_.sceneDir = fileBrowser_.directory().string();
                openScene(path);
                break;
            case PendingDialog::SaveAs:
                settings_.sceneDir = fileBrowser_.directory().string();
                saveSceneAs(path);
                break;
            case PendingDialog::ImportModel:
                settings_.modelDir = fileBrowser_.directory().string();
                importModel(path);
                break;
            case PendingDialog::Environment:
                settings_.environmentDir = fileBrowser_.directory().string();
                setEnvironment(path, environmentAsBackground_);
                break;
            case PendingDialog::Texture:
                settings_.textureDir = fileBrowser_.directory().string();
                assignTextureToSlot(path);
                break;
            case PendingDialog::Script:
                settings_.scriptDir = fileBrowser_.directory().string();
                // The dialog spans frames; the object it was opened for may be
                // gone (deleted, or replaced by a play/stop) by now.
                if (auto* target = findByUuid(document_.scene(), scriptTargetUuid_)) {
                    assignScript(*target, path);
                } else {
                    log("script not attached - the object is no longer in the scene");
                }
                scriptTargetUuid_.clear();
                break;
            case PendingDialog::RecordDir:
#ifdef THREEPP_EDITOR_WITH_PHYSX
                // The dialog picks a FILE (it has no directory mode); what the
                // recorder wants is the folder it sits in, because it writes one
                // CSV per sensor named after the sensor.
                if (sensors_) {
                    const auto dir = path.has_filename() ? path.parent_path() : path;
                    sensors_->setRecordDirectory(dir);
                    log("sensor recordings will go to " + dir.string());
                }
#endif
                break;
            case PendingDialog::None:
                break;
        }
        pendingDialog_ = PendingDialog::None;
    }

    // Unsaved-changes guard for New / Open / Quit.
    if (pendingAction_ != PendingAction::None && document_.dirty()) {
        ImGui::OpenPopup("Unsaved changes");
    } else if (pendingAction_ != PendingAction::None) {
        const auto action = pendingAction_;
        pendingAction_ = PendingAction::None;
        switch (action) {
            case PendingAction::New: newScene(); break;
            case PendingAction::Open:
                pendingDialog_ = PendingDialog::Open;
                fileBrowser_.open("Open Scene", FileBrowser::Mode::Open,
                                  settings_.sceneDir, {".json"});
                break;
            case PendingAction::OpenPath: openScene(pendingPath_); break;
            case PendingAction::OpenExample: openExample(pendingExample_); break;
            case PendingAction::Quit: canvas_.close(); break;
            case PendingAction::None: break;
        }
    }

    if (ImGui::BeginPopupModal("Unsaved changes", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("The scene has unsaved changes.");
        ImGui::Spacing();
        if (ImGui::Button("Save", {110 * contentScale_, 0})) {
            saveScene();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Discard", {110 * contentScale_, 0})) {
            document_.setDirty(false);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", {110 * contentScale_, 0})) {
            pendingAction_ = PendingAction::None;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (!importError_.empty() && !ImGui::IsPopupOpen("Import failed")) {
        ImGui::OpenPopup("Import failed");
    }
    if (ImGui::BeginPopupModal("Import failed", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted(importError_.c_str());
        ImGui::Spacing();
        if (ImGui::Button("OK", {110 * contentScale_, 0})) {
            importError_.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    handleShortcuts();

    // After every panel has drawn, so the texture slot rows a drop may have
    // landed on are known for this frame.
    resolveTextureDrops(io.MousePos.x, io.MousePos.y);

    // Structural edits parked by the hierarchy walk.
    if (deferred_) {
        auto operation = std::move(deferred_);
        deferred_ = nullptr;
        operation();
    }

    // Picking runs last: it must see the WantCaptureMouse produced by every
    // panel drawn this frame.
    if (!io.WantCaptureMouse && !fileBrowser_.isOpen() &&
        ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !gizmo_->isDragging()) {
        // A click, not the end of an orbit drag.
        const auto drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.f);
        if (std::abs(drag.x) < 3.f * contentScale_ && std::abs(drag.y) < 3.f * contentScale_) {
            pickAt(io.MousePos.x, io.MousePos.y);
        }
    }
}


// ---------------------------------------------------------------- scene setup

void EditorApp::newScene() {

    // Replacing the document under a running session would leave every session
    // holding a Scene& to a graph that no longer exists.
    if (rejectWhilePlaying("New Scene")) return;

    selectObject(nullptr);
    commands_.clear();
    document_.newScene();
    buildTemplateScene();
    // A new document is a document: it renders the way the editor starts,
    // whatever the one before it had dialled in.
    applyDocumentRender();
    log("new scene");
}

void EditorApp::buildTemplateScene() {

    auto& scene = document_.scene();
    scene.background = Color(0x1c1f24);

    auto sun = DirectionalLight::create(0xffffff, 2.5f);
    sun->name = "Sun";
    sun->position.set(6, 10, 5);
    sun->castShadow = true;
    scene.add(sun);

    auto ambient = AmbientLight::create(0xffffff, 0.35f);
    ambient->name = "Ambient Light";
    scene.add(ambient);

    auto groundMaterial = MeshStandardMaterial::create();
    groundMaterial->color = Color(0x9aa0a6);
    groundMaterial->roughness = 0.95f;
    groundMaterial->metalness = 0.f;
    auto ground = Mesh::create(PlaneGeometry::create(20, 20), groundMaterial);
    ground->name = "Ground";
    ground->rotation.x = -math::PI / 2;
    ground->receiveShadow = true;
    scene.add(ground);

    auto boxMaterial = MeshStandardMaterial::create();
    boxMaterial->color = Color(0xd08b52);
    boxMaterial->roughness = 0.5f;
    auto box = Mesh::create(BoxGeometry::create(1, 1, 1), boxMaterial);
    box->name = "Box";
    box->position.set(0, 0.5f, 0);
    box->castShadow = true;
    box->receiveShadow = true;
    scene.add(box);

    // A new document starts where the editor started: a free perspective view,
    // and nothing to chase.
    setOrthographic(false);
    setViewPreset(ViewPreset::User);
    camera_.position.set(6, 5, 8);
    orbit_->target.set(0, 0.5f, 0);
    followSelection_ = false;
    documentView_ = false;

    document_.setDirty(false);
    selectObject(box.get());
}

void EditorApp::openScene(const std::filesystem::path& path) {

    if (rejectWhilePlaying("Open Scene")) return;

    selectObject(nullptr);
    std::string error;
    if (!document_.open(path, &error)) {
        log("open failed: " + error);
        return;
    }
    commands_.clear();
    logWarnings();
    // Where this document asks to be seen from, if it says. A file that says
    // nothing keeps the camera where it was, which is what every version of the
    // editor before this one did.
    applyDocumentView();
    applyDocumentRender();
    settings_.addRecentFile(path);
    settings_.sceneDir = path.parent_path().string();
    log("opened " + path.filename().string());
}

void EditorApp::openExample(const std::string& slug) {

    if (rejectWhilePlaying("Open Example")) return;

    const auto* example = examples::find(slug);
    const auto json = examples::json(slug);
    if (!example || json.empty()) {
        log("no example named \"" + slug + "\" ships with this build");
        return;
    }

    selectObject(nullptr);
    std::string error;
    // The same loader path a file goes through — and the same listeners fire,
    // so the markers, the overlays and (for a scene that had one) the robot
    // rearticulation all happen exactly as they do for Open.
    if (!document_.openJson(json, &error)) {
        log("open example failed: " + error);
        return;
    }
    commands_.clear();
    logWarnings();
    // Untitled and clean: it came from the binary, not from a file, so Save
    // will ask where to put it rather than writing over anything.
    //
    // An example is something you asked to LOOK at, so it is framed — unless it
    // authored a vantage of its own, which is a considered answer to the same
    // question and beats an automatic three-quarter view.
    if (!applyDocumentView()) frameDocument();
    applyDocumentRender();
    log("opened example \"" + std::string(example->label) + "\" - " + std::string(example->summary));
}

bool EditorApp::parseViewSpec(const std::string& text, Vector3& position, Vector3& target) {

    // The parse itself is in the library (extras/editor/ViewSpec.hpp), because
    // the editor is no longer the only front end that places a camera from a
    // document's authored vantage — threepp_player reads the same
    // userData["editorView"] and --shot takes the same text. This stays as the
    // editor's public spelling of it; main.cpp's --shot handling and every
    // caller below are unchanged.
    return editor::parseViewSpec(text, position, target);
}

bool EditorApp::applyDocumentView() {

    documentView_ = false;
    // A document that says nothing about following clears it: follow belongs to
    // a subject, and opening a different document is a different subject. It is
    // deliberately not sticky across an Open for the same reason the selection
    // is not.
    followSelection_ = false;

    const auto& userData = document_.scene().userData;
    const auto read = [&](const char* key) {
        const auto it = userData.find(key);
        if (it == userData.end() || it->second.type() != typeid(std::string)) return std::string();
        return std::any_cast<const std::string&>(it->second);
    };

    if (const auto follow = read("editorFollow"); !follow.empty()) {
        if (auto* subject = document_.scene().getObjectByName(follow)) {
            selectObject(subject);
            followSelection_ = true;
        } else {
            log("editorFollow names \"" + follow + "\", which is not in this scene - ignored");
        }
    }

    const auto view = read("editorView");
    if (view.empty()) return false;

    Vector3 position;
    Vector3 target;
    if (!parseViewSpec(view, position, target)) {
        log("editorView is \"" + view + "\" - want px,py,pz@tx,ty,tz; ignored");
        return false;
    }

    // A placement, not a projection: an authored vantage is a perspective one,
    // and arriving in a leftover axis view would be a different picture than
    // the one that was authored.
    setOrthographic(false);
    setViewPreset(ViewPreset::User);
    camera_.position.copy(position);
    orbit_->target.copy(target);
    camera_.lookAt(target);
    documentView_ = true;
    return true;
}

void EditorApp::applyDocumentRender() {

    // value_or(renderDefaults_), not "leave the renderer alone": a document that
    // says nothing about fog wants the editor's fog, not the ground mist the
    // document opened before it was carrying. read() already layers whatever the
    // document DOES say over the same defaults, key by key.
    const auto config = RenderConfig::read(document_.scene(), renderDefaults_)
                                .value_or(renderDefaults_);
    config.apply(*renderer_);
}

void EditorApp::frameDocument() {

    Box3 box;
    box.setFromObject(document_.scene());
    if (box.isEmpty()) return;

    const Vector3 centre = box.getCenter();
    const Vector3 size = box.getSize();
    const float radius = std::max({size.x, size.y, size.z}) * 0.5f;
    const float distance =
            std::max(radius / std::tan(math::degToRad(camera_.fov) * 0.5f), 1.f) * 1.15f;

    // A three-quarter view from above, rather than "keep the current angle":
    // there is no previous framing to preserve, the document was just replaced.
    Vector3 direction(0.45f, 0.5f, 1.f);
    direction.normalize();

    setOrthographic(false);
    setViewPreset(ViewPreset::User);
    orbit_->target.copy(centre);
    camera_.position.copy(centre).add(direction.multiplyScalar(distance));
}

void EditorApp::saveScene() {

    // What is in the scene right now is the simulation, not the document: a
    // save here would quietly write the fallen boxes back as the authored pose.
    if (rejectWhilePlaying("Save")) return;

    if (!document_.hasPath()) {
        pendingDialog_ = PendingDialog::SaveAs;
        fileBrowser_.open("Save Scene As", FileBrowser::Mode::Save,
                          settings_.sceneDir, {".json"}, "scene.json");
        return;
    }
    saveSceneAs(document_.path());
}

void EditorApp::saveSceneAs(const std::filesystem::path& path) {

    if (rejectWhilePlaying("Save As")) return;

    // The look goes into the file with the geometry. Written as a difference
    // from the editor's startup state, so a document nobody adjusted the
    // renderer for carries no render block at all (write() erases it) and one
    // that did carries only what was adjusted.
    RenderConfig::capture(*renderer_).write(document_.scene(), renderDefaults_);

    std::string error;
    if (!document_.saveAs(path, &error)) {
        log("save failed: " + error);
        return;
    }
    logWarnings();
    settings_.addRecentFile(path);
    settings_.sceneDir = path.parent_path().string();
    log("saved " + path.filename().string());
}

void EditorApp::setImageStorage(ImageStorage storage) {

    document_.setImageStorage(storage);
    settings_.imageStorage = storage;
    log(storage == ImageStorage::Reference ? "textures will be referenced on save"
                                           : "textures will be embedded on save");
}

void EditorApp::setModelStorage(ModelStorage storage) {

    document_.setModelStorage(storage);
    settings_.modelStorage = storage;
    log(storage == ModelStorage::Reference ? "imported models will be referenced on save"
                                           : "imported models will be embedded on save");
}

void EditorApp::unlinkSelectedAsset() {

    if (rejectWhilePlaying("Unlink Imported Asset")) return;

    auto* selected = selection_.get();
    if (!selected) return;

    const auto source = assetSource(*selected);
    if (source.empty()) return;

    auto* target = selected;
    commands_.execute(makeProperty<std::string>(
            "Unlink Asset", "assetSource:" + selected->uuid,
            [target](const std::string& value) { setAssetSource(*target, value); },
            source.generic_string(), std::string{}));
    document_.setDirty(true);

    log("unlinked " + (selected->name.empty() ? selected->type() : selected->name) +
        " from " + source.filename().string() + " - it will be written out in full");
}

void EditorApp::importModel(const std::filesystem::path& path) {

    // Loading happens on a worker so the UI never freezes; pollImports picks
    // the queue up, shows the toast and finalizes on the main thread.
    importQueue_.push_back(path);
    log("import queued: " + path.filename().string());
}

void EditorApp::pollImports(float dt) {

    uiTime_ += dt;
    if (statusFlashRemaining_ > 0.f && (statusFlashRemaining_ -= dt) <= 0.f) statusFlash_.clear();

    if (!activeImport_ && !importQueue_.empty()) {
        auto path = importQueue_.front();
        importQueue_.pop_front();
        activeImport_ = std::make_unique<ActiveImport>();
        activeImport_->path = path;
        // Loader exceptions surface through the future and are rethrown on
        // the main thread in the get() below.
        activeImport_->future = std::async(std::launch::async, [path]() -> std::shared_ptr<Object3D> {
            auto extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension == ".urdf" || extension == ".xacro") {
                URDFLoader loader;
                return loader.load(path);
            }
            ModelLoader loader;
            return loader.load(path);
        });
    }
    if (!activeImport_) return;
    activeImport_->elapsed += dt;

    // Finishing during play would add into a scene the stop-restore throws
    // away (and leave a dangling undo entry); park the result until stopped.
    if (isPlaying()) return;
    if (activeImport_->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;

    const auto path = activeImport_->path;
    const auto elapsed = activeImport_->elapsed;
    std::shared_ptr<Object3D> group;
    std::string error;
    try {
        group = activeImport_->future.get();
        if (!group || group->children.empty()) error = "no importable content in the file";
    } catch (const std::exception& e) {
        error = e.what();
    }
    activeImport_.reset();

    if (!error.empty()) {
        log("import failed: " + path.filename().string() + " - " + error);
        importError_ = path.filename().string() + "\n\n" + error;
        return;
    }

    std::size_t nodes = 0;
    group->traverse([&nodes](Object3D&) { ++nodes; });

    // Remember the file. Whether the document then writes the subtree out or
    // just points at it is the save-time storage choice; either way this is
    // what makes the choice available later.
    //
    // Normalised to match what ObjectLoader stamps when it re-imports, so that
    // saving a scene, reopening it and saving again produces the same bytes
    // rather than converging on the second try.
    std::error_code pathEc;
    const auto canonical = std::filesystem::weakly_canonical(path, pathEc);
    setAssetSource(*group, pathEc ? path : canonical);

    // A robot records where it came from: the joint table cannot be
    // serialised, so the document keeps a reference and rebuilds on load.
    std::size_t dof = 0;
    if (auto* robot = group->as<Robot>()) {
        dof = robot->numDOF();
        RobotConfig config;
        config.urdf = path.string();
        config.joints = robot->jointValues();
        config.write(*robot);
        // URDFLoader builds the collision hulls visible; they are wireframe
        // duplicates sitting on the visual meshes, so start them hidden.
        robot->showColliders(config.showColliders);
    }

    group->name = ObjectFactory::uniqueName(document_.scene(), path.stem().string());
    addObject(group, document_.scene(), "Import " + path.filename().string());
    settings_.modelDir = path.parent_path().string();

    char message[192];
    if (dof > 0) {
        std::snprintf(message, sizeof(message), "Imported %s (%zu nodes, %zu joints, %.1fs)",
                      path.filename().string().c_str(), nodes, dof, elapsed);
    } else {
        std::snprintf(message, sizeof(message), "Imported %s (%zu nodes, %.1fs)",
                      path.filename().string().c_str(), nodes, elapsed);
    }
    log(message);
    flashStatus(message);
}

float EditorApp::hierarchyPx() const {

    return settings_.hierarchyWidth * contentScale_;
}

float EditorApp::inspectorPx() const {

    return settings_.inspectorWidth * contentScale_;
}

float EditorApp::bottomHeightLimit() const {

    const float s = contentScale_;
    // Against the render surface rather than the ImGui viewport, because
    // cameraDockRect() asks for this outside the frame.
    const float total = static_cast<float>(renderer_->size().height());
    // A viewport strip has to survive: dragging the panel up to the toolbar
    // would leave nothing to drag it back down against.
    const float free = total - menuHeight_ - toolbarHeight_ - statusHeight_ - 140.f * s;

    return std::max(free / s, EditorSettings::minBottomHeight);
}

float EditorApp::bottomPanelPx() const {

    return std::clamp(settings_.bottomPanelHeight,
                      EditorSettings::minBottomHeight, bottomHeightLimit()) *
           contentScale_;
}

float EditorApp::collapsedBottomPx() const {

    return ImGui::GetFrameHeight() + 6 * contentScale_;
}

float EditorApp::bottomBandPx() const {

    // Exactly the panel, with nothing reserved for the splitter: a band held
    // clear for the grip is a strip of bare viewport between the side panels
    // and the bottom one — a seam across the whole window. The grip overlays
    // the boundary instead (drawHeightSplitter), which costs the side panels
    // their last few pixels and costs the layout nothing.
    return bottomPanelOpen_ ? bottomPanelPx() : collapsedBottomPx();
}

// One implementation for both axes. `horizontal` is the strip's long axis being
// horizontal — the handle that moves things up and down.
void EditorApp::drawSplitterStrip(const char* id, float x, float top, float width, float height,
                                  bool horizontal, float& value, float sign, float lo, float hi) {

    const float s = contentScale_;

    ImGui::SetNextWindowPos({x, top});
    ImGui::SetNextWindowSize({width, height});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0.f, 0.f, 0.f, 0.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.f, 0.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);

    if (ImGui::Begin(id, nullptr, layout::barFlags)) {

        ImGui::InvisibleButton("##grip", {std::max(width, 1.f), std::max(height, 1.f)});

        const bool active = ImGui::IsItemActive();
        if (active || ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(horizontal ? ImGuiMouseCursor_ResizeNS : ImGuiMouseCursor_ResizeEW);
            // Only hint at the handle once the pointer finds it; an always-on
            // divider would just be chrome.
            auto* draw = ImGui::GetWindowDrawList();
            const auto min = ImGui::GetWindowPos();
            const auto colour = ImGui::GetColorU32(active ? theme::accent() : theme::muted());
            if (horizontal) {
                draw->AddRectFilled({min.x, min.y + height * 0.5f - 1.f * s},
                                    {min.x + width, min.y + height * 0.5f + 1.f * s}, colour);
            } else {
                draw->AddRectFilled({min.x + width * 0.5f - 1.f * s, min.y},
                                    {min.x + width * 0.5f + 1.f * s, min.y + height}, colour);
            }
        }
        if (active) {
            const auto& delta = ImGui::GetIO().MouseDelta;
            value = std::clamp(value + (horizontal ? delta.y : delta.x) * sign / s, lo, hi);
        }
    }
    ImGui::End();

    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

void EditorApp::drawSplitter(const char* id, float x, float top, float height,
                             float& width, float sign) {

    drawSplitterStrip(id, x, top, layout::splitterThickness * contentScale_, height,
                      false, width, sign,
                      EditorSettings::minPanelWidth, EditorSettings::maxPanelWidth);
}

void EditorApp::drawHeightSplitter(const char* id, float x, float top, float width,
                                   float& height) {

    // Dragging up grows the panel, hence the -1.
    drawSplitterStrip(id, x, top, width, layout::splitterThickness * contentScale_,
                      true, height, -1.f,
                      EditorSettings::minBottomHeight, bottomHeightLimit());
}

void EditorApp::flashStatus(std::string message) {

    statusFlash_ = std::move(message);
    statusFlashRemaining_ = 4.f;
}

void EditorApp::drawImportToast() {

    if (!activeImport_ && importQueue_.empty()) return;

    const auto* viewport = ImGui::GetMainViewport();
    const float s = contentScale_;

    std::string text = activeImport_
                               ? "Importing " + activeImport_->path.filename().string()
                               : std::string("Import pending");
    if (activeImport_ && activeImport_->elapsed >= 1.f) {
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "  (%.0fs)", activeImport_->elapsed);
        text += suffix;
    }
    if (!importQueue_.empty()) text += "  +" + std::to_string(importQueue_.size()) + " queued";

    const float radius = 7.f * s;
    const ImVec2 textSize = ImGui::CalcTextSize(text.c_str());
    const float height = std::max(textSize.y, radius * 2) + 16 * s;
    const float width = textSize.x + radius * 2 + 34 * s;

    // Bottom-left of the viewport (the camera preview owns the bottom-right).
    const float bottom = viewport->Size.y - statusHeight_ - bottomBandPx();
    ImGui::SetNextWindowPos({viewport->Pos.x + hierarchyPx() + 12 * s,
                             viewport->Pos.y + bottom - height - 12 * s});
    ImGui::SetNextWindowSize({width, height});
    ImGui::SetNextWindowBgAlpha(0.9f);

    if (ImGui::Begin("##importToast", nullptr,
                     layout::barFlags | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoFocusOnAppearing)) {
        auto* draw = ImGui::GetWindowDrawList();
        const ImVec2 pos = ImGui::GetWindowPos();
        const ImVec2 center(pos.x + 10 * s + radius, pos.y + height * 0.5f);
        const float t = uiTime_ * 6.f;
        draw->PathArcTo(center, radius, t, t + 4.7f, 24);
        draw->PathStroke(ImGui::GetColorU32(theme::accent()), 0, 2.5f * s);
        ImGui::SetCursorPos({10 * s + radius * 2 + 8 * s, (height - textSize.y) * 0.5f});
        ImGui::TextUnformatted(text.c_str());
    }
    ImGui::End();
}

void EditorApp::setEnvironment(const std::filesystem::path& path, bool alsoBackground) {

    // The snapshot carries the environment, so this would be undone by Stop.
    if (rejectWhilePlaying("Set Environment")) return;

    RGBELoader loader;
    std::shared_ptr<Texture> texture;
    try {
        texture = loader.load(path);
    } catch (const std::exception& e) {
        log("environment failed: " + std::string(e.what()));
        return;
    }
    if (!texture) {
        log("environment failed: " + path.filename().string());
        return;
    }

    auto& scene = document_.scene();
    scene.environment = texture;
    if (alsoBackground) scene.background = Background(texture);
    document_.setDirty(true);
    settings_.environmentDir = path.parent_path().string();
    log("environment set from " + path.filename().string());
}

void EditorApp::clearEnvironment() {

    if (rejectWhilePlaying("Clear Environment")) return;

    auto& scene = document_.scene();
    scene.environment = nullptr;
    scene.background = Background(Color(0x1c1f24));
    document_.setDirty(true);
    log("environment cleared");
}

void EditorApp::assignTextureToSlot(const std::filesystem::path& path) {

    const auto pending = pendingTextureSlot_;
    pendingTextureSlot_ = {};

    if (pending.objectUuid.empty()) {
        assignTextureToSelection(path);
        return;
    }

    // The dialog spanned frames, so nothing about the target can be assumed
    // still there: the object may have been deleted, and a play/stop in between
    // replaces the whole graph (the uuid survives that, a Material* would not).
    auto* owner = findByUuid(document_.scene(), pending.objectUuid);
    auto material = owner ? owner->material() : nullptr;
    if (!material) {
        log("texture not assigned - the object is no longer in the scene");
        return;
    }

    // The material can also have been swapped for one without this slot.
    const auto slots = textureSlotsOf(*material);
    const auto match = std::find_if(slots.begin(), slots.end(),
                                    [&pending](const MaterialTextureSlot& slot) {
                                        return slot.name == pending.slot;
                                    });
    if (match == slots.end()) {
        log("texture not assigned - " + pending.slot + " is no longer a slot on this material");
        return;
    }

    applyTextureToSlot(path, {material.get(), match->set, match->current, match->name, match->srgb}, "");
}

void EditorApp::assignTextureToSelection(const std::filesystem::path& path) {

    auto* selected = selection_.get();
    if (!selected) {
        log("no selection to texture");
        return;
    }
    auto material = selected->material();
    if (!material) {
        log("selected object has no material");
        return;
    }

    // Nothing said where this should go, so let the file name say. A texture
    // named "..._normal" almost never wants to be the base colour map, and
    // dropping it there is a change the user then has to notice and undo.
    const auto slots = textureSlotsOf(*material);
    if (slots.empty()) {
        log("selected object's material has no texture slots");
        return;
    }

    const auto wanted = textureSlotFromFilename(path.stem().string());
    auto match = std::find_if(slots.begin(), slots.end(),
                              [&wanted](const MaterialTextureSlot& slot) { return slot.name == wanted; });

    const bool inferred = match != slots.end();
    // Falls back to the first slot, which is the base colour map wherever the
    // material has one.
    const auto& slot = inferred ? *match : slots.front();

    applyTextureToSlot(path, {material.get(), slot.set, slot.current, slot.name, slot.srgb},
                       inferred ? " (matched by name)" : "");
}

void EditorApp::resolveTextureDrops(float mouseX, float mouseY) {

    if (pendingTextureDrops_.empty()) return;

    auto dropped = std::move(pendingTextureDrops_);
    pendingTextureDrops_.clear();

    for (const auto& path : dropped) {

        // A drop straight onto a slot row is an explicit answer; take it over
        // anything the file name might suggest. When the cursor is nowhere near
        // a row - or the window was not focused, so the position is stale - this
        // finds nothing and the name-based path below decides instead.
        const auto hit = std::find_if(frameTextureSlots_.begin(), frameTextureSlots_.end(),
                                      [mouseX, mouseY](const FrameTextureSlot& slot) {
                                          return slot.contains(mouseX, mouseY);
                                      });

        if (hit != frameTextureSlots_.end()) {
            applyTextureToSlot(path, hit->target, "");
            continue;
        }

        assignTextureToSelection(path);
    }
}

void EditorApp::applyTextureToSlot(const std::filesystem::path& path,
                                   const TextureSlotTarget& target,
                                   const std::string& note) {

    // Both texture paths (dialog and drop) end here, so this is the gate.
    if (rejectWhilePlaying(target.slot.empty() ? "Set texture" : target.slot.c_str())) return;
    if (!target.material || !target.setter) return;

    TextureLoader loader;
    std::shared_ptr<Texture> texture;
    try {
        // The slot decides: a data map decoded as sRGB shades wrong, and that
        // is a bug you chase in the renderer rather than here.
        texture = loader.load(path, target.srgb ? ColorSpace::sRGB : ColorSpace::NoColorSpace);
    } catch (const std::exception& e) {
        log("texture failed: " + std::string(e.what()));
        return;
    }
    if (!texture) {
        log("texture failed: " + path.filename().string());
        return;
    }

    commands_.execute(std::make_unique<SetMaterialMapCommand>(
            *target.material, target.slot, target.setter, target.current, texture));
    document_.setDirty(true);
    settings_.textureDir = path.parent_path().string();
    log(target.slot + " set from " + path.filename().string() + note);
}

void EditorApp::assignScript(Object3D& object, const std::filesystem::path& path) {

    if (rejectWhilePlaying("Attach Script")) return;

    const auto before = ScriptConfig::read(object).value_or(ScriptConfig{});

    ScriptConfig after;
    if (!path.empty()) {
        // Store forward slashes: a saved document should not read differently
        // depending on which platform wrote it.
        auto text = path.generic_string();
        after.path = text;
        // Re-attaching the same file keeps whatever the user had dialled in;
        // a different file starts clean, since its parameters are its own.
        if (before.path == text) after.fields = before.fields;
    }

    auto* target = &object;
    commands_.execute(makeProperty<ScriptConfig>(
            path.empty() ? "Clear Script" : "Set Script", "script:" + object.uuid,
            [target](const ScriptConfig& value) { value.write(*target); },
            before, after));
    document_.setDirty(true);

    if (path.empty()) {
        log("script cleared on " + (object.name.empty() ? object.type() : object.name));
        return;
    }
    settings_.scriptDir = path.parent_path().string();
    log("script " + path.filename().string() + " attached to " +
        (object.name.empty() ? object.type() : object.name));
}

void EditorApp::setInlineScript(Object3D& object, const std::string& source, const std::string& label) {

    // Belt and braces: both callers already hold off (the script panel goes
    // read-only, an external save parks until Stop), but this is where the
    // userData edit actually happens.
    if (rejectWhilePlaying(label.c_str())) return;

    const auto before = ScriptConfig::read(object).value_or(ScriptConfig{});

    ScriptConfig after = before;
    after.setSource(source);
    // Editing the same inline script keeps the parameter values the user
    // dialled in; arriving from a .py drops them, because they belonged to that
    // file's class and mean nothing to this one. Anything the new class no
    // longer exposes is pruned by the inspector on the next edit.
    if (!before.isInline()) after.fields.clear();
    if (source.empty()) after = ScriptConfig{};

    auto* target = &object;
    commands_.execute(makeProperty<ScriptConfig>(
            label, "scriptSource:" + object.uuid,
            [target](const ScriptConfig& value) { value.write(*target); },
            before, after));
    document_.setDirty(true);
}

#ifdef THREEPP_EDITOR_WITH_PYTHON
const scripting::Inspection& EditorApp::inspectScriptSource(const std::string& uuid,
                                                            const std::string& label,
                                                            const std::string& source) {

    // Keyed on a hash of the text rather than a write time, since there is no
    // file — but the same contract: the section refreshes as soon as the source
    // changes, which for inline scripts is the moment Apply commits.
    const auto stamp = std::hash<std::string>{}(source);

    auto it = scriptSourceInspections_.find(uuid);
    if (it != scriptSourceInspections_.end() && it->second.hash == stamp) return it->second.inspection;

    CachedSourceInspection entry;
    entry.hash = stamp;
    entry.inspection = scripting::inspectSource(source, uuid, label);
    return (scriptSourceInspections_[uuid] = std::move(entry)).inspection;
}

const scripting::Inspection& EditorApp::inspectScript(const std::string& path) {

    // Keyed on the file's write time, so saving the .py in another editor is
    // picked up on the next inspector frame — the same expectation Play sets.
    std::filesystem::file_time_type stamp{};
    std::error_code ec;
    const auto written = std::filesystem::last_write_time(path, ec);
    if (!ec) stamp = written;

    auto it = scriptInspections_.find(path);
    if (it != scriptInspections_.end() && it->second.stamp == stamp) return it->second.inspection;

    CachedInspection entry;
    entry.stamp = stamp;
    entry.inspection = scripting::inspect(path);
    return (scriptInspections_[path] = std::move(entry)).inspection;
}
#endif


// ------------------------------------------------------------- graph editing

void EditorApp::addObject(const std::shared_ptr<Object3D>& object, Object3D& parent, const std::string& label) {

    // The one gate for every "Add": the menus, the hierarchy, spline points and
    // the drop handler all arrive here. `pollImports` deliberately does not —
    // it parks a queued import until Stop rather than refusing it.
    if (rejectWhilePlaying(label.c_str())) return;

    if (!object) return;
    auto command = std::make_unique<AddObjectCommand>(parent, object, label);
    auto* raw = object.get();
    commands_.execute(std::move(command));
    document_.setDirty(true);
    selectObject(raw);
    scrollTo_ = raw;
}

namespace {

    // Replace a generator's output in ONE undo step. Two commands (remove the old,
    // add the new) would be two, because the stack's transactions coalesce by
    // merge key and these do not share one — and half-undoing a regenerate leaves
    // a scene with two generations of content in it.
    //
    // Both nodes are held by shared_ptr for the whole life of the command, which
    // is what lets undo put the previous generation back rather than re-running
    // the script to approximate it. Targets re-resolve by uuid, so the command
    // survives the play-stop graph swap like every other one.
    // Also carries the CONFIG, so that clearing a generator — which removes its
    // script and its output together — stays one undo step. Regenerate passes the
    // same config for before and after, where writing it is a no-op.
    class RegenerateCommand: public threepp::editor::Command {

    public:
        RegenerateCommand(Object3D& carrier, std::shared_ptr<Object3D> previous,
                          std::shared_ptr<Object3D> next, GeneratorConfig before,
                          GeneratorConfig after, std::string label)
            : carrier_(&carrier), previous_(std::move(previous)), next_(std::move(next)),
              before_(std::move(before)), after_(std::move(after)), label_(std::move(label)),
              carrierUuid_(carrier.uuid) {}

        void redo() override {

            if (previous_) previous_->removeFromParent();
            if (!carrier_) return;
            if (next_) carrier_->add(next_);
            after_.write(*carrier_);
        }

        void undo() override {

            if (next_) next_->removeFromParent();
            if (!carrier_) return;
            if (previous_) carrier_->add(previous_);
            before_.write(*carrier_);
        }

        [[nodiscard]] std::string name() const override { return label_; }

        // Never merged: two regenerates are two distinct generations of content,
        // and collapsing them would drop the middle one's output on the floor.
        bool mergeWith(const Command&) override { return false; }

        [[nodiscard]] bool rebind(Object3D& root) override {

            Object3D* found = nullptr;
            root.traverse([&](Object3D& o) {
                if (!found && o.uuid == carrierUuid_) found = &o;
            });
            if (!found) return false;
            carrier_ = found;
            return true;
        }

    private:
        Object3D* carrier_;
        std::shared_ptr<Object3D> previous_;
        std::shared_ptr<Object3D> next_;
        GeneratorConfig before_;
        GeneratorConfig after_;
        std::string label_;
        std::string carrierUuid_;
    };

}// namespace

std::string EditorApp::generatorTemplate() {

    return "# Scene generator. Runs when you press Regenerate, never on open.\n"
           "# Module-level names become editable parameters in the inspector.\n"
           "import math\n"
           "import random\n"
           "import threepp\n"
           "from threepp import editor\n"
           "\n"
           "count = 60\n"
           "seed = 1\n"
           "radius = 8.0\n"
           "\n"
           "random.seed(seed)  # same seed, same scene\n"
           "\n"
           "geometry = threepp.BoxGeometry(0.5, 0.5, 0.5)\n"
           "material = threepp.MeshStandardMaterial()\n"
           "material.color = threepp.Color(0x88aa55)\n"
           "\n"
           "field = threepp.InstancedMesh(geometry, material, count)\n"
           "field.name = \"Scatter\"\n"
           "for i in range(count):\n"
           "    angle = random.uniform(0, 2 * math.pi)\n"
           "    r = radius * math.sqrt(random.random())\n"
           "    m = threepp.Matrix4()\n"
           "    m.make_rotation_y(random.uniform(0, 2 * math.pi))\n"
           "    m.set_position(r * math.cos(angle), 0.25, r * math.sin(angle))\n"
           "    field.set_matrix_at(i, m)\n"
           "field.instance_matrix_needs_update()\n"
           "\n"
           "editor.add(field)\n";
}

bool EditorApp::regenerate(Object3D& carrier) {

    if (rejectWhilePlaying("Regenerate")) return false;

    const auto config = GeneratorConfig::read(carrier);
    if (!config) {
        log("regenerate: no generator script on this object");
        return false;
    }

#ifndef THREEPP_EDITOR_WITH_PYTHON
    log("regenerate: built without Python scripting - the script is saved, not run");
    return false;
#else
    // The output is built DETACHED and attached only on success, so a script that
    // raises leaves the document exactly as it was. This is why there is no
    // rollback path below: there is nothing to roll back.
    auto output = Group::create();
    output->name = "Generated";
    output->userData[GeneratorConfig::generatedKey] = std::string("1");

    std::string error;
    {
        scripting::authoringSink() = output.get();
        scripting::authoringScene() = &document_.scene();
        // Cleared on every exit path: a stale sink would let a later behaviour
        // script append into a node nobody is going to commit.
        struct SinkGuard {
            ~SinkGuard() {
                scripting::authoringSink() = nullptr;
                scripting::authoringScene() = nullptr;
            }
        } guard;

        error = scripting::runAuthoringSource(config->source, "generator");
    }

    if (!error.empty()) {
        log("regenerate failed: " + error);
        return false;
    }

    if (output->children.empty()) {
        // Not an error — a script may legitimately have decided to build nothing
        // — but silence here reads as a broken button, so say it.
        log("regenerate: the script added nothing");
    }

    // Hold the outgoing generation alive for undo. `generatedChild` returns a raw
    // pointer into the graph, and removeFromParent would be the last owner.
    std::shared_ptr<Object3D> previous;
    if (auto* old = GeneratorConfig::generatedChild(carrier)) {
        for (auto& child : carrier.children) {
            if (child == old) {
                previous = old->shared_from_this();
                break;
            }
        }
    }

    commands_.execute(std::make_unique<RegenerateCommand>(carrier, previous, output, *config,
                                                          *config, "Regenerate"));
    document_.setDirty(true);
    log("regenerate: " + std::to_string(output->children.size()) + " object(s) generated");
    return true;
#endif
}

// The output goes with the script. Leaving it behind orphans content that still
// carries the generated tag, which a later generator on the same object would
// silently adopt and replace — and "clear the generator" plainly means the whole
// thing. Same call the spline makes when its mesh kind goes to none: the derived
// node is destroyed, and undo re-creates it.
void EditorApp::clearGenerator(Object3D& carrier) {

    if (rejectWhilePlaying("Clear Generator")) return;

    const auto config = GeneratorConfig::read(carrier);
    if (!config) return;

    std::shared_ptr<Object3D> previous;
    if (auto* old = GeneratorConfig::generatedChild(carrier)) {
        for (auto& child : carrier.children) {
            if (child == old) {
                previous = old->shared_from_this();
                break;
            }
        }
    }

    // Selecting the output and then clearing would leave the gizmo attached to a
    // detached node ("must be a part of the scene graph" spam).
    if (previous && selection_.get() == previous.get()) selectObject(nullptr);

    commands_.execute(std::make_unique<RegenerateCommand>(carrier, previous, nullptr, *config,
                                                          GeneratorConfig{}, "Clear Generator"));
    document_.setDirty(true);
    log("generator cleared" + std::string(previous ? " with its output" : ""));
}

std::string EditorApp::applyGeneratorSource(Object3D& target, const std::string& text) {

    const auto before = GeneratorConfig::read(target).value_or(GeneratorConfig{});
    auto after = before;
    after.source = ScriptWorkspace::normalize(text);

    auto* carrier = &target;
    commands_.execute(makeProperty<GeneratorConfig>(
            "Edit Generator", "generator:" + target.uuid,
            [carrier](const GeneratorConfig& value) { value.write(*carrier); }, before, after));
    document_.setDirty(true);
    return after.source;
}

void EditorApp::addSplinePoint(Object3D& spline, std::size_t index, const std::string& label) {

    // Builds its own AddObjectCommand rather than going through addObject().
    if (rejectWhilePlaying(label.c_str())) return;

    if (!SplineConfig::isSpline(spline)) return;

    const auto points = SplineConfig::controlPoints(spline);
    const auto count = points.size();
    const auto slot = std::min(index, count);

    // Where the new point goes has one rule: the curve must visibly change.
    // Between two points that is their midpoint; past either end it is the end
    // segment continued, so "Add Point" extends the spline rather than
    // stacking a second point on the last one.
    Vector3 position;
    if (count == 0) {
        // Nothing to extend; the origin of the spline's own space.
    } else if (slot == 0) {
        position.copy(points.front());
        if (count > 1) position.sub(points[1]).add(points.front());
    } else if (slot >= count) {
        position.copy(points.back());
        if (count > 1) position.sub(points[count - 2]).add(points.back());
        else position.x += 1.f;
    } else {
        position.copy(points[slot - 1]).add(points[slot]).multiplyScalar(0.5f);
    }

    auto point = ObjectFactory::createSplinePoint(spline);
    point->position.copy(position);

    auto* raw = point.get();
    // `slot` is a POINT index; AddObjectCommand takes a CHILD index, and the
    // generated mesh sits among the same children without being a point.
    commands_.execute(std::make_unique<AddObjectCommand>(
            spline, point, label, SplineConfig::childSlotForPointIndex(spline, slot)));
    document_.setDirty(true);
    selectObject(raw);
    scrollTo_ = raw;
}

void EditorApp::deleteSelected() {

    // The one that is not merely confusing: PhysX holds an actor per body, and
    // taking the node out from under it while the world is stepping is a
    // use-after-free rather than a lost edit.
    if (rejectWhilePlaying("Delete")) return;

    auto* selected = selection_.get();
    if (!selected || selected == &document_.scene()) return;
    if (document_.isEditorOnly(*selected)) return;

    auto command = std::make_unique<RemoveObjectCommand>(
            *selected, "Delete " + (selected->name.empty() ? selected->type() : selected->name));
    if (!command->valid()) return;

    selectObject(nullptr);
    commands_.execute(std::move(command));
    document_.setDirty(true);
}

void EditorApp::duplicateSelected() {

    if (rejectWhilePlaying("Duplicate")) return;

    auto* selected = selection_.get();
    if (!selected || selected == &document_.scene() || !selected->parent) return;
    if (document_.isEditorOnly(*selected)) return;

    auto copy = selected->clone();
    if (!copy) {
        log("cannot duplicate " + selected->type());
        return;
    }
    copy->name = ObjectFactory::uniqueName(
            document_.scene(), selected->name.empty() ? selected->type() : selected->name);

    // Object3D::clone() shares materials with the source (three.js does the
    // same). In an editor that is a trap: recolouring the copy would recolour
    // the original. Geometry stays shared — that IS the desired behaviour, and
    // it is what keeps duplication cheap.
    copy->traverse([](Object3D& object) {
        auto* withMaterials = dynamic_cast<ObjectWithMaterials*>(&object);
        if (!withMaterials || withMaterials->materials().empty()) return;
        std::vector<std::shared_ptr<Material>> clones;
        clones.reserve(withMaterials->materials().size());
        for (const auto& material : withMaterials->materials()) {
            clones.push_back(material ? material->clone() : nullptr);
        }
        withMaterials->setMaterials(clones);
    });

    addObject(copy, *selected->parent, "Duplicate " + copy->name);
}

void EditorApp::focusSelected() {

    auto* selected = selection_.get();
    if (!selected) return;

    Box3 box;
    box.setFromObject(*selected);
    if (box.isEmpty()) {
        selected->getWorldPosition(orbit_->target);
        return;
    }

    Vector3 centre = box.getCenter();
    const Vector3 size = box.getSize();
    const float radius = std::max({size.x, size.y, size.z}) * 0.5f;
    // Pull back far enough that the bounding sphere fits the vertical FOV, with
    // a little air around it.
    float distance = std::max(radius / std::tan(math::degToRad(camera_.fov) * 0.5f), 0.5f) * 1.6f;

    Camera& camera = viewCamera();

    Vector3 direction;
    direction.subVectors(camera.position, orbit_->target);
    if (direction.length() < 1e-4f) direction.set(1, 1, 1);
    direction.normalize();

    if (orthographic_) {
        // Framing an ortho view is a frustum change, not a move. The camera
        // still goes clear of the scene so nothing in front of the subject is
        // clipped away.
        setOrthoHeight(std::max(radius, 0.25f) * 2.f * 1.6f);
        distance = std::max(distance, sceneClearDistance(centre));
    }

    orbit_->target.copy(centre);
    camera.position.copy(centre).add(direction.multiplyScalar(distance));
}

void EditorApp::undo() {

    // Not just for symmetry: the stack replays setters into the scene, and
    // while playing that is the simulation's scene, which Stop then discards.
    if (rejectWhilePlaying("Undo")) return;
    if (commands_.undo()) document_.setDirty(true);
}

void EditorApp::redo() {

    if (rejectWhilePlaying("Redo")) return;
    if (commands_.redo()) document_.setDirty(true);
}

void EditorApp::reparent(Object3D& object, Object3D& newParent) {

    if (rejectWhilePlaying("Reparent")) return;

    auto command = std::make_unique<ReparentCommand>(
            object, newParent,
            "Reparent " + (object.name.empty() ? object.type() : object.name));
    if (!command->valid()) return;
    commands_.execute(std::move(command));
    document_.setDirty(true);
}


// ------------------------------------------------------------------ selection

Box3 EditorApp::instanceWorldBox(const InstancedMesh& mesh, int instance) {

    Box3 box;
    if (instance < 0 || static_cast<std::size_t>(instance) >= mesh.count()) return box;
    auto geometry = mesh.geometry();
    if (!geometry) return box;
    if (!geometry->boundingBox) geometry->computeBoundingBox();
    if (!geometry->boundingBox || geometry->boundingBox->isEmpty()) return box;

    Matrix4 instanceMatrix;
    mesh.getMatrixAt(static_cast<std::size_t>(instance), instanceMatrix);

    box.copy(*geometry->boundingBox);
    box.applyMatrix4(instanceMatrix.premultiply(*mesh.matrixWorld));
    return box;
}

void EditorApp::selectObject(Object3D* object, std::optional<int> instance) {

    if (object && document_.isEditorOnly(*object)) object = nullptr;

    // A preview belongs to the inspector of the object it runs on; changing
    // the selection (including deleting the object) ends it, pose restored.
    if (animPreview_ && animPreview_->root != object) stopAnimationPreview();

    selection_.set(object);

    // Always drop the previous outlines first: the overlay group co-owns them,
    // so overwriting the pointer alone would leave the old box in the scene.
    if (selectionBox_) {
        selectionBox_->removeFromParent();
        selectionBox_.reset();
    }
    if (instanceOutline_) {
        instanceOutline_->removeFromParent();
        instanceOutline_.reset();
    }
    selectedInstance_.reset();

    if (object && object != &document_.scene()) {
        // Picking keeps working while playing — watching what a simulation does
        // to an object is half of why anyone presses Play — but the handles
        // stay parked for the session, and the outline is what says "this one".
        if (isPlaying()) {
            gizmo_->detach();
        } else {
            gizmo_->attach(*object);
        }
        // An InstancedMesh gets its picked instance boxed instead of the whole
        // cloud. Selecting one from the hierarchy (no instance to speak of)
        // falls through to the whole-object outline, which is the honest answer
        // there — nothing was pointed at.
        auto* instanced = object->as<InstancedMesh>();
        if (instanced && instance && *instance >= 0 &&
            static_cast<std::size_t>(*instance) < instanced->count()) {

            selectedInstance_ = instance;
            instanceBox_.copy(instanceWorldBox(*instanced, *instance));
            if (!instanceBox_.isEmpty()) {
                instanceOutline_ = Box3Helper::create(instanceBox_, 0xffcc44);
                overlay_->add(instanceOutline_);
            }
        } else {
            // A camera or a light bounds to nothing, and BoxHelper silently keeps
            // its degenerate initial geometry in that case — a speck at the origin.
            // Those objects carry a marker icon (and cameras a frustum) instead.
            Box3 bounds;
            bounds.setFromObject(*object);
            if (!bounds.isEmpty()) {
                selectionBox_ = BoxHelper::create(*object, 0xffcc44);
                // The outline is editor furniture: it lives under the overlay so
                // it is never saved and never picked.
                overlay_->add(selectionBox_);
            }
        }
    } else {
        gizmo_->detach();
    }
    applyGizmoMode();
    // The outline above was built this instant and is visible by default. The
    // frame loop would hide it before anything drew, but saying it here means
    // "selected while playing" is answered where the selection is made rather
    // than one call later.
    applyAuthoringVisibility();
}

void EditorApp::setJointValue(Robot& robot, std::size_t index, float radians) {

    if (index >= robot.numDOF()) return;
    robot.setJointValue(index, radians);

    // Keep the document's copy in step with what is on screen, so a save
    // captures the pose the user is looking at.
    auto config = RobotConfig::read(robot).value_or(RobotConfig{});
    config.joints = robot.jointValues();
    if (!config.urdf.empty()) config.write(robot);
}

void EditorApp::rearticulateRobots(Scene& scene) {

    // A document round trip flattens a Robot into a plain Object3D: the pose
    // survives, the joint table does not. Rebuild from the referenced URDF and
    // transplant, keeping the placeholder's identity and placement so uuid
    // lookups (selection, undo rebinding) still resolve.
    std::vector<Object3D*> placeholders;
    scene.traverse([&placeholders](Object3D& object) {
        if (auto* robot = object.as<Robot>()) {
            // Already live — ObjectLoader re-imported it from the URDF the
            // document referenced. The node transforms are right (the override
            // table restored them) but the joint table is still at the file's
            // rest pose, so the inspector would show the wrong angles and snap
            // the robot on the first drag. Re-drive it from the saved values.
            if (const auto config = RobotConfig::read(object)) {
                for (std::size_t i = 0; i < config->joints.size() && i < robot->numDOF(); ++i) {
                    robot->setJointValue(i, config->joints[i]);
                }
                robot->showColliders(config->showColliders);
            }
            return;
        }
        if (RobotConfig::read(object)) placeholders.push_back(&object);
    });

    for (auto* placeholder : placeholders) {

        auto* parent = placeholder->parent;
        if (!parent) continue;

        const auto config = *RobotConfig::read(*placeholder);

        std::shared_ptr<Robot> robot;
        std::string error;
        try {
            URDFLoader loader;
            robot = loader.load(config.urdf);
        } catch (const std::exception& e) {
            error = e.what();
        }
        if (!robot) {
            // The geometry is still in the document, so the scene renders as
            // saved — it just cannot be re-jointed until the file is back.
            log("robot not re-articulated (" + std::filesystem::path(config.urdf).filename().string() +
                (error.empty() ? ")" : "): " + error));
            continue;
        }

        // The identity/placement transplant AND the descendant-userData
        // preservation (a sensor authored on a link, not the root) live in a free
        // function so they can be tested headlessly — see transplantRobot.
        transplantRobot(*placeholder, robot, [this](const std::string& m) { log(m); });
    }
}

void EditorApp::startAnimationPreview(Object3D& root, const std::string& clipName,
                                      bool loop, float speed) {

    stopAnimationPreview();
    if (isPlaying()) return;

    auto clip = clipName.empty()
                        ? (root.animations.empty() ? nullptr : root.animations.front())
                        : AnimationClip::findByName(root.animations, clipName);
    if (!clip && !root.animations.empty()) clip = root.animations.front();
    if (!clip) return;

    animPreview_ = std::make_unique<AnimPreview>();
    animPreview_->root = &root;
    animPreview_->clip = clip->name();

    root.traverse([this](Object3D& node) {
        animPreview_->saved.push_back({&node, node.position, node.quaternion, node.scale});
        if (auto* morph = dynamic_cast<ObjectWithMorphTargetInfluences*>(&node)) {
            animPreview_->savedMorphs.emplace_back(morph, morph->morphTargetInfluences());
        }
    });

    animPreview_->mixer = std::make_unique<AnimationMixer>(root);
    auto* action = animPreview_->mixer->clipAction(clip);
    if (!action) {
        animPreview_.reset();
        return;
    }
    action->setLoop(loop ? Loop::Repeat : Loop::Once);
    action->setClampWhenFinished(true);
    action->setEffectiveTimeScale(speed);
    action->play();
}

void EditorApp::stopAnimationPreview(bool restore) {

    if (!animPreview_) return;

    if (restore) {
        animPreview_->mixer->stopAllAction();
        for (const auto& s : animPreview_->saved) {
            s.node->position.copy(s.position);
            s.node->quaternion.copy(s.quaternion);
            s.node->scale.copy(s.scale);
            s.node->updateMatrix();
        }
        for (auto& [object, values] : animPreview_->savedMorphs) {
            object->morphTargetInfluences() = values;
        }
    }
    animPreview_.reset();
}

bool EditorApp::isPreviewing(const Object3D& root) const {

    return animPreview_ && animPreview_->root == &root;
}

bool EditorApp::cameraDockRect(float& x, float& y, float& w, float& h) const {

    // Collapsing the bottom panel collapses this with it — the two share a
    // band, and a camera view the height of a tab strip is worth nothing.
    if (!bottomPanelOpen_) return false;

    const auto size = renderer_->size();
    const float s = contentScale_;

    w = inspectorPx();
    h = bottomPanelPx();
    x = static_cast<float>(size.width()) - w;
    y = static_cast<float>(size.height()) - statusHeight_ - h;

    return w >= 80.f * s && h >= 60.f * s;
}

void EditorApp::renderCameraPreview() {

    preview_.active = false;
    preview_.visible = false;

    float x = 0, y = 0, w = 0, h = 0;
    if (!cameraDockRect(x, y, w, h)) return;

    // The dock paints itself even with nothing selected, so record it before
    // the camera check.
    preview_.x = x;
    preview_.y = y;
    preview_.w = w;
    preview_.h = h;
    preview_.visible = true;

    auto* selected = selection_.get();
    auto* cam = selected ? selected->as<PerspectiveCamera>() : nullptr;
    if (!cam) return;

    const auto size = renderer_->size();
    const auto height = static_cast<float>(size.height());

    // Editor furniture (grid, gizmo, outline) must not appear in the preview,
    // and the camera keeps its own aspect outside of it.
    const bool overlayVisible = overlay_->visible;
    overlay_->visible = false;
    const float aspectBefore = cam->aspect;
    cam->aspect = w / h;
    cam->updateProjectionMatrix();

    // GL viewport origin is bottom-left.
    const int glX = static_cast<int>(x);
    const int glY = static_cast<int>(height - y - h);
    const int glW = static_cast<int>(w);
    const int glH = static_cast<int>(h);

    renderer_->setScissorTest(true);
    renderer_->setScissor(glX, glY, glW, glH);
    renderer_->setViewport(glX, glY, glW, glH);
    renderer_->render(document_.scene(), *cam);
    renderer_->setScissorTest(false);
    renderer_->setScissor(0, 0, size.width(), size.height());
    renderer_->setViewport(0, 0, size.width(), size.height());

    cam->aspect = aspectBefore;
    cam->updateProjectionMatrix();
    overlay_->visible = overlayVisible;

    preview_.label = cam->name.empty() ? std::string("Camera") : cam->name;
    preview_.active = true;
}

void EditorApp::refreshSelectionHelpers() {

    if (selectionBox_ && selection_.get()) {
        selectionBox_->update();
    }

    // Box3Helper reads instanceBox_ by reference, so re-deriving the box here is
    // what makes the outline follow an instance whose mesh (or whose instance
    // matrix) moved. Whether the result is worth drawing is
    // applyAuthoringVisibility()'s call, below.
    if (instanceOutline_ && selectedInstance_) {
        if (auto* instanced = selection_.get() ? selection_.get()->as<InstancedMesh>() : nullptr) {
            instanceBox_.copy(instanceWorldBox(*instanced, *selectedInstance_));
        } else {
            instanceBox_.makeEmpty();
        }
    }

    syncViewportMarkers();
    syncSplineOverlays();
    syncPhysicsDebug();
    syncDebugDraw();
    syncSensorOverlay();
    syncCameraHelper();

    // After the syncs, because two of them BUILD the nodes it hides: a camera
    // selected mid-play gets a fresh frustum helper, and an object that only
    // now needs a marker gets a fresh icon. Every frame rather than at Play and
    // Stop, so nothing the authoring layer grows during a session can come back
    // for a frame.
    applyAuthoringVisibility();

    // Snap is a hold-to-engage modifier, exactly like the transform example.
    const bool snap = snapEnabled_ || ImGui::GetIO().KeyShift;
    if (snap) {
        gizmo_->setTranslationSnap(0.25f);
        gizmo_->setRotationSnap(math::degToRad(15.f));
        gizmo_->setScaleSnap(0.1f);
    } else {
        gizmo_->setTranslationSnap(std::nullopt);
        gizmo_->setRotationSnap(std::nullopt);
        gizmo_->setScaleSnap(std::nullopt);
    }

    // The gizmo is a nuisance while the simulation owns the transforms, and
    // pointless with nothing selected or in Select mode. `enabled` is what the
    // mouse listeners gate on, so this is also what makes it inert; being
    // detached for the whole session (see startPlay) is the other half.
    gizmo_->enabled = gizmoActive();
    gizmo_->visible = gizmo_->enabled;
}

Object3D* EditorApp::resolveSelectable(Object3D* hit) const {

    if (!hit) return nullptr;

    auto* scene = &document_.scene();

    // Highest ancestor still below the scene root — clicking an imported model
    // selects the model, not one of its 200 sub-meshes.
    Object3D* top = hit;
    for (Object3D* o = hit; o && o->parent && o->parent != scene; o = o->parent) {
        top = o->parent;
    }

    // ...unless the user is already working inside that subtree, in which case
    // the click drills down to the exact node.
    if (auto* current = selection_.get()) {
        if (current != top && isDescendantOf(*current, *top)) return hit;
        if (current == top) return hit;
    }
    return top;
}

void EditorApp::pickAt(float mouseX, float mouseY) {

    const auto* viewport = ImGui::GetMainViewport();
    const float width = viewport->Size.x;
    const float height = viewport->Size.y;
    if (width <= 0.f || height <= 0.f) return;

    const Vector2 ndc{
            ((mouseX - viewport->Pos.x) / width) * 2.f - 1.f,
            -(((mouseY - viewport->Pos.y) / height) * 2.f - 1.f)};

    raycaster_.setFromCamera(ndc, viewCamera());
    auto intersections = raycaster_.intersectObject(document_.scene(), true);

    // Markers win over geometry regardless of depth: they are drawn on top of
    // everything, so clicking one has to select its owner even when the icon
    // floats inside a wall. Only while they are ON screen, though — the
    // raycaster does not test visibility, so without this a click during Play
    // would land on an icon nobody can see and select a light instead of the
    // wall the user was pointing at.
    if (markers_ && markers_->visible) {
        for (const auto& hit : intersections) {
            if (!hit.object) continue;
            if (auto* owner = markerOwnerOf(hit.object)) {
                selectObject(owner);
                return;
            }
        }
    }

    for (const auto& hit : intersections) {
        if (!hit.object) continue;
        if (document_.isEditorOnly(*hit.object)) continue;
        if (!hit.object->visible) continue;
        auto* selectable = resolveSelectable(hit.object);
        // The instance index only means anything when the thing being selected
        // is the InstancedMesh that was hit. resolveSelectable can walk UP to an
        // import root, and instance 7 of a child says nothing about the root.
        std::optional<int> instance;
        if (selectable == hit.object && hit.object->is<InstancedMesh>()) instance = hit.instanceId;
        selectObject(selectable, instance);
        return;
    }

    selectObject(nullptr);
}


// ----------------------------------------------------------- viewport camera

Camera& EditorApp::viewCamera() {

    return orthographic_ ? static_cast<Camera&>(ortho_) : static_cast<Camera&>(camera_);
}

const char* EditorApp::viewPresetLabel(ViewPreset preset) {

    switch (preset) {
        case ViewPreset::Front: return "Front";
        case ViewPreset::Back: return "Back";
        case ViewPreset::Left: return "Left";
        case ViewPreset::Right: return "Right";
        case ViewPreset::Top: return "Top";
        case ViewPreset::Bottom: return "Bottom";
        default: return "User";
    }
}

Vector3 EditorApp::viewPresetDirection(ViewPreset preset) {

    // A hair of tilt on the pole views. Looking exactly down +Y with an up
    // vector of +Y is the degenerate case for both lookAt() and the orbit
    // spherical; a ten-thousandth of a radian costs a tenth of a pixel and
    // removes the whole class of problem.
    constexpr float eps = 1e-4f;

    switch (preset) {
        case ViewPreset::Front: return {0, 0, 1};
        case ViewPreset::Back: return {0, 0, -1};
        case ViewPreset::Right: return {1, 0, 0};
        case ViewPreset::Left: return {-1, 0, 0};
        case ViewPreset::Top: return Vector3(0, 1, eps).normalize();
        case ViewPreset::Bottom: return Vector3(0, -1, eps).normalize();
        default: return {0, 0, 1};
    }
}

float EditorApp::sceneClearDistance(const Vector3& from) const {

    // Everything the document holds, without the overlay: the grid is 40 units
    // across and would otherwise set the distance for every scene.
    Box3 bounds;
    for (const auto& child : document_.scene().children) {
        if (!child || child == overlay_.get()) continue;
        if (document_.isEditorOnly(*child)) continue;
        Box3 childBounds;
        childBounds.setFromObject(*child);
        if (!childBounds.isEmpty()) bounds.union_(childBounds);
    }
    if (bounds.isEmpty()) return 0.f;

    // Half-diagonal, so the answer holds whichever way the camera is pointing.
    const float radius = bounds.getSize().length() * 0.5f;
    return radius + from.distanceTo(bounds.getCenter()) + 1.f;
}

void EditorApp::setOrthoHeight(float height) {

    const float halfHeight = std::max(height, 1e-3f) * 0.5f;
    const float aspect = std::max(canvas_.aspect(), 1e-3f);

    ortho_.top = halfHeight;
    ortho_.bottom = -halfHeight;
    ortho_.right = halfHeight * aspect;
    ortho_.left = -halfHeight * aspect;
    ortho_.zoom = 1.f;
    ortho_.updateProjectionMatrix();
}

void EditorApp::setOrthographic(bool ortho) {

    if (ortho == orthographic_) return;

    const Vector3 target = orbit_->target;
    const float halfFovTan = std::tan(math::degToRad(camera_.fov) * 0.5f);

    if (ortho) {
        // Same framing, different projection: the frustum is sized to what the
        // perspective camera covers at the orbit distance.
        const float distance = std::max(camera_.position.distanceTo(target), 0.01f);
        ortho_.up.copy(camera_.up);
        ortho_.position.copy(camera_.position);
        ortho_.quaternion.copy(camera_.quaternion);
        setOrthoHeight(2.f * distance * halfFovTan);

        // Ortho framing does not depend on how far back the camera stands, but
        // clipping does — and an axis view is usually asked for precisely to
        // see past whatever is between here and the far side of the scene.
        const float clear = sceneClearDistance(target);
        if (clear > distance) {
            Vector3 direction;
            direction.subVectors(ortho_.position, target);
            if (direction.length() < 1e-4f) direction.set(0, 0, 1);
            ortho_.position.copy(target).addScaledVector(direction.normalize(), clear);
        }
        ortho_.lookAt(target);
    } else {
        // And back: the distance that reproduces the ortho frustum height under
        // the perspective fov.
        const float height = (ortho_.top - ortho_.bottom) / std::max(ortho_.zoom, 1e-4f);
        const float distance = std::max(height * 0.5f / halfFovTan, 0.01f);
        Vector3 direction;
        direction.subVectors(ortho_.position, target);
        if (direction.length() < 1e-4f) direction.set(0, 0, 1);
        camera_.up.copy(ortho_.up);
        camera_.position.copy(target).addScaledVector(direction.normalize(), distance);
        camera_.lookAt(target);
    }

    orthographic_ = ortho;
    bindViewportControls();
    updateGridPlacement();
}

void EditorApp::setViewPreset(ViewPreset preset) {

    viewPreset_ = preset;

    if (preset != ViewPreset::User) {

        Camera& camera = viewCamera();
        const Vector3 target = orbit_->target;

        // Orbiting keeps the distance it had — the view turns, it does not
        // dolly. In ortho the distance only decides what is clipped, so it is
        // pushed clear of the scene instead.
        float distance = std::max(camera.position.distanceTo(target), 0.01f);
        if (orthographic_) distance = std::max(distance, sceneClearDistance(target));

        camera.position.copy(target).addScaledVector(viewPresetDirection(preset), distance);
        camera.lookAt(target);
    }

    updateGridPlacement();
}

void EditorApp::updateViewPreset() {

    if (viewPreset_ == ViewPreset::User) return;

    // The label describes where the camera is standing, so orbiting away from
    // the axis retires it. Pan and zoom keep the direction and so keep the
    // label. The threshold is a degree — wide enough that damping settling out
    // of a just-applied preset does not trip it.
    Vector3 direction;
    direction.subVectors(viewCamera().position, orbit_->target);
    if (direction.length() < 1e-6f) return;

    if (direction.normalize().dot(viewPresetDirection(viewPreset_)) < 0.99985f) {
        viewPreset_ = ViewPreset::User;
        updateGridPlacement();
    }
}

void EditorApp::updateGridPlacement() {

    if (!grid_) return;

    // GridHelper lies in XZ. In an axis ortho view that plane is edge-on — a
    // single line — so the grid is stood up into the plane being looked at.
    // Only in ortho: a perspective axis view still reads as a 3D view, and a
    // ground plane is the right reference there.
    Vector3 normal(0, 1, 0);
    if (orthographic_) {
        switch (viewPreset_) {
            case ViewPreset::Front:
            case ViewPreset::Back:
                normal.set(0, 0, 1);
                break;
            case ViewPreset::Left:
            case ViewPreset::Right:
                normal.set(1, 0, 0);
                break;
            default:
                break;
        }
    }

    if (normal.z != 0.f) {
        grid_->rotation.set(math::PI * 0.5f, 0, 0);
    } else if (normal.x != 0.f) {
        grid_->rotation.set(0, 0, math::PI * 0.5f);
    } else {
        grid_->rotation.set(0, 0, 0);
    }

    // A scene's ground plane usually sits exactly on the grid, and which of two
    // coplanar surfaces survives is decided by the depth buffer rather than by
    // draw order — no render order puts the grid back on top, and turning the
    // depth test off would float it over objects that really are in front.
    // Nudging it towards the viewer does both jobs. The nudge grows with the
    // square of the distance, which is how depth precision degrades, and is
    // capped at a fraction of that distance so it stays invisible.
    const Vector3& eye = viewCamera().position;
    const float distance = std::max(eye.length(), 0.01f);// the grid sits at the origin
    const float lift = std::clamp(distance * distance * 2e-5f, 2e-3f, distance * 0.02f);

    grid_->position.copy(normal).multiplyScalar(eye.dot(normal) < 0.f ? -lift : lift);
    // The origin axes lie in the same plane and lose to the same ground.
    if (axes_) axes_->position.copy(grid_->position);
}

void EditorApp::bindViewportControls() {

    Camera& camera = viewCamera();

    // OrbitControls and TransformControls each capture a Camera& for their
    // whole life, so a projection switch means new ones. It happens on a
    // keypress, and keeping exactly one of each alive matters: both subscribe
    // to the canvas, and two live sets would both act on every drag.
    // What the view is looking at is a property of the session, not of the
    // controls, so it is carried across. On the first call there is nothing to
    // carry and the constructor sets it.
    Vector3 target;
    if (orbit_) target.copy(orbit_->target);

    orbit_ = std::make_unique<OrbitControls>(camera, canvas_);
    orbit_->target.copy(target);
    orbit_->enableDamping = true;
    orbit_->enabled = !gizmoDragging_;
    // An ortho dolly scales the frustum rather than moving the camera, and an
    // unbounded zoom either inverts the frustum or divides it away.
    orbit_->minZoom = 1e-3f;
    orbit_->maxZoom = 1e4f;

    if (gizmo_) {
        gizmo_->removeEventListener("dragging-changed", *gizmoDragListener_);
        gizmo_->detach();
        gizmo_->removeFromParent();
    }
    gizmo_ = std::make_unique<TransformControls>(camera, canvas_);
    gizmo_->setSize(0.9f);
    overlay_->addRef(*gizmo_);
    gizmo_->addEventListener("dragging-changed", *gizmoDragListener_);
    // Snap and enabled/visible are re-applied every frame by
    // refreshSelectionHelpers(); mode, space and the attachment are not. A
    // projection switch while playing rebuilds a gizmo that has to arrive
    // parked like the one it replaces — Stop is what puts it back.
    if (auto* selected = selection_.get(); selected && !isPlaying()) gizmo_->attach(*selected);
    applyGizmoMode();
}

void EditorApp::setFollowSelection(bool follow) {

    if (followSelection_ == follow) return;

    followSelection_ = follow;
    // Whichever way it went, the next chase starts from where the camera stands:
    // forgetting the heading is what makes the first frame snap instead of swing
    // (see updateFollow).
    followHeadingFor_.clear();
    if (!follow) {
        log("follow selection off");
        return;
    }
    if (auto* selected = selection_.get()) {
        log("following " + (selected->name.empty() ? selected->type() : selected->name));
    } else {
        // Armed rather than refused: selecting something is what starts it, and
        // that is one click away.
        log("follow selection on - select something to chase");
    }
}

void EditorApp::updateFollow(float dt) {

    if (!followSelection_) return;

    // Nothing selected is a PAUSE, not an off: the target stays where the
    // chase left it and reselecting picks the chase back up.
    auto* selected = selection_.get();
    if (!selected) return;

    Vector3 world;
    selected->getWorldPosition(world);

    // Exponential approach with a ~90 ms time constant, framed as
    // 1 - exp(-dt/tau) so the rate is the same at 30 fps and at 300 rather than
    // a per-frame fraction that makes the camera lag on a slow machine. Short
    // enough to read as attached to the subject, long enough that the IMU-level
    // tremble of a hovering physics body does not reach the picture: a hard
    // lock on a rigid body is not a chase cam, it is a vibration.
    constexpr float kTau = 0.09f;
    // The heading gets its own, and a slower one: position error is a metre the
    // eye reads as lag, heading error is the whole frame swinging, and a swing
    // that tracks every twitch is worse than one that arrives late.
    //
    // 150 ms, and the number was measured rather than guessed. The Hover Arena
    // drone was traced hands-off and at full yaw stick: hovering, its heading
    // WANDERS — a degree or two either way at about 1.3 deg/s — and that is a
    // slow enough signal that no exponential filter takes it out (the camera's
    // yaw jitter only falls from 1.33 to 1.21 degrees rms going from no filter
    // at all to 180 ms), so buying wobble rejection with lag buys nothing. At
    // full stick it yaws at 200 deg/s, and the steady lag is exactly tau times
    // that: 30 degrees here, 45 at 220 ms. Looked at side by side mid-turn, 100
    // ms is a rigid lock that hides the turn entirely, 220 ms shows the drone's
    // flank, and this one keeps it astern while still reading as cornering.
    constexpr float kHeadingTau = 0.15f;
    // A hitch (a shader compile, a breakpoint) must not fling the camera; the
    // clamp costs nothing and bounds the step to one settled frame.
    const float step = std::clamp(dt, 0.f, 0.1f);

    // --- which way the subject is facing ------------------------------------
    // YAW ONLY, and never pitch or roll. A hover drone banks to move, constantly;
    // a camera that inherited attitude would roll the horizon with every input
    // and make the picture unwatchable. Camera up is left alone (OrbitControls
    // aims it at the target with the world up it has always used), so "level"
    // needs no correcting — it is never disturbed.
    //
    // A parallel projection is the one exception to the rotation: an axis view IS
    // a direction of view, and rotating it means it is no longer the view that
    // was asked for. Ortho follows by translation, exactly as it did before.
    // The editor is Y-up throughout: the grid lies in XZ and the axis presets
    // name ±Y as the poles. The heading is measured about this and the offset is
    // rotated about it, so the two cannot disagree.
    const Vector3 worldUp(0.f, 1.f, 0.f);
    const bool rotate = !orthographic_;
    float placedWith = followHeading_;
    if (!rotate) {
        // The camera is not standing at any heading while a parallel projection
        // is on, so the one on record is not the one to un-rotate by later:
        // forget it, and let a switch back to perspective snap to whatever the
        // subject is doing then.
        followHeadingFor_.clear();
    } else {
        float measured = 0.f;
        const bool valid = headingOf(*selected, measured);

        // A different subject snaps. Reading the offset back through the NEW
        // heading (below) leaves the camera precisely where it stands, so
        // selecting something that happens to face east is not a 90-degree whip.
        if (const auto uuid = selection_.uuid(); uuid != followHeadingFor_) {
            followHeadingFor_ = uuid;
            followHeading_ = valid ? measured : 0.f;
        }
        placedWith = followHeading_;

        if (valid) {
            // Shortest way round: std::remainder folds the difference into
            // [-pi, pi], so a subject crossing the seam behind it turns the near
            // way rather than unwinding the long way about.
            const float error = std::remainder(measured - followHeading_, math::TWO_PI);
            followHeading_ += error * (1.f - std::exp(-step / kHeadingTau));
        }
    }

    // --- where the camera stands, in the subject's frame --------------------
    // Read back out of the camera every frame, through the heading it was PLACED
    // with. That is what makes the user's orbiting, panning and zooming compose
    // in the subject's frame: a drag that puts the camera over the drone's left
    // shoulder is stored as "over its left shoulder" and stays there through
    // every turn, and a zoom is still just a shorter offset.
    auto& camera = viewCamera();
    Vector3 offset;
    offset.subVectors(camera.position, orbit_->target);
    if (rotate) offset.applyAxisAngle(worldUp, -placedWith);

    // The target chases the subject; the camera is then placed relative to where
    // the target ended up. With a subject that never turns (every object in an
    // edit-mode scene, and the heading frame is the identity) this is exactly
    // the old rigid translation of target and camera by the same delta.
    Vector3 delta;
    delta.subVectors(world, orbit_->target).multiplyScalar(1.f - std::exp(-step / kTau));
    orbit_->target.add(delta);

    if (rotate) offset.applyAxisAngle(worldUp, followHeading_);
    camera.position.copy(orbit_->target).add(offset);
}

float EditorApp::viewportWorldPerPixel(const Vector3& world) const {

    const auto height = static_cast<float>(renderer_->size().height());

    if (orthographic_) {
        // Parallel projection: the same everywhere in the frame.
        return (ortho_.top - ortho_.bottom) / std::max(ortho_.zoom, 1e-4f) / std::max(1.f, height);
    }
    const float distance = camera_.position.distanceTo(world);
    return 2.f * distance * std::tan(math::degToRad(camera_.fov) * 0.5f) / std::max(1.f, height);
}


// ----------------------------------------------------------------- play mode

bool EditorApp::isPlaying() const {

    return play_.state() != PlayController::State::Stopped;
}

bool EditorApp::rejectWhilePlaying(const char* what) {

    if (!isPlaying()) return false;

    // Once per attempt, not once per frame: every caller is an action, not a
    // poll. Greying the menu item is the hint; this is for the paths that have
    // no affordance to grey (shortcuts, file drops, the dialogs already open).
    log(std::string(what) + " is not available while playing - press Stop first");
    return true;
}

void EditorApp::startPlay() {

    if (isPlaying()) {
        play_.resume();
        return;
    }

    // The play snapshot must capture the authored pose, not the preview's.
    stopAnimationPreview();

    // Per-session, so a scene whose script does not read the keyboard keeps its shortcuts.
    scriptsPolledKeys_ = false;

    std::string error;
    if (!play_.play(document_, &error)) {
        log("play failed: " + error);
        return;
    }

    // Park the gizmo for the session — after the session is known to have
    // started, so a refused Play leaves the editor exactly as it was. Hiding it
    // would not be enough: attached, it keeps riding a body the simulation is
    // moving and keeps offering handles that the edit gate would only refuse.
    // It comes back on Stop through the scene-replaced listener, which
    // re-resolves the selection by uuid and re-attaches on the way; the mode,
    // the space and the snap are the editor's own state and were never touched.
    gizmo_->detach();
    applyGizmoMode();

    log("play started");
}

void EditorApp::togglePause() {

    if (!isPlaying()) return;
    play_.togglePause();
    log(play_.paused() ? "paused" : "resumed");
}

void EditorApp::stopPlay() {

    if (!isPlaying()) return;

    // Before the sessions go: the lines describe a PhysX world that stop() is
    // about to destroy, and the cloud describes sensors it is about to drop.
    clearPhysicsDebug();
    clearSensorOverlay();

    std::string error;
    if (!play_.stop(document_, &error)) {
        log("stop failed: " + error);
        return;
    }
    log("play stopped, scene restored");

    // A save that arrived while playing was parked rather than committed; the
    // scene it belongs to exists again now, so take the poll off its accumulator
    // and let the next frame apply it.
    if (externalEdit_.active) externalEdit_.poll = ExternalEditState::pollSeconds;
}


// ------------------------------------------------------------------- plumbing

bool EditorApp::authoringVisible() const {

    return !hideAuthoring_ && !isPlaying();
}

void EditorApp::applyAuthoringVisibility() {

    const bool visible = authoringVisible();

    if (selectionBox_) selectionBox_->visible = visible;
    // An instance whose index went out from under it (the count shrank) leaves
    // an empty box, and Box3Helper::updateMatrixWorld early-returns on empty —
    // hide it rather than stranding the last shape it drew.
    if (instanceOutline_) instanceOutline_->visible = visible && !instanceBox_.isEmpty();
    // The whole group: one flag hides every icon, including the ones a frame of
    // play is still busy placing.
    if (markers_) markers_->visible = visible;
    if (cameraHelper_) cameraHelper_->visible = visible;
}

bool EditorApp::gizmoActive() const {

    return authoringVisible() && gizmoMode_ != "select" && !selection_.empty();
}

void EditorApp::applyGizmoMode() {

    // "select" is not a TransformControls mode — it is the absence of one. The
    // gizmo keeps its last real mode so switching back does not surprise the
    // user, and is simply hidden and disabled.
    const bool selectOnly = gizmoMode_ == "select";
    if (!selectOnly) gizmo_->setMode(gizmoMode_.empty() ? "translate" : gizmoMode_);
    gizmo_->setSpace(gizmoWorldSpace_ ? "world" : "local");
    // Through the same predicate the frame loop uses: this runs from the
    // toolbar buttons and the W/E/R/Q shortcuts, and a mode change while
    // playing records the mode without putting the parked gizmo back on screen.
    gizmo_->visible = gizmoActive();
}

void EditorApp::handleShortcuts() {

    const ImGuiIO& io = ImGui::GetIO();
    // Panels keep keyboard focus after a click (selecting in the hierarchy
    // does it), and WantCaptureKeyboard stays true the whole time — gating on
    // it made every shortcut dead until the viewport was clicked again. Only
    // real text entry and open popups may swallow the keys.
    if (io.WantTextInput) return;
    if (ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) return;
    if (fileBrowser_.isOpen()) return;

    const bool ctrl = io.KeyCtrl;
    const bool shift = io.KeyShift;
    const bool alt = io.KeyAlt;

    // Follow Selection, ahead of the yield below on purpose. Chasing the thing
    // you are flying is a VIEW command, and the session it is wanted in most is
    // exactly the one that hands the plain keys to a script — a chase cam you
    // can only switch on before you start flying is not much of a chase cam.
    // Shift+F rather than a key of its own, so no teleop binding is spent on it
    // and it reads as "Frame selection, but keep doing it".
    if (shift && !ctrl && !alt && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
        setFollowSelection(!followSelection_);
    }

    // A playing script that reads the keyboard owns the PLAIN keys. Teleop reaches for
    // W/A/S/D, Q/E and the numpad; those are the gizmo modes and the axis viewpoints, so
    // without this, driving a robot forward also switched the gizmo to translate and snapped
    // the camera to a front view. Modified commands stay live — Ctrl+S, Ctrl+Z and the
    // Alt+digit viewpoints are not what a script polls, and silently losing save-while-playing
    // would be its own surprise.
    if (scriptsPolledKeys_ && isPlaying() && !ctrl && !alt && !io.KeySuper) return;

    // --- viewpoints -------------------------------------------------------
    // The numpad bindings every 3D editor shares, with Ctrl for the opposite
    // side. Alt+digit does the same on the keyboards that have no numpad.
    const auto viewKey = [&](ImGuiKey keypad, ImGuiKey digit, ViewPreset side, ViewPreset opposite) {
        if (!(ImGui::IsKeyPressed(keypad, false) || (alt && ImGui::IsKeyPressed(digit, false)))) return;
        // An axis view is an orthographic view — that is what it is for. The
        // projection stays put afterwards; Numpad5 is the way back.
        setOrthographic(true);
        setViewPreset(ctrl ? opposite : side);
    };
    viewKey(ImGuiKey_Keypad1, ImGuiKey_1, ViewPreset::Front, ViewPreset::Back);
    viewKey(ImGuiKey_Keypad3, ImGuiKey_3, ViewPreset::Right, ViewPreset::Left);
    viewKey(ImGuiKey_Keypad7, ImGuiKey_7, ViewPreset::Top, ViewPreset::Bottom);
    if (ImGui::IsKeyPressed(ImGuiKey_Keypad5, false) || (alt && ImGui::IsKeyPressed(ImGuiKey_5, false))) {
        setOrthographic(!orthographic_);
    }

    // New and Open put up a "discard changes?" modal before they mutate
    // anything, so they are refused here rather than at the far end — asking
    // the question and then declining to act on the answer is worse than not
    // asking. The rest fall through to the gate in the operation itself.
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_N, false)) {
        if (!rejectWhilePlaying("New Scene")) pendingAction_ = PendingAction::New;
    }
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
        if (!rejectWhilePlaying("Open Scene")) pendingAction_ = PendingAction::Open;
    }
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_S, false)) saveScene();
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_D, false)) duplicateSelected();

    if (ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_Z, false)) undo();
    if (ctrl && (ImGui::IsKeyPressed(ImGuiKey_Y, false) ||
                 (shift && ImGui::IsKeyPressed(ImGuiKey_Z, false)))) redo();

    if (ctrl) return;// the remaining shortcuts are unmodified single keys

    if (ImGui::IsKeyPressed(ImGuiKey_W, false)) {
        gizmoMode_ = "translate";
        applyGizmoMode();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_E, false)) {
        gizmoMode_ = "rotate";
        applyGizmoMode();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
        gizmoMode_ = "scale";
        applyGizmoMode();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Q, false)) {
        gizmoWorldSpace_ = !gizmoWorldSpace_;
        applyGizmoMode();
    }
    // Shift+F is Follow Selection, handled above; one press must not also frame.
    if (!shift && ImGui::IsKeyPressed(ImGuiKey_F, false)) focusSelected();
    if (ImGui::IsKeyPressed(ImGuiKey_Delete, false)) deleteSelected();
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) selectObject(nullptr);
}

void EditorApp::handleFileDrop(const std::vector<std::string>& paths) {

    // One message for the whole drop rather than one per file, and it stops a
    // scene drop from raising the discard-changes modal.
    if (!paths.empty() && rejectWhilePlaying("Dropping files")) return;

    for (const auto& entry : paths) {
        std::filesystem::path path(entry);
        const auto extension = formats::extensionOf(path);

        if (extension == ".json") {
            if (document_.dirty()) {
                pendingAction_ = PendingAction::OpenPath;
                pendingPath_ = path;
            } else {
                openScene(path);
            }
        } else if (extension == ".hdr") {
            setEnvironment(path, true);
        } else if (formats::contains(formats::importable(), extension)) {
            importModel(path);
        } else if (formats::contains(formats::scripts(), extension)) {
            // Same rule as an image drop: it lands on whatever is selected.
            if (auto* selected = selection_.get()) {
                assignScript(*selected, path);
            } else {
                log("no selection to attach " + path.filename().string() + " to");
            }
        } else if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
                   extension == ".bmp" || extension == ".tga" || extension == ".gif") {
            // Deferred: which slot this belongs in depends on where the cursor
            // is over the inspector, and those rows are only known once the UI
            // has drawn. resolveTextureDrops() picks it up at the end of the
            // frame.
            pendingTextureDrops_.push_back(path);
        } else {
            log("ignored dropped file " + path.filename().string());
        }
    }
}

void EditorApp::log(const std::string& message) {

    console_.push_back(message);
    while (console_.size() > kConsoleLimit) console_.pop_front();
    consoleScrollToBottom_ = true;
}

void EditorApp::logWarnings() {

    for (const auto& warning : document_.warnings()) log("warning: " + warning);
}

void EditorApp::beginEditIfActivated() {

    if (ImGui::IsItemActivated()) commands_.beginTransaction();
}

void EditorApp::endEditIfDeactivated() {

    if (ImGui::IsItemDeactivated()) commands_.endTransaction();
}

void EditorApp::loadSettings() {

    settingsPath_ = EditorSettings::defaultPath();
    settings_.load(settingsPath_);
    bottomPanelOpen_ = settings_.bottomPanelOpen;
    document_.setImageStorage(settings_.imageStorage);
    document_.setModelStorage(settings_.modelStorage);
}

void EditorApp::persistSettings() {

    settings_.bottomPanelOpen = bottomPanelOpen_;
    settings_.save(settingsPath_);
}
