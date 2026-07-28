
#include "EditorApp.hpp"

#include "EditorTheme.hpp"
#include "ImportFormats.hpp"
#include "PanelLayout.hpp"

#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/AnimationPlaySession.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/ScriptWorkspace.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"

#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#endif

#include "threepp/canvas/Monitor.hpp"
#include "threepp/core/Clock.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/helpers/AxesHelper.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/helpers/GridHelper.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
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
#include "threepp/objects/Robot.hpp"
#include "threepp/renderers/RendererFactory.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>

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

}// namespace


EditorApp::EditorApp(const Options& options)
    : options_(options),
      canvas_(Canvas::Parameters()
                      .title("threepp editor")
                      .size(kDefaultWidth, kDefaultHeight)
                      .antialiasing(4)
                      .exitOnKeyEscape(false)),
      renderer_(createRenderer(canvas_, requestedApi(options.vulkan))),
      camera_(55.f, canvas_.aspect(), 0.05f, 5000.f),
      orbit_(camera_, canvas_) {

    contentScale_ = monitor::contentScale().first;

    renderer_->shadowMap().enabled = true;
    renderer_->shadowMap().type = ShadowMap::PFC;
    renderer_->toneMapping = ToneMapping::ACESFilmic;
    renderer_->toneMappingExposure = 1.0f;

    camera_.position.set(6, 5, 8);
    orbit_.target.set(0, 0.5f, 0);
    orbit_.enableDamping = true;

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

    gizmo_ = std::make_unique<TransformControls>(camera_, canvas_);
    gizmo_->setSize(0.9f);
    overlay_->addRef(*gizmo_);

    // Orbiting while dragging a handle fights the gizmo; and a drag is exactly
    // the span an undo entry should cover.
    gizmoDragListener_ = std::make_unique<LambdaEventListener>([this](Event& event) {
        const bool dragging = std::any_cast<bool>(event.target);
        orbit_.enabled = !dragging;
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
    gizmo_->addEventListener("dragging-changed", *gizmoDragListener_);

    // --- ImGui --------------------------------------------------------------
    ui_ = std::make_unique<ImguiFunctionalContext>(canvas_, *renderer_, [this] { drawUi(); });

    ioCapture_.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture_.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture_.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas_.setIOCapture(&ioCapture_);

    canvas_.onWindowResize([this](WindowSize size) {
        camera_.aspect = size.aspect();
        camera_.updateProjectionMatrix();
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
        // The overlay group survives the scene swap, so the outline must be
        // detached from it, not just dropped.
        if (selectionBox_) {
            selectionBox_->removeFromParent();
            selectionBox_.reset();
        }

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
    play_.addSession(physics_);
#endif
    play_.addSession(std::make_shared<AnimationPlaySession>());
#ifdef THREEPP_EDITOR_WITH_PYTHON
    // Last, so a script's transform edits are the final word for the frame —
    // physics and the animation player have already had their say.
    scripts_ = std::make_shared<ScriptPlaySession>();
    scripts_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(scripts_);
#endif

    loadSettings();

    assetDir_ = settings_.sceneDir.empty() ? std::filesystem::current_path()
                                           : std::filesystem::path(settings_.sceneDir);

    if (!options_.openOnStart.empty() && options_.openOnStart.extension() == ".json") {
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
    log("built without PhysX - Play runs with no physics session");
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

int EditorApp::runSelfTest() {

    Clock clock;
    int failed = 0;

    const auto step = [&](int frames = 1) {
        for (int i = 0; i < frames; ++i) {
            canvas_.animateOnce([&] { frame(clock.getDelta()); });
        }
    };
    const auto check = [&](bool ok, const char* what) {
        std::cout << (ok ? "[selftest] PASS " : "[selftest] FAIL ") << what << std::endl;
        if (!ok) ++failed;
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
                child == static_cast<Object3D*>(physicsDebugLines_.get()) ||
                child == static_cast<Object3D*>(cameraHelper_.get()) ||
                child == static_cast<Object3D*>(gizmo_.get())) continue;
            ++n;
        }
        return n;
    };

    step(2);

    auto* box = document_.scene().getObjectByName("Box");
    auto* ground = document_.scene().getObjectByName("Ground");
    check(box != nullptr, "template Box exists");
    check(ground != nullptr, "template Ground exists");
    if (!box || !ground) return 1;

    // Outline-leak regression: re-selecting must not accumulate helpers.
    selectObject(box);
    step();
    selectObject(ground);
    step();
    selectObject(box);
    step();
    check(selection_.get() == box, "selection is Box");
    check(outlineCount() == 1, "one outline after repeated re-select");

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

    // Viewport markers and the camera frustum. The marker count is also the
    // check that the embedded SVG parses — a failure there is silent by
    // design (the object stays selectable from the hierarchy).
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

    selectObject(cameraRaw);
    step();
    deleteSelected();
    step();
    check(viewportMarkers_.size() == lightMarkers, "deleting the camera drops its marker");
    check(cameraHelper_ == nullptr, "deleting the camera drops the frustum helper");
    commands_.undo();// leave the scene as we found it
    step();

    // Every light kind carries its own icon, and each is a separate SVG. One
    // marker per added light is what proves all of them parse — a broken path
    // set would silently produce no marker for just that kind.
    {
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
    {
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
            step(30);

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
        // Script Editor and stored in the scene. Through the window's own apply
        // path, which is what the user's Ctrl+Enter runs.
        if (auto* box = document_.scene().getObjectByName("Box")) {

            const auto undosBefore = commands_.undoCount();

            openScriptEditor(*box);
            check(scriptEditor_.open, "the Script Editor opens on the object");
            // Indented with TABS on purpose: Apply has to normalize them, or
            // this source is a TabError waiting for the first space-indented
            // line somebody adds later.
            scriptEditor_.buffer = "class Inline:\n"
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
            check(scriptEditor_.status.empty(), "valid source passes the syntax check");
            check(commands_.undoCount() == undosBefore + 1, "Apply is one undo entry");

            commands_.undo();
            step();
            check(!ScriptConfig::read(*box).has_value(), "undo takes the inline script back off");
            check(!scriptEditor_.open, "the window closes when its script is undone away");
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
            step(30);
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
                scriptEditor_.buffer = "class Broken:\n    def update(self dt):\n        pass\n";
                applyScriptEditor();
                step();

                check(!scriptEditor_.status.empty(), "Apply reports the syntax error");
                check(scriptEditor_.status.find("line 2") != std::string::npos,
                      "the syntax error carries its line number");
                check(ScriptConfig::read(*restored).value_or(ScriptConfig{}).source ==
                              scriptEditor_.buffer,
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
                    step(30);
                    auto* driven = document_.scene().getObjectByName("Box");
                    check(driven && driven->rotation.y > beforeTemplate + 1e-3f,
                          "the template script runs as generated (import threepp resolves)");
                    stopPlay();
                    step();
                }

                // start(self, obj, scene): the scene arrives only when the
                // signature asks for it, and it is a real handle — this script
                // ignores its own object and drives Ground through a lookup.
                if (auto* target = document_.scene().getObjectByName("Box")) {
                    setInlineScript(*target,
                                    "class Reacher:\n"
                                    "    def start(self, obj, scene):\n"
                                    "        self.other = scene.get_object_by_name('Ground')\n"
                                    "    def update(self, dt):\n"
                                    "        self.other.position.y += dt\n",
                                    "Scene-Reaching Script");
                    step();
                    const float groundBefore =
                            document_.scene().getObjectByName("Ground")->position.y;
                    startPlay();
                    step(30);
                    auto* ground = document_.scene().getObjectByName("Ground");
                    check(ground && ground->position.y > groundBefore + 1e-3f,
                          "a two-argument start receives the scene and reaches other objects");
                    stopPlay();
                    step();
                    auto* groundRestored = document_.scene().getObjectByName("Ground");
                    check(groundRestored &&
                                  std::abs(groundRestored->position.y - groundBefore) < 1e-4f,
                          "stop restores what a scene-reaching script moved");
                }

                if (auto* clear = document_.scene().getObjectByName("Box")) {
                    assignScript(*clear, {});
                    step();
                    check(!ScriptConfig::read(*clear).has_value(), "clearing removes the inline script");
                    check(!scriptEditor_.open, "clearing the script closes the Script Editor");
                }
            }
        }
    }
#endif

    // "Edit in VS Code", end to end and without VS Code: the export, the
    // workspace, the poll, the sync back through the Script Editor's Apply, and
    // every way a session ends. The launch itself is the one thing gated out —
    // a test run must not open windows on the machine running it.
    //
    // Outside the Python guard on purpose: none of this needs an interpreter.
    // The compile check is the only part that does, and it is not what a file
    // watcher is for.
    {
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
            check(scriptEditor_.open, "the Script Editor comes along to show the session");

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
    {
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

            // Tube to road: same node, different geometry, and the width the
            // config asks for is the width the mesh has.
            if (mesh) {
                const auto geometry = mesh->geometry();
                config.mesh = SplineConfig::MeshKind::Road;
                config.width = 7.f;
                setConfig(config, "Spline Mesh");
                step();
                spline = splineNow(splineUuid);

                check(spline && SplineConfig::derivedMesh(*spline) == mesh,
                      "switching tube to road keeps the node");
                check(mesh->geometry() != geometry, "and rebuilds its geometry");

                // The road emits the two vertices of a cross-section back to
                // back, so a pair measures the width there. It NEVER narrows:
                // a straight run and any bend the width fits through measure
                // the authored width exactly, and a bend tighter than the
                // half-width reaches from the outer edge to the centre of the
                // bend — the pie sector, still over half the width. The ribbon
                // this replaced pinned those rings down to a pivot.
                const auto* position = mesh->geometry()->getAttribute<float>("position");
                float widest = 0.f;
                float narrowest = std::numeric_limits<float>::max();
                for (int i = 0; position && i + 1 < position->count(); i += 2) {
                    const Vector3 left(position->getX(i), position->getY(i), position->getZ(i));
                    const Vector3 right(position->getX(i + 1), position->getY(i + 1), position->getZ(i + 1));
                    const float across = left.distanceTo(right);
                    widest = std::max(widest, across);
                    narrowest = std::min(narrowest, across);
                }
                check(std::abs(widest - 7.f) < 1e-3f,
                      "with the road exactly as wide as the config says");
                check(narrowest >= 3.5f - 1e-3f && narrowest <= 7.f + 1e-3f,
                      "and never pinched narrower than the half-width it always reaches");
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
        authored.mesh = SplineConfig::MeshKind::Road;
        authored.width = 5.f;
        authored.uvLength = 3.f;
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
            bool narrowed = false;
            if (auto* asMesh = reloadedMesh ? reloadedMesh->as<Mesh>() : nullptr) {
                const auto* position = asMesh->geometry()->getAttribute<float>("position");
                if (position && position->count() >= 2) {
                    // The WIDEST cross-section is the authored width exactly —
                    // straight runs and every bend the width fits through
                    // measure it — and the narrowest still reaches the outer
                    // half, since a bend tighter than that spans from the outer
                    // edge to its own centre rather than pinching.
                    float widest = 0.f;
                    float narrowest = std::numeric_limits<float>::max();
                    for (int i = 0; i + 1 < position->count(); i += 2) {
                        const Vector3 left(position->getX(i), position->getY(i), position->getZ(i));
                        const Vector3 right(position->getX(i + 1), position->getY(i + 1), position->getZ(i + 1));
                        const float across = left.distanceTo(right);
                        widest = std::max(widest, across);
                        narrowest = std::min(narrowest, across);
                    }
                    narrowed = std::abs(widest - authored.width) < 1e-3f &&
                               narrowest >= authored.width * 0.5f - 1e-3f;
                }
            }
            check(narrowed, "rebuilt to the width the reloaded config asks for");
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
                            "    def start(self, obj, scene):\n"
                            "        self.obj = obj\n"
                            "        self.u = 0.0\n"
                            "        self.path = threepp.editor.spline_from_object(\n"
                            "            scene.get_object_by_name(self.spline_name))\n"
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

    // With a model path on the command line, exercise the async import path
    // end to end: queue -> worker -> finalize -> selected group in the scene.
    if (!options_.openOnStart.empty() && options_.openOnStart.extension() != ".json") {
        const auto childrenBefore = document_.scene().children.size();
        importModel(options_.openOnStart);
        int budget = 3000;// frames; the worker is genuinely asynchronous
        while ((activeImport_ || !importQueue_.empty()) && budget-- > 0) step();
        check(budget > 0, "import completed in time");
        check(importError_.empty(), "import reported no error");
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
    }

    // URDF: import, drive a joint, and prove the pose survives a full document
    // round trip. Play/Stop is that round trip — it serialises the scene and
    // rebuilds it — so this covers re-articulation as the user meets it.
    if (!options_.urdf.empty()) {

        importModel(options_.urdf);
        int budget = 6000;
        while ((activeImport_ || !importQueue_.empty()) && budget-- > 0) step();
        check(budget > 0, "urdf import completed in time");
        check(importError_.empty(), "urdf import reported no error");

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

            // Collision hulls: hidden on import, and the opt-in has to outlive
            // the rebuild or it would reset every time play is pressed.
            const auto colliderVisible = [](Object3D& root) {
                bool visible = false;
                root.traverse([&visible](Object3D& node) {
                    if (node.userData.contains("collider") && node.visible) visible = true;
                });
                return visible;
            };
            if (live) {
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
        }
    }

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // The physics collider overlay. It exists because a body resting on nothing
    // and a body whose collider is somewhere else look identical, so the checks
    // are about the buffer being there, being emptied, and surviving the scene
    // replace that Stop performs under it.
    {
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

        const auto colliderVertices = [this] {
            return physicsDebugLines_ ? physicsDebugLines_->geometry()->drawRange.count : 0;
        };

        physicsDebug_ = true;
        step(2);
        check(physicsDebugLines_ == nullptr, "the collider overlay stays off while stopped");

        startPlay();
        // PhysX fills the render buffer during simulate(), so the lines arrive
        // one substep after the parameter is set.
        step(8);
        check(physicsDebugLines_ != nullptr, "toggling Physics Colliders on while playing draws an overlay");
        check(physicsDebugLines_ && physicsDebugLines_->visible, "which is visible");
        check(colliderVertices() > 0, "with a non-empty line buffer");

        physicsDebug_ = false;
        step(2);
        check(physicsDebugLines_ == nullptr, "toggling it off clears the overlay");

        // Back on, then Stop: that is a full scene replace with live world
        // pointers in the overlay, which is the shape of every scar this file
        // carries.
        physicsDebug_ = true;
        step(5);
        check(colliderVertices() > 0, "toggling it back on refills the buffer");
        stopPlay();
        step(2);
        check(physicsDebugLines_ == nullptr, "stopping play clears it, scene replace and all");

        // And once more across an explicit document replace while playing.
        startPlay();
        step(5);
        check(colliderVertices() > 0, "the overlay comes back on the next play");
        stopPlay();
        newScene();
        step(2);
        check(physicsDebugLines_ == nullptr, "a new scene with the toggle on leaves nothing behind");

        // A road's collider count follows the road's SHAPE, not its
        // tessellation. This S bends twice, both tighter than half its six
        // metre width — the case that used to be cooked as one sliver hull per
        // sample of a folded ribbon, which drew a moiré of them. The pieces
        // behind it are a handful, so the overlay's line count is BOUNDED: this
        // is a regression guard on the count, not on the drawing.
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
                config.mesh = SplineConfig::MeshKind::Road;
                config.width = 6.f;
                // Sampled as finely as the editor allows, which is what makes
                // the bound below mean something: one hull per sample would be
                // 600 of them here.
                config.samples = SplineConfig::maxSamples;
                config.write(*spline);
                step();

                if (auto* road = SplineConfig::derivedMesh(*spline)) {
                    PhysicsConfig physics;
                    physics.enabled = true;
                    physics.body = PhysicsConfig::Body::Static;
                    physics.shape = PhysicsConfig::Shape::Auto;
                    physics.write(*road);
                }
            }

            startPlay();
            step(8);
            const int lines = colliderVertices();
            check(lines > 0, "an S-curve road draws collider lines of its own");
            check(lines < 4000, "and a bounded number of them, not one hull per sample");
            stopPlay();
            step(2);
        }
        physicsDebug_ = false;
    }
#endif

    std::cout << "[selftest] " << (failed == 0 ? "ALL PASS" : "FAILED") << std::endl;
    return failed == 0 ? 0 : 1;
}

void EditorApp::frame(float dt) {

    play_.update(dt);
    pollImports(dt);
    pollExternalEdit(dt);
    if (animPreview_) animPreview_->mixer->update(dt);
    orbit_.update();

    refreshSelectionHelpers();

    renderer_->render(document_.scene(), camera_);
    renderCameraPreview();
    ui_->render();

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

    drawMenuBar();
    drawToolbar();
    drawHierarchy();
    drawInspector();
    drawBottomPanel();
    drawStatusBar();
    drawPlayBanner();
    drawImportToast();
    // A floating window, so it goes over the fixed panels and under the modals.
    drawScriptEditor();

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

    selectObject(nullptr);
    commands_.clear();
    document_.newScene();
    buildTemplateScene();
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

    camera_.position.set(6, 5, 8);
    orbit_.target.set(0, 0.5f, 0);

    document_.setDirty(false);
    selectObject(box.get());
}

void EditorApp::openScene(const std::filesystem::path& path) {

    selectObject(nullptr);
    std::string error;
    if (!document_.open(path, &error)) {
        log("open failed: " + error);
        return;
    }
    commands_.clear();
    logWarnings();
    settings_.addRecentFile(path);
    settings_.sceneDir = path.parent_path().string();
    log("opened " + path.filename().string());
}

void EditorApp::saveScene() {

    if (!document_.hasPath()) {
        pendingDialog_ = PendingDialog::SaveAs;
        fileBrowser_.open("Save Scene As", FileBrowser::Mode::Save,
                          settings_.sceneDir, {".json"}, "scene.json");
        return;
    }
    saveSceneAs(document_.path());
}

void EditorApp::saveSceneAs(const std::filesystem::path& path) {

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

void EditorApp::drawSplitter(const char* id, float x, float top, float height,
                             float& width, float sign) {

    const float s = contentScale_;
    const float thickness = layout::splitterThickness * s;

    ImGui::SetNextWindowPos({x, top});
    ImGui::SetNextWindowSize({thickness, height});
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4{0.f, 0.f, 0.f, 0.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.f, 0.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);

    if (ImGui::Begin(id, nullptr, layout::barFlags)) {

        ImGui::InvisibleButton("##grip", {thickness, std::max(height, 1.f)});

        const bool active = ImGui::IsItemActive();
        if (active || ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            // Only hint at the handle once the pointer finds it; an always-on
            // divider would just be chrome.
            auto* draw = ImGui::GetWindowDrawList();
            const auto min = ImGui::GetWindowPos();
            draw->AddRectFilled({min.x + thickness * 0.5f - 1.f * s, min.y},
                                {min.x + thickness * 0.5f + 1.f * s, min.y + height},
                                ImGui::GetColorU32(active ? theme::accent() : theme::muted()));
        }
        if (active) {
            width = std::clamp(width + ImGui::GetIO().MouseDelta.x * sign / s,
                               EditorSettings::minPanelWidth, EditorSettings::maxPanelWidth);
        }
    }
    ImGui::End();

    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
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
    const float bottom = viewport->Size.y - statusHeight_ - (bottomPanelOpen_ ? layout::bottomHeight * s : 0.f);
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

    auto& scene = document_.scene();
    scene.environment = nullptr;
    scene.background = Background(Color(0x1c1f24));
    document_.setDirty(true);
    log("environment cleared");
}

void EditorApp::assignTextureToSlot(const std::filesystem::path& path) {

    auto target = textureSlotTarget_;
    textureSlotTarget_ = {};

    if (!target.material || !target.setter) {
        assignTextureToSelection(path);
        return;
    }

    TextureLoader loader;
    std::shared_ptr<Texture> texture;
    try {
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
    log(target.slot + " set from " + path.filename().string());
}

void EditorApp::assignTextureToSelection(const std::filesystem::path& path) {

    auto* selected = selection_.get();
    if (!selected) {
        log("no selection to texture");
        return;
    }
    auto material = selected->material();
    auto* withMap = material ? dynamic_cast<MaterialWithMap*>(material.get()) : nullptr;
    if (!withMap) {
        log("selected object has no base-colour map slot");
        return;
    }

    TextureLoader loader;
    std::shared_ptr<Texture> texture;
    try {
        texture = loader.load(path, ColorSpace::sRGB);
    } catch (const std::exception& e) {
        log("texture failed: " + std::string(e.what()));
        return;
    }
    if (!texture) {
        log("texture failed: " + path.filename().string());
        return;
    }

    commands_.execute(std::make_unique<SetMaterialMapCommand>(
            *material, "map",
            [withMap](const std::shared_ptr<Texture>& t) { withMap->map = t; },
            withMap->map, texture));
    document_.setDirty(true);
    settings_.textureDir = path.parent_path().string();
    log("map set from " + path.filename().string());
}

void EditorApp::assignScript(Object3D& object, const std::filesystem::path& path) {

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

    if (!object) return;
    auto command = std::make_unique<AddObjectCommand>(parent, object, label);
    auto* raw = object.get();
    commands_.execute(std::move(command));
    document_.setDirty(true);
    selectObject(raw);
    scrollTo_ = raw;
}

void EditorApp::addSplinePoint(Object3D& spline, std::size_t index, const std::string& label) {

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
        selected->getWorldPosition(orbit_.target);
        return;
    }

    Vector3 centre = box.getCenter();
    const Vector3 size = box.getSize();
    const float radius = std::max({size.x, size.y, size.z}) * 0.5f;
    // Pull back far enough that the bounding sphere fits the vertical FOV, with
    // a little air around it.
    const float distance = std::max(radius / std::tan(math::degToRad(camera_.fov) * 0.5f), 0.5f) * 1.6f;

    Vector3 direction;
    direction.subVectors(camera_.position, orbit_.target);
    if (direction.length() < 1e-4f) direction.set(1, 1, 1);
    direction.normalize();

    orbit_.target.copy(centre);
    camera_.position.copy(centre).add(direction.multiplyScalar(distance));
}

void EditorApp::reparent(Object3D& object, Object3D& newParent) {

    auto command = std::make_unique<ReparentCommand>(
            object, newParent,
            "Reparent " + (object.name.empty() ? object.type() : object.name));
    if (!command->valid()) return;
    commands_.execute(std::move(command));
    document_.setDirty(true);
}


// ------------------------------------------------------------------ selection

void EditorApp::selectObject(Object3D* object) {

    if (object && document_.isEditorOnly(*object)) object = nullptr;

    // A preview belongs to the inspector of the object it runs on; changing
    // the selection (including deleting the object) ends it, pose restored.
    if (animPreview_ && animPreview_->root != object) stopAnimationPreview();

    selection_.set(object);

    // Always drop the previous outline first: the overlay group co-owns it, so
    // overwriting the pointer alone would leave the old box in the scene.
    if (selectionBox_) {
        selectionBox_->removeFromParent();
        selectionBox_.reset();
    }

    if (object && object != &document_.scene()) {
        gizmo_->attach(*object);
        // A camera or a light bounds to nothing, and BoxHelper silently keeps
        // its degenerate initial geometry in that case — a speck at the origin.
        // Those objects carry a marker icon (and cameras a frustum) instead.
        Box3 bounds;
        bounds.setFromObject(*object);
        if (!bounds.isEmpty()) {
            selectionBox_ = BoxHelper::create(*object, 0xffcc44);
            // The outline is editor furniture: it lives under the overlay so it
            // is never saved and never picked.
            overlay_->add(selectionBox_);
        }
    } else {
        gizmo_->detach();
    }
    applyGizmoMode();
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
        if (object.as<Robot>()) return;// already live
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

        robot->name = placeholder->name;
        robot->uuid = placeholder->uuid;
        robot->position.copy(placeholder->position);
        robot->quaternion.copy(placeholder->quaternion);
        robot->scale.copy(placeholder->scale);
        robot->visible = placeholder->visible;
        robot->userData = placeholder->userData;

        for (std::size_t i = 0; i < config.joints.size() && i < robot->numDOF(); ++i) {
            robot->setJointValue(i, config.joints[i]);
        }
        robot->showColliders(config.showColliders);

        const auto index = childIndex(*parent, *placeholder);
        placeholder->removeFromParent();
        insertChildAt(*parent, robot, index);
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
    h = layout::bottomHeight * s;
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

    syncViewportMarkers();
    syncSplineOverlays();
    syncPhysicsDebug();
    syncCameraHelper();

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
    // pointless with nothing selected or in Select mode.
    gizmo_->enabled = !isPlaying() && gizmoMode_ != "select" && !selection_.empty();
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

    raycaster_.setFromCamera(ndc, camera_);
    auto intersections = raycaster_.intersectObject(document_.scene(), true);

    // Markers win over geometry regardless of depth: they are drawn on top of
    // everything, so clicking one has to select its owner even when the icon
    // floats inside a wall.
    for (const auto& hit : intersections) {
        if (!hit.object) continue;
        if (auto* owner = markerOwnerOf(hit.object)) {
            selectObject(owner);
            return;
        }
    }

    for (const auto& hit : intersections) {
        if (!hit.object) continue;
        if (document_.isEditorOnly(*hit.object)) continue;
        if (!hit.object->visible) continue;
        selectObject(resolveSelectable(hit.object));
        return;
    }

    selectObject(nullptr);
}


// ----------------------------------------------------------------- play mode

bool EditorApp::isPlaying() const {

    return play_.state() != PlayController::State::Stopped;
}

void EditorApp::startPlay() {

    if (isPlaying()) {
        play_.resume();
        return;
    }

    // The play snapshot must capture the authored pose, not the preview's.
    stopAnimationPreview();

    std::string error;
    if (!play_.play(document_, &error)) {
        log("play failed: " + error);
        return;
    }
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
    // about to destroy.
    clearPhysicsDebug();

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

void EditorApp::applyGizmoMode() {

    // "select" is not a TransformControls mode — it is the absence of one. The
    // gizmo keeps its last real mode so switching back does not surprise the
    // user, and is simply hidden and disabled.
    const bool selectOnly = gizmoMode_ == "select";
    if (!selectOnly) gizmo_->setMode(gizmoMode_.empty() ? "translate" : gizmoMode_);
    gizmo_->setSpace(gizmoWorldSpace_ ? "world" : "local");
    gizmo_->visible = !selectOnly && !selection_.empty();
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

    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_N, false)) pendingAction_ = PendingAction::New;
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_O, false)) pendingAction_ = PendingAction::Open;
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_S, false)) saveScene();
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_D, false)) duplicateSelected();

    if (ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
        if (commands_.undo()) document_.setDirty(true);
    }
    if (ctrl && (ImGui::IsKeyPressed(ImGuiKey_Y, false) ||
                 (shift && ImGui::IsKeyPressed(ImGuiKey_Z, false)))) {
        if (commands_.redo()) document_.setDirty(true);
    }

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
    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) focusSelected();
    if (ImGui::IsKeyPressed(ImGuiKey_Delete, false)) deleteSelected();
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) selectObject(nullptr);
}

void EditorApp::handleFileDrop(const std::vector<std::string>& paths) {

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
            assignTextureToSelection(path);
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
}

void EditorApp::persistSettings() {

    settings_.bottomPanelOpen = bottomPanelOpen_;
    settings_.save(settingsPath_);
}
