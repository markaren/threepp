
#include "EditorApp.hpp"

#include "EditorTheme.hpp"
#include "PanelLayout.hpp"

#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/AnimationPlaySession.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"

#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
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
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/renderers/RendererFactory.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>

// GLFW's window-title setter. Declared rather than included: the GLFW headers
// are private to the threepp target, but the symbol is linked into it and the
// signature is a plain C entry point taking an opaque window handle.
extern "C" void glfwSetWindowTitle(void* window, const char* title);

using namespace threepp;
using namespace threepp::editor;

namespace {

    constexpr int kDefaultWidth = 1600;
    constexpr int kDefaultHeight = 900;

    // Pixel budgets at 100% DPI; everything is multiplied by the monitor
    // content scale at draw time.
    constexpr float kHierarchyWidth = 280.f;
    constexpr float kInspectorWidth = 340.f;
    constexpr float kBottomHeight = 200.f;

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
        // Markers and the frustum helper point into the outgoing graph too.
        // Drop them before anything can dereference an owner that is gone.
        clearViewportMarkers();
        if (cameraHelper_) {
            cameraHelper_->removeFromParent();
            cameraHelper_.reset();
            cameraHelperFor_ = nullptr;
        }
        // Recorded commands point into the old scene too. Re-resolve their
        // targets by uuid against the new graph; commands that cannot (raw
        // captures in property setters) are dropped rather than left dangling.
        commands_.rebind(scene);
        const auto uuid = selection_.uuid();
        selection_.set(nullptr);
        gizmo_->detach();
        // The overlay group survives the scene swap, so the outline must be
        // detached from it, not just dropped.
        if (selectionBox_) {
            selectionBox_->removeFromParent();
            selectionBox_.reset();
        }
        if (!uuid.empty()) {
            Object3D* found = nullptr;
            scene.traverse([&](Object3D& o) {
                if (!found && o.uuid == uuid) found = &o;
            });
            if (found) selectObject(found);
        }
    });

#ifdef THREEPP_EDITOR_WITH_PHYSX
    play_.addSession(std::make_shared<PhysicsPlaySession>());
#endif
    play_.addSession(std::make_shared<AnimationPlaySession>());

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
}

EditorApp::~EditorApp() {

    canvas_.setIOCapture(nullptr);

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
                child == markers_.get() ||
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

    std::cout << "[selftest] " << (failed == 0 ? "ALL PASS" : "FAILED") << std::endl;
    return failed == 0 ? 0 : 1;
}

void EditorApp::frame(float dt) {

    play_.update(dt);
    pollImports(dt);
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

    if (preview_.active) {
        auto* draw = ImGui::GetForegroundDrawList();
        const ImVec2 min(preview_.x, preview_.y);
        const ImVec2 max(preview_.x + preview_.w, preview_.y + preview_.h);
        draw->AddRect(ImVec2(min.x - 1, min.y - 1), ImVec2(max.x + 1, max.y + 1),
                      ImGui::GetColorU32(ImGuiCol_Border));
        const ImVec2 pad(6.f * contentScale_, 3.f * contentScale_);
        const ImVec2 textSize = ImGui::CalcTextSize(preview_.label.c_str());
        draw->AddRectFilled(min, ImVec2(min.x + textSize.x + 2 * pad.x, min.y + textSize.y + 2 * pad.y),
                            ImGui::GetColorU32(ImGuiCol_WindowBg, 0.85f));
        draw->AddText(ImVec2(min.x + pad.x, min.y + pad.y),
                      ImGui::GetColorU32(ImGuiCol_Text), preview_.label.c_str());
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
        activeImport_->future = std::async(std::launch::async, [path] {
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
    std::shared_ptr<Group> group;
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

    group->name = ObjectFactory::uniqueName(document_.scene(), path.stem().string());
    addObject(group, document_.scene(), "Import " + path.filename().string());
    settings_.modelDir = path.parent_path().string();

    char message[192];
    std::snprintf(message, sizeof(message), "Imported %s (%zu nodes, %.1fs)",
                  path.filename().string().c_str(), nodes, elapsed);
    log(message);
    flashStatus(message);
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
    const float bottom = viewport->Size.y - statusHeight_ - (bottomPanelOpen_ ? kBottomHeight * s : 0.f);
    ImGui::SetNextWindowPos({viewport->Pos.x + kHierarchyWidth * s + 12 * s,
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

void EditorApp::renderCameraPreview() {

    preview_.active = false;

    auto* selected = selection_.get();
    auto* cam = selected ? selected->as<PerspectiveCamera>() : nullptr;
    if (!cam) return;

    const auto size = renderer_->size();
    const auto width = static_cast<float>(size.width());
    const auto height = static_cast<float>(size.height());

    // The 3D viewport region, window coordinates (origin top-left).
    const float s = contentScale_;
    const float left = kHierarchyWidth * s;
    const float right = width - kInspectorWidth * s;
    const float top = menuHeight_ + toolbarHeight_;
    const float bottom = height - statusHeight_ - (bottomPanelOpen_ ? kBottomHeight * s : 0.f);
    const float viewW = right - left;
    const float viewH = bottom - top;
    if (viewW < 160.f * s || viewH < 120.f * s) return;

    // Bottom-right inset, shaped like the viewport so the framing matches
    // what a fullscreen render from this camera would show.
    const float margin = 12.f * s;
    float w = std::clamp(viewW * 0.30f, 160.f * s, 480.f * s);
    float h = w * (viewH / viewW);
    if (h > viewH * 0.5f) {
        h = viewH * 0.5f;
        w = h * (viewW / viewH);
    }
    const float x = right - margin - w;
    const float y = bottom - margin - h;

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

    preview_.x = x;
    preview_.y = y;
    preview_.w = w;
    preview_.h = h;
    preview_.label = cam->name.empty() ? std::string("Camera") : cam->name;
    preview_.active = true;
}

void EditorApp::refreshSelectionHelpers() {

    if (selectionBox_ && selection_.get()) {
        selectionBox_->update();
    }

    syncViewportMarkers();
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

    std::string error;
    if (!play_.stop(document_, &error)) {
        log("stop failed: " + error);
        return;
    }
    log("play stopped, scene restored");
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
        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (extension == ".json") {
            if (document_.dirty()) {
                pendingAction_ = PendingAction::OpenPath;
                pendingPath_ = path;
            } else {
                openScene(path);
            }
        } else if (extension == ".hdr") {
            setEnvironment(path, true);
        } else if (extension == ".obj" || extension == ".dae" || extension == ".gltf" ||
                   extension == ".glb" || extension == ".stl" || extension == ".fbx") {
            importModel(path);
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
