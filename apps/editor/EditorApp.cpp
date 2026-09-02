
#include "EditorApp.hpp"

#include "EditorTheme.hpp"
#include "ExampleScenes.hpp"
#include "ImportFormats.hpp"
#include "PanelLayout.hpp"

#include "threepp/extras/editor/AnimationPlaySession.hpp"
#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/SoundConfig.hpp"
#include "threepp/extras/editor/GeneratorConfig.hpp"
#include "threepp/extras/editor/MaterialTextureSlots.hpp"
#include "threepp/extras/editor/FlockPlaySession.hpp"
#include "threepp/extras/editor/ParticleFieldPlaySession.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/ScriptWorkspace.hpp"
#include "threepp/extras/editor/SplatImportConfig.hpp"
#include "threepp/extras/editor/SplatSurfaceCache.hpp"
#include "threepp/extras/editor/SplatSurfaceConfig.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"
#include "threepp/extras/editor/ViewSpec.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

#ifdef THREEPP_EDITOR_WITH_PYTHON
#include "ScriptHost.hpp"// runAuthoringSource, for the Generator's Regenerate
#endif

#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"

#ifdef THREEPP_WITH_AUDIO
#include "threepp/audio/Audio.hpp"
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
#include "threepp/loaders/EXRLoader.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/SogLoader.hpp"
#include "threepp/loaders/SplatLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatLod.hpp"
#include "threepp/renderers/GLRenderer.hpp"
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_set>

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

    // The backend, named here in code. Deliberately NOT the examples' interactive
    // createRenderer (examples/libs/renderer_factory.hpp): handed no preference it
    // prints a console menu and blocks on std::cin, which a windowed app must
    // never do — it stalls the editor behind a prompt nobody sees and hangs any
    // piped or scripted run.
    std::unique_ptr<Renderer> makeRenderer(Canvas& canvas, bool vulkan) {

#ifdef THREEPP_WITH_VULKAN
        if (vulkan) return std::make_unique<VulkanRenderer>(canvas);
#else
        if (vulkan) {
            std::cerr << "threepp editor: built without Vulkan support, using OpenGL\n";
        }
#endif
        return std::make_unique<GLRenderer>(canvas);
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

    bool isDescription(const std::filesystem::path& path) {

        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        return extension == ".urdf" || extension == ".xacro";
    }

    // What a queued file wants to be told before it can be expanded. Only a
    // robot description is asked: scanArgs on a 200 MB .glb would parse the
    // whole thing as XML, fail, and answer the same empty vector.
    std::vector<xacro::ArgDecl> declaredXacroArgs(const std::filesystem::path& path) {

        if (!isDescription(path)) return {};
        return xacro::scanArgs(path);
    }

    // A SOG / SSOG scan: a directory of chunks, or the .zip / .sog archive of
    // one. Decided by content, like the PLY path below it.
    //
    // It gets the SAME half-turn about X the .ply path applies, and the reason
    // is worth writing down because the format documentation points the other
    // way. SOG v2 declares itself right-handed with +Y up, which reads like a
    // promise that no flip is needed — it is not. That line describes the
    // CONTAINER's axis convention, not which way the scan inside it was
    // reconstructed. splat-transform re-encodes an existing 3DGS .ply without
    // reorienting it, and on the Sanctuaire scan the decoded means match that
    // .ply's to 8e-13 with no negation on any axis. So a SOG made from a COLMAP
    // capture is +Y DOWN exactly like its .ply, and skipping the flip lands the
    // basilica on its roof. Measured, not reasoned: the first import without it
    // came out upside down.
    //
    // A multi-level asset is read at level 0, its finest — "import my scan"
    // means the scan, not a proxy — with the level recorded so a future
    // serialization pass can reproduce the choice.
    std::shared_ptr<Object3D> loadSplatSog(const std::filesystem::path& path, bool vulkanBackend) {

        const auto info = SogLoader::describe(path);

        // A multi-level asset imports for DYNAMIC LOD: every other level in one
        // cloud (splats::loadSogWithLod says why every other), with the level
        // table travelling on the cloud so the per-frame policy in render()
        // finds it. A single-level asset imports exactly as before.
        //
        // VULKAN ONLY. The GL path ignores submission ranges, so a multi-level
        // cloud there draws every resident level stacked — the scan two or
        // three times over, fatter and slower, with no way to select. On a GL
        // editor a multi-level asset imports at its finest level, exactly as it
        // did before dynamic LOD existed. (The editor DEFAULTS to GL; --vulkan
        // is the flag, and it is where all of the splat perf work lives.)
        splats::SogLodResult loaded;
        if (vulkanBackend) {
            loaded = splats::loadSogWithLod(path);
        } else {
            loaded.data = SogLoader::load(path);
        }
        auto& data = loaded.data;
        auto& lodTable = loaded.table;

        editor::SplatImportConfig config;
        const auto srcU8 = path.u8string();
        config.source = std::string(srcU8.begin(), srcU8.end());
        config.lod = info.lodLevels > 1 ? 0 : -1;

        // The same cull the .ply path applies, and for the same reason: a scan
        // is a scan whatever container it arrived in. NOT under dynamic LOD
        // though: the cull reorders and removes splats, which would invalidate
        // every offset in the level table. The outliers a coarse level carries
        // are the price of the table staying true.
        if (lodTable.empty()) {
            config.culled = true;
            config.removed = data.removeOutliers();
        }

        auto cloud = SplatCloud::create(std::move(data));
        cloud->setLodTable(std::move(lodTable));

        // On the NODE, not in the data, for the same reason the .ply path puts
        // it there: which way is up belongs to the scene, and the gizmo can
        // undo it.
        cloud->rotation.x = math::PI;
        config.flippedX = true;

        config.write(*cloud);

        return cloud;
    }

    // A .ply, decided by its header rather than its name. Runs on the import
    // worker, so everything expensive here — the parse, the outlier cull, the
    // covariance and data-texture build inside SplatCloud — is off the UI
    // thread. Nothing touches GL: the cloud's DataTextures are plain CPU
    // buffers until the renderer first draws them.
    std::shared_ptr<Object3D> loadSplatPly(const std::filesystem::path& path) {

        // isSplatPly answers false for a file it cannot OPEN, so keep
        // "unreadable" and "not a splat" apart here — otherwise a filesystem
        // problem (a path that did not survive an encoding trip, most of all)
        // gets reported as a file-format verdict, which is exactly the wrong
        // trail to send someone down.
        if (std::ifstream probe(path, std::ios::binary); !probe) {
            throw std::runtime_error("cannot open the file, so no header was read"
                                     " - a path or permissions problem, not a format one");
        }

        const bool splat = SplatLoader::isSplatPly(path);
        const bool pointCloud = !splat && SplatLoader::isPointCloudPly(path);
        if (!splat && !pointCloud) {

            // Not a splat scan and not a point cloud, so it belongs to the
            // mesh path — which is unchanged, and which is where it stops:
            // threepp has no mesh PLY loader today, and ModelLoader refuses
            // the extension. Saying so here beats letting "no importable
            // content in the file" stand for both "your scan is malformed"
            // and "we cannot read mesh PLYs".
            ModelLoader loader;
            if (auto group = loader.load(path)) return group;
            throw std::runtime_error(
                    "no f_dc_0 property and no vertex-only element in the PLY header, so this"
                    " is neither a Gaussian splat scan nor a point cloud"
                    " - and threepp has no mesh PLY loader");
        }

        // A colour-only point cloud (a laser scan, a photogrammetry export
        // without Gaussians) takes the same object: one degree-0 Gaussian per
        // point, sized from the cloud's median neighbour spacing, so it
        // renders as a surface at point mix 0 and as its dots at 1 (the
        // Splats section of the inspector has the slider). No outlier cull:
        // the rule is written for the scale tail of an optimiser's output,
        // and a point cloud has no scales of its own.
        SplatLoader::PointCloudInfo pcInfo;
        auto data = pointCloud ? SplatLoader::loadPointCloudPly(path, {}, &pcInfo)
                               : SplatLoader::loadPly(path);

        // The two defaults a user means by "import my scan", both of them the
        // gaussian_splats example's, and both recorded so a future
        // serialization pass can reproduce this import from the file alone.
        editor::SplatImportConfig config;
        config.pointCloud = pointCloud;
        // Stored as UTF-8; .string() narrows through the ANSI code page and
        // would mangle the same paths the drop handler just went out of its
        // way to decode correctly.
        const auto srcU8 = path.u8string();
        config.source = std::string(srcU8.begin(), srcU8.end());

        // Photogrammetry output carries a long tail of enormous near-opaque
        // splats that render as fog over the subject. The rule is
        // percentile-based against the cloud's own distribution, so it is a
        // no-op on a clean scan and carries no unit.
        if (!pointCloud) {
            config.culled = true;
            config.removed = data.removeOutliers();
        }

        auto cloud = SplatCloud::create(std::move(data));

        // COLMAP — and the 3DGS pipelines built on it — put +Y down, so a scan
        // arrives upside down in a +Y-up scene. The conventional half-turn
        // about X is the fix, and it belongs on the NODE rather than in the
        // data: which way is up is a property of the scene the cloud is being
        // placed into, not of the file. The gizmo can undo it.
        cloud->rotation.x = math::PI;
        config.flippedX = true;

        config.write(*cloud);

        return cloud;
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
      renderer_(makeRenderer(canvas_, options.vulkan)),
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

    // Arms the camera dock's exact-pixel path when (and only when) the backend
    // can do it. On OpenGL this leaves the pane inert and the dock keeps its
    // scissored second render.
    dockPane_.attach(renderer_.get());

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
    // Off until asked for: the grid already says where the ground is, and three
    // coloured sticks at the origin are furniture in every scene that does not
    // happen to be authored around it. View > Origin Axes switches them on.
    axes_->visible = false;
    overlay_->add(axes_);

    markers_ = Group::create();
    markers_->name = "__editor_markers";
    overlay_->add(markers_);

    splines_ = Group::create();
    splines_->name = "__editor_splines";
    overlay_->add(splines_);

    conveyors_ = Group::create();
    conveyors_->name = "__editor_conveyors";
    overlay_->add(conveyors_);

    // The preview fields. Editor-only through the overlay, so a ParticleField —
    // which has no ObjectLoader case and would export as its zero-area
    // placeholder — can never reach a saved document.
    particles_ = Group::create();
    particles_->name = "__editor_particles";
    overlay_->add(particles_);

    // Editor-only like the overlay, but a SIBLING of it rather than a child: the
    // overlay is hidden for the duration of every sensor scan (a depth camera
    // pointed at the grid otherwise measures the grid), and a sensor must not be
    // hidden from itself. See SensorPlaySession.
    sensorRig_ = Group::create();
    sensorRig_->name = "__editor_sensor_rig";
    document_.addEditorOnly(*sensorRig_);

    // Orbiting while dragging a handle fights the gizmo; and a drag is exactly
    // the span an undo entry should cover.
    gizmoDragHandler_ = [this](Event& event) {
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
    };

    // Builds orbit_ and gizmo_ against the perspective camera. Called again
    // whenever the projection changes.
    bindViewportControls();
    orbit_->target.set(0, 0.5f, 0);

    // --- ImGui --------------------------------------------------------------
    ui_ = std::make_unique<ImguiFunctionalContext>(canvas_, *renderer_, [this] { drawUi(); });

    ioCapture_.preventMouseEvent = [this] {
        // The view gizmo draws on the background list, so ImGui's own capture
        // knows nothing about it - while the pointer is on it, a drag must
        // not orbit and a click must not pick.
        // Same reasoning for the sculpt brush: with it armed over the selected
        // terrain, the press belongs to the stroke and the orbit must not spin
        // the camera under it. A miss falls through and navigation is normal.
        return ImGui::GetIO().WantCaptureMouse || viewGizmoHovered_ || toolPaletteHovered_ ||
               sculptOwnsMouse();
    };
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
        clearConveyorOverlays();
        // Keyed by the outgoing scene's uuids, and each preview field is a
        // structural scene change the next sync rebuilds from whatever the new
        // graph authored.
        clearParticleOverlays();
        clearTreeOverlays();
        // The rings are keyed by the outgoing scene's uuid; the audition is
        // playing a file for a node that is about to stop existing.
        clearSoundRings();
        // Keyed by uuid too, and placed off a node that is going away.
        clearJointHelper();
        clearVehicleHelper();
        stopAudition();
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
        // Baked scan surfaces are keyed by cloud uuid. A cloud that survives
        // the swap — a Play/Stop hands the same live object back — keeps its
        // entry warm; the pruning happens below, once the new graph is here.
        splatBakeNode_.clear();
        splatBakeStats_.clear();
        // The previews drew those meshes, and they hang off the surviving
        // overlay rather than off the outgoing graph.
        clearSplatSurfacePreviews();
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

        // Entries for clouds that did not come back are unreachable megabytes.
        if (splatSurfaces_) splatSurfaces_->retainClouds(scene);

        commands_.rebind(scene);
        if (!uuid.empty()) {
            Object3D* found = nullptr;
            scene.traverse([&](Object3D& o) {
                if (!found && o.uuid == uuid) found = &o;
            });
            if (found) selectObject(found);
        }
    });

    // A dropped undo entry is a promise withdrawn, so say WHICH one out loud. The
    // budget only ever bites on splat scans, where one deletion held in history is
    // a couple of gigabytes of host memory.
    commands_.onPrune([this](const std::vector<std::string>& dropped, std::size_t bytesFreed) {
        constexpr double gib = 1024.0 * 1024.0 * 1024.0;
        constexpr std::size_t named = 3;// enough to be specific, short enough to read

        std::ostringstream message;
        message << "undo history: dropped ";
        for (std::size_t i = 0; i < std::min(named, dropped.size()); ++i) {
            message << (i ? ", " : "") << '"' << dropped[i] << '"';
        }
        if (dropped.size() > named) message << " and " << (dropped.size() - named) << " more";
        message << std::fixed << std::setprecision(2)
                << " to free " << static_cast<double>(bytesFreed) / gib << " GiB (budget "
                << static_cast<double>(commands_.byteLimit()) / gib << " GiB)";
        log(message.str());
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

    // The surface-bake memo. Built in every configuration, because the
    // inspector authors and reports in every configuration; it declines to bake
    // where it cannot (see SplatSurfaceCache).
    splatSurfaces_ = std::make_shared<editor::SplatSurfaceCache>();

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // Kept as a member too: the collider overlay reads the world it builds.
    physics_ = std::make_shared<PhysicsPlaySession>();
    // Soft bodies can decline to cook, and the GPU world can decline to come
    // up; both are worth a line in the log rather than silence.
    physics_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(physics_);
    // Right after physics: its start() borrows the world physics just built,
    // and stopping in reverse order tears the belts down while that world is
    // still alive.
    conveyorSession_ = std::make_shared<ConveyorPlaySession>();
    conveyorSession_->setPhysics(physics_.get());
    play_.addSession(conveyorSession_);
#endif
    // Right after the conveyor, and OUTSIDE its guard: a particle field needs a
    // renderer rather than a physics world, so it plays in every build. The
    // previews are parked for the duration (ParticleOverlay) and this session
    // owns its own fields and its own clock from t = 0.
    particleSession_ = std::make_shared<ParticleFieldPlaySession>();
    particleSession_->setRenderer(renderer_.get());
    // The viewport camera, asked for per frame rather than pinned: the ortho
    // views have cameras of their own, and the preview overlay follows the same
    // one (syncParticleOverlays), so a follow field wraps about the same point
    // whether the scene is being authored or played.
    particleSession_->setViewpoint([this] {
        Vector3 position;
        viewCamera().getWorldPosition(position);
        return position;
    });
    particleSession_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(particleSession_);

#ifdef THREEPP_EDITOR_WITH_PHYSX
    // Right after the particle fields and back inside the PhysX guard: grains
    // are a PBD simulation in the world physics built, so this borrows that
    // world exactly as the conveyor does — and emits BETWEEN steps, which is
    // what registering after the physics session buys. It needs the renderer
    // too, but only to ask which backend it is: that resolves the authored
    // "auto" visual to a particle field on Vulkan and an InstancedMesh on GL.
    granularSession_ = std::make_shared<GranularPlaySession>();
    granularSession_->setPhysics(physics_.get());
    granularSession_->setRenderer(renderer_.get());
    granularSession_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(granularSession_);

#ifdef THREEPP_WITH_VULKAN
    // Baked scan surfaces, right after the grains and still inside the PhysX
    // guard: the collider is cooked into the world physics_ built, so the same
    // borrow-and-stop-first ordering applies. The memo it bakes into is the
    // app's — the inspector's "Bake now" fills the very same one, which is what
    // makes a warmed bake instant at Play.
    splatSurfaceSession_ = std::make_shared<SplatSurfacePlaySession>();
    splatSurfaceSession_->setPhysics(physics_.get());
    splatSurfaceSession_->setRenderer(renderer_.get());
    splatSurfaceSession_->setCache(splatSurfaces_.get());
    splatSurfaceSession_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(splatSurfaceSession_);
#endif

    // Characters, last of the world-borrowers: each one is a capsule controller
    // in the world physics_ built, so the same borrow-and-stop-first ordering
    // applies. It updates AFTER the physics step (registration order), which is
    // what a kinematic sweep wants — the world it sweeps through has already
    // settled this frame.
    characterSession_ = std::make_shared<CharacterPlaySession>();
    characterSession_->setPhysics(physics_.get());
    characterSession_->setLogger([this](const std::string& message) { log(message); });
    play_.addSession(characterSession_);
#endif

    {
        auto animations = std::make_shared<AnimationPlaySession>();
        // Defer authored characters to the session registered above — but only
        // when there IS one. Without PhysX nothing else would drive them, and
        // an authored character would stand in its bind pose.
        animations->setSkipCharacters(characterSession_ != nullptr);
        play_.addSession(animations);
    }
    // Ambient flocks. Dependency-free (no PhysX, no renderer coupling) and
    // stateless between plays. The mesh filter is NOT optional here: the
    // editor scene carries overlay meshes (gizmo handles, light markers,
    // waypoint pucks) in the same graph the perch bake traverses, and a
    // marker hovering at altitude reads as the highest surface in its column
    // — the whole flock then climbs to a phantom floor.
    {
        auto flockSession = std::make_shared<FlockPlaySession>();
        flockSession->setMeshFilter(
                [this](const Mesh& mesh) { return !document_.isEditorOnly(mesh); });
        play_.addSession(flockSession);
    }
#ifdef THREEPP_WITH_AUDIO
    // Sounds. Kept as a member for the status readout and the selftest. Its
    // listener rides the perspective viewport camera, which is the closest
    // thing the editor has to "where the user is standing" — the ortho views
    // are a drafting aid, not a vantage.
    audio_ = std::make_shared<AudioPlaySession>();
    audio_->setLogger([this](const std::string& message) { log(message); });
    audio_->setListenerHost(&camera_);
    play_.addSession(audio_);
#endif
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
    } else if (!options_.openOnStart.empty() && formats::isScene(options_.openOnStart)) {
        openScene(options_.openOnStart);
    } else {
        buildTemplateScene();
        // Any other startup path goes through the same dispatch as a file
        // drop, so `threepp_editor model.glb` imports into the template scene.
        // The self-test drives its own import instead (with assertions).
        if (!options_.openOnStart.empty() && !options_.selfTest) {
            // handleFileDrop expects UTF-8, because that is what GLFW drops
            // deliver; .string() would narrow through the ANSI code page.
            const auto u8 = options_.openOnStart.u8string();
            handleFileDrop({std::string(u8.begin(), u8.end())});
        }
    }

    // After the document, so it lights whatever ended up open — and before Play, so a
    // --screenshot run is lit by the time it shoots.
    if (!options_.environment.empty()) {
        setEnvironment(options_.environment, /*alsoBackground*/ true);
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

    // Before the renderer goes: the pane holds a view handle into it. (The pane
    // is a member and would release itself, but member destruction order runs
    // after this body and the renderer is declared above it.)
    dockPane_.release();

    // Stops the poll and takes the scratch .py with it. Whatever was saved is
    // already in the document; what is left on disk is a copy nobody will read.
    stopExternalEdit("the editor is closing");

    // A window closed mid-Play never went through Stop. Run it now, while the
    // sessions can still tear down in LIFO order: dropping them below unstopped
    // would destroy the PhysX world under the character session's controller
    // manager, and the foundation then refuses to release ("pending module
    // references"). No-op when already stopped.
    stopPlay();

    // The sensor session parents nodes into sensorRig_, which is a member of this
    // same object: member destruction order would take the rig down first and
    // leave the session's destructor unlinking from freed memory. Drop the
    // sessions here, while everything they point into is still alive.
    play_.clearSessions();
    sensors_.reset();
    physics_.reset();
#ifdef THREEPP_WITH_AUDIO
    // Same reasoning: its sounds are parented into the scene document_ owns.
    audio_.reset();
    // And the audition's, before the listener it was opened on goes.
    stopAudition();
#endif
    if (sensorRig_) {
        sensorRig_->clear();
        document_.removeEditorOnly(*sensorRig_);
    }

    // Tear the overlay down before the members it points at (the gizmo owns a
    // pimpl that unregisters canvas listeners in its destructor).
    if (gizmo_) {
        gizmoDragSub_.unsubscribe();
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

    // Before the sessions step, so this frame's step drives on this frame's keys.
    updateVehicleTeleop(dt);
    updateCharacterTeleop(dt);
    play_.update(dt);
    // And after it, so the camera chases where the character ended up rather
    // than where it started — a one-frame-stale chase reads as lag on top of
    // the lag the filter is deliberately adding.
    updateCharacterCamera(dt);
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
    // The gizmo's animated snap, before the orbit re-derives its spherical
    // from wherever this leaves the camera.
    updateViewTween(dt);
    orbit_->update();
    updateViewPreset();
    // The nudge that keeps the grid off a coplanar ground is measured against
    // the camera, so it is re-derived every frame rather than on view changes.
    updateGridPlacement();

    refreshSelectionHelpers(dt);

    // Dynamic splat LOD, before the render so this frame draws the choice.
    // Only clouds imported with a multi-level table participate (lodTable()
    // empty otherwise); the policy picks the coarsest level that still covers
    // the cloud's projected footprint — so leaning in is always the finest
    // level — and frustum-culls its chunks. Uses the same camera the frame
    // renders with, which during Play is the play camera.
    {
        auto& cam = viewCamera();
        const int viewH = canvas_.size().height();
        document_.scene().traverse([&](Object3D& o) {
            if (auto* sc = dynamic_cast<SplatCloud*>(&o); sc && !sc->lodTable().empty())
                splats::selectLod(*sc, sc->lodTable(), cam, viewH);
        });
    }

    // Before the render, not after: on Vulkan the camera dock is a secondary
    // view that the renderer records INSIDE render(), and the camera it points
    // at has to be this frame's camera. (Play and Stop replace the scene, and
    // with it every camera object in it.)
    syncCameraDockPane();
    renderer_->render(document_.scene(), viewCamera());
    dockPane_.endFrame();
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
    // Before the panels: the Scripts tab lives in the bottom panel, and which
    // scripts are in it — if any — is decided here.
    updateScriptEditors();
    drawHierarchy();
    drawInspector();
    drawBottomPanel();
    drawStatusBar();
    drawViewGizmo();
    drawToolPalette();
    drawTransportBar();
    drawViewpointPicker();
    drawImportToast();

    if (preview_.visible) {
        // Background list, not foreground: the camera image is drawn by the
        // renderer before any ImGui at all, so this layer sits over it while
        // still passing under dialogs and menus.
        auto* draw = ImGui::GetBackgroundDrawList();
        const auto* viewport = ImGui::GetMainViewport();
        const ImVec2 min(viewport->Pos.x + preview_.x, viewport->Pos.y + preview_.y);
        const ImVec2 max(min.x + preview_.w, min.y + preview_.h);

        if (!preview_.active) {
            // Nothing rendered into it: paint the dock so the corner reads as
            // panel rather than as a scrap of viewport nobody can reach.
            draw->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_WindowBg));
            // ...but only claim the dock is empty when it is. A Vulkan
            // secondary view is allocated at a frame boundary, so for the frame
            // or two before it first draws the dock holds a camera and no
            // picture, and the hint would be wrong.
            if (!preview_.pending) {
                // Two different states, and telling them apart is the whole
                // point: a scene with cameras and an empty dock is something
                // the picker below can fix.
                const char* hint = preview_.hasCameras ? "No camera in dock" : "No camera in scene";
                const auto textSize = ImGui::CalcTextSize(hint);
                draw->AddText({min.x + (preview_.w - textSize.x) * 0.5f,
                               min.y + (preview_.h - textSize.y) * 0.5f},
                              ImGui::GetColorU32(theme::muted()), hint);
            }
        }

        draw->AddRect(min, max, ImGui::GetColorU32(ImGuiCol_Border));
        // The label used to be painted here. It is a picker now — which camera
        // the dock shows is a thing you set, not a readout of the selection.
        drawCameraDockPicker();
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
            case PendingDialog::Sound:
                settings_.soundDir = fileBrowser_.directory().string();
                if (auto* target = findByUuid(document_.scene(), soundTargetUuid_)) {
                    assignSound(*target, path);
                } else {
                    log("sound not attached - the object is no longer in the scene");
                }
                soundTargetUuid_.clear();
                break;
            case PendingDialog::SavePrefab:
                settings_.prefabDir = fileBrowser_.directory().string();
                if (auto* source = findByUuid(document_.scene(), prefabSourceUuid_)) {
                    savePrefab(*source, path);
                } else {
                    log("prefab not saved - the object is no longer in the scene");
                }
                prefabSourceUuid_.clear();
                break;
            case PendingDialog::AddPrefab: {
                settings_.prefabDir = fileBrowser_.directory().string();
                // No target, or one that has gone away since the dialog opened:
                // the scene root, which is where a prefab with nowhere else to
                // go belongs — better than dropping it on the floor.
                auto* parent = prefabTargetUuid_.empty()
                                       ? nullptr
                                       : findByUuid(document_.scene(), prefabTargetUuid_);
                addPrefab(path, parent ? *parent : static_cast<Object3D&>(document_.scene()));
                prefabTargetUuid_.clear();
                break;
            }
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
                                  settings_.sceneDir, formats::scenes());
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

    drawArgPrompt();

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

    drawViewportContextMenu();

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

    // The conveyor radius handle owns the mouse for as long as it is being
    // dragged (and for the release frame) — picking through a drag would
    // reselect and tear the handle out from under it.
    const bool radiusDragOwnsMouse = updateConveyorRadiusDrag();

    // The brush, before the pick gate: a stroke that ended on this frame still
    // owns the release, and re-picking under it would reselect out from under
    // the terrain being sculpted.
    const bool sculptOwnedPress = sculptOwnsMouse();
    updateSculpt();

    // Picking runs last: it must see the WantCaptureMouse produced by every
    // panel drawn this frame.
    if (!io.WantCaptureMouse && !fileBrowser_.isOpen() && !radiusDragOwnsMouse &&
        !sculptOwnedPress &&
        !viewGizmoHovered_ && !toolPaletteHovered_ &&
        ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !gizmo_->isDragging()) {
        // A click, not the end of an orbit drag.
        const auto drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.f);
        if (std::abs(drag.x) < 3.f * contentScale_ && std::abs(drag.y) < 3.f * contentScale_) {
            pickAt(io.MousePos.x, io.MousePos.y);
        }
    }

    // Same gate for the right button, where the drag threshold separates a
    // context click from the end of an orbit pan. A hit selects (so the menu,
    // the outline and the inspector all agree on the subject); a miss keeps
    // the selection and opens the menu for the empty space under the cursor.
    if (!io.WantCaptureMouse && !fileBrowser_.isOpen() && !radiusDragOwnsMouse &&
        !viewGizmoHovered_ && !toolPaletteHovered_ &&
        ImGui::IsMouseReleased(ImGuiMouseButton_Right) && !gizmo_->isDragging()) {
        const auto drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right, 0.f);
        if (std::abs(drag.x) < 3.f * contentScale_ && std::abs(drag.y) < 3.f * contentScale_) {
            viewportCtx_ = pickAt(io.MousePos.x, io.MousePos.y, /*deselectOnMiss=*/false);
            ImGui::OpenPopup("##viewportCtx");
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
    // The dock's camera belongs to the document that had it. Play and Stop
    // deliberately keep it (same uuids, new objects); a different document is
    // where it goes back to following whatever camera the scene brings.
    dockCamera_.reset();
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
    // Only now, on the way in: a failed open leaves the document (and so the
    // dock) exactly as it was.
    dockCamera_.reset();
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
    dockCamera_.reset();
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

    // The DOCUMENT, as the name says — not the editor's own furniture. The
    // overlay is an ordinary child of the scene (SceneDocument detaches it only
    // for export), so setFromObject(scene) measures it too, and the transform
    // gizmo inside it carries a bound of order 1e6: its picker planes are sized
    // against the camera distance rather than against anything in the world.
    // Framing that puts the camera a million units out and the actual scene a
    // sub-pixel speck at the far plane.
    //
    // Found by importing a splat scan and photographing a black viewport. It
    // needs a selection to bite (no selection, no attached gizmo), which is why
    // it survived: the paths that frame — open a scene, open an example — do it
    // before anything is selected, and an import does not reframe at all.
    Box3 box;
    for (auto* child : document_.scene().children) {

        if (!child || document_.isEditorOnly(*child)) continue;

        Box3 childBox;
        childBox.setFromObject(*child);
        if (!childBox.isEmpty()) box.union_(childBox);
    }
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
                          settings_.sceneDir, formats::scenes(), "scene.json");
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
    importQueue_.push_back({path, {}, false});
    log("import queued: " + path.filename().string());
}

void EditorApp::pollImports(float dt) {

    uiTime_ += dt;
    if (statusFlashRemaining_ > 0.f && (statusFlashRemaining_ -= dt) <= 0.f) statusFlash_.clear();

    if (!activeImport_ && !argPrompt_ && !importQueue_.empty()) {

        auto entry = importQueue_.front();
        importQueue_.pop_front();

        // A description that declares arguments cannot be expanded until someone
        // has said what they are — UR's ur_type defaults to a deliberately
        // invalid value so that an unanswered import fails loudly. Ask first,
        // and launch nothing this frame; the prompt's OK pushes the entry back.
        if (!entry.argsResolved) {
            auto declared = declaredXacroArgs(entry.path);
            if (!declared.empty() && !options_.selfTest) {
                argPrompt_ = std::make_unique<ArgPrompt>();
                argPrompt_->path = entry.path;
                argPrompt_->values.resize(declared.size(), std::vector<char>(1024, '\0'));
                argPrompt_->declared = std::move(declared);
                return;
            }
            // Nothing declared, or a headless self-test run, which has nobody to
            // ask and takes the file's own defaults.
            entry.argsResolved = true;
        }

        const auto path = entry.path;
        std::map<std::string, std::string> args;
        for (const auto& [name, value] : entry.args) args[name] = value;

        activeImport_ = std::make_unique<ActiveImport>();
        activeImport_->path = path;
        activeImport_->args = std::move(entry.args);
        // Which backend, decided HERE on the main thread and captured by value:
        // the worker must not touch renderer_, and the splat import needs to
        // know because multi-level dynamic-LOD import is Vulkan-only (the GL
        // path ignores submission ranges and would draw every level stacked).
#ifdef THREEPP_WITH_VULKAN
        const bool vulkanBackend = dynamic_cast<VulkanRenderer*>(renderer_.get()) != nullptr;
#else
        const bool vulkanBackend = false;
#endif
        // Loader exceptions surface through the future and are rethrown on
        // the main thread in the get() below.
        activeImport_->future = std::async(std::launch::async, [path, args, vulkanBackend]() -> std::shared_ptr<Object3D> {
            if (isDescription(path)) {
                URDFLoader loader;
                if (!args.empty()) loader.setArgs(args);
                auto robot = loader.load(path);
                // A xacro that fails knows exactly why — a missing package, an
                // argument nobody supplied, a file and a line — and that reason is
                // worth far more in the console than "nothing importable".
                if (!robot) {
                    if (const auto reason = loader.lastError(); !reason.empty()) {
                        throw std::runtime_error(reason);
                    }
                }
                return robot;
            }
            // Content decides, not the name. A SOG asset can be a directory,
            // a .sog or a .zip, and the only thing they share is a meta.json.
            if (SogLoader::isSog(path)) return loadSplatSog(path, vulkanBackend);

            // A directory that got this far is not a SOG, and no other importer
            // takes one — say which of the two it is rather than letting
            // ModelLoader report an extension it never saw.
            std::error_code dirEc;
            if (std::filesystem::is_directory(path, dirEc)) {
                throw std::runtime_error("no meta.json or lod-meta.json in this folder,"
                                         " so it is not a Gaussian splat scan");
            }

            const auto extension = formats::extensionOf(path);
            if (extension == ".zip" || extension == ".sog") {
                throw std::runtime_error("this archive holds no meta.json or lod-meta.json,"
                                         " so it is not a Gaussian splat scan"
                                         " - and threepp imports no other kind of archive");
            }

            if (formats::isSplatCandidate(path)) return loadSplatPly(path);
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
    const auto args = activeImport_->args;
    std::shared_ptr<Object3D> group;
    std::string error;
    try {
        group = activeImport_->future.get();
        // A splat cloud IS the imported object rather than a Group of meshes,
        // so it has no children and the emptiness test would throw away a
        // perfectly good import.
        auto* splat = group ? group->as<SplatCloud>() : nullptr;
        if (!group || (group->children.empty() && !splat)) {
            error = "no importable content in the file";
        } else if (splat && splat->splatCount() == 0) {
            error = "the file declares no splats";
        }
    } catch (const std::exception& e) {
        error = e.what();
    }
    activeImport_.reset();

    if (!error.empty()) {
        log("import failed: " + path.filename().string() + " - " + error);
        // A description says why it failed, down to the file and line, and the
        // console has already said it. A modal on top of that only asks for a
        // click before you can go and fix the thing it named.
        if (!isDescription(path)) importError_ = path.filename().string() + "\n\n" + error;
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
    //
    // A splat cloud is deliberately NOT marked: assetSource means "ObjectLoader
    // can rebuild this subtree by re-importing the file", and it cannot — see
    // SplatImportConfig, which records the same path for the serialization pass
    // without claiming something the loader would then fail to honour.
    std::error_code pathEc;
    const auto canonical = std::filesystem::weakly_canonical(path, pathEc);
    auto* splatCloud = group->as<SplatCloud>();
    if (!splatCloud) setAssetSource(*group, pathEc ? path : canonical);

    // A robot records where it came from: the joint table cannot be
    // serialised, so the document keeps a reference and rebuilds on load.
    std::size_t dof = 0;
    if (auto* robot = group->as<Robot>()) {
        dof = robot->numDOF();
        RobotConfig config;
        config.urdf = path.string();
        config.joints = robot->jointValues();
        // Whatever the arg prompt collected. Without this the file is only half
        // the answer, and every later rebuild would expand it differently.
        config.xacroArgs = args;
        config.write(*robot);
        // URDFLoader builds the collision hulls visible; they are wireframe
        // duplicates sitting on the visual meshes, so start them hidden.
        robot->showColliders(config.showColliders);
    }

    group->name = ObjectFactory::uniqueName(document_.scene(), path.stem().string());
    addObject(group, document_.scene(), "Import " + path.filename().string());
    settings_.modelDir = path.parent_path().string();

    char message[192];
    if (splatCloud) {
        // Splats, not nodes: a cloud is one node no matter how big the scan is,
        // and "1 node" is the least informative thing that could be said about
        // a quarter of a million Gaussians.
        std::snprintf(message, sizeof(message), "Imported %s (%zu splats, SH degree %d, %.1fs)",
                      path.filename().string().c_str(), splatCloud->splatCount(),
                      splatCloud->data().shDegree, elapsed);
    } else if (dof > 0) {
        std::snprintf(message, sizeof(message), "Imported %s (%zu nodes, %zu joints, %.1fs)",
                      path.filename().string().c_str(), nodes, dof, elapsed);
    } else {
        std::snprintf(message, sizeof(message), "Imported %s (%zu nodes, %.1fs)",
                      path.filename().string().c_str(), nodes, elapsed);
    }
    log(message);
    flashStatus(message);

    if (splatCloud) {
        if (const auto config = editor::SplatImportConfig::read(*splatCloud); config && config->culled) {
            log("culled " + std::to_string(config->removed) + " outlier splats, and flipped 180" +
                " degrees about X for +Y-up");
        }
        // Saved as a reference to the file plus these ops (ObjectExporter's
        // threeppSplat block), so the document stays small and reopening
        // re-imports the scan; a Play/Stop keeps the live object.
        log("saved by reference: \"" + splatCloud->name + "\" reloads from its file when the scene opens");
    }
}

void EditorApp::drawArgPrompt() {

    if (!argPrompt_) return;

    constexpr const char* title = "Import arguments";
    if (!argPrompt_->opened) {
        ImGui::OpenPopup(title);
        argPrompt_->opened = true;
    }

    const ImVec2 centre = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(centre, ImGuiCond_Appearing, {0.5f, 0.5f});
    ImGui::SetNextWindowSizeConstraints({460 * contentScale_, 0}, {820 * contentScale_, FLT_MAX});

    if (!ImGui::BeginPopupModal(title, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        // Dismissed with Esc, which reads as Cancel — reopening it would make the
        // dialog impossible to get out of by the one key that always means "no".
        if (!ImGui::IsPopupOpen(title)) {
            log("import cancelled: " + argPrompt_->path.filename().string());
            argPrompt_.reset();
        }
        return;
    }

    ImGui::TextUnformatted(argPrompt_->path.filename().string().c_str());
    ImGui::TextDisabled("%s", "Leave a field empty to use the file's own default.");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    for (std::size_t i = 0; i < argPrompt_->declared.size(); ++i) {

        const auto& decl = argPrompt_->declared[i];
        auto& buffer = argPrompt_->values[i];

        ImGui::TextUnformatted(decl.name.c_str());
        ImGui::SetNextItemWidth(-1);
        // The declared text is a HINT, never the field's contents. Pre-filling
        // would turn a default that is still being derived — UR's joint limits
        // path is built out of ur_type — into a frozen literal the moment the
        // dialog was opened.
        const std::string id = "##arg" + std::to_string(i);
        ImGui::InputTextWithHint(id.c_str(),
                                 decl.hasDefault ? decl.defaultValue.c_str() : "(no default - required)",
                                 buffer.data(), buffer.size());
        ImGui::Spacing();
    }

    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Button("Import", {110 * contentScale_, 0})) {

        PendingImport entry;
        entry.path = argPrompt_->path;
        entry.argsResolved = true;
        for (std::size_t i = 0; i < argPrompt_->declared.size(); ++i) {
            const std::string value = argPrompt_->values[i].data();
            if (value.empty()) continue;
            entry.args.emplace_back(argPrompt_->declared[i].name, value);
        }

        // Back to the front: it was popped to get here, and it is still the
        // import the user asked for first.
        importQueue_.push_front(std::move(entry));
        argPrompt_.reset();
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", {110 * contentScale_, 0})) {
        // Dropped, not failed: nobody asked for a robot built out of defaults
        // they just declined to accept, and an error modal would be a lie.
        log("import cancelled: " + argPrompt_->path.filename().string());
        argPrompt_.reset();
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }

    ImGui::EndPopup();
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
    const float free = total - menuHeight_ - statusHeight_ - 140.f * s;

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

    std::shared_ptr<Texture> texture;
    try {
        if (formats::extensionOf(path) == ".exr") {
            EXRLoader loader;
            texture = loader.load(path);
        } else {
            RGBELoader loader;
            texture = loader.load(path);
        }
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

void EditorApp::assignSound(Object3D& object, const std::filesystem::path& path) {

    if (rejectWhilePlaying("Attach Sound")) return;

    const auto before = SoundAuthoring::read(object);

    SoundAuthoring after = before;
    // A file dropped on an object that was not a sound yet makes it one, at the
    // defaults — the alternative is a file key nothing ever reads.
    if (!after.config) after.config = SoundConfig{};
    // Forward slashes, like the script reference: a saved document should not
    // read differently depending on which platform wrote it.
    after.file = path.empty() ? std::string{} : path.generic_string();

    // A new file is a new sound; an audition still playing the old one would
    // be lying about what the button says.
    if (isAuditioning(object)) stopAudition();

    auto* target = &object;
    commands_.execute(makeProperty<SoundAuthoring>(
            path.empty() ? "Clear Sound File" : "Set Sound File", "sound:" + object.uuid,
            [target](const SoundAuthoring& value) { value.write(*target); },
            before, after));
    document_.setDirty(true);

    if (path.empty()) {
        log("sound file cleared on " + (object.name.empty() ? object.type() : object.name));
        return;
    }
    settings_.soundDir = path.parent_path().string();
    log("sound " + path.filename().string() + " attached to " +
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

        // Both generations are held; whichever one is currently out of the scene
        // is the one the history is paying for. After a play-stop swap that can be
        // both, since the dead graph detached everything on its way out.
        void retainedRoots(std::vector<Object3D*>& out) const override {

            if (previous_) out.push_back(previous_.get());
            if (next_) out.push_back(next_.get());
        }

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

void EditorApp::addConveyorPoint(Object3D& conveyor, std::size_t index, const std::string& label) {

    // The spline twin, waypoint for control point — see addSplinePoint for the
    // placement rule (the path must visibly change).
    if (rejectWhilePlaying(label.c_str())) return;

    if (!ConveyorConfig::isConveyor(conveyor)) return;

    std::vector<Vector3> points;
    for (const auto* node : ConveyorConfig::waypointNodes(conveyor)) {
        points.push_back(node->position);
    }
    const auto count = points.size();
    const auto slot = std::min(index, count);

    Vector3 position;
    if (count == 0) {
        // Nothing to extend; the origin of the conveyor's own space.
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

    auto point = ObjectFactory::createConveyorPoint(conveyor);
    point->position.copy(position);

    auto* raw = point.get();
    commands_.execute(std::make_unique<AddObjectCommand>(
            conveyor, point, label, ConveyorConfig::childSlotForPointIndex(conveyor, slot)));
    document_.setDirty(true);
    selectObject(raw);
    scrollTo_ = raw;
}

void EditorApp::addConveyorWall(Object3D& conveyor, const std::string& label) {

    if (rejectWhilePlaying(label.c_str())) return;
    if (!ConveyorConfig::isConveyor(conveyor)) return;

    auto wall = ObjectFactory::createConveyorWall(conveyor);
    auto* raw = wall.get();
    commands_.execute(std::make_unique<AddObjectCommand>(conveyor, wall, label,
                                                         AddObjectCommand::atEnd));
    document_.setDirty(true);
    selectObject(raw);
    scrollTo_ = raw;
}

void EditorApp::addConveyorWallPoint(Object3D& wall, std::size_t index, const std::string& label) {

    if (rejectWhilePlaying(label.c_str())) return;
    if (!ConveyorWallConfig::isWall(wall)) return;

    const auto points = ConveyorWallConfig::pointNodes(wall);
    const auto count = points.size();
    const auto slot = std::min(index, count);

    // Placement thinks in the belt's own coordinates — station along the path
    // and lateral offset — NOT in straight-line extrapolation: a wall grown
    // around a bend must land its next point ON the edge it is following, and
    // a chord-extended point would leave the belt entirely. Everything runs in
    // the conveyor's space; the wall group's own transform bridges both ways
    // (the group may have been slid along the belt with the gizmo).
    namespace cv = threepp::conveyor;
    auto* conveyor = wall.parent;
    std::vector<Vector3> centerline;
    if (conveyor && ConveyorConfig::isConveyor(*conveyor)) {
        const auto config = ConveyorConfig::read(*conveyor).value_or(ConveyorConfig{});
        const auto spec = config.spec(*conveyor);
        centerline = cv::resamplePath(spec.waypoints, spec.smooth, spec.samples);
    }

    Matrix4 wallMatrix;
    wallMatrix.compose(wall.position, wall.quaternion, wall.scale);
    Matrix4 wallInverse(wallMatrix);
    wallInverse.invert();

    const auto projectPoint = [&](std::size_t i) {
        Vector3 p = points[i]->position;
        p.applyMatrix4(wallMatrix);
        return cv::projectOntoPath(p, centerline);
    };

    Vector3 position;
    if (centerline.size() >= 2 && count > 0) {
        // The march step a growing wall takes past its end.
        const float step = 0.6f;
        cv::PathProjection at;
        if (slot == 0) {
            at = projectPoint(0);
            at.station -= step;
        } else if (slot >= count) {
            at = projectPoint(count - 1);
            at.station += step;
        } else {
            const auto before = projectPoint(slot - 1);
            const auto after = projectPoint(slot);
            at.station = (before.station + after.station) * 0.5f;
            at.offset = (before.offset + after.offset) * 0.5f;
        }
        position = cv::pointOnPath(centerline, at.station, at.offset);
        position.applyMatrix4(wallInverse);
    } else if (count > 0) {
        // No path to follow (a wall orphaned from its conveyor): the plain
        // extrapolation fallback.
        if (slot == 0) {
            position.copy(points.front()->position);
            if (count > 1) position.sub(points[1]->position).add(points.front()->position);
        } else if (slot >= count) {
            position.copy(points.back()->position);
            if (count > 1) {
                position.sub(points[count - 2]->position).add(points.back()->position);
            } else {
                position.x += 0.5f;
            }
        } else {
            position.copy(points[slot - 1]->position)
                    .add(points[slot]->position)
                    .multiplyScalar(0.5f);
        }
    }

    auto point = ObjectFactory::createConveyorWallPoint(wall);
    point->position.copy(position);

    auto* raw = point.get();
    commands_.execute(std::make_unique<AddObjectCommand>(wall, point, label, slot));
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


// ------------------------------------------------------------------- prefabs

namespace {

    // The Save box's default name is the object's, and object names are free
    // text: a mesh called "Wheel / FL" would otherwise arrive there as a path.
    std::string prefabFileName(const Object3D& object) {

        const auto& name = object.name.empty() ? object.type() : object.name;

        std::string out;
        out.reserve(name.size());
        for (const unsigned char c : name) {
            out.push_back(c < 0x20 || std::strchr("<>:\"/\\|?*", c) ? '_' : static_cast<char>(c));
        }
        // Windows also refuses a name ending in a dot or a space.
        while (!out.empty() && (out.back() == '.' || out.back() == ' ')) out.pop_back();
        if (out.empty()) out = "prefab";

        return out + ".json";
    }

    // Fresh identity for a just-loaded prefab, and the reason the feature works
    // at all when the same file is instantiated twice.
    //
    // ObjectLoader adopts serialized uuids verbatim and ObjectExporter dedupes
    // by uuid. Two instances of one prefab therefore look like the same objects
    // to the NEXT save of the scene: one material entry would be written for
    // both, and reopening would hand both instances the same Material — where
    // recolouring one recolours the other. That is precisely the trap
    // duplicateSelected() clones materials to avoid, arriving by another road.
    //
    // GEOMETRY and TEXTURES deliberately keep their uuids, so the exporter's
    // dedupe still merges identical vertex and pixel data across instances.
    // That is the same sharing trade Duplicate already makes, and it is what
    // keeps a scene full of one prefab from being a scene full of copies of its
    // mesh data.
    void reidentifyPrefab(Object3D& root) {

        // One Material can sit on several meshes in the subtree; each DISTINCT
        // one gets a new identity once, not once per mesh that holds it.
        std::unordered_set<Material*> seen;

        root.traverse([&seen](Object3D& object) {
            object.uuid = math::generateUUID();

            auto* withMaterials = dynamic_cast<ObjectWithMaterials*>(&object);
            if (!withMaterials) return;
            for (const auto& material : withMaterials->materials()) {
                if (material && seen.insert(material.get()).second) {
                    material->setUuid(math::generateUUID());
                }
            }
        });
    }

}// namespace

std::string EditorApp::prefabStartDir() const {

    return settings_.prefabDir.empty() ? settings_.sceneDir : settings_.prefabDir;
}

void EditorApp::beginSavePrefab(Object3D& object) {

    // The scene root is the document; saving it as a prefab is File ▸ Save As.
    if (&object == &document_.scene() || document_.isEditorOnly(object)) return;

    // By uuid, like every other dialog that spans frames: a Play/Stop in
    // between replaces the whole graph and the pointer with it.
    prefabSourceUuid_ = object.uuid;
    pendingDialog_ = PendingDialog::SavePrefab;
    fileBrowser_.open("Save as Prefab", FileBrowser::Mode::Save,
                      prefabStartDir(), formats::scenes(), prefabFileName(object));
}

void EditorApp::savePrefab(Object3D& object, const std::filesystem::path& path) {

    std::string error;
    if (!document_.exportSubtree(object, path, &error)) {
        log("prefab save failed: " + error);
        return;
    }
    logWarnings();
    log("saved prefab " + path.filename().string());
}

void EditorApp::beginAddPrefab(Object3D& parent) {

    prefabTargetUuid_ = &parent == &document_.scene() ? std::string{} : parent.uuid;
    pendingDialog_ = PendingDialog::AddPrefab;
    fileBrowser_.open("Add Prefab", FileBrowser::Mode::Open,
                      prefabStartDir(), formats::scenes());
}

void EditorApp::addPrefab(const std::filesystem::path& path, Object3D& parent) {

    // addObject refuses too, but a parse only to throw the result away is worth
    // skipping — and the console line then names the real reason.
    if (rejectWhilePlaying("Add Prefab")) return;

    ObjectLoader loader;
    std::shared_ptr<Object3D> root;
    std::string error;
    try {
        root = loader.load(path);
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (!root) {
        if (error.empty()) error = "not a scene document";
        log("prefab failed: " + path.filename().string() + " - " + error);
        importError_ = path.filename().string() + "\n\n" + error;
        return;
    }
    for (const auto& warning : loader.warnings()) log("warning: " + warning);

    const auto stem = path.stem().string();

    // Somebody picked a whole scene, which is a perfectly good prefab — it is
    // the same document — except that a Scene cannot be a child of one. Its
    // content goes into a Group instead, so any saved scene can be dropped into
    // another one.
    if (auto scene = std::dynamic_pointer_cast<Scene>(root)) {
        auto group = Group::create();
        group->name = stem;
        // Copied first: removeFromParent() edits the vector being walked, and
        // the returned handle is what keeps each child alive across the move.
        const auto children = scene->children;
        for (auto* child : children) {
            if (auto kept = child->removeFromParent()) group->add(kept);
        }
        root = group;
    }

    reidentifyPrefab(*root);

    // A prefab whose root was never named answers to the file it came from.
    if (root->name.empty()) root->name = stem;
    root->name = ObjectFactory::uniqueName(document_.scene(), root->name);

    addObject(root, parent, "Add Prefab " + root->name);
    log("added prefab " + path.filename().string());
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
    // Same rule for the sound audition: it belongs to the inspector of the
    // object it is playing for.
    if (!auditionUuid_.empty() && (!object || object->uuid != auditionUuid_)) stopAudition();

    selection_.set(object);

    // Clicking a camera still aims the dock at it: that gesture is how most
    // people find the dock at all, and it costs nothing to keep. What it no
    // longer does is the reverse — clicking anything else leaves the dock
    // alone, because framing a shot means selecting the thing being framed.
    if (auto* camera = selectedCamera()) setDockCamera(camera);

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

    // A document written before the articulation extension flattens a Robot into
    // a plain Object3D: the pose survives, the joint table does not. Re-import
    // the referenced URDF and move its joint table onto the subtree the document
    // carries, keeping the placeholder's identity and placement so uuid lookups
    // (selection, undo rebinding) still resolve.
    //
    // Documents written since do not come through here at all — ObjectLoader
    // hands back a live Robot, which is the branch just below. That is the whole
    // point of the extension: a play/stop cycle no longer reads the URDF, so it
    // cannot rebuild the subtree from it, so it cannot delete anything authored
    // into it.
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
            // The same arguments the import used, or this rebuilds a different
            // robot out of the same file — the exact thing a play/stop cycle
            // must not do.
            loader.setArgs(config.argMap());
            robot = loader.load(config.urdf);
            if (!robot) error = loader.lastError();
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

        // Only the joint table is taken from the re-import; the subtree stays the
        // document's. That, the identity/placement handover and the reporting all
        // live in a free function so they can be tested headlessly — see
        // transplantRobot.
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


// ------------------------------------------------------------ sound audition

void EditorApp::startAudition(const Object3D& object) {

    stopAudition();

#ifdef THREEPP_WITH_AUDIO
    if (auditionUnavailable_) return;

    const auto config = SoundConfig::read(object);
    if (!config) return;

    const auto file = SoundConfig::resolveFile(SoundConfig::file(object),
                                               document_.path().empty()
                                                       ? std::filesystem::path{}
                                                       : document_.path().parent_path());
    if (file.empty()) {
        log("no sound file to audition on \"" + object.name + "\"");
        return;
    }

    if (!auditionListener_) {
        try {
            auditionListener_ = std::make_unique<AudioListener>();
        } catch (const std::exception& e) {
            // Said once per session: the button goes dead rather than logging
            // the same failure on every click.
            auditionUnavailable_ = true;
            log("audio device unavailable - audition disabled (" + std::string(e.what()) + ")");
            return;
        }
    }

    try {
        auditionSound_ = std::make_unique<Audio>(*auditionListener_, file);
    } catch (const std::exception& e) {
        log("could not play " + file.string() + " (" + e.what() + ")");
        return;
    }

    // FLAT on purpose: a plain Audio never spatializes, so what you hear is the
    // file at the authored volume and rate. Spatial audition would need the
    // listener to follow the editor camera through every orbit, and answers a
    // question the distance rings already answer better.
    auditionSound_->setLooping(config->loop);
    auditionSound_->setVolume(config->volume);
    auditionSound_->setPlaybackRate(config->rate);
    auditionSound_->play();

    auditionUuid_ = object.uuid;
#else
    (void) object;
#endif
}

void EditorApp::stopAudition() {

    auditionUuid_.clear();
#ifdef THREEPP_WITH_AUDIO
    // The listener stays: opening a device costs tens of milliseconds and the
    // next audition is usually one click away. Only the sound goes.
    auditionSound_.reset();
#endif
}

bool EditorApp::isAuditioning(const Object3D& object) const {

    return !auditionUuid_.empty() && auditionUuid_ == object.uuid;
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

namespace {

    // Perspective or orthographic; anything else is not a dock subject.
    Camera* asCamera(Object3D& object) {

        if (auto* persp = object.as<PerspectiveCamera>()) return persp;
        if (auto* ortho = object.as<OrthographicCamera>()) return ortho;
        return nullptr;
    }

}// namespace

Camera* EditorApp::selectedCamera() const {

    auto* selected = selection_.get();
    return selected ? asCamera(*selected) : nullptr;
}

std::vector<Camera*> EditorApp::sceneCameras() const {

    std::vector<Camera*> cameras;
    document_.scene().traverse([&](Object3D& object) {
        // The frustum helper and the marker icons are not scene cameras, and
        // the viewport camera is not in the scene at all.
        if (document_.isEditorOnly(object)) return;
        if (auto* cam = asCamera(object)) cameras.push_back(cam);
    });
    return cameras;
}

Camera* EditorApp::dockCamera() const {

    const auto cameras = sceneCameras();
    Camera* first = cameras.empty() ? nullptr : cameras.front();

    // Never chosen: follow the scene. Opening a scene that has a camera and
    // finding the dock blank is the thing this whole indirection exists to
    // avoid.
    if (!dockCamera_) return first;

    for (auto* cam : cameras) {
        if (cam->uuid == *dockCamera_) return cam;
    }

    // Chosen, but not in the scene right now. "None" means none; a camera that
    // was deleted (or is one undo away from coming back) falls back rather than
    // leaving the dock dead, and re-docks by uuid the moment it returns.
    return dockCamera_->empty() ? nullptr : first;
}

void EditorApp::setDockCamera(Camera* camera) {

    dockCamera_ = camera ? camera->uuid : std::string{};
}

void EditorApp::drawCameraDockPicker() {

    if (!preview_.visible) return;

    const auto cameras = sceneCameras();
    // Nothing to choose between, and the dock already says so in the middle.
    // A combo reading "None" over an empty list is worse than no combo.
    if (cameras.empty()) return;

    const float s = contentScale_;
    const float pad = 6.f * s;
    const auto& style = ImGui::GetStyle();
    // Never more than a third of the dock: this is a label that happens to be
    // clickable, and what the dock is for is the picture behind it.
    const float itemWidth = std::min(150.f * s, preview_.w / 3.f);
    if (itemWidth < 60.f * s || preview_.h < 40.f * s) return;

    const auto* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({viewport->Pos.x + preview_.x + pad,
                             viewport->Pos.y + preview_.y + pad});
    ImGui::SetNextWindowSize({itemWidth + style.WindowPadding.x * 2.f, 0.f});
    // The same plate the static label used to draw, now holding a widget.
    ImGui::SetNextWindowBgAlpha(0.85f);

    // NoFocusOnAppearing keeps a glance at the dock from pulling keyboard focus
    // off the hierarchy, and the window is deliberately NOT NoInputs: hovering
    // it raises WantCaptureMouse, which is also what stops a click on the dock
    // from picking through into the scene behind it.
    if (ImGui::Begin("##cameraDock", nullptr,
                     layout::barFlags | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoFocusOnAppearing)) {

        auto* current = dockCamera();
        const auto label = [](const Camera* camera) {
            return camera->name.empty() ? std::string("Camera") : camera->name;
        };

        ImGui::SetNextItemWidth(itemWidth);
        if (ImGui::BeginCombo("##dockCamera", current ? label(current).c_str() : "None")) {
            if (ImGui::Selectable("None", current == nullptr)) setDockCamera(nullptr);
            for (auto* camera : cameras) {
                // Names repeat freely in a scene; the uuid is what makes two
                // "Camera" rows two different rows to ImGui.
                ImGui::PushID(camera->uuid.c_str());
                if (ImGui::Selectable(label(camera).c_str(), camera == current)) {
                    setDockCamera(camera);
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("The camera this dock renders.\nIndependent of what is selected.");
        }
    }
    ImGui::End();
}

void EditorApp::syncCameraDockPane() {

    if (!dockPane_.supported()) return;

    float x = 0, y = 0, w = 0, h = 0;
    const bool haveRect = cameraDockRect(x, y, w, h);
    // A camera only, and only while the dock has room. Anything else releases
    // the view — a collapsed dock has no business holding a deferred chain.
    Camera* cam = haveRect ? dockCamera() : nullptr;
    dockPane_.sync(cam,
                   static_cast<int>(x), static_cast<int>(y),
                   static_cast<int>(w), static_cast<int>(h));
}

void EditorApp::renderCameraPreview() {

    preview_.active = false;
    preview_.pending = false;
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
    // Read by drawUi for the empty state, which has two quite different things
    // to say: a scene with no camera in it, versus a dock deliberately pointed
    // at nothing.
    preview_.hasCameras = !sceneCameras().empty();

    auto* dockCam = dockCamera();
    if (!dockCam) return;

    // Vulkan: the pixels are already in the frame. The renderer rendered this
    // camera as a full secondary view — its own G-buffer, its own shadows, GI
    // and reflections — and copied the result into the dock rect during
    // render(). Nothing left to draw here; the dock is EXACT rather than the
    // lit-pane approximation it used to be.
    if (dockPane_.supported()) {
        preview_.active = dockPane_.active();
        // Docked, but the view is still being stood up (it is allocated at a
        // frame boundary). Those pixels are the primary viewport's; say so.
        preview_.pending = !preview_.active;
        return;
    }

    // OpenGL: a second, scissored render of the same scene. Exact by
    // construction — it is the same renderer drawing the same frame — and
    // deliberately left alone.
    auto* cam = dockCam->as<PerspectiveCamera>();
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

    preview_.active = true;
}

void EditorApp::refreshSelectionHelpers(float dt) {

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
    syncConveyorOverlays();
    syncParticleOverlays(dt);
    syncTreeOverlays();
    syncSplatSurfacePreviews();
    syncPhysicsDebug();
    syncDebugDraw();
    syncSensorOverlay();
    syncCameraHelper();
    syncSoundRings();
    syncJointHelper();
    syncVehicleHelper();

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

Object3D* EditorApp::pickAt(float mouseX, float mouseY, bool deselectOnMiss) {

    const auto* viewport = ImGui::GetMainViewport();
    const float width = viewport->Size.x;
    const float height = viewport->Size.y;
    if (width <= 0.f || height <= 0.f) return nullptr;

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
                // Picking is the one selection route where the object was found
                // by pointing at it, not at its row — bring the row to it.
                scrollTo_ = owner;
                return owner;
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
        scrollTo_ = selectable;
        return selectable;
    }

    if (deselectOnMiss) selectObject(nullptr);
    return nullptr;
}

void EditorApp::drawViewportContextMenu() {

    if (!ImGui::BeginPopup("##viewportCtx")) return;

    // The stored pointer is only trusted while it is still THE selection:
    // whoever removes an object re-points the selection first (see Selection),
    // so a mismatch means the world changed under the open menu.
    if (viewportCtx_ && selection_.get() != viewportCtx_) {
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return;
    }

    const bool editable = !isPlaying();

    if (auto* object = viewportCtx_) {
        // The same actions as the hierarchy row's menu, on the same subject —
        // two menus that disagree about what can be done to an object would
        // read as two different editors.
        if (ImGui::BeginMenu("Add", editable)) {
            drawAddMenu(*object);
            ImGui::EndMenu();
        }
        ImGui::Separator();
        // The rename edit box lives in the hierarchy row; scrollTo_ makes sure
        // that row is on screen when it appears.
        if (ImGui::MenuItem("Rename", nullptr, false, editable)) {
            renaming_ = object;
            renameBuffer_ = object->name;
            scrollTo_ = object;
        }
        if (ImGui::MenuItem("Copy Name", nullptr, false, !object->name.empty())) {
            ImGui::SetClipboardText(object->name.c_str());
        }
        if (ImGui::MenuItem("Save as Prefab...", nullptr, false, editable)) {
            beginSavePrefab(*object);
        }
        if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, editable)) {
            deferred_ = [this] { duplicateSelected(); };
        }
        if (ImGui::MenuItem("Focus", "F")) focusSelected();
        if (ImGui::MenuItem(object->visible ? "Hide" : "Show", nullptr, false, editable)) {
            auto* target = object;
            const bool before = object->visible;
            commands_.execute(makeProperty<bool>(
                    before ? "Hide" : "Show", {},
                    [target](const bool& value) { target->visible = value; },
                    before, !before));
            document_.setDirty(true);
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Delete", "Del", false, editable)) {
            deferred_ = [this] { deleteSelected(); };
        }
    } else {
        // Empty space: the click has nothing to act on, so the menu is about
        // what could be here instead.
        if (ImGui::BeginMenu("Add", editable)) {
            drawAddMenu(document_.scene());
            ImGui::EndMenu();
        }
    }

    ImGui::EndPopup();
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

    // An explicit view set outranks a flight still on its way to an older one.
    viewTween_.active = false;

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
        gizmoDragSub_.unsubscribe();
        gizmo_->detach();
        gizmo_->removeFromParent();
    }
    gizmo_ = std::make_unique<TransformControls>(camera, canvas_);
    gizmo_->setSize(0.9f);
    overlay_->addRef(*gizmo_);
    gizmoDragSub_ = gizmo_->subscribe("dragging-changed", gizmoDragHandler_);
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
    // An audition talking over the scene's own sounds helps nobody.
    stopAudition();

#ifdef THREEPP_WITH_AUDIO
    // Where a relative userData["soundFile"] resolves from. Set per play rather
    // than once: Save As moves the document, and with it the anchor.
    if (audio_) {
        audio_->setResourcePath(document_.path().empty()
                                        ? std::filesystem::path{}
                                        : document_.path().parent_path());
    }
#endif

    // Per-session, so a scene whose script does not read the keyboard keeps its shortcuts.
    scriptsPolledKeys_ = false;
    vehicleTeleopActive_ = false;


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

void EditorApp::updateVehicleTeleop(float dt) {

    vehicleDriving_ = false;
#ifdef THREEPP_EDITOR_WITH_PHYSX
    if (!isPlaying() || !physics_ || physics_->vehicleCount() == 0) return;
    // A vehicle exists in the played scene, so the plain keys are the pedals
    // whether or not one is pressed right now — see handleShortcuts.
    vehicleDriving_ = true;

    // ImGui's key state, for the reasons the script key provider documents:
    // the canvas's held-key set goes stale while a panel keeps focus. Same
    // suppression rules as the shortcuts.
    const ImGuiIO& io = ImGui::GetIO();
    const bool suppressed =
            io.WantTextInput ||
            ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) ||
            fileBrowser_.isOpen();
    const auto down = [&](ImGuiKey key) { return !suppressed && ImGui::IsKeyDown(key); };

    const bool forward = down(ImGuiKey_W) || down(ImGuiKey_UpArrow);
    const bool backward = down(ImGuiKey_S) || down(ImGuiKey_DownArrow);
    const bool left = down(ImGuiKey_A) || down(ImGuiKey_LeftArrow);
    const bool right = down(ImGuiKey_D) || down(ImGuiKey_RightArrow);
    const bool handbrake = down(ImGuiKey_Space);

    // Write controls only while keys are involved — plus the tail where the
    // released steer is still slewing back to centre — so a script driving
    // the same vehicle through its handle is not overwritten by silence.
    const bool any = forward || backward || left || right || handbrake;
    if (!any && !vehicleTeleopActive_) return;

    const float steer = (left ? 1.f : 0.f) - (right ? 1.f : 0.f);
    physics_->driveVehicles(forward, backward, steer, handbrake, dt);
    vehicleTeleopActive_ = any || physics_->vehiclesSteering();
#else
    (void) dt;
#endif
}

float EditorApp::viewYaw() {

    // The direction the view LOOKS, flattened to the ground plane, in the
    // convention every yaw in this codebase uses (yaw of +Z is zero). Read
    // from camera-to-target rather than from the camera's quaternion so a
    // panned or orbited view still answers with what is on screen.
    Vector3 forward;
    forward.subVectors(orbit_->target, viewCamera().position);
    forward.y = 0.f;
    if (forward.lengthSq() < 1e-8f) return 0.f;
    return std::atan2(forward.x, forward.z);
}

void EditorApp::updateCharacterTeleop(float dt) {

    characterDriving_ = false;
#ifdef THREEPP_EDITOR_WITH_PHYSX
    if (!isPlaying() || !characterSession_ || characterSession_->characterCount() == 0) return;
    // A character exists in the played scene, so the plain keys are the
    // controls whether or not one is pressed right now — see handleShortcuts.
    characterDriving_ = true;

    // ImGui's key state, for the reason the vehicle teleop documents: the
    // canvas's held-key set goes stale while a panel keeps focus.
    const ImGuiIO& io = ImGui::GetIO();
    const bool suppressed =
            io.WantTextInput ||
            ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) ||
            fileBrowser_.isOpen();
    const auto down = [&](ImGuiKey key) { return !suppressed && ImGui::IsKeyDown(key); };

    CharacterPlaySession::Input input;
    input.forward = (down(ImGuiKey_W) || down(ImGuiKey_UpArrow) ? 1.f : 0.f) -
                    (down(ImGuiKey_S) || down(ImGuiKey_DownArrow) ? 1.f : 0.f);
    // +strafe is the character's LEFT, which is what A asks for.
    input.strafe = (down(ImGuiKey_A) || down(ImGuiKey_LeftArrow) ? 1.f : 0.f) -
                   (down(ImGuiKey_D) || down(ImGuiKey_RightArrow) ? 1.f : 0.f);
    input.run = !suppressed && io.KeyShift;
    input.jump = down(ImGuiKey_Space);
    input.viewYaw = viewYaw();

    // Write while keys are involved, plus ONE released frame — the vehicle's
    // rule and the vehicle's reason. The released frame is what decelerates
    // the character to Idle (the session keeps applying the zero demand from
    // then on), and stopping after it is what keeps a script or a test driving
    // the same character from being overwritten by silence every frame.
    const bool any = input.forward != 0.f || input.strafe != 0.f || input.run || input.jump;
    if (!any && !characterTeleopActive_) return;
    characterSession_->drive(input);
    characterTeleopActive_ = any;
#else
    (void) dt;
#endif
}

void EditorApp::updateCharacterCamera(float dt) {

#ifdef THREEPP_EDITOR_WITH_PHYSX
    if (!isPlaying() || !characterSession_) return;
    // The user's own chase wins: Follow Selection is an explicit choice, and
    // two filters pulling the same target apart is not a camera.
    if (followSelection_) return;
    const auto* player = characterSession_->player();
    if (!player || !player->root) return;

    // Aim at the chest rather than the feet: a target on the floor puts the
    // horizon through the character's shins at any normal orbit pitch.
    Vector3 world;
    player->root->getWorldPosition(world);
    world.y += 0.7f * player->height;

    // First frame this actually takes the view over: remember where it stood,
    // so Stop can put it back (see restoreCharacterCamera). Saved lazily
    // rather than at startPlay, because a scene with no character must leave
    // the view alone — and then have nothing to restore.
    if (!characterCameraSaved_) {
        characterCameraSaved_ = true;
        characterCameraPosition_.copy(viewCamera().position);
        characterCameraTarget_.copy(orbit_->target);
    }

    // Exponential approach on the same ~90 ms constant updateFollow uses, and
    // framed the same way so the rate does not depend on the frame rate. The
    // camera is then translated by exactly the delta the target moved, which
    // is what leaves the user's orbit angle, pan and zoom untouched.
    constexpr float kTau = 0.09f;
    const float step = std::clamp(dt, 0.f, 0.1f);
    Vector3 delta;
    delta.subVectors(world, orbit_->target).multiplyScalar(1.f - std::exp(-step / kTau));
    orbit_->target.add(delta);
    viewCamera().position.add(delta);
#else
    (void) dt;
#endif
}

void EditorApp::restoreCharacterCamera() {

    if (!characterCameraSaved_) return;
    characterCameraSaved_ = false;
    viewCamera().position.copy(characterCameraPosition_);
    orbit_->target.copy(characterCameraTarget_);
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

    // The chase camera put itself over the character; Stop puts the scene back,
    // so the view goes back with it. Otherwise the camera is left staring at
    // the empty ground the character walked to before being restored to where
    // it was authored.
    restoreCharacterCamera();

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
    // Only ever hidden here: syncSoundRings owns the other direction (a config
    // whose radii are all past the cap leaves nothing to draw).
    if (soundRings_ && !visible) soundRings_->visible = false;
    // Same contract as the rings: syncJointHelper re-asserts visibility.
    if (jointHelper_ && !visible) jointHelper_->visible = false;
    // And the wheel rings, through syncVehicleHelper.
    if (vehicleHelper_ && !visible) vehicleHelper_->visible = false;
    // And the spawn slab, through syncParticleHelper. The preview FIELDS are
    // deliberately not here: they are the scene's weather, not a statement
    // about what is being edited, and Play parks them by count instead.
    if (particleHelper_ && !visible) particleHelper_->visible = false;
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
    // Driving a played vehicle owns the same plain keys for the same reason:
    // W is the throttle, and it must not also switch the gizmo to translate.
    if (vehicleDriving_ && isPlaying() && !ctrl && !alt && !io.KeySuper) return;
    // And a played character, for the same reason: W walks, and it must not
    // also switch the gizmo to translate.
    if (characterDriving_ && isPlaying() && !ctrl && !alt && !io.KeySuper) return;

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
        // GLFW hands dropped paths over as UTF-8. Constructing a path from a
        // plain std::string decodes through the ANSI code page on Windows,
        // which mangles anything past ASCII — a scan in a folder named
        // "…Beaupré" arrived as an unopenable path and got blamed for not
        // being a splat. Decode as what the bytes actually are.
        std::filesystem::path path(std::u8string(entry.begin(), entry.end()));
        const auto extension = formats::extensionOf(path);

        // A dropped FOLDER, which no extension can describe. A SOG scan is a
        // directory of chunks in its unpacked form, and dropping the folder is
        // the obvious gesture — so ask the only question that can answer it.
        // Anything else stays "ignored", as before: this is not a general
        // import-a-directory feature.
        std::error_code dirEc;
        if (std::filesystem::is_directory(path, dirEc)) {
            if (SogLoader::isSog(path)) {
                importModel(path);
            } else {
                log("ignored dropped folder " + path.filename().string() +
                    " - no meta.json or lod-meta.json, so it is not a splat scan");
            }
            continue;
        }

        if (formats::contains(formats::scenes(), extension)) {
            if (document_.dirty()) {
                pendingAction_ = PendingAction::OpenPath;
                pendingPath_ = path;
            } else {
                openScene(path);
            }
        } else if (formats::isEnvironment(extension)) {
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
        } else if (formats::isImage(extension)) {
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
