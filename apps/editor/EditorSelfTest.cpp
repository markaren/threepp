
// The editor driving itself: `--selftest` walks every seam the app has (play,
// undo across a scene swap, splines, sensors, scripts, the play-mode lock) and
// prints a PASS/FAIL line per assertion; `--screenshot` builds the spline-tube
// scenario, plays it and writes PNGs to look at. Both run on whichever backend
// the binary was started with, so `--selftest --vulkan` is a second full pass.
//
// Its own translation unit because it is half the code EditorApp used to be and
// none of the behaviour: the app and its acceptance harness now recompile
// independently, and a change to one cannot conflict with a change to the
// other. It is still built unconditionally and on purpose — an acceptance suite
// behind an off-by-default flag is one nobody runs, and this one has caught
// every regression in the editor so far.

#include "EditorApp.hpp"

#include "ImportFormats.hpp"

#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/ScriptWorkspace.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/extras/editor/SensorPlaySession.hpp"
#ifdef THREEPP_EDITOR_WITH_PHYSX
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#endif

#include "threepp/core/Clock.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/editor/GeneratorConfig.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/loaders/AssetSource.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/scenes/Scene.hpp"
#ifdef THREEPP_WITH_VULKAN
// The screenshot passes ask the renderer which way up its pixels come back.
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

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

}// namespace


// stb_image_write is already compiled into threepp (utils/StbImageWrite.cpp);
// these mirror its two entry points rather than adding an include path for the
// one function this file wants.
extern "C" {
    int stbi_write_png(char const* filename, int w, int h, int comp, const void* data, int stride_in_bytes);
    void stbi_flip_vertically_on_write(int flag);
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
    // Marker icons too. A hemisphere light and an ambient light both sit at the
    // origin, so their billboards hang in mid-air in the middle of the arena —
    // useful when you are authoring, noise when you are judging a picture. The
    // point cloud stays: it is what the scene is DOING, not furniture.
    if (markers_) markers_->visible = false;
    // And the selection furniture, for the same reason: a document that opens
    // with something selected (userData["editorFollow"]) would otherwise be
    // photographed through a yellow box and a set of transform handles. Select
    // mode is how the toolbar itself spells "no gizmo", and the outline nodes
    // are only rebuilt when the selection changes, which it does not here.
    gizmoMode_ = "select";
    applyGizmoMode();
    if (selectionBox_) selectionBox_->visible = false;
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
    if (!options_.keys.empty()) {
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

int EditorApp::runScreenshot() {

    // A document of its own to photograph beats the built-in scenario. The
    // scenario is the DEFAULT, not the contract: `--screenshot=x.png` with
    // nothing else on the line still builds the tubes and writes every sibling
    // view, which is what the road/spline acceptance passes ask for.
    if (!options_.example.empty() ||
        (!options_.openOnStart.empty() && options_.openOnStart.extension() == ".json")) {
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

    return wrote ? 0 : 1;
}

int EditorApp::runSelfTest() {

    Clock clock;
    int failed = 0;

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

    // Play is a viewer, not an editor. Every mutating entry point is driven
    // here directly rather than through its menu item, because greying a menu
    // is not the guarantee — the refusal in the operation is, and the shortcut
    // handler and file-drop path reach these with no menu in between.
    {
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
    {
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

    // Follow Selection: the orbit target chases what is selected, the camera
    // rides along keeping the user's angle and distance, and it keeps doing it
    // while a Play session owns the transform. Driven with a hand-moved object
    // rather than a simulated one so the assertion is about the CAMERA and not
    // about whatever a solver did this frame.
    {
        auto* subject = document_.scene().getObjectByName("Box");
        check(subject != nullptr, "Box available for the follow drive");

        // The approach is exponential and the frame rate is the machine's, so
        // convergence is waited for rather than assumed after a frame count.
        Vector3 world;
        const auto settle = [&](Object3D& node, float tolerance) {
            for (int i = 0; i < 900; ++i) {
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

    // A texture dialog outliving its target. The picker records (uuid, slot)
    // rather than a Material*, so confirming after the graph has been rebuilt
    // resolves against the new one — or, when the object is gone, does nothing
    // at all. Before this it dereferenced a pointer into the destroyed scene.
    {
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
    {
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
    {
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
    {
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
    {
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
    {
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
    {
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
    {
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
    }

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
                    stepFixed(30);
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
                    stepFixed(30);
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
                                    "    def start(self, obj, scene):\n"
                                    "        self.door = threepp.editor.script_from_object(\n"
                                    "            scene.get_object_by_name('Box'))\n"
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
    {
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
        check(budget > 0 && importError_.empty(), "the millimetre urdf imported");

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
    {
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

    // --- material shadowSide ----------------------------------------------
    // The inspector's answer to a Double-sided material self-shadowing into a
    // moire. It is only worth exposing if it survives being written down, and
    // Play is the harshest test of that: the snapshot goes through the same
    // exporter/loader pair a saved scene does.
    {
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
    {
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
            check(after.size() == before.size() && moved > (n / 3) / 500,
                  "a material edit shows up without a selection change");
        }
    }

    // --- orthographic axis views ------------------------------------------
    // The screenshots are what say the views look right; these say the state
    // machine behind them holds — that the projection swaps without moving what
    // is framed, that picking still works through the ortho unprojection, and
    // that the label follows the camera rather than the last button pressed.
    {
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
    {
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
    {
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
    {
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
            auto eye = ObjectFactory::createPrimitive(Primitive::Sphere, document_.scene());
            eye->name = "Depth Cam";
            eye->position.set(-2.f, 4.f, 2.f);
            // -Z is the viewing direction; -90 degrees about X points it at -Y.
            eye->rotation.x = -1.5707963f;
            eye->scale.set(0.12f, 0.12f, 0.12f);

            SensorConfig depth;
            depth.enabled = true;
            depth.type = SensorConfig::Type::Depth;
            depth.width = 96;
            depth.height = 72;
            depth.fovY = 60.f;
            depth.rateHz = 0.f;
            depth.seed = 777;
            depth.rangeStddev = 0.005f;
            depth.nearPlane = 0.2f;
            depth.farPlane = 20.f;
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
        // agree far inside the noise sigma instead. A seed that failed to
        // replay would deviate by ~sigma (0.02 m), 20x this bound.
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

            // One step is one scan for an ungated sensor. Hash the FIRST scan and
            // nothing later: every scan draws from the range-noise stream, so
            // scan N is only comparable with scan N of another run, and how many
            // frames the IMU wait below takes is a property of the machine.
            stepFixed();
            const auto* lidarEntry = entryFor(lidarUuid);
            check(lidarEntry != nullptr && lidarEntry->scans == 1,
                  "the first frame of play is exactly one scan");
            const std::size_t scanHash = lidarEntry ? hashCloud(lidarEntry->returns) : 0;
            const std::size_t scanPoints = lidarEntry ? lidarEntry->returns.size() : 0;

            const auto* depthEntry = entryFor(depthUuid);
            check(depthEntry != nullptr && depthEntry->scans == 1,
                  "the depth camera scanned on the same frame");
            const std::size_t depthHash = depthEntry ? hashPoints(depthEntry->cloud) : 0;
            const std::size_t depthPoints = depthEntry ? depthEntry->cloud.size() : 0;
            check(depthPoints > 0, "and its cloud is not empty");

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
                if (lidarEntry) firstReturns = lidarEntry->returns;
                if (depthEntry) firstDepthCloud = depthEntry->cloud;
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
                    const float lidarDev = lidarEntry
                            ? maxDeviation(lidarEntry->returns, firstReturns, returnPos) : 1e9f;
                    const float depthDev = depthEntry
                            ? maxDeviation(depthEntry->cloud, firstDepthCloud, pointPos) : 1e9f;
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

    // ------------------------------------------------ scene-root userData
    //
    // Whether an authoring rule can live ON the scene rather than on a node
    // depends entirely on this: the root is serialised as an object of type
    // "Scene", so its userData should round-trip like any other node's. Play
    // serialises and Stop reloads, which is the same path a save and open take,
    // so this answers the question for both. Multi-line values are the case
    // that matters — an inline script is newlines and quotes, not a scalar.
    {
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
    {
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
    {
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
    {
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
                // controller that has stopped controlling.
                check(world.y > 1.4f && world.y < 3.2f, "and holds its hover band");
                check(std::abs(world.y - before) < 1.5f, "without drifting off its start height");
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

    std::cout << "[selftest] " << (failed == 0 ? "ALL PASS" : "FAILED") << std::endl;
    return failed == 0 ? 0 : 1;
}
