// SYNTHETIC PERCEPTION LOOP — the threepp thesis demo in one window:
//
//   render → read back → detect → overlay, every frame, one GPU, one process.
//
// The Vulkan deferred path renders an ANIMATED scene (two glTF soldiers — one
// walking a patrol line, one running a circle — among props under an HDR sky),
// the rendered frame is read back and fed to the Vulkan-compute RF-DETR-Nano
// port (examples/vulkan/rfdetr), and the detections are drawn as a classic
// detector overlay (box + class/confidence tag) in a HUD pass over the live
// render. No .jpg anywhere: the detector has never seen anything but threepp
// pixels.
//
// Usage: vulkan_synthetic_inference [weights] [--shot out.png]
//   weights — export_rfdetr_weights.py output
//             (default: <project>/scripts/rfdetr-nano.weights)
//   --shot  — headless validation: run ~150 frames, print detections,
//             save the overlaid frame, exit.

#include "threepp/threepp.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "rfdetr/RfDetrVk.hpp"
#include "utility/DetectionOverlay.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // COCO 91-id class names (RF-DETR uses the original COCO ids with gaps).
    const char* kCoco91[91] = {
            "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
            "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
            "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"};

    // Load one soldier instance and start the named clip. The glb is loaded
    // once per instance — skinned meshes can't share a skeleton between
    // independently-posed characters.
    struct Actor {
        std::shared_ptr<Group> root;
        std::unique_ptr<AnimationMixer> mixer;
    };

    Actor loadSoldier(GLTFLoader& loader, const std::string& path, const std::string& clipName) {
        Actor a;
        auto result = loader.load(path);
        if (!result || !result->scene) {
            std::cerr << "Failed to load " << path << "\n";
            return a;
        }
        a.root = result->scene;
        a.root->animations = result->animations;
        a.mixer = std::make_unique<AnimationMixer>(*a.root);
        std::shared_ptr<AnimationClip> clip;
        for (auto& c : a.root->animations) {
            if (c->name() == clipName) {
                clip = c;
                break;
            }
        }
        if (!clip && !a.root->animations.empty()) clip = a.root->animations.front();
        if (clip) a.mixer->clipAction(clip)->play();
        return a;
    }

}// namespace

int main(int argc, char** argv) {
    std::string weightsPath = std::string(PROJECT_FOLDER) + "/scripts/rfdetr-nano.weights";
    std::string shotPath;
    {
        std::vector<std::string> args(argv + 1, argv + argc);
        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "--shot" && i + 1 < args.size()) {
                shotPath = args[++i];
            } else {
                weightsPath = args[i];
            }
        }
    }
    if (!std::filesystem::exists(weightsPath)) {
        std::cerr << "Weights not found: " << weightsPath
                  << "\nUsage: " << argv[0] << " [weights] [--shot out.png]\n";
        return 1;
    }

    Canvas canvas("Synthetic perception - render+detect", {{"vsync", false}, {"size", WindowSize{1280, 720}}});
    VulkanRenderer renderer(canvas);
    renderer.outputColorSpace = ColorSpace::sRGB;
    renderer.toneMapping = ToneMapping::Neutral;
    renderer.setDenoise(true);
    renderer.setRestirDIEnabled(true);
    renderer.setDeferredAO(true);

    // ---- Scene ----
    Scene scene;
    RGBELoader rgbe;
    if (auto env = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = env;
        scene.environment = env;
    }
    auto sun = DirectionalLight::create(Color(1.f, 0.95f, 0.85f), 2.5f);
    sun->position.set(30.f, 50.f, 20.f);
    scene.add(sun);

    auto ground = Mesh::create(BoxGeometry::create(60.f, 0.5f, 60.f),
                               MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                       .color(Color(0.45f, 0.45f, 0.42f))
                                       .roughness(0.85f)
                                       .metalness(0.f)));
    ground->position.y = -0.25f;
    scene.add(ground);

    // Props: pillars + crates so the detector sees clutter, occlusion and
    // shadows — not a person on an empty plane.
    {
        auto red = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.7f, 0.15f, 0.1f)).roughness(0.6f).metalness(0.f));
        auto blue = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.15f, 0.3f, 0.7f)).roughness(0.5f).metalness(0.f));
        for (float x : {-8.f, -3.f, 5.f, 9.f}) {
            auto pillar = Mesh::create(BoxGeometry::create(0.8f, 4.f, 0.8f), red);
            pillar->position.set(x, 2.f, -10.f);
            scene.add(pillar);
        }
        for (int i = 0; i < 3; ++i) {
            auto crate = Mesh::create(BoxGeometry::create(1.2f, 1.2f, 1.2f), blue);
            crate->position.set(-5.f + i * 4.5f, 0.6f, -6.5f);
            scene.add(crate);
        }
    }

    // ---- Actors ----
    GLTFLoader loader;
    const std::string soldierPath = std::string(DATA_FOLDER) + "/models/gltf/Soldier.glb";
    Actor walker = loadSoldier(loader, soldierPath, "Walk");
    Actor runner = loadSoldier(loader, soldierPath, "Run");
    if (!walker.root || !runner.root) return 1;
    scene.add(walker.root);
    scene.add(runner.root);

    // ---- Camera (static elevated observer — synthetic CCTV) ----
    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 200.f);
    camera.position.set(0.f, 2.6f, 6.f);// close-in observer: soldiers ~150-300 px tall
    camera.lookAt(Vector3(0.f, 1.2f, -3.f));
    camera.updateMatrixWorld();

    // ---- Detector ----
    rfdetr::RfDetrVk model(renderer);
    std::cout << "Loading RF-DETR-Nano weights: " << weightsPath << "\n";
    model.loadWeights(weightsPath);

    // ---- Detection HUD (ortho overlay over the live render) ----
    // EVERYTHING below works in surface PIXELS (framebufferSize), not logical
    // window units: readRGBPixels() returns the swapchain image, which is
    // larger than canvas.size() under OS display scaling — mixing the two
    // spaces puts every box in the wrong place.
    Scene ui;
    OrthographicCamera uiCam(0.f, 1.f, 1.f, 0.f, -1.f, 10.f);
    uiCam.position.z = 1.f;
    FontLoader fontLoader;
    const Font font = fontLoader.defaultFont();
    std::shared_ptr<Group> overlay = Group::create();
    ui.add(overlay);

    std::vector<unsigned char> rgba;// reused RGB→RGBA scratch
    Clock clock;
    int frame = 0;
    float inferMsAvg = 0.f;
    int inferCount = 0;

    canvas.animate([&] {
        const float dt = clock.getDelta();
        const float t = clock.elapsedTime;

        // Walker: patrol line x ∈ [-6, 6] at z = -4, about-face at the ends.
        // (The Soldier model's authored forward is -Z: Ry(θ) faces it toward
        // (-sin θ, 0, -cos θ), so +x motion needs θ = -π/2.)
        {
            const float speed = 1.5f, span = 6.f;
            const float u = std::fmod(t * speed, 4.f * span);// there (2span) + back (2span)
            const bool fwd = u < 2.f * span;
            const float x = fwd ? (u - span) : (3.f * span - u);
            walker.root->position.set(x, 0.f, -4.f);
            walker.root->rotation.y = fwd ? -math::PI / 2.f : math::PI / 2.f;
            walker.mixer->update(dt);
        }
        // Runner: circle r=5 around (0, -3), facing the tangent (-Z-forward model).
        {
            const float w = 4.f / 5.f;// speed 4 m/s on r=5
            const float a = t * w;
            runner.root->position.set(std::sin(a) * 5.f, 0.f, -3.f + std::cos(a) * 5.f);
            runner.root->rotation.y = a - math::PI / 2.f;// velocity dir (cos a, -sin a)
            runner.mixer->update(dt);
        }

        renderer.render(scene, camera);

        // Detect every 3rd frame (RF-DETR ~25 ms; the readback serializes
        // CPU/GPU, so per-frame inference would halve the render rate).
        if (frame % 3 == 0 && frame > 5) {
            const auto fb = renderer.framebufferSize();// surface PIXELS (≠ size() under DPI scaling)
            const float H = static_cast<float>(fb.height());
            const auto rgb = renderer.readRGBPixels();
            rgba.resize(rgb.size() / 3 * 4);
            for (size_t i = 0, j = 0; i < rgb.size(); i += 3, j += 4) {
                rgba[j] = rgb[i];
                rgba[j + 1] = rgb[i + 1];
                rgba[j + 2] = rgb[i + 2];
                rgba[j + 3] = 255;
            }
            const auto t0 = std::chrono::steady_clock::now();
            auto dets = model.infer(rgba.data(), fb.width(), fb.height(), 0.5f);
            const float ms = std::chrono::duration<float, std::milli>(
                                     std::chrono::steady_clock::now() - t0)
                                     .count();
            inferMsAvg += ms;
            ++inferCount;

            // HUD ortho tracks the surface pixel size (also handles resize).
            uiCam.left = 0.f;
            uiCam.right = static_cast<float>(fb.width());
            uiCam.top = H;
            uiCam.bottom = 0.f;
            uiCam.updateProjectionMatrix();
            uiCam.updateMatrixWorld();

            // Rebuild the overlay. Detections are in surface-pixel space
            // (y down); the HUD ortho is y-up → flip.
            ui.remove(*overlay);
            overlay = Group::create();
            for (const auto& d : dets) {
                const Color& col = detviz::kPalette[d.classId % 6];
                const float y1 = H - d.y2, y2 = H - d.y1;
                overlay->add(detviz::makeBoxLines(d.x1, y1, d.x2, y2, col));
                const char* name = (d.classId >= 0 && d.classId < 91) ? kCoco91[d.classId] : "?";
                overlay->add(detviz::makeLabel(font, detviz::labelText(name, d.confidence), col, d.x1, y2, H));
            }
            ui.add(overlay);

            if (inferCount % 30 == 1 || !shotPath.empty()) {
                std::printf("[detect %.1f ms] %zu det(s):", ms, dets.size());
                for (const auto& d : dets)
                    std::printf("  %s %.2f (%.0f,%.0f - %.0f,%.0f)",
                                (d.classId >= 0 && d.classId < 91) ? kCoco91[d.classId] : "?",
                                d.confidence, d.x1, d.y1, d.x2, d.y2);
                std::printf("\n");
            }
        }

        renderer.render(ui, uiCam);// HUD overlay pass

        if (!shotPath.empty() && frame == 150) {
            renderer.writeFramebuffer(shotPath);
            std::printf("wrote %s (avg detect %.1f ms over %d runs)\n",
                        shotPath.c_str(), inferMsAvg / float(inferCount), inferCount);
            std::exit(0);
        }
        ++frame;
    });
    return 0;
}
