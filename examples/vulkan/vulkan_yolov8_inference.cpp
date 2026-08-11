/// YOLOv8n Vulkan Object Detection Inference Example
///
/// Usage: vulkan_yolov8_inference [image_path] [weights_path]
///   image_path   – JPEG/PNG input image
///   weights_path – .bin file exported by scripts/export_yolov8n_weights.py
///
///
/// Runs the full
/// YOLOv8n forward pass on the Vulkan compute pipeline, then displays the
/// input image with detection bounding-box overlays.

#include "threepp/threepp.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/materials/SpriteMaterial.hpp"

#include "capture_util.hpp"
#include "coco_labels.hpp"
#include "utility/DetectionOverlay.hpp"
#include "yolov8/YoloV8nVk.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

// The label table indexes the model's class dimension directly, so a mismatch
// between the two would mislabel every detection with no other symptom.
static_assert(std::ssize(coco::kCoco80) == yolo::YoloV8nVk::NUM_CLASSES,
              "COCO-80 label table does not match YoloV8nVk::NUM_CLASSES");

int main(int argc, char** argv) {
    std::string imgPath, weightsPath, shotPath;
    std::vector<std::string> args(argv + 1, argv + argc);
    // --shot <png>: headless capture — render a few viewer frames, save, exit.
    for (size_t i = 0; i + 1 < args.size();) {
        if (args[i] == "--shot") {
            shotPath = args[i + 1];
            args.erase(args.begin() + i, args.begin() + i + 2);
        } else {
            ++i;
        }
    }
    if (args.size() > 0) imgPath     = args[0];
    if (args.size() > 1) weightsPath = args[1];
    if (imgPath.empty() || weightsPath.empty()) {
        std::cerr << "Usage: " << argv[0] << " imgPath weightsPath [--shot out.png]\n";
        return 1;
    }

    ImageLoader imgLoader;
    auto imgOpt = imgLoader.load(imgPath, 4, false);// RGBA, no flip
    if (!imgOpt) {
        std::cerr << "ERROR: Could not load image '" << imgPath << "'\n";
        return 1;
    }
    auto& img = *imgOpt;
    std::cout << "Loaded image: " << img.width() << "x" << img.height() << "\n";

    constexpr int WIN = 900;
    Canvas::Parameters params;
    params.title("YOLOv8n Vulkan Inference").size(WIN, WIN);

    Canvas canvas(params);
    VulkanRenderer renderer(canvas);
    renderer.setClearColor(Color(0x1a1a2e));

    using clk = std::chrono::steady_clock;
    auto ms = [](clk::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };

    std::vector<yolo::Detection> detections;
    {
        constexpr int WARMUP = 2;
        constexpr int RUNS   = 10;

        yolo::YoloV8nVk model(renderer);
        std::cout << "Loading weights from '" << weightsPath << "' ...\n";
        model.loadWeights(weightsPath);

        auto& rgba = img.data<unsigned char>();
        std::string bar(52, '=');
        std::cout << "\n" << bar << "\n"
                  << "  YOLOv8n Vulkan Inference Benchmark\n"
                  << bar << "\n"
                  << "  Device : Vulkan\n"
                  << "  Warmup : " << WARMUP << " runs\n"
                  << "  Runs   : " << RUNS << "\n"
                  << bar << "\n\n";

        std::cout << "Warming up..." << std::flush;
        for (int i = 0; i < WARMUP; ++i)
            detections = model.infer(rgba.data(), static_cast<int>(img.width()), static_cast<int>(img.height()));
        std::cout << " done.\n\n";

        std::vector<double> times_ms;
        times_ms.reserve(RUNS);
        for (int i = 0; i < RUNS; ++i) {
            auto ta = clk::now();
            detections = model.infer(rgba.data(), static_cast<int>(img.width()), static_cast<int>(img.height()));
            auto tb = clk::now();
            double dt = ms(tb - ta);
            times_ms.push_back(dt);
            std::cout << "  Run " << std::setw(2) << (i + 1) << "/" << RUNS << "  "
                      << std::fixed << std::setprecision(2) << std::setw(8) << dt
                      << " ms   detections: " << detections.size() << "\n";
            for (auto& d : detections) {
                std::cout << "             > " << std::left << std::setw(20) << coco::name80(d.cls_id) << std::right
                          << "  conf: " << std::fixed << std::setprecision(3) << d.conf << "\n";
            }
        }

        auto sorted = times_ms;
        std::ranges::sort(sorted);
        double sum = 0.0;
        for (double v : times_ms) sum += v;
        double mean = sum / RUNS;
        std::string sep(52, '-');
        std::cout << "\n" << bar << "\n"
                  << "  Mean   : " << std::fixed << std::setprecision(2) << mean
                  << " ms   (" << std::setprecision(1) << (1000.0 / mean) << " FPS)\n"
                  << "  Min    : " << std::setprecision(2) << sorted.front() << " ms\n"
                  << "  Max    : " << sorted.back() << " ms\n"
                  << bar << "\n\n";
    }

    // ---- Scene: image Sprite + detection-box LineSegments under an ortho camera ----
    // VulkanRenderer draws this through its ortho HUD overlay: the Sprite samples
    // its texture map, and the boxes render on top as LineSegments (the ortho line
    // overlay wired into the renderer for exactly this). Both the sprite and the
    // boxes live in the same 640x640 world space, so they align.
    auto scene = Scene::create();
    auto camera = OrthographicCamera::create(0, 640, 640, 0, -1, 1);

    TextureLoader texLoader;
    auto displayTex = texLoader.load(imgPath, ColorSpace::Linear, true);

    auto spriteMat = SpriteMaterial::create();
    spriteMat->map = displayTex;
    auto imageSprite = Sprite::create(spriteMat);
    imageSprite->scale.set(640.f, 640.f, 1.f);   // fill the 640x640 ortho view
    imageSprite->position.set(320.f, 320.f, 0.f);// centered -> spans [0,640]
    scene->add(imageSprite);

    FontLoader fontLoader;
    const Font font = fontLoader.defaultFont();

    const float sx = 640.f / static_cast<float>(img.width());
    const float sy = 640.f / static_cast<float>(img.height());
    for (auto& d : detections) {
        float x1 = std::max(0.f, d.x1 * sx), y1 = std::max(0.f, d.y1 * sy);
        float x2 = std::min(640.f, d.x2 * sx), y2 = std::min(640.f, d.y2 * sy);
        float sy1 = 640.f - y2;
        float sy2 = 640.f - y1;
        const Color& col = detviz::classColor(d.cls_id);
        scene->add(detviz::makeBoxLines(x1, sy1, x2, sy2, col));
        scene->add(detviz::makeLabel(font, detviz::labelText(coco::name80(d.cls_id), d.conf), col, x1, sy2));
    }

    // A few frames so the swapchain holds the finished overlay, then write the
    // path the caller asked for verbatim (not an aaa_caps name) and exit.
    capture::Shot shot;
    shot.name = shotPath;
    shot.frames = 5;
    canvas.animate([&] {
        renderer.render(*scene, *camera);
        if (shot.ready()) capture::finishShotAtPath(renderer, shot.name);
    });

    return 0;
}
