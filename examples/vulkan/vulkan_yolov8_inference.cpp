/// YOLOv8n Vulkan Object Detection Inference Example
///
/// Usage: vulkan_yolov8_inference [image_path] [weights_path]
///   image_path   – JPEG/PNG input image
///   weights_path – .bin file exported by scripts/export_yolov8n_weights.py
///                  (same format as the WGPU yolov8_inference example)
///
/// Direct Vulkan port of examples/wgpu/yolov8_inference.cpp: runs the full
/// YOLOv8n forward pass on the Vulkan compute pipeline, then displays the
/// input image with detection bounding-box overlays.

#include "threepp/threepp.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/core/BufferAttribute.hpp"

#include "yolov8/YoloV8nVk.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

static const char* kCocoNames[80] = {
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

static std::shared_ptr<LineSegments> makeBoxLines(
        float x1, float y1, float x2, float y2, const Color& col) {
    std::vector<float> positions = {
        x1, y1, 0,  x2, y1, 0,
        x2, y1, 0,  x2, y2, 0,
        x2, y2, 0,  x1, y2, 0,
        x1, y2, 0,  x1, y1, 0
    };
    auto geo = BufferGeometry::create();
    geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    auto mat = LineBasicMaterial::create();
    mat->color = col;
    mat->depthTest = false;
    return LineSegments::create(geo, mat);
}

int main(int argc, char** argv) {
    std::string imgPath, weightsPath;
    if (argc > 1) imgPath     = argv[1];
    if (argc > 2) weightsPath = argv[2];
    if (imgPath.empty() || weightsPath.empty()) {
        std::cerr << "Usage: " << argv[0] << " imgPath weightsPath\n";
        return 1;
    }

    ImageLoader imgLoader;
    auto imgOpt = imgLoader.load(imgPath, 4, false);// RGBA, no flip
    if (!imgOpt) {
        std::cerr << "ERROR: Could not load image '" << imgPath << "'\n";
        return 1;
    }
    auto& img = *imgOpt;
    std::cout << "Loaded image: " << img.width << "x" << img.height << "\n";

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
            detections = model.infer(rgba.data(), static_cast<int>(img.width), static_cast<int>(img.height));
        std::cout << " done.\n\n";

        std::vector<double> times_ms;
        times_ms.reserve(RUNS);
        for (int i = 0; i < RUNS; ++i) {
            auto ta = clk::now();
            detections = model.infer(rgba.data(), static_cast<int>(img.width), static_cast<int>(img.height));
            auto tb = clk::now();
            double dt = ms(tb - ta);
            times_ms.push_back(dt);
            std::cout << "  Run " << std::setw(2) << (i + 1) << "/" << RUNS << "  "
                      << std::fixed << std::setprecision(2) << std::setw(8) << dt
                      << " ms   detections: " << detections.size() << "\n";
            for (auto& d : detections) {
                const char* name = (d.cls_id >= 0 && d.cls_id < 80) ? kCocoNames[d.cls_id] : "unknown";
                std::cout << "             > " << std::left << std::setw(20) << name << std::right
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

    static const Color kPalette[] = {
        Color(0xff3333), Color(0x33ff33), Color(0x3333ff),
        Color(0xffff33), Color(0xff33ff), Color(0x33ffff)
    };

    const float sx = 640.f / static_cast<float>(img.width);
    const float sy = 640.f / static_cast<float>(img.height);
    for (auto& d : detections) {
        float x1 = std::max(0.f, d.x1 * sx), y1 = std::max(0.f, d.y1 * sy);
        float x2 = std::min(640.f, d.x2 * sx), y2 = std::min(640.f, d.y2 * sy);
        float sy1 = 640.f - y2;
        float sy2 = 640.f - y1;
        Color col = kPalette[d.cls_id % 6];
        scene->add(makeBoxLines(x1, sy1, x2, sy2, col));
    }

    canvas.animate([&] {
        renderer.render(*scene, *camera);
    });

    return 0;
}
