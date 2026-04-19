/// YOLOv8n WebGPU Object Detection Inference Example
///
/// Usage: yolov8_inference [image_path] [weights_path]
///   image_path   – JPEG/PNG input image (default: data/textures/uv_grid_opengl.jpg)
///   weights_path – .bin file exported by scripts/export_yolov8n_weights.py
///                  (default: yolov8n.bin next to the executable)
///
/// Workflow:
///   1. Load image from disk via ImageLoader (RGBA, stb_image)
///   2. Load pre-trained weights from binary file into GPU buffers
///   3. Run full YOLOv8n forward pass on GPU using WGSL compute shaders
///   4. Display the input image with detection bounding box overlays

#include "threepp/threepp.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/core/BufferAttribute.hpp"

#include "yolov8/YoloV8n.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

// COCO class names (80 classes, 0-indexed)
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

/// Build a LineSegments object drawing 4 edges of a 2D bounding box.
/// Coordinates are in 2D (Z=0). The box is axis-aligned.
static std::shared_ptr<LineSegments> makeBoxLines(
        float x1, float y1, float x2, float y2, const Color& col) {

    // 4 edges × 2 vertices each = 8 vertices
    std::vector<float> positions = {
        x1, y1, 0,  x2, y1, 0,   // top edge
        x2, y1, 0,  x2, y2, 0,   // right edge
        x2, y2, 0,  x1, y2, 0,   // bottom edge
        x1, y2, 0,  x1, y1, 0    // left edge
    };

    auto geo = BufferGeometry::create();
    geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto mat = LineBasicMaterial::create();
    mat->color = col;
    mat->depthTest = false;

    return LineSegments::create(geo, mat);
}

int main(int argc, char** argv) {
    std::string imgPath;
    std::string weightsPath;

    if (argc > 1) imgPath     = argv[1];
    if (argc > 2) weightsPath = argv[2];

    if (imgPath.empty() || weightsPath.empty()) {
        std::cerr << "Usage: " << argv[0] << " imgPath weightsPath\n";
    }

    // ----------------------------------------------------------------
    // Load input image on CPU
    // ----------------------------------------------------------------
    ImageLoader imgLoader;
    auto imgOpt = imgLoader.load(imgPath, 4, false);  // RGBA, no flip
    if (!imgOpt) {
        std::cerr << "ERROR: Could not load image '" << imgPath << "'\n";
        return 1;
    }
    auto& img = *imgOpt;
    std::cout << "Loaded image: " << img.width << "×" << img.height << "\n";

    // ----------------------------------------------------------------
    // Setup WebGPU window
    // ----------------------------------------------------------------
    const int WIN = 900;
    Canvas::Parameters params;
    params.title("YOLOv8n WebGPU Inference")
          .size(WIN, WIN);

    Canvas canvas(params);
    WgpuRenderer renderer(canvas);
    renderer.setClearColor(Color(0x1a1a2e));

    // ----------------------------------------------------------------
    // Run inference (before the render loop, can take several seconds
    // on first run due to WGSL shader compilation)
    // ----------------------------------------------------------------
    std::cout << "Loading weights from '" << weightsPath << "' ...\n";

    using clk = std::chrono::steady_clock;
    auto ms = [](clk::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    std::vector<yolo::Detection> detections;
    {
        const int WARMUP = 3;
        const int RUNS   = 50;

        yolo::YoloV8n model(renderer);
        model.loadWeights(weightsPath);

        auto& rgba = img.data<unsigned char>();
        std::filesystem::path imgFs(imgPath);
        double imgKB = std::filesystem::exists(imgFs)
            ? double(std::filesystem::file_size(imgFs)) / 1024.0 : 0.0;

        std::string bar(52, '=');
        std::cout << "\n" << bar << "\n"
                  << "  YOLOv8n WebGPU Inference Benchmark\n"
                  << bar << "\n"
                  << "  Image  : " << imgFs.filename().string()
                  << " (" << std::fixed << std::setprecision(1) << imgKB << " KB)\n"
                  << "  Device : WebGPU (wgpu-native)\n"
                  << "  Warmup : " << WARMUP << " runs\n"
                  << "  Runs   : " << RUNS << "\n"
                  << bar << "\n\n";

        std::cout << "Warming up (" << WARMUP << " runs)..." << std::flush;
        for (int i = 0; i < WARMUP; ++i)
            detections = model.infer(rgba.data(), int(img.width), int(img.height));
        std::cout << " done.\n\n";

        std::vector<double> times_ms;
        times_ms.reserve(RUNS);
        for (int i = 0; i < RUNS; ++i) {
            auto ta = clk::now();
            detections = model.infer(rgba.data(), int(img.width), int(img.height));
            auto tb = clk::now();
            double dt = ms(tb - ta);
            times_ms.push_back(dt);

            std::cout << "  Run " << std::setw(2) << (i + 1) << "/" << RUNS << "  "
                      << std::fixed << std::setprecision(2) << std::setw(7) << dt
                      << " ms   detections: " << detections.size() << "\n";
            for (auto& d : detections) {
                const char* name = (d.cls_id >= 0 && d.cls_id < 80) ? kCocoNames[d.cls_id] : "unknown";
                std::cout << "             > " << std::left << std::setw(20) << name << std::right
                          << "  conf: " << std::fixed << std::setprecision(3) << d.conf << "\n";
            }
        }

        // Stats
        auto sorted = times_ms;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0.0;
        for (double v : times_ms) sum += v;
        double mean   = sum / RUNS;
        double median = (RUNS % 2)
            ? sorted[RUNS / 2]
            : 0.5 * (sorted[RUNS / 2 - 1] + sorted[RUNS / 2]);
        double sq = 0.0;
        for (double v : times_ms) sq += (v - mean) * (v - mean);
        double stdev = std::sqrt(sq / (RUNS - 1));   // sample stdev
        double mn = sorted.front(), mx = sorted.back();

        std::string sep(52, '-');
        std::cout << "\n" << bar << "\n"
                  << "  Results (" << RUNS << " runs, ms)\n"
                  << sep << "\n"
                  << "  Mean   : " << std::fixed << std::setprecision(2) << std::setw(8) << mean
                  << " ms   (" << std::setw(6) << std::setprecision(1) << (1000.0 / mean) << " FPS)\n"
                  << "  Median : " << std::setprecision(2) << std::setw(8) << median
                  << " ms   (" << std::setw(6) << std::setprecision(1) << (1000.0 / median) << " FPS)\n"
                  << "  Std    : " << std::setprecision(2) << std::setw(8) << stdev << " ms\n"
                  << "  Min    : " << std::setw(8) << mn << " ms\n"
                  << "  Max    : " << std::setw(8) << mx << " ms\n"
                  << bar << "\n\n";
    }

    // ----------------------------------------------------------------
    // Scene: display image + bounding boxes with an orthographic camera
    // Camera: (0,0) = top-left, (640,640) = bottom-right (matches YOLO space)
    // ----------------------------------------------------------------
    auto scene = Scene::create();

    // Orthographic camera: (left, right, top, bottom, near, far).
    // top=640 means world y=640 maps to screen top, bottom=0 means y=0 = screen bottom.
    // Image loaded with flipY=true so image top appears at screen top.
    // YOLO box y coords (image space, 0=top) must be flipped: screen_y = 640 - image_y.
    auto camera = OrthographicCamera::create(0, 640, 640, 0, -1, 1);

    // Upload image as texture (resize to 640×640 display by placing it on a plane)
    // Use TextureLoader to create a displayable texture
    TextureLoader texLoader;
    auto displayTex = texLoader.load(imgPath, true);  // flipY=true: correct OpenGL orientation

    // Plane covering [0,640] × [0,640]: centre at (320,320), size 640×640
    auto planeGeo = PlaneGeometry::create(640, 640);
    auto planeMat = MeshBasicMaterial::create();
    planeMat->map = displayTex;
    planeMat->side = Side::Double;

    auto quad = Mesh::create(planeGeo, planeMat);
    quad->position.set(320, 320, 0);
    // PlaneGeometry is in XY, faces -Z by default (normal +Z).
    // No rotation needed for ortho camera looking at -Z.
    scene->add(quad);

    // Detection colour palette (one colour per class mod 6)
    static const Color kPalette[] = {
        Color(0xff3333), Color(0x33ff33), Color(0x3333ff),
        Color(0xffff33), Color(0xff33ff), Color(0x33ffff)
    };

    // Detections are in original-image pixel space; map them onto the 640×640 display plane.
    const float sx = 640.f / float(img.width);
    const float sy = 640.f / float(img.height);
    for (auto& d : detections) {
        float x1 = std::max(0.f, d.x1 * sx), y1 = std::max(0.f, d.y1 * sy);
        float x2 = std::min(640.f, d.x2 * sx), y2 = std::min(640.f, d.y2 * sy);

        // Flip y: image y=0 is top → world y=640 (screen top)
        float sy1 = 640.f - y2;
        float sy2 = 640.f - y1;

        Color col = kPalette[d.cls_id % 6];
        auto box = makeBoxLines(x1, sy1, x2, sy2, col);
        scene->add(box);
    }

    // ----------------------------------------------------------------
    // Render loop
    // ----------------------------------------------------------------
    canvas.animate([&] {
        renderer.render(*scene, *camera);
    });

    return 0;
}
