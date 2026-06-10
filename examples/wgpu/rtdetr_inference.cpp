/// RT-DETR-L WebGPU Object Detection Inference
///
/// Usage: rtdetr_inference [image_path]
///   image_path   - JPEG/PNG input image (default: data/bus.jpg)
///
/// Workflow:
///   1. Load image from disk via ImageLoader (RGBA, stb_image)
///   2. Load pre-trained weights from binary file into GPU buffers
///   3. Run full RT-DETR-L forward pass on GPU using WGSL compute shaders
///   4. Display the input image with detection bounding box overlays

#include "threepp/threepp.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"

#include "utility/DetectionOverlay.hpp"
#include "rtdetr/RtDetr.hpp"

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


int main(int argc, char** argv) {
    std::string imgPath;
    std::string weightsPath;

    if (argc > 1) imgPath     = argv[1];
    if (argc > 2) weightsPath = argv[2];

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
    std::cout << "Loaded image: " << img.width << "x" << img.height << "\n";

    // ----------------------------------------------------------------
    // Setup WebGPU window
    // ----------------------------------------------------------------
    const int WIN = 900;
    Canvas::Parameters params;
    params.title("RT-DETR-L WebGPU Inference")
          .size(WIN, WIN);

    Canvas canvas(params);
    WgpuRenderer renderer(canvas);
    renderer.setClearColor(Color(0x1a1a2e));

    // ----------------------------------------------------------------
    // Run inference
    // ----------------------------------------------------------------
    std::cout << "Loading weights from '" << weightsPath << "' ...\n";

    using clk = std::chrono::steady_clock;

    std::vector<rtdetr::RtDetr::Detection> detections;
    {
        const int WARMUP = 3;
        const int RUNS   = 10;

        rtdetr::RtDetr model(renderer);
        model.loadWeights(weightsPath);

        auto& rgba = img.data<unsigned char>();
        std::filesystem::path imgFs(imgPath);
        double imgKB = std::filesystem::exists(imgFs)
            ? double(std::filesystem::file_size(imgFs)) / 1024.0 : 0.0;

        std::string bar(52, '=');
        std::cout << "\n" << bar << "\n"
                  << "  RT-DETR-L WebGPU Inference Benchmark\n"
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
            double dt = std::chrono::duration<double, std::milli>(tb - ta).count();
            times_ms.push_back(dt);

            std::cout << "  Run " << std::setw(2) << (i + 1) << "/" << RUNS << "  "
                      << std::fixed << std::setprecision(2) << std::setw(7) << dt
                      << " ms   detections: " << detections.size() << "\n";
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
        double stdev = std::sqrt(sq / std::max(1, RUNS - 1));
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

        // Print detections
        std::cout << "Detections (" << detections.size() << "):\n";
        for (size_t i = 0; i < detections.size(); ++i) {
            auto& d = detections[i];
            const char* name = (d.classId >= 0 && d.classId < 80)
                ? kCocoNames[d.classId] : "unknown";
            std::cout << "  [" << i << "] " << std::left << std::setw(18) << name
                      << std::right << std::fixed
                      << "  conf=" << std::setprecision(3) << d.confidence
                      << "  box=(" << std::setprecision(1)
                      << d.x1 << "," << d.y1 << ","
                      << d.x2 << "," << d.y2 << ")\n";
        }
    }

    // ----------------------------------------------------------------
    // Scene: display image + bounding boxes with an orthographic camera
    // ----------------------------------------------------------------
    auto scene = Scene::create();

    // Orthographic camera: (left, right, top, bottom, near, far).
    // top=640 = screen top, bottom=0 = screen bottom.
    auto camera = OrthographicCamera::create(0, 640, 640, 0, -1, 1);

    // Upload image as texture
    TextureLoader texLoader;
    auto displayTex = texLoader.load(imgPath, true);  // flipY for correct orientation

    // Plane covering [0,640] x [0,640]
    auto planeGeo = PlaneGeometry::create(640, 640);
    auto planeMat = MeshBasicMaterial::create();
    planeMat->map = displayTex;
    planeMat->side = Side::Double;

    auto quad = Mesh::create(planeGeo, planeMat);
    quad->position.set(320, 320, 0);
    scene->add(quad);

    FontLoader fontLoader;
    const Font font = fontLoader.defaultFont();

    // Map detections from original-image pixel space onto the 640x640 display
    const float sx = 640.f / float(img.width);
    const float sy = 640.f / float(img.height);
    for (auto& d : detections) {
        float x1 = std::max(0.f, d.x1 * sx), y1 = std::max(0.f, d.y1 * sy);
        float x2 = std::min(640.f, d.x2 * sx), y2 = std::min(640.f, d.y2 * sy);

        // Flip y: image y=0 is top -> world y=640 (screen top)
        float sy1 = 640.f - y2;
        float sy2 = 640.f - y1;

        const Color& col = detviz::kPalette[d.classId % 6];
        scene->add(detviz::makeBoxLines(x1, sy1, x2, sy2, col));
        const char* name = (d.classId >= 0 && d.classId < 80) ? kCocoNames[d.classId] : "?";
        scene->add(detviz::makeLabel(font, detviz::labelText(name, d.confidence), col, x1, sy2));
    }

    // ----------------------------------------------------------------
    // Render loop
    // ----------------------------------------------------------------
    canvas.animate([&] {
        renderer.render(*scene, *camera);
    });

    return 0;
}
