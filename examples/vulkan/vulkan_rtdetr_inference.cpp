/// RT-DETR-L Vulkan Object Detection Inference Example
///
/// Usage:
///   vulkan_rtdetr_inference <image> <weights>        run detection + visualize
///   vulkan_rtdetr_inference --validate <weights> <ref.bin>  numeric self-check
///   vulkan_rtdetr_inference --selftest               MSDeformAttn analytical test
///
///   image    – JPEG/PNG input image
///   weights  – scripts/rtdetr_l.weights   (export_rtdetr_weights.py)
///   ref.bin  – scripts/rtdetr_l_ref.bin   (capture_rtdetr_activations.py)
///
/// Direct Vulkan port of examples/wgpu/rtdetr: runs the full RT-DETR-L forward
/// pass (HGNetv2 backbone → AIFI encoder → CCFM neck → deformable decoder) on
/// the Vulkan compute pipeline, then displays the image with detection boxes
/// through the renderer's ortho HUD overlay (Sprite + LineSegments).

#include "threepp/threepp.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/materials/SpriteMaterial.hpp"

#include "utility/DetectionOverlay.hpp"
#include "rtdetr/RtDetrVk.hpp"
#include "rtdetr/WeightLoader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

// ── Validation mode (--validate): diff every layer against the captured ref ──
static void compare(const char* name, const std::vector<float>& got,
                    const rtdetr::Weights& ref, const std::string& refKey) {
    auto it = ref.data.find(refKey);
    if (it == ref.data.end()) { std::cout << "  " << name << ": ref '" << refKey << "' missing\n"; return; }
    const auto& want = it->second;
    if (want.size() != got.size()) {
        std::cout << "  " << name << ": SIZE MISMATCH " << got.size() << " vs " << want.size() << "\n";
        return;
    }
    double maxAbs = 0, sumAbs = 0, sumRef = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::abs(double(got[i]) - double(want[i]));
        maxAbs = std::max(maxAbs, d); sumAbs += d; sumRef += std::abs(double(want[i]));
    }
    std::cout << "  " << std::left << std::setw(5) << name << std::right
              << "  maxAbsErr " << std::scientific << std::setprecision(3) << maxAbs
              << "  relL1 " << (sumRef > 0 ? sumAbs / sumRef : 0.0)
              << (maxAbs < 2e-2 ? "   OK" : "   CHECK") << "\n";
}

static int runValidation(rtdetr::RtDetrVk& model, const std::string& weightsPath, const std::string& refPath) {
    std::cout << "Loading weights '" << weightsPath << "' ..." << std::endl;
    model.loadWeights(weightsPath);
    std::cout << "Loading reference '" << refPath << "' ..." << std::endl;
    auto ref = rtdetr::parseWeightBinary(refPath);
    const auto inIt = ref.data.find("input");
    if (inIt == ref.data.end()) { std::cerr << "ERROR: ref has no 'input'\n"; return 1; }

    std::cout << "\nRunning full forward pass..." << std::endl;
    const auto fw = model.runForward(inIt->second, /*captureIntermediates=*/true);

    std::cout << "\nNumeric validation vs captured reference:\n";
    compare("P3", fw.p3, ref, "model.3");
    compare("P4", fw.p4, ref, "model.7");
    compare("P5", fw.p5, ref, "model.10");
    compare("AIFI", fw.aifi, ref, "model.11");
    compare("S3", fw.s3, ref, "model.21");
    compare("S4", fw.s4, ref, "model.24");
    compare("S5", fw.s5, ref, "model.27");
    compare("IP0", fw.ip0, ref, "decoder.input_proj.0");
    compare("IP1", fw.ip1, ref, "decoder.input_proj.1");
    compare("IP2", fw.ip2, ref, "decoder.input_proj.2");
    compare("MEM", fw.memory, ref, "decoder.memory");
    compare("ENC", fw.encOutput, ref, "decoder.enc_output");

    // model.28 = [300,84] = [bbox(4) | sigmoid(scores)(80)]. Top-K ordering is
    // unstable on the seeded-noise input, so report row-aligned + set-based.
    auto it28 = ref.data.find("model.28");
    if (it28 != ref.data.end() && it28->second.size() == 300 * 84) {
        const auto& m28 = it28->second;
        auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        int rowMatched = 0, setMatched = 0;
        for (int q = 0; q < 300; ++q) {
            double qmax = 0;
            for (int c = 0; c < 4; ++c) qmax = std::max(qmax, std::abs(double(fw.decBboxes[q * 4 + c]) - m28[q * 84 + c]));
            if (qmax < 1e-2) ++rowMatched;
        }
        for (int r = 0; r < 300; ++r) {
            double best = 1e30;
            for (int q = 0; q < 300; ++q) {
                double d = 0;
                for (int c = 0; c < 4; ++c) d += std::abs(double(fw.decBboxes[q * 4 + c]) - m28[r * 84 + c]);
                best = std::min(best, d);
            }
            if (best < 4e-2) ++setMatched;
        }
        size_t oob = 0;
        for (float v : fw.decBboxes) if (v <= 0.f || v >= 1.f) ++oob;
        std::cout << "  DEC   row-aligned " << std::dec << rowMatched << "/300, set-twins "
                  << setMatched << "/300, boxes-in-(0,1) " << (oob == 0 ? "OK" : "CHECK")
                  << "  (ordering differs on noise input — see notes)\n";
        (void) sig;
    }
    std::cout.flush();
    return 0;
}

// ── Detection mode: run inference and visualize boxes over the image ──
static int runDetection(Canvas& canvas, VulkanRenderer& renderer, rtdetr::RtDetrVk& model,
                        const std::string& imgPath, const std::string& weightsPath,
                        const std::string& shotPath) {
    ImageLoader imgLoader;
    auto imgOpt = imgLoader.load(imgPath, 4, false);
    if (!imgOpt) { std::cerr << "ERROR: cannot load image '" << imgPath << "'\n"; return 1; }
    auto& img = *imgOpt;
    std::cout << "Loaded image: " << img.width << "x" << img.height << "\n";

    std::cout << "Loading weights '" << weightsPath << "' ..." << std::endl;
    model.loadWeights(weightsPath);

    auto& rgba = img.data<unsigned char>();
    using clk = std::chrono::steady_clock;

    // Warmup (first inference compiles all compute pipelines — ~70 ms cold),
    // then average a few timed runs for a stable ms / FPS figure.
    std::vector<rtdetr::Detection> detections;
    for (int i = 0; i < 4; ++i) {
        detections = model.infer(rgba.data(), static_cast<int>(img.width), static_cast<int>(img.height));
    }
    constexpr int kRuns = 5;
    double total = 0.0;
    for (int i = 0; i < kRuns; ++i) {
        auto t0 = clk::now();
        detections = model.infer(rgba.data(), int(img.width), int(img.height));
        total += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    }
    const double ms = total / kRuns;

    std::cout << "\nDetections (" << detections.size() << ") in "
              << std::fixed << std::setprecision(1) << ms << " ms  ("
              << std::setprecision(1) << (1000.0 / ms) << " FPS):\n";
    for (auto& d : detections) {
        const char* name = (d.classId >= 0 && d.classId < 80) ? kCocoNames[d.classId] : "?";
        std::cout << "  " << std::left << std::setw(16) << name << std::right
                  << "  conf " << std::setprecision(3) << d.confidence
                  << "  [" << int(d.x1) << "," << int(d.y1) << "," << int(d.x2) << "," << int(d.y2) << "]\n";
    }
    std::cout << "Opening viewer window (close it to exit)..." << std::endl;// flushes

    // Scene: image Sprite + box LineSegments in 640x640 ortho space.
    auto scene = Scene::create();
    auto camera = OrthographicCamera::create(0, 640, 640, 0, -1, 1);

    TextureLoader texLoader;
    auto displayTex = texLoader.load(imgPath, ColorSpace::Linear, true);
    auto spriteMat = SpriteMaterial::create();
    spriteMat->map = displayTex;
    auto imageSprite = Sprite::create(spriteMat);
    imageSprite->scale.set(640.f, 640.f, 1.f);
    imageSprite->position.set(320.f, 320.f, 0.f);
    scene->add(imageSprite);

    FontLoader fontLoader;
    const Font font = fontLoader.defaultFont();

    const float sx = 640.f / float(img.width), sy = 640.f / float(img.height);
    for (auto& d : detections) {
        float bx1 = d.x1 * sx, bx2 = d.x2 * sx;
        float by1 = 640.f - d.y2 * sy, by2 = 640.f - d.y1 * sy;// flip Y for ortho
        const Color& col = detviz::kPalette[d.classId % 6];
        scene->add(detviz::makeBoxLines(bx1, by1, bx2, by2, col));
        const char* name = (d.classId >= 0 && d.classId < 80) ? kCocoNames[d.classId] : "?";
        scene->add(detviz::makeLabel(font, detviz::labelText(name, d.confidence), col, bx1, by2));
    }

    int shotFrame = 0;
    canvas.animate([&] {
        renderer.render(*scene, *camera);
        if (!shotPath.empty() && ++shotFrame >= 5) {
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << "\n";
            std::exit(0);
        }
    });
    return 0;
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    const bool selfTest = !args.empty() && args[0] == "--selftest";
    const bool validate = !args.empty() && args[0] == "--validate";

    // --shot <png>: headless capture — render a few viewer frames, save, exit.
    std::string shotPath;
    for (size_t i = 0; i + 1 < args.size();) {
        if (args[i] == "--shot") {
            shotPath = args[i + 1];
            args.erase(args.begin() + i, args.begin() + i + 2);
        } else {
            ++i;
        }
    }

    if (!selfTest &&
        ((validate && args.size() < 3) || (!validate && args.size() < 2))) {
        std::cerr << "Usage: " << argv[0] << " <image> <weights> [--shot out.png]\n"
                  << "   or: " << argv[0] << " --validate <weights> <ref.bin>\n"
                  << "   or: " << argv[0] << " --selftest\n";
        return 1;
    }

    Canvas::Parameters params;
    params.title("RT-DETR Vulkan Inference").size(900, 900);
    Canvas canvas(params);
    VulkanRenderer renderer(canvas);
    renderer.setClearColor(Color(0x1a1a2e));

    try {
        rtdetr::RtDetrVk model(renderer);

        if (selfTest) {
            auto r = model.selfTestMsDeform();
            double maxErr = 0;
            constexpr float exp[4] = {12.f, 13.f, 14.f, 15.f};
            for (size_t i = 0; i < 4 && i < r.size(); ++i) maxErr = std::max(maxErr, std::abs(double(r[i]) - exp[i]));
            std::cout << "MSDeformAttn self-test: got [";
            for (float v : r) std::cout << v << " ";
            std::cout << "] expected [12 13 14 15]  maxErr " << std::scientific << maxErr
                      << (maxErr < 1e-4 ? "  PASS" : "  FAIL") << "\n";
            return maxErr < 1e-4 ? 0 : 1;
        }
        if (validate)
            return runValidation(model, args[1], args[2]);
        return runDetection(canvas, renderer, model, args[0], args[1], shotPath);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
