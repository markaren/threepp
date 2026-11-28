/// RF-DETR Vulkan Object Detection / Pose Inference Example.
///
/// Usage:
///   vulkan_rfdetr_inference <image> <weights>               run detection + visualize
///   vulkan_rfdetr_inference <image> <weights> --bench <N>   time N inferences, print, exit
///   vulkan_rfdetr_inference --validate <weights> <ref.bin>  numeric self-check
///
///   --variant nano|small|medium|keypoint picks the model (default nano);
///   keypoint is RF-DETR Keypoint: people with their 17 COCO keypoints.
///   --fp32 keeps every matrix product in fp32 where the GPU would use fp16
///   tensor cores (exact against PyTorch's fp32; ~2x slower).
///   --bench, --validate and --shot run in a hidden window.
///
///   image    – JPEG/PNG input image
///   weights  – scripts/rfdetr-nano.weights   (export_rfdetr_weights.py)
///   ref.bin  – scripts/rfdetr_nano_ref.bin   (capture_rfdetr_activations.py)
///
/// Direct Vulkan port of the RF-DETR / LW-DETR model (DINOv2-windowed ViT-S
/// backbone + C2f projector + two-stage deformable decoder). The backbone and
/// projector validate bit-exact against the captured PyTorch reference; the
/// decoder matches except for the top-K proposal ordering, which is numerically
/// unstable on the seeded-noise validation input (stable on real images: capture
/// with --image to validate the decoder).

#include "threepp/threepp.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/materials/SpriteMaterial.hpp"

#include "utility/DetectionOverlay.hpp"
#include "rfdetr/RfDetrVk.hpp"
#include "rfdetr/WeightLoader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

// COCO category-id (1..90) -> name; index 0 and the unused ids map to "N/A".
static const char* kCoco91[91] = {
    "N/A","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","N/A","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","N/A","backpack",
    "umbrella","N/A","N/A","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "N/A","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "N/A","dining table","N/A","N/A","toilet","N/A","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","N/A","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
};

// The COCO person skeleton over the 17 keypoints (nose, eyes, ears, shoulders,
// elbows, wrists, hips, knees, ankles; left before right).
static const int kCocoSkeleton[19][2] = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
    {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}};
// Keypoints below this confidence are not drawn (the threshold Roboflow's examples use).
static constexpr float kKeypointThresh = 0.1f;

// One detection's skeleton in the viewer's 640x640 y-up space: the limbs whose two
// ends are confident, and a filled dot on every confident keypoint (a 1 px line alone
// disappears on dark clothing).
static std::shared_ptr<Object3D> makeSkeleton(const rfdetr::Detection& d, float sx, float sy, const Color& col) {
    auto at = [&](int k) { return std::pair<float, float>{d.keypoints[k].x * sx, 640.f - d.keypoints[k].y * sy}; };
    auto seen = [&](int k) { return k < int(d.keypoints.size()) && d.keypoints[k].conf >= kKeypointThresh; };
    auto group = Group::create();

    std::vector<float> limbs;
    if (d.keypoints.size() == 17)
        for (const auto& e : kCocoSkeleton) {
            if (!seen(e[0]) || !seen(e[1])) continue;
            auto [x0, y0] = at(e[0]);
            auto [x1, y1] = at(e[1]);
            limbs.insert(limbs.end(), {x0, y0, 0.2f, x1, y1, 0.2f});
        }
    if (!limbs.empty()) {
        auto geo = BufferGeometry::create();
        geo->setAttribute("position", FloatBufferAttribute::create(limbs, 3));
        auto mat = LineBasicMaterial::create();
        mat->color = col;
        mat->depthTest = false;
        group->add(LineSegments::create(geo, mat));
    }

    auto dotMat = SpriteMaterial::create();
    dotMat->map = detviz::solidTexture(col);
    for (int k = 0; k < int(d.keypoints.size()); ++k) {
        if (!seen(k)) continue;
        auto [x, y] = at(k);
        auto dot = Sprite::create(dotMat);
        dot->scale.set(6.f, 6.f, 1.f);
        dot->position.set(x, y, 0.3f);
        group->add(dot);
    }
    return group->children.empty() ? nullptr : group;
}

// ── Validation mode: diff every layer against the captured reference ──
static int runValidation(rfdetr::RfDetrVk& model, const std::string& weightsPath, const std::string& refPath) {
    std::cout << "Loading weights '" << weightsPath << "' ..." << std::endl;
    model.loadWeights(weightsPath);
    std::cout << "Loading reference '" << refPath << "' ..." << std::endl;
    auto ref = rfdetr::parseWeightBinary(refPath);
    auto inIt = ref.data.find("input");
    if (inIt == ref.data.end()) { std::cerr << "ERROR: ref has no 'input'\n"; return 1; }

    std::cout << "\nRunning full forward pass..." << std::endl;
    auto fw = model.runForward(inIt->second, /*captureIntermediates=*/true);

    // fp32 products match PyTorch to ~1e-5; fp16 tensor-core products to ~1e-3 relative,
    // so those runs are judged on the relative error.
    const bool fp16 = model.tensorCores();
    std::cout << "\nMatrix products: " << (fp16 ? "fp16 tensor cores, judged on relL1 < 5e-3 (--fp32 for the fp32 check)"
                                                 : "fp32, judged on maxAbsErr < 2e-2") << "\n";
    auto compare = [&](const char* name, const std::vector<float>& got, const std::string& key) {
        auto it = ref.data.find(key);
        if (it == ref.data.end()) { std::cout << "  " << name << ": ref '" << key << "' missing\n"; return; }
        const auto& want = it->second;
        if (want.size() != got.size()) {
            std::cout << "  " << name << ": SIZE " << got.size() << " vs " << want.size() << "\n";
            return;
        }
        double maxAbs = 0, sumAbs = 0, sumRef = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            double d = std::abs(double(got[i]) - double(want[i]));
            maxAbs = std::max(maxAbs, d); sumAbs += d; sumRef += std::abs(double(want[i]));
        }
        const double relL1 = sumRef > 0 ? sumAbs / sumRef : 0.0;
        std::cout << "  " << std::left << std::setw(7) << name << std::right
                  << "  maxAbsErr " << std::scientific << std::setprecision(3) << maxAbs
                  << "  relL1 " << relL1 << ((fp16 ? relL1 < 5e-3 : maxAbs < 2e-2) ? "   OK" : "   CHECK") << "\n";
    };

    std::cout << "\nNumeric validation vs captured reference:\n";
    compare("patch", fw.patchEmbed, "patch");
    compare("embed", fw.embed, "embed");
    compare("vit0", fw.vit0, "vit0");
    compare("vit3", fw.vit3, "vit3");
    compare("tap0", fw.taps[0], "tap.0");
    compare("tap1", fw.taps[1], "tap.1");
    compare("tap2", fw.taps[2], "tap.2");
    compare("tap3", fw.taps[3], "tap.3");
    compare("proj0", fw.proj0, "proj.0");
    const bool keypoints = model.config().numKeypoints > 0;
    if (keypoints) {
        compare("projkp", fw.projKp0, "projkp.0");
        compare("dec.ref", fw.decRef, "dec.ref");
        compare("dec.hs", fw.decHs, "dec.hs");
        compare("dec.kp", fw.decKp, "dec.kp");
    }
    compare("logits", fw.predLogits, "pred_logits");
    compare("boxes", fw.predBoxes, "pred_boxes");
    if (keypoints) compare("kpts", fw.predKeypoints, "pred_keypoints");

    // Row-level diagnostic: the two-stage top-K selection is numerically unstable
    // on random (non-image) input — near-tied proposal scores get ordered
    // differently under ~1e-5 drift, so a handful of the 300 query rows are
    // permuted. Count how many rows actually diverge vs. a structural bug.
    auto rowDiag = [&](const char* name, const std::vector<float>& got, const std::string& key, uint32_t cols) {
        auto it = ref.data.find(key);
        if (it == ref.data.end() || it->second.size() != got.size()) return;
        const auto& want = it->second;
        uint32_t rows = uint32_t(got.size() / cols), bad = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            double mx = 0;
            for (uint32_t c = 0; c < cols; ++c)
                mx = std::max(mx, std::abs(double(got[r * cols + c]) - double(want[r * cols + c])));
            if (mx > 0.05) ++bad;
        }
        std::cout << "  " << name << ": " << bad << "/" << rows << " query rows diverge (>0.05)"
                  << " (near-tied proposals swap in the top-K under small drift - see notes)\n";
    };
    rowDiag("logits", fw.predLogits, "pred_logits", uint32_t(model.config().numClasses));
    rowDiag("boxes", fw.predBoxes, "pred_boxes", 4);
    if (keypoints) rowDiag("kpts", fw.predKeypoints, "pred_keypoints", uint32_t(model.config().numKeypoints) * 8u);

    // The keypoint head's channels differ in scale and in what they feed (the
    // uncertainty rescoring reads 2 and 4-6), so its error is worth seeing per channel.
    if (auto it = ref.data.find("pred_keypoints"); keypoints && it != ref.data.end() &&
                                                    it->second.size() == fw.predKeypoints.size()) {
        static const char* kChannel[8] = {"x", "y", "findable", "visible", "log_l11", "l21", "log_l22", "class"};
        std::cout << "  kpts max abs error per channel:";
        for (size_t c = 0; c < 8; ++c) {
            double mx = 0;
            for (size_t i = c; i < fw.predKeypoints.size(); i += 8)
                mx = std::max(mx, std::abs(double(fw.predKeypoints[i]) - double(it->second[i])));
            std::cout << "  " << kChannel[c] << " " << std::scientific << std::setprecision(1) << mx;
        }
        std::cout << "\n";
    }

    // The host postprocess on the reference input, to compare with rfdetr's
    // PostProcess on the same tensor (none above the threshold on a noise input).
    const int R = model.config().resolution;
    const auto dets = model.postprocess(fw, R, R, 0.5f);

    // The same postprocess on PyTorch's own heads: the detections are what a swap of
    // near-tied queries in the top-K leaves unchanged, so they are the end-to-end check.
    {
        rfdetr::RfDetrVk::ForwardOut ref32;
        auto take = [&](const std::string& key, std::vector<float>& dst) {
            if (auto it = ref.data.find(key); it != ref.data.end()) dst = it->second;
        };
        take("pred_logits", ref32.predLogits);
        take("pred_boxes", ref32.predBoxes);
        if (keypoints) take("pred_keypoints", ref32.predKeypoints);
        if (ref32.predLogits.size() == fw.predLogits.size() && ref32.predBoxes.size() == fw.predBoxes.size() &&
            ref32.predKeypoints.size() == fw.predKeypoints.size()) {
            const auto want = model.postprocess(ref32, R, R, 0.5f);
            double dConf = 0, dBox = 0, dKp = 0;
            bool matched = want.size() == dets.size();
            for (const auto& w : want) {// each of PyTorch's detections against our best-overlapping one
                double bestIou = 0;
                const rfdetr::Detection* best = nullptr;
                for (const auto& d : dets) {
                    const double ix = std::max(0.0, double(std::min(w.x2, d.x2) - std::max(w.x1, d.x1)));
                    const double iy = std::max(0.0, double(std::min(w.y2, d.y2) - std::max(w.y1, d.y1)));
                    const double un = double(w.x2 - w.x1) * (w.y2 - w.y1) + double(d.x2 - d.x1) * (d.y2 - d.y1) - ix * iy;
                    if (d.classId == w.classId && un > 0 && ix * iy / un > bestIou) { bestIou = ix * iy / un; best = &d; }
                }
                if (!best) { matched = false; continue; }
                dConf = std::max(dConf, double(std::abs(best->confidence - w.confidence)));
                dBox = std::max({dBox, double(std::abs(best->x1 - w.x1)), double(std::abs(best->y1 - w.y1)),
                                 double(std::abs(best->x2 - w.x2)), double(std::abs(best->y2 - w.y2))});
                for (size_t k = 0; k < w.keypoints.size() && k < best->keypoints.size(); ++k)
                    dKp = std::max({dKp, double(std::abs(best->keypoints[k].x - w.keypoints[k].x)),
                                    double(std::abs(best->keypoints[k].y - w.keypoints[k].y))});
            }
            std::cout << "\nDetections against PyTorch's on the same input: " << dets.size() << " vs " << want.size()
                      << std::fixed << std::setprecision(4) << ", confidence within " << dConf << std::setprecision(2)
                      << ", boxes within " << dBox << " px";
            if (keypoints) std::cout << ", keypoints within " << dKp << " px";
            std::cout << ((matched && dConf < 1e-2 && dBox < 1.0 && dKp < 2.0) ? "   OK" : "   CHECK") << "\n"
                      << std::scientific;
        }
    }
    std::cout << "\nDetections on the reference input (" << R << "x" << R << " input pixels, score > 0.5): "
              << dets.size() << "\n" << std::fixed;
    for (const auto& d : dets) {
        std::cout << "  class " << d.classId << "  conf " << std::setprecision(4) << d.confidence << "  ["
                  << std::setprecision(1) << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "]\n";
        if (!d.keypoints.empty()) {
            std::cout << "    keypoints (x y conf):";
            for (const auto& k : d.keypoints)
                std::cout << "  " << std::setprecision(1) << k.x << " " << k.y << " " << std::setprecision(2) << k.conf;
            std::cout << "\n";
        }
    }
    std::cout.flush();
    return 0;
}

// ── Detection mode: run inference and visualize boxes over the image ──
static int runDetection(Canvas& canvas, VulkanRenderer& renderer, rfdetr::RfDetrVk& model,
                        const std::string& imgPath, const std::string& weightsPath,
                        const std::string& shotPath, int benchRuns) {
    ImageLoader imgLoader;
    auto imgOpt = imgLoader.load(imgPath, 4, false);
    if (!imgOpt) { std::cerr << "ERROR: cannot load image '" << imgPath << "'\n"; return 1; }
    auto& img = *imgOpt;
    std::cout << "Loaded image: " << img.width() << "x" << img.height() << "\n";

    std::cout << "Loading weights '" << weightsPath << "' ..." << std::endl;
    model.loadWeights(weightsPath);

    auto& rgba = img.data<unsigned char>();
    using clk = std::chrono::steady_clock;

    const float scoreThresh = 0.5f;// rfdetr's predict() default, for every variant
    std::vector<rfdetr::Detection> dets;
    // Sustained warmup: short bursty runs leave the GPU idle-clocked (~1/3 of boost),
    // which understates throughput and makes comparisons (esp. vs PyTorch, which boosts
    // during its 50-iter loop) unfair. Drive load ~2s so the boost clock settles, then
    // time enough runs to stay boosted throughout.
    {
        auto wt0 = clk::now();
        do { dets = model.infer(rgba.data(), int(img.width()), int(img.height()), scoreThresh); }
        while (std::chrono::duration<double>(clk::now() - wt0).count() < 2.0);
    }
    const int runs = benchRuns > 0 ? benchRuns : 50;
    std::vector<double> times(static_cast<size_t>(runs));
    double total = 0.0;
    for (int i = 0; i < runs; ++i) {
        auto t0 = clk::now();
        dets = model.infer(rgba.data(), int(img.width()), int(img.height()), scoreThresh);
        times[size_t(i)] = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        total += times[size_t(i)];
    }
    const double ms = total / runs;

    if (benchRuns > 0) {
        // The distribution, not just the mean: one slow outlier (a driver hiccup)
        // moves a mean of short runs but not the median.
        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        auto at = [&](double q) { return sorted[size_t(q * double(sorted.size() - 1) + 0.5)]; };
        std::cout << std::fixed << std::setprecision(2)
                  << "bench: " << runs << " runs  median " << at(0.5) << " ms  mean " << ms
                  << "  p10 " << at(0.1) << "  p90 " << at(0.9) << "  min " << sorted.front() << "\n";
        for (auto& d : dets) {
            const char* name = (d.classId >= 0 && d.classId < 91) ? kCoco91[d.classId] : "?";
            std::cout << "  " << std::left << std::setw(16) << name << std::right
                      << "  conf " << std::setprecision(4) << d.confidence << "  [" << std::setprecision(1)
                      << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "]\n";
            if (!d.keypoints.empty()) {
                std::cout << "    keypoints (x y conf):";
                for (const auto& k : d.keypoints)
                    std::cout << "  " << std::setprecision(1) << k.x << " " << k.y << " " << std::setprecision(2) << k.conf;
                std::cout << "\n";
            }
        }
        return 0;
    }

    std::cout << "\nDetections (" << dets.size() << ") in " << std::fixed << std::setprecision(1)
              << ms << " ms  (" << (1000.0 / ms) << " FPS):\n";
    for (auto& d : dets) {
        const char* name = (d.classId >= 0 && d.classId < 91) ? kCoco91[d.classId] : "?";
        std::cout << "  " << std::left << std::setw(16) << name << std::right
                  << "  conf " << std::setprecision(3) << d.confidence
                  << "  [" << int(d.x1) << "," << int(d.y1) << "," << int(d.x2) << "," << int(d.y2) << "]\n";
    }
    std::cout << "Opening viewer window (close it to exit)..." << std::endl;

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

    const float sx = 640.f / float(img.width()), sy = 640.f / float(img.height());
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        float bx1 = d.x1 * sx, bx2 = d.x2 * sx;
        float by1 = 640.f - d.y2 * sy, by2 = 640.f - d.y1 * sy;// flip Y for ortho
        // Colour by class; people with keypoints by detection, so overlapping skeletons stay apart.
        const Color& col = detviz::kPalette[(d.keypoints.empty() ? size_t(d.classId) : i) % 6];
        scene->add(detviz::makeBoxLines(bx1, by1, bx2, by2, col));
        if (auto skeleton = makeSkeleton(d, sx, sy, col)) scene->add(skeleton);
        const char* name = (d.classId >= 0 && d.classId < 91) ? kCoco91[d.classId] : "?";
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

    // Optional --variant nano|small|medium|keypoint (default nano), stripped from
    // args so the positional <image>/<weights> handling below is unchanged.
    auto variant = rfdetr::RfDetrVariant::Nano;
    for (size_t i = 0; i + 1 < args.size();) {
        if (args[i] == "--variant") {
            const std::string& v = args[i + 1];
            if (v == "nano") variant = rfdetr::RfDetrVariant::Nano;
            else if (v == "small") variant = rfdetr::RfDetrVariant::Small;
            else if (v == "medium") variant = rfdetr::RfDetrVariant::Medium;
            else if (v == "keypoint") variant = rfdetr::RfDetrVariant::Keypoint;
            else { std::cerr << "Unknown --variant '" << v << "' (nano|small|medium|keypoint)\n"; return 1; }
            args.erase(args.begin() + i, args.begin() + i + 2);
        } else {
            ++i;
        }
    }

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

    // --fp32: every matrix product in fp32 even where the GPU has tensor cores (the
    // fp32 match with PyTorch, and the path GPUs without them take).
    bool fp32 = false;
    for (size_t i = 0; i < args.size();) {
        if (args[i] == "--fp32") {
            fp32 = true;
            args.erase(args.begin() + i);
        } else {
            ++i;
        }
    }

    // --bench <N>: time N inferences after the warmup, print them, exit (no viewer).
    int benchRuns = 0;
    for (size_t i = 0; i + 1 < args.size();) {
        if (args[i] == "--bench") {
            benchRuns = std::max(1, std::atoi(args[i + 1].c_str()));
            args.erase(args.begin() + i, args.begin() + i + 2);
        } else {
            ++i;
        }
    }

    const bool validate = !args.empty() && args[0] == "--validate";
    if ((validate && args.size() < 3) || (!validate && args.size() < 2)) {
        std::cerr << "Usage: " << argv[0] << " [--variant nano|small|medium|keypoint] [--fp32] <image> <weights> [--shot out.png | --bench N]\n"
                  << "   or: " << argv[0] << " [--variant nano|small|medium|keypoint] [--fp32] --validate <weights> <ref.bin>\n";
        return 1;
    }

    Canvas::Parameters params;
    params.title("RF-DETR Vulkan Inference").size(900, 900);
    // Nobody looks at these runs: a hidden window keeps the keyboard focus where it was.
    params.headless(validate || benchRuns > 0 || !shotPath.empty());
    Canvas canvas(params);
    VulkanRenderer renderer(canvas);
    renderer.setClearColor(Color(0x1a1a2e));

    try {
        rfdetr::RfDetrVk model(renderer, variant, /*tensorCores=*/!fp32);
        if (validate)
            return runValidation(model, args[1], args[2]);
        return runDetection(canvas, renderer, model, args[0], args[1], shotPath, benchRuns);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
