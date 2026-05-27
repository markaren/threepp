/// RF-DETR-Nano Vulkan Object Detection Inference Example (work in progress).
///
/// Usage: vulkan_rfdetr_inference --validate <weights> <ref.bin>
///   weights  – scripts/rfdetr-nano.weights   (export_rfdetr_weights.py)
///   ref.bin  – scripts/rfdetr_nano_ref.bin    (capture_rfdetr_activations.py)
///
/// Port of the RF-DETR / LWDETR model (DINOv2-windowed ViT backbone + C2f
/// projector + two-stage deformable decoder) to the Vulkan compute pipeline.
/// Built in phases; current surface validates the backbone stages against the
/// captured reference activations.

#include "threepp/threepp.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "rfdetr/RfDetrVk.hpp"
#include "rfdetr/WeightLoader.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

static void printStats(const char* name, const std::vector<float>& v,
                       const std::array<uint32_t, 3>& dim) {
    double sum = 0.0;
    float mn = v.empty() ? 0.f : v[0], mx = mn;
    for (float f : v) { sum += f; mn = std::min(mn, f); mx = std::max(mx, f); }
    double mean = v.empty() ? 0.0 : sum / double(v.size());
    std::cout << "  " << std::left << std::setw(6) << name << std::right
              << " [" << dim[0] << "," << dim[1] << "," << dim[2] << "]"
              << "  mean " << std::fixed << std::setprecision(5) << std::setw(10) << mean
              << "  min " << std::setw(10) << mn << "  max " << std::setw(10) << mx << "\n";
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() < 3 || args[0] != "--validate") {
        std::cerr << "Usage: " << argv[0] << " --validate <weights> <ref.bin>\n";
        return 1;
    }

    Canvas::Parameters params;
    params.title("RF-DETR Vulkan Inference").size(900, 900);
    Canvas canvas(params);
    VulkanRenderer renderer(canvas);

    try {
        rfdetr::RfDetrVk model(renderer);
        std::cout << "Loading weights '" << args[1] << "' ..." << std::endl;
        model.loadWeights(args[1]);

        std::cout << "Loading reference '" << args[2] << "' ..." << std::endl;
        auto ref = rfdetr::parseWeightBinary(args[2]);
        auto inIt = ref.data.find("input");
        if (inIt == ref.data.end()) { std::cerr << "ERROR: ref has no 'input'\n"; return 1; }

        std::cout << "\nRunning forward (patch embed stage)..." << std::endl;
        auto fw = model.runForward(inIt->second);

        std::cout << "\nStage statistics:\n";
        printStats("patch", fw.patchEmbed, fw.patchDim);
        std::cout << "  (proj.0 / pred_logits / pred_boxes comparison pending backbone+decoder)\n";
        std::cout.flush();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
