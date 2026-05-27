#pragma once

#include "VkInfer.hpp"

#include "threepp/renderers/VulkanRenderer.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace rfdetr {

    struct Detection {
        int   classId;
        float confidence;
        float x1, y1, x2, y2;
    };

    /// RF-DETR-Nano object detector on the Vulkan compute pipeline. Port of the
    /// LWDETR / DINOv2-windowed-ViT model (rfdetr package). Same VkInfer harness
    /// and most shaders as the RT-DETR port; the new piece is the windowed ViT
    /// backbone (patch-embed + CLS token + 12 transformer layers w/ LayerScale,
    /// 2x2 windowed attention with global layers {3,6,9}, no register tokens),
    /// the C2f projector (tap layers 3/6/9/final -> single P4), and a two-stage
    /// deformable decoder (L=1, reparam bbox).
    ///
    /// Built in phases; current surface validates the backbone+projector against
    /// the captured proj.0 reference.
    class RfDetrVk {
    public:
        static constexpr int   RESOLUTION = 384;
        static constexpr int   PATCH      = 16;
        static constexpr int   GRID       = RESOLUTION / PATCH;// 24
        static constexpr int   EMBED_DIM  = 384;
        static constexpr int   NUM_LAYERS = 12;
        static constexpr int   NUM_WINDOWS = 2;
        static constexpr int   NUM_HEADS  = 6;// DINOv2 ViT-S
        static constexpr int   NUM_CLASSES = 91;

        explicit RfDetrVk(threepp::VulkanRenderer& renderer);
        ~RfDetrVk();
        RfDetrVk(const RfDetrVk&) = delete;
        RfDetrVk& operator=(const RfDetrVk&) = delete;

        void loadWeights(const std::string& path);

        // Phase-1 validation: run the patch embedding (+ backbone as wired so far)
        // on a [3,384,384] CHW input and read back the projector P4 feature.
        struct ForwardOut {
            std::vector<float>     patchEmbed;// [EMBED_DIM, GRID, GRID]
            std::array<uint32_t, 3> patchDim{};
        };
        ForwardOut runForward(const std::vector<float>& chw);

    private:
        VkInfer vk_;
        VkPipe convPipe_, conv1x1Pipe_, conv3x3Pipe_, im2colPipe_;
        VkPipe linearPipe_, layerNormPipe_, softmaxPipe_, geluPipe_, layerScalePipe_;
        VkPipe attnScoresPipe_, attnApplyPipe_, addPipe_, concatPipe_;

        std::unordered_map<std::string, VkTensor> weights_;

        // ── ops ──
        Tensor conv_(const Tensor& x, const std::string& wKey, const std::string& bKey,
                     int strideH, int strideW, int padH, int padW);
        Tensor patchEmbed_(const Tensor& x);
    };

}// namespace rfdetr
