#pragma once

#include "VkInfer.hpp"

#include "threepp/renderers/VulkanRenderer.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace yolo {

    struct Detection {
        float x1, y1, x2, y2;///< Box in 640x640 model space (un-letterboxed by infer)
        float conf;          ///< Object confidence (sigmoid of best class logit)
        int   cls_id;        ///< COCO class index (0-based)
    };

    /// YOLOv8n object detector running on the Vulkan compute pipeline. A direct
    /// port of the WGPU YoloV8n: same weight format (export_yolov8n_weights.py),
    /// same offline BN-folding + f16 weight packing, same op set and forward
    /// pass. The whole forward pass is recorded into one command buffer and
    /// submitted once (see VkInfer); activations are arena-owned Tensor views,
    /// weights and detection buffers are persistent owning VkTensors.
    class YoloV8nVk {
    public:
        static constexpr int INPUT_SIZE  = 640;
        static constexpr int NUM_CLASSES = 80;
        static constexpr int REG_MAX     = 16;

        explicit YoloV8nVk(threepp::VulkanRenderer& renderer);
        ~YoloV8nVk();

        YoloV8nVk(const YoloV8nVk&) = delete;
        YoloV8nVk& operator=(const YoloV8nVk&) = delete;

        void loadWeights(const std::string& path);

        std::vector<Detection> infer(const unsigned char* rgba,
                                     int width, int height,
                                     float confThresh = 0.25f,
                                     float iouThresh  = 0.45f);

    private:
        VkInfer vk_;

        // One pipeline per op type (reused for every layer of that type).
        VkPipe convPipe_, conv3x3s1Pipe_, conv3x3s2Pipe_, conv1x1Pipe_;
        VkPipe bnSiluPipe_, addPipe_, maxpoolPipe_, upsamplePipe_;
        VkPipe concatPipe_, slicePipe_, detectPipe_, detectFusedPipe_, preprocessPipe_;

        // Letterbox state (written by preprocess_, consumed to un-letterbox boxes).
        float lbScale_ = 1.f, lbPadX_ = 0.f, lbPadY_ = 0.f;

        // GPU detection pre-filter: shared counter + dets across all 3 scales.
        static constexpr uint32_t MAX_DETS = 1024;
        VkTensor detCounter_;// persistent (owning) 1 x uint32 atomic
        VkTensor detDets_;   // persistent (owning) MAX_DETS x 6 floats

        std::unordered_map<std::string, VkTensor> weights_;// persistent (owning)

        // ── low-level GPU ops (activations are arena-owned Tensor views) ────
        Tensor conv_(const Tensor& x, const std::string& weightKey, const std::string& biasKey,
                     int strideH = 1, int strideW = 1, bool silu = false,
                     const Tensor* residual = nullptr);
        Tensor bnSilu_(const Tensor& x, const std::string& bnPrefix);
        Tensor add_(const Tensor& a, const Tensor& b);
        Tensor maxpool_(const Tensor& x, int k, int stride, int pad);
        Tensor upsample2x_(const Tensor& x);
        Tensor concat_(const Tensor& a, const Tensor& b);
        Tensor sliceChannels_(const Tensor& x, uint32_t offsetC, uint32_t countC);

        // ── compound blocks ─────────────────────────────────────────────────
        Tensor convBnSilu_(const Tensor& x, const std::string& prefix, int stride = 1,
                           const Tensor* residual = nullptr);
        Tensor plainConv_(const Tensor& x, const std::string& prefix);
        Tensor c2f_(const Tensor& x, const std::string& prefix, int n, bool shortcut);
        Tensor sppf_(const Tensor& x, const std::string& prefix);

        struct SourceView {
            VkBuffer buf;
            uint32_t srcTotalC, srcOffsetC, countC;
        };
        Tensor concatMany_(std::vector<Tensor>& parts);
        Tensor concatViews_(const std::vector<SourceView>& views, uint32_t H, uint32_t W);

        // ── detect head + NMS ───────────────────────────────────────────────
        void decodeScaleGpu_(const Tensor& feat, int scaleIdx, float stride, float confThresh);
        std::vector<Detection> nms_(std::vector<Detection>& dets, float iouThresh);

        // ── preprocessing ───────────────────────────────────────────────────
        Tensor preprocess_(const unsigned char* rgba, int width, int height);
    };

}// namespace yolo
