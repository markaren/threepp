#pragma once

#include "VkInfer.hpp"

#include "threepp/renderers/VulkanRenderer.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtdetr {

    enum class Activation : uint32_t {
        None = 0,
        ReLU = 1,
        SiLU = 2,
    };

    struct Detection {
        int   classId;
        float confidence;
        float x1, y1, x2, y2;///< image-pixel box (un-normalised by infer)
    };

    /// RT-DETR-L object detector running on the Vulkan compute pipeline. A direct
    /// port of the WGPU RtDetr: same weight format (export_rtdetr_weights.py),
    /// same offline BN-folding (fp32 conv weights, no f16 packing), same op set
    /// and forward pass. The forward pass is recorded into one command buffer and
    /// submitted once (see VkInfer); activations are arena-owned Tensor views,
    /// weights are persistent owning VkTensors.
    ///
    /// Built in phases: A = HGNetv2 backbone (this file's current surface).
    class RtDetrVk {
    public:
        static constexpr int   INPUT_SIZE  = 640;
        static constexpr int   NUM_CLASSES = 80;
        static constexpr float BN_EPS      = 1e-3f;

        explicit RtDetrVk(threepp::VulkanRenderer& renderer);
        ~RtDetrVk();

        RtDetrVk(const RtDetrVk&) = delete;
        RtDetrVk& operator=(const RtDetrVk&) = delete;

        void loadWeights(const std::string& path);

        // End-to-end detection on an RGBA image: CPU preprocess (plain resize +
        // /255, RT-DETR does not letterbox) → full forward → sigmoid/threshold +
        // light same-class NMS. Boxes are in source-image pixels.
        std::vector<Detection> infer(const unsigned char* rgba, int width, int height,
                                     float confThresh = 0.5f);

        // ── phased validation entry ─────────────────────────────────────────
        // Runs the forward pass (as far as the phases wired so far) on a pre-built
        // [3, 640, 640] CHW float input and reads back intermediate features. The
        // caller supplies the raw input (the activation-capture fixture stores it
        // un-normalised, so it uploads verbatim — same bytes the reference torch
        // run saw), letting the user diff numerics against the captured layers.
        //
        // Captured: backbone P3/P4/P5 (A) + AIFI (B) + CCFM S3/S4/S5 (C) +
        // input_proj / memory / enc_output / decoder bboxes+scores (D).
        struct ForwardOut {
            std::vector<float>     p3, p4, p5, aifi, s3, s4, s5;
            std::array<uint32_t, 3> p3dim{}, p4dim{}, p5dim{}, aifidim{};// {C, H, W}
            std::array<uint32_t, 3> s3dim{}, s4dim{}, s5dim{};
            std::vector<float>     ip0, ip1, ip2, memory, encOutput;
            std::array<uint32_t, 3> ip0dim{}, ip1dim{}, ip2dim{};
            std::vector<float>     decScores;// [300, 80] class logits
            std::vector<float>     decBboxes;// [300, 4]  cx,cy,w,h in [0,1]
        };
        ForwardOut runForward(const std::vector<float>& chw);

        // Analytical MSDeformAttn self-test (no weights). Mirrors the WGPU
        // milestone-9 case; expected output is {12, 13, 14, 15}. Isolates the
        // deformable-attention shader from top-K / weight dependencies.
        std::vector<float> selfTestMsDeform();

    private:
        VkInfer vk_;

        // One pipeline per op type (reused for every layer of that type).
        VkPipe convPipe_, dwConvPipe_, maxPoolPipe_, concatPipe_;
        VkPipe addPipe_, addSiluPipe_, upsamplePipe_;
        // Phase B: AIFI transformer encoder.
        VkPipe linearPipe_, layerNormPipe_, softmaxPipe_, geluPipe_;
        VkPipe transposeCHW2TokensPipe_, transposeTokens2CHWPipe_;
        VkPipe qkvSplicePipe_, attnScoresPipe_, attnApplyPipe_;
        // Phase D: deformable decoder.
        VkPipe reluPipe_, msDeformAttnPipe_, offsetPreprocessPipe_, transposeMaskPipe_;

        std::unordered_map<std::string, VkTensor> weights_;// persistent (owning)

        // ── low-level GPU ops (activations are arena-owned Tensor views) ────
        Tensor conv_(const Tensor& x, const std::string& weightKey, const std::string& biasKey,
                     int strideH, int strideW, int padH, int padW, Activation act);
        Tensor conv_(const Tensor& x, const std::string& weightKey, const std::string& biasKey,
                     int strideH, int strideW,
                     int padTop, int padLeft, int padBottom, int padRight, Activation act);
        Tensor dwConv_(const Tensor& x, const std::string& weightKey, const std::string& biasKey,
                       int strideH, int strideW, int padH, int padW, Activation act);
        Tensor maxPool_(const Tensor& x, int kH, int kW, int strideH, int strideW,
                        int padTop, int padLeft, int padBottom, int padRight);
        Tensor concatC_(const Tensor& a, const Tensor& b);
        Tensor addTensor_(const Tensor& a, const Tensor& b);
        Tensor addSilu_(const Tensor& a, const Tensor& b);
        Tensor upsample2x_(const Tensor& x);

        // ── compound backbone blocks ────────────────────────────────────────
        Tensor hgStem_(const Tensor& x);
        Tensor lightConv_(const Tensor& x, const std::string& prefix, int k);
        Tensor hgBlock_(const Tensor& x, const std::string& prefix,
                        int n, int k, bool shortcut, bool lightconv);

        struct BackboneFeatures { Tensor p3, p4, p5; };
        BackboneFeatures backbone_(const Tensor& x);

        // ── Phase B: transformer ops + AIFI encoder ─────────────────────────
        Tensor uploadArena_(const std::vector<uint32_t>& shape, const float* data);
        Tensor linear_(const Tensor& x, const std::string& wKey, const std::string& bKey);
        Tensor layerNorm_(const Tensor& x, const std::string& wKey, const std::string& bKey,
                          float eps = 1e-5f);
        Tensor softmaxLast_(const Tensor& x);
        Tensor gelu_(const Tensor& x);
        Tensor transposeCHW2Tokens_(const Tensor& x);
        Tensor transposeTokens2CHW_(const Tensor& x, uint32_t C, uint32_t H, uint32_t W);
        Tensor attnScores_(const Tensor& qkv, uint32_t H);
        Tensor attnApply_(const Tensor& qkv, const Tensor& attn, uint32_t H);
        Tensor aifi_(const Tensor& x);

        // ── Phase C: CCFM neck (reuses conv/concat/upsample/add/add_silu) ────
        Tensor repC3_(const Tensor& x, const std::string& prefix, int n);
        struct NeckFeatures { Tensor s3, s4, s5; };
        NeckFeatures ccfm_(const Tensor& p3, const Tensor& p4, const Tensor& f5);

        // ── Phase D: deformable decoder + heads ──────────────────────────────
        Tensor relu_(const Tensor& x);
        Tensor inputProj_(const Tensor& x, int scaleIdx);
        // Writes a masked [C,H,W]->[H*W,C] transpose for one level into `out` at
        // a token offset (out must hold all levels). Used to assemble memory.
        void   buildMemoryLevel_(const Tensor& p, VkBuffer out, uint32_t outTokenOffset, uint32_t level);
        Tensor msDeformAttn_(const Tensor& value,
                             const std::vector<std::pair<uint32_t, uint32_t>>& spatialShapes,
                             const Tensor& refPts, const Tensor& samplingOffsets,
                             const Tensor& attnWeights, uint32_t numHeads);

        struct DecoderOut { std::vector<float> scores, bboxes; };
        // Runs the 6-layer decoder. memory/encOutput/encScores arrive on the host
        // (the decoder interleaves GPU bursts with CPU top-K + iterative bbox
        // refinement, so it cannot be a single batched frame). Returns the final
        // class logits [300,80] and refined boxes [300,4].
        DecoderOut decoder_(const std::vector<float>& memoryCpu, uint32_t totalTokens,
                            const std::vector<float>& encOutCpu,
                            const std::vector<float>& encScoresCpu,
                            const std::vector<std::pair<uint32_t, uint32_t>>& spatialShapes);
    };

}// namespace rtdetr
