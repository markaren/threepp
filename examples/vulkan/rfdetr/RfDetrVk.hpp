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

    enum class RfDetrVariant { Nano, Small, Medium };

    /// Per-variant architecture config. Nano/Small/Medium share the
    /// dinov2_windowed_small backbone (384-d, 6 heads, 12 layers) and a single-scale
    /// P4 projector; they differ only in input resolution and decoder depth.
    /// forVariant() pins the numbers from the rfdetr package (rfdetr/config.py).
    struct RfDetrConfig {
        int resolution = 384, patch = 16;
        int embedDim = 384, numLayers = 12, numHeads = 6, numWindows = 2;
        int hiddenDim = 256, numQueries = 300, decLayers = 2;
        int saHeads = 8, caHeads = 16, decPoints = 2, dimFF = 2048, numClasses = 91;
        std::array<int, 4> tapLayers{2, 5, 8, 11};

        [[nodiscard]] uint32_t grid() const { return uint32_t(resolution / patch); }
        [[nodiscard]] uint32_t tokensPerWindow() const { uint32_t s = grid() / uint32_t(numWindows); return s * s + 1u; }
        [[nodiscard]] uint32_t maxTokens() const { return uint32_t(numWindows * numWindows) * tokensPerWindow(); }

        static RfDetrConfig forVariant(RfDetrVariant v);
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
        explicit RfDetrVk(threepp::VulkanRenderer& renderer,
                          RfDetrVariant variant = RfDetrVariant::Nano);
        ~RfDetrVk();
        RfDetrVk(const RfDetrVk&) = delete;
        RfDetrVk& operator=(const RfDetrVk&) = delete;

        void loadWeights(const std::string& path);

        // Phase-1 validation: run the patch embedding (+ backbone as wired so far)
        // on a [3,384,384] CHW input and read back the projector P4 feature.
        struct ForwardOut {
            std::vector<float>     patchEmbed;// [EMBED_DIM, GRID, GRID]
            std::array<uint32_t, 3> patchDim{};
            std::vector<float>     embed;// [W*W, 1+T, EMBED_DIM] windowed tokens
            std::array<uint32_t, 3> embedDim{};
            std::vector<float>     vit0;// hidden states after ViT layer 0  [W*W,1+T,C]
            std::vector<float>     vit3;// hidden states after ViT layer 3  [W*W,1+T,C]
            std::array<uint32_t, 3> vitDim{};
            std::array<std::vector<float>, 4> taps;// de-windowed tap features [C,G,G]
            std::array<uint32_t, 3> tapDim{};
            std::vector<float>     proj0;// projector P4 output [256,G,G]
            std::array<uint32_t, 3> proj0Dim{};
            std::vector<float>     predLogits;// [NUM_QUERIES, NUM_CLASSES]
            std::vector<float>     predBoxes; // [NUM_QUERIES, 4] (cx,cy,w,h normalized)
        };
        ForwardOut runForward(const std::vector<float>& chw, bool captureIntermediates = true);

        // End-to-end detection on an RGBA image: ImageNet preprocess (square resize
        // to 384, /255, normalize) -> forward -> sigmoid + score-threshold + box
        // de-normalization to the original image pixels. classId is the COCO id.
        std::vector<Detection> infer(const unsigned char* rgba, int width, int height,
                                     float scoreThresh = 0.5f);

    private:
        VkInfer vk_;
        RfDetrConfig cfg_;
        VkPipe convPipe_, conv1x1Pipe_, conv3x3Pipe_, im2colPipe_;
        VkPipe linearPipe_, layerNormPipe_, softmaxPipe_, geluPipe_, layerScalePipe_;
        VkPipe attnScoresPipe_, attnApplyPipe_, addPipe_, concatPipe_;
        VkPipe buildEmbeddingsPipe_;
        VkPipe attnVitScoresPipe_, attnVitApplyPipe_;
        VkPipe unwindowPipe_, lnChwPipe_, chCopyPipe_;
        VkPipe transposeT2tPipe_, qkvSplicePipe_, reluPipe_, offsetPreprocessPipe_, msDeformAttnPipe_;
        VkPipe scaleAddPipe_;
        VkPipe linearSplitKPipe_, reduceSplitKPipe_;

        // Scratch for split-K partial sums [splitK, M, N], reused across all
        // split-K linears (serialized by VkInfer's per-dispatch barriers).
        VkTensor splitKPartials_;
        uint64_t splitKPartialFloats_ = 0;// capacity of splitKPartials_ in floats
        bool useSplitK_ = true;// off via env RF_NOSPLITK=1 for A/B baseline

        // Deformable-attention spatial_shapes [grid,grid] + level_start [0], constant
        // for the single-scale P4 decoder — uploaded once (ctor), not per layer.
        VkTensor msShapeBuf_, msStartBuf_;

        std::unordered_map<std::string, VkTensor> weights_;
        std::vector<float> learnedRefCpu_;// refpoint_embed[:NUM_QUERIES] cached at load

        // ── ops ──
        Tensor conv_(const Tensor& x, const std::string& wKey, const std::string& bKey,
                     int strideH, int strideW, int padH, int padW);
        Tensor patchEmbed_(const Tensor& x);
        Tensor buildEmbeddings_(const Tensor& patchEmbed);

        // ── ViT primitives (token buffer is flat [M, D] window-major) ──
        // act: 0 = none, 1 = GELU fused on the GEMM output (saves a round-trip).
        Tensor linear_(const Tensor& x, const std::string& wKey, const std::string& bKey, uint32_t act = 0);
        Tensor layerNorm_(const Tensor& x, const std::string& wKey, const std::string& bKey, uint32_t D);
        Tensor gelu_(const Tensor& x);
        Tensor layerScale_(const Tensor& x, const std::string& gammaKey, uint32_t C);
        Tensor add_(const Tensor& a, const Tensor& b);
        // Fused residual: out = residual + branch * gamma (LayerScale folded in).
        Tensor scaleAdd_(const Tensor& residual, const Tensor& branch, const std::string& gammaKey, uint32_t C);
        Tensor softmaxRows_(const Tensor& x, uint32_t rows, uint32_t n);
        // Batched DINOv2 attention from a fused qkv [M,3D]: scores [B,H,T,T].
        Tensor attnVitScores_(const Tensor& qkv, uint32_t T, uint32_t B);
        // out[b,i,h*d+e] = Σ_j attn[b,h,i,j] * V (qkv offset 2D).
        Tensor attnVitApply_(const Tensor& attn, const Tensor& qkv, uint32_t T, uint32_t B);
        // One DINOv2 transformer block. global=true runs full attention (B=1,T=M).
        Tensor vitLayer_(const Tensor& x, int layerIdx, bool global);

        // ── projector primitives (CHW feature maps [C, HW]) ──
        // Channel-wise (ConvNeXt) LayerNorm; act 0=none, 2=SiLU fused on output.
        Tensor layerNormChw_(const Tensor& x, const std::string& gKey, const std::string& bKey,
                             uint32_t C, uint32_t HW, uint32_t act);
        // De-window an LN'd hidden [numWin²,tokens,C] into a [C,grid,grid] feature map.
        Tensor unwindow_(const Tensor& lnHidden, uint32_t outC, uint32_t grid);
        // Copy Cn channels of src (from srcOff) into a fresh [Cn,HW] buffer.
        Tensor sliceCh_(const Tensor& src, uint32_t srcOff, uint32_t Cn, uint32_t HW);
        // Place Cn channels of src into dst starting at channel dstOff (concat).
        void placeCh_(const Tensor& src, const Tensor& dst, uint32_t Cn, uint32_t HW, uint32_t dstOff);
        // ConvX = conv(bias-free) -> channel LayerNorm (+optional SiLU). kernel k (1 or 3).
        Tensor convX_(const Tensor& x, const std::string& pfx, int k, uint32_t HW, bool silu);
        Tensor bottleneck_(const Tensor& x, const std::string& pfx, uint32_t HW);
        Tensor c2f_(const Tensor& x, const std::string& pfx, uint32_t HW);
        // Full projector: concat 4 taps -> C2f -> final channel LayerNorm -> [256,grid,grid].
        Tensor projector_(const std::array<Tensor, 4>& taps, uint32_t grid);

        // ── decoder primitives ──
        Tensor uploadArena_(const std::vector<uint32_t>& shape, const float* data);
        Tensor relu_(const Tensor& x);
        Tensor transposeChw2Tokens_(const Tensor& x);// [C,H,W] -> [H*W,C]
        Tensor bboxMLP_(const Tensor& x, const std::string& pfx);// 3-layer MLP, relu between
        // Combined-QKV self-attention scores [H,M,M] and apply [M,D] (qkv is [M,3D]).
        Tensor attnScores_(const Tensor& qkv, uint32_t M, uint32_t H);
        Tensor attnApply_(const Tensor& qkv, const Tensor& attn, uint32_t M, uint32_t H);
        // Deformable cross-attention; refPtsXy [Nq,2], offsets [Nq,H*L*P*2] (already
        // pixel-scaled), attnW [Nq,H*L*P]. Single feature level (proj0 = P4).
        Tensor msDeformAttn_(const Tensor& value, uint32_t H, uint32_t W,
                             const Tensor& refPtsXy, const Tensor& offsets, const Tensor& attnW,
                             uint32_t numHeads);
        // Two-stage deformable decoder + heads; memory is proj0 in [HW,256] token layout.
        // Fills out.predLogits / out.predBoxes. encClass/encBboxDelta read back from frame A.
        void decoder_(const Tensor& memory, uint32_t gridH, uint32_t gridW, ForwardOut& out);
    };

}// namespace rfdetr
