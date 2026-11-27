#pragma once

#include "VkInfer.hpp"

#include "threepp/renderers/VulkanRenderer.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace rfdetr {

    struct Keypoint {
        float x, y;// image pixels
        float conf;// sigmoid of the keypoint's findable logit
    };

    struct Detection {
        int   classId;
        float confidence;
        float x1, y1, x2, y2;
        // RfDetrVariant::Keypoint: the class's keypoints in schema order (COCO-17
        // for a person); empty for the detection variants.
        std::vector<Keypoint> keypoints;
    };

    enum class RfDetrVariant { Nano, Small, Medium, Keypoint };

    /// Per-variant architecture config. Every variant shares the
    /// dinov2_windowed_small backbone (384-d, 6 heads, 12 layers) and a single-scale
    /// P4 projector; Nano/Small/Medium differ only in input resolution and decoder
    /// depth. Keypoint (RF-DETR Keypoint, pose estimation) patchifies at 12, keeps
    /// 100 queries over 2 classes (class 0 and person) and adds GroupPose keypoint
    /// queries: numKeypoints per instance, refined by every decoder layer against
    /// a second projector's features. forVariant() pins the numbers from the
    /// rfdetr package (rfdetr/config.py).
    struct RfDetrConfig {
        int resolution = 384, patch = 16;
        int embedDim = 384, numLayers = 12, numHeads = 6, numWindows = 2;
        int hiddenDim = 256, numQueries = 300, decLayers = 2;
        int saHeads = 8, caHeads = 16, decPoints = 2, dimFF = 2048, numClasses = 91;
        int numKeypoints = 0;// keypoint queries per instance; 0 = detection only
        std::array<int, 4> tapLayers{2, 5, 8, 11};

        [[nodiscard]] uint32_t grid() const { return uint32_t(resolution / patch); }
        [[nodiscard]] uint32_t tokensPerWindow() const { uint32_t s = grid() / uint32_t(numWindows); return s * s + 1u; }
        [[nodiscard]] uint32_t maxTokens() const { return uint32_t(numWindows * numWindows) * tokensPerWindow(); }

        static RfDetrConfig forVariant(RfDetrVariant v);
    };

    /// RF-DETR object detector on the Vulkan compute pipeline. Port of the
    /// LWDETR / DINOv2-windowed-ViT model (rfdetr package): the windowed ViT
    /// backbone (patch-embed + CLS token + 12 transformer layers w/ LayerScale,
    /// 2x2 windowed attention with global layers {3,6,9}, no register tokens),
    /// the C2f projector (tap layers 3/6/9/final -> single P4), and a two-stage
    /// deformable decoder (L=1, reparam bbox). The Keypoint variant adds the
    /// pose branch: a second projector on the same taps, keypoint queries per
    /// instance (group self-attention with their instance, deformable
    /// cross-attention into the second projector, FFN) and a keypoint head.
    class RfDetrVk {
    public:
        explicit RfDetrVk(threepp::VulkanRenderer& renderer,
                          RfDetrVariant variant = RfDetrVariant::Nano);
        ~RfDetrVk();
        RfDetrVk(const RfDetrVk&) = delete;
        RfDetrVk& operator=(const RfDetrVk&) = delete;

        void loadWeights(const std::string& path);

        // Validation: the forward pass on a [3,R,R] CHW input (R = the variant's
        // resolution), with the stages the reference captures read back.
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
            // Keypoint variant only (empty otherwise):
            std::vector<float>     projKp0;      // second (keypoint) projector output [256,G,G]
            std::vector<float>     decHs;        // decoder output after the final norm [Nq,256]
            std::vector<float>     decRef;       // decoder reference boxes [Nq,4]
            std::vector<float>     decKp;        // last layer's keypoint states [Nq,K,256]
            std::vector<float>     predKeypoints;// [Nq,K,8]: x, y (normalized), findable, visible,
                                                 // log_l11, l21, log_l22, class-logit contribution
        };
        ForwardOut runForward(const std::vector<float>& chw, bool captureIntermediates = true);

        // End-to-end detection on an RGBA image: ImageNet preprocess (square resize
        // to the variant's resolution, /255, normalize), forward pass and heads run
        // on the GPU as one submit; the host then applies sigmoid + score-threshold
        // + box de-normalization to the original image pixels. classId is the COCO id.
        // The Keypoint variant postprocesses like rfdetr's PostProcess: the top
        // num-queries (query, class) pairs, each score fused with the keypoints'
        // predicted localisation uncertainty, then the threshold. DETR has no NMS:
        // below ~0.45 a half-visible person can come back as many overlapping
        // detections.
        std::vector<Detection> infer(const unsigned char* rgba, int width, int height,
                                     float scoreThresh = 0.5f);
        // infer()'s host postprocess on a forward pass's heads, for an image of
        // width x height pixels (the validator runs it on the reference input).
        [[nodiscard]] std::vector<Detection> postprocess(const ForwardOut& fw, int width, int height,
                                                         float scoreThresh) const;

        [[nodiscard]] const RfDetrConfig& config() const { return cfg_; }

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
        VkPipe preprocessPipe_, proposalsPipe_, topkRankPipe_, queryRefsPipe_;
        VkPipe repeatRowsPipe_, groupPackPipe_, groupUnpackPipe_;// keypoint decoder
        // Fused ViT attention (created when VkInfer::subgroupClustered() and head dim 64;
        // else scores + softmax + apply).
        VkPipe attnFlashPipe_;
        bool useFlashAttn_ = false;

        // Keypoint variant, from the checkpoint's keypoint schema (_kp_active_mask
        // [classes with a schema slot, max K]): the one class that has keypoints,
        // and how many classes have a slot (the postprocess fuses the scores of all
        // of them; a slot without keypoints adds no uncertainty).
        int kpClass_ = -1;
        int kpSchemaClasses_ = 0;

        // Scratch for split-K partial sums [splitK, M, N], reused across all
        // split-K linears (serialized by VkInfer's per-dispatch barriers).
        VkTensor splitKPartials_;
        uint64_t splitKPartialFloats_ = 0;// capacity of splitKPartials_ in floats

        // Deformable-attention spatial_shapes [grid,grid] + level_start [0], constant
        // for the single-scale P4 decoder — uploaded once (ctor), not per layer.
        // msShapeBuf_ doubles as the offset preprocess's level shape.
        VkTensor msShapeBuf_, msStartBuf_;
        // Sine-embedding frequencies 10000^(2*floor(k/2)/128), k < 128 (ctor).
        VkTensor dimtBuf_;
        // The RGBA image infer() uploads; grows with the largest image seen.
        VkTensor rgbaInput_;

        std::unordered_map<std::string, VkTensor> weights_;

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
        // B independent attentions of T tokens, H heads of dim d, from a fused qkv
        // [B*T, 3*H*d] (the ViT's windows, the keypoint decoder's groups): scores [B,H,T,T].
        Tensor attnVitScores_(const Tensor& qkv, uint32_t T, uint32_t B, uint32_t H, uint32_t d);
        // out[b,i,h*d+e] = Σ_j attn[b,h,i,j] * V (qkv offset 2*H*d).
        Tensor attnVitApply_(const Tensor& attn, const Tensor& qkv, uint32_t T, uint32_t B, uint32_t H, uint32_t d);
        // Both in one pass with an online softmax (attn_vit_flash.comp): out [B*T, D].
        Tensor attnVitFlash_(const Tensor& qkv, uint32_t T, uint32_t B);
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
        // Projector `pfx` (backbone.0.projector, or the Keypoint variant's
        // backbone.0.cross_attn_projector) on the concatenated taps [4C,grid,grid]:
        // C2f -> final channel LayerNorm -> [256,grid,grid].
        Tensor projector_(const Tensor& tapsCat, const std::string& pfx);

        // ── decoder primitives ──
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
        // [R,W] -> [R*K,W]: row r*K+k is row r (a per-instance row per keypoint query).
        Tensor repeatRows_(const Tensor& x, uint32_t K);
        // The Keypoint variant's sub-layers of decoder layer `layer`, after the
        // instance FFN (GroupPose): group self-attention of each instance with its
        // K keypoint queries, keypoint deformable cross-attention into kpMemory
        // around the instance's reference box, keypoint FFN. Updates the instance
        // states tgt [Nq,256] and the keypoint states kp [Nq*K,256]. qpK, refK and
        // refXyK are query_pos [Nq,256] and the reference [Nq,4] / its centre [Nq,2]
        // repeated over the K keypoints.
        void keypointLayer_(int layer, Tensor& tgt, Tensor& kp, const Tensor& qpK, const Tensor& refK,
                            const Tensor& refXyK, const Tensor& kpMemory);
        // ── one inference = one frame ──
        // The backbone stages the validation path reads back, and the decoder's
        // memory tokens [G*G,256] (kpMemory: the Keypoint variant's second projector).
        struct BackboneTensors {
            Tensor patch, embed, vit0, vit3, proj0, projKp0;
            std::array<Tensor, 4> taps;
            Tensor memory, kpMemory;
        };
        // What the host needs after the frame: class logits [Nq,NC], the box head's
        // deltas [Nq,4] and the fixed decoder reference [Nq,4] they apply to; the
        // Keypoint variant's keypoint head output kpOut [Nq*K,8]. hs and kp (the
        // final instance / keypoint states) are read back only for validation.
        struct HeadTensors {
            Tensor logits, bboxDelta, ref, kpOut, hs, kp;
        };
        [[nodiscard]] VkDeviceSize headReadbackBytes_() const;
        // Patch embed -> 12 ViT blocks -> taps -> projector(s), recorded into the
        // open frame; fills keep, including the decoder memory tokens.
        void recordBackbone_(const Tensor& input, BackboneTensors& keep);
        // Two-stage deformable decoder + heads, recorded into the open frame:
        // first-stage heads over every memory token, top-Nq proposals, the fixed
        // reference + its sine embedding, the decoder layers, the final heads.
        HeadTensors recordDecoder_(const BackboneTensors& bb);
        // Records backbone + decoder behind `input` (already in the open frame),
        // ends the frame and assembles ForwardOut from the readbacks.
        ForwardOut finishForward_(const Tensor& input, bool captureIntermediates);
    };

}// namespace rfdetr
