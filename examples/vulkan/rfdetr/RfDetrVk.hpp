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
        // tensorCores: where the device has fp16 tensor cores (VkInfer::coopMatF16()), the
        // GEMMs, convolutions and attention take their products in fp16 with fp32 sums,
        // ~2x faster at ~1e-3 relative error (confidences within ~0.002). false keeps
        // every product in fp32, matching PyTorch's fp32 forward to ~1e-5.
        explicit RfDetrVk(threepp::VulkanRenderer& renderer,
                          RfDetrVariant variant = RfDetrVariant::Nano, bool tensorCores = true);
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
        // on the GPU as one submit (recorded once per image size, then submitted
        // again); the host then applies sigmoid + score-threshold + box
        // de-normalization to the original image pixels. classId is the COCO id.
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
        // Whether the matrix products run on fp16 tensor cores (see the constructor).
        [[nodiscard]] bool tensorCores() const { return tensorCores_; }

    private:
        VkInfer vk_;
        RfDetrConfig cfg_;
        bool tensorCores_ = false;
        // gemm.comp's builds, [B layout (GemmB)][tile: 128x64, 64x64], and its split-K
        // reduction.
        std::array<std::array<VkPipe, 2>, 3> gemmPipe_{};
        // gemm_cm.comp's tensor-core builds, [B layout (GemmB)] (created when
        // VkInfer::coopMatF16()).
        std::array<VkPipe, 3> gemmCmPipe_{};
        VkPipe gemmReducePipe_;
        VkPipe layerNormPipe_, softmaxPipe_;
        VkPipe attnScoresPipe_, attnApplyPipe_, addPipe_;
        VkPipe buildEmbeddingsPipe_;
        VkPipe attnVitScoresPipe_, attnVitApplyPipe_;
        VkPipe unwindowPipe_, lnChwPipe_;
        VkPipe transposeT2tPipe_, offsetPreprocessPipe_, msDeformAttnPipe_;
        VkPipe preprocessPipe_, proposalsPipe_, topkRankPipe_, queryRefsPipe_;
        VkPipe repeatRowsPipe_, groupPackPipe_, groupUnpackPipe_;// keypoint decoder
        // Fused ViT attention (created when VkInfer::subgroupClustered() and head dim 64;
        // else scores + softmax + apply).
        VkPipe attnFlashPipe_;
        // The decoders' head-dim-32 self-attention in one pass on tensor cores
        // (attn_vit_flash_cm.comp built with HD=32); null without them.
        VkPipe attnCm32Pipe_;
        bool useFlashAttn_ = false;

        // Keypoint variant, from the checkpoint's keypoint schema (_kp_active_mask
        // [classes with a schema slot, max K]): the one class that has keypoints,
        // and how many classes have a slot (the postprocess fuses the scores of all
        // of them; a slot without keypoints adds no uncertainty).
        int kpClass_ = -1;
        int kpSchemaClasses_ = 0;

        // Scratch for split-K partial sums [splitK, M, N], reused by every split
        // GEMM (serialized by VkInfer's per-dispatch barriers).
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
        // fp16 copies of the 2D weights for the tensor-core GEMMs (when VkInfer::coopMatF16()).
        std::unordered_map<std::string, VkTensor> weights16_;

        // ── GEMM ──
        // gemm.comp's B operand: an nn.Linear weight [N,K], a 1x1 conv's CHW input
        // [K,N], or a kxk conv's implicit im2col.
        enum class GemmB { NK = 0, KN = 1, Conv = 2 };
        // Y = epi(A·B): A [M,K] row-major; epi = act(· + bias), then res + gamma ⊙ that
        // when res is set (gamma optional). Strides and offsets are in floats.
        struct Gemm {
            GemmB layout = GemmB::NK;
            VkBuffer a = VK_NULL_HANDLE, b = VK_NULL_HANDLE, y = VK_NULL_HANDLE;
            VkBuffer b16 = VK_NULL_HANDLE;// fp16 copy of an NK weight: allows the tensor-core GEMM
            VkBuffer a16 = VK_NULL_HANDLE;// fp16 copy of a conv weight (A of KN / Conv): the same
            uint32_t M = 0, N = 0, K = 0;
            uint32_t lda = 0, ldb = 0, ldc = 0;
            uint32_t aOff = 0, bOff = 0, cOff = 0;
            VkBuffer bias = VK_NULL_HANDLE;
            uint32_t biasOff = 0;
            bool biasPerRow = false;// bias[m] (a conv's output channel) instead of bias[n]
            uint32_t act = 0;// 1 = GELU, 2 = ReLU
            VkBuffer res = VK_NULL_HANDLE, gamma = VK_NULL_HANDLE;
            uint32_t ldr = 0, rOff = 0;
            uint32_t inH = 0, inW = 0, outW = 0, kh = 1, kw = 1, stride = 1, pad = 0;// GemmB::Conv
        };
        // Records g, choosing the tile and the K split for its shape.
        void gemm_(const Gemm& g);

        // ── ops ──
        // conv (bias optional, no groups) as one GEMM over x's channels from inOff
        // (the weight gives how many); a fresh [out_c, out_h, out_w] output.
        Tensor conv_(const Tensor& x, const std::string& wKey, const std::string& bKey,
                     int stride, int pad, uint32_t inOff = 0);
        Tensor patchEmbed_(const Tensor& x);
        Tensor buildEmbeddings_(const Tensor& patchEmbed);

        // ── ViT primitives (token buffer is flat [M, D] window-major) ──
        // act(x·Wᵀ + b) for act 0 = none, 1 = GELU, 2 = ReLU; with res, the residual
        // res + gamma ⊙ that (gamma a LayerScale key, or none), all in the GEMM's epilogue.
        Tensor linear_(const Tensor& x, const std::string& wKey, const std::string& bKey, uint32_t act = 0,
                       const Tensor* res = nullptr, const std::string& gammaKey = "");
        // a·Wᵀ + b for weight (and bias) rows [n0, n0+n) only, into columns
        // [cOff, cOff+n) of y [M, ldc] (ldc = y.shape.back()): the parts of a fused
        // in_proj that one input needs, written straight into the qkv layout.
        void linearRowsInto_(const Tensor& a, const std::string& wKey, const std::string& bKey,
                             uint32_t n0, uint32_t n, const Tensor& y, uint32_t cOff);
        Tensor layerNorm_(const Tensor& x, const std::string& wKey, const std::string& bKey, uint32_t D);
        Tensor add_(const Tensor& a, const Tensor& b);
        Tensor softmaxRows_(const Tensor& x, uint32_t rows, uint32_t n);
        // B independent attentions of T tokens, H heads of dim d, from a fused qkv
        // [B*T, 3*H*d] (the ViT's windows, the keypoint decoder's groups): scores [B,H,T,T].
        Tensor attnVitScores_(const Tensor& qkv, uint32_t T, uint32_t B, uint32_t H, uint32_t d);
        // out[b,i,h*d+e] = Σ_j attn[b,h,i,j] * V (qkv offset 2*H*d).
        Tensor attnVitApply_(const Tensor& attn, const Tensor& qkv, uint32_t T, uint32_t B, uint32_t H, uint32_t d);
        // Both in one pass with an online softmax (attn_vit_flash.comp): out [B*T, D].
        Tensor attnVitFlash_(const Tensor& qkv, uint32_t T, uint32_t B);
        // B groups of T tokens, H heads of dim 32 (the decoders), in one pass: out [B*T, H*32].
        // Needs attnCm32Pipe_.
        Tensor attnHead32_(const Tensor& qkv, uint32_t T, uint32_t B, uint32_t H);
        // One DINOv2 transformer block. global=true runs full attention (B=1,T=M).
        Tensor vitLayer_(const Tensor& x, int layerIdx, bool global);

        // ── projector primitives (CHW feature maps [C, HW]) ──
        // Channel-wise (ConvNeXt) LayerNorm; act 0=none, 2=SiLU fused on output.
        // Writes channels [outOff, outOff+C) of out (a fresh [C,H,W] when out is null).
        Tensor layerNormChw_(const Tensor& x, const std::string& gKey, const std::string& bKey,
                             uint32_t C, uint32_t HW, uint32_t act,
                             const Tensor* out = nullptr, uint32_t outOff = 0);
        // De-window an LN'd hidden [numWin²,tokens,C] into channels [chOffset, chOffset+C)
        // of the [*,grid,grid] feature map dst.
        void unwindowInto_(const Tensor& lnHidden, const Tensor& dst, uint32_t chOffset);
        // ConvX = conv(bias-free) on x's channels from inOff -> channel LayerNorm
        // (+optional SiLU) into channels [outOff, ...) of out (fresh when null).
        Tensor convX_(const Tensor& x, const std::string& pfx, int k, uint32_t HW, bool silu,
                      uint32_t inOff = 0, const Tensor* out = nullptr, uint32_t outOff = 0);
        // C2f: cv1 -> 3 chained bottlenecks -> cv2, with every branch written in place
        // into the concat buffer cv2 reads.
        Tensor c2f_(const Tensor& x, const std::string& pfx, uint32_t HW);
        // Projector `pfx` (backbone.0.projector, or the Keypoint variant's
        // backbone.0.cross_attn_projector) on the concatenated taps [4C,grid,grid]:
        // C2f -> final channel LayerNorm -> [256,grid,grid].
        Tensor projector_(const Tensor& tapsCat, const std::string& pfx);

        // ── decoder primitives ──
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
            Tensor tapsCat;// the four taps [4C,grid,grid], tap t at channel t*C
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
        // What a recorded forward pass leaves for the host: its tensors and where the
        // frame's readbacks of the heads land.
        struct Recorded {
            BackboneTensors bb;
            HeadTensors heads;
            VkDeviceSize offLogits = 0, offDelta = 0, offRef = 0, offKp = 0;
        };
        // Records backbone + decoder behind `input` (already in the open frame) and
        // the head readbacks.
        Recorded recordForward_(const Tensor& input);
        // ForwardOut's heads from a finished frame's readbacks.
        [[nodiscard]] ForwardOut readHeads_(const Recorded& r) const;
        // recordForward_, end the frame, readHeads_ (+ the validation readbacks).
        ForwardOut finishForward_(const Tensor& input, bool captureIntermediates);
        // infer()'s frame, replayed while the image size and every buffer it was
        // recorded against stay the same (VkInfer::replayable).
        Recorded inferRec_;
        VkDeviceSize inferUploadOff_ = 0;
    };

}// namespace rfdetr
