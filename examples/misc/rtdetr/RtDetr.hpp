#pragma once

#include "GPUTensor.hpp"
#include "WeightLoader.hpp"

#include "threepp/renderers/wgpu/WgpuComputePipeline.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rtdetr {

    /// Activation functions supported by the conv kernels.
    /// (ReLU is the HGNet default; SiLU reserved for future use; GELU lives
    /// in the transformer FFN and will get its own kernel.)
    enum class Activation : uint32_t {
        None = 0,
        ReLU = 1,
        SiLU = 2,
    };

    /// RT-DETR-L object detector running on the WebGPU compute pipeline.
    /// Load pre-trained weights exported by scripts/export_rtdetr_weights.py,
    /// then call infer() with any RGBA image.
    ///
    /// Milestone 1: weight loading + BN-fold + fp16 packing.
    /// Milestone 2: DWConv kernel (depthwise convolution).
    /// Subsequent milestones add backbone, encoder, decoder, etc.
    class RtDetr {
    public:
        static constexpr int   INPUT_SIZE  = 640;
        static constexpr int   NUM_CLASSES = 80;
        static constexpr float BN_EPS      = 1e-3f;

        explicit RtDetr(threepp::WgpuRenderer& renderer);
        ~RtDetr();

        RtDetr(const RtDetr&) = delete;
        RtDetr& operator=(const RtDetr&) = delete;

        void loadWeights(const std::string& path);

        [[nodiscard]] size_t numWeights() const { return weights_.size(); }
        [[nodiscard]] std::vector<std::string> fusedKeys() const {
            std::vector<std::string> out;
            for (const auto& kv : weights_) {
                const auto& k = kv.first;
                if (k.size() >= 13 && k.compare(k.size() - 13, 13, ".fused.weight") == 0)
                    out.push_back(k);
                else if (k.size() >= 11 && k.compare(k.size() - 11, 11, ".fused.bias") == 0)
                    out.push_back(k);
            }
            std::sort(out.begin(), out.end());
            return out;
        }
        [[nodiscard]] std::vector<std::string> weightKeysStartingWith(const std::string& prefix) const {
            std::vector<std::string> out;
            for (const auto& kv : weights_) {
                if (kv.first.rfind(prefix, 0) == 0) out.push_back(kv.first);
            }
            std::sort(out.begin(), out.end());
            return out;
        }
        [[nodiscard]] bool   hasWeight(const std::string& key) const { return weights_.count(key) > 0; }
        [[nodiscard]] const std::vector<uint32_t>& weightShape(const std::string& key) const {
            return weights_.at(key).shape;
        }

        /// Run a depthwise conv using an already-loaded fused weight.
        /// Weight shape: [C, 1, kH, kW]. Bias shape: [C] (required).
        /// `x` must have x.C() == weight.shape[0].
        GPUTensor dwConv_(const GPUTensor& x,
                          const std::string& weightKey,
                          const std::string& biasKey,
                          int strideH, int strideW,
                          int padH, int padW,
                          Activation act = Activation::None);

        /// Run a standard (non-grouped) 2D conv using an already-loaded fused weight.
        /// Weight shape: [out_c, in_c, kH, kW]. Bias shape: [out_c] (optional: pass ""
        /// as biasKey to skip). `x` must have x.C() == weight.shape[1].
        GPUTensor conv_(const GPUTensor& x,
                        const std::string& weightKey,
                        const std::string& biasKey,
                        int strideH, int strideW,
                        int padH, int padW,
                        Activation act = Activation::None);

        /// Conv2D with asymmetric padding (PyTorch F.pad style: top/left/bottom/right).
        GPUTensor conv_(const GPUTensor& x,
                        const std::string& weightKey,
                        const std::string& biasKey,
                        int strideH, int strideW,
                        int padTop, int padLeft, int padBottom, int padRight,
                        Activation act = Activation::None);

        /// HGStem block (model.0). Input [3, H, W] → output [48, H/4, W/4].
        GPUTensor hgStem_(const GPUTensor& x);

        /// Generic HGBlock: n parallel-dense 3×3 (or k×k) convs feeding each
        /// other, concat {x, m.0, m.1, ..., m.{n-1}}, then sc (1×1 squeeze),
        /// ec (1×1 expand). Optional residual add. Plain Conv variant only
        /// (lightconv/DWConv variant is a separate helper for later stages).
        GPUTensor hgBlock_(const GPUTensor& x,
                           const std::string& prefix,
                           int n, int k = 3, bool shortcut = false,
                           bool lightconv = false);

        /// Elementwise add (in-place into a fresh tensor; a and b must match shape).
        GPUTensor addTensor_(const GPUTensor& a, const GPUTensor& b);

        /// PyTorch-style linear: Y[M,N] = X[M,K] @ W[N,K]^T + b[N].
        /// `wKey` must be an [N, K] 2D weight; `bKey` may be "" to skip bias.
        /// Input x is interpreted as [M, K] (any leading-dim reshape is the
        /// caller's responsibility; shape metadata is preserved in output as
        /// [..., N] matching x's leading dims).
        GPUTensor linear_(const GPUTensor& x,
                          const std::string& wKey,
                          const std::string& bKey);

        /// LayerNorm along the last axis: y = (x - mean) / sqrt(var + eps) * gamma + beta.
        /// `wKey` = [D] scale, `bKey` = [D] shift.  x's last dim must equal D.
        GPUTensor layerNorm_(const GPUTensor& x,
                             const std::string& wKey,
                             const std::string& bKey,
                             float eps = 1e-5f);

        /// Row-wise numerically-stable softmax along the last axis.
        GPUTensor softmaxLast_(const GPUTensor& x);

        /// Elementwise exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
        GPUTensor gelu_(const GPUTensor& x);

        /// Elementwise ReLU: max(0, x).
        GPUTensor relu_(const GPUTensor& x);

        /// Compute multi-head attention scores (pre-softmax).
        /// `qkv` shape [M, 3*D] (output of a combined QKV linear). Returns
        /// [H, M, M] with values scaled by 1/sqrt(d) where d = D / H.
        GPUTensor attnScores_(const GPUTensor& qkv, uint32_t H);

        /// Apply attention: output [M, D], given attn weights [H, M, M] and
        /// the same combined `qkv` [M, 3*D]. Heads are concatenated.
        GPUTensor attnApply_(const GPUTensor& qkv,
                             const GPUTensor& attn,
                             uint32_t H);

        /// Full AIFI transformer encoder (module model.11).
        /// Input spatial tensor [D, H, W] → output [D, H, W] (same shape).
        /// Adds 2D sinusoidal positional embedding, runs pre-norm or post-norm
        /// as configured by ultralytics (RT-DETR uses post-norm).
        GPUTensor aifi_(const GPUTensor& x);

        /// Full HGNetv2 backbone + P5 1×1 projection.
        /// Input [3, H, W] (H, W multiples of 32). Returns {P3, P4, P5} with
        /// channels {512, 1024, 256} and strides {8, 16, 32}.
        struct BackboneFeatures { GPUTensor p3, p4, p5; };
        BackboneFeatures backbone_(const GPUTensor& x);

        /// LightConv: Conv1x1 (no activation, BN-folded) → DWConv k×k (ReLU).
        /// Ultralytics LightConv at <prefix>.conv1 / <prefix>.conv2.
        GPUTensor lightConv_(const GPUTensor& x,
                             const std::string& prefix, int k);

        /// Allocate a GPU tensor of the given shape and fill it with `value`.
        /// Intended for milestone 2 sanity tests.
        GPUTensor allocFilled(std::initializer_list<uint32_t> shape, float value);

        /// Allocate a GPU tensor and upload arbitrary CPU-side data.
        GPUTensor uploadTensor(std::initializer_list<uint32_t> shape, const float* data);

        /// 2D max-pool. Supports asymmetric padding (pad_{top,left,bottom,right}).
        /// Out-of-bounds positions contribute -inf (i.e. are ignored by max).
        GPUTensor maxPool_(const GPUTensor& x,
                           int kH, int kW,
                           int strideH, int strideW,
                           int padTop, int padLeft, int padBottom, int padRight);

        /// 2x nearest-neighbor upsample (H and W doubled).
        GPUTensor upsample2x_(const GPUTensor& x);

        /// Fused elementwise (a + b) -> SiLU.  a and b must have matching numel.
        GPUTensor addSilu_(const GPUTensor& a, const GPUTensor& b);

        /// RepC3 block: y = silu(cv1(x) + bottleneckChain) + cv2(x)?  Actually ultralytics
        /// RepC3: a = Conv1x1(cv1, silu); b = Conv1x1(cv2, silu); for n: a = silu(conv3x3(a)+conv1x1(a)); return a + b.
        GPUTensor repC3_(const GPUTensor& x, const std::string& prefix, int n);

        /// Flatten three projected feature maps [256, Hi, Wi] into the decoder's
        /// memory token stream [sum(Hi*Wi), 256] in ultralytics' row-major
        /// (N, H*W, C) layout, concatenated in order (low->high level).
        GPUTensor buildMemory_(const GPUTensor& p0, const GPUTensor& p1, const GPUTensor& p2);

        /// Apply RTDETRDecoder input_proj.i (1x1 Conv + BN, no activation) to
        /// a feature map. Returns a [256, H, W] tensor suitable for flattening
        /// into the decoder's multi-scale memory.
        GPUTensor inputProj_(const GPUTensor& x, int scaleIdx);

        /// Full CCFM neck: modules 12..27. Inputs are backbone P3/P4 and AIFI-processed P5.
        /// Returns {S3, S4, S5} feature maps fed to the decoder.
        struct NeckFeatures { GPUTensor s3, s4, s5; };
        NeckFeatures ccfm_(const GPUTensor& p3, const GPUTensor& p4, const GPUTensor& f5);

        /// Multi-Scale Deformable Attention (MSDeformAttn).
        /// Core cross-attention mechanism of the RT-DETR decoder.
        ///
        /// value:           [total_tokens, D]  multi-scale projected values (D = H*d)
        /// spatialShapes:   per-level (H, W) pairs
        /// refPts:          [Nq, 2] reference points in normalized [0,1] coords
        /// samplingOffsets: [Nq, H*L*P*2] predicted sampling offsets
        /// attnWeights:     [Nq, H*L*P] attention weights (must be softmaxed)
        /// Returns:         [Nq, D]
        GPUTensor msDeformAttn_(const GPUTensor& value,
                                const std::vector<std::pair<uint32_t,uint32_t>>& spatialShapes,
                                const GPUTensor& refPts,
                                const GPUTensor& samplingOffsets,
                                const GPUTensor& attnWeights,
                                uint32_t numHeads);

        /// A single detection result.
        struct Detection {
            int   classId;
            float confidence;
            float x1, y1, x2, y2;   ///< box in original image pixels
        };

        /// RTDETRDecoder output: final class scores and bounding boxes.
        struct DecoderOutput {
            std::vector<float> scores;  ///< [300 * 80] class logits
            std::vector<float> bboxes;  ///< [300 * 4]  normalized (cx, cy, w, h)
        };

        /// Run end-to-end inference on an RGBA image.
        /// Returns detections above confThresh after NMS.
        std::vector<Detection> infer(const unsigned char* rgba,
                                     int width, int height,
                                     float confThresh = 0.5f);

        /// Full RTDETRDecoder forward pass (model.28).
        /// Selects top-300 queries from enc_output, runs 6 decoder layers
        /// with self-attention, MSDeformAttn cross-attention, and FFN.
        /// Returns final class scores and iteratively-refined bounding boxes.
        DecoderOutput decoder_(const GPUTensor& memory,
                               const GPUTensor& encOutput,
                               const std::vector<std::pair<uint32_t,uint32_t>>& spatialShapes);

        /// GPU transpose: [C, H, W] → [H*W, C] token layout.
        GPUTensor transposeCHW2Tokens_(const GPUTensor& x);

        /// GPU transpose: [H*W, C] → [C, H, W] spatial layout.
        GPUTensor transposeTokens2CHW_(const GPUTensor& x, uint32_t C, uint32_t H, uint32_t W);

        /// Concatenate two tensors along the channel axis. a and b must have
        /// matching H and W. Output channel count = a.C() + b.C().
        GPUTensor concatC_(const GPUTensor& a, const GPUTensor& b);

        /// Copy a Storage buffer to CPU. Sized in floats.
        std::vector<float> readback(threepp::WgpuBuffer& buf, size_t floatCount);

        /// Access the CPU-side copy of a fused weight or bias, for offline
        /// verification against GPU output. Only "*.fused.weight" and
        /// "*.fused.bias" keys are populated.
        [[nodiscard]] const std::vector<float>& cpuWeight(const std::string& key) const {
            return cpuWeights_.at(key);
        }

    private:
        threepp::WgpuRenderer& renderer_;

        /// Weight tensors uploaded to GPU (keyed by PyTorch state_dict name,
        /// with BN-folded convs stored under "<prefix>.fused.weight/bias").
        std::unordered_map<std::string, GPUTensor> weights_;

        /// CPU-side copy of conv weight/bias tensors used for analytical
        /// test assertions. Keyed by the same names as weights_ but only
        /// populated for "*.fused.weight" and "*.fused.bias".
        std::unordered_map<std::string, std::vector<float>> cpuWeights_;

        threepp::WgpuComputePipeline dwConvPipe_;
        threepp::WgpuComputePipeline convPipe_;
        threepp::WgpuComputePipeline maxPoolPipe_;
        threepp::WgpuComputePipeline concatCPipe_;
        threepp::WgpuComputePipeline addPipe_;
        threepp::WgpuComputePipeline linearPipe_;
        threepp::WgpuComputePipeline layerNormPipe_;
        threepp::WgpuComputePipeline softmaxPipe_;
        threepp::WgpuComputePipeline geluPipe_;
        threepp::WgpuComputePipeline attnScoresPipe_;
        threepp::WgpuComputePipeline attnApplyPipe_;
        threepp::WgpuComputePipeline upsample2xPipe_;
        threepp::WgpuComputePipeline addSiluPipe_;
        threepp::WgpuComputePipeline msDeformAttnPipe_;
        threepp::WgpuComputePipeline reluPipe_;
        threepp::WgpuComputePipeline transposeCHW2TokensPipe_;
        threepp::WgpuComputePipeline transposeTokens2CHWPipe_;
        threepp::WgpuComputePipeline transposeMaskPipe_;
        threepp::WgpuComputePipeline qkvSplicePipe_;
        threepp::WgpuComputePipeline offsetPreprocessPipe_;
        threepp::WgpuBuffer          convParamBuf_;
        threepp::WgpuBuffer          poolParamBuf_;
        threepp::WgpuBuffer          concatParamBuf_;
        threepp::WgpuBuffer          addParamBuf_;
        threepp::WgpuBuffer          linearParamBuf_;
        threepp::WgpuBuffer          lnParamBuf_;
        threepp::WgpuBuffer          softmaxParamBuf_;
        threepp::WgpuBuffer          geluParamBuf_;
        threepp::WgpuBuffer          attnParamBuf_;
        threepp::WgpuBuffer          upsampleParamBuf_;
        threepp::WgpuBuffer          msDeformParamBuf_;
        threepp::WgpuBuffer          transposeParamBuf_;
        threepp::WgpuBuffer          transposeMaskParamBuf_;
        threepp::WgpuBuffer          qkvSpliceParamBuf_;
        threepp::WgpuBuffer          offsetPreprocessParamBuf_;
    };

}// namespace rtdetr
