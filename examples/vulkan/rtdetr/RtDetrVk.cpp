#include "RtDetrVk.hpp"

#include "WeightLoader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

// SPIR-V blobs generated from the .comp sources by compile_vulkan_shader
// (VARIANT_SUFFIX=rtdetr so they don't collide with the YOLO kernels of the
// same basename).
#include "threepp/renderers/vulkan/shaders/conv2d.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/dwconv.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/maxpool.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/concat.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/add.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/add_silu.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/upsample.comp.rtdetr.spv.h"
// Phase B: AIFI transformer encoder.
#include "threepp/renderers/vulkan/shaders/linear.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/layernorm.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/softmax.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/gelu.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/transpose_chw2tokens.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/transpose_tokens2chw.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/qkv_splice.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_scores.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_apply.comp.rtdetr.spv.h"
// Phase D: deformable decoder.
#include "threepp/renderers/vulkan/shaders/relu.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/transpose_mask.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/offset_preprocess.comp.rtdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/msdeformattn.comp.rtdetr.spv.h"

#include <algorithm>

using threepp::VulkanRenderer;

namespace rtdetr {

namespace {

    // Push-constant param structs — byte layout matches the GLSL push_constant
    // blocks (all 4-byte scalars, tightly packed under std430).
    struct ConvParams {
        uint32_t in_c, out_c, in_h, in_w;
        uint32_t out_h, out_w, k_h, k_w;
        uint32_t stride_h, stride_w, pad_h, pad_w;
        uint32_t has_bias, activation, _p0, _p1;
    };
    struct PoolParams {
        uint32_t c, in_h, in_w, out_h;
        uint32_t out_w, k_h, k_w, stride_h;
        uint32_t stride_w, pad_t, pad_l, _p0;
        uint32_t _p1, _p2, _p3, _p4;
    };
    struct ConcatParams { uint32_t c_a, c_b, h, w; };
    struct AddParams    { uint32_t n, _p0, _p1, _p2; };
    struct UpParams     { uint32_t C, inH, inW, _p; };
    // Phase B
    struct LinearParams   { uint32_t M, N, K, hasBias; };
    struct LNParams       { uint32_t M, D; float eps, _pad; };
    struct SoftmaxParams  { uint32_t M, N, _p0, _p1; };
    struct GeluParams     { uint32_t n, _p0, _p1, _p2; };
    struct TransposeParams{ uint32_t C, H, W, _p; };
    struct QkvSpliceParams{ uint32_t M, D, _p0, _p1; };
    struct AttnParams     { uint32_t M, H, d, stride; float scale; uint32_t _p0, _p1, _p2; };
    // Phase D
    struct MsDeformParams { uint32_t Nq, H, d, L, P, _p0, _p1, _p2; };
    struct OffPreParams   { uint32_t Nq, numHeads, numLevels, numPoints; };
    struct TMaskParams    { uint32_t C, H, W, outOffset, level, _p0, _p1, _p2; };

    uint32_t divCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

    bool endsWith(const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    }

}// namespace

// ============================================================
//  Construction
// ============================================================
RtDetrVk::RtDetrVk(VulkanRenderer& r)
    : vk_(static_cast<VkDevice>(r.nativeDevice()),
          static_cast<VkPhysicalDevice>(r.nativePhysicalDevice()),
          static_cast<VkQueue>(r.nativeGraphicsQueue()),
          r.graphicsQueueFamily()) {

    convPipe_     = vk_.createPipe(kRtConv2dSpv,   sizeof(kRtConv2dSpv),   4, sizeof(ConvParams));
    dwConvPipe_   = vk_.createPipe(kRtDwConvSpv,   sizeof(kRtDwConvSpv),   4, sizeof(ConvParams));
    maxPoolPipe_  = vk_.createPipe(kRtMaxPoolSpv,  sizeof(kRtMaxPoolSpv),  2, sizeof(PoolParams));
    concatPipe_   = vk_.createPipe(kRtConcatSpv,   sizeof(kRtConcatSpv),   3, sizeof(ConcatParams));
    addPipe_      = vk_.createPipe(kRtAddSpv,      sizeof(kRtAddSpv),      3, sizeof(AddParams));
    addSiluPipe_  = vk_.createPipe(kRtAddSiluSpv,  sizeof(kRtAddSiluSpv),  3, sizeof(AddParams));
    upsamplePipe_ = vk_.createPipe(kRtUpsampleSpv, sizeof(kRtUpsampleSpv), 2, sizeof(UpParams));

    linearPipe_              = vk_.createPipe(kRtLinearSpv,    sizeof(kRtLinearSpv),    4, sizeof(LinearParams));
    layerNormPipe_           = vk_.createPipe(kRtLayerNormSpv, sizeof(kRtLayerNormSpv), 4, sizeof(LNParams));
    softmaxPipe_             = vk_.createPipe(kRtSoftmaxSpv,   sizeof(kRtSoftmaxSpv),   2, sizeof(SoftmaxParams));
    geluPipe_                = vk_.createPipe(kRtGeluSpv,      sizeof(kRtGeluSpv),      2, sizeof(GeluParams));
    transposeCHW2TokensPipe_ = vk_.createPipe(kRtT2tSpv,      sizeof(kRtT2tSpv),       2, sizeof(TransposeParams));
    transposeTokens2CHWPipe_ = vk_.createPipe(kRtT2cSpv,      sizeof(kRtT2cSpv),       2, sizeof(TransposeParams));
    qkvSplicePipe_           = vk_.createPipe(kRtQkvSpliceSpv, sizeof(kRtQkvSpliceSpv), 3, sizeof(QkvSpliceParams));
    attnScoresPipe_          = vk_.createPipe(kRtAttnScoresSpv, sizeof(kRtAttnScoresSpv), 2, sizeof(AttnParams));
    attnApplyPipe_           = vk_.createPipe(kRtAttnApplySpv, sizeof(kRtAttnApplySpv), 3, sizeof(AttnParams));

    reluPipe_             = vk_.createPipe(kRtReluSpv,    sizeof(kRtReluSpv),    2, sizeof(GeluParams));
    transposeMaskPipe_    = vk_.createPipe(kRtTMaskSpv,   sizeof(kRtTMaskSpv),   2, sizeof(TMaskParams));
    offsetPreprocessPipe_ = vk_.createPipe(kRtOffPreSpv,  sizeof(kRtOffPreSpv),  3, sizeof(OffPreParams));
    msDeformAttnPipe_     = vk_.createPipe(kRtMsDeformSpv, sizeof(kRtMsDeformSpv), 7, sizeof(MsDeformParams));
}

RtDetrVk::~RtDetrVk() {
    vk_.destroyPipe(convPipe_);
    vk_.destroyPipe(dwConvPipe_);
    vk_.destroyPipe(maxPoolPipe_);
    vk_.destroyPipe(concatPipe_);
    vk_.destroyPipe(addPipe_);
    vk_.destroyPipe(addSiluPipe_);
    vk_.destroyPipe(upsamplePipe_);
    vk_.destroyPipe(linearPipe_);
    vk_.destroyPipe(layerNormPipe_);
    vk_.destroyPipe(softmaxPipe_);
    vk_.destroyPipe(geluPipe_);
    vk_.destroyPipe(transposeCHW2TokensPipe_);
    vk_.destroyPipe(transposeTokens2CHWPipe_);
    vk_.destroyPipe(qkvSplicePipe_);
    vk_.destroyPipe(attnScoresPipe_);
    vk_.destroyPipe(attnApplyPipe_);
    vk_.destroyPipe(reluPipe_);
    vk_.destroyPipe(msDeformAttnPipe_);
    vk_.destroyPipe(offsetPreprocessPipe_);
    vk_.destroyPipe(transposeMaskPipe_);
}

// ============================================================
//  Weight loading (offline BN-fold — identical to WGPU, fp32 conv weights)
// ============================================================
void RtDetrVk::loadWeights(const std::string& path) {
    auto w = parseWeightBinary(path);

    // Ultralytics RT-DETR wraps Conv2d + BN in a `Conv` module:
    //   <prefix>.conv.weight + <prefix>.bn.*. DWConv uses the same naming.
    // The decoder's input_proj uses nn.Sequential(Conv2d, BatchNorm2d) with
    // names <prefix>.0.weight and <prefix>.1.*; synthesize aliases first so the
    // fold loop below picks them up.
    for (auto& kv : std::vector<std::pair<std::string, std::vector<float>>>(w.data.begin(), w.data.end())) {
        const auto& name = kv.first;
        if (endsWith(name, ".0.weight")) {
            const std::string prefix = name.substr(0, name.size() - std::string(".0.weight").size());
            const std::string bn1 = prefix + ".1.weight";
            if (w.data.count(bn1) == 0) continue;
            const std::string convAlias = prefix + ".conv.weight";
            if (w.data.count(convAlias)) continue;
            w.data[convAlias]   = w.data.at(name);
            w.shapes[convAlias] = w.shapes.at(name);
            auto aliasBN = [&](const char* suffix) {
                std::string from = prefix + ".1" + suffix;
                std::string to   = prefix + ".bn" + suffix;
                if (w.data.count(from) && !w.data.count(to)) {
                    w.data[to]   = w.data.at(from);
                    w.shapes[to] = w.shapes.at(from);
                }
            };
            aliasBN(".weight");
            aliasBN(".bias");
            aliasBN(".running_mean");
            aliasBN(".running_var");
        }
    }

    std::vector<std::string> convKeys;
    for (auto& kv : w.data)
        if (endsWith(kv.first, ".conv.weight")) convKeys.push_back(kv.first);

    size_t foldedCount = 0;
    for (auto& convKey : convKeys) {
        const std::string prefix = convKey.substr(0, convKey.size() - std::string(".conv.weight").size());
        const std::string bnW = prefix + ".bn.weight";
        const std::string bnB = prefix + ".bn.bias";
        const std::string bnM = prefix + ".bn.running_mean";
        const std::string bnV = prefix + ".bn.running_var";
        if (!w.data.count(bnW) || !w.data.count(bnB) ||
            !w.data.count(bnM) || !w.data.count(bnV)) continue;

        const auto& wt    = w.data.at(convKey);
        const auto& gamma = w.data.at(bnW);
        const auto& beta  = w.data.at(bnB);
        const auto& mean  = w.data.at(bnM);
        const auto& var_  = w.data.at(bnV);
        const auto& sh    = w.shapes.at(convKey);
        uint32_t oc  = sh[0];
        uint32_t per = uint32_t(wt.size()) / oc;

        std::vector<float> fw(wt.size());
        std::vector<float> fb(oc);
        for (uint32_t c = 0; c < oc; ++c) {
            float s = gamma[c] / std::sqrt(var_[c] + BN_EPS);
            fb[c]   = beta[c] - mean[c] * s;
            for (uint32_t i = 0; i < per; ++i)
                fw[c * per + i] = wt[c * per + i] * s;
        }

        w.data[prefix + ".fused.weight"]   = std::move(fw);
        w.shapes[prefix + ".fused.weight"] = sh;
        w.data[prefix + ".fused.bias"]     = std::move(fb);
        w.shapes[prefix + ".fused.bias"]   = {oc};
        ++foldedCount;
    }

    weights_.clear();
    for (auto& [name, data] : w.data) {
        const auto& sh = w.shapes.at(name);
        VkTensor t = vk_.createOwned(sh);
        vk_.upload(t.buffer, data.data(), data.size() * sizeof(float));
        weights_.emplace(name, std::move(t));
    }

    std::cout << "RtDetrVk::loadWeights: " << weights_.size() << " GPU tensors"
              << " (folded " << foldedCount << " BN blocks, fp32 conv weights)\n";
}

// ============================================================
//  Low-level GPU ops
// ============================================================
Tensor RtDetrVk::conv_(const Tensor& x, const std::string& weightKey, const std::string& biasKey,
                       int strideH, int strideW, int padH, int padW, Activation act) {
    return conv_(x, weightKey, biasKey, strideH, strideW, padH, padW, padH, padW, act);
}

Tensor RtDetrVk::conv_(const Tensor& x, const std::string& weightKey, const std::string& biasKey,
                       int strideH, int strideW,
                       int padTop, int padLeft, int padBottom, int padRight, Activation act) {
    auto& wt = weights_.at(weightKey);
    if (wt.shape.size() != 4)
        throw std::runtime_error("RtDetrVk::conv_: weight '" + weightKey + "' is not 4D");
    uint32_t out_c = wt.shape[0], in_c = wt.shape[1], k_h = wt.shape[2], k_w = wt.shape[3];
    if (x.C() != in_c)
        throw std::runtime_error("RtDetrVk::conv_: input C mismatch for '" + weightKey + "'");

    uint32_t in_h = x.H(), in_w = x.W();
    uint32_t out_h = (in_h + uint32_t(padTop)  + uint32_t(padBottom) - k_h) / uint32_t(strideH) + 1;
    uint32_t out_w = (in_w + uint32_t(padLeft) + uint32_t(padRight)  - k_w) / uint32_t(strideW) + 1;

    bool hasBias = !biasKey.empty() && weights_.count(biasKey);
    ConvParams cp{in_c, out_c, in_h, in_w, out_h, out_w, k_h, k_w,
                  uint32_t(strideH), uint32_t(strideW), uint32_t(padTop), uint32_t(padLeft),
                  hasBias ? 1u : 0u, uint32_t(act), 0u, 0u};

    Tensor out = vk_.createTensor({out_c, out_h, out_w});
    VkBuffer biasBuf = hasBias ? weights_.at(biasKey).buffer : vk_.dummy();

    std::vector<VkBuffer> ssbos = {x.buffer, wt.buffer, biasBuf, out.buffer};
    uint32_t totalPos = out_h * out_w;
    vk_.dispatch(convPipe_, ssbos, &cp, sizeof(cp), divCeil(totalPos, 16), divCeil(out_c, 16), 1);
    return out;
}

Tensor RtDetrVk::dwConv_(const Tensor& x, const std::string& weightKey, const std::string& biasKey,
                         int strideH, int strideW, int padH, int padW, Activation act) {
    auto& wt = weights_.at(weightKey);
    if (wt.shape.size() != 4 || wt.shape[1] != 1u)
        throw std::runtime_error("RtDetrVk::dwConv_: weight '" + weightKey + "' is not depthwise");
    uint32_t C = wt.shape[0], k_h = wt.shape[2], k_w = wt.shape[3];
    if (x.C() != C)
        throw std::runtime_error("RtDetrVk::dwConv_: input C mismatch for '" + weightKey + "'");

    uint32_t in_h = x.H(), in_w = x.W();
    uint32_t out_h = (in_h + 2 * uint32_t(padH) - k_h) / uint32_t(strideH) + 1;
    uint32_t out_w = (in_w + 2 * uint32_t(padW) - k_w) / uint32_t(strideW) + 1;

    bool hasBias = !biasKey.empty() && weights_.count(biasKey);
    ConvParams cp{C, C, in_h, in_w, out_h, out_w, k_h, k_w,
                  uint32_t(strideH), uint32_t(strideW), uint32_t(padH), uint32_t(padW),
                  hasBias ? 1u : 0u, uint32_t(act), 0u, 0u};

    Tensor out = vk_.createTensor({C, out_h, out_w});
    VkBuffer biasBuf = hasBias ? weights_.at(biasKey).buffer : vk_.dummy();

    std::vector<VkBuffer> ssbos = {x.buffer, wt.buffer, biasBuf, out.buffer};
    vk_.dispatch(dwConvPipe_, ssbos, &cp, sizeof(cp), divCeil(out_w, 8), divCeil(out_h, 8), divCeil(C, 4));
    return out;
}

Tensor RtDetrVk::maxPool_(const Tensor& x, int kH, int kW, int strideH, int strideW,
                          int padTop, int padLeft, int padBottom, int padRight) {
    uint32_t C = x.C(), in_h = x.H(), in_w = x.W();
    uint32_t out_h = (in_h + uint32_t(padTop)  + uint32_t(padBottom) - uint32_t(kH)) / uint32_t(strideH) + 1;
    uint32_t out_w = (in_w + uint32_t(padLeft) + uint32_t(padRight)  - uint32_t(kW)) / uint32_t(strideW) + 1;

    PoolParams pp{C, in_h, in_w, out_h, out_w, uint32_t(kH), uint32_t(kW), uint32_t(strideH),
                  uint32_t(strideW), uint32_t(padTop), uint32_t(padLeft), 0u, 0u, 0u, 0u, 0u};

    Tensor out = vk_.createTensor({C, out_h, out_w});
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(maxPoolPipe_, ssbos, &pp, sizeof(pp), divCeil(out_w, 8), divCeil(out_h, 8), divCeil(C, 4));
    return out;
}

Tensor RtDetrVk::concatC_(const Tensor& a, const Tensor& b) {
    if (a.H() != b.H() || a.W() != b.W())
        throw std::runtime_error("RtDetrVk::concatC_: H/W mismatch");
    uint32_t cA = a.C(), cB = b.C(), H = a.H(), W = a.W();
    ConcatParams cp{cA, cB, H, W};
    Tensor out = vk_.createTensor({cA + cB, H, W});
    std::vector<VkBuffer> ssbos = {a.buffer, b.buffer, out.buffer};
    vk_.dispatch(concatPipe_, ssbos, &cp, sizeof(cp), divCeil(W, 8), divCeil(H, 8), divCeil(cA + cB, 4));
    return out;
}

Tensor RtDetrVk::addTensor_(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel())
        throw std::runtime_error("RtDetrVk::addTensor_: size mismatch");
    uint32_t N = a.numel();
    AddParams ap{N, 0, 0, 0};
    Tensor out = vk_.createTensorV(a.shape);
    std::vector<VkBuffer> ssbos = {a.buffer, b.buffer, out.buffer};
    vk_.dispatch(addPipe_, ssbos, &ap, sizeof(ap), divCeil(N, 64), 1, 1);
    return out;
}

Tensor RtDetrVk::addSilu_(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel())
        throw std::runtime_error("RtDetrVk::addSilu_: size mismatch");
    uint32_t N = a.numel();
    AddParams ap{N, 0, 0, 0};
    Tensor out = vk_.createTensorV(a.shape);
    std::vector<VkBuffer> ssbos = {a.buffer, b.buffer, out.buffer};
    vk_.dispatch(addSiluPipe_, ssbos, &ap, sizeof(ap), divCeil(N, 64), 1, 1);
    return out;
}

Tensor RtDetrVk::upsample2x_(const Tensor& x) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    UpParams up{C, H, W, 0};
    Tensor out = vk_.createTensor({C, H * 2, W * 2});
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(upsamplePipe_, ssbos, &up, sizeof(up), divCeil(W * 2, 8), divCeil(H * 2, 8), divCeil(C, 4));
    return out;
}

// ============================================================
//  Compound backbone blocks
// ============================================================
Tensor RtDetrVk::hgStem_(const Tensor& x) {
    using A = Activation;
    auto y   = conv_(x, "model.0.stem1.fused.weight", "model.0.stem1.fused.bias",
                     2, 2, 1, 1, A::ReLU);
    // F.pad [0,1,0,1] + Conv2x2 p=0 ≡ Conv2x2 with pad [top=0, left=0, bottom=1, right=1]
    auto x2a = conv_(y, "model.0.stem2a.fused.weight", "model.0.stem2a.fused.bias",
                     1, 1, 0, 0, 1, 1, A::ReLU);
    auto x2b = conv_(x2a, "model.0.stem2b.fused.weight", "model.0.stem2b.fused.bias",
                     1, 1, 0, 0, 1, 1, A::ReLU);
    auto x1  = maxPool_(y, 2, 2, 1, 1, 0, 0, 1, 1);
    auto cat = concatC_(x1, x2b);
    auto s3  = conv_(cat, "model.0.stem3.fused.weight", "model.0.stem3.fused.bias",
                     2, 2, 1, 1, A::ReLU);
    auto s4  = conv_(s3, "model.0.stem4.fused.weight", "model.0.stem4.fused.bias",
                     1, 1, 0, 0, A::ReLU);
    return s4;
}

Tensor RtDetrVk::lightConv_(const Tensor& x, const std::string& prefix, int k) {
    // conv1: 1×1, BN-folded, NO activation
    auto m = conv_(x, prefix + ".conv1.fused.weight", prefix + ".conv1.fused.bias",
                   1, 1, 0, 0, Activation::None);
    // conv2: DW k×k s=1 p=(k-1)/2, ReLU
    int pad = (k - 1) / 2;
    auto y = dwConv_(m, prefix + ".conv2.fused.weight", prefix + ".conv2.fused.bias",
                     1, 1, pad, pad, Activation::ReLU);
    return y;
}

Tensor RtDetrVk::hgBlock_(const Tensor& x, const std::string& prefix,
                          int n, int k, bool shortcut, bool lightconv) {
    using A = Activation;
    int pad = (k - 1) / 2;

    const Tensor* prev = &x;
    std::vector<Tensor> owned;
    owned.reserve(size_t(n));
    for (int i = 0; i < n; ++i) {
        std::string mPrefix = prefix + ".m." + std::to_string(i);
        if (lightconv) {
            owned.push_back(lightConv_(*prev, mPrefix, k));
        } else {
            owned.push_back(conv_(*prev, mPrefix + ".fused.weight", mPrefix + ".fused.bias",
                                  1, 1, pad, pad, A::ReLU));
        }
        prev = &owned.back();
    }

    Tensor acc = concatC_(x, owned[0]);
    for (int i = 1; i < n; ++i)
        acc = concatC_(acc, owned[size_t(i)]);

    auto s = conv_(acc, prefix + ".sc.fused.weight", prefix + ".sc.fused.bias",
                   1, 1, 0, 0, A::ReLU);
    auto e = conv_(s, prefix + ".ec.fused.weight", prefix + ".ec.fused.bias",
                   1, 1, 0, 0, A::ReLU);

    if (shortcut) {
        if (e.C() != x.C() || e.H() != x.H() || e.W() != x.W())
            throw std::runtime_error("hgBlock_: shortcut shape mismatch for '" + prefix + "'");
        return addTensor_(e, x);
    }
    return e;
}

RtDetrVk::BackboneFeatures RtDetrVk::backbone_(const Tensor& x) {
    using A = Activation;
    auto f0 = hgStem_(x);                                       // model.0  HGStem        3→48   s4
    auto f1 = hgBlock_(f0, "model.1", 6, 3, false, false);      // model.1  HGBlock        48→128 s4
    auto f2 = dwConv_(f1, "model.2.fused.weight", "model.2.fused.bias",
                      2, 2, 1, 1, A::None);                      // model.2  DWConv s2      128→128 s8
    auto p3 = hgBlock_(f2, "model.3", 6, 3, false, false);      // model.3  HGBlock        128→512 s8  <- P3
    auto f4 = dwConv_(p3, "model.4.fused.weight", "model.4.fused.bias",
                      2, 2, 1, 1, A::None);                      // model.4  DWConv s2      512→512 s16
    auto f5 = hgBlock_(f4, "model.5", 6, 5, false, true);       // model.5  HGBlockLight   512→1024 s16
    auto f6 = hgBlock_(f5, "model.6", 6, 5, true, true);        // model.6  HGBlockLight   1024→1024 s16
    auto p4 = hgBlock_(f6, "model.7", 6, 5, true, true);        // model.7  HGBlockLight   1024→1024 s16 <- P4
    auto f8 = dwConv_(p4, "model.8.fused.weight", "model.8.fused.bias",
                      2, 2, 1, 1, A::None);                      // model.8  DWConv s2      1024→1024 s32
    auto f9 = hgBlock_(f8, "model.9", 6, 5, false, true);       // model.9  HGBlockLight   1024→2048 s32
    auto p5 = conv_(f9, "model.10.fused.weight", "model.10.fused.bias",
                    1, 1, 0, 0, A::None);                        // model.10 Conv1x1        2048→256 s32 <- P5
    return { std::move(p3), std::move(p4), std::move(p5) };
}

// ============================================================
//  Phase B: transformer ops
// ============================================================
Tensor RtDetrVk::uploadArena_(const std::vector<uint32_t>& shape, const float* data) {
    Tensor t = vk_.createTensorV(shape);
    vk_.upload(t.buffer, data, size_t(t.numel()) * sizeof(float));
    return t;
}

Tensor RtDetrVk::linear_(const Tensor& x, const std::string& wKey, const std::string& bKey) {
    auto& wt = weights_.at(wKey);
    if (wt.shape.size() != 2)
        throw std::runtime_error("RtDetrVk::linear_: weight '" + wKey + "' must be 2D");
    uint32_t N = wt.shape[0], K = wt.shape[1];
    if (x.numel() % K != 0)
        throw std::runtime_error("RtDetrVk::linear_: x numel not divisible by K for '" + wKey + "'");
    uint32_t M = x.numel() / K;

    bool hasBias = !bKey.empty() && weights_.count(bKey);
    LinearParams lp{M, N, K, hasBias ? 1u : 0u};
    Tensor out = vk_.createTensorV({M, N});
    VkBuffer bBuf = hasBias ? weights_.at(bKey).buffer : vk_.dummy();
    std::vector<VkBuffer> ssbos = {x.buffer, wt.buffer, bBuf, out.buffer};
    vk_.dispatch(linearPipe_, ssbos, &lp, sizeof(lp), divCeil(N, 16), divCeil(M, 16), 1);
    return out;
}

Tensor RtDetrVk::layerNorm_(const Tensor& x, const std::string& wKey, const std::string& bKey, float eps) {
    auto& g = weights_.at(wKey);
    auto& b = weights_.at(bKey);
    uint32_t D = 1; for (auto s : g.shape) D *= s;
    uint32_t M = x.numel() / D;
    LNParams lp{M, D, eps, 0.f};
    Tensor out = vk_.createTensorV({M, D});
    std::vector<VkBuffer> ssbos = {x.buffer, g.buffer, b.buffer, out.buffer};
    vk_.dispatch(layerNormPipe_, ssbos, &lp, sizeof(lp), divCeil(M, 64), 1, 1);
    return out;
}

Tensor RtDetrVk::softmaxLast_(const Tensor& x) {
    uint32_t N = x.shape.back();
    uint32_t M = x.numel() / N;
    SoftmaxParams sp{M, N, 0, 0};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(softmaxPipe_, ssbos, &sp, sizeof(sp), divCeil(M, 64), 1, 1);
    return out;
}

Tensor RtDetrVk::gelu_(const Tensor& x) {
    uint32_t N = x.numel();
    GeluParams gp{N, 0, 0, 0};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(geluPipe_, ssbos, &gp, sizeof(gp), divCeil(N, 64), 1, 1);
    return out;
}

Tensor RtDetrVk::transposeCHW2Tokens_(const Tensor& x) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    TransposeParams tp{C, H, W, 0};
    Tensor out = vk_.createTensorV({H * W, C});
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(transposeCHW2TokensPipe_, ssbos, &tp, sizeof(tp), divCeil(C * H * W, 64), 1, 1);
    return out;
}

Tensor RtDetrVk::transposeTokens2CHW_(const Tensor& x, uint32_t C, uint32_t H, uint32_t W) {
    TransposeParams tp{C, H, W, 0};
    Tensor out = vk_.createTensorV({C, H, W});
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(transposeTokens2CHWPipe_, ssbos, &tp, sizeof(tp), divCeil(C * H * W, 64), 1, 1);
    return out;
}

Tensor RtDetrVk::attnScores_(const Tensor& qkv, uint32_t H) {
    uint32_t M = qkv.shape[0];
    uint32_t threeD = qkv.shape[1];
    uint32_t D = threeD / 3;
    uint32_t d = D / H;
    AttnParams ap{M, H, d, threeD, 1.0f / std::sqrt(float(d)), 0, 0, 0};
    Tensor out = vk_.createTensorV({H, M, M});
    std::vector<VkBuffer> ssbos = {qkv.buffer, out.buffer};
    vk_.dispatch(attnScoresPipe_, ssbos, &ap, sizeof(ap), divCeil(M, 8), divCeil(M, 8), H);
    return out;
}

Tensor RtDetrVk::attnApply_(const Tensor& qkv, const Tensor& attn, uint32_t H) {
    uint32_t M = qkv.shape[0];
    uint32_t D = qkv.shape[1] / 3;
    uint32_t d = D / H;
    AttnParams ap{M, H, d, qkv.shape[1], 1.0f / std::sqrt(float(d)), 0, 0, 0};
    Tensor out = vk_.createTensorV({M, D});
    std::vector<VkBuffer> ssbos = {qkv.buffer, attn.buffer, out.buffer};
    vk_.dispatch(attnApplyPipe_, ssbos, &ap, sizeof(ap), divCeil(d, 8), divCeil(M, 8), H);
    return out;
}

static std::vector<float> make2dSinCosPosEmbed(uint32_t H, uint32_t W, uint32_t embedDim,
                                               float temperature = 10000.0f) {
    if (embedDim % 4 != 0)
        throw std::runtime_error("sinCosPosEmbed: embed_dim must be multiple of 4");
    uint32_t posDim = embedDim / 4;
    std::vector<float> omega(posDim);
    for (uint32_t i = 0; i < posDim; ++i)
        omega[i] = 1.0f / std::pow(temperature, float(i) / float(posDim));

    std::vector<float> out(size_t(H) * W * embedDim, 0.0f);
    for (uint32_t gy = 0; gy < H; ++gy) {
        for (uint32_t gx = 0; gx < W; ++gx) {
            // Ultralytics' build_2d_sincos uses meshgrid(arange(w), arange(h), "ij"):
            // grid_w = outer loop = y, grid_h = inner loop = x in image coords.
            float posX = float(gy);
            float posY = float(gx);
            size_t base = (size_t(gy) * W + gx) * embedDim;
            for (uint32_t i = 0; i < posDim; ++i) {
                float ax = posX * omega[i];
                float ay = posY * omega[i];
                out[base + 0 * posDim + i] = std::sin(ax);
                out[base + 1 * posDim + i] = std::cos(ax);
                out[base + 2 * posDim + i] = std::sin(ay);
                out[base + 3 * posDim + i] = std::cos(ay);
            }
        }
    }
    return out;
}

Tensor RtDetrVk::aifi_(const Tensor& x) {
    const uint32_t D = x.C(), Hs = x.H(), Ws = x.W(), M = Hs * Ws;
    const uint32_t numHeads = 8;// ultralytics RT-DETR-L AIFI default

    auto tok = transposeCHW2Tokens_(x);// [M, D]

    auto pos  = make2dSinCosPosEmbed(Hs, Ws, D);
    auto posT = uploadArena_({M, D}, pos.data());
    auto qk   = addTensor_(tok, posT);

    // QKV projection: Q,K from qk (target + pos), V from tok (target). Splice on GPU.
    auto yqk = linear_(qk,  "model.11.ma.in_proj_weight", "model.11.ma.in_proj_bias");
    auto yt  = linear_(tok, "model.11.ma.in_proj_weight", "model.11.ma.in_proj_bias");
    QkvSpliceParams sp{M, D, 0, 0};
    Tensor qkv = vk_.createTensorV({M, 3 * D});
    {
        std::vector<VkBuffer> ssbos = {yqk.buffer, yt.buffer, qkv.buffer};
        vk_.dispatch(qkvSplicePipe_, ssbos, &sp, sizeof(sp), divCeil(M * 3u * D, 64), 1, 1);
    }

    auto scores = attnScores_(qkv, numHeads);
    auto attnSm = softmaxLast_(scores);
    auto heads  = attnApply_(qkv, attnSm, numHeads);// [M, D]

    auto attnOut = linear_(heads, "model.11.ma.out_proj.weight", "model.11.ma.out_proj.bias");
    auto r1      = addTensor_(tok, attnOut);
    auto n1      = layerNorm_(r1, "model.11.norm1.weight", "model.11.norm1.bias");

    auto h1 = linear_(n1, "model.11.fc1.weight", "model.11.fc1.bias");
    auto hg = gelu_(h1);
    auto h2 = linear_(hg, "model.11.fc2.weight", "model.11.fc2.bias");
    auto r2 = addTensor_(n1, h2);
    auto n2 = layerNorm_(r2, "model.11.norm2.weight", "model.11.norm2.bias");

    return transposeTokens2CHW_(n2, D, Hs, Ws);
}

// ============================================================
//  Phase C: CCFM neck
// ============================================================
// RepC3: a=SiLU(cv1 1x1); b=SiLU(cv2 1x1); for n: a=SiLU(conv1 3x3 + conv2 1x1);
// return a+b. (cv3 is Identity in RT-DETR since c_ == c_out.)
Tensor RtDetrVk::repC3_(const Tensor& x, const std::string& prefix, int n) {
    using A = Activation;
    auto a = conv_(x, prefix + ".cv1.fused.weight", prefix + ".cv1.fused.bias",
                   1, 1, 0, 0, A::SiLU);
    auto b = conv_(x, prefix + ".cv2.fused.weight", prefix + ".cv2.fused.bias",
                   1, 1, 0, 0, A::SiLU);
    for (int i = 0; i < n; ++i) {
        std::string mp = prefix + ".m." + std::to_string(i);
        auto r1 = conv_(a, mp + ".conv1.fused.weight", mp + ".conv1.fused.bias",
                        1, 1, 1, 1, A::None);
        auto r2 = conv_(a, mp + ".conv2.fused.weight", mp + ".conv2.fused.bias",
                        1, 1, 0, 0, A::None);
        a = addSilu_(r1, r2);
    }
    return addTensor_(a, b);
}

RtDetrVk::NeckFeatures RtDetrVk::ccfm_(const Tensor& p3, const Tensor& p4, const Tensor& f5) {
    using A = Activation;
    auto x12 = conv_(f5, "model.12.fused.weight", "model.12.fused.bias", 1, 1, 0, 0, A::SiLU);
    auto u13 = upsample2x_(x12);
    auto l14 = conv_(p4, "model.14.fused.weight", "model.14.fused.bias", 1, 1, 0, 0, A::None);
    auto c15 = concatC_(u13, l14);
    auto x16 = repC3_(c15, "model.16", 3);
    auto x17 = conv_(x16, "model.17.fused.weight", "model.17.fused.bias", 1, 1, 0, 0, A::SiLU);
    auto u18 = upsample2x_(x17);
    auto l19 = conv_(p3, "model.19.fused.weight", "model.19.fused.bias", 1, 1, 0, 0, A::None);
    auto c20 = concatC_(u18, l19);
    auto s3  = repC3_(c20, "model.21", 3);
    auto d22 = conv_(s3, "model.22.fused.weight", "model.22.fused.bias", 2, 2, 1, 1, A::SiLU);
    auto c23 = concatC_(d22, x17);
    auto s4  = repC3_(c23, "model.24", 3);
    auto d25 = conv_(s4, "model.25.fused.weight", "model.25.fused.bias", 2, 2, 1, 1, A::SiLU);
    auto c26 = concatC_(d25, x12);
    auto s5  = repC3_(c26, "model.27", 3);
    return { std::move(s3), std::move(s4), std::move(s5) };
}

// ============================================================
//  Phase D: deformable decoder + heads
// ============================================================
Tensor RtDetrVk::relu_(const Tensor& x) {
    uint32_t N = x.numel();
    GeluParams gp{N, 0, 0, 0};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(reluPipe_, ssbos, &gp, sizeof(gp), divCeil(N, 64), 1, 1);
    return out;
}

Tensor RtDetrVk::inputProj_(const Tensor& x, int scaleIdx) {
    const std::string p = "model.28.input_proj." + std::to_string(scaleIdx);
    return conv_(x, p + ".fused.weight", p + ".fused.bias", 1, 1, 0, 0, Activation::None);
}

void RtDetrVk::buildMemoryLevel_(const Tensor& p, VkBuffer out, uint32_t outTokenOffset, uint32_t level) {
    uint32_t C = p.C(), H = p.H(), W = p.W();
    TMaskParams mp{C, H, W, outTokenOffset, level, 0, 0, 0};
    std::vector<VkBuffer> ssbos = {p.buffer, out};
    vk_.dispatch(transposeMaskPipe_, ssbos, &mp, sizeof(mp), divCeil(C * H * W, 64), 1, 1);
}

Tensor RtDetrVk::msDeformAttn_(const Tensor& value,
                               const std::vector<std::pair<uint32_t, uint32_t>>& spatialShapes,
                               const Tensor& refPts, const Tensor& samplingOffsets,
                               const Tensor& attnWeights, uint32_t numHeads) {
    uint32_t D  = value.shape.back();
    uint32_t d  = D / numHeads;
    uint32_t L  = uint32_t(spatialShapes.size());
    uint32_t Nq = refPts.shape[0];
    uint32_t HLP = attnWeights.shape.back();
    uint32_t P   = HLP / (numHeads * L);

    std::vector<uint32_t> shapesBuf(L * 2), startsBuf(L);
    uint32_t cumulative = 0;
    for (uint32_t i = 0; i < L; ++i) {
        shapesBuf[i * 2 + 0] = spatialShapes[i].first;
        shapesBuf[i * 2 + 1] = spatialShapes[i].second;
        startsBuf[i] = cumulative;
        cumulative += spatialShapes[i].first * spatialShapes[i].second;
    }
    Tensor shapeGpu = vk_.createRaw(L * 2 * sizeof(uint32_t));
    vk_.upload(shapeGpu.buffer, shapesBuf.data(), L * 2 * sizeof(uint32_t));
    Tensor startGpu = vk_.createRaw(L * sizeof(uint32_t));
    vk_.upload(startGpu.buffer, startsBuf.data(), L * sizeof(uint32_t));

    MsDeformParams mp{Nq, numHeads, d, L, P, 0, 0, 0};
    Tensor out = vk_.createTensorV({Nq, D});
    std::vector<VkBuffer> ssbos = {value.buffer, shapeGpu.buffer, startGpu.buffer,
                                   refPts.buffer, samplingOffsets.buffer, attnWeights.buffer, out.buffer};
    vk_.dispatch(msDeformAttnPipe_, ssbos, &mp, sizeof(mp), divCeil(d, 8), divCeil(Nq, 8), numHeads);
    return out;
}

RtDetrVk::DecoderOut RtDetrVk::decoder_(const std::vector<float>& memoryCpu, uint32_t totalTokens,
                                        const std::vector<float>& encOutCpu,
                                        const std::vector<float>& encScoresCpu,
                                        const std::vector<std::pair<uint32_t, uint32_t>>& spatialShapes) {
    const uint32_t D = 256, numHeads = 8;
    const uint32_t numLevels = uint32_t(spatialShapes.size());
    const uint32_t numPoints = 4, numQueries = 300, numClasses = 80, numLayers = 6;

    auto sigmoidF   = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto invSigmoid = [](float x) { x = std::clamp(x, 1e-5f, 1.0f - 1e-5f); return std::log(x / (1.0f - x)); };

    // memory is consumed by value_proj in every layer → keep it GPU-resident in
    // an owning buffer (survives the per-layer beginFrame arena rewinds).
    VkTensor memOwned = vk_.createOwnedRaw(memoryCpu.size() * sizeof(float));
    vk_.upload(memOwned.buffer, memoryCpu.data(), memoryCpu.size() * sizeof(float));
    Tensor memView{memOwned.buffer, {totalTokens, D}};

    // Step 1: enc_score top-K selection (CPU; enc_score already computed in frame 1).
    std::vector<std::pair<float, uint32_t>> maxScores(totalTokens);
    for (uint32_t t = 0; t < totalTokens; ++t) {
        float mx = encScoresCpu[size_t(t) * numClasses];
        for (uint32_t c = 1; c < numClasses; ++c)
            mx = std::max(mx, encScoresCpu[size_t(t) * numClasses + c]);
        maxScores[t] = {mx, t};
    }
    std::partial_sort(maxScores.begin(), maxScores.begin() + numQueries, maxScores.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<uint32_t> topkIdx(numQueries);
    for (uint32_t i = 0; i < numQueries; ++i) topkIdx[i] = maxScores[i].second;

    std::vector<float> targetCpu(size_t(numQueries) * D);
    for (uint32_t i = 0; i < numQueries; ++i)
        std::copy_n(encOutCpu.data() + size_t(topkIdx[i]) * D, D, targetCpu.data() + size_t(i) * D);

    // Step 2: anchors in logit space, gather top-300.
    const float eps = 0.01f;
    std::vector<float> allAnchors(size_t(totalTokens) * 4);
    {
        size_t idx = 0;
        for (uint32_t lvl = 0; lvl < numLevels; ++lvl) {
            uint32_t H = spatialShapes[lvl].first, W = spatialShapes[lvl].second;
            float wh = 0.05f * float(1u << lvl);
            for (uint32_t gy = 0; gy < H; ++gy)
                for (uint32_t gx = 0; gx < W; ++gx) {
                    float cx = (float(gx) + 0.5f) / float(W);
                    float cy = (float(gy) + 0.5f) / float(H);
                    bool valid = cx > eps && cx < (1.f - eps) && cy > eps && cy < (1.f - eps)
                              && wh > eps && wh < (1.f - eps);
                    if (valid) {
                        allAnchors[idx * 4 + 0] = invSigmoid(cx);
                        allAnchors[idx * 4 + 1] = invSigmoid(cy);
                        allAnchors[idx * 4 + 2] = invSigmoid(wh);
                        allAnchors[idx * 4 + 3] = invSigmoid(wh);
                    } else {
                        allAnchors[idx * 4 + 0] = allAnchors[idx * 4 + 1] =
                        allAnchors[idx * 4 + 2] = allAnchors[idx * 4 + 3] = 1e8f;
                    }
                    ++idx;
                }
        }
    }
    std::vector<float> topkAnchors(size_t(numQueries) * 4);
    for (uint32_t i = 0; i < numQueries; ++i)
        std::copy_n(allAnchors.data() + size_t(topkIdx[i]) * 4, 4, topkAnchors.data() + size_t(i) * 4);

    auto bboxMLP = [&](const Tensor& x, const std::string& prefix) -> Tensor {
        auto h1 = linear_(x, prefix + ".layers.0.weight", prefix + ".layers.0.bias");
        h1 = relu_(h1);
        auto h2 = linear_(h1, prefix + ".layers.1.weight", prefix + ".layers.1.bias");
        h2 = relu_(h2);
        return linear_(h2, prefix + ".layers.2.weight", prefix + ".layers.2.bias");
    };

    // enc_bbox_head → reference points.
    std::vector<float> refPts(size_t(numQueries) * 4);
    {
        vk_.beginFrame();
        Tensor target = uploadArena_({numQueries, D}, targetCpu.data());
        auto encBboxRaw = bboxMLP(target, "model.28.enc_bbox_head");
        vk_.endFrame();
        std::vector<float> bboxRawCpu(encBboxRaw.numel());
        vk_.readback(encBboxRaw.buffer, bboxRawCpu.data(), encBboxRaw.bytes());
        for (size_t i = 0; i < refPts.size(); ++i)
            refPts[i] = sigmoidF(bboxRawCpu[i] + topkAnchors[i]);
    }

    // Step 3: 6 decoder layers (one batched frame each; CPU bbox-refine between).
    for (uint32_t layer = 0; layer < numLayers; ++layer) {
        std::string p = "model.28.decoder.layers." + std::to_string(layer);
        std::vector<float> nextTarget(targetCpu.size());
        std::vector<float> bboxDeltaCpu(size_t(numQueries) * 4);

        vk_.beginFrame();
        Tensor target = uploadArena_({numQueries, D}, targetCpu.data());
        Tensor refTensor = uploadArena_({numQueries, 4u}, refPts.data());

        auto qp1 = linear_(refTensor, "model.28.query_pos_head.layers.0.weight",
                                       "model.28.query_pos_head.layers.0.bias");
        qp1 = relu_(qp1);
        auto queryPos = linear_(qp1, "model.28.query_pos_head.layers.1.weight",
                                      "model.28.query_pos_head.layers.1.bias");

        // self-attention
        auto qkInput = addTensor_(target, queryPos);
        auto yqk = linear_(qkInput, p + ".self_attn.in_proj_weight", p + ".self_attn.in_proj_bias");
        auto yt  = linear_(target,  p + ".self_attn.in_proj_weight", p + ".self_attn.in_proj_bias");
        QkvSpliceParams sp{numQueries, D, 0, 0};
        Tensor qkv = vk_.createTensorV({numQueries, 3 * D});
        {
            std::vector<VkBuffer> ssbos = {yqk.buffer, yt.buffer, qkv.buffer};
            vk_.dispatch(qkvSplicePipe_, ssbos, &sp, sizeof(sp), divCeil(numQueries * 3u * D, 64), 1, 1);
        }
        auto scores  = attnScores_(qkv, numHeads);
        auto attnSm  = softmaxLast_(scores);
        auto heads   = attnApply_(qkv, attnSm, numHeads);
        auto attnOut = linear_(heads, p + ".self_attn.out_proj.weight", p + ".self_attn.out_proj.bias");
        auto n1 = layerNorm_(addTensor_(target, attnOut), p + ".norm1.weight", p + ".norm1.bias");

        // cross-attention (MSDeformAttn)
        auto crossQuery = addTensor_(n1, queryPos);
        auto value = linear_(memView, p + ".cross_attn.value_proj.weight", p + ".cross_attn.value_proj.bias");
        auto rawOffsets = linear_(crossQuery, p + ".cross_attn.sampling_offsets.weight",
                                              p + ".cross_attn.sampling_offsets.bias");
        auto rawAttnW = linear_(crossQuery, p + ".cross_attn.attention_weights.weight",
                                            p + ".cross_attn.attention_weights.bias");
        uint32_t LP = numLevels * numPoints;
        rawAttnW.shape = {numQueries * numHeads, LP};
        auto attnWSm = softmaxLast_(rawAttnW);
        attnWSm.shape = {numQueries, numHeads * LP};
        {
            Tensor refPts4 = uploadArena_({numQueries, 4u}, refPts.data());
            std::vector<uint32_t> levelData(numLevels * 2);
            for (uint32_t ll = 0; ll < numLevels; ++ll) {
                levelData[ll * 2 + 0] = spatialShapes[ll].first;
                levelData[ll * 2 + 1] = spatialShapes[ll].second;
            }
            Tensor levelBuf = vk_.createRaw(numLevels * 2 * sizeof(uint32_t));
            vk_.upload(levelBuf.buffer, levelData.data(), levelData.size() * sizeof(uint32_t));
            OffPreParams opp{numQueries, numHeads, numLevels, numPoints};
            uint32_t offStride = numHeads * numLevels * numPoints * 2;
            std::vector<VkBuffer> ssbos = {refPts4.buffer, levelBuf.buffer, rawOffsets.buffer};
            vk_.dispatch(offsetPreprocessPipe_, ssbos, &opp, sizeof(opp), divCeil(numQueries * offStride, 64), 1, 1);
        }
        std::vector<float> refXy2(size_t(numQueries) * 2);
        for (uint32_t q = 0; q < numQueries; ++q) {
            refXy2[size_t(q) * 2 + 0] = refPts[size_t(q) * 4 + 0];
            refXy2[size_t(q) * 2 + 1] = refPts[size_t(q) * 4 + 1];
        }
        Tensor refPtsT = uploadArena_({numQueries, 2u}, refXy2.data());
        auto crossOut  = msDeformAttn_(value, spatialShapes, refPtsT, rawOffsets, attnWSm, numHeads);
        auto crossProj = linear_(crossOut, p + ".cross_attn.output_proj.weight", p + ".cross_attn.output_proj.bias");
        auto n2 = layerNorm_(addTensor_(n1, crossProj), p + ".norm2.weight", p + ".norm2.bias");

        // FFN
        auto ff1 = linear_(n2, p + ".linear1.weight", p + ".linear1.bias");
        auto ffg = gelu_(ff1);
        auto ff2 = linear_(ffg, p + ".linear2.weight", p + ".linear2.bias");
        auto n3 = layerNorm_(addTensor_(n2, ff2), p + ".norm3.weight", p + ".norm3.bias");

        auto bboxDelta = bboxMLP(n3, "model.28.dec_bbox_head." + std::to_string(layer));
        vk_.endFrame();

        vk_.readback(n3.buffer, nextTarget.data(), n3.bytes());
        vk_.readback(bboxDelta.buffer, bboxDeltaCpu.data(), bboxDelta.bytes());

        targetCpu = std::move(nextTarget);
        for (uint32_t q = 0; q < numQueries; ++q)
            for (uint32_t c = 0; c < 4; ++c) {
                size_t i = size_t(q) * 4 + c;
                refPts[i] = sigmoidF(bboxDeltaCpu[i] + invSigmoid(refPts[i]));
            }
        vk_.resetArena();
    }

    // Step 4: final class logits from the last layer's score head.
    std::vector<float> finalScores(size_t(numQueries) * numClasses);
    {
        vk_.beginFrame();
        Tensor target = uploadArena_({numQueries, D}, targetCpu.data());
        auto fs = linear_(target, "model.28.dec_score_head.5.weight", "model.28.dec_score_head.5.bias");
        vk_.endFrame();
        vk_.readback(fs.buffer, finalScores.data(), fs.bytes());
    }

    return DecoderOut{std::move(finalScores), std::move(refPts)};
}

std::vector<Detection> RtDetrVk::infer(const unsigned char* rgba, int srcW, int srcH,
                                       float confThresh) {
    const int dstW = INPUT_SIZE, dstH = INPUT_SIZE;

    // CPU preprocessing: bilinear resize (plain stretch — no letterbox) + /255.
    std::vector<float> chw(size_t(3) * dstH * dstW);
    for (int dy = 0; dy < dstH; ++dy) {
        float sy = (float(dy) + 0.5f) * float(srcH) / float(dstH) - 0.5f;
        int y0 = std::max(0, int(std::floor(sy)));
        int y1 = std::min(srcH - 1, y0 + 1);
        float fy = sy - float(y0);
        for (int dx = 0; dx < dstW; ++dx) {
            float sx = (float(dx) + 0.5f) * float(srcW) / float(dstW) - 0.5f;
            int x0 = std::max(0, int(std::floor(sx)));
            int x1 = std::min(srcW - 1, x0 + 1);
            float fx = sx - float(x0);
            for (int c = 0; c < 3; ++c) {
                float tl = float(rgba[(y0 * srcW + x0) * 4 + c]);
                float tr = float(rgba[(y0 * srcW + x1) * 4 + c]);
                float bl = float(rgba[(y1 * srcW + x0) * 4 + c]);
                float br = float(rgba[(y1 * srcW + x1) * 4 + c]);
                float val = (1 - fy) * ((1 - fx) * tl + fx * tr)
                          + fy       * ((1 - fx) * bl + fx * br);
                chw[size_t(c) * dstH * dstW + size_t(dy) * dstW + dx] = val / 255.0f;
            }
        }
    }

    auto fw = runForward(chw);

    auto sigmoidF = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto iou = [](const Detection& a, const Detection& b) {
        float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
        float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
        float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (areaA + areaB - inter + 1e-6f);
    };

    std::vector<Detection> candidates;
    for (int q = 0; q < 300; ++q) {
        int bestC = 0;
        float bestP = sigmoidF(fw.decScores[size_t(q) * NUM_CLASSES]);
        for (int c = 1; c < NUM_CLASSES; ++c) {
            float pr = sigmoidF(fw.decScores[size_t(q) * NUM_CLASSES + c]);
            if (pr > bestP) { bestP = pr; bestC = c; }
        }
        if (bestP < confThresh) continue;

        float cx = fw.decBboxes[size_t(q) * 4 + 0], cy = fw.decBboxes[size_t(q) * 4 + 1];
        float w  = fw.decBboxes[size_t(q) * 4 + 2], h  = fw.decBboxes[size_t(q) * 4 + 3];
        float x1 = std::clamp((cx - w * 0.5f) * float(srcW), 0.f, float(srcW));
        float y1 = std::clamp((cy - h * 0.5f) * float(srcH), 0.f, float(srcH));
        float x2 = std::clamp((cx + w * 0.5f) * float(srcW), 0.f, float(srcW));
        float y2 = std::clamp((cy + h * 0.5f) * float(srcH), 0.f, float(srcH));
        candidates.push_back({bestC, bestP, x1, y1, x2, y2});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.confidence > b.confidence; });
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<Detection> results;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        results.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j)
            if (!suppressed[j] && candidates[j].classId == candidates[i].classId
                && iou(candidates[i], candidates[j]) > 0.7f)
                suppressed[j] = true;
    }
    return results;
}

std::vector<float> RtDetrVk::selfTestMsDeform() {
    const uint32_t H = 2, d = 2, D = H * d, L = 2, P = 1, Nq = 1, totalTokens = 5;
    vk_.beginFrame();
    std::vector<float> valData(totalTokens * D);
    for (uint32_t i = 0; i < totalTokens * D; ++i) valData[i] = float(i + 1);
    Tensor value = uploadArena_({totalTokens, D}, valData.data());
    std::vector<std::pair<uint32_t, uint32_t>> shapes = {{2, 2}, {1, 1}};
    std::vector<float> refData = {0.5f, 0.5f};
    Tensor refPts = uploadArena_({Nq, 2u}, refData.data());
    std::vector<float> offData(Nq * H * L * P * 2, 0.0f);
    Tensor offs = uploadArena_({Nq, H * L * P * 2}, offData.data());
    std::vector<float> awData(Nq * H * L * P, 0.5f);
    Tensor aw = uploadArena_({Nq, H * L * P}, awData.data());
    auto out = msDeformAttn_(value, shapes, refPts, offs, aw, H);
    vk_.endFrame();
    std::vector<float> r(out.numel());
    vk_.readback(out.buffer, r.data(), out.bytes());
    vk_.resetArena();
    return r;
}

// ============================================================
//  Phased validation entry
// ============================================================
RtDetrVk::ForwardOut RtDetrVk::runForward(const std::vector<float>& chw) {
    const uint32_t dstW = INPUT_SIZE, dstH = INPUT_SIZE;
    if (chw.size() != size_t(3) * dstH * dstW)
        throw std::runtime_error("RtDetrVk::runForward: input must be 3*640*640 floats");

    ForwardOut out;
    std::vector<std::pair<uint32_t, uint32_t>> shapes;
    std::vector<float> encScoresCpu;
    uint32_t totalTokens = 0;

    auto rb = [&](const Tensor& t, std::vector<float>& dst, std::array<uint32_t, 3>& dim) {
        dst.resize(t.numel());
        vk_.readback(t.buffer, dst.data(), t.bytes());
        dim = {t.C(), t.H(), t.W()};
    };

    // ── Frame 1: backbone → AIFI → CCFM → input_proj → memory → enc_output → enc_score ──
    {
        vk_.beginFrame();
        Tensor input = vk_.createTensorV({3u, dstH, dstW});
        vk_.upload(input.buffer, chw.data(), chw.size() * sizeof(float));
        auto bp   = backbone_(input);
        auto f5   = aifi_(bp.p5);
        auto neck = ccfm_(bp.p3, bp.p4, f5);
        auto ip0  = inputProj_(neck.s3, 0);
        auto ip1  = inputProj_(neck.s4, 1);
        auto ip2  = inputProj_(neck.s5, 2);

        uint32_t M0 = ip0.H() * ip0.W(), M1 = ip1.H() * ip1.W(), M2 = ip2.H() * ip2.W();
        totalTokens = M0 + M1 + M2;
        Tensor memory = vk_.createTensorV({totalTokens, 256u});
        buildMemoryLevel_(ip0, memory.buffer, 0, 0);
        buildMemoryLevel_(ip1, memory.buffer, M0, 1);
        buildMemoryLevel_(ip2, memory.buffer, M0 + M1, 2);

        auto encLin    = linear_(memory, "model.28.enc_output.0.weight", "model.28.enc_output.0.bias");
        auto enc       = layerNorm_(encLin, "model.28.enc_output.1.weight", "model.28.enc_output.1.bias");
        auto encScores = linear_(enc, "model.28.enc_score_head.weight", "model.28.enc_score_head.bias");
        vk_.endFrame();

        rb(bp.p3, out.p3, out.p3dim);
        rb(bp.p4, out.p4, out.p4dim);
        rb(bp.p5, out.p5, out.p5dim);
        rb(f5, out.aifi, out.aifidim);
        rb(neck.s3, out.s3, out.s3dim);
        rb(neck.s4, out.s4, out.s4dim);
        rb(neck.s5, out.s5, out.s5dim);
        rb(ip0, out.ip0, out.ip0dim);
        rb(ip1, out.ip1, out.ip1dim);
        rb(ip2, out.ip2, out.ip2dim);
        out.memory.resize(memory.numel());
        vk_.readback(memory.buffer, out.memory.data(), memory.bytes());
        out.encOutput.resize(enc.numel());
        vk_.readback(enc.buffer, out.encOutput.data(), enc.bytes());
        encScoresCpu.resize(encScores.numel());
        vk_.readback(encScores.buffer, encScoresCpu.data(), encScores.bytes());

        shapes = {{ip0.H(), ip0.W()}, {ip1.H(), ip1.W()}, {ip2.H(), ip2.W()}};
        vk_.resetArena();
    }

    // ── Decoder (multi-frame: GPU bursts + CPU top-K / iterative bbox refine) ──
    auto dec = decoder_(out.memory, totalTokens, out.encOutput, encScoresCpu, shapes);
    out.decScores = std::move(dec.scores);
    out.decBboxes = std::move(dec.bboxes);

    return out;
}

}// namespace rtdetr
