#include "RfDetrVk.hpp"

#include "WeightLoader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "threepp/renderers/vulkan/shaders/conv2d.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/conv1x1.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/conv3x3.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/im2col.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/linear.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/layernorm.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/softmax.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/gelu.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/layerscale.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_scores.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_apply.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/add.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/concat.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/build_embeddings.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_vit_scores.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_vit_apply.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/unwindow.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/layernorm_chw.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/chcopy.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/transpose_chw2tokens.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/qkv_splice.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/relu.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/offset_preprocess.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/msdeformattn.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/scale_add.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/linear_splitk.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/reduce_splitk.comp.rfdetr.spv.h"

using threepp::VulkanRenderer;

namespace rfdetr {

namespace {
    struct ConvParams {
        uint32_t in_c, out_c, in_h, in_w;
        uint32_t out_h, out_w, k_h, k_w;
        uint32_t stride_h, stride_w, pad_h, pad_w;
        uint32_t has_bias, activation, _p0, _p1;
    };
    struct LinearParams    { uint32_t M, N, K, hasBias, act, _p0, _p1, _p2; };
    struct SplitKParams    { uint32_t M, N, K, splitK, kChunk, _p0, _p1, _p2; };
    struct ReduceParams    { uint32_t M, N, splitK, hasBias, act, _p0, _p1, _p2; };
    struct LNParams        { uint32_t M, D; float eps, _pad; };
    struct SoftmaxParams   { uint32_t M, N, _p0, _p1; };
    struct GeluParams      { uint32_t n, _p0, _p1, _p2; };
    struct LayerScaleParams{ uint32_t M, C, _p0, _p1; };
    struct AttnParams      { uint32_t M, H, d, stride; float scale; uint32_t _p0, _p1, _p2; };
    struct AttnVitParams   { uint32_t T, H, d, B; float scale; uint32_t _p0, _p1, _p2; };
    struct AddParams       { uint32_t n, _p0, _p1, _p2; };
    struct ConcatParams    { uint32_t c_a, c_b, h, w; };
    struct EmbedParams     { uint32_t C, grid, winSide, numWin; };
    struct LnChwParams     { uint32_t C, HW; float eps; uint32_t act; };
    struct UnwindowParams  { uint32_t C, grid, winSide, numWin, chOffset, _p0, _p1, _p2; };
    struct ChCopyParams    { uint32_t Cn, HW, srcOff, dstOff; };
    struct TransposeParams { uint32_t C, H, W, _p; };
    struct QkvSpliceParams { uint32_t M, D, _p0, _p1; };
    struct OffPreParams    { uint32_t Nq, numHeads, numLevels, numPoints; };
    struct MsDeformParams  { uint32_t Nq, H, d, L, P, _p0, _p1, _p2; };

    uint32_t divCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
}// namespace

// Per-variant overrides. Nano/Small/Medium share the dinov2_windowed_small
// backbone + single-scale P4 head (the struct defaults = Nano); only input
// resolution and decoder depth differ. Pinned from rfdetr/config.py.
RfDetrConfig RfDetrConfig::forVariant(RfDetrVariant v) {
    RfDetrConfig c;// defaults are Nano
    switch (v) {
        case RfDetrVariant::Nano:   c.resolution = 384; c.decLayers = 2; break;
        case RfDetrVariant::Small:  c.resolution = 512; c.decLayers = 3; break;
        case RfDetrVariant::Medium: c.resolution = 576; c.decLayers = 4; break;
    }
    return c;
}

RfDetrVk::RfDetrVk(VulkanRenderer& r, RfDetrVariant variant)
    : vk_(static_cast<VkDevice>(r.nativeDevice()),
          static_cast<VkPhysicalDevice>(r.nativePhysicalDevice()),
          static_cast<VkQueue>(r.nativeGraphicsQueue()),
          r.graphicsQueueFamily()),
      cfg_(RfDetrConfig::forVariant(variant)) {

    convPipe_       = vk_.createPipe(kRfConv2dSpv,    sizeof(kRfConv2dSpv),    4, sizeof(ConvParams));
    conv1x1Pipe_    = vk_.createPipe(kRfConv1x1Spv,   sizeof(kRfConv1x1Spv),   4, sizeof(ConvParams));
    conv3x3Pipe_    = vk_.createPipe(kRfConv3x3Spv,   sizeof(kRfConv3x3Spv),   4, sizeof(ConvParams));
    im2colPipe_     = vk_.createPipe(kRfIm2colSpv,    sizeof(kRfIm2colSpv),    2, sizeof(ConvParams));
    linearPipe_     = vk_.createPipe(kRfLinearSpv,    sizeof(kRfLinearSpv),    4, sizeof(LinearParams));
    layerNormPipe_  = vk_.createPipe(kRfLayerNormSpv, sizeof(kRfLayerNormSpv), 4, sizeof(LNParams));
    softmaxPipe_    = vk_.createPipe(kRfSoftmaxSpv,   sizeof(kRfSoftmaxSpv),   2, sizeof(SoftmaxParams));
    geluPipe_       = vk_.createPipe(kRfGeluSpv,      sizeof(kRfGeluSpv),      2, sizeof(GeluParams));
    layerScalePipe_ = vk_.createPipe(kRfLayerScaleSpv, sizeof(kRfLayerScaleSpv), 3, sizeof(LayerScaleParams));
    attnScoresPipe_ = vk_.createPipe(kRfAttnScoresSpv, sizeof(kRfAttnScoresSpv), 2, sizeof(AttnParams));
    attnApplyPipe_  = vk_.createPipe(kRfAttnApplySpv, sizeof(kRfAttnApplySpv), 3, sizeof(AttnParams));
    addPipe_        = vk_.createPipe(kRfAddSpv,       sizeof(kRfAddSpv),       3, sizeof(AddParams));
    concatPipe_     = vk_.createPipe(kRfConcatSpv,    sizeof(kRfConcatSpv),    3, sizeof(ConcatParams));
    buildEmbeddingsPipe_ = vk_.createPipe(kRfBuildEmbedSpv, sizeof(kRfBuildEmbedSpv), 4, sizeof(EmbedParams));
    attnVitScoresPipe_ = vk_.createPipe(kRfAttnVitScoresSpv, sizeof(kRfAttnVitScoresSpv), 2, sizeof(AttnVitParams));
    attnVitApplyPipe_  = vk_.createPipe(kRfAttnVitApplySpv,  sizeof(kRfAttnVitApplySpv),  3, sizeof(AttnVitParams));
    unwindowPipe_ = vk_.createPipe(kRfUnwindowSpv, sizeof(kRfUnwindowSpv), 2, sizeof(UnwindowParams));
    lnChwPipe_    = vk_.createPipe(kRfLnChwSpv,    sizeof(kRfLnChwSpv),    4, sizeof(LnChwParams));
    chCopyPipe_   = vk_.createPipe(kRfChCopySpv,   sizeof(kRfChCopySpv),   2, sizeof(ChCopyParams));
    transposeT2tPipe_     = vk_.createPipe(kRfT2tSpv,       sizeof(kRfT2tSpv),       2, sizeof(TransposeParams));
    qkvSplicePipe_        = vk_.createPipe(kRfQkvSpliceSpv, sizeof(kRfQkvSpliceSpv), 3, sizeof(QkvSpliceParams));
    reluPipe_             = vk_.createPipe(kRfReluSpv,      sizeof(kRfReluSpv),      2, sizeof(GeluParams));
    offsetPreprocessPipe_ = vk_.createPipe(kRfOffPreSpv,   sizeof(kRfOffPreSpv),    3, sizeof(OffPreParams));
    msDeformAttnPipe_     = vk_.createPipe(kRfMsDeformSpv,  sizeof(kRfMsDeformSpv),  7, sizeof(MsDeformParams));
    scaleAddPipe_         = vk_.createPipe(kRfScaleAddSpv,  sizeof(kRfScaleAddSpv),  4, sizeof(LayerScaleParams));
    linearSplitKPipe_     = vk_.createPipe(kRfLinearSplitKSpv, sizeof(kRfLinearSplitKSpv), 3, sizeof(SplitKParams));
    reduceSplitKPipe_     = vk_.createPipe(kRfReduceSplitKSpv, sizeof(kRfReduceSplitKSpv), 3, sizeof(ReduceParams));

    // Partial-sum scratch sized for the largest split-K linear we route: splitK=4
    // planes of [maxTokens, N<=512]. Scales with the variant's token count (Nano
    // ~580, Medium ~1300). One buffer reused across calls; barriers serialize access.
    splitKPartialFloats_ = uint64_t(4) * uint64_t(cfg_.maxTokens()) * 512ull;
    splitKPartials_ = vk_.createOwnedRaw(VkDeviceSize(splitKPartialFloats_) * sizeof(float));
    useSplitK_ = std::getenv("RF_NOSPLITK") == nullptr;
}

RfDetrVk::~RfDetrVk() {
    vk_.destroyPipe(convPipe_);
    vk_.destroyPipe(conv1x1Pipe_);
    vk_.destroyPipe(conv3x3Pipe_);
    vk_.destroyPipe(im2colPipe_);
    vk_.destroyPipe(linearPipe_);
    vk_.destroyPipe(layerNormPipe_);
    vk_.destroyPipe(softmaxPipe_);
    vk_.destroyPipe(geluPipe_);
    vk_.destroyPipe(layerScalePipe_);
    vk_.destroyPipe(attnScoresPipe_);
    vk_.destroyPipe(attnApplyPipe_);
    vk_.destroyPipe(addPipe_);
    vk_.destroyPipe(concatPipe_);
    vk_.destroyPipe(buildEmbeddingsPipe_);
    vk_.destroyPipe(attnVitScoresPipe_);
    vk_.destroyPipe(attnVitApplyPipe_);
    vk_.destroyPipe(unwindowPipe_);
    vk_.destroyPipe(lnChwPipe_);
    vk_.destroyPipe(chCopyPipe_);
    vk_.destroyPipe(transposeT2tPipe_);
    vk_.destroyPipe(qkvSplicePipe_);
    vk_.destroyPipe(reluPipe_);
    vk_.destroyPipe(offsetPreprocessPipe_);
    vk_.destroyPipe(msDeformAttnPipe_);
    vk_.destroyPipe(scaleAddPipe_);
    vk_.destroyPipe(linearSplitKPipe_);
    vk_.destroyPipe(reduceSplitKPipe_);
}

void RfDetrVk::loadWeights(const std::string& path) {
    auto w = parseWeightBinary(path);
    vk_.resetDescriptorCache();// cached sets reference the previous weight buffers
    weights_.clear();
    for (auto& [name, data] : w.data) {
        const auto& sh = w.shapes.at(name);
        VkTensor t = vk_.createOwned(sh.empty() ? std::vector<uint32_t>{uint32_t(data.size())} : sh);
        vk_.upload(t.buffer, data.data(), data.size() * sizeof(float));
        weights_.emplace(name, std::move(t));
    }

    // Validate the chosen variant against the checkpoint — trust the weights over the
    // enum (this is how RF-DETR-Nano's dec_layers=2 override was first caught). Nano/
    // Small/Medium have 2/3/4 decoder layers, so this alone catches a variant mismatch.
    int ckptDecLayers = 0;
    while (weights_.count("transformer.decoder.layers." + std::to_string(ckptDecLayers) + ".self_attn.out_proj.weight"))
        ++ckptDecLayers;
    if (ckptDecLayers != cfg_.decLayers)
        throw std::runtime_error(
                "RfDetrVk: variant/weights mismatch — config expects dec_layers=" +
                std::to_string(cfg_.decLayers) + " but checkpoint has " + std::to_string(ckptDecLayers) +
                ". Pass the RfDetrVariant matching these weights.");

    // Pre-fuse per-layer Q/K/V projections into one [3D, D] weight (+ [3D] bias) so
    // each ViT block runs a single GEMM instead of three. Concatenated along the
    // output dim: rows [0,D)=Q, [D,2D)=K, [2D,3D)=V, matching the fused attn shaders.
    auto concat3 = [](const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c) {
        std::vector<float> out;
        out.reserve(a.size() + b.size() + c.size());
        out.insert(out.end(), a.begin(), a.end());
        out.insert(out.end(), b.begin(), b.end());
        out.insert(out.end(), c.begin(), c.end());
        return out;
    };
    for (int L = 0; L < cfg_.numLayers; ++L) {
        std::string p = "backbone.0.encoder.encoder.encoder.layer." + std::to_string(L) + ".attention.attention.";
        auto qw = w.data.find(p + "query.weight"), kw = w.data.find(p + "key.weight"), vw = w.data.find(p + "value.weight");
        if (qw == w.data.end() || kw == w.data.end() || vw == w.data.end()) continue;
        uint32_t K = w.shapes.at(p + "query.weight")[1];
        auto fw = concat3(qw->second, kw->second, vw->second);
        VkTensor tw = vk_.createOwned({uint32_t(fw.size() / K), K});
        vk_.upload(tw.buffer, fw.data(), fw.size() * sizeof(float));
        weights_.emplace(p + "qkv.weight", std::move(tw));
        auto qb = w.data.find(p + "query.bias"), kb = w.data.find(p + "key.bias"), vb = w.data.find(p + "value.bias");
        if (qb != w.data.end() && kb != w.data.end() && vb != w.data.end()) {
            auto fb = concat3(qb->second, kb->second, vb->second);
            VkTensor tb = vk_.createOwned({uint32_t(fb.size())});
            vk_.upload(tb.buffer, fb.data(), fb.size() * sizeof(float));
            weights_.emplace(p + "qkv.bias", std::move(tb));
        }
    }
    // refpoint_embed is a constant — cache the first cfg_.numQueries rows on the CPU so
    // the decoder doesn't read it back from the GPU on every inference.
    auto rp = w.data.find("refpoint_embed.weight");
    if (rp != w.data.end() && rp->second.size() >= size_t(cfg_.numQueries) * 4)
        learnedRefCpu_.assign(rp->second.begin(), rp->second.begin() + size_t(cfg_.numQueries) * 4);

    std::cout << "RfDetrVk::loadWeights: " << weights_.size() << " GPU tensors (fp32)\n";
}

// Generic conv (used by patch embed: 3->384, k16 s16). f32 weights [out_c,in_c,kH,kW].
Tensor RfDetrVk::conv_(const Tensor& x, const std::string& wKey, const std::string& bKey,
                       int strideH, int strideW, int padH, int padW) {
    auto& wt = weights_.at(wKey);
    uint32_t out_c = wt.shape[0], in_c = wt.shape[1], k_h = wt.shape[2], k_w = wt.shape[3];
    uint32_t in_h = x.H(), in_w = x.W();
    uint32_t out_h = (in_h + 2 * uint32_t(padH) - k_h) / uint32_t(strideH) + 1;
    uint32_t out_w = (in_w + 2 * uint32_t(padW) - k_w) / uint32_t(strideW) + 1;
    bool hasBias = !bKey.empty() && weights_.count(bKey);
    ConvParams cp{in_c, out_c, in_h, in_w, out_h, out_w, k_h, k_w,
                  uint32_t(strideH), uint32_t(strideW), uint32_t(padH), uint32_t(padW),
                  hasBias ? 1u : 0u, 0u, 0u, 0u};
    Tensor out = vk_.createTensor({out_c, out_h, out_w});
    VkBuffer biasBuf = hasBias ? weights_.at(bKey).buffer : vk_.dummy();
    std::vector<VkBuffer> ssbos = {x.buffer, wt.buffer, biasBuf, out.buffer};
    uint32_t totalPos = out_h * out_w;
    vk_.dispatch(convPipe_, ssbos, &cp, sizeof(cp), divCeil(totalPos, 16), divCeil(out_c, 16), 1);
    return out;
}

Tensor RfDetrVk::patchEmbed_(const Tensor& x) {
    return conv_(x,
                 "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight",
                 "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.bias",
                 cfg_.patch, cfg_.patch, 0, 0);
}

// patch_embed [C,G,G] -> windowed tokens [W*W, 1+winSide^2, C] with CLS + pos.
Tensor RfDetrVk::buildEmbeddings_(const Tensor& patchEmbed) {
    const uint32_t C = cfg_.embedDim, grid = cfg_.grid(), winSide = cfg_.grid() / cfg_.numWindows, numWin = cfg_.numWindows;
    const uint32_t tokensPerWin = winSide * winSide + 1u;
    const uint32_t numWinSq = numWin * numWin;
    EmbedParams ep{C, grid, winSide, numWin};
    Tensor out = vk_.createTensorV({numWinSq, tokensPerWin, C});
    auto& pos = weights_.at("backbone.0.encoder.encoder.embeddings.position_embeddings");
    auto& cls = weights_.at("backbone.0.encoder.encoder.embeddings.cls_token");
    std::vector<VkBuffer> ssbos = {patchEmbed.buffer, pos.buffer, cls.buffer, out.buffer};
    uint32_t total = numWinSq * tokensPerWin * C;
    vk_.dispatch(buildEmbeddingsPipe_, ssbos, &ep, sizeof(ep), divCeil(total, 64), 1, 1);
    return out;
}

// Y[M,N] = X[M,K] · W[N,K]^T + b[N]. W is nn.Linear weight [out=N, in=K] row-major.
Tensor RfDetrVk::linear_(const Tensor& x, const std::string& wKey, const std::string& bKey, uint32_t act) {
    auto& wt = weights_.at(wKey);
    uint32_t N = wt.shape[0], K = wt.shape[1];
    uint32_t M = x.numel() / K;
    bool hasBias = !bKey.empty() && weights_.count(bKey);
    Tensor out = vk_.createTensorV({M, N});
    VkBuffer biasBuf = hasBias ? weights_.at(bKey).buffer : vk_.dummy();

    // Split-K for low-N GEMMs (fc2, attention out-proj, ...). The monolithic GEMM
    // makes ceil(M/64)*ceil(N/64) workgroups; at N<=512 that's too few to fill the
    // GPU, so the op is occupancy-bound (fc2 runs far slower than fc1 despite equal
    // MACs — only the workgroup count differs). Partition K into `splitK` planes to
    // multiply the workgroup count, then reduce. Guarded so the partials fit scratch.
    const uint32_t splitK = 4u;
    const uint32_t kChunk = divCeil(K, splitK);
    if (useSplitK_ && N <= 512u && K >= 256u && M >= 256u &&
        static_cast<uint64_t>(splitK) * M * N <= splitKPartialFloats_) {
        SplitKParams sp{M, N, K, splitK, kChunk, 0u, 0u, 0u};
        std::vector<VkBuffer> pssbos = {x.buffer, wt.buffer, splitKPartials_.buffer};
        vk_.dispatch(linearSplitKPipe_, pssbos, &sp, sizeof(sp), divCeil(N, 64), divCeil(M, 64), splitK);
        ReduceParams rp{M, N, splitK, hasBias ? 1u : 0u, act, 0u, 0u, 0u};
        std::vector<VkBuffer> rssbos = {splitKPartials_.buffer, biasBuf, out.buffer};
        vk_.dispatch(reduceSplitKPipe_, rssbos, &rp, sizeof(rp), divCeil(M * N, 256u), 1, 1);
        return out;
    }

    LinearParams lp{M, N, K, hasBias ? 1u : 0u, act, 0u, 0u, 0u};
    std::vector<VkBuffer> ssbos = {x.buffer, wt.buffer, biasBuf, out.buffer};
    vk_.dispatch(linearPipe_, ssbos, &lp, sizeof(lp), divCeil(N, 64), divCeil(M, 64), 1);// BN=64, BM=64
    return out;
}

Tensor RfDetrVk::layerNorm_(const Tensor& x, const std::string& wKey, const std::string& bKey, uint32_t D) {
    uint32_t M = x.numel() / D;
    LNParams lp{M, D, 1e-6f, 0.f};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, weights_.at(wKey).buffer, weights_.at(bKey).buffer, out.buffer};
    vk_.dispatch(layerNormPipe_, ssbos, &lp, sizeof(lp), divCeil(M, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::gelu_(const Tensor& x) {
    uint32_t n = x.numel();
    GeluParams gp{n, 0u, 0u, 0u};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(geluPipe_, ssbos, &gp, sizeof(gp), divCeil(n, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::layerScale_(const Tensor& x, const std::string& gammaKey, uint32_t C) {
    uint32_t M = x.numel() / C;
    LayerScaleParams lp{M, C, 0u, 0u};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, weights_.at(gammaKey).buffer, out.buffer};
    vk_.dispatch(layerScalePipe_, ssbos, &lp, sizeof(lp), divCeil(x.numel(), 64), 1, 1);
    return out;
}

Tensor RfDetrVk::add_(const Tensor& a, const Tensor& b) {
    uint32_t n = a.numel();
    AddParams ap{n, 0u, 0u, 0u};
    Tensor out = vk_.createTensorV(a.shape);
    std::vector<VkBuffer> ssbos = {a.buffer, b.buffer, out.buffer};
    vk_.dispatch(addPipe_, ssbos, &ap, sizeof(ap), divCeil(n, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::scaleAdd_(const Tensor& residual, const Tensor& branch, const std::string& gammaKey, uint32_t C) {
    uint32_t M = residual.numel() / C;
    LayerScaleParams lp{M, C, 0u, 0u};
    Tensor out = vk_.createTensorV(residual.shape);
    std::vector<VkBuffer> ssbos = {residual.buffer, branch.buffer, weights_.at(gammaKey).buffer, out.buffer};
    vk_.dispatch(scaleAddPipe_, ssbos, &lp, sizeof(lp), divCeil(residual.numel(), 64), 1, 1);
    return out;
}

Tensor RfDetrVk::softmaxRows_(const Tensor& x, uint32_t rows, uint32_t n) {
    SoftmaxParams sp{rows, n, 0u, 0u};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(softmaxPipe_, ssbos, &sp, sizeof(sp), divCeil(rows, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::attnVitScores_(const Tensor& qkv, uint32_t T, uint32_t B) {
    const uint32_t H = cfg_.numHeads, d = cfg_.embedDim / cfg_.numHeads;
    AttnVitParams ap{T, H, d, B, 1.0f / std::sqrt(float(d)), 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({B * H * T, T});
    std::vector<VkBuffer> ssbos = {qkv.buffer, out.buffer};
    vk_.dispatch(attnVitScoresPipe_, ssbos, &ap, sizeof(ap), divCeil(T, 8), divCeil(T, 8), B * H);
    return out;
}

Tensor RfDetrVk::attnVitApply_(const Tensor& attn, const Tensor& qkv, uint32_t T, uint32_t B) {
    const uint32_t H = cfg_.numHeads, d = cfg_.embedDim / cfg_.numHeads;
    AttnVitParams ap{T, H, d, B, 1.0f / std::sqrt(float(d)), 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({B * T, H * d});
    std::vector<VkBuffer> ssbos = {attn.buffer, qkv.buffer, out.buffer};
    vk_.dispatch(attnVitApplyPipe_, ssbos, &ap, sizeof(ap), divCeil(d, 8), divCeil(T, 8), B * H);
    return out;
}

// One DINOv2 transformer block on a flat [M, C] token buffer (M = numWin²·tokens).
// Windowed layers attend within each window (B=numWin², T=tokens/window); global
// layers attend across all tokens (B=1, T=M). Because the windowed→global reshape
// is a plain row-major view, the flat buffer is reused unchanged — only (B,T) flip.
Tensor RfDetrVk::vitLayer_(const Tensor& x, int layerIdx, bool global) {
    const uint32_t C = cfg_.embedDim;
    const uint32_t M = x.numel() / C;
    const uint32_t numWinSq = cfg_.numWindows * cfg_.numWindows;
    const uint32_t B = global ? 1u : numWinSq;
    const uint32_t T = M / B;
    const std::string pfx = "backbone.0.encoder.encoder.encoder.layer." + std::to_string(layerIdx) + ".";

    Tensor n1 = layerNorm_(x, pfx + "norm1.weight", pfx + "norm1.bias", C);
    // Fused QKV: one [M,3D] GEMM instead of three [M,D] GEMMs (better occupancy,
    // n1 read once); the attention shaders index Q/K/V out of the fused buffer.
    Tensor qkv = linear_(n1, pfx + "attention.attention.qkv.weight", pfx + "attention.attention.qkv.bias");
    Tensor scores = attnVitScores_(qkv, T, B);
    Tensor probs  = softmaxRows_(scores, B * cfg_.numHeads * T, T);
    Tensor ctx = attnVitApply_(probs, qkv, T, B);
    Tensor ao = linear_(ctx, pfx + "attention.output.dense.weight", pfx + "attention.output.dense.bias");
    Tensor x1 = scaleAdd_(x, ao, pfx + "layer_scale1.lambda1", C);// x + ao*lambda1

    Tensor n2 = layerNorm_(x1, pfx + "norm2.weight", pfx + "norm2.bias", C);
    Tensor h1 = linear_(n2, pfx + "mlp.fc1.weight", pfx + "mlp.fc1.bias", /*act=GELU*/ 1u);
    Tensor h2 = linear_(h1, pfx + "mlp.fc2.weight", pfx + "mlp.fc2.bias");
    return scaleAdd_(x1, h2, pfx + "layer_scale2.lambda1", C);// x1 + h2*lambda2
}

// ── projector primitives (CHW feature maps stored [C, HW]) ──

Tensor RfDetrVk::layerNormChw_(const Tensor& x, const std::string& gKey, const std::string& bKey,
                               uint32_t C, uint32_t HW, uint32_t act) {
    LnChwParams lp{C, HW, 1e-6f, act};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, weights_.at(gKey).buffer, weights_.at(bKey).buffer, out.buffer};
    vk_.dispatch(lnChwPipe_, ssbos, &lp, sizeof(lp), divCeil(HW, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::unwindow_(const Tensor& lnHidden, uint32_t outC, uint32_t grid) {
    UnwindowParams up{outC, grid, grid / uint32_t(cfg_.numWindows), uint32_t(cfg_.numWindows), 0u, 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({outC, grid, grid});
    std::vector<VkBuffer> ssbos = {lnHidden.buffer, out.buffer};
    vk_.dispatch(unwindowPipe_, ssbos, &up, sizeof(up), divCeil(outC * grid * grid, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::sliceCh_(const Tensor& src, uint32_t srcOff, uint32_t Cn, uint32_t HW) {
    ChCopyParams cp{Cn, HW, srcOff, 0u};
    // Keep the [C,H,W] spatial layout so downstream 3x3 convs infer H,W correctly
    // (a flat {Cn,HW} shape would be read as H=HW,W=1 and break neighbour sampling).
    Tensor out = vk_.createTensorV({Cn, src.H(), src.W()});
    std::vector<VkBuffer> ssbos = {src.buffer, out.buffer};
    vk_.dispatch(chCopyPipe_, ssbos, &cp, sizeof(cp), divCeil(Cn * HW, 64), 1, 1);
    return out;
}

void RfDetrVk::placeCh_(const Tensor& src, const Tensor& dst, uint32_t Cn, uint32_t HW, uint32_t dstOff) {
    ChCopyParams cp{Cn, HW, 0u, dstOff};
    std::vector<VkBuffer> ssbos = {src.buffer, dst.buffer};
    vk_.dispatch(chCopyPipe_, ssbos, &cp, sizeof(cp), divCeil(Cn * HW, 64), 1, 1);
}

// conv(no bias, pad=k/2) -> channel-wise LayerNorm (+optional SiLU).
Tensor RfDetrVk::convX_(const Tensor& x, const std::string& pfx, int k, uint32_t HW, bool silu) {
    int pad = k / 2;
    Tensor c = conv_(x, pfx + ".conv.weight", "", 1, 1, pad, pad);
    uint32_t outC = c.C();
    return layerNormChw_(c, pfx + ".bn.weight", pfx + ".bn.bias", outC, HW, silu ? 2u : 0u);
}

Tensor RfDetrVk::bottleneck_(const Tensor& x, const std::string& pfx, uint32_t HW) {
    Tensor h = convX_(x, pfx + ".cv1", 3, HW, true);
    return convX_(h, pfx + ".cv2", 3, HW, true);// shortcut=False for the projector
}

// C2f(c1, 256, n=3): cv1 1x1 -> split(128,128) -> 3 bottlenecks chained -> cat(5x128) -> cv2 1x1.
Tensor RfDetrVk::c2f_(const Tensor& x, const std::string& pfx, uint32_t HW) {
    const uint32_t c = 128;// hidden = out_channels(256) * e(0.5)
    Tensor cv1 = convX_(x, pfx + ".cv1", 1, HW, true);// [256, HW] = [y0|y1]
    Tensor y1 = sliceCh_(cv1, c, c, HW);              // channels 128..255
    Tensor y2 = bottleneck_(y1, pfx + ".m.0", HW);
    Tensor y3 = bottleneck_(y2, pfx + ".m.1", HW);
    Tensor y4 = bottleneck_(y3, pfx + ".m.2", HW);
    Tensor cat = vk_.createTensorV({5u * c, cv1.H(), cv1.W()});// [640, H, W]
    placeCh_(cv1, cat, 2u * c, HW, 0u);               // y0,y1 -> 0..255
    placeCh_(y2,  cat, c, HW, 2u * c);                // y2 -> 256..383
    placeCh_(y3,  cat, c, HW, 3u * c);                // y3 -> 384..511
    placeCh_(y4,  cat, c, HW, 4u * c);                // y4 -> 512..639
    return convX_(cat, pfx + ".cv2", 1, HW, true);    // [256, HW]
}

Tensor RfDetrVk::projector_(const std::array<Tensor, 4>& taps, uint32_t grid) {
    const uint32_t HW = grid * grid;
    const uint32_t C = cfg_.embedDim;
    Tensor big = vk_.createTensorV({4u * C, grid, grid});// [1536, grid, grid]
    for (uint32_t j = 0; j < 4; ++j) placeCh_(taps[j], big, C, HW, j * C);
    Tensor c2fOut = c2f_(big, "backbone.0.projector.stages.0.0", HW);
    return layerNormChw_(c2fOut, "backbone.0.projector.stages.0.1.weight",
                         "backbone.0.projector.stages.0.1.bias", c2fOut.C(), HW, 0u);
}

// ── decoder primitives ──

Tensor RfDetrVk::uploadArena_(const std::vector<uint32_t>& shape, const float* data) {
    Tensor t = vk_.createTensorV(shape);
    vk_.upload(t.buffer, data, t.bytes());
    return t;
}

Tensor RfDetrVk::relu_(const Tensor& x) {
    uint32_t n = x.numel();
    GeluParams gp{n, 0u, 0u, 0u};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(reluPipe_, ssbos, &gp, sizeof(gp), divCeil(n, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::transposeChw2Tokens_(const Tensor& x) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    TransposeParams tp{C, H, W, 0u};
    Tensor out = vk_.createTensorV({H * W, C});
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(transposeT2tPipe_, ssbos, &tp, sizeof(tp), divCeil(H * W * C, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::bboxMLP_(const Tensor& x, const std::string& pfx) {
    Tensor h = relu_(linear_(x, pfx + ".layers.0.weight", pfx + ".layers.0.bias"));
    h = relu_(linear_(h, pfx + ".layers.1.weight", pfx + ".layers.1.bias"));
    return linear_(h, pfx + ".layers.2.weight", pfx + ".layers.2.bias");
}

Tensor RfDetrVk::attnScores_(const Tensor& qkv, uint32_t M, uint32_t H) {
    uint32_t D = qkv.shape.back() / 3u;
    uint32_t d = D / H;
    AttnParams ap{M, H, d, 3u * D, 1.0f / std::sqrt(float(d)), 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({H, M, M});
    std::vector<VkBuffer> ssbos = {qkv.buffer, out.buffer};
    vk_.dispatch(attnScoresPipe_, ssbos, &ap, sizeof(ap), divCeil(M, 8), divCeil(M, 8), H);
    return out;
}

Tensor RfDetrVk::attnApply_(const Tensor& qkv, const Tensor& attn, uint32_t M, uint32_t H) {
    uint32_t D = qkv.shape.back() / 3u;
    uint32_t d = D / H;
    AttnParams ap{M, H, d, 3u * D, 1.0f / std::sqrt(float(d)), 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({M, D});
    std::vector<VkBuffer> ssbos = {qkv.buffer, attn.buffer, out.buffer};
    vk_.dispatch(attnApplyPipe_, ssbos, &ap, sizeof(ap), divCeil(d, 8), divCeil(M, 8), H);
    return out;
}

Tensor RfDetrVk::msDeformAttn_(const Tensor& value, uint32_t H, uint32_t W,
                               const Tensor& refPtsXy, const Tensor& offsets, const Tensor& attnW,
                               uint32_t numHeads) {
    uint32_t D = value.shape.back();
    uint32_t d = D / numHeads;
    uint32_t Nq = refPtsXy.shape[0];
    uint32_t L = 1u, P = cfg_.decPoints;
    std::vector<uint32_t> shapesBuf = {H, W};
    Tensor shapeGpu = vk_.createRaw(2 * sizeof(uint32_t));
    vk_.upload(shapeGpu.buffer, shapesBuf.data(), 2 * sizeof(uint32_t));
    uint32_t zero = 0u;
    Tensor startGpu = vk_.createRaw(sizeof(uint32_t));
    vk_.upload(startGpu.buffer, &zero, sizeof(uint32_t));
    MsDeformParams mp{Nq, numHeads, d, L, P, 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({Nq, D});
    std::vector<VkBuffer> ssbos = {value.buffer, shapeGpu.buffer, startGpu.buffer,
                                   refPtsXy.buffer, offsets.buffer, attnW.buffer, out.buffer};
    vk_.dispatch(msDeformAttnPipe_, ssbos, &mp, sizeof(mp), divCeil(d, 8), divCeil(Nq, 8), numHeads);
    return out;
}

// Two-stage deformable decoder + heads. memory = proj0 in [HW, 256] token layout.
void RfDetrVk::decoder_(const Tensor& memory, uint32_t gridH, uint32_t gridW, ForwardOut& out) {
    const uint32_t D = cfg_.hiddenDim, Nq = cfg_.numQueries, NC = cfg_.numClasses;
    const uint32_t totalTokens = gridH * gridW;

    // ── Frame A: enc_output projection + first-stage class/bbox over all tokens ──
    std::vector<float> encClass(size_t(totalTokens) * NC), encBboxDelta(size_t(totalTokens) * 4);
    {
        vk_.beginFrame();
        Tensor om = layerNorm_(linear_(memory, "transformer.enc_output.0.weight", "transformer.enc_output.0.bias"),
                               "transformer.enc_output_norm.0.weight", "transformer.enc_output_norm.0.bias", D);
        Tensor cls = linear_(om, "transformer.enc_out_class_embed.0.weight", "transformer.enc_out_class_embed.0.bias");
        Tensor bbx = bboxMLP_(om, "transformer.enc_out_bbox_embed.0");
        vk_.endFrame();
        // One GPU round-trip for both first-stage outputs instead of two.
        vk_.readback2(cls.buffer, encClass.data(), cls.bytes(),
                      bbx.buffer, encBboxDelta.data(), bbx.bytes());
        vk_.resetArena();
    }

    // learned refpoint_embed[:Nq] (modulates the two-stage proposals via reparam),
    // cached on the CPU at load — no per-inference readback.
    const std::vector<float>& learnedRef = learnedRefCpu_;

    // ── CPU: grid proposals → reparam coords → top-Nq by max class logit ──
    std::vector<float> encCoord(size_t(totalTokens) * 4);
    for (uint32_t gy = 0; gy < gridH; ++gy)
        for (uint32_t gx = 0; gx < gridW; ++gx) {
            uint32_t t = gy * gridW + gx;
            float px = (float(gx) + 0.5f) / float(gridW), py = (float(gy) + 0.5f) / float(gridH);
            const float pw = 0.05f, ph = 0.05f;// 0.05 * 2^lvl, lvl=0
            const float* d = &encBboxDelta[size_t(t) * 4];
            encCoord[t * 4 + 0] = d[0] * pw + px;
            encCoord[t * 4 + 1] = d[1] * ph + py;
            encCoord[t * 4 + 2] = std::exp(d[2]) * pw;
            encCoord[t * 4 + 3] = std::exp(d[3]) * ph;
        }
    std::vector<std::pair<float, uint32_t>> sc(totalTokens);
    for (uint32_t t = 0; t < totalTokens; ++t) {
        float m = encClass[size_t(t) * NC];
        for (uint32_t c = 1; c < NC; ++c) m = std::max(m, encClass[size_t(t) * NC + c]);
        sc[t] = {m, t};
    }
    std::partial_sort(sc.begin(), sc.begin() + Nq, sc.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Decoder reference (fixed across layers: lite_refpoint_refine) + sine query_pos.
    std::vector<float> refCpu(size_t(Nq) * 4), refXyCpu(size_t(Nq) * 2), qsine(size_t(Nq) * 512);
    const float twoPi = 2.0f * 3.14159265358979323846f;
    const uint32_t dim = 128;
    std::vector<float> dimt(dim);
    for (uint32_t j = 0; j < dim; ++j) dimt[j] = std::pow(10000.0f, 2.0f * float(j / 2) / float(dim));
    auto sineEmbed = [&](float v, float* dst) {// v already includes the 2π scale
        for (uint32_t k = 0; k < dim / 2; ++k) {
            dst[2 * k]     = std::sin(v / dimt[2 * k]);
            dst[2 * k + 1] = std::cos(v / dimt[2 * k + 1]);
        }
    };
    for (uint32_t i = 0; i < Nq; ++i) {
        const float* ts = &encCoord[size_t(sc[i].second) * 4];
        const float* lr = &learnedRef[size_t(i) * 4];
        float cx = lr[0] * ts[2] + ts[0];
        float cy = lr[1] * ts[3] + ts[1];
        float w  = std::exp(lr[2]) * ts[2];
        float h  = std::exp(lr[3]) * ts[3];
        refCpu[i * 4 + 0] = cx; refCpu[i * 4 + 1] = cy; refCpu[i * 4 + 2] = w; refCpu[i * 4 + 3] = h;
        refXyCpu[i * 2 + 0] = cx; refXyCpu[i * 2 + 1] = cy;
        float* qs = &qsine[size_t(i) * 512];// order: pos_y, pos_x, pos_w, pos_h
        sineEmbed(cy * twoPi, qs + 0);
        sineEmbed(cx * twoPi, qs + 128);
        sineEmbed(w  * twoPi, qs + 256);
        sineEmbed(h  * twoPi, qs + 384);
    }

    // ── Frame B: query_pos + 3 decoder layers (refs fixed → single frame) + heads ──
    Tensor tgtInit{weights_.at("query_feat.weight").buffer, {Nq, D}};// first Nq rows
    std::vector<float> bboxDeltaCpu(size_t(Nq) * 4);
    {
        vk_.beginFrame();
        Tensor qsineT = uploadArena_({Nq, 512u}, qsine.data());
        Tensor refT   = uploadArena_({Nq, 4u}, refCpu.data());
        Tensor refXyT = uploadArena_({Nq, 2u}, refXyCpu.data());
        std::vector<uint32_t> lvl = {gridH, gridW};
        Tensor lvlT = vk_.createRaw(2 * sizeof(uint32_t));
        vk_.upload(lvlT.buffer, lvl.data(), 2 * sizeof(uint32_t));

        Tensor qp = relu_(linear_(qsineT, "transformer.decoder.ref_point_head.layers.0.weight",
                                          "transformer.decoder.ref_point_head.layers.0.bias"));
        qp = linear_(qp, "transformer.decoder.ref_point_head.layers.1.weight",
                         "transformer.decoder.ref_point_head.layers.1.bias");

        Tensor tgt = tgtInit;
        for (int layer = 0; layer < cfg_.decLayers; ++layer) {
            std::string p = "transformer.decoder.layers." + std::to_string(layer);
            // self-attention (combined QKV; q=k=tgt+query_pos, v=tgt)
            Tensor yqk = linear_(add_(tgt, qp), p + ".self_attn.in_proj_weight", p + ".self_attn.in_proj_bias");
            Tensor yt  = linear_(tgt, p + ".self_attn.in_proj_weight", p + ".self_attn.in_proj_bias");
            Tensor qkv = vk_.createTensorV({Nq, 3u * D});
            QkvSpliceParams sp{Nq, D, 0u, 0u};
            { std::vector<VkBuffer> s = {yqk.buffer, yt.buffer, qkv.buffer};
              vk_.dispatch(qkvSplicePipe_, s, &sp, sizeof(sp), divCeil(Nq * 3u * D, 64), 1, 1); }
            Tensor sSm   = softmaxRows_(attnScores_(qkv, Nq, cfg_.saHeads), cfg_.saHeads * Nq, Nq);
            Tensor heads = attnApply_(qkv, sSm, Nq, cfg_.saHeads);
            Tensor attnOut = linear_(heads, p + ".self_attn.out_proj.weight", p + ".self_attn.out_proj.bias");
            Tensor n1 = layerNorm_(add_(tgt, attnOut), p + ".norm1.weight", p + ".norm1.bias", D);

            // cross-attention (deformable, single level)
            Tensor cq = add_(n1, qp);
            Tensor value = linear_(memory, p + ".cross_attn.value_proj.weight", p + ".cross_attn.value_proj.bias");
            Tensor rawOff = linear_(cq, p + ".cross_attn.sampling_offsets.weight", p + ".cross_attn.sampling_offsets.bias");
            Tensor rawAw  = linear_(cq, p + ".cross_attn.attention_weights.weight", p + ".cross_attn.attention_weights.bias");
            rawAw.shape = {Nq * uint32_t(cfg_.caHeads), uint32_t(cfg_.decPoints)};// softmax over L*P (=P, since L=1) per head
            Tensor awSm = softmaxRows_(rawAw, Nq * cfg_.caHeads, cfg_.decPoints);
            awSm.shape = {Nq, uint32_t(cfg_.caHeads * cfg_.decPoints)};
            { OffPreParams opp{Nq, uint32_t(cfg_.caHeads), 1u, uint32_t(cfg_.decPoints)};
              uint32_t stride = cfg_.caHeads * 1u * cfg_.decPoints * 2u;
              std::vector<VkBuffer> s = {refT.buffer, lvlT.buffer, rawOff.buffer};
              vk_.dispatch(offsetPreprocessPipe_, s, &opp, sizeof(opp), divCeil(Nq * stride, 64), 1, 1); }
            Tensor crossOut  = msDeformAttn_(value, gridH, gridW, refXyT, rawOff, awSm, cfg_.caHeads);
            Tensor crossProj = linear_(crossOut, p + ".cross_attn.output_proj.weight", p + ".cross_attn.output_proj.bias");
            Tensor n2 = layerNorm_(add_(n1, crossProj), p + ".norm2.weight", p + ".norm2.bias", D);

            // FFN
            Tensor ff = relu_(linear_(n2, p + ".linear1.weight", p + ".linear1.bias"));
            ff = linear_(ff, p + ".linear2.weight", p + ".linear2.bias");
            tgt = layerNorm_(add_(n2, ff), p + ".norm3.weight", p + ".norm3.bias", D);
        }
        Tensor hs = layerNorm_(tgt, "transformer.decoder.norm.weight", "transformer.decoder.norm.bias", D);
        Tensor logits = linear_(hs, "class_embed.weight", "class_embed.bias");
        Tensor bboxDelta = bboxMLP_(hs, "bbox_embed");
        vk_.endFrame();
        out.predLogits.resize(logits.numel());
        // One round-trip for both head outputs.
        vk_.readback2(logits.buffer, out.predLogits.data(), logits.bytes(),
                      bboxDelta.buffer, bboxDeltaCpu.data(), bboxDelta.bytes());
        vk_.resetArena();
    }

    // pred_boxes = reparam(bbox_embed(hs), ref) with the fixed decoder reference.
    out.predBoxes.resize(size_t(Nq) * 4);
    for (uint32_t i = 0; i < Nq; ++i) {
        const float* d = &bboxDeltaCpu[size_t(i) * 4];
        const float* r = &refCpu[size_t(i) * 4];
        out.predBoxes[i * 4 + 0] = d[0] * r[2] + r[0];
        out.predBoxes[i * 4 + 1] = d[1] * r[3] + r[1];
        out.predBoxes[i * 4 + 2] = std::exp(d[2]) * r[2];
        out.predBoxes[i * 4 + 3] = std::exp(d[3]) * r[3];
    }
}

RfDetrVk::ForwardOut RfDetrVk::runForward(const std::vector<float>& chw, bool captureIntermediates) {
    const uint32_t R = cfg_.resolution;
    if (chw.size() != size_t(3) * R * R)
        throw std::runtime_error("RfDetrVk::runForward: input must be 3*384*384 floats");

    vk_.beginFrame();
    Tensor input = vk_.createTensorV({3u, R, R});
    vk_.upload(input.buffer, chw.data(), chw.size() * sizeof(float));
    auto pe  = patchEmbed_(input);    // [384, 24, 24]
    vk_.markTimestamp("patch");
    auto emb = buildEmbeddings_(pe);  // [4, 145, 384] windowed tokens
    vk_.markTimestamp("embed");

    // Flatten the windowed tokens to [M, C] (window-major) for the layer loop.
    const uint32_t C = cfg_.embedDim;
    const uint32_t M = emb.numel() / C;
    Tensor tok = emb;
    tok.shape = {M, C};

    // 12 DINOv2 blocks; full attention at layers {3,6,9}. Capture layers 0 & 3
    // for debugging and the tap outputs at layers {2,5,8,11} for the projector.
    static const char* kVitLbl[12] = {"vit0","vit1","vit2","vit3","vit4","vit5",
                                      "vit6","vit7","vit8","vit9","vit10","vit11"};
    Tensor vit0, vit3;
    std::array<Tensor, 4> tapHidden;
    const int tapLayers[4] = {2, 5, 8, 11};
    for (int i = 0; i < cfg_.numLayers; ++i) {
        bool global = (i == 3 || i == 6 || i == 9);
        tok = vitLayer_(tok, i, global);
        if (i == 0) vit0 = tok;
        if (i == 3) vit3 = tok;
        for (int t = 0; t < 4; ++t)
            if (i == tapLayers[t]) tapHidden[t] = tok;
        vk_.markTimestamp(kVitLbl[i]);
    }

    // Each tap: final ViT LayerNorm (per-token over C), then de-window to [C,G,G].
    std::array<Tensor, 4> taps;
    for (int t = 0; t < 4; ++t) {
        Tensor ln = layerNorm_(tapHidden[t], "backbone.0.encoder.encoder.layernorm.weight",
                               "backbone.0.encoder.encoder.layernorm.bias", C);
        taps[t] = unwindow_(ln, C, cfg_.grid());
    }
    vk_.markTimestamp("taps");
    Tensor proj0 = projector_(taps, cfg_.grid());
    vk_.markTimestamp("proj");
    // Flatten proj0 [256,G,G] -> memory tokens [G*G,256] and stash in an owning
    // buffer so it survives the decoder's per-frame arena resets.
    Tensor memTok = transposeChw2Tokens_(proj0);
    VkTensor memoryOwned = vk_.createOwnedRaw(VkDeviceSize(memTok.numel()) * sizeof(float));
    vk_.recordCopy(memoryOwned.buffer, memTok.buffer, memTok.bytes());
    vk_.endFrame();

    ForwardOut out;
    if (captureIntermediates) {// validation only — the detect path skips these readbacks
        out.patchEmbed.resize(pe.numel());
        vk_.readback(pe.buffer, out.patchEmbed.data(), pe.bytes());
        out.patchDim = {pe.C(), pe.H(), pe.W()};
        out.embed.resize(emb.numel());
        vk_.readback(emb.buffer, out.embed.data(), emb.bytes());
        out.embedDim = {emb.C(), emb.H(), emb.W()};
        out.vit0.resize(vit0.numel());
        vk_.readback(vit0.buffer, out.vit0.data(), vit0.bytes());
        out.vit3.resize(vit3.numel());
        vk_.readback(vit3.buffer, out.vit3.data(), vit3.bytes());
        out.vitDim = {emb.C(), emb.H(), emb.W()};// [4,145,384]
        for (int t = 0; t < 4; ++t) {
            out.taps[t].resize(taps[t].numel());
            vk_.readback(taps[t].buffer, out.taps[t].data(), taps[t].bytes());
        }
        out.tapDim = {taps[0].C(), taps[0].H(), taps[0].W()};// [384,24,24]
        out.proj0.resize(proj0.numel());
        vk_.readback(proj0.buffer, out.proj0.data(), proj0.bytes());
        out.proj0Dim = {proj0.C(), proj0.H(), proj0.W()};// [256,24,24]
    }
    vk_.resetArena();

    // Two-stage deformable decoder + heads (memory is GPU-resident, owned).
    Tensor memoryView{memoryOwned.buffer, {cfg_.grid() * cfg_.grid(), uint32_t(cfg_.hiddenDim)}};
    decoder_(memoryView, cfg_.grid(), cfg_.grid(), out);
    return out;
}

// ImageNet preprocess (square resize → /255 → normalize) + sigmoid/threshold
// postprocess. classId is the COCO category id (1..90); boxes are in image pixels.
std::vector<Detection> RfDetrVk::infer(const unsigned char* rgba, int width, int height,
                                       float scoreThresh) {
    const int R = cfg_.resolution;
    const float mean[3] = {0.485f, 0.456f, 0.406f}, istd[3] = {1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f};
    std::vector<float> chw(size_t(3) * R * R);
    for (int y = 0; y < R; ++y) {
        float fy = (float(y) + 0.5f) * float(height) / float(R) - 0.5f;
        int y0 = std::max(0, std::min(height - 1, int(std::floor(fy))));
        int y1 = std::min(height - 1, y0 + 1);
        float wy = std::max(0.f, std::min(1.f, fy - float(y0)));
        for (int x = 0; x < R; ++x) {
            float fx = (float(x) + 0.5f) * float(width) / float(R) - 0.5f;
            int x0 = std::max(0, std::min(width - 1, int(std::floor(fx))));
            int x1 = std::min(width - 1, x0 + 1);
            float wx = std::max(0.f, std::min(1.f, fx - float(x0)));
            for (int c = 0; c < 3; ++c) {
                auto px = [&](int xx, int yy) { return float(rgba[(size_t(yy) * width + xx) * 4 + c]); };
                float top = px(x0, y0) * (1 - wx) + px(x1, y0) * wx;
                float bot = px(x0, y1) * (1 - wx) + px(x1, y1) * wx;
                float v = (top * (1 - wy) + bot * wy) / 255.f;
                chw[size_t(c) * R * R + size_t(y) * R + x] = (v - mean[c]) * istd[c];
            }
        }
    }

    ForwardOut fw = runForward(chw, /*captureIntermediates=*/false);

    std::vector<Detection> dets;
    auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };
    for (uint32_t q = 0; q < cfg_.numQueries; ++q) {
        for (uint32_t c = 0; c < cfg_.numClasses; ++c) {
            float s = sigmoid(fw.predLogits[size_t(q) * cfg_.numClasses + c]);
            if (s < scoreThresh) continue;
            const float* b = &fw.predBoxes[size_t(q) * 4];
            float cx = b[0], cy = b[1], w = b[2], h = b[3];
            dets.push_back(Detection{int(c), s,
                                     (cx - w * 0.5f) * width, (cy - h * 0.5f) * height,
                                     (cx + w * 0.5f) * width, (cy + h * 0.5f) * height});
        }
    }
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    return dets;
}

}// namespace rfdetr
