#include "RfDetrVk.hpp"

#include "WeightLoader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

// gemm.comp's builds: B layout (nn.Linear weight, 1x1 conv input, implicit im2col) x tile.
#include "threepp/renderers/vulkan/shaders/gemm.comp.rfdetr_nk128x64.spv.h"
#include "threepp/renderers/vulkan/shaders/gemm.comp.rfdetr_nk64x64.spv.h"
#include "threepp/renderers/vulkan/shaders/gemm.comp.rfdetr_kn128x64.spv.h"
#include "threepp/renderers/vulkan/shaders/gemm.comp.rfdetr_kn64x64.spv.h"
#include "threepp/renderers/vulkan/shaders/gemm.comp.rfdetr_conv128x64.spv.h"
#include "threepp/renderers/vulkan/shaders/gemm.comp.rfdetr_conv64x64.spv.h"
#include "threepp/renderers/vulkan/shaders/gemm_reduce.comp.rfdetr.spv.h"
// gemm_cm.comp: the tensor-core GEMM (fp16 products, fp32 sums) for nn.Linear layers.
#include "threepp/renderers/vulkan/shaders/gemm_cm.comp.rfdetr_cm64x64.spv.h"
#include "threepp/renderers/vulkan/shaders/gemm_cm.comp.rfdetr_cmkn.spv.h"
#include "threepp/renderers/vulkan/shaders/gemm_cm.comp.rfdetr_cmconv.spv.h"
#include "threepp/renderers/vulkan/shaders/layernorm.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/softmax.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_scores.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_apply.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/add.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/build_embeddings.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_vit_scores.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_vit_apply.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/unwindow.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/layernorm_chw.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/transpose_chw2tokens.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/offset_preprocess.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/msdeformattn.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/preprocess.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/proposals.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/topk_rank.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/query_refs.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_vit_flash.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_vit_flash_cm.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/attn_vit_flash_cm.comp.rfdetr_hd32.spv.h"
#include "threepp/renderers/vulkan/shaders/repeat_rows.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/group_pack.comp.rfdetr.spv.h"
#include "threepp/renderers/vulkan/shaders/group_unpack.comp.rfdetr.spv.h"

using threepp::VulkanRenderer;

namespace rfdetr {

namespace {
    struct GemmParams {
        uint32_t M, N, K, kChunk;
        uint32_t lda, ldb, ldc, ldr;
        uint32_t aOff, bOff, cOff, rOff;
        uint32_t biasOff, flags, _p0, _p1;
        uint32_t inH, inW, outW, kh;
        uint32_t kw, stride, pad, _p2;
    };
    struct GemmReduceParams {
        uint32_t M, N, splitK, flags;
        uint32_t ldc, ldr, cOff, rOff;
        uint32_t biasOff, _p0, _p1, _p2;
    };
    // gemm.comp / gemm_reduce.comp flags; the activation sits in bits 8..11.
    constexpr uint32_t kGemmBias = 1u, kGemmRes = 2u, kGemmGamma = 4u, kGemmPartial = 8u, kGemmBiasRow = 16u;
    struct LNParams        { uint32_t M, D; float eps, _pad; };
    struct SoftmaxParams   { uint32_t M, N, _p0, _p1; };
    struct AttnParams      { uint32_t M, H, d, stride; float scale; uint32_t _p0, _p1, _p2; };
    struct AttnVitParams   { uint32_t T, H, d, B; float scale; uint32_t _p0, _p1, _p2; };
    struct AddParams       { uint32_t n, _p0, _p1, _p2; };
    struct EmbedParams     { uint32_t C, grid, winSide, numWin; };
    struct LnChwParams     { uint32_t C, HW; float eps; uint32_t act, outOff, _p0, _p1, _p2; };
    struct UnwindowParams  { uint32_t C, grid, winSide, numWin, chOffset, _p0, _p1, _p2; };
    struct TransposeParams { uint32_t C, H, W, _p; };
    struct OffPreParams    { uint32_t Nq, numHeads, numLevels, numPoints; };
    struct MsDeformParams  { uint32_t Nq, H, d, L, P, _p0, _p1, _p2; };
    struct PreprocessParams{ uint32_t W, H, R, _p0; };
    struct ProposalParams  { uint32_t T, gridW, gridH, NC; };
    struct TopKParams      { uint32_t T, K, _p0, _p1; };
    struct QueryRefParams  { uint32_t Nq, _p0, _p1, _p2; };
    struct RepeatParams    { uint32_t R, W, K, _p0; };
    struct GroupParams     { uint32_t N, K, W, _p0; };

    uint32_t divCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

    // IEEE binary16, round to nearest even.
    uint16_t toHalf(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));
        const uint32_t sign = (x >> 16) & 0x8000u, mant = x & 0x7FFFFFu;
        const int32_t exp = int32_t((x >> 23) & 0xFFu) - 127 + 15;
        if (((x >> 23) & 0xFFu) == 0xFFu) return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));// inf, nan
        if (exp >= 31) return uint16_t(sign | 0x7C00u);                                          // overflow
        if (exp <= 0) {                                                                          // subnormal
            if (exp < -10) return uint16_t(sign);
            const uint32_t m = mant | 0x800000u, shift = uint32_t(14 - exp);
            uint32_t h = m >> shift;
            const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
            if (rem > half || (rem == half && (h & 1u))) ++h;
            return uint16_t(sign | h);
        }
        uint32_t h = (uint32_t(exp) << 10) | (mant >> 13);
        const uint32_t rem = mant & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;// a carry rounds up into the exponent
        return uint16_t(sign | h);
    }

    // A weight shape as text, for the loader's errors.
    std::string shapeStr(const std::vector<uint32_t>& s) {
        std::string out = "[";
        for (size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
        return out + "]";
    }
}// namespace

// Per-variant overrides. Every variant shares the dinov2_windowed_small backbone
// + single-scale P4 head (the struct defaults = Nano); Nano/Small/Medium differ in
// input resolution and decoder depth, Keypoint (RF-DETR Keypoint preview) also in
// patch size, queries, classes (class 0 and person) and the 17 COCO keypoints per person.
// Pinned from rfdetr/config.py.
RfDetrConfig RfDetrConfig::forVariant(RfDetrVariant v) {
    RfDetrConfig c;// defaults are Nano
    switch (v) {
        case RfDetrVariant::Nano:   c.resolution = 384; c.decLayers = 2; break;
        case RfDetrVariant::Small:  c.resolution = 512; c.decLayers = 3; break;
        case RfDetrVariant::Medium: c.resolution = 576; c.decLayers = 4; break;
        case RfDetrVariant::Keypoint:
            c.resolution = 576; c.patch = 12; c.decLayers = 4;
            c.numQueries = 100; c.numClasses = 2; c.numKeypoints = 17;
            break;
    }
    return c;
}

RfDetrVk::RfDetrVk(VulkanRenderer& r, RfDetrVariant variant, bool tensorCores)
    : vk_(static_cast<VkInstance>(r.nativeInstance()),
          static_cast<VkDevice>(r.nativeDevice()),
          static_cast<VkPhysicalDevice>(r.nativePhysicalDevice()),
          static_cast<VkQueue>(r.nativeGraphicsQueue()),
          r.graphicsQueueFamily()),
      cfg_(RfDetrConfig::forVariant(variant)),
      tensorCores_(tensorCores && vk_.coopMatF16()) {

    // gemm.comp: [B layout][tile] (kTiles in gemm_()).
    gemmPipe_[0][0] = vk_.createPipe(kRfGemmNK128x64Spv, sizeof(kRfGemmNK128x64Spv), 6, sizeof(GemmParams), "gemm_nk_128x64");
    gemmPipe_[0][1] = vk_.createPipe(kRfGemmNK64x64Spv, sizeof(kRfGemmNK64x64Spv), 6, sizeof(GemmParams), "gemm_nk_64x64");
    gemmPipe_[1][0] = vk_.createPipe(kRfGemmKN128x64Spv, sizeof(kRfGemmKN128x64Spv), 6, sizeof(GemmParams), "gemm_kn_128x64");
    gemmPipe_[1][1] = vk_.createPipe(kRfGemmKN64x64Spv, sizeof(kRfGemmKN64x64Spv), 6, sizeof(GemmParams), "gemm_kn_64x64");
    gemmPipe_[2][0] = vk_.createPipe(kRfGemmCONV128x64Spv, sizeof(kRfGemmCONV128x64Spv), 6, sizeof(GemmParams), "gemm_conv_128x64");
    gemmPipe_[2][1] = vk_.createPipe(kRfGemmCONV64x64Spv, sizeof(kRfGemmCONV64x64Spv), 6, sizeof(GemmParams), "gemm_conv_64x64");
    if (tensorCores_) {// gemm_cm.comp: [B layout]
        gemmCmPipe_[0] = vk_.createPipe(kRfGemmCm64x64Spv, sizeof(kRfGemmCm64x64Spv), 6, sizeof(GemmParams), "gemm_cm_nk");
        gemmCmPipe_[1] = vk_.createPipe(kRfGemmCmKnSpv, sizeof(kRfGemmCmKnSpv), 6, sizeof(GemmParams), "gemm_cm_kn");
        gemmCmPipe_[2] = vk_.createPipe(kRfGemmCmConvSpv, sizeof(kRfGemmCmConvSpv), 6, sizeof(GemmParams), "gemm_cm_conv");
    }
    std::cout << "RfDetrVk: matrix products " << (tensorCores_ ? "on tensor cores (fp16 products, fp32 sums)" : "in fp32")
              << "\n";
    gemmReducePipe_ = vk_.createPipe(kRfGemmReduceSpv, sizeof(kRfGemmReduceSpv), 5, sizeof(GemmReduceParams), "gemm_reduce");
    layerNormPipe_  = vk_.createPipe(kRfLayerNormSpv, sizeof(kRfLayerNormSpv), 4, sizeof(LNParams), "layernorm");
    softmaxPipe_    = vk_.createPipe(kRfSoftmaxSpv,   sizeof(kRfSoftmaxSpv),   2, sizeof(SoftmaxParams), "softmax");
    attnScoresPipe_ = vk_.createPipe(kRfAttnScoresSpv, sizeof(kRfAttnScoresSpv), 2, sizeof(AttnParams), "attn_scores");
    attnApplyPipe_  = vk_.createPipe(kRfAttnApplySpv, sizeof(kRfAttnApplySpv), 3, sizeof(AttnParams), "attn_apply");
    addPipe_        = vk_.createPipe(kRfAddSpv,       sizeof(kRfAddSpv),       3, sizeof(AddParams), "add");
    buildEmbeddingsPipe_ = vk_.createPipe(kRfBuildEmbedSpv, sizeof(kRfBuildEmbedSpv), 4, sizeof(EmbedParams), "build_embeddings");
    attnVitScoresPipe_ = vk_.createPipe(kRfAttnVitScoresSpv, sizeof(kRfAttnVitScoresSpv), 2, sizeof(AttnVitParams), "attn_vit_scores");
    attnVitApplyPipe_  = vk_.createPipe(kRfAttnVitApplySpv,  sizeof(kRfAttnVitApplySpv),  3, sizeof(AttnVitParams), "attn_vit_apply");
    unwindowPipe_ = vk_.createPipe(kRfUnwindowSpv, sizeof(kRfUnwindowSpv), 2, sizeof(UnwindowParams), "unwindow");
    lnChwPipe_    = vk_.createPipe(kRfLnChwSpv,    sizeof(kRfLnChwSpv),    4, sizeof(LnChwParams), "layernorm_chw");
    transposeT2tPipe_     = vk_.createPipe(kRfT2tSpv,       sizeof(kRfT2tSpv),       2, sizeof(TransposeParams), "transpose_chw2tokens");
    offsetPreprocessPipe_ = vk_.createPipe(kRfOffPreSpv,   sizeof(kRfOffPreSpv),    3, sizeof(OffPreParams), "offset_preprocess");
    msDeformAttnPipe_     = vk_.createPipe(kRfMsDeformSpv,  sizeof(kRfMsDeformSpv),  7, sizeof(MsDeformParams), "msdeformattn");
    preprocessPipe_       = vk_.createPipe(kRfPreprocessSpv, sizeof(kRfPreprocessSpv), 2, sizeof(PreprocessParams), "preprocess");
    proposalsPipe_        = vk_.createPipe(kRfProposalsSpv,  sizeof(kRfProposalsSpv),  4, sizeof(ProposalParams), "proposals");
    topkRankPipe_         = vk_.createPipe(kRfTopkRankSpv,   sizeof(kRfTopkRankSpv),   2, sizeof(TopKParams), "topk_rank");
    queryRefsPipe_        = vk_.createPipe(kRfQueryRefsSpv,  sizeof(kRfQueryRefsSpv),  7, sizeof(QueryRefParams), "query_refs");
    if (cfg_.numKeypoints > 0) {
        repeatRowsPipe_   = vk_.createPipe(kRfRepeatRowsSpv,  sizeof(kRfRepeatRowsSpv),  2, sizeof(RepeatParams), "repeat_rows");
        groupPackPipe_    = vk_.createPipe(kRfGroupPackSpv,   sizeof(kRfGroupPackSpv),   5, sizeof(GroupParams), "group_pack");
        groupUnpackPipe_  = vk_.createPipe(kRfGroupUnpackSpv, sizeof(kRfGroupUnpackSpv), 6, sizeof(GroupParams), "group_unpack");
    }

    // Fused attention when the device has clustered subgroup adds and the head dim
    // is the kernel's 64; otherwise scores + softmax + apply.
    // With tensor cores, its matrix products in fp16 (attn_vit_flash_cm.comp).
    useFlashAttn_ = vk_.subgroupClustered() && cfg_.embedDim / cfg_.numHeads == 64;
    const bool flashCm = useFlashAttn_ && tensorCores_;
    if (flashCm)
        attnFlashPipe_ = vk_.createPipe(kRfAttnFlashCmSpv, sizeof(kRfAttnFlashCmSpv), 2, sizeof(AttnVitParams), "attn_vit_flash_cm");
    else if (useFlashAttn_)
        attnFlashPipe_ = vk_.createPipe(kRfAttnFlashSpv, sizeof(kRfAttnFlashSpv), 2, sizeof(AttnVitParams), "attn_vit_flash");
    std::cout << "RfDetrVk: ViT attention " << (flashCm ? "fused, on tensor cores" : useFlashAttn_ ? "fused" : "in three passes")
              << "\n";
    // The decoders' self-attention (head dim 32): one pass on tensor cores, else three.
    if (tensorCores_ && vk_.subgroupClustered())
        attnCm32Pipe_ = vk_.createPipe(kRfAttnFlashCm32Spv, sizeof(kRfAttnFlashCm32Spv), 2, sizeof(AttnVitParams),
                                       "attn_head32_cm");

    // Partial-sum scratch for split-K GEMMs: room for 4 planes of [maxTokens, 512]
    // (gemm_() picks no split that would not fit). One buffer reused by every split
    // GEMM; the per-dispatch barriers serialize access.
    splitKPartialFloats_ = uint64_t(4) * uint64_t(cfg_.maxTokens()) * 512ull;
    splitKPartials_ = vk_.createOwnedRaw(VkDeviceSize(splitKPartialFloats_) * sizeof(float));

    // Deformable-attention spatial_shapes [grid,grid] + level_start [0] are constant
    // for the single-scale P4 decoder — upload once here instead of twice per layer.
    const uint32_t g = cfg_.grid();
    const uint32_t shapeInit[2] = {g, g};
    const uint32_t startInit = 0u;
    msShapeBuf_ = vk_.createOwnedRaw(sizeof(shapeInit));
    vk_.upload(msShapeBuf_.buffer, shapeInit, sizeof(shapeInit));
    msStartBuf_ = vk_.createOwnedRaw(sizeof(startInit));
    vk_.upload(msStartBuf_.buffer, &startInit, sizeof(startInit));

    // Frequencies of the decoder's sine query embedding (query_refs.comp).
    std::array<float, 128> dimt{};
    for (uint32_t k = 0; k < 128; ++k) dimt[k] = std::pow(10000.0f, 2.0f * float(k / 2) / 128.0f);
    dimtBuf_ = vk_.createOwnedRaw(sizeof(dimt));
    vk_.upload(dimtBuf_.buffer, dimt.data(), sizeof(dimt));
}

RfDetrVk::~RfDetrVk() {
    for (auto& row : gemmPipe_)
        for (auto& pipe : row) vk_.destroyPipe(pipe);
    for (auto& pipe : gemmCmPipe_) vk_.destroyPipe(pipe);
    vk_.destroyPipe(gemmReducePipe_);
    vk_.destroyPipe(attnFlashPipe_);
    vk_.destroyPipe(attnCm32Pipe_);
    vk_.destroyPipe(layerNormPipe_);
    vk_.destroyPipe(softmaxPipe_);
    vk_.destroyPipe(attnScoresPipe_);
    vk_.destroyPipe(attnApplyPipe_);
    vk_.destroyPipe(addPipe_);
    vk_.destroyPipe(buildEmbeddingsPipe_);
    vk_.destroyPipe(attnVitScoresPipe_);
    vk_.destroyPipe(attnVitApplyPipe_);
    vk_.destroyPipe(unwindowPipe_);
    vk_.destroyPipe(lnChwPipe_);
    vk_.destroyPipe(transposeT2tPipe_);
    vk_.destroyPipe(offsetPreprocessPipe_);
    vk_.destroyPipe(msDeformAttnPipe_);
    vk_.destroyPipe(preprocessPipe_);
    vk_.destroyPipe(proposalsPipe_);
    vk_.destroyPipe(topkRankPipe_);
    vk_.destroyPipe(queryRefsPipe_);
    vk_.destroyPipe(repeatRowsPipe_);
    vk_.destroyPipe(groupPackPipe_);
    vk_.destroyPipe(groupUnpackPipe_);
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
                "RfDetrVk: variant/weights mismatch - config expects dec_layers=" +
                std::to_string(cfg_.decLayers) + " but checkpoint has " + std::to_string(ckptDecLayers) +
                ". Pass the RfDetrVariant matching these weights.");
    // Medium and Keypoint both have 4 decoder layers: the patch size and the
    // keypoint head tell them apart.
    const auto& patchW = weights_.at("backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight").shape;
    if (patchW.size() != 4 || patchW[2] != uint32_t(cfg_.patch) || patchW[3] != uint32_t(cfg_.patch))
        throw std::runtime_error("RfDetrVk: variant/weights mismatch - config patchifies at " +
                                 std::to_string(cfg_.patch) + " but the checkpoint's patch embedding is " +
                                 shapeStr(patchW) + ". Pass the RfDetrVariant matching these weights.");
    const bool ckptKeypoints = weights_.count("keypoint_embed.layers.0.weight") > 0;
    if (ckptKeypoints != (cfg_.numKeypoints > 0))
        throw std::runtime_error(ckptKeypoints
                                         ? "RfDetrVk: the checkpoint has a keypoint head; pass RfDetrVariant::Keypoint"
                                         : "RfDetrVk: RfDetrVariant::Keypoint needs RF-DETR Keypoint weights "
                                           "(scripts/export_rfdetr_weights.py --variant keypoint)");

    if (cfg_.numKeypoints > 0) {
        const uint32_t K = uint32_t(cfg_.numKeypoints), D = uint32_t(cfg_.hiddenDim);
        auto need = [&](const std::string& name, const std::vector<uint32_t>& shape) {
            auto it = w.shapes.find(name);
            if (it == w.shapes.end())
                throw std::runtime_error("RfDetrVk: checkpoint has no " + name + " " + shapeStr(shape));
            if (it->second != shape)
                throw std::runtime_error("RfDetrVk: " + name + " is " + shapeStr(it->second) + ", expected " + shapeStr(shape));
        };
        need("transformer.decoder.keypoint_pos_embed", {K, D});
        // The decoder's initial keypoint queries depend on learned weights only, so
        // the exporter computes them once (scripts/export_rfdetr_weights.py).
        need("derived.keypoint_queries", {uint32_t(cfg_.numQueries), K, D});

        // The group self-attention runs unmasked: with keypoints on one class, the
        // class mask over each group [1+K,1+K] blocks nothing.
        need("transformer.decoder.keypoint_class_mask", {1u + K, 1u + K});
        const auto& classMask = w.data.at("transformer.decoder.keypoint_class_mask");
        if (std::any_of(classMask.begin(), classMask.end(), [](float v) { return v != 0.f; }))
            throw std::runtime_error("RfDetrVk: the keypoint class mask blocks attention inside a group "
                                     "(keypoints on more than one class), which this port does not do");

        // The keypoint schema: one row per class with a slot, its active keypoints
        // first. One class may have keypoints, all K of them.
        auto am = w.shapes.find("_kp_active_mask");
        if (am == w.shapes.end() || am->second.size() != 2 || am->second[1] != K ||
            am->second[0] > uint32_t(cfg_.numClasses))
            throw std::runtime_error("RfDetrVk: checkpoint has no keypoint schema _kp_active_mask [<=" +
                                     std::to_string(cfg_.numClasses) + "," + std::to_string(K) + "]");
        const auto& active = w.data.at("_kp_active_mask");
        kpSchemaClasses_ = int(am->second[0]);
        kpClass_ = -1;
        for (int c = 0; c < kpSchemaClasses_; ++c) {
            const float* row = active.data() + size_t(c) * K;
            const auto n = std::count_if(row, row + K, [](float v) { return v != 0.f; });
            if (n == 0) continue;
            if (kpClass_ >= 0 || n != std::ptrdiff_t(K))
                throw std::runtime_error("RfDetrVk: the keypoint schema puts keypoints on more than one class, "
                                         "which this port does not do");
            kpClass_ = c;
        }
        if (kpClass_ < 0) throw std::runtime_error("RfDetrVk: the keypoint schema has no keypoints");

        // instance_kp_layer_scale is one scalar per layer; group_unpack takes a gamma per channel.
        for (int L = 0; L < cfg_.decLayers; ++L) {
            const std::string name = "transformer.decoder.layers." + std::to_string(L) + ".instance_kp_layer_scale";
            need(name, {1u});
            const std::vector<float> gamma(D, w.data.at(name)[0]);
            VkTensor t = vk_.createOwned({D});
            vk_.upload(t.buffer, gamma.data(), gamma.size() * sizeof(float));
            weights_.emplace(name + ".vec", std::move(t));
        }
    }

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
    // query_refs.comp reads the first cfg_.numQueries rows of refpoint_embed [N,4].
    auto rp = w.shapes.find("refpoint_embed.weight");
    if (rp == w.shapes.end() || rp->second.size() != 2 || rp->second[0] < uint32_t(cfg_.numQueries) ||
        rp->second[1] != 4u)
        throw std::runtime_error("RfDetrVk: checkpoint has no refpoint_embed [>=" +
                                 std::to_string(cfg_.numQueries) + ", 4]");

    // fp16 copies of the 2D weights (and the fused qkv) for the tensor-core GEMMs.
    weights16_.clear();
    if (tensorCores_) {
        auto upload16 =[&](const std::string& name, const std::vector<float>& data, const std::vector<uint32_t>& shape) {
            std::vector<uint16_t> h(data.size());
            for (size_t i = 0; i < data.size(); ++i) h[i] = toHalf(data[i]);
            VkTensor t = vk_.createOwnedRaw(VkDeviceSize(h.size()) * sizeof(uint16_t));
            t.shape = shape;
            vk_.upload(t.buffer, h.data(), h.size() * sizeof(uint16_t));
            weights16_.emplace(name, std::move(t));
        };
        for (const auto& [name, data] : w.data) {
            const auto& sh = w.shapes.at(name);
            if (sh.size() == 2 && sh[1] % 4u == 0u) upload16(name, data, sh);
            if (sh.size() == 4 && (sh[1] * sh[2] * sh[3]) % 4u == 0u)// a conv weight as [out_c, in_c*kh*kw]
                upload16(name, data, {sh[0], sh[1] * sh[2] * sh[3]});
        }
        for (int L = 0; L < cfg_.numLayers; ++L) {
            std::string p = "backbone.0.encoder.encoder.encoder.layer." + std::to_string(L) + ".attention.attention.";
            auto qw = w.data.find(p + "query.weight"), kw = w.data.find(p + "key.weight"), vw = w.data.find(p + "value.weight");
            if (qw == w.data.end() || kw == w.data.end() || vw == w.data.end()) continue;
            const uint32_t K = w.shapes.at(p + "query.weight")[1];
            auto fw = concat3(qw->second, kw->second, vw->second);
            upload16(p + "qkv.weight", fw, {uint32_t(fw.size() / K), K});
        }
    }
    std::cout << "RfDetrVk::loadWeights: " << weights_.size() << " GPU tensors (fp32)"
              << (weights16_.empty() ? "" : ", " + std::to_string(weights16_.size()) + " fp16 copies") << "\n";
}

// Records g with the tile and K split whose estimated time is least, among those
// that give every SM at least ~2 workgroups to switch between (few resident
// workgroups leave the SM waiting on memory: an unsplit 2308x384x1536 runs at half
// the rate of the same GEMM split in two). The estimate is the busiest SM's whole
// workgroups at the tile's rate, a launch, and for a split the reduction's traffic;
// the rates and costs were timed on an RTX 5080 (PORTING_NOTES §10).
void RfDetrVk::gemm_(const Gemm& g) {
    if (((g.K | g.lda | g.aOff) & 3u) || (g.layout != GemmB::Conv && ((g.ldb | g.bOff) & 3u)) ||
        (g.layout == GemmB::KN && (g.N & 3u)))
        throw std::runtime_error("RfDetrVk::gemm_: an operand is not float4-aligned");

    // 128x64: 8x8 per invocation, 32 k per tile, no register prefetch (the most FMAs
    // per shared read); 64x64: 4x4, 16 k, prefetch (small footprint, many per SM).
    struct TileCfg { uint32_t bm, bn, bk; float fmaPerNs; };
    static constexpr TileCfg kTiles[2] = {{128u, 64u, 32u, 215.f}, {64u, 64u, 16u, 170.f}};
    const float sms = float(vk_.smCount());
    // GEMMs with an fp16 copy of their weight (B of a linear layer, A of a conv) go to
    // gemm_cm.comp's 64x64 tiles when the device has tensor cores. Those workgroups are
    // short and want ~3 per SM to hide their loads (timed per shape on an RTX 5080,
    // PORTING_NOTES §11): the smallest split that gets there.
    const bool cm = tensorCores_ && (g.layout == GemmB::NK ? g.b16 != VK_NULL_HANDLE : g.a16 != VK_NULL_HANDLE);
    int tile = 0;
    uint32_t splits = 1, chunk = g.K;
    if (cm) {
        for (uint32_t s : {1u, 2u, 4u, 8u}) {
            const uint32_t c = divCeil(divCeil(g.K, s), 32u) * 32u;
            const uint32_t n = divCeil(g.K, c);
            if (n != s || (n > 1 && uint64_t(n) * g.M * g.N > splitKPartialFloats_)) continue;
            splits = n;
            chunk = c;
            if (float(divCeil(g.M, 64u) * divCeil(g.N, 64u) * n) >= 3.f * sms) break;
        }
    } else {
        float best = std::numeric_limits<float>::max(), mostGroups = 0.f;
        bool enough = false;// some candidate reaches the workgroup floor
        for (int t = 0; t < 2; ++t) {
            const TileCfg& tc = kTiles[t];
            for (uint32_t s : {1u, 2u, 4u, 8u}) {
                const uint32_t c = divCeil(divCeil(g.K, s), tc.bk) * tc.bk;// k per split, whole k-tiles
                const uint32_t n = divCeil(g.K, c);
                if (n != s || (n > 1 && uint64_t(n) * g.M * g.N > splitKPartialFloats_)) continue;
                const float groups = float(divCeil(g.M, tc.bm) * divCeil(g.N, tc.bn) * n);
                float ns = std::ceil(groups / sms) * float(tc.bm * tc.bn) * float(c + 64u) / tc.fmaPerNs + 1000.f;
                if (n > 1) ns += 1000.f + float(n + 1) * float(g.M) * float(g.N) * 4.f / 1.5f;// ~1.5 B/ns
                const bool ok = groups >= 1.7f * sms;
                // Past the floor the estimate decides; below it (tiny GEMMs) the most workgroups.
                if (ok ? (!enough || ns < best) : (!enough && groups > mostGroups)) {
                    best = ok ? ns : best;
                    mostGroups = groups;
                    enough = enough || ok;
                    tile = t;
                    splits = n;
                    chunk = c;
                }
            }
        }
    }
    const uint32_t epi = (g.bias ? (g.biasPerRow ? kGemmBiasRow : kGemmBias) : 0u) | (g.res ? kGemmRes : 0u) |
                         (g.gamma ? kGemmGamma : 0u) | (g.act << 8u);
    const VkBuffer none = vk_.dummy();
    const VkBuffer bias = g.bias ? g.bias : none, res = g.res ? g.res : none, gamma = g.gamma ? g.gamma : none;
    GemmParams gp{g.M, g.N, g.K, chunk,
                  g.lda, g.ldb, g.ldc, g.ldr,
                  g.aOff, g.bOff, g.cOff, g.rOff,
                  g.biasOff, splits > 1 ? kGemmPartial : epi, 0u, 0u,
                  g.inH, g.inW, g.outW, g.kh,
                  g.kw, g.stride, g.pad, 0u};
    const uint32_t bm = cm ? 64u : kTiles[tile].bm, bn = cm ? 64u : kTiles[tile].bn;
    const VkPipe& pipe = cm ? gemmCmPipe_[int(g.layout)] : gemmPipe_[int(g.layout)][tile];
    // The tensor-core GEMM reads the fp16 copy of the weight operand.
    const VkBuffer opA = cm && g.layout != GemmB::NK ? g.a16 : g.a;
    const VkBuffer opB = cm && g.layout == GemmB::NK ? g.b16 : g.b;
    vk_.dispatch(pipe, {opA, opB, bias, splits > 1 ? splitKPartials_.buffer : g.y, res, gamma},
                 &gp, sizeof(gp), divCeil(g.N, bn), divCeil(g.M, bm), splits);
    if (splits > 1) {
        GemmReduceParams rp{g.M, g.N, splits, epi, g.ldc, g.ldr, g.cOff, g.rOff, g.biasOff, 0u, 0u, 0u};
        vk_.dispatch(gemmReducePipe_, {splitKPartials_.buffer, bias, g.y, res, gamma}, &rp, sizeof(rp),
                     divCeil(g.M * g.N, 256u), 1, 1);
    }
}

// conv (no groups) as one GEMM: Y[oc, pos] = W[oc, :] · im2col(x)[:, pos], reading
// x's channels from inOff on. A 1x1 stride-1 conv reads x as it is ([C, HW] is
// the B operand); any other is gathered in the GEMM's B fetch.
Tensor RfDetrVk::conv_(const Tensor& x, const std::string& wKey, const std::string& bKey,
                       int stride, int pad, uint32_t inOff) {
    const auto& wt = weights_.at(wKey);
    const uint32_t outC = wt.shape[0], inC = wt.shape[1], kh = wt.shape[2], kw = wt.shape[3];
    const uint32_t inH = x.H(), inW = x.W();
    const uint32_t outH = (inH + 2u * uint32_t(pad) - kh) / uint32_t(stride) + 1u;
    const uint32_t outW = (inW + 2u * uint32_t(pad) - kw) / uint32_t(stride) + 1u;
    Tensor out = vk_.createTensorV({outC, outH, outW});
    Gemm g;
    g.a = wt.buffer;
    g.b = x.buffer;
    g.y = out.buffer;
    g.M = outC;
    g.N = outH * outW;
    g.K = inC * kh * kw;
    g.lda = g.K;
    g.ldc = g.N;
    g.bOff = inOff * inH * inW;
    if (!bKey.empty() && weights_.count(bKey)) {
        g.bias = weights_.at(bKey).buffer;
        g.biasPerRow = true;// one per output channel
    }
    if (auto it = weights16_.find(wKey); it != weights16_.end()) g.a16 = it->second.buffer;
    if (kh == 1 && kw == 1 && stride == 1 && pad == 0) {
        g.layout = GemmB::KN;
        g.ldb = inH * inW;
    } else {
        g.layout = GemmB::Conv;
        g.inH = inH;
        g.inW = inW;
        g.outW = outW;
        g.kh = kh;
        g.kw = kw;
        g.stride = uint32_t(stride);
        g.pad = uint32_t(pad);
    }
    gemm_(g);
    return out;
}

Tensor RfDetrVk::patchEmbed_(const Tensor& x) {
    return conv_(x,
                 "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight",
                 "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.bias",
                 cfg_.patch, 0);
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

// act(x·Wᵀ + b), and with res the residual res + gamma ⊙ that, in one GEMM. W is the
// nn.Linear weight [out=N, in=K] row-major.
Tensor RfDetrVk::linear_(const Tensor& x, const std::string& wKey, const std::string& bKey, uint32_t act,
                         const Tensor* res, const std::string& gammaKey) {
    const auto& wt = weights_.at(wKey);
    const uint32_t N = wt.shape[0], K = wt.shape[1];
    const uint32_t M = x.numel() / K;
    Tensor out = vk_.createTensorV({M, N});
    Gemm g;
    g.a = x.buffer;
    g.b = wt.buffer;
    g.y = out.buffer;
    g.M = M;
    g.N = N;
    g.K = K;
    g.lda = K;
    g.ldb = K;
    g.ldc = N;
    if (auto it = weights16_.find(wKey); it != weights16_.end()) g.b16 = it->second.buffer;
    if (!bKey.empty() && weights_.count(bKey)) g.bias = weights_.at(bKey).buffer;
    g.act = act;
    if (res) {
        g.res = res->buffer;
        g.ldr = N;
        if (!gammaKey.empty()) g.gamma = weights_.at(gammaKey).buffer;
    }
    gemm_(g);
    return out;
}

void RfDetrVk::linearRowsInto_(const Tensor& a, const std::string& wKey, const std::string& bKey,
                               uint32_t n0, uint32_t n, const Tensor& y, uint32_t cOff) {
    const auto& wt = weights_.at(wKey);
    const uint32_t K = wt.shape[1];
    Gemm g;
    g.a = a.buffer;
    g.b = wt.buffer;
    g.y = y.buffer;
    g.M = a.numel() / K;
    g.N = n;
    g.K = K;
    g.lda = K;
    g.ldb = K;
    g.ldc = y.shape.back();
    g.bOff = n0 * K;
    g.cOff = cOff;
    g.bias = weights_.at(bKey).buffer;
    g.biasOff = n0;
    if (auto it = weights16_.find(wKey); it != weights16_.end()) g.b16 = it->second.buffer;
    gemm_(g);
}

Tensor RfDetrVk::layerNorm_(const Tensor& x, const std::string& wKey, const std::string& bKey, uint32_t D) {
    if (D > 512u) throw std::runtime_error("RfDetrVk::layerNorm_: rows wider than 512");
    uint32_t M = x.numel() / D;
    LNParams lp{M, D, 1e-6f, 0.f};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, weights_.at(wKey).buffer, weights_.at(bKey).buffer, out.buffer};
    vk_.dispatch(layerNormPipe_, ssbos, &lp, sizeof(lp), divCeil(M, 4), 1, 1);// four rows per workgroup
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

Tensor RfDetrVk::softmaxRows_(const Tensor& x, uint32_t rows, uint32_t n) {
    SoftmaxParams sp{rows, n, 0u, 0u};
    Tensor out = vk_.createTensorV(x.shape);
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(softmaxPipe_, ssbos, &sp, sizeof(sp), divCeil(rows, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::attnVitScores_(const Tensor& qkv, uint32_t T, uint32_t B, uint32_t H, uint32_t d) {
    AttnVitParams ap{T, H, d, B, 1.0f / std::sqrt(float(d)), 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({B * H * T, T});
    std::vector<VkBuffer> ssbos = {qkv.buffer, out.buffer};
    vk_.dispatch(attnVitScoresPipe_, ssbos, &ap, sizeof(ap), divCeil(T, 8), divCeil(T, 8), B * H);
    return out;
}

Tensor RfDetrVk::attnVitApply_(const Tensor& attn, const Tensor& qkv, uint32_t T, uint32_t B, uint32_t H, uint32_t d) {
    AttnVitParams ap{T, H, d, B, 1.0f / std::sqrt(float(d)), 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({B * T, H * d});
    std::vector<VkBuffer> ssbos = {attn.buffer, qkv.buffer, out.buffer};
    vk_.dispatch(attnVitApplyPipe_, ssbos, &ap, sizeof(ap), divCeil(d, 8), divCeil(T, 8), B * H);
    return out;
}

Tensor RfDetrVk::attnHead32_(const Tensor& qkv, uint32_t T, uint32_t B, uint32_t H) {
    AttnVitParams ap{T, H, 32u, B, 1.0f / std::sqrt(32.0f), 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({B * T, H * 32u});
    vk_.dispatch(attnCm32Pipe_, {qkv.buffer, out.buffer}, &ap, sizeof(ap), divCeil(T, 64), H, B);
    return out;
}

Tensor RfDetrVk::attnVitFlash_(const Tensor& qkv, uint32_t T, uint32_t B) {
    const uint32_t H = cfg_.numHeads, d = cfg_.embedDim / cfg_.numHeads;
    AttnVitParams ap{T, H, d, B, 1.0f / std::sqrt(float(d)), 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({B * T, H * d});
    // 64 queries per workgroup, one head and one group per y/z.
    vk_.dispatch(attnFlashPipe_, {qkv.buffer, out.buffer}, &ap, sizeof(ap), divCeil(T, 64), H, B);
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
    Tensor ctx;
    if (useFlashAttn_) {
        ctx = attnVitFlash_(qkv, T, B);
    } else {
        const uint32_t H = cfg_.numHeads, d = C / H;
        Tensor scores = attnVitScores_(qkv, T, B, H, d);
        Tensor probs  = softmaxRows_(scores, B * H * T, T);
        ctx = attnVitApply_(probs, qkv, T, B, H, d);
    }
    // The residuals (x + lambda ⊙ branch, LayerScale) are the GEMMs' epilogues.
    Tensor x1 = linear_(ctx, pfx + "attention.output.dense.weight", pfx + "attention.output.dense.bias", 0u,
                        &x, pfx + "layer_scale1.lambda1");

    Tensor n2 = layerNorm_(x1, pfx + "norm2.weight", pfx + "norm2.bias", C);
    Tensor h1 = linear_(n2, pfx + "mlp.fc1.weight", pfx + "mlp.fc1.bias", /*act=GELU*/ 1u);
    return linear_(h1, pfx + "mlp.fc2.weight", pfx + "mlp.fc2.bias", 0u, &x1, pfx + "layer_scale2.lambda1");
}

// ── projector primitives (CHW feature maps stored [C, HW]) ──

Tensor RfDetrVk::layerNormChw_(const Tensor& x, const std::string& gKey, const std::string& bKey,
                               uint32_t C, uint32_t HW, uint32_t act, const Tensor* out, uint32_t outOff) {
    if (C > 512u) throw std::runtime_error("RfDetrVk::layerNormChw_: more than 512 channels");
    LnChwParams lp{C, HW, 1e-6f, act, outOff, 0u, 0u, 0u};
    Tensor y = out ? *out : vk_.createTensorV(x.shape);
    vk_.dispatch(lnChwPipe_, {x.buffer, weights_.at(gKey).buffer, weights_.at(bKey).buffer, y.buffer},
                 &lp, sizeof(lp), divCeil(HW, 32), 1, 1);// 32 positions per workgroup
    return y;
}

void RfDetrVk::unwindowInto_(const Tensor& lnHidden, const Tensor& dst, uint32_t chOffset) {
    const uint32_t C = cfg_.embedDim, grid = cfg_.grid();
    UnwindowParams up{C, grid, grid / uint32_t(cfg_.numWindows), uint32_t(cfg_.numWindows), chOffset, 0u, 0u, 0u};
    vk_.dispatch(unwindowPipe_, {lnHidden.buffer, dst.buffer}, &up, sizeof(up), divCeil(C * grid * grid, 64), 1, 1);
}

// conv(no bias, pad=k/2) -> channel-wise LayerNorm (+optional SiLU).
Tensor RfDetrVk::convX_(const Tensor& x, const std::string& pfx, int k, uint32_t HW, bool silu,
                        uint32_t inOff, const Tensor* out, uint32_t outOff) {
    Tensor c = conv_(x, pfx + ".conv.weight", "", 1, k / 2, inOff);
    return layerNormChw_(c, pfx + ".bn.weight", pfx + ".bn.bias", c.C(), HW, silu ? 2u : 0u, out, outOff);
}

// C2f(c1, 256, n=3): cv1 1x1 -> split(128,128) -> 3 bottlenecks chained -> cat(5x128) -> cv2 1x1.
// cv1's output [y0|y1] and each bottleneck's land in their channels of the concat
// buffer, which is where the next bottleneck reads its input.
Tensor RfDetrVk::c2f_(const Tensor& x, const std::string& pfx, uint32_t HW) {
    const uint32_t c = 128;// hidden = out_channels(256) * e(0.5)
    Tensor cat = vk_.createTensorV({5u * c, x.H(), x.W()});// [y0 y1 y2 y3 y4], 640 channels
    convX_(x, pfx + ".cv1", 1, HW, true, 0u, &cat, 0u);
    for (uint32_t i = 0; i < 3; ++i) {// bottleneck i (shortcut=False): y(i+1) -> y(i+2)
        const std::string m = pfx + ".m." + std::to_string(i);
        Tensor h = convX_(cat, m + ".cv1", 3, HW, true, (i + 1u) * c);
        convX_(h, m + ".cv2", 3, HW, true, 0u, &cat, (i + 2u) * c);
    }
    return convX_(cat, pfx + ".cv2", 1, HW, true);// [256, HW]
}

Tensor RfDetrVk::projector_(const Tensor& tapsCat, const std::string& pfx) {
    const uint32_t HW = tapsCat.H() * tapsCat.W();
    Tensor c2fOut = c2f_(tapsCat, pfx + ".stages.0.0", HW);
    return layerNormChw_(c2fOut, pfx + ".stages.0.1.weight", pfx + ".stages.0.1.bias", c2fOut.C(), HW, 0u);
}

// ── decoder primitives ──

Tensor RfDetrVk::transposeChw2Tokens_(const Tensor& x) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    TransposeParams tp{C, H, W, 0u};
    Tensor out = vk_.createTensorV({H * W, C});
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(transposeT2tPipe_, ssbos, &tp, sizeof(tp), divCeil(H * W * C, 64), 1, 1);
    return out;
}

Tensor RfDetrVk::bboxMLP_(const Tensor& x, const std::string& pfx) {
    Tensor h = linear_(x, pfx + ".layers.0.weight", pfx + ".layers.0.bias", /*act=ReLU*/ 2u);
    h = linear_(h, pfx + ".layers.1.weight", pfx + ".layers.1.bias", 2u);
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
    // spatial_shapes [H,W] + level_start [0] are constant (single-scale P4, H==W==grid)
    // and preuploaded into msShapeBuf_/msStartBuf_ in the ctor — no per-layer upload.
    (void) H; (void) W;
    MsDeformParams mp{Nq, numHeads, d, L, P, 0u, 0u, 0u};
    Tensor out = vk_.createTensorV({Nq, D});
    std::vector<VkBuffer> ssbos = {value.buffer, msShapeBuf_.buffer, msStartBuf_.buffer,
                                   refPtsXy.buffer, offsets.buffer, attnW.buffer, out.buffer};
    vk_.dispatch(msDeformAttnPipe_, ssbos, &mp, sizeof(mp), divCeil(d, 8), divCeil(Nq, 8), numHeads);
    return out;
}

Tensor RfDetrVk::repeatRows_(const Tensor& x, uint32_t K) {
    const uint32_t W = x.shape.back(), R = x.numel() / W;
    RepeatParams rp{R, W, K, 0u};
    Tensor out = vk_.createTensorV({R * K, W});
    vk_.dispatch(repeatRowsPipe_, {x.buffer, out.buffer}, &rp, sizeof(rp), divCeil(R * K * W, 256), 1, 1);
    return out;
}

// RF-DETR Keypoint's per-layer keypoint processing (TransformerDecoderLayer.forward_post
// after norm3, rfdetr/models/transformer.py), eval path: no inter-instance keypoint
// attention, keypoint cross-attention on, keypoint dim = hidden dim (the in/out
// projections around it are identities).
void RfDetrVk::keypointLayer_(int layer, Tensor& tgt, Tensor& kp, const Tensor& qpK, const Tensor& refK,
                              const Tensor& refXyK, const Tensor& kpMemory) {
    const uint32_t D = cfg_.hiddenDim, Nq = cfg_.numQueries, K = cfg_.numKeypoints, G = 1u + K;
    const uint32_t grid = cfg_.grid();
    const std::string p = "transformer.decoder.layers." + std::to_string(layer);

    // a. Group self-attention: each instance with its K keypoint queries, q = k =
    //    feature + position (the instance token's position is zero), v = feature.
    Tensor feat = vk_.createTensorV({Nq * G, D});
    Tensor featPos = vk_.createTensorV({Nq * G, D});
    { GroupParams gp{Nq, K, D, 0u};
      vk_.dispatch(groupPackPipe_, {tgt.buffer, kp.buffer, weights_.at("transformer.decoder.keypoint_pos_embed").buffer,
                                    feat.buffer, featPos.buffer},
                   &gp, sizeof(gp), divCeil(Nq * G * D, 256), 1, 1); }
    // in_proj: q and k from feature + position, v from the feature, each GEMM
    // only the rows its input needs, written straight into the fused qkv layout.
    Tensor qkv = vk_.createTensorV({Nq * G, 3u * D});
    linearRowsInto_(featPos, p + ".kp_inst_self_attn.in_proj_weight", p + ".kp_inst_self_attn.in_proj_bias",
                    0u, 2u * D, qkv, 0u);
    linearRowsInto_(feat, p + ".kp_inst_self_attn.in_proj_weight", p + ".kp_inst_self_attn.in_proj_bias",
                    2u * D, D, qkv, 2u * D);
    const uint32_t H = cfg_.saHeads, d = D / H;
    Tensor ctx = attnCm32Pipe_.pipeline ? attnHead32_(qkv, G, Nq, H)
                                        : attnVitApply_(softmaxRows_(attnVitScores_(qkv, G, Nq, H, d), Nq * H * G, G),
                                                        qkv, G, Nq, H, d);
    Tensor groupOut = linear_(ctx, p + ".kp_inst_self_attn.out_proj.weight", p + ".kp_inst_self_attn.out_proj.bias");
    // Back to instance and keypoint rows with the residuals; the instance takes the
    // group's answer through a learned (tiny) layer scale.
    Tensor tgtSum = vk_.createTensorV({Nq, D});
    Tensor kpSum = vk_.createTensorV({Nq * K, D});
    { GroupParams gp{Nq, K, D, 0u};
      vk_.dispatch(groupUnpackPipe_, {groupOut.buffer, tgt.buffer, kp.buffer,
                                      weights_.at(p + ".instance_kp_layer_scale.vec").buffer,
                                      tgtSum.buffer, kpSum.buffer},
                   &gp, sizeof(gp), divCeil(Nq * G * D, 256), 1, 1); }
    tgt = layerNorm_(tgtSum, p + ".kp_inst_norm.weight", p + ".kp_inst_norm.bias", D);
    kp = layerNorm_(kpSum, p + ".kp_norm.weight", p + ".kp_norm.bias", D);

    // b. Keypoint deformable cross-attention into the keypoint memory, sampled around
    //    the instance's reference box; the query is the keypoint state plus its
    //    instance's query_pos.
    const uint32_t NK = Nq * K, Hc = cfg_.caHeads, P = cfg_.decPoints;
    Tensor cq = add_(kp, qpK);
    Tensor value = linear_(kpMemory, p + ".kp_cross_attn.value_proj.weight", p + ".kp_cross_attn.value_proj.bias");
    Tensor rawOff = linear_(cq, p + ".kp_cross_attn.sampling_offsets.weight", p + ".kp_cross_attn.sampling_offsets.bias");
    Tensor rawAw  = linear_(cq, p + ".kp_cross_attn.attention_weights.weight", p + ".kp_cross_attn.attention_weights.bias");
    rawAw.shape = {NK * Hc, P};// softmax over the points of each head (one level)
    Tensor awSm = softmaxRows_(rawAw, NK * Hc, P);
    awSm.shape = {NK, Hc * P};
    { OffPreParams opp{NK, Hc, 1u, P};
      vk_.dispatch(offsetPreprocessPipe_, {refK.buffer, msShapeBuf_.buffer, rawOff.buffer}, &opp, sizeof(opp),
                   divCeil(NK * Hc * P * 2u, 64), 1, 1); }
    Tensor crossOut = msDeformAttn_(value, grid, grid, refXyK, rawOff, awSm, Hc);
    kp = layerNorm_(linear_(crossOut, p + ".kp_cross_attn.output_proj.weight", p + ".kp_cross_attn.output_proj.bias",
                            0u, &kp),
                    p + ".kp_cross_attn_norm.weight", p + ".kp_cross_attn_norm.bias", D);

    // c. Keypoint FFN.
    Tensor ff = linear_(kp, p + ".kp_linear1.weight", p + ".kp_linear1.bias", /*act=ReLU*/ 2u);
    kp = layerNorm_(linear_(ff, p + ".kp_linear3.weight", p + ".kp_linear3.bias", 0u, &kp),
                    p + ".kp_norm5.weight", p + ".kp_norm5.bias", D);
}

// Two-stage deformable decoder + heads, recorded into the open frame: nothing
// in here waits on the host. memory = proj0 in [HW, 256] token layout.
RfDetrVk::HeadTensors RfDetrVk::recordDecoder_(const BackboneTensors& bb) {
    const Tensor& memory = bb.memory;
    const uint32_t D = cfg_.hiddenDim, Nq = cfg_.numQueries, NC = cfg_.numClasses;
    const uint32_t K = cfg_.numKeypoints;
    const uint32_t grid = cfg_.grid(), totalTokens = grid * grid;

    // enc_output projection + first-stage class/bbox heads over all tokens.
    Tensor om = layerNorm_(linear_(memory, "transformer.enc_output.0.weight", "transformer.enc_output.0.bias"),
                           "transformer.enc_output_norm.0.weight", "transformer.enc_output_norm.0.bias", D);
    Tensor cls = linear_(om, "transformer.enc_out_class_embed.0.weight", "transformer.enc_out_class_embed.0.bias");
    Tensor bbx = bboxMLP_(om, "transformer.enc_out_bbox_embed.0");

    // Grid proposals -> reparam coords + each token's best class logit.
    Tensor encCoord = vk_.createTensorV({totalTokens, 4u});
    Tensor encScore = vk_.createTensorV({totalTokens});
    { ProposalParams pp{totalTokens, grid, grid, NC};
      vk_.dispatch(proposalsPipe_, {cls.buffer, bbx.buffer, encCoord.buffer, encScore.buffer},
                   &pp, sizeof(pp), divCeil(totalTokens, 64), 1, 1); }
    // The top-Nq tokens by that logit, best first (equal logits by token index).
    Tensor sel = vk_.createTensorV({Nq});// uint32 token indices
    { TopKParams tp{totalTokens, Nq, 0u, 0u};
      vk_.dispatch(topkRankPipe_, {encScore.buffer, sel.buffer}, &tp, sizeof(tp), divCeil(totalTokens, 256), 1, 1); }
    // Decoder reference (fixed across layers: lite_refpoint_refine) = the learned
    // refpoint_embed[:Nq] reparametrised on the selected proposals, and its sine
    // embedding, the input of ref_point_head.
    Tensor refT   = vk_.createTensorV({Nq, 4u});
    Tensor refXyT = vk_.createTensorV({Nq, 2u});
    Tensor qsineT = vk_.createTensorV({Nq, 512u});
    { QueryRefParams qrp{Nq, 0u, 0u, 0u};
      vk_.dispatch(queryRefsPipe_, {sel.buffer, encCoord.buffer, weights_.at("refpoint_embed.weight").buffer,
                                    dimtBuf_.buffer, refT.buffer, refXyT.buffer, qsineT.buffer},
                   &qrp, sizeof(qrp), divCeil(512u, 128u), Nq, 1); }
    vk_.markTimestamp("select");

    // query_pos + the decoder layers (refs fixed) + heads.
    Tensor tgtInit{weights_.at("query_feat.weight").buffer, {Nq, D}};// first Nq rows
    Tensor qp = linear_(qsineT, "transformer.decoder.ref_point_head.layers.0.weight",
                        "transformer.decoder.ref_point_head.layers.0.bias", /*act=ReLU*/ 2u);
    qp = linear_(qp, "transformer.decoder.ref_point_head.layers.1.weight",
                     "transformer.decoder.ref_point_head.layers.1.bias");

    // Keypoint variant: K keypoint queries per instance, initialised from the learned
    // queries alone (image-independent, so precomputed at export), and the per-
    // instance inputs of their cross-attention, fixed across layers like the reference.
    Tensor kp, qpK, refK, refXyK;
    if (K > 0) {
        kp = Tensor{weights_.at("derived.keypoint_queries").buffer, {Nq * K, D}};
        qpK = repeatRows_(qp, K);
        refK = repeatRows_(refT, K);
        refXyK = repeatRows_(refXyT, K);
    }

    Tensor tgt = tgtInit;
    for (int layer = 0; layer < cfg_.decLayers; ++layer) {
        std::string p = "transformer.decoder.layers." + std::to_string(layer);
        // self-attention (q = k = tgt + query_pos, v = tgt): the q/k and the v rows of
        // in_proj, each into its part of the fused qkv
        Tensor qkv = vk_.createTensorV({Nq, 3u * D});
        linearRowsInto_(add_(tgt, qp), p + ".self_attn.in_proj_weight", p + ".self_attn.in_proj_bias",
                        0u, 2u * D, qkv, 0u);
        linearRowsInto_(tgt, p + ".self_attn.in_proj_weight", p + ".self_attn.in_proj_bias", 2u * D, D, qkv, 2u * D);
        Tensor heads = attnCm32Pipe_.pipeline
                               ? attnHead32_(qkv, Nq, 1u, cfg_.saHeads)
                               : attnApply_(qkv, softmaxRows_(attnScores_(qkv, Nq, cfg_.saHeads), cfg_.saHeads * Nq, Nq),
                                            Nq, cfg_.saHeads);
        Tensor n1 = layerNorm_(linear_(heads, p + ".self_attn.out_proj.weight", p + ".self_attn.out_proj.bias", 0u, &tgt),
                               p + ".norm1.weight", p + ".norm1.bias", D);

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
          std::vector<VkBuffer> s = {refT.buffer, msShapeBuf_.buffer, rawOff.buffer};
          vk_.dispatch(offsetPreprocessPipe_, s, &opp, sizeof(opp), divCeil(Nq * stride, 64), 1, 1); }
        Tensor crossOut  = msDeformAttn_(value, grid, grid, refXyT, rawOff, awSm, cfg_.caHeads);
        Tensor n2 = layerNorm_(linear_(crossOut, p + ".cross_attn.output_proj.weight", p + ".cross_attn.output_proj.bias",
                                       0u, &n1),
                               p + ".norm2.weight", p + ".norm2.bias", D);

        // FFN (the residuals are the GEMMs' epilogues)
        Tensor ff = linear_(n2, p + ".linear1.weight", p + ".linear1.bias", /*act=ReLU*/ 2u);
        tgt = layerNorm_(linear_(ff, p + ".linear2.weight", p + ".linear2.bias", 0u, &n2),
                         p + ".norm3.weight", p + ".norm3.bias", D);

        if (K > 0) keypointLayer_(layer, tgt, kp, qpK, refK, refXyK, bb.kpMemory);
    }
    Tensor hs = layerNorm_(tgt, "transformer.decoder.norm.weight", "transformer.decoder.norm.bias", D);
    Tensor logits = linear_(hs, "class_embed.weight", "class_embed.bias");
    Tensor bboxDelta = bboxMLP_(hs, "bbox_embed");
    // The keypoint states get no final norm.
    Tensor kpOut = K > 0 ? bboxMLP_(kp, "keypoint_embed") : Tensor{};// [Nq*K, 8]
    vk_.markTimestamp("decoder");
    return HeadTensors{logits, bboxDelta, refT, kpOut, hs, kp};
}

VkDeviceSize RfDetrVk::headReadbackBytes_() const {
    const VkDeviceSize Nq = VkDeviceSize(cfg_.numQueries);
    const VkDeviceSize kpOut = Nq * VkDeviceSize(cfg_.numKeypoints) * 8;
    return (Nq * VkDeviceSize(cfg_.numClasses) + Nq * 4 + Nq * 4 + kpOut) * sizeof(float);
}

void RfDetrVk::recordBackbone_(const Tensor& input, BackboneTensors& keep) {
    keep.patch = patchEmbed_(input);          // [384, 24, 24]
    vk_.markTimestamp("patch");
    keep.embed = buildEmbeddings_(keep.patch);// [4, 145, 384] windowed tokens
    vk_.markTimestamp("embed");

    // Flatten the windowed tokens to [M, C] (window-major) for the layer loop.
    const uint32_t C = cfg_.embedDim;
    const uint32_t M = keep.embed.numel() / C;
    Tensor tok = keep.embed;
    tok.shape = {M, C};

    // 12 DINOv2 blocks; full attention at layers {3,6,9}. Keep layers 0 & 3
    // for validation and the tap outputs at layers {2,5,8,11} for the projector.
    static const char* kVitLbl[12] = {"vit0","vit1","vit2","vit3","vit4","vit5",
                                      "vit6","vit7","vit8","vit9","vit10","vit11"};
    std::array<Tensor, 4> tapHidden;
    const int tapLayers[4] = {2, 5, 8, 11};
    for (int i = 0; i < cfg_.numLayers; ++i) {
        bool global = (i == 3 || i == 6 || i == 9);
        tok = vitLayer_(tok, i, global);
        if (i == 0) keep.vit0 = tok;
        if (i == 3) keep.vit3 = tok;
        for (int t = 0; t < 4; ++t)
            if (i == tapLayers[t]) tapHidden[t] = tok;
        vk_.markTimestamp(kVitLbl[i]);
    }

    // Each tap: final ViT LayerNorm (per-token over C), then de-windowed into its
    // channels of the concatenated taps [4C, grid, grid] the projectors read.
    const uint32_t grid = cfg_.grid();
    keep.tapsCat = vk_.createTensorV({4u * C, grid, grid});
    for (uint32_t t = 0; t < 4; ++t) {
        Tensor ln = layerNorm_(tapHidden[t], "backbone.0.encoder.encoder.layernorm.weight",
                               "backbone.0.encoder.encoder.layernorm.bias", C);
        unwindowInto_(ln, keep.tapsCat, t * C);
    }
    vk_.markTimestamp("taps");
    keep.proj0 = projector_(keep.tapsCat, "backbone.0.projector");
    vk_.markTimestamp("proj");
    // proj0 [256,G,G] -> memory tokens [G*G,256]. It can stay in the arena: the
    // decoder is recorded into the same frame, before the arena is rewound.
    keep.memory = transposeChw2Tokens_(keep.proj0);
    if (cfg_.numKeypoints > 0) {
        // The keypoint cross-attention reads its own projection of the same taps.
        keep.projKp0 = projector_(keep.tapsCat, "backbone.0.cross_attn_projector");
        keep.kpMemory = transposeChw2Tokens_(keep.projKp0);
        vk_.markTimestamp("projkp");
    }
}

RfDetrVk::Recorded RfDetrVk::recordForward_(const Tensor& input) {
    Recorded r;
    recordBackbone_(input, r.bb);
    r.heads = recordDecoder_(r.bb);
    const HeadTensors& heads = r.heads;
    r.offLogits = vk_.recordReadback(heads.logits.buffer, heads.logits.bytes());
    r.offDelta  = vk_.recordReadback(heads.bboxDelta.buffer, heads.bboxDelta.bytes());
    r.offRef    = vk_.recordReadback(heads.ref.buffer, heads.ref.bytes());
    r.offKp     = heads.kpOut.valid() ? vk_.recordReadback(heads.kpOut.buffer, heads.kpOut.bytes()) : 0;
    return r;
}

RfDetrVk::ForwardOut RfDetrVk::readHeads_(const Recorded& r) const {
    const HeadTensors& heads = r.heads;
    const VkDeviceSize offLogits = r.offLogits, offDelta = r.offDelta, offRef = r.offRef, offKp = r.offKp;
    ForwardOut out;
    out.predLogits.resize(heads.logits.numel());
    std::memcpy(out.predLogits.data(), vk_.frameReadback(offLogits), heads.logits.bytes());

    // pred_boxes = reparam(bbox_embed(hs), ref) with the fixed decoder reference.
    const uint32_t Nq = cfg_.numQueries;
    const auto* delta = static_cast<const float*>(vk_.frameReadback(offDelta));
    const auto* ref = static_cast<const float*>(vk_.frameReadback(offRef));
    out.predBoxes.resize(size_t(Nq) * 4);
    for (uint32_t i = 0; i < Nq; ++i) {
        const float* d = delta + size_t(i) * 4;
        const float* r = ref + size_t(i) * 4;
        out.predBoxes[i * 4 + 0] = d[0] * r[2] + r[0];
        out.predBoxes[i * 4 + 1] = d[1] * r[3] + r[1];
        out.predBoxes[i * 4 + 2] = std::exp(d[2]) * r[2];
        out.predBoxes[i * 4 + 3] = std::exp(d[3]) * r[3];
    }

    if (heads.kpOut.valid()) {
        // pred_keypoints: xy = delta * ref_wh + ref_xy against the same reference, the
        // other six channels as the head gave them. Channel 7 summed over a query's
        // keypoints adds to the logit of the keypoints' class.
        const uint32_t K = cfg_.numKeypoints, NC = cfg_.numClasses;
        const auto* kpRaw = static_cast<const float*>(vk_.frameReadback(offKp));
        out.predKeypoints.assign(kpRaw, kpRaw + heads.kpOut.numel());
        for (uint32_t q = 0; q < Nq; ++q) {
            const float* r = ref + size_t(q) * 4;
            float boost = 0.f;
            for (uint32_t k = 0; k < K; ++k) {
                float* o = &out.predKeypoints[(size_t(q) * K + k) * 8];
                o[0] = o[0] * r[2] + r[0];
                o[1] = o[1] * r[3] + r[1];
                boost += o[7];
            }
            out.predLogits[size_t(q) * NC + uint32_t(kpClass_)] += boost;
        }
        out.decRef.assign(ref, ref + size_t(Nq) * 4);
    }
    return out;
}

RfDetrVk::ForwardOut RfDetrVk::finishForward_(const Tensor& input, bool captureIntermediates) {
    const Recorded r = recordForward_(input);
    vk_.endFrame();// the inference's one submit and wait
    ForwardOut out = readHeads_(r);
    const BackboneTensors& bb = r.bb;
    const HeadTensors& heads = r.heads;
    if (captureIntermediates) {// validation only — the detect path skips these readbacks
        auto grab = [&](const Tensor& t, std::vector<float>& dst) {
            dst.resize(t.numel());
            vk_.readback(t.buffer, dst.data(), t.bytes());
        };
        grab(bb.patch, out.patchEmbed);
        out.patchDim = {bb.patch.C(), bb.patch.H(), bb.patch.W()};
        grab(bb.embed, out.embed);
        out.embedDim = {bb.embed.C(), bb.embed.H(), bb.embed.W()};
        grab(bb.vit0, out.vit0);
        grab(bb.vit3, out.vit3);
        out.vitDim = {bb.embed.C(), bb.embed.H(), bb.embed.W()};// [4,145,384]
        const uint32_t C = cfg_.embedDim, grid = cfg_.grid();
        const size_t tapFloats = size_t(C) * grid * grid;
        for (uint32_t t = 0; t < 4; ++t) {
            out.taps[t].resize(tapFloats);
            vk_.readback(bb.tapsCat.buffer, out.taps[t].data(), tapFloats * sizeof(float),
                         VkDeviceSize(t) * tapFloats * sizeof(float));
        }
        out.tapDim = {C, grid, grid};
        grab(bb.proj0, out.proj0);
        out.proj0Dim = {bb.proj0.C(), bb.proj0.H(), bb.proj0.W()};// [256,24,24]
        if (cfg_.numKeypoints > 0) {
            grab(bb.projKp0, out.projKp0);
            grab(heads.hs, out.decHs);
            grab(heads.kp, out.decKp);
        }
    }
    vk_.resetArena();
    return out;
}

RfDetrVk::ForwardOut RfDetrVk::runForward(const std::vector<float>& chw, bool captureIntermediates) {
    const uint32_t R = cfg_.resolution;
    const VkDeviceSize bytes = VkDeviceSize(chw.size()) * sizeof(float);
    if (chw.size() != size_t(3) * R * R)
        throw std::runtime_error("RfDetrVk::runForward: input must be 3*R*R floats, R the variant's resolution");
    vk_.reserveFrameIO(bytes, headReadbackBytes_());
    vk_.beginFrame();
    Tensor input = vk_.createTensorV({3u, R, R});
    vk_.recordUpload(input.buffer, chw.data(), bytes);
    return finishForward_(input, captureIntermediates);
}

// ImageNet preprocess (square resize → /255 → normalize) and the forward pass
// run on the GPU as one submit; then sigmoid/threshold postprocess on the host.
// classId is the COCO category id (1..90); boxes are in image pixels.
std::vector<Detection> RfDetrVk::infer(const unsigned char* rgba, int width, int height,
                                       float scoreThresh) {
    const uint32_t R = cfg_.resolution;
    const VkDeviceSize rgbaBytes = VkDeviceSize(width) * VkDeviceSize(height) * 4;
    if (rgbaInput_.capacity < rgbaBytes) {
        // A new buffer handle: the descriptor sets cached for the old one go stale.
        rgbaInput_ = vk_.createOwnedRaw(rgbaBytes);
        vk_.resetDescriptorCache();
    }
    vk_.reserveFrameIO(rgbaBytes, headReadbackBytes_());
    // The frame is the same for every image of this size: record it once, then only
    // refill the image's staging and submit it again.
    const uint64_t key = (1ull << 63) | (uint64_t(uint32_t(width)) << 32) | uint64_t(uint32_t(height));
    if (vk_.replayable(key)) {
        std::memcpy(vk_.frameUploadData(inferUploadOff_), rgba, size_t(rgbaBytes));
        vk_.replayFrame();
    } else {
        vk_.beginFrame(key);
        inferUploadOff_ = vk_.recordUpload(rgbaInput_.buffer, rgba, rgbaBytes);
        Tensor input = vk_.createTensorV({3u, R, R});
        { PreprocessParams pp{uint32_t(width), uint32_t(height), R, 0u};
          vk_.dispatch(preprocessPipe_, {rgbaInput_.buffer, input.buffer}, &pp, sizeof(pp),
                       divCeil(R, 16), divCeil(R, 16), 1); }
        vk_.markTimestamp("preprocess");
        inferRec_ = recordForward_(input);
        vk_.endFrame();
    }
    const ForwardOut out = readHeads_(inferRec_);
    vk_.resetArena();
    return postprocess(out, width, height, scoreThresh);
}

std::vector<Detection> RfDetrVk::postprocess(const ForwardOut& fw, int width, int height, float scoreThresh) const {
    std::vector<Detection> dets;
    auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };
    auto detection = [&](uint32_t q, uint32_t c, float s) {
        const float* b = &fw.predBoxes[size_t(q) * 4];
        float cx = b[0], cy = b[1], w = b[2], h = b[3];
        return Detection{int(c), s,
                         (cx - w * 0.5f) * float(width), (cy - h * 0.5f) * float(height),
                         (cx + w * 0.5f) * float(width), (cy + h * 0.5f) * float(height), {}};
    };
    const uint32_t Nq = cfg_.numQueries, NC = cfg_.numClasses;
    if (cfg_.numKeypoints == 0) {
        for (uint32_t q = 0; q < Nq; ++q) {
            for (uint32_t c = 0; c < NC; ++c) {
                float s = sigmoid(fw.predLogits[size_t(q) * NC + c]);
                if (s < scoreThresh) continue;
                dets.push_back(detection(q, c, s));
            }
        }
    } else {
        // rfdetr's PostProcess for keypoint models (models/postprocess.py): the top
        // num_select (= the query count) (query, class) pairs by probability, best
        // first and ties by flat index; then every pair whose class has a schema
        // slot gets its score fused with the keypoints' localisation uncertainty,
        //   s' = min(sigmoid(log s - 0.2 T), nextafter(1, 0)),
        // T the findability-weighted log mean trace of the covariances the head's
        // precision-Cholesky factors imply (0 for a slot without keypoints).
        const uint32_t K = cfg_.numKeypoints, pairs = Nq * NC;
        std::vector<float> prob(pairs);
        for (uint32_t i = 0; i < pairs; ++i) prob[i] = sigmoid(fw.predLogits[i]);
        std::vector<uint32_t> order(pairs);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return prob[a] > prob[b]; });

        auto logSumExp = [](const float* v, uint32_t n) {
            float m = -std::numeric_limits<float>::infinity();
            for (uint32_t i = 0; i < n; ++i) m = std::max(m, v[i]);
            if (!std::isfinite(m)) return m;
            float s = 0.f;
            for (uint32_t i = 0; i < n; ++i) s += std::exp(v[i] - m);
            return m + std::log(s);
        };
        std::vector<float> weightedTrace(K), logW(K);
        auto logMeanTrace = [&](uint32_t q) {
            for (uint32_t k = 0; k < K; ++k) {
                const float* o = &fw.predKeypoints[(size_t(q) * K + k) * 8];
                const float logT1 = -2.f * o[4], logT2 = -2.f * o[6];
                const float logT[3] = {logT1, logT2, 2.f * std::log(std::max(std::abs(o[5]), 1e-12f)) + logT1 + logT2};
                logW[k] = std::log(std::max(sigmoid(o[2]), 1e-12f));
                weightedTrace[k] = logSumExp(logT, 3) + logW[k];
            }
            return logSumExp(weightedTrace.data(), K) - logSumExp(logW.data(), K);
        };

        const float belowOne = std::nextafter(1.f, 0.f);
        for (uint32_t n = 0; n < std::min(Nq, pairs); ++n) {
            const uint32_t idx = order[n], q = idx / NC, c = idx % NC;
            float s = prob[idx];
            if (int(c) < kpSchemaClasses_) {
                const float T = int(c) == kpClass_ ? logMeanTrace(q) : 0.f;
                s = std::min(sigmoid(std::log(s) - 0.2f * T), belowOne);
            }
            if (s < scoreThresh) continue;
            Detection d = detection(q, c, s);
            if (int(c) == kpClass_) {
                d.keypoints.resize(K);
                for (uint32_t k = 0; k < K; ++k) {
                    const float* o = &fw.predKeypoints[(size_t(q) * K + k) * 8];
                    d.keypoints[k] = Keypoint{o[0] * float(width), o[1] * float(height), sigmoid(o[2])};
                }
            }
            dets.push_back(std::move(d));
        }
    }
    std::stable_sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    return dets;
}

}// namespace rfdetr
