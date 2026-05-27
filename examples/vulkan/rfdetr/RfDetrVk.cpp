#include "RfDetrVk.hpp"

#include "WeightLoader.hpp"

#include <algorithm>
#include <cmath>
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

using threepp::VulkanRenderer;

namespace rfdetr {

namespace {
    struct ConvParams {
        uint32_t in_c, out_c, in_h, in_w;
        uint32_t out_h, out_w, k_h, k_w;
        uint32_t stride_h, stride_w, pad_h, pad_w;
        uint32_t has_bias, activation, _p0, _p1;
    };
    struct LinearParams    { uint32_t M, N, K, hasBias; };
    struct LNParams        { uint32_t M, D; float eps, _pad; };
    struct SoftmaxParams   { uint32_t M, N, _p0, _p1; };
    struct GeluParams      { uint32_t n, _p0, _p1, _p2; };
    struct LayerScaleParams{ uint32_t M, C, _p0, _p1; };
    struct AttnParams      { uint32_t M, H, d, stride; float scale; uint32_t _p0, _p1, _p2; };
    struct AddParams       { uint32_t n, _p0, _p1, _p2; };
    struct ConcatParams    { uint32_t c_a, c_b, h, w; };

    uint32_t divCeil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
}// namespace

RfDetrVk::RfDetrVk(VulkanRenderer& r)
    : vk_(static_cast<VkDevice>(r.nativeDevice()),
          static_cast<VkPhysicalDevice>(r.nativePhysicalDevice()),
          static_cast<VkQueue>(r.nativeGraphicsQueue()),
          r.graphicsQueueFamily()) {

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
}

void RfDetrVk::loadWeights(const std::string& path) {
    auto w = parseWeightBinary(path);
    weights_.clear();
    for (auto& [name, data] : w.data) {
        const auto& sh = w.shapes.at(name);
        VkTensor t = vk_.createOwned(sh.empty() ? std::vector<uint32_t>{uint32_t(data.size())} : sh);
        vk_.upload(t.buffer, data.data(), data.size() * sizeof(float));
        weights_.emplace(name, std::move(t));
    }
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
                 PATCH, PATCH, 0, 0);
}

RfDetrVk::ForwardOut RfDetrVk::runForward(const std::vector<float>& chw) {
    const uint32_t R = RESOLUTION;
    if (chw.size() != size_t(3) * R * R)
        throw std::runtime_error("RfDetrVk::runForward: input must be 3*384*384 floats");

    vk_.beginFrame();
    Tensor input = vk_.createTensorV({3u, R, R});
    vk_.upload(input.buffer, chw.data(), chw.size() * sizeof(float));
    auto pe = patchEmbed_(input);// [384, 24, 24]
    vk_.endFrame();

    ForwardOut out;
    out.patchEmbed.resize(pe.numel());
    vk_.readback(pe.buffer, out.patchEmbed.data(), pe.bytes());
    out.patchDim = {pe.C(), pe.H(), pe.W()};
    vk_.resetArena();
    return out;
}

}// namespace rfdetr
