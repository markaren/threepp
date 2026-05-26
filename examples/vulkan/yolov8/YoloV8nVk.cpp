#include "YoloV8nVk.hpp"

#include "WeightLoader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

// SPIR-V blobs generated from the .comp sources by compile_vulkan_shader.
#include "threepp/renderers/vulkan/shaders/conv2d.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/conv1x1.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/conv3x3s1.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/conv3x3s2.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/preprocess.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/bn_silu.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/add.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/maxpool.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/upsample.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/concat.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/slice.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/detect.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/detect_fused.comp.spv.h"

using threepp::VulkanRenderer;

namespace yolo {

namespace {

    // Push-constant param structs — byte layout matches the GLSL push_constant
    // blocks (all 4-byte scalars, tightly packed under std430).
    struct ConvParams {
        uint32_t in_c, out_c, in_h, in_w;
        uint32_t out_h, out_w, k_h, k_w;
        uint32_t stride_h, stride_w, pad_h, pad_w;
        uint32_t has_bias, has_silu, has_residual, _p2;
    };
    struct BnParams { uint32_t channels, height, width; float eps; };
    struct AddParams { uint32_t count, _p0, _p1, _p2; };
    struct PoolParams { uint32_t channels, in_h, in_w, out_h, out_w, k, stride, pad; };
    struct UpParams { uint32_t channels, in_h, in_w, _p; };
    struct CatParams { uint32_t c_a, c_b, height, width; };
    struct SliceParams { uint32_t c_in, c_out, offset_c, height, width, dst_offset_c, _p1, _p2; };
    struct DetectParams {
        uint32_t grid_h, grid_w, reg_max, num_classes;
        float stride, conf_thresh, _p1, _p2;
        uint32_t in_c_box, in_c_cls, max_dets, _p4;
    };
    struct PrepParams {
        uint32_t src_w, src_h, dst_w, dst_h;
        float inv_scale, pad_x, pad_y;
        uint32_t _p2;
    };

    uint32_t divCeil(uint32_t n, uint32_t d) { return (n + d - 1) / d; }

    uint16_t f32_to_f16(float f) {
        uint32_t x;
        std::memcpy(&x, &f, 4);
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t  exp  = int32_t((x >> 23) & 0xFFu) - 127;
        uint32_t mant = x & 0x7FFFFFu;
        if (exp == 128) return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));
        if (exp > 15) return uint16_t(sign | 0x7C00u);
        if (exp < -14) return uint16_t(sign);
        uint32_t he = uint32_t(exp + 15) & 0x1Fu;
        uint32_t hm = mant >> 13;
        return uint16_t(sign | (he << 10) | hm);
    }

}// namespace

// ============================================================
//  Construction
// ============================================================
YoloV8nVk::YoloV8nVk(VulkanRenderer& r)
    : vk_(static_cast<VkDevice>(r.nativeDevice()),
          static_cast<VkPhysicalDevice>(r.nativePhysicalDevice()),
          static_cast<VkQueue>(r.nativeGraphicsQueue()),
          r.graphicsQueueFamily()) {

    convPipe_        = vk_.createPipe(kConv2dSpv,      sizeof(kConv2dSpv),      4, sizeof(ConvParams));
    conv3x3s1Pipe_   = vk_.createPipe(kConv3x3s1Spv,   sizeof(kConv3x3s1Spv),   5, sizeof(ConvParams));
    conv3x3s2Pipe_   = vk_.createPipe(kConv3x3s2Spv,   sizeof(kConv3x3s2Spv),   4, sizeof(ConvParams));
    conv1x1Pipe_     = vk_.createPipe(kConv1x1Spv,     sizeof(kConv1x1Spv),     4, sizeof(ConvParams));
    bnSiluPipe_      = vk_.createPipe(kBnSiluSpv,      sizeof(kBnSiluSpv),      6, sizeof(BnParams));
    addPipe_         = vk_.createPipe(kAddSpv,         sizeof(kAddSpv),         3, sizeof(AddParams));
    maxpoolPipe_     = vk_.createPipe(kMaxpoolSpv,     sizeof(kMaxpoolSpv),     2, sizeof(PoolParams));
    upsamplePipe_    = vk_.createPipe(kUpsampleSpv,    sizeof(kUpsampleSpv),    2, sizeof(UpParams));
    concatPipe_      = vk_.createPipe(kConcatSpv,      sizeof(kConcatSpv),      3, sizeof(CatParams));
    slicePipe_       = vk_.createPipe(kSliceSpv,       sizeof(kSliceSpv),       2, sizeof(SliceParams));
    detectPipe_      = vk_.createPipe(kDetectSpv,      sizeof(kDetectSpv),      4, sizeof(DetectParams));
    detectFusedPipe_ = vk_.createPipe(kDetectFusedSpv, sizeof(kDetectFusedSpv), 7, sizeof(DetectParams));
    preprocessPipe_  = vk_.createPipe(kPreprocessSpv,  sizeof(kPreprocessSpv),  2, sizeof(PrepParams));

    detCounter_ = vk_.createOwnedRaw(sizeof(uint32_t));
    detDets_    = vk_.createOwnedRaw(MAX_DETS * 6 * sizeof(float));
}

YoloV8nVk::~YoloV8nVk() {
    vk_.destroyPipe(convPipe_);
    vk_.destroyPipe(conv3x3s1Pipe_);
    vk_.destroyPipe(conv3x3s2Pipe_);
    vk_.destroyPipe(conv1x1Pipe_);
    vk_.destroyPipe(bnSiluPipe_);
    vk_.destroyPipe(addPipe_);
    vk_.destroyPipe(maxpoolPipe_);
    vk_.destroyPipe(upsamplePipe_);
    vk_.destroyPipe(concatPipe_);
    vk_.destroyPipe(slicePipe_);
    vk_.destroyPipe(detectPipe_);
    vk_.destroyPipe(detectFusedPipe_);
    vk_.destroyPipe(preprocessPipe_);
}

// ============================================================
//  Weight loading (offline BN-fold + f16 packing — identical to WGPU)
// ============================================================
void YoloV8nVk::loadWeights(const std::string& path) {
    auto w = parseWeightBinary(path);

    const float eps = 1e-5f;
    std::vector<std::string> convKeys;
    for (auto& kv : w.data)
        if (kv.first.size() > 12 && kv.first.compare(kv.first.size() - 12, 12, ".conv.weight") == 0)
            convKeys.push_back(kv.first);

    for (auto& convKey : convKeys) {
        std::string prefix = convKey.substr(0, convKey.size() - 12);
        std::string bnW = prefix + ".bn.weight", bnB = prefix + ".bn.bias";
        std::string bnM = prefix + ".bn.running_mean", bnV = prefix + ".bn.running_var";
        if (!w.data.count(bnW) || !w.data.count(bnB) || !w.data.count(bnM) || !w.data.count(bnV)) continue;

        const auto& wt    = w.data.at(convKey);
        const auto& gamma = w.data.at(bnW);
        const auto& beta  = w.data.at(bnB);
        const auto& mean  = w.data.at(bnM);
        const auto& var_  = w.data.at(bnV);
        const auto& sh    = w.shapes.at(convKey);
        uint32_t oc = sh[0];
        uint32_t per = wt.size() / oc;

        std::vector<float> fw(wt.size());
        std::vector<float> fb(oc);
        for (uint32_t c = 0; c < oc; ++c) {
            float s = gamma[c] / std::sqrt(var_[c] + eps);
            fb[c] = beta[c] - mean[c] * s;
            for (uint32_t i = 0; i < per; ++i) fw[c * per + i] = wt[c * per + i] * s;
        }
        w.data[prefix + ".fused.weight"]   = std::move(fw);
        w.shapes[prefix + ".fused.weight"] = sh;
        w.data[prefix + ".fused.bias"]     = std::move(fb);
        w.shapes[prefix + ".fused.bias"]   = {oc};
    }

    weights_.clear();
    for (auto& [name, data] : w.data) {
        auto& sh = w.shapes.at(name);
        bool isConvWeight = (sh.size() == 4 && name.size() >= 7 &&
                             name.compare(name.size() - 7, 7, ".weight") == 0);

        if (isConvWeight && !data.empty() && (data.size() % 2) == 0) {
            size_t n = data.size();
            std::vector<uint32_t> packed(n / 2);
            for (size_t i = 0; i < n; i += 2) {
                uint16_t lo = f32_to_f16(data[i]);
                uint16_t hi = f32_to_f16(data[i + 1]);
                packed[i / 2] = uint32_t(lo) | (uint32_t(hi) << 16);
            }
            VkTensor t = vk_.createOwnedRaw(packed.size() * sizeof(uint32_t));
            t.shape = sh;
            vk_.upload(t.buffer, packed.data(), packed.size() * sizeof(uint32_t));
            weights_.emplace(name, std::move(t));
        } else {
            VkTensor t = vk_.createOwned(sh);
            vk_.upload(t.buffer, data.data(), data.size() * sizeof(float));
            weights_.emplace(name, std::move(t));
        }
    }
}

// ============================================================
//  Low-level GPU ops
// ============================================================
Tensor YoloV8nVk::conv_(const Tensor& x, const std::string& weightKey, const std::string& biasKey,
                        int strideH, int strideW, bool silu, const Tensor* residual) {
    auto& wt = weights_.at(weightKey);
    uint32_t out_c = wt.shape[0], in_c = wt.shape[1], k_h = wt.shape[2], k_w = wt.shape[3];
    uint32_t in_h = x.H(), in_w = x.W();
    uint32_t pad_h = (k_h == 1) ? 0u : k_h / 2;
    uint32_t pad_w = (k_w == 1) ? 0u : k_w / 2;
    uint32_t out_h = (in_h + 2 * pad_h - k_h) / uint32_t(strideH) + 1;
    uint32_t out_w = (in_w + 2 * pad_w - k_w) / uint32_t(strideW) + 1;

    bool hasBias = !biasKey.empty() && weights_.count(biasKey);
    ConvParams cp{in_c, out_c, in_h, in_w, out_h, out_w, k_h, k_w,
                  uint32_t(strideH), uint32_t(strideW), pad_h, pad_w,
                  hasBias ? 1u : 0u, silu ? 1u : 0u, residual ? 1u : 0u, 0u};

    Tensor out = vk_.createTensor({out_c, out_h, out_w});
    VkBuffer biasBuf = hasBias ? weights_.at(biasKey).buffer : vk_.dummy();

    const VkPipe* pipe;
    uint32_t dx, dy, dz;
    if (k_h == 3 && k_w == 3 && strideH == 1 && strideW == 1) {
        pipe = &conv3x3s1Pipe_;
        dx = divCeil(out_w, 16); dy = divCeil(out_h, 16); dz = divCeil(out_c, 4);
    } else if (k_h == 3 && k_w == 3 && strideH == 2 && strideW == 2) {
        pipe = &conv3x3s2Pipe_;
        dx = divCeil(out_w, 8); dy = divCeil(out_h, 8); dz = divCeil(out_c, 4);
    } else if (k_h == 1 && k_w == 1 && strideH == 1 && strideW == 1) {
        pipe = &conv1x1Pipe_;
        dx = divCeil(out_w, 8); dy = divCeil(out_h, 8); dz = divCeil(out_c, 4);
    } else {
        pipe = &convPipe_;
        dx = divCeil(out_w, 8); dy = divCeil(out_h, 8); dz = divCeil(out_c, 4);
    }

    std::vector<VkBuffer> ssbos = {x.buffer, wt.buffer, biasBuf, out.buffer};
    if (pipe == &conv3x3s1Pipe_)
        ssbos.push_back(residual ? residual->buffer : vk_.dummy());
    vk_.dispatch(*pipe, ssbos, &cp, sizeof(cp), dx, dy, dz);
    return out;
}

Tensor YoloV8nVk::bnSilu_(const Tensor& x, const std::string& bnPrefix) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    BnParams bp{C, H, W, 1e-5f};
    Tensor out = vk_.createTensor({C, H, W});
    std::vector<VkBuffer> ssbos = {
        x.buffer,
        weights_.at(bnPrefix + ".running_mean").buffer,
        weights_.at(bnPrefix + ".running_var").buffer,
        weights_.at(bnPrefix + ".weight").buffer,
        weights_.at(bnPrefix + ".bias").buffer,
        out.buffer};
    vk_.dispatch(bnSiluPipe_, ssbos, &bp, sizeof(bp), divCeil(W, 8), divCeil(H, 8), divCeil(C, 4));
    return out;
}

Tensor YoloV8nVk::add_(const Tensor& a, const Tensor& b) {
    uint32_t n = a.numel();
    AddParams ap{n, 0, 0, 0};
    Tensor out = vk_.createTensorV(a.shape);
    std::vector<VkBuffer> ssbos = {a.buffer, b.buffer, out.buffer};
    vk_.dispatch(addPipe_, ssbos, &ap, sizeof(ap), divCeil(n, 256), 1, 1);
    return out;
}

Tensor YoloV8nVk::maxpool_(const Tensor& x, int k, int stride, int pad) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    uint32_t outH = (H + 2 * uint32_t(pad) - uint32_t(k)) / uint32_t(stride) + 1;
    uint32_t outW = (W + 2 * uint32_t(pad) - uint32_t(k)) / uint32_t(stride) + 1;
    PoolParams pp{C, H, W, outH, outW, uint32_t(k), uint32_t(stride), uint32_t(pad)};
    Tensor out = vk_.createTensor({C, outH, outW});
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(maxpoolPipe_, ssbos, &pp, sizeof(pp), divCeil(outW, 8), divCeil(outH, 8), divCeil(C, 4));
    return out;
}

Tensor YoloV8nVk::upsample2x_(const Tensor& x) {
    uint32_t C = x.C(), H = x.H(), W = x.W();
    UpParams up{C, H, W, 0};
    Tensor out = vk_.createTensor({C, H * 2, W * 2});
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(upsamplePipe_, ssbos, &up, sizeof(up), divCeil(W * 2, 8), divCeil(H * 2, 8), divCeil(C, 4));
    return out;
}

Tensor YoloV8nVk::concat_(const Tensor& a, const Tensor& b) {
    uint32_t ca = a.C(), cb = b.C(), H = a.H(), W = a.W();
    CatParams cp{ca, cb, H, W};
    Tensor out = vk_.createTensor({ca + cb, H, W});
    std::vector<VkBuffer> ssbos = {a.buffer, b.buffer, out.buffer};
    vk_.dispatch(concatPipe_, ssbos, &cp, sizeof(cp), divCeil(W, 8), divCeil(H, 8), divCeil(ca + cb, 4));
    return out;
}

Tensor YoloV8nVk::sliceChannels_(const Tensor& x, uint32_t offsetC, uint32_t countC) {
    uint32_t H = x.H(), W = x.W();
    SliceParams sp{x.C(), countC, offsetC, H, W, 0u, 0u, 0u};
    Tensor out = vk_.createTensor({countC, H, W});
    std::vector<VkBuffer> ssbos = {x.buffer, out.buffer};
    vk_.dispatch(slicePipe_, ssbos, &sp, sizeof(sp), divCeil(W, 8), divCeil(H, 8), divCeil(countC, 4));
    return out;
}

// ============================================================
//  Compound blocks
// ============================================================
Tensor YoloV8nVk::convBnSilu_(const Tensor& x, const std::string& prefix, int stride, const Tensor* residual) {
    return conv_(x, prefix + ".fused.weight", prefix + ".fused.bias", stride, stride, true, residual);
}

Tensor YoloV8nVk::plainConv_(const Tensor& x, const std::string& prefix) {
    return conv_(x, prefix + ".weight", prefix + ".bias", 1, 1);
}

Tensor YoloV8nVk::concatMany_(std::vector<Tensor>& parts) {
    uint32_t H = parts[0].H(), W = parts[0].W();
    uint32_t totalC = 0;
    for (auto& p : parts) totalC += p.C();
    Tensor out = vk_.createTensor({totalC, H, W});

    uint32_t dstOffset = 0;
    for (auto& p : parts) {
        SliceParams sp{p.C(), p.C(), 0u, H, W, dstOffset, 0u, 0u};
        std::vector<VkBuffer> ssbos = {p.buffer, out.buffer};
        vk_.dispatch(slicePipe_, ssbos, &sp, sizeof(sp), divCeil(W, 8), divCeil(H, 8), divCeil(p.C(), 4));
        dstOffset += p.C();
    }
    return out;
}

Tensor YoloV8nVk::concatViews_(const std::vector<SourceView>& views, uint32_t H, uint32_t W) {
    uint32_t totalC = 0;
    for (auto& v : views) totalC += v.countC;
    Tensor out = vk_.createTensor({totalC, H, W});

    uint32_t dstOffset = 0;
    for (auto& v : views) {
        SliceParams sp{v.srcTotalC, v.countC, v.srcOffsetC, H, W, dstOffset, 0u, 0u};
        std::vector<VkBuffer> ssbos = {v.buf, out.buffer};
        vk_.dispatch(slicePipe_, ssbos, &sp, sizeof(sp), divCeil(W, 8), divCeil(H, 8), divCeil(v.countC, 4));
        dstOffset += v.countC;
    }
    return out;
}

Tensor YoloV8nVk::c2f_(const Tensor& x, const std::string& prefix, int n, bool shortcut) {
    Tensor cv1out = convBnSilu_(x, prefix + ".cv1");
    uint32_t cmid = cv1out.C() / 2;
    uint32_t H = cv1out.H(), W = cv1out.W();

    std::vector<Tensor> bottleneckOuts;
    bottleneckOuts.reserve(size_t(n));
    Tensor cur = sliceChannels_(cv1out, cmid, cmid);// second half feeds the bottleneck chain

    for (int i = 0; i < n; ++i) {
        std::string mi = prefix + ".m." + std::to_string(i);
        Tensor y = convBnSilu_(cur, mi + ".cv1");
        const Tensor* res = shortcut ? &cur : nullptr;
        y = convBnSilu_(y, mi + ".cv2", 1, res);
        bottleneckOuts.push_back(y);
        cur = y;
    }

    std::vector<SourceView> views;
    views.reserve(2 + bottleneckOuts.size());
    views.push_back({cv1out.buffer, cv1out.C(), 0u, cmid});
    views.push_back({cv1out.buffer, cv1out.C(), cmid, cmid});
    for (auto& bo : bottleneckOuts) views.push_back({bo.buffer, bo.C(), 0u, bo.C()});

    Tensor combined = concatViews_(views, H, W);
    return convBnSilu_(combined, prefix + ".cv2");
}

Tensor YoloV8nVk::sppf_(const Tensor& x, const std::string& prefix) {
    Tensor y  = convBnSilu_(x, prefix + ".cv1");
    Tensor y1 = maxpool_(y, 5, 1, 2);
    Tensor y2 = maxpool_(y1, 5, 1, 2);
    Tensor y3 = maxpool_(y2, 5, 1, 2);
    std::vector<Tensor> parts{y, y1, y2, y3};
    Tensor combined = concatMany_(parts);
    return convBnSilu_(combined, prefix + ".cv2");
}

// ============================================================
//  Detect head + NMS
// ============================================================
void YoloV8nVk::decodeScaleGpu_(const Tensor& feat, int scaleIdx, float stride, float confThresh) {
    std::string idx = std::to_string(scaleIdx);
    std::string cv2pfx = "model.22.cv2." + idx;
    std::string cv3pfx = "model.22.cv3." + idx;

    Tensor bx = convBnSilu_(feat, cv2pfx + ".0");
    bx = convBnSilu_(bx, cv2pfx + ".1");
    bx = plainConv_(bx, cv2pfx + ".2");

    Tensor cx = convBnSilu_(feat, cv3pfx + ".0");
    cx = convBnSilu_(cx, cv3pfx + ".1");
    cx = plainConv_(cx, cv3pfx + ".2");

    uint32_t gH = feat.H(), gW = feat.W();
    DetectParams dp{gH, gW, uint32_t(REG_MAX), uint32_t(NUM_CLASSES),
                    stride, confThresh, 0.f, 0.f, 0u, 0u, MAX_DETS, 0u};
    std::vector<VkBuffer> ssbos = {bx.buffer, cx.buffer, detCounter_.buffer, detDets_.buffer};
    vk_.dispatch(detectPipe_, ssbos, &dp, sizeof(dp), divCeil(gW, 8), divCeil(gH, 8), 1);
}

std::vector<Detection> YoloV8nVk::nms_(std::vector<Detection>& dets, float iouThresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> out;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        out.push_back(dets[i]);
        const auto& di = dets[i];
        float area_i = (di.x2 - di.x1) * (di.y2 - di.y1);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            const auto& dj = dets[j];
            float ix1 = std::max(di.x1, dj.x1), iy1 = std::max(di.y1, dj.y1);
            float ix2 = std::min(di.x2, dj.x2), iy2 = std::min(di.y2, dj.y2);
            float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
            float inter = iw * ih;
            float area_j = (dj.x2 - dj.x1) * (dj.y2 - dj.y1);
            float iou = inter / (area_i + area_j - inter + 1e-6f);
            if (iou > iouThresh) suppressed[j] = true;
        }
    }
    return out;
}

// ============================================================
//  Preprocessing
// ============================================================
Tensor YoloV8nVk::preprocess_(const unsigned char* rgba, int srcW, int srcH) {
    const int dstW = INPUT_SIZE, dstH = INPUT_SIZE;

    size_t srcBytes = size_t(srcW) * size_t(srcH) * 4;
    Tensor srcBuf = vk_.createRaw(srcBytes);
    vk_.upload(srcBuf.buffer, rgba, srcBytes);

    float scale = std::min(float(dstW) / float(srcW), float(dstH) / float(srcH));
    float newW = float(srcW) * scale, newH = float(srcH) * scale;
    float padX = (float(dstW) - newW) * 0.5f, padY = (float(dstH) - newH) * 0.5f;
    lbScale_ = scale; lbPadX_ = padX; lbPadY_ = padY;

    PrepParams pp{uint32_t(srcW), uint32_t(srcH), uint32_t(dstW), uint32_t(dstH),
                  1.0f / scale, padX, padY, 0u};
    Tensor out = vk_.createTensor({3u, uint32_t(dstH), uint32_t(dstW)});
    std::vector<VkBuffer> ssbos = {srcBuf.buffer, out.buffer};
    vk_.dispatch(preprocessPipe_, ssbos, &pp, sizeof(pp), divCeil(uint32_t(dstW), 16), divCeil(uint32_t(dstH), 16), 1);
    return out;
}

// ============================================================
//  Main inference
// ============================================================
std::vector<Detection> YoloV8nVk::infer(const unsigned char* rgba, int width, int height,
                                        float confThresh, float iouThresh) {
    vk_.zero(detCounter_.buffer);// reset the shared atomic counter for this inference
    vk_.beginFrame();

    // ---- Backbone ----
    Tensor x = preprocess_(rgba, width, height);// [3,640,640]
    x = convBnSilu_(x, "model.0", 2);           // [16,320,320]
    x = convBnSilu_(x, "model.1", 2);           // [32,160,160]
    x = c2f_(x, "model.2", 1, true);
    x = convBnSilu_(x, "model.3", 2);           // [64,80,80]
    Tensor feat3 = c2f_(x, "model.4", 2, true);
    x = convBnSilu_(feat3, "model.5", 2);       // [128,40,40]
    Tensor feat4 = c2f_(x, "model.6", 2, true);
    x = convBnSilu_(feat4, "model.7", 2);       // [256,20,20]
    x = c2f_(x, "model.8", 1, true);
    Tensor feat5 = sppf_(x, "model.9");         // [256,20,20]

    // ---- Neck (PAN-FPN) — shortcut=false for all head C2f ----
    x = upsample2x_(feat5);
    x = concat_(x, feat4);
    Tensor n4 = c2f_(x, "model.12", 1, false);

    x = upsample2x_(n4);
    x = concat_(x, feat3);
    Tensor h3 = c2f_(x, "model.15", 1, false);

    x = convBnSilu_(h3, "model.16", 2);
    x = concat_(x, n4);
    Tensor h4 = c2f_(x, "model.18", 1, false);

    x = convBnSilu_(h4, "model.19", 2);
    x = concat_(x, feat5);
    Tensor h5 = c2f_(x, "model.21", 1, false);

    // ---- Detect head (model.22): GPU pre-filter ----
    const Tensor* scales[3] = {&h3, &h4, &h5};
    float strides[3] = {8.f, 16.f, 32.f};
    for (int i = 0; i < 3; ++i)
        decodeScaleGpu_(*scales[i], i, strides[i], confThresh);

    vk_.endFrame();// single submit + wait for the whole forward pass

    // ---- Readback + CPU NMS ----
    uint32_t count = 0;
    vk_.readback(detCounter_.buffer, &count, sizeof(uint32_t));
    if (count > MAX_DETS) count = MAX_DETS;

    std::vector<float> raw(size_t(count) * 6, 0.f);
    if (count > 0) vk_.readback(detDets_.buffer, raw.data(), size_t(count) * 6 * sizeof(float));

    std::vector<Detection> all;
    all.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Detection d;
        d.x1 = raw[i * 6 + 0]; d.y1 = raw[i * 6 + 1];
        d.x2 = raw[i * 6 + 2]; d.y2 = raw[i * 6 + 3];
        d.conf = raw[i * 6 + 4];
        d.cls_id = static_cast<int>(raw[i * 6 + 5]);
        if (d.x2 > d.x1 && d.y2 > d.y1) all.push_back(d);
    }
    auto out = nms_(all, iouThresh);

    const float invS = 1.0f / lbScale_;
    for (auto& d : out) {
        d.x1 = (d.x1 - lbPadX_) * invS;
        d.y1 = (d.y1 - lbPadY_) * invS;
        d.x2 = (d.x2 - lbPadX_) * invS;
        d.y2 = (d.y2 - lbPadY_) * invS;
    }

    vk_.resetArena();// free this inference's intermediate activations
    return out;
}

}// namespace yolo
