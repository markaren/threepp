#include "threepp/renderers/vulkan/PostComposite.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/post_composite.comp.spv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace threepp::vulkan {

    namespace {

        // float32 → float16 bits (round-to-nearest-even, denormal-flushing —
        // ample for a [0,1] display-referred LUT).
        uint16_t floatToHalf(float f) {
            uint32_t x;
            std::memcpy(&x, &f, sizeof(x));
            const uint32_t sign = (x >> 16) & 0x8000u;
            int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
            uint32_t mant = x & 0x7FFFFFu;
            if (exp <= 0) return static_cast<uint16_t>(sign);          // underflow → ±0
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);// overflow → ±max
            // Round mantissa 23 → 10 bits to nearest even.
            uint32_t m = mant >> 13;
            const uint32_t rem = mant & 0x1FFFu;
            if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
                ++m;
                if (m == 0x400u) { m = 0; ++exp; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFu); }
            }
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | m);
        }

        // Row-major 3×3 helpers for the white-balance matrix build.
        struct Mat3 { float m[9]; };
        Mat3 mul(const Mat3& a, const Mat3& b) {
            Mat3 r{};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    r.m[i * 3 + j] = a.m[i * 3 + 0] * b.m[0 * 3 + j] +
                                     a.m[i * 3 + 1] * b.m[1 * 3 + j] +
                                     a.m[i * 3 + 2] * b.m[2 * 3 + j];
            return r;
        }

        // CCT (Kelvin) → CIE 1931 xy on the Planckian locus (Kim et al. cubic
        // fit, 1667–25000 K). `tint` offsets y: negative → green, positive →
        // magenta, ±1 ≈ ±0.05 in y (UE-comparable scale).
        void cctToXy(float kelvin, float tint, float& x, float& y) {
            const double T  = std::clamp(kelvin, 1667.f, 25000.f);
            const double T2 = T * T, T3 = T2 * T;
            double xd;
            if (T <= 4000.0)
                xd = -0.2661239e9 / T3 - 0.2343589e6 / T2 + 0.8776956e3 / T + 0.179910;
            else
                xd = -3.0258469e9 / T3 + 2.1070379e6 / T2 + 0.2226347e3 / T + 0.240390;
            const double x2 = xd * xd, x3 = x2 * xd;
            double yd;
            if (T <= 2222.0)
                yd = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * xd - 0.20219683;
            else if (T <= 4000.0)
                yd = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * xd - 0.16748867;
            else
                yd = 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * xd - 0.37001483;
            x = static_cast<float>(xd);
            y = static_cast<float>(yd) - tint * 0.05f;
        }

        // Bradford chromatic adaptation from the scene illuminant (xy) to D65,
        // wrapped into linear sRGB: M = sRGB←XYZ · CAT · XYZ←sRGB.
        Mat3 bradfordToD65(float srcX, float srcY) {
            const Mat3 bradford = {{ 0.8951f,  0.2664f, -0.1614f,
                                    -0.7502f,  1.7135f,  0.0367f,
                                     0.0389f, -0.0685f,  1.0296f}};
            const Mat3 bradfordInv = {{ 0.9869929f, -0.1470543f, 0.1599627f,
                                        0.4323053f,  0.5183603f, 0.0492912f,
                                       -0.0085287f,  0.0400428f, 0.9684867f}};
            const Mat3 srgbToXyz = {{0.4124564f, 0.3575761f, 0.1804375f,
                                     0.2126729f, 0.7151522f, 0.0721750f,
                                     0.0193339f, 0.1191920f, 0.9503041f}};
            const Mat3 xyzToSrgb = {{ 3.2404542f, -1.5371385f, -0.4985314f,
                                     -0.9692660f,  1.8760108f,  0.0415560f,
                                      0.0556434f, -0.2040259f,  1.0572252f}};

            auto whiteXyz = [](float x, float y, float* out) {
                out[0] = x / y;
                out[1] = 1.f;
                out[2] = (1.f - x - y) / y;
            };
            float src[3], dst[3];
            whiteXyz(srcX, srcY, src);
            whiteXyz(0.3127f, 0.3290f, dst);// D65

            auto cone = [&](const float* w, float* lms) {
                for (int i = 0; i < 3; ++i)
                    lms[i] = bradford.m[i * 3 + 0] * w[0] +
                             bradford.m[i * 3 + 1] * w[1] +
                             bradford.m[i * 3 + 2] * w[2];
            };
            float lmsSrc[3], lmsDst[3];
            cone(src, lmsSrc);
            cone(dst, lmsDst);

            Mat3 scale{};
            for (int i = 0; i < 3; ++i)
                scale.m[i * 3 + i] = lmsDst[i] / lmsSrc[i];

            const Mat3 cat = mul(bradfordInv, mul(scale, bradford));
            return mul(xyzToSrgb, mul(cat, srgbToXyz));
        }

    }// namespace

    PostComposite::PostComposite(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight) {
        hdrOut_.resize(framesInFlight_);
        createLutImage();
        createPipeline();
        createDescriptorPool();
    }

    PostComposite::~PostComposite() {
        VkDevice d = ctx_.device();
        if (pipe_)           vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_)     vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)        vkDestroySampler(d, sampler_, nullptr);
        if (nearestSampler_) vkDestroySampler(d, nearestSampler_, nullptr);
        destroyImage2D(ctx_.allocator(), d, gradeLut_);
        for (auto& img : hdrOut_) destroyImage2D(ctx_.allocator(), d, img);
    }

    void PostComposite::createLutImage() {
        gradeLut_.width  = kLutSize;
        gradeLut_.height = kLutSize;
        gradeLut_.format = VK_FORMAT_R16G16B16A16_SFLOAT;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_3D;
        ici.format        = gradeLut_.format;
        ici.extent        = {kLutSize, kLutSize, kLutSize};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &gradeLut_.image, &gradeLut_.alloc, nullptr),
              "vmaCreateImage(postComposite.gradeLut)");

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = gradeLut_.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format   = gradeLut_.format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vkCreateImageView(ctx_.device(), &vci, nullptr, &gradeLut_.view),
              "vkCreateImageView(postComposite.gradeLut)");
        ctx_.setObjectName(gradeLut_.image, "postComposite.gradeLut");

        // The LUT is always bound but only sampled when lutActive_ — still,
        // the descriptor needs a valid layout from frame one. Transition to
        // GENERAL (content undefined until the first setColorGrade upload;
        // never read before then).
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(postComposite)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(postComposite)");
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = gradeLut_.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        check(vkEndCommandBuffer(cb), "end one-shot cb(postComposite)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(postComposite)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(postComposite)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void PostComposite::resizeHdrOutput(uint32_t width, uint32_t height) {
        if (width == hdrOutW_ && height == hdrOutH_ && hdrOut_[0].image != VK_NULL_HANDLE) return;
        VkDevice d = ctx_.device();
        for (auto& img : hdrOut_) destroyImage2D(ctx_.allocator(), d, img);
        hdrOutW_ = width;
        hdrOutH_ = height;

        for (auto& img : hdrOut_) {
            img.width  = width;
            img.height = height;
            img.format = VK_FORMAT_B8G8R8A8_UNORM;// matches the swapchain channel order

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = img.format;
            ici.extent        = {width, height, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &img.image, &img.alloc, nullptr),
                  "vmaCreateImage(postComposite.hdrOut)");

            // One-shot UNDEFINED → GENERAL, same pattern as TaaResolve's
            // transitionFreshImage (this pass writes it via imageStore and
            // TaaResolve's RCAS/copy finalize reads it, both GENERAL-layout
            // compute-only consumers).
            VkCommandBufferAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool        = cmdPool_;
            ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            check(vkAllocateCommandBuffers(d, &ai, &cb), "alloc one-shot cb(postComposite.hdrOut)");
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(postComposite.hdrOut)");
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img.image;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
            check(vkEndCommandBuffer(cb), "end one-shot cb(postComposite.hdrOut)");
            VkSubmitInfo si{};
            si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers    = &cb;
            check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(postComposite.hdrOut)");
            check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(postComposite.hdrOut)");
            vkFreeCommandBuffers(d, cmdPool_, 1, &cb);

            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = img.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = img.format;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(d, &vci, nullptr, &img.view), "vkCreateImageView(postComposite.hdrOut)");
            ctx_.setObjectName(img.image, "vmaCreateImage(postComposite.hdrOut)");
            ctx_.setObjectName(img.view,  "vmaCreateImage(postComposite.hdrOut)");
        }
    }

    void PostComposite::setWhiteBalance(float temperatureK, float tint) {
        const bool neutral = std::abs(temperatureK - 6500.f) < 1.f && std::abs(tint) < 1e-4f;
        if (neutral) {
            wbActive_ = false;
            const float ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            std::memcpy(wbMat_, ident, sizeof(wbMat_));
            return;
        }
        float x, y;
        cctToXy(temperatureK, tint, x, y);
        const Mat3 m = bradfordToD65(x, y);
        std::memcpy(wbMat_, m.m, sizeof(wbMat_));
        wbActive_ = true;
    }

    void PostComposite::setColorGrade(const ColorGrade& g) {
        auto near1 = [](float v) { return std::abs(v - 1.f) < 1e-4f; };
        auto near0 = [](float v) { return std::abs(v) < 1e-4f; };
        const bool identity =
                near0(g.lift[0]) && near0(g.lift[1]) && near0(g.lift[2]) &&
                near1(g.gamma[0]) && near1(g.gamma[1]) && near1(g.gamma[2]) &&
                near1(g.gain[0]) && near1(g.gain[1]) && near1(g.gain[2]) &&
                near1(g.saturation) && near1(g.contrast);
        if (identity) {
            lutActive_ = false;
            return;
        }
        uploadLut(g);
        lutActive_ = true;
    }

    void PostComposite::uploadLut(const ColorGrade& g) {
        // Bake in the sRGB-ENCODED display-referred domain (matches the
        // shader's post-encode tap): lift/gain toe+shoulder, gamma power,
        // luma-preserving saturation, pivot-0.5 contrast.
        constexpr uint32_t N = kLutSize;
        std::vector<uint16_t> texels(N * N * N * 4);
        auto grade1 = [&](float c, int ch) {
            float v = c * g.gain[ch] + g.lift[ch] * (1.f - c);
            v = std::pow(std::max(v, 0.f), 1.f / std::max(g.gamma[ch], 1e-3f));
            return v;
        };
        size_t idx = 0;
        for (uint32_t bz = 0; bz < N; ++bz)
            for (uint32_t gy = 0; gy < N; ++gy)
                for (uint32_t rx = 0; rx < N; ++rx) {
                    float v[3] = {grade1(static_cast<float>(rx) / (N - 1), 0),
                                  grade1(static_cast<float>(gy) / (N - 1), 1),
                                  grade1(static_cast<float>(bz) / (N - 1), 2)};
                    const float luma = 0.2126f * v[0] + 0.7152f * v[1] + 0.0722f * v[2];
                    for (float& c : v) {
                        c = luma + (c - luma) * g.saturation;
                        c = (c - 0.5f) * g.contrast + 0.5f;
                        c = std::clamp(c, 0.f, 1.f);
                    }
                    texels[idx++] = floatToHalf(v[0]);
                    texels[idx++] = floatToHalf(v[1]);
                    texels[idx++] = floatToHalf(v[2]);
                    texels[idx++] = floatToHalf(1.f);
                }

        const VkDeviceSize bytes = texels.size() * sizeof(uint16_t);
        Buffer staging = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                      VMA_MEMORY_USAGE_AUTO,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                              VMA_ALLOCATION_CREATE_MAPPED_BIT);
        VmaAllocationInfo info{};
        vmaGetAllocationInfo(ctx_.allocator(), staging.alloc, &info);
        std::memcpy(info.pMappedData, texels.data(), bytes);

        // One-shot copy + layout dance. UI-rate (a slider release), not per
        // frame — the queue-idle wait is deliberate simplicity.
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(gradeLut)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(gradeLut)");

        auto barrier = [&](VkImageLayout from, VkImageLayout to,
                           VkAccessFlags srcA, VkAccessFlags dstA,
                           VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = from;
            b.newLayout = to;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = gradeLut_.image;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
            b.srcAccessMask = srcA;
            b.dstAccessMask = dstA;
            vkCmdPipelineBarrier(cb, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
        };
        barrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {N, N, N};
        vkCmdCopyBufferToImage(cb, staging.handle, gradeLut_.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        check(vkEndCommandBuffer(cb), "end one-shot cb(gradeLut)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(gradeLut)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(gradeLut)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
        destroyBuffer(ctx_.allocator(), staging);
    }

    void PostComposite::createPipeline() {
        VkDevice d = ctx_.device();

        if (sampler_ == VK_NULL_HANDLE) {
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter    = VK_FILTER_LINEAR;
            sci.minFilter    = VK_FILTER_LINEAR;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod       = 0.f;
            check(vkCreateSampler(d, &sci, nullptr, &sampler_), "vkCreateSampler(postComposite)");
        }
        if (nearestSampler_ == VK_NULL_HANDLE) {
            // Uint texture (raster ids) — texelFetch only, but the combined
            // sampler must still be filter-legal for the integer format.
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter    = VK_FILTER_NEAREST;
            sci.minFilter    = VK_FILTER_NEAREST;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            check(vkCreateSampler(d, &sci, nullptr, &nearestSampler_),
                  "vkCreateSampler(postComposite.nearest)");
        }

        // Layout: combined samplers @0,1 ; storage images @2,3 ; sampler3D @4 ;
        // usampler2D @5 (raster ids) ; HDR-mode-only combined sampler @6
        // (resolved HDR history) ; HDR-mode-only storage image @7 (hdrOut_).
        VkDescriptorSetLayoutBinding bnd[8]{};
        bnd[0].binding = 0;
        bnd[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd[1].binding = 1;
        bnd[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd[2].binding = 2;
        bnd[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bnd[3].binding = 3;
        bnd[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bnd[4].binding = 4;
        bnd[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd[5].binding = 5;
        bnd[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd[6].binding = 6;
        bnd[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd[7].binding = 7;
        bnd[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        for (auto& b : bnd) { b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 8;
        dlci.pBindings = bnd;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(postComposite)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = 88;// 8×u32 + 3×vec4 white-balance rows + 2×u32 HDR-mode src extent
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(postComposite)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kPostCompositeCompSpv);
        smci.pCode    = kPostCompositeCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), "vkCreateShaderModule(post_composite)");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipeLayout_;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(post_composite)");
        vkDestroyShaderModule(d, mod, nullptr);
    }

    void PostComposite::createDescriptorPool() {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = framesInFlight_ * 5;// 4 base + 1 HDR-mode scene
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = framesInFlight_ * 3;// 2 base + 1 HDR-mode output

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(postComposite)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descPool_;
        ai.descriptorSetCount = framesInFlight_;
        ai.pSetLayouts        = layouts.data();
        sets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(ctx_.device(), &ai, sets_.data()),
              "vkAllocateDescriptorSets(postComposite)");
    }

    void PostComposite::rewriteDescriptors(const DescriptorWriteInputs& in) {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            auto sampled = [&](VkImageView v) {
                VkDescriptorImageInfo i{};
                i.sampler = sampler_;
                i.imageView = v;
                i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                return i;
            };
            auto storage = [&](VkImageView v) {
                VkDescriptorImageInfo i{};
                i.imageView = v;
                i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                return i;
            };
            VkDescriptorImageInfo cScene = sampled(in.sceneHdrPerFrame[f]);
            VkDescriptorImageInfo cBloom = sampled(in.bloomPerFrame[f]);
            VkDescriptorImageInfo cGbuf  = storage(in.gbufPerFrame[f]);
            VkDescriptorImageInfo cOut   = storage(in.taaInputPerFrame[f]);
            VkDescriptorImageInfo cLut   = sampled(gradeLut_.view);
            VkDescriptorImageInfo cIds{};
            cIds.sampler     = nearestSampler_;
            cIds.imageView   = in.rasterIdsPerFrame[f];
            cIds.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            // HDR-mode-only bindings. Fall back to the always-valid LDR views
            // when the caller hasn't supplied hdrScenePerFrame yet, or
            // resizeHdrOutput hasn't run yet (hdrOut_ still null on first
            // construction) — harmless, since hdrMode being false means the
            // shader never samples/writes these.
            VkDescriptorImageInfo cHdrScene = sampled(in.hdrScenePerFrame ? in.hdrScenePerFrame[f]
                                                                            : in.sceneHdrPerFrame[f]);
            VkDescriptorImageInfo cHdrOut   = storage(hdrOut_[f].view != VK_NULL_HANDLE
                                                               ? hdrOut_[f].view
                                                               : in.taaInputPerFrame[f]);

            VkWriteDescriptorSet w[8]{};
            auto setw = [&](int n, uint32_t bind, VkDescriptorType t,
                            const VkDescriptorImageInfo* info) {
                w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstSet = sets_[f];
                w[n].dstBinding = bind;
                w[n].descriptorCount = 1;
                w[n].descriptorType = t;
                w[n].pImageInfo = info;
            };
            setw(0, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cScene);
            setw(1, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cBloom);
            setw(2, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &cGbuf);
            setw(3, 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &cOut);
            setw(4, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cLut);
            setw(5, 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cIds);
            setw(6, 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cHdrScene);
            setw(7, 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          &cHdrOut);
            vkUpdateDescriptorSets(ctx_.device(), 8, w, 0, nullptr);
        }
    }

    void PostComposite::recordDispatch(VkCommandBuffer cb, uint32_t frame,
                                       uint32_t width, uint32_t height,
                                       uint32_t toneMapping, uint32_t exposureBits,
                                       uint32_t preExposureBits,
                                       bool bgIsSolidColor, float effBloomIntensity,
                                       bool skyFromRasterIds,
                                       uint32_t srcWidth, uint32_t srcHeight,
                                       bool hdrMode) {
        if (srcWidth == 0)  srcWidth  = width;
        if (srcHeight == 0) srcHeight = height;
        // The shade/resolve (and, when active, the bloom pyramid / HDR-mode
        // TAA resolve) wrote via compute; make those writes visible to this
        // dispatch's reads.
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo di{};
        di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &di);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);

        uint32_t intensityBits;
        std::memcpy(&intensityBits, &effBloomIntensity, sizeof(intensityBits));
        const uint32_t postFlags = (wbActive_ ? 1u : 0u) | (lutActive_ ? 2u : 0u) |
                                   (skyFromRasterIds ? 4u : 0u) | (hdrMode ? 8u : 0u);

        struct Pc {
            uint32_t u[8];
            float    wb[12];// 3 × vec4 rows
            uint32_t srcWH[2];// sky-mask source (render) extent, for HDR-mode
                              // dispatch-at-display-res texel-index scaling
        } pc{};
        pc.u[0] = toneMapping;
        pc.u[1] = exposureBits;
        pc.u[2] = bgIsSolidColor ? 1u : 0u;
        pc.u[3] = intensityBits;
        pc.u[4] = width;
        pc.u[5] = height;
        pc.u[6] = postFlags;
        pc.u[7] = preExposureBits;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                pc.wb[r * 4 + c] = wbMat_[r * 3 + c];
        pc.srcWH[0] = srcWidth;
        pc.srcWH[1] = srcHeight;
        static_assert(sizeof(Pc) == 88, "post_composite push-constant layout");
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
        // LDR mode: this wrote the TAA input; TaaResolve::recordResolve's
        // pre-barrier makes it visible to the temporal resolve. HDR mode:
        // this wrote the swapchain (or the pre-RCAS intermediate) directly —
        // the resolve already ran, upstream of this pass.
    }

}// namespace threepp::vulkan
