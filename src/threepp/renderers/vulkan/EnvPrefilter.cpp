#include "threepp/renderers/vulkan/EnvPrefilter.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/prefilter_env.comp.spv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

    constexpr float kPi     = 3.14159265358979323846f;
    constexpr float kTwoPi  = 6.28318530717958647692f;
    constexpr float kDeg2Rad = kPi / 180.f;

    inline float texelLum(const float* p) {
        return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
    }

    // Direction + solid angle of equirect texel (x, y). MUST match the shaders'
    // sampleEnvLod mapping (u = 0.5 + atan(z,x)/2π, v = 0.5 + asin(y)/π), or
    // the re-injected analytic sun points away from the disc it replaced.
    inline void texelDir(uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                         float out[3], float* dOmega) {
        const float u   = (static_cast<float>(x) + 0.5f) / static_cast<float>(w);
        const float v   = (static_cast<float>(y) + 0.5f) / static_cast<float>(h);
        const float phi = (u - 0.5f) * kTwoPi;
        const float lat = (v - 0.5f) * kPi;
        const float cl  = std::cos(lat);
        out[0] = cl * std::cos(phi);
        out[1] = std::sin(lat);
        out[2] = cl * std::sin(phi);
        if (dOmega) *dOmega = cl * (kTwoPi / static_cast<float>(w)) * (kPi / static_cast<float>(h));
    }

}// namespace

namespace threepp::vulkan {

    EnvPrefilter::EnvPrefilter(VulkanContext& ctx, VkCommandPool cmdPool)
        : ctx_(ctx), cmdPool_(cmdPool) {
        createPipeline();
    }

    EnvPrefilter::~EnvPrefilter() {
        VkDevice d = ctx_.device();
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (srcSampler_)     vkDestroySampler(d, srcSampler_, nullptr);
    }

    void EnvPrefilter::createPipeline() {
        // Sampler used to read mip 0 during prefilter. REPEAT in u so the
        // longitudinal seam blends; CLAMP in v to avoid pole bleed.
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.minLod       = 0.0f;
        sci.maxLod       = 0.0f;// only mip 0 read during prefilter
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &srcSampler_),
              "vkCreateSampler(prefilter)");

        // Descriptor layout: 0 = sampled mip-0 view, 1 = storage view of target mip.
        std::array<VkDescriptorSetLayoutBinding, 2> b{};
        b[0].binding         = 0;
        b[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1;
        b[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding         = 1;
        b[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1;
        b[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(b.size());
        dlci.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(prefilter)");

        // Push constants: alpha (4) + numSamples (4) + 8 byte pad → 16.
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = 16;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(prefilter)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kPrefilterEnvCompSpv);
        smci.pCode    = kPrefilterEnvCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(prefilter)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(),
                                       1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(prefilter)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // Descriptor pool: kMaxEnvMips sets, each with one sampled + one storage.
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = kMaxEnvMips;
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[1].descriptorCount = kMaxEnvMips;

        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = kMaxEnvMips;
        pci.poolSizeCount = static_cast<uint32_t>(ps.size());
        pci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &pci, nullptr, &descPool_),
              "vkCreateDescriptorPool(prefilter)");
    }

    Image2D EnvPrefilter::buildPmrem(uint32_t w, uint32_t h,
                                     const float* pixels,
                                     VkDeviceSize byteSize,
                                     bool extractSun,
                                     SunExtract* outSun) {
        if (outSun) *outSun = {};

        // ── HDRI sun extraction (CPU, one-shot per env upload) ──────────────
        // Detect a dominant compact bright source and build a sun-clamped copy
        // to prefilter from. Detection: the peak texel must dominate the mean
        // luminance by 200× (bright clouds/sky bands sit at ~10-50×) and clear
        // an absolute HDR floor. The disc: every texel within 8° of the peak
        // whose luminance exceeds 5% of it. Fill: the mean of the 8°-16°
        // surround, so the replaced disc blends into the sky gradient (the
        // sub-threshold halo/glow around the sun stays in the env — desirable).
        std::vector<float> clamped;// sun-clamped prefilter source (empty = none)
        bool sunFound = false;
        if (extractSun && outSun && w > 1 && h > 1) {
            const size_t n = static_cast<size_t>(w) * h;
            float  maxL = 0.f; size_t maxI = 0; double sumL = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const float L = texelLum(pixels + 4 * i);
                sumL += L;
                if (L > maxL) { maxL = L; maxI = i; }
            }
            const float meanL = static_cast<float>(sumL / static_cast<double>(n));
            if (maxL > std::max(200.f * meanL, 32.f)) {
                float pd[3];
                texelDir(static_cast<uint32_t>(maxI % w), static_cast<uint32_t>(maxI / w), w, h, pd, nullptr);
                const float cosRegion = std::cos(8.f * kDeg2Rad);
                const float cosFill   = std::cos(16.f * kDeg2Rad);
                const float thresh    = 0.05f * maxL;
                double fill[3]{};
                size_t fillN = 0;
                std::vector<uint32_t> disc;
                for (uint32_t y = 0; y < h; ++y) {
                    for (uint32_t x = 0; x < w; ++x) {
                        float d[3], dO;
                        texelDir(x, y, w, h, d, &dO);
                        const float c = d[0] * pd[0] + d[1] * pd[1] + d[2] * pd[2];
                        if (c <= cosFill) continue;
                        const float* p = pixels + 4 * (static_cast<size_t>(y) * w + x);
                        if (c > cosRegion && texelLum(p) > thresh) {
                            disc.push_back(y * w + x);
                        } else {
                            for (int k = 0; k < 3; ++k) fill[k] += p[k];
                            ++fillN;
                        }
                    }
                }
                const float f[3] = {
                        fillN ? static_cast<float>(fill[0] / static_cast<double>(fillN)) : 0.f,
                        fillN ? static_cast<float>(fill[1] / static_cast<double>(fillN)) : 0.f,
                        fillN ? static_cast<float>(fill[2] / static_cast<double>(fillN)) : 0.f};
                // Energy accounting: the disc texels are REPLACED by the fill, so
                // that much radiance stays in the env — the analytic light must
                // carry only the EXCESS Σ max(L−fill,0)·dΩ, or the sun is counted
                // ~twice and every lit surface brightens (washed-out pastels).
                double E[3]{}, dAcc[3]{}, omega = 0.0;
                for (const uint32_t idx : disc) {
                    float d[3], dO;
                    texelDir(idx % w, idx / w, w, h, d, &dO);
                    const float* p = pixels + 4 * static_cast<size_t>(idx);
                    const float ex[3] = {std::max(p[0] - f[0], 0.f),
                                         std::max(p[1] - f[1], 0.f),
                                         std::max(p[2] - f[2], 0.f)};
                    const float exL = texelLum(ex);
                    for (int k = 0; k < 3; ++k) {
                        E[k]    += static_cast<double>(ex[k]) * dO;
                        dAcc[k] += static_cast<double>(d[k]) * exL * dO;
                    }
                    omega += dO;
                }
                const double dl = std::sqrt(dAcc[0] * dAcc[0] + dAcc[1] * dAcc[1] + dAcc[2] * dAcc[2]);
                if (omega > 0.0 && !disc.empty() && dl > 1e-12 &&
                    (E[0] + E[1] + E[2]) > 1e-4) {
                    for (int k = 0; k < 3; ++k) {
                        outSun->dir[k]    = static_cast<float>(dAcc[k] / dl);
                        outSun->colorE[k] = static_cast<float>(E[k]);
                    }
                    outSun->angularRadiusDeg = std::clamp(
                            std::sqrt(static_cast<float>(omega) / kPi) / kDeg2Rad, 0.05f, 10.f);
                    clamped.assign(pixels, pixels + 4 * n);
                    for (const uint32_t idx : disc) {
                        clamped[4 * static_cast<size_t>(idx) + 0] = f[0];
                        clamped[4 * static_cast<size_t>(idx) + 1] = f[1];
                        clamped[4 * static_cast<size_t>(idx) + 2] = f[2];
                    }
                    outSun->found = true;
                    sunFound      = true;
                }
            }
        }

        Image2D out{};
        out.width  = w;
        out.height = h;
        out.format = VK_FORMAT_R32G32B32A32_SFLOAT;

        const uint32_t maxDim   = std::max(w, h);
        const uint32_t fullMips = static_cast<uint32_t>(
                std::floor(std::log2(static_cast<float>(maxDim)))) + 1u;
        out.mipLevels = std::min(fullMips, kMaxEnvMips);

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = out.format;
        ici.extent        = {w, h, 1};
        ici.mipLevels     = out.mipLevels;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_STORAGE_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci,
                             &out.image, &out.alloc, nullptr),
              "vmaCreateImage(envPmrem)");
        ctx_.setObjectName(out.image, "envPmrem (HDR env prefilter)");

        // Staging buffer for mip 0.
        Buffer staging = createBuffer(
                ctx_.allocator(), ctx_.device(), byteSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);
        // Sun path: mip 0 first receives the CLAMPED copy so the prefilter
        // dispatches integrate a sun-free source; the ORIGINAL is copied back
        // over mip 0 in a second submission below (mirror/sky keep the disc).
        void* mapped = nullptr;
        vmaMapMemory(ctx_.allocator(), staging.alloc, &mapped);
        std::memcpy(mapped, sunFound ? clamped.data() : pixels, byteSize);
        vmaUnmapMemory(ctx_.allocator(), staging.alloc);

        // One-shot command buffer: upload mip 0, dispatch prefilter per mip,
        // transition all mips to SHADER_READ_ONLY_OPTIMAL.
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cmdPool_;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &cbai, &cb),
              "alloc one-shot cb(prefilter)");

        VkCommandBufferBeginInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &cbi), "begin one-shot cb(prefilter)");

        // Mip 0: UNDEFINED → TRANSFER_DST for upload.
        {
            VkImageMemoryBarrier br{};
            br.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            br.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            br.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.image               = out.image;
            br.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            br.subresourceRange.baseMipLevel = 0;
            br.subresourceRange.levelCount = 1;
            br.subresourceRange.layerCount = 1;
            br.srcAccessMask = 0;
            br.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &br);
        }
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cb, staging.handle, out.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // Transition: mip 0 (TRANSFER_DST → GENERAL for compute read), mips 1..N-1
        // (UNDEFINED → GENERAL for compute write). GENERAL is universal so reads
        // and writes coexist in the same dispatch.
        {
            std::array<VkImageMemoryBarrier, 2> brs{};
            brs[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            brs[0].oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            brs[0].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            brs[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            brs[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            brs[0].image               = out.image;
            brs[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            brs[0].subresourceRange.baseMipLevel = 0;
            brs[0].subresourceRange.levelCount = 1;
            brs[0].subresourceRange.layerCount = 1;
            brs[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            brs[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            brs[1] = brs[0];
            brs[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            brs[1].subresourceRange.baseMipLevel = 1;
            brs[1].subresourceRange.levelCount = out.mipLevels - 1;
            brs[1].srcAccessMask = 0;
            brs[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            const uint32_t brCount = (out.mipLevels > 1) ? 2u : 1u;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, brCount, brs.data());
        }

        // Full-chain sampling view (used by closest_hit at sample time).
        {
            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = out.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = out.format;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.baseMipLevel = 0;
            vci.subresourceRange.levelCount = out.mipLevels;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx_.device(), &vci, nullptr, &out.view),
                  "vkCreateImageView(envPmrem)");
            ctx_.setObjectName(out.view, "envPmrem (HDR env prefilter)");
        }

        // Sampler with LINEAR mip filtering so trilinear blends across mips.
        // maxLod = mipLevels - 1 lets textureLod sample the full chain.
        {
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter    = VK_FILTER_LINEAR;
            sci.minFilter    = VK_FILTER_LINEAR;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.minLod       = 0.0f;
            sci.maxLod       = static_cast<float>(out.mipLevels - 1);
            check(vkCreateSampler(ctx_.device(), &sci, nullptr, &out.sampler),
                  "vkCreateSampler(envPmrem)");
        }

        // Per-mip storage views — created and destroyed inside this scope.
        std::vector<VkImageView> mipStorageViews;
        mipStorageViews.reserve(out.mipLevels);
        if (out.mipLevels > 1) {
            vkResetDescriptorPool(ctx_.device(), descPool_, 0);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

            for (uint32_t mip = 1; mip < out.mipLevels; ++mip) {
                VkImageViewCreateInfo vci{};
                vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image    = out.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format   = out.format;
                vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                vci.subresourceRange.baseMipLevel = mip;
                vci.subresourceRange.levelCount = 1;
                vci.subresourceRange.layerCount = 1;
                VkImageView mipView = VK_NULL_HANDLE;
                check(vkCreateImageView(ctx_.device(), &vci, nullptr, &mipView),
                      "vkCreateImageView(prefilter mip)");
                mipStorageViews.push_back(mipView);

                VkDescriptorSetAllocateInfo ai{};
                ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool     = descPool_;
                ai.descriptorSetCount = 1;
                ai.pSetLayouts        = &dsLayout_;
                VkDescriptorSet set = VK_NULL_HANDLE;
                check(vkAllocateDescriptorSets(ctx_.device(), &ai, &set),
                      "vkAllocateDescriptorSets(prefilter)");

                VkDescriptorImageInfo srcInfo{};
                srcInfo.sampler     = srcSampler_;
                srcInfo.imageView   = out.view;// full-chain view; sampler reads mip 0
                srcInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkDescriptorImageInfo dstInfo{};
                dstInfo.sampler     = VK_NULL_HANDLE;
                dstInfo.imageView   = mipView;
                dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                std::array<VkWriteDescriptorSet, 2> ws{};
                ws[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[0].dstSet          = set;
                ws[0].dstBinding      = 0;
                ws[0].descriptorCount = 1;
                ws[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                ws[0].pImageInfo      = &srcInfo;
                ws[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[1].dstSet          = set;
                ws[1].dstBinding      = 1;
                ws[1].descriptorCount = 1;
                ws[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                ws[1].pImageInfo      = &dstInfo;
                vkUpdateDescriptorSets(ctx_.device(), 2, ws.data(), 0, nullptr);

                struct Pc {
                    float    alpha;
                    uint32_t numSamples;
                    uint32_t _pad0;
                    uint32_t _pad1;
                } pc{};
                const float r = static_cast<float>(mip) /
                                static_cast<float>(out.mipLevels - 1);
                pc.alpha      = r * r;// GGX α = roughness²
                // 512 (was 64): a tiny, very bright HDR sun under a narrow lobe needs
                // many samples to integrate smoothly — 64 left it quantized into
                // discrete bright lumps. Paired with the per-texel Cranley-Patterson
                // rotation in the shader (decorrelates the residual into absorbable
                // noise). One-shot per env upload, so essentially free at runtime.
                pc.numSamples = 512u;
                vkCmdPushConstants(cb, pipelineLayout_,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipelineLayout_, 0, 1, &set,
                                        0, nullptr);

                const uint32_t mipW = std::max(1u, w >> mip);
                const uint32_t mipH = std::max(1u, h >> mip);
                const uint32_t gx = (mipW + 7u) / 8u;
                const uint32_t gy = (mipH + 7u) / 8u;
                vkCmdDispatch(cb, gx, gy, 1u);
                // No barrier between dispatches: each writes a different mip
                // and reads only mip 0, no hazard.
            }
        }

        if (!sunFound) {
            // Final transition: all mips → SHADER_READ_ONLY_OPTIMAL for sampling.
            VkImageMemoryBarrier br{};
            br.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            br.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
            br.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.image               = out.image;
            br.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            br.subresourceRange.baseMipLevel = 0;
            br.subresourceRange.levelCount = out.mipLevels;
            br.subresourceRange.layerCount = 1;
            br.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                 0, 0, nullptr, 0, nullptr, 1, &br);
        }

        check(vkEndCommandBuffer(cb), "end one-shot cb(prefilter)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        const VkResult sr = vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE);
        if (sr != VK_SUCCESS) check(sr, "submit one-shot(prefilter)");
        const VkResult wr = vkQueueWaitIdle(ctx_.graphicsQueue());
        if (wr != VK_SUCCESS) check(wr, "wait one-shot(prefilter)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);

        // Sun path, second submission: mip 0 still holds the clamped source —
        // restore the ORIGINAL equirect (sky / clear glass / mirror lookups keep
        // the real sun disc; only mips 1+ stay sun-free), then finalize layouts.
        if (sunFound) {
            vmaMapMemory(ctx_.allocator(), staging.alloc, &mapped);
            std::memcpy(mapped, pixels, byteSize);
            vmaUnmapMemory(ctx_.allocator(), staging.alloc);

            VkCommandBuffer cb2 = VK_NULL_HANDLE;
            check(vkAllocateCommandBuffers(ctx_.device(), &cbai, &cb2),
                  "alloc one-shot cb2(prefilter sun)");
            check(vkBeginCommandBuffer(cb2, &cbi), "begin one-shot cb2(prefilter sun)");

            VkImageMemoryBarrier br{};
            br.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.image               = out.image;
            br.subresourceRange.aspectMask   = VK_IMAGE_ASPECT_COLOR_BIT;
            br.subresourceRange.baseMipLevel = 0;
            br.subresourceRange.levelCount   = 1;
            br.subresourceRange.layerCount   = 1;
            // mip 0: GENERAL → TRANSFER_DST (prior submission already retired).
            br.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            br.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            br.srcAccessMask = 0;
            br.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb2,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &br);
            vkCmdCopyBufferToImage(cb2, staging.handle, out.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            std::array<VkImageMemoryBarrier, 2> fin{};
            fin[0] = br;
            fin[0].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            fin[0].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            fin[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fin[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            fin[1] = fin[0];
            fin[1].oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            fin[1].srcAccessMask = 0;
            fin[1].subresourceRange.baseMipLevel = 1;
            fin[1].subresourceRange.levelCount   = out.mipLevels - 1;
            const uint32_t finCount = (out.mipLevels > 1) ? 2u : 1u;
            vkCmdPipelineBarrier(cb2,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                 0, 0, nullptr, 0, nullptr, finCount, fin.data());

            check(vkEndCommandBuffer(cb2), "end one-shot cb2(prefilter sun)");
            si.pCommandBuffers = &cb2;
            const VkResult sr2 = vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE);
            if (sr2 != VK_SUCCESS) check(sr2, "submit one-shot cb2(prefilter sun)");
            const VkResult wr2 = vkQueueWaitIdle(ctx_.graphicsQueue());
            if (wr2 != VK_SUCCESS) check(wr2, "wait one-shot cb2(prefilter sun)");
            vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb2);
        }

        destroyBuffer(ctx_.allocator(), staging);
        for (auto v : mipStorageViews) {
            vkDestroyImageView(ctx_.device(), v, nullptr);
        }

        return out;
    }

}// namespace threepp::vulkan
