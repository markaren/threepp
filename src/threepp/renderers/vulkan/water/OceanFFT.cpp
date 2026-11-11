// FFT-based ocean primitive implementations. See OceanFFT.hpp for the
// public surface and architecture overview.

#include "threepp/renderers/vulkan/water/OceanFFT.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"// flushHostWrites

#include "threepp/renderers/vulkan/shaders/phillips_spectrum.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/dynamic_spectrum.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ifft_twiddle.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ifft_horizontal.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ifft_vertical.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ifft_permute.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ifft_shared.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ocean_chop_moments.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/dynamic_spectrum.comp.chop.spv.h"
#include "threepp/renderers/vulkan/shaders/foam_crest.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/foam_crest.comp.climate.spv.h"

#include "threepp/math/Rng.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace threepp::water {

    namespace {

        // ── Tiny helpers. The shared ones (vulkan::check, createBuffer) are in
        // VulkanResources.hpp; these stay local for the "OceanFFT:" prefix on
        // the error text and the storage + sampled image the cascades use.

        void check(VkResult r, const char* what) {
            if (r != VK_SUCCESS) {
                throw std::runtime_error(std::string("OceanFFT: ") + what +
                                         " failed (VkResult=" + std::to_string(static_cast<int>(r)) + ")");
            }
        }

        OceanImage makeStorageSampledImage(vulkan::VulkanContext& ctx,
                                           uint32_t w, uint32_t h, VkFormat fmt,
                                           const char* debugName = nullptr) {
            OceanImage img{};
            img.format = fmt;
            img.width  = w;
            img.height = h;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = fmt;
            ici.extent        = {w, h, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx.allocator(), &ici, &aci, &img.image, &img.alloc, nullptr),
                  "vmaCreateImage(storage|sampled)");

            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = img.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = fmt;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            check(vulkan::createImageView(ctx.device(), &vci, nullptr, &img.view),
                  "vkCreateImageView");

            if (debugName) {
                ctx.setObjectName(img.image, debugName);
                ctx.setObjectName(img.view,  debugName);
            }
            return img;
        }

        void destroyImage(vulkan::VulkanContext& ctx, OceanImage& img) {
            if (img.view != VK_NULL_HANDLE) {
                vkDestroyImageView(ctx.device(), img.view, nullptr);
                img.view = VK_NULL_HANDLE;
            }
            if (img.image != VK_NULL_HANDLE) {
                vmaDestroyImage(ctx.allocator(), img.image, img.alloc);
                img.image = VK_NULL_HANDLE;
                img.alloc = VK_NULL_HANDLE;
            }
        }

        // Transition a storage/sampled image to GENERAL layout so it can be
        // both read and written by the compute pipelines. We keep all the
        // FFT images in GENERAL throughout the frame to avoid ping-ponging
        // layouts every dispatch.
        void cmdTransitionToGeneral(VkCommandBuffer cb, OceanImage& img) {
            if (img.currentLayout == VK_IMAGE_LAYOUT_GENERAL) return;
            VkImageMemoryBarrier br{};
            br.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            br.oldLayout     = img.currentLayout;
            br.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            br.srcAccessMask = 0;
            br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            br.image         = img.image;
            br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &br);
            img.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        // Read-after-write image barrier between two compute dispatches that
        // share the same image (storage write → sampled/storage read).
        void cmdShaderRWBarrier(VkCommandBuffer cb, VkImage img) {
            VkImageMemoryBarrier br{};
            br.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            br.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            br.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            br.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            br.image         = img;
            br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &br);
        }

        VkSampler makeNearestSampler(vulkan::VulkanContext& ctx) {
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter    = VK_FILTER_NEAREST;
            sci.minFilter    = VK_FILTER_NEAREST;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sci.maxLod       = VK_LOD_CLAMP_NONE;
            VkSampler s = VK_NULL_HANDLE;
            check(vkCreateSampler(ctx.device(), &sci, nullptr, &s), "vkCreateSampler");
            return s;
        }

        VkShaderModule makeShader(vulkan::VulkanContext& ctx,
                                  const uint32_t* spv, size_t bytes) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = bytes;
            smci.pCode    = spv;
            VkShaderModule m = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx.device(), &smci, nullptr, &m),
                  "vkCreateShaderModule");
            return m;
        }

        VkPipeline makeComputePipeline(vulkan::VulkanContext& ctx,
                                       VkShaderModule mod,
                                       VkPipelineLayout layout) {
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = mod;
            stage.pName  = "main";

            VkComputePipelineCreateInfo cpci{};
            cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage  = stage;
            cpci.layout = layout;

            VkPipeline p = VK_NULL_HANDLE;
            check(vkCreateComputePipelines(ctx.device(), ctx.pipelineCache(), 1, &cpci, nullptr, &p),
                  "vkCreateComputePipelines");
            return p;
        }

        constexpr uint32_t kGroup = 8;
        uint32_t groupCountFor(uint32_t n) { return (n + kGroup - 1u) / kGroup; }

    }// namespace

    // ─────────────────────────────────────────────────────────────────
    //   PhillipsSpectrum
    // ─────────────────────────────────────────────────────────────────

    PhillipsSpectrum::PhillipsSpectrum(vulkan::VulkanContext& ctx, const Settings& s)
        : ctx_(ctx), settings_(s) {
        sampler_ = makeNearestSampler(ctx_);
        createImages();
        uploadNoise();
        createPipeline();
    }

    void PhillipsSpectrum::createImages() {
        // Output h0 packs h0(k) and conj(h0(-k)) into RGBA32F.
        h0_ = makeStorageSampledImage(ctx_, settings_.textureSize, settings_.textureSize,
                                      VK_FORMAT_R32G32B32A32_SFLOAT,
                                      "ocean.phillips.h0");
        // Noise: complex Gaussian, 2 channels.
        noise_ = makeStorageSampledImage(ctx_, settings_.textureSize, settings_.textureSize,
                                         VK_FORMAT_R32G32_SFLOAT,
                                         "ocean.phillips.noise");
    }

    void PhillipsSpectrum::uploadNoise() {
        const uint32_t N = settings_.textureSize;
        std::vector<float> data(N * N * 2);
        math::Rng rng(0xC0FFEEu);
        for (uint32_t i = 0; i < N * N; ++i) {
            data[2 * i + 0] = rng.nextGaussian();
            data[2 * i + 1] = rng.nextGaussian();
        }

        const VkDeviceSize bytes = data.size() * sizeof(float);
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = bytes;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer       sb       = VK_NULL_HANDLE;
        VmaAllocation  sa       = VK_NULL_HANDLE;
        VmaAllocationInfo info{};
        check(vmaCreateBuffer(ctx_.allocator(), &bci, &aci, &sb, &sa, &info),
              "vmaCreateBuffer(noiseStaging)");
        std::memcpy(info.pMappedData, data.data(), bytes);
        vulkan::flushHostWrites(ctx_.allocator(), sa, 0, bytes);

        // One-shot upload.
        VkCommandPoolCreateInfo pci{};
        pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = ctx_.queueFamilies().graphics;
        pci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        VkCommandPool pool = VK_NULL_HANDLE;
        check(vkCreateCommandPool(ctx_.device(), &pci, nullptr, &pool),
              "vkCreateCommandPool(noiseUpload)");

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = pool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &cbai, &cb),
              "vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo bbi{};
        bbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bbi), "vkBeginCommandBuffer(noise upload)");

        // Transition noise to TRANSFER_DST.
        VkImageMemoryBarrier br{};
        br.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        br.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        br.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        br.image         = noise_.image;
        br.srcAccessMask = 0;
        br.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        br.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &br);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent      = {N, N, 1};
        vkCmdCopyBufferToImage(cb, sb, noise_.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // Transition noise to SHADER_READ for sampling in the compute pass.
        br.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        br.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        br.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &br);
        noise_.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(noise upload)");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "vkQueueSubmit(noise upload)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "vkQueueWaitIdle(noise upload)");

        vkDestroyCommandPool(ctx_.device(), pool, nullptr);
        vmaDestroyBuffer(ctx_.allocator(), sb, sa);
    }

    void PhillipsSpectrum::createPipeline() {
        // Bindings: 0 = h0 (storage image, write), 1 = noise (sampled).
        // The spectrum params are push constants (see recordCompute).
        const std::array<VkDescriptorSetLayoutBinding, 2> bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsl_),
              "vkCreateDescriptorSetLayout(phillips)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(PushParams);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsl_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &layout_),
              "vkCreatePipelineLayout(phillips)");

        VkShaderModule mod = makeShader(ctx_, kPhillipsSpectrumCompSpv, sizeof(kPhillipsSpectrumCompSpv));
        pipe_ = makeComputePipeline(ctx_, mod, layout_);
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        const std::array<VkDescriptorPoolSize, 2> poolSizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        dpci.pPoolSizes    = poolSizes.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &pool_),
              "vkCreateDescriptorPool(phillips)");

        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &dsl_;
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, &ds_),
              "vkAllocateDescriptorSets(phillips)");

        VkDescriptorImageInfo h0Info{};
        h0Info.imageView   = h0_.view;
        h0Info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo noiseInfo{};
        noiseInfo.sampler     = sampler_;
        noiseInfo.imageView   = noise_.view;
        noiseInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet     = ds_;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &h0Info;

        writes[1].sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet     = ds_;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &noiseInfo;

        vulkan::updateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void PhillipsSpectrum::recordCompute(VkCommandBuffer cb) {
        cmdTransitionToGeneral(cb, h0_);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                                0, 1, &ds_, 0, nullptr);
        // Push constants: baked into THIS command buffer, so a sea-state change
        // recorded for the next frame cannot reach a dispatch still in flight.
        PushParams p{};
        p.textureSize     = settings_.textureSize;
        p.tileSize        = settings_.tileSize;
        p.windTheta       = settings_.windTheta;
        p.windSpeed       = settings_.windSpeed;
        p.smallWaveCutoff = settings_.smallWaveCutoff;
        p.kMin            = settings_.kMin;
        p.kMax            = settings_.kMax;
        p.fetch           = settings_.fetch;
        p.spread          = settings_.spread;
        p.swellScale      = swellScale(settings_);
        p.swellPeriod     = settings_.swellPeriod;
        p.swellTheta      = settings_.swellTheta;
        p.swellSpread     = swellSpreadClamped(settings_.swellSpread);
        p.shortWave       = settings_.shortWaveSpectrum ? 1u : 0u;
        vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        const uint32_t g = groupCountFor(settings_.textureSize);
        vkCmdDispatch(cb, g, g, 1);
        // Emit a write→read barrier so subsequent samples see the data.
        cmdShaderRWBarrier(cb, h0_.image);
    }

    void PhillipsSpectrum::updateSeaState(float windTheta, float windSpeed, float fetch, uint32_t spread) {
        settings_.windTheta = windTheta;
        settings_.windSpeed = windSpeed;
        settings_.fetch     = fetch;
        settings_.spread    = spread;
    }

    void PhillipsSpectrum::updateSwell(float height, float period, float theta, float spread) {
        settings_.swellHeight = height;
        settings_.swellPeriod = period;
        settings_.swellTheta  = theta;
        settings_.swellSpread = spread;
    }

    namespace {
        // CPU mirror of swellShape in phillips_spectrum.comp; keep the two
        // identical (the sources of each factor are cited there).
        double swellShapeCpu(double kx, double kz, double tp, double theta, double s) {
            constexpr double g  = 9.81;
            constexpr double pi = 3.14159265358979323846;
            const double kl = std::sqrt(kx * kx + kz * kz);
            const double w  = std::sqrt(g * kl);
            const double x  = tp * w / (2.0 * pi);
            const double sg = (x <= 1.0) ? 0.07 : 0.09;
            const double d  = (x - 1.0) / sg;
            const double gammaPeak = std::pow(6.0, std::exp(-0.5 * d * d));
            const double x2 = x * x;
            const double sf = std::exp(-1.25 / (x2 * x2)) * gammaPeak / (x2 * x2 * x);
            const double c  = std::clamp(-(kx * std::cos(theta) + kz * std::sin(theta)) / kl, -1.0, 1.0);
            const double halfCos = std::max(std::sqrt(0.5 * (1.0 + c)), 1e-6);
            return sf * std::pow(halfCos, 2.0 * s) / (w * kl);
        }

        // CPU mirror of donelanBannerWeight in phillips_spectrum.comp (float,
        // as there); the sources are cited there.
        float donelanBannerWeightCpu(float cosTheta, float wOverWp) {
            float beta;
            if (wOverWp < 0.56f) beta = 1.24f;
            else if (wOverWp < 0.95f) beta = 2.61f * std::pow(wOverWp, 1.3f);
            else if (wOverWp < 1.6f) beta = 2.28f * std::pow(wOverWp, -1.3f);
            else beta = std::pow(10.f, -0.4f + 0.8393f * std::exp(-0.567f * std::log(wOverWp * wOverWp)));
            const float theta = std::acos(std::clamp(cosTheta, -1.f, 1.f));
            const float ch = std::cosh(beta * theta);
            return 3.14159265358979f * 0.5f * beta / (ch * ch * std::tanh(beta * 3.14159265358979f));
        }

        // CPU mirror of elfouhailyShortCurvature in phillips_spectrum.comp
        // (double here); the equations and their sources are cited there.
        double elfouhailyShortCurvatureCpu(double k, double kp, double U) {
            constexpr double g = 9.81, km = 370.0, cm = 0.23;
            const double omegaC = U * std::sqrt(kp / g);
            const double z0 = 3.7e-5 * U * U / g * std::pow(omegaC, 0.9);
            const double us = 0.4 * U / std::log(10.0 / z0);
            const double lr = std::log(us / cm);
            const double am = std::max(0.01 * (1.0 + (us <= cm ? lr : 3.0 * lr)), 0.0);
            const double kr = k / km;
            const double c  = std::sqrt(g / k * (1.0 + kr * kr));
            const double pr = kp / k;
            return 0.5 * am * (cm / c) * std::exp(-1.25 * pr * pr) * std::exp(-0.25 * (kr - 1.0) * (kr - 1.0));
        }

        // The fade the shader puts on it: smoothstep over ln k from 2 k_p to 16 k_p.
        double shortWaveFadeCpu(double k, double kp) {
            const double t = std::clamp((std::log(k) - std::log(2.0 * kp)) / std::log(8.0), 0.0, 1.0);
            return t * t * (3.0 - 2.0 * t);
        }

        // The shader's peak length scale L (hs-fix and fetch included).
        double spectrumLengthCpu(const PhillipsSpectrum::Settings& S) {
            double L = double(S.windSpeed) * S.windSpeed / 9.81;
            L *= 1.1624764;
            if (S.fetch > 0.f) {
                const double Ffd = 1609.0 * S.windSpeed * S.windSpeed;
                L *= std::pow(std::clamp(S.fetch / Ffd, 0.005, 1.0), 2.0 / 3.0);
            }
            return L;
        }
    }// namespace

    float PhillipsSpectrum::unresolvedShortWaveMss() const {
        const Settings& S = settings_;
        if (!S.shortWaveSpectrum || S.kMax > 0.f || !(S.windSpeed > 0.f) || !(S.tileSize > 0.f)) return 0.f;
        // Slope variance of a curvature spectrum: mss = integral B(k) d ln k
        // (both axes; the directional weight integrates out). What the grid
        // draws of it is the roll-off exp(-(k l)^2) up to its Nyquist pi N /
        // tile (the square grid reaches past that on its diagonals only);
        // the rest is banked.
        const double kp   = 0.894 / spectrumLengthCpu(S);
        const double kNyq = 3.14159265358979 * double(S.textureSize) / double(S.tileSize);
        const double l    = S.smallWaveCutoff;
        const double lo   = std::log(std::max(double(S.kMin), 2.0 * kp));
        const double hi   = std::log(370.0 * 30.0);// B_h is under 1e-20 of its peak by then
        constexpr int n   = 4096;
        double mss = 0.0;
        for (int i = 0; i < n; ++i) {
            const double lk = lo + (i + 0.5) * (hi - lo) / n;
            const double k  = std::exp(lk);
            const double drawn = k < kNyq ? std::exp(-k * k * l * l) : 0.0;
            mss += elfouhailyShortCurvatureCpu(k, kp, S.windSpeed) * shortWaveFadeCpu(k, kp) * (1.0 - drawn);
        }
        return static_cast<float>(mss * (hi - lo) / n);
    }

    float PhillipsSpectrum::swellSpreadClamped(float s) {
        // s < 1 widens cos^(2s)(theta/2) past anything measured for swell
        // (Goda 2000 puts it at 10 and up) and s <= 0 makes pow(0, 2s)
        // undefined; max(1, NaN) is 1 too.
        return std::max(1.f, s);
    }

    float PhillipsSpectrum::swellScale(const Settings& s) {
        if (!(s.swellHeight > 0.f) || !(s.swellPeriod > 0.f) || !(s.tileSize > 0.f)) return 0.f;
        constexpr double g  = 9.81;
        constexpr double pi = 3.14159265358979323846;
        const double dk = 2.0 * pi / s.tileSize;
        // Peak wavenumber kp = (2 pi / Tp)^2 / g must sit inside the band and
        // at least two modes out from the origin, or the swell cannot be
        // drawn on this grid (and normalising its tail up to Hs would be wrong).
        const double kp = std::pow(2.0 * pi / s.swellPeriod, 2.0) / g;
        if (kp < 2.0 * dk || kp < s.kMin || (s.kMax > 0.f && kp > s.kMax)) {
            // Say so, once per period and tile asked for: a swell the user set
            // that silently draws nothing is worse than a line on stderr.
            static float warnedPeriod = 0.f, warnedTile = 0.f;
            if (s.swellPeriod != warnedPeriod || s.tileSize != warnedTile) {
                warnedPeriod = s.swellPeriod;
                warnedTile   = s.tileSize;
                const double kHi = std::min(s.kMax > 0.f ? double(s.kMax) : 1e9, 1e9);
                const double kLo = std::max(2.0 * dk, double(s.kMin));
                std::fprintf(stderr,
                             "[ocean] swell of %.1f s is not drawn: its peak wavelength %.0f m is outside what "
                             "cascade 0 (tile %.0f m) carries, periods %.1f to %.1f s\n",
                             double(s.swellPeriod), 2.0 * pi / kp, double(s.tileSize),
                             2.0 * pi / std::sqrt(g * kHi), 2.0 * pi / std::sqrt(g * kLo));
            }
            return 0.f;
        }
        const int n = static_cast<int>(s.textureSize);
        const double l = s.smallWaveCutoff;
        double sum = 0.0;
        for (int iz = 0; iz < n; ++iz) {
            for (int ix = 0; ix < n; ++ix) {
                const double kx = (ix - 0.5 * n) * dk;
                const double kz = (iz - 0.5 * n) * dk;
                const double k2 = kx * kx + kz * kz;
                const double kl = std::sqrt(k2);
                // The shader's band mask and its length(k) < 1e-4 guard.
                if (kl < 1e-4 || kl < s.kMin || (s.kMax > 0.f && kl > s.kMax)) continue;
                sum += swellShapeCpu(kx, kz, s.swellPeriod, s.swellTheta, swellSpreadClamped(s.swellSpread)) *
                       std::exp(-k2 * l * l);
            }
        }
        if (!(sum > 0.0)) return 0.f;
        // The rendered height is the un-normalised IFFT over tileSize, and
        // h0 = xi sqrt(P / 2) with E|xi|^2 = 2 gives E|h(k, t)|^2 = P(k) + P(-k),
        // so the height variance is 2 sum P / tile^2. Hs = 4 sqrt(variance).
        const double variance = (double(s.swellHeight) * s.swellHeight) / 16.0;
        return static_cast<float>(variance * double(s.tileSize) * s.tileSize / (2.0 * sum));
    }

    float PhillipsSpectrum::meanWavenumber() const {
        // phillips_spectrum.comp's phillips2D, line for line (float, as there),
        // its spreading and swell included.
        const Settings& S = settings_;
        const float g = 9.81f;
        const float l = S.smallWaveCutoff;
        const float swell = swellScale(S);
        auto phillips2D = [&](float kx, float kz) -> float {
            const float k2 = kx * kx + kz * kz;
            if (std::sqrt(k2) < 1e-4f) return 0.f;
            float L = S.windSpeed * S.windSpeed / g;
            L *= 1.1624764f;
            float Aeff = 0.01f;
            float gamma = 1.f;
            if (S.fetch > 0.f) {
                const float Ffd = 1609.f * S.windSpeed * S.windSpeed;
                const float r = std::clamp(S.fetch / Ffd, 0.005f, 1.f);
                L *= std::pow(r, 2.f / 3.f);
                Aeff *= std::pow(r, -0.22f);
                gamma = 1.f + 2.3f * std::sqrt(1.f - r);
            }
            const float kL2 = k2 * L * L;
            const float kn = std::sqrt(k2);
            float kw = (kx * std::cos(S.windTheta) + kz * std::sin(S.windTheta)) / kn;
            if (S.spread != 0u) {
                // Measured spreading, from -k as in the shader.
                const float wp = std::sqrt(g * 0.894f / L);
                kw = donelanBannerWeightCpu(-kw, std::sqrt(g * kn) / wp);
            } else {
                kw *= kw;
            }
            float p = Aeff * std::exp(-1.f / kL2) * kw * std::exp(-k2 * l * l) / (k2 * k2);
            if (gamma > 1.f) {
                const float w = std::sqrt(g * kn);
                const float wp = std::sqrt(g * 0.894f / L);
                const float sg = (w <= wp) ? 0.07f : 0.09f;
                const float d = (w - wp) / (sg * wp);
                p *= std::pow(gamma, std::exp(-0.5f * d * d));
            }
            p *= 2.5446900f;
            if (S.shortWaveSpectrum) {
                const double kp = 0.894 / double(L);
                p += float(2.0 * 3.14159265358979 * elfouhailyShortCurvatureCpu(kn, kp, S.windSpeed) *
                           shortWaveFadeCpu(kn, kp)) * kw * std::exp(-k2 * l * l) / (k2 * k2);
            }
            if (swell > 0.f)
                p += float(swell * swellShapeCpu(kx, kz, S.swellPeriod, S.swellTheta,
                                                 swellSpreadClamped(S.swellSpread))) *
                     std::exp(-k2 * l * l);
            return p;
        };
        const uint32_t n = S.textureSize;
        const double dk = 2.0 * 3.14159265358979 / double(S.tileSize);
        double sumP = 0.0, sumWP = 0.0;
        for (uint32_t iz = 0; iz < n; ++iz) {
            for (uint32_t ix = 0; ix < n; ++ix) {
                const float kx = float((double(ix) - double(n) * 0.5) * dk);
                const float kz = float((double(iz) - double(n) * 0.5) * dk);
                const float kl = std::sqrt(kx * kx + kz * kz);
                if (kl < S.kMin || (S.kMax > 0.f && kl > S.kMax)) continue;
                // Mode k carries P(k) and P(-k) (h0(k) + conj h0(-k)).
                const double p = double(phillips2D(kx, kz)) + double(phillips2D(-kx, -kz));
                sumP += p;
                sumWP += std::sqrt(double(g) * kl) * p;
            }
        }
        if (!(sumP > 0.0)) return 0.f;
        const double wm = sumWP / sumP;
        return float(wm * wm / g);
    }

    PhillipsSpectrum::~PhillipsSpectrum() {
        if (pipe_   != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipe_, nullptr);
        if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layout_, nullptr);
        if (dsl_    != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dsl_, nullptr);
        if (pool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), pool_, nullptr);
        if (sampler_!= VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
        destroyImage(ctx_, h0_);
        destroyImage(ctx_, noise_);
    }

    // ─────────────────────────────────────────────────────────────────
    //   DynamicSpectrum
    // ─────────────────────────────────────────────────────────────────

    DynamicSpectrum::DynamicSpectrum(vulkan::VulkanContext& ctx,
                                     const PhillipsSpectrum& src,
                                     uint32_t textureSize, float tileSize)
        : ctx_(ctx), src_(src), textureSize_(textureSize), tileSize_(tileSize) {
        sampler_ = makeNearestSampler(ctx_);
        createImages();
        createPipeline();
    }

    void DynamicSpectrum::createImages() {
        ht_           = makeStorageSampledImage(ctx_, textureSize_, textureSize_, VK_FORMAT_R32G32_SFLOAT, "ocean.dyn.ht");
        displacement_ = makeStorageSampledImage(ctx_, textureSize_, textureSize_, VK_FORMAT_R32G32_SFLOAT, "ocean.dyn.displacement");
    }

    void DynamicSpectrum::createPipeline() {
        // Binding indices 2 and 5 (the unused gradient / Jacobian-diagonal
        // spectra) were retired; gaps in a set layout are legal, so the
        // remaining bindings keep their historical numbers and the shader's
        // layout qualifiers stay put.
        // (Binding 4, the params UBO, is gone too: the per-frame time rides in
        // push constants — see recordCompute.)
        const std::array<VkDescriptorSetLayoutBinding, 3> bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // H0
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // HT
            VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Displacement
        };
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsl_),
              "vkCreateDescriptorSetLayout(dyn)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(PushParams);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsl_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &layout_),
              "vkCreatePipelineLayout(dyn)");

        VkShaderModule mod = makeShader(ctx_, kDynamicSpectrumCompSpv, sizeof(kDynamicSpectrumCompSpv));
        pipe_ = makeComputePipeline(ctx_, mod, layout_);
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        const std::array<VkDescriptorPoolSize, 2> poolSizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          2},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        dpci.pPoolSizes    = poolSizes.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &pool_),
              "vkCreateDescriptorPool(dyn)");

        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &dsl_;
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, &ds_),
              "vkAllocateDescriptorSets(dyn)");

        // src h0 must be in GENERAL layout for combined-sampler reads to work.
        // We sample at SHADER_READ_ONLY_OPTIMAL conventionally — but since the
        // PhillipsSpectrum recordCompute leaves it in GENERAL, do the same.
        VkDescriptorImageInfo h0Info{};
        h0Info.sampler     = sampler_;
        h0Info.imageView   = src_.h0View();
        h0Info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo htInfo{};
        htInfo.imageView   = ht_.view;
        htInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo dispInfo{};
        dispInfo.imageView   = displacement_.view;
        dispInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (auto& w : writes) {
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = ds_;
            w.descriptorCount = 1;
        }
        writes[0].dstBinding = 0; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[0].pImageInfo = &h0Info;
        writes[1].dstBinding = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          writes[1].pImageInfo = &htInfo;
        writes[2].dstBinding = 3; writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          writes[2].pImageInfo = &dispInfo;
        vulkan::updateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void DynamicSpectrum::enableChop(float taperSpacing) {
        chopTaperSpacing_ = taperSpacing;
        if (chopPipe_ != VK_NULL_HANDLE) return;
        slope_   = makeStorageSampledImage(ctx_, textureSize_, textureSize_, VK_FORMAT_R32G32_SFLOAT, "ocean.dyn.slope");
        jacDiag_ = makeStorageSampledImage(ctx_, textureSize_, textureSize_, VK_FORMAT_R32G32_SFLOAT, "ocean.dyn.jacDiag");

        // The OCEAN_CHOP variant's own set: the plain bindings plus Slope (2)
        // and JacDiag (4). Its own layout, pipeline and pool, so the plain
        // path above keeps exactly the ones it had.
        const std::array<VkDescriptorSetLayoutBinding, 5> bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // H0
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // HT
            VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Displacement
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Slope
            VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // JacDiag
        };
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &chopDsl_),
              "vkCreateDescriptorSetLayout(dyn chop)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(ChopPushParams);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &chopDsl_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &chopLayout_),
              "vkCreatePipelineLayout(dyn chop)");

        VkShaderModule mod = makeShader(ctx_, kDynamicSpectrumChopCompSpv, sizeof(kDynamicSpectrumChopCompSpv));
        chopPipe_ = makeComputePipeline(ctx_, mod, chopLayout_);
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        const std::array<VkDescriptorPoolSize, 2> poolSizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          4},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        dpci.pPoolSizes    = poolSizes.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &chopPool_),
              "vkCreateDescriptorPool(dyn chop)");

        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = chopPool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &chopDsl_;
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, &chopDs_),
              "vkAllocateDescriptorSets(dyn chop)");

        VkDescriptorImageInfo h0Info{};
        h0Info.sampler     = sampler_;
        h0Info.imageView   = src_.h0View();
        h0Info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        const std::array<VkImageView, 4> storage{ht_.view, displacement_.view, slope_.view, jacDiag_.view};
        const std::array<uint32_t, 4> storageBinding{1u, 3u, 2u, 4u};
        std::array<VkDescriptorImageInfo, 4> infos{};
        std::array<VkWriteDescriptorSet, 5> writes{};
        for (auto& w : writes) {
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = chopDs_;
            w.descriptorCount = 1;
        }
        writes[0].dstBinding = 0; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[0].pImageInfo = &h0Info;
        for (uint32_t i = 0; i < 4; ++i) {
            infos[i].imageView           = storage[i];
            infos[i].imageLayout         = VK_IMAGE_LAYOUT_GENERAL;
            writes[i + 1].dstBinding     = storageBinding[i];
            writes[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i + 1].pImageInfo     = &infos[i];
        }
        vulkan::updateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void DynamicSpectrum::recordCompute(VkCommandBuffer cb, float elapsedSeconds, bool chop) {
        chop = chop && chopPipe_ != VK_NULL_HANDLE;
        cmdTransitionToGeneral(cb, ht_);
        cmdTransitionToGeneral(cb, displacement_);
        if (chop) {
            cmdTransitionToGeneral(cb, slope_);
            cmdTransitionToGeneral(cb, jacDiag_);
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, chop ? chopPipe_ : pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, chop ? chopLayout_ : layout_,
                                0, 1, chop ? &chopDs_ : &ds_, 0, nullptr);
        // The frame time is a PUSH CONSTANT, recorded into this command buffer.
        // It used to be a single host-mapped UBO rewritten here every frame —
        // with two frames in flight, recording frame N+1 overwrote it before
        // frame N's dispatch had necessarily read it, so frame N's cascades
        // could evolve to frame N+1's time (a pacing-dependent skew that
        // showed up as sample_height() mismatches run to run).
        if (chop) {
            ChopPushParams p{};
            p.textureSize    = textureSize_;
            p.tileSize       = tileSize_;
            p.elapsedSeconds = elapsedSeconds;
            p.taperSpacing   = chopTaperSpacing_;
            vkCmdPushConstants(cb, chopLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        } else {
            PushParams p{};
            p.textureSize    = textureSize_;
            p.tileSize       = tileSize_;
            p.elapsedSeconds = elapsedSeconds;
            vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        }
        const uint32_t g = groupCountFor(textureSize_);
        vkCmdDispatch(cb, g, g, 1);

        cmdShaderRWBarrier(cb, ht_.image);
        cmdShaderRWBarrier(cb, displacement_.image);
        if (chop) {
            cmdShaderRWBarrier(cb, slope_.image);
            cmdShaderRWBarrier(cb, jacDiag_.image);
        }
    }

    DynamicSpectrum::~DynamicSpectrum() {
        if (pipe_   != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipe_, nullptr);
        if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layout_, nullptr);
        if (dsl_    != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dsl_, nullptr);
        if (pool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), pool_, nullptr);
        if (chopPipe_   != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), chopPipe_, nullptr);
        if (chopLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), chopLayout_, nullptr);
        if (chopDsl_    != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), chopDsl_, nullptr);
        if (chopPool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), chopPool_, nullptr);
        if (sampler_!= VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
        destroyImage(ctx_, ht_);
        destroyImage(ctx_, displacement_);
        destroyImage(ctx_, slope_);// no-ops while the chop variant was never enabled
        destroyImage(ctx_, jacDiag_);
    }

    // ─────────────────────────────────────────────────────────────────
    //   IFFT
    // ─────────────────────────────────────────────────────────────────

    IFFT::IFFT(vulkan::VulkanContext& ctx, uint32_t textureSize, uint32_t maxGroups)
        : ctx_(ctx), textureSize_(textureSize),
          logSize_(static_cast<uint32_t>(std::log2(static_cast<double>(textureSize)))),
          maxGroups_(std::max(maxGroups, 1u)) {
        sampler_ = makeNearestSampler(ctx_);
        createTwiddleImage();
        createPipelines();
        checkSharedLimits();
        setPath(defaultPath());
    }

    void IFFT::setPath(Path p) {
        path_ = (p == Path::Shared && sharedSupported_ && ensureSharedPipelines()) ? Path::Shared : Path::Butterfly;
    }

    IFFT::Path IFFT::defaultPath() {
        static const Path path = [] {
            const char* e = std::getenv("THREEPP_OCEAN_FFT");
            if (!e || !*e || std::strcmp(e, "shared") == 0) return Path::Shared;
            if (std::strcmp(e, "butterfly") == 0) return Path::Butterfly;
            std::fprintf(stderr, "[IFFT] THREEPP_OCEAN_FFT=%s is neither butterfly nor shared; using shared\n", e);
            return Path::Shared;
        }();
        return path;
    }

    void IFFT::createTwiddleImage() {
        // Twiddle table: width = log2(N), height = N. Storage in RGBA32F.
        twiddle_ = makeStorageSampledImage(ctx_, logSize_, textureSize_, VK_FORMAT_R32G32B32A32_SFLOAT, "ocean.ifft.twiddle");
    }

    void IFFT::createPipelines() {
        // ── Twiddle compute pipeline (1 storage image)
        VkDescriptorSetLayoutBinding twb{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dlciT{};
        dlciT.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlciT.bindingCount = 1;
        dlciT.pBindings    = &twb;
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlciT, nullptr, &dslTwiddle_),
              "vkCreateDescriptorSetLayout(twiddle)");

        VkPushConstantRange pcrT{};
        pcrT.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcrT.offset     = 0;
        pcrT.size       = sizeof(int32_t);

        VkPipelineLayoutCreateInfo plciT{};
        plciT.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plciT.setLayoutCount         = 1;
        plciT.pSetLayouts            = &dslTwiddle_;
        plciT.pushConstantRangeCount = 1;
        plciT.pPushConstantRanges    = &pcrT;
        check(vkCreatePipelineLayout(ctx_.device(), &plciT, nullptr, &layoutTwiddle_),
              "vkCreatePipelineLayout(twiddle)");

        VkShaderModule modT = makeShader(ctx_, kIfftTwiddleCompSpv, sizeof(kIfftTwiddleCompSpv));
        pipeTwiddle_ = makeComputePipeline(ctx_, modT, layoutTwiddle_);
        vkDestroyShaderModule(ctx_.device(), modT, nullptr);

        // ── Butterfly (horizontal/vertical) layout: 2 sampled, 1 storage
        const std::array<VkDescriptorSetLayoutBinding, 3> bbb{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dlciB{};
        dlciB.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlciB.bindingCount = static_cast<uint32_t>(bbb.size());
        dlciB.pBindings    = bbb.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlciB, nullptr, &dslButterfly_),
              "vkCreateDescriptorSetLayout(butterfly)");

        VkPushConstantRange pcrB{};
        pcrB.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcrB.offset     = 0;
        pcrB.size       = sizeof(int32_t);

        VkPipelineLayoutCreateInfo plciB{};
        plciB.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plciB.setLayoutCount         = 1;
        plciB.pSetLayouts            = &dslButterfly_;
        plciB.pushConstantRangeCount = 1;
        plciB.pPushConstantRanges    = &pcrB;
        check(vkCreatePipelineLayout(ctx_.device(), &plciB, nullptr, &layoutButterfly_),
              "vkCreatePipelineLayout(butterfly)");

        VkShaderModule modH = makeShader(ctx_, kIfftHorizontalCompSpv, sizeof(kIfftHorizontalCompSpv));
        pipeHorizontal_ = makeComputePipeline(ctx_, modH, layoutButterfly_);
        vkDestroyShaderModule(ctx_.device(), modH, nullptr);

        VkShaderModule modV = makeShader(ctx_, kIfftVerticalCompSpv, sizeof(kIfftVerticalCompSpv));
        pipeVertical_ = makeComputePipeline(ctx_, modV, layoutButterfly_);
        vkDestroyShaderModule(ctx_.device(), modV, nullptr);

        // ── Permute layout: 1 storage image, read and written (no PC). The
        // permute is a strict 1:1 texel map, so it runs in place on a single
        // image rather than sampling one and storing into another.
        const std::array<VkDescriptorSetLayoutBinding, 1> pbb{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dlciP{};
        dlciP.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlciP.bindingCount = static_cast<uint32_t>(pbb.size());
        dlciP.pBindings    = pbb.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlciP, nullptr, &dslPermute_),
              "vkCreateDescriptorSetLayout(permute)");

        VkPipelineLayoutCreateInfo plciP{};
        plciP.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plciP.setLayoutCount = 1;
        plciP.pSetLayouts    = &dslPermute_;
        check(vkCreatePipelineLayout(ctx_.device(), &plciP, nullptr, &layoutPermute_),
              "vkCreatePipelineLayout(permute)");

        VkShaderModule modP = makeShader(ctx_, kIfftPermuteCompSpv, sizeof(kIfftPermuteCompSpv));
        pipePermute_ = makeComputePipeline(ctx_, modP, layoutPermute_);
        vkDestroyShaderModule(ctx_.device(), modP, nullptr);

        // ── Pool: 1 twiddle set + maxGroups_ write-once groups, each
        // 4 butterfly (2 horizontal + 2 vertical) + 2 permute sets. A butterfly
        // set is 2 sampled + 1 storage; an in-place permute set is 1 storage
        // and samples nothing.
        // The shared path's sets come from the same pool: up to maxGroups_ of
        // them, each the twiddle table + kMaxBatch storage images.
        const std::array<VkDescriptorPoolSize, 2> poolSizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxGroups_ * (4 * 2) + maxGroups_},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1 + maxGroups_ * (4 * 1 + 2 * 1) + maxGroups_ * kMaxBatch},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1 + maxGroups_ * 6 + maxGroups_;
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        dpci.pPoolSizes    = poolSizes.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &pool_),
              "vkCreateDescriptorPool(ifft)");

        // Allocate the twiddle DS now (it's bound to a stable image).
        VkDescriptorSetAllocateInfo daiT{};
        daiT.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        daiT.descriptorPool     = pool_;
        daiT.descriptorSetCount = 1;
        daiT.pSetLayouts        = &dslTwiddle_;
        check(vkAllocateDescriptorSets(ctx_.device(), &daiT, &dsTwiddle_),
              "vkAllocateDescriptorSets(twiddle)");

        VkDescriptorImageInfo tw{};
        tw.imageView   = twiddle_.view;
        tw.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet wT{};
        wT.sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wT.dstSet     = dsTwiddle_;
        wT.dstBinding = 0;
        wT.descriptorCount = 1;
        wT.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wT.pImageInfo = &tw;
        vulkan::updateDescriptorSets(ctx_.device(), 1, &wT, 0, nullptr);

        // Butterfly/permute sets are allocated and wired per image pair in
        // groupFor — nothing more to allocate up front.
    }

    void IFFT::recordTwiddleOnce(VkCommandBuffer cb) {
        cmdTransitionToGeneral(cb, twiddle_);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTwiddle_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layoutTwiddle_,
                                0, 1, &dsTwiddle_, 0, nullptr);
        const int32_t sz = static_cast<int32_t>(textureSize_);
        vkCmdPushConstants(cb, layoutTwiddle_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(sz), &sz);
        // 8-tall workgroup, x = logSize, y = textureSize/2.
        vkCmdDispatch(cb, logSize_, (textureSize_ / 2 + 7) / 8, 1);
        cmdShaderRWBarrier(cb, twiddle_.image);
        twiddleComputed_ = true;
    }

    IFFT::DescGroup& IFFT::groupFor(const OceanImage& a, const OceanImage& b) {
        for (auto& g : groups_) {
            if (g.input == a.view && g.scratch == b.view) return g;
        }
        if (groups_.size() >= maxGroups_) {
            // An image pair this IFFT has never seen, with the cache full,
            // means images were recreated without recreating the IFFT — the
            // DisplacedMeshState contract says that cannot happen. Reusing or
            // rewriting a group here would race in-flight command buffers.
            throw std::runtime_error("[IFFT] descriptor-group cache exhausted — image pair churn without IFFT recreation");
        }

        DescGroup g{};
        g.input   = a.view;
        g.scratch = b.view;

        VkDescriptorSetAllocateInfo daiB{};
        daiB.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        daiB.descriptorPool     = pool_;
        daiB.descriptorSetCount = 1;
        daiB.pSetLayouts        = &dslButterfly_;
        for (auto* sets : {&g.h, &g.v}) {
            for (auto& ds : *sets) {
                check(vkAllocateDescriptorSets(ctx_.device(), &daiB, &ds),
                      "vkAllocateDescriptorSets(butterfly)");
            }
        }
        VkDescriptorSetAllocateInfo daiP{};
        daiP.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        daiP.descriptorPool     = pool_;
        daiP.descriptorSetCount = 1;
        daiP.pSetLayouts        = &dslPermute_;
        for (auto& ds : g.p) {
            check(vkAllocateDescriptorSets(ctx_.device(), &daiP, &ds),
                  "vkAllocateDescriptorSets(permute)");
        }

        const VkDescriptorImageInfo twInfo{ sampler_, twiddle_.view, VK_IMAGE_LAYOUT_GENERAL };
        const VkDescriptorImageInfo aSampled{ sampler_, a.view, VK_IMAGE_LAYOUT_GENERAL };
        const VkDescriptorImageInfo bSampled{ sampler_, b.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo aStorage{}; aStorage.imageView = a.view; aStorage.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo bStorage{}; bStorage.imageView = b.view; bStorage.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        auto writeButterfly = [&](VkDescriptorSet ds, const VkDescriptorImageInfo& read, const VkDescriptorImageInfo& write) {
            std::array<VkWriteDescriptorSet, 3> w{};
            for (auto& e : w) { e.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; e.dstSet = ds; e.descriptorCount = 1; }
            w[0].dstBinding = 0; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &twInfo;
            w[1].dstBinding = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &read;
            w[2].dstBinding = 2; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[2].pImageInfo = &write;
            vulkan::updateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
        };
        writeButterfly(g.h[0], aSampled, bStorage);
        writeButterfly(g.h[1], bSampled, aStorage);
        writeButterfly(g.v[0], aSampled, bStorage);
        writeButterfly(g.v[1], bSampled, aStorage);

        // The permute reads and writes one image in place, so a permute set is
        // a single storage binding. Both are written even though the butterfly
        // parity means only p[0] is ever bound today: an allocated set left
        // undescribed is a landmine for whoever changes that parity.
        auto writePermute = [&](VkDescriptorSet ds, const VkDescriptorImageInfo& inPlace) {
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = ds;
            w.descriptorCount = 1;
            w.dstBinding      = 0;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w.pImageInfo      = &inPlace;
            vulkan::updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        };
        writePermute(g.p[0], aStorage);
        writePermute(g.p[1], bStorage);

        groups_.push_back(g);
        return groups_.back();
    }

    void IFFT::checkSharedLimits() {
        // 256 threads (ifft_shared.comp's fixed size), ROWS lines a
        // workgroup so that every thread has a butterfly up to N = 512, and in
        // shared memory each line's ping-pong pair plus one twiddle column:
        // (ROWS*2*N + N/2) complex fp32.
        constexpr uint32_t kThreads = 256;
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &props);
        const uint32_t half = std::max(textureSize_ / 2u, 1u);
        sharedRows_ = std::clamp(kThreads / half, 1u, std::max(textureSize_, 1u));
        const uint32_t sharedBytes = (sharedRows_ * 2u * textureSize_ + textureSize_ / 2u) * 8u;
        sharedSupported_ = textureSize_ >= 2u && std::has_single_bit(textureSize_) &&
                           (1u << logSize_) == textureSize_ &&
                           kThreads <= props.limits.maxComputeWorkGroupInvocations &&
                           kThreads <= props.limits.maxComputeWorkGroupSize[0] &&
                           sharedBytes <= props.limits.maxComputeSharedMemorySize &&
                           textureSize_ / sharedRows_ <= props.limits.maxComputeWorkGroupCount[1];
    }

    void IFFT::destroySharedPipelines() {
        VkDevice d = ctx_.device();
        if (pipeSharedRows_ != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeSharedRows_, nullptr);
        if (pipeSharedCols_ != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeSharedCols_, nullptr);
        if (layoutShared_   != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, layoutShared_, nullptr);
        if (dslShared_      != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(d, dslShared_, nullptr);
        pipeSharedRows_ = pipeSharedCols_ = VK_NULL_HANDLE;
        layoutShared_ = VK_NULL_HANDLE;
        dslShared_ = VK_NULL_HANDLE;
    }

    bool IFFT::ensureSharedPipelines() {
        if (pipeSharedRows_ != VK_NULL_HANDLE && pipeSharedCols_ != VK_NULL_HANDLE) return true;
        if (!sharedSupported_) return false;
        try {
            buildSharedPipelines();
            return true;
        } catch (const std::exception& e) {
            // A driver that will not build the shared kernel (its shared array
            // is sized by specialization constants) must not take the
            // butterfly path down with it: this IFFT stays on that path.
            destroySharedPipelines();
            sharedSupported_ = false;
            std::fprintf(stderr, "[IFFT] N=%u: the shared-memory FFT did not build (%s); the butterfly path runs\n",
                         textureSize_, e.what());
            return false;
        }
    }

    void IFFT::buildSharedPipelines() {
        const std::array<VkDescriptorSetLayoutBinding, 2> bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,         VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          kMaxBatch, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dslShared_),
              "vkCreateDescriptorSetLayout(ifft shared)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(uint32_t);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dslShared_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &layoutShared_),
              "vkCreatePipelineLayout(ifft shared)");

        VkShaderModule mod = makeShader(ctx_, kIfftSharedCompSpv, sizeof(kIfftSharedCompSpv));
        for (uint32_t vertical = 0; vertical < 2; ++vertical) {
            // constant_id 0: N, 1: log2 N, 2: lines a workgroup, 3: rows (0) or columns (1).
            const std::array<int32_t, 4> values{
                static_cast<int32_t>(textureSize_), static_cast<int32_t>(logSize_),
                static_cast<int32_t>(sharedRows_), static_cast<int32_t>(vertical)};
            std::array<VkSpecializationMapEntry, 4> entries{};
            for (uint32_t k = 0; k < 4; ++k) entries[k] = VkSpecializationMapEntry{k, k * 4u, 4u};
            VkSpecializationInfo si{};
            si.mapEntryCount = static_cast<uint32_t>(entries.size());
            si.pMapEntries   = entries.data();
            si.dataSize      = sizeof(values);
            si.pData         = values.data();

            VkComputePipelineCreateInfo cpci{};
            cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = mod;
            cpci.stage.pName  = "main";
            cpci.stage.pSpecializationInfo = &si;
            cpci.layout       = layoutShared_;
            VkPipeline& p = vertical ? pipeSharedCols_ : pipeSharedRows_;
            const VkResult r = vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &cpci, nullptr, &p);
            if (r != VK_SUCCESS) {
                p = VK_NULL_HANDLE;
                vkDestroyShaderModule(ctx_.device(), mod, nullptr);
                check(r, "vkCreateComputePipelines(ifft shared)");
            }
        }
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);
    }

    VkDescriptorSet IFFT::sharedSetFor(OceanImage* const* images, uint32_t count) {
        SharedSet key{};
        key.count = count;
        for (uint32_t i = 0; i < count; ++i) key.views[i] = images[i]->view;
        for (const auto& s : sharedSets_) {
            if (s.count == key.count && s.views == key.views) return s.ds;
        }
        if (sharedSets_.size() >= maxGroups_) {
            // Same contract as groupFor: a set is written once and only bound
            // afterwards, so a new list of views past the cap means images
            // were recreated without recreating the IFFT.
            throw std::runtime_error("[IFFT] shared descriptor-set cache exhausted: image churn without IFFT recreation");
        }

        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &dslShared_;
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, &key.ds),
              "vkAllocateDescriptorSets(ifft shared)");

        const VkDescriptorImageInfo twInfo{sampler_, twiddle_.view, VK_IMAGE_LAYOUT_GENERAL};
        // Every slot must name a valid image: the shader's switch reaches
        // each array element statically. Unused slots repeat the first image
        // (the layer slots never select them).
        std::array<VkDescriptorImageInfo, kMaxBatch> io{};
        for (uint32_t i = 0; i < kMaxBatch; ++i) {
            io[i].imageView   = key.views[i < count ? i : 0u];
            io[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        std::array<VkWriteDescriptorSet, 2> w{};
        for (auto& e : w) { e.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; e.dstSet = key.ds; }
        w[0].dstBinding = 0; w[0].descriptorCount = 1;         w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &twInfo;
        w[1].dstBinding = 1; w[1].descriptorCount = kMaxBatch; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[1].pImageInfo = io.data();
        vulkan::updateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);

        sharedSets_.push_back(key);
        return key.ds;
    }

    uint32_t IFFT::prepareShared(VkCommandBuffer cb, OceanImage* const* images, uint32_t count, uint32_t mask,
                                 VkDescriptorSet& ds, uint32_t& slots) {
        if (count > kMaxBatch) {
            throw std::runtime_error("[IFFT] recordApplyBatch: more images than kMaxBatch");
        }
        if (!twiddleComputed_) recordTwiddleOnce(cb);
        ds = sharedSetFor(images, count);

        // The z slots: the k-th live image is transformed by z = k.
        uint32_t layers = 0;
        slots = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (images[i]->width != textureSize_ || images[i]->height != textureSize_) {
                throw std::runtime_error("[IFFT] recordApplyBatch: image size differs from the IFFT's");
            }
            // All of them, live or not: every slot is bound, so every image
            // must be in the layout its descriptor names.
            cmdTransitionToGeneral(cb, *images[i]);
            if (mask & (1u << i)) slots |= i << (4u * layers++);
        }
        return layers;
    }

    void IFFT::dispatchShared(VkCommandBuffer cb, VkDescriptorSet ds, uint32_t slots, uint32_t layers,
                              bool columns) const {
        // Workgroup (0, y, z) owns lines [y*ROWS, (y+1)*ROWS) of image slot z:
        // rows on the first pass, columns (and the permute's sign) on the second.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, columns ? pipeSharedCols_ : pipeSharedRows_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layoutShared_, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, layoutShared_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(slots), &slots);
        vkCmdDispatch(cb, 1, textureSize_ / sharedRows_, layers);
    }

    namespace {
        // The live bits of a batch of `count` images.
        uint32_t liveMask(uint32_t count, uint32_t mask) {
            return count >= 32u ? mask : mask & ((1u << count) - 1u);
        }
    }// namespace

    uint32_t IFFT::newGroupsFor(OceanImage* const* images, uint32_t count, const OceanImage& scratch,
                                uint32_t mask) const {
        uint32_t fresh = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (!(mask & (1u << i))) continue;
            bool known = false;
            for (const auto& g : groups_) known = known || (g.input == images[i]->view && g.scratch == scratch.view);
            // A repeat of an image earlier in this batch is one group, not two.
            for (uint32_t k = 0; k < i && !known; ++k) {
                known = (mask & (1u << k)) && images[k]->view == images[i]->view;
            }
            if (!known) ++fresh;
        }
        return fresh;
    }

    void IFFT::checkButterflyGroups(OceanImage* const* images, uint32_t count, const OceanImage& scratch,
                                    uint32_t mask) const {
        const uint32_t fresh = newGroupsFor(images, count, scratch, mask);
        if (groups_.size() + fresh > maxGroups_) {
            throw std::runtime_error(
                    "[IFFT] butterfly path: this batch needs " + std::to_string(groups_.size() + fresh) +
                    " (image, scratch) descriptor groups but the IFFT was built with maxGroups " +
                    std::to_string(maxGroups_) +
                    "; every image a batch hands it, other IFFTs' included, takes one, so build it with "
                    "maxGroups at least the number of distinct images it will transform");
        }
    }

    void IFFT::recordApplyJobs(VkCommandBuffer cb, const Job* jobs, uint32_t jobCount) {
        if (jobCount > 8u) throw std::runtime_error("[IFFT] recordApplyJobs: more than 8 jobs");
        bool allShared = true;
        for (uint32_t j = 0; j < jobCount; ++j) allShared = allShared && jobs[j].fft->path_ == Path::Shared;
        if (!allShared) {
            // Checked for every job before any is recorded, so a batch the
            // butterfly caches cannot take throws without leaving half a
            // transform in the command buffer. (A job on the Shared path
            // goes through its own recordApplyBatch below unchanged.)
            for (uint32_t j = 0; j < jobCount; ++j) {
                if (jobs[j].fft->path_ != Path::Shared) {
                    jobs[j].fft->checkButterflyGroups(jobs[j].images, jobs[j].count, *jobs[j].scratch,
                                                      liveMask(jobs[j].count, jobs[j].mask));
                }
            }
            for (uint32_t j = 0; j < jobCount; ++j) {
                jobs[j].fft->recordApplyBatch(cb, jobs[j].images, jobs[j].count, *jobs[j].scratch, jobs[j].mask);
            }
            return;
        }

        struct Prepared {
            VkDescriptorSet ds = VK_NULL_HANDLE;
            uint32_t slots = 0, layers = 0;
        };
        std::array<Prepared, 8> prep{};
        uint32_t live = 0;
        for (uint32_t j = 0; j < jobCount; ++j) {
            if (jobs[j].count > kMaxBatch) {
                throw std::runtime_error("[IFFT] recordApplyBatch: more than kMaxBatch images");
            }
            // An empty job binds nothing: its descriptor set would name no image.
            if (liveMask(jobs[j].count, jobs[j].mask) == 0u) continue;
            prep[j].layers = jobs[j].fft->prepareShared(cb, jobs[j].images, jobs[j].count, jobs[j].mask,
                                                        prep[j].ds, prep[j].slots);
            live += prep[j].layers;
        }
        if (live == 0) return;

        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        // Every job's rows, one barrier, every job's columns, one barrier: the
        // jobs touch disjoint images, so their dispatches overlap freely.
        for (const bool columns : {false, true}) {
            for (uint32_t j = 0; j < jobCount; ++j) {
                if (prep[j].layers) jobs[j].fft->dispatchShared(cb, prep[j].ds, prep[j].slots, prep[j].layers, columns);
            }
            // After the columns, as recordApply's final barrier: every
            // downstream reader of the images (water_displace, the height
            // readback, the ripple slopes) reads what the columns pass wrote.
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
    }

    void IFFT::recordApplyBatch(VkCommandBuffer cb, OceanImage* const* images, uint32_t count,
                                OceanImage& scratch, uint32_t mask) {
        if (count > kMaxBatch) {
            throw std::runtime_error("[IFFT] recordApplyBatch: more images than kMaxBatch");
        }
        mask = liveMask(count, mask);
        if (mask == 0u) return;// no image, or none live: nothing to record
        if (path_ == Path::Shared) {
            const Job job{this, images, count, &scratch, mask};
            recordApplyJobs(cb, &job, 1u);
            return;
        }
        checkButterflyGroups(images, count, scratch, mask);
        for (uint32_t i = 0; i < count; ++i) {
            if (mask & (1u << i)) recordApply(cb, *images[i], scratch);
        }
    }

    void IFFT::recordApply(VkCommandBuffer cb, OceanImage& input, OceanImage& scratch) {
        if (path_ == Path::Shared) {
            OceanImage* one = &input;
            recordApplyBatch(cb, &one, 1u, scratch, 1u);
            return;
        }
        if (!twiddleComputed_) recordTwiddleOnce(cb);

        // Write-once group for this image pair — record-time only BINDS, so
        // in-flight frames (and this frame's earlier chain) keep their sets.
        DescGroup& grp = groupFor(input, scratch);

        cmdTransitionToGeneral(cb, input);
        cmdTransitionToGeneral(cb, scratch);

        // pingPong: starts at false; flipped before each pass in the reference
        // implementation this follows.
        // We model the same: iter 0 → write to scratch (read input), iter 1 →
        // write to input (read scratch), and so on.
        bool pingPong = false;

        // Horizontal passes
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeHorizontal_);
        for (uint32_t step = 0; step < logSize_; ++step) {
            pingPong = !pingPong;
            VkDescriptorSet ds = pingPong ? grp.h[0] : grp.h[1];
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layoutButterfly_,
                                    0, 1, &ds, 0, nullptr);
            const int32_t s = static_cast<int32_t>(step);
            vkCmdPushConstants(cb, layoutButterfly_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(s), &s);
            const uint32_t g = groupCountFor(textureSize_);
            vkCmdDispatch(cb, g, g, 1);
            // Barrier on the destination image — next pass reads it.
            cmdShaderRWBarrier(cb, pingPong ? scratch.image : input.image);
        }

        // Vertical passes
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeVertical_);
        for (uint32_t step = 0; step < logSize_; ++step) {
            pingPong = !pingPong;
            VkDescriptorSet ds = pingPong ? grp.v[0] : grp.v[1];
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layoutButterfly_,
                                    0, 1, &ds, 0, nullptr);
            const int32_t s = static_cast<int32_t>(step);
            vkCmdPushConstants(cb, layoutButterfly_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(s), &s);
            const uint32_t g = groupCountFor(textureSize_);
            vkCmdDispatch(cb, g, g, 1);
            cmdShaderRWBarrier(cb, pingPong ? scratch.image : input.image);
        }

        // Permute, in place. The loops above flip pingPong exactly 2*logSize
        // times — an even count whatever logSize is — so it returns to the
        // `false` it started at and the butterfly result always lands back in
        // `input`, which is where the caller's contract says the spatial-domain
        // field must end up. Guard it rather than assume it: an in-place permute
        // on the wrong image would strand the result in `scratch` silently, and
        // the ocean would go quietly wrong instead of loudly.
        if (pingPong) {
            throw std::runtime_error("[IFFT] butterfly parity left the result in scratch — the in-place permute would strand it there");
        }
        // Unlike a butterfly, the permute maps texel (x,y) to texel (x,y) and
        // nothing else, so each invocation only read-modify-writes its own
        // texel. That makes it safe to run over `input` directly, which is why
        // there is no scratch → input copy back here: the result never leaves
        // the image it is already in, sparing a full N x N RG32F blit and the
        // four COMPUTE <-> TRANSFER layout barriers it had to be wrapped in.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipePermute_);
        // grp.p[0] permutes a (=input) in place; grp.p[1] is its mirror on b.
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layoutPermute_,
                                0, 1, &grp.p[0], 0, nullptr);
        const uint32_t g = groupCountFor(textureSize_);
        vkCmdDispatch(cb, g, g, 1);

        // Final barrier — the caller, and every downstream reader of `input`
        // (water_displace, the height readback), reads what the permute wrote.
        cmdShaderRWBarrier(cb, input.image);
    }

    IFFT::~IFFT() {
        destroySharedPipelines();
        if (pipeTwiddle_   != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipeTwiddle_, nullptr);
        if (pipeHorizontal_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipeHorizontal_, nullptr);
        if (pipeVertical_   != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipeVertical_, nullptr);
        if (pipePermute_    != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipePermute_, nullptr);
        if (layoutTwiddle_   != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layoutTwiddle_, nullptr);
        if (layoutButterfly_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layoutButterfly_, nullptr);
        if (layoutPermute_   != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layoutPermute_, nullptr);
        if (dslTwiddle_   != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dslTwiddle_, nullptr);
        if (dslButterfly_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dslButterfly_, nullptr);
        if (dslPermute_   != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dslPermute_, nullptr);
        if (pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), pool_, nullptr);
        if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
        destroyImage(ctx_, twiddle_);
    }

    // ─────────────────────────────────────────────────────────────────
    //   ChopMoments
    // ─────────────────────────────────────────────────────────────────

    ChopMoments::ChopMoments(vulkan::VulkanContext& ctx, uint32_t textureSize,
                             VkImageView htView, VkImageView slopeView, VkImageView jacView)
        : ctx_(ctx), textureSize_(textureSize) {
        for (uint32_t n = textureSize_; n > 1u; n >>= 1u) ++mipLevels_;

        moments_.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        moments_.width  = textureSize_;
        moments_.height = textureSize_;
        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = moments_.format;
        ici.extent        = {textureSize_, textureSize_, 1};
        ici.mipLevels     = mipLevels_;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &moments_.image, &moments_.alloc, nullptr),
              "vmaCreateImage(chopMoments)");

        // A storage view names exactly one level; the shade samples them all.
        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = moments_.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = moments_.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vulkan::createImageView(ctx_.device(), &vci, nullptr, &moments_.view), "vkCreateImageView(chopMoments mip0)");
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels_, 0, 1};
        check(vulkan::createImageView(ctx_.device(), &vci, nullptr, &sampledView_), "vkCreateImageView(chopMoments chain)");
        ctx_.setObjectName(moments_.image, "ocean.chopMoments (RGBA32F + mips)");

        // Repeat (the cascade tiles the world), trilinear, anisotropic: the
        // footprint of a pixel on water seen low is a long strip along the
        // line of sight, and an anisotropic fetch averages along it.
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &props);
        VkSamplerCreateInfo sci{};
        sci.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter        = VK_FILTER_LINEAR;
        sci.minFilter        = VK_FILTER_LINEAR;
        sci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.anisotropyEnable = VK_TRUE;
        sci.maxAnisotropy    = std::min(16.0f, props.limits.maxSamplerAnisotropy);
        sci.minLod           = 0.f;
        sci.maxLod           = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_), "vkCreateSampler(chopMoments)");

        const std::array<VkDescriptorSetLayoutBinding, 4> bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},// ht (h, dDx/dz)
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},// slope
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},// jacDiag
            VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},// moments mip 0
        };
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsl_), "vkCreateDescriptorSetLayout(chopMoments)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size       = sizeof(PushParams);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsl_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &layout_), "vkCreatePipelineLayout(chopMoments)");

        VkShaderModule mod = makeShader(ctx_, kOceanChopMomentsCompSpv, sizeof(kOceanChopMomentsCompSpv));
        pipe_ = makeComputePipeline(ctx_, mod, layout_);
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4};
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &poolSize;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &pool_), "vkCreateDescriptorPool(chopMoments)");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &dsl_;
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, &ds_), "vkAllocateDescriptorSets(chopMoments)");

        const std::array<VkImageView, 4> views{htView, slopeView, jacView, moments_.view};
        std::array<VkDescriptorImageInfo, 4> infos{};
        std::array<VkWriteDescriptorSet, 4> w{};
        for (uint32_t i = 0; i < 4; ++i) {
            infos[i].imageView   = views[i];
            infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = ds_;
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[i].pImageInfo      = &infos[i];
        }
        vulkan::updateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
    }

    void ChopMoments::record(VkCommandBuffer cb, float slopeScale, float chopScale,
                             VkPipelineStageFlags readerStages, float bankedVar) {
        auto barrier = [&](uint32_t baseMip, uint32_t mipCount, VkImageLayout oldLayout,
                           VkAccessFlags src, VkAccessFlags dst,
                           VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = oldLayout;
            b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            b.srcAccessMask       = src;
            b.dstAccessMask       = dst;
            b.image               = moments_.image;
            b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, 1};
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
        };
        // First use: UNDEFINED -> GENERAL for the whole chain; afterwards a
        // guard against the previous frame's sampled reads.
        if (moments_.currentLayout != VK_IMAGE_LAYOUT_GENERAL) {
            barrier(0, mipLevels_, VK_IMAGE_LAYOUT_UNDEFINED, 0,
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
            moments_.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
        } else {
            barrier(0, mipLevels_, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    readerStages,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &ds_, 0, nullptr);
        PushParams p{};
        p.textureSize = textureSize_;
        p.slopeScale  = slopeScale;
        p.chopScale   = chopScale;
        p.bankedVar   = bankedVar;
        vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        const uint32_t g = groupCountFor(textureSize_);
        vkCmdDispatch(cb, g, g, 1);

        // The chain by linear blits: level L is the box average of L-1, the
        // prefilter the shade's footprint fetch assumes. All in GENERAL.
        barrier(0, 1, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        int32_t w = static_cast<int32_t>(textureSize_);
        for (uint32_t level = 1; level < mipLevels_; ++level) {
            const int32_t nw = std::max(w >> 1, 1);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
            blit.srcOffsets[1]  = {w, w, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
            blit.dstOffsets[1]  = {nw, nw, 1};
            vkCmdBlitImage(cb, moments_.image, VK_IMAGE_LAYOUT_GENERAL,
                           moments_.image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_LINEAR);
            barrier(level, 1, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            w = nw;
        }
        barrier(0, mipLevels_, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, readerStages);
    }

    ChopMoments::~ChopMoments() {
        if (pipe_    != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipe_, nullptr);
        if (layout_  != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layout_, nullptr);
        if (dsl_     != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dsl_, nullptr);
        if (pool_    != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), pool_, nullptr);
        if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
        if (sampledView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx_.device(), sampledView_, nullptr);
        destroyImage(ctx_, moments_);
    }

    // ─────────────────────────────────────────────────────────────────
    //   FieldMips
    // ─────────────────────────────────────────────────────────────────

    FieldMips::FieldMips(vulkan::VulkanContext& ctx, uint32_t size, std::vector<VkImage> sources)
        : ctx_(ctx), size_(size), sources_(std::move(sources)) {
        for (uint32_t n = size_; n > 1u; n >>= 1u) ++mipLevels_;
        copies_.resize(sources_.size());
        views_.resize(sources_.size(), VK_NULL_HANDLE);
        for (size_t i = 0; i < sources_.size(); ++i) {
            auto& c  = copies_[i];
            c.format = VK_FORMAT_R32G32_SFLOAT;
            c.width  = size_;
            c.height = size_;
            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = c.format;
            ici.extent        = {size_, size_, 1};
            ici.mipLevels     = mipLevels_;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &c.image, &c.alloc, nullptr),
                  "vmaCreateImage(fieldMips)");
            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = c.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = c.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels_, 0, 1};
            check(vulkan::createImageView(ctx_.device(), &vci, nullptr, &views_[i]), "vkCreateImageView(fieldMips)");
            ctx_.setObjectName(c.image, "ocean.fieldMips (RG32F + mips)");
        }
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod       = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_), "vkCreateSampler(fieldMips)");
    }

    void FieldMips::record(VkCommandBuffer cb) {
        auto barrier = [&](VkImage image, uint32_t baseMip, uint32_t mipCount, VkImageLayout oldLayout,
                           VkAccessFlags src, VkAccessFlags dst,
                           VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = oldLayout;
            b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            b.srcAccessMask       = src;
            b.dstAccessMask       = dst;
            b.image               = image;
            b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, 1};
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
        };
        for (size_t i = 0; i < sources_.size(); ++i) {
            auto& c = copies_[i];
            // The source: what the inverse FFT just wrote, to be copied from.
            barrier(sources_[i], 0, 1, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            if (c.currentLayout != VK_IMAGE_LAYOUT_GENERAL) {
                barrier(c.image, 0, mipLevels_, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                c.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
            } else {
                barrier(c.image, 0, mipLevels_, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT);
            }
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent         = {size_, size_, 1};
            vkCmdCopyImage(cb, sources_[i], VK_IMAGE_LAYOUT_GENERAL, c.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            barrier(c.image, 0, 1, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            int32_t w = static_cast<int32_t>(size_);
            for (uint32_t level = 1; level < mipLevels_; ++level) {
                const int32_t nw = std::max(w >> 1, 1);
                VkImageBlit blit{};
                blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
                blit.srcOffsets[1]  = {w, w, 1};
                blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                blit.dstOffsets[1]  = {nw, nw, 1};
                vkCmdBlitImage(cb, c.image, VK_IMAGE_LAYOUT_GENERAL, c.image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                               VK_FILTER_LINEAR);
                barrier(c.image, level, 1, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                w = nw;
            }
            barrier(c.image, 0, mipLevels_, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            // The source goes back to its compute readers (the crest foam pass, the moments).
            barrier(sources_[i], 0, 1, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
    }

    FieldMips::~FieldMips() {
        if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
        for (auto& v : views_)
            if (v != VK_NULL_HANDLE) vkDestroyImageView(ctx_.device(), v, nullptr);
        for (auto& c : copies_) destroyImage(ctx_, c);
    }

    // ─────────────────────────────────────────────────────────────────
    //   CrestFoam
    // ─────────────────────────────────────────────────────────────────

    static_assert(sizeof(CrestFoam::PushParams) == 72,
                  "CrestFoam::PushParams must match foam_crest.comp's Pc layout (72 bytes)");

    CrestFoam::CrestFoam(vulkan::VulkanContext& ctx, uint32_t resolution,
                         VkImageView jac0, VkImageView ht0, VkImageView jac1, VkImageView ht1,
                         VkSampler c1Sampler, bool climateLevels)
        : ctx_(ctx), res_(resolution), climateLevels_(climateLevels) {
        for (uint32_t n = res_; n > 1u; n >>= 1u) ++mipLevels_;

        // .r the foam, .g unused.
        // With the wave climate's levels, .b and .a their foam.
        foam_.format = climateLevels_ ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R16G16_SFLOAT;
        foam_.width  = res_;
        foam_.height = res_;
        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = foam_.format;
        ici.extent        = {res_, res_, 1};
        ici.mipLevels     = mipLevels_;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &foam_.image, &foam_.alloc, nullptr),
              "vmaCreateImage(crestFoam)");

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = foam_.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = foam_.format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vulkan::createImageView(ctx_.device(), &vci, nullptr, &foam_.view), "vkCreateImageView(crestFoam mip0)");
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels_, 0, 1};
        check(vulkan::createImageView(ctx_.device(), &vci, nullptr, &sampledView_), "vkCreateImageView(crestFoam chain)");
        ctx_.setObjectName(foam_.image, "ocean.crestFoam (R16F + mips)");

        // The shade's sampler: repeat (the texture tiles rest space as cascade
        // 0 does), trilinear, anisotropic, for the same reason as the chop
        // moments': a pixel's footprint on water seen low is a long strip.
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &props);
        VkSamplerCreateInfo sci{};
        sci.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter        = VK_FILTER_LINEAR;
        sci.minFilter        = VK_FILTER_LINEAR;
        sci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.anisotropyEnable = VK_TRUE;
        sci.maxAnisotropy    = std::min(16.0f, props.limits.maxSamplerAnisotropy);
        sci.minLod           = 0.f;
        sci.maxLod           = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_), "vkCreateSampler(crestFoam)");
        // The pass's sampler on the cascade fields (single level).
        sci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.anisotropyEnable = VK_FALSE;
        sci.maxAnisotropy    = 1.0f;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &fieldSampler_), "vkCreateSampler(crestFoam fields)");

        const std::array<VkDescriptorSetLayoutBinding, 5> bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},// jac0
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},// ht0
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},// jac1
            VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},// ht1
            VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},         // foam mip 0
        };
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsl_), "vkCreateDescriptorSetLayout(crestFoam)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size       = sizeof(PushParams);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsl_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &layout_), "vkCreatePipelineLayout(crestFoam)");

        VkShaderModule mod = climateLevels_ ? makeShader(ctx_, kFoamCrestClimateCompSpv, sizeof(kFoamCrestClimateCompSpv))
                                            : makeShader(ctx_, kFoamCrestCompSpv, sizeof(kFoamCrestCompSpv));
        pipe_ = makeComputePipeline(ctx_, mod, layout_);
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        const std::array<VkDescriptorPoolSize, 2> poolSizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        dpci.pPoolSizes    = poolSizes.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &pool_), "vkCreateDescriptorPool(crestFoam)");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &dsl_;
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, &ds_), "vkAllocateDescriptorSets(crestFoam)");

        const std::array<VkImageView, 5> views{jac0, ht0, jac1, ht1, foam_.view};
        std::array<VkDescriptorImageInfo, 5> infos{};
        std::array<VkWriteDescriptorSet, 5> w{};
        for (uint32_t i = 0; i < 5; ++i) {
            infos[i].sampler     = i == 4u ? VK_NULL_HANDLE
                                 : ((i == 2u || i == 3u) && c1Sampler != VK_NULL_HANDLE) ? c1Sampler : fieldSampler_;
            infos[i].imageView   = views[i];
            infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = ds_;
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = i != 4u ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[i].pImageInfo      = &infos[i];
        }
        vulkan::updateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
    }

    void CrestFoam::record(VkCommandBuffer cb, PushParams p, VkPipelineStageFlags readerStages) {
        auto barrier = [&](uint32_t baseMip, uint32_t mipCount, VkImageLayout oldLayout,
                           VkAccessFlags src, VkAccessFlags dst,
                           VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = oldLayout;
            b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            b.srcAccessMask       = src;
            b.dstAccessMask       = dst;
            b.image               = foam_.image;
            b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, 1};
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
        };
        // The pass reads the level it writes (last frame's foam), so level 0
        // must hold something before the first dispatch: cleared once, no foam.
        if (foam_.currentLayout != VK_IMAGE_LAYOUT_GENERAL) {
            barrier(0, mipLevels_, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            const VkClearColorValue zero{};
            const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels_, 0, 1};
            vkCmdClearColorImage(cb, foam_.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
            barrier(0, mipLevels_, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
            foam_.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
        } else {
            // A guard against the previous frame's sampled reads.
            barrier(0, mipLevels_, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    readerStages,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &ds_, 0, nullptr);
        p.res = res_;
        vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        const uint32_t g = groupCountFor(res_);
        vkCmdDispatch(cb, g, g, 1);

        // The chain by linear blits: level L is the box average of L-1, so a
        // footprint fetch returns the mean foam under it. All in GENERAL.
        barrier(0, 1, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        int32_t w = static_cast<int32_t>(res_);
        for (uint32_t level = 1; level < mipLevels_; ++level) {
            const int32_t nw = std::max(w >> 1, 1);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
            blit.srcOffsets[1]  = {w, w, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
            blit.dstOffsets[1]  = {nw, nw, 1};
            vkCmdBlitImage(cb, foam_.image, VK_IMAGE_LAYOUT_GENERAL,
                           foam_.image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_LINEAR);
            barrier(level, 1, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            w = nw;
        }
        barrier(0, mipLevels_, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, readerStages);
    }

    CrestFoam::~CrestFoam() {
        if (pipe_    != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipe_, nullptr);
        if (layout_  != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layout_, nullptr);
        if (dsl_     != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dsl_, nullptr);
        if (pool_    != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), pool_, nullptr);
        if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
        if (fieldSampler_ != VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), fieldSampler_, nullptr);
        if (sampledView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx_.device(), sampledView_, nullptr);
        destroyImage(ctx_, foam_);
    }

}// namespace threepp::water
