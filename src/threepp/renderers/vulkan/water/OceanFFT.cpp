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

#include "threepp/math/Rng.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace threepp::water {

    namespace {

        // ── Tiny helpers (mirroring VulkanRenderer.cpp's createBuffer/createImage,
        // but kept self-contained so this TU doesn't need access to the renderer's
        // private helpers).

        void check(VkResult r, const char* what) {
            if (r != VK_SUCCESS) {
                throw std::runtime_error(std::string("OceanFFT: ") + what +
                                         " failed (VkResult=" + std::to_string(static_cast<int>(r)) + ")");
            }
        }

        OceanBuffer makeUbo(vulkan::VulkanContext& ctx, VkDeviceSize size) {
            OceanBuffer b{};
            b.size = size;
            VkBufferCreateInfo bci{};
            bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size        = size;
            bci.usage       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo aci{};
            aci.usage         = VMA_MEMORY_USAGE_AUTO;
            aci.flags         = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo info{};
            check(vmaCreateBuffer(ctx.allocator(), &bci, &aci, &b.handle, &b.alloc, &info),
                  "vmaCreateBuffer(ubo)");
            b.mapped = info.pMappedData;
            return b;
        }

        void destroyBuffer(vulkan::VulkanContext& ctx, OceanBuffer& b) {
            if (b.handle != VK_NULL_HANDLE) {
                vmaDestroyBuffer(ctx.allocator(), b.handle, b.alloc);
                b = {};
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
            check(vkCreateImageView(ctx.device(), &vci, nullptr, &img.view),
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
        vkBeginCommandBuffer(cb, &bbi);

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

        vkEndCommandBuffer(cb);

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(ctx_.graphicsQueue());

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

        vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
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
        vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        const uint32_t g = groupCountFor(settings_.textureSize);
        vkCmdDispatch(cb, g, g, 1);
        // Emit a write→read barrier so subsequent samples see the data.
        cmdShaderRWBarrier(cb, h0_.image);
    }

    void PhillipsSpectrum::updateSeaState(float windTheta, float windSpeed, float fetch) {
        settings_.windTheta = windTheta;
        settings_.windSpeed = windSpeed;
        settings_.fetch     = fetch;
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
        vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void DynamicSpectrum::recordCompute(VkCommandBuffer cb, float elapsedSeconds) {
        cmdTransitionToGeneral(cb, ht_);
        cmdTransitionToGeneral(cb, displacement_);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                                0, 1, &ds_, 0, nullptr);
        // The frame time is a PUSH CONSTANT, recorded into this command buffer.
        // It used to be a single host-mapped UBO rewritten here every frame —
        // with two frames in flight, recording frame N+1 overwrote it before
        // frame N's dispatch had necessarily read it, so frame N's cascades
        // could evolve to frame N+1's time (a pacing-dependent skew that
        // showed up as sample_height() mismatches run to run).
        PushParams p{};
        p.textureSize    = textureSize_;
        p.tileSize       = tileSize_;
        p.elapsedSeconds = elapsedSeconds;
        vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        const uint32_t g = groupCountFor(textureSize_);
        vkCmdDispatch(cb, g, g, 1);

        cmdShaderRWBarrier(cb, ht_.image);
        cmdShaderRWBarrier(cb, displacement_.image);
    }

    DynamicSpectrum::~DynamicSpectrum() {
        if (pipe_   != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipe_, nullptr);
        if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layout_, nullptr);
        if (dsl_    != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dsl_, nullptr);
        if (pool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), pool_, nullptr);
        if (sampler_!= VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
        destroyImage(ctx_, ht_);
        destroyImage(ctx_, displacement_);
    }

    // ─────────────────────────────────────────────────────────────────
    //   IFFT
    // ─────────────────────────────────────────────────────────────────

    IFFT::IFFT(vulkan::VulkanContext& ctx, uint32_t textureSize)
        : ctx_(ctx), textureSize_(textureSize),
          logSize_(static_cast<uint32_t>(std::log2(static_cast<double>(textureSize)))) {
        sampler_ = makeNearestSampler(ctx_);
        createTwiddleImage();
        createPipelines();
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

        // ── Pool: 1 twiddle set + kMaxDescGroups write-once groups, each
        // 4 butterfly (2 horizontal + 2 vertical) + 2 permute sets. A butterfly
        // set is 2 sampled + 1 storage; an in-place permute set is 1 storage
        // and samples nothing.
        const std::array<VkDescriptorPoolSize, 2> poolSizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxDescGroups * (4 * 2)},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1 + kMaxDescGroups * (4 * 1 + 2 * 1)},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1 + kMaxDescGroups * 6;
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
        vkUpdateDescriptorSets(ctx_.device(), 1, &wT, 0, nullptr);

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
        if (groups_.size() >= kMaxDescGroups) {
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
            vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
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
            vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        };
        writePermute(g.p[0], aStorage);
        writePermute(g.p[1], bStorage);

        groups_.push_back(g);
        return groups_.back();
    }

    void IFFT::recordApply(VkCommandBuffer cb, OceanImage& input, OceanImage& scratch) {
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
        if (pipeTwiddle_    != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipeTwiddle_, nullptr);
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
        destroyBuffer(ctx_, paramsUbo_);
        destroyImage(ctx_, twiddle_);
    }

}// namespace threepp::water
