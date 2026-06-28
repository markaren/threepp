// FFT-based ocean primitive implementations. See OceanFFT.hpp for the
// public surface and architecture overview.

#include "threepp/renderers/vulkan/water/OceanFFT.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/phillips_spectrum.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/dynamic_spectrum.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ifft_twiddle.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ifft_horizontal.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ifft_vertical.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/ifft_permute.comp.spv.h"

#include <array>
#include <cmath>
#include <cstring>
#include <random>
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

        // Params UBO (64 bytes — 7 payload fields + 1 pad, doubled for alignment)
        paramsUbo_ = makeUbo(ctx_, 64);
        writeParams();

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
        std::mt19937 rng(0xC0FFEEu);
        std::uniform_real_distribution<float> uni(0.f, 1.f);
        auto gauss = [&] {
            const float u1 = std::max(uni(rng), 1e-6f);
            const float u2 = uni(rng);
            return std::cos(2.f * 3.14159265f * u2) * std::sqrt(-2.f * std::log(u1));
        };
        for (uint32_t i = 0; i < N * N; ++i) {
            data[2 * i + 0] = gauss();
            data[2 * i + 1] = gauss();
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

    void PhillipsSpectrum::writeParams() {
        struct {
            uint32_t textureSize;
            float    tileSize;
            float    windTheta;
            float    windSpeed;
            float    smallWaveCutoff;
            float    kMin;
            float    kMax;
            float    _pad;
        } p{};
        p.textureSize     = settings_.textureSize;
        p.tileSize        = settings_.tileSize;
        p.windTheta       = settings_.windTheta;
        p.windSpeed       = settings_.windSpeed;
        p.smallWaveCutoff = settings_.smallWaveCutoff;
        p.kMin            = settings_.kMin;
        p.kMax            = settings_.kMax;
        std::memcpy(paramsUbo_.mapped, &p, sizeof(p));
    }

    void PhillipsSpectrum::createPipeline() {
        // Bindings: 0 = h0 (storage image, write), 1 = noise (sampled), 2 = params UBO
        const std::array<VkDescriptorSetLayoutBinding, 3> bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsl_),
              "vkCreateDescriptorSetLayout(phillips)");

        VkPipelineLayoutCreateInfo plci{};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &dsl_;
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
        // We also need a UBO entry — extend the pool sizes with one. Combined here
        // for one-shot allocation.
        std::array<VkDescriptorPoolSize, 3> poolSizesAll{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1},
        };
        dpci.poolSizeCount = static_cast<uint32_t>(poolSizesAll.size());
        dpci.pPoolSizes    = poolSizesAll.data();
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

        VkDescriptorBufferInfo paramInfo{};
        paramInfo.buffer = paramsUbo_.handle;
        paramInfo.range  = paramsUbo_.size;

        std::array<VkWriteDescriptorSet, 3> writes{};
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

        writes[2].sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet     = ds_;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &paramInfo;

        vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void PhillipsSpectrum::recordCompute(VkCommandBuffer cb) {
        cmdTransitionToGeneral(cb, h0_);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                                0, 1, &ds_, 0, nullptr);
        const uint32_t g = groupCountFor(settings_.textureSize);
        vkCmdDispatch(cb, g, g, 1);
        // Emit a write→read barrier so subsequent samples see the data.
        cmdShaderRWBarrier(cb, h0_.image);
    }

    PhillipsSpectrum::~PhillipsSpectrum() {
        if (pipe_   != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipe_, nullptr);
        if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layout_, nullptr);
        if (dsl_    != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dsl_, nullptr);
        if (pool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), pool_, nullptr);
        if (sampler_!= VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
        destroyBuffer(ctx_, paramsUbo_);
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
        paramsUbo_ = makeUbo(ctx_, 16);
        createPipeline();
    }

    void DynamicSpectrum::createImages() {
        ht_           = makeStorageSampledImage(ctx_, textureSize_, textureSize_, VK_FORMAT_R32G32_SFLOAT, "ocean.dyn.ht");
        dht_          = makeStorageSampledImage(ctx_, textureSize_, textureSize_, VK_FORMAT_R32G32_SFLOAT, "ocean.dyn.dht");
        displacement_ = makeStorageSampledImage(ctx_, textureSize_, textureSize_, VK_FORMAT_R32G32_SFLOAT, "ocean.dyn.displacement");
        jacDiag_      = makeStorageSampledImage(ctx_, textureSize_, textureSize_, VK_FORMAT_R32G32_SFLOAT, "ocean.dyn.jacDiag");
    }

    void DynamicSpectrum::createPipeline() {
        const std::array<VkDescriptorSetLayoutBinding, 6> bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // H0
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // HT
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // DHT
            VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Displacement
            VkDescriptorSetLayoutBinding{4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Params
            VkDescriptorSetLayoutBinding{5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // JacDiag
        };
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(bindings.size());
        dlci.pBindings    = bindings.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsl_),
              "vkCreateDescriptorSetLayout(dyn)");

        VkPipelineLayoutCreateInfo plci{};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &dsl_;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &layout_),
              "vkCreatePipelineLayout(dyn)");

        VkShaderModule mod = makeShader(ctx_, kDynamicSpectrumCompSpv, sizeof(kDynamicSpectrumCompSpv));
        pipe_ = makeComputePipeline(ctx_, mod, layout_);
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        const std::array<VkDescriptorPoolSize, 3> poolSizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          4},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1},
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

        VkDescriptorImageInfo dhtInfo{};
        dhtInfo.imageView   = dht_.view;
        dhtInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo dispInfo{};
        dispInfo.imageView   = displacement_.view;
        dispInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo jacInfo{};
        jacInfo.imageView   = jacDiag_.view;
        jacInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo pInfo{};
        pInfo.buffer = paramsUbo_.handle;
        pInfo.range  = paramsUbo_.size;

        std::array<VkWriteDescriptorSet, 6> writes{};
        for (auto& w : writes) {
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = ds_;
            w.descriptorCount = 1;
        }
        writes[0].dstBinding = 0; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[0].pImageInfo = &h0Info;
        writes[1].dstBinding = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          writes[1].pImageInfo = &htInfo;
        writes[2].dstBinding = 2; writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          writes[2].pImageInfo = &dhtInfo;
        writes[3].dstBinding = 3; writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          writes[3].pImageInfo = &dispInfo;
        writes[4].dstBinding = 4; writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         writes[4].pBufferInfo = &pInfo;
        writes[5].dstBinding = 5; writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          writes[5].pImageInfo = &jacInfo;
        vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void DynamicSpectrum::recordCompute(VkCommandBuffer cb, float elapsedSeconds) {
        struct {
            uint32_t textureSize;
            float    tileSize;
            float    elapsedSeconds;
            float    _pad;
        } p{};
        p.textureSize    = textureSize_;
        p.tileSize       = tileSize_;
        p.elapsedSeconds = elapsedSeconds;
        std::memcpy(paramsUbo_.mapped, &p, sizeof(p));

        cmdTransitionToGeneral(cb, ht_);
        cmdTransitionToGeneral(cb, dht_);
        cmdTransitionToGeneral(cb, displacement_);
        cmdTransitionToGeneral(cb, jacDiag_);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                                0, 1, &ds_, 0, nullptr);
        const uint32_t g = groupCountFor(textureSize_);
        vkCmdDispatch(cb, g, g, 1);

        cmdShaderRWBarrier(cb, ht_.image);
        cmdShaderRWBarrier(cb, dht_.image);
        cmdShaderRWBarrier(cb, displacement_.image);
        cmdShaderRWBarrier(cb, jacDiag_.image);
    }

    DynamicSpectrum::~DynamicSpectrum() {
        if (pipe_   != VK_NULL_HANDLE) vkDestroyPipeline(ctx_.device(), pipe_, nullptr);
        if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_.device(), layout_, nullptr);
        if (dsl_    != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_.device(), dsl_, nullptr);
        if (pool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx_.device(), pool_, nullptr);
        if (sampler_!= VK_NULL_HANDLE) vkDestroySampler(ctx_.device(), sampler_, nullptr);
        destroyBuffer(ctx_, paramsUbo_);
        destroyImage(ctx_, ht_);
        destroyImage(ctx_, dht_);
        destroyImage(ctx_, displacement_);
        destroyImage(ctx_, jacDiag_);
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

        // ── Permute layout: 1 sampled + 1 storage (no PC)
        const std::array<VkDescriptorSetLayoutBinding, 2> pbb{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
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

        // ── Pool: enough for 1 twiddle + 4 butterfly (2 horizontal + 2 vertical) +
        // 2 permute (one per direction) descriptor sets.
        const std::array<VkDescriptorPoolSize, 2> poolSizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * 2 + 2 * 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1 + 4 * 1 + 2 * 1},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1 + 4 + 2;
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

        // Pre-allocate 2 horizontal + 2 vertical butterfly DSes. We bind images
        // lazily in rebindDescriptorSets when the caller hands us new image pair.
        dsHorizontal_.resize(2);
        VkDescriptorSetAllocateInfo daiB{};
        daiB.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        daiB.descriptorPool     = pool_;
        daiB.descriptorSetCount = 1;
        daiB.pSetLayouts        = &dslButterfly_;
        for (auto& ds : dsHorizontal_) {
            check(vkAllocateDescriptorSets(ctx_.device(), &daiB, &ds),
                  "vkAllocateDescriptorSets(butterflyH)");
        }
        dsVertical_.resize(2);
        for (auto& ds : dsVertical_) {
            check(vkAllocateDescriptorSets(ctx_.device(), &daiB, &ds),
                  "vkAllocateDescriptorSets(butterflyV)");
        }

        VkDescriptorSetAllocateInfo daiP{};
        daiP.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        daiP.descriptorPool     = pool_;
        daiP.descriptorSetCount = 1;
        daiP.pSetLayouts        = &dslPermute_;
        for (auto& ds : dsPermute_) {
            check(vkAllocateDescriptorSets(ctx_.device(), &daiP, &ds),
                  "vkAllocateDescriptorSets(permute)");
        }
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

    void IFFT::rebindDescriptorSets(OceanImage& a, OceanImage& b) {
        // Two horizontal sets:
        //   [0] reads a, writes b (called when pingPong is true on iter 0)
        //   [1] reads b, writes a
        // Same pattern for vertical.
        // Permute:
        //   [0] reads (image holding result before permute), writes (other one)
        // We don't know yet which holds the result; that's resolved at record time.
        struct ImgBinding {
            VkSampler sampler; VkImageView view; VkImageLayout layout;
        };

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
        writeButterfly(dsHorizontal_[0], aSampled, bStorage);
        writeButterfly(dsHorizontal_[1], bSampled, aStorage);
        writeButterfly(dsVertical_[0],   aSampled, bStorage);
        writeButterfly(dsVertical_[1],   bSampled, aStorage);

        auto writePermute = [&](VkDescriptorSet ds, const VkDescriptorImageInfo& read, const VkDescriptorImageInfo& write) {
            std::array<VkWriteDescriptorSet, 2> w{};
            for (auto& e : w) { e.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; e.dstSet = ds; e.descriptorCount = 1; }
            w[0].dstBinding = 0; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &read;
            w[1].dstBinding = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[1].pImageInfo = &write;
            vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
        };
        // dsPermute_[0]: read a, write b
        // dsPermute_[1]: read b, write a
        writePermute(dsPermute_[0], aSampled, bStorage);
        writePermute(dsPermute_[1], bSampled, aStorage);

        prevInput_   = &a;
        prevScratch_ = &b;
    }

    void IFFT::recordApply(VkCommandBuffer cb, OceanImage& input, OceanImage& scratch) {
        if (!twiddleComputed_) recordTwiddleOnce(cb);

        if (prevInput_ != &input || prevScratch_ != &scratch) {
            rebindDescriptorSets(input, scratch);
        }

        cmdTransitionToGeneral(cb, input);
        cmdTransitionToGeneral(cb, scratch);

        // pingPong: starts at false; flipped before each pass in the WGSL ref.
        // We model the same: iter 0 → write to scratch (read input), iter 1 →
        // write to input (read scratch), and so on.
        bool pingPong = false;

        // Horizontal passes
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeHorizontal_);
        for (uint32_t step = 0; step < logSize_; ++step) {
            pingPong = !pingPong;
            VkDescriptorSet ds = pingPong ? dsHorizontal_[0] : dsHorizontal_[1];
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
            VkDescriptorSet ds = pingPong ? dsVertical_[0] : dsVertical_[1];
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layoutButterfly_,
                                    0, 1, &ds, 0, nullptr);
            const int32_t s = static_cast<int32_t>(step);
            vkCmdPushConstants(cb, layoutButterfly_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(s), &s);
            const uint32_t g = groupCountFor(textureSize_);
            vkCmdDispatch(cb, g, g, 1);
            cmdShaderRWBarrier(cb, pingPong ? scratch.image : input.image);
        }

        // Permute. After 2*logSize butterfly passes, pingPong is back to its
        // starting parity (false for even logSize). The result lives in `input`
        // when pingPong is false, in `scratch` when true.
        const bool resultInScratch = pingPong;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipePermute_);
        // dsPermute_[0]: read a (=input), write b (=scratch)
        // dsPermute_[1]: read b (=scratch), write a (=input)
        VkDescriptorSet dsP = resultInScratch ? dsPermute_[1] : dsPermute_[0];
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layoutPermute_,
                                0, 1, &dsP, 0, nullptr);
        const uint32_t g = groupCountFor(textureSize_);
        vkCmdDispatch(cb, g, g, 1);

        // Final barrier — caller reads the result.
        cmdShaderRWBarrier(cb, resultInScratch ? input.image : scratch.image);

        // The permute writes to whichever image was NOT holding the result.
        // We want the spatial-domain result in `input` (caller's contract).
        // If permute wrote to scratch, copy scratch → input.
        if (!resultInScratch) {
            // permute wrote to scratch (since result was in input). Copy back.
            VkImageCopy copy{};
            copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.extent         = {textureSize_, textureSize_, 1};
            // Both already in GENERAL — vkCmdCopyImage requires TRANSFER_SRC/DST.
            // Use a barrier sandwich to flip layouts for the copy then restore.
            VkImageMemoryBarrier toSrc{};
            toSrc.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toSrc.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            toSrc.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toSrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            toSrc.image         = scratch.image;
            toSrc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            VkImageMemoryBarrier toDst{};
            toDst.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toDst.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            toDst.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toDst.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toDst.image         = input.image;
            toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            VkImageMemoryBarrier flip[2] = {toSrc, toDst};
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 2, flip);

            vkCmdCopyImage(cb,
                scratch.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                input.image,   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &copy);

            // Back to GENERAL for downstream samplers.
            VkImageMemoryBarrier back[2]{};
            back[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            back[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            back[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            back[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            back[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            back[0].image = scratch.image;
            back[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            back[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            back[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            back[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            back[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            back[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            back[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            back[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            back[1].image = input.image;
            back[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            back[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            back[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 2, back);
        }
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
