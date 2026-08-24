#include "threepp/renderers/vulkan/LidarScanner.hpp"

#include "threepp/renderers/vulkan/ParticleFieldPass.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/lidar.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/lidar.rchit.spv.h"
#include "threepp/renderers/vulkan/shaders/lidar.rmiss.spv.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace threepp::vulkan {

    namespace {

        // Round up to the next power of two so buffer resizes amortise.
        uint32_t roundUpPow2(uint32_t v) {
            if (v == 0) return 1;
            --v;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            return v + 1;
        }

    }// namespace

    static_assert(LidarScanner::kDensityVolumes == kMaxDensityFields,
                  "LIDAR density array must match the renderer's volume cap.");

    LidarScanner::LidarScanner(VulkanContext& ctx) : ctx_(ctx) {
        createDescriptorLayout();
        createPipeline();
        createSbt();
        createCommandObjects();

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW =
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &densitySampler_),
              "vkCreateSampler(lidar density)");

        // The stand-in majorant array. Host-visible and written once: 16 bytes
        // that are never read by any shader (counts.x is 0 whenever this is
        // bound) but must be a valid descriptor all the same.
        const std::array<uint32_t, kDensityVolumes> zeros{};
        majorantFallback_ = createBuffer(
                ctx_.allocator(), ctx_.device(), sizeof(zeros),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        uploadHostVisible(ctx_.allocator(), majorantFallback_, zeros.data(), sizeof(zeros));
    }

    LidarScanner::~LidarScanner() {
        VkDevice d = ctx_.device();
        // A dispatch nobody collected is still reading its beam buffer and
        // writing its result buffer. Let it finish before either is freed.
        for (auto& s : slots_) {
            if (s.submitted && s.fence) vkWaitForFences(d, 1, &s.fence, VK_TRUE, UINT64_MAX);
            if (s.fence) vkDestroyFence(d, s.fence, nullptr);
            destroyBuffer(ctx_.allocator(), s.readbackBuf);
            destroyBuffer(ctx_.allocator(), s.resultBuf);
            destroyBuffer(ctx_.allocator(), s.beamBuf);
        }
        if (cmdPool_)  vkDestroyCommandPool(d, cmdPool_, nullptr);
        if (densitySampler_) vkDestroySampler(d, densitySampler_, nullptr);
        destroyBuffer(ctx_.allocator(), majorantFallback_);
        destroyBuffer(ctx_.allocator(), sbtBuf_);
        if (pipeline_) vkDestroyPipeline(d, pipeline_, nullptr);
        if (descPool_) vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (descSetLayout_)  vkDestroyDescriptorSetLayout(d, descSetLayout_, nullptr);
    }

    void LidarScanner::createDescriptorLayout() {
        // Six bindings:
        //   0 = TLAS (acceleration structure)
        //   1 = GeomDescBuf  (SSBO, read)
        //   2 = MatDescBuf   (SSBO, read)
        //   3 = BeamBuf      (SSBO, read)
        //   4 = ResultBuf    (SSBO, write)
        //   5 = FogUbo       (uniform, read — shared with main RT's GpuFogUbo)
        //   6 = PdMajorantBuf (SSBO, read — per-volume delta-tracking bounds)
        //  67 = particle density volumes (r32ui, sampled)   ┐ the deferred
        //  68 = ParticleDensityUbo (boxes + medium params)  ┘ set's numbers,
        // reused verbatim so shaders/particle_density.glsl is included here
        // UNMODIFIED — one medium model, one header, two consumers.
        std::array<VkDescriptorSetLayoutBinding, 9> b{};
        b[0].binding = 0;
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        b[1].binding = 1;
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        b[2].binding = 2;
        b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[2].descriptorCount = 1;
        b[2].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        b[3].binding = 3;
        b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[3].descriptorCount = 1;
        b[3].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[4].binding = 4;
        b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[4].descriptorCount = 1;
        b[4].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[5].binding = 5;
        b[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[5].descriptorCount = 1;
        b[5].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[6].binding = 6;
        b[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[6].descriptorCount = 1;
        b[6].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[7].binding = 67;
        b[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[7].descriptorCount = kDensityVolumes;
        b[7].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[8].binding = 68;
        b[8].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[8].descriptorCount = 1;
        b[8].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(b.size());
        dlci.pBindings = b.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &descSetLayout_),
              "vkCreateDescriptorSetLayout(lidar)");

        // Pipeline layout: 1 descriptor set, 1 push constant range covering
        // both raygen and closest-hit.
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                         VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                         VK_SHADER_STAGE_MISS_BIT_KHR;
        pcr.offset = 0;
        pcr.size = sizeof(vulkan_lidar::LidarPushConstants);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &descSetLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(lidar)");

        // One set per scan slot: a set referenced by a pending submit must not
        // be updated (VUID-vkUpdateDescriptorSets-None-03047), and every
        // dispatch rewrites the TLAS + desc bindings.
        std::array<VkDescriptorPoolSize, 4> ps{};
        ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        ps[0].descriptorCount = kScanSlots;
        ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[1].descriptorCount = 5 * kScanSlots;
        ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps[2].descriptorCount = 2 * kScanSlots;
        ps[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[3].descriptorCount = kDensityVolumes * kScanSlots;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = kScanSlots;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
        dpci.pPoolSizes = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(lidar)");

        std::array<VkDescriptorSetLayout, kScanSlots> layouts{};
        layouts.fill(descSetLayout_);
        std::array<VkDescriptorSet, kScanSlots> sets{};
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = descPool_;
        dsai.descriptorSetCount = kScanSlots;
        dsai.pSetLayouts = layouts.data();
        check(vkAllocateDescriptorSets(ctx_.device(), &dsai, sets.data()),
              "vkAllocateDescriptorSets(lidar)");
        for (uint32_t i = 0; i < kScanSlots; ++i) slots_[i].descSet = sets[i];
    }

    void LidarScanner::createPipeline() {
        auto loadModule = [this](const uint32_t* code, size_t size) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = size;
            smci.pCode    = code;
            VkShaderModule m = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &m),
                  "vkCreateShaderModule(lidar)");
            return m;
        };
        VkShaderModule rgenMod = loadModule(kLidarRgenSpv,  sizeof(kLidarRgenSpv));
        VkShaderModule missMod = loadModule(kLidarRmissSpv, sizeof(kLidarRmissSpv));
        VkShaderModule chitMod = loadModule(kLidarRchitSpv, sizeof(kLidarRchitSpv));

        std::array<VkPipelineShaderStageCreateInfo, 3> stg{};
        for (auto& s : stg) {
            s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            s.pName = "main";
        }
        stg[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;      stg[0].module = rgenMod;
        stg[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;        stg[1].module = missMod;
        stg[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; stg[2].module = chitMod;

        std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> grp{};
        for (auto& g : grp) {
            g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.generalShader = g.closestHitShader = g.anyHitShader = g.intersectionShader = VK_SHADER_UNUSED_KHR;
        }
        grp[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;             grp[0].generalShader    = 0;
        grp[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;             grp[1].generalShader    = 1;
        grp[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR; grp[2].closestHitShader = 2;

        VkRayTracingPipelineCreateInfoKHR rci{};
        rci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rci.stageCount = static_cast<uint32_t>(stg.size());
        rci.pStages    = stg.data();
        rci.groupCount = static_cast<uint32_t>(grp.size());
        rci.pGroups    = grp.data();
        rci.maxPipelineRayRecursionDepth = 1;
        rci.layout = pipelineLayout_;

        check(ctx_.rt().createRayTracingPipelines(
                      ctx_.device(), VK_NULL_HANDLE, VK_NULL_HANDLE,
                      1, &rci, nullptr, &pipeline_),
              "vkCreateRayTracingPipelinesKHR(lidar)");

        vkDestroyShaderModule(ctx_.device(), rgenMod, nullptr);
        vkDestroyShaderModule(ctx_.device(), missMod, nullptr);
        vkDestroyShaderModule(ctx_.device(), chitMod, nullptr);
    }

    void LidarScanner::createSbt() {
        const auto& props = ctx_.rtPipelineProperties();
        const uint32_t handleSize        = props.shaderGroupHandleSize;
        const uint32_t handleAlignment   = props.shaderGroupHandleAlignment;
        const uint32_t baseAlignment     = props.shaderGroupBaseAlignment;
        const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

        constexpr uint32_t groupCount = 3;  // rgen, miss, hit
        const uint32_t handlesSize = groupCount * handleSize;
        std::vector<uint8_t> handles(handlesSize);
        check(ctx_.rt().getRayTracingShaderGroupHandles(
                      ctx_.device(), pipeline_, 0, groupCount,
                      handlesSize, handles.data()),
              "vkGetRayTracingShaderGroupHandlesKHR(lidar)");

        const uint32_t rgenBytes = alignUp(handleSizeAligned, baseAlignment);
        const uint32_t missBytes = alignUp(handleSizeAligned, baseAlignment);
        const uint32_t hitBytes  = alignUp(handleSizeAligned, baseAlignment);
        const VkDeviceSize sbtSize =
                static_cast<VkDeviceSize>(rgenBytes) +
                static_cast<VkDeviceSize>(missBytes) +
                static_cast<VkDeviceSize>(hitBytes);

        sbtBuf_ = createBuffer(
                ctx_.allocator(), ctx_.device(), sbtSize,
                VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        void* mapped = nullptr;
        vmaMapMemory(ctx_.allocator(), sbtBuf_.alloc, &mapped);
        std::memset(mapped, 0, sbtSize);
        uint8_t* dst = static_cast<uint8_t*>(mapped);
        std::memcpy(dst,                         handles.data() + 0 * handleSize, handleSize);
        std::memcpy(dst + rgenBytes,             handles.data() + 1 * handleSize, handleSize);
        std::memcpy(dst + rgenBytes + missBytes, handles.data() + 2 * handleSize, handleSize);
        flushHostWrites(ctx_.allocator(), sbtBuf_.alloc, 0, sbtSize);
        vmaUnmapMemory(ctx_.allocator(), sbtBuf_.alloc);

        const VkDeviceAddress base = sbtBuf_.address;
        rgenRgn_.deviceAddress = base;
        rgenRgn_.stride        = rgenBytes;
        rgenRgn_.size          = rgenBytes;
        missRgn_.deviceAddress = base + rgenBytes;
        missRgn_.stride        = handleSizeAligned;
        missRgn_.size          = missBytes;
        hitRgn_.deviceAddress  = base + rgenBytes + missBytes;
        hitRgn_.stride         = handleSizeAligned;
        hitRgn_.size           = hitBytes;
        callRgn_ = {};
    }

    void LidarScanner::createCommandObjects() {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = ctx_.queueFamilies().graphics;
        check(vkCreateCommandPool(ctx_.device(), &cpci, nullptr, &cmdPool_),
              "vkCreateCommandPool(lidar)");

        std::array<VkCommandBuffer, kScanSlots> cbs{};
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cmdPool_;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = kScanSlots;
        check(vkAllocateCommandBuffers(ctx_.device(), &cbai, cbs.data()),
              "vkAllocateCommandBuffers(lidar)");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        for (uint32_t i = 0; i < kScanSlots; ++i) {
            slots_[i].cmdBuf = cbs[i];
            check(vkCreateFence(ctx_.device(), &fci, nullptr, &slots_[i].fence),
                  "vkCreateFence(lidar)");
        }
    }

    void LidarScanner::ensureCapacity(Slot& slot, uint32_t numBeams, uint32_t slotsPerBeam) {
        const uint32_t beamsNeeded   = roundUpPow2(std::max(1u, numBeams));
        const uint32_t resultsNeeded = roundUpPow2(std::max(1u, numBeams * std::max(1u, slotsPerBeam)));
        const bool beamsGrew   = (beamsNeeded   > slot.capacityBeams);
        const bool resultsGrew = (resultsNeeded > slot.capacityResults);
        if (!beamsGrew && !resultsGrew) return;

        // Safe to free: dispatch reaches here only with a slot that is either
        // free or reclaimed, and a slot is reclaimed only once its fence has
        // signaled — so nothing in flight is reading these buffers.
        if (beamsGrew) {
            destroyBuffer(ctx_.allocator(), slot.beamBuf);
            const VkDeviceSize beamBytes =
                    static_cast<VkDeviceSize>(beamsNeeded) * sizeof(vulkan_lidar::LidarBeam);
            slot.beamBuf = createBuffer(
                    ctx_.allocator(), ctx_.device(), beamBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            slot.capacityBeams = beamsNeeded;
        }

        if (resultsGrew) {
            destroyBuffer(ctx_.allocator(), slot.resultBuf);
            destroyBuffer(ctx_.allocator(), slot.readbackBuf);
            const VkDeviceSize resultBytes =
                    static_cast<VkDeviceSize>(resultsNeeded) * sizeof(vulkan_lidar::LidarResult);
            slot.resultBuf = createBuffer(
                    ctx_.allocator(), ctx_.device(), resultBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    0);
            slot.readbackBuf = createBuffer(
                    ctx_.allocator(), ctx_.device(), resultBytes,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            slot.capacityResults = resultsNeeded;
        }

        // Update descriptor set bindings 3 (beams) and 4 (results) to point
        // at the new buffers. The scene bindings (0/1/2) are touched per
        // scan in updateSceneBindings.
        VkDescriptorBufferInfo beamInfo{};
        beamInfo.buffer = slot.beamBuf.handle;
        beamInfo.offset = 0;
        beamInfo.range  = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo resultInfo{};
        resultInfo.buffer = slot.resultBuf.handle;
        resultInfo.offset = 0;
        resultInfo.range  = VK_WHOLE_SIZE;

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = slot.descSet;
        w[0].dstBinding = 3;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[0].pBufferInfo = &beamInfo;

        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = slot.descSet;
        w[1].dstBinding = 4;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo = &resultInfo;

        vkUpdateDescriptorSets(ctx_.device(),
                               static_cast<uint32_t>(w.size()), w.data(),
                               0, nullptr);
    }

    void LidarScanner::updateSceneBindings(Slot& slot,
                                           VkAccelerationStructureKHR tlas,
                                           VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                                           VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                                           VkBuffer fogUbo, VkDeviceSize fogUboSize,
                                           const DensityBinding& density) {
        VkWriteDescriptorSetAccelerationStructureKHR asi{};
        asi.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        asi.accelerationStructureCount = 1;
        asi.pAccelerationStructures = &tlas;

        VkDescriptorBufferInfo geomInfo{};
        geomInfo.buffer = geomDescsBuffer;
        geomInfo.offset = 0;
        geomInfo.range  = geomDescsSize ? geomDescsSize : VK_WHOLE_SIZE;

        VkDescriptorBufferInfo matInfo{};
        matInfo.buffer = matDescsBuffer;
        matInfo.offset = 0;
        matInfo.range  = matDescsSize ? matDescsSize : VK_WHOLE_SIZE;

        VkDescriptorBufferInfo fogInfo{};
        fogInfo.buffer = fogUbo;
        fogInfo.offset = 0;
        fogInfo.range  = fogUboSize ? fogUboSize : VK_WHOLE_SIZE;

        // ── ParticleField density (parent plan phase 3) ─────────────────────
        // Every slot filled, live volumes first and the caller's dummy behind
        // them, exactly as the deferred set does it — a set with a hole in it
        // is a validation error the moment it is bound, whether or not the
        // shader reads that slot.
        std::array<VkDescriptorImageInfo, kDensityVolumes> pdInfos{};
        for (uint32_t i = 0; i < kDensityVolumes; ++i) {
            pdInfos[i].sampler     = densitySampler_;
            pdInfos[i].imageView   = (density.views && i < density.viewCount)
                                             ? density.views[i]
                                             : VK_NULL_HANDLE;
            pdInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        VkDescriptorBufferInfo pdUboInfo{};
        pdUboInfo.buffer = density.ubo;
        pdUboInfo.offset = 0;
        pdUboInfo.range  = density.uboSize ? density.uboSize : VK_WHOLE_SIZE;

        VkDescriptorBufferInfo majInfo{};
        majInfo.buffer = density.majorants ? density.majorants : majorantFallback_.handle;
        majInfo.offset = 0;
        majInfo.range  = (density.majorants && density.majorantsSize)
                                 ? density.majorantsSize
                                 : VK_WHOLE_SIZE;

        std::array<VkWriteDescriptorSet, 7> w{};
        w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].pNext           = &asi;
        w[0].dstSet          = slot.descSet;
        w[0].dstBinding      = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet          = slot.descSet;
        w[1].dstBinding      = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo     = &geomInfo;

        w[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[2].dstSet          = slot.descSet;
        w[2].dstBinding      = 2;
        w[2].descriptorCount = 1;
        w[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[2].pBufferInfo     = &matInfo;

        w[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[3].dstSet          = slot.descSet;
        w[3].dstBinding      = 5;
        w[3].descriptorCount = 1;
        w[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[3].pBufferInfo     = &fogInfo;

        w[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[4].dstSet          = slot.descSet;
        w[4].dstBinding      = 6;
        w[4].descriptorCount = 1;
        w[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[4].pBufferInfo     = &majInfo;

        w[5].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[5].dstSet          = slot.descSet;
        w[5].dstBinding      = 67;
        w[5].descriptorCount = kDensityVolumes;
        w[5].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[5].pImageInfo      = pdInfos.data();

        w[6].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[6].dstSet          = slot.descSet;
        w[6].dstBinding      = 68;
        w[6].descriptorCount = 1;
        w[6].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[6].pBufferInfo     = &pdUboInfo;

        vkUpdateDescriptorSets(ctx_.device(),
                               static_cast<uint32_t>(w.size()), w.data(),
                               0, nullptr);
    }

    void LidarScanner::scan(VkQueue queue,
                            VkAccelerationStructureKHR tlas,
                            VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                            VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                            VkBuffer fogUbo, VkDeviceSize fogUboSize,
                            const DensityBinding& density,
                            const vulkan_lidar::LidarPushConstants& pc,
                            const vulkan_lidar::LidarBeam* beams, uint32_t numBeams,
                            vulkan_lidar::LidarResult* outResults) {

        if (outResults == nullptr) return;
        // The synchronous contract, expressed in terms of the pipelined one:
        // fire and take delivery immediately. `collect` writes the all-miss
        // result when `dispatch` declined to submit, so an unbuilt scene reads
        // the same here as it always did.
        const int slot = dispatch(queue, tlas, geomDescsBuffer, geomDescsSize,
                                  matDescsBuffer, matDescsSize,
                                  fogUbo, fogUboSize, density, pc, beams, numBeams);
        collect(slot, outResults);
    }

    int LidarScanner::dispatch(VkQueue queue,
                               VkAccelerationStructureKHR tlas,
                               VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                               VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                               VkBuffer fogUbo, VkDeviceSize fogUboSize,
                               const DensityBinding& density,
                               const vulkan_lidar::LidarPushConstants& pc,
                               const vulkan_lidar::LidarBeam* beams, uint32_t numBeams) {

        if (numBeams == 0 || beams == nullptr) return kNoSlot;

        uint32_t index = kScanSlots;
        for (uint32_t i = 0; i < kScanSlots; ++i) {
            if (!slots_[i].reserved) { index = i; break; }
        }
        if (index == kScanSlots) {
            // Every slot is reserved. Reclaim the stalest one that has actually
            // finished on the GPU — see the header: a sensor destroyed between
            // firing and collecting abandons its slot, and four of those would
            // wedge the scanner permanently. Bumping the generation below kills
            // the abandoned handle rather than aliasing it onto this scan.
            uint64_t oldest = UINT64_MAX;
            for (uint32_t i = 0; i < kScanSlots; ++i) {
                const Slot& c = slots_[i];
                const bool finished = !c.submitted ||
                                      vkGetFenceStatus(ctx_.device(), c.fence) == VK_SUCCESS;
                if (finished && c.issued < oldest) {
                    oldest = c.issued;
                    index  = i;
                }
            }
            if (index == kScanSlots) return kNoSlot;// all still running
        }
        Slot& s = slots_[index];
        s.gen = (s.gen + 1) & 0x3FFFFFFu;
        s.issued = ++dispatchCounter_;
        const int handle = static_cast<int>(index) | static_cast<int>(s.gen << kHandleShift);

        const uint32_t maxReturns   = std::max(pc.maxReturns, 1u);
        const uint32_t samples      = std::max(pc.samplesPerBeam, 1u);
        const uint32_t slotsPerBeam = maxReturns * samples;
        // The paired trace doubles the launch AND the result rows: the clean
        // leg occupies the second half, at the same stride, so the caller can
        // difference row i against row i + numBeams * slotsPerBeam.
        const uint32_t legs         = ((pc.flags & vulkan_lidar::kLidarFlagPairedClean) != 0u) ? 2u : 1u;
        const uint32_t totalSlots   = numBeams * slotsPerBeam * legs;

        // Reserve first: collect() owes the caller a result of the right length
        // whether or not anything was traced.
        s.reserved  = true;
        s.submitted = false;
        s.slots     = totalSlots;

        // The density set members are on this list because the rgen STATICALLY
        // reads binding 68 (pd.counts) and every slot of 67, whatever the scene
        // holds — an incomplete set there is a validation error at bind time,
        // not a quiet zero. A caller that has not built them yet gets the same
        // all-miss result an unbuilt TLAS gets.
        const bool sceneReady = (tlas != VK_NULL_HANDLE) &&
                                (geomDescsBuffer != VK_NULL_HANDLE) &&
                                (matDescsBuffer != VK_NULL_HANDLE) &&
                                (fogUbo != VK_NULL_HANDLE) &&
                                (density.ubo != VK_NULL_HANDLE) &&
                                (density.views != nullptr) &&
                                (density.viewCount >= kDensityVolumes);
        if (!sceneReady) return handle;

        // Sized by the result count, which is what the leg factor multiplies.
        // The beam buffer comes out one leg too large and is uploaded with
        // numBeams rows regardless — the clean leg reads the same beams (that
        // is what pairing means), so there is nothing extra to send.
        ensureCapacity(s, numBeams * legs, slotsPerBeam);

        // Upload beams (mapped, sequential write + flush).
        uploadHostVisible(ctx_.allocator(), s.beamBuf, beams,
                          numBeams * sizeof(vulkan_lidar::LidarBeam));

        updateSceneBindings(s, tlas, geomDescsBuffer, geomDescsSize,
                            matDescsBuffer, matDescsSize,
                            fogUbo, fogUboSize, density);

        VkCommandBuffer cmd = s.cmdBuf;

        // Reset cmd buffer + fence and re-record.
        check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer(lidar)");
        check(vkResetFences(ctx_.device(), 1, &s.fence), "vkResetFences(lidar)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer(lidar)");

        // The trace reads the renderer's TLAS, which an in-flight frame may
        // still be building. This barrier is what makes that safe WITHOUT a CPU
        // wait: the first synchronization scope of a barrier covers everything
        // submitted earlier on the same queue, so the acceleration-structure
        // build of any frame ahead of us has completed before the rgen runs.
        // Before this existed the caller reached for vkDeviceWaitIdle, which is
        // correct and costs every queued frame — the ~28 ms hitch this class's
        // header talks about.
        VkMemoryBarrier2 asToTrace{};
        asToTrace.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        asToTrace.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        asToTrace.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                                  VK_ACCESS_2_SHADER_WRITE_BIT |
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT;
        asToTrace.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        asToTrace.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                  VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo asDep{};
        asDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        asDep.memoryBarrierCount = 1;
        asDep.pMemoryBarriers = &asToTrace;
        vkCmdPipelineBarrier2(cmd, &asDep);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                pipelineLayout_, 0, 1, &s.descSet, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout_,
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                   VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                   VK_SHADER_STAGE_MISS_BIT_KHR,
                           0, sizeof(pc), &pc);
        ctx_.rt().cmdTraceRays(cmd, &rgenRgn_, &missRgn_, &hitRgn_, &callRgn_,
                                numBeams * legs, 1, 1);

        // Barrier: RT shader writes to resultBuf_ → transfer read.
        VkBufferMemoryBarrier2 rtToCopy{};
        rtToCopy.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        rtToCopy.srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        rtToCopy.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        rtToCopy.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        rtToCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        rtToCopy.srcQueueFamilyIndex = rtToCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rtToCopy.buffer = s.resultBuf.handle;
        rtToCopy.size   = VK_WHOLE_SIZE;
        VkDependencyInfo rtDep{};
        rtDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        rtDep.bufferMemoryBarrierCount = 1;
        rtDep.pBufferMemoryBarriers = &rtToCopy;
        vkCmdPipelineBarrier2(cmd, &rtDep);

        // Copy device → host-visible. Copy `numBeams * maxReturns` slots
        // — the rgen lays out results as beamIdx * maxReturns + returnSlot.
        VkBufferCopy region{};
        region.size = static_cast<VkDeviceSize>(totalSlots) * sizeof(vulkan_lidar::LidarResult);
        vkCmdCopyBuffer(cmd, s.resultBuf.handle, s.readbackBuf.handle, 1, &region);

        // Barrier: transfer write → host read.
        VkBufferMemoryBarrier2 copyToHost{};
        copyToHost.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        copyToHost.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        copyToHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        copyToHost.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
        copyToHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        copyToHost.srcQueueFamilyIndex = copyToHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyToHost.buffer = s.readbackBuf.handle;
        copyToHost.size   = VK_WHOLE_SIZE;
        VkDependencyInfo hostDep{};
        hostDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        hostDep.bufferMemoryBarrierCount = 1;
        hostDep.pBufferMemoryBarriers = &copyToHost;
        vkCmdPipelineBarrier2(cmd, &hostDep);

        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(lidar)");

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        check(vkQueueSubmit(queue, 1, &si, s.fence), "vkQueueSubmit(lidar)");

        s.submitted = true;
        return handle;
    }

    bool LidarScanner::ready(int handle) const {

        const Slot* s = slotFor(handle);
        if (s == nullptr) return false;
        // A reserved-but-unsubmitted slot (unbuilt scene) has nothing to wait
        // for — its all-miss result is available immediately.
        if (!s->submitted) return true;
        return vkGetFenceStatus(ctx_.device(), s->fence) == VK_SUCCESS;
    }

    uint32_t LidarScanner::resultSlots(int handle) const {

        const Slot* s = slotFor(handle);
        return s ? s->slots : 0u;
    }

    bool LidarScanner::collect(int handle, vulkan_lidar::LidarResult* outResults) {

        if (outResults == nullptr) return false;
        Slot* slot = slotFor(handle);
        if (slot == nullptr) return false;
        Slot& s = *slot;

        const uint32_t totalSlots = s.slots;
        const bool submitted = s.submitted;
        s.reserved  = false;
        s.submitted = false;
        s.slots     = 0;
        if (totalSlots == 0) return false;

        // Nothing was submitted (unbuilt scene): the caller is still owed a
        // well-formed all-miss cloud of the right length.
        if (!submitted) {
            for (uint32_t i = 0; i < totalSlots; ++i) {
                vulkan_lidar::LidarResult& r = outResults[i];
                r.position[0] = r.position[1] = r.position[2] = 0.f;
                r.normal[0]   = r.normal[1]   = r.normal[2]   = 0.f;
                r.distance    = 0.f;
                r.intensity   = 0.f;
                r.instanceId  = -1;
                r.returnNo    = 0;
                r._pad0      = 0.f;
                r.returnKind = 0;
            }
            return true;
        }

        check(vkWaitForFences(ctx_.device(), 1, &s.fence, VK_TRUE, UINT64_MAX),
              "vkWaitForFences(lidar)");

        // Memcpy from mapped readback buffer. VMA HOST_ACCESS_RANDOM keeps it
        // mapped continuously, but the spec requires an invalidate before
        // reading non-coherent host-visible memory; a no-op when coherent.
        invalidateHostReads(ctx_.allocator(), s.readbackBuf.alloc, 0,
                            static_cast<VkDeviceSize>(totalSlots) * sizeof(vulkan_lidar::LidarResult));
        void* mapped = nullptr;
        check(vmaMapMemory(ctx_.allocator(), s.readbackBuf.alloc, &mapped),
              "vmaMapMemory(lidar readback)");
        std::memcpy(outResults, mapped, totalSlots * sizeof(vulkan_lidar::LidarResult));
        vmaUnmapMemory(ctx_.allocator(), s.readbackBuf.alloc);
        return true;
    }

}// namespace threepp::vulkan
