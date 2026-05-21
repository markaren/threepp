#include "threepp/renderers/vulkan/LidarScanner.hpp"

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

    LidarScanner::LidarScanner(VulkanContext& ctx) : ctx_(ctx) {
        createDescriptorLayout();
        createPipeline();
        createSbt();
        createCommandObjects();
    }

    LidarScanner::~LidarScanner() {
        VkDevice d = ctx_.device();
        if (fence_)    vkDestroyFence(d, fence_, nullptr);
        if (cmdPool_)  vkDestroyCommandPool(d, cmdPool_, nullptr);
        destroyBuffer(ctx_.allocator(), readbackBuf_);
        destroyBuffer(ctx_.allocator(), resultBuf_);
        destroyBuffer(ctx_.allocator(), beamBuf_);
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
        std::array<VkDescriptorSetLayoutBinding, 6> b{};
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

        // Descriptor pool sized for exactly one set.
        std::array<VkDescriptorPoolSize, 3> ps{};
        ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        ps[0].descriptorCount = 1;
        ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[1].descriptorCount = 4;
        ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps[2].descriptorCount = 1;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
        dpci.pPoolSizes = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(lidar)");

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = descPool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &descSetLayout_;
        check(vkAllocateDescriptorSets(ctx_.device(), &dsai, &descSet_),
              "vkAllocateDescriptorSets(lidar)");
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

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cmdPool_;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(ctx_.device(), &cbai, &cmdBuf_),
              "vkAllocateCommandBuffers(lidar)");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check(vkCreateFence(ctx_.device(), &fci, nullptr, &fence_),
              "vkCreateFence(lidar)");
    }

    void LidarScanner::ensureCapacity(uint32_t numBeams, uint32_t slotsPerBeam) {
        const uint32_t beamsNeeded   = roundUpPow2(std::max(1u, numBeams));
        const uint32_t resultsNeeded = roundUpPow2(std::max(1u, numBeams * std::max(1u, slotsPerBeam)));
        const bool beamsGrew   = (beamsNeeded   > capacityBeams_);
        const bool resultsGrew = (resultsNeeded > capacityResults_);
        if (!beamsGrew && !resultsGrew) return;

        // Wait for any in-flight work on these buffers before freeing. The
        // scanner's own dispatch is the only thing that touches them, and
        // we've either never submitted (first call) or already waited on
        // fence_ at the end of the previous scan().
        if (beamsGrew) {
            destroyBuffer(ctx_.allocator(), beamBuf_);
            const VkDeviceSize beamBytes =
                    static_cast<VkDeviceSize>(beamsNeeded) * sizeof(vulkan_lidar::LidarBeam);
            beamBuf_ = createBuffer(
                    ctx_.allocator(), ctx_.device(), beamBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            capacityBeams_ = beamsNeeded;
        }

        if (resultsGrew) {
            destroyBuffer(ctx_.allocator(), resultBuf_);
            destroyBuffer(ctx_.allocator(), readbackBuf_);
            const VkDeviceSize resultBytes =
                    static_cast<VkDeviceSize>(resultsNeeded) * sizeof(vulkan_lidar::LidarResult);
            resultBuf_ = createBuffer(
                    ctx_.allocator(), ctx_.device(), resultBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    0);
            readbackBuf_ = createBuffer(
                    ctx_.allocator(), ctx_.device(), resultBytes,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);
            capacityResults_ = resultsNeeded;
        }

        // Update descriptor set bindings 3 (beams) and 4 (results) to point
        // at the new buffers. The scene bindings (0/1/2) are touched per
        // scan in updateSceneBindings.
        VkDescriptorBufferInfo beamInfo{};
        beamInfo.buffer = beamBuf_.handle;
        beamInfo.offset = 0;
        beamInfo.range  = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo resultInfo{};
        resultInfo.buffer = resultBuf_.handle;
        resultInfo.offset = 0;
        resultInfo.range  = VK_WHOLE_SIZE;

        std::array<VkWriteDescriptorSet, 2> w{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = descSet_;
        w[0].dstBinding = 3;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[0].pBufferInfo = &beamInfo;

        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = descSet_;
        w[1].dstBinding = 4;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo = &resultInfo;

        vkUpdateDescriptorSets(ctx_.device(),
                               static_cast<uint32_t>(w.size()), w.data(),
                               0, nullptr);
    }

    void LidarScanner::updateSceneBindings(VkAccelerationStructureKHR tlas,
                                           VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                                           VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                                           VkBuffer fogUbo, VkDeviceSize fogUboSize) {
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

        std::array<VkWriteDescriptorSet, 4> w{};
        w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].pNext           = &asi;
        w[0].dstSet          = descSet_;
        w[0].dstBinding      = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet          = descSet_;
        w[1].dstBinding      = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo     = &geomInfo;

        w[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[2].dstSet          = descSet_;
        w[2].dstBinding      = 2;
        w[2].descriptorCount = 1;
        w[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[2].pBufferInfo     = &matInfo;

        w[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[3].dstSet          = descSet_;
        w[3].dstBinding      = 5;
        w[3].descriptorCount = 1;
        w[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[3].pBufferInfo     = &fogInfo;

        vkUpdateDescriptorSets(ctx_.device(),
                               static_cast<uint32_t>(w.size()), w.data(),
                               0, nullptr);
    }

    void LidarScanner::scan(VkQueue queue,
                            VkAccelerationStructureKHR tlas,
                            VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                            VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                            VkBuffer fogUbo, VkDeviceSize fogUboSize,
                            const vulkan_lidar::LidarPushConstants& pc,
                            const vulkan_lidar::LidarBeam* beams, uint32_t numBeams,
                            vulkan_lidar::LidarResult* outResults) {
        if (numBeams == 0 || outResults == nullptr || beams == nullptr) return;

        const uint32_t maxReturns   = std::max(pc.maxReturns, 1u);
        const uint32_t samples      = std::max(pc.samplesPerBeam, 1u);
        const uint32_t slotsPerBeam = maxReturns * samples;
        const uint32_t totalSlots   = numBeams * slotsPerBeam;

        // Bail out gracefully if the scene isn't ready — write all misses.
        const bool sceneReady = (tlas != VK_NULL_HANDLE) &&
                                (geomDescsBuffer != VK_NULL_HANDLE) &&
                                (matDescsBuffer != VK_NULL_HANDLE) &&
                                (fogUbo != VK_NULL_HANDLE);
        if (!sceneReady) {
            for (uint32_t i = 0; i < totalSlots; ++i) {
                vulkan_lidar::LidarResult& r = outResults[i];
                r.position[0] = r.position[1] = r.position[2] = 0.f;
                r.normal[0]   = r.normal[1]   = r.normal[2]   = 0.f;
                r.distance    = 0.f;
                r.intensity   = 0.f;
                r.instanceId  = -1;
                r.returnNo    = 0;
                r._pad[0] = r._pad[1] = 0.f;
            }
            return;
        }

        ensureCapacity(numBeams, slotsPerBeam);

        // Upload beams (mapped, sequential write).
        {
            void* mapped = nullptr;
            check(vmaMapMemory(ctx_.allocator(), beamBuf_.alloc, &mapped),
                  "vmaMapMemory(lidar beams)");
            std::memcpy(mapped, beams, numBeams * sizeof(vulkan_lidar::LidarBeam));
            vmaUnmapMemory(ctx_.allocator(), beamBuf_.alloc);
        }

        updateSceneBindings(tlas, geomDescsBuffer, geomDescsSize,
                            matDescsBuffer, matDescsSize,
                            fogUbo, fogUboSize);

        // Reset cmd buffer + fence and re-record.
        check(vkResetCommandBuffer(cmdBuf_, 0), "vkResetCommandBuffer(lidar)");
        check(vkResetFences(ctx_.device(), 1, &fence_), "vkResetFences(lidar)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmdBuf_, &bi), "vkBeginCommandBuffer(lidar)");

        vkCmdBindPipeline(cmdBuf_, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_);
        vkCmdBindDescriptorSets(cmdBuf_, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                pipelineLayout_, 0, 1, &descSet_, 0, nullptr);
        vkCmdPushConstants(cmdBuf_, pipelineLayout_,
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                   VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                   VK_SHADER_STAGE_MISS_BIT_KHR,
                           0, sizeof(pc), &pc);
        ctx_.rt().cmdTraceRays(cmdBuf_, &rgenRgn_, &missRgn_, &hitRgn_, &callRgn_,
                                numBeams, 1, 1);

        // Barrier: RT shader writes to resultBuf_ → transfer read.
        VkBufferMemoryBarrier2 rtToCopy{};
        rtToCopy.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        rtToCopy.srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        rtToCopy.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        rtToCopy.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        rtToCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        rtToCopy.srcQueueFamilyIndex = rtToCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rtToCopy.buffer = resultBuf_.handle;
        rtToCopy.size   = VK_WHOLE_SIZE;
        VkDependencyInfo rtDep{};
        rtDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        rtDep.bufferMemoryBarrierCount = 1;
        rtDep.pBufferMemoryBarriers = &rtToCopy;
        vkCmdPipelineBarrier2(cmdBuf_, &rtDep);

        // Copy device → host-visible. Copy `numBeams * maxReturns` slots
        // — the rgen lays out results as beamIdx * maxReturns + returnSlot.
        VkBufferCopy region{};
        region.size = static_cast<VkDeviceSize>(totalSlots) * sizeof(vulkan_lidar::LidarResult);
        vkCmdCopyBuffer(cmdBuf_, resultBuf_.handle, readbackBuf_.handle, 1, &region);

        // Barrier: transfer write → host read.
        VkBufferMemoryBarrier2 copyToHost{};
        copyToHost.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        copyToHost.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        copyToHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        copyToHost.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
        copyToHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        copyToHost.srcQueueFamilyIndex = copyToHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyToHost.buffer = readbackBuf_.handle;
        copyToHost.size   = VK_WHOLE_SIZE;
        VkDependencyInfo hostDep{};
        hostDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        hostDep.bufferMemoryBarrierCount = 1;
        hostDep.pBufferMemoryBarriers = &copyToHost;
        vkCmdPipelineBarrier2(cmdBuf_, &hostDep);

        check(vkEndCommandBuffer(cmdBuf_), "vkEndCommandBuffer(lidar)");

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmdBuf_;
        check(vkQueueSubmit(queue, 1, &si, fence_), "vkQueueSubmit(lidar)");

        check(vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, UINT64_MAX),
              "vkWaitForFences(lidar)");

        // Memcpy from mapped readback buffer. VMA HOST_ACCESS_RANDOM keeps it
        // mapped continuously, but the spec requires an invalidate before
        // reading non-coherent host-visible memory; vmaInvalidateAllocation
        // is a no-op when the memory is coherent.
        vmaInvalidateAllocation(ctx_.allocator(), readbackBuf_.alloc,
                                0,
                                static_cast<VkDeviceSize>(totalSlots) * sizeof(vulkan_lidar::LidarResult));
        void* mapped = nullptr;
        check(vmaMapMemory(ctx_.allocator(), readbackBuf_.alloc, &mapped),
              "vmaMapMemory(lidar readback)");
        std::memcpy(outResults, mapped, totalSlots * sizeof(vulkan_lidar::LidarResult));
        vmaUnmapMemory(ctx_.allocator(), readbackBuf_.alloc);
    }

}// namespace threepp::vulkan
