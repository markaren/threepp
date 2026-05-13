#include "threepp/renderers/vulkan/PhotonCaustics.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"
#include "threepp/renderers/vulkan/shaders/photon_emit.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/photon_miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/photon_chit.rchit.spv.h"

#include <array>
#include <cstring>
#include <vector>

namespace threepp::vulkan {

    PhotonCaustics::PhotonCaustics(VulkanContext& ctx, VkPipelineLayout sharedRtLayout)
        : ctx_(ctx), sharedLayout_(sharedRtLayout) {
        createBuffers();
        createPipeline();
        createSbt();
    }

    PhotonCaustics::~PhotonCaustics() {
        VkDevice d = ctx_.device();
        destroyBuffer(ctx_.allocator(), sbtBuf_);
        if (emitPipeline_) vkDestroyPipeline(d, emitPipeline_, nullptr);
        destroyBuffer(ctx_.allocator(), countBuf_);
        destroyBuffer(ctx_.allocator(), dataBuf_);
    }

    void PhotonCaustics::createBuffers() {
        countBuf_ = createBuffer(
                ctx_.allocator(), ctx_.device(),
                static_cast<VkDeviceSize>(kPhotonGridSize) * sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                0);
        // 3 vec3 per slot: position, flux, direction.
        dataBuf_ = createBuffer(
                ctx_.allocator(), ctx_.device(),
                static_cast<VkDeviceSize>(kPhotonGridSize) * kPhotonsPerCell * 3 * 12,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                0);
    }

    void PhotonCaustics::createPipeline() {
        auto loadModule = [this](const uint32_t* code, size_t size) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = size;
            smci.pCode    = code;
            VkShaderModule m = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &m),
                  "vkCreateShaderModule(photon)");
            return m;
        };
        VkShaderModule rgenMod = loadModule(kPhotonEmitRgenSpv, sizeof(kPhotonEmitRgenSpv));
        VkShaderModule missMod = loadModule(kPhotonMissRmissSpv, sizeof(kPhotonMissRmissSpv));
        VkShaderModule chitMod = loadModule(kPhotonChitRchitSpv, sizeof(kPhotonChitRchitSpv));

        // Groups: 0=rgen, 1=miss, 2=photon closest-hit.
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
        rci.maxPipelineRayRecursionDepth = 1; // loop in raygen, no nested traces
        rci.layout = sharedLayout_;           // shared with main RT pipeline

        check(ctx_.rt().createRayTracingPipelines(
                      ctx_.device(), VK_NULL_HANDLE, VK_NULL_HANDLE,
                      1, &rci, nullptr, &emitPipeline_),
              "vkCreateRayTracingPipelinesKHR(photon)");

        vkDestroyShaderModule(ctx_.device(), rgenMod, nullptr);
        vkDestroyShaderModule(ctx_.device(), missMod, nullptr);
        vkDestroyShaderModule(ctx_.device(), chitMod, nullptr);
    }

    void PhotonCaustics::createSbt() {
        const auto& props = ctx_.rtPipelineProperties();
        const uint32_t handleSize        = props.shaderGroupHandleSize;
        const uint32_t handleAlignment   = props.shaderGroupHandleAlignment;
        const uint32_t baseAlignment     = props.shaderGroupBaseAlignment;
        const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

        constexpr uint32_t groupCount = 3; // rgen, miss, hit
        const uint32_t handlesSize = groupCount * handleSize;
        std::vector<uint8_t> handles(handlesSize);
        check(ctx_.rt().getRayTracingShaderGroupHandles(
                      ctx_.device(), emitPipeline_, 0, groupCount,
                      handlesSize, handles.data()),
              "vkGetRayTracingShaderGroupHandlesKHR(photon)");

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
        std::memcpy(dst,                        handles.data() + 0 * handleSize, handleSize);
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

    void PhotonCaustics::recordZeroFillCounts(VkCommandBuffer cb) {
        vkCmdFillBuffer(cb, countBuf_.handle, 0, VK_WHOLE_SIZE, 0u);
        VkBufferMemoryBarrier2 fillDone{};
        fillDone.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        fillDone.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        fillDone.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        fillDone.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        fillDone.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        fillDone.srcQueueFamilyIndex = fillDone.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fillDone.buffer = countBuf_.handle;
        fillDone.size   = VK_WHOLE_SIZE;
        VkDependencyInfo fillDep{};
        fillDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        fillDep.bufferMemoryBarrierCount = 1;
        fillDep.pBufferMemoryBarriers = &fillDone;
        vkCmdPipelineBarrier2(cb, &fillDep);
        initialized_ = true;
    }

    void PhotonCaustics::recordEmitPass(VkCommandBuffer cb,
                                        VkDescriptorSet descSet,
                                        const EmitPushConstants& push) {
        initialized_ = true;

        // 1. Zero per-cell photon counters (counts only; data is overwritten
        //    in place, old overflow slots beyond the new count are never read).
        vkCmdFillBuffer(cb, countBuf_.handle, 0, VK_WHOLE_SIZE, 0u);

        // 2. Barrier: fill → raygen read/write in photon emit shader.
        {
            VkBufferMemoryBarrier2 fillDone{};
            fillDone.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            fillDone.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            fillDone.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            fillDone.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            fillDone.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            fillDone.srcQueueFamilyIndex = fillDone.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            fillDone.buffer = countBuf_.handle;
            fillDone.size   = VK_WHOLE_SIZE;
            VkDependencyInfo fillDep{};
            fillDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            fillDep.bufferMemoryBarrierCount = 1;
            fillDep.pBufferMemoryBarriers = &fillDone;
            vkCmdPipelineBarrier2(cb, &fillDep);
        }

        // 3. Photon emit dispatch.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, emitPipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                sharedLayout_, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cb, sharedLayout_,
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                   VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                   VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                   VK_SHADER_STAGE_MISS_BIT_KHR,
                           0, sizeof(push), &push);
        ctx_.rt().cmdTraceRays(cb, &rgenRgn_, &missRgn_, &hitRgn_, &callRgn_,
                                kPhotonEmitDim, kPhotonEmitDim, 1);

        // 4. Barrier: photon writes → closest_hit reads (caustic gather).
        std::array<VkBufferMemoryBarrier2, 2> photonDone{};
        for (auto& b : photonDone) {
            b.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            b.srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.size = VK_WHOLE_SIZE;
        }
        photonDone[0].buffer = countBuf_.handle;
        photonDone[1].buffer = dataBuf_.handle;

        VkDependencyInfo photonDep{};
        photonDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        photonDep.bufferMemoryBarrierCount = 2;
        photonDep.pBufferMemoryBarriers = photonDone.data();
        vkCmdPipelineBarrier2(cb, &photonDep);
    }

}// namespace threepp::vulkan
