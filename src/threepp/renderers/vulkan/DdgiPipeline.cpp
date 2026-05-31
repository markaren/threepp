#include "threepp/renderers/vulkan/DdgiPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"
#include "threepp/renderers/vulkan/shaders/ddgi_update.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/ddgi_miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/ddgi_blend.comp.spv.h"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace threepp::vulkan {

    namespace {
        // Host mirror of ddgi_update.rgen's `DdgiUbo` (scalar block layout —
        // every member 4-byte aligned, tightly packed). Uploaded per frame once
        // the grid is sized to the scene AABB (a later increment); allocated
        // here so the buffer + binding exist.
        struct DdgiUboHost {
            float   gridOrigin[3];
            float   gridSpacing[3];
            int32_t probeCounts[3];
            int32_t raysPerProbe;
            int32_t totalProbes;
            float   rayRot0[3];
            float   rayRot1[3];
            float   rayRot2[3];
        };
    }// namespace

    DdgiPipeline::DdgiPipeline(VulkanContext& ctx, VkCommandPool cmdPool)
        : ctx_(ctx), cmdPool_(cmdPool) {
        computeGridDims();
        createImages();
        createBuffers();
        createUpdatePipeline();
        createSbt();
        createBlendPipeline();
        createDescriptors();
    }

    DdgiPipeline::~DdgiPipeline() {
        VkDevice d = ctx_.device();
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (blendPipeline_)  vkDestroyPipeline(d, blendPipeline_, nullptr);
        if (blendLayout_)    vkDestroyPipelineLayout(d, blendLayout_, nullptr);
        if (blendDsLayout_)  vkDestroyDescriptorSetLayout(d, blendDsLayout_, nullptr);
        destroyBuffer(ctx_.allocator(), sbtBuf_);
        if (updatePipeline_) vkDestroyPipeline(d, updatePipeline_, nullptr);
        if (updateLayout_)   vkDestroyPipelineLayout(d, updateLayout_, nullptr);
        if (updateDsLayout_) vkDestroyDescriptorSetLayout(d, updateDsLayout_, nullptr);
        destroyBuffer(ctx_.allocator(), ddgiUbo_);
        for (auto& b : rayRadiance_) destroyBuffer(ctx_.allocator(), b);
        for (auto& img : irradiance_) destroyImage2D(ctx_.allocator(), d, img);
    }

    void DdgiPipeline::computeGridDims() {
        probesX_ = kDdgiDefaultProbesX;
        probesY_ = kDdgiDefaultProbesY;
        probesZ_ = kDdgiDefaultProbesZ;
        totalProbes_  = probesX_ * probesY_ * probesZ_;
        raysPerProbe_ = kDdgiRaysPerProbe;
        irrRes_   = kDdgiIrradianceRes;
        border_   = kDdgiProbeBorder;
        tileSide_ = irrRes_ + 2 * border_;
        // Lay the per-probe tiles out in a roughly-square atlas.
        tilesPerRow_ = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(totalProbes_))));
        tileRows_    = (totalProbes_ + tilesPerRow_ - 1) / tilesPerRow_;
        atlasW_ = static_cast<uint32_t>(tilesPerRow_ * tileSide_);
        atlasH_ = static_cast<uint32_t>(tileRows_ * tileSide_);

        // Default grid: a ~20×10×20 m box centred horizontally on the origin,
        // floor a little below y=0. Spacing puts probe (0,0,0) at the min
        // corner and probe (counts-1) at the max corner. The renderer overrides
        // this via setGrid() once scene-AABB sizing lands.
        const float spanX = 20.f, spanY = 10.f, spanZ = 20.f;
        gridOrigin_[0] = -spanX * 0.5f;
        gridOrigin_[1] = -2.f;
        gridOrigin_[2] = -spanZ * 0.5f;
        gridSpacing_[0] = spanX / float(probesX_ > 1 ? probesX_ - 1 : 1);
        gridSpacing_[1] = spanY / float(probesY_ > 1 ? probesY_ - 1 : 1);
        gridSpacing_[2] = spanZ / float(probesZ_ > 1 ? probesZ_ - 1 : 1);
    }

    void DdgiPipeline::setGrid(const float originXYZ[3], const float spacingXYZ[3]) {
        for (int i = 0; i < 3; ++i) {
            gridOrigin_[i]  = originXYZ[i];
            gridSpacing_[i] = spacingXYZ[i];
        }
        uboWritten_ = false;// re-upload on next update
    }

    void DdgiPipeline::transitionFreshImage(VkImage img) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(ddgi)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(ddgi)");

        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);

        check(vkEndCommandBuffer(cb), "end one-shot cb(ddgi)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(ddgi)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(ddgi)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void DdgiPipeline::createImages() {
        for (auto& img : irradiance_) {
            img = Image2D{};
            img.width  = atlasW_;
            img.height = atlasH_;
            img.format = VK_FORMAT_R16G16B16A16_SFLOAT;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = img.format;
            ici.extent        = {atlasW_, atlasH_, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &img.image, &img.alloc, nullptr),
                  "vmaCreateImage(ddgi.irradiance)");

            transitionFreshImage(img.image);

            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = img.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = img.format;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx_.device(), &vci, nullptr, &img.view),
                  "vkCreateImageView(ddgi.irradiance)");
        }
    }

    void DdgiPipeline::createBuffers() {
        for (auto& b : rayRadiance_)
            b = createBuffer(
                    ctx_.allocator(), ctx_.device(),
                    static_cast<VkDeviceSize>(totalProbes_) * raysPerProbe_ * sizeof(float) * 4,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

        ddgiUbo_ = createBuffer(
                ctx_.allocator(), ctx_.device(), sizeof(DdgiUboHost),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }

    void DdgiPipeline::createUpdatePipeline() {
        // Own descriptor set layout: 0=TLAS, 1=DDGI UBO, 2=ray-radiance (write),
        // 3=env sampler (read in the miss shader).
        std::array<VkDescriptorSetLayoutBinding, 4> b{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[3].descriptorCount = 1; b[3].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(b.size());
        dlci.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &updateDsLayout_),
              "vkCreateDescriptorSetLayout(ddgi.update)");

        VkPipelineLayoutCreateInfo plci{};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &updateDsLayout_;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &updateLayout_),
              "vkCreatePipelineLayout(ddgi.update)");

        auto loadModule = [this](const uint32_t* code, size_t size) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = size;
            smci.pCode    = code;
            VkShaderModule m = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &m), "vkCreateShaderModule(ddgi)");
            return m;
        };
        VkShaderModule rgenMod = loadModule(kDdgiUpdateRgenSpv, sizeof(kDdgiUpdateRgenSpv));
        VkShaderModule missMod = loadModule(kDdgiMissRmissSpv,  sizeof(kDdgiMissRmissSpv));

        std::array<VkPipelineShaderStageCreateInfo, 2> stg{};
        for (auto& s : stg) { s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; s.pName = "main"; }
        stg[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR; stg[0].module = rgenMod;
        stg[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;   stg[1].module = missMod;

        // Two general groups (rgen, miss); no hit group — the update rgen traces
        // with gl_RayFlagsSkipClosestHitShaderEXT, so the hit SBT region is empty.
        std::array<VkRayTracingShaderGroupCreateInfoKHR, 2> grp{};
        for (auto& g : grp) {
            g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.generalShader = g.closestHitShader = g.anyHitShader = g.intersectionShader = VK_SHADER_UNUSED_KHR;
        }
        grp[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR; grp[0].generalShader = 0;
        grp[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR; grp[1].generalShader = 1;

        VkRayTracingPipelineCreateInfoKHR rci{};
        rci.sType      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rci.stageCount = static_cast<uint32_t>(stg.size());
        rci.pStages    = stg.data();
        rci.groupCount = static_cast<uint32_t>(grp.size());
        rci.pGroups    = grp.data();
        rci.maxPipelineRayRecursionDepth = 1;
        rci.layout     = updateLayout_;
        check(ctx_.rt().createRayTracingPipelines(
                      ctx_.device(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rci, nullptr, &updatePipeline_),
              "vkCreateRayTracingPipelinesKHR(ddgi.update)");

        vkDestroyShaderModule(ctx_.device(), rgenMod, nullptr);
        vkDestroyShaderModule(ctx_.device(), missMod, nullptr);
    }

    void DdgiPipeline::createSbt() {
        const auto& props = ctx_.rtPipelineProperties();
        const uint32_t handleSize        = props.shaderGroupHandleSize;
        const uint32_t handleAlignment   = props.shaderGroupHandleAlignment;
        const uint32_t baseAlignment     = props.shaderGroupBaseAlignment;
        const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

        constexpr uint32_t groupCount = 2;// rgen, miss
        const uint32_t handlesSize = groupCount * handleSize;
        std::vector<uint8_t> handles(handlesSize);
        check(ctx_.rt().getRayTracingShaderGroupHandles(
                      ctx_.device(), updatePipeline_, 0, groupCount, handlesSize, handles.data()),
              "vkGetRayTracingShaderGroupHandlesKHR(ddgi)");

        const uint32_t rgenBytes = alignUp(handleSizeAligned, baseAlignment);
        const uint32_t missBytes = alignUp(handleSizeAligned, baseAlignment);
        const VkDeviceSize sbtSize = static_cast<VkDeviceSize>(rgenBytes) + missBytes;

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
        std::memcpy(dst,            handles.data() + 0 * handleSize, handleSize);
        std::memcpy(dst + rgenBytes, handles.data() + 1 * handleSize, handleSize);
        vmaUnmapMemory(ctx_.allocator(), sbtBuf_.alloc);

        const VkDeviceAddress base = sbtBuf_.address;
        rgenRgn_.deviceAddress = base;
        rgenRgn_.stride        = rgenBytes;
        rgenRgn_.size          = rgenBytes;
        missRgn_.deviceAddress = base + rgenBytes;
        missRgn_.stride        = handleSizeAligned;
        missRgn_.size          = missBytes;
        hitRgn_  = {};// no hit shaders (SkipClosestHit)
        callRgn_ = {};
    }

    void DdgiPipeline::createBlendPipeline() {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(b.size());
        dlci.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &blendDsLayout_),
              "vkCreateDescriptorSetLayout(ddgi.blend)");

        // ddgi_blend.comp push constants: ivec4 + ivec4 + vec4 + 3×vec4 = 96 B.
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = 96;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &blendDsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &blendLayout_),
              "vkCreatePipelineLayout(ddgi.blend)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kDdgiBlendCompSpv);
        smci.pCode    = kDdgiBlendCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(ddgi.blend)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = blendLayout_;
        check(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &blendPipeline_),
              "vkCreateComputePipelines(ddgi.blend)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);
    }

    void DdgiPipeline::createDescriptors() {
        // kPingPong update sets (TLAS + UBO + ray buffer + env) and kPingPong
        // blend sets (ray buffer + 2 atlas images).
        std::array<VkDescriptorPoolSize, 5> sizes{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; sizes[0].descriptorCount = kPingPong;
        sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;             sizes[1].descriptorCount = kPingPong;
        sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;             sizes[2].descriptorCount = 2 * kPingPong;
        sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;     sizes[3].descriptorCount = kPingPong;
        sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;              sizes[4].descriptorCount = 2 * kPingPong;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 2 * kPingPong;// kPingPong update + kPingPong blend
        dpci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        dpci.pPoolSizes    = sizes.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(ddgi)");

        for (auto& set : updateSets_) {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &updateDsLayout_;
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, &set),
                  "vkAllocateDescriptorSets(ddgi.update)");
        }
        for (auto& set : blendSets_) {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &blendDsLayout_;
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, &set),
                  "vkAllocateDescriptorSets(ddgi.blend)");
        }
        // Descriptor *contents* are written lazily on first record* per parity.
    }

    void DdgiPipeline::recordUpdate(VkCommandBuffer cb, VkAccelerationStructureKHR tlas,
                                    VkImageView envView, VkSampler envSampler, uint32_t frame) {
        const uint32_t idx = frame % kPingPong;

        // Upload the static DDGI uniform once (grid params + identity ray
        // rotation; per-frame rotation is a tuning follow-up needing a
        // double-buffered UBO).
        if (!uboWritten_) {
            DdgiUboHost u{};
            for (int i = 0; i < 3; ++i) { u.gridOrigin[i] = gridOrigin_[i]; u.gridSpacing[i] = gridSpacing_[i]; }
            u.probeCounts[0] = probesX_; u.probeCounts[1] = probesY_; u.probeCounts[2] = probesZ_;
            u.raysPerProbe = raysPerProbe_;
            u.totalProbes  = totalProbes_;
            u.rayRot0[0] = 1.f; u.rayRot1[1] = 1.f; u.rayRot2[2] = 1.f;// identity
            void* mapped = nullptr;
            vmaMapMemory(ctx_.allocator(), ddgiUbo_.alloc, &mapped);
            std::memcpy(mapped, &u, sizeof(u));
            vmaUnmapMemory(ctx_.allocator(), ddgiUbo_.alloc);
            uboWritten_ = true;
        }

        // Populate this parity's update set once (TLAS / env stable for bring-up;
        // rewiring on scene-rebuild TLAS change is a follow-up).
        if (!updateWired_[idx]) {
            VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
            asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            asInfo.accelerationStructureCount = 1;
            asInfo.pAccelerationStructures    = &tlas;
            VkDescriptorBufferInfo uboInfo{ddgiUbo_.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo rayInfo{rayRadiance_[idx].handle, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo  envInfo{envSampler, envView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

            std::array<VkWriteDescriptorSet, 4> w{};
            for (auto& x : w) {
                x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                x.dstSet = updateSets_[idx];
                x.descriptorCount = 1;
            }
            w[0].dstBinding = 0; w[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; w[0].pNext = &asInfo;
            w[1].dstBinding = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;             w[1].pBufferInfo = &uboInfo;
            w[2].dstBinding = 2; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;             w[2].pBufferInfo = &rayInfo;
            w[3].dstBinding = 3; w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;     w[3].pImageInfo = &envInfo;
            vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
            updateWired_[idx] = true;
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, updatePipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                updateLayout_, 0, 1, &updateSets_[idx], 0, nullptr);
        ctx_.rt().cmdTraceRays(cb, &rgenRgn_, &missRgn_, &hitRgn_, &callRgn_,
                               static_cast<uint32_t>(raysPerProbe_),
                               static_cast<uint32_t>(totalProbes_), 1);

        VkBufferMemoryBarrier2 bb{};
        bb.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        bb.srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        bb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        bb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = rayRadiance_[idx].handle;
        bb.size   = VK_WHOLE_SIZE;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers    = &bb;
        vkCmdPipelineBarrier2(cb, &dep);
    }

    void DdgiPipeline::recordBlend(VkCommandBuffer cb, uint32_t frame) {
        const uint32_t idx = frame % kPingPong;

        if (!blendWired_[idx]) {
            VkDescriptorBufferInfo rayInfo{rayRadiance_[idx].handle, 0, VK_WHOLE_SIZE};
            // Bring-up has no temporal accumulation (hysteresis 0), so prev and
            // curr point at the same image; the read value is multiplied by 0.
            VkDescriptorImageInfo prevInfo{VK_NULL_HANDLE, irradiance_[idx].view, VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo currInfo{VK_NULL_HANDLE, irradiance_[idx].view, VK_IMAGE_LAYOUT_GENERAL};
            std::array<VkWriteDescriptorSet, 3> w{};
            for (auto& x : w) {
                x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                x.dstSet = blendSets_[idx];
                x.descriptorCount = 1;
            }
            w[0].dstBinding = 0; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &rayInfo;
            w[1].dstBinding = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  w[1].pImageInfo = &prevInfo;
            w[2].dstBinding = 2; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  w[2].pImageInfo = &currInfo;
            vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
            blendWired_[idx] = true;
        }

        struct BlendPc {
            int32_t cfg0[4];// interiorRes, border, tilesPerRow, totalProbes
            int32_t cfg1[4];// raysPerProbe, _, _, _
            float   blend[4];// hysteresis, _, _, _
            float   rot0[4]; float rot1[4]; float rot2[4];
        } pc{};
        pc.cfg0[0] = irrRes_; pc.cfg0[1] = border_; pc.cfg0[2] = tilesPerRow_; pc.cfg0[3] = totalProbes_;
        pc.cfg1[0] = raysPerProbe_;
        pc.blend[0] = 0.f;// no temporal accumulation for bring-up
        pc.rot0[0] = 1.f; pc.rot1[1] = 1.f; pc.rot2[2] = 1.f;// identity (matches update)

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, blendPipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                blendLayout_, 0, 1, &blendSets_[idx], 0, nullptr);
        vkCmdPushConstants(cb, blendLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        // One workgroup per probe — the blend shader's local size covers a full
        // probe tile and loads that probe's rays into shared memory once.
        vkCmdDispatch(cb, static_cast<uint32_t>(totalProbes_), 1, 1);

        VkImageMemoryBarrier2 ib{};
        ib.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        ib.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        ib.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        ib.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        ib.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = irradiance_[idx].image;
        ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ib.subresourceRange.levelCount = 1;
        ib.subresourceRange.layerCount = 1;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &ib;
        vkCmdPipelineBarrier2(cb, &dep);
    }

}// namespace threepp::vulkan
