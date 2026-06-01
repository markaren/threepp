#include "threepp/renderers/vulkan/DdgiPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"
#include "threepp/renderers/vulkan/shaders/ddgi_update.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/ddgi_blend.comp.spv.h"
// Main RT hit/miss groups — the probe update reuses them so probe rays shade
// through the real closest_hit (probe mode), instead of a duplicated shader.
#include "threepp/renderers/vulkan/shaders/miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/shadow_miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/closest_hit.rchit.spv.h"
#include "threepp/renderers/vulkan/shaders/closest_hit_alpha.rahit.spv.h"
#include "threepp/renderers/vulkan/shaders/shadow_anyhit.rahit.spv.h"

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
            float   intensity;// artistic multiplier on the sampled irradiance
        };
    }// namespace

    DdgiPipeline::DdgiPipeline(VulkanContext& ctx, VkCommandPool cmdPool, VkPipelineLayout sharedRtLayout)
        : ctx_(ctx), cmdPool_(cmdPool), sharedLayout_(sharedRtLayout) {
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

    void DdgiPipeline::setGridFromBounds(const float minXYZ[3], const float maxXYZ[3]) {
        const int counts[3] = {probesX_, probesY_, probesZ_};
        for (int i = 0; i < 3; ++i) {
            gridOrigin_[i]  = minXYZ[i];
            const float span = maxXYZ[i] - minXYZ[i];
            // N probes span N-1 intervals; origin sits on probe 0, the opposite
            // corner on probe N-1, so the lattice exactly covers [min, max].
            gridSpacing_[i] = span / float(counts[i] > 1 ? counts[i] - 1 : 1);
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

        // UNDEFINED → TRANSFER_DST so we can zero-clear the fresh atlas. Without
        // the clear, the first frames of temporal accumulation (hysteresis) read
        // undefined VMA contents as "previous irradiance" and that garbage —
        // possibly NaN/Inf — persists for many frames before averaging out.
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);

        VkClearColorValue clearVal{};// {0,0,0,0}
        VkImageSubresourceRange clearRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &clearRange);

        // TRANSFER_DST → GENERAL for shader storage-image read/write.
        VkImageMemoryBarrier b2{};
        b2.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b2.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b2.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b2.image               = img;
        b2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b2.subresourceRange.levelCount = 1;
        b2.subresourceRange.layerCount = 1;
        b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b2);

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
        // Reuse the SHARED RT layout + the MAIN hit/miss groups, swapping in
        // ddgi_update.rgen as the entry point. The 5-group structure mirrors the
        // main pipeline EXACTLY so closest_hit's shadow-ray traceRayEXT
        // (sbtRecordOffset 1 / missIndex 1) resolves the same way. closest_hit
        // runs in probe mode (Payload.inFlags bit 64).
        auto loadModule = [this](const uint32_t* code, size_t size) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = size;
            smci.pCode    = code;
            VkShaderModule m = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &m), "vkCreateShaderModule(ddgi)");
            return m;
        };
        VkShaderModule rgenMod  = loadModule(kDdgiUpdateRgenSpv,        sizeof(kDdgiUpdateRgenSpv));
        VkShaderModule missMod  = loadModule(kMissRmissSpv,            sizeof(kMissRmissSpv));
        VkShaderModule sMissMod = loadModule(kShadowMissRmissSpv,      sizeof(kShadowMissRmissSpv));
        VkShaderModule chitMod  = loadModule(kClosestHitRchitSpv,      sizeof(kClosestHitRchitSpv));
        VkShaderModule ahitMod  = loadModule(kClosestHitAlphaRahitSpv, sizeof(kClosestHitAlphaRahitSpv));
        VkShaderModule sahitMod = loadModule(kShadowAnyhitRahitSpv,    sizeof(kShadowAnyhitRahitSpv));

        // closest_hit spec constants: restirDI=false (classic NEE — the per-pixel
        // ReSTIR reservoir doesn't apply to probe rays) and sceneFeatures=0 (skip
        // clearcoat/sheen/iridescence/glass-caustic paths — irrelevant to
        // low-frequency diffuse GI probes, smaller SPV, and no rebuild needed
        // when the scene's feature set changes).
        struct ChitSpecData { VkBool32 restirDIEnabled; uint32_t sceneFeatures; }
            chitSpecData{VK_FALSE, 0u};
        VkSpecializationMapEntry chitMapEntries[2]{};
        chitMapEntries[0].constantID = 0; chitMapEntries[0].offset = offsetof(ChitSpecData, restirDIEnabled); chitMapEntries[0].size = sizeof(VkBool32);
        chitMapEntries[1].constantID = 1; chitMapEntries[1].offset = offsetof(ChitSpecData, sceneFeatures);   chitMapEntries[1].size = sizeof(uint32_t);
        VkSpecializationInfo chitSpecInfo{};
        chitSpecInfo.mapEntryCount = 2;
        chitSpecInfo.pMapEntries   = chitMapEntries;
        chitSpecInfo.dataSize      = sizeof(ChitSpecData);
        chitSpecInfo.pData         = &chitSpecData;

        std::array<VkPipelineShaderStageCreateInfo, 6> stages{};
        for (auto& s : stages) { s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; s.pName = "main"; }
        stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;      stages[0].module = rgenMod;
        stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;        stages[1].module = missMod;
        stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;        stages[2].module = sMissMod;
        stages[3].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; stages[3].module = chitMod;
        stages[3].pSpecializationInfo = &chitSpecInfo;
        stages[4].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;     stages[4].module = ahitMod;
        stages[5].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;     stages[5].module = sahitMod;

        std::array<VkRayTracingShaderGroupCreateInfoKHR, 5> groups{};
        for (auto& g : groups) {
            g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.generalShader = g.closestHitShader = g.anyHitShader = g.intersectionShader = VK_SHADER_UNUSED_KHR;
        }
        groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;             groups[0].generalShader   = 0;
        groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;             groups[1].generalShader   = 1;
        groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;             groups[2].generalShader   = 2;
        groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR; groups[3].closestHitShader = 3; groups[3].anyHitShader = 4;
        groups[4].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR; groups[4].anyHitShader     = 5;

        VkRayTracingPipelineCreateInfoKHR rci{};
        rci.sType      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rci.stageCount = static_cast<uint32_t>(stages.size());
        rci.pStages    = stages.data();
        rci.groupCount = static_cast<uint32_t>(groups.size());
        rci.pGroups    = groups.data();
        // Probe ray → closest_hit (probe mode) → NEE shadow ray is depth 1;
        // match the main pipeline's 4 for margin (probe mode terminates, never
        // recursing into the GI sub-traces).
        rci.maxPipelineRayRecursionDepth = 4;
        rci.layout     = sharedLayout_;
        check(ctx_.rt().createRayTracingPipelines(
                      ctx_.device(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rci, nullptr, &updatePipeline_),
              "vkCreateRayTracingPipelinesKHR(ddgi.update)");

        vkDestroyShaderModule(ctx_.device(), rgenMod,  nullptr);
        vkDestroyShaderModule(ctx_.device(), missMod,  nullptr);
        vkDestroyShaderModule(ctx_.device(), sMissMod, nullptr);
        vkDestroyShaderModule(ctx_.device(), chitMod,  nullptr);
        vkDestroyShaderModule(ctx_.device(), ahitMod,  nullptr);
        vkDestroyShaderModule(ctx_.device(), sahitMod, nullptr);
    }

    void DdgiPipeline::createSbt() {
        const auto& props = ctx_.rtPipelineProperties();
        const uint32_t handleSize        = props.shaderGroupHandleSize;
        const uint32_t handleAlignment   = props.shaderGroupHandleAlignment;
        const uint32_t baseAlignment     = props.shaderGroupBaseAlignment;
        const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

        // 5 groups: rgen, primary miss, shadow miss, path hit, shadow hit —
        // identical layout to the main pipeline's SBT so closest_hit's shadow
        // traceRayEXT (sbtRecordOffset 1, missIndex 1) resolves correctly.
        constexpr uint32_t groupCount = 5;
        const uint32_t handlesSize = groupCount * handleSize;
        std::vector<uint8_t> handles(handlesSize);
        check(ctx_.rt().getRayTracingShaderGroupHandles(
                      ctx_.device(), updatePipeline_, 0, groupCount, handlesSize, handles.data()),
              "vkGetRayTracingShaderGroupHandlesKHR(ddgi)");

        const uint32_t rgenBytes = alignUp(handleSizeAligned, baseAlignment);
        const uint32_t missBytes = alignUp(2 * handleSizeAligned, baseAlignment);
        const uint32_t hitBytes  = alignUp(2 * handleSizeAligned, baseAlignment);
        const VkDeviceSize sbtSize = static_cast<VkDeviceSize>(rgenBytes) + missBytes + hitBytes;

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
        // rgen.
        std::memcpy(dst, handles.data() + 0 * handleSize, handleSize);
        // miss[0]=primary, miss[1]=shadow.
        std::memcpy(dst + rgenBytes + 0 * handleSizeAligned, handles.data() + 1 * handleSize, handleSize);
        std::memcpy(dst + rgenBytes + 1 * handleSizeAligned, handles.data() + 2 * handleSize, handleSize);
        // hit[0]=path (sbtOffset 0), hit[1]=shadow (sbtOffset 1).
        std::memcpy(dst + rgenBytes + missBytes + 0 * handleSizeAligned, handles.data() + 3 * handleSize, handleSize);
        std::memcpy(dst + rgenBytes + missBytes + 1 * handleSizeAligned, handles.data() + 4 * handleSize, handleSize);
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
        // Only the blend pass needs DDGI-owned descriptor sets — the update pass
        // uses the shared RT set. Each blend set: 1 storage buffer (ray buffer)
        // + 2 storage images (irradiance prev/curr).
        std::array<VkDescriptorPoolSize, 2> sizes{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; sizes[0].descriptorCount = kPingPong;
        sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  sizes[1].descriptorCount = 2 * kPingPong;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = kPingPong;
        dpci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        dpci.pPoolSizes    = sizes.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(ddgi)");

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

    void DdgiPipeline::recordUpdate(VkCommandBuffer cb, VkDescriptorSet sharedSet, uint32_t frame,
                                    const void* pcData, uint32_t pcSize) {
        const uint32_t idx = frame % kPingPong;

        // Upload the static DDGI uniform once (grid params + identity ray
        // rotation; per-frame rotation is a tuning follow-up needing a
        // double-buffered UBO). Lives at binding 46 of the shared set.
        if (!uboWritten_) {
            DdgiUboHost u{};
            for (int i = 0; i < 3; ++i) { u.gridOrigin[i] = gridOrigin_[i]; u.gridSpacing[i] = gridSpacing_[i]; }
            u.probeCounts[0] = probesX_; u.probeCounts[1] = probesY_; u.probeCounts[2] = probesZ_;
            u.raysPerProbe = raysPerProbe_;
            u.totalProbes  = totalProbes_;
            u.rayRot0[0] = 1.f; u.rayRot1[1] = 1.f; u.rayRot2[2] = 1.f;// identity
            u.intensity    = intensity_;
            void* mapped = nullptr;
            vmaMapMemory(ctx_.allocator(), ddgiUbo_.alloc, &mapped);
            std::memcpy(mapped, &u, sizeof(u));
            // Defensive flush in case the allocation lands on non-coherent
            // host-visible memory; harmless no-op on coherent memory. (Note:
            // the binding must also list RAYGEN in its stageFlags or the probe
            // rgen reads the UBO as zeros regardless of flushing — see the
            // shared RT layout, binding 46.)
            vmaFlushAllocation(ctx_.allocator(), ddgiUbo_.alloc, 0, VK_WHOLE_SIZE);
            vmaUnmapMemory(ctx_.allocator(), ddgiUbo_.alloc);
            uboWritten_ = true;
        }

        // Forward the path-trace push constants — closest_hit reads them in
        // probe mode (emissiveCount, env CDF, fireflyClamp, and motionFlags incl.
        // the DDGI-enabled bit that drives the infinite-bounce ambient term).
        vkCmdPushConstants(cb, sharedLayout_,
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                   VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                           0, pcSize, pcData);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, updatePipeline_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                sharedLayout_, 0, 1, &sharedSet, 0, nullptr);
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
            // Temporal accumulation: `prev` is the OTHER parity's atlas — i.e.
            // last frame's blended irradiance — and `curr` is this frame's
            // target. The blend mixes mix(result, prev, hysteresis), so probe
            // irradiance is filtered over many frames instead of recomputed
            // from scratch each frame. Without this the per-frame Monte-Carlo
            // noise in the probe-ray NEE makes the field flicker violently
            // ("lightning storm"). prev/curr are fixed per parity, so caching
            // the descriptor write per idx stays valid.
            VkDescriptorImageInfo prevInfo{VK_NULL_HANDLE, irradiance_[idx ^ 1].view, VK_IMAGE_LAYOUT_GENERAL};
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
        pc.blend[0] = hysteresis_;// temporal blend weight on the previous atlas
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
