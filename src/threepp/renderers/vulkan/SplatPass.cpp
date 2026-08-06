#include "threepp/renderers/vulkan/SplatPass.hpp"

#include "threepp/extras/DataUtils.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/vulkan/GpuTimings.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/splat_checksum.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_expand.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_indirect.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_project.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_radix_hist.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_radix_scatter.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_range.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_raster.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_scan.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/splat_scan_add.comp.spv.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace threepp::vulkan {

    namespace {

        // KEEP IN SYNC with splat_common.glsl. A mismatch here is not a
        // compile error, it is a wrong picture — the shader would index a
        // struct the host packed differently.
        constexpr uint32_t kTileW = 16, kTileH = 16;
        constexpr uint32_t kScanBlock  = 1024;// 256 threads x 4
        constexpr uint32_t kScanThreads = 256;
        constexpr uint32_t kRadixBlock = 512;// 128 threads x 4
        constexpr uint32_t kRadixBins  = 16;
        constexpr uint32_t kRadixPasses = 8; // 8 x 4 bits = the whole 32-bit key
        constexpr uint32_t kProjStride = 64; // sizeof(SplatProj)
        constexpr uint32_t kGeomStride = 40; // vec3 + float + float[6], scalar layout
        constexpr uint32_t kGlobalWords = 12;
        constexpr uint32_t kBindings   = 23;// 22 = the indirect dispatch args
        constexpr uint32_t kMaxRanges  = 64;// KEEP IN SYNC with splat_common.glsl

        constexpr uint32_t kSplatFlagOrtho     = 1u;
        constexpr uint32_t kSplatFlagDebugNaN  = 2u;
        constexpr uint32_t kSplatFlagDepthTest = 4u;
        constexpr uint32_t kSplatFlagMotion    = 8u;
        constexpr uint32_t kSplatFlagFog       = 16u;
        constexpr uint32_t kSplatFlagChecksum  = 32u;
        constexpr uint32_t kSplatFlagBgSolid   = 64u;

        struct SplatPc {
            uint32_t count, srcOff, dstOff, sumOff;
            uint32_t arg0, arg1, arg2, arg3;
        };

        // Mirrors SplatUbo in splat_common.glsl (scalar layout: no vec3
        // padding surprises, but mat4 is still 64 B and vec2/vec4 keep their
        // natural alignment — the layout below is written to match exactly).
        struct SplatUboData {
            float modelView[16];
            float proj[16];
            float projInv[16];
            float model[16];
            float prevVPfromView[16];
            float camWorld[16];
            float camPosWs[4];
            float camFwdWs[4];
            float viewport[2];
            float focal[2];
            float percentile[2];
            float jitterClip[2];
            float nearPlane;
            float preExposure;
            uint32_t splatCount;
            uint32_t shCoeffs;
            uint32_t shDegree;
            uint32_t tilesX;
            uint32_t tilesY;
            uint32_t tileBits;
            uint32_t depthBits;
            uint32_t budget;
            uint32_t flags;
            uint32_t envMipCount;
            uint32_t rangeCount;
            uint32_t ranges[kMaxRanges * 2];// (first compact index, source base)
        };

        uint32_t divUp(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

        // Number of bits the tile id needs at the TOP of the sort key. The
        // depth gets everything else, so the key always fills 32 bits and the
        // depth quantisation is as fine as the screen allows.
        uint32_t tileBitsFor(uint32_t tiles) {
            uint32_t b = 1;
            while ((1u << b) < tiles && b < 24) ++b;
            return b;
        }

        // Words scanScratch needs to scan an n-element array through all its
        // levels (each level's block sums live in their own region).
        uint32_t scanScratchWords(uint32_t n) {
            uint32_t total = 0, cur = std::max(n, 1u);
            while (true) {
                const uint32_t nb = divUp(cur, kScanBlock);
                total += nb;
                if (nb == 1) break;
                cur = nb;
            }
            return total;
        }

        VkPipeline makeComputePipe(VkDevice d, VkPipelineCache cache, VkPipelineLayout layout,
                                   const uint32_t* spv, size_t spvBytes, const char* label) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = spvBytes;
            smci.pCode    = spv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(d, &smci, nullptr, &mod), label);

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = mod;
            stage.pName  = "main";

            VkComputePipelineCreateInfo cpci{};
            cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage  = stage;
            cpci.layout = layout;
            VkPipeline pipe = VK_NULL_HANDLE;
            check(vkCreateComputePipelines(d, cache, 1, &cpci, nullptr, &pipe), label);
            vkDestroyShaderModule(d, mod, nullptr);
            return pipe;
        }

    }// namespace


    SplatPass::SplatPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight) {

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &props);
        const VkDeviceSize align = std::max<VkDeviceSize>(
                props.limits.minUniformBufferOffsetAlignment, 16);
        uboStride_ = ((sizeof(SplatUboData) + align - 1) / align) * align;

        createPipelines();
        createDescriptorPool();

        uboBuf_.resize(framesInFlight_);
        debugBuf_.resize(framesInFlight_);
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            uboBuf_[f] = createBuffer(ctx_.allocator(), ctx_.device(),
                                      uboStride_ * kMaxClouds,
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                              VMA_ALLOCATION_CREATE_MAPPED_BIT);
            debugBuf_[f] = createBuffer(ctx_.allocator(), ctx_.device(),
                                        kGlobalWords * sizeof(uint32_t),
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }

        globalBuf_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                  kGlobalWords * sizeof(uint32_t),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_AUTO);
        ctx_.setObjectName(globalBuf_.handle, "splat.globals");

        // Two VkDispatchIndirectCommands = 6 uints. STORAGE so the sizing
        // kernel can write it, INDIRECT so the dispatches can read it.
        indirectBuf_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                    6 * sizeof(uint32_t),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VMA_MEMORY_USAGE_AUTO);
        ctx_.setObjectName(indirectBuf_.handle, "splat.indirectArgs");
    }

    SplatPass::~SplatPass() {
        VkDevice d = ctx_.device();
        VmaAllocator a = ctx_.allocator();
        for (auto& kv : resident_) {
            destroyBuffer(a, kv.second->geom);
            destroyBuffer(a, kv.second->sh);
        }
        resident_.clear();
        for (auto* b : {&projBuf_, &countBuf_, &offsetBuf_, &keyA_, &valA_, &keyB_, &valB_,
                        &rangeBuf_, &globalBuf_, &histBuf_, &scanBuf_, &indirectBuf_})
            destroyBuffer(a, *b);
        for (auto& b : uboBuf_) destroyBuffer(a, b);
        for (auto& b : debugBuf_) destroyBuffer(a, b);

        for (VkPipeline p : {projectPipe_, scanPipe_, scanAddPipe_, expandPipe_, indirectPipe_,
                             histPipe_, scatterPipe_, rangePipe_, rasterPipe_, checksumPipe_})
            if (p) vkDestroyPipeline(d, p, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)    vkDestroySampler(d, sampler_, nullptr);
    }

    void SplatPass::createPipelines() {
        VkDevice d = ctx_.device();

        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        check(vkCreateSampler(d, &sci, nullptr, &sampler_), "vkCreateSampler(splat)");

        std::array<VkDescriptorSetLayoutBinding, kBindings> bnd{};
        for (uint32_t i = 0; i < kBindings; ++i) {
            bnd[i].binding         = i;
            bnd[i].descriptorCount = 1;
            bnd[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            bnd[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        bnd[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bnd[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bnd[15].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bnd[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bnd[18].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;// fog
        bnd[19].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;// clouds
        bnd[20].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;// lights (ambient + suns)
        bnd[21].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;// env

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = kBindings;
        dlci.pBindings    = bnd.data();
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(splat)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size       = sizeof(SplatPc);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(splat)");

        VkPipelineCache cache = ctx_.pipelineCache();
        projectPipe_  = makeComputePipe(d, cache, pipeLayout_, kSplatProjectCompSpv,
                                        sizeof(kSplatProjectCompSpv), "splat_project");
        scanPipe_     = makeComputePipe(d, cache, pipeLayout_, kSplatScanCompSpv,
                                        sizeof(kSplatScanCompSpv), "splat_scan");
        scanAddPipe_  = makeComputePipe(d, cache, pipeLayout_, kSplatScanAddCompSpv,
                                        sizeof(kSplatScanAddCompSpv), "splat_scan_add");
        expandPipe_   = makeComputePipe(d, cache, pipeLayout_, kSplatExpandCompSpv,
                                        sizeof(kSplatExpandCompSpv), "splat_expand");
        indirectPipe_ = makeComputePipe(d, cache, pipeLayout_, kSplatIndirectCompSpv,
                                        sizeof(kSplatIndirectCompSpv), "splat_indirect");
        histPipe_     = makeComputePipe(d, cache, pipeLayout_, kSplatRadixHistCompSpv,
                                        sizeof(kSplatRadixHistCompSpv), "splat_radix_hist");
        scatterPipe_  = makeComputePipe(d, cache, pipeLayout_, kSplatRadixScatterCompSpv,
                                        sizeof(kSplatRadixScatterCompSpv), "splat_radix_scatter");
        rangePipe_    = makeComputePipe(d, cache, pipeLayout_, kSplatRangeCompSpv,
                                        sizeof(kSplatRangeCompSpv), "splat_range");
        rasterPipe_   = makeComputePipe(d, cache, pipeLayout_, kSplatRasterCompSpv,
                                        sizeof(kSplatRasterCompSpv), "splat_raster");
        checksumPipe_ = makeComputePipe(d, cache, pipeLayout_, kSplatChecksumCompSpv,
                                        sizeof(kSplatChecksumCompSpv), "splat_checksum");
    }

    void SplatPass::createDescriptorPool() {
        const uint32_t sets = kMaxClouds * framesInFlight_;
        VkDescriptorPoolSize sz[4]{};
        sz[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sz[0].descriptorCount = sets * 4;// splat + fog + clouds + lights
        sz[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sz[1].descriptorCount = sets * 14;// 13 sort/scratch buffers + indirect args
        sz[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sz[2].descriptorCount = sets * 2;// sceneHdr + gbuf motion
        sz[3].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sz[3].descriptorCount = sets * 3;// gbuf depth + gbuf ids + env

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = sets;
        dpci.poolSizeCount = 4;
        dpci.pPoolSizes    = sz;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(splat)");
    }

    void SplatPass::oneShot(const std::function<void(VkCommandBuffer)>& body) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(splat)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(splat)");
        body(cb);
        check(vkEndCommandBuffer(cb), "end one-shot cb(splat)");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(splat)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(splat)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void SplatPass::uploadCloud(Cloud& c, const SplatCloud& src) {
        const SplatData& data = src.data();
        const uint32_t n      = static_cast<uint32_t>(data.count());
        const uint32_t coeffs = static_cast<uint32_t>(data.coeffCount());

        c.count    = n;
        c.shCoeffs = coeffs;
        c.shDegree = static_cast<uint32_t>(data.shDegree);
        if (n == 0) return;

        const VkDeviceSize geomBytes = VkDeviceSize(n) * kGeomStride;
        const VkDeviceSize shBytes   = VkDeviceSize(n) * coeffs * 2 * sizeof(uint32_t);

        // Staging + device-local. Every geometry buffer in this renderer is
        // host-visible mapped; splat data is different in kind — uploaded once,
        // read every frame by every tile, never written again — so it belongs
        // in device-local memory even though nothing else here does.
        Buffer stageGeom = createBuffer(ctx_.allocator(), ctx_.device(), geomBytes,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
        Buffer stageSh   = createBuffer(ctx_.allocator(), ctx_.device(), shBytes,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                VMA_ALLOCATION_CREATE_MAPPED_BIT);

        {
            VmaAllocationInfo gi{}, si{};
            vmaGetAllocationInfo(ctx_.allocator(), stageGeom.alloc, &gi);
            vmaGetAllocationInfo(ctx_.allocator(), stageSh.alloc, &si);
            auto* gp = static_cast<float*>(gi.pMappedData);
            auto* sp = static_cast<uint32_t*>(si.pMappedData);

            for (uint32_t i = 0; i < n; ++i) {
                float* dst = gp + size_t(i) * 10;
                dst[0] = data.means[i].x;
                dst[1] = data.means[i].y;
                dst[2] = data.means[i].z;
                dst[3] = data.opacities[i];
                data.computeCovariance(i, dst + 4);

                const float* sh = data.shAt(i);
                uint32_t* so = sp + size_t(i) * coeffs * 2;
                for (uint32_t k = 0; k < coeffs; ++k) {
                    // packHalf2x16 semantics: low 16 bits = .x, high = .y.
                    // Two words per coefficient (r,g | b,0) is the same 8 bytes
                    // the GL path's RGBA16F texel spends, and the same reason:
                    // one fetch per coefficient, nothing straddling a boundary.
                    const uint16_t r = DataUtils::toHalfFloat(sh[k * 3 + 0]);
                    const uint16_t gg = DataUtils::toHalfFloat(sh[k * 3 + 1]);
                    const uint16_t b = DataUtils::toHalfFloat(sh[k * 3 + 2]);
                    so[k * 2 + 0] = uint32_t(r) | (uint32_t(gg) << 16);
                    so[k * 2 + 1] = uint32_t(b);
                }
            }
            flushHostWrites(ctx_.allocator(), stageGeom.alloc);
            flushHostWrites(ctx_.allocator(), stageSh.alloc);
        }

        c.geom = createBuffer(ctx_.allocator(), ctx_.device(), geomBytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VMA_MEMORY_USAGE_AUTO);
        c.sh   = createBuffer(ctx_.allocator(), ctx_.device(), shBytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VMA_MEMORY_USAGE_AUTO);
        ctx_.setObjectName(c.geom.handle, "splat.geom");
        ctx_.setObjectName(c.sh.handle, "splat.sh");

        oneShot([&](VkCommandBuffer cb) {
            VkBufferCopy cg{0, 0, geomBytes};
            VkBufferCopy cs{0, 0, shBytes};
            vkCmdCopyBuffer(cb, stageGeom.handle, c.geom.handle, 1, &cg);
            vkCmdCopyBuffer(cb, stageSh.handle, c.sh.handle, 1, &cs);
        });

        destroyBuffer(ctx_.allocator(), stageGeom);
        destroyBuffer(ctx_.allocator(), stageSh);
    }

    void SplatPass::allocateScratch(uint32_t maxSplats, uint32_t entryBudget) {
        VmaAllocator a = ctx_.allocator();
        VkDevice d = ctx_.device();
        maxSplats_   = maxSplats;
        entryBudget_ = std::max(entryBudget, 1u);

        for (auto* b : {&projBuf_, &countBuf_, &offsetBuf_, &keyA_, &valA_, &keyB_, &valB_,
                        &histBuf_, &scanBuf_})
            destroyBuffer(a, *b);

        constexpr VkBufferUsageFlags kSsbo =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto mk = [&](VkDeviceSize bytes, const char* name) {
            Buffer b = createBuffer(a, d, std::max<VkDeviceSize>(bytes, 256), kSsbo,
                                    VMA_MEMORY_USAGE_AUTO);
            ctx_.setObjectName(b.handle, name);
            return b;
        };

        projBuf_   = mk(VkDeviceSize(maxSplats) * kProjStride, "splat.proj");
        countBuf_  = mk(VkDeviceSize(maxSplats) * 4, "splat.counts");
        offsetBuf_ = mk(VkDeviceSize(maxSplats) * 4, "splat.offsets");
        keyA_      = mk(VkDeviceSize(entryBudget_) * 4, "splat.keyA");
        valA_      = mk(VkDeviceSize(entryBudget_) * 4, "splat.valA");
        keyB_      = mk(VkDeviceSize(entryBudget_) * 4, "splat.keyB");
        valB_      = mk(VkDeviceSize(entryBudget_) * 4, "splat.valB");

        const uint32_t radixBlocks = divUp(entryBudget_, kRadixBlock);
        const uint32_t histWords   = kRadixBins * radixBlocks;
        histBuf_ = mk(VkDeviceSize(histWords) * 4, "splat.hist");
        scanBuf_ = mk(VkDeviceSize(std::max(scanScratchWords(histWords),
                                            scanScratchWords(maxSplats)) + 8) * 4,
                      "splat.scanScratch");
    }

    void SplatPass::writeSets(Cloud& c) {
        // Every binding needs a real object: descriptorBindingPartiallyBound is
        // not enabled on this device, so a VK_NULL_HANDLE anywhere in the set
        // is a validation error, not a "don't sample it".
        if (sceneHdrViews_.empty() || c.sets.empty() || envView_ == VK_NULL_HANDLE ||
            fogUbos_.empty() || cloudUbos_.empty() || lightsUbos_.empty())
            return;

        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VkDescriptorBufferInfo ubo{uboBuf_[f].handle, uboStride_ * c.slot, sizeof(SplatUboData)};
            const VkBuffer ssbo[13] = {
                    c.geom.handle, c.sh.handle, projBuf_.handle, countBuf_.handle,
                    offsetBuf_.handle, keyA_.handle, valA_.handle, keyB_.handle,
                    valB_.handle, rangeBuf_.handle, globalBuf_.handle, histBuf_.handle,
                    scanBuf_.handle};

            VkDescriptorBufferInfo bi[13]{};
            std::array<VkWriteDescriptorSet, kBindings> w{};
            for (uint32_t i = 0; i < kBindings; ++i) {
                w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet          = c.sets[f];
                w[i].dstBinding      = i;
                w[i].descriptorCount = 1;
            }
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[0].pBufferInfo    = &ubo;
            for (uint32_t i = 0; i < 13; ++i) {
                bi[i] = {ssbo[i], 0, VK_WHOLE_SIZE};
                w[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i + 1].pBufferInfo    = &bi[i];
            }

            VkDescriptorImageInfo hdr{VK_NULL_HANDLE, sceneHdrViews_[f], VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo dep{sampler_, depthViews_[f],
                                      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo mot{VK_NULL_HANDLE, motionViews_[f], VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo ids{sampler_, idsViews_[f],
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            w[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[14].pImageInfo     = &hdr;
            w[15].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[15].pImageInfo     = &dep;
            w[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[16].pImageInfo     = &mot;
            w[17].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[17].pImageInfo     = &ids;

            VkDescriptorBufferInfo ub[3]{{fogUbos_[f], 0, VK_WHOLE_SIZE},
                                         {cloudUbos_[f], 0, VK_WHOLE_SIZE},
                                         {lightsUbos_[f], 0, VK_WHOLE_SIZE}};
            for (uint32_t i = 0; i < 3; ++i) {
                w[18 + i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w[18 + i].pBufferInfo    = &ub[i];
            }
            VkDescriptorImageInfo env{envSampler_ ? envSampler_ : sampler_, envView_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            w[21].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[21].pImageInfo     = &env;

            VkDescriptorBufferInfo ind{indirectBuf_.handle, 0, VK_WHOLE_SIZE};
            w[22].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[22].pBufferInfo    = &ind;

            vkUpdateDescriptorSets(ctx_.device(), kBindings, w.data(), 0, nullptr);
        }
    }

    void SplatPass::syncClouds(const std::vector<CloudEntry>& clouds) {
        frameClouds_.clear();
        ++syncSerial_;
        if (clouds.empty()) return;

        // A cloud arriving or a cloud growing is a rare, load-time event; both
        // reallocate shared scratch and rewrite descriptor sets that other
        // frames may still be reading, so the device is drained first. This is
        // the one stall in the pass and it happens on asset load, not per frame.
        bool structural = false;
        uint32_t needSplats = maxSplats_;
        for (const auto& e : clouds) {
            if (!e.cloud) continue;
            if (!resident_.count(e.cloud)) structural = true;
            needSplats = std::max(needSplats, static_cast<uint32_t>(e.cloud->splatCount()));
        }
        uint32_t needEntries = entryBudget_;
        if (needSplats > maxSplats_) {
            structural  = true;
            needEntries = std::max<uint32_t>(
                    needEntries,
                    static_cast<uint32_t>(std::min<uint64_t>(
                            uint64_t(needSplats) * kEntriesPerSplat, kMaxEntries)));
        }

        // A frame that overflowed was TRUNCATED — splats are missing from
        // whichever tiles the key list ran out in, which looks like blocky
        // holes rather than like a resource problem. The expansion reports the
        // exact shortfall, so the fix is to resize to what the frame WANTED
        // (plus a quarter of headroom for the next camera angle) rather than to
        // climb a doubling ladder and pay a wrong frame for every rung.
        bool grewOnOverflow = false;
        if (const uint32_t over = lastOverflow(); over > 0 && !budgetCapped_) {
            grewOnOverflow = true;
            const uint64_t wanted = (uint64_t(entryBudget_) + over) * 5 / 4;
            uint32_t next = static_cast<uint32_t>(std::min<uint64_t>(wanted, kMaxEntries));
            if (next > entryBudget_) {
                needEntries = std::max(needEntries, next);
                structural  = true;
            }
            if (wanted > kMaxEntries) {
                budgetCapped_ = true;
                std::cerr << "[threepp] SplatPass: this frame's tile expansion wants "
                          << wanted << " (splat, tile) pairs, past the " << kMaxEntries
                          << " ceiling (" << (kMaxEntryBytes >> 20)
                          << " MB of sort buffers). Frames will be TRUNCATED — splats "
                             "missing from some tiles. Fewer or smaller splats, or a "
                             "coarser tile grid, is the fix."
                          << std::endl;
            }
        }

        if (structural) {
            check(vkDeviceWaitIdle(ctx_.device()), "vkDeviceWaitIdle(splat sync)");

            for (const auto& e : clouds) {
                if (!e.cloud || resident_.count(e.cloud)) continue;
                if (slotsUsed_ >= kMaxClouds) {
                    std::cerr << "[threepp] SplatPass: more than " << kMaxClouds
                              << " splat clouds in one scene; the extra ones are not drawn"
                              << std::endl;
                    break;
                }
                auto c = std::make_unique<Cloud>();
                c->slot = slotsUsed_++;
                uploadCloud(*c, *e.cloud);

                std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
                VkDescriptorSetAllocateInfo ai{};
                ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool     = descPool_;
                ai.descriptorSetCount = framesInFlight_;
                ai.pSetLayouts        = layouts.data();
                c->sets.resize(framesInFlight_);
                check(vkAllocateDescriptorSets(ctx_.device(), &ai, c->sets.data()),
                      "vkAllocateDescriptorSets(splat)");
                resident_.emplace(e.cloud, std::move(c));
            }

            if (needSplats > maxSplats_ || needEntries > entryBudget_) {
                // Worth one line on stderr, but only when an actual frame was
                // truncated: until the resize lands those frames ARE wrong
                // (splats missing from tiles), and a silent self-correction
                // leaves whoever saw them with no explanation. Sizing a cloud
                // for the first time is not that and says nothing.
                if (grewOnOverflow)
                    std::cerr << "[threepp] SplatPass: tile expansion budget "
                              << entryBudget_ << " -> " << needEntries
                              << " (splat, tile) pairs after a truncated frame"
                              << std::endl;
                allocateScratch(std::max(needSplats, maxSplats_), needEntries);
                // The overflow counters just consumed describe a frame that no
                // longer exists. Clearing them stops the next few syncs from
                // reading the same stale shortfall and growing again.
                for (uint32_t f = 0; f < framesInFlight_; ++f) {
                    VmaAllocationInfo ai{};
                    vmaGetAllocationInfo(ctx_.allocator(), debugBuf_[f].alloc, &ai);
                    if (ai.pMappedData) std::memset(ai.pMappedData, 0, kGlobalWords * 4);
                }
            }
            for (auto& kv : resident_) writeSets(*kv.second);
        }

        // Opt-in invariant report. The two numbers that matter are structural,
        // not cosmetic: a non-zero scan violation means the expansion ranges
        // overlap (splats overwriting each other), a non-zero order violation
        // means the radix sort dropped or duplicated entries. Both show up as
        // "some splats are just missing", which is indistinguishable by eye
        // from a culling bug — hence the assertion rather than the inference.
        if (const char* e = std::getenv("THREEPP_VK_SPLAT_CHECKSUM"); e && *e && *e != '0') {
            VmaAllocationInfo ai{};
            vmaGetAllocationInfo(ctx_.allocator(), debugBuf_[lastFrame_].alloc, &ai);
            invalidateHostReads(ctx_.allocator(), debugBuf_[lastFrame_].alloc);
            if (const auto* w = static_cast<const uint32_t*>(ai.pMappedData); w && w[2]) {
                std::cerr << "[splat] entries " << w[2] << " overflow " << w[3]
                          << " visible " << w[4] << " scanBad " << w[10]
                          << " orderBad " << w[11]
                          // The three hashes are what makes this an A/B
                          // instrument rather than only an invariant report: on
                          // a real scan the final FRAME is not reproducible
                          // run to run, so a changed picture proves nothing —
                          // these hash the splat pass's own output (sorted keys,
                          // sorted payload, composited pixels) and are the level
                          // at which a sort change must be byte-identical.
                          << " hashKey " << w[7] << " hashVal " << w[8]
                          << " hashColor " << w[9] << std::endl;
            }
        }

        for (const auto& e : clouds) {
            auto it = resident_.find(e.cloud);
            if (it == resident_.end() || it->second->count == 0) continue;
            it->second->lastSeen = syncSerial_;
            FrameCloud fc{};
            fc.cloud = it->second.get();
            std::memcpy(fc.model, e.model, sizeof(fc.model));
            fc.pLo = e.pLo;
            fc.pHi = e.pHi;
            fc.debugNonFinite = e.debugNonFinite;
            // Validate the submission list HERE rather than trusting it in the
            // shader: a range past the end of the cloud would read another
            // cloud's memory through the same descriptor, and a caller that
            // hands over more than kMaxRanges would silently overflow the UBO.
            fc.submitCount = it->second->count;
            if (!e.ranges.empty()) {

                uint32_t total = 0;
                for (const auto& [off, n] : e.ranges) {
                    if (fc.ranges.size() >= kMaxRanges) break;
                    if (n == 0 || off >= it->second->count) continue;
                    const uint32_t clamped = std::min(n, it->second->count - off);
                    fc.ranges.emplace_back(off, clamped);
                    total += clamped;
                }
                fc.submitCount = total;
                if (fc.ranges.empty() || total == 0) continue;// nothing submitted
            }
            frameClouds_.push_back(std::move(fc));
        }
    }

    void SplatPass::resize(uint32_t width, uint32_t height, const ResizeInputs& in) {
        if (!in.sceneHdrPerFrame || !in.depthPerFrame) return;

        const uint32_t tx = divUp(std::max(width, 1u), kTileW);
        const uint32_t ty = divUp(std::max(height, 1u), kTileH);
        const bool sameExtent = (width == width_ && height == height_);

        width_  = width;
        height_ = height;
        tilesX_ = tx;
        tilesY_ = ty;

        sceneHdrViews_.assign(in.sceneHdrPerFrame, in.sceneHdrPerFrame + framesInFlight_);
        depthViews_.assign(in.depthPerFrame, in.depthPerFrame + framesInFlight_);
        if (in.fogUbos)    fogUbos_.assign(in.fogUbos, in.fogUbos + framesInFlight_);
        if (in.cloudUbos)  cloudUbos_.assign(in.cloudUbos, in.cloudUbos + framesInFlight_);
        if (in.lightsUbos) lightsUbos_.assign(in.lightsUbos, in.lightsUbos + framesInFlight_);
        if (in.envView) {
            envView_    = in.envView;
            envSampler_ = in.envSampler;
            envMips_    = std::max(in.envMips, 1u);
        }
        if (in.motionPerFrame) motionViews_.assign(in.motionPerFrame, in.motionPerFrame + framesInFlight_);
        else                   motionViews_ = sceneHdrViews_;
        if (in.motionImages) motionImages_.assign(in.motionImages, in.motionImages + framesInFlight_);
        else                 motionImages_.assign(framesInFlight_, VK_NULL_HANDLE);
        if (in.idsPerFrame) idsViews_.assign(in.idsPerFrame, in.idsPerFrame + framesInFlight_);
        else                idsViews_ = depthViews_;

        if (!sameExtent || rangeBuf_.handle == VK_NULL_HANDLE) {
            destroyBuffer(ctx_.allocator(), rangeBuf_);
            rangeBuf_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                     VkDeviceSize(tx) * ty * 8,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VMA_MEMORY_USAGE_AUTO);
            ctx_.setObjectName(rangeBuf_.handle, "splat.tileRange");
        }
        if (maxSplats_ == 0) allocateScratch(1, kEntriesPerSplat);

        for (auto& kv : resident_) writeSets(*kv.second);
    }

    void SplatPass::barrierIndirect(VkCommandBuffer cb) const {
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        mb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        VkDependencyInfo di{};
        di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }

    void SplatPass::barrier(VkCommandBuffer cb) const {
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                           VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo di{};
        di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }

    void SplatPass::recordScan(VkCommandBuffer cb, uint32_t n, uint32_t mode0) {
        if (n == 0) return;

        struct Level { uint32_t count, srcOff, sumOff, mode; };
        std::vector<Level> levels;
        uint32_t cur = n, srcOff = 0, regionOff = 0, mode = mode0;
        while (true) {
            const uint32_t nb = divUp(cur, kScanBlock);
            levels.push_back({cur, srcOff, regionOff, mode});
            if (nb == 1) break;
            srcOff    = regionOff;
            regionOff += nb;
            cur       = nb;
            mode      = 1;// scanScratch, in place
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, scanPipe_);
        for (const auto& l : levels) {
            SplatPc pc{l.count, l.srcOff, l.srcOff, l.sumOff, l.mode, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, divUp(l.count, kScanBlock), 1, 1);
            barrier(cb);
        }
        // Walk back, skipping the top level: its own block-sum array has a
        // single entry, whose exclusive scan is 0.
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, scanAddPipe_);
        for (int j = static_cast<int>(levels.size()) - 2; j >= 0; --j) {
            const auto& l = levels[static_cast<size_t>(j)];
            SplatPc pc{l.count, l.srcOff, l.srcOff, l.sumOff, l.mode, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, divUp(l.count, kScanThreads), 1, 1);
            barrier(cb);
        }
    }

    void SplatPass::record(VkCommandBuffer cb, uint32_t frame, const RecordParams& p) {
        if (frameClouds_.empty() || !valid()) return;
        lastFrame_ = frame;

        // Debug hashes cost a full extra pass over the key list and an atomic
        // per composited pixel, so they are opt-in: the determinism test asks
        // through setSplatDebugChecksum, a human debugging by hand through the
        // environment variable.
        const char* cse = std::getenv("THREEPP_VK_SPLAT_CHECKSUM");
        const bool checksum = p.checksum || (cse && *cse && *cse != '0');

        // Per-stage timestamps go to the FIRST drawn cloud only — the slots are
        // one pair per stage per frame, and a second writer would both violate
        // VUID-vkCmdWriteTimestamp2-None-03864 and report the wrong interval.
        bool stagesWritten = false;// slots for this frame already used
        bool timing        = false;// this cloud's stages go to the pool
        const auto stageBegin = [&](TimingPass tp) { if (timing) p.timings->begin(cb, tp, frame); };
        const auto stageEnd   = [&](TimingPass tp) { if (timing) p.timings->end(cb, tp, frame); };

        for (const auto& fc : frameClouds_) {
            Cloud& c = *fc.cloud;
            if (c.count == 0) continue;
            // Everything per-splat runs over what this frame SUBMITS; the
            // resident count only sizes the buffers. Equal unless the frame
            // handed over a range list.
            const uint32_t nSubmit = fc.submitCount;
            if (nSubmit == 0) continue;
            timing = p.timings && !stagesWritten;

            // ── per-cloud UBO slot ───────────────────────────────────────────
            SplatUboData u{};
            // modelView = view * model, column-major throughout.
            for (int col = 0; col < 4; ++col)
                for (int row = 0; row < 4; ++row) {
                    float s = 0.f;
                    for (int k = 0; k < 4; ++k) s += p.view[k * 4 + row] * fc.model[col * 4 + k];
                    u.modelView[col * 4 + row] = s;
                }
            std::memcpy(u.proj, p.proj, sizeof(u.proj));
            std::memcpy(u.projInv, p.projInverse, sizeof(u.projInv));
            std::memcpy(u.model, fc.model, sizeof(u.model));
            std::memcpy(u.prevVPfromView, p.prevVPfromView, sizeof(u.prevVPfromView));
            std::memcpy(u.camWorld, p.camWorld, sizeof(u.camWorld));
            u.jitterClip[0] = p.jitterClip[0];
            u.jitterClip[1] = p.jitterClip[1];
            u.envMipCount   = envMips_;
            u.camPosWs[0] = p.camPos[0];
            u.camPosWs[1] = p.camPos[1];
            u.camPosWs[2] = p.camPos[2];
            u.camFwdWs[0] = p.camFwd[0];
            u.camFwdWs[1] = p.camFwd[1];
            u.camFwdWs[2] = p.camFwd[2];
            u.viewport[0] = static_cast<float>(width_);
            u.viewport[1] = static_cast<float>(height_);
            u.focal[0]    = 0.5f * u.viewport[0] * p.proj[0];
            u.focal[1]    = 0.5f * u.viewport[1] * p.proj[5];
            u.percentile[0] = fc.pLo;
            u.percentile[1] = fc.pHi;
            u.nearPlane   = p.nearPlane;
            u.preExposure = p.preExposure;
            // The SUBMITTED total, not the resident total: every stage after
            // project indexes compactly, so this is the only count they need.
            u.splatCount  = fc.submitCount;
            u.rangeCount  = static_cast<uint32_t>(fc.ranges.size());
            {
                uint32_t first = 0, k = 0;
                for (const auto& [off, n] : fc.ranges) {
                    u.ranges[k * 2 + 0] = first;// compact start, ascending
                    u.ranges[k * 2 + 1] = off;  // source base
                    first += n;
                    ++k;
                }
            }
            u.shCoeffs    = c.shCoeffs;
            u.shDegree    = c.shDegree;
            u.tilesX      = tilesX_;
            u.tilesY      = tilesY_;
            u.tileBits    = tileBitsFor(tilesX_ * tilesY_);
            u.depthBits   = 32u - u.tileBits;
            u.budget      = entryBudget_;
            u.flags       = (p.orthographic ? kSplatFlagOrtho : 0u) |
                      (p.depthTest ? kSplatFlagDepthTest : 0u) |
                      (fc.debugNonFinite ? kSplatFlagDebugNaN : 0u) |
                      (checksum ? kSplatFlagChecksum : 0u) |
                      (p.bgIsSolidColor ? kSplatFlagBgSolid : 0u) |
                      (p.motionVectors ? kSplatFlagMotion : 0u) |
                      (p.fog ? kSplatFlagFog : 0u);

            VmaAllocationInfo ui{};
            vmaGetAllocationInfo(ctx_.allocator(), uboBuf_[frame].alloc, &ui);
            std::memcpy(static_cast<uint8_t*>(ui.pMappedData) + uboStride_ * c.slot, &u, sizeof(u));
            flushHostWrites(ctx_.allocator(), uboBuf_[frame].alloc,
                            uboStride_ * c.slot, sizeof(u));

            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1,
                                    &c.sets[frame], 0, nullptr);

            // ── clear the frame's GPU-owned state ────────────────────────────
            // minDistBits starts at +inf's bit pattern so the atomicMin has
            // something to lose to; everything else at zero.
            uint32_t g0[kGlobalWords] = {};
            g0[0] = 0x7F7FFFFFu;// FLT_MAX
            g0[1] = 0u;
            vkCmdUpdateBuffer(cb, globalBuf_.handle, 0, sizeof(g0), g0);
            vkCmdFillBuffer(cb, rangeBuf_.handle, 0, VkDeviceSize(tilesX_) * tilesY_ * 8, 0u);
            barrier(cb);

            const uint32_t radixBlocks = divUp(entryBudget_, kRadixBlock);
            SplatPc pc{};

            // ── project + cull + tile counts ─────────────────────────────────
            stageBegin(TP_SplatProject);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, projectPipe_);
            pc = {nSubmit, 0, 0, 0, 0, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, divUp(nSubmit, 256), 1, 1);
            barrier(cb);

            if (checksum) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, checksumPipe_);
                pc = {nSubmit, 0, 0, 0, 0, /*mode*/ 0, 0, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, divUp(nSubmit, 256), 1, 1);
                barrier(cb);
            }

            // ── prefix sum over the per-splat tile counts ────────────────────
            recordScan(cb, nSubmit, 0);

            if (checksum) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, checksumPipe_);
                pc = {nSubmit, 0, 0, 0, 0, /*mode*/ 2, 0, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, divUp(nSubmit, 256), 1, 1);
                barrier(cb);
            }

            // ── deterministic expansion ──────────────────────────────────────
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, expandPipe_);
            pc = {nSubmit, 0, 0, 0, 0, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, divUp(nSubmit, 256), 1, 1);
            barrier(cb);

            // ── size the sort from the data ──────────────────────────────────
            // One thread reads g.entryCount and writes the group counts the
            // radix and range dispatches below consume. Everything after the
            // expansion used to run over entryBudget_ instead, which measured
            // 7.9 ms of sorting on a frame with nothing on screen.
            // THREEPP_VK_SPLAT_NOINDIRECT restores the worst-case dispatches.
            // Same escape-hatch shape as NOMOTION/NOFOG, and it earns its keep
            // twice: it is how the indirect path was A/B'd for byte-identical
            // output on a real scan, and it is the first thing to try if some
            // other driver disagrees about vkCmdDispatchIndirect.
            const char* nie = std::getenv("THREEPP_VK_SPLAT_NOINDIRECT");
            const bool indirectDispatch = !(nie && *nie && *nie != '0');

            if (indirectDispatch) {

                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, indirectPipe_);
                pc = {entryBudget_, 0, 0, 0, 0, 0, 0, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, 1, 1, 1);
                barrierIndirect(cb);
            }

            // ── 8 x 4-bit LSD radix, ping-ponging A <-> B ────────────────────
            // The histogram is indexed [bin][block] with pc.arg1 = the WORST-CASE
            // block count, and the scan below still runs the worst-case extent
            // (recordScan's per-level offsets are host arithmetic). Only the
            // dispatches shrink — which is exactly why the histogram has to be
            // ZEROED first: the tail blocks used to run and write zero counts,
            // and now they do not run at all, so their bins would carry LAST
            // FRAME'S values into a scan that still reads them. Every offset
            // after the live region would be wrong, and the picture with it.
            stageEnd(TP_SplatProject);
            stageBegin(TP_SplatSort);
            for (uint32_t pass = 0; pass < kRadixPasses; ++pass) {
                const uint32_t shift = pass * 4;
                const uint32_t side  = pass & 1u;// 0: A->B, 1: B->A

                vkCmdFillBuffer(cb, histBuf_.handle, 0,
                                VkDeviceSize(kRadixBins) * radixBlocks * 4, 0u);
                barrier(cb);

                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, histPipe_);
                pc = {entryBudget_, 0, 0, 0, shift, radixBlocks, side, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                if (indirectDispatch) vkCmdDispatchIndirect(cb, indirectBuf_.handle, 0);
                else                  vkCmdDispatch(cb, radixBlocks, 1, 1);
                barrier(cb);

                recordScan(cb, kRadixBins * radixBlocks, 2);

                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, scatterPipe_);
                pc = {entryBudget_, 0, 0, 0, shift, radixBlocks, side, 0};
                vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                if (indirectDispatch) vkCmdDispatchIndirect(cb, indirectBuf_.handle, 0);
                else                  vkCmdDispatch(cb, radixBlocks, 1, 1);
                barrier(cb);
            }

            stageEnd(TP_SplatSort);

            // ── tile ranges ──────────────────────────────────────────────────
            stageBegin(TP_SplatRaster);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rangePipe_);
            pc = {entryBudget_, 0, 0, 0, 0, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            // rangeBuf_ was zero-filled at the top of this cloud, so the tiles
            // this dispatch no longer reaches are empty rather than stale.
            if (indirectDispatch) vkCmdDispatchIndirect(cb, indirectBuf_.handle, 3 * sizeof(uint32_t));
            else                  vkCmdDispatch(cb, divUp(entryBudget_, 256), 1, 1);
            barrier(cb);

            if (checksum) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, checksumPipe_);
                for (uint32_t mode : {1u, 3u}) {
                    pc = {entryBudget_, 0, 0, 0, 0, mode, 0, 0};
                    vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cb, divUp(entryBudget_, 256), 1, 1);
                    barrier(cb);
                }
            }

            // ── composite ────────────────────────────────────────────────────
            // The motion attachment rests in SHADER_READ_ONLY_OPTIMAL (the
            // G-buffer render pass's finalLayout) and this pass writes it as a
            // STORAGE image, which only works in GENERAL. Flip it, dispatch,
            // flip it back — leaving it in GENERAL would be a silent lie to
            // every consumer downstream that samples it in the layout the
            // render pass promised.
            const bool flipMotion = p.motionVectors && motionImages_[frame] != VK_NULL_HANDLE;
            auto motionLayout = [&](VkImageLayout from, VkImageLayout to,
                                    VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess) {
                VkImageMemoryBarrier2 b{};
                b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = srcAccess;
                b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                b.dstAccessMask = dstAccess;
                b.oldLayout     = from;
                b.newLayout     = to;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image         = motionImages_[frame];
                b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
                VkDependencyInfo di{};
                di.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                di.imageMemoryBarrierCount  = 1;
                di.pImageMemoryBarriers     = &b;
                vkCmdPipelineBarrier2(cb, &di);
            };
            if (flipMotion)
                motionLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rasterPipe_);
            pc = {0, 0, 0, 0, 0, 0, 0, 0};
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, tilesX_, tilesY_, 1);
            barrier(cb);
            stageEnd(TP_SplatRaster);
            stagesWritten = stagesWritten || timing;

            if (flipMotion)
                motionLayout(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            VkBufferCopy copy{0, 0, kGlobalWords * sizeof(uint32_t)};
            vkCmdCopyBuffer(cb, globalBuf_.handle, debugBuf_[frame].handle, 1, &copy);
        }
    }

    void SplatPass::readDebug(uint64_t out[4]) const {
        out[0] = out[1] = out[2] = out[3] = 0;
        if (debugBuf_.empty()) return;
        // The device is drained so the last frame's copy has provably landed;
        // reading the slot that frame recorded into is then unambiguous.
        check(vkDeviceWaitIdle(ctx_.device()), "vkDeviceWaitIdle(splat readDebug)");
        VmaAllocationInfo ai{};
        vmaGetAllocationInfo(ctx_.allocator(), debugBuf_[lastFrame_].alloc, &ai);
        invalidateHostReads(ctx_.allocator(), debugBuf_[lastFrame_].alloc);
        const auto* w = static_cast<const uint32_t*>(ai.pMappedData);
        if (!w) return;
        out[0] = w[7]; // hashKey
        out[1] = w[8]; // hashVal
        out[2] = w[9]; // hashColor
        out[3] = w[2]; // entryCount
    }

    uint32_t SplatPass::lastOverflow() const {
        if (debugBuf_.empty()) return 0;
        uint32_t worst = 0;
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            VmaAllocationInfo ai{};
            vmaGetAllocationInfo(ctx_.allocator(), debugBuf_[f].alloc, &ai);
            if (const auto* w = static_cast<const uint32_t*>(ai.pMappedData)) worst = std::max(worst, w[3]);
        }
        return worst;
    }

}// namespace threepp::vulkan
