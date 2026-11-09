#include "threepp/renderers/vulkan/PathTracer.hpp"

#include "threepp/renderers/vulkan/DescriptorShadow.hpp"
#include "threepp/renderers/vulkan/ParticleFieldPass.hpp"// OverlayGradeGpu
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"// kMaxMaterialTextures
#include "threepp/renderers/vulkan/shaders/path_trace.comp.photons.spv.h"
#include "threepp/renderers/vulkan/shaders/path_trace.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/path_trace_resolve.comp.spv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace threepp::vulkan {

    namespace {

        // path_trace.comp's push block, in its order.
        struct PushConstants {
            uint32_t width;
            uint32_t height;
            uint32_t sampleIndex;
            uint32_t samples;
            uint32_t maxBounces;
            uint32_t flags;
            uint32_t emissiveCount;
            float    emissiveTotalPower;
            uint32_t envMipCount;
            float    fireflyClamp;
            float    sunTanHalfAngle;
            uint32_t seed;
            float    oceanFineTileSize;
            float    starIntensity;
            float    volDensity;
            float    volAniso;
            float    lensK[4];
            float    lensRadial[4];
            float    lensTangential[2];
            float    lensOverscan;
            float    lensRadius;
            float    focusDistance;
            float    aspectScale;
            uint32_t rows;// first row | count << 16
            float    domeRadius;
        };
        static_assert(sizeof(PushConstants) == 128);// the least maxPushConstantsSize a device has

        // path_trace_resolve.comp's: the display transform and the post
        // stack's white balance and grade (OverlayGrade).
        struct ResolvePush {
            uint32_t        width;
            uint32_t        height;
            uint32_t        toneMapping;
            float           exposure;
            OverlayGradeGpu grade;
        };
        static_assert(sizeof(ResolvePush) == 112);

        constexpr uint32_t kNumBindings = 19;// 0-9, then 67-69, 58, 13, 78, 79, 10 and 11

        // The photon grid's header (path_trace.comp's PhotonGridBuf), ahead of
        // its buckets' count, start and end arrays. The shader's set-up pass
        // fills the sun's fields (sunScale, region, sunLight) and may widen
        // the radius.
        struct PhotonGridHeader {
            uint32_t stored;
            uint32_t cursor;
            uint32_t capacity;
            uint32_t hashMask;
            float    radius;
            float    cellInv;
            float    powerScale;
            float    sunScale;
            float    region[4];
            float    shrink;
            uint32_t sunCount;
            uint32_t sunLight;
            uint32_t pad;
        };
        static_assert(sizeof(PhotonGridHeader) == 64);
        constexpr VkDeviceSize kPhotonBytes = 32;// path_trace.comp's Photon

        VkPipeline makeComputePipeline(VulkanContext& ctx, const uint32_t* spv, size_t bytes,
                                       VkPipelineLayout layout, const char* what) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = bytes;
            smci.pCode    = spv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx.device(), &smci, nullptr, &mod), what);
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = mod;
            stage.pName  = "main";
            VkComputePipelineCreateInfo cpci{};
            cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage  = stage;
            cpci.layout = layout;
            // THREEPP_VULKAN_PIPELINE_STATS: the driver's statistics (register
            // count, spills), past the cache, which keeps none.
            const bool stats = ctx.pipelineStatsEnabled();
            if (stats) cpci.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
            VkPipeline pipe = VK_NULL_HANDLE;
            check(vkCreateComputePipelines(ctx.device(), stats ? VK_NULL_HANDLE : ctx.pipelineCache(), 1, &cpci,
                                           nullptr, &pipe), what);
            vkDestroyShaderModule(ctx.device(), mod, nullptr);
            if (stats) {
                if (const int64_t n = ctx.pipelineStat(pipe, "Register Count"); n >= 0)
                    std::fprintf(stderr, "[path-trace] %s: %lld registers\n", what, static_cast<long long>(n));
                ctx.dumpPipelineStats(pipe, what);
            }
            return pipe;
        }

    }// namespace

    PathTracer::PathTracer(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        if (!ctx_.rayQuerySupported()) return;
        for (int i = 0; i < 2; ++i) {
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter    = i == 0 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            sci.minFilter    = sci.magFilter;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod       = 0.f;
            check(vkCreateSampler(ctx_.device(), &sci, nullptr, i == 0 ? &nearestSampler_ : &linearSampler_),
                  "vkCreateSampler(pathTracer)");
        }
        createPipeline();
        createResolvePipeline();
        createDescriptorPool();
        boundAccum_.assign(framesInFlight_, 0);
        boundPhotons_.assign(framesInFlight_, 0);
        for (auto& b : photonStandIn_)
            b = createBuffer(ctx_.allocator(), ctx_.device(), 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    }

    PathTracer::~PathTracer() {
        VkDevice d = ctx_.device();
        freeRetired(0, /*all=*/true);
        destroyBuffer(ctx_.allocator(), accum_);
        destroyBuffer(ctx_.allocator(), photonBuf_);
        destroyBuffer(ctx_.allocator(), photonGrid_);
        for (auto& b : photonStandIn_) destroyBuffer(ctx_.allocator(), b);
        for (auto s : sets_) forgetDescriptorSet(s);
        if (resolveSet_) forgetDescriptorSet(resolveSet_);
        if (descPool_) vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (pipe_) vkDestroyPipeline(d, pipe_, nullptr);
        if (pipePhotons_) vkDestroyPipeline(d, pipePhotons_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_) vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (resolvePipe_) vkDestroyPipeline(d, resolvePipe_, nullptr);
        if (resolvePipeLayout_) vkDestroyPipelineLayout(d, resolvePipeLayout_, nullptr);
        if (resolveLayout_) vkDestroyDescriptorSetLayout(d, resolveLayout_, nullptr);
        if (nearestSampler_) vkDestroySampler(d, nearestSampler_, nullptr);
        if (linearSampler_) vkDestroySampler(d, linearSampler_, nullptr);
    }

    void PathTracer::createPipeline() {
        VkDevice d = ctx_.device();

        VkDescriptorSetLayoutBinding b[kNumBindings]{};
        auto set = [&](uint32_t i, VkDescriptorType t) {
            b[i].binding         = i;
            b[i].descriptorType  = t;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        };
        // The rest sit at the deferred set's binding numbers, where the
        // includes the two share declare them: the ParticleField density
        // volumes (67-69, particle_density.glsl), the cloud block (58,
        // cloud_ubo.glsl), the ocean's fine-cascade height (13) and its wake
        // field and ripples (78, 79, ocean_wake.glsl).
        auto setAt = [&](uint32_t slot, uint32_t binding, VkDescriptorType t, uint32_t count) {
            b[slot].binding         = binding;
            b[slot].descriptorType  = t;
            b[slot].descriptorCount = count;
            b[slot].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        };
        set(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);            // camera (the primary view's)
        set(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);            // lights
        set(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);    // env (PMREM)
        set(3, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);// TLAS
        set(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);            // MaterialDesc[]
        set(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);            // GeometryDesc[]
        set(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);    // bindless material textures...
        b[6].descriptorCount = kMaxMaterialTextures;          // ...fixed-size array
        set(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);            // EmTri[]
        set(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);            // accumulation
        set(9, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);            // fog (the air medium)
        setAt(10, 67, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxDensityFields);// density volumes
        setAt(11, 68, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);                         // their boxes + medium
        setAt(12, 69, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxDensityFields);// r16f mirrors
        setAt(13, 58, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1);                         // clouds (GpuCloudUbo)
        setAt(14, 13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1);                 // ocean fine-cascade height
        setAt(15, 78, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1);                 // ocean wake field
        setAt(16, 79, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1);                 // its ripples
        setAt(17, 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);                         // the photons
        setAt(18, 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);                         // their grid

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = kNumBindings;
        dlci.pBindings    = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(pathTracer)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset     = 0;
        pc.size       = sizeof(PushConstants);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(pathTracer)");

        refl_ = reflectSpirvBindings(kPathTraceCompSpv);
        pipe_ = makeComputePipeline(ctx_, kPathTraceCompSpv, sizeof(kPathTraceCompSpv), pipeLayout_,
                                    "vkCreateComputePipelines(path_trace)");
    }

    void PathTracer::createResolvePipeline() {
        VkDevice d = ctx_.device();
        VkDescriptorSetLayoutBinding b[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2;
        dlci.pBindings    = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &resolveLayout_),
              "vkCreateDescriptorSetLayout(pathTracer.resolve)");
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size       = sizeof(ResolvePush);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &resolveLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &resolvePipeLayout_),
              "vkCreatePipelineLayout(pathTracer.resolve)");
        resolvePipe_ = makeComputePipeline(ctx_, kPathTraceResolveCompSpv, sizeof(kPathTraceResolveCompSpv),
                                           resolvePipeLayout_, "vkCreateComputePipelines(path_trace_resolve)");
    }

    void PathTracer::createDescriptorPool() {
        VkDescriptorPoolSize sizes[4]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = framesInFlight_ * 5;// camera + lights + fog + density + clouds
        sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = framesInFlight_ * (4 + kMaxMaterialTextures + 2 * kMaxDensityFields);// env + ocean x3 + bindless + volumes
        sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[2].descriptorCount = framesInFlight_ * 6 + 2;// materials + geometry + emissive + accumulation + photons x2; the resolve's two
        sizes[3].type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        sizes[3].descriptorCount = framesInFlight_ * 1;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_ + 1;
        dpci.poolSizeCount = 4;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(pathTracer)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descPool_;
        ai.descriptorSetCount = framesInFlight_;
        ai.pSetLayouts        = layouts.data();
        sets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(ctx_.device(), &ai, sets_.data()),
              "vkAllocateDescriptorSets(pathTracer)");

        VkDescriptorSetAllocateInfo ri{};
        ri.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ri.descriptorPool     = descPool_;
        ri.descriptorSetCount = 1;
        ri.pSetLayouts        = &resolveLayout_;
        check(vkAllocateDescriptorSets(ctx_.device(), &ri, &resolveSet_),
              "vkAllocateDescriptorSets(pathTracer.resolve)");
    }

    void PathTracer::rewriteDescriptors(const DescriptorWriteInputs& in, int onlyFrame) {
        if (sets_.empty()) return;
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            if (onlyFrame >= 0 && f != static_cast<uint32_t>(onlyFrame)) continue;
            VkDescriptorBufferInfo camInfo{};
            camInfo.buffer = in.cameraUbo[f];
            camInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo lightInfo{};
            lightInfo.buffer = in.lightsUbo[f];
            lightInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorImageInfo envInfo{};
            envInfo.sampler     = in.envSampler;
            envInfo.imageView   = in.envView;
            envInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkDescriptorBufferInfo matInfo{};
            matInfo.buffer = in.materialBuf[f];
            matInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo geomInfo{};
            geomInfo.buffer = in.geomDescBuf[f];
            geomInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo emInfo{};
            emInfo.buffer = in.emissiveTriBuf[f];
            emInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo fogInfo{};
            fogInfo.buffer = in.fogUbo[f];
            fogInfo.range  = VK_WHOLE_SIZE;

            VkAccelerationStructureKHR tlasLocal = in.tlas;
            VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
            asInfo.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            asInfo.accelerationStructureCount = 1;
            asInfo.pAccelerationStructures    = &tlasLocal;

            VkWriteDescriptorSet w[16]{};
            auto setw = [&](int n, uint32_t bind, VkDescriptorType t,
                            const VkDescriptorImageInfo* img, const VkDescriptorBufferInfo* buf) {
                w[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstSet          = sets_[f];
                w[n].dstBinding      = bind;
                w[n].descriptorCount = 1;
                w[n].descriptorType  = t;
                w[n].pImageInfo      = img;
                w[n].pBufferInfo     = buf;
            };
            setw(0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfo);
            setw(1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &lightInfo);
            setw(2, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr);
            setw(3, 3, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr);
            w[3].pNext = &asInfo;
            setw(4, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo);
            setw(5, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &geomInfo);
            w[6].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[6].dstSet          = sets_[f];
            w[6].dstBinding      = 6;
            w[6].descriptorCount = in.materialTexCount;
            w[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[6].pImageInfo      = in.materialTex;
            setw(7, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &emInfo);
            setw(8, 9, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &fogInfo);
            // Binding 8, the session's accumulator, is prepare()'s.
            uint32_t nw = 9;
            std::array<VkDescriptorImageInfo, kMaxDensityFields> pdInfos{};
            std::array<VkDescriptorImageInfo, kMaxDensityFields> pdLinInfos{};
            VkDescriptorBufferInfo pdUboInfo{};
            if (in.particleDensity && in.particleDensityLin && in.particleDensityUbo) {
                for (uint32_t i = 0; i < kMaxDensityFields; ++i) {
                    pdInfos[i]    = {nearestSampler_, in.particleDensity[i], VK_IMAGE_LAYOUT_GENERAL};
                    pdLinInfos[i] = {linearSampler_, in.particleDensityLin[i], VK_IMAGE_LAYOUT_GENERAL};
                }
                pdUboInfo.buffer = in.particleDensityUbo[f];
                pdUboInfo.range  = VK_WHOLE_SIZE;
                w[9].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[9].dstSet          = sets_[f];
                w[9].dstBinding      = 67;
                w[9].descriptorCount = kMaxDensityFields;
                w[9].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[9].pImageInfo      = pdInfos.data();
                setw(10, 68, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &pdUboInfo);
                w[11].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[11].dstSet          = sets_[f];
                w[11].dstBinding      = 69;
                w[11].descriptorCount = kMaxDensityFields;
                w[11].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[11].pImageInfo      = pdLinInfos.data();
                nw = 12;
            }
            VkDescriptorBufferInfo cloudInfo{};
            if (in.cloudUbo) {
                cloudInfo.buffer = in.cloudUbo[f];
                cloudInfo.range  = in.cloudRange;
                setw(static_cast<int>(nw), 58, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &cloudInfo);
                ++nw;
            }
            VkDescriptorImageInfo oceanInfo{};
            if (in.oceanFineView != VK_NULL_HANDLE && in.oceanFineSampler != VK_NULL_HANDLE) {
                oceanInfo = {in.oceanFineSampler, in.oceanFineView, VK_IMAGE_LAYOUT_GENERAL};
                setw(static_cast<int>(nw), 13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &oceanInfo, nullptr);
                ++nw;
            }
            VkDescriptorImageInfo wakeInfo{}, rippleInfo{};
            if (in.oceanWakeView != VK_NULL_HANDLE && in.oceanWakeSampler != VK_NULL_HANDLE &&
                in.oceanRippleView != VK_NULL_HANDLE && in.oceanRippleSampler != VK_NULL_HANDLE) {
                wakeInfo   = {in.oceanWakeSampler, in.oceanWakeView, VK_IMAGE_LAYOUT_GENERAL};
                rippleInfo = {in.oceanRippleSampler, in.oceanRippleView, VK_IMAGE_LAYOUT_GENERAL};
                setw(static_cast<int>(nw), 78, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &wakeInfo, nullptr);
                ++nw;
                setw(static_cast<int>(nw), 79, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rippleInfo, nullptr);
                ++nw;
            }
            updateDescriptorSets(ctx_.device(), nw, w, 0, nullptr);
        }
    }

    void PathTracer::rewriteEmissive(uint32_t frame, VkBuffer emissiveTriBuf) {
        if (sets_.empty() || frame >= framesInFlight_) return;
        VkDescriptorBufferInfo emInfo{};
        emInfo.buffer = emissiveTriBuf;
        emInfo.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = sets_[frame];
        w.dstBinding      = 7;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &emInfo;
        updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
    }

    void PathTracer::bindAccumulator(uint32_t frame) {
        if (sets_.empty() || accum_.handle == VK_NULL_HANDLE) return;
        if (boundAccum_[frame] == accumSerial_) return;
        VkDescriptorBufferInfo info{};
        info.buffer = accum_.handle;
        info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = sets_[frame];
        w.dstBinding      = 8;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &info;
        updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        boundAccum_[frame] = accumSerial_;
    }

    void PathTracer::bindPhotons(uint32_t frame) {
        if (sets_.empty()) return;
        // photonSerial_ 1 is the stand-ins, until a scene with emitters and
        // glass allocates the photons.
        if (boundPhotons_[frame] == photonSerial_) return;
        const bool             real = photonBuf_.handle != VK_NULL_HANDLE;
        VkDescriptorBufferInfo info[2]{};
        info[0].buffer = real ? photonBuf_.handle : photonStandIn_[0].handle;
        info[1].buffer = real ? photonGrid_.handle : photonStandIn_[1].handle;
        info[0].range = info[1].range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = sets_[frame];
            w[i].dstBinding      = 10 + i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo     = &info[i];
        }
        updateDescriptorSets(ctx_.device(), 2, w, 0, nullptr);
        boundPhotons_[frame] = photonSerial_;
    }

    void PathTracer::freeRetired(uint64_t frameSerial, bool all) {
        for (auto it = retired_.begin(); it != retired_.end();) {
            if (all || frameSerial >= it->second + framesInFlight_ + 1) {
                destroyBuffer(ctx_.allocator(), it->first);
                it = retired_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void PathTracer::begin(const Params& params) {
        params_                 = params;
        params_.samplesPerFrame = std::max(1u, params_.samplesPerFrame);
        params_.maxBounces      = std::min(params_.maxBounces, 64u);
        params_.height          = std::min(params_.height, 65535u);
        params_.domePort        = std::max(params_.domePort, 0.f);
        active_       = true;
        resetPending_ = true;
        samples_      = 0;
        rowCursor_    = 0;
        ++seed_;
    }

    void PathTracer::prepare(uint32_t frame, uint32_t renderWidth, uint32_t renderHeight, uint64_t frameSerial,
                             bool photons, bool sunPhotons) {
        if (!available() || !active_) return;
        freeRetired(frameSerial, false);
        if (renderWidth == 0 || renderHeight == 0) return;
        const bool     own    = params_.width > 0 && params_.height > 0;
        const uint32_t width  = own ? params_.width : renderWidth;
        const uint32_t height = std::min(own ? params_.height : renderHeight, 65535u);
        renderWidth_  = renderWidth;
        renderHeight_ = renderHeight;
        // A band no larger than the frame: the dispatch the samples-per-frame
        // count was chosen against.
        bandRows_ = std::clamp<uint32_t>(uint32_t((uint64_t(renderWidth) * renderHeight) / width), 1u, height);
        if (width != width_ || height != height_ || accum_.handle == VK_NULL_HANDLE) {
            if (accum_.handle != VK_NULL_HANDLE) retired_.emplace_back(accum_, frameSerial);
            accum_ = Buffer{};
            const VkDeviceSize bytes = VkDeviceSize(width) * height * 16;
            accum_ = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            width_  = width;
            height_ = height;
            ++accumSerial_;
            resetPending_ = true;
            samples_      = 0;
            rowCursor_    = 0;
        }
        bindAccumulator(frame);
        // The photons: each kind half a photon a pixel each pass (the
        // emitters' at most a million, the sun's three), a power of two of
        // buckets at least as many as both, and the pipeline that builds and
        // gathers them.
        if ((photons || sunPhotons) && pipePhotons_ == VK_NULL_HANDLE) {
            reflPhotons_ = reflectSpirvBindings(kPathTracePhotonsCompSpv);
            pipePhotons_ = makeComputePipeline(ctx_, kPathTracePhotonsCompSpv, sizeof(kPathTracePhotonsCompSpv),
                                               pipeLayout_, "vkCreateComputePipelines(path_trace photons)");
        }
        photonsWanted_    = photons;
        sunPhotonsWanted_ = sunPhotons;
        if (photons || sunPhotons) {
            const uint32_t half = uint32_t((uint64_t(width_) * height_) / 2);
            const uint32_t ne   = photons ? std::clamp<uint32_t>(half, 16384u, 1u << 20) : 0u;
            const uint32_t ns   = sunPhotons ? std::clamp<uint32_t>(half, 65536u, 3u << 20) : 0u;
            const uint32_t cap  = ne + ns;
            if (cap != photonCap_ || photonBuf_.handle == VK_NULL_HANDLE) {
                if (photonBuf_.handle != VK_NULL_HANDLE) retired_.emplace_back(photonBuf_, frameSerial);
                if (photonGrid_.handle != VK_NULL_HANDLE) retired_.emplace_back(photonGrid_, frameSerial);
                uint32_t buckets = 4096;
                while (buckets < cap) buckets <<= 1;
                photonBuf_  = createBuffer(ctx_.allocator(), ctx_.device(), VkDeviceSize(cap) * 2 * kPhotonBytes,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
                photonGrid_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                           sizeof(PhotonGridHeader) + VkDeviceSize(buckets) * 3 * 4,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
                photonCap_     = cap;
                photonBuckets_ = buckets;
                ++photonSerial_;
                photonsLive_ = emitLive_ = sunLive_ = false;// a pass begun on the old ones gathers nothing more
            }
            photonEmitCount_ = ne;
            photonSunCount_  = ns;
        }
        bindPhotons(frame);
    }

    void PathTracer::declare(rg::RenderGraph& graph, rg::PassBuilder& pass, uint32_t frame) const {
        if (sets_.empty()) return;
        // The photon variant also writes the photons (bindings 10 and 11),
        // which the other leaves out of its code.
        const bool photons = pipePhotons_ != VK_NULL_HANDLE && (photonsWanted_ || sunPhotonsWanted_ || photonsLive_);
        declareDescriptorSet(graph, pass, sets_[frame], 0, photons ? reflPhotons_ : refl_, rg::kCompute);
    }

    void PathTracer::recordDispatch(VkCommandBuffer cb, uint32_t frame, const FrameInputs& in) {
        if (!available() || !active_ || accum_.handle == VK_NULL_HANDLE || frame >= framesInFlight_) return;
        const uint32_t rows = std::min(bandRows_, height_ - rowCursor_);
        PushConstants  pc{};
        pc.width              = width_;
        pc.height             = height_;
        pc.sampleIndex        = samples_;
        pc.samples            = params_.samplesPerFrame;
        pc.maxBounces         = params_.maxBounces;
        // The reset holds for every band of the first pass: each overwrites
        // its own rows once.
        pc.flags              = (resetPending_ ? 1u : 0u) | (in.sunExtracted ? 2u : 0u) | ((in.lensModel & 3u) << 2)
                              | (in.envSunLastDirLight ? 16u : 0u);
        pc.emissiveCount      = in.emissiveCount;
        pc.emissiveTotalPower = in.emissiveTotalPower;
        pc.envMipCount        = std::max(1u, in.envMipCount);
        pc.fireflyClamp       = params_.fireflyClamp;
        pc.sunTanHalfAngle    = in.sunTanHalfAngle;
        pc.seed               = seed_;
        pc.oceanFineTileSize  = in.oceanFineTileSize;
        pc.starIntensity      = in.starIntensity;
        pc.volDensity         = in.volDensity;
        pc.volAniso           = in.volAniso;
        // A picture of another shape than the frame keeps the vertical field:
        // its horizontal NDC span is the aspect ratio of the two, and the
        // normalised intrinsics of the lens follow (fx/W and the principal
        // point's offset from the centre over it).
        const double frameAspect = renderHeight_ ? double(renderWidth_) / renderHeight_ : 1.0;
        const float  k = (height_ && frameAspect > 0.0) ? float((double(width_) / height_) / frameAspect) : 1.f;
        pc.aspectScale    = k;
        pc.lensK[0]       = in.lensK[0] / k;
        pc.lensK[1]       = in.lensK[1];
        pc.lensK[2]       = 0.5f - (0.5f - in.lensK[2]) / k;
        pc.lensK[3]       = in.lensK[3];
        std::copy(in.lensRadial, in.lensRadial + 4, pc.lensRadial);
        std::copy(in.lensTangential, in.lensTangential + 2, pc.lensTangential);
        pc.lensOverscan   = in.lensOverscan;
        pc.lensRadius     = in.lensRadius;
        pc.focusDistance  = in.focusDistance;
        pc.rows           = rowCursor_ | (rows << 16);
        pc.domeRadius     = params_.domePort;

        // A pass begins with its photons: the emitters' in a scene with
        // emitters and glass, the sun's in one with an ocean and a sun. Its
        // bands gather them, on the pipeline that holds the photon code.
        if (rowCursor_ == 0) {
            const bool ready = pipePhotons_ != VK_NULL_HANDLE && photonBuf_.handle != VK_NULL_HANDLE;
            emitLive_    = ready && photonsWanted_ && photonEmitCount_ > 0 && in.emissiveCount > 0 &&
                           in.emissiveTotalPower > 0.f;
            sunLive_     = ready && sunPhotonsWanted_ && photonSunCount_ > 0;
            photonsLive_ = emitLive_ || sunLive_;
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, photonsLive_ ? pipePhotons_ : pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        if (rowCursor_ == 0 && photonsLive_) recordPhotons(cb, in, &pc);
        if (photonsLive_) pc.flags |= 256u;
        if (sunLive_) pc.flags |= 512u;
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (width_ + 7) / 8, (rows + 7) / 8, 1);

        rowCursor_ += rows;
        if (rowCursor_ >= height_) {
            rowCursor_    = 0;
            resetPending_ = false;
            samples_ += params_.samplesPerFrame;
        }
    }

    void PathTracer::recordPhotons(VkCommandBuffer cb, const FrameInputs& in, const void* cameraPush) {
        // The gather radius: two of the picture's pixels at the emitters'
        // distance in the first pass (the shader's set-up widens it for the
        // sun's photons, to hold some four of them), shrinking as Knaus and
        // Zwicker's progressive photon mapping has it, r² by (i + α) / (i + 1)
        // a pass (α = 2/3): r_i² = r_0² Γ(i + 1 + α) / (Γ(1 + α) Γ(i + 2)).
        constexpr double kAlpha        = 2.0 / 3.0;
        constexpr double kRadiusPixels = 2.0;
        const double pass    = double(samples_ / std::max(1u, params_.samplesPerFrame));
        const double shrink2 = std::exp(std::lgamma(pass + 1.0 + kAlpha) - std::lgamma(1.0 + kAlpha) -
                                        std::lgamma(pass + 2.0));
        const double pixel   = in.photonViewHeight > 0.f ? double(in.photonViewHeight) / std::max(1u, height_) : 0.01;
        const float  r       = emitLive_ ? float(std::max(kRadiusPixels * pixel * std::sqrt(shrink2), 1e-5)) : 1e-4f;

        PhotonGridHeader hdr{};
        hdr.capacity   = photonCap_;
        hdr.hashMask   = photonBuckets_ - 1u;
        hdr.radius     = r;
        hdr.cellInv    = 1.f / (2.f * r);
        hdr.powerScale = photonEmitCount_ > 0 ? in.emissiveTotalPower / float(photonEmitCount_) : 0.f;
        hdr.shrink     = float(std::sqrt(shrink2));
        hdr.sunCount   = sunLive_ ? photonSunCount_ : 0u;

        // The last pass's gathers are done with the grid before it is cleared.
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        vkCmdUpdateBuffer(cb, photonGrid_.handle, 0, sizeof(hdr), &hdr);
        vkCmdFillBuffer(cb, photonGrid_.handle, sizeof(hdr), VkDeviceSize(photonBuckets_) * 4, 0u);
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);

        // Set up (mode 5: the sun's disc and scale), trace the emitters' (1)
        // and the sun's (4), allocate the buckets (2), sort (3): each a thread
        // a photon or bucket, in rows of 1024, each reading what the last
        // wrote.
        auto step = [&](uint32_t mode, uint32_t count) {
            PushConstants pc = *static_cast<const PushConstants*>(cameraPush);
            pc.flags = (pc.flags & ~(7u << 5)) | (mode << 5) | (sunLive_ ? 512u : 0u);
            pc.rows  = count;
            vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            const uint32_t threadRows = (count + 1023u) / 1024u;
            vkCmdDispatch(cb, 128, (threadRows + 7u) / 8u, 1);
            VkMemoryBarrier b{};
            b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &b, 0, nullptr, 0, nullptr);
        };
        step(5u, 1u);
        if (emitLive_) step(1u, photonEmitCount_);
        if (sunLive_) step(4u, photonSunCount_);
        step(2u, photonBuckets_);
        step(3u, photonCap_);
    }

    template<class F>
    void PathTracer::submitAndWait(F&& record, const char* what) {
        VkDevice d = ctx_.device();
        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx_.queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        check(vkCreateCommandPool(d, &cpci, nullptr, &cpool), what);
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(d, &cbai, &cb), what);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), what);

        // Everything that wrote the accumulator is visible to what follows.
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        record(cb);
        check(vkEndCommandBuffer(cb), what);

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        check(vkCreateFence(d, &fci, nullptr, &fence), what);
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, fence), what);
        check(vkWaitForFences(d, 1, &fence, VK_TRUE, UINT64_MAX), what);
        vkDestroyFence(d, fence, nullptr);
        vkDestroyCommandPool(d, cpool, nullptr);
    }

    bool PathTracer::read(std::vector<float>& rgb, uint32_t& w, uint32_t& h) {
        w = h = 0;
        if (!available() || accum_.handle == VK_NULL_HANDLE || samples_ == 0) return false;
        const VkDeviceSize bytes = VkDeviceSize(width_) * height_ * 16;
        Buffer staging = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                              VMA_ALLOCATION_CREATE_MAPPED_BIT);
        submitAndWait([&](VkCommandBuffer cb) {
            VkBufferCopy region{};
            region.size = bytes;
            vkCmdCopyBuffer(cb, accum_.handle, staging.handle, 1, &region);
        }, "pathTrace.read");

        void* mapped = nullptr;
        check(vmaMapMemory(ctx_.allocator(), staging.alloc, &mapped), "vmaMapMemory(pathTrace.read)");
        invalidateHostReads(ctx_.allocator(), staging.alloc, 0, bytes);
        const size_t px = size_t(width_) * height_;
        rgb.resize(px * 3);
        const auto* src = static_cast<const float*>(mapped);
        for (size_t i = 0; i < px; ++i) {
            const float n   = src[i * 4 + 3];
            const float inv = n > 0.f ? 1.f / n : 0.f;
            rgb[i * 3 + 0] = src[i * 4 + 0] * inv;
            rgb[i * 3 + 1] = src[i * 4 + 1] * inv;
            rgb[i * 3 + 2] = src[i * 4 + 2] * inv;
        }
        vmaUnmapMemory(ctx_.allocator(), staging.alloc);
        destroyBuffer(ctx_.allocator(), staging);
        w = width_;
        h = height_;
        return true;
    }

    bool PathTracer::readRGB(std::vector<uint8_t>& rgb, uint32_t& w, uint32_t& h,
                             uint32_t toneMapping, float exposure, const OverlayGradeGpu& grade) {
        w = h = 0;
        if (!available() || accum_.handle == VK_NULL_HANDLE || samples_ == 0) return false;
        const size_t       px    = size_t(width_) * height_;
        const VkDeviceSize bytes = VkDeviceSize(px) * 4;
        // The resolve's output lands straight in host-visible memory.
        Buffer out = createBuffer(ctx_.allocator(), ctx_.device(), bytes,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                          VMA_ALLOCATION_CREATE_MAPPED_BIT);
        {
            VkDescriptorBufferInfo inInfo{};
            inInfo.buffer = accum_.handle;
            inInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo outInfo{};
            outInfo.buffer = out.handle;
            outInfo.range  = VK_WHOLE_SIZE;
            VkWriteDescriptorSet ws[2]{};
            for (uint32_t i = 0; i < 2; ++i) {
                ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[i].dstSet          = resolveSet_;
                ws[i].dstBinding      = i;
                ws[i].descriptorCount = 1;
                ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                ws[i].pBufferInfo     = i == 0 ? &inInfo : &outInfo;
            }
            updateDescriptorSets(ctx_.device(), 2, ws, 0, nullptr);
        }
        submitAndWait([&](VkCommandBuffer cb) {
            ResolvePush pc{};
            pc.width       = width_;
            pc.height      = height_;
            pc.toneMapping = toneMapping;
            pc.exposure    = exposure;
            pc.grade       = grade;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeLayout_, 0, 1, &resolveSet_, 0, nullptr);
            vkCmdPushConstants(cb, resolvePipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, (width_ + 7) / 8, (height_ + 7) / 8, 1);
            VkMemoryBarrier mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);
        }, "pathTrace.readRGB");

        void* mapped = nullptr;
        check(vmaMapMemory(ctx_.allocator(), out.alloc, &mapped), "vmaMapMemory(pathTrace.readRGB)");
        invalidateHostReads(ctx_.allocator(), out.alloc, 0, bytes);
        rgb.resize(px * 3);
        const auto* src = static_cast<const uint32_t*>(mapped);
        for (size_t i = 0; i < px; ++i) {
            const uint32_t v = src[i];
            rgb[i * 3 + 0] = static_cast<uint8_t>(v & 0xFFu);
            rgb[i * 3 + 1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
            rgb[i * 3 + 2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
        }
        vmaUnmapMemory(ctx_.allocator(), out.alloc);
        destroyBuffer(ctx_.allocator(), out);
        w = width_;
        h = height_;
        return true;
    }

}// namespace threepp::vulkan
