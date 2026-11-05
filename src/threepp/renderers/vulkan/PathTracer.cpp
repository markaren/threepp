#include "threepp/renderers/vulkan/PathTracer.hpp"

#include "threepp/renderers/vulkan/DescriptorShadow.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"// kMaxMaterialTextures
#include "threepp/renderers/vulkan/shaders/path_trace.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/path_trace_resolve.comp.spv.h"

#include <algorithm>
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
        };
        static_assert(sizeof(PushConstants) == 48);

        // path_trace_resolve.comp's.
        struct ResolvePush {
            uint32_t width;
            uint32_t height;
            uint32_t toneMapping;
            float    exposure;
        };
        static_assert(sizeof(ResolvePush) == 16);

        constexpr uint32_t kNumBindings = 10;

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
            VkPipeline pipe = VK_NULL_HANDLE;
            check(vkCreateComputePipelines(ctx.device(), ctx.pipelineCache(), 1, &cpci, nullptr, &pipe), what);
            vkDestroyShaderModule(ctx.device(), mod, nullptr);
            return pipe;
        }

    }// namespace

    PathTracer::PathTracer(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        if (!ctx_.rayQuerySupported()) return;
        createPipeline();
        createResolvePipeline();
        createDescriptorPool();
        boundAccum_.assign(framesInFlight_, 0);
    }

    PathTracer::~PathTracer() {
        VkDevice d = ctx_.device();
        freeRetired(0, /*all=*/true);
        destroyBuffer(ctx_.allocator(), accum_);
        for (auto s : sets_) forgetDescriptorSet(s);
        if (resolveSet_) forgetDescriptorSet(resolveSet_);
        if (descPool_) vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (pipe_) vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_) vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (resolvePipe_) vkDestroyPipeline(d, resolvePipe_, nullptr);
        if (resolvePipeLayout_) vkDestroyPipelineLayout(d, resolvePipeLayout_, nullptr);
        if (resolveLayout_) vkDestroyDescriptorSetLayout(d, resolveLayout_, nullptr);
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
        sizes[0].descriptorCount = framesInFlight_ * 3;// camera + lights + fog
        sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = framesInFlight_ * (1 + kMaxMaterialTextures);// env + bindless
        sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[2].descriptorCount = framesInFlight_ * 4 + 2;// materials + geometry + emissive + accumulation; the resolve's two
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

            VkWriteDescriptorSet w[9]{};
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
            updateDescriptorSets(ctx_.device(), 9, w, 0, nullptr);
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
        active_       = true;
        resetPending_ = true;
        samples_      = 0;
        ++seed_;
    }

    void PathTracer::prepare(uint32_t frame, uint32_t width, uint32_t height, uint64_t frameSerial) {
        if (!available() || !active_) return;
        freeRetired(frameSerial, false);
        if (width == 0 || height == 0) return;
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
        }
        bindAccumulator(frame);
    }

    void PathTracer::declare(rg::RenderGraph& graph, rg::PassBuilder& pass, uint32_t frame) const {
        if (sets_.empty()) return;
        declareDescriptorSet(graph, pass, sets_[frame], 0, refl_, rg::kCompute);
    }

    void PathTracer::recordDispatch(VkCommandBuffer cb, uint32_t frame,
                                    uint32_t emissiveCount, float emissiveTotalPower,
                                    uint32_t envMipCount, float sunTanHalfAngle, bool sunExtracted) {
        if (!available() || !active_ || accum_.handle == VK_NULL_HANDLE || frame >= framesInFlight_) return;
        PushConstants pc{};
        pc.width              = width_;
        pc.height             = height_;
        pc.sampleIndex        = samples_;
        pc.samples            = params_.samplesPerFrame;
        pc.maxBounces         = params_.maxBounces;
        pc.flags              = (resetPending_ ? 1u : 0u) | (sunExtracted ? 2u : 0u);
        pc.emissiveCount      = emissiveCount;
        pc.emissiveTotalPower = emissiveTotalPower;
        pc.envMipCount        = std::max(1u, envMipCount);
        pc.fireflyClamp       = params_.fireflyClamp;
        pc.sunTanHalfAngle    = sunTanHalfAngle;
        pc.seed               = seed_;

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (width_ + 7) / 8, (height_ + 7) / 8, 1);

        resetPending_ = false;
        samples_ += params_.samplesPerFrame;
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
                             uint32_t toneMapping, float exposure) {
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
