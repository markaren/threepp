#include "threepp/renderers/vulkan/ProbeGI.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"// kMaxMaterialTextures

#include "threepp/renderers/vulkan/shaders/probe_update.comp.spv.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace threepp::vulkan {

    namespace {
        // Host mirror of the shaders' ProbeGridUbo (scalar layout).
        struct GridUboData {
            float   origin[3];
            float   enabled;
            float   spacing[3];
            float   _pad0;
            int32_t dims[3];
            int32_t _pad1;
        };
        static_assert(sizeof(GridUboData) == 48, "ProbeGridUbo layout drift");
    }// namespace

    ProbeGI::ProbeGI(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        // SH-L1 store: 4 vec4 per probe. TRANSFER_DST for the zero-fill on
        // (re)fit; STORAGE for the update pass + deferred_shade's reads.
        // TRANSFER_SRC: the determinism audit hashes this buffer per frame
        // (debugHashShadeImages' probeSh row) — pure capability bit.
        shBuf_ = createBuffer(ctx_.allocator(), ctx_.device(),
                              static_cast<VkDeviceSize>(kProbeCount) * 4 * 16,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        // Chebyshev depth store: kDepthTexels × packHalf2x16(mean, mean²) per
        // probe = 4 MB. Cleared alongside the SH on every grid (re)fit — 0u is
        // the "no data yet" sentinel the sampler treats as fully visible.
        depthBuf_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                 static_cast<VkDeviceSize>(kProbeCount) * kDepthTexels * 4,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,// audit readback
                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        // Snapshot twins (see the header comment): copy targets, shader reads.
        prevShBuf_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                  static_cast<VkDeviceSize>(kProbeCount) * 4 * 16,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        prevDepthBuf_ = createBuffer(ctx_.allocator(), ctx_.device(),
                                     static_cast<VkDeviceSize>(kProbeCount) * kDepthTexels * 4,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        gridUbos_.resize(framesInFlight_);
        gridUboHandles_.resize(framesInFlight_);
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            gridUbos_[f] = createBuffer(ctx_.allocator(), ctx_.device(),
                                        sizeof(GridUboData),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            gridUboHandles_[f] = gridUbos_[f].handle;
            updateGridUbo(f, false);// valid (disabled) contents before first use
        }
        createPipeline();
        createDescriptorPool();
    }

    ProbeGI::~ProbeGI() {
        VkDevice d = ctx_.device();
        if (pipe_)       vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        for (auto& b : gridUbos_) destroyBuffer(ctx_.allocator(), b);
        destroyBuffer(ctx_.allocator(), shBuf_);
        destroyBuffer(ctx_.allocator(), depthBuf_);
        destroyBuffer(ctx_.allocator(), prevShBuf_);
        destroyBuffer(ctx_.allocator(), prevDepthBuf_);
    }

    void ProbeGI::createPipeline() {
        VkDevice d = ctx_.device();

        VkDescriptorSetLayoutBinding b[12]{};
        auto set = [&](uint32_t i, VkDescriptorType t) {
            b[i].binding = i;
            b[i].descriptorType = t;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        };
        set(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);            // ProbeGridUbo
        set(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);            // lights
        set(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);    // env (PMREM)
        set(3, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);// TLAS
        set(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);            // MaterialDesc[]
        set(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);            // GeometryDesc[]
        set(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);    // bindless material textures...
        b[6].descriptorCount = kMaxMaterialTextures;          // ...fixed-size array
        set(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);            // EmTri[]
        set(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);            // SH probe store (write side)
        set(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);            // Chebyshev depth store (write side)
        set(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);           // SH snapshot (read side)
        set(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);           // depth snapshot (read side)

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 12;
        dlci.pBindings = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(probeGI)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.offset = 0;
        pc.size = 32;// 8×u32 (offset, count, frame, flags, emCount, emPower, envMips, pad)
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(probeGI)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kProbeUpdateCompSpv);
        smci.pCode    = kProbeUpdateCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), "vkCreateShaderModule(probe_update)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipeLayout_;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(probe_update)");
        vkDestroyShaderModule(d, mod, nullptr);
    }

    void ProbeGI::createDescriptorPool() {
        VkDescriptorPoolSize sizes[4]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = framesInFlight_ * 2;// grid + lights
        sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = framesInFlight_ * (1 + kMaxMaterialTextures);// env + bindless
        sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[2].descriptorCount = framesInFlight_ * 7;// materials + geometry + emissive + SH/depth write + SH/depth snapshot
        sizes[3].type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        sizes[3].descriptorCount = framesInFlight_ * 1;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_;
        dpci.poolSizeCount = 4;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(probeGI)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_, dsLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = descPool_;
        ai.descriptorSetCount = framesInFlight_;
        ai.pSetLayouts        = layouts.data();
        sets_.resize(framesInFlight_);
        check(vkAllocateDescriptorSets(ctx_.device(), &ai, sets_.data()),
              "vkAllocateDescriptorSets(probeGI)");
    }

    void ProbeGI::rewriteDescriptors(const DescriptorWriteInputs& in, int onlyFrame) {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            if (onlyFrame >= 0 && f != static_cast<uint32_t>(onlyFrame)) continue;
            VkDescriptorBufferInfo gridInfo{};
            gridInfo.buffer = gridUbos_[f].handle;
            gridInfo.range  = VK_WHOLE_SIZE;
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
            geomInfo.buffer = in.geomDescBuf[f];// per-FIF ring (auto-LOD level switches flush per slot)
            geomInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo emInfo{};
            emInfo.buffer = in.emissiveTriBuf[f];
            emInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo shInfo{};
            shInfo.buffer = shBuf_.handle;
            shInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo depthInfo{};
            depthInfo.buffer = depthBuf_.handle;
            depthInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo shPrevInfo{};
            shPrevInfo.buffer = prevShBuf_.handle;
            shPrevInfo.range  = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo depthPrevInfo{};
            depthPrevInfo.buffer = prevDepthBuf_.handle;
            depthPrevInfo.range  = VK_WHOLE_SIZE;

            VkAccelerationStructureKHR tlasLocal = in.tlas;
            VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
            asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            asInfo.accelerationStructureCount = 1;
            asInfo.pAccelerationStructures = &tlasLocal;

            VkWriteDescriptorSet w[12]{};
            auto setw = [&](int n, uint32_t bind, VkDescriptorType t,
                            const VkDescriptorImageInfo* img, const VkDescriptorBufferInfo* buf) {
                w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstSet = sets_[f];
                w[n].dstBinding = bind;
                w[n].descriptorCount = 1;
                w[n].descriptorType = t;
                w[n].pImageInfo = img;
                w[n].pBufferInfo = buf;
            };
            setw(0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr,  &gridInfo);
            setw(1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         nullptr,  &lightInfo);
            setw(2, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr);
            setw(3, 3, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr);
            w[3].pNext = &asInfo;
            setw(4, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         nullptr,  &matInfo);
            setw(5, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         nullptr,  &geomInfo);
            w[6].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[6].dstSet          = sets_[f];
            w[6].dstBinding      = 6;
            w[6].descriptorCount = in.materialTexCount;
            w[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[6].pImageInfo      = in.materialTex;
            setw(7, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         nullptr,  &emInfo);
            setw(8, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         nullptr,  &shInfo);
            setw(9, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         nullptr,  &depthInfo);
            setw(10, 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,       nullptr,  &shPrevInfo);
            setw(11, 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,       nullptr,  &depthPrevInfo);
            vkUpdateDescriptorSets(ctx_.device(), 12, w, 0, nullptr);
        }
    }

    void ProbeGI::rewriteEmissive(uint32_t frame, VkBuffer emissiveTriBuf) {
        VkDescriptorBufferInfo info{};
        info.buffer = emissiveTriBuf;
        info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = sets_[frame];
        w.dstBinding = 7;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &info;
        vkUpdateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
    }

    void ProbeGI::setGridBounds(const float aabbMin[3], const float aabbMax[3]) {
        const uint32_t dims[3] = {kDimX, kDimY, kDimZ};
        float newSpacing[3], newOrigin[3];
        for (int a = 0; a < 3; ++a) {
            // Cell-centered probes: spacing = extent/dims, first probe half a
            // cell inside the AABB. The 0.05 m floor keeps a flat scene (zero
            // Y extent) from collapsing the trilinear cell to a divide-by-0.
            const float extent = std::max(aabbMax[a] - aabbMin[a], 0.f);
            newSpacing[a] = std::max(extent / static_cast<float>(dims[a]), 0.05f);
            newOrigin[a]  = aabbMin[a] + 0.5f * newSpacing[a];
        }
        // HYSTERESIS — adopting a new grid WIPES the SH cache (needsClear_),
        // and the wipe collapses the measured ambient scene-wide for the ~8+
        // frames the round-robin needs to refill. Every STRUCTURAL scene
        // change re-fits (VulkanCoreImpl probeGridDirty_), and gameplay churns
        // structure constantly (tracers, casings, decals, muzzle flashes) —
        // an unconditional wipe made every sun-occluded mesh visibly strobe
        // (measured: a 1 cm cube toggled every 5 frames pulsed a Sponza
        // corridor's shadowed ambient 0.4→5.2 mean luminance). Keep the
        // current grid while it still represents the scene: origin within
        // half a cell and spacing within −20 %/+25 % per axis. Out-of-grid
        // points clamp in probeIrradiance, so a slightly-outgrown AABB
        // degrades gracefully; a real scene swap trips the thresholds and
        // takes the full wipe.
        if (gridFitted_) {
            bool keep = true;
            for (int a = 0; a < 3; ++a) {
                if (newSpacing[a] > gridSpacing_[a] * 1.25f ||
                    newSpacing[a] < gridSpacing_[a] * 0.80f ||
                    std::abs(newOrigin[a] - gridOrigin_[a]) > 0.5f * gridSpacing_[a]) {
                    keep = false;
                    break;
                }
            }
            if (keep) return;
        }
        std::memcpy(gridSpacing_, newSpacing, sizeof(newSpacing));
        std::memcpy(gridOrigin_,  newOrigin,  sizeof(newOrigin));
        gridFitted_   = true;
        needsClear_   = true;
        probeOffset_  = 0;
        updateCounter_ = 0;
        haveWindow_    = false;// the fill below invalidates the prev snapshots
    }

    void ProbeGI::updateGridUbo(uint32_t frame, bool enabled) {
        GridUboData d{};
        std::memcpy(d.origin,  gridOrigin_,  sizeof(gridOrigin_));
        std::memcpy(d.spacing, gridSpacing_, sizeof(gridSpacing_));
        d.dims[0] = static_cast<int32_t>(kDimX);
        d.dims[1] = static_cast<int32_t>(kDimY);
        d.dims[2] = static_cast<int32_t>(kDimZ);
        d.enabled = enabled ? 1.f : 0.f;
        void* mapped = nullptr;
        if (vmaMapMemory(ctx_.allocator(), gridUbos_[frame].alloc, &mapped) == VK_SUCCESS) {
            std::memcpy(mapped, &d, sizeof(d));
            flushHostWrites(ctx_.allocator(), gridUbos_[frame].alloc, 0, sizeof(d));
            vmaUnmapMemory(ctx_.allocator(), gridUbos_[frame].alloc);
        }
    }

    void ProbeGI::recordDispatch(VkCommandBuffer cb, uint32_t frame,
                                 uint32_t emissiveCount, float emissiveTotalPower,
                                 bool shadows, uint32_t envMipCount) {
        // A full prev-store refresh is needed whenever the canonical stores
        // could differ from prev OUTSIDE the last dispatch's window: the first
        // dispatch of all (prev is uninitialised), and any frame the clear
        // below zeroes the canonical stores under prev's feet.
        bool fullCopy = !haveWindow_;
        if (needsClear_) {
            // Fresh fit: zero SH + validity + history so probes bootstrap with
            // α = 1 instead of blending into a stale grid. The depth store is
            // zeroed too — 0u is its "no data" sentinel (sampled as fully
            // visible, so bootstrap can't Chebyshev-reject every probe).
            vkCmdFillBuffer(cb, shBuf_.handle, 0, VK_WHOLE_SIZE, 0u);
            vkCmdFillBuffer(cb, depthBuf_.handle, 0, VK_WHOLE_SIZE, 0u);
            VkMemoryBarrier2 bar{};
            bar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            bar.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            bar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            bar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &bar;
            vkCmdPipelineBarrier2(cb, &dep);
            needsClear_ = false;
            fullCopy    = true;
        }

        // Snapshot the canonical stores for this update's reads — but only the
        // part that can have changed. The update reads prev ANYWHERE (the
        // feedback tap samples at ray-hit points, which land wherever the
        // geometry does), so prev must equal canonical over the whole grid.
        // After dispatch k, though, the only probes where they differ are
        // dispatch k's own window: nothing else wrote the canonical stores.
        // So copying just that window before dispatch k+1 restores equality by
        // induction, and the snapshot is byte-for-byte what a full copy would
        // have produced. 10 MB of copy per frame becomes ~1.3 MB.
        //
        // Barrier 1:
        // whatever wrote them last (the previous update's compute, or the
        // needsClear_ fill above) must be visible to the transfer; barrier 2:
        // the copies must land before the dispatch reads the snapshots, and
        // the canonical stores' transfer-READ must retire before the dispatch
        // WRITES them (WAR — execution ordering, covered by the same barrier).
        {
            VkMemoryBarrier2 toCopy{};
            toCopy.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            toCopy.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            toCopy.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT;
            toCopy.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            toCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT |
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT;
            VkDependencyInfo dep{};
            dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers    = &toCopy;
            vkCmdPipelineBarrier2(cb, &dep);

            // [first, first + n) in probe indices, wrapped at kProbeCount into
            // at most two regions. The window never wraps as long as
            // kProbesPerFrame divides kProbeCount, but the cursor arithmetic
            // does not promise that and a silent half-copy would be a
            // needle-in-a-haystack bug.
            const uint32_t first = fullCopy ? 0u : lastOffset_;
            const uint32_t n     = fullCopy ? kProbeCount : lastCount_;
            uint32_t spans[2][2] = {{first, std::min(n, kProbeCount - first)}, {0u, 0u}};
            spans[1][1]          = n - spans[0][1];
            const uint32_t regionCount = spans[1][1] > 0 ? 2u : 1u;

            constexpr VkDeviceSize kShStride    = 4 * 16;
            const VkDeviceSize     kDepthStride = static_cast<VkDeviceSize>(kDepthTexels) * 4;

            VkBufferCopy shCopy[2]{}, depthCopy[2]{};
            for (uint32_t i = 0; i < regionCount; ++i) {
                shCopy[i].srcOffset = shCopy[i].dstOffset = spans[i][0] * kShStride;
                shCopy[i].size                            = spans[i][1] * kShStride;
                depthCopy[i].srcOffset = depthCopy[i].dstOffset = spans[i][0] * kDepthStride;
                depthCopy[i].size                               = spans[i][1] * kDepthStride;
            }
            vkCmdCopyBuffer(cb, shBuf_.handle, prevShBuf_.handle, regionCount, shCopy);
            vkCmdCopyBuffer(cb, depthBuf_.handle, prevDepthBuf_.handle, regionCount, depthCopy);

            VkMemoryBarrier2 toCompute{};
            toCompute.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            toCompute.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            toCompute.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT |
                                      VK_ACCESS_2_TRANSFER_READ_BIT;
            toCompute.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            toCompute.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            dep.pMemoryBarriers = &toCompute;
            vkCmdPipelineBarrier2(cb, &dep);
        }

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout_, 0, 1, &sets_[frame], 0, nullptr);
        const uint32_t count = std::min(kProbesPerFrame, kProbeCount);
        uint32_t emPowerBits;
        std::memcpy(&emPowerBits, &emissiveTotalPower, sizeof(emPowerBits));
        const uint32_t pc[8] = {probeOffset_, count, updateCounter_,
                                shadows ? 1u : 0u,
                                emissiveCount, emPowerBits, envMipCount, 0u};
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cb, count, 1, 1);

        // The window this dispatch is about to write, so the NEXT dispatch's
        // snapshot knows what to restore. Tracked per dispatch, not per frame.
        lastOffset_ = probeOffset_;
        lastCount_  = count;
        haveWindow_ = true;

        probeOffset_ = (probeOffset_ + count) % kProbeCount;
        ++updateCounter_;
    }

}// namespace threepp::vulkan
