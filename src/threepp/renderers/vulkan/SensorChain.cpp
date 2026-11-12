#include "threepp/renderers/vulkan/SensorChain.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/sensor_evs.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/sensor_rgb.comp.spv.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace threepp::vulkan {

    namespace {

        struct RgbPC {
            uint32_t baseW, baseH, ss, outW, outH, mode, clear;
            float    invGain, nInv, k, g, fullWell, readNoise;
            uint32_t seed;
        };

        struct EvsPC {
            uint32_t baseW, baseH, ss, outW, outH, first;
            float    dt;
            uint32_t seed;
            float    minLuma;
            uint32_t maxEvents;
            float    refractory, baHz, baDark;
            uint32_t bwOn;
            float    tauDay, tauMax, illumRef, illumGain, jitter;
            uint32_t background;
            float    vizDecay;
            uint32_t writeViz;
            uint32_t opIndex;
        };
        static_assert(sizeof(EvsPC) <= 128, "push constants");

        struct TapRec {
            int32_t i00;
            float   fx, fy, pad;
        };

        struct PixelState {
            float   thr, ref, last, filt, lastRel, hotRate;
            int32_t hotPol;
            float   viz;
        };

        constexpr uint32_t kEventsHeader = 16;

        Buffer hostBuffer(VulkanContext& ctx, VkDeviceSize bytes, VkBufferUsageFlags usage) {
            return createBuffer(ctx.allocator(), ctx.device(), bytes, usage,
                                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }

        void* mapped(VulkanContext& ctx, const Buffer& b) {
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(ctx.allocator(), b.alloc, &info);
            return info.pMappedData;
        }

        void computeToHostBarrier(VkCommandBuffer c) {
            VkMemoryBarrier2 mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
            mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
            VkDependencyInfo dep{};
            dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(c, &dep);
        }

        void globalBarrier(VkCommandBuffer c) {
            VkMemoryBarrier2 mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT |
                               VK_ACCESS_2_TRANSFER_WRITE_BIT;
            VkDependencyInfo dep{};
            dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(c, &dep);
        }

    }// namespace

    SensorChain::SensorChain(VulkanContext& ctx, const Setup& s)
        : ctx_(ctx), kind_(s.kind), baseW_(s.baseW), baseH_(s.baseH), ss_(std::max(1u, s.ss)), outW_(s.outW), outH_(s.outH) {
        const size_t n = static_cast<size_t>(outW_) * outH_;
        if (n == 0 || s.tapIndex.size() != n || s.tapFx.size() != n || s.tapFy.size() != n)
            throw std::runtime_error("SensorChain: one tap per output pixel");
        if (kind_ == Kind::Events && (s.threshold.size() != n || s.hotRate.size() != n))
            throw std::runtime_error("SensorChain: one threshold and hot rate per output pixel");
        groupsX_ = (outW_ + 7u) / 8u;
        groupsY_ = (outH_ + 7u) / 8u;
        createPipeline();

        const VkBufferUsageFlags dev = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        tapsBytes_ = sizeof(TapRec) * n;
        taps_ = createBuffer(ctx_.allocator(), ctx_.device(), tapsBytes_, dev, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        stateBytes_ = kind_ == Kind::Events ? sizeof(PixelState) * n : 0;
        staging_ = hostBuffer(ctx_, tapsBytes_ + stateBytes_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        auto* st = static_cast<unsigned char*>(mapped(ctx_, staging_));
        auto* taps = reinterpret_cast<TapRec*>(st);
        for (size_t i = 0; i < n; ++i) taps[i] = {s.tapIndex[i], s.tapFx[i], s.tapFy[i], 0.f};
        if (kind_ == Kind::Rgb) {
            acc_  = createBuffer(ctx_.allocator(), ctx_.device(), 16 * n, dev, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            out_  = hostBuffer(ctx_, 4 * n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            sums_ = hostBuffer(ctx_, 4ull * groupsX_ * groupsY_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            linHost_ = hostBuffer(ctx_, 16 * n, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        } else {
            auto* ps = reinterpret_cast<PixelState*>(st + tapsBytes_);
            for (size_t i = 0; i < n; ++i) ps[i] = {s.threshold[i], 0.f, 0.f, 0.f, -1e30f, s.hotRate[i], 1, 0.f};
            state_ = createBuffer(ctx_.allocator(), ctx_.device(), stateBytes_, dev, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            // a render's worst case is maxEvents signal events on every pixel; twice the pixels holds any
            // realistic frame and overflows loudly beyond it
            eventCapacity_ = static_cast<uint32_t>(8 * n);   // several sensor frames' worth (read once a tick)
            events_ = hostBuffer(ctx_, kEventsHeader + sizeof(Event) * static_cast<VkDeviceSize>(eventCapacity_),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            counter_ = createBuffer(ctx_.allocator(), ctx_.device(), kEventsHeader, dev, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            tau_ = hostBuffer(ctx_, 4 * 40, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            std::memset(mapped(ctx_, tau_), 0, 4 * 40);
            flushHostWrites(ctx_.allocator(), tau_.alloc, 0, 4 * 40);
            viz_ = hostBuffer(ctx_, 4 * n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }
        flushHostWrites(ctx_.allocator(), staging_.alloc, 0, tapsBytes_ + stateBytes_);
    }

    SensorChain::~SensorChain() {
        const VkDevice d = ctx_.device();
        for (Buffer* b : {&taps_, &staging_, &acc_, &out_, &sums_, &linHost_, &state_, &events_, &tau_, &viz_, &counter_})
            destroyBuffer(ctx_.allocator(), *b);
        if (pool_) vkDestroyDescriptorPool(d, pool_, nullptr);
        if (pipeline_) vkDestroyPipeline(d, pipeline_, nullptr);
        if (layout_) vkDestroyPipelineLayout(d, layout_, nullptr);
        if (dsLayout_) vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
    }

    void SensorChain::createPipeline() {
        const bool rgb = kind_ == Kind::Rgb;
        const uint32_t nb = rgb ? 5u : 7u;
        std::vector<VkDescriptorSetLayoutBinding> b(nb);
        for (uint32_t i = 0; i < nb; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = i == 0 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = nb;
        dlci.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_), "vkCreateDescriptorSetLayout(sensor chain)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size       = rgb ? sizeof(RgbPC) : sizeof(EvsPC);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &layout_), "vkCreatePipelineLayout(sensor chain)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = rgb ? sizeof(kSensorRgbCompSpv) : sizeof(kSensorEvsCompSpv);
        smci.pCode    = rgb ? kSensorRgbCompSpv : kSensorEvsCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod), "vkCreateShaderModule(sensor chain)");
        VkComputePipelineCreateInfo cpci{};
        cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = mod;
        cpci.stage.pName  = "main";
        cpci.layout       = layout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(sensor chain)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        const uint32_t nSets = static_cast<uint32_t>(sets_.size());
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[0].descriptorCount = nSets;
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[1].descriptorCount = nSets * (nb - 1);
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = nSets;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &pool_), "vkCreateDescriptorPool(sensor chain)");
        std::vector<VkDescriptorSetLayout> layouts(nSets, dsLayout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = pool_;
        dsai.descriptorSetCount = nSets;
        dsai.pSetLayouts        = layouts.data();
        check(vkAllocateDescriptorSets(ctx_.device(), &dsai, sets_.data()), "vkAllocateDescriptorSets(sensor chain)");
    }

    bool SensorChain::queue(const RgbOp& op, uint32_t srcView) {
        if (kind_ != Kind::Rgb || ops_.size() >= kMaxOps) return false;
        ops_.push_back({true, op, {}, srcView});
        return true;
    }

    bool SensorChain::queue(const EventOp& op, uint32_t srcView) {
        if (kind_ != Kind::Events || ops_.size() >= kMaxOps) return false;
        ops_.push_back({false, {}, op, srcView});
        return true;
    }

    void SensorChain::addPass(rg::RenderGraph& g, uint32_t frame, const std::function<Src(uint32_t)>& src) {
        if (ops_.empty()) return;
        const bool rgb = kind_ == Kind::Rgb;
        std::vector<Op> ops;
        ops.swap(ops_);
        std::vector<VkDescriptorSet> sets(ops.size());
        std::vector<Src> srcs(ops.size());
        for (size_t k = 0; k < ops.size(); ++k) {
            srcs[k] = src(ops[k].src);
            const VkDescriptorSet ds = sets_[(frame % 2u) * kMaxOps + k];
            sets[k] = ds;
            VkDescriptorImageInfo img{};
            img.imageView   = srcs[k].view;
            img.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            std::vector<VkDescriptorBufferInfo> bufs;
            if (rgb) bufs = {{taps_.handle, 0, VK_WHOLE_SIZE}, {acc_.handle, 0, VK_WHOLE_SIZE}, {out_.handle, 0, VK_WHOLE_SIZE}, {sums_.handle, 0, VK_WHOLE_SIZE}};
            else bufs = {{taps_.handle, 0, VK_WHOLE_SIZE}, {state_.handle, 0, VK_WHOLE_SIZE}, {events_.handle, 0, VK_WHOLE_SIZE}, {tau_.handle, 0, VK_WHOLE_SIZE}, {viz_.handle, 0, VK_WHOLE_SIZE}, {counter_.handle, 0, VK_WHOLE_SIZE}};
            std::vector<VkWriteDescriptorSet> w(1 + bufs.size());
            for (size_t i = 0; i < w.size(); ++i) {
                w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet          = ds;
                w[i].dstBinding      = static_cast<uint32_t>(i);
                w[i].descriptorCount = 1;
                w[i].descriptorType  = i == 0 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                if (i == 0) w[i].pImageInfo = &img;
                else w[i].pBufferInfo = &bufs[i - 1];
            }
            vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(w.size()), w.data(), 0, nullptr);
        }

        const bool resetCounter = !rgb && consumed_;
        if (resetCounter) {   // a fresh list after a read (the counter cleared on the GPU); otherwise the frame appends to it
            consumed_ = false, vizWritten_ = false, opBase_ = 0;
        }
        const uint32_t opBase = opBase_;   // after the reset: the first frame after a read counts from 0
        if (!rgb) {
            for (const auto& o: ops) vizWritten_ = vizWritten_ || o.e.writeViz;
            opBase_ += static_cast<uint32_t>(ops.size());
        }

        const bool upload = !uploaded_;
        uploaded_ = true;
        auto pass = g.addPass(rgb ? "sensorChain.rgb" : "sensorChain.events", [this, sets, upload, ops, rgb, opBase, resetCounter](VkCommandBuffer c) {
            globalBarrier(c);
            if (resetCounter) {
                const uint32_t init[4] = {0u, eventCapacity_, 0u, 0u};
                vkCmdUpdateBuffer(c, counter_.handle, 0, sizeof(init), init);
                globalBarrier(c);
            }
            if (upload) {
                VkBufferCopy r{0, 0, tapsBytes_};
                vkCmdCopyBuffer(c, staging_.handle, taps_.handle, 1, &r);
                if (stateBytes_) {
                    VkBufferCopy rs{tapsBytes_, 0, stateBytes_};
                    vkCmdCopyBuffer(c, staging_.handle, state_.handle, 1, &rs);
                }
                globalBarrier(c);
            }
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
            for (size_t k = 0; k < ops.size(); ++k) {
                if (k > 0) globalBarrier(c);   // each operation reads what the one before left
                vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &sets[k], 0, nullptr);
                if (rgb) {
                    const RgbOp& rop = ops[k].r;
                    RgbPC pc{baseW_, baseH_, ss_, outW_, outH_, 0u, rop.clear ? 1u : 0u, rop.invGain, rop.nInv, rop.k, rop.g,
                             rop.fullWell, rop.readNoise, rop.seed};
                    if (rop.accumulate) {
                        vkCmdPushConstants(c, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                        vkCmdDispatch(c, groupsX_, groupsY_, 1);
                    }
                    if (rop.finish) {
                        if (rop.accumulate) globalBarrier(c);
                        pc.mode = 1u;
                        vkCmdPushConstants(c, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                        vkCmdDispatch(c, groupsX_, groupsY_, 1);
                        if (rop.exportLinear) {
                            globalBarrier(c);
                            VkBufferCopy r{0, 0, acc_.size};
                            vkCmdCopyBuffer(c, acc_.handle, linHost_.handle, 1, &r);
                        }
                    }
                } else {
                    const EventOp& eop = ops[k].e;
                    EvsPC pc{baseW_, baseH_, ss_, outW_, outH_, eop.first ? 1u : 0u, eop.dt, eop.seed, eop.minLuma, eop.maxEvents,
                             eop.refractory, eop.baHz, eop.baDark, eop.bandwidth ? 1u : 0u, eop.tauDay, eop.tauMax, eop.illumRef,
                             eop.illumGain, eop.jitter, eop.background ? 1u : 0u, eop.vizDecay, eop.writeViz ? 1u : 0u,
                             opBase + static_cast<uint32_t>(k)};
                    vkCmdPushConstants(c, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(c, groupsX_, groupsY_, 1);
                }
            }
            if (!rgb) {   // the list's header for the host
                globalBarrier(c);
                VkBufferCopy r{0, 0, kEventsHeader};
                vkCmdCopyBuffer(c, counter_.handle, events_.handle, 1, &r);
            }
            computeToHostBarrier(c);
        });
        // the views' colour targets: read after their resolves wrote them
        for (const auto& sc: srcs)
            pass.use(g.importImage("sensorChain.src", sc.image, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_GENERAL),
                     rg::storageRead());
    }

    bool SensorChain::readRgb(std::vector<unsigned char>& rgb8, double& linearSum, std::vector<float>* linear) {
        if (kind_ != Kind::Rgb) return false;
        const size_t n = static_cast<size_t>(outW_) * outH_;
        invalidateHostReads(ctx_.allocator(), out_.alloc, 0, VK_WHOLE_SIZE);
        invalidateHostReads(ctx_.allocator(), sums_.alloc, 0, VK_WHOLE_SIZE);
        const auto* px = static_cast<const unsigned char*>(mapped(ctx_, out_));
        rgb8.resize(3 * n);
        for (size_t i = 0; i < n; ++i) rgb8[3 * i] = px[4 * i], rgb8[3 * i + 1] = px[4 * i + 1], rgb8[3 * i + 2] = px[4 * i + 2];
        const auto* sums = static_cast<const float*>(mapped(ctx_, sums_));
        linearSum = 0;
        for (uint32_t i = 0; i < groupsX_ * groupsY_; ++i) linearSum += sums[i];
        if (linear) {
            invalidateHostReads(ctx_.allocator(), linHost_.alloc, 0, VK_WHOLE_SIZE);
            const auto* l = static_cast<const float*>(mapped(ctx_, linHost_));
            linear->resize(3 * n);
            for (size_t i = 0; i < n; ++i) (*linear)[3 * i] = l[4 * i], (*linear)[3 * i + 1] = l[4 * i + 1], (*linear)[3 * i + 2] = l[4 * i + 2];
        }
        return true;
    }

    void SensorChain::readEvents(std::vector<Event>& out, bool& overflow) {
        out.clear();
        overflow = false;
        if (kind_ != Kind::Events) return;
        invalidateHostReads(ctx_.allocator(), events_.alloc, 0, VK_WHOLE_SIZE);
        const auto* base = static_cast<const unsigned char*>(mapped(ctx_, events_));
        const auto* h = reinterpret_cast<const uint32_t*>(base);
        const uint32_t count = std::min(h[0], eventCapacity_);
        overflow = h[2] != 0 || h[0] > eventCapacity_;
        const auto* ev = reinterpret_cast<const Event*>(base + kEventsHeader);
        out.assign(ev, ev + count);
        consumed_ = true;
    }

    bool SensorChain::readViz(std::vector<unsigned char>& rgba8) {
        if (kind_ != Kind::Events || !vizWritten_) return false;
        const size_t n = static_cast<size_t>(outW_) * outH_;
        invalidateHostReads(ctx_.allocator(), viz_.alloc, 0, VK_WHOLE_SIZE);
        const auto* px = static_cast<const unsigned char*>(mapped(ctx_, viz_));
        rgba8.assign(px, px + 4 * n);
        return true;
    }

    void SensorChain::readTauHistogram(std::array<uint32_t, 40>& hist) {
        hist.fill(0);
        if (kind_ != Kind::Events) return;
        invalidateHostReads(ctx_.allocator(), tau_.alloc, 0, VK_WHOLE_SIZE);
        std::memcpy(hist.data(), mapped(ctx_, tau_), sizeof(uint32_t) * 40);
    }

}// namespace threepp::vulkan
