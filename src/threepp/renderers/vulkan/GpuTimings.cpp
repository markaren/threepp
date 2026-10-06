#include "threepp/renderers/vulkan/GpuTimings.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <algorithm>
#include <array>

namespace threepp::vulkan {

    GpuTimings::GpuTimings(VulkanContext& ctx, uint32_t framesInFlight)
        : ctx_(ctx), framesInFlight_(framesInFlight) {
        pools_.resize(framesInFlight_, VK_NULL_HANDLE);
        maskRecorded_.resize(framesInFlight_, 0u);

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx_.physicalDevice(), &props);
        timestampPeriodNs_ = props.limits.timestampPeriod;
        timingsSupported_  = (timestampPeriodNs_ > 0.f) &&
                             (props.limits.timestampComputeAndGraphics != 0u);
        if (!timingsSupported_) return;

        VkQueryPoolCreateInfo qpci{};
        qpci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = kTimingSlots;
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            check(vkCreateQueryPool(ctx_.device(), &qpci, nullptr, &pools_[f]),
                  "vkCreateQueryPool(timing)");
        }
        // The pass table's pools are made at each slot's first beginFrame
        // (ensurePassPool), sized to passWanted_.
        passPools_.resize(framesInFlight_, VK_NULL_HANDLE);
        passCapacity_.resize(framesInFlight_, 0u);
        passSlots_.resize(framesInFlight_);
    }

    GpuTimings::~GpuTimings() {
        VkDevice d = ctx_.device();
        for (auto p : pools_)
            if (p) vkDestroyQueryPool(d, p, nullptr);
        for (auto p : passPools_)
            if (p) vkDestroyQueryPool(d, p, nullptr);
    }

    void GpuTimings::ensurePassPool(uint32_t frame) {
        if (passCapacity_[frame] >= passWanted_) return;
        // The slot's previous use has signalled its fence and been read back
        // (readBack runs before beginFrame), so its pool is free to go.
        VkDevice d = ctx_.device();
        if (passPools_[frame] != VK_NULL_HANDLE) vkDestroyQueryPool(d, passPools_[frame], nullptr);
        passPools_[frame]    = VK_NULL_HANDLE;
        passCapacity_[frame] = 0u;
        VkQueryPoolCreateInfo qpci{};
        qpci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = passWanted_ * 2u;
        check(vkCreateQueryPool(d, &qpci, nullptr, &passPools_[frame]), "vkCreateQueryPool(pass table)");
        passCapacity_[frame] = passWanted_;
        passSlots_[frame].reserve(passWanted_);
    }

    void GpuTimings::passBegin(VkCommandBuffer cb, const char* name, uint32_t group) {
        passOpen_ = false;
        if (!timingsSupported_) return;
        auto& slots = passSlots_[recordFrame_];
        if (slots.size() >= passCapacity_[recordFrame_]) {
            // Full: this span and the frame's remaining ones go unrecorded, and
            // the slot's next use gets a pool twice the size.
            passWanted_ = std::max(passWanted_, std::min(passCapacity_[recordFrame_] * 2u, kPassTableMax));
            return;
        }
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, passPools_[recordFrame_],
                             static_cast<uint32_t>(slots.size()) * 2u);
        slots.push_back({name ? name : "", group});
        passOpen_ = true;
    }

    void GpuTimings::passEnd(VkCommandBuffer cb) {
        if (!passOpen_) return;
        passOpen_ = false;
        const auto& slots = passSlots_[recordFrame_];
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, passPools_[recordFrame_],
                             static_cast<uint32_t>(slots.size() - 1u) * 2u + 1u);
    }

    void GpuTimings::beginFrame(VkCommandBuffer cb, uint32_t frame) {
        recordStartTp_ = std::chrono::high_resolution_clock::now();
        recordFrame_   = frame;
        passOpen_      = false;
        // Timing pool reset must run on the command stream (CPU-side
        // vkResetQueryPool also works on 1.2+ but we keep the GPU-side
        // reset for portability with older Vulkan toolchains). Clear
        // the host-side recorded-mask in lockstep.
        if (timingsSupported_) {
            ensurePassPool(frame);
            vkCmdResetQueryPool(cb, passPools_[frame], 0, passCapacity_[frame] * 2u);
            passSlots_[frame].clear();
        }
        if (timingsSupported_ && pools_[frame] != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(cb, pools_[frame], 0, kTimingSlots);
            maskRecorded_[frame] = 0u;
            // Open the whole-command-buffer bracket. AFTER the reset, in the same
            // command buffer — writing a timestamp into a query the stream has not
            // yet reset is VUID-vkCmdWriteTimestamp2-None-03864 and garbage
            // results. Written directly rather than through begin(): suppressed_
            // is a secondary-view device and must not be able to make the frame
            // total vanish. Its mask bit is set by endFrameTotal.
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                 pools_[frame], TP_Frame * 2u);
        }
    }

    void GpuTimings::endFrameTotal(VkCommandBuffer cb, uint32_t frame) {
        if (!timingsSupported_ || pools_[frame] == VK_NULL_HANDLE) return;
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             pools_[frame], TP_Frame * 2u + 1u);
        maskRecorded_[frame] |= (1u << TP_Frame);
    }

    void GpuTimings::begin(VkCommandBuffer cb, TimingPass pass, uint32_t frame) {
        if (!timingsSupported_ || suppressed_) return;
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             pools_[frame], pass * 2u);
        maskRecorded_[frame] |= (1u << pass);
    }

    void GpuTimings::end(VkCommandBuffer cb, TimingPass pass, uint32_t frame) {
        if (!timingsSupported_ || suppressed_) return;
        vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             pools_[frame], pass * 2u + 1u);
    }

    void GpuTimings::finishRecord() {
        using namespace std::chrono;
        lastTimings_.cpuRecordMs =
                duration<float, std::milli>(high_resolution_clock::now() - recordStartTp_).count();
    }

    void GpuTimings::readBack(uint32_t frame, float pendingCpuEnsureMs) {
        // Pre-populate CPU fields the caller can keep updated even if GPU
        // timings aren't available.
        lastTimings_.cpuEnsureSceneMs = pendingCpuEnsureMs;
        // Zero the GPU fields — only the passes that ran will overwrite.
        lastTimings_.rasterGbufMs  = 0.f;
        lastTimings_.gbufResolveMs = 0.f;
        lastTimings_.shadeBMs      = 0.f;
        lastTimings_.overlayMs     = 0.f;
        lastTimings_.pathTraceMs   = 0.f;
        lastTimings_.denoiseMs     = 0.f;
        lastTimings_.taaMs         = 0.f;
        lastTimings_.dofMs         = 0.f;
        lastTimings_.froxelMs      = 0.f;
        lastTimings_.splatMs       = 0.f;
        // These three were missing from the zeroing block: assigned below but
        // never cleared, so on the two early returns (no timestamp support, first
        // use of a slot) they kept the PREVIOUS frame's value — stale non-zero
        // splat numbers on exactly the frames a warm-up window includes.
        lastTimings_.splatProjectMs = 0.f;
        lastTimings_.splatSortMs    = 0.f;
        lastTimings_.splatRasterMs  = 0.f;
        lastTimings_.instanceExpandMs = 0.f;
        lastTimings_.particleDensityMs = 0.f;
        lastTimings_.particleEmitMs   = 0.f;
        lastTimings_.oceanFftMs      = 0.f;
        lastTimings_.oceanDisplaceMs = 0.f;
        lastTimings_.oceanFoamMs     = 0.f;
        lastTimings_.rtaoMs          = 0.f;
        lastTimings_.probeGiMs       = 0.f;
        lastTimings_.oceanBlasMs     = 0.f;
        lastTimings_.tlasRefitMs     = 0.f;
        lastTimings_.dynGeomRefitMs  = 0.f;
        lastTimings_.gpuTotalMs     = 0.f;
        lastTimings_.gpuPassSumMs   = 0.f;
        lastTimings_.passes.clear();
        if (!timingsSupported_) return;
        const float toMs = timestampPeriodNs_ * 1e-6f;

        // The pass table: every span this slot's previous use recorded, in one
        // fetch. Read with availability rather than WAIT: the fence has
        // signalled, so every query that was executed is available, and a
        // command buffer that was recorded but never submitted (a swapchain
        // out-of-date frame) leaves its queries unavailable instead of hanging
        // the readback. Rows whose begin or end is unavailable are dropped.
        if (const auto& slots = passSlots_[frame]; !slots.empty()) {
            const uint32_t n = static_cast<uint32_t>(slots.size()) * 2u;
            passResults_.assign(size_t(n) * 2u, 0u);// (value, availability) per query
            const VkResult r = vkGetQueryPoolResults(
                    ctx_.device(), passPools_[frame], 0, n,
                    passResults_.size() * sizeof(uint64_t), passResults_.data(),
                    2u * sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            if (r == VK_SUCCESS || r == VK_NOT_READY) {
                lastTimings_.passes.reserve(slots.size());
                for (size_t i = 0; i < slots.size(); ++i) {
                    const uint64_t t0 = passResults_[i * 4u + 0u], a0 = passResults_[i * 4u + 1u];
                    const uint64_t t1 = passResults_[i * 4u + 2u], a1 = passResults_[i * 4u + 3u];
                    if (!a0 || !a1) continue;
                    VulkanRenderer::FrameTimings::PassTiming row;
                    row.name  = slots[i].name;
                    row.view  = slots[i].group;
                    row.gpuMs = t1 >= t0 ? static_cast<float>(t1 - t0) * toMs : 0.f;
                    lastTimings_.passes.push_back(std::move(row));
                }
            }
        }

        const uint32_t mask = maskRecorded_[frame];
        if (mask == 0u) return;// first use of this slot
        // We read pairs individually (not in one bulk fetch) because slots
        // for passes that didn't run this cycle are RESET but never WRITTEN,
        // and VK_QUERY_RESULT_WAIT_BIT on a reset query blocks indefinitely.
        auto pairMs = [&](TimingPass p) -> float {
            if ((mask & (1u << p)) == 0u) return 0.f;
            std::array<uint64_t, 2> pair{};
            const VkResult r = vkGetQueryPoolResults(
                    ctx_.device(), pools_[frame],
                    p * 2u, 2u, sizeof(pair), pair.data(),
                    sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (r != VK_SUCCESS) return 0.f;
            if (pair[1] < pair[0]) return 0.f;
            return static_cast<float>(pair[1] - pair[0]) * toMs;
        };
        lastTimings_.rasterGbufMs  = pairMs(TP_RasterGbuf);
        lastTimings_.gbufResolveMs = pairMs(TP_GbufResolve);
        lastTimings_.shadeBMs      = pairMs(TP_ShadeB);
        // Overlay timings collapse the depth prepass + draw pair into a
        // single "overlay" column for the public API.
        lastTimings_.overlayMs    = pairMs(TP_OverlayDepth) + pairMs(TP_OverlayDraw);
        lastTimings_.pathTraceMs  = pairMs(TP_DeferredShade);// public field name kept for API stability
        lastTimings_.denoiseMs    = pairMs(TP_Denoise);
        lastTimings_.taaMs        = pairMs(TP_TAA);
        lastTimings_.dofMs        = pairMs(TP_Dof);
        lastTimings_.froxelMs     = pairMs(TP_Froxel);
        lastTimings_.splatMs      = pairMs(TP_Splat);
        lastTimings_.splatProjectMs = pairMs(TP_SplatProject);
        lastTimings_.splatSortMs    = pairMs(TP_SplatSort);
        lastTimings_.splatRasterMs  = pairMs(TP_SplatRaster);
        lastTimings_.instanceExpandMs = pairMs(TP_InstanceExpand);
        lastTimings_.particleDensityMs = pairMs(TP_ParticleDensity);
        lastTimings_.particleEmitMs    = pairMs(TP_ParticleEmit);
        lastTimings_.oceanFftMs      = pairMs(TP_OceanFFT);
        lastTimings_.oceanDisplaceMs = pairMs(TP_OceanDisplace);
        lastTimings_.oceanFoamMs     = pairMs(TP_OceanFoam);
        lastTimings_.rtaoMs          = pairMs(TP_Rtao);
        lastTimings_.probeGiMs       = pairMs(TP_ProbeGI);
        lastTimings_.oceanBlasMs     = pairMs(TP_OceanBlas);
        lastTimings_.tlasRefitMs     = pairMs(TP_TlasRefit);
        lastTimings_.dynGeomRefitMs  = pairMs(TP_DynGeomRefit);
        lastTimings_.gpuTotalMs     = pairMs(TP_Frame);
        // The DISJOINT bracketed passes only. TP_SplatProject/Sort/Raster are
        // recorded INSIDE TP_Splat and partition it, so they are excluded or splat
        // frames double-count. TP_SensorImage is named explicitly: it is bracketed
        // in the record pass but has no public field, so a sum built from the
        // struct alone would silently drop it. gpuTotalMs minus this is the GPU
        // work no bracket covers, plus the pipeline bubbles between passes.
        lastTimings_.gpuPassSumMs =
                lastTimings_.rasterGbufMs + lastTimings_.gbufResolveMs +
                lastTimings_.shadeBMs + lastTimings_.overlayMs +
                lastTimings_.pathTraceMs + lastTimings_.denoiseMs +
                lastTimings_.taaMs + lastTimings_.dofMs +
                lastTimings_.froxelMs + lastTimings_.splatMs +
                lastTimings_.instanceExpandMs +
                lastTimings_.particleDensityMs +
                lastTimings_.particleEmitMs +
                lastTimings_.oceanFftMs + lastTimings_.oceanDisplaceMs +
                lastTimings_.oceanFoamMs + lastTimings_.oceanBlasMs +
                lastTimings_.tlasRefitMs + lastTimings_.dynGeomRefitMs +
                lastTimings_.rtaoMs + lastTimings_.probeGiMs +
                pairMs(TP_SensorImage);
    }

}// namespace threepp::vulkan
