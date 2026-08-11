// LidarScanner — secondary ray-tracing pipeline that emits one beam per
// invocation against the main renderer's TLAS, evaluates the LIDAR equation
// in a custom closest-hit shader, and writes per-beam (range, intensity,
// normal, instance) results back to the host.
//
// Owns its own pipeline + pipeline layout + descriptor set layout (unlike
// PhotonCaustics, which piggybacks on the main RT layout) — the LIDAR pass
// is decoupled from the path tracer's per-frame state, so it doesn't need
// to thread through the same push constant block. The host wires the
// shared TLAS + geom/mat buffers into LidarScanner's descriptor set just
// before each scan() call.
//
// Two ways to drive it, and the difference is worth a paragraph because it is
// the difference between a smooth frame and a 30 ms hitch:
//
//   scan()                 — synchronous. Dispatch, block on the fence, copy
//                            out. Simple, and right for a test or an offline
//                            capture that has no frame to keep.
//   dispatch() / collect() — pipelined. dispatch() submits and RETURNS; the
//                            caller collects on a LATER frame, by which time
//                            the fence is long signaled and collect() costs a
//                            memcpy.
//
// Why the split exists: a blocking readback does not cost "the trace". It costs
// EVERY FRAME ALREADY QUEUED ON THE GPU, because the fence it waits on sits
// behind them. On a renderer running two frames in flight at ~14 ms that is a
// ~28 ms stall for a 1.2 ms trace (measured, RTX 4070, editor Hover Arena), and
// it lands on whatever frame the sensor happened to be due on. A sensor that
// scans at 10 Hz must not cost a stall ten times a second, so the frame-loop
// caller (SensorPlaySession) fires on one frame and takes delivery on the next.
//
// Ordering, with no CPU wait to provide it: the dispatch command buffer opens
// with a barrier whose first synchronization scope is everything submitted
// earlier on the same queue (Vulkan's submission-order rule), so an in-flight
// frame's TLAS build is complete before the trace reads it.

#ifndef THREEPP_VULKAN_LIDAR_SCANNER_HPP
#define THREEPP_VULKAN_LIDAR_SCANNER_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"
#include "threepp/renderers/vulkan/shaders/lidar_shared.h"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <utility>// std::as_const

namespace threepp::vulkan {

    class VulkanContext;

    class LidarScanner {

    public:
        // How many ParticleField density volumes a scan can see at once.
        // KEEP IN SYNC with vulkan::kMaxDensityFields (ParticleFieldPass.hpp)
        // and kMaxDensityFields in shaders/particle_density.glsl; a
        // static_assert in the .cpp ties this one to the first.
        static constexpr uint32_t kDensityVolumes = 4;

        // The world-anchored ParticleField density volumes this scan should
        // delta-track through (parent plan phase 3). The scanner binds exactly
        // what the deferred shade binds, at the same binding numbers and from
        // the same source of truth, so the sensor and the picture agree about
        // where the dust is by construction rather than by convention.
        //
        // EVERY view slot must be a valid image view — the caller fills unused
        // ones with its 1x1x1 dummy, as the deferred set does. What actually
        // gates sampling is `ubo`'s counts.x, which is 0 on a scene with no
        // dust and makes the whole medium a no-op.
        struct DensityBinding {
            VkBuffer     ubo     = VK_NULL_HANDLE;// ParticleDensityUboGpu
            VkDeviceSize uboSize = 0;
            // kDensityVolumes r32ui volume views, VK_IMAGE_LAYOUT_GENERAL.
            const VkImageView* views = nullptr;
            uint32_t           viewCount = 0;
            // Per-volume Q20.12 majorants (ParticleFieldPass::densityMajorants).
            // VK_NULL_HANDLE until some field has asked for a volume; the
            // scanner then binds its own zeroed stand-in, which reads as "no
            // bound medium" and costs nothing.
            VkBuffer     majorants     = VK_NULL_HANDLE;
            VkDeviceSize majorantsSize = 0;
        };

        explicit LidarScanner(VulkanContext& ctx);
        ~LidarScanner();

        LidarScanner(const LidarScanner&) = delete;
        LidarScanner& operator=(const LidarScanner&) = delete;

        // Synchronous scan. Submits an RT dispatch of `numBeams` invocations,
        // waits for completion, and writes up to `pc.maxReturns` per-beam
        // results into outResults[0 .. numBeams * pc.maxReturns - 1]. Slot
        // layout: outResults[beamIdx * maxReturns + returnSlot]. Unused
        // slots are filled with miss sentinels (hitInstanceId = -1) so the
        // host iterates a fixed-stride array and filters.
        //
        // The caller owns the storage and must size it for at least
        // numBeams * max(pc.maxReturns, 1) LidarResult entries.
        //
        // Bails out gracefully (writes all-miss results) when the scene is
        // not yet built (tlas == VK_NULL_HANDLE or buffers null/empty) so
        // the first frame after construction can call scan() safely.
        void scan(VkQueue queue,
                  VkAccelerationStructureKHR tlas,
                  VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                  VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                  VkBuffer fogUbo, VkDeviceSize fogUboSize,
                  const DensityBinding& density,
                  const vulkan_lidar::LidarPushConstants& pc,
                  const vulkan_lidar::LidarBeam* beams, uint32_t numBeams,
                  vulkan_lidar::LidarResult* outResults);

        // How many scans may be outstanding at once. A rig with a LIDAR and a
        // depth camera both due on the same frame must not have to take turns —
        // that would make "which frame did this sensor scan on" depend on how
        // many other sensors are in the scene. Four covers a sensor suite.
        static constexpr uint32_t kScanSlots = 4;
        static constexpr int      kNoSlot    = -1;

        // handle = slot index in the low kHandleShift bits, generation above.
        static constexpr int kHandleShift = 4;// kScanSlots <= 1 << kHandleShift
        static constexpr int kHandleMask  = (1 << kHandleShift) - 1;

        // Same trace, submitted and NOT waited on. Returns a handle to pass to
        // ready()/collect(), or kNoSlot when the beam list is empty or no slot
        // could be had. A handle is returned even when the scene is not built
        // yet: collect() then writes the all-miss result the synchronous path
        // writes, so a caller's cloud is always well-formed.
        //
        // A handle carries a GENERATION, and slots are reclaimed when they run
        // out: a sensor destroyed between firing and collecting (Stop, in the
        // editor) would otherwise hold its slot forever, and four of those
        // would wedge the scanner for the rest of the process. Reclaiming
        // invalidates the abandoned handle, so its owner — if it somehow still
        // exists — is told "nothing outstanding" rather than handed a stranger's
        // cloud. Only a slot whose fence has signaled is reclaimed, so the GPU
        // is never racing a buffer that is being reused.
        int dispatch(VkQueue queue,
                     VkAccelerationStructureKHR tlas,
                     VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                     VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                     VkBuffer fogUbo, VkDeviceSize fogUboSize,
                     const DensityBinding& density,
                     const vulkan_lidar::LidarPushConstants& pc,
                     const vulkan_lidar::LidarBeam* beams, uint32_t numBeams);

        // Whether that dispatch has finished. A vkGetFenceStatus poll, never a
        // wait; false for a handle that is not outstanding.
        [[nodiscard]] bool ready(int handle) const;

        // Take delivery, waiting if it has not finished. `outResults` must hold
        // resultSlots(handle) entries. False when the handle is not outstanding.
        bool collect(int handle, vulkan_lidar::LidarResult* outResults);

        // Result-slot count the dispatch on this handle was made with.
        [[nodiscard]] uint32_t resultSlots(int handle) const;

        // Which slot a handle names, for a caller keeping per-slot storage of
        // its own. Meaningless for kNoSlot; pairs with ready()/collect(), which
        // check the generation and refuse a reclaimed handle.
        [[nodiscard]] static uint32_t slotIndex(int handle) {
            return static_cast<uint32_t>(handle) & static_cast<uint32_t>(kHandleMask);
        }

    private:
        VulkanContext& ctx_;

        // Pipeline + descriptor objects. The pipeline, its layout and the SBT
        // are shared by every slot; only the per-scan state below is per-slot.
        VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
        VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_  = VK_NULL_HANDLE;
        VkPipeline            pipeline_  = VK_NULL_HANDLE;

        // Shader binding table + per-stage regions.
        Buffer                          sbtBuf_{};
        VkStridedDeviceAddressRegionKHR rgenRgn_{};
        VkStridedDeviceAddressRegionKHR missRgn_{};
        VkStridedDeviceAddressRegionKHR hitRgn_{};
        VkStridedDeviceAddressRegionKHR callRgn_{};

        VkCommandPool cmdPool_ = VK_NULL_HANDLE;

        // NEAREST + clamp, for the r32ui density volumes: an integer format
        // cannot be hardware-filtered, and particle_density.glsl does its
        // trilinear by hand with texelFetch, which ignores the sampler
        // entirely. It exists only because the binding is a combined sampler.
        VkSampler densitySampler_ = VK_NULL_HANDLE;
        // 16 zero bytes, bound at binding 6 when the caller has no majorant
        // buffer yet (no field has ever asked for a density volume). A
        // descriptor that is statically used must be valid even on the path
        // where the shader never reads it.
        Buffer    majorantFallback_{};

        // One outstanding scan. Everything a dispatch writes lives here, so a
        // second scan in flight cannot land on the first one's results — and a
        // descriptor set is per-slot because updating one that a pending submit
        // still references is exactly the in-flight-update violation
        // (VUID-vkUpdateDescriptorSets-None-03047) this codebase has been bitten
        // by before.
        //
        // beamBuf is host→device upload (mapped, sequential write); resultBuf is
        // the device-local SSBO the rgen writes; readbackBuf is host-visible and
        // filled by a vkCmdCopyBuffer after the trace. No separate device-local
        // beam buffer: the trace is short-lived and the upload is sequential, so
        // a host-visible SSBO performs well.
        struct Slot {
            VkDescriptorSet descSet = VK_NULL_HANDLE;
            VkCommandBuffer cmdBuf  = VK_NULL_HANDLE;
            VkFence         fence   = VK_NULL_HANDLE;
            Buffer   beamBuf{};
            Buffer   resultBuf{};
            Buffer   readbackBuf{};
            uint32_t capacityBeams   = 0;// beam-buffer rows
            uint32_t capacityResults = 0;// result-buffer rows (= beams × slotsPerBeam)
            // Outstanding: `submitted` says the fence is unsignaled, `slots` how
            // many result entries collect() owes the caller. A dispatch that
            // declined to submit (unbuilt scene) still reserves the slot with
            // submitted == false, so collect() can hand back all-misses.
            bool     reserved  = false;
            bool     submitted = false;
            uint32_t slots     = 0;
            // Bumped on every dispatch. A handle whose generation no longer
            // matches was reclaimed and is dead.
            uint32_t gen   = 0;
            // Dispatch order, so reclaiming takes the stalest slot.
            uint64_t issued = 0;
        };
        std::array<Slot, kScanSlots> slots_{};
        uint64_t dispatchCounter_ = 0;

        // The slot a handle names, or nullptr when the handle is dead (never
        // issued, already collected, or reclaimed under it).
        [[nodiscard]] const Slot* slotFor(int handle) const {
            if (handle < 0) return nullptr;
            const auto index = static_cast<uint32_t>(handle & kHandleMask);
            if (index >= kScanSlots) return nullptr;
            const Slot& s = slots_[index];
            if (!s.reserved) return nullptr;
            if (s.gen != static_cast<uint32_t>(handle >> kHandleShift)) return nullptr;
            return &s;
        }
        [[nodiscard]] Slot* slotFor(int handle) {
            return const_cast<Slot*>(std::as_const(*this).slotFor(handle));
        }

        void createDescriptorLayout();
        void createPipeline();
        void createSbt();
        void createCommandObjects();

        // Round (numBeams, slotsPerBeam) up to powers of two and reallocate
        // buffers if larger than current capacity. `slotsPerBeam` is the
        // total result slots per beam = samplesPerBeam · maxReturns.
        // Updates the descriptor set bindings for beamBuf_ + resultBuf_
        // since the VkBuffer handles change.
        void ensureCapacity(Slot& slot, uint32_t numBeams, uint32_t slotsPerBeam);

        // Update the shared bindings (TLAS, geomDescs, matDescs, fogUbo)
        // before dispatch. The beam/result bindings are updated only when
        // ensureCapacity recreates the buffers.
        void updateSceneBindings(Slot& slot,
                                 VkAccelerationStructureKHR tlas,
                                 VkBuffer geomDescsBuffer, VkDeviceSize geomDescsSize,
                                 VkBuffer matDescsBuffer, VkDeviceSize matDescsSize,
                                 VkBuffer fogUbo, VkDeviceSize fogUboSize,
                                 const DensityBinding& density);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_LIDAR_SCANNER_HPP
