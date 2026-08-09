// VulkanImplCommon — constants shared by the types split out of
// VulkanCoreImpl.hpp. Anything a moved struct needs at namespace scope lives
// here; VulkanRenderer::Impl forwards each name at its original spot so every
// reference site (Impl methods and the VulkanCore*.cpp TUs) is unchanged.

#ifndef THREEPP_VULKAN_IMPL_COMMON_HPP
#define THREEPP_VULKAN_IMPL_COMMON_HPP

#include <cstdint>

namespace threepp::vulkan::impl {

    // Frames-in-flight depth. Bumped from 2 → 3 to deepen CPU/GPU
    // pipelining: while frame N+2 is being recorded on the CPU, frame N
    // and frame N+1 can be in different stages of GPU execution. Hides
    // CPU jitter (scene-build, ImGui, frustum cull) without changing the
    // GPU schedule (queue is still serial — async compute would do that,
    // and is a much larger change).
    //
    // The 2-slot ping-pong (the ReSTIR DI reservoir images) stays at 2
    // entries — Vulkan queue execution is
    // strictly in-order within a queue, so when frame N+2 writes slot
    // (N+2)&1 the prior owner of that slot (frame N) has fully completed
    // on the GPU. Temporal reproject still reads "the previous frame"
    // because readSlot = 1 - writeSlot, which alternates correctly.
    //
    // MUST stay EVEN. The ping-pong slot is `currentFrame & 1`, and
    // currentFrame cycles mod kFramesInFlight. Only an even count makes
    // `currentFrame & 1` track the true monotonic frame parity, so the
    // slot actually alternates frame-to-frame. An ODD count (e.g. 3)
    // desyncs it: the write-slot sequence becomes 0,1,0,0,1,0,… so every
    // 3rd frame the temporal read samples a 2-frame-STALE slot while the
    // immediately-previous frame's output is overwritten unread —
    // corrupting accum/gbuf/moments/albedo/ReSTIR/TAA history on a 3-frame
    // beat (periodic ghosting + reprojection reading the wrong frame).
    // If a deeper pipeline is ever wanted, decouple the ping-pong parity
    // from this sync ring (drive the slot from a monotonic `++parity & 1`
    // and rewrite the temporal image bindings per frame) instead of
    // bumping this to an odd value.
    constexpr uint32_t kFramesInFlight = 2;

    // Light-count bounds for GpuLightsUbo (VulkanGpuLayouts.hpp).
    static constexpr uint32_t kMaxDirLights   = 8;
    static constexpr uint32_t kMaxPointLights = 8;
    static constexpr uint32_t kMaxSpotLights  = 8;
    static constexpr uint32_t kMaxRectLights  = 4;

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_IMPL_COMMON_HPP
