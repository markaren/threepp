// TlasState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the top-level acceleration structure: its buffers, refit scratch, instance staging and dirty rows, and
// the per-entry ray-flag classification (cutout and ray-shadow distance).
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_TLASSTATE_HPP
#define THREEPP_VULKAN_STATE_TLASSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct TlasState {
        // Single TLAS over all mesh instances in the scene.
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        Buffer tlasBuffer;
        // Per-in-flight-frame instance buffers. The per-frame TLAS refit is
        // recorded into the frame command buffer (not a one-shot drain), so the
        // host instance write for frame N must not clobber the buffer frame N-1
        // is still reading — hence one buffer per in-flight slot, indexed by
        // currentFrame. The structural full build (buildTlas) writes all slots.
        Buffer tlasInstancesBuffers[kFramesInFlight];
        // Persistent scratch for the in-frame TLAS refit (sized once; reused —
        // safe only because the frame graph's entry barrier orders each frame's
        // AS builds after the previous frame's work; submit order alone is not
        // an execution dependency).
        Buffer tlasRefitScratch_{};
        VkDeviceSize tlasRefitScratchSize_ = 0;
        // Instance count of the TLAS's last full BUILD. A MODE_UPDATE with any
        // other count is a spec violation that corrupts traversal, so
        // recordTlasRefit promotes such a refit to a full build (or skips it
        // when the existing storage can't hold the build). The structural
        // fingerprint now catches membership flips too — this is the backstop.
        uint32_t tlasBuiltInstanceCount_ = 0;
        // Per-frame TLAS refit, staged by ensureSceneBuilt and recorded into the
        // frame command buffer by recordCommandBuffer (after the deformable BLAS
        // rebuilds). Replaces the old mid-frame refitTlas one-shot drain.
        std::vector<VkAccelerationStructureInstanceKHR> pendingTlasInstances_;
        bool pendingTlasRefit_ = false;
        bool pendingTlasFullBuild_ = false;

        // ── Cutout ray distance (setCutoutRayDistance; 0 = off, the default) ─
        // Every BLAS is non-opaque so that a cutout caster's hits come back to
        // the shader as candidates to alpha-test. A shadow ray through a tree
        // crosses many leaf cards, and each is a round trip (material read, UV
        // fetch, texture sample). Beyond this distance from the camera a
        // cutout caster's instance is FORCE_OPAQUE instead: the RT core
        // commits the first card it meets. Its shadow is then the cards' own
        // outline, not the leaves'. Reflection and GI rays already trace with
        // gl_RayFlagsOpaqueEXT and are unchanged.
        // The entries are classified for a camera position and re-classified
        // once the camera has moved 5 % of the distance from it (a flag flip
        // is a full TLAS build, so not every frame of a moving camera pays
        // one); an entry that moves by itself waits for that too.
        float cutoutRayDistance_ = 0.f;
        // setCutoutRayCap: the per-ray answer to the same cost (shadowVis in
        // deferred_shade_10_lighting_utils.glsl; ShadePush flags bits 16-19).
        uint32_t cutoutRayCap_ = 0u;
        float    cutoutRayCapFrom_ = 0.f;// ...for what is shaded at least this far from the eye
        float cutoutRayClassifiedFor_ = 0.f;// the distance the entries' bits were derived for
        float cutoutRayCamPos_[3] = {0.f, 0.f, 0.f};// ...and the camera position
        bool  cutoutRayClassified_ = false;// false after a full re-expansion (the bits were reset)

        // The lean fill's layout key: pendingTlasInstances_ is rewritten in
        // place only while these hold (buildTlas clears tlasFillValid_), and
        // tlasInstDirty_ names the instance ranges each frame slot still lacks.
        bool     tlasFillValid_        = false;
        uint64_t tlasFillSpansVersion_ = ~0ull;
        size_t   tlasFillEntryCount_   = 0;
        bool     tlasFillGenOn_        = false;
        std::array<DescDirtyRanges, kFramesInFlight> tlasInstDirty_{DescDirtyRanges{.maxRanges = 1024},
                                                                    DescDirtyRanges{.maxRanges = 1024}};
        // Entries whose TLAS row changed for a reason meshMovedBits_ does not
        // carry (a crossfade side, a ray shadow or cutout reclassification),
        // a bit each; consumed by the next fill.
        std::vector<uint32_t> tlasRowDirty_;
        std::vector<uint32_t> tlasSpanOrder_;// the lean fill's span order (stable, then volatile)

        uint64_t rayShadowEpochSeen_    = 0;
        uint64_t rayShadowSpansVersion_ = ~0ull;
        size_t   rayShadowEntryCount_   = 0;
        float    rayShadowCamPos_[3]    = {0.f, 0.f, 0.f};// the camera the states were derived for
        float    rayShadowStep_         = 0.f;            // 5 % of the smallest distance; 0: none set
        bool     rayShadowClassified_   = false;

        // TLAS refit instance staging, swapped with pendingTlasInstances_ so
        // both vectors keep their capacity across frames (a fresh 4.8 MB
        // allocation per moving frame at 100k instances otherwise).
        std::vector<VkAccelerationStructureInstanceKHR> tlasInstanceScratch_;

        // buildShadowDrawData's scratch: the view's own
        // (ViewPrepScratch::shadowChunks). Definition moved to
        // VulkanViewContext.hpp.
        using ShadowChunk = vulkan::impl::ShadowChunk;
        // The TLAS fill's per-chunk GeometryDesc dirty rows (same chunking).
        std::vector<std::vector<uint32_t>> tlasChunkDirty_;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_TLASSTATE_HPP
