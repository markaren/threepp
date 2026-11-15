// IndirectDrawState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the indirect draw builder's state: the per-span draw cache and its mutex, the scene build serial, and
// the stable, instance and class ids it assigns.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct IndirectDrawState {
        // ── Stable / semantic object IDs for the segmentation AOVs ───────
        // outIds.x is the per-frame visible-set index; outIds.y must be STABLE
        // across frames. We assign a dense 31-bit id per Object3D (keyed by the
        // process-stable Object3D::id) the first time it is drawn, so the label
        // survives add/remove/hide/LOD. Callers may override the instance id and
        // set an 8-bit semantic class (folded into outIds.z bits 8..15 by
        // buildIndirectDrawData). Maps are keyed by Object3D::id (never a raw
        // pointer) so a deleted object leaves an inert entry, never a dangling read.
        std::unordered_map<unsigned int, uint32_t> autoStableIds_;      // Object3D::id -> dense id
        std::unordered_map<unsigned int, uint32_t> instanceIdOverride_; // user-set instance id
        std::unordered_map<unsigned int, uint16_t> classIds_;           // user-set semantic class
        uint32_t nextAutoStableId_ = 1;// 0 reserved for sky / unassigned
        // Stable ids stop at INT32_MAX: LidarReturn::hitInstanceId reports the
        // same id as int32 with negative sentinels (-1 miss, -2 volume).
        static constexpr uint32_t kMaxStableId = 0x7FFFFFFFu;

        // The frame prep's per-frame scratch (buildIndirectDrawData's bucket
        // vectors among it) is the view's own: ViewContext::prep.
        // Definition moved to VulkanViewContext.hpp.
        using SpanDrawCache = vulkan::impl::SpanDrawCache;
        // Every view's build reads it, and the first this frame brings a span
        // up to date: under spanDrawCacheMutex_, the map and that refresh, as
        // views can build at once (THREEPP_VK_CPU_VIEWS); a span's rows are
        // read after the lock, by which time no other build writes them.
        std::unordered_map<unsigned int, SpanDrawCache> spanDrawCache_;
        std::mutex spanDrawCacheMutex_;
        uint64_t spanDrawCacheFrame_ = 0;
        uint64_t sceneBuildSerial_ = 0;// ensureSceneBuilt calls: meshMovedBits_ holds one call's moves
    };

}// namespace threepp::vulkan::impl
