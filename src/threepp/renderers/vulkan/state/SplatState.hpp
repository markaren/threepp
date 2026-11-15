// SplatState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns Gaussian splats: the splat pass, this frame's visible and parked clouds, the depth stamp, the splat
// volume (reflection) table and the depth AOV mode.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_SPLATSTATE_HPP
#define THREEPP_VULKAN_STATE_SPLATSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct SplatState {
        VkDescriptorSetLayout splatStampSetLayout_      = VK_NULL_HANDLE;
        VkDescriptorPool      splatStampPool_           = VK_NULL_HANDLE;
        VkPipelineLayout      splatStampPipelineLayout_ = VK_NULL_HANDLE;
        VkPipeline            splatStampPipeline_       = VK_NULL_HANDLE;
        // The stamp into BillboardHdrLayer's own one-sample depth (no colour).
        VkPipeline            splatStampLayerPipeline_  = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kFramesInFlight> splatStampSets_{};
        std::array<uint64_t, kFramesInFlight>        splatStampSetSerials_{};// imageViewSerial
        // Sticky: set the first frame a scene holds BOTH splat clouds and
        // overlay content, and never cleared. It forces the depth AOV on
        // (splatDepthAov()) because the stamp has nothing to read otherwise.
        // Sticky rather than per-frame because turning the AOV off again means
        // reallocating the render-extent resources — a device idle — every
        // time a gizmo is hidden and shown. The cost of leaving it on is one
        // R32 image per frame in flight plus a guarded store per covered pixel.
        bool splatOverlayDepth_ = false;

        // Gaussian splats (SplatCloud), composited into sceneHdr between the
        // deferred shade and the DoF — linear HDR, pre-post, so splats get
        // DoF/bloom/tonemap/TAA through the same path as everything else. See
        // vulkan/SplatPass.hpp. Primary view only: the pass owns one set of
        // scratch buffers sized to the extent it was last resized to, and a
        // secondary view would fight it for them.
        // Allocates nothing until a scene actually contains a SplatCloud.
        std::unique_ptr<vulkan::SplatPass> splat_;
        // This frame's SplatClouds, gathered by the sidecar collector in
        // VulkanRenderer::render (see collectSplatClouds) — outside the
        // snapshot machinery, like collectWorldSprites, because a SplatCloud
        // is not a kind the G-buffer path knows how to draw.
        std::vector<vulkan::SplatPass::CloudEntry> lastVisibleSplats_;
        // Splat clouds in the scene but not effectively visible this frame —
        // parked, not evicted: their GPU buffers stay so a visibility toggle
        // costs nothing (see SplatPass::syncClouds).
        std::vector<const SplatCloud*> lastParkedSplats_;
        // Camera-derived splat parameters, stashed by the collector (the last
        // point in the frame that holds a Camera&). The jitter shear is added
        // at record time because the raster does not choose it until later.
        vulkan::SplatPass::RecordParams splatParams_{};
        // This frame's REVERSE-Z projection, stashed alongside — recordSplats
        // composes it with taaSkyReproj_ to get view space -> previous clip.
        float splatProjRevZ_[16]{};
        // setSplatDebugChecksum — the determinism gate's switch. Off by
        // default; the hashes cost two extra dispatches and an atomic per
        // composited pixel.
        bool splatChecksum_ = false;

        // ── Splat reflection volumes (plans/splat-volume-reflections.md) ────
        // Bindings 70/71 of EVERY view's deferred set. The particle-density
        // block above, mirrored member for member and for the same reasons:
        // both handles are owned HERE rather than by SplatPass because they
        // must be valid from the renderer's first descriptor write, which
        // happens before any cloud exists and keeps happening on scenes that
        // never get one.
        //
        //   splatVolumeUbos_  — per-FIF SplatVolumeUboGpu (std140, 912 B),
        //                       rewritten every frame; handle never changes.
        //   splatVolumeDummy_ — 1x1x1 RGBA16F in GENERAL, bound to every array
        //                       slot no live volume occupies.
        Buffer  splatVolumeUbos_[kFramesInFlight]{};
        Image2D splatVolumeDummy_{};
        // At least one baked volume is bound this frame. Feeds the shade's
        // flags bit 12 — and ONLY on the primary view (recordSceneDispatch),
        // because doc/vulkan_splats.md's scope wall keeps splats out of sensor
        // AOVs and a secondary's water reflection is a sensor AOV.
        bool     splatVolumeActiveThisFrame_ = false;
        // splatVolumeBindKey() each FIF's deferred set was last written with.
        // Not equal ⇒ that set names the wrong volume views and must be
        // rewritten before it is recorded.
        uint64_t splatVolumeDescKey_[kFramesInFlight]{};

        // Gaussian-splat depth AOV. OFF by default, and off means the backing
        // image is one texel: a full-res r32f per frame in flight is ~25 MB at
        // 1080p that a scene with no splats would never read.
        // Crossing Off reallocates the render-extent resources, exactly like
        // setGbufferMsaa — a setup knob, not a per-frame one. Expected <->
        // Median is a UBO flag and reallocates nothing.
        using SplatDepthMode = VulkanRenderer::SplatDepthMode;

        SplatDepthMode splatDepthMode_ = SplatDepthMode::Off;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_SPLATSTATE_HPP
