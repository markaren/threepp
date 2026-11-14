// LensState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the camera sensor image: the sensor pass, lens distortion and sensor noise, and the lens rig (its face
// cameras and views).
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct LensState {
        // Final image formation (lens warp + sensor noise). Runs after the
        // overlay pass so it applies to everything the camera sees, not just
        // the parts of the frame that existed before the overlay composited.
        std::unique_ptr<vulkan::SensorPass> sensorPass_;

        // ── Lens + sensor (the non-pinhole half of the camera) ──────────────
        // Both default to inactive, and while inactive the RCAS stage keeps
        // its original texelFetch/CAS path → byte-identical output.
        LensDistortion lens_{};
        VulkanRenderer::SensorNoise sensorNoise_{};
        // Frame counter driving the noise seed. Advances once per recorded
        // frame while noise is on; resetSensorNoise() rewinds it so the same
        // seed replays the same sequence from the top of an episode.
        uint32_t sensorNoiseFrame_ = 0u;

        // Overscan factor for the lens warp: the scene is rendered with the
        // frustum widened by this much so barrel distortion has real geometry
        // to gather into the output corners. 1 = off.
        float lensOverscan_ = 1.f;

        // The layout render() last asked for, and what it was computed from
        // (recomputed only when one of those changes: lens, intrinsics, size).
        bool                   lensRigWanted_ = false;
        vulkan::lensrig::Frame lensRigLayout_{};
        LensDistortion         lensRigLens_{};
        float                  lensRigNormK_[4]{};
        VkExtent2D             lensRigExt_{0, 0};
        // Per face: its camera (kept for the renderer's life: a removed view
        // still names it until the frame boundary frees it), its view handle
        // (0 = none) and the size that view was created at.
        std::array<std::unique_ptr<PerspectiveCamera>, vulkan::lensrig::kFaceCount> lensRigCams_{};
        std::array<uint32_t, vulkan::lensrig::kFaceCount>                         lensRigViews_{};
        std::array<VkExtent2D, vulkan::lensrig::kFaceCount>                       lensRigViewExt_{};
        // ...and the extent it renders at: the primary's render scale of
        // that size, upsampled by the view's own TAA resolve.
        std::array<VkExtent2D, vulkan::lensrig::kFaceCount>                       lensRigViewRender_{};
        // The rig as the last frame's sensor pass resampled it (faceMask = the
        // faces whose views rendered that frame): the AOV readback composes
        // its labels from the same views, with the same ownership.
        vulkan::lensrig::Frame lensRigLast_{};
        bool                   lensRigLastActive_ = false;
        // The camera's clip planes that frame (every view shares them), for
        // re-expressing a side view's depth along the camera's own axis.
        float lensRigNear_ = 0.1f, lensRigFar_ = 1000.f;

        // The sensor stage is built after the secondary views (it samples the
        // rig's), so addTailPasses only marks that this frame wants it.
        bool sensorStagePending_ = false;
    };

}// namespace threepp::vulkan::impl
