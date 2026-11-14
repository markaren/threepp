// PostChainState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the upscalers (FSR, DLSS with Ray Reconstruction), TAA, the render scale, and the post knobs: depth of
// field, bloom, sharpen, motion blur.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct PostChainState {
        // (taa_ moved to ViewContext — TaaResolve owns the temporal HISTORY
        //  ping-pong, which is the single most view-specific thing in the
        //  renderer. See the pass-ownership note there.)
#if defined(THREEPP_WITH_FSR)
        // AMD FidelityFX FSR 3.1 upscaler. When its context creates successfully
        // (Windows only) it REPLACES the TAA temporal resolve — see the
        // fsrActive_ branch in VulkanCoreRecord.cpp. FSR writes its upscaled
        // linear-HDR output into TaaResolve's history write slot, so PostComposite
        // tonemaps it via the HDR-input path (its postFlags-bit-3 hdrMode +
        // recordPostFinalize) with no
        // descriptor rewrite. On create failure fsrActive_ stays false and the
        // TAA path runs unchanged. See FsrUpscaler.{hpp,cpp}.
        std::unique_ptr<vulkan::FsrUpscaler> fsr_;
        bool   fsrActive_    = false;// true once the FSR context created
        bool   fsrResetNext_ = true; // force FSR history reset on the next dispatch
        double fsrPrevTimeSec_ = -1.0;// for FSR frameTimeDelta (ms)
        // Camera params stashed at raster-camera-upload time (no camera at the
        // record site) — FSR needs near/far/vertical-FOV for depth
        // reconstruction — and the sub-pixel [-0.5,0.5] jitter applied to the
        // projection this frame, sourced from FSR's own sequence so the
        // dispatch jitterOffset matches: prepOut_.fsrCam* and .fsrJitter*.
        // Runtime on/off (setFsr), distinct from fsrActive_ (compiled + context
        // created). Default on so a THREEPP_WITH_FSR build uses FSR by default.
        bool   fsrEnabled_ = true;
#endif
#if defined(THREEPP_WITH_DLSS)
        // NVIDIA DLSS Super Resolution. Same seam as FSR (replaces the TAA
        // temporal resolve, writes into TaaResolve's history write slot) and
        // OUTRANKS it: when both are available + enabled, DLSS runs — see the
        // useDlss() branch in VulkanCoreRecord.cpp. On NGX init/feature-create
        // failure (non-RTX GPU, old driver) dlssActive_ stays false and the
        // FSR/TAA fallback runs unchanged. See DlssUpscaler.{hpp,cpp}.
        std::unique_ptr<vulkan::DlssUpscaler> dlss_;
        bool   dlssActive_    = false;// true once the NGX feature created
        bool   dlssResetNext_ = true; // force DLSS history reset on the next dispatch
        double dlssPrevTimeSec_ = -1.0;// for DLSS frameTimeDelta (ms)
        // The sub-pixel [-0.5,0.5] jitter applied to the projection this frame
        // (built-in Halton — DLSS's recommended sequence/phase count matches
        // jitterPhaseCount_'s FSR2-style scaling): prepOut_.dlssJitter*.
        // Runtime on/off (setDlss), distinct from dlssActive_. Default on so a
        // THREEPP_WITH_DLSS build uses DLSS by default.
        bool   dlssEnabled_ = true;
        // Self-heal budget for sticky NGX evaluate failures (renderFrame);
        // replenished by the display-resize recreate funnel.
        uint32_t dlssHealTries_ = 0;
        // DLSS Ray Reconstruction (setDlssRayReconstruction). Default on; it
        // runs only while DLSS itself runs (useDlss) and the denoiser is on
        // (denoiseEnabled_: RR takes the denoiser's place), and renderFrame
        // creates and releases its NGX feature to follow (syncDlssRayReconstruction).
        bool   dlssRrEnabled_ = true;
#endif
        // The upscalers are made when a frame first needs them, not by the
        // constructor. NGX's init runs on its own thread from there
        // (DlssUpscaler::ngxInit_); resolveUpscalers(), at the start of every
        // frame, creates the DLSS feature the first time DLSS is enabled
        // (dlssPending_ until then), the FSR context the first time FSR is the
        // upscaler to use, and the HDR plumbing either needs. So an
        // application that turns DLSS off before its first frame never waits
        // for NGX, and one that turns it on later gets the feature at the
        // next frame. Defined in VulkanCoreFrame.cpp.
        bool   dlssPending_ = false;

        // (bloom_ and post_ moved to ViewContext — bloom_ owns sceneHdr, the
        //  render-extent HDR target every view needs its own copy of, and
        //  post_ owns the display-extent hdrOut fed from it.)
        // Thin-lens depth of field on sceneHdr, recorded between the scene
        // dispatch and the bloom pyramid so bokeh still blooms + tone-maps
        // as HDR (see vulkan/DofPass.hpp). CoC is camera-derived: aperture
        // from camAperture_, focal length + sensor size from the camera's own
        // film gauge (filmHeightM_ below), focus plane at focusDistance_.
        // OFF by default (the whole pass is skipped; zero cost).
        std::unique_ptr<vulkan::DofPass> dof_;

        bool  dofEnabled_    = false;
        float focusDistance_ = 10.f;

        float bloomIntensity_ = 0.0f;

        float bloomThreshold_ = 1.0f;// soft-knee bright-pass cutoff (linear HDR)
        float bloomClamp_ = 0.0f;    // per-tap HDR cap before the bright pass; <= 0 = off
        float sharpenStrength_ = 0.5f;// post-TAA RCAS amount; 0 = off
        float motionBlurAmount_ = 0.f;// post-TAA motion blur: shutter open fraction
                                      // of the frame interval (0.5 = 180°); 0 = off
        float taaBlendAlpha_ = 0.16f;// 10% current, 90% history at the reference rate;
                                    // frame-rate-corrected per frame (see taaPrevTimeSec_)
        // Wall-clock anchor for the frame-rate-aware TAA blend. taaBlendAlpha_ is a
        // per-frame new-sample weight, so holding it fixed ties the history half-life
        // to frame count: the same 90 %/frame retention is an invisible ~10 ms ghost
        // at 200 fps but a long visible smear at 30 fps. Each frame the
        // weight is re-solved (in recordCommandBuffer, against kTaaRefFps) so
        // (1-alpha) is held constant in wall-clock time instead of per frame; the
        // shader's velocity/deviation gates are already per-frame-displacement based,
        // so only this base weight needs the correction.
        double taaPrevTimeSec_ = -1.0;

        // ── Lower-resolution render mode ────────────────────────────────
        // renderScale_ < 1 runs the deferred shade + the raster G-buffer at
        // renderExtent() instead of the swapchain extent. The TAA pass then
        // upsamples to full resolution (it dispatches at the swapchain extent
        // with a full-res history — see taa_resolve.comp); with TAA off a
        // bilinear blit upscales the low-res denoise output instead. 1.0 →
        // renderExtent() equals the swapchain extent and every pass behaves
        // exactly as before.
        float renderScale_ = 1.0f;
    };

}// namespace threepp::vulkan::impl
