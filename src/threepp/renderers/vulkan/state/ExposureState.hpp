// ExposureState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns exposure and tone mapping: the tone-mapping operator, the physical camera and photometric light units,
// the pre-exposure history, white balance, and auto exposure.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_EXPOSURESTATE_HPP
#define THREEPP_VULKAN_STATE_EXPOSURESTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct ExposureState {
        // Mirrored from VulkanRenderer (Renderer base) at the start of each
        // render() so the renderFrame path can read them without a pointer back
        // into the public class. Synced unconditionally — toggling these never
        // resets the accumulator (tone mapping is display-only).
        ToneMapping toneMapping_ = ToneMapping::None;
        float       toneMappingExposure_ = 1.f;

        // ── Physical camera exposure (setPhysicalCamera) ────────────────────
        // While enabled, exposure is derived from real camera parameters
        // instead of toneMappingExposure_:
        //   EV100 = log2(N²/t · 100/S) − evComp, exposure = 1/(1.2·2^EV100)
        // and the shade/resolve PRE-EXPOSES sceneHdr (multiplies the store by
        // the full exposure) so rgba16f survives 100k-lux daylight radiance.
        // Defaults are the sunny-16 daylight setting (f/16, 1/125 s, ISO 100):
        // a 100k-lux sun-lit scene lands at mid-gray. Legacy mode (default
        // off) leaves every multiply at 1.0 — numerically byte-identical.
        bool  physicalCamera_ = false;
        float camAperture_    = 16.f;       // f-number N
        float camShutter_     = 1.f / 125.f;// seconds
        float camIso_         = 100.f;
        float camEvComp_      = 0.f;// EV compensation (+1 doubles brightness)
        // Raw white-balance inputs (PostComposite only keeps the derived
        // Bradford matrix) so whiteBalance() can report them back.
        float wbTemperatureK_ = 6500.f;
        float wbTint_         = 0.f;

        float    preExpHist_[kFramesInFlight] = {1.f, 1.f};
        uint32_t preExpBits_     = 0x3F800000u;// float bits of this frame's preExposure

        // Analytic light intensities are photometric while enabled: dir = lux
        // (as-is), point = lumens (→ candela Φ/4π at upload), spot = lumens
        // (→ Φ/π, Frostbite's cone-angle-invariant convention), rect/emissive
        // = nits (as-is). Pair with setPhysicalCamera — 100k lux needs a
        // physical exposure to land on screen.
        bool physicalLightUnits_ = false;

        // ── Auto-exposure state ───────────────────────────────────────────────
        std::unique_ptr<vulkan::AutoExposure> autoExposure_;
        bool   autoExposureEnabled_ = false;
        float  autoExpSpeed_        = 2.0f;
        float  autoExpMinEV_        = -3.0f;
        float  autoExpMaxEV_        =  3.0f;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_EXPOSURESTATE_HPP
