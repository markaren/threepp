// GIState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns global illumination: the probe GI and path tracer passes and the probe grid's camera, light signature
// and fast-blend state.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct GIState {
        // (deferredShade_ moved to ViewContext — its per-frame-in-flight
        //  descriptor sets bind this view's G-buffer, reservoirs and sceneHdr,
        //  so the sets themselves are per-view. The pipeline could be shared;
        //  it is not, because a set can never be rewritten while it may be in
        //  flight (VUID-03047), and a per-view pass object rules that out
        //  structurally.)
        // World-space irradiance probe grid (multi-bounce GI for the deferred
        // gather — see vulkan/ProbeGI.hpp). Created alongside deferredShade_;
        // its SH buffer + grid UBO back the deferred set's bindings 36/37, so
        // it must exist whenever deferredShade_ does. probeGIEnabled_ gates the
        // per-frame update dispatch AND the shader-side sampling (grid UBO
        // enable) — default ON (VulkanRenderer::setProbeGI(false) turns it off).
        // probeGridDirty_ re-fits the grid to the scene AABB after each
        // structural scene rebuild (new/removed meshes ⇒ new bounds).
        std::unique_ptr<vulkan::ProbeGI> probeGI_;
        // Progressive path trace of the primary view (pathTraceBegin / pathTraceRead):
        // a pass of the frame graph while a session is active. See vulkan/PathTracer.hpp.
        std::unique_ptr<vulkan::PathTracer> pathTracer_;
        bool probeGIEnabled_  = true;
        bool probeGridDirty_  = true;
        // The primary camera's world position this frame: the centre the
        // probe cascades follow (ProbeGI::recentre) when the scene is too
        // large for one scene-fit grid. Taken from the primary's camera UBO
        // upload, never a secondary view's.
        float probeCamPos_[3] = {0.f, 0.f, 0.f};
        // Lighting-step detector for the probe update. The probes blend 1/32
        // of each update, one update per probe every 8 frames, so a switched
        // light lingered in the grid for thousands of frames; a step in any
        // light's power (probeLightingStepped) runs the next
        // kProbeFastBlendFrames of probe updates at the fast blend
        // (probe_update.comp, push-constant flags bit 1). updateLightsUbo fills
        // the current signature; the dispatch compares it with the last one.
        struct ProbeLightSig {
            std::array<uint32_t, 4> counts{};// dir, point, spot, rect
            std::vector<float>      lum;     // ambient, then one entry per light
            uint32_t                emissiveCount = 0;
            float                   emissivePower = 0.f;
            bool                    valid         = false;
        };
        ProbeLightSig probeLightSig_;
        ProbeLightSig probeLightSigPrev_;
        int           probeFastBlendFrames_ = 0;
        static constexpr int kProbeFastBlendFrames = 32;// four full sweeps
    };

}// namespace threepp::vulkan::impl
