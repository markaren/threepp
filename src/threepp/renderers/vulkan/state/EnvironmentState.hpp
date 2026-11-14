// EnvironmentState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the environment map and what feeds it: background colour, the baked Preetham sky, the prefiltered env
// and its extracted sun, the scene's env knobs and the env-sun policy.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct EnvironmentState {
        // Environment equirect (HDR float) used by the primary miss
        // for backgrounds and by closest-hit for a single mirror-reflection
        // IBL probe. Default is a 1×1 black dummy so descriptors are always
        // valid; replaced lazily when the scene's environment / background
        // texture is set or changed.
        Image2D envImage{};
        unsigned int envTextureIdUploaded = 0xFFFFFFFFu;
        bool envIsDefault  = true;
        bool envIsBgColor  = false;
        Color envBgColor{0.f, 0.f, 0.f};
        // threepp::Sky (vulkan/SkyBake): the visible Sky the last full scene
        // expansion met, if any (its box builds no entry), and whether envImage
        // is a NON-OWNING copy of the bake's persistent image — dropEnvImage
        // knows. refreshEnvTextureFromScene reads the Sky's uniforms every
        // frame; a change marks a re-bake, recorded in place at the top of the
        // frame's command buffer, with none of what a new env costs.
        Sky* skySeen_ = nullptr;
        bool envIsSky = false;
        PreethamSkyParams skyBaked_{};
        bool skyBakedValid_  = false;
        bool skyBakePending_ = false;
        std::unique_ptr<vulkan::SkyBake> skyBake_;

        // HDRI sun extracted at PMREM build (deferred only — see
        // envSunExtractionWanted). found=false when no sun / extraction off.
        // updateLightsUbo re-injects it as an analytic directional light.
        vulkan::EnvPrefilter::SunExtract envSun_{};
        // updateLightsUbo appended envSun_ as the LAST directional light this
        // frame (the one-sun policy can defer it to the scene's own sun). The
        // path tracer reads it: that light's disc is still in env mip 0, so a
        // ray it sends through glass can find it (sun caustics).
        bool envSunInjected_ = false;
        // The scene's environment/background knobs (SceneEnvKnobs.hpp), as
        // updateLightsUbo last packed them. envSun_ above stays in ENV space;
        // the lights UBO and envSunDirection()/envSunColor() turn and scale it
        // by these. envKnobsSeen_ is the frame loop's own copy for change
        // detection (a change wipes temporal history like a new env does);
        // invalid until the first frame, so the first read is not a change.
        vulkan::SceneEnvKnobs envKnobs_{};
        vulkan::SceneEnvKnobs envKnobsSeen_{};
        bool envKnobsSeenValid_ = false;

        // Env luminance CDF (Phase A: env importance sampling).
        // Conditional CDF: w×h R32F texture; row r holds the cumulative
        // distribution over columns at that latitude.
        // Marginal CDF: h×1 R32F texture; row r holds the cumulative marginal
        // 64×64 R8 blue-noise tile for sub-pixel jitter. Generated once via
        // void-and-cluster (Ulichney 1993) and uploaded at startup. Adjacent
        // pixels share correlated values (silhouette stability) but globally
        // decorrelated (no coherent shake). Animated temporally by offsetting
        // the lookup coords per frame; AA convergence via accumulator.
        Image2D blueNoiseImage{};

        // PMREM: GGX-prefiltered env mip chain. Built once per env
        // upload — see vulkan/EnvPrefilter.{hpp,cpp}. Owns the prefilter
        // compute pipeline + descriptor pool + source sampler; the host calls
        // envPrefilter_->buildPmrem(...) when scene.environment changes.
        std::unique_ptr<vulkan::EnvPrefilter> envPrefilter_;

        // HDRI sun extraction. refreshEnvTextureFromScene detects the env's
        // dominant compact bright source, prefilters PMREM mips 1+ from a
        // sun-clamped copy (no glossy / rough env lookup ever integrates the
        // raw ~10⁴:1 disc — the "bright spec blobs in reflections" artifact),
        // and updateLightsUbo re-injects the removed energy as an analytic
        // directional light (sharp correct sun highlight, jittered soft RT
        // shadows, GI bounce, water glints).
        //
        // ONE-SUN POLICY (consulted only when extraction is wanted): Auto
        // injects the extracted sun only while the scene has NO visible
        // DirectionalLight of its own — an explicit scene light claims the sun
        // role (scenes authored for raster renderers carry a stand-in sun light
        // because raster can't shadow from an env map; injecting the env sun on
        // top lights and shadows the scene with TWO suns). Always injects
        // regardless; Off skips extraction entirely.
        VulkanRenderer::EnvSunPolicy envSunPolicy_ = VulkanRenderer::EnvSunPolicy::Auto;
    };

}// namespace threepp::vulkan::impl
