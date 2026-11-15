// ShadowState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns shadow maps: the distance, the dummy bindings, the static-caster cache, the shadow pipelines and the
// sun and lamp slot assignments.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#ifndef THREEPP_VULKAN_STATE_SHADOWSTATE_HPP
#define THREEPP_VULKAN_STATE_SHADOWSTATE_HPP

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct ShadowState {
        float    shadowDistance_   = 0.f;// 0 = the camera's far plane, clamped

        // Bindings 80-83 of every view's deferred set when the view has no
        // live maps: a 1x1 D32 2D array (cascades and spots), a 1x1 D32 cube
        // array (points), both resting in DEPTH_STENCIL_READ_ONLY, and a zeroed
        // GpuShadowMapsUbo (flags 0: every lookup answers "no map").
        vulkan::ShadowDepthArray shadowDummyArray_{};
        vulkan::ShadowDepthArray shadowDummyCube_{};
        Buffer                   shadowDummyUbo_{};

        // The hash walk's by-products: the moving casters' entry indices in
        // span order, and each entry's span (the view's prep scratch,
        // ViewPrepScratch::shadowDynamicIdx / shadowSpanOfEntry). In steady
        // state (every layer skipped or served from its cache)
        // buildShadowDrawData walks only those; the caches hold the static
        // casters.
        // The static set as a generation: +1 whenever shadowStaticHash changes.
        // A cached layer is valid for one generation, never for a hash value:
        // a layer unseen while a mesh moves and settles would meet its old
        // hash again with the mesh at its old place. Every view's fit
        // compares, the first to see a new hash steps it: under
        // shadowStaticMutex_, as views can fit at once.
        uint64_t   shadowStaticHashPrev_ = 0, shadowStaticGen_ = 1;
        std::mutex shadowStaticMutex_;
        // THREEPP_VK_SHADOW_CACHE_DEBUG=1: a line per generation step (what
        // flipped) and per frame (layers cached / rebuilt / drawn / skipped).
        bool                 shadowCacheDebug_ = vulkan::env::flag("THREEPP_VK_SHADOW_CACHE_DEBUG");
        std::vector<uint8_t> shadowDynPrev_;

        // setShadowCacheStaleFrames: 0 (default) redraws a cascade whenever
        // the static caster set changes; n > 0 lets it hold stale casters for
        // up to n frames (ShadowMapPass::casterSnap). updateShadowCasterSnap
        // clears the bits of the casters that left the static set, marks the
        // cascades they leave stale and lists ShadowMapPass::onTop.
        uint32_t shadowCacheStaleFrames_ = 0;

        // Depth-only, reverse-Z, dynamic rendering into one D32 layer: the
        // indirect G-buffer draw reduced to position + the cutout test, with
        // depth clamp where the device has it and dynamic cull mode, front
        // face and depth bias.
        VkPipeline shadowPipeline_      = VK_NULL_HANDLE;// casters through shadow_depth.frag's alpha test
        VkPipeline shadowDepthPipeline_ = VK_NULL_HANDLE;// the rest: no fragment stage

        // Dir light 0 as updateLightsUbo packed it, with its LightShadow
        // settings (the environment's sun: the defaults, 2048 texels).
        struct ShadowSun {
            bool     valid = false;
            float    dir[3]{0.f, 1.f, 0.f};// toward the sun
            float    bias = 0.f, normalBias = 0.f, radius = 1.f, intensity = 1.f;
            uint32_t mapSize = 2048u;
        };
        ShadowSun shadowSun_;
        // The lamps holding a shadow map this frame, by slot (index = the
        // layer in the spot array / the cube in the point array), chosen from
        // the primary camera in updateLightsUbo (assignShadowSlots).
        struct ShadowLamp {
            bool         used = false;
            unsigned int id   = 0;// Object3D::id of the light
            float        pos[3]{}, dir[3]{};
            float        fov = 0.f;// spot: full cone angle of the map (rad)
            float        far = 1.f;
            float        bias = 0.f, normalBias = 0.f, radius = 1.f, intensity = 1.f;
        };
        std::vector<ShadowLamp> shadowSpotSlots_, shadowPointSlots_;
    };

}// namespace threepp::vulkan::impl

#endif// THREEPP_VULKAN_STATE_SHADOWSTATE_HPP
