// OceanState: one of the state groups VulkanRenderer::Impl inherits (VulkanCoreImpl.hpp).
// Owns the ocean surface bindings the shade reads: fine height and foam fields, crest foam, climate, wake and
// ripple views, and the foam detail and crest-pattern images.
//
// The members, their comments and their relative order are as they were in Impl;
// Impl's methods and the VulkanCore*.cpp TUs reach them unqualified, as before.
#pragma once

#include "StateCommon.hpp"

namespace threepp::vulkan::impl {

    struct OceanState {
        // Ocean fine-cascade normal-map source (deferred shade ocean binding).
        // Default 1×1 dummy R32F, replaced with the active DisplacedMesh's
        // cascade-2 height image when one is in the scene. deferred_shade.comp
        // samples this on the ocean (transmissive DisplacedMesh) at world-space XZ via
        // finite differences to perturb the macro normal — adds sub-mesh
        // chop detail (FFT cells finer than the 1 m mesh resolves).
        Image2D oceanFineHeightDummy{};
        VkImageView oceanFineHeightView   = VK_NULL_HANDLE;// either dummy or cascade-2 view
        VkSampler   oceanFineHeightSampler = VK_NULL_HANDLE;
        float       oceanFineTileSize     = 0.f;          // 0 disables sampling in shader
        float       oceanWaveScale        = 1.f;          // its params.waveScale (the path tracer's fine slope)
        bool        oceanChopNormals      = false;        // binding 13 is a ChopMoments chain and it is on this frame

        // World-space foam (deferred shade ocean binding). Built by
        // FoamWorldPipeline each frame from a DisplacedMesh's foamImage.
        // deferred_shade.comp samples it at world XZ on water hits — replaces
        // the per-vertex foam buffer that used to live on the BLAS. Held as a
        // dummy 1×1 R32F when no DisplacedMesh is in the scene.
        Image2D oceanFoamDummy{};
        VkImageView oceanFoamView   = VK_NULL_HANDLE;
        VkSampler   oceanFoamSampler = VK_NULL_HANDLE;
        float       oceanFoamTileSize = 0.f;              // 0 disables sampling
        // Crest foam (deferred shade bindings 91 and 92): the live ocean's
        // foam texture over its rest coordinates (water::CrestFoam), or the
        // foam dummy, and the foam photograph old foam opens against, uploaded
        // the first time an ocean turns the feature on (the foam detail tile
        // stands in until then). oceanCrestFoam: on this frame, the shade's spec constant 8.
        VkImageView oceanCrestFoamView    = VK_NULL_HANDLE;
        VkSampler   oceanCrestFoamSampler = VK_NULL_HANDLE;
        bool        oceanCrestFoam        = false;
        // DisplacedMesh::Params::waveReflections on this frame (the shade's
        // spec constant 9): the water's mirror ray shades water it meets.
        bool        oceanWaveReflections  = false;
        // The wave climate (DisplacedMesh::WaveClimate; deferred binding 93):
        // the live ocean's map, or the foam dummy, with the layout it is in.
        // oceanClimate: the shade's spec constant 12, bit 0 the map is on this
        // frame, bit 1 its crest foam keeps the climate's levels.
        VkImageView   oceanClimateView    = VK_NULL_HANDLE;
        VkSampler     oceanClimateSampler = VK_NULL_HANDLE;
        VkImageLayout oceanClimateLayout  = VK_IMAGE_LAYOUT_GENERAL;
        uint32_t      oceanClimate        = 0u;
        Image2D     crestFoamPatternImage{};
        // The wake field's state images (deferred shade binding 78): the live
        // ocean's, or WakeFieldPipeline's 1x1 dummy. The shade samples it only
        // through the patch table the ocean's GeometryDesc::foamAddress names.
        VkImageView oceanWakeView = VK_NULL_HANDLE;
        // Its ripples' slopes (binding 79), or the same dummy.
        VkImageView oceanRippleView = VK_NULL_HANDLE;

        // Tileable foam detail texture (deferred shade foam binding).
        // R = micro bubble brightness (three value-noise octaves matching the
        // old procedural micro look over a 4 m world mapping), G = ridged
        // lace/filament pattern (12 m mapping). Baked once at startup with a
        // full mip chain — replaces the per-pixel procedural noise octaves in
        // the foam shading: a sampled texture is band-limited at distance (no
        // shimmer under TAA) and 2 fetches replace ~20 hash evaluations.
        Image2D foamDetailImage{};
    };

}// namespace threepp::vulkan::impl
