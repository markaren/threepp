// VulkanFrameTypes — shared POD structs used by multiple Vulkan renderer modules.
//
// Only add types here when a second consumer appears. Structs that are internal
// to one module stay in that module's .hpp/.cpp. This file grows on demand.

#ifndef THREEPP_VULKAN_FRAME_TYPES_HPP
#define THREEPP_VULKAN_FRAME_TYPES_HPP

#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <cstdint>

namespace threepp::vulkan {

    // Per-BufferGeometry vertex/index/color buffer upload used by both the
    // ortho HUD overlay (OverlayPass) and the 3D hybrid overlay
    // (recordCommandBuffer's line-draw section). Keyed on raw
    // BufferGeometry*; geomId guards against recycled-pointer aliasing.
    struct LineRec {
        Buffer   vertex;
        Buffer   index;// VK_NULL_HANDLE if non-indexed
        Buffer   color;// VK_NULL_HANDLE if no "color" attribute
        Buffer   normal;// VK_NULL_HANDLE if no float "normal" attribute — the
                        // lit split-screen pane's mesh path binds it; lines and
                        // HUD art never carry one, so it costs those nothing
        uint32_t vertexCount     = 0;
        uint32_t indexCount      = 0;
        uint32_t positionVersion = 0;
        uint32_t indexVersion    = 0;
        uint32_t colorVersion    = 0;
        uint32_t normalVersion   = 0;
        // BufferGeometry::id (monotonic per construction). Detects pointer
        // recycle — a freed geometry's address reused for a new geometry
        // would have fresh version=0 fields that match the stale record,
        // but the id differs.
        unsigned int geomId    = 0;
        uint64_t     lastTouch = 0;// overlay-frame counter; for stale eviction
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_FRAME_TYPES_HPP
