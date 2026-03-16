// Shared WebGPU state used by all Dawn subsystem classes.

#ifndef THREEPP_DAWNSTATE_HPP
#define THREEPP_DAWNSTATE_HPP

#include <cstddef>
#include <cstdint>
#include <webgpu/webgpu.h>

namespace threepp::dawn {

    // Runtime-configurable light limits.
    struct LightLimits {
        int maxDirLights = 8;
        int maxPointLights = 8;
        int maxSpotLights = 8;
        int maxHemiLights = 8;

        // Compute the light uniform buffer size in bytes from current limits.
        // Layout: header(32) + dir(N*32) + point(N*48) + spot(N*64) + hemi(N*48)
        [[nodiscard]] size_t lightUniformSize() const {
            return 32
                + static_cast<size_t>(maxDirLights) * 32
                + static_cast<size_t>(maxPointLights) * 48
                + static_cast<size_t>(maxSpotLights) * 64
                + static_cast<size_t>(maxHemiLights) * 48;
        }
    };

    // Holds the core WebGPU handles shared across subsystems.
    // Owned by DawnRenderer::Impl; subsystems hold a reference.
    struct DawnState {
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
        WGPUTextureFormat surfaceFormat = WGPUTextureFormat_BGRA8Unorm;
        LightLimits lightLimits;
    };

}// namespace threepp::dawn

#endif//THREEPP_DAWNSTATE_HPP
