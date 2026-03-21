// Shared WebGPU state used by all Wgpu subsystem classes.

#ifndef THREEPP_WGPUSTATE_HPP
#define THREEPP_WGPUSTATE_HPP

#include <cstddef>
#include <cstdint>
#include <webgpu/webgpu.h>

namespace threepp::wgpu {

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

    // Runtime-configurable shadow limits.
    struct ShadowLimits {
        uint32_t mapSize = 1024;
        int maxShadowLights = 4;
        int maxShadowPointLights = 2;

        // Uniform buffer sizes derived from current limits.
        static constexpr size_t shadowUniformPerLight = 80;
        static constexpr size_t pointShadowPerLight = 32;

        [[nodiscard]] size_t shadowUniformSize() const {
            return 16 + static_cast<size_t>(maxShadowLights) * shadowUniformPerLight;
        }
        [[nodiscard]] size_t pointShadowUniformSize() const {
            return 16 + static_cast<size_t>(maxShadowPointLights) * pointShadowPerLight;
        }
    };

    // Holds the core WebGPU handles shared across subsystems.
    // Owned by WgpuRenderer::Impl; subsystems hold a reference.
    struct WgpuState {
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
        WGPUTextureFormat surfaceFormat = WGPUTextureFormat_BGRA8Unorm;
        LightLimits lightLimits;
        ShadowLimits shadowLimits;
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPUSTATE_HPP
