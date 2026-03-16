// Shared WebGPU state used by all Dawn subsystem classes.

#ifndef THREEPP_DAWNSTATE_HPP
#define THREEPP_DAWNSTATE_HPP

#include <webgpu/webgpu.h>

namespace threepp::dawn {

    // Holds the core WebGPU handles shared across subsystems.
    // Owned by DawnRenderer::Impl; subsystems hold a reference.
    struct DawnState {
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
        WGPUTextureFormat surfaceFormat = WGPUTextureFormat_BGRA8Unorm;
    };

}// namespace threepp::dawn

#endif//THREEPP_DAWNSTATE_HPP
