// Light collection and GPU buffer packing for the Wgpu renderer.
// Traverses the scene graph to collect light data and uploads it
// to a shared GPU uniform buffer in world-space coordinates.

#ifndef THREEPP_WGPULIGHTS_HPP
#define THREEPP_WGPULIGHTS_HPP

#include "WgpuState.hpp"

#include <cstddef>
#include <vector>
#include <webgpu/webgpu.h>

namespace threepp {
    class Object3D;
}

namespace threepp::wgpu {

    class WgpuLights {
    public:
        explicit WgpuLights(WgpuState& state);

        // Collect lights from the scene and upload to GPU buffer.
        void update(Object3D& scene);

        // GPU buffer handle for bind group entries (binding 2).
        [[nodiscard]] WGPUBuffer uniformBuffer() const { return lightBuffer_; }

        // Current light uniform buffer size (derived from state's light limits).
        [[nodiscard]] size_t lightUniformSize() const;

        // Recreate the GPU buffer after light limits have changed.
        void recreateBuffer();

        void dispose();

    private:
        WgpuState& state_;
        WGPUBuffer lightBuffer_ = nullptr;
        std::vector<float> scratch_;   // reused scratch buffer — avoids per-frame heap alloc
        std::vector<float> uploaded_;  // last-uploaded data — skip GPU write when unchanged
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPULIGHTS_HPP
