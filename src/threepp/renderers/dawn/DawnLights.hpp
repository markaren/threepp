// Light collection and GPU buffer packing for the Dawn renderer.
// Traverses the scene graph to collect light data and uploads it
// to a shared GPU uniform buffer in world-space coordinates.

#ifndef THREEPP_DAWNLIGHTS_HPP
#define THREEPP_DAWNLIGHTS_HPP

#include "DawnState.hpp"

#include <webgpu/webgpu.h>

namespace threepp {
    class Object3D;
}

namespace threepp::dawn {

    class DawnLights {
    public:
        explicit DawnLights(DawnState& state);

        // Collect lights from the scene and upload to GPU buffer.
        void update(Object3D& scene);

        // GPU buffer handle for bind group entries (binding 2).
        [[nodiscard]] WGPUBuffer uniformBuffer() const { return lightBuffer_; }

        void dispose();

    private:
        DawnState& state_;
        WGPUBuffer lightBuffer_ = nullptr;
    };

}// namespace threepp::dawn

#endif//THREEPP_DAWNLIGHTS_HPP
