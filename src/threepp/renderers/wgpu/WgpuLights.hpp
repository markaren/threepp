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
        // useLegacyLights: when true, packs a flag the shader uses to skip the
        // diffuse 1/PI divisor and use the legacy distance falloff (matches
        // GL with PHYSICALLY_CORRECT_LIGHTS undefined). When false, shader
        // applies physical Lambert + Frostbite distance falloff.
        void update(Object3D& scene, bool useLegacyLights);

        // GPU buffer handle for bind group entries (binding 2).
        [[nodiscard]] WGPUBuffer uniformBuffer() const { return lightBuffer_; }

        // Current light uniform buffer size (derived from state's light limits).
        [[nodiscard]] size_t lightUniformSize() const;

        // Raw light data computed by the last update() call.
        // Used by the renderer to upload a per-render snapshot to a pool buffer,
        // so concurrent render passes in the same frame each see their own light data.
        [[nodiscard]] const std::vector<float>& currentData() const { return scratch_; }

        // Recreate the GPU buffer after light limits have changed.
        void recreateBuffer();

        // True if the last update() collected at least one RectAreaLight.
        // Used by the renderer to feature-gate LTC bindings + shader path.
        [[nodiscard]] bool hasRectAreaLights() const { return hasRectAreaLights_; }

        void dispose();

    private:
        WgpuState& state_;
        WGPUBuffer lightBuffer_ = nullptr;
        std::vector<float> scratch_;   // reused scratch buffer — avoids per-frame heap alloc
        std::vector<float> uploaded_;  // last-uploaded data — skip GPU write when unchanged
        bool hasRectAreaLights_ = false;
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPULIGHTS_HPP
