// Shadow map rendering subsystem for the Wgpu renderer.
// Manages the depth array texture, shadow pipelines, and per-light
// shadow pass rendering.

#ifndef THREEPP_WGPUSHADOWMAP_HPP
#define THREEPP_WGPUSHADOWMAP_HPP

#include "WgpuShaders.hpp"
#include "WgpuState.hpp"

#include "threepp/math/Matrix4.hpp"

#include <cstdint>
#include <vector>
#include <webgpu/webgpu.h>

namespace threepp {
    class Object3D;
    class Light;
    class LightWithShadow;
}// namespace threepp

namespace threepp::wgpu {

    class WgpuGeometries;

    struct ShadowLightEntry {
        WGPUTextureView layerView = nullptr;
        Matrix4 lightVP;
        float bias = 0.005f;
        float normalBias = 0.0f;
    };

    class WgpuShadowMap {
    public:
        bool autoUpdate = true;
        bool needsUpdate = false;

        WgpuShadowMap(WgpuState& state, WgpuGeometries& geometries);

        // Process shadow-casting lights and render depth passes.
        // Call once per frame before the main render pass.
        void beginFrame(Object3D& scene);

        [[nodiscard]] bool isActive() const { return active_; }
        [[nodiscard]] int activeLightCount() const { return activeLightCount_; }

        // Accessors for bind group creation in the main render pass.
        [[nodiscard]] WGPUBuffer uniformBuffer() const { return uniformBuffer_; }
        [[nodiscard]] WGPUTextureView depthArrayView() const { return depthArrayView_; }
        [[nodiscard]] WGPUSampler comparisonSampler() const { return comparisonSampler_; }

        // Point light shadow accessors
        [[nodiscard]] WGPUBuffer ptUniformBuffer() const { return ptUniformBuffer_; }
        [[nodiscard]] WGPUTextureView ptDepthArrayView() const { return ptDepthArrayView_; }
        [[nodiscard]] int numPointShadows() const { return numPointShadows_; }

        void dispose();

    private:
        WgpuState& state_;
        WgpuGeometries& geometries_;

        bool initialized_ = false;
        bool active_ = false;
        int activeLightCount_ = 0;
        int numPointShadows_ = 0;

        // Dir/Spot shadow: depth array texture (one layer per shadow-casting light)
        WGPUTexture depthArrayTexture_ = nullptr;
        WGPUTextureView depthArrayView_ = nullptr;
        WGPUSampler comparisonSampler_ = nullptr;
        WGPUBuffer uniformBuffer_ = nullptr;

        // Point light shadow: depth 2D array texture (6 layers per point light)
        WGPUTexture ptDepthArrayTexture_ = nullptr;
        WGPUTextureView ptDepthArrayView_ = nullptr;
        WGPUBuffer ptUniformBuffer_ = nullptr;
        std::vector<WGPUTextureView> ptLayerViews_;

        // Depth-only render pipeline
        WGPURenderPipeline depthPipeline_ = nullptr;
        WGPUPipelineLayout depthPipelineLayout_ = nullptr;
        WGPUBindGroupLayout depthBindGroupLayout_ = nullptr;
        WGPUShaderModule depthShader_ = nullptr;
        WGPUBuffer depthTransformBuffer_ = nullptr;

        // Per-light entries (dir/spot)
        std::vector<ShadowLightEntry> lights_;

        void init();
        void renderPass(WGPUCommandEncoder encoder, Object3D& scene,
                        const Matrix4& lightVP, int lightIndex);
        void renderObject(WGPURenderPassEncoder pass, Object3D& object,
                          const Matrix4& lightVP);
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPUSHADOWMAP_HPP
