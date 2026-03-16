// Pipeline creation and caching for the Dawn renderer backend.
// Manages both standard material pipelines (keyed by feature bitmask)
// and custom ShaderMaterial pipelines (keyed by material identity + shader hash).

#ifndef THREEPP_DAWNPIPELINES_HPP
#define THREEPP_DAWNPIPELINES_HPP

#include "DawnShaders.hpp"
#include "DawnState.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <webgpu/webgpu.h>

namespace threepp {
    class Material;
    class ShaderMaterial;
    class Mesh;
    class Camera;
    class GPUTexture;
}// namespace threepp

namespace threepp {
    class Matrix4;
}

namespace threepp::dawn {

    struct PipelineEntry {
        WGPUShaderModule shader = nullptr;
        WGPURenderPipeline pipeline = nullptr;
        WGPUPipelineLayout layout = nullptr;
        WGPUBindGroupLayout bindGroupLayout = nullptr;
    };

    struct CustomPipelineEntry {
        WGPUShaderModule shader = nullptr;
        WGPURenderPipeline pipeline = nullptr;
        WGPUPipelineLayout layout = nullptr;
        WGPUBindGroupLayout bindGroupLayout = nullptr;
        size_t shaderHash = 0;
        std::vector<WGPUBindGroupLayoutEntry> bglEntries;
    };

    class DawnPipelines {
    public:
        explicit DawnPipelines(DawnState& state);

        // Get or create a standard pipeline for the given feature bitmask.
        // surfaceFormat and sampleCount configure the render target.
        PipelineEntry& getOrCreatePipeline(uint64_t features,
                                           WGPUTextureFormat surfaceFormat,
                                           uint32_t sampleCount);

        // Get or create a custom pipeline for a ShaderMaterial.
        // Returns the pipeline entry. Rebuilds if the shader source changed.
        CustomPipelineEntry& getOrCreateCustomPipeline(
                ShaderMaterial* sm,
                WGPUTextureFormat surfaceFormat,
                uint32_t sampleCount);

        // Release all cached pipelines (e.g. when sample count changes).
        void invalidateAll();

        void dispose();

        // Remove cached pipeline for a disposed material (by material id).
        void onMaterialDispose(unsigned int materialId);

        [[nodiscard]] size_t standardCount() const { return pipelineCache_.size(); }
        [[nodiscard]] size_t customCount() const { return customPipelineCache_.size(); }

    private:
        DawnState& state_;
        std::unordered_map<uint64_t, PipelineEntry> pipelineCache_;
        std::unordered_map<unsigned int, CustomPipelineEntry> customPipelineCache_;

        static std::vector<WGPUBindGroupLayoutEntry> buildBindGroupLayoutEntries(uint64_t features);
    };

}// namespace threepp::dawn

#endif//THREEPP_DAWNPIPELINES_HPP
