// Pipeline creation and caching for the Wgpu renderer backend.
// Manages both standard material pipelines (keyed by feature bitmask)
// and custom ShaderMaterial pipelines (keyed by material identity + shader hash).

#ifndef THREEPP_WGPUPIPELINES_HPP
#define THREEPP_WGPUPIPELINES_HPP

#include "WgpuShaders.hpp"
#include "WgpuState.hpp"

#ifdef THREEPP_WGPU_GLSL_COMPAT
#include "WgpuShaderTranslator.hpp"
#endif

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
    class WgpuTexture;
}// namespace threepp

namespace threepp {
    class Matrix4;
}

namespace threepp::wgpu {

    struct PipelineEntry {
        WGPUShaderModule shader = nullptr;
        WGPURenderPipeline pipeline = nullptr;
        WGPUPipelineLayout layout = nullptr;
        WGPUBindGroupLayout bindGroupLayout = nullptr;
    };

    struct CustomPipelineEntry {
        WGPUShaderModule shader = nullptr;         // Combined WGSL: single vert+frag module
        WGPUShaderModule vertShader = nullptr;     // Separate modules: SPIR-V or per-stage WGSL
        WGPUShaderModule fragShader = nullptr;     // Separate modules: SPIR-V or per-stage WGSL
        WGPURenderPipeline pipeline = nullptr;
        WGPUPipelineLayout layout = nullptr;
        WGPUBindGroupLayout bindGroupLayout = nullptr;
        size_t shaderHash = 0;
        std::vector<WGPUBindGroupLayoutEntry> bglEntries;
        // GLSL-compat path: names of uniforms placed in the binding-2 UBO, sorted alphabetically.
        // The CPU packer must write these uniforms in the same order to match the SPIR-V layout.
        std::vector<std::string> customUniformNames;
        // Byte size of the binding-2 CustomUniforms UBO (0 if none).
        uint32_t customUniformSize = 0;
    };

    class WgpuPipelines {
    public:
        explicit WgpuPipelines(WgpuState& state);

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
        WgpuState& state_;
        std::unordered_map<uint64_t, PipelineEntry> pipelineCache_;
        std::unordered_map<unsigned int, CustomPipelineEntry> customPipelineCache_;

        std::vector<WGPUBindGroupLayoutEntry> buildBindGroupLayoutEntries(uint64_t features) const;

#ifdef THREEPP_WGPU_GLSL_COMPAT
        WgpuShaderTranslator translator_;
#endif
    };

}// namespace threepp::wgpu

#endif//THREEPP_WGPUPIPELINES_HPP
