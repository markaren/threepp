#ifndef THREEPP_DAWNCOMPUTEPIPELINE_HPP
#define THREEPP_DAWNCOMPUTEPIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>

// Forward declare WebGPU types
typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUQueueImpl* WGPUQueue;
typedef struct WGPUComputePipelineImpl* WGPUComputePipeline;
typedef struct WGPUBindGroupLayoutImpl* WGPUBindGroupLayout;
typedef struct WGPUPipelineLayoutImpl* WGPUPipelineLayout;
typedef struct WGPUShaderModuleImpl* WGPUShaderModule;

namespace threepp {

    class DawnRenderer;
    class DawnTexture;
    class DawnBuffer;

    /// Compute pipeline wrapper for dispatching WGSL compute shaders.
    class DawnComputePipeline {

    public:
        /// Binding types for resources.
        enum class BindingType {
            StorageTextureWrite,   ///< Read-write storage texture (write_only in WGSL)
            StorageTextureRead,    ///< Read-only storage texture (read in WGSL via texture_2d)
            Texture,               ///< Sampled texture (texture_2d + textureLoad)
            UniformBuffer          ///< Uniform buffer
        };

        /// Create a compute pipeline from WGSL source.
        /// @param renderer DawnRenderer providing the WebGPU device/queue
        /// @param wgslSource Complete WGSL shader source code
        /// @param entryPoint Name of the @compute entry point function
        DawnComputePipeline(DawnRenderer& renderer, const std::string& wgslSource,
                        const std::string& entryPoint);

        ~DawnComputePipeline();

        DawnComputePipeline(const DawnComputePipeline&) = delete;
        DawnComputePipeline& operator=(const DawnComputePipeline&) = delete;

        /// Set a storage texture binding (write-only).
        void setStorageTexture(uint32_t binding, DawnTexture& texture);

        /// Set a read-only texture binding (texture_2d).
        void setTexture(uint32_t binding, DawnTexture& texture);

        /// Set a uniform buffer binding.
        void setUniformBuffer(uint32_t binding, DawnBuffer& buffer);

        /// Dispatch compute workgroups.
        void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);

    private:
        struct BindingInfo;
        struct Impl;
        Impl* impl_;
    };

}// namespace threepp

#endif//THREEPP_DAWNCOMPUTEPIPELINE_HPP
