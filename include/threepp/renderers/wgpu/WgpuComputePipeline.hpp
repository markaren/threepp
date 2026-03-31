#ifndef THREEPP_WGPUCOMPUTEPIPELINE_HPP
#define THREEPP_WGPUCOMPUTEPIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>

// Forward declare WebGPU types
typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUQueueImpl* WGPUQueue;
typedef struct WGPUComputePipelineImpl* WGPUComputePipeline;
typedef struct WGPUComputePassEncoderImpl* WGPUComputePassEncoder;
typedef struct WGPUBindGroupLayoutImpl* WGPUBindGroupLayout;
typedef struct WGPUPipelineLayoutImpl* WGPUPipelineLayout;
typedef struct WGPUShaderModuleImpl* WGPUShaderModule;

namespace threepp {

    class WgpuRenderer;
    class WgpuTexture;
    class WgpuBuffer;

    /// Compute pipeline wrapper for dispatching WGSL compute shaders.
    class WgpuComputePipeline {

    public:
        /// Binding types for resources.
        enum class BindingType {
            StorageTextureWrite,   ///< Read-write storage texture (write_only in WGSL)
            StorageTextureRead,    ///< Read-only storage texture (read in WGSL via texture_2d)
            Texture,               ///< Sampled texture (texture_2d + textureLoad)
            UniformBuffer,         ///< Uniform buffer
            StorageBuffer,         ///< Read-write storage buffer (var<storage, read_write>)
            StorageBufferRead,     ///< Read-only storage buffer (var<storage, read>)
        };

        /// Create a compute pipeline from WGSL source.
        /// @param renderer WgpuRenderer providing the WebGPU device/queue
        /// @param wgslSource Complete WGSL shader source code
        /// @param entryPoint Name of the @compute entry point function
        WgpuComputePipeline(WgpuRenderer& renderer, const std::string& wgslSource,
                        const std::string& entryPoint);

        ~WgpuComputePipeline();

        WgpuComputePipeline(const WgpuComputePipeline&) = delete;
        WgpuComputePipeline& operator=(const WgpuComputePipeline&) = delete;

        /// Set a storage texture binding (write-only).
        void setStorageTexture(uint32_t binding, WgpuTexture& texture);

        /// Set a read-only texture binding (texture_2d).
        void setTexture(uint32_t binding, WgpuTexture& texture);

        /// Set a uniform buffer binding.
        void setUniformBuffer(uint32_t binding, WgpuBuffer& buffer);

        /// Set a read-write storage buffer binding (var<storage, read_write>).
        void setStorageBuffer(uint32_t binding, WgpuBuffer& buffer);

        /// Set a read-only storage buffer binding (var<storage, read>).
        void setStorageBufferRead(uint32_t binding, WgpuBuffer& buffer);

        /// Encode a dispatch into an existing compute pass (no submission).
        void encode(WGPUComputePassEncoder pass, uint32_t x, uint32_t y = 1, uint32_t z = 1);

        /// Replace the shader module with new WGSL source (forces pipeline rebuild on next dispatch).
        void replaceShader(const std::string& wgslSource);

        /// Create an encoder, dispatch, and submit (convenience wrapper).
        void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);

    private:
        struct BindingInfo;
        struct Impl;
        Impl* impl_;
    };

}// namespace threepp

#endif//THREEPP_WGPUCOMPUTEPIPELINE_HPP
