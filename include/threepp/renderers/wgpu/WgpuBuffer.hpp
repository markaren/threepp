#ifndef THREEPP_WGPUBUFFER_HPP
#define THREEPP_WGPUBUFFER_HPP

#include <cstddef>
#include <cstdint>

// Forward declare WebGPU types
typedef struct WGPUBufferImpl* WGPUBuffer;
typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUQueueImpl* WGPUQueue;

namespace threepp {

    class WgpuRenderer;

    /// GPU-resident buffer for use with compute and render pipelines.
    class WgpuBuffer {

    public:
        enum class Usage { Uniform, Storage, Vertex };

        /// Create a GPU buffer.
        /// @param renderer WgpuRenderer providing the WebGPU device/queue
        /// @param size Buffer size in bytes
        /// @param usage Uniform (default) or Storage
        WgpuBuffer(WgpuRenderer& renderer, size_t size, Usage usage = Usage::Uniform);

        ~WgpuBuffer();

        WgpuBuffer(const WgpuBuffer&) = delete;
        WgpuBuffer& operator=(const WgpuBuffer&) = delete;
        WgpuBuffer(WgpuBuffer&&) noexcept;
        WgpuBuffer& operator=(WgpuBuffer&&) noexcept;

        /// Write data to the buffer.
        void write(const void* data, size_t size, size_t offset = 0);

        /// Get the raw WGPUBuffer handle.
        [[nodiscard]] WGPUBuffer buffer() const { return buffer_; }

        [[nodiscard]] size_t size() const { return size_; }

    private:
        WGPUDevice device_ = nullptr;
        WGPUQueue queue_ = nullptr;
        WGPUBuffer buffer_ = nullptr;
        size_t size_ = 0;

        void release();
    };

}// namespace threepp

#endif//THREEPP_WGPUBUFFER_HPP
