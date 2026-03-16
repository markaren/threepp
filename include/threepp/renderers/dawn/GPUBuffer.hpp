#ifndef THREEPP_GPUBUFFER_HPP
#define THREEPP_GPUBUFFER_HPP

#include <cstddef>
#include <cstdint>

// Forward declare WebGPU types
typedef struct WGPUBufferImpl* WGPUBuffer;
typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUQueueImpl* WGPUQueue;

namespace threepp {

    class DawnRenderer;

    /// GPU-resident uniform buffer for use with compute and render pipelines.
    class GPUBuffer {

    public:
        /// Create a GPU buffer.
        /// @param renderer DawnRenderer providing the WebGPU device/queue
        /// @param size Buffer size in bytes
        GPUBuffer(DawnRenderer& renderer, size_t size);

        ~GPUBuffer();

        GPUBuffer(const GPUBuffer&) = delete;
        GPUBuffer& operator=(const GPUBuffer&) = delete;
        GPUBuffer(GPUBuffer&&) noexcept;
        GPUBuffer& operator=(GPUBuffer&&) noexcept;

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

#endif//THREEPP_GPUBUFFER_HPP
