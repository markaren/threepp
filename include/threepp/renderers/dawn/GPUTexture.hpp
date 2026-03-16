#ifndef THREEPP_GPUTEXTURE_HPP
#define THREEPP_GPUTEXTURE_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Forward declare WebGPU types to avoid leaking webgpu.h
typedef struct WGPUTextureImpl* WGPUTexture;
typedef struct WGPUTextureViewImpl* WGPUTextureView;
typedef struct WGPUSamplerImpl* WGPUSampler;
typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUQueueImpl* WGPUQueue;

namespace threepp {

    class DawnRenderer;

    /// GPU-resident texture that can be used with compute and render pipelines.
    /// Supports storage (read/write from compute), sampling (read from fragment), and CPU upload.
    class GPUTexture {

    public:
        enum class Format {
            RGBA32Float,
            RG32Float,
            RGBA8Unorm
        };

        enum Usage : uint32_t {
            Storage        = 1 << 0,
            TextureBinding = 1 << 1,
            CopyDst        = 1 << 2,
            CopySrc        = 1 << 3
        };

        /// Create a GPU texture.
        /// @param renderer DawnRenderer providing the WebGPU device/queue
        /// @param width Texture width in pixels
        /// @param height Texture height in pixels
        /// @param format Pixel format
        /// @param usage Bitwise OR of Usage flags
        GPUTexture(DawnRenderer& renderer, uint32_t width, uint32_t height,
                   Format format, uint32_t usage = Storage | TextureBinding | CopyDst);

        ~GPUTexture();

        GPUTexture(const GPUTexture&) = delete;
        GPUTexture& operator=(const GPUTexture&) = delete;
        GPUTexture(GPUTexture&&) noexcept;
        GPUTexture& operator=(GPUTexture&&) noexcept;

        /// Upload CPU data to the texture.
        void write(const void* data, size_t size);

        /// Get the raw WGPUTexture handle.
        [[nodiscard]] WGPUTexture texture() const { return texture_; }

        /// Get the texture view for binding.
        [[nodiscard]] WGPUTextureView view() const { return view_; }

        /// Get the sampler for sampling in shaders.
        [[nodiscard]] WGPUSampler sampler() const { return sampler_; }

        [[nodiscard]] uint32_t width() const { return width_; }
        [[nodiscard]] uint32_t height() const { return height_; }
        [[nodiscard]] Format format() const { return format_; }

    private:
        WGPUDevice device_ = nullptr;
        WGPUQueue queue_ = nullptr;
        WGPUTexture texture_ = nullptr;
        WGPUTextureView view_ = nullptr;
        WGPUSampler sampler_ = nullptr;
        uint32_t width_ = 0;
        uint32_t height_ = 0;
        Format format_;
        uint32_t bytesPerPixel_ = 0;

        void release();
    };

}// namespace threepp

#endif//THREEPP_GPUTEXTURE_HPP
