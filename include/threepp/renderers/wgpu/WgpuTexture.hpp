#ifndef THREEPP_WGPUTEXTURE_HPP
#define THREEPP_WGPUTEXTURE_HPP

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

    class WgpuRenderer;

    /// GPU-resident texture that can be used with compute and render pipelines.
    /// Supports storage (read/write from compute), sampling (read from fragment), and CPU upload.
    class WgpuTexture {

    public:
        enum class Format {
            RGBA32Float,
            RGBA16Float,   // 8 bytes/pixel — storage-writable + linearly filterable
            RG32Float,
            R32Float,      // 4 bytes/pixel — single channel float
            RGBA8Unorm
        };

        enum class Dimension {
            D2,
            D2Array,
            Cube
        };

        enum Usage : uint32_t {
            Storage        = 1 << 0,
            TextureBinding = 1 << 1,
            CopyDst        = 1 << 2,
            CopySrc        = 1 << 3
        };

        /// Create a GPU texture.
        /// @param renderer WgpuRenderer providing the WebGPU device/queue
        /// @param width Texture width in pixels
        /// @param height Texture height in pixels
        /// @param format Pixel format
        /// @param usage Bitwise OR of Usage flags
        WgpuTexture(WgpuRenderer& renderer, uint32_t width, uint32_t height,
                   Format format, uint32_t usage = Storage | TextureBinding | CopyDst);

        /// Create a GPU texture with explicit dimension (2D, 2DArray, or Cube).
        WgpuTexture(WgpuRenderer& renderer, uint32_t width, uint32_t height,
                   Format format, Dimension dimension,
                   uint32_t usage = TextureBinding | CopyDst,
                   uint32_t layers = 0);

        ~WgpuTexture();

        WgpuTexture(const WgpuTexture&) = delete;
        WgpuTexture& operator=(const WgpuTexture&) = delete;
        WgpuTexture(WgpuTexture&&) noexcept;
        WgpuTexture& operator=(WgpuTexture&&) noexcept;

        /// Upload CPU data to the texture.
        void write(const void* data, size_t size);

        /// Upload CPU data to a single face of a cube texture.
        /// @param face Face index (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z)
        /// @param data Pointer to pixel data (RGB data is automatically converted to RGBA)
        /// @param size Size of data in bytes
        void writeFace(uint32_t face, const void* data, size_t size);

        /// Upload CPU data to a single layer of a 2D array texture.
        void writeLayer(uint32_t layer, const void* data, size_t size);

        /// Get the raw WGPUTexture handle.
        [[nodiscard]] WGPUTexture texture() const { return texture_; }

        /// Get the texture view for binding.
        [[nodiscard]] WGPUTextureView view() const { return view_; }

        /// Get the sampler for sampling in shaders.
        [[nodiscard]] WGPUSampler sampler() const { return sampler_; }

        [[nodiscard]] uint32_t width() const { return width_; }
        [[nodiscard]] uint32_t height() const { return height_; }
        [[nodiscard]] Format format() const { return format_; }
        [[nodiscard]] Dimension dimension() const { return dimension_; }
        [[nodiscard]] uint32_t layers() const { return layers_; }

    private:
        WGPUDevice device_ = nullptr;
        WGPUQueue queue_ = nullptr;
        WGPUTexture texture_ = nullptr;
        WGPUTextureView view_ = nullptr;
        WGPUSampler sampler_ = nullptr;
        uint32_t width_ = 0;
        uint32_t height_ = 0;
        Format format_;
        Dimension dimension_ = Dimension::D2;
        uint32_t layers_ = 1;
        uint32_t bytesPerPixel_ = 0;

        void release();
    };

}// namespace threepp

#endif//THREEPP_WGPUTEXTURE_HPP
