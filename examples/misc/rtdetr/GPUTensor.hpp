#pragma once

#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"

#include <cassert>
#include <initializer_list>
#include <memory>
#include <vector>

namespace rtdetr {

    /// Lightweight GPU tensor: a flat float32 storage buffer with shape metadata.
    /// Layout is NCHW (batch=1 implicit) for conv activations; for transformer
    /// tensors, shape is interpreted as [tokens, dim] or [dim] as appropriate.
    struct GPUTensor {
        std::vector<uint32_t> shape;
        std::unique_ptr<threepp::WgpuBuffer> buf;

        GPUTensor() = default;
        GPUTensor(GPUTensor&&) = default;
        GPUTensor& operator=(GPUTensor&&) = default;

        GPUTensor(const GPUTensor&) = delete;
        GPUTensor& operator=(const GPUTensor&) = delete;

        [[nodiscard]] bool valid() const { return buf != nullptr; }

        [[nodiscard]] uint32_t numel() const {
            uint32_t n = 1;
            for (auto s : shape) n *= s;
            return n;
        }
        [[nodiscard]] size_t bytes() const { return numel() * sizeof(float); }

        [[nodiscard]] uint32_t C() const { return shape.empty() ? 1 : shape[0]; }
        [[nodiscard]] uint32_t H() const { return shape.size() > 1 ? shape[1] : 1; }
        [[nodiscard]] uint32_t W() const { return shape.size() > 2 ? shape[2] : 1; }

        void upload(const float* data) {
            assert(buf);
            buf->write(data, bytes());
        }

        [[nodiscard]] threepp::WgpuBuffer&       buffer()       { return *buf; }
        [[nodiscard]] const threepp::WgpuBuffer& buffer() const { return *buf; }
    };

    inline GPUTensor makeTensor(threepp::WgpuRenderer& r, std::initializer_list<uint32_t> shape) {
        GPUTensor t;
        t.shape = shape;
        uint32_t n = 1;
        for (auto s : shape) n *= s;
        size_t byteCount = std::max<size_t>(n, 1u) * sizeof(float);
        t.buf = std::make_unique<threepp::WgpuBuffer>(
            r, byteCount, threepp::WgpuBuffer::Usage::Storage);
        return t;
    }

    /// Storage tensor that also supports CPU readback (Storage | CopySrc).
    inline GPUTensor makeReadbackTensor(threepp::WgpuRenderer& r,
                                        std::initializer_list<uint32_t> shape) {
        GPUTensor t;
        t.shape = shape;
        uint32_t n = 1;
        for (auto s : shape) n *= s;
        size_t byteCount = std::max<size_t>(n, 1u) * sizeof(float);
        t.buf = std::make_unique<threepp::WgpuBuffer>(
            r, byteCount, threepp::WgpuBuffer::Usage::StorageReadback);
        return t;
    }

    inline GPUTensor makeReadbackTensorV(threepp::WgpuRenderer& r, const std::vector<uint32_t>& shape) {
        GPUTensor t;
        t.shape = shape;
        uint32_t n = 1;
        for (auto s : shape) n *= s;
        size_t byteCount = std::max<size_t>(n, 1u) * sizeof(float);
        t.buf = std::make_unique<threepp::WgpuBuffer>(
            r, byteCount, threepp::WgpuBuffer::Usage::StorageReadback);
        return t;
    }

    inline GPUTensor makeTensorV(threepp::WgpuRenderer& r, const std::vector<uint32_t>& shape) {
        GPUTensor t;
        t.shape = shape;
        uint32_t n = 1;
        for (auto s : shape) n *= s;
        size_t byteCount = std::max<size_t>(n, 1u) * sizeof(float);
        t.buf = std::make_unique<threepp::WgpuBuffer>(
            r, byteCount, threepp::WgpuBuffer::Usage::Storage);
        return t;
    }

    /// Storage tensor sized for f16-packed weights: numel/2 u32 elements.
    /// Shape is kept in logical f32-element terms.
    inline GPUTensor makeF16WeightTensor(threepp::WgpuRenderer& r,
                                         const std::vector<uint32_t>& shape) {
        GPUTensor t;
        t.shape = shape;
        uint32_t n = 1;
        for (auto s : shape) n *= s;
        size_t byteCount = std::max<size_t>(n, 2u) * 2u;   // 2 bytes per f16
        if (byteCount % 4 != 0) byteCount += 2;            // 4-byte align
        t.buf = std::make_unique<threepp::WgpuBuffer>(
            r, byteCount, threepp::WgpuBuffer::Usage::Storage);
        return t;
    }

}// namespace rtdetr
