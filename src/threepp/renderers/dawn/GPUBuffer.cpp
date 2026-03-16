
#include "threepp/renderers/dawn/GPUBuffer.hpp"
#include "threepp/renderers/DawnRenderer.hpp"

#include <webgpu/webgpu.h>

using namespace threepp;

GPUBuffer::GPUBuffer(DawnRenderer& renderer, size_t size)
    : device_(static_cast<WGPUDevice>(renderer.nativeDevice())),
      queue_(static_cast<WGPUQueue>(renderer.nativeQueue())),
      size_(size) {

    WGPUBufferDescriptor desc{};
    desc.label = {.data = "gpu_buffer", .length = 10};
    desc.size = size;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    buffer_ = wgpuDeviceCreateBuffer(device_, &desc);
}

GPUBuffer::~GPUBuffer() {
    release();
}

GPUBuffer::GPUBuffer(GPUBuffer&& other) noexcept
    : device_(other.device_), queue_(other.queue_),
      buffer_(other.buffer_), size_(other.size_) {
    other.buffer_ = nullptr;
}

GPUBuffer& GPUBuffer::operator=(GPUBuffer&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        queue_ = other.queue_;
        buffer_ = other.buffer_;
        size_ = other.size_;
        other.buffer_ = nullptr;
    }
    return *this;
}

void GPUBuffer::write(const void* data, size_t size, size_t offset) {
    wgpuQueueWriteBuffer(queue_, buffer_, offset, data, size);
}

void GPUBuffer::release() {
    if (buffer_) { wgpuBufferRelease(buffer_); buffer_ = nullptr; }
}
