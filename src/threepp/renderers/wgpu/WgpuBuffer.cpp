
#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "WgpuCompat.hpp"

#include <webgpu/webgpu.h>

using namespace threepp;

WgpuBuffer::WgpuBuffer(WgpuRenderer& renderer, size_t size)
    : device_(static_cast<WGPUDevice>(renderer.nativeDevice())),
      queue_(static_cast<WGPUQueue>(renderer.nativeQueue())),
      size_(size) {

    WGPUBufferDescriptor desc{};
    desc.label = WGPU_LABEL("gpu_buffer");
    desc.size = size;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    buffer_ = wgpuDeviceCreateBuffer(device_, &desc);
}

WgpuBuffer::~WgpuBuffer() {
    release();
}

WgpuBuffer::WgpuBuffer(WgpuBuffer&& other) noexcept
    : device_(other.device_), queue_(other.queue_),
      buffer_(other.buffer_), size_(other.size_) {
    other.buffer_ = nullptr;
}

WgpuBuffer& WgpuBuffer::operator=(WgpuBuffer&& other) noexcept {
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

void WgpuBuffer::write(const void* data, size_t size, size_t offset) {
    wgpuQueueWriteBuffer(queue_, buffer_, offset, data, size);
}

void WgpuBuffer::release() {
    if (buffer_) { wgpuBufferRelease(buffer_); buffer_ = nullptr; }
}
