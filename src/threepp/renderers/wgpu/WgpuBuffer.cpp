
#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"

#include <webgpu/webgpu.h>

using namespace threepp;

WgpuBuffer::WgpuBuffer(WgpuRenderer& renderer, size_t size, Usage usage)
    : device_(static_cast<WGPUDevice>(renderer.nativeDevice())),
      queue_(static_cast<WGPUQueue>(renderer.nativeQueue())),
      size_(size) {

    WGPUBufferDescriptor desc{};
    desc.label = WGPUStringView{"gpu_buffer", WGPU_STRLEN} ;
    desc.size = size;
    WGPUBufferUsage baseUsage = WGPUBufferUsage_Uniform;
    switch (usage) {
        case Usage::Storage: baseUsage = WGPUBufferUsage_Storage; break;
        case Usage::StorageReadback: baseUsage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc; break;
        case Usage::Vertex:  baseUsage = WGPUBufferUsage_Vertex;  break;
        case Usage::Uniform: baseUsage = WGPUBufferUsage_Uniform; break;
    }
    desc.usage = baseUsage | WGPUBufferUsage_CopyDst;
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
