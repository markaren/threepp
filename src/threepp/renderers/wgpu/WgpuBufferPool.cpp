#include "WgpuBufferPool.hpp"

#include <algorithm>

namespace threepp::wgpu {

WgpuBufferPool::WgpuBufferPool(WGPUDevice device, WGPUQueue queue)
    : device_(device), queue_(queue) {}

WgpuBufferPool::~WgpuBufferPool() {
    dispose();
}

size_t WgpuBufferPool::roundToSizeClass(size_t size) {
    // Round up to nearest multiple of 256 (WebGPU uniform buffer alignment).
    // For larger buffers, round to nearest power of 2 to reduce fragmentation.
    constexpr size_t kMinAlign = 256;
    if (size <= kMinAlign) return kMinAlign;
    if (size <= 4096) return (size + kMinAlign - 1) & ~(kMinAlign - 1);
    // Power-of-2 for larger allocations
    size_t v = size - 1;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16; v |= v >> 32;
    return v + 1;
}

void WgpuBufferPool::beginFrame() {
    for (auto& entry : inUse_) {
        PoolKey key{entry.sizeClass, entry.usage};
        freePools_[key].push_back(entry.buffer);
    }
    inUse_.clear();

    // Trim each size class to at most kMaxFreePerClass buffers to prevent
    // the pool from accumulating memory after frames with transient peaks.
    constexpr size_t kMaxFreePerClass = 4;
    for (auto& [key, buffers] : freePools_) {
        while (buffers.size() > kMaxFreePerClass) {
            wgpuBufferRelease(buffers.back());
            buffers.pop_back();
        }
    }
}

WGPUBuffer WgpuBufferPool::acquire(size_t size, WgpuBufferUsageFlags usage, const void* data) {
    size_t sc = roundToSizeClass(size);
    PoolKey key{sc, usage};

    WGPUBuffer buf = nullptr;
    auto it = freePools_.find(key);
    if (it != freePools_.end() && !it->second.empty()) {
        buf = it->second.back();
        it->second.pop_back();
    } else {
        WGPUBufferDescriptor desc{};
        desc.size = sc;
        desc.usage = usage | WGPUBufferUsage_CopyDst;
        buf = wgpuDeviceCreateBuffer(device_, &desc);
    }

    if (data && buf) {
        wgpuQueueWriteBuffer(queue_, buf, 0, data, size);
    }

    InUseEntry entry;
    entry.buffer = buf;
    entry.sizeClass = sc;
    entry.usage = usage;
    inUse_.push_back(entry);
    return buf;
}

void WgpuBufferPool::dispose() {
    for (auto& entry : inUse_) {
        wgpuBufferRelease(entry.buffer);
    }
    inUse_.clear();

    for (auto& [key, buffers] : freePools_) {
        for (auto buf : buffers) {
            wgpuBufferRelease(buf);
        }
    }
    freePools_.clear();
}

} // namespace threepp::wgpu
