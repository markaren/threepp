#pragma once

#include <webgpu/webgpu.h>

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace threepp::wgpu {

// Frame-scoped GPU buffer pool to avoid per-draw buffer creation/destruction.
// Call beginFrame() at the start of each frame to recycle all buffers.
// Acquired buffers remain valid until the next beginFrame() call.
class WgpuBufferPool {
public:
    WgpuBufferPool(WGPUDevice device, WGPUQueue queue);

    // Recycle all in-use buffers back to the free pools.
    void beginFrame();

    // Acquire a buffer of at least `size` bytes with the given usage flags.
    // Writes `data` into the buffer if non-null.
    WGPUBuffer acquire(size_t size, WGPUBufferUsage usage, const void* data = nullptr);

    // Release all GPU buffers.
    void dispose();

    ~WgpuBufferPool();

private:
    WGPUDevice device_;
    WGPUQueue queue_;

    struct PoolKey {
        size_t sizeClass;
        WGPUBufferUsage usage;
        bool operator==(const PoolKey& o) const {
            return sizeClass == o.sizeClass && usage == o.usage;
        }
    };
    struct PoolKeyHash {
        size_t operator()(const PoolKey& k) const {
            return std::hash<size_t>()(k.sizeClass) ^ (std::hash<uint64_t>()(k.usage) << 16);
        }
    };

    struct InUseEntry {
        WGPUBuffer buffer;
        size_t sizeClass;
        WGPUBufferUsage usage;
    };

    std::unordered_map<PoolKey, std::vector<WGPUBuffer>, PoolKeyHash> freePools_;
    std::vector<InUseEntry> inUse_;

    static size_t roundToSizeClass(size_t size);
};

} // namespace threepp::wgpu
