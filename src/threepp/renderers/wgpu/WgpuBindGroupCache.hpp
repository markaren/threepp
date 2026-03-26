// Per-object bind group cache for the WebGPU renderer.
// Stores one WGPUBindGroup per (object, material) pair.  The cached bind
// group is reused as long as the set of GPU resource handles is identical to
// the previous frame; otherwise it is recreated and the old one released.
// Stale entries (object left the scene) are evicted after 2 frames.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <webgpu/webgpu.h>

namespace threepp::wgpu {

class WgpuBindGroupCache {
public:
    explicit WgpuBindGroupCache(WGPUDevice device);
    ~WgpuBindGroupCache();

    // Get or create the bind group for (objectPtr, materialPtr).
    // - If the content of the entries is unchanged since last call: returns cached handle.
    // - Otherwise: releases the old handle and creates a new one.
    // The returned handle is owned by the cache — do NOT call wgpuBindGroupRelease on it.
    WGPUBindGroup get(const void* objectPtr, const void* materialPtr,
                      WGPUBindGroupLayout layout,
                      const WGPUBindGroupEntry* entries, uint32_t count,
                      WGPUStringView label = {nullptr, 0});

    // Advance the frame counter and evict entries unused for more than 2 frames.
    // Call once per frame alongside WgpuBufferPool::beginFrame().
    void beginFrame();

    // Release all cached bind groups (call before device destruction).
    void dispose();

private:
    WGPUDevice device_;
    uint32_t frame_ = 0;

    struct Entry {
        WGPUBindGroup bg = nullptr;
        size_t contentHash = 0;
        uint32_t lastUsed = 0;
    };

    struct Key {
        const void* object;
        const void* material;
        bool operator==(const Key& o) const { return object == o.object && material == o.material; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h = reinterpret_cast<uintptr_t>(k.object);
            h ^= reinterpret_cast<uintptr_t>(k.material) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<Key, Entry, KeyHash> cache_;

    // FNV-1a hash over all handle/offset/size fields of the bind group entries.
    static size_t computeHash(WGPUBindGroupLayout layout,
                               const WGPUBindGroupEntry* entries, uint32_t count);
};

} // namespace threepp::wgpu
