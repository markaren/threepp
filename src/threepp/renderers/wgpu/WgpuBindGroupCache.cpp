#include "WgpuBindGroupCache.hpp"

namespace threepp::wgpu {

WgpuBindGroupCache::WgpuBindGroupCache(WGPUDevice device) : device_(device) {
    cache_.reserve(256);
    deferredRelease_.reserve(64);
}

WgpuBindGroupCache::~WgpuBindGroupCache() {
    dispose();
}

size_t WgpuBindGroupCache::computeHash(WGPUBindGroupLayout layout,
                                        const WGPUBindGroupEntry* entries, uint32_t count) {
    size_t h = 14695981039346656037ULL;
    auto mix = [&](uint64_t v) {
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        for (int i = 0; i < 8; ++i) {
            h ^= static_cast<size_t>(p[i]);
            h *= 1099511628211ULL;
        }
    };
    mix(reinterpret_cast<uintptr_t>(layout));
    mix(static_cast<uint64_t>(count));
    for (uint32_t i = 0; i < count; ++i) {
        const auto& e = entries[i];
        mix(static_cast<uint64_t>(e.binding));
        mix(reinterpret_cast<uintptr_t>(e.buffer));
        mix(static_cast<uint64_t>(e.offset));
        mix(static_cast<uint64_t>(e.size));
        mix(reinterpret_cast<uintptr_t>(e.textureView));
        mix(reinterpret_cast<uintptr_t>(e.sampler));
    }
    return h;
}

WGPUBindGroup WgpuBindGroupCache::get(const void* objectPtr, const void* materialPtr,
                                       WGPUBindGroupLayout layout,
                                       const WGPUBindGroupEntry* entries, uint32_t count,
                                       WGPUStringView label) {
    (void)objectPtr;
    (void)materialPtr;

    WGPUBindGroupDescriptor desc{};
    desc.label      = label;
    desc.layout     = layout;
    desc.entryCount = count;
    desc.entries    = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device_, &desc);

    if (bg) deferredRelease_.push_back(bg);
    return bg;
}

void WgpuBindGroupCache::beginFrame() {
    for (auto bg : deferredRelease_) {
        wgpuBindGroupRelease(bg);
    }
    deferredRelease_.clear();

    ++frame_;
    for (auto it = cache_.begin(); it != cache_.end(); ) {
        if (frame_ - it->second.lastUsed > 2) {
            wgpuBindGroupRelease(it->second.bg);
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void WgpuBindGroupCache::dispose() {
    for (auto bg : deferredRelease_) {
        wgpuBindGroupRelease(bg);
    }
    deferredRelease_.clear();

    for (auto& [key, entry] : cache_) {
        wgpuBindGroupRelease(entry.bg);
    }
    cache_.clear();
}

} // namespace threepp::wgpu
