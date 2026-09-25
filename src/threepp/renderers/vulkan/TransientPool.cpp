#include "threepp/renderers/vulkan/TransientPool.hpp"

#include "threepp/renderers/vulkan/RenderGraph.hpp"

#include <algorithm>
#include <cstdio>

namespace threepp::vulkan {

    namespace {

        VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) {
            return a > 1 ? (v + a - 1) / a * a : v;
        }

        // Every live pool, so destroyImage2D can hand a pooled image back to
        // the pool that made it without knowing which one that was.
        std::mutex& registryMutex() {
            static std::mutex m;
            return m;
        }
        std::vector<TransientPool*>& registry() {
            static std::vector<TransientPool*> r;
            return r;
        }

    }// namespace

    // ── AliasPacker ──────────────────────────────────────────────────────────

    AliasPacker::Placement AliasPacker::place(uint32_t group, uint32_t slots, VkDeviceSize size,
                                              VkDeviceSize alignment, uint32_t memoryTypeBits) const {
        for (uint32_t b = 0; b < blocks_.size(); ++b) {
            const Block& blk = blocks_[b];
            if (!blk.alive || !blk.open || !(memoryTypeBits & (1u << blk.memoryType)) || size > blk.size) continue;
            VkDeviceSize at = 0;
            const auto it = blk.used.find(group);
            if (it != blk.used.end()) {
                for (const Range& r : it->second) {
                    if (!(r.slots & slots)) continue;// never in the same frame
                    if (alignUp(at, alignment) + size <= r.offset) break;
                    at = std::max(at, r.offset + r.size);
                }
            }
            at = alignUp(at, alignment);
            if (at + size <= blk.size) return {b, at};
        }
        return {};
    }

    uint32_t AliasPacker::addBlock(VkDeviceSize size, uint32_t memoryType) {
        Block b;
        b.size       = size;
        b.memoryType = memoryType;
        blocks_.push_back(std::move(b));
        return static_cast<uint32_t>(blocks_.size() - 1);
    }

    void AliasPacker::occupy(uint32_t block, uint32_t group, uint32_t slots, VkDeviceSize offset,
                             VkDeviceSize size) {
        auto& v = blocks_[block].used[group];
        const auto at = std::lower_bound(v.begin(), v.end(), offset,
                                         [](const Range& r, VkDeviceSize o) { return r.offset < o; });
        v.insert(at, {offset, size, slots});
    }

    void AliasPacker::release(uint32_t block, uint32_t group, uint32_t slots, VkDeviceSize offset) {
        auto& used = blocks_[block].used;
        const auto g = used.find(group);
        if (g == used.end()) return;
        auto& v = g->second;
        // (offset, slots) is unique in a group: two images at one offset have
        // disjoint slot masks.
        const auto r = std::find_if(v.begin(), v.end(),
                                    [&](const Range& x) { return x.offset == offset && x.slots == slots; });
        if (r != v.end()) v.erase(r);
        if (v.empty()) used.erase(g);
    }

    bool AliasPacker::blockEmpty(uint32_t block) const {
        return blocks_[block].used.empty();
    }

    void AliasPacker::removeBlock(uint32_t block) {
        blocks_[block].alive = false;
        blocks_[block].used.clear();
    }

    void AliasPacker::closeAll() {
        for (auto& b : blocks_) b.open = false;
    }

    VkDeviceSize AliasPacker::liveBytes() const {
        VkDeviceSize n = 0;
        for (const auto& b : blocks_)
            if (b.alive) n += b.size;
        return n;
    }

    // ── TransientPool ────────────────────────────────────────────────────────

    TransientPool::TransientPool(VmaAllocator allocator, VkDevice device, const char* name,
                                 VkDeviceSize blockSize)
        : allocator_(allocator), device_(device), name_(name), blockSize_(blockSize) {
        std::lock_guard lock(registryMutex());
        registry().push_back(this);
    }

    TransientPool::~TransientPool() {
        {
            std::lock_guard lock(registryMutex());
            auto& r = registry();
            r.erase(std::remove(r.begin(), r.end(), this), r.end());
        }
        if (!images_.empty()) {
            std::fprintf(stderr, "[TransientPool] '%s' destroyed with %zu images still bound\n", name_,
                         images_.size());
            for (const auto& [image, e] : images_) vkDestroyImage(device_, image, nullptr);
        }
        for (VmaAllocation a : blocks_)
            if (a) vmaFreeMemory(allocator_, a);
    }

    VkImage TransientPool::createImage(const VkImageCreateInfo& info, uint32_t group, uint32_t slots) {
        VkImage image = VK_NULL_HANDLE;
        if (vkCreateImage(device_, &info, nullptr, &image) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device_, image, &req);

        std::lock_guard lock(mtx_);
        auto at = packer_.place(group, slots, req.size, req.alignment, req.memoryTypeBits);
        if (at.block == AliasPacker::kNone) {
            VkMemoryRequirements blockReq = req;
            blockReq.size = std::max(req.size, blockSize_);
            VmaAllocationCreateInfo aci{};
            aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            VmaAllocation     alloc = VK_NULL_HANDLE;
            VmaAllocationInfo ai{};
            if (vmaAllocateMemory(allocator_, &blockReq, &aci, &alloc, &ai) != VK_SUCCESS) {
                vkDestroyImage(device_, image, nullptr);
                return VK_NULL_HANDLE;
            }
            const uint32_t b = packer_.addBlock(blockReq.size, ai.memoryType);
            blocks_.resize(b + 1, VK_NULL_HANDLE);
            blocks_[b] = alloc;
            at         = {b, 0};
        }
        if (vmaBindImageMemory2(allocator_, blocks_[at.block], at.offset, image, nullptr) != VK_SUCCESS) {
            vkDestroyImage(device_, image, nullptr);
            if (packer_.blockEmpty(at.block)) {
                vmaFreeMemory(allocator_, blocks_[at.block]);
                blocks_[at.block] = VK_NULL_HANDLE;
                packer_.removeBlock(at.block);
            }
            return VK_NULL_HANDLE;
        }
        packer_.occupy(at.block, group, slots, at.offset, req.size);
        images_[image] = {at.block, group, slots, at.offset, req.size};
        return image;
    }

    void TransientPool::setBlockSize(VkDeviceSize bytes) {
        std::lock_guard lock(mtx_);
        if (bytes == blockSize_) return;
        blockSize_ = bytes;
        packer_.closeAll();
    }

    VkDeviceSize TransientPool::packedBytes(const std::vector<VkImageCreateInfo>& infos) const {
        VkDeviceSize at = 0;
        for (const auto& info : infos) {
            VkDeviceImageMemoryRequirements q{};
            q.sType       = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS;
            q.pCreateInfo = &info;
            VkMemoryRequirements2 r{};
            r.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
            vkGetDeviceImageMemoryRequirements(device_, &q, &r);
            at = alignUp(at, r.memoryRequirements.alignment) + r.memoryRequirements.size;
        }
        return at;
    }

    bool TransientPool::releaseImpl(VkImage image) {
        std::lock_guard lock(mtx_);
        const auto it = images_.find(image);
        if (it == images_.end()) return false;
        const Entry e = it->second;
        images_.erase(it);
        vkDestroyImage(device_, image, nullptr);
        packer_.release(e.block, e.group, e.slots, e.offset);
        if (packer_.blockEmpty(e.block)) {
            vmaFreeMemory(allocator_, blocks_[e.block]);
            blocks_[e.block] = VK_NULL_HANDLE;
            packer_.removeBlock(e.block);
        }
        return true;
    }

    bool TransientPool::release(VkImage image) {
        if (image == VK_NULL_HANDLE) return false;
        std::lock_guard lock(registryMutex());
        for (TransientPool* p : registry())
            if (p->releaseImpl(image)) return true;
        return false;
    }

    bool TransientPool::pooled(VkImage image) {
        if (image == VK_NULL_HANDLE) return false;
        std::lock_guard lock(registryMutex());
        for (TransientPool* p : registry()) {
            std::lock_guard l(p->mtx_);
            if (p->images_.count(image)) return true;
        }
        return false;
    }

    void TransientPool::declare(rg::RenderGraph& graph) const {
        std::lock_guard lock(mtx_);
        for (const auto& [image, e] : images_) {
            graph.setMemoryRange(image, blocks_[e.block], e.offset, e.size);
        }
    }

    VkDeviceSize TransientPool::reservedBytes() const {
        std::lock_guard lock(mtx_);
        return packer_.liveBytes();
    }

    VkDeviceSize TransientPool::requestedBytes() const {
        std::lock_guard lock(mtx_);
        VkDeviceSize n = 0;
        for (const auto& [image, e] : images_) n += e.size;
        return n;
    }

    VkResult createImageMaybePooled(VmaAllocator allocator, TransientPool* pool, uint32_t group, uint32_t slots,
                                    const VkImageCreateInfo& info, VkImage* image, VmaAllocation* alloc) {
        *alloc = VK_NULL_HANDLE;
        if (pool) {
            *image = pool->createImage(info, group, slots);
            if (*image != VK_NULL_HANDLE) return VK_SUCCESS;
        }
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        return vmaCreateImage(allocator, &info, &aci, image, alloc, nullptr);
    }

}// namespace threepp::vulkan
