#include "threepp/renderers/vulkan/DescriptorShadow.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace threepp::vulkan {

    namespace {

        struct ViewInfo {
            VkImage            image = VK_NULL_HANDLE;
            VkImageAspectFlags aspect = 0;
            uint32_t           baseMip = 0, mipCount = 1, imageMips = 1;
            uint64_t           serial = 0;// see imageViewSerial
        };

        struct Element {
            VkImageView   view = VK_NULL_HANDLE;
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkBuffer      buffer = VK_NULL_HANDLE;
            VkAccelerationStructureKHR as = VK_NULL_HANDLE;
        };
        struct Binding {
            VkDescriptorType     type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
            std::vector<Element> elements;
        };
        using SetShadow = std::unordered_map<uint32_t, Binding>;

        std::mutex& mtx() {
            static std::mutex m;
            return m;
        }
        std::unordered_map<VkImageView, ViewInfo>& views() {
            static std::unordered_map<VkImageView, ViewInfo> v;
            return v;
        }
        struct AccelInfo {
            VkBuffer    storage = VK_NULL_HANDLE;
            const char* contents = nullptr;
            uint64_t    serial = 0;// see accelerationStructureSerial
        };
        // One counter for views and acceleration structures: a serial is never
        // handed out twice, so 0 can stand for "none".
        uint64_t nextSerial = 0;// guarded by mtx()
        std::unordered_map<VkAccelerationStructureKHR, AccelInfo>& accels() {
            static std::unordered_map<VkAccelerationStructureKHR, AccelInfo> a;
            return a;
        }
        std::unordered_map<VkDescriptorSet, SetShadow>& sets() {
            static std::unordered_map<VkDescriptorSet, SetShadow> s;
            return s;
        }
        // One warning per offending handle, not per frame.
        std::unordered_set<uint64_t>& warned() {
            static std::unordered_set<uint64_t> w;
            return w;
        }

        // "set<s>.b<b>" for the graph's dump, built once: the graph keeps names
        // by pointer, and building strings per frame was the pass's CPU cost.
        const char* bindingName(uint32_t set, uint32_t binding) {
            static const auto table = [] {
                std::vector<std::string> t(4 * 128);
                for (uint32_t s = 0; s < 4; ++s)
                    for (uint32_t b = 0; b < 128; ++b)
                        t[s * 128 + b] = "set" + std::to_string(s) + ".b" + std::to_string(b);
                return t;
            }();
            return (set < 4 && binding < 128) ? table[set * 128 + binding].c_str() : "set?.b?";
        }

        bool isImageType(VkDescriptorType t) {
            return t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
                   t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        }
        bool isBufferType(VkDescriptorType t) {
            return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
                   t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        }

    }// namespace

    void registerImageView(VkImageView view, VkImage image, VkImageAspectFlags aspect,
                           uint32_t baseMip, uint32_t mipCount, uint32_t imageMipLevels) {
        if (view == VK_NULL_HANDLE) return;
        std::lock_guard lock(mtx());
        views()[view] = {image, aspect, baseMip, mipCount == 0 ? 1u : mipCount,
                         imageMipLevels == 0 ? 1u : imageMipLevels, ++nextSerial};
    }

    uint64_t imageViewSerial(VkImageView view) {
        if (view == VK_NULL_HANDLE) return 0;
        std::lock_guard lock(mtx());
        const auto it = views().find(view);
        // Unregistered: a fresh number, so a cache keyed on it misses (and
        // rewrites) rather than matching a view it has never seen.
        return it == views().end() ? ++nextSerial : it->second.serial;
    }

    VkResult createImageView(VkDevice device, const VkImageViewCreateInfo* info,
                             const VkAllocationCallbacks* allocator, VkImageView* view) {
        const VkResult r = vkCreateImageView(device, info, allocator, view);
        if (r == VK_SUCCESS) {
            const auto& sr = info->subresourceRange;
            const uint32_t count = sr.levelCount == VK_REMAINING_MIP_LEVELS ? 1u : sr.levelCount;
            registerImageView(*view, info->image, sr.aspectMask, sr.baseMipLevel, count,
                              sr.baseMipLevel + count);
        }
        return r;
    }

    void registerAccelerationStructure(VkAccelerationStructureKHR as, VkBuffer storage,
                                       const char* contents) {
        if (as == VK_NULL_HANDLE) return;
        std::lock_guard lock(mtx());
        accels()[as] = {storage, contents, ++nextSerial};
    }

    uint64_t accelerationStructureSerial(VkAccelerationStructureKHR as) {
        if (as == VK_NULL_HANDLE) return 0;
        std::lock_guard lock(mtx());
        const auto it = accels().find(as);
        return it == accels().end() ? ++nextSerial : it->second.serial;
    }

    VkBuffer accelerationStructureBuffer(VkAccelerationStructureKHR as) {
        std::lock_guard lock(mtx());
        const auto it = accels().find(as);
        return it == accels().end() ? VK_NULL_HANDLE : it->second.storage;
    }

    void unregisterImageView(VkImageView view) {
        if (view == VK_NULL_HANDLE) return;
        std::lock_guard lock(mtx());
        views().erase(view);
    }

    void updateDescriptorSets(VkDevice device, uint32_t writeCount, const VkWriteDescriptorSet* writes,
                              uint32_t copyCount, const VkCopyDescriptorSet* copies) {
        {
            std::lock_guard lock(mtx());
            for (uint32_t i = 0; i < writeCount; ++i) {
                const auto& w = writes[i];
                auto& b = sets()[w.dstSet][w.dstBinding];
                b.type = w.descriptorType;
                if (b.elements.size() < w.dstArrayElement + w.descriptorCount) {
                    b.elements.resize(w.dstArrayElement + w.descriptorCount);
                }
                for (uint32_t e = 0; e < w.descriptorCount; ++e) {
                    Element& el = b.elements[w.dstArrayElement + e];
                    el = {};
                    if (isImageType(w.descriptorType) && w.pImageInfo) {
                        el.view   = w.pImageInfo[e].imageView;
                        el.layout = w.pImageInfo[e].imageLayout;
                    } else if (isBufferType(w.descriptorType) && w.pBufferInfo) {
                        el.buffer = w.pBufferInfo[e].buffer;
                    } else if (w.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
                        for (auto* n = static_cast<const VkBaseInStructure*>(w.pNext); n; n = n->pNext) {
                            if (n->sType == VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR) {
                                const auto* a = reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureKHR*>(n);
                                if (e < a->accelerationStructureCount) el.as = a->pAccelerationStructures[e];
                            }
                        }
                    }
                }
            }
            for (uint32_t i = 0; i < copyCount; ++i) {
                auto it = sets().find(copies[i].dstSet);
                if (it != sets().end()) it->second.erase(copies[i].dstBinding);
            }
        }
        vkUpdateDescriptorSets(device, writeCount, writes, copyCount, copies);
    }

    void forgetDescriptorSet(VkDescriptorSet set) {
        std::lock_guard lock(mtx());
        sets().erase(set);
    }

    void declareDescriptorSet(rg::RenderGraph& graph, rg::PassBuilder& pass,
                              VkDescriptorSet set, uint32_t setIndex,
                              const std::vector<SpirvBinding>& bindings,
                              VkPipelineStageFlags2 stages, uint32_t maxArrayElements,
                              std::initializer_list<VkImage> exclude) {
        std::lock_guard lock(mtx());
        const auto s = sets().find(set);
        if (s == sets().end()) {
            if (warned().insert(reinterpret_cast<uint64_t>(set)).second) {
                std::fprintf(stderr, "[RenderGraph] descriptor set %p has no shadow; its pass declares nothing\n",
                             static_cast<void*>(set));
            }
            return;
        }
        for (const auto& rb : bindings) {
            if (rb.set != setIndex) continue;
            const auto b = s->second.find(rb.binding);
            if (b == s->second.end()) continue;
            const Binding& bind = b->second;
            if (bind.elements.size() > maxArrayElements) continue;

            if (isBufferType(bind.type)) {
                rg::Access a;
                a.stages = stages;
                if (bind.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || bind.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
                    a.access = VK_ACCESS_2_UNIFORM_READ_BIT;
                } else {
                    if (rb.read) a.access |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                    if (rb.write) a.access |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    a.write = rb.write;
                }
                for (const auto& el : bind.elements) {
                    if (el.buffer == VK_NULL_HANDLE) continue;
                    pass.use(graph.importBuffer(bindingName(setIndex, rb.binding), el.buffer), a);
                }
                continue;
            }
            if (bind.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
                for (const auto& el : bind.elements) {
                    const auto a = accels().find(el.as);
                    if (el.as == VK_NULL_HANDLE || a == accels().end()) continue;
                    pass.use(graph.importBuffer(bindingName(setIndex, rb.binding), a->second.storage),
                             rg::Access{stages, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                                        VK_IMAGE_LAYOUT_UNDEFINED, false});
                    // Traversal reads the BLASes the instances reference, and
                    // hit shading reads their vertex data by device address.
                    if (a->second.contents) {
                        pass.use(graph.importMemory(a->second.contents),
                                 rg::Access{stages,
                                            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                            VK_IMAGE_LAYOUT_UNDEFINED, false});
                    }
                }
                continue;
            }
            if (!isImageType(bind.type)) continue;

            for (const auto& el : bind.elements) {
                if (el.view == VK_NULL_HANDLE) continue;
                const auto v = views().find(el.view);
                if (v == views().end()) {
                    if (warned().insert(reinterpret_cast<uint64_t>(el.view)).second) {
                        std::fprintf(stderr,
                                     "[RenderGraph] image view %p (set %u binding %u) is not registered; "
                                     "its accesses are not declared\n",
                                     static_cast<void*>(el.view), setIndex, rb.binding);
                    }
                    continue;
                }
                bool excluded = false;
                for (const VkImage x : exclude) excluded = excluded || x == v->second.image;
                if (excluded) continue;
                rg::Access a;
                a.stages = stages;
                a.layout = el.layout;
                if (bind.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                    if (rb.read) a.access |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                    if (rb.write) a.access |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    a.write = rb.write;
                } else if (bind.type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT) {
                    a.access = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;
                } else {
                    a.access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                }
                const auto h = graph.importImage(bindingName(setIndex, rb.binding), v->second.image,
                                                 v->second.aspect, v->second.imageMips, el.layout);
                pass.use(h, a, v->second.baseMip, v->second.mipCount);
            }
        }
    }

}// namespace threepp::vulkan
