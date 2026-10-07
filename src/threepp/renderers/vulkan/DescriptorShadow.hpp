// DescriptorShadow — what each descriptor binding points at, on the host.
//
// Vulkan has no query for "which image does this view / descriptor refer to",
// and a render-graph pass needs exactly that to turn "dispatch pipeline P with
// descriptor set S" into resource declarations. Two small registries answer
// it:
//   * image views: registerImageView() when a view is created, forget on
//     destruction (destroyImage2D does it for every Image2D);
//   * descriptor sets: updateDescriptorSets() records each write and then
//     forwards to vkUpdateDescriptorSets.
// declareDescriptorSet() then walks a pipeline's reflected bindings (see
// SpirvReflect) against a set's shadow and declares every image and buffer the
// dispatch can touch, with the access its shader qualifiers allow.
//
// Only sets written through updateDescriptorSets() have a shadow; a pass that
// declares an unshadowed set gets a one-time warning and no declarations.

#ifndef THREEPP_VULKAN_DESCRIPTOR_SHADOW_HPP
#define THREEPP_VULKAN_DESCRIPTOR_SHADOW_HPP

#include "threepp/renderers/vulkan/RenderGraph.hpp"
#include "threepp/renderers/vulkan/SpirvReflect.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace threepp::vulkan {

    void registerImageView(VkImageView view, VkImage image, VkImageAspectFlags aspect,
                           uint32_t baseMip, uint32_t mipCount, uint32_t imageMipLevels);
    void unregisterImageView(VkImageView view);

    // The buffer an acceleration structure lives in. A shader's ray query
    // reads that buffer, so an AS binding is declared as an ACCELERATION_
    // STRUCTURE_READ of it — the same buffer an AS build declares its write on.
    // `contents` (a TLAS): the graph memory resource (RenderGraph::
    // importMemory) holding what its instances reference — the BLASes and the
    // vertex data hit shading reads by device address. A ray query through
    // the AS is declared as a read of that too.
    void registerAccelerationStructure(VkAccelerationStructureKHR as, VkBuffer storage,
                                       const char* contents = nullptr);
    [[nodiscard]] VkBuffer accelerationStructureBuffer(VkAccelerationStructureKHR as);
    // imageViewSerial's counterpart (VulkanResources.hpp): a number for THIS
    // acceleration structure, assigned by registerAccelerationStructure and
    // never reused. 0 for null; a new number per call for an unregistered one.
    [[nodiscard]] uint64_t accelerationStructureSerial(VkAccelerationStructureKHR as);

    // updateDescriptorSets and createImageView (the recording wrappers) are
    // declared in VulkanResources.hpp, next to the other resource helpers.

    // Forget a set's shadow (the set was freed or its pool reset).
    void forgetDescriptorSet(VkDescriptorSet set);

    // Declare, on `pass`, every image and buffer that the shader's bindings in
    // descriptor set `setIndex` refer to through `set`. Array bindings larger
    // than maxArrayElements (bindless material textures, written only outside
    // the frame) are skipped. Images are imported in the layout their
    // descriptor names, unless the caller imported them first. Images in
    // `exclude` are skipped too: a pass that changes an image's layout inside
    // itself (and restores it) declares that image itself, in its resting
    // layout, rather than in the layout its descriptor names.
    void declareDescriptorSet(rg::RenderGraph& graph, rg::PassBuilder& pass,
                              VkDescriptorSet set, uint32_t setIndex,
                              const std::vector<SpirvBinding>& bindings,
                              VkPipelineStageFlags2 stages,
                              uint32_t maxArrayElements = 16,
                              std::initializer_list<VkImage> exclude = {});

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DESCRIPTOR_SHADOW_HPP
