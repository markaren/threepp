// VmaImpl — the single translation unit that compiles Vulkan Memory Allocator.
//
// VMA is a ~20k-line single-header library: whichever TU defines
// VMA_IMPLEMENTATION pays its full compile cost. That define used to sit at the
// top of VulkanRenderer.cpp, the file holding every public setter, so editing a
// one-line getter rebuilt all of VMA with it. Nothing else belongs in here —
// keep it at exactly these two lines so the cost stays where it can't be
// triggered by ordinary edits.
//
// No VMA_* configuration macros: the rest of the tree includes
// <vk_mem_alloc.h> (via VulkanResources.hpp / VulkanContext.hpp) with the
// defaults, and any config macro would have to be defined identically in every
// including TU to avoid an ODR mismatch.

#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
