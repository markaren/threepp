// Process-wide exit policy for the Vulkan device.
//
// On Windows/NVIDIA, vkDestroyDevice blocks for ~20 s whenever the driver
// still holds shader binaries it has not yet written to its own on-disk
// cache: a run that presented enough frames for the driver's background
// re-optimisation, or one that compiled something new. The driver writes on a
// timer of roughly 25 s; a destroy before that waits it out and then gives
// the write up, so the wait buys nothing. A process that is exiting anyway
// can leave the device to the OS instead (see VulkanContext).

#ifndef THREEPP_VULKAN_EXIT_POLICY_HPP
#define THREEPP_VULKAN_EXIT_POLICY_HPP

namespace threepp::vulkan {

    /// Declare that the process is on its way out: every VulkanContext
    /// destroyed from now on leaves its device, surface and instance for the
    /// OS to reclaim instead of calling vkDestroyDevice. The Python binding
    /// sets this at interpreter finalization. A C++ canvas asks for the same
    /// by default (Canvas::Parameters::fastExit, false to opt out), so this
    /// switch is for the odd app that has to tear down a renderer created on
    /// an opted-out canvas while exiting.
    void setProcessExiting(bool exiting);

    [[nodiscard]] bool processExiting();

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_EXIT_POLICY_HPP
