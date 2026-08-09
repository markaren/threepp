// Validation-layer report — the counters behind the Vulkan CI gate.
//
// VulkanContext installs a VK_EXT_debug_utils messenger whenever
// VK_LAYER_KHRONOS_validation is enabled (debug builds, or any build with
// THREEPP_VULKAN_VALIDATION set). Every message it delivers is printed, as
// before, and now also counted here — which is what lets a test assert that a
// frame sequence produced no spec violations.
//
// Why that assertion is worth having: it is the only Vulkan claim that holds on
// ANY conformant driver. The pixel goldens cannot be a cross-hardware gate —
// they are references captured on one GPU — so CI has never run the Vulkan
// backend at all, only compiled it. "Zero validation errors" is hardware-
// independent, so a software rasteriser (lavapipe) can gate it, and the class of
// defect it catches is precisely the one that has recurred here: descriptor
// updates issued while a frame is still in flight.
//
// Deliberately free of <vulkan/vulkan.h> and <vk_mem_alloc.h>. Both are PRIVATE
// to the threepp target, so a public header that pulled them in could not be
// included by a consumer — or by a test — at all. That is also why the counts
// are plain integers rather than the messenger's own types.
//
// The counts are process-wide, not per-context. The layer reports instance-level
// violations before any context exists and object-lifetime ones after one is
// gone, and both belong in the total.

#ifndef THREEPP_VULKAN_VALIDATION_REPORT_HPP
#define THREEPP_VULKAN_VALIDATION_REPORT_HPP

#include <cstdint>

namespace threepp::vulkan {

    // Messages counted since process start.
    //
    // Errors are spec violations — ERROR severity AND the VALIDATION message
    // type — which is what a gate should fail on. Everything else the messenger
    // hears lands in the warning count and is deliberately NOT fatal: that
    // bucket holds performance advice (driver-specific) and GENERAL-type loader
    // diagnostics (machine-specific — a stale third-party layer manifest
    // arrives as an ERROR-severity GENERAL message), and folding either into
    // the error count would make the verdict depend on where the gate ran
    // rather than on the code.
    [[nodiscard]] std::uint32_t validationErrorCount();
    [[nodiscard]] std::uint32_t validationWarningCount();

    // True once a VulkanContext has actually installed the debug messenger —
    // i.e. the layer was both requested AND found. A test asserting "no
    // validation errors" must check this as well, or it passes vacuously
    // wherever the layer is not installed, which is the normal state of a
    // release build and would make the gate worthless exactly where it is
    // cheapest to leave enabled.
    [[nodiscard]] bool validationActive();

    // No reset: a caller wanting per-phase deltas subtracts two readings, and a
    // reset would let a gate discard the messages emitted while the instance and
    // device were being created — which are the ones no later phase can re-report.

    // Strict mode (THREEPP_VULKAN_STRICT_VALIDATION=1) terminates the process
    // with this code at normal exit if any error was counted, so that any
    // Vulkan-backed executable becomes a validation gate without being edited.
    // Distinct from 42, which every Vulkan test uses for "no device — skip":
    // a skip and a spec violation must never arrive as the same verdict.
    constexpr int kStrictValidationExitCode = 67;

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_VALIDATION_REPORT_HPP
