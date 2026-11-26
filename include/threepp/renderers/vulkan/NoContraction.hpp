// Process-wide shader precision option of the Vulkan backend.
//
// A driver may fuse or reorder a shader's floating-point arithmetic (a * b + c
// into one fma, for instance) and is free to decide that per pipeline compile.
// Two processes running the same binary on the same inputs can then render
// frames that differ by an ulp or so where the driver chose differently, which
// breaks cross-process replay of the rendered frame. With this option on,
// every float arithmetic result (add, subtract, multiply, divide, mod, negate,
// dot and the matrix/vector products) in every shader module the backend
// creates is decorated NoContraction before the module reaches the driver --
// what GLSL `precise` emits -- so it is evaluated as written.
//
// Off by default. On, it changes rendered values at the ulp level (a different
// reference, not the old one made stable) and costs 0-2 % frame time on light
// scenes and up to about 6 % GPU time on heavy ones. Modules built inside
// third-party libraries (FSR, NGX) are not covered. The ocean wake field's own
// modules (its step, its ripples and its shore: WakeFieldPipeline) are
// decorated whether or not it is on, because they carry their values from one
// step to the next.
//
// Turned on by THREEPP_VK_NO_CONTRACTION=1 or by setNoContraction(true). The
// value is fixed when the first shader module of the process is created
// (during the first renderer's construction) and holds for the rest of it.

#ifndef THREEPP_VULKAN_NO_CONTRACTION_HPP
#define THREEPP_VULKAN_NO_CONTRACTION_HPP

namespace threepp::vulkan {

    /// Choose the option in code, before the first renderer is created. Takes
    /// precedence over THREEPP_VK_NO_CONTRACTION. Once a shader module has
    /// been created the value is fixed: a call asking for the other value is
    /// ignored with a warning and returns false.
    bool setNoContraction(bool enabled);

    /// Whether shader float arithmetic is (or, before the first shader module,
    /// will be) decorated NoContraction. A harness records this with its result.
    [[nodiscard]] bool noContraction();

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_NO_CONTRACTION_HPP
