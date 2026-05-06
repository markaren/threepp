#ifndef THREEPP_WGPUPATHTRACERSHADERS_HPP
#define THREEPP_WGPUPATHTRACERSHADERS_HPP

// Public WGSL fragment declarations and shader builders for the WGPU path
// tracer. Source files live under shaders/ as .wgsl and are embedded as C++
// raw string literals at build time by cmake/EmbedWgslPathTracer.cmake.

#include <string>

namespace threepp::wgpu_pt {

    // Single-fragment shaders consumed directly by WgpuPathTracer.cpp.
    extern const char* const svgfAtrousWGSL;
    extern const char* const depthFillWGSL;
    extern const char* const displayWGSL;
    extern const char* const upscaleWGSL;

    /// Concatenate the RT shader fragments and patch the ENV_CDF_FLAG marker.
    std::string buildRtShader(bool hasEnvCdf);

    /// Concatenate shared defs + vertex-transform compute shader.
    std::string buildVtShader();

    /// Concatenate shared defs + BVH refit compute shader.
    std::string buildRefitShader();

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERSHADERS_HPP
