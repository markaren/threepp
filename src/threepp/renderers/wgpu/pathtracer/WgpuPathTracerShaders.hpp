#ifndef THREEPP_WGPUPATHTRACERSHADERS_HPP
#define THREEPP_WGPUPATHTRACERSHADERS_HPP

// Private header — declarations of the embedded WGSL fragments and shader
// builders used by the path tracer. Fragments are split across several .cpp
// files (Display, Denoise, VtRefit, Rt) grouped by pipeline. Symbols are
// declared here only after the extraction step that owns them lands; until
// then they remain inline in WgpuPathTracer.cpp inside an anonymous
// namespace.

#include <string>

namespace threepp::wgpu_pt {

    // -- RT (path-tracer) fragments and builder --------------------------------
    extern const char* const csCommonWGSL;
    extern const char* const csPathTraceWGSL;
    extern const char* const csPathTraceWGSL2;
    extern const char* const csPathTraceWGSL2b;
    extern const char* const csPathTraceWGSL3;
    /// Concatenate the RT shader fragments and patch the ENV_CDF_FLAG marker.
    std::string buildRtShader(bool hasEnvCdf);

    // -- Vertex-transform / BVH-refit fragments and builders -------------------
    // csSharedDefsWGSL is also consumed by the RT shader builder.
    extern const char* const csSharedDefsWGSL;
    extern const char* const vtWGSL_;
    extern const char* const refitWGSL_;
    extern const char* const blasRefitWGSL_;
    std::string buildVtShader();
    std::string buildRefitShader();
    std::string buildBlasRefitShader();

    // -- Denoiser pipeline fragments -------------------------------------------
    extern const char* const svgfAtrousWGSL;

    // -- Display / depth-fill / temporal-upscale pipeline fragments ------------
    extern const char* const depthFillWGSL;
    extern const char* const displayWGSL;
    extern const char* const upscaleWGSL;

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERSHADERS_HPP
