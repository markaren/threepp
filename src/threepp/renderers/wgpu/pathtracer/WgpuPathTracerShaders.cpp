#include "WgpuPathTracerShaders.hpp"

#include <string>

namespace threepp::wgpu_pt {

    // Internal fragments — defined in the auto-generated WgpuPathTracerShadersGen.cpp
    // (see cmake/EmbedWgslPathTracer.cmake). Declared here so the build
    // functions below can concatenate them.
    extern const char* const csSharedDefsWGSL;
    extern const char* const csCommonWGSL;
    extern const char* const csPathTraceWGSL;
    extern const char* const csRunBouncesWGSL1;
    extern const char* const csRunBouncesWGSL2;
    extern const char* const csPrimaryShadeWGSL1;
    extern const char* const csPrimaryShadeWGSL2;
    extern const char* const csPrimaryShadeWGSL3;
    extern const char* const csPathTraceWGSL2;
    extern const char* const csPathTraceWGSL3;
    extern const char* const csBounce1WGSL;
    extern const char* const csAccumWGSL1;
    extern const char* const vtWGSL_;
    extern const char* const refitWGSL_;

    std::string buildRtShader(bool hasEnvCdf) {
        std::string src = std::string(csSharedDefsWGSL) + "\n" +
                          csCommonWGSL + "\n" +
                          csPathTraceWGSL + "\n" +
                          csRunBouncesWGSL1 + "\n" +
                          csRunBouncesWGSL2 + "\n" +
                          csPrimaryShadeWGSL1 + "\n" +
                          csPrimaryShadeWGSL2 + "\n" +
                          csPrimaryShadeWGSL3 + "\n" +
                          csPathTraceWGSL2 + "\n" +
                          csPathTraceWGSL3 + "\n" +
                          csBounce1WGSL + "\n" +
                          csAccumWGSL1;
        const std::string marker = "/*ENV_CDF_FLAG*/false";
        auto pos = src.find(marker);
        if (pos != std::string::npos) {
            src.replace(pos, marker.size(), hasEnvCdf ? "true" : "false");
        }
        return src;
    }

    std::string buildVtShader() {
        return std::string(csSharedDefsWGSL) + "\n" + vtWGSL_;
    }

    std::string buildRefitShader() {
        return std::string(csSharedDefsWGSL) + "\n" + refitWGSL_;
    }

}// namespace threepp::wgpu_pt
