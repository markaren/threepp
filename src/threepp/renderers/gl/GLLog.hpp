
#ifndef THREEPP_GL_LOG_HPP
#define THREEPP_GL_LOG_HPP

// The OpenGL renderer's log categories, all in one place so the list in
// doc/environment_variables.md has one source. THREEPP_LOG=<name>:<level>
// sets a threshold; "gl" covers them all.

#include "threepp/utils/Log.hpp"

namespace threepp::gl {

    // Shader compile and link logs, shader code generation.
    inline constinit log::Category kLogProgram{"gl.program"};
    // Textures, render targets and their framebuffers.
    inline constinit log::Category kLogTexture{"gl.texture"};
    // Uniform values the renderer cannot upload.
    inline constinit log::Category kLogUniform{"gl.uniform"};
    // Blending and other GL state.
    inline constinit log::Category kLogState{"gl.state"};
    // Shadow maps.
    inline constinit log::Category kLogShadow{"gl.shadow"};
    // Draw calls and framebuffer capture.
    inline constinit log::Category kLogRenderer{"gl.renderer"};
    // Post-processing passes.
    inline constinit log::Category kLogPostProcessing{"gl.postprocessing"};

}// namespace threepp::gl

#endif//THREEPP_GL_LOG_HPP
