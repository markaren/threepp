
#ifndef THREEPP_VULKAN_LOG_HPP
#define THREEPP_VULKAN_LOG_HPP

// The Vulkan renderer's log categories, all in one place so the list in
// doc/environment_variables.md has one source. THREEPP_LOG=<name>:<level>
// sets a threshold; "vulkan" covers them all.

#include "threepp/utils/Log.hpp"

namespace threepp::vulkan {

    // Validation layer messages (errors and warnings), strict-validation verdicts.
    inline constinit log::Category kLogValidation{"vulkan.validation"};
    // Device, surface and swapchain: the GPU, ray tracing, extensions, present mode.
    inline constinit log::Category kLogDevice{"vulkan.device"};
    // Pipeline cache load and save.
    inline constinit log::Category kLogPipelineCache{"vulkan.pipeline_cache"};
    // THREEPP_VULKAN_PIPELINE_STATS output: pipeline executable statistics.
    inline constinit log::Category kLogPipelineStats{"vulkan.pipeline_stats"};
    // Time of each renderer startup step (Debug).
    inline constinit log::Category kLogStartup{"vulkan.startup"};
    // Time of each renderer and context teardown step (Debug).
    inline constinit log::Category kLogExit{"vulkan.exit"};
    // Calls the renderer ignores or cannot honour.
    inline constinit log::Category kLogRenderer{"vulkan.renderer"};
    // Frame loop: swapchain recreation, secondary views, acquires (Trace).
    inline constinit log::Category kLogFrame{"vulkan.frame"};
    // HUD and 2D overlay passes.
    inline constinit log::Category kLogOverlay{"vulkan.overlay"};
    // DLSS, FSR and NGX.
    inline constinit log::Category kLogUpscale{"vulkan.upscale"};
    // Deferred shade variants and their compile times.
    inline constinit log::Category kLogShade{"vulkan.shade"};
    // Render graph diagnostics and dumps (Debug).
    inline constinit log::Category kLogRenderGraph{"vulkan.rendergraph"};
    // GPU memory: THREEPP_VK_MEMDUMP reports, transient pools, memory placement.
    inline constinit log::Category kLogMemory{"vulkan.memory"};
    // Geometry, BLAS and TLAS builds, auto-LOD, skinning.
    inline constinit log::Category kLogGeometry{"vulkan.geometry"};
    // displacementMap refinement.
    inline constinit log::Category kLogDisplacement{"vulkan.displacement"};
    // Vertex and frame interop (exported device memory).
    inline constinit log::Category kLogInterop{"vulkan.interop"};
    // Textures and environment maps.
    inline constinit log::Category kLogTexture{"vulkan.texture"};
    // Scene build: slot edits, walk checks, morph targets.
    inline constinit log::Category kLogScene{"vulkan.scene"};
    // THREEPP_VK_DRAW_STATS output.
    inline constinit log::Category kLogDrawStats{"vulkan.draw_stats"};
    // THREEPP_CPU_PHASE_PROFILE output.
    inline constinit log::Category kLogCpuProf{"vulkan.cpuprof"};
    // Probe GI grid fits (Debug).
    inline constinit log::Category kLogProbe{"vulkan.probe"};
    // Shadow mode and the static shadow caches (Debug, Trace per frame).
    inline constinit log::Category kLogShadow{"vulkan.shadow"};
    // Gaussian splats.
    inline constinit log::Category kLogSplat{"vulkan.splat"};
    // ParticleField.
    inline constinit log::Category kLogParticles{"vulkan.particles"};
    // Ocean spectrum and FFT.
    inline constinit log::Category kLogOcean{"vulkan.ocean"};
    // Path tracer.
    inline constinit log::Category kLogPathTracer{"vulkan.pathtracer"};

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_LOG_HPP
