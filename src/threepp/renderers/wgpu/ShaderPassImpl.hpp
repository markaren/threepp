#ifndef THREEPP_SHADERPASSIMPL_HPP
#define THREEPP_SHADERPASSIMPL_HPP

#include "threepp/renderers/wgpu/ShaderPass.hpp"
#include "WgpuCompat.hpp"

#include <webgpu/webgpu.h>

namespace threepp {

    struct ShaderPass::Impl {
        std::string wgslSource;
        WGPUShaderModule shaderModule = nullptr;
        WGPURenderPipeline pipeline = nullptr;

        ~Impl() {
            if (pipeline) wgpuRenderPipelineRelease(pipeline);
            if (shaderModule) wgpuShaderModuleRelease(shaderModule);
        }

        void ensurePipeline(WGPUDevice device, WGPUPipelineLayout layout, WGPUTextureFormat format) {
            if (pipeline) return;

            // Create shader module
            WgpuShaderSourceWGSL wgslSrc{};
            wgslSrc.chain.sType = WGPU_STYPE_SHADER_SOURCE_WGSL;
            wgslSrc.chain.next = nullptr;
            WGPU_SHADER_CODE(wgslSrc, wgslSource.c_str(), wgslSource.size());

            WGPUShaderModuleDescriptor shaderDesc{};
            shaderDesc.nextInChain = &wgslSrc.chain;
            shaderDesc.label = WGPU_LABEL("shaderpass_module");
            shaderModule = wgpuDeviceCreateShaderModule(device, &shaderDesc);

            // Color target — opaque overwrite, no blending
            WGPUColorTargetState colorTarget{};
            colorTarget.format = format;
            colorTarget.writeMask = WGPUColorWriteMask_All;

            auto vsEntry = WGPU_ENTRY("vs");
            auto fsEntry = WGPU_ENTRY("fs");

            WGPUFragmentState fragmentState{};
            fragmentState.module = shaderModule;
            fragmentState.entryPoint = fsEntry;
            fragmentState.targetCount = 1;
            fragmentState.targets = &colorTarget;

            WGPURenderPipelineDescriptor pipeDesc{};
            pipeDesc.label = WGPU_LABEL("shaderpass_pipe");
            pipeDesc.layout = layout;
            pipeDesc.vertex.module = shaderModule;
            pipeDesc.vertex.entryPoint = vsEntry;
            pipeDesc.vertex.bufferCount = 0;
            pipeDesc.vertex.buffers = nullptr;
            pipeDesc.fragment = &fragmentState;
            pipeDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
            pipeDesc.primitive.cullMode = WGPUCullMode_None;
            pipeDesc.multisample.count = 1;
            pipeDesc.multisample.mask = ~0u;

            pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeDesc);
        }
    };

}// namespace threepp

#endif//THREEPP_SHADERPASSIMPL_HPP
