#ifndef THREEPP_SHADERPASSIMPL_HPP
#define THREEPP_SHADERPASSIMPL_HPP

#include "threepp/renderers/wgpu/ShaderPass.hpp"

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
            WGPUShaderSourceWGSL wgslSrc{};
            wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
            wgslSrc.chain.next = nullptr;
            wgslSrc.code = {.data = wgslSource.c_str(), .length = wgslSource.size()};

            WGPUShaderModuleDescriptor shaderDesc{};
            shaderDesc.nextInChain = &wgslSrc.chain;
            shaderDesc.label = {.data = "shaderpass_module", .length = 17};
            shaderModule = wgpuDeviceCreateShaderModule(device, &shaderDesc);

            // Color target — opaque overwrite, no blending
            WGPUColorTargetState colorTarget{};
            colorTarget.format = format;
            colorTarget.writeMask = WGPUColorWriteMask_All;

            WGPUStringView vsEntry = {.data = "vs", .length = 2};
            WGPUStringView fsEntry = {.data = "fs", .length = 2};

            WGPUFragmentState fragmentState{};
            fragmentState.module = shaderModule;
            fragmentState.entryPoint = fsEntry;
            fragmentState.targetCount = 1;
            fragmentState.targets = &colorTarget;

            WGPURenderPipelineDescriptor pipeDesc{};
            pipeDesc.label = {.data = "shaderpass_pipe", .length = 15};
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
