
#include "threepp/renderers/wgpu/WgpuComputePipeline.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"
#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"

#include <webgpu/webgpu.h>

#include <unordered_map>
#include <stdexcept>

using namespace threepp;

namespace {

    WGPUTextureFormat toWGPUFormat(WgpuTexture::Format fmt) {
        switch (fmt) {
            case WgpuTexture::Format::RGBA32Float: return WGPUTextureFormat_RGBA32Float;
            case WgpuTexture::Format::RGBA16Float: return WGPUTextureFormat_RGBA16Float;
            case WgpuTexture::Format::RG32Float:   return WGPUTextureFormat_RG32Float;
            case WgpuTexture::Format::RGBA8Unorm:  return WGPUTextureFormat_RGBA8Unorm;
        }
        return WGPUTextureFormat_RGBA8Unorm;
    }

    // rgba16float and rgba8unorm are filterable (WGPUTextureSampleType_Float).
    // rg32float / rgba32float require the float32-filterable feature; declare as UnfilterableFloat
    // to avoid needing that feature for compute shaders that use textureLoad.
    WGPUTextureSampleType sampleType(WgpuTexture::Format fmt) {
        switch (fmt) {
            case WgpuTexture::Format::RGBA16Float:
            case WgpuTexture::Format::RGBA8Unorm:
                return WGPUTextureSampleType_Float;
            default:
                return WGPUTextureSampleType_UnfilterableFloat;
        }
    }

}// namespace

struct WgpuComputePipeline::BindingInfo {
    BindingType type;
    // For textures
    WGPUTextureView textureView = nullptr;
    WGPUTextureFormat textureFormat = WGPUTextureFormat_Undefined;
    // For buffers
    WGPUBuffer buffer = nullptr;
    size_t bufferSize = 0;
};

struct WgpuComputePipeline::Impl {
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUShaderModule shaderModule = nullptr;
    WGPUComputePipeline pipeline = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    std::string entryPoint;

    std::unordered_map<uint32_t, BindingInfo> bindings;
    bool pipelineBuilt = false;

    ~Impl() {
        if (pipeline) wgpuComputePipelineRelease(pipeline);
        if (pipelineLayout) wgpuPipelineLayoutRelease(pipelineLayout);
        if (bindGroupLayout) wgpuBindGroupLayoutRelease(bindGroupLayout);
        if (shaderModule) wgpuShaderModuleRelease(shaderModule);
    }

    void buildPipeline() {
        if (pipelineBuilt) {
            // Rebuild bind group layout if bindings changed
            if (bindGroupLayout) wgpuBindGroupLayoutRelease(bindGroupLayout);
            if (pipelineLayout) wgpuPipelineLayoutRelease(pipelineLayout);
            if (pipeline) wgpuComputePipelineRelease(pipeline);
        }

        // Build bind group layout entries
        std::vector<WGPUBindGroupLayoutEntry> bglEntries;
        for (auto& [binding, info] : bindings) {
            WGPUBindGroupLayoutEntry e{};
            e.binding = binding;
            e.visibility = WGPUShaderStage_Compute;

            switch (info.type) {
                case BindingType::StorageTextureWrite:
                    e.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
                    e.storageTexture.format = info.textureFormat;
                    e.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
                    break;
                case BindingType::StorageTextureRead:
                case BindingType::Texture: {
                    bool filterable = (info.textureFormat == WGPUTextureFormat_RGBA16Float ||
                                       info.textureFormat == WGPUTextureFormat_RGBA8Unorm);
                    e.texture.sampleType = filterable ? WGPUTextureSampleType_Float
                                                      : WGPUTextureSampleType_UnfilterableFloat;
                    e.texture.viewDimension = WGPUTextureViewDimension_2D;
                    break;
                }
                case BindingType::UniformBuffer:
                    e.buffer.type = WGPUBufferBindingType_Uniform;
                    e.buffer.minBindingSize = info.bufferSize;
                    break;
            }
            bglEntries.push_back(e);
        }

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = WGPUStringView{"compute_bgl", WGPU_STRLEN} ;
        bglDesc.entryCount = bglEntries.size();
        bglDesc.entries = bglEntries.data();
        bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = WGPUStringView{"compute_pl", WGPU_STRLEN} ;
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bindGroupLayout;
        pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);

        auto ep = WGPUStringView{entryPoint.c_str(), static_cast<size_t>(entryPoint.size())};
        WGPUComputePipelineDescriptor pipeDesc{};
        pipeDesc.label = WGPUStringView{"compute_pipe", WGPU_STRLEN} ;
        pipeDesc.layout = pipelineLayout;
        pipeDesc.compute.module = shaderModule;
        pipeDesc.compute.entryPoint = ep;
        pipeline = wgpuDeviceCreateComputePipeline(device, &pipeDesc);

        pipelineBuilt = true;
    }
};

WgpuComputePipeline::WgpuComputePipeline(WgpuRenderer& renderer, const std::string& wgslSource,
                                 const std::string& entryPoint)
    : impl_(new Impl()) {

    impl_->device = static_cast<WGPUDevice>(renderer.nativeDevice());
    impl_->queue = static_cast<WGPUQueue>(renderer.nativeQueue());
    impl_->entryPoint = entryPoint;

    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.chain.next = nullptr;
    wgslSrc.code = {.data = wgslSource.c_str(), .length = static_cast<size_t>(wgslSource.size())};

    WGPUShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &wgslSrc.chain;
    shaderDesc.label = WGPUStringView{"compute_shader", WGPU_STRLEN} ;
    impl_->shaderModule = wgpuDeviceCreateShaderModule(impl_->device, &shaderDesc);
}

WgpuComputePipeline::~WgpuComputePipeline() {
    delete impl_;
}

void WgpuComputePipeline::setStorageTexture(uint32_t binding, WgpuTexture& texture) {
    auto fmt = toWGPUFormat(texture.format());
    auto it = impl_->bindings.find(binding);
    // Only invalidate the pipeline when the binding layout changes (type or format).
    // A view-only swap does not affect the BGL or pipeline — bind group handles it.
    if (it == impl_->bindings.end() ||
        it->second.type != BindingType::StorageTextureWrite ||
        it->second.textureFormat != fmt) {
        impl_->pipelineBuilt = false;
    }
    BindingInfo info{};
    info.type = BindingType::StorageTextureWrite;
    info.textureView = texture.view();
    info.textureFormat = fmt;
    impl_->bindings[binding] = info;
}

void WgpuComputePipeline::setTexture(uint32_t binding, WgpuTexture& texture) {
    auto fmt = toWGPUFormat(texture.format());
    auto it = impl_->bindings.find(binding);
    if (it == impl_->bindings.end() ||
        it->second.type != BindingType::Texture ||
        it->second.textureFormat != fmt) {
        impl_->pipelineBuilt = false;
    }
    BindingInfo info{};
    info.type = BindingType::Texture;
    info.textureView = texture.view();
    info.textureFormat = fmt;
    impl_->bindings[binding] = info;
}

void WgpuComputePipeline::setUniformBuffer(uint32_t binding, WgpuBuffer& buffer) {
    auto it = impl_->bindings.find(binding);
    if (it == impl_->bindings.end() ||
        it->second.type != BindingType::UniformBuffer ||
        it->second.bufferSize != buffer.size()) {
        impl_->pipelineBuilt = false;
    }
    BindingInfo info{};
    info.type = BindingType::UniformBuffer;
    info.buffer = buffer.buffer();
    info.bufferSize = buffer.size();
    impl_->bindings[binding] = info;
}

void WgpuComputePipeline::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    if (!impl_->pipelineBuilt) {
        impl_->buildPipeline();
    }

    // Build bind group from current bindings
    std::vector<WGPUBindGroupEntry> entries;
    for (auto& [binding, info] : impl_->bindings) {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        switch (info.type) {
            case BindingType::StorageTextureWrite:
            case BindingType::StorageTextureRead:
            case BindingType::Texture:
                e.textureView = info.textureView;
                break;
            case BindingType::UniformBuffer:
                e.buffer = info.buffer;
                e.offset = 0;
                e.size = info.bufferSize;
                break;
        }
        entries.push_back(e);
    }

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.label = WGPUStringView{"compute_bg", WGPU_STRLEN} ;
    bgDesc.layout = impl_->bindGroupLayout;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(impl_->device, &bgDesc);

    // Create command encoder and dispatch
    WGPUCommandEncoderDescriptor encDesc{};
    encDesc.label = WGPUStringView{"compute_enc", WGPU_STRLEN} ;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(impl_->device, &encDesc);

    WGPUComputePassDescriptor passDesc{};
    passDesc.label = WGPUStringView{"compute_pass", WGPU_STRLEN} ;
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);

    wgpuComputePassEncoderSetPipeline(pass, impl_->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, z);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBufferDescriptor cmdDesc{};
    cmdDesc.label = WGPUStringView{"compute_cmd", WGPU_STRLEN} ;
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(impl_->queue, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bindGroup);
}
