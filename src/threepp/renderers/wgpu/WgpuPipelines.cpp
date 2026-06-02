// Pipeline creation and caching for the Wgpu renderer backend.

#include "WgpuPipelines.hpp"
#include "WgpuGeometries.hpp"
#include "WgpuShaders.hpp"

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"
#include "threepp/textures/DepthTexture.hpp"

#include <set>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#ifdef THREEPP_WGPU_GLSL_COMPAT
#include "WgpuShaderTranslator.hpp"
#endif

namespace threepp::wgpu {

    WgpuPipelines::WgpuPipelines(WgpuState& state) : state_(state) {}

    std::vector<WGPUBindGroupLayoutEntry> WgpuPipelines::buildBindGroupLayoutEntries(uint64_t features) const {
        std::vector<WGPUBindGroupLayoutEntry> entries;
        bool lit = ShaderFeatures::isLit(features);

        // Helper lambda: add a texture + sampler binding pair (fragment-visible)
        auto addTexSamplerBindings = [&](uint32_t texBinding, uint32_t sampBinding) {
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = texBinding;
                e.visibility = WGPUShaderStage_Fragment;
                e.texture.sampleType = WGPUTextureSampleType_Float;
                e.texture.viewDimension = WGPUTextureViewDimension_2D;
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = sampBinding;
                e.visibility = WGPUShaderStage_Fragment;
                e.sampler.type = WGPUSamplerBindingType_Filtering;
                entries.push_back(e);
            }
        };

        // Binding 0: transform uniforms
        {
            WGPUBindGroupLayoutEntry e{};
            e.binding = 0;
            e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
            e.buffer.type = WGPUBufferBindingType_Uniform;
            e.buffer.minBindingSize = TRANSFORM_UNIFORM_SIZE;
            entries.push_back(e);
        }

        // Binding 1: material uniforms (vertex stage needed for displacement map)
        {
            WGPUBindGroupLayoutEntry e{};
            e.binding = 1;
            e.visibility = WGPUShaderStage_Fragment |
                           ((features & ShaderFeatures::DisplacementMap) ? WGPUShaderStage_Vertex : 0);
            e.buffer.type = WGPUBufferBindingType_Uniform;
            e.buffer.minBindingSize = MATERIAL_UNIFORM_SIZE;
            entries.push_back(e);
        }

        // Binding 2: light uniforms (if lit)
        if (lit) {
            WGPUBindGroupLayoutEntry e{};
            e.binding = 2;
            e.visibility = WGPUShaderStage_Fragment;
            e.buffer.type = WGPUBufferBindingType_Uniform;
            e.buffer.minBindingSize = state_.lightLimits.lightUniformSize();
            entries.push_back(e);
        }

        // Binding 3: texture, Binding 4: sampler (if textured)
        if (features & ShaderFeatures::Texture) {
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 3;
                e.visibility = WGPUShaderStage_Fragment;
                e.texture.sampleType = WGPUTextureSampleType_Float;
                e.texture.viewDimension = WGPUTextureViewDimension_2D;
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 4;
                e.visibility = WGPUShaderStage_Fragment;
                e.sampler.type = WGPUSamplerBindingType_Filtering;
                entries.push_back(e);
            }
        }

        // Binding 5-6: normal map
        if (features & ShaderFeatures::NormalMap) {
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 5;
                e.visibility = WGPUShaderStage_Fragment;
                e.texture.sampleType = WGPUTextureSampleType_Float;
                e.texture.viewDimension = WGPUTextureViewDimension_2D;
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 6;
                e.visibility = WGPUShaderStage_Fragment;
                e.sampler.type = WGPUSamplerBindingType_Filtering;
                entries.push_back(e);
            }
        }

        // Binding 7-9: shadow map
        if (features & ShaderFeatures::Shadow) {
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 7;
                e.visibility = WGPUShaderStage_Fragment;
                e.buffer.type = WGPUBufferBindingType_Uniform;
                e.buffer.minBindingSize = state_.shadowLimits.shadowUniformSize();
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 8;
                e.visibility = WGPUShaderStage_Fragment;
                e.texture.sampleType = WGPUTextureSampleType_Depth;
                e.texture.viewDimension = WGPUTextureViewDimension_2DArray;
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 9;
                e.visibility = WGPUShaderStage_Fragment;
                e.sampler.type = WGPUSamplerBindingType_Comparison;
                entries.push_back(e);
            }
        }

        // Optional texture maps (each is a texture + sampler pair)
        if (features & ShaderFeatures::EmissiveMap)  addTexSamplerBindings(10, 11);
        if (features & ShaderFeatures::RoughnessMap) addTexSamplerBindings(12, 13);
        if (features & ShaderFeatures::MetalnessMap) addTexSamplerBindings(14, 15);
        if (features & ShaderFeatures::AOMap)        addTexSamplerBindings(16, 17);
        if (features & ShaderFeatures::AlphaMap)     addTexSamplerBindings(18, 19);
        if (features & ShaderFeatures::SpecularMap)  addTexSamplerBindings(20, 21);
        if (features & ShaderFeatures::LightMap)     addTexSamplerBindings(22, 23);
        if (features & ShaderFeatures::BumpMap)      addTexSamplerBindings(24, 25);
        if (features & ShaderFeatures::GradientMap)  addTexSamplerBindings(26, 27);

        // Binding 28: instance storage buffer
        if (features & ShaderFeatures::Instanced) {
            WGPUBindGroupLayoutEntry e{};
            e.binding = 28;
            e.visibility = WGPUShaderStage_Vertex;
            e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
            entries.push_back(e);
        }

        // Binding 29: morph target storage buffer
        if (features & ShaderFeatures::MorphTargets) {
            WGPUBindGroupLayoutEntry e{};
            e.binding = 29;
            e.visibility = WGPUShaderStage_Vertex;
            e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
            entries.push_back(e);
        }

        // Binding 30-31: displacement map (vertex stage)
        if (features & ShaderFeatures::DisplacementMap) {
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 30;
                e.visibility = WGPUShaderStage_Vertex;
                e.texture.sampleType = WGPUTextureSampleType_Float;
                e.texture.viewDimension = WGPUTextureViewDimension_2D;
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 31;
                e.visibility = WGPUShaderStage_Vertex;
                e.sampler.type = WGPUSamplerBindingType_Filtering;
                entries.push_back(e);
            }
        }

        // Binding 32-33: environment map (equirect 2D or cube)
        if (features & (ShaderFeatures::EnvMap | ShaderFeatures::EnvMapCube)) {
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 32;
                e.visibility = WGPUShaderStage_Fragment;
                e.texture.sampleType = WGPUTextureSampleType_Float;
                e.texture.viewDimension = (features & ShaderFeatures::EnvMapCube)
                    ? WGPUTextureViewDimension_Cube
                    : WGPUTextureViewDimension_2D;
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 33;
                e.visibility = WGPUShaderStage_Fragment;
                e.sampler.type = WGPUSamplerBindingType_Filtering;
                entries.push_back(e);
            }
        }

        // Binding 36-37: point light shadow map
        if (features & ShaderFeatures::Shadow) {
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 36;
                e.visibility = WGPUShaderStage_Fragment;
                e.buffer.type = WGPUBufferBindingType_Uniform;
                e.buffer.minBindingSize = state_.shadowLimits.pointShadowUniformSize();
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 37;
                e.visibility = WGPUShaderStage_Fragment;
                e.texture.sampleType = WGPUTextureSampleType_Depth;
                e.texture.viewDimension = WGPUTextureViewDimension_2DArray;
                entries.push_back(e);
            }
        }

        // Binding 34-35: skinning (joint matrices + weights)
        if (features & ShaderFeatures::Skinning) {
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 34;
                e.visibility = WGPUShaderStage_Vertex;
                e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 35;
                e.visibility = WGPUShaderStage_Vertex;
                e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
                entries.push_back(e);
            }
        }

        // Binding 38-39: transmission sampler map
        if (features & ShaderFeatures::Transmission) {
            addTexSamplerBindings(38, 39);
        }

        // Bindings 40-42: RectAreaLight LTC lookup textures (shared sampler).
        if (features & ShaderFeatures::RectAreaLights) {
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 40;
                e.visibility = WGPUShaderStage_Fragment;
                e.texture.sampleType = WGPUTextureSampleType_Float;
                e.texture.viewDimension = WGPUTextureViewDimension_2D;
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 41;
                e.visibility = WGPUShaderStage_Fragment;
                e.texture.sampleType = WGPUTextureSampleType_Float;
                e.texture.viewDimension = WGPUTextureViewDimension_2D;
                entries.push_back(e);
            }
            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 42;
                e.visibility = WGPUShaderStage_Fragment;
                e.sampler.type = WGPUSamplerBindingType_Filtering;
                entries.push_back(e);
            }
        }

        return entries;
    }

    PipelineEntry& WgpuPipelines::getOrCreatePipeline(uint64_t features,
                                                       WGPUTextureFormat surfaceFormat,
                                                       uint32_t sampleCount) {
        // Encode sampleCount and target format flavor so pipelines compiled
        // for different MSAA counts or color targets cache separately. The
        // surface direct path uses *UnormSrgb (GPU does sRGB encode); RTs use
        // the linear *Unorm form (in-shader sRGB encode); intermediate uses
        // RGBA16Float. Three buckets, two bits.
        const bool isFloat16 = (surfaceFormat == WGPUTextureFormat_RGBA16Float);
        const bool isSrgb = (surfaceFormat == WGPUTextureFormat_BGRA8UnormSrgb
                          || surfaceFormat == WGPUTextureFormat_RGBA8UnormSrgb);
        const uint64_t cacheKey = features
                                | (sampleCount > 1 ? (1ULL << 62) : 0ULL)
                                | (isFloat16 ? (1ULL << 61) : 0ULL)
                                | (isSrgb ? (1ULL << 60) : 0ULL);
        auto it = pipelineCache_.find(cacheKey);
        if (it != pipelineCache_.end()) return it->second;

        PipelineEntry entry{};

        // Shader module
        std::string wgsl = buildWGSL(features, state_.lightLimits, state_.shadowLimits);
        WGPUShaderSourceWGSL wgslSource{};
        wgslSource.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslSource.chain.next = nullptr;
        wgslSource.code = {.data = wgsl.c_str(), .length = static_cast<size_t>(wgsl.size())};

        WGPUShaderModuleDescriptor shaderDesc{};
        shaderDesc.nextInChain = &wgslSource.chain;
        shaderDesc.label = WGPUStringView{"wgpu_shader", WGPU_STRLEN} ;
        entry.shader = wgpuDeviceCreateShaderModule(state_.device, &shaderDesc);
        if (!entry.shader) {
            std::cerr << "WgpuPipelines: Failed to create shader module for features 0x"
                      << std::hex << features << std::dec << std::endl;
            return pipelineCache_[features];
        }

        // Bind group layout
        auto bglEntries = buildBindGroupLayoutEntries(features);

        WGPUBindGroupLayoutDescriptor bglDesc{};
        bglDesc.label = WGPUStringView{"bind_group_layout", WGPU_STRLEN} ;
        bglDesc.entryCount = bglEntries.size();
        bglDesc.entries = bglEntries.data();
        entry.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(state_.device, &bglDesc);

        // Pipeline layout
        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.label = WGPUStringView{"pipeline_layout", WGPU_STRLEN} ;
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &entry.bindGroupLayout;
        entry.layout = wgpuDeviceCreatePipelineLayout(state_.device, &plDesc);

        // Vertex buffer layout: pos(vec3) + normal(vec3) + uv(vec2) + color(vec3) + uv2(vec2) = 52 bytes
        WGPUVertexAttribute attrs[5]{};
        attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0;  attrs[0].shaderLocation = 0;
        attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
        attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;
        attrs[3].format = WGPUVertexFormat_Float32x3; attrs[3].offset = 32; attrs[3].shaderLocation = 3;
        attrs[4].format = WGPUVertexFormat_Float32x2; attrs[4].offset = 44; attrs[4].shaderLocation = 4;

        WGPUVertexBufferLayout vbLayout{};
        vbLayout.arrayStride = VERTEX_STRIDE;
        vbLayout.stepMode = WGPUVertexStepMode_Vertex;
        vbLayout.attributeCount = 5;
        vbLayout.attributes = attrs;

        // Blend state from feature bits
        WGPUBlendState blendState{};
        uint64_t blendBits = features & ShaderFeatures::BlendMask;
        WGPUColorTargetState colorTarget{};
        colorTarget.format = surfaceFormat;
        colorTarget.writeMask = WGPUColorWriteMask_All;

        if (blendBits == ShaderFeatures::BlendDisabled) {
            colorTarget.blend = nullptr;
        } else {
            if (blendBits == ShaderFeatures::BlendAdditive) {
                blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
                blendState.color.dstFactor = WGPUBlendFactor_One;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_One;
                blendState.alpha.dstFactor = WGPUBlendFactor_One;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else if (blendBits == ShaderFeatures::BlendSubtractive) {
                blendState.color.srcFactor = WGPUBlendFactor_Zero;
                blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrc;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
                blendState.alpha.dstFactor = WGPUBlendFactor_One;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else if (blendBits == ShaderFeatures::BlendMultiply) {
                blendState.color.srcFactor = WGPUBlendFactor_Zero;
                blendState.color.dstFactor = WGPUBlendFactor_Src;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
                blendState.alpha.dstFactor = WGPUBlendFactor_SrcAlpha;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else {
                // BlendNormal (default)
                blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
                blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_One;
                blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            }
            colorTarget.blend = &blendState;
        }

        auto fsEntry = WGPUStringView{"fs_main", WGPU_STRLEN} ;
        WGPUFragmentState fragmentState{};
        fragmentState.module = entry.shader;
        fragmentState.entryPoint = fsEntry;
        fragmentState.targetCount = 1;
        fragmentState.targets = &colorTarget;

        WGPUDepthStencilState depthStencil{};
        depthStencil.format = WGPUTextureFormat_Depth24Plus;
        depthStencil.depthWriteEnabled = (features & ShaderFeatures::DepthWriteOff)
                                              ? WGPUOptionalBool_False
                                              : WGPUOptionalBool_True;
        depthStencil.depthCompare = (features & ShaderFeatures::DepthTestOff)
                                          ? WGPUCompareFunction_Always
                                          : WGPUCompareFunction_LessEqual;

        WGPURenderPipelineDescriptor pipelineDesc{};
        pipelineDesc.label = WGPUStringView{"wgpu_pipeline", WGPU_STRLEN} ;
        pipelineDesc.layout = entry.layout;

        auto vsEntry = WGPUStringView{"vs_main", WGPU_STRLEN} ;
        pipelineDesc.vertex.module = entry.shader;
        pipelineDesc.vertex.entryPoint = vsEntry;
        pipelineDesc.vertex.bufferCount = 1;
        pipelineDesc.vertex.buffers = &vbLayout;

        // Topology selection
        uint64_t topoBits = features & ShaderFeatures::TopoMask;
        if (features & ShaderFeatures::Wireframe) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineList;
        } else if (topoBits == ShaderFeatures::TopoLineList) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineList;
        } else if (topoBits == ShaderFeatures::TopoLineStrip) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineStrip;
            pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Uint32;
        } else if (topoBits == ShaderFeatures::TopoPointList) {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_PointList;
        } else {
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        }
        pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;

        // Face culling
        uint64_t cullBits = features & ShaderFeatures::CullMask;
        if (cullBits == ShaderFeatures::CullFront) {
            pipelineDesc.primitive.cullMode = WGPUCullMode_Front;
        } else if (cullBits == ShaderFeatures::CullBack) {
            pipelineDesc.primitive.cullMode = WGPUCullMode_Back;
        } else {
            pipelineDesc.primitive.cullMode = WGPUCullMode_None;
        }

        pipelineDesc.depthStencil = &depthStencil;
        pipelineDesc.multisample.count = sampleCount;
        pipelineDesc.multisample.mask = 0xFFFFFFFF;
        pipelineDesc.fragment = &fragmentState;

        entry.pipeline = wgpuDeviceCreateRenderPipeline(state_.device, &pipelineDesc);
        if (!entry.pipeline) {
            std::cerr << "WgpuPipelines: Failed to create render pipeline for features 0x"
                      << std::hex << features << std::dec << std::endl;
        }

        pipelineCache_[cacheKey] = entry;
        return pipelineCache_[cacheKey];
    }

    CustomPipelineEntry& WgpuPipelines::getOrCreateCustomPipeline(
            ShaderMaterial* sm,
            WGPUTextureFormat surfaceFormat,
            uint32_t sampleCount) {

        const bool isGlsl = sm->vertexShader.find("gl_Position") != std::string::npos ||
                             sm->fragmentShader.find("gl_FragColor") != std::string::npos;

        // Encode sampleCount in bit 32, RGBA16Float in bit 33, sRGB target in bit 34.
        // Same three-bucket scheme as getOrCreatePipeline above: surface direct
        // uses *UnormSrgb, RTs use linear, intermediate uses RGBA16Float.
        const bool isFloat16Custom = (surfaceFormat == WGPUTextureFormat_RGBA16Float);
        const bool isSrgbCustom = (surfaceFormat == WGPUTextureFormat_BGRA8UnormSrgb
                                || surfaceFormat == WGPUTextureFormat_RGBA8UnormSrgb);
        const uint64_t customCacheKey = static_cast<uint64_t>(sm->id)
                                      | (sampleCount > 1 ? (1ULL << 32) : 0ULL)
                                      | (isFloat16Custom ? (1ULL << 33) : 0ULL)
                                      | (isSrgbCustom ? (1ULL << 34) : 0ULL);

#ifndef THREEPP_WGPU_GLSL_COMPAT
        if (isGlsl) {
            // GLSL compat disabled at build time — skip silently
            return customPipelineCache_[customCacheKey];
        }
#endif

        // Combine vertex + fragment shader into one module (WGSL path)
        // For the GLSL path the hash still covers both shader strings.
        std::string wgsl = isGlsl ? "" : (sm->vertexShader + "\n" + sm->fragmentShader);
        size_t shaderHash = std::hash<std::string>{}(sm->vertexShader + "##" + sm->fragmentShader);

        auto it = customPipelineCache_.find(customCacheKey);
        bool needRebuild = (it == customPipelineCache_.end() || it->second.shaderHash != shaderHash);

        if (needRebuild) {
            // Clean up old entry
            if (it != customPipelineCache_.end()) {
                auto& old = it->second;
                if (old.pipeline)    wgpuRenderPipelineRelease(old.pipeline);
                if (old.layout)      wgpuPipelineLayoutRelease(old.layout);
                if (old.bindGroupLayout) wgpuBindGroupLayoutRelease(old.bindGroupLayout);
                if (old.shader)      wgpuShaderModuleRelease(old.shader);
                if (old.vertShader)  wgpuShaderModuleRelease(old.vertShader);
                if (old.fragShader)  wgpuShaderModuleRelease(old.fragShader);
            }

            CustomPipelineEntry entry{};
            entry.shaderHash = shaderHash;

#ifdef THREEPP_WGPU_GLSL_COMPAT
            if (isGlsl) {
                // Collect non-texture uniform names and texture names
                std::vector<std::string> uniformNames;
                std::vector<std::string> texNames;
                for (auto& [name, uniform] : sm->uniforms) {
                    if (!uniform.hasValue()) continue;
                    auto& val = const_cast<Uniform&>(uniform).value();
                    if (std::get_if<Texture*>(&val)) {
                        texNames.push_back(name);
                    } else {
                        uniformNames.push_back(name);
                    }
                }
                for (auto& [name, ptr] : sm->customTextures) {
                    texNames.push_back(name);
                }
                std::sort(texNames.begin(), texNames.end());
                texNames.erase(std::unique(texNames.begin(), texNames.end()), texNames.end());

                auto translated = translator_.translate(
                        sm->vertexShader, sm->fragmentShader,
                        uniformNames, texNames);

                if (!translated.success()) {
                    std::cerr << "[WgpuPipelines] GLSL translation failed for material "
                              << sm->id << ":\n" << translated.errorMessage << std::endl;
                    customPipelineCache_[customCacheKey] = CustomPipelineEntry{};
                    return customPipelineCache_[customCacheKey];
                }

                entry.customUniformNames = translated.customUniformNames;
                entry.customUniformSize = translated.customUniformSize;

                // Create vertex shader module from SPIR-V
                {
                    WGPUShaderSourceSPIRV spirvSrc{};
                    spirvSrc.chain.sType = WGPUSType_ShaderSourceSPIRV;
                    spirvSrc.chain.next  = nullptr;
                    spirvSrc.codeSize    = static_cast<uint32_t>(translated.vertexSpirv.size());
                    spirvSrc.code        = translated.vertexSpirv.data();

                    WGPUShaderModuleDescriptor desc{};
                    desc.nextInChain = &spirvSrc.chain;
                    desc.label       = WGPUStringView{"glsl_vert_spirv", WGPU_STRLEN} ;
                    entry.vertShader = wgpuDeviceCreateShaderModule(state_.device, &desc);
                    if (!entry.vertShader) {
                        std::cerr << "[WgpuPipelines] Failed to create vertex SPIR-V module\n";
                        customPipelineCache_[customCacheKey] = CustomPipelineEntry{};
                        return customPipelineCache_[customCacheKey];
                    }
                }

                // Create fragment shader module from SPIR-V
                {
                    WGPUShaderSourceSPIRV spirvSrc{};
                    spirvSrc.chain.sType = WGPUSType_ShaderSourceSPIRV;
                    spirvSrc.chain.next  = nullptr;
                    spirvSrc.codeSize    = static_cast<uint32_t>(translated.fragmentSpirv.size());
                    spirvSrc.code        = translated.fragmentSpirv.data();

                    WGPUShaderModuleDescriptor desc{};
                    desc.nextInChain = &spirvSrc.chain;
                    desc.label       = WGPUStringView{"glsl_frag_spirv", WGPU_STRLEN} ;
                    entry.fragShader = wgpuDeviceCreateShaderModule(state_.device, &desc);
                    if (!entry.fragShader) {
                        std::cerr << "[WgpuPipelines] Failed to create fragment SPIR-V module\n";
                        wgpuShaderModuleRelease(entry.vertShader);
                        customPipelineCache_[customCacheKey] = CustomPipelineEntry{};
                        return customPipelineCache_[customCacheKey];
                    }
                }
            } else
#endif
            {
                // Detect separate per-stage WGSL (naga-generated): each shader
                // string contains its own @vertex or @fragment entry point.
                bool hasSeparateWgsl = !isGlsl &&
                    sm->vertexShader.find("@vertex") != std::string::npos &&
                    sm->fragmentShader.find("@fragment") != std::string::npos;

                if (hasSeparateWgsl) {
                    // Create separate vertex WGSL module
                    {
                        WGPUShaderSourceWGSL wgslSrc{};
                        wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
                        wgslSrc.chain.next  = nullptr;
                        wgslSrc.code = {.data = sm->vertexShader.c_str(), .length = static_cast<size_t>(sm->vertexShader.size())};

                        WGPUShaderModuleDescriptor desc{};
                        desc.nextInChain = &wgslSrc.chain;
                        desc.label       = WGPUStringView{"wgsl_vert", WGPU_STRLEN} ;
                        entry.vertShader = wgpuDeviceCreateShaderModule(state_.device, &desc);
                        if (!entry.vertShader) {
                            std::cerr << "[WgpuPipelines] Failed to create separate vertex WGSL module\n";
                            customPipelineCache_[customCacheKey] = CustomPipelineEntry{};
                            return customPipelineCache_[customCacheKey];
                        }
                    }
                    // Create separate fragment WGSL module
                    {
                        WGPUShaderSourceWGSL wgslSrc{};
                        wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
                        wgslSrc.chain.next  = nullptr;
                        wgslSrc.code = {.data = sm->fragmentShader.c_str(), .length = static_cast<size_t>(sm->fragmentShader.size())};

                        WGPUShaderModuleDescriptor desc{};
                        desc.nextInChain = &wgslSrc.chain;
                        desc.label       = WGPUStringView{"wgsl_frag", WGPU_STRLEN} ;
                        entry.fragShader = wgpuDeviceCreateShaderModule(state_.device, &desc);
                        if (!entry.fragShader) {
                            std::cerr << "[WgpuPipelines] Failed to create separate fragment WGSL module\n";
                            wgpuShaderModuleRelease(entry.vertShader);
                            customPipelineCache_[customCacheKey] = CustomPipelineEntry{};
                            return customPipelineCache_[customCacheKey];
                        }
                    }
                } else {
                    // Combined WGSL path: single module with vs_main + fs_main
                    WGPUShaderSourceWGSL wgslSrc{};
                    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
                    wgslSrc.chain.next  = nullptr;
                    wgslSrc.code = {.data = wgsl.c_str(), .length = static_cast<size_t>(wgsl.size())};

                    WGPUShaderModuleDescriptor shaderDesc{};
                    shaderDesc.nextInChain = &wgslSrc.chain;
                    shaderDesc.label       = WGPUStringView{"custom_shader", WGPU_STRLEN} ;
                    entry.shader = wgpuDeviceCreateShaderModule(state_.device, &shaderDesc);
                }
            }

            // Build bind group layout entries
            // Binding 0: TransformUniforms (256 bytes) -- vertex + fragment
            // Binding 1: LightData (704 bytes) -- fragment
            // Bindings 2+: user-defined (discovered from customTextures)
            std::vector<WGPUBindGroupLayoutEntry> bglEntries;

            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 0;
                e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
                e.buffer.type = WGPUBufferBindingType_Uniform;
                e.buffer.minBindingSize = TRANSFORM_UNIFORM_SIZE;
                bglEntries.push_back(e);
            }

            {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 1;
                e.visibility = WGPUShaderStage_Fragment;
                e.buffer.type = WGPUBufferBindingType_Uniform;
                e.buffer.minBindingSize = state_.lightLimits.lightUniformSize();
                bglEntries.push_back(e);
            }

            // Custom uniform buffer at binding 2 (for ocean params etc.)
            // Exclude Texture* values — they are texture bindings, not UBO members.
            // Compute actual UBO size: each scalar/vec3 -> 16 bytes, mat4 -> 64 bytes.
            if (entry.customUniformSize == 0) {
                // WGSL path: compute from the material's uniform map
                for (auto& [name, uniform] : sm->uniforms) {
                    if (!uniform.hasValue()) continue;
                    auto& val = const_cast<Uniform&>(uniform).value();
                    if (std::get_if<Texture*>(&val)) continue;
                    if (std::get_if<Matrix4>(&val) || std::get_if<Matrix4*>(&val))
                        entry.customUniformSize += 64;
                    else
                        entry.customUniformSize += 16;
                }
            }
            bool hasCustomUniforms = entry.customUniformSize > 0;
            if (hasCustomUniforms) {
                WGPUBindGroupLayoutEntry e{};
                e.binding = 2;
                e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
                e.buffer.type = WGPUBufferBindingType_Uniform;
                e.buffer.minBindingSize = entry.customUniformSize;
                bglEntries.push_back(e);
            }

            // GPU texture bindings (texture + sampler pairs, sorted by name for determinism)
            // Collect from both customTextures and Texture* uniforms
            std::vector<std::string> texNames;
            std::set<std::string> depthTexNames; // DepthTexture uniforms -> unfilterable
            for (auto& [name, ptr] : sm->customTextures) {
                texNames.push_back(name);
            }
            for (auto& [name, uniform] : sm->uniforms) {
                if (!uniform.hasValue()) continue;
                auto& val = const_cast<Uniform&>(uniform).value();
                if (auto* tp = std::get_if<Texture*>(&val)) {
                    texNames.push_back(name);
                    if (*tp && dynamic_cast<DepthTexture*>(*tp)) depthTexNames.insert(name);
                }
            }
            std::sort(texNames.begin(), texNames.end());
            texNames.erase(std::unique(texNames.begin(), texNames.end()), texNames.end());

            uint32_t nextBinding = hasCustomUniforms ? 3 : 2;
            for (auto& name : texNames) {
                bool isCube = false;
                bool fromCustomTextures = sm->customTextures.count(name) > 0;
                bool isFilterable = false;
                if (fromCustomTextures) {
                    auto* gpuTex = static_cast<WgpuTexture*>(sm->customTextures[name]);
                    isCube = gpuTex->dimension() == WgpuTexture::Dimension::Cube;
                    auto fmt = gpuTex->format();
                    isFilterable = isCube ||
                        fmt == WgpuTexture::Format::RGBA16Float ||
                        fmt == WgpuTexture::Format::RGBA8Unorm;
                }
                // A DepthTexture uniform is bound to the RT's R32Float depth-resolve
                // texture, which is unfilterable-float and must use a non-filtering sampler.
                const bool isDepth = depthTexNames.count(name) > 0;
                // Textures from uniforms use filtering (unless they're a DepthTexture);
                // customTextures use filtering only when the format natively supports it.
                bool useFiltering = !isDepth && (isFilterable || !fromCustomTextures);
                {
                    WGPUBindGroupLayoutEntry e{};
                    e.binding = nextBinding++;
                    e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
                    e.texture.sampleType = useFiltering ? WGPUTextureSampleType_Float
                                                        : WGPUTextureSampleType_UnfilterableFloat;
                    e.texture.viewDimension = isCube ? WGPUTextureViewDimension_Cube
                                                     : WGPUTextureViewDimension_2D;
                    bglEntries.push_back(e);
                }
                {
                    WGPUBindGroupLayoutEntry e{};
                    e.binding = nextBinding++;
                    e.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
                    e.sampler.type = useFiltering ? WGPUSamplerBindingType_Filtering
                                                  : WGPUSamplerBindingType_NonFiltering;
                    bglEntries.push_back(e);
                }
            }

            entry.bglEntries = bglEntries;

            WGPUBindGroupLayoutDescriptor bglDesc{};
            bglDesc.label = WGPUStringView{"custom_bgl", WGPU_STRLEN} ;
            bglDesc.entryCount = bglEntries.size();
            bglDesc.entries = bglEntries.data();
            entry.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(state_.device, &bglDesc);

            WGPUPipelineLayoutDescriptor plDesc{};
            plDesc.label = WGPUStringView{"custom_pl", WGPU_STRLEN} ;
            plDesc.bindGroupLayoutCount = 1;
            plDesc.bindGroupLayouts = &entry.bindGroupLayout;
            entry.layout = wgpuDeviceCreatePipelineLayout(state_.device, &plDesc);

            // Vertex buffer layout: pos(0) + normal(12) + uv(24) + color(32) = 44 bytes
            // All 4 attributes are declared because the GLSL translator always emits
            // layout(location=3) in vec3 color; and wgpu validates SPIR-V vertex inputs.
            WGPUVertexAttribute attrs[4]{};
            attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0;  attrs[0].shaderLocation = 0;
            attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
            attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;
            attrs[3].format = WGPUVertexFormat_Float32x3; attrs[3].offset = 32; attrs[3].shaderLocation = 3;

            WGPUVertexBufferLayout vbLayout{};
            vbLayout.arrayStride = VERTEX_STRIDE;
            vbLayout.stepMode = WGPUVertexStepMode_Vertex;
            vbLayout.attributeCount = 4;
            vbLayout.attributes = attrs;

            WGPUBlendState blendState{};
            if (sm->blending == Blending::Additive) {
                blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
                blendState.color.dstFactor = WGPUBlendFactor_One;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_One;
                blendState.alpha.dstFactor = WGPUBlendFactor_One;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else if (sm->blending == Blending::Subtractive) {
                blendState.color.srcFactor = WGPUBlendFactor_Zero;
                blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrc;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
                blendState.alpha.dstFactor = WGPUBlendFactor_One;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else if (sm->blending == Blending::Multiply) {
                blendState.color.srcFactor = WGPUBlendFactor_Zero;
                blendState.color.dstFactor = WGPUBlendFactor_Src;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
                blendState.alpha.dstFactor = WGPUBlendFactor_SrcAlpha;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            } else {
                blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
                blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
                blendState.color.operation = WGPUBlendOperation_Add;
                blendState.alpha.srcFactor = WGPUBlendFactor_One;
                blendState.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
                blendState.alpha.operation = WGPUBlendOperation_Add;
            }

            WGPUColorTargetState colorTarget{};
            colorTarget.format = surfaceFormat;
            colorTarget.writeMask = WGPUColorWriteMask_All;
            colorTarget.blend = (sm->blending == Blending::None) ? nullptr : &blendState;

            // Choose entry points and shader modules based on pipeline type:
            // - SPIR-V (GLSL compat): separate modules, entry point "main"
            // - Separate WGSL (naga): separate modules, entry point "vs_main"/"fs_main"
            // - Combined WGSL: single module, entry point "vs_main"/"fs_main"
            const bool useSeparateModules = (entry.vertShader != nullptr);
            WGPUShaderModule vsModule = useSeparateModules ? entry.vertShader : entry.shader;
            WGPUShaderModule fsModule = useSeparateModules ? entry.fragShader : entry.shader;

            // SPIR-V modules (from glslang) use "main"; WGSL always uses vs_main/fs_main.
            const bool isSpirvPath = useSeparateModules && (entry.shader == nullptr);
            auto vsEntry = (isSpirvPath && isGlsl) ? WGPUStringView{"main", WGPU_STRLEN}  : WGPUStringView{"vs_main", WGPU_STRLEN} ;
            auto fsEntry = (isSpirvPath && isGlsl) ? WGPUStringView{"main", WGPU_STRLEN}  : WGPUStringView{"fs_main", WGPU_STRLEN} ;

            WGPUFragmentState fragmentState{};
            fragmentState.module = fsModule;
            fragmentState.entryPoint = fsEntry;
            fragmentState.targetCount = 1;
            fragmentState.targets = &colorTarget;

            auto mapDepthFunc = [](DepthFunc f) -> WGPUCompareFunction {
                switch (f) {
                    case DepthFunc::Never:        return WGPUCompareFunction_Never;
                    case DepthFunc::Always:       return WGPUCompareFunction_Always;
                    case DepthFunc::Less:         return WGPUCompareFunction_Less;
                    case DepthFunc::LessEqual:    return WGPUCompareFunction_LessEqual;
                    case DepthFunc::Equal:        return WGPUCompareFunction_Equal;
                    case DepthFunc::GreaterEqual: return WGPUCompareFunction_GreaterEqual;
                    case DepthFunc::Greater:      return WGPUCompareFunction_Greater;
                    case DepthFunc::NotEqual:     return WGPUCompareFunction_NotEqual;
                    default:                      return WGPUCompareFunction_LessEqual;
                }
            };

            WGPUDepthStencilState depthStencil{};
            depthStencil.format = WGPUTextureFormat_Depth24Plus;
            depthStencil.depthWriteEnabled = sm->depthWrite
                ? WGPUOptionalBool_True : WGPUOptionalBool_False;
            depthStencil.depthCompare = sm->depthTest
                ? mapDepthFunc(sm->depthFunc) : WGPUCompareFunction_Always;

            WGPUCullMode cullMode = WGPUCullMode_None;
            if (sm->side == Side::Front) cullMode = WGPUCullMode_Back;
            else if (sm->side == Side::Back) cullMode = WGPUCullMode_Front;

            WGPURenderPipelineDescriptor pipelineDesc{};
            pipelineDesc.label = WGPUStringView{"custom_pipeline", WGPU_STRLEN} ;
            pipelineDesc.layout = entry.layout;

            pipelineDesc.vertex.module = vsModule;
            pipelineDesc.vertex.entryPoint = vsEntry;
            pipelineDesc.vertex.bufferCount = 1;
            pipelineDesc.vertex.buffers = &vbLayout;
            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
            pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
            pipelineDesc.primitive.cullMode = cullMode;
            pipelineDesc.depthStencil = &depthStencil;
            pipelineDesc.multisample.count = sampleCount;
            pipelineDesc.multisample.mask = 0xFFFFFFFF;
            pipelineDesc.fragment = &fragmentState;

            entry.pipeline = wgpuDeviceCreateRenderPipeline(state_.device, &pipelineDesc);

            customPipelineCache_[customCacheKey] = entry;
        }

        return customPipelineCache_[customCacheKey];
    }

    void WgpuPipelines::invalidateAll() {
        for (auto& [key, entry] : pipelineCache_) {
            if (entry.pipeline) wgpuRenderPipelineRelease(entry.pipeline);
            if (entry.layout) wgpuPipelineLayoutRelease(entry.layout);
            if (entry.bindGroupLayout) wgpuBindGroupLayoutRelease(entry.bindGroupLayout);
            if (entry.shader) wgpuShaderModuleRelease(entry.shader);
        }
        pipelineCache_.clear();

        for (auto& [key, entry] : customPipelineCache_) {
            if (entry.pipeline)    wgpuRenderPipelineRelease(entry.pipeline);
            if (entry.layout)      wgpuPipelineLayoutRelease(entry.layout);
            if (entry.bindGroupLayout) wgpuBindGroupLayoutRelease(entry.bindGroupLayout);
            if (entry.shader)      wgpuShaderModuleRelease(entry.shader);
            if (entry.vertShader)  wgpuShaderModuleRelease(entry.vertShader);
            if (entry.fragShader)  wgpuShaderModuleRelease(entry.fragShader);
        }
        customPipelineCache_.clear();
    }

    void WgpuPipelines::onMaterialDispose(unsigned int materialId) {
        // Remove both the 1x and 4x MSAA cached variants (bit 32 encodes sampleCount > 1).
        for (uint64_t extra : {0ULL, 1ULL << 32}) {
            auto it = customPipelineCache_.find(static_cast<uint64_t>(materialId) | extra);
            if (it != customPipelineCache_.end()) {
                auto& entry = it->second;
                if (entry.pipeline)    wgpuRenderPipelineRelease(entry.pipeline);
                if (entry.layout)      wgpuPipelineLayoutRelease(entry.layout);
                if (entry.bindGroupLayout) wgpuBindGroupLayoutRelease(entry.bindGroupLayout);
                if (entry.shader)      wgpuShaderModuleRelease(entry.shader);
                if (entry.vertShader)  wgpuShaderModuleRelease(entry.vertShader);
                if (entry.fragShader)  wgpuShaderModuleRelease(entry.fragShader);
                customPipelineCache_.erase(it);
            }
        }
    }

    void WgpuPipelines::dispose() {
        invalidateAll();
    }

}// namespace threepp::wgpu
