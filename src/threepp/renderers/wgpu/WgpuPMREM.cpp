
#include "WgpuPMREM.hpp"

#include <algorithm>
#include <cstring>

using namespace threepp;
using namespace threepp::wgpu;

WgpuPMREM::WgpuPMREM(WgpuState& state) : state_(state) {}

WgpuPMREM::~WgpuPMREM() {
    if (pipeline_)       { wgpuRenderPipelineRelease(pipeline_);       pipeline_ = nullptr; }
    if (pipelineLayout_) { wgpuPipelineLayoutRelease(pipelineLayout_); pipelineLayout_ = nullptr; }
    if (bgl_)            { wgpuBindGroupLayoutRelease(bgl_);            bgl_ = nullptr; }
    if (sampler_)        { wgpuSamplerRelease(sampler_);                sampler_ = nullptr; }
    if (shader_)         { wgpuShaderModuleRelease(shader_);            shader_ = nullptr; }
}

void WgpuPMREM::ensurePipeline(WGPUTextureFormat format) {
    if (pipeline_ && pipelineFormat_ == format) return;

    if (pipeline_)       { wgpuRenderPipelineRelease(pipeline_);       pipeline_ = nullptr; }
    if (pipelineLayout_) { wgpuPipelineLayoutRelease(pipelineLayout_); pipelineLayout_ = nullptr; }
    if (bgl_)            { wgpuBindGroupLayoutRelease(bgl_);            bgl_ = nullptr; }
    if (sampler_)        { wgpuSamplerRelease(sampler_);                sampler_ = nullptr; }
    if (shader_)         { wgpuShaderModuleRelease(shader_);            shader_ = nullptr; }

    const char* wgsl = R"(
struct PMREMUniforms {
    roughness: f32,
    numSamples: u32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(0) var<uniform> u: PMREMUniforms;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var srcSampler: sampler;

fn vdc(bits: u32) -> f32 {
    var b = bits;
    b = (b << 16u) | (b >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return f32(b) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, N: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(N), vdc(i));
}

fn importanceSampleGGX(Xi: vec2<f32>, N: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 6.28318530718 * Xi.x;
    let cosTheta = sqrt(max((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y), 0.0));
    let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    let H_t = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    let upN = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(N.z) < 0.999);
    let tangent = normalize(cross(upN, N));
    let bitangent = cross(N, tangent);
    return normalize(tangent * H_t.x + bitangent * H_t.y + N * H_t.z);
}

fn sampleEquirect(dir: vec3<f32>, mip: f32) -> vec3<f32> {
    // three.js / GL convention: u = atan2(z, x) / (2π) + 0.5; v = asin(y) / π + 0.5
    let uPhi = atan2(dir.z, dir.x) * 0.15915494 + 0.5;
    let vTheta = asin(clamp(dir.y, -1.0, 1.0)) * 0.31830989 + 0.5;
    return textureSampleLevel(srcTex, srcSampler, vec2<f32>(uPhi, vTheta), mip).rgb;
}

fn ggxD(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265359 * d * d);
}

struct V {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1., -1.), vec2<f32>(3., -1.), vec2<f32>(-1., 3.));
    var o: V;
    o.pos = vec4<f32>(p[vi], 0., 1.);
    o.uv = p[vi] * 0.5 + vec2<f32>(0.5, 0.5);
    o.uv.y = 1.0 - o.uv.y;
    return o;
}

@fragment fn fs(in: V) -> @location(0) vec4<f32> {
    // Invert the equirect projection used by sampleEquirect():
    //   uv.x = atan2(dir.z, dir.x) / (2π) + 0.5   =>  atan2(z, x) = (uv.x - 0.5) · 2π
    //   uv.y = asin(dir.y) / π + 0.5              =>  dir.y = sin((uv.y - 0.5) · π)
    let phiAz = (in.uv.x - 0.5) * 6.28318530718;
    let thetaAsin = (in.uv.y - 0.5) * 3.14159265359;
    let cosT = cos(thetaAsin);
    let N = vec3<f32>(cosT * cos(phiAz), sin(thetaAsin), cosT * sin(phiAz));

    // For roughness = 0 (mip 0 won't call this, but guard), return source directly.
    if (u.roughness <= 0.0) {
        return vec4<f32>(sampleEquirect(N, 0.0), 1.0);
    }

    // Colbert/Krivanek mip-biased importance sampling (mirrors GLPMREM): each
    // GGX sample reads the source at a mip whose texels cover ~the sample's
    // solid angle, so a small ultra-bright sun is pre-averaged into its
    // neighbourhood instead of being hit (or MISSED) by a handful of discrete
    // samples. Sampling only mip 0 systematically lost the sun's irradiance —
    // env-lit floors read ~2x darker than GL on sunny HDRs.
    let srcDim = vec2<f32>(textureDimensions(srcTex, 0u));
    let saTexel = 4.0 * 3.14159265359 / (srcDim.x * srcDim.y);
    let maxSrcMip = f32(textureNumLevels(srcTex) - 1u);
    // Same source-relative streak floor as GL: roughness=1 reads the ~16x8 mip.
    let roughFloor = u.roughness * max(log2(srcDim.x) - 4.0, 0.0);

    var accumColor = vec3<f32>(0.0);
    var accumWeight = 0.0;
    for (var i = 0u; i < u.numSamples; i++) {
        let Xi = hammersley(i, u.numSamples);
        let H = importanceSampleGGX(Xi, N, u.roughness);
        let L = normalize(-N + 2.0 * dot(N, H) * H);
        let NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            let NdotH = max(dot(N, H), 0.0);
            let pdf = ggxD(NdotH, u.roughness) * 0.25 + 1e-4;  // V=N -> VdotH=NdotH
            let saSample = 1.0 / (f32(u.numSamples) * pdf);
            var mip = 0.5 * log2(saSample / saTexel);
            mip = clamp(max(mip, roughFloor), 0.0, maxSrcMip);
            // Clamp per-channel to suppress firefly / pixelation from ultra-bright HDR pixels
            // (sun disk, specular highlights). Per-channel min also handles Inf values in the
            // source — which occur in RGBE loaders when the exponent overflows float range.
            let s = min(sampleEquirect(L, mip), vec3<f32>(50.0));
            accumColor = accumColor + s * NdotL;
            accumWeight = accumWeight + NdotL;
        }
    }
    let color = accumColor / max(accumWeight, 0.001);
    return vec4<f32>(color, 1.0);
}
)";

    WGPUShaderSourceWGSL wgslSrc{};
    wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslSrc.code = {.data = wgsl, .length = static_cast<size_t>(strlen(wgsl))};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslSrc.chain;
    shader_ = wgpuDeviceCreateShaderModule(state_.device, &smDesc);

    WGPUBindGroupLayoutEntry bglEntries[3]{};
    bglEntries[0].binding    = 0;
    bglEntries[0].visibility = WGPUShaderStage_Fragment;
    bglEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
    bglEntries[0].buffer.minBindingSize = 16;
    bglEntries[1].binding    = 1;
    bglEntries[1].visibility = WGPUShaderStage_Fragment;
    bglEntries[1].texture.sampleType    = WGPUTextureSampleType_Float;
    bglEntries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
    bglEntries[2].binding    = 2;
    bglEntries[2].visibility = WGPUShaderStage_Fragment;
    bglEntries[2].sampler.type = WGPUSamplerBindingType_Filtering;

    WGPUBindGroupLayoutDescriptor bgld{};
    bgld.entryCount = 3;
    bgld.entries    = bglEntries;
    bgl_ = wgpuDeviceCreateBindGroupLayout(state_.device, &bgld);

    WGPUPipelineLayoutDescriptor pld{};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts     = &bgl_;
    pipelineLayout_ = wgpuDeviceCreatePipelineLayout(state_.device, &pld);

    WGPUSamplerDescriptor sd{};
    sd.magFilter    = WGPUFilterMode_Linear;
    sd.minFilter    = WGPUFilterMode_Linear;
    // Trilinear: the Colbert mip-biased sampling computes fractional source LODs.
    sd.mipmapFilter = WGPUMipmapFilterMode_Linear;
    sd.addressModeU = WGPUAddressMode_Repeat;
    sd.addressModeV = WGPUAddressMode_ClampToEdge;
    sd.lodMaxClamp = 32.0f;
    sd.maxAnisotropy = 1;
    sampler_ = wgpuDeviceCreateSampler(state_.device, &sd);

    WGPUVertexState vs{};
    vs.module     = shader_;
    vs.entryPoint = {.data = "vs", .length = 2};

    WGPUColorTargetState ct{};
    ct.format    = format;
    ct.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fs{};
    fs.module      = shader_;
    fs.entryPoint  = {.data = "fs", .length = 2};
    fs.targetCount = 1;
    fs.targets     = &ct;

    WGPURenderPipelineDescriptor rpd{};
    rpd.label               = WGPUStringView{"pmrem_pipe", WGPU_STRLEN};
    rpd.layout              = pipelineLayout_;
    rpd.vertex              = vs;
    rpd.fragment            = &fs;
    rpd.primitive.topology  = WGPUPrimitiveTopology_TriangleList;
    rpd.multisample.count   = 1;
    rpd.multisample.mask    = 0xFFFFFFFF;
    pipeline_ = wgpuDeviceCreateRenderPipeline(state_.device, &rpd);

    pipelineFormat_ = format;
}

void WgpuPMREM::prefilter2D(WGPUTexture texture, uint32_t width, uint32_t height,
                            uint32_t mipLevels, WGPUTextureFormat format) {
    if (mipLevels <= 1) return;
    ensurePipeline(format);

    // roughness reaches 1.0 at the last mip that still has angular resolution
    // (>= 32x16, i.e. floor(log2(W)) - 5), NOT at the tail of the chain: a 1x1
    // mip is one convolved direction sprayed onto every normal, which flattens
    // the diffuse IBL E(n) to a direction-independent constant (caught by the
    // top-lit furnace test — floors read horizon irradiance instead of
    // bright-sky-up). Must match maxMip in WgpuShaders.cpp's env sampling.
    uint32_t log2W = 0;
    for (uint32_t w = width; w > 1; w >>= 1) ++log2W;
    const uint32_t lastUseful = log2W > 5 ? std::min(log2W - 5, mipLevels - 1) : 0;

    // Snapshot the (box-mipped) source pyramid into a temp texture: the
    // mip-biased sampling reads arbitrary source mips while the passes below
    // overwrite mips 1..N-1 of the live texture, and a subresource can't be
    // both sampled and rendered to. flushPendingMipmaps runs the box-mip
    // generator before calling us, so the temp carries a full pyramid.
    WGPUTextureDescriptor tmpDesc{};
    tmpDesc.label = WGPUStringView{"pmrem_src_snapshot", WGPU_STRLEN};
    tmpDesc.size = {width, height, 1};
    tmpDesc.mipLevelCount = mipLevels;
    tmpDesc.sampleCount = 1;
    tmpDesc.dimension = WGPUTextureDimension_2D;
    tmpDesc.format = format;
    tmpDesc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    WGPUTexture tmpTex = wgpuDeviceCreateTexture(state_.device, &tmpDesc);

    WGPUCommandEncoderDescriptor ceDesc{};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(state_.device, &ceDesc);

    for (uint32_t level = 0; level < mipLevels; ++level) {
        WGPUTexelCopyTextureInfo src{};
        src.texture = texture;
        src.mipLevel = level;
        WGPUTexelCopyTextureInfo dst{};
        dst.texture = tmpTex;
        dst.mipLevel = level;
        WGPUExtent3D extent{std::max(1u, width >> level), std::max(1u, height >> level), 1};
        wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &extent);
    }

    WGPUTextureViewDescriptor srcVd{};
    srcVd.dimension      = WGPUTextureViewDimension_2D;
    srcVd.baseMipLevel   = 0;
    srcVd.mipLevelCount  = mipLevels;
    srcVd.baseArrayLayer = 0;
    srcVd.arrayLayerCount = 1;
    srcVd.aspect         = WGPUTextureAspect_All;
    WGPUTextureView srcView = wgpuTextureCreateView(tmpTex, &srcVd);

    for (uint32_t level = 1; level < mipLevels; ++level) {
        // Levels past lastUseful repeat roughness 1 so any LOD overshoot or
        // trilinear tail stays consistent with the irradiance level.
        const float roughness = lastUseful > 0
                                        ? std::min(1.f, static_cast<float>(level) / static_cast<float>(lastUseful))
                                        : 1.f;
        // Fewer samples for very rough mips (they average over a wide lobe anyway).
        const uint32_t samples = roughness < 0.5f ? 256u : 128u;

        struct { float roughness; uint32_t numSamples; float _pad0; float _pad1; } ubData{roughness, samples, 0.f, 0.f};
        WGPUBufferDescriptor ubd{};
        ubd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        ubd.size  = 16;
        WGPUBuffer ubuf = wgpuDeviceCreateBuffer(state_.device, &ubd);
        wgpuQueueWriteBuffer(state_.queue, ubuf, 0, &ubData, sizeof(ubData));

        WGPUTextureViewDescriptor dstVd{};
        dstVd.dimension      = WGPUTextureViewDimension_2D;
        dstVd.baseMipLevel   = level;
        dstVd.mipLevelCount  = 1;
        dstVd.baseArrayLayer = 0;
        dstVd.arrayLayerCount = 1;
        dstVd.aspect         = WGPUTextureAspect_All;
        WGPUTextureView dstView = wgpuTextureCreateView(texture, &dstVd);

        WGPUBindGroupEntry bgEntries[3]{};
        bgEntries[0].binding = 0;
        bgEntries[0].buffer  = ubuf;
        bgEntries[0].size    = 16;
        bgEntries[1].binding     = 1;
        bgEntries[1].textureView = srcView;
        bgEntries[2].binding = 2;
        bgEntries[2].sampler = sampler_;

        WGPUBindGroupDescriptor bgd{};
        bgd.layout     = bgl_;
        bgd.entryCount = 3;
        bgd.entries    = bgEntries;
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(state_.device, &bgd);

        WGPURenderPassColorAttachment ca{};
        ca.view       = dstView;
        ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        ca.loadOp     = WGPULoadOp_Clear;
        ca.storeOp    = WGPUStoreOp_Store;
        ca.clearValue = {0, 0, 0, 0};

        WGPURenderPassDescriptor rpassDesc{};
        rpassDesc.colorAttachmentCount = 1;
        rpassDesc.colorAttachments     = &ca;

        WGPURenderPassEncoder rpe = wgpuCommandEncoderBeginRenderPass(encoder, &rpassDesc);
        wgpuRenderPassEncoderSetPipeline(rpe, pipeline_);
        wgpuRenderPassEncoderSetBindGroup(rpe, 0, bg, 0, nullptr);
        wgpuRenderPassEncoderDraw(rpe, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(rpe);
        wgpuRenderPassEncoderRelease(rpe);

        wgpuBindGroupRelease(bg);
        wgpuTextureViewRelease(dstView);
        wgpuBufferRelease(ubuf);
    }

    WGPUCommandBufferDescriptor cbDesc{};
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
    wgpuQueueSubmit(state_.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(encoder);

    wgpuTextureViewRelease(srcView);
    wgpuTextureRelease(tmpTex);
}
