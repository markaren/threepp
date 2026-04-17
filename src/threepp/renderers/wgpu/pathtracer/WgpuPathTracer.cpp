
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/wgpu/WgpuComputePipeline.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"

#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerAtlas.hpp"
#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerBCn.hpp"
#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerBvh.hpp"
#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerEnvCdf.hpp"
#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerGeometry.hpp"
#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerShaders.hpp"
#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerTypes.hpp"

#include <tuple>

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <webgpu/webgpu.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#ifndef __EMSCRIPTEN__
#include <future>
#endif
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <webgpu/wgpu.h>

using namespace threepp;
using namespace threepp::wgpu_pt;

// Software BCn / DXT decompressor (DXT1/3/5, BC4, BC5, BC7) lives in
// WgpuPathTracerBCn.{hpp,cpp}. The atlas builder calls bcnDecompress() from there.


// Limits, atlas constants and small helpers (nextPow2, triTexPages) live in
// WgpuPathTracerTypes.hpp and are imported via the using-directive above.

// All embedded WGSL shader strings and shader builder helpers now live in
// WgpuPathTracerShaders_{Rt,VtRefit,Denoise,Display}.cpp, declared in
// WgpuPathTracerShaders.hpp. POD types (uniform structs, BvhNode, PingPong,
// atlas constants) live in WgpuPathTracerTypes.hpp.

// ---------------------------------------------------------------------------
// WgpuPathTracer::Impl
// ---------------------------------------------------------------------------
struct WgpuPathTracer::Impl {
    WgpuRenderer& renderer;
    WGPUDevice device;
    WGPUQueue queue;

    // GPU textures
    WgpuTexture triTex;
    WgpuTexture matTex;
    WgpuTexture texAtlasTex;
    int atlasLayers_ = 0;  // current atlas row count (0 = initial placeholder)
    int atlasCols_ = ATLAS_WIDTH / DEFAULT_TILE_SIZE;
    int tileSize_ = DEFAULT_TILE_SIZE;
    int textureResolution_ = DEFAULT_TILE_SIZE;  // user config: 1024 or 2048
    PingPong accum;
    PingPong hitMesh;
    WgpuTexture envTexGpu;   // equirectangular env map for IBL (1x1 placeholder when unused)
    WgpuTexture bgTexGpu;   // equirectangular background for ray misses (1x1 placeholder when unused)
    WgpuTexture envCdfTex;  // conditional CDF (width × height), R32Float
    WgpuTexture envMargTex; // marginal CDF (height × 1), R32Float
    float       envLumSum_ = 0.f;  // total luminance sum for PDF normalization
    Texture*    prevEnvTex_ = nullptr;
    Texture*    prevBgTex_  = nullptr;

    // G-buffer ping-pong (read = current frame, write = previous frame)
    PingPong gBuf;

    // ReSTIR DI reservoir ping-pong
    PingPong reservoir;   // rgba32float — lightPos.xyz + encoded type/index
    PingPong reservoirW;  // rgba32float — W_sum, M, W, p_hat

    // ReSTIR GI reservoir ping-pong
    PingPong giRes;    // rgba32float — secHitPos.xyz + octahedral-packed normal
    PingPong giResW;   // rgba32float — W_sum, M, W, p_hat
    PingPong giResLo;  // rgba16float — Lo radiance at secondary hit

    // Albedo buffer (primary-hit albedo for demodulation/remodulation)
    WgpuTexture albedoTex;

    // Temporal variance moments ping-pong (μ, μ² of luminance)
    PingPong moments;

    // Diffuse/specular split accumulation
    PingPong diffAccum;
    PingPong specAccum;

    // Spatial filter ping-pong (shared between diffuse and specular passes)
    PingPong filtered;
    // Denoised output textures for display
    WgpuTexture denoisedDiff;
    WgpuTexture denoisedSpec;

    // Temporal upscale (TAAU) — full-res ping-pong, active when pixelScale < 0.85
    PingPong upscale;
    WgpuTexture zeroTex;           // 1×1 dummy; bound when upscale inactive
    WgpuComputePipeline upscalePipeline;
    WgpuBuffer          upscaleUniBuf;
    UpscaleGpuUniforms  upscaleUBO{};

    // Denoiser pipeline (spatial à-trous only)
    WgpuComputePipeline atrousPipeline;
    WgpuBuffer  atrousUniBuf;
    bool denoiserEnabled_ = true;
    bool restirEnabled_ = true;
    bool restirGiEnabled_ = false;
    int spp_ = 1;
    float envIntensity_ = 0.5f;
    int maxBounces_ = 4;
    float exposure_ = 1.0f;
    // Per-contribution firefly clamp (luminance cap) on indirect MIS paths.
    // Default 8.0 matches production renderers (Arnold/Cycles/RenderMan).
    // Set to a very large value (e.g. 1e30f) to disable clipping when
    // unbiased HDR output is required (ML training data, light-transport
    // validation). Primary-ray emissive hits are never clamped.
    float fireflyCap_ = 8.0f;
    int aovMode_ = 0;  // 0=off, 1=depth, 2=normals, 3=albedo, 4=instanceId, 5=roughness, 6=adaptiveBounce

    // Dynamic capacity tracking — buffers grow as scenes demand more
    int triCapacity_  = INIT_TRI_CAP;
    int matCapacity_  = INIT_MAT_CAP;
    int meshCapacity_ = INIT_MESH_CAP;
    int bvhCapacity_  = 2 * INIT_TRI_CAP - 1;

    // Actual scene counts (distinct from capacity which includes headroom)
    int matCount_  = 0;  // number of unique materials/meshes in use
    int meshCount_ = 0;  // number of mesh entries (instances) in use

    // Overlay BVH state — append-only fast path
    int  bvhRootIdx_              = 0;    // 0 = normal root; >0 = combined overlay root
    bool pendingConsolidation_    = false;// full rebuild scheduled after append
    int  stableFramesSinceAppend_ = 0;   // frames since last append without topology change
    float pixelScale_ = 1.0f;
    int fullWidth_ = 0;   // unscaled window size
    int fullHeight_ = 0;

    // Previous camera vectors for accumulation + TAAU reprojection
    float prevCamOri_[3] = {0.f, 0.f, 0.f};
    float prevCamFwd_[3] = {0.f, 0.f, -1.f};
    float prevCamRgt_[3] = {1.f, 0.f, 0.f};
    float prevCamUp_[3]  = {0.f, 1.f, 0.f};

    // GPU storage buffers
    WgpuBuffer bvhNodeBuf;
    WgpuBuffer bvhCounterBuf;
    WgpuBuffer refitMetaBuf;
    WgpuBuffer pathCounterBuf;     // atomic<u32> work-queue head for rt_main (persistent-thread path regeneration)
    WgpuBuffer primaryCounterBuf;  // separate atomic<u32> for rt_primary_main — needed because CPU-side counter writes can't interleave with GPU compute passes
    WgpuBuffer bounceCounterBuf;   // atomic<u32> for rt_bounces_main
    WgpuBuffer primaryHitBuf;      // kernel-split: RawHit per pixel written by primaryPipeline, read by rtPipeline (16 B/px)
    int primaryHitBufPixels_ = 0;  // current allocated pixel capacity (resized with viewport)
    WgpuBuffer pathStateBuf;       // bounce-split: PathStateEntry (12 vec4 = 192 B/px) carrying state from primary_shade → rt_bounces_main
    WgpuBuffer objTriBuf;
    WgpuBuffer objTriBuf2;  // overflow buffer for large scenes
    int objTriSplit_ = 0;   // split point: tris [0, split) in buf1, [split, count) in buf2
    WgpuBuffer matrixBuf;
    WgpuBuffer motionMatBuf;            // per-mesh motion matrices for RT accum reprojection
    std::vector<float> motionMatCpu;    // CPU staging: prevWorld * inverse(curWorld) per mesh
    WgpuBuffer leafIndexBuf;

    // Emissive triangle NEE
    WgpuBuffer emissiveTriBuf;
    std::vector<float> emissiveTriCpu;  // packed vec4: (triIndex, area, 0, 0) per emissive tri
    int emissiveTriCount_ = 0;
    float emissiveTotalArea_ = 0.f;
    float emissiveTotalPower_ = 0.f;
    std::unordered_set<int> emissiveMeshSet_;  // mesh indices that contribute emissive light

    // GPU uniform buffers
    WgpuBuffer vtUniBuf;
    WgpuBuffer refitUniBuf;
    WgpuBuffer rtUniformBuf;

    // Compute pipelines
    WgpuComputePipeline vtPipeline;
    WgpuComputePipeline refitPipeline;
    WgpuComputePipeline primaryPipeline;  // kernel-split step 1: BVH primary traversal → primaryHitBuf
    WgpuComputePipeline rtPipeline;       // kernel-split step 2: primaryShade + serialize to pathStateBuf
    WgpuComputePipeline bouncesPipeline;  // kernel-split step 3: runBounces + accumulation

    // Depth-fill pipeline — writes NDC depth from gBuffer primary-ray t values
    WGPURenderPipeline      depthFillPipeline_ = nullptr;
    WGPUPipelineLayout      depthFillPipeLayout_ = nullptr;
    WGPUBindGroupLayout     depthFillBGL_ = nullptr;
    WGPUShaderModule        depthFillShader_ = nullptr;
    WGPUBuffer              depthFillUniBuf_ = nullptr;
    WGPUBindGroup           depthFillBG_ = nullptr;     // cached bind group (stable until resize)
    uint32_t                depthFillSampleCount_ = 0;  // 0 = not yet built

    // Display pipeline
    OrthographicCamera displayCam;
    Scene displayScene;
    std::shared_ptr<ShaderMaterial> displayMat;

    // CPU staging buffers
    std::vector<float> triBuffer;
    std::vector<float> matBuffer;
    std::vector<float> rawObjTriBuf;
    std::vector<float> matrixCpuBuf;
    std::vector<uint32_t> bvhNodeCpuBuf;
    std::vector<int32_t> refitMetaCpuBuf;
    std::vector<uint32_t> bvhCounterZeros;

    // BVH state (wide BVH4)
    std::vector<Bvh4Node> bvhNodes;
    std::vector<int> bvhIndices;
    std::vector<int> leafIndices;

    // Async scene build result — CPU work done on background thread
    struct AsyncBuildResult {
        std::vector<unsigned char> atlasData;
        int atlasLayers = 0;
        int atlasCols = ATLAS_WIDTH / DEFAULT_TILE_SIZE;
        int tileSize = DEFAULT_TILE_SIZE;
        std::unordered_map<Texture*, int> texSlotMap;
        std::vector<float> triBuffer;
        std::vector<float> matBuffer;
        std::vector<float> rawObjTriBuf;
        std::vector<float> matrixCpuBuf;
        std::vector<Bvh4Node> bvhNodes;
        std::vector<int> bvhIndices;
        std::vector<int> leafIndices;
        std::vector<uint32_t> bvhNodeCpuBuf;
        std::vector<int32_t> refitMetaCpuBuf;
        std::vector<float> emissiveTriCpu;
        std::unordered_set<int> emissiveMeshSet;  // mesh indices with emissive contribution
        int triCount = 0;
        int numBvhNodes = 0;
        int emissiveTriCount = 0;
        float emissiveTotalArea = 0.f;
        float emissiveTotalPower = 0.f;
        std::vector<Mesh*> meshes;       // unique meshes (for atlas)
        std::vector<RtMeshEntry> entries; // expanded instances
        // Capacities used for buffer sizing (next-power-of-2)
        int triCapacity = 0;
        int matCapacity = 0;
        int meshCapacity = 0;
        int bvhCapacity = 0;
        int objTriSplit = 0;  // split point for two-buffer objTri scheme
        int matCount = 0;     // actual number of unique materials (matCapacity ≥ matCount)
        int meshCount = 0;    // actual number of mesh entries (meshCapacity ≥ meshCount)
    };
#ifndef __EMSCRIPTEN__
    // Async env CDF build
    std::future<EnvCdfResult> asyncEnvCdf_;
    bool envCdfPending_ = false;
#endif
    bool shaderHasEnvCdf_ = false;  // tracks which shader variant is active

    // Shader compilation wait tracking
    uint32_t shaderWaitFrames_ = 0;
    // True from when RT async build starts until the very first dispatch completes.
    // Ensures the VT pass (objTriBuf → triTex) runs on first dispatch even if no mesh moved.
    bool firstDispatchPending_ = false;

    // Frame state
    std::unordered_map<Texture*, int> texSlotMap;
    std::vector<Mesh*> prevMeshes;
    int prevEntryCount_ = 0;
    std::vector<Matrix4> prevEntryMatrices;
    int triCount_ = 0;
    int numBvhNodes_ = 0;
    uint32_t vtDispatchX_ = 1, vtDispatchY_ = 1;
    uint32_t rfDispatchX_ = 1, rfDispatchY_ = 1;
    float frameCount_ = 0.f;
    uint32_t globalFrameCounter_ = 0;
    int prevNLights_ = -1;
    bool foveatedEnabled_ = false;
    int foveatedConvergeFrames_ = 4;
    Vector3 prevCamPos_;
    Vector3 prevCamDir_;
    int overlayLayer_ = -1;  // -1 = disabled; objects on this layer bypass path tracing and go to raster overlay
    bool overlayFoundLastFrame_ = true;   // cached: did last frame's traversal find any overlay objects? (true = scan on first frame)
    int camMovingFrames_ = 0;             // consecutive frames camera was moving (for foveated flush cooldown)

    int width_, height_;

    int maxTriCap() const {
        WGPULimits limits{};
        wgpuDeviceGetLimits(device, &limits);
        return static_cast<int>(std::min(
            limits.maxStorageBufferBindingSize / BYTES_PER_TRI,
            uint64_t(INT_MAX)));
    }

    // Pack the stored previous-frame camera vectors into any UBO struct that has
    // float[4] prevCamOri/Fwd/Rgt/Up fields (RT and upscale UBOs share this layout).
    void fillPrevCamUbo(float* ori4, float* fwd4, float* rgt4, float* up4) const {
        ori4[0] = prevCamOri_[0]; ori4[1] = prevCamOri_[1]; ori4[2] = prevCamOri_[2]; ori4[3] = 0.f;
        fwd4[0] = prevCamFwd_[0]; fwd4[1] = prevCamFwd_[1]; fwd4[2] = prevCamFwd_[2]; fwd4[3] = 0.f;
        rgt4[0] = prevCamRgt_[0]; rgt4[1] = prevCamRgt_[1]; rgt4[2] = prevCamRgt_[2]; rgt4[3] = 0.f;
        up4[0]  = prevCamUp_[0];  up4[1]  = prevCamUp_[1];  up4[2]  = prevCamUp_[2];  up4[3]  = 0.f;
    }

    Impl(WgpuRenderer& r, int w, int h)
        : renderer(r),
          device(static_cast<WGPUDevice>(r.nativeDevice())),
          queue(static_cast<WGPUQueue>(r.nativeQueue())),
          // Geometry textures (small placeholders — grown dynamically on first build)
          triTex(r, TEX_PAGE_WIDTH, TRI_TEX_HEIGHT * triTexPages(INIT_TRI_CAP),
                 WgpuTexture::Format::RGBA32Float,
                 WgpuTexture::Storage | WgpuTexture::TextureBinding),
          matTex(r, INIT_MAT_CAP, MAT_TEX_HEIGHT,
                 WgpuTexture::Format::RGBA32Float,
                 WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          texAtlasTex(r, 1u, 1u,
                      WgpuTexture::Format::RGBA8Unorm,
                      WgpuTexture::Dimension::D2Array,
                      WgpuTexture::TextureBinding | WgpuTexture::CopyDst, 1u),
          envTexGpu(r, 1u, 1u, WgpuTexture::Format::RGBA8Unorm,
                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          bgTexGpu(r, 1u, 1u, WgpuTexture::Format::RGBA8Unorm,
                   WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          envCdfTex(r, 1u, 1u, WgpuTexture::Format::R32Float,
                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          envMargTex(r, 1u, 1u, WgpuTexture::Format::R32Float,
                     WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          // Albedo buffer for demodulation
          albedoTex(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    WgpuTexture::Format::RGBA16Float),
          denoisedDiff(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float),
          denoisedSpec(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float),
          zeroTex(r, 1u, 1u,
                  WgpuTexture::Format::RGBA16Float,
                  WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          upscalePipeline(r, upscaleWGSL, "upscale_main"),
          upscaleUniBuf(r, sizeof(UpscaleGpuUniforms)),
          // Denoiser pipeline (spatial à-trous only)
          atrousPipeline(r, svgfAtrousWGSL, "svgf_atrous_main"),
          atrousUniBuf(r, sizeof(AtrousGpuUniforms)),
          // Storage buffers (small placeholders — grown dynamically on first build)
          bvhNodeBuf(r, static_cast<size_t>(2 * INIT_TRI_CAP - 1) * BVH4_GPU_U32S * sizeof(uint32_t),
                     WgpuBuffer::Usage::Storage),
          bvhCounterBuf(r, static_cast<size_t>(2 * INIT_TRI_CAP - 1) * sizeof(uint32_t),
                        WgpuBuffer::Usage::Storage),
          refitMetaBuf(r, static_cast<size_t>(2 * INIT_TRI_CAP - 1) * BVH4_REFIT_INTS * sizeof(int32_t),
                       WgpuBuffer::Usage::Storage),
          pathCounterBuf(r, sizeof(uint32_t), WgpuBuffer::Usage::Storage),
          primaryCounterBuf(r, sizeof(uint32_t), WgpuBuffer::Usage::Storage),
          bounceCounterBuf(r, sizeof(uint32_t), WgpuBuffer::Usage::Storage),
          // primaryHitBuf: 16 bytes × pixel count.  Sized lazily to actual viewport
          // resolution (see recreatePerFrameBuffers).  Placeholder size here.
          primaryHitBuf(r, static_cast<size_t>(w * h) * 16u, WgpuBuffer::Usage::Storage),
          pathStateBuf(r, static_cast<size_t>(w * h) * 192u, WgpuBuffer::Usage::Storage),
          objTriBuf(r, static_cast<size_t>(INIT_TRI_CAP) * 32 * sizeof(float),
                    WgpuBuffer::Usage::Storage),
          objTriBuf2(r, 128u, WgpuBuffer::Usage::Storage),  // placeholder — grown when needed
          matrixBuf(r, static_cast<size_t>(INIT_MESH_CAP) * 32 * sizeof(float),
                    WgpuBuffer::Usage::Storage),
          motionMatBuf(r, static_cast<size_t>(INIT_MESH_CAP) * 16 * sizeof(float),
                       WgpuBuffer::Usage::Storage),
          leafIndexBuf(r, static_cast<size_t>(INIT_TRI_CAP) * sizeof(int),
                       WgpuBuffer::Usage::Storage),
          emissiveTriBuf(r, static_cast<size_t>(INIT_TRI_CAP) * 4 * sizeof(float),
                         WgpuBuffer::Usage::Storage),
          // Uniform buffers
          vtUniBuf(r, sizeof(VtGpuUniforms)),
          refitUniBuf(r, sizeof(RefitGpuUniforms)),
          rtUniformBuf(r, sizeof(RtGpuUniforms)),
          // Compute pipelines — primary + rt share the same shader source but
          // use different entry points.  Source rebuilt on first topology
          // commit with the env-CDF flag; here we construct with false.
          vtPipeline(r, buildVtShader(), "vt_main"),
          refitPipeline(r, buildRefitShader(), "bvh_refit"),
          primaryPipeline(r, buildRtShader(false), "rt_primary_main"),
          rtPipeline(r, buildRtShader(false), "rt_main"),
          bouncesPipeline(r, buildRtShader(false), "rt_bounces_main"),
          // Display pipeline
          displayCam(-1.f, 1.f, 1.f, -1.f, 0.1f, 10.f),
          // CPU buffers — empty; sized dynamically by async build
          width_(w), height_(h),
          fullWidth_(w), fullHeight_(h) {

        primaryHitBufPixels_ = w * h;

        // Allocate ping-pong textures (must happen in body; PingPong can't use
        // member-initializer-list syntax for sub-members)
        auto ww = static_cast<uint32_t>(w), wh = static_cast<uint32_t>(h);
        using F = WgpuTexture::Format;
        constexpr uint32_t STB = WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst;
        constexpr uint32_t STBR = STB | WgpuTexture::RenderAttachment;
        accum.a      = WgpuTexture(r, ww, wh, F::RGBA16Float);
        accum.b      = WgpuTexture(r, ww, wh, F::RGBA16Float);
        hitMesh.a    = WgpuTexture(r, ww, wh, F::RGBA16Float);
        hitMesh.b    = WgpuTexture(r, ww, wh, F::RGBA16Float);
        gBuf.a       = WgpuTexture(r, ww, wh, F::RGBA16Float, STBR);
        gBuf.b       = WgpuTexture(r, ww, wh, F::RGBA16Float, STBR);
        reservoir.a  = WgpuTexture(r, ww, wh, F::RGBA32Float);
        reservoir.b  = WgpuTexture(r, ww, wh, F::RGBA32Float);
        reservoirW.a = WgpuTexture(r, ww, wh, F::RGBA32Float);
        reservoirW.b = WgpuTexture(r, ww, wh, F::RGBA32Float);
        giRes.a      = WgpuTexture(r, ww, wh, F::RGBA32Float);
        giRes.b      = WgpuTexture(r, ww, wh, F::RGBA32Float);
        giResW.a     = WgpuTexture(r, ww, wh, F::RGBA32Float);
        giResW.b     = WgpuTexture(r, ww, wh, F::RGBA32Float);
        giResLo.a    = WgpuTexture(r, ww, wh, F::RGBA16Float);
        giResLo.b    = WgpuTexture(r, ww, wh, F::RGBA16Float);
        moments.a    = WgpuTexture(r, ww, wh, F::RGBA16Float);
        moments.b    = WgpuTexture(r, ww, wh, F::RGBA16Float);
        diffAccum.a  = WgpuTexture(r, ww, wh, F::RGBA16Float);
        diffAccum.b  = WgpuTexture(r, ww, wh, F::RGBA16Float);
        specAccum.a  = WgpuTexture(r, ww, wh, F::RGBA16Float);
        specAccum.b  = WgpuTexture(r, ww, wh, F::RGBA16Float);
        filtered.a   = WgpuTexture(r, ww, wh, F::RGBA16Float);
        filtered.b   = WgpuTexture(r, ww, wh, F::RGBA16Float);
        upscale.a    = WgpuTexture(r, ww, wh, F::RGBA16Float, STB);
        upscale.b    = WgpuTexture(r, ww, wh, F::RGBA16Float, STB);
        // read/write pointers default to &a / &b — no explicit init needed

        // Wire compute pipeline bindings
        vtPipeline.setStorageBufferRead(0, objTriBuf);
        vtPipeline.setStorageBufferRead(1, matrixBuf);
        vtPipeline.setStorageTexture(2, triTex);
        vtPipeline.setUniformBuffer(3, vtUniBuf);
        vtPipeline.setStorageBufferRead(4, objTriBuf2);

        refitPipeline.setTexture(0, triTex);
        refitPipeline.setStorageBuffer(1, bvhNodeBuf);
        refitPipeline.setStorageBuffer(2, bvhCounterBuf);
        refitPipeline.setStorageBufferRead(3, leafIndexBuf);
        refitPipeline.setUniformBuffer(4, refitUniBuf);
        refitPipeline.setStorageBufferRead(5, refitMetaBuf);

        // RT pipelines — set ALL bindings upfront (per-frame ones get overwritten)
        rtPipeline.setUniformBuffer(0, rtUniformBuf);
        rtPipeline.setTexture(1, *accum.read);
        rtPipeline.setStorageTexture(2, *accum.write);
        rtPipeline.setStorageBufferRead(3, bvhNodeBuf);
        rtPipeline.setTexture(4, matTex);
        rtPipeline.setTexture(5, triTex);
        rtPipeline.setTexture(6, texAtlasTex);
        rtPipeline.setTexture(7, *hitMesh.read);
        rtPipeline.setStorageTexture(8, *hitMesh.write);
        rtPipeline.setTexture(9, envTexGpu);
        rtPipeline.setStorageTexture(10, *gBuf.read);
        rtPipeline.setStorageBufferRead(11, emissiveTriBuf);
        rtPipeline.setTexture(12, envCdfTex);
        rtPipeline.setTexture(13, envMargTex);
        rtPipeline.setStorageTexture(14, albedoTex);
        rtPipeline.setTexture(15, *gBuf.write);
        rtPipeline.setTexture(16, bgTexGpu);
        rtPipeline.setTexture(17, *reservoir.read);
        rtPipeline.setStorageTexture(18, *reservoir.write);
        rtPipeline.setTexture(19, *reservoirW.read);
        rtPipeline.setStorageTexture(20, *reservoirW.write);
        rtPipeline.setTexture(21, *moments.read);
        rtPipeline.setStorageTexture(22, *moments.write);
        rtPipeline.setTexture(23, *diffAccum.read);
        rtPipeline.setStorageTexture(24, *diffAccum.write);
        rtPipeline.setTexture(25, *specAccum.read);
        rtPipeline.setStorageTexture(26, *specAccum.write);
        rtPipeline.setStorageBufferRead(27, motionMatBuf);
        rtPipeline.setTexture(28, *giRes.read);
        rtPipeline.setStorageTexture(29, *giRes.write);
        rtPipeline.setTexture(30, *giResW.read);
        rtPipeline.setStorageTexture(31, *giResW.write);
        rtPipeline.setTexture(32, *giResLo.read);
        rtPipeline.setStorageTexture(33, *giResLo.write);
        rtPipeline.setStorageBuffer(34, pathCounterBuf);
        rtPipeline.setStorageBuffer(35, primaryHitBuf);  // kernel-split: read primary hits written by primaryPipeline
        rtPipeline.setStorageBuffer(36, primaryCounterBuf);
        rtPipeline.setStorageBuffer(37, pathStateBuf);   // bounce-split: primary_shade writes; rt_bounces_main reads

        // Primary-hit kernel (kernel split step 1).  Same shader source, different
        // entry point — so the full bind group layout matches rtPipeline.  Most
        // bindings are unused by rt_primary_main but WebGPU still requires them
        // bound because they're declared in the module.
        primaryPipeline.setUniformBuffer(0, rtUniformBuf);
        primaryPipeline.setTexture(1, *accum.read);
        primaryPipeline.setStorageTexture(2, *accum.write);
        primaryPipeline.setStorageBufferRead(3, bvhNodeBuf);
        primaryPipeline.setTexture(4, matTex);
        primaryPipeline.setTexture(5, triTex);
        primaryPipeline.setTexture(6, texAtlasTex);
        primaryPipeline.setTexture(7, *hitMesh.read);
        primaryPipeline.setStorageTexture(8, *hitMesh.write);
        primaryPipeline.setTexture(9, envTexGpu);
        primaryPipeline.setStorageTexture(10, *gBuf.read);
        primaryPipeline.setStorageBufferRead(11, emissiveTriBuf);
        primaryPipeline.setTexture(12, envCdfTex);
        primaryPipeline.setTexture(13, envMargTex);
        primaryPipeline.setStorageTexture(14, albedoTex);
        primaryPipeline.setTexture(15, *gBuf.write);
        primaryPipeline.setTexture(16, bgTexGpu);
        primaryPipeline.setTexture(17, *reservoir.read);
        primaryPipeline.setStorageTexture(18, *reservoir.write);
        primaryPipeline.setTexture(19, *reservoirW.read);
        primaryPipeline.setStorageTexture(20, *reservoirW.write);
        primaryPipeline.setTexture(21, *moments.read);
        primaryPipeline.setStorageTexture(22, *moments.write);
        primaryPipeline.setTexture(23, *diffAccum.read);
        primaryPipeline.setStorageTexture(24, *diffAccum.write);
        primaryPipeline.setTexture(25, *specAccum.read);
        primaryPipeline.setStorageTexture(26, *specAccum.write);
        primaryPipeline.setStorageBufferRead(27, motionMatBuf);
        primaryPipeline.setTexture(28, *giRes.read);
        primaryPipeline.setStorageTexture(29, *giRes.write);
        primaryPipeline.setTexture(30, *giResW.read);
        primaryPipeline.setStorageTexture(31, *giResW.write);
        primaryPipeline.setTexture(32, *giResLo.read);
        primaryPipeline.setStorageTexture(33, *giResLo.write);
        primaryPipeline.setStorageBuffer(34, pathCounterBuf);
        primaryPipeline.setStorageBuffer(35, primaryHitBuf);
        primaryPipeline.setStorageBuffer(36, primaryCounterBuf);
        primaryPipeline.setStorageBuffer(37, pathStateBuf);

        // Bounces kernel (bounce-split step 3).  Full mirror of rtPipeline's
        // bindings because runBounces + accumulation touches the same resources.
        // Adds binding 38 (bounceCounter) — not referenced by the other entry
        // points, so bound only here.
        bouncesPipeline.setUniformBuffer(0, rtUniformBuf);
        bouncesPipeline.setTexture(1, *accum.read);
        bouncesPipeline.setStorageTexture(2, *accum.write);
        bouncesPipeline.setStorageBufferRead(3, bvhNodeBuf);
        bouncesPipeline.setTexture(4, matTex);
        bouncesPipeline.setTexture(5, triTex);
        bouncesPipeline.setTexture(6, texAtlasTex);
        bouncesPipeline.setTexture(7, *hitMesh.read);
        bouncesPipeline.setStorageTexture(8, *hitMesh.write);
        bouncesPipeline.setTexture(9, envTexGpu);
        bouncesPipeline.setStorageTexture(10, *gBuf.read);
        bouncesPipeline.setStorageBufferRead(11, emissiveTriBuf);
        bouncesPipeline.setTexture(12, envCdfTex);
        bouncesPipeline.setTexture(13, envMargTex);
        bouncesPipeline.setStorageTexture(14, albedoTex);
        bouncesPipeline.setTexture(15, *gBuf.write);
        bouncesPipeline.setTexture(16, bgTexGpu);
        bouncesPipeline.setTexture(17, *reservoir.read);
        bouncesPipeline.setStorageTexture(18, *reservoir.write);
        bouncesPipeline.setTexture(19, *reservoirW.read);
        bouncesPipeline.setStorageTexture(20, *reservoirW.write);
        bouncesPipeline.setTexture(21, *moments.read);
        bouncesPipeline.setStorageTexture(22, *moments.write);
        bouncesPipeline.setTexture(23, *diffAccum.read);
        bouncesPipeline.setStorageTexture(24, *diffAccum.write);
        bouncesPipeline.setTexture(25, *specAccum.read);
        bouncesPipeline.setStorageTexture(26, *specAccum.write);
        bouncesPipeline.setStorageBufferRead(27, motionMatBuf);
        bouncesPipeline.setTexture(28, *giRes.read);
        bouncesPipeline.setStorageTexture(29, *giRes.write);
        bouncesPipeline.setTexture(30, *giResW.read);
        bouncesPipeline.setStorageTexture(31, *giResW.write);
        bouncesPipeline.setTexture(32, *giResLo.read);
        bouncesPipeline.setStorageTexture(33, *giResLo.write);
        bouncesPipeline.setStorageBuffer(34, pathCounterBuf);
        bouncesPipeline.setStorageBuffer(35, primaryHitBuf);
        bouncesPipeline.setStorageBuffer(36, primaryCounterBuf);
        bouncesPipeline.setStorageBuffer(37, pathStateBuf);
        bouncesPipeline.setStorageBuffer(38, bounceCounterBuf);

        // Spatial filter — set ALL bindings upfront
        atrousPipeline.setUniformBuffer(0, atrousUniBuf);
        atrousPipeline.setTexture(1, *accum.read);
        atrousPipeline.setStorageTexture(2, *accum.write);
        atrousPipeline.setTexture(3, *gBuf.write);
        atrousPipeline.setTexture(4, albedoTex);
        atrousPipeline.setTexture(5, *hitMesh.read);
        atrousPipeline.setTexture(6, *moments.read);

        // TAAU pipeline — initial bindings (per-frame ones refreshed in render)
        upscalePipeline.setUniformBuffer(0, upscaleUniBuf);
        upscalePipeline.setTexture(1, denoisedDiff);
        upscalePipeline.setTexture(2, denoisedSpec);
        upscalePipeline.setTexture(3, *gBuf.read);
        upscalePipeline.setTexture(4, *upscale.read);
        upscalePipeline.setStorageTexture(5, *upscale.write);

        // Kick off async shader compilation for the small helper pipelines.
        // The large RT shaders are compiled after the first topology build so that
        // async uses the real (scene-sized) buffer bindings — this avoids a costly
        // double-compilation where the initial async result gets discarded because
        // layoutDirty=true (topology bindings changed) and the pipeline must be
        // rebuilt synchronously on the main thread a second time.
        vtPipeline.startAsyncBuild();
        refitPipeline.startAsyncBuild();
        atrousPipeline.startAsyncBuild();
        upscalePipeline.startAsyncBuild();
        std::cerr << "[PathTracer] Async shader compilation started for helper pipelines" << std::endl;

        // Zero-fill accumulators and SVGF textures
        {
            std::vector<float> zeros(w * h * 4, 0.f);
            accum.a.write(zeros.data(), zeros.size() * sizeof(float));
            accum.b.write(zeros.data(), zeros.size() * sizeof(float));
            gBuf.a.write(zeros.data(), zeros.size() * sizeof(float));
            gBuf.b.write(zeros.data(), zeros.size() * sizeof(float));
            moments.a.write(zeros.data(), zeros.size() * sizeof(float));
            moments.b.write(zeros.data(), zeros.size() * sizeof(float));
        }
        // Init albedo to white (no demodulation effect until first frame writes real values)
        {
            std::vector<float> ones(w * h * 4, 1.f);
            albedoTex.write(ones.data(), ones.size() * sizeof(float));
        }
        // Fill hitMesh textures with sentinel 128.0f (= "no hit")
        {
            std::vector<float> hitSentinel(w * h * 4, 128.f);
            hitMesh.a.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));
            hitMesh.b.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));
        }

        // Display quad
        displayCam.position.z = 1.f;
        displayMat = ShaderMaterial::create();
        displayMat->vertexShader = displayWGSL;
        displayMat->fragmentShader = displayWGSL;
        displayMat->customTextures["accumTex"] = accum.read;
        displayMat->customTextures["gBufTex"]  = gBuf.write;
        displayMat->customTextures["diffTex"]      = diffAccum.read;
        displayMat->customTextures["specTex"]      = specAccum.read;
        displayMat->customTextures["upscaleTex"]   = &zeroTex;
        displayScene.add(Mesh::create(PlaneGeometry::create(2.f, 2.f), displayMat));

        // Depth-fill: build shader, BGL, layout, and uniform buffer now.
        // The pipeline itself is created lazily on first use (needs sample count from live frame).
        {
            WGPUShaderModuleDescriptor smDesc{};
            smDesc.label = WGPUStringView{"depth_fill_sm", WGPU_STRLEN};
            WGPUShaderSourceWGSL wgslSrc{};
            wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
            wgslSrc.code = WGPUStringView{depthFillWGSL, WGPU_STRLEN};
            smDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgslSrc);
            depthFillShader_ = wgpuDeviceCreateShaderModule(device, &smDesc);

            WGPUBindGroupLayoutEntry bglEntries[2]{};
            bglEntries[0].binding = 0;
            bglEntries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
            bglEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
            bglEntries[0].buffer.minBindingSize = sizeof(DepthFillUniforms);
            bglEntries[1].binding = 1;
            bglEntries[1].visibility = WGPUShaderStage_Fragment;
            bglEntries[1].texture.sampleType = WGPUTextureSampleType_Float;  // RGBA16Float is filterable
            bglEntries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
            WGPUBindGroupLayoutDescriptor bglDesc{};
            bglDesc.label = WGPUStringView{"depth_fill_bgl", WGPU_STRLEN};
            bglDesc.entryCount = 2;
            bglDesc.entries = bglEntries;
            depthFillBGL_ = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

            WGPUPipelineLayoutDescriptor plDesc{};
            plDesc.label = WGPUStringView{"depth_fill_pl", WGPU_STRLEN};
            plDesc.bindGroupLayoutCount = 1;
            plDesc.bindGroupLayouts = &depthFillBGL_;
            depthFillPipeLayout_ = wgpuDeviceCreatePipelineLayout(device, &plDesc);

            WGPUBufferDescriptor ubDesc{};
            ubDesc.label = WGPUStringView{"depth_fill_uni", WGPU_STRLEN};
            ubDesc.size = sizeof(DepthFillUniforms);
            ubDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
            depthFillUniBuf_ = wgpuDeviceCreateBuffer(device, &ubDesc);
        }
    }

    // Build (or rebuild) the depth-fill render pipeline for a given MSAA sample count.
    void ensureDepthFillPipeline(uint32_t sampleCount) {
        if (depthFillPipeline_ && depthFillSampleCount_ == sampleCount) return;
        if (depthFillPipeline_) { wgpuRenderPipelineRelease(depthFillPipeline_); depthFillPipeline_ = nullptr; }
        WGPURenderPipelineDescriptor rpDesc{};
        rpDesc.label = WGPUStringView{"depth_fill_rp", WGPU_STRLEN};
        rpDesc.layout = depthFillPipeLayout_;
        rpDesc.vertex.module = depthFillShader_;
        rpDesc.vertex.entryPoint = WGPUStringView{"vs", WGPU_STRLEN};
        WGPUPrimitiveState prim{};
        prim.topology = WGPUPrimitiveTopology_TriangleList;
        rpDesc.primitive = prim;
        WGPUDepthStencilState ds{};
        ds.format = WGPUTextureFormat_Depth24Plus;
        ds.depthWriteEnabled = WGPUOptionalBool_True;
        ds.depthCompare = WGPUCompareFunction_Always;
        rpDesc.depthStencil = &ds;
        WGPUFragmentState frag{};
        frag.module = depthFillShader_;
        frag.entryPoint = WGPUStringView{"fs", WGPU_STRLEN};
        frag.targetCount = 0;
        rpDesc.fragment = &frag;
        rpDesc.multisample.count = sampleCount;
        rpDesc.multisample.mask  = 0xFFFFFFFF;
        depthFillPipeline_ = wgpuDeviceCreateRenderPipeline(device, &rpDesc);
        depthFillSampleCount_ = sampleCount;
    }

    void recreateAccumTextures(int w, int h) {
        width_ = w;
        height_ = h;
        auto uw = static_cast<uint32_t>(w);
        auto uh = static_cast<uint32_t>(h);
        auto fmt = WgpuTexture::Format::RGBA16Float;

        // Resize primary-hit buffer (16 bytes per pixel) and path-state buffer
        // (192 bytes per pixel).  Both scale linearly with viewport pixel count.
        const int newPixels = w * h;
        if (newPixels != primaryHitBufPixels_) {
            primaryHitBufPixels_ = newPixels;
            primaryHitBuf = WgpuBuffer(renderer, static_cast<size_t>(newPixels) * 16u,
                                        WgpuBuffer::Usage::Storage);
            pathStateBuf = WgpuBuffer(renderer, static_cast<size_t>(newPixels) * 192u,
                                       WgpuBuffer::Usage::Storage);
            // Re-bind on all three pipelines — buffer object identities changed.
            primaryPipeline.setStorageBuffer(35, primaryHitBuf);
            rtPipeline.setStorageBuffer(35, primaryHitBuf);
            bouncesPipeline.setStorageBuffer(35, primaryHitBuf);
            primaryPipeline.setStorageBuffer(37, pathStateBuf);
            rtPipeline.setStorageBuffer(37, pathStateBuf);
            bouncesPipeline.setStorageBuffer(37, pathStateBuf);
        }

        accum.a = WgpuTexture(renderer, uw, uh, fmt);
        accum.b = WgpuTexture(renderer, uw, uh, fmt);
        accum.read = &accum.a;
        accum.write = &accum.b;

        hitMesh.a = WgpuTexture(renderer, uw, uh, fmt);
        hitMesh.b = WgpuTexture(renderer, uw, uh, fmt);
        hitMesh.read  = &hitMesh.a;
        hitMesh.write = &hitMesh.b;

        const uint32_t gBufUsage = WgpuTexture::Storage | WgpuTexture::TextureBinding
                                 | WgpuTexture::CopyDst | WgpuTexture::RenderAttachment;
        gBuf.a = WgpuTexture(renderer, uw, uh, fmt, gBufUsage);
        gBuf.b = WgpuTexture(renderer, uw, uh, fmt, gBufUsage);
        gBuf.read  = &gBuf.a;
        gBuf.write = &gBuf.b;

        auto fmt32 = WgpuTexture::Format::RGBA32Float;
        reservoir.a  = WgpuTexture(renderer, uw, uh, fmt32);
        reservoir.b  = WgpuTexture(renderer, uw, uh, fmt32);
        reservoirW.a = WgpuTexture(renderer, uw, uh, fmt32);
        reservoirW.b = WgpuTexture(renderer, uw, uh, fmt32);
        reservoir.read   = &reservoir.a;
        reservoir.write  = &reservoir.b;
        reservoirW.read  = &reservoirW.a;
        reservoirW.write = &reservoirW.b;

        giRes.a  = WgpuTexture(renderer, uw, uh, fmt32);
        giRes.b  = WgpuTexture(renderer, uw, uh, fmt32);
        giResW.a = WgpuTexture(renderer, uw, uh, fmt32);
        giResW.b = WgpuTexture(renderer, uw, uh, fmt32);
        giResLo.a = WgpuTexture(renderer, uw, uh, fmt);
        giResLo.b = WgpuTexture(renderer, uw, uh, fmt);
        giRes.read   = &giRes.a;
        giRes.write  = &giRes.b;
        giResW.read  = &giResW.a;
        giResW.write = &giResW.b;
        giResLo.read  = &giResLo.a;
        giResLo.write = &giResLo.b;

        albedoTex = WgpuTexture(renderer, uw, uh, fmt);

        moments.a = WgpuTexture(renderer, uw, uh, fmt);
        moments.b = WgpuTexture(renderer, uw, uh, fmt);
        moments.read  = &moments.a;
        moments.write = &moments.b;

        diffAccum.a = WgpuTexture(renderer, uw, uh, fmt);
        diffAccum.b = WgpuTexture(renderer, uw, uh, fmt);
        diffAccum.read  = &diffAccum.a;
        diffAccum.write = &diffAccum.b;
        specAccum.a = WgpuTexture(renderer, uw, uh, fmt);
        specAccum.b = WgpuTexture(renderer, uw, uh, fmt);
        specAccum.read  = &specAccum.a;
        specAccum.write = &specAccum.b;

        filtered.a = WgpuTexture(renderer, uw, uh, fmt);
        filtered.b = WgpuTexture(renderer, uw, uh, fmt);
        denoisedDiff = WgpuTexture(renderer, uw, uh, fmt);
        denoisedSpec = WgpuTexture(renderer, uw, uh, fmt);

        std::vector<float> zeros(w * h * 4, 0.f);
        accum.a.write(zeros.data(), zeros.size() * sizeof(float));
        accum.b.write(zeros.data(), zeros.size() * sizeof(float));
        gBuf.a.write(zeros.data(), zeros.size() * sizeof(float));
        gBuf.b.write(zeros.data(), zeros.size() * sizeof(float));
        moments.a.write(zeros.data(), zeros.size() * sizeof(float));
        moments.b.write(zeros.data(), zeros.size() * sizeof(float));
        diffAccum.a.write(zeros.data(), zeros.size() * sizeof(float));
        diffAccum.b.write(zeros.data(), zeros.size() * sizeof(float));
        specAccum.a.write(zeros.data(), zeros.size() * sizeof(float));
        specAccum.b.write(zeros.data(), zeros.size() * sizeof(float));
        denoisedDiff.write(zeros.data(), zeros.size() * sizeof(float));
        denoisedSpec.write(zeros.data(), zeros.size() * sizeof(float));
        filtered.a.write(zeros.data(), zeros.size() * sizeof(float));
        filtered.b.write(zeros.data(), zeros.size() * sizeof(float));
        std::vector<float> hitSentinel(w * h * 4, 128.f);
        hitMesh.a.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));
        hitMesh.b.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));

        rtPipeline.setStorageTexture(10, *gBuf.read);
        rtPipeline.setTexture(15, *gBuf.write);
        rtPipeline.setStorageTexture(14, albedoTex);
        rtPipeline.setTexture(17, *reservoir.read);
        rtPipeline.setStorageTexture(18, *reservoir.write);
        rtPipeline.setTexture(19, *reservoirW.read);
        rtPipeline.setStorageTexture(20, *reservoirW.write);
        rtPipeline.setTexture(21, *moments.read);
        rtPipeline.setStorageTexture(22, *moments.write);
        rtPipeline.setTexture(23, *diffAccum.read);
        rtPipeline.setStorageTexture(24, *diffAccum.write);
        rtPipeline.setTexture(25, *specAccum.read);
        rtPipeline.setStorageTexture(26, *specAccum.write);
        rtPipeline.setStorageBufferRead(27, motionMatBuf);
        rtPipeline.setTexture(28, *giRes.read);
        rtPipeline.setStorageTexture(29, *giRes.write);
        rtPipeline.setTexture(30, *giResW.read);
        rtPipeline.setStorageTexture(31, *giResW.write);
        rtPipeline.setTexture(32, *giResLo.read);
        rtPipeline.setStorageTexture(33, *giResLo.write);

        // Mirror texture re-bindings on primaryPipeline so its bind-group layout
        // picks up the new texture object identities.  Primary only consumes a
        // handful of these at runtime but WebGPU requires them all bound.
        primaryPipeline.setTexture(1, *accum.read);
        primaryPipeline.setStorageTexture(2, *accum.write);
        primaryPipeline.setTexture(7, *hitMesh.read);
        primaryPipeline.setStorageTexture(8, *hitMesh.write);
        primaryPipeline.setStorageTexture(10, *gBuf.read);
        primaryPipeline.setTexture(15, *gBuf.write);
        primaryPipeline.setStorageTexture(14, albedoTex);
        primaryPipeline.setTexture(17, *reservoir.read);
        primaryPipeline.setStorageTexture(18, *reservoir.write);
        primaryPipeline.setTexture(19, *reservoirW.read);
        primaryPipeline.setStorageTexture(20, *reservoirW.write);
        primaryPipeline.setTexture(21, *moments.read);
        primaryPipeline.setStorageTexture(22, *moments.write);
        primaryPipeline.setTexture(23, *diffAccum.read);
        primaryPipeline.setStorageTexture(24, *diffAccum.write);
        primaryPipeline.setTexture(25, *specAccum.read);
        primaryPipeline.setStorageTexture(26, *specAccum.write);
        primaryPipeline.setStorageBufferRead(27, motionMatBuf);
        primaryPipeline.setTexture(28, *giRes.read);
        primaryPipeline.setStorageTexture(29, *giRes.write);
        primaryPipeline.setTexture(30, *giResW.read);
        primaryPipeline.setStorageTexture(31, *giResW.write);
        primaryPipeline.setTexture(32, *giResLo.read);
        primaryPipeline.setStorageTexture(33, *giResLo.write);

        // Mirror texture re-bindings on bouncesPipeline.  Bounces kernel
        // consumes almost the same resources as rtPipeline, so mirror the full set.
        bouncesPipeline.setTexture(1, *accum.read);
        bouncesPipeline.setStorageTexture(2, *accum.write);
        bouncesPipeline.setTexture(7, *hitMesh.read);
        bouncesPipeline.setStorageTexture(8, *hitMesh.write);
        bouncesPipeline.setStorageTexture(10, *gBuf.read);
        bouncesPipeline.setTexture(15, *gBuf.write);
        bouncesPipeline.setStorageTexture(14, albedoTex);
        bouncesPipeline.setTexture(17, *reservoir.read);
        bouncesPipeline.setStorageTexture(18, *reservoir.write);
        bouncesPipeline.setTexture(19, *reservoirW.read);
        bouncesPipeline.setStorageTexture(20, *reservoirW.write);
        bouncesPipeline.setTexture(21, *moments.read);
        bouncesPipeline.setStorageTexture(22, *moments.write);
        bouncesPipeline.setTexture(23, *diffAccum.read);
        bouncesPipeline.setStorageTexture(24, *diffAccum.write);
        bouncesPipeline.setTexture(25, *specAccum.read);
        bouncesPipeline.setStorageTexture(26, *specAccum.write);
        bouncesPipeline.setStorageBufferRead(27, motionMatBuf);
        bouncesPipeline.setTexture(28, *giRes.read);
        bouncesPipeline.setStorageTexture(29, *giRes.write);
        bouncesPipeline.setTexture(30, *giResW.read);
        bouncesPipeline.setStorageTexture(31, *giResW.write);
        bouncesPipeline.setTexture(32, *giResLo.read);
        bouncesPipeline.setStorageTexture(33, *giResLo.write);

        atrousPipeline.setTexture(4, albedoTex);
        atrousPipeline.setTexture(6, *moments.read);

        // Depth fill bind group references gBuf.write — invalidate on resize
        if (depthFillBG_) { wgpuBindGroupRelease(depthFillBG_); depthFillBG_ = nullptr; }

        frameCount_ = 0.f;
    }

    void recreateUpscaleTextures(int fw, int fh) {
        auto ufw = static_cast<uint32_t>(fw);
        auto ufh = static_cast<uint32_t>(fh);
        const uint32_t usage = WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst;
        auto fmt = WgpuTexture::Format::RGBA16Float;
        upscale.a = WgpuTexture(renderer, ufw, ufh, fmt, usage);
        upscale.b = WgpuTexture(renderer, ufw, ufh, fmt, usage);
        upscale.read  = &upscale.a;
        upscale.write = &upscale.b;
        std::vector<float> zeros(fw * fh * 4, 0.f);
        upscale.a.write(zeros.data(), zeros.size() * sizeof(float));
        upscale.b.write(zeros.data(), zeros.size() * sizeof(float));
    }

    void resetUpscaleHistory() {
        // Zero-fill histLen (w=0 → first frame takes 100% current, no stale blending)
        const int fw = fullWidth_;
        const int fh = fullHeight_;
        if (fw <= 0 || fh <= 0) return;
        std::vector<float> zeros(fw * fh * 4, 0.f);
        upscale.a.write(zeros.data(), zeros.size() * sizeof(float));
        upscale.b.write(zeros.data(), zeros.size() * sizeof(float));
    }
};

// ---------------------------------------------------------------------------
// WgpuPathTracer public API
// ---------------------------------------------------------------------------

WgpuPathTracer::WgpuPathTracer(WgpuRenderer& renderer, std::pair<int, int> size)
    : pimpl_(std::make_unique<Impl>(renderer, size.first, size.second)) {}

WgpuPathTracer::~WgpuPathTracer() = default;

void WgpuPathTracer::render(Object3D& scene, Camera& camera) {
    auto& d = *pimpl_;

    // Derive camera vectors from camera world matrix (controller-agnostic)
    const Vector3& camPos = camera.position;
    Vector3 fwd;
    camera.getWorldDirection(fwd);
    Vector3 rgt = Vector3(fwd).cross(Vector3(0.f, 1.f, 0.f)).normalize();
    Vector3 up = Vector3(rgt).cross(fwd);

    const bool camMoved =
            (camPos - d.prevCamPos_).length() > 1e-4f ||
            (fwd - d.prevCamDir_).length() > 1e-4f;
    if (camMoved) {
        d.prevCamPos_ = camPos;
        d.prevCamDir_ = fwd;
        // Camera motion is handled via per-pixel reprojection in the RT accumulation
        // pass (triggered by camMoved flag in params.w). No FC reset needed.
    }

    // Foveated flush: when camera transitions from sustained motion → stopped,
    // follower pixels hold block-copied values. Reset frameCount so the EMA
    // restarts fresh. Require several consecutive moving frames before arming
    // the reset — prevents flicker during slow camera movement where camMoved
    // toggles on/off around the threshold every few frames.
    if (camMoved) {
        d.camMovingFrames_++;
    } else {
        if (d.foveatedEnabled_ && d.camMovingFrames_ >= 3) {
            d.frameCount_ = 0.f;
        }
        d.camMovingFrames_ = 0;
    }

    // Collect RT-eligible meshes via full traverse (NOT traverseVisible).
    // We deliberately ignore Object3D::visible here so the rtMeshes list — and the
    // BVH that gets built from it — stays stable across .visible toggles.  Hidden
    // meshes are handled per-frame by overriding their world matrix to translate
    // them far out of the scene, so the refit pass culls them naturally without
    // a topology rebuild or fast-append.  Effective visibility (own + ancestors)
    // is determined in a parallel traverseVisible pass below.
    std::vector<Mesh*> rtMeshes;
    auto isRtEligibleMaterial = [](Mesh& m) {
        auto* mat = m.material().get();
        if (!mat->visible) return false;
        auto* mww = mat->as<MaterialWithWireframe>();
        if (mww && mww->wireframe) return false;
        if (mat->is<LineBasicMaterial>()) return false;
        // Transparent meshes with no texture and no transmission are raster-only
        // overlay effects (e.g. separate clearcoat geometry layers). They have no
        // physical meaning in path tracing and render as opaque shells that occlude
        // everything beneath them.
        if (mat->transparent) {
            auto* mwm = dynamic_cast<MaterialWithMap*>(mat);
            auto* mwt = dynamic_cast<MaterialWithTransmission*>(mat);
            const bool hasMap = mwm && mwm->map;
            const bool hasTransmission = mwt && mwt->transmission > 0.f;
            const bool hasBlend = mat->opacity < 0.999f;
            if (!hasMap && !hasTransmission && !hasBlend) return false;
        }
        return true;
    };
    scene.traverse([&](Object3D& o) {
        auto* m_ptr = dynamic_cast<Mesh*>(&o);
        if (!m_ptr) return;
        auto& m = *m_ptr;
        if (d.overlayLayer_ >= 0 && m.layers.isEnabled(static_cast<unsigned>(d.overlayLayer_))) return;
        if (!isRtEligibleMaterial(m)) return;
        rtMeshes.push_back(&m);
    });
    // Build the set of meshes that are effectively visible (mesh.visible AND all
    // ancestors visible).  traverseVisible handles ancestor propagation for free.
    std::unordered_set<Mesh*> visibleMeshSet;
    scene.traverseVisible([&](Object3D& o) {
        if (auto* mp = dynamic_cast<Mesh*>(&o)) visibleMeshSet.insert(mp);
    });
    // Hide-by-translate: replace world matrices of effectively-hidden entries
    // with a far-away pure translation.  Refit then places those triangles' AABB
    // at ~(1e18,1e18,1e18) where no real ray can reach them.  Visibility flips
    // become matrix changes which the per-frame diff loop catches and refits —
    // no BVH topology rebuild / fast-append required.
    auto applyHideOverride = [&](std::vector<RtMeshEntry>& entries) {
        Matrix4 hideMat;
        hideMat.makeTranslation(1e18f, 1e18f, 1e18f);
        for (auto& e : entries) {
            if (!visibleMeshSet.count(e.mesh)) e.worldMatrix = hideMat;
        }
    };
    std::vector<PointLight*> pointLights;
    scene.traverseType<PointLight>([&](PointLight& l) { if (l.visible) pointLights.push_back(&l); });
    std::vector<DirectionalLight*> dirLights;
    scene.traverseType<DirectionalLight>([&](DirectionalLight& l) { if (l.visible) dirLights.push_back(&l); });
    std::vector<SpotLight*> spotLights;
    scene.traverseType<SpotLight>([&](SpotLight& l) { if (l.visible) spotLights.push_back(&l); });

    // Compute entry count for instanced mesh awareness
    int totalEntryCount = 0;
    for (auto* m : rtMeshes) {
        auto* inst = dynamic_cast<InstancedMesh*>(m);
        totalEntryCount += (inst && inst->count() > 0) ? static_cast<int>(inst->count()) : 1;
    }

    // Detect topology change (mesh list or instance configuration changed)
    const bool topoChanged = (rtMeshes != d.prevMeshes) || (totalEntryCount != d.prevEntryCount_);

    // --- Append-only fast path ---
    // If only new meshes were added (none removed), skip the full BVH rebuild and instead
    // build a small overlay BVH for just the new triangles.  This avoids O(N_total) rebuild
    // and frameCount_ reset — new objects converge via normal temporal accumulation.
    // Overlay design: new tris sit at [oldTriCount..] in objTriBuf.  A small sub-BVH is built
    // for them.  A new combined-root node references both the old root (always node 0) and the
    // overlay root.  The traversal uniform bvhAux.x is updated to point at the combined root.
    bool didFastAppend = false;
    if (topoChanged && !d.prevMeshes.empty() && d.triCount_ > 0 && d.bvhRootIdx_ == 0
        && !d.matBuffer.empty() && !d.matrixCpuBuf.empty()) {  // mat/matrix buffers kept alive
        // Check all previous meshes are still present (pure append, no removals)
        std::unordered_set<Mesh*> prevSet(d.prevMeshes.begin(), d.prevMeshes.end());
        bool allOldPresent = (rtMeshes.size() >= d.prevMeshes.size());
        if (allOldPresent) {
            for (auto* m : d.prevMeshes) {
                if (std::find(rtMeshes.begin(), rtMeshes.end(), m) == rtMeshes.end()) {
                    allOldPresent = false; break;
                }
            }
        }

        if (allOldPresent && totalEntryCount >= d.prevEntryCount_) {
            // Identify new meshes
            std::vector<Mesh*> addedMeshes;
            for (auto* m : rtMeshes)
                if (!prevSet.count(m)) addedMeshes.push_back(m);

            // Update world matrices and expand instances before counting triangles
            for (auto* m : addedMeshes) m->updateWorldMatrix(true, true);
            auto addedEntries = expandMeshEntries(addedMeshes);
            applyHideOverride(addedEntries);

            // Count total new triangles across all expanded entries (includes instanced copies)
            int newTris = 0;
            for (auto& entry : addedEntries) {
                auto* geo = entry.mesh->geometry().get();
                auto* idx = geo->getIndex();
                auto* pos = geo->getAttribute<float>("position");
                if (!pos) continue;
                newTris += idx ? static_cast<int>(idx->count()) / 3
                               : static_cast<int>(pos->count()) / 3;
            }

            const int oldTriCount = d.triCount_;
            const int oldBvhNodes = d.numBvhNodes_;
            const int oldMatCount  = d.matCount_;
            const int oldMeshCount = d.meshCount_;
            // Worst-case BVH nodes: ~2 per leaf, max 8 tris per leaf
            const int projBvhNodes = oldBvhNodes + (newTris / MAX_LEAF_TRIS + 2) * 4;

            // trisFit: check total capacity AND that new tris fit in buffer 1
            // (fast-append uploads to objTriBuf only; two-buffer overflow not supported here)
            const bool trisFit = (oldTriCount + newTris) <= d.triCapacity_ &&
                                 (oldTriCount + newTris) <= d.objTriSplit_;
            const bool bvhFit  = projBvhNodes <= d.bvhCapacity_;
            // matsFit: at most one material slot per unique Mesh* (materials are deduplicated by Mesh*)
            const bool matsFit = (oldMatCount  + static_cast<int>(addedMeshes.size())) <= d.matCapacity_;
            // meshesFit: one matrix slot per expanded entry (one per instance for InstancedMesh)
            const bool meshesFit = (oldMeshCount + static_cast<int>(addedEntries.size())) <= d.meshCapacity_;

            if (!addedMeshes.empty() && newTris > 0 && trisFit && bvhFit && matsFit && meshesFit) {
                // Build new tris into LOCAL temp buffers using local indices (0..newTris-1).
                // These are small (proportional to newTris, not totalTris).
                const int localPages = triTexPages(newTris);
                std::vector<float> localTriBuf(
                    static_cast<size_t>(TEX_PAGE_WIDTH) * TRI_TEX_HEIGHT * localPages * 4, 0.f);
                std::vector<float> localObjBuf(static_cast<size_t>(newTris) * 32, 0.f);

                // Build geometry into local buffers.
                // matOffset and meshOffset are global — new mat/mesh rows go at correct columns
                // in d.matBuffer and d.matrixCpuBuf (which have 2× headroom and are kept alive).
                int appendedTris = buildGeometryBuffers(
                    addedEntries, d.texSlotMap,
                    localTriBuf, d.matBuffer, localObjBuf, d.matrixCpuBuf,
                    newTris, d.matCapacity_, d.meshCapacity_,
                    0 /*triOffset — local*/, oldMatCount, oldMeshCount);

                if (appendedTris == newTris) {
                    // Build overlay BVH on local buffers (leaf triStart = local 0..newTris-1)
                    std::vector<Bvh4Node> overlayNodes;
                    std::vector<int> overlayLeafIndices;
                    buildOverlayBVH(localTriBuf, localObjBuf, 0, newTris,
                                    overlayNodes, overlayLeafIndices);

                    // Offset leaf triStart values from local to global (add oldTriCount)
                    for (auto& node : overlayNodes) {
                        for (int c = 0; c < 4; c++) {
                            const int ci = node.childIdx[c];
                            if (ci >= 0 || ci == INT_MIN) continue;
                            const int raw = -ci;
                            const int localStart = (raw - 1) / MAX_LEAF_TRIS;
                            const int cnt = ((raw - 1) % MAX_LEAF_TRIS) + 1;
                            node.childIdx[c] = -(((localStart + oldTriCount) * MAX_LEAF_TRIS) + cnt);
                        }
                    }

                    const int overlayRootNodeIdx = oldBvhNodes;
                    const int combinedRootIdx    = oldBvhNodes + static_cast<int>(overlayNodes.size());

                    // Build combined root referencing old root (node 0) and overlay root
                    Bvh4Node combinedRoot{};
                    combinedRoot.parent = -1;
                    combinedRoot.childCount = 2;
                    combinedRoot.numInternalChildren = 2;
                    combinedRoot.childIdx[0] = 0;  // old BVH root
                    // Use maximally conservative (infinite) AABB for the old subtree.
                    // The saved bvhRootMinX_/Max_ are computed from the original root's child
                    // slot extents which may not tightly bound ALL geometry in that subtree
                    // (e.g. nodes added later, or float rounding). A too-tight AABB here would
                    // cause sceneAnyHit to cull shadow rays that should hit the old geometry
                    // (= light leaking through walls). The cost is one extra AABB test per ray.
                    combinedRoot.childMinX[0] = -1e30f;
                    combinedRoot.childMinY[0] = -1e30f;
                    combinedRoot.childMinZ[0] = -1e30f;
                    combinedRoot.childMaxX[0] =  1e30f;
                    combinedRoot.childMaxY[0] =  1e30f;
                    combinedRoot.childMaxZ[0] =  1e30f;
                    combinedRoot.childIdx[1] = overlayRootNodeIdx;
                    if (!overlayNodes.empty()) {
                        const auto& ov = overlayNodes[0];
                        combinedRoot.childMinX[1] = *std::min_element(ov.childMinX, ov.childMinX + 4);
                        combinedRoot.childMinY[1] = *std::min_element(ov.childMinY, ov.childMinY + 4);
                        combinedRoot.childMinZ[1] = *std::min_element(ov.childMinZ, ov.childMinZ + 4);
                        combinedRoot.childMaxX[1] = *std::max_element(ov.childMaxX, ov.childMaxX + 4);
                        combinedRoot.childMaxY[1] = *std::max_element(ov.childMaxY, ov.childMaxY + 4);
                        combinedRoot.childMaxZ[1] = *std::max_element(ov.childMaxZ, ov.childMaxZ + 4);
                    }
                    for (int c = 2; c < 4; c++) {
                        combinedRoot.childIdx[c] = INT_MIN;
                        combinedRoot.childMinX[c] = 1e30f; combinedRoot.childMinY[c] = 1e30f; combinedRoot.childMinZ[c] = 1e30f;
                        combinedRoot.childMaxX[c] = -1e30f; combinedRoot.childMaxY[c] = -1e30f; combinedRoot.childMaxZ[c] = -1e30f;
                    }

                    // Pack overlay nodes + combined root and upload at offset oldBvhNodes
                    const int packedCount = static_cast<int>(overlayNodes.size()) + 1;
                    std::vector<Bvh4Node> toPackVec(overlayNodes.begin(), overlayNodes.end());
                    toPackVec.push_back(combinedRoot);
                    std::vector<uint32_t> packedBvh(packedCount * BVH4_GPU_U32S, 0u);
                    packBvh4Buffer(toPackVec, packedBvh, packedCount);
                    d.bvhNodeBuf.write(packedBvh.data(),
                        packedCount * BVH4_GPU_U32S * sizeof(uint32_t),
                        static_cast<size_t>(oldBvhNodes) * BVH4_GPU_U32S * sizeof(uint32_t));

                    // Upload sorted local obj-space triangle data at offset in objTriBuf
                    d.objTriBuf.write(localObjBuf.data(),
                        static_cast<size_t>(newTris) * BYTES_PER_TRI,
                        static_cast<size_t>(oldTriCount) * BYTES_PER_TRI);

                    // Re-upload matBuffer (small, new material rows were appended in-place)
                    d.matTex.write(d.matBuffer.data(), d.matBuffer.size() * sizeof(float));

                    // Upload new mesh matrix rows (only the new portion)
                    const int addedMeshCount = static_cast<int>(addedEntries.size());
                    d.matrixBuf.write(
                        d.matrixCpuBuf.data() + static_cast<size_t>(oldMeshCount) * 32,
                        static_cast<size_t>(addedMeshCount) * 32 * sizeof(float),
                        static_cast<size_t>(oldMeshCount) * 32 * sizeof(float));

                    // Update leaf indices for refit (append overlay leaf node global indices)
                    for (int li : overlayLeafIndices)
                        d.leafIndices.push_back(oldBvhNodes + li);
                    d.leafIndexBuf.write(d.leafIndices.data(), d.leafIndices.size() * sizeof(int));

                    // Update Impl state
                    d.triCount_   += newTris;
                    d.matCount_   += static_cast<int>(addedMeshes.size());  // rough: unique meshes
                    d.meshCount_  += addedMeshCount;
                    d.numBvhNodes_ = combinedRootIdx + 1;
                    d.bvhRootIdx_  = combinedRootIdx;
                    d.pendingConsolidation_    = true;
                    d.stableFramesSinceAppend_ = 0;
                    d.prevMeshes = rtMeshes;
                    d.prevEntryCount_ = totalEntryCount;

                    // Force VT pass next frame — new obj-space triangles need world-space transform
                    d.firstDispatchPending_ = true;

                    didFastAppend = true;
                    std::cerr << "[PathTracer] Fast-append: +" << newTris
                              << " tris, combined root=" << combinedRootIdx << std::endl;
                }
            }
        }
    }

    // Build scene data (BVH, geometry buffers, atlas, emissives) when topology changes.
    // Native: async on background thread.  Emscripten: synchronous (no pthreads).
    bool topoJustFinished = false;
#ifdef __EMSCRIPTEN__
    if (topoChanged && !didFastAppend) {
        d.prevMeshes = rtMeshes;
        d.prevEntryCount_ = totalEntryCount;
        auto meshes = rtMeshes;

        // Expand InstancedMesh objects into individual entries
        for (auto* m : meshes) m->updateWorldMatrix(true, true);
        auto entries = expandMeshEntries(meshes);
        applyHideOverride(entries);

        Impl::AsyncBuildResult r;
        r.meshes = meshes;
        r.entries = entries;
        r.texSlotMap.clear();
        auto [atlasData, atlasLayers, atlasCols_, tileSize_] = buildAtlas(meshes, r.texSlotMap, d.textureResolution_);
        r.atlasData = std::move(atlasData);
        r.atlasLayers = atlasLayers;
        r.atlasCols = atlasCols_;
        r.tileSize = tileSize_;

        int totalTris = 0;
        for (auto& entry : entries) {
            auto* geo = entry.mesh->geometry().get();
            auto* idx = geo->getIndex();
            auto* pos = geo->getAttribute<float>("position");
            if (!pos) continue;
            totalTris += idx ? static_cast<int>(idx->count()) / 3
                             : static_cast<int>(pos->count()) / 3;
        }
        const int matCount = static_cast<int>(meshes.size());
        const int meshCount = static_cast<int>(entries.size());
        const int triCap = d.maxTriCap();
        const int maxTotalTris = triCap * 2;  // two split buffers
        const int rawTriCap = std::clamp(totalTris, 1, maxTotalTris);
        if (totalTris > maxTotalTris) {
            std::cerr << "[PathTracer] Warning: scene has " << totalTris
                      << " tris, capped to " << maxTotalTris << " (2x GPU buffer limit)\n";
        }
        // Reserve 50% headroom so append-only topology changes don't require buffer recreation
        r.triCapacity  = std::min(rawTriCap + rawTriCap / 2, maxTotalTris);
        r.objTriSplit = std::min(r.triCapacity, triCap);  // first buffer holds up to triCap
        r.matCapacity  = std::max(matCount * 2, 1);    // 2× headroom for new materials on append
        r.meshCapacity = std::max(meshCount * 2, 1);   // 2× headroom for new meshes on append

        const int pages = triTexPages(r.triCapacity);
        r.triBuffer.resize(static_cast<size_t>(TEX_PAGE_WIDTH) * TRI_TEX_HEIGHT * pages * 4, 0.f);
        r.matBuffer.resize(static_cast<size_t>(r.matCapacity) * MAT_TEX_HEIGHT * 4, 0.f);
        r.rawObjTriBuf.resize(static_cast<size_t>(r.triCapacity) * 32, 0.f);
        r.matrixCpuBuf.resize(static_cast<size_t>(r.meshCapacity) * 32, 0.f);
        r.triCount = buildGeometryBuffers(entries, r.texSlotMap, r.triBuffer, r.matBuffer,
                                           r.rawObjTriBuf, r.matrixCpuBuf,
                                           r.triCapacity, r.matCapacity, r.meshCapacity);
        r.matCount  = matCount;   // upper-bound: number of unique meshes (some may share materials)
        r.meshCount = meshCount;  // number of expanded mesh entries (instances)
        buildBVH(r.triBuffer, r.triCount, r.bvhNodes, r.bvhIndices, r.leafIndices, r.rawObjTriBuf);
        r.numBvhNodes = static_cast<int>(r.bvhNodes.size());
        // Reserve 2× BVH nodes so overlay BVH can be appended without buffer recreation
        r.bvhCapacity = std::max(r.numBvhNodes * 2, 1);
        r.bvhNodeCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * BVH4_GPU_U32S, 0u);
        packBvh4Buffer(r.bvhNodes, r.bvhNodeCpuBuf, r.bvhCapacity);
        r.refitMetaCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * BVH4_REFIT_INTS, 0);
        packRefitMetadata(r.bvhNodes, r.refitMetaCpuBuf, r.bvhCapacity);

        r.emissiveTriCount = 0;
        r.emissiveTotalArea = 0.f;
        r.emissiveTotalPower = 0.f;
        for (int ti = 0; ti < r.triCount; ti++) {
            const int matIdx = static_cast<int>(r.triBuffer[pagedIdx(ti, 0) + 3]);
            const float er = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 0];
            const float eg = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 1];
            const float eb = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 2];
            const float luminance = 0.2126f * er + 0.7152f * eg + 0.0722f * eb;
            if (luminance > 0.001f) {
                const float* v0p = r.triBuffer.data() + pagedIdx(ti, 0);
                const float* v1p = r.triBuffer.data() + pagedIdx(ti, 1);
                const float* v2p = r.triBuffer.data() + pagedIdx(ti, 2);
                Vector3 v0(v0p[0], v0p[1], v0p[2]);
                Vector3 v1(v1p[0], v1p[1], v1p[2]);
                Vector3 v2(v2p[0], v2p[1], v2p[2]);
                Vector3 cross;
                cross.crossVectors(v1 - v0, v2 - v0);
                const float area = cross.length() * 0.5f;
                if (area > 1e-8f) {
                    const float power = area * luminance;
                    r.emissiveTotalArea += area;
                    r.emissiveTotalPower += power;
                    r.emissiveTriCpu.push_back(static_cast<float>(ti));
                    r.emissiveTriCpu.push_back(area);
                    r.emissiveTriCpu.push_back(r.emissiveTotalPower);
                    r.emissiveTriCpu.push_back(power);
                    r.emissiveTriCount++;
                    // Record which mesh this emissive tri belongs to (triData row1.w = meshIdx)
                    r.emissiveMeshSet.insert(static_cast<int>(v1p[3]));
                }
            }
        }
        topoJustFinished = true;
        d.frameCount_ = 0.f;
#else
    if (topoChanged && !didFastAppend) {
        d.prevMeshes = rtMeshes;
        d.prevEntryCount_ = totalEntryCount;

        auto meshes = rtMeshes;
        for (auto* m : meshes) m->updateWorldMatrix(true, true);
        auto entries = expandMeshEntries(meshes);
        applyHideOverride(entries);

        Impl::AsyncBuildResult r;
        r.meshes = meshes;
        r.entries = entries;
        r.texSlotMap.clear();
        auto [atlasData, atlasLayers, atlasCols_, tileSize_] = buildAtlas(meshes, r.texSlotMap, d.textureResolution_);
        r.atlasData = std::move(atlasData);
        r.atlasLayers = atlasLayers;
        r.atlasCols = atlasCols_;
        r.tileSize = tileSize_;

        int totalTris = 0;
        for (auto& entry : entries) {
            auto* geo = entry.mesh->geometry().get();
            auto* idx = geo->getIndex();
            auto* pos = geo->getAttribute<float>("position");
            if (!pos) continue;
            totalTris += idx ? static_cast<int>(idx->count()) / 3
                             : static_cast<int>(pos->count()) / 3;
        }
        const int matCount = static_cast<int>(meshes.size());
        const int meshCount = static_cast<int>(entries.size());
        const int triCap = d.maxTriCap();
        const int maxTotalTris = triCap * 2;  // two split buffers
        const int rawTriCap = std::clamp(totalTris, 1, maxTotalTris);
        if (totalTris > maxTotalTris) {
            std::cerr << "[PathTracer] Warning: scene has " << totalTris
                      << " tris, capped to " << maxTotalTris << " (2x GPU buffer limit)\n";
        }
        // Reserve 50% headroom so append-only topology changes don't require buffer recreation
        r.triCapacity  = std::min(rawTriCap + rawTriCap / 2, maxTotalTris);
        r.objTriSplit = std::min(r.triCapacity, triCap);  // first buffer holds up to triCap
        r.matCapacity  = std::max(matCount * 2, 1);    // 2× headroom for new materials on append
        r.meshCapacity = std::max(meshCount * 2, 1);   // 2× headroom for new meshes on append

        const int pages = triTexPages(r.triCapacity);
        r.triBuffer.resize(static_cast<size_t>(TEX_PAGE_WIDTH) * TRI_TEX_HEIGHT * pages * 4, 0.f);
        r.matBuffer.resize(static_cast<size_t>(r.matCapacity) * MAT_TEX_HEIGHT * 4, 0.f);
        r.rawObjTriBuf.resize(static_cast<size_t>(r.triCapacity) * 32, 0.f);
        r.matrixCpuBuf.resize(static_cast<size_t>(r.meshCapacity) * 32, 0.f);
        r.triCount = buildGeometryBuffers(entries, r.texSlotMap, r.triBuffer, r.matBuffer,
                                           r.rawObjTriBuf, r.matrixCpuBuf,
                                           r.triCapacity, r.matCapacity, r.meshCapacity);
        r.matCount  = matCount;   // upper-bound: number of unique meshes (some may share materials)
        r.meshCount = meshCount;  // number of expanded mesh entries (instances)
        buildBVH(r.triBuffer, r.triCount, r.bvhNodes, r.bvhIndices, r.leafIndices, r.rawObjTriBuf);
        r.numBvhNodes = static_cast<int>(r.bvhNodes.size());
        // Reserve 2× BVH nodes so overlay BVH can be appended without buffer recreation
        r.bvhCapacity = std::max(r.numBvhNodes * 2, 1);
        r.bvhNodeCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * BVH4_GPU_U32S, 0u);
        packBvh4Buffer(r.bvhNodes, r.bvhNodeCpuBuf, r.bvhCapacity);
        r.refitMetaCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * BVH4_REFIT_INTS, 0);
        packRefitMetadata(r.bvhNodes, r.refitMetaCpuBuf, r.bvhCapacity);

        r.emissiveTriCount = 0;
        r.emissiveTotalArea = 0.f;
        r.emissiveTotalPower = 0.f;
        for (int ti = 0; ti < r.triCount; ti++) {
            const int matIdx = static_cast<int>(r.triBuffer[pagedIdx(ti, 0) + 3]);
            const float er = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 0];
            const float eg = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 1];
            const float eb = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 2];
            const float luminance = 0.2126f * er + 0.7152f * eg + 0.0722f * eb;
            if (luminance > 0.001f) {
                const float* v0p = r.triBuffer.data() + pagedIdx(ti, 0);
                const float* v1p = r.triBuffer.data() + pagedIdx(ti, 1);
                const float* v2p = r.triBuffer.data() + pagedIdx(ti, 2);
                Vector3 v0(v0p[0], v0p[1], v0p[2]);
                Vector3 v1(v1p[0], v1p[1], v1p[2]);
                Vector3 v2(v2p[0], v2p[1], v2p[2]);
                Vector3 cross;
                cross.crossVectors(v1 - v0, v2 - v0);
                const float area = cross.length() * 0.5f;
                if (area > 1e-8f) {
                    const float power = area * luminance;
                    r.emissiveTotalArea += area;
                    r.emissiveTotalPower += power;
                    r.emissiveTriCpu.push_back(static_cast<float>(ti));
                    r.emissiveTriCpu.push_back(area);
                    r.emissiveTriCpu.push_back(r.emissiveTotalPower);
                    r.emissiveTriCpu.push_back(power);
                    r.emissiveTriCount++;
                    r.emissiveMeshSet.insert(static_cast<int>(v1p[3]));
                }
            }
        }
        topoJustFinished = true;
        d.frameCount_ = 0.f;
#endif

        // Move CPU results into Impl
        d.texSlotMap = std::move(r.texSlotMap);
        d.triBuffer = std::move(r.triBuffer);
        d.matBuffer = std::move(r.matBuffer);
        d.rawObjTriBuf = std::move(r.rawObjTriBuf);
        d.matrixCpuBuf = std::move(r.matrixCpuBuf);
        d.bvhNodes = std::move(r.bvhNodes);
        d.bvhIndices = std::move(r.bvhIndices);
        d.leafIndices = std::move(r.leafIndices);
        d.bvhNodeCpuBuf = std::move(r.bvhNodeCpuBuf);
        d.refitMetaCpuBuf = std::move(r.refitMetaCpuBuf);
        d.emissiveTriCpu = std::move(r.emissiveTriCpu);
        d.emissiveMeshSet_ = std::move(r.emissiveMeshSet);
        d.triCount_ = r.triCount;
        d.objTriSplit_ = r.objTriSplit;
        d.numBvhNodes_ = r.numBvhNodes;
        d.emissiveTriCount_ = r.emissiveTriCount;
        d.matCount_  = r.matCount;
        d.meshCount_ = r.meshCount;
        d.bvhRootIdx_ = 0;          // fresh build always resets to root = node 0
        d.pendingConsolidation_ = false;
        d.stableFramesSinceAppend_ = 0;

        std::cerr << "[PathTracer] Scene: " << r.triCount << " tris, "
                  << r.numBvhNodes << " BVH nodes, "
                  << r.matCapacity << " materials, "
                  << r.meshCapacity << " meshes" << std::endl;
        d.emissiveTotalArea_ = r.emissiveTotalArea;
        d.emissiveTotalPower_ = r.emissiveTotalPower;

        // Grow GPU buffers when scene exceeds current capacity (same pattern as atlas)
        if (r.triCapacity != d.triCapacity_) {
            d.triCapacity_ = r.triCapacity;
            const int pages = triTexPages(r.triCapacity);
            d.triTex = WgpuTexture(d.renderer, TEX_PAGE_WIDTH, TRI_TEX_HEIGHT * pages,
                                    WgpuTexture::Format::RGBA32Float,
                                    WgpuTexture::Storage | WgpuTexture::TextureBinding);
            // Split objTri data across two buffers to stay within per-buffer size limits
            const size_t buf1Tris = static_cast<size_t>(r.objTriSplit);
            const size_t buf2Tris = static_cast<size_t>(std::max(r.triCapacity - r.objTriSplit, 0));
            d.objTriBuf = WgpuBuffer(d.renderer, buf1Tris * BYTES_PER_TRI,
                                      WgpuBuffer::Usage::Storage);
            d.objTriBuf2 = WgpuBuffer(d.renderer, std::max(buf2Tris * BYTES_PER_TRI, size_t(128)),
                                       WgpuBuffer::Usage::Storage);
            d.leafIndexBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.triCapacity) * sizeof(int),
                                         WgpuBuffer::Usage::Storage);
            d.emissiveTriBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.triCapacity) * 4 * sizeof(float),
                                           WgpuBuffer::Usage::Storage);
            d.vtPipeline.setStorageBufferRead(0, d.objTriBuf);
            d.vtPipeline.setStorageTexture(2, d.triTex);
            d.vtPipeline.setStorageBufferRead(4, d.objTriBuf2);
            d.refitPipeline.setTexture(0, d.triTex);
            d.refitPipeline.setStorageBufferRead(3, d.leafIndexBuf);
            d.rtPipeline.setTexture(5, d.triTex);
            d.rtPipeline.setStorageBufferRead(11, d.emissiveTriBuf);
            d.primaryPipeline.setTexture(5, d.triTex);
            d.primaryPipeline.setStorageBufferRead(11, d.emissiveTriBuf);
            d.bouncesPipeline.setTexture(5, d.triTex);
            d.bouncesPipeline.setStorageBufferRead(11, d.emissiveTriBuf);
        }
        if (r.bvhCapacity != d.bvhCapacity_) {
            d.bvhCapacity_ = r.bvhCapacity;
            d.bvhNodeBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.bvhCapacity) * BVH4_GPU_U32S * sizeof(uint32_t),
                                       WgpuBuffer::Usage::Storage);
            d.bvhCounterBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.bvhCapacity) * sizeof(uint32_t),
                                          WgpuBuffer::Usage::Storage);
            d.bvhCounterZeros.resize(r.bvhCapacity, 0u);
            d.refitMetaBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.bvhCapacity) * BVH4_REFIT_INTS * sizeof(int32_t),
                                         WgpuBuffer::Usage::Storage);
            d.refitPipeline.setStorageBuffer(1, d.bvhNodeBuf);
            d.refitPipeline.setStorageBuffer(2, d.bvhCounterBuf);
            d.refitPipeline.setStorageBufferRead(5, d.refitMetaBuf);
            d.rtPipeline.setStorageBufferRead(3, d.bvhNodeBuf);
            d.primaryPipeline.setStorageBufferRead(3, d.bvhNodeBuf);
            d.bouncesPipeline.setStorageBufferRead(3, d.bvhNodeBuf);
        }
        if (r.matCapacity != d.matCapacity_) {
            d.matCapacity_ = r.matCapacity;
            d.matTex = WgpuTexture(d.renderer, r.matCapacity, MAT_TEX_HEIGHT,
                                    WgpuTexture::Format::RGBA32Float,
                                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
            d.rtPipeline.setTexture(4, d.matTex);
            d.primaryPipeline.setTexture(4, d.matTex);
            d.bouncesPipeline.setTexture(4, d.matTex);
        }
        if (r.meshCapacity != d.meshCapacity_) {
            d.meshCapacity_ = r.meshCapacity;
            d.matrixBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.meshCapacity) * 32 * sizeof(float),
                                      WgpuBuffer::Usage::Storage);
            d.motionMatBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.meshCapacity) * 16 * sizeof(float),
                                         WgpuBuffer::Usage::Storage);
            d.vtPipeline.setStorageBufferRead(1, d.matrixBuf);
            d.rtPipeline.setStorageBufferRead(27, d.motionMatBuf);
            d.primaryPipeline.setStorageBufferRead(27, d.motionMatBuf);
            d.bouncesPipeline.setStorageBufferRead(27, d.motionMatBuf);
        }

        // Upload atlas
        if (r.atlasLayers != d.atlasLayers_ || r.atlasCols != d.atlasCols_ || r.tileSize != d.tileSize_) {
            d.atlasLayers_ = r.atlasLayers;
            d.atlasCols_ = r.atlasCols;
            d.tileSize_ = r.tileSize;
            const int layerW = d.atlasCols_ * d.tileSize_;
            const int layerH = d.atlasCols_ * d.tileSize_;
            d.texAtlasTex = WgpuTexture(d.renderer,
                    layerW, layerH,
                    WgpuTexture::Format::RGBA8Unorm,
                    WgpuTexture::Dimension::D2Array,
                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst,
                    r.atlasLayers);
            d.rtPipeline.setTexture(6, d.texAtlasTex);
            d.primaryPipeline.setTexture(6, d.texAtlasTex);
            d.bouncesPipeline.setTexture(6, d.texAtlasTex);
        }
        // Upload atlas layers
        const size_t layerBytes = static_cast<size_t>(d.atlasCols_ * d.tileSize_) * (d.atlasCols_ * d.tileSize_) * 4;
        for (int layer = 0; layer < r.atlasLayers; ++layer) {
            d.texAtlasTex.writeLayer(layer, r.atlasData.data() + layer * layerBytes, layerBytes);
        }

        // Upload geometry + BVH
        d.bvhNodeBuf.write(d.bvhNodeCpuBuf.data(), d.numBvhNodes_ * BVH4_GPU_U32S * sizeof(uint32_t));
        d.refitMetaBuf.write(d.refitMetaCpuBuf.data(), d.numBvhNodes_ * BVH4_REFIT_INTS * sizeof(int32_t));
        // Upload objTri data split across two buffers
        {
            const size_t splitAt = static_cast<size_t>(d.objTriSplit_);
            const size_t totalTris = static_cast<size_t>(d.triCount_);
            const size_t buf1Tris = std::min(totalTris, splitAt);
            const size_t buf2Tris = totalTris > splitAt ? totalTris - splitAt : 0;
            d.objTriBuf.write(d.rawObjTriBuf.data(), buf1Tris * BYTES_PER_TRI);
            if (buf2Tris > 0) {
                d.objTriBuf2.write(d.rawObjTriBuf.data() + splitAt * 32, buf2Tris * BYTES_PER_TRI);
            }
        }
        d.leafIndexBuf.write(d.leafIndices.data(), d.leafIndices.size() * sizeof(int));
        d.matTex.write(d.matBuffer.data(), d.matBuffer.size() * sizeof(float));

        if (d.emissiveTriCount_ > 0) {
            d.emissiveTriBuf.write(d.emissiveTriCpu.data(),
                                   d.emissiveTriCpu.size() * sizeof(float));
        }

        // Free large CPU-side build buffers now that data lives on GPU.
        // matBuffer and matrixCpuBuf are kept alive — they're small (~KB) and needed for
        // the append-only fast path to write new material/mesh-matrix rows.
        { std::vector<float>().swap(d.triBuffer); }
        { std::vector<float>().swap(d.rawObjTriBuf); }
        { std::vector<uint32_t>().swap(d.bvhNodeCpuBuf); }
        { std::vector<int32_t>().swap(d.refitMetaCpuBuf); }
        { std::vector<float>().swap(d.emissiveTriCpu); }

        // Start async RT pipeline compilation NOW — after all topology bindings (buffer sizes,
        // texture formats) are finalised.  This ensures:
        //  • layoutDirty stays false between now and the first dispatch (per-frame ping-pong
        //    swaps only change texture views, never the layout type/format → no dirty).
        //  • When async completes, the result is used directly without a second compile.
        // Previously these were started in the constructor with 1-tri placeholder bindings,
        // causing a layoutDirty discard → a second full synchronous compile on the main thread
        // (~30s extra freeze on first cold-cache launch).
        d.rtPipeline.startAsyncBuild();
        d.primaryPipeline.startAsyncBuild();
        d.bouncesPipeline.startAsyncBuild();
        // Mark that triTex has not yet been populated via the VT pass.
        // The first real dispatch must run the VT pass even if no mesh moved.
        d.firstDispatchPending_ = true;
    }

    // Before first build, skip RT dispatch
    if (d.triCount_ == 0) {
        d.displayMat->customTextures["accumTex"] = d.accum.read;
        d.displayMat->customTextures["gBufTex"]  = d.gBuf.write;
        d.displayMat->customTextures["diffTex"]  = d.diffAccum.read;
        d.displayMat->customTextures["specTex"]  = d.specAccum.read;
        d.displayMat->uniformsNeedUpdate = true;
        d.renderer.render(d.displayScene, d.displayCam);
        return;
    }

    scene.updateMatrixWorld();

    // Expand entries for movement detection (uses current world matrices)
    auto rtEntries = expandMeshEntries(rtMeshes);
    applyHideOverride(rtEntries);

    // Detect per-entry matrix changes; build bitmask of which entries moved.
    // movedBits is used by the GPU for per-pixel accumulation reset.
    // anyMeshMoved drives the vertex-transform and BVH-refit pipelines.
    uint32_t movedBits[4] = {0u, 0u, 0u, 0u};
    bool anyMeshMoved = (d.prevEntryMatrices.size() != rtEntries.size());

    if (topoJustFinished) {
        // Topology change: all pixels need to re-accumulate (mesh-to-triangle mapping changed)
        movedBits[0] = movedBits[1] = movedBits[2] = movedBits[3] = 0xFFFFFFFFu;
        anyMeshMoved = true;
    } else if (anyMeshMoved) {
        // Entry count mismatch (shouldn't happen without topo change, but be safe)
        movedBits[0] = movedBits[1] = movedBits[2] = movedBits[3] = 0xFFFFFFFFu;
    } else {
        for (size_t i = 0; i < rtEntries.size() && i < static_cast<size_t>(d.meshCapacity_) && i < 128u; ++i) {
            if (rtEntries[i].worldMatrix != d.prevEntryMatrices[i]) {
                anyMeshMoved = true;
                movedBits[i >> 5u] |= (1u << (i & 31u));
            }
        }
    }
    // Note: camMoved triggers per-pixel reprojection via camMovedFlag (params.w bit 1); no movedBits needed for that.

    // --- Consolidation rebuild: after a fast-append the BVH has an overlay combined-root.
    // Once the scene has been stable (no topology or matrix changes) for 120 frames (~2 s at
    // 60 fps), schedule a full rebuild so the BVH reverts to a clean single-root structure.
    // The rebuild fires by clearing prevMeshes, which makes topoChanged = true next frame.
    // frameCount_ will reset (brief flash is acceptable — scene has converged by then).
    if (d.pendingConsolidation_) {
        if (!topoChanged && !anyMeshMoved) {
            ++d.stableFramesSinceAppend_;
            if (d.stableFramesSinceAppend_ >= 120) {
                std::cerr << "[PathTracer] Consolidation rebuild triggered after "
                          << d.stableFramesSinceAppend_ << " stable frames\n";
                d.prevMeshes.clear();       // force topoChanged = true next frame
                d.pendingConsolidation_    = false;
                d.stableFramesSinceAppend_ = 0;
            }
        } else {
            // Scene changed again — restart the stable-frame counter so we only
            // consolidate after a truly quiet period following the last append.
            d.stableFramesSinceAppend_ = 0;
        }
    }

    // Compute per-entry motion matrices: prevWorld * inverse(curWorld)
    // Used by the RT accumulation reprojection to map pixels on moving objects
    // to their previous-frame screen position so temporal history follows the surface.
    d.motionMatCpu.resize(static_cast<size_t>(d.meshCapacity_) * 16, 0.f);
    for (size_t i = 0; i < rtEntries.size() && i < static_cast<size_t>(d.meshCapacity_); ++i) {
        Matrix4 mot;  // identity by default
        if (i < d.prevEntryMatrices.size()) {
            Matrix4 curInv(rtEntries[i].worldMatrix);
            curInv.invert();
            mot.multiplyMatrices(d.prevEntryMatrices[i], curInv);
        }
        // else: identity (new entry, no previous frame)
        std::memcpy(d.motionMatCpu.data() + i * 16, mot.elements.data(), 16 * sizeof(float));
    }
    d.motionMatBuf.write(d.motionMatCpu.data(), d.motionMatCpu.size() * sizeof(float));

    d.prevEntryMatrices.resize(rtEntries.size());
    for (size_t i = 0; i < rtEntries.size(); ++i)
        d.prevEntryMatrices[i] = rtEntries[i].worldMatrix;
    if (anyMeshMoved) {
        if (!topoChanged) {
            std::ranges::fill(d.matrixCpuBuf, 0.f);
            int mi = 0;
            for (auto& entry : rtEntries) {
                if (mi >= d.meshCapacity_) break;
                const auto& w = entry.worldMatrix;
                Matrix4 nm(w);
                nm.invert().transpose();
                float* p = d.matrixCpuBuf.data() + mi * 32;
                std::memcpy(p, w.elements.data(), 16 * sizeof(float));
                std::memcpy(p + 16, nm.elements.data(), 16 * sizeof(float));
                ++mi;
            }
        }
        d.matrixBuf.write(d.matrixCpuBuf.data(), static_cast<size_t>(d.meshCapacity_) * 32 * sizeof(float));

        // Compute 2D dispatch dimensions (WebGPU max per-dimension is 65535)
        const uint32_t vtTotal = (static_cast<uint32_t>(d.triCount_) + 63u) / 64u;
        const uint32_t vtGx = (std::min)(vtTotal, 65535u);
        const uint32_t vtGy = (vtTotal + vtGx - 1u) / vtGx;
        d.vtDispatchX_ = vtGx;
        d.vtDispatchY_ = vtGy;

        VtGpuUniforms vtU{};
        vtU.triCount = static_cast<uint32_t>(d.triCount_);
        vtU.groupsX  = vtGx;
        vtU.splitAt  = static_cast<uint32_t>(d.objTriSplit_);
        d.vtUniBuf.write(&vtU, sizeof(vtU));

        const uint32_t rfTotal = (static_cast<uint32_t>(d.leafIndices.size()) + 63u) / 64u;
        const uint32_t rfGx = (std::min)(rfTotal, 65535u);
        const uint32_t rfGy = (rfTotal + rfGx - 1u) / rfGx;
        d.rfDispatchX_ = rfGx;
        d.rfDispatchY_ = rfGy;

        RefitGpuUniforms rfU{};
        rfU.leafCount = static_cast<uint32_t>(d.leafIndices.size());
        rfU.groupsX   = rfGx;
        d.refitUniBuf.write(&rfU, sizeof(rfU));
        d.bvhCounterBuf.write(d.bvhCounterZeros.data(),
                              static_cast<size_t>(d.numBvhNodes_) * sizeof(uint32_t));
    }

    // Compute tanHalfFov from the camera
    float tanHalfFov = 1.f;
    if (auto* persp = dynamic_cast<PerspectiveCamera*>(&camera)) {
        tanHalfFov = std::tan(persp->fov * 3.14159265358979f / 360.f);
    }

    // Pack uniform buffer
    RtGpuUniforms u{};
    u.camOri[0] = camPos.x; u.camOri[1] = camPos.y; u.camOri[2] = camPos.z;
    u.camFwd[0] = fwd.x; u.camFwd[1] = fwd.y; u.camFwd[2] = fwd.z;
    u.camRgt[0] = rgt.x; u.camRgt[1] = rgt.y; u.camRgt[2] = rgt.z;
    u.camUp[0] = up.x; u.camUp[1] = up.y; u.camUp[2] = up.z;
    d.fillPrevCamUbo(u.prevCamOri, u.prevCamFwd, u.prevCamRgt, u.prevCamUp);
    u.iRes[0] = static_cast<float>(d.width_);
    u.iRes[1] = static_cast<float>(d.height_);
    u.tanHalfFov[0] = tanHalfFov;
    u.frameCount[0] = d.frameCount_;
    u.triCount[0] = static_cast<float>(d.triCount_);
    u.mode[0] = 1.f;  // always path tracer
    u.spp[1] = static_cast<float>(d.tileSize_);
    u.movedMeshBits[0] = movedBits[0];
    u.movedMeshBits[1] = movedBits[1];
    u.movedMeshBits[2] = movedBits[2];
    u.movedMeshBits[3] = movedBits[3];
    // Detect if any moved mesh is an emissive source — triggers tighter accum cap in shader.
    bool anyEmissiveMoved = false;
    if (movedBits[0] | movedBits[1] | movedBits[2] | movedBits[3]) {
        for (int mi = 0; mi < 128 && !anyEmissiveMoved; ++mi) {
            const uint32_t word = movedBits[mi >> 5];
            const uint32_t bit  = static_cast<uint32_t>(mi & 31);
            if ((word >> bit) & 1u) {
                anyEmissiveMoved = d.emissiveMeshSet_.count(mi) > 0;
            }
        }
    }

    // Helper: upload an equirectangular texture to a GPU texture object.
    auto uploadEquirect = [&](Texture* tex, WgpuTexture& gpuTex) {
        auto& img = tex->image();
        if (img.width <= 0 || img.height <= 0) return;
        bool isHdr = false;
        try { (void)img.data<float>(); isHdr = true; }
        catch (const std::bad_variant_access&) {}
        if (isHdr) {
            const auto& src = img.data<float>();
            gpuTex = WgpuTexture(d.renderer,
                                 static_cast<uint32_t>(img.width),
                                 static_cast<uint32_t>(img.height),
                                 WgpuTexture::Format::RGBA32Float,
                                 WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
            gpuTex.write(src.data(), src.size() * sizeof(float));
        } else {
            const auto& src = img.data<unsigned char>();
            gpuTex = WgpuTexture(d.renderer,
                                 static_cast<uint32_t>(img.width),
                                 static_cast<uint32_t>(img.height),
                                 WgpuTexture::Format::RGBA8Unorm,
                                 WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
            gpuTex.write(src.data(), src.size());
        }
    };

    // --- Environment (IBL lighting): scene.environment ---
    // Fallback: use background texture for IBL when no environment is set.
    u.envColor[3] = 0.f;  // default: no IBL
    if (auto* s = dynamic_cast<Scene*>(&scene)) {
        Texture* envTex = s->environment.get();
        if (!envTex && s->background.isTexture()) {
            envTex = s->background.texture().get();
        }
        if (envTex) {
            if (envTex != d.prevEnvTex_) {
                uploadEquirect(envTex, d.envTexGpu);
                d.rtPipeline.setTexture(9, d.envTexGpu);
                d.primaryPipeline.setTexture(9, d.envTexGpu);
                d.bouncesPipeline.setTexture(9, d.envTexGpu);

                // Build env CDF for importance sampling.
                auto& img = envTex->image();
                if (img.width > 0 && img.height > 0) {
                    int w = img.width, h = img.height;
                    bool isHdr = false;
                    try { (void)img.data<float>(); isHdr = true; }
                    catch (const std::bad_variant_access&) {}
#ifdef __EMSCRIPTEN__
                    EnvCdfResult cdf;
                    if (isHdr) {
                        const float* ptr = img.data<float>().data();
                        std::vector<float> tmp(ptr, ptr + static_cast<size_t>(w) * h * 4);
                        cdf = buildEnvCdf(tmp, w, h);
                    } else {
                        const unsigned char* ptr = img.data<unsigned char>().data();
                        std::vector<unsigned char> tmp(ptr, ptr + static_cast<size_t>(w) * h * 4);
                        cdf = buildEnvCdf(tmp, w, h);
                    }
                    d.envLumSum_ = cdf.totalSum;
                    d.envCdfTex = WgpuTexture(d.renderer,
                                              static_cast<uint32_t>(cdf.width),
                                              static_cast<uint32_t>(cdf.height),
                                              WgpuTexture::Format::R32Float,
                                              WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
                    d.envCdfTex.write(cdf.conditional.data(), cdf.conditional.size() * sizeof(float));
                    d.rtPipeline.setTexture(12, d.envCdfTex);
                    d.primaryPipeline.setTexture(12, d.envCdfTex);
                    d.bouncesPipeline.setTexture(12, d.envCdfTex);
                    d.envMargTex = WgpuTexture(d.renderer,
                                               static_cast<uint32_t>(cdf.height), 1u,
                                               WgpuTexture::Format::R32Float,
                                               WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
                    d.envMargTex.write(cdf.marginal.data(), cdf.marginal.size() * sizeof(float));
                    d.rtPipeline.setTexture(13, d.envMargTex);
                    d.primaryPipeline.setTexture(13, d.envMargTex);
                    d.bouncesPipeline.setTexture(13, d.envMargTex);
                    if (!d.shaderHasEnvCdf_) {
                        d.rtPipeline.replaceShader(buildRtShader(true));
                        d.primaryPipeline.replaceShader(buildRtShader(true));
                        d.bouncesPipeline.replaceShader(buildRtShader(true));
                        d.shaderHasEnvCdf_ = true;
                    }
#else
                    if (isHdr) {
                        const float* ptr = img.data<float>().data();
                        d.asyncEnvCdf_ = std::async(std::launch::async, [ptr, w, h]() {
                            std::vector<float> tmp(ptr, ptr + static_cast<size_t>(w) * h * 4);
                            return buildEnvCdf(tmp, w, h);
                        });
                    } else {
                        const unsigned char* ptr = img.data<unsigned char>().data();
                        d.asyncEnvCdf_ = std::async(std::launch::async, [ptr, w, h]() {
                            std::vector<unsigned char> tmp(ptr, ptr + static_cast<size_t>(w) * h * 4);
                            return buildEnvCdf(tmp, w, h);
                        });
                    }
                    d.envCdfPending_ = true;
#endif
                }
                d.prevEnvTex_ = envTex;
            }
            u.envColor[3] = 2.f;
        } else {
            // No environment = no IBL; clear CDF
            if (d.prevEnvTex_) {
                d.prevEnvTex_ = nullptr;
                d.envLumSum_ = 0.f;
                if (d.shaderHasEnvCdf_) {
                    d.rtPipeline.replaceShader(buildRtShader(false));
                    d.primaryPipeline.replaceShader(buildRtShader(false));
                    d.bouncesPipeline.replaceShader(buildRtShader(false));
                    d.shaderHasEnvCdf_ = false;
                }
            }
        }

        // --- Background (ray miss): scene.background ---
        u.bgColor[3] = 0.f;  // default: procedural sky gradient
        if (s->background.isTexture()) {
            Texture* bgTex = s->background.texture().get();
            if (bgTex != d.prevBgTex_) {
                uploadEquirect(bgTex, d.bgTexGpu);
                d.rtPipeline.setTexture(16, d.bgTexGpu);
                d.primaryPipeline.setTexture(16, d.bgTexGpu);
                d.bouncesPipeline.setTexture(16, d.bgTexGpu);
                d.prevBgTex_ = bgTex;
            }
            u.bgColor[3] = 2.f;
        } else if (s->background.isColor()) {
            const Color& c = s->background.color();
            u.bgColor[0] = c.r; u.bgColor[1] = c.g; u.bgColor[2] = c.b;
            u.bgColor[3] = 1.f;
            d.prevBgTex_ = nullptr;
        }

    }
#ifndef __EMSCRIPTEN__
    // Check if async CDF build finished — upload to GPU
    if (d.envCdfPending_ && d.asyncEnvCdf_.valid() &&
        d.asyncEnvCdf_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        auto cdf = d.asyncEnvCdf_.get();
        d.envCdfPending_ = false;
        d.envLumSum_ = cdf.totalSum;
        d.envCdfTex = WgpuTexture(d.renderer,
                                  static_cast<uint32_t>(cdf.width),
                                  static_cast<uint32_t>(cdf.height),
                                  WgpuTexture::Format::R32Float,
                                  WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
        d.envCdfTex.write(cdf.conditional.data(), cdf.conditional.size() * sizeof(float));
        d.rtPipeline.setTexture(12, d.envCdfTex);
        d.primaryPipeline.setTexture(12, d.envCdfTex);
        d.bouncesPipeline.setTexture(12, d.envCdfTex);
        d.envMargTex = WgpuTexture(d.renderer,
                                   static_cast<uint32_t>(cdf.height), 1u,
                                   WgpuTexture::Format::R32Float,
                                   WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
        d.envMargTex.write(cdf.marginal.data(), cdf.marginal.size() * sizeof(float));
        d.rtPipeline.setTexture(13, d.envMargTex);
        d.primaryPipeline.setTexture(13, d.envMargTex);
        d.bouncesPipeline.setTexture(13, d.envMargTex);
        // Swap to env CDF shader variant
        if (!d.shaderHasEnvCdf_) {
            d.rtPipeline.replaceShader(buildRtShader(true));
            d.primaryPipeline.replaceShader(buildRtShader(true));
            d.bouncesPipeline.replaceShader(buildRtShader(true));
            d.shaderHasEnvCdf_ = true;
        }
    }
#endif

    u.envIntensity[0] = d.envIntensity_;
    // Pass envmap dimensions and total luminance sum for importance sampling
    if (d.envLumSum_ > 0.f && d.prevEnvTex_) {
        auto& eImg = d.prevEnvTex_->image();
        u.envIntensity[1] = static_cast<float>(eImg.width);
        u.envIntensity[2] = static_cast<float>(eImg.height);
        u.envIntensity[3] = d.envLumSum_;
    } else {
        u.envIntensity[1] = 0.f;
        u.envIntensity[2] = 0.f;
        u.envIntensity[3] = 0.f;
    }
    u.params[0] = static_cast<float>(d.maxBounces_);
    u.params[1] = static_cast<float>(d.globalFrameCounter_++);
    u.params[2] = (d.foveatedEnabled_ && d.frameCount_ > 0.f) ? 1.f : 0.f;
    // params.w encodes two boolean flags:
    //   1.0 = forceReset (first frame / topology rebuild)
    //   2.0 = camMoved   (camera translated/rotated this frame)
    //   3.0 = both
    {
        const float fr = (d.frameCount_ == 0.f) ? 1.f : 0.f;
        const float cm = camMoved ? 2.f : 0.f;
        u.params[3] = fr + cm;
    }
    u.emissiveInfo[0] = static_cast<float>(d.emissiveTriCount_);
    u.emissiveInfo[1] = d.emissiveTotalPower_;
    u.emissiveInfo[2] = d.fireflyCap_;  // luminance cap for indirect MIS contributions
    u.emissiveInfo[3] = static_cast<float>(d.aovMode_);  // AOV visualization mode
    u.spp[0] = d.restirGiEnabled_ ? 1.f : 0.f;
    u.restirParams[0] = d.restirEnabled_ ? 1.f : 0.f;
    u.restirParams[1] = camMoved ? 5.f : 20.f;  // M clamp — low during motion to flush stale reservoirs
    u.restirParams[2] = anyEmissiveMoved ? 1.f : 0.f;  // emissive source moved → tight accum cap
    u.restirParams[3] = 0.f;  // reserved (was rawMode/temporal denoiser flag)
    u.bvhAux[0] = static_cast<uint32_t>(d.bvhRootIdx_);  // traversal root (0=normal, >0=overlay)

    int nLights = 0;
    auto packLight = [&](float px, float py, float pz, float r, float g, float b, float type) {
        if (nLights >= 4) return;
        u.lightPos[nLights][0] = px; u.lightPos[nLights][1] = py; u.lightPos[nLights][2] = pz;
        u.lightCol[nLights][0] = r;  u.lightCol[nLights][1] = g;  u.lightCol[nLights][2] = b;
        u.lightType[nLights][0] = type;
        ++nLights;
    };
    for (auto* l : pointLights) {
        if (!l->visible) continue;
        const auto& lp = l->position;
        const auto& lc = l->color;
        const float li = l->intensity;
        packLight(lp.x, lp.y, lp.z, lc.r * li, lc.g * li, lc.b * li, 0.f);
        u.lightType[nLights - 1][3] = l->distance;
        u.lightDir[nLights - 1][3] = l->decay;
    }
    for (auto* l : dirLights) {
        if (!l->visible) continue;
        Vector3 dir = Vector3(l->position).sub(l->target().position).normalize();
        const auto& lc = l->color;
        const float li = l->intensity;
        packLight(dir.x, dir.y, dir.z, lc.r * li, lc.g * li, lc.b * li, 1.f);
    }
    for (auto* l : spotLights) {
        if (nLights >= 4 || !l->visible) break;
        const auto& lp = l->position;
        const auto& lc = l->color;
        const float li = l->intensity;
        Vector3 dir = Vector3(l->target().position).sub(lp).normalize();
        const float cosAngle = std::cos(l->angle);
        const float cosOuter = std::cos(l->angle * (1.f + l->penumbra));
        u.lightPos[nLights][0] = lp.x; u.lightPos[nLights][1] = lp.y; u.lightPos[nLights][2] = lp.z;
        u.lightCol[nLights][0] = lc.r * li; u.lightCol[nLights][1] = lc.g * li; u.lightCol[nLights][2] = lc.b * li;
        u.lightType[nLights][0] = 2.f;
        u.lightType[nLights][1] = cosAngle;
        u.lightType[nLights][2] = cosOuter;
        u.lightType[nLights][3] = l->distance;
        u.lightDir[nLights][0] = dir.x; u.lightDir[nLights][1] = dir.y; u.lightDir[nLights][2] = dir.z;
        u.lightDir[nLights][3] = l->decay;
        ++nLights;
    }
    u.lightCount[0] = static_cast<float>(nLights);

    // Flush stale ReSTIR reservoirs when the light count changes (e.g. light.visible toggle).
    // Reservoirs hold candidates for lights that no longer exist, causing them to linger.
    // Re-stamp frameCount in the uniform AFTER the reset so the GPU sees fc=0 this frame
    // (forceReset fires immediately rather than one frame late).
    if (nLights != d.prevNLights_) {
        d.frameCount_ = 0.f;
        d.prevNLights_ = nLights;
        u.frameCount[0] = 0.f;
        // params[3] was packed before this detection; recompute from scratch to ensure
        // forceReset (bit 0) is set correctly regardless of prior state of params[3].
        // += 1 would corrupt the encoding if params[3] was already 1 (fr already set),
        // flipping forceReset off and camMoved on instead.
        u.params[3] = 1.f + (camMoved ? 2.f : 0.f);
    }

    d.rtUniformBuf.write(&u, sizeof(u));

    // Set per-frame texture bindings (accum + hitMesh ping-pong)
    auto& activePipeline = d.rtPipeline;

    // Skip RT dispatch if either pipeline is still compiling asynchronously.
    // On cold shader cache (first launch / code change) this can take 10-30 seconds.
    if (!activePipeline.isReady() || !d.primaryPipeline.isReady() || !d.bouncesPipeline.isReady()) {
        ++d.shaderWaitFrames_;
        if (d.shaderWaitFrames_ == 1) {
            std::cerr << "[PathTracer] Compiling RT shaders — first launch may take 10-30s "
                         "(subsequent launches are instant via driver cache)." << std::endl;
        } else if (d.shaderWaitFrames_ % 180 == 0) {
            std::cerr << "[PathTracer] Still compiling shaders... ("
                      << (d.shaderWaitFrames_ / 60) << "s elapsed)" << std::endl;
        }
        // Tick the device so backends that need main-thread event processing make progress.
        wgpuDevicePoll(d.device, false, nullptr);
        // Safety net: after ~60 seconds still black, force-block until async finishes.
        // This prevents a permanent black screen if something prevents the future from
        // being polled to completion in the normal render loop.
        if (d.shaderWaitFrames_ >= 3600) {
            std::cerr << "[PathTracer] 60s timeout — force-completing shader compilation." << std::endl;
            activePipeline.forceFinishBuild();
            d.primaryPipeline.forceFinishBuild();
            d.bouncesPipeline.forceFinishBuild();
            d.shaderWaitFrames_ = 0;
            // Fall through: isReady() now returns true; encode() will do the fast sync rebuild.
        } else {
            d.displayMat->customTextures["accumTex"] = d.accum.read;
            d.displayMat->customTextures["gBufTex"]  = d.gBuf.write;
            d.displayMat->uniformsNeedUpdate = true;
            d.renderer.render(d.displayScene, d.displayCam);
            return;
        }
    }
    if (d.shaderWaitFrames_ > 0) {
        std::cerr << "[PathTracer] RT shaders ready after "
                  << (d.shaderWaitFrames_ / 60) << "s — rendering started." << std::endl;
        d.shaderWaitFrames_ = 0;
    }
    // First dispatch after async shader compilation: triTex has never been written by the VT
    // pass (every prior frame returned early).  Force the VT pass to run now so the RT shader
    // sees valid world-space triangle data, and reset accumulation for a clean start.
    if (d.firstDispatchPending_) {
        d.firstDispatchPending_ = false;
        anyMeshMoved = true;
        movedBits[0] = movedBits[1] = movedBits[2] = movedBits[3] = 0xFFFFFFFFu;
        d.frameCount_ = 0.f;
    }

    // Persistent-thread dispatch size.  Fixed count (not pixel-proportional):
    // we want enough workgroups to saturate the GPU but few enough that each
    // thread pulls multiple pixels from the queue (→ within-warp rebalancing).
    // 1024 workgroups × 64 threads = 65k threads.  On 1080p (2M pixels) that's
    // ~31 paths per thread; threads finishing short paths early pull more work
    // while warp-mates continue tracing long paths, reducing idle lanes.
    constexpr uint32_t rtWorkgroups = 1024u;

    for (int sampleIdx = 0; sampleIdx < d.spp_; ++sampleIdx) {

        // Update frame count in uniforms for each sample (different random seed)
        if (sampleIdx > 0) {
            d.frameCount_ += 1.f;
            u.frameCount[0] = d.frameCount_;
            d.rtUniformBuf.write(&u, sizeof(u));
        }

        activePipeline.setTexture(1, *d.accum.read);
        activePipeline.setStorageTexture(2, *d.accum.write);
        activePipeline.setTexture(7, *d.hitMesh.read);
        activePipeline.setStorageTexture(8, *d.hitMesh.write);
        d.primaryPipeline.setTexture(1, *d.accum.read);
        d.primaryPipeline.setStorageTexture(2, *d.accum.write);
        d.primaryPipeline.setTexture(7, *d.hitMesh.read);
        d.primaryPipeline.setStorageTexture(8, *d.hitMesh.write);
        d.bouncesPipeline.setTexture(1, *d.accum.read);
        d.bouncesPipeline.setStorageTexture(2, *d.accum.write);
        d.bouncesPipeline.setTexture(7, *d.hitMesh.read);
        d.bouncesPipeline.setStorageTexture(8, *d.hitMesh.write);

        // GPU dispatch
        {
            WGPUCommandEncoderDescriptor encDesc{};
            encDesc.label = WGPUStringView{"pt_enc", WGPU_STRLEN};
            WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(d.device, &encDesc);

            WGPUComputePassDescriptor passDesc{};

            // VT + BVH refit only on first sample (geometry doesn't change between samples)
            if (sampleIdx == 0 && anyMeshMoved) {
                passDesc.label = WGPUStringView{"vt_pass", WGPU_STRLEN};
                WGPUComputePassEncoder vtPass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
                d.vtPipeline.encode(vtPass, d.vtDispatchX_, d.vtDispatchY_);
                wgpuComputePassEncoderEnd(vtPass);
                wgpuComputePassEncoderRelease(vtPass);

                if (!topoJustFinished) {
                    passDesc.label = WGPUStringView{"rf_pass", WGPU_STRLEN};
                    WGPUComputePassEncoder rfPass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
                    d.refitPipeline.encode(rfPass, d.rfDispatchX_, d.rfDispatchY_);
                    wgpuComputePassEncoderEnd(rfPass);
                    wgpuComputePassEncoderRelease(rfPass);
                }
            }

            // Reset both work-queue counters to 0 before the encoder runs.
            // These are Queue.writeBuffer operations which execute in submission
            // order before the subsequent compute passes in the same encoder.
            // Separate counters are needed (not one shared reset between passes)
            // because CPU-side queue writes cannot interleave with GPU compute
            // passes inside a single command encoder — both writes would land
            // before either pass, leaving the shared counter consumed after the
            // primary pass and exhausted before rt_main starts.
            const uint32_t counterZero = 0u;
            d.primaryCounterBuf.write(&counterZero, sizeof(counterZero));
            d.pathCounterBuf.write(&counterZero, sizeof(counterZero));
            d.bounceCounterBuf.write(&counterZero, sizeof(counterZero));

            // Kernel split — step 1: primary BVH traversal → primaryHitBuf.
            passDesc.label = WGPUStringView{"rt_primary_pass", WGPU_STRLEN};
            WGPUComputePassEncoder primaryPass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
            d.primaryPipeline.encode(primaryPass, rtWorkgroups, 1u);
            wgpuComputePassEncoderEnd(primaryPass);
            wgpuComputePassEncoderRelease(primaryPass);

            // Kernel split — step 2: primaryShade + NEE + ReSTIR DI + serialize → pathStateBuf.
            passDesc.label = WGPUStringView{"rt_pass", WGPU_STRLEN};
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
            activePipeline.encode(pass, rtWorkgroups, 1u);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);

            // Kernel split — step 3: runBounces + accumulation.
            passDesc.label = WGPUStringView{"rt_bounces_pass", WGPU_STRLEN};
            WGPUComputePassEncoder bouncesPass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
            d.bouncesPipeline.encode(bouncesPass, rtWorkgroups, 1u);
            wgpuComputePassEncoderEnd(bouncesPass);
            wgpuComputePassEncoderRelease(bouncesPass);

            WGPUCommandBufferDescriptor cmdDesc{};
            cmdDesc.label = WGPUStringView{"pt_cmd", WGPU_STRLEN};
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
            wgpuQueueSubmit(d.queue, 1, &cmd);

            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
        }

        // Swap all ping-pong buffers so next sample reads this sample's output
        d.accum.swap();
        d.hitMesh.swap();
        d.diffAccum.swap();
        d.specAccum.swap();

        d.gBuf.swap();
        d.rtPipeline.setStorageTexture(10, *d.gBuf.read);
        d.rtPipeline.setTexture(15, *d.gBuf.write);

        d.reservoir.swap();
        d.reservoirW.swap();
        d.rtPipeline.setTexture(17, *d.reservoir.read);
        d.rtPipeline.setStorageTexture(18, *d.reservoir.write);
        d.rtPipeline.setTexture(19, *d.reservoirW.read);
        d.rtPipeline.setStorageTexture(20, *d.reservoirW.write);

        d.giRes.swap();
        d.giResW.swap();
        d.giResLo.swap();
        d.rtPipeline.setTexture(28, *d.giRes.read);
        d.rtPipeline.setStorageTexture(29, *d.giRes.write);
        d.rtPipeline.setTexture(30, *d.giResW.read);
        d.rtPipeline.setStorageTexture(31, *d.giResW.write);
        d.rtPipeline.setTexture(32, *d.giResLo.read);
        d.rtPipeline.setStorageTexture(33, *d.giResLo.write);

        d.moments.swap();
        d.rtPipeline.setTexture(21, *d.moments.read);
        d.rtPipeline.setStorageTexture(22, *d.moments.write);
        d.rtPipeline.setTexture(23, *d.diffAccum.read);
        d.rtPipeline.setStorageTexture(24, *d.diffAccum.write);
        d.rtPipeline.setTexture(25, *d.specAccum.read);
        d.rtPipeline.setStorageTexture(26, *d.specAccum.write);

        // Mirror all ping-pong rebindings on primaryPipeline — most are unused
        // by rt_primary_main but WebGPU's bind-group validation requires current
        // texture identities across the whole layout.
        d.primaryPipeline.setTexture(1, *d.accum.read);
        d.primaryPipeline.setStorageTexture(2, *d.accum.write);
        d.primaryPipeline.setTexture(7, *d.hitMesh.read);
        d.primaryPipeline.setStorageTexture(8, *d.hitMesh.write);
        d.primaryPipeline.setStorageTexture(10, *d.gBuf.read);
        d.primaryPipeline.setTexture(15, *d.gBuf.write);
        d.primaryPipeline.setTexture(17, *d.reservoir.read);
        d.primaryPipeline.setStorageTexture(18, *d.reservoir.write);
        d.primaryPipeline.setTexture(19, *d.reservoirW.read);
        d.primaryPipeline.setStorageTexture(20, *d.reservoirW.write);
        d.primaryPipeline.setTexture(21, *d.moments.read);
        d.primaryPipeline.setStorageTexture(22, *d.moments.write);
        d.primaryPipeline.setTexture(23, *d.diffAccum.read);
        d.primaryPipeline.setStorageTexture(24, *d.diffAccum.write);
        d.primaryPipeline.setTexture(25, *d.specAccum.read);
        d.primaryPipeline.setStorageTexture(26, *d.specAccum.write);
        d.primaryPipeline.setTexture(28, *d.giRes.read);
        d.primaryPipeline.setStorageTexture(29, *d.giRes.write);
        d.primaryPipeline.setTexture(30, *d.giResW.read);
        d.primaryPipeline.setStorageTexture(31, *d.giResW.write);
        d.primaryPipeline.setTexture(32, *d.giResLo.read);
        d.primaryPipeline.setStorageTexture(33, *d.giResLo.write);

        // Mirror all ping-pong rebindings on bouncesPipeline.
        d.bouncesPipeline.setTexture(1, *d.accum.read);
        d.bouncesPipeline.setStorageTexture(2, *d.accum.write);
        d.bouncesPipeline.setTexture(7, *d.hitMesh.read);
        d.bouncesPipeline.setStorageTexture(8, *d.hitMesh.write);
        d.bouncesPipeline.setStorageTexture(10, *d.gBuf.read);
        d.bouncesPipeline.setTexture(15, *d.gBuf.write);
        d.bouncesPipeline.setTexture(17, *d.reservoir.read);
        d.bouncesPipeline.setStorageTexture(18, *d.reservoir.write);
        d.bouncesPipeline.setTexture(19, *d.reservoirW.read);
        d.bouncesPipeline.setStorageTexture(20, *d.reservoirW.write);
        d.bouncesPipeline.setTexture(21, *d.moments.read);
        d.bouncesPipeline.setStorageTexture(22, *d.moments.write);
        d.bouncesPipeline.setTexture(23, *d.diffAccum.read);
        d.bouncesPipeline.setStorageTexture(24, *d.diffAccum.write);
        d.bouncesPipeline.setTexture(25, *d.specAccum.read);
        d.bouncesPipeline.setStorageTexture(26, *d.specAccum.write);
        d.bouncesPipeline.setTexture(28, *d.giRes.read);
        d.bouncesPipeline.setStorageTexture(29, *d.giRes.write);
        d.bouncesPipeline.setTexture(30, *d.giResW.read);
        d.bouncesPipeline.setStorageTexture(31, *d.giResW.write);
        d.bouncesPipeline.setTexture(32, *d.giResLo.read);
        d.bouncesPipeline.setStorageTexture(33, *d.giResLo.write);
    }

    // Update denoiser's moments reference (reads converged moments)
    d.atrousPipeline.setTexture(6, *d.moments.read);

    // Spatial denoiser (path tracer mode only) — à-trous wavelet filter on
    // split diffuse/specular channels independently. Each channel gets its own
    // mode (diffuse=0, specular=1) with roughness-adaptive edge-stopping.
    // 5 cascade passes with step sizes 1, 2, 4, 8, 16 give effective 32-pixel radius.
    WgpuTexture* displayTex = d.accum.read;
    WgpuTexture* displayDiff = d.diffAccum.read;
    WgpuTexture* displaySpec = d.specAccum.read;
    const bool hasMotion = (movedBits[0] | movedBits[1] | movedBits[2] | movedBits[3]) != 0u;
    (void) hasMotion; (void) camMoved;
    // Run the denoiser on every frame when enabled. The previous heuristic gated
    // it off after 64 static frames (banking on accumulator convergence to take
    // over), which produced a perceptible "denoiser pops off" transition once the
    // camera stopped moving. With blue-noise input + 5 passes the filter is
    // detail-preserving at convergence, so always-on is the cleaner trade.
    if (d.denoiserEnabled_ && d.aovMode_ == 0) {
        const uint32_t gx = (static_cast<uint32_t>(d.width_)  + 7u) / 8u;
        const uint32_t gy = (static_cast<uint32_t>(d.height_) + 7u) / 8u;

        AtrousGpuUniforms au{};
        au.frameCount = d.frameCount_;

        // Helper: run N-pass à-trous cascade on a channel.
        // Uses dstDenoised and tmpFiltered as ping-pong targets.
        // Input: srcAccum (raw accumulation), output ALWAYS lands in dstDenoised
        // regardless of `passes` parity. The trick: parity of (passes-1-p) makes
        // the final pass (p == passes-1) always write to dst.
        auto runAtrous = [&](WgpuTexture& srcAccum, WgpuTexture& dstDenoised,
                             WgpuTexture& tmpFiltered, uint32_t mode, int passes) {
            au.mode = mode;
            d.atrousPipeline.setTexture(3, *d.gBuf.write);
            d.atrousPipeline.setTexture(5, *d.hitMesh.read);

            WgpuTexture* readTex  = &srcAccum;
            for (int p = 0; p < passes; ++p) {
                WgpuTexture* writeTex = (((passes - 1 - p) % 2) == 0) ? &dstDenoised : &tmpFiltered;
                au.stepSize = static_cast<uint32_t>(1 << p);
                d.atrousUniBuf.write(&au, sizeof(au));
                d.atrousPipeline.setTexture(1, *readTex);
                d.atrousPipeline.setStorageTexture(2, *writeTex);
                d.atrousPipeline.dispatch(gx, gy);
                readTex = writeTex;
            }
        };

        // 4 atrous passes — step sizes 1,2,4,8 → effective 16-pixel radius.
        // Input: raw diff/spec accum directly. Scratch: filtered.a/b as ping-pong
        // targets (final pass always lands in denoisedDiff/Spec via parity trick).
        runAtrous(*d.diffAccum.read, d.denoisedDiff, d.filtered.a, 0, 4);
        runAtrous(*d.specAccum.read, d.denoisedSpec, d.filtered.b, 1, 4);

        displayDiff = &d.denoisedDiff;
        displaySpec = &d.denoisedSpec;
    }

    // --- TAAU: temporal upscale (active when pixelScale < 0.65, i.e. ≥1.5× upscale) ---
    if (d.pixelScale_ < 0.65f) {
        UpscaleGpuUniforms& uu = d.upscaleUBO;
        d.fillPrevCamUbo(uu.prevCamOri, uu.prevCamFwd, uu.prevCamRgt, uu.prevCamUp);
        uu.curCamOri[0] = camPos.x; uu.curCamOri[1] = camPos.y; uu.curCamOri[2] = camPos.z; uu.curCamOri[3] = 0.f;
        uu.curCamFwd[0] = fwd.x;    uu.curCamFwd[1] = fwd.y;    uu.curCamFwd[2] = fwd.z;    uu.curCamFwd[3] = 0.f;
        uu.curCamRgt[0] = rgt.x;    uu.curCamRgt[1] = rgt.y;    uu.curCamRgt[2] = rgt.z;    uu.curCamRgt[3] = 0.f;
        uu.curCamUp[0]  = up.x;     uu.curCamUp[1]  = up.y;     uu.curCamUp[2]  = up.z;     uu.curCamUp[3]  = 0.f;
        uu.iRes[0] = static_cast<float>(d.fullWidth_);
        uu.iRes[1] = static_cast<float>(d.fullHeight_);
        uu.iRes[2] = d.pixelScale_;
        uu.iRes[3] = 0.f;
        uu.tanHalfFov[0] = tanHalfFov;
        uu.tanHalfFov[1] = uu.tanHalfFov[2] = uu.tanHalfFov[3] = 0.f;
        uu.frameCount[0] = d.frameCount_;
        uu.frameCount[1] = uu.frameCount[2] = uu.frameCount[3] = 0.f;
        d.upscaleUniBuf.write(&uu, sizeof(uu));

        d.upscalePipeline.setTexture(1, *displayDiff);
        d.upscalePipeline.setTexture(2, *displaySpec);
        d.upscalePipeline.setTexture(3, *d.gBuf.write);
        d.upscalePipeline.setTexture(4, *d.upscale.read);
        d.upscalePipeline.setStorageTexture(5, *d.upscale.write);

        const int ugx = (d.fullWidth_  + 7) / 8;
        const int ugy = (d.fullHeight_ + 7) / 8;
        d.upscalePipeline.dispatch(ugx, ugy, 1);
        d.upscale.swap();

        d.displayMat->customTextures["upscaleTex"] = d.upscale.read;
    } else {
        d.displayMat->customTextures["upscaleTex"] = &d.zeroTex;
    }

    // Store camera for next frame's reprojection (must run every frame, not just when denoiser is active)
    d.prevCamOri_[0] = camPos.x; d.prevCamOri_[1] = camPos.y; d.prevCamOri_[2] = camPos.z;
    d.prevCamFwd_[0] = fwd.x;    d.prevCamFwd_[1] = fwd.y;    d.prevCamFwd_[2] = fwd.z;
    d.prevCamRgt_[0] = rgt.x;    d.prevCamRgt_[1] = rgt.y;    d.prevCamRgt_[2] = rgt.z;
    d.prevCamUp_[0]  = up.x;     d.prevCamUp_[1]  = up.y;     d.prevCamUp_[2]  = up.z;

    d.displayMat->customTextures["accumTex"] = displayTex;
    d.displayMat->customTextures["gBufTex"]  = d.gBuf.write;
    d.displayMat->customTextures["diffTex"]  = displayDiff;
    d.displayMat->customTextures["specTex"]  = displaySpec;
    d.displayMat->uniformsNeedUpdate = true;
    d.frameCount_ += 1.f;

    // Pass exposure via camera z (transform.cameraPos.z); always positive so the display quad
    // stays in front of the near plane.  AOV mode is encoded in _pad: _pad = pixelScale + aovMode*10,
    // allowing the display shader to extract both pixelScale and aovMode from a single float.
    d.displayCam.position.z = d.exposure_;
    d.renderer.setPixelRatioHint(d.pixelScale_ + static_cast<float>(d.aovMode_) * 10.f);

    // Blit to screen
    d.renderer.render(d.displayScene, d.displayCam);

    // Raster overlay: draw wireframe meshes and Line geometry on top.
    // Collect overlay objects and temporarily hide everything else so the
    // renderer only draws the overlay. Depth is cleared so overlays render
    // without being occluded by the blit quad's depth writes; color is loaded
    // so the path-traced image is preserved.
    {
        struct Entry { Object3D* obj; bool wasVisible; };
        std::vector<Entry> hidden;
        bool hasOverlay = false;

        // Skip traversal entirely when no overlay layer is configured and last frame
        // confirmed the scene has no wireframe/Line overlay objects.
        const bool mightHaveOverlay = (d.overlayLayer_ >= 0) || d.overlayFoundLastFrame_;
        if (mightHaveOverlay) {

        // Only toggle renderable objects (Mesh / Line), never containers/groups,
        // so parent nodes remain visible and their children are reachable.
        scene.traverse([&](Object3D& obj) {
            if (!obj.visible) return;
            const bool onOverlayLayer = (d.overlayLayer_ >= 0 &&
                                         obj.layers.isEnabled(static_cast<unsigned>(d.overlayLayer_)));
            if (auto* mesh = obj.as<Mesh>()) {
                auto* mat = mesh->material().get();
                auto* mww = dynamic_cast<MaterialWithWireframe*>(mat);
                bool isOverlay = onOverlayLayer ||
                                 (mww && mww->wireframe) ||
                                 dynamic_cast<LineBasicMaterial*>(mat) != nullptr;
                if (isOverlay) {
                    hasOverlay = true;
                } else {
                    hidden.push_back({mesh, true});
                    mesh->visible = false;
                }
            } else if (obj.is<Line>() || onOverlayLayer) {
                hasOverlay = true;
            }
        });

        } // end if (mightHaveOverlay)
        d.overlayFoundLastFrame_ = hasOverlay;

        if (hasOverlay) {
            // Reconstruct rasterizer depth from path-traced primary-hit t values.
            // This lets wireframe/line objects be correctly occluded by path-traced geometry.
            auto* encoder = static_cast<WGPUCommandEncoder>(d.renderer.nativeRenderCommandEncoder());
            auto* depthView = static_cast<WGPUTextureView>(d.renderer.nativeFrameDepthView());
            const uint32_t depthSamples = d.renderer.nativeFrameDepthSampleCount();
            d.ensureDepthFillPipeline(depthSamples);
            if (encoder && depthView && d.depthFillPipeline_) {
                // Upload depth-fill uniforms (projView + camera vectors)
                DepthFillUniforms dfu{};
                {
                    Matrix4 proj = camera.projectionMatrix;
                    // Apply the same NDC z remap as WgpuRenderer::render()
                    auto& e = proj.elements;
                    e[2]  = 0.5f * e[2]  + 0.5f * e[3];
                    e[6]  = 0.5f * e[6]  + 0.5f * e[7];
                    e[10] = 0.5f * e[10] + 0.5f * e[11];
                    e[14] = 0.5f * e[14] + 0.5f * e[15];
                    Matrix4 pv;
                    pv.multiplyMatrices(proj, camera.matrixWorldInverse);
                    const auto& pe = pv.elements;
                    for (int i = 0; i < 16; ++i) dfu.projView[i] = pe[i];
                }
                dfu.camOri[0] = camPos.x; dfu.camOri[1] = camPos.y; dfu.camOri[2] = camPos.z;
                dfu.camFwd[0] = fwd.x;    dfu.camFwd[1] = fwd.y;    dfu.camFwd[2] = fwd.z;
                dfu.camRgt[0] = rgt.x;    dfu.camRgt[1] = rgt.y;    dfu.camRgt[2] = rgt.z;
                dfu.camUp[0]  = up.x;     dfu.camUp[1]  = up.y;     dfu.camUp[2]  = up.z;
                dfu.iRes[0]   = static_cast<float>(d.width_);
                dfu.iRes[1]   = static_cast<float>(d.height_);
                dfu.tanHalfFov[0] = tanHalfFov;
                wgpuQueueWriteBuffer(d.queue, d.depthFillUniBuf_, 0, &dfu, sizeof(dfu));

                // Build bind group: uniform (0) + gBuf.write (1) — cached until resize
                if (!d.depthFillBG_) {
                    WGPUBindGroupEntry bgEntries[2]{};
                    bgEntries[0].binding = 0;
                    bgEntries[0].buffer  = d.depthFillUniBuf_;
                    bgEntries[0].offset  = 0;
                    bgEntries[0].size    = sizeof(DepthFillUniforms);
                    bgEntries[1].binding     = 1;
                    bgEntries[1].textureView = d.gBuf.write->view();
                    WGPUBindGroupDescriptor bgDesc{};
                    bgDesc.layout     = d.depthFillBGL_;
                    bgDesc.entryCount = 2;
                    bgDesc.entries    = bgEntries;
                    d.depthFillBG_ = wgpuDeviceCreateBindGroup(d.device, &bgDesc);
                }
                WGPUBindGroup bg = d.depthFillBG_;

                // Depth-fill render pass: loads existing depth, then overwrites with
                // reconstructed path-traced depth. No color attachment.
                WGPURenderPassDepthStencilAttachment depthAtt{};
                depthAtt.view            = depthView;
                depthAtt.depthLoadOp     = WGPULoadOp_Clear;
                depthAtt.depthStoreOp    = WGPUStoreOp_Store;
                depthAtt.depthClearValue = 1.0f;
                WGPURenderPassDescriptor passDesc{};
                passDesc.label                    = WGPUStringView{"depth_fill_pass", WGPU_STRLEN};
                passDesc.colorAttachmentCount     = 0;
                passDesc.depthStencilAttachment   = &depthAtt;
                WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
                wgpuRenderPassEncoderSetPipeline(pass, d.depthFillPipeline_);
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                wgpuRenderPassEncoderSetViewport(pass, 0.f, 0.f,
                    static_cast<float>(d.width_), static_cast<float>(d.height_), 0.f, 1.f);
                wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
                wgpuRenderPassEncoderEnd(pass);
                wgpuRenderPassEncoderRelease(pass);
                // depthFillBG_ is persistent — not released here
            }

            // Overlay render: preserve color AND depth (use reconstructed depth for occlusion).
            const bool savedAutoClear       = d.renderer.autoClear;
            const bool savedShadowAutoUpdate = d.renderer.shadowMapAutoUpdate;
            d.renderer.autoClear           = false;  // no color or depth clear
            d.renderer.shadowMapAutoUpdate = false;  // skip shadow re-render
            d.renderer.render(scene, camera);
            d.renderer.autoClear           = savedAutoClear;
            d.renderer.shadowMapAutoUpdate = savedShadowAutoUpdate;
        }

        for (auto& e : hidden) e.obj->visible = e.wasVisible;
    }
}

void WgpuPathTracer::setEnvIntensity(float intensity) {
    if (intensity != pimpl_->envIntensity_) {
        pimpl_->envIntensity_ = intensity;
        pimpl_->frameCount_ = 0.f;
    }
}

float WgpuPathTracer::envIntensity() const {
    return pimpl_->envIntensity_;
}

void WgpuPathTracer::setMaxBounces(int bounces) {
    bounces = std::max(1, std::min(bounces, 16));
    if (pimpl_->maxBounces_ != bounces) {
        pimpl_->maxBounces_ = bounces;
        pimpl_->frameCount_ = 0.f;
    }
}

int WgpuPathTracer::maxBounces() const {
    return pimpl_->maxBounces_;
}

void WgpuPathTracer::setExposure(float exposure) {
    pimpl_->exposure_ = exposure;
}

float WgpuPathTracer::exposure() const {
    return pimpl_->exposure_;
}

void WgpuPathTracer::setDenoiserEnabled(bool enabled) {
    if (pimpl_->denoiserEnabled_ == enabled) return;
    pimpl_->denoiserEnabled_ = enabled;
    pimpl_->frameCount_ = 0.f;  // jitter state changes — flush stale history
}

bool WgpuPathTracer::denoiserEnabled() const {
    return pimpl_->denoiserEnabled_;
}

void WgpuPathTracer::setReSTIREnabled(bool enabled) {
    pimpl_->frameCount_ = 0.f;  // flush stale reservoir before first temporal reuse
    pimpl_->restirEnabled_ = enabled;
}

bool WgpuPathTracer::restirEnabled() const {
    return pimpl_->restirEnabled_;
}

void WgpuPathTracer::setReSTIRGIEnabled(bool enabled) {
    pimpl_->frameCount_ = 0.f;
    pimpl_->restirGiEnabled_ = enabled;
}

bool WgpuPathTracer::restirGiEnabled() const {
    return pimpl_->restirGiEnabled_;
}

void WgpuPathTracer::setSamplesPerPixel(int spp) {
    pimpl_->spp_ = std::max(1, spp);
}

int WgpuPathTracer::samplesPerPixel() const {
    return pimpl_->spp_;
}

void WgpuPathTracer::setFireflyClamp(float cap) {
    pimpl_->fireflyCap_ = (cap > 0.f) ? cap : 1e30f;
}

float WgpuPathTracer::fireflyClamp() const {
    return pimpl_->fireflyCap_;
}

void WgpuPathTracer::setAOVMode(int mode) {
    mode = std::max(0, std::min(mode, 6));
    if (pimpl_->aovMode_ != mode) {
        pimpl_->aovMode_ = mode;
        pimpl_->frameCount_ = 0.f;
    }
}

int WgpuPathTracer::aovMode() const {
    return pimpl_->aovMode_;
}

void WgpuPathTracer::setSize(std::pair<int, int> size) {
    if (size.first <= 0 || size.second <= 0) return; // minimised — skip texture recreation
    pimpl_->fullWidth_ = size.first;
    pimpl_->fullHeight_ = size.second;
    const int sw = std::max(1, static_cast<int>(size.first * pimpl_->pixelScale_));
    const int sh = std::max(1, static_cast<int>(size.second * pimpl_->pixelScale_));
    pimpl_->recreateAccumTextures(sw, sh);
    pimpl_->recreateUpscaleTextures(size.first, size.second);
}

std::pair<int, int> WgpuPathTracer::size() const {
    return {pimpl_->fullWidth_, pimpl_->fullHeight_};
}

void WgpuPathTracer::setPixelScale(float scale) {
    scale = std::clamp(scale, 0.1f, 2.0f);
    if (scale == pimpl_->pixelScale_) return;
    pimpl_->pixelScale_ = scale;
    // Set renderer hint so the display shader reads pixelScale from transform._pad
    pimpl_->renderer.setPixelRatioHint(scale);
    if (pimpl_->fullWidth_ > 0 && pimpl_->fullHeight_ > 0) {
        const int sw = std::max(1, static_cast<int>(pimpl_->fullWidth_ * scale));
        const int sh = std::max(1, static_cast<int>(pimpl_->fullHeight_ * scale));
        pimpl_->recreateAccumTextures(sw, sh);
        pimpl_->resetUpscaleHistory();
    }
}

float WgpuPathTracer::pixelScale() const {
    return pimpl_->pixelScale_;
}

void WgpuPathTracer::setFoveatedRendering(bool enabled) {
    pimpl_->foveatedEnabled_ = enabled;
}

bool WgpuPathTracer::foveatedRendering() const {
    return pimpl_->foveatedEnabled_;
}

int WgpuPathTracer::frameCount() const {
    return static_cast<int>(pimpl_->frameCount_);
}

void WgpuPathTracer::resetAccumulation() {
    pimpl_->frameCount_ = 0.f;
}

void WgpuPathTracer::setOverlayLayer(int channel) {
    pimpl_->overlayLayer_ = channel;
}

int WgpuPathTracer::overlayLayer() const {
    return pimpl_->overlayLayer_;
}

void WgpuPathTracer::setTextureResolution(int size) {
    size = (size >= 2048) ? 2048 : 1024;
    pimpl_->textureResolution_ = size;
}

int WgpuPathTracer::textureResolution() const {
    return pimpl_->textureResolution_;
}

void WgpuPathTracer::markDirty() {
    pimpl_->prevMeshes.clear();
    pimpl_->prevEntryCount_ = 0;
}

void WgpuPathTracer::dispose() {
    auto& d = *pimpl_;
    if (d.depthFillBG_)         { wgpuBindGroupRelease(d.depthFillBG_);             d.depthFillBG_ = nullptr; }
    if (d.depthFillPipeline_)   { wgpuRenderPipelineRelease(d.depthFillPipeline_);   d.depthFillPipeline_ = nullptr; }
    if (d.depthFillPipeLayout_) { wgpuPipelineLayoutRelease(d.depthFillPipeLayout_); d.depthFillPipeLayout_ = nullptr; }
    if (d.depthFillBGL_)        { wgpuBindGroupLayoutRelease(d.depthFillBGL_);       d.depthFillBGL_ = nullptr; }
    if (d.depthFillShader_)     { wgpuShaderModuleRelease(d.depthFillShader_);       d.depthFillShader_ = nullptr; }
    if (d.depthFillUniBuf_)     { wgpuBufferRelease(d.depthFillUniBuf_);             d.depthFillUniBuf_ = nullptr; }
}
