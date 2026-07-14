// DeferredShade — deferred lighting pass used by VulkanRenderer.
//
// Reads the material G-buffer (normal+roughness, albedo+metalness, depth, IDs)
// produced by the raster prepass and shades a clean, analytic, noise-free base
// (direct analytic lights + split-sum specular IBL + approximate diffuse IBL)
// straight into the BloomPass sceneHdr image. The existing bloom + TAA tail
// then finishes the frame the same way regardless of the shading stage
// upstream of it — only the surface-shading stage differs between renderer
// configurations.
//
// This pass owns no images: it writes BloomPass::sceneHdrView(frame), the
// same linear-HDR target every shading stage writes into. It uses its own
// focused descriptor layout, keeping its footprint independent of the rest
// of the renderer.

#ifndef THREEPP_VULKAN_DEFERRED_SHADE_HPP
#define THREEPP_VULKAN_DEFERRED_SHADE_HPP

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class DeferredShade {

    public:
        DeferredShade(VulkanContext& ctx, uint32_t framesInFlight);
        ~DeferredShade();
        DeferredShade(const DeferredShade&) = delete;
        DeferredShade& operator=(const DeferredShade&) = delete;

        // Per-frame inputs. The UBO buffers are stable; the G-buffer views and
        // sceneHdr views change on resize and the env view/sampler change when
        // the scene environment is rebuilt — call rewriteDescriptors again in
        // any of those cases (idempotent).
        struct DescriptorWriteInputs {
            const VkBuffer*    cameraUbo  = nullptr;// [framesInFlight] viewInverse/projInverse
            const VkBuffer*    lightsUbo  = nullptr;// [framesInFlight] GpuLightsUbo (scalar)
            VkImageView        envView    = VK_NULL_HANDLE;// prefiltered PMREM mip chain
            VkSampler          envSampler = VK_NULL_HANDLE;
            const VkImageView* gbufNormal = nullptr;// [framesInFlight]
            const VkImageView* gbufDepth  = nullptr;// [framesInFlight] (depth-aspect view)
            const VkImageView* gbufIds    = nullptr;// [framesInFlight] (usampler2D)
            const VkImageView* gbufAlbedo = nullptr;// [framesInFlight]
            const VkImageView* gbufUv     = nullptr;// [framesInFlight] (.rg = UV, emissive-map sample)
            const VkImageView* gbufMotion = nullptr;// [framesInFlight] (.rg = NDC motion, GI reproject)
            const VkImageView* indirect   = nullptr;// [framesInFlight] GI accumulator / denoiser scratch (storage+sampled)
            const VkImageView* momentsSq  = nullptr;// [framesInFlight] SVGF E[L²] accumulator (storage+sampled, ping-pong like indirect)
            const VkImageView* atrousA    = nullptr;// [framesInFlight] SVGF à-trous ping-pong A (storage)
            const VkImageView* atrousB    = nullptr;// [framesInFlight] SVGF à-trous ping-pong B (storage)
            const VkImageView* reflect    = nullptr;// [framesInFlight] sharp mirror-ray reflection radiance (storage)
            const VkImageView* reflAux    = nullptr;// [framesInFlight] reflection-denoiser auxiliary (storage+sampled, ping-pong like reflect)
            // Denoised direct-shadow channel (bindings 43-47): shadow-ratio
            // accumulator (ping-ponged like indirect), the unshadowed analytic
            // direct sum the recombine multiplies by the filtered ratio, and
            // the ratio's own rg16f à-trous ping-pong pair.
            const VkImageView* shadowVis     = nullptr;// [framesInFlight] storage+sampled (prev fif read)
            const VkImageView* directU       = nullptr;// [framesInFlight] storage
            const VkImageView* shadowAtrousA = nullptr;// [framesInFlight] storage (rg16f)
            const VkImageView* shadowAtrousB = nullptr;// [framesInFlight] storage (rg16f)
            const VkImageView* sceneHdr   = nullptr;// [framesInFlight] output (storage)
            // Scene fog (homogeneous medium) — the per-frame UBO
            // (GpuFogUbo: sigmaT/enabled/color/anisotropy/waterSurfaceY).
            const VkBuffer*    fogBuf     = nullptr;// [framesInFlight]
            VkDeviceSize       fogRange   = 0;
            // ReSTIR DI reservoir ping-pong — [2] PHYSICAL images (not per-frame).
            // rewriteDescriptors picks write=slot(f&1), read=other per frame.
            // STORAGE images.
            const VkImageView* reservoirPos = nullptr;// [2] lightPos.xyz + lightType.w (rgba32f)
            const VkImageView* reservoirW   = nullptr;// [2] W_sum/M/W/p_hat (rgba16f)
            VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;// shared scene TLAS (shadow + reflection rays)
            const VkBuffer*    materialBuf = nullptr;// [framesInFlight] MaterialDesc[] (emissive)
            const VkBuffer*    geomDescBuf = nullptr;// [framesInFlight] GeometryDesc[] (reflection-hit normals/UVs; ringed for auto-LOD level switches)
            const VkDescriptorImageInfo* materialTex = nullptr;// bindless array (reflection-hit textures)
            uint32_t           materialTexCount = 0;          // == kMaxMaterialTextures
            const VkBuffer*    emissiveTriBuf = nullptr;// [framesInFlight] EmTri[] (emissive NEE)
            // Ocean textures (thin-shell water branch). Single shared handles —
            // 1×1 dummies when no DisplacedMesh is in the scene (the tile-size
            // push constants gate sampling). Mirror RT bindings 32 + 44.
            VkImageView        oceanFineView    = VK_NULL_HANDLE;// FFT fine-cascade height
            VkSampler          oceanFineSampler = VK_NULL_HANDLE;
            VkImageView        oceanFoamView    = VK_NULL_HANDLE;// world-space foam accumulator
            VkSampler          oceanFoamSampler = VK_NULL_HANDLE;
            // Baked tileable foam detail (R=bubbles, G=lace) — created once at
            // renderer startup, mipped, SHADER_READ_ONLY. Mirrors RT binding 45.
            VkImageView        foamDetailView    = VK_NULL_HANDLE;
            VkSampler          foamDetailSampler = VK_NULL_HANDLE;
            // 64×64 R8 void-and-cluster blue-noise tile (the renderer's shared
            // blueNoiseImage) — dithers the stochastic GI hemisphere directions.
            // Sampled through gbufSampler_ (NEAREST is correct: the shader
            // computes exact texel-center UVs).
            VkImageView        blueNoise = VK_NULL_HANDLE;
            // World-space irradiance probe grid (ProbeGI, bindings 36/37/54).
            // Always valid — ProbeGI allocates all three at construction; the
            // grid UBO's `enabled` flag gates all sampling when probe GI is off.
            VkBuffer           probeShBuf    = VK_NULL_HANDLE;// SH-L1 store
            const VkBuffer*    probeGridUbo  = nullptr;       // [framesInFlight]
            VkBuffer           probeDepthBuf = VK_NULL_HANDLE;// Chebyshev depth store
            // Hybrid SSR (bindings 55-57): the HiZ closest-depth pyramid +
            // its NEAREST/unclamped-LOD sampler (HiZPyramid), and the raster
            // camera UBOs (forward jittered VP for the screen-space march).
            // The prev-frame colour source is sceneHdr[other fif] — already
            // provided above. All always valid; sampling gates on flags bit 9.
            VkImageView        hizView      = VK_NULL_HANDLE;
            VkSampler          hizSampler   = VK_NULL_HANDLE;
            const VkBuffer*    rasterCamUbo = nullptr;        // [framesInFlight]
            // MSAA raw raster attachments (bindings 38-42) — only meaningful
            // when setGbufferMsaa > 1; pass a 1x1 dummy MS view/sampler set
            // otherwise (same "always bound, harmlessly unused" convention
            // as the ocean/foam dummies above). Consumed only by dispatch B
            // (shadeMode==1).
            const VkImageView* gbufNormalMS = nullptr;// [framesInFlight]
            const VkImageView* gbufDepthMS  = nullptr;// [framesInFlight]
            const VkImageView* gbufIdsMS    = nullptr;// [framesInFlight]
            const VkImageView* gbufAlbedoMS = nullptr;// [framesInFlight]
            const VkImageView* gbufUvMS     = nullptr;// [framesInFlight]
            // Clustered lights (bindings 48/49): per-cell index grid written
            // by recordClusterBuild + the full power-sorted point/spot light
            // list (GpuClusterLight[], no 8-per-type cap).
            const VkBuffer*    clusterGrid   = nullptr;// [framesInFlight] storage (device-local)
            const VkBuffer*    clusterLights = nullptr;// [framesInFlight] storage (host-visible)
            // Froxel volumetrics (bindings 50-53): the per-froxel in-scatter
            // accumulator (3D, ping-ponged across fif for the temporal EMA)
            // and the front-to-back-integrated LUT the shade samples.
            const VkImageView* froxelScatter = nullptr;// [framesInFlight] storage+sampled 3D
            const VkImageView* froxelLut     = nullptr;// [framesInFlight] storage+sampled 3D
            // Volumetric clouds (binding 58) — the per-frame GpuCloudUbo
            // (enabled/coverage/density/bottomY/topY/evolveSpeed/timeSec/wind).
            // Always bound (tiny); clouds.enabled == 0 makes the march a no-op.
            const VkBuffer*    cloudUbo   = nullptr;// [framesInFlight]
            VkDeviceSize       cloudRange = 0;
            // Half-res cloud march (bindings 59-63) — the per-FIF cloud
            // in-scatter/transmittance image (rgba16f) + its rg16f aux
            // (mean-depth/history), each ping-ponged across frames-in-flight
            // for the temporal reprojection. Always bound (small); the march
            // is only dispatched when clouds are enabled.
            const VkImageView* cloudColor = nullptr;// [framesInFlight] rgba16f half-res
            const VkImageView* cloudAux   = nullptr;// [framesInFlight] rg16f half-res
            // Cloud shadow map (bindings 64/65) — 512² R8 top-down cloud
            // transmittance, per FIF, regenerated each frame.
            const VkImageView* cloudShadow = nullptr;// [framesInFlight] r8 512²
        };
        // onlyFrame >= 0 rewrites just that frame-in-flight slot's set (the
        // caller has fence-proven that slot is idle — used by the per-FIF
        // deferred-descriptor refresh that replaced material-texture-swap
        // vkDeviceWaitIdle stalls). onlyFrame < 0 rewrites every slot (legal
        // only after a device drain: scene build, resize).
        void rewriteDescriptors(const DescriptorWriteInputs& in, int onlyFrame = -1);

        // Rebind just the emissive-triangle buffer (binding 12) for one frame —
        // the emissive buffer grows in the per-frame path, so call this when it
        // reallocates (mirrors the RT rewriteEmissiveTriDescriptors).
        void rewriteEmissive(uint32_t frame, VkBuffer emissiveTriBuf);

        // Dispatch the deferred shade over the render extent. width/height =
        // the deferred render extent (== G-buffer extent). envMipCount drives the
        // roughness→mip mapping for specular IBL. The caller is responsible for
        // making the G-buffer visible to COMPUTE (the raster G-buffer render
        // pass declares a COMPUTE consumer dependency) and for the sceneHdr
        // write→read barrier (BloomPass::recordDispatch's leading barrier).
        // camDeltaLen/camRotAngle: the camera's WORLD motion this frame (m,
        // radians) — the reflection history policy needs it because a chase-cam
        // surface (sunroof on a followed car) is screen-stationary (motion
        // vectors ~0) while its view-dependent reflection content slides.
        // timeSec: wall-clock seconds (frame-rate independent) — drives the
        // foam-noise drift so its speed doesn't scale with fps.
        // sunTanHalfAngle: tan of the directional-light angular RADIUS — jitters
        // the primary sun-shadow ray within that cone (0 = hard 1-ray shadow).
        // gbufMsaaSamples: setGbufferMsaa's sample count (1/2/4). Packed into
        // spare pc.flags bits (5-6) so dispatch A can weight complex pixels
        // by their dominant-cluster fraction and blend sky-minority coverage;
        // 1 = today's behaviour (no weighting, MS G-buffer bindings unused).
        // shadeMode: 0 = dispatch A (always). 1 = dispatch B, the MSAA
        // per-sample edge-shading pass — caller only issues this when
        // gbufMsaaSamples > 1, AFTER dispatch A and a compute->compute
        // barrier (dispatch B reads dispatch A's outImage write).
        // shadeBActive: whether dispatch B WILL run this frame — dispatch A
        // reserves the geometry-minority coverage weight only then (flags
        // bit 7); otherwise it folds that weight into the dominant surface.
        // clusterLightCount: # lights in the cluster buffer this frame — the
        // shade's analytic split reads its cell's list when > 0 (the caller
        // must have recorded recordClusterBuild + a compute barrier first).
        // ssrActive: HiZ pyramid built this frame + setSsrReflections on —
        // gates the shade's hybrid SSR fast path (flags bit 9).
        // preExpBits/prevPreExpBits: float-bits of the pre-exposure to bake
        // into every sceneHdr store (physical-camera mode; keeps 100k-lux
        // radiance inside fp16) and the PREV frame's factor the SSR
        // prev-scene fetch divides back out. 0x3F800000 = 1.0f = legacy.
        // bgIsSolidColor: the background is a DISPLAY-referred solid colour
        // (the composite bypass restores it verbatim) — the sky store skips
        // the pre-exposure so sky and geometry share one value domain
        // (flags bit 10; without it any DoF/bloom mixing into sky pixels is
        // amplified 1/preExposure at the bypass → white silhouette rims).
        void recordDispatch(VkCommandBuffer cb, uint32_t frame,
                            uint32_t width, uint32_t height, uint32_t envMipCount,
                            bool shadows, bool ao, uint32_t frameCounter,
                            uint32_t emissiveCount, float emissiveTotalPower,
                            float fireflyClamp,
                            float oceanFineTileSize, float oceanFoamTileSize,
                            bool denoise, bool restirDI, bool volFog,
                            float volDensity, float volAniso,
                            float starIntensity,
                            float camDeltaLen, float camRotAngle,
                            float timeSec, float sunTanHalfAngle,
                            uint32_t gbufMsaaSamples = 1, uint32_t shadeMode = 0,
                            bool shadeBActive = false,
                            uint32_t clusterLightCount = 0,
                            bool froxelsActive = false,
                            bool ssrActive = false,
                            uint32_t preExpBits = 0x3F800000u,
                            uint32_t prevPreExpBits = 0x3F800000u,
                            bool bgIsSolidColor = false);

        // Clustered light culling: one thread per cluster cell tests every
        // light's cull sphere against the cell's view-space AABB and writes
        // the per-cell index list (cluster_build.comp). Record BEFORE the
        // shade dispatch with a compute→compute barrier between them; skip
        // when lightCount == 0 (the shade's cluster loop is count-gated).
        void recordClusterBuild(VkCommandBuffer cb, uint32_t frame,
                                uint32_t lightCount,
                                uint32_t width, uint32_t height);

        // Froxel volumetric lighting: inject (per-froxel in-scatter — RT sun
        // shafts + clustered lights, temporal EMA) + integrate (front-to-back
        // LUT), with the inject→integrate barrier inside. Record AFTER
        // recordClusterBuild's barrier (inject reads the cluster grid) and
        // BEFORE the shade dispatch, with a compute→compute barrier after
        // (the shade samples the LUT). Only when the medium is active; pass
        // froxelsActive=true to recordDispatch the same frame.
        void recordFroxels(VkCommandBuffer cb, uint32_t frame,
                           uint32_t width, uint32_t height,
                           bool volFog, float volDensity, float volAniso,
                           uint32_t frameCounter,
                           float camDeltaLen, float camRotAngle,
                           uint32_t clusterLightCount);

        // Half-resolution volumetric cloud march (cloud_march.comp). One
        // dispatch over the HALF render extent that writes the cloud
        // in-scatter/transmittance image the full-res shade upsamples, with a
        // temporal reprojection EMA against the previous frame's result. Record
        // BEFORE the shade dispatch (and after the froxel barrier) with a
        // compute→compute barrier after (the shade samples cloudColor). Only
        // dispatch when clouds are enabled — off = free / image-identical.
        // width/height = the FULL render extent (the shader derives half res).
        void recordCloudMarch(VkCommandBuffer cb, uint32_t frame,
                              uint32_t width, uint32_t height, uint32_t envMipCount,
                              uint32_t frameCounter,
                              float camDeltaLen, float camRotAngle);

        // Cloud shadow map (cloud_shadow.comp): a 512² top-down cloud
        // transmittance field regenerated each frame. Record BEFORE the froxel
        // pass (the froxel sun term samples it) and the shade, with a
        // compute→compute barrier after. Only dispatch when clouds are on.
        void recordCloudShadow(VkCommandBuffer cb, uint32_t frame, uint32_t frameCounter);

        // Spatial denoise of the demodulated diffuse-indirect (binding 16) +
        // recombine into sceneHdr. Run AFTER recordDispatch (same descriptor
        // set); the caller inserts a compute→compute barrier between them.
        // gbufMsaaSamples/shadeBActive mirror recordDispatch: the recombine
        // weights its GI/reflection adds by the geometry coverage at MSAA
        // complex pixels (must match how the shade pass split the weights).
        // preExpBits: the recombine's sceneHdr adds bake the same
        // pre-exposure the shade stored with (0x3F800000 = 1.0f = legacy).
        void recordDenoiseDispatch(VkCommandBuffer cb, uint32_t frame,
                                   uint32_t width, uint32_t height,
                                   uint32_t gbufMsaaSamples = 1,
                                   bool shadeBActive = false,
                                   uint32_t preExpBits = 0x3F800000u);

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        VkSampler             gbufSampler_  = VK_NULL_HANDLE;// nearest (texelFetch ignores it)
        VkSampler             lutSampler_   = VK_NULL_HANDLE;// LINEAR clamp — froxel LUT trilinear sampling
        VkDescriptorSetLayout dsLayout_     = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_   = VK_NULL_HANDLE;
        VkPipeline            pipe_         = VK_NULL_HANDLE;
        VkPipeline            denoisePipe_  = VK_NULL_HANDLE;// spatial denoise + recombine
        VkPipeline            clusterPipe_  = VK_NULL_HANDLE;// clustered light culling (cluster_build.comp)
        VkPipeline            froxelInjectPipe_    = VK_NULL_HANDLE;// froxel in-scatter (froxel_inject.comp)
        VkPipeline            froxelIntegratePipe_ = VK_NULL_HANDLE;// froxel LUT integrate (froxel_integrate.comp)
        VkPipeline            cloudMarchPipe_      = VK_NULL_HANDLE;// half-res cloud march (cloud_march.comp)
        VkPipeline            cloudShadowPipe_     = VK_NULL_HANDLE;// cloud shadow map (cloud_shadow.comp)
        VkDescriptorPool      descPool_     = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]

        // Pool sizes derived from the descriptor-set-layout bindings in
        // createPipeline (summed per type × framesInFlight), consumed by
        // createDescriptorPool. Deriving them from the single binding table
        // means they can't desync from it — replaces the old hand-summed counts.
        std::vector<VkDescriptorPoolSize> poolSizes_;

        void createPipeline();
        void createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DEFERRED_SHADE_HPP
