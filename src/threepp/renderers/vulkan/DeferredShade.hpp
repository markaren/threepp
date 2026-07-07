// DeferredShade — deferred lighting pass used by VulkanRenderer.
//
// Reads the material G-buffer (normal+roughness, albedo+metalness, depth, IDs)
// produced by the raster prepass and shades a clean, analytic, noise-free base
// (direct analytic lights + split-sum specular IBL + approximate diffuse IBL)
// straight into the BloomPass sceneHdr image. The existing bloom + TAA tail
// then finishes the frame exactly as it does for the path-traced ReferencePT
// mode — only the surface-shading stage differs.
//
// This pass owns no images: it writes BloomPass::sceneHdrView(frame), the same
// linear-HDR target denoise.comp writes in ReferencePT. It uses its own
// focused descriptor layout (it does NOT touch the shared RT descriptor set),
// so it cannot regress the path tracer.

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
            // Scene fog (homogeneous medium) — the SAME per-frame UBO the PT path
            // consumes (GpuFogUbo: sigmaT/enabled/color/anisotropy/waterSurfaceY).
            const VkBuffer*    fogBuf     = nullptr;// [framesInFlight]
            VkDeviceSize       fogRange   = 0;
            // ReSTIR DI reservoir ping-pong — [2] PHYSICAL images (not per-frame): the
            // shared PT reservoir images. rewriteDescriptors picks write=slot(f&1),
            // read=other per frame, matching the RT set's ping-pong. STORAGE images.
            const VkImageView* reservoirPos = nullptr;// [2] lightPos.xyz + lightType.w (rgba32f)
            const VkImageView* reservoirW   = nullptr;// [2] W_sum/M/W/p_hat (rgba16f)
            VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;// shared scene TLAS (shadow + reflection rays)
            const VkBuffer*    materialBuf = nullptr;// [framesInFlight] MaterialDesc[] (emissive)
            VkBuffer           geomDescBuf = VK_NULL_HANDLE;// GeometryDesc[] (reflection-hit normals/UVs)
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
            // blueNoiseImage, also bound by the PT path) — dithers the stochastic
            // GI hemisphere directions. Sampled through gbufSampler_ (NEAREST is
            // correct: the shader computes exact texel-center UVs).
            VkImageView        blueNoise = VK_NULL_HANDLE;
            // World-space irradiance probe grid (ProbeGI, bindings 36/37/54).
            // Always valid — ProbeGI allocates all three at construction; the
            // grid UBO's `enabled` flag gates all sampling when probe GI is off.
            VkBuffer           probeShBuf    = VK_NULL_HANDLE;// SH-L1 store
            const VkBuffer*    probeGridUbo  = nullptr;       // [framesInFlight]
            VkBuffer           probeDepthBuf = VK_NULL_HANDLE;// Chebyshev depth store
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
        };
        void rewriteDescriptors(const DescriptorWriteInputs& in);

        // Rebind just the emissive-triangle buffer (binding 12) for one frame —
        // the emissive buffer grows in the per-frame path, so call this when it
        // reallocates (mirrors the RT rewriteEmissiveTriDescriptors).
        void rewriteEmissive(uint32_t frame, VkBuffer emissiveTriBuf);

        // Dispatch the deferred shade over the render extent. width/height =
        // path-trace render extent (== G-buffer extent). envMipCount drives the
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
                            bool froxelsActive = false);

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

        // Spatial denoise of the demodulated diffuse-indirect (binding 16) +
        // recombine into sceneHdr. Run AFTER recordDispatch (same descriptor
        // set); the caller inserts a compute→compute barrier between them.
        // gbufMsaaSamples/shadeBActive mirror recordDispatch: the recombine
        // weights its GI/reflection adds by the geometry coverage at MSAA
        // complex pixels (must match how the shade pass split the weights).
        void recordDenoiseDispatch(VkCommandBuffer cb, uint32_t frame,
                                   uint32_t width, uint32_t height,
                                   uint32_t gbufMsaaSamples = 1,
                                   bool shadeBActive = false);

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
        VkDescriptorPool      descPool_     = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]

        void createPipeline();
        void createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DEFERRED_SHADE_HPP
