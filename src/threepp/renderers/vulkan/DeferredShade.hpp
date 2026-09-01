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
            // Raster camera UBOs (binding 57, forward jittered VP + prevVP) —
            // cloud_march.comp's temporal reprojection reads rcam.prevVP.
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
            // Half-res RTAO (bindings 72-76) — the per-FIF AO+bent-normal image
            // (rgba16f) + its rgba16f aux (histLen/E[ao²]/near-field ao/skyVis),
            // each ping-ponged across frames-in-flight for the temporal
            // reprojection. Always bound (small); the pass is only dispatched
            // when AO is enabled.
            const VkImageView* rtao    = nullptr;// [framesInFlight] rgba16f half-res
            const VkImageView* rtaoAux = nullptr;// [framesInFlight] rgba16f half-res
            // ParticleField density volumes (bindings 67/68) — plan §3.3. A
            // FIXED-SIZE array of r32ui 3D images sampled by the froxel passes
            // through mediumExtinction, plus the small std140 UBO that says
            // where in the world each one sits. Always bound: unused slots get
            // the renderer's 1×1×1 dummy, so a scene with no dust is still a
            // complete descriptor set. Shared across views by construction —
            // the volume is world-anchored so K cameras cost one scatter.
            const VkImageView* particleDensity    = nullptr;// [kMaxDensityFields]
            // The r16f mirrors (binding 69), LINEAR-sampled by the shade's
            // per-pixel dust march. Same count and same dummy-fill contract.
            const VkImageView* particleDensityLin = nullptr;// [kMaxDensityFields]
            uint32_t           particleDensityCount = 0;    // == kMaxDensityFields
            const VkBuffer*    particleDensityUbo = nullptr;// [framesInFlight]
            // Splat reflection volumes (bindings 70/71) —
            // plans/splat-volume-reflections.md. A SECOND, parallel table to
            // the density one above: a fixed-size array of rgba16f 3D images
            // SplatPass baked once per cloud, LINEAR-sampled by svLeg on the
            // water/glass reflection legs, plus the std140 UBO carrying each
            // one's world→UVW matrix and conservative world AABB. Same
            // always-bound / dummy-fill contract, and world-anchored the same
            // way — the handles are shared by every view, and only the shade's
            // flags bit 12 differs per view (splats stay invisible to sensors).
            const VkImageView* splatVolume      = nullptr;// [kMaxSplatVolumes]
            uint32_t           splatVolumeCount = 0;      // == kMaxSplatVolumes
            const VkBuffer*    splatVolumeUbo   = nullptr;// [framesInFlight]
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

        // Everything the deferred-shade dispatch needs, filled by NAME at the
        // call site (a 24-argument positional list is a silent-swap trap).
        // The bools become pc.flags bits; the rest map 1:1 onto the shared
        // ShadePush block (vulkan_shared.h). The caller is responsible for
        // making the G-buffer visible to COMPUTE (the raster G-buffer render
        // pass declares a COMPUTE consumer dependency) and for the sceneHdr
        // write→read barrier (BloomPass::recordDispatch's leading barrier).
        struct DispatchParams {
            uint32_t width  = 0;// deferred render extent (== G-buffer extent)
            uint32_t height = 0;
            uint32_t envMipCount = 1;// PMREM mips — roughness→mip for specular IBL
            bool shadows = true;
            bool ao      = true;// RT env-visibility (AO/GI)
            uint32_t frameCounter  = 0;
            uint32_t emissiveCount = 0;
            float emissiveTotalPower = 0.f;
            float fireflyClamp       = 0.f;
            float oceanFineTileSize  = 0.f;
            float oceanFoamTileSize  = 0.f;
            bool denoise  = true;
            bool restirDI = false;
            bool volFog   = false;// volumetric spot-light beams
            float volDensity    = 0.f;
            float volAniso      = 0.f;
            float starIntensity = 0.f;
            // Camera WORLD motion this frame (m, radians) — the reflection
            // history policy needs it because a chase-cam surface (sunroof on
            // a followed car) is screen-stationary (motion vectors ~0) while
            // its view-dependent reflection content slides.
            float camDeltaLen = 0.f;
            float camRotAngle = 0.f;
            // Wall-clock seconds (frame-rate independent) — foam-noise drift.
            float timeSec = 0.f;
            // tan of the directional-light angular RADIUS — jitters the
            // primary sun-shadow ray within that cone (0 = hard 1-ray shadow).
            float sunTanHalfAngle = 0.f;
            // setGbufferMsaa's sample count (1/2/4) → pc.flags bits 5-6 so
            // dispatch A can weight complex pixels by their dominant-cluster
            // fraction; 1 = no weighting, MS G-buffer bindings unused.
            uint32_t gbufMsaaSamples = 1;
            // 0 = dispatch A (always). 1 = dispatch B, the MSAA per-sample
            // edge-shading pass — only when gbufMsaaSamples > 1, AFTER
            // dispatch A and a compute→compute barrier (B reads A's write).
            uint32_t shadeMode = 0;
            // Whether dispatch B WILL run this frame — dispatch A reserves
            // the geometry-minority coverage weight only then (flags bit 7);
            // otherwise it folds that weight into the dominant surface.
            bool shadeBActive = false;
            // # lights in the cluster buffer — the shade's analytic split
            // reads its cell's list when > 0 (the caller must have recorded
            // recordClusterBuild + a compute barrier first).
            uint32_t clusterLightCount = 0;
            bool froxelsActive = false;// froxel LUT valid this frame (flags bit 8)
            // Float-bits of the pre-exposure baked into every sceneHdr store
            // (physical camera keeps 100k lux in fp16). 0x3F800000 = legacy 1.
            uint32_t preExpBits = 0x3F800000u;
            // DISPLAY-referred solid background (the composite bypass restores
            // it verbatim) — the sky store skips the pre-exposure so sky and
            // geometry share one value domain (flags bit 10; without it any
            // DoF/bloom mixing into sky pixels is amplified 1/preExposure at
            // the bypass → white silhouette rims).
            bool bgIsSolidColor = false;
            // A ParticleField density volume is live this frame (flags bit 11).
            // Gates applyParticleFog, which reads the dust's integrated
            // transmittance out of the froxel LUT's .a channel — so on every
            // scene without dust the term is an early return, not merely a
            // multiply by 1.
            bool particleDensity = false;
            // A baked splat reflection volume is live this frame (flags bit 12).
            // Gates the svLeg marches on the water and glass reflection legs.
            //
            // PRIMARY VIEW ONLY. The descriptors are world-anchored and shared
            // across views harmlessly (exactly like the density table), but the
            // FLAG is what turns the march on, and doc/vulkan_splats.md's scope
            // wall says splats are invisible to the RT sensors and that
            // secondary views skip the pass rather than paint splats into an
            // AOV nobody asked for. A sensor's water reflection suddenly showing
            // a cloud would be a silent scope change — so the caller clears this
            // for every secondary view (VulkanRenderer.cpp, recordSceneDispatch).
            bool splatVolume = false;
        };
        // Dispatch the deferred shade over the render extent.
        void recordDispatch(VkCommandBuffer cb, uint32_t frame, const DispatchParams& p);

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

        // Half-resolution ray-traced ambient occlusion + bent normals
        // (rtao.comp). One dispatch over the HALF render extent that writes the
        // AO + bent-normal image the full-res shade upsamples for its ambient/
        // spec-occlusion + bent-normal probe/env diffuse indirect, with a
        // temporal reprojection EMA against the previous frame's result. Record
        // AFTER the probe-GI update (shares the TLAS) and BEFORE the shade
        // dispatch, with a compute→compute barrier after (the shade samples
        // rtao). Only dispatch when AO is enabled. width/height = the FULL
        // render extent (the shader derives half res).
        void recordRtao(VkCommandBuffer cb, uint32_t frame,
                        uint32_t width, uint32_t height,
                        uint32_t frameCounter);

        // Particle billboard lighting (particle_light.comp): one thread per
        // live overlay particle evaluates the deferred light field at the
        // particle's world center (sun × RT shadow × cloud shadow, clustered
        // point/spot lights, probe/env ambient) + the camera→particle fog leg,
        // writing 2×vec4 per particle into the caller-owned output SSBO the
        // billboard vertex shader reads. Record AFTER the shade dispatch (its
        // barriers already made the TLAS/cluster/froxel/cloud/probe inputs
        // compute-visible); the caller owns the output-write → vertex-read
        // barrier. ioSet = a set of particleIoLayout() {centers, lightOut}
        // written by the caller (buffers are the caller's, fixed at creation).
        void recordParticleLight(VkCommandBuffer cb, uint32_t frame,
                                 VkDescriptorSet ioSet,
                                 uint32_t count, uint32_t centerBase,
                                 uint32_t clusterLightCount,
                                 bool froxelsActive, uint32_t envMipCount);

        // Layout of the particle IO set (binding 0 = readonly centers SSBO,
        // binding 1 = writeonly light/fog result SSBO) — the caller allocates
        // and writes its sets from this.
        [[nodiscard]] VkDescriptorSetLayout particleIoLayout() const { return particleIoLayout_; }

        // Filter the demodulated lighting channels and COMPOSITE them into
        // sceneHdr. Two pipelines run back to back over the same descriptor set:
        //   • giFilterPipe_   — SVGF variance-guided à-trous over the demodulated
        //     diffuse GI (binding 16) + the co-filtered soft-shadow visibility
        //     ratio, then recombine blur(GI)·albedo + directU×R̃ into sceneHdr.
        //   • reflFilterPipe_ — separable roughness-guided gloss reconstruction
        //     of the 1-mirror-ray reflection (binding 25), then recombine.
        // Both recombines carry fog extinction, MSAA coverage weighting, and
        // pre-exposure. Run AFTER recordDispatch (same descriptor set); the
        // caller inserts a compute→compute barrier between them.
        // gbufMsaaSamples/shadeBActive mirror recordDispatch: the recombine
        // weights its GI/reflection adds by the geometry coverage at MSAA
        // complex pixels (must match how the shade pass split the weights).
        // preExpBits: the recombine's sceneHdr adds bake the same
        // pre-exposure the shade stored with (0x3F800000 = 1.0f = legacy).
        void recordFilterAndComposite(VkCommandBuffer cb, uint32_t frame,
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
        VkPipeline            giFilterPipe_ = VK_NULL_HANDLE;// SVGF GI + shadow-ratio filter + recombine
        VkPipeline            reflFilterPipe_ = VK_NULL_HANDLE;// reflection gloss reconstruction + recombine
        VkPipeline            clusterPipe_  = VK_NULL_HANDLE;// clustered light culling (cluster_build.comp)
        VkPipeline            froxelInjectPipe_    = VK_NULL_HANDLE;// froxel in-scatter (froxel_inject.comp)
        VkPipeline            froxelIntegratePipe_ = VK_NULL_HANDLE;// froxel LUT integrate (froxel_integrate.comp)
        VkPipeline            cloudMarchPipe_      = VK_NULL_HANDLE;// half-res cloud march (cloud_march.comp)
        VkPipeline            cloudShadowPipe_     = VK_NULL_HANDLE;// cloud shadow map (cloud_shadow.comp)
        VkPipeline            rtaoPipe_            = VK_NULL_HANDLE;// half-res ray-traced AO + bent normals (rtao.comp)
        // Particle billboard lighting (particle_light.comp) — set 0 is the
        // shared deferred set, set 1 the caller's particle IO buffers.
        VkDescriptorSetLayout particleIoLayout_    = VK_NULL_HANDLE;
        VkPipelineLayout      particlePipeLayout_  = VK_NULL_HANDLE;
        VkPipeline            particleLightPipe_   = VK_NULL_HANDLE;
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
