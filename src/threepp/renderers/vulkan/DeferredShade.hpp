// DeferredShade — raster-first deferred lighting pass (RenderMode::RasterFirst).
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
        void recordDispatch(VkCommandBuffer cb, uint32_t frame,
                            uint32_t width, uint32_t height, uint32_t envMipCount,
                            bool shadows, bool ao, uint32_t frameCounter,
                            uint32_t emissiveCount, float emissiveTotalPower,
                            float fireflyClamp,
                            float oceanFineTileSize, float oceanFoamTileSize,
                            bool denoise, bool restirDI,
                            float volDensity, float volAniso,
                            float starIntensity,
                            float camDeltaLen, float camRotAngle,
                            float timeSec);

        // Spatial denoise of the demodulated diffuse-indirect (binding 16) +
        // recombine into sceneHdr. Run AFTER recordDispatch (same descriptor
        // set); the caller inserts a compute→compute barrier between them.
        void recordDenoiseDispatch(VkCommandBuffer cb, uint32_t frame,
                                   uint32_t width, uint32_t height);

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        VkSampler             gbufSampler_  = VK_NULL_HANDLE;// nearest (texelFetch ignores it)
        VkDescriptorSetLayout dsLayout_     = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_   = VK_NULL_HANDLE;
        VkPipeline            pipe_         = VK_NULL_HANDLE;
        VkPipeline            denoisePipe_  = VK_NULL_HANDLE;// spatial denoise + recombine
        VkDescriptorPool      descPool_     = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]

        void createPipeline();
        void createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DEFERRED_SHADE_HPP
