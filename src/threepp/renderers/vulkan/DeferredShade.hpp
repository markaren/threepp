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
            const VkImageView* sceneHdr   = nullptr;// [framesInFlight] output (storage)
            VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;// shared scene TLAS (shadow + reflection rays)
            const VkBuffer*    materialBuf = nullptr;// [framesInFlight] MaterialDesc[] (emissive)
            VkBuffer           geomDescBuf = VK_NULL_HANDLE;// GeometryDesc[] (reflection-hit normals/UVs)
            const VkDescriptorImageInfo* materialTex = nullptr;// bindless array (reflection-hit textures)
            uint32_t           materialTexCount = 0;          // == kMaxMaterialTextures
            const VkBuffer*    emissiveTriBuf = nullptr;// [framesInFlight] EmTri[] (emissive NEE)
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
        void recordDispatch(VkCommandBuffer cb, uint32_t frame,
                            uint32_t width, uint32_t height, uint32_t envMipCount,
                            bool shadows, bool ao, uint32_t frameCounter,
                            uint32_t emissiveCount, float emissiveTotalPower,
                            float fireflyClamp);

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        VkSampler             gbufSampler_  = VK_NULL_HANDLE;// nearest (texelFetch ignores it)
        VkDescriptorSetLayout dsLayout_     = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_   = VK_NULL_HANDLE;
        VkPipeline            pipe_         = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_     = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]

        void createPipeline();
        void createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_DEFERRED_SHADE_HPP
