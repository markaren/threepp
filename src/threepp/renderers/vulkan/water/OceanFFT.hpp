// FFT-based ocean simulation primitives for the Vulkan deferred renderer.
//
// PhillipsSpectrum   — Generates the static h0(k) initial spectrum (one-shot).
// DynamicSpectrum    — Time-evolves h0(k) into ht/dht/displacement (per frame).
// IFFT               — log2(N) horizontal + vertical butterfly + permute pass,
//                      transforms a frequency-domain RG32F image into spatial.
// OceanCascade       — Aggregate of one Phillips/Dynamic/IFFT chain plus
//                      spatial-domain output images consumed by the
//                      water_displace.comp pass.
//
// Each class owns its own VkImages, VkBuffers, descriptor sets, and
// pipelines. Construction is non-blocking (no GPU dispatch); callers
// invoke recordX(VkCommandBuffer) to enqueue per-frame work into an
// existing command buffer (typically the renderer's main frame buffer
// just before the BLAS rebuild dispatch).
//
// Private to the Vulkan renderer — included by VulkanRenderer.cpp only.

#ifndef THREEPP_VULKAN_OCEAN_FFT_HPP
#define THREEPP_VULKAN_OCEAN_FFT_HPP

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <vector>
#include <memory>

namespace threepp::vulkan {
    class VulkanContext;
}

namespace threepp::water {

    // Generic image / buffer pair with the bare-minimum bookkeeping
    // OceanFFT needs. Local copy of the same idea VulkanRenderer.cpp
    // uses internally (Buffer / Image2D) — kept self-contained so this
    // module doesn't depend on private renderer types.
    struct OceanImage {
        VkImage       image  = VK_NULL_HANDLE;
        VkImageView   view   = VK_NULL_HANDLE;
        VmaAllocation alloc  = VK_NULL_HANDLE;
        VkFormat      format = VK_FORMAT_UNDEFINED;
        uint32_t      width  = 0;
        uint32_t      height = 0;
        VkImageLayout currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct OceanBuffer {
        VkBuffer        handle  = VK_NULL_HANDLE;
        VmaAllocation   alloc   = VK_NULL_HANDLE;
        VkDeviceAddress address = 0;
        VkDeviceSize    size    = 0;
        void*           mapped  = nullptr;
    };

    // ─── PhillipsSpectrum ──────────────────────────────────────────────
    class PhillipsSpectrum {
    public:
        struct Settings {
            uint32_t textureSize = 256;
            float    tileSize    = 40.0f;
            float    windTheta   = 0.0f;
            float    windSpeed   = 12.0f;
            float    smallWaveCutoff = 0.01f;
            float    kMin = 0.0f;
            float    kMax = 0.0f;     // 0 = no upper bound
            float    fetch = 0.0f;    // m of upwind open water; 0 = fully developed (plain Phillips)
        };

        PhillipsSpectrum(vulkan::VulkanContext& ctx, const Settings& s);
        ~PhillipsSpectrum();

        PhillipsSpectrum(const PhillipsSpectrum&) = delete;
        PhillipsSpectrum& operator=(const PhillipsSpectrum&) = delete;

        // Records the one-shot dispatch that fills h0_. Call once after
        // construction, then use h0Image/h0View as a sampled input to
        // DynamicSpectrum.
        void recordCompute(VkCommandBuffer cb);

        // Live sea-state change (wind and/or fetch): rewrites the params UBO
        // so the next recordCompute regenerates h0 (caller re-dispatches). The
        // noise image persists, so successive regenerations are phase-
        // correlated and the sea state morphs smoothly. Same mapped-UBO-
        // rewrite convention as DynamicSpectrum's per-frame time update.
        void updateSeaState(float windTheta, float windSpeed, float fetch);

        VkImage     h0Image() const { return h0_.image; }
        VkImageView h0View()  const { return h0_.view;  }
        const OceanImage& h0() const { return h0_; }

        uint32_t textureSize() const { return settings_.textureSize; }
        const Settings& settings() const { return settings_; }

    private:
        vulkan::VulkanContext& ctx_;
        Settings  settings_;
        OceanImage h0_;
        OceanImage noise_;
        OceanBuffer paramsUbo_;
        VkSampler   sampler_      = VK_NULL_HANDLE;
        VkDescriptorPool      pool_   = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl_    = VK_NULL_HANDLE;
        VkDescriptorSet       ds_     = VK_NULL_HANDLE;
        VkPipelineLayout      layout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_   = VK_NULL_HANDLE;

        void createImages();
        void uploadNoise();
        void createPipeline();
        void writeParams();
    };

    // ─── DynamicSpectrum ───────────────────────────────────────────────
    // Outputs two RG32F images per dispatch:
    //   ht           — height-field spectrum (IFFT → spatial-domain height)
    //   displacement — packed horizontal-displacement spectrum (IFFT → (dx,dz))
    // (The gradient and Jacobian-diagonal spectra the WebTide reference also
    // emitted were computed here for a while but never consumed — normals come
    // from finite-differencing the spatial height, foam from finite-
    // differencing the spatial displacement — so they were dropped.)
    class DynamicSpectrum {
    public:
        DynamicSpectrum(vulkan::VulkanContext& ctx,
                        const PhillipsSpectrum& src,
                        uint32_t textureSize,
                        float tileSize);
        ~DynamicSpectrum();

        DynamicSpectrum(const DynamicSpectrum&) = delete;
        DynamicSpectrum& operator=(const DynamicSpectrum&) = delete;

        void recordCompute(VkCommandBuffer cb, float elapsedSeconds);

        const OceanImage& ht()           const { return ht_; }
        const OceanImage& displacement() const { return displacement_; }

    private:
        vulkan::VulkanContext& ctx_;
        const PhillipsSpectrum& src_;
        uint32_t textureSize_;
        float    tileSize_;

        OceanImage  ht_, displacement_;
        OceanBuffer paramsUbo_;
        VkSampler   sampler_ = VK_NULL_HANDLE;

        VkDescriptorPool      pool_   = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl_    = VK_NULL_HANDLE;
        VkDescriptorSet       ds_     = VK_NULL_HANDLE;
        VkPipelineLayout      layout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_   = VK_NULL_HANDLE;

        void createImages();
        void createPipeline();
    };

    // ─── IFFT ──────────────────────────────────────────────────────────
    // log2(N)+1 dispatches per direction. Caller provides the input image
    // (frequency-domain RG32F) and a scratch image (same format/size)
    // used for ping-pong. Writes the spatial-domain result back into the
    // input image.
    class IFFT {
    public:
        IFFT(vulkan::VulkanContext& ctx, uint32_t textureSize);
        ~IFFT();

        IFFT(const IFFT&) = delete;
        IFFT& operator=(const IFFT&) = delete;

        // Records: log2N horizontal butterflies + log2N vertical butterflies
        // + 1 permute pass. After this completes, `input` holds the spatial
        // domain image (sign-corrected). `scratch` contents are left garbage
        // (caller may share scratch across multiple cascades).
        //
        // Both images must be RG32F. Layout is handled here: each image is
        // transitioned to GENERAL based on its tracked currentLayout (first
        // use records the UNDEFINED→GENERAL barrier), and both end the call
        // in GENERAL. Callers must not force currentLayout beforehand — a
        // wrong claim of GENERAL suppresses that first-use barrier.
        void recordApply(VkCommandBuffer cb, OceanImage& input, OceanImage& scratch);

        uint32_t textureSize() const { return textureSize_; }

    private:
        vulkan::VulkanContext& ctx_;
        uint32_t textureSize_;
        uint32_t logSize_;

        OceanImage twiddle_;
        OceanBuffer paramsUbo_;     // unused (step index passed via push constant)
        VkSampler   sampler_ = VK_NULL_HANDLE;

        VkDescriptorPool      pool_       = VK_NULL_HANDLE;
        VkDescriptorSetLayout dslTwiddle_ = VK_NULL_HANDLE;  // for the precompute pass
        VkDescriptorSetLayout dslButterfly_ = VK_NULL_HANDLE; // for h/v passes
        VkDescriptorSetLayout dslPermute_ = VK_NULL_HANDLE;

        VkPipelineLayout      layoutTwiddle_   = VK_NULL_HANDLE;
        VkPipelineLayout      layoutButterfly_ = VK_NULL_HANDLE;
        VkPipelineLayout      layoutPermute_   = VK_NULL_HANDLE;

        VkPipeline pipeTwiddle_    = VK_NULL_HANDLE;
        VkPipeline pipeHorizontal_ = VK_NULL_HANDLE;
        VkPipeline pipeVertical_   = VK_NULL_HANDLE;
        VkPipeline pipePermute_    = VK_NULL_HANDLE;

        VkDescriptorSet dsTwiddle_ = VK_NULL_HANDLE;

        // Butterfly/permute sets are wired ONCE per distinct (input, scratch)
        // view pair and only *bound* afterwards. They are referenced by the
        // frame command buffer with N frames in flight — and by the earlier
        // height chain of the SAME frame when the displacement chain follows —
        // so rewriting a live set races both the pending frame and the
        // already-recorded one (VUID-vkUpdateDescriptorSets-None-03047).
        // Lifecycle: the cascade's images and its IFFT are created and
        // destroyed together (DisplacedMeshState), so cached views cannot
        // outlive the images they name.
        struct DescGroup {
            VkImageView input   = VK_NULL_HANDLE;
            VkImageView scratch = VK_NULL_HANDLE;
            std::array<VkDescriptorSet, 2> h{};// [0] reads input writes scratch, [1] the reverse
            std::array<VkDescriptorSet, 2> v{};
            std::array<VkDescriptorSet, 2> p{};// same orientation convention
        };
        static constexpr uint32_t kMaxDescGroups = 4;// 2 pairs/cascade (height, displacement) + headroom
        std::vector<DescGroup> groups_;

        bool twiddleComputed_ = false;

        void createTwiddleImage();
        void createPipelines();
        void recordTwiddleOnce(VkCommandBuffer cb);
        DescGroup& groupFor(const OceanImage& input, const OceanImage& scratch);
    };

    // ─── OceanCascade ──────────────────────────────────────────────────
    // Aggregate. One Phillips + Dynamic + IFFT + spatial-domain outputs
    // for a single wavenumber band. The full WebTide setup uses three.
    struct OceanCascade {
        std::unique_ptr<PhillipsSpectrum> phillips;
        std::unique_ptr<DynamicSpectrum>  dyn;
        std::unique_ptr<IFFT>             ifft;
        // Spatial-domain results — RG32F. height.r is the height value;
        // displacement.r/.g are dx/dz.
        OceanImage heightSpatial;
        OceanImage displacementSpatial;
        // Scratch used by the IFFT (RG32F, same dim).
        OceanImage scratch;
    };

}// namespace threepp::water

#endif//THREEPP_VULKAN_OCEAN_FFT_HPP
