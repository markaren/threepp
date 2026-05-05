// FFT-based ocean simulation primitives for the Vulkan path tracer.
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
        };

        PhillipsSpectrum(vulkan::VulkanContext& ctx, const Settings& s);
        ~PhillipsSpectrum();

        PhillipsSpectrum(const PhillipsSpectrum&) = delete;
        PhillipsSpectrum& operator=(const PhillipsSpectrum&) = delete;

        // Records the one-shot dispatch that fills h0_. Call once after
        // construction, then use h0Image/h0View as a sampled input to
        // DynamicSpectrum.
        void recordCompute(VkCommandBuffer cb);

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
    // Outputs four RG32F images per dispatch:
    //   ht           — height-field spectrum (IFFT → spatial-domain height)
    //   dht          — gradient spectrum
    //   displacement — packed horizontal-displacement spectrum (IFFT → (dx,dz))
    //   jacDiag      — Jacobian diagonal spectrum (foam input)
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
        const OceanImage& dht()          const { return dht_; }
        const OceanImage& displacement() const { return displacement_; }
        const OceanImage& jacDiag()      const { return jacDiag_; }

    private:
        vulkan::VulkanContext& ctx_;
        const PhillipsSpectrum& src_;
        uint32_t textureSize_;
        float    tileSize_;

        OceanImage  ht_, dht_, displacement_, jacDiag_;
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
        // domain image (sign-corrected). `scratch` is left in an undefined
        // state (caller may share scratch across multiple cascades).
        //
        // Both images must be RG32F, both must be in GENERAL layout already
        // (this method does not transition them).
        void recordApply(VkCommandBuffer cb, OceanImage& input, OceanImage& scratch);

        uint32_t textureSize() const { return textureSize_; }

    private:
        vulkan::VulkanContext& ctx_;
        uint32_t textureSize_;
        uint32_t logSize_;

        OceanImage twiddle_;
        OceanBuffer paramsUbo_;     // 16 bytes; rewritten per pass
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

        // Per-step descriptor set ring. We need 2*logSize ping-pong
        // configurations (horizontal then vertical), each binding two
        // images either as A→B or B→A. The permute pass needs one more.
        std::vector<VkDescriptorSet> dsHorizontal_; // 2 entries (ping/pong)
        std::vector<VkDescriptorSet> dsVertical_;   // 2 entries
        VkDescriptorSet dsTwiddle_ = VK_NULL_HANDLE;
        VkDescriptorSet dsPermute_[2]{}; // [0]: scratch->input, [1]: input->input (after even count)

        bool twiddleComputed_ = false;
        OceanImage*   prevInput_   = nullptr;
        OceanImage*   prevScratch_ = nullptr;

        void createTwiddleImage();
        void createPipelines();
        void recordTwiddleOnce(VkCommandBuffer cb);
        void rebindDescriptorSets(OceanImage& a, OceanImage& b);
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
