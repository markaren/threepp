// FFT-based ocean simulation primitives for the Vulkan deferred renderer.
//
// PhillipsSpectrum   — Generates the static h0(k) initial spectrum (one-shot).
// DynamicSpectrum    — Time-evolves h0(k) into ht/dht/displacement (per frame).
// IFFT               — log2(N) horizontal + vertical butterfly + permute pass,
//                      transforms a frequency-domain RG32F image into spatial.
//                      Path::Shared (the default; THREEPP_OCEAN_FFT) does the
//                      same transform in one rows and one columns dispatch,
//                      each line whole in workgroup shared memory.
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

        // Push-constant block of phillips_spectrum.comp (std430, 32 bytes).
        struct PushParams {
            uint32_t textureSize;
            float    tileSize;
            float    windTheta;
            float    windSpeed;
            float    smallWaveCutoff;
            float    kMin;
            float    kMax;
            float    fetch;
        };

        PhillipsSpectrum(vulkan::VulkanContext& ctx, const Settings& s);
        ~PhillipsSpectrum();

        PhillipsSpectrum(const PhillipsSpectrum&) = delete;
        PhillipsSpectrum& operator=(const PhillipsSpectrum&) = delete;

        // Records the one-shot dispatch that fills h0_. Call once after
        // construction, then use h0Image/h0View as a sampled input to
        // DynamicSpectrum.
        void recordCompute(VkCommandBuffer cb);

        // Live sea-state change (wind and/or fetch): the next recordCompute
        // regenerates h0 with it (caller re-dispatches). The noise image
        // persists, so successive regenerations are phase-correlated and the
        // sea state morphs smoothly. The params are push constants recorded
        // into the command buffer, so a frame in flight never sees the
        // change meant for the next one.
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
        VkSampler   sampler_      = VK_NULL_HANDLE;
        VkDescriptorPool      pool_   = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl_    = VK_NULL_HANDLE;
        VkDescriptorSet       ds_     = VK_NULL_HANDLE;
        VkPipelineLayout      layout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_   = VK_NULL_HANDLE;

        void createImages();
        void uploadNoise();
        void createPipeline();
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

        // Push-constant block of dynamic_spectrum.comp (std430, 12 bytes).
        struct PushParams {
            uint32_t textureSize;
            float    tileSize;
            float    elapsedSeconds;
        };

        void recordCompute(VkCommandBuffer cb, float elapsedSeconds);

        const OceanImage& ht()           const { return ht_; }
        const OceanImage& displacement() const { return displacement_; }

    private:
        vulkan::VulkanContext& ctx_;
        const PhillipsSpectrum& src_;
        uint32_t textureSize_;
        float    tileSize_;

        OceanImage  ht_, displacement_;
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
    // Caller provides the input image (frequency-domain RG32F) and a scratch
    // image (same format/size). Writes the spatial-domain result back into the
    // input image: out(x,y) = (-1)^(x+y) * sum over (k,l) of
    // in(k,l) * exp(+2*pi*i*(k*x + l*y)/N), unnormalised (no 1/N).
    //
    // Two implementations of that one transform, equal to float rounding:
    //   Butterfly  log2(N)+1 dispatches per direction, a pipeline barrier
    //              after each, ping-ponging through `scratch`.
    //   Shared     ifft_shared.comp: one dispatch for all rows and one for all
    //              columns, each row or column whole in workgroup shared
    //              memory, in place (`scratch` is not touched). Several images
    //              of the same size go in the same two dispatches
    //              (recordApplyBatch). Sizes it cannot hold (a 256-thread
    //              workgroup, or its shared memory: 10 KB at N = 512, 20 KB
    //              at 1024, past the device's limits) fall back to Butterfly.
    // THREEPP_OCEAN_FFT=butterfly|shared picks the default for every IFFT
    // (shared when unset); setPath overrides it per instance.
    class IFFT {
    public:
        enum class Path { Butterfly, Shared };
        // THREEPP_OCEAN_FFT, read once per process.
        static Path defaultPath();

        // The most images one recordApplyBatch takes (ifft_shared.comp's
        // image array). A cascade set has at most six, the wake ripples eight.
        static constexpr uint32_t kMaxBatch = 8;

        // maxGroups: how many distinct (input, scratch) image pairs this IFFT
        // will be applied to over its life (a cascade has two; the wake
        // field's ripples have one a patch).
        IFFT(vulkan::VulkanContext& ctx, uint32_t textureSize, uint32_t maxGroups = 4);
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

        // recordApply on images[i] for every bit i of `mask` (i < count <=
        // kMaxBatch), all textureSize square. On the Shared path that is one
        // rows and one columns dispatch for all of them, and the barriers end
        // the call as recordApply's do; on the Butterfly path it is
        // recordApply on each in turn. The images may belong to other IFFTs of
        // the same size (cascades that share a resolution). Every one of the
        // `count` images is bound whatever the mask, so a set of images whose
        // live subset changes frame to frame (the wake patches) costs one
        // descriptor set; a distinct set of views counts against maxGroups as
        // a distinct pair does. On the Butterfly path every live (image,
        // scratch) pair counts against maxGroups, other IFFTs' images
        // included: a call that would pass it throws before recording
        // anything. No image (count 0) or no live bit records nothing.
        void recordApplyBatch(VkCommandBuffer cb, OceanImage* const* images, uint32_t count,
                              OceanImage& scratch, uint32_t mask = ~0u);

        // Several batches, each through its own IFFT (the cascades' different
        // resolutions), with one barrier after all their rows and one after
        // all their columns. Exactly recordApplyBatch on each job in turn
        // unless every job's IFFT is on the Shared path; at most 8 jobs.
        struct Job {
            IFFT*              fft     = nullptr;
            OceanImage* const* images  = nullptr;
            uint32_t           count   = 0;
            OceanImage*        scratch = nullptr;
            uint32_t           mask    = ~0u;
        };
        static void recordApplyJobs(VkCommandBuffer cb, const Job* jobs, uint32_t jobCount);

        uint32_t textureSize() const { return textureSize_; }

        // The path this instance records. setPath(Shared) is ignored where
        // the shared path is not supported (see above). The shared pipelines
        // are built on the first setPath(Shared), the constructor's included,
        // so a THREEPP_OCEAN_FFT=butterfly run never builds them, and a
        // driver that rejects them leaves this IFFT on the butterfly path
        // (sharedSupported() then turns false) instead of failing it.
        Path path() const { return path_; }
        void setPath(Path p);
        bool sharedSupported() const { return sharedSupported_; }

    private:
        vulkan::VulkanContext& ctx_;
        uint32_t textureSize_;
        uint32_t logSize_;

        OceanImage twiddle_;
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
        uint32_t maxGroups_ = 4;// 2 pairs/cascade (height, displacement) + headroom
        std::vector<DescGroup> groups_;

        bool twiddleComputed_ = false;

        // The shared path: its two pipelines (rows, columns) share a layout of
        // the twiddle table + kMaxBatch storage images, and one set per
        // distinct list of image views, written once like the groups above.
        Path path_ = Path::Butterfly;
        bool sharedSupported_ = false;
        uint32_t sharedRows_ = 1;// lines a workgroup (ifft_shared.comp ROWS)
        VkDescriptorSetLayout dslShared_    = VK_NULL_HANDLE;
        VkPipelineLayout      layoutShared_ = VK_NULL_HANDLE;
        VkPipeline pipeSharedRows_ = VK_NULL_HANDLE;
        VkPipeline pipeSharedCols_ = VK_NULL_HANDLE;
        struct SharedSet {
            std::array<VkImageView, kMaxBatch> views{};
            uint32_t        count = 0;
            VkDescriptorSet ds    = VK_NULL_HANDLE;
        };
        std::vector<SharedSet> sharedSets_;

        void createTwiddleImage();
        void createPipelines();
        void checkSharedLimits();    // sharedSupported_ and sharedRows_ from the device limits
        bool ensureSharedPipelines();// builds them once; false (and unsupported) if the driver refuses
        void buildSharedPipelines(); // throws on a refused create; ensureSharedPipelines catches
        void destroySharedPipelines();
        void recordTwiddleOnce(VkCommandBuffer cb);
        DescGroup& groupFor(const OceanImage& input, const OceanImage& scratch);
        // How many of these live pairs have no group yet (recordApplyBatch's
        // up-front check against maxGroups on the Butterfly path).
        uint32_t newGroupsFor(OceanImage* const* images, uint32_t count, const OceanImage& scratch,
                              uint32_t mask) const;
        void checkButterflyGroups(OceanImage* const* images, uint32_t count, const OceanImage& scratch,
                                  uint32_t mask) const;
        VkDescriptorSet sharedSetFor(OceanImage* const* images, uint32_t count);
        // Twiddle, set, layouts and z slots for one batch; returns its layer count.
        uint32_t prepareShared(VkCommandBuffer cb, OceanImage* const* images, uint32_t count, uint32_t mask,
                               VkDescriptorSet& ds, uint32_t& slots);
        void dispatchShared(VkCommandBuffer cb, VkDescriptorSet ds, uint32_t slots, uint32_t layers,
                            bool columns) const;
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
