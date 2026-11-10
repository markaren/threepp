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
            bool     hsFix = false;   // Pierson-Moskowitz constants (DisplacedMesh::Params::hsFix)
            uint32_t spread = 0;      // 0 = symmetric cos^2, 1 = Donelan-Banner sech^2 (see the shader)
            // Optional swell system (see swellShape in the shader). Height 0 = none.
            float    swellHeight = 0.0f;  // significant height, m
            float    swellPeriod = 12.0f; // peak period Tp, s
            float    swellTheta  = 0.0f;  // direction of travel, rad (sample domain)
            float    swellSpread = 50.0f; // s of cos^(2s)(theta/2); below 1 is taken as 1
            // The measured short-wave spectrum (DisplacedMesh::Params::
            // shortWaveSpectrum, see elfouhailyShortCurvature in the shader).
            bool     shortWaveSpectrum = false;
        };

        // Push-constant block of phillips_spectrum.comp (std430, 56 bytes).
        struct PushParams {
            uint32_t textureSize;
            float    tileSize;
            float    windTheta;
            float    windSpeed;
            float    smallWaveCutoff;
            float    kMin;
            float    kMax;
            float    fetch;
            uint32_t hsFix;
            uint32_t spread;
            float    swellScale;
            float    swellPeriod;
            float    swellTheta;
            float    swellSpread;
            uint32_t shortWave;
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
        void updateSeaState(float windTheta, float windSpeed, float fetch, bool hsFix, uint32_t spread);

        // Live swell change; the next recordCompute regenerates h0 with it.
        void updateSwell(float height, float period, float theta, float spread);

        // Factor on the shader's swellShape that makes the swell's discrete
        // sum over this cascade's grid (its band and short-wave taper
        // included) carry exactly the requested significant height: a
        // swell is so narrow that it lands on a handful of modes, and a
        // continuous normalisation would miss Hs by tens of percent
        // depending on where the lattice falls. 0 when there is no swell or
        // its peak is not inside the band with at least two modes below it.
        static float swellScale(const Settings& s);

        // swellSpread as the shader gets it: at least 1.
        static float swellSpreadClamped(float s);

        // The band's mean wavenumber in Tayfun's (1980) sense, k_m = w_m^2 / g
        // with w_m = sum(w P) / sum(P) over this cascade's own k grid and band
        // (kMin, kMax, the small-wave taper): the k of the bound-wave harmonic
        // (DisplacedMesh::Params::boundWaves). A CPU copy of the shader's
        // phillips2D (spreading and swell included), evaluated for the current settings. 0 for an empty band.
        float meanWavenumber() const;

        // Params::shortWaveSpectrum: the slope variance (both axes, m^2/m^2)
        // of the short-wave spectrum that this cascade's grid does not draw,
        // what its roll-off exp(-(k l)^2) takes and everything past its
        // Nyquist (the capillary peak), integrated over ln k on the CPU.
        // Only for the open band (kMax == 0, the finest cascade): a lower
        // band's cut tail is the next cascade's. 0 when the switch is off.
        float unresolvedShortWaveMss() const;

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
        // ...and of its OCEAN_CHOP variant (16 bytes).
        struct ChopPushParams {
            uint32_t textureSize;
            float    tileSize;
            float    elapsedSeconds;
            float    taperSpacing;// m, the vertex grid's nominal spacing; 0 = no taper
        };

        // Chop normals (DisplacedMesh::Params::chopNormals): creates the slope
        // and jacDiag images and the OCEAN_CHOP variant's pipeline and set,
        // the first time only, so a mesh that never asks for the feature runs
        // exactly the plain pipeline, set and images. taperSpacing (m, may
        // change on later calls) tapers the derivative fields to the vertex
        // grid's Nyquist for the cascades that feed the vertex normal; 0 for
        // the per-pixel cascade (see dynamic_spectrum.comp).
        void enableChop(float taperSpacing);
        bool chopEnabled() const { return chopPipe_ != VK_NULL_HANDLE; }

        // chop (only once enableChop ran): the OCEAN_CHOP variant, which also
        // stores the slope spectrum i*k*h, the displacement gradient's
        // diagonal, and packs its off-diagonal into ht's imaginary part (see
        // dynamic_spectrum.comp). Otherwise the plain pass.
        void recordCompute(VkCommandBuffer cb, float elapsedSeconds, bool chop = false);

        const OceanImage& ht()           const { return ht_; }
        const OceanImage& displacement() const { return displacement_; }
        OceanImage&       slope()              { return slope_; }  // null until enableChop
        OceanImage&       jacDiag()            { return jacDiag_; }// null until enableChop

    private:
        vulkan::VulkanContext& ctx_;
        const PhillipsSpectrum& src_;
        uint32_t textureSize_;
        float    tileSize_;

        OceanImage  ht_, displacement_, slope_, jacDiag_;
        VkSampler   sampler_ = VK_NULL_HANDLE;

        VkDescriptorPool      pool_   = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl_    = VK_NULL_HANDLE;
        VkDescriptorSet       ds_     = VK_NULL_HANDLE;
        VkPipelineLayout      layout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_   = VK_NULL_HANDLE;

        // The OCEAN_CHOP variant (enableChop).
        VkDescriptorPool      chopPool_   = VK_NULL_HANDLE;
        VkDescriptorSetLayout chopDsl_    = VK_NULL_HANDLE;
        VkDescriptorSet       chopDs_     = VK_NULL_HANDLE;
        VkPipelineLayout      chopLayout_ = VK_NULL_HANDLE;
        VkPipeline            chopPipe_   = VK_NULL_HANDLE;
        float                 chopTaperSpacing_ = 0.f;

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
        // will be applied to over its life (a cascade has two, four with the
        // chop normals' slope and jacDiag; the wake field's ripples have one
        // a patch).
        IFFT(vulkan::VulkanContext& ctx, uint32_t textureSize, uint32_t maxGroups = 6);
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
        uint32_t maxGroups_ = 6;// 4 pairs/cascade (height, displacement, chop slope, chop jacDiag) + headroom
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

    // ─── ChopMoments ───────────────────────────────────────────────────
    // The fine cascade's per-pixel surface for the deferred shade while the
    // chop normals are on (DisplacedMesh::Params::chopNormals; built the
    // first time they are). One RGBA32F image with a full mip chain:
    //   .r = the height exactly as the IFFT left it (so every reader of the
    //        old fine-height binding, the path tracer included, sees the
    //        same value at LOD 0)
    //   .g, .b = the slope of the DISPLACED (choppy) surface, world units
    //   .a = its squared length, plus the variance the fold fade banked
    // Level L is the box average of 2^L x 2^L texels (linear blits), so one
    // filtered fetch over a pixel's footprint gives the mean slope (tilts N)
    // and E[s^2] - |E[s]|^2, the slope variance the pixel cannot hold (goes
    // to roughness): LEAN mapping, Olano and Baker 2010, "LEAN Mapping".
    class ChopMoments {
    public:
        ChopMoments(vulkan::VulkanContext& ctx, uint32_t textureSize,
                    VkImageView htView, VkImageView slopeView, VkImageView jacView);
        ~ChopMoments();

        ChopMoments(const ChopMoments&) = delete;
        ChopMoments& operator=(const ChopMoments&) = delete;

        struct PushParams {
            uint32_t textureSize;
            float    slopeScale;// raw IFFT slope -> world slope (waveScale / tile)
            float    chopScale; // raw IFFT dD/dx -> world, times lambda (choppiness / tile)
            float    bankedVar; // slope variance added to .a everywhere (shortWaveSpectrum); 0 = none
        };

        // After the cascade's IFFTs, same command buffer. Leaves every level
        // in GENERAL, visible to compute and ray-tracing sampled reads.
        void record(VkCommandBuffer cb, float slopeScale, float chopScale, VkPipelineStageFlags readerStages,
                    float bankedVar = 0.f);

        VkImageView sampledView() const { return sampledView_; }
        VkSampler   sampler()     const { return sampler_; }

    private:
        vulkan::VulkanContext& ctx_;
        uint32_t textureSize_;
        uint32_t mipLevels_ = 1;

        OceanImage  moments_;                      // .view = mip 0 storage view
        VkImageView sampledView_ = VK_NULL_HANDLE; // all levels
        VkSampler   sampler_     = VK_NULL_HANDLE; // linear, trilinear, repeat, anisotropic

        VkDescriptorPool      pool_   = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl_    = VK_NULL_HANDLE;
        VkDescriptorSet       ds_     = VK_NULL_HANDLE;
        VkPipelineLayout      layout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_   = VK_NULL_HANDLE;
    };

    // ─── FieldMips ─────────────────────────────────────────────────────
    // Mipped copies of a cascade's spatial fields (RG32F, one per source),
    // rebuilt every frame: level 0 is the field as the inverse FFT left it,
    // level L its box average over 2^L x 2^L texels. For short waves as
    // geometry (DisplacedMesh::Params::shortWaves): water_displace reads the
    // middle cascade at the level whose texel matches the vertex spacing
    // where the vertex is, so the mesh never carries a wave it cannot hold
    // (the far, thin part of a warped mesh would alias it into moire).
    class FieldMips {
    public:
        FieldMips(vulkan::VulkanContext& ctx, uint32_t size, std::vector<VkImage> sources);
        ~FieldMips();

        FieldMips(const FieldMips&) = delete;
        FieldMips& operator=(const FieldMips&) = delete;

        // After the cascade's IFFTs, same command buffer. Leaves every level
        // in GENERAL, visible to compute sampled reads.
        void record(VkCommandBuffer cb);

        VkImageView view(size_t i) const { return views_[i]; }// all levels of copy i
        VkSampler   sampler()      const { return sampler_; } // trilinear, repeat

    private:
        vulkan::VulkanContext& ctx_;
        uint32_t size_;
        uint32_t mipLevels_ = 1;
        std::vector<VkImage>     sources_;
        std::vector<OceanImage>  copies_;// .view unused
        std::vector<VkImageView> views_;
        VkSampler sampler_ = VK_NULL_HANDLE;
    };

    // ─── CrestFoam ─────────────────────────────────────────────────────
    // Crest foam (DisplacedMesh::Params::crestFoam; built the first time a
    // mesh turns it on): the foam texture over the water's rest coordinates
    // and the pass that deposits into it where a crest breaks and decays the
    // rest (foam_crest.comp says how and why). One RG16F image (.r the foam, .g
    // the compression of cascades 0 and 1 for Params::waveModulation), tileSize0
    // across, REPEAT, with a full mip chain rebuilt every frame by linear
    // blits, so the shade's footprint fetch over a pixel returns the covered
    // fraction of that footprint.
    class CrestFoam {
    public:
        // The views are cascade 0's and cascade 1's jacDiag and height images
        // (a mesh without cascade 1 passes cascade 0's twice; the mask gates it).
        CrestFoam(vulkan::VulkanContext& ctx, uint32_t resolution,
                  VkImageView jac0, VkImageView ht0, VkImageView jac1, VkImageView ht1,
                  // Cascade 1's sampler when its views are mipped copies
                  // (Params::foldFoam reads them at a level); null = single level.
                  VkSampler c1Sampler = VK_NULL_HANDLE,
                  // The wave climate's levels (DisplacedMesh::WaveClimate):
                  // RGBA16F and foam_crest.comp's climate variant, which keeps
                  // the foam of the sea scaled by 2/3 and 1/3 in .b and .a.
                  bool climateLevels = false);
        ~CrestFoam();

        CrestFoam(const CrestFoam&) = delete;
        CrestFoam& operator=(const CrestFoam&) = delete;

        // foam_crest.comp's push block, field for field (72 bytes).
        struct PushParams {
            VkDeviceAddress hullAddr;
            uint32_t        res;
            float           tileSize;
            float           tileSize1;
            float           steepScale;
            uint32_t        cascadeMask;
            uint32_t        hullCount;
            float           decay;
            float           threshold;
            float           invRamp;
            float           maxFoam;
            float           centerX;
            float           centerZ;
            uint32_t        model;// bit 0: Params::waveModulation, bit 1: Params::foldFoam, bit 2: Params::crestLineFoam
            float           sharpness = 0.f;// foldFoam: the drawn choppiness is 1 + this
            float           c1Lod     = 0.f;// foldFoam: cascade 1's mip level
            float           crestHalf = 0.f;// crestLineFoam: reach along the crest either way (m)
        };

        // After the cascades' IFFTs, same command buffer; fills p.res itself.
        // Leaves every level in GENERAL, visible to readerStages' sampled reads.
        void record(VkCommandBuffer cb, PushParams p, VkPipelineStageFlags readerStages);

        VkImageView sampledView() const { return sampledView_; }
        VkSampler   sampler()     const { return sampler_; }
        VkImage     image()       const { return foam_.image; }
        uint32_t    resolution()  const { return res_; }
        bool        climateLevels() const { return climateLevels_; }

    private:
        vulkan::VulkanContext& ctx_;
        uint32_t res_;
        uint32_t mipLevels_ = 1;
        bool     climateLevels_ = false;

        OceanImage  foam_;                         // .view = mip 0 storage view
        VkImageView sampledView_ = VK_NULL_HANDLE; // all levels
        VkSampler   sampler_     = VK_NULL_HANDLE; // trilinear, repeat, anisotropic (the shade's)
        VkSampler   fieldSampler_ = VK_NULL_HANDLE;// linear, repeat (the cascade fields)

        VkDescriptorPool      pool_   = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsl_    = VK_NULL_HANDLE;
        VkDescriptorSet       ds_     = VK_NULL_HANDLE;
        VkPipelineLayout      layout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_   = VK_NULL_HANDLE;
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
