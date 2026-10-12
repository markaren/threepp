// ProbeGI — world-space irradiance probe grid (DDGI-lite) for the deferred
// renderer's multi-bounce GI.
//
// Owns 32×16×32 grids of SH-L1 irradiance probes — ONE fitted to the scene
// AABB (cell-centered) while that keeps probes within kSceneFitMaxSpacing of
// each other, or kCascades camera-centred, toroidally addressed cascades at
// kCascadeBaseSpacing·4^c for a scene too large for one grid (a harbour, a
// fjord: the scene fit put one probe every 2.5 km there, and the deferred
// shade's indirect term had nothing to tap) — and the probe_update.comp
// compute pass that refreshes
// a round-robin window of probes each frame: 64 rays per probe via ray query,
// hits shaded with deferred_shade.comp's giRadiance logic (direct analytic
// lights + emissive NEE) plus a probe-grid feedback tap — the tap makes the
// cache converge to the multi-bounce fixed point over a few grid sweeps.
// deferred_shade.comp samples the same SH buffer at its GI-ray hit points
// (bindings 36/37), which is what lets enclosed interiors (the Sponza
// ground-floor corridors) receive courtyard light instead of rendering
// near-black off the 1-bounce gather. Each probe also carries an 8×8
// octahedral map of ray-hit distance moments (mean/mean²) that the sampler
// Chebyshev-tests per tap (DDGI visibility, Majercik et al. 2019) so probes
// on the far side of a wall can't leak irradiance through it.
//
// Follows the SkinningPipeline / EnvPrefilter house pattern: one class, own
// pipeline + descriptor pool, recordDispatch(cb, ...). Buffers exist from
// construction (the SH store is ~1 MB) so the deferred set can always bind
// them; the grid UBO's `enabled` flag gates all sampling when the feature is
// off (VulkanRenderer::setProbeGI, default ON).

#ifndef THREEPP_VULKAN_PROBE_GI_HPP
#define THREEPP_VULKAN_PROBE_GI_HPP

#include "threepp/renderers/vulkan/RenderGraph.hpp"
#include "threepp/renderers/vulkan/SpirvReflect.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;

    class ProbeGI {

    public:
        // Grid resolution (probes per axis) — 16384 probes, 64 B each = 1 MB.
        static constexpr uint32_t kDimX = 32, kDimY = 16, kDimZ = 32;
        static constexpr uint32_t kProbeCount = kDimX * kDimY * kDimZ;
        // Rays per probe — must match probe_update.comp's local_size_x.
        static constexpr uint32_t kRaysPerProbe = 64;
        // Chebyshev depth map: kDepthRes² octahedral texels per probe, each
        // packHalf2x16(mean, mean²) ray-hit distance. Must match
        // probe_common.glsl's kProbeDepthRes; texel count must equal
        // kRaysPerProbe (one thread per texel in the update's blend).
        static constexpr uint32_t kDepthRes    = 8;
        static constexpr uint32_t kDepthTexels = kDepthRes * kDepthRes;
        // Round-robin budget: the full grid refreshes every
        // kProbeCount / kProbesPerFrame = 8 frames. 2048 × 64 = 131 k rays
        // per frame — well inside the ~1 ms probe budget on an RTX 4070.
        static constexpr uint32_t kProbesPerFrame = 2048;
        // Camera-centred cascades (layout Cascades): up to kCascadesMax
        // windows of kProbeCount probes each, finest first, back to back in
        // the SH and depth stores. Spacing kCascadeBaseSpacing·4^c, so 1, 4
        // and 16 m by default: 32×16×32 m around the camera at 1 m, then
        // 128×64×128 m, then 512×256×512 m. The scene-fit grid is cascade 0
        // alone. Env overrides (read once at construction):
        //   THREEPP_VK_PROBE_CASCADES       cascade count 1..3; 0 = scene fit always
        //   THREEPP_VK_PROBE_CASCADE_BASE   finest spacing, metres
        //   THREEPP_VK_PROBE_SCENE_FIT_MAX  the scene fit's largest allowed spacing, metres
        static constexpr uint32_t kCascadesMax         = 3;
        static constexpr uint32_t kProbeCountAll       = kCascadesMax * kProbeCount;
        static constexpr float    kCascadeBaseSpacing  = 1.f;
        static constexpr float    kSceneFitMaxSpacing  = 2.f;
        enum class Layout { SceneFit, Cascades };

        ProbeGI(VulkanContext& ctx, uint32_t framesInFlight);
        ~ProbeGI();
        ProbeGI(const ProbeGI&) = delete;
        ProbeGI& operator=(const ProbeGI&) = delete;

        // Per-scene inputs — the subset of DeferredShade's set the probe rays
        // need (TLAS + lights + env + material/geometry/emissive buffers).
        // Call whenever rewriteDeferredDescriptors runs (TLAS rebuild, env
        // swap, buffer reallocation); idempotent.
        struct DescriptorWriteInputs {
            const VkBuffer* lightsUbo   = nullptr;// [framesInFlight]
            VkImageView     envView     = VK_NULL_HANDLE;// prefiltered PMREM
            VkSampler       envSampler  = VK_NULL_HANDLE;
            VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
            const VkBuffer* materialBuf = nullptr;// [framesInFlight] MaterialDesc[]
            const VkBuffer* geomDescBuf = nullptr;// [framesInFlight] GeometryDesc[] (ringed for auto-LOD level switches)
            const VkDescriptorImageInfo* materialTex = nullptr;// bindless array
            uint32_t        materialTexCount = 0;
            const VkBuffer* emissiveTriBuf = nullptr;// [framesInFlight] EmTri[]
        };
        // onlyFrame >= 0 rewrites just that frame-in-flight slot's set (kept in
        // lockstep with DeferredShade's per-FIF refresh); < 0 rewrites all slots.
        void rewriteDescriptors(const DescriptorWriteInputs& in, int onlyFrame = -1);

        // Rebind just the emissive-triangle buffer for one frame (the per-frame
        // buffer grows) — mirrors DeferredShade::rewriteEmissive.
        void rewriteEmissive(uint32_t frame, VkBuffer emissiveTriBuf);

        // Fit the grid to the scene's world AABB (probes at cell centers,
        // spacing = extent/dims per axis, min-clamped so a degenerate axis
        // doesn't collapse the grid). Schedules a zero-fill of the SH store
        // on the next recordDispatch — stale radiance from the previous fit
        // must not bleed into the new layout. A scene whose fit would space
        // probes more than kSceneFitMaxSpacing apart switches the layout to
        // camera-centred cascades instead (and back, when it shrinks).
        void setGridBounds(const float aabbMin[3], const float aabbMax[3]);
        [[nodiscard]] Layout   layout() const { return layout_; }
        [[nodiscard]] uint32_t cascadeCount() const { return cascadeCount_; }
        // Cascades follow the primary camera: once per frame, BEFORE
        // updateGridUbo, with the primary camera's world position. A window
        // moves by whole cells when the camera leaves its middle cell; the
        // slots the window keeps hold their world cells and their history, the
        // entering slots are reset by the next recordDispatch (the shader's
        // scroll mode). No-op for the scene-fit layout.
        void recentre(const float camPos[3]);
        // Zero the SH store, validity and history on the next dispatch, exactly
        // as a fresh grid fit does: probes bootstrap with alpha = 1 instead of
        // blending into whatever the store held. Part of the renderer's
        // resetTemporalHistory().
        void invalidateHistory() {
            needsClear_ = true;
            // The update cursor is temporal state as well: it advances by
            // kProbesPerFrame per frame, so its phase encodes how many frames
            // preceded the reset, and two captures that began after different
            // frame counts would update different probe slices on the same
            // window frame.
            probeOffset_ = 0;
            // Same argument for the update counter, and it is the stronger
            // half: it is the shader's `frame`, so it seeds every probe ray and
            // rotates the ray sphere. Left running, a reset grid rebuilt itself
            // from a different set of directions depending on how many frames
            // preceded the reset, and two runs of the same scene in one process
            // could not produce the same grid. setGridBounds already zeroed
            // both; this path zeroed only the cursor.
            updateCounter_ = 0;
            // The clear above already forces a full prev refresh; this keeps
            // the two facts from having to be reasoned about together.
            haveWindow_ = false;
            // Cascades are re-placed on the next recentre; the full clear
            // above supersedes any pending scroll reset.
            cascadesPlaced_ = false;
            scrollPending_  = false;
        }
        [[nodiscard]] bool gridFitted() const { return gridFitted_; }
        // The fitted grid, or with cascades the FINEST window: the first probe's
        // world position and the step between probes (window coordinate
        // (i, j, k) sits at origin + (i, j, k) · spacing).
        [[nodiscard]] const float* gridOrigin() const { return gridOrigin_; }
        [[nodiscard]] const float* gridSpacing() const { return gridSpacing_; }

        // Upload this frame's grid UBO (origin/spacing/dims + the sampling
        // enable). Call every frame BEFORE recording — deferred_shade reads
        // the same UBO at binding 37.
        void updateGridUbo(uint32_t frame, bool enabled);

        // Record the round-robin probe update (kProbesPerFrame workgroups ×
        // kRaysPerProbe rays). The caller provides the AS-build→compute
        // barrier before and the probe-write→shade-read barrier after.
        // fastBlend: the scene's lighting just stepped (a light switched);
        // this update's probes take the fast blend unconditionally.
        void recordDispatch(VkCommandBuffer cb, uint32_t frame,
                            uint32_t emissiveCount, float emissiveTotalPower,
                            bool shadows, uint32_t envMipCount, bool fastBlend = false);

        // For DeferredShade's descriptor write (bindings 36/37/54).
        // Render-graph declaration of recordDispatch: its descriptor set as
        // reflected from probe_update.comp, plus the SH and depth stores and
        // their snapshots, which it also clears and copies (transfer). Its
        // barriers between clear, snapshot and dispatch stay inside it.
        void declare(rg::RenderGraph& graph, rg::PassBuilder& pass, uint32_t frame) const;

        [[nodiscard]] VkBuffer shBuffer() const { return shBuf_.handle; }
        [[nodiscard]] const VkBuffer* gridUbos() const { return gridUboHandles_.data(); }
        [[nodiscard]] VkBuffer depthBuffer() const { return depthBuf_.handle; }

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        Buffer                shBuf_{};   // kProbeCount × 4 × vec4 SH-L1 store
        Buffer                depthBuf_{};// kProbeCount × kDepthTexels × uint Chebyshev depth
        // Start-of-dispatch snapshots (recordDispatch copies canonical → prev
        // before every update). The update's feedback tap / EMA history reads
        // go here, its writes to the canonical stores above — race-free, so
        // two same-seed runs produce bit-identical grids. Consumers keep
        // reading the canonical stores; no plumbing outside this class.
        Buffer                prevShBuf_{};
        Buffer                prevDepthBuf_{};
        std::vector<SpirvBinding> refl_;// probe_update.comp's bindings
        std::vector<Buffer>   gridUbos_;        // [framesInFlight]
        std::vector<VkBuffer> gridUboHandles_;  // handles view for DeferredShade

        VkDescriptorSetLayout dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout      pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool      descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]

        float    gridOrigin_[3]  = {0.f, 0.f, 0.f};
        float    gridSpacing_[3] = {1.f, 1.f, 1.f};
        bool     gridFitted_     = false;
        // Layout state. cascadeCount_ is 1 for the scene fit, whose window is
        // cascade 0 with base 0. For cascades: per window, the world position
        // of window coordinate (0,0,0), the spacing, the world cell of (0,0,0)
        // and that cell before the last scroll (== cBase_ once the scroll
        // reset has been recorded).
        Layout   layout_         = Layout::SceneFit;
        uint32_t cascadeCount_   = 1;
        float    cOrigin_[kCascadesMax][3]{};
        float    cSpacing_[kCascadesMax][3]{};
        int32_t  cBase_[kCascadesMax][3]{};
        int32_t  cPrevBase_[kCascadesMax][3]{};
        bool     cascadesPlaced_ = false;// recentre has placed the windows since the layout was chosen
        bool     scrollPending_  = false;// a window moved: the next recordDispatch resets the entering slots
        uint32_t cascadesWanted_ = kCascadesMax;// env THREEPP_VK_PROBE_CASCADES
        float    cascadeBase_    = kCascadeBaseSpacing;// env THREEPP_VK_PROBE_CASCADE_BASE
        float    sceneFitMax_    = kSceneFitMaxSpacing;// env THREEPP_VK_PROBE_SCENE_FIT_MAX
        bool     needsClear_     = true;// zero the SH store on first dispatch
        uint32_t probeOffset_    = 0;   // round-robin cursor
        uint32_t updateCounter_  = 0;   // rotates the per-probe ray sphere
        // The window the LAST dispatch wrote — the only place the canonical
        // stores can differ from the prev snapshots, and therefore the only
        // part the next dispatch's snapshot has to copy. haveWindow_ false
        // means "prev is not known to match": copy everything.
        uint32_t lastOffset_     = 0;
        uint32_t lastCount_      = 0;
        bool     haveWindow_     = false;

        void createPipeline();
        void createDescriptorPool();
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_PROBE_GI_HPP
