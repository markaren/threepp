// SmokeGrid — the device-side Eulerian smoke solver behind ParticleField::GridSmoke.
//
// One SmokeGrid is one field's solver state: the scalars (smoke, temperature,
// fuel, oxygen used) and pressure on a res^3 grid, the velocity on the faces
// between those cells ((res+1)^3, see shaders/smoke_common.glsl), each a
// ping-pong pair, one scratch volume, the per-frame parameter block, and eight
// descriptor sets — one per parity of the three pairs, written once at creation
// and never again, so the solver costs no descriptor traffic. SmokePipelines is
// the part that does not depend on a field: the compute pipelines, their
// layouts and the sampler.
//
// A frame is `plan()` (host: how many fixed steps are owed, the parameter block
// and the push constants, all by value) and later `record()` (device: those
// steps, then the pass that writes the density volume in the Q20.12 fixed point
// the particle scatter uses, so the r16f convert after it is shared code, and
// the temperature volume the glow reads).
// The render graph does not see the solver itself: those images are private to
// this block and live in VK_IMAGE_LAYOUT_GENERAL, ordered by plain memory
// barriers. The two images it writes for the rest of the frame — the field's
// r32ui density volume and the temperature volume — are declared by
// ParticleFieldPass, the first exactly as it is for a scatter.
//
// A burning field also lights its surroundings: `recordGlow()` sums its glow
// over each octant of the box into a small host-visible block per frame in
// flight (shaders/smoke_glow.comp), and `takeGlow()` reads a block back once
// its frame's fence has signalled, never waiting for the device. ParticleFieldPass
// makes the octants point lights.

#ifndef THREEPP_VULKAN_SMOKEGRID_HPP
#define THREEPP_VULKAN_SMOKEGRID_HPP

#include "VulkanImplCommon.hpp"
#include "VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <memory>

namespace threepp {
    class ParticleField;
}

namespace threepp::vulkan {

    class VulkanContext;

    // Mirrors `Pc` in shaders/smoke_common.glsl.
    struct SmokePc {
        std::int32_t res[4];
        float        invCell[4];
        float        box[4];
        float        phys0[4];
        float        phys1[4];
        float        wind[4];
        float        turb[4];// x eddy speed (m/s, rms), y eddy size (m)
    };
    static_assert(sizeof(SmokePc) == 112, "SmokePc layout drift");

    // How many sources and colliders a field may carry (the rest are ignored). A labyrinth's walls are colliders.
    inline constexpr std::uint32_t kSmokeMaxSources   = 16;
    inline constexpr std::uint32_t kSmokeMaxColliders = 64;
    inline constexpr std::uint32_t kSmokeMaxBoxes     = 128;

    // Mirrors `Params` in shaders/smoke_common.glsl (std430).
    struct SmokeGpuSource {
        float posRadius[4];
        float rates[4];// x density / s, y temperature / s, z velocity blend / s, w fuel / s
        float vel[4];
        float force[4];// xyz m/s^2 at the centre: the force over the mass of air the weight spreads it on; w unused
    };
    struct SmokeGpuCollider {
        float a[4];
        float b[4];
        float vel[4];
    };
    struct SmokeGpuBox {
        float lo[4];
        float hi[4];
        float vel[4];
    };
    struct SmokeGpuParams {
        float            head[4];// x sources, y capsule colliders, z time, w boxes
        // The fire (ParticleField::GridSmoke): x ignition temperature, y burn rate, z heat release, w soot yield;
        // then x oxygen per fuel, y expansion, z kelvin per temperature unit, w the ambient (K). Here and not in the
        // push constants, which are 16 bytes short of the 128 every device guarantees.
        float            burn[8];
        SmokeGpuSource   src[kSmokeMaxSources];
        SmokeGpuCollider col[kSmokeMaxColliders];
        SmokeGpuBox      box[kSmokeMaxBoxes];
    };
    static_assert(sizeof(SmokeGpuParams) == 16 + 32 + kSmokeMaxSources * 64 + kSmokeMaxColliders * 48 + kSmokeMaxBoxes * 48, "SmokeGpuParams layout drift");

    // Mirrors `Pc` in shaders/smoke_glow.comp.
    struct SmokeGlowPc {
        std::int32_t  res[4]; // xyz cells per axis, w the pass (0 the layers' partial sums, 1 the octants)
        float         cell[4];// xyz cell size (m), w sigma_t in Q20.12 per unit density (the write pass's)
        float         box[4]; // xyz world minimum of the grid
        std::uint64_t partials;
        std::uint64_t octants;
    };
    static_assert(sizeof(SmokeGlowPc) == 64, "SmokeGlowPc layout drift");

    // One frame's light of a burning field, as shaders/smoke_glow.comp leaves it: per octant of the box (bit 0 the
    // upper half along x, bit 1 along y, bit 2 along z), its glow summed over the cells. Per unit of
    // DensityRepr::emissiveIntensity, which the host multiplies in.
    struct SmokeGlowOctant {
        float intensity[3];// the cells' emission times their volume (W/sr per unit emissiveIntensity)
        float luminance;   // the same in luminance: the weight of the centroid
        float centroid[3]; // world, the glow's luminance-weighted centre (the octant's centre when it is dark)
        float spread;      // m, the glow's rms distance from the centroid
    };
    struct SmokeGlow {
        SmokeGlowOctant octant[8];
    };
    static_assert(sizeof(SmokeGlow) == 256, "SmokeGlow layout drift");

    // One frame of one field, resolved on the host so recording reads no field.
    struct SmokePlan {
        std::uint32_t  steps      = 0;
        std::uint32_t  iterations = 24;
        bool           reset      = false;
        SmokePc        pc{};
        SmokeGpuParams params{};
        // The fire's light this frame: whether to sum the glow (the field burns, and its light is wanted), into
        // which frame-in-flight slot, and for which frame serial.
        bool           glow     = false;
        std::uint32_t  glowSlot = 0;
        std::uint64_t  glowSerial = 0;
        SmokeGlowPc    glowPc{};
    };

    class SmokePipelines {

    public:
        SmokePipelines(VulkanContext& ctx, VkDescriptorSetLayout densitySetLayout);
        ~SmokePipelines();
        SmokePipelines(const SmokePipelines&)            = delete;
        SmokePipelines& operator=(const SmokePipelines&) = delete;

        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout      layout      = VK_NULL_HANDLE;// set 0
        VkPipelineLayout      writeLayout = VK_NULL_HANDLE;// set 0 + the field's density set (set 1)
        VkPipelineLayout      glowLayout  = VK_NULL_HANDLE;// set 0, SmokeGlowPc
        VkSampler             samplerBorder = VK_NULL_HANDLE;
        VkPipeline            forces = VK_NULL_HANDLE, curl = VK_NULL_HANDLE, confine = VK_NULL_HANDLE,
                   advect = VK_NULL_HANDLE, advectFix = VK_NULL_HANDLE, diverge = VK_NULL_HANDLE,
                   jacobi = VK_NULL_HANDLE, project = VK_NULL_HANDLE, write = VK_NULL_HANDLE, solid = VK_NULL_HANDLE,
                   glow = VK_NULL_HANDLE;

    private:
        VulkanContext& ctx_;
    };

    class SmokeGrid {

    public:
        using CreateImage3DFn = std::function<Image2D(std::uint32_t, std::uint32_t, std::uint32_t, VkFormat,
                                                      VkImageUsageFlags, const char*)>;

        // Null when the images or sets could not be created.
        static std::unique_ptr<SmokeGrid> create(VulkanContext& ctx, const SmokePipelines& pl,
                                                 const CreateImage3DFn& createImage3D, std::uint32_t res);
        ~SmokeGrid();
        SmokeGrid(const SmokeGrid&)            = delete;
        SmokeGrid& operator=(const SmokeGrid&) = delete;

        [[nodiscard]] std::uint32_t resolution() const { return res_; }

        // The temperature the glow sees, kelvin, r16f res^3: written by the last pass of every frame (all of it, so
        // its contents are dead between frames) and sampled by the shade's march for the emission (binding 99,
        // particle_density.glsl). Same lifetime as this grid.
        [[nodiscard]] VkImage     temperatureImage() const { return temperature_.image; }
        [[nodiscard]] VkImageView temperatureView() const { return temperature_.view; }

        // Advance the field's clock and pack this frame's parameters. `frame` is the frame-in-flight slot and
        // `serial` the frame serial being prepared: the glow, when the field burns, goes to that slot.
        SmokePlan plan(const ParticleField& field, std::uint32_t frame, std::uint64_t serial);

        // Run the owed steps and write the density volume (set 1 of the write pass) and the temperature volume.
        void record(VkCommandBuffer cb, const SmokePipelines& pl, const SmokePlan& plan, VkDescriptorSet densitySet);

        // The fire's light: sum the glow record() wrote into the plan's slot (shaders/smoke_glow.comp), and make it
        // visible to the host. After record(), behind the barrier that follows the write pass. No-op unless the plan
        // asks for it.
        void recordGlow(VkCommandBuffer cb, const SmokePipelines& pl, const SmokePlan& plan);

        // The glow `frame`'s slot holds, when the frame that used the slot last summed one and that frame is
        // `serial` - kFramesInFlight: called in the frame's post-fence window, before this frame records into the
        // same slot, so it reads a finished sum and never waits. Each sum is taken once.
        bool takeGlow(std::uint32_t frame, std::uint64_t serial, SmokeGlow& out);

        // TEST/AUDIT: the images holding the state the last recorded step left (the "in" half of each pair).
        // Valid to copy once the device has drained. See VulkanRenderer::readGridSmoke.
        struct Images {
            VkImage velocity = VK_NULL_HANDLE;// rgba16f, (res+1)^3, xyz m/s on the min faces, relative to the wind;
                                              // w = the expansion (1/s) the last step's burn released in that cell
            VkImage scalars  = VK_NULL_HANDLE;// rgba16f, density, temperature, fuel, oxygen deficit
            VkImage pressure = VK_NULL_HANDLE;// r32f
            VkImage scratch  = VK_NULL_HANDLE;// rgba16f, .x = what the last step's pressure solve removed: the
                                              // divergence before its projection, less the expansion
            VkImage solid    = VK_NULL_HANDLE;// rgba16f, xyz the solid's velocity, w covered
        };
        [[nodiscard]] Images latest() const {
            return {vel_[v_].image, sca_[s_].image, p_[p_in_].image, aux_.image, solid_.image};
        }
        // False until the first frame has cleared the images.
        [[nodiscard]] bool ready() const { return ready_; }

    private:
        SmokeGrid(VulkanContext& ctx, std::uint32_t res) : ctx_(ctx), res_(res) {}
        void step(VkCommandBuffer cb, const SmokePipelines& pl, const SmokePlan& plan);

        VulkanContext& ctx_;
        std::uint32_t  res_;
        Image2D        vel_[2]{}, sca_[2]{}, p_[2]{}, aux_{}, solid_{}, temperature_{};
        Buffer         params_{};
        // The glow's sums: the layers' partials (device-local, by address), and the octants, one host-visible
        // block per frame in flight, with the frame serial that wrote each (0 = nothing to take).
        Buffer         glowPartials_{};
        Buffer         glowOut_[impl::kFramesInFlight]{};
        void*          glowMapped_[impl::kFramesInFlight]{};
        std::uint64_t  glowSerial_[impl::kFramesInFlight]{};
        VkDescriptorPool pool_ = VK_NULL_HANDLE;
        VkDescriptorSet  sets_[8]{};
        // The ping-pong parities: which half of each pair is "in".
        int v_ = 0, s_ = 0, p_in_ = 0;
        bool ready_ = false;
        // The field's clock.
        bool          clockStarted_ = false;
        double        lastTime_     = 0.0;
        std::uint64_t seenResets_   = 0;
        bool          warnedCaps_   = false;
    };

}// namespace threepp::vulkan

#endif
