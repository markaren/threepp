// ParticleField — a scene object whose particle count the CPU never walks.
//
// The thesis, one level above GrassMesh: the renderer recognises the type, so
// a field of 300k particles is exactly ONE MeshEntry and ONE EntrySpan. Every
// per-frame CPU loop in the Vulkan backend (lean matrix refresh, frustum cull,
// motion matrices, indirect build, TLAS instance fill) is O(entries), so a
// field costs O(1) per frame whatever its capacity — which is the entire point.
// An InstancedMesh of the same 300k grains pays ~0.2-0.5 µs of entry
// bookkeeping per instance per frame plus ~22 MB/frame of DrawInfo / motion /
// TLAS-instance uploads; a ParticleField pays neither.
//
// The positions live on the device. The CPU knows the CAPACITY and nothing
// else: liveness, radius and (later) orientation all ride in the position
// buffer or in optional device-side side buffers, and the per-particle work is
// done by shaders that dispatch over a CPU-constant domain.
//
// ── VULKAN ONLY ──────────────────────────────────────────────────────────────
// Decided 2026-08-10 (plans/particle-field.md R10). This type has NO GL
// implementation and never will: GL keeps the InstancedMesh path, which was
// measured at 500k moving grains and is a first-class capability there. On the
// GL backend a ParticleField holds a zero-area placeholder triangle, so it is
// a valid Mesh that draws no pixels and crashes nothing — but it renders no
// particles. Use an InstancedMesh under `--api gl`.
//
// ── PHASE 0 ──────────────────────────────────────────────────────────────────
// This is the entity + the buffer, drawing nothing. `Ownership::HostRing` is
// the only implemented mode; every representation below is stored and none is
// yet consumed, so a field in a Vulkan scene flows through the whole frame and
// produces zero pixels. Phase 1 turns MeshRepr into an indirect draw.
//
// ── CHURN CONTRACT (read before calling create) ──────────────────────────────
// A field is created ONCE at its final capacity and is never resized: there is
// no setCapacity, and there never will be. Creating or destroying a field is a
// STRUCTURAL scene change — full entry re-expansion, a vkDeviceWaitIdle, and a
// cleared TAA history — so a field that is spawned and dropped per burst costs
// far more than the particles it draws. Park instead: a field with
// liveCount == 0 stays in the scene and costs one entry. If a field must grow,
// it is a new field and the application pays the rebuild knowingly. (Same cost
// model, and the same reason for stating it here, as
// SplatCloud::setSubmitRanges.)

#ifndef THREEPP_PARTICLEFIELD_HPP
#define THREEPP_PARTICLEFIELD_HPP

#include "threepp/constants.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Texture;

    // Byte-identical to physx::PxVec4 and to GLSL `vec4` under scalar layout.
    // This identity is the whole point: the CUDA device-to-device copy from
    // PxParticleBuffer::getPositionInvMasses() is a straight memcpy with no
    // repack, and the host path below is a single memcpy for the same reason.
    struct ParticlePos {
        float x, y, z;
        // PhysX writes invMass here. ParticleField REINTERPRETS it per
        // Config::wSemantic:
        //   WSemantic::InvMass (default) → radius comes from uniformRadius
        //   WSemantic::Radius            → w IS the world radius
        // A slot with w < 0 is DEAD under either semantic. There is no separate
        // liveness buffer: one buffer, one rule, and every consumer (expansion,
        // AABB write, density scatter, billboard) tests the same predicate.
        float w;
    };
    static_assert(sizeof(ParticlePos) == 16, "ParticlePos must be 16 B / stride 16");

    class ParticleField: public Mesh {

    public:
        enum class WSemantic : std::uint32_t { InvMass = 0, Radius = 1 };
        enum class Ownership : std::uint32_t { Interop, HostRing, Renderer };

        struct Config {
            std::uint32_t capacity      = 0;    // fixed for life; see the churn contract
            Ownership     ownership     = Ownership::HostRing;
            WSemantic     wSemantic     = WSemantic::InvMass;
            // The world radius the MeshRepr proxy geometry is authored at.
            // Under WSemantic::InvMass, w is PhysX's inverse mass and says
            // nothing about size, so the proxy draws at scale 1 and this value
            // is what the later representations (AABB dilation, billboard size,
            // density kernel) use. Under WSemantic::Radius, w IS the world
            // radius and the proxy scales by w / uniformRadius.
            float         uniformRadius = 0.01f;
            bool          orientations  = false;// allocate the snorm16x4 buffer
            bool          attributes    = false;
        };

        // Representations. Each is opt-in, each is independently costed, and
        // each exists because a sensor can see it. PHASE 0 STORES THEM AND
        // CONSUMES NONE — enabling one changes no pixels yet.
        struct MeshRepr {// granular
            std::shared_ptr<BufferGeometry> geometry;// the per-particle proxy
            std::shared_ptr<Material>       material;// ONE material for the field
            bool enabled = false;
        };
        struct BillboardRepr {// sparse dust / spray
            std::shared_ptr<Texture> texture;
            Blending blending  = Blending::Normal;
            float    sizeScale = 1.f;
            bool     enabled   = false;
        };
        // Dense dust / smoke. The field is scattered ONCE per frame into a
        // world-anchored 3D density volume that every view's froxel pass then
        // samples (plan §3.3): the per-particle cost is one splat, independent
        // of screen coverage and independent of how many cameras look at it.
        //
        // ── PHASE 2 API ADDITION (the plan's struct has no bounds member) ────
        // The volume needs a world box and a voxel count, so §3.3's "sized per
        // field from its configured world bounds" is spelled out here:
        //
        //   center/halfExtent — the WORLD-space axis-aligned box the volume
        //     covers, i.e. [center - halfExtent, center + halfExtent]. World,
        //     not field-local, because the volume is shared across views and
        //     across fields; the field's own matrixWorld is applied to the
        //     PARTICLES on the way in (exactly as the mesh representation
        //     applies it), so a parented field still lands in the right box.
        //     Both may be changed per frame — the volume is re-scattered from
        //     scratch every frame, so moving the box is free.
        //   resolution — voxels per axis. LATCHED at the first frame the
        //     representation is enabled and never changed afterwards: the
        //     image is allocated once and never resized (the same fixed-size
        //     contract as the position ring). Clamped to [8, 256]; 128 is
        //     8 MB of r32ui.
        //
        // Density is quantised: a voxel accumulates fixed-point sigma_t with
        // 12 fractional bits (quantum 1/4096 per metre) so the accumulation is
        // an INTEGER add and therefore associative — which is what makes two
        // renders of the same scene byte-identical. See particle_density.glsl.
        struct DensityRepr {// dense dust / smoke
            float sigmaPerParticle = 1.f;// sigma_t contributed by one particle
            Color albedo{1.f, 1.f, 1.f};
            float anisotropy = 0.f;      // HG g for THIS medium
            Vector3       center{0.f, 0.f, 0.f};   // world centre of the volume
            Vector3       halfExtent{5.f, 5.f, 5.f};// world half-size per axis
            std::uint32_t resolution = 128;        // voxels/axis; latched at first enable
            bool          enabled    = false;

            // ── EMISSION (fire) — plans/particle-atmosphere.md F-A ───────────
            // A flame is emission-dominated, and emission at a sample point is
            // sigma(x) * L_e(x). sigma is ALREADY marched by the dust path, so
            // fire needs NO second volume: make L_e an analytic function of the
            // normalised HEIGHT inside this field's box — a blackbody ramp,
            // hot at the bottom, cooling into the smoke line — and the flame's
            // SHAPE comes from the particle distribution while its COLOUR
            // structure comes from the ramp. (A per-particle temperature
            // channel, for real combustion sims, is deliberately v2: the
            // analytic ramp covers authored fire and costs one vec4.)
            //
            // emissiveIntensity == 0 is the exact no-op: the shader takes a
            // uniform branch around the whole emissive path, so a pure dust
            // field renders the same bits it did before emission existed.
            float emissiveIntensity = 0.f;  // HDR radiance scale of a 2000 K flame;
                                            // ACES/AgX wants tens, not ones.
            float tempBottomK       = 1900.f;// blackbody T at the box BOTTOM (core)
            float tempTopK          = 800.f; // blackbody T at the box TOP (sooty tips)
            // Exponent of the bottom->top temperature ramp over normalised box
            // height: T = mix(bottom, top, pow(heightFraction, tempFalloff)).
            // > 1 holds the core temperature through the lower flame and drops
            // it late, which is what a real diffusion flame does.
            float tempFalloff       = 1.6f;
        };
        struct TracedRepr {// procedural-AABB BLAS
            float radiusScale = 1.f;// AABB dilation vs the render radius
            bool  enabled     = false;
        };

        // Throws std::invalid_argument on capacity == 0 or on an ownership mode
        // this phase does not implement (Interop / Renderer). See the churn
        // contract in the file header before choosing a capacity.
        static std::shared_ptr<ParticleField> create(const Config& config);

        [[nodiscard]] const Config& config() const { return config_; }
        [[nodiscard]] std::uint32_t capacity() const { return config_.capacity; }

        // Turn the mesh representation on: every live particle draws `proxy`
        // once, with `material`, as ONE indirect draw (Vulkan only).
        //
        // Use this rather than mutating meshRepr() by hand. `material` must ALSO
        // be the field's Mesh material — the G-buffer shades a particle through
        // the field entry's MaterialDesc slot, which is derived from
        // Object3D::material() like every other mesh — and this is what keeps
        // the two in step. The field's own Mesh GEOMETRY is deliberately left
        // as the zero-area placeholder (see the file header): the proxy is
        // pulled bindlessly by the particle vertex stage, never rasterised at
        // the field origin, and never put in the field's BLAS.
        void setMeshRepr(std::shared_ptr<BufferGeometry> proxy,
                         std::shared_ptr<Material>       material);

        // Turn the density representation on: every live particle adds
        // `sigmaPerParticle` (1/m) of extinction to the world box
        // [center - halfExtent, center + halfExtent], sampled by the froxel
        // volumetrics of EVERY view (Vulkan only). Objects behind the box are
        // attenuated, clustered lights inside it glow, and the ambient/sky
        // in-scatter fills it — with no per-view particle work.
        //
        // Use this rather than mutating densityRepr() by hand for the first
        // enable: `resolution` is latched the frame the volume is allocated,
        // and this is the call that makes that moment explicit. center /
        // halfExtent / sigmaPerParticle stay live-editable afterwards.
        void setDensityRepr(const Vector3& center, const Vector3& halfExtent,
                            float sigmaPerParticle, std::uint32_t resolution = 128);

        [[nodiscard]] MeshRepr&      meshRepr() { return meshRepr_; }
        [[nodiscard]] const MeshRepr& meshRepr() const { return meshRepr_; }
        [[nodiscard]] BillboardRepr& billboardRepr() { return billboardRepr_; }
        [[nodiscard]] const BillboardRepr& billboardRepr() const { return billboardRepr_; }
        [[nodiscard]] DensityRepr&   densityRepr() { return densityRepr_; }
        [[nodiscard]] const DensityRepr& densityRepr() const { return densityRepr_; }
        [[nodiscard]] TracedRepr&    tracedRepr() { return tracedRepr_; }
        [[nodiscard]] const TracedRepr& tracedRepr() const { return tracedRepr_; }

        // ── Ownership::HostRing ─────────────────────────────────────────────
        // Point the field at n host positions laid out as ParticlePos (== 16 B
        // PxVec4). The ONLY per-particle CPU cost in the whole design, and it
        // is one memcpy, not a loop. n > capacity() is clamped to capacity.
        // Also sets the live count to n.
        void submit(const void* pxVec4Array, std::uint32_t n);

        // Per-particle orientation, as n quaternions in (x, y, z, w) order.
        // Requires Config::orientations. WRITE-ONCE by contract: the device
        // buffer is a single instance (not ringed), because the plan's model is
        // "written once, at slot claim" — an orientation set is authored with
        // the field, not animated. Rewriting it while frames are in flight is
        // a host write to a buffer the GPU may be reading; if a sim ever needs
        // per-frame orientation it needs a ring, exactly like positions.
        // Encoded to snorm16x4 (8 B/particle on the device, versus the 36 B of
        // host floats per instance the InstancedMesh path carried).
        void setOrientations(const float* quatXyzw, std::uint32_t n);

        // For Ownership::Renderer / Interop the count comes from the device and
        // is never read back. For HostRing, submit() sets it; this setter exists
        // to park a field (setLiveCount(0)) without re-submitting positions.
        void setLiveCount(std::uint32_t n);
        [[nodiscard]] std::uint32_t liveCount() const { return liveCount_; }

        [[nodiscard]] std::string type() const override { return "ParticleField"; }

        // ── Renderer-internal ───────────────────────────────────────────────
        // The staging block submit() writes. Sized to capacity once, at create,
        // and never reallocated. The Vulkan backend copies the live prefix into
        // the current ring slot inside its post-fence window; the serial is what
        // lets a slot skip a copy it already has.
        [[nodiscard]] const std::vector<ParticlePos>& hostPositions() const { return host_; }
        [[nodiscard]] std::uint64_t dataSerial() const { return dataSerial_; }

        // snorm16x4 quaternions, 4 shorts per particle, sized to capacity when
        // Config::orientations is set and empty otherwise. The serial is 0
        // until setOrientations() is first called, which is what lets the
        // backend skip allocating the buffer for a field that never uses one.
        [[nodiscard]] const std::vector<std::int16_t>& hostOrientations() const { return ori_; }
        [[nodiscard]] std::uint64_t orientationSerial() const { return oriSerial_; }

        explicit ParticleField(const Config& config);
        ~ParticleField() override = default;

    private:
        Config        config_;
        std::uint32_t liveCount_  = 0;
        std::uint64_t dataSerial_ = 1;// bumped by submit/setLiveCount

        std::vector<ParticlePos> host_;
        std::vector<std::int16_t> ori_;// snorm16x4, 4 per particle
        std::uint64_t             oriSerial_ = 0;

        MeshRepr      meshRepr_;
        BillboardRepr billboardRepr_;
        DensityRepr   densityRepr_;
        TracedRepr    tracedRepr_;
    };

}// namespace threepp

#endif//THREEPP_PARTICLEFIELD_HPP
