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
            float         uniformRadius = 0.01f;// used when wSemantic == InvMass
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
        struct DensityRepr {// dense dust / smoke
            float sigmaPerParticle = 1.f;// sigma_t contributed by one particle
            Color albedo{1.f, 1.f, 1.f};
            float anisotropy = 0.f;      // HG g for THIS medium
            bool  enabled    = false;
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

        [[nodiscard]] MeshRepr&      meshRepr() { return meshRepr_; }
        [[nodiscard]] BillboardRepr& billboardRepr() { return billboardRepr_; }
        [[nodiscard]] DensityRepr&   densityRepr() { return densityRepr_; }
        [[nodiscard]] TracedRepr&    tracedRepr() { return tracedRepr_; }

        // ── Ownership::HostRing ─────────────────────────────────────────────
        // Point the field at n host positions laid out as ParticlePos (== 16 B
        // PxVec4). The ONLY per-particle CPU cost in the whole design, and it
        // is one memcpy, not a loop. n > capacity() is clamped to capacity.
        // Also sets the live count to n.
        void submit(const void* pxVec4Array, std::uint32_t n);

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

        explicit ParticleField(const Config& config);
        ~ParticleField() override = default;

    private:
        Config        config_;
        std::uint32_t liveCount_  = 0;
        std::uint64_t dataSerial_ = 1;// bumped by submit/setLiveCount

        std::vector<ParticlePos> host_;

        MeshRepr      meshRepr_;
        BillboardRepr billboardRepr_;
        DensityRepr   densityRepr_;
        TracedRepr    tracedRepr_;
    };

}// namespace threepp

#endif//THREEPP_PARTICLEFIELD_HPP
