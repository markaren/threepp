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
// ── OWNERSHIP: WHO WRITES THE POSITIONS ──────────────────────────────────────
// Two modes are implemented, and they divide the API cleanly in half. Calling
// one mode's half on the other mode's field THROWS rather than silently doing
// nothing — a field whose positions never arrive renders an empty scene, and
// that is a bug worth an exception at the call site instead of a debugging
// session at the capture.
//
//   Ownership::HostRing  — the CPU owns the positions. submit() memcpys them
//     into a ring of kFramesInFlight + 1 host-visible buffers (the ring exists
//     because a HOST write races an in-flight read). Its half of the API is
//     submit() / setLiveCount(); setEmitter() and setEmitterTime() throw.
//
//   Ownership::Renderer  — the GPU owns the positions. A threepp compute pass
//     (particle_emit.comp) writes ONE device-local position buffer and its
//     prevPositions sibling at the head of every frame, from the closed-form
//     trajectory EmitterParams describes. Nothing crosses the bus per frame and
//     the CPU never walks a particle. Its half of the API is setEmitter() /
//     setEmitterTime(); submit() throws.
//
//   Ownership::Interop   — declared, not implemented (CUDA zero-copy); create()
//     still rejects it.
//
// setOrientations() belongs to NEITHER half: an orientation set is authored
// once with the field and is orthogonal to who advances the positions, so a
// Renderer field may still carry one.
//
// ── PHASE STATE ──────────────────────────────────────────────────────────────
// MeshRepr (one indirect draw of a proxy per particle) and DensityRepr (a
// world-anchored sigma_t volume, with an optional blackbody emission ramp) are
// live. BillboardRepr and TracedRepr are stored and consumed by nothing.
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

        // ── EMITTER (Ownership::Renderer) — plans/particle-atmosphere.md F-C ──
        //
        // The device emitter is STATELESS and CLOSED FORM: a slot's position is
        // f(seed, slot, t), evaluated from scratch every frame by one compute
        // thread. There is no integration, no ping-pong pair of buffers and no
        // per-frame state anywhere, which is not an implementation detail but
        // the reason the mode is worth having:
        //
        //   • DETERMINISM. The same params and the same t give the same bytes,
        //     in this process and in the next one. An integrator's state is a
        //     function of every frame that came before it; a closed form is a
        //     function of nothing but its arguments.
        //   • EXACT MOTION VECTORS. The same dispatch writes f(t) and f(t - dt)
        //     into positions and prevPositions, so the two can never disagree
        //     about which frame they describe — the whole staleness bug class
        //     that copy-the-previous-buffer schemes carry (see the prevVertex
        //     notes in the deforming-geometry path) simply does not exist.
        //   • FREE SEEKING. setEmitterTime(8.f, dt) is valid with no warm-up, so
        //     a headless capture can jump straight to t = 8 s.
        //
        // ARCHETYPES ARE PARAMETERS, NOT SHADER VARIANTS. Snow, rain, embers and
        // dust motes are four points in this struct, not four shaders — which is
        // what keeps the dispatch count at one per field however many kinds of
        // weather a scene has.
        //
        //   snow  velocity {0,-1.1,0}, accel 0, driftAmplitude ~0.35,
        //         driftFrequency ~0.12, lifetime = fall height / speed
        //   rain  velocity {0,-9,0}, driftAmplitude ~0.02, short lifetime
        //   ember velocity {0,+1.4,0}, accel {0,+0.4,0}, speedSpread ~0.5,
        //         driftGrowth 1 (the tips wander, the base is steady), duty < 1
        //   mote  velocity ~0, driftAmplitude ~0.15, driftFrequency ~0.05,
        //         spawnHalfExtent = the whole room
        //
        // WHERE THE PARTICLES ARE. Positions are written in the FIELD'S LOCAL
        // space; the field's matrixWorld is applied downstream by every consumer
        // (the raster proxy's model matrix, the density splat's world matrix),
        // exactly as it is for a HostRing field. So a Renderer field can be
        // parented and moved and its emitter parameters do not change.
        //
        // THE STEADY-STATE CLOUD IS THE SPAWN BOX SWEPT ALONG THE TRAJECTORY.
        // Slots are born uniformly in [spawnCenter ± spawnHalfExtent] with
        // uniformly distributed ages, so at constant velocity the visible volume
        // is that box smeared over velocity * lifetime. Author a SNOW field as a
        // thin slab at the TOP of the volume with lifetime = height / speed —
        // not as a box the size of the volume, which gives a triangular density
        // ramp instead of an even snowfall.
        struct EmitterParams {
            // Birth region, field-local. A thin slab is an emission plane.
            Vector3 spawnCenter{0.f, 0.f, 0.f};
            Vector3 spawnHalfExtent{5.f, 0.1f, 5.f};

            // Initial velocity (m/s) and its per-particle spread. speedSpread is
            // an ISOTROPIC perturbation in m/s added to the vector, so it reads
            // as a cone on a directional emitter and as a puff on a still one —
            // one knob instead of a cone angle plus a speed jitter.
            Vector3 velocity{0.f, -1.f, 0.f};
            float   speedSpread = 0.f;
            // Constant acceleration (m/s^2): gravity is -Y, buoyancy is +Y.
            // Falling snow and rain use ZERO — they are at terminal velocity,
            // which is a constant, and an accelerating flake is a bug you have
            // to author around.
            Vector3 accel{0.f, 0.f, 0.f};
            // Uniform horizontal drift (m/s). Kept separate from `velocity`
            // because it is a property of the SCENE (and is usually animated)
            // rather than of the emitter; the two are summed on the way to the
            // shader, so `velocity + wind` is what a particle actually flies.
            Vector3 wind{0.f, 0.f, 0.f};

            // ── Curl-style drift ────────────────────────────────────────────
            // Sums of phase-shifted sines with incommensurate frequencies: a
            // wandering, non-repeating displacement that is a PURE FUNCTION of
            // t (no noise texture, no integrated turbulence, nothing to seek
            // past). driftScale couples the phase to the particle's own
            // position, which turns a field of independently wobbling flakes
            // into gusts travelling through the volume.
            float driftAmplitude = 0.f;// metres
            float driftFrequency = 0.f;// Hz of the slowest term
            float driftGrowth    = 0.f;// 0 = constant, 1 = ramps in over the life
            float driftScale     = 0.f;// metres of spatial wavelength; 0 = per-slot phase only

            // ── Life cycle ──────────────────────────────────────────────────
            // A slot repeats with period `lifetime` (jittered per slot) and is
            // ALIVE for the first `dutyCycle` of it. Outside that window it
            // writes the w < 0 dead sentinel — the one liveness rule every
            // consumer already tests — so duty < 1 gives a field whose mass
            // breathes rather than sitting at a constant particle count.
            float lifetime       = 4.f;
            float lifetimeJitter = 0.f;// +/- fraction of lifetime, [0,1]
            float dutyCycle      = 1.f;// alive fraction of the period, (0,1]

            // Per-particle radius, written into w. Under
            // WSemantic::Radius the mesh proxy scales by w / uniformRadius, so
            // sizeJitter is where flake-size variety comes from; under
            // WSemantic::InvMass w is only ever tested for the sentinel and the
            // value is inert.
            float size       = 0.02f;
            float sizeJitter = 0.f;// +/- fraction of size, [0,1]

            std::uint32_t seed = 20260812u;
        };

        // Throws std::invalid_argument on capacity == 0 or on an ownership mode
        // this phase does not implement (Interop). See the churn contract in the
        // file header before choosing a capacity.
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
        //
        // THROWS on an Ownership::Renderer field: its positions live in device
        // memory the host cannot map, and a silent no-op here would render an
        // empty field with no diagnostic anywhere.
        void submit(const void* pxVec4Array, std::uint32_t n);

        // ── Ownership::Renderer ─────────────────────────────────────────────
        // Install the closed-form trajectory the device emitter evaluates. Free
        // to call every frame — the parameters are O(1) bytes and are published
        // to the GPU as push constants, so animating the wind costs nothing.
        //
        // THROWS on a HostRing / Interop field: there is no emitter there, and
        // parameters that nothing reads are worse than an exception.
        void setEmitter(const EmitterParams& params);
        [[nodiscard]] const EmitterParams& emitter() const { return emitter_; }

        // Advance the emitter to ABSOLUTE time `timeSec`. Not a delta: the
        // trajectory is closed form in t, so any t is valid in any order and a
        // capture may seek.
        //
        // `dtSec` is the interval the MOTION VECTORS are taken over — the same
        // dispatch writes f(t) and f(t - dtSec) — so pass the frame's own delta.
        // Pass 0 to freeze the field: every particle then reprojects onto itself
        // and TAA converges completely, which is what a still capture wants.
        //
        // NEVER a wall clock: the caller owns the clock, so a headless capture
        // is a function of its frame index and nothing else. THROWS on a
        // non-Renderer field.
        void setEmitterTime(float timeSec, float dtSec);
        [[nodiscard]] float emitterTime() const { return emitTime_; }
        [[nodiscard]] float emitterDt() const { return emitDt_; }

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

        // For HostRing, submit() sets it; this setter exists to park a field
        // (setLiveCount(0)) without re-submitting positions.
        //
        // For Ownership::Renderer the count is CAPACITY, set at construction and
        // written to the device once: a stateless emitter has no compaction and
        // no free list, so every slot is dispatched every frame and the ones
        // outside their lifetime window identify themselves with the w < 0
        // sentinel. Nothing is gained by tracking a second, redundant number
        // that would have to be recomputed on the host from the same closed form
        // the GPU is already evaluating. setLiveCount(0) still parks the field —
        // that is the one legitimate use of this setter in Renderer mode, and it
        // skips the emit dispatch entirely.
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

        // Ownership::Renderer only. Deliberately NOT behind dataSerial_: the
        // emitter's whole state is O(1) bytes that ride in a push constant, so
        // there is nothing to version-gate and no upload to skip.
        EmitterParams emitter_;
        float         emitTime_ = 0.f;
        float         emitDt_   = 1.f / 60.f;
    };

}// namespace threepp

#endif//THREEPP_PARTICLEFIELD_HPP
