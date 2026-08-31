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
//   Ownership::Interop   — a FOREIGN DEVICE API owns the positions. The
//     renderer allocates the field's positions as an EXPORTABLE dedicated
//     allocation and hands the OS handle to the application
//     (VulkanRenderer::enableParticleFieldInterop), which imports it into CUDA
//     and does one cuMemcpyDtoD per frame straight out of its sim — for PhysX
//     PBD that is PxParticleBuffer::getPositionInvMasses(), and ParticlePos is
//     byte-identical to PxVec4 by design, so the copy is a memcpy with no
//     repack and no host round trip. Its half of the API is setLiveCount() plus
//     the renderer-side enable; submit() and setEmitter() throw.
//
//     ── THE ACCEPTED TRADE: AN INTEROP FIELD IS NOT REPRODUCIBLE ────────────
//     GPU PhysX is not bit-deterministic — its solver's atomics and its
//     work partitioning both reorder run to run — so a field whose positions
//     come out of one FORFEITS the byte-reproducibility contract every other
//     mode holds. Two runs of the same scene with the same seed will differ,
//     and no amount of care in this class can prevent it, because the bytes
//     are authored on the other side of the import.
//
//     Use it for SIM-COUPLED CONTENT ONLY — grains on a belt, debris, a
//     fluid — where the sim IS the truth being rendered. Weather (snow, rain,
//     embers, dust) stays on Ownership::Renderer's analytic emitter for
//     exactly this reason: it is a closed form, it is reproducible, and it
//     costs nothing on the bus either. Sensor goldens over an Interop field
//     must be tolerance gates, never byte compares.
//
//     Everything derived from the emitter's closed form is also unavailable
//     here — age fade, size taper, colour ramp, surface landing — because
//     there is no closed form to evaluate. An Interop field is positions, a
//     radius and an orientation set, and that is the whole model.
//
// setOrientations() belongs to NEITHER half: an orientation set is authored
// once with the field and is orthogonal to who advances the positions, so a
// Renderer field may still carry one.
//
// ── PHASE STATE ──────────────────────────────────────────────────────────────
// MeshRepr (one indirect draw of a proxy per particle), DensityRepr (a
// world-anchored sigma_t volume, with an optional blackbody emission ramp) and
// BillboardRepr (one indirect draw of a vertex-less camera-facing quad per
// particle, ADDITIVE and unlit — see the struct) are live. TracedRepr is
// stored and consumed by nothing.
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
            // ── PER-PARTICLE APPEARANCE (plans/particle-volumetric-sprites R2) ─
            // Allocate ONE device buffer of capacity * 16 B holding a vec4 per
            // particle: rgb = linear HDR radiance, a = reserved (phase-2
            // opacity). It rides the POSITIONS' path exactly — same exportable
            // flags under Ownership::Interop, same export call, same per-frame
            // snapshot — so a mode where positions are safe and attributes are
            // not cannot exist by construction.
            //
            // Under HostRing / Renderer the v1 contract is WRITE-ONCE
            // (setAttributes, mirroring setOrientations): a per-frame host ring
            // for colours is scope this slice does not carry, and the demo that
            // needs colours every frame is on the interop leg.
            //
            // BillboardRepr reads it INSTEAD of the colorHot/colorCool ramp
            // when it is present (R3) — one scheme or the other, never a blend.
            bool          attributes    = false;
            // ── Ownership::HostRing: the STABLE SLOT promise ────────────────
            // The host declares that index i names the SAME particle in every
            // submit — a fixed pool written in place, dead slots left in the
            // buffer with w < 0, no compaction between frames. It costs the
            // host nothing to say so when it is already true, and it buys the
            // one thing a host-driven field otherwise cannot have: the previous
            // ring slot becomes a genuine prevPositions buffer, so
            // BillboardRepr::stretchSeconds (the velocity streak) works here
            // exactly as it does for a Renderer field. Without it (pos -
            // prevPos) would be the difference between two UNRELATED particles
            // and the field would streak in every direction at once.
            //
            // A newly spawned slot has one frame of stale prevPos, because the
            // slot's previous occupant is what the previous submit left there;
            // the streak is bounded by BillboardRepr::stretchMax and is meant
            // to be spent under a spawn fade-in. The renderer also drops the
            // stretch for any frame whose two ring slots are not CONSECUTIVE
            // submits (a frame the host skipped, a field just created) rather
            // than smearing over a two-frame or backwards displacement — the
            // sprite is simply round that frame.
            //
            // Ignored under Ownership::Renderer (which has real prevPositions)
            // and under Ownership::Interop (whose slots are snapshots of a
            // foreign buffer this class cannot make promises about).
            bool          hostStableSlots = false;
        };

        // Representations. Each is opt-in, each is independently costed, and
        // each exists because a sensor can see it. MeshRepr, DensityRepr and
        // BillboardRepr are live; TracedRepr is stored and consumed by nothing
        // — see the PHASE STATE block at the top of this header.
        struct MeshRepr {// granular
            std::shared_ptr<BufferGeometry> geometry;// the per-particle proxy
            std::shared_ptr<Material>       material;// ONE material for the field
            // ── F4: distance LOD, the FAR half of the split ─────────────────
            // Beyond this many metres from the camera the proxy collapses onto
            // its own centre (zero area, no fragments, no deferred shade) and
            // the BILLBOARD representation is expected to take over — see
            // BillboardRepr::lodNear, which is the complementary gate on the
            // same field. 0 = no LOD, every live particle rasterises, which is
            // exactly the pre-F4 behaviour.
            //
            // The collapse is a per-vertex predicate rather than a compaction:
            // a field is ONE indirect draw of `liveCount` instances and there
            // is no free list to compact against, so what LOD buys is the
            // fragment and deferred-shade cost of distant proxies (the 5M snow
            // bench put 11 of 16.8 ms exactly there) and not their vertex cost.
            float lodFar = 0.f;
            // Metres of soft ramp below lodFar, over which the proxy shrinks to
            // nothing instead of vanishing on one frame. A hard cut pops; a
            // shrink under a billboard that is fading IN over the same band is
            // invisible.
            float lodFade = 0.f;
            // ── F4: the NEAR cull ───────────────────────────────────────────
            // Particles closer to the camera than this shrink away. A 1.6 cm
            // flake 40 cm from the lens covers ~100 px and reads as a floating
            // crystal — the "scale outlier" of the F3 snow capture, which is a
            // PROXIMITY artefact and not (as first suspected) an emitter size
            // hash that can overshoot: `size * (1 + sizeJitter * rndS)` is
            // bounded by construction. A real camera cannot resolve a flake
            // inside its near focus either, so shrinking it out is the honest
            // fix. 0 = off.
            float nearCull = 0.f;
            bool enabled = false;
        };
        // Sparse emissive spray: embers, sparks, rain streaks, fireflies.
        //
        // ── F3 SCOPE: ADDITIVE AND UNLIT ────────────────────────────────────
        // One indirect draw of (4 vertices x liveCount) per field, composited
        // in the post-upscaler overlay slot. Two things are deliberately NOT
        // done and are not oversights:
        //
        //   • NO PER-QUAD LIGHTING. The legacy ParticleSystem billboards run
        //     particle_light.comp, which casts an RT shadow ray and walks the
        //     cluster list per particle. For emissive content that is µs-class
        //     work to answer a question whose answer does not matter — an
        //     ember IS a light source, and the light an emitter casts on the
        //     world is its own PointLight (FireEffect's), through the ordinary
        //     deferred path. Normal-blend billboards, which DO want it, are a
        //     later slice.
        //   • NO SORT. Addition commutes, so the frame buffer holds the same
        //     sum whatever order the quads arrive in. `blending` is therefore
        //     ignored in this phase and every field billboard draws additive;
        //     when normal blend lands it must reuse SplatPass's deterministic
        //     radix sort rather than grow a second one.
        struct BillboardRepr {
            // Optional. MODULATES the procedural sprite (rgb tint x alpha
            // coverage); null binds a 1x1 white default, so the untextured
            // look is the shipped one and needs no asset.
            std::shared_ptr<Texture> texture;
            Blending blending  = Blending::Normal;// ignored in the additive slice
            float    sizeScale = 1.f;             // multiplies the particle radius

            // Linear HDR radiance at age 0 and at end of life. The default is
            // the blackbody arc a cooling ember actually walks — white-hot
            // yellow into deep red — so an ember field with no colour authoring
            // still reads as fire rather than as orange dots.
            Color colorHot{1.00f, 0.72f, 0.34f};
            Color colorCool{1.00f, 0.16f, 0.02f};
            float intensity = 1.f;// HDR scale on both colours

            // 0 = a tight spark, 1 = a broad glow. Shapes the radial falloff.
            float softness = 0.45f;
            // Weight of the fixed tight core term (t^9) the falloff adds under
            // the softness-shaped skirt. The core is what keeps a few-pixel
            // ember from reading as a soft blob — but it plants a 1-2 px hard
            // dot at every sprite's centre REGARDLESS of drawn radius, and in
            // a dense field of big soft smoke sprites those dots are speckle
            // that only averages away at particle counts fill-rate cannot
            // afford (measured: the prop-wash far wake mottles at 2M and needs
            // 5M to smooth with the core on). Softness deliberately does not
            // reach this term, so a field of large parcels turns it down here.
            // 0.85 is the pre-knob constant: the default is byte-identical to
            // every field authored before the knob existed.
            float coreWeight = 0.85f;
            // Brightness over life: (1 - ageFrac)^fadePower. > 1 holds the
            // spark bright and drops it late, which is what a burning ember
            // does; the age is re-derived from the emitter's closed form, so
            // this only does anything on an Ownership::Renderer field.
            float fadePower = 1.6f;
            // Per-particle brightness spread, hashed. Real embers are not N
            // identical lamps, and this is the whole cost of saying so.
            float brightJitter = 0.45f;
            // Radius over life: r *= (1 - sizeTaper * ageFrac). A cooling ember
            // gets smaller, not bigger.
            float sizeTaper = 0.55f;

            // ── Velocity stretch (rain) ─────────────────────────────────────
            // Seconds of travel to smear the quad over, along the particle's
            // own screen-projected velocity. The velocity is FREE and exact: a
            // Renderer field's emit dispatch already wrote f(t) and f(t - dt),
            // so (pos - prevPos)/dt is the analytic answer with no extra state.
            // 0 = a round sprite. A 9 m/s raindrop crosses ~20 px per frame and
            // reads as HAIL without this.
            float stretchSeconds = 0.f;
            // Cap, in multiples of the radius. A fast particle near the near
            // plane projects to an arbitrarily long segment, and one 2000-px
            // streak across the frame is a bug the eye reads instantly.
            float stretchMax = 24.f;
            // ── F4: the SCREEN-SPACE streak cap ─────────────────────────────
            // stretchMax is expressed in radii, i.e. in WORLD units, so it caps
            // a streak's length in metres and says nothing about how many
            // PIXELS that becomes. The nearest drop in a field the camera
            // stands inside is metres closer than the rest, so it projects its
            // (correctly capped) 12 cm to a bright bar across a quarter of the
            // frame — the "one anomalously bright near streak per frame" the F3
            // sequence shows. This is the cap in the domain the defect lives
            // in: a fraction of the frame HEIGHT, applied in NDC. 0 = off.
            float stretchMaxScreen = 0.f;

            // ── F4: near fade ───────────────────────────────────────────────
            // Fade the sprite out below this camera distance, in metres, over
            // the whole [0, nearFade] band. Additive quads compound with
            // proximity in two ways at once — coverage grows as 1/d^2 and the
            // streak grows as 1/d — so the closest particle in a field the
            // camera stands inside is the brightest thing on screen by a wide
            // margin, whatever the authored intensity is. 0 = off.
            float nearFade = 0.f;
            // ── F4: distance LOD, the NEAR half of the split ────────────────
            // Quads CLOSER than this many metres collapse to zero area, on the
            // understanding that MeshRepr is drawing that particle as a solid
            // proxy there (MeshRepr::lodFar is the complementary gate). Both
            // read the same position buffer in the same frame, so the split
            // costs no CPU, no second field and no compaction. 0 = off, i.e.
            // the quad is drawn at every distance.
            float lodNear = 0.f;
            // Metres of ramp above lodNear over which the quad fades IN, so it
            // arrives as the mesh proxy is shrinking out over the same band.
            float lodFade = 0.f;

            // ── F4: bloom, without re-entering the upscaler domain ──────────
            // > 0 renders this field a SECOND time into a small offscreen HDR
            // target, runs the shared bloom_down/up pyramid on that target
            // alone and composites the result additively in the same overlay
            // slot the quads land in. The value scales the radiance written to
            // the glow target, so a field can bloom harder or softer than its
            // own sprite is bright.
            //
            // 0 (the default) skips the whole chain — no target is allocated,
            // no pass is recorded, no pixel changes. Weather fields want that:
            // 300k rain streaks have nothing to bloom and the chain is a pure
            // cost. Sparks want the opposite.
            float glow = 0.f;
            // Bright-pass knee for this field's own pyramid, in the same linear
            // HDR domain the sprite writes.
            //
            // 0 (the default) means NO bright pass and no firefly suppression,
            // and that is the right default here rather than a lazy one: the
            // glow target contains only this field's emissive quads, so every
            // pixel in it is already "the highlight". Switching the scene
            // pyramid's first-level behaviour on instead Karis-suppresses a
            // 3-px spark — it is a firefly by construction — and then
            // thresholds what is left to nothing. Raise it only when a field
            // wants its DIM particles left out of its own halo.
            float glowThreshold = 0.f;

            // ── F5: the splash ring ─────────────────────────────────────────
            // When EmitterParams::Surface::splashSeconds is on, a landed
            // particle's quad stops facing the camera and lies FLAT in the
            // world XZ plane, drawn as an expanding annulus. This is the
            // annulus width as a fraction of the ring's own radius: 1 is a
            // filled disc, 0.3 is a rim. Nothing here turns the splash on —
            // that is the emitter's business, and this shader only draws what
            // the emitter wrote into w.
            float splashRingWidth = 0.30f;

            // ── 4c: the SPRITE slice — alpha-over, and lit ──────────────────
            // Two opt-in flags that together turn this representation from
            // "emissive dot" into "a parcel of water/dust the sun shines on".
            // Both default OFF and the additive path is byte-identical when
            // they are: the pipeline, the vertex maths and the fragment output
            // all branch on them and nothing else changes.
            //
            // alphaOver: composite premultiplied SRC_ALPHA-over instead of
            // ONE/ONE. An additive sprite can only ever ADD light, so a white
            // puff over a bright sky is invisible and the same puff over a dark
            // hull is a lamp — the two failures that make additive dots read as
            // a snow flurry rather than as water. Alpha-over OCCLUDES, which is
            // what a chunk of aerated water does.
            //
            // ORDER. This pass sorts nothing (that is still true): draws are
            // issued in field order and, within a field, instances in SLOT
            // order, which Vulkan's primitive-order guarantee makes the blend
            // order too. So a host that wants correct alpha-over submits its
            // particles back-to-front — which for a few hundred sprites is one
            // argsort. Documented rather than solved, deliberately: a device
            // radix sort (SplatPass's) is the answer at 10^5 sprites, and this
            // slice is for 10^2.
            bool alphaOver = false;
            // lit: per-vertex radiance = colour x (ambient + sun x phase(V.L))
            // from the SCENE's sun and ambient, filled engine-side (the same
            // one-sun policy the deferred path follows). Still not a lighting
            // path — no shadow ray, no cluster walk, no per-fragment work: one
            // Henyey-Greenstein lobe per particle, which is what makes a sheet
            // of spray FLARE when the lens looks through it into the sun and go
            // grey when the sun is behind the camera.
            bool lit = false;
            // HG asymmetry for that lobe. ~0.3-0.4 is the forward-ish, still
            // half-isotropic scattering of a water parcel; 0 is isotropic.
            float litPhaseG = 0.35f;
            // An ambient radiance FLOOR, added to the scene's own summed
            // AmbientLights in the same linear units. Not a scale on them: an
            // IBL-lit scene usually carries no AmbientLight at all (the env map
            // is the ambience, and this pass cannot sample it), so scaling zero
            // would leave the shaded side of every sprite black. This is the
            // one number that says how dark that side is allowed to get.
            float litAmbient = 0.2f;
            // Master coverage scale in alphaOver mode: the sprite's own alpha,
            // before the procedural/texture falloff. Ignored when additive
            // (there `intensity` is the only knob and coverage is folded into
            // the radiance).
            float opacity = 1.0f;

            // ── VOLUMETRIC TRANSPORT (plans/particle-volumetric-sprites R4/R5) ─
            // The sprite carries the IMAGE; the field's own DensityRepr volume
            // carries the LIGHT TRANSPORT. Two short fixed-step marches of that
            // volume per sprite — one toward the camera, one toward the sun —
            // turn a flat additive projection into something with dust lanes and
            // a lit rim, and cost no sort, because both factors depend only on
            // the sprite's OWN position and so leave the blend commutative.
            //
            //   rgb = attr.rgb * intensity
            //       * pow(T_cam, volumeExtinction)
            //       * mix(1, T_sun * (volumeAmbient + volumeSunGain * HG(V.L)),
            //             volumeShadow)
            //
            // BOTH DEFAULT TO 0 AND 0 IS AN EXACT NO-OP: the shader takes a
            // uniform branch around both marches, and a field that leaves them
            // alone renders the bits it did before this existed — the same
            // discipline as DensityRepr::emissiveIntensity. When either is on
            // and the field has no DensityRepr, a 1x1x1 dummy volume binds and
            // the factor is exactly 1; nothing crashes and nothing asks.
            //
            // volumeExtinction is an EXPONENT on the transmittance rather than a
            // linear mix, so 1 is the physically honest answer and >1 is the
            // "more dust" grade a shot usually wants without re-authoring
            // DensityRepr::sigmaPerParticle (which the deferred fog march also
            // reads, and which therefore cannot be pushed for the sprites alone).
            float volumeExtinction = 0.f;
            // 0 = unshadowed (the pre-change look), 1 = fully replace the
            // sprite's own radiance with the sun term. A mix rather than a
            // multiply because an emissive nebula is not only reflecting the key.
            float volumeShadow = 0.f;
            // What the shadowed side sits at, so nothing in the volume goes
            // black — the litAmbient argument applied to the 3D transport.
            float volumeAmbient = 0.25f;
            // Scale on the sun's own contribution through T_sun. Unitless: it
            // multiplies the HG lobe, not a radiance, because the sprite's
            // colour IS its radiance here.
            float volumeSunGain = 1.0f;

            bool enabled = false;
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

            // ── CAMERA FOLLOW: the field as a TORUS around a moving centre ──
            // A weather field is a patch of world by default, which is right for
            // a snow flurry over one shore and wrong for weather — walk out of
            // the patch and the snow stops, and F4 shipped exactly that ("a
            // fixed grid that does nothing for the overall scene").
            //
            // With `follow` on, the trajectory is evaluated in world space
            // EXACTLY as before and the result is then WRAPPED TOROIDALLY into
            // a lateral box centred on followCenter():
            //
            //     xz = boxMin.xz + mod(xz - boxMin.xz, boxSize.xz)
            //
            // The wrap PERIOD is the spawn slab's own lateral size
            // (2·spawnHalfExtent.xz) and that is not a convenience — a uniform
            // distribution over exactly one period folds to a uniform
            // distribution, so the wrap is measure preserving and the snowfall
            // stays even. Any other period would bunch the flakes.
            //
            // Y IS NEVER WRAPPED. The fall is what makes snow snow; only the
            // lateral extent is a tiling of the same weather.
            //
            // Nothing about the trajectory changes, so determinism, seeking and
            // exact motion vectors all survive: the wrap is a pure function of
            // the position and the centre, applied identically to f(t) and
            // f(t − dt).
            bool follow = false;
            // The centre is SNAPPED to this lattice (metres, 0 = no snapping)
            // before it reaches the shader. Between snaps the field is exactly
            // world anchored — a slow pan shows real parallax rather than a
            // volume sliding with the eye — and the wrap seam sits still
            // instead of churning with every camera jiggle.
            //
            // CHOOSE IT AS AN INTEGER NUMBER OF DENSITY VOXELS when the field
            // also carries a DensityRepr that follows the same centre: a snap
            // of half a voxel re-phases the whole volume against its own
            // lattice and the haze visibly swims. (Same class of error as a
            // per-frame-fitted density box, which is why FireEffect's boxes are
            // fixed.)
            float followSnap = 4.f;

            // ── F5: ANALYTIC SURFACE INTERACTION (rest + splatter) ──────────
            // Snow that RESTS on what is under it and rain that SPLASHES where
            // it lands, without giving up the closed form.
            //
            // A compute pass bakes a top-down HEIGHT MAP of the scene over this
            // field's lateral footprint (particle_height_bake.comp — one ray
            // query straight down per texel, the cloud-shadow map's structure
            // with a TLAS trace where its cloud march is). The emitter then
            // SOLVES, inside f(seed, t), for the age at which the trajectory's
            // own height meets the baked height under its own drifted xz, and
            // clamps the particle there. Landing, resting, fading and (for
            // rain) splashing are all still pure functions of (seed, t), so
            // every property the mode exists for survives intact:
            //
            //   • DETERMINISM and SEEKING — nothing accumulates; setEmitterTime
            //     to any t reproduces the resting flakes exactly as well as the
            //     falling ones.
            //   • EXACT MOTION VECTORS — a resting particle writes pos ==
            //     prevPos bit-identically, i.e. zero motion, which is the
            //     truth about a flake lying on a roof.
            //   • ZERO PER-PARTICLE CPU — the host writes one 64 B record.
            //
            // ── LIMITS, stated up front rather than discovered ──────────────
            //   • Flakes rest and FADE. They do not accumulate into a snowpack:
            //     a snowpack is a change to the SURFACE (geometry, albedo,
            //     material), not to the particles, and it is a separate
            //     project.
            //   • The bake sees the scene AS IT WAS AT BAKE TIME. Dynamic
            //     objects do not carve flake shadows unless something re-bakes;
            //     the renderer re-bakes on a structural scene change (the same
            //     trigger as an entry-list rebuild) and whenever the follow
            //     centre snaps, and that is all.
            //   • A top-down height map has no overhangs. A flake rests on the
            //     FIRST surface below the cloud, which is what snow does; it
            //     will not settle on a shelf under a table.
            struct Surface {
                bool enabled = false;

                // Half-size of the SQUARE bake footprint, field-local metres.
                // 0 = use spawnHalfExtent.xz — which is also the toroidal wrap
                // period, so a FOLLOWING field's bake covers exactly the box
                // its flakes are folded into, no more and no less.
                float extent = 0.f;
                // Texels per axis. Clamped to [16, 1024]; 256 over a 48 m box
                // is 19 cm per texel and 256 KB per ring slot.
                //
                // The map is sampled NEAREST, deliberately. Bilinear across a
                // roof edge interpolates between the roof and the ground three
                // metres below it and rests flakes in mid-air on the ramp
                // between them; nearest quantises the LANDING POINT in xz to a
                // texel and puts every flake on a real surface.
                std::uint32_t resolution = 256;

                // The vertical search band, FIELD-LOCAL. Rays start at
                // `searchTop` and run down `searchTop - searchBottom` metres.
                // searchTop == searchBottom (the default pair) means "derive
                // it": the top is the spawn slab's own ceiling and the bottom
                // is one lifetime of fall below it, which is the band the
                // trajectory can actually occupy.
                float searchTop    = 0.f;
                float searchBottom = 0.f;

                // Metres above the baked surface the particle's CENTRE rests.
                // A flake proxy is a solid with a radius, so 0 buries half of
                // it; the sensible value is around the particle's own size.
                float bias = 0.f;

                // How long a landed particle holds its position before it
                // starts to fade, seconds, and the hashed +/- fraction of that
                // per slot. A field whose flakes all vanish together reads as a
                // blink; the jitter is the whole cost of avoiding it.
                float restSeconds = 3.f;
                float restJitter  = 0.6f;
                // Seconds of shrink-to-nothing after the rest. The fade is a
                // RADIUS ramp — w is the radius under WSemantic::Radius, so the
                // mesh proxy and the billboard both take it for free and no
                // alpha channel is needed anywhere.
                float fadeSeconds = 1.2f;

                // ── Rain: the splash ────────────────────────────────────────
                // > 0 makes a landed particle spend this long as an expanding,
                // flattened RING at its landing point instead of resting as a
                // drop. The ring is drawn by the billboard representation,
                // which recovers the splash's phase from the radius the emitter
                // wrote into w (see BillboardRepr and the vertex shader): a
                // scale ABOVE 1 can only mean a splash, because nothing else in
                // the lifecycle ever grows a particle.
                float splashSeconds = 0.f;
                // Ring radius at the end of the splash, in multiples of the
                // particle's own radius. Also the decode divisor, so the
                // renderer keeps it in step with the billboard record.
                float splashGrow = 7.f;
            };
            Surface surface;

            std::uint32_t seed = 20260812u;
        };

        // Throws std::invalid_argument on capacity == 0. See the churn contract
        // in the file header before choosing a capacity — and, for
        // Ownership::Interop, the reproducibility forfeit stated there.
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

        // Turn the billboard representation on: every live particle draws one
        // camera-facing additive quad, sized from its own radius times
        // `sizeScale`, composited after the upscalers (Vulkan only).
        //
        // Use this rather than setting billboardRepr().enabled by hand — it is
        // the call that makes the two colours explicit, and they are what the
        // look is. Everything else on BillboardRepr stays live-editable.
        void setBillboardRepr(const Color& colorHot, const Color& colorCool,
                              float intensity = 1.f, float sizeScale = 1.f);

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
        //
        // THROWS on an Ownership::Interop field for the same reason — its
        // positions are written by the foreign device API, and host data
        // submitted here would go nowhere — with ONE exception:
        // hostFallback() below.
        //
        // `dtSec` is the interval this submit ADVANCED the pool over — the same
        // role setEmitterTime's dt plays for a Renderer field, and used for the
        // same one thing: turning BillboardRepr::stretchSeconds into a streak
        // length under Config::hostStableSlots. Leave it 0 (the default) and
        // the field assumes 1/60 s; it is only read when the stretch is on.
        void submit(const void* pxVec4Array, std::uint32_t n, float dtSec = 0.f);
        // Seconds the last submit advanced the pool over. See submit().
        [[nodiscard]] float hostDt() const { return hostDt_; }

        // ── Ownership::Interop: the fallback leg ────────────────────────────
        // Set by the RENDERER, once, when it discovers that this device cannot
        // export memory to a foreign API at all (no VK_KHR_external_memory_*),
        // so no import and no device-to-device copy is possible. The renderer
        // then allocates the field exactly as it would a HostRing one and says
        // so on stderr; submit() becomes legal, and an application that kept
        // its pull-to-host path (it should — it is the A/B leg) feeds the field
        // through it with no other change. A field that silently rendered
        // nothing on such a device would be the worst of the three outcomes,
        // which is why this exists rather than a throw at create().
        //
        // An APPLICATION may set it too, for the mirror-image failure: the
        // export succeeded but the foreign API refused to IMPORT it, so again
        // nothing will ever write the device buffer and the host path is the
        // only one left. It is a one-way switch either way.
        [[nodiscard]] bool hostFallback() const { return hostFallback_; }
        // The staging block is allocated HERE rather than at construction, so
        // an Interop field that works costs the host not one byte for the
        // ability to fall back.
        void setHostFallback() {
            hostFallback_ = true;
            host_.resize(config_.capacity);
        }

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

        // Move the centre of the toroidal follow box (EmitterParams::follow).
        // Pass the CAMERA's world position — snowfall that follows anything
        // else is not weather. WORLD space; the backend converts to the field's
        // own space with the field's world matrix.
        //
        // The value is snapped to EmitterParams::followSnap here, and
        // followCenter() reads back the SNAPPED point, which is the number a
        // caller must use to place anything that has to agree with the wrap box
        // — above all the field's own DensityRepr::center. Reading the camera
        // position twice and snapping it twice would work until someone changed
        // one of the two expressions.
        //
        // Free to call every frame (it writes two floats) and DETERMINISTIC by
        // construction: under a scripted camera the snapped centre is a pure
        // function of the frame index, so a --shot/--seq capture is still a
        // function of its frame index and nothing else. THROWS on a
        // non-Renderer field.
        void setFollowCenter(const Vector3& worldCenter);
        [[nodiscard]] const Vector3& followCenter() const { return followCenter_; }

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

        // Per-particle appearance, as n vec4s in (r, g, b, a) order: rgb is
        // LINEAR HDR radiance (the same domain BillboardRepr::colorHot is in)
        // and a is reserved for the phase-2 alpha-over opacity. Requires
        // Config::attributes.
        //
        // WRITE-ONCE by contract, for exactly the reason setOrientations is:
        // the device buffer is a single instance, not a ring, so rewriting it
        // while frames are in flight is a host write to memory the GPU may be
        // reading. A sim that needs per-frame colours wants the INTEROP leg,
        // where the attribute buffer is exported alongside the positions and
        // the foreign API writes it device-to-device — which is the mode this
        // feature exists for.
        //
        // THROWS on an Ownership::Interop field that is not in hostFallback():
        // its attributes are written by the foreign API, and host bytes here
        // would go nowhere — the same rule, and the same one exception, as
        // submit().
        void setAttributes(const float* rgba, std::uint32_t n);

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

        // rgba floats, 4 per particle, sized to capacity once setAttributes has
        // been called and empty otherwise. The serial is 0 until then, which is
        // what lets the backend skip the upload for an interop field whose
        // attributes only ever arrive on the device.
        [[nodiscard]] const std::vector<float>& hostAttributes() const { return attr_; }
        [[nodiscard]] std::uint64_t attributeSerial() const { return attrSerial_; }

        explicit ParticleField(const Config& config);
        ~ParticleField() override = default;

    private:
        Config        config_;
        std::uint32_t liveCount_  = 0;
        std::uint64_t dataSerial_ = 1;// bumped by submit/setLiveCount
        bool          hostFallback_ = false;// Interop on a device that cannot export

        std::vector<ParticlePos> host_;
        // Ownership::HostRing: what the last submit says its own step was.
        float                     hostDt_ = 1.f / 60.f;
        std::vector<std::int16_t> ori_;// snorm16x4, 4 per particle
        std::uint64_t             oriSerial_ = 0;
        std::vector<float>        attr_;// rgba, 4 per particle
        std::uint64_t             attrSerial_ = 0;

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
        // SNAPPED world centre of the follow box. Its default of the origin is
        // the honest one: a follow field nobody drives is a patch centred on
        // the world origin, i.e. exactly the pre-follow behaviour of a field
        // sitting at its own position.
        Vector3       followCenter_{0.f, 0.f, 0.f};
    };

}// namespace threepp

#endif//THREEPP_PARTICLEFIELD_HPP
