// GPU granular material on a conveyor: PhysX 5.5 PxPBDParticleSystem grains
// poured from a chute onto the belt from extras/conveyor, carried along it,
// AROUND A BEND, and spilling off the discharge end into a stockpile. Two lanes
// run side by side with contrasting PBD materials — grippy GRAVEL against slick
// PELLETS — so the difference is in how each one piles, not just what colour it
// is.
//
// The belt drives the grains by the same mechanism it drives soft bodies — a
// kinematic collider whose target is advanced along travel each substep and
// teleported back afterwards (see ConveyorPhysics). No special case was needed
// for particles: PBD reads the contact velocity of a kinematic rigid exactly
// like a rigid body does, so the belt "just works" once the granular material's
// friction is high enough to be dragged. The bend is the same idea turned
// through 90 degrees of thinking: one kinematic body ROTATING about the arc
// centre, so every contact across the belt width reads its own surface speed.
//
// PBD particles are a CUDA-only PhysX feature with no CPU fallback. This file
// COMPILES anywhere PhysX does — the headers ship regardless — and at runtime
// prints why and exits 0 when there is no GPU, so CI can build and run it.
//
//   granular_conveyor                       interactive (pick a backend; live HUD)
//   granular_conveyor --shot pbd_belt       headless captures -> aaa_caps/
//   granular_conveyor --selftest            headless numeric gates, no window
//   granular_conveyor --bench               particles-vs-fps table
//   granular_conveyor --count 200000        particle budget (split over lanes)
//   granular_conveyor --rate 4000           grains per second per lane
//   granular_conveyor --drop                no belt: phase-1 block-drop test
//   granular_conveyor --gpu-instances off   A/B the GPU per-instance passes
//   granular_conveyor --field im            InstancedMesh field (Vulkan A/B control)
//   granular_conveyor --field pf            ParticleField (Vulkan default)
//   ... --cam 6,3,7 --look 4,0.5,-3 --fov 40    reframe without rebuilding
//
// Each lane turns 45 degrees away from its neighbour on the way to its
// stockpile, so the two belts make a V and the default framing sits out on +X
// looking back into it: both chutes, both bends, both piles, nothing cropped
// even at the end of the run.
//
// One other framing worth keeping, found by looking at captures:
//   --cam 1.2,9,2.6 --look 1.2,1.9,2.55 --fov 34   straight down onto the
//       pellet lane's bend. This is the view that shows what a curved belt
//       does to a granular load: the bed rides hard against the OUTER
//       skirtboard through the arc and leaves a clean gap along the inner one.
// It wants a frame while the chute is still pouring: a lane empties in
// (capacity/2)/rate seconds, so --frames past that shows a stopped machine.

#include "threepp/threepp.hpp"

#include "threepp/extras/conveyor/ConveyorPhysics.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/physx/PhysxParticles.hpp"
#include "threepp/objects/ParticleField.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include "capture_util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace threepp;
namespace cv = threepp::conveyor;

namespace {

    // ── Scene dimensions, in one place ───────────────────────────────────────
    //
    // These are not free: a belt is a VOLUMETRIC pump, and the pour has to fit
    // through it. Capacity is (width x usable bed depth x speed) m^3/s, and one
    // grain occupies about spacing^3 loose, so
    //
    //     grains/s <= width * (0.85 * guideHeight) * speed / spacing^3
    //
    // Overrun it and the bed climbs over the guides and stops conveying — the
    // belt can only drag the layer it touches, so a deep column just sits there
    // slipping (measured: pour 3x capacity and a cohort moves at 0.6x belt
    // speed with grain heaped 0.8 m over a 0.16 m guide). At the numbers below
    // the ceiling is ~3000 grains/s per lane, which is what --rate defaults to.
    //
    // The guides are TALLER than that formula's usable depth on purpose: real
    // skirtboards are, and a measured 2800 grains/s settles into a 0.29 m bed,
    // so a 0.22 m guide had grain walking over the edge all the way down.
    constexpr float kBeltY = 1.90f;    // conveying surface height (a stacker)
    constexpr float kBeltX0 = -3.5f;   // inlet end
    constexpr float kBeltWidth = 1.00f;
    constexpr float kBeltSpeed = 2.2f; // m/s along travel
    constexpr float kGuideHeight = 0.32f;
    // A metre of belt behind the pour, so grain that bounces backwards off the
    // bed is picked up again instead of dribbling off the tail drum.
    constexpr float kChuteX = -2.50f;

    // ── The bend ─────────────────────────────────────────────────────────────
    //
    // Each lane runs straight out of its chute, turns 45 degrees AWAY from its
    // neighbour, and discharges. The turn is the part worth filming. PhysX's
    // own conveyor feature is a surface velocity carried on a rigid SHAPE,
    // which the GPU particle and deformable solvers never read — NVIDIA lists
    // "deformable bodies and particles do not support conveyor belts and will
    // not contact/fall through" among Omniverse Physics' current limitations,
    // and tells you to use rigid bodies instead. ConveyorPhysics does not use
    // that feature at all: a bend is ONE kinematic body rotating about the
    // vertical axis through the arc centre, tiled by GPU-cooked convex wedges,
    // and a PBD grain reads its contact velocity exactly like it reads any
    // other kinematic rigid's. There is no particle special case anywhere in
    // this file.
    //
    // Holding its line through the turn costs a grain v^2/r of lateral
    // acceleration, and only friction against the belt supplies it: 2.2 m/s^2
    // (0.22 g) at 2.2 m/s and r = 2.2 m. The bed leans visibly into the outer
    // skirtboard on the way round, which is the shot.
    //
    // The radius is chosen for that lean and for a long readable arc, NOT to
    // keep the load on. Halving it was tried: r = 1.0 m more than doubles the
    // demand (0.49 g) and lost 22 grains against 27 at r = 2.2, which is noise.
    // The bed simply heaps harder against the outer guide. It is the SKIRTBOARD
    // that holds a granular load through a bend, not friction — which is what a
    // real curved conveyor is built on, and worth knowing before blaming the
    // solver for a spill.
    constexpr float kBendX = 0.90f;      // corner waypoint, on the inlet run
    constexpr float kBendRadius = 2.20f; // centreline radius of the fillet
    constexpr float kBendDeg = 45.f;     // turn angle, away from the other lane
    constexpr float kRunOut = 3.20f;     // corner waypoint to discharge
    // Lane separation, measured at the INLET. The outward turn splays the two
    // discharges a further 2 x 2.3 m apart, so two 2 m-radius stockpiles clear
    // each other comfortably even though the chutes stand close enough to read
    // as one machine. (A merged heap has one profile, and comparing the two
    // profiles is the whole point of running the lanes side by side.)
    constexpr float kLaneZ = 1.75f;
    // Where a stockpile begins: this far past the discharge drum measured ALONG
    // travel, so pile probes never measure the falling stream.
    constexpr float kPileGap = 1.40f;

    // Plan-view length of a polyline (the belts are horizontal).
    float polylineLength(const std::vector<Vector3>& pts) {
        float len = 0.f;
        for (std::size_t i = 0; i + 1 < pts.size(); ++i) len += pts[i].distanceTo(pts[i + 1]);
        return len;
    }

    // ── Grain field: two implementations, one interface ──────────────────────
    //
    // The visual half of a lane — "point N proxy meshes at N particle
    // positions" — has two implementations that must be selectable in the SAME
    // binary, because the only honest way to judge the second is to interleave
    // it against the first (--field im|pf):
    //
    //   GrainField  one InstancedMesh, N instance matrices written per frame.
    //               The only path GL has, and a first-class capability there:
    //               measured at 500k moving grains, 26 fps whole-demo, with the
    //               GL render half at ~10.8 ms. --api gl always uses it.
    //   PfField     one threepp::ParticleField. VULKAN ONLY. The CPU writes one
    //               memcpy of positions and the renderer's per-frame work drops
    //               to O(1) in the grain count.
    struct IGrainVisual {
        virtual ~IGrainVisual() = default;
        // Per-frame: point the field at `n` live grain positions (PxVec4, w =
        // inverse mass).
        virtual void update(const ::physx::PxVec4* positions, unsigned n) = 0;
        virtual Object3D& object() = 0;
        [[nodiscard]] virtual std::shared_ptr<Object3D> shared() const = 0;
    };

    // Uniform random orientation (Shoemake), one per grain, from `seed`. Shared
    // by both implementations so the two fields orient their grains
    // IDENTICALLY — without that the A/B capture compares two different piles
    // of rocks and says nothing about the renderer.
    std::vector<Quaternion> grainOrientations(unsigned capacity, unsigned seed) {
        std::vector<Quaternion> out(capacity);
        std::mt19937 rng{seed};
        std::uniform_real_distribution<float> uni(0.f, 1.f);
        for (unsigned i = 0; i < capacity; ++i) {
            const float u1 = uni(rng), u2 = uni(rng), u3 = uni(rng);
            const float s1 = std::sqrt(1.f - u1), s2 = std::sqrt(u1);
            out[i].set(s1 * std::sin(math::TWO_PI * u2), s1 * std::cos(math::TWO_PI * u2),
                       s2 * std::sin(math::TWO_PI * u3), s2 * std::cos(math::TWO_PI * u3));
        }
        return out;
    }

    // ── Instanced grain field ────────────────────────────────────────────────
    //
    // One InstancedMesh per particle group, created ONCE at the group's full
    // capacity. Two reasons it is never re-created and its count() only ever
    // steps coarsely:
    //   • The Vulkan deferred renderer expands an InstancedMesh into count()
    //     TLAS entries. Changing count() invalidates that expansion — an
    //     entry-list rebuild plus a device wait, and the TAA history for
    //     everything after it in the list. Once per few thousand grains is
    //     fine; once per frame is not.
    //   • Per-frame work then reduces to writing three floats per grain. The
    //     3x3 block (a fixed random orientation, so a pile does not read as a
    //     lattice of identically-facing rocks) is written once at setup and
    //     never touched again — PBD particles carry no orientation anyway.
    class GrainField: public IGrainVisual {

    public:
        GrainField(const std::shared_ptr<BufferGeometry>& geometry,
                   const std::shared_ptr<Material>& material, unsigned capacity,
                   unsigned seed)
            : rot_(std::size_t(capacity) * 9), capacity_(capacity) {

            mesh_ = InstancedMesh::create(geometry, material, capacity);
            mesh_->setCount(0);
            mesh_->frustumCulled = false;// the field spans the whole scene

            const auto quats = grainOrientations(capacity, seed);
            auto& e = mesh_->instanceMatrix()->array();
            std::memset(e.data(), 0, e.size() * sizeof(float));
            Matrix4 m;
            for (unsigned i = 0; i < capacity_; ++i) {
                // Uniform random orientation, baked in permanently.
                m.makeRotationFromQuaternion(quats[i]);
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 3; ++r)
                        rot_[std::size_t(i) * 9 + c * 3 + r] = m.elements[c * 4 + r];
                e[std::size_t(i) * 16 + 15] = 1.f;
            }
        }

        void update(const ::physx::PxVec4* positions, unsigned n) override {

            n = std::min(n, capacity_);
            auto& e = mesh_->instanceMatrix()->array();
            float* base = e.data();
            // Newly claimed slots get their (fixed) rotation written in; from
            // then on only the translation moves. Slots past `n` keep a zero
            // 3x3, which collapses the instance to a point — no pixels — so the
            // up-to-kStep-1 spares inside the current count step are invisible.
            for (unsigned i = claimed_; i < n; ++i) {
                float* b = base + std::size_t(i) * 16;
                const float* r = rot_.data() + std::size_t(i) * 9;
                b[0] = r[0]; b[1] = r[1]; b[2] = r[2];
                b[4] = r[3]; b[5] = r[4]; b[6] = r[5];
                b[8] = r[6]; b[9] = r[7]; b[10] = r[8];
            }
            claimed_ = std::max(claimed_, n);

            for (unsigned i = 0; i < n; ++i) {
                float* b = base + std::size_t(i) * 16;
                b[12] = positions[i].x;
                b[13] = positions[i].y;
                b[14] = positions[i].z;
            }
            mesh_->instanceMatrix()->needsUpdate();

            const unsigned want = std::min(capacity_, ((n + kStep - 1) / kStep) * kStep);
            if (want != mesh_->count()) mesh_->setCount(want);
        }

        Object3D& object() override { return *mesh_; }
        [[nodiscard]] std::shared_ptr<Object3D> shared() const override { return mesh_; }

    private:
        static constexpr unsigned kStep = 4096;

        std::shared_ptr<InstancedMesh> mesh_;
        std::vector<float> rot_;// 3x3 per instance, written once, read on claim
        unsigned capacity_ = 0;
        unsigned claimed_ = 0;
    };

    // ── ParticleField grain field (Vulkan only) ──────────────────────────────
    //
    // The same picture, produced without the renderer ever learning how many
    // grains there are. Everything GrainField does per grain per frame — 3
    // floats into a 64-byte instance matrix, an InstancedMesh count step, and
    // downstream of that one MeshEntry / DrawInfo / motion matrix / TLAS
    // instance descriptor per grain — collapses to:
    //
    //   • ONE memcpy of the PBD position block. PxVec4 IS ParticlePos, so
    //     there is no repack; that memcpy is the whole `field` bench column.
    //   • ONE entry in the renderer, whatever the capacity. The grain count
    //     reaches the GPU as a 4-byte device-side copy into the draw command's
    //     instanceCount and never reaches the CPU at all.
    //
    // Orientations are written ONCE, at construction, as 8 B of snorm16x4 per
    // grain on the device — against GrainField's 36 B of host floats per grain
    // plus the claim-time write into the instance matrix.
    class PfField: public IGrainVisual {

    public:
        PfField(const std::shared_ptr<BufferGeometry>& geometry,
                const std::shared_ptr<Material>& material, unsigned capacity,
                unsigned seed, float radius) {

            ParticleField::Config cfg;
            cfg.capacity = capacity;
            cfg.ownership = ParticleField::Ownership::HostRing;
            // PhysX writes inverse mass into w, which says nothing about size,
            // so the proxy geometry (authored at `radius`) draws at scale 1 and
            // a negative w is the dead-slot predicate. Interop mode will hand
            // this same buffer straight to CUDA.
            cfg.wSemantic = ParticleField::WSemantic::InvMass;
            cfg.uniformRadius = radius;
            cfg.orientations = true;
            field_ = ParticleField::create(cfg);
            field_->setMeshRepr(geometry, material);
            field_->frustumCulled = false;

            // Byte-identical orientations to GrainField's, from the same seed,
            // before the snorm16x4 quantisation the buffer applies.
            const auto quats = grainOrientations(capacity, seed);
            std::vector<float> xyzw(std::size_t(capacity) * 4u);
            for (unsigned i = 0; i < capacity; ++i) {
                xyzw[std::size_t(i) * 4u + 0u] = quats[i].x;
                xyzw[std::size_t(i) * 4u + 1u] = quats[i].y;
                xyzw[std::size_t(i) * 4u + 2u] = quats[i].z;
                xyzw[std::size_t(i) * 4u + 3u] = quats[i].w;
            }
            field_->setOrientations(xyzw.data(), capacity);
        }

        void update(const ::physx::PxVec4* positions, unsigned n) override {
            // The entire per-frame CPU cost of the representation. No loop, no
            // count step, no entry-list consequence.
            field_->submit(positions, n);
        }

        Object3D& object() override { return *field_; }
        [[nodiscard]] std::shared_ptr<Object3D> shared() const override { return field_; }

    private:
        std::shared_ptr<ParticleField> field_;
    };

    // ── Discharge dust (--dust, Vulkan only) ─────────────────────────────────
    //
    // Grains slamming into the stockpile kick up fines. The plume is a
    // ParticleField with ONLY DensityRepr enabled: nothing draws these
    // particles — they exist as σ_t in a world-anchored volume the froxel
    // passes sample, so the dust occludes the pile, catches the sun and
    // haloes the discharge for one compute dispatch (~0.25 ns/particle).
    //
    // The particles are given STRUCTURE deliberately: emitted in per-frame
    // clumps at the impact point, carried up by buoyancy that dies with age,
    // spread by a cheap curl of sines. Uniform-random dust in a box is the
    // worst-looking dust there is — Poisson blotch with no shape — and this
    // emitter is the difference between that and a plume.
    class DustPlume {

    public:
        DustPlume(Scene& scene, const Vector3& impact, const Vector3& outDir, unsigned seed)
            : impact_(impact), out_(outDir), rng_(seed) {

            ParticleField::Config cfg;
            cfg.capacity = kCap;
            field_ = ParticleField::create(cfg);
            const Vector3 center(impact.x + outDir.x * 0.5f, 1.25f, impact.z + outDir.z * 0.5f);
            field_->setDensityRepr(center, Vector3(3.0f, 1.7f, 3.0f), kSigma, 64);
            field_->densityRepr().albedo = Color(0.50f, 0.43f, 0.33f);// dirty fines, not steam
            field_->densityRepr().anisotropy = 0.35f;// sunlit dust scatters forward
            scene.add(field_);

            pts_.assign(kCap, ParticlePos{0.f, 0.f, 0.f, -1.f});// all dead
            vel_.assign(kCap, Vector3());
            age_.assign(kCap, 0.f);
            life_.assign(kCap, 1.f);
        }

        void tick(float dt) {
            t_ += dt;
            // Ramp in as the first grains actually reach the discharge (~3 s of
            // belt travel): dust before there is anything to raise it reads as
            // a smoke machine.
            const float ramp = std::clamp((t_ - 3.2f) / 2.5f, 0.f, 1.f);
            emitAcc_ += kRate * ramp * dt;
            std::uniform_real_distribution<float> u(-1.f, 1.f);
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            // A puff every ~1/3 s on top of the base rate: impacts are lumpy,
            // and the lumps are what make it read as kicked-up rather than leaked.
            if (ramp > 0.f && u01(rng_) < dt * 3.f) emitAcc_ += 55.f;

            for (int n = int(emitAcc_); n > 0; --n) {
                const unsigned i = cursor_++ % kCap;
                const float a = u(rng_) * math::PI;
                // Two sources, the way a real transfer point sheds fines: most
                // rise off the SPLASH at the pile (wide, slow), the rest are
                // stripped off the falling STREAM between drum and pile
                // (narrow, already moving) — which visually ties the plume to
                // the pour instead of leaving a puff hovering beside it.
                const bool stream = u01(rng_) < 0.35f;
                if (stream) {
                    const float h = 0.35f + 1.25f * u01(rng_);// along the fall
                    const float r = 0.16f + 0.10f * u01(rng_);
                    pts_[i] = {impact_.x - out_.x * 0.35f + std::cos(a) * r,
                               h,
                               impact_.z - out_.z * 0.35f + std::sin(a) * r, 1.f};
                    vel_[i].set(out_.x * 0.35f + 0.25f * u(rng_), -0.15f + 0.45f * u01(rng_),
                                out_.z * 0.35f + 0.25f * u(rng_));
                } else {
                    const float r = 0.65f * std::sqrt(u01(rng_));
                    pts_[i] = {impact_.x + std::cos(a) * r + out_.x * 0.25f,
                               impact_.y + 0.30f + 0.30f * u01(rng_),
                               impact_.z + std::sin(a) * r + out_.z * 0.25f, 1.f};
                    vel_[i].set(out_.x * (0.30f + 0.40f * u01(rng_)) + 0.30f * u(rng_),
                                0.50f + 0.70f * u01(rng_),
                                out_.z * (0.30f + 0.40f * u01(rng_)) + 0.30f * u(rng_));
                }
                age_[i] = 0.f;
                life_[i] = 2.8f + 2.2f * u01(rng_);
            }
            emitAcc_ -= float(int(emitAcc_));

            const float drag = std::exp(-1.1f * dt);
            for (unsigned i = 0; i < kCap; ++i) {
                if (pts_[i].w < 0.f) continue;
                age_[i] += dt;
                if (age_[i] > life_[i]) {
                    pts_[i].w = -1.f;
                    continue;
                }
                // Buoyancy dies with age; the curl is three phase-shifted sines
                // sampled at the particle — cheap, divergence-ish-free, and
                // enough to shear the column into wisps.
                const float k = 2.1f;
                const Vector3 p(pts_[i].x, pts_[i].y, pts_[i].z);
                const float sway = 0.45f * (1.f - age_[i] / life_[i]);
                vel_[i].x += (std::sin(p.y * k + t_ * 1.3f) + 0.5f * std::sin(p.z * k * 1.7f + t_)) * sway * dt;
                vel_[i].z += (std::cos(p.y * k * 1.3f - t_ * 1.1f) + 0.5f * std::sin(p.x * k * 1.9f - t_)) * sway * dt;
                vel_[i].y += (0.45f * (1.f - age_[i] / life_[i]) - 0.12f) * dt;
                vel_[i].multiplyScalar(drag);
                pts_[i].x += vel_[i].x * dt;
                pts_[i].y += vel_[i].y * dt;
                pts_[i].z += vel_[i].z * dt;
            }
            field_->submit(pts_.data(), kCap);
        }

    private:
        static constexpr unsigned kCap = 40000;
        static constexpr float kRate = 3200.f;// particles per second, before puffs
        // Thin: the dust's only light is the isotropic ambient/skylight term
        // (the sun march owns height fog, not particle density — a phase-3
        // candidate), and an optically thick plume under a flat light reads as
        // cotton. Translucent wisps read as dust.
        static constexpr float kSigma = 0.30f;

        std::shared_ptr<ParticleField> field_;
        Vector3 impact_, out_;
        std::vector<ParticlePos> pts_;
        std::vector<Vector3> vel_;
        std::vector<float> age_, life_;
        std::mt19937 rng_;
        unsigned cursor_ = 0;
        float emitAcc_ = 0.f, t_ = 0.f;
    };

    // ── The pour ─────────────────────────────────────────────────────────────
    //
    // A chute discharges `rate` grains per second. Emission has to be
    // non-overlapping — PBD depenetrates an overlap violently unless the clamp
    // does all the work — so grains are placed on a LATTICE inside a thin slab
    // and jittered by a fraction of the cell, never sampled uniformly. The slab
    // grows in layers only as far as the burst needs.
    class Chute {

    public:
        Chute(const Vector3& mouth, float width, float spacing, unsigned seed)
            : mouth_(mouth), spacing_(spacing), rng_(seed) {

            cellsX_ = std::max(1u, unsigned(0.34f / (spacing * 1.05f)));
            cellsZ_ = std::max(1u, unsigned(width * 0.70f / (spacing * 1.05f)));
        }

        // Accumulates a fractional rate so a 42.7-grains-per-frame pour does not
        // quantise to 42. Returns the positions to hand to Group::emit.
        const std::vector<Vector3>& tick(float dt, float rate) {

            pending_ += dt * rate;
            const auto want = unsigned(pending_);
            pending_ -= float(want);
            out_.clear();
            if (want == 0) return out_;

            const float cell = spacing_ * 1.05f;
            const unsigned perLayer = cellsX_ * cellsZ_;
            const unsigned layers = (want + perLayer - 1) / perLayer;
            slots_.resize(std::size_t(perLayer) * layers);
            std::iota(slots_.begin(), slots_.end(), 0u);
            std::shuffle(slots_.begin(), slots_.end(), rng_);

            std::uniform_real_distribution<float> j(-0.2f * spacing_, 0.2f * spacing_);
            out_.reserve(want);
            for (unsigned i = 0; i < want; ++i) {
                const unsigned s = slots_[i];
                const unsigned ix = s % cellsX_;
                const unsigned iz = (s / cellsX_) % cellsZ_;
                const unsigned iy = s / perLayer;
                out_.emplace_back(mouth_.x + (float(ix) - float(cellsX_ - 1) * 0.5f) * cell + j(rng_),
                                  mouth_.y + float(iy) * cell + j(rng_),
                                  mouth_.z + (float(iz) - float(cellsZ_ - 1) * 0.5f) * cell + j(rng_));
            }
            return out_;
        }

        [[nodiscard]] float slabTop(unsigned peakLayers = 3) const {
            return mouth_.y + float(peakLayers) * spacing_ * 1.05f;
        }

    private:
        Vector3 mouth_;
        float spacing_;
        std::mt19937 rng_;
        unsigned cellsX_ = 1, cellsZ_ = 1;
        float pending_ = 0.f;
        std::vector<unsigned> slots_;
        std::vector<Vector3> out_;
    };

    // ── Conveyor spec + visuals ──────────────────────────────────────────────

    // One lane: a belt leaving the inlet along +X, a rounded corner turning away
    // from the other lane, and a run-out to the discharge — with a side guide on
    // each edge so a granular BED (unlike a crate) does not simply run off the
    // sides. The corner is authored as a cornerRadius on the middle waypoint,
    // which cornerFillet resolves into an arc TANGENT to both straights, so the
    // belt cannot kink however the geometry above is retuned.
    //
    // The guides are authored the way the editor authors them: two points in
    // plan, resolved by followWall against the same centreline the ribbon uses.
    // That is what makes a guide hug the bend instead of cutting its chord —
    // two world-space points at the ends would be a straight line across the
    // turn, and grain would leave through the gap.
    cv::ConveyorSpec laneSpec(float z) {

        const float side = z < 0.f ? -1.f : 1.f;// turn away from the other lane
        const float turn = kBendDeg * math::DEG2RAD;
        const Vector3 corner(kBendX, kBeltY, z);
        const Vector3 discharge(corner.x + kRunOut * std::cos(turn), kBeltY,
                                corner.z + side * kRunOut * std::sin(turn));

        cv::ConveyorSpec s;
        s.waypoints = {cv::Waypoint{Vector3(kBeltX0, kBeltY, z)},
                       cv::Waypoint{corner, kBendRadius},
                       cv::Waypoint{discharge}};
        s.width = kBeltWidth;
        s.speed = kBeltSpeed;
        s.smooth = false;// the corner is an exact fillet, not a spline
        s.frame = true;

        const auto path = cv::resamplePath(s.waypoints, s.smooth, s.samples);
        const float half = kBeltWidth * 0.5f;
        const float len = polylineLength(path);
        for (int e = -1; e <= 1; e += 2) {
            cv::WallSpec w;
            w.height = kGuideHeight;
            w.points = {cv::pointOnPath(path, 0.f, float(e) * half),
                        cv::pointOnPath(path, len, float(e) * half)};
            s.walls.push_back(std::move(w));
        }
        return s;
    }

    // Everything drawn for one conveyor, built from the SAME helpers the
    // colliders come from (so they agree by construction) — a trimmed version of
    // the editor's generator: belt ribbon, side guides, rails, legs, end drums.
    // Returns the belt material, whose texture offset is what makes the surface
    // read as moving (the geometry never does).
    std::shared_ptr<MeshStandardMaterial> addConveyorVisual(Scene& scene,
                                                            const cv::ConveyorSpec& spec) {

        const auto path = cv::resamplePath(spec.waypoints, spec.smooth, spec.samples);

        auto tex = cv::beltTexture();
        tex->repeat.set(std::max(1.f, std::round(spec.width / cv::kBeltTileLength)),
                        1.f / cv::kBeltTileLength);
        auto beltMat = MeshStandardMaterial::create();
        beltMat->color = Color(0x33383d);
        beltMat->roughness = 0.85f;
        beltMat->metalness = 0.f;
        beltMat->side = Side::Double;
        beltMat->map = tex;
        // The pattern scrolls via the texture offset with no geometric motion —
        // temporal passes need telling, or the moving pattern smears.
        beltMat->textureAnimatedHint = true;

        auto belt = Mesh::create(cv::ribbonGeometry(path, spec.width), beltMat);
        belt->receiveShadow = true;
        scene.add(belt);

        auto guideMat = MeshStandardMaterial::create();
        guideMat->color = Color(0xa8b0b6);
        guideMat->roughness = 0.45f;
        guideMat->metalness = 0.8f;
        guideMat->side = Side::Double;
        for (const auto& w : spec.walls) {
            auto followed = cv::followWall(w.points, path);
            if (followed.size() < 2) continue;
            auto mesh = Mesh::create(cv::wallGeometry(followed, w.height), guideMat);
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            scene.add(mesh);
        }

        auto frameMat = MeshStandardMaterial::create();
        frameMat->color = Color(0x8e959c);
        frameMat->roughness = 0.5f;
        frameMat->metalness = 0.85f;
        const auto profile = cv::FrameProfile::forWidth(spec.width);
        for (int side = -1; side <= 1; side += 2) {
            auto rail = Mesh::create(cv::railGeometry(path, spec.width, side, profile), frameMat);
            rail->castShadow = true;
            scene.add(rail);
        }
        auto legBox = BoxGeometry::create(profile.legThickness, 1.f, profile.legThickness);
        for (const auto& leg : cv::legTransforms(path, spec.width, 0.f, profile)) {
            auto mesh = Mesh::create(legBox, frameMat);
            mesh->position.copy(leg.center);
            mesh->quaternion.copy(leg.orientation);
            mesh->scale.y = leg.length;
            mesh->castShadow = true;
            scene.add(mesh);
        }
        auto drum = CylinderGeometry::create(profile.drumRadius, profile.drumRadius,
                                             spec.width + 2.f * profile.railThickness, 20);
        for (const auto& d : cv::endDrumTransforms(path, profile)) {
            auto mesh = Mesh::create(drum, frameMat);
            mesh->position.copy(d.center);
            mesh->quaternion.copy(d.orientation);
            mesh->castShadow = true;
            scene.add(mesh);
        }
        return beltMat;
    }

    // A tail skirt: a plate across the inlet end of the belt. ConveyorPhysics
    // only builds guides that FOLLOW the path, so the one collider a granular
    // belt needs and a crate belt does not — a back wall at the tail drum — is
    // added here. Without it the heap under the chute walks backwards against
    // travel and dribbles off the end.
    void addTailSkirt(Scene& scene, PhysxWorld& world, float z,
                      const std::shared_ptr<Material>& material) {

        using namespace ::physx;

        constexpr float kThick = 0.05f;
        constexpr float kHeight = 0.45f;
        const Vector3 center(kBeltX0 - kThick * 0.5f, kBeltY + kHeight * 0.5f, z);

        auto mesh = Mesh::create(BoxGeometry::create(kThick, kHeight, kBeltWidth), material);
        mesh->position.copy(center);
        mesh->castShadow = true;
        scene.add(mesh);
        // Slick, like ConveyorPhysics' own guide material: a grippy back wall
        // would hold the heap against it instead of letting the belt pull out.
        auto* mat = world.physics().createMaterial(0.08f, 0.08f, 0.f);
        mat->setFrictionCombineMode(PxCombineMode::eMIN);
        world.addStatic(PxBoxGeometry(kThick * 0.5f, kHeight * 0.5f, kBeltWidth * 0.5f),
                        PxTransform(PxVec3(center.x, center.y, center.z)), mat);
    }

    // The chute cone + its two support posts: grain appearing in mid-air does
    // not read as a pour, and a cone floating in mid-air does not read as a
    // chute. Purely visual.
    void addChuteVisual(Scene& scene, float z, float mouthTop,
                        const std::shared_ptr<Material>& material) {

        auto cone = Mesh::create(CylinderGeometry::create(0.52f, 0.26f, 0.50f, 20, 1, true),
                                 material);
        cone->position.set(kChuteX, mouthTop + 0.22f, z);
        cone->castShadow = true;
        scene.add(cone);

        const float postTop = mouthTop + 0.47f;
        auto post = BoxGeometry::create(0.07f, postTop, 0.07f);
        for (int side = -1; side <= 1; side += 2) {
            auto leg = Mesh::create(post, material);
            leg->position.set(kChuteX, postTop * 0.5f, z + float(side) * 0.62f);
            leg->castShadow = true;
            scene.add(leg);
        }
    }

    // ── Where a grain stands, along its own belt ──────────────────────────────
    //
    // A bent belt cannot be measured along +X. Past the corner, travel has a Z
    // component, and every question below — has this grain moved DOWN the belt,
    // how deep is the bed, did it reach the discharge, did it leave over a
    // skirtboard — is a statement about the PATH, not about an axis. So each
    // lane carries a coarse copy of the same centreline the ribbon and the
    // colliders are built from, and grains are projected onto it.
    //
    // The projection is hand-rolled rather than cv::projectOntoPath because
    // that one builds its station table per call: fine for a wall edit, not for
    // 200 000 grains twice a second.
    struct LaneGeom {
        std::vector<Vector3> pts;  // decimated centreline
        std::vector<float> station;// arc length at each point
        Vector3 discharge;         // the last point
        Vector3 outDir;            // horizontal unit travel there
        float chuteStation = 0.f;  // where the pour lands

        [[nodiscard]] bool valid() const { return pts.size() >= 2; }
        [[nodiscard]] float length() const { return station.back(); }

        // Metres past the discharge along travel; negative while short of it.
        [[nodiscard]] float past(float x, float z) const {
            return (x - discharge.x) * outDir.x + (z - discharge.z) * outDir.z;
        }

        // Station (arc length, clamped to the ends) and lateral distance, in plan.
        void project(float x, float z, float& s, float& lateral) const {
            float best = std::numeric_limits<float>::max();
            s = 0.f;
            for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
                const float ax = pts[i].x, az = pts[i].z;
                const float dx = pts[i + 1].x - ax, dz = pts[i + 1].z - az;
                const float len2 = dx * dx + dz * dz;
                float t = len2 > 1e-12f ? ((x - ax) * dx + (z - az) * dz) / len2 : 0.f;
                t = std::clamp(t, 0.f, 1.f);
                const float ex = x - (ax + dx * t), ez = z - (az + dz * t);
                const float d2 = ex * ex + ez * ez;
                if (d2 < best) {
                    best = d2;
                    s = station[i] + t * (station[i + 1] - station[i]);
                }
            }
            lateral = std::sqrt(best);
        }
    };

    // Decimate the dense centreline to ~0.4 m: the projection above is O(points)
    // per grain, and half a metre of arc is far finer than any gate here reads.
    // The final point is always kept, because the discharge is a measurement
    // datum rather than a sample.
    LaneGeom makeLaneGeom(const std::vector<Vector3>& dense, const Vector3& chuteMouth) {

        LaneGeom g;
        if (dense.size() < 2) return g;
        for (const auto& p : dense) {
            if (g.pts.empty() || g.pts.back().distanceTo(p) > 0.4f) g.pts.push_back(p);
        }
        if (g.pts.back().distanceTo(dense.back()) > 1e-4f) g.pts.push_back(dense.back());

        g.station.assign(g.pts.size(), 0.f);
        for (std::size_t i = 1; i < g.pts.size(); ++i)
            g.station[i] = g.station[i - 1] + g.pts[i - 1].distanceTo(g.pts[i]);
        g.discharge = g.pts.back();
        Vector3 d = g.pts.back();
        d.sub(g.pts[g.pts.size() - 2]);
        d.y = 0.f;
        d.normalize();
        g.outDir = d;
        float lat = 0.f;
        g.project(chuteMouth.x, chuteMouth.z, g.chuteStation, lat);
        return g;
    }

    // ── Telemetry: the numbers the self-test gates on ─────────────────────────
    struct Stats {
        unsigned n = 0;
        unsigned bad = 0;// non-finite components — must stay 0
        float minY = 0.f, maxY = 0.f;
        unsigned spilled = 0;// past the discharge AND below the belt
        // Grain found under the MIDDLE of the belt run — below the deck,
        // between half a metre past the pour and a metre and a half short of
        // the discharge. Nothing can reach there except over a skirtboard, and
        // the window is drawn to exclude the two things that look like leakage
        // and are not: chute splash at the inlet, and the stockpile spreading
        // back underneath the discharge drum. The bend sits in the middle of
        // the window, so this is the bend's own failure mode — a straight belt
        // cannot throw its load sideways — kept apart from `spilled`, which is
        // where grain is SUPPOSED to leave.
        unsigned lost = 0;
        // Mean station of those losses: WHERE along the run they went over.
        float lostStation = 0.f;
        // Deepest point of the bed riding the middle of the belt, measured from
        // the conveying surface. This is the overrun alarm: once it passes the
        // guide height the belt is being poured into faster than it can pump,
        // and everything downstream of that (transport rate, pile shape) is
        // measuring a jam rather than a conveyor.
        float bedTop = 0.f;
        // Stockpile shape. `pileTop` is the crest above the floor and
        // `pileRadius` the RMS horizontal distance from the pile's own centroid,
        // so pileTop / pileRadius is a repose-angle proxy: THE number that
        // separates a grippy material from a slick one independently of colour.
        unsigned pileN = 0;
        float pileTop = 0.f;
        float pileRadius = 0.f;
    };

    // `geom` may be invalid (the --drop mode has no belt at all), in which case
    // only the belt-free numbers — count, finiteness, minY/maxY — are filled in.
    Stats measure(const ::physx::PxVec4* p, unsigned n, const LaneGeom& geom) {
        Stats s;
        s.n = n;
        if (n == 0) return s;
        s.minY = s.maxY = p[0].y;
        double px = 0, pz = 0, lostS = 0;
        for (unsigned i = 0; i < n; ++i) {
            const auto& v = p[i];
            if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
                ++s.bad;
                continue;
            }
            s.minY = std::min(s.minY, v.y);
            s.maxY = std::max(s.maxY, v.y);
            if (!geom.valid()) continue;

            const float past = geom.past(v.x, v.z);
            float st = 0.f, lat = 0.f;
            geom.project(v.x, v.z, st, lat);

            if (v.y < kBeltY - 0.3f) {
                if (past > 0.f) ++s.spilled;
                if (st > geom.chuteStation + 0.5f && st < geom.length() - 1.5f) {
                    ++s.lost;
                    lostS += st;
                }
            }
            if (past > kPileGap) {
                ++s.pileN;
                px += v.x;
                pz += v.z;
                s.pileTop = std::max(s.pileTop, v.y);
            }
            // Skip the metre downstream of the chute: grain in free fall, and
            // the heap it makes on landing, are not the conveyed bed. The
            // lateral cut is what the old x-range did implicitly on a straight
            // belt — off the deck sideways is not the bed either.
            if (st > geom.chuteStation + 1.0f && st < geom.length() - 0.3f &&
                lat < kBeltWidth && v.y > kBeltY)
                s.bedTop = std::max(s.bedTop, v.y - kBeltY);
        }
        if (s.lost) s.lostStation = float(lostS / s.lost);
        if (s.pileN > 32) {
            const double cx = px / s.pileN, cz = pz / s.pileN;
            double sq = 0;
            for (unsigned i = 0; i < n; ++i) {
                const auto& v = p[i];
                if (!std::isfinite(v.x) || !std::isfinite(v.z)) continue;
                if (geom.past(v.x, v.z) <= kPileGap) continue;
                const double dx = v.x - cx, dz = v.z - cz;
                sq += dx * dx + dz * dz;
            }
            s.pileRadius = float(std::sqrt(sq / s.pileN));
        }
        return s;
    }

    // Mean STATION of a fixed cohort — the first `n` particles ever emitted.
    // Particle index in a PxParticleBuffer is stable for the buffer's lifetime,
    // so this follows the SAME grains down the belt; a mean over everything
    // would be dragged backwards by every new grain the chute drops at the
    // inlet. Station rather than X, because past the bend X stops being travel:
    // a grain that rides the whole turn perfectly gains only cos(45 deg) of its
    // distance in X, which would read as a belt that half works.
    float cohortMeanStation(const ::physx::PxVec4* p, unsigned n, const LaneGeom& geom) {
        if (n == 0 || !geom.valid()) return 0.f;
        double sum = 0;
        unsigned good = 0;
        for (unsigned i = 0; i < n; ++i) {
            if (!std::isfinite(p[i].x) || !std::isfinite(p[i].z)) continue;
            float st = 0.f, lat = 0.f;
            geom.project(p[i].x, p[i].z, st, lat);
            sum += st;
            ++good;
        }
        return good ? float(sum / good) : 0.f;
    }

    // A jittered block of grid positions, for the phase-1 drop mode.
    std::vector<Vector3> block(unsigned count, float spacing, const Vector3& centerBottom,
                               unsigned seed) {
        std::vector<Vector3> out;
        out.reserve(count);
        const auto side = unsigned(std::ceil(std::cbrt(double(count))));
        const float d = spacing * 1.08f;
        std::mt19937 rng{seed};
        std::uniform_real_distribution<float> j(-0.12f * spacing, 0.12f * spacing);
        for (unsigned y = 0; y < side && out.size() < count; ++y)
            for (unsigned x = 0; x < side && out.size() < count; ++x)
                for (unsigned z = 0; z < side && out.size() < count; ++z)
                    out.emplace_back(centerBottom.x + (float(x) - float(side - 1) * 0.5f) * d + j(rng),
                                     centerBottom.y + float(y) * d + j(rng),
                                     centerBottom.z + (float(z) - float(side - 1) * 0.5f) * d + j(rng));
        return out;
    }

    // ── One lane: belt + chute + material + its own instanced field ──────────
    struct Lane {
        const char* name = "";
        float z = 0.f;
        PbdParticles::Group* group = nullptr;
        std::unique_ptr<IGrainVisual> field;
        std::unique_ptr<Chute> chute;
        std::shared_ptr<MeshStandardMaterial> beltMat;
        LaneGeom geom;
        float scroll = 0.f;
        Stats stats;
        float cohortS0 = 0.f, cohortS1 = 0.f;
    };

}// namespace

int main(int argc, char** argv) {

    std::string shot;
    std::vector<int> shotFrames{150, 450, 1200};
    bool selftest = false;
    bool bench = false;
    bool dropMode = false;
    bool wantDust = false;// --dust: fines rising off the discharge (Vulkan only)
    unsigned budget = 100000;
    float rate = 2800.f;
    int frames = 0;// 0 = per-mode default
    float fov = 0.f;// 0 = per-mode default
    bool forceGl = false;// --api gl: bench/capture through OpenGL instead
    bool gpuInstances = true;// --gpu-instances off: A/B the GPU instance passes
    // --field pf|im. The Vulkan default is the ParticleField; --api gl forces
    // the InstancedMesh path unconditionally (ParticleField has no GL
    // implementation, by decision — see its class header). Both are selectable
    // on Vulkan in this one binary because an A/B has to be interleaved.
    bool useParticleField = true;
    // --cam / --look come from the shared capture harness, so reframing a
    // beauty shot never needs a rebuild.
    const capture::Args cap = capture::parseArgs(argc, argv);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shot = argv[++i];
        else if (std::strcmp(argv[i], "--fov") == 0 && i + 1 < argc)
            fov = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else if (std::strcmp(argv[i], "--drop") == 0) dropMode = true;
        else if (std::strcmp(argv[i], "--dust") == 0) wantDust = true;
        else if (std::strcmp(argv[i], "--count") == 0 && i + 1 < argc)
            budget = unsigned(std::max(2, std::atoi(argv[++i])));
        else if (std::strcmp(argv[i], "--rate") == 0 && i + 1 < argc)
            rate = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) frames = std::atoi(argv[++i]);
        // --api gl lets --bench measure the SAME sim through the other backend.
        // The two are not doing the same work (GL is forward + shadow maps; the
        // Vulkan path adds ray-traced shadows/AO/GI, ReSTIR, denoise and TAA),
        // so this is not an apples-to-apples renderer comparison — it isolates
        // what a 100k-instance field costs the CPU when instances are NOT
        // first-class traced scene entities. Vulkan stays the default.
        else if (std::strcmp(argv[i], "--api") == 0 && i + 1 < argc) {
            const char* a = argv[++i];
            forceGl = std::strcmp(a, "gl") == 0 || std::strcmp(a, "opengl") == 0;
        }
        else if (std::strcmp(argv[i], "--field") == 0 && i + 1 < argc) {
            const char* a = argv[++i];
            useParticleField = !(std::strcmp(a, "im") == 0 ||
                                 std::strcmp(a, "instanced") == 0);
        }
        else if (std::strcmp(argv[i], "--gpu-instances") == 0 && i + 1 < argc) {
            const char* a = argv[++i];
            gpuInstances = !(std::strcmp(a, "off") == 0 || std::strcmp(a, "0") == 0);
        }
        else if (std::strcmp(argv[i], "--shots") == 0 && i + 1 < argc) {
            shotFrames.clear();
            for (const char* s = argv[++i]; *s;) {
                shotFrames.push_back(std::atoi(s));
                while (*s && *s != ',') ++s;
                if (*s == ',') ++s;
            }
        }
    }
    const bool offscreen = !shot.empty() || selftest || bench;
    if (frames <= 0) frames = dropMode ? 300 : 1200;

    // ── The GPU gate ─────────────────────────────────────────────────────────
    // Constructing a GPU world is the only honest probe (driver, device and the
    // PhysX GPU library all have to line up), so try it and report.
    PhysxWorld::Settings ws;
    ws.enableGpuDynamics = true;
    // A PBD system's neighbourhood + contact pools live in the GPU heap, and the
    // neighbourhood alone is maxParticles * maxNeighborhood * 4 bytes (~115 MB
    // at 300k x 96). The 64 MB default is sized for a handful of rigid bodies.
    // PxGpuDynamicsMemoryConfig::isValid() requires a POWER OF TWO here, so this
    // steps rather than scales.
    ws.gpuHeapCapacityMB = budget > 400000u ? 2048u : (budget > 150000u ? 1024u : 512u);
    std::unique_ptr<PhysxWorld> world;
    try {
        world = std::make_unique<PhysxWorld>(ws);
    } catch (const std::exception& e) {
        std::cout << "granular_conveyor: PhysX GPU dynamics is unavailable (" << e.what()
                  << ").\n  PxPBDParticleSystem is CUDA-only and has no CPU path, so this demo "
                     "needs an NVIDIA GPU.\n  Nothing to run - exiting cleanly."
                  << std::endl;
        return 0;
    }

    PbdParticles::Settings ps;
    // 5 cm grains: big enough to read as individual stones at the demo's
    // framing, small enough that 100k of them make a stockpile a 1.9 m belt can
    // stand over instead of one that buries its own discharge.
    ps.spacing = 0.05f;
    ps.solverIterations = 8;
    auto particles = std::make_unique<PbdParticles>(*world, ps);
    const float radius = particles->solidRestOffset();

    // ── The two materials ────────────────────────────────────────────────────
    // PxPBDMaterial has no restitution, so the contrast has to come from
    // friction and damping — which is the honest place for it anyway: the
    // repose angle of a granular heap IS its internal friction. Gravel grips
    // and stacks; pellets slip and run out flat.
    PbdParticles::MaterialSpec gravelSpec;
    // > 1 is legal; PBD friction is soft, so a "grippy" granular material wants
    // more than a rigid body would. It also saturates: 2.0 measured the same
    // pile aspect as 1.2 (0.749 vs 0.756), so 1.2 is where the knob stops
    // paying and where it is left.
    gravelSpec.friction = 1.2f;
    gravelSpec.damping = 0.25f;
    PbdParticles::MaterialSpec pelletSpec;
    // Not zero: a belt conveys by FRICTION, so a frictionless material is not a
    // slick material, it is one the belt cannot pick up at all (measured at
    // 0.02: the pour sloshed over the skirtboards along the whole belt and a
    // cohort covered 3.6 m against a 5.5 m gate). This is the low end that
    // still conveys, and it runs out into a shallow pancake where gravel builds
    // a mound.
    pelletSpec.friction = 0.22f;
    pelletSpec.damping = 0.f;

    const unsigned perLane = std::max(1u, budget / 2u);

    // ── Scene ────────────────────────────────────────────────────────────────
    // Captures are 1080p (they are the deliverable); --selftest / --bench stay
    // at 720p, where they are measuring physics and relative frame cost rather
    // than producing an image.
    const WindowSize size = !shot.empty() ? WindowSize{1920, 1080}
                                          : (offscreen ? WindowSize{1280, 720}
                                                       : WindowSize{1600, 900});
    Canvas canvas(Canvas::Parameters()
                          .title("PBD Granular Conveyor")
                          .size(size)
                          .vsync(!offscreen)
                          .headless(offscreen));
    // Offscreen runs pick a backend for themselves — createRenderer with no
    // argument reads stdin. Vulkan is the target (deferred lighting + shadows on
    // a 100k-instance field is what the demo is for); the fallback is the
    // CPU-only-CI / no-Vulkan-build path.
    std::unique_ptr<Renderer> renderer;
    if (offscreen && forceGl) {
        renderer = createRenderer(canvas, GraphicsAPI::OpenGL);
    } else if (offscreen) {
        try {
            renderer = createRenderer(canvas, GraphicsAPI::Vulkan);
        } catch (const std::exception& e) {
            std::cout << "granular_conveyor: no Vulkan backend (" << e.what()
                      << ") - capturing through OpenGL" << std::endl;
            renderer = createRenderer(canvas, GraphicsAPI::OpenGL);
        }
    } else {
        renderer = createRenderer(canvas);
    }
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->shadowMap().enabled = true;

#ifdef THREEPP_WITH_VULKAN
    auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get());
    // --gpu-instances off: the A/B lever for the GPU-driven instance work
    // (plans/gpu-driven-instances.md). Each stage adds GPU-side production of a
    // per-instance fact, and "did it get cheaper" is only answerable by running
    // the same binary with the pass on and off, interleaved. Default (on) is
    // whatever the renderer's default is.
    if (vk && !gpuInstances) vk->setGpuInstanceExpansion(false);
    if (vk && offscreen) {
        // PIN the exposure. Auto-exposure chases a scene whose average
        // brightness changes as the piles grow, which makes both A/B frame
        // times and before/after captures unreadable.
        vk->setAutoExposure(false);
    }
#endif

    Scene scene;
    constexpr int kSky = 0x86a3c0;
    scene.background = Color(kSky);
    // NO scene fog, deliberately. Fading the far ground into the sky would be
    // the natural way to hide a finite ground plate's edge, but on the Vulkan
    // deferred backend Scene::fog desaturates the WHOLE frame regardless of
    // Fog::nearPlane/farPlane — probed with near=500/far=1000, which cannot
    // touch a 16 m-deep scene, and the yard still lost its colour. So the edge
    // is pushed out of frame geometrically instead (see the floor below).

    auto sun = DirectionalLight::create(0xfff3e0, 3.2f);
    sun->position.set(7.f, 10.f, 5.f);
    sun->castShadow = true;
    scene.add(sun);
    scene.add(AmbientLight::create(0x8899bb, 0.35f));

    auto floorMat = MeshStandardMaterial::create();
    floorMat->color = Color(0x5f5a52);
    floorMat->roughness = 0.95f;
    // Far larger than the framing needs: with no fog to hide it, the plate's
    // edge has to be pushed past where a ~30-degree-elevation camera can see
    // ground at all. A visible edge reads as a diorama.
    auto floor = Mesh::create(BoxGeometry::create(400.f, 0.4f, 400.f), floorMat);
    floor->position.y = -0.2f;
    floor->receiveShadow = true;
    scene.add(floor);
    // Grippy floor: a slick one lets the stockpile skate outward instead of
    // stacking, which is the whole thing the pile angle is supposed to show.
    auto* floorPhys = world->physics().createMaterial(0.9f, 0.9f, 0.f);
    world->addStatic(*floor, floorPhys);

    // Gravel: dark, matte, faceted. detail 0 = a 20-face icosahedron, ~1/2 the
    // triangles of even a coarse UV sphere, which matters 50 000 times over.
    auto gravelMat = MeshStandardMaterial::create();
    gravelMat->color = Color(0x756a5c);
    gravelMat->roughness = 0.95f;
    gravelMat->metalness = 0.f;
    gravelMat->flatShading = true;

    // Pellets: pale, smooth, slightly glossy — a rounded bead.
    auto pelletMat = MeshStandardMaterial::create();
    pelletMat->color = Color(0xd8bd82);
    pelletMat->roughness = 0.35f;
    pelletMat->metalness = 0.05f;

    // ParticleField is Vulkan-only by decision: GL keeps the InstancedMesh
    // path, which is a first-class capability there and the reference ceiling
    // this demo's bench table reports against. Decided from the RENDERER, not
    // the --api flag: the interactive run picks its backend on stdin inside
    // createRenderer, and the offscreen path falls back to GL when Vulkan is
    // unavailable — in both, forceGl is false while the renderer is GL, and a
    // ParticleField there would draw exactly nothing.
#ifdef THREEPP_WITH_VULKAN
    const bool pfFields = useParticleField && vk != nullptr;
#else
    const bool pfFields = false;
#endif
    const auto makeField = [&](const std::shared_ptr<BufferGeometry>& geom,
                               const std::shared_ptr<Material>& mat, unsigned cap,
                               unsigned seed) -> std::unique_ptr<IGrainVisual> {
        if (pfFields) return std::make_unique<PfField>(geom, mat, cap, seed, radius);
        return std::make_unique<GrainField>(geom, mat, cap, seed);
    };
    std::cout << "grain field: " << (pfFields ? "ParticleField (one entry per lane)"
                                              : "InstancedMesh (one entry per grain)")
              << std::endl;

    std::vector<Lane> lanes;
    if (dropMode) {
        Lane l;
        l.name = "gravel";
        l.z = 0.f;
        l.group = &particles->addGroup(budget, gravelSpec);
        l.field = makeField(IcosahedronGeometry::create(radius, 0), gravelMat, budget, 7u);
        lanes.push_back(std::move(l));
    } else {
        struct LaneDef {
            const char* name;
            float z;
            const PbdParticles::MaterialSpec* mat;
            std::shared_ptr<Material> visual;
            std::shared_ptr<BufferGeometry> geom;
            unsigned seed;
        };
        const LaneDef defs[2] = {
                {"gravel", -kLaneZ, &gravelSpec, gravelMat,
                 IcosahedronGeometry::create(radius, 0), 7u},
                {"pellets", +kLaneZ, &pelletSpec, pelletMat,
                 SphereGeometry::create(radius, 6, 4), 23u}};
        for (const auto& d : defs) {
            Lane l;
            l.name = d.name;
            l.z = d.z;
            l.group = &particles->addGroup(perLane, *d.mat);
            l.field = makeField(d.geom, d.visual, perLane, d.seed);
            l.chute = std::make_unique<Chute>(Vector3(kChuteX, kBeltY + 0.42f, d.z), kBeltWidth,
                                              ps.spacing, d.seed * 41u + 1u);
            lanes.push_back(std::move(l));
        }
    }
    for (auto& l : lanes) {
        l.field->object().castShadow = true;
        l.field->object().receiveShadow = true;
        scene.add(l.field->shared());
    }

    std::unique_ptr<cv::ConveyorPhysics> beltSim;
    if (!dropMode) {
        auto chuteMat = MeshStandardMaterial::create();
        chuteMat->color = Color(0x767d84);
        chuteMat->roughness = 0.5f;
        chuteMat->metalness = 0.85f;
        chuteMat->side = Side::Double;

        std::vector<cv::ConveyorSpec> specs;
        for (auto& l : lanes) {
            auto spec = laneSpec(l.z);
            l.beltMat = addConveyorVisual(scene, spec);
            // UV.v is cumulative arc length, so scrolling it runs the pattern
            // along travel whatever the path does in plan — including round
            // the bend, where +X is no longer travel.
            l.scroll = -spec.speed * l.beltMat->map->repeat.y;
            l.geom = makeLaneGeom(cv::resamplePath(spec.waypoints, spec.smooth, spec.samples),
                                  Vector3(kChuteX, kBeltY, l.z));
            addTailSkirt(scene, *world, l.z, chuteMat);
            addChuteVisual(scene, l.z, l.chute->slabTop(), chuteMat);
            specs.push_back(std::move(spec));
        }
        // One ConveyorPhysics for every lane: it takes the whole spec list and
        // registers ONE pre/post substep hook pair for all of them.
        beltSim = std::make_unique<cv::ConveyorPhysics>(*world, std::move(specs));
        std::cout << "belts: " << beltSim->beltCount() << " drag segments, "
                  << beltSim->wallCount() << " guide segments, " << kBeltSpeed << " m/s"
                  << std::endl;
    }

    // Dust rises where the grains land. Density representation only — no
    // geometry, no billboards — so it costs one scatter dispatch however many
    // particles the plumes carry. ParticleField is Vulkan-only, so --dust on a
    // GL run is declined out loud rather than silently ignored.
    std::vector<std::unique_ptr<DustPlume>> plumes;
    if (wantDust && !dropMode) {
        if (!pfFields) {
            std::cout << "granular_conveyor: --dust needs the Vulkan backend - skipping"
                      << std::endl;
        } else {
            for (auto& l : lanes) {
                if (!l.geom.valid()) continue;
                const Vector3 impact(l.geom.discharge.x + l.geom.outDir.x * 0.85f, 0.30f,
                                     l.geom.discharge.z + l.geom.outDir.z * 0.85f);
                plumes.push_back(std::make_unique<DustPlume>(
                        scene, impact, l.geom.outDir, 97u + unsigned(plumes.size()) * 31u));
            }
            std::cout << "dust: " << plumes.size() << " discharge plume(s)" << std::endl;
        }
    }

    // Framing has to hold chute -> belt -> BEND -> stockpile for both lanes in
    // one shot, and the eye also has to clear the skirtboards: a 0.32 m guide on
    // a 1.0 m belt hides the bed entirely below ~18 degrees of elevation, and
    // the conveyed stream is the thing being demonstrated. The two lanes splay
    // apart, so the default sits out on +X looking back INTO the V they make —
    // both turns face the camera from there.
    const Vector3 look = cap.camTarget.value_or(
            dropMode ? Vector3(0.f, 0.35f, 0.f) : Vector3(1.0f, 1.10f, 0.f));
    PerspectiveCamera camera(fov > 0.f ? fov : (dropMode ? 42.f : 40.f), canvas.aspect(), 0.05f,
                             300.f);
    // Framed on the LATE state (both stockpiles at full 50k, ~frame 1150), not
    // the pretty middle: a framing that only fits while the piles are small
    // spends the end of the run cropping them.
    camera.position.copy(cap.camPos.value_or(
            dropMode ? Vector3(3.6f, 2.0f, 4.4f) : Vector3(14.2f, 6.6f, 0.f)));
    camera.lookAt(look);

    std::unique_ptr<OrbitControls> controls;
    if (!offscreen) {
        controls = std::make_unique<OrbitControls>(camera, canvas);
        controls->target.copy(look);
    }
    canvas.onWindowResize([&](WindowSize s) {
        camera.aspect = s.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(s);
    });

    if (dropMode) {
        const auto seeded = block(budget, ps.spacing, Vector3(0.f, 0.9f, 0.f), 4242u);
        lanes[0].group->emit(seeded.data(), unsigned(seeded.size()), Vector3(0.f, -0.5f, 0.f),
                             0.02f);
    }
    std::cout << "granular_conveyor: " << lanes.size() << " lane(s), capacity "
              << lanes[0].group->capacity() << "/lane, radius " << radius << " m, rate " << rate
              << "/s/lane, GPU heap " << ws.gpuHeapCapacityMB << " MB" << std::endl;

    // ── Loop ─────────────────────────────────────────────────────────────────
    // Fixed dt, so a headless run is frame-for-frame reproducible and "frame N"
    // means the same sim time every time.
    constexpr float kDt = 1.f / 60.f;
    constexpr unsigned kCohort = 2000;// grains followed down the belt
    // The cohort window. Both ends are constrained, and the bend tightened
    // both: transit is only (pathLength - chuteStation) / speed ~ 3 s, so a
    // window longer than that stops measuring how FAST the belt carries and
    // starts measuring whether the cohort has reached the end — every grain
    // that runs off clamps to the final station and the mean saturates. Two
    // seconds is under the transit time with room to spare. The start waits
    // until the whole cohort exists (2000 grains at 2800/s = 0.71 s), so the
    // two samples average the same grains rather than a growing set.
    constexpr int kCohortStart = 60; // frame the cohort measurement starts
    constexpr int kCohortSpan = 120; // 2 sim seconds later
    int frame = 0;
    Stats at60, at240;// drop mode only (one lane)
    bool ok = true;
    const auto fail = [&](const std::string& what) {
        std::cout << "SELFTEST FAIL: " << what << std::endl;
        ok = false;
    };

    // --bench: fps sampled in windows, tagged with the live particle count.
    //
    // Every column is measured, and together they are meant to CLOSE: wall should
    // equal sim + emit + field + stats + rend + tail, and rend (a host chrono
    // around render()) should equal frame (the renderer's own cpuFrameMs) to
    // within timer noise. Where those two disagree, one of them is lying about
    // its own extent; where the sum falls short of wall, the missing time is
    // outside every timer in this file.
    struct BenchRow {
        unsigned n;
        double fps;
        double wallMs;   // frame period: window wall / frames
        double simMs;    // world.step() + the position readback
        double emitMs;   // chute tick + Group::emit (CUDA lock + blocking HtoD)
        double fieldMs;  // the per-particle instanceMatrix write, alone
        double statsMs;  // telemetry passes over every grain (2 Hz gated)
        double rendMs;   // host wall around renderer->render()
        double frameMs;  // renderer CPU frame (cpuFrameMs), Vulkan only
        double ensMs;    // ensureSceneBuilt, the scene.* half of frameMs
        double tailMs;   // outside the animate body: submit+present / swap, poll
        double gpuMs;    // whole-command-buffer GPU span (Vulkan only)
        double gpuSumMs; // the bracketed GPU passes only; gpu - gpusum is unbracketed
        double gbufMs;   // raster G-buffer (Vulkan only)
        double shadeMs;  // deferred shade / trace (Vulkan only)
        double iexpMs;   // GPU per-instance world-matrix expansion (Vulkan only)
    };
    std::vector<BenchRow> benchRows;
    constexpr int kBenchWindow = 120;
    std::chrono::high_resolution_clock::time_point windowStart;
    double simAccum = 0, frameAccum = 0, gbufAccum = 0, shadeAccum = 0, gpuAccum = 0;
    double iexpAccum = 0;
    double emitAccum = 0, fieldAccum = 0, statsAccum = 0, rendAccum = 0, tailAccum = 0;
    double ensAccum = 0, gpuSumAccum = 0;
    int windowFrames = 0;
    // Closed at the TOP of the animate body, not the bottom. Closing at the
    // bottom made the first frame of every window contribute its wall time but
    // not its phase times (the accumulators were zeroed after that frame's sim had
    // already been added), and divided kBenchWindow frames by kBenchWindow-1 frame
    // periods — ~0.8% low on every phase, ~0.8% high on fps, and a spurious
    // residual in exactly the wall-minus-phases number this table exists to
    // report. Tagged with hudGrains (the previous frame's population) because
    // `total` is body-local.

    // ── Live HUD (interactive only) ──────────────────────────────────────────
    // RendererSettingsUi draws the FPS row itself; these add the sim-vs-render
    // split the --bench table reports, because that split is the whole question
    // this demo raises: at 300k grains, is the wall PhysX or the renderer?
    // Smoothed over 0.5 s — a per-frame readout of either number is unreadable.
    //
    // NOTE ON THAT FPS ROW AND VSYNC: interactive runs present with vsync on
    // (see the Canvas parameters), so FPS pins to the refresh rate as soon as
    // the frame fits in its budget. The millisecond rows keep telling the truth
    // after that — read those, not FPS, when judging headroom where the sim is
    // heavy. --bench runs unvsynced.
    double hudSimMs = 0.0, hudFrameMs = 0.0, hudGbufMs = 0.0;
    unsigned hudGrains = 0;
    double hudWall = 0.0, hudSimAccum = 0.0, hudFrameAccum = 0.0, hudGbufAccum = 0.0;
    int hudFrames = 0;
    auto hudLast = std::chrono::high_resolution_clock::now();

    // Headless paths build no UI at all: --shot frames are the deliverable and
    // must not have a panel painted into them, and --bench/--selftest measure a
    // frame that the panel would otherwise be part of.
    std::unique_ptr<RendererSettingsUi> ui;
    if (!offscreen) {
        ui = std::make_unique<RendererSettingsUi>(canvas, *renderer, [&] {
            ImGui::Text("grains    %8u", hudGrains);
            ImGui::Text("sim       %8.2f ms", hudSimMs);
            if (hudFrameMs > 0.0) ImGui::Text("renderer  %8.2f ms", hudFrameMs);
            if (hudGbufMs > 0.0) ImGui::Text("gbuf      %8.2f ms", hudGbufMs);
        },
        "Granular");
    }

    using Clk = std::chrono::high_resolution_clock;
    const auto msSince = [](const Clk::time_point& t) {
        return std::chrono::duration<double, std::milli>(Clk::now() - t).count();
    };
    // Stamped at the very end of the animate body; the tail is measured from it to
    // the point animateOnce() hands control back. See the explicit loop below.
    Clk::time_point bodyEnd{};
    bool haveBodyEnd = false;

    const auto frameBody = [&] {
        // Window bookkeeping first, so a window spans exactly kBenchWindow frame
        // tops and every one of those frames' phases lands inside it.
        if (bench) {
            const auto nowW = Clk::now();
            if (windowFrames >= kBenchWindow) {
                const double wall = std::chrono::duration<double>(nowW - windowStart).count();
                const double k = 1.0 / double(windowFrames);
                benchRows.push_back({hudGrains, double(windowFrames) / wall,
                                     wall * 1000.0 * k, simAccum * k, emitAccum * k,
                                     fieldAccum * k, statsAccum * k, rendAccum * k,
                                     frameAccum * k, ensAccum * k, tailAccum * k,
                                     gpuAccum * k, gpuSumAccum * k,
                                     gbufAccum * k, shadeAccum * k, iexpAccum * k});
                windowFrames = 0;
            }
            if (windowFrames == 0) {
                windowStart = nowW;
                simAccum = frameAccum = gbufAccum = shadeAccum = gpuAccum = iexpAccum = 0;
                emitAccum = fieldAccum = statsAccum = rendAccum = tailAccum = 0;
                ensAccum = gpuSumAccum = 0;
            }
            ++windowFrames;
        }

        // Outside the sim timer below, and not free: Group::emit takes a scoped
        // CUDA lock and issues three SYNCHRONOUS host-to-device copies per lane.
        const auto tEmit = Clk::now();
        for (auto& l : lanes) {
            if (!l.chute) continue;
            const auto& burst = l.chute->tick(kDt, rate);
            if (!burst.empty())
                l.group->emit(burst.data(), unsigned(burst.size()), Vector3(0.4f, -1.4f, 0.f),
                              0.02f);
        }
        emitAccum += msSince(tEmit);

        const auto tSim = std::chrono::high_resolution_clock::now();
        world->step(kDt);
        particles->pull();
        const double simMs = std::chrono::duration<double, std::milli>(
                                     std::chrono::high_resolution_clock::now() - tSim)
                                     .count();
        simAccum += simMs;
        if (ui) hudSimAccum += simMs;

        // The per-particle instanceMatrix write, ALONE and in its own loop. This is
        // the setMatrixAt-equivalent: 12 bytes at a 64-byte stride for every live
        // grain, per lane, per frame, plus instanceMatrix()->needsUpdate(). It is
        // the prime suspect for the work outside render(), so it must not share a
        // timer with the belt scroll below — that one dirties a material, which is
        // what makes the renderer's MaterialDesc flush fire, an unrelated cost on
        // the other side of render(). (That flush used to re-send every entry's
        // 608-byte MaterialDesc for these two belts — 2.8 ms and 47.7 MB a frame
        // at 78.4k grains. It now sends their entry ranges only, which is why
        // frame.I_uploadMatDesc reads ~0.003 ms; keeping the two timers apart is
        // what made the cost attributable in the first place.)
        const auto tField = Clk::now();
        for (auto& l : lanes)
            l.field->update(l.group->positions(), l.group->active());
        fieldAccum += msSince(tField);
        for (auto& p : plumes) p->tick(kDt);
        for (auto& l : lanes) {
            if (!l.beltMat) continue;
            l.beltMat->map->offset.y += l.scroll * kDt;
            l.beltMat->needsUpdate();
        }

        // Host-side wall around render(), as a CONTROL on cpuFrameMs: the renderer
        // stamps that itself, from its own frameStart to just before it returns. If
        // the two agree, every unaccounted millisecond is provably outside render();
        // if they do not, cpuFrameMs is wrong about its own extent. It is also the
        // only render number a --api gl run has.
        const auto tRend = Clk::now();
        renderer->render(scene, camera);
        rendAccum += msSince(tRend);
        if (ui) ui->render();
        ++frame;

#ifdef THREEPP_WITH_VULKAN
        if (vk) {
            const auto t = vk->lastFrameTimings();
            frameAccum += double(t.cpuFrameMs);
            gbufAccum += double(t.rasterGbufMs);
            shadeAccum += double(t.pathTraceMs);
            // Whole-command-buffer GPU span. NOTE it describes the RETIRED
            // occupant of this frame-in-flight slot — two frames back, since
            // readBack runs right after the fence wait. Only window means are
            // comparable with the CPU columns, never a per-frame difference.
            gpuAccum += double(t.gpuTotalMs);
            gpuSumAccum += double(t.gpuPassSumMs);
            // The GPU cost of the instance-expansion dispatch, on its own. It is
            // pure ADDED cost until a consumer moves onto it, so it has to be
            // visible next to what it will eventually replace.
            iexpAccum += double(t.instanceExpandMs);
            // The scene.* half of cpuFrameMs, so a hole inside render() can be
            // localised to ensureSceneBuilt (structural rebuilds do work outside
            // every scene.* scope) rather than to the frame path.
            ensAccum += double(t.cpuEnsureSceneMs);
            if (ui) {
                hudFrameAccum += double(t.cpuFrameMs);
                hudGbufAccum += double(t.rasterGbufMs);
            }
        }
#endif

        // HUD window: the means of whatever was accumulated over the span.
        // Closes on elapsed TIME, not a frame count, so the readout keeps
        // refreshing at a usable rate when a heavy scene drops to single-digit
        // fps (a 120-frame window would freeze it for 20 s there).
        if (ui) {
            const auto nowT = std::chrono::high_resolution_clock::now();
            hudWall += std::chrono::duration<double>(nowT - hudLast).count();
            hudLast = nowT;
            ++hudFrames;
            if (hudWall >= 0.5) {
                const double k = 1.0 / double(hudFrames);
                hudSimMs = hudSimAccum * k;
                hudFrameMs = hudFrameAccum * k;
                hudGbufMs = hudGbufAccum * k;
                hudWall = hudSimAccum = hudFrameAccum = hudGbufAccum = 0.0;
                hudFrames = 0;
            }
        }

        // Telemetry is not free: measure() is two passes over every grain, so at
        // 300k it is milliseconds of CPU per frame and would show up in the
        // --bench table as a rendering cost. Sample it at 2 Hz plus the exact
        // frames a gate reads, which still catches a NaN within 30 frames.
        // Timed OUTSIDE the gate on purpose: the column then reports the true
        // per-frame mean, zero frames included, which is the number that belongs
        // in a wall-minus-phases residual. (Per-occurrence it is ~30x this.)
        const auto tStats = Clk::now();
        const bool sample = frame % 30 == 0 || frame == kCohortStart ||
                            frame == kCohortStart + kCohortSpan || frame >= frames;
        unsigned total = 0;
        for (auto& l : lanes) {
            if (sample) l.stats = measure(l.group->positions(), l.group->active(), l.geom);
            total += l.group->active();
            if (l.stats.bad) fail(std::string(l.name) + ": non-finite particle positions");
            if (frame == kCohortStart)
                l.cohortS0 = cohortMeanStation(l.group->positions(),
                                               std::min(kCohort, l.stats.n), l.geom);
            if (frame == kCohortStart + kCohortSpan)
                l.cohortS1 = cohortMeanStation(l.group->positions(),
                                               std::min(kCohort, l.stats.n), l.geom);
        }
        if (dropMode) {
            if (frame == 60)
                at60 = measure(lanes[0].group->positions(), lanes[0].group->active(),
                               lanes[0].geom);
            if (frame == 240)
                at240 = measure(lanes[0].group->positions(), lanes[0].group->active(),
                                lanes[0].geom);
        }
        // Every frame, not on the 2 Hz `sample` gate above: active() is a
        // counter read, not a pass over the grains. (The panel drew earlier in
        // this frame, so it shows this value one frame later — invisible.)
        hudGrains = total;
        statsAccum += msSince(tStats);

        if (frame % 150 == 0) {
            std::printf("[f%4d] n=%6u", frame, total);
            for (const auto& l : lanes)
                std::printf("  %s{bed=%.3f pile=%.2f/%.2f spill=%5u lost=%4u@%.1f}", l.name,
                            double(l.stats.bedTop), double(l.stats.pileTop),
                            double(l.stats.pileRadius), l.stats.spilled, l.stats.lost,
                            double(l.stats.lostStation));
            std::printf("\n");
            std::fflush(stdout);
        }

        if (!shot.empty() &&
            std::find(shotFrames.begin(), shotFrames.end(), frame) != shotFrames.end()) {
            char name[256];
            std::snprintf(name, sizeof(name), "%s_f%04d.png", shot.c_str(), frame);
            const auto p = capture::shotOutputPath(name);
            renderer->writeFramebuffer(p);
            std::cout << "wrote " << p.string() << " (n=" << total << ")" << std::endl;
        }
        if (offscreen && frame >= frames) canvas.close();
        bodyEnd = Clk::now();
        haveBodyEnd = true;
    };

    // canvas.animate(f) IS `while (animateOnce(f)) {}` — spelled out here because
    // the interval between the body returning and animateOnce handing control back
    // is invisible to any timer inside the body, and for Vulkan that interval holds
    // vkQueueSubmit2 + vkQueuePresentKHR (endFrame runs from the Canvas frame-end
    // callback), which are the only blocking Vulkan calls in the frame that are
    // NOT inside render(). For GL it holds glfwSwapBuffers. Plus glfwPollEvents
    // for both. Interactive runs present with vsync ON, where a large tail is just
    // the refresh wait and means nothing; read this only from --bench.
    //
    // AND EVEN FROM --bench, `tail` and `sim` are two halves of ONE number here.
    // A --bench run is headless, which on Windows/NVIDIA means presenting to a
    // window nothing composites, and vkQueuePresentKHR then waits on the HOST for
    // the presented frame's rendering to finish (see the derivation in
    // VulkanContext::VulkanContext). So `tail` ~= gpuTotalMs, and it is a WAIT,
    // not work. Suppress the present
    // (THREEPP_VULKAN_SUPPRESS_PRESENT=1) and `tail` collapses to 0.07 ms while
    // `sim` triples: world->step() + particles->pull() is a blocking CUDA sync,
    // so the moment the CPU is allowed to run a frame ahead, the wait for the
    // GPU reappears there instead. Measured at 78.4k: tail 12.5 / sim 4.1
    // becomes tail 0.07 / sim 19.3 for the same wall. Whichever column it lands
    // in, the wall is the sum of this app's two GPU tenants plus its CPU frame —
    // read `wall`, and read `gpu` for the graphics half.
    while (canvas.animateOnce(frameBody)) {
        if (!haveBodyEnd) continue;// the final call returns false without running f
        tailAccum += msSince(bodyEnd);
        haveBodyEnd = false;
    }

    // Both borrow the world and must die before it (they unregister substep
    // hooks and release actors / device buffers through its CUDA context).
    beltSim.reset();

    if (bench) {
        // Header and row are independent format strings — edit them together.
        // `resid` is wall minus everything measured: the honest hole.
        std::printf("\n%9s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s"
                    "   (%dx%d, %d-frame windows)\n",
                    "particles", "fps", "wall", "sim", "emit", "field",
                    "stats", "rend", "frame", "ens", "tail", "resid",
                    "gpu", "gpusum", "gbuf", "shade", "iexp",
                    size.width(), size.height(), kBenchWindow);
        for (const auto& r : benchRows) {
            const double resid = r.wallMs - (r.simMs + r.emitMs + r.fieldMs + r.statsMs +
                                             r.rendMs + r.tailMs);
            std::printf("%9u %7.1f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f"
                        " %7.2f %7.2f %7.2f %7.3f %7.3f %7.3f\n",
                        r.n, r.fps, r.wallMs, r.simMs, r.emitMs, r.fieldMs, r.statsMs,
                        r.rendMs, r.frameMs, r.ensMs, r.tailMs, resid,
                        r.gpuMs, r.gpuSumMs, r.gbufMs, r.shadeMs, r.iexpMs);
        }
        std::fflush(stdout);
    }

    if (!selftest) {
        particles.reset();
        return 0;
    }

    // ── Numeric gates ────────────────────────────────────────────────────────
    for (const auto& l : lanes) {
        if (l.stats.bad) fail(std::string(l.name) + ": non-finite particle positions");
        // The floor's top face is y = 0 and a resting grain's centre sits one
        // radius above it, so anything materially below 0 has tunnelled.
        if (l.stats.minY < -0.5f * radius)
            fail(std::string(l.name) + ": particles tunnelled through the floor");
    }

    if (dropMode) {
        const auto& s = lanes[0].stats;
        if (!(s.maxY < at60.maxY)) fail("the block never collapsed (maxY did not fall)");
        std::printf("selftest[drop]: n=%u minY=%.4f maxY=%.4f (f60 %.4f, f240 %.4f)\n", s.n,
                    double(s.minY), double(s.maxY), double(at60.maxY), double(at240.maxY));
    } else {
        for (const auto& l : lanes) {
            // THE belt gate: a cohort of grains must be carried ALONG THE PATH
            // at more than half the belt speed — which is the gate the bend
            // makes interesting, since clearing it now means riding the turn
            // rather than sliding straight on. Half, not all, because a grain
            // spends its first metre being accelerated from the chute's velocity
            // and its last one falling off the end.
            const float carried = l.cohortS1 - l.cohortS0;
            const float floorDist = 0.5f * kBeltSpeed * (float(kCohortSpan) * kDt);
            if (frames < kCohortStart + kCohortSpan) fail("run too short to measure transport");
            if (!(carried > floorDist)) fail(std::string(l.name) + ": the belt did not carry it");
            if (l.stats.spilled == 0)
                fail(std::string(l.name) + ": nothing reached the discharge end");
            // Overrun check: a jammed belt can still pass the transport gate on
            // the creep of its bottom layer, so pin the bed depth too.
            if (!(l.stats.bedTop < 2.f * kGuideHeight))
                fail(std::string(l.name) + ": the belt is overloaded (bed over the guides)");
            // THE bend gate: grain goes round the corner, not over the side of
            // it. Cumulative — everything that ever left the belt along the run,
            // against everything the lane ever emitted. Measured at 27 and 12
            // grains in 32 666 (under 0.1%), all of it at station ~1.7, which is
            // chute splash at the inlet and not the turn at all; the bend's own
            // contribution is indistinguishable from zero. The threshold is 1%,
            // an order of magnitude clear, because this is a tripwire for a
            // BROKEN bend — guides that stop following the arc, or colliders
            // that stop matching the ribbon — and not a tuning knob.
            if (!(l.stats.lost < l.stats.n / 100u))
                fail(std::string(l.name) + ": grain is being thrown off the bend");
            std::printf("selftest[%s]: n=%u carried=%.2f m/%.1f s (floor %.2f) spilled=%u "
                        "lost=%u@%.2f/%.2f m bed=%.3f minY=%.4f pile=%.2f/%.2f aspect=%.3f\n",
                        l.name, l.stats.n, double(carried), double(float(kCohortSpan) * kDt),
                        double(floorDist), l.stats.spilled, l.stats.lost,
                        double(l.stats.lostStation), double(l.geom.length()),
                        double(l.stats.bedTop),
                        double(l.stats.minY), double(l.stats.pileTop), double(l.stats.pileRadius),
                        l.stats.pileRadius > 0.f ? double(l.stats.pileTop / l.stats.pileRadius)
                                                 : 0.0);
        }
        // THE materials gate: the grippy material must build a measurably
        // STEEPER pile than the slick one. Colour is not evidence.
        if (lanes.size() == 2) {
            const auto aspect = [](const Stats& s) {
                return s.pileRadius > 1e-3f ? s.pileTop / s.pileRadius : 0.f;
            };
            const float a0 = aspect(lanes[0].stats), a1 = aspect(lanes[1].stats);
            if (!(a0 > a1 * 1.15f)) {
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                              "the two materials pile the same (gravel %.3f vs pellets %.3f)", a0,
                              a1);
                fail(buf);
            }
        }
    }

    std::cout << (ok ? "SELFTEST PASS" : "SELFTEST FAIL") << std::endl;
    particles.reset();
    return ok ? 0 : 1;
}
