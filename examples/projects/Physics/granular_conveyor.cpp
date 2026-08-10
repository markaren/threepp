// GPU granular material on a conveyor: PhysX 5.5 PxPBDParticleSystem grains
// poured from a chute onto the belt from extras/conveyor, carried along it, and
// spilling off the discharge end into a stockpile. Two lanes run side by side
// with contrasting PBD materials — grippy GRAVEL against slick PELLETS — so the
// difference is in how each one piles, not just what colour it is.
//
// The belt drives the grains by the same mechanism it drives soft bodies — a
// kinematic collider whose target is advanced along travel each substep and
// teleported back afterwards (see ConveyorPhysics). No special case was needed
// for particles: PBD reads the contact velocity of a kinematic rigid exactly
// like a rigid body does, so the belt "just works" once the granular material's
// friction is high enough to be dragged.
//
// PBD particles are a CUDA-only PhysX feature with no CPU fallback. This file
// COMPILES anywhere PhysX does — the headers ship regardless — and at runtime
// prints why and exits 0 when there is no GPU, so CI can build and run it.
//
//   granular_conveyor                       interactive (pick a backend)
//   granular_conveyor --shot pbd_belt       headless captures -> aaa_caps/
//   granular_conveyor --selftest            headless numeric gates, no window
//   granular_conveyor --bench               particles-vs-fps table
//   granular_conveyor --count 200000        particle budget (split over lanes)
//   granular_conveyor --rate 4000           grains per second per lane
//   granular_conveyor --drop                no belt: phase-1 block-drop test
//   ... --cam 6,3,7 --look 4,0.5,-3 --fov 40    reframe without rebuilding
//
// The default framing is the overview: both chutes, both belts, both
// stockpiles. Two framings worth keeping, found by looking at captures:
//   --cam -2,4.4,6.2  --look 2.6,1.95,2.8 --fov 44   the conveyed BED, three
//       quarters down the near belt with the far lane's pile behind it. The eye
//       has to sit above ~18 degrees of elevation or the skirtboards hide the
//       bed entirely, which is the one thing this view exists to show.
//   --cam 7.4,2.3,1.6 --look 4.9,1,-2.9   --fov 44   the discharge curtain,
//       close enough to read individual stones.
// Both want a frame while the chute is still pouring: a lane empties in
// (capacity/2)/rate seconds, so --frames past that shows a stopped machine.

#include "threepp/threepp.hpp"

#include "threepp/extras/conveyor/ConveyorPhysics.hpp"
#include "threepp/extras/physx/PhysxParticles.hpp"

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
    constexpr float kBeltX1 = 3.5f;    // discharge end
    constexpr float kBeltWidth = 1.00f;
    constexpr float kBeltSpeed = 2.2f; // m/s along +X
    constexpr float kGuideHeight = 0.32f;
    // A metre of belt behind the pour, so grain that bounces backwards off the
    // bed is picked up again instead of dribbling off the tail drum.
    constexpr float kChuteX = -2.50f;
    // Lane separation. Wide enough that two 2 m-radius stockpiles only just
    // meet: the whole point of the side-by-side is comparing their PROFILES,
    // and a merged heap has one profile.
    constexpr float kLaneZ = 2.80f;
    // Where a stockpile begins: past the discharge drum and past where the
    // 2.2 m/s stream lands, so pile probes never measure the trajectory.
    constexpr float kPileX = kBeltX1 + 1.4f;

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
    class GrainField {

    public:
        GrainField(const std::shared_ptr<BufferGeometry>& geometry,
                   const std::shared_ptr<Material>& material, unsigned capacity,
                   unsigned seed)
            : rot_(std::size_t(capacity) * 9), capacity_(capacity) {

            mesh_ = InstancedMesh::create(geometry, material, capacity);
            mesh_->setCount(0);
            mesh_->frustumCulled = false;// the field spans the whole scene

            std::mt19937 rng{seed};
            std::uniform_real_distribution<float> uni(0.f, 1.f);
            auto& e = mesh_->instanceMatrix()->array();
            std::memset(e.data(), 0, e.size() * sizeof(float));
            Quaternion q;
            Matrix4 m;
            for (unsigned i = 0; i < capacity_; ++i) {
                // Uniform random orientation (Shoemake), baked in permanently.
                const float u1 = uni(rng), u2 = uni(rng), u3 = uni(rng);
                const float s1 = std::sqrt(1.f - u1), s2 = std::sqrt(u1);
                q.set(s1 * std::sin(math::TWO_PI * u2), s1 * std::cos(math::TWO_PI * u2),
                      s2 * std::sin(math::TWO_PI * u3), s2 * std::cos(math::TWO_PI * u3));
                m.makeRotationFromQuaternion(q);
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 3; ++r)
                        rot_[std::size_t(i) * 9 + c * 3 + r] = m.elements[c * 4 + r];
                e[std::size_t(i) * 16 + 15] = 1.f;
            }
        }

        // Per-frame: point the field at `n` live grain positions (PxVec4, w =
        // inverse mass, ignored here).
        void update(const ::physx::PxVec4* positions, unsigned n) {

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

        [[nodiscard]] InstancedMesh& mesh() { return *mesh_; }
        [[nodiscard]] std::shared_ptr<InstancedMesh> shared() const { return mesh_; }

    private:
        static constexpr unsigned kStep = 4096;

        std::shared_ptr<InstancedMesh> mesh_;
        std::vector<float> rot_;// 3x3 per instance, written once, read on claim
        unsigned capacity_ = 0;
        unsigned claimed_ = 0;
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

    // A straight belt along +X at the given z, with a side guide on each edge so
    // a granular BED (unlike a crate) does not simply run off the sides. The
    // guides are authored the way the editor authors them: two points in plan,
    // resolved by followWall against the same centreline the ribbon uses.
    cv::ConveyorSpec laneSpec(float z) {

        cv::ConveyorSpec s;
        s.waypoints = {cv::Waypoint{Vector3(kBeltX0, kBeltY, z)},
                       cv::Waypoint{Vector3(kBeltX1, kBeltY, z)}};
        s.width = kBeltWidth;
        s.speed = kBeltSpeed;
        s.smooth = false;
        s.frame = true;

        const float half = kBeltWidth * 0.5f;
        for (int side = -1; side <= 1; side += 2) {
            cv::WallSpec w;
            w.height = kGuideHeight;
            w.points = {Vector3(kBeltX0, kBeltY, z + float(side) * half),
                        Vector3(kBeltX1, kBeltY, z + float(side) * half)};
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

    // ── Telemetry: the numbers the self-test gates on ─────────────────────────
    struct Stats {
        unsigned n = 0;
        unsigned bad = 0;// non-finite components — must stay 0
        float minY = 0.f, maxY = 0.f;
        unsigned spilled = 0;// past the discharge AND below the belt
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

    Stats measure(const ::physx::PxVec4* p, unsigned n) {
        Stats s;
        s.n = n;
        if (n == 0) return s;
        s.minY = s.maxY = p[0].y;
        double px = 0, pz = 0;
        for (unsigned i = 0; i < n; ++i) {
            const auto& v = p[i];
            if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
                ++s.bad;
                continue;
            }
            s.minY = std::min(s.minY, v.y);
            s.maxY = std::max(s.maxY, v.y);
            if (v.x > kBeltX1 && v.y < kBeltY - 0.3f) ++s.spilled;
            if (v.x > kPileX) {
                ++s.pileN;
                px += v.x;
                pz += v.z;
                s.pileTop = std::max(s.pileTop, v.y);
            }
            // Skip the metre downstream of the chute: grain in free fall, and
            // the heap it makes on landing, are not the conveyed bed.
            if (v.x > kChuteX + 1.0f && v.x < kBeltX1 - 0.3f && v.y > kBeltY)
                s.bedTop = std::max(s.bedTop, v.y - kBeltY);
        }
        if (s.pileN > 32) {
            const double cx = px / s.pileN, cz = pz / s.pileN;
            double sq = 0;
            for (unsigned i = 0; i < n; ++i) {
                const auto& v = p[i];
                if (!std::isfinite(v.x) || !std::isfinite(v.z) || v.x <= kPileX) continue;
                const double dx = v.x - cx, dz = v.z - cz;
                sq += dx * dx + dz * dz;
            }
            s.pileRadius = float(std::sqrt(sq / s.pileN));
        }
        return s;
    }

    // Mean X of a fixed cohort — the first `n` particles ever emitted. Particle
    // index in a PxParticleBuffer is stable for the buffer's lifetime, so this
    // follows the SAME grains down the belt; a mean over everything would be
    // dragged backwards by every new grain the chute drops at the inlet.
    float cohortMeanX(const ::physx::PxVec4* p, unsigned n) {
        if (n == 0) return 0.f;
        double sx = 0;
        unsigned good = 0;
        for (unsigned i = 0; i < n; ++i) {
            if (!std::isfinite(p[i].x)) continue;
            sx += p[i].x;
            ++good;
        }
        return good ? float(sx / good) : 0.f;
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
        std::unique_ptr<GrainField> field;
        std::unique_ptr<Chute> chute;
        std::shared_ptr<MeshStandardMaterial> beltMat;
        float scroll = 0.f;
        Stats stats;
        float cohortX0 = 0.f, cohortX1 = 0.f;
    };

}// namespace

int main(int argc, char** argv) {

    std::string shot;
    std::vector<int> shotFrames{150, 450, 1200};
    bool selftest = false;
    bool bench = false;
    bool dropMode = false;
    unsigned budget = 100000;
    float rate = 2800.f;
    int frames = 0;// 0 = per-mode default
    float fov = 0.f;// 0 = per-mode default
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
        else if (std::strcmp(argv[i], "--count") == 0 && i + 1 < argc)
            budget = unsigned(std::max(2, std::atoi(argv[++i])));
        else if (std::strcmp(argv[i], "--rate") == 0 && i + 1 < argc)
            rate = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) frames = std::atoi(argv[++i]);
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
    if (offscreen) {
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

    std::vector<Lane> lanes;
    if (dropMode) {
        Lane l;
        l.name = "gravel";
        l.z = 0.f;
        l.group = &particles->addGroup(budget, gravelSpec);
        l.field = std::make_unique<GrainField>(IcosahedronGeometry::create(radius, 0), gravelMat,
                                               budget, 7u);
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
            l.field = std::make_unique<GrainField>(d.geom, d.visual, perLane, d.seed);
            l.chute = std::make_unique<Chute>(Vector3(kChuteX, kBeltY + 0.42f, d.z), kBeltWidth,
                                              ps.spacing, d.seed * 41u + 1u);
            lanes.push_back(std::move(l));
        }
    }
    for (auto& l : lanes) {
        l.field->mesh().castShadow = true;
        l.field->mesh().receiveShadow = true;
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
            l.scroll = -spec.speed * l.beltMat->map->repeat.y;// travel is +X
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

    // Framing has to hold chute -> belt -> stockpile for BOTH lanes in one
    // shot, and the eye also has to clear the skirtboards: a 0.32 m guide on a
    // 1.0 m belt hides the bed entirely below ~18 degrees of elevation, and the
    // conveyed stream is the thing being demonstrated.
    const Vector3 look = cap.camTarget.value_or(
            dropMode ? Vector3(0.f, 0.35f, 0.f) : Vector3(1.5f, 1.05f, 0.3f));
    PerspectiveCamera camera(fov > 0.f ? fov : (dropMode ? 42.f : 33.f), canvas.aspect(), 0.05f,
                             300.f);
    camera.position.copy(cap.camPos.value_or(
            dropMode ? Vector3(3.6f, 2.0f, 4.4f) : Vector3(13.5f, 6.2f, 8.0f)));
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
    constexpr int kCohortStart = 30;  // frame the cohort measurement starts
    constexpr int kCohortSpan = 300;  // 5 sim seconds later
    int frame = 0;
    Stats at60, at240;// drop mode only (one lane)
    bool ok = true;
    const auto fail = [&](const std::string& what) {
        std::cout << "SELFTEST FAIL: " << what << std::endl;
        ok = false;
    };

    // --bench: fps sampled in windows, tagged with the live particle count.
    struct BenchRow {
        unsigned n;
        double fps;
        double simMs;    // world.step() + the position readback
        double frameMs;  // renderer CPU frame
        double gbufMs;   // raster G-buffer (Vulkan only)
        double shadeMs;  // deferred shade / trace (Vulkan only)
    };
    std::vector<BenchRow> benchRows;
    constexpr int kBenchWindow = 120;
    std::chrono::high_resolution_clock::time_point windowStart;
    double simAccum = 0, frameAccum = 0, gbufAccum = 0, shadeAccum = 0;
    int windowFrames = 0;

    canvas.animate([&] {
        for (auto& l : lanes) {
            if (!l.chute) continue;
            const auto& burst = l.chute->tick(kDt, rate);
            if (!burst.empty())
                l.group->emit(burst.data(), unsigned(burst.size()), Vector3(0.4f, -1.4f, 0.f),
                              0.02f);
        }

        const auto tSim = std::chrono::high_resolution_clock::now();
        world->step(kDt);
        particles->pull();
        simAccum += std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - tSim)
                            .count();

        for (auto& l : lanes) {
            l.field->update(l.group->positions(), l.group->active());
            if (!l.beltMat) continue;
            l.beltMat->map->offset.y += l.scroll * kDt;
            l.beltMat->needsUpdate();
        }

        renderer->render(scene, camera);
        ++frame;

#ifdef THREEPP_WITH_VULKAN
        if (vk) {
            const auto t = vk->lastFrameTimings();
            frameAccum += double(t.cpuFrameMs);
            gbufAccum += double(t.rasterGbufMs);
            shadeAccum += double(t.pathTraceMs);
        }
#endif

        // Telemetry is not free: measure() is two passes over every grain, so at
        // 300k it is milliseconds of CPU per frame and would show up in the
        // --bench table as a rendering cost. Sample it at 2 Hz plus the exact
        // frames a gate reads, which still catches a NaN within 30 frames.
        const bool sample = frame % 30 == 0 || frame == kCohortStart ||
                            frame == kCohortStart + kCohortSpan || frame >= frames;
        unsigned total = 0;
        for (auto& l : lanes) {
            if (sample) l.stats = measure(l.group->positions(), l.group->active());
            total += l.group->active();
            if (l.stats.bad) fail(std::string(l.name) + ": non-finite particle positions");
            if (frame == kCohortStart)
                l.cohortX0 = cohortMeanX(l.group->positions(), std::min(kCohort, l.stats.n));
            if (frame == kCohortStart + kCohortSpan)
                l.cohortX1 = cohortMeanX(l.group->positions(), std::min(kCohort, l.stats.n));
        }
        if (dropMode) {
            if (frame == 60) at60 = measure(lanes[0].group->positions(), lanes[0].group->active());
            if (frame == 240) at240 = measure(lanes[0].group->positions(), lanes[0].group->active());
        }

        // Benchmark windows: close one whenever it fills, tagged with the
        // population it was measured at.
        if (bench) {
            if (windowFrames == 0) {
                windowStart = std::chrono::high_resolution_clock::now();
                simAccum = frameAccum = gbufAccum = shadeAccum = 0;
            }
            if (++windowFrames >= kBenchWindow) {
                const double wall = std::chrono::duration<double>(
                                            std::chrono::high_resolution_clock::now() - windowStart)
                                            .count();
                const double k = 1.0 / double(windowFrames);
                benchRows.push_back({total, double(windowFrames) / wall, simAccum * k,
                                     frameAccum * k, gbufAccum * k, shadeAccum * k});
                windowFrames = 0;
            }
        }

        if (frame % 150 == 0) {
            std::printf("[f%4d] n=%6u", frame, total);
            for (const auto& l : lanes)
                std::printf("  %s{bed=%.3f pile=%.2f/%.2f spill=%5u}", l.name,
                            double(l.stats.bedTop), double(l.stats.pileTop),
                            double(l.stats.pileRadius), l.stats.spilled);
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
    });

    // Both borrow the world and must die before it (they unregister substep
    // hooks and release actors / device buffers through its CUDA context).
    beltSim.reset();

    if (bench) {
        std::printf("\n%10s %8s %9s %9s %9s %9s   (%dx%d, %d-frame windows)\n", "particles", "fps",
                    "sim ms", "frame ms", "gbuf ms", "shade ms", size.width(), size.height(),
                    kBenchWindow);
        for (const auto& r : benchRows)
            std::printf("%10u %8.1f %9.2f %9.2f %9.3f %9.3f\n", r.n, r.fps, r.simMs, r.frameMs,
                        r.gbufMs, r.shadeMs);
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
            // THE belt gate: a cohort of grains must be carried along travel at
            // more than half the belt speed. Half, not all, because a grain
            // spends its first metre being accelerated from the chute's velocity
            // and its last one falling off the end.
            const float carried = l.cohortX1 - l.cohortX0;
            const float floorDist = 0.5f * kBeltSpeed * (float(kCohortSpan) * kDt);
            if (frames < kCohortStart + kCohortSpan) fail("run too short to measure transport");
            if (!(carried > floorDist)) fail(std::string(l.name) + ": the belt did not carry it");
            if (l.stats.spilled == 0)
                fail(std::string(l.name) + ": nothing reached the discharge end");
            // Overrun check: a jammed belt can still pass the transport gate on
            // the creep of its bottom layer, so pin the bed depth too.
            if (!(l.stats.bedTop < 2.f * kGuideHeight))
                fail(std::string(l.name) + ": the belt is overloaded (bed over the guides)");
            std::printf("selftest[%s]: n=%u carried=%.2f m/%.1f s (floor %.2f) spilled=%u "
                        "bed=%.3f minY=%.4f pile=%.2f/%.2f aspect=%.3f\n",
                        l.name, l.stats.n, double(carried), double(float(kCohortSpan) * kDt),
                        double(floorDist), l.stats.spilled, double(l.stats.bedTop),
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
