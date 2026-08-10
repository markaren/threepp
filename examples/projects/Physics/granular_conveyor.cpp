// GPU granular material on a conveyor: PhysX 5.5 PxPBDParticleSystem grains
// poured from a chute onto the belt from extras/conveyor, carried along it, and
// spilling off the discharge end into a stockpile.
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
//   granular_conveyor --count 200000        particle budget
//   granular_conveyor --rate 4000           grains per second poured
//   granular_conveyor --drop                phase-1 mode: no belt, block drop

#include "threepp/threepp.hpp"

#include "threepp/extras/conveyor/ConveyorPhysics.hpp"
#include "threepp/extras/physx/PhysxParticles.hpp"

#include "capture_util.hpp"

#include <algorithm>
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
    // the ceiling is ~3000 grains/s, which is the default --rate.
    //
    // The guides are TALLER than that formula's usable depth on purpose: real
    // skirtboards are, and a measured 2800 grains/s settles into a 0.29 m bed,
    // so a 0.22 m guide had grain walking over the edge all the way down.
    constexpr float kBeltY = 1.90f;    // conveying surface height (a stacker)
    constexpr float kBeltX0 = -4.0f;   // inlet end
    constexpr float kBeltX1 = 4.0f;    // discharge end
    constexpr float kBeltWidth = 1.00f;
    constexpr float kBeltSpeed = 2.2f; // m/s along +X
    constexpr float kGuideHeight = 0.32f;
    // A metre of belt behind the pour, so grain that bounces backwards off the
    // bed is picked up again instead of dribbling off the tail drum.
    constexpr float kChuteX = -3.00f;

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

    // ── Telemetry: the numbers the self-test gates on ─────────────────────────
    struct Stats {
        unsigned n = 0;
        unsigned bad = 0;// non-finite components — must stay 0
        float minY = 0.f, maxY = 0.f;
        float meanX = 0.f;
        float spread = 0.f;  // RMS horizontal distance from the mean
        unsigned spilled = 0;// past the discharge AND below the belt
        // Deepest point of the bed riding the middle of the belt, measured from
        // the conveying surface. This is the overrun alarm: once it passes the
        // guide height the belt is being poured into faster than it can pump,
        // and everything downstream of that (transport rate, pile shape) is
        // measuring a jam rather than a conveyor.
        float bedTop = 0.f;
        float pileTop = 0.f;// highest grain in the stockpile, above the floor
    };

    Stats measure(const ::physx::PxVec4* p, unsigned n) {
        Stats s;
        s.n = n;
        if (n == 0) return s;
        s.minY = s.maxY = p[0].y;
        double sx = 0, sz = 0;
        for (unsigned i = 0; i < n; ++i) {
            const auto& v = p[i];
            if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
                ++s.bad;
                continue;
            }
            s.minY = std::min(s.minY, v.y);
            s.maxY = std::max(s.maxY, v.y);
            sx += v.x;
            sz += v.z;
            if (v.x > kBeltX1 && v.y < kBeltY - 0.3f) ++s.spilled;
            // Past the discharge drum: the stockpile, whatever height it has
            // reached. Measured without the belt filter above, or the probe
            // reads back its own threshold.
            // The discharge stream is still 2 m up for the first metre past the
            // lip, so start beyond where it lands or the probe measures the
            // trajectory instead of the pile.
            if (v.x > kBeltX1 + 1.4f) s.pileTop = std::max(s.pileTop, v.y);
            // Skip the metre downstream of the chute: grain in free fall, and
            // the heap it makes on landing, are not the conveyed bed.
            if (v.x > kChuteX + 1.0f && v.x < kBeltX1 - 0.3f && v.y > kBeltY)
                s.bedTop = std::max(s.bedTop, v.y - kBeltY);
        }
        const unsigned good = n - s.bad;
        if (!good) return s;
        s.meanX = float(sx / good);
        const double mz = sz / good;
        double sq = 0;
        for (unsigned i = 0; i < n; ++i) {
            const auto& v = p[i];
            if (!std::isfinite(v.x) || !std::isfinite(v.z)) continue;
            const double dx = v.x - s.meanX, dz = v.z - mz;
            sq += dx * dx + dz * dz;
        }
        s.spread = float(std::sqrt(sq / good));
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

}// namespace

int main(int argc, char** argv) {

    std::string shot;
    std::vector<int> shotFrames{150, 450, 900};
    bool selftest = false;
    bool dropMode = false;
    unsigned budget = 60000;
    float rate = 2800.f;
    int frames = 0;// 0 = per-mode default
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shot = argv[++i];
        else if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--drop") == 0) dropMode = true;
        else if (std::strcmp(argv[i], "--count") == 0 && i + 1 < argc)
            budget = unsigned(std::max(1, std::atoi(argv[++i])));
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
    const bool offscreen = !shot.empty() || selftest;
    if (frames <= 0) frames = dropMode ? 300 : 900;

    // ── The GPU gate ─────────────────────────────────────────────────────────
    // Constructing a GPU world is the only honest probe (driver, device and the
    // PhysX GPU library all have to line up), so try it and report.
    PhysxWorld::Settings ws;
    ws.enableGpuDynamics = true;
    // A PBD system's neighbourhood + contact pools live in the GPU heap; the
    // 64 MB default is sized for a handful of rigid bodies.
    ws.gpuHeapCapacityMB = 512;
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
    PbdParticles particles(*world, ps);
    const float radius = particles.solidRestOffset();

    // Gravel: high friction so the belt can drag it and the pile can hold an
    // angle. Damping bleeds the energy a 1 m drop onto the belt puts in.
    PbdParticles::MaterialSpec gravelSpec;
    gravelSpec.friction = 0.95f;
    gravelSpec.damping = 0.2f;
    auto& gravel = particles.addGroup(budget, gravelSpec);

    // ── Scene ────────────────────────────────────────────────────────────────
    Canvas canvas(Canvas::Parameters()
                          .title("PBD Granular Conveyor")
                          .size(offscreen ? WindowSize{1280, 720} : WindowSize{1600, 900})
                          .vsync(!offscreen)
                          .headless(offscreen));
    // Offscreen runs pick a backend for themselves — createRenderer with no
    // argument reads stdin. Vulkan is the target (deferred lighting + shadows on
    // a 100k-instance field is what the demo is for), but THREEPP_WITH_VULKAN is
    // PRIVATE to the library, so ask for it and fall back on the throw instead
    // of guessing at compile time. That is also the CPU-only-CI path.
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

    Scene scene;
    scene.background = Color(0x86a3c0);

    auto sun = DirectionalLight::create(0xfff3e0, 3.2f);
    sun->position.set(7.f, 10.f, 5.f);
    sun->castShadow = true;
    scene.add(sun);
    scene.add(AmbientLight::create(0x8899bb, 0.35f));

    auto floorMat = MeshStandardMaterial::create();
    floorMat->color = Color(0x6f6b64);
    floorMat->roughness = 0.95f;
    auto floor = Mesh::create(BoxGeometry::create(28.f, 0.4f, 20.f), floorMat);
    floor->position.y = -0.2f;
    floor->receiveShadow = true;
    scene.add(floor);
    // Grippy floor: a slick one lets the stockpile skate outward instead of
    // stacking, which is the whole thing the pile angle is supposed to show.
    auto* floorPhys = world->physics().createMaterial(0.9f, 0.9f, 0.f);
    world->addStatic(*floor, floorPhys);

    auto gravelMat = MeshStandardMaterial::create();
    gravelMat->color = Color(0x6b6258);
    gravelMat->roughness = 0.9f;
    gravelMat->metalness = 0.f;
    gravelMat->flatShading = true;
    // detail 0 = a 20-face icosahedron: a faceted pebble at ~1/5 the triangles
    // of even a coarse UV sphere, which matters 60 000 times over.
    GrainField field(IcosahedronGeometry::create(radius, 0), gravelMat, budget, 7u);
    field.mesh().castShadow = true;
    field.mesh().receiveShadow = true;
    scene.add(field.shared());

    std::unique_ptr<cv::ConveyorPhysics> beltSim;
    std::shared_ptr<MeshStandardMaterial> beltMat;
    float beltScroll = 0.f;
    Chute chute(Vector3(kChuteX, kBeltY + 0.42f, 0.f), kBeltWidth, ps.spacing, 991u);

    if (!dropMode) {
        auto spec = laneSpec(0.f);
        beltMat = addConveyorVisual(scene, spec);
        beltScroll = -spec.speed * beltMat->map->repeat.y;// travel is +X (not reversed)
        beltSim = std::make_unique<cv::ConveyorPhysics>(*world, std::vector<cv::ConveyorSpec>{spec});
        std::cout << "belt: " << beltSim->beltCount() << " drag segments, "
                  << beltSim->wallCount() << " guide segments, " << kBeltSpeed << " m/s"
                  << std::endl;

        // The chute itself: an open cone whose mouth covers the emission slab,
        // so grain reads as falling OUT of something instead of appearing in
        // mid-air. Purely visual.
        auto chuteMat = MeshStandardMaterial::create();
        chuteMat->color = Color(0x767d84);
        chuteMat->roughness = 0.5f;
        chuteMat->metalness = 0.85f;
        chuteMat->side = Side::Double;
        auto cone = Mesh::create(CylinderGeometry::create(0.52f, 0.26f, 0.50f, 20, 1, true),
                                 chuteMat);
        cone->position.set(kChuteX, chute.slabTop() + 0.22f, 0.f);
        cone->castShadow = true;
        scene.add(cone);

        addTailSkirt(scene, *world, 0.f, chuteMat);
    }

    // Framing has to hold chute -> belt -> stockpile in one shot, which is an
    // ~13 m span, so the FOV is narrow and the eye is far back rather than wide
    // and close (a wide lens this near puts the near guide rail across half the
    // frame).
    // The eye also has to clear the skirtboards: a 0.32 m guide on a 1.0 m belt
    // hides the bed entirely below ~18 degrees of elevation, and the conveyed
    // stream IS the thing being demonstrated. ~27 degrees looks down into it.
    const Vector3 look = dropMode ? Vector3(0.f, 0.35f, 0.f) : Vector3(1.6f, 1.15f, 0.f);
    PerspectiveCamera camera(dropMode ? 42.f : 32.f, canvas.aspect(), 0.05f, 300.f);
    camera.position.copy(dropMode ? Vector3(3.6f, 2.0f, 4.4f) : Vector3(8.0f, 7.4f, 10.4f));
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
        gravel.emit(seeded.data(), unsigned(seeded.size()), Vector3(0.f, -0.5f, 0.f), 0.02f);
    }
    std::cout << "granular_conveyor: capacity " << budget << ", radius " << radius
              << " m, rate " << rate << "/s, GPU heap " << ws.gpuHeapCapacityMB << " MB"
              << std::endl;

    // ── Loop ─────────────────────────────────────────────────────────────────
    // Fixed dt, so a headless run is frame-for-frame reproducible and "frame N"
    // means the same sim time every time.
    constexpr float kDt = 1.f / 60.f;
    constexpr unsigned kCohort = 2000;// grains followed down the belt
    constexpr int kCohortStart = 30;  // frame the cohort measurement starts
    constexpr int kCohortSpan = 300;  // 5 sim seconds later
    int frame = 0;
    Stats last;
    Stats at60, at240;
    float cohortX0 = 0.f, cohortX1 = 0.f;
    bool ok = true;
    const auto fail = [&](const char* what) {
        std::cout << "SELFTEST FAIL: " << what << std::endl;
        ok = false;
    };

    canvas.animate([&] {
        if (!dropMode) {
            const auto& burst = chute.tick(kDt, rate);
            if (!burst.empty())
                gravel.emit(burst.data(), unsigned(burst.size()),
                            Vector3(0.4f, -1.4f, 0.f), 0.02f);
        }

        world->step(kDt);
        particles.pull();
        field.update(gravel.positions(), gravel.active());

        if (beltMat) {
            beltMat->map->offset.y += beltScroll * kDt;
            beltMat->needsUpdate();
        }

        renderer->render(scene, camera);
        ++frame;

        last = measure(gravel.positions(), gravel.active());
        if (last.bad) fail("non-finite particle positions");
        if (frame == 60) at60 = last;
        if (frame == 240) at240 = last;
        if (frame == kCohortStart) cohortX0 = cohortMeanX(gravel.positions(), std::min(kCohort, last.n));
        if (frame == kCohortStart + kCohortSpan)
            cohortX1 = cohortMeanX(gravel.positions(), std::min(kCohort, last.n));

        if (frame % 150 == 0) {
            std::printf("[f%4d] n=%6u bed=%.3f pile=%.2f meanX=%+.2f spread=%.2f spilled=%6u "
                        "minY=%.3f\n",
                        frame, last.n, double(last.bedTop), double(last.pileTop),
                        double(last.meanX), double(last.spread), last.spilled, double(last.minY));
            std::fflush(stdout);
        }

        if (!shot.empty() &&
            std::find(shotFrames.begin(), shotFrames.end(), frame) != shotFrames.end()) {
            char name[256];
            std::snprintf(name, sizeof(name), "%s_f%04d.png", shot.c_str(), frame);
            const auto p = capture::shotOutputPath(name);
            renderer->writeFramebuffer(p);
            std::cout << "wrote " << p.string() << " (n=" << last.n << ")" << std::endl;
        }
        if (offscreen && frame >= frames) canvas.close();
    });

    // ConveyorPhysics borrows the world and must die first (it unregisters its
    // substep hooks and releases its actors); so must the particle system.
    beltSim.reset();

    if (!selftest) return 0;

    // ── Numeric gates ────────────────────────────────────────────────────────
    if (last.bad) fail("non-finite particle positions");
    // The floor's top face is y = 0 and a resting grain's centre sits one radius
    // above it, so anything materially below 0 has tunnelled.
    if (last.minY < -0.5f * radius) fail("particles tunnelled through the floor");

    if (dropMode) {
        if (!(last.maxY < at60.maxY)) fail("the block never collapsed (maxY did not fall)");
        if (!(at240.spread > at60.spread * 1.05f)) fail("the pile never spread out");
        if (!(std::abs(last.spread - at240.spread) < 0.03f * at240.spread))
            fail("the pile never settled (spread still moving at the end)");
        std::printf("selftest[drop]: n=%u minY=%.4f maxY=%.4f spread60=%.3f spread240=%.3f "
                    "spreadEnd=%.3f\n",
                    last.n, double(last.minY), double(last.maxY), double(at60.spread),
                    double(at240.spread), double(last.spread));
    } else {
        // THE belt gate: a cohort of grains must be carried along travel at
        // more than half the belt speed. Half, not all, because a grain spends
        // its first metre being accelerated from the chute's velocity and its
        // last one falling off the end.
        const float carried = cohortX1 - cohortX0;
        const float floorDist = 0.5f * kBeltSpeed * (float(kCohortSpan) * kDt);
        if (frames < kCohortStart + kCohortSpan) fail("run too short to measure transport");
        if (!(carried > floorDist)) fail("the belt did not carry the grains");
        if (last.spilled == 0) fail("nothing reached the discharge end");
        // Overrun check: a jammed belt can still pass the transport gate if the
        // bottom layer creeps, so pin the bed depth too.
        if (!(last.bedTop < 2.f * kGuideHeight))
            fail("the belt is overloaded (bed climbed over the guides)");
        std::printf("selftest[belt]: n=%u carried=%.2f m in %.1f s (floor %.2f m) spilled=%u "
                    "bed=%.3f (guide %.2f) minY=%.4f\n",
                    last.n, double(carried), double(float(kCohortSpan) * kDt), double(floorDist),
                    last.spilled, double(last.bedTop), double(kGuideHeight), double(last.minY));
    }

    std::cout << (ok ? "SELFTEST PASS" : "SELFTEST FAIL") << std::endl;
    return ok ? 0 : 1;
}
