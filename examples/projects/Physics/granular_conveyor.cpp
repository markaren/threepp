// GPU granular material (PhysX 5.5 PxPBDParticleSystem) — phase 1: the drop
// test. N particles are seeded as a block above a static floor and let go; the
// pile they form is the evidence the solver, the emitter and the readback all
// agree. The conveyor arrives in the next phase; the scaffolding here (instanced
// spheres created ONCE at capacity, headless capture, numeric self-test) is what
// that will be poured onto.
//
// PBD particles are a CUDA-only PhysX feature with no CPU fallback. This file
// COMPILES anywhere PhysX does — the headers ship regardless — and at runtime
// prints why and exits 0 when there is no GPU, so CI can build and run it.
//
//   granular_conveyor                       interactive (pick a backend)
//   granular_conveyor --shot pbd_drop       headless capture -> aaa_caps/
//   granular_conveyor --selftest            headless numeric gates, no window
//   granular_conveyor --count 200000        particle budget

#include "threepp/threepp.hpp"

#include "threepp/extras/physx/PhysxParticles.hpp"

#include "capture_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace threepp;

namespace {

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
            // Grains beyond `count()` are never drawn; the up-to-kStep-1 spare
            // slots inside the current step are parked at zero scale instead
            // (a zero 3x3 collapses the instance to a point — no pixels, no
            // triangles) until a grain claims them.
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
                float* b = e.data() + std::size_t(i) * 16;
                // Copy the rotation only; translation stays 0 and w stays 1
                // until the grain is claimed (see claim() / setPosition()).
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 3; ++r) rot_[std::size_t(i) * 9 + c * 3 + r] = m.elements[c * 4 + r];
                b[15] = 1.f;
            }
        }

        // Per-frame: point the field at `n` live grain positions (PxVec4, w =
        // inverse mass, ignored here).
        void update(const ::physx::PxVec4* positions, unsigned n) {

            n = std::min(n, capacity_);
            auto& e = mesh_->instanceMatrix()->array();
            float* base = e.data();
            // Newly claimed slots get their (fixed) rotation written in; from
            // then on only the translation moves.
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

            // Coarse count steps: pay the entry-list rebuild once per kStep
            // grains, not once per emitted burst.
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

    // ── Telemetry: the numbers the self-test gates on ─────────────────────────
    struct Stats {
        unsigned n = 0;
        unsigned bad = 0;// non-finite components — must stay 0
        float minY = 0.f, maxY = 0.f;
        float meanX = 0.f, meanZ = 0.f;
        float spread = 0.f;// RMS horizontal distance from the mean: pile width
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
        }
        const unsigned good = n - s.bad;
        if (!good) return s;
        s.meanX = float(sx / good);
        s.meanZ = float(sz / good);
        double sq = 0;
        for (unsigned i = 0; i < n; ++i) {
            const auto& v = p[i];
            if (!std::isfinite(v.x) || !std::isfinite(v.z)) continue;
            const double dx = v.x - s.meanX, dz = v.z - s.meanZ;
            sq += dx * dx + dz * dz;
        }
        s.spread = float(std::sqrt(sq / good));
        return s;
    }

    // A block of grid positions, jittered so the initial packing is not a
    // perfect lattice (a lattice releases as one rigid slab and rings).
    std::vector<Vector3> block(unsigned count, float spacing, const Vector3& centerBottom,
                               unsigned seed) {
        std::vector<Vector3> out;
        out.reserve(count);
        const auto side = unsigned(std::ceil(std::cbrt(double(count))));
        const float d = spacing * 1.08f;// a hair apart, so nothing starts overlapping
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
    bool selftest = false;
    unsigned budget = 50000;
    int frames = 0;// 0 = per-mode default
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shot = argv[++i];
        else if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--count") == 0 && i + 1 < argc)
            budget = unsigned(std::max(1, std::atoi(argv[++i])));
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) frames = std::atoi(argv[++i]);
    }
    const bool offscreen = !shot.empty() || selftest;
    if (frames <= 0) frames = 300;

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
    ps.spacing = 0.06f;
    ps.solverIterations = 8;
    PbdParticles particles(*world, ps);
    const float radius = particles.solidRestOffset();

    // Gravel: high friction, a touch of cohesion so it holds a steep pile.
    PbdParticles::MaterialSpec gravelSpec;
    gravelSpec.friction = 0.85f;
    gravelSpec.damping = 0.15f;
    gravelSpec.cohesion = 0.f;
    auto& gravel = particles.addGroup(budget, gravelSpec);

    // ── Scene ────────────────────────────────────────────────────────────────
    Canvas canvas(Canvas::Parameters()
                          .title("PBD Granular")
                          .size(offscreen ? WindowSize{1280, 720} : WindowSize{1600, 900})
                          .vsync(!offscreen)
                          .headless(offscreen));
    // Offscreen runs pick a backend for themselves — createRenderer with no
    // argument reads stdin. Vulkan is the target (deferred lighting + shadows
    // on a 100k-instance field is what the demo is for), but THREEPP_WITH_VULKAN
    // is PRIVATE to the library, so ask for it and fall back on the throw
    // instead of guessing at compile time. That is also the CPU-only-CI path.
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
    sun->position.set(6.f, 9.f, 4.f);
    sun->castShadow = true;
    scene.add(sun);
    scene.add(AmbientLight::create(0x8899bb, 0.35f));

    auto floorMat = MeshStandardMaterial::create();
    floorMat->color = Color(0x6f6b64);
    floorMat->roughness = 0.95f;
    auto floor = Mesh::create(BoxGeometry::create(10.f, 0.4f, 10.f), floorMat);
    floor->position.y = -0.2f;
    floor->receiveShadow = true;
    scene.add(floor);
    // Grippy floor: a slick one lets the pile skate outward instead of stacking.
    auto* floorPhys = world->physics().createMaterial(0.9f, 0.9f, 0.f);
    world->addStatic(*floor, floorPhys);

    auto gravelMat = MeshStandardMaterial::create();
    gravelMat->color = Color(0x6b6258);
    gravelMat->roughness = 0.9f;
    gravelMat->metalness = 0.f;
    gravelMat->flatShading = true;
    // detail 0 = a 20-face icosahedron: a faceted pebble at ~1/5 the triangles
    // of even a coarse UV sphere, which matters 50 000 times over.
    GrainField field(IcosahedronGeometry::create(radius, 0), gravelMat, budget, 7u);
    field.mesh().castShadow = true;
    field.mesh().receiveShadow = true;
    scene.add(field.shared());

    PerspectiveCamera camera(45.f, canvas.aspect(), 0.05f, 200.f);
    camera.position.set(3.6f, 2.0f, 4.4f);
    camera.lookAt(Vector3(0.f, 0.35f, 0.f));

    std::unique_ptr<OrbitControls> controls;
    if (!offscreen) {
        controls = std::make_unique<OrbitControls>(camera, canvas);
        controls->target.set(0.f, 0.35f, 0.f);
    }
    canvas.onWindowResize([&](WindowSize s) {
        camera.aspect = s.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(s);
    });

    // Seed the block. One shot: this phase is about whether a released block
    // falls, collides and settles, not about pouring.
    const auto seeded = block(budget, ps.spacing, Vector3(0.f, 0.9f, 0.f), 4242u);
    const unsigned emitted = gravel.emit(seeded.data(), unsigned(seeded.size()),
                                         Vector3(0.f, -0.5f, 0.f), 0.02f);
    std::cout << "granular_conveyor: " << emitted << " particles, radius " << radius
              << " m, GPU heap " << ws.gpuHeapCapacityMB << " MB" << std::endl;

    // ── Loop ─────────────────────────────────────────────────────────────────
    // Fixed dt, so a headless run is frame-for-frame reproducible and "frame N"
    // means the same sim time every time.
    constexpr float kDt = 1.f / 60.f;
    int frame = 0;
    Stats last;
    Stats at60, at240;
    bool ok = true;
    const auto fail = [&](const char* what) {
        std::cout << "SELFTEST FAIL: " << what << std::endl;
        ok = false;
    };

    canvas.animate([&] {
        world->step(kDt);
        particles.pull();
        field.update(gravel.positions(), gravel.active());

        renderer->render(scene, camera);
        ++frame;

        last = measure(gravel.positions(), gravel.active());
        if (last.bad) fail("non-finite particle positions");
        if (frame == 60) at60 = last;
        if (frame == 240) at240 = last;

        if (frame % 60 == 0 || frame == 30) {
            std::printf("[f%4d] n=%6u y[%.3f..%.3f] spread=%.3f\n", frame, last.n,
                        double(last.minY), double(last.maxY), double(last.spread));
            std::fflush(stdout);
        }

        if (!shot.empty() && (frame == 30 || frame == 120 || frame == frames)) {
            char name[256];
            std::snprintf(name, sizeof(name), "%s_f%04d.png", shot.c_str(), frame);
            const auto p = capture::shotOutputPath(name);
            renderer->writeFramebuffer(p);
            std::cout << "wrote " << p.string() << std::endl;
        }
        if (offscreen && frame >= frames) canvas.close();
    });

    if (!selftest) return 0;

    // ── Numeric gates ────────────────────────────────────────────────────────
    // The floor's top face is y = 0 and a resting grain's centre sits one
    // radius above it, so anything materially below 0 has tunnelled.
    if (last.n != emitted) fail("particle count changed during the run");
    if (last.minY < -0.5f * radius) fail("particles tunnelled through the floor");
    if (!(last.maxY < at60.maxY)) fail("the block never collapsed (maxY did not fall)");
    if (!(at240.spread > at60.spread * 1.05f)) fail("the pile never spread out");
    // Settled: the last 60 steps must barely move the footprint. An exploding
    // or jittering pile keeps growing.
    if (!(std::abs(last.spread - at240.spread) < 0.03f * at240.spread))
        fail("the pile never settled (spread still moving at the end)");

    std::printf("selftest: n=%u minY=%.4f maxY=%.4f spread60=%.3f spread240=%.3f spreadEnd=%.3f\n",
                last.n, double(last.minY), double(last.maxY), double(at60.spread),
                double(at240.spread), double(last.spread));
    std::cout << (ok ? "SELFTEST PASS" : "SELFTEST FAIL") << std::endl;
    return ok ? 0 : 1;
}
