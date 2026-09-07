// Replay audit — the go/no-go check for the sensor-determinism claim.
//
// Runs the sensor_rig leg (same articulation, same drives, same sensors, no
// window and no renderer) for a fixed number of physics substeps, drains every
// sensor each substep, and folds every field of every packet into one FNV-1a
// hash chain per sensor. The manifest it writes is the whole result:
//
//     replay_audit --seconds 30 --out a.txt
//     replay_audit --seconds 30 --out b.txt      # fresh process, same seed
//     replay_audit --compare a.txt b.txt         # exit 0 = bit-identical
//
// If two fresh processes on the same machine and commit disagree, the
// "seeded and sim-clock-stamped" contract (Sensor.hpp) is broken somewhere
// between PhysX stepping and the drain() ring, and that leak must be found
// before any of it is worth writing up. Noise is ON (the seeded MEMS
// defaults): seeded noise reproducing bit-for-bit is part of the contract,
// not an obstacle to the test.
//
// Fields are serialized one by one — never memcpy a struct: padding bytes are
// indeterminate and would make the hash flaky by construction.

#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/ContactSensor.hpp"
#include "threepp/extras/sensors/ForceTorqueSensor.hpp"
#include "threepp/extras/sensors/Imu.hpp"
#include "threepp/extras/sensors/JointEncoder.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <PxPhysicsAPI.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr float kSubstep = 1.f / 240.f;

    // FNV-1a 64. Chained per sensor: order matters, which is the point — a
    // reordered stream is as much a determinism failure as a changed value.
    class Fnv {
    public:
        void bytes(const void* p, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) {
                h_ ^= b[i];
                h_ *= 0x100000001B3ULL;
            }
        }
        void f32(float v) { bytes(&v, sizeof v); }
        void f64(double v) { bytes(&v, sizeof v); }
        void u32(std::uint32_t v) { bytes(&v, sizeof v); }
        void b8(bool v) {
            const unsigned char c = v ? 1 : 0;
            bytes(&c, 1);
        }
        void vec3(const Vector3& v) {
            f32(v.x);
            f32(v.y);
            f32(v.z);
        }
        [[nodiscard]] std::uint64_t value() const { return h_; }

    private:
        std::uint64_t h_ = 0xCBF29CE484222325ULL;
    };

    struct Stream {
        Fnv hash;
        std::uint64_t count = 0;
    };

    int compare(const std::string& pathA, const std::string& pathB) {
        auto load = [](const std::string& path) {
            std::map<std::string, std::string> rows;
            std::ifstream f(path);
            if (!f) throw std::runtime_error("cannot open " + path);
            std::string name, rest;
            while (f >> name && std::getline(f, rest)) rows[name] = rest;
            return rows;
        };
        const auto a = load(pathA);
        const auto b = load(pathB);
        int bad = 0;
        for (const auto& [name, row] : a) {
            const auto it = b.find(name);
            if (it == b.end()) {
                std::cout << "MISSING  " << name << " (only in " << pathA << ")\n";
                ++bad;
            } else if (it->second != row) {
                std::cout << "DIFF     " << name << "\n  " << pathA << ":" << row
                          << "\n  " << pathB << ":" << it->second << "\n";
                ++bad;
            } else {
                std::cout << "OK       " << name << row << "\n";
            }
        }
        for (const auto& [name, row] : b) {
            if (!a.count(name)) {
                std::cout << "MISSING  " << name << " (only in " << pathB << ")\n";
                ++bad;
            }
        }
        std::cout << (bad ? "NO-GO: " : "GO: ") << (a.size() - bad) << "/" << a.size()
                  << " streams bit-identical\n";
        return bad ? 1 : 0;
    }

}// namespace

int main(int argc, char** argv) {

    double seconds = 30.0;
    std::string outPath;
    // Island-invariance probe: --bystander drops an unrelated dynamic box far
    // from the leg (its own island; it only ever touches the ground). Same-seed
    // fresh-process replay must hold with or without it — the question the pair
    // of flags answers is whether the box's PRESENCE changes the LEG's streams:
    // without eENABLE_ENHANCED_DETERMINISM the solver batches constraints
    // across islands and may; with --enhanced it must not.
    bool bystander = false;
    bool enhanced = false;
    // --threads: solver worker count (Settings::numThreads). The question this
    // knob asks: is the simulation bit-exact ACROSS thread counts, and does
    // eENABLE_ENHANCED_DETERMINISM change the answer?
    unsigned threads = 2;
    // --seed: offsets the IMU noise seeds. Instrument validation, not an
    // experiment: a changed seed MUST flip the imu row and MUST NOT touch the
    // noiseless streams — a hash that can't detect change proves nothing when
    // it matches.
    std::uint64_t seedOffset = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (arg == "--bystander") bystander = true;
        else if (arg == "--enhanced") enhanced = true;
        else if (arg == "--threads" && i + 1 < argc) threads = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (arg == "--seed" && i + 1 < argc) seedOffset = static_cast<std::uint64_t>(std::atoll(argv[++i]));
        else if (arg == "--compare" && i + 2 < argc) return compare(argv[i + 1], argv[i + 2]);
    }

    // ---- the sensor_rig leg, verbatim minus the window --------------------

    auto scene = Scene::create();

    auto limb = [&](float w, float h, float d) {
        auto mesh = Mesh::create(BoxGeometry::create(w, h, d), MeshStandardMaterial::create());
        scene->add(mesh);
        return mesh;
    };

    PhysxWorld::Settings settings;
    settings.fixedTimestep = kSubstep;
    settings.maxSubSteps = 8;
    settings.enhancedDeterminism = enhanced;
    settings.numThreads = threads;
    PhysxWorld world(settings);

    auto ground = limb(40, 1, 40);
    ground->position.y = -0.5f;
    world.addStatic(*ground);

    ::physx::PxRigidDynamic* box = nullptr;
    if (bystander) {
        // Far enough that it can never touch the leg; bouncing on the ground
        // keeps its island alive (contacts every few substeps) for the whole
        // run, so cross-island solver batching has something to batch with.
        using namespace ::physx;
        box = world.addDynamic(PxBoxGeometry(0.3f, 0.3f, 0.3f),
                               PxTransform(PxVec3(15.f, 3.f, 0.f)), 500.f);
    }

    constexpr float kThighLen = 0.9f;
    constexpr float kShinLen = 0.9f;
    constexpr float kHipY = 1.78f;// foot overlaps the floor at full reach → contact events

    auto baseMesh = limb(0.5f, 0.4f, 0.5f);
    baseMesh->position.set(0.f, kHipY + 0.3f, 0.f);
    auto thighMesh = limb(0.18f, kThighLen, 0.18f);
    thighMesh->position.set(0.f, kHipY - kThighLen * 0.5f, 0.f);
    auto shinMesh = limb(0.16f, kShinLen, 0.16f);
    shinMesh->position.set(0.f, kHipY - kThighLen - kShinLen * 0.5f, 0.f);

    Articulation leg(world, /*fixedBase*/ true, /*solverPositionIters*/ 32,
                     /*disableSelfCollision*/ true);
    auto base = leg.addLink(nullptr, *baseMesh, 800.f, {0, 0, 1}, {0, kHipY, 0},
                            false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, "revolute", 0.f, nullptr);
    auto thigh = leg.addLink(&base, *thighMesh, 900.f, {0, 0, 1}, {0, kHipY, 0},
                             false, 0.f, 0.f, 4000.f, 450.f, 1e6f, 0.f, "revolute", 0.f, nullptr);
    auto shin = leg.addLink(&thigh, *shinMesh, 900.f, {0, 0, 1}, {0, kHipY - kThighLen, 0},
                            false, 0.f, 0.f, 3000.f, 350.f, 1e6f, 0.f, "revolute", 0.f, nullptr);
    leg.finalize();

    // Noise ON, sensor_rig's exact seeds (plus any --seed offset): seeded
    // noise replaying bit-for-bit is part of what is under test.
    Imu imu(*shinMesh, 200.0);
    if (seedOffset) {
        imu.gyroNoise.seed += seedOffset;
        imu.accelNoise.seed += seedOffset;
        imu.reset();
    }
    JointEncoder hipEnc(*thighMesh, thigh, 100.0);
    JointEncoder kneeEnc(*shinMesh, shin, 100.0);
    hipEnc.setCountsPerRev(1024);
    kneeEnc.setCountsPerRev(1024);
    ForceTorqueSensor hipFt(*thighMesh, leg, thigh, 200.0);
    ContactSensor foot(*shinMesh);

    world.registerSensor(&imu);
    world.registerSensor(&hipEnc);
    world.registerSensor(&kneeEnc);
    world.registerSensor(&hipFt);
    world.registerSensor(&foot);

    double simClock = 0.0;
    world.onPreSubstep([&](float dt) {
        simClock += dt;
        const auto phase = static_cast<float>(simClock) * 0.6f * 2.f * math::PI;
        thigh.setDriveTarget(0.45f * std::sin(phase));
        shin.setDriveTarget(-1.0f * (1.f - std::cos(phase)) * 0.5f);
    });

    // ---- step + hash -------------------------------------------------------

    Stream sImu, sHip, sKnee, sFt, sContact, sTruth;

    std::vector<ImuSample> imuBuf;
    std::vector<JointSample> jointBuf;
    std::vector<WrenchSample> ftBuf;
    std::vector<ContactSample> contactBuf;

    const auto steps = static_cast<std::uint64_t>(seconds / kSubstep + 0.5);
    for (std::uint64_t i = 0; i < steps; ++i) {
        world.step(kSubstep);

        imu.drain(imuBuf);
        for (const auto& s : imuBuf) {
            sImu.hash.f64(s.t);
            sImu.hash.vec3(s.angularVelocity);
            sImu.hash.vec3(s.linearAcceleration);
            ++sImu.count;
        }
        for (auto [enc, stream] : {std::pair{&hipEnc, &sHip}, std::pair{&kneeEnc, &sKnee}}) {
            enc->drain(jointBuf);
            for (const auto& s : jointBuf) {
                stream->hash.f64(s.t);
                stream->hash.f32(s.position);
                stream->hash.f32(s.velocity);
                ++stream->count;
            }
        }
        hipFt.drain(ftBuf);
        for (const auto& s : ftBuf) {
            sFt.hash.f64(s.t);
            sFt.hash.vec3(s.force);
            sFt.hash.vec3(s.torque);
            ++sFt.count;
        }
        foot.drain(contactBuf);
        for (const auto& s : contactBuf) {
            sContact.hash.f64(s.t);
            sContact.hash.b8(s.inContact);
            sContact.hash.b8(s.touchBegan);
            sContact.hash.b8(s.touchEnded);
            sContact.hash.vec3(s.force);
            sContact.hash.u32(s.pointCount);
            sContact.hash.u32(s.observedPoints);
            // Points are world-space geometry; the against-actor pointer is
            // process-specific by nature and deliberately NOT hashed.
            for (std::uint32_t p = 0; p < s.pointCount; ++p) {
                sContact.hash.vec3(s.points[p].position);
                sContact.hash.vec3(s.points[p].normal);
                sContact.hash.f32(s.points[p].impulse);
            }
            ++sContact.count;
        }
        // Physics truth, unfiltered by any sensor: separates "the sensors
        // leak" from "the simulation itself diverged".
        sTruth.hash.f32(thigh.jointPosition());
        sTruth.hash.f32(shin.jointPosition());
        ++sTruth.count;
    }

    std::ostringstream manifest;
    auto row = [&](const char* name, const Stream& s) {
        manifest << name << " count=" << s.count << " fnv=" << std::hex << s.hash.value()
                 << std::dec << "\n";
    };
    row("imu", sImu);
    row("encoder.hip", sHip);
    row("encoder.knee", sKnee);
    row("forceTorque.hip", sFt);
    row("contact.foot", sContact);
    row("physicsTruth", sTruth);

    std::cout << "replay_audit: " << steps << " substeps @ " << 1.f / kSubstep << " Hz ("
              << seconds << " s sim)"
              << " [threads=" << threads << "]"
              << (enhanced ? " [enhancedDeterminism]" : "")
              << (bystander ? " [bystander]" : "")
              << (seedOffset ? " [seedOffset]" : "") << "\n";
    if (box) {
        // Config echo, stdout only (NOT the manifest — the manifest must stay
        // comparable across configs). A box that never simulated would still
        // sit at its spawn pose (15, 3, 0); one that fell and settled on the
        // ground plane proves the probe was live.
        const auto p = box->getGlobalPose().p;
        std::cout << "bystander final pose: " << p.x << " " << p.y << " " << p.z << "\n";
    }
    std::cout << manifest.str();
    if (!outPath.empty()) {
        std::ofstream f(outPath);
        f << manifest.str();
        std::cout << "wrote " << outPath << "\n";
    }

    world.unregisterSensor(&imu);
    world.unregisterSensor(&hipEnc);
    world.unregisterSensor(&kneeEnc);
    world.unregisterSensor(&hipFt);
    world.unregisterSensor(&foot);
    return 0;
}
