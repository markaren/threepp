// A cathedral of identical wooden planks, designed from one seed, proven to
// stand by PhysX, and knocked down by a steel wrecking ball if you ask for it.
//
// Nothing is loaded from disk: the planks, their wood, the hall and its light
// are all made here. Cathedral.hpp lays out every plank, Stability.hpp settles
// the design in PhysX and rejects one that moves, Look.hpp makes the wood, the
// floor and the hall.
//
//   mouse   orbit and zoom
//   B       let the wrecking ball go
//
// Flags:
//   --seed N          which cathedral (default 1)
//   --design N        take the seed's Nth design without proving the ones before it
//   --scale S         smaller buildings for quick probes (default 1)
//   --ball            let the wrecking ball go at the start
//   --ball-speed V    its speed at the bottom of its swing, m/s (default 4.5)
//   --ball-radius R   metres (default 0.1, steel)
//   --rope L          metres from the pivot (default 6)
//   --ball-aim DX DZ  the direction it swings in (default from the south-west)
//   --ball-height H   of its centre at the bottom of the swing (default 0.16)
//   --ball-offset O   of its path beside the south-west tower's axis, metres
//   --physics         design and settle on the CPU, report, and exit (no window, no GPU)
//   --collapse T      with --physics: also run T seconds after the ball hits, and report
//   --dump-final f    with --collapse: every plank's final pose, one line each
//   --settle T        seconds of awake settling (default 2; 0 skips the proof)
//   --threads N       PhysX worker threads
//   --hz N, --iters N the step rate (default 960) and solver position iterations (default 48) of the proof
//   --scene-hz N, --scene-iters N  the same once it is proven, for the ball and the collapse (default 960, 24:
//                     at 480 Hz and 12 iterations a woken wall sags on its own, which would not be honest)
//   --explain         say which planks the generator left out, and why
//   --dump file.csv   write the design, one plank a line: element, course, x, z, angle
//   --test stack|ring|wall|tower|roof N   one element of N courses instead of the cathedral
//   --exposure E      tone mapping exposure
//   --ssaa L          headless: supersample with 2^L samples (default 4; 0: MSAA and FXAA)
//
// Headless:
//   --shot file.png [--time T] [--cam X Y Z] [--target X Y Z] [--fov F] [--size WxH]
//   --sequence dir [--fps F] [--time T]
//   --stills dir      the standard set of views for this seed
//   --hero file.png   the hero view alone
//   --probe dir       the collapse: one wide view before, and 0.25, 0.5, 1, 2, 4 and 6 s after the ball hits

#include "threepp/threepp.hpp"

#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/geometries/RoundedBoxGeometry.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"

#include "Cathedral.hpp"
#include "Look.hpp"
#include "Stability.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace threepp;
using namespace planks;

namespace {

    using WallClock = std::chrono::steady_clock;

    double since(WallClock::time_point t0) {
        return std::chrono::duration<double>(WallClock::now() - t0).count();
    }

    struct View {
        std::string name;
        Vector3 eye, target;
        float fov = 30;
    };

    // What the building looks like from the floor plan's point of view.
    struct Layout {
        Box3 box;
        Vector3 centre;
        std::vector<Vector3> points;// plank centres, for framing
        float naveX0 = 0, naveX1 = 0, crossingX = 0, apseX = 0, eaves = 0, ridge = 0, zWall = 0;
        Vector3 tower;// the south-west tower's axis, on the floor
        float towerHeight = 0, towerReach = 0, tier2 = 0;
    };

    Layout layoutOf(const Design& d, const Plan& p) {
        Layout l;
        for (const auto& pl : d.planks) {
            l.box.expandByPoint(Vector3(pl.x - LEN / 2, courseY(pl.course) - THICK / 2, pl.z - LEN / 2));
            l.box.expandByPoint(Vector3(pl.x + LEN / 2, courseY(pl.course) + THICK / 2, pl.z + LEN / 2));
        }
        l.box.getCenter(l.centre);
        for (const auto& pl : d.planks) l.points.emplace_back(pl.x, courseY(pl.course), pl.z);
        l.naveX0 = p.naveStart;
        l.naveX1 = p.naveEnd;
        l.crossingX = p.crossingX;
        l.apseX = p.apseX;
        l.eaves = static_cast<float>(p.naveCourses) * THICK;
        l.ridge = static_cast<float>(p.ridgeCourse) * THICK;
        l.zWall = p.zWall;
        l.tower.set(0, 0, p.towerZ);
        l.towerReach = p.towerReach;
        int tower = -1;
        for (size_t i = 0; i < d.elements.size(); ++i)
            if (d.elements[i] == "south-west tower") tower = static_cast<int>(i);
        for (const auto& pl : d.planks)
            if (pl.element == tower) l.towerHeight = std::max(l.towerHeight, courseY(pl.course) + THICK / 2);
        l.tier2 = static_cast<float>(p.naveCourses + 8) * THICK;
        return l;
    }

    // How far back along `from` (a direction away from the target, its height
    // kept) the eye must stand for every point to fall inside the frame, with
    // `margin` of the frame to spare at the edges.
    View fit(const std::string& name, const Vector3& target, Vector3 away, float eyeHeight, float fov, float aspect,
             const std::vector<Vector3>& points, float margin) {
        away.y = 0;
        away.normalize();
        PerspectiveCamera cam(fov, aspect, 0.01f, 200);
        const auto inside = [&](float distance) {
            cam.position.set(target.x + away.x * distance, eyeHeight, target.z + away.z * distance);
            cam.lookAt(target);
            cam.updateMatrixWorld();
            for (auto p : points) {
                p.project(cam);
                if (std::abs(p.x) > 1 - margin || std::abs(p.y) > 1 - margin || p.z > 1) return false;
            }
            return true;
        };
        float lo = 0.3f, hi = 60.f;
        for (int i = 0; i < 40; ++i) {
            const float mid = 0.5f * (lo + hi);
            (inside(mid) ? hi : lo) = mid;
        }
        View v{name, Vector3(target.x + away.x * hi, eyeHeight, target.z + away.z * hi), target, fov};
        return v;
    }

    // A view along `dir` (from the target toward the eye) that holds every point
    // with `margin` of the frame to spare, the points centred in the frame and
    // the eye as close as that allows.
    View frame(const std::string& name, Vector3 target, Vector3 dir, float fov, float aspect,
               const std::vector<Vector3>& points, float margin) {
        dir.normalize();
        PerspectiveCamera cam(fov, aspect, 0.01f, 200);
        const auto place = [&](float distance) {
            cam.position.copy(target).addScaledVector(dir, distance);
            cam.lookAt(target);
            cam.updateMatrixWorld();
        };
        float distance = 5;
        for (int pass = 0; pass < 4; ++pass) {
            float lo = 0.3f, hi = 60.f;
            for (int i = 0; i < 36; ++i) {
                const float mid = 0.5f * (lo + hi);
                place(mid);
                bool inside = true;
                for (auto p : points) {
                    p.project(cam);
                    if (std::abs(p.x) > 1 - margin || std::abs(p.y) > 1 - margin || p.z > 1) {
                        inside = false;
                        break;
                    }
                }
                (inside ? hi : lo) = mid;
            }
            distance = hi;
            // move the target so that what is seen sits in the middle of the frame
            place(distance);
            float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
            for (auto p : points) {
                p.project(cam);
                x0 = std::min(x0, p.x), x1 = std::max(x1, p.x), y0 = std::min(y0, p.y), y1 = std::max(y1, p.y);
            }
            const float halfHeight = std::tan(fov * math::DEG2RAD / 2) * distance;
            Vector3 right, up, forward;
            cam.matrixWorld->extractBasis(right, up, forward);
            target.addScaledVector(right, 0.5f * (x0 + x1) * halfHeight * aspect).addScaledVector(up, 0.5f * (y0 + y1) * halfHeight);
        }
        place(distance);
        return {name, cam.position, target, fov};
    }

    std::vector<Vector3> corners(const Box3& b) {
        std::vector<Vector3> out;
        for (int i = 0; i < 8; ++i)
            out.emplace_back((i & 1) ? b.max().x : b.min().x, (i & 2) ? b.max().y : b.min().y, (i & 4) ? b.max().z : b.min().z);
        return out;
    }

    // Low, from the south-west, about the height of the nave's eaves: the spires
    // stand against the dark above the hall's far wall.
    View heroView(const Layout& l, float aspect) {
        const Vector3 target(l.centre.x - 0.25f, l.box.max().y * 0.40f, l.centre.z);
        return frame("hero", target, Vector3(-0.50f, -0.03f, 0.87f), 30, aspect, l.points, 0.04f);
    }

    std::vector<View> standardViews(const Layout& l, float aspect) {
        std::vector<View> views;
        views.push_back(heroView(l, aspect));
        // the west front, square on, from a little below the eaves
        {
            Box3 front;
            front.expandByPoint(Vector3(-0.3f, 0, -(l.tower.z + l.towerReach)));
            front.expandByPoint(Vector3(0.3f, l.towerHeight, l.tower.z + l.towerReach));
            const Vector3 target(0, l.towerHeight * 0.46f, 0);
            std::vector<Vector3> points;
            for (const auto& p : l.points)
                if (p.x < 0.35f) points.push_back(p);
            views.push_back(frame("west_front", target, Vector3(-1, -0.04f, 0.05f), 30, aspect, points, 0.04f));
        }
        // down the nave from just above the floor, by the west front
        views.push_back({"nave_low", {l.naveX0 + 0.10f, 0.045f, 0.f}, {l.crossingX, l.eaves * 0.62f, 0.f}, 64});
        // the south-west tower's upper tiers, close enough to count the planks
        {
            const Vector3 axis = l.tower;
            const float y = l.tier2 + 0.12f;
            const Vector3 out = Vector3(-0.62f, 0.f, 0.78f).normalize();
            views.push_back({"tower_close", Vector3(axis).addScaledVector(out, 0.95f).add(Vector3(0, y - 0.1f, 0)), Vector3(axis.x, y + 0.02f, axis.z), 34});
        }
        // along the nave's ridge from beside the crossing tower, to the west gable between the spires
        views.push_back({"roof_ridge", {l.crossingX - 0.37f, l.ridge + 0.28f, 0.03f}, {0.f, l.ridge - 0.09f, 0.f}, 40});
        // from high up, three quarters from the south-east
        // from high up, three quarters from the south-east, looking down at about 38 degrees
        views.push_back(frame("aerial", l.centre, Vector3(0.55f * 0.79f, 0.62f, 0.83f * 0.79f), 30, aspect, l.points, 0.05f));
        return views;
    }

    // The wrecking ball: a pendulum whose lowest point is beside the south-west
    // tower's axis, at the height of its first courses.
    struct Swing {
        Vector3 pivot, start, bottom;
        float radius = 0.1f;
        float rope = 6.f;
    };

    Swing swingFor(const Layout& l, float speed, float radius, float rope, Vector3 aim, float height, float offset) {
        Swing s;
        s.radius = radius;
        s.rope = rope;
        aim.y = 0;
        aim.normalize();
        const Vector3 side(-aim.z, 0, aim.x);
        s.bottom.set(l.tower.x + side.x * offset, height, l.tower.z + side.z * offset);
        s.pivot.set(s.bottom.x, height + rope, s.bottom.z);
        // let go from where it falls far enough to pass the bottom at `speed`
        const float drop = speed * speed / (2 * 9.81f);
        const float angle = std::acos(std::clamp(1 - drop / rope, -1.f, 1.f));
        s.start = Vector3(s.pivot).addScaledVector(aim, -rope * std::sin(angle));
        s.start.y = s.pivot.y - rope * std::cos(angle);
        return s;
    }

    // When the swinging ball first reaches the tower: the pendulum on its own,
    // stepped finely, until its centre is a radius from the tower's outside.
    float predictHit(const Swing& s, const Layout& l) {
        const float g = 9.81f;
        Vector3 out = Vector3(s.start).sub(s.pivot);
        float theta = std::atan2(std::sqrt(out.x * out.x + out.z * out.z), -out.y), omega = 0, t = 0;
        Vector3 back = out;
        back.y = 0;
        back.normalize();
        const float h = 1e-4f;
        for (int i = 0; i < 200000; ++i) {
            const Vector3 at = Vector3(s.pivot).addScaledVector(back, s.rope * std::sin(theta)).add(Vector3(0, -s.rope * std::cos(theta), 0));
            const float dx = at.x - l.tower.x, dz = at.z - l.tower.z;
            if (std::sqrt(dx * dx + dz * dz) <= l.towerReach + s.radius) return t;
            omega += -g / s.rope * std::sin(theta) * h;
            theta += omega * h;
            t += h;
        }
        return t;
    }

    // Wide, from the south-south-west, with room for the ball swinging in.
    View probeView(const Layout& l, const Swing& s, float aspect) {
        std::vector<Vector3> points = l.points;
        Vector3 in = Vector3(s.start).sub(s.bottom);
        in.y = 0;
        in.normalize();
        points.push_back(Vector3(s.bottom).addScaledVector(in, 1.1f).add(Vector3(0, 0.15f, 0)));
        const Vector3 target(l.centre.x - 0.3f, l.box.max().y * 0.38f, l.centre.z + 0.15f);
        return frame("collapse", target, Vector3(-0.50f, 0.02f, 0.87f), 32, aspect, points, 0.04f);
    }

    // Step timing for the report.
    struct StepStats {
        double asleepMs = 0;
        int asleepSteps = 0;
        double peakMs = 0;
        int peakAwake = 0;
        float peakTime = 0;
        int maxAwake = 0;
        int steps = 0;

        void add(double ms, int awake, float t, bool beforeHit) {
            ++steps;
            if (beforeHit) {
                asleepMs += ms;
                ++asleepSteps;
            } else if (ms > peakMs) {
                peakMs = ms;
                peakAwake = awake;
                peakTime = t;
            }
            maxAwake = std::max(maxAwake, awake);
        }

        void print(float hitAt) const {
            std::printf("physics: %d steps; before the hit %.3f ms/step; collapse peak %.2f ms/step at %.2f s after the hit with %d planks awake (most awake at once: %d)\n",
                        steps, asleepSteps ? asleepMs / asleepSteps : 0.0, peakMs, peakTime - hitAt, peakAwake, maxAwake);
        }
    };

}// namespace

int run(int argc, char** argv);

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
}

int run(int argc, char** argv) {

    std::uint64_t seed = 1;
    int firstDesign = 0;
    float scale = 1;
    std::string shotPath, sequenceDir, stillsDir, probeDir, heroPath;
    float shotTime = 0;
    int width = 1920, height = 1080, fps = 30;
    std::optional<Vector3> camArg, targetArg;
    std::optional<float> fovArg;
    bool ballOn = false, physicsOnly = false;
    float ballSpeed = 4.5f, ballRadius = 0.1f, ropeLength = 6.f, ballHeight = 0.16f, ballOffset = 0.f;
    Vector3 ballAim(0.80f, 0, -0.60f);
    float settleSeconds = 2.f, collapseSeconds = 0;
    float exposure = 1.18f;
    int ssaa = 4;
    std::string test;
    int testCourses = 100;
    int hz = 960, sceneHz = 960, sceneIters = 24;
    bool explain = false;
    std::string dumpPath, dumpFinalPath;
    unsigned threads = std::max(2u, std::thread::hardware_concurrency() > 4 ? std::thread::hardware_concurrency() - 4 : 2u);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--seed" && more) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--design" && more) firstDesign = std::max(0, std::atoi(argv[++i]));
        else if (a == "--scale" && more) scale = std::stof(argv[++i]);
        else if (a == "--shot" && more) shotPath = argv[++i];
        else if (a == "--hero" && more) heroPath = argv[++i];
        else if (a == "--sequence" && more) sequenceDir = argv[++i];
        else if (a == "--stills" && more) stillsDir = argv[++i];
        else if (a == "--probe" && more) probeDir = argv[++i];
        else if (a == "--time" && more) shotTime = std::stof(argv[++i]);
        else if (a == "--fps" && more) fps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--size" && more) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if (a == "--fov" && more) fovArg = std::stof(argv[++i]);
        else if (a == "--exposure" && more) exposure = std::stof(argv[++i]);
        else if (a == "--ssaa" && more) ssaa = std::clamp(std::atoi(argv[++i]), 0, 5);
        else if (a == "--ball") ballOn = true;
        else if (a == "--ball-speed" && more) ballSpeed = std::stof(argv[++i]);
        else if (a == "--ball-radius" && more) ballRadius = std::stof(argv[++i]);
        else if (a == "--rope" && more) ropeLength = std::stof(argv[++i]);
        else if (a == "--ball-height" && more) ballHeight = std::stof(argv[++i]);
        else if (a == "--ball-offset" && more) ballOffset = std::stof(argv[++i]);
        else if (a == "--ball-aim" && i + 2 < argc) {
            ballAim.set(std::stof(argv[i + 1]), 0, std::stof(argv[i + 2]));
            i += 2;
        }
        else if (a == "--physics") physicsOnly = true;
        else if (a == "--pgs") tuning.tgs = false;
        else if (a == "--hz" && more) hz = std::max(30, std::atoi(argv[++i]));
        else if (a == "--scene-hz" && more) sceneHz = std::max(30, std::atoi(argv[++i]));
        else if (a == "--scene-iters" && more) sceneIters = std::max(1, std::atoi(argv[++i]));
        else if (a == "--nospec") tuning.speculative = false;
        else if (a == "--nopcm") tuning.pcm = false;
        else if (a == "--explain") explain = true;
        else if (a == "--dump" && more) dumpPath = argv[++i];
        else if (a == "--dump-final" && more) dumpFinalPath = argv[++i];
        else if (a == "--nostab") tuning.stabilization = false;
        else if (a == "--tol" && more) tuning.toleranceLength = std::stof(argv[++i]);
        else if (a == "--offset" && more) tuning.contactOffset = std::stof(argv[++i]);
        else if (a == "--iters" && more) tuning.positionIterations = std::atoi(argv[++i]);
        else if (a == "--test" && i + 2 < argc) {
            test = argv[++i];
            testCourses = std::atoi(argv[++i]);
        }
        else if (a == "--collapse" && more) collapseSeconds = std::stof(argv[++i]);
        else if (a == "--settle" && more) settleSeconds = std::stof(argv[++i]);
        else if (a == "--threads" && more) threads = static_cast<unsigned>(std::max(0, std::atoi(argv[++i])));
        else if ((a == "--cam" || a == "--target") && i + 3 < argc) {
            const Vector3 v(std::stof(argv[i + 1]), std::stof(argv[i + 2]), std::stof(argv[i + 3]));
            (a == "--cam" ? camArg : targetArg) = v;
            i += 3;
        }
    }
    if (!probeDir.empty()) ballOn = true;
    if (physicsOnly && collapseSeconds > 0) ballOn = true;
    const bool headless = !shotPath.empty() || !sequenceDir.empty() || !stillsDir.empty() || !probeDir.empty() || !heroPath.empty();
    const float aspect = static_cast<float>(width) / static_cast<float>(height);

    // ------------------------------------------------------------ design + proof

    const float dt = 1.f / static_cast<float>(hz);
    Design design;
    Plan plan{};
    std::unique_ptr<Physics> physics;
    for (int attempt = firstDesign; attempt < firstDesign + 3; ++attempt) {
        const std::uint64_t variant = seed + static_cast<std::uint64_t>(attempt) * 1000003ull;
        const auto t0 = WallClock::now();
        design = test.empty() ? buildCathedral(variant, scale, &plan, explain) : buildTest(test, testCourses);
        const double generation = since(t0);
        if (!dumpPath.empty()) {
            std::FILE* f = std::fopen(dumpPath.c_str(), "w");
            for (const auto& p : design.planks) std::fprintf(f, "%d,%d,%.6f,%.6f,%.6f\n", p.element, p.course, p.x, p.z, p.angle);
            std::fclose(f);
        }

        std::map<int, int> perElement;
        for (const auto& p : design.planks) ++perElement[p.element];
        std::printf("seed %llu (design %d): %zu planks, generated in %.1f ms (left out: %d overlapping, %d unsupported)\n",
                    static_cast<unsigned long long>(seed), attempt, design.planks.size(), generation * 1000, design.rejectedOverlap, design.rejectedSupport);
        if (test.empty())
            std::printf("  nave %.2f m wide, %d bays of %.2f m, eaves at %d courses, ridge at %d, roofs at %.0f deg; %s; west towers %d tiers from %d planks%s%s; crossing of %d planks, %s\n",
                        plan.naveInner, plan.bays, plan.bayLength, plan.naveCourses, plan.ridgeCourse, plan.pitch, plan.aisles ? "aisles" : "no aisles",
                        plan.tiers, plan.towerPlanks, plan.galleries ? ", galleries" : "", plan.lantern ? ", lanterns" : ", spires", plan.crossingPlanks, plan.dome ? "dome" : "spire");
        {
            // per element name (the two walls of a pair add up)
            std::map<std::string, std::pair<int, int>> byName;
            for (const auto& [e, n] : perElement) {
                auto& entry = byName[design.elements[e]];
                entry.first += n;
                entry.second += e < static_cast<int>(design.leftOut.size()) ? design.leftOut[e] : 0;
            }
            for (const auto& [name, v] : byName) std::printf("    %-22s %6d  (left out %d)\n", name.c_str(), v.first, v.second);
        }

        const auto t1 = WallClock::now();
        physics.reset();// PhysX allows one foundation at a time
        physics = std::make_unique<Physics>(design, dt, threads);
        std::printf("  physics built in %.2f s (%u threads)\n", since(t1), threads);
        if (settleSeconds <= 0) break;

        const auto r = physics->settle(settleSeconds);
        std::printf("  settled %.1f s awake in %.1f s (%.2f ms/step): largest displacement %.3f mm (%s), %d planks moved over 1 mm\n",
                    settleSeconds, r.seconds, r.msPerStep, r.maxDisplacement * 1000,
                    r.worst >= 0 ? design.elements[design.planks[r.worst].element].c_str() : "-", r.movedOverMm);
        if (r.worst >= 0) {
            const auto& w = design.planks[r.worst];
            const auto q = physics->pose(r.worst).p;
            std::printf("  worst plank: course %d at (%.3f, %.3f), moved (%.3f, %.3f, %.3f) mm\n", w.course, w.x, w.z, (q.x - w.x) * 1000, (q.y - courseY(w.course)) * 1000, (q.z - w.z) * 1000);
        }
        // where it moved, element by element
        const auto moved = physics->displacements();
        std::map<std::string, std::pair<float, int>> worstOf;
        for (size_t i = 0; i < moved.size(); ++i) {
            auto& w = worstOf[design.elements[design.planks[i].element]];
            if (moved[i] >= w.first) w = {moved[i], design.planks[i].course};
        }
        for (const auto& [name, m] : worstOf) std::printf("    %-22s %8.3f mm (course %d)\n", name.c_str(), m.first * 1000, m.second);
        if (r.maxDisplacement <= 0.001f) break;
        std::printf("  design %d rejected: it moved more than a millimetre\n", attempt);
        if (attempt == firstDesign + 2) std::printf("  no design of this seed stood; keeping the last\n");
    }
    physics->sleepAll();
    physics->time = 0;// the scene's clock starts with the cathedral standing
    physics->setRate(1.f / static_cast<float>(sceneHz), sceneIters);
    const float sceneDt = physics->dt;
    std::fflush(stdout);

    const Layout layout = layoutOf(design, plan);
    const Swing swing = swingFor(layout, ballSpeed, ballRadius, ropeLength, ballAim, ballHeight, ballOffset);

    StepStats stats;
    bool hung = false;
    std::optional<float> hitTime;// when the ball first woke a plank
    const auto stepPhysics = [&] {
        if (ballOn && !hung) {
            physics->hangBall(swing.pivot, swing.start, swing.radius);
            hung = true;
        }
        const auto t0 = WallClock::now();
        physics->step();
        const double ms = since(t0) * 1000;
        const bool before = !hitTime;
        const int awake = (stats.steps % 4 == 0 || !before) ? physics->awake() : 0;
        if (ballOn && !hitTime && awake > 0) {
            hitTime = physics->time;
            std::printf("the ball hit at t = %.3f s\n", *hitTime);
        }
        stats.add(ms, awake, physics->time, before);
    };

    if (physicsOnly) {
        if (collapseSeconds > 0) {
            const auto t0 = WallClock::now();
            while (!hitTime && physics->time < 10.f) stepPhysics();
            const float until = (hitTime ? *hitTime : physics->time) + collapseSeconds;
            float report = hitTime ? *hitTime + 1.f : until;
            while (physics->time < until) {
                stepPhysics();
                if (physics->time >= report) {
                    std::printf("  %.1f s after the hit: %d planks moved more than 5 cm, %d awake (%.0f s so far)\n",
                                physics->time - *hitTime, physics->movedMoreThan(0.05f), physics->awake(), since(t0));
                    report += 1.f;
                }
            }
            stats.print(hitTime ? *hitTime : 0.f);
            std::printf("collapse: %d of %zu planks moved more than 5 cm (%.0f s of CPU for %.1f s)\n",
                        physics->movedMoreThan(0.05f), design.planks.size(), since(t0), physics->time);
            if (!dumpFinalPath.empty()) {
                physics->dump(dumpFinalPath);
                std::printf("wrote %s\n", dumpFinalPath.c_str());
            }
        }
        return 0;
    }

    // ------------------------------------------------------------ the hall

    const bool supersample = headless && ssaa > 0;
    Canvas canvas(Canvas::Parameters()
                          .title("Plank cathedral")
                          .size(width, height)
                          .antialiasing(supersample ? 0 : 4)
                          .headless(headless));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = exposure;
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    Scene scene;
    const Color horizon(0x2a2119);
    scene.background = Color(0x0d0a08);
    scene.fog = Fog(horizon, 5.5f, 22);
    {
        PMREMGenerator pmrem(renderer);
        RoomEnvironment room;
        scene.environment = pmrem.fromScene(room, 0.04f);
        room.dispose();
    }
    scene.environmentIntensity = 0.16f;
    scene.add(backdrop(horizon, Color(0x0e0b09)));

    auto floor = Mesh::create(PlaneGeometry::create(120, 120), floorMaterial());
    floor->rotation.x = -math::PI / 2;
    floor->receiveShadow = true;
    scene.add(floor);

    // the planks: one instanced mesh
    const auto plankGeometry = RoundedBoxGeometry::create(LEN, THICK, WIDE, 2, 0.0007f);
    auto planksMesh = InstancedMesh::create(plankGeometry, woodMaterial(), design.planks.size());
    planksMesh->castShadow = true;
    planksMesh->receiveShadow = true;
    planksMesh->frustumCulled = false;
    {
        Rng tint(seed * 977 + 11);
        for (size_t i = 0; i < design.planks.size(); ++i) planksMesh->setColorAt(i, plankTint(tint));
        planksMesh->instanceColor()->needsUpdate();
    }
    scene.add(planksMesh);

    const auto syncPlanks = [&] {
        Matrix4 m;
        for (size_t i = 0; i < physics->bodies.size(); ++i) {
            const auto p = physics->pose(i);
            m.compose(Vector3(p.p.x, p.p.y, p.p.z), Quaternion(p.q.x, p.q.y, p.q.z, p.q.w), Vector3(1, 1, 1));
            planksMesh->setMatrixAt(i, m);
        }
        planksMesh->instanceMatrix()->needsUpdate();
    };
    syncPlanks();

    // the wrecking ball, dark worn steel, and its rope
    auto ballMaterial = MeshStandardMaterial::create();
    ballMaterial->color = Color(0x3c3d3f);
    ballMaterial->metalness = 0.85f;
    ballMaterial->roughness = 0.42f;
    auto ballMesh = Mesh::create(SphereGeometry::create(swing.radius, 64, 32), ballMaterial);
    ballMesh->castShadow = true;
    ballMesh->receiveShadow = true;
    ballMesh->visible = false;
    scene.add(ballMesh);
    auto ropeMaterial = MeshStandardMaterial::create();
    ropeMaterial->color = Color(0x2a241d);
    ropeMaterial->roughness = 0.9f;
    auto ropeMesh = Mesh::create(CylinderGeometry::create(0.006f, 0.006f, 1, 8, 1, true), ropeMaterial);
    ropeMesh->castShadow = true;
    ropeMesh->visible = false;
    scene.add(ropeMesh);

    // one low, warm light from the south-west, its shadow fitted to the building
    auto key = DirectionalLight::create(Color(0xffd3a6), 4.6f);
    Object3D keyTarget;
    scene.addRef(keyTarget);
    {
        const Vector3 toLight = Vector3(-0.86f, 0.40f, 0.34f).normalize();
        key->position.copy(layout.centre).addScaledVector(toLight, 14.f);
        key->setTarget(keyTarget);
        key->castShadow = true;
        key->shadow->mapSize.set(4096, 4096);
        key->shadow->bias = -0.00015f;
        key->shadow->normalBias = 0.0012f;
        key->shadow->radius = 2.5f;
        Box3 box = layout.box;
        box.expandByScalar(0.05f);
        if (ballOn) box.expandByPoint(swing.start).expandByScalar(swing.radius);
        fitShadow(*key, keyTarget, box);
    }
    scene.add(key);
    // what the floor throws back into the shadows: warm, from below
    auto bounce = HemisphereLight::create(Color(0x14110e), Color(0x8a6646), 0.55f);
    scene.add(bounce);

    PerspectiveCamera camera(30, aspect, 0.01f, 120);
    const View hero = heroView(layout, aspect);
    camera.position.copy(camArg.value_or(hero.eye));
    OrbitControls controls(camera, canvas);
    controls.target.copy(targetArg.value_or(hero.target));
    controls.update();

    EffectComposer::Options composerOptions;
    composerOptions.samples = supersample ? 0 : 4;
    EffectComposer composer(renderer, composerOptions);
    if (supersample) {
        auto pass = std::make_shared<SSAARenderPass>(scene, camera);
        pass->sampleLevel = ssaa;
        composer.addPass(pass);
    } else {
        composer.addPass(std::make_shared<RenderPass>(scene, camera));
    }
    GTAOParameters aoParameters;
    aoParameters.radius = 0.06f;
    aoParameters.distanceExponent = 1.f;
    aoParameters.thickness = 1.f;
    aoParameters.scale = 1.f;
    aoParameters.samples = 16;
    aoParameters.distanceFallOff = 1.f;
    aoParameters.screenSpaceRadius = false;
    PoissonDenoiseParameters denoiseParameters;
    denoiseParameters.lumaPhi = 10.f;
    denoiseParameters.depthPhi = 2.f;
    denoiseParameters.normalPhi = 3.f;
    denoiseParameters.radius = 4.f;
    denoiseParameters.radiusExponent = 1.f;
    denoiseParameters.rings = 2.f;
    denoiseParameters.samples = 16;
    composer.addPass(std::make_shared<GTAOPass>(scene, camera, width, height, aoParameters, denoiseParameters));
    composer.addPass(std::make_shared<OutputPass>());
    if (!supersample) composer.addPass(std::make_shared<FXAAPass>());

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
        composer.setSize(size.width(), size.height());
    });

    const auto aim = [&](const View& v) {
        camera.position.copy(v.eye);
        camera.fov = fovArg.value_or(v.fov);
        camera.updateProjectionMatrix();
        camera.lookAt(v.target);
    };

    const auto showBall = [&] {
        if (!ballOn) return;
        Vector3 at = swing.start;
        if (physics->ball) {
            const auto p = physics->ball->getGlobalPose();
            at.set(p.p.x, p.p.y, p.p.z);
            ballMesh->quaternion.set(p.q.x, p.q.y, p.q.z, p.q.w);
        }
        ballMesh->visible = true;
        ballMesh->position.copy(at);
        // the rope from the pivot to the top of the ball
        const Vector3 top = Vector3(swing.pivot).sub(at).normalize().multiplyScalar(swing.radius).add(at);
        const float length = swing.pivot.distanceTo(top);
        ropeMesh->visible = true;
        ropeMesh->position.copy(top).add(swing.pivot).multiplyScalar(0.5f);
        ropeMesh->scale.set(1, length, 1);
        ropeMesh->quaternion.setFromUnitVectors(Vector3(0, 1, 0), Vector3(swing.pivot).sub(top).normalize());
    };

    // ------------------------------------------------------------ headless

    if (headless) {
        struct Capture {
            float time;     // seconds on the scene's clock, or after the hit
            bool afterHit;
            View view;
            std::string path;
        };
        std::vector<Capture> captures;
        const std::string tag = "seed" + std::to_string(seed);
        if (!stillsDir.empty()) {
            std::filesystem::create_directories(stillsDir);
            for (const auto& v : standardViews(layout, aspect)) captures.push_back({shotTime, false, v, (std::filesystem::path(stillsDir) / (tag + "_" + v.name + ".png")).string()});
        }
        if (!heroPath.empty()) captures.push_back({shotTime, false, hero, heroPath});
        if (!shotPath.empty()) {
            View v = hero;
            if (camArg) v.eye = *camArg;
            if (targetArg) v.target = *targetArg;
            captures.push_back({shotTime, false, v, shotPath});
        }
        if (!probeDir.empty()) {
            std::filesystem::create_directories(probeDir);
            const View wide = probeView(layout, swing, aspect);
            captures.push_back({-0.25f, true, wide, (std::filesystem::path(probeDir) / (tag + "_collapse_0_before.png")).string()});
            int index = 1;
            for (const float after : {0.25f, 0.5f, 1.f, 2.f, 4.f, 6.f}) {
                char name[64];
                std::snprintf(name, sizeof(name), "%s_collapse_%d_+%.2fs.png", tag.c_str(), index++, after);
                captures.push_back({after, true, wide, (std::filesystem::path(probeDir) / name).string()});
            }
        }
        if (!sequenceDir.empty()) {
            std::filesystem::create_directories(sequenceDir);
            View v = hero;
            if (camArg) v.eye = *camArg;
            if (targetArg) v.target = *targetArg;
            const int frames = std::max(1, static_cast<int>(shotTime * static_cast<float>(fps)));
            for (int f = 0; f <= frames; ++f) {
                char name[32];
                std::snprintf(name, sizeof(name), "frame_%04d.png", f);
                captures.push_back({static_cast<float>(f) / static_cast<float>(fps), false, v, (std::filesystem::path(sequenceDir) / name).string()});
            }
        }
        std::stable_sort(captures.begin(), captures.end(), [](const Capture& a, const Capture& b) {
            if (a.afterHit != b.afterHit) return !a.afterHit;
            return a.time < b.time;
        });

        const float predicted = predictHit(swing, layout);
        if (ballOn) std::printf("the ball should reach the tower at t = %.3f s\n", predicted);
        size_t next = 0;
        int warm = 0;
        bool timed = false;
        float synced = -1;
        canvas.animate([&] {
            if (next >= captures.size()) {
                if (ballOn) stats.print(hitTime.value_or(0.f));
                if (ballOn) std::printf("collapse: %d of %zu planks moved more than 5 cm by t = %.2f s\n", physics->movedMoreThan(0.05f), design.planks.size(), physics->time);
                std::fflush(stdout);
                std::exit(0);
            }
            const auto& c = captures[next];
            if (c.afterHit && c.time >= 0)
                while (!hitTime && physics->time < 10.f) stepPhysics();
            const float when = !c.afterHit ? c.time : c.time < 0 ? predicted + c.time : hitTime.value_or(0.f) + c.time;
            while (physics->time + 0.5f * sceneDt < when) stepPhysics();
            if (synced != physics->time) {
                syncPlanks();
                synced = physics->time;
            }
            showBall();
            aim(c.view);

            const auto t0 = WallClock::now();
            composer.render();
            if (warm == 1 && !timed) {
                unsigned char pixel[4];
                renderer.readPixels(Vector2(0, 0), {1, 1}, Format::RGBA, pixel);
                std::printf("one %dx%d frame (warm, to the pixel readback, %d samples): %.1f ms\n", width, height, supersample ? 1 << ssaa : 1, since(t0) * 1000);
                timed = true;
            }
            if (++warm >= 3) {
                renderer.writeFramebuffer(c.path);
                std::printf("wrote %s (t = %.2f s, %d awake, %d moved over 5 cm)\n", c.path.c_str(), physics->time, physics->awake(), physics->movedMoreThan(0.05f));
                std::fflush(stdout);
                warm = 0;
                ++next;
            }
        });
        return 0;
    }

    // ------------------------------------------------------------ windowed

    canvas.onKeyPressed([&](KeyEvent evt) {
        if (evt.key == Key::B && !ballOn) ballOn = true;
    });

    WallClock::time_point last = WallClock::now();
    float pending = 0;
    canvas.animate([&] {
        const float elapsed = std::min(static_cast<float>(since(last)), 0.1f);
        last = WallClock::now();
        pending += elapsed;
        int steps = 0;
        while (pending >= sceneDt && steps < 16) {
            stepPhysics();
            pending -= sceneDt;
            ++steps;
        }
        if (steps >= 16) pending = 0;
        if (steps > 0) syncPlanks();
        showBall();
        controls.update();
        composer.render();
    });
    return 0;
}
