// marine::VesselWash against python/examples/rigs/usv_rig.py's Wash, the model
// it is a port of: the same vessel, two frames, and the sources the Python
// class put on a stub sea (sample_height 0), printed to seven digits. A change
// to one of the two that is not made in the other fails here. CPU only: no
// renderer or GPU involved.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/marine/VesselWash.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/DisplacedMesh.hpp"

#include <array>
#include <cmath>
#include <vector>

using namespace threepp;
using marine::VesselWash;

namespace {

    // x0, z0, x1, z1, vx, vz, radius, foam, aeration, turbulence, lane, push, ax, az, patch
    struct Expected {
        float x0, z0, x1, z1, vx, vz, radius, foam, aeration, turbulence, lane, push, ax, az;
        int patch;
    };

    void compare(const std::vector<DisplacedMesh::WakeSource>& got, const std::vector<Expected>& want) {
        REQUIRE(got.size() == want.size());
        const auto near = [](float v) { return Catch::Approx(v).epsilon(2e-5).margin(2e-5); };
        for (size_t i = 0; i < want.size(); ++i) {
            INFO("source " << i);
            const auto& g = got[i];
            const auto& w = want[i];
            CHECK(g.x0 == near(w.x0));
            CHECK(g.z0 == near(w.z0));
            CHECK(g.x1 == near(w.x1));
            CHECK(g.z1 == near(w.z1));
            CHECK(g.vx == near(w.vx));
            CHECK(g.vz == near(w.vz));
            CHECK(g.radius == near(w.radius));
            CHECK(g.foam == near(w.foam));
            CHECK(g.aeration == near(w.aeration));
            CHECK(g.turbulence == near(w.turbulence));
            CHECK(g.lane == near(w.lane));
            CHECK(g.push == near(w.push));
            CHECK(g.ax == near(w.ax));
            CHECK(g.az == near(w.az));
            CHECK(g.patch == w.patch);
        }
    }

    std::shared_ptr<DisplacedMesh> makeSea(uint32_t resolution, uint32_t ripples) {
        auto sea = DisplacedMesh::create(PlaneGeometry::create(10.f, 10.f), MeshBasicMaterial::create());
        sea->wakeField.resolution       = resolution;
        sea->wakeField.rippleResolution = ripples;
        return sea;
    }

    // A level vessel at heading a: her bow along world (cos a, 0, -sin a), her
    // starboard side along (sin a, 0, cos a) (usv_rig's frames).
    struct Vessel {
        float heading;
        Vector3 origin{40.f, 0.f, -25.f};
        Vector3 velocity;

        [[nodiscard]] Vector3 forward() const { return {std::cos(heading), 0.f, -std::sin(heading)}; }
        [[nodiscard]] Vector3 starboard() const { return {std::sin(heading), 0.f, std::cos(heading)}; }
        [[nodiscard]] Vector3 world(float x, float y, float z) const {
            return origin + forward() * x + Vector3(0.f, y, 0.f) + starboard() * z;
        }
        void advance(float surge, float sway, float dt) {
            velocity = forward() * surge + starboard() * sway;
            origin += velocity * dt;
        }
    };

}// namespace

TEST_CASE("VesselWash: a ship on two propellers, with waves", "[marine]") {
    auto sea = makeSea(1024, 512);
    VesselWash::Hull hull;
    hull.length     = 28.f;
    hull.halfLength = 14.f;
    hull.halfBeam   = 4.5f;
    hull.centre     = 0.5f;
    hull.mass       = 355000.f;
    hull.strips     = {{-7.f, 1.6f, 0.3f, 1.5f, 7.f}, {-7.f, -1.6f, 0.3f, 1.5f, 7.f}, {7.f, 1.1f, 0.2f, 1.f, 7.f}, {7.f, -1.1f, 0.2f, 1.f, 7.f}};
    VesselWash wash(*sea, 0, hull);
    CHECK(wash.size() == Catch::Approx(120.f));
    CHECK(wash.rippleSize() == Catch::Approx(120.f));

    Vessel v{0.3f};
    const std::array<std::vector<Expected>, 2> want = {{
            {
                    {17.513f, -16.09232f, 17.513f, -16.09232f, -1.286954f, 0.3981014f, 2.309165f, 0.f, 0.08432358f, 0.6062044f, 1.f, 0.f, 0.f, 0.f, -1},
                    {29.28263f, -19.37838f, 29.28263f, -19.37838f, -0.9780848f, 0.3025571f, 1.3f, 0.f, 0.01718155f, 0.f, 0.f, 0.f, 0.f, 0.f, -1},
                    {16.19899f, -20.34014f, 16.19899f, -20.34014f, -1.286954f, 0.3981014f, 2.309165f, 0.f, 0.08432358f, 0.6062044f, 1.f, 0.f, 0.f, 0.f, -1},
                    {27.98234f, -23.58187f, 27.98234f, -23.58187f, -0.9780848f, 0.3025571f, 1.3f, 0.f, 0.01718155f, 0.f, 0.f, 0.f, 0.f, 0.f, -1},
                    {27.19948f, -21.03684f, 27.19948f, -21.03684f, 0.8686684f, -0.2373081f, 4.5f, 0.f, 0.001707191f, 0.42f, 1.f, 0.f, 0.f, 0.f, -1},
                    {33.882f, -21.42919f, 33.882f, -21.42919f, 0.f, 0.f, 1.5f, 0.f, 0.f, 0.f, 0.f, 1044765.f, 6.687355f, -2.068641f, 0},
                    {32.93633f, -24.48626f, 32.93633f, -24.48626f, 0.f, 0.f, 1.5f, 0.f, 0.f, 0.f, 0.f, 1044765.f, 6.687355f, -2.068641f, 0},
                    {47.10895f, -26.04414f, 47.10895f, -26.04414f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 696510.f, 6.687355f, -2.068641f, 0},
                    {46.4588f, -28.14588f, 46.4588f, -28.14588f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 696510.f, 6.687355f, -2.068641f, 0},
            },
            {
                    {17.513f, -16.09232f, 17.60016f, -16.14891f, -1.286954f, 0.3981014f, 2.309165f, 0.f, 0.08432358f, 0.6062044f, 1.f, 0.f, 0.f, 0.f, -1},
                    {29.28263f, -19.37838f, 29.37914f, -19.40475f, -0.9780848f, 0.3025571f, 1.3f, 0.f, 0.01718155f, 0.f, 0.f, 0.f, 0.f, 0.f, -1},
                    {16.19899f, -20.34014f, 16.30143f, -20.34738f, -1.286954f, 0.3981014f, 2.309165f, 0.f, 0.08432358f, 0.6062044f, 1.f, 0.f, 0.f, 0.f, -1},
                    {27.98234f, -23.58187f, 28.07886f, -23.60823f, -0.9780848f, 0.3025571f, 1.3f, 0.f, 0.01718155f, 0.f, 0.f, 0.f, 0.f, 0.f, -1},
                    {27.19948f, -21.03684f, 27.29599f, -21.06321f, 0.8686684f, -0.2373081f, 4.5f, 0.f, 0.001707191f, 0.42f, 1.f, 0.f, 0.f, 0.f, -1},
                    {33.882f, -21.42919f, 33.97851f, -21.45556f, 0.f, 0.f, 1.5f, 0.f, 0.f, 0.f, 0.f, 1044765.f, 6.687355f, -2.068641f, 0},
                    {32.93633f, -24.48626f, 33.03285f, -24.51263f, 0.f, 0.f, 1.5f, 0.f, 0.f, 0.f, 0.f, 1044765.f, 6.687355f, -2.068641f, 0},
                    {47.10895f, -26.04414f, 47.20547f, -26.07051f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 696510.f, 6.687355f, -2.068641f, 0},
                    {46.4588f, -28.14588f, 46.55532f, -28.17225f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 696510.f, 6.687355f, -2.068641f, 0},
            },
    }};
    const std::array<std::array<float, 4>, 2> patchCentre = {{{18.696981f, -18.406715f, 5.704405f, -14.387640f},
                                                               {18.793500f, -18.433083f, 5.800924f, -14.414008f}}};
    for (int frame = 0; frame < 2; ++frame) {
        INFO("frame " << frame);
        v.advance(6.f, 0.2f, 1.f / 60.f);
        const Vector3 thrust = v.forward() * 60000.f;
        sea->clearWakeSources();
        wash.update(1.f / 60.f, v.origin, v.forward(), v.velocity,
                    {{"prop_s", v.world(-12.f, -1.9f, 2.2f), thrust, 1.f, false},
                     {"prop_p", v.world(-12.f, -1.9f, -2.2f), thrust, 1.f, false}});
        compare(sea->wakeSources, want[frame]);
        const auto& p = sea->wakePatches[0];
        CHECK(p.centerX == Catch::Approx(patchCentre[frame][0]).margin(1e-4));
        CHECK(p.centerZ == Catch::Approx(patchCentre[frame][1]).margin(1e-4));
        CHECK(p.size == Catch::Approx(120.f));
        CHECK(p.eddy == Catch::Approx(10.8f).epsilon(1e-5));
        CHECK(p.rippleCenterX == Catch::Approx(patchCentre[frame][2]).margin(1e-4));
        CHECK(p.rippleCenterZ == Catch::Approx(patchCentre[frame][3]).margin(1e-4));
        CHECK(p.rippleSize == Catch::Approx(120.f));
    }
    const auto& r = wash.report().at("prop_s");
    CHECK(r.race == Catch::Approx(2.55953f).epsilon(1e-5));
    CHECK(r.atSurface == Catch::Approx(1.347121f).epsilon(1e-5));
    CHECK(r.run == Catch::Approx(3.f).epsilon(1e-5));
    CHECK(r.froude == Catch::Approx(0.7420084f).epsilon(1e-5));
}

TEST_CASE("VesselWash: a small catamaran on a jet, going astern, no waves", "[marine]") {
    auto sea = makeSea(512, 0);
    VesselWash::Hull hull;
    hull.length     = 2.f;
    hull.halfLength = 1.f;
    hull.halfBeam   = 0.54f;
    hull.centre     = 0.1f;
    hull.hullOffset = 0.4f;
    hull.mass       = 60.f;
    VesselWash wash(*sea, 3, hull);
    CHECK(wash.size() == Catch::Approx(40.f));
    CHECK(wash.rippleSize() == Catch::Approx(20.f));

    Vessel v{-1.1f};
    const std::array<std::vector<Expected>, 2> want = {{
            {
                    {39.8166f, -25.35483f, 39.8166f, -25.35483f, 0.5553467f, 1.091123f, 0.2890634f, 0.9994499f, 1.f, 0.5509439f, 1.f, 0.f, 0.f, 0.f, -1},
                    {39.58118f, -25.82288f, 39.58118f, -25.82288f, 0.6664161f, 1.309347f, 0.09765625f, 0.f, 0.5321523f, 0.f, 0.f, 0.f, 0.f, 0.f, -1},
                    {40.84485f, -24.22191f, 40.84485f, -24.22191f, -0.09525519f, -0.1871535f, 0.14f, 0.f, 0.f, 0.098f, 1.f, 0.f, 0.f, 0.f, -1},
                    {40.13189f, -23.85903f, 40.13189f, -23.85903f, -0.09525519f, -0.1871535f, 0.14f, 0.f, 0.f, 0.098f, 1.f, 0.f, 0.f, 0.f, -1},
            },
            {
                    {39.8166f, -25.35483f, 39.80379f, -25.37449f, 0.5553467f, 1.091123f, 0.2890634f, 0.9994499f, 1.f, 0.5509439f, 1.f, 0.f, 0.f, 0.f, -1},
                    {39.58118f, -25.82288f, 39.5706f, -25.84368f, 0.6664161f, 1.309347f, 0.09765625f, 0.f, 0.5321523f, 0.f, 0.f, 0.f, 0.f, 0.f, -1},
                    {40.84485f, -24.22191f, 40.83427f, -24.2427f, -0.09525519f, -0.1871535f, 0.14f, 0.f, 0.f, 0.098f, 1.f, 0.f, 0.f, 0.f, -1},
                    {40.13189f, -23.85903f, 40.1213f, -23.87982f, -0.09525519f, -0.1871535f, 0.14f, 0.f, 0.f, 0.098f, 1.f, 0.f, 0.f, 0.f, -1},
            },
    }};
    for (int frame = 0; frame < 2; ++frame) {
        INFO("frame " << frame);
        v.advance(-1.4f, 0.f, 1.f / 60.f);
        sea->clearWakeSources();
        wash.update(1.f / 60.f, v.origin, v.forward(), v.velocity,
                    {{"jet", v.world(-0.9f, -0.15f, 0.f), v.forward() * -150.f, 0.05f, true}});
        compare(sea->wakeSources, want[frame]);
        const auto& p = sea->wakePatches[3];
        CHECK(p.size == Catch::Approx(40.f));
        CHECK(p.eddy == Catch::Approx(0.693752f).epsilon(1e-5));
        CHECK(p.rippleSize == 0.f);
    }
    CHECK(wash.report().at("jet").foam == Catch::Approx(0.9994499f).epsilon(1e-5));

    // Taken away: her patch is off and her producers start afresh.
    wash.clear();
    CHECK(sea->wakePatches[3].size == 0.f);
}

TEST_CASE("VesselWash: a hull's weight from her footprint alone", "[marine]") {
    VesselWash::Hull mono;
    mono.halfLength = 14.f;
    mono.halfBeam   = 4.5f;
    mono.centre     = 0.5f;
    const auto strips = VesselWash::footprintStrips(mono);
    REQUIRE(strips.size() == 2u * VesselWash::kStrips);
    float share = 0.f, moment = 0.f;
    for (const auto& s : strips) {
        share += s.share;
        moment += s.share * s.z;
        CHECK(std::abs(s.x - mono.centre) < mono.halfLength);
        CHECK(std::abs(s.z) < mono.halfBeam);
        CHECK(s.halfLength == Catch::Approx(14.f / VesselWash::kStrips));
    }
    CHECK(share == Catch::Approx(1.f));               // all of her weight
    CHECK(moment == Catch::Approx(0.f).margin(1e-6)); // and none of it to one side
    // The bow tapers, the stern does not: the foremost stretch bears less.
    CHECK(strips.back().share < 0.5f * strips.front().share);

    VesselWash::Hull cat = mono;
    cat.hullOffset = 3.f;
    for (const auto& s : VesselWash::footprintStrips(cat)) CHECK(std::abs(s.z) == Catch::Approx(3.f));

    // A hull that sits deep is a wider load: her short waves go, her weight does not.
    VesselWash::Hull deep = mono;
    deep.draft = 2.5f;
    const auto deepStrips = VesselWash::footprintStrips(deep);
    REQUIRE(deepStrips.size() == strips.size());
    for (size_t i = 0; i < strips.size(); ++i) {
        CHECK(deepStrips[i].radius == Catch::Approx(std::hypot(strips[i].radius, 2.f)));
        CHECK(deepStrips[i].share == Catch::Approx(strips[i].share));
    }
}
