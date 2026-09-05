// The sonar's renderer-free half: the ray table and the fold from returns to
// echogram. Pinned here without a GPU because these are the parts that
// decide what the image MEANS — which beam a ray belongs to, which bin a
// range lands in, how bright an echo reads — and the one property the
// determinism audit leans on: the fold is a max, so the order the rays come
// back in cannot change a bit of the picture.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/helpers/SonarModel.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    // A surface hit at `range` along ray `ray`, with the normal facing the
    // sensor at incidence `cosInc` unless told otherwise.
    LidarReturn hitAlong(const std::vector<Vector3>& dirs, std::size_t ray, float range,
                         int instanceId, float cosInc = 1.f) {
        LidarReturn r{};
        const Vector3& d = dirs[ray];
        r.position = Vector3(d).multiplyScalar(range);
        // A normal at angle acos(cosInc) to the ray, built in the plane of d and
        // an arbitrary perpendicular.
        Vector3 perp = std::abs(d.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
        perp.cross(d).normalize();
        const float sinInc = std::sqrt(std::max(0.f, 1.f - cosInc * cosInc));
        r.normal = Vector3(d).multiplyScalar(-cosInc).add(perp.multiplyScalar(sinInc));
        r.distance = range;
        r.intensity = 1.f;
        r.hitInstanceId = instanceId;
        r.returnNo = 1;
        r.returnKind = 0;
        return r;
    }

    LidarReturn miss() {
        LidarReturn r{};
        r.hitInstanceId = -1;
        return r;
    }

    SonarModel small() {
        SonarModel m;
        m.horizontalFov = 90.f;
        m.beams = 9;
        m.verticalAperture = 20.f;
        m.verticalSamples = 4;
        m.maxRange = 10.f;
        m.rangeBins = 100;
        m.attenuation = 0.f;
        m.incidenceFloor = 0.f;
        return m;
    }

}// namespace

TEST_CASE("Sonar ray table: beam-major, centred, lidar frame", "[sonar]") {
    const SonarModel m = small();
    const auto dirs = sonarRayDirections(m);
    REQUIRE(dirs.size() == m.rayCount());
    REQUIRE(dirs.size() == 36);

    for (const auto& d : dirs) CHECK_THAT(d.length(), WithinAbs(1.f, 1e-5f));

    // The middle beam looks straight down local -Z: its samples have x == 0
    // and are symmetric in elevation about the horizon.
    const unsigned mid = m.beams / 2;
    for (unsigned s = 0; s < m.verticalSamples; ++s) {
        const Vector3& d = dirs[mid * m.verticalSamples + s];
        CHECK_THAT(d.x, WithinAbs(0.f, 1e-6f));
        CHECK(d.z < 0.f);
    }
    CHECK_THAT(dirs[mid * m.verticalSamples + 0].y, WithinAbs(-dirs[mid * m.verticalSamples + 3].y, 1e-6f));
    // Sample 0 is the lowest elevation.
    CHECK(dirs[mid * m.verticalSamples + 0].y < dirs[mid * m.verticalSamples + 3].y);

    // Beam 0 is the left-most (most negative azimuth, toward local -X); the
    // first beam centre sits half a step inside the swath edge.
    const float halfStep = 0.5f * m.horizontalFov / static_cast<float>(m.beams);
    const float expectedAz = (-0.5f * m.horizontalFov + halfStep) * math::DEG2RAD;
    const Vector3& first = dirs[1];// sample 1, elevation small
    const float az = std::atan2(first.x, -first.z);
    CHECK_THAT(az, WithinAbs(expectedAz, 1e-5f));
    CHECK(first.x < 0.f);
    CHECK(dirs[(m.beams - 1) * m.verticalSamples + 1].x > 0.f);
}

TEST_CASE("Sonar fold: bins by range, strongest echo wins, misses vanish", "[sonar]") {
    const SonarModel m = small();
    const auto dirs = sonarRayDirections(m);
    const Vector3 origin(0, 0, 0);
    SonarReflectivity refl;

    std::vector<LidarReturn> returns(m.rayCount(), miss());
    // Beam 2: two samples land in different bins, two in the same bin with
    // different strengths (via reflectivity).
    refl.set(7, 0.5f);
    returns[2 * 4 + 0] = hitAlong(dirs, 8, 2.55f, 1);   // bin 25
    returns[2 * 4 + 1] = hitAlong(dirs, 9, 7.01f, 1);   // bin 70
    returns[2 * 4 + 2] = hitAlong(dirs, 10, 7.05f, 7);  // bin 70, weaker (0.5)
    returns[2 * 4 + 3] = hitAlong(dirs, 11, 7.09f, 1);  // bin 70, full again

    SonarImage img;
    sonarAccumulate(m, refl, returns, origin, img);
    REQUIRE(img.beams == 9);
    REQUIRE(img.bins == 100);
    REQUIRE(img.intensity.size() == 900);

    CHECK_THAT(img.at(2, 25), WithinAbs(1.f, 1e-6f));
    CHECK_THAT(img.at(2, 70), WithinAbs(1.f, 1e-6f));// max of {1, 0.5, 1}, not a sum
    // Nothing anywhere else.
    const float total = std::accumulate(img.intensity.begin(), img.intensity.end(), 0.f);
    CHECK_THAT(total, WithinAbs(2.f, 1e-6f));
    CHECK_THAT(img.rangeOfBin(70), WithinAbs(7.05f, 1e-5f));
}

TEST_CASE("Sonar fold: incidence floor, attenuation, reflectivity, volume", "[sonar]") {
    SonarModel m = small();
    m.attenuation = 0.1f;
    m.incidenceFloor = 0.35f;
    const auto dirs = sonarRayDirections(m);
    const Vector3 origin(0, 0, 0);
    SonarReflectivity refl;
    refl.defaultValue = 0.8f;
    refl.volume = 0.2f;
    refl.set(3, 0.35f);

    std::vector<LidarReturn> returns(m.rayCount(), miss());
    returns[0] = hitAlong(dirs, 0, 4.f, 99, /*cosInc*/ 1.f);   // default refl, head-on
    returns[4] = hitAlong(dirs, 4, 4.f, 3, /*cosInc*/ 0.f);    // fish, grazing
    returns[8] = hitAlong(dirs, 8, 4.f, -2);                   // volume scatter
    returns[8].returnKind = static_cast<std::int32_t>(LidarReturnKind::VolumeScatter);

    SonarImage img;
    sonarAccumulate(m, refl, returns, origin, img);
    const float att = std::exp(-2.f * 0.1f * 4.f);
    CHECK_THAT(img.at(0, 40), WithinAbs(0.8f * att, 1e-5f));
    CHECK_THAT(img.at(1, 40), WithinAbs(0.35f * 0.35f * att, 1e-5f));
    CHECK_THAT(img.at(2, 40), WithinAbs(0.2f * att, 1e-5f));
}

TEST_CASE("Sonar fold: several returns per ray share the ray's beam", "[sonar]") {
    const SonarModel m = small();
    const auto dirs = sonarRayDirections(m);
    const Vector3 origin(0, 0, 0);
    SonarReflectivity refl;

    // Two returns per ray: a net at 3 m and a fish behind it at 6 m on ray 20
    // (beam 5). Layout: [ray * 2 + slot].
    std::vector<LidarReturn> returns(m.rayCount() * 2, miss());
    returns[20 * 2 + 0] = hitAlong(dirs, 20, 3.f, 1);
    returns[20 * 2 + 1] = hitAlong(dirs, 20, 6.f, 2);
    returns[20 * 2 + 1].returnNo = 2;

    SonarImage img;
    sonarAccumulate(m, refl, returns, origin, img);
    CHECK_THAT(img.at(5, 30), WithinAbs(1.f, 1e-6f));
    CHECK_THAT(img.at(5, 60), WithinAbs(1.f, 1e-6f));
}

TEST_CASE("Sonar fold: out-of-window ranges are dropped", "[sonar]") {
    SonarModel m = small();
    m.minRange = 1.f;
    const auto dirs = sonarRayDirections(m);
    SonarReflectivity refl;
    std::vector<LidarReturn> returns(m.rayCount(), miss());
    returns[0] = hitAlong(dirs, 0, 0.5f, 1); // inside the blind zone
    returns[1] = hitAlong(dirs, 1, 10.5f, 1);// beyond maxRange
    returns[2] = hitAlong(dirs, 2, 10.0f, 1);// exactly maxRange: last bin
    SonarImage img;
    sonarAccumulate(m, refl, returns, Vector3(), img);
    const float total = std::accumulate(img.intensity.begin(), img.intensity.end(), 0.f);
    CHECK_THAT(total, WithinAbs(1.f, 1e-6f));
    CHECK_THAT(img.at(0, 99), WithinAbs(1.f, 1e-6f));
}

TEST_CASE("Sonar fold is order-independent within a beam", "[sonar]") {
    // The property the replay audit relies on: the strongest echo per bin is
    // a max, so whatever order the tracer returns the samples of a beam in,
    // the image is bit-identical. Exercised by permuting the SAMPLES of each
    // beam (the layout fixes which beam a ray belongs to; what scheduling can
    // reorder is the work inside it).
    SonarModel m = small();
    m.attenuation = 0.07f;
    m.incidenceFloor = 0.2f;
    const auto dirs = sonarRayDirections(m);
    SonarReflectivity refl;
    refl.set(2, 0.4f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> range(0.5f, 9.9f), cosv(0.f, 1.f);
    std::vector<LidarReturn> returns(m.rayCount());
    for (std::size_t i = 0; i < returns.size(); ++i) {
        returns[i] = (i % 5 == 0) ? miss() : hitAlong(dirs, i, range(gen), int(i % 3), cosv(gen));
    }
    SonarImage a;
    sonarAccumulate(m, refl, returns, Vector3(), a);

    // Shuffle samples within each beam, swapping the return records but
    // keeping each record's own geometry (the fold reads the record, not the
    // slot).
    std::vector<LidarReturn> shuffled = returns;
    for (unsigned b = 0; b < m.beams; ++b) {
        auto first = shuffled.begin() + static_cast<std::ptrdiff_t>(b * m.verticalSamples);
        std::shuffle(first, first + m.verticalSamples, gen);
    }
    SonarImage bimg;
    sonarAccumulate(m, refl, shuffled, Vector3(), bimg);
    REQUIRE(a.intensity.size() == bimg.intensity.size());
    CHECK(std::equal(a.intensity.begin(), a.intensity.end(), bimg.intensity.begin()));
    CHECK(std::count_if(a.intensity.begin(), a.intensity.end(), [](float v) { return v > 0.f; }) > 20);
}
