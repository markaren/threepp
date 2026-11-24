// TerrainGenerator::displaceTo sets every vertex's height on the thread pool.
// With no field built each height is the procedural one, which heightAt returns
// for a single point: the two must agree at every vertex, however the pool cuts
// the range into chunks.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/terrain/TerrainGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

using namespace threepp;
using namespace threepp::terrain;

TEST_CASE("displaceTo gives every vertex the height heightAt gives its point") {

    TerrainParams tp;
    TerrainGenerator gen(tp.seed);

    // Whole triangles (displaceTo recomputes the normals), and a count that no
    // chunk size divides evenly. The points lie scattered over the terrain.
    const int n = 3 * 6661;
    std::vector<float> positions(static_cast<size_t>(n) * 3);
    for (int i = 0; i < n; ++i) {
        const float u = static_cast<float>(i % 173) / 172.f - 0.5f;
        const float v = static_cast<float>(i / 173) / static_cast<float>(n / 173) - 0.5f;
        positions[i * 3 + 0] = u * tp.worldSize * 0.9f;
        positions[i * 3 + 1] = 12345.f;// every vertex is overwritten, or the check below sees it
        positions[i * 3 + 2] = v * tp.worldSize * 0.9f;
    }

    auto geo = BufferGeometry::create();
    geo->setAttribute("position", FloatBufferAttribute::create(std::move(positions), 3));

    gen.displaceTo(*geo, tp);

    const auto& a = geo->getAttribute<float>("position")->array();
    float lo = 1e30f, hi = -1e30f;
    int wrong = 0;
    for (int i = 0; i < n; ++i) {
        const float want = gen.heightAt(a[i * 3 + 0], a[i * 3 + 2], tp);
        if (std::abs(a[i * 3 + 1] - want) > 1e-3f) ++wrong;
        lo = std::min(lo, a[i * 3 + 1]);
        hi = std::max(hi, a[i * 3 + 1]);
    }
    CHECK(wrong == 0);
    CHECK(hi - lo > 1.f);// a relief, not one constant
}
