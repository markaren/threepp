// Cloth_test — the contracts Cloth.hpp states: constraints hold, pins hold,
// colliders keep the sheet out, wind pushes downwind, a replay is bit-identical,
// and ClothMesh welds split vertices and writes the geometry.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/cloth/Cloth.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"

#include <cmath>

using namespace threepp;

namespace {

    constexpr float kDt = 1.f / 60.f;
    constexpr int kNx = 16, kNy = 12;
    constexpr float kW = 1.5f, kH = 1.f;

    // A sheet in the XY plane, (kNx + 1) x (kNy + 1) particles, row 0 on top.
    Cloth makeSheet(const Cloth::Params& params = {}) {
        std::vector<Vector3> p;
        std::vector<unsigned int> idx;
        for (int j = 0; j <= kNy; ++j) {
            for (int i = 0; i <= kNx; ++i) {
                p.emplace_back(kW * (static_cast<float>(i) / kNx - 0.5f), kH * (0.5f - static_cast<float>(j) / kNy), 0.f);
            }
        }
        for (int j = 0; j < kNy; ++j) {
            for (int i = 0; i < kNx; ++i) {
                const unsigned int a = j * (kNx + 1) + i, b = a + 1, c = a + kNx + 1, d = c + 1;
                idx.insert(idx.end(), {a, c, b, b, c, d});
            }
        }
        return {std::move(p), std::move(idx), params};
    }

    void pinTopRow(Cloth& c) {
        for (int i = 0; i <= kNx; ++i) c.pin(i);
    }

    // Largest relative stretch over the grid's horizontal and vertical edges.
    float maxStretch(const Cloth& c) {
        const auto& x = c.positions();
        const float dx = kW / kNx, dy = kH / kNy;
        float worst = 0.f;
        for (int j = 0; j <= kNy; ++j) {
            for (int i = 0; i <= kNx; ++i) {
                const int a = j * (kNx + 1) + i;
                if (i < kNx) worst = std::max(worst, std::abs(x[a].distanceTo(x[a + 1]) / dx - 1.f));
                if (j < kNy) worst = std::max(worst, std::abs(x[a].distanceTo(x[a + kNx + 1]) / dy - 1.f));
            }
        }
        return worst;
    }

    bool allFinite(const Cloth& c) {
        for (const auto& p : c.positions()) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) return false;
        }
        return true;
    }

}// namespace

TEST_CASE("Cloth builds one stretch constraint per edge and one bend pair per interior edge", "[cloth]") {

    const Cloth c = makeSheet();
    const size_t quads = kNx * kNy;
    const size_t edges = kNx * (kNy + 1) + kNy * (kNx + 1) + quads;// horizontal + vertical + diagonals
    const size_t boundary = 2 * (kNx + kNy);

    CHECK(c.particleCount() == static_cast<size_t>((kNx + 1) * (kNy + 1)));
    CHECK(c.triangleCount() == 2 * quads);
    CHECK(c.stretchConstraintCount() == edges);
    CHECK(c.bendConstraintCount() == edges - boundary);
}

TEST_CASE("Cloth at rest with no gravity and no wind does not move", "[cloth]") {

    Cloth::Params p;
    p.gravity.set(0.f, 0.f, 0.f);
    Cloth c = makeSheet(p);
    const auto before = c.positions();

    for (int i = 0; i < 120; ++i) c.step(kDt);

    for (size_t i = 0; i < before.size(); ++i) {
        CHECK(c.positions()[i].x == before[i].x);
        CHECK(c.positions()[i].y == before[i].y);
        CHECK(c.positions()[i].z == before[i].z);
    }
}

TEST_CASE("A hanging cloth keeps its pins and its edge lengths", "[cloth]") {

    Cloth c = makeSheet();
    pinTopRow(c);
    const auto before = c.positions();

    for (int i = 0; i < 300; ++i) c.step(kDt);

    REQUIRE(allFinite(c));
    for (int i = 0; i <= kNx; ++i) {
        CHECK(c.positions()[i].distanceTo(before[i]) == 0.f);
    }
    // Rigid stretch compliance: gravity alone leaves well under 2 %.
    CHECK(maxStretch(c) < 0.02f);
    // The free edge still hangs below the pins by about the sheet's height.
    const float bottom = c.positions()[kNy * (kNx + 1) + kNx / 2].y;
    CHECK(bottom < 0.5f - 0.95f * kH);
}

TEST_CASE("An unpinned cloth falls at g", "[cloth]") {

    Cloth::Params p;
    p.damping = 0.f;
    p.normalDrag = 0.f;
    p.tangentDrag = 0.f;
    Cloth c = makeSheet(p);
    const float y0 = c.positions()[0].y;

    const int steps = 60;
    for (int i = 0; i < steps; ++i) c.step(kDt);

    const float t = steps * kDt;
    const float fallen = y0 - c.positions()[0].y;
    // Semi-implicit Euler over substeps overshoots 0.5 g t^2 by g t h / 2.
    CHECK(std::abs(fallen - 0.5f * 9.81f * t * t) < 0.02f);
    CHECK(maxStretch(c) < 1e-3f);
}

TEST_CASE("A pin given a target carries its particle there", "[cloth]") {

    Cloth c = makeSheet();
    pinTopRow(c);
    const Vector3 target = c.positions()[0] + Vector3{0.1f, 0.2f, -0.3f};

    c.pin(0, target);
    c.step(kDt);

    CHECK(c.positions()[0].distanceTo(target) < 1e-6f);
    CHECK(c.isPinned(0));

    c.unpin(0);
    for (int i = 0; i < 30; ++i) c.step(kDt);
    CHECK(c.positions()[0].distanceTo(target) > 1e-3f);
}

TEST_CASE("Wind carries a hanging cloth downwind, and still air does not", "[cloth]") {

    const auto meanZ = [](const Cloth& c) {
        float z = 0.f;
        for (const auto& p : c.positions()) z += p.z;
        return z / static_cast<float>(c.particleCount());
    };

    Cloth still = makeSheet();
    pinTopRow(still);
    Cloth blown = makeSheet();
    pinTopRow(blown);
    blown.setWind({0.f, 0.f, 6.f});

    for (int i = 0; i < 240; ++i) {
        still.step(kDt);
        blown.step(kDt);
    }

    REQUIRE(allFinite(blown));
    CHECK(std::abs(meanZ(still)) < 1e-4f);
    CHECK(meanZ(blown) > 0.1f);
    CHECK(maxStretch(blown) < 0.05f);
}

TEST_CASE("A wind field replaces the built-in wind", "[cloth]") {

    Cloth c = makeSheet();
    pinTopRow(c);
    c.setWind({0.f, 0.f, 6.f});
    c.setWindField([](const Vector3&, float) { return Vector3{0.f, 0.f, -6.f}; });

    for (int i = 0; i < 240; ++i) c.step(kDt);

    float z = 0.f;
    for (const auto& p : c.positions()) z += p.z;
    CHECK(z / static_cast<float>(c.particleCount()) < -0.1f);
}

TEST_CASE("Colliders keep the cloth out", "[cloth]") {

    // A horizontal sheet dropped on a sphere above a floor.
    std::vector<Vector3> p;
    std::vector<unsigned int> idx;
    const int n = 20;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            p.emplace_back(static_cast<float>(i) / n - 0.5f, 0.6f, static_cast<float>(j) / n - 0.5f);
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const unsigned int a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
            idx.insert(idx.end(), {a, c, b, b, c, d});
        }
    }
    Cloth c(std::move(p), std::move(idx));

    const Vector3 centre{0.05f, 0.25f, -0.03f};
    const float radius = 0.25f;
    c.addSphere(centre, radius);
    c.addPlane({0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});

    for (int i = 0; i < 300; ++i) c.step(kDt);

    REQUIRE(allFinite(c));
    float nearest = 1e9f, lowest = 1e9f;
    for (const auto& q : c.positions()) {
        nearest = std::min(nearest, q.distanceTo(centre));
        lowest = std::min(lowest, q.y);
    }
    CHECK(nearest >= radius);
    CHECK(lowest >= 0.f);
    // It did come down: the sheet drapes over the sphere, its rim near the floor.
    CHECK(nearest < radius + 0.05f);
    CHECK(lowest < 0.3f);
}

TEST_CASE("A replay with the same inputs is bit-identical", "[cloth]") {

    const auto run = [] {
        Cloth c = makeSheet();
        for (int j = 0; j <= kNy; ++j) c.pin(j * (kNx + 1));// the hoist
        c.setWind({7.f, 0.f, 1.f});
        c.addSphere({0.4f, -0.2f, 0.2f}, 0.15f);
        for (int i = 0; i < 200; ++i) c.step(i % 3 == 0 ? 1.f / 50.f : kDt);
        return c.positions();
    };

    const auto a = run();
    const auto b = run();
    REQUIRE(a.size() == b.size());
    bool same = true;
    for (size_t i = 0; i < a.size(); ++i) {
        same = same && a[i].x == b[i].x && a[i].y == b[i].y && a[i].z == b[i].z;
    }
    CHECK(same);
}

TEST_CASE("A long step is clamped and does not blow up", "[cloth]") {

    Cloth c = makeSheet();
    pinTopRow(c);
    c.setWind({0.f, 0.f, 12.f});

    for (int i = 0; i < 60; ++i) c.step(0.5f);

    CHECK(allFinite(c));
    CHECK(c.time() < 60 * c.params.maxStep + 1e-3f);
    CHECK(maxStretch(c) < 0.1f);
}

TEST_CASE("ClothMesh maps a grid one to one and writes the geometry", "[cloth]") {

    auto geometry = PlaneGeometry::create(kW, kH, kNx, kNy);
    ClothMesh mesh(*geometry);

    CHECK(mesh.cloth().particleCount() == static_cast<size_t>((kNx + 1) * (kNy + 1)));

    const auto hoist = mesh.particlesWhere([](const Vector3& p) { return p.x < -0.5f * kW + 1e-4f; });
    CHECK(hoist.size() == static_cast<size_t>(kNy + 1));
    for (const size_t i : hoist) mesh.cloth().pin(i);

    mesh.cloth().setWind({6.f, 0.f, 0.5f});
    for (int i = 0; i < 120; ++i) mesh.update(kDt);

    const auto& pos = geometry->getAttribute<float>("position")->array();
    const auto& nrm = geometry->getAttribute<float>("normal")->array();
    const auto& x = mesh.cloth().positions();
    bool positionsMatch = true, normalsUnit = true, moved = false;
    for (size_t v = 0; v < pos.size() / 3; ++v) {
        const auto& p = x[mesh.particleOf(v)];
        positionsMatch = positionsMatch && pos[v * 3] == p.x && pos[v * 3 + 1] == p.y && pos[v * 3 + 2] == p.z;
        const float len = std::sqrt(nrm[v * 3] * nrm[v * 3] + nrm[v * 3 + 1] * nrm[v * 3 + 1] + nrm[v * 3 + 2] * nrm[v * 3 + 2]);
        normalsUnit = normalsUnit && std::abs(len - 1.f) < 1e-4f;
        moved = moved || std::abs(pos[v * 3 + 2]) > 1e-3f;
    }
    CHECK(positionsMatch);
    CHECK(normalsUnit);
    CHECK(moved);

    REQUIRE(geometry->boundingBox.has_value());
    CHECK(geometry->boundingBox->max().z - geometry->boundingBox->min().z > 1e-3f);
}

TEST_CASE("ClothMesh welds vertices that share a position", "[cloth]") {

    // A box has 24 vertices (split normals) on 8 corners.
    auto geometry = BoxGeometry::create(1.f, 1.f, 1.f);
    ClothMesh mesh(*geometry);

    CHECK(geometry->getAttribute<float>("position")->count() == 24);
    CHECK(mesh.cloth().particleCount() == 8);
    CHECK(mesh.cloth().triangleCount() == 12);
    // A closed box: 12 edges + 6 face diagonals, every edge interior.
    CHECK(mesh.cloth().stretchConstraintCount() == 18);
    CHECK(mesh.cloth().bendConstraintCount() == 18);
}
