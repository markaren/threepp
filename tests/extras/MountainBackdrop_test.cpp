#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/terrain/MountainBackdrop.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/textures/Texture.hpp"

#include <cmath>

using namespace threepp;
using namespace threepp::terrain;

namespace {

    // A small ring, so the tests build in milliseconds.
    MountainBackdropOptions small() {
        MountainBackdropOptions o;
        o.innerRadius = 200;
        o.fullRadius = 500;
        o.outerRadius = 1500;
        o.segments = 128;
        o.featureScale = 400;
        o.height = 150;
        o.crestHeight = 40;
        return o;
    }

    const std::vector<float>& positions(const MountainBackdrop& b) {
        return b.mesh()->geometry()->getAttribute<float>("position")->array();
    }

}// namespace

TEST_CASE("MountainBackdrop sits under the ground inside its inner radius") {
    auto o = small();
    o.ground = [](float, float) { return 3.f; };
    o.sink = 4;
    const auto b = MountainBackdrop::create(o);
    CHECK(b->heightAt(0, 0) == -1.f);
    CHECK(b->heightAt(150, -60) == -1.f);
    // and rises out of it further out
    float highest = -1e9f;
    for (int i = 0; i < 64; ++i) {
        const float a = static_cast<float>(i) / 64.f * 6.2831853f;
        highest = std::max(highest, b->heightAt(1200 * std::cos(a), 1200 * std::sin(a)));
    }
    CHECK(highest > 3.f + 50.f);
}

TEST_CASE("A massif raises the ranges round its centre and nowhere else") {
    auto plain = small();
    auto peaked = small();
    peaked.massifs.push_back({Vector2(900, 0), 150, 500});
    const auto a = MountainBackdrop::create(plain);
    const auto b = MountainBackdrop::create(peaked);
    CHECK(b->heightAt(900, 0) > a->heightAt(900, 0) + 100.f);
    CHECK(std::abs(b->heightAt(-900, 0) - a->heightAt(-900, 0)) < 1e-3f);
}

TEST_CASE("The mesh samples heightAt and spaces its rings like its segments") {
    const auto o = small();
    const auto b = MountainBackdrop::create(o);
    const auto& p = positions(*b);
    const size_t S = static_cast<size_t>(o.segments);
    REQUIRE(p.size() % (S * 3) == 0);
    const size_t rings = p.size() / (S * 3);
    REQUIRE(rings > 10);

    for (size_t v : {size_t{0}, S * 5 + 17, (rings / 2) * S + 3, (rings - 1) * S + S - 1}) {
        CHECK(p[3 * v + 1] == b->heightAt(p[3 * v], p[3 * v + 2]));
    }
    for (size_t j : {size_t{2}, rings / 2, rings - 2}) {
        const float r0 = std::hypot(p[3 * j * S], p[3 * j * S + 2]);
        const float r1 = std::hypot(p[3 * (j + 1) * S], p[3 * (j + 1) * S + 2]);
        const float segment = r0 * 6.2831853f / static_cast<float>(S);
        CHECK(std::abs((r1 - r0) / segment - 1.f) < 1e-3f);
    }
    CHECK(std::hypot(p[3 * (rings - 1) * S], p[3 * (rings - 1) * S + 2]) < o.outerRadius);
}

TEST_CASE("Shading picks the material: per-pixel shader or baked maps") {
    auto o = small();
    const auto perPixel = MountainBackdrop::create(o);
    CHECK(perPixel->mesh()->material()->as<ShaderMaterial>() != nullptr);
    CHECK(perPixel->mesh()->geometry()->hasAttribute("aCavity"));
    CHECK_FALSE(perPixel->mesh()->geometry()->hasAttribute("uv"));

    o.shading = MountainBackdropOptions::Shading::Textured;
    o.texelsPerSegment = 2;
    const auto textured = MountainBackdrop::create(o);
    const auto* standard = textured->mesh()->material()->as<MeshStandardMaterial>();
    REQUIRE(standard != nullptr);
    REQUIRE(standard->map);
    REQUIRE(standard->normalMap);
    CHECK(standard->map->image().width() == static_cast<unsigned>(o.segments * 2));
    CHECK(standard->map->colorSpace == ColorSpace::sRGB);
    CHECK(standard->normalMap->colorSpace == ColorSpace::NoColorSpace);

    // one seam column more than segments, its u at 1, on the same ground as the first
    const auto& p = textured->mesh()->geometry()->getAttribute<float>("position")->array();
    const auto& uv = textured->mesh()->geometry()->getAttribute<float>("uv")->array();
    const size_t cols = static_cast<size_t>(o.segments) + 1;
    REQUIRE(p.size() % (cols * 3) == 0);
    CHECK(uv[2 * (cols - 1)] == 1.f);
    CHECK(p[3 * (cols - 1)] == p[0]);
    CHECK(p[3 * (cols - 1) + 1] == p[1]);

    // the normal map leans the surface, it does not overturn it
    const auto& n = standard->normalMap->image().data<unsigned char>();
    size_t up = 0;
    for (size_t i = 2; i < n.size(); i += 4) up += n[i] > 180;
    CHECK(up == n.size() / 4);

    const auto& cav = textured->mesh()->geometry()->getAttribute<float>("aCavity")->array();
    bool hollow = false, crest = false;
    for (float c : cav) {
        CHECK(c >= -1.f);
        CHECK(c <= 1.f);
        hollow = hollow || c > 0.1f;
        crest = crest || c < -0.1f;
    }
    CHECK(hollow);
    CHECK(crest);
}

TEST_CASE("setHaze retunes the per-pixel material") {
    const auto b = MountainBackdrop::create(small());
    b->setHaze(Color(0.1f, 0.2f, 0.3f), Color(1, 0.5f, 0.2f), Vector3(0, 0, 2), 1234.f);
    auto* m = b->mesh()->material()->as<ShaderMaterial>();
    REQUIRE(m != nullptr);
    CHECK(m->uniforms.at("uHaze").value<float>() == 1234.f);
    CHECK(m->uniforms.at("uHorizon").value<Color>() == Color(0.1f, 0.2f, 0.3f));
    CHECK(m->uniforms.at("uSunDirW").value<Vector3>() == Vector3(0, 0, 1));
    CHECK(b->options().hazeDistance == 1234.f);
}
