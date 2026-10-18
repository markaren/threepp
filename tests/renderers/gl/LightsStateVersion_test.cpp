// The lights version is what tells a material its copy of the light state is
// stale. A material copies the state's vectors of per-light uniform maps,
// shadow textures and shadow matrices when it is wired to the lights
// (GLRenderer getProgram), where three.js shares the arrays by reference. So
// anything that changes what a slot points at has to bump the version, not
// only a change in how many lights there are: a light swapped for another of
// the same type within one frame kept materials lit from the removed light,
// and a shadow map rebuilt at a new size left them sampling a freed texture.
//
// No GL here: Lights::setup reads the lights and writes the state.

#include <catch2/catch_test_macros.hpp>

#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/common/Lights.hpp"

using namespace threepp;

TEST_CASE("Lights::setup keeps the version across identical frames") {

    Lights lights;
    auto sun = DirectionalLight::create();
    std::vector<Light*> frame{sun.get()};

    lights.setup(frame);
    const auto v0 = lights.state.version;

    lights.setup(frame);
    REQUIRE(lights.state.version == v0);
}

TEST_CASE("Lights::setup bumps the version when a light is swapped for another of the same type") {

    Lights lights;
    auto a = DirectionalLight::create();
    auto b = DirectionalLight::create();

    std::vector<Light*> withA{a.get()};
    lights.setup(withA);
    const auto v0 = lights.state.version;

    // Same count, same ambient: the counting hash alone saw no change here.
    std::vector<Light*> withB{b.get()};
    lights.setup(withB);
    REQUIRE(lights.state.version != v0);
    REQUIRE(lights.state.directional.size() == 1);
}

TEST_CASE("Lights::setup bumps the version when a light's shadow map is replaced") {

    Lights lights;
    auto sun = DirectionalLight::create();
    sun->castShadow = true;
    sun->shadow->map = RenderTarget::create(4, 4, {});

    std::vector<Light*> frame{sun.get()};
    lights.setup(frame);
    const auto v0 = lights.state.version;
    const auto* first = lights.state.directionalShadowMap.at(0);
    REQUIRE(first != nullptr);

    lights.setup(frame);
    REQUIRE(lights.state.version == v0);

    // What GLShadowMap does on a mapSize change: the target, and with it the
    // texture the receivers sample, is a new object.
    sun->shadow->map = RenderTarget::create(8, 8, {});
    lights.setup(frame);
    REQUIRE(lights.state.version != v0);
    REQUIRE(lights.state.directionalShadowMap.at(0) != first);
}
