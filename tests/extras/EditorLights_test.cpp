
#include <catch2/catch_test_macros.hpp>

#include "threepp/renderers/common/Lights.hpp"

#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"

using namespace threepp;

// The renderer re-uploads light uniforms only when Lights::state.version
// changes. Ambient light contributes no entry to any of the per-type counts —
// it collapses into one summed colour — so a version keyed purely on those
// counts leaves an ambient change invisible until something unrelated forces a
// program switch. That is what these cover.

TEST_CASE("removing an ambient light bumps the light state version", "[editor]") {

    Lights lights;

    auto ambient = AmbientLight::create(0xffffff);
    std::vector<Light*> withAmbient{ambient.get()};
    lights.setup(withAmbient);
    const auto before = lights.state.version;

    std::vector<Light*> none;
    lights.setup(none);

    CHECK(lights.state.version != before);
}

TEST_CASE("recolouring an ambient light bumps the light state version", "[editor]") {

    Lights lights;

    auto ambient = AmbientLight::create(0x404040);
    std::vector<Light*> list{ambient.get()};
    lights.setup(list);
    const auto before = lights.state.version;

    ambient->color = Color(0xff0000);
    lights.setup(list);

    CHECK(lights.state.version != before);
}

TEST_CASE("an unchanged light set keeps its version", "[editor]") {

    // The version drives program rebuilds, so it must not churn every frame.
    Lights lights;

    auto ambient = AmbientLight::create(0x202020);
    auto sun = DirectionalLight::create(0xffffff);
    std::vector<Light*> list{ambient.get(), sun.get()};

    lights.setup(list);
    const auto settled = lights.state.version;

    lights.setup(list);
    lights.setup(list);

    CHECK(lights.state.version == settled);
}
