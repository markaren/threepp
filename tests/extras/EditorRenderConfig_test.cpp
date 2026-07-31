// The look of a document: the userData round-trip the editor writes on save.
//
// Renderer-free on purpose — this is the half CI can run without a GPU, and it
// is the half that carries the format contract. RenderConfig differs from the
// other editor configs in that it is written as a DIFFERENCE from a baseline,
// so the contract has three parts:
//
//   * encode() emits only what differs from `base`, and nothing at all when
//     nothing does — a document that never touched the Renderer Settings panel
//     saves no render block.
//   * decode() layers the text over `base`, so an absent key means "the
//     editor's default", not "the struct's default" and never "whatever the
//     previously open document left on the renderer".
//   * an unknown key is ignored rather than fatal, so a document written by a
//     newer editor still loads.
//
// capture()/apply() need a live renderer and are exercised by the editor itself.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/RenderConfig.hpp"

#include "threepp/core/Object3D.hpp"

#include <string>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;


TEST_CASE("a config identical to the baseline encodes to nothing", "[render]") {

    RenderConfig base;
    base.renderScale = 0.8f;// the editor's Vulkan default

    REQUIRE(RenderConfig{}.encode().empty());
    REQUIRE(base.encode(base).empty());

    // ...and the difference is what a save writes, not the whole state.
    RenderConfig fogged = base;
    fogged.heightFog = true;
    fogged.fogDensity = 0.05f;
    REQUIRE(fogged.encode(base) == "heightfog=1;fogdensity=0.05");
}

TEST_CASE("an absent key means the baseline, not the struct default", "[render]") {

    RenderConfig base;
    base.renderScale = 0.8f;
    base.toneMapping = ToneMapping::ACESFilmic;
    base.bloom = 0.25f;

    // A document that only asks for bloom keeps the baseline's render scale and
    // tone map — the trap this whole design exists to avoid.
    const auto decoded = RenderConfig::decode("bloom=0.5", base);
    REQUIRE_THAT(decoded.bloom, WithinAbs(0.5f, 1e-6f));
    REQUIRE_THAT(decoded.renderScale, WithinAbs(0.8f, 1e-6f));
    REQUIRE(decoded.toneMapping == ToneMapping::ACESFilmic);
}

TEST_CASE("every field survives the round trip", "[render]") {

    RenderConfig config;
    config.toneMapping = ToneMapping::AgX;
    config.exposure = 1.6f;
    config.shadows = true;
    config.shadowType = ShadowMap::VSM;
    config.renderScale = 0.65f;
    config.upscaler = RenderConfig::Upscaler::Dlss;
    config.gbufferMsaa = 4;
    config.textureAnisotropy = 8.f;
    config.autoExposure = true;
    config.autoExposureSpeed = 3.5f;
    config.whiteBalanceK = 5200.f;
    config.whiteBalanceTint = -0.25f;
    config.sharpen = 0.2f;
    config.ao = false;
    config.probeGI = false;
    config.denoise = false;
    config.restirDI = false;
    config.sunAngularRadius = 0.53f;
    config.envSun = RenderConfig::EnvSun::Off;
    config.fireflyClamp = 12.f;
    config.physicalCamera = true;
    config.physicalLightUnits = true;
    config.aperture = 2.8f;
    config.shutterSeconds = 1.f / 250.f;
    config.iso = 800.f;
    config.exposureCompensation = -1.5f;
    config.depthOfField = true;
    config.focusDistance = 4.25f;
    config.motionBlur = 0.35f;
    config.bloom = 0.15f;
    config.bloomThreshold = 2.f;
    config.heightFog = true;
    config.fogDensity = 0.03f;
    config.fogBaseY = -2.f;
    config.fogFalloff = 120.f;
    config.fogNoise = 0.8f;
    config.fogAnisotropy = 0.4f;
    config.beamDensity = 0.02f;
    config.beamAnisotropy = 0.7f;
    config.clouds = true;
    config.cloudCoverage = 0.6f;
    config.cloudDensity = 1.5f;
    config.cloudBottomY = 800.f;
    config.cloudTopY = 2000.f;
    config.cloudWindX = 3.f;
    config.cloudWindZ = -4.f;
    config.cloudEvolve = 0.5f;
    config.occlusionCulling = true;
    config.autoLod = false;
    config.autoLodError = 1.5f;
    config.normalMapToksvig = false;

    // Encoded against the defaults, decoded against them: the whole struct is
    // in the string, so nothing silently fails to travel.
    REQUIRE(RenderConfig::decode(config.encode()) == config);

    // And the same holds against a non-default baseline, where the string is a
    // partial description of the config.
    RenderConfig base;
    base.renderScale = 0.8f;
    base.bloom = 0.15f;
    base.clouds = true;
    REQUIRE(RenderConfig::decode(config.encode(base), base) == config);
}

TEST_CASE("an unknown key is ignored, and a malformed value keeps the baseline", "[render]") {

    RenderConfig base;
    base.exposure = 1.25f;

    const auto decoded = RenderConfig::decode("newfangled=7;exposure=oops;bloom=0.5", base);
    REQUIRE_THAT(decoded.exposure, WithinAbs(1.25f, 1e-6f));
    REQUIRE_THAT(decoded.bloom, WithinAbs(0.5f, 1e-6f));
}

TEST_CASE("the entry rides on the scene root, and a default look leaves none", "[render]") {

    Object3D scene;
    RenderConfig base;
    base.renderScale = 0.8f;

    // Nothing to say: no entry, so the file has no render block.
    base.write(scene, base);
    REQUIRE(scene.userData.find(RenderConfig::userDataKey) == scene.userData.end());
    REQUIRE_FALSE(RenderConfig::read(scene, base).has_value());

    RenderConfig config = base;
    config.bloom = 0.4f;
    config.renderScale = 0.5f;
    config.write(scene, base);

    const auto read = RenderConfig::read(scene, base);
    REQUIRE(read.has_value());
    REQUIRE(*read == config);

    // Back to the baseline: the stale entry is removed rather than left behind
    // saying something the document no longer means.
    base.write(scene, base);
    REQUIRE(scene.userData.find(RenderConfig::userDataKey) == scene.userData.end());
}
