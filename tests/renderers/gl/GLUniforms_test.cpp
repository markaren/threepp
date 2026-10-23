#include <catch2/catch_test_macros.hpp>

#include "threepp/renderers/gl/GLUniforms.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    // Counts the loads it is asked for; the GL program it stands in for is
    // not needed to decide whether a uniform is loaded at all.
    struct RecordingUniform: UniformObject {

        int loads = 0;

        using UniformObject::UniformObject;

        void setValue(const UniformValue&, GLTextures*) override { ++loads; }
    };

}// namespace

TEST_CASE("upload skips a uniform that has no value yet and loads it once it has one") {

    RecordingUniform u("tDiffuse");
    std::vector<UniformObject*> seq{&u};

    UniformMap values;
    values["tDiffuse"] = Uniform();// declared, value still to come

    GLUniforms::upload(seq, values, nullptr);
    REQUIRE(u.loads == 0);

    values["tDiffuse"].setValue(1.f);
    GLUniforms::upload(seq, values, nullptr);
    REQUIRE(u.loads == 1);
}

TEST_CASE("uniformCacheDiffers detects a change in any single component") {

    const std::vector<float> cache{1.f, 2.f, 3.f, 4.f};

    // no change
    REQUIRE_FALSE(uniformCacheDiffers(cache, {1.f, 2.f}));
    REQUIRE_FALSE(uniformCacheDiffers(cache, {1.f, 2.f, 3.f}));
    REQUIRE_FALSE(uniformCacheDiffers(cache, {1.f, 2.f, 3.f, 4.f}));

    // single-component changes, one at a time, for each arity
    REQUIRE(uniformCacheDiffers(cache, {9.f, 2.f}));
    REQUIRE(uniformCacheDiffers(cache, {1.f, 9.f}));

    REQUIRE(uniformCacheDiffers(cache, {9.f, 2.f, 3.f}));
    REQUIRE(uniformCacheDiffers(cache, {1.f, 9.f, 3.f}));
    REQUIRE(uniformCacheDiffers(cache, {1.f, 2.f, 9.f}));

    // regression: a change in y-only or z-only (with x and w unchanged) must
    // still be detected for a 4-component uniform. A previous version of this
    // check used (cache[1] != y && cache[2] != z) instead of ||, which meant a
    // change in only one of y/z was silently dropped and glUniform4f was never
    // called, leaving stale data on the GPU.
    REQUIRE(uniformCacheDiffers(cache, {1.f, 9.f, 3.f, 4.f}));
    REQUIRE(uniformCacheDiffers(cache, {1.f, 2.f, 9.f, 4.f}));
    REQUIRE(uniformCacheDiffers(cache, {1.f, 2.f, 3.f, 9.f}));
    REQUIRE(uniformCacheDiffers(cache, {9.f, 2.f, 3.f, 4.f}));
}

TEST_CASE("parseUniformPath splits a GL uniform name into its containers and leaf") {

    using Containers = std::vector<std::string>;

    SECTION("bare name") {
        const auto p = parseUniformPath("diffuse");
        REQUIRE(p.containers == Containers{});
        REQUIRE(p.leaf == "diffuse");
        REQUIRE_FALSE(p.leafIsArray);
    }

    SECTION("bottom-level array") {
        const auto p = parseUniformPath("directionalShadowMap[0]");
        REQUIRE(p.containers == Containers{});
        REQUIRE(p.leaf == "directionalShadowMap");
        REQUIRE(p.leafIsArray);
    }

    SECTION("array of structs") {
        const auto p = parseUniformPath("spotLights[12].position");
        REQUIRE(p.containers == Containers{"spotLights", "12"});
        REQUIRE(p.leaf == "position");
        REQUIRE_FALSE(p.leafIsArray);
    }

    // regression: the end-of-name test used the part's length instead of
    // its end offset. They agree only for the first part, so an array inside
    // a struct was read as a struct "pts" holding a scalar named "0", and a
    // vector given for it reached the unsupported-variant branch.
    SECTION("array inside a struct") {
        const auto p = parseUniformPath("s.pts[0]");
        REQUIRE(p.containers == Containers{"s"});
        REQUIRE(p.leaf == "pts");
        REQUIRE(p.leafIsArray);
    }

    SECTION("array inside an array of structs") {
        const auto p = parseUniformPath("lights[1].pts[0]");
        REQUIRE(p.containers == Containers{"lights", "1"});
        REQUIRE(p.leaf == "pts");
        REQUIRE(p.leafIsArray);
    }
}
