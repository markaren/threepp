#include <catch2/catch_test_macros.hpp>

#include "threepp/renderers/gl/GLUniforms.hpp"

using namespace threepp::gl;

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
