// SpirvReflect against the backend's own embedded shaders. CPU only.

#include <catch2/catch_test_macros.hpp>

#include "threepp/renderers/vulkan/SpirvReflect.hpp"

#include "threepp/renderers/vulkan/shaders/bloom_down.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/cluster_build.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/deferred_shade.comp.spv.h"

using namespace threepp::vulkan;
using Kind = SpirvBinding::Kind;

namespace {

    const SpirvBinding* find(const std::vector<SpirvBinding>& v, uint32_t set, uint32_t binding) {
        for (const auto& b : v) {
            if (b.set == set && b.binding == binding) return &b;
        }
        return nullptr;
    }

}// namespace

TEST_CASE("bloom_down: a sampler read and a writeonly storage image") {
    const auto b = reflectSpirvBindings(kBloomDownCompSpv);
    REQUIRE(b.size() == 2);
    const auto* src = find(b, 0, 0);
    REQUIRE(src);
    CHECK(src->kind == Kind::CombinedSampler);
    CHECK(src->read);
    CHECK_FALSE(src->write);
    const auto* dst = find(b, 0, 1);
    REQUIRE(dst);
    CHECK(dst->kind == Kind::StorageImage);
    CHECK_FALSE(dst->read);
    CHECK(dst->write);
}

TEST_CASE("cluster_build: a uniform block, a read-write and a readonly storage buffer") {
    const auto b = reflectSpirvBindings(kClusterBuildCompSpv);
    const auto* cam = find(b, 0, 0);
    REQUIRE(cam);
    CHECK(cam->kind == Kind::UniformBuffer);
    const auto* grid = find(b, 0, 48);
    REQUIRE(grid);
    CHECK(grid->kind == Kind::StorageBuffer);
    CHECK(grid->write);
    const auto* lights = find(b, 0, 49);
    REQUIRE(lights);
    CHECK(lights->kind == Kind::StorageBuffer);
    CHECK(lights->read);
    CHECK_FALSE(lights->write);
}

TEST_CASE("deferred_shade: only statically used bindings, arrays sized") {
    const auto b = reflectSpirvBindings(kDeferredShadeCompSpv);
    REQUIRE(b.size() > 20);
    for (size_t i = 1; i < b.size(); ++i) {
        CHECK((b[i - 1].set < b[i].set || b[i - 1].binding < b[i].binding));
    }
    const auto* normal = find(b, 0, 3);
    REQUIRE(normal);
    CHECK(normal->kind == Kind::CombinedSampler);
    CHECK(normal->arraySize == 1);
    // The bindless material texture array.
    const auto* tex = find(b, 0, 11);
    REQUIRE(tex);
    CHECK(tex->kind == Kind::CombinedSampler);
    CHECK(tex->arraySize == 2048);
    // The TLAS the shadow and reflection rays traverse.
    bool accel = false;
    for (const auto& x : b) accel = accel || x.kind == Kind::AccelerationStructure;
    CHECK(accel);
}

TEST_CASE("a malformed module reflects to nothing") {
    const uint32_t junk[] = {0xdeadbeef, 1, 2, 3, 4, 5};
    CHECK(reflectSpirvBindings(junk).empty());
    CHECK(reflectSpirvBindings(nullptr, 0).empty());
}
