// Instance-id uniqueness under concurrent construction.
//
// Object3D, BufferGeometry, Material and Texture each hand out an id from a
// process-wide counter. Those ids are used as IDENTITY downstream — GLRenderer
// skips re-uploading a material's uniforms when the id matches the last bound
// one, RenderLists sorts draw order by material id, and the Vulkan backend skips
// re-uploading the environment map when the texture id matches. A duplicate id
// therefore does not crash; it silently renders one object with another's state.
//
// It is reachable: loadAsync() runs loaders on a detached std::thread, so a model
// load constructs meshes, geometries, materials and textures concurrently with
// whatever the main thread is doing. With plain `id{counter++}` the
// read-modify-write tears under contention and hands out duplicates.

#include <catch2/catch_test_macros.hpp>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kThreads = 4;
    constexpr int kPerThread = 4000;

    // Construct `kPerThread` objects on each of `kThreads` threads, collect every
    // id, and report how many were handed out more than once.
    template<class MakeOne>
    std::size_t duplicateIdCount(MakeOne makeOne) {

        std::vector<std::vector<unsigned int>> perThread(kThreads);
        std::vector<std::thread> threads;
        threads.reserve(kThreads);

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&perThread, t, &makeOne] {
                auto& out = perThread[t];
                out.reserve(kPerThread);
                for (int i = 0; i < kPerThread; ++i) {
                    out.push_back(makeOne());
                }
            });
        }
        for (auto& th : threads) th.join();

        std::vector<unsigned int> all;
        all.reserve(static_cast<std::size_t>(kThreads) * kPerThread);
        for (const auto& v : perThread) all.insert(all.end(), v.begin(), v.end());

        std::unordered_set<unsigned int> seen;
        seen.reserve(all.size());
        std::size_t duplicates = 0;
        for (const unsigned int id : all) {
            if (!seen.insert(id).second) ++duplicates;
        }
        return duplicates;
    }

}// namespace

TEST_CASE("Object3D ids are unique across concurrent construction") {
    const auto dup = duplicateIdCount([] { return Object3D::create()->id; });
    INFO("duplicate Object3D ids: " << dup);
    CHECK(dup == 0);
}

TEST_CASE("BufferGeometry ids are unique across concurrent construction") {
    const auto dup = duplicateIdCount([] { return BufferGeometry::create()->id; });
    INFO("duplicate BufferGeometry ids: " << dup);
    CHECK(dup == 0);
}

TEST_CASE("Material ids are unique across concurrent construction") {
    const auto dup = duplicateIdCount([] { return MeshBasicMaterial::create()->id; });
    INFO("duplicate Material ids: " << dup);
    CHECK(dup == 0);
}

TEST_CASE("Texture ids are unique across concurrent construction") {
    const auto dup = duplicateIdCount([] { return Texture::create()->id; });
    INFO("duplicate Texture ids: " << dup);
    CHECK(dup == 0);
}

TEST_CASE("Instance ids keep their documented starting point") {

    // BufferGeometry is 1-based (it used pre-increment); the other three are
    // 0-based (post-increment). The atomic rewrite has to preserve that, since
    // a 0 id is a legitimate value the renderers compare against.
    const auto firstGeometry = BufferGeometry::create()->id;
    const auto secondGeometry = BufferGeometry::create()->id;
    CHECK(secondGeometry == firstGeometry + 1);

    const auto firstObject = Object3D::create()->id;
    const auto secondObject = Object3D::create()->id;
    CHECK(secondObject == firstObject + 1);
}
