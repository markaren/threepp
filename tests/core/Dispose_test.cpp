// dispose() dispatches "dispose" on every call, as in three.js.
//
// Material, BufferGeometry, RenderTarget and InstancedMesh latched on a
// private flag and fired only the first time. A renderer's teardown disposes
// every material, geometry and render target it knows, so the first dispose
// often came from there, and any later owner (a second renderer, or code that
// disposed an object and used it again) was never told the object was gone.
// Texture had the same latch and lost it in 1292a04f.

#include <catch2/catch_test_macros.hpp>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp;

namespace {

    template<class T>
    int disposeTwice(T& object) {

        int calls = 0;
        auto subscription = object.subscribe("dispose", [&calls](Event&) { ++calls; });

        object.dispose();
        object.dispose();

        return calls;
    }

}// namespace

TEST_CASE("dispose() dispatches on every call") {

    auto material = MeshBasicMaterial::create();
    CHECK(disposeTwice(*material) == 2);

    auto geometry = BoxGeometry::create();
    CHECK(disposeTwice(*geometry) == 2);

    auto texture = Texture::create();
    CHECK(disposeTwice(*texture) == 2);

    auto target = RenderTarget::create(4, 4, RenderTarget::Options{});
    CHECK(disposeTwice(*target) == 2);

    auto instanced = InstancedMesh::create(BoxGeometry::create(), MeshBasicMaterial::create(), 2);
    CHECK(disposeTwice(*instanced) == 2);
}

TEST_CASE("the destructor dispatches after an earlier dispose()") {

    int calls = 0;
    LambdaEventListener listener{[&calls](Event&) { ++calls; }};
    {
        auto material = MeshBasicMaterial::create();
        material->addEventListener("dispose", listener);
        material->dispose();
    }// ~Material dispatches again
    CHECK(calls == 2);
}
