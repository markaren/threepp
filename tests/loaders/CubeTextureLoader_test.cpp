// CubeTextureLoader on a face that will not load.
//
// ImageLoader::load returns std::optional<Image> and hands back nullopt for a
// path that does not exist or will not decode. The cube loader dereferenced that
// unconditionally — `images.emplace_back(*load)` — so one mistyped face path,
// which is the likeliest way to call this wrongly, was undefined behaviour rather
// than a failed load.

#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/CubeTextureLoader.hpp"

#include <filesystem>

using namespace threepp;

namespace {

    std::filesystem::path cubeDir() {
        return std::filesystem::path(DATA_FOLDER) / "textures" / "cube" / "Bridge2";
    }

}// namespace

TEST_CASE("CubeTextureLoader reports a missing face instead of dereferencing nullopt") {

    const auto dir = cubeDir();
    if (!std::filesystem::exists(dir / "posx.jpg")) {
        SUCCEED("cube asset not present; nothing to load");
        return;
    }

    CubeTextureLoader loader;

    // Five real faces and one that does not exist.
    std::shared_ptr<CubeTexture> tex;
    REQUIRE_NOTHROW(tex = loader.load({dir / "posx.jpg", dir / "negx.jpg",
                                       dir / "posy.jpg", dir / "negy.jpg",
                                       dir / "posz.jpg", dir / "no_such_face.jpg"}));
    CHECK(tex == nullptr);
}

TEST_CASE("CubeTextureLoader loads six real faces") {

    const auto dir = cubeDir();
    if (!std::filesystem::exists(dir / "posx.jpg")) {
        SUCCEED("cube asset not present; nothing to load");
        return;
    }

    CubeTextureLoader loader;
    const auto tex = loader.load({dir / "posx.jpg", dir / "negx.jpg",
                                  dir / "posy.jpg", dir / "negy.jpg",
                                  dir / "posz.jpg", dir / "negz.jpg"});

    REQUIRE(tex != nullptr);
    REQUIRE(tex->images().size() == 6);
    CHECK(tex->format == Format::RGB);

    // Three-channel faces are only safe to upload under GL's default 4-byte
    // unpack alignment when the row stride happens to be a multiple of 4.
    const auto width = tex->images()[0].width();
    if ((width * 3u) % 4u != 0u) {
        INFO("face width " << width << " gives a row stride of " << width * 3u);
        CHECK(tex->unpackAlignment == 1);
    }
}
