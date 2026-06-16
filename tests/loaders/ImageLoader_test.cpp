
#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/textures/Texture.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb/stb_image_write.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

using namespace threepp;

namespace {

    // 2x2 RGB test pattern with four distinct, byte-exact colors:
    //   top-left  red    | top-right  green
    //   bot-left  blue   | bot-right  white
    constexpr std::array<unsigned char, 12> kPattern{
            255, 0, 0, /**/ 0, 255, 0,
            0, 0, 255, /**/ 255, 255, 255};

    std::filesystem::path writePatternPng() {
        auto path = std::filesystem::temp_directory_path() / "threepp_imageloader_test.png";
        REQUIRE(stbi_write_png(path.string().c_str(), 2, 2, 3, kPattern.data(), 2 * 3) != 0);
        return path;
    }

    // pixel accessor into a loaded image
    std::array<unsigned char, 3> rgbAt(const Image& img, unsigned x, unsigned y, int channels) {
        const auto& d = const_cast<Image&>(img).data();
        const size_t o = (static_cast<size_t>(y) * img.width() + x) * channels;
        return {d[o], d[o + 1], d[o + 2]};
    }

}// namespace

TEST_CASE("ImageLoader preserves pixel layout") {

    const auto path = writePatternPng();
    ImageLoader loader;

    SECTION("flipY=false, 3 channels: rows in file order") {
        auto img = loader.load(path, 3, false);
        REQUIRE(img.has_value());
        CHECK(img->width() == 2);
        CHECK(img->height() == 2);
        CHECK(img->channels() == 3);
        CHECK(rgbAt(*img, 0, 0, 3) == std::array<unsigned char, 3>{255, 0, 0});  // red
        CHECK(rgbAt(*img, 1, 0, 3) == std::array<unsigned char, 3>{0, 255, 0});  // green
        CHECK(rgbAt(*img, 0, 1, 3) == std::array<unsigned char, 3>{0, 0, 255});  // blue
        CHECK(rgbAt(*img, 1, 1, 3) == std::array<unsigned char, 3>{255, 255, 255});
    }

    SECTION("flipY=true, 3 channels: rows vertically mirrored") {
        auto img = loader.load(path, 3, true);
        REQUIRE(img.has_value());
        CHECK(rgbAt(*img, 0, 0, 3) == std::array<unsigned char, 3>{0, 0, 255});  // blue now on top
        CHECK(rgbAt(*img, 1, 0, 3) == std::array<unsigned char, 3>{255, 255, 255});
        CHECK(rgbAt(*img, 0, 1, 3) == std::array<unsigned char, 3>{255, 0, 0});
        CHECK(rgbAt(*img, 1, 1, 3) == std::array<unsigned char, 3>{0, 255, 0});
    }

    SECTION("flipY=true, 4 channels: expanded with opaque alpha, mirrored") {
        auto img = loader.load(path, 4, true);
        REQUIRE(img.has_value());
        CHECK(img->channels() == 4);
        CHECK(rgbAt(*img, 0, 0, 4) == std::array<unsigned char, 3>{0, 0, 255});
        CHECK(rgbAt(*img, 1, 1, 4) == std::array<unsigned char, 3>{0, 255, 0});
        const auto& d = img->data();
        CHECK(d[3] == 255);// alpha
    }

    SECTION("loading an HDR with flipY must not affect later image loads") {
        // Regression: RGBELoader used the process-global
        // stbi_set_flip_vertically_on_load, which leaked into every later
        // stbi decode — glTF textures (loaded with flipY=false) came back
        // upside-down once any HDR environment had been loaded.
        const auto hdrPath = std::filesystem::temp_directory_path() / "threepp_imageloader_test.hdr";
        const std::array<float, 6> hdrPattern{1.f, 0.f, 0.f, /* row 1 */ 0.f, 1.f, 0.f};// red over green
        REQUIRE(stbi_write_hdr(hdrPath.string().c_str(), 1, 2, 3, hdrPattern.data()) != 0);

        RGBELoader rgbe;
        auto hdr = rgbe.load(hdrPath, true);
        REQUIRE(hdr != nullptr);
        // flipY=true: green row first
        const auto& hdrData = hdr->image().data<float>();
        CHECK(hdrData[0] == 0.f);
        CHECK(hdrData[1] == 1.f);

        // the glTF pattern: flipY=false must still be in file order
        auto img = loader.load(path, 4, false);
        REQUIRE(img.has_value());
        CHECK(rgbAt(*img, 0, 0, 4) == std::array<unsigned char, 3>{255, 0, 0});// red stays on top
    }

    SECTION("memory overload matches file overload") {
        // read the png bytes back and load via the memory path
        std::ifstream f(path, std::ios::binary);
        REQUIRE(f.is_open());
        std::vector<unsigned char> bytes(std::istreambuf_iterator<char>(f), {});
        REQUIRE(!bytes.empty());

        auto a = loader.load(path, 3, true);
        auto b = loader.load(bytes, 3, true);
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        CHECK(a->data() == b->data());
    }
}
