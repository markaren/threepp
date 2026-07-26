
#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/DDSLoader.hpp"
#include "threepp/textures/Texture.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace threepp;

namespace {

    // Build a minimal DXT1 DDS in memory: 4-byte magic + 124-byte header,
    // followed by `payloadBytes` zeroed payload bytes.
    std::vector<unsigned char> craftDds(uint32_t width, uint32_t height, uint32_t mipCount, size_t payloadBytes) {

        std::vector<unsigned char> data(4 + 124 + payloadBytes, 0);
        auto put32 = [&data](size_t offset, uint32_t v) {
            std::memcpy(data.data() + offset, &v, 4);
        };

        put32(0, 0x20534444u);// "DDS "
        put32(4, 124);        // header dwSize
        put32(12, height);
        put32(16, width);
        put32(28, mipCount);
        put32(76, 32);         // ddspf.dwSize
        put32(80, 0x00000004u);// ddspf.dwFlags = DDPF_FOURCC
        put32(84, 0x31545844u);// "DXT1"

        return data;
    }

}// namespace

TEST_CASE("DDSLoader rejects dimensions whose mip size wraps 32-bit") {

    // 262144 x 262144 makes blocksX * blocksY * blockBytes wrap to 0 in 32-bit
    // arithmetic; the zero "size" then passed the truncation check and the
    // block flip wrote far outside a zero-byte allocation.
    DDSLoader loader;
    CHECK(loader.loadFromMemory(craftDds(262144, 262144, 1, 0)) == nullptr);
}

TEST_CASE("DDSLoader rejects zero dimensions") {

    DDSLoader loader;
    CHECK(loader.loadFromMemory(craftDds(0, 4, 1, 8)) == nullptr);
    CHECK(loader.loadFromMemory(craftDds(4, 0, 1, 8)) == nullptr);
}

TEST_CASE("DDSLoader rejects a payload-less file") {

    // 8x8 DXT1 needs 4 blocks (32 bytes); none are present.
    DDSLoader loader;
    CHECK(loader.loadFromMemory(craftDds(8, 8, 1, 0)) == nullptr);
}

TEST_CASE("DDSLoader still accepts a valid minimal file") {

    // 4x4 DXT1 = one 8-byte block.
    DDSLoader loader;
    CHECK(loader.loadFromMemory(craftDds(4, 4, 1, 8)) != nullptr);
}

TEST_CASE("DDSLoader caps an absurd mip count") {

    // Only mip 0 is present; a hostile count must neither loop 4 billion times
    // nor read past the buffer - it truncates at the data that exists.
    DDSLoader loader;
    CHECK(loader.loadFromMemory(craftDds(4, 4, 0xFFFFFFFFu, 8)) != nullptr);
}
