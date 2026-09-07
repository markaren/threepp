
#include <catch2/catch_test_macros.hpp>

#include "threepp/utils/ZipReader.hpp"
#include "threepp/utils/ZipWriter.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    std::vector<unsigned char> fileBytes(const std::filesystem::path& path) {

        std::ifstream in(path, std::ios::binary);
        return std::vector<unsigned char>{std::istreambuf_iterator<char>(in),
                                          std::istreambuf_iterator<char>()};
    }

}// namespace


TEST_CASE("An archive round-trips its bytes, and says the same thing twice") {

    // Bytes chosen to break a writer that is not honestly binary: a NUL, the
    // DOS end-of-file 0x1A, a CR LF pair, and a local file header signature in
    // the middle of the payload — the last of which is also what the reader's
    // backward EOCD scan has to survive.
    const std::vector<unsigned char> payload{
            0x00, 0x1A, 0x0D, 0x0A, 0xFF, 0x50, 0x4B, 0x03, 0x04, 0x00, 0x7F, 0x80};
    const std::string document = R"({"metadata":{"version":4.5}})";

    const auto dir = std::filesystem::temp_directory_path() / "threepp-zipwriter-test";
    std::filesystem::create_directories(dir);

    const auto first = dir / "first.tpz";
    {
        ZipWriter writer;
        writer.add("scene.json", document);
        writer.add("buffers/abcdef.bin", payload);
        writer.add("images/abcdef-image.png", std::vector<unsigned char>{1, 2, 3});
        writer.writeTo(first);
    }

    ZipReader reader(first);
    REQUIRE(reader.names().size() == 3);
    CHECK(reader.read("buffers/abcdef.bin") == payload);

    // The order entries were added in must not reach the file, and neither must
    // the clock: the exporter advertises byte-identical documents and autosaves
    // diff on that.
    const auto second = dir / "second.tpz";
    {
        ZipWriter writer;
        writer.add("images/abcdef-image.png", std::vector<unsigned char>{1, 2, 3});
        writer.add("buffers/abcdef.bin", payload);
        writer.add("scene.json", document);
        writer.writeTo(second);
    }

    CHECK(fileBytes(first) == fileBytes(second));

    // THE LOCAL FILE HEADER MUST NOT LIE. General purpose bit 3 (0x08) says the
    // crc and the two sizes trail the data instead, and on an entry written that
    // way all three are ZERO in the header — which is how 153 of the 154 entries
    // in the reference SOG archive read as empty files to a reader that trusts
    // them. Nothing threepp writes may be that archive: flags word at offset 6,
    // uncompressed size at 22, both of the first entry (scene.json, which the
    // writer's fixed order puts first).
    const auto bytes = fileBytes(first);
    REQUIRE(bytes.size() > 30);
    const unsigned flags = bytes[6] | (bytes[7] << 8);
    const unsigned size = bytes[22] | (bytes[23] << 8) | (bytes[24] << 16) | (bytes[25] << 24);
    CHECK(flags == 0);
    CHECK(size == document.size());
}
