#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/STLLoader.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    std::filesystem::path writeTempFile(const std::string& name, const std::vector<char>& bytes) {
        auto path = std::filesystem::temp_directory_path() / name;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        out.close();
        return path;
    }

    std::filesystem::path writeTempFile(const std::string& name, const std::string& text) {
        return writeTempFile(name, std::vector<char>(text.begin(), text.end()));
    }

    void appendFloat(std::vector<char>& buf, float v) {
        char tmp[4];
        std::memcpy(tmp, &v, 4);
        buf.insert(buf.end(), tmp, tmp + 4);
    }

}// namespace

TEST_CASE("STLLoader parses ASCII STL without crashing") {

    // A binary parse of this file would read a garbage face count from offset 80
    // and try to allocate/index a huge vector -> crash. It must be detected as ASCII.
    const std::string stl =
            "solid triangle\n"
            "  facet normal 0.0 0.0 1.0\n"
            "    outer loop\n"
            "      vertex 0.0 0.0 0.0\n"
            "      vertex 1.0 0.0 0.0\n"
            "      vertex 0.0 1.0 0.0\n"
            "    endloop\n"
            "  endfacet\n"
            "endsolid triangle\n";

    auto path = writeTempFile("threepp_ascii.stl", stl);

    STLLoader loader;
    auto geometry = loader.load(path);

    REQUIRE(geometry);
    auto* position = geometry->getAttribute<float>("position");
    REQUIRE(position);
    CHECK(position->count() == 3);// one triangle -> three vertices

    std::filesystem::remove(path);
}

TEST_CASE("STLLoader parses binary STL without crashing") {

    // Build a minimal one-triangle binary STL: 80-byte header + uint32 count + one
    // 50-byte face record. This guards the binary path (and the former pre-size +
    // emplace_back double-grow bug).
    std::vector<char> bytes(80, 0);// header

    const uint32_t faces = 1;
    char countBytes[4];
    std::memcpy(countBytes, &faces, 4);
    bytes.insert(bytes.end(), countBytes, countBytes + 4);

    // normal
    appendFloat(bytes, 0.f);
    appendFloat(bytes, 0.f);
    appendFloat(bytes, 1.f);
    // v0, v1, v2
    appendFloat(bytes, 0.f);
    appendFloat(bytes, 0.f);
    appendFloat(bytes, 0.f);
    appendFloat(bytes, 1.f);
    appendFloat(bytes, 0.f);
    appendFloat(bytes, 0.f);
    appendFloat(bytes, 0.f);
    appendFloat(bytes, 1.f);
    appendFloat(bytes, 0.f);
    // attribute byte count
    bytes.push_back(0);
    bytes.push_back(0);

    auto path = writeTempFile("threepp_binary.stl", bytes);

    STLLoader loader;
    auto geometry = loader.load(path);

    REQUIRE(geometry);
    auto* position = geometry->getAttribute<float>("position");
    REQUIRE(position);
    CHECK(position->count() == 3);

    auto* normal = geometry->getAttribute<float>("normal");
    REQUIRE(normal);
    CHECK(normal->count() == 3);

    std::filesystem::remove(path);
}
