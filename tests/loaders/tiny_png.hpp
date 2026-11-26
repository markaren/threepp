// A 1x1 RGB PNG on disk, for loader tests that need a texture file to exist.
//
// Declarations only from stb_image_write: threepp compiles the implementation
// (src/threepp/utils/StbImageWrite.cpp) and every test links threepp; see the
// note in sog_writer.hpp, whose include spelling this follows.

#ifndef THREEPP_TESTS_LOADERS_TINY_PNG_HPP
#define THREEPP_TESTS_LOADERS_TINY_PNG_HPP

#include "external/stb/stb_image_write.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace loadertest {

    inline void appendPngBytes(void* context, void* data, int size) {
        auto* out = static_cast<std::vector<unsigned char>*>(context);
        const auto* bytes = static_cast<const unsigned char*>(data);
        out->insert(out->end(), bytes, bytes + size);
    }

    // Encoded in memory and written through ofstream, which takes any path
    // encoding; stbi_write_png's fopen does not.
    inline std::filesystem::path writeTinyPng(const std::string& name) {
        std::vector<unsigned char> png;
        const unsigned char rgb[3] = {200, 100, 50};
        stbi_write_png_to_func(&appendPngBytes, &png, 1, 1, 3, rgb, 3);
        const auto path = std::filesystem::temp_directory_path() / name;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
        return path;
    }

}// namespace loadertest

#endif// THREEPP_TESTS_LOADERS_TINY_PNG_HPP
