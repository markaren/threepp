// The whole of a file as bytes, for the loaders that take a file whole.

#ifndef THREEPP_FILE_BYTES_HPP
#define THREEPP_FILE_BYTES_HPP

#include <filesystem>
#include <fstream>
#include <optional>
#include <vector>

namespace threepp::utils {

    // nullopt when the file cannot be opened, sized or read to the end; an
    // empty file is an empty vector.
    inline std::optional<std::vector<unsigned char>> readFileBytes(const std::filesystem::path& path) {

        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) return std::nullopt;

        const std::streamoff end = in.tellg();
        if (end < 0) return std::nullopt;

        std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
        in.seekg(0, std::ios::beg);
        if (end > 0) {

            in.read(reinterpret_cast<char*>(bytes.data()), end);
            if (in.gcount() != end) return std::nullopt;
        }

        return bytes;
    }

}// namespace threepp::utils

#endif//THREEPP_FILE_BYTES_HPP
