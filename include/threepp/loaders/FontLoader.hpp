
#ifndef THREEPP_FONTLOADER_HPP
#define THREEPP_FONTLOADER_HPP

#include "threepp/extras/core/Font.hpp"

#include <filesystem>
#include <optional>

namespace threepp {

    class FontLoader {

    public:
        Font defaultFont();
        std::optional<Font> load(const std::filesystem::path& path);
        std::optional<Font> load(const std::vector<unsigned char>& data);
    };
}// namespace threepp

#endif//THREEPP_FONTLOADER_HPP
