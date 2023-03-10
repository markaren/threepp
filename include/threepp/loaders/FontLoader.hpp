//

#ifndef THREEPP_FONTLOADER_HPP
#define THREEPP_FONTLOADER_HPP

#include "threepp/extras/core/Font.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

namespace threepp {


    class FontLoader {

    public:
        std::optional<FontData> load(const std::filesystem::path& path);
    };

}// namespace threepp

#endif//THREEPP_FONTLOADER_HPP
