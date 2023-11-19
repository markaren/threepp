//

#ifndef THREEPP_FONTLOADER_HPP
#define THREEPP_FONTLOADER_HPP

#include "threepp/extras/core/Font.hpp"

#if __has_include(<nlohmann/json.hpp>)
#include "threepp/utils/StringUtils.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#endif

#include <filesystem>
#include <iostream>
#include <optional>


namespace threepp {

    class FontLoader {

    public:
        std::optional<Font> load(const std::filesystem::path& path);

    private:
#if __has_include(<nlohmann/json.hpp>)
        static FontData toFontData(const nlohmann::json& json) {

            FontData data;
            data.familyName = json["familyName"];
            data.resolution = json["resolution"];
            data.boundingBox = FontData::BoundingBox{
                    json["boundingBox"]["xMin"].get<float>(),
                    json["boundingBox"]["xMax"].get<float>(),
                    json["boundingBox"]["yMin"].get<float>(),
                    json["boundingBox"]["yMax"].get<float>(),
            };

            auto& glyphs = data.glyphs;
            for (auto& [str_key, value] : json["glyphs"].items()) {
                char key = str_key[0];
                glyphs[key] = FontData::Glyph{
                        value["x_min"].get<float>(),
                        value["x_max"].get<float>(),
                        value["ha"]};
                if (value.contains("o")) {
                    std::string o = value["o"].get<std::string>();
                    glyphs[key].o = utils::split(o, ' ');
                }
            }

            return data;
        }
#endif
    };

}// namespace threepp

#if __has_include(<nlohmann/json.hpp>)

std::optional<threepp::Font> threepp::FontLoader::load(const std::filesystem::path& path) {

    if (!std::filesystem::exists(path)) {
        std::cerr << "[FontLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
        return std::nullopt;
    }

    std::ifstream file(path);
    auto json = nlohmann::json::parse(file);

    return Font(toFontData(json));
}

#else

std::optional<threepp::Font> threepp::FontLoader::load(const std::filesystem::path& path) {

    std::cerr << "[FontLoader] JSON library not found. No loading will occour." << std::endl;

    return std::nullopt;
}

#endif

#endif//THREEPP_FONTLOADER_HPP
