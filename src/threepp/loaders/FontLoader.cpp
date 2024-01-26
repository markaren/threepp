
#include "threepp/loaders/FontLoader.hpp"

#include "threepp/utils/StringUtils.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>


using namespace threepp;

namespace {

    Font toFont(const nlohmann::json& json) {

        Font data;
        data.familyName = json["familyName"];
        data.resolution = json["resolution"];
        data.lineHeight = json["lineHeight"];
        data.boundingBox = Font::BoundingBox{
                json["boundingBox"]["xMin"].get<float>(),
                json["boundingBox"]["xMax"].get<float>(),
                json["boundingBox"]["yMin"].get<float>(),
                json["boundingBox"]["yMax"].get<float>(),
        };

        auto& glyphs = data.glyphs;
        for (auto& [str_key, value] : json["glyphs"].items()) {
            char key = str_key[0];
            glyphs[key] = Font::Glyph{
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

}// namespace


std::optional<Font> FontLoader::load(const std::filesystem::path& path) {

    if (!std::filesystem::exists(path)) {
        std::cerr << "[FontLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
        return std::nullopt;
    }

    std::ifstream file(path);
    auto json = nlohmann::json::parse(file);

    return toFont(json);
}

std::optional<Font> FontLoader::parse(const std::vector<unsigned char>& data) {
    auto json = nlohmann::json::parse(data);

    return toFont(json);
}
