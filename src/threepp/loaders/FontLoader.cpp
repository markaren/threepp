
#include "threepp/loaders/FontLoader.hpp"

#include "threepp/utils/StringUtils.hpp"

#include "nlohmann/json.hpp"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

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
                    .x_min = value["x_min"].get<float>(),
                    .x_max = value["x_max"].get<float>(),
                    .ha = value["ha"]};
            if (value.contains("o")) {
                std::string o = value["o"].get<std::string>();
                glyphs[key].o = utils::split(o, ' ');
            }
        }

        return data;
    }

    std::optional<Font> loadFromTTF(const std::filesystem::path& ttfFile) {

        std::ifstream file(ttfFile, std::ios::binary);
        if (!file.is_open()) {
            return std::nullopt;
        }

        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read the file contents into a vector
        std::vector<char> fontData(fileSize);
        file.read(fontData.data(), fileSize);
        file.close();

        stbtt_fontinfo info;
        if (!stbtt_InitFont(&info, reinterpret_cast<const unsigned char*>(fontData.data()), 0)) {
            return std::nullopt;
        }

        Font font;
        font.familyName = ttfFile.stem().string();
        font.resolution = ttUSHORT(info.data + info.head + 18);
        stbtt_GetFontVMetrics(&info, &font.lineHeight, nullptr, nullptr);

        int width, height, xOffset, yOffset;
        float scale = stbtt_ScaleForPixelHeight(&info, 16);

        stbtt_vertex* vertices;
        for (int ch = 32; ch < 128; ++ch) {
            Font::Glyph glyph;

            int glyphIndex = stbtt_FindGlyphIndex(&info, ch);
            if (glyphIndex == 0) continue;

            stbtt_GetGlyphHMetrics(&info, glyphIndex, &glyph.ha, nullptr);
            stbtt_GetGlyphBitmapBox(&info, glyphIndex, scale, scale, &xOffset, &yOffset, &width, &height);
            glyph.x_min = static_cast<float>(xOffset);
            glyph.x_max = static_cast<float>(xOffset + width);

            int numVertices = stbtt_GetGlyphShape(&info, glyphIndex, &vertices);

            for (int j = 0; j < numVertices; ++j) {
                const auto type = vertices[j].type;
                if (type == STBTT_vcurve) {
                    glyph.o.emplace_back("q");
                    glyph.o.emplace_back(std::to_string(vertices[j].cx));
                    glyph.o.emplace_back(std::to_string(vertices[j].cy));
                    glyph.o.emplace_back(std::to_string(vertices[j - 1].x));
                    glyph.o.emplace_back(std::to_string(vertices[j - 1].y));

                } else if (type == STBTT_vline) {
                    glyph.o.emplace_back("l");
                    glyph.o.emplace_back(std::to_string(vertices[j].x));
                    glyph.o.emplace_back(std::to_string(vertices[j].y));

                } else if (type == STBTT_vmove) {
                    glyph.o.emplace_back("m");
                    glyph.o.emplace_back(std::to_string(vertices[j].x));
                    glyph.o.emplace_back(std::to_string(vertices[j].y));
                }
            }

            font.glyphs[static_cast<char>(ch)] = glyph;
        }

        STBTT_free(vertices, info);

        return font;
    }


}// namespace


std::optional<Font> FontLoader::load(const std::filesystem::path& path) {

    if (!std::filesystem::exists(path)) {
        std::cerr << "[FontLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
        return std::nullopt;
    }

    const auto ext = path.extension();
    if (path.extension() == ".ttf" || path.extension() == ".TTF") {
        return loadFromTTF(path);
    }

    std::ifstream file(path);
    const auto json = nlohmann::json::parse(file);

    return toFont(json);
}

std::optional<Font> FontLoader::load(const std::vector<unsigned char>& data) {
    const auto json = nlohmann::json::parse(data);

    return toFont(json);
}
