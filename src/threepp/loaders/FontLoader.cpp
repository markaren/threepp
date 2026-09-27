
#include "threepp/loaders/FontLoader.hpp"

#include "threepp/utils/StringUtils.hpp"

#include "nlohmann/json.hpp"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include <fstream>
#include <initializer_list>
#include <iostream>
#include <utility>


using namespace threepp;

namespace {

    Font toFont(const nlohmann::json& json) {

        Font data;
        data.familyName = json["familyName"];
        data.resolution = json["resolution"];
        data.lineHeight = json["lineHeight"];
        data.ascender = json.value("ascender", 0);
        data.descender = json.value("descender", 0);
        data.boundingBox = Font::BoundingBox{
                json["boundingBox"]["xMin"].get<float>(),
                json["boundingBox"]["xMax"].get<float>(),
                json["boundingBox"]["yMin"].get<float>(),
                json["boundingBox"]["yMax"].get<float>(),
        };

        auto& glyphs = data.glyphs;
        for (auto& [str_key, value] : json["glyphs"].items()) {
            const auto codepoints = utils::decodeUtf8(str_key);
            if (codepoints.empty()) continue;
            const char32_t key = codepoints[0];
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
        stbtt_GetFontVMetrics(&info, &font.ascender, &font.descender, nullptr);

        int width, height, xOffset, yOffset;
        float scale = stbtt_ScaleForPixelHeight(&info, 16);

        // Printable ASCII, Latin-1 + Latin Extended-A (accents, ø, µ, °, ×),
        // Greek (λ, θ, ω for maths and physics labels), and the common
        // typographic and mathematical symbols.
        static constexpr std::pair<char32_t, char32_t> ranges[] = {
                {0x0020, 0x007E}, {0x00A0, 0x017F}, {0x0370, 0x03FF}, {0x2010, 0x2027},
                {0x2030, 0x203A}, {0x20AC, 0x20AC}, {0x2190, 0x21FF}, {0x2200, 0x22FF}};

        // The outline format is three.js's typeface one, which lists the END
        // point of a curve before its control point(s):
        //   q x y cpx cpy          b x y cp1x cp1y cp2x cp2y
        const auto put = [](Font::Glyph& g, const char* op, std::initializer_list<int> values) {
            g.o.emplace_back(op);
            for (const int v : values) g.o.emplace_back(std::to_string(v));
        };

        for (const auto& [first, last] : ranges) {
            for (char32_t ch = first; ch <= last; ++ch) {
                const int glyphIndex = stbtt_FindGlyphIndex(&info, static_cast<int>(ch));
                if (glyphIndex == 0) continue;

                Font::Glyph glyph;
                stbtt_GetGlyphHMetrics(&info, glyphIndex, &glyph.ha, nullptr);
                stbtt_GetGlyphBitmapBox(&info, glyphIndex, scale, scale, &xOffset, &yOffset, &width, &height);
                glyph.x_min = static_cast<float>(xOffset);
                glyph.x_max = static_cast<float>(xOffset + width);

                stbtt_vertex* vertices = nullptr;
                const int numVertices = stbtt_GetGlyphShape(&info, glyphIndex, &vertices);
                for (int j = 0; j < numVertices; ++j) {
                    const auto& v = vertices[j];
                    switch (v.type) {
                        case STBTT_vmove: put(glyph, "m", {v.x, v.y}); break;
                        case STBTT_vline: put(glyph, "l", {v.x, v.y}); break;
                        case STBTT_vcurve: put(glyph, "q", {v.x, v.y, v.cx, v.cy}); break;
                        case STBTT_vcubic: put(glyph, "b", {v.x, v.y, v.cx, v.cy, v.cx1, v.cy1}); break;
                        default: break;
                    }
                }
                STBTT_free(vertices, info.userdata);

                font.glyphs[ch] = std::move(glyph);
            }
        }

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
