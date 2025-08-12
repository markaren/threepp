
#include "threepp/objects/TextNode.hpp"

#include "external/stb/stb_truetype.h"
#include "threepp/utils/ImageUtils.hpp"

#include <fstream>

using namespace threepp;

struct TextNode::Impl {

    Impl(Sprite* that, const std::filesystem::path& fontFile): that(that) {
        if (!std::filesystem::exists(fontFile)) {
            throw std::runtime_error("Font file not found: " + fontFile.string());
        }

        std::ifstream file(fontFile, std::ios::binary);
        fontBuffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});

        stbtt_InitFont(&font_, fontBuffer.data(), stbtt_GetFontOffsetForIndex(fontBuffer.data(), 0));

        that->material()->as<MaterialWithMap>()->map = Texture::create(createText("empty"));

        that->center.set(0, 1);

        that->material()->as<MaterialWithMap>()->map->offset.set(0.5, 0.5);
        that->material()->as<MaterialWithMap>()->map->needsUpdate();

        that->material()->as<SpriteMaterial>()->opacity = 1;
    }

    void setText(const std::string& text, float worldScale) {
        auto material = that->material()->as<MaterialWithMap>();
        material->map->image() = createText(text, worldScale);
        material->map->needsUpdate();
    }

    void setColor(const Color& color) {
        const auto& map = that->material()->as<MaterialWithMap>()->map;
        auto& image = map->image();
        for (int i = 0; i < image.width * image.height; ++i) {
            image.data()[i * 4 + 0] = 255 * color.r;
            image.data()[i * 4 + 1] = 255 * color.g;
            image.data()[i * 4 + 2] = 255 * color.b;
        }
        map->needsUpdate();
    }

    Image createText(const std::string& text, float worldScale = 1) {
        // Use stb_truetype to render the text into the texture

        if (text.empty()) {
            return {std::vector<unsigned char>(4, 255), 1, 1};
        }

        // Calculate scale
        const float scale = stbtt_ScaleForPixelHeight(&font_, 128);

        int ascent, descent, lineGap;
        stbtt_GetFontVMetrics(&font_, &ascent, &descent, &lineGap);

        // Measure text dimensions
        int totalWidth = 0;
        int minY = 0, maxY = 0;

        // First pass: calculate dimensions
        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            int ax, lsb;
            stbtt_GetCodepointHMetrics(&font_, c, &ax, &lsb);

            int c_x1, c_y1, c_x2, c_y2;
            stbtt_GetCodepointBitmapBox(&font_, c, scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);

            minY = std::min(minY, c_y1);
            maxY = std::max(maxY, c_y2);

            totalWidth += static_cast<int>(ax * scale);

            // Add kerning
            if (i + 1 < text.size()) {
                totalWidth += static_cast<int>(stbtt_GetCodepointKernAdvance(&font_, c, text[i + 1]) * scale);
            }
        }

        // Add padding
        const int padding = 4;
        int width = totalWidth + 2 * padding;
        int height = (maxY - minY) + 2 * padding;

        // Ensure minimum dimensions
        width = std::max(width, 1);
        height = std::max(height, 1);

        // Create buffer
        std::vector<unsigned char> pixels(width * height, 0);

        // Second pass: render text
        int x = padding;
        int baselineY = padding - minY;

        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            int ax, lsb;
            stbtt_GetCodepointHMetrics(&font_, c, &ax, &lsb);

            int c_x1, c_y1, c_x2, c_y2;
            stbtt_GetCodepointBitmapBox(&font_, c, scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);

            int glyphX = x + static_cast<int>(lsb * scale);
            int glyphY = baselineY + c_y1;
            int glyphW = c_x2 - c_x1;
            int glyphH = c_y2 - c_y1;

            // Bounds check
            if (glyphX >= 0 && glyphY >= 0 &&
                glyphX + glyphW <= width && glyphY + glyphH <= height) {

                int byteOffset = glyphX + glyphY * width;
                stbtt_MakeCodepointBitmap(
                        &font_,
                        pixels.data() + byteOffset,
                        glyphW, glyphH,
                        width,
                        scale, scale,
                        c);
            }

            x += static_cast<int>(ax * scale);

            // Add kerning
            if (i + 1 < text.size()) {
                x += static_cast<int>(stbtt_GetCodepointKernAdvance(&font_, c, text[i + 1]) * scale);
            }
        }

        // Convert grayscale to RGBA
        std::vector<unsigned char> rgba(width * height * 4, 0);
        for (int i = 0; i < width * height; ++i) {
            rgba[i * 4 + 0] = 255;
            rgba[i * 4 + 1] = 255;
            rgba[i * 4 + 2] = 255;
            rgba[i * 4 + 3] = pixels[i];
        }

        flipImage(rgba, 4, width, height);

        const auto aspect = (float) width / height;
        that->scale.set(worldScale * aspect, worldScale, 1);

        return {rgba, static_cast<unsigned int>(width), static_cast<unsigned int>(height)};
    }

private:
    Sprite* that;

    stbtt_fontinfo font_{};
    std::vector<unsigned char> fontBuffer;
};

TextNode::TextNode(const std::filesystem::path& fontPath)
    : Sprite(nullptr), pimpl_(std::make_unique<Impl>(this, fontPath)) {
}

void TextNode::setText(const std::string& text, float worldScale) {
    pimpl_->setText(text, worldScale);
}
std::shared_ptr<TextNode> TextNode::create(const std::filesystem::path& fontPath) {
    return std::make_shared<TextNode>(fontPath);
}

void TextNode::setColor(const Color& color) {
    pimpl_->setColor(color);
}

TextNode::~TextNode() = default;
