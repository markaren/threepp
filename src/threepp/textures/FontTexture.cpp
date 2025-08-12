
#include "threepp/textures/FontTexture.hpp"

#include "external/stb/stb_truetype.h"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/utils/ImageUtils.hpp"

#include <fstream>

using namespace threepp;

struct FontTexture::Impl {

    Impl(const std::filesystem::path& fontFile, float fontSize, unsigned int padding)
        : fontSize_(fontSize), padding_(padding) {

        if (!std::filesystem::exists(fontFile)) {
            throw std::runtime_error("Font file not found: " + fontFile.string());
        }


        std::ifstream file(fontFile, std::ios::binary);
        fontBuffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});


        stbtt_InitFont(&font_, fontBuffer.data(), stbtt_GetFontOffsetForIndex(fontBuffer.data(), 0));
    }

    Image createText(const std::string& text) {
        // Use stb_truetype to render the text into the texture
        // This method should update the texture data based on the provided text


        // Calculate scale
        const float scale = stbtt_ScaleForPixelHeight(&font_, fontSize_);

        int ascent, descent, lineGap;
        stbtt_GetFontVMetrics(&font_, &ascent, &descent, &lineGap);

        // Calculate text width and height
        int width = 0;
        int min_x1 = 0, min_y1 = 0, max_x2 = 0, max_y2 = 0;
        int x = 0;

        // First pass: measure text dimensions
        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            int ax, lsb;
            stbtt_GetCodepointHMetrics(&font_, c, &ax, &lsb);

            int c_x1, c_y1, c_x2, c_y2;
            stbtt_GetCodepointBitmapBox(&font_, c, scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);

            int x_left = x + static_cast<int>(lsb * scale);
            int x_right = x_left + (c_x2 - c_x1);

            if (i == 0) {
                min_x1 = x_left;
                min_y1 = c_y1;
                max_x2 = x_right;
                max_y2 = c_y2;
            } else {
                min_x1 = std::min(min_x1, x_left);
                min_y1 = std::min(min_y1, c_y1);
                max_x2 = std::max(max_x2, x_right);
                max_y2 = std::max(max_y2, c_y2);
            }

            x += static_cast<int>(ax * scale);

            // Add kerning
            if (i + 1 < text.size()) {
                x += static_cast<int>(stbtt_GetCodepointKernAdvance(&font_, c, text[i + 1]) * scale);
            }
        }

        // Calculate buffer dimensions with padding
        const int padding = 4;
        width = max_x2 - min_x1 + 2 * padding;
        int height = max_y2 - min_y1 + 2 * padding;

        // Ensure minimum dimensions
        width = std::max(width, 1);
        height = std::max(height, 1);

        // Allocate grayscale buffer
        std::vector<unsigned char> pixels(width * height, 0);

        // Second pass: render text
        x = padding - min_x1;
        int y = padding - min_y1 + static_cast<int>(ascent * scale);

        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            int ax, lsb;
            stbtt_GetCodepointHMetrics(&font_, c, &ax, &lsb);

            int c_x1, c_y1, c_x2, c_y2;
            stbtt_GetCodepointBitmapBox(&font_, c, scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);

            // Calculate position and dimensions
            int glyph_x = x + static_cast<int>(lsb * scale);
            int glyph_y = y + c_y1;
            int glyph_w = c_x2 - c_x1;
            int glyph_h = c_y2 - c_y1;

            // Safety bounds check
            if (glyph_x >= 0 && glyph_y >= 0 &&
                glyph_x + glyph_w <= width && glyph_y + glyph_h <= height) {
                int byteOffset = glyph_x + glyph_y * width;
                stbtt_MakeCodepointBitmap(&font_, pixels.data() + byteOffset,
                                          glyph_w, glyph_h, width, scale, scale, c);
            }

            x += static_cast<int>(ax * scale);

            // Add kerning
            if (i + 1 < text.size()) {
                x += static_cast<int>(stbtt_GetCodepointKernAdvance(&font_, c, text[i + 1]) * scale);
            }
        }

        // Draw text
        x = 0, y = int(fontSize_);
        for (char c : text) {
            int ax, lsb;
            stbtt_GetCodepointHMetrics(&font_, c, &ax, &lsb);

            int c_x1, c_y1, c_x2, c_y2;
            stbtt_GetCodepointBitmapBox(&font_, c, scale, scale, &c_x1, &c_y1, &c_x2, &c_y2);

            int byteOffset = x + lsb * scale + (y + c_y1) * width;
            stbtt_MakeCodepointBitmap(&font_, pixels.data() + byteOffset, c_x2 - c_x1, c_y2 - c_y1, width, scale, scale, c);

            x += ax * scale;
        }

        // Convert grayscale to RGBA
        std::vector<unsigned char> rgba(width * height * 4, 0);
        for (int i = 0; i < width * height; ++i) {
            rgba[i * 4 + 0] = 255;
            rgba[i * 4 + 1] = 255;
            rgba[i * 4 + 2] = 255;
            rgba[i * 4 + 3] = pixels[i];
        }

        return {rgba, static_cast<unsigned int>(width), static_cast<unsigned int>(height)};
    }


private:
    stbtt_fontinfo font_{};
    std::vector<unsigned char> fontBuffer;

    float fontSize_;
    unsigned int padding_;
    Color color_;
};

FontTexture::FontTexture(const std::filesystem::path& fontFile, float fontSize, unsigned int padding)
    : Texture({Image{{}, 256, 64}}), pimpl_(std::make_unique<Impl>(fontFile, fontSize, padding)) {

    setText(text_);
}


void FontTexture::setText(const std::string& text) {

    this->text_ = text;

    image() = pimpl_->createText(text);
    flipImage(image().data<unsigned char>(), 4, image().width, image().height);

    needsUpdate();
}

void FontTexture::setColor(const Color& color) {
    auto& image = this->image();
    for (int i = 0; i < image.width * image.height; ++i) {
        image.data()[i * 4 + 0] = color.r * 255;
        image.data()[i * 4 + 1] = color.g * 255;
        image.data()[i * 4 + 2] = color.b * 255;
    }
}

FontTexture::~FontTexture() = default;
