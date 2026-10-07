
#include "threepp/objects/TextSprite.hpp"

#include "threepp/utils/ImageUtils.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

using namespace threepp;

namespace {

    // Font::rasterize's pixelHeight for the glyph atlas.
    constexpr float kAtlasPixels = 64.f;

    // The pixelHeight that gives a TextBox::Line atlas the glyph resolution of a
    // TextBox::Ink one, so the box changes which rows are kept and not how sharp
    // the glyphs are. For the ink box, kAtlasPixels is the height of the font's
    // bounding box (its em, for a font that carries none); for the line box it
    // is the height of ascender to descender.
    float linePixels(const Font& font) {
        const float bbox = font.boundingBox.yMax - font.boundingBox.yMin;
        const float ink = bbox + static_cast<float>(font.underlineThickness);
        const float line = font.ascender > font.descender ? static_cast<float>(font.ascender - font.descender) : bbox;
        const float reference = ink > 0.f ? ink : static_cast<float>(font.resolution);
        if (line <= 0.f || reference <= 0.f) return kAtlasPixels;
        return std::max(1.f, std::round(kAtlasPixels * line / reference));
    }

}// namespace

struct TextSprite::Impl {

    Color color_;
    float worldScale_{};
    TextBox box_;
    std::string text_{"empty"};
    bool rasterized_ = false;// "empty" above is a placeholder, not an atlas

    Impl(TextSprite* that, Font font, std::optional<float> worldScale, TextBox box)
        : worldScale_(worldScale.value_or(1.f)), box_(box), that(that), font_(std::move(font)), linePixels_(linePixels(font_)) {

        that->setHorizontalAlignment(HorizontalAlignment::Left);
        that->setVerticalAlignment(VerticalAlignment::Below);

        const auto material = that->materialAs<MaterialWithMap>();
        material->map = Texture::create({});
    }

    void setText(const std::string& text) {

        // Re-rasterizing is expensive for the renderer as well (the glyph
        // atlas is re-uploaded whenever the texture version bumps), so skip
        // when the content hasn't changed. HUDs tend to call setText with
        // the same string every frame.
        if (text == text_ && rasterized_) return;
        this->text_ = text;
        rasterized_ = true;

        rasterize();
    }

    void setTextBox(TextBox box) {
        if (box == box_) return;
        box_ = box;
        if (rasterized_) rasterize();
    }

    void rasterize() {

        auto image = createText(text_);
        imgAspect_ = static_cast<float>(image.width()) / static_cast<float>(image.height());
        // TextBox::Ink: the image, whatever its height, is worldScale tall.
        // TextBox::Line: one line of it is, so two lines are twice that.
        lines_ = box_ == TextBox::Line ? std::max(1.f, static_cast<float>(image.height()) / linePixels_) : 1.f;

        const auto material = that->materialAs<MaterialWithMap>();
        material->map->images() = {image};
        material->map->needsUpdate();

        applyScale();
    }

    void setColor(const Color& color) {
        this->color_ = color;
        const auto& map = that->materialAs<MaterialWithMap>()->map;
        if (map->images().empty()) return;
        auto& image = map->image();
        for (int i = 0; i < static_cast<int>(image.width() * image.height()); ++i) {
            image.data()[i * 4 + 0] = 255 * color.r;
            image.data()[i * 4 + 1] = 255 * color.g;
            image.data()[i * 4 + 2] = 255 * color.b;
        }
        map->needsUpdate();
    }

    [[nodiscard]] Image createText(const std::string& text) const {
        return font_.rasterize(text, box_ == TextBox::Line ? linePixels_ : kAtlasPixels, color_, 2, box_);
    }

    void setWorldScale(float worldScale) {
        worldScale_ = worldScale;
        applyScale();
    }

    void applyScale() {
        const float height = worldScale_ * lines_;
        that->scale.set(imgAspect_ * height, height, 1.f);
    }

private:
    Sprite* that;
    Font font_;
    float linePixels_;
    float imgAspect_{1.f};
    float lines_{1.f};
};

TextSprite::TextSprite(const Font& font, std::optional<float> worldScale, TextBox box)
    : Sprite(nullptr), pimpl_(std::make_unique<Impl>(this, font, worldScale, box)) {
}

void TextSprite::setText(const std::string& text) {
    pimpl_->setText(text);
}

const Color& TextSprite::getColor() const {
    return pimpl_->color_;
}

std::string TextSprite::getText() const {
    return pimpl_->text_;
}

std::shared_ptr<TextSprite> TextSprite::create(const Font& fontPath, std::optional<float> worldScale, TextBox box) {
    return std::make_shared<TextSprite>(fontPath, worldScale, box);
}

void TextSprite::setColor(const Color& color) {
    pimpl_->setColor(color);
}

void TextSprite::setWorldScale(float worldScale) {
    pimpl_->setWorldScale(worldScale);
}

void TextSprite::setTextBox(TextBox box) {
    pimpl_->setTextBox(box);
}

TextBox TextSprite::getTextBox() const {
    return pimpl_->box_;
}

void TextSprite::setHorizontalAlignment(HorizontalAlignment h) {
    switch (h) {
        case HorizontalAlignment::Left:
            center.x = 0.f;
            break;
        case HorizontalAlignment::Center:
            center.x = 0.5f;
            break;
        case HorizontalAlignment::Right:
            center.x = 1.f;
            break;
    }
}
TextSprite::VerticalAlignment TextSprite::getVerticalAlignment() const {
    return center.y == 0.f ? VerticalAlignment::Above : (center.y == 1.f ? VerticalAlignment::Below : VerticalAlignment::Center);
}

TextSprite::HorizontalAlignment TextSprite::getHorizontalAlignment() const {
    return center.x == 0.f ? HorizontalAlignment::Left : (center.x == 1.f ? HorizontalAlignment::Right : HorizontalAlignment::Center);
}


void TextSprite::setVerticalAlignment(VerticalAlignment v) {
    switch (v) {
        case VerticalAlignment::Above:
            center.y = 0.f;
            break;
        case VerticalAlignment::Center:
            center.y = 0.5f;
            break;
        case VerticalAlignment::Below:
            center.y = 1.f;
            break;
    }
}

TextSprite::~TextSprite() = default;
