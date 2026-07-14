
#include "threepp/objects/TextSprite.hpp"

#include "threepp/utils/ImageUtils.hpp"

#include <utility>

using namespace threepp;

struct TextSprite::Impl {

    Color color_;
    float worldScale_{};
    std::string text_{"empty"};
    bool rasterized_ = false;// "empty" above is a placeholder, not an atlas

    Impl(TextSprite* that, Font font, std::optional<float> worldScale)
        : worldScale_(worldScale.value_or(1.f)), that(that), font_(std::move(font)) {

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

        auto image = createText(text);
        imgAspect_ = static_cast<float>(image.width()) / static_cast<float>(image.height());

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
        return font_.rasterize(text, 64, color_, 2);
    }

    void setWorldScale(float worldScale) {
        worldScale_ = worldScale;
        applyScale();
    }

    void applyScale() {
        that->scale.set(imgAspect_ * worldScale_, worldScale_, 1.f);
    }

private:
    Sprite* that;
    Font font_;
    float imgAspect_{1.f};
};

TextSprite::TextSprite(const Font& font, std::optional<float> worldScale)
    : Sprite(nullptr), pimpl_(std::make_unique<Impl>(this, font, worldScale)) {
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

std::shared_ptr<TextSprite> TextSprite::create(const Font& fontPath, std::optional<float> worldScale) {
    return std::make_shared<TextSprite>(fontPath, worldScale);
}

void TextSprite::setColor(const Color& color) {
    pimpl_->setColor(color);
}

void TextSprite::setWorldScale(float worldScale) {
    pimpl_->setWorldScale(worldScale);
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
