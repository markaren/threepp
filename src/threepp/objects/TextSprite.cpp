
#include "threepp/objects/TextSprite.hpp"

#include "threepp/utils/ImageUtils.hpp"

#include <utility>

using namespace threepp;

struct TextSprite::Impl {

    Color color_;
    float worldScale_{};
    std::string text_{"empty"};

    Impl(TextSprite* that, Font font, std::optional<float> worldScale)
        : that(that), font_(std::move(font)), worldScale_(worldScale.value_or(1.f)) {

        that->setHorizontalAlignment(HorizontalAlignment::Left);
        that->setVerticalAlignment(VerticalAlignment::Below);

        const auto material = that->material()->as<MaterialWithMap>();
        material->map = Texture::create({});
        material->map->offset.set(0.5f, 0.5f);
    }

    void setText(const std::string& text) {

        this->text_ = text;

        auto image = createText(text);
        imgAspect_ = static_cast<float>(image.width) / static_cast<float>(image.height);

        const auto material = that->material()->as<MaterialWithMap>();
        material->map->images() = {image};
        material->map->needsUpdate();

        applyScale();
    }

    void setColor(const Color& color) {
        this->color_ = color;
        const auto& map = that->material()->as<MaterialWithMap>()->map;
        if (map->images().empty()) return;
        auto& image = map->image();
        for (int i = 0; i < image.width * image.height; ++i) {
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
