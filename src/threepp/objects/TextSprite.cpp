
#include "threepp/objects/TextSprite.hpp"

#include "threepp/utils/ImageUtils.hpp"

#include <utility>

using namespace threepp;

struct TextSprite::Impl {

    Color color_;
    std::string text_{"empty"};

    Impl(Sprite* that, Font font): that(that), font_(std::move(font)) {

        that->center.set(0, 1);
    }

    void setText(const std::string& text, float worldScale) {

        this->text_ = text;

        auto image = createText(text);
        flipImage(image.data(), 4, image.width, image.height);
        const float imgAspect = static_cast<float>(image.width) / static_cast<float>(image.height);
        that->scale.set(imgAspect, 1, 1);

        const auto material = that->material()->as<MaterialWithMap>();
        material->map = Texture::create(image);
        material->map->offset.set(0.5f, 0.5f);
        material->map->needsUpdate();

        Box3 bb;
        bb.setFromObject(*that);
        Vector3 size;
        bb.getSize(size);

        const float correction = worldScale / size.y;
        that->scale.set(imgAspect * correction, 1.0f * correction, 1);

    }

    void setColor(const Color& color) {
        this->color_ = color;
        const auto& map = that->material()->as<MaterialWithMap>()->map;
        if (!map) return;
        auto& image = map->image();
        for (int i = 0; i < image.width * image.height; ++i) {
            image.data()[i * 4 + 0] = 255 * color.r;
            image.data()[i * 4 + 1] = 255 * color.g;
            image.data()[i * 4 + 2] = 255 * color.b;
        }
        map->needsUpdate();
    }

    [[nodiscard]] Image createText(const std::string& text) const {
        return font_.rasterize(text, 128, color_, 4);
    }

private:
    Sprite* that;
    Font font_;
};

TextSprite::TextSprite(const Font& font)
    : Sprite(nullptr), pimpl_(std::make_unique<Impl>(this, font)) {
}

void TextSprite::setText(const std::string& text, float worldScale) {
    pimpl_->setText(text, worldScale);
}

const Color& TextSprite::getColor() const {
    return pimpl_->color_;
}

std::string TextSprite::getText() const {
    return pimpl_->text_;
}

std::shared_ptr<TextSprite> TextSprite::create(const Font& fontPath) {
    return std::make_shared<TextSprite>(fontPath);
}

void TextSprite::setColor(const Color& color) {
    pimpl_->setColor(color);
}

TextSprite::~TextSprite() = default;
