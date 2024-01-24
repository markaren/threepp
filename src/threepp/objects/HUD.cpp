
#include "threepp/objects/HUD.hpp"

#include "threepp/loaders/FontLoader.hpp"
#include "threepp/materials/SpriteMaterial.hpp"

using namespace threepp;

namespace {

    std::optional<Font> loadFont(const std::filesystem::path& fontPath) {
        FontLoader loader;
        return loader.load(fontPath);
    }

}// namespace

TextRef::TextRef(const std::filesystem::path& fontPath, unsigned int size)
    : font_(*loadFont(fontPath)), size_(size), mesh_(std::make_shared<Mesh>(BufferGeometry::create(), SpriteMaterial::create())) {

    mesh_->position.set(0, 0, -1);
}

void TextRef::setSize(unsigned int size) {
    size_ = size;
}

void TextRef::setText(const std::string& str) {

    TextGeometry::Options opts(font_, size_);
    opts.height = 1;
    auto geometry = TextGeometry::create(str, opts);
    mesh_->setGeometry(geometry);

    if (verticalAlignment_ == VerticalAlignment::CENTER) {
        offset_.y = float(size_) / 2;
    } else if (verticalAlignment_ == VerticalAlignment::TOP) {
        offset_.y = float(size_);
    } else {
        offset_.y = 0;
    }

    Vector3 size;
    if (horizontalAlignment_ != HorizontallAlignment::LEFT) {
        if (!mesh_->geometry()->boundingBox) {
            mesh_->geometry()->computeBoundingBox();
            auto bb = mesh_->geometry()->boundingBox;
            bb->getSize(size);
            if (horizontalAlignment_ == HorizontallAlignment::CENTER) {
                offset_.x = size.x / 2;
            } else {
                offset_.x = size.x;
            }
        }
    }

    setPosition(pos_.x, pos_.y);
}

void TextRef::setColor(const Color& color) {
    mesh_->material()->as<SpriteMaterial>()->color.copy(color);
}

void TextRef::setPosition(float x, float y) {
    mesh_->position.x = x * 100 - offset_.x;
    mesh_->position.y = y * 100 - offset_.y;
    pos_.set(x, y);
}

void TextRef::setVerticalAlignment(VerticalAlignment verticalAlignment) {
    verticalAlignment_ = verticalAlignment;
}

void TextRef::setHorizontalAlignment(HorizontallAlignment horizontalAlignment) {
    horizontalAlignment_ = horizontalAlignment;
}
