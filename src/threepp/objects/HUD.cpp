
#include "threepp/objects/HUD.hpp"

#include "threepp/geometries/ShapeGeometry.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/renderers/GLRenderer.hpp"

using namespace threepp;

namespace {

    std::optional<Font> loadFont(const std::filesystem::path& fontPath) {
        FontLoader loader;
        return loader.load(fontPath);
    }

}// namespace

HudText::HudText(const std::filesystem::path& fontPath, unsigned int size)
    : font_(*loadFont(fontPath)), size_(size), mesh_(std::make_shared<Mesh>(BufferGeometry::create(), SpriteMaterial::create())) {

    mesh_->position.set(0, 0, -1);
    setColor(Color::gray);
}

void HudText::scale(float scale) {
    scale_ = scale;
    mesh_->geometry()->scale(scale, scale, scale);

    updateSettings();
}

void HudText::setText(const std::string& str) {

    auto shapes = font_.generateShapes(str, size_);
    std::vector<const Shape*> ss;
    ss.reserve(shapes.size());
    for (const auto& s : shapes) {
        ss.emplace_back(&s);
    }
    auto geometry = ShapeGeometry::create(ss, 5);
    geometry->scale(scale_, scale_, scale_);
    mesh_->setGeometry(geometry);

    updateSettings();
}

void HudText::setColor(const Color& color) {
    mesh_->material()->as<SpriteMaterial>()->color.copy(color);
}

void HudText::setPosition(float x, float y) {

    updateSettings();

    pos_.set(x, y);
}

void HudText::setVerticalAlignment(VerticalAlignment verticalAlignment) {
    verticalAlignment_ = verticalAlignment;


    updateSettings();
}

void HudText::setHorizontalAlignment(HorizontallAlignment horizontalAlignment) {
    horizontalAlignment_ = horizontalAlignment;

    updateSettings();
}
void HudText::updateSettings() {

    if (verticalAlignment_ == VerticalAlignment::CENTER) {
        offset_.y = float(size_) / 2;
    } else if (verticalAlignment_ == VerticalAlignment::TOP) {
        offset_.y = float(size_);
    } else {
        offset_.y = 0;
    }

    if (horizontalAlignment_ != HorizontallAlignment::LEFT) {

        mesh_->geometry()->computeBoundingBox();
        const auto& bb = mesh_->geometry()->boundingBox;
        Vector3 size;
        bb->getSize(size);

        if (horizontalAlignment_ == HorizontallAlignment::CENTER) {
            offset_.x = size.x / 2;
        } else if (horizontalAlignment_ == HorizontallAlignment::RIGHT) {
            offset_.x = size.x;
        }
    } else {
        offset_.x = 0;
    }

    mesh_->position.x = pos_.x * 100 - offset_.x - (margin_.x * ((0.5 > pos_.x) ? -1.f : 1.f));
    mesh_->position.y = pos_.y * 100 - offset_.y - (margin_.y * ((0.5 > pos_.y) ? -1.f : 1.f));
}

void HudText::setMargin(const Vector2& margin) {
    margin_.copy(margin);

    updateSettings();
}
void HUD::apply(GLRenderer& renderer) {
    renderer.clearDepth();
    renderer.render(*this, camera_);
}
