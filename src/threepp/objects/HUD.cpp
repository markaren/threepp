
#include "threepp/objects/HUD.hpp"

#include "threepp/renderers/GLRenderer.hpp"

using namespace threepp;


void HUD::Options::updateElement(Object3D& o, WindowSize windowSize) {

    static Box3 bb;
    bb.setFromObject(o);
    Vector3 size;
    bb.getSize(size);

    Vector2 offset;
    if (verticalAlignment == VerticalAlignment::CENTER) {
        offset.y = (float(size.y) / 2);
    } else if (verticalAlignment == VerticalAlignment::TOP) {
        offset.y = float(size.y);
    } else {
        offset.y = 0;
    }

    if (horizontalAlignment != HorizontalAlignment::LEFT) {

        if (horizontalAlignment == HorizontalAlignment::CENTER) {
            offset.x = (size.x / 2);
        } else if (horizontalAlignment == HorizontalAlignment::RIGHT) {
            offset.x = size.x;
        }
    } else {
        offset.x = 0;
    }

    o.position.x = pos.x * float(windowSize.width) - offset.x - (margin.x * ((0.5 > pos.x) ? -1.f : 1.f));
    o.position.y = pos.y * float(windowSize.height) - (offset.y) - (margin.y * ((0.5 > pos.y) ? -1.f : 1.f));
}

void HUD::apply(GLRenderer& renderer) {
    renderer.clearDepth();
    renderer.render(*this, camera_);
}

void HUD::remove(Object3D& object) {
    Object3D::remove(object);

    map_.erase(&object);
}

void HUD::setSize(WindowSize size) {
    camera_.right = size.width;
    camera_.top = size.height;

    for (auto [obj, opts] : map_) {
        opts.updateElement(*obj, size);
    }

    camera_.updateProjectionMatrix();

    size_ = size;
}

void HUD::add(Object3D& object, HUD::Options opts) {
    Object3D::add(object);

    opts.updateElement(object, size_);

    map_[&object] = opts;
}

void HUD::add(const std::shared_ptr<Object3D>& object, HUD::Options opts) {
    Object3D::add(object);

    opts.updateElement(*object, size_);

    map_[object.get()] = opts;
}

void HUD::needsUpdate(Object3D& o) {

    if (map_.count(&o)) {
        map_.at(&o).updateElement(o, size_);
    }
}
