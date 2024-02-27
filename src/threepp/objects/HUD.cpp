
#include "threepp/objects/HUD.hpp"

#include "threepp/renderers/GLRenderer.hpp"

using namespace threepp;


void HUD::Options::updateElement(Object3D& o, WindowSize windowSize) {

    static Box3 bb;
    bb.setFromObject(o);
    const auto size = bb.getSize();

    Vector2 offset;
    if (verticalAlignment_ == VerticalAlignment::CENTER) {
        offset.y = (float(size.y) / 2);
    } else if (verticalAlignment_ == VerticalAlignment::TOP) {
        offset.y = float(size.y);
    } else {
        offset.y = 0;
    }

    if (horizontalAlignment_ != HorizontalAlignment::LEFT) {

        if (horizontalAlignment_ == HorizontalAlignment::CENTER) {
            offset.x = (size.x / 2);
        } else if (horizontalAlignment_ == HorizontalAlignment::RIGHT) {
            offset.x = size.x;
        }
    } else {
        offset.x = 0;
    }

    o.position.x = pos.x * float(windowSize.width) - offset.x - (margin_.x * ((0.5 > pos.x) ? -1.f : 1.f));
    o.position.y = pos.y * float(windowSize.height) - (offset.y) - (margin_.y * ((0.5 > pos.y) ? -1.f : 1.f));
}


HUD::HUD(PeripheralsEventSource& eventSource)
    : size_(eventSource.size()), camera_(0, size_.width, size_.height, 0, 0.1, 10) {

    eventSource.addMouseListener(*this);

    camera_.position.z = 1;
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

void HUD::onMouseDown(int button, const Vector2& pos) {

    raycaster_.setFromCamera(mouse_, camera_);

    auto intersects = raycaster_.intersectObjects(children, false);
    if (!intersects.empty()) {
        auto front = intersects.front();
        if (map_.count(front.object)) {
            map_.at(front.object).onMouseDown_(button);
        }
    }
}

void HUD::onMouseUp(int button, const Vector2& pos) {

    raycaster_.setFromCamera(mouse_, camera_);

    auto intersects = raycaster_.intersectObjects(children, false);
    if (!intersects.empty()) {
        auto front = intersects.front();
        if (map_.count(front.object)) {
            map_.at(front.object).onMouseUp_(button);
        }
    }
}

void HUD::onMouseMove(const Vector2& pos) {

    mouse_.x = (pos.x / static_cast<float>(size_.width)) * 2 - 1;
    mouse_.y = -(pos.y / static_cast<float>(size_.height)) * 2 + 1;
}
