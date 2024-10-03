
#include <utility>

#include "threepp/objects/HUD.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;


void HUD::Options::updateElement(Object3D& o, std::pair<int, int> windowSize) {

    static Box3 bb;
    bb.setFromObject(o);
    const auto size = bb.getSize();

    Vector2 offset;
    if (verticalAlignment_ == VerticalAlignment::CENTER) {
        offset.y = (static_cast<float>(size.y) / 2);
    } else if (verticalAlignment_ == VerticalAlignment::TOP) {
        offset.y = static_cast<float>(size.y);
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

    o.position.x = pos.x * static_cast<float>(windowSize.first) - offset.x - margin_.x * (0.5 > pos.x ? -1.f : 1.f);
    o.position.y = pos.y * static_cast<float>(windowSize.second) - offset.y - margin_.y * (0.5 > pos.y ? -1.f : 1.f);
}


struct HUD::Impl: Scene, MouseListener {

    Impl(PeripheralsEventSource* eventSource, const std::pair<int, int>& size)
        : eventSource_(eventSource),
          size_(size),
          camera_(0, size_.first, size_.second, 0, 0.1, 10) {

        if (eventSource) eventSource->addMouseListener(*this);

        camera_.position.z = 1;
    }

    void apply(GLRenderer& renderer) {
        renderer.clearDepth();
        renderer.render(*this, camera_);
    }

    void remove(Object3D& object) override {
        Object3D::remove(object);

        map_.erase(&object);
    }

    void setSize(std::pair<int, int> size) {
        camera_.right = static_cast<float>(size.first);
        camera_.top = static_cast<float>(size.second);

        for (auto [obj, opts] : map_) {
            opts.updateElement(*obj, size);
        }

        camera_.updateProjectionMatrix();

        size_ = size;
    }

    void add(Object3D& object, Options opts) {
        Object3D::add(object);

        opts.updateElement(object, size_);

        map_[&object] = opts;
    }

    void add(const std::shared_ptr<Object3D>& object, Options opts) {
        Object3D::add(object);

        opts.updateElement(*object, size_);

        map_[object.get()] = opts;
    }

    void needsUpdate(Object3D& o) {

        if (map_.contains(&o)) {
            map_.at(&o).updateElement(o, size_);
        }
    }

    void handleMouseEvent(int button, const std::function<void(Options&, int)>& handler) {
        raycaster_.setFromCamera(mouse_, camera_);

        const auto intersects = raycaster_.intersectObjects(children, false);
        if (!intersects.empty()) {
            const auto front = intersects.front();
            if (map_.contains(front.object)) {
                handler(map_.at(front.object), button);
            }
        }
    }

    void onMouseDown(int button, const Vector2&) override {

        handleMouseEvent(button, [](Options& handler, int button) {
            handler.onMouseDown_(button);
        });
    }

    void onMouseUp(int button, const Vector2&) override {

        handleMouseEvent(button, [](Options& handler, int button) {
            handler.onMouseUp_(button);
        });
    }

    void onMouseMove(const Vector2& pos) override {

        mouse_.x = (pos.x / static_cast<float>(size_.first)) * 2 - 1;
        mouse_.y = -(pos.y / static_cast<float>(size_.second)) * 2 + 1;
    }

    ~Impl() override {
        if (eventSource_) eventSource_->removeMouseListener(*this);
    }

private:
    PeripheralsEventSource* eventSource_;

    std::pair<int, int> size_;
    OrthographicCamera camera_;

    Raycaster raycaster_;
    Vector2 mouse_{-Infinity<float>, -Infinity<float>};

    std::unordered_map<Object3D*, Options> map_;
};

HUD::HUD(std::pair<int, int> size)
    : pimpl_(std::make_unique<Impl>(nullptr, size)) {}


HUD::HUD(PeripheralsEventSource* eventSource)
    : pimpl_(std::make_unique<Impl>(eventSource, eventSource->size())) {}

void HUD::apply(GLRenderer& renderer) {
    pimpl_->apply(renderer);
}

void HUD::add(Object3D& object, Options opts) {
    pimpl_->add(object, std::move(opts));
}

void HUD::add(const std::shared_ptr<Object3D>& object, Options opts) {
    pimpl_->add(object, std::move(opts));
}

void HUD::remove(Object3D& object) {
    pimpl_->remove(object);
}

void HUD::setSize(std::pair<int, int> size) {
    pimpl_->setSize(size);
}

void HUD::needsUpdate(Object3D& o) {
    pimpl_->needsUpdate(o);
}

HUD::~HUD() = default;
