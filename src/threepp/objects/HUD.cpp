
#include <utility>

#include "threepp/objects/HUD.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include <iostream>

using namespace threepp;


void HUD::Options::updateElement(std::pair<int, int> windowSize) {

    static Box3 bb;
    bb.setFromObject(*object_, true);
    const auto size = bb.getSize();

    Vector2 offset;
    if (verticalAlignment_ == VerticalAlignment::CENTER) {
        offset.y = (static_cast<float>(size.y) * 0.5f);
    } else if (verticalAlignment_ == VerticalAlignment::BELOW) {
        offset.y = static_cast<float>(size.y);
    } else {
        offset.y = 0;
    }

    if (horizontalAlignment_ != HorizontalAlignment::LEFT) {

        if (horizontalAlignment_ == HorizontalAlignment::CENTER) {
            offset.x = (size.x * 0.5f);
        } else if (horizontalAlignment_ == HorizontalAlignment::RIGHT) {
            offset.x = size.x;
        }
    } else {
        offset.x = 0;
    }

    object_->position.x = pos.x * static_cast<float>(windowSize.first) - offset.x - margin_.x * (0.5f > pos.x ? -1.f : 1.f);
    object_->position.y = pos.y * static_cast<float>(windowSize.second) - offset.y - margin_.y * (0.5f > pos.y ? -1.f : 1.f);
}


struct HUD::Impl: MouseListener {

    Scene scene;

    Impl(GLRenderer& renderer, PeripheralsEventSource* eventSource)
        : renderer_(&renderer),
    eventSource_(eventSource),
          camera_(0, static_cast<float>(renderer.size().first), static_cast<float>(renderer.size().second), 0, 0.1f, 10.f) {

        if (eventSource) eventSource->addMouseListener(*this);

        camera_.position.z = 1;


        if (renderer_->autoClear) {
            std::cerr << "[HUD] Warning: autoClear is enabled on the renderer. HUD will not work properly. Please set autoClear to false." << std::endl;
        }
    }

    void render() {

        if (renderer_->size() != lastSize_) setSize(renderer_->size());

        for (const auto& opt : options_) {
            if (opt->needsUpdate_) {
                opt->updateElement(renderer_->size());
                opt->needsUpdate_ = false;
            }
        }

        renderer_->clearDepth();
        renderer_->render(scene, camera_);
    }

    void remove(Object3D& object) {
        scene.remove(object);
    }

    void setSize(std::pair<int, int> size) {
        camera_.right = static_cast<float>(size.first);
        camera_.top = static_cast<float>(size.second);

        for (const auto& opt : options_) {
            opt->updateElement(size);
        }

        camera_.updateProjectionMatrix();

        lastSize_ = size;
    }

    Options& add(Object3D& object) {
        scene.add(object);

        auto& opts = *options_.emplace_back(std::make_unique<Options>());
        opts.object_ = &object;

        return opts;
    }

    Options& add(const std::shared_ptr<Object3D>& object) {
        scene.add(object);

        auto& opts = *options_.emplace_back(std::make_unique<Options>());
        opts.object_ = object.get();

        return opts;
    }

    Options* getStoredOptions(Object3D& object) {

        for (const auto& opt : options_) {
            if (opt->object_ == &object) return opt.get();
        }

        return nullptr;
    }

    void handleMouseEvent(int button, const std::function<void(Options&, int)>& handler) {
        raycaster_.setFromCamera(mouse_, camera_);

        const auto intersects = raycaster_.intersectObjects(scene.children, false);
        if (!intersects.empty()) {
            const auto& front = intersects.front();

            for (const auto& opt : options_) {
                if (opt->object_ == front.object) {
                    handler(*opt, button);
                    break;
                }
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

        const auto size = renderer_->size();
        mouse_.x = (pos.x / static_cast<float>(size.first)) * 2 - 1;
        mouse_.y = -(pos.y / static_cast<float>(size.second)) * 2 + 1;
    }

    ~Impl() override {
        if (eventSource_) eventSource_->removeMouseListener(*this);
    }

private:
    GLRenderer* renderer_;
    PeripheralsEventSource* eventSource_;

    std::pair<int, int> lastSize_;
    OrthographicCamera camera_;

    Raycaster raycaster_;
    Vector2 mouse_{-Infinity<float>, -Infinity<float>};

    std::vector<std::unique_ptr<Options>> options_;
};

HUD::HUD(GLRenderer& renderer, PeripheralsEventSource* eventSource)
    : pimpl_(std::make_unique<Impl>(renderer, eventSource)) {}


void HUD::render() {
    pimpl_->render();
}

HUD::Options& HUD::add(Object3D& object) {
    return pimpl_->add(object);
}

HUD::Options& HUD::add(const std::shared_ptr<Object3D>& object) {
    return pimpl_->add(object);
}

HUD::Options* HUD::getStoredOptions(Object3D& obj) {
    return pimpl_->getStoredOptions(obj);
}

void HUD::remove(Object3D& object) {
    pimpl_->remove(object);
}

HUD::~HUD() = default;
