
#include "threepp/canvas/Canvas.hpp"

#include "threepp/favicon.hpp"
#include "threepp/loaders/ImageLoader.hpp"

#include <optional>

using namespace threepp;


void CanvasBase::handleTasks() {
    while (!tasks_.empty()) {
        auto& task = tasks_.top();
        if (task.second < getTime()) {
            task.first();
            tasks_.pop();
        } else {
            break;
        }
    }
}

void CanvasBase::animate(const std::function<void()>& f) {

    while (animateOnce(f)) {}
}

void CanvasBase::onWindowResize(std::function<void(WindowSize)> f) {

    this->resizeListener_ = std::move(f);
}

void CanvasBase::invokeLater(const std::function<void()>& f, float t) {

    tasks_.emplace(f, static_cast<float>(getTime()) + t);
}

const WindowSize& CanvasBase::getSize() const {

    return size_;
}

float CanvasBase::getAspect() const {

    return size_.getAspect();
}

std::optional<Image> CanvasBase::loadFavicon() {
    ImageLoader loader;
    return loader.load(faviconSource(), 4, false);
}
CanvasBase::CanvasBase(const WindowSize& size): size_(size) {}

CanvasBase::~CanvasBase() = default;
