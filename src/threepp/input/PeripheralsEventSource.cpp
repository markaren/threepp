
#include "threepp/input/PeripheralsEventSource.hpp"

#include <algorithm>

using namespace threepp;

void PeripheralsEventSource::setIOCapture(IOCapture* capture) {
    ioCapture_ = capture;
}

void PeripheralsEventSource::onDrop(std::function<void(std::vector<std::string>)> paths) {
    dropListener_ = std::move(paths);
}
