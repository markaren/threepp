
#include <memory>

#include "threepp/objects/LineLoop.hpp"

using namespace threepp;


LineLoop::LineLoop(
        const std::shared_ptr<BufferGeometry>& geometry,
        const std::shared_ptr<Material>& material)
    : Line(geometry, material) {}


std::string LineLoop::type() const {

    return "LineLoop";
}

std::shared_ptr<Object3D> LineLoop::clone(bool recursive) {
    auto clone = create();
    clone->copy(*this, recursive);

    return clone;
}

std::shared_ptr<LineLoop> LineLoop::create(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material) {

    return std::make_shared<LineLoop>(geometry, (material));
}
