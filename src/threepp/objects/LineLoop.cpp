
#include <memory>

#include "threepp/objects/LineLoop.hpp"

using namespace threepp;


LineLoop::LineLoop(
        const std::shared_ptr<BufferGeometry>& geometry,
        const std::shared_ptr<Material>& material)
    : Line(geometry, material) {}


const std::string& LineLoop::type() const {

    static const std::string typeName = "LineLoop";
    return typeName;
}

std::shared_ptr<LineLoop> LineLoop::create(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material) {

    return std::make_shared<LineLoop>(geometry, (material));
}

std::shared_ptr<Object3D> LineLoop::createDefault() {

    return create();
}
