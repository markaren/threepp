
#include "threepp/postprocessing/Pass.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/GLRenderer.hpp"

using namespace threepp;

namespace {

    std::shared_ptr<BufferGeometry> fullScreenTriangle() {

        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create({-1.f, 3.f, 0.f, -1.f, -1.f, 0.f, 3.f, -1.f, 0.f}, 3));
        geometry->setAttribute("uv", FloatBufferAttribute::create({0.f, 2.f, 0.f, 0.f, 2.f, 0.f}, 2));

        return geometry;
    }

}// namespace


FullScreenQuad::FullScreenQuad(const std::shared_ptr<Material>& material)
    : mesh_(Mesh::create(fullScreenTriangle(), material)),
      camera_(OrthographicCamera::create(-1, 1, 1, -1, 0, 1)) {

    // The triangle reaches outside the camera's frustum by design; culling it
    // on a bounding sphere that was never meant to fit is only a way to lose
    // the pass on some frames.
    mesh_->frustumCulled = false;
}

void FullScreenQuad::setMaterial(const std::shared_ptr<Material>& material) {

    mesh_->setMaterial(material);
}

std::shared_ptr<Material> FullScreenQuad::material() const {

    return mesh_->material();
}

void FullScreenQuad::render(GLRenderer& renderer) {

    renderer.render(*mesh_, *camera_);
}

FullScreenQuad::~FullScreenQuad() = default;
