
#include "MapView.hpp"

#include "lod/LODControl.hpp"
#include "lod/LODFrustum.hpp"
#include "lod/LODRaycast.hpp"

using namespace threepp;


namespace {

    void subdivide(MapNode& node, float depth) {
        if (depth <= 0) {
            return;
        }

        node.subdivide();

        for (unsigned i = 0; i < node.children.size(); i++) {
            if (auto child = node.children[i]->as<MapNode>()) {

                subdivide(*child, depth - 1);
            }
        }
    }

}// namespace


MapView::MapView(std::unique_ptr<MapProvider> provider, std::unique_ptr<LODControl> lod): provider(std::move(provider)), lod(std::move(lod)) {

    onBeforeRender = [this](void* renderer, Object3D* scene, Camera* camera, BufferGeometry*, Material*, std::optional<GeometryGroup>) {
        this->lod->updateLOD(*this, *camera, *static_cast<GLRenderer*>(renderer), *scene);
    };

    root = std::make_unique<MapPlaneNode>(nullptr, this);

    geometry_ = root->baseGeometry();
    material()->transparent = false;
    material()->as<MaterialWithWireframe>()->wireframe = true;
    material()->opacity = 0.0;
    material()->depthWrite = false;
    material()->colorWrite = false;

    scale.copy(root->baseScale());

    add(*root);
    root->initialize();

    preSubDivide();
}

void MapView::preSubDivide() {

    const auto minZoom = this->provider->minZoom;
    if (minZoom > 0) {
        subdivide(*this->root, minZoom);
    }
}

MapProvider* MapView::getProvider() const {

    return provider ? provider.get() : nullptr;
}
