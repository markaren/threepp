
#include "geo/MapView.hpp"

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


MapView::MapView(std::unique_ptr<MapProvider> provider, std::unique_ptr<LODControl> lod)
    : provider(std::move(provider)), lod(std::move(lod)) {

    root = std::make_unique<MapPlaneNode>(nullptr, this);

    onBeforeRender = [this](void* renderer, Object3D* scene, Camera* camera, BufferGeometry*, Material*, std::optional<GeometryGroup>) {
        this->lod->updateLOD(*this, *camera, *static_cast<GLRenderer*>(renderer), *scene);
    };

    geometry_ = root->baseGeometry();
    material()->transparent = true;
    material()->depthWrite = false;
    material()->colorWrite = false;
    material()->opacity = 0;

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

void MapView::reload() {
    traverse([&](Object3D& object) {
      if (auto node = object.as<MapNode>()) {
          node->initialize();
      }
    });
}
