
#include "geo/nodes/MapNode.hpp"
#include "geo/MapView.hpp"

#include <iostream>

using namespace threepp;


MapNode::MapNode(
        MapNode* parentNode, MapView* mapView, int location, int level, int x, int y,
        const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material)
    : Mesh(geometry, material), parentNode(parentNode),
      mapView(mapView), location(location), level(level), x(x), y(y) {}

void MapNode::subdivide() {
    const auto maxZoom = this->mapView->maxZoom();
    if (!this->children.empty() || (this->level + 1 > maxZoom) || this->parentNode && (this->parentNode->nodesLoaded < MapNode::childrens)) {
        return;
    }

    this->createChildNodes();

    layers.disable(0);

    this->subdivided = true;
}

void MapNode::simplify() {
    const auto minZoom = this->mapView->minZoom();
    if (this->level - 1 < minZoom) {
        return;
    }

    // Clear children and reset flags
    this->subdivided = false;
    this->layers.enable(0);
    this->clear();
    this->nodesLoaded = 0;
}

void MapNode::loadData() {
    if ((this->level < this->mapView->getProvider()->minZoom) || (this->level > this->mapView->getProvider()->maxZoom)) {
        std::cerr << "Geo-Three: Loading tile outside of provider range." << std::endl;

        this->material()->as<MaterialWithMap>()->map = nullptr;
        this->material()->needsUpdate();
        return;
    }

    auto image = mapView->getProvider()->fetchTile(this->level, this->x, this->y);

    if (this->disposed) {
        return;
    }

    const auto texture = Texture::create(image);
    texture->generateMipmaps = true;
    texture->format = Format::RGBA;
    texture->magFilter = Filter::Linear;
    texture->minFilter = Filter::Linear;
    texture->needsUpdate();

    this->material()->as<MaterialWithMap>()->map = texture;
    this->material()->needsUpdate();
}

void MapNode::nodeReady() {
    if (this->disposed) {
        std::cerr << "Geo-Three: nodeReady() called for disposed node" << std::endl;
        disposed = true;
        return;
    }

    if (this->parentNode) {
        this->parentNode->nodesLoaded++;

        if (this->parentNode->nodesLoaded == MapNode::childrens) {
            if (this->parentNode->subdivided == true) {

                this->layers.disable(0);
            }

            for (unsigned i = 0; i < this->parentNode->children.size(); i++) {
                this->parentNode->children[i]->visible = true;
            }
        }

        if (this->parentNode->nodesLoaded > MapNode::childrens) {
            std::cerr << "Geo-Three: Loaded more children objects than expected." << std::endl;
        }
    }
    // If its the root object just set visible
    else {
        this->visible = true;
    }
}
