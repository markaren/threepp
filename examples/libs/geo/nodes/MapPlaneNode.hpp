//https://github.com/tentone/geo-three/blob/master/source/nodes/MapPlaneNode.ts

#ifndef THREEPP_MAPPLANENODE_HPP
#define THREEPP_MAPPLANENODE_HPP

#include "geo/geometries/MapNodeGeometry.hpp"
#include "geo/nodes/MapNode.hpp"
#include "geo/utils/UnitUtils.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <iostream>

namespace threepp {

    class MapPlaneNode: public MapNode {

    public:
        MapPlaneNode(MapNode* parent, MapView* mapView, int location = QuadTreePosition::root, int level = 0, int x = 0, int y = 0)
            : MapNode(parent, mapView, location, level, x, y, baseGeometry(), MeshBasicMaterial::create({{"wireframe", false}})) {

            this->matrixAutoUpdate = false;
            this->layers.enable(0);
            this->visible = false;
        }

        void initialize() override {

            loadData();
            if (!initialized) {
                nodeReady();
                initialized = true;
            }
        }

        [[nodiscard]] Vector3 baseScale() const override {

            return {utils::EARTH_PERIMETER, 1, utils::EARTH_PERIMETER};
        }

        std::shared_ptr<BufferGeometry> baseGeometry() override {

            static std::shared_ptr<BufferGeometry> baseGeom = MapNodeGeometry::create(1, 1, 1, 1, false);

            return baseGeom;
        }

        void createChildNodes() override {

            const auto level = this->level + 1;
            const auto x = this->x * 2;
            const auto y = this->y * 2;

            auto node = std::make_shared<MapPlaneNode>(this, this->mapView, QuadTreePosition::topLeft, level, x, y);
            node->scale.set(0.5, 1.0, 0.5);
            node->position.set(-0.25, 0, -0.25);
            this->add(node);
            node->updateMatrix();
            node->updateMatrixWorld(true);
            node->initialize();

            node = std::make_shared<MapPlaneNode>(this, this->mapView, QuadTreePosition::topRight, level, x + 1, y);
            node->scale.set(0.5, 1.0, 0.5);
            node->position.set(0.25, 0, -0.25);
            this->add(node);
            node->updateMatrix();
            node->updateMatrixWorld(true);
            node->initialize();

            node = std::make_shared<MapPlaneNode>(this, this->mapView, QuadTreePosition::bottomLeft, level, x, y + 1);
            node->scale.set(0.5, 1.0, 0.5);
            node->position.set(-0.25, 0, 0.25);
            this->add(node);
            node->updateMatrix();
            node->updateMatrixWorld(true);
            node->initialize();

            node = std::make_shared<MapPlaneNode>(this, this->mapView, QuadTreePosition::bottomRight, level, x + 1, y + 1);
            node->scale.set(0.5, 1.0, 0.5);
            node->position.set(0.25, 0, 0.25);
            this->add(node);
            node->updateMatrix();
            node->updateMatrixWorld(true);
            node->initialize();
        }

    private:
        bool initialized = false;
    };

}// namespace threepp

#endif//THREEPP_MAPPLANENODE_HPP
