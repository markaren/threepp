//https://github.com/tentone/geo-three/blob/master/source/nodes/MapPlaneNode.ts

#ifndef THREEPP_MAPPLANENODE_HPP
#define THREEPP_MAPPLANENODE_HPP

#include "../utils/UnitUtils.hpp"
#include "MapNode.hpp"
#include "geo/geometries/MapNodeGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include <future>

namespace threepp {

    namespace {
        std::shared_ptr<BufferGeometry> baseGeom = MapNodeGeometry::create(1, 1, 1, 1, false);
    }

    class MapPlaneNode: public MapNode {

    public:
        MapPlaneNode(MapNode* parent, MapView* mapView, int location = QuadTreePosition::root, int level = 0, float x = 0, float y = 0)
            : MapNode(parent, mapView, location, level, x, y, baseGeom, MeshBasicMaterial::create({{"wireframe", false}})) {

            this->matrixAutoUpdate = false;
            this->layers.enable(0);
            this->visible = false;

            initialize();
        }

        void initialize() override {

            loadData();
            nodeReady();
            this->visible = true;
        }

        [[nodiscard]] Vector3 baseScale() const override {

            return {utils::EARTH_PERIMETER, 1, utils::EARTH_PERIMETER};
        }

        std::shared_ptr<BufferGeometry> baseGeometry() override {

            return baseGeom;
        }

        void createChildNodes() override {

            const auto level = this->level + 1;
            const auto x = this->x * 2;
            const auto y = this->y * 2;

            auto node = std::make_shared<MapPlaneNode>(this, this->mapView, QuadTreePosition::topLeft, level, x, y);
//            node->initialize();
            node->scale.set(0.5, 1.0, 0.5);
            node->position.set(-0.25, 0, -0.25);
            this->add(node);
            node->updateMatrix();
            node->updateMatrixWorld(true);

            node = std::make_shared<MapPlaneNode>(this, this->mapView, QuadTreePosition::topRight, level, x + 1, y);
//            node->initialize();
            node->scale.set(0.5, 1.0, 0.5);
            node->position.set(0.25, 0, -0.25);
            this->add(node);
            node->updateMatrix();
            node->updateMatrixWorld(true);

            node = std::make_shared<MapPlaneNode>(this, this->mapView, QuadTreePosition::bottomLeft, level, x, y + 1);
//            node->initialize();
            node->scale.set(0.5, 1.0, 0.5);
            node->position.set(-0.25, 0, 0.25);
            this->add(node);
            node->updateMatrix();
            node->updateMatrixWorld(true);

            node = std::make_shared<MapPlaneNode>(this, this->mapView, QuadTreePosition::bottomRight, level, x + 1, y + 1);
//            node->initialize();
            node->scale.set(0.5, 1.0, 0.5);
            node->position.set(0.25, 0, 0.25);
            this->add(node);
            node->updateMatrix();
            node->updateMatrixWorld(true);
        }

    };

}// namespace threepp

#endif//THREEPP_MAPPLANENODE_HPP
