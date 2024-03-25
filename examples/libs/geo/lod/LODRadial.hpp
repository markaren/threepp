// https://github.com/tentone/geo-three/blob/master/source/lod/LODRadial.ts

#ifndef THREEPP_LODRADIAL_HPP
#define THREEPP_LODRADIAL_HPP

#include "threepp/core/Object3D.hpp"

#include "geo/MapView.hpp"
#include "geo/lod/LODControl.hpp"
#include "geo/nodes/MapNode.hpp"

#include <cmath>
#include <iostream>

namespace threepp {

    class LODRadial: public LODControl {

    public:
        float subdivideDistance;
        float simplifyDistance;

        explicit LODRadial(float subdivideDistance = 50, float simplifyDistance = 300)
            : subdivideDistance(subdivideDistance), simplifyDistance(simplifyDistance) {}

        void updateLOD(MapView& view, Camera& camera, const GLRenderer& renderer, const Object3D& scene) override {

            camera.getWorldPosition(pov);

            view.children[0]->traverseType<MapNode>([&](MapNode& node) {
                node.getWorldPosition(position);

                auto distance = pov.distanceTo(position);
                distance /= std::pow(2.f, view.getProvider()->maxZoom - node.getLevel());
                //
                if (distance < this->subdivideDistance) {
                    node.subdivide();
                } else if (distance > this->simplifyDistance && node.parentNode) {
                    node.parentNode->simplify();
                }
            });
        }

    protected:
        Vector3 pov;
        Vector3 position;
    };

}// namespace threepp

#endif//THREEPP_LODRADIAL_HPP
