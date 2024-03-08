
#ifndef THREEPP_LODRADIAL_HPP
#define THREEPP_LODRADIAL_HPP

#include "LODControl.hpp"
#include "MapNode.hpp"

#include <cmath>

namespace threepp {

    class MapNode;

    class LODRadial: public LODControl {

    public:
        LODRadial(float subdivideDistance = 50, float simplifyDistance = 300)
            : subdivideDistance(subdivideDistance), simplifyDistance(simplifyDistance) {}

        void updateLOD(MapView& view, Camera& camera, GLRenderer& renderer, Object3D& scene) {

            camera.getWorldPosition(pov);

            view.children[0]->traverseType<MapNode>([&](MapNode& node) {
                                                                         node.getWorldPosition(position);

                                                                         auto distance = pov.distanceTo(position);
                                                                         distance /= std::pow(2, view.provider->maxZoom - node.getLevel());
                //
                                                                         if (distance < this->subdivideDistance)
                                                                         {
                //                                                             node.subdivide();
                                                                         }
//                                                                         else if (distance > this.simplifyDistance && node.parentNode)
//                                                                         {
                //                                                             node.parentNode.simplify();
//                                                                         }
            });
        }

    private:
        float subdivideDistance;
        float simplifyDistance;

        Vector3 pov;
        Vector3 position;
    };

}// namespace threepp

#endif//THREEPP_LODRADIAL_HPP
