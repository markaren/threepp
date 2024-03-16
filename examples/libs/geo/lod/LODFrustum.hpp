// https://github.com/tentone/geo-three/blob/master/source/lod/LODFrustum.ts

#ifndef THREEPP_LODFRUSTUM_HPP
#define THREEPP_LODFRUSTUM_HPP

#include "threepp/math/Frustum.hpp"
#include "threepp/math/Matrix4.hpp"

#include "geo/lod/LODRadial.hpp"
#include "geo/MapView.hpp"

namespace threepp {

    class LODFrustum: public LODRadial {

    public:

        explicit LODFrustum(float subdivideDistance = 150, float simplifyDistance = 400)
            : LODRadial(subdivideDistance, simplifyDistance) {}

        void updateLOD(MapView& view, Camera& camera, const GLRenderer& renderer, const Object3D& scene) override {

            projection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
            frustum.setFromProjectionMatrix(projection);
            camera.getWorldPosition(pov);

            view.children[0]->traverseType<MapNode>([&](MapNode& node) {
                node.getWorldPosition(position);

                auto distance = pov.distanceTo(position);
                distance /= std::pow(2, view.getProvider()->maxZoom - node.getLevel());

                const auto inFrustum = this->pointOnly ? frustum.containsPoint(position) : frustum.intersectsObject(node);

                if (distance < this->subdivideDistance && inFrustum) {
                    node.subdivide();
                } else if (distance > this->simplifyDistance && node.parentNode) {
                    node.parentNode->simplify();
                }
            });
        }

    private:
        bool pointOnly = true;

        Matrix4 projection;
        Frustum frustum;
    };

}// namespace threepp

#endif//THREEPP_LODFRUSTUM_HPP
