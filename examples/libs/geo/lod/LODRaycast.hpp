// https://github.com/tentone/geo-three/blob/master/source/lod/LODRaycast.ts

#ifndef THREEPP_LODRAYCAST_HPP
#define THREEPP_LODRAYCAST_HPP

#include "threepp/core/Raycaster.hpp"
#include "threepp/math/MathUtils.hpp"

#include "geo/MapView.hpp"
#include "geo/lod/LODControl.hpp"
#include "geo/nodes/MapNode.hpp"

namespace threepp {

    class LODRaycast: public LODControl {

    public:
        int subdivisionRays = 1;
        float thresholdUp = 0.6;
        float thresholdDown = 0.15;
        bool powerDistance = false;
        bool scaleDistance = true;

        void updateLOD(MapView& view, Camera& camera, const GLRenderer& renderer, const Object3D& scene) override;

    private:
        Raycaster raycaster;
        Vector2 mouse;
    };

}// namespace threepp

#endif//THREEPP_LODRAYCAST_HPP
