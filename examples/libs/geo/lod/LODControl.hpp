// https://github.com/tentone/geo-three/blob/master/source/lod/LODControl.ts

#ifndef THREEPP_LODCONTROL_HPP
#define THREEPP_LODCONTROL_HPP

#include "threepp/cameras/Camera.hpp"
#include "threepp/renderers/GLRenderer.hpp"

namespace threepp {

    class MapView;

    class LODControl {

    public:
        virtual void updateLOD(MapView& view, Camera& camera, const GLRenderer& renderer, const Object3D& scene) = 0;

        virtual ~LODControl() = default;
    };

}// namespace threepp

#endif//THREEPP_LODCONTROL_HPP
