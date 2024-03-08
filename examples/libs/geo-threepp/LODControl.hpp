
#ifndef THREEPP_LODCONTROL_HPP
#define THREEPP_LODCONTROL_HPP

#include "MapView.hpp"
#include "threepp/cameras/Camera.hpp"
#include "threepp/renderers/GLRenderer.hpp"

namespace threepp {

    class LODControl {

    public:
        virtual void updateLOD(MapView& view, Camera& camera, GLRenderer& renderer, Object3D& scene) = 0;

        virtual ~LODControl() = default;
    };

}// namespace threepp

#endif//THREEPP_LODCONTROL_HPP
