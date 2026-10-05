// https://github.com/mrdoob/three.js/blob/r186/examples/jsm/environments/RoomEnvironment.js

#ifndef THREEPP_ROOMENVIRONMENT_HPP
#define THREEPP_ROOMENVIRONMENT_HPP

#include "threepp/scenes/Scene.hpp"

#include <memory>

namespace threepp {

    // A neutral studio: a white box room with a few boxes on the floor, one
    // point light and six emissive panels. It is not meant to be looked at but
    // turned into an environment map, for image-based lighting without an HDR
    // file (the same room three.js's editor and most of its examples use):
    //
    //   PMREMGenerator pmrem(renderer);
    //   RoomEnvironment room;
    //   scene->environment = pmrem.fromScene(room, 0.04f);
    //
    // Based on model-viewer's EnvironmentScene, as the three.js original is.
    class RoomEnvironment: public Scene {

    public:
        RoomEnvironment();

        [[nodiscard]] const std::string& type() const override;

        // Frees the geometry and materials the room owns.
        void dispose();

        static std::shared_ptr<RoomEnvironment> create();
    };

}// namespace threepp

#endif//THREEPP_ROOMENVIRONMENT_HPP
