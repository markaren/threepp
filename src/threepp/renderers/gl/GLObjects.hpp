// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLObjects.js

#ifndef THREEPP_GLOBJECTS_HPP
#define THREEPP_GLOBJECTS_HPP

#include "threepp/core/Object3D.hpp"

#include "threepp/renderers/gl/GLAttributes.hpp"
#include "threepp/renderers/gl/GLGeometries.hpp"

namespace threepp::gl {

    struct GLObjects {

        GLObjects(GLGeometries geometries, GLAttributes attributes, GLInfo &info)
            : attributes_(attributes), geometries_(geometries), info_(info) {}

        BufferGeometry *update(Object3D *object) {

            const auto frame = info_.render.frame;

            auto geometry = object->geometry();
            auto buffergeometry = geometries_.get(object, geometry);

            // Update once per frame

            if (!updateMap_.count(buffergeometry) || updateMap_[buffergeometry] != frame) {

                geometries_.update(buffergeometry);

                updateMap_[buffergeometry] = frame;
            }

            return buffergeometry;
        }

        void dispose() {

            updateMap_.clear();
        }

    private:
        GLInfo info_;
        GLGeometries geometries_;
        GLAttributes attributes_;

        std::unordered_map<BufferGeometry *, int> updateMap_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLOBJECTS_HPP
