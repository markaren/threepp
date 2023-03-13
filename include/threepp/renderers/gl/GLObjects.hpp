// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLObjects.js

#ifndef THREEPP_GLOBJECTS_HPP
#define THREEPP_GLOBJECTS_HPP

#include "threepp/core/Object3D.hpp"

namespace threepp::gl {

    class GLGeometries;
    class GLAttributes;
    class GLInfo;

    struct GLObjects {

        GLObjects(GLGeometries& geometries, GLAttributes& attributes, GLInfo& info);

        BufferGeometry* update(Object3D* object);

        void dispose();

        ~GLObjects();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLOBJECTS_HPP
