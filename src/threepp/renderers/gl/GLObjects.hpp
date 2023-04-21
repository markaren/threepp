// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLObjects.js

#ifndef THREEPP_GLOBJECTS_HPP
#define THREEPP_GLOBJECTS_HPP

#include <threepp/core/BufferGeometry.hpp>

#include <memory>

namespace threepp {

    class Object3D;

    namespace gl {

        class GLInfo;
        class GLGeometries;
        class GLAttributes;


        struct GLObjects {

            GLObjects(GLGeometries& geometries, GLAttributes& attributes, GLInfo& info);

            BufferGeometry* update(Object3D* object);

            void dispose();

            ~GLObjects();

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLOBJECTS_HPP
