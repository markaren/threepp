// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLGeometries.js

#ifndef THREEPP_GLGEOMETRIES_HPP
#define THREEPP_GLGEOMETRIES_HPP

#include "threepp/core/BufferGeometry.hpp"

#include <memory>

namespace threepp {

    class Object3D;

    namespace gl {

        class GLInfo;
        class GLAttributes;
        class GLBindingStates;

        class GLGeometries {
        public:
            GLGeometries(GLAttributes& attributes, GLInfo& info, GLBindingStates& bindingStates);

            void get(Object3D* object, BufferGeometry* geometry);

            void update(BufferGeometry* geometry);

            void updateWireframeAttribute(BufferGeometry* geometry);

            IntBufferAttribute* getWireframeAttribute(BufferGeometry* geometry);

            ~GLGeometries();

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLGEOMETRIES_HPP
