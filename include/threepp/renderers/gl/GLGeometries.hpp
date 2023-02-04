// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLGeometries.js

#ifndef THREEPP_GLGEOMETRIES_HPP
#define THREEPP_GLGEOMETRIES_HPP

#include "GLAttributes.hpp"
#include "GLBindingStates.hpp"
#include "GLInfo.hpp"

#include "threepp/core/InstancedBufferGeometry.hpp"

#include <unordered_map>

namespace threepp::gl {

    struct GLGeometries {

        struct OnGeometryDispose : EventListener {

            explicit OnGeometryDispose(GLGeometries &scope);

            void onEvent(Event &event) override;

        private:
            GLGeometries &scope_;
        };

        GLGeometries(GLAttributes &attributes, GLInfo &info, GLBindingStates &bindingStates);

        void get(Object3D* object, BufferGeometry* geometry);

        void update(BufferGeometry *geometry);

        void updateWireframeAttribute(BufferGeometry *geometry);

        IntBufferAttribute *getWireframeAttribute(BufferGeometry *geometry);

    private:
        GLInfo& info_;
        GLAttributes& attributes_;
        GLBindingStates& bindingStates_;

        OnGeometryDispose onGeometryDispose_;

        std::unordered_map<BufferGeometry*, bool> geometries_;
        std::unordered_map<BufferGeometry*, std::unique_ptr<IntBufferAttribute>> wireframeAttributes_;

    };

}// namespace threepp::gl

#endif//THREEPP_GLGEOMETRIES_HPP
