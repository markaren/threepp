// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLGeometries.js

#ifndef THREEPP_GLGEOMETRIES_HPP
#define THREEPP_GLGEOMETRIES_HPP

#include "GLAttributes.hpp"
#include "GLBindingStates.hpp"
#include "GLInfo.hpp"

#include <unordered_map>

namespace threepp::gl {

    struct GLGeometries {

        struct OnGeometryDispose : EventListener {

            explicit OnGeometryDispose(GLGeometries &scope) : scope_(scope) {}

            void onEvent(Event &event) override {

                auto geometry = static_cast<BufferGeometry *>(event.target);

                if (geometry->hasIndex()) {

                    scope_.attributes_.remove(geometry->getIndex());
                }

                for (const auto &[name, value] : geometry->getAttributes()) {

                    scope_.attributes_.remove(value.get());
                }

                geometry->removeEventListener("dispose", this);

                scope_.geometries_.erase(geometry->id);


                if (scope_.wireframeAttributes_.count(geometry->id)) {

                    const auto &attribute = scope_.wireframeAttributes_.at(geometry->id);

                    scope_.attributes_.remove(attribute);
                    scope_.wireframeAttributes_.erase(geometry->id);
                }

                scope_.bindingStates_.releaseStatesOfGeometry(geometry);

                //                if ( geometry.isInstancedBufferGeometry === true ) {
                //
                //                    delete geometry._maxInstanceCount;
                //
                //                }

                scope_.info_.memory.geometries--;
            }

        private:
            GLGeometries &scope_;
        };

        GLGeometries(GLAttributes &attributes, GLInfo &info, GLBindingStates &bindingStates)
            : info_(info), attributes_(attributes), bindingStates_(bindingStates), onGeometryDispose_(*this) {
        }

        BufferGeometry *get(Object3D *object, BufferGeometry *geometry) {

            if (geometries_.count(geometry->id)) {
                return geometry;
            }

            geometry->addEventListener("dispose", &onGeometryDispose_);

            geometries_[geometry->id] = true;

            info_.memory.geometries++;

            return geometry;
        }

        void update(BufferGeometry *geometry) {

            auto &geometryAttributes = geometry->getAttributes();

            // Updating index buffer in VAO now. See WebGLBindingStates.

            for (auto &[name, value] : geometryAttributes) {

                attributes_.update(value.get(), GL_ARRAY_BUFFER);
            }
        }


        IntBufferAttribute *getWireframeAttribute(BufferGeometry *geometry) {

            if (wireframeAttributes_.count(geometry->id)) {

                const auto &currentAttribute = wireframeAttributes_.at(geometry->id);

                if (geometry->hasIndex()) {

                    auto geometryIndex = geometry->getIndex();

                    // if the attribute is obsolete, create a new one

                    if (currentAttribute->version() < geometryIndex->version()) {

                        updateWireframeAttribute(geometry);
                    }
                }

            } else {

                updateWireframeAttribute(geometry);
            }

            return wireframeAttributes_.at(geometry->id);
        }

    private:
        GLInfo info_;
        GLAttributes attributes_;
        GLBindingStates bindingStates_;

        OnGeometryDispose onGeometryDispose_;

        std::unordered_map<unsigned int, bool> geometries_;
        std::unordered_map<unsigned int, IntBufferAttribute *> wireframeAttributes_;


        void updateWireframeAttribute(BufferGeometry *geometry) {
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLGEOMETRIES_HPP
