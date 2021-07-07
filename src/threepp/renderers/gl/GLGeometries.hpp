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

                    scope_.attributes_.remove(attribute.get());
                    scope_.wireframeAttributes_.erase(geometry->id);
                }

                scope_.bindingStates_.releaseStatesOfGeometry(geometry);

                if (instanceof <InstancedBufferGeometry>(geometry)) {

                    dynamic_cast<InstancedBufferGeometry *>(geometry)->_maxInstanceCount = 0;
                }

                scope_.info_.memory.geometries--;
            }

        private:
            GLGeometries &scope_;
        };

        GLGeometries(GLAttributes &attributes, GLInfo &info, GLBindingStates &bindingStates)
            : info_(info), attributes_(attributes), bindingStates_(bindingStates), onGeometryDispose_(*this) {
        }

        BufferGeometry *get(BufferGeometry *geometry) {

            if (geometries_[geometry->id]) {
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

        void updateWireframeAttribute(BufferGeometry *geometry) {

            std::vector<int> indices;

            const auto geometryIndex = geometry->getIndex();
            const auto geometryPosition = geometry->getAttribute<float>("position");
            unsigned int version = 0;

            if ( geometryIndex != nullptr ) {

                const auto& array = geometryIndex->array();
                version = geometryIndex->version;

                for ( int i = 0, l = (int) array.size(); i < l; i += 3 ) {

                    const auto a = array[ i + 0 ];
                    const auto b = array[ i + 1 ];
                    const auto c = array[ i + 2 ];

                    indices.insert(indices.end(), { a, b, b, c, c, a });

                }

            } else {

                const auto& array = geometryPosition->array();
                version = geometryPosition->version;

                for ( int i = 0, l = ( (int) array.size() / 3 ) - 1; i < l; i += 3 ) {

                    const auto a = i + 0;
                    const auto b = i + 1;
                    const auto c = i + 2;

                    indices.insert(indices.end(), {a, b, b, c, c, a} );

                }

            }

            auto attribute = IntBufferAttribute::create(indices, 1);
            attribute->version = version;

            // Updating index buffer in VAO now. See WebGLBindingStates

            if ( wireframeAttributes_.count(geometry->id) ) {
                auto previousAttribute = wireframeAttributes_.at(geometry->id).get();
                attributes_.remove( previousAttribute );
            }

            wireframeAttributes_[geometry->id] = std::move(attribute);
        }

        IntBufferAttribute *getWireframeAttribute(BufferGeometry *geometry) {

            if (wireframeAttributes_.count(geometry->id)) {

                const auto &currentAttribute = wireframeAttributes_.at(geometry->id);

                if (geometry->hasIndex()) {

                    auto geometryIndex = geometry->getIndex();

                    // if the attribute is obsolete, create a new one

                    if (currentAttribute->version < geometryIndex->version) {

                        updateWireframeAttribute(geometry);
                    }
                }

            } else {

                updateWireframeAttribute(geometry);
            }

            return wireframeAttributes_.at(geometry->id).get();
        }

    private:
        GLInfo& info_;
        GLAttributes& attributes_;
        GLBindingStates& bindingStates_;

        OnGeometryDispose onGeometryDispose_;

        std::unordered_map<unsigned int, bool> geometries_;
        std::unordered_map<unsigned int, std::unique_ptr<IntBufferAttribute>> wireframeAttributes_;

    };

}// namespace threepp::gl

#endif//THREEPP_GLGEOMETRIES_HPP
