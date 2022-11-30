// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLObjects.js

#ifndef THREEPP_GLOBJECTS_HPP
#define THREEPP_GLOBJECTS_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/objects/InstancedMesh.hpp"

#include "threepp/renderers/gl/GLAttributes.hpp"
#include "threepp/renderers/gl/GLGeometries.hpp"

namespace threepp::gl {

    struct GLObjects {

        struct OnInstancedMeshDispose : public EventListener {

            explicit OnInstancedMeshDispose(GLObjects &scope) : scope(scope) {}

            void onEvent(Event &event) override {
                auto instancedMesh = static_cast<InstancedMesh *>(event.target);

                instancedMesh->removeEventListener("dispose", this);

                scope.attributes_.remove(instancedMesh->instanceMatrix.get());

                if (instancedMesh->instanceColor) scope.attributes_.remove(instancedMesh->instanceColor.get());
            }

        private:
            GLObjects &scope;
        };

        GLObjects(GLGeometries &geometries, GLAttributes &attributes, GLInfo &info)
            : attributes_(attributes), geometries_(geometries), info_(info), onInstancedMeshDispose(*this) {}

        std::shared_ptr<BufferGeometry> update(const std::shared_ptr<Object3D> &object) {

            const int frame = info_.render.frame;

            auto geometry = object->geometry();
            geometries_.get(*geometry);

            // Update once per frame

            if (!updateMap_.count(geometry->id) || updateMap_[geometry->id] != frame) {

                geometries_.update(*geometry);

                updateMap_[geometry->id] = frame;
            }

            if (std::dynamic_pointer_cast<InstancedMesh>(object)) {

                if (!object->hasEventListener("dispose", &onInstancedMeshDispose)) {

                    object->addEventListener("dispose", &onInstancedMeshDispose);
                }

                auto o = dynamic_cast<InstancedMesh *>(object.get());

                attributes_.update(o->instanceMatrix.get(), GL_ARRAY_BUFFER);

                if (o->instanceColor != nullptr) {

                    attributes_.update(o->instanceColor.get(), GL_ARRAY_BUFFER);
                }
            }

            return geometry;
        }

        void dispose() {

            updateMap_.clear();
        }

    private:
        GLInfo &info_;
        GLGeometries &geometries_;
        GLAttributes &attributes_;

        OnInstancedMeshDispose onInstancedMeshDispose;

        std::unordered_map<unsigned int, int> updateMap_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLOBJECTS_HPP
