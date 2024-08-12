
#include "threepp/renderers/gl/GLObjects.hpp"

#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/renderers/gl/GLAttributes.hpp"
#include "threepp/renderers/gl/GLGeometries.hpp"
#include "threepp/renderers/gl/GLInfo.hpp"

#ifndef EMSCRIPTEN
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

using namespace threepp;
using namespace threepp::gl;


struct GLObjects::Impl {

    struct OnInstancedMeshDispose: public EventListener {

        explicit OnInstancedMeshDispose(GLObjects::Impl* scope): scope(scope) {}

        void onEvent(Event& event) override {
            auto instancedMesh = static_cast<InstancedMesh*>(event.target);

            instancedMesh->removeEventListener("dispose", *this);

            scope->attributes_.remove(instancedMesh->instanceMatrix());

            if (instancedMesh->instanceColor()) scope->attributes_.remove(instancedMesh->instanceColor());
        }

    private:
        GLObjects::Impl* scope;
    };

    GLInfo& info_;
    GLGeometries& geometries_;
    GLAttributes& attributes_;

    OnInstancedMeshDispose onInstancedMeshDispose;

    std::unordered_map<BufferGeometry*, size_t> updateMap_;

    Impl(GLGeometries& geometries, GLAttributes& attributes, GLInfo& info)
        : attributes_(attributes),
          geometries_(geometries), info_(info),
          onInstancedMeshDispose(this) {}

    BufferGeometry* update(Object3D* object) {

        const auto frame = info_.render.frame;

        const auto geometry = object->geometry().get();
        geometries_.get(object, geometry);

        // Update once per frame

        if (!updateMap_.contains(geometry) || updateMap_[geometry] != frame) {

            geometries_.update(geometry);

            updateMap_[geometry] = frame;
        }

        if (auto instancedMesh = object->as<InstancedMesh>()) {

            if (!object->hasEventListener("dispose", onInstancedMeshDispose)) {

                object->addEventListener("dispose", onInstancedMeshDispose);
            }

            attributes_.update(instancedMesh->instanceMatrix(), GL_ARRAY_BUFFER);

            if (instancedMesh->instanceColor() != nullptr) {

                attributes_.update(instancedMesh->instanceColor(), GL_ARRAY_BUFFER);
            }
        }

        return geometry;
    }

    void dispose() {

        updateMap_.clear();
    }
};

BufferGeometry* GLObjects::update(Object3D* object) {

    return pimpl_->update(object);
}

gl::GLObjects::GLObjects(GLGeometries& geometries, GLAttributes& attributes, GLInfo& info)
    : pimpl_(std::make_unique<Impl>(geometries, attributes, info)) {}

void gl::GLObjects::dispose() {

    pimpl_->dispose();
}

GLObjects::~GLObjects() = default;
