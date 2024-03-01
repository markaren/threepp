
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

    GLInfo& info_;
    GLGeometries& geometries_;
    GLAttributes& attributes_;

    std::unordered_map<BufferGeometry*, size_t> updateMap_;

    ///  Track subscriptions to instanceMeshDispose
    std::unordered_map<Object3D*, Subscription> instanceMeshDisposeSubscriptions_;

    Impl(GLGeometries& geometries, GLAttributes& attributes, GLInfo& info)
        : attributes_(attributes),
          geometries_(geometries), info_(info) {}

    BufferGeometry* update(Object3D* object) {

        const auto frame = info_.render.frame;

        auto geometry = object->geometry();
        geometries_.get(object, geometry);

        // Update once per frame

        if (!updateMap_.count(geometry) || updateMap_[geometry] != frame) {

            geometries_.update(geometry);

            updateMap_[geometry] = frame;
        }

        if (auto instancedMesh = object->as<InstancedMesh>()) {

            if (instanceMeshDisposeSubscriptions_.find(object) == instanceMeshDisposeSubscriptions_.end()) {

                auto onInstanceMeshDispose = [this,object](Event& event) {
                    auto instancedMesh = static_cast<InstancedMesh*>(event.target);
                    this->attributes_.remove(instancedMesh->instanceMatrix());
                    if (instancedMesh->instanceColor()) this->attributes_.remove(instancedMesh->instanceColor());
                    // Remove our subscription
                    this->instanceMeshDisposeSubscriptions_.erase(object);
                };

                instanceMeshDisposeSubscriptions_.insert({object, object->addEventListener("dispose", onInstanceMeshDispose)});
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
