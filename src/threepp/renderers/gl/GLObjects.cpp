
#include "threepp/renderers/gl/GLObjects.hpp"

#include <glad/glad.h>

using namespace threepp;
using namespace threepp::gl;


BufferGeometry* GLObjects::update(Object3D* object) {

    const int frame = info_.render.frame;

    auto geometry = object->geometry();
    geometries_.get(object, geometry);

    // Update once per frame

    if (!updateMap_.count(geometry) || updateMap_[geometry] != frame) {

        geometries_.update(geometry);

        updateMap_[geometry] = frame;
    }

    auto instancedMesh = dynamic_cast<InstancedMesh*>(object);
    if (instancedMesh) {

        if (!object->hasEventListener("dispose", &onInstancedMeshDispose)) {

            object->addEventListener("dispose", &onInstancedMeshDispose);
        }

        attributes_.update(instancedMesh->instanceMatrix.get(), GL_ARRAY_BUFFER);

        if (instancedMesh->instanceColor != nullptr) {

            attributes_.update(instancedMesh->instanceColor.get(), GL_ARRAY_BUFFER);
        }
    }

    return geometry;
}
