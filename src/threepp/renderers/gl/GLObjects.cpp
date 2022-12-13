
#include "threepp/renderers/gl/GLObjects.hpp"

#include <glad/glad.h>

using namespace threepp;
using namespace threepp::gl;


std::shared_ptr<BufferGeometry> GLObjects::update(Object3D* object) {

    const int frame = info_.render.frame;

    auto geometry = object->geometry();
    geometries_.get(*geometry);

    // Update once per frame

    if (!updateMap_.count(geometry->id) || updateMap_[geometry->id] != frame) {

        geometries_.update(*geometry);

        updateMap_[geometry->id] = frame;
    }

    if (dynamic_cast<InstancedMesh*>(object)) {

        if (!object->hasEventListener("dispose", &onInstancedMeshDispose)) {

            object->addEventListener("dispose", &onInstancedMeshDispose);
        }

        auto o = dynamic_cast<InstancedMesh *>(object);

        attributes_.update(o->instanceMatrix.get(), GL_ARRAY_BUFFER);

        if (o->instanceColor != nullptr) {

            attributes_.update(o->instanceColor.get(), GL_ARRAY_BUFFER);
        }
    }

    return geometry;
}
