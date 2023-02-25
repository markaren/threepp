
#include "threepp/renderers/gl/GLGeometries.hpp"

#include <glad/glad.h>

using namespace threepp;
using namespace threepp::gl;

GLGeometries::GLGeometries(GLAttributes& attributes, GLInfo& info, GLBindingStates& bindingStates)
    : info_(info), attributes_(attributes), bindingStates_(bindingStates), onGeometryDispose_(*this) {
}

void GLGeometries::get(Object3D* object, BufferGeometry* geometry) {

    if (geometries_.count(geometry) && geometries_.at(geometry)) return;

    geometry->addEventListener("dispose", &onGeometryDispose_);

    geometries_[geometry] = true;

    ++info_.memory.geometries;
}

void GLGeometries::update(BufferGeometry* geometry) {

    auto& geometryAttributes = geometry->getAttributes();

    // Updating index buffer in VAO now. See WebGLBindingStates.

    for (auto& [name, value] : geometryAttributes) {

        attributes_.update(value.get(), GL_ARRAY_BUFFER);
    }
}

void GLGeometries::updateWireframeAttribute(BufferGeometry* geometry) {

    std::vector<unsigned int> indices;

    const auto geometryIndex = geometry->getIndex();
    const auto geometryPosition = geometry->getAttribute<float>("position");
    unsigned int version = 0;

    if (geometryIndex != nullptr) {

        const auto& array = geometryIndex->array();
        version = geometryIndex->version;

        for (unsigned i = 0, l = array.size(); i < l; i += 3) {

            const auto a = array[i + 0];
            const auto b = array[i + 1];
            const auto c = array[i + 2];

            indices.insert(indices.end(), {a, b, b, c, c, a});
        }

    } else {

        const auto& array = geometryPosition->array();
        version = geometryPosition->version;

        for (unsigned i = 0, l = array.size() / 3 - 1; i < l; i += 3) {

            const auto a = i + 0;
            const auto b = i + 1;
            const auto c = i + 2;

            indices.insert(indices.end(), {a, b, b, c, c, a});
        }
    }

    auto attribute = IntBufferAttribute::create(indices, 1);
    attribute->version = version;

    // Updating index buffer in VAO now. See WebGLBindingStates

    if (wireframeAttributes_.count(geometry)) {
        auto previousAttribute = wireframeAttributes_.at(geometry).get();
        attributes_.remove(previousAttribute);
    }

    wireframeAttributes_[geometry] = std::move(attribute);
}

IntBufferAttribute* GLGeometries::getWireframeAttribute(BufferGeometry* geometry) {

    if (wireframeAttributes_.count(geometry)) {

        const auto& currentAttribute = wireframeAttributes_.at(geometry);

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

    return wireframeAttributes_.at(geometry).get();
}


GLGeometries::OnGeometryDispose::OnGeometryDispose(GLGeometries& scope): scope_(scope) {}

void GLGeometries::OnGeometryDispose::onEvent(Event& event) {

    auto geometry = static_cast<BufferGeometry*>(event.target);

    if (geometry->hasIndex()) {

        scope_.attributes_.remove(geometry->getIndex());
    }

    for (const auto& [name, value] : geometry->getAttributes()) {

        scope_.attributes_.remove(value.get());
    }

    geometry->removeEventListener("dispose", this);

    scope_.geometries_.erase(geometry);


    if (scope_.wireframeAttributes_.count(geometry)) {

        const auto& attribute = scope_.wireframeAttributes_.at(geometry);

        scope_.attributes_.remove(attribute.get());
        scope_.wireframeAttributes_.erase(geometry);
    }

    scope_.bindingStates_.releaseStatesOfGeometry(geometry);

    auto ig = dynamic_cast<InstancedBufferGeometry*>(geometry);
    if (ig) {
        ig->_maxInstanceCount = 0;
    }

    --scope_.info_.memory.geometries;
}
