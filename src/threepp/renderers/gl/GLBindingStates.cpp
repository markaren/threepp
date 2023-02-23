
#pragma warning(disable : 4312)

#include "threepp/renderers/gl/GLBindingStates.hpp"

#include "threepp/core/InterleavedBufferAttribute.hpp"
#include "threepp/renderers/gl/GLUtils.hpp"

using namespace threepp;
using namespace threepp::gl;

GLBindingStates::GLBindingStates(GLAttributes &attributes)
    : maxVertexAttributes_(glGetParameter(GL_MAX_VERTEX_ATTRIBS)),
      attributes_(attributes),
      defaultState_(createBindingState(std::nullopt)),
      currentState_(defaultState_) {
}

void GLBindingStates::setup(Object3D *object, Material *material, std::shared_ptr<GLProgram> &program, BufferGeometry *geometry, BufferAttribute *index) {

    auto state = getBindingState(geometry, program, material);

    if (!(currentState_ == state)) {

        currentState_ = state;
        bindVertexArrayObject(*currentState_->object);
    }

    bool updateBuffers = needsUpdate(geometry, index);

    if (updateBuffers) {
        saveCache(geometry, index);
    }

    if (object->as<InstancedMesh>()) {

        updateBuffers = true;
    }

    if (index) {

        attributes_.update(index, GL_ELEMENT_ARRAY_BUFFER);
    }

    if (updateBuffers) {

        setupVertexAttributes(object, material, program, geometry);

        if (index) {

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, attributes_.get(index).buffer);
        }
    }
}
GLuint GLBindingStates::createVertexArrayObject() const {

    GLuint vao;
    glGenVertexArrays(1, &vao);
    return vao;
}

void GLBindingStates::bindVertexArrayObject(GLuint vao) const {

    glBindVertexArray(vao);
}

void GLBindingStates::deleteVertexArrayObject(GLuint vao) {

    glDeleteVertexArrays(1, &vao);
}

std::shared_ptr<GLBindingState> GLBindingStates::getBindingState(BufferGeometry *geometry, std::shared_ptr<GLProgram> &program, Material *material) {

    bool wireframe = false;

    auto wm = material->as<MaterialWithWireframe>();
    if (wm) {
        wireframe = wm->wireframe;
    }

    auto &programMap = bindingStates[geometry->id];

    auto &stateMap = programMap[program->id];

    if (!stateMap.count(wireframe)) {

        stateMap[wireframe] = createBindingState(createVertexArrayObject());
    }

    return stateMap.at(wireframe);
}

std::shared_ptr<GLBindingState> GLBindingStates::createBindingState(std::optional<GLuint> vao) const {

    return std::make_shared<GLBindingState>(
            std::vector<int>(maxVertexAttributes_),
            std::vector<int>(maxVertexAttributes_),
            std::vector<int>(maxVertexAttributes_),
            vao);
}

bool GLBindingStates::needsUpdate(BufferGeometry *geometry, BufferAttribute *index) {

    const auto &cachedAttributes = currentState_->attributes;
    const auto &geometryAttributes = geometry->getAttributes();

    int attributesNum = 0;

    for (const auto &[key, value] : geometryAttributes) {

        if (!cachedAttributes.count(key)) return true;


        const auto &geometryAttribute = geometryAttributes.at(key);

        if (!cachedAttributes.count(key)) return true;

        const auto &cachedAttribute = cachedAttributes.at(key);

        if (cachedAttribute != geometryAttribute.get()) return true;

//          if (cachedAttribute.data != geometryAttribute.data) return true;

        ++attributesNum;
    }

    if (currentState_->attributesNum != attributesNum) return true;

    if (currentState_->index != index) return true;

    return false;
}

void GLBindingStates::saveCache(BufferGeometry *geometry, BufferAttribute *index) {

    std::unordered_map<std::string, BufferAttribute *> cache;
    const auto &attributes = geometry->getAttributes();
    int attributesNum = 0;

    for (const auto &[key, attribute] : attributes) {

        cache[key] = attribute.get();

        ++attributesNum;
    }

    currentState_->attributes = cache;
    currentState_->attributesNum = attributesNum;

    currentState_->index = index;
}

void GLBindingStates::initAttributes() {

    auto &newAttributes = currentState_->newAttributes;

    for (int &newAttribute : newAttributes) {

        newAttribute = 0;
    }
}

void GLBindingStates::enableAttribute(int attribute) {

    enableAttributeAndDivisor(attribute, 0);
}

void GLBindingStates::enableAttributeAndDivisor(int attribute, int meshPerAttribute) {

    auto &newAttributes = currentState_->newAttributes;
    auto &enabledAttributes = currentState_->enabledAttributes;
    auto &attributeDivisors = currentState_->attributeDivisors;

    newAttributes[attribute] = 1;

    if (enabledAttributes.at(attribute) == 0) {

        glEnableVertexAttribArray(attribute);
        enabledAttributes[attribute] = 1;
    }

    if (attributeDivisors[attribute] != meshPerAttribute) {

        glVertexAttribDivisor(attribute, meshPerAttribute);
        attributeDivisors[attribute] = meshPerAttribute;
    }
}

void GLBindingStates::disableUnusedAttributes() {

    const auto &newAttributes = currentState_->newAttributes;
    auto &enabledAttributes = currentState_->enabledAttributes;

    for (unsigned i = 0, il = enabledAttributes.size(); i < il; ++i) {

        if (enabledAttributes[i] != newAttributes[i]) {

            glDisableVertexAttribArray(i);
            enabledAttributes[i] = 0;
        }
    }
}

void GLBindingStates::vertexAttribPointer(GLuint index, GLint size, GLenum type, bool normalized, GLsizei stride, size_t offset) {

    if (type == GL_INT || type == GL_UNSIGNED_INT) {

        glVertexAttribIPointer(index, size, type, stride, (GLvoid *) offset);

    } else {

        glVertexAttribPointer(index, size, type, normalized, stride, (GLvoid *) offset);
    }
}

void GLBindingStates::setupVertexAttributes(Object3D *object, Material *material, std::shared_ptr<GLProgram> &program, BufferGeometry *geometry) {

    initAttributes();

    auto &geometryAttributes = geometry->getAttributes();

    const auto &programAttributes = program->getAttributes();

    auto &materialDefaultAttributeValues = material->defaultAttributeValues;

    for (const auto &[name, programAttribute] : programAttributes) {

        if (programAttribute >= 0) {

            if (geometryAttributes.count(name)) {

                auto &geometryAttribute = geometryAttributes.at(name);

                const auto normalized = geometryAttribute->normalized();
                const auto size = geometryAttribute->itemSize();

                auto attribute = attributes_.get(geometryAttribute.get());

                const auto buffer = attribute.buffer;
                const auto type = attribute.type;
                const auto bytesPerElement = attribute.bytesPerElement;

                if (dynamic_cast<InterleavedBufferAttribute *>(geometryAttribute.get())) {

                    auto attr = dynamic_cast<InterleavedBufferAttribute *>(geometryAttribute.get());
                    auto data = attr->data;
                    const auto stride = data->stride();
                    const auto offset = attr->offset;

                    if (false /*data && data.isInstancedInterleavedBuffer*/) {

                        //                        enableAttributeAndDivisor( programAttribute, data.meshPerAttribute );
                        //
                        //                        if ( geometry._maxInstanceCount === undefined ) {
                        //
                        //                            geometry._maxInstanceCount = data.meshPerAttribute * data.count;
                        //
                        //                        }

                    } else {

                        enableAttribute(programAttribute);
                    }

                    glBindBuffer(GL_ARRAY_BUFFER, buffer);
                    vertexAttribPointer(programAttribute, size, type, normalized, stride * bytesPerElement, offset * bytesPerElement);

                } else {

                    if (false /*geometryAttribute.isInstancedBufferAttribute*/) {

                        // TODO

                    } else {

                        enableAttribute(programAttribute);
                    }

                    glBindBuffer(GL_ARRAY_BUFFER, buffer);
                    vertexAttribPointer(programAttribute, size, type, normalized, 0, 0);
                }

            } else if (name == "instanceMatrix") {

                auto attribute = attributes_.get(object->as<InstancedMesh>()->instanceMatrix.get());

                auto buffer = attribute.buffer;
                auto type = attribute.type;

                enableAttributeAndDivisor(programAttribute + 0, 1);
                enableAttributeAndDivisor(programAttribute + 1, 1);
                enableAttributeAndDivisor(programAttribute + 2, 1);
                enableAttributeAndDivisor(programAttribute + 3, 1);

                glBindBuffer(GL_ARRAY_BUFFER, buffer);

                glVertexAttribPointer(programAttribute + 0, 4, type, false, 64, (void *) 0);
                glVertexAttribPointer(programAttribute + 1, 4, type, false, 64, (void *) 16);
                glVertexAttribPointer(programAttribute + 2, 4, type, false, 64, (void *) 32);
                glVertexAttribPointer(programAttribute + 3, 4, type, false, 64, (void *) 48);

            } else if (name == "instanceColor") {

                auto attribute = attributes_.get(object->as<InstancedMesh>()->instanceColor.get());

                auto buffer = attribute.buffer;
                auto type = attribute.type;

                enableAttributeAndDivisor( programAttribute, 1 );

                glBindBuffer( GL_ARRAY_BUFFER, buffer );

                glVertexAttribPointer( programAttribute, 3, type, false, 12, 0 );

            } else if (!materialDefaultAttributeValues.empty()) {

                    if (materialDefaultAttributeValues.count("name")) {

                        UniformValue &value = materialDefaultAttributeValues.at("name");

                        // TODO
                    }
                }
            }
        }

        disableUnusedAttributes();
    }

    void GLBindingStates::dispose() {

        reset();

        for (const auto &geometryId : bindingStates) {

            auto &programMap = bindingStates.at(geometryId.first);

            for (const auto &programId : programMap) {

                auto &stateMap = programMap.at(programId.first);

                for (const auto &wireframe : stateMap) {

                    deleteVertexArrayObject(*stateMap.at(wireframe.first)->object);

                    stateMap.erase(wireframe.first);
                }

                programMap.erase(programId.first);
            }

            bindingStates.erase(geometryId.first);
        }
    }

    void GLBindingStates::releaseStatesOfGeometry(BufferGeometry * geometry) {

        if (!bindingStates.count(geometry->id)) return;

        auto &programMap = bindingStates[geometry->id];

        for (const auto &programId : programMap) {

            auto &stateMap = programMap.at(programId.first);

            for (const auto &wireframe : stateMap) {

                deleteVertexArrayObject(*stateMap.at(wireframe.first)->object);

                stateMap.erase(wireframe.first);
            }

            programMap.erase(programId.first);
        }

        bindingStates.erase(geometry->id);
    }

    void GLBindingStates::releaseStatesOfProgram(GLProgram & program) {

        for (const auto &geometryId : bindingStates) {

            auto &programMap = bindingStates[geometryId.first];

            if (!programMap.count(program.id)) continue;

            auto &stateMap = programMap.at(program.id);

            for (const auto &wireframe : stateMap) {

                auto &value = stateMap.at(wireframe.first);
                deleteVertexArrayObject(*value->object);

                stateMap.erase(wireframe.first);
            }

            programMap.erase(program.id);
        }
    }

    void GLBindingStates::reset() {

        if (currentState_ == defaultState_) return;

        currentState_ = defaultState_;
        bindVertexArrayObject(currentState_->object.value_or(0));
    }
