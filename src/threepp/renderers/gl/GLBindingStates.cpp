
#include "threepp/renderers/gl/GLBindingStates.hpp"

#include "threepp/renderers/gl/GLUtils.hpp"

#include "threepp/core/InterleavedBufferAttribute.hpp"
#include "threepp/materials/materials.hpp"
#include "threepp/objects/InstancedMesh.hpp"

#if EMSCRIPTEN
#include <GLES3/gl32.h>
#endif

using namespace threepp;
using namespace threepp::gl;

typedef std::unordered_map<bool, std::shared_ptr<GLBindingState>> StateMap;
typedef std::unordered_map<int, StateMap> ProgramMap;

struct GLBindingStates::Impl {

    GLAttributes& attributes_;
    unsigned int maxVertexAttributes_;

    const std::shared_ptr<GLBindingState> defaultState_;
    std::shared_ptr<GLBindingState> currentState_;

    std::unordered_map<unsigned int, ProgramMap> bindingStates;

    explicit Impl(GLAttributes& attributes)
        : maxVertexAttributes_(glGetParameteri(GL_MAX_VERTEX_ATTRIBS)),
          attributes_(attributes),
          defaultState_(createBindingState(std::nullopt)),
          currentState_(defaultState_) {}


    void setup(Object3D* object, Material* material, GLProgram* program, BufferGeometry* geometry, BufferAttribute* index) {

        auto state = getBindingState(geometry, program, material);

        if (!(currentState_ == state)) {

            currentState_ = state;
            bindVertexArrayObject(*currentState_->object);
        }

        bool updateBuffers = needsUpdate(geometry, index);

        if (updateBuffers) {
            saveCache(geometry, index);
        }

        if (object->is<InstancedMesh>()) {

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

    [[nodiscard]] GLuint createVertexArrayObject() const {

        GLuint vao;
        glGenVertexArrays(1, &vao);
        return vao;
    }

    void bindVertexArrayObject(GLuint vao) const {

        glBindVertexArray(vao);
    }

    void deleteVertexArrayObject(GLuint vao) {

        glDeleteVertexArrays(1, &vao);
    }

    std::shared_ptr<GLBindingState> getBindingState(BufferGeometry* geometry, GLProgram* program, Material* material) {

        bool wireframe = false;

        auto wm = material->as<MaterialWithWireframe>();
        if (wm) {
            wireframe = wm->wireframe;
        }

        auto& programMap = bindingStates[geometry->id];

        auto& stateMap = programMap[program->id];

        if (!stateMap.contains(wireframe)) {

            stateMap[wireframe] = createBindingState(createVertexArrayObject());
        }

        return stateMap.at(wireframe);
    }

    [[nodiscard]] std::shared_ptr<GLBindingState> createBindingState(std::optional<GLuint> vao) const {

        return std::make_shared<GLBindingState>(
                std::vector<int>(maxVertexAttributes_),
                std::vector<int>(maxVertexAttributes_),
                std::vector<int>(maxVertexAttributes_),
                vao);
    }

    bool needsUpdate(BufferGeometry* geometry, BufferAttribute* index) const {

        const auto& cachedAttributes = currentState_->attributes;
        const auto& geometryAttributes = geometry->getAttributes();

        int attributesNum = 0;

        for (const auto& [key, value] : geometryAttributes) {

            if (!cachedAttributes.contains(key)) return true;


            const auto& geometryAttribute = geometryAttributes.at(key);

            if (!cachedAttributes.contains(key)) return true;

            const auto& cachedAttribute = cachedAttributes.at(key);

            if (cachedAttribute != geometryAttribute.get()) return true;

            //          if (cachedAttribute.data != geometryAttribute.data) return true;

            ++attributesNum;
        }

        if (currentState_->attributesNum != attributesNum) return true;

        if (currentState_->index != index) return true;

        return false;
    }

    void saveCache(BufferGeometry* geometry, BufferAttribute* index) const {

        std::unordered_map<std::string, BufferAttribute*> cache;
        const auto& attributes = geometry->getAttributes();
        int attributesNum = 0;

        for (const auto& [key, attribute] : attributes) {

            cache[key] = attribute.get();

            ++attributesNum;
        }

        currentState_->attributes = cache;
        currentState_->attributesNum = attributesNum;

        currentState_->index = index;
    }

    void initAttributes() const {

        auto& newAttributes = currentState_->newAttributes;

        for (int& newAttribute : newAttributes) {

            newAttribute = 0;
        }
    }

    void enableAttribute(int attribute) {

        enableAttributeAndDivisor(attribute, 0);
    }

    void enableAttributeAndDivisor(int attribute, int meshPerAttribute) const {

        auto& newAttributes = currentState_->newAttributes;
        auto& enabledAttributes = currentState_->enabledAttributes;
        auto& attributeDivisors = currentState_->attributeDivisors;

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

    void disableUnusedAttributes() const {

        const auto& newAttributes = currentState_->newAttributes;
        auto& enabledAttributes = currentState_->enabledAttributes;

        for (unsigned i = 0, il = enabledAttributes.size(); i < il; ++i) {

            if (enabledAttributes[i] != newAttributes[i]) {

                glDisableVertexAttribArray(i);
                enabledAttributes[i] = 0;
            }
        }
    }

    void vertexAttribPointer(GLuint index, GLint size, GLenum type, bool normalized, GLsizei stride, size_t offset) {

        if (type == GL_INT || type == GL_UNSIGNED_INT) {

            glVertexAttribIPointer(index, size, type, stride, (GLvoid*) offset);

        } else {

            glVertexAttribPointer(index, size, type, normalized, stride, (GLvoid*) offset);
        }
    }

    void setupVertexAttributes(Object3D* object, Material* material, GLProgram* program, BufferGeometry* geometry) {

        initAttributes();

        auto& geometryAttributes = geometry->getAttributes();

        const auto& programAttributes = program->getAttributes();

        auto& materialDefaultAttributeValues = material->defaultAttributeValues;

        for (const auto& [name, programAttribute] : programAttributes) {

            if (programAttribute >= 0) {

                if (geometryAttributes.contains(name)) {

                    auto& geometryAttribute = geometryAttributes.at(name);

                    const auto normalized = geometryAttribute->normalized();
                    const auto size = geometryAttribute->itemSize();

                    auto attribute = attributes_.get(geometryAttribute.get());

                    const auto buffer = attribute.buffer;
                    const auto type = attribute.type;
                    const auto bytesPerElement = attribute.bytesPerElement;

                    if (dynamic_cast<InterleavedBufferAttribute*>(geometryAttribute.get())) {

                        auto attr = dynamic_cast<InterleavedBufferAttribute*>(geometryAttribute.get());
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

                    auto attribute = attributes_.get(object->as<InstancedMesh>()->instanceMatrix());

                    auto buffer = attribute.buffer;
                    auto type = attribute.type;

                    enableAttributeAndDivisor(programAttribute + 0, 1);
                    enableAttributeAndDivisor(programAttribute + 1, 1);
                    enableAttributeAndDivisor(programAttribute + 2, 1);
                    enableAttributeAndDivisor(programAttribute + 3, 1);

                    glBindBuffer(GL_ARRAY_BUFFER, buffer);

                    glVertexAttribPointer(programAttribute + 0, 4, type, false, 64, (void*) 0);
                    glVertexAttribPointer(programAttribute + 1, 4, type, false, 64, (void*) 16);
                    glVertexAttribPointer(programAttribute + 2, 4, type, false, 64, (void*) 32);
                    glVertexAttribPointer(programAttribute + 3, 4, type, false, 64, (void*) 48);

                } else if (name == "instanceColor") {

                    auto attribute = attributes_.get(object->as<InstancedMesh>()->instanceColor());

                    auto buffer = attribute.buffer;
                    auto type = attribute.type;

                    enableAttributeAndDivisor(programAttribute, 1);

                    glBindBuffer(GL_ARRAY_BUFFER, buffer);

                    glVertexAttribPointer(programAttribute, 3, type, false, 12, 0);

                } else if (!materialDefaultAttributeValues.empty()) {

                    if (materialDefaultAttributeValues.contains("name")) {

                        // UniformValue& value = materialDefaultAttributeValues.at("name");

                        // TODO
                    }
                }
            }
        }

        disableUnusedAttributes();
    }

    void dispose() {

        reset();

        for (auto geoIt = bindingStates.begin(); geoIt != bindingStates.end();) {
            auto& programMap = geoIt->second;

            for (auto progIt = programMap.begin(); progIt != programMap.end();) {
                auto& stateMap = progIt->second;

                for (auto it = stateMap.begin(); it != stateMap.end();) {
                    // const auto& wireframe = *it;

                    deleteVertexArrayObject(*it->second->object);

                    it = stateMap.erase(it);
                }

                progIt = programMap.erase(progIt);
            }

            geoIt = bindingStates.erase(geoIt);
        }
    }

    void releaseStatesOfGeometry(BufferGeometry* geometry) {

        if (!bindingStates.contains(geometry->id)) return;

        ProgramMap& programMap = bindingStates[geometry->id];

        for (const auto& programId : programMap) {

            StateMap& stateMap = programMap.at(programId.first);

            for (const auto& wireframe : stateMap) {

                deleteVertexArrayObject(*stateMap.at(wireframe.first)->object);
            }

            stateMap.clear();
        }

        programMap.clear();

        bindingStates.erase(geometry->id);
    }

    void releaseStatesOfProgram(GLProgram& program) {

        for (const auto& geometryId : bindingStates) {

            auto& programMap = bindingStates[geometryId.first];

            if (!programMap.contains(program.id)) continue;

            auto& stateMap = programMap.at(program.id);

            for (const auto& wireframe : stateMap) {

                auto& value = stateMap.at(wireframe.first);
                deleteVertexArrayObject(*value->object);
            }
            stateMap.clear();

            programMap.erase(program.id);
        }
    }

    void reset() {

        if (currentState_ == defaultState_) return;

        currentState_ = defaultState_;
        bindVertexArrayObject(currentState_->object.value_or(0));
    }
};

GLBindingStates::GLBindingStates(GLAttributes& attributes)
    : pimpl_(std::make_unique<Impl>(attributes)) {
}

void GLBindingStates::setup(Object3D* object, Material* material, GLProgram* program, BufferGeometry* geometry, BufferAttribute* index) {

    pimpl_->setup(object, material, program, geometry, index);
}

void GLBindingStates::initAttributes() {

    pimpl_->initAttributes();
}

void GLBindingStates::enableAttribute(int attribute) {

    pimpl_->enableAttribute(attribute);
}

void GLBindingStates::disableUnusedAttributes() {

    pimpl_->disableUnusedAttributes();
}

void GLBindingStates::dispose() {

    pimpl_->dispose();
}

void GLBindingStates::releaseStatesOfGeometry(BufferGeometry* geometry) {

    pimpl_->releaseStatesOfGeometry(geometry);
}

void GLBindingStates::releaseStatesOfProgram(GLProgram& program) {

    pimpl_->releaseStatesOfProgram(program);
}

void GLBindingStates::reset() {

    pimpl_->reset();
}

GLBindingStates::~GLBindingStates() = default;
