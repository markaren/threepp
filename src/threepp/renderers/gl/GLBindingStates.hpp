// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBindingStates.js

#ifndef THREEPP_GLBINDINGSTATES_HPP
#define THREEPP_GLBINDINGSTATES_HPP

#include "GLAttributes.hpp"
#include "GLCapabilities.hpp"
#include "GLProgram.hpp"
#include "glHelper.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/materials.hpp"

#include "threepp/utils/InstanceOf.hpp"

#include <optional>
#include <threepp/core/InstancedBufferAttribute.hpp>
#include <threepp/objects/InstancedMesh.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace threepp::gl {

    struct GLBindingState {

        std::vector<int> newAttributes;
        std::vector<int> enabledAttributes;
        std::vector<int> attributeDivisors;
        std::optional<GLuint> object;
        std::unordered_map<std::string, BufferAttribute *> attributes;
        BufferAttribute *index = nullptr;

        int attributesNum = 0;

        GLBindingState(
                std::vector<int> newAttributes,
                std::vector<int> enabledAttributes,
                std::vector<int> attributeDivisors,
                const std::optional<GLuint> &object)
            : newAttributes(std::move(newAttributes)),
              enabledAttributes(std::move(enabledAttributes)),
              attributeDivisors(std::move(attributeDivisors)),
              object(object) {}
    };


    typedef std::unordered_map<bool, std::shared_ptr<GLBindingState>> StateMap;
    typedef std::unordered_map<int, StateMap> ProgramMap;

    struct GLBindingStates {

        explicit GLBindingStates(GLAttributes &attributes)
            : maxVertexAttributes_(glGetParameter(GL_MAX_VERTEX_ATTRIBS)),
              attributes_(attributes),
              defaultState_(createBindingState(std::nullopt)),
              currentState_(defaultState_) {
        }

        void setup(Object3D *object, Material *material, std::shared_ptr<GLProgram> &program, BufferGeometry *geometry, BufferAttribute *index) {

            auto state = getBindingState(geometry, program, material);

            if (!(currentState_ == state)) {

                currentState_ = state;
                bindVertexArrayObject(*currentState_->object);
            }

            bool updateBuffers = needsUpdate(geometry, index);

            if (updateBuffers) saveCache(geometry, index);

            if (instanceof <InstancedMesh>(object)) {

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
            glCreateVertexArrays(1, &vao);
            return vao;
        }

        void bindVertexArrayObject(const GLuint vao) const {
            glBindVertexArray(vao);
        }

        void deleteVertexArrayObject(GLuint vao) {
            glDeleteVertexArrays(1, &vao);
        }

        std::shared_ptr<GLBindingState> getBindingState(BufferGeometry *geometry, std::shared_ptr<GLProgram> &program, Material *material) {

            bool wireframe = false;

            if (instanceof <MaterialWithWireframe>(material)) {
                wireframe = dynamic_cast<MaterialWithWireframe *>(material)->wireframe;
            }

            auto& programMap = bindingStates[geometry->id];

            auto& stateMap = programMap[program->id];

            if (!stateMap.count(wireframe)) {

                stateMap[wireframe] = createBindingState(createVertexArrayObject());
            }

            return stateMap[wireframe];
        }

        std::shared_ptr<GLBindingState> createBindingState(std::optional<GLuint> vao) const {

            return std::make_shared<GLBindingState>(
                    std::vector<int>(maxVertexAttributes_),
                    std::vector<int>(maxVertexAttributes_),
                    std::vector<int>(maxVertexAttributes_),
                    vao);
        }

        bool needsUpdate(BufferGeometry *geometry, BufferAttribute *index) {

            const auto &cachedAttributes = currentState_->attributes;
            const auto &geometryAttributes = geometry->getAttributes();

            int attributesNum = 0;

            for (const auto &[key, value] : geometryAttributes) {

                if (!cachedAttributes.count(key)) return true;


                const auto &geometryAttribute = geometryAttributes.at(key);

                if (!cachedAttributes.count(key)) return true;

                const auto &cachedAttribute = cachedAttributes.at(key);

                if (cachedAttribute != geometryAttribute.get()) return true;

                //  if (cachedAttribute.data != geometryAttribute.data) return true;

                attributesNum++;
            }

            if (currentState_->attributesNum != attributesNum) return true;

            if (currentState_->index != index) return true;

            return false;
        }

        void saveCache(BufferGeometry *geometry, BufferAttribute *index) {

            std::unordered_map<std::string, BufferAttribute *> cache;
            const auto &attributes = geometry->getAttributes();
            int attributesNum = 0;

            for (const auto &[key, attribute] : attributes) {

                cache[key] = attribute.get();

                attributesNum++;
            }

            currentState_->attributes = cache;
            currentState_->attributesNum = attributesNum;

            currentState_->index = index;
        }

        void initAttributes() {

            auto &newAttributes = currentState_->newAttributes;

            for (int &newAttribute : newAttributes) {

                newAttribute = 0;
            }
        }

        void enableAttribute(int attribute) {

            enableAttributeAndDivisor(attribute, 0);
        }

        void enableAttributeAndDivisor(int attribute, int meshPerAttribute) {

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

        void disableUnusedAttributes() {

            const auto &newAttributes = currentState_->newAttributes;
            auto &enabledAttributes = currentState_->enabledAttributes;

            for (int i = 0, il = (int) enabledAttributes.size(); i < il; i++) {

                if (enabledAttributes[i] != newAttributes[i]) {

                    glDisableVertexAttribArray(i);
                    enabledAttributes[i] = 0;
                }
            }
        }

        void vertexAttribPointer(GLuint index, GLint size, GLenum type, bool normalized, GLsizei stride, int offset) {

            if (type == GL_INT || type == GL_UNSIGNED_INT) {

                glVertexAttribIPointer(index, size, type, stride, (GLvoid*) offset);

            } else {

                glVertexAttribPointer(index, size, type, normalized, stride, (GLvoid*) offset);
            }
        }

        void setupVertexAttributes(Object3D *object, Material *material, std::shared_ptr<GLProgram> &program, BufferGeometry *geometry) {

            initAttributes();

            auto &geometryAttributes = geometry->getAttributes();

            const auto &programAttributes = program->getAttributes();

            //            const auto &materialDefaultAttributeValues = material.defaultAttributeValues;

            for (const auto &[name, programAttribute] : programAttributes) {

                if (programAttribute >= 0) {

                    if (geometryAttributes.count(name)) {

                        auto &geometryAttribute = geometryAttributes.at(name);

                        const auto normalized = geometryAttribute->normalized();
                        const auto size = geometryAttribute->itemSize();

                        auto attribute = attributes_.get(geometryAttribute.get());

                        // TODO Attribute may not be available on context restore

//                        if (!attribute) continue;

                        const auto buffer = attribute.buffer;
                        const auto type = attribute.type;
                        const auto bytesPerElement = attribute.bytesPerElement;

                        if (false /*geometryAttribute.isInterleavedBufferAttribute*/) {

                            // TODO

                        } else {

                            if (false /*geometryAttribute.isInstancedBufferAttribute*/) {

                                // TODO

                            } else {

                                enableAttribute( programAttribute );
                            }

                            glBindBuffer( GL_ARRAY_BUFFER, buffer );
                            vertexAttribPointer( programAttribute, size, type, normalized, 0, 0 );

                        }
                    }
                }
            }

            disableUnusedAttributes();
        }

        void dispose() {

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


        void releaseStatesOfGeometry(BufferGeometry *geometry) {

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

        void releaseStatesOfProgram(GLProgram &program) {

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

        void reset() {

            if (currentState_ == defaultState_) return;

            currentState_ = defaultState_;
            bindVertexArrayObject(*currentState_->object);
        }

    private:
        GLuint maxVertexAttributes_;

        const std::shared_ptr<GLBindingState> defaultState_;
        std::shared_ptr<GLBindingState> currentState_;

        GLAttributes &attributes_;

        std::unordered_map<unsigned int, ProgramMap> bindingStates;
    };

}// namespace threepp::gl

#endif//THREEPP_GLBINDINGSTATES_HPP
