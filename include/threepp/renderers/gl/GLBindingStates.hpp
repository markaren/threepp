// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBindingStates.js

#ifndef THREEPP_GLBINDINGSTATES_HPP
#define THREEPP_GLBINDINGSTATES_HPP

#include "GLAttributes.hpp"
#include "GLCapabilities.hpp"
#include "GLProgram.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/InstancedBufferAttribute.hpp"
#include "threepp/materials/materials.hpp"
#include "threepp/objects/InstancedMesh.hpp"

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace threepp::gl {

    struct GLBindingState {

        std::vector<int> newAttributes;
        std::vector<int> enabledAttributes;
        std::vector<int> attributeDivisors;
        std::optional<unsigned int> object;
        std::unordered_map<std::string, BufferAttribute *> attributes;
        BufferAttribute *index = nullptr;

        int attributesNum = 0;

        GLBindingState(
                std::vector<int> newAttributes,
                std::vector<int> enabledAttributes,
                std::vector<int> attributeDivisors,
                const std::optional<unsigned int> &object)
            : newAttributes(std::move(newAttributes)),
              enabledAttributes(std::move(enabledAttributes)),
              attributeDivisors(std::move(attributeDivisors)),
              object(object) {}
    };


    typedef std::unordered_map<bool, std::shared_ptr<GLBindingState>> StateMap;
    typedef std::unordered_map<int, StateMap> ProgramMap;

    struct GLBindingStates {

        explicit GLBindingStates(GLAttributes &attributes);

        void setup(Object3D *object, Material *material, std::shared_ptr<GLProgram> &program, BufferGeometry *geometry, BufferAttribute *index);

        [[nodiscard]] unsigned int createVertexArrayObject() const;

        void bindVertexArrayObject(unsigned int vao) const;

        void deleteVertexArrayObject(unsigned int vao);

        std::shared_ptr<GLBindingState> getBindingState(BufferGeometry *geometry, std::shared_ptr<GLProgram> &program, Material *material);

        [[nodiscard]] std::shared_ptr<GLBindingState> createBindingState(std::optional<unsigned int> vao) const;

        bool needsUpdate(BufferGeometry *geometry, BufferAttribute *index);

        void saveCache(BufferGeometry *geometry, BufferAttribute *index);

        void initAttributes();

        void enableAttribute(int attribute);

        void enableAttributeAndDivisor(int attribute, int meshPerAttribute);

        void disableUnusedAttributes();

        void vertexAttribPointer(unsigned int index, int size, unsigned int type, bool normalized, int stride, size_t offset);

        void setupVertexAttributes(Object3D *object, Material *material, std::shared_ptr<GLProgram> &program, BufferGeometry *geometry);

        void dispose();

        void releaseStatesOfGeometry(BufferGeometry *geometry);

        void releaseStatesOfProgram(GLProgram &program);

        void reset();

    private:
        GLAttributes &attributes_;
        unsigned int maxVertexAttributes_;

        const std::shared_ptr<GLBindingState> defaultState_;
        std::shared_ptr<GLBindingState> currentState_;

        std::unordered_map<unsigned int, ProgramMap> bindingStates;
    };

}// namespace threepp::gl

#endif//THREEPP_GLBINDINGSTATES_HPP
