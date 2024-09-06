// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBindingStates.js

#ifndef THREEPP_GLBINDINGSTATES_HPP
#define THREEPP_GLBINDINGSTATES_HPP

#include "GLAttributes.hpp"
#include "GLCapabilities.hpp"
#include "GLProgram.hpp"

#include "threepp/core/BufferGeometry.hpp"

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
        std::unordered_map<std::string, BufferAttribute*> attributes;
        BufferAttribute* index = nullptr;

        int attributesNum = 0;

        GLBindingState(
                std::vector<int> newAttributes,
                std::vector<int> enabledAttributes,
                std::vector<int> attributeDivisors,
                const std::optional<unsigned int>& object)
            : newAttributes(std::move(newAttributes)),
              enabledAttributes(std::move(enabledAttributes)),
              attributeDivisors(std::move(attributeDivisors)),
              object(object) {}
    };

    class GLBindingStates {

    public:
        explicit GLBindingStates(GLAttributes& attributes);

        void setup(Object3D* object, Material* material, GLProgram* program, BufferGeometry* geometry, BufferAttribute* index);

        void initAttributes();

        void enableAttribute(int attribute);

        void disableUnusedAttributes();

        void dispose();

        void releaseStatesOfGeometry(BufferGeometry* geometry);

        void releaseStatesOfProgram(GLProgram& program);

        void reset();

        ~GLBindingStates();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLBINDINGSTATES_HPP
