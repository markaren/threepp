// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLRenderLists.js
//
// GL-specific thin wrapper around the backend-neutral RenderList.
// The sort comparator uses GLProgram::id via the ProgramIdResolver callback.

#ifndef THREEPP_GLRENDERLIST_HPP
#define THREEPP_GLRENDERLIST_HPP

#include "threepp/renderers/common/RenderLists.hpp"

#include "GLProgram.hpp"
#include "GLProperties.hpp"

namespace threepp::gl {

    // Backward-compatible aliases — existing code using gl::RenderItem, gl::GLRenderList, etc. continues to work.
    using RenderItem = threepp::RenderItem;

    struct GLRenderList : public threepp::RenderList {

        explicit GLRenderList(GLProperties& properties)
            : threepp::RenderList([&properties](Material* mat) -> uint64_t {
                  auto mp = properties.materialProperties.get(mat);
                  return mp && mp->program ? static_cast<uint64_t>(mp->program->id) : 0;
              }) {}
    };

    struct GLRenderLists {

        explicit GLRenderLists(GLProperties& properties);

        GLRenderList* get(Object3D* scene, size_t renderCallDepth);

        void dispose();

    private:
        GLProperties& properties;

        std::unordered_map<std::string, std::vector<std::unique_ptr<GLRenderList>>> lists;
    };

}// namespace threepp::gl

#endif//THREEPP_GLRENDERLIST_HPP
