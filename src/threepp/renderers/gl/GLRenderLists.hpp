// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLRenderLists.js
//
// GL-specific thin wrapper around the backend-neutral RenderList.

#ifndef THREEPP_GLRENDERLIST_HPP
#define THREEPP_GLRENDERLIST_HPP

#include "threepp/renderers/common/RenderLists.hpp"

namespace threepp::gl {

    struct GLRenderList : public threepp::RenderList {};

    struct GLRenderLists {

        GLRenderList* get(Object3D* scene, size_t renderCallDepth);

        void dispose();

    private:
        std::unordered_map<std::string, std::vector<std::unique_ptr<GLRenderList>>> lists;
    };

}// namespace threepp::gl

#endif//THREEPP_GLRENDERLIST_HPP
