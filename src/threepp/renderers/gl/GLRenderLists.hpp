// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLRenderLists.js

#ifndef THREEPP_GLRENDERLIST_HPP
#define THREEPP_GLRENDERLIST_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"

#include "GLProgram.hpp"
#include "GLProperties.hpp"

namespace threepp::gl {

    struct RenderItem {

        unsigned int id;
        Object3D *object;
        BufferGeometry *geometry;
        Material *material;
        GLProgram program;
        int groupOrder;
        int renderOrder;
        float z;
        int group;
    };


    struct GLRenderList {

        explicit GLRenderList(GLProperties &properties);

        void init();

        RenderItem &getNextRenderItem(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, int group);

    private:
        GLProperties &properties;

        std::vector<RenderItem> renderItems;
        int renderItemIndex = 0;

        std::vector<Object3D *> opaque;
        std::vector<Object3D *> transmissive;
        std::vector<Object3D *> transparent;
    };

}// namespace threepp::gl

#endif//THREEPP_GLRENDERLIST_HPP
