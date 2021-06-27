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
        std::optional<GLProgram> program;
        int groupOrder;
        unsigned int renderOrder;
        float z;
        int group;
    };


    struct GLRenderList {

        explicit GLRenderList(GLProperties &properties);

        void init();

        RenderItem &getNextRenderItem(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, int group);

        void push( Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, int group );

    private:
        GLProperties &properties;

        std::vector<RenderItem> renderItems;
        int renderItemsIndex = 0;

        std::vector<RenderItem> opaque;
        std::vector<RenderItem> transmissive;
        std::vector<RenderItem> transparent;
    };

    struct GLRenderLists {

        void dispose() {

        }

    };

}// namespace threepp::gl

#endif//THREEPP_GLRENDERLIST_HPP
