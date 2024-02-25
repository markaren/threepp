// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLRenderLists.js

#ifndef THREEPP_GLRENDERLIST_HPP
#define THREEPP_GLRENDERLIST_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/core/misc.hpp"
#include "threepp/materials/Material.hpp"

#include "GLProgram.hpp"
#include "GLProperties.hpp"

namespace threepp::gl {

    struct RenderItem {

        std::optional<unsigned int> id;
        Object3D* object;
        BufferGeometry* geometry;
        Material* material;
        GLProgram* program;
        unsigned int groupOrder;
        unsigned int renderOrder;
        float z;
        std::optional<GeometryGroup> group;
    };

    struct GLRenderList {

        std::vector<RenderItem*> opaque;
        std::vector<RenderItem*> transparent;

        std::vector<std::unique_ptr<RenderItem>> renderItems;
        size_t renderItemsIndex = 0;

        explicit GLRenderList(GLProperties& properties);

        void init();

        RenderItem* getNextRenderItem(
                Object3D* object,
                BufferGeometry* geometry,
                Material* material,
                unsigned int groupOrder, float z, std::optional<GeometryGroup> group);

        void push(
                Object3D* object,
                BufferGeometry* geometry,
                Material* material,
                unsigned int groupOrder, float z, std::optional<GeometryGroup> group);

        void unshift(
                Object3D* object,
                BufferGeometry* geometry,
                Material* material,
                unsigned int groupOrder, float z, std::optional<GeometryGroup> group);

        void sort();

        void finish();

    private:
        GLProperties& properties;
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
