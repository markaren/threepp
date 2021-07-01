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

        unsigned int id;
        Object3D *object;
        BufferGeometry *geometry;
        Material *material;
        std::shared_ptr<GLProgram> program;
        int groupOrder;
        unsigned int renderOrder;
        float z;
        std::optional<GeometryGroup> group;
    };


    struct GLRenderList {

        std::vector<RenderItem> opaque;
        std::vector<RenderItem> transmissive;
        std::vector<RenderItem> transparent;

        std::vector<RenderItem> renderItems;
        int renderItemsIndex = 0;

        explicit GLRenderList(GLProperties &properties);

        void init();

        RenderItem &getNextRenderItem(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, GeometryGroup &group);

        void push(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, GeometryGroup &group);

        void unshift(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, GeometryGroup &group);

        void sort();

        void finish();

    private:
        GLProperties &properties;
    };

    struct GLRenderLists {

        explicit GLRenderLists(GLProperties &properties) : properties(properties) {}

        std::shared_ptr<GLRenderList> get(Scene *scene, int renderCallDepth) {

            if (!lists.count(scene)) {

                auto &l = lists[scene] = std::vector<std::shared_ptr<GLRenderList>>{ std::make_shared<GLRenderList>(properties)};
                return l.back();

            } else {

                auto &l = lists.at(scene);
                if (renderCallDepth >= l.size()) {

                    l.emplace_back(std::make_shared<GLRenderList>(properties));
                    return l.back();

                } else {

                    return l[renderCallDepth];
                }
            }
        }


        void dispose() {

            lists.clear();
        }

    private:
        GLProperties &properties;

        std::unordered_map<Scene *, std::vector<std::shared_ptr<GLRenderList>>> lists;
    };

}// namespace threepp::gl

#endif//THREEPP_GLRENDERLIST_HPP
