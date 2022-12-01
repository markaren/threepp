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

        int id;
        std::shared_ptr<Object3D> object;
        std::shared_ptr<BufferGeometry> geometry;
        std::shared_ptr<Material> material;
        std::shared_ptr<GLProgram> program;
        int groupOrder;
        int renderOrder;
        float z;
        std::optional<GeometryGroup> group;
    };

    struct GLRenderList {

        std::vector<std::shared_ptr<RenderItem>> opaque;
        std::vector<std::shared_ptr<RenderItem>> transmissive;
        std::vector<std::shared_ptr<RenderItem>> transparent;

        std::vector<std::shared_ptr<RenderItem>> renderItems;
        int renderItemsIndex = 0;

        explicit GLRenderList(GLProperties &properties);

        void init();

        std::shared_ptr<RenderItem> getNextRenderItem(
                const std::shared_ptr<Object3D> &object,
                const std::shared_ptr<BufferGeometry> &geometry,
                const std::shared_ptr<Material> &material,
                int groupOrder, float z, std::optional<GeometryGroup> group);

        void push(
                const std::shared_ptr<Object3D> &object,
                const std::shared_ptr<BufferGeometry> &geometry,
                const std::shared_ptr<Material> &material,
                int groupOrder, float z, std::optional<GeometryGroup> group);

        void unshift(
                const std::shared_ptr<Object3D> &object,
                const std::shared_ptr<BufferGeometry> &geometry,
                const std::shared_ptr<Material> &material,
                int groupOrder, float z, std::optional<GeometryGroup> group);

        void sort();

        void finish();

    private:
        GLProperties &properties;
    };

    struct GLRenderLists {

        explicit GLRenderLists(GLProperties &properties) : properties(properties) {}

        std::shared_ptr<GLRenderList> get(Scene *scene, int renderCallDepth) {

            if (!lists.count(scene->uuid)) {

                auto &l = lists[scene->uuid] = std::vector<std::shared_ptr<GLRenderList>>{std::make_shared<GLRenderList>(properties)};
                return l.back();

            } else {

                auto &l = lists.at(scene->uuid);
                if (renderCallDepth >= l.size()) {

                    l.emplace_back(std::make_shared<GLRenderList>(properties));
                    return l.back();

                } else {

                    return l.at(renderCallDepth);
                }
            }
        }

        void dispose() {

            lists.clear();
        }

    private:
        GLProperties &properties;

        std::unordered_map<std::string, std::vector<std::shared_ptr<GLRenderList>>> lists;
    };

}// namespace threepp::gl

#endif//THREEPP_GLRENDERLIST_HPP
