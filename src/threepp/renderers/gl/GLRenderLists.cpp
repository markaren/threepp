
#include "threepp/renderers/gl/GLRenderLists.hpp"

#include <memory>

using namespace threepp;
using namespace threepp::gl;

GLRenderLists::GLRenderLists(GLProperties& properties): properties(properties) {}

GLRenderList* GLRenderLists::get(Object3D* scene, size_t renderCallDepth) {

    if (!lists.contains(scene->uuid)) {

        auto& l = lists[scene->uuid].emplace_back(std::make_unique<GLRenderList>(properties));
        return l.get();

    } else {

        auto& l = lists.at(scene->uuid);
        if (renderCallDepth >= l.size()) {

            l.emplace_back(std::make_unique<GLRenderList>(properties));
            return l.back().get();

        } else {

            return l.at(renderCallDepth).get();
        }
    }
}

void GLRenderLists::dispose() {

    lists.clear();
}
