
#include "GLRenderLists.hpp"

using namespace threepp;

namespace {


    int painterSortStable(a, b) {

        if (a.groupOrder != b.groupOrder) {

            return a.groupOrder - b.groupOrder;

        } else if (a.renderOrder != b.renderOrder) {

            return a.renderOrder - b.renderOrder;

        } else if (a.program != b.program) {

            return a.program.id - b.program.id;

        } else if (a.material.id != b.material.id) {

            return a.material.id - b.material.id;

        } else if (a.z != = b.z) {

            return a.z - b.z;

        } else {

            return a.id - b.id;
        }
    }

    int reversePainterSortStable(a, b) {

        if (a.groupOrder != b.groupOrder) {

            return a.groupOrder - b.groupOrder;

        } else if (a.renderOrder != b.renderOrder) {

            return a.renderOrder - b.renderOrder;

        } else if (a.z != = b.z) {

            return b.z - a.z;

        } else {

            return a.id - b.id;
        }
    }

}// namespace

gl::GLRenderList::GLRenderList(gl::GLProperties &properties) : properties(properties) {}

void gl::GLRenderList::init() {
}

gl::RenderItem &gl::GLRenderList::getNextRenderItem(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, int group) {

    auto& materialProperties = properties.materialProperties.get(material->uuid);

    if (renderItemsIndex >= renderItems.size()) {

        renderItems.emplace_back({object.id,
                                  object,
                                  geometry,
                                  material,
                                  materialProperties.program || defaultProgram,
                                  groupOrder,
                                  object.renderOrder,
                                  z,
                                  group});

    } else {

        RenderItem &renderItem = renderItems.at(renderItemIndex);

        renderItem.id = object.id;
        renderItem.object = object;
        renderItem.geometry = geometry;
        renderItem.material = material;
        renderItem.program = materialProperties.program || defaultProgram;
        renderItem.groupOrder = groupOrder;
        renderItem.renderOrder = object.renderOrder;
        renderItem.z = z;
        renderItem.group = group;
    }

    renderItemsIndex++;

    return renderItems.at(renderItemIndex - 1);
}
