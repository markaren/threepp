
#include "threepp/renderers/gl/GLRenderLists.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    int painterSortStable(const RenderItem &a, const RenderItem &b) {

//        if (a.groupOrder != b.groupOrder) {
//
//            return a.groupOrder - b.groupOrder;
//
//        } else if (a.renderOrder != b.renderOrder) {
//
//            return a.renderOrder - b.renderOrder;
//
//        } else if (a.program != b.program) {
//
//            return a.program.id - b.program.id;
//
//        } else if (a.material->id != b.material->id) {
//
//            return a.material->id - b.material->id;
//
//        } else if (a.z != b.z) {
//
//            return a.z - b.z;
//
//        } else {
//
//            return a.id - b.id;
//        }
return 0;
    }

    int reversePainterSortStable(const RenderItem &a, const RenderItem &b) {

        if (a.groupOrder != b.groupOrder) {

            return a.groupOrder - b.groupOrder;

        } else if (a.renderOrder != b.renderOrder) {

            return a.renderOrder - b.renderOrder;

        } else if (a.z != b.z) {

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

        renderItems.emplace_back(RenderItem {object->id,
                                  object,
                                  geometry,
                                  material,
                                  materialProperties.program,
                                  groupOrder,
                                  object->renderOrder,
                                  z,
                                  group});

    } else {

        RenderItem &renderItem = renderItems.at(renderItemsIndex);

        renderItem.id = object->id;
        renderItem.object = object;
        renderItem.geometry = geometry;
        renderItem.material = material;
        if (materialProperties.program) {
//            renderItem.program = materialProperties.program.value();
        }
        renderItem.groupOrder = groupOrder;
        renderItem.renderOrder = object->renderOrder;
        renderItem.z = z;
        renderItem.group = group;
    }

    renderItemsIndex++;

    return renderItems.at(renderItemsIndex - 1);
}

void gl::GLRenderList::push(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, int group) {

    auto& renderItem = getNextRenderItem( object, geometry, material, groupOrder, z, group );

    if ( material->transparent ) {

        transparent.emplace_back( renderItem );

    } else {

        opaque.emplace_back( renderItem );

    }
}
