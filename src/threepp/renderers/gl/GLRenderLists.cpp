
#include "threepp/renderers/gl/GLRenderLists.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    struct {

        bool operator()(const RenderItem &a, const RenderItem &b) {
            if (a.groupOrder != b.groupOrder) {

                return a.groupOrder < b.groupOrder;

            } else if (a.renderOrder != b.renderOrder) {

                return a.renderOrder < b.renderOrder;

            } else if (*a.program != *b.program) {

                return a.program->id < b.program->id;

            } else if (a.material->id != b.material->id) {

                return a.material->id < b.material->id;

            } else if (a.z != b.z) {

                return a.z < b.z;

            } else {

                return a.id < b.id;
            }
        }
    } painterSortStable;

    struct {
        bool operator()(const RenderItem &a, const RenderItem &b) {

            if (a.groupOrder != b.groupOrder) {

                return a.groupOrder < b.groupOrder;

            } else if (a.renderOrder != b.renderOrder) {

                return a.renderOrder < b.renderOrder;

            } else if (a.z != b.z) {

                return b.z < a.z;

            } else {

                return a.id < b.id;
            }
        }
    } reversePainterSortStable;

}// namespace

gl::GLRenderList::GLRenderList(gl::GLProperties &properties) : properties(properties) {}

void gl::GLRenderList::init() {
}

gl::RenderItem &gl::GLRenderList::getNextRenderItem(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, GeometryGroup &group) {

    auto &materialProperties = properties.materialProperties.get(material->uuid);

    if (renderItemsIndex >= renderItems.size()) {

        renderItems.emplace_back(RenderItem{object->id,
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
            renderItem.program = materialProperties.program;
        }
        renderItem.groupOrder = groupOrder;
        renderItem.renderOrder = object->renderOrder;
        renderItem.z = z;
        renderItem.group = group;
    }

    renderItemsIndex++;

    return renderItems.at(renderItemsIndex - 1);
}

void gl::GLRenderList::push(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, GeometryGroup &group) {

    auto &renderItem = getNextRenderItem(object, geometry, material, groupOrder, z, group);

    if (material->transparent) {

        transparent.emplace_back(renderItem);

    } else {

        opaque.emplace_back(renderItem);
    }
}

void GLRenderList::unshift(Object3D *object, BufferGeometry *geometry, Material *material, int groupOrder, float z, GeometryGroup &group) {

    auto &renderItem = getNextRenderItem(object, geometry, material, groupOrder, z, group);

    if (false /*material->transmission > 0.0*/) {

        transmissive.insert(transmissive.begin(), renderItem);

    } else if (material->transparent == true) {

        transparent.insert(transparent.begin(), renderItem);

    } else {

        opaque.insert(opaque.begin(), renderItem);
    }
}


void GLRenderList::sort() {

    if (opaque.size() > 1) std::sort(opaque.begin(), opaque.end(), painterSortStable);
    if (transmissive.size() > 1) std::sort(transmissive.begin(), transmissive.end(), reversePainterSortStable);
    if (transparent.size() > 1) std::sort(transparent.begin(), transparent.end(), reversePainterSortStable);
}

void GLRenderList::finish() {

    // Clear references from inactive renderItems in the list

    for (auto &renderItem : renderItems) {

        if (renderItem.id == -1) break;

        renderItem.id = -1;
        renderItem.object = nullptr;
        renderItem.geometry = nullptr;
        renderItem.material = nullptr;
        renderItem.program = nullptr;
        renderItem.group = std::nullopt;
    }
}
