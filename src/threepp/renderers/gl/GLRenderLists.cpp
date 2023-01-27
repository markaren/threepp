
#include "threepp/renderers/gl/GLRenderLists.hpp"

#include <algorithm>
#include <memory>

using namespace threepp;
using namespace threepp::gl;

namespace {

    struct {

        bool operator()(const RenderItem* a, const RenderItem* b) {
            if (a->groupOrder != b->groupOrder) {
                return b->groupOrder > a->groupOrder;
            } else if (a->renderOrder != b->renderOrder) {
                return b->renderOrder > a->renderOrder;
            } else if (a->program != nullptr && b->program != nullptr && (a->program->id != b->program->id)) {
                return b->program->id > a->program->id;
            } else if (a->material->id != b->material->id) {
                return b->material->id > a->material->id;
            } else if (a->z != b->z) {
                return b->z > a->z;
            } else {
                return b->id > a->id;
            }
        }
    } painterSortStable;

    struct {
        bool operator()(const RenderItem *a, const RenderItem *b) {

            if (a->groupOrder != b->groupOrder) {
                return b->groupOrder > a->groupOrder;
            } else if (a->renderOrder != b->renderOrder) {
                return b->renderOrder > a->renderOrder;
            } else if (a->program != nullptr && b->program != nullptr && (a->program->id != b->program->id)) {
                return b->program->id > a->program->id;
            } else if (a->material->id != b->material->id) {
                return b->material->id > a->material->id;
            } else if (a->z != b->z) {
                return b->z > a->z;
            } else {
                return b->id > a->id;
            }
        }
    } reversePainterSortStable;

}// namespace

gl::GLRenderList::GLRenderList(gl::GLProperties &properties) : properties(properties) {}

void gl::GLRenderList::init() {

    renderItemsIndex = 0;

    renderItems.clear();
    opaque.clear();
    transparent.clear();
}

gl::RenderItem* gl::GLRenderList::getNextRenderItem(
        Object3D *object,
        BufferGeometry *geometry,
        Material *material,
        int groupOrder, float z, std::optional<GeometryGroup> group) {

    gl::RenderItem* renderItem = nullptr;
    auto &materialProperties = properties.materialProperties.get(material->uuid);

    if (renderItemsIndex >= renderItems.size()) {
        auto r = std::make_unique<RenderItem>(RenderItem{(int) object->id,
                                                object,
                                                geometry,
                                                material,
                                                materialProperties.program.get(),
                                                groupOrder,
                                                object->renderOrder,
                                                z,
                                                group});
        renderItems.emplace_back(std::move(r));
        renderItem = renderItems.back().get();

    } else {

        renderItem = renderItems.at(renderItemsIndex).get();

        renderItem->id = (int) object->id;
        renderItem->object = object;
        renderItem->geometry = geometry;
        renderItem->material = material;
        if (materialProperties.program) {
            renderItem->program = materialProperties.program.get();
        }
        renderItem->groupOrder = groupOrder;
        renderItem->renderOrder = object->renderOrder;
        renderItem->z = z;
        renderItem->group = group;
    }

    renderItemsIndex++;

    return renderItem;
}

void gl::GLRenderList::push(
        Object3D *object,
        BufferGeometry *geometry,
        Material *material,
        int groupOrder, float z, std::optional<GeometryGroup> group) {

    auto renderItem = getNextRenderItem(object, geometry, material, groupOrder, z, group);

    if (material->transparent) {

        transparent.emplace_back(renderItem);

    } else {

        opaque.emplace_back(renderItem);
    }
}

void GLRenderList::unshift(
        Object3D *object,
        BufferGeometry *geometry,
        Material *material,
        int groupOrder, float z, std::optional<GeometryGroup> group) {

    auto renderItem = getNextRenderItem(object, geometry, material, groupOrder, z, group);

    if (material->transparent) {

        transparent.insert(transparent.begin(), renderItem);

    } else {

        opaque.insert(opaque.begin(), renderItem);
    }
}

void GLRenderList::sort() {

    if (opaque.size() > 1) std::stable_sort(opaque.begin(), opaque.end(), painterSortStable);
    if (transparent.size() > 1) std::stable_sort(transparent.begin(), transparent.end(), reversePainterSortStable);
}

void GLRenderList::finish() {

    // Clear references from inactive renderItems in the list

    for (auto i = renderItemsIndex, il = renderItems.size(); i < il; i++) {

        auto& renderItem = renderItems.at(i);

        if (renderItem->id == -1) break;

        renderItem->id = -1;
        renderItem->object = nullptr;
        renderItem->geometry = nullptr;
        renderItem->material = nullptr;
        renderItem->program = nullptr;
        renderItem->group = std::nullopt;
    }
}
