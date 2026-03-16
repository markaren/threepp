
#include "RenderLists.hpp"

#include <algorithm>
#include <memory>

using namespace threepp;

namespace {

    struct {

        bool operator()(const RenderItem* a, const RenderItem* b) {
            if (a->groupOrder != b->groupOrder) {
                return a->groupOrder < b->groupOrder;
            } else if (a->renderOrder != b->renderOrder) {
                return a->renderOrder < b->renderOrder;
            } else if (a->programId != b->programId) {
                return a->programId < b->programId;
            } else if (a->material->id != b->material->id) {
                return a->material->id < b->material->id;
            } else if (a->z != b->z) {
                return a->z < b->z;
            } else {
                return a->id < b->id;
            }
        }
    } painterSortStable;

    struct {
        bool operator()(const RenderItem* a, const RenderItem* b) {

            if (a->groupOrder != b->groupOrder) {
                return a->groupOrder < b->groupOrder;
            } else if (a->renderOrder != b->renderOrder) {
                return a->renderOrder < b->renderOrder;
            } else if (a->z != b->z) {
                return a->z > b->z;
            } else {
                return a->id < b->id;
            }
        }
    } reversePainterSortStable;

}// namespace

RenderList::RenderList(ProgramIdResolver resolver): resolver_(std::move(resolver)) {}

void RenderList::init() {

    renderItemsIndex = 0;

    opaque.clear();
    transparent.clear();
}

RenderItem* RenderList::getNextRenderItem(
        Object3D* object,
        BufferGeometry* geometry,
        Material* material,
        unsigned int groupOrder, float z, std::optional<GeometryGroup> group) {

    uint64_t progId = resolver_ ? resolver_(material) : 0;

    RenderItem* renderItem = nullptr;

    if (renderItemsIndex >= renderItems.size()) {
        auto r = std::make_unique<RenderItem>(RenderItem{object->id,
                                                         object,
                                                         geometry,
                                                         material,
                                                         progId,
                                                         groupOrder,
                                                         object->renderOrder,
                                                         z,
                                                         group});
        renderItems.emplace_back(std::move(r));
        renderItem = renderItems.back().get();

    } else {

        renderItem = renderItems.at(renderItemsIndex).get();

        renderItem->id = object->id;
        renderItem->object = object;
        renderItem->geometry = geometry;
        renderItem->material = material;
        renderItem->programId = progId;
        renderItem->groupOrder = groupOrder;
        renderItem->renderOrder = object->renderOrder;
        renderItem->z = z;
        renderItem->group = group;
    }

    ++renderItemsIndex;

    return renderItem;
}

void RenderList::push(
        Object3D* object,
        BufferGeometry* geometry,
        Material* material,
        unsigned int groupOrder, float z, std::optional<GeometryGroup> group) {

    auto renderItem = getNextRenderItem(object, geometry, material, groupOrder, z, group);

    if (material->transparent) {

        transparent.emplace_back(renderItem);

    } else {

        opaque.emplace_back(renderItem);
    }
}

void RenderList::unshift(
        Object3D* object,
        BufferGeometry* geometry,
        Material* material,
        unsigned int groupOrder, float z, std::optional<GeometryGroup> group) {

    auto renderItem = getNextRenderItem(object, geometry, material, groupOrder, z, group);

    if (material->transparent) {

        transparent.insert(transparent.begin(), renderItem);

    } else {

        opaque.insert(opaque.begin(), renderItem);
    }
}

void RenderList::sort() {

    if (opaque.size() > 1) std::stable_sort(opaque.begin(), opaque.end(), painterSortStable);
    if (transparent.size() > 1) std::stable_sort(transparent.begin(), transparent.end(), reversePainterSortStable);
}

void RenderList::finish() {

    // Clear references from inactive renderItems in the list

    for (auto i = renderItemsIndex, il = renderItems.size(); i < il; ++i) {

        auto& renderItem = renderItems.at(i);

        if (!renderItem->id) break;

        renderItem->id = std::nullopt;
        renderItem->object = nullptr;
        renderItem->geometry = nullptr;
        renderItem->material = nullptr;
        renderItem->programId = 0;
        renderItem->group = std::nullopt;
    }
}

RenderLists::RenderLists(ProgramIdResolver resolver): resolver_(std::move(resolver)) {}

RenderList* RenderLists::get(Object3D* scene, size_t renderCallDepth) {

    if (!lists.contains(scene->uuid)) {

        auto& l = lists[scene->uuid].emplace_back(std::make_unique<RenderList>(resolver_));
        return l.get();

    } else {

        auto& l = lists.at(scene->uuid);
        if (renderCallDepth >= l.size()) {

            l.emplace_back(std::make_unique<RenderList>(resolver_));
            return l.back().get();

        } else {

            return l.at(renderCallDepth).get();
        }
    }
}

void RenderLists::dispose() {

    lists.clear();
}
