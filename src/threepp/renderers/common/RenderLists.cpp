
#include "RenderLists.hpp"

#include "threepp/materials/interfaces.hpp"

#include <algorithm>
#include <memory>

using namespace threepp;

namespace {

    // three.js r186's painterSortStable minus its materialVariant tie-break.
    // r186 has no program term here, so neither does this.
    struct {

        bool operator()(const RenderItem* a, const RenderItem* b) const {
            if (a->groupOrder != b->groupOrder) {
                return a->groupOrder < b->groupOrder;
            } else if (a->renderOrder != b->renderOrder) {
                return a->renderOrder < b->renderOrder;
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
        bool operator()(const RenderItem* a, const RenderItem* b) const {

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

void RenderList::init() {

    renderItemsIndex = 0;

    opaque.clear();
    transmissive.clear();
    transparent.clear();
}

RenderItem* RenderList::getNextRenderItem(
        Object3D* object,
        BufferGeometry* geometry,
        Material* material,
        int groupOrder, float z, std::optional<GeometryGroup> group) {

    RenderItem* renderItem = nullptr;

    if (renderItemsIndex >= renderItems.size()) {
        auto r = std::make_unique<RenderItem>(RenderItem{object->id,
                                                         object,
                                                         geometry,
                                                         material,
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
        int groupOrder, float z, std::optional<GeometryGroup> group) {

    auto renderItem = getNextRenderItem(object, geometry, material, groupOrder, z, group);

    auto transmissionMaterial = dynamic_cast<MaterialWithTransmission*>(material);
    if (transmissionMaterial && transmissionMaterial->transmission > 0.f) {

        transmissive.insert(transmissive.begin(), renderItem);

    } else if (material->transparent) {

        transparent.emplace_back(renderItem);

    } else {

        opaque.emplace_back(renderItem);
    }
}

void RenderList::unshift(
        Object3D* object,
        BufferGeometry* geometry,
        Material* material,
        int groupOrder, float z, std::optional<GeometryGroup> group) {

    auto renderItem = getNextRenderItem(object, geometry, material, groupOrder, z, group);

    if (material->transparent) {

        transparent.insert(transparent.begin(), renderItem);

    } else {

        opaque.insert(opaque.begin(), renderItem);
    }
}

void RenderList::sort() {

    if (opaque.size() > 1) std::stable_sort(opaque.begin(), opaque.end(), painterSortStable);
    if (transmissive.size() > 1) std::stable_sort(transmissive.begin(), transmissive.end(), reversePainterSortStable);
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
        renderItem->group = std::nullopt;
    }
}

RenderList* RenderLists::get(Object3D* scene, size_t renderCallDepth) {

    if (!lists.contains(scene->uuid)) {

        auto& l = lists[scene->uuid].emplace_back(std::make_unique<RenderList>());
        return l.get();

    } else {

        auto& l = lists.at(scene->uuid);
        if (renderCallDepth >= l.size()) {

            l.emplace_back(std::make_unique<RenderList>());
            return l.back().get();

        } else {

            return l.at(renderCallDepth).get();
        }
    }
}

void RenderLists::dispose() {

    lists.clear();
}
