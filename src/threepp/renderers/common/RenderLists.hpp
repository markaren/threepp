// Backend-neutral render list for sorting opaque/transparent objects.
// Extracted from gl/GLRenderLists with GL dependencies removed.

#ifndef THREEPP_RENDERLISTS_HPP
#define THREEPP_RENDERLISTS_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/core/misc.hpp"
#include "threepp/materials/Material.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    struct RenderItem {

        std::optional<unsigned int> id;
        Object3D* object = nullptr;
        BufferGeometry* geometry = nullptr;
        Material* material = nullptr;
        uint64_t programId = 0;  // Opaque program identifier for sort ordering
        unsigned int groupOrder = 0;
        unsigned int renderOrder = 0;
        float z = 0;
        std::optional<GeometryGroup> group;
    };

    // Callback to resolve a material's current program ID.
    // GL backend returns GLProgram::id; Wgpu backend can return pipeline hash.
    using ProgramIdResolver = std::function<uint64_t(Material*)>;

    struct RenderList {

        std::vector<RenderItem*> opaque;
        std::vector<RenderItem*> transparent;

        std::vector<std::unique_ptr<RenderItem>> renderItems;
        size_t renderItemsIndex = 0;

        explicit RenderList(ProgramIdResolver resolver = nullptr);

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
        ProgramIdResolver resolver_;
    };

    struct RenderLists {

        explicit RenderLists(ProgramIdResolver resolver = nullptr);

        RenderList* get(Object3D* scene, size_t renderCallDepth);

        void dispose();

    private:
        ProgramIdResolver resolver_;

        std::unordered_map<std::string, std::vector<std::unique_ptr<RenderList>>> lists;
    };

}// namespace threepp

#endif//THREEPP_RENDERLISTS_HPP
