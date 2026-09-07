#ifndef THREEPP_MATERIALVARIANTS_HPP
#define THREEPP_MATERIALVARIANTS_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    struct MeshVariantEntry {
        std::string meshUuid;
        std::shared_ptr<Material> material;
    };

    struct MaterialVariants {
        /// Ordered list of variant names as declared in the file.
        std::vector<std::string> names;

        /// variantName -> list of (meshUuid, material) assignments.
        /// Meshes absent from a variant's list revert to their default material.
        std::unordered_map<std::string, std::vector<MeshVariantEntry>> table;

        /// meshUuid -> material assigned at load time.
        std::unordered_map<std::string, std::shared_ptr<Material>> defaults;

        [[nodiscard]] bool empty() const { return names.empty(); }

        /// Apply a named variant to the scene. Meshes not listed revert to defaults.
        void apply(const std::string& variantName, Object3D& sceneRoot) const;

        /// Restore all tracked meshes to their load-time materials.
        void reset(Object3D& sceneRoot) const;
    };

}// namespace threepp

#endif// THREEPP_MATERIALVARIANTS_HPP
