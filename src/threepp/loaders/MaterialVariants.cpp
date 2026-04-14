#include "threepp/loaders/MaterialVariants.hpp"

#include "threepp/objects/ObjectWithMaterials.hpp"

namespace threepp {

    void MaterialVariants::reset(Object3D& sceneRoot) const {
        sceneRoot.traverse([&](Object3D& obj) {
            auto* owm = dynamic_cast<ObjectWithMaterials*>(&obj);
            if (!owm) return;
            auto it = defaults.find(obj.uuid);
            if (it != defaults.end())
                owm->setMaterial(it->second);
        });
    }

    void MaterialVariants::apply(const std::string& variantName, Object3D& sceneRoot) const {
        reset(sceneRoot);

        auto it = table.find(variantName);
        if (it == table.end()) return;

        std::unordered_map<std::string, std::shared_ptr<Material>> lookup;
        lookup.reserve(it->second.size());
        for (const auto& entry : it->second)
            lookup[entry.meshUuid] = entry.material;

        sceneRoot.traverse([&](Object3D& obj) {
            auto* owm = dynamic_cast<ObjectWithMaterials*>(&obj);
            if (!owm) return;
            auto mit = lookup.find(obj.uuid);
            if (mit != lookup.end())
                owm->setMaterial(mit->second);
        });
    }

}// namespace threepp
