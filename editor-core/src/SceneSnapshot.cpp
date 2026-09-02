
#include "threepp/extras/editor/SceneSnapshot.hpp"

#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <functional>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    using TextureSlot = std::shared_ptr<Texture>;
    using SlotVisitor = std::function<void(TextureSlot&)>;

    // Every texture-carrying member reachable through the material mixins.
    // Kept in one place so collect and rebind can never drift apart — a slot
    // added here is handled by both directions at once.
    void forEachTextureSlot(Material& material, const SlotVisitor& visit) {

        if (auto* m = dynamic_cast<MaterialWithMap*>(&material)) visit(m->map);
        if (auto* m = dynamic_cast<MaterialWithAlphaMap*>(&material)) visit(m->alphaMap);
        if (auto* m = dynamic_cast<MaterialWithSpecularMap*>(&material)) visit(m->specularMap);
        if (auto* m = dynamic_cast<MaterialWithEnvMap*>(&material)) visit(m->envMap);
        if (auto* m = dynamic_cast<MaterialWithGradientMap*>(&material)) visit(m->gradientMap);
        if (auto* m = dynamic_cast<MaterialWithAoMap*>(&material)) visit(m->aoMap);
        if (auto* m = dynamic_cast<MaterialWithBumpMap*>(&material)) visit(m->bumpMap);
        if (auto* m = dynamic_cast<MaterialWithLightMap*>(&material)) visit(m->lightMap);
        if (auto* m = dynamic_cast<MaterialWithDisplacementMap*>(&material)) visit(m->displacementMap);
        if (auto* m = dynamic_cast<MaterialWithNormalMap*>(&material)) visit(m->normalMap);
        if (auto* m = dynamic_cast<MaterialWithMatCap*>(&material)) visit(m->matcap);
        if (auto* m = dynamic_cast<MaterialWithRoughness*>(&material)) visit(m->roughnessMap);
        if (auto* m = dynamic_cast<MaterialWithMetalness*>(&material)) visit(m->metalnessMap);
        if (auto* m = dynamic_cast<MaterialWithEmissive*>(&material)) visit(m->emissiveMap);
        if (auto* m = dynamic_cast<MaterialWithThickness*>(&material)) visit(m->thicknessMap);
        if (auto* m = dynamic_cast<MaterialWithTransmission*>(&material)) visit(m->transmissionMap);
        if (auto* m = dynamic_cast<MaterialWithClearcoat*>(&material)) {
            visit(m->clearcoatMap);
            visit(m->clearcoatRoughnessMap);
            visit(m->clearcoatNormalMap);
        }
        if (auto* m = dynamic_cast<MaterialWithDetailMap*>(&material)) {
            visit(m->detailMap);
            visit(m->detailNormalMap);
        }
    }

    void forEachMaterial(Object3D& root, const std::function<void(Material&)>& visit) {

        root.traverse([&](Object3D& o) {
            if (auto* withMaterials = dynamic_cast<ObjectWithMaterials*>(&o)) {
                for (const auto& material : withMaterials->materials()) {
                    if (material) visit(*material);
                }
                return;
            }
            if (auto material = o.material()) visit(*material);
        });
    }

}// namespace


void SceneSnapshot::collectTextures(Object3D& root,
                                    std::unordered_map<std::string, std::shared_ptr<Texture>>& out) {

    forEachMaterial(root, [&](Material& material) {
        forEachTextureSlot(material, [&](TextureSlot& slot) {
            if (slot) out.emplace(slot->uuid(), slot);
        });
    });

    if (auto* scene = dynamic_cast<Scene*>(&root)) {
        if (scene->environment) out.emplace(scene->environment->uuid(), scene->environment);
        if (auto background = scene->background.texture()) out.emplace(background->uuid(), background);
    }
}

void SceneSnapshot::rebindTextures(Object3D& root,
                                   const std::unordered_map<std::string, std::shared_ptr<Texture>>& textures) {

    if (textures.empty()) return;

    forEachMaterial(root, [&](Material& material) {
        bool changed = false;
        forEachTextureSlot(material, [&](TextureSlot& slot) {
            if (!slot) return;
            const auto it = textures.find(slot->uuid());
            if (it == textures.end() || it->second == slot) return;
            slot = it->second;
            changed = true;
        });
        // The reloaded material references a different Texture instance than the
        // one the renderer already has state for; force a re-bind.
        if (changed) material.needsUpdate();
    });

    if (auto* scene = dynamic_cast<Scene*>(&root)) {
        if (scene->environment) {
            const auto it = textures.find(scene->environment->uuid());
            if (it != textures.end()) scene->environment = it->second;
        }
        if (auto background = scene->background.texture()) {
            const auto it = textures.find(background->uuid());
            if (it != textures.end() && it->second != background) {
                scene->background = Background(it->second);
            }
        }
    }
}

bool SceneSnapshot::capture(Scene& scene, std::string* error) {

    clear();

    ObjectExporter exporter;
    ObjectExporterOptions options;
    // See the header: the live textures are kept by pointer instead.
    options.images = ImageStorage::Omit;
    // Never reference: a snapshot has to restore the scene exactly as it is
    // right now, including whatever the running session did to an imported
    // subtree. Re-importing from disk would hand back the file's version.
    options.models = ModelStorage::Embed;
    options.prettyPrint = false;

    try {
        json_ = exporter.toJson(scene, options);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        json_.clear();
        return false;
    }

    if (json_.empty()) {
        if (error) *error = "scene export produced no document";
        return false;
    }

    collectTextures(scene, textures_);

    // The clouds, by uuid. Scene children are owned by shared_ptr, so
    // shared_from_this is valid here (Object3D says so for exactly this case).
    scene.traverse([&](Object3D& object) {
        if (!object.as<SplatCloud>()) return;
        if (auto cloud = std::dynamic_pointer_cast<SplatCloud>(object.shared_from_this())) {
            splats_.emplace(cloud->uuid, cloud);
        }
    });
    return true;
}

std::shared_ptr<Scene> SceneSnapshot::restore(std::string* error) const {

    if (json_.empty()) {
        if (error) *error = "no snapshot captured";
        return nullptr;
    }

    ObjectLoader loader;
    if (!splats_.empty()) {
        loader.setSplatCloudResolver([this](const std::string& uuid) -> std::shared_ptr<SplatCloud> {
            const auto it = splats_.find(uuid);
            return it == splats_.end() ? nullptr : it->second;
        });
    }
    std::shared_ptr<Object3D> parsed;
    try {
        parsed = loader.parse(json_);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return nullptr;
    }

    auto scene = std::dynamic_pointer_cast<Scene>(parsed);
    if (!scene) {
        if (error) *error = "snapshot did not parse back into a Scene";
        return nullptr;
    }

    rebindTextures(*scene, textures_);
    return scene;
}

void SceneSnapshot::clear() {

    json_.clear();
    textures_.clear();
    splats_.clear();
}
