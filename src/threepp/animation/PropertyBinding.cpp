
#include "threepp/animation/PropertyBinding.hpp"
#include "threepp/animation/MaterialAnimationProxy.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/objects/SkinnedMesh.hpp"

#include <functional>
#include <iostream>
#include <regex>

using namespace threepp;

namespace {

    // Route a float-buffer write to a named field on a Material. Returns true
    // on hit. Covers the glTF KHR_animation_pointer subset we emit from
    // GLTFLoader: baseColorFactor, metalness, roughness, emissive,
    // emissiveIntensity, opacity, alphaTest. Unknown names are a no-op — the
    // loader won't emit them, so hitting the fallback means a typo somewhere.
    bool setMaterialProperty(Material* mat, const std::string& prop,
                             const std::vector<float>& buffer, size_t offset) {
        if (!mat) return false;

        if (prop == "baseColorFactor") {
            // glTF packs RGBA; route RGB → color, A → opacity. Opacity drives
            // the transparent flag only when < 1 so fully-opaque animations
            // don't stomp the material's transparent=false default.
            if (auto* c = dynamic_cast<MaterialWithColor*>(mat)) {
                c->color.setRGB(buffer[offset + 0], buffer[offset + 1], buffer[offset + 2]);
            }
            mat->opacity = buffer[offset + 3];
            if (mat->opacity < 1.0f) mat->transparent = true;
            mat->needsUpdate();
            return true;
        }
        if (prop == "color") {
            if (auto* c = dynamic_cast<MaterialWithColor*>(mat)) {
                c->color.setRGB(buffer[offset + 0], buffer[offset + 1], buffer[offset + 2]);
                mat->needsUpdate();
            }
            return true;
        }
        if (prop == "opacity") {
            mat->opacity = buffer[offset];
            if (mat->opacity < 1.0f) mat->transparent = true;
            mat->needsUpdate();
            return true;
        }
        if (prop == "emissive") {
            if (auto* e = dynamic_cast<MaterialWithEmissive*>(mat)) {
                e->emissive.setRGB(buffer[offset + 0], buffer[offset + 1], buffer[offset + 2]);
                mat->needsUpdate();
            }
            return true;
        }
        if (prop == "emissiveIntensity") {
            if (auto* e = dynamic_cast<MaterialWithEmissive*>(mat)) {
                e->emissiveIntensity = buffer[offset];
                mat->needsUpdate();
            }
            return true;
        }
        if (prop == "metalness") {
            if (auto* m = dynamic_cast<MaterialWithMetalness*>(mat)) {
                m->metalness = buffer[offset];
                mat->needsUpdate();
            }
            return true;
        }
        if (prop == "roughness") {
            if (auto* r = dynamic_cast<MaterialWithRoughness*>(mat)) {
                r->roughness = buffer[offset];
                mat->needsUpdate();
            }
            return true;
        }
        if (prop == "alphaTest") {
            mat->alphaTest = buffer[offset];
            mat->needsUpdate();
            return true;
        }
        return false;
    }

    bool getMaterialProperty(Material* mat, const std::string& prop,
                             std::vector<float>& buffer, size_t offset) {
        if (!mat) return false;

        if (prop == "baseColorFactor") {
            if (auto* c = dynamic_cast<MaterialWithColor*>(mat)) {
                buffer[offset + 0] = c->color.r;
                buffer[offset + 1] = c->color.g;
                buffer[offset + 2] = c->color.b;
            }
            buffer[offset + 3] = mat->opacity;
            return true;
        }
        if (prop == "color") {
            if (auto* c = dynamic_cast<MaterialWithColor*>(mat)) {
                buffer[offset + 0] = c->color.r;
                buffer[offset + 1] = c->color.g;
                buffer[offset + 2] = c->color.b;
            }
            return true;
        }
        if (prop == "opacity")            { buffer[offset] = mat->opacity; return true; }
        if (prop == "emissive") {
            if (auto* e = dynamic_cast<MaterialWithEmissive*>(mat)) {
                buffer[offset + 0] = e->emissive.r;
                buffer[offset + 1] = e->emissive.g;
                buffer[offset + 2] = e->emissive.b;
            }
            return true;
        }
        if (prop == "emissiveIntensity") {
            if (auto* e = dynamic_cast<MaterialWithEmissive*>(mat)) buffer[offset] = e->emissiveIntensity;
            return true;
        }
        if (prop == "metalness") {
            if (auto* m = dynamic_cast<MaterialWithMetalness*>(mat)) buffer[offset] = m->metalness;
            return true;
        }
        if (prop == "roughness") {
            if (auto* r = dynamic_cast<MaterialWithRoughness*>(mat)) buffer[offset] = r->roughness;
            return true;
        }
        if (prop == "alphaTest")          { buffer[offset] = mat->alphaTest; return true; }
        return false;
    }

}// namespace

namespace {

    Object3D* searchNodeSubtree(const std::string& nodeName, const std::vector<Object3D*>& children) {

        for (auto childNode : children) {

            if (childNode->name == nodeName || childNode->uuid == nodeName) {

                return childNode;
            }

            const auto result = searchNodeSubtree(nodeName, childNode->children);

            if (result) return result;
        }

        return nullptr;
    }


}// namespace

PropertyBinding::TrackResults PropertyBinding::parseTrackName(const std::string& trackName) {

    const auto lastDot = trackName.find_last_of('.');
    return {trackName.substr(0, lastDot), trackName.substr(lastDot+1)};
}

Object3D* PropertyBinding::findNode(Object3D* root, const std::string& nodeName) {

    if (nodeName.empty() || nodeName == "." || nodeName == root->name || nodeName == root->uuid) {

        return root;
    }

    // search into skeleton bones.
    if (root->is<SkinnedMesh>() && root->as<SkinnedMesh>()->skeleton) {

        const auto& bone = root->as<SkinnedMesh>()->skeleton->getBoneByName(nodeName);

        if (bone != nullptr) {

            return bone;
        }
    }

    // search into node subtree.
    if (!root->children.empty()) {

        const auto subTreeNode = searchNodeSubtree(nodeName, root->children);

        if (subTreeNode) {

            return subTreeNode;
        }
    }

    return nullptr;
}

void PropertyBinding::bind() {

    Object3D* targetObject = this->node;
    const auto& parsedPath = this->parsedPath;

    const auto& propertyName = parsedPath.propertyName;

    if (!this->node) {

        auto find = findNode(this->rootNode, parsedPath.nodeName);

        if (!find) {
            targetObject = rootNode;
        } else {
            targetObject = find;
        }

        this->node = targetObject;
    }

    // set fail state so we can just 'return' on error
    this->_getValue = _getValue_unavailable;
    this->_setValue = _setValue_unavailable;

    // ensure there is a value node
    if (!targetObject) {

        std::cerr << "THREE.PropertyBinding: Trying to update node for track: " << this->path << " but it wasn't found." << std::endl;
        return;
    }

    // glTF nodes are pre-created as Group; the actual Mesh with morph
    // influences is a child. When the property is a morph index and the
    // found node doesn't implement the interface, search its subtree.
    if (parseMorphIndex(propertyName) >= 0 &&
        !dynamic_cast<ObjectWithMorphTargetInfluences*>(targetObject)) {
        std::function<Object3D*(Object3D*)> findMorphChild = [&](Object3D* n) -> Object3D* {
            for (auto* c : n->children) {
                if (dynamic_cast<ObjectWithMorphTargetInfluences*>(c)) return c;
                if (auto* found = findMorphChild(c)) return found;
            }
            return nullptr;
        };
        if (auto* m = findMorphChild(targetObject))
            targetObject = m;
    }

    this->targetObject = targetObject;
    this->propertyName = propertyName;

    // select getter / setter
    if (dynamic_cast<MaterialAnimationProxy*>(targetObject)) {
        this->_getValue = _getValue_material;
        this->_setValue = _setValue_material;
    } else {
        this->_getValue = _getValue_direct;
        this->_setValue = _setValue_direct_setMatrixWorldNeedsUpdate;
    }
}

void PropertyBinding::_setValue_material(PropertyBinding* that, const std::vector<float>& buffer, size_t offset) {
    auto* proxy = dynamic_cast<MaterialAnimationProxy*>(that->targetObject);
    if (!proxy || !proxy->targetMaterial) return;
    setMaterialProperty(proxy->targetMaterial.get(), that->propertyName, buffer, offset);
}

void PropertyBinding::_getValue_material(PropertyBinding* that, std::vector<float>& buffer, size_t offset) {
    auto* proxy = dynamic_cast<MaterialAnimationProxy*>(that->targetObject);
    if (!proxy || !proxy->targetMaterial) return;
    getMaterialProperty(proxy->targetMaterial.get(), that->propertyName, buffer, offset);
}

void PropertyBinding::unbind() {

    this->node = nullptr;

    // back to the prototype version of getValue / setValue
    // note: avoiding to mutate the shape of 'this' via 'delete'
    this->_getValue = _getValue_unbound;
    this->_setValue = _setValue_unbound;
}
