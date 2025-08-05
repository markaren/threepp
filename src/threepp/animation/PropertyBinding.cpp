
#include "threepp/animation/PropertyBinding.hpp"
#include "threepp/objects/SkinnedMesh.hpp"

#include <iostream>
#include <regex>

using namespace threepp;

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

    this->targetObject = targetObject;
    this->propertyName = propertyName;

    // select getter / setter
    this->_getValue = _getValue_direct;
    this->_setValue = _setValue_direct_setMatrixWorldNeedsUpdate;
}

void PropertyBinding::unbind() {

    this->node = nullptr;

    // back to the prototype version of getValue / setValue
    // note: avoiding to mutate the shape of 'this' via 'delete'
    this->_getValue = _getValue_unbound;
    this->_setValue = _setValue_unbound;
}
