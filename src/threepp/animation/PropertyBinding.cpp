
#include "threepp/animation/PropertyBinding.hpp"
#include "threepp/objects/SkinnedMesh.hpp"

#include <regex>
#include <stdexcept>

using namespace threepp;

namespace {

    enum BindingType {
        Direct = 0,
        EntireArray = 1,
        ArrayElement = 2,
        HasFromToArray = 3
    };

    enum Versioning {
        None = 0,
        NeedsUpdate = 1,
        MatrixWorldNeedsUpdate = 2
    };

    // Characters [].:/ are reserved for track binding syntax.
    const std::string RESERVED_CHARS_RE = "\\[\\]\\.:\\/";
    const std::string RESERVED_RE = "[" + RESERVED_CHARS_RE + "]";

    // Attempts to allow node names from any language.
    // Excludes reserved characters and matches everything else.
    const std::string WORD_CHAR = "[^" + RESERVED_CHARS_RE + "]";
    const std::string WORD_CHAR_OR_DOT = "[^" + RESERVED_CHARS_RE + "\\." + "]";

    // Parent directories, delimited by '/' or ':'.
    // Currently unused, but must be matched to parse the rest of the track name.
    const std::string DIRECTORY_RE = "((?:" + WORD_CHAR + "+[\\/:])*)";

    // Target node. May contain word characters (a-zA-Z0-9_) and '.' or '-'.
    const std::string NODE_RE = "(" + WORD_CHAR_OR_DOT + "+)?";

    // Object on target node, and accessor.
    // May not contain reserved characters.
    // Accessor may contain any character except closing bracket.
    const std::string OBJECT_RE = std::string("(?:\\.") + "(" + WORD_CHAR + "+)" + "(?:\\[([^\\]]+)\\])?)?";

    // Property and accessor.
    // May not contain reserved characters.
    // Accessor may contain any non-bracket characters.
    const std::string PROPERTY_RE = std::string("\\.") + "(" + WORD_CHAR + "+)" + "(?:\\[([^\\]]+)\\])?";

    const std::string TRACK_REGEX_PATTERN = "^" + DIRECTORY_RE + NODE_RE + OBJECT_RE + PROPERTY_RE + "$";

    const std::regex trackRegex(TRACK_REGEX_PATTERN);

    const std::vector<std::string> _supportedObjectNames{"material", "materials", "bones"};

    Object3D* searchNodeSubtree(const std::string& nodeName, std::vector<Object3D*> children) {

        for (unsigned i = 0; i < children.size(); i++) {

            const auto childNode = children[i];

            if (childNode->name == nodeName || childNode->uuid == nodeName) {

                return childNode;
            }

            const auto result = searchNodeSubtree(nodeName, childNode->children);

            if (result) return result;
        }

        return nullptr;
    };

}// namespace

threepp::PropertyBinding::TrackResults threepp::PropertyBinding::parseTrackName(const std::string& trackName) {

    std::smatch matches;
    if (!std::regex_match(trackName, matches, trackRegex)) {
        throw std::runtime_error("PropertyBinding: Cannot parse trackName: " + trackName);
    }

    TrackResults results;
    results.nodeName = matches[2];
    results.objectName = matches[3];
    results.objectIndex = matches[4];
    results.propertyName = matches[5];
    results.propertyIndex = matches[6];

    auto lastDot = results.nodeName.find_last_of('.');
    if (lastDot != std::string::npos) {
        std::string objectName = results.nodeName.substr(lastDot + 1);
        // Check if objectName is in the allowlist
        // (You'll need to define _supportedObjectNames)
        if (std::find(_supportedObjectNames.begin(), _supportedObjectNames.end(), objectName) != _supportedObjectNames.end()) {
            results.nodeName = results.nodeName.substr(0, lastDot);
            results.objectName = objectName;
        }
    }

    if (results.propertyName.empty()) {
        throw std::runtime_error("PropertyBinding: Cannot parse propertyName from trackName: " + trackName);
    }

    return results;
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
    auto targetObject = this->node;
    const auto parsedPath = this->parsedPath;

    const auto objectName = parsedPath.objectName;
    const auto propertyName = parsedPath.propertyName;
    auto propertyIndex = parsedPath.propertyIndex;

    if (!targetObject) {

        targetObject = PropertyBinding::findNode(this->rootNode, parsedPath.nodeName);

        if (!targetObject) targetObject = this->rootNode;

        this->node = targetObject;
    }

    // set fail state so we can just 'return' on error
    //            this->_getValue = this->_getValue_unavailable;
    //            this->_setValue = this->_setValue_unavailable;

    // ensure there is a value node
    if (!targetObject) {

        //        console.error( 'THREE.PropertyBinding: Trying to update node for track: ' + this->path + ' but it wasn\'t found.' );
        return;
    }

    if (objectName) {

        auto objectIndex = parsedPath.objectIndex;

        // special cases were we need to reach deeper into the hierarchy to get the face materials....
        if (objectName == "materials") {

            if (!targetObject.material) {

                console.error('THREE.PropertyBinding: Can not bind to material as node does not have a material.', this);
                return;
            }

            if (!targetObject.material.materials) {

                console.error('THREE.PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.', this);
                return;
            }

            targetObject = targetObject.material.materials;
        } else if (objectName == "bones") {

            if (!targetObject.skeleton) {

//                console.error('THREE.PropertyBinding: Can not bind to bones as node does not have a skeleton.', this);
                return;
            }

            // potential future optimization: skip this if propertyIndex is already an integer
            // and convert the integer string to a true integer.

            targetObject = targetObject.skeleton.bones;

            // support resolving morphTarget names into indices.
            for (let i = 0; i < targetObject.length; i++) {

                if (targetObject[i].name == = objectIndex) {

                    objectIndex = i;
                    break;
                }
            }

        } else {

            if (targetObject[objectName] == = undefined) {

//                console.error('THREE.PropertyBinding: Can not bind to objectName of node undefined.', this);
                return;
            }

            targetObject = targetObject[objectName];
        }


        if (objectIndex != = undefined) {

            if (targetObject[objectIndex] == = undefined) {

//                console.error('THREE.PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.', this, targetObject);
                return;
            }

            targetObject = targetObject[objectIndex];
        }
    }

    // resolve property
    const auto& nodeProperty = targetObject[propertyName];

    if (nodeProperty == = undefined) {

        const auto& nodeName = parsedPath.nodeName;

//        console.error('THREE.PropertyBinding: Trying to update property for track: ' + nodeName +
//                              '.' + propertyName + ' but it wasn\'t found.',
//                      targetObject);
        return;
    }

    // determine versioning scheme
    auto versioning = Versioning::None;

    this->targetObject = targetObject;

    if (targetObject.needsUpdate != = undefined) {// material

        versioning = this->Versioning.NeedsUpdate;

    } else if (targetObject.matrixWorldNeedsUpdate != = undefined) {// node transform

        versioning = this->Versioning.MatrixWorldNeedsUpdate;
    }

    // determine how the property gets bound
    auto bindingType = BindingType::Direct;

    if (propertyIndex != undefined) {

        // access a sub element of the property array (only primitives are supported right now)

        if (propertyName == "morphTargetInfluences") {

            // potential optimization, skip this if propertyIndex is already an integer, and convert the integer string to a true integer.

            // support resolving morphTarget names into indices.
            if (!targetObject.geometry) {

                //                        console.error( 'THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.', this );
                return;
            }

            if (targetObject.geometry.isBufferGeometry) {

                if (!targetObject.geometry.morphAttributes) {

                    //                            console.error( 'THREE.PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.', this );
                    return;
                }

                if (targetObject.morphTargetDictionary[propertyIndex] != = undefined) {

                    propertyIndex = targetObject.morphTargetDictionary[propertyIndex];
                }


            } else {

                //                        std::cout ( 'THREE.PropertyBinding: Can not bind to morphTargetInfluences on THREE.Geometry. Use THREE.BufferGeometry instead.', this );
                return;
            }
        }

        bindingType = BindingType::ArrayElement;

        this->resolvedProperty = nodeProperty;
        this->propertyIndex = propertyIndex;

    } else if (nodeProperty.fromArray != = undefined&& nodeProperty.toArray != = undefined) {

        // must use copy for Object3D.Euler/Quaternion

        bindingType = BindingType::HasFromToArray;

        this->resolvedProperty = nodeProperty;

    } else if (Array.isArray(nodeProperty)) {

        bindingType = BindingType::EntireArray;

        this->resolvedProperty = nodeProperty;

    } else {

        this->propertyName = propertyName;
    }

    // select getter / setter
    this->_getValue = this->GetterByBindingType[bindingType];
    this->_setValue = this->SetterByBindingTypeAndVersioning[bindingType][versioning];
}

void PropertyBinding::unbind() {

    this->node = nullptr;

    // back to the prototype version of getValue / setValue
    // note: avoiding to mutate the shape of 'this' via 'delete'
    this->_getValue = _getValue_unbound;
    this->_setValue = _setValue_unbound;
}
