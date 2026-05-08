
#ifndef THREEPP_PROPERTYBINDING_HPP
#define THREEPP_PROPERTYBINDING_HPP

#include <iostream>
#include <variant>
#include <vector>

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"

namespace threepp {

    class PropertyBinding {

        using GetterFn = void (*)(PropertyBinding*, std::vector<float>&, size_t);
        using SetterFn = void (*)(PropertyBinding*, const std::vector<float>&, size_t);

    public:
        struct TrackResults {
            std::string nodeName;
            std::string propertyName;
        };

        GetterFn _getValue;
        SetterFn _setValue;

        void setValue(const std::vector<float>& buffer, size_t offset) {

            _setValue(this, buffer, offset);
        }

        void getValue(std::vector<float>& buffer, size_t offset) {

            _getValue(this, buffer, offset);
        }

        static std::shared_ptr<PropertyBinding> create(Object3D* root, const std::string& path, const std::optional<TrackResults>& parsedPath) {

            return std::shared_ptr<PropertyBinding>(new PropertyBinding(root, path, parsedPath));
        }

    private:
        friend class AnimationMixer;

        Object3D* rootNode;
        Object3D* node = nullptr;

        std::string path{};
        TrackResults parsedPath;

        using ResolvedProperty = std::variant<Vector3*, Quaternion*>;
        ResolvedProperty resolvedProperty;

        Object3D* targetObject = nullptr;
        std::string propertyName;

        PropertyBinding(Object3D* rootNode, const std::string& path, const std::optional<TrackResults>& parsedPath)
            : rootNode(rootNode), path(path), parsedPath(parsedPath.value_or(parseTrackName(path))) {

            this->_getValue = _getValue_unbound;
            this->_setValue = _setValue_unbound;
        }

        static void _getValue_unavailable(PropertyBinding*, std::vector<float>&, size_t) {}

        static void _setValue_unavailable(PropertyBinding*, const std::vector<float>&, size_t) {}

        static void _getValue_unbound(PropertyBinding* that, std::vector<float>& targetArray, size_t offset) {

            that->bind();
            that->getValue(targetArray, offset);
        }

        static void _setValue_unbound(PropertyBinding* that, const std::vector<float>& sourceArray, size_t offset) {

            that->bind();
            that->setValue(sourceArray, offset);
        }

        static int parseMorphIndex(const std::string& name) {
            if (name.size() > 23 && name.compare(0, 22, "morphTargetInfluences[") == 0 && name.back() == ']') {
                return std::stoi(name.substr(22, name.size() - 23));
            }
            return -1;
        }

        static void _getValue_direct(PropertyBinding* that, std::vector<float>& buffer, size_t offset) {

            if (that->propertyName == "quaternion") {
                that->targetObject->quaternion.toArray(buffer, offset);
            } else if (that->propertyName == "position") {
                that->targetObject->position.toArray(buffer, offset);
            } else if (that->propertyName == "scale") {
                that->targetObject->scale.toArray(buffer, offset);
            } else if (int idx = parseMorphIndex(that->propertyName); idx >= 0) {
                if (auto* m = dynamic_cast<ObjectWithMorphTargetInfluences*>(that->targetObject)) {
                    auto& inf = m->morphTargetInfluences();
                    buffer[offset] = idx < static_cast<int>(inf.size()) ? inf[idx] : 0.f;
                }
            } else {
                std::cerr << that->propertyName << " is not readable." << std::endl;
            }
        }

        static void _setValue_direct(PropertyBinding* that, const std::vector<float>& buffer, size_t offset) {

            if (that->propertyName == "quaternion") {
                that->targetObject->quaternion.fromArray(buffer, offset);
            } else if (that->propertyName == "position") {
                that->targetObject->position.fromArray(buffer, offset);
            } else if (that->propertyName == "scale") {
                that->targetObject->scale.fromArray(buffer, offset);
            } else if (int idx = parseMorphIndex(that->propertyName); idx >= 0) {
                if (auto* m = dynamic_cast<ObjectWithMorphTargetInfluences*>(that->targetObject)) {
                    auto& inf = m->morphTargetInfluences();
                    if (idx >= static_cast<int>(inf.size())) inf.resize(idx + 1, 0.f);
                    inf[idx] = buffer[offset];
                }
            } else {
                std::cerr << that->propertyName << " is not writable." << std::endl;
            }
        }

        static void _setValue_direct_setMatrixWorldNeedsUpdate(PropertyBinding* that, const std::vector<float>& buffer, size_t offset) {

            if (that->propertyName == "quaternion") {
                that->targetObject->quaternion.fromArray(buffer, offset);
            } else if (that->propertyName == "position") {
                that->targetObject->position.fromArray(buffer, offset);
            } else if (that->propertyName == "scale") {
                that->targetObject->scale.fromArray(buffer, offset);
            } else if (int idx = parseMorphIndex(that->propertyName); idx >= 0) {
                if (auto* m = dynamic_cast<ObjectWithMorphTargetInfluences*>(that->targetObject)) {
                    auto& inf = m->morphTargetInfluences();
                    if (idx >= static_cast<int>(inf.size())) inf.resize(idx + 1, 0.f);
                    inf[idx] = buffer[offset];
                }
            } else {
                std::cerr << "_setValue_direct_setMatrixWorldNeedsUpdate: " << that->propertyName << " is not writable." << std::endl;
            }
            that->targetObject->matrixWorldNeedsUpdate = true;
        }

        // Material property setters/getters (for glTF KHR_animation_pointer).
        // Defined in PropertyBinding.cpp; targetObject must be a MaterialAnimationProxy.
        static void _setValue_material(PropertyBinding* that, const std::vector<float>& buffer, size_t offset);
        static void _getValue_material(PropertyBinding* that, std::vector<float>& buffer, size_t offset);

        void bind();

        void unbind();

        static TrackResults parseTrackName(const std::string& trackName);

        static Object3D* findNode(Object3D* root, const std::string& nodeName);

    };

}// namespace threepp

#endif//THREEPP_PROPERTYBINDING_HPP
