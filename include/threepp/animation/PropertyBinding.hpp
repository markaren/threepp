
#ifndef THREEPP_PROPERTYBINDING_HPP
#define THREEPP_PROPERTYBINDING_HPP

#include <iostream>
#include <variant>
#include <vector>

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/objects/Bone.hpp"

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

        static void _getValue_direct(PropertyBinding* that, std::vector<float>& buffer, size_t offset) {

            if (that->propertyName == "quaternion") {
                that->targetObject->quaternion.toArray(buffer, offset);
            } else if (that->propertyName == "position") {
                that->targetObject->position.toArray(buffer, offset);
            } else if (that->propertyName == "scale") {
                that->targetObject->scale.toArray(buffer, offset);
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
            } else {
                std::cerr << that->propertyName << " is not writable." << std::endl;
            }
        }

        static void _setValue_direct_setMatrixWorldNeedsUpdate(PropertyBinding* that, const std::vector<float>& buffer, size_t offset) {

            if (that->propertyName == "quaternion") {
                that->targetObject->quaternion.fromArray(buffer, offset);
            } else if (that->propertyName == "position") {
                that->targetObject->position.fromArray(buffer, offset);
            }  else if (that->propertyName == "scale") {
                that->targetObject->scale.fromArray(buffer, offset);
            }  else {
                std::cerr << "_setValue_direct_setMatrixWorldNeedsUpdate: " << that->propertyName << " is not writable." << std::endl;
            }
            that->targetObject->matrixWorldNeedsUpdate = true;
        }

        void bind();

        void unbind();

        static TrackResults parseTrackName(const std::string& trackName);

        static Object3D* findNode(Object3D* root, const std::string& nodeName);

    };

}// namespace threepp

#endif//THREEPP_PROPERTYBINDING_HPP
