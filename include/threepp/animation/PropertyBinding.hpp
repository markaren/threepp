
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
            std::optional<std::string> objectName;
            std::string objectIndex;
            std::string propertyName;
            std::optional<std::string> propertyIndex;
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

        std::string path;
        TrackResults parsedPath;

        using ResolvedProperty = std::variant<float*, Vector3*, Quaternion*, Matrix4*>;
        ResolvedProperty resolvedProperty;

        using TargetObject = std::variant<std::monostate, Object3D*, Material*, Bone*, std::vector<std::shared_ptr<Bone>>>;
        TargetObject targetObject{std::monostate()};
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
                std::get<Object3D*>(that->targetObject)->quaternion.toArray(buffer, offset);
            } else if (that->propertyName == "vector") {
                std::get<Object3D*>(that->targetObject)->position.toArray(buffer, offset);
            } else {
                std::cerr << that->propertyName << " is not readable." << std::endl;
            }
        }

        static void _getValue_array(PropertyBinding* that, std::vector<float>& buffer, size_t offset, size_t n) {
            if (auto ptr = std::get_if<float*>(&that->resolvedProperty)) {
                float* source = *ptr;
                for (size_t i = 0; i < n; ++i) {
                    buffer[offset + i] = source[i];
                }
            }
        }

        static void _getValue_arrayElement(PropertyBinding* that, std::vector<float>& buffer, size_t offset) {
            if (auto ptr = std::get_if<float*>(&that->resolvedProperty)) {
                // Assuming propertyIndex is stored as an integer member
                buffer[offset] = (*ptr)[std::stoi(that->parsedPath.propertyIndex.value_or("0"))];
            }
        }

        static void _getValue_toArray(PropertyBinding* that, std::vector<float>& buffer, size_t offset) {
            if (auto v3 = std::get_if<Vector3*>(&that->resolvedProperty)) {
                (*v3)->toArray(buffer, offset);
            } else if (auto q = std::get_if<Quaternion*>(&that->resolvedProperty)) {
                (*q)->toArray(buffer, offset);
            } else if (auto m4 = std::get_if<Matrix4*>(&that->resolvedProperty)) {
                (*m4)->toArray(buffer, offset);
            }
        }

        static void _setValue_direct(PropertyBinding* that, const std::vector<float>& buffer, size_t offset) {

            if (that->propertyName == "quaternion") {
                std::get<Object3D*>(that->targetObject)->quaternion.fromArray(buffer, offset);
            } else if (that->propertyName == "vector") {
                std::get<Object3D*>(that->targetObject)->position.fromArray(buffer, offset);
            } else {
                std::cerr << that->propertyName << " is not writable." << std::endl;
            }
        }

        static void _setValue_direct_setNeedsUpdate(PropertyBinding* that, const std::vector<float>& buffer, size_t offset) {
            std::cerr << that->propertyName << " is not writable." << std::endl;
            std::get<Material*>(that->targetObject)->needsUpdate();
        }

        void bind();

        void unbind();

        static TrackResults parseTrackName(const std::string& trackName);

        static Object3D* findNode(Object3D* root, const std::string& nodeName);

        static const GetterFn GetterByBindingType[];

        static const SetterFn SetterByBindingTypeAndVersioning[4][3];

        static TargetObject resolveTargetObject(TargetObject& rootNode, const std::string& property) {

            return {};
        }

        static ResolvedProperty resolveProperty(TargetObject& obj, const std::string& property) {
            if (property == "position") {
                if (auto objPtr = std::get_if<Object3D*>(&obj)) {
                    return &(*objPtr)->position;
                }
            } else if (property == "quaternion") {
                if (auto objPtr = std::get_if<Object3D*>(&obj)) {
                    return &(*objPtr)->quaternion;
                }
            } else if (property == "scale") {
                if (auto objPtr = std::get_if<Object3D*>(&obj)) {
                    return &(*objPtr)->scale;
                }
            } else if (property == "matrix") {
                if (auto objPtr = std::get_if<Object3D*>(&obj)) {
                    return (*objPtr)->matrix.get();
                }
            } else if (property == "matrixWorld") {
                if (auto objPtr = std::get_if<Object3D*>(&obj)) {
                    return (*objPtr)->matrixWorld.get();
                }
            }
        }
    };

}// namespace threepp

#endif//THREEPP_PROPERTYBINDING_HPP
