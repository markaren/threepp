
#ifndef THREEPP_PROPERTYBINDING_HPP
#define THREEPP_PROPERTYBINDING_HPP

#include <vector>

#include "threepp/core/Object3D.hpp"

namespace threepp {

    class PropertyBinding {

    public:
        struct TrackResults {
            std::string nodeName;
            std::optional<std::string> objectName;
            std::string objectIndex;
            std::string propertyName;
            std::optional<std::string> propertyIndex;
        };

        std::function<void(PropertyBinding*, std::vector<float>&, size_t)> _getValue;
        std::function<void(PropertyBinding*, const std::vector<float>&, size_t)> _setValue;

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

        PropertyBinding(Object3D* rootNode, const std::string& path, const std::optional<TrackResults>& parsedPath)
            : rootNode(rootNode), path(path), parsedPath(parsedPath.value_or(parseTrackName(path))) {}

        static void _getValue_unbound(PropertyBinding* that, std::vector<float>& targetArray, size_t offset) {

            that->bind();
            that->getValue(targetArray, offset);
        }

        static void _setValue_unbound(PropertyBinding* that, const std::vector<float>& sourceArray, size_t offset) {

            that->bind();
            that->setValue(sourceArray, offset);
        }

        void bind();

        void unbind();

        static TrackResults parseTrackName(const std::string& trackName);

        static Object3D* findNode( Object3D* root, const std::string& nodeName );
    };

}// namespace threepp

#endif//THREEPP_PROPERTYBINDING_HPP
