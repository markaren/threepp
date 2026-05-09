
#ifndef THREEPP_MODELLOADER_HPP
#define THREEPP_MODELLOADER_HPP

#include "threepp/loaders/Loader.hpp"
#include "threepp/objects/Group.hpp"

#include <filesystem>
#include <memory>

namespace threepp {

    class ModelLoader : public Loader<Group> {

    public:
        // Dispatches to the appropriate loader based on file extension.
        // Supported: .obj, .dae, .gltf, .glb, .stl
        [[nodiscard]] std::shared_ptr<Group> load(const std::filesystem::path& path) override;

        // Propagates to inner loaders that have a file-level up-axis (Collada,
        // USD). Use when this ModelLoader is being driven by an outer system
        // (URDF/SDF/MJCF) that owns the coordinate frame.
        ModelLoader& setIgnoreUpDirection(bool ignore);

    private:
        bool ignoreUpDirection_ = false;
    };

}// namespace threepp

#endif//THREEPP_MODELLOADER_HPP
