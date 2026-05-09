#pragma once

#include "threepp/loaders/MaterialVariants.hpp"

#include <filesystem>
#include <memory>

namespace threepp {

    class Group;
    class Texture;

    struct USDResult {
        std::shared_ptr<Group> scene;
        MaterialVariants variants;///< Named material variants (empty in current implementation)
        /// HDR/equirect texture authored by the file's UsdLuxDomeLight, if any.
        /// Assign to `scene.background` and `scene.environment` for IBL. Null
        /// when the file has no DomeLight or its `inputs:texture:file` could
        /// not be resolved.
        std::shared_ptr<Texture> environment;
    };

    class USDLoader {
    public:
        USDLoader();
        ~USDLoader();

        /// When true, the file-level upAxis metadata is ignored and the
        /// returned scene is left in its file-native orientation. Use this when
        /// the mesh is being composed into an outer system (URDF/SDF/MJCF) that
        /// owns the coordinate frame.
        USDLoader& setIgnoreUpDirection(bool ignore);

        /// Load a USD/USDA/USDC/USDZ file. Returns the scene root only.
        std::shared_ptr<Group> load(const std::filesystem::path& path);

        /// Load a USD file and return full result including (future) variant info.
        USDResult loadFull(const std::filesystem::path& path);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp
