#pragma once

#include "threepp/loaders/MaterialVariants.hpp"

#include <filesystem>
#include <memory>

namespace threepp {

    class Group;

    struct USDResult {
        std::shared_ptr<Group> scene;
        MaterialVariants variants;///< Named material variants (empty in current implementation)
    };

    class USDLoader {
    public:
        USDLoader();
        ~USDLoader();

        /// Load a USD/USDA/USDC/USDZ file. Returns the scene root only.
        std::shared_ptr<Group> load(const std::filesystem::path& path);

        /// Load a USD file and return full result including (future) variant info.
        USDResult loadFull(const std::filesystem::path& path);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp
