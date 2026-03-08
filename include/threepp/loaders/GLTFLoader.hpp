#ifndef THREEPP_GLTFLOADER_HPP
#define THREEPP_GLTFLOADER_HPP

#include "threepp/threepp.hpp"

#include <filesystem>
#include <memory>
#include <optional>

namespace threepp {

    struct GLTFResult {
        std::shared_ptr<Group> scene;   ///< Root node of the loaded model
        std::vector<std::shared_ptr<Group>> scenes; ///< All scenes in the file
    };

    class GLTFLoader {
    public:
        std::optional<GLTFResult> load(const std::filesystem::path& path);

    private:
        struct Impl;
    };

} // namespace threepp

#endif
