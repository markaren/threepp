
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
    };

}// namespace threepp

#endif//THREEPP_MODELLOADER_HPP
