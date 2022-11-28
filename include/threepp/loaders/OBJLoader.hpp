
#ifndef THREEPP_OBJLOADER_HPP
#define THREEPP_OBJLOADER_HPP

#include <filesystem>

#include "threepp/objects/Group.hpp"
#include "threepp/loaders/MTLLoader.hpp"

namespace threepp {

    class OBJLoader {

    public:

        std::optional<MaterialCreator> materials;

        std::shared_ptr<Group> load(const std::filesystem::path& path, bool tryLoadMtl = true);
    };

}// namespace threepp

#endif//THREEPP_OBJLOADER_HPP
