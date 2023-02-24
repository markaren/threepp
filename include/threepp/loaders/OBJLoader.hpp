
#ifndef THREEPP_OBJLOADER_HPP
#define THREEPP_OBJLOADER_HPP

#include <filesystem>
#include <unordered_map>

#include "threepp/loaders/MTLLoader.hpp"
#include "threepp/objects/Group.hpp"

namespace threepp {

    class OBJLoader {

    public:
        std::optional<MaterialCreator> materials;

        std::shared_ptr<Group> load(const std::filesystem::path& path, bool tryLoadMtl = true);

    private:
        std::unordered_map<std::string, std::shared_ptr<Group>> cache_;
    };

}// namespace threepp

#endif//THREEPP_OBJLOADER_HPP
