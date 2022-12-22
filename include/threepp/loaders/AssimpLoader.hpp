
#ifndef THREEPP_ASSIMPLOADER_HPP
#define THREEPP_ASSIMPLOADER_HPP

#include "threepp/objects/Group.hpp"

#include <filesystem>

namespace threepp {

    class AssimpLoader {

    public:
        explicit AssimpLoader(bool basicMaterial = false);

        std::shared_ptr<Group> load(const std::filesystem::path& path);

        ~AssimpLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

    };

}

#endif//THREEPP_ASSIMPLOADER_HPP
