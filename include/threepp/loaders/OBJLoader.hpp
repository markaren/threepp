
#ifndef THREEPP_OBJLOADER_HPP
#define THREEPP_OBJLOADER_HPP

#include "threepp/objects/Group.hpp"
#include <filesystem>

namespace threepp {

    class OBJLoader {

    public:
        std::shared_ptr<Group> load(const std::filesystem::path& path);
    };

}// namespace threepp

#endif//THREEPP_OBJLOADER_HPP
