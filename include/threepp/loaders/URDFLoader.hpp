
#ifndef THREEPP_URDFLOADER_HPP
#define THREEPP_URDFLOADER_HPP

#include "threepp/loaders/Loader.hpp"
#include "threepp/objects/Robot.hpp"

#include <filesystem>
#include <memory>

namespace threepp {

    class Group;

    class URDFLoader {

    public:
        explicit URDFLoader();

        std::shared_ptr<Robot> load(Loader<Group>& loader, const std::filesystem::path& path);

        ~URDFLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_URDFLOADER_HPP
