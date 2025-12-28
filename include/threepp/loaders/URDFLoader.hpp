
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

        URDFLoader& setGeometryLoader(std::shared_ptr<Loader<Group>> loader);

        std::shared_ptr<Robot> load(const std::filesystem::path& path);

        std::shared_ptr<Robot> parse(const std::filesystem::path& baseDir, const std::string& xml);

        ~URDFLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_URDFLOADER_HPP
