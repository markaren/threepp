
#ifndef THREEPP_URDFLOADER_HPP
#define THREEPP_URDFLOADER_HPP

#include <memory>
#include <filesystem>

namespace threepp {

    class Object3D;

    class URDFLoader {

    public:
        explicit URDFLoader();

        std::shared_ptr<Object3D> load(const std::filesystem::path& path);

        ~URDFLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}

#endif//THREEPP_URDFLOADER_HPP
