
#ifndef THREEPP_URDFLOADER_HPP
#define THREEPP_URDFLOADER_HPP

#include <memory>
#include <filesystem>

namespace threepp {

    class Group;

    template <class T>
    class Loader {
    public:
        virtual std::shared_ptr<T> load(const std::filesystem::path& path) = 0;

        virtual ~Loader() = default;
    };

    class URDFLoader {

    public:
        explicit URDFLoader();

        std::shared_ptr<Group> load(Loader<Group>& loader, const std::filesystem::path& path);

        ~URDFLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}

#endif//THREEPP_URDFLOADER_HPP
