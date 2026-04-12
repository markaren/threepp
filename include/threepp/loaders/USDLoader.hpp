#pragma once

#include <filesystem>
#include <memory>

namespace threepp {

    class Group;

    class USDLoader {
    public:
        USDLoader();
        ~USDLoader();

        std::shared_ptr<Group> load(const std::filesystem::path& path);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp
