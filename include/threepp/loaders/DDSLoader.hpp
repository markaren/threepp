#pragma once

#include <filesystem>
#include <memory>
#include <vector>

namespace threepp {

    class Texture;

    class DDSLoader {
    public:
        DDSLoader();
        ~DDSLoader();

        std::shared_ptr<Texture> load(const std::filesystem::path& path);
        std::shared_ptr<Texture> loadFromMemory(const std::vector<unsigned char>& data);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp
