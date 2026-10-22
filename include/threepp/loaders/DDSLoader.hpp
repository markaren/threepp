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

        // flipY true reverses the rows to the bottom-up order stb_image's
        // flipY=true gives every other format; false keeps the DDS's own
        // top-down order (glTF's UV convention).
        std::shared_ptr<Texture> load(const std::filesystem::path& path, bool flipY = true);
        std::shared_ptr<Texture> loadFromMemory(const std::vector<unsigned char>& data, bool flipY = true);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp
