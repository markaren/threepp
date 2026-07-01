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

    // Cheap CPU-side scan for BC1/DXT1 punch-through alpha, without decompressing
    // the image: reads only the block colour endpoints and checks how many 4x4
    // blocks use the color0<=color1 "3-colour + transparent" encoding. Real cutout
    // textures (lattice/mesh patterns) land far above the incidental-tie rate of
    // fully opaque textures. Returns false for non-DXT1 formats (DXT3/DXT5/BC7
    // carry an explicit alpha channel and don't need this heuristic) or if the
    // file can't be read as a DDS.
    bool ddsHasCutoutAlpha(const std::filesystem::path& path);

}// namespace threepp
