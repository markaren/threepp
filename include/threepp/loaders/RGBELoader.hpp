
#ifndef THREEPP_RGBELOADER_HPP
#define THREEPP_RGBELOADER_HPP

#include <filesystem>
#include <memory>

namespace threepp {

    class Texture;

    class RGBELoader {

    public:
        // Load a Radiance HDR (.hdr) file.
        // Returns a float-type RGB texture with EquirectangularReflection mapping,
        // ready to assign to scene.background / scene.environment.
        std::shared_ptr<Texture> load(const std::filesystem::path& path, bool flipY = true);
    };

}// namespace threepp

#endif//THREEPP_RGBELOADER_HPP
