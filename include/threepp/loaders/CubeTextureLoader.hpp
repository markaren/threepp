
#ifndef THREEPP_CUBETEXTURELOADER_HPP
#define THREEPP_CUBETEXTURELOADER_HPP

#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include <array>
#include <filesystem>
#include <memory>

namespace threepp {

    class CubeTextureLoader {

    public:
        // Six faces in +x, -x, +y, -y, +z, -z order. The format is decided for
        // the SET: six JPEG faces upload as RGB, anything else (PNG faces, or a
        // mix) as RGBA with every face decoded to match. Returns nullptr when a
        // face does not exist or will not decode.
        std::shared_ptr<CubeTexture> load(const std::array<std::filesystem::path, 6>& paths,
                                          ColorSpace colorSpace = ColorSpace::sRGB);

    private:
        ImageLoader loader;
    };

}// namespace threepp

#endif//THREEPP_CUBETEXTURELOADER_HPP
