
#ifndef THREEPP_HDRTEXTURE_HPP
#define THREEPP_HDRTEXTURE_HPP

#include "threepp/textures/Texture.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace threepp::detail {

    // Shared tail of the HDR image loaders: linear float RGBA pixels in, an
    // environment-ready Texture out. RGBELoader (.hdr) and EXRLoader (.exr)
    // differ only in how they get to the floats — if they disagreed on any
    // property below, the same sky would light a scene differently depending on
    // which file format the user happened to download.
    //
    // RGBA rather than RGB even when the file carries three channels: GL handles
    // rgba32f more reliably than a 3-channel float format, and not every API
    // exposes an rgb32float at all. The Vulkan environment path additionally
    // requires exactly this (float RGBA) and ignores anything else.
    inline std::shared_ptr<Texture> makeHdrTexture(std::vector<float> rgba,
                                                   unsigned int width,
                                                   unsigned int height,
                                                   std::string name) {

        // Moved through the vector<Image> overload rather than create(const
        // Image&): an 8k equirect is ~1 GB of floats, and the by-reference
        // overload would copy every one of them.
        std::vector<Image> images;
        images.emplace_back(std::move(rgba), width, height, 0u);

        auto texture = Texture::create(std::move(images));
        texture->name = std::move(name);
        texture->format = Format::RGBA;
        texture->type = Type::Float;
        texture->colorSpace = ColorSpace::Linear;// both decoders emit linear scene-referred floats
        texture->mapping = Mapping::EquirectangularReflection;
        // Equirect maps wrap 360° in azimuth — Repeat on S keeps the atan2 seam (at
        // -X) continuous when sampled directly as a background and when GGX-prefiltered
        // into the GL PMREM atlas (otherwise the seam bakes a vertical streak). T stays
        // clamped (the poles do not wrap).
        texture->wrapS = TextureWrapping::Repeat;
        texture->needsUpdate();

        return texture;
    }

}// namespace threepp::detail

#endif//THREEPP_HDRTEXTURE_HPP
