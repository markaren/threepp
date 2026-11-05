
#ifndef THREEPP_TIFF_TIFFDECODE_HPP
#define THREEPP_TIFF_TIFFDECODE_HPP

#include "threepp/textures/Image.hpp"

#include <cstddef>
#include <optional>

namespace threepp::detail {

    // "II*\0" (little-endian) or "MM\0*" (big-endian) at the start of `data`.
    bool isTiff(const unsigned char* data, std::size_t size);

    // Decode the first image of a baseline TIFF into 8-bit samples, the way
    // ImageLoader hands back stb's images: `channels` 1..4 picks grey, grey +
    // alpha, RGB or RGBA output, and `flipY` puts the bottom row first.
    //
    // Reads strips and tiles, chunky and planar layouts, no compression,
    // PackBits, LZW and Deflate, the horizontal predictor, 8- and 16-bit
    // unsigned samples (16-bit keep their high byte), and grey, RGB and
    // palette images. The first extra sample is taken as alpha whatever
    // ExtraSamples calls it: exporters (Blender among them) write a texture's
    // cutout mask as "unspecified". Associated alpha is un-premultiplied.
    //
    // Everything else (JPEG-in-TIFF, CMYK, YCbCr, float samples, BigTIFF, bit
    // depths other than 8 and 16) returns nullopt with a line on stderr naming
    // what was found.
    std::optional<Image> decodeTiff(const unsigned char* data, std::size_t size, int channels, bool flipY);

}// namespace threepp::detail

#endif//THREEPP_TIFF_TIFFDECODE_HPP
