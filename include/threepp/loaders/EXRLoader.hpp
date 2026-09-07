
#ifndef THREEPP_EXRLOADER_HPP
#define THREEPP_EXRLOADER_HPP

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Texture;

    // Load an OpenEXR (.exr) image.
    //
    // Returns a float-type RGBA texture with EquirectangularReflection mapping,
    // ready to assign to scene.background / scene.environment — the same shape
    // RGBELoader returns for .hdr, so the two are interchangeable at the call
    // site.
    //
    // Supported subset: single-part scanline images with NONE, RLE, ZIPS, ZIP or
    // PIZ compression and HALF / FLOAT / UINT channels. That covers what Blender
    // writes, what the HDRI sites ship, and the lossless codec VFX pipelines use.
    // Tiled, deep and multi-part files, and the PXR24 / B44 / DWAA / DWAB codecs,
    // are rejected with a message naming what was found rather than decoded into
    // garbage.
    //
    // Channels are matched by name: R, G, B and A, or Y alone for a luminance
    // image (broadcast to RGB). Alpha defaults to 1 when the file has none.
    class EXRLoader {

    public:
        std::shared_ptr<Texture> load(const std::filesystem::path& path, bool flipY = true);

        std::shared_ptr<Texture> loadFromMemory(const std::vector<unsigned char>& data,
                                                const std::string& name = {},
                                                bool flipY = true);
    };

}// namespace threepp

#endif//THREEPP_EXRLOADER_HPP
