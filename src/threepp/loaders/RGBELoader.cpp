
#include "threepp/loaders/RGBELoader.hpp"

#include "threepp/loaders/HdrTexture.hpp"

// stb_image.h is already compiled via ImageLoader.cpp — just declare the API.
#include "stb_image.h"

#include <iostream>

using namespace threepp;

std::shared_ptr<Texture> RGBELoader::load(const std::filesystem::path& path, bool flipY) {

    if (!std::filesystem::exists(path)) {
        std::cerr << "[RGBELoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
        return nullptr;
    }

    int width{}, height{}, channels{};
    // stbi_loadf decodes RGBE encoding internally and returns linear float RGB(A).
    // Request 3 channels from stbi — then pad to RGBA ourselves.
    // GL handles rgba32f more reliably than a 3-channel float format, and not
    // every API exposes an rgb32float at all.
    // Flip manually instead of via stbi_set_flip_vertically_on_load: that flag
    // is process-global and leaked into every later stbi decode (glTF embedded
    // textures came back upside-down once any HDR had loaded with flipY=true —
    // previously masked because ImageLoader re-set the global on every call).
    float* pixels = stbi_loadf(path.string().c_str(), &width, &height, &channels, 3);

    if (!pixels) {
        std::cerr << "[RGBELoader] Failed to load '" << path.string() << "': " << stbi_failure_reason() << std::endl;
        return nullptr;
    }

    const int nPixels = width * height;
    std::vector<float> data(nPixels * 4);
    for (int y = 0; y < height; ++y) {
        const int srcY = flipY ? (height - 1 - y) : y;
        for (int x = 0; x < width; ++x) {
            const int dst = (y * width + x) * 4;
            const int src = (srcY * width + x) * 3;
            data[dst + 0] = pixels[src + 0];
            data[dst + 1] = pixels[src + 1];
            data[dst + 2] = pixels[src + 2];
            data[dst + 3] = 1.0f;
        }
    }
    stbi_image_free(pixels);

    // stbi_loadf already decoded RGBE → linear floats, which is what
    // makeHdrTexture (shared with EXRLoader) tags the result as.
    return detail::makeHdrTexture(std::move(data),
                                  static_cast<unsigned int>(width),
                                  static_cast<unsigned int>(height),
                                  path.stem().string());
}
