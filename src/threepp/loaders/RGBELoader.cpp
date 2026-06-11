
#include "threepp/loaders/RGBELoader.hpp"

#include "threepp/textures/Texture.hpp"

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
    // WebGPU has no rgb32float format; GL also handles rgba32f more reliably.
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

    Image image{std::move(data), static_cast<unsigned int>(width), static_cast<unsigned int>(height), 0};

    auto texture = Texture::create(image);
    texture->name = path.stem().string();
    texture->format = Format::RGBA;
    texture->type = Type::Float;
    texture->colorSpace = ColorSpace::Linear;// stbi_loadf already decoded RGBE → linear floats
    texture->mapping = Mapping::EquirectangularReflection;
    // Equirect maps wrap 360° in azimuth — Repeat on S keeps the atan2 seam (at
    // -X) continuous when sampled directly as a background and when GGX-prefiltered
    // into the GL PMREM atlas (otherwise the seam bakes a vertical streak). T stays
    // clamped (the poles do not wrap).
    texture->wrapS = TextureWrapping::Repeat;
    texture->needsUpdate();

    return texture;
}
