
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

    stbi_set_flip_vertically_on_load(flipY);

    int width{}, height{}, channels{};
    // stbi_loadf decodes RGBE encoding internally and returns linear float RGB(A).
    // Request 3 channels from stbi — then pad to RGBA ourselves.
    // WebGPU has no rgb32float format; GL also handles rgba32f more reliably.
    float* pixels = stbi_loadf(path.string().c_str(), &width, &height, &channels, 3);

    if (!pixels) {
        std::cerr << "[RGBELoader] Failed to load '" << path.string() << "': " << stbi_failure_reason() << std::endl;
        return nullptr;
    }

    const int nPixels = width * height;
    std::vector<float> data(nPixels * 4);
    for (int i = 0; i < nPixels; ++i) {
        data[4 * i + 0] = pixels[3 * i + 0];
        data[4 * i + 1] = pixels[3 * i + 1];
        data[4 * i + 2] = pixels[3 * i + 2];
        data[4 * i + 3] = 1.0f;
    }
    stbi_image_free(pixels);

    Image image{std::move(data), static_cast<unsigned int>(width), static_cast<unsigned int>(height), 0};

    auto texture = Texture::create(image);
    texture->name = path.stem().string();
    texture->format = Format::RGBA;
    texture->type = Type::Float;
    texture->colorSpace = ColorSpace::Linear;// stbi_loadf already decoded RGBE → linear floats
    texture->mapping = Mapping::EquirectangularReflection;
    texture->needsUpdate();

    return texture;
}
