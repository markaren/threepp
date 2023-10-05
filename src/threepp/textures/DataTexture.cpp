
#include "threepp/textures/DataTexture.hpp"

using namespace threepp;

DataTexture::DataTexture(const std::vector<unsigned char>& data, unsigned int width, unsigned int height) {

    this->image = Image{data, width, height};

    this->magFilter = NearestFilter;
    this->minFilter = NearestFilter;

    this->generateMipmaps = false;
    this->unpackAlignment = 1;

    this->needsUpdate();
}

std::shared_ptr<DataTexture> DataTexture::create(size_t size, unsigned int width, unsigned int height) {

    return std::shared_ptr<DataTexture>(new DataTexture(std::vector<unsigned char>(size), width, height));
}

std::shared_ptr<DataTexture> DataTexture::create(const std::vector<unsigned char>& data, unsigned int width, unsigned int height) {

    return std::shared_ptr<DataTexture>(new DataTexture(data, width, height));
}
