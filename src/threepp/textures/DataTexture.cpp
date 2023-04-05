
#include "threepp/textures/DataTexture.hpp"

using namespace threepp;

DataTexture::DataTexture(const std::vector<unsigned char>& data, unsigned int width, unsigned int height) {

    auto tmp = static_cast<unsigned char*>(malloc(sizeof(unsigned char) * data.size()));
    for (unsigned i = 0; i < data.size(); i++) {
        tmp[i] = data[i];
    }

    this->image = Image{std::shared_ptr<unsigned char>(tmp, free), width, height};

    this->magFilter = NearestFilter;
    this->minFilter = NearestFilter;

    this->generateMipmaps = false;
    this->unpackAlignment = 1;

    this->needsUpdate();
}

std::shared_ptr<DataTexture> DataTexture::create(const std::vector<unsigned char>& data, unsigned int width, unsigned int height) {

    return std::shared_ptr<DataTexture>(new DataTexture(data, width, height));
}
