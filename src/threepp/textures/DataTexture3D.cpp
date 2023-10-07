
#include "threepp/textures/DataTexture3D.hpp"

using namespace threepp;


DataTexture3D::DataTexture3D(const std::vector<unsigned char>& data,
                             unsigned int width, unsigned int height, unsigned int depth)
    : Texture(std::nullopt) {

    // We're going to add .setXXX() methods for setting properties later.
    // Users can still set in DataTexture3D directly.
    //
    //	const texture = new THREE.DataTexture3D( data, width, height, depth );
    // 	texture.anisotropy = 16;
    //
    // See #14839

    this->image = Image{data, width, height, depth};

    this->magFilter = Filter::Nearest;
    this->minFilter = Filter::Nearest;

    this->wrapR = TextureWrapping::ClampToEdge;

    this->generateMipmaps = false;
    this->unpackAlignment = 1;

    this->needsUpdate();
}

std::shared_ptr<DataTexture3D> DataTexture3D::create(
        const std::vector<unsigned char>& data,
        unsigned int width, unsigned int height, unsigned int depth) {

    return std::shared_ptr<DataTexture3D>(new DataTexture3D(data, width, height, depth));
}
