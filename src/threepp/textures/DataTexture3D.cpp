
#include "threepp/textures/DataTexture3D.hpp"

using namespace threepp;


DataTexture3D::DataTexture3D(std::vector<unsigned char> data, unsigned int width, unsigned int height, unsigned int depth)
    : Texture(std::nullopt) {

    // We're going to add .setXXX() methods for setting properties later.
    // Users can still set in DataTexture3D directly.
    //
    //	const texture = new THREE.DataTexture3D( data, width, height, depth );
    // 	texture.anisotropy = 16;
    //
    // See #14839

    this->image = Image{ std::make_shared<unsigned char>(*data.data()), width, height, depth };

    this->magFilter = NearestFilter;
    this->minFilter = NearestFilter;

    this->wrapR = ClampToEdgeWrapping;

    this->generateMipmaps = false;
    this->flipY = false;
    this->unpackAlignment = 1;

    this->needsUpdate();
}
