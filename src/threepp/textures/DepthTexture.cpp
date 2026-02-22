
#include "threepp/textures/DepthTexture.hpp"

using namespace threepp;

DepthTexture::DepthTexture(unsigned width, unsigned height,std::optional<Type> type, Format format)
    : Texture({Image(std::vector<unsigned char>(width * height * 4, 255), width, height)}) {

    if ( !type && format == Format::Depth ) type = Type::UnsignedShort;
    if ( !type && format == Format::DepthStencil ) type = Type::UnsignedInt248;

    magFilter = Filter::Nearest;
    minFilter = Filter::Nearest;

    generateMipmaps = false;
}

std::shared_ptr<DepthTexture> DepthTexture::create(unsigned width, unsigned height, std::optional<Type> type, Format format) {
    return std::shared_ptr<DepthTexture>(new DepthTexture(width, height, type, format));
}
