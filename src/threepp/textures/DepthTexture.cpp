
#include "threepp/textures/DepthTexture.hpp"

using namespace threepp;

DepthTexture::DepthTexture(std::optional<Type> type, Format format)
    : Texture({Image({}, 0, 0)}) {

    if ( !type && format == Format::Depth ) type = Type::UnsignedShort;
    if ( !type && format == Format::DepthStencil ) type = Type::UnsignedInt248;

    magFilter = Filter::Nearest;
    minFilter = Filter::Nearest;

    generateMipmaps = false;
}

std::shared_ptr<DepthTexture> DepthTexture::create(std::optional<Type> type, Format format) {
    return std::shared_ptr<DepthTexture>(new DepthTexture(type, format));
}
