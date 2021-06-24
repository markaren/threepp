
#include "threepp/renderers/gl/GLTextures.hpp"

#include "threepp/math/MathUtils.hpp"

using namespace threepp;

namespace {

    bool isPowerOfTwo(const Image &image) {

        return math::isPowerOfTwo(image.width()) && math::isPowerOfTwo(image.height());
    }

    bool textureNeedsGenerateMipmaps(const Texture &texture, bool supportsMips) {

        return texture.generateMipmaps && supportsMips &&
               texture.minFilter != NearestFilter && texture.minFilter != LinearFilter;
    }

}// namespace

void gl::GLTextures::uploadTexture(const gl::GLTextures::TextureProperties &textureProperties, GLint texture, GLint slot) {
}
