
#include "threepp/renderers/gl/GLTextures.hpp"

#include "threepp/renderers/gl/GLUtils.hpp"

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

    //    void generateMipmap(int target, int texture, int width, int height) {
    //
    //        glGenerateMipmap(target);
    //
    //        const textureProperties = properties.get(texture);
    //
    //        textureProperties.__maxMipLevel = std::log2(std::max(width, height));
    //    }

    GLuint getInternalFormat(GLuint glFormat, GLuint glType) {

        auto internalFormat = glFormat;

        if (glFormat == GL_RED) {

            if (glType == GL_FLOAT) internalFormat = GL_R32F;
            if (glType == GL_HALF_FLOAT) internalFormat = GL_R16F;
            if (glType == GL_UNSIGNED_BYTE) internalFormat = GL_R8;
        }

        if (glFormat == GL_RGB) {

            if (glType == GL_FLOAT) internalFormat = GL_RGB32F;
            if (glType == GL_HALF_FLOAT) internalFormat = GL_RGB16F;
            if (glType == GL_UNSIGNED_BYTE) internalFormat = GL_RGB8;
        }

        if (glFormat == GL_RGBA) {

            if (glType == GL_FLOAT) internalFormat = GL_RGBA32F;
            if (glType == GL_HALF_FLOAT) internalFormat = GL_RGBA16F;
            if (glType == GL_UNSIGNED_BYTE) internalFormat = GL_RGBA8;
        }

        return internalFormat;
    }

}// namespace


void gl::GLTextures::uploadTexture(GLTextureProperties::Properties &textureProperties, Texture &texture, GLint slot) {

    if (!texture.image) return;

    GLint textureType = GL_TEXTURE_2D;

    initTexture(textureProperties, texture);

    state.activeTexture(GL_TEXTURE0 + slot);
    state.bindTexture(textureType, textureProperties.glTexture);

    glPixelStorei(GL_UNPACK_ALIGNMENT, texture.unpackAlignment);

    Image image = *texture.image;

    const auto supportsMips = true;

    GLuint glFormat = convert(texture.format);

    GLuint glType = convert(texture.type);
    auto glInternalFormat = getInternalFormat(glFormat, glType);
    //
    //    setTextureParameters(textureType, texture, supportsMips);
    //
    //    Image *mipmap;
    //    auto &mipmaps = texture.mipmaps;
    //
    //    // regular Texture (image, video, canvas)
    //
    //    // use manually created mipmaps if available
    //    // if there are no manual mipmaps
    //    // set 0 level mipmap and then use GL to generate other mipmap levels
    //
    //    if (mipmaps.size() > 0 && supportsMips) {
    //
    //        for (int i = 0, il = mipmaps.size(); i < il; i++) {
    //
    //            mipmap = &mipmaps[i];
    //            state.texImage2D(GL_TEXTURE_2D, i, glInternalFormat, glFormat, glType, mipmap);
    //        }
    //
    //        texture.generateMipmaps = false;
    //        textureProperties.maxMipLevel = mipmaps.size() - 1;
    //
    //    } else {
    //
    //        state.texImage2D(GL_TEXTURE_2D, 0, glInternalFormat, glFormat, glType, texture.image->getData());
    //        textureProperties.maxMipLevel = 0;
    //    }
    //
    //
    //    if (textureNeedsGenerateMipmaps(texture, supportsMips)) {
    //
    //        generateMipmap(textureType, texture, image.width(), image.height());
    //    }
    //
    //    textureProperties.version = texture.version();
    //
    //    if (texture.onUpdate) texture.onUpdate(texture);
}

void gl::GLTextures::initTexture(gl::GLTextureProperties::Properties &textureProperties, Texture &texture) {

    if (!textureProperties.glInit) {

        textureProperties.glInit = true;

        texture.addEventListener("dispose", &onTextureDispose);

        glGenTextures(1, &textureProperties.glTexture);

        info.memory.textures++;
    }
}
gl::GLTextures::GLTextures(gl::GLState &state, gl::GLProperties &properties, gl::GLInfo &info)
    : state(state), properties(properties), info(info) {

    //    onTextureDispose = [&](Event &event) {
    //      auto texture = static_cast<Texture*>(event.target);
    //
    //      texture->removeEventListener( "dispose", onTextureDispose );
    //
    //      deallocateTexture( texture );
    //
    //      info.memory.textures --;
    //    };
}


void gl::GLTextures::deallocateTexture(Texture &texture) {

    auto textureProperties = properties.textureProperties.get(texture.uuid);

    if (!textureProperties.glInit) return;

    glDeleteTextures(1, &textureProperties.glTexture);

    properties.textureProperties.remove(texture.uuid);
}

void gl::GLTextures::resetTextureUnits() {

    textureUnits = 0;
}

int gl::GLTextures::allocateTextureUnit() {
    int textureUnit = textureUnits;

    if (textureUnit >= maxTextures) {

        std::cerr << "THREE.WebGLTextures: Trying to use " << textureUnit << " texture units while this GPU supports only " << maxTextures << std::endl;
    }

    textureUnits += 1;

    return textureUnit;
}

