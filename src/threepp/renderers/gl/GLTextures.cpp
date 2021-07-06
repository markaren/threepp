
#include "threepp/renderers/gl/GLTextures.hpp"

#include "threepp/renderers/gl/GLUtils.hpp"

#include "threepp/math/MathUtils.hpp"

using namespace threepp;

namespace {

    std::unordered_map<int, GLuint> wrappingToGL{
            {RepeatWrapping, GL_REPEAT},
            {ClampToEdgeWrapping, GL_CLAMP_TO_EDGE},
            {MirroredRepeatWrapping, GL_MIRRORED_REPEAT}};

    std::unordered_map<int, GLuint> filterToGL{
            {NearestFilter, GL_NEAREST},
            {NearestMipmapNearestFilter, GL_NEAREST_MIPMAP_NEAREST},
            {NearestMipmapLinearFilter, GL_NEAREST_MIPMAP_LINEAR},

            {LinearFilter, GL_LINEAR},
            {LinearMipmapNearestFilter, GL_LINEAR_MIPMAP_NEAREST},
            {LinearMipmapLinearFilter, GL_LINEAR_MIPMAP_LINEAR}};

    bool isPowerOfTwo(const Image &image) {

        return math::isPowerOfTwo(image.width) && math::isPowerOfTwo(image.height);
    }

    bool textureNeedsGenerateMipmaps(const Texture &texture, bool supportsMips) {

        return texture.generateMipmaps && supportsMips &&
               texture.minFilter != NearestFilter && texture.minFilter != LinearFilter;
    }

    GLuint filterFallback(int f) {

        if (f == NearestFilter || f == NearestMipmapNearestFilter || f == NearestMipmapLinearFilter) {

            return GL_NEAREST;
        }

        return GL_LINEAR;
    }

    GLint getInternalFormat(GLint glFormat, GLuint glType) {

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

gl::GLTextures::GLTextures(gl::GLState &state, gl::GLProperties &properties, gl::GLInfo &info)
    : state(state), properties(properties), info(info), onTextureDispose_(*this) {
}

void gl::GLTextures::generateMipmap(GLuint target, const Texture &texture, GLuint width, GLuint height) {

    glGenerateMipmap(target);

    auto &textureProperties = properties.textureProperties.get(texture.uuid);

    textureProperties.maxMipLevel = (int) std::log2(std::max(width, height));
}

void gl::GLTextures::setTextureParameters(GLuint textureType, Texture &texture, bool supportsMips) {

    if (supportsMips) {

        glTexParameteri(textureType, GL_TEXTURE_WRAP_S, wrappingToGL[texture.wrapS]);
        glTexParameteri(textureType, GL_TEXTURE_WRAP_T, wrappingToGL[texture.wrapT]);

        if (textureType == GL_TEXTURE_3D || textureType == GL_TEXTURE_2D_ARRAY) {

            //            glTexParameteri( textureType, GL_TEXTURE_WRAP_R, wrappingToGL[ texture.wrapR ] );
        }

        glTexParameteri(textureType, GL_TEXTURE_MAG_FILTER, filterToGL[texture.magFilter]);
        glTexParameteri(textureType, GL_TEXTURE_MIN_FILTER, filterToGL[texture.minFilter]);

    } else {

        glTexParameteri(textureType, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(textureType, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        if (textureType == GL_TEXTURE_3D || textureType == GL_TEXTURE_2D_ARRAY) {

            glTexParameteri(textureType, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        }

        if (texture.wrapS != ClampToEdgeWrapping || texture.wrapT != ClampToEdgeWrapping) {

            std::cerr << "THREE.WebGLRenderer: Texture is not power of two. Texture.wrapS and Texture.wrapT should be set to THREE.ClampToEdgeWrapping." << std::endl;
        }

        glTexParameteri(textureType, GL_TEXTURE_MAG_FILTER, filterFallback(texture.magFilter));
        glTexParameteri(textureType, GL_TEXTURE_MIN_FILTER, filterFallback(texture.minFilter));

        if (texture.minFilter != NearestFilter && texture.minFilter != LinearFilter) {

            std::cerr << "THREE.WebGLRenderer: Texture is not power of two. Texture.minFilter should be set to THREE.NearestFilter or THREE.LinearFilter." << std::endl;
        }
    }
}

void gl::GLTextures::uploadTexture(TextureProperties &textureProperties, Texture &texture, GLuint slot) {

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

    setTextureParameters(textureType, texture, supportsMips);

        Image *mipmap;
        auto &mipmaps = texture.mipmaps;

        // regular Texture (image, video, canvas)

        // use manually created mipmaps if available
        // if there are no manual mipmaps
        // set 0 level mipmap and then use GL to generate other mipmap levels

        if (mipmaps.size() > 0 && supportsMips) {

            for (size_t i = 0, il = mipmaps.size(); i < il; i++) {

                mipmap = &mipmaps[i];
                state.texImage2D(GL_TEXTURE_2D, (GLint) i, glInternalFormat, mipmap->width, mipmap->height, glFormat, glType, mipmap);
            }

            texture.generateMipmaps = false;
            textureProperties.maxMipLevel = (int) mipmaps.size() - 1;

        } else {

            state.texImage2D(GL_TEXTURE_2D, 0, glInternalFormat, image.width, image.height, glFormat, glType, texture.image->getData().data());
            textureProperties.maxMipLevel = 0;
        }


        if (textureNeedsGenerateMipmaps(texture, supportsMips)) {

            generateMipmap(textureType, texture, image.width, image.height);
        }

        textureProperties.version = texture.version();

        if (texture.onUpdate) texture.onUpdate.value()(texture);
}

void gl::GLTextures::initTexture(TextureProperties &textureProperties, Texture &texture) {

    if (!textureProperties.glInit) {

        textureProperties.glInit = true;

        texture.addEventListener("dispose", &onTextureDispose_);

        glGenTextures(1, &textureProperties.glTexture);

        info.memory.textures++;
    }
}

void gl::GLTextures::deallocateTexture(Texture &texture) {

    auto &textureProperties = properties.textureProperties.get(texture.uuid);

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

void gl::GLTextures::setTexture2D(Texture &texture, GLuint slot) {

    auto &textureProperties = properties.textureProperties.get(texture.uuid);

    if (texture.version() > 0 && textureProperties.version != texture.version()) {

        const auto &image = texture.image;

        if (image) {

            std::cerr << "THREE.WebGLRenderer: Texture marked for update but image is undefined" << std::endl;

        } else {

            uploadTexture(textureProperties, texture, slot);
            return;
        }
    }

    state.activeTexture(GL_TEXTURE0 + slot);
    state.bindTexture(GL_TEXTURE_2D, textureProperties.glTexture);
}

void gl::GLTextures::setTexture2DArray(Texture &texture, GLuint slot) {

    auto &textureProperties = properties.textureProperties.get(texture.uuid);

    if (texture.version() > 0 && textureProperties.version != texture.version()) {

        uploadTexture(textureProperties, texture, slot);
        return;
    }

    state.activeTexture(GL_TEXTURE0 + slot);
    state.bindTexture(GL_TEXTURE_2D_ARRAY, textureProperties.glTexture);
}

void gl::GLTextures::setTexture3D(Texture &texture, GLuint slot) {

    auto textureProperties = properties.textureProperties.get(texture.uuid);

    if (texture.version() > 0 && textureProperties.version != texture.version()) {

        uploadTexture(textureProperties, texture, slot);
        return;
    }

    state.activeTexture(GL_TEXTURE0 + slot);
    state.bindTexture(GL_TEXTURE_3D, textureProperties.glTexture);
}

void gl::GLTextures::setTextureCube(Texture &texture, GLuint slot) {

    auto &textureProperties = properties.textureProperties.get(texture.uuid);

    if (texture.version() > 0 && textureProperties.version != texture.version()) {

        uploadCubeTexture(textureProperties, texture, slot);
        return;
    }

    state.activeTexture(GL_TEXTURE0 + slot);
    state.bindTexture(GL_TEXTURE_CUBE_MAP, textureProperties.glTexture);
}

void gl::GLTextures::uploadCubeTexture(TextureProperties &textureProperties, Texture &texture, GLuint slot) {
    // TODO
}

void gl::GLTextures::setupFrameBufferTexture(GLuint framebuffer, GLRenderTarget &renderTarget, Texture &texture, GLuint attachment, GLuint textureTarget) {

    //TODO
}

void gl::GLTextures::TextureEventListener::onEvent(Event &event) {

    auto texture = static_cast<Texture *>(event.target);

    texture->removeEventListener("dispose", this);

    scope_.deallocateTexture(*texture);

    scope_.info.memory.textures--;
}
