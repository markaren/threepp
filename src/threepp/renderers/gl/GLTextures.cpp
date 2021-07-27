
#include "threepp/renderers/gl/GLTextures.hpp"

#include "threepp/renderers/gl/GLUtils.hpp"

#include "threepp/math/MathUtils.hpp"

using namespace threepp;

namespace {

    std::unordered_map<int, int> wrappingToGL{
            {RepeatWrapping, GL_REPEAT},
            {ClampToEdgeWrapping, GL_CLAMP_TO_EDGE},
            {MirroredRepeatWrapping, GL_MIRRORED_REPEAT}};

    std::unordered_map<int, int> filterToGL{
            {NearestFilter, GL_NEAREST},
            {NearestMipmapNearestFilter, GL_NEAREST_MIPMAP_NEAREST},
            {NearestMipmapLinearFilter, GL_NEAREST_MIPMAP_LINEAR},

            {LinearFilter, GL_LINEAR},
            {LinearMipmapNearestFilter, GL_LINEAR_MIPMAP_NEAREST},
            {LinearMipmapLinearFilter, GL_LINEAR_MIPMAP_LINEAR}};

    bool textureNeedsGenerateMipmaps(const Texture &texture) {

        return texture.generateMipmaps &&
               texture.minFilter != NearestFilter && texture.minFilter != LinearFilter;
    }

    GLuint filterFallback(int f) {

        if (f == NearestFilter || f == NearestMipmapNearestFilter || f == NearestMipmapLinearFilter) {

            return GL_NEAREST;
        }

        return GL_LINEAR;
    }

    GLint getInternalFormat(GLuint glFormat, GLuint glType) {

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
    : state(state), properties(properties), info(info), onTextureDispose_(*this), onRenderTargetDispose_(*this) {
}

void gl::GLTextures::generateMipmap(GLuint target, const Texture &texture, GLuint width, GLuint height) {

    glGenerateMipmap(target);

    auto &textureProperties = properties.textureProperties.get(texture.uuid);

    textureProperties.maxMipLevel = static_cast<int>(std::log2(std::max(width, height)));
}

void gl::GLTextures::setTextureParameters(GLuint textureType, Texture &texture) {

    glTexParameteri(textureType, GL_TEXTURE_WRAP_S, wrappingToGL[texture.wrapS]);
    glTexParameteri(textureType, GL_TEXTURE_WRAP_T, wrappingToGL[texture.wrapT]);

    if (textureType == GL_TEXTURE_3D || textureType == GL_TEXTURE_2D_ARRAY) {

        //            glTexParameteri( textureType, GL_TEXTURE_WRAP_R, wrappingToGL[ texture.wrapR ] );
    }

    glTexParameteri(textureType, GL_TEXTURE_MAG_FILTER, filterToGL[texture.magFilter]);
    glTexParameteri(textureType, GL_TEXTURE_MIN_FILTER, filterToGL[texture.minFilter]);
}

void gl::GLTextures::uploadTexture(TextureProperties &textureProperties, Texture &texture, GLuint slot) {

    if (!texture.image) return;

    GLint textureType = GL_TEXTURE_2D;

    initTexture(textureProperties, texture);

    state.activeTexture(GL_TEXTURE0 + slot);
    state.bindTexture(textureType, textureProperties.glTexture);

    glPixelStorei(GL_UNPACK_ALIGNMENT, texture.unpackAlignment);

    const auto &image = *texture.image;

    GLuint glFormat = convert(texture.format);

    GLuint glType = convert(texture.type);
    auto glInternalFormat = getInternalFormat(glFormat, glType);

    setTextureParameters(textureType, texture);

    auto &mipmaps = texture.mipmaps;

    // regular Texture (image, video, canvas)

    // use manually created mipmaps if available
    // if there are no manual mipmaps
    // set 0 level mipmap and then use GL to generate other mipmap levels

    if (!mipmaps.empty()) {

        for (size_t i = 0, il = mipmaps.size(); i < il; i++) {

            const auto &mipmap = mipmaps[i];
            state.texImage2D(GL_TEXTURE_2D, (GLint) i, glInternalFormat, mipmap.width, mipmap.height, glFormat, glType, mipmap.getData());
        }

        texture.generateMipmaps = false;
        textureProperties.maxMipLevel = static_cast<int>(mipmaps.size()) - 1;

    } else {

        state.texImage2D(GL_TEXTURE_2D, 0, glInternalFormat, image.width, image.height, glFormat, glType, texture.image->getData());
        textureProperties.maxMipLevel = 0;
    }

    if (textureNeedsGenerateMipmaps(texture)) {

        generateMipmap(textureType, texture, image.width, image.height);
    }

    textureProperties.version = texture.version();

    if (texture.onUpdate) texture.onUpdate.value()(texture);
}

void gl::GLTextures::initTexture(TextureProperties &textureProperties, Texture &texture) {

    if (!textureProperties.glInit) {

        textureProperties.glInit = true;

        texture.addEventListener("dispose", &onTextureDispose_);

        GLuint glTexture;
        glGenTextures(1, &glTexture);
        textureProperties.glTexture = glTexture;

        info.memory.textures++;
    }
}

void gl::GLTextures::deallocateTexture(Texture *texture) {

    auto &textureProperties = properties.textureProperties.get(texture->uuid);

    if (!textureProperties.glInit) return;

    glDeleteTextures(1, &textureProperties.glTexture.value());

    properties.textureProperties.remove(texture->uuid);
}

void gl::GLTextures::deallocateRenderTarget(GLRenderTarget *renderTarget) {

    if (!renderTarget) return;

    const auto &texture = renderTarget->texture;

    auto &renderTargetProperties = properties.renderTargetProperties.get(renderTarget->uuid);
    const auto &textureProperties = properties.textureProperties.get(texture->uuid);

    if (textureProperties.glTexture) {

        glDeleteTextures(1, &textureProperties.glTexture.value());

        info.memory.textures--;
    }

    if (renderTarget->depthTexture) {

        renderTarget->depthTexture->dispose();
    }

    glDeleteFramebuffers(1, &renderTargetProperties.glFramebuffer.value());
    if (renderTargetProperties.glDepthbuffer) glDeleteRenderbuffers(1, &renderTargetProperties.glDepthbuffer.value());

    properties.textureProperties.remove(texture->uuid);
    properties.renderTargetProperties.remove(renderTarget->uuid);
}

void gl::GLTextures::resetTextureUnits() {

    textureUnits = 0;
}

int gl::GLTextures::allocateTextureUnit() {

    int textureUnit = textureUnits;

    if (textureUnit >= maxTextures) {

        std::cerr << "THREE.GLTextures: Trying to use " << textureUnit << " texture units while this GPU supports only " << maxTextures << std::endl;
    }

    textureUnits += 1;

    return textureUnit;
}

void gl::GLTextures::setTexture2D(Texture &texture, GLuint slot) {

    auto &textureProperties = properties.textureProperties.get(texture.uuid);

    if (texture.version() > 0 && textureProperties.version != texture.version()) {

        const auto &image = texture.image;

        if (!image) {

            std::cerr << "THREE.GLRenderer: Texture marked for update but image is undefined" << std::endl;

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

    auto &textureProperties = properties.textureProperties.get(texture.uuid);

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

void gl::GLTextures::setupFrameBufferTexture(
        GLuint framebuffer,
        const std::shared_ptr<GLRenderTarget> &renderTarget,
        Texture &texture, GLuint attachment, GLuint textureTarget) {

    const auto glFormat = convert(texture.format);
    const auto glType = convert(texture.type);
    const auto glInternalFormat = getInternalFormat(/*texture.internalFormat,*/ glFormat, glType);

    if (textureTarget == GL_TEXTURE_3D || textureTarget == GL_TEXTURE_2D_ARRAY) {

        state.texImage3D(textureTarget, 0, glInternalFormat, renderTarget->width, renderTarget->height, renderTarget->depth, glFormat, glType, nullptr);

    } else {

        state.texImage2D(textureTarget, 0, glInternalFormat, renderTarget->width, renderTarget->height, glFormat, glType, nullptr);
    }

    state.bindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, textureTarget, *properties.textureProperties.get(texture.uuid).glTexture, 0);
    state.bindFramebuffer(GL_FRAMEBUFFER, std::nullopt);
}

void gl::GLTextures::setupRenderBufferStorage(GLuint renderbuffer, const std::shared_ptr<GLRenderTarget> &renderTarget) {

    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);

    if (renderTarget->depthBuffer && !renderTarget->stencilBuffer) {

        auto glInternalFormat = GL_DEPTH_COMPONENT16;

        glRenderbufferStorage(GL_RENDERBUFFER, glInternalFormat, renderTarget->width, renderTarget->height);


        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);

    } else if (renderTarget->depthBuffer && renderTarget->stencilBuffer) {


        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_STENCIL, renderTarget->width, renderTarget->height);


        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);

    } else {

        // Use the first texture for MRT so far
        const auto texture = renderTarget->texture;

        const auto glFormat = convert(texture->format);
        const auto glType = convert(texture->type);
        const auto glInternalFormat = getInternalFormat(glFormat, glType);

        glRenderbufferStorage(GL_RENDERBUFFER, glInternalFormat, renderTarget->width, renderTarget->height);
    }

    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void gl::GLTextures::setupDepthTexture(GLuint framebuffer, const std::shared_ptr<GLRenderTarget> &renderTarget) {

    state.bindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    if (!(renderTarget->depthTexture && renderTarget->depthTexture)) {

        throw std::runtime_error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");
    }

    // upload an empty depth texture with framebuffer size
    if (!properties.textureProperties.get(renderTarget->depthTexture->uuid).glTexture ||
        renderTarget->depthTexture->image->width != renderTarget->width ||
        renderTarget->depthTexture->image->height != renderTarget->height) {

        renderTarget->depthTexture->image->width = renderTarget->width;
        renderTarget->depthTexture->image->height = renderTarget->height;
        renderTarget->depthTexture->needsUpdate();
    }

    setTexture2D(*renderTarget->depthTexture, 0);

    const auto &glDepthTexture = properties.textureProperties.get(renderTarget->depthTexture->uuid).glTexture;

    if (renderTarget->depthTexture->format == DepthFormat) {

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, *glDepthTexture, 0);

    } else if (renderTarget->depthTexture->format == DepthStencilFormat) {

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, *glDepthTexture, 0);

    } else {

        throw std::runtime_error("Unknown depthTexture format");
    }
}

void gl::GLTextures::setupDepthRenderbuffer(const std::shared_ptr<GLRenderTarget> &renderTarget) {

    auto &renderTargetProperties = properties.renderTargetProperties.get(renderTarget->uuid);

    if (renderTarget->depthTexture) {

        setupDepthTexture(*renderTargetProperties.glFramebuffer, renderTarget);

    } else {

        state.bindFramebuffer(GL_FRAMEBUFFER, renderTargetProperties.glFramebuffer);
        GLuint glDepthbuffer;
        glGenRenderbuffers(1, &glDepthbuffer);
        renderTargetProperties.glDepthbuffer = glDepthbuffer;
        setupRenderBufferStorage(*renderTargetProperties.glDepthbuffer, renderTarget);
    }

    state.bindFramebuffer(GL_FRAMEBUFFER, 0);
}

void gl::GLTextures::setupRenderTarget(const std::shared_ptr<GLRenderTarget> &renderTarget) {

    const auto &texture = renderTarget->texture;

    auto &renderTargetProperties = properties.renderTargetProperties.get(renderTarget->uuid);
    auto &textureProperties = properties.textureProperties.get(texture->uuid);

    renderTarget->addEventListener("dispose", &onRenderTargetDispose_);

    GLuint glTexture;
    glGenTextures(1, &glTexture);
    textureProperties.glTexture = glTexture;
    textureProperties.version = texture->version();
    info.memory.textures++;

    // Handles WebGL2 RGBFormat fallback - #18858

    if (texture->format == RGBFormat && (texture->type == FloatType || texture->type == HalfFloatType)) {

        texture->format = RGBAFormat;

        std::cerr << "THREE.GLRenderer: Rendering to textures with RGB format is not supported. Using RGBA format instead." << std::endl;
    }

    // Setup framebuffer

    GLuint glFramebuffer;
    glGenFramebuffers(1, &glFramebuffer);
    renderTargetProperties.glFramebuffer = glFramebuffer;

    // Setup color buffer

    auto glTextureType = GL_TEXTURE_2D;

    state.bindTexture(glTextureType, textureProperties.glTexture);
    setTextureParameters(glTextureType, *texture);
    setupFrameBufferTexture(*renderTargetProperties.glFramebuffer, renderTarget, *texture, GL_COLOR_ATTACHMENT0, glTextureType);

    if (textureNeedsGenerateMipmaps(*texture)) {

        generateMipmap(GL_TEXTURE_2D, *texture, renderTarget->width, renderTarget->height);
    }

    state.bindTexture(GL_TEXTURE_2D, 0);


    // Setup depth and stencil buffers

    if (renderTarget->depthBuffer) {

        setupDepthRenderbuffer(renderTarget);
    }
}

void gl::GLTextures::TextureEventListener::onEvent(Event &event) {

    auto texture = static_cast<Texture *>(event.target);

    texture->removeEventListener("dispose", this);

    scope_.deallocateTexture(texture);

    scope_.info.memory.textures--;
}

void gl::GLTextures::RenderTargetEventListener::onEvent(Event &event) {

    auto renderTarget = static_cast<GLRenderTarget *>(event.target);

    renderTarget->removeEventListener("dispose", this);

    scope_.deallocateRenderTarget(renderTarget);
}
