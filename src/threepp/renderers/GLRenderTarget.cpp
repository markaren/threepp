
#include "threepp/renderers/GLRenderTarget.hpp"

#include "threepp/math/MathUtils.hpp"

using namespace threepp;


std::unique_ptr<GLRenderTarget> GLRenderTarget::create(unsigned int width, unsigned int height, const Options& options) {

    return std::make_unique<GLRenderTarget>(width, height, options);
}

GLRenderTarget::GLRenderTarget(unsigned int width, unsigned int height, const Options& options)
    : uuid(math::generateUUID()),
      width(width), height(height),
      scissor(0.f, 0.f, static_cast<float>(width), static_cast<float>(height)),
      viewport(0.f, 0.f, static_cast<float>(width), static_cast<float>(height)),
      depthBuffer(options.depthBuffer), stencilBuffer(options.stencilBuffer),
      texture(Texture::create({Image({}, width, height)})) {

    if (options.mapping) texture->mapping = *options.mapping;
    if (options.wrapS) texture->wrapS = *options.wrapS;
    if (options.wrapT) texture->wrapT = *options.wrapT;
    if (options.magFilter) texture->magFilter = *options.magFilter;
    if (options.minFilter) texture->minFilter = *options.minFilter;
    if (options.format) texture->format = *options.format;
    if (options.type) texture->type = *options.type;
    if (options.anisotropy) texture->anisotropy = *options.anisotropy;
    if (options.encoding) texture->encoding = *options.encoding;
}

void GLRenderTarget::setSize(unsigned int width, unsigned int height, unsigned int depth) {

    if (this->width != width || this->height != height || this->depth != depth) {

        this->width = width;
        this->height = height;
        this->depth = depth;

        this->texture->image().width = width;
        this->texture->image().height = height;
        this->texture->image().depth = depth;

        this->dispose();
    }

    this->viewport.set(0, 0, static_cast<float>(width), static_cast<float>(height));
    this->scissor.set(0, 0, static_cast<float>(width), static_cast<float>(height));
}

GLRenderTarget& GLRenderTarget::copy(const GLRenderTarget& source) {

    this->width = source.width;
    this->height = source.height;
    this->depth = source.depth;

    this->viewport.copy(source.viewport);

    this->texture = source.texture;
    //                this->texture.image = { ...this->texture.image }; // See #20328.

    this->depthBuffer = source.depthBuffer;
    this->stencilBuffer = source.stencilBuffer;

    return *this;
}

void GLRenderTarget::dispose() {

    if (!disposed) {

        disposed = true;
        this->dispatchEvent("dispose", this);
    }
}

GLRenderTarget::~GLRenderTarget() {

    dispose();
}
