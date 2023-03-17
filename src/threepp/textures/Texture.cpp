
#include "threepp/textures/Texture.hpp"

#include "threepp/math/MathUtils.hpp"

#include <cmath>

using namespace threepp;


Texture::Texture(std::optional<Image> image)
    : uuid(math::generateUUID()),
      image(std::move(image)) {}

std::shared_ptr<Texture> Texture::create(std::optional<Image> image) {

    return std::shared_ptr<Texture>(new Texture(std::move(image)));
}

void Texture::updateMatrix() {

    this->matrix.setUvTransform(this->offset.x, this->offset.y, this->repeat.x, this->repeat.y, this->rotation, this->center.x, this->center.y);
}

void Texture::dispose() {

    if (!disposed_) {
        disposed_ = true;
        this->dispatchEvent("dispose", this);
    }
}

void Texture::transformUv(Vector2& uv) const {

    if (this->mapping != UVMapping) return;

    uv.applyMatrix3(this->matrix);

    if (uv.x < 0 || uv.x > 1) {

        switch (this->wrapS) {

            case RepeatWrapping:

                uv.x = uv.x - std::floor(uv.x);
                break;

            case ClampToEdgeWrapping:

                uv.x = uv.x < 0 ? 0.f : 1.f;
                break;

            case MirroredRepeatWrapping:

                if (std::abs((int) std::floor(uv.x) % 2) == 1) {

                    uv.x = std::ceil(uv.x) - uv.x;

                } else {

                    uv.x = uv.x - std::floor(uv.x);
                }

                break;
        }
    }

    if (uv.y < 0 || uv.y > 1) {

        switch (this->wrapT) {

            case RepeatWrapping:

                uv.y = uv.y - std::floor(uv.y);
                break;

            case ClampToEdgeWrapping:

                uv.y = uv.y < 0 ? 0.f : 1.f;
                break;

            case MirroredRepeatWrapping:

                if (std::abs((int) std::floor(uv.y) % 2) == 1) {

                    uv.y = std::ceil(uv.y) - uv.y;

                } else {

                    uv.y = uv.y - std::floor(uv.y);
                }

                break;
        }
    }

    if (this->image && (*this->image).flipped()) {

        uv.y = 1 - uv.y;
    }
}

void Texture::needsUpdate() {

    this->version_++;
}

unsigned int Texture::version() const {

    return version_;
}

Texture& Texture::copy(const Texture& source) {

    this->image = source.image;
    this->mipmaps = source.mipmaps;

    this->mapping = source.mapping;

    this->wrapS = source.wrapS;
    this->wrapT = source.wrapT;

    this->magFilter = source.magFilter;
    this->minFilter = source.minFilter;

    this->anisotropy = source.anisotropy;

    this->format = source.format;
    this->internalFormat = source.internalFormat;
    this->type = source.type;

    this->offset.copy(source.offset);
    this->repeat.copy(source.repeat);
    this->center.copy(source.center);
    this->rotation = source.rotation;

    this->matrixAutoUpdate = source.matrixAutoUpdate;
    this->matrix.copy(source.matrix);

    this->generateMipmaps = source.generateMipmaps;
    this->premultiplyAlpha = source.premultiplyAlpha;
    this->unpackAlignment = source.unpackAlignment;
    this->encoding = source.encoding;

    return *this;
}

std::shared_ptr<Texture> Texture::clone() const {
    auto tex = Texture::create();
    tex->copy(*this);

    return tex;
}

Texture::~Texture() {
    dispose();
}
