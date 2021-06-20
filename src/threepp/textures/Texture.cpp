
#include "threepp/textures/Texture.hpp"

using namespace threepp;

void Texture::updateMatrix() {

    this->matrix.setUvTransform(this->offset.x, this->offset.y, this->repeat.x, this->repeat.y, this->rotation, this->center.x, this->center.y);
}

void Texture::dispose() {

    this->dispatchEvent("dispose");
}

void Texture::transformUv(Vector2 &uv) const {

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

    if (this->flipY) {

        uv.y = 1 - uv.y;
    }
}

void Texture::needsUpdate() {
    this->version_++;
}

