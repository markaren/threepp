
#ifndef THREEPP_FRAMEBUFFERTEXTURE_HPP
#define THREEPP_FRAMEBUFFERTEXTURE_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    // Snapshot of the renderer's framebuffer, populated by
    // Renderer::copyFramebufferToTexture. The captured bytes are post-tone-map
    // and post-output-encode (sRGB when the renderer's outputColorSpace is
    // sRGB), so the default colorSpace is sRGB — sampling decodes to linear
    // and the next render stage re-encodes once for a clean round-trip.
    //
    // Diverges from three.js (which leaves FramebufferTexture as NoColorSpace
    // and silently double-encodes); set colorSpace = NoColorSpace explicitly
    // if you need raw bytes (e.g. uploading via a non-color-aware path like
    // ImGui).
    class FramebufferTexture: public Texture {

    public:
        static std::shared_ptr<FramebufferTexture> create(unsigned int width, unsigned int height) {

            return std::shared_ptr<FramebufferTexture>(new FramebufferTexture(width, height));
        }

    private:
        explicit FramebufferTexture(unsigned int width, unsigned int height)
            : Texture({Image(std::vector<unsigned char>(width * height * 4, 0), width, height)}) {

            this->magFilter = Filter::Nearest;
            this->minFilter = Filter::Nearest;

            this->generateMipmaps = false;
            this->unpackAlignment = 1;

            this->colorSpace = ColorSpace::sRGB;

            this->needsUpdate();
        }
    };

}// namespace threepp

#endif//THREEPP_FRAMEBUFFERTEXTURE_HPP
