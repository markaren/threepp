// A pool of soft, camera-facing sprites that the caller drives: powder kicked up by a foot,
// dust behind a wheel, drifting snow, smoke. Write each sprite's centre, size, opacity and
// colour as often as you like, then commit() once per frame.
//
// It is the same billboard as ParticleSystem (whose emitter makes its own particles), so both
// renderers draw it: GLRenderer with the material below, VulkanRenderer with its particle
// pass. With Normal blending the sprites are lit by the scene's lights on both (colour is the
// albedo, and the anisotropy below says how the light depends on the view). With Additive
// blending they are emissive.

#ifndef THREEPP_PARTICLESPRITES_HPP
#define THREEPP_PARTICLESPRITES_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Vector3.hpp"

#include <memory>

namespace threepp {

    class Mesh;
    class Texture;
    class BufferGeometry;

    class ParticleSprites: public Object3D {

    public:
        // `texture` is the sprite's image (white = the sprite's colour; its alpha shapes it);
        // nullptr uses softDisc().
        static std::shared_ptr<ParticleSprites> create(size_t capacity, std::shared_ptr<Texture> texture = nullptr);

        [[nodiscard]] size_t capacity() const;

        // Sprite i: its centre (in this object's space), its diameter in world units, its
        // opacity (0 hides it) and colour, and a spin of the image in radians.
        void set(size_t i, const Vector3& centre, float size, float opacity,
                 const Color& color = Color(1, 1, 1), float angle = 0.f);

        void hide(size_t i);

        void hideAll();

        // How the sprites scatter light, as a Henyey-Greenstein g in [-0.9, 0.9]. Positive
        // (the default, 0.35) brightens them looking toward the sun, as thin haze and single
        // grains do. A dense billow is lit like a body instead, brightest with the sun behind
        // the viewer: give it a small negative value. 0 is the same from every side.
        void setAnisotropy(float g);

        [[nodiscard]] float anisotropy() const;

        // Sends this frame's writes to the GPU.
        void commit();

        // The billboard mesh; its material's blending picks lit (Normal) or emissive (Additive).
        [[nodiscard]] Mesh& mesh() const;

        // A round puff, opaque at the centre and fading to nothing at the edge.
        static std::shared_ptr<Texture> softDisc(int size = 64);

        ~ParticleSprites() override;

    private:
        explicit ParticleSprites(size_t capacity, std::shared_ptr<Texture> texture);

        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_PARTICLESPRITES_HPP
