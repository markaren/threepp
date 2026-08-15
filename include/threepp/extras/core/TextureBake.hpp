// Shared plumbing for the in-engine procedural texture generators
// (architecture, vegetation, terrain dressing, ...): finishing a generated
// DataTexture for sampling, converting a height field into a tangent-space
// normal map, and the Rng-lattice periodic value noise the terrain
// detail/facade bakers tile with.
//
// Extracted (multiply) from CabinTextures / DetailTexture / FacadeTexture;
// draws now come from math::Rng, so texels differ from the mt19937 era but
// are identical across platforms and standard libraries.

#ifndef THREEPP_EXTRAS_CORE_TEXTUREBAKE_HPP
#define THREEPP_EXTRAS_CORE_TEXTUREBAKE_HPP

#include "threepp/extras/core/NoiseUtils.hpp"
#include "threepp/math/Rng.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace threepp::texgen {

    // Finish a generated texture: filtering, wrap mode, colour space.
    // generateMipmaps is NOT optional — DataTexture defaults it to false,
    // which renders BLACK on GL. `anisotropy` > 1 is for content viewed at
    // permanent grazing incidence (road ribbons and the like).
    inline void finishTexture(const std::shared_ptr<DataTexture>& t, bool srgb, bool repeat,
                              int anisotropy = 1) {
        t->magFilter = Filter::Linear;
        t->minFilter = Filter::LinearMipmapLinear;
        t->generateMipmaps = true;
        t->wrapS = repeat ? TextureWrapping::Repeat : TextureWrapping::ClampToEdge;
        t->wrapT = repeat ? TextureWrapping::Repeat : TextureWrapping::ClampToEdge;
        t->colorSpace = srgb ? ColorSpace::sRGB : ColorSpace::NoColorSpace;
        if (anisotropy > 1) t->anisotropy = anisotropy;
        t->needsUpdate();
    }

    // Wrap a pixel buffer as a size x size RGBA texture finished as LINEAR
    // data (0.5-neutral overlays / normal maps must not sRGB-decode) with
    // Repeat wrapping — the terrain detail/facade convention.
    inline std::shared_ptr<DataTexture> makeLinearRepeatTexture(std::vector<unsigned char> px,
                                                                unsigned int size) {
        auto t = DataTexture::create(ImageData{std::move(px)}, size, size);
        t->colorSpace = ColorSpace::Linear;
        t->wrapS = t->wrapT = TextureWrapping::Repeat;
        t->magFilter = Filter::Linear;
        t->minFilter = Filter::LinearMipmapLinear;
        t->generateMipmaps = true;// DataTexture defaults false → GL black
        return t;
    }

    // Tangent-space normal map from a height function, by central difference.
    // `tile` makes the differencing wrap at the tile edge so the normal map
    // is seamless wherever the height field itself is.
    template<class HeightFn>
    void writeNormalFromHeight(const std::shared_ptr<DataTexture>& normal,
                               unsigned int size, float bumpScale,
                               HeightFn&& h, bool tile) {
        auto& cn = normal->image().data<unsigned char>();
        const auto S = static_cast<float>(size);
        const float texel = 1.f / S;
        auto at = [&](float u, float v) {
            if (tile) return h(noise::wrap01(u), noise::wrap01(v));
            return h(std::clamp(u, 0.f, 1.f), std::clamp(v, 0.f, 1.f));
        };
        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / S;
                const float v = static_cast<float>(y) / S;
                float nx = (at(u - texel, v) - at(u + texel, v)) * bumpScale;
                float ny = (at(u, v - texel) - at(u, v + texel)) * bumpScale;
                float nz = 1.f;
                const float inv = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
                const size_t idx = (static_cast<size_t>(y) * size + x) * 4;
                cn[idx + 0] = noise::toByte(nx * inv * 0.5f + 0.5f);
                cn[idx + 1] = noise::toByte(ny * inv * 0.5f + 0.5f);
                cn[idx + 2] = noise::toByte(nz * inv * 0.5f + 0.5f);
                cn[idx + 3] = 255;
            }
        }
    }

    // ── Rng-lattice periodic value noise ─────────────────────────────────
    //
    // A cells x cells lattice of uniform draws, sampled with WRAPPED bilinear
    // interpolation so the field tiles. Unlike the hash-based noise::valueNoise
    // this consumes the caller's RNG stream: the lattice values depend on how
    // many draws preceded them, so call order is part of a generator's output
    // contract — preserve it when refactoring.

    inline std::vector<float> noiseLattice(math::Rng& rng, int cells) {
        std::vector<float> v(static_cast<size_t>(cells) * cells);
        for (auto& x : v) x = rng.nextFloat();
        return v;
    }

    inline float sampleLattice(const std::vector<float>& lat, int cells, float u, float v) {
        const float fu = u * static_cast<float>(cells), fv = v * static_cast<float>(cells);
        const int iu = static_cast<int>(fu) % cells, iv = static_cast<int>(fv) % cells;
        const int ju = (iu + 1) % cells, jv = (iv + 1) % cells;
        const float tu = fu - std::floor(fu), tv = fv - std::floor(fv);
        const float a = lat[static_cast<size_t>(iv) * cells + iu], b = lat[static_cast<size_t>(iv) * cells + ju];
        const float c = lat[static_cast<size_t>(jv) * cells + iu], d = lat[static_cast<size_t>(jv) * cells + ju];
        return (a + (b - a) * tu) * (1.f - tv) + (c + (d - c) * tu) * tv;
    }

}// namespace threepp::texgen

#endif//THREEPP_EXTRAS_CORE_TEXTUREBAKE_HPP
