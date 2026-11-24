// The watermill's sky: a summer afternoon's sky baked on the CPU into an equirectangular float image, used as the
// scene's background and (prefiltered by the renderer) its environment light. Blue overhead, paler and warmer down
// at the horizon, a glow round the sun, and a field of fair-weather cumulus: lit white on their sun side, grey
// beneath. The sun's disc is left out: the DirectionalLight is the sun, and the environment must not light the
// scene with it twice.

#ifndef THREEPP_EXAMPLE_WATERMILL_SKYBAKE_HPP
#define THREEPP_EXAMPLE_WATERMILL_SKYBAKE_HPP

#include "threepp/math/Vector3.hpp"
#include "threepp/textures/DataTexture.hpp"

#include "TextureLab.hpp"

namespace mill {

    struct SkyLook {
        threepp::Vector3 sun;// unit, toward the sun
        V3 zenith = srgb(78, 128, 196) * 1.1f;
        V3 horizon = srgb(196, 214, 228) * 1.15f;
        V3 glow = srgb(255, 236, 200);
        float cloudCover = 0.48f;// 0 clear .. 1 overcast
        float brightness = 1.f;
    };

    // the sky's radiance toward a direction (unit, y up), clouds included
    inline V3 skyRadiance(const SkyLook& s, float dx, float dy, float dz) {
        const float up = std::max(dy, 0.f);
        const float h = std::pow(1.f - up, 3.5f);
        V3 col = mix(s.zenith, s.horizon, h);
        const float cosSun = dx * s.sun.x + dy * s.sun.y + dz * s.sun.z;
        col = col + s.glow * (0.35f * std::pow(std::max(cosSun, 0.f), 8.f) + 0.6f * std::pow(std::max(cosSun, 0.f), 64.f));
        // below the horizon: the far woods' dim green-grey (the scene's ground covers most of it)
        if (dy < 0) {
            const V3 below = srgb(70, 82, 66) * 0.6f;
            col = mix(s.horizon * 0.8f, below, smooth(0.f, 0.08f, -dy));
            return col * s.brightness;
        }
        // clouds on a layer 1.5 km up, over a 30 km field
        if (dy > 0.015f) {
            const float t = 1.5f / dy;// km to the layer along the ray
            const float px = dx * t, pz = dz * t;
            const float n = fbm2(px * 0.55f, pz * 0.55f, 6, 777u, 0, 0, 0.55f);
            const float billow = fbm2(px * 2.2f + 3.f, pz * 2.2f, 4, 778u);
            float density = smooth(1.f - s.cloudCover - 0.05f, 1.f - s.cloudCover + 0.3f, 0.5f + 0.5f * n + 0.18f * billow);
            density *= smooth(0.015f, 0.12f, dy);// thin out toward the horizon, where they are far
            // light: brighter where the cloud's edge faces the sun, greyer where it is thick
            const float sx = s.sun.x, sz = s.sun.z;
            const float slen = std::max(std::hypot(sx, sz), 1e-3f);
            const float ahead = fbm2((px + 0.25f * sx / slen) * 0.55f, (pz + 0.25f * sz / slen) * 0.55f, 5, 777u, 0, 0, 0.55f);
            const float lit = clamp01(0.55f + 2.2f * (n - ahead));
            const V3 white = srgb(250, 248, 242) * 1.35f;
            const V3 base = srgb(150, 158, 172) * 0.95f;
            V3 cloud = mix(base, white, lit);
            cloud = cloud + s.glow * (0.8f * std::pow(std::max(cosSun, 0.f), 16.f) * (1.f - density));// silver lining
            col = mix(col, cloud, density);
        }
        return col * s.brightness;
    }

    inline std::shared_ptr<threepp::DataTexture> bakeSky(const SkyLook& s, int w = 2048, int h = 1024) {
        using namespace threepp;
        std::vector<float> px(static_cast<size_t>(w) * h * 4);
        forRows(h, [&](int y) {
            // row 0 is the bottom (v = 0): the nadir
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(h);
            const float theta = (1.f - v) * 3.14159265f;// from the zenith
            for (int x = 0; x < w; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(w);
                const float phi = u * 6.2831853f;
                // three.js's equirect convention: u = atan(dir.z, dir.x) / 2pi + 0.5
                const float dx = std::sin(theta) * std::cos(phi - 3.14159265f);
                const float dz = std::sin(theta) * std::sin(phi - 3.14159265f);
                const float dy = std::cos(theta);
                const V3 c = skyRadiance(s, dx, dy, dz);
                const size_t i = (static_cast<size_t>(y) * w + x) * 4;
                px[i] = c.r;
                px[i + 1] = c.g;
                px[i + 2] = c.b;
                px[i + 3] = 1.f;
            }
        });
        auto tex = DataTexture::create(ImageData(std::move(px)), w, h);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->wrapS = TextureWrapping::Repeat;
        tex->wrapT = TextureWrapping::ClampToEdge;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Linear;
        tex->generateMipmaps = false;
        tex->needsUpdate();
        return tex;
    }

    // The light the valley floor sees: the same sky, but only above the trees. The woods stand round the stream
    // to about 20 degrees up the valley and over 40 across it; below that line the environment is canopy (sunlit
    // leaves and dark gaps, brighter on the side the sun is on), and below the horizon the dark banks and water.
    // Used as the scene's environment (ambient light and reflections); the background stays the open sky.
    inline std::shared_ptr<threepp::DataTexture> bakeForestEnvironment(const SkyLook& s, int w = 1024, int h = 512) {
        using namespace threepp;
        std::vector<float> px(static_cast<size_t>(w) * h * 4);
        const V3 gap = srgb(22, 30, 16), leaf = srgb(96, 132, 50), ground = srgb(40, 38, 28);
        forRows(h, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(h);
            const float theta = (1.f - v) * 3.14159265f;
            for (int x = 0; x < w; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(w);
                const float phi = u * 6.2831853f;
                const float dx = std::sin(theta) * std::cos(phi - 3.14159265f);
                const float dz = std::sin(theta) * std::sin(phi - 3.14159265f);
                const float dy = std::cos(theta);
                const float az = std::atan2(dz, dx);
                const float elev = std::asin(std::clamp(dy, -1.f, 1.f));
                // the treeline: low up and down the valley (along z), high across it, ragged
                const float across = std::abs(std::cos(az));// 1 looking along x
                float line = 0.16f + 0.55f * across * across + 0.06f * fbm2(az * 3.f, 0.5f, 4, 313u, 0, 0);
                line *= 1.f + 0.15f * fbm2(az * 12.f, 1.5f, 3, 314u);
                V3 c;
                if (elev > line) {
                    c = skyRadiance(s, dx, dy, dz);
                    // the edge of the canopy is broken: sky through the top leaves
                    const float frayed = smooth(line + 0.12f, line, elev) * smooth(0.f, 0.6f, fbm2(az * 40.f, elev * 40.f, 3, 315u));
                    c = mix(c, leaf, 0.7f * frayed);
                } else if (dy > -0.02f) {
                    const float n = 0.5f + 0.5f * fbm2(az * 18.f, elev * 18.f, 4, 316u);
                    const float sunSide = 0.5f + 0.5f * (dx * s.sun.x + dz * s.sun.z) / std::max(1e-3f, std::hypot(dx, dz) * std::hypot(s.sun.x, s.sun.z));
                    // looking toward the sun the leaves glow (light through them); away from it they are lit faces
                    c = mix(gap, leaf * (0.6f + 0.8f * sunSide), smooth(0.35f, 0.75f, n));
                    c = c * (0.7f + 0.3f * smooth(0.f, line, elev));
                } else {
                    c = mix(gap, ground, smooth(-0.02f, -0.3f, dy));
                }
                c = c * s.brightness;
                const size_t i = (static_cast<size_t>(y) * w + x) * 4;
                px[i] = c.r;
                px[i + 1] = c.g;
                px[i + 2] = c.b;
                px[i + 3] = 1.f;
            }
        });
        auto tex = DataTexture::create(ImageData(std::move(px)), w, h);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->wrapS = TextureWrapping::Repeat;
        tex->wrapT = TextureWrapping::ClampToEdge;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Linear;
        tex->generateMipmaps = false;
        tex->needsUpdate();
        return tex;
    }

}// namespace mill

#endif
