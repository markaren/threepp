// The watermill's texture workshop: a set of float maps (linear albedo, height in metres, roughness, cavity, alpha)
// painted per pixel by a recipe, then finished into the three textures a MeshStandardMaterial takes: the albedo (sRGB,
// alpha in A), a tangent-space normal map from the height, and an ORM map (R occlusion, G roughness, B metalness)
// used as the aoMap, roughnessMap and metalnessMap at once. Everything is generated here; nothing is loaded.

#ifndef THREEPP_EXAMPLE_WATERMILL_TEXTURELAB_HPP
#define THREEPP_EXAMPLE_WATERMILL_TEXTURELAB_HPP

#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/utils/ThreadPool.hpp"

#include "Noise.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace mill {

    struct V3 {
        float r = 0, g = 0, b = 0;
        V3() = default;
        V3(float r, float g, float b): r(r), g(g), b(b) {}
        V3 operator+(const V3& o) const { return {r + o.r, g + o.g, b + o.b}; }
        V3 operator-(const V3& o) const { return {r - o.r, g - o.g, b - o.b}; }
        V3 operator*(float s) const { return {r * s, g * s, b * s}; }
        V3 operator*(const V3& o) const { return {r * o.r, g * o.g, b * o.b}; }
    };

    inline V3 mix(const V3& a, const V3& b, float t) { return a + (b - a) * t; }

    // an sRGB colour as linear floats
    inline V3 srgb(int r, int g, int b) {
        const auto lin = [](int c) {
            const float s = static_cast<float>(c) / 255.f;
            return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
        };
        return {lin(r), lin(g), lin(b)};
    }

    inline unsigned char toSrgbByte(float v) {
        v = clamp01(v);
        const float s = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
        return static_cast<unsigned char>(s * 255.f + 0.5f);
    }
    inline unsigned char toByte(float v) { return static_cast<unsigned char>(clamp01(v) * 255.f + 0.5f); }

    template<class Fn>
    void forRows(int rows, Fn&& fn) {
        threepp::ThreadPool::shared().parallelFor(0, rows, std::forward<Fn>(fn));
    }

    struct TexSet {
        std::shared_ptr<threepp::Texture> albedo, normal, orm;

        // a standard material wearing the set
        [[nodiscard]] std::shared_ptr<threepp::MeshStandardMaterial> material(float normalScale = 1.f) const {
            auto m = threepp::MeshStandardMaterial::create();
            m->map = albedo;
            m->normalMap = normal;
            m->normalScale.set(normalScale, normalScale);
            m->roughnessMap = orm;
            m->metalnessMap = orm;
            m->aoMap = orm;
            m->aoMapIntensity = 1.f;
            m->roughness = 1.f;
            m->metalness = 1.f;// scaled by the map's B, which is 0 for every non-metal
            return m;
        }
    };

    inline std::shared_ptr<threepp::DataTexture> makeTexture(std::vector<unsigned char> px, int w, int h, bool srgbData, bool repeat = true) {
        using namespace threepp;
        auto tex = DataTexture::create(ImageData(std::move(px)), w, h);
        tex->format = Format::RGBA;
        tex->type = Type::UnsignedByte;
        tex->wrapS = tex->wrapT = repeat ? TextureWrapping::Repeat : TextureWrapping::ClampToEdge;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::LinearMipmapLinear;
        tex->generateMipmaps = true;
        tex->anisotropy = 8;
        if (srgbData) tex->colorSpace = ColorSpace::sRGB;
        tex->needsUpdate();
        return tex;
    }

    struct Maps {
        int w, h;
        float texelX, texelY;// metres a pixel along x and y
        std::vector<V3> albedo;
        std::vector<float> height, rough, cavity, metal, alpha;

        Maps(int w, int h, float widthMetres, float heightMetres)
            : w(w), h(h), texelX(widthMetres / static_cast<float>(w)), texelY(heightMetres / static_cast<float>(h)),
              albedo(static_cast<size_t>(w) * h, V3(0.5f, 0.5f, 0.5f)), height(static_cast<size_t>(w) * h, 0.f),
              rough(static_cast<size_t>(w) * h, 0.9f), cavity(static_cast<size_t>(w) * h, 1.f),
              metal(static_cast<size_t>(w) * h, 0.f), alpha(static_cast<size_t>(w) * h, 1.f) {}

        [[nodiscard]] size_t at(int x, int y) const { return static_cast<size_t>(y) * w + x; }

        // the height blurred over about `radius` pixels (three box passes, wrapping)
        [[nodiscard]] std::vector<float> blurredHeight(int radius, bool wrap = true) const {
            std::vector<float> a = height, b(a.size());
            for (int pass = 0; pass < 3; ++pass) {
                forRows(h, [&](int y) {
                    float sum = 0;
                    const auto sample = [&](int x) { return a[at(wrap ? wrapi(x, w) : std::clamp(x, 0, w - 1), y)]; };
                    for (int k = -radius; k <= radius; ++k) sum += sample(k);
                    for (int x = 0; x < w; ++x) {
                        b[at(x, y)] = sum / static_cast<float>(2 * radius + 1);
                        sum += sample(x + radius + 1) - sample(x - radius);
                    }
                });
                forRows(w, [&](int x) {
                    float sum = 0;
                    const auto sample = [&](int y) { return b[at(x, wrap ? wrapi(y, h) : std::clamp(y, 0, h - 1))]; };
                    for (int k = -radius; k <= radius; ++k) sum += sample(k);
                    for (int y = 0; y < h; ++y) {
                        a[at(x, y)] = sum / static_cast<float>(2 * radius + 1);
                        sum += sample(y + radius + 1) - sample(y - radius);
                    }
                });
            }
            return a;
        }

        // darken the albedo and the occlusion where the surface sits below its surroundings: `depth` metres below the
        // local mean (over `radius` px) is fully occluded down to `floor`
        void bakeCavity(int radius, float depth, float floorValue, bool wrap = true) {
            const auto mean = blurredHeight(radius, wrap);
            forRows(h, [&](int y) {
                for (int x = 0; x < w; ++x) {
                    const size_t c = at(x, y);
                    const float below = std::max(0.f, mean[c] - height[c]);
                    cavity[c] *= 1.f - (1.f - floorValue) * smooth(0.f, depth, below);
                }
            });
        }

        [[nodiscard]] TexSet finish(float normalStrength = 1.f, bool wrap = true, bool albedoCavity = true) const {
            std::vector<unsigned char> a(static_cast<size_t>(w) * h * 4), n(a.size()), o(a.size());
            forRows(h, [&](int y) {
                for (int x = 0; x < w; ++x) {
                    const size_t c = at(x, y);
                    const auto hAt = [&](int xx, int yy) {
                        if (wrap) return height[at(wrapi(xx, w), wrapi(yy, h))];
                        return height[at(std::clamp(xx, 0, w - 1), std::clamp(yy, 0, h - 1))];
                    };
                    const float sx = (hAt(x + 1, y) - hAt(x - 1, y)) / (2.f * texelX) * normalStrength;
                    const float sy = (hAt(x, y + 1) - hAt(x, y - 1)) / (2.f * texelY) * normalStrength;
                    const float len = std::sqrt(sx * sx + sy * sy + 1.f);
                    n[4 * c] = toByte(0.5f - 0.5f * sx / len);
                    n[4 * c + 1] = toByte(0.5f - 0.5f * sy / len);
                    n[4 * c + 2] = toByte(0.5f + 0.5f / len);
                    n[4 * c + 3] = 255;
                    const float cav = cavity[c];
                    const V3 col = albedoCavity ? albedo[c] * (0.35f + 0.65f * cav) : albedo[c];
                    a[4 * c] = toSrgbByte(col.r);
                    a[4 * c + 1] = toSrgbByte(col.g);
                    a[4 * c + 2] = toSrgbByte(col.b);
                    a[4 * c + 3] = toByte(alpha[c]);
                    o[4 * c] = toByte(0.4f + 0.6f * cav);
                    o[4 * c + 1] = toByte(rough[c]);
                    o[4 * c + 2] = toByte(metal[c]);
                    o[4 * c + 3] = 255;
                }
            });
            TexSet set;
            set.albedo = makeTexture(std::move(a), w, h, true, wrap);
            set.normal = makeTexture(std::move(n), w, h, false, wrap);
            set.orm = makeTexture(std::move(o), w, h, false, wrap);
            return set;
        }
    };

}// namespace mill

#endif
