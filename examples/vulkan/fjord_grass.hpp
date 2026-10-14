// FJORD GRASS — the meadow as TUFTS, not blades.
//
// The meadow used to be 300 000 single blades (8 triangles each, 2.4 M swaying
// triangles) at about three blades per square metre: sparse cartoon spikes on
// bare ground the moment the camera came near, and over a millisecond of the
// frame. A real meadow is thousands of blades per square metre, which only
// cards can afford: each tuft here is two crossed cards showing a painted
// clump of fine blades (mixed greens, a few dry straws), ~30 per card, so the
// same screen reads ten times as dense at a tenth of the triangles.
//
// The atlas is painted at startup (no asset): four tuft variants side by side.
// The cards carry the GrassMesh "heightFrac" attribute, so the wind compute
// sways their tops exactly as it swayed the blades.

#ifndef THREEPP_FJORD_GRASS_HPP
#define THREEPP_FJORD_GRASS_HPP

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace fjord {

    constexpr int kTuftVariants = 4;

    // RGBA8 sRGB atlas, kTuftVariants cells side by side; row 0 is the ground
    // (v = 0 at the base of a card).
    inline std::shared_ptr<threepp::DataTexture> makeTuftAtlas(int cellW = 256, int cellH = 192,
                                                               unsigned int seed = 9u) {
        using namespace threepp;
        const int W = cellW * kTuftVariants, H = cellH;
        std::vector<float> rgb(static_cast<size_t>(W) * H * 3, 0.f), a(static_cast<size_t>(W) * H, 0.f);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        std::normal_distribution<float> gauss(0.f, 1.f);
        const auto toSrgb = [](float c) {
            c = std::clamp(c, 0.f, 1.f);
            return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
        };

        for (int cell = 0; cell < kTuftVariants; ++cell) {
            const int blades = 26 + cell * 4;
            for (int b = 0; b < blades; ++b) {
                // Base clustered toward the middle of the card, heights varied,
                // tips leaning out and bending over.
                const float bx = std::clamp(0.5f + 0.17f * gauss(rng), 0.06f, 0.94f);
                const float h = 0.40f + 0.58f * std::pow(u01(rng), 0.7f);
                const float lean = (bx - 0.5f) * 0.6f + 0.18f * gauss(rng);
                const float w0 = (2.0f + 2.2f * u01(rng)) / static_cast<float>(cellW);
                const bool dry = u01(rng) < 0.12f;
                // Linear colours: dark base, the blade's own green toward the tip.
                const float g = u01(rng);
                const float tip[3] = {dry ? 0.30f : 0.07f + 0.10f * g, dry ? 0.26f : 0.16f + 0.10f * g,
                                      dry ? 0.10f : 0.035f + 0.02f * g};
                const float base[3] = {0.020f, 0.040f, 0.012f};
                const int rows = static_cast<int>(h * static_cast<float>(cellH));
                for (int r = 0; r < rows; ++r) {
                    const float t = static_cast<float>(r) / static_cast<float>(std::max(1, rows - 1));
                    const float cx = (bx + lean * t * t) * static_cast<float>(cellW);
                    const float half = 0.5f * w0 * static_cast<float>(cellW) * (1.f - 0.85f * t) + 0.35f;
                    const int y = r;
                    for (int x = static_cast<int>(std::floor(cx - half - 1.f)); x <= static_cast<int>(std::ceil(cx + half + 1.f)); ++x) {
                        if (x < 0 || x >= cellW) continue;
                        const float cov = std::clamp(half + 0.5f - std::abs(static_cast<float>(x) + 0.5f - cx), 0.f, 1.f);
                        if (cov <= 0.f) continue;
                        const size_t i = static_cast<size_t>(y) * W + static_cast<size_t>(cell * cellW + x);
                        const float shade = 0.75f + 0.25f * t;
                        for (int k = 0; k < 3; ++k) {
                            const float c = (base[k] + (tip[k] - base[k]) * std::pow(t, 0.8f)) * shade;
                            rgb[i * 3 + k] = rgb[i * 3 + k] * (1.f - cov) + c * cov;// later blades in front
                        }
                        a[i] = a[i] + (1.f - a[i]) * cov;
                    }
                }
            }
        }
        // Push colour into the empty texels (nearest painted neighbour in the
        // row) so mipmaps do not pull black into the blade edges.
        for (int y = 0; y < H; ++y) {
            float last[3] = {0.05f, 0.09f, 0.03f};
            for (int x = 0; x < W; ++x) {
                const size_t i = static_cast<size_t>(y) * W + x;
                if (a[i] > 0.05f) {
                    for (int k = 0; k < 3; ++k) last[k] = rgb[i * 3 + k];
                } else {
                    for (int k = 0; k < 3; ++k) rgb[i * 3 + k] = last[k];
                }
            }
        }
        std::vector<unsigned char> px(static_cast<size_t>(W) * H * 4);
        for (size_t i = 0; i < a.size(); ++i) {
            for (int k = 0; k < 3; ++k) px[i * 4 + k] = static_cast<unsigned char>(toSrgb(rgb[i * 3 + k]) * 255.f + 0.5f);
            px[i * 4 + 3] = static_cast<unsigned char>(std::clamp(a[i], 0.f, 1.f) * 255.f + 0.5f);
        }
        auto t = DataTexture::create(ImageData{std::move(px)}, static_cast<unsigned int>(W), static_cast<unsigned int>(H));
        t->colorSpace = ColorSpace::sRGB;
        t->wrapS = t->wrapT = TextureWrapping::ClampToEdge;
        t->magFilter = Filter::Linear;
        t->minFilter = Filter::LinearMipmapLinear;
        t->generateMipmaps = true;
        t->needsUpdate();
        return t;
    }

    struct Tuft {
        threepp::Vector3 position;// base, on the ground
        float yaw = 0.f;
        float width = 0.6f, height = 0.45f;
        int variant = 0;
        threepp::Vector3 tint{1.f, 1.f, 1.f};
    };

    // Two crossed cards per tuft, merged per world cell into GrassMesh tiles.
    inline std::vector<std::shared_ptr<threepp::GrassMesh>> buildTuftTiles(
            const std::vector<Tuft>& tufts, float tileSize, const std::shared_ptr<threepp::Material>& material,
            const threepp::GrassMesh::Params& params) {
        using namespace threepp;
        std::map<std::pair<int, int>, std::vector<const Tuft*>> cells;
        for (const auto& t : tufts) {
            cells[{static_cast<int>(std::floor(t.position.x / tileSize)),
                   static_cast<int>(std::floor(t.position.z / tileSize))}]
                    .push_back(&t);
        }
        std::vector<std::shared_ptr<GrassMesh>> out;
        for (auto& [key, list] : cells) {
            std::vector<float> pos, nrm, uv, col, hf;
            std::vector<unsigned int> idx;
            for (const Tuft* t : list) {
                const float u0 = static_cast<float>(t->variant) / static_cast<float>(kTuftVariants);
                const float u1 = static_cast<float>(t->variant + 1) / static_cast<float>(kTuftVariants);
                for (int card = 0; card < 2; ++card) {
                    const float ang = t->yaw + static_cast<float>(card) * 1.5707963f;
                    const float ca = std::cos(ang), sa = std::sin(ang);
                    // Up-biased normal, as the blades had: a card lit from its
                    // back face does not go black.
                    Vector3 n(-sa * 0.53f, 0.85f, ca * 0.53f);
                    const auto base = static_cast<unsigned int>(pos.size() / 3);
                    for (int v = 0; v < 2; ++v) {
                        for (int s = 0; s < 2; ++s) {
                            const float side = s == 0 ? -0.5f : 0.5f;
                            pos.insert(pos.end(), {t->position.x + ca * side * t->width,
                                                   t->position.y + static_cast<float>(v) * t->height,
                                                   t->position.z + sa * side * t->width});
                            nrm.insert(nrm.end(), {n.x, n.y, n.z});
                            uv.insert(uv.end(), {s == 0 ? u0 : u1, static_cast<float>(v)});
                            const float ao = v == 0 ? 0.55f : 1.f;// the base sits in the tuft's own shade
                            col.insert(col.end(), {t->tint.x * ao, t->tint.y * ao, t->tint.z * ao});
                            hf.push_back(static_cast<float>(v));
                        }
                    }
                    idx.insert(idx.end(), {base, base + 1, base + 2, base + 1, base + 3, base + 2});
                }
            }
            auto geo = BufferGeometry::create();
            geo->setIndex(idx);
            geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
            geo->setAttribute("heightFrac", FloatBufferAttribute::create(hf, 1));
            auto tile = GrassMesh::create(geo, material);
            tile->params = params;
            tile->name = "tuft_tile_" + std::to_string(key.first) + "_" + std::to_string(key.second);
            out.push_back(tile);
        }
        return out;
    }

}// namespace fjord

#endif//THREEPP_FJORD_GRASS_HPP
