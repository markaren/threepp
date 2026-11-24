// The footbridge: three timber spans across the stream on two tall stone piers and two abutments. Heavy stringers,
// a deck of cross planks with gaps between them, and on each side a rail of posts, top and bottom rails and close
// vertical pickets. Grey with age; green low on the piers where the stream wets them.

#ifndef THREEPP_EXAMPLE_WATERMILL_BRIDGE_HPP
#define THREEPP_EXAMPLE_WATERMILL_BRIDGE_HPP

#include "threepp/objects/Group.hpp"

#include "Geometry.hpp"
#include "Masonry.hpp"
#include "Materials.hpp"
#include "Site.hpp"

namespace mill {

    class BridgeModel {
    public:
        std::shared_ptr<threepp::Group> root = threepp::Group::create();

        explicit BridgeModel(const Materials& m) {
            using namespace site;
            using threepp::Vector3;
            Rng rng(1212);
            const float zc = bridgeZ;
            const float deckUnder = deckTop - 0.06f;     // the planks' underside
            const float stringerTop = deckUnder;
            const float stringerDepth = 0.5f;
            const float seat = stringerTop - stringerDepth;// where the stringers bear on the piers
            MeshBuild stones, mortar, wood, rails;
            const auto stoneWeather = [](const Vector3& p, const Vector3& n, const V3& c) {
                const float water = site::nominalWater(p.z);
                V3 col = c * (0.68f + 0.32f * smooth(water - 0.1f, water + 1.2f, p.y));
                col = mix(col, srgb(76, 94, 52), 0.65f * smooth(water + 1.4f, water + 0.1f, p.y) * clamp01(0.6f + fbm3(p.x * 2.f, p.y * 2.f, p.z * 2.f, 3, 4u)));
                col = mix(col, srgb(92, 116, 50), 0.55f * smooth(0.5f, 0.9f, n.y));
                return col;
            };
            // the piers: battered, coursed blocks
            for (float px : bridgePiers) {
                Loop L;
                L.x0 = px - pierHalfX;
                L.x1 = px + pierHalfX;
                L.z0 = zc - pierHalfZ;
                L.z1 = zc + pierHalfZ;
                L.base = site::groundHeight(px, zc) - 0.5f;
                L.top = seat - 0.02f;
                L.thickness = 0.45f;
                L.batter = 0.035f;
                L.courseMin = 0.28f;
                L.courseMax = 0.42f;
                L.stoneMin = 0.45f;
                L.stoneMax = 0.95f;
                L.seed = static_cast<uint32_t>(500 + static_cast<int>(px * 10));
                L.tint = V3(0.96f, 0.95f, 0.93f);
                L.weather = stoneWeather;
                buildLoop(stones, mortar, L);
                // a timber bolster on top that the stringers bear on
                appendBeam(wood, Vector3(px, seat - 0.01f - 0.1f, zc - pierHalfZ + 0.25f), Vector3(px, seat - 0.11f, zc + pierHalfZ - 0.25f), Vector3(0, 1, 0), 0.4f, 0.2f,
                           WoodLook{V3(0.62f, 0.58f, 0.54f), rng.next()});
            }
            // the abutments at both ends, set into the banks
            const std::array<std::array<float, 2>, 2> ends{{{bridgeX0 - 1.4f, bridgeX0 + 0.5f}, {bridgeX1 - 0.6f, bridgeX1 + 1.2f}}};
            for (const auto& e : ends) {
                Loop L;
                L.x0 = e[0];
                L.x1 = e[1];
                L.z0 = zc - 1.25f;
                L.z1 = zc + 1.25f;
                const float g = std::min(site::groundHeight(e[0], zc), site::groundHeight(e[1], zc));
                L.base = g - 0.8f;
                L.top = seat - 0.02f;
                L.thickness = 0.45f;
                L.courseMin = 0.26f;
                L.courseMax = 0.4f;
                L.stoneMin = 0.4f;
                L.stoneMax = 0.9f;
                L.seed = static_cast<uint32_t>(700 + static_cast<int>(e[0] * 10));
                L.tint = V3(0.94f, 0.93f, 0.9f);
                L.weather = stoneWeather;
                buildLoop(stones, mortar, L);
            }
            // stringers: three a span, a little sag in each
            std::vector<float> bearings{bridgeX0, bridgePiers[0], bridgePiers[1], bridgeX1};
            for (size_t s = 0; s + 1 < bearings.size(); ++s) {
                const float x0 = bearings[s] - (s == 0 ? 0.6f : 0.f), x1 = bearings[s + 1] + (s + 2 == bearings.size() ? 0.6f : 0.f);
                for (float dz : {-0.62f, 0.f, 0.62f}) {
                    WoodLook look{V3(0.66f, 0.63f, 0.6f) * rng.range(0.9f, 1.05f), rng.next(), 0.02f};
                    look.weather = [](const Vector3& p, const Vector3& n, const V3& c) {
                        (void) p;
                        return n.y < -0.5f ? c * 0.75f : c;
                    };
                    appendBeam(wood, Vector3(x0, seat + 0.5f * stringerDepth, zc + dz), Vector3(x1, seat + 0.5f * stringerDepth, zc + dz), Vector3(0, 1, 0), 0.22f,
                               stringerDepth, look);
                }
            }
            // the deck: cross planks, worn and gapped; a few newer, a few with moss in the shade under the trees
            const float halfDeck = 0.5f * deckWidth;
            for (float x = bridgeX0 - 0.55f; x < bridgeX1 + 0.55f;) {
                const float w = rng.range(0.2f, 0.29f);
                V3 tint = V3(0.92f, 0.9f, 0.86f) * rng.range(0.8f, 1.06f);
                if (rng() < 0.06f) tint = V3(0.9f, 0.78f, 0.62f);
                if (rng() < 0.08f) tint = mix(tint, V3(0.55f, 0.66f, 0.38f), 0.5f);
                WoodLook look{tint, rng.next(), -0.004f, rng.range(-0.02f, 0.02f)};
                appendBeam(wood, Vector3(x + 0.5f * w, deckTop - 0.03f, zc - halfDeck - rng.range(0.f, 0.06f)), Vector3(x + 0.5f * w, deckTop - 0.03f, zc + halfDeck + rng.range(0.f, 0.06f)),
                           Vector3(0, 1, 0), w, 0.06f, look);
                x += w + rng.range(0.012f, 0.025f);
            }
            // the rails: posts on the outside of the outer stringers, rails, close pickets
            for (float side : {-1.f, 1.f}) {
                const float zr = zc + side * (halfDeck + 0.04f);
                const float postTop = deckTop + 1.06f;
                std::vector<float> posts;
                const float x0 = bridgeX0 - 0.45f, x1 = bridgeX1 + 0.45f;
                const int n = static_cast<int>((x1 - x0) / 1.25f);
                for (int k = 0; k <= n; ++k) posts.push_back(x0 + (x1 - x0) * static_cast<float>(k) / static_cast<float>(n));
                for (float px : posts) {
                    appendBeam(rails, Vector3(px, seat + 0.15f, zr + side * 0.06f), Vector3(px, postTop, zr + side * 0.06f), Vector3(0, 0, side), 0.12f, 0.12f,
                               WoodLook{V3(0.8f, 0.77f, 0.73f) * rng.range(0.9f, 1.05f), rng.next()});
                }
                for (size_t k = 0; k + 1 < posts.size(); ++k) {
                    const float a = posts[k], b = posts[k + 1];
                    WoodLook look{V3(0.85f, 0.82f, 0.78f) * rng.range(0.9f, 1.05f), rng.next(), 0.008f};
                    appendBeam(rails, Vector3(a, postTop - 0.04f, zr), Vector3(b, postTop - 0.04f, zr), Vector3(0, 1, 0), 0.07f, 0.09f, look);
                    appendBeam(rails, Vector3(a, deckTop + 0.12f, zr), Vector3(b, deckTop + 0.12f, zr), Vector3(0, 1, 0), 0.06f, 0.08f, look);
                    // pickets
                    for (float x = a + 0.11f; x < b - 0.07f; x += 0.105f) {
                        if (rng() < 0.015f) continue;// one gone here and there
                        appendBeam(rails, Vector3(x, deckTop + 0.06f, zr - side * 0.01f), Vector3(x, postTop - 0.1f, zr - side * 0.01f), Vector3(0, 0, side), 0.05f, 0.022f,
                                   WoodLook{V3(0.88f, 0.86f, 0.82f) * rng.range(0.82f, 1.06f), rng.next(), 0.f, rng.range(-0.05f, 0.05f)});
                    }
                }
            }
            root->add(stones.mesh(m.masonry));
            root->add(mortar.mesh(m.mortar));
            root->add(wood.mesh(m.timber));
            root->add(rails.mesh(m.siding));
        }
    };

}// namespace mill

#endif
