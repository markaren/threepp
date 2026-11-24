// The flume: an open plank trough that brings the water down the right bank from upstream to the top of the wheel,
// held in yokes every 1.2 m on two stringers, carried on timber bents (two battered posts, a cap, cross braces) that
// stand on flat footing stones. Where it crosses the footbridge's landing the bents step aside so walkers pass under.
// Over the wheel its last beam rests on a post on the wheel's pier and a bracket on the mill's wall.

#ifndef THREEPP_EXAMPLE_WATERMILL_FLUME_HPP
#define THREEPP_EXAMPLE_WATERMILL_FLUME_HPP

#include "threepp/objects/Group.hpp"

#include "Geometry.hpp"
#include "Masonry.hpp"
#include "Materials.hpp"
#include "Site.hpp"

namespace mill {

    class FlumeModel {
    public:
        std::shared_ptr<threepp::Group> root = threepp::Group::create();

        // the trough's inside: floor height and the half width between its walls
        static float floorAt(float z) { return site::flumeFloor(z); }

        explicit FlumeModel(const Materials& m) {
            using namespace site;
            using threepp::Vector3;
            Rng rng(808);
            MeshBuild wet, dry, footings;
            const float x = flumeX, hi = 0.5f * flumeInner;
            const float zA = flumeStartZ, zB = flumeEndZ;
            const auto plankLook = [&](float darkness) {
                WoodLook l;
                l.tint = V3(1, 1, 1) * rng.range(0.85f, 1.05f) * darkness;
                l.seed = rng.next();
                l.weather = [](const Vector3& p, const Vector3& n, const V3& c) {
                    // the trough's wood is dark with wet; algae greens its lower boards and the drips under the seams
                    V3 col = c;
                    col = mix(col, srgb(60, 74, 44), 0.5f * smooth(0.3f, 0.f, p.y - site::flumeFloor(p.z)) * clamp01(0.5f + fbm3(p.x * 2.f, p.y * 2.f, p.z * 2.f, 2, 3u)));
                    (void) n;
                    return col;
                };
                return l;
            };
            // floor and walls: planks in lengths of about 4 m, butt joints staggered
            const auto planks = [&](float u, float yOff, float width, float thick, const Vector3& up, float stagger) {
                float z = zA - stagger;
                while (z < zB) {
                    const float len = rng.range(3.4f, 4.4f);
                    const float z0 = std::max(z, zA), z1 = std::min(z + len, zB);
                    if (z1 - z0 > 0.2f) {
                        const float y0 = floorAt(z0) + yOff, y1 = floorAt(z1) + yOff;
                        appendBeam(wet, Vector3(x + u, y0, z0 + 0.005f), Vector3(x + u, y1, z1 - 0.005f), up, width, thick, plankLook(0.62f));
                    }
                    z += len;
                }
            };
            for (int k = 0; k < 3; ++k) planks(-hi + (static_cast<float>(k) + 0.5f) * flumeInner / 3.f, -0.025f, flumeInner / 3.f - 0.004f, 0.05f, Vector3(0, 1, 0), 1.3f * static_cast<float>(k));
            for (float s : {-1.f, 1.f}) {
                for (int k = 0; k < 3; ++k) {
                    planks(s * (hi + 0.025f), 0.075f + 0.15f * static_cast<float>(k), 0.05f, 0.148f, Vector3(s, 0, 0), 0.9f * static_cast<float>(k) + (s > 0 ? 2.f : 0.f));
                }
            }
            // yokes: a sill under the floor, posts outside the walls, a tie across the top at every other one
            int yoke = 0;
            for (float z = zB - 0.15f; z > zA + 0.1f; z -= 1.2f, ++yoke) {
                const float f = floorAt(z);
                appendBeam(dry, Vector3(x - hi - 0.22f, f - 0.11f, z), Vector3(x + hi + 0.22f, f - 0.11f, z), Vector3(0, 1, 0), 0.1f, 0.12f, plankLook(0.8f));
                for (float s : {-1.f, 1.f})
                    appendBeam(dry, Vector3(x + s * (hi + 0.1f), f - 0.17f, z), Vector3(x + s * (hi + 0.1f), f + flumeWall + 0.02f, z), Vector3(s, 0, 0), 0.1f, 0.08f, plankLook(0.8f));
                if (yoke % 2 == 0) appendBeam(dry, Vector3(x - hi - 0.16f, f + flumeWall - 0.02f, z), Vector3(x + hi + 0.16f, f + flumeWall - 0.02f, z), Vector3(0, 1, 0), 0.06f, 0.08f, plankLook(0.8f));
            }
            // stringers under the yokes
            for (float s : {-1.f, 1.f}) {
                float z = zA;
                while (z < zB - 0.1f) {
                    const float z1 = std::min(z + rng.range(5.f, 6.5f), zB - 0.1f);
                    appendBeam(dry, Vector3(x + s * 0.33f, floorAt(z) - 0.3f, z), Vector3(x + s * 0.33f, floorAt(z1) - 0.3f, z1), Vector3(0, 1, 0), 0.15f, 0.26f, plankLook(0.75f));
                    z = z1;
                }
            }
            // the bents, every 2.4 m, stepping aside at the bridge and stopping short of the wheel
            std::vector<float> bentZ;
            for (float z = wheelZ - wheelRadius - 0.35f; z > zA + 0.5f; z -= 2.4f) {
                if (std::abs(z - bridgeZ) < 1.4f) continue;
                bentZ.push_back(z);
            }
            bentZ.push_back(bridgeZ - 1.45f);
            bentZ.push_back(bridgeZ + 1.45f);
            int bentIndex = 0;
            for (float z : bentZ) {
                const float top = floorAt(z) - 0.43f;
                const float spreadTop = 0.55f, spreadFoot = 0.8f;
                float footY[2];
                for (int sIdx = 0; sIdx < 2; ++sIdx) {
                    const float s = sIdx == 0 ? -1.f : 1.f;
                    const float fx = x + s * spreadFoot;
                    float g = site::groundHeight(fx, z);
                    // inside the stream: stand on the bed (the water covers the foot)
                    footY[sIdx] = g;
                    // a flat footing stone
                    StoneLook look;
                    look.seed = rng.next();
                    look.tint = V3(0.92f, 0.9f, 0.86f) * rng.range(0.85f, 1.05f);
                    look.weather = [](const Vector3& p, const Vector3& n, const V3& c) {
                        return mix(c * (0.7f + 0.3f * smooth(-0.2f, 1.f, p.y - site::nominalWater(p.z))), srgb(90, 110, 50), 0.5f * smooth(0.5f, 0.9f, n.y));
                    };
                    Frame fr{Vector3(fx, g + 0.05f, z), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)};
                    appendStone(footings, fr, 0.3f, 0.14f, 0.3f, look);
                }
                for (int sIdx = 0; sIdx < 2; ++sIdx) {
                    const float s = sIdx == 0 ? -1.f : 1.f;
                    appendBeam(dry, Vector3(x + s * spreadFoot, footY[sIdx] + 0.18f, z), Vector3(x + s * spreadTop, top - 0.1f, z), Vector3(0, 0, 1), 0.18f, 0.18f, plankLook(0.72f));
                }
                appendBeam(dry, Vector3(x - 0.75f, top, z), Vector3(x + 0.75f, top, z), Vector3(0, 1, 0), 0.2f, 0.2f, plankLook(0.72f));
                // cross braces on the bent's face, where it is tall enough
                const float low = std::max(footY[0], footY[1]) + 0.5f;
                if (top - low > 1.2f) {
                    for (float s : {-1.f, 1.f}) {
                        const float zz = z + s * 0.11f;
                        appendBeam(dry, Vector3(x - 0.75f, low, zz), Vector3(x + 0.6f, top - 0.25f, zz), Vector3(0, 0, 1), 0.15f, 0.045f, plankLook(0.75f));
                        appendBeam(dry, Vector3(x + 0.75f, low, zz + s * 0.05f), Vector3(x - 0.6f, top - 0.25f, zz + s * 0.05f), Vector3(0, 0, 1), 0.15f, 0.045f, plankLook(0.75f));
                    }
                }
                // sway bracing between every third pair of bents
                if (bentIndex % 3 == 0 && bentIndex + 1 < static_cast<int>(bentZ.size()) && std::abs(bentZ[bentIndex + 1] - z) < 2.6f) {
                    const float z2 = bentZ[bentIndex + 1];
                    const float top2 = floorAt(z2) - 0.43f;
                    appendBeam(dry, Vector3(x - 0.68f, low, z), Vector3(x - 0.62f, top2 - 0.3f, z2), Vector3(1, 0, 0), 0.13f, 0.045f, plankLook(0.75f));
                }
                ++bentIndex;
            }
            // over the wheel the trough runs on past its last bent, held by a cantilever from the mill's wall: a beam
            // out of the wall under the stringers and a raking strut below it
            {
                const float zb = wheelZ - 0.9f;
                const float top = floorAt(zb) - 0.4f;
                appendBeam(dry, Vector3(millWest + 0.15f, top, zb), Vector3(x - 0.55f, top, zb), Vector3(0, 1, 0), 0.2f, 0.22f, plankLook(0.7f));
                appendBeam(dry, Vector3(millWest + 0.02f, top - 1.3f, zb), Vector3(x + 0.1f, top - 0.12f, zb), Vector3(0, 0, 1), 0.15f, 0.14f, plankLook(0.7f));
            }
            // the mouth: a short sill and side boards past the last yoke
            appendBeam(wet, Vector3(x, floorAt(zB) - 0.02f, zB - 0.02f), Vector3(x, floorAt(zB) - 0.02f, zB + 0.06f), Vector3(0, 1, 0), flumeInner + 0.1f, 0.06f, plankLook(0.5f));
            root->add(wet.mesh(m.timberWet));
            root->add(dry.mesh(m.timber));
            root->add(footings.mesh(m.masonry));
        }
    };

}// namespace mill

#endif
