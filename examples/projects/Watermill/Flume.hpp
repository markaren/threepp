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
            for (float z = wheelZ - wheelRadius - 0.35f; z > headZ1 + 0.6f; z -= 2.4f) bentZ.push_back(z);
            int bentIndex = 0;
            for (float z : bentZ) {
                const float top = floorAt(z) - 0.43f;
                const float bentLean = rng.range(-0.09f, 0.09f);
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
                    // the bent leans a little along the flume and each post is a little off its line
                    appendBeam(dry, Vector3(x + s * spreadFoot + rng.range(-0.04f, 0.04f), footY[sIdx] + 0.18f, z - 0.5f * bentLean), Vector3(x + s * spreadTop, top - 0.1f, z + 0.5f * bentLean),
                               Vector3(0, 0, 1), 0.18f, 0.18f, plankLook(0.72f));
                }
                appendBeam(dry, Vector3(x - 0.75f, top + rng.range(-0.02f, 0.02f), z + 0.5f * bentLean), Vector3(x + 0.75f, top + rng.range(-0.02f, 0.02f), z + 0.5f * bentLean), Vector3(0, 1, 0), 0.2f, 0.2f, plankLook(0.72f));
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
            buildHeadBox(m, dry, footings, rng);
            root->add(wet.mesh(m.timberWet));
            root->add(dry.mesh(m.timber));
            root->add(footings.mesh(m.masonry));
        }

    private:
        // the head box, its stone base, its roof, and the pipe that feeds it
        void buildHeadBox(const Materials& m, MeshBuild& dry, MeshBuild& stones, Rng& rng) {
            using namespace site;
            using threepp::Vector3;
            const float x = flumeX, hx = headHalfX;
            const float f = floorAt(flumeStartZ);// the trough's floor where it starts
            const float baseTop = f - 0.43f;     // the stone base carries the box and the trough's first yokes
            const float lowGround = std::min({groundHeight(x - hx, headZ0), groundHeight(x + hx, headZ0), groundHeight(x - hx, headZ1), groundHeight(x + hx, headZ1)});
            MeshBuild mortar;
            Loop L;
            L.x0 = x - hx;
            L.x1 = x + hx;
            L.z0 = headZ0 - 0.05f;
            L.z1 = headZ1 + 0.05f;
            L.base = lowGround - 0.5f;
            L.top = baseTop;
            L.thickness = 0.42f;
            L.batter = 0.03f;
            L.courseMin = 0.26f;
            L.courseMax = 0.4f;
            L.stoneMin = 0.4f;
            L.stoneMax = 0.85f;
            L.seed = 1711;
            L.tint = V3(0.95f, 0.94f, 0.9f);
            L.moss = 0.75f;
            L.weather = [](const Vector3& p, const Vector3& n, const V3& c) {
                V3 col = c * (0.72f + 0.28f * smooth(groundHeight(p.x, p.z), groundHeight(p.x, p.z) + 1.5f, p.y));
                col = mix(col, srgb(80, 98, 52), 0.5f * smooth(0.5f, 0.9f, n.y));
                // the box leaks: green-black streaks down the base under it
                col = mix(col, srgb(52, 60, 42), 0.4f * smooth(-0.6f, 0.f, p.y - (floorAt(flumeStartZ) - 1.2f)) * smooth(0.f, 0.5f, fbm3(p.x * 6.f, p.y * 0.6f, p.z * 6.f, 2, 7u)));
                return col;
            };
            buildLoop(stones, mortar, L);
            root->add(mortar.mesh(m.mortar));
            // the box: a sill frame on the stone, corner posts, walls of vertical boards (the trough leaves through the
            // downstream wall), a plate round the top
            const float y0 = baseTop, y1 = f + 1.05f;
            const auto look = [&](float shade) {
                WoodLook l;
                l.tint = V3(1, 1, 1) * rng.range(0.82f, 1.02f) * shade;
                l.seed = rng.next();
                l.weather = [](const Vector3& p, const Vector3& n, const V3& c) {
                    (void) n;
                    // dark and green where the water has soaked the boards, low down
                    return mix(c, c * V3(0.55f, 0.6f, 0.5f), smooth(floorAt(flumeStartZ) + 0.6f, floorAt(flumeStartZ) - 0.3f, p.y));
                };
                return l;
            };
            for (float sx : {-1.f, 1.f}) {
                for (float sz : {-1.f, 1.f}) {
                    const float px = x + sx * (hx - 0.08f), pz = (sz < 0 ? headZ0 : headZ1) - sz * 0.08f;
                    appendBeam(dry, Vector3(px, y0, pz), Vector3(px, y1 + 0.1f, pz), Vector3(sx, 0, 0), 0.14f, 0.14f, look(0.8f));
                }
            }
            for (float yy : {y0 + 0.08f, y1 + 0.05f}) {
                appendBeam(dry, Vector3(x - hx, yy, headZ0 + 0.07f), Vector3(x + hx, yy, headZ0 + 0.07f), Vector3(0, 1, 0), 0.14f, 0.12f, look(0.8f));
                appendBeam(dry, Vector3(x - hx, yy, headZ1 - 0.07f), Vector3(x + hx, yy, headZ1 - 0.07f), Vector3(0, 1, 0), 0.14f, 0.12f, look(0.8f));
                appendBeam(dry, Vector3(x - hx + 0.07f, yy, headZ0), Vector3(x - hx + 0.07f, yy, headZ1), Vector3(0, 1, 0), 0.14f, 0.12f, look(0.8f));
                appendBeam(dry, Vector3(x + hx - 0.07f, yy, headZ0), Vector3(x + hx - 0.07f, yy, headZ1), Vector3(0, 1, 0), 0.14f, 0.12f, look(0.8f));
            }
            // boards: along each wall, on the outside of the frame
            const auto wall = [&](const Vector3& a, const Vector3& along, const Vector3& out, float length, bool downstream) {
                for (float u = 0.f; u < length - 0.02f;) {
                    const float w = std::min(rng.range(0.18f, 0.27f), length - u);
                    const Vector3 c = a + along * (u + 0.5f * w) + out * 0.012f;
                    float lo = y0 + 0.02f + std::max(0.f, rng.range(-0.02f, 0.05f));
                    // the trough's opening through the downstream wall
                    if (downstream && std::abs((c - Vector3(x, 0, c.z)).x) < 0.5f * flumeInner + 0.07f) lo = f + flumeWall + 0.08f;
                    appendBeam(dry, Vector3(c.x, lo, c.z), Vector3(c.x, y1 + 0.1f, c.z), out, w - 0.008f, 0.025f, look(1.f));
                    u += w;
                }
            };
            wall(Vector3(x - hx, 0, headZ0), Vector3(1, 0, 0), Vector3(0, 0, -1), 2.f * hx, false);
            wall(Vector3(x - hx, 0, headZ1), Vector3(1, 0, 0), Vector3(0, 0, 1), 2.f * hx, true);
            wall(Vector3(x - hx, 0, headZ0), Vector3(0, 0, 1), Vector3(-1, 0, 0), headZ1 - headZ0, false);
            wall(Vector3(x + hx, 0, headZ0), Vector3(0, 0, 1), Vector3(1, 0, 0), headZ1 - headZ0, false);
            // the roof: a little gable, ridge across the creek (along x), split shakes
            MeshBuild shakes;
            const float over = 0.25f, rise = 0.55f, ridgeY = y1 + 0.2f + rise;
            const float zm = 0.5f * (headZ0 + headZ1), half = 0.5f * (headZ1 - headZ0) + over;
            const float slope = std::hypot(half, rise);
            for (float s : {-1.f, 1.f}) {
                const Vector3 up(0, half / slope, s * rise / slope);// the slope's normal
                for (float d = 0.f; d < slope - 0.05f; d += 0.16f) {
                    const float t = d / slope;
                    const float z = zm + s * half * (1.f - t), y = ridgeY - rise * (1.f - t);
                    for (float xx = x - hx - over; xx < x + hx + over - 0.02f;) {
                        const float w = std::min(rng.range(0.1f, 0.22f), x + hx + over - xx);
                        const float t2 = std::min(1.f, (d + 0.4f) / slope);
                        const Vector3 a(xx + 0.5f * w, y + 0.02f, z), b(xx + 0.5f * w + rng.range(-0.01f, 0.01f), ridgeY - rise * (1.f - t2) + 0.012f, zm + s * half * (1.f - t2));
                        WoodLook l;
                        l.tint = mix(V3(1, 1, 1) * rng.range(0.7f, 1.f), V3(0.45f, 0.56f, 0.28f), 0.5f * smooth(-0.2f, 0.5f, fbm2(xx * 2.f, d * 2.f + s, 2, 88u)));
                        l.seed = rng.next();
                        l.sag = -0.005f;
                        appendBeam(shakes, a, b, up, w - 0.006f, 0.012f, l);
                        xx += w;
                    }
                }
            }
            for (float s : {-1.f, 1.f})
                appendBeam(dry, Vector3(x - hx - over, ridgeY - rise - 0.05f, zm + s * half), Vector3(x + hx + over, ridgeY - rise - 0.05f, zm + s * half), Vector3(0, 1, 0), 0.04f, 0.14f, look(0.85f));
            // the gables: boards up to the roof's underside at both ends, and the barge boards along the rake
            for (float gx : {x - hx + 0.01f, x + hx - 0.01f}) {
                const float out = gx < x ? -1.f : 1.f;
                for (float z = headZ0; z < headZ1 - 0.02f;) {
                    const float w = std::min(rng.range(0.18f, 0.26f), headZ1 - z);
                    const float zc = z + 0.5f * w;
                    const float top = ridgeY - rise * std::abs(zc - zm) / half - 0.03f;
                    appendBeam(dry, Vector3(gx + out * 0.012f, y1 + 0.05f, zc), Vector3(gx + out * 0.012f, top, zc), Vector3(out, 0, 0), w - 0.008f, 0.025f, look(0.95f));
                    z += w;
                }
                for (float s : {-1.f, 1.f})
                    appendBeam(dry, Vector3(gx + out * (over - 0.02f), ridgeY + 0.02f, zm), Vector3(gx + out * (over - 0.02f), ridgeY - rise - 0.02f, zm + s * half), Vector3(out, 0, 0), 0.03f, 0.16f, look(0.85f));
            }
            root->add(shakes.mesh(m.shingles));
            // the supply pipe: out of the bank behind, into the base, a flange at the stone
            MeshBuild iron;
            const float pz = 0.5f * (headZ0 + headZ1), py = lowGround + 0.55f;
            const Vector3 wallPoint(x + hx + 0.02f, py, pz);
            const Vector3 bend(x + hx + 1.6f, py, pz), buried(x + hx + 3.5f, groundHeight(x + hx + 3.5f, pz) - 0.6f, pz);
            appendCylinder(iron, wallPoint, bend, 0.19f, 0.19f, 18, WoodLook{V3(1, 1, 1), 1801}, false);
            appendCylinder(iron, bend, buried, 0.19f, 0.19f, 18, WoodLook{V3(1, 1, 1), 1802}, false);
            for (const Vector3& fl : {wallPoint + Vector3(0.05f, 0, 0), bend})
                appendCylinder(iron, fl - Vector3(0.03f, 0, 0), fl + Vector3(0.03f, 0, 0), 0.27f, 0.27f, 18, WoodLook{V3(1, 1, 1), 1803});
            root->add(iron.mesh(m.iron));
        }
    };

}// namespace mill

#endif
