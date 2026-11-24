// The overshot wheel: an iron wheel gone to rust, turning in the plane x = wheelX between the mill's wall and its
// stone pier. Two shrouds (annular plates) carry 36 bent-plate buckets on a closed sole; twelve flat arms a side run
// to cast hubs on the axle. Its angle is a rotation about +x: positive turns the top toward +z (downstream), the way
// the water from the flume drives it.
//
// The buckets' geometry (in the wheel's own (y, z) plane, angle phi from the top toward +z):
//   plate k: from the sole (rSole, theta_k) out to its knee (rKnee, theta_k), then its lip bends BACK (against the
//   turn) to (rRim, theta_k - lipBack). Cell k lies between plate k and plate k+1: its floor on the descending side is
//   plate k+1, whose bent lip keeps the water in until the wheel has carried it well down; its mouth is the gap from
//   plate k's knee to plate k+1's lip.

#ifndef THREEPP_EXAMPLE_WATERMILL_WHEEL_HPP
#define THREEPP_EXAMPLE_WATERMILL_WHEEL_HPP

#include "threepp/objects/Group.hpp"

#include "Geometry.hpp"
#include "Masonry.hpp"
#include "Materials.hpp"
#include "Site.hpp"

#include <array>

namespace mill {

    // the bearings' feet stand this far below the axle, on stone
    constexpr float bearingSeat = 0.3f;

    struct WheelShape {
        static constexpr int buckets = site::wheelBuckets;
        static constexpr float rRim = site::wheelRadius;
        static constexpr float rKnee = site::wheelRadius - 0.2f;
        static constexpr float rSole = site::wheelRadius - 0.46f;
        static constexpr float lipBack = 0.105f;// radians the lip bends back
        static constexpr float halfWidth = 0.5f * site::wheelWidth;
        static constexpr float shroud = 0.014f;// plate thickness
        static constexpr float pitch = 6.2831853f / static_cast<float>(buckets);

        // a point of the wheel's own plane (radius, angle from the top toward +z) as (y, z) offsets from the axle
        static std::array<float, 2> yz(float r, float phi) { return {r * std::cos(phi), r * std::sin(phi)}; }
    };

    class WheelModel {
    public:
        std::shared_ptr<threepp::Group> root = threepp::Group::create(); // at the axle, not turning
        std::shared_ptr<threepp::Group> rotor = threepp::Group::create();// turns with the wheel

        explicit WheelModel(const Materials& m) {
            using namespace site;
            using threepp::Vector3;
            root->position.set(wheelX, wheelY, wheelZ);
            root->add(rotor);
            MeshBuild iron, dark;
            Rng rng(5);
            const WoodLook rust{V3(1, 1, 1), 3};
            const float hw = WheelShape::halfWidth;
            // the shrouds: rings of plate, segmented with a lapped joint every 30 degrees
            const int segs = 144;
            for (float side : {-1.f, 1.f}) {
                const float x = side * (hw - 0.5f * WheelShape::shroud);
                const unsigned base = iron.vertexCount();
                for (int k = 0; k <= segs; ++k) {
                    const float phi = 6.2831853f * static_cast<float>(k) / static_cast<float>(segs);
                    for (float r : {WheelShape::rSole - 0.04f, WheelShape::rRim}) {
                        const auto p = WheelShape::yz(r, phi);
                        const V3 tint = V3(1, 1, 1) * (0.9f + 0.15f * fbm2(phi * 4.f, r * 3.f + side, 2, 41u));
                        iron.vertex(Vector3(x, p[0], p[1]), Vector3(side, 0, 0), phi * r / 0.6f, r / 0.6f, tint);
                    }
                }
                for (int k = 0; k < segs; ++k) {
                    const unsigned a = base + 2 * k, b = a + 1, c = a + 2, d = a + 3;
                    if (side < 0) iron.index.insert(iron.index.end(), {a, c, b, b, c, d});
                    else iron.index.insert(iron.index.end(), {a, b, c, b, d, c});
                }
                // the inner face too (the shroud is seen from inside the buckets)
                const unsigned base2 = iron.vertexCount();
                const float xi = side * (hw - WheelShape::shroud);
                for (int k = 0; k <= segs; ++k) {
                    const float phi = 6.2831853f * static_cast<float>(k) / static_cast<float>(segs);
                    for (float r : {WheelShape::rSole - 0.04f, WheelShape::rRim}) {
                        const auto p = WheelShape::yz(r, phi);
                        iron.vertex(Vector3(xi, p[0], p[1]), Vector3(-side, 0, 0), phi * r / 0.6f + 0.3f, r / 0.6f, V3(0.7f, 0.66f, 0.6f));
                    }
                }
                for (int k = 0; k < segs; ++k) {
                    const unsigned a = base2 + 2 * k, b = a + 1, c = a + 2, d = a + 3;
                    if (side < 0) iron.index.insert(iron.index.end(), {a, b, c, b, d, c});
                    else iron.index.insert(iron.index.end(), {a, c, b, b, c, d});
                }
                // the rim's outer edge band and the sole's edge, as thin cylinders of the plate's thickness
                // (an iron angle riveted round the rim)
                for (int k = 0; k < segs; ++k) {
                    const float p0 = 6.2831853f * static_cast<float>(k) / static_cast<float>(segs), p1 = 6.2831853f * static_cast<float>(k + 1) / static_cast<float>(segs);
                    const auto a = WheelShape::yz(WheelShape::rRim + 0.012f, p0), b = WheelShape::yz(WheelShape::rRim + 0.012f, p1);
                    appendBeam(iron, Vector3(side * (hw - 0.03f), a[0], a[1]), Vector3(side * (hw - 0.03f), b[0], b[1]), Vector3(side, 0, 0), 0.025f, 0.06f, rust);
                }
                // bolt heads round the shroud
                for (int k = 0; k < WheelShape::buckets * 2; ++k) {
                    const float phi = WheelShape::pitch * 0.5f * static_cast<float>(k);
                    const auto p = WheelShape::yz(WheelShape::rKnee - 0.08f * static_cast<float>(k % 2), phi);
                    appendCylinder(dark, Vector3(side * hw, p[0], p[1]), Vector3(side * (hw + 0.012f), p[0], p[1]), 0.016f, 0.014f, 6, WoodLook{V3(1, 1, 1), 7});
                }
            }
            // the sole: the closed inner ring the buckets stand on
            {
                const unsigned base = iron.vertexCount();
                for (int k = 0; k <= segs; ++k) {
                    const float phi = 6.2831853f * static_cast<float>(k) / static_cast<float>(segs);
                    const auto p = WheelShape::yz(WheelShape::rSole, phi);
                    const Vector3 n(0, std::cos(phi), std::sin(phi));
                    for (float x : {-hw + WheelShape::shroud, hw - WheelShape::shroud}) iron.vertex(Vector3(x, p[0], p[1]), n, x / 0.6f, phi * WheelShape::rSole / 0.6f, V3(0.75f, 0.7f, 0.65f));
                }
                for (int k = 0; k < segs; ++k) {
                    const unsigned a = base + 2 * k, b = a + 1, c = a + 2, d = a + 3;
                    iron.index.insert(iron.index.end(), {a, c, b, b, c, d});
                    // and its underside, seen from below
                    iron.index.insert(iron.index.end(), {a, b, c, b, d, c});
                }
            }
            // the buckets: bent plates, each two strips across the wheel's width
            for (int k = 0; k < WheelShape::buckets; ++k) {
                const float th = WheelShape::pitch * static_cast<float>(k);
                const auto s = WheelShape::yz(WheelShape::rSole, th);
                const auto knee = WheelShape::yz(WheelShape::rKnee, th);
                const auto lip = WheelShape::yz(WheelShape::rRim - 0.01f, th - WheelShape::lipBack);
                const float hx = hw - WheelShape::shroud;
                const WoodLook plate{V3(1, 1, 1) * rng.range(0.85f, 1.08f), rng.next()};
                const auto strip = [&](const std::array<float, 2>& a, const std::array<float, 2>& b) {
                    const Vector3 mid(0, 0.5f * (a[0] + b[0]), 0.5f * (a[1] + b[1]));
                    const Vector3 along(0, b[0] - a[0], b[1] - a[1]);
                    Vector3 n = Vector3(1, 0, 0).cross(along);
                    n.normalize();
                    appendBeam(iron, Vector3(-hx, mid.y, mid.z), Vector3(hx, mid.y, mid.z), n, along.length(), 0.01f, plate);
                };
                strip(s, knee);
                strip(knee, lip);
            }
            // the arms: twelve flat bars a side from the hub to the sole, and a ring of struts between them
            for (float side : {-1.f, 1.f}) {
                const float x = side * (hw - 0.06f);
                const float xh = side * 0.38f;// hubs sit a little inboard, the arms splay out to the shrouds
                for (int k = 0; k < 12; ++k) {
                    const float phi = 6.2831853f * (static_cast<float>(k) + 0.25f) / 12.f;
                    const auto a = WheelShape::yz(0.3f, phi), b = WheelShape::yz(WheelShape::rSole - 0.02f, phi);
                    appendBeam(iron, Vector3(xh, a[0], a[1]), Vector3(x, b[0], b[1]), Vector3(1, 0, 0), 0.11f, 0.028f, WoodLook{V3(1, 1, 1) * rng.range(0.85f, 1.05f), rng.next()});
                }
                for (int k = 0; k < 12; ++k) {
                    const float p0 = 6.2831853f * (static_cast<float>(k) + 0.25f) / 12.f, p1 = 6.2831853f * (static_cast<float>(k) + 1.25f) / 12.f;
                    const float r = 1.55f;
                    const auto a = WheelShape::yz(r, p0), b = WheelShape::yz(r, p1);
                    const float xr = xh + (x - xh) * (r - 0.3f) / (WheelShape::rSole - 0.32f);
                    appendCylinder(iron, Vector3(xr, a[0], a[1]), Vector3(xr, b[0], b[1]), 0.016f, 0.016f, 6, rust, false);
                }
                // the hub: a cast boss with bolts
                appendCylinder(dark, Vector3(side * 0.30f, 0, 0), Vector3(side * 0.46f, 0, 0), 0.34f, 0.3f, 24, WoodLook{V3(1, 1, 1), 11});
                for (int b = 0; b < 8; ++b) {
                    const float phi = 6.2831853f * static_cast<float>(b) / 8.f;
                    const auto p = WheelShape::yz(0.24f, phi);
                    appendCylinder(dark, Vector3(side * 0.46f, p[0], p[1]), Vector3(side * 0.5f, p[0], p[1]), 0.025f, 0.022f, 6, WoodLook{V3(1, 1, 1), 13});
                }
            }
            // the axle turns with the wheel: from the pier's bearing through the wall
            appendCylinder(dark, Vector3(axleOuterX - 0.1f - wheelX, 0, 0), Vector3(millWest - wheelX + 0.2f, 0, 0), 0.11f, 0.11f, 20, WoodLook{V3(1, 1, 1), 17});
            // a century on the creek: the paint dulled and gone dark with wet, green algae where the water runs and
            // stands (the buckets, the rims' inner edges), the arms streaked
            for (size_t i = 0; i < iron.col.size(); i += 3) {
                const Vector3 p(iron.pos[i], iron.pos[i + 1], iron.pos[i + 2]);
                const float r = std::hypot(p.y, p.z);
                V3 c(iron.col[i], iron.col[i + 1], iron.col[i + 2]);
                c = c * V3(0.62f, 0.58f, 0.56f) * (0.8f + 0.35f * clamp01(0.5f + fbm3(p.x * 3.f, p.y * 3.f, p.z * 3.f, 3, 5101u)));
                const float wetZone = smooth(WheelShape::rSole - 0.3f, WheelShape::rKnee, r);
                const float algae = wetZone * smooth(-0.25f, 0.35f, fbm3(p.x * 4.f, p.y * 4.f, p.z * 4.f, 3, 5102u));
                c = mix(c, srgb(48, 62, 34), 0.6f * algae);
                c = c * (1.f - 0.3f * smooth(0.f, 0.6f, fbm3(p.x * 9.f, r * 1.2f, std::atan2(p.z, p.y) * 4.f, 2, 5103u)) * smooth(1.f, 2.4f, r));
                iron.col[i] = c.r;
                iron.col[i + 1] = c.g;
                iron.col[i + 2] = c.b;
            }
            auto ironMesh = iron.mesh(m.iron);
            auto darkMesh = dark.mesh(m.blackIron, false);
            rotor->add(ironMesh);
            rotor->add(darkMesh);

            // The bearings, which do not turn: cast plummer blocks, each bolted down on stone, the pier's cap stone
            // outside and a corbel in the mill's wall. Their feet stand on the stone at bearingSeat below the axle.
            MeshBuild fixed;
            const float seat = -bearingSeat;
            for (float bx : {axleOuterX - wheelX, millWest - 0.05f - wheelX}) {
                const WoodLook cast{V3(1, 1, 1), 19};
                // the sole: a plate as long as the block's feet, on the stone
                appendBeam(fixed, Vector3(bx, seat + 0.025f, -0.31f), Vector3(bx, seat + 0.025f, 0.31f), Vector3(0, 1, 0), 0.3f, 0.05f, cast);
                // the body from the sole up to the shaft, and the housing round the shaft
                appendBeam(fixed, Vector3(bx, seat + 0.05f + 0.5f * (-0.05f - seat - 0.05f), -0.19f), Vector3(bx, seat + 0.05f + 0.5f * (-0.05f - seat - 0.05f), 0.19f),
                           Vector3(0, 1, 0), 0.24f, -0.05f - seat - 0.05f, cast);
                appendCylinder(fixed, Vector3(bx - 0.13f, 0, 0), Vector3(bx + 0.13f, 0, 0), 0.17f, 0.17f, 18, WoodLook{V3(1, 1, 1), 23});
                // the cap's flanges and their bolts, and the hold-down bolts through the feet
                appendBeam(fixed, Vector3(bx, 0.f, -0.27f), Vector3(bx, 0.f, 0.27f), Vector3(0, 1, 0), 0.2f, 0.05f, cast);
                for (float bz : {-0.23f, 0.23f}) {
                    appendCylinder(fixed, Vector3(bx, 0.02f, bz), Vector3(bx, 0.06f, bz), 0.024f, 0.022f, 6, cast);
                    for (float bxo : {-0.09f, 0.09f}) appendCylinder(fixed, Vector3(bx + bxo, seat + 0.045f, bz + 0.04f * (bz > 0 ? 1.f : -1.f)), Vector3(bx + bxo, seat + 0.085f, bz + 0.04f * (bz > 0 ? 1.f : -1.f)), 0.022f, 0.02f, 6, cast);
                }
            }
            root->add(fixed.mesh(m.blackIron, false));
        }

        void setAngle(float angle) const { rotor->rotation.x = angle; }
    };

    // the stone pier under the axle's outer bearing
    inline std::shared_ptr<threepp::Mesh> wheelPier(const Materials& m, std::shared_ptr<threepp::Mesh>& mortarOut) {
        using namespace site;
        MeshBuild stones, mortar;
        Loop L;
        L.x0 = pierX0;
        L.x1 = pierX1;
        L.z0 = pierZ0;
        L.z1 = pierZ1;
        L.base = -0.7f;
        L.top = wheelY - bearingSeat - 0.16f;// the cap stone goes on top
        L.thickness = 0.5f * (pierX1 - pierX0);// so the pier is solid: every course's stones meet in the middle
        L.courseMin = 0.22f;
        L.courseMax = 0.36f;
        L.stoneMin = 0.3f;
        L.stoneMax = 0.6f;
        L.seed = 301;
        L.moss = 1.15f;
        L.mossLine = wheelY - 0.6f;// the wheel's spray
        L.weather = [](const threepp::Vector3& p, const threepp::Vector3& n, const V3& c) {
            V3 col = c * (0.65f + 0.35f * smooth(0.f, 2.f, p.y));
            col = mix(col, srgb(70, 88, 48), 0.6f * smooth(2.8f, 0.4f, p.y) * clamp01(0.6f + fbm3(p.x * 3.f, p.y * 3.f, p.z * 3.f, 2, 9u)));
            col = mix(col, srgb(88, 112, 48), 0.6f * smooth(0.5f, 0.9f, n.y));
            return col;
        };
        buildLoop(stones, mortar, L);
        // the cap: one long slab over the whole pier, a little proud of it all round; the bearing is bolted to it
        {
            StoneLook cap;
            cap.seed = 3101;
            cap.tint = V3(0.86f, 0.84f, 0.8f);
            cap.round = 0.02f;
            cap.pillow = 0.008f;
            cap.chips = 0.012f;
            cap.weather = L.weather;
            Frame f{threepp::Vector3(0.5f * (pierX0 + pierX1), wheelY - bearingSeat - 0.08f, 0.5f * (pierZ0 + pierZ1)), threepp::Vector3(0, 0, 1), threepp::Vector3(0, 1, 0), threepp::Vector3(-1, 0, 0)};
            appendStone(stones, f, 0.5f * (pierZ1 - pierZ0) + 0.07f, 0.08f, 0.5f * (pierX1 - pierX0) + 0.06f, cap);
        }
        // the corbel in the mill's wall that carries the inner bearing, filling the bottom of the axle's hole
        {
            StoneLook corbel;
            corbel.seed = 3102;
            corbel.tint = V3(0.8f, 0.78f, 0.74f);
            corbel.round = 0.02f;
            corbel.weather = L.weather;
            Frame f{threepp::Vector3(millWest, wheelY - bearingSeat - 0.3f, wheelZ), threepp::Vector3(0, 0, 1), threepp::Vector3(0, 1, 0), threepp::Vector3(-1, 0, 0)};
            appendStone(stones, f, 0.37f, 0.3f, 0.31f, corbel);
        }
        mortarOut = mortar.mesh(m.mortar);
        return stones.mesh(m.masonry);
    }

}// namespace mill

#endif
