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
            auto ironMesh = iron.mesh(m.iron);
            auto darkMesh = dark.mesh(m.blackIron, false);
            rotor->add(ironMesh);
            rotor->add(darkMesh);

            // the bearings, which do not turn: a cast pedestal on the pier and a block in the wall
            MeshBuild fixed;
            for (float bx : {axleOuterX - wheelX, millWest - 0.05f - wheelX}) {
                appendBeam(fixed, Vector3(bx, -0.16f, -0.28f), Vector3(bx, -0.16f, 0.28f), Vector3(0, 1, 0), 0.26f, 0.12f, WoodLook{V3(1, 1, 1), 19});
                appendCylinder(fixed, Vector3(bx - 0.13f, 0, 0), Vector3(bx + 0.13f, 0, 0), 0.17f, 0.17f, 16, WoodLook{V3(1, 1, 1), 23});
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
        L.top = wheelY - 0.28f;
        L.thickness = 0.32f;
        L.courseMin = 0.22f;
        L.courseMax = 0.36f;
        L.stoneMin = 0.3f;
        L.stoneMax = 0.6f;
        L.seed = 301;
        L.weather = [](const threepp::Vector3& p, const threepp::Vector3& n, const V3& c) {
            V3 col = c * (0.65f + 0.35f * smooth(0.f, 2.f, p.y));
            col = mix(col, srgb(70, 88, 48), 0.6f * smooth(2.8f, 0.4f, p.y) * clamp01(0.6f + fbm3(p.x * 3.f, p.y * 3.f, p.z * 3.f, 2, 9u)));
            col = mix(col, srgb(88, 112, 48), 0.6f * smooth(0.5f, 0.9f, n.y));
            return col;
        };
        buildLoop(stones, mortar, L);
        mortarOut = mortar.mesh(m.mortar);
        return stones.mesh(m.masonry);
    }

}// namespace mill

#endif
