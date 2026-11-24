// The mill's shell: the stone foundation (the basement, where the gearing turns), the timber frame on it clad in
// vertical boards and battens gone silver, the windows with their old sashes and some panes gone, the doors, and the
// steep gabled roof of split shakes with moss in it. The floors, stairs and machinery inside are Interior.hpp's.
//
// Old and neglected, but sound: boards are weathered, stained by rain under the sills and green-black by the wheel,
// a few battens are missing and a few panes are out; nothing has fallen.

#ifndef THREEPP_EXAMPLE_WATERMILL_MILL_HPP
#define THREEPP_EXAMPLE_WATERMILL_MILL_HPP

#include "threepp/objects/Group.hpp"

#include <map>

#include "Geometry.hpp"
#include "Masonry.hpp"
#include "Materials.hpp"
#include "Site.hpp"

namespace mill {

    using threepp::Vector3;

    // A century has settled the timber frame on its stone base: it racks toward the creek and downstream, more the
    // higher up (a point of the frame at height y is moved by this), and each wall bows between its corners.
    inline Vector3 frameSettle(float y) {
        const float h = std::clamp((y - site::sillTop) / (site::eaves - site::sillTop), 0.f, 1.6f);
        return Vector3(-0.05f, 0.f, 0.022f) * h;
    }

    // one face of the building: its outer plane, a direction along it and the outward normal
    struct Facade {
        Vector3 origin;// the face's start at y = 0
        Vector3 along; // unit, along the wall
        Vector3 out;   // unit, outward
        float length;
        float bulge;   // m the wall bows out (in, if negative) at its middle

        [[nodiscard]] Vector3 at(float u, float y, float w) const {
            Vector3 p = origin + along * u + Vector3(0, y, 0) + out * w;
            if (y > site::sillTop) {
                const float h = std::clamp((y - site::sillTop) / (site::eaves - site::sillTop), 0.f, 1.f);
                const float t = std::clamp(u / length, 0.f, 1.f);
                p += frameSettle(y) + out * (bulge * std::sin(3.14159265f * t) * std::sin(3.14159265f * h));
            }
            return p;
        }
    };

    struct WindowSpec {
        int facade;   // 0 west, 1 south, 2 east, 3 north
        float u;      // centre along the facade
        float y0, y1; // sill and head of the opening
        float width;
    };

    struct DoorSpec {
        int facade;
        float u, y0, y1, width;
    };

    inline const std::array<Facade, 4>& facades() {
        using namespace site;
        static const std::array<Facade, 4> f{{
                {Vector3(millWest, 0, millNorth), Vector3(0, 0, 1), Vector3(-1, 0, 0), millSouth - millNorth, 0.045f},
                {Vector3(millWest, 0, millSouth), Vector3(1, 0, 0), Vector3(0, 0, 1), millEast - millWest, 0.025f},
                {Vector3(millEast, 0, millSouth), Vector3(0, 0, -1), Vector3(1, 0, 0), millSouth - millNorth, -0.02f},
                {Vector3(millEast, 0, millNorth), Vector3(-1, 0, 0), Vector3(0, 0, -1), millEast - millWest, 0.02f},
        }};
        return f;
    }

    inline const std::vector<WindowSpec>& millWindows() {
        using namespace site;
        static const std::vector<WindowSpec> w{
                // south gable, toward the view
                {1, 1.6f, 5.15f, 6.65f, 0.86f},
                {1, 4.9f, 5.15f, 6.65f, 0.86f},
                {1, 1.6f, 7.75f, 9.2f, 0.86f},
                {1, 4.9f, 7.75f, 9.2f, 0.86f},
                {1, 3.25f, 10.05f, 11.25f, 0.8f},
                {1, 3.25f, 12.15f, 12.85f, 0.56f},
                // west, over the wheel
                {0, 3.5f, 5.0f, 6.3f, 0.86f},
                {0, 1.8f, 7.75f, 9.2f, 0.86f},
                {0, 5.0f, 7.75f, 9.2f, 0.86f},
                {0, 7.9f, 7.75f, 9.2f, 0.86f},
                {0, 2.6f, 10.0f, 10.95f, 0.8f},
                {0, 6.4f, 10.0f, 10.95f, 0.8f},
                // east, into the hill
                {2, 2.0f, 5.15f, 6.65f, 0.86f},
                {2, 7.0f, 5.15f, 6.65f, 0.86f},
                {2, 2.0f, 7.75f, 9.2f, 0.86f},
                {2, 7.0f, 7.75f, 9.2f, 0.86f},
                {2, 4.5f, 10.0f, 10.95f, 0.8f},
                // north gable
                {3, 1.7f, 5.15f, 6.65f, 0.86f},
                {3, 4.8f, 7.75f, 9.2f, 0.86f},
                {3, 3.25f, 10.05f, 11.25f, 0.8f},
        };
        return w;
    }

    inline const std::vector<DoorSpec>& millDoors() {
        using namespace site;
        static const std::vector<DoorSpec> d{
                {2, 4.5f, stoneFloor + 0.02f, stoneFloor + 2.3f, 1.8f},// the loading doors on the hill side
                {0, 0.85f, stoneFloor + 0.02f, stoneFloor + 2.05f, 0.9f},// the miller's door to the flume
                {1, 1.6f, basementFloor + 0.02f, 2.95f, 0.92f},            // the basement door in the foundation, ajar
        };
        return d;
    }

    class MillShell {
    public:
        std::shared_ptr<threepp::Group> root = threepp::Group::create();

        explicit MillShell(const Materials& m) {
            buildFoundation(m);
            buildWalls(m);
            buildWindows(m);
            buildDoors(m);
            buildRoof(m);
            for (auto& [mat, b] : wood_) root->add(b.mesh(mat));
        }

    private:
        std::map<std::shared_ptr<threepp::Material>, MeshBuild> wood_;

        MeshBuild& wood(const std::shared_ptr<threepp::Material>& mat) { return wood_[mat]; }

        // weathering of the outside boards by where they are
        static V3 boardWeather(const Vector3& p, const Vector3& n, const V3& c) {
            using namespace site;
            V3 col = c;
            // a century of sun and rain did not weather it evenly: broad patches gone silver, others still brown
            const float patch = fbm3(p.x * 0.45f, p.y * 0.3f, p.z * 0.45f, 3, 901u);
            col = mix(col, col * V3(1.08f, 0.9f, 0.72f), 0.75f * smooth(-0.05f, 0.45f, patch));
            col = col * (0.9f + 0.18f * smooth(-0.4f, 0.4f, fbm3(p.x * 1.3f, p.y * 0.25f, p.z * 1.3f, 2, 902u)));
            // black streaks run down from the eaves and the floor bands, where the water drips off
            const float streakField = fbm3(p.x * 6.f, p.y * 0.35f, p.z * 6.f, 3, 903u);
            const float underDrip = std::max({smooth(eaves - 1.6f, eaves - 0.1f, p.y), smooth(binFloor - 1.2f, binFloor - 0.05f, p.y) * (p.y < binFloor ? 1.f : 0.f),
                                              smooth(garnerFloor - 1.2f, garnerFloor - 0.05f, p.y) * (p.y < garnerFloor ? 1.f : 0.f)});
            col = col * (1.f - 0.55f * underDrip * smooth(0.05f, 0.5f, streakField));
            // green algae and moss low on the boards, over the damp foundation, thickest on the north and the creek side
            const float shade = n.z < -0.5f || n.x < -0.5f ? 1.f : 0.55f;
            const float low = smooth(sillTop + 1.3f, sillTop + 0.05f, p.y) * shade;
            const float clump = smooth(-0.2f, 0.35f, fbm3(p.x * 2.f, p.y * 2.f, p.z * 2.f, 3, 904u));
            col = mix(col, srgb(74, 94, 46), 0.7f * low * clump);
            // splash and damp near the foot of the wall
            col = col * (0.72f + 0.28f * smooth(sillTop, sillTop + 1.0f, p.y));
            // the wheel keeps the west wall round it wet: dark, green-black
            if (n.x < -0.5f) {
                const float wet = smooth(3.4f, 2.4f, std::abs(p.z - wheelZ)) * smooth(7.6f, 5.2f, p.y);
                col = mix(col, srgb(52, 60, 40), 0.65f * wet);
            }
            // the sunny south bleaches, the north greens
            if (n.z > 0.5f) col = col * 1.06f;
            if (n.z < -0.5f) col = mix(col, srgb(96, 104, 80), 0.25f);
            // rain streaks running down from the window sills
            for (const auto& w : millWindows()) {
                const Facade& F = facades()[w.facade];
                if (n.dot(F.out) < 0.5f) continue;
                const float u = (p - F.origin).dot(F.along);
                const float du = std::abs(u - w.u);
                if (du < 0.5f * w.width + 0.05f && p.y < w.y0 && p.y > w.y0 - 1.6f) {
                    const float streak = smooth(0.5f * w.width + 0.05f, 0.1f, du) * smooth(w.y0 - 1.6f, w.y0 - 0.1f, p.y);
                    col = col * (1.f - 0.35f * streak);
                }
            }
            return col;
        }

        // ---- the foundation -----------------------------------------------------------------------------------------
        void buildFoundation(const Materials& m) {
            using namespace site;
            MeshBuild stones, mortar;
            Loop L;
            L.x0 = millWest;
            L.x1 = millEast;
            L.z0 = millNorth;
            L.z1 = millSouth;
            L.base = -0.6f;
            L.top = sillTop;
            L.thickness = foundationThickness;
            L.courseMin = 0.24f;
            L.courseMax = 0.42f;
            L.stoneMin = 0.4f;
            L.stoneMax = 1.05f;
            L.seed = 7;
            L.tint = V3(1.f, 0.98f, 0.95f);
            L.moss = 0.75f;
            const float b = -L.base;
            // west: the axle's hole and two small windows; south: the basement door; east and north: a window each
            L.openings[0] = {{wheelZ - millNorth - 0.35f, wheelZ - millNorth + 0.35f, wheelY - 0.35f + b, wheelY + 0.55f + b},
                             {1.3f, 2.0f, 2.0f + b, 2.9f + b},
                             {7.4f, 8.1f, 2.0f + b, 2.9f + b}};
            L.openings[1] = {{1.1f, 2.1f, basementFloor + b, 3.15f + b}};
            L.openings[2] = {{4.2f, 4.8f, 3.2f + b, 4.0f + b}};
            L.openings[3] = {{3.0f, 3.6f, 2.6f + b, 3.5f + b}};
            L.weather = [](const Vector3& p, const Vector3& n, const V3& c) {
                V3 col = c * (0.7f + 0.3f * smooth(-0.2f, 1.6f, p.y));
                // algae: the stream side, and round the wheel up to its spray
                const bool west = n.x < -0.3f || p.x < millWest + 0.05f;
                float green = west ? smooth(1.8f, 0.2f, p.y) : smooth(1.0f, 0.f, p.y);
                if (west) green = std::max(green, smooth(3.6f, 2.0f, std::abs(p.z - wheelZ)) * smooth(4.4f, 2.5f, p.y));
                green *= 0.6f + 0.4f * clamp01(0.5f + fbm3(p.x * 2.f, p.y * 2.f, p.z * 2.f, 3, 5u));
                col = mix(col, srgb(80, 96, 54), 0.7f * green);
                // moss on the tops of the stones
                col = mix(col, srgb(92, 116, 50), 0.6f * smooth(0.5f, 0.9f, n.y) * smooth(-0.2f, 0.4f, fbm3(p.x * 3.f, p.y * 3.f, p.z * 3.f, 2, 6u)));
                return col;
            };
            buildLoop(stones, mortar, L);
            // timber lintels over the openings
            const auto& F = facades();
            const auto lintel = [&](int k, float u0, float u1, float yTop) {
                const Facade& f = F[k];
                appendBeam(wood(m.timber), f.at(u0 - 0.25f, yTop - 0.11f, -0.5f * foundationThickness), f.at(u1 + 0.25f, yTop - 0.11f, -0.5f * foundationThickness), Vector3(0, 1, 0),
                           foundationThickness - 0.04f, 0.2f, WoodLook{V3(0.75f, 0.7f, 0.66f), static_cast<uint32_t>(k * 13 + static_cast<int>(u0 * 7))});
            };
            for (int k = 0; k < 4; ++k) {
                for (const auto& o : L.openings[k]) lintel(k, o.u0, o.u1, o.v1 - b);
            }
            root->add(stones.mesh(m.masonry));
            root->add(mortar.mesh(m.mortar));
            // the basement floor: packed earth and old flags, seen through the door and windows
            MeshBuild floor;
            appendQuad(floor, Vector3(millWest + 0.3f, basementFloor, millSouth - 0.3f), Vector3(millEast - 0.3f, basementFloor, millSouth - 0.3f),
                       Vector3(millEast - 0.3f, basementFloor, millNorth + 0.3f), Vector3(millWest + 0.3f, basementFloor, millNorth + 0.3f), 3.f, 3.f, V3(0.7f, 0.62f, 0.55f));
            auto floorMesh = floor.mesh(m.groundMat);
            root->add(floorMesh);
        }

        // ---- the timber walls ---------------------------------------------------------------------------------------
        // the roof line over a gable at u (0..length): the eaves plus the rise
        static float gableTop(const Facade& f, float u) {
            using namespace site;
            const float half = 0.5f * f.length;
            return eaves + (half - std::abs(u - half));
        }

        void buildWalls(const Materials& m) {
            using namespace site;
            Rng rng(41);
            const auto& F = facades();
            const std::array<float, 3> bands{binFloor, garnerFloor, eaves};
            for (int k = 0; k < 4; ++k) {
                const Facade& f = F[k];
                const bool gable = (k == 1 || k == 3);
                // the openings in this wall (windows and doors), widened by the casing
                std::vector<std::array<float, 4>> holes;
                for (const auto& w : millWindows())
                    if (w.facade == k) holes.push_back({w.u - 0.5f * w.width - 0.02f, w.u + 0.5f * w.width + 0.02f, w.y0 - 0.02f, w.y1 + 0.02f});
                for (const auto& d : millDoors())
                    if (d.facade == k) holes.push_back({d.u - 0.5f * d.width - 0.02f, d.u + 0.5f * d.width + 0.02f, d.y0 - 0.05f, d.y1 + 0.02f});
                // walk along the wall, board by board
                float u = 0.f;
                std::vector<float> joints;
                while (u < f.length - 0.05f) {
                    float bw = rng.range(0.2f, 0.32f);
                    if (f.length - (u + bw) < 0.14f) bw = f.length - u;
                    const float gap = rng.range(0.003f, 0.011f);
                    const float uc = u + 0.5f * bw;
                    // the board's tint: most silvered alike, the odd one darker or a later, browner replacement
                    V3 tint = V3(1, 1, 1) * rng.range(0.86f, 1.08f);
                    if (rng() < 0.08f) tint = V3(0.82f, 0.72f, 0.6f);
                    if (rng() < 0.05f) tint = tint * 0.75f;
                    // three lifts, split at the floor bands; the bottoms have rotted back unevenly
                    float y = sillTop + 0.02f + std::max(0.f, rng.range(-0.03f, 0.06f));
                    for (int lift = 0; lift < 3; ++lift) {
                        float top = bands[lift] - 0.04f;
                        if (lift == 2 && gable) top = gableTop(f, uc) - 0.05f;
                        // cut round the openings
                        std::vector<std::pair<float, float>> spans{{y, top}};
                        for (const auto& h : holes) {
                            if (uc + 0.5f * bw <= h[0] || uc - 0.5f * bw >= h[1]) continue;
                            std::vector<std::pair<float, float>> next;
                            for (auto [a, b] : spans) {
                                if (h[3] <= a || h[2] >= b) {
                                    next.emplace_back(a, b);
                                    continue;
                                }
                                if (h[2] > a) next.emplace_back(a, h[2]);
                                if (h[3] < b) next.emplace_back(h[3], b);
                            }
                            spans = next;
                        }
                        for (auto [a, b] : spans) {
                            if (b - a < 0.04f) continue;
                            WoodLook look;
                            look.tint = tint;
                            look.seed = rng.next();
                            look.bow = rng.range(-0.004f, 0.006f);
                            look.twist = rng.range(-0.02f, 0.02f);
                            look.weather = boardWeather;
                            appendBeam(wood(m.siding), f.at(uc, a, 0.016f), f.at(uc, b, 0.016f), f.out, bw - gap, 0.022f, look);
                        }
                        y = bands[lift] + 0.04f;
                    }
                    u += bw;
                    if (u < f.length - 0.05f) joints.push_back(u);
                }
                // battens over the joints (a few have fallen off), and the drip boards at the floor bands
                for (float uj : joints) {
                    float y = sillTop + 0.02f;
                    for (int lift = 0; lift < 3; ++lift) {
                        float top = bands[lift] - 0.12f;
                        if (lift == 2 && gable) top = gableTop(f, uj) - 0.08f;
                        if (rng() > 0.06f) {
                            std::vector<std::pair<float, float>> spans{{y, top}};
                            for (const auto& h : holes) {
                                if (uj < h[0] - 0.03f || uj > h[1] + 0.03f) continue;
                                std::vector<std::pair<float, float>> next;
                                for (auto [a, b] : spans) {
                                    if (h[3] <= a || h[2] >= b) {
                                        next.emplace_back(a, b);
                                        continue;
                                    }
                                    if (h[2] > a) next.emplace_back(a, h[2]);
                                    if (h[3] < b) next.emplace_back(h[3] + 0.04f, b);
                                }
                                spans = next;
                            }
                            for (auto [a, b] : spans) {
                                if (b - a < 0.1f) continue;
                                WoodLook look;
                                look.tint = V3(1, 1, 1) * rng.range(0.85f, 1.02f);
                                look.seed = rng.next();
                                look.weather = boardWeather;
                                // nailed up by eye: never quite plumb
                                appendBeam(wood(m.siding), f.at(uj + rng.range(-0.012f, 0.012f), a, 0.037f), f.at(uj + rng.range(-0.012f, 0.012f), b, 0.037f), f.out, 0.055f, 0.02f, look);
                            }
                        }
                        y = bands[lift] + 0.12f;
                    }
                }
                for (int lift = 0; lift < 2; ++lift) {
                    WoodLook look;
                    look.tint = V3(0.92f, 0.9f, 0.86f);
                    look.seed = rng.next();
                    look.sag = 0.01f;
                    look.weather = boardWeather;
                    appendBeam(wood(m.siding), f.at(-0.05f, bands[lift], 0.045f), f.at(f.length + 0.05f, bands[lift], 0.045f), Vector3(0, 1, 0), 0.05f, 0.17f, look);
                }
                // the corner boards
                for (float uc : {0.07f, f.length - 0.07f}) {
                    WoodLook look;
                    look.tint = V3(0.95f, 0.93f, 0.9f);
                    look.seed = rng.next();
                    look.weather = boardWeather;
                    appendBeam(wood(m.siding), f.at(uc, sillTop - 0.05f, 0.05f), f.at(uc, eaves + 0.05f, 0.05f), f.out, 0.14f, 0.03f, look);
                }
                // the frame behind the boards: sill, corner and middle posts, girts every 1.2 m, the plates
                const WoodLook frame{V3(0.7f, 0.64f, 0.58f), static_cast<uint32_t>(k + 1)};
                const float in = -0.1f;// the frame's middle, inside the boards
                // the sill, cut where a door opens in this wall (the doors stand on a threshold, not in the sill)
                {
                    std::vector<std::pair<float, float>> runs{{0.f, f.length}};
                    for (const auto& d : millDoors()) {
                        if (d.facade != k || d.y0 > sillTop + 0.5f || d.y1 < sillTop) continue;
                        std::vector<std::pair<float, float>> next;
                        for (auto [a, b] : runs) {
                            const float c0 = d.u - 0.5f * d.width - 0.02f, c1 = d.u + 0.5f * d.width + 0.02f;
                            if (c1 <= a || c0 >= b) {
                                next.emplace_back(a, b);
                                continue;
                            }
                            if (c0 > a) next.emplace_back(a, c0);
                            if (c1 < b) next.emplace_back(c1, b);
                        }
                        runs = next;
                    }
                    for (auto [a, b] : runs)
                        appendBeam(wood(m.interior), f.at(a, sillTop + 0.125f, -0.15f), f.at(b, sillTop + 0.125f, -0.15f), Vector3(0, 1, 0), 0.3f, 0.25f, frame);
                }
                const int posts = static_cast<int>(f.length / 2.2f);
                for (int p = 0; p <= posts; ++p) {
                    const float up = 0.13f + (f.length - 0.26f) * static_cast<float>(p) / static_cast<float>(posts);
                    float top = eaves;
                    if (gable) top = std::min(gableTop(f, up), ridge) - 0.1f;
                    bool blocked = false;
                    for (const auto& h : holes) blocked |= (up > h[0] - 0.12f && up < h[1] + 0.12f);
                    if (blocked && p != 0 && p != posts) continue;
                    WoodLook look = frame;
                    look.seed = rng.next();
                    appendBeam(wood(m.interior), f.at(up, sillTop + 0.25f, in - 0.06f), f.at(up, top, in - 0.06f), f.out, 0.22f, 0.22f, look);
                }
                for (float gy = sillTop + 1.2f; gy < eaves - 0.3f; gy += 1.25f) {
                    // girts, broken by the openings they meet
                    std::vector<std::pair<float, float>> spans{{0.1f, f.length - 0.1f}};
                    for (const auto& h : holes) {
                        if (gy < h[2] - 0.08f || gy > h[3] + 0.08f) continue;
                        std::vector<std::pair<float, float>> next;
                        for (auto [a, b] : spans) {
                            if (h[1] <= a || h[0] >= b) {
                                next.emplace_back(a, b);
                                continue;
                            }
                            if (h[0] > a) next.emplace_back(a, h[0]);
                            if (h[1] < b) next.emplace_back(h[1], b);
                        }
                        spans = next;
                    }
                    for (auto [a, b] : spans) {
                        if (b - a < 0.2f) continue;
                        WoodLook look = frame;
                        look.seed = rng.next();
                        appendBeam(wood(m.interior), f.at(a, gy, in + 0.035f), f.at(b, gy, in + 0.035f), Vector3(0, 1, 0), 0.06f, 0.14f, look);
                    }
                }
                if (!gable) {
                    appendBeam(wood(m.interior), f.at(-0.1f, eaves - 0.1f, -0.12f), f.at(f.length + 0.1f, eaves - 0.1f, -0.12f), Vector3(0, 1, 0), 0.24f, 0.2f, frame);
                } else {
                    // the tie beam across the gable at the eaves
                    appendBeam(wood(m.interior), f.at(0.1f, eaves - 0.12f, -0.15f), f.at(f.length - 0.1f, eaves - 0.12f, -0.15f), Vector3(0, 1, 0), 0.22f, 0.24f, frame);
                }
            }
        }

        // ---- windows ------------------------------------------------------------------------------------------------
        void buildWindows(const Materials& m) {
            using namespace site;
            Rng rng(97);
            MeshBuild glass;
            const auto& F = facades();
            for (const auto& w : millWindows()) {
                const Facade& f = F[w.facade];
                const float hw = 0.5f * w.width;
                // the frame has racked with the wall: its head stands a little to one side of its sill
                const float rack = rng.range(-0.022f, 0.022f) / std::max(w.y1 - w.y0, 0.1f);
                const auto fa = [&](float u, float y, float wOut) { return f.at(u + rack * (y - w.y0), y, wOut); };
                const WoodLook trim{V3(0.95f, 0.94f, 0.9f) * rng.range(0.85f, 1.f), rng.next()};
                MeshBuild& t = wood(m.trim);
                // casing: head with its drip cap, the sides, the sill sloping out
                appendBeam(t, fa(w.u - hw - 0.14f, w.y1 + 0.07f, 0.06f), fa(w.u + hw + 0.14f, w.y1 + 0.07f, 0.06f), Vector3(0, 1, 0), 0.035f, 0.14f, trim);
                appendBeam(t, fa(w.u - hw - 0.17f, w.y1 + 0.15f, 0.085f), fa(w.u + hw + 0.17f, w.y1 + 0.15f, 0.085f), Vector3(0, 1, 0), 0.07f, 0.025f, trim);
                for (float s : {-1.f, 1.f})
                    appendBeam(t, fa(w.u + s * (hw + 0.065f), w.y0 - 0.02f, 0.06f), fa(w.u + s * (hw + 0.065f), w.y1, 0.06f), f.out, 0.13f, 0.035f, trim);
                appendBeam(t, fa(w.u - hw - 0.16f, w.y0 - 0.03f, 0.04f), fa(w.u + hw + 0.16f, w.y0 - 0.03f, 0.04f), Vector3(0, 1, 0), 0.16f, 0.04f, trim);
                // the jambs lining the opening through the wall
                for (float s : {-1.f, 1.f})
                    appendBeam(t, fa(w.u + s * (hw - 0.012f), w.y0, -0.07f), fa(w.u + s * (hw - 0.012f), w.y1, -0.07f), f.along, 0.024f, 0.2f, trim);
                appendBeam(t, fa(w.u - hw, w.y1 - 0.012f, -0.07f), fa(w.u + hw, w.y1 - 0.012f, -0.07f), f.out, 0.2f, 0.024f, trim);
                // two sashes of six panes; the lower one sits in front of the upper
                const float h = w.y1 - w.y0;
                for (int sash = 0; sash < 2; ++sash) {
                    const float sy0 = w.y0 + 0.02f + static_cast<float>(sash) * (0.5f * h - 0.02f);
                    const float sy1 = sy0 + 0.5f * h;
                    const float depth = sash == 0 ? -0.02f : -0.065f;
                    const float bar = 0.045f;
                    // stiles and rails
                    for (float s : {-1.f, 1.f})
                        appendBeam(t, fa(w.u + s * (hw - 0.025f - bar * 0.5f), sy0, depth), fa(w.u + s * (hw - 0.025f - bar * 0.5f), sy1, depth), f.out, bar, 0.035f, trim);
                    appendBeam(t, fa(w.u - hw + 0.025f, sy0 + 0.5f * bar, depth), fa(w.u + hw - 0.025f, sy0 + 0.5f * bar, depth), Vector3(0, 1, 0), 0.035f, bar, trim);
                    appendBeam(t, fa(w.u - hw + 0.025f, sy1 - 0.5f * bar, depth), fa(w.u + hw - 0.025f, sy1 - 0.5f * bar, depth), Vector3(0, 1, 0), 0.035f, bar, trim);
                    // muntins: three panes across, two up
                    const float ix0 = w.u - hw + 0.025f + bar, ix1 = w.u + hw - 0.025f - bar;
                    const float iy0 = sy0 + bar, iy1 = sy1 - bar;
                    for (int c = 1; c < 3; ++c) {
                        const float x = ix0 + (ix1 - ix0) * static_cast<float>(c) / 3.f;
                        appendBeam(t, fa(x, iy0, depth), fa(x, iy1, depth), f.out, 0.018f, 0.025f, trim);
                    }
                    const float my = 0.5f * (iy0 + iy1);
                    appendBeam(t, fa(ix0, my, depth), fa(ix1, my, depth), Vector3(0, 1, 0), 0.025f, 0.018f, trim);
                    // the panes: most whole and grimy, a few out, a few broken
                    for (int c = 0; c < 3; ++c) {
                        for (int r = 0; r < 2; ++r) {
                            const float px0 = ix0 + (ix1 - ix0) * static_cast<float>(c) / 3.f + 0.009f, px1 = ix0 + (ix1 - ix0) * static_cast<float>(c + 1) / 3.f - 0.009f;
                            const float py0 = r == 0 ? iy0 : my + 0.009f, py1 = r == 0 ? my - 0.009f : iy1;
                            const float fate = rng();
                            if (fate < 0.12f) continue;// gone
                            const Vector3 a = fa(px0, py0, depth), b = fa(px1, py0, depth), cc = fa(px1, py1, depth), d = fa(px0, py1, depth);
                            const V3 tint = V3(1, 1, 1) * rng.range(0.8f, 1.1f);
                            if (fate < 0.18f) {
                                // broken: a shard left in the bottom corner
                                const unsigned base = glass.vertexCount();
                                const Vector3 n = f.out;
                                glass.vertex(a, n, 0, 0, tint);
                                glass.vertex(b, n, 1, 0, tint);
                                glass.vertex(a + (d - a) * rng.range(0.3f, 0.8f) + (b - a) * rng.range(0.1f, 0.4f), n, 0.3f, 0.6f, tint);
                                glass.index.insert(glass.index.end(), {base, base + 1, base + 2});
                                continue;
                            }
                            const unsigned base = glass.vertexCount();
                            const float ou = rng(), ov = rng();
                            glass.vertex(a, f.out, ou, ov, tint);
                            glass.vertex(b, f.out, ou + 0.5f, ov, tint);
                            glass.vertex(cc, f.out, ou + 0.5f, ov + 0.5f, tint);
                            glass.vertex(d, f.out, ou, ov + 0.5f, tint);
                            glass.index.insert(glass.index.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
                        }
                    }
                }
            }
            auto mesh = glass.mesh(m.glassMat);
            mesh->castShadow = false;
            root->add(mesh);
        }

        // ---- doors --------------------------------------------------------------------------------------------------
        void buildDoors(const Materials& m) {
            using namespace site;
            Rng rng(131);
            const auto& F = facades();
            for (const auto& d : millDoors()) {
                const Facade& f = F[d.facade];
                const float hw = 0.5f * d.width;
                const WoodLook trim{V3(0.9f, 0.88f, 0.84f), rng.next()};
                MeshBuild& t = wood(m.trim);
                appendBeam(t, f.at(d.u - hw - 0.12f, d.y1 + 0.07f, 0.06f), f.at(d.u + hw + 0.12f, d.y1 + 0.07f, 0.06f), Vector3(0, 1, 0), 0.035f, 0.14f, trim);
                for (float s : {-1.f, 1.f})
                    appendBeam(t, f.at(d.u + s * (hw + 0.06f), d.y0 - 0.02f, 0.06f), f.at(d.u + s * (hw + 0.06f), d.y1, 0.06f), f.out, 0.12f, 0.035f, trim);
                // the leaves: vertical boards on ledges and a brace, closed (the hill side) or a little open (the flume side)
                const int leaves = d.width > 1.2f ? 2 : 1;
                for (int l = 0; l < leaves; ++l) {
                    const float lw = d.width / static_cast<float>(leaves);
                    const float hingeU = l == 0 ? d.u - hw : d.u + hw;
                    const float dir = l == 0 ? 1.f : -1.f;
                    const float open = leaves == 1 ? 0.35f : 0.f;
                    // the leaf's own frame: from the hinge, along its width, swung inward by `open`
                    const Vector3 hinge = f.at(hingeU, 0, -0.06f);
                    const Vector3 across = (f.along * (dir * std::cos(open)) - f.out * std::sin(open)).normalize();
                    const Vector3 outN = (f.out * std::cos(open) + f.along * (dir * std::sin(open))).normalize();
                    const auto at = [&](float a, float y, float w) { return hinge + across * a + Vector3(0, y, 0) + outN * w; };
                    float a = 0.01f;
                    while (a < lw - 0.02f) {
                        const float bw = std::min(rng.range(0.14f, 0.22f), lw - 0.01f - a);
                        appendBeam(wood(m.siding), at(a + 0.5f * bw, d.y0 + 0.02f, 0.f), at(a + 0.5f * bw, d.y1 - 0.02f, 0.f), outN, bw - 0.004f, 0.025f,
                                   WoodLook{V3(0.85f, 0.8f, 0.74f) * rng.range(0.9f, 1.05f), rng.next()});
                        a += bw;
                    }
                    const float hy = d.y1 - d.y0;
                    for (float ly : {0.25f, 0.5f, 0.82f})
                        appendBeam(wood(m.timber), at(0.04f, d.y0 + ly * hy, -0.03f), at(lw - 0.04f, d.y0 + ly * hy, -0.03f), Vector3(0, 1, 0), 0.03f, 0.13f, WoodLook{V3(0.8f, 0.75f, 0.7f), rng.next()});
                    appendBeam(wood(m.timber), at(0.1f, d.y0 + 0.29f * hy, -0.03f), at(lw - 0.1f, d.y0 + 0.78f * hy, -0.03f), outN, 0.11f, 0.028f, WoodLook{V3(0.8f, 0.75f, 0.7f), rng.next()});
                }
            }
            // the loading doors' landing on the hill side, and its steps down to the terrace
            const Facade& e = F[2];
            const DoorSpec& load = millDoors()[0];
            MeshBuild& t = wood(m.timber);
            const float landY = load.y0 - 0.04f;
            for (float a = -1.3f; a <= 1.31f; a += 0.2f)
                appendBeam(t, e.at(load.u + a, landY, 0.05f), e.at(load.u + a, landY, 1.5f), Vector3(0, 1, 0), 0.19f, 0.045f, WoodLook{V3(0.85f, 0.8f, 0.75f) * rng.range(0.85f, 1.05f), rng.next()});
            for (float a : {-1.3f, 1.3f}) {
                for (float w : {0.3f, 1.4f}) {
                    const Vector3 top = e.at(load.u + a, landY - 0.05f, w);
                    appendBeam(t, Vector3(top.x, 2.2f, top.z), top, Vector3(1, 0, 0), 0.16f, 0.16f, WoodLook{V3(0.7f, 0.66f, 0.6f), rng.next()});
                }
            }
            for (int s = 0; s < 6; ++s) {
                const float y = landY - 0.3f * static_cast<float>(s + 1);
                if (y < site::groundHeight(e.at(load.u + 1.9f, 0, 1.0f).x, e.at(load.u + 1.9f, 0, 1.0f).z) + 0.1f) break;
                appendBeam(t, e.at(load.u + 1.45f + 0.28f * static_cast<float>(s), y, 0.4f), e.at(load.u + 1.45f + 0.28f * static_cast<float>(s), y, 1.4f), Vector3(0, 1, 0), 0.26f, 0.05f, WoodLook{V3(0.8f, 0.76f, 0.7f), rng.next()});
            }
        }

        // ---- the roof -----------------------------------------------------------------------------------------------
        void buildRoof(const Materials& m) {
            using namespace site;
            Rng rng(173);
            const float midX = 0.5f * (millWest + millEast);
            const float half = 0.5f * (millEast - millWest);
            const float over = 0.45f, gableOver = 0.35f;
            const float z0 = millNorth - gableOver, z1 = millSouth + gableOver;
            const float rafterTop = eaves + 0.16f;// the roof plane over the outer wall line
            // a point on a slope: s = -1 west, +1 east; d = metres in from the eave's edge along the slope
            const float slopeLen = (half + over) * std::sqrt(2.f);
            // the roof has sagged between its gables, the ridge most, and its planes are no longer flat
            const float zMid = 0.5f * (z0 + z1), zHalf = 0.5f * (z1 - z0);
            const auto sagAt = [&](float d, float z) {
                const float along = std::max(0.f, 1.f - ((z - zMid) / zHalf) * ((z - zMid) / zHalf));
                return along * (0.035f + 0.065f * std::clamp(d / slopeLen, 0.f, 1.f)) + 0.012f * fbm2(z * 0.45f, d * 0.6f, 2, 91u) * along;
            };
            const auto onSlope = [&](float s, float d, float z, float lift) {
                const float x = (s < 0 ? millWest - over : millEast + over) - s * d / std::sqrt(2.f);
                const float y = rafterTop - over + d / std::sqrt(2.f);
                const Vector3 n(s / std::sqrt(2.f), 1.f / std::sqrt(2.f), 0);
                return Vector3(x, y - sagAt(d, z), z) + n * lift + frameSettle(y);
            };
            MeshBuild& frame = wood(m.interior);
            // rafters (their tails show under the eaves), collar ties, the ridge board
            for (float z = z0 + 0.05f; z <= z1 - 0.04f; z += 0.6f) {
                for (float s : {-1.f, 1.f}) {
                    const Vector3 a = onSlope(s, 0.05f, z, -0.08f), b = onSlope(s, slopeLen - 0.02f, z, -0.08f);
                    appendBeam(frame, a, b, Vector3(s / std::sqrt(2.f), 1.f / std::sqrt(2.f), 0), 0.06f, 0.15f, WoodLook{V3(0.72f, 0.66f, 0.6f), rng.next()});
                }
                if (std::fmod(z - z0, 1.2f) < 0.3f && z > millNorth + 0.2f && z < millSouth - 0.2f)
                    appendBeam(frame, Vector3(millWest + 1.6f, eaves + 1.5f, z + 0.07f), Vector3(millEast - 1.6f, eaves + 1.5f, z + 0.07f), Vector3(0, 1, 0), 0.05f, 0.16f, WoodLook{V3(0.72f, 0.66f, 0.6f), rng.next()});
            }
            // the ridge board, in lengths that follow the sag
            for (float z = z0; z < z1 - 0.01f; z += 0.5f * (z1 - z0) / 4.f) {
                const float zb = std::min(z + 0.5f * (z1 - z0) / 4.f, z1);
                const Vector3 a = onSlope(-1.f, slopeLen, z, 0.f), b = onSlope(-1.f, slopeLen, zb, 0.f);
                appendBeam(frame, Vector3(a.x + 0.5f * over * 0.f, a.y + 0.06f, z), Vector3(b.x, b.y + 0.06f, zb), Vector3(1, 0, 0), 0.2f, 0.04f, WoodLook{V3(0.7f, 0.64f, 0.58f), rng.next()});
            }
            // skip sheathing across the rafters: boards with gaps (light shows between them from inside where shakes are thin)
            for (float s : {-1.f, 1.f}) {
                for (float d = 0.08f; d < slopeLen - 0.05f; d += 0.24f) {
                    appendBeam(frame, onSlope(s, d, z0, 0.0f), onSlope(s, d, z1, 0.0f), Vector3(s / std::sqrt(2.f), 1.f / std::sqrt(2.f), 0), 0.15f, 0.02f, WoodLook{V3(0.75f, 0.7f, 0.64f) * rng.range(0.85f, 1.05f), rng.next(), sagAt(d, zMid)});
                }
            }
            // the shakes: courses from the eaves up, each shake its own piece, butts proud of the course above
            MeshBuild& shakes = wood(m.shingles);
            const float exposure = 0.24f, length = 0.6f;
            for (float s : {-1.f, 1.f}) {
                int course = 0;
                for (float d = -0.03f; d < slopeLen - 0.1f; d += exposure, ++course) {
                    float z = z0 - 0.02f - rng.range(0.f, 0.12f);
                    while (z < z1) {
                        const float w = std::min(rng.range(0.1f, 0.3f), z1 + 0.02f - z);
                        if (w < 0.05f) break;
                        const float zc = z + 0.5f * w;
                        z += w + rng.range(0.004f, 0.012f);
                        if (rng() < 0.012f && d > 0.5f) continue;// a lost shake
                        // the butts do not keep a ruled line: each sits a little up or down the slope
                        const float dButt = d + rng.range(-0.025f, 0.02f);
                        const float dTop = std::min(dButt + length, slopeLen + 0.02f);
                        // each lies on the ones below: the butt lifted more than the tip
                        const float liftButt = 0.022f + rng.range(0.f, 0.009f), liftTip = 0.008f;
                        Vector3 a = onSlope(s, dButt, zc, liftButt), b = onSlope(s, dTop, zc, liftTip);
                        const float yaw = rng.range(-0.04f, 0.04f);
                        b.z += yaw * (dTop - d);
                        // moss: clumps, thickest low on the roof and where the shakes are old
                        const float mossN = fbm2(zc * 0.8f + s * 7.f, d * 0.9f, 3, 77u);
                        const float moss = clamp01(smooth(0.0f, 0.5f, mossN) * (0.5f + 0.5f * smooth(3.0f, 0.3f, d)) + 0.15f * rng());
                        V3 tint = V3(1, 1, 1) * rng.range(0.7f, 1.05f);
                        tint = mix(tint, V3(0.42f, 0.55f, 0.26f), 0.75f * moss);
                        if (rng() < 0.04f) tint = V3(0.95f, 0.82f, 0.66f);// a newer patch shake
                        WoodLook look;
                        look.tint = tint;
                        look.seed = rng.next();
                        look.sag = -rng.range(0.002f, 0.01f);// cupped
                        look.twist = rng.range(-0.06f, 0.06f);
                        appendBeam(shakes, a, b, Vector3(s / std::sqrt(2.f), 1.f / std::sqrt(2.f), 0), w, 0.014f, look);
                    }
                }
            }
            // the ridge cap, the rake (barge) boards on the gables, the fascia on the eaves
            const WoodLook cap{V3(0.6f, 0.58f, 0.55f), rng.next()};
            for (float s : {-1.f, 1.f}) {
                // the ridge cap in four lengths, following the sag
                for (int k = 0; k < 4; ++k) {
                    const float za = z0 - 0.05f + (z1 - z0 + 0.1f) * static_cast<float>(k) / 4.f, zb = za + (z1 - z0 + 0.1f) / 4.f + 0.03f;
                    appendBeam(shakes, onSlope(s, slopeLen - 0.12f, za, 0.05f), onSlope(s, slopeLen - 0.12f, zb, 0.05f), Vector3(s / std::sqrt(2.f), 1.f / std::sqrt(2.f), 0), 0.24f, 0.025f, cap);
                }
                for (float z : {z0 - 0.02f, z1 + 0.02f}) {
                    appendBeam(wood(m.trim), onSlope(s, 0.f, z, -0.05f), onSlope(s, slopeLen + 0.03f, z, -0.05f), Vector3(0, 0, 1), 0.035f, 0.2f, WoodLook{V3(0.85f, 0.83f, 0.8f), rng.next()});
                }
                appendBeam(wood(m.trim), onSlope(s, 0.02f, z0 - 0.02f, -0.1f), onSlope(s, 0.02f, z1 + 0.02f, -0.1f), Vector3(s, 0.f, 0.f), 0.035f, 0.18f, WoodLook{V3(0.85f, 0.83f, 0.8f), rng.next(), sagAt(0.02f, zMid)});
            }
        }
    };

}// namespace mill

#endif
