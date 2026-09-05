// Urban props for a region pack: pier decks, parked cars, moored boats.
//
// The three things a town of 8000 footprints still lacks once the ground, the
// roofs and the trees are right. All three are placed from SURVEYED data, never
// scattered: a car stands in a mapped parking bay or at a mapped kerb, a boat
// lies along a mapped pier inside a mapped marina, a deck follows a mapped pier
// line. Nothing is invented — which is also why nothing lands on a roof or in
// the water: the same footprint / sea / pavement gates the trees use apply.
//
// GEOMETRY, NOT INSTANCES. The obvious build is an InstancedMesh per cell, and
// the plan asked for one; it does not survive contact with this engine. Six body
// colours need per-instance colour, and `instanceColor` is wired on the GL path
// only (GLBindingStates/GLProgram) — the Vulkan deferred renderer these shots
// are judged on ignores it, so every car in the frame would be one colour. So
// the prototype is stamped into a per-cell vertex-coloured buffer instead,
// exactly as GeoBuildings batches its footprints: one draw per 250 m cell either
// way, and the colour is real. A car is ~116 triangles, so a full town of ~3000
// cars is ~350 k triangles spread over ~40 draws.
//
// Header-only, extras. `ground` must be the PROVIDER height (road carve and
// detail relief folded in), not the raw DEM, or every wheel floats or sinks.

#ifndef THREEPP_EXTRAS_TERRAIN_URBANPROPS_HPP
#define THREEPP_EXTRAS_TERRAIN_URBANPROPS_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/road/RoadNetwork.hpp"
#include "threepp/extras/terrain/GeoTerrain.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace threepp::terrain {

    struct UrbanPropsOptions {
        float seaLevel = 0.f;
        // Static ROI, like the forest's: a 4 km pack holds far more town than a
        // frame ever looks at, and a car past ~1.5 km is under a pixel.
        float centerX = 0.f, centerZ = 0.f;
        float halfExtent = 1500.f;
        float cellSize = 250.f;// one draw per cell

        // Pier decks.
        bool decks = true;
        float deckTop = 1.0f;      // m above sea level
        float deckThickness = 0.4f;// m
        float deckWidth = 3.f;     // fallback when the line carries no width

        // Cars.
        bool cars = true;
        float lotBayPitch = 2.5f;  // along the lot's long axis
        float lotRowPitch = 5.0f;  // across it (car length + a share of aisle)
        float lotOccupancy = 0.55f;
        float kerbPitch = 6.f;
        float kerbOccupancy = 0.25f;
        float kerbOffset = 1.0f;    // beyond the paved half width
        float junctionClear = 8.f;  // no kerb car this close to a shared node
        float urbanMin = 0.3f;      // kerb parking needs a town around it

        // Boats.
        bool boats = true;
        float boatPitch = 3.f;
        float boatOccupancy = 0.6f;
        float marinaRadius = 15.f;// a pier this close to a marina counts as in it

        // Gates. `urban` and `footprints` may be null (the gate is then open).
        const UrbanMask* urban = nullptr;
        const FootprintMask* footprints = nullptr;
        std::function<float(float, float)> ground;
    };

    struct UrbanPropsStats {
        int deckLines = 0;
        int carsLot = 0, carsKerb = 0, carCells = 0;
        int boats = 0, marinas = 0;
        int rejectRoof = 0, rejectSea = 0, rejectPaved = 0, rejectJunction = 0;
        size_t meshes = 0, triangles = 0;
    };

    namespace detail {

        inline float upHash01(unsigned int a, unsigned int b, unsigned int c) {
            unsigned int n = a * 374761393u + b * 668265263u + c * 2246822519u + 0x9E3779B9u;
            n = (n ^ (n >> 13)) * 1274126177u;
            return static_cast<float>((n ^ (n >> 16)) & 0xffffffu) / 16777215.f;
        }

        inline unsigned int upStrHash(const std::string& s) {
            unsigned int h = 2166136261u;
            for (char c : s) h = (h ^ static_cast<unsigned char>(c)) * 16777619u;
            return h;
        }

        inline bool upInsideRing(const std::vector<Vector2>& ring, float x, float z) {
            bool in = false;
            for (size_t i = 0, j = ring.size() - 1; i < ring.size(); j = i++) {
                const Vector2& a = ring[i];
                const Vector2& c = ring[j];
                if ((a.y > z) != (c.y > z) && x < (c.x - a.x) * (z - a.y) / (c.y - a.y) + a.x)
                    in = !in;
            }
            return in;
        }

        // Vertex-coloured triangle soup. Non-indexed and flat-shaded on purpose:
        // a car body wants hard edges, and sharing vertices across a box corner
        // would need split normals anyway.
        struct PropBuf {
            std::vector<float> pos, nrm, col, uv;

            void tri(const Vector3& a, const Vector3& b, const Vector3& c,
                     const std::array<float, 3>& k) {
                const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
                const float vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
                float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
                const float l = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (l > 1e-12f) {
                    nx /= l;
                    ny /= l;
                    nz /= l;
                } else {
                    nx = 0.f;
                    ny = 1.f;
                    nz = 0.f;
                }
                pos.insert(pos.end(), {a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z});
                for (int i = 0; i < 3; ++i) {
                    nrm.insert(nrm.end(), {nx, ny, nz});
                    col.insert(col.end(), {k[0], k[1], k[2]});
                }
                uv.insert(uv.end(), {0.f, 0.f, 1.f, 0.f, 0.5f, 1.f});
            }
            void quad(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d,
                      const std::array<float, 3>& k) {
                tri(a, b, c, k);
                tri(a, c, d, k);
            }
            [[nodiscard]] bool empty() const { return pos.empty(); }
            [[nodiscard]] size_t tris() const { return pos.size() / 9; }
        };

        // Extrude a plan polygon (local: +u forward, +v lateral) between two
        // heights at (ox, oz) heading (fx, fz). The ring is normalised to
        // NEGATIVE shoelace in world (x, z), which is the winding whose top-cap
        // fan faces +Y in this right-handed Y-up frame — get it backwards and
        // every roof is back-face culled and the prop reads as a hole.
        inline void upPrism(PropBuf& buf, const std::vector<Vector2>& plan, float ox, float oz,
                            float fx, float fz, float y0, float y1,
                            const std::array<float, 3>& top, const std::array<float, 3>& side) {
            if (plan.size() < 3) return;
            const float lx = -fz, lz = fx;
            std::vector<Vector2> w;
            w.reserve(plan.size());
            for (const auto& p : plan)
                w.emplace_back(ox + p.x * fx + p.y * lx, oz + p.x * fz + p.y * lz);
            float sh = 0.f;
            for (size_t i = 0, n = w.size(); i < n; ++i) {
                const Vector2& a = w[i];
                const Vector2& b = w[(i + 1) % n];
                sh += a.x * b.y - b.x * a.y;
            }
            if (sh > 0.f) std::reverse(w.begin(), w.end());
            const auto T = [&](size_t i) { return Vector3(w[i].x, y1, w[i].y); };
            const auto B = [&](size_t i) { return Vector3(w[i].x, y0, w[i].y); };
            for (size_t i = 1; i + 1 < w.size(); ++i) buf.tri(T(0), T(i), T(i + 1), top);
            for (size_t i = 0, n = w.size(); i < n; ++i) {
                const size_t j = (i + 1) % n;
                buf.quad(T(i), B(i), B(j), T(j), side);
            }
        }

        // Axis-aligned-to-heading box: the same prism with a 4-point plan.
        inline void upBox(PropBuf& buf, float ox, float oz, float fx, float fz, float halfLen,
                          float halfWid, float y0, float y1, const std::array<float, 3>& top,
                          const std::array<float, 3>& side) {
            const std::vector<Vector2> plan{{halfLen, halfWid},
                                            {halfLen, -halfWid},
                                            {-halfLen, -halfWid},
                                            {-halfLen, halfWid}};
            upPrism(buf, plan, ox, oz, fx, fz, y0, y1, top, side);
        }

        // Wheel: a short cylinder whose axis is the vehicle's lateral direction.
        inline void upWheel(PropBuf& buf, const Vector3& c, float ax, float az, float radius,
                            float halfLen, int sides, const std::array<float, 3>& k) {
            const Vector3 u(0.f, 1.f, 0.f);           // up
            const Vector3 v(-az, 0.f, ax);            // forward (axis rotated 90 deg)
            const Vector3 cp(c.x + ax * halfLen, c.y, c.z + az * halfLen);
            const Vector3 cm(c.x - ax * halfLen, c.y, c.z - az * halfLen);
            const auto rim = [&](int i, bool plus) {
                const float t = 6.28318531f * static_cast<float>(i) / static_cast<float>(sides);
                const float ct = std::cos(t) * radius, st = std::sin(t) * radius;
                const Vector3& o = plus ? cp : cm;
                return Vector3(o.x + u.x * ct + v.x * st, o.y + u.y * ct + v.y * st,
                               o.z + u.z * ct + v.z * st);
            };
            for (int i = 0; i < sides; ++i) {
                const int j = (i + 1) % sides;
                buf.quad(rim(i, true), rim(i, false), rim(j, false), rim(j, true), k);
                buf.tri(cp, rim(i, true), rim(j, true), k);
                buf.tri(cm, rim(j, false), rim(i, false), k);
            }
        }

        // ~116 triangles: body, cabin with dark glass, four 6-gon wheels.
        // 4.4 x 1.8 x 1.45 m, wheels on the ground at `y`.
        inline void upCar(PropBuf& buf, float x, float y, float z, float fx, float fz,
                          const std::array<float, 3>& body) {
            const std::array<float, 3> glass{0.035f, 0.041f, 0.050f};
            const std::array<float, 3> tyre{0.028f, 0.028f, 0.030f};
            const std::array<float, 3> cabinTop{body[0] * 0.92f, body[1] * 0.92f, body[2] * 0.92f};
            upBox(buf, x, z, fx, fz, 2.20f, 0.90f, y + 0.30f, y + 0.95f, body, body);
            upBox(buf, x - fx * 0.15f, z - fz * 0.15f, fx, fz, 1.10f, 0.80f, y + 0.95f, y + 1.45f,
                  cabinTop, glass);
            const float lx = -fz, lz = fx;
            for (int s = 0; s < 4; ++s) {
                const float along = (s < 2) ? 1.45f : -1.45f;
                const float lat = (s % 2 == 0) ? 0.82f : -0.82f;
                upWheel(buf,
                        Vector3(x + fx * along + lx * lat, y + 0.32f, z + fz * along + lz * lat),
                        lx, lz, 0.32f, 0.11f, 6, tyre);
            }
        }

        // ~30 triangles: a 6 m open motorboat, hull + wheelhouse.
        inline void upBoat(PropBuf& buf, float x, float y, float z, float fx, float fz,
                           const std::array<float, 3>& hull) {
            const std::vector<Vector2> plan{{3.00f, 0.00f},  {2.10f, 0.90f},  {-2.40f, 1.05f},
                                            {-3.00f, 0.80f}, {-3.00f, -0.80f}, {-2.40f, -1.05f},
                                            {2.10f, -0.90f}};
            const std::array<float, 3> deck{hull[0] * 0.85f + 0.10f, hull[1] * 0.85f + 0.10f,
                                            hull[2] * 0.85f + 0.10f};
            const std::array<float, 3> glass{0.045f, 0.055f, 0.065f};
            upPrism(buf, plan, x, z, fx, fz, y - 0.35f, y + 0.55f, deck, hull);
            upBox(buf, x - fx * 0.40f, z - fz * 0.40f, fx, fz, 0.90f, 0.70f, y + 0.55f, y + 1.25f,
                  deck, glass);
        }

        // Mitred quad strip along a polyline: L forward then R backward is
        // already the negative-shoelace ring, so the top cap is a plain quad
        // strip (no triangulator) and the sides follow the ring edges.
        inline void upDeckStrip(PropBuf& buf, const std::vector<Vector2>& pts, float halfWidth,
                                float y0, float y1, const std::array<float, 3>& top,
                                const std::array<float, 3>& side) {
            const size_t n = pts.size();
            if (n < 2) return;
            std::vector<Vector2> L(n), R(n);
            for (size_t i = 0; i < n; ++i) {
                float dx = 0.f, dz = 0.f;
                if (i > 0) {
                    const float ax = pts[i].x - pts[i - 1].x, az = pts[i].y - pts[i - 1].y;
                    const float l = std::sqrt(ax * ax + az * az);
                    if (l > 1e-5f) {
                        dx += ax / l;
                        dz += az / l;
                    }
                }
                if (i + 1 < n) {
                    const float ax = pts[i + 1].x - pts[i].x, az = pts[i + 1].y - pts[i].y;
                    const float l = std::sqrt(ax * ax + az * az);
                    if (l > 1e-5f) {
                        dx += ax / l;
                        dz += az / l;
                    }
                }
                const float l = std::sqrt(dx * dx + dz * dz);
                if (l < 1e-5f) {
                    dx = 1.f;
                    dz = 0.f;
                } else {
                    dx /= l;
                    dz /= l;
                }
                // Averaged tangent, NOT a true miter (which divides by the
                // half-angle cosine): on a near-straight pier the two agree to
                // millimetres, and on a hairpin the true miter shoots a spike
                // across the harbour while this only pinches the corner.
                const float nx = -dz, nz = dx;
                L[i].set(pts[i].x + nx * halfWidth, pts[i].y + nz * halfWidth);
                R[i].set(pts[i].x - nx * halfWidth, pts[i].y - nz * halfWidth);
            }
            for (size_t i = 0; i + 1 < n; ++i) {
                const size_t j = i + 1;
                buf.quad(Vector3(L[i].x, y1, L[i].y), Vector3(L[j].x, y1, L[j].y),
                         Vector3(R[j].x, y1, R[j].y), Vector3(R[i].x, y1, R[i].y), top);
                // Left flank, right flank (fender edge colour).
                buf.quad(Vector3(L[i].x, y1, L[i].y), Vector3(L[i].x, y0, L[i].y),
                         Vector3(L[j].x, y0, L[j].y), Vector3(L[j].x, y1, L[j].y), side);
                buf.quad(Vector3(R[j].x, y1, R[j].y), Vector3(R[j].x, y0, R[j].y),
                         Vector3(R[i].x, y0, R[i].y), Vector3(R[i].x, y1, R[i].y), side);
            }
            // End caps.
            buf.quad(Vector3(R[0].x, y1, R[0].y), Vector3(R[0].x, y0, R[0].y),
                     Vector3(L[0].x, y0, L[0].y), Vector3(L[0].x, y1, L[0].y), side);
            const size_t e = n - 1;
            buf.quad(Vector3(L[e].x, y1, L[e].y), Vector3(L[e].x, y0, L[e].y),
                     Vector3(R[e].x, y0, R[e].y), Vector3(R[e].x, y1, R[e].y), side);
        }

        inline std::shared_ptr<Mesh> upMakeMesh(const PropBuf& b,
                                                const std::shared_ptr<Material>& mat,
                                                const std::string& name) {
            auto geo = BufferGeometry::create();
            geo->setAttribute("position", FloatBufferAttribute::create(b.pos, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(b.nrm, 3));
            geo->setAttribute("color", FloatBufferAttribute::create(b.col, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(b.uv, 2));
            geo->computeBoundingSphere();
            auto mesh = Mesh::create(geo, mat);
            mesh->name = name;
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            return mesh;
        }

    }// namespace detail

    // The real body-colour mix on a Norwegian street, in order: white, grey,
    // black, silver, red, blue (30/20/20/10/10/10).
    inline std::array<float, 3> urbanCarColor(float u) {
        if (u < 0.30f) return {0.760f, 0.760f, 0.745f};// white
        if (u < 0.50f) return {0.185f, 0.190f, 0.200f};// dark grey
        if (u < 0.70f) return {0.030f, 0.030f, 0.034f};// black
        if (u < 0.80f) return {0.420f, 0.435f, 0.450f};// silver
        if (u < 0.90f) return {0.330f, 0.038f, 0.036f};// red
        return {0.042f, 0.085f, 0.230f};               // blue
    }

    // Build the props under `root`. Returns what it planted and why it refused.
    inline UrbanPropsStats buildUrbanProps(Object3D& root, const GeoTerrainPack& pack,
                                           const road::RoadNetwork& net,
                                           const UrbanPropsOptions& o) {
        UrbanPropsStats st;
        if (!o.ground) return st;
        const float sea = o.seaLevel;
        const float roi = o.halfExtent;
        const auto inRoi = [&](float x, float z) {
            return std::fabs(x - o.centerX) <= roi && std::fabs(z - o.centerZ) <= roi;
        };

        auto carMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.35f).metalness(0.f));
        carMat->vertexColors = true;
        auto deckMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.88f).metalness(0.f));
        deckMat->vertexColors = true;
        auto boatMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.42f).metalness(0.f));
        boatMat->vertexColors = true;

        // ── pier decks ──────────────────────────────────────────────────────
        // Only over water: a pier line that runs up onto the quay is already
        // ground (the apron raise gave it a surface), and a deck floating a
        // metre over it would be a shelf in mid-air.
        if (o.decks && pack.hasLandUse()) {
            detail::PropBuf deck;
            const std::array<float, 3> top{0.180f, 0.172f, 0.158f};// weathered concrete/timber
            const std::array<float, 3> side{0.055f, 0.050f, 0.045f};// dark fender edge
            const float y1 = sea + o.deckTop;
            const float y0 = y1 - o.deckThickness;
            for (const auto& l : pack.landuse.lines) {
                if (l.cls != "pier" || l.points.size() < 2) continue;
                // Split the line into runs of consecutive over-water vertices.
                std::vector<Vector2> run;
                const auto flush = [&] {
                    if (run.size() >= 2) {
                        detail::upDeckStrip(deck, run,
                                            0.5f * (l.width > 0.f ? l.width : o.deckWidth), y0, y1,
                                            top, side);
                        ++st.deckLines;
                    }
                    run.clear();
                };
                for (const auto& p : l.points) {
                    const bool wet = inRoi(p.x, p.y) && o.ground(p.x, p.y) < sea + 0.5f;
                    if (wet) run.push_back(p);
                    else flush();
                }
                flush();
            }
            if (!deck.empty()) {
                root.add(detail::upMakeMesh(deck, deckMat, "pier_decks"));
                ++st.meshes;
                st.triangles += deck.tris();
            }
        }

        // ── cars ────────────────────────────────────────────────────────────
        if (o.cars) {
            std::map<std::pair<int, int>, detail::PropBuf> cells;
            const auto cellOf = [&](float x, float z) {
                return std::make_pair(static_cast<int>(std::floor(x / o.cellSize)),
                                      static_cast<int>(std::floor(z / o.cellSize)));
            };
            const auto place = [&](float x, float z, float fx, float fz, unsigned int seed) {
                const float g = o.ground(x, z);
                if (g < sea + 0.3f) {
                    ++st.rejectSea;
                    return false;
                }
                if (o.footprints && o.footprints->inside(x, z)) {
                    ++st.rejectRoof;
                    return false;
                }
                detail::upCar(cells[cellOf(x, z)], x, g, z, fx, fz,
                              urbanCarColor(detail::upHash01(seed, 0x51u, 0u)));
                return true;
            };

            // (a) mapped parking lots: a 2.5 m bay pitch along the lot's long
            // axis, 5 m rows across it, cars nose-in (heading = short axis).
            for (const auto& poly : pack.landuse.polygons) {
                if (poly.cls != "parking" || poly.outer.size() < 3) continue;
                Vector2 c;
                for (const auto& p : poly.outer) c.add(p);
                c.divideScalar(static_cast<float>(poly.outer.size()));
                if (!inRoi(c.x, c.y)) continue;
                Vector2 axis(1.f, 0.f);
                float bestLen = -1.f;
                float minU = 1e30f, maxU = -1e30f, minV = 1e30f, maxV = -1e30f;
                for (size_t i = 0, n = poly.outer.size(); i < n; ++i) {
                    const Vector2& a = poly.outer[i];
                    const Vector2& b = poly.outer[(i + 1) % n];
                    const float dx = b.x - a.x, dz = b.y - a.y;
                    const float l = dx * dx + dz * dz;
                    if (l > bestLen) {
                        bestLen = l;
                        axis.set(dx, dz);
                    }
                }
                if (bestLen < 1e-6f) continue;
                axis.divideScalar(std::sqrt(bestLen));
                const Vector2 lat(-axis.y, axis.x);
                for (const auto& p : poly.outer) {
                    const float u = p.x * axis.x + p.y * axis.y;
                    const float v = p.x * lat.x + p.y * lat.y;
                    minU = std::min(minU, u);
                    maxU = std::max(maxU, u);
                    minV = std::min(minV, v);
                    maxV = std::max(maxV, v);
                }
                const unsigned int lotSeed = detail::upStrHash(poly.id);
                int iu = 0;
                for (float u = minU + 1.25f; u <= maxU; u += o.lotBayPitch, ++iu) {
                    int iv = 0;
                    for (float v = minV + 2.5f; v <= maxV; v += o.lotRowPitch, ++iv) {
                        const float x = axis.x * u + lat.x * v;
                        const float z = axis.y * u + lat.y * v;
                        if (!detail::upInsideRing(poly.outer, x, z)) continue;
                        if (detail::upHash01(lotSeed, static_cast<unsigned int>(iu),
                                             static_cast<unsigned int>(iv)) > o.lotOccupancy)
                            continue;
                        if (net.pavedWeight(x, z, 1.f) > 0.25f) {
                            ++st.rejectPaved;
                            continue;
                        }
                        // Nose-in: the car points along the SHORT axis.
                        if (place(x, z, lat.x, lat.y,
                                  lotSeed ^ (static_cast<unsigned int>(iu) << 8) ^
                                          static_cast<unsigned int>(iv)))
                            ++st.carsLot;
                    }
                }
            }

            // (b) kerb parking on K and P roads inside the town. Junction nodes
            // (a centreline endpoint that another road's centreline touches) are
            // kept clear: a car parked across a side street is the one placement
            // error a viewer reads instantly.
            std::vector<Vector2> junctions;
            {
                const auto infos = net.roadInfos();
                std::vector<std::vector<Vector3>> cls;
                cls.reserve(infos.size());
                for (const auto& i : infos) cls.push_back(net.roadCenterline(i.id));
                for (size_t a = 0; a < cls.size(); ++a) {
                    if (cls[a].size() < 2) continue;
                    for (int e = 0; e < 2; ++e) {
                        const Vector3& p = e ? cls[a].back() : cls[a].front();
                        if (!inRoi(p.x, p.z)) continue;
                        bool shared = false;
                        for (size_t b = 0; b < cls.size() && !shared; ++b) {
                            if (b == a) continue;
                            for (const auto& q : cls[b])
                                if (std::fabs(q.x - p.x) < 6.f && std::fabs(q.z - p.z) < 6.f) {
                                    shared = true;
                                    break;
                                }
                        }
                        if (shared) junctions.emplace_back(p.x, p.z);
                    }
                }
                for (size_t a = 0; a < cls.size(); ++a) {
                    const auto& info = infos[a];
                    if (info.category != "K" && info.category != "P") continue;
                    const auto& cl = cls[a];
                    if (cl.size() < 2) continue;
                    const unsigned int rs = detail::upStrHash(info.id);
                    const float sideSign = (rs & 1u) ? 1.f : -1.f;
                    const float off = 0.5f * info.width + o.kerbOffset;
                    float acc = 0.f;
                    unsigned int idx = 0;
                    for (size_t k = 1; k < cl.size(); ++k) {
                        const float dx = cl[k].x - cl[k - 1].x, dz = cl[k].z - cl[k - 1].z;
                        const float len = std::sqrt(dx * dx + dz * dz);
                        if (len < 1e-4f) continue;
                        const float ux = dx / len, uz = dz / len;
                        float pos = 0.f;
                        while (acc + (len - pos) >= o.kerbPitch) {
                            pos += o.kerbPitch - acc;
                            acc = 0.f;
                            ++idx;
                            const float cx = cl[k - 1].x + ux * pos, cz = cl[k - 1].z + uz * pos;
                            if (!inRoi(cx, cz)) continue;
                            if (detail::upHash01(rs, idx, 7u) > o.kerbOccupancy) continue;
                            bool nearJ = false;
                            for (const auto& j : junctions) {
                                const float jx = j.x - cx, jz = j.y - cz;
                                if (jx * jx + jz * jz < o.junctionClear * o.junctionClear) {
                                    nearJ = true;
                                    break;
                                }
                            }
                            if (nearJ) {
                                ++st.rejectJunction;
                                continue;
                            }
                            const float px = cx - uz * off * sideSign;
                            const float pz = cz + ux * off * sideSign;
                            if (o.urban && o.urban->sample(px, pz) < o.urbanMin) continue;
                            if (place(px, pz, ux, uz, rs ^ (idx * 2654435761u))) ++st.carsKerb;
                        }
                        acc += len - pos;
                    }
                }
            }

            for (const auto& [key, buf] : cells) {
                if (buf.empty()) continue;
                root.add(detail::upMakeMesh(buf, carMat,
                                            "cars_" + std::to_string(key.first) + "_" +
                                                    std::to_string(key.second)));
                ++st.meshes;
                ++st.carCells;
                st.triangles += buf.tris();
            }
        }

        // ── boats ───────────────────────────────────────────────────────────
        // Along the pier lines that lie in (or beside) a mapped marina, moored
        // alternately to either side so the pier reads as a berth, not a road
        // with boats driving down it.
        if (o.boats && pack.hasLandUse()) {
            std::vector<const GeoLandUse::Polygon*> marinas;
            for (const auto& poly : pack.landuse.polygons)
                if (poly.cls == "marina" && poly.outer.size() >= 3) marinas.push_back(&poly);
            const std::array<std::array<float, 3>, 3> hulls = {{{0.720f, 0.725f, 0.720f},// white
                                                                {0.300f, 0.315f, 0.330f},// grey
                                                                {0.055f, 0.115f, 0.250f}}};// blue
            for (size_t mi = 0; mi < marinas.size(); ++mi) {
                const auto& mp = *marinas[mi];
                detail::PropBuf buf;
                int planted = 0;
                for (const auto& l : pack.landuse.lines) {
                    if (l.cls != "pier" || l.points.size() < 2) continue;
                    bool belongs = false;
                    for (const auto& p : l.points) {
                        if (detail::upInsideRing(mp.outer, p.x, p.y)) {
                            belongs = true;
                            break;
                        }
                        for (const auto& q : mp.outer)
                            if (std::fabs(q.x - p.x) < o.marinaRadius &&
                                std::fabs(q.y - p.y) < o.marinaRadius) {
                                belongs = true;
                                break;
                            }
                        if (belongs) break;
                    }
                    if (!belongs) continue;
                    const unsigned int ls = detail::upStrHash(l.id) ^ 0xB0A7u;
                    const float off = 0.5f * (l.width > 0.f ? l.width : o.deckWidth) + 1.6f;
                    float acc = 0.f;
                    unsigned int idx = 0;
                    for (size_t k = 1; k < l.points.size(); ++k) {
                        const float dx = l.points[k].x - l.points[k - 1].x;
                        const float dz = l.points[k].y - l.points[k - 1].y;
                        const float len = std::sqrt(dx * dx + dz * dz);
                        if (len < 1e-4f) continue;
                        const float ux = dx / len, uz = dz / len;
                        float pos = 0.f;
                        while (acc + (len - pos) >= o.boatPitch) {
                            pos += o.boatPitch - acc;
                            acc = 0.f;
                            ++idx;
                            const float cx = l.points[k - 1].x + ux * pos;
                            const float cz = l.points[k - 1].y + uz * pos;
                            if (!inRoi(cx, cz)) continue;
                            if (detail::upHash01(ls, idx, 3u) > o.boatOccupancy) continue;
                            const float sgn = (idx & 1u) ? 1.f : -1.f;
                            const float bx = cx - uz * off * sgn, bz = cz + ux * off * sgn;
                            if (o.ground(bx, bz) > sea + 0.4f) continue;// not water: no berth
                            const size_t ci = static_cast<size_t>(
                                    detail::upHash01(ls, idx, 11u) * 2.999f);
                            detail::upBoat(buf, bx, sea + 0.30f, bz, ux, uz,
                                           hulls[std::min<size_t>(ci, 2)]);
                            ++planted;
                        }
                        acc += len - pos;
                    }
                }
                if (planted > 0 && !buf.empty()) {
                    root.add(detail::upMakeMesh(buf, boatMat, "boats_" + std::to_string(mi)));
                    ++st.meshes;
                    ++st.marinas;
                    st.boats += planted;
                    st.triangles += buf.tris();
                }
            }
        }

        return st;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_URBANPROPS_HPP
