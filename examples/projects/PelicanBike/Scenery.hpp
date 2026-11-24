// The setting: a cycle path along a sandy coast. The path is asphalt with a worn dashed centre line and edge lines,
// a low kerb on the near side; beyond the far edge, dunes with marram grass fall away to the sea. Every texture is
// painted here per pixel.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_SCENERY_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_SCENERY_HPP

#include "Parts.hpp"
#include "Surfaces.hpp"

namespace pb {

    using mill::fbm2;
    using mill::fbm3;
    using mill::hash2f;
    using mill::worley2;

    constexpr float roadHalf = 2.2f;   // the path is 4.4 m wide

    // ---- the asphalt: 4.4 m square, markings centred (u across the path, v along it) ----------------------------------
    inline TexSet asphaltTexture(uint32_t seed) {
        const int N = 2048;
        const float size = 4.4f;
        Maps m(N, N, size, size);
        mill::forRows(N, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
                const size_t i = m.at(x, y);
                // the surface: a fine aggregate in a dark binder
                const auto stone = worley2(u * 190.f, v * 190.f, seed, 190, 190, 0.95f);
                const auto stone2 = worley2(u * 70.f, v * 70.f, seed + 1, 70, 70, 0.9f);
                const float tone = 0.5f + 0.5f * fbm2(u * 9.f, v * 9.f, 5, seed + 2, 9, 9);
                const float blotch = 0.5f + 0.5f * fbm2(u * 3.f, v * 3.f, 4, seed + 3, 3, 3);
                const float sh = mill::unit(stone.id), sh2 = mill::unit(stone2.id);
                const float chip = smooth(0.34f, 0.12f, stone.f1) * (0.5f + 0.5f * sh);
                const float chip2 = smooth(0.30f, 0.10f, stone2.f1) * (0.4f + 0.6f * sh2);
                V3 c = mill::mix(srgb(58, 58, 62), srgb(86, 85, 86), tone * 0.8f + 0.2f * blotch);
                c = mill::mix(c, srgb(138, 132, 122), chip * 0.55f);
                c = mill::mix(c, srgb(110, 104, 98), chip2 * 0.4f);
                // the wheel tracks: polished darker bands, and a little oil in the middle of the lane
                const float track = std::exp(-std::pow((std::abs(u - 0.5f) - 0.22f) * 4.4f / 0.28f, 2.f));
                c = c * (1.f - 0.18f * track);
                const float oil = smooth(0.62f, 0.8f, 0.5f + 0.5f * fbm2(u * 6.f, v * 6.f, 4, seed + 4, 6, 6)) * track;
                c = c * (1.f - 0.25f * oil);
                // sand blown across the edges
                const float distEdge = (roadHalf - std::abs(u - 0.5f) * size);
                const float sand = smooth(0.7f, 0.0f, distEdge + 0.25f * (fbm2(u * 22.f, v * 22.f, 3, seed + 5, 22, 22))) * (0.4f + 0.6f * blotch);
                c = mill::mix(c, srgb(196, 174, 130), sand * 0.55f);
                // cracks
                const auto cr = worley2(u * 5.f, v * 5.f, seed + 6, 5, 5, 1.f);
                const float crackMask = smooth(0.45f, 0.75f, 0.5f + 0.5f * fbm2(u * 4.f, v * 4.f, 3, seed + 7, 4, 4));
                const float crack = smooth(0.012f, 0.0f, cr.f2 - cr.f1 + 0.008f * fbm2(u * 120.f, v * 120.f, 2, seed + 8, 120, 120)) * crackMask;
                c = c * (1.f - 0.6f * crack);
                float h = 0.0009f * (chip + 0.7f * chip2) - 0.0012f * crack;
                float rough = 0.9f - 0.12f * track - 0.3f * oil + 0.05f * chip;
                // markings: a dashed centre line and solid edge lines, worn
                const float across = std::abs(u - 0.5f) * size;
                const bool dash = std::fmod(v * 2.f + 0.5f, 1.f) < 0.5f;
                const float lineC = (across < 0.05f && dash) ? 1.f : 0.f;
                const float lineE = (std::abs(across - (roadHalf - 0.22f)) < 0.045f) ? 1.f : 0.f;
                const float wear = smooth(0.30f, 0.62f, 0.5f + 0.5f * fbm2(u * 40.f, v * 40.f, 4, seed + 9, 40, 40) + 0.35f * (chip - 0.5f));
                const float paint = std::max(lineC, lineE) * wear;
                c = mill::mix(c, srgb(222, 220, 208), paint * 0.9f);
                rough = mill::lerpf(rough, 0.7f, paint);
                h += 0.0004f * paint;
                m.albedo[i] = c;
                m.height[i] = h;
                m.rough[i] = rough;
                m.cavity[i] = 0.75f + 0.25f * (1.f - crack) + 0.1f * chip;
            }
        });
        return m.finish(1.0f);
    }

    // sand with dry grass in it, 3 m a tile
    inline TexSet sandTexture(uint32_t seed) {
        const int N = 1024;
        Maps m(N, N, 3.f, 3.f);
        mill::forRows(N, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
                const size_t i = m.at(x, y);
                const auto grain = worley2(u * 260.f, v * 260.f, seed, 260, 260, 1.f);
                const float ripple = 0.5f + 0.5f * std::sin((v * 6.f + 0.6f * fbm2(u * 5.f, v * 5.f, 3, seed + 1, 5, 5)) * 6.2831853f * 3.f);
                const float tone = 0.5f + 0.5f * fbm2(u * 6.f, v * 6.f, 5, seed + 2, 6, 6);
                const float sh = mill::unit(grain.id);
                V3 c = mill::mix(srgb(188, 165, 120), srgb(224, 204, 156), tone * 0.7f + 0.3f * sh);
                // tufts of dry grass and flecks of shell and pebble
                const float turf = smooth(0.55f, 0.75f, 0.5f + 0.5f * fbm2(u * 7.f, v * 7.f, 4, seed + 3, 7, 7));
                c = mill::mix(c, srgb(128, 128, 70), turf * 0.7f);
                const auto peb = worley2(u * 40.f, v * 40.f, seed + 4, 40, 40, 1.f);
                const float pebble = smooth(0.2f, 0.08f, peb.f1) * (mill::unit(peb.id) > 0.82f ? 1.f : 0.f);
                c = mill::mix(c, srgb(150, 140, 126), pebble);
                m.albedo[i] = c;
                m.height[i] = 0.0016f * ripple * (1.f - turf) + 0.0008f * (1.f - grain.f1) + 0.003f * pebble;
                m.rough[i] = 0.95f;
                m.cavity[i] = 0.85f + 0.15f * (1.f - turf);
            }
        });
        return m.finish(1.0f);
    }

    // ---- a tuft of marram grass: blades that arch over -------------------------------------------------------------------
    inline std::shared_ptr<BufferGeometry> grassTuft(uint32_t seed) {
        Rng rng(seed);
        MeshBuild mb;
        const int blades = 22;
        for (int b = 0; b < blades; ++b) {
            const float yaw = rng.range(0.f, TAU);
            const float lean = rng.range(0.05f, 0.9f);
            const float height = rng.range(0.28f, 0.62f);
            const float wid = rng.range(0.008f, 0.014f);
            const float rootR = rng.range(0.0f, 0.05f);
            const Vector3 root(std::cos(yaw) * rootR, 0.f, std::sin(yaw) * rootR);
            const Vector3 dir(std::cos(yaw), 0.f, std::sin(yaw));
            const Vector3 side(-dir.z, 0.f, dir.x);
            const int seg = 7;
            const unsigned base = mb.vertexCount();
            for (int k = 0; k <= seg; ++k) {
                const float t = static_cast<float>(k) / static_cast<float>(seg);
                // arching: rise, then lean over
                const Vector3 p = root + dir * (height * lean * t * t * 0.9f) + Vector3(0.f, height * (t - 0.3f * lean * t * t), 0.f);
                const float w = wid * (1.f - 0.92f * std::pow(t, 1.6f));
                const V3 colour = mill::mix(srgb(120, 118, 60), srgb(176, 168, 96), t);
                Vector3 n = Vector3(0, 0, 0) - dir;
                n.y = 0.4f;
                n.normalize();
                mb.vertex(p - side * w, n, 0.f, t, colour);
                mb.vertex(p + side * w, n, 1.f, t, colour);
            }
            for (int k = 0; k < seg; ++k) {
                const unsigned a = base + 2 * k, b2 = a + 1, c = a + 2, d = a + 3;
                mb.index.insert(mb.index.end(), {a, b2, c, b2, d, c});
            }
        }
        auto mesh = mb.mesh(standard(Color(1, 1, 1), 0.7f), true);
        return mesh->geometry();
    }

    struct Scenery {
        std::shared_ptr<Group> root = Group::create();
        std::shared_ptr<Group> roadGroup = Group::create(), groundGroup = Group::create(), grassGroup = Group::create();
        TexSet road, sand;

        // the world is made to repeat: each part follows the rider in steps of its own period
        void follow(float x) const {
            roadGroup->position.x = std::round(x / 4.4f) * 4.4f;
            groundGroup->position.x = std::round(x / 240.f) * 240.f;
            grassGroup->position.x = std::round(x / 80.f) * 80.f;
        }
        std::shared_ptr<MeshStandardMaterial> roadMat, sandMat, seaMat, grassMat, kerbMat, pebbleMat;

        // height of the dunes and beach at (x, z), z negative away from the path
        static float duneHeight(float x, float z) {
            if (z > -2.6f) return -0.02f;
            const float d = -z - 2.6f;
            const float rise = smooth(0.f, 6.f, d);
            const float fall = smooth(12.f, 55.f, d);
            const float rolling = 0.5f + 0.5f * fbm2(x * 0.025f, z * 0.05f, 5, 41, 2, 0);
            const float tops = 0.55f * rise * rolling * (1.f - fall) + 0.25f * rise * (1.f - fall) * (0.5f + 0.5f * fbm2(x * 0.1f, z * 0.1f, 3, 43, 8, 0));
            return tops - 2.6f * fall + 0.1f * fbm2(x * 0.5f, z * 0.5f, 2, 44, 40, 0);
        }

        Scenery() {
            road = asphaltTexture(5);
            sand = sandTexture(7);
            roadMat = road.material(1.f);
            roadMat->normalScale.set(0.8f, 0.8f);
            sandMat = sand.material(0.8f);
            roadMat->map->wrapS = roadMat->map->wrapT = TextureWrapping::Repeat;

            // the path: one long strip, the texture repeating every 4.4 m along it
            {
                const float length = 520.f;
                MeshBuild mb;
                const int nx = static_cast<int>(length / 2.2f);
                for (int i = 0; i <= nx; ++i) {
                    const float x = -length * 0.5f + length * static_cast<float>(i) / static_cast<float>(nx);
                    for (int k = 0; k <= 1; ++k) {
                        const float z = (k == 0 ? -roadHalf : roadHalf);
                        mb.vertex(Vector3(x, 0.f, z), Vector3(0, 1, 0), (k == 0 ? 0.f : 1.f), x / 4.4f, V3(1, 1, 1));
                    }
                }
                for (int i = 0; i < nx; ++i) {
                    const unsigned a = 2 * i, b = a + 1, c = a + 2, d = a + 3;
                    // u runs with z, v with x: keep the winding so the normal is +y
                    mb.index.insert(mb.index.end(), {a, b, c, b, d, c});
                }
                auto mesh = mb.mesh(roadMat, false);
                mesh->castShadow = false;
                roadGroup->add(mesh);
            }
            // the near kerb: a low concrete edge, and a strip of sand and grass beyond it
            {
                kerbMat = standard(lin(150, 148, 142), 0.9f);
                auto kerb = Mesh::create(BoxGeometry::create(520.f, 0.1f, 0.2f), kerbMat);
                kerb->position.set(0.f, 0.05f, roadHalf + 0.1f);
                kerb->castShadow = true;
                kerb->receiveShadow = true;
                roadGroup->add(kerb);
            }
            // the ground beside the path: a grid of sand, dunes on the far side
            {
                MeshBuild mb;
                const float x0 = -260.f, x1 = 260.f;
                const int nx = 260;
                // z from the near side outward: rows spaced finer near the path
                std::vector<float> zs;
                for (float z = roadHalf + 0.2f + 18.f; z > roadHalf + 0.2f; z -= 1.5f) zs.push_back(z);
                zs.push_back(roadHalf + 0.2f);
                zs.push_back(-roadHalf);
                for (float z = -roadHalf - 0.6f; z > -90.f; z -= (z > -20.f ? 0.9f : 3.2f)) zs.push_back(z);
                for (size_t r = 0; r < zs.size(); ++r) {
                    for (int i = 0; i <= nx; ++i) {
                        const float x = x0 + (x1 - x0) * static_cast<float>(i) / static_cast<float>(nx);
                        const float z = zs[r];
                        float y = z < -roadHalf ? duneHeight(x, z) : (z > roadHalf + 0.2f ? 0.02f + 0.04f * fbm2(x * 0.3f, z * 0.3f, 3, 45, 24, 0) * smooth(roadHalf + 0.5f, roadHalf + 4.f, z) : 0.f);
                        // the shoulder sits a little under the asphalt
                        if (std::abs(z) < roadHalf + 0.25f && z <= roadHalf + 0.2f && z >= -roadHalf) y = -0.01f;
                        mb.vertex(Vector3(x, y, z), Vector3(0, 1, 0), x / 3.f, z / 3.f, V3(1, 1, 1));
                    }
                }
                const unsigned row = static_cast<unsigned>(nx + 1);
                for (size_t r = 0; r + 1 < zs.size(); ++r)
                    for (int i = 0; i < nx; ++i) {
                        const unsigned a = static_cast<unsigned>(r) * row + i, b = a + 1, c = a + row, d = c + 1;
                        mb.index.insert(mb.index.end(), {a, b, c, b, d, c});
                    }
                mb.smoothNormals();
                // the near verge is trampled sand; the far dunes get the same texture, greener with height
                auto mesh = mb.mesh(sandMat, false);
                mesh->castShadow = false;
                mesh->receiveShadow = true;
                groundGroup->add(mesh);
            }
            root->add(roadGroup);
            root->add(groundGroup);
            root->add(grassGroup);
            // the sea
            {
                seaMat = standard(lin(34, 88, 100), 0.12f);
                auto sea = Mesh::create(PlaneGeometry::create(30000.f, 30000.f), seaMat);
                sea->rotation.x = -PI / 2.f;
                sea->position.set(0.f, -1.7f, -9000.f);
                sea->receiveShadow = false;
                root->add(sea);
            }
            // marram grass on the dunes and along the path's far edge, as instances of one tuft
            {
                grassMat = standard(Color(1, 1, 1), 0.75f);
                grassMat->vertexColors = true;
                grassMat->side = Side::Double;
                grassMat->alphaTest = 0.0f;
                std::vector<Matrix4> ms;
                std::vector<float> rgb;
                Rng rng(91);
                const int clumps = 800;   // in 80 m, repeated at -80, 0 and +80
                for (int c = 0; c < clumps; ++c) {
                    const float cx = rng.range(-40.f, 40.f);
                    // denser near the path's far edge, thinning into the dunes
                    const float d = std::pow(rng(), 1.6f) * 30.f;
                    const float cz = -roadHalf - 0.5f - d;
                    const int n = 1 + static_cast<int>(rng() * rng() * 9.f);
                    const float base = rng.range(0.5f, 1.2f) * (1.f - 0.3f * d / 30.f);
                    for (int j = 0; j < n; ++j) {
                        const float x0 = cx + 0.3f * rng.normal();
                        const float z = std::min(cz + 0.28f * rng.normal(), -roadHalf - 0.4f);
                        const float s = base * rng.range(0.7f, 1.3f);
                        const float sy = s * rng.range(0.8f, 1.35f);
                        const float yaw = rng.range(0.f, TAU);
                        const float dry = rng();
                        for (int k = -1; k <= 1; ++k) {
                            const float x = x0 + 80.f * static_cast<float>(k);
                            Matrix4 m;
                            Quaternion q;
                            q.setFromAxisAngle(Vector3(0, 1, 0), yaw);
                            m.compose(Vector3(x, duneHeight(x, z) - 0.02f, z), q, Vector3(s, sy, s));
                            ms.push_back(m);
                            rgb.insert(rgb.end(), {0.85f + 0.15f * dry, 0.85f + 0.12f * dry, 0.8f});
                        }
                    }
                }
                auto inst = InstancedMesh::create(grassTuft(13), grassMat, ms.size());
                for (size_t i = 0; i < ms.size(); ++i) inst->setMatrixAt(i, ms[i]);
                inst->setColors(0, ms.size(), rgb.data());
                inst->castShadow = true;
                inst->receiveShadow = true;
                inst->frustumCulled = false;
                grassGroup->add(inst);
            }
        }
    };

}// namespace pb

#endif
