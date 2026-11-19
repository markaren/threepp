// A sluice dam on a forest stream. A brook fills a mill pond held by a log crib dam; open the sluice gate and the
// pond pushes a jet out under it, a surge runs down the boulder rapid, over the rock ledge and into the lake below,
// and the pond drains. Shut it and the pond rises until it spills over the dam's top.
//
// The water is simulated (ShallowWater.hpp: the shallow-water equations on a 15 cm grid, stepped every frame) and
// drawn as one mesh whose vertices follow it. Its shader drags the ripple texture along the simulated current, so the
// ripples run with the water, and the simulation's foam decides where the water turns white. The ground and the
// water share one grid, so the water's edge is where its level meets the ground.
//
//   left mouse     orbit        right mouse: pan        wheel: closer / further
//   G              open / shut the gate
//   Up / Down      the gate 10 cm up / down
//   Space          pause        R: start over
//
// Headless (dev):
//   SluiceDam --shot out.png [--time T] [--open T] [--view dam|overview|falls|pond|rapid] [--size WxH]
//   (--open raises the gate fully at that time; the shot is taken at --time seconds of water)

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/textures/DataTexture.hpp"

#include "Course.hpp"
#include "ShallowWater.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <tuple>
#include <map>
#include <random>
#include <string>

using namespace threepp;
using namespace sluice;

namespace {

    // a colour given in sRGB, as a linear Color
    Color srgb(int r, int g, int b) {
        const auto lin = [](int c) { return std::pow(static_cast<float>(c) / 255.f, 2.2f); };
        return {lin(r), lin(g), lin(b)};
    }

    template<class T>
    std::shared_ptr<T> shadowed(std::shared_ptr<T> object) {
        object->castShadow = true;
        object->receiveShadow = true;
        return object;
    }

    // ---- the ground --------------------------------------------------------------------------------------------------
    // the tile's terrain on the water's grid (vertices at the cells' centres), coloured by what it is: moss and
    // grass, dark earth on the steep banks, stones in the stream's bed, mud under the pond and the lake
    std::shared_ptr<Mesh> makeTerrain(const Course& course, const std::vector<float>& ground) {
        const int count = columns * rows;
        std::vector<float> positions(count * 3), normals(count * 3), colors(count * 3);
        std::mt19937 rng(3);
        std::uniform_real_distribution<float> jitter(-1, 1);
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < columns; ++i) {
                const int c = j * columns + i;
                const float x = tileX0 + (static_cast<float>(i) + 0.5f) * cellSize;
                const float z = tileZ0 + (static_cast<float>(j) + 0.5f) * cellSize;
                positions[3 * c] = x;
                positions[3 * c + 1] = ground[c];
                positions[3 * c + 2] = z;
                const int il = std::max(i - 1, 0), ir = std::min(i + 1, columns - 1);
                const int jb = std::max(j - 1, 0), jt = std::min(j + 1, rows - 1);
                const float gx = (ground[j * columns + ir] - ground[j * columns + il]) / (static_cast<float>(ir - il) * cellSize);
                const float gz = (ground[jt * columns + i] - ground[jb * columns + i]) / (static_cast<float>(jt - jb) * cellSize);
                Vector3 n(-gx, 1, -gz);
                n.normalize();
                normals[3 * c] = n.x;
                normals[3 * c + 1] = n.y;
                normals[3 * c + 2] = n.z;

                // the colour: a mottle of moss and grass, earth where it is steep, wet stones and mud in the water
                const float mottle = 0.5f + 0.25f * std::sin(0.9f * x + 0.4f) * std::sin(0.7f * z) + 0.15f * std::sin(2.3f * x - 1.9f * z) + 0.1f * jitter(rng);
                Color col = srgb(70, 84, 38).lerp(srgb(104, 112, 52), std::clamp(mottle, 0.f, 1.f));
                const float steep = smoothstep(0.85f, 0.6f, n.y);
                col.lerp(srgb(74, 60, 44), 0.75f * steep);
                if (course.wet(x, z, -0.15f)) {
                    const float deep = smoothstep(-0.3f, -1.5f, std::min(Course::pondSd(x, z), Course::lakeSd(x, z)));
                    Color bed = srgb(72, 66, 58).lerp(srgb(48, 41, 33), deep);
                    bed.multiplyScalar(0.85f + 0.3f * mottle);
                    col = bed;
                } else if (course.wet(x, z, 0.5f)) {
                    col.lerp(srgb(52, 58, 30), 0.6f);// the damp margin
                }
                colors[3 * c] = col.r;
                colors[3 * c + 1] = col.g;
                colors[3 * c + 2] = col.b;
            }
        }
        std::vector<unsigned int> index;
        index.reserve((columns - 1) * (rows - 1) * 6);
        for (int j = 0; j + 1 < rows; ++j) {
            for (int i = 0; i + 1 < columns; ++i) {
                const unsigned a = j * columns + i, b = a + 1, c = a + columns, d = c + 1;
                index.insert(index.end(), {a, c, b, b, c, d});
            }
        }
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
        geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));
        geometry->setIndex(index);
        auto material = MeshStandardMaterial::create();
        material->vertexColors = true;
        material->roughness = 0.95f;
        return shadowed(Mesh::create(geometry, material));
    }

    // the tile's cut sides: soil from the terrain's edge down to a base
    std::shared_ptr<Mesh> makeSkirt(const std::vector<float>& ground) {
        std::vector<float> positions;
        std::vector<unsigned int> index;
        const auto side = [&](int n, const std::function<std::pair<float, float>(int)>& xz, const std::function<int(int)>& cell, bool flip) {
            const unsigned start = positions.size() / 3;
            for (int k = 0; k < n; ++k) {
                const auto [x, z] = xz(k);
                positions.insert(positions.end(), {x, ground[cell(k)], z, x, -4.5f, z});
            }
            for (int k = 0; k + 1 < n; ++k) {
                const unsigned a = start + 2 * k, b = a + 1, c = a + 2, d = a + 3;
                if (flip) index.insert(index.end(), {a, c, b, b, c, d});
                else index.insert(index.end(), {a, b, c, b, d, c});
            }
        };
        const auto cx = [](int i) { return tileX0 + (static_cast<float>(i) + 0.5f) * cellSize; };
        const auto cz = [](int j) { return tileZ0 + (static_cast<float>(j) + 0.5f) * cellSize; };
        side(columns, [&](int i) { return std::pair{cx(i), cz(0)}; }, [](int i) { return i; }, true);
        side(columns, [&](int i) { return std::pair{cx(i), cz(rows - 1)}; }, [](int i) { return (rows - 1) * columns + i; }, false);
        side(rows, [&](int j) { return std::pair{cx(0), cz(j)}; }, [](int j) { return j * columns; }, false);
        side(rows, [&](int j) { return std::pair{cx(columns - 1), cz(j)}; }, [](int j) { return j * columns + columns - 1; }, true);
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geometry->setIndex(index);
        geometry->computeVertexNormals();
        auto material = MeshStandardMaterial::create();
        material->color = srgb(86, 66, 48);
        material->roughness = 1;
        material->side = Side::Double;
        return Mesh::create(geometry, material);
    }

    // ---- stones ------------------------------------------------------------------------------------------------------
    std::shared_ptr<BufferGeometry> stoneGeometry(unsigned seed) {
        auto geometry = IcosahedronGeometry::create(1, 2);
        auto& p = geometry->getAttribute<float>("position")->array();
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(0, 6.283f);
        const float a1 = u(rng), a2 = u(rng), a3 = u(rng);
        for (size_t k = 0; k < p.size(); k += 3) {
            Vector3 v(p[k], p[k + 1], p[k + 2]);
            v.normalize();
            const float bump = 1 + 0.12f * std::sin(3 * v.x + a1) * std::sin(2 * v.z + a2) + 0.06f * std::sin(7 * v.y + 4 * v.x + a3);
            v.multiplyScalar(bump);
            if (v.y < -0.3f) v.y = -0.3f + 0.3f * (v.y + 0.3f);// a flatter underside
            p[k] = v.x;
            p[k + 1] = v.y;
            p[k + 2] = v.z;
        }
        geometry->computeVertexNormals();
        return geometry;
    }

    // ---- the dam -----------------------------------------------------------------------------------------------------
    // a log crib between the banks, its sluice with a plank gate on a rack and the windlass's hand wheel, a catwalk
    // with a rail and the apron the jet runs down. Its frame: the sill's top centre at the origin, +z downstream
    struct Dam {
        std::shared_ptr<Group> root = Group::create();
        std::shared_ptr<Group> gate = Group::create();
        std::shared_ptr<Group> wheel = Group::create();

        Dam() {
            std::mt19937 rng(11);
            std::uniform_real_distribution<float> u(0, 1);
            auto wood = [&](int r, int g, int b) {
                auto m = MeshStandardMaterial::create();
                m->color = srgb(r, g, b);
                m->roughness = 0.9f;
                return m;
            };
            std::vector<std::shared_ptr<MeshStandardMaterial>> dry{wood(132, 126, 115), wood(121, 116, 106), wood(140, 132, 118)};
            std::vector<std::shared_ptr<MeshStandardMaterial>> wet{wood(82, 76, 68), wood(74, 69, 62), wood(88, 80, 70)};
            auto plank = wood(118, 108, 94);
            auto iron = MeshStandardMaterial::create();
            iron->color = srgb(58, 52, 48);
            iron->metalness = 0.5f;
            iron->roughness = 0.6f;

            const float top = crest - sill;// the crest in this frame
            const auto log = [&](float x0, float x1, float y, float z, bool isWet) {
                const float r = 0.11f + 0.02f * u(rng);
                auto geometry = CylinderGeometry::create(r, r * (0.92f + 0.08f * u(rng)), x1 - x0, 10);
                geometry->rotateZ(math::PI / 2);
                auto mesh = shadowed(Mesh::create(geometry, (isWet ? wet : dry)[rng() % 3]));
                mesh->position.set(0.5f * (x0 + x1), y, z);
                root->add(mesh);
            };
            // the faces: courses of logs, split by the sluice above its sill
            const float r = 0.12f;
            for (const auto& [z, bottom, isUpstream] : {std::tuple{cribHalf - r, -0.55f, false}, std::tuple{-cribHalf + r, -0.15f, true}}) {
                for (float y = bottom + r; y <= top - r + 0.02f; y += 2 * r) {
                    const bool isWet = isUpstream || y < 0.35f;
                    if (y + r > 0) {
                        log(-cribWidth, -0.45f, y, z, isWet);
                        log(0.45f, cribWidth, y, z, isWet);
                    } else {
                        log(-cribWidth, cribWidth, y, z, isWet);
                    }
                }
            }
            // the ties across the crib, their ends showing past both faces
            for (float x : {-2.8f, -1.4f, 1.4f, 2.8f}) {
                for (float y = 0.0f; y <= top - 0.1f; y += 0.48f) {
                    auto geometry = CylinderGeometry::create(0.1f, 0.1f, 2 * cribHalf + 0.22f, 8);
                    geometry->rotateX(math::PI / 2);
                    auto mesh = shadowed(Mesh::create(geometry, dry[rng() % 3]));
                    mesh->position.set(x, y + 0.12f, 0);
                    root->add(mesh);
                }
            }
            const auto box = [&](float sx, float sy, float sz, float x, float y, float z, const std::shared_ptr<Material>& m, Object3D& parent) {
                auto mesh = shadowed(Mesh::create(BoxGeometry::create(sx, sy, sz), m));
                mesh->position.set(x, y, z);
                parent.add(mesh);
                return mesh;
            };
            // the sill, the sluice's cheeks, the posts and the crossbeam over the gate
            box(1.04f, 0.2f, 2 * cribHalf, 0, -0.1f, 0, wet[0], *root);
            box(0.06f, top + 0.1f, 2 * cribHalf, -0.45f, top / 2, 0, wet[1], *root);
            box(0.06f, top + 0.1f, 2 * cribHalf, 0.45f, top / 2, 0, wet[1], *root);
            box(0.2f, 2.7f, 0.2f, -0.57f, 1.05f, gatePlane, dry[0], *root);
            box(0.2f, 2.7f, 0.2f, 0.57f, 1.05f, gatePlane, dry[0], *root);
            box(1.5f, 0.2f, 0.3f, 0, 2.5f, gatePlane, dry[1], *root);
            // the gate: planks, battens and the rack's stem; it slides up its posts
            gate->position.set(0, 0, gatePlane);
            root->add(gate);
            for (int k = 0; k < 4; ++k) box(0.94f, 0.32f, 0.06f, 0, 0.165f + 0.325f * static_cast<float>(k), 0, plank, *gate);
            box(0.08f, 1.2f, 0.05f, -0.28f, 0.65f, 0.05f, plank, *gate);
            box(0.08f, 1.2f, 0.05f, 0.28f, 0.65f, 0.05f, plank, *gate);
            box(0.08f, 1.7f, 0.08f, 0, 1.3f + 0.85f, 0, iron, *gate);
            // the windlass: an axle on the crossbeam, the hand wheel beside the posts
            auto axle = CylinderGeometry::create(0.035f, 0.035f, 1.1f, 10);
            axle->rotateZ(math::PI / 2);
            auto axleMesh = shadowed(Mesh::create(axle, iron));
            axleMesh->position.set(0.5f, 2.72f, gatePlane);
            root->add(axleMesh);
            wheel->position.set(1.0f, 2.72f, gatePlane);
            root->add(wheel);
            auto rim = TorusGeometry::create(0.42f, 0.035f, 8, 36);
            rim->rotateY(math::PI / 2);
            wheel->add(shadowed(Mesh::create(rim, dry[2])));
            for (int k = 0; k < 6; ++k) {
                auto spoke = shadowed(Mesh::create(BoxGeometry::create(0.04f, 0.84f, 0.04f), dry[2]));
                spoke->rotation.x = static_cast<float>(k) * math::PI / 6;
                wheel->add(spoke);
            }
            auto hub = CylinderGeometry::create(0.07f, 0.07f, 0.12f, 12);
            hub->rotateZ(math::PI / 2);
            wheel->add(shadowed(Mesh::create(hub, iron)));
            // the catwalk over the crib and its rail on the downstream side
            for (int k = 0; k < 3; ++k) box(2 * cribWidth + 0.3f, 0.05f, 0.17f, 0, top + 0.03f, -0.04f + 0.19f * static_cast<float>(k), plank, *root);
            for (float x : {-3.4f, -1.7f, 0.0f, 1.7f, 3.4f}) {
                auto post = shadowed(Mesh::create(CylinderGeometry::create(0.04f, 0.045f, 0.95f, 8), dry[1]));
                post->position.set(x, top + 0.5f, 0.42f);
                root->add(post);
            }
            auto railGeometry = CylinderGeometry::create(0.035f, 0.035f, 2 * cribWidth, 8);
            railGeometry->rotateZ(math::PI / 2);
            auto rail = shadowed(Mesh::create(railGeometry, dry[1]));
            rail->position.set(0, top + 0.97f, 0.42f);
            root->add(rail);
            // the apron below the gate: planks sloping 15 cm down from the sill, with low sides
            const float length = 0.95f, drop = 0.15f;
            auto apron = box(0.9f, 0.05f, length, 0, -0.025f - drop / 2, cribHalf + length / 2, wet[2], *root);
            apron->rotation.x = std::atan2(drop, length);
            box(0.05f, 0.25f, length, -0.47f, 0.02f - drop / 2, cribHalf + length / 2, wet[1], *root);
            box(0.05f, 0.25f, length, 0.47f, 0.02f - drop / 2, cribHalf + length / 2, wet[1], *root);
        }

        // the gate raised by `opening` m and the wheel turned to match (its pinion is 7 cm in radius)
        void pose(float opening) const {
            gate->position.y = opening;
            wheel->rotation.x = opening / 0.07f;
        }
    };

    // ---- the trees ---------------------------------------------------------------------------------------------------
    // simple pines round the course: a trunk and two cones each, instanced
    void plantTrees(Scene& scene, const Course& course, const std::vector<float>& ground) {
        std::mt19937 rng(17);
        std::uniform_real_distribution<float> u(0, 1);
        struct Tree {
            float x, z, y, height;
        };
        std::vector<Tree> trees;
        for (int tries = 0; trees.size() < 48 && tries < 6000; ++tries) {
            const float x = tileX0 + 0.6f + (tileX1 - tileX0 - 1.2f) * u(rng);
            const float z = tileZ0 + 0.6f + (tileZ1 - tileZ0 - 1.2f) * u(rng);
            if (course.clear(x, z, 0.6f)) continue;
            bool apart = true;
            for (const auto& t : trees) apart &= std::hypot(x - t.x, z - t.z) > 2.6f;
            if (!apart) continue;
            const int i = std::clamp(static_cast<int>((x - tileX0) / cellSize), 0, columns - 1);
            const int j = std::clamp(static_cast<int>((z - tileZ0) / cellSize), 0, rows - 1);
            trees.push_back({x, z, ground[j * columns + i] - 0.05f, 4.5f + 4.f * u(rng)});
        }
        auto trunkGeometry = CylinderGeometry::create(0.09f, 0.16f, 1, 7);
        trunkGeometry->translate(0, 0.5f, 0);
        auto coneGeometry = ConeGeometry::create(1, 1, 9);
        coneGeometry->translate(0, 0.5f, 0);
        auto bark = MeshStandardMaterial::create();
        bark->color = srgb(96, 70, 50);
        bark->roughness = 0.95f;
        auto needles = MeshStandardMaterial::create();
        needles->roughness = 0.9f;
        auto trunks = shadowed(InstancedMesh::create(trunkGeometry, bark, trees.size()));
        auto lower = shadowed(InstancedMesh::create(coneGeometry, needles, trees.size()));
        auto upper = shadowed(InstancedMesh::create(coneGeometry, needles, trees.size()));
        Matrix4 m;
        for (size_t k = 0; k < trees.size(); ++k) {
            const auto& t = trees[k];
            const float h = t.height, w = 0.32f * h * (0.85f + 0.3f * u(rng));
            m.makeScale(1, 0.55f * h, 1).setPosition(t.x, t.y, t.z);
            trunks->setMatrixAt(k, m);
            m.makeScale(w, 0.55f * h, w).setPosition(t.x, t.y + 0.3f * h, t.z);
            lower->setMatrixAt(k, m);
            m.makeScale(0.68f * w, 0.42f * h, 0.68f * w).setPosition(t.x, t.y + 0.6f * h, t.z);
            upper->setMatrixAt(k, m);
            const Color green = srgb(46, 74, 40).lerp(srgb(70, 94, 50), u(rng));
            lower->setColorAt(k, green);
            upper->setColorAt(k, Color(green).multiplyScalar(1.08f));
            trunks->setColorAt(k, Color(1, 1, 1).multiplyScalar(0.8f + 0.4f * u(rng)));
        }
        scene.add(trunks);
        scene.add(lower);
        scene.add(upper);
    }

    // ---- the water's look --------------------------------------------------------------------------------------------
    // the ripple map: a tileable field of small ripples (RGB, a tangent-space normal over world XZ) and a foam pattern
    // carried with them (A: a lace of veins over soft blobs; the simulation's foam decides how much of it is white)
    std::shared_ptr<DataTexture> makeRipples(int size = 256) {
        struct Wave {
            float m, n, a, phase;
        };
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(0, 1);
        std::vector<Wave> ripples, lace;
        while (ripples.size() < 48) {
            const float m = std::round(u(rng) * 32 - 16), n = std::round(u(rng) * 32 - 16), k = std::hypot(m, n);
            if (k < 3 || k > 16) continue;
            ripples.push_back({m, n, std::pow(k, -1.4f), 6.283f * u(rng)});
        }
        while (lace.size() < 10) {
            const float m = std::round(u(rng) * 14 - 7), n = std::round(u(rng) * 14 - 7), k = std::hypot(m, n);
            if (k < 2 || k > 7) continue;
            lace.push_back({m, n, 1, 6.283f * u(rng)});
        }
        std::vector<float> gx(size * size), gy(size * size), veins(size * size);
        float maxSlope = 0, laceNorm = 0;
        for (const auto& w : lace) laceNorm += w.a;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float s = static_cast<float>(x) / static_cast<float>(size), t = static_cast<float>(y) / static_cast<float>(size);
                float dx = 0, dy = 0, field = 0;
                for (const auto& w : ripples) {
                    const float arg = 6.283f * (w.m * s + w.n * t) + w.phase;
                    dx -= w.a * 6.283f * w.m * std::sin(arg);
                    dy -= w.a * 6.283f * w.n * std::sin(arg);
                }
                for (const auto& w : lace) field += w.a * std::cos(6.283f * (w.m * s + w.n * t) + w.phase);
                const int c = y * size + x;
                gx[c] = dx;
                gy[c] = dy;
                veins[c] = field / laceNorm;
                maxSlope = std::max(maxSlope, std::hypot(dx, dy));
            }
        }
        std::vector<unsigned char> pixels(size * size * 4);
        const float scale = 0.55f / maxSlope;
        for (int c = 0; c < size * size; ++c) {
            const float nx = -gx[c] * scale, ny = -gy[c] * scale;
            const float nz = std::sqrt(std::max(0.05f, 1 - nx * nx - ny * ny));
            const float foam = std::clamp(0.8f * std::exp(-std::abs(veins[c]) / 0.09f) + 0.25f * (0.5f + 0.5f * std::sin(9.f * veins[c])), 0.f, 1.f);
            pixels[4 * c] = static_cast<unsigned char>(std::clamp(nx * 0.5f + 0.5f, 0.f, 1.f) * 255);
            pixels[4 * c + 1] = static_cast<unsigned char>(std::clamp(ny * 0.5f + 0.5f, 0.f, 1.f) * 255);
            pixels[4 * c + 2] = static_cast<unsigned char>(std::clamp(nz * 0.5f + 0.5f, 0.f, 1.f) * 255);
            pixels[4 * c + 3] = static_cast<unsigned char>(foam * 255);
        }
        auto texture = DataTexture::create(ImageData{std::move(pixels)}, size, size);
        texture->wrapS = texture->wrapT = TextureWrapping::Repeat;
        texture->magFilter = Filter::Linear;
        texture->minFilter = Filter::LinearMipmapLinear;
        texture->generateMipmaps = true;
        texture->needsUpdate();
        return texture;
    }

    const char* waterVertex = R"(
        attribute vec2 flowVel;
        attribute vec2 water;
        varying vec3 vWorld;
        varying vec3 vNormal;
        varying vec2 vVel;
        varying vec2 vWater;
        void main() {
            vec4 world = modelMatrix * vec4(position, 1.0);
            vWorld = world.xyz;
            vNormal = normalize(mat3(modelMatrix) * normal);
            vVel = flowVel;
            vWater = water;
            gl_Position = projectionMatrix * viewMatrix * world;
        })";

    // The water: the ripple map laid over world XZ and carried along the simulated velocity in two phases half a
    // period apart, crossfaded so neither phase's restart shows; the sky and the sun reflected by Fresnel; the bed seen
    // through the water's depth, the light absorbed on its way down and back up, the water's own brown scattered back;
    // the foam's lace whitening it where the simulation says the water is white
    const char* waterFragment = R"(
        uniform sampler2D ripples;
        uniform float time;
        uniform float tile;
        uniform float period;
        uniform vec3 sunDirection;
        uniform vec3 sunColor;
        uniform vec3 skyZenith;
        uniform vec3 skyHorizon;
        uniform vec3 treeline;
        uniform vec3 scatter;
        uniform vec3 absorption;
        varying vec3 vWorld;
        varying vec3 vNormal;
        varying vec2 vVel;
        varying vec2 vWater;
        void main() {
            float depth = vWater.y;
            if (depth < 0.0005) discard;
            float ph0 = fract(time / period);
            float ph1 = fract(time / period + 0.5);
            float blend = abs(1.0 - 2.0 * ph0);
            vec2 base = vWorld.xz / tile;
            vec4 t0 = texture2D(ripples, base - vVel * (ph0 * period / tile));
            vec4 t1 = texture2D(ripples, base - vVel * (ph1 * period / tile) + vec2(0.37, 0.61));
            vec4 tap = mix(t0, t1, blend);
            float speed = length(vVel);
            vec2 tilt = (tap.xy * 2.0 - 1.0) * (0.45 + 0.55 * clamp(speed / 1.5, 0.0, 1.0));
            vec3 N = normalize(vNormal);
            vec3 T = normalize(vec3(1.0, 0.0, 0.0) - N * N.x);
            vec3 B = normalize(vec3(0.0, 0.0, 1.0) - N * N.z);
            N = normalize(T * tilt.x + B * tilt.y + N * sqrt(max(1e-4, 1.0 - dot(tilt, tilt))));

            vec3 V = normalize(cameraPosition - vWorld);
            float cosV = max(dot(N, V), 0.0);
            float fresnel = 0.02 + 0.98 * pow(1.0 - cosV, 5.0);
            vec3 R = reflect(-V, N);
            // low reflections are of the dark treeline round the water, higher ones of the sky
            float up = clamp(R.y, 0.0, 1.0);
            vec3 sky = mix(treeline, mix(skyHorizon, skyZenith, pow(up, 0.6)), smoothstep(0.15, 0.65, up));
            vec3 glint = sunColor * pow(max(dot(R, sunDirection), 0.0), 500.0) * 3.0;
            float sunlit = max(dot(N, sunDirection), 0.0);

            // the bed through the water, along the view down and back up
            vec3 through = exp(-absorption * depth * (1.0 + 1.0 / max(cosV, 0.25)));
            float seen = dot(through, vec3(1.0 / 3.0)) * (1.0 - fresnel);
            vec3 body = scatter * (1.0 - through) * (0.4 + 0.6 * sunlit);
            float alpha = 1.0 - seen;
            vec3 color = (sky * fresnel + glint + body * (1.0 - fresnel)) / max(alpha, 1e-3);

            float foam = 0.75 * vWater.x;
            float white = clamp((tap.a + 1.2 * foam - 1.05) / 0.35, 0.0, 1.0) * clamp(3.0 * foam, 0.0, 1.0);
            vec3 froth = vec3(0.62) * (0.45 * skyZenith + sunColor * (0.25 + 0.6 * sunlit) * 0.6);
            color = mix(color, froth, white);
            alpha = mix(alpha, 1.0, 0.85 * white);
            gl_FragColor = vec4(color, alpha);
            #include <tonemapping_fragment>
            #include <encodings_fragment>
        })";

    struct View {
        Vector3 eye, target;
    };

    const std::map<std::string, View> views{
            {"overview", {{15.f, 24.f, 20.f}, {0.f, -0.5f, -2.f}}},
            {"dam", {{3.4f, 2.6f, -3.0f}, {0.f, 0.7f, -8.f}}},
            {"pond", {{-2.6f, 3.0f, -21.5f}, {0.f, 1.2f, -8.5f}}},
            {"rapid", {{-3.4f, 2.4f, 4.5f}, {0.2f, -0.2f, -4.f}}},
            {"falls", {{-2.2f, -0.6f, 13.5f}, {0.2f, -1.3f, 9.f}}},
    };

}// namespace

int main(int argc, char** argv) {

    std::string shotPath, viewName = "overview";
    float shotTime = 4, openTime = -1;
    int width = 1280, height = 720;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--time" && i + 1 < argc) shotTime = std::stof(argv[++i]);
        else if (a == "--open" && i + 1 < argc) openTime = std::stof(argv[++i]);
        else if (a == "--view" && i + 1 < argc) viewName = argv[++i];
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
    }
    const bool headless = !shotPath.empty();
    if (!views.count(viewName)) viewName = "overview";

    Canvas canvas(Canvas::Parameters()
                          .title("Sluice dam")
                          .size(width, height)
                          .antialiasing(4)
                          .headless(headless));
    GLRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 0.9f;
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    const Color skyHorizon(0.72f, 0.78f, 0.84f), skyZenith(0.32f, 0.46f, 0.66f);
    Scene scene;
    scene.background = skyHorizon;
    scene.fog = Fog(skyHorizon, 40, 110);

    const Vector3 sunDirection = Vector3(-0.45f, 0.75f, 0.48f).normalize();
    auto sun = DirectionalLight::create(0xfff1e0, 2.6f);
    sun->position.copy(sunDirection).multiplyScalar(40);
    sun->castShadow = true;
    sun->shadow->mapSize.set(4096, 4096);
    sun->shadow->bias = -0.0004f;
    auto shadowCamera = sun->shadow->camera->as<OrthographicCamera>();
    shadowCamera->left = shadowCamera->bottom = -30;
    shadowCamera->right = shadowCamera->top = 30;
    shadowCamera->nearPlane = 1;
    shadowCamera->farPlane = 90;
    shadowCamera->updateProjectionMatrix();
    scene.add(sun);
    scene.add(HemisphereLight::create(Color(0.78f, 0.85f, 0.95f), Color(0.25f, 0.22f, 0.16f), 1.1f));

    // ---- the ground, the stones, the dam, the trees -----------------------------------------------------------------
    Course course;
    const int count = columns * rows;
    std::vector<float> ground(count);
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < columns; ++i) {
            ground[j * columns + i] = course.height(tileX0 + (static_cast<float>(i) + 0.5f) * cellSize, tileZ0 + (static_cast<float>(j) + 0.5f) * cellSize);
        }
    }
    scene.add(makeTerrain(course, ground));
    scene.add(makeSkirt(ground));

    auto stoneMaterial = MeshStandardMaterial::create();
    stoneMaterial->color = srgb(96, 94, 86);
    stoneMaterial->roughness = 0.88f;
    std::vector<std::shared_ptr<BufferGeometry>> stones;
    for (unsigned k = 0; k < 4; ++k) stones.push_back(stoneGeometry(100 + k));
    for (size_t k = 0; k < course.rocks().size(); ++k) {
        const auto& rock = course.rocks()[k];
        auto mesh = shadowed(Mesh::create(stones[k % stones.size()], stoneMaterial));
        mesh->position.set(rock.x, rock.y, rock.z);
        mesh->scale.set(rock.radius, rock.radius * rock.squash, rock.radius);
        mesh->rotation.y = rock.yaw;
        scene.add(mesh);
    }

    Dam dam;
    dam.root->position.set(damX, sill, damZ);
    scene.add(dam.root);

    plantTrees(scene, course, ground);

    // ---- the water ---------------------------------------------------------------------------------------------------
    // the bed: the ground, raised by the crib (to its crest), the sill and the apron, and by the stones (each a
    // squashed sphere's cap, grown a cell so no water shows inside a stone)
    std::vector<float> bed = ground, stoneTop(count, -1e9f);
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < columns; ++i) {
            const int c = j * columns + i;
            const float lx = tileX0 + (static_cast<float>(i) + 0.5f) * cellSize - damX;
            const float lz = tileZ0 + (static_cast<float>(j) + 0.5f) * cellSize - damZ;
            if (std::abs(lz) <= cribHalf && std::abs(lx) <= cribWidth) bed[c] = std::abs(lx) < gateHalf ? sill : crest;
            if (lz > cribHalf && lz <= cribHalf + 0.95f && std::abs(lx) < 0.45f) bed[c] = std::max(bed[c], sill - 0.15f * (lz - cribHalf) / 0.95f);
        }
    }
    for (const auto& rock : course.rocks()) {
        const int i0 = std::max(0, static_cast<int>((rock.x - rock.radius - tileX0) / cellSize) - 1);
        const int i1 = std::min(columns - 1, static_cast<int>((rock.x + rock.radius - tileX0) / cellSize) + 1);
        const int j0 = std::max(0, static_cast<int>((rock.z - rock.radius - tileZ0) / cellSize) - 1);
        const int j1 = std::min(rows - 1, static_cast<int>((rock.z + rock.radius - tileZ0) / cellSize) + 1);
        for (int j = j0; j <= j1; ++j) {
            for (int i = i0; i <= i1; ++i) {
                const float d = std::hypot(tileX0 + (static_cast<float>(i) + 0.5f) * cellSize - rock.x, tileZ0 + (static_cast<float>(j) + 0.5f) * cellSize - rock.z);
                if (d < rock.radius) stoneTop[j * columns + i] = std::max(stoneTop[j * columns + i], rock.y + rock.squash * rock.radius * std::sqrt(1 - (d / rock.radius) * (d / rock.radius)));
            }
        }
    }
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < columns; ++i) {
            float top = -1e9f;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di)
                    if (i + di >= 0 && i + di < columns && j + dj >= 0 && j + dj < rows) top = std::max(top, stoneTop[(j + dj) * columns + i + di]);
            bed[j * columns + i] = std::max(bed[j * columns + i], top);
        }
    }
    std::vector<float> cover(count);
    for (int c = 0; c < count; ++c) cover[c] = std::max(ground[c], bed[c]);

    ShallowWater water(columns, rows, cellSize, tileX0, tileZ0);
    float gate = gateStart, gateTarget = gateStart;
    const auto start = [&] {
        water = ShallowWater(columns, rows, cellSize, tileX0, tileZ0);
        water.setBed(bed, ground, cover);
        std::vector<uint8_t> near(count), active(count);
        for (int j = 0; j < rows; ++j)
            for (int i = 0; i < columns; ++i)
                near[j * columns + i] = course.wet(tileX0 + (static_cast<float>(i) + 0.5f) * cellSize, tileZ0 + (static_cast<float>(j) + 0.5f) * cellSize, 0.6f);
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < columns; ++i) {
                bool on = false;
                for (int dj = -3; dj <= 3 && !on; ++dj)
                    for (int di = -3; di <= 3 && !on; ++di)
                        on = i + di >= 0 && i + di < columns && j + dj >= 0 && j + dj < rows && near[(j + dj) * columns + i + di];
                active[j * columns + i] = on && i > 0 && j > 0 && i < columns - 1 && j < rows - 1;
            }
        }
        water.setActive(active);
        int springs = 0;
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < columns; ++i) {
                    const int c = j * columns + i;
                    const float x = tileX0 + (static_cast<float>(i) + 0.5f) * cellSize, z = tileZ0 + (static_cast<float>(j) + 0.5f) * cellSize;
                    if (!active[c]) continue;
                    if (Course::spring(x, z)) {
                        if (pass == 0) ++springs;
                        else water.setSource(c, inflow / (static_cast<float>(springs) * cellSize * cellSize));
                    }
                    if (pass == 1 && Course::drain(x, z)) water.setSink(c, lakeLevel);
                    if (pass == 1) {
                        // the water in place to begin with: the pond full to just under its crest, the lake full, a film in
                        // the stream and the brook; a few seconds of flow then set it going
                        float level = -1e9f;
                        if (Course::pondSd(x, z) < 0 && z < damZ) level = crest - 0.05f;
                        if (Course::lakeSd(x, z) < 0) level = lakeLevel;
                        if (course.wet(x, z, -0.3f) && level < -1e8f) level = bed[c] + 0.04f;
                        if (level > -1e8f) water.setDepth(c, level - bed[c]);
                    }
                }
            }
        }
        const int gateRow = static_cast<int>(std::round((damZ + gatePlane - tileZ0) / cellSize));
        water.addGate(gateRow, static_cast<int>((damX - gateHalf - tileX0) / cellSize), static_cast<int>((damX + gateHalf - tileX0) / cellSize) - 1);
        gate = gateTarget = gateStart;
        water.setGate(gate);
        water.settle(20);
    };
    start();

    std::vector<float> positions(count * 3), normals(count * 3), velocity(count * 2), wetness(count * 2);
    water.surface(positions.data(), normals.data(), velocity.data(), wetness.data());
    auto waterGeometry = BufferGeometry::create();
    waterGeometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    waterGeometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
    waterGeometry->setAttribute("flowVel", FloatBufferAttribute::create(velocity, 2));
    waterGeometry->setAttribute("water", FloatBufferAttribute::create(wetness, 2));
    for (const auto& name : {"position", "normal", "flowVel", "water"}) waterGeometry->getAttribute(name)->setUsage(DrawUsage::Dynamic);
    {
        std::vector<unsigned int> index;
        for (int j = 0; j + 1 < rows; ++j) {
            for (int i = 0; i + 1 < columns; ++i) {
                const unsigned a = j * columns + i, b = a + 1, c = a + columns, d = c + 1;
                if (course.wet(positions[3 * a], positions[3 * a + 2], 1.2f)) index.insert(index.end(), {a, c, b, b, c, d});
            }
        }
        waterGeometry->setIndex(index);
    }
    auto ripples = makeRipples();
    auto waterMaterial = ShaderMaterial::create();
    waterMaterial->vertexShader = waterVertex;
    waterMaterial->fragmentShader = waterFragment;
    waterMaterial->transparent = true;
    waterMaterial->uniforms = {
            {"ripples", Uniform(static_cast<Texture*>(ripples.get()))},
            {"time", Uniform(0.f)},
            {"tile", Uniform(0.5f)},
            {"period", Uniform(0.8f)},
            {"sunDirection", Uniform(sunDirection)},
            {"sunColor", Uniform(Vector3(2.6f, 2.45f, 2.25f))},
            {"skyZenith", Uniform(Vector3(skyZenith.r, skyZenith.g, skyZenith.b))},
            {"skyHorizon", Uniform(Vector3(skyHorizon.r, skyHorizon.g, skyHorizon.b))},
            {"treeline", Uniform(Vector3(0.03f, 0.045f, 0.028f))},
            {"scatter", Uniform(Vector3(0.14f, 0.095f, 0.05f))},
            {"absorption", Uniform(Vector3(0.7f, 1.3f, 2.4f))}};
    auto waterMesh = Mesh::create(waterGeometry, waterMaterial);
    waterMesh->frustumCulled = false;
    scene.add(waterMesh);

    // ---- the camera, the panel, the keys -----------------------------------------------------------------------------
    PerspectiveCamera camera(45, canvas.aspect(), 0.1f, 300);
    camera.position.copy(views.at(viewName).eye);
    OrbitControls controls{camera, canvas};
    controls.target.copy(views.at(viewName).target);
    controls.update();

    float solverMs = 0, simTime = 0;
    bool paused = false;
    const auto pondLevel = [&] {
        const int i = static_cast<int>((damX - tileX0) / cellSize), j = static_cast<int>((damZ - 2.0f - tileZ0) / cellSize);
        return water.level(j * columns + i);
    };

    std::unique_ptr<ImguiFunctionalContext> ui;
    IOCapture capture{};
    if (!headless) {
        ui = std::make_unique<ImguiFunctionalContext>(canvas, renderer, [&] {
            ImGui::SetNextWindowPos({10, 10}, 0, {0, 0});
            ImGui::SetNextWindowSize({300, 0}, 0);
            ImGui::Begin("Sluice dam");
            ImGui::SliderFloat("gate (m)", &gateTarget, 0, gateMax, "%.2f");
            if (ImGui::Button("Open")) gateTarget = gateMax;
            ImGui::SameLine();
            if (ImGui::Button("Shut")) gateTarget = 0;
            ImGui::SameLine();
            if (ImGui::Button(paused ? "Run" : "Pause")) paused = !paused;
            ImGui::SameLine();
            if (ImGui::Button("Start over")) start();
            ImGui::Text("gate at %.2f m, %.2f m3/s under it", gate, water.gateFlux());
            ImGui::Text("pond %+.2f m on the dam's top", pondLevel() - crest);
            ImGui::Text("water %.0f m3, solver %.1f ms", water.volume(), solverMs);
            ImGui::Text("G: open / shut, Up / Down: 10 cm");
            ImGui::End();
        });
        capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
        capture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
        canvas.setIOCapture(&capture);
    }
    canvas.onKeyPressed([&](KeyEvent evt) {
        switch (evt.key) {
            case Key::G: gateTarget = gateTarget > 0.5f * gateMax ? 0.f : gateMax; break;
            case Key::UP: gateTarget = std::min(gateMax, gateTarget + 0.1f); break;
            case Key::DOWN: gateTarget = std::max(0.f, gateTarget - 0.1f); break;
            case Key::SPACE: paused = !paused; break;
            case Key::R: start(); break;
            default: break;
        }
    });
    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    double solverTotal = 0;
    int frames = 0;
    canvas.animate([&] {
        const float dt = headless ? 1.f / 60 : std::min(clock.getDelta(), 0.05f);
        if (openTime >= 0 && simTime >= openTime) {
            gateTarget = gateMax;
            openTime = -1;
        }
        if (!paused) {
            const float step = winch * dt;
            gate += std::clamp(gateTarget - gate, -step, step);
            water.setGate(std::max(gate, gateLeak));
            const auto t0 = std::chrono::steady_clock::now();
            water.step(dt);
            const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
            solverMs += 0.05f * (ms - solverMs);
            solverTotal += ms;
            simTime += dt;
        }
        water.surface(positions.data(), normals.data(), velocity.data(), wetness.data());
        waterGeometry->getAttribute<float>("position")->array() = positions;
        waterGeometry->getAttribute<float>("normal")->array() = normals;
        waterGeometry->getAttribute<float>("flowVel")->array() = velocity;
        waterGeometry->getAttribute<float>("water")->array() = wetness;
        for (const auto& name : {"position", "normal", "flowVel", "water"}) waterGeometry->getAttribute(name)->needsUpdate();
        waterMaterial->uniforms["time"].setValue(simTime);
        dam.pose(gate);

        renderer.render(scene, camera);
        if (ui) ui->render();
        ++frames;

        if (headless && simTime >= shotTime) {
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << "\n"
                      << "solver " << solverTotal / frames << " ms a frame; gate " << gate << " m, " << water.gateFlux()
                      << " m3/s under it; pond " << pondLevel() - crest << " m on the crest; water " << water.volume() << " m3" << std::endl;
            std::exit(0);
        }
    });
}
