// The summer woods round the watermill: the trees (built from scratch, every leaf its own), the shrubs on the rocks,
// the ferns and grass of the banks.
//
// A handful of trees is grown once (TreesGrow.hpp), each at three levels of detail, and planted many times over as
// instances, turned, scaled and tinted each its own way: full detail along the hero view's stretch of the stream and
// round the mill, fewer and larger leaves further out, a sparse crown on the far hillsides. One tree is grown for
// its place alone: the leaning trunk at the left of the hero view whose crown hangs over the top of the picture.
// The instances are gathered in tiles so the camera and the sun's shadow camera each draw only the tiles they see.

#ifndef THREEPP_EXAMPLE_WATERMILL_TREES_HPP
#define THREEPP_EXAMPLE_WATERMILL_TREES_HPP

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Frustum.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Sphere.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/utils/ThreadPool.hpp"

#include "FoliagePlants.hpp"
#include "FoliageTextures.hpp"
#include "Materials.hpp"
#include "Site.hpp"
#include "Terrain.hpp"
#include "TreesGrow.hpp"

#include <chrono>
#include <cstdio>
#include <functional>
#include <map>

namespace mill {

    namespace woods {

        // ---- the trees ----------------------------------------------------------------------------------------------
        enum TreeVariant : int {
            TulipTall = 0,// tulip poplar: tall, straight, a high narrow crown
            TulipBank,    // tulip poplar on the bank, leaning out over the water
            MapleFork,    // sugar maple: forked low, a broad round crown
            OakSpread,    // white oak: crooked, spreading limbs
            BeechLow,     // beech: smooth grey, a low dense crown with drooping ends
            BirchBank,    // birch: white, leaning, a light crown
            BasswoodTall, // basswood: tall, a full rounded crown
            DogwoodUnder, // a small understory tree, layered
            Sapling,      // a young beech in the shade
            TreeVariants
        };

        inline std::vector<TreeSpec> treeSpecs() {
            std::vector<TreeSpec> v(TreeVariants);
            {
                auto& t = v[TulipTall];
                t.seed = 101;
                t.height = 27.f;
                t.crownBase = 0.5f;
                t.crownRadius = 5.6f;
                t.crownFlat = 0.85f;
                t.lobes = 7;
                t.trunkRadius = 0.3f;
                t.lean = 0.04f;
                t.wobble = 0.08f;
                t.leader = 0.86f;
                t.upTropism = 0.24f;
                t.leafKind = woods::Tulip;
                t.leafLength = 0.23f;
                t.leaves = 32000;
                t.bark = Furrowed;
                t.barkTint = V3(0.95f, 0.93f, 0.88f);
                t.leafTint = V3(1.02f, 1.02f, 0.95f);
            }
            {
                auto& t = v[TulipBank];
                t = v[TulipTall];
                t.seed = 102;
                t.height = 22.f;
                t.crownBase = 0.4f;
                t.crownRadius = 5.8f;
                t.crownShift = 2.6f;
                t.lean = 0.17f;
                t.leanRecover = 0.3f;
                t.leader = 0.7f;
                t.lobes = 6;
                t.trunkRadius = 0.26f;
                t.wobble = 0.14f;
            }
            {
                auto& t = v[MapleFork];
                t.seed = 203;
                t.height = 19.f;
                t.crownBase = 0.36f;
                t.crownRadius = 6.2f;
                t.crownFlat = 0.8f;
                t.lobes = 7;
                t.trunkRadius = 0.27f;
                t.forkAt = 0.3f;
                t.forkAngle = 0.34f;
                t.leader = 0.62f;
                t.wobble = 0.16f;
                t.upTropism = 0.16f;
                t.leafKind = woods::Maple;
                t.leafLength = 0.2f;
                t.leaves = 32000;
                t.bark = Furrowed;
                t.barkTint = V3(0.82f, 0.82f, 0.82f);
                t.leafTint = V3(0.98f, 1.f, 1.f);
            }
            {
                auto& t = v[OakSpread];
                t.seed = 304;
                t.height = 21.f;
                t.crownBase = 0.36f;
                t.crownRadius = 7.0f;
                t.crownFlat = 0.7f;
                t.lobes = 8;
                t.trunkRadius = 0.34f;
                t.forkAt = 0.38f;
                t.forkAngle = 0.45f;
                t.leader = 0.6f;
                t.wobble = 0.26f;
                t.upTropism = 0.08f;
                t.sag = 0.04f;
                t.leafKind = woods::Oak;
                t.leafLength = 0.2f;
                t.leaves = 32000;
                t.bark = Furrowed;
                t.barkTint = V3(0.74f, 0.72f, 0.7f);
                t.leafTint = V3(0.95f, 0.97f, 1.f);
            }
            {
                auto& t = v[BeechLow];
                t.seed = 405;
                t.height = 17.f;
                t.crownBase = 0.26f;
                t.crownRadius = 6.2f;
                t.crownFlat = 0.75f;
                t.lobes = 8;
                t.trunkRadius = 0.26f;
                t.leader = 0.7f;
                t.wobble = 0.12f;
                t.upTropism = 0.1f;
                t.sag = 0.08f;
                t.leafKind = woods::Beech;
                t.leafLength = 0.16f;
                t.leaves = 34000;
                t.leavesPerShoot = 7;
                t.leafDroop = 0.15f;
                t.bark = SmoothGrey;
                t.barkTint = V3(0.95f, 0.95f, 0.95f);
                t.flare = 0.7f;
            }
            {
                auto& t = v[BirchBank];
                t.seed = 506;
                t.height = 16.f;
                t.crownBase = 0.45f;
                t.crownRadius = 4.3f;
                t.crownShift = 1.8f;
                t.crownFlat = 0.95f;
                t.lobes = 6;
                t.trunkRadius = 0.16f;
                t.lean = 0.22f;
                t.leanRecover = 0.4f;
                t.leader = 0.8f;
                t.wobble = 0.12f;
                t.upTropism = 0.12f;
                t.sag = 0.1f;
                t.gaps = 0.4f;
                t.leafKind = woods::Birch;
                t.leafLength = 0.13f;
                t.leaves = 22000;
                t.leafDroop = 0.4f;
                t.bark = BirchBark;
                t.barkTint = V3(0.95f, 0.95f, 0.93f);
                t.flare = 0.25f;
                t.twigRadius = 0.004f;
            }
            {
                auto& t = v[BasswoodTall];
                t.seed = 607;
                t.height = 23.f;
                t.crownBase = 0.42f;
                t.crownRadius = 6.0f;
                t.crownFlat = 0.9f;
                t.lobes = 7;
                t.trunkRadius = 0.3f;
                t.leader = 0.8f;
                t.wobble = 0.1f;
                t.leafKind = woods::Basswood;
                t.leafLength = 0.21f;
                t.leaves = 30000;
                t.bark = Furrowed;
                t.barkTint = V3(0.88f, 0.86f, 0.82f);
            }
            {
                auto& t = v[DogwoodUnder];
                t.seed = 708;
                t.height = 6.5f;
                t.crownBase = 0.3f;
                t.crownRadius = 3.2f;
                t.crownFlat = 0.5f;
                t.lobes = 6;
                t.trunkRadius = 0.08f;
                t.lean = 0.12f;
                t.leader = 0.55f;
                t.forkAt = 0.28f;
                t.forkAngle = 0.5f;
                t.wobble = 0.25f;
                t.upTropism = 0.04f;
                t.step = 0.3f;
                t.influence = 1.5f;
                t.kill = 0.5f;
                t.pointDensity = 5.f;
                t.leafKind = woods::Dogwood;
                t.leafLength = 0.15f;
                t.leaves = 9000;
                t.leafZone = 1.f;
                t.shootLength = 0.16f;
                t.bark = SmoothGrey;
                t.barkTint = V3(0.7f, 0.66f, 0.62f);
                t.flare = 0.2f;
                t.twigRadius = 0.004f;
            }
            {
                auto& t = v[Sapling];
                t.seed = 809;
                t.height = 4.2f;
                t.crownBase = 0.25f;
                t.crownRadius = 1.9f;
                t.crownFlat = 0.8f;
                t.lobes = 5;
                t.trunkRadius = 0.04f;
                t.lean = 0.06f;
                t.leader = 0.8f;
                t.wobble = 0.2f;
                t.upTropism = 0.1f;
                t.step = 0.25f;
                t.influence = 1.1f;
                t.kill = 0.4f;
                t.pointDensity = 8.f;
                t.leafKind = woods::Beech;
                t.leafLength = 0.14f;
                t.leaves = 4000;
                t.leafZone = 0.8f;
                t.shootLength = 0.14f;
                t.leavesPerShoot = 5;
                t.bark = SmoothGrey;
                t.flare = 0.1f;
                t.twigRadius = 0.003f;
            }
            return v;
        }

        // the leaning trunk at the left of the hero view
        inline TreeSpec framingTreeSpec() {
            TreeSpec t;
            t.seed = 977;
            t.height = 24.f;
            t.crownBase = 0.4f;
            t.crownRadius = 7.0f;
            t.crownShift = 3.6f;
            t.crownFlat = 0.8f;
            t.lobes = 8;
            t.trunkRadius = 0.32f;
            t.lean = 0.14f;
            t.leanRecover = 0.15f;
            t.wobble = 0.14f;
            t.leader = 0.66f;
            t.forkAt = 0.36f;
            t.forkAngle = 0.3f;
            t.upTropism = 0.14f;
            t.leafKind = woods::Maple;
            t.leafLength = 0.2f;
            t.leaves = 42000;
            t.bark = Furrowed;
            t.barkTint = V3(0.8f, 0.79f, 0.76f);
            t.flare = 0.6f;
            return t;
        }

        // ---- where things may stand -----------------------------------------------------------------------------------
        constexpr float heroEyeX = -3.6f, heroEyeZ = 27.5f, heroLookX = 2.8f, heroLookZ = -8.f;

        // distance from the hero view's line of sight (in plan), or a big number behind the camera or past z = 25
        inline float corridorDistance(float x, float z) {
            if (z >= 25.f) return 1e9f;
            const float dx = heroLookX - heroEyeX, dz = heroLookZ - heroEyeZ;
            const float L = std::hypot(dx, dz);
            const float ux = dx / L, uz = dz / L;
            const float rx = x - heroEyeX, rz = z - heroEyeZ;
            if (rx * ux + rz * uz < 0.f) return 1e9f;
            return std::abs(rx * uz - rz * ux);
        }

        // the mill as the hero view sees it: no trunk in the wedge from the eye to the mill's front and stream side
        // (a tree leaning across the mill would hide what the picture is of)
        inline bool inMillView(float x, float z, float margin) {
            const float ax = heroEyeX, az = heroEyeZ, bx = site::millWest - 2.f, bz = site::millNorth - 1.f, cx = site::millEast + 2.5f, cz = site::millSouth;
            // signed distances to the three edges, positive inside whichever way round the triangle runs
            const float turn = (bx - ax) * (cz - az) - (bz - az) * (cx - ax) > 0.f ? 1.f : -1.f;
            const auto inside = [&](float px, float pz, float qx, float qz) {
                const float ex = qx - px, ez = qz - pz;
                return turn * ((x - px) * ez - (z - pz) * ex) / -std::hypot(ex, ez);
            };
            return inside(ax, az, bx, bz) > -margin && inside(bx, bz, cx, cz) > -margin && inside(cx, cz, ax, az) > -margin;
        }

        inline bool nearFlume(float x, float z, float margin) {
            return std::abs(x - site::flumeX) < margin && z > site::flumeStartZ - 1.f && z < site::flumeEndZ + 1.f;
        }

        inline bool nearBridge(float x, float z, float margin) {
            return std::abs(z - site::bridgeZ) < margin && x > site::bridgeX0 - 3.f && x < site::bridgeX1 + 0.5f;
        }

        // how far outside the stream's channel (negative: in it), each bank where it really is; and anything lower
        // than the water's edge counts as in it, wherever the channel's shape puts it
        inline float bankDistance(float x, float z, float* cxOut = nullptr) {
            float cx, hw;
            site::streamAt(z, cx, hw);
            if (cxOut) *cxOut = cx;
            const float d = x - cx;
            const float e = std::abs(d) - site::bankHalfWidth(z, hw, d < 0 ? -1.f : 1.f);
            if (site::groundHeight(x, z) < site::bedAt(x, z) + 0.45f) return std::min(e, -0.1f);
            return e;
        }

        // the boulders, for keeping trunks and plants out of them
        struct RockIndex {
            float cell = 3.f;
            std::map<std::pair<int, int>, std::vector<Rock>> cells;

            explicit RockIndex(const std::vector<Rock>& rocks) {
                for (const auto& r : rocks) cells[key(r.x, r.z)].push_back(r);
            }
            [[nodiscard]] std::pair<int, int> key(float x, float z) const {
                return {static_cast<int>(std::floor(x / cell)), static_cast<int>(std::floor(z / cell))};
            }
            // true if (x, z) lies within `share` of a rock's footprint grown by `margin`, where the rock stands
            // above the ground (a rock the slope has buried at this point is no obstacle)
            [[nodiscard]] bool hits(float x, float z, float margin, float share = 1.f) const {
                const auto k = key(x, z);
                float ground = 1e9f;
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const auto it = cells.find({k.first + dx, k.second + dz});
                        if (it == cells.end()) continue;
                        for (const auto& r : it->second) {
                            if (std::hypot(x - r.x, z - r.z) >= r.r * r.elong * share + margin) continue;
                            if (ground > 1e8f) ground = site::groundHeight(x, z);
                            if (r.y + r.squash * r.r * 0.8f > ground + 0.15f) return true;
                        }
                    }
                return false;
            }
        };

        // the trunk's foot: the lowest ground under it, so no root floats on a slope
        inline float footHeight(float x, float z, float r) {
            float h = site::groundHeight(x, z);
            for (int k = 0; k < 6; ++k) {
                const float a = 1.0472f * static_cast<float>(k);
                h = std::min(h, site::groundHeight(x + r * std::cos(a), z + r * std::sin(a)));
            }
            return h;
        }

        // ---- instances ----------------------------------------------------------------------------------------------
        struct Instance {
            float x, y, z, yaw, scale;
            V3 tint;    // the leaves'
            V3 woodTint;// the bark's
        };

        // the hero view's stretch: from the camera up the stream past the mill to the bridge
        inline float coreDistance(float x, float z) {
            const float ax = heroEyeX, az = heroEyeZ, bx = 7.f, bz = -14.f;
            const float vx = bx - ax, vz = bz - az;
            const float t = std::clamp(((x - ax) * vx + (z - az) * vz) / (vx * vx + vz * vz), 0.f, 1.f);
            return std::hypot(x - ax - t * vx, z - az - t * vz);
        }

        struct Planter {
            std::shared_ptr<threepp::Group> root = threepp::Group::create();
            float tile = 24.f;
            size_t meshes = 0;

            // one instanced mesh per tile of instances (so frustum culling works per tile)
            void plant(const std::shared_ptr<threepp::BufferGeometry>& geometry, const std::shared_ptr<threepp::Material>& material,
                       const std::vector<Instance>& list, bool leafTint, bool castShadow, float tileSize = 0.f) {
                using namespace threepp;
                if (!geometry || list.empty()) return;
                const float ts = tileSize > 0 ? tileSize : tile;
                std::map<std::pair<int, int>, std::vector<const Instance*>> tiles;
                for (const auto& in : list) tiles[{static_cast<int>(std::floor(in.x / ts)), static_cast<int>(std::floor(in.z / ts))}].push_back(&in);
                for (const auto& [key, items] : tiles) {
                    auto mesh = InstancedMesh::create(geometry, material, items.size());
                    Matrix4 m, r, s;
                    std::vector<float> colours(items.size() * 3);
                    for (size_t i = 0; i < items.size(); ++i) {
                        const Instance& in = *items[i];
                        r.makeRotationY(in.yaw);
                        s.makeScale(in.scale, in.scale, in.scale);
                        m.makeTranslation(in.x, in.y, in.z).multiply(r).multiply(s);
                        mesh->setMatrixAt(i, m);
                        const V3& c = leafTint ? in.tint : in.woodTint;
                        colours[3 * i] = c.r;
                        colours[3 * i + 1] = c.g;
                        colours[3 * i + 2] = c.b;
                    }
                    mesh->setColors(0, items.size(), colours.data());
                    mesh->instanceMatrix()->needsUpdate();
                    mesh->computeBoundingSphere();
                    mesh->castShadow = castShadow;
                    mesh->receiveShadow = true;
                    root->add(mesh);
                    ++meshes;
                }
            }
        };

    }// namespace woods

    inline std::shared_ptr<threepp::Group> plantWoods(const Materials& materials) {
        (void) materials;
        using namespace threepp;
        using namespace woods;
        const auto t0 = std::chrono::steady_clock::now();
        const auto seconds = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };

        // ---- surfaces ----------------------------------------------------------------------------------------------
        const TexSet atlas = foliageAtlas(17);
        const std::array<TexSet, 3> barks{woodsBark(Furrowed, 3), woodsBark(SmoothGrey, 4), woodsBark(BirchBark, 5)};
        auto leafMat = atlas.material(0.5f);
        leafMat->alphaTest = 0.5f;
        leafMat->side = Side::Double;
        leafMat->vertexColors = true;
        leafMat->translucency = 0.35f;
        leafMat->translucencyColor = Color(0.75f, 0.92f, 0.32f);
        leafMat->aoMap = nullptr;
        std::array<std::shared_ptr<MeshStandardMaterial>, 3> barkMat;
        for (int k = 0; k < 3; ++k) {
            barkMat[k] = barks[k].material(1.f);
            barkMat[k]->vertexColors = true;
        }
        auto greenMat = MeshStandardMaterial::create();
        greenMat->vertexColors = true;
        greenMat->side = Side::Double;
        greenMat->roughness = 0.78f;
        greenMat->metalness = 0.f;
        const double tTex = seconds(t0);

        // ---- grow every kind once, in parallel ---------------------------------------------------------------------
        const auto specs = treeSpecs();
        std::vector<TreeMeshes> trees(specs.size());
        TreeMeshes framing;
        std::array<ShrubSpec, 3> shrubSpecs;
        for (int k = 0; k < 3; ++k) {
            shrubSpecs[k].seed = 900 + k * 17;
            shrubSpecs[k].height = 2.0f + 0.5f * static_cast<float>(k);
            shrubSpecs[k].stems = 4 + k;
            shrubSpecs[k].splay = 0.5f + 0.15f * static_cast<float>(k);
            shrubSpecs[k].leafTint = V3(1.f, 1.f, 1.f) * (0.95f + 0.05f * static_cast<float>(k));
        }
        std::array<PlantMeshes, 3> shrubs;
        std::array<FernSpec, 3> fernSpecs;
        fernSpecs[0] = {1201, FernLady, 9, 0.8f, 1.05f, 1.25f, V3(1, 1, 1)};
        fernSpecs[1] = {1202, FernLady, 7, 0.6f, 0.9f, 1.0f, V3(0.95f, 1.f, 0.95f)};
        fernSpecs[2] = {1203, FernXmas, 11, 0.55f, 0.75f, 1.0f, V3(1, 1, 1)};
        std::array<PlantMeshes, 3> ferns;
        std::array<GrassSpec, 2> grassSpecs{GrassSpec{1301, 30, 0.42f, 0.009f, 0.15f}, GrassSpec{1302, 22, 0.3f, 0.006f, 0.05f}};
        std::array<PlantMeshes, 2> grasses;
        std::array<HerbSpec, 2> herbSpecs{HerbSpec{1401, 4, 0.65f, Dogwood, 0.11f, V3(1.05f, 1.08f, 1.f)}, HerbSpec{1402, 6, 0.4f, Basswood, 0.09f, V3(1.f, 1.05f, 1.f)}};
        std::array<PlantMeshes, 2> herbs;

        std::vector<std::function<void()>> jobs;
        jobs.emplace_back([&] { framing = growTree(framingTreeSpec()); });
        for (size_t i = 0; i < specs.size(); ++i) jobs.emplace_back([&, i] { trees[i] = growTree(specs[i]); });
        for (size_t i = 0; i < shrubs.size(); ++i) jobs.emplace_back([&, i] { shrubs[i] = growShrub(shrubSpecs[i]); });
        for (size_t i = 0; i < ferns.size(); ++i) jobs.emplace_back([&, i] { ferns[i] = growFern(fernSpecs[i]); });
        for (size_t i = 0; i < grasses.size(); ++i) jobs.emplace_back([&, i] { grasses[i] = growGrass(grassSpecs[i]); });
        for (size_t i = 0; i < herbs.size(); ++i) jobs.emplace_back([&, i] { herbs[i] = growHerb(herbSpecs[i]); });
        ThreadPool::shared().parallelFor(0, static_cast<int>(jobs.size()), [&](int i) { jobs[static_cast<size_t>(i)](); }, 1);

        // ---- where the trees stand ---------------------------------------------------------------------------------
        const RockIndex rocks(placeRocks());
        // [variant][lod]
        std::vector<std::array<std::vector<Instance>, 3>> placed(specs.size());
        std::vector<std::pair<float, float>> trunks;// for spacing the understory
        Rng rng(4242);
        const auto tint = [&](float hueAmount) {
            const float h = rng.range(-1.f, 1.f), b = rng.range(0.86f, 1.08f);
            return V3(1.f + 0.07f * h * hueAmount, 1.f + 0.02f * h * hueAmount, 1.f - 0.12f * h * hueAmount) * b;
        };
        const auto lodOf = [](float x, float z) {
            const float d = coreDistance(x, z);
            int lod = d < 16.f ? 0 : (d < 55.f ? 1 : 2);
            // behind the camera: only their shadows reach the view
            if (z > heroEyeZ + 3.f && lod == 0) lod = 1;
            return lod;
        };
        const auto trunkClear = [&](float x, float z, float millMargin, float corridor) {
            if (site::inMill(x, z, millMargin)) return false;
            if (nearFlume(x, z, 1.8f)) return false;
            if (nearBridge(x, z, 2.5f)) return false;
            if (corridorDistance(x, z) < corridor) return false;
            if (inMillView(x, z, 3.f)) return false;
            // along the reach the views look down, the creek is a gap open to the sky, as it is at the mill
            const float openReach = (z > -28.f && z < 62.f) ? 2.6f : 0.6f;
            if (bankDistance(x, z) < openReach) return false;
            if (rocks.hits(x, z, 0.2f)) return false;
            return true;
        };
        // the framing tree's place
        // (on the left bank, clear of the water however the bank wanders there)
        const float fz = 19.0f;
        const float fx = [&] {
            float cx, hw;
            site::streamAt(fz, cx, hw);
            float x = std::min(-7.f, cx - site::bankHalfWidth(fz, hw, -1.f) - 0.9f);
            while (bankDistance(x, fz) < 0.6f) x -= 0.25f;
            return x;
        }();
        trunks.emplace_back(fx, fz);
        // the canopy: a jittered grid over the whole valley
        const float cell = 6.3f;
        for (float gz = -150.f; gz < 75.f; gz += cell) {
            for (float gx = -92.f; gx < 92.f; gx += cell) {
                const float x = gx + rng.range(0.1f, 0.9f) * cell, z = gz + rng.range(0.1f, 0.9f) * cell;
                if (x < -92.f || x > 92.f || z < -148.f || z > 73.f) continue;
                if (rng() < 0.06f) continue;
                if (!trunkClear(x, z, 4.5f, 3.f)) continue;
                if (std::hypot(x - fx, z - fz) < 4.5f) continue;
                // far off, fewer and bigger trees: the same cover for fewer crowns to draw
                const bool far = coreDistance(x, z) > 60.f;
                if (far && rng() < 0.33f) continue;
                float cx;
                const float e = bankDistance(x, z, &cx);
                // what grows where: leaning trees on the banks, the big hardwoods on the slopes
                int v;
                const float u = rng();
                if (e < 4.f) v = u < 0.35f ? TulipBank : (u < 0.62f ? BirchBank : (u < 0.82f ? BeechLow : MapleFork));
                else if (e < 26.f) v = u < 0.3f ? TulipTall : (u < 0.52f ? MapleFork : (u < 0.7f ? BeechLow : (u < 0.86f ? OakSpread : BasswoodTall)));
                else v = u < 0.36f ? TulipTall : (u < 0.62f ? OakSpread : (u < 0.8f ? MapleFork : BasswoodTall));
                // the leaning ones lean toward the stream; the rest stand any way round
                float yaw = rng() * 6.2831853f;
                if (v == TulipBank || v == BirchBank) {
                    yaw = (cx > x ? 0.f : 3.14159265f) + rng.range(-0.5f, 0.5f);
                }
                Instance in{};
                in.x = x;
                in.z = z;
                in.scale = rng.range(0.85f, 1.15f) * (far ? 1.2f : 1.f);
                in.y = footHeight(x, z, specs[v].trunkRadius * in.scale * 1.4f) - 0.05f;
                in.yaw = yaw;
                in.tint = tint(1.f);
                in.woodTint = V3(1, 1, 1) * rng.range(0.85f, 1.1f);
                placed[v][lodOf(x, z)].push_back(in);
                trunks.emplace_back(x, z);
            }
        }
        // the banks: trees at the water's edge leaning out over it, both sides
        for (float z0 = -120.f; z0 < 70.f; z0 += 6.5f) {
            for (int sideK = 0; sideK < 2; ++sideK) {
                const float z = z0 + rng.range(0.f, 6.f);
                float cx, hw;
                site::streamAt(z, cx, hw);
                const float side = sideK == 0 ? -1.f : 1.f;
                const bool reach = z > -28.f && z < 62.f;
                const float x = cx + side * (hw + (reach ? rng.range(2.7f, 4.5f) : rng.range(0.7f, 3.2f)));
                if (rng() < (reach ? 0.6f : 0.15f) || !trunkClear(x, z, 4.5f, 3.f)) continue;
                if (std::hypot(x - fx, z - fz) < 5.f) continue;
                bool crowded = false;
                for (const auto& t : trunks)
                    if (std::hypot(x - t.first, z - t.second) < 3.6f) crowded = true;
                if (crowded) continue;
                const float u = rng();
                const int v = u < 0.4f ? TulipBank : (u < 0.75f ? BirchBank : BeechLow);
                Instance in{};
                in.x = x;
                in.z = z;
                in.scale = rng.range(0.85f, 1.12f);
                in.y = footHeight(x, z, specs[v].trunkRadius * in.scale * 1.4f) - 0.05f;
                in.yaw = (side < 0 ? 0.f : 3.14159265f) + rng.range(-0.45f, 0.45f);
                in.tint = tint(1.f);
                in.woodTint = V3(1, 1, 1) * rng.range(0.85f, 1.1f);
                placed[v][lodOf(x, z)].push_back(in);
                trunks.emplace_back(x, z);
            }
        }
        // the understory: small trees between the big ones, near the view only
        for (float gz = -70.f; gz < 60.f; gz += 5.f) {
            for (float gx = -45.f; gx < 45.f; gx += 5.f) {
                const float x = gx + rng.range(0.f, 5.f), z = gz + rng.range(0.f, 5.f);
                if (coreDistance(x, z) > 46.f || rng() < 0.35f) continue;
                if (!trunkClear(x, z, 3.f, 3.f)) continue;
                bool crowded = false;
                for (const auto& t : trunks)
                    if (std::hypot(x - t.first, z - t.second) < 2.2f) crowded = true;
                if (crowded) continue;
                const int v = rng() < 0.45f ? DogwoodUnder : Sapling;
                Instance in{};
                in.x = x;
                in.z = z;
                in.scale = rng.range(0.75f, 1.25f);
                in.y = footHeight(x, z, 0.1f) - 0.05f;
                in.yaw = rng() * 6.2831853f;
                in.tint = tint(1.f) * 1.04f;
                in.woodTint = V3(1, 1, 1) * rng.range(0.85f, 1.1f);
                placed[v][lodOf(x, z)].push_back(in);
            }
        }
        // further off, the understory goes on through the whole valley, a little bigger and as dense, so between the
        // far trunks and the ground there is leaf, not bare slope and sky
        for (float gz = -150.f; gz < 75.f; gz += 6.5f) {
            for (float gx = -90.f; gx < 90.f; gx += 6.5f) {
                const float x = gx + rng.range(0.f, 6.5f), z = gz + rng.range(0.f, 6.5f);
                // past about 95 m the canopy and the ridges behind close the view anyway
                if (coreDistance(x, z) <= 46.f || coreDistance(x, z) > 95.f || rng() < 0.3f) continue;
                if (!trunkClear(x, z, 3.f, 3.f)) continue;
                const int v = rng() < 0.4f ? DogwoodUnder : Sapling;
                Instance in{};
                in.x = x;
                in.z = z;
                in.scale = rng.range(1.1f, 1.6f);
                in.y = footHeight(x, z, 0.1f) - 0.05f;
                in.yaw = rng() * 6.2831853f;
                in.tint = tint(1.f) * 0.96f;
                in.woodTint = V3(1, 1, 1) * rng.range(0.85f, 1.1f);
                placed[v][lodOf(x, z)].push_back(in);
            }
        }

        // ---- the undergrowth ---------------------------------------------------------------------------------------
        std::array<std::vector<Instance>, 3> shrubAt, fernAt;
        std::array<std::vector<Instance>, 2> grassAt, herbAt;
        const auto groundClear = [&](float x, float z, float rockMargin, float rockShare) {
            // the other set views' eyes (side, bridge)
            if (std::hypot(x + 14.f, z - 2.f) < 3.5f || std::hypot(x + 6.f, z - 2.f) < 3.f) return false;
            if (site::inMill(x, z, 2.6f)) return false;// crowns reach about 2 m: none through the walls
            if (nearFlume(x, z, 1.3f)) return false;
            if (nearBridge(x, z, 1.4f) && std::abs(x - site::bridgeX0) > 2.f) return false;
            if (bankDistance(x, z) < 0.3f) return false;
            if (rocks.hits(x, z, rockMargin, rockShare)) return false;
            return true;
        };
        // rhododendron: the right bank's outcrop and the slope behind the mill, thinly further up the right bank
        const auto inThicket = [](float x, float z) {
            if (x > 8.f && x < 19.f && z > 5.f && z < 31.f) return 1.f;
            if (x > 13.f && x < 21.f && z > -11.f && z < 11.f) return 1.f;
            // the valley sides upstream, which the views up the creek look onto: thickets, broken by open patches
            if (x > 8.f && x < 36.f && z > -75.f && z < -12.f) return 0.8f;
            if (x < -9.f && x > -34.f && z > -75.f && z < 15.f) return 0.5f;
            return 0.f;
        };
        for (float z = -75.f; z < 33.f; z += 1.45f) {
            for (float x = -34.f; x < 36.f; x += 1.45f) {
                const float px = x + rng.range(-0.6f, 0.6f), pz = z + rng.range(-0.6f, 0.6f);
                const float w = inThicket(px, pz) * smooth(-0.7f, 0.05f, fbm2(px * 0.13f, pz * 0.13f, 2, 61u));
                if (rng() > w) continue;
                if (!groundClear(px, pz, 0.f, 0.55f)) continue;
                if (corridorDistance(px, pz) < 4.f) continue;
                Instance in{};
                in.x = px;
                in.z = pz;
                in.scale = rng.range(0.9f, 1.4f);
                in.y = footHeight(px, pz, 0.3f);
                in.yaw = rng() * 6.2831853f;
                in.tint = V3(1, 1, 1) * rng.range(0.85f, 1.1f);
                in.woodTint = V3(1, 1, 1);
                shrubAt[rng.next() % 3].push_back(in);
            }
        }
        // ferns, herbs and grass: in patches on the banks and under the trees near the view, grass where it is open
        for (float z = -60.f; z < 40.f; z += 1.25f) {
            for (float x = -40.f; x < 40.f; x += 1.25f) {
                const float px = x + rng.range(-0.55f, 0.55f), pz = z + rng.range(-0.55f, 0.55f);
                const float d = coreDistance(px, pz);
                if (d > 40.f) continue;
                const float e = bankDistance(px, pz);
                if (e < 0.3f) continue;
                const float patch = fbm2(px * 0.09f, pz * 0.09f, 3, 71u);
                const float patch2 = fbm2(px * 0.15f + 7.f, pz * 0.15f, 2, 72u);
                const float bank = smooth(6.f, 1.f, e);
                const float fade = smooth(40.f, 25.f, d);
                const float u = rng();
                // the clearings at the bridge's ends: grass
                const float clearing = std::max(smooth(7.f, 2.f, std::hypot(px - (site::bridgeX0 - 2.f), pz - site::bridgeZ)),
                                                smooth(5.f, 1.5f, std::hypot(px - (site::bridgeX1 + 1.f), pz - (site::bridgeZ - 2.5f))));
                const float pFern = fade * (0.1f + 0.3f * bank) * smooth(-0.25f, 0.3f, patch) * (1.f - clearing);
                const float pHerb = fade * (0.12f + 0.45f * bank) * smooth(-0.35f, 0.2f, patch2) * (1.f - clearing) * (px < 0.f ? 1.f : 0.6f);
                const float pGrass = fade * (0.05f + 0.25f * bank * smooth(0.f, 0.4f, -patch)) + 0.85f * clearing;
                int what = -1;
                if (u < pFern) what = 0;
                else if (u < pFern + pHerb) what = 1;
                else if (u < pFern + pHerb + pGrass) what = 2;
                if (what < 0) continue;
                if (!groundClear(px, pz, 0.f, what == 2 ? 0.7f : 0.85f)) continue;
                // keep the camera's foot clear, and the line of sight
                if (std::hypot(px - heroEyeX, pz - heroEyeZ) < 3.f) continue;
                if (what != 2 && corridorDistance(px, pz) < 1.5f && pz < 20.f) continue;
                if (inThicket(px, pz) >= 1.f && what != 2) continue;
                Instance in{};
                in.x = px;
                in.z = pz;
                in.y = site::groundHeight(px, pz);
                in.yaw = rng() * 6.2831853f;
                in.tint = V3(1, 1, 1) * rng.range(0.85f, 1.1f);
                in.woodTint = V3(1, 1, 1) * rng.range(0.9f, 1.1f);
                if (what == 0) {
                    in.scale = rng.range(0.7f, 1.2f);
                    fernAt[(patch2 > 0.15f ? 2 : rng.next() % 2)].push_back(in);
                } else if (what == 1) {
                    in.scale = rng.range(0.8f, 1.25f);
                    herbAt[rng.next() % 2].push_back(in);
                } else {
                    in.scale = rng.range(0.7f, 1.3f);
                    grassAt[rng.next() % 2].push_back(in);
                }
            }
        }

        // ---- the meshes ---------------------------------------------------------------------------------------------
        Planter planter;
        std::array<size_t, 3> lodTris{0, 0, 0}, lodCount{0, 0, 0};
        size_t heroTris = 0, allTris = 0;
        // an estimate of what the hero view and the sun's shadow camera draw
        auto heroCam = PerspectiveCamera::create(50.f, 1.5f, 0.05f, 2000.f);
        heroCam->position.set(heroEyeX, 1.45f, heroEyeZ);
        heroCam->lookAt(Vector3(heroLookX, 4.2f, heroLookZ));
        heroCam->updateMatrixWorld();
        Frustum heroFrustum;
        heroFrustum.setFromProjectionMatrix(Matrix4().multiplyMatrices(heroCam->projectionMatrix, heroCam->matrixWorldInverse));
        const auto inHero = [&](float x, float y, float z, float r) { return heroFrustum.intersectsSphere(Sphere(Vector3(x, y, z), r)); };

        for (size_t v = 0; v < specs.size(); ++v) {
            const auto& bark = barkMat[static_cast<size_t>(specs[v].bark)];
            for (int lod = 0; lod < 3; ++lod) {
                const auto& list = placed[v][lod];
                if (list.empty()) continue;
                const auto wood = trees[v].wood[lod].geometry();
                const auto leaves = trees[v].leaves[lod].geometry();
                planter.plant(wood, bark, list, false, true);
                planter.plant(leaves, leafMat, list, true, true);
                const size_t tris = trees[v].wood[lod].triangles() + trees[v].leaves[lod].triangles();
                lodTris[lod] += tris * list.size();
                lodCount[lod] += list.size();
                allTris += tris * list.size();
                for (const auto& in : list)
                    if (inHero(in.x, in.y + 0.6f * specs[v].height * in.scale, in.z, std::max(specs[v].crownRadius + 2.f, 0.5f * specs[v].height) * in.scale)) heroTris += tris;
            }
        }
        {
            // the framing tree, grown for its place: leaning toward the stream and a little up it
            auto wood = Mesh::create(framing.wood[0].geometry(), barkMat[Furrowed]);
            auto leaves = Mesh::create(framing.leaves[0].geometry(), leafMat);
            for (auto* m : {wood.get(), leaves.get()}) {
                m->position.set(fx, footHeight(fx, fz, 0.5f) - 0.05f, fz);
                m->rotation.y = 0.38f;
                m->castShadow = true;
                m->receiveShadow = true;
            }
            planter.root->add(wood);
            planter.root->add(leaves);
            const size_t tris = framing.wood[0].triangles() + framing.leaves[0].triangles();
            heroTris += tris;
            allTris += tris;
        }
        size_t plantTris = 0;
        const auto plantSet = [&](const PlantMeshes& pm, const std::vector<Instance>& list, bool shadows) {
            if (list.empty()) return;
            if (!pm.wood.empty()) planter.plant(pm.wood.geometry(), barkMat[Furrowed], list, false, shadows, 16.f);
            if (!pm.leaves.empty()) planter.plant(pm.leaves.geometry(), leafMat, list, true, shadows, 16.f);
            if (!pm.green.empty()) planter.plant(pm.green.geometry(), greenMat, list, true, shadows, 16.f);
            const size_t tris = pm.wood.triangles() + pm.leaves.triangles() + pm.green.triangles();
            plantTris += tris * list.size();
            for (const auto& in : list)
                if (inHero(in.x, in.y + 1.f, in.z, 2.f)) heroTris += tris;
        };
        size_t nShrubs = 0, nFerns = 0, nGrass = 0, nHerbs = 0;
        for (size_t k = 0; k < shrubs.size(); ++k) plantSet(shrubs[k], shrubAt[k], true), nShrubs += shrubAt[k].size();
        for (size_t k = 0; k < ferns.size(); ++k) plantSet(ferns[k], fernAt[k], true), nFerns += fernAt[k].size();
        for (size_t k = 0; k < grasses.size(); ++k) plantSet(grasses[k], grassAt[k], false), nGrass += grassAt[k].size();
        for (size_t k = 0; k < herbs.size(); ++k) plantSet(herbs[k], herbAt[k], true), nHerbs += herbAt[k].size();
        allTris += plantTris;

        std::printf("woods: framing tree at x %.2f; %zu/%zu/%zu trees near/mid/far + 1 framing, %zu rhododendrons, %zu ferns, %zu herbs, %zu grass tufts; "
                    "%.1f M triangles in all, about %.1f M in the hero view; %zu meshes; %.2f s (textures %.2f s)\n",
                    static_cast<double>(fx), lodCount[0], lodCount[1], lodCount[2], nShrubs, nFerns, nHerbs, nGrass, static_cast<double>(allTris) * 1e-6,
                    static_cast<double>(heroTris) * 1e-6, planter.meshes, seconds(t0), tTex);
        return planter.root;
    }

}// namespace mill

#endif
