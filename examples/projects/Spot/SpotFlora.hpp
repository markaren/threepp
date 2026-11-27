// What grows round spot_play's course: the forest that closes over the hills
// round the meadow, so the course reads as a clearing rather than a bowl of
// bare hills against the sky.
//
// The trees are the card trees from threepp_data
// (models/fjord/trees): Poly Haven's scanned fir_tree_01, pine_tree_01 and
// tree_small_02 (CC0), rebuilt as trunks and leaf cards of the real needles
// and leaves, read by examples/vulkan/fjord_forest.hpp. They are visual only:
// nothing here is a PhysX static, so the policy's height scan never sees them.
//
// Where they stand follows the rule the fjord demo took from its reference
// photographs: forest closes over the slopes, thins at a ragged edge, and
// birch stands at the margin with the conifers behind it.

#ifndef THREEPP_EXAMPLES_SPOT_FLORA_HPP
#define THREEPP_EXAMPLES_SPOT_FLORA_HPP

#include "SpotCourse.hpp"

#include "../../vulkan/fjord_forest.hpp"

#include "threepp/objects/InstancedMesh.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace spot::flora {

    // ── The forest ──────────────────────────────────────────────────────────
    // Plants the forest under `root` and shades the ground under it for a sun in
    // `toSun` (unit, toward the sun). Returns the number of trees.
    //
    // The trees cast no shadow-map shadow: that map is a box that follows Spot, and a
    // crown's shadow ended in a straight line at its edge. The sky's sun never moves,
    // so the crowns' shade is baked into the ground instead, everywhere at once: each
    // crown an ellipsoid round its leaf cards, the ground darkened by how much crown
    // the sun's ray crosses on its way down, so the shade is dappled at a crown's rim
    // and deep under its middle.
    inline int plantForest(threepp::Object3D& root, course::Course& land, const std::string& data,
                           const threepp::Vector3& toSun) {
        using namespace threepp;
        using course::fbm;
        using course::smoothstep;

        const std::filesystem::path dir = std::filesystem::path(data) / "models" / "fjord" / "trees";
        // Slots 0 spruce, 1 pine, 2 birch. Bark tiling and the leaf cut-off (0.4) as the
        // twin's kit has them: the atlases pad every card with alpha just under 0.4, so a
        // lower cut-off shows the cards as rectangles.
        const std::array<vegetation::SpeciesVariants, 3> species{
                fjord::loadKitSpecies(dir, fjord::KitLook{"spruce", {1.2f, 0.1f}}),
                fjord::loadKitSpecies(dir, fjord::KitLook{"pine", {1.2f, 0.1f}}),
                fjord::loadKitSpecies(dir, fjord::KitLook{"birch", {1.f, 1.f}}),
        };
        // Heights in metres (min, max): a grown spruce and pine stand, younger birch at the edge.
        constexpr std::array<std::array<float, 2>, 3> HEIGHT{{{11.f, 19.f}, {10.f, 16.f}, {6.f, 11.f}}};

        // The clearing: the meadow and its hills are the same bowl the terrain's rim is
        // (centre (12, 0), y stretched by 1.1). The forest edge wanders through 23-30 m.
        const auto bowl = [](float x, float y) { return std::hypot(x - 12.f, y * 1.1f); };
        const auto edgeAt = [](float x, float y) { return 26.5f + 3.5f * fbm(x / 11.f, y / 11.f, 3, 91u); };

        struct Planted {
            int species, variant;
            Matrix4 m;
            Vector3 crown, radii;// the crown's ellipsoid in the world: centre, half-axes (x = y, z)
        };
        std::vector<Planted> trees;
        course::Rand rnd(20261117u);
        const auto onStone = [&](float x, float y) {
            for (const auto& s : land.stones()) {
                if (std::hypot(x - s.x, y - s.y) < s.z + 0.8f) return true;
            }
            return false;
        };
        const auto plant = [&](int s, int variant, float x, float y, float size, float yaw) {
            const auto& sv = species[static_cast<std::size_t>(s)];
            if (sv.near.empty()) return;
            const int v = variant % static_cast<int>(sv.near.size());
            const float h = HEIGHT[s][0] + (HEIGHT[s][1] - HEIGHT[s][0]) * size;
            const float scale = h / sv.near[static_cast<std::size_t>(v)].height;
            // Rooted a little into the ground, so the downhill side of the trunk does not float.
            const float z = land.groundAt(x, y) - 0.25f;
            Quaternion q;
            q.setFromAxisAngle(Vector3(0, 0, 1), yaw);
            Matrix4 m;
            m.compose(Vector3(x, y, z), q, Vector3(scale, scale, scale));
            // The leaf cards' box (kit y-up) as the crown: a little narrower than the
            // outermost card, which is mostly air.
            const auto& bb = *sv.near[static_cast<std::size_t>(v)].leafGeo->boundingBox;
            const float rh = 0.4f * std::max(bb.max().x - bb.min().x, bb.max().z - bb.min().z) * scale;
            const float rv = 0.5f * (bb.max().y - bb.min().y) * scale;
            trees.push_back({s, v, m, Vector3(x, y, z + 0.5f * (bb.min().y + bb.max().y) * scale), Vector3(rh, rh, rv)});
        };

        // The forest round the clearing.
        constexpr float SPACING = 4.2f;
        for (float gy = course::Y0 + 1.f; gy < course::Y0 + course::SIZE - 1.f; gy += SPACING) {
            for (float gx = course::X0 + 1.f; gx < course::X0 + course::SIZE - 1.f; gx += SPACING) {
                const float x = gx + rnd.range(0.f, SPACING), y = gy + rnd.range(0.f, SPACING);
                const float keep = rnd.next(), pick = rnd.next(), size = rnd.next(), yaw = rnd.range(0.f, 2 * math::PI);
                const int variant = static_cast<int>(rnd.next() * 3.f) % 3;
                if (x > course::X0 + course::SIZE - 1.f || y > course::Y0 + course::SIZE - 1.f) continue;

                const float into = bowl(x, y) - edgeAt(x, y);// metres past the forest's edge
                const float density = 0.06f + 0.84f * smoothstep(0.f, 7.f, into);
                if (into < -2.f || keep > density) continue;
                if (land.distanceToPath(x, y) < 3.f || onStone(x, y)) continue;

                // Birch at the margin, conifers behind it; spruce the commoner of the two.
                const float birch = 0.75f * (1.f - smoothstep(0.f, 6.f, into));
                plant(pick < birch ? 2 : (pick < birch + (1.f - birch) * 0.3f ? 1 : 0), variant, x, y, size, yaw);
            }
        }

        // A few trees standing alone in the clearing, birch and the odd pine, so trees and
        // their shade are in the frames the follow camera sees, not only on the skyline.
        // Off the path, the terrace and its stair, out of the opening view, and well apart.
        std::vector<Vector2> alone;
        for (int tries = 0; alone.size() < 7 && tries < 4000; ++tries) {
            const float a = rnd.range(0.f, 2 * math::PI), r = rnd.range(7.f, 21.f);
            const float x = 12.f + std::cos(a) * r, y = std::sin(a) * r / 1.1f;
            const float pick = rnd.next(), size = rnd.next(), yaw = rnd.range(0.f, 2 * math::PI);
            const int variant = static_cast<int>(rnd.next() * 3.f) % 3;
            if (land.distanceToPath(x, y) < 4.5f || onStone(x, y)) continue;
            if (x > course::STAIR_X0 - 3.f && x < course::SLOPE_END + 2.f && std::abs(y) < course::TERRACE_HALF_Y + course::TERRACE_SIDE) continue;
            if (x < course::STAIR_X0 && std::abs(y) < 7.f) continue;// the view from the start to the stair stays open
            bool apart = true;
            for (const auto& p : alone) {
                if (std::hypot(x - p.x, y - p.y) < 9.f) apart = false;
            }
            if (!apart) continue;
            alone.emplace_back(x, y);
            plant(pick < 0.7f ? 2 : 1, variant, x, y, 0.4f + 0.6f * size, yaw);
        }

        // One instanced mesh per species, variant and part; the kit is y-up, this world z-up.
        Matrix4 zUp;
        zUp.makeRotationX(math::PI / 2);
        for (int s = 0; s < 3; ++s) {
            const auto& sv = species[static_cast<std::size_t>(s)];
            for (int v = 0; v < static_cast<int>(sv.near.size()); ++v) {
                std::vector<Matrix4> at;
                for (const auto& t : trees) {
                    if (t.species == s && t.variant == v) at.push_back(t.m);
                }
                if (at.empty()) continue;
                const auto& tv = sv.near[static_cast<std::size_t>(v)];
                for (const auto& [geo, mat] : {std::pair{tv.leafGeo, tv.leafMat}, std::pair{tv.trunkGeo, tv.barkMat}}) {
                    if (!geo || !mat) continue;
                    auto g = geo->clone();
                    g->applyMatrix4(zUp);
                    auto mesh = InstancedMesh::create(g, mat, static_cast<int>(at.size()));
                    for (std::size_t i = 0; i < at.size(); ++i) mesh->setMatrixAt(static_cast<int>(i), at[i]);
                    mesh->instanceMatrix()->needsUpdate();
                    mesh->receiveShadow = true;
                    mesh->frustumCulled = false;// the instances span the whole ring
                    root.add(mesh);
                }
            }
        }

        // The crowns' shade on the ground: the length of crown each ground sample's ray to
        // the sun crosses, turned into a darkening that saturates at a crown's heart near
        // what the shadow map gives Spot's own shadow (the sun gone, the sky left).
        using course::N;
        std::vector<float> depth(static_cast<std::size_t>(N) * N, 0.f);
        const Vector3 d = Vector3(toSun).normalize();
        const float lean = std::hypot(d.x, d.y) / std::max(d.z, 0.1f);// horizontal reach per metre of height
        for (const auto& t : trees) {
            const float g = land.groundAt(t.crown.x, t.crown.y);
            const float reach = t.radii.x + t.radii.z * lean + 0.5f;// the ellipsoid's shadow round its centre
            const float cx = t.crown.x - d.x / std::max(d.z, 0.1f) * (t.crown.z - g);// the shadow's centre
            const float cy = t.crown.y - d.y / std::max(d.z, 0.1f) * (t.crown.z - g);
            const int i0 = std::max(0, static_cast<int>((cx - reach - course::X0) / course::CELL));
            const int i1 = std::min(N - 1, static_cast<int>((cx + reach - course::X0) / course::CELL) + 1);
            const int j0 = std::max(0, static_cast<int>((cy - reach - course::Y0) / course::CELL));
            const int j1 = std::min(N - 1, static_cast<int>((cy + reach - course::Y0) / course::CELL) + 1);
            for (int j = j0; j <= j1; ++j) {
                for (int i = i0; i <= i1; ++i) {
                    const float px = course::X0 + i * course::CELL, py = course::Y0 + j * course::CELL;
                    // The ray from the ground toward the sun against the crown's ellipsoid, in
                    // the ellipsoid's unit-sphere space.
                    const Vector3 o((px - t.crown.x) / t.radii.x, (py - t.crown.y) / t.radii.y,
                                    (land.groundAt(px, py) - t.crown.z) / t.radii.z);
                    const Vector3 r(d.x / t.radii.x, d.y / t.radii.y, d.z / t.radii.z);
                    const float a = r.dot(r), b = o.dot(r), c = o.dot(o) - 1.f;
                    const float disc = b * b - a * c;
                    if (disc <= 0.f) continue;
                    const float sq = std::sqrt(disc);
                    const float t1 = std::max((-b - sq) / a, 0.f), t2 = (-b + sq) / a;
                    if (t2 > t1) depth[static_cast<std::size_t>(j) * N + i] += t2 - t1;// metres, d is unit
                }
            }
        }
        // A crown is clumps and gaps, not a solid: the light through it is mottled, from
        // the ground sample's half metre up to a few metres.
        std::vector<float> shade(depth.size());
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                const std::size_t k = static_cast<std::size_t>(j) * N + i;
                if (depth[k] <= 0.f) {
                    shade[k] = 1.f;
                    continue;
                }
                const float x = course::X0 + i * course::CELL, y = course::Y0 + j * course::CELL;
                const float gaps = std::clamp(0.5f + 0.95f * fbm(x / 1.3f, y / 1.3f, 3, 97u), 0.f, 1.f);
                shade[k] = 1.f - 0.58f * (1.f - std::exp(-0.45f * depth[k] * gaps));
            }
        }
        land.shadeGround(shade);
        return static_cast<int>(trees.size());
    }

}// namespace spot::flora

#endif// THREEPP_EXAMPLES_SPOT_FLORA_HPP
