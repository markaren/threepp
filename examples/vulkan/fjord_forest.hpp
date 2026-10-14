// FJORD FOREST — the kit trees (fjord_trees.hpp) as CanopyForest prototypes,
// and the site scatter that decides where a west-Norwegian valley grows them.
//
// Placement is a RULE here, not a measurement: the twins read a lidar canopy
// height model, this valley is generated, so the forest is generated with it.
// The rules are the ones the reference photographs show (Nærøyfjord,
// Aurlandsfjord, Geiranger): forest closes over every slope gentle enough to
// hold soil, thins to scattered pines on the ledges of the walls, gives up on
// sheer rock, and stops at a ragged treeline with mountain birch last.

#ifndef THREEPP_FJORD_FOREST_HPP
#define THREEPP_FJORD_FOREST_HPP

#include "fjord_trees.hpp"

#include "threepp/extras/vegetation/CanopyForest.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"
#include "threepp/objects/Mesh.hpp"

#include <array>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace fjord {

    // How a species' kit is drawn. The numbers are the twin's materials
    // (norvasundet kit manifest): alpha cutoff, translucency, bark tiling.
    struct KitLook {
        std::string name;                 // file stem in threepp_data/models/fjord/trees
        threepp::Vector2 barkRepeat{1, 1};// the trunk material's base_color_tiling
        threepp::Color limbColor{0.42f, 0.36f, 0.31f};// wood1 (limbs): see foldLimbs
        float alphaTest = 0.4f;
        float translucency = 0.5f;
        threepp::Color leafTint{1, 1, 1};// multiplies the atlas (per-species colour grade)
    };

    namespace detail {

        inline std::shared_ptr<threepp::BufferGeometry> foldLimbs(
                const std::shared_ptr<threepp::BufferGeometry>& trunk,
                const std::shared_ptr<threepp::BufferGeometry>& limbs, const KitLook& look) {
            using namespace threepp;
            // Every vertex carries a colour the bark multiplies: white on the
            // trunk, the limbs' own wood colour on the limbs. The kit's limbs
            // have no unwrap (every UV is (0,1)), so in the twin they render as
            // one colour anyway: the mean of their wood texture, which is this.
            const auto paint = [](const std::shared_ptr<BufferGeometry>& g, const Color& c) {
                auto out = g->clone();
                const auto n = static_cast<size_t>(out->getAttribute<float>("position")->count());
                std::vector<float> col(n * 3);
                for (size_t i = 0; i < n; ++i) {
                    col[i * 3] = c.r;
                    col[i * 3 + 1] = c.g;
                    col[i * 3 + 2] = c.b;
                }
                out->setAttribute("color", FloatBufferAttribute::create(col, 3));
                return out;
            };
            auto t = paint(trunk, Color(1.f, 1.f, 1.f));
            if (!limbs) return t;
            auto l = paint(limbs, look.limbColor);
            if (auto* uv = l->getAttribute<float>("uv")) {
                auto& a = uv->array();
                for (size_t i = 0; i + 1 < a.size(); i += 2) {// a quiet grey patch of the bark
                    a[i] = 0.31f;
                    a[i + 1] = 0.47f;
                }
            }
            auto merged = mergeBufferGeometries(std::vector<std::shared_ptr<BufferGeometry>>{t, l});
            if (!merged) return trunk;
            merged->computeBoundingBox();
            merged->computeBoundingSphere();
            return merged;
        }

        inline float topOf(const std::shared_ptr<threepp::BufferGeometry>& g) {
            if (!g) return 0.f;
            g->computeBoundingBox();
            return g->boundingBox->max().y;
        }

    }// namespace detail

    // One species: card variants (near) and lite variants (far), three each.
    inline threepp::vegetation::SpeciesVariants loadKitSpecies(const std::filesystem::path& dir,
                                                               const KitLook& look) {
        using namespace threepp;
        const KitMeshes meshes = loadTreeFile(dir / (look.name + ".tree"));
        TextureLoader tl;

        auto bark = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.92f).metalness(0.f));
        // The kit's UVs are glTF-style (v runs down the image), so every image
        // is decoded without the flip. Bark tiling is the glb's texture
        // transform: scale (sx, sy) about the image's top edge.
        bark->map = tl.load(dir / (look.name + "_bark.webp"), ColorSpace::sRGB, false);
        bark->normalMap = tl.load(dir / (look.name + "_bark_n.webp"), ColorSpace::Linear, false);
        bark->normalScale.set(0.6f, 0.6f);// the WebP normals are soft; keep them a hint
        bark->vertexColors = true;         // limbs tinted, trunk white (foldLimbs)
        for (auto* t : {bark->map.get(), bark->normalMap.get()}) {
            if (!t) continue;
            t->wrapS = t->wrapT = TextureWrapping::Repeat;
            t->repeat.copy(look.barkRepeat);
            t->offset.set(0.f, 1.f - look.barkRepeat.y);
        }

        auto leafMat = [&](const std::string& tier) {
            auto m = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(look.leafTint).roughness(0.85f).metalness(0.f));
            m->map = tl.load(dir / (look.name + "_" + tier + ".webp"), ColorSpace::sRGB, false);
            m->alphaTest = look.alphaTest;
            m->side = Side::Double;
            m->vertexColors = true;// the kit's baked canopy occlusion × species tint
            m->translucency = look.translucency;
            m->translucencyColor = Color(0.50f, 0.80f, 0.28f);
            return m;
        };
        const auto cardLeaf = leafMat("card");
        const auto liteLeaf = leafMat("lite");

        vegetation::SpeciesVariants sv;
        for (const char* tier : {"card", "lite"}) {
            for (int v = 0; v < 3; ++v) {
                const std::string p = std::string(tier) + std::to_string(v) + "_";
                const auto leaf = meshes.find(p + "leaf");
                const auto trunk = meshes.find(p + "trunk");
                if (leaf == meshes.end() || trunk == meshes.end()) continue;
                const auto limbs = meshes.find(p + "wood1");
                vegetation::TreeVariant tv;
                tv.leafGeo = leaf->second;
                tv.trunkGeo = detail::foldLimbs(trunk->second,
                                                limbs == meshes.end() ? nullptr : limbs->second, look);
                tv.barkMat = bark;
                tv.leafMat = std::string(tier) == "card" ? cardLeaf : liteLeaf;
                tv.height = std::max(detail::topOf(tv.leafGeo), detail::topOf(tv.trunkGeo));
                (std::string(tier) == "card" ? sv.near : sv.far).push_back(tv);
            }
        }
        return sv;
    }

    // ── far tier: crossed impostor cards, one mesh per forest cell ──────────
    // The pictures are baked from the lite trees by scripts/fjord/
    // bake_tree_impostors.py (impostor_albedo/normal.webp beside the trees): a
    // row per species slot, two columns (front, side) per lite variant, each
    // tree scaled uniformly to fit its cell, base on the bottom edge. The
    // extents below are recomputed from the same meshes, so card and picture
    // agree to the texel.
    constexpr int kImpostorCellW = 128, kImpostorCellH = 256;

    inline std::function<std::shared_ptr<threepp::Object3D>(const std::vector<threepp::vegetation::FarTree>&)>
    makeImpostorTier(const std::filesystem::path& dir, const std::array<threepp::vegetation::SpeciesVariants, 3>& species) {
        using namespace threepp;
        struct Frame {
            float halfW = 1.f, ymin = 0.f, ymax = 1.f;
        };
        std::array<std::array<Frame, 3>, 3> frames{};
        for (int s = 0; s < 3; ++s)
            for (size_t v = 0; v < species[s].far.size() && v < 3; ++v) {
                Frame f{0.f, 1e9f, -1e9f};
                for (const auto& g : {species[s].far[v].leafGeo, species[s].far[v].trunkGeo}) {
                    if (!g) continue;
                    const auto& a = g->getAttribute<float>("position")->array();
                    for (size_t i = 0; i + 2 < a.size(); i += 3) {
                        f.halfW = std::max({f.halfW, std::abs(a[i]), std::abs(a[i + 2])});
                        f.ymin = std::min(f.ymin, a[i + 1]);
                        f.ymax = std::max(f.ymax, a[i + 1]);
                    }
                }
                frames[s][v] = f;
            }
        TextureLoader tl;
        auto mat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.85f).metalness(0.f));
        mat->map = tl.load(dir / "impostor_albedo.webp", ColorSpace::sRGB, false);
        mat->normalMap = tl.load(dir / "impostor_normal.webp", ColorSpace::Linear, false);
        // Low cut: mips average the alpha to COVERAGE, and a far crown is a
        // few texels of it; at the leaves' 0.4 the far forest would thin out.
        mat->alphaTest = 0.18f;
        mat->side = Side::Double;
        mat->translucency = 0.35f;
        mat->translucencyColor = Color(0.50f, 0.80f, 0.28f);

        return [mat, frames](const std::vector<vegetation::FarTree>& trees) -> std::shared_ptr<Object3D> {
            std::vector<float> pos, nrm, uv;
            std::vector<unsigned int> idx;
            pos.reserve(trees.size() * 24);
            const Vector3 up(0.f, 1.f, 0.f);
            for (const auto& t : trees) {
                const Frame& f = frames[static_cast<size_t>(t.species)][static_cast<size_t>(std::clamp(t.variant, 0, 2))];
                const float ppm = std::min(kImpostorCellW / (2.f * f.halfW), kImpostorCellH / (f.ymax - f.ymin));
                const float w = kImpostorCellW / ppm * t.scale, h = kImpostorCellH / ppm * t.scale;
                const float y0 = t.position.y + f.ymin * t.scale;
                const float c = std::cos(t.yaw), s = std::sin(t.yaw);
                for (int view = 0; view < 2; ++view) {
                    // The bake's frames, turned by the tree's yaw (three.js rotation about +y).
                    const Vector3 right = view == 0 ? Vector3(c, 0.f, -s) : Vector3(-s, 0.f, -c);
                    const Vector3 toward = view == 0 ? Vector3(s, 0.f, c) : Vector3(c, 0.f, -s);
                    const int col = std::clamp(t.variant, 0, 2) * 2 + view;
                    const float u0 = static_cast<float>(col) / 6.f, u1 = static_cast<float>(col + 1) / 6.f;
                    const float vTop = static_cast<float>(t.species) / 3.f, vBot = static_cast<float>(t.species + 1) / 3.f;
                    const auto base = static_cast<unsigned int>(pos.size() / 3);
                    const float corners[4][2] = {{-0.5f, 0.f}, {0.5f, 0.f}, {0.5f, 1.f}, {-0.5f, 1.f}};
                    for (const auto& k : corners) {
                        pos.insert(pos.end(), {t.position.x + right.x * w * k[0], y0 + h * k[1], t.position.z + right.z * w * k[0]});
                        // Rounded: a card shades like the crown it shows, lit side
                        // to dark side, top to foot, not like a flat sheet.
                        Vector3 n(toward);
                        n.multiplyScalar(0.55f).addScaledVector(right, 0.45f * k[0] * 2.f).addScaledVector(up, 0.25f + 0.45f * k[1]);
                        n.normalize();
                        nrm.insert(nrm.end(), {n.x, n.y, n.z});
                        uv.insert(uv.end(), {k[0] < 0.f ? u0 : u1, k[1] > 0.5f ? vTop : vBot});
                    }
                    idx.insert(idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
                }
            }
            auto geo = BufferGeometry::create();
            geo->setIndex(idx);
            geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            geo->computeBoundingBox();
            geo->computeBoundingSphere();
            auto mesh = Mesh::create(geo, mat);
            mesh->name = "forest_impostors";
            return mesh;
        };
    }

    // Jittered-grid scatter: one candidate per `spacing` cell, kept with the
    // probability `density(x, z)` returns (0..1). `height(x, z)` gives the
    // tree's height in metres for a kept site (TreeSite::canopyHeight — the
    // forest builder scales each prototype to it).
    inline std::vector<threepp::vegetation::TreeSite> scatterSites(
            float minX, float maxX, float minZ, float maxZ, float spacing, unsigned int seed,
            const std::function<float(float, float)>& density,
            const std::function<float(float, float, float)>& height) {
        std::vector<threepp::vegetation::TreeSite> out;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        const int nx = static_cast<int>((maxX - minX) / spacing);
        const int nz = static_cast<int>((maxZ - minZ) / spacing);
        for (int iz = 0; iz < nz; ++iz) {
            for (int ix = 0; ix < nx; ++ix) {
                const float x = minX + (static_cast<float>(ix) + u01(rng)) * spacing;
                const float z = minZ + (static_cast<float>(iz) + u01(rng)) * spacing;
                const float keep = u01(rng);
                const float r = u01(rng);
                const float d = density(x, z);
                if (keep >= d) continue;
                threepp::vegetation::TreeSite s;
                s.x = x;
                s.z = z;
                s.canopyHeight = height(x, z, r);
                s.standHeight = s.canopyHeight;
                out.push_back(s);
            }
        }
        return out;
    }

}// namespace fjord

#endif//THREEPP_FJORD_FOREST_HPP
