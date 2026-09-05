// Quadtree tile LOD terrain (CPU bake, renderer-agnostic).
//
// The single-mesh TerrainGenerator bake caps ground fidelity at one global
// vertex/texel density: at a 3 km world and a 512² grid the ground is ~6 m per
// vertex AND per albedo texel everywhere — mush up close. This system splits
// the world into a quadtree of independently baked tile meshes:
//
//   • Near tiles subdivide by camera distance (with hysteresis), multiplying
//     vertex density ×2 per level and albedo texel density with it — a
//     maxDepth-4 leaf is 16× denser than its root on both axes.
//   • All heights/colours are pulled through a TerrainProvider — two plain
//     std::functions — so the SAME machinery serves procedural noise fields,
//     the eroded TerrainGenerator output, or real elevation data (DEMs). A
//     HeightGrid helper wraps any row-major float grid with bilinear/bicubic
//     (Catmull-Rom) sampling; bicubic keeps the C1 continuity that stops
//     up-close bilinear "diamond" creases. Providers add sub-grid detail
//     (extra noise octaves) to give near tiles relief the base grid can't hold.
//   • Tiles carry a SKIRT: the border ring extruded straight down. Adjacent
//     tiles of different LOD leave T-junction cracks; a skirt hides them in
//     raster AND stops ray-traced shadow rays leaking through the gap. Skirt
//     quads are emitted with both windings so no edge can be back-face culled.
//   • Each bake makes ONE provider sweep into a height LATTICE at albedo-texel
//     spacing; positions, normals, albedo/weight texels, the world-space
//     normal map, the height AABB and the mesh-error metric all derive from
//     it (~6× fewer provider evaluations than per-product sampling).
//   • Vertex normals are ANALYTIC (lattice central differences), but their
//     epsilon tracks tile resolution, so vertex shading alone still differs
//     slightly across a LOD border. The bake therefore also emits a WORLD-
//     SPACE NORMAL MAP at texel density: the Vulkan terrain path shades from
//     it per-pixel, and mip selection band-limits it continuously — LOD
//     borders then shade identically. (GL keeps vertex normals.)
//   • With TerrainProvider::weights + TileTerrainOptions band sets, tiles also
//     bake an RGBA STRUCTURE-WEIGHT map, and the Vulkan terrain path resolves
//     per-band repeating albedo/normal/roughness sets at screen density
//     (stochastic-tiled, triplanar, height-blended) over the macro splat.
//   • LOD selection measures min distance to each tile's HEIGHT AABB from any
//     number of viewpoints (update(vector<Vector3>) — sensor rigs), and tiles
//     whose bake measured little sub-quad relief postpone splitting
//     (errorLod) — triangles go where relief demands them.
//   • Tile bakes run on background threads (std::async) with a bounded
//     in-flight count, and at most maxSwapsPerFrame add/remove swaps are
//     applied per update() so LOD transitions never hitch the frame. A
//     split/merge only applies once every replacement mesh is ready — the
//     parent stays visible until then, so there are never holes. Abandoned
//     bakes are parked and reaped, never awaited on the update thread.
//
// Usage:
//   terrain::TerrainProvider prov;
//   prov.height = [&](float x, float z) { return grid.sampleBicubic(x, z) + detail(x, z); };
//   prov.albedo = [&](float x, float z, float h, float slope, float* rgb) { ... };
//   auto tiles = terrain::TileTerrain::create(prov, opts);
//   scene.add(tiles);
//   // per frame:
//   tiles->update(camera.position);
//
// Header-only, threepp core only. The provider callbacks are invoked from
// worker threads when opts.asyncBake is true — they must be pure/thread-safe.

#ifndef THREEPP_EXTRAS_TERRAIN_TERRAINTILES_HPP
#define THREEPP_EXTRAS_TERRAIN_TERRAINTILES_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace threepp::terrain {

    // ── Height source: any row-major float grid (procedural bake or DEM) ────
    class HeightGrid {

    public:
        HeightGrid() = default;

        // `heights` is dim×dim, row-major [iz*dim+ix], in world units (metres),
        // spanning [-worldSize/2, +worldSize/2] on X and Z around (cx, cz).
        HeightGrid(std::vector<float> heights, int dim, float worldSize,
                   float cx = 0.f, float cz = 0.f)
            : h_(std::move(heights)), dim_(dim), worldSize_(worldSize),
              step_(worldSize / static_cast<float>(dim - 1)),
              half_(worldSize * 0.5f), cx_(cx), cz_(cz) {}

        [[nodiscard]] bool valid() const { return dim_ >= 4 && static_cast<int>(h_.size()) == dim_ * dim_; }
        [[nodiscard]] int dim() const { return dim_; }
        [[nodiscard]] float worldSize() const { return worldSize_; }
        [[nodiscard]] const std::vector<float>& data() const { return h_; }
        [[nodiscard]] std::vector<float>& data() { return h_; }

        [[nodiscard]] float sampleBilinear(float x, float z) const {
            float gx, gz;
            toGrid(x, z, gx, gz);
            const int ix = static_cast<int>(gx), iz = static_cast<int>(gz);
            const float fx = gx - static_cast<float>(ix), fz = gz - static_cast<float>(iz);
            const float a = at(ix, iz) + (at(ix + 1, iz) - at(ix, iz)) * fx;
            const float b = at(ix, iz + 1) + (at(ix + 1, iz + 1) - at(ix, iz + 1)) * fx;
            return a + (b - a) * fz;
        }

        // Catmull-Rom bicubic: C1-continuous — no per-cell "diamond" creases
        // when a fine tile mesh magnifies a coarse grid.
        [[nodiscard]] float sampleBicubic(float x, float z) const {
            float gx, gz;
            toGrid(x, z, gx, gz);
            const int ix = static_cast<int>(gx), iz = static_cast<int>(gz);
            const float fx = gx - static_cast<float>(ix), fz = gz - static_cast<float>(iz);
            float rows[4];
            for (int j = -1; j <= 2; ++j) {
                rows[j + 1] = catmull(at(ix - 1, iz + j), at(ix, iz + j),
                                      at(ix + 1, iz + j), at(ix + 2, iz + j), fx);
            }
            return catmull(rows[0], rows[1], rows[2], rows[3], fz);
        }

        // Surface normal Y (1 = flat, →0 vertical) via central differences.
        [[nodiscard]] float slopeNy(float x, float z, float e = 4.f) const {
            const float hx = sampleBilinear(x + e, z) - sampleBilinear(x - e, z);
            const float hz = sampleBilinear(x, z + e) - sampleBilinear(x, z - e);
            return (2.f * e) / std::sqrt(hx * hx + hz * hz + 4.f * e * e);
        }

    private:
        [[nodiscard]] float at(int ix, int iz) const {
            ix = std::clamp(ix, 0, dim_ - 1);
            iz = std::clamp(iz, 0, dim_ - 1);
            return h_[static_cast<size_t>(iz) * dim_ + ix];
        }
        void toGrid(float x, float z, float& gx, float& gz) const {
            gx = std::clamp((x - cx_ + half_) / step_, 0.f, static_cast<float>(dim_ - 1) - 1e-3f);
            gz = std::clamp((z - cz_ + half_) / step_, 0.f, static_cast<float>(dim_ - 1) - 1e-3f);
        }
        static float catmull(float p0, float p1, float p2, float p3, float t) {
            return p1 + 0.5f * t * (p2 - p0 + t * (2.f * p0 - 5.f * p1 + 4.f * p2 - p3 + t * (3.f * (p1 - p2) + p3 - p0)));
        }

        std::vector<float> h_;
        int dim_ = 0;
        float worldSize_ = 0.f, step_ = 1.f, half_ = 0.f, cx_ = 0.f, cz_ = 0.f;
    };

    // ── Provider: the tile baker's only view of the terrain ─────────────────
    struct TerrainProvider {
        // World-space height (metres). REQUIRED. Must be thread-safe.
        std::function<float(float x, float z)> height;

        // Albedo at a sample, written to rgb[0..2] in [0,1] (sRGB). `h` is the
        // provider height at (x,z); `slope` is 0 flat → 1 vertical, measured on
        // the DETAILED surface (finite differences of height()). Optional —
        // grey when unset. Must be thread-safe.
        std::function<void(float x, float z, float h, float slope, float* rgb)> albedo;

        // STRUCTURE-band coverage at a sample, written to w4[0..3] in [0,1]
        // (band order: grass/rock/scree/snow — the makeTerrainBandSet
        // convention). Baked into a per-tile RGBA weight map the Vulkan
        // terrain shading path uses to pick per-band repeating texture sets
        // (SplatRules::weightsFunction() is the ready-made source). Optional;
        // must be thread-safe.
        std::function<void(float x, float z, float h, float slope, float* w4)> weights;
    };

    struct TileTerrainOptions {
        float worldSize = 3000.f;
        float centerX = 0.f, centerZ = 0.f;
        int rootGrid = 4;  // rootGrid × rootGrid top-level tiles
        int maxDepth = 4;  // leaf tile = worldSize/rootGrid/2^maxDepth wide
        int tileRes = 64;  // quads per tile side (verts = tileRes+1 squared)

        // LOD control: split when camera distance < splitFactor·tileSize,
        // merge back when distance > mergeFactor·tileSize. Keep a dead band
        // (merge > split) or tiles flip-flop at the threshold.
        float splitFactor = 1.7f;
        float mergeFactor = 2.3f;

        // Optional per-tile LOD refinement bias. Called with the tile CENTRE
        // (cx,cz) and its half-size; return a multiplier >= 1 that scales BOTH
        // the split and merge distances for that tile, so flagged tiles subdivide
        // earlier and deeper (a road corridor kept sharp) while everything else
        // keeps the exact default behaviour. Scaling both thresholds by the same
        // factor preserves the split/merge dead band, so tiles don't flicker.
        // Empty (default) = no bias, i.e. byte-identical to the un-hooked path.
        // Invoked from the update() thread only (not the async bake workers).
        std::function<float(float cx, float cz, float halfSize)> refineBias;

        int splatTexelsPerQuad = 2;// albedo texture texels per mesh quad
        float skirtDepth = 0.f;    // 0 → auto (4% of tile size)

        int maxBakesInFlight = 4; // background bakes running at once
        int maxSwapsPerFrame = 2; // split/merge swaps applied per update()
        bool asyncBake = true;    // false → bake lazily on the update() thread

        float roughness = 0.96f;  // tile material
        float metalness = 0.f;

        // Optional tiled detail albedo layered over the per-tile splat by the
        // Vulkan deferred renderer (MaterialWithDetailMap). The splat's texel
        // density is bounded by tile resolution (~0.5 m/texel at typical leaf
        // depth — visibly coarse up close); a repeating cm-scale detail field,
        // world-XZ anchored so every tile shares it seamlessly, is what makes
        // game terrain read sharp. LINEAR color space, 0.5 = neutral; the
        // layer distance-fades in the shader, so it never patterns far away.
        // Ignored by renderers without detail-map support (GL).
        std::shared_ptr<Texture> detailMap;
        float detailRepeat = 0.8f;  // repeats per world meter
        float detailStrength = 1.f; // 0..1 modulation strength

        // Optional detail NORMAL + ROUGHNESS map, sharing detailRepeat's world-XZ
        // projection (Vulkan deferred renderer only). RGBA LINEAR: RGB =
        // tangent-space normal (0.5 = flat), A = roughness modulation (0.5 =
        // neutral). Gives near ground relief lighting + roughness breakup; both
        // fade with distance so they never shimmer far away.
        std::shared_ptr<Texture> detailNormalMap;
        float detailNormalScale = 1.f;   // tangent xy perturbation scale
        float detailRoughStrength = 0.6f;// 0..1 roughness modulation strength

        // ── Per-band STRUCTURE sets (Vulkan terrain shading path) ────────────
        // When any band albedo is set AND the provider has a weights callback,
        // tiles bake an RGBA weight map and shade through
        // MaterialWithTerrainMaps: each pixel blends up to four repeating
        // texture sets (0.5-neutral albedo overlay with height in A + normal/
        // roughness), world-XZ anchored, stochastic-tiled, height-blended —
        // colour structure resolves at SCREEN density instead of bake density.
        // The per-tile splat `map` stays the macro colour underneath. Bands
        // SUPERSEDE the single detailMap layer (it is skipped when they run).
        // makeTerrainBandSet() builds a matching procedural set.
        std::array<std::shared_ptr<Texture>, 4> bandAlbedo{};
        std::array<std::shared_ptr<Texture>, 4> bandNormalRough{};
        std::array<float, 4> bandRepeat{0.75f, 0.45f, 0.9f, 0.6f};   // repeats per metre
        std::array<float, 4> bandRoughness{0.92f, 0.82f, 0.9f, 0.45f};// base roughness
        float bandStrength = 1.f;      // 0..1 albedo-overlay modulation
        float bandNormalScale = 1.f;   // tangent perturbation scale
        float bandRoughStrength = 0.6f;// 0..1 roughness modulation
        float heightBlend = 6.f;       // band height-blend sharpness (0 = linear)

        // Scale split/merge radii by each tile's measured mesh error (how much
        // sub-quad relief its lattice holds that the mesh doesn't): flat tiles
        // postpone subdivision, jagged ones keep the full radius. Never applied
        // to refineBias-boosted tiles (their bias encodes a coverage CONTRACT —
        // see carveRoads' inflate sizing — that assumes the default radii).
        bool errorLod = true;

        [[nodiscard]] bool bandsActive() const {
            return (bandAlbedo[0] || bandNormalRough[0]) != false;
        }
    };

    class TileTerrain : public Group {

    public:
        explicit TileTerrain(TerrainProvider provider, TileTerrainOptions options = {})
            : provider_(std::move(provider)), o_(options) {

            buildSharedTopology();// index + uv identical for every tile at this res

            const float rootSize = o_.worldSize / static_cast<float>(o_.rootGrid);
            const float wx0 = o_.centerX - o_.worldSize * 0.5f;
            const float wz0 = o_.centerZ - o_.worldSize * 0.5f;
            roots_.reserve(static_cast<size_t>(o_.rootGrid) * o_.rootGrid);
            for (int iz = 0; iz < o_.rootGrid; ++iz)
                for (int ix = 0; ix < o_.rootGrid; ++ix) {
                    auto n = std::make_unique<Node>();
                    n->level = 0;
                    n->x0 = wx0 + static_cast<float>(ix) * rootSize;
                    n->z0 = wz0 + static_cast<float>(iz) * rootSize;
                    n->size = rootSize;
                    n->midH = provider_.height(n->x0 + rootSize * 0.5f, n->z0 + rootSize * 0.5f);
                    n->minH = n->maxH = n->midH;// refined by the bake below
                    applyBake(*n);// synchronous: frame 0 must have full coverage
                    roots_.push_back(std::move(n));
                }
        }

        static std::shared_ptr<TileTerrain> create(TerrainProvider provider, TileTerrainOptions options = {}) {
            return std::make_shared<TileTerrain>(std::move(provider), options);
        }

        // LOD selection + application of finished bakes. Call once per frame.
        void update(const Vector3& camPos) { updateImpl(&camPos, 1); }

        // Multi-viewpoint variant: refine against the CLOSEST of several
        // viewpoints (editor viewport + robot sensor cameras rendering the same
        // scene). A tile is as fine as its most demanding observer needs.
        void update(const std::vector<Vector3>& viewpoints) {
            if (!viewpoints.empty()) updateImpl(viewpoints.data(), viewpoints.size());
        }

        [[nodiscard]] float heightAt(float x, float z) const { return provider_.height(x, z); }
        [[nodiscard]] int activeTiles() const { return activeTiles_; }
        // Bakes RUNNING on workers right now. A finished bake that is waiting
        // for its swap gate no longer counts: see releaseSlot().
        [[nodiscard]] int pendingBakes() const { return inFlight_; }
        // FNV-1a over (level, x0, z0) of every tile with a live mesh, in tree
        // order. Two runs with equal tile counts can still hold different
        // trees (the split/merge dead band keeps whatever state a node
        // arrived in), and only the signature tells them apart.
        [[nodiscard]] std::uint64_t treeSignature() const {
            std::uint64_t h = 0xCBF29CE484222325ULL;
            for (const auto& r : roots_) signatureRec(*r, h);
            return h;
        }
        [[nodiscard]] const TileTerrainOptions& options() const { return o_; }

    private:
        // Texture gutter: per-tile maps carry this many texels of beyond-tile
        // provider data on every side so their first mip levels agree across
        // borders (see the bake). 4 → three clean mip levels.
        static constexpr int kGutter = 4;

        struct BakeData {
            std::vector<float> pos, nrm;// grid + skirt verts (index/uv are SHARED — see buildSharedTopology)
            std::vector<unsigned char> albedo;
            std::vector<unsigned char> weights; // RGBA band weights; empty = no provider.weights
            std::vector<unsigned char> wsNormal;// RGBA world-space normal map; empty = no albedo provider
            int albedoDim = 0;
            float minH = 0.f, maxH = 0.f;// tile height range (lattice extremes)
            float meshErr = 0.f;         // max |lattice − mesh bilerp| (error-scaled LOD)
        };

        // Node states (encoded by mesh/kid presence):
        //   ACTIVE leaf   — mesh, no kids
        //   SPLITTING     — mesh + kids (children baking; parent still visible)
        //   SUBDIVIDED    — no mesh, kids (children active)
        //   MERGING       — SUBDIVIDED with its own bake pending
        struct Node {
            int level = 0;
            float x0 = 0.f, z0 = 0.f, size = 0.f;
            float midH = 0.f;// provider height at tile centre (pre-bake fallback)
            float minH = 0.f, maxH = 0.f;// height AABB (bake-refined; midH until then)
            float err = 1e30f;// baked mesh error; 1e30 = unknown → never postpones a split
            std::shared_ptr<Mesh> mesh;
            std::array<std::unique_ptr<Node>, 4> kid;
            std::future<BakeData> baking;
            // True while `baking` occupies an in-flight slot (requestBake to
            // the first frame it is observed ready, or to apply/discard).
            bool bakeCounted = false;
            // Border stitching state: the tile's natural border heights (per
            // edge, vdim values, captured at bake) and whether each edge is
            // currently CONFORMED to a coarser neighbour's interpolation.
            std::array<std::vector<float>, 4> borderOrig;
            std::array<int8_t, 4> edgeConf{0, 0, 0, 0};

            [[nodiscard]] bool bakeReady() const {
                // deferred futures (sync mode) count as ready — .get() bakes inline.
                return baking.valid() &&
                       baking.wait_for(std::chrono::seconds(0)) != std::future_status::timeout;
            }
        };

        void updateImpl(const Vector3* views, size_t viewCount) {
            swapsLeft_ = o_.maxSwapsPerFrame;
            // Drain abandoned bakes that have since finished (or never started —
            // deferred futures drop without running). Parking them here instead
            // of blocking in discardBake keeps update() hitch-free even when a
            // worker is mid-bake at abandon time.
            for (auto it = graveyard_.begin(); it != graveyard_.end();) {
                if (it->wait_for(std::chrono::seconds(0)) != std::future_status::timeout)
                    it = graveyard_.erase(it);
                else
                    ++it;
            }
            for (auto& r : roots_) resolve(*r, views, viewCount);
            // Border stitching: with the LOD-delta invariant enforced by the
            // split/merge gates, a visible tile's edge meets at most one
            // coarser neighbour — drop its odd border verts onto the coarse
            // interpolation so the border is WATERTIGHT. T-junction cracks at
            // LOD transitions otherwise expose the skirt wall below the rim,
            // where the RT sun-shadow test is double-occluded: every such
            // border rendered as a dark dashed hairline (crisp under MSAA).
            for (auto& r : roots_) conformBorders(*r);
        }

        // ── neighbour queries (visible-surface level) ────────────────────────
        // Level of the tile whose MESH currently covers (x, z): leaves and
        // SPLITTING parents count (their mesh is what's on screen); SUBDIVIDED
        // nodes descend. -1 outside the terrain.
        [[nodiscard]] int visibleLevelAt(float x, float z) const {
            const Node* n = visibleNodeAt(x, z);
            return n ? n->level : -1;
        }

        [[nodiscard]] const Node* visibleNodeAt(float x, float z) const {
            return const_cast<TileTerrain*>(this)->visibleNodeAtMut(x, z);
        }

        Node* visibleNodeAtMut(float x, float z) {
            const float rootSize = o_.worldSize / static_cast<float>(o_.rootGrid);
            const float wx0 = o_.centerX - o_.worldSize * 0.5f;
            const float wz0 = o_.centerZ - o_.worldSize * 0.5f;
            const int ix = static_cast<int>(std::floor((x - wx0) / rootSize));
            const int iz = static_cast<int>(std::floor((z - wz0) / rootSize));
            if (ix < 0 || ix >= o_.rootGrid || iz < 0 || iz >= o_.rootGrid) return nullptr;
            Node* n = roots_[static_cast<size_t>(iz) * o_.rootGrid + ix].get();
            while (n) {
                if (n->mesh || !n->kid[0]) return n;// visible leaf / SPLITTING parent
                const float s = n->size * 0.5f;
                const int cx = x >= n->x0 + s ? 1 : 0;
                const int cz = z >= n->z0 + s ? 1 : 0;
                n = n->kid[static_cast<size_t>(cz * 2 + cx)].get();
            }
            return nullptr;
        }

        // Two probe points just outside edge e of node n (¼ and ¾ along it —
        // under the ≤1 delta invariant at most two neighbour leaves share an
        // edge, and these hit both).
        void edgeProbes(const Node& n, int e, float* px, float* pz) const {
            const float d = n.size * 0.01f;
            const float q1 = 0.25f * n.size, q3 = 0.75f * n.size;
            switch (e) {
                case 0: px[0] = n.x0 + q1; pz[0] = n.z0 - d;        px[1] = n.x0 + q3; pz[1] = n.z0 - d; break;        // -Z
                case 1: px[0] = n.x0 + q1; pz[0] = n.z0 + n.size + d; px[1] = n.x0 + q3; pz[1] = n.z0 + n.size + d; break;// +Z
                case 2: px[0] = n.x0 - d;  pz[0] = n.z0 + q1;       px[1] = n.x0 - d;  pz[1] = n.z0 + q3; break;        // -X
                default: px[0] = n.x0 + n.size + d; pz[0] = n.z0 + q1; px[1] = n.x0 + n.size + d; pz[1] = n.z0 + q3; break;// +X
            }
        }

        // All side neighbours' visible level >= minLevel? Optionally FORCE
        // coarser ones to start splitting (their bakes stream via their own
        // resolve visits) — the cascade is grounded: each forced node's own
        // gate requires strictly lower levels, terminating at the roots.
        bool neighborsAtLeast(const Node& n, int minLevel, bool force) {
            bool ok = true;
            float px[2], pz[2];
            for (int e = 0; e < 4; ++e) {
                edgeProbes(n, e, px, pz);
                for (int p = 0; p < 2; ++p) {
                    Node* v = visibleNodeAtMut(px[p], pz[p]);
                    if (!v || v->level >= minLevel) continue;
                    ok = false;
                    if (force && v->mesh && !v->kid[0] && v->level < o_.maxDepth)
                        makeChildren(*v);// ACTIVE → SPLITTING
                }
            }
            return ok;
        }

        // No side neighbour's visible level > maxLevel? (Merge gate — never
        // forces; the finer neighbour merges on its own radius and unblocks.)
        bool neighborsAtMost(const Node& n, int maxLevel) {
            float px[2], pz[2];
            for (int e = 0; e < 4; ++e) {
                edgeProbes(n, e, px, pz);
                for (int p = 0; p < 2; ++p) {
                    const Node* v = visibleNodeAt(px[p], pz[p]);
                    if (v && v->level > maxLevel) return false;
                }
            }
            return true;
        }

        // ── LOD state machine ────────────────────────────────────────────────
        void resolve(Node& n, const Vector3* views, size_t viewCount) {
            const float d = nodeDistance(n, views, viewCount);
            const bool hasKids = static_cast<bool>(n.kid[0]);

            // Optional refinement bias: scale BOTH thresholds equally so the
            // dead band (and thus hysteresis) is preserved. >= 1 => refine sooner.
            const float bias = o_.refineBias
                                       ? std::max(1.f, o_.refineBias(n.x0 + n.size * 0.5f,
                                                                     n.z0 + n.size * 0.5f, n.size * 0.5f))
                                       : 1.f;
            // Error scaling: a tile whose bake measured little sub-quad relief
            // gains nothing from splitting — shrink its radii (floor 0.5 keeps
            // near ground always refining for the albedo/structure density).
            // Skipped for biased tiles: refineBias radii back the carveRoads
            // quad-coverage contract and must stay exact. Both radii scale
            // equally, so the dead band survives.
            float errScale = 1.f;
            if (o_.errorLod && bias <= 1.001f && n.err < 1e29f)
                errScale = std::clamp(n.err / (n.size * 0.003f), 0.5f, 1.f);
            const float splitRadius = bias * errScale * o_.splitFactor * n.size;
            const float mergeRadius = bias * errScale * o_.mergeFactor * n.size;

            if (n.mesh && !hasKids) {// ACTIVE leaf
                if (n.level < o_.maxDepth && d < splitRadius) {
                    // Kick coarser side neighbours' streaming now (LOD-delta
                    // invariant); our own children bake concurrently and the
                    // SWAP below waits for the neighbours to catch up.
                    neighborsAtLeast(n, n.level, /*force=*/true);
                    makeChildren(n);
                } else
                    return;
            }

            if (n.mesh && n.kid[0]) {// SPLITTING
                // Camera receded before the children landed — abandon (small
                // margin over the split radius so boundary hover doesn't churn).
                if (d > splitRadius * 1.15f) {
                    abandonChildren(n);
                    return;
                }
                bool allReady = true;
                for (auto& c : n.kid) {
                    if (c->mesh) continue;
                    requestBake(*c);
                    if (c->bakeReady()) releaseSlot(*c);// done: free the slot while we wait on the gate
                    else allReady = false;
                }
                // Swap gate: children going visible at level+1 requires every
                // side neighbour's visible level >= our level, or the border
                // delta exceeds 1 and conformBorders can't bridge it. Keep the
                // parent on screen and keep nudging the laggards.
                if (allReady && swapsLeft_ > 0 && neighborsAtLeast(n, n.level, /*force=*/true)) {
                    --swapsLeft_;
                    for (auto& c : n.kid)
                        if (!c->mesh) applyBake(*c);
                    detachMesh(n);
                }
                return;// children not live yet — do not recurse
            }

            // SUBDIVIDED: merge back when every child is a plain active leaf
            // and the camera has receded past the dead band.
            bool childrenAreSimpleLeaves = true;
            for (auto& c : n.kid)
                if (c->kid[0] || !c->mesh) childrenAreSimpleLeaves = false;

            // Merge gate: dropping to our level requires no side neighbour
            // finer than level+1 (delta invariant). The finer neighbour merges
            // on its own radius and unblocks us a frame later — no forcing.
            if (d > mergeRadius && childrenAreSimpleLeaves && neighborsAtMost(n, n.level + 1)) {
                if (swapsLeft_ > 0) {
                    requestBake(n);
                    if (n.bakeReady()) {
                        releaseSlot(n);
                        --swapsLeft_;
                        applyBake(n);
                        for (auto& c : n.kid) detachMesh(*c);
                        for (auto& c : n.kid) c.reset();
                    }
                }
                return;
            }
            // Camera came back — a pending merge bake is stale; discard it.
            if (n.baking.valid()) discardBake(n);
            for (auto& c : n.kid) resolve(*c, views, viewCount);
        }

        void makeChildren(Node& n) {
            const float s = n.size * 0.5f;
            for (int k = 0; k < 4; ++k) {
                auto c = std::make_unique<Node>();
                c->level = n.level + 1;
                c->x0 = n.x0 + static_cast<float>(k & 1) * s;
                c->z0 = n.z0 + static_cast<float>(k >> 1) * s;
                c->size = s;
                c->midH = provider_.height(c->x0 + s * 0.5f, c->z0 + s * 0.5f);
                // Parent's baked height range is a better pre-bake bound than a
                // single centre sample (a valley-centre tile with a camera-level
                // ridge under-refines on a point metric).
                c->minH = std::min(n.minH, c->midH);
                c->maxH = std::max(n.maxH, c->midH);
                n.kid[static_cast<size_t>(k)] = std::move(c);
            }
        }

        void abandonChildren(Node& n) {
            for (auto& c : n.kid) {
                if (c && c->baking.valid()) discardBake(*c);
                if (c && c->mesh) detachMesh(*c);
                c.reset();
            }
        }

        void discardBake(Node& n) {
            // Destroying an async future would BLOCK until its worker finishes —
            // and a geo-provider bake (bicubic + road-network queries per
            // sample) is not always ms-scale. Park it in the graveyard instead;
            // updateImpl() reaps finished ones. Deferred futures (sync mode)
            // are dropped by the reaper without ever running. The in-flight
            // slot frees immediately — momentary oversubscription by parked
            // workers is bounded by the graveyard, which drains every frame.
            graveyard_.push_back(std::move(n.baking));
            n.baking = {};
            releaseSlot(n);
        }

        // The in-flight budget bounds bakes RUNNING on workers, not bakes
        // waiting to be swapped in. A split's children can sit ready for many
        // frames while the swap gate waits for a coarser neighbour to catch
        // up, and that neighbour needs a slot to do so: counting the ready
        // futures against the budget deadlocked the streamer with exactly
        // maxBakesInFlight bakes "in flight" forever (four, on the norddal
        // pack). Release the slot the first frame a bake is seen ready.
        void releaseSlot(Node& n) {
            if (!n.bakeCounted) return;
            n.bakeCounted = false;
            --inFlight_;
        }

        static void fnvBytes(std::uint64_t& h, const void* p, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) {
                h ^= b[i];
                h *= 0x100000001B3ULL;
            }
        }
        static void signatureRec(const Node& n, std::uint64_t& h) {
            if (n.mesh) {
                fnvBytes(h, &n.level, sizeof(n.level));
                fnvBytes(h, &n.x0, sizeof(n.x0));
                fnvBytes(h, &n.z0, sizeof(n.z0));
            }
            for (const auto& c : n.kid)
                if (c) signatureRec(*c, h);
        }

        // Min distance from any viewpoint to the tile's height-AABB
        // ([x0,x0+size] × [minH,maxH] × [z0,z0+size]).
        [[nodiscard]] float nodeDistance(const Node& n, const Vector3* views, size_t viewCount) const {
            float best = std::numeric_limits<float>::max();
            for (size_t i = 0; i < viewCount; ++i) {
                const Vector3& cam = views[i];
                const float px = std::clamp(cam.x, n.x0, n.x0 + n.size);
                const float pz = std::clamp(cam.z, n.z0, n.z0 + n.size);
                const float py = std::clamp(cam.y, n.minH, n.maxH);
                const float dx = cam.x - px, dz = cam.z - pz, dy = cam.y - py;
                best = std::min(best, dx * dx + dz * dz + dy * dy);
            }
            return std::sqrt(best);
        }

        void requestBake(Node& n) {
            if (n.mesh || n.baking.valid()) return;
            if (inFlight_ >= o_.maxBakesInFlight) return;// retry next update
            ++inFlight_;
            n.bakeCounted = true;
            // Plain-data bake on a worker; mesh creation stays on the update()
            // thread (scene graph and GPU upload are not thread-safe). Sync
            // mode uses a deferred future: .get() bakes inline at swap time.
            n.baking = std::async(o_.asyncBake ? std::launch::async : std::launch::deferred,
                                  [this, x0 = n.x0, z0 = n.z0, size = n.size] {
                                      return bakeTile(x0, z0, size);
                                  });
        }

        // ── shared topology (constructor, main thread) ───────────────────────
        // Index + uv are byte-identical for every tile at one tileRes — grid
        // and skirt alike. Build them ONCE; each applyBake copies the vectors
        // into its geometry instead of every worker re-deriving them.
        void buildSharedTopology() {
            const int res = o_.tileRes;
            const int vdim = res + 1;
            const int tpq = std::max(o_.splatTexelsPerQuad, 1);
            const int adim = res * tpq + 1;
            const size_t gridVerts = static_cast<size_t>(vdim) * vdim;
            sharedUv_.reserve((gridVerts + 8 * vdim) * 2);
            sharedIdx_.reserve(static_cast<size_t>(res) * res * 6 + 4 * res * 12);

            // TEXEL-CENTER registration: vertex (i,j)'s bake sample lands in
            // texel kGutter + i*tpq of the gutter-padded map, so its uv must
            // hit that texel's CENTER. The plain i/res mapping misses centers
            // by half a texel, which makes the border sample engage CLAMP
            // filtering asymmetrically per tile: adjacent tiles disagree on
            // the shared border's albedo/weight/normal texel and a thin seam
            // line appears at every border (crisp under unjittered MSAA).
            // Centered uvs make the border sample the exact baked border value
            // on BOTH sides — level-0 continuity is exact; the gutter extends
            // that agreement into the mip chain.
            const int tdim = adim + 2 * kGutter;
            const auto uvOf = [&](int i) {
                return (static_cast<float>(kGutter + i * tpq) + 0.5f) / static_cast<float>(tdim);
            };
            for (int j = 0; j < vdim; ++j) {
                const float v = uvOf(j);
                for (int i = 0; i < vdim; ++i)
                    sharedUv_.insert(sharedUv_.end(), {uvOf(i), v});
            }
            for (int j = 0; j < res; ++j)
                for (int i = 0; i < res; ++i) {
                    const auto a = static_cast<unsigned int>(j * vdim + i);
                    const auto bI = a + 1;
                    const auto c = a + static_cast<unsigned int>(vdim);
                    const auto dI = c + 1;
                    sharedIdx_.insert(sharedIdx_.end(), {a, c, bI, bI, c, dI});
                }
            // Skirt: TWO rings per edge — a LIP ring marginally above the
            // surface and the dropped bottom ring; quads span lip→bottom with
            // both windings so no edge can be back-face culled. The lip is the
            // T-junction dark-line fix: a crack pixel used to land on the wall
            // BELOW the neighbouring surface, where the sun shadow test is
            // occluded by both rims — every LOD border read as a dark dashed
            // hairline (crisp under MSAA). With the wall's top at surface+lip,
            // crack pixels hit it at effectively surface height with the
            // border's own normal/uv — shaded exactly like the ground around
            // it. The lip sliver itself is sub-pixel at the distances where
            // its tile LOD is active.
            auto addSkirtEdge = [&](const std::function<int(int)>& indexOf) {
                const auto base = static_cast<unsigned int>(sharedUv_.size() / 2);
                for (int ring = 0; ring < 2; ++ring)
                    for (int k = 0; k < vdim; ++k) {
                        const auto src = static_cast<size_t>(indexOf(k));
                        sharedUv_.insert(sharedUv_.end(), {sharedUv_[src * 2], sharedUv_[src * 2 + 1]});
                    }
                for (int k = 0; k < res; ++k) {
                    const auto t0 = base + static_cast<unsigned int>(k);            // lip ring
                    const auto t1 = t0 + 1;
                    const auto s0 = base + static_cast<unsigned int>(vdim + k);// bottom ring
                    const auto s1 = s0 + 1;
                    sharedIdx_.insert(sharedIdx_.end(), {t0, s0, t1, t1, s0, s1});
                    sharedIdx_.insert(sharedIdx_.end(), {t0, t1, s0, t1, s1, s0});
                }
            };
            addSkirtEdge([&](int k) { return k; });              // -Z edge
            addSkirtEdge([&](int k) { return res * vdim + k; }); // +Z edge
            addSkirtEdge([&](int k) { return k * vdim; });       // -X edge
            addSkirtEdge([&](int k) { return k * vdim + res; }); // +X edge
        }

        // ── tile bake (worker thread) ────────────────────────────────────────
        // ONE provider sweep: heights land in a lattice at albedo-texel spacing
        // (plus a stencil margin), and every product — vertex positions, vertex
        // normals, albedo/weight texels, the world-space normal map, the height
        // AABB and the mesh-error metric — derives from it. The legacy bake
        // re-evaluated the provider ~5× per vertex and ~5× per albedo texel
        // (bicubic + road-corridor queries each time); the lattice cuts
        // provider calls ~6× for the default splatTexelsPerQuad = 2.
        [[nodiscard]] BakeData bakeTile(float tx0, float tz0, float tsize) const {
            const int res = o_.tileRes;
            const int vdim = res + 1;
            const float step = tsize / static_cast<float>(res);
            const float skirt = o_.skirtDepth > 0.f ? o_.skirtDepth : tsize * 0.04f;
            const int tpq = std::max(o_.splatTexelsPerQuad, 1);
            const int adim = res * tpq + 1;
            const float astep = step / static_cast<float>(tpq);

            // Stencil radii in lattice cells. Vertex normals keep the legacy
            // epsilon max(step/2, 0.75 m) (floored so the finest tiles don't
            // alias sub-grid detail into shimmer); albedo slope + the normal
            // map keep max(astep, 0.75 m). Rounded UP to whole lattice cells.
            const auto cellsFor = [astep](float eps) {
                return std::max(1, static_cast<int>(std::ceil(eps / astep - 1e-3f)));
            };
            const int kV = cellsFor(std::max(step * 0.5f, 0.75f));
            const int kS = cellsFor(std::max(astep, 0.75f));
            // The margin serves the stencils AND the texture GUTTER (below).
            const int margin = std::max({kV, kS, kGutter}) + kS;// gutter texels need slope stencils too
            const int ldim = adim + 2 * margin;

            std::vector<float> lat(static_cast<size_t>(ldim) * ldim);
            for (int j = 0; j < ldim; ++j) {
                const float z = tz0 + static_cast<float>(j - margin) * astep;
                for (int i = 0; i < ldim; ++i) {
                    const float x = tx0 + static_cast<float>(i - margin) * astep;
                    lat[static_cast<size_t>(j) * ldim + i] = provider_.height(x, z);
                }
            }
            const auto L = [&](int i, int j) {
                return lat[static_cast<size_t>(j + margin) * ldim + (i + margin)];
            };// i,j in texel space [-margin, adim+margin)

            BakeData b;

            // Height AABB over the tile interior.
            b.minH = b.maxH = L(0, 0);
            for (int j = 0; j < adim; ++j)
                for (int i = 0; i < adim; ++i) {
                    const float h = L(i, j);
                    b.minH = std::min(b.minH, h);
                    b.maxH = std::max(b.maxH, h);
                }

            // Mesh error: the relief this bake's lattice holds that the VERTEX
            // grid can't — max |lattice − bilerp(enclosing vertex quad)|. Feeds
            // the error-scaled LOD radii (flat tiles postpone splitting).
            if (tpq > 1) {
                float err = 0.f;
                for (int j = 0; j < adim; ++j) {
                    const int vj = std::min(j / tpq, res - 1);
                    const float fj = (static_cast<float>(j) - static_cast<float>(vj * tpq)) / static_cast<float>(tpq);
                    for (int i = 0; i < adim; ++i) {
                        const int vi = std::min(i / tpq, res - 1);
                        const float fi = (static_cast<float>(i) - static_cast<float>(vi * tpq)) / static_cast<float>(tpq);
                        const float h00 = L(vi * tpq, vj * tpq), h10 = L((vi + 1) * tpq, vj * tpq);
                        const float h01 = L(vi * tpq, (vj + 1) * tpq), h11 = L((vi + 1) * tpq, (vj + 1) * tpq);
                        const float interp = (h00 * (1.f - fi) + h10 * fi) * (1.f - fj) +
                                             (h01 * (1.f - fi) + h11 * fi) * fj;
                        err = std::max(err, std::abs(L(i, j) - interp));
                    }
                }
                b.meshErr = err;
            }

            // ── grid vertices (lattice rows at vertex stride) ────────────────
            const size_t gridVerts = static_cast<size_t>(vdim) * vdim;
            b.pos.reserve((gridVerts + 8 * vdim) * 3);
            b.nrm.reserve(b.pos.capacity());
            const float eV = static_cast<float>(kV) * astep;
            for (int j = 0; j < vdim; ++j) {
                const float z = tz0 + static_cast<float>(j) * step;
                const int lj = j * tpq;
                for (int i = 0; i < vdim; ++i) {
                    const float x = tx0 + static_cast<float>(i) * step;
                    const int li = i * tpq;
                    const float hx = L(li + kV, lj) - L(li - kV, lj);
                    const float hz = L(li, lj + kV) - L(li, lj - kV);
                    Vector3 nn(-hx, 2.f * eV, -hz);
                    nn.normalize();
                    b.pos.insert(b.pos.end(), {x, L(li, lj), z});
                    b.nrm.insert(b.nrm.end(), {nn.x, nn.y, nn.z});
                }
            }
            // Skirt verts: TWO rings per edge — the LIP ring just above the
            // surface (crack pixels shade as ground, not shadowed wall; see
            // buildSharedTopology) and the dropped bottom ring. Order must
            // match buildSharedTopology's edges exactly.
            // NO lip above the surface. A raised skirt rim was tried against
            // the LOD-crack shadow hairlines and made things WORSE: the rim
            // pokes into the RT sun-shadow rays at EVERY border (self-eps is
            // |coord|-scaled and smaller than any useful rim), turning a rare
            // LOD-transition artifact into a universal grid of shadow lines —
            // confirmed by G-buffer debug views (albedo/normals clean, line
            // gone with the rim removed). Crack hairlines at LOD borders want
            // a geometric fix (neighbour LOD-delta constraint + edge
            // stitching), not taller walls.
            const float lip = 0.f;
            auto addSkirtVerts = [&](const std::function<int(int)>& indexOf) {
                for (int k = 0; k < vdim; ++k) {
                    const auto src = static_cast<size_t>(indexOf(k));
                    b.pos.insert(b.pos.end(), {b.pos[src * 3], b.pos[src * 3 + 1] + lip, b.pos[src * 3 + 2]});
                    b.nrm.insert(b.nrm.end(), {b.nrm[src * 3], b.nrm[src * 3 + 1], b.nrm[src * 3 + 2]});
                }
                for (int k = 0; k < vdim; ++k) {
                    const auto src = static_cast<size_t>(indexOf(k));
                    b.pos.insert(b.pos.end(), {b.pos[src * 3], b.pos[src * 3 + 1] - skirt, b.pos[src * 3 + 2]});
                    b.nrm.insert(b.nrm.end(), {b.nrm[src * 3], b.nrm[src * 3 + 1], b.nrm[src * 3 + 2]});
                }
            };
            addSkirtVerts([&](int k) { return k; });              // -Z edge
            addSkirtVerts([&](int k) { return res * vdim + k; }); // +Z edge
            addSkirtVerts([&](int k) { return k * vdim; });       // -X edge
            addSkirtVerts([&](int k) { return k * vdim + res; }); // +X edge

            // ── per-tile textures: albedo splat, band weights, world normals ─
            // Baked with a GUTTER: kGutter texels of real beyond-tile provider
            // data on every side. Texel-centred uvs already make level 0 agree
            // across a border, but each tile's MIPS average inward only — and
            // grazing-angle aniso reaches into the mips even close up, so
            // borders still resolved a divergent texel and drew a hairline
            // seam (crisp under unjittered MSAA). With the gutter, the first
            // log2(kGutter) mip levels average identical data on both sides;
            // deeper mips only engage at distances where the residual is
            // sub-pixel.
            const int tdim = adim + 2 * kGutter;
            b.albedoDim = tdim;
            b.albedo.assign(static_cast<size_t>(tdim) * tdim * 4, 255u);
            if (provider_.albedo) {
                const bool bakeWeights = static_cast<bool>(provider_.weights);
                if (bakeWeights) b.weights.assign(b.albedo.size(), 0u);
                b.wsNormal.assign(b.albedo.size(), 255u);
                const float ae = static_cast<float>(kS) * astep;
                for (int tj = 0; tj < tdim; ++tj) {
                    const int j = tj - kGutter;// texel space, may reach outside the tile
                    const float z = tz0 + static_cast<float>(j) * astep;
                    for (int ti = 0; ti < tdim; ++ti) {
                        const int i = ti - kGutter;
                        const float x = tx0 + static_cast<float>(i) * astep;
                        const float h = L(i, j);
                        const float hx = L(i + kS, j) - L(i - kS, j);
                        const float hz = L(i, j + kS) - L(i, j - kS);
                        const float ny = (2.f * ae) / std::sqrt(hx * hx + hz * hz + 4.f * ae * ae);
                        const float slope = 1.f - ny;
                        const size_t oI = (static_cast<size_t>(tj) * tdim + ti) * 4;

                        // World-space normal map texel. Per-PIXEL mip-filtered
                        // normals replace the interpolated vertex normal in the
                        // Vulkan G-buffer: adjacent tiles at different LODs then
                        // agree at their shared border (a mip of the fine tile's
                        // small-epsilon normals ≈ the coarse tile's large-epsilon
                        // normal), where per-vertex normals cannot — their
                        // epsilon tracks tile resolution and jumps at LOD seams.
                        {
                            const float il = 1.f / std::sqrt(hx * hx + 4.f * ae * ae + hz * hz);
                            b.wsNormal[oI + 0] = static_cast<unsigned char>((-hx * il * 0.5f + 0.5f) * 255.f + 0.5f);
                            b.wsNormal[oI + 1] = static_cast<unsigned char>((2.f * ae * il * 0.5f + 0.5f) * 255.f + 0.5f);
                            b.wsNormal[oI + 2] = static_cast<unsigned char>((-hz * il * 0.5f + 0.5f) * 255.f + 0.5f);
                        }

                        // 2×2 SUPERSAMPLED albedo eval — coverage AA for painted
                        // content. A point-sampled bake turns the ~1.5-texel-wide
                        // painted road into a BINARY per-texel line on coarse LOD
                        // tiles (3-5 m/texel): staircase scallops that read as
                        // beads/dashes at distance and crawl under motion on the
                        // un-jittered GL path (the beads are in the DATA — mips
                        // and aniso never touched them because the quadtree keeps
                        // tile albedo near screen density, so LOD 0 is what you
                        // see). Averaging 4 sub-texel evals boxes the paint's
                        // analytic edge into partial coverage. Height/slope stay
                        // single-sample (smooth fields; the paint is xz-only).
                        //
                        // Weights are SINGLE-TAP on purpose: their content is
                        // feathered band windows + the paved suppression edge,
                        // consumed through a height-blend that softens boundaries
                        // again — supersampling was measured invisible there, but
                        // it doubled total bake cost (evaluateWeights re-runs the
                        // splat layer stack + curvature), and slow bakes stretch
                        // the tile-settle window during which every swap resets
                        // the temporal accumulator (visible as DLSS shimmer).
                        float rgb[3] = {0.f, 0.f, 0.f};
                        for (int sj = 0; sj < 2; ++sj)
                            for (int si = 0; si < 2; ++si) {
                                float s[3] = {0.5f, 0.5f, 0.5f};
                                provider_.albedo(x + (si ? 0.25f : -0.25f) * astep,
                                                 z + (sj ? 0.25f : -0.25f) * astep,
                                                 h, slope, s);
                                rgb[0] += s[0]; rgb[1] += s[1]; rgb[2] += s[2];
                            }
                        b.albedo[oI + 0] = static_cast<unsigned char>(std::clamp(rgb[0] * 0.25f, 0.f, 1.f) * 255.f + 0.5f);
                        b.albedo[oI + 1] = static_cast<unsigned char>(std::clamp(rgb[1] * 0.25f, 0.f, 1.f) * 255.f + 0.5f);
                        b.albedo[oI + 2] = static_cast<unsigned char>(std::clamp(rgb[2] * 0.25f, 0.f, 1.f) * 255.f + 0.5f);
                        if (bakeWeights) {
                            float w4[4] = {0.f, 0.f, 0.f, 0.f};
                            provider_.weights(x, z, h, slope, w4);
                            b.weights[oI + 0] = static_cast<unsigned char>(std::clamp(w4[0], 0.f, 1.f) * 255.f + 0.5f);
                            b.weights[oI + 1] = static_cast<unsigned char>(std::clamp(w4[1], 0.f, 1.f) * 255.f + 0.5f);
                            b.weights[oI + 2] = static_cast<unsigned char>(std::clamp(w4[2], 0.f, 1.f) * 255.f + 0.5f);
                            b.weights[oI + 3] = static_cast<unsigned char>(std::clamp(w4[3], 0.f, 1.f) * 255.f + 0.5f);
                        }
                    }
                }
            }
            return b;
        }

        // ── main-thread mesh application ────────────────────────────────────
        // Consumes the node's pending bake (or bakes inline when none) and
        // attaches the tile mesh.
        void applyBake(Node& n) {
            BakeData b;
            if (n.baking.valid()) {
                b = n.baking.get();
                n.baking = {};
                releaseSlot(n);
            } else {
                b = bakeTile(n.x0, n.z0, n.size);
            }
            n.minH = b.minH;
            n.maxH = b.maxH;
            n.err = b.meshErr;
            // Natural border heights for the stitching pass (captured before
            // the positions move into the attribute); a fresh mesh starts
            // un-conformed.
            {
                const int vdim = o_.tileRes + 1;
                for (int e = 0; e < 4; ++e) {
                    auto& row = n.borderOrig[static_cast<size_t>(e)];
                    row.resize(static_cast<size_t>(vdim));
                    for (int k = 0; k < vdim; ++k)
                        row[static_cast<size_t>(k)] = b.pos[static_cast<size_t>(borderIdx(e, k)) * 3 + 1];
                }
                n.edgeConf = {0, 0, 0, 0};
            }

            auto geo = BufferGeometry::create();
            geo->setIndex(sharedIdx_);// copies; topology identical for every tile
            geo->setAttribute("position", FloatBufferAttribute::create(std::move(b.pos), 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(std::move(b.nrm), 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(sharedUv_, 2));
            geo->computeBoundingBox();
            geo->computeBoundingSphere();

            auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                            .color(Color::white)
                                                            .roughness(o_.roughness)
                                                            .metalness(o_.metalness));
            const bool bands = o_.bandsActive() && !b.weights.empty();
            if (provider_.albedo) {
                auto tex = DataTexture::create(ImageData{std::move(b.albedo)},
                                               static_cast<unsigned int>(b.albedoDim),
                                               static_cast<unsigned int>(b.albedoDim));
                tex->colorSpace = ColorSpace::sRGB;
                tex->magFilter = Filter::Linear;
                // Mip-filtered: the baked-road pipeline paints roads INTO this
                // texture and relies on minification integrating them smoothly at
                // distance — without a mip chain the GL path point-samples the
                // splat and a distant painted road aliases into dashes. (Vulkan
                // builds full mip chains for material textures regardless, which
                // is why only GL showed it.) Aniso keeps grazing-angle roads
                // sharp instead of mip-blurred: 16, not 8 — a road seen down-route
                // from a driving camera has footprint anisotropy ratios well past
                // 8, so at 8 the far road still collapses into mip mush (washed
                // out) and the mid-range shimmers with the residual undersampling.
                // 16 matches the Vulkan material-sampler policy (the validated
                // look); GL honors it too, clamped to the hardware max.
                tex->minFilter = Filter::LinearMipmapLinear;
                tex->generateMipmaps = true;
                tex->anisotropy = 16;
                mat->map = tex;
            }
            if (!b.wsNormal.empty()) {
                // World-space normal map (LINEAR — raw vector data). Mip
                // filtering is the point: screen-footprint mip selection
                // band-limits the normal field continuously, so tiles of
                // different LOD shade identically at shared borders.
                auto tex = DataTexture::create(ImageData{std::move(b.wsNormal)},
                                               static_cast<unsigned int>(b.albedoDim),
                                               static_cast<unsigned int>(b.albedoDim));
                tex->colorSpace = ColorSpace::Linear;
                tex->magFilter = Filter::Linear;
                tex->minFilter = Filter::LinearMipmapLinear;
                tex->generateMipmaps = true;
                tex->anisotropy = 8;
                mat->terrainNormalMap = tex;
            }
            if (bands) {
                auto tex = DataTexture::create(ImageData{std::move(b.weights)},
                                               static_cast<unsigned int>(b.albedoDim),
                                               static_cast<unsigned int>(b.albedoDim));
                tex->colorSpace = ColorSpace::Linear;// coverage data, not colour
                tex->magFilter = Filter::Linear;
                tex->minFilter = Filter::LinearMipmapLinear;
                tex->generateMipmaps = true;
                mat->terrainWeightMap = tex;
                mat->terrainBandAlbedo = o_.bandAlbedo;
                mat->terrainBandNormalRough = o_.bandNormalRough;
                mat->terrainBandRepeat = o_.bandRepeat;
                mat->terrainBandRoughness = o_.bandRoughness;
                mat->terrainBandStrength = o_.bandStrength;
                mat->terrainBandNormalScale = o_.bandNormalScale;
                mat->terrainBandRoughStrength = o_.bandRoughStrength;
                mat->terrainHeightBlend = o_.heightBlend;
            }
            // The single detail layer is the pre-band fallback; bands carry
            // their own per-band relief, so both at once would double-perturb.
            if (o_.detailMap && !bands) {
                mat->detailMap = o_.detailMap;
                mat->detailRepeat = o_.detailRepeat;
                mat->detailStrength = o_.detailStrength;
            }
            if (o_.detailNormalMap && !bands) {
                mat->detailNormalMap = o_.detailNormalMap;
                mat->detailNormalScale = o_.detailNormalScale;
                mat->detailRoughStrength = o_.detailRoughStrength;
            }

            n.mesh = Mesh::create(geo, mat);
            // Tiles ARE the LOD system here (quadtree level per ring) — the
            // renderer's auto-LOD stacking on top double-simplifies: adjacent
            // tiles land on different auto levels and shade differently at
            // the seams (positional error stays sub-pixel, shading response
            // doesn't), and every rebake churns a simplification chain.
            n.mesh->autoLod = false;
            n.mesh->name = "terrain_tile_L" + std::to_string(n.level);
            add(n.mesh);
            ++activeTiles_;
        }

        void detachMesh(Node& n) {
            if (!n.mesh) return;
            remove(*n.mesh);
            n.mesh.reset();
            --activeTiles_;
        }

        // ── border stitching (main thread, after resolve) ────────────────────
        // Grid index of border vertex k on edge e (skirt edge order).
        [[nodiscard]] int borderIdx(int e, int k) const {
            const int res = o_.tileRes;
            const int vdim = res + 1;
            switch (e) {
                case 0: return k;                 // -Z
                case 1: return res * vdim + k;    // +Z
                case 2: return k * vdim;          // -X
                default: return k * vdim + res;   // +X
            }
        }

        void conformBorders(Node& n) {
            if (n.mesh) {
                auto* pos = n.mesh->geometry()->getAttribute<float>("position");
                if (pos) {
                    bool dirty = false;
                    float px[2], pz[2];
                    for (int e = 0; e < 4; ++e) {
                        edgeProbes(n, e, px, pz);
                        int lmin = INT_MAX;
                        for (int p = 0; p < 2; ++p) {
                            const Node* v = visibleNodeAt(px[p], pz[p]);
                            if (v) lmin = std::min(lmin, v->level);
                        }
                        const int8_t want = (lmin != INT_MAX && lmin < n.level) ? int8_t(1) : int8_t(0);
                        if (n.edgeConf[e] != want && !n.borderOrig[e].empty()) {
                            applyEdgeConform(n, e, want, *pos);
                            n.edgeConf[e] = want;
                            dirty = true;
                        }
                    }
                    if (dirty) pos->needsUpdate();
                }
            }
            for (auto& c : n.kid)
                if (c) conformBorders(*c);
        }

        // Drop the edge's ODD border verts onto the midpoint of their even
        // neighbours (the coarser tile's linear edge) — or restore the natural
        // heights. Corner verts are even and never move, so two conformed
        // edges meeting at a corner stay consistent. The skirt rings mirror
        // the border, so they follow.
        void applyEdgeConform(Node& n, int e, bool conform, TypedBufferAttribute<float>& pos) {
            const int res = o_.tileRes;
            const int vdim = res + 1;
            const size_t gridVerts = static_cast<size_t>(vdim) * vdim;
            const float skirtD = o_.skirtDepth > 0.f ? o_.skirtDepth : n.size * 0.04f;
            auto& a = pos.array();
            const auto& orig = n.borderOrig[static_cast<size_t>(e)];
            for (int k = 0; k < vdim; ++k) {
                float y = orig[static_cast<size_t>(k)];
                if (conform && (k & 1) && k + 1 < vdim)
                    y = 0.5f * (orig[static_cast<size_t>(k - 1)] + orig[static_cast<size_t>(k + 1)]);
                a[static_cast<size_t>(borderIdx(e, k)) * 3 + 1] = y;
                const size_t skirtBase = gridVerts + static_cast<size_t>(e) * 2 * vdim;
                a[(skirtBase + static_cast<size_t>(k)) * 3 + 1] = y;              // lip ring (at surface)
                a[(skirtBase + static_cast<size_t>(vdim + k)) * 3 + 1] = y - skirtD;// bottom ring
            }
        }

        TerrainProvider provider_;
        TileTerrainOptions o_;
        std::vector<std::unique_ptr<Node>> roots_;
        std::vector<std::future<BakeData>> graveyard_;// abandoned bakes, reaped when done
        std::vector<unsigned int> sharedIdx_;// one topology for every tile (grid + skirt)
        std::vector<float> sharedUv_;
        int inFlight_ = 0;
        int activeTiles_ = 0;
        int swapsLeft_ = 0;
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_TERRAINTILES_HPP
