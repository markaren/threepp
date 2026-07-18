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
//   • Normals are ANALYTIC (central differences of the provider, not the tile
//     mesh), so shared borders shade identically whatever each side's LOD.
//   • Tile bakes run on background threads (std::async) with a bounded
//     in-flight count, and at most maxSwapsPerFrame add/remove swaps are
//     applied per update() so LOD transitions never hitch the frame. A
//     split/merge only applies once every replacement mesh is ready — the
//     parent stays visible until then, so there are never holes.
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
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
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
        // Ignored by renderers without detail-map support (GL/WGPU).
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
    };

    class TileTerrain : public Group {

    public:
        explicit TileTerrain(TerrainProvider provider, TileTerrainOptions options = {})
            : provider_(std::move(provider)), o_(options) {

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
                    applyBake(*n);// synchronous: frame 0 must have full coverage
                    roots_.push_back(std::move(n));
                }
        }

        static std::shared_ptr<TileTerrain> create(TerrainProvider provider, TileTerrainOptions options = {}) {
            return std::make_shared<TileTerrain>(std::move(provider), options);
        }

        // LOD selection + application of finished bakes. Call once per frame.
        void update(const Vector3& camPos) {
            swapsLeft_ = o_.maxSwapsPerFrame;
            for (auto& r : roots_) resolve(*r, camPos);
        }

        [[nodiscard]] float heightAt(float x, float z) const { return provider_.height(x, z); }
        [[nodiscard]] int activeTiles() const { return activeTiles_; }
        [[nodiscard]] int pendingBakes() const { return inFlight_; }
        [[nodiscard]] const TileTerrainOptions& options() const { return o_; }

    private:
        struct BakeData {
            std::vector<float> pos, nrm, uv;
            std::vector<unsigned int> idx;
            std::vector<unsigned char> albedo;
            int albedoDim = 0;
        };

        // Node states (encoded by mesh/kid presence):
        //   ACTIVE leaf   — mesh, no kids
        //   SPLITTING     — mesh + kids (children baking; parent still visible)
        //   SUBDIVIDED    — no mesh, kids (children active)
        //   MERGING       — SUBDIVIDED with its own bake pending
        struct Node {
            int level = 0;
            float x0 = 0.f, z0 = 0.f, size = 0.f;
            float midH = 0.f;// provider height at tile centre (LOD distance metric)
            std::shared_ptr<Mesh> mesh;
            std::array<std::unique_ptr<Node>, 4> kid;
            std::future<BakeData> baking;

            [[nodiscard]] bool bakeReady() const {
                // deferred futures (sync mode) count as ready — .get() bakes inline.
                return baking.valid() &&
                       baking.wait_for(std::chrono::seconds(0)) != std::future_status::timeout;
            }
        };

        // ── LOD state machine ────────────────────────────────────────────────
        void resolve(Node& n, const Vector3& cam) {
            const float d = nodeDistance(n, cam);
            const bool hasKids = static_cast<bool>(n.kid[0]);

            // Optional refinement bias: scale BOTH thresholds equally so the
            // dead band (and thus hysteresis) is preserved. >= 1 => refine sooner.
            const float bias = o_.refineBias
                                       ? std::max(1.f, o_.refineBias(n.x0 + n.size * 0.5f,
                                                                     n.z0 + n.size * 0.5f, n.size * 0.5f))
                                       : 1.f;
            const float splitRadius = bias * o_.splitFactor * n.size;
            const float mergeRadius = bias * o_.mergeFactor * n.size;

            if (n.mesh && !hasKids) {// ACTIVE leaf
                if (n.level < o_.maxDepth && d < splitRadius) makeChildren(n);
                else return;
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
                    if (!c->bakeReady()) allReady = false;
                }
                if (allReady && swapsLeft_ > 0) {
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

            if (d > mergeRadius && childrenAreSimpleLeaves) {
                if (swapsLeft_ > 0) {
                    requestBake(n);
                    if (n.bakeReady()) {
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
            for (auto& c : n.kid) resolve(*c, cam);
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
            // Destroying an async future blocks until the worker finishes (tile
            // bakes are ms-scale); a deferred one is dropped without running.
            n.baking = {};
            --inFlight_;
        }

        [[nodiscard]] float nodeDistance(const Node& n, const Vector3& cam) const {
            const float px = std::clamp(cam.x, n.x0, n.x0 + n.size);
            const float pz = std::clamp(cam.z, n.z0, n.z0 + n.size);
            const float dx = cam.x - px, dz = cam.z - pz;
            const float dy = cam.y - n.midH;
            return std::sqrt(dx * dx + dz * dz + dy * dy);
        }

        void requestBake(Node& n) {
            if (n.mesh || n.baking.valid()) return;
            if (inFlight_ >= o_.maxBakesInFlight) return;// retry next update
            ++inFlight_;
            // Plain-data bake on a worker; mesh creation stays on the update()
            // thread (scene graph and GPU upload are not thread-safe). Sync
            // mode uses a deferred future: .get() bakes inline at swap time.
            n.baking = std::async(o_.asyncBake ? std::launch::async : std::launch::deferred,
                                  [this, x0 = n.x0, z0 = n.z0, size = n.size] {
                                      return bakeTile(x0, z0, size);
                                  });
        }

        // ── tile bake (worker thread) ────────────────────────────────────────
        [[nodiscard]] BakeData bakeTile(float tx0, float tz0, float tsize) const {
            const int res = o_.tileRes;
            const int vdim = res + 1;
            const float step = tsize / static_cast<float>(res);
            const float skirt = o_.skirtDepth > 0.f ? o_.skirtDepth : tsize * 0.04f;
            // Normal sampling offset: half a cell, floored so coarse tiles
            // don't alias sub-grid detail into shimmer.
            const float e = std::max(step * 0.5f, 0.75f);

            BakeData b;
            const size_t gridVerts = static_cast<size_t>(vdim) * vdim;
            b.pos.reserve((gridVerts + 4 * vdim) * 3);
            b.nrm.reserve(b.pos.capacity());
            b.uv.reserve((gridVerts + 4 * vdim) * 2);

            auto pushVert = [&](float x, float z, float y, float u, float v) {
                const float hx = provider_.height(x + e, z) - provider_.height(x - e, z);
                const float hz = provider_.height(x, z + e) - provider_.height(x, z - e);
                Vector3 nn(-hx, 2.f * e, -hz);
                nn.normalize();
                b.pos.insert(b.pos.end(), {x, y, z});
                b.nrm.insert(b.nrm.end(), {nn.x, nn.y, nn.z});
                b.uv.insert(b.uv.end(), {u, v});
            };

            for (int j = 0; j < vdim; ++j) {
                const float z = tz0 + static_cast<float>(j) * step;
                const float v = static_cast<float>(j) / static_cast<float>(res);
                for (int i = 0; i < vdim; ++i) {
                    const float x = tx0 + static_cast<float>(i) * step;
                    pushVert(x, z, provider_.height(x, z), static_cast<float>(i) / static_cast<float>(res), v);
                }
            }
            for (int j = 0; j < res; ++j)
                for (int i = 0; i < res; ++i) {
                    const auto a = static_cast<unsigned int>(j * vdim + i);
                    const auto bI = a + 1;
                    const auto c = a + static_cast<unsigned int>(vdim);
                    const auto dI = c + 1;
                    b.idx.insert(b.idx.end(), {a, c, bI, bI, c, dI});
                }

            // Skirt: duplicate each border vertex, dropped by `skirt`, keeping
            // the border vertex's normal/uv. Both windings are emitted so the
            // quad can never be back-face culled, whichever way the edge faces.
            auto addSkirtEdge = [&](const std::function<int(int)>& indexOf) {
                const auto base = static_cast<unsigned int>(b.pos.size() / 3);
                for (int k = 0; k < vdim; ++k) {
                    const auto src = static_cast<size_t>(indexOf(k));
                    b.pos.insert(b.pos.end(), {b.pos[src * 3], b.pos[src * 3 + 1] - skirt, b.pos[src * 3 + 2]});
                    b.nrm.insert(b.nrm.end(), {b.nrm[src * 3], b.nrm[src * 3 + 1], b.nrm[src * 3 + 2]});
                    b.uv.insert(b.uv.end(), {b.uv[src * 2], b.uv[src * 2 + 1]});
                }
                for (int k = 0; k < res; ++k) {
                    const auto t0 = static_cast<unsigned int>(indexOf(k));
                    const auto t1 = static_cast<unsigned int>(indexOf(k + 1));
                    const auto s0 = base + static_cast<unsigned int>(k);
                    const auto s1 = s0 + 1;
                    b.idx.insert(b.idx.end(), {t0, s0, t1, t1, s0, s1});
                    b.idx.insert(b.idx.end(), {t0, t1, s0, t1, s1, s0});
                }
            };
            addSkirtEdge([&](int k) { return k; });              // -Z edge
            addSkirtEdge([&](int k) { return res * vdim + k; }); // +Z edge
            addSkirtEdge([&](int k) { return k * vdim; });       // -X edge
            addSkirtEdge([&](int k) { return k * vdim + res; }); // +X edge

            // Per-tile albedo splat, on the DETAILED surface (slope included).
            const int adim = res * o_.splatTexelsPerQuad + 1;
            b.albedoDim = adim;
            b.albedo.assign(static_cast<size_t>(adim) * adim * 4, 255u);
            if (provider_.albedo) {
                const float astep = tsize / static_cast<float>(adim - 1);
                const float ae = std::max(astep, 0.75f);
                for (int j = 0; j < adim; ++j) {
                    const float z = tz0 + static_cast<float>(j) * astep;
                    for (int i = 0; i < adim; ++i) {
                        const float x = tx0 + static_cast<float>(i) * astep;
                        const float h = provider_.height(x, z);
                        const float hx = provider_.height(x + ae, z) - provider_.height(x - ae, z);
                        const float hz = provider_.height(x, z + ae) - provider_.height(x, z - ae);
                        const float ny = (2.f * ae) / std::sqrt(hx * hx + hz * hz + 4.f * ae * ae);
                        // 2×2 SUPERSAMPLED albedo eval — coverage AA for painted
                        // content. A point-sampled bake turns the ~1.5-texel-wide
                        // painted road into a BINARY per-texel line on coarse LOD
                        // tiles (3-5 m/texel): staircase scallops that read as
                        // beads/dashes at distance and crawl under motion on the
                        // un-jittered GL/WGPU paths (Vulkan's TAA merely blurred
                        // them — capture-diffed: the beads are in the DATA, mips
                        // and aniso never touched them because the quadtree keeps
                        // tile albedo near screen density, so LOD 0 is what you
                        // see). Averaging 4 sub-texel evals boxes the paint's
                        // analytic edge into partial coverage — a smooth, properly
                        // anti-aliased line at every tile resolution. Height/slope
                        // stay single-sample (smooth fields; the paint is xz-only).
                        float rgb[3] = {0.f, 0.f, 0.f};
                        for (int sj = 0; sj < 2; ++sj)
                            for (int si = 0; si < 2; ++si) {
                                float s[3] = {0.5f, 0.5f, 0.5f};
                                provider_.albedo(x + (si ? 0.25f : -0.25f) * astep,
                                                 z + (sj ? 0.25f : -0.25f) * astep,
                                                 h, 1.f - ny, s);
                                rgb[0] += s[0]; rgb[1] += s[1]; rgb[2] += s[2];
                            }
                        rgb[0] *= 0.25f; rgb[1] *= 0.25f; rgb[2] *= 0.25f;
                        const size_t oI = (static_cast<size_t>(j) * adim + i) * 4;
                        b.albedo[oI + 0] = static_cast<unsigned char>(std::clamp(rgb[0], 0.f, 1.f) * 255.f + 0.5f);
                        b.albedo[oI + 1] = static_cast<unsigned char>(std::clamp(rgb[1], 0.f, 1.f) * 255.f + 0.5f);
                        b.albedo[oI + 2] = static_cast<unsigned char>(std::clamp(rgb[2], 0.f, 1.f) * 255.f + 0.5f);
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
                --inFlight_;
            } else {
                b = bakeTile(n.x0, n.z0, n.size);
            }

            auto geo = BufferGeometry::create();
            geo->setIndex(b.idx);
            geo->setAttribute("position", FloatBufferAttribute::create(b.pos, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(b.nrm, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(b.uv, 2));
            geo->computeBoundingBox();
            geo->computeBoundingSphere();

            auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                            .color(Color::white)
                                                            .roughness(o_.roughness)
                                                            .metalness(o_.metalness));
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
                // look); GL and WGPU both honor it, clamped to the hardware max.
                tex->minFilter = Filter::LinearMipmapLinear;
                tex->generateMipmaps = true;
                tex->anisotropy = 16;
                mat->map = tex;
            }
            if (o_.detailMap) {
                mat->detailMap = o_.detailMap;
                mat->detailRepeat = o_.detailRepeat;
                mat->detailStrength = o_.detailStrength;
            }
            if (o_.detailNormalMap) {
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

        TerrainProvider provider_;
        TileTerrainOptions o_;
        std::vector<std::unique_ptr<Node>> roots_;
        int inFlight_ = 0;
        int activeTiles_ = 0;
        int swapsLeft_ = 0;
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_TERRAINTILES_HPP
