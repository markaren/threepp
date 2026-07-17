// Extruded-footprint building meshes from a geodata region pack.
//
// Turns GeoTerrainPack::buildings (OSM footprints + measured nDSM heights,
// see the pack contract in GeoTerrainPack.hpp) into renderable geometry:
// per building a flat-roof prism — earcut roof cap at groundMin + height,
// walls dropped to groundMin − sink so sloped ground never shows a gap under
// the downhill edge. No bottom cap (it is always buried).
//
// Batching: buildings are grouped into square world-space CHUNKS (chunkSize
// metres); each chunk becomes up to THREE non-indexed Meshes — windowed
// walls, plain walls, roofs — sharing three materials total. Few draw calls,
// still coarse enough spatial granularity for frustum/occlusion culling to
// drop off-screen town districts. Flat shading falls out of the non-indexed
// layout (each triangle owns its vertices).
//
// Facades (FacadeTexture.hpp): wall UVs are emitted in (window-bay, floor)
// units, SNAPPED per building/edge — floors = round(height/floorHeight) and
// bays = round(edgeLen/bayWidth), so every facade gets a complete, centred
// window grid: rows never cut at the roofline, columns never slice at
// corners. Buildings of utility types (garage/shed/warehouse/...) and
// sub-bay wall stubs get the windowless plain-cladding variant. Roof UVs are
// planar world (x,z)/roofTileSize. Glass panes carry roughness ≈0.09 via the
// roughness map, so windows pick up env/sun reflections in the deferred path.
//
// Colouring: vertexColors — walls and roofs get per-building colours out of a
// muted Nordic palette, hashed from the OSM id (stable across runs; XOR the
// `seed` option to re-roll every facade for domain randomization). Real OSM
// `building:colour` / `roof:colour` tags override the hash when parseable.
// A subtle ground-contact grime gradient darkens wall bases. NOTE: use
// MeshStandardMaterial (the Vulkan deferred path ignores Phong), colours are
// linear-space.
//
// Header-only, extras (links against threepp core for shapeutils::earcut).

#ifndef THREEPP_EXTRAS_TERRAIN_GEOBUILDINGS_HPP
#define THREEPP_EXTRAS_TERRAIN_GEOBUILDINGS_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/ShapeUtils.hpp"
#include "threepp/extras/terrain/FacadeTexture.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace threepp::terrain {

    struct GeoBuildingsOptions {
        float sink = 0.6f;      // walls extend this far below groundMin (m)
        float chunkSize = 500.f;// batching cell size (m) — culling granularity

        bool facadeTextures = true;// procedural window/cladding/roof maps
        unsigned int seed = 0;     // re-rolls every hashed facade (domain rand.)
        float floorHeight = 3.f;   // nominal storey height the floor snap targets
        float bayWidth = 2.5f;     // nominal window-bay width the bay snap targets
        float minWindowEdge = 1.6f;// facades shorter than this get plain cladding
        float roofTileSize = 3.f;  // world metres per roof texture repeat

        float roughness = 0.85f;// fallback material roughness (facadeTextures off)
        float metalness = 0.f;
        float grime = 0.18f;// ground-contact darkening at the wall base (0 = off)
    };

    namespace detail {

        inline std::uint32_t geoBldHash(const std::string& s) {
            std::uint32_t h = 2166136261u;// FNV-1a
            for (const char c : s) {
                h ^= static_cast<std::uint8_t>(c);
                h *= 16777619u;
            }
            return h;
        }

        // Muted Nordic town palette (LINEAR space): white/cream painted wood,
        // ochre, oxide red, grey render — and dark slate/tile/felt roofs.
        inline const std::array<std::array<float, 3>, 5>& geoBldWallPalette() {
            static const std::array<std::array<float, 3>, 5> p{{
                    {0.62f, 0.60f, 0.56f},// white-painted wood
                    {0.55f, 0.47f, 0.32f},// cream
                    {0.42f, 0.27f, 0.10f},// ochre
                    {0.26f, 0.07f, 0.05f},// oxide red (falu)
                    {0.35f, 0.35f, 0.36f},// grey render
            }};
            return p;
        }
        inline const std::array<std::array<float, 3>, 4>& geoBldRoofPalette() {
            static const std::array<std::array<float, 3>, 4> p{{
                    {0.055f, 0.060f, 0.068f},// slate
                    {0.085f, 0.085f, 0.090f},// graphite felt
                    {0.160f, 0.055f, 0.038f},// tile red
                    {0.095f, 0.068f, 0.048f},// brown tile
            }};
            return p;
        }

        // Signed shoelace area of an open (x, z) ring (Vector2 = (x, z)).
        inline float geoBldRingArea(const std::vector<Vector2>& r) {
            float a = 0.f;
            for (size_t i = 0, n = r.size(); i < n; ++i) {
                const Vector2& p = r[i];
                const Vector2& q = r[(i + 1) % n];
                a += p.x * q.y - q.x * p.y;
            }
            return 0.5f * a;
        }

        // Parse an OSM colour value ("#rrggbb", "#rgb", or a small set of
        // common names) into LINEAR rgb. Returns false when unparseable
        // (compound values like "Brown and Black" fall back to the palette).
        inline bool geoBldParseColour(const std::string& s, float out[3]) {
            if (s.empty()) return false;
            auto toLinear = [](float srgb) { return std::pow(srgb, 2.2f); };
            if (s[0] == '#' && (s.size() == 7 || s.size() == 4)) {
                auto nib = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    return -1;
                };
                int v[6];
                const bool longForm = s.size() == 7;
                for (int i = 0; i < (longForm ? 6 : 3); ++i) {
                    v[i] = nib(s[1 + i]);
                    if (v[i] < 0) return false;
                }
                for (int c = 0; c < 3; ++c) {
                    const float srgb = longForm
                                               ? static_cast<float>(v[c * 2] * 16 + v[c * 2 + 1]) / 255.f
                                               : static_cast<float>(v[c] * 17) / 255.f;
                    out[c] = toLinear(srgb);
                }
                return true;
            }
            std::string k;
            k.reserve(s.size());
            for (const char c : s) k += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            static const std::unordered_map<std::string, std::array<float, 3>> named = {
                    {"white", {0.95f, 0.95f, 0.93f}},
                    {"cream", {0.96f, 0.93f, 0.82f}},
                    {"beige", {0.96f, 0.96f, 0.86f}},
                    {"yellow", {0.93f, 0.85f, 0.35f}},
                    {"red", {0.70f, 0.15f, 0.12f}},
                    {"darkred", {0.55f, 0.10f, 0.08f}},
                    {"maroon", {0.50f, 0.10f, 0.10f}},
                    {"brown", {0.45f, 0.30f, 0.18f}},
                    {"tan", {0.82f, 0.71f, 0.55f}},
                    {"grey", {0.62f, 0.62f, 0.62f}},
                    {"gray", {0.62f, 0.62f, 0.62f}},
                    {"lightgrey", {0.80f, 0.80f, 0.80f}},
                    {"lightgray", {0.80f, 0.80f, 0.80f}},
                    {"darkgrey", {0.42f, 0.42f, 0.42f}},
                    {"darkgray", {0.42f, 0.42f, 0.42f}},
                    {"silver", {0.75f, 0.75f, 0.77f}},
                    {"black", {0.12f, 0.12f, 0.12f}},
                    {"green", {0.30f, 0.45f, 0.25f}},
                    {"darkgreen", {0.18f, 0.32f, 0.16f}},
                    {"blue", {0.35f, 0.45f, 0.60f}},
                    {"lightblue", {0.65f, 0.75f, 0.85f}},
                    {"orange", {0.90f, 0.55f, 0.20f}},
                    {"pink", {0.90f, 0.70f, 0.72f}},
            };
            const auto it = named.find(k);
            if (it == named.end()) return false;
            for (int c = 0; c < 3; ++c) out[c] = toLinear(it->second[c]);
            return true;
        }

        // Building types that read wrong with a residential window grid.
        inline bool geoBldIsPlainType(const std::string& t) {
            static const std::unordered_set<std::string> plain = {
                    "garage", "garages", "shed", "hut", "carport", "roof",
                    "service", "farm_auxiliary", "warehouse", "industrial",
                    "barn", "greenhouse", "storage_tank", "hangar"};
            return plain.count(t) != 0;
        }

    }// namespace detail

    // Build the batched building meshes for a pack. Returns a Group of chunk
    // meshes (empty Group if the pack carries no buildings); all meshes share
    // three materials (windowed walls / plain walls / roofs).
    inline std::shared_ptr<Group> buildGeoBuildingMeshes(const GeoTerrainPack& pack,
                                                         const GeoBuildingsOptions& o = {}) {
        auto group = Group::create();
        group->name = "geo_buildings";
        if (pack.buildings.empty()) return group;

        FacadeMaps maps;
        if (o.facadeTextures) {
            FacadeMapOptions fo;
            fo.seed = 1337u ^ o.seed;
            maps = makeFacadeMaps(fo);
        }
        const auto mkMat = [&](const FacadeSet* set) {
            auto m = MeshStandardMaterial::create();
            m->vertexColors = true;
            if (set) {
                m->map = set->albedo;
                m->normalMap = set->normal;
                m->roughnessMap = set->roughMetal;// g = roughness
                m->metalnessMap = set->roughMetal;// b = metalness
                m->roughness = 1.f;               // the maps carry the values
                m->metalness = 1.f;
            } else {
                m->roughness = o.roughness;
                m->metalness = o.metalness;
            }
            return m;
        };
        auto matWin = mkMat(o.facadeTextures ? &maps.windowed : nullptr);
        auto matPlain = mkMat(o.facadeTextures ? &maps.plain : nullptr);
        auto matRoof = mkMat(o.facadeTextures ? &maps.roof : nullptr);

        struct Buf {
            std::vector<float> pos, nrm, col, uv;
            void push(float x, float y, float z, const float n[3],
                      const float c[3], float u, float v) {
                pos.insert(pos.end(), {x, y, z});
                nrm.insert(nrm.end(), {n[0], n[1], n[2]});
                col.insert(col.end(), {c[0], c[1], c[2]});
                uv.insert(uv.end(), {u, v});
            }
        };
        struct ChunkBuf {
            Buf win, plain, roof;
        };
        std::unordered_map<std::int64_t, ChunkBuf> chunks;

        for (const auto& b : pack.buildings) {
            if (b.outer.size() < 3) continue;

            // Per-building colours: real OSM tags win; otherwise hashed out of
            // the palette, stable in the OSM id (XOR seed re-rolls them all).
            const std::uint32_t h = detail::geoBldHash(b.id) ^ o.seed;
            const float jitter = 0.85f + 0.3f * static_cast<float>((h >> 8) & 0xff) / 255.f;
            float wall[3], roof[3];
            if (!detail::geoBldParseColour(b.colour, wall)) {
                const auto& wp = detail::geoBldWallPalette()[h % detail::geoBldWallPalette().size()];
                for (int c = 0; c < 3; ++c) wall[c] = wp[c] * jitter;
            }
            if (!detail::geoBldParseColour(b.roofColour, roof)) {
                const auto& rp = detail::geoBldRoofPalette()[(h >> 16) % detail::geoBldRoofPalette().size()];
                for (int c = 0; c < 3; ++c) roof[c] = rp[c] * jitter;
            }
            const float wallBase[3] = {wall[0] * (1.f - o.grime),
                                       wall[1] * (1.f - o.grime),
                                       wall[2] * (1.f - o.grime)};

            const float yBase = b.groundMin - o.sink;
            const float yTop = b.groundMin + b.height;

            // Floor snap: an INTEGER number of texture floors spans base→roof
            // exactly (window rows never cut at the roofline; the sunk strip
            // lands on the tile's plain lower band).
            const float wallH = yTop - yBase;
            const int floors = std::max(1, static_cast<int>(std::lround(wallH / o.floorHeight)));
            const float floorH = wallH / static_cast<float>(floors);

            // Normalize winding defensively (the pack contract already says
            // outer positive / holes negative shoelace in (x,z)).
            std::vector<Vector2> outer = b.outer;
            if (detail::geoBldRingArea(outer) < 0.f)
                std::reverse(outer.begin(), outer.end());
            std::vector<std::vector<Vector2>> holes = b.holes;
            for (auto& hole : holes)
                if (detail::geoBldRingArea(hole) > 0.f)
                    std::reverse(hole.begin(), hole.end());

            // Chunk by footprint centroid.
            float cx = 0.f, cz = 0.f;
            for (const auto& p : outer) { cx += p.x; cz += p.y; }
            cx /= static_cast<float>(outer.size());
            cz /= static_cast<float>(outer.size());
            const auto cellX = static_cast<std::int32_t>(std::floor(cx / o.chunkSize));
            const auto cellZ = static_cast<std::int32_t>(std::floor(cz / o.chunkSize));
            ChunkBuf& chunk = chunks[(static_cast<std::int64_t>(cellX) << 32) ^
                                     static_cast<std::uint32_t>(cellZ)];

            // ── roof cap ────────────────────────────────────────────────────
            // triangulateShape mutates its inputs; feed it copies. Face indices
            // reference outer ⧺ holes concatenated.
            {
                std::vector<Vector2> contour = outer;
                std::vector<std::vector<Vector2>> triHoles = holes;
                const auto faces = shapeutils::triangulateShape(contour, triHoles);
                std::vector<Vector2> all = contour;
                for (const auto& hole : triHoles) all.insert(all.end(), hole.begin(), hole.end());
                static constexpr float up[3] = {0.f, 1.f, 0.f};
                const float rt = 1.f / o.roofTileSize;
                for (const auto& f : faces) {
                    if (f.size() < 3) continue;
                    Vector2 a = all[f[0]], bb = all[f[1]], c = all[f[2]];
                    // Orient so the world normal points UP: a triangle wound
                    // counter-clockwise in (x,z) has a DOWNWARD y normal, so
                    // flip positive-shoelace triangles.
                    const float cross = (bb.x - a.x) * (c.y - a.y) - (bb.y - a.y) * (c.x - a.x);
                    if (cross > 0.f) std::swap(bb, c);
                    chunk.roof.push(a.x, yTop, a.y, up, roof, a.x * rt, a.y * rt);
                    chunk.roof.push(bb.x, yTop, bb.y, up, roof, bb.x * rt, bb.y * rt);
                    chunk.roof.push(c.x, yTop, c.y, up, roof, c.x * rt, c.y * rt);
                }
            }

            // ── walls (outer ring + courtyard rings) ───────────────────────
            const bool plainType = detail::geoBldIsPlainType(b.type);
            const float vTop = static_cast<float>(floors);// (yTop-yBase)/floorH exactly
            const auto emitWalls = [&](const std::vector<Vector2>& ring) {
                for (size_t i = 0, n = ring.size(); i < n; ++i) {
                    const Vector2& p = ring[i];
                    const Vector2& q = ring[(i + 1) % n];
                    const float dx = q.x - p.x, dz = q.y - p.y;
                    const float len = std::sqrt(dx * dx + dz * dz);
                    if (len < 1e-4f) continue;
                    // Bay snap per facade: an INTEGER number of bays spans the
                    // edge exactly — complete, centred window columns. Sub-bay
                    // stubs go windowless.
                    const bool windowed = !plainType && len >= o.minWindowEdge;
                    Buf& buf = windowed ? chunk.win : chunk.plain;
                    const int bays = std::max(1, static_cast<int>(std::lround(len / o.bayWidth)));
                    const float u1 = static_cast<float>(bays);
                    // Outward for a positive-shoelace ring traversal (and
                    // courtyard-facing for the negative hole rings).
                    const float nrm[3] = {dz / len, 0.f, -dx / len};
                    // Two triangles, wound to match the outward normal; grimed
                    // colour at the base fades up the wall.
                    buf.push(p.x, yBase, p.y, nrm, wallBase, 0.f, 0.f);
                    buf.push(q.x, yTop, q.y, nrm, wall, u1, vTop);
                    buf.push(q.x, yBase, q.y, nrm, wallBase, u1, 0.f);
                    buf.push(p.x, yBase, p.y, nrm, wallBase, 0.f, 0.f);
                    buf.push(p.x, yTop, p.y, nrm, wall, 0.f, vTop);
                    buf.push(q.x, yTop, q.y, nrm, wall, u1, vTop);
                }
            };
            emitWalls(outer);
            for (const auto& hole : holes) emitWalls(hole);
        }

        const auto addMesh = [&](Buf& buf, const std::shared_ptr<MeshStandardMaterial>& mat,
                                 const char* name) {
            if (buf.pos.empty()) return;
            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position", FloatBufferAttribute::create(buf.pos, 3));
            geometry->setAttribute("normal", FloatBufferAttribute::create(buf.nrm, 3));
            geometry->setAttribute("color", FloatBufferAttribute::create(buf.col, 3));
            geometry->setAttribute("uv", FloatBufferAttribute::create(buf.uv, 2));
            geometry->computeBoundingSphere();
            auto mesh = Mesh::create(geometry, mat);
            mesh->name = name;
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            group->add(mesh);
        };
        for (auto& [key, chunk] : chunks) {
            addMesh(chunk.win, matWin, "geo_buildings_walls");
            addMesh(chunk.plain, matPlain, "geo_buildings_walls_plain");
            addMesh(chunk.roof, matRoof, "geo_buildings_roof");
        }
        return group;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_GEOBUILDINGS_HPP
