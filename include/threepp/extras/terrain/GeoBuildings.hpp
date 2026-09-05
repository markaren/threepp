// Extruded-footprint building meshes from a geodata region pack.
//
// Turns GeoTerrainPack::buildings (OSM footprints + measured nDSM heights,
// see the pack contract in GeoTerrainPack.hpp) into renderable geometry:
// walls dropped to groundMin − sink so sloped ground never shows a gap under
// the downhill edge, no bottom cap (it is always buried), and per building
// either
//   • a GABLED roof — near-rectangular non-utility footprints get a ridge
//     along the long axis of the minimum-area bounding rectangle at
//     groundMin + height (nDSM heights measure the ridge), roof planes
//     falling to eaves walls, plain-clad gable triangles filling the ends; or
//   • a FLAT cap at groundMin + height — utility types, courtyard rings,
//     footprints the rectangle fit rejects, and OSM roof:shape=flat.
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
#include "threepp/extras/terrain/StraightSkeleton.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace threepp::terrain {

    struct GeoBuildingsStats;

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

        // Gabled roofs (see the header comment for the eligibility rules).
        bool pitchedRoofs = true;      // false = every building keeps the flat cap
        float roofPitchDeg = 35.f;     // nominal gable pitch
        float minRoofRise = 1.f;       // shallower gables read flat — stay flat
        float maxRoofRise = 5.f;       // rise clamp for wide footprints
        float maxGableSpan = 18.f;     // rect short side beyond this stays flat (big-box)
        float gableCoverageMin = 0.78f;// footprint/rect area ratio to count as rectangular
        float minEavesWall = 2.4f;     // eaves height floor; the rise shrinks to keep it

        // Chimneys on gabled ridges (aerial lived-in clutter, 30 verts each,
        // shares the plain-wall material — zero extra draw calls).
        bool chimneys = true;
        float chimneyHeight = 0.9f;// stack top above the ridge (m)
        float chimneySide = 0.55f; // square cross-section (m)

        // Roofs measured off the pack's 1 m DOM (GeoBuilding::roof). Buildings
        // WITHOUT a roof block keep the heuristic above, bit for bit.
        bool measuredRoofs = true;
        float eavesOverhang = 0.45f;  // roof projection past the wall (m)
        float utilityOverhang = 0.25f;// ... on garages / sheds / industry
        float fasciaHeight = 0.18f;   // board hanging from the roof edge (m)
        float parapetHeight = 0.30f;  // lip around a measured FLAT roof (m)
        float corniceDepth = 0.12f;   // masonry cornice projection (m)
        float corniceHeight = 0.35f;  // ... and its band height (m)

        GeoBuildingsStats* stats = nullptr;// optional counters (see below)
    };

    // Filled by buildGeoBuildingMeshes when a pointer is supplied.
    struct GeoBuildingsStats {
        int buildings = 0;
        int flat = 0, gabled = 0, hipped = 0;// roof kinds actually emitted
        int heuristic = 0;                   // no roof block: old rect-fit path
        int skeletonTried = 0, skeletonFailed = 0;
        // Per-cause breakdown of the skeleton fallbacks (index = SkeletonFail).
        std::array<int, SK_COUNT> skelFail{};
        int skelFailRing = 0; // the offset ring simplified away to nothing
        int skelFailPitch = 0;// a valid skeleton whose pitch was out of range
        int cornerBoards = 0;
        int towers = 0;
        size_t meshes = 0, triangles = 0, materials = 0;
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
        // ochre, oxide red, grey render — and slate/zinc/tile roofs.
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

        inline constexpr float kGeoBldDeg2Rad = 3.14159265358979323846f / 180.f;

        // Convex hull (Andrew monotone chain) of the footprint points.
        inline std::vector<Vector2> geoBldHull(std::vector<Vector2> pts) {
            std::sort(pts.begin(), pts.end(), [](const Vector2& a, const Vector2& b) {
                return a.x < b.x || (a.x == b.x && a.y < b.y);
            });
            pts.erase(std::unique(pts.begin(), pts.end(), [](const Vector2& a, const Vector2& b) {
                          return a.x == b.x && a.y == b.y;
                      }),
                      pts.end());
            const size_t n = pts.size();
            if (n < 3) return pts;
            const auto cross = [](const Vector2& o, const Vector2& a, const Vector2& b) {
                return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
            };
            std::vector<Vector2> hull(2 * n);
            size_t k = 0;
            for (size_t i = 0; i < n; ++i) {// lower chain
                while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0.f) --k;
                hull[k++] = pts[i];
            }
            for (size_t i = n - 1, t = k + 1; i-- > 0;) {// upper chain
                while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0.f) --k;
                hull[k++] = pts[i];
            }
            hull.resize(k - 1);
            return hull;
        }

        struct GeoBldRectFit {
            Vector2 u{1.f, 0.f}, v{0.f, 1.f};// unit axes in (x, z)
            float uMin = 0.f, uMax = 0.f, vMin = 0.f, vMax = 0.f;
        };

        // Minimum-area oriented bounding rectangle of a footprint (rotating
        // calipers over the convex hull — one candidate frame per hull edge).
        inline bool geoBldMinRect(const std::vector<Vector2>& poly, GeoBldRectFit& out) {
            const auto hull = geoBldHull(poly);
            if (hull.size() < 3) return false;
            float best = std::numeric_limits<float>::max();
            for (size_t i = 0, n = hull.size(); i < n; ++i) {
                const Vector2& a = hull[i];
                const Vector2& b = hull[(i + 1) % n];
                const float ex = b.x - a.x, ey = b.y - a.y;
                const float el = std::sqrt(ex * ex + ey * ey);
                if (el < 1e-4f) continue;
                const Vector2 u(ex / el, ey / el);
                const Vector2 v(-u.y, u.x);
                float uMin = 1e30f, uMax = -1e30f, vMin = 1e30f, vMax = -1e30f;
                for (const Vector2& p : hull) {
                    const float pu = p.x * u.x + p.y * u.y;
                    const float pv = p.x * v.x + p.y * v.y;
                    uMin = std::min(uMin, pu);
                    uMax = std::max(uMax, pu);
                    vMin = std::min(vMin, pv);
                    vMax = std::max(vMax, pv);
                }
                const float area = (uMax - uMin) * (vMax - vMin);
                if (area < best) {
                    best = area;
                    out.u = u;
                    out.v = v;
                    out.uMin = uMin;
                    out.uMax = uMax;
                    out.vMin = vMin;
                    out.vMax = vMax;
                }
            }
            return best < std::numeric_limits<float>::max();
        }

        // Sutherland–Hodgman clip of a ring against the half-plane
        // side·(dot(p, axis) − c) ≤ 0. Ring orientation is preserved.
        inline std::vector<Vector2> geoBldClipHalfPlane(const std::vector<Vector2>& poly,
                                                        const Vector2& axis, float c, float side) {
            std::vector<Vector2> out;
            out.reserve(poly.size() + 2);
            for (size_t i = 0, n = poly.size(); i < n; ++i) {
                const Vector2& a = poly[i];
                const Vector2& b = poly[(i + 1) % n];
                const float da = side * (a.x * axis.x + a.y * axis.y - c);
                const float db = side * (b.x * axis.x + b.y * axis.y - c);
                if (da <= 0.f) out.push_back(a);
                if ((da < 0.f) != (db < 0.f) && std::abs(da - db) > 1e-9f) {
                    const float t = da / (da - db);
                    out.emplace_back(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
                }
            }
            return out;
        }

        // Building types that read wrong with a residential window grid.
        inline bool geoBldIsPlainType(const std::string& t) {
            static const std::unordered_set<std::string> plain = {
                    "garage", "garages", "shed", "hut", "carport", "roof",
                    "service", "farm_auxiliary", "warehouse", "industrial",
                    "barn", "greenhouse", "storage_tank", "hangar"};
            return plain.count(t) != 0;
        }

        // ── facade classes ──────────────────────────────────────────────────
        // Ålesund's centre is rendered masonry in Jugend pastels; its suburbs
        // are painted wood; its quays are profiled sheet. One clapboard tile
        // for all three is the flat read the close-up loses on.
        enum class GeoBldClass { Wood,
                                 Masonry,
                                 Industrial };

        inline bool geoBldIsMasonryType(const std::string& t) {
            static const std::unordered_set<std::string> s = {
                    "apartments", "retail", "office", "commercial", "hotel",
                    "public", "civic", "school", "hospital", "church", "chapel",
                    "cathedral", "university", "college", "kindergarten",
                    "government", "museum", "train_station", "sports_centre",
                    "supermarket", "dormitory", "parking"};
            return s.count(t) != 0;
        }
        inline bool geoBldIsIndustrialType(const std::string& t) {
            static const std::unordered_set<std::string> s = {
                    "industrial", "warehouse", "hangar", "factory", "manufacture",
                    "depot", "silo", "boat_storage", "works"};
            return s.count(t) != 0;
        }
        inline bool geoBldIsHouseType(const std::string& t) {
            static const std::unordered_set<std::string> s = {
                    "house", "detached", "semidetached_house", "terrace", "cabin",
                    "boathouse", "farm", "bungalow", "hut", "static_caravan",
                    "summer_house", "garage", "garages", "shed", "carport"};
            return s.count(t) != 0;
        }

        inline GeoBldClass geoBldClassify(const std::string& type, int floors, float area) {
            if (geoBldIsIndustrialType(type)) return GeoBldClass::Industrial;
            if (geoBldIsMasonryType(type)) return GeoBldClass::Masonry;
            if (!geoBldIsHouseType(type) && floors >= 3 && area >= 200.f)
                return GeoBldClass::Masonry;
            return GeoBldClass::Wood;
        }

        // Pastel Jugend render (LINEAR). Calibrated like the roof palette: a
        // linear albedo near 0.5 reads as the photo's pale cream under a bright
        // sky; anything under 0.3 reads as a dirty grey block.
        inline const std::array<std::array<float, 3>, 7>& geoBldMasonryPalette() {
            static const std::array<std::array<float, 3>, 7> p{{
                    {0.720f, 0.710f, 0.685f},// white render
                    {0.660f, 0.600f, 0.455f},// cream
                    {0.520f, 0.375f, 0.155f},// ochre
                    {0.545f, 0.325f, 0.260f},// salmon
                    {0.395f, 0.455f, 0.335f},// pale green
                    {0.375f, 0.425f, 0.460f},// pale blue-grey
                    {0.600f, 0.520f, 0.360f},// sand
            }};
            return p;
        }
        // Painted wood: white dominates a Norwegian suburb (~55%), then falu
        // red, ochre, grey and a dark brown.
        inline const std::array<std::array<float, 3>, 5>& geoBldWoodPalette() {
            static const std::array<std::array<float, 3>, 5> p{{
                    {0.645f, 0.635f, 0.605f},// white
                    {0.260f, 0.070f, 0.050f},// falu red
                    {0.420f, 0.270f, 0.100f},// ochre
                    {0.300f, 0.300f, 0.310f},// mid grey
                    {0.140f, 0.090f, 0.060f},// dark brown
            }};
            return p;
        }
        inline size_t geoBldWoodPick(std::uint32_t h) {
            const std::uint32_t r = h % 20u;// white ≈55%, then 15/10/10/10
            if (r < 11u) return 0;
            if (r < 14u) return 1;
            if (r < 16u) return 2;
            if (r < 18u) return 3;
            return 4;
        }
        // ── roof coverings ──────────────────────────────────────────────────
        // MEASURED against the render, not against an aerial photo. These are
        // LINEAR albedos that are multiplied by a roof texture whose own mean is
        // ~0.70, so the product is what the sun sees. Calibration (aksla-near,
        // 2026-09-05): a sunlit slate plane must read 0.50-0.60x a sunlit white
        // masonry wall in sRGB. The old 0.195 slate over a 0.87 texture read
        // 0.68 — and being blue-biased (b > r by 15%) it came out pale BLUE-grey
        // on every block. Slate is neutral to faintly warm.
        enum GeoBldRoofCover { RC_SLATE = 0,
                               RC_TILE,
                               RC_METAL };

        struct GeoBldRoofPick {
            int cover = RC_SLATE;
            float col[3] = {0.f, 0.f, 0.f};
        };

        // Distribution by hash, the mix the town actually shows:
        //   masonry  slate 55 / near-black 20 / red tile 10 / light metal 15
        //   wood     dark concrete tile 45 / red 25 / mid grey 15 / brown 15
        //   flat + industrial  felt / zinc / grey membrane, as before.
        inline GeoBldRoofPick geoBldPickRoof(bool masonry, bool coverFlat, std::uint32_t h) {
            const std::uint32_t r = (h >> 16) % 20u;
            GeoBldRoofPick k;
            auto set = [&k](int cover, float a, float b, float c) {
                k.cover = cover;
                k.col[0] = a;
                k.col[1] = b;
                k.col[2] = c;
            };
            if (coverFlat) {
                if (r < 10u) set(RC_METAL, 0.095f, 0.098f, 0.102f);     // felt
                else if (r < 16u) set(RC_METAL, 0.300f, 0.306f, 0.312f);// zinc
                else set(RC_METAL, 0.215f, 0.218f, 0.222f);             // membrane
            } else if (masonry) {
                if (r < 11u) set(RC_SLATE, 0.146f, 0.143f, 0.138f);     // slate
                else if (r < 15u) set(RC_SLATE, 0.079f, 0.077f, 0.074f);// near-black
                else if (r < 17u) set(RC_TILE, 0.205f, 0.058f, 0.038f); // red tile
                else set(RC_METAL, 0.278f, 0.282f, 0.288f);             // light metal
            } else {
                if (r < 9u) set(RC_TILE, 0.104f, 0.101f, 0.098f);      // dark concrete
                else if (r < 14u) set(RC_TILE, 0.205f, 0.058f, 0.038f);// red tile
                else if (r < 17u) set(RC_TILE, 0.178f, 0.176f, 0.172f);// mid grey
                else set(RC_TILE, 0.150f, 0.086f, 0.048f);             // brown
            }
            return k;
        }

        // Trim: white against a coloured wall, dark grey-brown against white.
        inline void geoBldTrimColour(const float wall[3], float out[3]) {
            const float lum = 0.2126f * wall[0] + 0.7152f * wall[1] + 0.0722f * wall[2];
            if (lum > 0.42f) {
                out[0] = 0.055f;
                out[1] = 0.046f;
                out[2] = 0.038f;
            } else {
                out[0] = 0.700f;
                out[1] = 0.690f;
                out[2] = 0.660f;
            }
        }

        // Mitred OUTWARD offset of a positive-shoelace ring — the eaves line.
        // Vertex count is preserved so the offset ring and the wall ring stay
        // 1:1 (the fascia/soffit strip is then a plain quad strip). Very sharp
        // corners would throw a spike; those are clamped to a bevel-ish length.
        inline std::vector<Vector2> geoBldOffsetRing(const std::vector<Vector2>& r, float d) {
            const size_t n = r.size();
            std::vector<Vector2> out(n);
            for (size_t i = 0; i < n; ++i) {
                const Vector2& p = r[i];
                const Vector2& a = r[(i + n - 1) % n];
                const Vector2& b = r[(i + 1) % n];
                Vector2 d0(p.x - a.x, p.y - a.y), d1(b.x - p.x, b.y - p.y);
                const float l0 = std::sqrt(d0.x * d0.x + d0.y * d0.y);
                const float l1 = std::sqrt(d1.x * d1.x + d1.y * d1.y);
                if (l0 < 1e-5f || l1 < 1e-5f) {
                    out[i] = p;
                    continue;
                }
                d0 = Vector2(d0.x / l0, d0.y / l0);
                d1 = Vector2(d1.x / l1, d1.y / l1);
                const Vector2 n0(d0.y, -d0.x), n1(d1.y, -d1.x);// OUTWARD
                const Vector2 s(n0.x + n1.x, n0.y + n1.y);
                const float k = 1.f + (n0.x * n1.x + n0.y * n1.y);
                const float sl = std::sqrt(s.x * s.x + s.y * s.y);
                if (sl < 1e-5f) {
                    out[i] = p;
                } else if (k < 0.4f) {
                    out[i] = Vector2(p.x + d * 2.f * s.x / sl, p.y + d * 2.f * s.y / sl);
                } else {
                    out[i] = Vector2(p.x + d * s.x / k, p.y + d * s.y / k);
                }
            }
            return out;
        }

    }// namespace detail

    // Build the batched building meshes for a pack. Returns a Group of chunk
    // meshes (empty Group if the pack carries no buildings); meshes share the
    // nine facade/roof materials (three per class plus three roof coverings).
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

        enum MatId {
            M_WOOD_WIN = 0,
            M_WOOD_PLAIN,
            M_MAS_UP,
            M_MAS_GND,
            M_MAS_PLAIN,
            M_IND,
            M_ROOF_SLATE,
            M_ROOF_TILE,
            M_ROOF_METAL,
            M_COUNT
        };
        const FacadeSet* sets[M_COUNT] = {
                &maps.windowed, &maps.plain, &maps.masonryUpper, &maps.masonryGround,
                &maps.masonryPlain, &maps.industrial, &maps.roofSlate, &maps.roofTile,
                &maps.roof};
        static const char* matNames[M_COUNT] = {
                "geo_buildings_wood", "geo_buildings_wood_plain", "geo_buildings_masonry",
                "geo_buildings_masonry_ground", "geo_buildings_masonry_plain",
                "geo_buildings_industrial", "geo_buildings_roof_slate",
                "geo_buildings_roof_tile", "geo_buildings_roof_metal"};
        std::array<std::shared_ptr<MeshStandardMaterial>, M_COUNT> mats;
        for (int i = 0; i < M_COUNT; ++i)
            mats[i] = mkMat(o.facadeTextures ? sets[i] : nullptr);

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
            std::array<Buf, M_COUNT> b;
        };
        std::unordered_map<std::int64_t, ChunkBuf> chunks;

        // Quad wound so (B−A)×(C−A) points along `n` — the same convention the
        // wall strips below use.
        const auto quad = [](Buf& buf, const float A[3], const float B[3], const float C[3],
                             const float D[3], const float n[3], const float c[3],
                             float u0, float u1, float v0, float v1) {
            buf.push(A[0], A[1], A[2], n, c, u0, v0);
            buf.push(B[0], B[1], B[2], n, c, u1, v0);
            buf.push(C[0], C[1], C[2], n, c, u1, v1);
            buf.push(A[0], A[1], A[2], n, c, u0, v0);
            buf.push(C[0], C[1], C[2], n, c, u1, v1);
            buf.push(D[0], D[1], D[2], n, c, u0, v1);
        };

        GeoBuildingsStats st;
        for (const auto& b : pack.buildings) {
            if (b.outer.size() < 3) continue;
            ++st.buildings;

            const std::uint32_t h = detail::geoBldHash(b.id) ^ o.seed;
            const float jitter = 0.85f + 0.3f * static_cast<float>((h >> 8) & 0xff) / 255.f;

            // Normalize winding defensively (the pack contract already says
            // outer positive / holes negative shoelace in (x,z)).
            std::vector<Vector2> outer = b.outer;
            if (detail::geoBldRingArea(outer) < 0.f)
                std::reverse(outer.begin(), outer.end());
            std::vector<std::vector<Vector2>> holes = b.holes;
            for (auto& hole : holes)
                if (detail::geoBldRingArea(hole) > 0.f)
                    std::reverse(hole.begin(), hole.end());
            const float plan = detail::geoBldRingArea(outer);

            const bool plainType = detail::geoBldIsPlainType(b.type);
            const int floorsEst = std::max(1, static_cast<int>(std::lround(b.height / o.floorHeight)));
            const auto cls = detail::geoBldClassify(b.type, floorsEst, plan);
            const bool masonry = cls == detail::GeoBldClass::Masonry;
            const bool industrial = cls == detail::GeoBldClass::Industrial;

            // ── palette ────────────────────────────────────────────────────
            float wall[3];
            if (!detail::geoBldParseColour(b.colour, wall)) {
                if (masonry) {
                    const auto& wp = detail::geoBldMasonryPalette()
                            [h % detail::geoBldMasonryPalette().size()];
                    for (int c = 0; c < 3; ++c) wall[c] = wp[c] * jitter;
                } else if (industrial) {
                    static const std::array<std::array<float, 3>, 3> ind{{
                            {0.330f, 0.335f, 0.345f},// galvanised
                            {0.190f, 0.215f, 0.245f},// blue-grey sheet
                            {0.430f, 0.420f, 0.395f},// off-white sheet
                    }};
                    const auto& wp = ind[h % ind.size()];
                    for (int c = 0; c < 3; ++c) wall[c] = wp[c] * jitter;
                } else {
                    const auto& wp = detail::geoBldWoodPalette()[detail::geoBldWoodPick(h)];
                    for (int c = 0; c < 3; ++c) wall[c] = wp[c] * jitter;
                }
            }
            float trim[3];
            detail::geoBldTrimColour(wall, trim);
            const float wallBase[3] = {wall[0] * (1.f - o.grime),
                                       wall[1] * (1.f - o.grime),
                                       wall[2] * (1.f - o.grime)};

            // ── roof shape ─────────────────────────────────────────────────
            // Measured roof block (Phase A) wins; without one the old
            // rectangle-fit heuristic runs unchanged, overhang 0, no trim.
            enum Kind { K_FLAT,
                        K_GABLE,
                        K_HIP };
            Kind kind = K_FLAT;
            const bool measured = o.measuredRoofs && b.hasRoof() && o.pitchedRoofs;
            if (b.roof.tower) ++st.towers;

            float overhang = 0.f;
            std::vector<Vector2> wallRing = outer;
            std::vector<Vector2> offRing;
            float yTop = b.groundMin + b.height;// ridge
            float yWallTop = yTop;              // eaves
            Vector2 gU(1.f, 0.f), gV(0.f, 1.f);
            float gVc = 0.f, gHalfW = 1.f, gSlope = 0.f, gUc = 0.f, gHalfL = 0.f;
            StraightSkeletonResult skel;
            float hipPitch = 0.f;

            const auto axesFrom = [&](const Vector2& axis) {
                float l = std::sqrt(axis.x * axis.x + axis.y * axis.y);
                if (l < 1e-4f) return false;
                gU = Vector2(axis.x / l, axis.y / l);
                gV = Vector2(-gU.y, gU.x);
                float vMin = 1e30f, vMax = -1e30f, uMin = 1e30f, uMax = -1e30f;
                for (const auto& p : wallRing) {
                    const float pv = p.x * gV.x + p.y * gV.y;
                    const float pu = p.x * gU.x + p.y * gU.y;
                    vMin = std::min(vMin, pv);
                    vMax = std::max(vMax, pv);
                    uMin = std::min(uMin, pu);
                    uMax = std::max(uMax, pu);
                }
                gVc = 0.5f * (vMin + vMax);
                gHalfW = 0.5f * (vMax - vMin);
                gUc = 0.5f * (uMin + uMax);
                gHalfL = 0.5f * (uMax - uMin);
                return gHalfW > 0.6f;
            };

            if (measured) {
                overhang = (plainType || industrial) ? o.utilityOverhang : o.eavesOverhang;
                auto simp = simplifyRing(outer);
                if (simp.size() >= 3 &&
                    std::abs(detail::geoBldRingArea(simp) - plan) < 0.05f * plan)
                    wallRing = simp;
                offRing = detail::geoBldOffsetRing(wallRing, overhang);
                if (detail::geoBldRingArea(offRing) <= plan) {
                    offRing = wallRing;// pathological notch: no overhang
                    overhang = 0.f;
                }
                const float eave = b.groundMin + b.roof.eave;
                const float ridge = b.groundMin + b.roof.ridge;
                const bool pitched = b.roof.kind != "flat" && (ridge - eave) > 0.8f &&
                                     b.roofShape != "flat";
                if (!pitched) {
                    kind = K_FLAT;
                    yTop = yWallTop = std::max(ridge, b.groundMin + 1.5f);
                } else if (b.roof.kind == "hipped" && holes.empty()) {
                    ++st.skeletonTried;
                    // The skeleton gets its OWN simplified ring. The wall ring
                    // has to keep its vertex count (the fascia strip runs 1:1
                    // with it) and the 5% area guard above rejected most
                    // simplifications, so the near-duplicate vertices and
                    // collinear runs of an OSM cadastre ring were reaching the
                    // event queue intact. That was the fallback rate.
                    const auto skelRing = simplifyRing(offRing, 0.25f, 0.5f);
                    if (skelRing.size() < 3) {
                        ++st.skelFailRing;
                    } else {
                        skel = computeRoofSkeleton(skelRing);
                        if (skel.ok && skel.maxOffset > overhang + 0.6f) {
                            hipPitch = (ridge - eave) / (skel.maxOffset - overhang);
                            if (hipPitch > 0.04f && hipPitch < 3.2f) {
                                kind = K_HIP;
                                yTop = ridge;
                                yWallTop = eave;
                            }
                        }
                        if (!skel.ok)
                            ++st.skelFail[static_cast<size_t>(
                                    std::clamp(skel.reason, 0, SK_COUNT - 1))];
                        else if (kind != K_HIP)
                            ++st.skelFailPitch;
                    }
                    if (kind != K_HIP) ++st.skeletonFailed;
                }
                if (kind == K_FLAT && pitched) {
                    // gabled (or a hip the skeleton refused): ridge along the
                    // measured axis through the footprint centre.
                    if (axesFrom(b.roof.axis)) {
                        kind = K_GABLE;
                        yTop = ridge;
                        yWallTop = eave;
                        gSlope = (ridge - eave) / gHalfW;
                    } else {
                        yTop = yWallTop = std::max(ridge, b.groundMin + 1.5f);
                    }
                }
            } else {
                // ── heuristic (no roof block) ──────────────────────────────
                // Same eaves treatment as the measured path: without it the
                // 974 blockless footprints here (and every building in packs
                // with no roof block at all, e.g. trollstigen) kept the
                // zero-thickness paper edge this phase set out to kill.
                ++st.heuristic;
                overhang = (plainType || industrial) ? o.utilityOverhang : o.eavesOverhang;
                offRing = detail::geoBldOffsetRing(wallRing, overhang);
                if (detail::geoBldRingArea(offRing) <= plan) {
                    offRing = wallRing;// pathological notch: no overhang
                    overhang = 0.f;
                }
                if (o.pitchedRoofs && !plainType && holes.empty() && b.roofShape != "flat") {
                    detail::GeoBldRectFit rect;
                    if (detail::geoBldMinRect(outer, rect)) {
                        float longE = rect.uMax - rect.uMin, shortE = rect.vMax - rect.vMin;
                        Vector2 axU = rect.u, axV = rect.v;
                        float c0 = rect.vMin, c1 = rect.vMax;
                        float cL0 = rect.uMin, cL1 = rect.uMax;
                        if (longE < shortE) {// ridge along the LONG rectangle axis
                            std::swap(longE, shortE);
                            std::swap(axU, axV);
                            c0 = rect.uMin;
                            c1 = rect.uMax;
                            cL0 = rect.vMin;
                            cL1 = rect.vMax;
                        }
                        const float cover = plan / std::max(1e-3f, longE * shortE);
                        const float coverMin = b.roofShape.empty() ? o.gableCoverageMin : 0.55f;
                        if (cover >= coverMin && shortE >= 3.f && shortE <= o.maxGableSpan) {
                            float rise = std::tan(o.roofPitchDeg * detail::kGeoBldDeg2Rad) *
                                         0.5f * shortE;
                            rise = std::min({rise, o.maxRoofRise, b.height - o.minEavesWall});
                            if (rise >= o.minRoofRise) {
                                kind = K_GABLE;
                                gU = axU;
                                gV = axV;
                                gVc = 0.5f * (c0 + c1);
                                gHalfW = 0.5f * (c1 - c0);
                                gUc = 0.5f * (cL0 + cL1);
                                gHalfL = 0.5f * (cL1 - cL0);
                                gSlope = rise / gHalfW;
                                yWallTop = yTop - rise;
                            }
                        }
                    }
                }
            }
            if (kind == K_GABLE) ++st.gabled;
            else if (kind == K_HIP)
                ++st.hipped;
            else
                ++st.flat;

            // Roof colour: real OSM tags win; slate on masonry, tile on wood,
            // felt/zinc on flat and industrial (a tile red on a flat cap reads
            // as a mistake).
            float roof[3];
            const bool coverFlat = kind == K_FLAT || industrial || plainType;
            const auto pick = detail::geoBldPickRoof(masonry, coverFlat, h);
            const int roofMat = pick.cover == detail::RC_SLATE  ? M_ROOF_SLATE
                                : pick.cover == detail::RC_TILE ? M_ROOF_TILE
                                                                : M_ROOF_METAL;
            if (!detail::geoBldParseColour(b.roofColour, roof))
                for (int c = 0; c < 3; ++c) roof[c] = pick.col[c] * jitter;

            const float yBase = b.groundMin - o.sink;
            const float hMinEave = yWallTop - gSlope * overhang;// gable verge bottom
            const auto tentH = [&](const Vector2& p) {
                const float v = p.x * gV.x + p.y * gV.y;
                return std::max(hMinEave, yTop - std::abs(v - gVc) * gSlope);
            };

            // Floor snap: an INTEGER number of texture floors spans base→eaves
            // exactly (window rows never cut at the roofline).
            const float wallH = yWallTop - yBase;
            const int floors = std::max(1, static_cast<int>(std::lround(wallH / o.floorHeight)));
            const float floorH = wallH / static_cast<float>(floors);

            float cx = 0.f, cz = 0.f;
            for (const auto& p : wallRing) {
                cx += p.x;
                cz += p.y;
            }
            cx /= static_cast<float>(wallRing.size());
            cz /= static_cast<float>(wallRing.size());
            const auto cellX = static_cast<std::int32_t>(std::floor(cx / o.chunkSize));
            const auto cellZ = static_cast<std::int32_t>(std::floor(cz / o.chunkSize));
            ChunkBuf& chunk = chunks[(static_cast<std::int64_t>(cellX) << 32) ^
                                     static_cast<std::uint32_t>(cellZ)];

            const int trimMat = masonry ? M_MAS_PLAIN : (industrial ? M_IND : M_WOOD_PLAIN);
            Buf& trimBuf = chunk.b[trimMat];
            const float rt = 1.f / o.roofTileSize;

            // ── roof surface ───────────────────────────────────────────────
            Buf& roofBuf = chunk.b[roofMat];
            if (kind == K_HIP) {
                for (const auto& f : skel.faces) {
                    const Vector2& p1 = skel.ring[f.edge];
                    const Vector2& p2 = skel.ring[(f.edge + 1) % skel.ring.size()];
                    Vector2 d(p2.x - p1.x, p2.y - p1.y);
                    const float dl = std::sqrt(d.x * d.x + d.y * d.y);
                    if (dl < 1e-4f) continue;
                    d = Vector2(d.x / dl, d.y / dl);
                    const Vector2 nIn(-d.y, d.x);// inward
                    const float slopeLen = std::sqrt(1.f + hipPitch * hipPitch);
                    float n[3] = {-hipPitch * nIn.x, 1.f, -hipPitch * nIn.y};
                    const float nl = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                    n[0] /= nl;
                    n[1] /= nl;
                    n[2] /= nl;
                    std::vector<Vector2> face = f.poly;
                    std::vector<std::vector<Vector2>> noHoles;
                    const auto tris = shapeutils::triangulateShape(face, noHoles);
                    const auto hAt = [&](const Vector2& p) {
                        const float dist = (p.x - p1.x) * nIn.x + (p.y - p1.y) * nIn.y;
                        return yWallTop + hipPitch * (dist - overhang);
                    };
                    for (const auto& t : tris) {
                        if (t.size() < 3) continue;
                        Vector2 a = face[t[0]], bb = face[t[1]], c = face[t[2]];
                        const float cr = (bb.x - a.x) * (c.y - a.y) - (bb.y - a.y) * (c.x - a.x);
                        if (cr > 0.f) std::swap(bb, c);
                        for (const Vector2* pt : {&a, &bb, &c}) {
                            const float tu = ((pt->x - p1.x) * d.x + (pt->y - p1.y) * d.y) * rt;
                            const float tv = ((pt->x - p1.x) * nIn.x + (pt->y - p1.y) * nIn.y) *
                                             slopeLen * rt;
                            roofBuf.push(pt->x, hAt(*pt), pt->y, n, roof, tu, tv);
                        }
                    }
                }
            } else if (kind == K_GABLE) {
                const float slopeLen = std::sqrt(1.f + gSlope * gSlope);
                for (const float side : {-1.f, 1.f}) {
                    std::vector<Vector2> half = detail::geoBldClipHalfPlane(offRing, gV, gVc, side);
                    if (half.size() < 3) continue;
                    std::vector<std::vector<Vector2>> noHoles;
                    const auto faces = shapeutils::triangulateShape(half, noHoles);
                    float n[3] = {gV.x * side * gSlope, 1.f, gV.y * side * gSlope};
                    const float nl = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                    n[0] /= nl;
                    n[1] /= nl;
                    n[2] /= nl;
                    for (const auto& f : faces) {
                        if (f.size() < 3) continue;
                        Vector2 a = half[f[0]], bb = half[f[1]], c = half[f[2]];
                        const float cr = (bb.x - a.x) * (c.y - a.y) - (bb.y - a.y) * (c.x - a.x);
                        if (cr > 0.f) std::swap(bb, c);
                        for (const Vector2* pt : {&a, &bb, &c}) {
                            const float tu = (pt->x * gU.x + pt->y * gU.y) * rt;
                            const float tv = std::abs(pt->x * gV.x + pt->y * gV.y - gVc) *
                                             slopeLen * rt;
                            roofBuf.push(pt->x, tentH(*pt), pt->y, n, roof, tu, tv);
                        }
                    }
                }
            } else {
                std::vector<Vector2> contour = wallRing;
                std::vector<std::vector<Vector2>> triHoles = holes;
                const auto faces = shapeutils::triangulateShape(contour, triHoles);
                std::vector<Vector2> all = contour;
                for (const auto& hole : triHoles) all.insert(all.end(), hole.begin(), hole.end());
                static constexpr float up[3] = {0.f, 1.f, 0.f};
                for (const auto& f : faces) {
                    if (f.size() < 3) continue;
                    Vector2 a = all[f[0]], bb = all[f[1]], c = all[f[2]];
                    const float cr = (bb.x - a.x) * (c.y - a.y) - (bb.y - a.y) * (c.x - a.x);
                    if (cr > 0.f) std::swap(bb, c);
                    roofBuf.push(a.x, yTop, a.y, up, roof, a.x * rt, a.y * rt);
                    roofBuf.push(bb.x, yTop, bb.y, up, roof, bb.x * rt, bb.y * rt);
                    roofBuf.push(c.x, yTop, c.y, up, roof, c.x * rt, c.y * rt);
                }
            }

            // ── walls ──────────────────────────────────────────────────────
            // Masonry gets its GROUND FLOOR as a separate strip so it can carry
            // shopfront glazing over a rusticated base; every edge gets a
            // hashed integer bay phase so the window grid stops lining up from
            // building to building (the wallpaper read).
            const float vTop = static_cast<float>(floors);
            const auto emitWalls = [&](const std::vector<Vector2>& ring) {
                for (size_t i = 0, n = ring.size(); i < n; ++i) {
                    const Vector2& p = ring[i];
                    const Vector2& q = ring[(i + 1) % n];
                    const float dx = q.x - p.x, dz = q.y - p.y;
                    const float len = std::sqrt(dx * dx + dz * dz);
                    if (len < 1e-4f) continue;
                    const bool windowed = !plainType && len >= o.minWindowEdge;
                    const int bays = std::max(1, static_cast<int>(std::lround(len / o.bayWidth)));
                    const float nrm[3] = {dz / len, 0.f, -dx / len};
                    const float phase = static_cast<float>((h >> (3u * (i & 7u))) & 1u);
                    int mat;
                    float uScale = 1.f;
                    if (industrial) mat = M_IND;
                    else if (masonry)
                        mat = windowed ? M_MAS_UP : M_MAS_PLAIN, uScale = windowed ? 0.5f : 1.f;
                    else
                        mat = windowed ? M_WOOD_WIN : M_WOOD_PLAIN, uScale = windowed ? 0.5f : 1.f;
                    const float u0 = phase * uScale;
                    const float u1 = u0 + uScale * static_cast<float>(bays);
                    const bool splitGround = masonry && windowed && floors >= 2;
                    const float yMid = splitGround ? yBase + floorH : yBase;
                    if (splitGround) {
                        Buf& g = chunk.b[M_MAS_GND];
                        g.push(p.x, yBase, p.y, nrm, wallBase, u0, 0.f);
                        g.push(q.x, yMid, q.y, nrm, wall, u1, 1.f);
                        g.push(q.x, yBase, q.y, nrm, wallBase, u1, 0.f);
                        g.push(p.x, yBase, p.y, nrm, wallBase, u0, 0.f);
                        g.push(p.x, yMid, p.y, nrm, wall, u0, 1.f);
                        g.push(q.x, yMid, q.y, nrm, wall, u1, 1.f);
                    }
                    Buf& buf = chunk.b[mat];
                    const float vHi = splitGround ? vTop - 1.f : vTop;
                    const float cLo[3] = {splitGround ? wall[0] : wallBase[0],
                                          splitGround ? wall[1] : wallBase[1],
                                          splitGround ? wall[2] : wallBase[2]};
                    buf.push(p.x, yMid, p.y, nrm, cLo, u0, 0.f);
                    buf.push(q.x, yWallTop, q.y, nrm, wall, u1, vHi);
                    buf.push(q.x, yMid, q.y, nrm, cLo, u1, 0.f);
                    buf.push(p.x, yMid, p.y, nrm, cLo, u0, 0.f);
                    buf.push(p.x, yWallTop, p.y, nrm, wall, u0, vHi);
                    buf.push(q.x, yWallTop, q.y, nrm, wall, u1, vHi);
                }
            };
            emitWalls(wallRing);
            for (const auto& hole : holes) emitWalls(hole);

            // ── corner boards ──────────────────────────────────────────────
            // Wood only, and GEOMETRY, not texture: the cladding tile's u wraps
            // once per BAY, so a board baked into it landed every 2.5 m and a
            // white house read as plaster panels with vertical seams. A real
            // 0.14 m board in the trim colour at each footprint corner is what
            // the eye uses to separate one wall plane from the next.
            if (!masonry && !industrial && !plainType && wallH > 2.f) {
                const float cw = 0.12f, prod = 0.035f;
                const size_t n = wallRing.size();
                // Only at a REAL corner. An OSM house ring carries collinear
                // vertices by the handful; a board at every one of them came
                // out as black posts down the middle of a wall.
                const auto isCorner = [&](size_t v) {
                    const Vector2& A = wallRing[(v + n - 1) % n];
                    const Vector2& B = wallRing[v];
                    const Vector2& C = wallRing[(v + 1) % n];
                    const float ax = B.x - A.x, az = B.y - A.y;
                    const float bx = C.x - B.x, bz = C.y - B.y;
                    const float la = std::sqrt(ax * ax + az * az);
                    const float lb = std::sqrt(bx * bx + bz * bz);
                    if (la < 0.5f || lb < 0.5f) return false;
                    const float c = (ax * bx + az * bz) / (la * lb);
                    return c < 0.94f;// turn > ~20 deg
                };
                for (size_t i = 0; i < n; ++i) {
                    const Vector2& P = wallRing[i];
                    const Vector2& Q = wallRing[(i + 1) % n];
                    const float dx = Q.x - P.x, dz = Q.y - P.y;
                    const float len = std::sqrt(dx * dx + dz * dz);
                    if (len < 2.f * cw + 0.6f) continue;
                    const float ux = dx / len, uz = dz / len;
                    const float nrm[3] = {uz, 0.f, -ux};
                    const float yLo = yBase + 0.05f;
                    const float yHi = std::min(yWallTop, tentH(P) + 0.0f);
                    // 60% of the way to the trim: the full trim colour on a
                    // white wall is near-black, and a black post is not a board.
                    const float cbc[3] = {0.4f * wall[0] + 0.6f * trim[0],
                                          0.4f * wall[1] + 0.6f * trim[1],
                                          0.4f * wall[2] + 0.6f * trim[2]};
                    for (const float s0 : {0.f, len - cw}) {
                        if (!isCorner(s0 == 0.f ? i : (i + 1) % n)) continue;
                        const float ax = P.x + ux * s0 + nrm[0] * prod;
                        const float az = P.y + uz * s0 + nrm[2] * prod;
                        const float bx = P.x + ux * (s0 + cw) + nrm[0] * prod;
                        const float bz = P.y + uz * (s0 + cw) + nrm[2] * prod;
                        const float top = kind == K_GABLE ? yHi : yWallTop;
                        const float A[3] = {ax, top, az}, B[3] = {bx, top, bz};
                        const float C[3] = {bx, yLo, bz}, D[3] = {ax, yLo, az};
                        quad(trimBuf, A, B, C, D, nrm, cbc, 0.f, cw / o.bayWidth,
                             (top - yLo) / o.floorHeight, 0.f);
                        ++st.cornerBoards;
                    }
                }
            }

            // ── gable ends ─────────────────────────────────────────────────
            if (kind == K_GABLE) {
                for (size_t i = 0, n = wallRing.size(); i < n; ++i) {
                    const Vector2& p = wallRing[i];
                    const Vector2& q = wallRing[(i + 1) % n];
                    const float dx = q.x - p.x, dz = q.y - p.y;
                    const float len = std::sqrt(dx * dx + dz * dz);
                    if (len < 1e-4f) continue;
                    const float nrm[3] = {dz / len, 0.f, -dx / len};
                    struct GablePt {
                        Vector2 xy;
                        float s, hh;
                    };
                    GablePt pts[3];
                    int m = 0;
                    pts[m++] = {p, 0.f, tentH(p)};
                    const float vp = p.x * gV.x + p.y * gV.y - gVc;
                    const float vq = q.x * gV.x + q.y * gV.y - gVc;
                    if ((vp < 0.f) != (vq < 0.f) && std::abs(vp - vq) > 1e-6f) {
                        const float t = vp / (vp - vq);
                        pts[m++] = {Vector2(p.x + dx * t, p.y + dz * t), len * t, yTop};
                    }
                    pts[m++] = {q, len, tentH(q)};
                    for (int k = 0; k + 1 < m; ++k) {
                        const GablePt& A = pts[k];
                        const GablePt& B = pts[k + 1];
                        const float ha = A.hh - yWallTop, hb = B.hh - yWallTop;
                        if (ha < 1e-3f && hb < 1e-3f) continue;
                        const float ua = A.s / o.bayWidth, ub = B.s / o.bayWidth;
                        if (hb > 1e-3f) {
                            trimBuf.push(A.xy.x, yWallTop, A.xy.y, nrm, wall, ua, 0.f);
                            trimBuf.push(B.xy.x, B.hh, B.xy.y, nrm, wall, ub, hb / floorH);
                            trimBuf.push(B.xy.x, yWallTop, B.xy.y, nrm, wall, ub, 0.f);
                        }
                        if (ha > 1e-3f) {
                            trimBuf.push(A.xy.x, yWallTop, A.xy.y, nrm, wall, ua, 0.f);
                            trimBuf.push(A.xy.x, A.hh, A.xy.y, nrm, wall, ua, ha / floorH);
                            trimBuf.push(B.xy.x, B.hh, B.xy.y, nrm, wall, ub, hb / floorH);
                        }
                    }
                }
            }

            // ── eaves: fascia board + soffit ───────────────────────────────
            // The roof plane stops in mid-air over the wall without these, and
            // the RT shadow they cast on the facade IS the eaves line — the
            // single cheapest cure for the paper-model read.
            if (overhang > 0.01f && kind != K_FLAT &&
                offRing.size() == wallRing.size()) {
                const size_t n = wallRing.size();
                const auto outerH = [&](const Vector2& p) {
                    return kind == K_HIP ? yWallTop - hipPitch * overhang : tentH(p);
                };
                const auto innerH = [&](const Vector2& p) {
                    return kind == K_HIP ? yWallTop : std::max(yWallTop, tentH(p));
                };
                for (size_t i = 0; i < n; ++i) {
                    const Vector2& O0 = offRing[i];
                    const Vector2& O1 = offRing[(i + 1) % n];
                    const Vector2& W0 = wallRing[i];
                    const Vector2& W1 = wallRing[(i + 1) % n];
                    const float dx = O1.x - O0.x, dz = O1.y - O0.y;
                    const float len = std::sqrt(dx * dx + dz * dz);
                    if (len < 1e-4f) continue;
                    const float h0 = outerH(O0), h1 = outerH(O1);
                    const float b0 = h0 - o.fasciaHeight, b1 = h1 - o.fasciaHeight;
                    const float nrm[3] = {dz / len, 0.f, -dx / len};
                    const float uu = len / o.bayWidth;
                    const float A[3] = {O0.x, h0, O0.y}, B[3] = {O1.x, h1, O1.y};
                    const float C[3] = {O1.x, b1, O1.y}, D[3] = {O0.x, b0, O0.y};
                    quad(trimBuf, A, B, C, D, nrm, trim, 0.f, uu, 0.30f,
                         0.30f - o.fasciaHeight / o.floorHeight);
                    // soffit: the boxed underside back to the wall
                    static constexpr float dn[3] = {0.f, -1.f, 0.f};
                    const float S0[3] = {O0.x, b0, O0.y}, S1[3] = {O1.x, b1, O1.y};
                    const float S2[3] = {W1.x, innerH(W1), W1.y};
                    const float S3[3] = {W0.x, innerH(W0), W0.y};
                    quad(trimBuf, S0, S1, S2, S3, dn, trim, 0.f, uu, 0.10f, 0.22f);
                }
            }

            // ── flat roof: parapet lip ─────────────────────────────────────
            if (kind == K_FLAT && o.parapetHeight > 0.01f) {
                const size_t n = wallRing.size();
                for (size_t i = 0; i < n; ++i) {
                    const Vector2& W0 = wallRing[i];
                    const Vector2& W1 = wallRing[(i + 1) % n];
                    const float dx = W1.x - W0.x, dz = W1.y - W0.y;
                    const float len = std::sqrt(dx * dx + dz * dz);
                    if (len < 1e-4f) continue;
                    const float nrm[3] = {dz / len, 0.f, -dx / len};
                    const float inv[3] = {-nrm[0], 0.f, -nrm[2]};
                    const float yP = yTop + o.parapetHeight;
                    const float uu = len / o.bayWidth;
                    const float A[3] = {W0.x, yP, W0.y}, B[3] = {W1.x, yP, W1.y};
                    const float C[3] = {W1.x, yTop, W1.y}, D[3] = {W0.x, yTop, W0.y};
                    quad(trimBuf, A, B, C, D, nrm, trim, 0.f, uu, 0.30f, 0.20f);
                    quad(trimBuf, B, A, D, C, inv, trim, 0.f, uu, 0.30f, 0.20f);
                }
            }

            // ── masonry cornice ────────────────────────────────────────────
            if (measured && masonry && o.corniceDepth > 0.01f && wallH > 4.f) {
                const auto cRing = detail::geoBldOffsetRing(wallRing, o.corniceDepth);
                const size_t n = wallRing.size();
                const float yC1 = yWallTop, yC0 = yWallTop - o.corniceHeight;
                for (size_t i = 0; i < n; ++i) {
                    const Vector2& C0 = cRing[i];
                    const Vector2& C1 = cRing[(i + 1) % n];
                    const Vector2& W0 = wallRing[i];
                    const Vector2& W1 = wallRing[(i + 1) % n];
                    const float dx = C1.x - C0.x, dz = C1.y - C0.y;
                    const float len = std::sqrt(dx * dx + dz * dz);
                    if (len < 1e-4f) continue;
                    const float nrm[3] = {dz / len, 0.f, -dx / len};
                    static constexpr float up[3] = {0.f, 1.f, 0.f};
                    static constexpr float dn[3] = {0.f, -1.f, 0.f};
                    const float uu = len / o.bayWidth;
                    const float A[3] = {C0.x, yC1, C0.y}, B[3] = {C1.x, yC1, C1.y};
                    const float C[3] = {C1.x, yC0, C1.y}, D[3] = {C0.x, yC0, C0.y};
                    quad(trimBuf, A, B, C, D, nrm, trim, 0.f, uu, 0.30f, 0.18f);
                    const float T0[3] = {W0.x, yC1, W0.y}, T1[3] = {W1.x, yC1, W1.y};
                    quad(trimBuf, T0, T1, B, A, up, trim, 0.f, uu, 0.30f, 0.34f);
                    const float U0[3] = {W0.x, yC0, W0.y}, U1[3] = {W1.x, yC0, W1.y};
                    quad(trimBuf, D, C, U1, U0, dn, trim, 0.f, uu, 0.30f, 0.34f);
                }
            }

            // ── chimneys ───────────────────────────────────────────────────
            if (kind == K_GABLE && o.chimneys && gHalfL > 2.f) {
                static const std::array<std::array<float, 3>, 3> chimPal{{
                        {0.240f, 0.095f, 0.055f},// brick
                        {0.420f, 0.410f, 0.390f},// rendered/light concrete
                        {0.170f, 0.170f, 0.180f},// dark metal flue
                }};
                const auto& cc = chimPal[(h >> 24) % chimPal.size()];
                const float capCol[3] = {cc[0] * 0.45f, cc[1] * 0.45f, cc[2] * 0.45f};
                const auto insideOuter = [&](float px, float pz) {
                    bool in = false;
                    for (size_t i = 0, j = wallRing.size() - 1; i < wallRing.size(); j = i++) {
                        const Vector2& a = wallRing[i];
                        const Vector2& c = wallRing[j];
                        if ((a.y > pz) != (c.y > pz) &&
                            px < (c.x - a.x) * (pz - a.y) / (c.y - a.y) + a.x)
                            in = !in;
                    }
                    return in;
                };
                const int count = gHalfL > 12.f ? 2 : 1;
                const float hs = 0.5f * o.chimneySide;
                const float yB = yTop - 0.45f;
                const float yT = yTop + o.chimneyHeight;
                for (int ci = 0; ci < count; ++ci) {
                    const float frac =
                            static_cast<float>((h >> (4 + 8 * ci)) & 0xff) / 255.f - 0.5f;
                    float cu = (count == 2)
                                       ? gUc + (ci == 0 ? -0.5f : 0.5f) * gHalfL + frac * 2.f
                                       : gUc + frac * 0.9f * (gHalfL - 1.f);
                    if (!insideOuter(cu * gU.x + gVc * gV.x, cu * gU.y + gVc * gV.y)) {
                        cu = gUc;
                        if (!insideOuter(cu * gU.x + gVc * gV.x, cu * gU.y + gVc * gV.y))
                            continue;
                    }
                    const std::array<std::array<float, 2>, 4> sq{{
                            {cu - hs, gVc - hs},
                            {cu + hs, gVc - hs},
                            {cu + hs, gVc + hs},
                            {cu - hs, gVc + hs},
                    }};
                    Vector2 w2[4];
                    for (int i = 0; i < 4; ++i)
                        w2[i] = Vector2(sq[i][0] * gU.x + sq[i][1] * gV.x,
                                        sq[i][0] * gU.y + sq[i][1] * gV.y);
                    const float u1 = 0.35f + o.chimneySide / o.bayWidth;
                    const float v1 = 0.30f + (yT - yB) / floorH;
                    for (int i = 0; i < 4; ++i) {
                        const Vector2& p = w2[i];
                        const Vector2& q = w2[(i + 1) % 4];
                        const float dx = q.x - p.x, dz = q.y - p.y;
                        const float len = std::sqrt(dx * dx + dz * dz);
                        const float nrm[3] = {dz / len, 0.f, -dx / len};
                        trimBuf.push(p.x, yB, p.y, nrm, cc.data(), 0.35f, 0.30f);
                        trimBuf.push(q.x, yT, q.y, nrm, cc.data(), u1, v1);
                        trimBuf.push(q.x, yB, q.y, nrm, cc.data(), u1, 0.30f);
                        trimBuf.push(p.x, yB, p.y, nrm, cc.data(), 0.35f, 0.30f);
                        trimBuf.push(p.x, yT, p.y, nrm, cc.data(), 0.35f, v1);
                        trimBuf.push(q.x, yT, q.y, nrm, cc.data(), u1, v1);
                    }
                    static constexpr float up[3] = {0.f, 1.f, 0.f};
                    trimBuf.push(w2[0].x, yT, w2[0].y, up, capCol, 0.5f, 0.15f);
                    trimBuf.push(w2[2].x, yT, w2[2].y, up, capCol, 0.5f, 0.15f);
                    trimBuf.push(w2[1].x, yT, w2[1].y, up, capCol, 0.5f, 0.15f);
                    trimBuf.push(w2[0].x, yT, w2[0].y, up, capCol, 0.5f, 0.15f);
                    trimBuf.push(w2[3].x, yT, w2[3].y, up, capCol, 0.5f, 0.15f);
                    trimBuf.push(w2[2].x, yT, w2[2].y, up, capCol, 0.5f, 0.15f);
                }
            }
        }

        std::array<bool, M_COUNT> used{};
        const auto addMesh = [&](Buf& buf, int mat) {
            if (buf.pos.empty()) return;
            auto geometry = BufferGeometry::create();
            geometry->setAttribute("position", FloatBufferAttribute::create(buf.pos, 3));
            geometry->setAttribute("normal", FloatBufferAttribute::create(buf.nrm, 3));
            geometry->setAttribute("color", FloatBufferAttribute::create(buf.col, 3));
            geometry->setAttribute("uv", FloatBufferAttribute::create(buf.uv, 2));
            geometry->computeBoundingSphere();
            auto mesh = Mesh::create(geometry, mats[mat]);
            mesh->name = matNames[mat];
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            group->add(mesh);
            st.triangles += buf.pos.size() / 9;
            ++st.meshes;
            used[mat] = true;
        };
        for (auto& [key, chunk] : chunks)
            for (int m = 0; m < M_COUNT; ++m) addMesh(chunk.b[m], m);
        for (int m = 0; m < M_COUNT; ++m)
            if (used[m]) ++st.materials;
        if (o.stats) *o.stats = st;
        return group;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_GEOBUILDINGS_HPP
