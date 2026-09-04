// NORWAY TERRAIN — real-world geodata region pack (elevation + roads).
//
// Loads a "region pack" (Kartverket DTM elevation + NVDB roads, baked to the
// frozen pack format by the fetch pipeline) and renders it natively:
//
//   • terrain::GeoTerrainPack::load  — DEM float grid + road polylines + metadata.
//   • road::RoadNetwork              — one ribbon per road; composes every road's
//                                      corridor flattening into the unified ground
//                                      height the terrain tiles bake from.
//   • terrain::makeGeoProvider       — height = flattened DEM + corridor-faded
//                                      sub-grid relief; albedo = Norwegian splat
//                                      tinted toward gravel at the roadside.
//   • terrain::TileTerrain           — quadtree LOD over the provider, async bake,
//                                      cm-scale detail maps (Vulkan deferred).
//
// A directional sun + HDRI environment light it; a sea plane fills low ground.
//
// Usage:
//   norway_terrain [<pack-dir>]          (or env THREEPP_REGION_PACK)
//   norway_terrain --shot out.png [--frames N]   headless capture (forces Vulkan)
//   default pack: <PROJECT_FOLDER>/geodata/trollstigen

#include "capture_util.hpp"
#include "renderer_factory.hpp"

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/road/RoadNetwork.hpp"
#include "threepp/extras/terrain/CliffShell.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/GeoBuildings.hpp"
#include "threepp/extras/terrain/GeoTerrain.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/extras/terrain/TerrainScatter.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/extras/vegetation/CanopyForest.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
// Vulkan-only: Ocean (FFT-displaced water) and the VulkanRenderer/Core deferred
// tuning knobs live behind THREEPP_WITH_VULKAN. The demo itself runs on the
// forward GL backend too — those references are all guarded below.
#ifdef THREEPP_WITH_VULKAN
#include "threepp/objects/Ocean.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr float kDeg2Rad = 3.14159265358979323846f / 180.f;

    // Spot-check: how well do the conformed road SURFACE elevations track the
    // pack's surveyed road heights (point.y)? A well-behaved conform keeps |Δ|
    // small. We compare the ribbon SURFACE height (net.surfaceHeight = conformed
    // centerline + surfaceRaise) rather than the provider height, because the
    // provider now TRENCHES the terrain under the pavement (groundHeight drops it
    // by trenchDepth) — the road MESH still sits at the un-trenched surface, so
    // that is what must match the survey. NOTE: road y legitimately diverges from
    // the DTM on BRIDGES and TUNNELS (the DTM reads ground/water, not the deck),
    // so we report MEDIAN + p90 — robust to those outliers — rather than max, and
    // count the > 3 m points as likely bridges/tunnels instead of failures.
    void reportRoadConformance(const terrain::GeoTerrainPack& pack, const road::RoadNetwork& net) {
        std::vector<float> d;
        for (const auto& rd : pack.roads)
            for (const auto& p : rd.points) {
                const float surf = net.surfaceHeight(p.x, p.z);
                if (std::isnan(surf)) continue;// no road near this sample (shouldn't happen on a centerline)
                d.push_back(std::abs(surf - p.y));// ribbon surface vs surveyed road height
            }
        if (d.empty()) {
            std::cout << "[norway] no road points to cross-check\n" << std::flush;
            return;
        }
        std::sort(d.begin(), d.end());
        const auto pct = [&](float q) { return d[std::min(d.size() - 1, static_cast<size_t>(q * d.size()))]; };
        int over3 = 0;
        for (float v : d) if (v > 3.f) ++over3;
        std::cout << "[norway] road conformance vs pack y over " << d.size() << " points (roads="
                  << net.roadCount() << "): median |dh| " << pct(0.5f) << " m, p90 " << pct(0.9f)
                  << " m, max " << d.back() << " m; " << over3
                  << " pts > 3 m (likely bridges/tunnels)\n"
                  << std::flush;
    }

}// namespace

int main(int argc, char** argv) {

    // ── args ──────────────────────────────────────────────────────────────────
    std::string packArg;
    std::string shotPath;
    std::string seqPrefix;// --shotseq <prefix>: dump a numbered frame sequence
    int seqStart = 130;   // first frame index dumped
    int seqStep = 1;      // dump every Nth frame
    int shotFrames = 160;
    bool haveCam = false;
    Vector3 camPosArg, camTargetArg;
    float fovArg = 55.f;// --fov <deg>: a long lens compresses a fjord wall the
                        // way the reference photo does; 55° makes it recede.
    bool noRoadBias = false;// disable the road-aware LOD refinement (A/B compare)
    std::string viewName;   // --view <name>: a named camera/sun preset (see below)
    bool haveFov = false;
    std::string profilePath;// --road-profile <csv>: dump the height field along the longest road and exit
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--shotseq" && i + 1 < argc) seqPrefix = argv[++i];
        else if (a == "--seqstart" && i + 1 < argc) seqStart = std::atoi(argv[++i]);
        else if (a == "--seqstep" && i + 1 < argc) seqStep = std::atoi(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--fov" && i + 1 < argc) {
            fovArg = static_cast<float>(std::atof(argv[++i]));
            haveFov = true;
        } else if (a == "--view" && i + 1 < argc) viewName = argv[++i];
        else if (a == "--no-road-bias") noRoadBias = true;
        else if (a == "--road-profile" && i + 1 < argc) profilePath = argv[++i];
        else if (a == "--cam" && i + 1 < argc) {
            // --cam x,y,z,tx,ty,tz  — place the camera at (x,y,z) looking at (tx,ty,tz).
            float v[6] = {0, 0, 0, 0, 0, 0};
            std::string s = argv[++i];
            int k = 0;
            size_t pos = 0;
            while (k < 6 && pos <= s.size()) {
                const size_t comma = s.find(',', pos);
                const std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!tok.empty()) v[k++] = std::stof(tok);
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            if (k == 6) {
                haveCam = true;
                camPosArg.set(v[0], v[1], v[2]);
                camTargetArg.set(v[3], v[4], v[5]);
            } else {
                std::cerr << "[norway] --cam needs 6 comma-separated floats: x,y,z,tx,ty,tz\n";
            }
        } else if (!a.empty() && a[0] != '-') packArg = a;
    }
    // ── named views ───────────────────────────────────────────────────────────
    // A shot worth judging has to be repeatable, and a 6-float --cam is not a
    // name. `reference` is the geiranger fjord-wall framing the realism plan is
    // judged against: drone height (150 m over the water), a long lens, and the
    // sun raking in from the image LEFT low enough that the wall's own relief
    // puts its right half in shadow. Distance is 950 m rather than the photo's
    // ~650 m because THIS wall is 570 m tall against the photo's ~400 m — the
    // angular size, not the metre count, is what has to match. --cam / --fov /
    // NT_SUN_AZ / NT_SUN_EL still win when given explicitly.
    float sunAzDeg = 215.f, sunElDeg = 34.f;
    bool viewSun = false;// a view preset owns the sun: don't re-aim it at the HDRI's
    if (!viewName.empty()) {
        if (viewName == "reference") {
            if (!haveCam) {
                haveCam = true;
                camPosArg.set(1000.f, 150.f, -1470.f);   // over the fjord
                camTargetArg.set(1000.f, 117.f, -520.f);// 2° down: the shoreline
                                                        // lands ~1/5 up the frame,
                                                        // so the wall stays the subject
            }
            if (!haveFov) fovArg = 35.f;// long lens: the wall must not recede
            sunAzDeg = 118.f;           // from +X / -Z = upper LEFT of this frame
            sunElDeg = 26.f;            // low enough to rake the foliation
            viewSun = true;
        } else {
            std::cerr << "[norway] unknown --view '" << viewName << "' (known: reference)\n";
        }
    }
    if (packArg.empty()) {
        if (const char* env = std::getenv("THREEPP_REGION_PACK")) packArg = env;
    }
    if (packArg.empty()) {
        packArg = (std::filesystem::path(PROJECT_FOLDER) / "geodata" / "aalesund").string();
    }

    // ── load the pack ──────────────────────────────────────────────────────────
    terrain::GeoTerrainPack pack;
    try {
        pack = terrain::GeoTerrainPack::load(packArg);
    } catch (const std::exception& e) {
        std::cerr << "[norway] failed to load region pack '" << packArg << "': " << e.what()
                  << std::endl;
        return 1;
    }
    const terrain::GeoRegion& reg = pack.region;
    std::cout << "[norway] loaded pack '" << reg.name << "' (" << packArg << ")\n"
              << "         worldSize " << reg.worldSize << " m, dim " << reg.dim
              << ", height " << reg.heightMin << ".." << reg.heightMax << " m, sea " << reg.seaLevel
              << ", roads " << pack.roads.size() << ", buildings " << pack.buildings.size() << "\n"
              << "         attribution: " << reg.attribution << "\n"
              << std::flush;

    // ── road network + unified ground height ───────────────────────────────────
    std::vector<road::RoadSpec> specs;
    specs.reserve(pack.roads.size());
    for (const auto& gr : pack.roads) {
        road::RoadSpec s;
        s.id = gr.id;
        s.category = gr.category;
        s.width = gr.width;
        s.points = gr.points;
        specs.push_back(std::move(s));
    }
    road::RoadNetwork network(std::move(specs));
    // BAKED-ROAD pipeline. Conform roads to the RAW DEM height first (roads
    // must meet real ground) with the elevation PROFILE enabled: the pack's
    // NVDB point heights classify every span — data height well above ground
    // (or ground = water) ⇒ BRIDGE deck that spans the vertical U instead of
    // draping into it; data height well below ground ⇒ TUNNEL, excluded; long
    // water runs that never clear the water ⇒ ferry legs, excluded (a road is
    // never submerged). THEN bake the roadbed into the grid: the paved band is
    // SET to the exact road surface (cut AND fill, dead flat across), so the
    // provider is a pure bicubic of the carved grid and the terrain itself IS
    // the road — the visual comes from the albedo paint (paintRoads below),
    // which is mip-filtered tile texture and therefore cannot coverage-shimmer
    // at distance the way 1 px ribbon geometry does. Only bridge decks remain
    // ribbon meshes (buildBridgeMeshes below).
    road::RoadProfileOptions rpo;
    rpo.enabled = true;
    rpo.seaLevel = reg.seaLevel;
    // NT_ROAD_SMOOTH sets the arc-length grade-smoothing window in metres (0 = the
    // legacy drape). Off by default here (the viewer keeps roads hugging terrain);
    // norway_drive turns it on for a drivable vertical alignment.
    if (const char* e = std::getenv("NT_ROAD_SMOOTH"); e && e[0] != '\0') rpo.gradeSmoothing = std::strtof(e, nullptr);
    network.conformTo([&pack](float x, float z) { return pack.grid.sampleBicubic(x, z); }, 14, rpo);
    terrain::RoadCarveOptions rco;
    rco.bakeSurface = true;
    // Defaults: full 6 m cut band (the tile-quad anti-poke guarantee — narrowing
    // it lets uphill quads slice through the pavement) + narrow asymmetric FILL
    // (fillInflate/fillFeather) so steep sidehills get a tight embankment instead
    // of a wide berm shelf. Env overrides for A/B.
    if (const char* e = std::getenv("NT_CARVE_INFLATE"); e && e[0] != '\0') rco.inflate = std::strtof(e, nullptr);
    if (const char* e = std::getenv("NT_CARVE_FEATHER"); e && e[0] != '\0') rco.feather = std::strtof(e, nullptr);
    if (const char* e = std::getenv("NT_CARVE_FILL"); e && e[0] != '\0') rco.fillInflate = std::strtof(e, nullptr);
    if (const char* e = std::getenv("NT_CARVE_FILLFEATHER"); e && e[0] != '\0') rco.fillFeather = std::strtof(e, nullptr);
    terrain::carveRoads(pack.grid, network, rco);

    // Synthetic bathymetry. The DTM stores water as a flat sheet at EXACTLY
    // seaLevel (no soundings), so the seabed would sit 0.15 m under the ocean
    // surface: the whole sea reads as a knee-deep pond (bottom splat visible
    // through the water), and at any real wind the wave troughs dip below the
    // bed and the terrain pokes through the sea as green polygonal plates.
    // Sink the water cells a few metres: open water goes optically deep while
    // the shore keeps a narrow shallow band (the drop spans the bicubic
    // support, ~2 cells ≈ 4 m). AFTER conformTo + carveRoads — road profile
    // classification (bridges/ferries) must see the real DTM water level, and
    // only excluded/deck-spanning segments cross water so no roadbed cell sits
    // at sea level. NT_SEA_DEPTH overrides; 0 restores the flat sheet.
    float seaDepth = 6.f;
    if (const char* e = std::getenv("NT_SEA_DEPTH"); e && e[0] != '\0') seaDepth = std::strtof(e, nullptr);
    if (seaDepth > 0.f && reg.heightMin < 1.0f) {
        for (float& h : pack.grid.data())
            if (h <= reg.seaLevel + 0.05f) h -= seaDepth;
    }

    // ── provider + tiles ───────────────────────────────────────────────────────
    terrain::GeoTerrainOptions gopt;
    gopt.snowHeightMin = std::max(reg.heightMax - 350.f, 900.f);// scene-relative snowline
    gopt.grassHeightMax = std::clamp(reg.heightMin + 0.45f * (reg.heightMax - reg.heightMin), 200.f, 900.f);
    gopt.wetlandBand = 6.f;
    gopt.paintRoads = true;// terrain IS the road (see the conformTo comment above)
    // Near-tile splat texels are ~0.6-1.3 m; a feather below the texel size
    // can't anti-alias the paint edge and reads as a staircase up close.
    gopt.roadEdgeFeather = 1.2f;
    // (envSet is declared later; this runs before it exists)
    gopt.paintUrban = !std::getenv("NT_NO_URBAN");// grey town fabric under dense buildings (A/B)
    // CLIFF relief + gneiss band, on 1 m packs only. The gate is the DEM's own
    // resolution, not the scene: sub-metre benches on a 2 m DEM would be
    // INVENTING structure below the data's sampling, which is exactly the soft
    // fBm look this replaces. At 1 m the benches sharpen structure the lidar
    // actually measured. NT_NO_CLIFF forces it off for an A/B.
    const float gridStep = reg.worldSize / static_cast<float>(reg.dim - 1);
    const bool cliffPack = gridStep <= 1.5f && !std::getenv("NT_NO_CLIFF");
    // CONTOUR-STRIP CLIFF SHELL (phase 2b): a free-parametrised skin over the
    // steep faces, u = contour arc length / v = world height. It OWNS the wall
    // relief once it is on — a positive terrain relief under it would poke
    // through the shell's 0.35 m offset — so the two are mutually exclusive.
    const bool shellOn = gridStep <= 1.5f && !std::getenv("NT_NO_SHELL");
    gopt.cliffRelief = cliffPack && !shellOn;
    const terrain::TerrainProvider prov = terrain::makeGeoProvider(pack, network, gopt);

    reportRoadConformance(pack, network);

    // Diagnostic: dump the provider height field along the longest road at fine
    // arc spacing (columns: arc s, centerline x/z/y, ribbon surface height, and
    // provider.height at lateral offsets 0/±2/±5/±9 m + raw DEM) then exit.
    // Lets an external script check the field the near tiles ACTUALLY sample for
    // longitudinal facets / segment-selection discontinuities, independent of
    // any tile machinery.
    if (!profilePath.empty()) {
        const auto cl = network.longestRoadCenterline();
        std::ofstream csv(profilePath);
        csv << "s,x,z,cly,surf,g0,gp2,gp5,gp9,gm2,gm5,gm9,dem\n";
        float s = 0.f;
        const float step = 0.25f;
        for (size_t i = 0; i + 1 < cl.size(); ++i) {
            const Vector3& a = cl[i];
            const Vector3& b = cl[i + 1];
            const float dx = b.x - a.x, dz = b.z - a.z;
            const float len = std::sqrt(dx * dx + dz * dz);
            if (len < 1e-6f) continue;
            const float px = -dz / len, pz = dx / len;// unit perpendicular (XZ)
            for (float t = 0.f; t < len; t += step) {
                const float x = a.x + dx * (t / len), z = a.z + dz * (t / len);
                const float y = a.y + (b.y - a.y) * (t / len);
                auto g = [&](float lat) { return prov.height(x + px * lat, z + pz * lat); };
                csv << (s + t) << ',' << x << ',' << z << ',' << y << ','
                    << network.surfaceHeight(x, z) << ','
                    << g(0.f) << ',' << g(2.f) << ',' << g(5.f) << ',' << g(9.f) << ','
                    << g(-2.f) << ',' << g(-5.f) << ',' << g(-9.f) << ','
                    << pack.grid.sampleBicubic(x, z) << '\n';
            }
            s += len;
        }
        std::cout << "[norway] wrote road profile (" << s << " m) to " << profilePath << "\n";
        return 0;
    }

    terrain::TileTerrainOptions tileOpts;
    tileOpts.worldSize = reg.worldSize;
    tileOpts.rootGrid = 4;
    tileOpts.maxDepth = 5;
    tileOpts.tileRes = 96;
    tileOpts.splitFactor = 1.2f;
    tileOpts.mergeFactor = 1.7f;
    // Splat texel density. NOTE (measured 2026-07-15, aalesund static-cam
    // frame-diff A/B): 2 vs 4 texels/quad is flicker-IDENTICAL to three
    // decimals — the distant-tile shimmer is NOT albedo texture aliasing
    // (material textures get full mip chains + trilinear/aniso on the Vulkan
    // path regardless of the texture's own minFilter). 4× only quadruples
    // bake time. NT_SPLAT_TPQ overrides for A/B.
    tileOpts.splatTexelsPerQuad = 2;
    if (const char* tpq = std::getenv("NT_SPLAT_TPQ"); tpq && tpq[0] != '\0')
        tileOpts.splatTexelsPerQuad = std::atoi(tpq);
    tileOpts.asyncBake = true;
    // Road-aware LOD: subdivide road-corridor tiles ~2.2× sooner/deeper so the
    // ribbon stays crisp at mid distance (terrain interp error shrinks with tile
    // size). The 1.2/1.7 split/merge dead band is preserved under the bias.
    if (!noRoadBias) {
        tileOpts.refineBias = [&network](float cx, float cz, float half) {
            return network.corridorIntersects(cx, cz, half) ? 2.2f : 1.0f;
        };
    }
    // Hoisted out of the block below: the cliff shell shades the SAME wall as
    // the tiles and must carry the same band set — two different rock
    // generators meeting at the shell boundary would draw the boundary.
    terrain::TerrainBandSet bandSet;
    bool bandSetBuilt = false;
    {
        // Per-band STRUCTURE sets (grass/rock/scree/snow): the terrain shader
        // resolves them at screen density over the macro splat, selected by
        // the baked weight map (provider.weights ← the Norwegian splat rules).
        // NT_NO_BANDS=1 falls back to the legacy single detail layer for A/B.
        const bool noBands = [] {
            const char* v = std::getenv("NT_NO_BANDS");
            return v && v[0] == '1';
        }();
        if (!noBands) {
            // On a cliff pack the rock slot carries the GNEISS generator
            // (foliation hanging vertically on the triplanar side projections,
            // joint blocks, wet veins) instead of the generic plate rock.
            bandSet = terrain::makeTerrainBandSet(
                    4242u, cliffPack ? terrain::BandKind::Cliff : terrain::BandKind::Rock);
            bandSetBuilt = true;
            const terrain::TerrainBandSet& bands = bandSet;
            for (size_t i = 0; i < 4; ++i) {
                tileOpts.bandAlbedo[i] = bands.band[i].albedo;
                tileOpts.bandNormalRough[i] = bands.band[i].normalRough;
            }
            tileOpts.bandRepeat = bands.repeat;
            tileOpts.bandRoughness = bands.roughness;
            // A wall gets the band layer HOT: the macro splat under it is baked
            // in XZ and therefore smears vertically on a near-vertical face, so
            // the triplanar band is the only layer that can put structure there
            // — it has to out-shout the smear, not politely modulate it.
            tileOpts.bandStrength = cliffPack ? 1.0f : 0.8f;
            tileOpts.bandNormalScale = cliffPack ? 2.2f : 1.4f;// relief lighting carries the depth read
            tileOpts.bandRoughStrength = 0.6f;
        }
        // Legacy cm-scale detail layer — the fallback wherever bands are off
        // (env override above, or the GL backend which ignores band fields).
        const terrain::DetailMaps dm = terrain::makeDetailMaps({});
        tileOpts.detailMap = dm.albedo;
        tileOpts.detailNormalMap = dm.normalRough;
        tileOpts.detailRepeat = 0.6f;
        tileOpts.detailStrength = 0.7f;
        tileOpts.detailNormalScale = 1.0f;
        tileOpts.detailRoughStrength = 0.5f;
    }

    // ── renderer ────────────────────────────────────────────────────────────────
    const bool headless = !shotPath.empty() || !seqPrefix.empty();
    // NT_SIZE="wxh" pins the framebuffer (Canvas::size() is a REQUEST — the
    // shot's real resolution is framebufferSize()); a judged 1:1 crop needs a
    // known native size, and the default window is smaller than 1080p.
    WindowSize canvasSize{1280, 800};
    if (const char* cs = std::getenv("NT_SIZE")) {
        int cw = 0, ch = 0;
        if (std::sscanf(cs, "%dx%d", &cw, &ch) == 2 && cw > 63 && ch > 63) canvasSize = {cw, ch};
    }
    Canvas canvas("threepp - NORWAY TERRAIN", {{"vsync", false}, {"size", canvasSize}});
    // Headless capture forces the Vulkan deferred renderer (detail-map layer is
    // Vulkan-only); NT_GL=1 captures the forward GL path instead (renderer-specific
    // artifact A/B — e.g. tile-albedo mip behaviour). Interactive runs keep the
    // renderer-select menu.
    auto renderer = !headless ? createRenderer(canvas)
                    : createRenderer(canvas, std::getenv("NT_GL") ? GraphicsAPI::OpenGL
                                                                  : GraphicsAPI::Vulkan);
    // Neutral on the forward GL path, ACESFilmic on Vulkan deferred —
    // three.js ACESFilmic's 1/0.6 viewing-environment gain washes this bright
    // scene out on the forward paths (see the NorwayDrive tone-mapping note).
#ifdef THREEPP_WITH_VULKAN
    renderer->toneMapping = dynamic_cast<VulkanRenderer*>(renderer.get())
                                    ? ToneMapping::ACESFilmic
                                    : ToneMapping::Neutral;
#else
    renderer->toneMapping = ToneMapping::Neutral;// forward GL path
#endif
    renderer->toneMappingExposure = 1.0f;

    // ── instrumentation / A-B knobs (env-driven, so one build sweeps configs) ──
    // NT_DLSS=0/NT_FSR=0 → disable that upscaler; both off = built-in TAA only.
    // NT_DENOISE=0 → deferred SVGF denoiser off (also THREEPP_DENOISE=0).
    // NT_FIREFLY=<v> → setFireflyClamp (0 = no clamp). NT_DEBUGVIEW=<n> → G-buffer
    // blit (1 normal, 2 motion, 3 ids, 4 albedo, 5 depth). NT_RENDERSCALE=<v>.
    // NT_SUNRADIUS=<deg> → setSunAngularRadius. NT_SEA_ROUGH / NT_SEA_METAL /
    // NT_NO_SEA / NT_SUN_ALIGN affect the demo scene (below).
    const auto envF = [](const char* k, float def) {
        const char* e = std::getenv(k);
        return e ? std::strtof(e, nullptr) : def;
    };
    const auto envSet = [](const char* k) { const char* e = std::getenv(k); return e && e[0] != '\0'; };
#ifdef THREEPP_WITH_VULKAN
    auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get());
    if (vk) {
        // DEFAULT: 2× G-buffer MSAA, no upscaler. With MSAA the raster runs
        // UNJITTERED — measured on this scene: the jittered TAA/DLSS paths leave
        // the final image trembling with the 8-phase Halton pattern (global
        // ±0.9 px shifts at a STATIC camera; frame-diff mean 3.2/255, p99 26 at
        // every high-contrast edge — "the roads shake"), while MSAA2-unjittered
        // is rock-solid (mean 0.28, p99 1.7, shift exactly 0) at the same fps
        // (53.6 vs 54.2). Terrain/roads are matte + mip/aniso-filtered, so the
        // loss of temporal AA doesn't bite here. NT_DLSS=1 / NT_MSAA=1 restore
        // the jittered path for A/B.
        // vk->setGbufferMsaa(2);
        // vk->setDlss(false);
        // vk->setFsr(false);
        if (envSet("NT_DLSS")) vk->setDlss(std::getenv("NT_DLSS")[0] != '0');
        if (envSet("NT_FSR")) vk->setFsr(std::getenv("NT_FSR")[0] != '0');
        if (envSet("NT_DENOISE")) vk->setDenoise(std::getenv("NT_DENOISE")[0] != '0');
        if (envSet("NT_FIREFLY")) vk->setFireflyClamp(envF("NT_FIREFLY", 30.f));
        if (envSet("NT_DEBUGVIEW")) vk->setHybridDebugView(std::atoi(std::getenv("NT_DEBUGVIEW")));
        if (envSet("NT_RENDERSCALE")) vk->setRenderScale(envF("NT_RENDERSCALE", 1.f));
        if (envSet("NT_SUNRADIUS")) vk->setSunAngularRadius(envF("NT_SUNRADIUS", 0.5f));
        if (envSet("NT_MSAA")) {
            vk->setGbufferMsaa(static_cast<uint32_t>(std::atoi(std::getenv("NT_MSAA"))));
            std::cout << "[norway] gbufferMsaa = " << vk->gbufferMsaa() << "\n";
        }
        if (envSet("NT_AO")) vk->setDeferredAO(std::getenv("NT_AO")[0] != '0');
        if (envSet("NT_PROBEGI")) vk->setProbeGI(std::getenv("NT_PROBEGI")[0] != '0');
    }
#endif// THREEPP_WITH_VULKAN

    Scene scene;
    RGBELoader rgbe;
    if (auto env = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = env;
        scene.environment = env;
    } else {
        scene.background = Color(0.55f, 0.70f, 0.92f);
        std::cerr << "[norway] HDRI env not found - flat sky fallback\n";
    }

    auto tiles = terrain::TileTerrain::create(prov, tileOpts);
    tiles->name = "norway_terrain";
    scene.add(tiles);

    // Near-field ground cover: instanced stones + grass tufts scattered from
    // the SAME provider (weights keep them off pavement and pick species by
    // band). Texture structure can't survive grazing-angle minification in the
    // last metres — physical props are what carry that range. NT_NO_SCATTER=1
    // disables for A/B.
    std::shared_ptr<terrain::TerrainScatter> scatter;
    if (!envSet("NT_NO_SCATTER")) {
        scatter = terrain::TerrainScatter::create(prov, {});
        scatter->name = "ground_cover";
        scene.add(scatter);
    }

    // Roads are BAKED into the terrain (carve + paint above); bridge decks are
    // always ribbon geometry. NEAR-DETAIL HYBRID: on-ground ribbon CHUNKS
    // (baked asphalt texture, crisp edges, lane markings — "stand-on" quality
    // the ~1 m painted splat texels cannot hold) are distance-culled per frame
    // below: within NT_ROAD_RIBBON_DIST the ribbon renders kSurfaceRaise above
    // the baked bed; beyond it the chunk vanishes and the painted bed alone is
    // the road — mip-filtered texture, so no sub-pixel coverage shimmer.
    // NT_RIBBON_ROADS=1 restores the legacy full-ribbon look for A/B.
    std::vector<std::pair<Object3D*, Vector3>> roadChunkCenters;// chunk + bounding center
    std::vector<float> roadChunkRadii;
    if (envSet("NT_RIBBON_ROADS")) {
        scene.add(network.buildMeshes());
    } else {
        scene.add(network.buildBridgeMeshes());
        auto chunks = network.buildGroundChunkMeshes();
        for (auto* child : chunks->children) {
            auto* mesh = child->as<Mesh>();
            if (!mesh) continue;
            auto geo = mesh->geometry();
            geo->computeBoundingSphere();
            roadChunkCenters.emplace_back(child, geo->boundingSphere->center);
            roadChunkRadii.push_back(geo->boundingSphere->radius);
        }
        std::cout << "[norway] road ribbon chunks: " << roadChunkCenters.size() << "\n" << std::flush;
        scene.add(chunks);
    }
    const float ribbonDist = envF("NT_ROAD_RIBBON_DIST", 600.f);// 6 m road ≈ 5 px here

    // Buildings (packs fetched with --buildings): extruded OSM footprints with
    // nDSM-measured heights, batched into 500 m chunk meshes with per-building
    // vertex colours. NT_NO_BUILDINGS=1 hides them, NT_FLAT_ROOFS=1 disables
    // the gable heuristic — both for A/B.
    if (!pack.buildings.empty() && !envSet("NT_NO_BUILDINGS")) {
        terrain::GeoBuildingsOptions bo;
        bo.pitchedRoofs = !envSet("NT_FLAT_ROOFS");
        auto buildings = terrain::buildGeoBuildingMeshes(pack, bo);
        std::cout << "[norway] buildings: " << pack.buildings.size() << " footprints in "
                  << buildings->children.size() << " chunk meshes\n" << std::flush;
        scene.add(buildings);
    }

    // Sea on low / coastal packs. A huge FLAT MeshStandardMaterial plane
    // (roughness ~0.15, metalness 0) drives the deferred renderer's STOCHASTIC
    // opaque-reflection channel — whose SVGF/ReBLUR temporal reproject can't hold
    // a stable history on a single vast smooth quad under sub-pixel TAA/DLSS
    // jitter: the reproject validity + motion magnitude alternate frame-to-frame,
    // so the reflection denoiser boils and the whole plane flickers (Bug B
    // symptoms 2/3). The engine's Ocean wears a TRANSMISSIVE MeshPhysicalMaterial,
    // which the deferred shader routes to the INLINE water-reflection path (no
    // stochastic reflection accumulator → stable), and its FFT relief also breaks
    // up the single-quad grazing-precision problem. NT_FLAT_SEA keeps the old
    // plane for A/B.
    const bool coastal = reg.heightMin < 1.0f && !envSet("NT_NO_SEA");
#ifdef THREEPP_WITH_VULKAN
    std::shared_ptr<Ocean> ocean;// kept alive for the settings panel's live sea controls
#endif
    if (coastal) {
        std::shared_ptr<Object3D> seaObj;
#ifdef THREEPP_WITH_VULKAN
        const bool flatSea = envSet("NT_FLAT_SEA");
#else
        // Ocean (FFT-displaced water) is a Vulkan-only object; the forward
        // The GL path always gets the flat MeshStandardMaterial sea plane.
        const bool flatSea = true;
#endif
        if (flatSea) {
            auto seaMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                               .color(Color(0.045f, 0.09f, 0.13f))
                                                               .roughness(envF("NT_SEA_ROUGH", 0.15f))
                                                               .metalness(envF("NT_SEA_METAL", 0.f)));
            auto sea = Mesh::create(PlaneGeometry::create(reg.worldSize * 1.2f, reg.worldSize * 1.2f), seaMat);
            sea->rotation.x = -math::PI / 2.f;
            seaObj = sea;
        }
#ifdef THREEPP_WITH_VULKAN
        else {
            Ocean::Options oo;
            oo.size = reg.worldSize * 1.2f;
            oo.resolution = 384;
            // NT_SEA_SCATTER=1 opts the cliff pack into Ocean::Look::Fjord: the
            // GLACIAL recipe, whose turquoise comes from volume in-scatter
            // instead of a stretched attenuation distance (see below). Off by
            // default so this demo's shot is unchanged; it is the A/B switch the
            // water-body-scatter work is judged on.
            const bool seaScatter = cliffPack && envF("NT_SEA_SCATTER", 0.f) > 0.5f;
            oo.look = seaScatter ? Ocean::Look::Fjord
                                 : Ocean::Look::Ocean;// open sea — never the small-body pond recipe
            // Beaufort 4-ish: Phillips amplitude ∝ V⁴ and the dominant swell
            // wavelength ∝ V², so below ~7 m/s the sea is glassy AND its short
            // chop falls under this sheet's ~25 m vertex spacing — reads dead
            // flat. Live-tunable in the settings panel; NT_SEA_WIND pins it.
            oo.windSpeed = envF("NT_SEA_WIND", cliffPack ? 5.5f : 9.0f);
            oo.windTheta = 215.f * kDeg2Rad;// swell rolling in from the SW (the sun heading)
            oo.choppiness = 0.45f;
            oo.tileSize1 = 90.f;
            oo.tileSize2 = 7.f;
            oo.fftSize = 512;// half the default — a big calm sea needs less spectral detail; recovers FPS
            ocean = Ocean::create(oo);
            if (auto* wm = ocean->material()->as<MeshPhysicalMaterial>()) {
                if (seaScatter) {
                    // Look::Fjord already carries the recipe (attenuation back
                    // near physical at 1.5 m + scatterColor/scatterDistance);
                    // nothing to override here. NT_SEA_ATTEN still pins the
                    // absorption length for A/B sweeps; NT_SEA_SCATTER_DIST and
                    // NT_SEA_SCATTER_COL=r,g,b (linear) sweep the in-scatter
                    // recipe without a rebuild of Ocean.cpp.
                    wm->attenuationDistance = envF("NT_SEA_ATTEN", wm->attenuationDistance);
                    wm->scatterDistance = envF("NT_SEA_SCATTER_DIST", wm->scatterDistance);
                    if (const char* e = std::getenv("NT_SEA_SCATTER_COL"); e && e[0] != '\0') {
                        float r = 0.f, g = 0.f, b = 0.f;
                        if (std::sscanf(e, "%f,%f,%f", &r, &g, &b) == 3) wm->scatterColor = Color(r, g, b);
                    }
                } else if (cliffPack) {
                    // GLACIAL fjord: rock flour in suspension scatters, so the
                    // water is opaque turquoise-teal rather than a dark mirror.
                    // The deferred body is tint = attenuationColor ^ (2·thickness
                    // / attenuationDistance) with Ocean's thickness = 2, so the
                    // EXPONENT is the knob that matters: 0.75 m gave 5.3 and
                    // powered the colour down to (2e-7, 0.010, 0.006) — black
                    // water, the "dark mirror" of the phase-1 crops, and the
                    // reason the knobs looked inert. 4 m ⇒ exponent 1, i.e. the
                    // body IS the attenuation colour, lit by skylight.
                    wm->attenuationColor = Color(0.10f, 0.50f, 0.48f);
                    wm->attenuationDistance = envF("NT_SEA_ATTEN", 4.0f);
                } else {
                    wm->attenuationColor = Color(0.045f, 0.13f, 0.16f);// dark Nordic fjord water
                    wm->attenuationDistance = 1.9f;
                }
            }
            seaObj = ocean;
        }
#endif// THREEPP_WITH_VULKAN
        // Slightly ABOVE the DTM's sea sheet — which the bathymetry carve above
        // sank by NT_SEA_DEPTH so wave troughs have water under them (with the
        // carve off, water reads as seaLevel exactly and the provider suppresses
        // relief noise there); real land starts well above this.
        seaObj->position.y = reg.seaLevel + 0.15f;
        seaObj->name = "sea";
        scene.add(seaObj);
    }

    // Directional sun (raking, ~SW) + HDRI ambient.
    auto sun = DirectionalLight::create(Color(1.0f, 0.96f, 0.88f), 2.8f);
    {
        // NT_SUN_AZ / NT_SUN_EL override the analytic sun direction (for aligning
        // it to the HDRI's baked sun disk — A/B of Bug B symptom 1).
        const float az = envF("NT_SUN_AZ", sunAzDeg) * kDeg2Rad, el = envF("NT_SUN_EL", sunElDeg) * kDeg2Rad;
        sun->position.set(std::cos(el) * std::sin(az), std::sin(el), std::cos(el) * std::cos(az));
        sun->position.multiplyScalar(std::max(reg.worldSize, 2000.f));
    }
    Object3D sunTarget;
    sunTarget.position.set(0.f, reg.heightMin, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    // ── camera above the longest road, looking along it ─────────────────────────
    PerspectiveCamera camera(fovArg, canvas.aspect(), 1.f, std::max(reg.worldSize * 3.f, 12000.f));
    Vector3 roadMid, roadDir;
    if (network.longestRoad(roadMid, roadDir)) {
        const float groundY = tiles->heightAt(roadMid.x, roadMid.z);
        camera.position.set(roadMid.x - roadDir.x * 130.f, groundY + 150.f, roadMid.z - roadDir.z * 130.f);
    } else {
        camera.position.set(0.f, reg.heightMax + 200.f, reg.worldSize * 0.4f);
        roadMid.set(0.f, reg.heightMin, 0.f);
        roadDir.set(0.f, 0.f, 1.f);
    }
    OrbitControls controls{camera, canvas};
    controls.target.set(roadMid.x + roadDir.x * 200.f,
                        tiles->heightAt(roadMid.x + roadDir.x * 200.f, roadMid.z + roadDir.z * 200.f),
                        roadMid.z + roadDir.z * 200.f);
    // Explicit viewpoint override (e.g. to re-shoot the failure viewpoint).
    if (haveCam) {
        camera.position.copy(camPosArg);
        controls.target.copy(camTargetArg);
    }
    controls.update();

    // ── contour-strip cliff shell ──────────────────────────────────────────────
    // The terrain is a heightfield, so every baked tile map is a function of
    // (x, z): on a near-vertical wall each texel is one stretched vertical
    // column and NOTHING baked on the tiles can vary along a column. That is the
    // vertical-stripe smear the phase-1b/2 shots kept showing. The shell is a
    // separate free mesh over the steep faces whose parametrisation is
    // u = contour arc length, v = world height — metric on the wall, so texel
    // density is uniform and content (ledge rows, seepage streaks) can vary
    // along v. NT_NO_SHELL=1 for the A/B, NT_SHELL_STEP=<m> for the level
    // spacing. 2 m packs (trollstigen/aalesund) never build one.
    if (shellOn && bandSetBuilt) {
        terrain::CliffShellOptions so;
        so.seaLevel = reg.seaLevel;
        so.levelStep = envF("NT_SHELL_STEP", 2.f);
        so.snowHeightMin = gopt.snowHeightMin;
        so.snowFeather = gopt.snowFeather;
        so.canopyForestMin = gopt.canopyForestMin;
        // Same ROI reasoning as the forest: a 4 km pack is 16 M cells, the shot
        // looks at part of it, and the shell is static geometry submitted whole.
        so.centerX = controls.target.x;
        so.centerZ = controls.target.z;
        so.halfExtent = envF("NT_SHELL_EXTENT", 1200.f);
        auto shellRoot = Group::create();
        shellRoot->name = "cliff_shell_root";
        const auto stShell = terrain::buildCliffShell(*shellRoot, pack, bandSet, so,
                                                      cliffPack ? 1.0f : 0.8f,
                                                      cliffPack ? 2.2f : 1.4f);
        scene.add(shellRoot);
        std::cout << "[norway] cliff shell: " << stShell.regions << " regions, "
                  << stShell.levels << " levels @ " << so.levelStep << " m, "
                  << stShell.maskCells << " mask cells, " << stShell.polylines
                  << " contours, " << stShell.strips << " strips, "
                  << stShell.vertices << " verts / " << stShell.triangles << " tris, "
                  << (stShell.atlasTexels / 1024) << " k atlas texels ("
                  << "trace " << stShell.traceSeconds << " s, stitch "
                  << stShell.stitchSeconds << " s, bake " << stShell.bakeSeconds << " s)\n"
                  << std::flush;
    }

    // ── canopy-driven forest ───────────────────────────────────────────────────
    // Packs fetched with --canopy carry a canopy height model (DOM - DTM): metres
    // of vegetation per cell, i.e. a MEASUREMENT of where forest stands and how
    // tall it is. Trees go exactly there, at that height, instead of on a
    // slope/elevation rule that invents a forest. Packs without a CHM get nothing
    // (trollstigen/aalesund unchanged). NT_NO_FOREST=1 for the A/B,
    // NT_FOREST_CAP=<n> for the instance budget.
    if (pack.hasCanopy() && !envSet("NT_NO_FOREST")) {
        const auto tf0 = std::chrono::high_resolution_clock::now();

        vegetation::CanopySiteOptions so;
        so.seaLevel = reg.seaLevel;
        // The shot only ever looks at part of a 4 km pack, and the instance budget
        // is better spent dense near the subject than thin across the whole square.
        so.centerX = controls.target.x;
        so.centerZ = controls.target.z;
        so.halfExtent = envF("NT_FOREST_EXTENT", 1100.f);
        // NT_TREELINE=<m> pins the treeline (9999 = effectively off);
        // NT_FOREST_NOGATE=1 restores the phase-3 detector exactly — no treeline,
        // no neighbourhood support, no tightened high-elevation slope gate — so
        // the "trees on the plateau" A/B needs no stash.
        so.treelineElevation = envF("NT_TREELINE", 0.f);
        if (envSet("NT_FOREST_NOGATE")) {
            so.treelineElevation = 1e9f;
            so.minNeighborSupport = 0;
            so.highSlopeElevation = 1e9f;
        }
        vegetation::CanopySiteReport srep;
        const auto sites = vegetation::detectTreeSites(pack.canopy, pack.grid, so, &srep);
        std::cout << "[norway] canopy sites: " << srep.peaks << " CHM peaks -> "
                  << sites.size() << " sites (rejected: support " << srep.rejectedSupport
                  << ", treeline " << srep.rejectedTreeline << ", high-slope "
                  << srep.rejectedHighSlope << ", slope " << srep.rejectedSlope
                  << ", ground " << srep.rejectedGround << "); treeline "
                  << srep.treelineElevation << " m +-" << so.treelineFeather
                  << ", highest site " << srep.highestSite << " m\n"
                  << std::flush;

        // NT_FOREST_LOD=0 restores the phase-2 single-tier planting (the A/B).
        // The default is the camera-following cell LOD: the old path splits its
        // tiers ONCE against the startup camera, which is a still-frame trick —
        // walk the camera in and the near trees are still blobs.
        const bool lodForest = !envSet("NT_FOREST_LOD") || std::getenv("NT_FOREST_LOD")[0] != '0';

        // Two prototypes per species for the near tier (card/frond canopies), three
        // for the far tier (blob puffs) — enough silhouette variety that a hillside
        // does not read as one stamp repeated. The LOD path takes the CHEAP blob
        // (~30 tris/puff, 110 attractors) for its 300-800 m level: at that range
        // the fjord demo's blob spends tens of thousands of triangles per tree on
        // a crown that covers ten pixels.
        std::array<vegetation::SpeciesVariants, 3> species;
        for (int s = 0; s < 3; ++s) {
            const auto sp = static_cast<vegetation::TreeSpecies>(s);
            const auto base = static_cast<unsigned int>(100 + s * 37);
            species[s].near = {vegetation::makeForestTreeVariant(sp, base + 1u, false),
                               vegetation::makeForestTreeVariant(sp, base + 2u, false)};
            species[s].far = {vegetation::makeForestTreeVariant(sp, base + 11u, true, lodForest),
                              vegetation::makeForestTreeVariant(sp, base + 12u, true, lodForest),
                              vegetation::makeForestTreeVariant(sp, base + 13u, true, lodForest)};
            // The LOD colour match, in numbers: the near tier's measured mean leaf
            // albedo, and what the far tier rendered before / after the match.
            const auto& n = species[s].near.front().leafMeanLinear;
            const auto& fr = species[s].far.front().leafMeanRaw;
            const auto& fa = species[s].far.front().leafMeanLinear;
            std::cout << "[norway] leaf mean (linear) species " << s << ": card "
                      << n.x << "," << n.y << "," << n.z << "  blob was " << fr.x << ","
                      << fr.y << "," << fr.z << " -> now " << fa.x << "," << fa.y << ","
                      << fa.z << "\n"
                      << std::flush;
        }

        auto forest = Group::create();
        forest->name = "canopy_forest";
        // Bases must come from the PROVIDER (cliff relief + road carving included),
        // not the raw DEM, or every trunk floats or sinks by that delta.
        if (lodForest) {
            // The canopy-surface material. The leaf atlas is a GRAIN map here, not
            // the colour: it is generated near-neutral so the vertex colour (the
            // species tint the blobs use, times a burial AO) survives the multiply.
            // NO alphaTest: the sprig atlas is ~half transparent, and at the 3 m
            // lattice a cutout punches the sheet into a bubble-wrap net of
            // square holes (looked at, 2 km shot). The ragged edge has to come
            // from the GEOMETRY — the valid region ends on the lattice and the
            // boundary carries a skirt — so the map is a pure grain layer.
            auto leafTex = vegetation::makeLeafClusterTexture(256, 77u, {0.90f, 0.93f, 0.86f},
                                                              vegetation::LeafShape::Ovate, 8, 2);
            leafTex->wrapS = TextureWrapping::Repeat;
            leafTex->wrapT = TextureWrapping::Repeat;
            auto canopyMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color::white).roughness(0.9f).metalness(0.f));
            canopyMat->map = leafTex;
            canopyMat->side = Side::Double;// the skirt is a curtain, seen from both
            canopyMat->vertexColors = true;
            canopyMat->translucencyColor = Color(0.50f, 0.80f, 0.28f);
            canopyMat->translucency = 0.3f;

            vegetation::ForestLodOptions lo;
            lo.cap = static_cast<int>(envF("NT_FOREST_CAP", 40000.f));
            lo.cellSize = envF("NT_FOREST_CELL", 128.f);
            lo.l0Distance = envF("NT_FOREST_L0", 300.f);
            lo.l1Distance = envF("NT_FOREST_L1", 800.f);
            // NT_FOREST_L2MESH=1 swaps the thinned blob level for the CHM canopy
            // SURFACE (buildCanopySurface). Kept as an A/B, not the default: it
            // is faster and closes the canopy on gentle ground, but on this
            // pack's benched wall the slope gate leaves tread-wide strips that
            // read as a staircase of lit green trays.
            lo.buildCanopyMesh = envSet("NT_FOREST_L2MESH");
            // 1 = no L2 tier, and that is what the whole tier costs: 113 -> 89 fps
            // here, 21% of the frame every frame for density at 800 m+ where a
            // crown is 2-4 px. The far "pop" the user reported was COLOUR, not
            // density (fixed in makeForestTreeVariant), so the thinning is back on
            // by default; NT_FOREST_L2KEEP=1 buys the density back.
            lo.l2Keep = static_cast<int>(envF("NT_FOREST_L2KEEP", 4.f));
            lo.mesh.seaLevel = reg.seaLevel;
            lo.mesh.maxSlopeDeg = so.maxSlopeDeg;// same gate as the sites, or the
            lo.mesh.minGroundHeight = so.minGroundHeight;// handoff grows new forest
            const auto st = vegetation::buildCanopyForestLod(*forest, sites, species, canopyMat,
                                                             pack.canopy, pack.grid, prov.height, lo);
            scene.add(forest);
            const auto tf1 = std::chrono::high_resolution_clock::now();
            std::cout << "[norway] forest LOD: " << st.sites << " sites, " << st.planted
                      << " instances in " << st.cells << " cells @ " << lo.cellSize << " m"
                      << " (L0 " << st.l0Meshes << " meshes < " << lo.l0Distance << " m, L1 "
                      << st.l1Meshes << " meshes, L2 " << st.l2Meshes
                      << (lo.buildCanopyMesh ? " canopy meshes / " : " thinned meshes 1-in-" )
                      << (lo.buildCanopyMesh ? st.canopyTris : lo.l2Keep)
                      << (lo.buildCanopyMesh ? " tris > " : " > ") << lo.l1Distance
                      << " m), L1 prototype " << st.l1ProtoTris << " tris, "
                      << std::chrono::duration<float>(tf1 - tf0).count() << " s\n"
                      << std::flush;
        } else {
            vegetation::ForestOptions fo;
            fo.cameraPos = camera.position;
            fo.cap = static_cast<int>(envF("NT_FOREST_CAP", 40000.f));
            const auto st = vegetation::buildCanopyForest(*forest, sites, species, prov.height, fo);
            scene.add(forest);
            const auto tf1 = std::chrono::high_resolution_clock::now();
            std::cout << "[norway] forest: " << st.sites << " sites (CHM local maxima), "
                      << st.planted << " instances (" << st.nearTier << " near + " << st.farTier
                      << " far) in " << st.meshes << " meshes, "
                      << std::chrono::duration<float>(tf1 - tf0).count() << " s\n"
                      << std::flush;
        }
    }

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer->setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    // Runtime renderer settings (shared panel). Interactive runs only — the
    // headless capture paths must not draw UI into the measured frames.
    std::unique_ptr<RendererSettingsUi> ui;
    if (!headless) {
        ui = std::make_unique<RendererSettingsUi>(canvas, *renderer, [&] {
            ImGui::TextDisabled("tiles %d  baking %d",
                                static_cast<int>(tiles->activeTiles()),
                                static_cast<int>(tiles->pendingBakes()));
#ifdef THREEPP_WITH_VULKAN
            // Live sea state. Wind writes plain Params fields — the renderer
            // detects the drift and re-bakes the Phillips spectra in place,
            // morphing the sea into the new state over a few swell periods
            // (no pop), so dragging the sliders every frame is fine.
            if (ocean) {
                ImGui::SeparatorText("Sea");
                ImGui::SliderFloat("Wind (m/s)", &ocean->params.windSpeed, 0.f, 24.f, "%.1f");
                float windDeg = ocean->params.windTheta / kDeg2Rad;
                if (ImGui::SliderFloat("Wind heading (deg)", &windDeg, 0.f, 360.f, "%.0f"))
                    ocean->params.windTheta = windDeg * kDeg2Rad;
                ImGui::SliderFloat("Choppiness", &ocean->params.choppiness, 0.f, 1.f, "%.2f");
                ImGui::SliderFloat("Wave scale", &ocean->params.waveScale, 0.f, 3.f, "%.2f");
            }
#endif
        }, "Norway terrain");
    }

    Clock clock;
    int frame = 0;
    float fpsAccum = 0.f, fps = 0.f;
    int fpsFrames = 0;
    [[maybe_unused]] bool sunAligned = false;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        fpsAccum += dt;
        if (++fpsFrames, fpsAccum >= 0.5f) {
            fps = static_cast<float>(fpsFrames) / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        // Bug B symptom 1: the sea reflects the HDRI's OWN baked sun disk (kept in
        // the env's low mips) via env specular. With the analytic sun on a hardcoded
        // heading that disagreed with the HDRI, the water showed a sun reflection
        // unrelated to the scene's lighting. Re-aim the analytic sun at the HDRI's
        // MEASURED sun direction once the renderer has prefiltered the env
        // (envSunFound), so the analytic highlight and the reflected env sun are ONE
        // coherent sun. COASTAL packs only: on mountain packs (no sea) the hardcoded
        // raking heading is a deliberate artistic choice and the HDRI sun would push
        // the valley into shadow. NT_SUN_AZ/EL (if set) keep the manual override.
#ifdef THREEPP_WITH_VULKAN
        if (!sunAligned && coastal && vk && vk->envSunFound() && !viewSun &&
            !envSet("NT_SUN_AZ") && !envSet("NT_SUN_EL")) {
            const Vector3 d = vk->envSunDirection();// unit vector TOWARD the sun
            if (d.length() > 0.5f) {
                sun->position.copy(d);
                sun->position.multiplyScalar(std::max(reg.worldSize, 2000.f));
                std::cout << "[norway] sun aligned to env: dir (" << d.x << ", " << d.y
                          << ", " << d.z << ")\n" << std::flush;
            }
            sunAligned = true;
        }
#endif

        controls.update();

        // Near-detail ribbon chunks: visible within ribbonDist of the camera
        // (10% hysteresis so a boundary chunk doesn't flip every frame). Beyond
        // it the baked+painted roadbed alone carries the road.
        for (size_t ci = 0; ci < roadChunkCenters.size(); ++ci) {
            auto* obj = roadChunkCenters[ci].first;
            const float d = camera.position.distanceTo(roadChunkCenters[ci].second) -
                            roadChunkRadii[ci];
            if (obj->visible) {
                if (d > ribbonDist * 1.1f) obj->visible = false;
            } else if (d < ribbonDist) {
                obj->visible = true;
            }
        }

        // NT_FREEZE_TILES: stop LOD updates once the sequence starts, so a
        // frame-diff measures PURE upscaler/shading shake with geometry frozen
        // (the road-corridor refineBias otherwise keeps baking/swapping tiles).
        const bool freezeTiles = envSet("NT_FREEZE_TILES") && !seqPrefix.empty() && frame >= seqStart;
        // NT_ZOOM_CYCLE="x,z": headless repro of the interactive "zoom in on a
        // flickering area, zoom back out — flicker gone" observation. Frames
        // 150-350 feed tiles->update() a position AT the target (tiles refine
        // there exactly as if the camera flew in), then it returns to the real
        // camera (tiles merge back per the split/merge hysteresis). Camera and
        // rendering never move — only the LOD driver — so a --shotseq at 600
        // isolates what the tile reload changed.
        Vector3 lodPos = camera.position;
        if (const char* zc = std::getenv("NT_ZOOM_CYCLE"); zc && frame >= 150 && frame < 350) {
            float zx = 0.f, zz = 0.f;
            if (std::sscanf(zc, "%f,%f", &zx, &zz) == 2)
                lodPos.set(zx, tiles->heightAt(zx, zz) + 30.f, zz);
        }
        // NT_ORBIT=<deg/frame>: slow orbital drift around the --cam target — the
        // moving-camera repro (tile swaps under motion) for shake measurement.
        if (haveCam && envSet("NT_ORBIT")) {
            const float w = envF("NT_ORBIT", 0.05f) * kDeg2Rad * static_cast<float>(frame);
            const Vector3 off = camPosArg - camTargetArg;
            camera.position.set(camTargetArg.x + off.x * std::cos(w) - off.z * std::sin(w),
                                camPosArg.y,
                                camTargetArg.z + off.x * std::sin(w) + off.z * std::cos(w));
            camera.lookAt(camTargetArg);
        }
        if (!freezeTiles) tiles->update(lodPos);
        if (scatter && !freezeTiles) scatter->update(lodPos);
        renderer->render(scene, camera);
        if (ui) ui->render();

        // Frame-sequence dump (Bug A shake / Bug B flicker metrics). Numbered
        // PNGs from a STATIC camera; tile count logged per dumped frame so
        // split/merge oscillation is visible in the log, not just the pixels.
        if (!seqPrefix.empty()) {
            ++frame;
            if (frame >= seqStart && ((frame - seqStart) % seqStep) == 0) {
                const int n = (frame - seqStart) / seqStep;
                char name[64];
                std::snprintf(name, sizeof(name), "%s_%03d.png", seqPrefix.c_str(), n);
                const auto path = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / "ntseq" / name;
                std::filesystem::create_directories(path.parent_path());
                renderer->writeFramebuffer(path);
                std::cout << "seq " << n << " frame " << frame << " tiles "
                          << tiles->activeTiles() << " baking " << tiles->pendingBakes();
#ifdef THREEPP_WITH_VULKAN
                if (vk) std::cout << " gbufResolveMs " << vk->lastFrameTimings().gbufResolveMs
                                  << " shadeBMs " << vk->lastFrameTimings().shadeBMs;
#endif
                std::cout << std::endl;
                if (n + 1 >= shotFrames) std::exit(0);
            }
            return;
        }

        if (!shotPath.empty() && ++frame >= shotFrames) {
            std::ostringstream stats;
            stats << " (" << fps << " fps, tiles " << tiles->activeTiles()
                  << ", baking " << tiles->pendingBakes() << ")";
            capture::finishShot(*renderer, shotPath, stats.str());
        }
    });

    return 0;
}
