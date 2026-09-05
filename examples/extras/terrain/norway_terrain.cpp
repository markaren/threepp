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
#include "threepp/extras/terrain/CellStreamer.hpp"
#include "threepp/extras/terrain/CliffShell.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/GeoBuildings.hpp"
#include "threepp/extras/terrain/GeoScene.hpp"
#include "threepp/extras/terrain/GeoTerrain.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/extras/terrain/TerrainScatter.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/extras/terrain/UrbanProps.hpp"
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
#include <map>
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
    bool listRoads = false; // --list-roads: id/category/width/length per road, then exit
    std::string viewName;   // --view <name>: a named camera/sun preset (see below)
    bool haveFov = false;
    std::string profilePath;// --road-profile <csv>: dump the height field along the longest road and exit
    // --fly x,y,z[,tx,ty,tz]: with --shotseq, the camera walks from the --cam
    // pose to this one over NT_FLY_FRAMES (600) frames starting at --seqstart.
    // Streamed content (trees, bushes, cars, tiles) is only ever exercised by a
    // camera that MOVES; a static shot photographs the first ring and nothing
    // else. Without a target the look-at stays at the --cam target.
    bool haveFly = false, haveFlyTarget = false;
    Vector3 flyPosArg, flyTargetArg;
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
        else if (a == "--list-roads") listRoads = true;
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
        } else if (a == "--fly" && i + 1 < argc) {
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
            if (k == 3 || k == 6) {
                haveFly = true;
                flyPosArg.set(v[0], v[1], v[2]);
                if (k == 6) {
                    haveFlyTarget = true;
                    flyTargetArg.set(v[3], v[4], v[5]);
                }
            } else {
                std::cerr << "[norway] --fly needs 3 or 6 comma-separated floats: x,y,z[,tx,ty,tz]\n";
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
        } else if (viewName.rfind("road", 0) != 0 && viewName != "aksla" &&
                   viewName != "aksla-near" && viewName != "suburb") {
            // road* / aksla* / suburb need the loaded pack (ground height, the
            // conformed network), so they resolve further down after carveRoads.
            std::cerr << "[norway] unknown --view '" << viewName
                      << "' (known: reference, aksla, aksla-near, suburb, "
                         "road[=id][@station], road-top, road-aerial, road-far)\n";
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
    {
        // What the pack actually carries for the close-up realism work: the CHM
        // (no CHM, no trees at all), the measured roof blocks (no blocks, the
        // rect-fit heuristic), and the OSM land use (no land use, one asphalt
        // fBm under the whole town). Printed on every run so a shot's log says
        // which of the three it was made with.
        std::map<std::string, int> roofKinds;
        for (const auto& b : pack.buildings)
            if (b.hasRoof()) ++roofKinds[b.roof.kind];
        std::cout << "         canopy: " << (pack.hasCanopy() ? "present" : "ABSENT")
                  << "; roof blocks: ";
        if (roofKinds.empty()) std::cout << "none";
        for (const auto& [k, n] : roofKinds) std::cout << k << ' ' << n << ' ';
        std::cout << "\n         landuse: " << pack.landuse.polygons.size() << " polygons, "
                  << pack.landuse.lines.size() << " lines, " << pack.landuse.points.size()
                  << " points\n"
                  << std::flush;
    }

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
    // ── N302 marking classes ───────────────────────────────────────────────────
    // A Norwegian road is marked by its ASPHALT WIDTH, not by its number: yellow
    // centre line only at 6 m and wider, white edge lines always, dashed 3+3 on
    // the narrow roads that carry no centre line at all. NT_ROAD_CENTRE_MIN is
    // the A/B for a road the pack's CATEGORY DEFAULT width mis-classifies (Fv63
    // gets 7 m from the fetcher, but its hairpin section is really narrower and
    // unmarked in the middle) until NVDB's measured widths are fetched.
    {
        road::MarkingRules mr;
        if (const char* e = std::getenv("NT_ROAD_CENTRE_MIN"); e && e[0] != '\0')
            mr.centreMin = std::strtof(e, nullptr);
        if (const char* e = std::getenv("NT_ROAD_WEAR"); e && e[0] != '\0')
            mr.wear = std::clamp(std::strtof(e, nullptr), 0.f, 1.f);
        if (const char* e = std::getenv("NT_ROAD_SEED"); e && e[0] != '\0')
            mr.seed = static_cast<unsigned int>(std::strtoul(e, nullptr, 10));
        if (const char* e = std::getenv("NT_ROAD_GRAVEL_EDGE"); e && e[0] == '0')
            mr.gravelEdge = false;
        network.setMarkingRules(mr);
    }
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

    // ── --list-roads ───────────────────────────────────────────────────────────
    // Which road is which, so a shot can name one (the K and P roads that show
    // the narrow/unmarked classes are anonymous ids in the pack).
    if (listRoads) {
        auto infos = network.roadInfos();
        std::sort(infos.begin(), infos.end(),
                  [](const auto& a, const auto& b) { return a.length > b.length; });
        std::cout << "[norway] roads (" << infos.size() << "), longest first:\n";
        for (const auto& i : infos) {
            static const char* kClass[] = {"main", "narrow", "unmarked", "track"};
            std::printf("  %-24s %-2s %5.1f m wide %8.0f m long  %-8s  at (%.0f, %.0f)\n",
                        i.id.c_str(), i.category.c_str(), i.width, i.length,
                        kClass[static_cast<int>(i.cls)], i.first.x, i.first.z);
        }
        std::cout << std::flush;
        return 0;
    }

    // ── named ROAD views ───────────────────────────────────────────────────────
    // The road look is judged from a road, not from 150 m up. Each preset anchors
    // on a station along a drivable run and PRINTS the --cam line it resolved to,
    // so the identical frame can be re-shot from any other build (which is how
    // the before/after sheets are made).
    if (viewName.rfind("road", 0) == 0) {
        std::string kind = viewName, roadId;
        float stationArg = -1.f;
        if (const size_t at = kind.find('@'); at != std::string::npos) {
            stationArg = std::strtof(kind.c_str() + at + 1, nullptr);
            kind = kind.substr(0, at);
        }
        if (const size_t eq = kind.find('='); eq != std::string::npos) {
            roadId = kind.substr(eq + 1);
            kind = kind.substr(0, eq);
        }
        std::vector<Vector3> cl = roadId.empty() ? network.longestDrivableRun()
                                                 : network.roadCenterline(roadId);
        if (cl.size() < 2) {
            std::cerr << "[norway] --view " << viewName << ": no such road (try --list-roads)\n";
        } else {
            std::vector<float> cum(cl.size(), 0.f);
            for (size_t k = 1; k < cl.size(); ++k) {
                const float dx = cl[k].x - cl[k - 1].x, dz = cl[k].z - cl[k - 1].z;
                cum[k] = cum[k - 1] + std::sqrt(dx * dx + dz * dz);
            }
            const float total = cum.back();
            const auto at = [&](float s) {
                s = std::clamp(s, 0.f, total);
                size_t k = 1;
                while (k + 1 < cum.size() && cum[k] < s) ++k;
                const float span = std::max(cum[k] - cum[k - 1], 1e-4f);
                const float t = std::clamp((s - cum[k - 1]) / span, 0.f, 1.f);
                return cl[k - 1].clone().lerp(cl[k], t);
            };
            const float station = (stationArg >= 0.f) ? stationArg : 0.4f * total;
            const Vector3 anchor = at(station);
            const Vector3 ahead = at(station + 60.f);
            Vector3 tan(ahead.x - anchor.x, 0.f, ahead.z - anchor.z);
            if (tan.length() < 1e-3f) tan.set(0.f, 0.f, 1.f);
            tan.normalize();
            const float eye = road::RoadNetwork::kSurfaceRaise + 1.6f;// driver's eye
            if (kind == "road-top") {
                // Near-vertical: this is the view the surface look is JUDGED in,
                // because a repeated object or an evenly spaced crack rhythm is
                // obvious from straight above and hides in perspective. Tilted
                // ~8 degrees off vertical ALONG the road so the up vector never
                // degenerates (a true nadir camera with up = +Y is a lookAt
                // singularity and the frame rolls arbitrarily).
                constexpr float topH = 28.f;
                const float back = topH * std::tan(8.f * math::DEG2RAD);
                camPosArg.set(anchor.x - tan.x * back, anchor.y + topH, anchor.z - tan.z * back);
                camTargetArg.copy(anchor);
            } else if (kind == "road-aerial") {
                camPosArg.set(anchor.x - tan.x * 40.f, anchor.y + 25.f, anchor.z - tan.z * 40.f);
                camTargetArg.copy(anchor);
            } else if (kind == "road-far") {
                // 900 m back: the 600 m ribbon cull is in the middle of the frame,
                // so a colour step between ribbon and painted bed cannot hide.
                camPosArg.set(anchor.x - tan.x * 900.f, anchor.y + 30.f, anchor.z - tan.z * 900.f);
                camTargetArg.copy(anchor);
            } else {
                camPosArg.set(anchor.x, anchor.y + eye, anchor.z);
                camTargetArg.set(ahead.x, ahead.y + eye, ahead.z);
            }
            haveCam = true;
            if (!haveFov) fovArg = 55.f;
            std::printf("[norway] --view %s -> road '%s' station %.0f/%.0f m; "
                        "--cam %.2f,%.2f,%.2f,%.2f,%.2f,%.2f --fov %.1f\n",
                        viewName.c_str(),
                        roadId.empty() ? "longest-drivable" : roadId.c_str(), station, total,
                        camPosArg.x, camPosArg.y, camPosArg.z,
                        camTargetArg.x, camTargetArg.y, camTargetArg.z, fovArg);
            std::cout << std::flush;
        }
    }

    // ── named ÅLESUND views ────────────────────────────────────────────────────
    // The close-up realism programme is judged from the SAME three framings every
    // phase, and two of them are the photograph everyone compares the render to:
    // the Aksla plateau edge, looking down over Brosundet and the
    // Jugend centre. `aksla` is the postcard, `aksla-near` the same eye
    // on a 28° lens for 1:1 crops of the centre blocks at ~600 m, and `suburb` is
    // eye level on a residential street — the only one of the three that shows a
    // wall and a window at reading distance. None of them touches the sun: the
    // default (215°/34°) is the one every previous aalesund shot used, so a
    // before/after pair differs by geometry alone.
    if (viewName == "aksla" || viewName == "aksla-near" || viewName == "suburb") {
        if (viewName != "suburb") {
            // The eye is on the WEST EDGE of the Aksla plateau, not on the
            // summit node (994, -214, 166 m): from the summit the town is a
            // distant strip behind the bare shoulder of the hill, which is not
            // the photograph. From the edge the town fills the lower two
            // thirds, the centre blocks sit at ~600 m and the islands run along
            // the top, as in the reference.
            //
            // The eye y is now an ABSOLUTE metre height, not ground + 2. Phase
            // B planted the Aksla forest, and the old eye (801, ground + 2,
            // -160) stands INSIDE two 20 m CHM trees: they fill the middle
            // third of the frame and there is no view at all. Moving 15 m west
            // and standing 20 m over the slope clears the crowns and keeps the
            // same composition — checked against the three candidates with the
            // forest on. A fixed y also means the frame does not move when the
            // DTM sampling changes.
            //   --cam 786,141,-150,300,5,-40 --fov 55   (aksla)
            //   --cam 786,141,-150,300,5,-40 --fov 28   (aksla-near)
            constexpr float ex = 786.f, ey = 141.f, ez = -150.f;
            const float ground = pack.grid.sampleBicubic(ex, ez);
            if (!haveCam) {
                haveCam = true;
                camPosArg.set(ex, ey, ez);
                camTargetArg.set(300.f, 5.f, -40.f);// the Jugend centre blocks
            }
            if (!haveFov) fovArg = (viewName == "aksla") ? 55.f : 28.f;
            std::printf("[norway] --view %s -> eye %.0f m absolute (ground %.0f, %.0f m over it); "
                        "--cam %.2f,%.2f,%.2f,%.2f,%.2f,%.2f --fov %.1f\n",
                        viewName.c_str(), camPosArg.y, ground, camPosArg.y - ground,
                        camPosArg.x, camPosArg.y, camPosArg.z,
                        camTargetArg.x, camTargetArg.y, camTargetArg.z, fovArg);
        } else {
            // A wooden-house street, RESOLVED rather than hardcoded: of the K
            // roads at least 150 m long, take the one with the most house-like
            // footprints (house / detached / semidetached / terrace) within 60 m
            // of its centreline, then the station along it where that count
            // peaks. Deterministic for a given pack, and it prints the id +
            // station so a later phase can re-shoot the identical frame with
            // --view road=<id>@<station>. On the 2026-09-05 aalesund pack this
            // resolves to K road 220317 at station 160 of 1105 m, with 18
            // house-like footprints inside 60 m.
            std::vector<Vector2> houses;
            houses.reserve(pack.buildings.size());
            for (const auto& b : pack.buildings) {
                if (b.type != "house" && b.type != "detached" &&
                    b.type != "semidetached_house" && b.type != "terrace")
                    continue;
                Vector2 c;
                for (const auto& p : b.outer) c.add(p);
                houses.push_back(c.divideScalar(static_cast<float>(b.outer.size())));
            }
            const auto countNear = [&houses](const Vector3& p, float r) {
                int n = 0;
                for (const auto& h : houses) {
                    const float dx = h.x - p.x, dz = h.y - p.z;
                    if (dx * dx + dz * dz <= r * r) ++n;
                }
                return n;
            };
            std::string bestId;
            int bestScore = -1;
            for (const auto& info : network.roadInfos()) {
                if (info.category != "K" || info.length < 150.f) continue;
                const auto cl = network.roadCenterline(info.id);
                if (cl.size() < 2) continue;
                int score = 0;
                for (size_t k = 0; k < cl.size(); k += 2) score += countNear(cl[k], 60.f);
                if (score > bestScore) {
                    bestScore = score;
                    bestId = info.id;
                }
            }
            const auto cl = bestId.empty() ? std::vector<Vector3>{}
                                           : network.roadCenterline(bestId);
            if (cl.size() < 2) {
                std::cerr << "[norway] --view suburb: no K road with houses in this pack\n";
            } else {
                std::vector<float> cum(cl.size(), 0.f);
                for (size_t k = 1; k < cl.size(); ++k)
                    cum[k] = cum[k - 1] + std::hypot(cl[k].x - cl[k - 1].x, cl[k].z - cl[k - 1].z);
                const auto at = [&](float s) {
                    s = std::clamp(s, 0.f, cum.back());
                    size_t k = 1;
                    while (k + 1 < cum.size() && cum[k] < s) ++k;
                    const float span = std::max(cum[k] - cum[k - 1], 1e-4f);
                    return cl[k - 1].clone().lerp(cl[k], std::clamp((s - cum[k - 1]) / span, 0.f, 1.f));
                };
                float bestS = 0.f;
                int bestN = -1;
                for (float s = 0.f; s <= cum.back(); s += 20.f) {
                    const int n = countNear(at(s), 60.f);
                    if (n > bestN) {
                        bestN = n;
                        bestS = s;
                    }
                }
                const Vector3 anchor = at(bestS);
                const Vector3 ahead = at(bestS + 60.f);
                const float eye = road::RoadNetwork::kSurfaceRaise + 1.7f;
                if (!haveCam) {
                    haveCam = true;
                    camPosArg.set(anchor.x, anchor.y + eye, anchor.z);
                    camTargetArg.set(ahead.x, ahead.y + eye, ahead.z);
                }
                if (!haveFov) fovArg = 55.f;
                std::printf("[norway] --view suburb -> K road '%s' station %.0f/%.0f m, "
                            "%d houses within 60 m; --cam %.2f,%.2f,%.2f,%.2f,%.2f,%.2f --fov %.1f\n",
                            bestId.c_str(), bestS, cum.back(), bestN,
                            camPosArg.x, camPosArg.y, camPosArg.z,
                            camTargetArg.x, camTargetArg.y, camTargetArg.z, fovArg);
            }
        }
        std::cout << std::flush;
    }

    // ── QUAY APRON ─────────────────────────────────────────────────────────
    // Ålesund's harbour front is reclaimed land, and the lidar reads reclaimed
    // quay as WATER: the DTM puts those cells at exactly seaLevel, and the sea
    // sink below then drops them 6 m — so the warehouses along Skansekaia stand
    // with their walls in the fjord (phaseC2_aksla_near.png). Nothing downstream
    // can fix that; the ground under a building has to exist.
    //
    // So: every sea-level cell that is under a building (footprints grown by
    // 8 m — the apron is the yard around the shed, not just its outline) or
    // inside a surveyed pier/quay polygon becomes an apron 0.9 m above the
    // water, and every breakwater cell a 1.5 m rock ridge. The heightfield gives
    // the apron a short ramp at its edge instead of a vertical quay wall; at the
    // ranges this pack is judged from that is the right trade, and it means no
    // extra geometry and no seam with the tiles.
    //
    // BEFORE the sink, and before makeGeoProvider: the sink must not see these
    // cells as water any more, and the paint needs the mask. NT_NO_APRON=1 for
    // the A/B.
    std::shared_ptr<const terrain::FootprintMask> apronMask;
    int apronCells = 0;
    if (reg.heightMin < 1.0f && !std::getenv("NT_NO_APRON") &&
        (!pack.buildings.empty() || pack.hasLandUse())) {
        const auto builtNear = terrain::buildFootprintMask(pack, 8.f, 2.f);
        const auto quayPoly = terrain::buildLandUseMask(pack, {"pier", "quay"}, 0.f, 2.f);
        const auto breakPoly = terrain::buildLandUseMask(pack, {"breakwater"}, 0.f, 2.f);
        if (builtNear || quayPoly || breakPoly) {
            const int gdim = pack.grid.dim();
            const float gstep = pack.grid.worldSize() / static_cast<float>(gdim - 1);
            const float ghalf = pack.grid.worldSize() * 0.5f;
            auto apron = terrain::detail::geoMakeMask(reg.worldSize, gstep);
            auto& hh = pack.grid.data();
            for (int iz = 0; iz < gdim && iz < apron->dim; ++iz) {
                const float z = -ghalf + static_cast<float>(iz) * gstep;
                for (int ix = 0; ix < gdim && ix < apron->dim; ++ix) {
                    float& hv = hh[static_cast<size_t>(iz) * gdim + ix];
                    if (hv > reg.seaLevel + 0.05f) continue;
                    const float x = -ghalf + static_cast<float>(ix) * gstep;
                    const bool rock = breakPoly && breakPoly->inside(x, z);
                    if (!rock && !(builtNear && builtNear->inside(x, z)) &&
                        !(quayPoly && quayPoly->inside(x, z)))
                        continue;
                    hv = reg.seaLevel + (rock ? 1.5f : 0.9f);
                    apron->m[static_cast<size_t>(iz) * apron->dim + ix] = 1u;
                    ++apronCells;
                }
            }
            apronMask = apron;
            std::cout << "[norway] quay apron: " << apronCells << " sea-level cells raised to +"
                      << 0.9f << " m (breakwater +1.5)\n"
                      << std::flush;
        }
    }

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
    // NT_SEA_BATHY=1: the GeoScene facade's distance-to-shore profile instead
    // of the flat sink (NT_SEA_SHORE_SLOPE m/m, NT_SEA_MAXDEPTH m). This is what
    // the netpen --terrain scene stands in, so shoreline water-colour work is
    // reproduced here, in the C++ demo, with the same seabed.
    if (const char* e = std::getenv("NT_SEA_BATHY"); e && *e == '1' && reg.heightMin < 1.0f) {
        float slope = 0.35f, maxDepth = 180.f;
        if (const char* v = std::getenv("NT_SEA_SHORE_SLOPE"); v && *v) slope = std::strtof(v, nullptr);
        if (const char* v = std::getenv("NT_SEA_MAXDEPTH"); v && *v) maxDepth = std::strtof(v, nullptr);
        terrain::GeoScene::makeBathymetry(pack.grid, reg.seaLevel, slope, maxDepth);
        seaDepth = 0.f;// the profile replaces the flat sink
    }
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
    // The far road is a flat tint mixed into the splat albedo. Take it from the
    // SAME bake the near ribbon uses instead of a literal: the old 0.075 was a
    // near-black that matched the old near-black ribbon, and any change to one
    // without the other makes the 600 m ribbon-cull hand-off step.
    gopt.roadColor = network.meanSurfaceColor(road::SurfaceKind::Asphalt);
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
    // Surveyed land cover (OSM bare_rock / scree / scrub / heath) painted over
    // the splat. NT_NO_LANDUSE_PAINT=1 is the A/B for that layer ALONE — the
    // CHM forest paint, the urban fabric and the road paint are untouched, so
    // the pair differs by the polygons and nothing else.
    gopt.landUsePaint = !std::getenv("NT_NO_LANDUSE_PAINT");
    gopt.apronMask = apronMask;
    const terrain::TerrainProvider prov = terrain::makeGeoProvider(pack, network, gopt);

    // ── land-use paint diagnostic ─────────────────────────────────────────
    // The parking paint went missing in the Round 5 shots (lots rendered as
    // grey-green ground with tufts while the cars stood on them), and the only
    // way to tell "the lot was never rasterised" from "the lot is painted and
    // something above it wins" is to look at the raster and at the layers.
    //   NT_LANDUSE_DUMP=<file.pgm>  class-code raster (P5, one code per byte)
    //                               + a per-class cell histogram, then exit.
    //   NT_LANDUSE_PROBE="x,z;x,z"  per point: the class code, the sampled Mix,
    //                               and albedo/weights with the layer ON and
    //                               OFF, so the layer's own contribution shows.
    {
        const char* dumpPath = std::getenv("NT_LANDUSE_DUMP");
        const char* probeArg = std::getenv("NT_LANDUSE_PROBE");
        if ((dumpPath && dumpPath[0]) || (probeArg && probeArg[0])) {
            const auto lup = terrain::buildLandUsePaint(pack, gopt.landUseCell, apronMask);
            if (!lup || !lup->valid()) {
                std::cout << "[landuse] NO PAINT (hasLandUse " << pack.hasLandUse()
                          << ", apron " << (apronMask ? 1 : 0) << ")\n";
            } else {
                std::array<long long, terrain::LandUsePaint::ClsCount> hist{};
                for (std::uint8_t v : lup->m)
                    if (v < terrain::LandUsePaint::ClsCount) ++hist[v];
                static const char* kNames[] = {"None", "Asphalt", "Bay", "Gravel",
                                               "Grass", "Pitch", "Concrete", "Rock"};
                std::cout << "[landuse] raster dim " << lup->dim << " cell " << lup->cell
                          << " half " << lup->half << "\n";
                for (int i = 0; i < terrain::LandUsePaint::ClsCount; ++i)
                    std::cout << "[landuse]   " << kNames[i] << " " << hist[i] << " cells\n";
                if (dumpPath && dumpPath[0]) {
                    std::ofstream f(dumpPath, std::ios::binary);
                    f << "P5\n" << lup->dim << " " << lup->dim << "\n255\n";
                    f.write(reinterpret_cast<const char*>(lup->m.data()),
                            static_cast<std::streamsize>(lup->m.size()));
                    std::cout << "[landuse] wrote " << dumpPath << "\n";
                }
            }
            if (probeArg && probeArg[0]) {
                terrain::GeoTerrainOptions gOff = gopt;
                gOff.landUsePaint = false;
                const terrain::TerrainProvider provOff =
                        terrain::makeGeoProvider(pack, network, gOff);
                std::string s = probeArg;
                size_t pos = 0;
                while (pos <= s.size()) {
                    const size_t semi = s.find(';', pos);
                    const std::string tok =
                            s.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
                    float px = 0.f, pz = 0.f;
                    if (std::sscanf(tok.c_str(), "%f,%f", &px, &pz) == 2) {
                        const float h = prov.height(px, pz);
                        const float dx = prov.height(px + 1.f, pz) - prov.height(px - 1.f, pz);
                        const float dz = prov.height(px, pz + 1.f) - prov.height(px, pz - 1.f);
                        const float slope = std::sqrt(dx * dx + dz * dz) / 2.f;
                        std::printf("[probe] (%.1f, %.1f) h %.2f slope %.3f\n", px, pz, h, slope);
                        if (lup && lup->valid()) {
                            const int ix = static_cast<int>(std::lround((px + lup->half) / lup->cell));
                            const int iz = static_cast<int>(std::lround((pz + lup->half) / lup->cell));
                            const auto mix = lup->sample(px, pz);
                            std::printf("[probe]   cell (%d, %d) code %d  mix", ix, iz,
                                        static_cast<int>(lup->code(ix, iz)));
                            for (int i = 0; i < terrain::LandUsePaint::ClsCount; ++i)
                                std::printf(" %.2f", mix[i]);
                            std::printf("\n");
                        }
                        float on[3] = {0, 0, 0}, off[3] = {0, 0, 0};
                        float wOn[4] = {0, 0, 0, 0}, wOff[4] = {0, 0, 0, 0};
                        prov.albedo(px, pz, h, slope, on);
                        provOff.albedo(px, pz, h, slope, off);
                        if (prov.weights) prov.weights(px, pz, h, slope, wOn);
                        if (provOff.weights) provOff.weights(px, pz, h, slope, wOff);
                        std::printf("[probe]   albedo OFF %.3f %.3f %.3f -> ON %.3f %.3f %.3f\n",
                                    off[0], off[1], off[2], on[0], on[1], on[2]);
                        std::printf("[probe]   weights OFF %.2f %.2f %.2f %.2f (sum %.2f)"
                                    " -> ON %.2f %.2f %.2f %.2f (sum %.2f)\n",
                                    wOff[0], wOff[1], wOff[2], wOff[3],
                                    wOff[0] + wOff[1] + wOff[2] + wOff[3], wOn[0], wOn[1], wOn[2],
                                    wOn[3], wOn[0] + wOn[1] + wOn[2] + wOn[3]);
                    }
                    if (semi == std::string::npos) break;
                    pos = semi + 1;
                }
            }
            std::cout << std::flush;
            if (dumpPath && dumpPath[0]) return 0;
        }
    }

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
    // The SURVEYED HARD SURFACING gets the same bias as a road corridor, and for
    // the same reason. Its content is fine paint — 2.5 m bay stripes, a lot's
    // edge, a quay apron — and none of it survives a coarse tile's texel. Worse,
    // `errorLod` shrinks the split radius of a tile whose MESH error is small,
    // and urban ground is graded flat by construction: exactly the ground that
    // carries the most paint detail was the last to refine. A 110 m camera over
    // the big lot sat on a ~2.6 m texel for hundreds of frames and the lot
    // rendered as one flat grey (phaseF_lot_nadir_before.png) — the paint was
    // right in the raster the whole time.
    //
    // OPT-IN (NT_LU_BIAS=1), because it is not free and the trade is the user's
    // to make: measured at --view aksla, 400-frame shots, 25.0 / 31.7 fps with
    // the bias against 31.0 / 39.4 without (same run order, tile settling still
    // in flight in both). It buys sharp lots and quays near the camera at about
    // a fifth of the frame rate in the flagship wide view.
    std::shared_ptr<const terrain::FootprintMask> hardMask;
    if (const char* lb = std::getenv("NT_LU_BIAS"); pack.hasLandUse() && lb && lb[0] == '1')
        hardMask = terrain::buildLandUseMask(
                pack, {"parking", "pier", "quay", "yard", "marina", "playground"}, 0.f, 4.f);
    if (!noRoadBias || hardMask) {
        tileOpts.refineBias = [&network, hardMask, noRoadBias](float cx, float cz, float half) {
            if (!noRoadBias && network.corridorIntersects(cx, cz, half)) return 2.2f;
            if (hardMask && half <= 400.f) {
                // Nine taps over the tile box: a lot is tens of metres across,
                // so a centre-only test misses the tile it sits in the corner of.
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i)
                        if (hardMask->inside(cx + static_cast<float>(i) * half * 0.8f,
                                             cz + static_cast<float>(j) * half * 0.8f))
                            return 2.2f;
            }
            return 1.0f;
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
    // Shot/sequence captures run on a HEADLESS canvas: hidden window, and the
    // Vulkan renderer prefers VK_EXT_headless_surface, so a batch of captures
    // never pops windows over whatever the user is doing (asked for 2026-09-05).
    Canvas canvas("threepp - NORWAY TERRAIN", {{"vsync", false}, {"size", canvasSize}, {"headless", headless}});
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
        const auto rc = network.meanSurfaceColor(road::SurfaceKind::Asphalt);
        const auto& mrules = network.markingRules();
        const double perTex = static_cast<double>(mrules.texWidth) * mrules.texHeight * 4.0;
        const size_t nSurfTex = network.surfaceSetCount() * 3;// albedo + normal + roughMetal
        const double patchMB = 2.0 * static_cast<double>(mrules.patchRes) * mrules.patchRes * 4.0 * 4.0;
        std::printf("[norway] road ribbon chunks: %zu sharing %zu baked surface sets "
                    "= %zu textures %.1f MB + patch atlas 2 x %d x %d (%.1f MB); "
                    "tile %.0f m (mean paved sRGB %.3f, %.3f, %.3f)\n",
                    roadChunkCenters.size(), network.surfaceSetCount(), nSurfTex,
                    static_cast<double>(nSurfTex) * perTex / 1048576.0,
                    mrules.patchRes * 4, mrules.patchRes, patchMB / 1048576.0,
                    static_cast<double>(mrules.tileLength), rc[0], rc[1], rc[2]);
        std::cout << std::flush;
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
        bo.measuredRoofs = !envSet("NT_NO_MEASURED_ROOFS");
        terrain::GeoBuildingsStats bs;
        bo.stats = &bs;
        auto buildings = terrain::buildGeoBuildingMeshes(pack, bo);
        std::printf("[norway] buildings: %d footprints -> flat %d, gabled %d, "
                    "hipped-skeleton %d (skeleton tried %d, fell back %d = %.1f%%), "
                    "no roof block %d, towers %d\n",
                    bs.buildings, bs.flat, bs.gabled, bs.hipped, bs.skeletonTried,
                    bs.skeletonFailed,
                    bs.skeletonTried ? 100.f * static_cast<float>(bs.skeletonFailed) /
                                               static_cast<float>(bs.skeletonTried)
                                     : 0.f,
                    bs.heuristic, bs.towers);
        {
            std::string causes;
            for (int r = 1; r < terrain::SK_COUNT; ++r)
                if (bs.skelFail[static_cast<size_t>(r)])
                    causes += std::string(causes.empty() ? "" : ", ") +
                              terrain::skeletonFailName(r) + " " +
                              std::to_string(bs.skelFail[static_cast<size_t>(r)]);
            if (bs.skelFailRing) causes += (causes.empty() ? "" : ", ") +
                                           std::string("ring-degenerate ") +
                                           std::to_string(bs.skelFailRing);
            if (bs.skelFailPitch) causes += (causes.empty() ? "" : ", ") +
                                            std::string("pitch-reject ") +
                                            std::to_string(bs.skelFailPitch);
            std::printf("[norway] buildings: skeleton fallback causes: %s\n",
                        causes.empty() ? "none" : causes.c_str());
        }
        std::printf("[norway] buildings: %zu meshes, %zu triangles, %zu materials, "
                    "%d corner boards\n",
                    bs.meshes, bs.triangles, bs.materials, bs.cornerBoards);
        std::fflush(stdout);
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
    //
    // TOWN GATES: a CHM is DOM − DTM, so on an urban pack every BUILDING is a
    // 10 m "canopy peak" and every ship, crane and pier crate is a taller one.
    // The gates below are what let the forest run over a town at all — without
    // them the detector plants a spruce on every roof in Ålesund, which is why
    // phase A shipped with the forest switched off on packs with footprints.
    // Measured on this pack: 30.2% of land cells carry canopy ≥ 2.5 m, and
    // 29.3% of those sit inside a footprint DILATED BY 4 M — the dilation is
    // not padding, it is the registration offset between the Kartverket DOM
    // and the OSM footprint, and at 2 m dilation a full ring of roof-edge
    // "canopy" survives.
    const bool urbanPack = !pack.buildings.empty() || pack.hasLandUse();
    // Streamed content: built per cell around the LIVE camera, updated in the
    // frame loop next to the tiles. Declared out here so the loop can see them.
    std::vector<std::shared_ptr<terrain::CellStreamer>> streamers;
    if (pack.hasCanopy() && !envSet("NT_NO_FOREST")) {
        const auto tf0 = std::chrono::high_resolution_clock::now();

        // Rasters, built once (2 m, the pack's own resolution).
        const auto fpMask = terrain::buildFootprintMask(pack, envF("NT_FOREST_DILATE", 4.f), 2.f);
        // Decks and lots: a pier is not ground, a marina is water, a parking lot
        // and a football pitch are surfaces nobody plants a tree in.
        const auto deckMask = terrain::buildLandUseMask(
                pack, {"pier", "quay", "breakwater", "marina", "parking", "pitch"}, 1.f, 2.f);
        // Within 40 m of a building = "garden": where the bush tier is allowed.
        const auto gardenMask = urbanPack ? terrain::buildFootprintMask(pack, 40.f, 4.f) : nullptr;

        int gateFootprint = 0, gatePaved = 0, gateDeck = 0;
        auto rejectSite = [&, fpMask, deckMask](float x, float z) {
            if (fpMask && fpMask->inside(x, z)) {
                ++gateFootprint;
                return true;
            }
            if (network.pavedWeight(x, z, 1.0f) > 0.2f) {
                ++gatePaved;
                return true;
            }
            if (deckMask && deckMask->inside(x, z)) {
                ++gateDeck;
                return true;
            }
            return false;
        };

        vegetation::CanopySiteOptions so;
        so.seaLevel = reg.seaLevel;
        // PACK-WIDE. This used to be a square ROI centred on the STARTUP camera
        // target — which is why the user asked whether "trees are a function of
        // where the camera is located at startup". They were. Detection is a
        // one-off scan and the sites are 16 bytes each; what has to be bounded
        // is the GEOMETRY, and that is the streamer's job below.
        so.centerX = 0.f;
        so.centerZ = 0.f;
        so.halfExtent = 1e9f;
        if (urbanPack) {
            // 2 m grid: a 3×3 window is a 4 m crown spacing, which is a town
            // tree. The 5×5 default is a plantation rule and it halves the
            // garden crowns (106 875 peaks vs 57 359 over this pack).
            so.windowRadius = 1;
            so.reject = rejectSite;
            // Cranes, spires, masts and ship superstructure are the tall end of
            // a CHM over a HARBOUR; on a fjord pack a 30 m peak is a spruce, so
            // this cap belongs to the pack, not to the detector.
            so.maxCanopyHeight = envF("NT_FOREST_MAXH", 28.f);
        } else {
            so.maxCanopyHeight = envF("NT_FOREST_MAXH", 1e9f);
        }
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
        auto sites = vegetation::detectTreeSites(pack.canopy, pack.grid, so, &srep);
        std::cout << "[norway] canopy sites: " << srep.peaks << " CHM peaks -> "
                  << sites.size() << " sites (rejected: support " << srep.rejectedSupport
                  << ", treeline " << srep.rejectedTreeline << ", high-slope "
                  << srep.rejectedHighSlope << ", slope " << srep.rejectedSlope
                  << ", ground " << srep.rejectedGround << ", tall " << srep.rejectedTall
                  << ", masked " << srep.rejectedMasked << " [footprint " << gateFootprint
                  << " + paved " << gatePaved << " + deck " << gateDeck << "]); treeline "
                  << srep.treelineElevation << " m +-" << so.treelineFeather
                  << ", highest site " << srep.highestSite << " m\n"
                  << std::flush;

        // ── Trees the surveyor drew ────────────────────────────────────────
        // OSM natural=tree points and tree_row lines are not a guess about the
        // CHM: someone stood there. They bypass the masks (a street tree IS on
        // the pavement, a park row IS beside a path) but not the ground gate,
        // and they take their height from the CHM where it has one.
        int explicitTrees = 0;
        if (pack.hasLandUse() && !envSet("NT_NO_OSM_TREES")) {
            const float roi = so.halfExtent;
            const auto addTree = [&](float x, float z) {
                if (std::fabs(x - so.centerX) > roi || std::fabs(z - so.centerZ) > roi) return;
                if (pack.grid.sampleBilinear(x, z) < so.seaLevel + so.minGroundHeight) return;
                float h = pack.hasCanopy() ? pack.canopy.sampleBilinear(x, z) : 0.f;
                if (h < 2.5f || h > so.maxCanopyHeight) h = 8.f;
                // standHeight 0: an OSM street tree is a single crown, never a
                // stand, so the species rule sends it to broadleaf.
                sites.push_back({x, z, h, 0.f});
                ++explicitTrees;
            };
            for (const auto& p : pack.landuse.points)
                if (p.cls == "tree") addTree(p.pos.x, p.pos.y);
            for (const auto& l : pack.landuse.lines) {
                if (l.cls != "tree_row" || l.points.size() < 2) continue;
                addTree(l.points.front().x, l.points.front().y);
                float acc = 0.f;// metres walked since the last tree
                for (size_t i = 1; i < l.points.size(); ++i) {
                    const float dx = l.points[i].x - l.points[i - 1].x;
                    const float dz = l.points[i].y - l.points[i - 1].y;
                    const float len = std::sqrt(dx * dx + dz * dz);
                    if (len < 1e-3f) continue;
                    float pos = 0.f;
                    while (acc + (len - pos) >= 6.f) {
                        pos += 6.f - acc;
                        acc = 0.f;
                        addTree(l.points[i - 1].x + dx * (pos / len),
                                l.points[i - 1].y + dz * (pos / len));
                    }
                    acc += len - pos;
                }
            }
            std::cout << "[norway] OSM trees: " << explicitTrees
                      << " explicit sites (natural=tree points + tree_row @ 6 m)\n"
                      << std::flush;
        }

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
        // What the vegetation actually costs the SCENE WALK, summed over the
        // forest and the bushes: an InstancedMesh is a draw, a descriptor set
        // and a TLAS instance whatever it holds, and on this pack the object
        // count is the frame, not the triangles (measured: dropping l0Distance
        // from 300 m to 60 m moved nothing, halving the ROI moved everything).
        int vegMeshes = 0, vegInstances = 0, vegCells = 0;
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
            lo.cap = static_cast<int>(envF("NT_FOREST_CAP", 1e9f));
            // A town's tall trees are limes, maples, chestnuts and rowans, not
            // Norway spruce: spruce needs BOTH height above sea and a stand
            // around it. Aksla's plantation (60 m+, closed canopy) still gets
            // conifers; the 16 m tree in a churchyard at 12 m elevation becomes
            // the broadleaf it is. Both terms are 0 on fjord packs = old rule.
            if (urbanPack) {
                lo.spruceMinElevation = envF("NT_SPRUCE_ELEV", 60.f);
                lo.spruceMinStandHeight = envF("NT_SPRUCE_STAND", 12.f);
            }
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
            // FAR CELLS ARE COARSER. The cell is the unit of object count, and
            // the object count is what this frame is spent on: geiranger runs
            // 28 k trees at ~100 fps over ~180 cells, Ålesund ran 26 k over
            // ~1500 and managed 18. Beyond NT_FOREST_NEAR from THE CAMERA the
            // streamer switches a whole 3×3 block to one 384 m cell: 9× fewer
            // objects for crowns that are 2-4 px wide.
            lo.cellSize = envF("NT_FOREST_CELL", 128.f);
            lo.farCellSize = lo.cellSize * 3.f;
            lo.mesh.seaLevel = reg.seaLevel;
            lo.mesh.maxSlopeDeg = so.maxSlopeDeg;// same gate as the sites, or the
            lo.mesh.minGroundHeight = so.minGroundHeight;// handoff grows new forest

            // ── sites → a fine spatial grid, once ──────────────────────────
            // The whole pack's sites, binned on the streamer's fine cell. A
            // cell build is then a lookup, not a scan.
            auto grid = std::make_shared<std::unordered_map<std::int64_t,
                                                            std::vector<vegetation::TreeSite>>>();
            const float gcs = lo.cellSize;
            const auto gkey = [](int cx, int cz) {
                return (static_cast<std::int64_t>(cx) << 32) ^ static_cast<std::uint32_t>(cz);
            };
            for (const auto& s : sites)
                (*grid)[gkey(static_cast<int>(std::floor(s.x / gcs)),
                             static_cast<int>(std::floor(s.z / gcs)))]
                        .push_back(s);

            auto speciesPtr = std::make_shared<std::array<vegetation::SpeciesVariants, 3>>(species);
            auto heightFn = prov.height;
            auto builder = [grid, gkey, speciesPtr, canopyMat, heightFn, lo, &pack](
                                   int level, int cx, int cz, float) -> std::shared_ptr<Object3D> {
                std::vector<vegetation::TreeSite> sub;
                const int span = level ? 3 : 1;
                const int bx = level ? cx * 3 : cx, bz = level ? cz * 3 : cz;
                for (int i = 0; i < span; ++i)
                    for (int j = 0; j < span; ++j) {
                        auto it = grid->find(gkey(bx + j, bz + i));
                        if (it != grid->end())
                            sub.insert(sub.end(), it->second.begin(), it->second.end());
                    }
                if (sub.empty()) return nullptr;
                auto g = Group::create();
                g->name = level ? "forest_coarse" : "forest_fine";
                vegetation::ForestLodOptions co = lo;
                co.coarseOnly = level != 0;
                // Per-cell seed: the yaw stream restarts inside every build, so
                // a shared seed would give every cell the same rotation
                // sequence — a rhythm the eye finds on a hillside.
                co.seed = lo.seed ^ (static_cast<unsigned int>(cx) * 73856093u) ^
                          (static_cast<unsigned int>(cz) * 19349663u) ^
                          (static_cast<unsigned int>(level) * 83492791u);
                vegetation::buildCanopyForestLod(*g, sub, *speciesPtr, canopyMat, pack.canopy,
                                                 pack.grid, heightFn, co);
                return g;
            };

            terrain::CellStreamerOptions cso;
            cso.fineCellSize = lo.cellSize;
            cso.fineRadius = envF("NT_FOREST_NEAR", 800.f);
            cso.coarseRadius = envF("NT_FOREST_FAR", urbanPack ? 1600.f : 1100.f);
            cso.maxCellBuildsPerFrame = static_cast<int>(envF("NT_STREAM_BUDGET", 2.f));
            auto forestStream = terrain::CellStreamer::create(builder, cso);
            forestStream->name = "forest_stream";
            scene.add(forestStream);
            forestStream->update(camera.position);// the startup ring, in one go
            streamers.push_back(forestStream);
            forest.reset();

            int fm = 0, fi = 0;
            forestStream->traverseType<InstancedMesh>([&](InstancedMesh& im) {
                ++fm;
                fi += static_cast<int>(im.count());
            });
            const auto tf1 = std::chrono::high_resolution_clock::now();
            vegMeshes += fm;
            vegInstances += fi;
            vegCells += forestStream->stats().active;
            std::cout << "[norway] forest stream: " << sites.size() << " pack-wide sites, "
                      << forestStream->stats().active << " live cells (" << fm << " meshes, "
                      << fi << " instances) @ " << cso.fineCellSize << " m < " << cso.fineRadius
                      << " m, " << (cso.fineCellSize * 3.f) << " m to " << cso.coarseRadius
                      << " m; L0 < " << lo.l0Distance << " m, L1, L2 1-in-" << lo.l2Keep << " > "
                      << lo.l1Distance << " m, budget " << cso.maxCellBuildsPerFrame
                      << " builds/frame, " << std::chrono::duration<float>(tf1 - tf0).count()
                      << " s\n"
                      << std::flush;
        } else {
            vegetation::ForestOptions fo;
            fo.cameraPos = camera.position;
            fo.cap = static_cast<int>(envF("NT_FOREST_CAP", urbanPack ? 80000.f : 40000.f));
            if (urbanPack) {
                fo.spruceMinElevation = envF("NT_SPRUCE_ELEV", 60.f);
                fo.spruceMinStandHeight = envF("NT_SPRUCE_STAND", 12.f);
            }
            const auto st = vegetation::buildCanopyForest(*forest, sites, species, prov.height, fo);
            scene.add(forest);
            vegMeshes += st.meshes;
            vegInstances += st.planted;
            const auto tf1 = std::chrono::high_resolution_clock::now();
            std::cout << "[norway] forest: " << st.sites << " sites (CHM local maxima), "
                      << st.planted << " instances (" << st.nearTier << " near + " << st.farTier
                      << " far) in " << st.meshes << " meshes, "
                      << std::chrono::duration<float>(tf1 - tf0).count() << " s\n"
                      << std::flush;
        }

        // Species split, for the record: the rule is invisible in a shot until
        // it is wrong, and "why is that garden full of spruce" is the failure.
        {
            std::array<int, 3> split{};
            const float se = urbanPack ? envF("NT_SPRUCE_ELEV", 60.f) : 0.f;
            const float ss = urbanPack ? envF("NT_SPRUCE_STAND", 12.f) : 0.f;
            for (const auto& s : sites)
                ++split[static_cast<size_t>(vegetation::pickTreeSpecies(
                        s.canopyHeight, prov.height(s.x, s.z), s.standHeight, 6.f, 14.f, 350.f,
                        se, ss))];
            std::cout << "[norway] species: " << split[0] << " scrub, " << split[1]
                      << " broadleaf, " << split[2] << " spruce (spruce needs >= " << se
                      << " m elevation AND >= " << ss << " m stand)\n"
                      << std::flush;
        }

        // ── Bush tier ──────────────────────────────────────────────────────
        // The 1.0-2.5 m band of the CHM: garden shrubs and the scrub along
        // walls. Restricted to gardens (within 40 m of a footprint), because
        // the same band over open hillside is 42 000 blobs nobody looks at.
        if (urbanPack && !envSet("NT_NO_BUSHES")) {
            const auto tb0 = std::chrono::high_resolution_clock::now();
            vegetation::CanopySiteOptions bo = so;
            bo.minHeight = 1.0f;
            bo.maxCanopyHeight = 2.5f;
            bo.windowRadius = 1;
            bo.minNeighborSupport = 0;// a shrub is not a stand
            bo.treelineElevation = 1e9f;
            bo.highSlopeElevation = 1e9f;
            // 2 m spacing: crownRadiusFactor · spacingFactor · (h1 + h2) with
            // h ≈ 2 m, and the thinner's own 4 m floor on the bin size.
            bo.crownRadiusFactor = 0.5f;
            bo.spacingFactor = 1.0f;
            bo.reject = [&, fpMask, deckMask, gardenMask](float x, float z) {
                if (gardenMask && !gardenMask->inside(x, z)) return true;// not a garden
                if (fpMask && fpMask->inside(x, z)) return true;
                if (network.pavedWeight(x, z, 1.0f) > 0.2f) return true;
                if (deckMask && deckMask->inside(x, z)) return true;
                return false;
            };
            vegetation::CanopySiteReport brep;
            auto bushes = vegetation::detectTreeSites(pack.canopy, pack.grid, bo, &brep);
            const int chmBushes = static_cast<int>(bushes.size());

            // Hedges: OSM barrier=hedge lines, one bush every 1.5 m. A hedge is
            // a LINE of the same prototype — no CHM peak survives a 0.6 m wide
            // object on a 2 m grid, so the data has to place these.
            int hedgeBushes = 0;
            if (pack.hasLandUse()) {
                const float roi = bo.halfExtent;
                for (const auto& l : pack.landuse.lines) {
                    if (l.cls != "hedge" || l.points.size() < 2) continue;
                    float acc = 1.5f;
                    for (size_t i = 1; i < l.points.size(); ++i) {
                        const float dx = l.points[i].x - l.points[i - 1].x;
                        const float dz = l.points[i].y - l.points[i - 1].y;
                        const float len = std::sqrt(dx * dx + dz * dz);
                        if (len < 1e-3f) continue;
                        float pos = 0.f;
                        while (acc + (len - pos) >= 1.5f) {
                            pos += 1.5f - acc;
                            acc = 0.f;
                            const float x = l.points[i - 1].x + dx * (pos / len);
                            const float z = l.points[i - 1].y + dz * (pos / len);
                            if (std::fabs(x - bo.centerX) > roi || std::fabs(z - bo.centerZ) > roi)
                                continue;
                            if (pack.grid.sampleBilinear(x, z) < bo.seaLevel + bo.minGroundHeight)
                                continue;
                            bushes.push_back({x, z, 1.6f, 0.f});
                            ++hedgeBushes;
                        }
                        acc += len - pos;
                    }
                }
            }

            if (!bushes.empty()) {
                // Three greens around the prototype's default, all of them
                // LIGHTER than a wet road: a hedge in sun is the brightest
                // green in a suburban frame, and the first pass had it darker
                // than the asphalt beside it.
                std::vector<vegetation::TreeVariant> bushVars{
                        vegetation::makeBushVariant(4101u),
                        vegetation::makeBushVariant(4102u, Color(0.36f, 0.54f, 0.19f)),
                        vegetation::makeBushVariant(4103u, Color(0.27f, 0.45f, 0.14f))};
                vegetation::BushOptions bopt;
                bopt.cap = static_cast<int>(envF("NT_BUSH_CAP", 1e9f));
                bopt.cullDistance = envF("NT_BUSH_CULL", 400.f);
                bopt.cellSize = 128.f;
                // Same treatment as the forest: pack-wide sites, fine cells
                // only (a 1.6 m shrub has no far tier worth the object).
                auto bgrid = std::make_shared<
                        std::unordered_map<std::int64_t, std::vector<vegetation::TreeSite>>>();
                const auto bkey = [](int cx, int cz) {
                    return (static_cast<std::int64_t>(cx) << 32) ^ static_cast<std::uint32_t>(cz);
                };
                for (const auto& s : bushes)
                    (*bgrid)[bkey(static_cast<int>(std::floor(s.x / bopt.cellSize)),
                                  static_cast<int>(std::floor(s.z / bopt.cellSize)))]
                            .push_back(s);
                auto bvars = std::make_shared<std::vector<vegetation::TreeVariant>>(bushVars);
                auto bHeight = prov.height;
                terrain::CellStreamerOptions bso;
                bso.fineCellSize = bopt.cellSize;
                bso.fineRadius = bopt.cullDistance;
                bso.coarseRadius = 0.f;// fine only
                bso.maxCellBuildsPerFrame = static_cast<int>(envF("NT_STREAM_BUDGET", 2.f));
                auto bushStream = terrain::CellStreamer::create(
                        [bgrid, bkey, bvars, bHeight, bopt](int, int cx, int cz,
                                                            float) -> std::shared_ptr<Object3D> {
                            auto it = bgrid->find(bkey(cx, cz));
                            if (it == bgrid->end() || it->second.empty()) return nullptr;
                            auto g = Group::create();
                            g->name = "bush_cell_root";
                            vegetation::BushOptions co = bopt;
                            co.seed = bopt.seed ^ (static_cast<unsigned int>(cx) * 73856093u) ^
                                      (static_cast<unsigned int>(cz) * 19349663u);
                            vegetation::buildBushField(*g, it->second, *bvars, bHeight, co);
                            return g;
                        },
                        bso);
                bushStream->name = "bush_field";
                scene.add(bushStream);
                bushStream->update(camera.position);
                streamers.push_back(bushStream);
                int bm = 0, bi = 0;
                bushStream->traverseType<InstancedMesh>([&](InstancedMesh& im) {
                    ++bm;
                    bi += static_cast<int>(im.count());
                });
                vegMeshes += bm;
                vegInstances += bi;
                vegCells += bushStream->stats().active;
                const auto tb1 = std::chrono::high_resolution_clock::now();
                std::cout << "[norway] bushes: " << brep.peaks << " CHM peaks 1-2.5 m -> "
                          << chmBushes << " garden sites + " << hedgeBushes
                          << " hedge sites (rejected: masked " << brep.rejectedMasked
                          << ", ground " << brep.rejectedGround << ", slope "
                          << brep.rejectedSlope << ") -> " << bi << " instances live in "
                          << bushStream->stats().active << " streamed cells, radius "
                          << bso.fineRadius << " m, "
                          << std::chrono::duration<float>(tb1 - tb0).count() << " s\n"
                          << std::flush;
            }
        }

        // The one line to read when the frame is slow.
        std::cout << "[norway] vegetation objects: " << vegMeshes
                  << " InstancedMesh in " << vegCells << " cells, " << vegInstances
                  << " instances ("
                  << (vegMeshes > 0 ? vegInstances / vegMeshes : 0)
                  << " per mesh)\n"
                  << std::flush;
    }

    // ── urban props: pier decks, parked cars, moored boats ─────────────────
    // Everything here is placed from the survey, never scattered, and every
    // placement passes the same gates the trees do (footprint, pavement, sea).
    // Static ROI around the view target, like the forest: a car at 2 km is a
    // sub-pixel smudge that still costs a draw. NT_NO_PROPS=1 for the A/B,
    // NT_PROPS_EXTENT / NT_PROPS_CELL to sweep.
    if (pack.hasLandUse() && !envSet("NT_NO_PROPS")) {
        const auto tp0 = std::chrono::high_resolution_clock::now();
        // 1 m dilation only: a car parked hard against a wall is normal, a car
        // INSIDE the wall is the failure this gate exists for.
        const auto propFp = terrain::buildFootprintMask(pack, 1.f, 2.f);
        const auto propUrban = gopt.paintUrban ? terrain::buildUrbanMask(pack, gopt) : nullptr;
        terrain::UrbanPropsOptions po;
        po.seaLevel = reg.seaLevel;
        // Pack-wide placement (see the forest: the ROI was the defect), 250 m
        // cells streamed by the camera out to NT_PROPS_EXTENT. Decks and boats
        // stay unstreamed: they are a few hundred objects for the whole pack
        // and they are part of the LAND, not scatter.
        po.centerX = 0.f;
        po.centerZ = 0.f;
        po.halfExtent = 1e9f;
        po.cellSize = envF("NT_PROPS_CELL", 250.f);
        po.cars = !envSet("NT_NO_CARS");
        po.boats = !envSet("NT_NO_BOATS");
        po.decks = !envSet("NT_NO_DECKS");
        po.urban = propUrban.get();
        po.footprints = propFp.get();
        po.ground = prov.height;
        auto props = Group::create();
        props->name = "urban_props";
        auto carField = std::make_shared<terrain::UrbanCarField>();
        const auto ps = terrain::buildUrbanProps(*props, pack, network, po, carField.get());
        scene.add(props);

        auto carMat = terrain::makeUrbanCarMaterial();
        terrain::CellStreamerOptions pso;
        pso.fineCellSize = po.cellSize;
        pso.fineRadius = envF("NT_PROPS_EXTENT", 1500.f);
        pso.coarseRadius = 0.f;// a car has no far tier: past the ring, nothing
        pso.maxCellBuildsPerFrame = static_cast<int>(envF("NT_STREAM_BUDGET", 2.f));
        auto carStream = terrain::CellStreamer::create(
                [carField, carMat](int, int cx, int cz, float) -> std::shared_ptr<Object3D> {
                    const auto* cars = carField->at(cx, cz);
                    if (!cars) return nullptr;
                    return terrain::buildCarCellMesh(*cars, carMat,
                                                     "cars_" + std::to_string(cx) + "_" +
                                                             std::to_string(cz));
                },
                pso);
        carStream->name = "car_stream";
        scene.add(carStream);
        carStream->update(camera.position);
        streamers.push_back(carStream);
        size_t liveCarTris = 0;
        int liveCarMeshes = 0;
        carStream->traverseType<Mesh>([&](Mesh& m) {
            ++liveCarMeshes;
            if (auto* p = m.geometry()->getAttribute<float>("position"))
                liveCarTris += p->count() / 3;
        });
        const auto tp1 = std::chrono::high_resolution_clock::now();
        std::printf("[norway] urban props: %d pier deck runs, %zu cars pack-wide (%d lot + %d "
                    "kerb) in %d cells @ %.0f m -> %d cells live (%d meshes, %zu tris) within "
                    "%.0f m of the camera; %d boats in %d marinas; deck/boat meshes %zu, "
                    "%zu tris; rejected roof %d, sea %d, paved %d, junction %d; %.2f s\n",
                    ps.deckLines, carField->count(), ps.carsLot, ps.carsKerb, ps.carCells,
                    po.cellSize, carStream->stats().active, liveCarMeshes, liveCarTris,
                    pso.fineRadius, ps.boats, ps.marinas, ps.meshes, ps.triangles, ps.rejectRoof,
                    ps.rejectSea, ps.rejectPaved, ps.rejectJunction,
                    std::chrono::duration<float>(tp1 - tp0).count());
        std::fflush(stdout);
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
        // --fly: linear traverse from the --cam pose to the --fly pose over
        // NT_FLY_FRAMES frames, starting at --seqstart. This is the only camera
        // MOTION the demo has (NT_ORBIT drifts, it does not travel), and it is
        // what the content streamers have to survive.
        if (haveFly && haveCam) {
            const int flyFrames = static_cast<int>(envF("NT_FLY_FRAMES", 600.f));
            const float t = std::clamp(static_cast<float>(frame - seqStart) /
                                               static_cast<float>(std::max(1, flyFrames)),
                                       0.f, 1.f);
            camera.position.lerpVectors(camPosArg, flyPosArg, t);
            Vector3 look = camTargetArg;
            if (haveFlyTarget) look.lerpVectors(camTargetArg, flyTargetArg, t);
            camera.lookAt(look);
            lodPos = camera.position;
        }
        if (!freezeTiles) tiles->update(lodPos);
        if (scatter && !freezeTiles) scatter->update(lodPos);
        // Vegetation and props follow the LIVE camera, on the same position and
        // the same freeze rule as the tiles.
        if (!freezeTiles)
            for (auto& s : streamers) s->update(lodPos);
        // Streamer churn, printed only on the frames where something changed:
        // adds and (especially) removes are what a fly-through can flash on,
        // because removing a scene entry clears the renderer's temporal history.
        if (!streamers.empty() && (!seqPrefix.empty() || envSet("NT_STREAM_LOG"))) {
            int nb = 0, na = 0, nr = 0, nact = 0, npend = 0;
            for (auto& s : streamers) {
                const auto& st = s->stats();
                nb += st.builds;
                na += st.adds;
                nr += st.removes;
                nact += st.active;
                npend += st.pending;
            }
            if (nb || na || nr) {
                int objs = 0;
                scene.traverse([&](Object3D&) { ++objs; });
                std::printf("[stream] frame %d built %d added %d removed %d active %d pending %d "
                            "objects %d\n",
                            frame, nb, na, nr, nact, npend, objs);
                std::fflush(stdout);
            }
        }
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
