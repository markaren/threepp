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

#include "threepp/threepp.hpp"

#include "threepp/extras/road/RoadNetwork.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/GeoTerrain.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
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
                  << net.roadCount() << "): median |Δh| " << pct(0.5f) << " m, p90 " << pct(0.9f)
                  << " m, max " << d.back() << " m; " << over3
                  << " pts > 3 m (likely bridges/tunnels)\n"
                  << std::flush;
    }

}// namespace

int main(int argc, char** argv) {

    // ── args ──────────────────────────────────────────────────────────────────
    std::string packArg;
    std::string shotPath;
    int shotFrames = 160;
    bool haveCam = false;
    Vector3 camPosArg, camTargetArg;
    bool noRoadBias = false;// disable the road-aware LOD refinement (A/B compare)
    std::string profilePath;// --road-profile <csv>: dump the height field along the longest road and exit
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
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
              << ", roads " << pack.roads.size() << "\n"
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
    // Conform roads to the RAW DEM height first (roads must meet real ground),
    // THEN carve the road cut into the grid: every DEM cell near a road is
    // clamped below the conformed ribbon surface, so the provider is a pure
    // bicubic of the carved grid — C1 everywhere, no runtime corridor warp, and
    // no sub-quad features for a tile split to pop in ("humps").
    network.conformTo([&pack](float x, float z) { return pack.grid.sampleBicubic(x, z); });
    terrain::carveRoads(pack.grid, network);

    // ── provider + tiles ───────────────────────────────────────────────────────
    terrain::GeoTerrainOptions gopt;
    gopt.snowHeightMin = std::max(reg.heightMax - 350.f, 900.f);// scene-relative snowline
    gopt.grassHeightMax = std::clamp(reg.heightMin + 0.45f * (reg.heightMax - reg.heightMin), 200.f, 900.f);
    gopt.wetlandBand = 6.f;
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
    tileOpts.splatTexelsPerQuad = 2;
    tileOpts.asyncBake = true;
    // Road-aware LOD: subdivide road-corridor tiles ~2.2× sooner/deeper so the
    // ribbon stays crisp at mid distance (terrain interp error shrinks with tile
    // size). The 1.2/1.7 split/merge dead band is preserved under the bias.
    if (!noRoadBias) {
        tileOpts.refineBias = [&network](float cx, float cz, float half) {
            return network.corridorIntersects(cx, cz, half) ? 2.2f : 1.0f;
        };
    }
    {
        const terrain::DetailMaps dm = terrain::makeDetailMaps({});
        tileOpts.detailMap = dm.albedo;
        tileOpts.detailNormalMap = dm.normalRough;
        tileOpts.detailRepeat = 0.6f;
        tileOpts.detailStrength = 0.7f;
        tileOpts.detailNormalScale = 1.0f;
        tileOpts.detailRoughStrength = 0.5f;
    }

    // ── renderer ────────────────────────────────────────────────────────────────
    Canvas canvas("threepp - NORWAY TERRAIN", {{"vsync", false}});
    // Headless capture forces the Vulkan deferred renderer (detail-map layer is
    // Vulkan-only); interactive runs keep the renderer-select menu.
    auto renderer = shotPath.empty() ? createRenderer(canvas)
                                     : createRenderer(canvas, GraphicsAPI::Vulkan);
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 1.0f;

    Scene scene;
    RGBELoader rgbe;
    if (auto env = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = env;
        scene.environment = env;
    } else {
        scene.background = Color(0.55f, 0.70f, 0.92f);
        std::cerr << "[norway] HDRI env not found — flat sky fallback\n";
    }

    auto tiles = terrain::TileTerrain::create(prov, tileOpts);
    tiles->name = "norway_terrain";
    scene.add(tiles);

    // Road ribbons.
    auto roads = network.buildMeshes();
    scene.add(roads);

    // Sea plane on low / coastal packs.
    if (reg.heightMin < 1.0f) {
        auto seaMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                           .color(Color(0.045f, 0.09f, 0.13f))
                                                           .roughness(0.15f)
                                                           .metalness(0.f));
        auto sea = Mesh::create(PlaneGeometry::create(reg.worldSize * 1.2f, reg.worldSize * 1.2f), seaMat);
        sea->rotation.x = -math::PI / 2.f;
        // Slightly ABOVE the DTM's flat sea sheet (water reads as seaLevel exactly,
        // and the provider suppresses relief noise there) so the seabed never dithers
        // through; real land starts well above this.
        sea->position.y = reg.seaLevel + 0.15f;
        sea->name = "sea";
        scene.add(sea);
    }

    // Directional sun (raking, ~SW) + HDRI ambient.
    auto sun = DirectionalLight::create(Color(1.0f, 0.96f, 0.88f), 2.8f);
    {
        const float az = 215.f * kDeg2Rad, el = 34.f * kDeg2Rad;
        sun->position.set(std::cos(el) * std::sin(az), std::sin(el), std::cos(el) * std::cos(az));
        sun->position.multiplyScalar(std::max(reg.worldSize, 2000.f));
    }
    Object3D sunTarget;
    sunTarget.position.set(0.f, reg.heightMin, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    // ── camera above the longest road, looking along it ─────────────────────────
    PerspectiveCamera camera(55.f, canvas.aspect(), 1.f, std::max(reg.worldSize * 3.f, 12000.f));
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

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer->setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;
    int frame = 0;
    float fpsAccum = 0.f, fps = 0.f;
    int fpsFrames = 0;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        fpsAccum += dt;
        if (++fpsFrames, fpsAccum >= 0.5f) {
            fps = static_cast<float>(fpsFrames) / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        controls.update();
        tiles->update(camera.position);
        renderer->render(scene, camera);

        if (!shotPath.empty() && ++frame >= shotFrames) {
            const auto path = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
            std::filesystem::create_directories(path.parent_path());
            renderer->writeFramebuffer(path);
            std::cout << "wrote " << path.string() << " (" << fps << " fps, tiles "
                      << tiles->activeTiles() << ", baking " << tiles->pendingBakes() << ")"
                      << std::endl;
            std::exit(0);
        }
    });

    return 0;
}
