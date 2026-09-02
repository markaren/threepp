// 3D Gaussian Splats, on either backend.
//
//   gaussian_splats [<file.ply>]           an INRIA-convention splat file
//   gaussian_splats                        three interpenetrating procedural clouds
//
//   --vulkan                               render through the Vulkan compute
//                                          tile rasterizer instead of GL
//   --shot <out.png> [--frames N]          headless capture, then exit
//   --screenshot=<out.png>                 same thing, spelled the other way
//   --cam x,y,z --look x,y,z               reframe a capture without rebuilding
//   --occluder                             a box through the middle of the cloud
//   --points                               every splat as an opaque disc (point
//                                          cloud view); P toggles it live with a
//                                          short dissolve, either backend
//   --point-size N                         disc diameter in pixels (default 2)
//   --point-mix M                          0..1, a fixed blend of the two looks
//                                          (the splat-behind-mesh depth test)
//   --wireframe                            the same box, wireframe: the same test
//                                          for the post-resolve OVERLAY path,
//                                          where the lines behind the cloud must
//                                          be hidden by it
//   --msaa N                               canvas antialiasing (0 = off, the
//                                          default, for GL parity). On Vulkan
//                                          N > 1 moves the overlay onto its
//                                          hardware-MSAA depth attachment —
//                                          pair with --wireframe to test it
//   --water                                a calm pond under the cloud (Vulkan
//                                          only): the splat-in-the-mirror check —
//                                          the traced water reflection marches the
//                                          cloud's baked volume (svLeg), so the
//                                          scan must appear in the pond
//   --metal                                a polished metal sphere beside the
//                                          cloud (Vulkan only): the same check on
//                                          the GLOSSY reflection path — the scan
//                                          must appear on the sphere
//   --no-flip                              keep a loaded file's own axes
//   --no-cull                              keep a loaded file's outlier splats
//   --morton                               Morton-reorder storage (see below)
//   --bench N                              N frames on a slow orbit, timings, exit
//   --level N                              which SOG detail level to read (0 = finest)
//   --scale S                              render scale; the splat pass is measured
//                                          at RENDER resolution, so this is the only
//                                          flag that changes its pixel count
//   --clouds K                             partition the same splats across K clouds
//                                          (constant total) — measures the per-cloud
//                                          tax, which is ~1.3 ms each: see
//                                          doc/vulkan_splats.md
//   --upscaler                             leave DLSS/FSR enabled (off by default for
//                                          GL parity). Alone it changes nothing: at
//                                          scale 1 DLSS selects DLAA and upscales
//                                          nothing — pair it with --scale
//
// ONE BINARY, TWO BACKENDS, ON PURPOSE. The GL path is the correctness oracle
// for the Vulkan one, and an oracle is only worth having if the thing being
// compared is genuinely the same: same asset, same outlier cull, same framing
// arithmetic, same camera. Everything above this line is shared; the only
// branch is which renderer is constructed.
//
// Comparing the two: run both with --shot and the SAME --cam/--look, and pass
// --vulkan a tone mapping of None (it is, by default here) so the sRGB decode
// the splat pass does on the way into linear HDR and the sRGB encode the post
// stack does on the way out are inverses. With a tone curve enabled the Vulkan
// splats tone-map along with the rest of the scene — correct, and deliberately
// not what GL does, which writes display-referred values straight out.
//
// Loaded scans are run through SplatData::removeOutliers by default: a
// photogrammetry scan carries a tail of enormous near-opaque splats that the
// optimiser parked in the sky, and they smear over the subject from most
// angles. --no-cull renders the file as authored.
//
// --morton runs SplatData::reorderMorton first, and it is OFF by default,
// because the fetch-locality hypothesis behind it MEASURED AS A REGRESSION on
// this draw-order path: 82-87 ms vs 50.9 ms orbiting the 5M Sanctuaire scan
// (three interleaved rounds, 2026-08-05), and no win on the 216k control.
// Correctness is unaffected (49.8 dB vs file order, tie-order scale). Best
// current explanation: the counting sort's ~80-splat tie groups become
// spatially contiguous on screen under Morton order, so consecutive
// overlapping quads serialize in the blend stage on the same framebuffer
// tiles; file order scatters them. The reorder stays available because a
// TILE-based rasterizer reads splats per screen region, where spatial storage
// locality aligns with access instead of fighting it.
//
// --bench orbits deliberately. update() early-outs when the camera has not
// moved, so a benchmark on a static camera measures the draw alone and quietly
// skips the sort and the per-frame index upload -- the two costs that actually
// scale with splat count.
//
// Loaded scans are turned 180 degrees about X by default, because the
// photogrammetry pipelines that produce them work +Y-down and threepp is
// +Y-up; without it every real capture arrives upside down.
//
// The procedural scene is built so that a rendering mistake is visible rather
// than plausible. Three ellipsoidal shells of flattened, randomly oriented
// Gaussians interpenetrate at different depths, in three saturated hues: if the
// back-to-front sort is wrong the shells stop looking like they pass through
// each other and start looking like whichever one happens to be drawn last is
// in front, from every angle. Orbiting is the test — a still frame of a broken
// sort can look perfectly reasonable.
//
// Press D to paint non-finite fragments magenta instead of discarding them.

#include "capture_util.hpp"

#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/SogLoader.hpp"
#include "threepp/splats/SplatLod.hpp"
#include "threepp/loaders/SplatLoader.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/threepp.hpp"

#if defined(THREEPP_WITH_VULKAN)
#include "threepp/objects/Ocean.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // Reshapes a generated cloud into a shell of flat, randomly oriented discs
    // in one hue. The generator supplies the determinism (and the rotations,
    // which stay as they are — randomly oriented discs are exactly what makes
    // the anisotropy visible).
    SplatData shell(const Vector3& centre, const Vector3& radii, const Vector3& hue,
                    size_t count, unsigned int seed, int degree) {

        SplatGenerator::Options options;
        options.count = count;
        options.seed = seed;
        options.shDegree = degree;
        options.extent.set(2.f, 2.f, 2.f);
        options.minScale = 0.03f;
        options.maxScale = 0.07f;
        options.anisotropy = 5.f;
        options.minOpacity = 0.10f;
        options.maxOpacity = 0.30f;
        options.higherOrderAmplitude = 0.f;

        auto data = SplatGenerator::generate(options);

        for (size_t i = 0; i < data.count(); ++i) {

            // The generated mean is a point in a box; use its direction, and
            // its length as the shell thickness. Points near the centre would
            // divide by ~0, so nudge them outward first.
            Vector3 dir = data.means[i];
            if (dir.length() < 1e-3f) dir.set(0.f, 1.f, 0.f);
            const float thickness = 0.86f + 0.14f * std::min(1.f, dir.length());
            dir.normalize();

            data.means[i].set(centre.x + dir.x * radii.x * thickness,
                              centre.y + dir.y * radii.y * thickness,
                              centre.z + dir.z * radii.z * thickness);

            // Flatten one axis: discs, not blobs. Aligned with the splat's own
            // (random) rotation, so the cloud has visible grain.
            data.scales[i].z *= 0.15f;

            // Hue with a little per-splat variation, so it does not read as a
            // flat wash.
            const float jitter = 0.85f + 0.3f * static_cast<float>(i % 7) / 6.f;
            data.setDcColor(i, Vector3{hue.x * jitter, hue.y * jitter, hue.z * jitter});

            if (degree >= 1) {

                // One band-1 term per cloud: a directional sheen that only
                // appears while orbiting. Nothing else in the demo exercises
                // the view-dependent half of the shader.
                float* c = data.shAt(i);
                c[1 * 3 + 0] = 0.25f * hue.z;
                c[1 * 3 + 1] = 0.25f * hue.x;
                c[1 * 3 + 2] = 0.25f * hue.y;
            }
        }

        return data;
    }

    void append(SplatData& dst, const SplatData& src) {

        if (dst.count() == 0) dst.shDegree = src.shDegree;
        if (dst.shDegree != src.shDegree) throw std::runtime_error("SH degree mismatch");

        dst.means.insert(dst.means.end(), src.means.begin(), src.means.end());
        dst.scales.insert(dst.scales.end(), src.scales.begin(), src.scales.end());
        dst.rotations.insert(dst.rotations.end(), src.rotations.begin(), src.rotations.end());
        dst.opacities.insert(dst.opacities.end(), src.opacities.begin(), src.opacities.end());
        dst.sh.insert(dst.sh.end(), src.sh.begin(), src.sh.end());
    }

    // The clouds share one SplatCloud on purpose. Three separate objects would
    // be ordered against each other by the renderer's per-object transparency
    // sort, which is far too coarse for shells that interpenetrate — the whole
    // point is that the ordering is per splat.
    SplatData proceduralScene() {

        constexpr int DEGREE = 1;

        // Spread mostly along x, so the default camera (mostly along +z) sees
        // three overlapping shells side by side rather than one behind another,
        // and staggered in z so which one is in front actually depends on where
        // you are standing.
        SplatData data;
        append(data, shell({-1.75f, 0.f, -1.3f}, {1.25f, 1.45f, 1.25f}, {1.0f, 0.22f, 0.16f}, 900, 101u, DEGREE));
        append(data, shell({0.f, 0.f, 0.f}, {1.45f, 1.15f, 1.45f}, {0.18f, 0.95f, 0.32f}, 900, 202u, DEGREE));
        append(data, shell({1.75f, 0.f, 1.3f}, {1.25f, 1.45f, 1.25f}, {0.20f, 0.40f, 1.0f}, 900, 303u, DEGREE));

        return data;
    }

}// namespace


int main(int argc, char** argv) {

    // --cam / --look / --frames / --out come from the shared capture harness.
    const auto args = capture::parseArgs(argc, argv);

    std::string plyPath;
    std::string shotPath = args.out.value_or("");
    int shotFrames = args.frames.value_or(10);
    int shotFrame = 0;
    bool flip = true;
    bool cull = true;
    bool morton = false;// measured slower on the GL draw-order path; see header
    int benchFrames = 0;
    bool useVulkan = false;
    bool occluder = false;
    bool wireOccluder = false;
    // Point rendering (SplatCloud::setPointMix). --points is mix 1; the P key
    // dissolves between 0 and 1 over kPointDissolveSeconds.
    float pointMix = 0.f;
    float pointSize = 2.f;
    // Canvas antialiasing. 0 by default for GL parity in captures; raising it
    // puts the Vulkan overlay pass on its hardware-MSAA path, which is the
    // other depth attachment the splat depth stamp has to write.
    int msaa = 0;
    bool debugNaN = false;
    bool fog = false;
    bool addSun = false;
    bool water = false;
    bool metal = false;
    // 0.15 by default (see the --metal block for why); --metal-rough drops it
    // toward a true mirror, which is how the VOLUME's resolution is judged
    // without the reflection denoiser's gloss blur on top.
    float metalRough = 0.15f;
    Vector3 metalPos;
    bool metalPosSet = false;
    // --level picks a SOG detail level; --upscaler lifts the GL-parity clamp
    // below so the pass can be measured at a render scale a real app would use.
    int lodLevel = 0;
    bool upscaler = false;
    float renderScale = 1.f;
    int cloudSplit = 1;
    bool lodDynamic = false;

    for (int i = 1; i < argc; ++i) {

        const std::string arg = argv[i];
        if (arg == "--vulkan") {
            useVulkan = true;
        } else if (arg == "--debug-nan") {
            debugNaN = true;
        } else if (arg == "--occluder") {
            occluder = true;
        } else if (arg == "--points") {
            pointMix = 1.f;
        } else if (arg == "--point-mix" && i + 1 < argc) {
            pointMix = std::clamp(std::stof(argv[++i]), 0.f, 1.f);
        } else if (arg == "--point-size" && i + 1 < argc) {
            pointSize = std::stof(argv[++i]);
        } else if (arg == "--msaa" && i + 1 < argc) {
            msaa = std::atoi(argv[++i]);
        } else if (arg == "--wireframe") {
            occluder = true;
            wireOccluder = true;
        } else if (arg == "--fog") {
            fog = true;
        } else if (arg == "--water") {
            water = true;
        } else if (arg == "--metal") {
            metal = true;
        } else if (arg == "--metal-rough" && i + 1 < argc) {
            metal = true;
            metalRough = std::clamp(static_cast<float>(std::atof(argv[++i])), 0.f, 1.f);
        } else if (arg == "--metal-pos" && i + 1 < argc) {
            // World position for the sphere. The default "beside the cloud"
            // placement assumes empty space there, which a town scan with a
            // wooded hill does not honour — this is the override for real
            // captures, like --cam is for the camera.
            metal = true;
            if (std::sscanf(argv[++i], "%f,%f,%f",
                            &metalPos.x, &metalPos.y, &metalPos.z) == 3)
                metalPosSet = true;
        } else if (arg == "--sun") {
            addSun = true;
        } else if (arg == "--level" && i + 1 < argc) {
            lodLevel = std::atoi(argv[++i]);
        } else if (arg == "--upscaler") {
            upscaler = true;
        } else if (arg == "--scale" && i + 1 < argc) {
            renderScale = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--clouds" && i + 1 < argc) {
            cloudSplit = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--lod-dynamic") {
            lodDynamic = true;
        } else if (arg == "--shot" && i + 1 < argc) {
            shotPath = argv[++i];
        } else if (arg.rfind("--screenshot=", 0) == 0) {
            shotPath = arg.substr(13);
        } else if (arg == "--no-flip") {
            flip = false;
        } else if (arg == "--no-cull") {
            cull = false;
        } else if (arg == "--morton") {
            morton = true;
        } else if (arg == "--no-morton") {// kept so A/B scripts don't break
            morton = false;
        } else if (arg == "--bench" && i + 1 < argc) {
            benchFrames = std::atoi(argv[++i]);
        } else if (arg == "--cam" || arg == "--look" || arg == "--frames" || arg == "--out") {
            ++i;// consumed by parseArgs
        } else if (arg.rfind("--", 0) != 0) {
            plyPath = arg;
        }
    }

    // Dynamic LOD is a VULKAN feature: the GL path ignores submission ranges,
    // so it would draw every resident level stacked on top of each other — the
    // town two or three times over, fatter and slower. Refuse the combination
    // rather than render it.
    if (lodDynamic && !useVulkan) {
        std::cerr << "--lod-dynamic needs --vulkan (the GL path draws every resident level);"
                     " loading level 0 only" << std::endl;
        lodDynamic = false;
    }

    // The pond exists to exercise the deferred water's traced reflection leg,
    // which is a Vulkan pass; on GL there is no reflection to put the cloud in.
    if (water && !useVulkan) {
        std::cerr << "--water needs --vulkan (the reflection march is a deferred-shade"
                     " feature); ignoring it" << std::endl;
        water = false;
    }
    if (metal && !useVulkan) {
        std::cerr << "--metal needs --vulkan (same reason as --water); ignoring it" << std::endl;
        metal = false;
    }

    SplatData data;
    splats::LodTable lodTable;// non-empty only under --lod-dynamic
    const bool loadedFromFile = !plyPath.empty();
    if (loadedFromFile) {

        std::cout << "loading " << plyPath << " ..." << std::endl;
        const auto t0 = std::chrono::steady_clock::now();
        // A SOG asset (a .zip, a .sog, or an unpacked folder) is recognised by
        // content, not by extension — and it is the only shape that carries
        // detail levels, which is what --level selects. The levels are
        // ALTERNATIVES: each covers the whole scene at its own density, so
        // --level 1 is the same building from half as many splats, not half a
        // building. A .ply has exactly one level and ignores the flag.
        if (SogLoader::isSog(plyPath) && lodDynamic) {

            // --lod-dynamic: several levels in ONE cloud, chosen per frame as a
            // range list — never a re-upload and never a second cloud, both of
            // which were measured and rejected (doc/vulkan_splats.md). The
            // loading, the every-other-level memory compromise and the chunk
            // table all live in splats::loadSogWithLod, shared with the editor
            // so the two cannot drift.
            auto loaded = splats::loadSogWithLod(plyPath);
            data = std::move(loaded.data);
            lodTable = std::move(loaded.table);
            std::cout << "  SOG asset, dynamic LOD, " << lodTable.levels.size()
                      << " resident level(s):";
            for (const auto& l : lodTable.levels)
                std::cout << " [" << l.lod << "] " << l.count << " (" << l.chunks.size()
                          << " chunks)";
            std::cout << std::endl;
        } else if (SogLoader::isSog(plyPath)) {

            const auto info = SogLoader::describe(plyPath);
            std::cout << "  SOG asset, " << info.lodLevels << " level(s):";
            for (const auto& l : info.levels) std::cout << " [" << l.lod << "] " << l.count;
            std::cout << "\n  reading level " << lodLevel << std::endl;
            data = SogLoader::load(plyPath, {lodLevel});
        } else if (!SplatLoader::isSplatPly(plyPath) && SplatLoader::isPointCloudPly(plyPath)) {

            // A colour-only point cloud (no f_dc_0): one degree-0 Gaussian per
            // point, sized from the median neighbour spacing, so it draws as
            // a surface at mix 0 and as dots at mix 1. No outlier cull — the
            // rule is written for an optimiser's scale tail.
            SplatLoader::PointCloudInfo info;
            data = SplatLoader::loadPointCloudPly(plyPath, {}, &info);
            cull = false;
            std::cout << "  point cloud: " << info.count << " points, spacing " << info.spacing
                      << ", sigma " << info.sigma
                      << (info.hadColor ? ", rgb" : info.hadIntensity ? ", intensity" : ", no colour")
                      << (info.hadNormals ? ", normals -> discs" : "") << std::endl;
        } else {
            data = SplatLoader::loadPly(plyPath);
        }
        const auto parseMs = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - t0)
                                     .count();
        std::cout << "  " << data.count() << " splats, SH degree " << data.shDegree;
        for (const auto& [name, values] : data.extras) std::cout << ", extra '" << name << "'";
        std::cout << "\n  parsed in " << std::fixed << std::setprecision(1) << parseMs << " ms"
                  << std::endl;

        // Scale and opacity distributions. Worth printing for any real scan:
        // photogrammetry output routinely carries a long tail of enormous,
        // near-opaque splats, and when a capture looks like fog rather than a
        // building this is the first thing to check.
        if (data.count() > 0) {

            std::vector<float> longest;
            longest.reserve(data.count());
            for (const auto& s : data.scales) longest.push_back(std::max({s.x, s.y, s.z}));
            std::sort(longest.begin(), longest.end());

            std::vector<float> alpha(data.opacities);
            std::sort(alpha.begin(), alpha.end());

            auto pct = [](const std::vector<float>& v, double q) {
                return v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))];
            };

            std::cout << std::setprecision(4)
                      << "  longest axis  p50 " << pct(longest, 0.5)
                      << "  p99 " << pct(longest, 0.99)
                      << "  p99.9 " << pct(longest, 0.999)
                      << "  max " << longest.back()
                      << "\n  opacity       p50 " << pct(alpha, 0.5)
                      << "  p99 " << pct(alpha, 0.99)
                      << "  max " << alpha.back() << std::endl;
        }

        // Scans get the outlier cull by default; procedural clouds never do,
        // since they have no tail to cut and the point of the demo is that
        // every generated splat is drawn. See SplatData::removeOutliers for
        // the rule — it is percentile-based, so it is a no-op on a clean scan.
        if (cull) {

            const size_t before = data.count();
            const auto tCull = std::chrono::steady_clock::now();
            const size_t removed = data.removeOutliers();
            std::cout << std::setprecision(1)
                      << "  culled " << removed << " of " << before << " outliers in "
                      << std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - tCull)
                                 .count()
                      << " ms (--no-cull to keep them)" << std::endl;
        }

        // After the cull, not before: the reorder then pays for the splats that
        // survive rather than for the ones about to be dropped. Order between
        // the two is otherwise free — the cull compacts survivors in place and
        // preserves their relative order, so it cannot undo a Morton pass.
        //
        // Procedural clouds are left in generated order on purpose: they are
        // 2700 splats of test fixture, the locality win is nothing there, and
        // file order is what the existing captures were taken in.
        if (morton) {

            const auto tMorton = std::chrono::steady_clock::now();
            data.reorderMorton();
            std::cout << std::setprecision(1)
                      << "  Morton-reordered " << data.count() << " splats in "
                      << std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - tMorton)
                                 .count()
                      << " ms (--no-morton to keep file order)" << std::endl;
        }

    } else {

        data = proceduralScene();
        std::cout << "procedural scene: " << data.count() << " splats, SH degree "
                  << data.shDegree << std::endl;
    }

    // Framing uses 1 sigma, not the renderer's 3: the 3-sigma sphere is the
    // conservative bound the culler needs, and using it here backs the camera
    // off far enough that the cloud sits in the middle of the frame like a coin.
    //
    // Real scans need more than the extremes, though. Photogrammetry leaves
    // stray splats hundreds of metres out, and a bounding sphere that honours
    // them frames the outliers instead of the building. The framing radius is
    // therefore a percentile of the distance from the median centre, which
    // ignores a sparse halo without any explicit outlier pass.
    Sphere fit;
    if (data.count() > 0) {

        // Component-wise median, not the bounding box centre. A scan's extremes
        // are exactly the splats that should not be trusted: this asset's box
        // is 2000 units across while the building inside it is a fraction of
        // that, so the box centre points at empty air between two outliers.
        auto median = [](std::vector<float> v) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        };
        std::vector<float> xs, ys, zs;
        xs.reserve(data.count());
        ys.reserve(data.count());
        zs.reserve(data.count());
        for (const auto& m : data.means) {
            xs.push_back(m.x);
            ys.push_back(m.y);
            zs.push_back(m.z);
        }
        fit.center.set(median(xs), median(ys), median(zs));

        std::vector<float> radii;
        radii.reserve(data.count());
        for (const auto& m : data.means) radii.push_back(m.distanceTo(fit.center));
        std::sort(radii.begin(), radii.end());

        auto pctR = [&](double q) { return radii[static_cast<size_t>(q * static_cast<double>(radii.size() - 1))]; };

        // 90th percentile: tight enough to fill the frame with the subject,
        // loose enough not to sit inside it.
        fit.radius = pctR(0.90);
        if (fit.radius <= 0.f) fit.radius = 1.f;

        std::cout << "  median centre " << fit.center
                  << "\n  radius about it  p50 " << pctR(0.5) << "  p90 " << pctR(0.9)
                  << "  p99 " << pctR(0.99) << "  max " << radii.back()
                  << "\n  framing radius " << fit.radius << std::endl;

    } else {
        fit.center.set(0, 0, 0);
        fit.radius = 1.f;
    }

    // vsync off whenever we are capturing or benchmarking: both report frame
    // times, and a 60 Hz clamp would report the monitor rather than the renderer.
    Canvas canvas(Canvas::Parameters()
                          .title("Gaussian splats")
                          .antialiasing(msaa)
                          .vsync(shotPath.empty() && benchFrames <= 0));

    std::unique_ptr<GLRenderer> glRenderer;
    Renderer* renderer = nullptr;
#if defined(THREEPP_WITH_VULKAN)
    std::unique_ptr<VulkanRenderer> vkRenderer;
    if (useVulkan) {
        vkRenderer = std::make_unique<VulkanRenderer>(canvas);
        // Tone mapping OFF, exposure 1: the splat pass sRGB-DECODES its colour
        // into linear sceneHdr and PostComposite sRGB-ENCODES on the way out,
        // so with no curve between them the round trip is the identity and this
        // capture is directly comparable to the GL one. Turn a curve on and the
        // splats tone-map with the scene — which is the point of compositing
        // pre-post, but it is not a GL comparison any more.
        vkRenderer->toneMapping = ToneMapping::None;
        vkRenderer->toneMappingExposure = 1.f;
        // --water is a beauty/interaction shot, not a GL comparison: the HDR
        // sky it loads would clip hard through the identity curve, so it takes
        // the ocean examples' ACES + 0.7 instead and gives up GL parity.
        if (water) {
            vkRenderer->toneMapping = ToneMapping::ACESFilmic;
            vkRenderer->toneMappingExposure = 0.7f;
        }
        // Render scale 1 with both upscalers off is a GL-PARITY clamp, not a
        // recommendation: the comparison has to be of rasterizers at the same
        // pixel count. It also means the default bench measures the splat pass
        // at its worst, because the pass runs at RENDER resolution — its tiles
        // come from the render extent and it composites into sceneHdr before the
        // upscale. --scale sets that render scale and --upscaler leaves the
        // upscalers enabled, so the pass can be measured the way a real app
        // would run it. Captures taken that way are no longer comparable to GL
        // ones.
        //
        // --upscaler ALONE changes nothing, and the reason is worth stating: the
        // renderer's own default scale is already 1, and DLSS picks its quality
        // mode from the ratio — at 1.0 that is DLAA, which renders at native
        // resolution and upscales nothing. Only --scale moves the pixel count.
        // Use --scale without --upscaler to price the pixels alone, with it to
        // price what an app actually ships.
        vkRenderer->setRenderScale(renderScale);
        if (!upscaler) {

            vkRenderer->setDlss(false);
            vkRenderer->setFsr(false);
        }
        if (fog) {
            // --fog: does the cloud sit IN the medium, or punch a clear hole
            // through it? deferred_shade_60_fog_volumetrics.glsl bakes fog into
            // sceneHdr DURING the shade, so anything composited afterwards gets
            // none of it unless the pass re-derives it (SplatPass does).
            // Falloff is deliberately large so the whole cloud is inside it.
            // Density scaled to the SCENE. The framing below backs off to about
            // 2.25x the fit radius, so 0.45/radius puts roughly one optical depth
            // over the camera->cloud leg whatever the scan is measured in. A fixed
            // density cannot do that: 0.012 was picked for a metres-scale building
            // and leaves the unit-radius procedural cloud at T = 0.97 — --fog then
            // renders a frame that looks right while testing nothing.
            VulkanRenderer::HeightFogSettings hf;
            hf.density = 0.45f / std::max(1e-3f, fit.radius);
            hf.baseY = -50.f;
            hf.falloff = 400.f;
            hf.noiseAmount = 0.f;
            vkRenderer->setHeightFog(hf);
            std::cout << "  height fog on (density " << hf.density << ")" << std::endl;
        }
        renderer = vkRenderer.get();
    }
#else
    if (useVulkan) {
        std::cerr << "this build has no Vulkan backend (-DTHREEPP_WITH_VULKAN=ON)" << std::endl;
        return 1;
    }
#endif
    if (!renderer) {
        glRenderer = std::make_unique<GLRenderer>(canvas);
        glRenderer->checkShaderErrors = true;
        renderer = glRenderer.get();
    }
    std::cout << "  backend " << (useVulkan ? "Vulkan (compute tile rasterizer)"
                                            : "GL (instanced quads)")
              << std::endl;

    auto scene = Scene::create();
    scene->background = Color(0x101014);

    // Clip planes scale with the scene. A fixed far plane is fine for a
    // procedural toy and useless for a scan measured in metres, where backing
    // the camera off to frame a basilica puts the whole building past it.
    auto camera = PerspectiveCamera::create(50, canvas.aspect(),
                                            std::max(0.01f, fit.radius * 0.002f),
                                            std::max(100.f, fit.radius * 40.f));

    const auto tBuild = std::chrono::steady_clock::now();
    // --clouds K partitions the SAME splats across K SplatCloud objects, total
    // count unchanged and every splat in its original place, so the picture is
    // the same and only the number of clouds differs. That measures the
    // PER-CLOUD tax: SplatPass::record runs the whole pipeline once per cloud —
    // the clears, the sizing dispatch, 8 rounds of hist/scan/scatter and a
    // full-screen tile walk — and per-chunk LOD would submit ten of them. If
    // the slope here is steep, per-chunk LOD wants one merged buffer instead.
    std::vector<std::shared_ptr<SplatCloud>> parts;
    if (cloudSplit <= 1) {
        parts.push_back(SplatCloud::create(std::move(data)));
    } else {

        const size_t total = data.count();
        const size_t per   = (total + cloudSplit - 1) / static_cast<size_t>(cloudSplit);
        const size_t cc    = static_cast<size_t>(data.coeffCount()) * 3;
        for (size_t b = 0; b < total; b += per) {

            const size_t e = std::min(total, b + per);
            SplatData d;
            d.shDegree = data.shDegree;
            d.means.assign(data.means.begin() + b, data.means.begin() + e);
            d.scales.assign(data.scales.begin() + b, data.scales.begin() + e);
            d.rotations.assign(data.rotations.begin() + b, data.rotations.begin() + e);
            d.opacities.assign(data.opacities.begin() + b, data.opacities.begin() + e);
            d.sh.assign(data.sh.begin() + b * cc, data.sh.begin() + e * cc);
            parts.push_back(SplatCloud::create(std::move(d)));
        }
        std::cout << "  split across " << parts.size() << " clouds of ~" << per
                  << " splats (same splats, same places)" << std::endl;
    }
    auto cloud = parts.front();
    std::cout << "  cloud built (covariances + data textures) in " << std::fixed
              << std::setprecision(1)
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tBuild).count()
              << " ms" << std::endl;

    // Photogrammetry rigs (COLMAP, and the 3DGS pipelines built on it) put +Y
    // down, so a scan loads upside down in a +Y-up scene. The conventional
    // half-turn about X is the fix, and it belongs here rather than in
    // SplatLoader: which way is up is a property of the scene the cloud is
    // being placed into, not of the file. Procedural clouds are authored
    // +Y-up already and are never flipped.
    if (loadedFromFile && flip) {

        for (auto& p : parts) p->rotation.x = math::PI;
        std::cout << "  flipped 180 degrees about X for +Y-up (--no-flip to keep the file's axes)"
                  << std::endl;
    }

    // The fit sphere was measured in the cloud's own coordinates; move its
    // centre into world space so the framing below still points at the scan.
    for (auto& p : parts) p->updateMatrixWorld();
    fit.center.applyMatrix4(*cloud->matrixWorld);

    for (auto& p : parts) scene->add(p);

    for (auto& p : parts) {
        p->setPointMix(pointMix);
        p->setPointSize(pointSize);
    }
    if (pointMix > 0.f) {
        std::cout << "  point mix " << pointMix << ", point size " << pointSize << " px"
                  << std::endl;
    }

    // --water: a calm pond under the cloud, scaled from the fit sphere so a
    // unit-radius toy and a metres-scale scan both float over water instead of
    // in it. The point is the reflection: the deferred water's traced leg
    // marches the cloud's baked volume (splat_volume.glsl's svLeg), so the
    // scan must appear in the pond — a still frame answers it, which is why
    // this is a flag here rather than a scene of its own.
    //
    // Vulkan builds only: Ocean lives with the Vulkan renderer, so a GL-only
    // build has no Ocean::create to LINK against — the runtime refusal above
    // (--water needs --vulkan) cannot save the linker.
#if defined(THREEPP_WITH_VULKAN)
    if (water) {

        // An HDR sky, for two reasons: the water surface needs an environment
        // worth reflecting around the cloud, and the env-miss leg of the
        // traced reflection is what the splat march composites in front of.
        RGBELoader rgbe;
        if (auto env = rgbe.load(std::string(DATA_FOLDER) +
                                 "/textures/env/autumn_field_puresky_2k.hdr")) {
            scene->background = env;
            scene->environment = env;
        }

        // Surface just under the cloud's content. 1.25x the p90 radius keeps
        // the bulk of the splats dry (the fit radius is a percentile, so a few
        // low strays may pierce the surface — visually that reads as intended).
        const float waterY = fit.center.y - fit.radius * 1.25f;

        Ocean::Options opts;
        opts.size = std::max(20.f, fit.radius * 30.f);// pond dwarfs the subject
        opts.windSpeed = 2.5f;// calm: a readable mirror, not a seascape
        auto ocean = Ocean::create(opts);
        ocean->position.y = waterY;
        scene->add(ocean);

        // Near-black bottom well below: the analytic deep-water body dominates
        // and the surface reads as water, not as a textured floor seen through
        // glass (same rationale as vulkan_ocean_minimal's deep preset).
        auto floorMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.02f, 0.02f, 0.02f)).roughness(1.0f));
        auto floor = Mesh::create(PlaneGeometry::create(opts.size, opts.size), floorMat);
        floor->rotation.x = -math::PI / 2.f;
        floor->position.y = waterY - std::max(2.f, fit.radius * 3.f);
        scene->add(floor);

        addSun = true;// the water body and any strays need a light
        std::cout << "  pond surface at y " << waterY << " (size " << opts.size << ")"
                  << std::endl;
    }
#endif// THREEPP_WITH_VULKAN

    // --metal: the glossy-path twin of --water. A polished sphere beside the
    // cloud reflects it through deferred_shade.comp's traced reflection leg —
    // the OTHER svLeg call site — with the reflection denoiser's
    // roughness-scaled blur on top. Slightly rough on purpose: a soft
    // volumetric ghost inside an already-filtered reflection is the look the
    // design promises; a perfect mirror would only showcase its stated limit.
    if (metal) {

        if (!scene->environment) {
            // Without --water there is no sky yet, and a chrome ball in a void
            // reflects the void. Same HDR, same two reasons.
            RGBELoader rgbe;
            if (auto env = rgbe.load(std::string(DATA_FOLDER) +
                                     "/textures/env/autumn_field_puresky_2k.hdr")) {
                scene->background = env;
                scene->environment = env;
            }
        }

        auto ballMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                            .color(Color(0.95f, 0.95f, 0.95f))
                                                            .metalness(1.f)
                                                            .roughness(metalRough));
        auto ball = Mesh::create(SphereGeometry::create(fit.radius * 0.8f, 48, 32), ballMat);
        if (metalPosSet)
            ball->position.copy(metalPos);
        else
            ball->position.set(fit.center.x + fit.radius * 2.1f, fit.center.y,
                               fit.center.z);
        scene->add(ball);

        addSun = true;
        std::cout << "  polished metal sphere beside the cloud" << std::endl;
    }

    // --occluder: an opaque slab driven through the middle of the cloud. The
    // splat compositor has no hardware depth test — it reads the G-buffer depth
    // and stops accumulating behind it — so "does a box in front of the cloud
    // actually hide the part of the cloud behind it" is a real question with a
    // wrong answer available, and this is how it gets asked. Sized and placed
    // from the fit sphere so it works on a procedural toy and a scan alike.
    if (occluder) {

        auto boxMat = MeshStandardMaterial::create({{"color", Color(0xE8E4D8)},
                                                    {"roughness", 0.85f}});
        // --wireframe asks the same question of the OVERLAY path. A wireframe
        // mesh does not go through the G-buffer on Vulkan; it is drawn after
        // the temporal resolve, depth-tested against a prepass depth buffer
        // that the splat compositor never wrote to. So the half of the slab
        // BEHIND the cloud is where a wrong answer shows: it must be hidden by
        // the splats, exactly as GL's transparent-pass ordering hides it.
        boxMat->wireframe = wireOccluder;
        auto box = Mesh::create(BoxGeometry::create(fit.radius * 0.35f,
                                                    fit.radius * 2.4f,
                                                    fit.radius * 2.4f),
                                boxMat);
        box->position.copy(fit.center);
        scene->add(box);
        std::cout << (wireOccluder ? "  WIREFRAME occluder slab through the cloud centre"
                                   : "  occluder slab through the cloud centre")
                  << std::endl;
    }

    // A sun, for either of two reasons. The occluder slab needs something to be
    // lit by (a black slab proves nothing), and --sun ALONE is the fog
    // configuration with a wrong answer available: the shade gives a mesh three
    // fog in-scatter terms — scene ambient, the env top and the sun march — so a
    // scene holding only a sun zeroes the first two, and a splat pass mirroring
    // only those two has nothing left to add back against the extinction it does
    // apply. --fog --sun is therefore where a cloud fades toward black while the
    // meshes in the same air sit in a lit haze. The ambient hides exactly that,
    // so it stays with the occluder instead of coming along with the sun.
    if (occluder || addSun) {

        auto sun = DirectionalLight::create(0xffffff, 2.5f);
        sun->position.set(fit.center.x + fit.radius * 2.f,
                          fit.center.y + fit.radius * 3.f,
                          fit.center.z + fit.radius * 2.f);
        scene->add(sun);
        std::cout << "  directional sun" << std::endl;
    }
    if (occluder) scene->add(AmbientLight::create(0xffffff, 0.35f));

    // Frame the cloud: back off far enough that its bounding sphere fits the
    // vertical field of view, with a little margin. Mostly along +z with a
    // slight rise, so the three shells read side by side.
    const float distance = std::max(0.5f, fit.radius / std::tan(math::degToRad(camera->fov * 0.5f)) * 1.05f);
    Vector3 eye{fit.center.x + distance * 0.10f,
                fit.center.y + distance * 0.28f,
                fit.center.z + distance * 0.95f};
    Vector3 target = fit.center;

    if (args.camPos) eye = *args.camPos;
    if (args.camTarget) target = *args.camTarget;

    camera->position.copy(eye);
    camera->lookAt(target);

    OrbitControls controls{*camera, canvas};
    controls.target.copy(target);
    controls.update();

    bool debugNonFinite = debugNaN;
    if (debugNaN) {
        cloud->setDebugNonFinite(true);
        std::cout << "  non-finite debug ON: splats with a corrupt SH coefficient paint magenta"
                  << std::endl;
    }
    // P dissolves between Gaussians and points over a short interval rather
    // than snapping: the mix is continuous by construction, and the sweep is
    // what shows that the two looks are the same splats.
    constexpr double kPointDissolveSeconds = 0.6;
    float pointTarget = pointMix;
    auto pointChangedAt = std::chrono::steady_clock::now();
    float pointFrom = pointMix;
    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
        if (evt.key == Key::D) {
            debugNonFinite = !debugNonFinite;
            cloud->setDebugNonFinite(debugNonFinite);
            std::cout << "non-finite debug " << (debugNonFinite ? "on (magenta)" : "off") << std::endl;
        } else if (evt.key == Key::P) {
            pointFrom = cloud->pointMix();
            pointTarget = pointTarget > 0.5f ? 0.f : 1.f;
            pointChangedAt = std::chrono::steady_clock::now();
            std::cout << "point mix -> " << pointTarget << std::endl;
        }
    });
    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    // Per-frame wall time for the capture path. The first few frames pay for
    // shader compilation and the initial texture upload, so they are recorded
    // and then dropped rather than averaged into the result.
    constexpr int WARMUP_FRAMES = 5;
    const bool benching = benchFrames > 0;
    const int measuredFrames = benching ? benchFrames : shotFrames;

    std::vector<double> frameMs;
    frameMs.reserve(static_cast<size_t>(std::max(measuredFrames, 1)));
    auto frameStart = std::chrono::steady_clock::now();

    // A benchmark has to move the camera. update() early-outs when the
    // modelView is unchanged, so a static camera measures the draw alone and
    // silently skips both the sort and the instanceColor upload -- which are
    // the two things a 5M-splat cloud is actually being asked about here.
    const float orbitRadians = benching ? (0.35f / static_cast<float>(std::max(1, benchFrames))) : 0.f;

    // Printed once, after the first render: the bake runs inside the first
    // frame's cloud upload, so the number does not exist before it. The same
    // cloud under the same knobs bakes the same volume everywhere (the bake is
    // deterministic), so this is also the editor's number for this file.
    bool volPrinted = false;

    canvas.animate([&] {
        if (cloud->pointMix() != pointTarget) {
            const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - pointChangedAt).count() / kPointDissolveSeconds;
            const float mix = t >= 1.0 ? pointTarget
                                       : pointFrom + (pointTarget - pointFrom) * static_cast<float>(t);
            for (auto& p : parts) p->setPointMix(mix);
        }

        if (benching) {

            const float c = std::cos(orbitRadians), s = std::sin(orbitRadians);
            const Vector3 d{camera->position.x - target.x, camera->position.y - target.y,
                            camera->position.z - target.z};
            camera->position.set(target.x + d.x * c - d.z * s, camera->position.y,
                                 target.z + d.x * s + d.z * c);
            camera->lookAt(target);
        }

        // ── dynamic LOD selection ───────────────────────────────────────────
        // The shared policy (threepp/splats/SplatLod.hpp): coarsest level that
        // still covers the cloud's projected footprint at ~1 splat/pixel (so
        // close-up saturates to level 0 by construction), hysteresis against
        // threshold flapping, and a frustum test per chunk — which is all the
        // chunk culling there is. Same code the editor runs.
        if (!lodTable.empty())
            splats::selectLod(*cloud, lodTable, *camera, canvas.size().height());

        // Before render(), not inside onBeforeRender: the renderer uploads the
        // sorted-index attribute while it builds the render list, which is
        // earlier than the hook, so the hook's sort would show up a frame late.
        // GL only. The Vulkan backend sorts on the GPU per tile, so the CPU
        // counting sort and the instanceColor upload it drives are pure waste
        // there — and worse than waste at 216k splats, since they would dominate
        // the frame the pass is supposed to be measured on.
        if (!useVulkan)
            for (auto& p : parts) p->update(*camera);

        renderer->render(*scene, *camera);

#if defined(THREEPP_WITH_VULKAN)
        if (vkRenderer && !volPrinted) {
            if (const auto bytes = vkRenderer->splatVolumeBytes(); bytes > 0) {
                std::cout << std::fixed << std::setprecision(1)
                          << "  reflection volume " << bytes << " bytes ("
                          << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MB)"
                          << std::endl;
                volPrinted = true;
            }
        }
#endif

        if (shotPath.empty() && !benching) return;

        const auto now = std::chrono::steady_clock::now();
        frameMs.push_back(std::chrono::duration<double, std::milli>(now - frameStart).count());
        frameStart = now;

        if (++shotFrame >= measuredFrames) {

            if (!shotPath.empty()) {

                renderer->writeFramebuffer(shotPath);
                std::cout << "wrote " << shotPath << std::endl;
            }

            if (frameMs.size() > WARMUP_FRAMES) {

                std::vector<double> timed(frameMs.begin() + WARMUP_FRAMES, frameMs.end());
                std::sort(timed.begin(), timed.end());
                double sum = 0;
                for (double v : timed) sum += v;

                std::cout << std::fixed << std::setprecision(2)
                          << "frames " << timed.size() << " (after " << WARMUP_FRAMES << " warmup)"
                          << "  mean " << sum / static_cast<double>(timed.size()) << " ms"
                          << "  median " << timed[timed.size() / 2] << " ms"
                          << "  min " << timed.front() << " ms"
                          << "  max " << timed.back() << " ms"
                          << "  (" << 1000.0 * static_cast<double>(timed.size()) / sum << " fps mean)"
                          << std::endl;
#if defined(THREEPP_WITH_VULKAN)
                // GPU-side breakdown, so "the splat pass costs X" is a
                // measurement of the pass and not of the whole frame around it.
                if (vkRenderer) {
                    const auto t = vkRenderer->lastFrameTimings();
                    std::cout << std::setprecision(3)
                              << "  gpu  splat " << t.splatMs
                              << "  shade " << t.pathTraceMs
                              << "  gbuf " << t.rasterGbufMs
                              << "  taa " << t.taaMs
                              << "  denoise " << t.denoiseMs
                              << " ms   cpu record " << t.cpuRecordMs << " ms" << std::endl;
                    // Inside the splat pass: which stage owns the frame decides
                    // where an optimisation belongs. project = per-splat work,
                    // sort = the radix passes, raster = tile ranges + composite.
                    std::cout << "  gpu  splat: project " << t.splatProjectMs
                              << "  sort " << t.splatSortMs
                              << "  raster " << t.splatRasterMs << " ms" << std::endl;
                }
#endif
            }
            std::exit(0);
        }
    });
}
