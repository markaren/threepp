// VulkanSplatVolume_test — a Gaussian-splat cloud has to show up in the water.
//
// The feature under test is plans/splat-volume-reflections.md: SplatPass bakes
// every resident SplatCloud once, at upload, into an rgba16f medium (rgb =
// linear radiance, a = sigma_t), and the deferred shade's water and glass
// reflection legs MARCH that medium (svLeg, splat_volume.glsl) instead of
// seeing nothing where the cloud is. VulkanSplat_test already asserts the bake
// itself — extent, occupancy, byte-identical determinism, eviction. This
// program asserts the two claims that only a rendered frame can settle, the
// plan's test-ladder items 3 and 4:
//
//   3. PRESENCE. The cloud appears in the pond below it.
//   4. NO DOUBLE COUNT. With no reflective surface in the scene, the same
//      cloud's live volume changes nothing at all — the primary leg never
//      marches it, because SplatPass already composites the real thing there.
//
// ── GRAZING, NOT OVERHEAD ────────────────────────────────────────────────────
// The camera sits low and looks nearly along the water. That is not framing
// taste, it is the whole economy of the test: Schlick over water gives F ≈ 0.02
// at normal incidence, so from ABOVE the reflection is invisible however
// correct the march is — the band would move by less than its own control
// spread and this program would fail for a reason that is not a bug. This eye
// sits 0.85 m up and the reflected leg leaves the surface at ~16 degrees, where
// F ≈ 0.2 and the cloud is unmistakable (the numbers the assertions print are
// an R−B swing of ~+230 against a control spread under 1).
// VulkanParticleField_test's fire-in-the-mirror block chose its pose the same
// way and says so: "Fresnel is the whole economy of a reflection, and a steep
// view returns ~2% of it."
//
// ── WHY REGION STATISTICS AND NOT A STORED GOLDEN ────────────────────────────
// A scene containing an Ocean advances its FFT surface on the frame clock, so
// two reads of a "static" water scene are two different sets of ripples unless
// that clock is held. This test holds it (setSimTime at one constant for the
// whole pond A/B): the hidden and the shown legs then see the SAME ripples, so
// the only thing that can move the water band between them is the reflected
// leg, and the same-scene control below measures only what the temporal
// filters do to a still frame. On the wall clock the control swung with the
// frame rate (0.03 to 0.81 in luma between runs on one machine) and a 1.5 luma
// effect could not reliably clear it. The control stays, following the house
// convention that block established — measure a SAME-SCENE CONTROL first, then
// require the effect to beat it — rather than committing a reference PPM.
//
// ── THE A/B ──────────────────────────────────────────────────────────────────
// Toggling the cloud's `visible` is EXACTLY the feature's own gate for the
// water pixels, not an approximation of it: volumeEntries() is built from the
// visible list, so a hidden cloud leaves counts.x == 0, svLeg returns 1.0 /
// vec3(0) and the shade runs its pre-change arithmetic. The measured regions
// are water the cloud's raster footprint never touches, so the only channel by
// which showing the cloud can move them is the reflected leg.
//
// The env-var A/B is asserted too, by running this same program under
// THREEPP_VK_SPLATVOL_OFF=1 (CTest registers both): there the effect must
// VANISH into the control spread. Same code, inverted expectation — which is
// what makes it a claim about svLeg and not about "the image changed".
//
// Run standalone (a plain exit-code program, not Catch2):
//   VulkanSplatVolume_test                     assert the reflection is there
//   THREEPP_VK_SPLATVOL_OFF=1 ...              assert it is NOT
//   VulkanSplatVolume_test --shot <dir>        also dump the frames as PNG
//   VulkanSplatVolume_test --emit <file.ppm>   internal: the child half of the
//                                              no-double-count A/B below
// Exits 42 (CTest SKIP_RETURN_CODE) without a Vulkan/RT GPU.

#include "threepp/threepp.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Ocean.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "capture_util.hpp"// examples/libs (shared via target include dir)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>// _getpid, and _putenv_s for the child process below
#else
#include <unistd.h>
#endif

using namespace threepp;
namespace fs = std::filesystem;

namespace {

    constexpr int kW = 512, kH = 384;
    constexpr int kSkipCode = 42;

    int selfPid() {
#ifdef _WIN32
        return _getpid();
#else
        return ::getpid();
#endif
    }

    int failures = 0;
    void report(bool ok, const std::string& what) {
        std::printf("%s  %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    // Raw-RGB PPM, the byte-exact round trip VulkanGolden_test/VulkanSplat_test
    // both use — no codec, no flip, no channel ambiguity.
    void writePPM(const fs::path& p, const std::vector<unsigned char>& rgb, int w, int h) {
        if (!p.parent_path().empty()) fs::create_directories(p.parent_path());
        std::ofstream f(p, std::ios::binary);
        f << "P6\n" << w << " " << h << "\n255\n";
        f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    }
    std::vector<unsigned char> readPPM(const fs::path& p, int& w, int& h) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return {};
        std::string magic;
        f >> magic;
        if (magic != "P6") return {};
        int maxv = 0;
        f >> w >> h >> maxv;
        f.get();
        std::vector<unsigned char> d(static_cast<size_t>(w) * h * 3);
        f.read(reinterpret_cast<char*>(d.data()), static_cast<std::streamsize>(d.size()));
        return f ? d : std::vector<unsigned char>{};
    }

    // The cloud: ONE strong hue, unlike VulkanSplat_test's three-band cloud.
    // The bake collapses a cloud to a mean linear radiance per voxel, so a
    // single dominant colour is what makes "the band's hue moved TOWARD the
    // cloud's" a statistic and not a hope. Generated, never loaded — the repo
    // needs no asset, and SplatGenerator's xorshift is fixed-point so the same
    // seed gives the same cloud on every platform.
    SplatData makeCloud() {
        SplatGenerator::Options o;
        o.count = 12000;
        o.seed = 7311u;
        o.shDegree = 0;// the bake keeps the DC term only; nothing here needs more
        o.extent.set(2.0f, 1.5f, 2.4f);
        o.minScale = 0.05f;
        o.maxScale = 0.13f;
        o.anisotropy = 2.f;
        o.minOpacity = 0.55f;
        o.maxOpacity = 0.98f;
        auto data = SplatGenerator::generate(o);
        for (size_t i = 0; i < data.count(); ++i)
            data.setDcColor(i, Vector3{0.97f, 0.32f, 0.06f});// hot orange
        return data;
    }

    // Mean luma, mean (R−B) and mean B over a screen rectangle given in
    // FRACTIONS of the frame, so the bands survive a resolution change.
    struct RegionStat {
        double lum = 0, rb = 0, b = 0;
    };
    RegionStat region(const std::vector<unsigned char>& px,
                      double fx0, double fx1, double fy0, double fy1) {
        const int x0 = int(fx0 * kW), x1 = int(fx1 * kW);
        const int y0 = int(fy0 * kH), y1 = int(fy1 * kH);
        RegionStat r;
        std::size_t n = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const std::size_t i = (static_cast<std::size_t>(y) * kW + x) * 3;
                if (i + 2 >= px.size()) continue;
                const double R = px[i], G = px[i + 1], B = px[i + 2];
                r.lum += 0.2126 * R + 0.7152 * G + 0.0722 * B;
                r.rb += R - B;
                r.b += B;
                ++n;
            }
        const double d = double(std::max<std::size_t>(n, 1));
        r.lum /= d;
        r.rb /= d;
        r.b /= d;
        return r;
    }

    // ── The bands ───────────────────────────────────────────────────────────
    // Derived from the pose below and confirmed against a capture. The cloud's
    // centre sits at (0, 1.6, −2.5) and the eye at (0, 0.85, 6.0), so the
    // reflected leg from water around z ≈ 1.6 … 3.8 is what reaches the virtual
    // cloud at y = −1.6 — rows 0.70 … 0.90 of the frame, columns 0.44 … 0.56.
    // The cloud's OWN raster footprint is rows ~0.28 … 0.44, well clear: what
    // lands in the mirror band arrived along the reflected leg, and could not
    // have arrived any other way.
    constexpr double kMirX0 = 0.44, kMirX1 = 0.56;
    constexpr double kBandY0 = 0.70, kBandY1 = 0.90;
    // Water at the same depth range, well off the reflection's axis: the
    // control that separates a MIRROR from a glow. A term that simply added
    // radiance — or a fog leak on the camera leg — would move both together.
    constexpr double kSideAX0 = 0.06, kSideAX1 = 0.22;
    constexpr double kSideBX0 = 0.78, kSideBX1 = 0.94;
    // The cloud itself, for the sanity check that it is actually on screen.
    constexpr double kCloudX0 = 0.38, kCloudX1 = 0.62;
    constexpr double kCloudY0 = 0.28, kCloudY1 = 0.44;

    double median(std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }
    double mean(const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = 0;
        for (double x : v) s += x;
        return s / double(v.size());
    }
    double stddev(const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        const double m = mean(v);
        double s = 0;
        for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / double(v.size() - 1));
    }

    // ── The reflector-free scene, in ONE place ──────────────────────────────
    // Both halves of the no-double-count A/B render it: this process, and the
    // THREEPP_VK_SPLATVOL_OFF child this process spawns. Any drift between the
    // two setups would look exactly like the bug the comparison is hunting, so
    // there is only one setup.
    constexpr int kFlatFrames = 140;

    void configureFlat(VulkanRenderer& r) {
        r.toneMapping = ToneMapping::None;
        r.toneMappingExposure = 1.f;
        r.setRenderScale(1.f);
        r.setDlss(false);
        r.setFsr(false);
        r.setDenoise(true);
        r.setRestirDIEnabled(true);
        r.setFireflyClamp(6.f);
        r.setClearColor(Color(0.06f, 0.06f, 0.08f));
        // Unjittered raster (MSAA rasterizes without the Halton offset by
        // design), which is what makes the byte claim below a BYTE claim:
        // VulkanSplat_test's determinism check rests on the same switch, and
        // with jitter on the last frames' TAA convergence moves by an LSB or two.
        r.setGbufferMsaa(2);
    }

    struct FlatScene {
        std::shared_ptr<Scene> scene;
        std::shared_ptr<PerspectiveCamera> camera;
        std::shared_ptr<SplatCloud> cloud;
    };
    FlatScene makeFlatScene() {
        FlatScene f;
        f.scene = Scene::create();
        f.scene->background = Color(0x0f0f14);// no env, no water, no glass: nothing traces a reflection leg
        f.camera = PerspectiveCamera::create(50.f, float(kW) / kH, 0.05f, 100.f);
        f.camera->position.set(0.f, 1.2f, 6.5f);
        f.camera->lookAt(Vector3{0.f, 0.f, 0.f});
        f.cloud = SplatCloud::create(makeCloud());
        f.scene->add(f.cloud);
        return f;
    }

}// namespace

int main(int argc, char** argv) {

    std::string emitPath, shotDir;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--emit") == 0 && i + 1 < argc) emitPath = argv[++i];
        else if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotDir = argv[++i];
    }
    // The knobs, read here for reporting only — the renderer reads them itself
    // (SplatPass's constructor and updateSplatVolumeUbo's static), once per run.
    const bool volOff = [] {
        const char* e = std::getenv("THREEPP_VK_SPLATVOL_OFF");
        return e && *e && *e != '0';
    }();
    const char* sigmaEnv = std::getenv("THREEPP_VK_SPLATVOL_SIGMA");

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanSplatVolume_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    VulkanRenderer& renderer = *rendererPtr;

    std::printf("mode: THREEPP_VK_SPLATVOL_OFF=%s  THREEPP_VK_SPLATVOL_SIGMA=%s\n",
                volOff ? "1" : "(unset)", sigmaEnv && *sigmaEnv ? sigmaEnv : "(unset)");

    // ── The child half of the no-double-count A/B ───────────────────────────
    // Renders the reflector-free scene and writes it, nothing else. Kept in
    // THIS program (rather than a second executable) so the two halves can
    // never drift apart in scene, pose or frame count.
    if (!emitPath.empty()) {
        configureFlat(renderer);
        auto f = makeFlatScene();
        for (int i = 0; i < kFlatFrames; ++i) renderer.render(*f.scene, *f.camera);
        writePPM(emitPath, renderer.readRGBPixels(), kW, kH);
        std::printf("[emit] %s  volumeBytes %llu\n", emitPath.c_str(),
                    static_cast<unsigned long long>(renderer.splatVolumeBytes()));
        return 0;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  1. PRESENCE — the cloud in the pond (plan, test-ladder item 3)
    // ════════════════════════════════════════════════════════════════════════
    renderer.setDenoise(true);
    renderer.setRestirDIEnabled(true);
    renderer.setFireflyClamp(6.0f);
    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 0.7f;// the ocean examples' pairing with an HDR sky

    auto scene = Scene::create();

    // An HDR sky for two reasons: the surface needs an environment worth
    // reflecting AROUND the cloud, and the env-miss leg of the traced
    // reflection is what the splat march composites in front of. Without it
    // "the band got darker" would have nothing to get darker THAN.
    RGBELoader rgbe;
    if (auto env = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene->background = env;
        scene->environment = env;
    } else {
        std::printf("[warn] env HDR missing; the sky is a flat colour\n");
        scene->background = Color(0.35f, 0.45f, 0.62f);
    }

    auto sun = DirectionalLight::create(Color(1.0f, 0.95f, 0.85f), 2.0f);
    sun->position.set(2.f, 3.f, 2.f);
    Object3D sunTarget;
    scene->add(sun);
    sun->setTarget(sunTarget);

    // The pond. The deferred water path is gated on the FFT-displaced ocean
    // marker, which only a DisplacedMesh carries — a plain Mesh wearing a
    // watery material can never reach shadeWater. Under 100 m Ocean resolves
    // its own POND recipe, so nothing here authors water parameters; the calm
    // wind is the only deviation, and it buys a readable mirror.
    Ocean::Options oo;
    oo.size = 60.f;
    oo.resolution = 256;
    oo.windSpeed = 2.0f;
    oo.choppiness = 0.30f;
    oo.waveScale = 0.45f;
    auto pond = Ocean::create(oo);
    scene->add(pond);

    // Near-black bottom well below: the analytic deep-water body dominates and
    // the surface reads as water rather than as a floor seen through glass
    // (vulkan_ocean_minimal's deep preset, same rationale).
    auto bed = Mesh::create(PlaneGeometry::create(oo.size, oo.size),
                            MeshStandardMaterial::create(
                                    MeshStandardMaterial::Params{}.color(Color(0.02f, 0.02f, 0.02f)).roughness(1.f)));
    bed->rotation.x = -math::PI / 2.f;
    bed->position.y = -4.f;
    scene->add(bed);

    auto cloud = SplatCloud::create(makeCloud());
    cloud->position.set(0.f, 1.6f, -2.5f);
    scene->add(cloud);

    // Low and grazing — see the header. 0.85 m over the water, looking almost
    // along it: the reflected leg leaves the surface at ~16 degrees, where
    // Fresnel returns ~0.2 instead of the ~0.02 an overhead pose returns.
    auto camera = PerspectiveCamera::create(45.f, float(kW) / kH, 0.05f, 400.f);
    camera->position.set(0.f, 0.85f, 6.0f);
    camera->lookAt(Vector3{0.f, 0.6f, -2.5f});
    camera->updateMatrixWorld();

    // The frame clock is held at one instant for the whole pond A/B, so the
    // FFT surface is the same set of ripples in the hidden legs, the control
    // and the shown leg, on every run; see the header. (A constant is a valid
    // non-decreasing clock; the temporal blends see dt = 0 and keep their
    // reference-rate weights.)
    constexpr double kPondTimeSec = 2.0;
    const auto draw = [&] {
        renderer.setSimTime(kPondTimeSec);
        renderer.render(*scene, *camera);
    };
    const auto settle = [&](int n) { for (int i = 0; i < n; ++i) draw(); };
    const auto shot = [&](const char* name) {
        if (shotDir.empty()) return;
        fs::create_directories(shotDir);
        renderer.writeFramebuffer(fs::path(shotDir) / (std::string(name) + ".png"));
    };

    // Upload and bake FIRST, then hide. A cloud that was never visible was
    // never uploaded, so hiding it up front would make the A/B an upload A/B;
    // hiding it after it is resident parks it, which is the flag-only A/B this
    // test wants (and VulkanSplat_test's "parked, not evicted" check pins).
    settle(25);

    // Leg 0a/0b — the cloud HIDDEN, read twice. The second read is the
    // CONTROL: whatever these two differ by is what an unchanged scene does to
    // itself on this backend with a live FFT surface under it. Every claim
    // below is stated against that number rather than against zero.
    cloud->visible = false;
    settle(90);
    const auto off1px = renderer.readRGBPixels();
    const auto off1 = region(off1px, kMirX0, kMirX1, kBandY0, kBandY1);
    settle(40);
    const auto off2px = renderer.readRGBPixels();
    const auto off2 = region(off2px, kMirX0, kMirX1, kBandY0, kBandY1);
    const auto off2SideA = region(off2px, kSideAX0, kSideAX1, kBandY0, kBandY1);
    const auto off2SideB = region(off2px, kSideBX0, kSideBX1, kBandY0, kBandY1);
    shot("10_hidden");

    const double ctrlLum = std::fabs(off2.lum - off1.lum);
    const double ctrlRb = std::fabs(off2.rb - off1.rb);
    const double ctrlB = std::fabs(off2.b - off1.b);

    report(renderer.splatResidentClouds() == 1,
           "a hidden cloud is still resident (parked), so the A/B is the FLAG "
           "and not an upload");

    // Leg 1 — the same frame with the cloud shown. Nothing else changes.
    cloud->visible = true;
    settle(60);
    const auto onPx = renderer.readRGBPixels();
    const auto on = region(onPx, kMirX0, kMirX1, kBandY0, kBandY1);
    const auto onSideA = region(onPx, kSideAX0, kSideAX1, kBandY0, kBandY1);
    const auto onSideB = region(onPx, kSideBX0, kSideBX1, kBandY0, kBandY1);
    const auto onCloud = region(onPx, kCloudX0, kCloudX1, kCloudY0, kCloudY1);
    const auto offCloud = region(off2px, kCloudX0, kCloudX1, kCloudY0, kCloudY1);
    shot("11_shown");

    const double dLum = on.lum - off2.lum;
    const double dRb = on.rb - off2.rb;
    const double dB = on.b - off2.b;
    const double dSideLum = 0.5 * ((onSideA.lum - off2SideA.lum) + (onSideB.lum - off2SideB.lum));
    const double dSideRb = 0.5 * ((onSideA.rb - off2SideA.rb) + (onSideB.rb - off2SideB.rb));

    std::printf("       mirror band  lum %.2f -> %.2f (%+.2f)   R-B %.2f -> %.2f (%+.2f)   B %.2f -> %.2f (%+.2f)\n",
                off2.lum, on.lum, dLum, off2.rb, on.rb, dRb, off2.b, on.b, dB);
    std::printf("       side  bands  dLum %+.2f  dR-B %+.2f      control spread  lum %.2f  R-B %.2f  B %.2f\n",
                dSideLum, dSideRb, ctrlLum, ctrlRb, ctrlB);
    std::printf("       cloud  band  lum %.2f -> %.2f            R-B %.2f -> %.2f\n",
                offCloud.lum, onCloud.lum, offCloud.rb, onCloud.rb);
    std::printf("       volume bytes %llu  gen %llu  residentClouds %zu\n",
                static_cast<unsigned long long>(renderer.splatVolumeBytes()),
                static_cast<unsigned long long>(renderer.splatVolumeGeneration()),
                renderer.splatResidentClouds());

    // The pose sanity gate. If the cloud were off screen, or the pond were not
    // where it is believed to be, every statistic below would be measuring
    // sky — and would pass the OFF mode for entirely the wrong reason.
    report(onCloud.rb > offCloud.rb + 10.0,
           "the cloud is on screen where the pose says it is (its own band goes "
           "orange), so the mirror band below it is water and not sky");

    if (!volOff) {
        report(renderer.splatVolumeBytes() > 0,
               "the cloud baked a reflection volume the shade can march");
        // THE FEATURE. Two independent signatures of the same event, because
        // either one alone has a cheap wrong explanation: a luma-only claim
        // could be a stray light, a hue-only claim could be tone mapping.
        report(dRb > 6.0 && dRb > 4.0 * std::max(ctrlRb, 0.25) && dRb > 4.0 * std::fabs(dSideRb),
               "the water band's HUE MOVED TOWARD THE CLOUD'S: R-B rises far "
               "past the same-scene control AND far past water off to the side, "
               "which a uniform glow or a camera-leg leak could not do");
        report(dB < -2.0 && std::fabs(dB) > 3.0 * std::max(ctrlB, 0.25),
               "and its TRANSMITTANCE DROPPED: the blue the sky was putting "
               "into the reflected leg is occluded by the marched medium");
        report(std::fabs(dLum) > 3.0 * std::max(ctrlLum, 0.25),
               "the band moved by much more than an unchanged frame of the same "
               "scene moves itself");
    } else {
        // The same measurements, the opposite claim: with the bake never run
        // and nothing bound, showing the cloud must leave the water exactly
        // where a cloud-free scene leaves it. This is what proves the effect
        // above belongs to svLeg and not to anything else the cloud does.
        report(renderer.splatVolumeBytes() == 0,
               "THREEPP_VK_SPLATVOL_OFF: no volume is baked at all");
        report(std::fabs(dRb) <= std::max(4.0 * ctrlRb, 2.0),
               "and the water band does NOT move: with no volume bound svLeg "
               "returns 1.0 / vec3(0) and the shade runs its pre-change "
               "arithmetic");
        report(std::fabs(dB) <= std::max(4.0 * ctrlB, 2.0),
               "in the blue channel either");
    }

    // The calibration knob, reported rather than gated (the plan asks for an
    // A/B lever, not authored data): a bigger sigma must darken the band
    // further, which is visible in these numbers across two runs.
    if (sigmaEnv && *sigmaEnv)
        std::printf("[info] SIGMA=%s -> mirror band dLum %+.2f  dR-B %+.2f  dB %+.2f\n",
                    sigmaEnv, dLum, dRb, dB);

    // ════════════════════════════════════════════════════════════════════════
    //  2. THE TIMING BRACKET — measured, not asserted (plan, "Costs")
    // ════════════════════════════════════════════════════════════════════════
    // The march has no bracket of its own and does not need one: it lives
    // INSIDE the deferred shade dispatch, which pathTraceMs already times end
    // to end. Same scene, same pose, cloud shown vs hidden — so the delta is
    // the svLeg marches on the water's reflected legs and nothing else (the
    // splat rasterizer's own cost is splatMs, a different field).
    {
        const auto sample = [&](int n) {
            std::vector<double> v;
            v.reserve(size_t(n));
            for (int i = 0; i < n; ++i) {
                draw();
                v.push_back(renderer.lastFrameTimings().pathTraceMs);
            }
            return v;
        };
        cloud->visible = true;
        settle(20);
        const auto tOn = sample(90);
        cloud->visible = false;
        settle(20);
        const auto tOff = sample(90);
        cloud->visible = true;
        std::printf("[time] deferred shade (pathTraceMs)  cloud SHOWN  mean %.3f  median %.3f  sd %.3f\n",
                    mean(tOn), median(tOn), stddev(tOn));
        std::printf("[time] deferred shade (pathTraceMs)  cloud HIDDEN mean %.3f  median %.3f  sd %.3f\n",
                    mean(tOff), median(tOff), stddev(tOff));
        // The MEDIAN is the reading to quote: a single scheduling stall on a
        // laptop puts a 30 ms frame in the sample and drags the mean with it
        // (the sd printed above is how you spot that it happened).
        std::printf("[time] splat volume march delta      median %+.3f ms  mean %+.3f ms "
                    "(quote the median; check the sd above first)\n",
                    median(tOn) - median(tOff), mean(tOn) - mean(tOff));
    }

    // ════════════════════════════════════════════════════════════════════════
    //  3. NO DOUBLE COUNT (plan, test-ladder item 4)
    // ════════════════════════════════════════════════════════════════════════
    // "The primary camera leg must never march it" is the one wrong turn the
    // design makes easy, and the plan states it twice. Two checks on one
    // reflector-free scene (makeFlatScene: solid background, no water, no
    // glass, so nothing in the frame traces a reflection leg at all), because
    // neither check is the whole claim on its own.
    //
    // The pond and the HDR sky are gone from here on, which is also what makes
    // the byte claim below possible: the FFT surface was the only thing in this
    // program that advanced on the wall clock.
    {
        scene->remove(*pond);
        scene->remove(*bed);
        scene->remove(*cloud);
        cloud.reset();

        // Spawn the OFF half FIRST, so its Vulkan device is gone again before
        // this process re-renders. Under THREEPP_VK_SPLATVOL_OFF the child
        // inherits the variable too and the comparison becomes OFF vs OFF —
        // still a real check that the OFF path renders at all, with the ON/OFF
        // pairing being the default run's job.
        // PID in the name: CTest registers this program twice (default and
        // .off) and `ctest -j` will happily run both at once.
        const fs::path out = fs::temp_directory_path() /
                             ("threepp_splatvol_ab_" + std::to_string(selfPid()) + ".ppm");
        std::error_code ec;
        fs::remove(out, ec);
#ifdef _WIN32
        _putenv_s("THREEPP_VK_SPLATVOL_OFF", "1");
#else
        setenv("THREEPP_VK_SPLATVOL_OFF", "1", 1);
#endif
        // cmd.exe strips ONE layer of outer quotes, so the whole command line
        // is wrapped again; the inner quotes are what survive a space in either
        // path. Harmless on a POSIX shell.
        const std::string cmd = "\"\"" + std::string(argv[0]) + "\" --emit \"" + out.string() + "\"\"";
        const int rc = std::system(cmd.c_str());
#ifdef _WIN32
        _putenv_s("THREEPP_VK_SPLATVOL_OFF", "");
#else
        unsetenv("THREEPP_VK_SPLATVOL_OFF");
#endif

        configureFlat(renderer);
        auto flat = makeFlatScene();
        const auto drawFlat = [&](int n) {
            for (int i = 0; i < n; ++i) renderer.render(*flat.scene, *flat.camera);
        };

        // (a) THE ENV-VAR A/B, on the scene the plan names: a cloud FULLY
        //     VISIBLE on screen, no reflector anywhere. Not a byte claim, and
        //     deliberately so — across two PROCESSES the TAA resolve's blend is
        //     frame-rate aware (VulkanSplat_test's golden comment), so when a
        //     frame lands moves the last LSBs. It does not have to be: a
        //     primary-leg march would add the cloud's own marched radiance on
        //     top of the raster composite, which is orders of magnitude outside
        //     the goldens' gate.
        drawFlat(kFlatFrames);
        const auto mine = renderer.readRGBPixels();
        const std::uint64_t liveBytes = renderer.splatVolumeBytes();
        shot("20_noreflector");

        int cw = 0, ch = 0;
        const auto childPx = readPPM(out, cw, ch);
        if (rc != 0 || childPx.empty() || cw != kW || ch != kH) {
            report(false, "the THREEPP_VK_SPLATVOL_OFF child rendered the "
                          "reflector-free scene (rc/size)");
            std::printf("       rc %d  file %s  %dx%d\n", rc, out.string().c_str(), cw, ch);
        } else {
            const capture::DiffResult d = capture::imageDiff(mine, childPx);
            std::printf("       cloud on screen, no reflector: this run vs a "
                        "THREEPP_VK_SPLATVOL_OFF child -> PSNR %.1f dB  maxD %d  hot %.3f%%  "
                        "(volumeBytes %llu vs 0)\n",
                        d.psnr, d.maxD, d.hotPct, static_cast<unsigned long long>(liveBytes));
            report(d.psnr >= 40.0 && d.maxD <= 32,
                   "a fully visible cloud renders the same with the volume "
                   "feature live as with it compiled out - the goldens' own "
                   "gate, on the scene the plan names");
        }
        fs::remove(out, ec);

        // (b) BYTE-EXACT, inside this one process. A cloud moved BEHIND the
        //     camera is still visible() and so still in volumeEntries() — the
        //     collector's test is a visibility test, NOT a frustum test, which
        //     is the property the whole design rests on (a cloud behind the
        //     camera is exactly what has to keep showing up in the water). So
        //     the volume stays live and flags bit 12 stays set while the
        //     rasterizer contributes not one pixel; with no reflector in the
        //     scene the frame must then be IDENTICAL to the same frame with no
        //     cloud at all. Unjittered (configureFlat's MSAA), so "identical"
        //     can mean identical.
        flat.cloud->position.set(0.f, 0.5f, 14.0f);// squarely behind the eye at z = 6.5
        drawFlat(kFlatFrames);
        const auto withVolume = renderer.readRGBPixels();
        const std::uint64_t behindBytes = renderer.splatVolumeBytes();
        shot("21_behind_camera");

        flat.scene->remove(*flat.cloud);
        flat.cloud.reset();
        drawFlat(kFlatFrames);
        const auto without = renderer.readRGBPixels();
        shot("22_no_cloud");

        const capture::DiffResult d = capture::imageDiff(withVolume, without);
        std::printf("       live volume vs no volume (no reflector): PSNR %.1f dB  maxD %d  hot %.3f%%  "
                    "(volumeBytes %llu -> %llu)\n",
                    d.psnr, d.maxD, d.hotPct,
                    static_cast<unsigned long long>(behindBytes),
                    static_cast<unsigned long long>(renderer.splatVolumeBytes()));
        if (!volOff) {
            report(liveBytes > 0 && behindBytes > 0,
                   "the volume is live for both halves - including for a cloud "
                   "BEHIND the camera, which is a visibility test and not a "
                   "frustum test");
            report(renderer.splatVolumeBytes() == 0,
                   "and it is freed with the cloud");
        }
        report(d.maxD == 0,
               "a live splat volume in a scene with NO reflector changes not one "
               "byte - the primary leg does not march it");
    }

    std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
