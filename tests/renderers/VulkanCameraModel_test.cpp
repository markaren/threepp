// VulkanCameraModel_test — numeric checks for the Vulkan physical camera's
// GEOMETRY (as opposed to VulkanGolden_test, which pins its pixels).
//
// The camera model's claim is that a threepp camera configured the way you
// configure a real one — sensor size via PerspectiveCamera::filmGauge, lens via
// setFocalLength — projects like that real camera, and that
// VulkanRendererCore::cameraIntrinsics() reports the fx/fy/cx/cy an OpenCV
// calibration of it would. That is a numeric claim, so test it numerically:
// project known 3D points through the renderer's own camera and check they land
// where the reported intrinsics say they should.
//
// Run standalone (plain exit-code program, not Catch2) or via CTest. Exits 42
// (→ CTest "Skipped") when no Vulkan/RT GPU or display is available.

#include "threepp/threepp.hpp"

#include "threepp/cameras/LensDistortion.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 384, kH = 256;// 1.5 aspect — deliberately NOT 1:1, so a
                                     // width/height mix-up cannot pass
    constexpr int kSkipCode = 42;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    void checkClose(float got, float want, float tol, const std::string& what) {
        const bool ok = std::abs(got - want) <= tol;
        std::printf("  %s %s (got %.4f, want %.4f, tol %.4f)\n",
                    ok ? "[ ok ]" : "[FAIL]", what.c_str(), got, want, tol);
        if (!ok) ++failures;
    }

    // Project a camera-space point (threepp/GL convention: +X right, +Y up,
    // −Z forward) to a pixel with top-left origin, using ONLY the reported
    // intrinsics in OpenCV's convention (+X right, +Y down, +Z forward).
    void projectWithIntrinsics(const VulkanRendererCore::CameraIntrinsics& k,
                               const Vector3& camSpace, float& u, float& v) {
        const float zCv = -camSpace.z;// OpenCV looks down +Z
        u = k.fx * (camSpace.x / zCv) + k.cx;
        v = k.fy * (-camSpace.y / zCv) + k.cy;// OpenCV's +Y is down
    }

    // The same point projected by the ACTUAL projection matrix the renderer
    // uploaded, taken through NDC to the same pixel convention. This is the
    // reference the intrinsics have to agree with.
    void projectWithMatrix(const PerspectiveCamera& cam, const Vector3& camSpace,
                           float& u, float& v) {
        Vector4 clip(camSpace.x, camSpace.y, camSpace.z, 1.f);
        clip.applyMatrix4(cam.projectionMatrix);
        const float xNdc = clip.x / clip.w;
        const float yNdc = clip.y / clip.w;
        u = (xNdc * 0.5f + 0.5f) * static_cast<float>(kW);
        v = (0.5f - yNdc * 0.5f) * static_cast<float>(kH);
    }

}// namespace

int main() {
    // Unbuffered: this test drives a GPU, and a device-lost or a crash would
    // otherwise discard every check already printed, leaving no clue as to how
    // far it got.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanCameraModel_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    VulkanRenderer& renderer = *rendererPtr;
    renderer.setRenderScale(1.0f);// intrinsics are reported in RENDER-extent
                                  // pixels; keep that equal to the framebuffer
                                  // so the expected values stay hand-checkable

    Scene scene;
    scene.background = Color(0.05f, 0.05f, 0.08f);

    // ── Film gauge / focal length round-trip (pure camera math, no GPU) ──────
    // A 1/2.3" sensor (6.3 mm wide) with a 4.8 mm lens — a typical small robot
    // camera, and nothing like the 35 mm full frame the DoF path used to assume.
    PerspectiveCamera camera(60.f, static_cast<float>(kW) / static_cast<float>(kH), 0.1f, 100.f);
    camera.filmGauge = 6.3f;
    camera.setFocalLength(4.8f);
    camera.updateProjectionMatrix();

    std::printf("film gauge / focal length round-trip:\n");
    checkClose(camera.getFocalLength(), 4.8f, 1e-3f, "getFocalLength round-trips setFocalLength");
    // filmGauge is the film WIDTH; the height follows from the aspect (>1 here).
    checkClose(camera.getFilmHeight(), 6.3f / (static_cast<float>(kW) / static_cast<float>(kH)), 1e-4f,
               "getFilmHeight = gauge / aspect");
    // fov = 2·atan(filmHeight / (2·focal))
    {
        const float wantFov = 2.f * std::atan(camera.getFilmHeight() / (2.f * 4.8f)) * 180.f / 3.14159265f;
        checkClose(camera.fov, wantFov, 1e-2f, "fov derived from sensor + lens");
    }

    // One frame so updateCameraUbo has stashed this camera's projection.
    renderer.render(scene, camera);

    // ── Reported intrinsics ────────────────────────────────────────────────
    const auto k = renderer.cameraIntrinsics();
    std::printf("intrinsics: fx=%.3f fy=%.3f cx=%.3f cy=%.3f (%ux%u)\n",
                k.fx, k.fy, k.cx, k.cy, k.width, k.height);

    check(k.width == static_cast<uint32_t>(kW) && k.height == static_cast<uint32_t>(kH),
          "intrinsics report the render extent");

    // Square pixels: the sensor and the framebuffer share an aspect, so
    // fx == fy. (Both equal (H/2)/tan(fovY/2).)
    checkClose(k.fx, k.fy, 1e-2f, "square pixels → fx == fy");

    // Closed form straight from the lens: focal in pixels = focal_mm × (pixels
    // per mm of sensor). This is the whole point of Stage 1 — the pixel focal
    // length is now traceable to a real lens on a real sensor.
    {
        const float pxPerMm = static_cast<float>(kH) / camera.getFilmHeight();
        checkClose(k.fy, 4.8f * pxPerMm, 0.05f, "fy = focal_mm x pixels-per-mm");
    }

    // Symmetric frustum → principal point dead centre.
    checkClose(k.cx, static_cast<float>(kW) * 0.5f, 1e-3f, "cx centred");
    checkClose(k.cy, static_cast<float>(kH) * 0.5f, 1e-3f, "cy centred");

    // ── Intrinsics vs. the actual projection matrix ────────────────────────
    // The real test: several off-axis points must land on the same pixel
    // whether you go through the renderer's projection or through fx/fy/cx/cy.
    std::printf("intrinsics agree with the projection matrix:\n");
    const std::vector<Vector3> pts = {
            {0.f, 0.f, -5.f},      // dead centre
            {1.f, 0.5f, -5.f},     // upper right
            {-1.2f, -0.7f, -3.f},  // lower left, nearer
            {0.4f, -1.1f, -12.f},  // lower right, far
            {-2.0f, 1.6f, -2.5f},  // wide corner-ish
    };
    for (size_t i = 0; i < pts.size(); ++i) {
        float uK, vK, uM, vM;
        projectWithIntrinsics(k, pts[i], uK, vK);
        projectWithMatrix(camera, pts[i], uM, vM);
        const float du = std::abs(uK - uM), dv = std::abs(vK - vM);
        const bool ok = du < 0.01f && dv < 0.01f;
        std::printf("  %s point %zu: intrinsics (%.3f, %.3f) vs matrix (%.3f, %.3f)\n",
                    ok ? "[ ok ]" : "[FAIL]", i, uK, vK, uM, vM);
        if (!ok) ++failures;
    }

    // ── Off-centre sensor: the principal point must follow filmOffset ───────
    // A shifted sensor (tilt-shift lens, stereo rig with a converged frustum)
    // moves the principal point off centre — carrying the projection's skew
    // terms is what keeps that honest.
    std::printf("off-centre sensor:\n");
    camera.filmOffset = 1.0f;// mm of horizontal sensor shift
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
    const auto kOff = renderer.cameraIntrinsics();
    check(std::abs(kOff.cx - static_cast<float>(kW) * 0.5f) > 1.f,
          "filmOffset shifts cx off centre");
    checkClose(kOff.cy, static_cast<float>(kH) * 0.5f, 1e-3f, "filmOffset leaves cy centred");
    {
        float uK, vK, uM, vM;
        const Vector3 p(0.6f, -0.3f, -4.f);
        projectWithIntrinsics(kOff, p, uK, vK);
        projectWithMatrix(camera, p, uM, vM);
        const bool ok = std::abs(uK - uM) < 0.01f && std::abs(vK - vM) < 0.01f;
        std::printf("  %s shifted-sensor point: intrinsics (%.3f, %.3f) vs matrix (%.3f, %.3f)\n",
                    ok ? "[ ok ]" : "[FAIL]", uK, vK, uM, vM);
        if (!ok) ++failures;
    }

    camera.filmOffset = 0.f;
    camera.updateProjectionMatrix();

    // ── Lens distortion agrees with OpenCV ─────────────────────────────────
    // The claim of the whole feature is that a lens calibrated with OpenCV
    // renders the way OpenCV says it should, so the check is against OpenCV's
    // own numbers, not against a restatement of our own formula.
    //
    // References generated once with cv2.projectPoints / cv2.fisheye.
    // projectPoints at K = I (fx = fy = 1, cx = cy = 0, so the outputs ARE
    // normalized coordinates) — baked in so the test needs no OpenCV at run
    // time. Regenerate by projecting the same points with the same
    // coefficients if these ever need to change.
    std::printf("forward map matches OpenCV:\n");
    {
        struct Ref {
            float x, y;  // ideal normalized in
            float xd, yd;// OpenCV's distorted normalized out
        };
        // distCoeffs = [k1, k2, p1, p2, k3] = [-0.28, 0.10, 0.0012, -0.0008, -0.02]
        const Ref bcRefs[] = {
            {0.0000f, 0.0000f, 0.000000000f, 0.000000000f},
            {0.3000f, 0.1500f, 0.290795145f, 0.145577572f},
            {-0.4500f, 0.3000f, -0.417651804f, 0.278629536f},
            {0.6000f, -0.5500f, 0.509647070f, -0.466867314f},
            {-0.7000f, -0.7000f, -0.562363312f, -0.560403312f},
            {0.1200f, -0.8800f, 0.098877356f, -0.728781675f},
        };
        LensDistortion bc;
        bc.model = LensModel::BrownConrady;
        bc.k1 = -0.28f; bc.k2 = 0.10f; bc.k3 = -0.02f; bc.p1 = 0.0012f; bc.p2 = -0.0008f;

        // fisheye D = [k1, k2, k3, k4] = [-0.03, 0.008, -0.001, 0.0002]
        const Ref feRefs[] = {
            {0.0000f, 0.0000f, 0.000000000f, 0.000000000f},
            {0.3000f, 0.1500f, 0.288568893f, 0.144284446f},
            {-0.4500f, 0.3000f, -0.409663755f, 0.273109170f},
            {0.6000f, -0.5500f, 0.497391454f, -0.455942166f},
            {-0.7000f, -0.7000f, -0.543236282f, -0.543236282f},
            {0.1200f, -0.8800f, 0.096775702f, -0.709688483f},
        };
        LensDistortion fe;
        fe.model = LensModel::Fisheye;
        fe.k1 = -0.03f; fe.k2 = 0.008f; fe.k3 = -0.001f; fe.k4 = 0.0002f;

        auto compare = [&](const char* name, const LensDistortion& d,
                           const Ref* refs, size_t n) {
            float worst = 0.f;
            for (size_t i = 0; i < n; ++i) {
                float xd = 0.f, yd = 0.f;
                lensDistort(d, refs[i].x, refs[i].y, xd, yd);
                worst = std::max(worst, std::max(std::abs(xd - refs[i].xd),
                                                 std::abs(yd - refs[i].yd)));
            }
            // float32 evaluation of a degree-7 polynomial against OpenCV's
            // float64; 1e-6 normalized is ~3e-4 px.
            const bool ok = worst < 1e-6f;
            std::printf("  %s %s (worst deviation from OpenCV %.2e)\n",
                        ok ? "[ ok ]" : "[FAIL]", name, worst);
            if (!ok) ++failures;
        };
        compare("cv2.projectPoints (Brown-Conrady)", bc, bcRefs, std::size(bcRefs));
        compare("cv2.fisheye.projectPoints (Kannala-Brandt)", fe, feRefs, std::size(feRefs));
    }

    // ── Lens distortion: the inverse map actually inverts ───────────────────
    // lensUndistort is iterative in both models, so "it converges" is a claim
    // that needs checking rather than assuming — and it is the map the
    // renderer actually uses, so an inverse that quietly stalled would bend
    // every frame by the wrong amount.
    std::printf("lens inverse round-trips the forward map:\n");
    {
        struct Case {
            const char* name;
            LensDistortion d;
        };
        LensDistortion bc;
        bc.model = LensModel::BrownConrady;
        bc.k1 = -0.28f; bc.k2 = 0.10f; bc.k3 = -0.02f; bc.p1 = 0.0012f; bc.p2 = -0.0008f;

        LensDistortion pin;// pincushion
        pin.model = LensModel::BrownConrady;
        pin.k1 = 0.22f; pin.k2 = 0.05f;

        LensDistortion fe;
        fe.model = LensModel::Fisheye;
        fe.k1 = -0.03f; fe.k2 = 0.008f; fe.k3 = -0.001f; fe.k4 = 0.0002f;

        const Case cases[] = {{"brown-conrady (barrel + tangential)", bc},
                              {"brown-conrady (pincushion)", pin},
                              {"fisheye (Kannala-Brandt)", fe}};
        const float samples[][2] = {{0.0f, 0.0f}, {0.3f, 0.15f}, {-0.45f, 0.30f},
                                    {0.6f, -0.55f}, {-0.7f, -0.7f}};
        for (const auto& cs : cases) {
            float worst = 0.f;
            for (const auto& s : samples) {
                float xd, yd, xr, yr;
                lensDistort(cs.d, s[0], s[1], xd, yd);
                lensUndistort(cs.d, xd, yd, xr, yr);
                worst = std::max(worst, std::max(std::abs(xr - s[0]), std::abs(yr - s[1])));
            }
            // 1e-5 normalized is ~0.003 px at these focal lengths. See
            // kLensInverseIterations for why 5 iterations (OpenCV's default)
            // could not hold this.
            const bool ok = worst < 1e-5f;
            std::printf("  %s %s (worst residual %.2e)\n", ok ? "[ ok ]" : "[FAIL]", cs.name, worst);
            if (!ok) ++failures;
        }
    }

    // ── The warp actually bends the image, in the right direction ───────────
    // A bright marker placed off-centre against black. Barrel distortion
    // (k1 < 0) maps an ideal radius r to r·(1 + k1·r²) < r, so an off-axis
    // POINT lands closer to the centre — the familiar outward bowing of
    // straight lines is the same effect seen along an edge. Pincushion
    // (k1 > 0) pushes it out. Measuring the marker's centroid turns "looks
    // bowed" into a number, and pins the sign, which is the part that is easy
    // to get backwards.
    std::printf("distortion moves the image the right way:\n");
    {
        auto marker = Mesh::create(SphereGeometry::create(0.28f, 32, 16),
                                   MeshBasicMaterial::create({{"color", Color(1.f, 0.f, 0.f)}}));
        marker->position.set(1.35f, 0.95f, -3.f);// well off-axis, where distortion bites
        scene.add(marker);

        // Used only by the overscan check far below: a blue marker placed
        // OUTSIDE the nominal frustum (ideal normalized 0.75, 0.20 — the
        // rendered pinhole image reaches |x| <= (W/2)/fx = 0.656). It is
        // therefore invisible in every check before that one, and no detector
        // here looks at blue.
        //
        // It is added HERE rather than at the point of use because adding a
        // mesh after an earlier add/remove crashes the renderer outright
        // (0xC0000409) — reproducible with the lens disabled, so unrelated to
        // the camera model. Keeping the scene's object set stable sidesteps it.
        auto outsider = Mesh::create(SphereGeometry::create(0.22f, 24, 12),
                                     MeshBasicMaterial::create({{"color", Color(0.f, 0.f, 1.f)}}));
        outsider->position.set(0.75f * 5.f, 0.20f * 5.f, -5.f);
        scene.add(outsider);
        scene.background = Color(0.f, 0.f, 0.f);
        camera.position.set(0.f, 0.f, 0.f);
        camera.lookAt({0.f, 0.f, -1.f});

        // Find the red marker's centroid in the presented frame.
        auto markerCentroid = [&](float& cxOut, float& cyOut) -> int {
            const auto rgb = renderer.readRGBPixels();
            const size_t n = static_cast<size_t>(kW) * kH;
            if (rgb.size() < n * 3) return 0;
            double sx = 0, sy = 0;
            int count = 0;
            for (size_t i = 0; i < n; ++i) {
                if (rgb[i * 3] > 120 && rgb[i * 3 + 1] < 90 && rgb[i * 3 + 2] < 90) {
                    sx += static_cast<double>(i % kW);
                    sy += static_cast<double>(i / kW);
                    ++count;
                }
            }
            if (count == 0) return 0;
            cxOut = static_cast<float>(sx / count);
            cyOut = static_cast<float>(sy / count);
            return count;
        };

        // A few frames per configuration: TAA needs to settle before the
        // centroid stops drifting.
        auto settle = [&] {
            for (int i = 0; i < 24; ++i) renderer.render(scene, camera);
        };

        renderer.setLensDistortion(LensDistortion{});// pinhole reference
        settle();
        float ux = 0, uy = 0;
        const int nUndist = markerCentroid(ux, uy);
        check(nUndist > 0, "marker visible in the undistorted frame");

        LensDistortion barrel;
        barrel.model = LensModel::BrownConrady;
        barrel.k1 = -0.35f;
        renderer.setLensDistortion(barrel);
        settle();
        float bx = 0, by = 0;
        const int nBarrel = markerCentroid(bx, by);
        check(nBarrel > 0, "marker visible under barrel distortion");

        LensDistortion pincushion;
        pincushion.model = LensModel::BrownConrady;
        pincushion.k1 = 0.30f;
        renderer.setLensDistortion(pincushion);
        settle();
        float px2 = 0, py = 0;
        const int nPin = markerCentroid(px2, py);
        check(nPin > 0, "marker visible under pincushion distortion");

        if (nUndist && nBarrel && nPin) {
            const float centreX = kW * 0.5f, centreY = kH * 0.5f;
            const float rU = std::hypot(ux - centreX, uy - centreY);
            const float rB = std::hypot(bx - centreX, by - centreY);
            const float rP = std::hypot(px2 - centreX, py - centreY);
            std::printf("  marker radius from centre: pinhole %.1f px, barrel %.1f px, pincushion %.1f px\n",
                        rU, rB, rP);
            // k1 = -0.35 at r ~ 0.55 predicts a radius factor of about
            // 1 - 0.35·0.30 = 0.89, so this is a magnitude check as well as a
            // sign check.
            check(rB < rU * 0.95f,
                  "barrel (k1<0) pulls an off-axis point toward the centre");
            check(rP > rU * 1.05f,
                  "pincushion (k1>0) pushes an off-axis point outward");
        }

        // ── Labels follow the pixels ────────────────────────────────────────
        // The whole reason the AOV readback warps too. The colour image is
        // warped on the GPU (rcas.comp), the ids AOV on the CPU
        // (LensDistortion.hpp) — if those two implementations disagree, the
        // marker's silhouette and the marker's instance id end up in
        // different places, and a synthetic dataset silently mislabels.
        std::printf("labels follow the pixels (GPU warp vs CPU warp):\n");
        renderer.setObjectInstanceId(*marker, 77u);
        renderer.setLensDistortion(barrel);
        settle();
        float cxRgb = 0, cyRgb = 0;
        const int nRgb = markerCentroid(cxRgb, cyRgb);

        std::vector<uint8_t> ids;
        int aw = 0, ah = 0, abpp = 0;
        const bool gotIds = renderer.readGBufferAOV(VulkanRendererCore::GBufferAOV::Ids,
                                                    ids, aw, ah, abpp);
        check(gotIds && aw == kW && ah == kH && abpp == 8, "ids AOV readback at the render extent");
        if (gotIds && nRgb > 0 && aw == kW && ah == kH && abpp == 8) {
            // Ids layout: 4x uint16. .y is the STABLE per-object instance id —
            // what setObjectInstanceId overrides. (.x is the per-frame visible
            // index + 1, which is not what we asked for.)
            const auto* px16 = reinterpret_cast<const uint16_t*>(ids.data());
            double sx = 0, sy = 0;
            int count = 0;
            for (int i = 0; i < kW * kH; ++i) {
                if (px16[static_cast<size_t>(i) * 4 + 1] == 77u) {
                    sx += static_cast<double>(i % kW);
                    sy += static_cast<double>(i / kW);
                    ++count;
                }
            }
            check(count > 0, "marker's instance id present in the warped ids AOV");
            if (count > 0) {
                const auto cxId = static_cast<float>(sx / count);
                const auto cyId = static_cast<float>(sy / count);
                const float d = std::hypot(cxId - cxRgb, cyId - cyRgb);
                // Both centroids describe the same silhouette through two
                // independent warp implementations; nearest-vs-bilinear
                // resampling and the marker's soft edge account for well under
                // a pixel of disagreement.
                const bool ok = d < 2.0f;
                std::printf("  %s colour centroid (%.2f, %.2f) vs ids centroid (%.2f, %.2f), %.2f px apart\n",
                            ok ? "[ ok ]" : "[FAIL]", cxRgb, cyRgb, cxId, cyId, d);
                if (!ok) ++failures;
            }
        }

        // ── Overlay content is warped too ───────────────────────────────────
        // The hybrid raster overlay pass (particle billboards — chimney smoke
        // and the like — plus Line/LineSegments and wireframe) composites onto
        // the swapchain AFTER the TAA resolve. When the warp lived in RCAS,
        // which runs BEFORE that, overlays stayed pinhole while the scene bent
        // around them and visibly slid off the geometry they belong to. The
        // warp now runs last (SensorPass), so a line drawn at a mesh's
        // position must still sit on that mesh under distortion.
        std::printf("overlay content follows the lens:\n");
        {
            // A green cross of LineSegments, centred on the same world point
            // as the red marker, small enough that per-vertex vs per-pixel
            // warping is not the thing under test.
            const std::vector<float> verts = {
                    -0.30f, 0.f, 0.f, 0.30f, 0.f, 0.f,
                    0.f, -0.30f, 0.f, 0.f, 0.30f, 0.f};
            auto lineGeo = BufferGeometry::create();
            lineGeo->setAttribute("position", FloatBufferAttribute::create(verts, 3));
            auto lineMat = LineBasicMaterial::create();
            lineMat->color = Color(0.f, 1.f, 0.f);
            auto cross = LineSegments::create(lineGeo, lineMat);
            cross->position.copy(marker->position);
            scene.add(cross);

            auto greenCentroid = [&](float& cxOut, float& cyOut) -> int {
                const auto rgb = renderer.readRGBPixels();
                const size_t n = static_cast<size_t>(kW) * kH;
                if (rgb.size() < n * 3) return 0;
                // Green-DOMINANT rather than green-bright: the line is one
                // pixel wide, so the warp's bilinear gather splits it across
                // two pixels and roughly halves its value. An absolute
                // threshold would report the line as missing purely because it
                // got resampled. Weight by green so the centroid is not pulled
                // by the dimmer half.
                double sx = 0, sy = 0, wsum = 0;
                int count = 0;
                for (size_t i = 0; i < n; ++i) {
                    const int r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
                    if (g > 40 && g > r * 2 && g > b * 2) {
                        const auto wgt = static_cast<double>(g);
                        sx += static_cast<double>(i % kW) * wgt;
                        sy += static_cast<double>(i / kW) * wgt;
                        wsum += wgt;
                        ++count;
                    }
                }
                if (count == 0 || wsum <= 0.0) return 0;
                cxOut = static_cast<float>(sx / wsum);
                cyOut = static_cast<float>(sy / wsum);
                return count;
            };

            renderer.setLensDistortion(LensDistortion{});
            settle();
            float lx0 = 0, ly0 = 0;
            const int nLine0 = greenCentroid(lx0, ly0);
            check(nLine0 > 0, "overlay line visible without a lens");

            renderer.setLensDistortion(barrel);
            settle();
            float lx1 = 0, ly1 = 0;
            const int nLine1 = greenCentroid(lx1, ly1);
            check(nLine1 > 0, "overlay line visible with a lens");

            if (nLine0 && nLine1) {
                const float moved = std::hypot(lx1 - lx0, ly1 - ly0);
                std::printf("  overlay line centroid moved %.1f px under the lens\n", moved);
                // The marker moved ~24 px inward at this k1; the line sits on
                // it, so it has to travel with it. An unwarped overlay would
                // sit still (moved ~= 0) — that was the bug.
                check(moved > 10.f, "overlay line moves with the scene under distortion");
            }

            renderer.setLensDistortion(LensDistortion{});
            scene.remove(*cross);
        }

        // ── Overscan ────────────────────────────────────────────────────────
        // Overscan renders a WIDER field so barrel distortion can gather real
        // geometry into the output corners instead of clamped edge texels. The
        // framing must not change: the same scene point has to land on the same
        // output pixel either way, or overscan would silently re-frame every
        // shot (and get the shrink direction wrong without anyone noticing).
        std::printf("overscan:\n");
        {
            renderer.setLensDistortion(barrel);
            renderer.setLensOverscan(1.f);
            settle();
            float ax = 0, ay = 0;
            const int nA = markerCentroid(ax, ay);
            const auto framePlain = renderer.readRGBPixels();

            renderer.setLensOverscan(1.3f);
            settle();
            float bx = 0, by = 0;
            const int nB = markerCentroid(bx, by);
            const auto frameOver = renderer.readRGBPixels();

            check(nA > 0 && nB > 0, "marker visible with and without overscan");
            if (nA > 0 && nB > 0) {
                const float d = std::hypot(bx - ax, by - ay);
                std::printf("  marker at (%.1f, %.1f) plain vs (%.1f, %.1f) overscanned, %.2f px apart\n",
                            ax, ay, bx, by, d);
                check(d < 2.5f, "overscan does not re-frame the shot");
            }

            (void) framePlain;
            (void) frameOver;

            // The decisive check: an object that lies OUTSIDE the nominal
            // frustum but inside the overscanned one. Without overscan it is
            // never rendered, so the barrel warp can only clamp edge texels
            // into that part of the frame; with overscan it exists and the
            // warp pulls it into view. This is exactly the content the smeared
            // border was standing in for.
            //
            // Placed at ideal normalized (0.75, 0.20): the rendered pinhole
            // image reaches |x| <= (W/2)/fx = 0.656, and a 1.3x overscan
            // reaches 0.853. Barrel then maps it to roughly (365, 174), inside
            // the output frame.
            // NOTE: `outsider` is added at scene-setup time, not here. Adding a
            // mesh at this point in the session crashes the renderer — with the
            // lens on OR off, so it is unrelated to this feature; see the
            // add/remove/add churn note where it is created.
            auto blueCount = [&] {
                const auto rgb = renderer.readRGBPixels();
                const size_t n = static_cast<size_t>(kW) * kH;
                if (rgb.size() < n * 3) return 0;
                int count = 0;
                for (size_t i = 0; i < n; ++i) {
                    const int r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
                    if (b > 90 && b > r * 2 && b > g * 2) ++count;
                }
                return count;
            };

            renderer.setLensOverscan(1.f);
            settle();
            const int blindPixels = blueCount();

            renderer.setLensOverscan(1.3f);
            settle();
            const int seenPixels = blueCount();

            std::printf("  out-of-frustum object: %d px without overscan, %d px with\n",
                        blindPixels, seenPixels);
            check(blindPixels == 0, "object outside the nominal frustum is not visible without overscan");
            check(seenPixels > 50, "overscan brings it into the distorted frame");

            renderer.setLensOverscan(1.f);
        }

        renderer.setLensDistortion(LensDistortion{});
        scene.remove(*marker);
    }

    // ── Sensor noise: the statistics are the ones claimed ───────────────────
    // "It looks grainy" is not the claim. The claim is shot-noise-dominated
    // statistics: variance proportional to signal, with 1/fullWell as the
    // constant, and noise growing as sqrt(ISO). Both are measurable, so
    // measure them.
    std::printf("sensor noise statistics:\n");
    {
        scene.background = Color(0.f, 0.f, 0.f);
        renderer.setAutoExposure(false);// a drifting exposure would masquerade
                                        // as signal-dependent noise

        // Deliberately small full well: it puts the noise well above 8-bit
        // quantization, so what we measure is the model and not the encode.
        constexpr float kFullWell = 2000.f;

        // Mean and temporal variance of the green channel over a central
        // patch, in LINEAR light (the domain the model is defined in).
        auto measure = [&](float grey, int frames, double& meanLin, double& varLin,
                           bool& framesDiffer) {
            scene.background = Color(grey, grey, grey);
            for (int i = 0; i < 12; ++i) renderer.render(scene, camera);// settle

            constexpr int x0 = kW / 4, x1 = 3 * kW / 4, y0 = kH / 4, y1 = 3 * kH / 4;
            std::vector<double> sum, sumSq;
            sum.assign(static_cast<size_t>((x1 - x0)) * (y1 - y0), 0.0);
            sumSq = sum;
            std::vector<unsigned char> prev;
            framesDiffer = false;

            auto toLinear = [](double c) {
                return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
            };
            for (int f = 0; f < frames; ++f) {
                renderer.render(scene, camera);
                const auto rgb = renderer.readRGBPixels();
                if (rgb.size() < static_cast<size_t>(kW) * kH * 3) return false;
                if (!prev.empty() && rgb != prev) framesDiffer = true;
                prev = rgb;
                size_t k = 0;
                for (int y = y0; y < y1; ++y) {
                    for (int x = x0; x < x1; ++x, ++k) {
                        const double lin =
                                toLinear(rgb[(static_cast<size_t>(y) * kW + x) * 3 + 1] / 255.0);
                        sum[k] += lin;
                        sumSq[k] += lin * lin;
                    }
                }
            }
            // Average the PER-PIXEL temporal variance: that isolates temporal
            // noise from any residual spatial structure (and from PRNU, which
            // is fixed per pixel and so contributes nothing here).
            double mAcc = 0, vAcc = 0;
            const auto n = static_cast<double>(frames);
            for (size_t k = 0; k < sum.size(); ++k) {
                const double m = sum[k] / n;
                mAcc += m;
                vAcc += std::max(0.0, sumSq[k] / n - m * m) * n / (n - 1.0);
            }
            meanLin = mAcc / static_cast<double>(sum.size());
            varLin  = vAcc / static_cast<double>(sum.size());
            return true;
        };

        VulkanRendererCore::SensorNoise sn;
        sn.enabled = true;
        sn.fullWellElectrons = kFullWell;
        sn.readNoiseElectrons = 0.f;         // isolate shot noise
        sn.darkCurrentElectronsPerSec = 0.f; // ditto
        sn.prnuPercent = 0.f;                // fixed pattern: no temporal variance
        sn.seed = 12345u;
        renderer.setSensorNoise(sn);
        renderer.setCameraExposure(16.f, 1.f / 125.f, 100.f);// ISO 100 → gain 1

        constexpr int kNoiseFrames = 48;
        bool differ = false;
        for (float grey : {0.25f, 0.5f, 0.75f}) {
            double m = 0, v = 0;
            if (!measure(grey, kNoiseFrames, m, v, differ)) {
                check(false, "readback during noise measurement");
                break;
            }
            // Shot noise: var = mean/fullWell (electrons are Poisson, and one
            // unit of linear signal is fullWell electrons at ISO 100).
            const double predicted = m / kFullWell;
            const double ratio = v / std::max(predicted, 1e-12);
            const bool ok = ratio > 0.6 && ratio < 1.6;
            std::printf("  %s grey %.2f: mean %.4f, var %.3e, predicted %.3e, ratio %.2f\n",
                        ok ? "[ ok ]" : "[FAIL]", grey, m, v, predicted, ratio);
            if (!ok) ++failures;
        }
        check(differ, "consecutive noisy frames actually differ (noise survives TAA)");

        // ISO: 4 stops of gain is 4x the noise amplitude, because the same
        // displayed level is reached from 1/16 as many electrons.
        {
            double mLo = 0, vLo = 0, mHi = 0, vHi = 0;
            bool d1 = false, d2 = false;
            renderer.setCameraExposure(16.f, 1.f / 125.f, 100.f);
            measure(0.5f, kNoiseFrames, mLo, vLo, d1);
            renderer.setCameraExposure(16.f, 1.f / 125.f, 1600.f);
            measure(0.5f, kNoiseFrames, mHi, vHi, d2);
            const double ampRatio = std::sqrt(vHi / std::max(vLo, 1e-12));
            const bool ok = ampRatio > 3.2 && ampRatio < 4.8;
            std::printf("  %s ISO 100 -> 1600 multiplies noise amplitude by %.2f (want ~4)\n",
                        ok ? "[ ok ]" : "[FAIL]", ampRatio);
            if (!ok) ++failures;
            renderer.setCameraExposure(16.f, 1.f / 125.f, 100.f);
        }

        // ── Replayability ──────────────────────────────────────────────────
        // An episode that cannot be replayed is not a reproducible experiment.
        //
        // Byte-comparing frames requires a pipeline that is otherwise
        // frame-invariant, and the default one is NOT: TAA walks a Halton
        // jitter phase every frame, so even a completely static scene resolves
        // to slightly different pixels each time. Switch to the unjittered
        // raster (MSAA, no upscaler) for these checks — that is the only
        // configuration in which "the same frame twice" is a meaningful
        // statement, and it isolates the noise as the sole varying term.
        {
            renderer.setDlss(false);
            renderer.setFsr(false);
            renderer.setGbufferMsaa(2);// > 1 without an upscaler ⇒ jitter gated off
            scene.background = Color(0.5f, 0.5f, 0.5f);

            auto settleFrames = [&] {
                for (int i = 0; i < 24; ++i) renderer.render(scene, camera);
            };

            // First establish that the baseline really is frame-invariant —
            // otherwise the replay checks below would be testing nothing.
            auto snOff = renderer.sensorNoise();
            snOff.enabled = false;
            renderer.setSensorNoise(snOff);
            settleFrames();
            renderer.render(scene, camera);
            const auto quietA = renderer.readRGBPixels();
            renderer.render(scene, camera);
            const auto quietB = renderer.readRGBPixels();
            const bool quiet = quietA == quietB;
            check(quiet, "noise off + unjittered raster -> consecutive frames identical");

            if (quiet) {
                sn.enabled = true;
                sn.seed    = 12345u;
                renderer.setSensorNoise(sn);
                settleFrames();

                // readRGBPixels can hand back a stale swapchain image for the
                // first few reads after the render/readback cadence changes
                // (MAILBOX presentation — a known constraint of this readback
                // path: capture in a steady state). So warm up WITH the
                // readback in the loop, then compare only the tail of the
                // sequence, once the rotation has settled. What is being
                // tested is that the same seed replays the same noise, not
                // that the swapchain is prompt.
                constexpr int kSettleReads = 6;// discarded: cadence not yet steady
                auto capture = [&](int frames) {
                    renderer.resetSensorNoise();
                    std::vector<std::vector<unsigned char>> seq;
                    for (int f = 0; f < kSettleReads + frames; ++f) {
                        renderer.render(scene, camera);
                        auto px = renderer.readRGBPixels();
                        if (f >= kSettleReads) seq.push_back(std::move(px));
                    }
                    return seq;
                };
                const auto runA = capture(4);
                const auto runB = capture(4);
                check(runA == runB, "same seed replays the same noise (steady state)");
                check(runA[0] != runA[1], "successive frames within a run differ");

                auto sn2 = renderer.sensorNoise();
                sn2.seed = 999u;
                renderer.setSensorNoise(sn2);
                const auto runC = capture(4);
                check(runC != runA, "a different seed produces different noise");

                // ── Off is off ─────────────────────────────────────────────
                sn2.enabled = false;
                renderer.setSensorNoise(sn2);
                settleFrames();
                renderer.render(scene, camera);
                const auto offA = renderer.readRGBPixels();
                check(offA == quietA, "disabling noise restores the exact original image");
            }
        }
    }

    std::printf(failures == 0 ? "\nOK\n" : "\n%d CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
