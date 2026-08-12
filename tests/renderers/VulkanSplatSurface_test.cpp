// VulkanSplatSurface_test — depth fusion of a Gaussian-splat scan into a
// triangle surface (plans/splat-surface-bake.md, P1). Four claims, in the
// order the plan asks for them:
//
//   1. DETERMINISM. The same cloud baked twice gives the same vertices and the
//      same indices, bit for bit. Everything downstream (sensor goldens, cooked
//      colliders) inherits this, and the splat rasterizer under it went to some
//      trouble to be bit-exact.
//
//   2. GEOMETRY. A cloud shaped as a plane fuses to a plane, a cloud shaped as
//      a sphere shell to a sphere of the right radius — both to within a voxel.
//      This is the whole point: the mesh is metric, not decorative.
//
//   3. COLLIDER. A PhysX ball dropped on the baked plane rests ON it. The mesh
//      is cooked by the existing PhysxWorld::addStaticTrimesh, so what is being
//      tested is the mesh's usability (winding, welding, no gaps), not PhysX.
//
//   4. FLOATERS. Photogrammetry strays either never reach the AOV's coverage
//      gate or come out as islands the component filter drops; either way the
//      mesh's AABB must not grow around them.
//
// Run standalone (a plain exit-code program, not Catch2):
//   VulkanSplatSurface_test            the assertions
//   VulkanSplatSurface_test --sweep    + the weight-floor and guard A/B tables
//                                        that placed the defaults (not asserted)
// Exits 42 (CTest SKIP_RETURN_CODE) without a Vulkan GPU.

#include "threepp/threepp.hpp"

#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/splats/SplatSurface.hpp"

#ifdef THREEPP_TEST_WITH_PHYSX
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 512, kH = 512;
    constexpr int kSkipCode = 42;
    constexpr float kVoxel = 0.05f;

    int failures = 0;
    void report(bool ok, const std::string& what) {
        std::printf("%s  %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    uint64_t hashMesh(const splats::SurfaceMesh& m) {
        uint64_t h = 1469598103934665603ull;// FNV-1a
        auto mix = [&h](const void* p, size_t n) {
            const auto* b = static_cast<const uint8_t*>(p);
            for (size_t i = 0; i < n; ++i) {
                h ^= b[i];
                h *= 1099511628211ull;
            }
        };
        mix(m.positions.data(), m.positions.size() * sizeof(float));
        mix(m.indices.data(), m.indices.size() * sizeof(uint32_t));
        return h;
    }

    // A slab of splats one splat thick at y = 0: the simplest shape whose
    // fusion error is readable as a single number.
    SplatData makePlane(size_t count = 40000, unsigned seed = 7u) {
        SplatGenerator::Options o;
        o.count = count;
        o.seed = seed;
        o.shDegree = 0;
        o.extent.set(4.f, 0.02f, 4.f);
        o.minScale = 0.02f;
        o.maxScale = 0.035f;
        o.anisotropy = 1.2f;
        o.minOpacity = 0.85f;
        o.maxOpacity = 1.f;
        auto d = SplatGenerator::generate(o);
        for (size_t i = 0; i < d.count(); ++i) d.setDcColor(i, Vector3{0.8f, 0.8f, 0.8f});
        return d;
    }

    // The same generator, means pushed onto a shell of radius r. Smaller splats
    // than the plane's, and more of them: the fused surface is the VISIBLE FRONT
    // of the outermost splats, so a shell of fat splats measures larger than the
    // radius its means lie on, by about a splat.
    SplatData makeShell(float r, size_t count = 120000, unsigned seed = 11u) {
        SplatGenerator::Options o;
        o.count = count;
        o.seed = seed;
        o.shDegree = 0;
        o.extent.set(2.f, 2.f, 2.f);
        o.minScale = 0.010f;
        o.maxScale = 0.018f;
        o.anisotropy = 1.2f;
        o.minOpacity = 0.85f;
        o.maxOpacity = 1.f;
        auto d = SplatGenerator::generate(o);
        for (size_t i = 0; i < d.count(); ++i) {
            Vector3 m = d.means[i];
            const float len = m.length();
            m = (len > 1e-4f) ? m * (r / len) : Vector3{r, 0.f, 0.f};
            d.means[i] = m;
            d.setDcColor(i, Vector3{0.8f, 0.8f, 0.8f});
        }
        return d;
    }

    // What photogrammetry actually leaves in the air above a scan: compact
    // opaque CLUMPS (each a few centimetres across — small enough to be debris,
    // big enough to reach the AOV's 0.5-coverage gate and be fused) plus a
    // sparse translucent halo (which individually never reaches the gate).
    // Appended to a cloud, so both live in one SplatCloud, as they do in a scan.
    SplatData withStrays(const SplatData& base, unsigned seed = 99u) {

        SplatGenerator::Options o;
        o.count = 5 * 25 + 100;
        o.seed = seed;
        o.shDegree = 0;
        o.extent.set(3.f, 1.6f, 3.f);
        o.minScale = 0.018f;
        o.maxScale = 0.030f;
        auto s = SplatGenerator::generate(o);

        const Vector3 clumps[5] = {{-1.3f, 0.55f, 0.9f}, {0.7f, 1.05f, -1.1f}, {1.4f, 0.42f, 1.2f}, {-0.5f, 0.85f, -1.4f}, {0.1f, 1.25f, 0.2f}};
        SplatData out = base;
        const size_t first = out.count();
        for (size_t i = 0; i < s.count(); ++i) {
            Vector3 m = s.means[i];
            // The halo's opacity is the point of it: at 0.08 a ray would have to
            // cross eight strays to drive transmittance under the AOV's 0.5
            // gate, so the halo is never fused at all — the gate IS the first
            // floater defense. Raise it to ~0.3 and clusters of three start
            // reading as surface, which is a real property of the statistic and
            // not a bug: at that density it is a translucent medium, not debris.
            float opacity = 0.08f;
            if (i < 5 * 25) {
                // A couple of centimetres of jitter about the clump centre: a
                // blob a voxel or so across, which meshes to an island well
                // under minComponentVoxels.
                m.set(m.x * 0.008f, m.y * 0.008f, m.z * 0.008f);
                m.add(clumps[i / 25]);
                opacity = 0.9f;
            } else {
                m.y = std::abs(m.y) + 0.35f;// halo: above the plane, never in it
            }
            out.means.push_back(m);
            out.scales.push_back(s.scales[i]);
            out.rotations.push_back(s.rotations[i]);
            out.opacities.push_back(opacity);
            // Degree 0: one coefficient a splat, so appending is layout-safe.
            for (int c = 0; c < 3; ++c) out.sh.push_back(0.f);
        }
        for (size_t i = first; i < out.count(); ++i) out.setDcColor(i, Vector3{0.7f, 0.7f, 0.7f});
        return out;
    }

    // What an OUTDOOR scan has behind its subject: a distant backdrop the pose
    // cameras see straight past the subject into. Big sparse splats on a shell
    // far outside it, appended to the near cloud so both live in ONE SplatCloud
    // the way a real scan does. Kept under 10 % of the cloud, so the robust fit
    // — a 90th-percentile radius — still describes the subject; a backdrop that
    // moved the fit would be a different scan, not a backdrop.
    SplatData withBackdrop(const SplatData& base, float r, size_t count = 3000, unsigned seed = 23u) {

        SplatGenerator::Options o;
        o.count = count;
        o.seed = seed;
        o.shDegree = 0;
        o.extent.set(2.f, 2.f, 2.f);
        o.minScale = 2.0f;
        o.maxScale = 3.0f;
        o.anisotropy = 1.f;
        o.minOpacity = 0.9f;
        o.maxOpacity = 1.f;
        auto s = SplatGenerator::generate(o);

        SplatData out = base;
        const size_t first = out.count();
        for (size_t i = 0; i < s.count(); ++i) {
            Vector3 m = s.means[i];
            const float len = m.length();
            out.means.push_back((len > 1e-4f) ? m * (r / len) : Vector3{r, 0.f, 0.f});
            out.scales.push_back(s.scales[i]);
            out.rotations.push_back(s.rotations[i]);
            out.opacities.push_back(s.opacities[i]);
            for (int c = 0; c < 3; ++c) out.sh.push_back(0.f);
        }
        for (size_t i = first; i < out.count(); ++i) out.setDcColor(i, Vector3{0.35f, 0.4f, 0.45f});
        return out;
    }

    // Depth-sensor camera clip range. The G-buffer's Depth AOV is REVERSED-Z
    // NDC (see readGBufferAOV), which the renderer builds from the camera's own
    // GL projection as z' = 0.5w - 0.5z; inverting that for a perspective
    // camera gives the view-space distance below.
    constexpr float kNear = 0.1f, kFar = 20.f;

    double viewDistanceFromNdc(double d) {
        return double(kFar) * double(kNear) / (d * (double(kFar) - double(kNear)) + double(kNear));
    }

    // The centre pixel of a view's depth AOV, as a view-space distance. 0 where
    // nothing was drawn (reversed-Z far plane).
    double viewCentreDistance(VulkanRenderer& renderer, uint32_t viewHandle) {
        std::vector<uint8_t> aov;
        int w = 0, h = 0, bpp = 0;
        if (!renderer.readViewGBufferAOV(viewHandle, VulkanRenderer::GBufferAOV::Depth, aov, w, h, bpp))
            return 0.0;
        if (bpp != 4 || w <= 0 || h <= 0) return 0.0;
        float d = 0.f;
        std::memcpy(&d, aov.data() + (size_t(h / 2) * size_t(w) + size_t(w / 2)) * 4u, 4);
        return d > 0.f ? viewDistanceFromNdc(double(d)) : 0.0;
    }

    int maxDelta(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
        if (a.size() != b.size() || a.empty()) return 255;
        int m = 0;
        for (size_t i = 0; i < a.size(); ++i)
            m = std::max(m, std::abs(int(a[i]) - int(b[i])));
        return m;
    }

    struct PlaneFit {
        double meanY{0}, rmsY{0}, maxAbs{0};
        size_t n{0};
    };
    PlaneFit fitPlane(const splats::SurfaceMesh& m) {
        PlaneFit f;
        f.n = m.vertexCount();
        if (!f.n) return f;
        for (size_t i = 0; i < m.positions.size(); i += 3) f.meanY += m.positions[i + 1];
        f.meanY /= static_cast<double>(f.n);
        for (size_t i = 0; i < m.positions.size(); i += 3) {
            const double d = m.positions[i + 1] - f.meanY;
            f.rmsY += d * d;
            f.maxAbs = std::max(f.maxAbs, std::abs(d));
        }
        f.rmsY = std::sqrt(f.rmsY / static_cast<double>(f.n));
        return f;
    }

}// namespace

int main(int argc, char** argv) {

    bool sweep = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--sweep") == 0) sweep = true;

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanSplatSurface_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    VulkanRenderer& renderer = *rendererPtr;
    renderer.setRenderScale(1.f);
    renderer.setDlss(false);
    renderer.setFsr(false);

    splats::SurfaceBakeOptions opts;
    opts.voxelSize = kVoxel;
    opts.poseCount = 16;

    // ── 1. determinism ──────────────────────────────────────────────────────
    auto planeCloud = SplatCloud::create(makePlane());
    auto scene = Scene::create();
    scene->add(planeCloud);

    const auto bake1 = splats::bakeSurface(renderer, *planeCloud, opts);
    const auto bake2 = splats::bakeSurface(renderer, *planeCloud, opts);
    std::printf("       plane  %zu verts  %zu tris  %llu blocks  %llu voxels"
                "  render %.0f ms  fuse %.0f ms  mesh %.0f ms\n",
                bake1.vertexCount(), bake1.triangleCount(),
                static_cast<unsigned long long>(bake1.stats.blocks),
                static_cast<unsigned long long>(bake1.stats.observedVoxels),
                bake1.stats.renderMs, bake1.stats.fuseMs, bake1.stats.meshMs);
    std::printf("       samples %llu  dropped fringe %llu  behind-neighbours %llu  far %llu"
                "  (gate %.2f)\n",
                static_cast<unsigned long long>(bake1.stats.depthSamples),
                static_cast<unsigned long long>(bake1.stats.skippedFringe),
                static_cast<unsigned long long>(bake1.stats.skippedOutlier),
                static_cast<unsigned long long>(bake1.stats.skippedFar),
                bake1.stats.maxDepth);
    std::printf("       carve blocks: %llu per-voxel, %llu bulk, %llu skipped"
                "  -> fast path %.1f%%   peak %.1f MB, %llu refused\n",
                static_cast<unsigned long long>(bake1.stats.carveVoxelBlocks),
                static_cast<unsigned long long>(bake1.stats.carveBulkBlocks),
                static_cast<unsigned long long>(bake1.stats.carveSkippedBlocks),
                100.0 * double(bake1.stats.carveBulkBlocks + bake1.stats.carveSkippedBlocks) /
                        std::max<double>(1.0, double(bake1.stats.carveVoxelBlocks +
                                                     bake1.stats.carveBulkBlocks +
                                                     bake1.stats.carveSkippedBlocks)),
                double(bake1.stats.peakBlockBytes) / (1024.0 * 1024.0),
                static_cast<unsigned long long>(bake1.stats.refusedBlocks));
    report(!bake1.empty(), "the plane cloud bakes to a non-empty surface");
    report(bake1.positions == bake2.positions && bake1.indices == bake2.indices,
           "the same cloud baked twice is identical, vertex and index");
    std::printf("       hash %llu / %llu\n",
                static_cast<unsigned long long>(hashMesh(bake1)),
                static_cast<unsigned long long>(hashMesh(bake2)));

    // The cloud is handed back where it was found.
    report(planeCloud->parent == scene.get(), "the bake puts the cloud back under its own parent");

    // ── 2. geometry ─────────────────────────────────────────────────────────
    const auto pf = fitPlane(bake1);
    std::printf("       plane fit  mean y %+.4f  rms %.4f  max |dy| %.4f   (voxel %.3f)\n",
                pf.meanY, pf.rmsY, pf.maxAbs, kVoxel);
    report(pf.rmsY < kVoxel, "the fused plane is flat to within a voxel");
    report(std::abs(pf.meanY) < 2.0 * kVoxel, "and sits at the height the splats are at");

    // Winding, which the collider case below depends on and no fit can see: the
    // triangles must face the FREE side (up, here), not into the solid.
    double nUp = 0;
    for (size_t t = 0; t + 2 < bake1.indices.size(); t += 3) {
        const auto* p = bake1.positions.data();
        const uint32_t a = bake1.indices[t] * 3, b = bake1.indices[t + 1] * 3, c = bake1.indices[t + 2] * 3;
        const Vector3 e1{p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]};
        const Vector3 e2{p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]};
        nUp += static_cast<double>(e1.z * e2.x - e1.x * e2.z);// (e1 x e2).y
    }
    std::printf("       winding: summed triangle normal .y %+.4f\n", nUp);
    report(nUp > 0.0, "the triangles face the side the cameras were on");

    auto shellCloud = SplatCloud::create(makeShell(1.f));
    auto shellScene = Scene::create();
    shellScene->add(shellCloud);
    const auto shell = splats::bakeSurface(renderer, *shellCloud, opts);
    double rMean = 0, rRms = 0;
    const size_t sn = shell.vertexCount();
    for (size_t i = 0; i < shell.positions.size(); i += 3)
        rMean += std::sqrt(shell.positions[i] * shell.positions[i] +
                           shell.positions[i + 1] * shell.positions[i + 1] +
                           shell.positions[i + 2] * shell.positions[i + 2]);
    if (sn) rMean /= static_cast<double>(sn);
    for (size_t i = 0; i < shell.positions.size(); i += 3) {
        const double r = std::sqrt(shell.positions[i] * shell.positions[i] +
                                   shell.positions[i + 1] * shell.positions[i + 1] +
                                   shell.positions[i + 2] * shell.positions[i + 2]);
        rRms += (r - rMean) * (r - rMean);
    }
    if (sn) rRms = std::sqrt(rRms / static_cast<double>(sn));
    std::printf("       shell  %zu verts  radius mean %.4f (want 1.000)  rms %.4f\n", sn, rMean, rRms);
    report(sn > 0 && std::abs(rMean - 1.0) < kVoxel, "the fused sphere shell has the radius it was built with");
    report(rRms < kVoxel, "and is round to within a voxel");

    // ── 3. collider ─────────────────────────────────────────────────────────
#ifdef THREEPP_TEST_WITH_PHYSX
    {
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(bake1.positions, 3));
        geometry->setIndex(std::vector<unsigned int>(bake1.indices.begin(), bake1.indices.end()));

        PhysxWorld world;
        auto* floor = world.addStaticTrimesh(*geometry);
        report(floor != nullptr, "the baked mesh cooks into a PhysX static trimesh");

        auto ball = Mesh::create(SphereGeometry::create(0.15f, 16, 12), MeshBasicMaterial::create());
        ball->position.set(0.2f, 1.5f, -0.3f);
        auto* body = world.add(*ball, 500.f);
        for (int i = 0; i < 240; ++i) world.step(1.f / 120.f);

        const float rest = body->getGlobalPose().p.y;
        const double err = std::abs(static_cast<double>(rest) - (pf.meanY + 0.15));
        std::printf("       ball rests at y %.4f  (plane %.4f + r 0.15 -> err %.4f, voxel %.3f)\n",
                    rest, pf.meanY, err, kVoxel);
        report(err < kVoxel, "a ball dropped on the baked scan rests on it, within a voxel");
    }
#else
    std::printf("[skip] PhysX SDK not configured — collider case not built\n");
#endif

    // ── 4. floaters ─────────────────────────────────────────────────────────
    auto strayCloud = SplatCloud::create(withStrays(makePlane()));
    auto strayScene = Scene::create();
    strayScene->add(strayCloud);
    // Sized to THIS scan's debris, the way a caller has to: the clumps mesh to
    // islands of ~250 cells each (truncation inflates a 5 cm blob into a 30 cm
    // one), which straddles the 256-cell default. The knob is an area in
    // disguise — see the header — and 640 cells is still only a patch 25 voxels
    // across, far under the 6400 the plane itself covers.
    auto strayOpts = opts;
    strayOpts.minComponentVoxels = 640;
    const auto stray = splats::bakeSurface(renderer, *strayCloud, strayOpts);
    size_t above = 0;
    for (size_t i = 1; i < stray.positions.size(); i += 3)
        if (stray.positions[i] > 0.2f) ++above;
    std::printf("       strays  %zu tris  components %u  culled %u (%llu tris)"
                "  %zu/%zu verts above y=0.2"
                "  aabb y [%.3f, %.3f] vs clean [%.3f, %.3f]\n",
                stray.triangleCount(), stray.stats.components, stray.stats.culledComponents,
                static_cast<unsigned long long>(stray.stats.culledTriangles),
                above, stray.vertexCount(),
                stray.stats.aabbMin.y, stray.stats.aabbMax.y,
                bake1.stats.aabbMin.y, bake1.stats.aabbMax.y);
    report(!stray.empty(), "the plane still bakes with strays above it");
    // The clumps sit between y = 0.42 and 1.25; a mesh that kept them would say
    // so here and nowhere else. Carved or filtered is not distinguished — the
    // plan asks only that they not end up in the mesh.
    report(stray.stats.aabbMax.y < bake1.stats.aabbMax.y + 2.f * kVoxel,
           "and the mesh's AABB is not inflated by them");
    report(above * 100 < stray.vertexCount(),
           "under 1% of the surface survives above the plane");

    // ── 5. the sensor world (P2) ────────────────────────────────────────────
    // The baked plane joins the scene as a sensor-only mesh. The primary camera
    // must not see it (the splats are what renders there), a lidar beam and a
    // secondary depth view must — and only once the scene opts in AND, on the
    // raster side, that view asks. A second secondary view at the same pose is
    // the control for the per-view half: it never asks and must stay blind.
    {
        auto camera = PerspectiveCamera::create(55.f, float(kW) / float(kH), 0.1f, 100.f);
        camera->position.set(0.f, 2.2f, 3.2f);
        camera->lookAt(0.f, 0.f, 0.f);
        auto draw = [&](int n) {
            for (int i = 0; i < n; ++i) renderer.render(*scene, *camera);
        };
        // Looking straight down, so up must not be the view direction.
        auto depthCam = PerspectiveCamera::create(40.f, 1.f, kNear, kFar);
        depthCam->up.set(0.f, 0.f, -1.f);
        depthCam->position.set(0.f, 2.f, 0.f);
        depthCam->lookAt(0.f, 0.f, 0.f);

        draw(1);// addView needs a rendered frame to share pipelines with
        const uint32_t viewH = renderer.addView(*depthCam, 256, 256);
        report(viewH != 0u, "the depth sensor view attaches");
        // The control for the per-view rule: same camera, same size, same
        // frame — everything about it equals the depth view except that it
        // never asks for sensor surfaces. This is the RGB camera preview and
        // the editor viewport pane, both of which are secondary views too.
        const uint32_t rgbH = renderer.addView(*depthCam, 256, 256);
        report(rgbH != 0u && rgbH != viewH, "an RGB preview view attaches alongside it");
        report(renderer.setViewSensorSurfaces(viewH, true) && !renderer.setViewSensorSurfaces(0u, true),
               "the depth view can ask for sensor surfaces; the primary cannot");

        draw(8);
        const auto imgA = renderer.readRGBPixels();
        draw(8);
        const auto imgControl = renderer.readRGBPixels();
        // The renderer's own frame-to-frame noise on an unchanged scene: the
        // bar the sensor mesh has to clear, measured rather than assumed.
        const int noise = maxDelta(imgA, imgControl);

        auto surface = splats::makeSensorMesh(bake1);
        report(surface != nullptr, "the baked surface becomes a sensor mesh");
        scene->add(surface);

        const double planeY = pf.meanY;
        std::vector<LidarBeam> beams;
        for (int i = 0; i < 9; ++i)
            beams.push_back({Vector3{-0.25f + 0.25f * float(i % 3), 2.f, -0.25f + 0.25f * float(i / 3)},
                             Vector3{0.f, -1.f, 0.f}});

        // Absence: added, not opted in. Nothing perceives it.
        draw(8);
        const int deltaOff = maxDelta(imgControl, renderer.readRGBPixels());
        std::vector<LidarReturn> retsOff;
        renderer.scanLidar(beams, retsOff);
        size_t hitsOff = 0;
        for (const auto& r : retsOff)
            if (r.returnNo > 0) ++hitsOff;
        double depthOff = viewCentreDistance(renderer, viewH);
        report(hitsOff == 0, "not opted in: the lidar sees nothing where the surface is");
        // The view asked and the scene did not: the master is a veto, not a
        // default the flag can override.
        report(depthOff == 0.0, "not opted in: even a flagged depth view sees nothing there");

        // Opted in.
        renderer.setSensorOnlySurfaces(true);
        draw(8);
        const auto imgOn = renderer.readRGBPixels();
        const int deltaOn = maxDelta(imgControl, imgOn);
        std::vector<LidarReturn> rets;
        renderer.scanLidar(beams, rets);
        double rangeErr = 0;
        size_t hits = 0;
        for (const auto& r : rets) {
            if (r.returnNo <= 0) continue;
            ++hits;
            rangeErr = std::max(rangeErr, std::abs(static_cast<double>(r.distance) - (2.0 - planeY)));
        }
        const double depthOn = viewCentreDistance(renderer, viewH);
        const double depthErr = std::abs(depthOn - (2.0 - planeY));
        // Same instant, same pose, same size — only the flag differs.
        const double depthRgb = viewCentreDistance(renderer, rgbH);

        std::printf("       lidar %zu/%zu beams return, max |range - (2 - %.4f)| %.4f  (voxel %.3f)\n",
                    hits, beams.size(), planeY, rangeErr, kVoxel);
        std::printf("       depth view centre %.4f vs %.4f -> err %.4f;  off %.4f;  unflagged view %.4f\n",
                    depthOn, 2.0 - planeY, depthErr, depthOff, depthRgb);
        std::printf("       primary maxDelta: control %d, mesh-off %d, mesh-on %d\n",
                    noise, deltaOff, deltaOn);

        report(hits == beams.size() && rangeErr < kVoxel,
               "opted in: every beam returns the baked plane's range, within a voxel");
        report(depthOn > 0.0 && depthErr < kVoxel,
               "opted in: the flagged depth view sees the plane at the right distance");
        report(depthRgb == 0.0,
               "and the unflagged view beside it sees nothing, same instant and pose");
        report(deltaOn <= noise && deltaOff <= noise,
               "and the primary image is unchanged by the mesh, opted in or not");

        renderer.setSensorOnlySurfaces(false);
        renderer.removeView(rgbH);
        renderer.removeView(viewH);
        scene->remove(*surface);
    }

    // ── 6. a distant backdrop is not fused ──────────────────────────────────
    // The defect this case exists for: the pose cameras' far plane is 20 fit
    // radii, so an outdoor scan returns depths on background splats far outside
    // the subject, and unfenced EVERY one of them allocated blocks along its
    // whole truncation band — gigabytes of TSDF over a volume nobody asked to
    // fuse, and then an O(poses x blocks x 512) carve over all of it. The gate
    // is on ALLOCATION only, so those samples still carve; they just do not buy
    // storage.
    //
    // The claims are counters, not seconds: they are exact functions of the
    // input and immune to whatever else is using the GPU. The wall times are
    // printed for orientation and asserted on by nothing.
    {
        auto backCloud = SplatCloud::create(withBackdrop(makePlane(), 30.f));
        auto backScene = Scene::create();
        backScene->add(backCloud);
        const auto back = splats::bakeSurface(renderer, *backCloud, opts);

        const auto work = [](const splats::SurfaceMesh& m) {
            return m.stats.carveVoxelBlocks + m.stats.carveBulkBlocks + m.stats.carveSkippedBlocks;
        };
        std::printf("       backdrop  %zu tris  gate %.2f  far samples %llu of %llu"
                    "  peak %.1f MB (plane %.1f)  refused %llu\n",
                    back.triangleCount(), back.stats.maxDepth,
                    static_cast<unsigned long long>(back.stats.skippedFar),
                    static_cast<unsigned long long>(back.stats.depthSamples + back.stats.skippedFar),
                    double(back.stats.peakBlockBytes) / (1024.0 * 1024.0),
                    double(bake1.stats.peakBlockBytes) / (1024.0 * 1024.0),
                    static_cast<unsigned long long>(back.stats.refusedBlocks));
        std::printf("       backdrop carve blocks: %llu per-voxel, %llu bulk, %llu skipped"
                    " (%llu total)   vs plane %llu / %llu / %llu\n",
                    static_cast<unsigned long long>(back.stats.carveVoxelBlocks),
                    static_cast<unsigned long long>(back.stats.carveBulkBlocks),
                    static_cast<unsigned long long>(back.stats.carveSkippedBlocks),
                    static_cast<unsigned long long>(work(back)),
                    static_cast<unsigned long long>(bake1.stats.carveVoxelBlocks),
                    static_cast<unsigned long long>(bake1.stats.carveBulkBlocks),
                    static_cast<unsigned long long>(bake1.stats.carveSkippedBlocks));

        // The fast paths' own correctness instrument. They claim to be bit-exact
        // restatements of the per-voxel path, and this is the only case where
        // all THREE fire — a flat plane's blocks all straddle its own surface,
        // so nothing there is ever provably free space; a backdrop 30 m behind
        // it is what puts whole blocks a truncation clear of what they can see.
        auto slowOpts = opts;
        slowOpts.carveFastPaths = false;
        const auto slow = splats::bakeSurface(renderer, *backCloud, slowOpts);
        report(back.stats.carveBulkBlocks > 0 && back.stats.carveSkippedBlocks > 0,
               "the backdrop case fires both whole-block fast paths");
        report(back.positions == slow.positions && back.indices == slow.indices,
               "and the mesh is bit-identical to the same bake with them off");
        std::printf("       fast-path A/B hash %llu / %llu  (%llu blocks per-voxel with them off)\n",
                    static_cast<unsigned long long>(hashMesh(back)),
                    static_cast<unsigned long long>(hashMesh(slow)),
                    static_cast<unsigned long long>(slow.stats.carveVoxelBlocks));
        std::printf("       backdrop aabb [%.2f %.2f %.2f]..[%.2f %.2f %.2f]"
                    " vs plane [%.2f %.2f %.2f]..[%.2f %.2f %.2f]\n",
                    back.stats.aabbMin.x, back.stats.aabbMin.y, back.stats.aabbMin.z,
                    back.stats.aabbMax.x, back.stats.aabbMax.y, back.stats.aabbMax.z,
                    bake1.stats.aabbMin.x, bake1.stats.aabbMin.y, bake1.stats.aabbMin.z,
                    bake1.stats.aabbMax.x, bake1.stats.aabbMax.y, bake1.stats.aabbMax.z);
        // Printed, never asserted on. Seconds measure whatever else is using the
        // GPU as much as they measure this code; the claims below are counters,
        // which are exact functions of the input.
        std::printf("       wall (indicative only): backdrop"
                    " render %.0f fuse %.0f mesh %.0f ms  vs plane %.0f / %.0f / %.0f\n",
                    back.stats.renderMs, back.stats.fuseMs, back.stats.meshMs,
                    bake1.stats.renderMs, bake1.stats.fuseMs, bake1.stats.meshMs);

        // The backdrop mesh is BIGGER than the plane's (measured: 18517 tris vs
        // 13460, reaching a truncation below it) and that is the fusion working,
        // not leaking: rays that pass the plane's silhouette now END somewhere
        // instead of nowhere, so the space beside and under the rim becomes
        // OBSERVED free space and the slab closes. The tolerance is therefore
        // one truncation plus a voxel, not a hair — what must not happen is
        // geometry out at the backdrop, 30 m away.
        const float pad = 4.f * kVoxel + kVoxel;
        const bool tight = !back.empty() &&
                           back.stats.aabbMin.x > bake1.stats.aabbMin.x - pad &&
                           back.stats.aabbMin.y > bake1.stats.aabbMin.y - pad &&
                           back.stats.aabbMin.z > bake1.stats.aabbMin.z - pad &&
                           back.stats.aabbMax.x < bake1.stats.aabbMax.x + pad &&
                           back.stats.aabbMax.y < bake1.stats.aabbMax.y + pad &&
                           back.stats.aabbMax.z < bake1.stats.aabbMax.z + pad;
        report(tight, "a scan with a distant backdrop meshes only the near subject");
        // 4x and not 2x because the backdrop also DEFEATS the fringe erode — it
        // infers coverage from the mask, and a full mask has no silhouette — so
        // the plane's own near-gate rim samples survive and allocate behind it.
        // The bound that matters is against the unfenced cost, which --sweep
        // prints: without the gate this same scan allocates two orders of
        // magnitude more.
        report(back.stats.skippedFar > 0 && back.stats.refusedBlocks == 0 &&
                       back.stats.peakBlockBytes < 4 * bake1.stats.peakBlockBytes,
               "the far samples are gated out of ALLOCATION and the volume stays plane-sized");
        report(work(back) < 5 * work(bake1) && back.stats.carveVoxelBlocks < 5 * bake1.stats.carveVoxelBlocks,
               "and the carve pass does the same order of work as the plane alone");

        if (sweep) {
            // The A/B that placed the gate: the same scan with allocation
            // unfenced. Not asserted — it is the measurement, and on a real
            // outdoor scan it is what exhausts the machine.
            auto o = opts;
            o.maxDepth = 1e6f;
            const auto unfenced = splats::bakeSurface(renderer, *backCloud, o);
            std::printf("       backdrop UNFENCED: %llu blocks (%.1f MB), carve %llu blocks,"
                        " %zu tris, fuse %.0f ms   vs gated %llu (%.1f MB) / %llu / %zu / %.0f ms\n",
                        static_cast<unsigned long long>(unfenced.stats.blocks),
                        double(unfenced.stats.peakBlockBytes) / (1024.0 * 1024.0),
                        static_cast<unsigned long long>(work(unfenced)),
                        unfenced.triangleCount(), unfenced.stats.fuseMs,
                        static_cast<unsigned long long>(back.stats.blocks),
                        double(back.stats.peakBlockBytes) / (1024.0 * 1024.0),
                        static_cast<unsigned long long>(work(back)),
                        back.triangleCount(), back.stats.fuseMs);
        }
    }

    if (sweep) {
        // The tables that placed the defaults. Not assertions — they are the
        // measurement the plan's open questions asked for.
        std::printf("\n       weight floor sweep (plane):\n");
        for (const float floorW : {0.5f, 1.f, 2.f, 4.f, 8.f}) {
            auto o = opts;
            o.weightFloor = floorW;
            const auto m = splats::bakeSurface(renderer, *planeCloud, o);
            const auto f = fitPlane(m);
            std::printf("         floor %4.1f : %7zu tris  rms %.4f  max |dy| %.4f\n",
                        floorW, m.triangleCount(), f.rmsY, f.maxAbs);
        }
        std::printf("       coverage-gate defense A/B (plane):\n");
        for (int mode = 0; mode < 4; ++mode) {
            auto o = opts;
            o.fringeErode = (mode & 1) ? 1 : 0;
            o.outlierTolerance = (mode & 2) ? 1.f : 0.f;
            const auto m = splats::bakeSurface(renderer, *planeCloud, o);
            const auto f = fitPlane(m);
            std::printf("         erode %d  outlier %.0f : %7zu tris  rms %.4f  max |dy| %.4f"
                        "  dropped %llu/%llu of %llu\n",
                        o.fringeErode, o.outlierTolerance, m.triangleCount(), f.rmsY, f.maxAbs,
                        static_cast<unsigned long long>(m.stats.skippedFringe),
                        static_cast<unsigned long long>(m.stats.skippedOutlier),
                        static_cast<unsigned long long>(m.stats.depthSamples));
        }
    }

    std::printf(failures ? "\n%d check(s) FAILED\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
