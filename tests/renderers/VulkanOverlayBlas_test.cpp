// VulkanOverlayBlas_test: a geometry that only overlay meshes draw gets no
// acceleration structure.
//
// A wireframe mesh, or a mesh on the renderer's overlay layer, is drawn by the
// raster overlay pass and is never a TLAS instance, so no ray meets it. Its
// BlasRecord still carried an acceleration structure: built at admission,
// refit on every edit, and holding device memory for nothing (spot_slam's
// wireframe marching-cubes surface, 600k vertices rebuilt every 3 s). Now
// such a record is its buffers, as every record is without ray query:
//   wire   a green wireframe sphere: it draws, no lidar beam meets it;
//   panel  a red quad on the overlay layer: it draws, no beam meets it;
//   pair   one sphere geometry shared by a wireframe mesh (there from the
//          start) and a solid blue mesh added later: the record built without
//          a structure for the wireframe is rebuilt with one when the solid
//          arrives, and the solid draws and returns a beam;
//   strip  a wireframe triangle soup of twelve cells, preallocated, one cell
//          placed, that places one more on every third frame: every edit
//          uploads its vertices (the last cell draws) and records no BLAS
//          refit (dynamicGeomStats().refitsRecorded stays 0).
// The lidar checks run with ray query only; without it no record has a
// structure and the raster checks stand alone.
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "vulkan_test_util.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 384, kH = 256;
    constexpr int kFramesPerPhase = 60;
    constexpr unsigned kOverlayLayer = 5;
    constexpr float kRowY = 0.9f, kSphereR = 0.45f;
    constexpr float kWireX = -1.5f, kPanelX = 0.f, kPairX = 1.5f, kSolidY = -0.3f;
    // The strip: kStripCells cells along the bottom, each a kStripGrid x
    // kStripGrid grid of quads, dense enough that a patch at a cell's centre
    // reads its lines.
    constexpr int kStripCells = 12, kStripGrid = 4;
    constexpr int kCellVerts = kStripGrid * kStripGrid * 6;
    constexpr float kStripX0 = -1.95f, kStripPitch = 0.33f, kStripW = 0.3f, kStripY = -1.1f, kStripH = 0.4f;

    std::shared_ptr<Texture> constantEnv(float le) {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, le);
        for (size_t i = 3; i < data.size(); i += 4) data[i] = 1.f;
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    std::array<double, 3> patch(const std::vector<unsigned char>& px, int cx, int cy, int r) {
        std::array<double, 3> sum{};
        int n = 0;
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                for (int c = 0; c < 3; ++c) sum[c] += px[(static_cast<size_t>(y) * kW + x) * 3 + c];
                ++n;
            }
        }
        for (auto& s : sum) s /= n;
        return sum;
    }

    // The channel that leads the other two by `margin`, or -1. A wireframe
    // covers part of a patch, so its colour leads by less than a fill's.
    int leads(const std::array<double, 3>& c, double margin) {
        for (int i = 0; i < 3; ++i) {
            const double o1 = c[(i + 1) % 3], o2 = c[(i + 2) % 3];
            if (c[i] > o1 + margin && c[i] > o2 + margin) return i;
        }
        return -1;
    }

    std::shared_ptr<MeshStandardMaterial> wireMat() {
        auto m = MeshStandardMaterial::create();
        m->color = Color(0, 1, 0);
        m->roughness = 1.f;
        m->metalness = 0.f;
        m->wireframe = true;
        return m;
    }

    float cellCenterX(int k) { return kStripX0 + static_cast<float>(k) * kStripPitch + kStripW * 0.5f; }

    // Place cell k of the strip and admit it to the draw range. The vertices
    // of a cell not yet placed sit at the origin: zero-length lines.
    void placeCell(BufferGeometry& g, int k) {
        auto* pos = g.getAttribute<float>("position");
        const float x0 = kStripX0 + static_cast<float>(k) * kStripPitch;
        const float y0 = kStripY - kStripH * 0.5f;
        const float dx = kStripW / kStripGrid, dy = kStripH / kStripGrid;
        int v = k * kCellVerts;
        for (int gy = 0; gy < kStripGrid; ++gy) {
            for (int gx = 0; gx < kStripGrid; ++gx) {
                const float ax = x0 + static_cast<float>(gx) * dx, bx = ax + dx;
                const float ay = y0 + static_cast<float>(gy) * dy, by = ay + dy;
                const float xy[6][2] = {{ax, ay}, {bx, ay}, {bx, by}, {ax, ay}, {bx, by}, {ax, by}};
                for (int j = 0; j < 6; ++j, ++v) pos->setXYZ(v, xy[j][0], xy[j][1], 0.f);
            }
        }
        pos->needsUpdate();
        g.setDrawRange(0, (k + 1) * kCellVerts);
    }

    std::shared_ptr<Mesh> makeStrip() {
        const int n = kStripCells * kCellVerts;
        auto geom = BufferGeometry::create();
        geom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(static_cast<size_t>(n) * 3, 0.f), 3));
        std::vector<float> nrm(static_cast<size_t>(n) * 3, 0.f);
        for (int i = 0; i < n; ++i) nrm[static_cast<size_t>(i) * 3 + 2] = 1.f;
        geom->setAttribute("normal", FloatBufferAttribute::create(std::move(nrm), 3));
        placeCell(*geom, 0);
        auto mesh = Mesh::create(geom, wireMat());
        mesh->frustumCulled = false;
        return mesh;
    }

}// namespace

int main() {

    auto gpu = vktest::openHeadlessVulkan("VulkanOverlayBlas_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;

    renderer.setDenoise(true);
    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;
    renderer.setOverlayLayer(static_cast<int>(kOverlayLayer));
    const bool rt = renderer.rayTracingSupported();

    const auto env = constantEnv(0.25f);
    Scene scene;
    scene.background = env;
    scene.environment = env;

    PerspectiveCamera camera(40.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 0, 4.4f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    auto wire = Mesh::create(SphereGeometry::create(kSphereR, 32, 16), wireMat());
    wire->position.set(kWireX, kRowY, 0.f);
    scene.add(wire);

    auto panelMat = MeshBasicMaterial::create();
    panelMat->color = Color(1, 0, 0);
    auto panel = Mesh::create(PlaneGeometry::create(0.9f, 0.9f), panelMat);
    panel->position.set(kPanelX, kRowY, 0.f);
    panel->layers.enable(kOverlayLayer);
    scene.add(panel);

    auto pairGeom = SphereGeometry::create(kSphereR, 32, 16);
    auto pairWire = Mesh::create(pairGeom, wireMat());
    pairWire->position.set(kPairX, kRowY, 0.f);
    scene.add(pairWire);
    auto solidMat = MeshStandardMaterial::create();
    solidMat->color = Color(0.05f, 0.1f, 0.9f);
    solidMat->roughness = 1.f;
    solidMat->metalness = 0.f;
    auto pairSolid = Mesh::create(pairGeom, solidMat);// added at phase 1
    pairSolid->position.set(kPairX, kSolidY, 0.f);

    auto strip = makeStrip();
    scene.add(strip);

    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };
    const auto samplePoint = [&](const std::vector<unsigned char>& px, float wx, float wy) {
        Vector3 p(wx, wy, 0.f);
        p.project(camera);
        const int x = static_cast<int>((p.x * 0.5f + 0.5f) * kW);
        const int y = static_cast<int>((1.f - (p.y * 0.5f + 0.5f)) * kH);
        return patch(px, x, y, 4);
    };
    const auto read = [&] {
        auto px = renderer.readRGBPixels();
        if (px.size() < static_cast<size_t>(kW) * kH * 3) {
            std::printf("readback too small: %zu\n", px.size());
            std::exit(1);
        }
        return px;
    };
    const auto report = [&](const std::vector<unsigned char>& px, const char* label) {
        const auto w = samplePoint(px, kWireX, kRowY), p = samplePoint(px, kPanelX, kRowY),
                   pw = samplePoint(px, kPairX, kRowY), ps = samplePoint(px, kPairX, kSolidY),
                   s0 = samplePoint(px, cellCenterX(0), kStripY),
                   s11 = samplePoint(px, cellCenterX(kStripCells - 1), kStripY);
        std::printf("%s\n  wire (%3.0f %3.0f %3.0f) panel (%3.0f %3.0f %3.0f) pair wire (%3.0f %3.0f %3.0f) "
                    "pair solid (%3.0f %3.0f %3.0f) strip[0] (%3.0f %3.0f %3.0f) strip[11] (%3.0f %3.0f %3.0f)\n",
                    label, w[0], w[1], w[2], p[0], p[1], p[2], pw[0], pw[1], pw[2], ps[0], ps[1], ps[2],
                    s0[0], s0[1], s0[2], s11[0], s11[1], s11[2]);
    };

    // Lidar beams from the eye through the four centres: they read the TLAS.
    const Vector3 eye(0.f, 0.f, 4.4f);
    const std::array<Vector3, 4> centres{Vector3(kWireX, kRowY, 0.f), Vector3(kPanelX, kRowY, 0.f),
                                         Vector3(kPairX, kRowY, 0.f), Vector3(kPairX, kSolidY, 0.f)};
    enum { kBeamWire, kBeamPanel, kBeamPairWire, kBeamPairSolid };
    std::vector<LidarBeam> beams;
    for (const auto& c : centres) beams.push_back({eye, (c - eye).normalize()});
    const auto lidarHits = [&] {
        std::vector<LidarReturn> rets;
        renderer.scanLidar(beams, rets);
        std::array<bool, 4> hit{};
        for (size_t i = 0; i < hit.size() && i < rets.size(); ++i) {
            hit[i] = rets[i].returnNo > 0 && rets[i].hitInstanceId >= 0;
        }
        std::printf("  lidar: wire %d panel %d pair wire %d pair solid %d\n", hit[kBeamWire], hit[kBeamPanel],
                    hit[kBeamPairWire], hit[kBeamPairSolid]);
        return hit;
    };

    int frame = 0;
    canvas.animate([&] {
        // The strip places one more cell on every third frame: each edit
        // uploads the soup's vertices and would have refit a BLAS.
        if (frame >= kFramesPerPhase && frame < kFramesPerPhase + 3 * (kStripCells - 1) &&
            (frame - kFramesPerPhase) % 3 == 0) {
            placeCell(*strip->geometry(), 1 + (frame - kFramesPerPhase) / 3);
        }
        renderer.render(scene, camera);
        ++frame;

        if (frame == kFramesPerPhase) {
            const auto px = read();
            report(px, "phase 1 (overlays alone):");
            check("the wireframe sphere draws", leads(samplePoint(px, kWireX, kRowY), 12.0) == 1);
            check("the overlay-layer panel draws", leads(samplePoint(px, kPanelX, kRowY), 40.0) == 0);
            check("the pair's wireframe draws", leads(samplePoint(px, kPairX, kRowY), 12.0) == 1);
            check("the pair's solid is not there yet", leads(samplePoint(px, kPairX, kSolidY), 12.0) == -1);
            check("the strip starts one cell long",
                  leads(samplePoint(px, cellCenterX(0), kStripY), 12.0) == 1 &&
                          leads(samplePoint(px, cellCenterX(kStripCells - 1), kStripY), 12.0) == -1);
            if (rt) {
                const auto h = lidarHits();
                check("no beam meets a wireframe or an overlay-layer mesh",
                      !h[kBeamWire] && !h[kBeamPanel] && !h[kBeamPairWire] && !h[kBeamPairSolid]);
            }
            scene.add(pairSolid);
            return;
        }

        if (frame == 2 * kFramesPerPhase) {
            const auto px = read();
            report(px, "phase 2 (the solid added, the strip grown):");
            check("the pair's solid draws", leads(samplePoint(px, kPairX, kSolidY), 12.0) == 2);
            check("the pair's wireframe still draws", leads(samplePoint(px, kPairX, kRowY), 12.0) == 1);
            check("the strip grew to twelve cells by edits on every third frame",
                  leads(samplePoint(px, cellCenterX(0), kStripY), 12.0) == 1 &&
                          leads(samplePoint(px, cellCenterX(kStripCells - 1), kStripY), 12.0) == 1);
            if (rt) {
                const auto h = lidarHits();
                check("the shared geometry's solid returns a beam, the overlays still none",
                      h[kBeamPairSolid] && !h[kBeamWire] && !h[kBeamPanel] && !h[kBeamPairWire]);
            } else {
                std::printf("  (no ray query: the lidar checks do not apply)\n");
            }
            const auto st = renderer.dynamicGeomStats();
            std::printf("  dynamic geometry: graduated %u, refits recorded %llu, full rebuilds %llu, drains %llu\n",
                        st.graduated, static_cast<unsigned long long>(st.refitsRecorded),
                        static_cast<unsigned long long>(st.fullRebuilds),
                        static_cast<unsigned long long>(st.drains));
            check("the strip's eleven edits recorded no BLAS refit", st.refitsRecorded == 0u);
            check("no edit drained the device", st.drains == 0u);
            if (vulkan::validationActive()) {
                check("no validation errors", vulkan::validationErrorCount() == 0u);
            }

            std::printf("overlay BLAS: %d failure(s)\n", failures);
            std::exit(failures == 0 ? 0 : 1);
        }
    });
    return 0;
}
