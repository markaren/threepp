// VulkanGeometryRefresh_test: in-place edits of an uploaded geometry reach
// the GPU, whichever attribute was edited.
//
// The in-place refresh re-uploaded positions and normals only: an edited
// vertex color never reached the GPU on either path, and an edited uv only on
// the drained (occasional) one, so a per-frame edit froze it at its first
// upload (the dune-buggy's ruts, 2026-10-15). Four quads, each its own
// geometry, white MeshStandardMaterial in a uniform environment:
//   [0] vertex colors red, edited ONCE to green (an occasional edit);
//   [1] vertex colors red, edited on 8 frames in a row (the record graduates
//       to the per-frame path) and left blue;
//   [2] textured with a red | blue map, uv on the red half, edited once to the
//       blue half;
//   [3] the same map repeating, uv on the blue half (in [0,1], so packed
//       unorm16 at build), edited once to u in [1.05, 1.2], the red half one
//       repeat on: the packed format cannot hold it, so the record must be
//       rebuilt rather than clamped to the u = 1 seam (half red, half blue).
// And one more below them:
//   [4] vertex colors green, its positions shifted 8 cm along x on every third
//       frame, ten times. An edit that misses three frames in a row used to
//       drain the device and rebuild the BLAS on every write (10 to 17 ms in
//       the dune-buggy); it now records into the frame command buffer: no
//       drain at all, no graduation, and the quad ends up 80 cm along.
//   ball  a blue 4096-triangle sphere that sits still until its auto-LOD chain
//       has landed, then is moved up once: an edit of a record with a chain
//       used to drain to free the chain (a rut chunk reused from a pool);
//       the chain is retired instead.
//   strip a row of twelve green quads along the top, one triangle soup
//       preallocated at capacity and drawn one quad long, that admits one more
//       quad on every third frame, eleven times: the live triangle count the
//       BLAS is rebuilt from grows past the count its storage was created for.
//       The refit used to build past the end of the structure (the SLAM
//       surface in spot_slam: VK_ERROR_DEVICE_LOST, or a neighbouring vertex
//       buffer drawn as a fan of garbage); the record moves to storage that
//       fits, and the renderer refuses the build when it does not.
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
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
    constexpr int kQuads = 5;
    constexpr float kRowY = 0.5f;
    constexpr float kMoverX = -1.2f, kMoverY = -0.9f;
    constexpr int kMoverEdits = 10;
    constexpr float kMoverStep = 0.08f;
    constexpr float kBallX = 1.3f, kBallY = -0.9f, kBallLift = 0.3f;
    constexpr int kBallEditFrame = kFramesPerPhase + 40;
    constexpr int kStripQuads = 12;
    constexpr float kStripX0 = -1.95f, kStripPitch = 0.33f, kStripW = 0.3f, kStripY = 1.3f, kStripH = 0.4f;

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

    // 8x1: the left half red, the right half blue.
    std::shared_ptr<Texture> redBlueMap() {
        constexpr int W = 8;
        std::vector<unsigned char> px(W * 4);
        for (int x = 0; x < W; ++x) {
            const bool left = x < W / 2;
            px[x * 4 + 0] = left ? 255 : 0;
            px[x * 4 + 1] = 0;
            px[x * 4 + 2] = left ? 0 : 255;
            px[x * 4 + 3] = 255;
        }
        Image img{std::move(px), static_cast<unsigned>(W), 1u};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::UnsignedByte;
        tex->generateMipmaps = false;
        tex->minFilter = Filter::Linear;
        tex->wrapS = TextureWrapping::Repeat;
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

    // Index of the dominant channel, or -1 when no channel leads the other two by 2x.
    int dominant(const std::array<double, 3>& c) {
        for (int i = 0; i < 3; ++i) {
            const double o1 = c[(i + 1) % 3], o2 = c[(i + 2) % 3];
            if (c[i] > 2.0 * o1 + 4.0 && c[i] > 2.0 * o2 + 4.0) return i;
        }
        return -1;
    }

    void shiftX(BufferGeometry& g, float dx) {
        auto* pos = g.getAttribute<float>("position");
        for (int i = 0; i < pos->count(); ++i) pos->setX(i, pos->getX(i) + dx);
        pos->needsUpdate();
    }

    void setColors(BufferGeometry& g, float r, float gr, float b) {
        auto* c = g.getAttribute<float>("color");
        for (int i = 0; i < c->count(); ++i) c->setXYZ(i, r, gr, b);
        c->needsUpdate();
    }

    // u in [u0, u1] across a PlaneGeometry(1, 1) (its x runs -0.5 .. 0.5).
    void setUvRange(BufferGeometry& g, float u0, float u1) {
        const auto* pos = g.getAttribute<float>("position");
        auto* uv = g.getAttribute<float>("uv");
        for (int i = 0; i < uv->count(); ++i) {
            const float u = pos->getX(i) + 0.5f;
            uv->setX(i, u0 + u * (u1 - u0));
        }
        uv->needsUpdate();
    }

    float stripCenterX(int k) { return kStripX0 + static_cast<float>(k) * kStripPitch + kStripW * 0.5f; }

    // Place quad k of the strip and admit it to the draw range. The vertices
    // of a quad not yet placed sit at the origin, outside the range.
    void growStrip(BufferGeometry& g, int k) {
        auto* pos = g.getAttribute<float>("position");
        const float x0 = kStripX0 + static_cast<float>(k) * kStripPitch, x1 = x0 + kStripW;
        const float y0 = kStripY - kStripH * 0.5f, y1 = kStripY + kStripH * 0.5f;
        const float xy[6][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y0}, {x1, y1}, {x0, y1}};
        for (int v = 0; v < 6; ++v) pos->setXYZ(k * 6 + v, xy[v][0], xy[v][1], 0.f);
        pos->needsUpdate();
        g.setDrawRange(0, (k + 1) * 6);
    }

    // kStripQuads quads of triangle soup, green vertex colors, every attribute
    // allocated at capacity, the first quad placed and drawn.
    std::shared_ptr<Mesh> makeStrip() {
        const int n = kStripQuads * 6;
        auto geom = BufferGeometry::create();
        geom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(n * 3, 0.f), 3));
        std::vector<float> nrm(static_cast<size_t>(n) * 3, 0.f);
        for (int i = 0; i < n; ++i) nrm[i * 3 + 2] = 1.f;
        geom->setAttribute("normal", FloatBufferAttribute::create(std::move(nrm), 3));
        std::vector<float> col(static_cast<size_t>(n) * 3, 0.f);
        for (int i = 0; i < n; ++i) col[i * 3 + 1] = 1.f;
        geom->setAttribute("color", FloatBufferAttribute::create(std::move(col), 3));
        growStrip(*geom, 0);
        auto mat = MeshStandardMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->roughness = 1.f;
        mat->metalness = 0.f;
        mat->vertexColors = true;
        auto mesh = Mesh::create(geom, mat);
        mesh->frustumCulled = false;
        return mesh;
    }

    std::shared_ptr<Mesh> makeQuad(float x, bool vertexColors) {
        auto geom = PlaneGeometry::create(1.f, 1.f);
        auto mat = MeshStandardMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->roughness = 1.f;
        mat->metalness = 0.f;
        if (vertexColors) {
            mat->vertexColors = true;
            const int n = geom->getAttribute<float>("position")->count();
            geom->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(n * 3, 1.f), 3));
        }
        auto mesh = Mesh::create(geom, mat);
        mesh->position.x = x;
        return mesh;
    }

}// namespace

int main() {

    auto gpu = vktest::openHeadlessVulkan("VulkanGeometryRefresh_test", kW, kH);
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

    const auto env = constantEnv(0.5f);
    Scene scene;
    scene.background = env;
    scene.environment = env;

    PerspectiveCamera camera(40.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 0, 4.4f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    std::array<std::shared_ptr<Mesh>, kQuads> quads;
    for (int i = 0; i < 4; ++i) {
        quads[i] = makeQuad((static_cast<float>(i) - 1.5f) * 1.2f, i < 2);
        quads[i]->position.y = kRowY;
        scene.add(quads[i]);
    }
    quads[4] = makeQuad(kMoverX, true);
    quads[4]->position.y = kMoverY;
    scene.add(quads[4]);
    setColors(*quads[4]->geometry(), 0, 1, 0);
    auto ballMat = MeshStandardMaterial::create();
    ballMat->color = Color(0.05f, 0.1f, 0.9f);
    ballMat->roughness = 1.f;
    auto ball = Mesh::create(SphereGeometry::create(0.4f, 64, 32), ballMat);// 4096 triangles
    ball->position.set(kBallX, kBallY, 0.f);
    scene.add(ball);
    auto strip = makeStrip();
    scene.add(strip);
    setColors(*quads[0]->geometry(), 1, 0, 0);
    setColors(*quads[1]->geometry(), 1, 0, 0);
    for (int i = 2; i < 4; ++i) {
        auto mat = std::dynamic_pointer_cast<MeshStandardMaterial>(quads[i]->material());
        mat->map = redBlueMap();
    }
    setUvRange(*quads[2]->geometry(), 0.05f, 0.2f);
    setUvRange(*quads[3]->geometry(), 0.8f, 0.95f);

    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };
    const auto samplePoint = [&](const std::vector<unsigned char>& px, float wx, float wy) {
        Vector3 p(wx, wy, 0.f);
        p.project(camera);
        const int x = static_cast<int>((p.x * 0.5f + 0.5f) * kW);
        const int y = static_cast<int>((1.f - (p.y * 0.5f + 0.5f)) * kH);
        return patch(px, x, y, 4);
    };
    const auto sample = [&](const std::vector<unsigned char>& px, int i) {
        return samplePoint(px, quads[i]->position.x, quads[i]->position.y);
    };
    const auto report = [&](const std::vector<unsigned char>& px, const char* label) {
        std::printf("%s\n", label);
        for (int i = 0; i < kQuads; ++i) {
            const auto c = sample(px, i);
            std::printf("  quad[%d] = (%5.1f %5.1f %5.1f)\n", i, c[0], c[1], c[2]);
        }
    };
    const auto read = [&] {
        auto px = renderer.readRGBPixels();
        if (px.size() < static_cast<size_t>(kW) * kH * 3) {
            std::printf("readback too small: %zu\n", px.size());
            std::exit(1);
        }
        return px;
    };

    int frame = 0;
    canvas.animate([&] {
        // Quad 1 is edited on 8 frames in a row, ending blue: the record
        // graduates to the per-frame path on the way.
        if (frame >= kFramesPerPhase && frame < kFramesPerPhase + 8) {
            const bool last = frame == kFramesPerPhase + 7;
            setColors(*quads[1]->geometry(), last ? 0.f : 1.f, 0.f, last ? 1.f : 0.f);
        }
        // The ball moves once, alone in its frame: quad[3]'s edit at phase 1
        // re-admits every dirty record through a structural rebuild, which
        // would hide the case. By now its chain has landed again.
        if (frame == kBallEditFrame) {
            const auto lod = renderer.autoLodStats();
            std::printf("  auto-LOD before the ball's edit: %u chain(s) ready, %u queued\n",
                        lod.chainsReady, lod.chainsQueued);
            check("the ball's auto-LOD chain landed (the case under test)", lod.chainsReady >= 1u);
            auto* pos = ball->geometry()->getAttribute<float>("position");
            for (int k = 0; k < pos->count(); ++k) pos->setY(k, pos->getY(k) + kBallLift);
            pos->needsUpdate();
        }
        // Quad 4 moves on every third frame.
        if (frame >= kFramesPerPhase && frame < kFramesPerPhase + 3 * kMoverEdits &&
            (frame - kFramesPerPhase) % 3 == 0) {
            shiftX(*quads[4]->geometry(), kMoverStep);
        }
        // The strip admits one more quad on every third frame: each edit
        // grows the live triangle count its BLAS is rebuilt from.
        if (frame >= kFramesPerPhase && frame < kFramesPerPhase + 3 * (kStripQuads - 1) &&
            (frame - kFramesPerPhase) % 3 == 0) {
            growStrip(*strip->geometry(), 1 + (frame - kFramesPerPhase) / 3);
        }
        renderer.render(scene, camera);
        ++frame;

        if (frame == kFramesPerPhase) {
            const auto px = read();
            report(px, "phase 1 (as built):");
            check("quad[0] vertex colors red", dominant(sample(px, 0)) == 0);
            check("quad[1] vertex colors red", dominant(sample(px, 1)) == 0);
            check("quad[2] uv on the red half", dominant(sample(px, 2)) == 0);
            check("quad[3] uv on the blue half", dominant(sample(px, 3)) == 2);
            check("quad[4] green where it starts", dominant(sample(px, 4)) == 1);
            check("ball not yet at its lifted spot", dominant(samplePoint(px, kBallX, kBallY + 0.55f)) == -1);
            check("strip starts one quad long",
                  dominant(samplePoint(px, stripCenterX(0), kStripY)) == 1 &&
                          dominant(samplePoint(px, stripCenterX(kStripQuads - 1), kStripY)) == -1);

            setColors(*quads[0]->geometry(), 0, 1, 0);
            setUvRange(*quads[2]->geometry(), 0.8f, 0.95f);
            setUvRange(*quads[3]->geometry(), 1.05f, 1.2f);
            return;
        }

        if (frame == 2 * kFramesPerPhase) {
            const auto px = read();
            report(px, "phase 2 (edited):");
            check("quad[0] recolored green by one edit", dominant(sample(px, 0)) == 1);
            check("quad[1] recolored blue by per-frame edits", dominant(sample(px, 1)) == 2);
            check("quad[2] uv moved to the blue half", dominant(sample(px, 2)) == 2);
            check("quad[3] uv past 1 not clamped by the packed format", dominant(sample(px, 3)) == 0);
            const float moved = kMoverX + kMoverEdits * kMoverStep;
            check("quad[4] moved 80 cm by edits on every third frame",
                  dominant(samplePoint(px, moved + 0.3f, kMoverY)) == 1 &&
                          dominant(samplePoint(px, kMoverX - 0.35f, kMoverY)) == -1);

            const auto st = renderer.dynamicGeomStats();
            std::printf("  dynamic geometry: graduated %u, refits recorded %llu, drains %llu\n", st.graduated,
                        static_cast<unsigned long long>(st.refitsRecorded),
                        static_cast<unsigned long long>(st.drains));
            check("quad[1] graduated, quad[4] (every third frame) did not", st.graduated == 1u);
            check("no in-place edit drained the device (the ball's chain included)", st.drains == 0u);
            check("ball moved up by one edit", dominant(samplePoint(px, kBallX, kBallY + 0.55f)) == 2);
            check("strip grew to twelve quads, its BLAS storage with it",
                  dominant(samplePoint(px, stripCenterX(0), kStripY)) == 1 &&
                          dominant(samplePoint(px, stripCenterX(kStripQuads - 1), kStripY)) == 1);
            if (vulkan::validationActive()) {
                check("no validation errors", vulkan::validationErrorCount() == 0u);
            }

            std::printf("geometry refresh: %d failure(s)\n", failures);
            std::exit(failures == 0 ? 0 : 1);
        }
    });
    return 0;
}
