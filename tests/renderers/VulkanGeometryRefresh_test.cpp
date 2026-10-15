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
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
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
    constexpr int kQuads = 4;

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
    for (int i = 0; i < kQuads; ++i) {
        const float x = (static_cast<float>(i) - 1.5f) * 1.2f;
        quads[i] = makeQuad(x, i < 2);
        scene.add(quads[i]);
    }
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
    const auto sample = [&](const std::vector<unsigned char>& px, int i) {
        Vector3 p = quads[i]->position;
        p.project(camera);
        const int x = static_cast<int>((p.x * 0.5f + 0.5f) * kW);
        const int y = static_cast<int>((1.f - (p.y * 0.5f + 0.5f)) * kH);
        return patch(px, x, y, 4);
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
        renderer.render(scene, camera);
        ++frame;

        if (frame == kFramesPerPhase) {
            const auto px = read();
            report(px, "phase 1 (as built):");
            check("quad[0] vertex colors red", dominant(sample(px, 0)) == 0);
            check("quad[1] vertex colors red", dominant(sample(px, 1)) == 0);
            check("quad[2] uv on the red half", dominant(sample(px, 2)) == 0);
            check("quad[3] uv on the blue half", dominant(sample(px, 3)) == 2);
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

            const auto st = renderer.dynamicGeomStats();
            std::printf("  dynamic geometry: graduated %u, refits recorded %llu\n", st.graduated,
                        static_cast<unsigned long long>(st.refitsRecorded));
            check("quad[1] graduated to the per-frame path", st.graduated >= 1u);

            std::printf("geometry refresh: %d failure(s)\n", failures);
            std::exit(failures == 0 ? 0 : 1);
        }
    });
    return 0;
}
