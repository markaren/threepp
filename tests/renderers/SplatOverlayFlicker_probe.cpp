// SplatOverlayFlicker_probe — measurement harness, not an asserted test.
//
// Reproduces the editor's baked-surface preview: a depth-tested LineSegments
// shell pushed one voxel off a baked splat surface, viewed at a grazing angle
// from a distance, under the renderer's default TAA. Renders N frames from a
// STATIC camera and counts, pixel by pixel, how often the line mask flips
// between consecutive frames. A stable overlay flips ~0 pixels; the shimmer
// under investigation flips many, every frame.
//
//   SplatOverlayFlicker_probe                the editor's preview: exempt via
//                                            kSplatUnoccludedOverlayLayer
//   SplatOverlayFlicker_probe --stamped      NOT exempt: depth-tested against
//                                            the splat stamp (the flicker)
//   SplatOverlayFlicker_probe --no-depthtest the depthTest=false baseline
//   SplatOverlayFlicker_probe --push X       push X out along vertex normals
//   SplatOverlayFlicker_probe --near         framed instead of grazing/distant
//
// Exits 42 without a Vulkan GPU, like the other Vulkan probes.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/splats/SplatSurface.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 960, kH = 600;
    constexpr int kSkipCode = 42;
    constexpr float kVoxel = 0.05f;

    // The surface test's plane: a slab of splats one splat thick at y = 0.
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

    // The editor preview's geometry, verbatim: deduplicated edges, each vertex
    // pushed along its area-weighted normal.
    std::shared_ptr<LineSegments> makePreview(const splats::SurfaceMesh& mesh,
                                              float push, bool depthTest) {
        const auto vertexCount = mesh.positions.size() / 3;
        std::vector<float> pushed = mesh.positions;
        if (push > 0.f && !pushed.empty()) {
            std::vector<float> normals(pushed.size(), 0.f);
            for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
                const auto i0 = mesh.indices[i], i1 = mesh.indices[i + 1], i2 = mesh.indices[i + 2];
                if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;
                const float* p0 = &mesh.positions[i0 * 3];
                const float* p1 = &mesh.positions[i1 * 3];
                const float* p2 = &mesh.positions[i2 * 3];
                const float ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
                const float vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
                const float nx = uy * vz - uz * vy;
                const float ny = uz * vx - ux * vz;
                const float nz = ux * vy - uy * vx;
                for (const auto index : {i0, i1, i2}) {
                    normals[index * 3] += nx;
                    normals[index * 3 + 1] += ny;
                    normals[index * 3 + 2] += nz;
                }
            }
            for (std::size_t v = 0; v < vertexCount; ++v) {
                const float nx = normals[v * 3], ny = normals[v * 3 + 1], nz = normals[v * 3 + 2];
                const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len <= 0.f) continue;
                const float s = push / len;
                pushed[v * 3] += nx * s;
                pushed[v * 3 + 1] += ny * s;
                pushed[v * 3 + 2] += nz * s;
            }
        }

        std::vector<float> positions;
        positions.reserve(mesh.indices.size() * 3);
        std::unordered_set<std::uint64_t> seen;
        seen.reserve(mesh.indices.size());
        const auto edge = [&](std::uint32_t a, std::uint32_t b) {
            if (a >= vertexCount || b >= vertexCount) return;
            const std::uint64_t key = a < b
                                              ? (static_cast<std::uint64_t>(a) << 32) | b
                                              : (static_cast<std::uint64_t>(b) << 32) | a;
            if (!seen.insert(key).second) return;
            positions.insert(positions.end(),
                             {pushed[a * 3], pushed[a * 3 + 1], pushed[a * 3 + 2],
                              pushed[b * 3], pushed[b * 3 + 1], pushed[b * 3 + 2]});
        };
        for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            const auto i0 = mesh.indices[i], i1 = mesh.indices[i + 1], i2 = mesh.indices[i + 2];
            edge(i0, i1);
            edge(i1, i2);
            edge(i2, i0);
        }

        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        auto material = LineBasicMaterial::create();
        material->color.setRGB(1.f, 0.f, 0.f);// pure red: nothing else in the scene is
        material->toneMapped = false;
        material->depthTest = depthTest;
        auto lines = LineSegments::create(geometry, material);
        lines->matrixAutoUpdate = false;
        lines->frustumCulled = false;
        return lines;
    }

    // A pixel is "line" when red dominates: the splats are grey, the sky is
    // whatever background the renderer clears to, the lines are pure red.
    inline bool isLine(const unsigned char* p) {
        return p[0] > 96 && p[0] > 2 * p[1] && p[0] > 2 * p[2];
    }

}// namespace

int main(int argc, char** argv) {

    bool depthTest = true;
    bool exempt = true;
    bool nearView = false;
    float push = 0.f;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-depthtest") == 0) depthTest = false;
        else if (std::strcmp(argv[i], "--stamped") == 0) exempt = false;
        else if (std::strcmp(argv[i], "--near") == 0) nearView = true;
        else if (std::strcmp(argv[i], "--push") == 0 && i + 1 < argc) push = std::stof(argv[++i]);
    }

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("SplatOverlayFlicker_probe").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    VulkanRenderer& renderer = *rendererPtr;
    renderer.setRenderScale(1.f);
    renderer.setDlss(false);
    renderer.setFsr(false);

    auto scene = Scene::create();
    auto cloud = SplatCloud::create(makePlane());
    scene->add(cloud);

    splats::SurfaceBakeOptions opts;
    opts.voxelSize = kVoxel;
    opts.poseCount = 16;
    const auto surface = splats::bakeSurface(renderer, *cloud, opts);
    if (surface.empty()) {
        std::printf("[FAIL] bake produced no surface\n");
        return 1;
    }
    std::printf("baked %zu tris, voxel %.3f, push %.3f, depthTest %d, exempt %d\n",
                surface.triangleCount(), surface.stats.voxelSize, push, depthTest ? 1 : 0,
                exempt ? 1 : 0);

    auto lines = makePreview(surface, push, depthTest);
    if (exempt) lines->layers.enable(VulkanRenderer::kSplatUnoccludedOverlayLayer);
    scene->add(lines);

    // A wall between the camera and the LEFT half of the scan: the depth-tested
    // preview must vanish behind it (the bug the depth test exists to fix), and
    // --no-depthtest shows through it. Dark grey, so the line mask cannot
    // mistake it.
    auto wall = Mesh::create(BoxGeometry::create(2.5f, 2.f, 0.3f),
                             MeshBasicMaterial::create({{"color", Color(0x202020)}}));
    wall->position.set(-1.4f, 0.6f, 6.f);
    scene->add(wall);

    auto camera = PerspectiveCamera::create(60.f, float(kW) / float(kH), 0.1f, 100.f);
    if (nearView) {
        camera->position.set(0.f, 3.f, 5.f);
    } else {
        // Grazing and distant: the plane spans z in [-2, 2]; from z = 14 at
        // eye height 0.6 the far edge is ~16 m out at ~2 degrees elevation.
        camera->position.set(0.f, 0.6f, 14.f);
    }
    camera->lookAt(Vector3(0.f, 0.f, 0.f));
    camera->updateMatrixWorld();

    for (int i = 0; i < 80; ++i) renderer.render(*scene, *camera);

    constexpr int kFrames = 32;
    std::vector<std::vector<unsigned char>> masks;
    masks.reserve(kFrames);
    std::size_t lastLinePixels = 0;
    for (int f = 0; f < kFrames; ++f) {
        renderer.render(*scene, *camera);
        const auto pixels = renderer.readRGBPixels();
        std::vector<unsigned char> mask(kW * kH, 0);
        std::size_t count = 0;
        for (int p = 0; p < kW * kH; ++p) {
            if (isLine(&pixels[p * 3])) {
                mask[p] = 1;
                ++count;
            }
        }
        lastLinePixels = count;
        masks.push_back(std::move(mask));
    }

    // Occlusion check: with the depth test turned off on the same material the
    // wall stops hiding anything, so the count rises by exactly the pixels the
    // wall was occluding. Zero difference would mean the depth test is not
    // actually testing.
    const auto writePpm = [&](const char* path, const std::vector<unsigned char>& px) {
        if (FILE* f = std::fopen(path, "wb")) {
            std::fprintf(f, "P6\n%d %d\n255\n", kW, kH);
            std::fwrite(px.data(), 1, px.size(), f);
            std::fclose(f);
        }
    };
    std::size_t linePixelsNoTest = lastLinePixels;
    if (depthTest) {
        {
            const auto px = renderer.readRGBPixels();
            writePpm("flicker_tested.ppm", px);
        }
        if (const auto material = lines->material()) material->depthTest = false;
        for (int i = 0; i < 8; ++i) renderer.render(*scene, *camera);
        const auto pixels = renderer.readRGBPixels();
        writePpm("flicker_untested.ppm", pixels);
        linePixelsNoTest = 0;
        for (int p = 0; p < kW * kH; ++p)
            if (isLine(&pixels[p * 3])) ++linePixelsNoTest;
        if (const auto material = lines->material()) material->depthTest = true;
    }
    std::printf("line pixels: %zu tested, %zu untested -> %lld hidden by the wall\n",
                lastLinePixels, linePixelsNoTest,
                static_cast<long long>(linePixelsNoTest) - static_cast<long long>(lastLinePixels));

    // Per-pixel flips between consecutive frames, and the set of pixels that
    // EVER flipped — the visible size of the shimmering region.
    std::uint64_t flips = 0;
    std::vector<unsigned char> everFlipped(kW * kH, 0);
    for (int f = 1; f < kFrames; ++f) {
        for (int p = 0; p < kW * kH; ++p) {
            if (masks[f][p] != masks[f - 1][p]) {
                ++flips;
                everFlipped[p] = 1;
            }
        }
    }
    std::size_t flickerRegion = 0;
    for (const auto v : everFlipped) flickerRegion += v;

    std::printf("line pixels (last frame): %zu\n", lastLinePixels);
    std::printf("mask flips over %d frame pairs: %llu  (%.1f per pair)\n",
                kFrames - 1, static_cast<unsigned long long>(flips),
                double(flips) / double(kFrames - 1));
    std::printf("pixels that ever flipped: %zu  (%.1f%% of line pixels)\n",
                flickerRegion,
                100.0 * double(flickerRegion) / double(std::max<std::size_t>(1, lastLinePixels)));
    return 0;
}
