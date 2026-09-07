
#include "threepp/splats/SplatSurface.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/extras/pointcloud/MarchingCubes.hpp"// the standard MC tables
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>

using namespace threepp;

namespace {

    constexpr int kB = 8;         // voxels per block edge
    constexpr int kBV = kB * kB * kB;
    // Two renders per pose: the first moves the camera (which invalidates the
    // cloud's sort), the second is the frame that is read back, so the AOV can
    // never be the previous pose's.
    constexpr int kFramesPerPose = 2;
    // Depth-summary tile edge, in pixels. Only the carve pass's fast paths read
    // it: bigger tiles summarise more coarsely (fewer blocks classify) and cost
    // less to build, and 16 is where a 512x512 capture's summary is 1024 entries.
    constexpr int kTile = 16;

    struct Block {

        int32_t bx{}, by{}, bz{};
        std::array<float, kBV> tsdf{};// truncated signed distance / truncation
        std::array<float, kBV> w{};
    };

    // Floor division / modulo by the block edge, negatives included: a shift
    // would do it on every compiler we build with, but not by the standard.
    inline int blockOf(int v) { return (v >= 0) ? (v / kB) : -((-v + kB - 1) / kB); }

    inline uint64_t blockKey(int bx, int by, int bz) {
        // 21 bits an axis, biased: a block index past +-2^20 is 8 million
        // voxels from the origin and no scan reaches it.
        const auto p = [](int v) { return static_cast<uint64_t>(static_cast<uint32_t>(v + (1 << 20)) & 0x1FFFFFu); };
        return (p(bx) << 42) | (p(by) << 21) | p(bz);
    }

    // A welded marching-cubes vertex is identified by the LOW node of the edge
    // it sits on plus the edge's axis: 20 bits an axis leaves room for the two
    // axis bits inside 64.
    inline uint64_t edgeKey(int x, int y, int z, int axis) {
        const auto p = [](int v) { return static_cast<uint64_t>(static_cast<uint32_t>(v + (1 << 19)) & 0xFFFFFu); };
        return (((p(x) << 40) | (p(y) << 20) | p(z)) << 2) | static_cast<uint64_t>(axis);
    }

    struct Volume {

        std::unordered_map<uint64_t, uint32_t> index;
        std::vector<Block> blocks;
        size_t maxBlocks{~size_t(0)};
        uint64_t refused{0};

        Block* find(uint64_t key) {
            const auto it = index.find(key);
            return it == index.end() ? nullptr : &blocks[it->second];
        }

        // Refuses past the budget rather than growing until the machine gives
        // out. Reserving geometrically but CLAMPED to the cap keeps the vector's
        // own growth from overshooting it on the last doubling.
        void ensure(uint64_t key, int bx, int by, int bz) {
            if (index.find(key) != index.end()) return;
            if (blocks.size() >= maxBlocks) {
                ++refused;
                return;
            }
            if (blocks.size() == blocks.capacity()) {
                const size_t want = std::max<size_t>(64, blocks.capacity() + blocks.capacity() / 2);
                blocks.reserve(std::min(maxBlocks, want));
            }
            index.emplace(key, static_cast<uint32_t>(blocks.size()));
            blocks.push_back(Block{});
            Block& b = blocks.back();
            b.bx = bx;
            b.by = by;
            b.bz = bz;
        }
    };

    // The robust fit examples/objects/gaussian_splats.cpp frames with:
    // component-wise median centre, 90th-percentile radius about it. A bounding
    // box centres on whichever photogrammetry outlier is furthest out.
    struct Fit {
        Vector3 center;
        float radius{1.f};
        Vector3 half{1.f, 1.f, 1.f};// per-axis 90th percentile about the centre
    };

    Fit robustFit(const SplatData& data, const Matrix4& toWorld) {

        Fit fit;
        if (data.count() == 0) return fit;

        std::vector<Vector3> pts;
        pts.reserve(data.count());
        for (const auto& m : data.means) {
            Vector3 p = m;
            p.applyMatrix4(toWorld);
            pts.push_back(p);
        }

        auto median = [](std::vector<float> v) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        };
        std::vector<float> xs, ys, zs;
        xs.reserve(pts.size());
        ys.reserve(pts.size());
        zs.reserve(pts.size());
        for (const auto& p : pts) {
            xs.push_back(p.x);
            ys.push_back(p.y);
            zs.push_back(p.z);
        }
        fit.center.set(median(xs), median(ys), median(zs));

        auto pct90 = [](std::vector<float>& v) {
            std::sort(v.begin(), v.end());
            return v[static_cast<size_t>(0.90 * static_cast<double>(v.size() - 1))];
        };
        std::vector<float> radii, hx, hy, hz;
        radii.reserve(pts.size());
        hx.reserve(pts.size());
        hy.reserve(pts.size());
        hz.reserve(pts.size());
        for (const auto& p : pts) {
            radii.push_back(p.distanceTo(fit.center));
            hx.push_back(std::abs(p.x - fit.center.x));
            hy.push_back(std::abs(p.y - fit.center.y));
            hz.push_back(std::abs(p.z - fit.center.z));
        }
        fit.radius = pct90(radii);
        if (!(fit.radius > 0.f)) fit.radius = 1.f;
        fit.half.set(pct90(hx), pct90(hy), pct90(hz));
        return fit;
    }

    std::vector<splats::BakePose> orbitPoses(const Fit& fit, int count, float distance) {

        std::vector<splats::BakePose> out;
        count = std::max(4, count);
        const float d = distance > 0.f ? distance : fit.radius * 2.2f;

        const float flat = std::max(fit.half.x, fit.half.z);
        if (fit.half.y < 0.5f * flat) {

            // Wider than tall: a sphere of poses spends half its budget under
            // the floor. A ring around it plus a top-down grid, instead.
            const int top = std::max(4, count / 4);
            const int ring = std::max(4, count - top);
            for (int i = 0; i < ring; ++i) {
                const float a = 2.f * math::PI * static_cast<float>(i) / static_cast<float>(ring);
                const float el = 35.f * math::DEG2RAD;
                splats::BakePose p;
                p.position.set(fit.center.x + d * std::cos(el) * std::cos(a),
                               fit.center.y + d * std::sin(el),
                               fit.center.z + d * std::cos(el) * std::sin(a));
                p.target = fit.center;
                out.push_back(p);
            }
            const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(top))));
            for (int j = 0; j < side; ++j)
                for (int i = 0; i < side; ++i) {
                    const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(side) * 2.f - 1.f;
                    const float v = (static_cast<float>(j) + 0.5f) / static_cast<float>(side) * 2.f - 1.f;
                    splats::BakePose p;
                    p.target.set(fit.center.x + u * flat * 0.6f, fit.center.y,
                                 fit.center.z + v * flat * 0.6f);
                    p.position.set(p.target.x, fit.center.y + d, p.target.z);
                    p.up.set(0.f, 0.f, -1.f);// looking straight down: y-up is degenerate
                    out.push_back(p);
                }
            return out;
        }

        // Compact scan: a Fibonacci sphere looking inward.
        const float golden = math::PI * (3.f - std::sqrt(5.f));
        for (int i = 0; i < count; ++i) {
            const float z = 1.f - 2.f * (static_cast<float>(i) + 0.5f) / static_cast<float>(count);
            const float r = std::sqrt(std::max(0.f, 1.f - z * z));
            const float phi = golden * static_cast<float>(i);
            splats::BakePose p;
            p.position.set(fit.center.x + d * r * std::cos(phi),
                           fit.center.y + d * z,
                           fit.center.z + d * r * std::sin(phi));
            p.target = fit.center;
            // Straight up or down would give lookAt a degenerate up vector.
            if (std::abs(z) > 0.999f) p.up.set(0.f, 0.f, -1.f);
            out.push_back(p);
        }
        return out;
    }

    // Van der Corput radical inverse: the deterministic, INDEX-DERIVED sequence
    // the interior stations step along. The bake's determinism contract
    // (SplatSurface.hpp) forbids an RNG, and it does not need one — the pose
    // list has to be a pure function of (fit, count) and nothing else.
    inline float radicalInverse(uint32_t i, uint32_t base) {
        const float inv = 1.f / static_cast<float>(base);
        float f = inv, r = 0.f;
        while (i) {
            r += static_cast<float>(i % base) * f;
            i /= base;
            f *= inv;
        }
        return r;
    }

    // Cameras INSIDE the scan, looking out. The three numbers below are
    // properties of that geometry rather than tastes, which is why they are
    // constants here and not options:
    //
    // kInteriorFov — from one point the directions have to OVERLAP or the TSDF's
    // weight floor (2 poses per voxel, SurfaceBakeOptions::weightFloor) is never
    // met: a fan of narrow cones from a station tiles the sphere without any
    // patch being seen twice, and the whole capture meshes to nothing. At 90
    // degrees each direction's footprint is about 2.1 sr against the sphere's
    // 12.6, so eight of them cover it with the overlap the floor needs.
    // kInteriorMinDirs is the other half of that argument.
    //
    // kInteriorJitter — station offset as a fraction of the fit's PER-AXIS half
    // extents: far enough that a second station sees round the furniture the
    // first one was behind, near enough that no station can end up inside a wall
    // (the half extents are 90th percentiles, so 0.35 of one is well short of
    // the surface it describes).
    constexpr float kInteriorFov = 90.f;
    constexpr float kInteriorJitter = 0.35f;
    constexpr int kInteriorMinDirs = 8;

    std::vector<splats::BakePose> interiorPoses(const Fit& fit, int count) {

        std::vector<splats::BakePose> out;
        count = std::max(kInteriorMinDirs, count);
        // At least two stations always: one is the centre, and a room is exactly
        // the subject where a single viewpoint has things standing in front of it.
        const int stations = std::clamp(1 + count / 24, 2, 8);
        const int dirs = std::max(kInteriorMinDirs, count / stations);

        const float golden = math::PI * (3.f - std::sqrt(5.f));
        for (int s = 0; s < stations; ++s) {

            Vector3 pos = fit.center;
            if (s > 0) {
                // Station 0 IS the centre; the rest walk a Halton sequence,
                // symmetric about it. Symmetric and not one-sided because
                // neither direction guards an invariant here — both are inside.
                pos.x += (2.f * radicalInverse(static_cast<uint32_t>(s), 2) - 1.f) * kInteriorJitter * fit.half.x;
                pos.y += (2.f * radicalInverse(static_cast<uint32_t>(s), 3) - 1.f) * kInteriorJitter * fit.half.y;
                pos.z += (2.f * radicalInverse(static_cast<uint32_t>(s), 5) - 1.f) * kInteriorJitter * fit.half.z;
            }

            for (int i = 0; i < dirs; ++i) {
                // The orbit's own Fibonacci arithmetic with the ENDPOINTS kept:
                // z runs 1 .. -1 inclusive instead of stepping inside the poles,
                // so one direction looks straight up and one straight down. A
                // room's floor is what a robot stands on and its ceiling is what
                // a lidar sees first; the pole-avoiding form points at neither.
                const float z = (dirs > 1)
                                        ? 1.f - 2.f * static_cast<float>(i) / static_cast<float>(dirs - 1)
                                        : 0.f;
                const float r = std::sqrt(std::max(0.f, 1.f - z * z));
                const float phi = golden * static_cast<float>(i);
                splats::BakePose p;
                p.position = pos;
                p.target.set(pos.x + r * std::cos(phi), pos.y + z, pos.z + r * std::sin(phi));
                p.fov = kInteriorFov;
                // Straight up or down would give lookAt a degenerate up vector.
                if (std::abs(z) > 0.999f) p.up.set(0.f, 0.f, -1.f);
                out.push_back(p);
            }
        }
        return out;
    }

    // The coverage-gate defense. The median depth is the transmittance-0.5
    // crossing; as coverage falls TOWARD that gate the crossing degenerates to
    // the last contributing splat and lands far behind the surface (measured up
    // to 2 world units on VulkanSplat_test's cloud, commit 1b5d509e). The AOV
    // exports no coverage channel, so coverage is inferred: it falls at the
    // cloud's silhouette fringe, and a sample sitting behind its neighbours is
    // the failure's signature. Both drops are one-signed and both are counted.
    void guardDepth(std::vector<float>& depth, int w, int h, int erode, float behindLimit,
                    uint64_t& droppedFringe, uint64_t& droppedOutlier) {

        const auto n = static_cast<size_t>(w) * static_cast<size_t>(h);
        if (erode > 0) {
            std::vector<uint8_t> keep(n, 0);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const size_t i = static_cast<size_t>(y) * w + x;
                    if (!(depth[i] > 0.f)) continue;
                    bool ok = true;
                    for (int dy = -erode; dy <= erode && ok; ++dy)
                        for (int dx = -erode; dx <= erode && ok; ++dx) {
                            const int nx = x + dx, ny = y + dy;
                            if (nx < 0 || ny < 0 || nx >= w || ny >= h) { ok = false; break; }
                            if (!(depth[static_cast<size_t>(ny) * w + nx] > 0.f)) ok = false;
                        }
                    keep[i] = ok ? 1 : 0;
                }
            for (size_t i = 0; i < n; ++i)
                if (depth[i] > 0.f && !keep[i]) {
                    depth[i] = 0.f;
                    ++droppedFringe;
                }
        }

        if (behindLimit > 0.f) {
            const std::vector<float> src = depth;
            float win[9];
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const size_t i = static_cast<size_t>(y) * w + x;
                    if (!(src[i] > 0.f)) continue;
                    int m = 0;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = x + dx, ny = y + dy;
                            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                            const float d = src[static_cast<size_t>(ny) * w + nx];
                            if (d > 0.f) win[m++] = d;
                        }
                    std::sort(win, win + m);
                    if (src[i] - win[m / 2] > behindLimit) {
                        depth[i] = 0.f;
                        ++droppedOutlier;
                    }
                }
        }
    }

}// namespace

namespace threepp::splats {

    SurfaceMesh bakeSurface(VulkanRenderer& renderer, SplatCloud& cloud, const SurfaceBakeOptions& options) {

        using clock = std::chrono::steady_clock;
        SurfaceMesh out;
        if (cloud.splatCount() == 0) return out;

        cloud.updateMatrixWorld(true);
        const Fit fit = robustFit(cloud.data(), *cloud.matrixWorld);

        const float voxel = options.voxelSize > 0.f
                                    ? options.voxelSize
                                    : std::clamp(fit.radius / 256.f, 0.005f, 0.10f);
        const float trunc = options.truncation > 0.f ? options.truncation
                                                     : std::max(1e-4f, options.truncationVoxels) * voxel;
        out.stats.voxelSize = voxel;
        out.stats.truncation = trunc;

        const bool interior = options.poseSet == SurfaceBakeOptions::PoseSet::Interior;

        // The allocation gate. Derived from the pose distance rather than from
        // the far plane, because what a pose is CLOSE ENOUGH to fuse is the
        // subject it was placed around; the far plane's job is only to let the
        // subject's own back side render.
        //
        // Standing INSIDE, that argument has no distance to make: the station is
        // at the centre of the subject, so the fit's own extents are the only
        // scale in the problem.
        const float poseDist = options.poseDistance > 0.f ? options.poseDistance : fit.radius * 2.2f;
        const float maxDepth = options.maxDepth > 0.f
                                       ? options.maxDepth
                                       : (interior ? 2.5f * fit.radius : 2.5f * poseDist);
        out.stats.maxDepth = maxDepth;

        const std::vector<BakePose> poses =
                options.poses.empty()
                        ? (interior ? interiorPoses(fit, options.poseCount)
                                    : orbitPoses(fit, options.poseCount, options.poseDistance))
                        : options.poses;
        out.stats.poses = static_cast<int>(poses.size());
        if (poses.empty()) return out;

        // The cloud renders alone: any other geometry in its own scene would
        // depth-test against it and punch holes in the capture. Its world
        // transform rides along on the private scene's matrix, so the fused
        // points come back in the coordinates the caller handed us.
        Object3D* origParent = cloud.parent;
        Matrix4 parentWorld;
        if (origParent) {
            origParent->updateMatrixWorld(true);
            parentWorld.copy(*origParent->matrixWorld);
        }
        std::shared_ptr<Object3D> owned = cloud.removeFromParent();
        auto stage = Scene::create();
        stage->matrixAutoUpdate = false;
        stage->matrix->copy(parentWorld);
        stage->addRef(cloud);

        const auto priorMode = renderer.splatDepthAovMode();
        renderer.setSplatDepthAov(VulkanRenderer::SplatDepthMode::Median);
        // MSAA rasterizes UNJITTERED by design (VulkanSplat_test says so, and
        // relies on it). Without that, the projection carries a per-frame Halton
        // offset and the capture becomes a function of the frame counter rather
        // than of the pose — which would forfeit the determinism contract this
        // bake exists to keep. Two reallocations for a one-time bake.
        const uint32_t priorMsaa = renderer.gbufferMsaa();
        if (priorMsaa < 2) renderer.setGbufferMsaa(2);

        const auto fbSize = renderer.framebufferSize();
        const float aspect = fbSize.height() > 0
                                     ? static_cast<float>(fbSize.width()) / static_cast<float>(fbSize.height())
                                     : 1.f;

        Volume vol;
        vol.maxBlocks = std::max<uint64_t>(1ull, options.maxBlockBytes / sizeof(Block));
        std::vector<float> depth;
        std::vector<uint8_t> raw;
        // Per-pose depth summary, one entry per kTile x kTile pixel tile: the
        // range of the covered depths in it and whether it is covered
        // everywhere. The carve pass's whole-block fast paths read nothing else.
        std::vector<float> tileMin, tileMax;
        std::vector<uint8_t> tileFull;
        double renderMs = 0, fuseMs = 0;

        // KinectFusion's weighted running average, written ONCE so that the
        // carve pass's bulk fast path cannot drift from its per-voxel path by so
        // much as a contraction: both call this, with the same operand types.
        const auto integrate = [&options](Block& blk, int idx, float s) {
            const float wOld = blk.w[idx];
            blk.tsdf[idx] = (blk.tsdf[idx] * wOld + s) / (wOld + 1.f);
            blk.w[idx] = std::min(options.maxWeight, wOld + 1.f);
        };

        auto poseCamera = [&](const BakePose& pose) {
            auto camera = PerspectiveCamera::create(pose.fov, aspect,
                                                    std::max(1e-3f, voxel * 0.5f),
                                                    std::max(10.f, fit.radius * 20.f));
            camera->position.copy(pose.position);
            camera->up.copy(pose.up);
            camera->lookAt(pose.target);
            camera->updateMatrixWorld(true);
            camera->updateProjectionMatrix();
            return camera;
        };

        // A priming pass over every pose before anything is read back. The splat
        // pass's (splat, tile) expansion budget GROWS when a frame truncates
        // against it, so the first bake of a cloud would fuse truncated frames
        // where the second one — inheriting the grown budget — fuses whole ones,
        // and the two would differ. Priming pays the growth up front, once, so
        // every fused frame sees the settled budget.
        {
            const auto t0 = clock::now();
            for (const auto& pose : poses) renderer.render(*stage, *poseCamera(pose));
            renderMs += std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        }

        for (const auto& pose : poses) {

            auto camera = poseCamera(pose);

            const auto t0 = clock::now();
            for (int f = 0; f < kFramesPerPose; ++f) renderer.render(*stage, *camera);

            int aw = 0, ah = 0, abpp = 0;
            const bool got = renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::SplatDepth,
                                                     raw, aw, ah, abpp);
            renderMs += std::chrono::duration<double, std::milli>(clock::now() - t0).count();
            if (!got || abpp != 4 || aw <= 0 || ah <= 0) continue;

            const auto t1 = clock::now();
            depth.resize(static_cast<size_t>(aw) * static_cast<size_t>(ah));
            std::memcpy(depth.data(), raw.data(), depth.size() * sizeof(float));
            for (auto& d : depth)
                if (!(d > 0.f) || !std::isfinite(d)) d = 0.f;
            // The wrong-mode tell. A ray landing past the pose's own distance to
            // the fit centre went THROUGH the near side of a hollow subject and
            // found its far side — which is what an interior scan looks like
            // when it is orbited from outside. Report only; it changes nothing.
            // Compared against the view-AXIS distance the AOV exports, which is
            // shorter than the euclidean one off-axis, so the count errs quiet.
            const float centreDist = pose.position.distanceTo(fit.center);
            for (const auto d : depth) {
                if (!(d > 0.f)) continue;
                ++out.stats.depthSamples;
                if (!interior && d > centreDist) ++out.stats.beyondCentreSamples;
            }
            guardDepth(depth, aw, ah, options.fringeErode, options.outlierTolerance * trunc,
                       out.stats.skippedFringe, out.stats.skippedOutlier);

            // Ray reconstruction. The AOV carries -viewPos.z (the view AXIS
            // distance, not the euclidean one — splat_project.comp), so a ray
            // parametrised by that same axis distance is dirView = (ndc.x/P00,
            // ndc.y/P11, -1) and viewPos(t) = dirView * t exactly.
            const auto& P = camera->projectionMatrix.elements;
            const float invP00 = 1.f / P[0], invP11 = 1.f / P[5];
            const auto& M = camera->matrixWorld->elements;
            const Vector3 camPos{M[12], M[13], M[14]};
            const Vector3 mx{M[0], M[1], M[2]}, my{M[4], M[5], M[6]}, mz{M[8], M[9], M[10]};

            auto worldDir = [&](int x, int y) {
                const float ndcx = (static_cast<float>(x) + 0.5f) / static_cast<float>(aw) * 2.f - 1.f;
                const float ndcy = 1.f - (static_cast<float>(y) + 0.5f) / static_cast<float>(ah) * 2.f;
                const float vx = ndcx * invP00, vy = ndcy * invP11;
                return Vector3{mx.x * vx + my.x * vy - mz.x,
                               mx.y * vx + my.y * vy - mz.y,
                               mx.z * vx + my.z * vy - mz.z};
            };

            // 1. Allocation: blocks within +-truncation of an observed hit, and
            // only for a hit NEAR ENOUGH to be the subject. Past maxDepth the
            // sample allocates nothing at all — it still gets to carve in the
            // update pass below, which is the half of its information that costs
            // no memory.
            for (int y = 0; y < ah; ++y)
                for (int x = 0; x < aw; ++x) {
                    const float d = depth[static_cast<size_t>(y) * aw + x];
                    if (!(d > 0.f)) continue;
                    if (d > maxDepth) {
                        ++out.stats.skippedFar;
                        continue;
                    }
                    const Vector3 dir = worldDir(x, y);
                    uint64_t last = ~0ull;
                    for (float t = d - trunc; t <= d + trunc + 1e-6f; t += voxel) {
                        const float tt = std::min(t, d + trunc);
                        const Vector3 p{camPos.x + dir.x * tt, camPos.y + dir.y * tt, camPos.z + dir.z * tt};
                        const int bx = blockOf(static_cast<int>(std::floor(p.x / voxel)));
                        const int by = blockOf(static_cast<int>(std::floor(p.y / voxel)));
                        const int bz = blockOf(static_cast<int>(std::floor(p.z / voxel)));
                        const uint64_t key = blockKey(bx, by, bz);
                        if (key == last) continue;
                        last = key;
                        vol.ensure(key, bx, by, bz);
                    }
                }

            // 2. Update: every allocated block, so that a floater allocated by
            // one pose is carved by the poses that see THROUGH its location.
            // KinectFusion's weighted running average, sequential in pose order.
            const auto& V = camera->matrixWorldInverse.elements;
            const Vector3 ex{V[0] * voxel, V[1] * voxel, V[2] * voxel};
            const Vector3 ey{V[4] * voxel, V[5] * voxel, V[6] * voxel};
            const Vector3 ez{V[8] * voxel, V[9] * voxel, V[10] * voxel};
            const float nearZ = camera->nearPlane;

            // The summary the fast paths below classify blocks against. An edge
            // tile summarises only its IN-IMAGE pixels, which is conservative
            // for both tests: a min over a superset is lower and a max over a
            // superset is higher, so a rect that clears the summary clears its
            // own subset of it too.
            const int tilesX = (aw + kTile - 1) / kTile, tilesY = (ah + kTile - 1) / kTile;
            constexpr float kInf = std::numeric_limits<float>::infinity();
            tileMin.assign(static_cast<size_t>(tilesX) * tilesY, kInf);
            tileMax.assign(static_cast<size_t>(tilesX) * tilesY, -kInf);
            tileFull.assign(static_cast<size_t>(tilesX) * tilesY, 1);
            for (int ty = 0; ty < tilesY; ++ty)
                for (int tx = 0; tx < tilesX; ++tx) {
                    const size_t t = static_cast<size_t>(ty) * tilesX + tx;
                    const int x1 = std::min(aw, (tx + 1) * kTile), y1 = std::min(ah, (ty + 1) * kTile);
                    for (int y = ty * kTile; y < y1; ++y)
                        for (int x = tx * kTile; x < x1; ++x) {
                            const float d = depth[static_cast<size_t>(y) * aw + x];
                            if (d > 0.f) {
                                tileMin[t] = std::min(tileMin[t], d);
                                tileMax[t] = std::max(tileMax[t], d);
                            } else {
                                tileFull[t] = 0;
                            }
                        }
                }

            for (auto& b : vol.blocks) {

                const float ox = (static_cast<float>(b.bx * kB) + 0.5f) * voxel;
                const float oy = (static_cast<float>(b.by * kB) + 0.5f) * voxel;
                const float oz = (static_cast<float>(b.bz * kB) + 0.5f) * voxel;
                Vector3 base{V[0] * ox + V[4] * oy + V[8] * oz + V[12],
                             V[1] * ox + V[5] * oy + V[9] * oz + V[13],
                             V[2] * ox + V[6] * oy + V[10] * oz + V[14]};

                // ── whole-block fast paths ──────────────────────────────────
                // Every one of them is a RESTATEMENT of the per-voxel loop
                // below, never an approximation: either it proves the loop
                // changes nothing for all 512 voxels (skip) or it proves the
                // loop takes the s == 1 branch for all 512 and does that
                // arithmetic verbatim (bulk). The mesh hash is what enforces it.
                //
                // The bounds: the 512 voxel centres are an affine lattice, so
                // view-space z is affine over them and its extremes are AT the
                // corners; the projection is projective, so a box wholly in
                // front of the near plane maps into the convex hull of its
                // projected corners, and their screen AABB contains every
                // voxel's pixel. The margins absorb float rounding on the
                // interior points, and they only ever make a fast path rarer.
                if (options.carveFastPaths) {

                    const auto vpAt = [&](float i, float j, float k) {
                        return Vector3{base.x + ex.x * i + ey.x * j + ez.x * k,
                                       base.y + ex.y * i + ey.y * j + ez.y * k,
                                       base.z + ex.z * i + ey.z * j + ez.z * k};
                    };
                    Vector3 corner[8];
                    float zLo = kInf, zHi = -kInf;
                    for (int c = 0; c < 8; ++c) {
                        corner[c] = vpAt((c & 1) ? float(kB - 1) : 0.f,
                                         (c & 2) ? float(kB - 1) : 0.f,
                                         (c & 4) ? float(kB - 1) : 0.f);
                        const float z = -corner[c].z;
                        zLo = std::min(zLo, z);
                        zHi = std::max(zHi, z);
                    }
                    const float zEps = 1e-4f * std::max(1.f, std::abs(zHi));

                    // Wholly behind the near plane: `z <= nearZ` for every voxel.
                    if (zHi <= nearZ - zEps) {
                        ++out.stats.carveSkippedBlocks;
                        continue;
                    }
                    if (zLo > nearZ + zEps) {

                        float pxLo = kInf, pxHi = -kInf, pyLo = kInf, pyHi = -kInf;
                        for (int c = 0; c < 8; ++c) {
                            const float z = -corner[c].z;
                            const float px = (P[0] * corner[c].x / z * 0.5f + 0.5f) * static_cast<float>(aw);
                            const float py = (0.5f - P[5] * corner[c].y / z * 0.5f) * static_cast<float>(ah);
                            pxLo = std::min(pxLo, px);
                            pxHi = std::max(pxHi, px);
                            pyLo = std::min(pyLo, py);
                            pyHi = std::max(pyHi, py);
                        }
                        constexpr float kPixEps = 1.f;
                        pxLo -= kPixEps;
                        pxHi += kPixEps;
                        pyLo -= kPixEps;
                        pyHi += kPixEps;

                        // Wholly outside the image. -1 and not 0 on the low side
                        // because the per-voxel bounds test truncates toward
                        // zero: px == -0.5 gives u == 0, which is IN bounds and
                        // samples column 0.
                        if (pxHi <= -1.f || pyHi <= -1.f ||
                            pxLo >= static_cast<float>(aw) || pyLo >= static_cast<float>(ah)) {
                            ++out.stats.carveSkippedBlocks;
                            continue;
                        }

                        const auto pixClamp = [](float v, int n) { return std::clamp(v, -2.f, static_cast<float>(n) + 2.f); };
                        const int x0 = std::max(0, static_cast<int>(std::floor(pixClamp(pxLo, aw))));
                        const int x1 = std::min(aw - 1, static_cast<int>(std::floor(pixClamp(pxHi, aw))));
                        const int y0 = std::max(0, static_cast<int>(std::floor(pixClamp(pyLo, ah))));
                        const int y1 = std::min(ah - 1, static_cast<int>(std::floor(pixClamp(pyHi, ah))));
                        if (x0 <= x1 && y0 <= y1) {

                            float dLo = kInf, dHi = -kInf;
                            bool full = true;
                            for (int ty = y0 / kTile; ty <= y1 / kTile; ++ty)
                                for (int tx = x0 / kTile; tx <= x1 / kTile; ++tx) {
                                    const size_t t = static_cast<size_t>(ty) * tilesX + tx;
                                    dLo = std::min(dLo, tileMin[t]);
                                    dHi = std::max(dHi, tileMax[t]);
                                    if (!tileFull[t]) full = false;
                                }
                            const float dEps = 1e-4f * std::max(1.f, std::max(std::abs(zHi), dHi > 0.f ? dHi : 0.f));

                            // Wholly behind every depth this block can read (dHi
                            // is -inf when it can read none): `sdf < -trunc` for
                            // every voxel, so every voxel continues. Needs no
                            // coverage flag — an uncovered pixel continues too.
                            if (dHi < zLo - trunc - dEps) {
                                ++out.stats.carveSkippedBlocks;
                                continue;
                            }

                            // Wholly in free space, and every pixel it can read
                            // is covered and in the image: `sdf >= trunc` for
                            // every voxel, so `s = min(1, sdf/trunc)` is EXACTLY
                            // 1.0f — the quotient of a value >= trunc is >= 1 in
                            // reals and rounding to nearest is monotone, so it
                            // cannot land under 1. The update is then the
                            // per-voxel one with s substituted, which is why
                            // both go through the same `integrate`.
                            const bool inside = pxLo >= 0.f && pxHi < static_cast<float>(aw) &&
                                                pyLo >= 0.f && pyHi < static_cast<float>(ah);
                            if (inside && full && dLo - zHi > trunc + dEps) {
                                for (int idx = 0; idx < kBV; ++idx) integrate(b, idx, 1.f);
                                ++out.stats.carveBulkBlocks;
                                continue;
                            }
                        }
                    }
                }
                ++out.stats.carveVoxelBlocks;

                for (int k = 0; k < kB; ++k)
                    for (int j = 0; j < kB; ++j)
                        for (int i = 0; i < kB; ++i) {

                            const Vector3 vp{base.x + ex.x * i + ey.x * j + ez.x * k,
                                             base.y + ex.y * i + ey.y * j + ez.y * k,
                                             base.z + ex.z * i + ey.z * j + ez.z * k};
                            const float z = -vp.z;
                            if (z <= nearZ) continue;
                            const float px = (P[0] * vp.x / z * 0.5f + 0.5f) * static_cast<float>(aw);
                            const float py = (0.5f - P[5] * vp.y / z * 0.5f) * static_cast<float>(ah);
                            const int u = static_cast<int>(px), v = static_cast<int>(py);
                            if (u < 0 || v < 0 || u >= aw || v >= ah) continue;
                            const float d = depth[static_cast<size_t>(v) * aw + u];
                            if (!(d > 0.f)) continue;

                            const float sdf = d - z;
                            if (sdf < -trunc) continue;// behind the surface: unobserved
                            const float s = std::min(1.f, sdf / trunc);
                            integrate(b, i + j * kB + k * kB * kB, s);
                        }
            }
            fuseMs += std::chrono::duration<double, std::milli>(clock::now() - t1).count();
        }

        renderer.setSplatDepthAov(priorMode);
        if (priorMsaa < 2) renderer.setGbufferMsaa(priorMsaa);
        stage->remove(cloud);
        if (origParent) {
            if (owned) {
                origParent->add(owned);
            } else {
                origParent->addRef(cloud);
            }
        }

        out.stats.blocks = vol.blocks.size();
        out.stats.peakBlockBytes = static_cast<uint64_t>(vol.blocks.size()) * sizeof(Block);
        out.stats.refusedBlocks = vol.refused;
        out.stats.renderMs = renderMs;
        out.stats.fuseMs = fuseMs;
        if (vol.blocks.empty()) return out;

        // ── marching cubes ──────────────────────────────────────────────────
        const auto t2 = clock::now();
        const auto& edgeTable = threepp::detail::mcEdgeTable();
        const auto& triTable = threepp::detail::mcTriTable();
        static constexpr int cx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
        static constexpr int cy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
        static constexpr int cz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
        static constexpr int edge[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

        // The field is -tsdf so the tables' "corner value > isolevel means
        // inside" convention (extras/pointcloud/MarchingCubes.hpp) puts the
        // solid side, tsdf < 0, inside.
        Block* cache = nullptr;
        uint64_t cacheKey = ~0ull;
        auto sample = [&](int vx, int vy, int vz, float& value) {
            const int bx = blockOf(vx), by = blockOf(vy), bz = blockOf(vz);
            const uint64_t key = blockKey(bx, by, bz);
            if (key != cacheKey) {
                cache = vol.find(key);
                cacheKey = key;
            }
            if (!cache) return false;
            const int idx = (vx - bx * kB) + (vy - by * kB) * kB + (vz - bz * kB) * kB * kB;
            if (cache->w[idx] < options.weightFloor) return false;
            value = -cache->tsdf[idx];
            return true;
        };

        std::vector<uint64_t> order;
        order.reserve(vol.index.size());
        for (const auto& [key, i] : vol.index) order.push_back(key);
        // Sorted, not hash order: the vertex and index arrays are part of the
        // determinism contract and an unordered_map's walk is not a promise.
        std::sort(order.begin(), order.end());

        std::unordered_map<uint64_t, uint32_t> vertexOf;
        std::vector<uint64_t> triCell;
        auto edgeVertex = [&](int vx, int vy, int vz, int e, const float val[8]) -> uint32_t {
            const int a = edge[e][0], b = edge[e][1];
            int ax = vx + cx[a], ay = vy + cy[a], az = vz + cz[a];
            int bx = vx + cx[b], by = vy + cy[b], bz = vz + cz[b];
            float va = val[a], vb = val[b];
            // Orient low-to-high along the edge axis so that both cells sharing
            // this edge compute the same vertex, bit for bit, and weld.
            if (bx < ax || by < ay || bz < az) {
                std::swap(ax, bx);
                std::swap(ay, by);
                std::swap(az, bz);
                std::swap(va, vb);
            }
            const int axis = (bx != ax) ? 0 : (by != ay) ? 1 : 2;
            const uint64_t key = edgeKey(ax, ay, az, axis);
            const auto it = vertexOf.find(key);
            if (it != vertexOf.end()) return it->second;

            const float denom = vb - va;
            float t = (std::abs(denom) > 1e-12f) ? (0.f - va) / denom : 0.5f;
            t = std::clamp(t, 0.f, 1.f);
            const Vector3 pa{(static_cast<float>(ax) + 0.5f) * voxel,
                             (static_cast<float>(ay) + 0.5f) * voxel,
                             (static_cast<float>(az) + 0.5f) * voxel};
            const Vector3 pb{(static_cast<float>(bx) + 0.5f) * voxel,
                             (static_cast<float>(by) + 0.5f) * voxel,
                             (static_cast<float>(bz) + 0.5f) * voxel};
            const auto id = static_cast<uint32_t>(out.positions.size() / 3);
            out.positions.push_back(pa.x + t * (pb.x - pa.x));
            out.positions.push_back(pa.y + t * (pb.y - pa.y));
            out.positions.push_back(pa.z + t * (pb.z - pa.z));
            vertexOf.emplace(key, id);
            return id;
        };

        float val[8];
        for (const uint64_t key : order) {
            const Block* blk = vol.find(key);
            if (!blk) continue;
            const int b0x = blk->bx * kB, b0y = blk->by * kB, b0z = blk->bz * kB;
            for (int k = 0; k < kB; ++k)
                for (int j = 0; j < kB; ++j)
                    for (int i = 0; i < kB; ++i) {

                        const int vx = b0x + i, vy = b0y + j, vz = b0z + k;
                        bool complete = true;
                        int cube = 0;
                        for (int c = 0; c < 8 && complete; ++c) {
                            if (!sample(vx + cx[c], vy + cy[c], vz + cz[c], val[c])) complete = false;
                            else if (val[c] > 0.f) cube |= (1 << c);
                        }
                        // A cell missing a corner sits at the edge of what was
                        // observed; guessing its other side would invent surface.
                        if (!complete) continue;
                        if (edgeTable[cube] == 0) continue;

                        const auto& tri = triTable[cube];
                        for (int t = 0; tri[t] != -1; t += 3) {
                            // Reversed against the table's own order. Bourke's
                            // tables set the cube bit for corners BELOW the
                            // isolevel; MarchingCubes.hpp classifies with `>`
                            // (it derives its normals from the field gradient,
                            // so winding never mattered there), which winds the
                            // triangles inward. A collider's triangles have to
                            // face the free side, so they are swapped back.
                            out.indices.push_back(edgeVertex(vx, vy, vz, tri[t], val));
                            out.indices.push_back(edgeVertex(vx, vy, vz, tri[t + 2], val));
                            out.indices.push_back(edgeVertex(vx, vy, vz, tri[t + 1], val));
                            triCell.push_back(blockKey(vx, vy, vz));
                        }
                    }
        }
        for (const auto& b : vol.blocks)
            for (int i = 0; i < kBV; ++i)
                if (b.w[i] >= options.weightFloor) ++out.stats.observedVoxels;

        if (out.indices.empty()) {
            out.stats.meshMs = std::chrono::duration<double, std::milli>(clock::now() - t2).count();
            return out;
        }

        // ── connected components ────────────────────────────────────────────
        // What survives carving is islands; a scan's floaters are small ones.
        const auto nv = static_cast<uint32_t>(out.positions.size() / 3);
        std::vector<uint32_t> parent(nv);
        for (uint32_t i = 0; i < nv; ++i) parent[i] = i;
        auto root = [&parent](uint32_t a) {
            while (parent[a] != a) {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            return a;
        };
        auto unite = [&](uint32_t a, uint32_t b) {
            a = root(a);
            b = root(b);
            if (a != b) parent[std::max(a, b)] = std::min(a, b);
        };
        for (size_t t = 0; t < out.indices.size(); t += 3) {
            unite(out.indices[t], out.indices[t + 1]);
            unite(out.indices[t], out.indices[t + 2]);
        }

        // Size a component in CELLS, not triangles: minComponentVoxels is a
        // metric size once multiplied by the voxel, which triangle count is not.
        std::vector<std::pair<uint32_t, uint64_t>> cells;
        cells.reserve(triCell.size());
        for (size_t t = 0; t < triCell.size(); ++t)
            cells.emplace_back(root(out.indices[t * 3]), triCell[t]);
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        std::unordered_map<uint32_t, uint32_t> cellCount;
        for (const auto& [r, c] : cells) ++cellCount[r];
        out.stats.components = static_cast<uint32_t>(cellCount.size());

        std::vector<uint32_t> keptIndices;
        keptIndices.reserve(out.indices.size());
        std::unordered_set<uint32_t> culled;
        for (size_t t = 0; t < triCell.size(); ++t) {
            const uint32_t r = root(out.indices[t * 3]);
            if (static_cast<int>(cellCount[r]) < options.minComponentVoxels) {
                ++out.stats.culledTriangles;
                culled.insert(r);
                continue;
            }
            keptIndices.push_back(out.indices[t * 3]);
            keptIndices.push_back(out.indices[t * 3 + 1]);
            keptIndices.push_back(out.indices[t * 3 + 2]);
        }
        out.stats.culledComponents = static_cast<uint32_t>(culled.size());

        // Compact: renumber in first-use order, so the arrays stay a function of
        // the field and not of which vertices happened to be culled.
        std::vector<uint32_t> remap(nv, ~0u);
        std::vector<float> pos;
        pos.reserve(out.positions.size());
        for (auto& idx : keptIndices) {
            if (remap[idx] == ~0u) {
                remap[idx] = static_cast<uint32_t>(pos.size() / 3);
                pos.push_back(out.positions[idx * 3]);
                pos.push_back(out.positions[idx * 3 + 1]);
                pos.push_back(out.positions[idx * 3 + 2]);
            }
            idx = remap[idx];
        }
        out.positions = std::move(pos);
        out.indices = std::move(keptIndices);

        if (!out.positions.empty()) {
            out.stats.aabbMin.set(out.positions[0], out.positions[1], out.positions[2]);
            out.stats.aabbMax = out.stats.aabbMin;
            for (size_t i = 0; i < out.positions.size(); i += 3) {
                out.stats.aabbMin.x = std::min(out.stats.aabbMin.x, out.positions[i]);
                out.stats.aabbMin.y = std::min(out.stats.aabbMin.y, out.positions[i + 1]);
                out.stats.aabbMin.z = std::min(out.stats.aabbMin.z, out.positions[i + 2]);
                out.stats.aabbMax.x = std::max(out.stats.aabbMax.x, out.positions[i]);
                out.stats.aabbMax.y = std::max(out.stats.aabbMax.y, out.positions[i + 1]);
                out.stats.aabbMax.z = std::max(out.stats.aabbMax.z, out.positions[i + 2]);
            }
        }
        out.stats.meshMs = std::chrono::duration<double, std::milli>(clock::now() - t2).count();
        return out;
    }

    std::shared_ptr<Mesh> makeSensorMesh(const SurfaceMesh& surface) {

        if (surface.empty()) return nullptr;

        auto geom = BufferGeometry::create();
        geom->setAttribute("position", FloatBufferAttribute::create(surface.positions, 3));
        geom->setIndex(std::vector<unsigned int>(surface.indices.begin(), surface.indices.end()));
        // The renderer drops a mesh with no normal attribute from the scene
        // entirely, and the depth sensor's G-buffer wants shaded normals for
        // the same surface the lidar's rchit derives from the triangle.
        geom->computeVertexNormals();

        auto mat = MeshStandardMaterial::create();
        mat->roughness = 1.f;
        mat->metalness = 0.f;
        mat->side = Side::Double;// a scan's shell may be seen from either face

        auto mesh = Mesh::create(geom, mat);
        mesh->name = "splat-sensor-surface";
        mesh->layers.set(VulkanRenderer::kSensorOnlyLayer);
        return mesh;
    }

}// namespace threepp::splats
