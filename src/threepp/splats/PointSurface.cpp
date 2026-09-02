
#include "threepp/splats/PointSurface.hpp"

#include "threepp/extras/pointcloud/MarchingCubes.hpp"// the standard MC tables
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatData.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace threepp;

namespace {

    // 20 bits per axis, offset so negatives pack: +-524288 voxels, which is
    // 2.6 km at the 5 mm floor. Two spare bits carry an edge axis.
    constexpr int kBits = 20;
    constexpr int64_t kOffset = int64_t{1} << (kBits - 1);
    constexpr uint64_t kMask = (uint64_t{1} << kBits) - 1;

    inline uint64_t packAxis(int64_t v) {
        return static_cast<uint64_t>(std::clamp<int64_t>(v + kOffset, 0, static_cast<int64_t>(kMask)));
    }

    inline uint64_t cellKey(int64_t x, int64_t y, int64_t z) {
        return (packAxis(x) << (2 * kBits)) | (packAxis(y) << kBits) | packAxis(z);
    }

    inline uint64_t edgeKey(int64_t x, int64_t y, int64_t z, int axis) {
        return cellKey(x, y, z) | (static_cast<uint64_t>(axis) << (3 * kBits));
    }

    inline void unpackKey(uint64_t key, int64_t& x, int64_t& y, int64_t& z) {
        x = static_cast<int64_t>((key >> (2 * kBits)) & kMask) - kOffset;
        y = static_cast<int64_t>((key >> kBits) & kMask) - kOffset;
        z = static_cast<int64_t>(key & kMask) - kOffset;
    }

}// namespace


namespace threepp::splats {

    SurfaceMesh buildPointSurface(const SplatCloud& cloud, const PointSurfaceOptions& options) {

        return buildPointSurface(cloud.data(), *cloud.matrixWorld, options);
    }

    SurfaceMesh buildPointSurface(const SplatData& data, const Matrix4& toWorld,
                                  const PointSurfaceOptions& options) {

        using clock = std::chrono::steady_clock;
        SurfaceMesh out;

        // ── the points, in world space ──────────────────────────────────────
        const auto t0 = clock::now();
        std::vector<Vector3> pts;
        pts.reserve(data.count());
        const bool floorOpacity = options.opacityFloor > 0.f;
        for (size_t i = 0; i < data.count(); ++i) {

            if (floorOpacity && !(data.opacities[i] >= options.opacityFloor)) continue;
            Vector3 p = data.means[i];
            p.applyMatrix4(toWorld);
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            pts.push_back(p);
        }
        if (pts.empty()) return out;

        float voxel = options.voxelSize;
        if (!(voxel > 0.f)) {
            const float spacing = medianNeighbourSpacing(pts);
            voxel = spacing > 0.f ? std::clamp(2.f * spacing, 0.005f, 0.5f) : 0.05f;
        }
        const float radiusVoxels = std::max(0.25f, options.radiusVoxels);
        const float radius = radiusVoxels * voxel;
        const float iso = std::clamp(options.isolevel, 0.f, 0.999f);
        const float inv = 1.f / voxel;
        out.stats.voxelSize = voxel;
        out.stats.truncation = radius;

        // ── voxel hash: sorted (cell key, point index) ──────────────────────
        std::vector<std::pair<uint64_t, uint32_t>> cells;
        cells.reserve(pts.size());
        const auto cellOf = [&](const Vector3& p, int64_t& x, int64_t& y, int64_t& z) {
            x = static_cast<int64_t>(std::floor(p.x * inv));
            y = static_cast<int64_t>(std::floor(p.y * inv));
            z = static_cast<int64_t>(std::floor(p.z * inv));
        };
        for (size_t i = 0; i < pts.size(); ++i) {
            int64_t x, y, z;
            cellOf(pts[i], x, y, z);
            cells.emplace_back(cellKey(x, y, z), static_cast<uint32_t>(i));
        }
        std::sort(cells.begin(), cells.end());

        std::vector<uint64_t> occupied;
        for (size_t i = 0; i < cells.size(); ++i)
            if (i == 0 || cells[i].first != cells[i - 1].first) occupied.push_back(cells[i].first);
        out.stats.observedVoxels = occupied.size();
        if (occupied.size() > options.maxVoxels) {
            out.stats.refusedBlocks = occupied.size();
            return out;
        }

        // ── the field, on the nodes the cells below will ask for ────────────
        // A node at integer corner (x, y, z) sees the points in the voxels
        // x - m .. x + m - 1 per axis, m = ceil(radiusVoxels): anything within
        // `radius` of it lies there.
        const int64_t m = static_cast<int64_t>(std::ceil(radiusVoxels));
        std::unordered_map<uint64_t, float> nodeValue;
        nodeValue.reserve(occupied.size() * 8);

        const auto nearestD2 = [&](const Vector3& q, int64_t nx, int64_t ny, int64_t nz) {
            float best = std::numeric_limits<float>::infinity();
            for (int64_t dz = -m; dz < m; ++dz)
                for (int64_t dy = -m; dy < m; ++dy)
                    for (int64_t dx = -m; dx < m; ++dx) {
                        const uint64_t key = cellKey(nx + dx, ny + dy, nz + dz);
                        auto it = std::lower_bound(cells.begin(), cells.end(),
                                                   std::make_pair(key, uint32_t{0}));
                        for (; it != cells.end() && it->first == key; ++it) {
                            const float d2 = q.distanceToSquared(pts[it->second]);
                            if (d2 < best) best = d2;
                        }
                    }
            return best;
        };

        const auto valueAt = [&](int64_t nx, int64_t ny, int64_t nz) -> float {
            const uint64_t key = cellKey(nx, ny, nz);
            const auto it = nodeValue.find(key);
            if (it != nodeValue.end()) return it->second;
            const Vector3 q{static_cast<float>(nx) * voxel, static_cast<float>(ny) * voxel,
                            static_cast<float>(nz) * voxel};
            const float d2 = nearestD2(q, nx, ny, nz);
            float v = 0.f;
            if (std::isfinite(d2)) v = std::max(0.f, 1.f - std::sqrt(d2) / radius);
            nodeValue.emplace(key, v);
            return v;
        };

        // Cells to polygonise: the occupied ones dilated by m, since the field
        // reaches that far. Sorted and unique, so the walk is deterministic.
        std::vector<uint64_t> cellList;
        cellList.reserve(occupied.size() * static_cast<size_t>((2 * m + 1) * (2 * m + 1) * (2 * m + 1)));
        for (const uint64_t key : occupied) {
            int64_t x, y, z;
            unpackKey(key, x, y, z);
            for (int64_t dz = -m; dz <= m; ++dz)
                for (int64_t dy = -m; dy <= m; ++dy)
                    for (int64_t dx = -m; dx <= m; ++dx) cellList.push_back(cellKey(x + dx, y + dy, z + dz));
        }
        std::sort(cellList.begin(), cellList.end());
        cellList.erase(std::unique(cellList.begin(), cellList.end()), cellList.end());
        out.stats.blocks = cellList.size();
        out.stats.fuseMs = std::chrono::duration<double, std::milli>(clock::now() - t0).count();

        // ── marching cubes ──────────────────────────────────────────────────
        const auto t1 = clock::now();
        const auto& edgeTable = threepp::detail::mcEdgeTable();
        const auto& triTable = threepp::detail::mcTriTable();
        static constexpr int cx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
        static constexpr int cy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
        static constexpr int cz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
        static constexpr int edge[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

        std::unordered_map<uint64_t, uint32_t> vertexOf;
        std::vector<uint64_t> triCell;
        auto edgeVertex = [&](int64_t vx, int64_t vy, int64_t vz, int e, const float val[8]) -> uint32_t {
            const int a = edge[e][0], b = edge[e][1];
            int64_t ax = vx + cx[a], ay = vy + cy[a], az = vz + cz[a];
            int64_t bx = vx + cx[b], by = vy + cy[b], bz = vz + cz[b];
            float va = val[a], vb = val[b];
            // Low-to-high along the edge axis, so both cells sharing the edge
            // compute the same vertex bit for bit and weld.
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
            float t = (std::abs(denom) > 1e-12f) ? (iso - va) / denom : 0.5f;
            t = std::clamp(t, 0.f, 1.f);
            const Vector3 pa{static_cast<float>(ax) * voxel, static_cast<float>(ay) * voxel,
                             static_cast<float>(az) * voxel};
            const Vector3 pb{static_cast<float>(bx) * voxel, static_cast<float>(by) * voxel,
                             static_cast<float>(bz) * voxel};
            const auto id = static_cast<uint32_t>(out.positions.size() / 3);
            out.positions.push_back(pa.x + t * (pb.x - pa.x));
            out.positions.push_back(pa.y + t * (pb.y - pa.y));
            out.positions.push_back(pa.z + t * (pb.z - pa.z));
            vertexOf.emplace(key, id);
            return id;
        };

        float val[8];
        for (const uint64_t key : cellList) {
            int64_t vx, vy, vz;
            unpackKey(key, vx, vy, vz);
            int cube = 0;
            for (int c = 0; c < 8; ++c) {
                val[c] = valueAt(vx + cx[c], vy + cy[c], vz + cz[c]);
                if (val[c] > iso) cube |= (1 << c);
            }
            if (edgeTable[cube] == 0) continue;

            const auto& tri = triTable[cube];
            for (int t = 0; tri[t] != -1; t += 3) {
                // Reversed against the table's order: the tables set a bit for
                // corners ABOVE the isolevel (the inside, here: near the
                // points), which winds inward. A collider faces the free side.
                out.indices.push_back(edgeVertex(vx, vy, vz, tri[t], val));
                out.indices.push_back(edgeVertex(vx, vy, vz, tri[t + 2], val));
                out.indices.push_back(edgeVertex(vx, vy, vz, tri[t + 1], val));
                triCell.push_back(key);
            }
        }

        if (out.indices.empty()) {
            out.stats.meshMs = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
            return out;
        }

        // ── connected components ────────────────────────────────────────────
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

        std::vector<std::pair<uint32_t, uint64_t>> compCells;
        compCells.reserve(triCell.size());
        for (size_t t = 0; t < triCell.size(); ++t) compCells.emplace_back(root(out.indices[t * 3]), triCell[t]);
        std::sort(compCells.begin(), compCells.end());
        compCells.erase(std::unique(compCells.begin(), compCells.end()), compCells.end());
        std::unordered_map<uint32_t, uint32_t> cellCount;
        for (const auto& [r, c] : compCells) ++cellCount[r];
        out.stats.components = static_cast<uint32_t>(cellCount.size());

        std::vector<uint32_t> kept;
        kept.reserve(out.indices.size());
        std::unordered_set<uint32_t> culled;
        for (size_t t = 0; t < triCell.size(); ++t) {
            const uint32_t r = root(out.indices[t * 3]);
            if (static_cast<int>(cellCount[r]) < options.minComponentVoxels) {
                ++out.stats.culledTriangles;
                culled.insert(r);
                continue;
            }
            kept.push_back(out.indices[t * 3]);
            kept.push_back(out.indices[t * 3 + 1]);
            kept.push_back(out.indices[t * 3 + 2]);
        }
        out.stats.culledComponents = static_cast<uint32_t>(culled.size());

        // Compact in first-use order, so the arrays are a function of the
        // field and not of which vertices were culled.
        std::vector<uint32_t> remap(nv, ~0u);
        std::vector<float> pos;
        pos.reserve(out.positions.size());
        for (auto& idx : kept) {
            if (remap[idx] == ~0u) {
                remap[idx] = static_cast<uint32_t>(pos.size() / 3);
                pos.push_back(out.positions[idx * 3]);
                pos.push_back(out.positions[idx * 3 + 1]);
                pos.push_back(out.positions[idx * 3 + 2]);
            }
            idx = remap[idx];
        }
        out.positions = std::move(pos);
        out.indices = std::move(kept);

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
        out.stats.meshMs = std::chrono::duration<double, std::milli>(clock::now() - t1).count();
        return out;
    }

}// namespace threepp::splats
