// A voxel-hash spatial index for 3D point clouds.
//
// threepp has no KD-tree, so this fills the gap for the common point-cloud
// operations: voxel downsampling, nearest-neighbour proximity queries, and
// incremental occupancy/point mapping (e.g. for LIDAR mapping / SLAM). It is
// header-only and dependency-free (threepp math + the standard library).

#ifndef THREEPP_VOXELGRID_HPP
#define THREEPP_VOXELGRID_HPP

#include "threepp/math/Vector3.hpp"

#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace threepp {

    namespace detail {

        struct VoxelHashKey {
            int x, y, z;
            bool operator==(const VoxelHashKey& o) const { return x == o.x && y == o.y && z == o.z; }
        };

        struct VoxelHashKeyHash {
            std::size_t operator()(const VoxelHashKey& k) const {
                // Classic large-prime spatial hash (Teschner et al.).
                return static_cast<std::size_t>((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
            }
        };

        inline VoxelHashKey voxelHashKey(const Vector3& p, float inv) {
            return {static_cast<int>(std::floor(p.x * inv)),
                    static_cast<int>(std::floor(p.y * inv)),
                    static_cast<int>(std::floor(p.z * inv))};
        }

    }// namespace detail

    /**
     * A voxel-hash spatial index for 3D points.
     *
     * Offers O(1) amortised insertion with an optional per-voxel capacity and a
     * minimum-spacing dedup filter, plus nearest-neighbour queries over the 27
     * voxels surrounding a query point. It doubles as an incremental point map
     * and as a general proximity structure for point clouds.
     */
    class VoxelGrid {

    public:
        /**
         * @param voxelSize          edge length of a voxel; also the natural
         *                           nearest-neighbour search radius.
         * @param maxPointsPerVoxel  cap on points stored per voxel (0 = no cap).
         * @param minSpacing         reject a point if an existing point in the
         *                           same voxel is closer than this (0 = keep all).
         */
        explicit VoxelGrid(float voxelSize, std::size_t maxPointsPerVoxel = 20, float minSpacing = 0.f)
            : inv_(1.f / voxelSize), voxelSize_(voxelSize),
              minSpacing2_(minSpacing * minSpacing), cap_(maxPointsPerVoxel) {}

        void clear() {
            cells_.clear();
            size_ = 0;
        }

        [[nodiscard]] bool empty() const { return size_ == 0; }
        [[nodiscard]] std::size_t size() const { return size_; }
        [[nodiscard]] float voxelSize() const { return voxelSize_; }

        /// Insert a point. Returns true if it was stored (passed cap + spacing).
        bool insert(const Vector3& p) {
            auto& cell = cells_[detail::voxelHashKey(p, inv_)];
            if (cap_ != 0 && cell.size() >= cap_) return false;
            for (const auto& q : cell) {
                const float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
                if (dx * dx + dy * dy + dz * dz < minSpacing2_) return false;
            }
            cell.push_back(p);
            ++size_;
            return true;
        }

        /**
         * Nearest stored point to `query` within `maxDist`. The search covers the
         * 27 voxels around the query, so the result is exact when
         * `maxDist <= voxelSize`. Returns false if no point is within range.
         */
        bool nearest(const Vector3& query, float maxDist, Vector3& out) const {
            const detail::VoxelHashKey c = detail::voxelHashKey(query, inv_);
            float best = maxDist * maxDist;
            bool found = false;
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        const auto it = cells_.find({c.x + dx, c.y + dy, c.z + dz});
                        if (it == cells_.end()) continue;
                        for (const auto& q : it->second) {
                            const float ex = q.x - query.x, ey = q.y - query.y, ez = q.z - query.z;
                            const float d2 = ex * ex + ey * ey + ez * ez;
                            if (d2 < best) {
                                best = d2;
                                out = q;
                                found = true;
                            }
                        }
                    }
            return found;
        }

        /// Append every stored point to `out`.
        void collect(std::vector<Vector3>& out) const {
            out.reserve(out.size() + size_);
            for (const auto& kv : cells_)
                for (const auto& p : kv.second) out.push_back(p);
        }

        /// Number of occupied voxels (cells holding at least one point).
        [[nodiscard]] std::size_t voxelCount() const { return cells_.size(); }

        /// Append the centre of every occupied voxel to `out` — an occupancy-grid
        /// view of the cloud (one point per cell, regardless of how many points
        /// it holds).
        void collectVoxelCenters(std::vector<Vector3>& out) const {
            out.reserve(out.size() + cells_.size());
            for (const auto& kv : cells_) {
                out.emplace_back((static_cast<float>(kv.first.x) + 0.5f) * voxelSize_,
                                 (static_cast<float>(kv.first.y) + 0.5f) * voxelSize_,
                                 (static_cast<float>(kv.first.z) + 0.5f) * voxelSize_);
            }
        }

    private:
        float inv_;
        float voxelSize_;
        float minSpacing2_;
        std::size_t cap_;
        std::size_t size_{0};
        std::unordered_map<detail::VoxelHashKey, std::vector<Vector3>, detail::VoxelHashKeyHash> cells_;
    };

    /**
     * Voxel-downsample a point cloud: return one representative point (the first
     * encountered) per occupied voxel of size `voxelSize`.
     */
    [[nodiscard]] inline std::vector<Vector3> voxelDownsample(const std::vector<Vector3>& points, float voxelSize) {
        std::unordered_map<detail::VoxelHashKey, Vector3, detail::VoxelHashKeyHash> grid;
        grid.reserve(points.size());
        const float inv = 1.f / voxelSize;
        for (const auto& p : points) {
            grid.emplace(detail::voxelHashKey(p, inv), p);
        }
        std::vector<Vector3> out;
        out.reserve(grid.size());
        for (auto& kv : grid) out.push_back(kv.second);
        return out;
    }

}// namespace threepp

#endif//THREEPP_VOXELGRID_HPP
