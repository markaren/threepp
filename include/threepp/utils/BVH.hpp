// Simple BVH implementation for threepp Geometries

#ifndef THREEPP_BVH_HPP
#define THREEPP_BVH_HPP


#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "threepp/math/Box3.hpp"
#include "threepp/math/Ray.hpp"
#include "threepp/math/Triangle.hpp"

#include <threepp/math/Matrix4.hpp>

namespace threepp {

    class BufferGeometry;

    struct BVHBox3: Box3 {

        BVHBox3(Box3 bb, bool isLeaf): Box3(bb), isLeaf_(isLeaf) {}

        [[nodiscard]] bool isLeaf() const {
            return isLeaf_;
        }

    private:
        bool isLeaf_{false};
    };

    class BVH {

    public:
        struct IntersectionResult {
            int idxA;
            int idxB;
            Vector3 position;// Center of intersection region
        };

        struct RayHit {
            float distance;
            int triangleIndex;
            Vector3 point;
            Vector3 normal;// geometric normal, flipped to face the ray origin
        };

        explicit BVH(int maxTrianglesPerNode = 8, int maxSubdivisions = 10)
            : maxTrianglesPerNode(maxTrianglesPerNode), maxSubdivisions(maxSubdivisions) {}

        // Keeps a raw pointer to `geometry` (see getGeometry); it must outlive the BVH.
        void build(const BufferGeometry& geometry);

        // Build from raw buffers: `positions` is xyz-interleaved, `indices` a flat
        // triangle list or empty for a soup (three consecutive vertices per face).
        // The triangles are copied; the inputs are not referenced afterwards.
        void build(const std::vector<float>& positions, const std::vector<unsigned int>& indices = {});

        // Helper methods for single-shape intersections
        [[nodiscard]] std::vector<int> intersect(const Box3& box, const Matrix4& m = Matrix4()) const;

        [[nodiscard]] std::vector<int> intersect(const Sphere& sphere, const Matrix4& m = Matrix4()) const;

        // Intersect this BVH with another BVH.
        // accurate = true: exact triangle-triangle test at the leaves, one result per
        // intersecting triangle pair, `position` the midpoint of the intersection
        // segment (average of the centroids for a coplanar pair).
        // accurate = false: one result per overlapping pair of leaf boxes, `position`
        // the centre of the box overlap, idxA/idxB = -1.
        [[nodiscard]] static std::vector<IntersectionResult> intersect(const BVH& b1, const Matrix4& m1, const BVH& b2, const Matrix4& m2, bool accurate = false);

        // Exact triangle-level intersection test; returns on the first hit.
        [[nodiscard]] static bool intersects(const BVH& b1, const BVH& b2, const Matrix4& m1 = Matrix4(), const Matrix4& m2 = Matrix4());

        // Smallest surface-to-surface distance: 0 when the meshes intersect, infinity
        // when either BVH is empty or nothing is closer than maxDistance. maxDistance
        // seeds the pruning bound, so a finite cutoff is much cheaper than an exact
        // search between distant meshes.
        [[nodiscard]] static float distance(const BVH& b1, const BVH& b2, const Matrix4& m1 = Matrix4(), const Matrix4& m2 = Matrix4(),
                                            float maxDistance = std::numeric_limits<float>::infinity());

        // Closest hit within maxDistance, or nullopt. The ray is in the BVH's local space.
        [[nodiscard]] std::optional<RayHit> raycast(const Ray& ray, float maxDistance = std::numeric_limits<float>::infinity()) const;

        // Early-out boolean occlusion query. The ray is in the BVH's local space.
        [[nodiscard]] bool raycastAny(const Ray& ray, float maxDistance) const;

        void collectBoxes(std::vector<BVHBox3>& boxes) const;

        [[nodiscard]] const BufferGeometry* getGeometry() const;

        [[nodiscard]] std::size_t triangleCount() const {
            return triangles.size();
        }

        // Root bounds in the BVH's own space; empty when nothing was built.
        [[nodiscard]] Box3 boundingBox() const {
            return root ? root->boundingBox : Box3{};
        }

    private:
        class BVHNode {
        public:
            Box3 boundingBox;
            std::unique_ptr<BVHNode> left;
            std::unique_ptr<BVHNode> right;
            std::vector<int> triangleIndices;

            BVHNode() = default;

            [[nodiscard]] bool isLeaf() const {
                return left == nullptr && right == nullptr;
            }
        };

        std::unique_ptr<BVHNode> root;
        int maxTrianglesPerNode;
        int maxSubdivisions;

        std::vector<Triangle> triangles;
        const BufferGeometry* geometry = nullptr;

        std::unique_ptr<BVHNode> buildNode(std::vector<int>& indices, int depth);

        static void collectBoxes(const BVHNode* node, std::vector<BVHBox3>& boxes);
    };


}// namespace threepp

#endif//THREEPP_BVH_HPP
