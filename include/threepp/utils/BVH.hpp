// Bounding volume hierarchy over the triangles of a threepp geometry.
//
// Built with a binned surface-area heuristic and stored as a flat node array,
// so ray queries walk it iteratively, near child first. Triangle indices handed
// back by any query are the triangle's position in the source geometry. A
// triangle with a NaN or infinite vertex is left out of the tree and is never
// reported.

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

        // maxTrianglesPerNode: a node with this many triangles or fewer becomes a
        // leaf; larger nodes are split where the surface-area heuristic says, and
        // may stop a little above it when no split pays off.
        // maxSubdivisions: a safety limit on the tree depth, not a target; values
        // below 64 are raised to 64 (and above 128 lowered to 128), so it never
        // caps a large mesh's leaf size.
        explicit BVH(int maxTrianglesPerNode = 8, int maxSubdivisions = 10)
            : maxTrianglesPerNode(maxTrianglesPerNode), maxSubdivisions(maxSubdivisions) {}

        // Keeps a raw pointer to `geometry` (see getGeometry); it must outlive the BVH.
        // One or two indices (or vertices, without an index) left over after the
        // last whole triangle are ignored.
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
        // Safe to call from several threads at once.
        [[nodiscard]] std::optional<RayHit> raycast(const Ray& ray, float maxDistance = std::numeric_limits<float>::infinity()) const;

        // Early-out boolean occlusion query. The ray is in the BVH's local space.
        [[nodiscard]] bool raycastAny(const Ray& ray, float maxDistance) const;

        void collectBoxes(std::vector<BVHBox3>& boxes) const;

        [[nodiscard]] const BufferGeometry* getGeometry() const;

        // The triangles in the tree: the source's, less any with a non-finite vertex.
        [[nodiscard]] std::size_t triangleCount() const {
            return triangles.size();
        }

        // Root bounds in the BVH's own space; empty when nothing was built.
        [[nodiscard]] Box3 boundingBox() const {
            return nodes.empty() ? Box3{} : nodes.front().box();
        }

    private:
        // 32 bytes; plain floats so the ray's slab test inlines.
        struct Node {
            float min[3];
            int first;// leaf: first triangle (leaf order); inner: left child, the right is first + 1
            float max[3];
            int count;// > 0 for a leaf

            [[nodiscard]] bool isLeaf() const {
                return count > 0;
            }

            [[nodiscard]] Box3 box() const {
                return {{min[0], min[1], min[2]}, {max[0], max[1], max[2]}};
            }
        };

        // What the ray test reads per triangle: a vertex and the two edges from it.
        struct RayTriangle {
            float v0[3];
            float e1[3];
            float e2[3];
        };

        int maxTrianglesPerNode;
        int maxSubdivisions;

        std::vector<Node> nodes;               // nodes[0] is the root
        std::vector<Triangle> triangles;       // leaf order
        std::vector<RayTriangle> rayTriangles; // leaf order
        std::vector<int> sourceIndex;          // leaf order -> index in the source geometry
        const BufferGeometry* geometry = nullptr;

        void buildTree();

        void collectBoxes(int node, std::vector<BVHBox3>& boxes) const;
    };


}// namespace threepp

#endif//THREEPP_BVH_HPP
