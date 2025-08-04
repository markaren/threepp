// Simple BVH implementation for threepp Geometries

#ifndef THREEPPP_BVH_HPP
#define THREEPPP_BVH_HPP


#include <memory>
#include <vector>

#include "threepp/math/Box3.hpp"
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

        explicit BVH(int maxTrianglesPerNode = 8, int maxSubdivisions = 10)
            : maxTrianglesPerNode(maxTrianglesPerNode), maxSubdivisions(maxSubdivisions) {}

        void build(const BufferGeometry& geometry);

        // Helper methods for single-shape intersections
        [[nodiscard]] std::vector<int> intersect(const Box3& box, const Matrix4& m = Matrix4()) const;

        [[nodiscard]] std::vector<int> intersect(const Sphere& sphere, const Matrix4& m = Matrix4()) const;

        // Intersect this BVH with another BVH
        [[nodiscard]] static std::vector<IntersectionResult> intersect(const BVH& b1, const Matrix4& m1, const BVH& b2, const Matrix4& m2, bool accurate = false);

        // Simple true/false intersection test with another BVH
        [[nodiscard]] static bool intersects(const BVH& b1, const BVH& b2, const Matrix4& m1 = Matrix4(), const Matrix4& m2 = Matrix4());

        void collectBoxes(std::vector<BVHBox3>& boxes) const;

        [[nodiscard]] const BufferGeometry* getGeometry() const;

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

        // Tests intersection between two BVH nodes
        static void intersectBVHNodes(const BVH& b1, const BVHNode* nodeA, const Matrix4& m1, const BVH& b2, const BVHNode* nodeB, const Matrix4& m2, std::vector<IntersectionResult>& results, bool accurate);

        static void collectBoxes(const BVHNode* node, std::vector<BVHBox3>& boxes);
    };


}// namespace threepp

#endif//THREEPPP_BVH_HPP
