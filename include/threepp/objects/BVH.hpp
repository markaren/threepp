// Simple BVH implementation for threepp Geometries

#ifndef THREEPPP_BVH_HPP
#define THREEPPP_BVH_HPP


#include <memory>
#include <vector>

#include "threepp/math/Box3.hpp"
#include "threepp/math/Triangle.hpp"

namespace threepp {

    class BufferGeometry;

    class BVH {

    public:
        BVH(int maxTrianglesPerNode = 8, int maxSubdivisions = 10)
            : maxTrianglesPerNode(maxTrianglesPerNode), maxSubdivisions(maxSubdivisions) {}

        void build(const BufferGeometry& geometry);

        // Intersect this BVH with another BVH
        [[nodiscard]] std::vector<std::pair<int, int>> intersect(const BVH& other) const;

        // Helper methods for single-shape intersections
        [[nodiscard]] std::vector<int> intersect(const Box3& box) const;

        [[nodiscard]] std::vector<int> intersect(const Sphere& sphere) const;

        // Simple true/false intersection test with another BVH
        [[nodiscard]] static bool intersects(const BVH& b1, const Matrix4& m1, const BVH& b2, const Matrix4& m2);

        void collectBoxes(std::vector<Box3>& boxes) const;

        [[nodiscard]] const BufferGeometry* getGeometry() const;

    private:
        class BVHNode {
        public:
            Box3 boundingBox;
            std::unique_ptr<BVHNode> left;
            std::unique_ptr<BVHNode> right;
            std::vector<int> triangleIndices;

            BVHNode() = default;
        };

        std::unique_ptr<BVHNode> root;
        int maxTrianglesPerNode;
        int maxSubdivisions;

        std::vector<Triangle> triangles;
        const BufferGeometry* geometry = nullptr;

        std::unique_ptr<BVHNode> buildNode(std::vector<int>& indices, int depth);

        // Tests intersection between two BVH nodes
        void intersectBVHNodes(const BVHNode* nodeA, const BVHNode* nodeB, std::vector<std::pair<int, int>>& results) const;

        static void collectBoxes(const BVHNode* node, std::vector<Box3>& boxes);
    };


}// namespace threepp

#endif//THREEPPP_BVH_HPP
