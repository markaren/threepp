#include "BVH.hpp"

#include "threepp/core/BufferGeometry.hpp"

#include <algorithm>
#include <functional>

using namespace threepp;


std::unique_ptr<BVH::BVHNode> BVH::buildNode(std::vector<int>& indices, int depth) {
    auto node = std::make_unique<BVHNode>();

    // Calculate bounding box for all triangles in this node
    if (!indices.empty()) {
        // Initialize with first triangle's bounding box
        Box3 tmpBox;
        tmpBox.expandByPoint(triangles[indices[0]].a());
        tmpBox.expandByPoint(triangles[indices[0]].b());
        tmpBox.expandByPoint(triangles[indices[0]].c());
        node->boundingBox = tmpBox;

        // Union with remaining triangles
        for (size_t i = 1; i < indices.size(); i++) {
            tmpBox.makeEmpty();
            tmpBox.expandByPoint(triangles[indices[i]].a());
            tmpBox.expandByPoint(triangles[indices[i]].b());
            tmpBox.expandByPoint(triangles[indices[i]].c());
            node->boundingBox.union_(tmpBox);
        }
    }

    // Stopping criteria: max depth reached or few enough triangles
    if (depth >= maxSubdivisions || static_cast<int>(indices.size()) <= maxTrianglesPerNode) {
        node->triangleIndices = indices;
        return node;
    }

    // Find longest axis of the bounding box
    Vector3 size;
    node->boundingBox.getSize(size);

    int axis = 0;
    if (size.y > size.x) axis = 1;
    if (size.z > size[axis]) axis = 2;

    // Sort triangle indices along the longest axis
    Vector3 centerA, centerB;
    std::ranges::sort(indices, [this, axis, &centerA, &centerB](int a, int b) {
        triangles[a].getMidpoint(centerA);
        triangles[b].getMidpoint(centerB);
        return centerA[axis] < centerB[axis];
    });

    // Split triangles into two groups
    const auto mid = static_cast<int>(indices.size() / 2);
    std::vector<int> leftIndices(indices.begin(), indices.begin() + mid);
    std::vector<int> rightIndices(indices.begin() + mid, indices.end());

    // Recursively build child nodes
    node->left = buildNode(leftIndices, depth + 1);
    node->right = buildNode(rightIndices, depth + 1);

    return node;
}

void BVH::build(const BufferGeometry& geom) {
    triangles.clear();
    geometry = &geom;

    const auto posAttr = geom.getAttribute<float>("position");
    if (!posAttr) return;

    std::vector<int> indices;
    if (geom.hasIndex()) {
        const auto index = geom.getIndex();
        for (int i = 0; i < index->count(); i += 3) {
            const auto a = index->getX(i);
            const auto b = index->getX(i + 1);
            const auto c = index->getX(i + 2);

            Triangle tri(
                    Vector3(posAttr->getX(a), posAttr->getY(a), posAttr->getZ(a)),
                    Vector3(posAttr->getX(b), posAttr->getY(b), posAttr->getZ(b)),
                    Vector3(posAttr->getX(c), posAttr->getY(c), posAttr->getZ(c)));

            triangles.emplace_back(tri);
            indices.emplace_back(static_cast<int>(triangles.size()) - 1);
        }
    } else {
        for (int i = 0; i < posAttr->count(); i += 3) {
            Triangle tri(
                    Vector3(posAttr->getX(i), posAttr->getY(i), posAttr->getZ(i)),
                    Vector3(posAttr->getX(i + 1), posAttr->getY(i + 1), posAttr->getZ(i + 1)),
                    Vector3(posAttr->getX(i + 2), posAttr->getY(i + 2), posAttr->getZ(i + 2)));

            triangles.emplace_back(tri);
            indices.emplace_back(static_cast<int>(triangles.size()) - 1);
        }
    }

    // Build the BVH from triangles
    root = buildNode(indices, 0);
}

std::vector<BVH::IntersectionResult> BVH::intersect(const BVH& b1, const Matrix4& m1, const BVH& b2, const Matrix4& m2, bool accurate) {
    std::vector<IntersectionResult> results;

    // Test intersection between the two BVH trees
    intersectBVHNodes(b1, b1.root.get(), m1, b2, b2.root.get(), m2, results, accurate);

    return results;
}

std::vector<int> BVH::intersect(const Box3& box, const Matrix4& m) const {
    std::vector<int> results;

    std::function<void(const BVHNode*)> traverse = [&](const BVHNode* node) {
        if (!node) return;

        if (!node->boundingBox.clone().applyMatrix4(m).intersectsBox(box)) {
            return;
        }

        if (node->left == nullptr && node->right == nullptr) {
            Box3 triBox;

            for (int idx : node->triangleIndices) {
                triBox.makeEmpty();
                triBox.expandByPoint(triangles[idx].a());
                triBox.expandByPoint(triangles[idx].b());
                triBox.expandByPoint(triangles[idx].c());
                triBox.applyMatrix4(m);

                if (box.intersectsBox(triBox)) {
                    results.emplace_back(idx);
                }
            }
            return;
        }

        traverse(node->left.get());
        traverse(node->right.get());
    };

    traverse(root.get());
    return results;
}

std::vector<int> BVH::intersect(const Sphere& sphere, const Matrix4& m) const {
    std::vector<int> results;

    std::function<void(const BVHNode*)> traverse = [&](const BVHNode* node) {
        if (!node) return;

        if (!node->boundingBox.clone().applyMatrix4(m).intersectsSphere(sphere)) {
            return;
        }

        if (node->isLeaf()) {

            Vector3 a, b, c;
            Vector3 closestPoint;
            Triangle worldTri;
            for (const int idx : node->triangleIndices) {
                const Triangle& tri = triangles[idx];
                worldTri.set(
                        a.copy(tri.a()).applyMatrix4(m),
                        b.copy(tri.b()).applyMatrix4(m),
                        c.copy(tri.c()).applyMatrix4(m));
                worldTri.closestPointToPoint(sphere.center, closestPoint);
                const float distSq = closestPoint.distanceToSquared(sphere.center);

                if (distSq <= (sphere.radius * sphere.radius)) {
                    results.push_back(idx);
                }
            }
            return;
        }

        traverse(node->left.get());
        traverse(node->right.get());
    };

    traverse(root.get());
    return results;
}

bool BVH::intersects(const BVH& b1, const BVH& b2, const Matrix4& m1, const Matrix4& m2) {
    if (!b1.root || !b2.root) return false;

    // Create transformed copies of the bounding boxes
    const Box3 thisRootBox = b1.root->boundingBox.clone().applyMatrix4(m1);
    const Box3 otherRootBox = b2.root->boundingBox.clone().applyMatrix4(m2);

    // Quick rejection test
    if (!thisRootBox.intersectsBox(otherRootBox)) {
        return false;
    }

    Box3 boxA, boxB;
    Vector3 sizeA, sizeB;

    // Helper function to test transformed triangles
    std::function<bool(const BVHNode*, const BVHNode*)> testNodes = [&](const BVHNode* nodeA, const BVHNode* nodeB) -> bool {
        // Transform bounding boxes for this test
        boxA.copy(nodeA->boundingBox).applyMatrix4(m1);
        boxB.copy(nodeB->boundingBox).applyMatrix4(m2);

        if (!boxA.intersectsBox(boxB)) {
            return false;
        }

        // If both nodes are leaves, test triangles
        if (nodeA->isLeaf() && nodeB->isLeaf()) {

            Box3 boxTriA, boxTriB;
            for (const int idxA : nodeA->triangleIndices) {

                const Triangle& triA = b1.triangles[idxA];

                boxTriA.makeEmpty();
                boxTriA.expandByPoint(triA.a());
                boxTriA.expandByPoint(triA.b());
                boxTriA.expandByPoint(triA.c());
                boxTriA.applyMatrix4(m1);

                for (const int idxB : nodeB->triangleIndices) {

                    const Triangle& triB = b2.triangles[idxB];

                    boxTriB.makeEmpty();
                    boxTriB.expandByPoint(triB.a());
                    boxTriB.expandByPoint(triB.b());
                    boxTriB.expandByPoint(triB.c());
                    boxTriB.applyMatrix4(m2);

                    if (boxTriA.intersectsBox(boxTriB)) {
                        return true;
                    }
                }
            }
            return false;
        }

        // Recursively descend into smaller node first (heuristic)

        boxA.getSize(sizeA);
        boxB.getSize(sizeB);
        const float volumeA = sizeA.x * sizeA.y * sizeA.z;
        const float volumeB = sizeB.x * sizeB.y * sizeB.z;

        if (volumeA < volumeB) {
            // A is smaller, descend A
            if (nodeA->left && nodeA->right) {
                return testNodes(nodeA->left.get(), nodeB) ||
                       testNodes(nodeA->right.get(), nodeB);
            } else {
                // B must have children, descend B
                return testNodes(nodeA, nodeB->left.get()) ||
                       testNodes(nodeA, nodeB->right.get());
            }
        } else {
            // B is smaller, descend B
            if (nodeB->left && nodeB->right) {
                return testNodes(nodeA, nodeB->left.get()) ||
                       testNodes(nodeA, nodeB->right.get());
            } else {
                // A must have children, descend A
                return testNodes(nodeA->left.get(), nodeB) ||
                       testNodes(nodeA->right.get(), nodeB);
            }
        }
    };

    return testNodes(b1.root.get(), b2.root.get());
}

void BVH::collectBoxes(std::vector<BVHBox3>& boxes) const {
    collectBoxes(root.get(), boxes);
}

void BVH::intersectBVHNodes(const BVH& b1, const BVHNode* nodeA, const Matrix4& m1, const BVH& b2, const BVHNode* nodeB, const Matrix4& m2, std::vector<IntersectionResult>& results, bool accurate) {

    Box3 bb1 = nodeA->boundingBox.clone();
    bb1.applyMatrix4(m1);

    Box3 bb2 = nodeB->boundingBox.clone();
    bb2.applyMatrix4(m2);

    // Quick rejection test using bounding boxes
    if (!bb1.intersectsBox(bb2)) {
        return;
    }

    // If both nodes are leaves, test all triangle pairs
    if (nodeA->isLeaf() && nodeB->isLeaf()) {

        Vector3 center;
        if (accurate) {
            Box3 boxA, boxB, intersectionBox;
            for (const int idxA : nodeA->triangleIndices) {

                const Triangle& triA = b1.triangles[idxA];

                boxA.makeEmpty();
                boxA.expandByPoint(triA.a());
                boxA.expandByPoint(triA.b());
                boxA.expandByPoint(triA.c());
                boxA.applyMatrix4(m1);

                for (const int idxB : nodeB->triangleIndices) {
                    // Could implement detailed triangle-triangle intersection here
                    // For now, using bounding box test as an approximation

                    const Triangle& triB = b2.triangles[idxB];

                    boxB.makeEmpty();
                    boxB.expandByPoint(triB.a());
                    boxB.expandByPoint(triB.b());
                    boxB.expandByPoint(triB.c());
                    boxB.applyMatrix4(m2);

                    if (boxA.intersectsBox(boxB)) {
                        // Compute intersection box
                        intersectionBox.set({std::max(boxA.min().x, boxB.min().x),
                                             std::max(boxA.min().y, boxB.min().y),
                                             std::max(boxA.min().z, boxB.min().z)},
                                            {std::min(boxA.max().x, boxB.max().x),
                                             std::min(boxA.max().y, boxB.max().y),
                                             std::min(boxA.max().z, boxB.max().z)});


                        intersectionBox.getCenter(center);
                        results.emplace_back(IntersectionResult{idxA, idxB, center});
                    }
                }
            }
        } else {
            const Box3 intersectionBox(
                    {std::max(bb1.min().x, bb2.min().x),
                     std::max(bb1.min().y, bb2.min().y),
                     std::max(bb1.min().z, bb2.min().z)},
                    {std::min(bb1.max().x, bb2.max().x),
                     std::min(bb1.max().y, bb2.max().y),
                     std::min(bb1.max().z, bb2.max().z)});

            intersectionBox.getCenter(center);
            // Use -1 for idxA/idxB to indicate node-level intersection
            results.emplace_back(IntersectionResult{-1, -1, center});
        }
        return;
    }

    // Recursively descend into smaller node first (heuristic)
    Vector3 sizeA, sizeB;
    nodeA->boundingBox.getSize(sizeA);
    nodeB->boundingBox.getSize(sizeB);
    const float volumeA = sizeA.x * sizeA.y * sizeA.z;
    const float volumeB = sizeB.x * sizeB.y * sizeB.z;

    if (volumeA < volumeB) {
        // A is smaller, descend A
        if (nodeA->left && nodeA->right) {
            intersectBVHNodes(b1, nodeA->left.get(), m1, b2, nodeB, m2, results, accurate);
            intersectBVHNodes(b1, nodeA->right.get(), m1, b2, nodeB, m2, results, accurate);
        } else {
            // B must have children, descend B
            intersectBVHNodes(b1, nodeA, m1, b2, nodeB->left.get(), m2, results, accurate);
            intersectBVHNodes(b1, nodeA, m1, b2, nodeB->right.get(), m2, results, accurate);
        }
    } else {
        // B is smaller, descend B
        if (nodeB->left && nodeB->right) {
            intersectBVHNodes(b1, nodeA, m1, b2, nodeB->left.get(), m2, results, accurate);
            intersectBVHNodes(b1, nodeA, m1, b2, nodeB->right.get(), m2, results, accurate);
        } else {
            // A must have children, descend A
            intersectBVHNodes(b1, nodeA->left.get(), m1, b2, nodeB, m2, results, accurate);
            intersectBVHNodes(b1, nodeA->right.get(), m1, b2, nodeB, m2, results, accurate);
        }
    }
}

void BVH::collectBoxes(const BVHNode* node, std::vector<BVHBox3>& boxes) {
    if (!node) return;

    boxes.emplace_back(node->boundingBox, node->isLeaf());
    collectBoxes(node->left.get(), boxes);
    collectBoxes(node->right.get(), boxes);
}

const BufferGeometry* BVH::getGeometry() const {
    return geometry;
}
