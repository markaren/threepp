#include "threepp/objects/BVH.hpp"

#include <algorithm>
#include <functional>

using namespace threepp;


std::unique_ptr<BVH::BVHNode> BVH::buildNode(std::vector<int>& indices, int depth) {
    auto node = std::make_unique<BVHNode>();

    // Calculate bounding box for all triangles in this node
    if (!indices.empty()) {
        // Initialize with first triangle's bounding box
        Box3 box;
        box.setFromPoints(std::vector{
                triangles[indices[0]].a(),
                triangles[indices[0]].b(),
                triangles[indices[0]].c()});
        node->boundingBox = box;

        // Union with remaining triangles
        for (size_t i = 1; i < indices.size(); i++) {
            box.setFromPoints(std::vector{
                    triangles[indices[i]].a(),
                    triangles[indices[i]].b(),
                    triangles[indices[i]].c()});
            node->boundingBox.union_(box);
        }
    }

    // Stopping criteria: max depth reached or few enough triangles
    if (depth >= maxSubdivisions || indices.size() <= maxTrianglesPerNode) {
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
    std::sort(indices.begin(), indices.end(), [this, axis](int a, int b) {
        Vector3 centerA, centerB;
        triangles[a].getMidpoint(centerA);
        triangles[b].getMidpoint(centerB);
        return centerA[axis] < centerB[axis];
    });

    // Split triangles into two groups
    auto mid = indices.size() / 2;
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
            int a = index->getX(i);
            int b = index->getX(i + 1);
            int c = index->getX(i + 2);

            Triangle tri(
                    Vector3(posAttr->getX(a), posAttr->getY(a), posAttr->getZ(a)),
                    Vector3(posAttr->getX(b), posAttr->getY(b), posAttr->getZ(b)),
                    Vector3(posAttr->getX(c), posAttr->getY(c), posAttr->getZ(c)));

            triangles.push_back(tri);
            indices.push_back(triangles.size() - 1);
        }
    } else {
        for (int i = 0; i < posAttr->count(); i += 3) {
            Triangle tri(
                    Vector3(posAttr->getX(i), posAttr->getY(i), posAttr->getZ(i)),
                    Vector3(posAttr->getX(i + 1), posAttr->getY(i + 1), posAttr->getZ(i + 1)),
                    Vector3(posAttr->getX(i + 2), posAttr->getY(i + 2), posAttr->getZ(i + 2)));

            triangles.push_back(tri);
            indices.push_back(triangles.size() - 1);
        }
    }

    // Build the BVH from triangles
    root = buildNode(indices, 0);
}

std::vector<std::pair<int, int>> BVH::intersect(const BVH& other) const {
    std::vector<std::pair<int, int>> results;

    // Test intersection between the two BVH trees
    intersectBVHNodes(root.get(), other.root.get(), results);

    return results;
}

std::vector<int> BVH::intersect(const Box3& box) const {
    std::vector<int> results;

    std::function<void(const BVHNode*)> traverse = [&](const BVHNode* node) {
        if (!node) return;

        if (!node->boundingBox.intersectsBox(box)) {
            return;
        }

        if (node->left == nullptr && node->right == nullptr) {
            for (int idx : node->triangleIndices) {
                Box3 triBox;
                triBox.setFromPoints(std::vector{
                        triangles[idx].a(),
                        triangles[idx].b(),
                        triangles[idx].c()});

                if (box.intersectsBox(triBox)) {
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

std::vector<int> BVH::intersect(const Sphere& sphere) const {
    std::vector<int> results;

    std::function<void(const BVHNode*)> traverse = [&](const BVHNode* node) {
        if (!node) return;

        if (!node->boundingBox.intersectsSphere(sphere)) {
            return;
        }

        if (node->left == nullptr && node->right == nullptr) {
            for (int idx : node->triangleIndices) {
                Vector3 closestPoint;
                triangles[idx].closestPointToPoint(sphere.center, closestPoint);
                float distSq = closestPoint.distanceToSquared(sphere.center);

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

bool BVH::intersects(const BVH& b1, const Matrix4& m1, const BVH& b2, const Matrix4& m2) {
    if (!b1.root || !b2.root) return false;

    // Create transformed copies of the bounding boxes
    Box3 thisRootBox = b1.root->boundingBox.clone().applyMatrix4(m1);
    Box3 otherRootBox = b2.root->boundingBox.clone().applyMatrix4(m2);

    // Quick rejection test
    if (!thisRootBox.intersectsBox(otherRootBox)) {
        return false;
    }

    // Helper function to test transformed triangles
    std::function<bool(const BVHNode*, const BVHNode*)> testNodes = [&b1, &b2, &m1, &m2, &testNodes](const BVHNode* nodeA, const BVHNode* nodeB) -> bool {
        // Transform bounding boxes for this test
        Box3 boxA = nodeA->boundingBox.clone().applyMatrix4(m1);
        Box3 boxB = nodeB->boundingBox.clone().applyMatrix4(m2);

        if (!boxA.intersectsBox(boxB)) {
            return false;
        }

        // If both nodes are leaves, test triangles
        if (nodeA->left == nullptr && nodeA->right == nullptr &&
            nodeB->left == nullptr && nodeB->right == nullptr) {

            for (int idxA : nodeA->triangleIndices) {
                // Transform triangle A to world space
                Triangle triA = b1.triangles[idxA];
                Vector3 a = triA.a().clone().applyMatrix4(m1);
                Vector3 b = triA.b().clone().applyMatrix4(m1);
                Vector3 c = triA.c().clone().applyMatrix4(m1);
                Triangle worldTriA(a, b, c);

                for (int idxB : nodeB->triangleIndices) {
                    // Transform triangle B to world space
                    Triangle triB = b2.triangles[idxB];
                    Vector3 d = triB.a().clone().applyMatrix4(m2);
                    Vector3 e = triB.b().clone().applyMatrix4(m2);
                    Vector3 f = triB.c().clone().applyMatrix4(m2);
                    Triangle worldTriB(d, e, f);

                    // Use bounding box test for simplicity
                    Box3 boxTriA, boxTriB;
                    boxTriA.setFromPoints(std::vector{a, b, c});
                    boxTriB.setFromPoints(std::vector{d, e, f});

                    if (boxTriA.intersectsBox(boxTriB)) {
                        return true;
                    }
                }
            }
            return false;
        }

        // Recursively descend into smaller node first (heuristic)
        Vector3 sizeA, sizeB;
        boxA.getSize(sizeA);
        boxB.getSize(sizeB);
        float volumeA = sizeA.x * sizeA.y * sizeA.z;
        float volumeB = sizeB.x * sizeB.y * sizeB.z;

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

void BVH::intersectBVHNodes(const BVHNode* nodeA, const BVHNode* nodeB, std::vector<std::pair<int, int>>& results) const {
    if (!nodeA || !nodeB) return;

    // Quick rejection test using bounding boxes
    if (!nodeA->boundingBox.intersectsBox(nodeB->boundingBox)) {
        return;
    }

    // If both nodes are leaves, test all triangle pairs
    if (nodeA->left == nullptr && nodeA->right == nullptr &&
        nodeB->left == nullptr && nodeB->right == nullptr) {

        for (int idxA : nodeA->triangleIndices) {
            for (int idxB : nodeB->triangleIndices) {
                // Could implement detailed triangle-triangle intersection here
                // For now, using bounding box test as an approximation
                Box3 boxA, boxB;
                boxA.setFromPoints(std::vector{
                        triangles[idxA].a(),
                        triangles[idxA].b(),
                        triangles[idxA].c()});
                boxB.setFromPoints(std::vector{
                        triangles[idxB].a(),
                        triangles[idxB].b(),
                        triangles[idxB].c()});

                if (boxA.intersectsBox(boxB)) {
                    results.emplace_back(idxA, idxB);
                }
            }
        }
        return;
    }

    // Recursively descend into smaller node first (heuristic)
    Vector3 sizeA, sizeB;
    nodeA->boundingBox.getSize(sizeA);
    nodeB->boundingBox.getSize(sizeB);
    float volumeA = sizeA.x * sizeA.y * sizeA.z;
    float volumeB = sizeB.x * sizeB.y * sizeB.z;

    if (volumeA < volumeB) {
        // A is smaller, descend A
        if (nodeA->left && nodeA->right) {
            intersectBVHNodes(nodeA->left.get(), nodeB, results);
            intersectBVHNodes(nodeA->right.get(), nodeB, results);
        } else {
            // B must have children, descend B
            intersectBVHNodes(nodeA, nodeB->left.get(), results);
            intersectBVHNodes(nodeA, nodeB->right.get(), results);
        }
    } else {
        // B is smaller, descend B
        if (nodeB->left && nodeB->right) {
            intersectBVHNodes(nodeA, nodeB->left.get(), results);
            intersectBVHNodes(nodeA, nodeB->right.get(), results);
        } else {
            // A must have children, descend A
            intersectBVHNodes(nodeA->left.get(), nodeB, results);
            intersectBVHNodes(nodeA->right.get(), nodeB, results);
        }
    }
}

bool BVH::trianglesIntersect(const Triangle& a, const Triangle& b) {
    // For simplicity, we're using a bounding box test
    // A more accurate triangle-triangle intersection test could be implemented
    Box3 boxA, boxB;
    boxA.setFromPoints(std::vector{a.a(), a.b(), a.c()});
    boxB.setFromPoints(std::vector{b.a(), b.b(), b.c()});

    return boxA.intersectsBox(boxB);
}

void BVH::collectBoxes(const BVHNode* node, std::vector<std::pair<Box3, Vector3>>& boxes) {
    if (!node) return;
    Box3 bb = node->boundingBox;
    boxes.emplace_back(bb, bb.getCenter());
    collectBoxes(node->left.get(), boxes);
    collectBoxes(node->right.get(), boxes);
}

const BufferGeometry* BVH::getGeometry() const {
    return geometry;
}
