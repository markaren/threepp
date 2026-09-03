#include "threepp/utils/BVH.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/utils/TriangleIntersect.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace threepp;

namespace {

    // Keeps a segment query between two points that lie exactly on a surface
    // from hitting the surfaces the endpoints sit on.
    constexpr float rayEps = 1e-4f;

    // Transforms are passed as `const Matrix4*`, null meaning identity, so the
    // identity case skips the per-node box and per-triangle transforms.
    const Matrix4* nonIdentity(const Matrix4& m) {
        static const Matrix4 identity;
        return m.elements == identity.elements ? nullptr : &m;
    }

    void transformBox(const Box3& src, const Matrix4* m, Box3& dst) {
        dst.copy(src);
        if (m) dst.applyMatrix4(*m);
    }

    void transformTriangle(const Triangle& src, const Matrix4* m, Triangle& dst) {
        if (!m) {
            dst.set(src.a(), src.b(), src.c());
            return;
        }
        Vector3 a{src.a()}, b{src.b()}, c{src.c()};
        dst.set(a.applyMatrix4(*m), b.applyMatrix4(*m), c.applyMatrix4(*m));
    }

    Box3 triangleBox(const Triangle& t) {
        Box3 box;
        box.expandByPoint(t.a());
        box.expandByPoint(t.b());
        box.expandByPoint(t.c());
        return box;
    }

    // One leaf's triangles, transformed and boxed. Keyed on the address of the
    // node's index vector, so a leaf tested against several opposing leaves is
    // transformed once; the buffers are reused across the traversal.
    struct LeafCache {
        const void* key{};
        std::vector<Triangle> triangles;
        std::vector<Box3> boxes;

        void load(const std::vector<Triangle>& source, const std::vector<int>& indices, const Matrix4* m) {
            if (key == &indices) return;
            key = &indices;
            triangles.resize(indices.size());
            boxes.resize(indices.size());
            for (std::size_t i = 0; i < indices.size(); ++i) {
                transformTriangle(source[indices[i]], m, triangles[i]);
                boxes[i] = triangleBox(triangles[i]);
            }
        }

        [[nodiscard]] std::size_t size() const {
            return triangles.size();
        }
    };

}// namespace


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
    // Reset before the early return: the old tree would otherwise index a cleared triangle list.
    root.reset();
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

void BVH::build(const std::vector<float>& positions, const std::vector<unsigned int>& indices) {
    triangles.clear();
    // The triangles are copied; there is no geometry to point at.
    geometry = nullptr;
    root.reset();

    const auto vertexCount = positions.size() / 3;
    const auto vertex = [&positions](std::size_t i) {
        return Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    };

    std::vector<int> triangleIndices;

    if (!indices.empty()) {
        if (indices.size() % 3 != 0) {
            throw std::invalid_argument("BVH::build: index count must be a multiple of 3");
        }
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            const auto a = indices[i], b = indices[i + 1], c = indices[i + 2];
            if (a >= vertexCount || b >= vertexCount || c >= vertexCount) {
                throw std::out_of_range("BVH::build: index out of range for the given positions");
            }
            triangles.emplace_back(vertex(a), vertex(b), vertex(c));
            triangleIndices.emplace_back(static_cast<int>(triangles.size()) - 1);
        }
    } else {
        if (positions.size() % 9 != 0) {
            throw std::invalid_argument("BVH::build: a non-indexed soup needs 9 floats (3 vertices) per triangle");
        }
        for (std::size_t i = 0; i + 2 < vertexCount; i += 3) {
            triangles.emplace_back(vertex(i), vertex(i + 1), vertex(i + 2));
            triangleIndices.emplace_back(static_cast<int>(triangles.size()) - 1);
        }
    }

    root = buildNode(triangleIndices, 0);
}

std::vector<BVH::IntersectionResult> BVH::intersect(const BVH& b1, const Matrix4& m1, const BVH& b2, const Matrix4& m2, bool accurate) {
    std::vector<IntersectionResult> results;
    if (!b1.root || !b2.root) return results;

    const Matrix4* t1 = nonIdentity(m1);
    const Matrix4* t2 = nonIdentity(m2);

    LeafCache cacheA, cacheB;
    Vector3 center;

    const auto visit = [&](auto&& self, const BVHNode* nodeA, const BVHNode* nodeB) -> void {
        Box3 bb1, bb2;
        transformBox(nodeA->boundingBox, t1, bb1);
        transformBox(nodeB->boundingBox, t2, bb2);

        // Quick rejection test using bounding boxes
        if (!bb1.intersectsBox(bb2)) return;

        if (nodeA->isLeaf() && nodeB->isLeaf()) {

            if (accurate) {
                // One result per intersecting triangle pair; `position` lies on both surfaces.
                cacheA.load(b1.triangles, nodeA->triangleIndices, t1);
                cacheB.load(b2.triangles, nodeB->triangleIndices, t2);

                for (std::size_t i = 0; i < cacheA.size(); ++i) {
                    if (!cacheA.boxes[i].intersectsBox(bb2)) continue;
                    for (std::size_t j = 0; j < cacheB.size(); ++j) {
                        if (!cacheA.boxes[i].intersectsBox(cacheB.boxes[j])) continue;
                        if (detail::triTriIntersectionPoint(cacheA.triangles[i], cacheB.triangles[j], center)) {
                            results.emplace_back(IntersectionResult{
                                    nodeA->triangleIndices[i], nodeB->triangleIndices[j], center});
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

        // Split the side with the larger box (if it has children).
        Vector3 sizeA, sizeB;
        bb1.getSize(sizeA);
        bb2.getSize(sizeB);
        const float volumeA = sizeA.x * sizeA.y * sizeA.z;
        const float volumeB = sizeB.x * sizeB.y * sizeB.z;

        const bool descendA = nodeA->left && nodeA->right && (volumeA >= volumeB || !(nodeB->left && nodeB->right));
        if (descendA) {
            self(self, nodeA->left.get(), nodeB);
            self(self, nodeA->right.get(), nodeB);
        } else {
            self(self, nodeA, nodeB->left.get());
            self(self, nodeA, nodeB->right.get());
        }
    };

    visit(visit, b1.root.get(), b2.root.get());
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

    const Matrix4* t1 = nonIdentity(m1);
    const Matrix4* t2 = nonIdentity(m2);

    Box3 rootA, rootB;
    transformBox(b1.root->boundingBox, t1, rootA);
    transformBox(b2.root->boundingBox, t2, rootB);
    if (!rootA.intersectsBox(rootB)) return false;

    LeafCache cacheA, cacheB;

    // Generic recursive lambda rather than std::function: called once per
    // unseparated node pair, where the indirect call is a measurable share of the query.
    const auto testNodes = [&](auto&& self, const BVHNode* nodeA, const BVHNode* nodeB) -> bool {
        Box3 boxA, boxB;
        transformBox(nodeA->boundingBox, t1, boxA);
        transformBox(nodeB->boundingBox, t2, boxB);

        if (!boxA.intersectsBox(boxB)) return false;

        if (nodeA->isLeaf() && nodeB->isLeaf()) {

            // Exact triangle test at the leaves; a box-only test reports near misses as hits.
            cacheA.load(b1.triangles, nodeA->triangleIndices, t1);
            cacheB.load(b2.triangles, nodeB->triangleIndices, t2);

            for (std::size_t i = 0; i < cacheA.size(); ++i) {
                if (!cacheA.boxes[i].intersectsBox(boxB)) continue;
                for (std::size_t j = 0; j < cacheB.size(); ++j) {
                    if (!cacheA.boxes[i].intersectsBox(cacheB.boxes[j])) continue;
                    if (detail::triTriOverlap(cacheA.triangles[i], cacheB.triangles[j])) return true;
                }
            }
            return false;
        }

        // Split the side with the larger box (if it has children).
        Vector3 sizeA, sizeB;
        boxA.getSize(sizeA);
        boxB.getSize(sizeB);
        const float volumeA = sizeA.x * sizeA.y * sizeA.z;
        const float volumeB = sizeB.x * sizeB.y * sizeB.z;

        const bool descendA = nodeA->left && nodeA->right && (volumeA >= volumeB || !(nodeB->left && nodeB->right));
        if (descendA) {
            return self(self, nodeA->left.get(), nodeB) ||
                   self(self, nodeA->right.get(), nodeB);
        }
        return self(self, nodeA, nodeB->left.get()) ||
               self(self, nodeA, nodeB->right.get());
    };

    return testNodes(testNodes, b1.root.get(), b2.root.get());
}

std::optional<BVH::RayHit> BVH::raycast(const Ray& ray, float maxDistance) const {
    if (!root) return std::nullopt;

    std::optional<RayHit> best;
    float bestDistance = maxDistance - rayEps;

    Vector3 point, boxPoint, edge1, edge2, normal;

    std::function<void(const BVHNode*)> traverse = [&](const BVHNode* node) {
        if (!node) return;

        // A node entered farther away than the current best cannot improve it.
        // The entry distance is only meaningful from outside the box.
        if (!node->boundingBox.containsPoint(ray.origin)) {
            ray.intersectBox(node->boundingBox, boxPoint);
            if (boxPoint.isNan()) return;
            if (boxPoint.distanceTo(ray.origin) > bestDistance) return;
        }

        if (node->isLeaf()) {

            for (const int idx : node->triangleIndices) {
                const Triangle& tri = triangles[idx];

                if (!ray.intersectTriangle(tri.a(), tri.b(), tri.c(), false, point)) continue;

                const float distance = point.distanceTo(ray.origin);
                if (distance < rayEps || distance > bestDistance) continue;

                edge1.subVectors(tri.b(), tri.a());
                edge2.subVectors(tri.c(), tri.a());
                normal.crossVectors(edge1, edge2).normalize();
                if (normal.dot(ray.direction) > 0) normal.negate();

                bestDistance = distance;
                best = RayHit{distance, idx, point, normal};
            }
            return;
        }

        traverse(node->left.get());
        traverse(node->right.get());
    };

    traverse(root.get());
    return best;
}

bool BVH::raycastAny(const Ray& ray, float maxDistance) const {
    if (!root) return false;

    const float limit = maxDistance - rayEps;

    Vector3 point;

    std::function<bool(const BVHNode*)> traverse = [&](const BVHNode* node) -> bool {
        if (!node) return false;

        if (!ray.intersectsBox(node->boundingBox)) return false;

        if (node->isLeaf()) {

            for (const int idx : node->triangleIndices) {
                const Triangle& tri = triangles[idx];

                if (!ray.intersectTriangle(tri.a(), tri.b(), tri.c(), false, point)) continue;

                const float distance = point.distanceTo(ray.origin);
                if (distance < rayEps || distance > limit) continue;

                return true;
            }
            return false;
        }

        return traverse(node->left.get()) || traverse(node->right.get());
    };

    return traverse(root.get());
}

void BVH::collectBoxes(std::vector<BVHBox3>& boxes) const {
    collectBoxes(root.get(), boxes);
}

float BVH::distance(const BVH& b1, const BVH& b2, const Matrix4& m1, const Matrix4& m2, float maxDistance) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    if (!b1.root || !b2.root || b1.triangles.empty() || b2.triangles.empty()) return inf;
    if (maxDistance <= 0.f) return inf;

    const Matrix4* t1 = nonIdentity(m1);
    const Matrix4* t2 = nonIdentity(m2);

    const float limitSq = maxDistance == inf ? inf : maxDistance * maxDistance;
    float bestSq = limitSq;
    LeafCache cacheA, cacheB;

    // Depth-first with a running best: node pairs whose boxes are farther apart
    // than bestSq are pruned, and the nearer child is visited first so bestSq drops early.
    const auto visit = [&](auto&& self, const BVHNode* nodeA, const Box3& boxA,
                           const BVHNode* nodeB, const Box3& boxB) -> void {
        if (bestSq <= 0.f) return;
        if (detail::boxDistanceSq(boxA, boxB) >= bestSq) return;

        if (nodeA->isLeaf() && nodeB->isLeaf()) {
            cacheA.load(b1.triangles, nodeA->triangleIndices, t1);
            cacheB.load(b2.triangles, nodeB->triangleIndices, t2);

            for (std::size_t i = 0; i < cacheA.size(); ++i) {
                if (detail::boxDistanceSq(cacheA.boxes[i], boxB) >= bestSq) continue;

                for (std::size_t j = 0; j < cacheB.size(); ++j) {
                    if (detail::boxDistanceSq(cacheA.boxes[i], cacheB.boxes[j]) >= bestSq) continue;

                    bestSq = std::min(bestSq, detail::triTriDistanceSq(cacheA.triangles[i], cacheB.triangles[j]));
                    if (bestSq <= 0.f) return;
                }
            }
            return;
        }

        // Split whichever side still has children, preferring the bigger box.
        Vector3 sizeA, sizeB;
        boxA.getSize(sizeA);
        boxB.getSize(sizeB);
        const float volumeA = sizeA.x * sizeA.y * sizeA.z;
        const float volumeB = sizeB.x * sizeB.y * sizeB.z;
        const bool splitA = nodeA->left && nodeA->right && (volumeA >= volumeB || !(nodeB->left && nodeB->right));

        const BVHNode* first;
        const BVHNode* second;
        Box3 firstBox, secondBox;
        if (splitA) {
            transformBox(nodeA->left->boundingBox, t1, firstBox);
            transformBox(nodeA->right->boundingBox, t1, secondBox);
            first = nodeA->left.get();
            second = nodeA->right.get();
            if (detail::boxDistanceSq(secondBox, boxB) < detail::boxDistanceSq(firstBox, boxB)) {
                std::swap(first, second);
                std::swap(firstBox, secondBox);
            }
            self(self, first, firstBox, nodeB, boxB);
            self(self, second, secondBox, nodeB, boxB);
        } else {
            transformBox(nodeB->left->boundingBox, t2, firstBox);
            transformBox(nodeB->right->boundingBox, t2, secondBox);
            first = nodeB->left.get();
            second = nodeB->right.get();
            if (detail::boxDistanceSq(boxA, secondBox) < detail::boxDistanceSq(boxA, firstBox)) {
                std::swap(first, second);
                std::swap(firstBox, secondBox);
            }
            self(self, nodeA, boxA, first, firstBox);
            self(self, nodeA, boxA, second, secondBox);
        }
    };

    Box3 rootA, rootB;
    transformBox(b1.root->boundingBox, t1, rootA);
    transformBox(b2.root->boundingBox, t2, rootB);
    visit(visit, b1.root.get(), rootA, b2.root.get(), rootB);

    // Nothing closer than the cutoff.
    return bestSq >= limitSq ? inf : std::sqrt(bestSq);
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
