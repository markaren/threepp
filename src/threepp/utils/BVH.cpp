#include "threepp/utils/BVH.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/utils/TriangleIntersect.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace threepp;

namespace {

    // Keeps a segment query between two points that lie exactly on a surface
    // from hitting the surfaces the endpoints sit on.
    constexpr float rayEps = 1e-4f;

    // Centroid bins per axis for the surface-area heuristic.
    constexpr int sahBins = 16;

    // The depth limit's range, and the traversal stack that limit bounds: a
    // near-first walk holds at most one deferred node per level.
    constexpr int minDepthLimit = 64;
    constexpr int maxDepthLimit = 128;
    constexpr int stackSize = maxDepthLimit + 2;

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

    // One leaf's triangles, transformed and boxed. Keyed on the leaf's first
    // triangle, so a leaf tested against several opposing leaves is transformed
    // once; the buffers are reused across the traversal.
    struct LeafCache {
        const void* key{};
        std::vector<Triangle> triangles;
        std::vector<Box3> boxes;

        void load(const std::vector<Triangle>& source, int first, int count, const Matrix4* m) {
            if (key == &source[first]) return;
            key = &source[first];
            triangles.resize(count);
            boxes.resize(count);
            for (int i = 0; i < count; ++i) {
                transformTriangle(source[first + i], m, triangles[i]);
                boxes[i] = triangleBox(triangles[i]);
            }
        }

        [[nodiscard]] std::size_t size() const {
            return triangles.size();
        }
    };

    float halfArea(const float* mn, const float* mx) {
        const float dx = std::max(0.f, mx[0] - mn[0]);
        const float dy = std::max(0.f, mx[1] - mn[1]);
        const float dz = std::max(0.f, mx[2] - mn[2]);
        return dx * dy + dy * dz + dz * dx;
    }

    // The centroid bin a triangle falls in along one axis. A NaN position (a
    // centroid whose sum overflowed, or an infinite offset times the zero scale
    // of an infinite centroid range) goes to bin 0 instead of reaching a
    // float-to-int conversion, which is undefined for NaN and infinity.
    int centroidBin(float centroid, float lo, float scale) {
        const float x = (centroid - lo) * scale;
        if (!(x > 0.f)) return 0;
        if (!(x < static_cast<float>(sahBins - 1))) return sahBins - 1;
        return static_cast<int>(x);
    }

    struct Bounds {
        float min[3]{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        float max[3]{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};

        void grow(const float* mn, const float* mx) {
            for (int k = 0; k < 3; ++k) {
                min[k] = std::min(min[k], mn[k]);
                max[k] = std::max(max[k], mx[k]);
            }
        }
    };

    // The ray as the traversal reads it: origin, direction and a finite inverse
    // direction (a tiny component's inverse is clamped to +-1e30, so an origin
    // lying on a slab plane gives 0 * 1e30 = 0 rather than 0 * inf = NaN).
    struct RayData {
        float o[3];
        float d[3];
        float inv[3];
        bool still[3];// no motion along this axis

        explicit RayData(const Ray& ray) {
            for (int k = 0; k < 3; ++k) {
                o[k] = ray.origin[k];
                d[k] = ray.direction[k];
                still[k] = d[k] == 0.f;
                const float i = 1.f / d[k];
                inv[k] = std::abs(i) > 1e30f ? std::copysign(1e30f, d[k]) : i;
            }
        }

        // Entry distance (in units of t) into the box, when the ray meets it within [0, tLimit].
        // An axis the ray does not move along has no slab interval: the box is
        // met only if the origin lies within it on that axis, faces included.
        // (A 1e30 inverse there gave an origin on a face the interval [-big, 0]
        // on one face or the other, depending on the sign of the zero.)
        [[nodiscard]] bool slab(const float* mn, const float* mx, float tLimit, float& tEntry) const {
            float tNear = -std::numeric_limits<float>::infinity();
            float tFar = std::numeric_limits<float>::infinity();
            for (int k = 0; k < 3; ++k) {
                if (still[k]) {
                    if (!(o[k] >= mn[k] && o[k] <= mx[k])) return false;
                    continue;
                }
                const float t0 = (mn[k] - o[k]) * inv[k], t1 = (mx[k] - o[k]) * inv[k];
                tNear = std::max(tNear, std::min(t0, t1));
                tFar = std::min(tFar, std::max(t0, t1));
            }
            tEntry = tNear;
            return tNear <= tFar && tFar >= 0.f && tNear <= tLimit;
        }
    };

}// namespace


void BVH::buildTree() {
    nodes.clear();
    rayTriangles.clear();
    sourceIndex.clear();

    const auto n = static_cast<int>(triangles.size());
    if (n == 0) return;

    // Per-triangle bounds and centroids, in source order. A triangle with a
    // NaN or infinite vertex (a no-data value) is left out of `order`, so no
    // node holds it and no query reports it; sourceIndex still maps the rest
    // to their positions in the source.
    std::vector<float> triMin(3 * static_cast<std::size_t>(n)), triMax(triMin.size()), centroid(triMin.size());
    std::vector<int> order;
    order.reserve(n);
    for (int i = 0; i < n; ++i) {
        const Triangle& t = triangles[i];
        const Vector3* v[3] = {&t.a(), &t.b(), &t.c()};
        bool finite = true;
        for (int k = 0; k < 3; ++k) {
            const float a = (*v[0])[k], b = (*v[1])[k], c = (*v[2])[k];
            finite = finite && std::isfinite(a) && std::isfinite(b) && std::isfinite(c);
            triMin[i * 3 + k] = std::min({a, b, c});
            triMax[i * 3 + k] = std::max({a, b, c});
            centroid[i * 3 + k] = (a + b + c) / 3.f;
        }
        if (finite) order.push_back(i);
    }

    const auto kept = static_cast<int>(order.size());
    if (kept == 0) {
        triangles.clear();
        return;
    }

    const int leafSize = std::max(1, maxTrianglesPerNode);
    const int depthLimit = std::clamp(maxSubdivisions, minDepthLimit, maxDepthLimit);

    nodes.reserve(2 * static_cast<std::size_t>(kept));
    nodes.emplace_back();

    const auto split = [&](auto&& self, int index, int begin, int end, int depth) -> void {
        const int count = end - begin;

        Bounds bounds, centroids;
        for (int i = begin; i < end; ++i) {
            const int t = order[i];
            bounds.grow(&triMin[t * 3], &triMax[t * 3]);
            centroids.grow(&centroid[t * 3], &centroid[t * 3]);
        }

        Node node{};
        std::copy_n(bounds.min, 3, node.min);
        std::copy_n(bounds.max, 3, node.max);

        const auto makeLeaf = [&] {
            node.first = begin;
            node.count = count;
            nodes[index] = node;
        };

        if (count <= leafSize || depth >= depthLimit) {
            makeLeaf();
            return;
        }

        // Binned SAH: the plane between centroid bins that minimises area x count
        // of the two sides, against area x count of the node kept as a leaf.
        int bestAxis = -1, bestSplit = 0;
        float bestCost = static_cast<float>(count) * halfArea(bounds.min, bounds.max);
        for (int axis = 0; axis < 3; ++axis) {
            const float extent = centroids.max[axis] - centroids.min[axis];
            if (!(extent > 0.f)) continue;

            Bounds binBounds[sahBins];
            int binCount[sahBins]{};
            const float scale = sahBins / extent;
            for (int i = begin; i < end; ++i) {
                const int t = order[i];
                const int b = centroidBin(centroid[t * 3 + axis], centroids.min[axis], scale);
                ++binCount[b];
                binBounds[b].grow(&triMin[t * 3], &triMax[t * 3]);
            }

            float leftCost[sahBins - 1];
            Bounds sweep;
            int sweepCount = 0;
            for (int b = 0; b < sahBins - 1; ++b) {
                sweepCount += binCount[b];
                if (binCount[b]) sweep.grow(binBounds[b].min, binBounds[b].max);
                leftCost[b] = sweepCount ? static_cast<float>(sweepCount) * halfArea(sweep.min, sweep.max) : -1.f;
            }
            sweep = Bounds{};
            sweepCount = 0;
            for (int b = sahBins - 1; b > 0; --b) {
                sweepCount += binCount[b];
                if (binCount[b]) sweep.grow(binBounds[b].min, binBounds[b].max);
                if (!sweepCount || leftCost[b - 1] < 0.f) continue;
                const float cost = leftCost[b - 1] + static_cast<float>(sweepCount) * halfArea(sweep.min, sweep.max);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplit = b;
                }
            }
        }

        int mid;
        if (bestAxis >= 0) {
            const float lo = centroids.min[bestAxis];
            const float scale = sahBins / (centroids.max[bestAxis] - lo);
            const auto it = std::partition(order.begin() + begin, order.begin() + end, [&](int t) {
                return centroidBin(centroid[t * 3 + bestAxis], lo, scale) < bestSplit;
            });
            mid = static_cast<int>(it - order.begin());
            if (mid == begin || mid == end) mid = begin + count / 2;
        } else {
            // Nothing beats a leaf. A small one stays a leaf; a big one takes an
            // object-median split on the widest centroid axis.
            if (count <= 2 * leafSize) {
                makeLeaf();
                return;
            }
            int axis = 0;
            for (int k = 1; k < 3; ++k) {
                if (centroids.max[k] - centroids.min[k] > centroids.max[axis] - centroids.min[axis]) axis = k;
            }
            mid = begin + count / 2;
            std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
                             [&](int a, int b) { return centroid[a * 3 + axis] < centroid[b * 3 + axis]; });
        }

        const auto left = static_cast<int>(nodes.size());
        nodes.emplace_back();
        nodes.emplace_back();
        node.first = left;
        node.count = 0;
        nodes[index] = node;
        self(self, left, begin, mid, depth + 1);
        self(self, left + 1, mid, end, depth + 1);
    };

    split(split, 0, 0, kept, 0);
    nodes.shrink_to_fit();

    // Reorder into leaf order, keeping the way back to the source index.
    std::vector<Triangle> sorted;
    sorted.reserve(kept);
    rayTriangles.resize(kept);
    for (int i = 0; i < kept; ++i) {
        const Triangle& t = triangles[order[i]];
        sorted.emplace_back(t);
        RayTriangle& r = rayTriangles[i];
        for (int k = 0; k < 3; ++k) {
            r.v0[k] = t.a()[k];
            r.e1[k] = t.b()[k] - t.a()[k];
            r.e2[k] = t.c()[k] - t.a()[k];
        }
    }
    triangles = std::move(sorted);
    sourceIndex = std::move(order);
}

void BVH::build(const BufferGeometry& geom) {
    triangles.clear();
    // Reset before the early return: the old tree would otherwise index a cleared triangle list.
    buildTree();
    geometry = &geom;

    const auto posAttr = geom.getAttribute<float>("position");
    if (!posAttr) return;

    // Whole triangles only: one or two indices (or vertices) left over at the
    // end are not a triangle, and reading their missing corners would run past
    // the end of the array.
    if (geom.hasIndex()) {
        const auto index = geom.getIndex();
        const int count = index->count() - index->count() % 3;
        triangles.reserve(count / 3);
        for (int i = 0; i < count; i += 3) {
            const auto a = index->getX(i);
            const auto b = index->getX(i + 1);
            const auto c = index->getX(i + 2);

            triangles.emplace_back(
                    Vector3(posAttr->getX(a), posAttr->getY(a), posAttr->getZ(a)),
                    Vector3(posAttr->getX(b), posAttr->getY(b), posAttr->getZ(b)),
                    Vector3(posAttr->getX(c), posAttr->getY(c), posAttr->getZ(c)));
        }
    } else {
        const int count = posAttr->count() - posAttr->count() % 3;
        triangles.reserve(count / 3);
        for (int i = 0; i < count; i += 3) {
            triangles.emplace_back(
                    Vector3(posAttr->getX(i), posAttr->getY(i), posAttr->getZ(i)),
                    Vector3(posAttr->getX(i + 1), posAttr->getY(i + 1), posAttr->getZ(i + 1)),
                    Vector3(posAttr->getX(i + 2), posAttr->getY(i + 2), posAttr->getZ(i + 2)));
        }
    }

    buildTree();
}

void BVH::build(const std::vector<float>& positions, const std::vector<unsigned int>& indices) {
    triangles.clear();
    // The triangles are copied; there is no geometry to point at.
    geometry = nullptr;
    buildTree();

    const auto vertexCount = positions.size() / 3;
    const auto vertex = [&positions](std::size_t i) {
        return Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    };

    if (!indices.empty()) {
        if (indices.size() % 3 != 0) {
            throw std::invalid_argument("BVH::build: index count must be a multiple of 3");
        }
        triangles.reserve(indices.size() / 3);
        for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
            const auto a = indices[i], b = indices[i + 1], c = indices[i + 2];
            if (a >= vertexCount || b >= vertexCount || c >= vertexCount) {
                throw std::out_of_range("BVH::build: index out of range for the given positions");
            }
            triangles.emplace_back(vertex(a), vertex(b), vertex(c));
        }
    } else {
        if (positions.size() % 9 != 0) {
            throw std::invalid_argument("BVH::build: a non-indexed soup needs 9 floats (3 vertices) per triangle");
        }
        triangles.reserve(vertexCount / 3);
        for (std::size_t i = 0; i + 2 < vertexCount; i += 3) {
            triangles.emplace_back(vertex(i), vertex(i + 1), vertex(i + 2));
        }
    }

    buildTree();
}

std::vector<BVH::IntersectionResult> BVH::intersect(const BVH& b1, const Matrix4& m1, const BVH& b2, const Matrix4& m2, bool accurate) {
    std::vector<IntersectionResult> results;
    if (b1.nodes.empty() || b2.nodes.empty()) return results;

    const Matrix4* t1 = nonIdentity(m1);
    const Matrix4* t2 = nonIdentity(m2);

    LeafCache cacheA, cacheB;
    Vector3 center;

    const auto visit = [&](auto&& self, int a, int b) -> void {
        const Node& nodeA = b1.nodes[a];
        const Node& nodeB = b2.nodes[b];
        Box3 bb1, bb2;
        transformBox(nodeA.box(), t1, bb1);
        transformBox(nodeB.box(), t2, bb2);

        // Quick rejection test using bounding boxes
        if (!bb1.intersectsBox(bb2)) return;

        if (nodeA.isLeaf() && nodeB.isLeaf()) {

            if (accurate) {
                // One result per intersecting triangle pair; `position` lies on both surfaces.
                cacheA.load(b1.triangles, nodeA.first, nodeA.count, t1);
                cacheB.load(b2.triangles, nodeB.first, nodeB.count, t2);

                for (std::size_t i = 0; i < cacheA.size(); ++i) {
                    if (!cacheA.boxes[i].intersectsBox(bb2)) continue;
                    for (std::size_t j = 0; j < cacheB.size(); ++j) {
                        if (!cacheA.boxes[i].intersectsBox(cacheB.boxes[j])) continue;
                        if (detail::triTriIntersectionPoint(cacheA.triangles[i], cacheB.triangles[j], center)) {
                            results.emplace_back(IntersectionResult{
                                    b1.sourceIndex[nodeA.first + i], b2.sourceIndex[nodeB.first + j], center});
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

        const bool descendA = !nodeA.isLeaf() && (volumeA >= volumeB || nodeB.isLeaf());
        if (descendA) {
            self(self, nodeA.first, b);
            self(self, nodeA.first + 1, b);
        } else {
            self(self, a, nodeB.first);
            self(self, a, nodeB.first + 1);
        }
    };

    visit(visit, 0, 0);
    return results;
}

std::vector<int> BVH::intersect(const Box3& box, const Matrix4& m) const {
    std::vector<int> results;
    if (nodes.empty()) return results;

    const Matrix4* t = nonIdentity(m);

    int stack[stackSize];
    int sp = 0;
    stack[sp++] = 0;
    Box3 nodeBox, triBox;
    while (sp > 0) {
        const Node& node = nodes[stack[--sp]];
        transformBox(node.box(), t, nodeBox);
        if (!nodeBox.intersectsBox(box)) continue;

        if (!node.isLeaf()) {
            stack[sp++] = node.first + 1;
            stack[sp++] = node.first;
            continue;
        }

        for (int i = node.first; i < node.first + node.count; ++i) {
            transformBox(triangleBox(triangles[i]), t, triBox);
            if (box.intersectsBox(triBox)) {
                results.emplace_back(sourceIndex[i]);
            }
        }
    }
    return results;
}

std::vector<int> BVH::intersect(const Sphere& sphere, const Matrix4& m) const {
    std::vector<int> results;
    if (nodes.empty()) return results;

    const Matrix4* t = nonIdentity(m);
    const float radiusSq = sphere.radius * sphere.radius;

    int stack[stackSize];
    int sp = 0;
    stack[sp++] = 0;
    Box3 nodeBox;
    Vector3 closestPoint;
    Triangle worldTri;
    while (sp > 0) {
        const Node& node = nodes[stack[--sp]];
        transformBox(node.box(), t, nodeBox);
        if (!nodeBox.intersectsSphere(sphere)) continue;

        if (!node.isLeaf()) {
            stack[sp++] = node.first + 1;
            stack[sp++] = node.first;
            continue;
        }

        for (int i = node.first; i < node.first + node.count; ++i) {
            transformTriangle(triangles[i], t, worldTri);
            worldTri.closestPointToPoint(sphere.center, closestPoint);
            if (closestPoint.distanceToSquared(sphere.center) <= radiusSq) {
                results.push_back(sourceIndex[i]);
            }
        }
    }
    return results;
}

bool BVH::intersects(const BVH& b1, const BVH& b2, const Matrix4& m1, const Matrix4& m2) {
    if (b1.nodes.empty() || b2.nodes.empty()) return false;

    const Matrix4* t1 = nonIdentity(m1);
    const Matrix4* t2 = nonIdentity(m2);

    Box3 rootA, rootB;
    transformBox(b1.nodes[0].box(), t1, rootA);
    transformBox(b2.nodes[0].box(), t2, rootB);
    if (!rootA.intersectsBox(rootB)) return false;

    LeafCache cacheA, cacheB;

    // Generic recursive lambda rather than std::function: called once per
    // unseparated node pair, where the indirect call is a measurable share of the query.
    const auto testNodes = [&](auto&& self, int a, int b) -> bool {
        const Node& nodeA = b1.nodes[a];
        const Node& nodeB = b2.nodes[b];
        Box3 boxA, boxB;
        transformBox(nodeA.box(), t1, boxA);
        transformBox(nodeB.box(), t2, boxB);

        if (!boxA.intersectsBox(boxB)) return false;

        if (nodeA.isLeaf() && nodeB.isLeaf()) {

            // Exact triangle test at the leaves; a box-only test reports near misses as hits.
            cacheA.load(b1.triangles, nodeA.first, nodeA.count, t1);
            cacheB.load(b2.triangles, nodeB.first, nodeB.count, t2);

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

        const bool descendA = !nodeA.isLeaf() && (volumeA >= volumeB || nodeB.isLeaf());
        if (descendA) {
            return self(self, nodeA.first, b) ||
                   self(self, nodeA.first + 1, b);
        }
        return self(self, a, nodeB.first) ||
               self(self, a, nodeB.first + 1);
    };

    return testNodes(testNodes, 0, 0);
}

namespace {

    // Walks the tree near child first, calling leafTest(first, count, tLimit) for
    // each leaf whose box the ray meets before tLimit. leafTest returns the new
    // tLimit (smaller after a closer hit), or a negative value to stop the walk.
    template<class Nodes, class LeafTest>
    void traverse(const Nodes& nodes, const RayData& ray, float tLimit, LeafTest&& leafTest) {
        struct Entry {
            int node;
            float t;
        };
        Entry stack[stackSize];
        int sp = 0;

        float tEntry;
        if (!ray.slab(nodes[0].min, nodes[0].max, tLimit, tEntry)) return;
        int current = 0;

        while (true) {
            const auto& node = nodes[current];
            if (node.isLeaf()) {
                tLimit = leafTest(node.first, node.count, tLimit);
                if (tLimit < 0.f) return;
            } else {
                const int left = node.first;
                float tLeft, tRight;
                const bool hitLeft = ray.slab(nodes[left].min, nodes[left].max, tLimit, tLeft);
                const bool hitRight = ray.slab(nodes[left + 1].min, nodes[left + 1].max, tLimit, tRight);
                if (hitLeft && hitRight) {
                    if (tRight < tLeft) {
                        stack[sp++] = {left, tLeft};
                        current = left + 1;
                    } else {
                        stack[sp++] = {left + 1, tRight};
                        current = left;
                    }
                    continue;
                }
                if (hitLeft || hitRight) {
                    current = hitLeft ? left : left + 1;
                    continue;
                }
            }

            // Pop the nearest deferred node that the current best has not ruled out.
            do {
                if (sp == 0) return;
                --sp;
            } while (stack[sp].t > tLimit);
            current = stack[sp].node;
        }
    }

    // Moller-Trumbore, both faces; the t along the (unnormalized) direction, or
    // a negative value on a miss.
    template<class RayTriangle>
    float hitTriangle(const RayTriangle& tri, const RayData& ray) {
        const float* d = ray.d;
        const float* e1 = tri.e1;
        const float* e2 = tri.e2;
        const float p[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
        const float det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if (det == 0.f) return -1.f;
        const float invDet = 1.f / det;
        const float s[3] = {ray.o[0] - tri.v0[0], ray.o[1] - tri.v0[1], ray.o[2] - tri.v0[2]};
        const float u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * invDet;
        if (u < 0.f || u > 1.f) return -1.f;
        const float q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
        const float v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * invDet;
        if (v < 0.f || u + v > 1.f) return -1.f;
        return (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * invDet;
    }

}// namespace

std::optional<BVH::RayHit> BVH::raycast(const Ray& ray, float maxDistance) const {
    if (nodes.empty()) return std::nullopt;

    const float length = ray.direction.length();
    if (!(length > 0.f)) return std::nullopt;

    // Distances along the ray are t * length; the walk compares t.
    const RayData data(ray);
    const float tMin = rayEps / length;
    int best = -1;
    float bestT = 0.f;

    traverse(nodes, data, (maxDistance - rayEps) / length, [&](int first, int count, float tLimit) {
        for (int i = first; i < first + count; ++i) {
            const float t = hitTriangle(rayTriangles[i], data);
            // Written so a NaN t (from edges that overflow) is never a hit.
            if (!(t >= tMin && t <= tLimit)) continue;
            tLimit = t;
            bestT = t;
            best = i;
        }
        return tLimit;
    });

    if (best < 0) return std::nullopt;

    // Normal for the winning triangle only, flipped to face the ray origin.
    Vector3 point, edge1, edge2, normal;
    ray.at(bestT, point);
    const Triangle& tri = triangles[best];
    edge1.subVectors(tri.b(), tri.a());
    edge2.subVectors(tri.c(), tri.a());
    normal.crossVectors(edge1, edge2).normalize();
    if (normal.dot(ray.direction) > 0) normal.negate();

    return RayHit{point.distanceTo(ray.origin), sourceIndex[best], point, normal};
}

bool BVH::raycastAny(const Ray& ray, float maxDistance) const {
    if (nodes.empty()) return false;

    const float length = ray.direction.length();
    if (!(length > 0.f)) return false;

    const RayData data(ray);
    const float tMin = rayEps / length;
    bool hit = false;

    traverse(nodes, data, (maxDistance - rayEps) / length, [&](int first, int count, float tLimit) {
        for (int i = first; i < first + count; ++i) {
            const float t = hitTriangle(rayTriangles[i], data);
            // Written so a NaN t (from edges that overflow) is never a hit.
            if (!(t >= tMin && t <= tLimit)) continue;
            hit = true;
            return -1.f;
        }
        return tLimit;
    });

    return hit;
}

void BVH::collectBoxes(std::vector<BVHBox3>& boxes) const {
    if (!nodes.empty()) collectBoxes(0, boxes);
}

float BVH::distance(const BVH& b1, const BVH& b2, const Matrix4& m1, const Matrix4& m2, float maxDistance) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    if (b1.nodes.empty() || b2.nodes.empty()) return inf;
    if (maxDistance <= 0.f) return inf;

    const Matrix4* t1 = nonIdentity(m1);
    const Matrix4* t2 = nonIdentity(m2);

    const float limitSq = maxDistance == inf ? inf : maxDistance * maxDistance;
    float bestSq = limitSq;
    LeafCache cacheA, cacheB;

    // Depth-first with a running best: node pairs whose boxes are farther apart
    // than bestSq are pruned, and the nearer child is visited first so bestSq drops early.
    const auto visit = [&](auto&& self, int a, const Box3& boxA, int b, const Box3& boxB) -> void {
        if (bestSq <= 0.f) return;
        if (detail::boxDistanceSq(boxA, boxB) >= bestSq) return;

        const Node& nodeA = b1.nodes[a];
        const Node& nodeB = b2.nodes[b];

        if (nodeA.isLeaf() && nodeB.isLeaf()) {
            cacheA.load(b1.triangles, nodeA.first, nodeA.count, t1);
            cacheB.load(b2.triangles, nodeB.first, nodeB.count, t2);

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
        const bool splitA = !nodeA.isLeaf() && (volumeA >= volumeB || nodeB.isLeaf());

        Box3 firstBox, secondBox;
        if (splitA) {
            int first = nodeA.first, second = nodeA.first + 1;
            transformBox(b1.nodes[first].box(), t1, firstBox);
            transformBox(b1.nodes[second].box(), t1, secondBox);
            if (detail::boxDistanceSq(secondBox, boxB) < detail::boxDistanceSq(firstBox, boxB)) {
                std::swap(first, second);
                std::swap(firstBox, secondBox);
            }
            self(self, first, firstBox, b, boxB);
            self(self, second, secondBox, b, boxB);
        } else {
            int first = nodeB.first, second = nodeB.first + 1;
            transformBox(b2.nodes[first].box(), t2, firstBox);
            transformBox(b2.nodes[second].box(), t2, secondBox);
            if (detail::boxDistanceSq(boxA, secondBox) < detail::boxDistanceSq(boxA, firstBox)) {
                std::swap(first, second);
                std::swap(firstBox, secondBox);
            }
            self(self, a, boxA, first, firstBox);
            self(self, a, boxA, second, secondBox);
        }
    };

    Box3 rootA, rootB;
    transformBox(b1.nodes[0].box(), t1, rootA);
    transformBox(b2.nodes[0].box(), t2, rootB);
    visit(visit, 0, rootA, 0, rootB);

    // Nothing closer than the cutoff.
    return bestSq >= limitSq ? inf : std::sqrt(bestSq);
}

void BVH::collectBoxes(int node, std::vector<BVHBox3>& boxes) const {
    const Node& n = nodes[node];
    boxes.emplace_back(n.box(), n.isLeaf());
    if (n.isLeaf()) return;
    collectBoxes(n.first, boxes);
    collectBoxes(n.first + 1, boxes);
}

const BufferGeometry* BVH::getGeometry() const {
    return geometry;
}
