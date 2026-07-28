
#include "threepp/extras/curves/RoadGeometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

using namespace threepp;

namespace {

    // Below this the cross product of up and tangent says nothing about which
    // way is sideways — the curve is running straight up or straight down.
    constexpr float kDegenerate = 1e-4f;

    // Below this |cross(up, tangent)| the horizontal part of the tangent is
    // noise, not signal — sin of about six degrees. See RibbonGeometry, where
    // renormalizing that noise fanned the ribbon into inside-out quads.
    constexpr float kLevelable = 0.1f;

    // Ceiling on the miter widening. The offset of a polyline IS its mitered
    // one, so this clamp is not a treatment for anything — it is only there so
    // a near-cusp cannot produce infinity. It sits far higher than
    // RibbonGeometry's, which had to keep a corner from spiking because it had
    // no trim to cut the spike off with.
    constexpr float kMaxMiter = 8.f;

    // One vertex of an edge polyline: where it is, and how far along the road
    // it is. `u` is what the two edges are stitched against — the only
    // correspondence left once a trim has taken vertices off one of them.
    struct Edge {
        Vector3 point;
        float u = 0.f;
    };

    // Do segments a0-a1 and b0-b1 cross in the XZ plane, away from their own
    // endpoints? The loop an offset ties is a plan phenomenon; a crossing AT an
    // endpoint is two segments sharing a vertex, which every polyline does.
    bool crossesXZ(const Vector3& a0, const Vector3& a1, const Vector3& b0, const Vector3& b1,
                   float& alongA, float& alongB) {

        const float ax = a1.x - a0.x, az = a1.z - a0.z;
        const float bx = b1.x - b0.x, bz = b1.z - b0.z;
        const float denominator = ax * bz - az * bx;
        if (std::abs(denominator) < 1e-12f) return false;// parallel

        const float dx = b0.x - a0.x, dz = b0.z - a0.z;
        alongA = (dx * bz - dz * bx) / denominator;
        alongB = (dx * az - dz * ax) / denominator;
        constexpr float edge = 1e-4f;
        return alongA > edge && alongA < 1.f - edge && alongB > edge && alongB < 1.f - edge;
    }

    // Cut the swallowtail loops out of an offset edge.
    //
    // The chain is rebuilt one vertex at a time, and each new segment is tested
    // against the ones already accepted, OLDEST first: where it crosses one,
    // everything between them is loop and comes off, and the crossing point
    // becomes the corner they meet at. Oldest first because the largest loop is
    // the real one — the segments of a swallowtail cross each other many times
    // over, and cutting at the nearest crossing would leave the rest of the
    // tail behind.
    //
    // The search is windowed by ARC LENGTH, not by sample count: a loop is tied
    // inside one bend and its two crossing segments are a few road widths apart
    // along the centreline, however finely that stretch was sampled. A road
    // that crosses itself half a lap later is a different question and out of
    // scope (see the header) — the window is what keeps the two apart. The
    // index cap only bounds the cost of a very densely sampled road.
    std::vector<Edge> trimLoops(const std::vector<Edge>& chain, float window) {

        if (chain.size() < 4) return chain;
        constexpr std::size_t kMaxLookback = 2048;

        std::vector<Edge> out;
        out.reserve(chain.size());
        out.push_back(chain.front());

        for (std::size_t k = 1; k < chain.size(); ++k) {

            const Edge& ahead = chain[k];
            const std::size_t accepted = out.size();
            std::size_t cut = accepted;// none
            float alongA = 0.f, alongB = 0.f;

            if (accepted >= 3) {
                const std::size_t last = accepted - 1;
                std::size_t oldest = last;
                while (oldest > 0 && out[last].u - out[oldest - 1].u <= window &&
                       last - (oldest - 1) <= kMaxLookback) {
                    --oldest;
                }
                for (std::size_t q = oldest; q + 1 < last; ++q) {
                    float s = 0.f, t = 0.f;
                    if (crossesXZ(out[q].point, out[q + 1].point, out[last].point, ahead.point, s, t)) {
                        cut = q;
                        alongA = s;
                        alongB = t;
                        break;
                    }
                }
            }

            if (cut < accepted) {
                const Edge tail = out.back();
                Edge corner;
                corner.point.copy(out[cut].point).lerp(out[cut + 1].point, alongA);
                // Both crossing segments carry a parameter for the corner, and
                // they straddle the loop; the surface either side of it is
                // stitched against the one in between.
                const float uA = out[cut].u + (out[cut + 1].u - out[cut].u) * alongA;
                const float uB = tail.u + (ahead.u - tail.u) * alongB;
                const float yB = tail.point.y + (ahead.point.y - tail.point.y) * alongB;
                corner.u = 0.5f * (uA + uB);
                corner.point.y = 0.5f * (corner.point.y + yB);
                out.resize(cut + 1);
                out.push_back(corner);
            }

            // Coincident neighbours would only make triangles with no area, and
            // leave the next crossing test a segment with no direction.
            if (out.back().point.distanceTo(ahead.point) > 1e-6f) out.push_back(ahead);
        }

        return out;
    }

    // Key for welding vertices that are bit-identical — a closed road's seam
    // duplicate, and nothing else. Exact, because those come out of the same
    // arithmetic rather than merely near each other.
    struct PositionKey {
        std::array<std::uint32_t, 3> bits{};

        bool operator==(const PositionKey& other) const { return bits == other.bits; }
    };

    struct PositionHash {
        std::size_t operator()(const PositionKey& key) const {
            std::size_t seed = 0;
            for (const auto value : key.bits) {
                seed ^= static_cast<std::size_t>(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    PositionKey keyOf(float x, float y, float z) {
        PositionKey key;
        // Negative zero and zero are the same place.
        const float values[3]{x == 0.f ? 0.f : x, y == 0.f ? 0.f : y, z == 0.f ? 0.f : z};
        std::memcpy(key.bits.data(), values, sizeof(values));
        return key;
    }

}// namespace


RoadGeometry::RoadGeometry(const Curve3& path, const Params& params)
    : width(params.width) {

    const unsigned int spans = std::max(params.divisions, 1u);
    const unsigned int rings = spans + 1;
    const float half = params.width * 0.5f;
    const float tile = std::max(params.uvLength, 1e-4f);

    std::vector<Vector3> centers(rings);
    for (unsigned int i = 0; i < rings; ++i) {
        path.getPoint(static_cast<float>(i) / static_cast<float>(spans), centers[i]);
    }

    const Vector3 up{0, 1, 0};

    // Per-span direction and the level sideways vector that goes with it,
    // carried across spans so a stretch running vertically keeps the sideways
    // direction it had going in. The same pass as RibbonGeometry's, and for the
    // same reasons; what differs is everything downstream of it.
    std::vector<Vector3> sides(spans);
    {
        Vector3 dir{0, 0, 1};
        Vector3 side{1, 0, 0};
        bool sideLocked = false;
        Vector3 delta;
        Vector3 candidate;
        for (unsigned int i = 0; i < spans; ++i) {
            delta.copy(centers[i + 1]).sub(centers[i]);
            if (delta.length() > kDegenerate) dir.copy(delta).normalize();
            candidate.copy(up).cross(dir);
            if (candidate.length() > kLevelable) {
                candidate.normalize();
                // Continuity outranks azimuth: a side that swaps to the other
                // side between two rings builds everything between them inside
                // out.
                if (sideLocked && candidate.dot(side) < 0.f) candidate.negate();
                side.copy(candidate);
                sideLocked = true;
            }
            sides[i].copy(side);
        }
    }

    // The two offset polylines. A cross-section belongs to the two spans
    // meeting at it: it bisects them and is widened by 1/cos(theta/2), which is
    // exactly where the neighbouring spans' offset edges cross — the miter join
    // IS the offset of a polyline, so this is the edge itself rather than an
    // approximation of it. Where that edge crosses itself, the trim below cuts.
    std::vector<Edge> left(rings);
    std::vector<Edge> right(rings);
    {
        Vector3 bisector;
        Vector3 side;
        float arc = 0.f;
        for (unsigned int i = 0; i < rings; ++i) {

            const Vector3& center = centers[i];
            if (i > 0) arc += center.distanceTo(centers[i - 1]);

            // Endpoints are one-sided; a closed curve wraps, which also makes
            // the seam ring's frame bit-identical to ring 0's rather than
            // merely close to it.
            const unsigned int inSpan = (i > 0) ? i - 1 : (params.closed ? spans - 1 : 0u);
            const unsigned int outSpan = (i < spans) ? i : (params.closed ? 0u : spans - 1);

            bisector.copy(sides[inSpan]).add(sides[outSpan]);
            const float cosHalf = bisector.length() * 0.5f;
            float miter = 1.f;
            if (cosHalf > kDegenerate) {
                side.copy(bisector).normalize();
                miter = std::min(1.f / cosHalf, kMaxMiter);
            } else {
                // The spans reverse: there is no bisector.
                side.copy(sides[outSpan]);
            }

            const float offset = half * miter;
            left[i].point.copy(center).addScaledVector(side, -offset);
            right[i].point.copy(center).addScaledVector(side, offset);
            left[i].u = right[i].u = arc / tile;
        }
    }

    // A swallowtail is tied within a few road widths of centreline; anything
    // that crosses from farther away is the road lying across itself, which is
    // not this pass's business. u carries arc length divided by the tile, so
    // the window converts through the same divisor.
    const float window = params.width * 4.f / tile;
    const std::vector<Edge> leftEdge = trimLoops(left, window);
    const std::vector<Edge> rightEdge = trimLoops(right, window);
    if (leftEdge.size() < 2 || rightEdge.size() < 2) return;

    std::vector<float> positions;
    std::vector<float> uvs;
    positions.reserve((leftEdge.size() + rightEdge.size()) * 3);
    uvs.reserve((leftEdge.size() + rightEdge.size()) * 2);
    for (const auto& vertex : leftEdge) {
        positions.insert(positions.end(), {vertex.point.x, vertex.point.y, vertex.point.z});
        uvs.insert(uvs.end(), {vertex.u, 0.f});
    }
    for (const auto& vertex : rightEdge) {
        positions.insert(positions.end(), {vertex.point.x, vertex.point.y, vertex.point.z});
        uvs.insert(uvs.end(), {vertex.u, 1.f});
    }

    // Stitch the strip between the two edges: at every step advance whichever
    // edge is BEHIND in arc length and close the triangle across. Where a trim
    // has taken a run of vertices off one edge, that edge stands still while
    // the other walks past it — a fan out of the corner vertex, which is the
    // shape the swept region has there.
    std::vector<unsigned int> indices;
    indices.reserve((leftEdge.size() + rightEdge.size()) * 3);
    {
        const auto rightBase = static_cast<unsigned int>(leftEdge.size());
        std::size_t i = 0, j = 0;
        while (i + 1 < leftEdge.size() || j + 1 < rightEdge.size()) {
            const bool advanceLeft = j + 1 >= rightEdge.size() ||
                                     (i + 1 < leftEdge.size() && leftEdge[i + 1].u <= rightEdge[j + 1].u);
            const auto l0 = static_cast<unsigned int>(i);
            const auto r0 = rightBase + static_cast<unsigned int>(j);
            // Wound so the face normal points up: reversing this leaves the
            // road back-face culled, i.e. invisible from above.
            if (advanceLeft) {
                indices.insert(indices.end(), {l0, static_cast<unsigned int>(i + 1), r0});
                ++i;
            } else {
                indices.insert(indices.end(), {l0, rightBase + static_cast<unsigned int>(j + 1), r0});
                ++j;
            }
        }
    }

    this->setIndex(indices);
    this->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    this->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
    // Read off the triangles rather than off the frames: a trim corner belongs
    // to faces from both sides of the cut, and averaging them is what keeps the
    // shading continuous across it.
    this->computeVertexNormals();
    this->computeBoundingBox();
    this->computeBoundingSphere();
}

std::shared_ptr<BufferGeometry> RoadGeometry::solid(const BufferGeometry& surface, float thickness) {

    auto out = BufferGeometry::create();

    const auto* position = surface.getAttribute<float>("position");
    const auto* index = surface.getIndex();
    if (!position || !index || index->array().size() < 3) return out;

    // Weld by position: the surface's vertices are unique except a closed
    // road's seam, and an unwelded seam reads as two boundaries and grows a
    // wall straight across the road.
    std::unordered_map<PositionKey, unsigned int, PositionHash> welded;
    std::vector<float> points;
    std::vector<unsigned int> top;
    points.reserve(static_cast<std::size_t>(position->count()) * 3);
    top.reserve(index->array().size());

    const auto canonical = [&](unsigned int vertex) {
        const auto i = static_cast<int>(vertex);
        const float x = position->getX(i), y = position->getY(i), z = position->getZ(i);
        const auto [it, inserted] = welded.emplace(keyOf(x, y, z),
                                                   static_cast<unsigned int>(points.size() / 3));
        if (inserted) points.insert(points.end(), {x, y, z});
        return it->second;
    };

    const auto& source = index->array();
    for (std::size_t t = 0; t + 2 < source.size(); t += 3) {
        const unsigned int a = canonical(source[t]);
        const unsigned int b = canonical(source[t + 1]);
        const unsigned int c = canonical(source[t + 2]);
        // A triangle that lost a vertex to the weld has no area, and no two
        // sides for the parity below to count.
        if (a == b || b == c || a == c) continue;
        top.insert(top.end(), {a, b, c});
    }
    if (top.empty()) return out;

    const auto count = static_cast<unsigned int>(points.size() / 3);
    std::vector<float> positions(points);
    positions.reserve(points.size() * 2);
    for (unsigned int i = 0; i < count; ++i) {
        positions.insert(positions.end(),
                         {points[i * 3], points[i * 3 + 1] - thickness, points[i * 3 + 2]});
    }

    std::vector<unsigned int> indices(top);
    indices.reserve(top.size() * 3);
    // The underside, reversed so it faces down.
    for (std::size_t t = 0; t + 2 < top.size(); t += 3) {
        indices.insert(indices.end(), {top[t] + count, top[t + 2] + count, top[t + 1] + count});
    }

    // A boundary edge is one whose OPPOSITE never appears: an interior edge is
    // walked once each way, by the two triangles sharing it. Every boundary
    // edge gets a wall down to the underside, wound outward — a triangle's
    // interior lies to the left of its edges, so the wall faces the other way.
    std::unordered_map<std::uint64_t, int> directed;
    directed.reserve(top.size() * 2);
    const auto edgeKey = [](unsigned int a, unsigned int b) {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    };
    for (std::size_t t = 0; t + 2 < top.size(); t += 3) {
        ++directed[edgeKey(top[t], top[t + 1])];
        ++directed[edgeKey(top[t + 1], top[t + 2])];
        ++directed[edgeKey(top[t + 2], top[t])];
    }
    for (std::size_t t = 0; t + 2 < top.size(); t += 3) {
        for (int e = 0; e < 3; ++e) {
            const unsigned int a = top[t + static_cast<std::size_t>(e)];
            const unsigned int b = top[t + static_cast<std::size_t>((e + 1) % 3)];
            if (directed.count(edgeKey(b, a))) continue;
            indices.insert(indices.end(), {a, a + count, b});
            indices.insert(indices.end(), {b, a + count, b + count});
        }
    }

    out->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    out->setIndex(indices);
    out->computeBoundingBox();
    out->computeBoundingSphere();
    return out;
}

std::string RoadGeometry::type() const {

    return "RoadGeometry";
}

std::shared_ptr<RoadGeometry> RoadGeometry::create(const Curve3& path, const Params& params) {

    return std::shared_ptr<RoadGeometry>(new RoadGeometry(path, params));
}

std::shared_ptr<RoadGeometry> RoadGeometry::create(const Curve3& path, float width,
                                                   unsigned int divisions, float uvLength,
                                                   bool closed) {

    return create(path, Params(width, divisions, uvLength, closed));
}

RoadGeometry::Params::Params(float width, unsigned int divisions, float uvLength, bool closed)
    : width(width), divisions(divisions), uvLength(uvLength), closed(closed) {}
