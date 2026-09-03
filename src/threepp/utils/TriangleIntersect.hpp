// Exact triangle-triangle predicates for the BVH leaf tests.
// Private to the library (src/, not include/); tests reach it through the src/
// include path set in tests/CMakeLists.txt.
#ifndef THREEPP_TRIANGLEINTERSECT_HPP
#define THREEPP_TRIANGLEINTERSECT_HPP

#include "threepp/math/Box3.hpp"
#include "threepp/math/Triangle.hpp"
#include "threepp/math/Vector3.hpp"

namespace threepp::detail {

    // Squared gap between two axis-aligned boxes; 0 when they overlap or touch.
    // The pruning bound for the distance query: no pair of triangles inside two
    // boxes can be closer than this.
    [[nodiscard]] float boxDistanceSq(const Box3& a, const Box3& b);

    // True when the two triangles share at least one point. Coplanar pairs are
    // resolved by a 2-D test in the plane they share. Degenerate (zero-area)
    // triangles never intersect.
    [[nodiscard]] bool triTriOverlap(const Triangle& a, const Triangle& b);

    // As triTriOverlap, and on a hit writes a representative point of the
    // intersection to `target`: the midpoint of the intersection segment for a
    // crossing pair, or the average of the two centroids when the pair is
    // coplanar (where the intersection is an area, not a segment).
    [[nodiscard]] bool triTriIntersectionPoint(const Triangle& a, const Triangle& b, Vector3& target);

    // Squared distance between the two triangle surfaces; 0 when they overlap.
    [[nodiscard]] float triTriDistanceSq(const Triangle& a, const Triangle& b);

}// namespace threepp::detail

#endif//THREEPP_TRIANGLEINTERSECT_HPP
