// https://github.com/mrdoob/three.js/tree/r129/src/extras/core

#ifndef THREEPP_CURVE_HPP
#define THREEPP_CURVE_HPP

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

#include <cmath>
#include <numeric>
#include <optional>
#include <vector>

namespace threepp {

    /**
     * Extensible curve object.
     *
     * Some common of curve methods:
     * .getPoint( t, optionalTarget ), .getTangent( t, optionalTarget )
     * .getPointAt( u, optionalTarget ), .getTangentAt( u, optionalTarget )
     * .getPoints(), .getSpacedPoints()
     * .getLength()
     * .updateArcLengths()
     *
     * This following curves inherit from THREE.Curve:
     *
     * -- 2D curves --
     * THREE.ArcCurve
     * THREE.CubicBezierCurve
     * THREE.EllipseCurve
     * THREE.LineCurve
     * THREE.QuadraticBezierCurve
     * THREE.SplineCurve
     *
     * -- 3D curves --
     * THREE.CatmullRomCurve3
     * THREE.CubicBezierCurve3
     * THREE.LineCurve3
     * THREE.QuadraticBezierCurve3
     *
     * A series of curves can be represented as a THREE.CurvePath.
     *
     **/
    template<class T>
    class Curve {

    public:
        bool needsUpdate{false};
        int arcLengthDivisions{200};

        struct FrenetFrames;

        // Virtual base class method to overwrite and implement in subclasses
        //	- t [0 .. 1]
        virtual void getPoint(float t, T &target) = 0;

        void getPointAt(float u, T &target) {

            const auto t = this->getUtoTmapping(u);
            return this->getPoint(t, target);
        }

        // Get sequence of points using getPoint( t )

        std::vector<T> getPoints(int divisions = 5) {

            std::vector<T> points;

            for (int d = 0; d <= divisions; d++) {
                T &point = points.emplace_back();
                this->getPoint(static_cast<float>(d) / static_cast<float>(divisions));
            }

            return points;
        }

        // Get sequence of points using getPointAt( u )

        std::vector<T> getSpacedPoints(int divisions = 5) {

            std::vector<T> points;

            for (int d = 0; d <= divisions; d++) {

                T &point = points.emplace_back();
                this->getPointAt(static_cast<float>(d) / static_cast<float>(divisions));
            }

            return points;
        }

        // Get total curve arc length
        float getLength() {

            const auto lengths = this->getLengths();
            return lengths[lengths.size() - 1];
        }

        // Get list of cumulative segment lengths

        std::vector<float> getLengths() {

            return getLengths(this->arcLengthDivisions);
        }

        std::vector<float> getLengths(int divisions) {

            if ((this->cacheArcLengths.size() == divisions + 1) && !this->needsUpdate) {

                return this->cacheArcLengths;
            }

            this->needsUpdate = false;

            std::vector<float> cache;
            T current, last;
            this->getPoint(0, last);
            float sum = 0;

            cache.emplace_back(0.f);

            for (int p = 1; p <= divisions; p++) {

                this->getPoint(static_cast<float>(p) / static_cast<float>(divisions), current);
                sum += current.distanceTo(last);
                cache.emplace_back(sum);
                last = current;
            }

            this->cacheArcLengths = cache;

            return cache;
        }

        void updateArcLengths() {

            this->needsUpdate = true;
            this->getLengths();
        }

        // Given u ( 0 .. 1 ), get a t to find p. This gives you points which are equidistant

        float getUtoTmapping(float u, std::optional<float> distance = std::nullopt) {

            const auto arcLengths = this->getLengths();

            float i;
            float il = static_cast<float>(arcLengths.size());

            float targetArcLength; // The targeted u distance value to get

            if (distance) {

                targetArcLength = *distance;

            } else {

                targetArcLength = u * arcLengths[static_cast<int>(il) - 1];
            }

            // binary search for the index with largest value smaller than target u distance

            float low = 0, high = il - 1, comparison;

            while (low <= high) {

                i = std::floor(low + (high - low) / 2);

                comparison = arcLengths[static_cast<int>(i)] - targetArcLength;

                if (comparison < 0) {

                    low = i + 1;

                } else if (comparison > 0) {

                    high = i - 1;

                } else {

                    high = i;
                    break;

                    // DONE
                }
            }

            i = high;

            if (arcLengths[static_cast<int>(i)] == targetArcLength) {

                return i / (il - 1);
            }

            // we could get finer grain at lengths, or use simple interpolation between two points

            const float lengthBefore = arcLengths[static_cast<int>(i)];
            const float lengthAfter = arcLengths[static_cast<int>(i) + 1];

            const float segmentLength = lengthAfter - lengthBefore;

            // determine where we are between the 'before' and 'after' points

            const float segmentFraction = (targetArcLength - lengthBefore) / segmentLength;

            // add that fractional amount to t

            const float t = (i + segmentFraction) / (il - 1);

            return t;
        }

        // Returns a unit vector tangent at t
        // In case any sub curve does not implement its tangent derivation,
        // 2 points a small delta apart will be used to find its gradient
        // which seems to give a reasonable approximation

        void getTangent(float t, T &tangent) {

            const float delta = 0.0001f;
            float t1 = t - delta;
            float t2 = t + delta;

            // Capping in case of danger

            if (t1 < 0) t1 = 0;
            if (t2 > 1) t2 = 1;

            T pt1, pt2;
            this->getPoint(t1, pt1);
            this->getPoint(t2, pt2);

            tangent.copy(pt2).sub(pt1).normalize();
        }

        void getTangentAt(float u, T &optionalTarget) {

            const float t = this->getUtoTmapping(u);
            this->getTangent(t, optionalTarget);
        }

        FrenetFrames computeFrenetFrames(int segments, bool closed) {

            // see http://www.cs.indiana.edu/pub/techreports/TR425.pdf

            Vector3 normal;

            std::vector<T> tangents;
            std::vector<T> normals;
            std::vector<T> binormals;

            Vector3 vec;
            Matrix4 mat;

            // compute the tangent vectors for each segment on the curve

            for (int i = 0; i <= segments; i++) {

                const float u = static_cast<float>(i) / static_cast<float>(segments);

                auto &tangent = tangents.emplace_back();
                this->getTangentAt(u, tangent);
                tangent.normalize();
            }

            // select an initial normal vector perpendicular to the first tangent vector,
            // and in the direction of the minimum tangent xyz component

            normals.emplace_back();
            binormals.emplace_back();
            float min = std::numeric_limits<float>::max();
            const float tx = std::abs(tangents[0].x);
            const float ty = std::abs(tangents[0].y);
            const float tz = std::abs(tangents[0].z);

            if (tx <= min) {

                min = tx;
                normal.set(1, 0, 0);
            }

            if (ty <= min) {

                min = ty;
                normal.set(0, 1, 0);
            }

            if (tz <= min) {

                normal.set(0, 0, 1);
            }

            vec.crossVectors(tangents[0], normal).normalize();

            normals[0].crossVectors(tangents[0], vec);
            binormals[0].crossVectors(tangents[0], normals[0]);


            // compute the slowly-varying normal and binormal vectors for each segment on the curve

            for (int i = 1; i <= segments; i++) {

                normals.emplace_back(normals[i - 1].clone());

                binormals.emplace_back( binormals[i - 1].clone());

                vec.crossVectors(tangents[i - 1], tangents[i]);

                if (vec.length() > std::numeric_limits<float>::epsilon()) {

                    vec.normalize();

                    const float theta = std::acos(std::clamp(tangents[i - 1].dot(tangents[i]), -1.f, 1.f));// clamp for floating pt errors

                    normals[i].applyMatrix4(mat.makeRotationAxis(vec, theta));
                }

                binormals[i].crossVectors(tangents[i], normals[i]);
            }

            // if the curve is closed, postprocess the vectors so the first and last normal vectors are the same

            if (closed) {

                float theta = std::acos(std::clamp(normals[0].dot(normals[segments]), -1.f, 1.f));
                theta /= segments;

                if (tangents[0].dot(vec.crossVectors(normals[0], normals[segments])) > 0) {

                    theta = -theta;
                }

                for (int i = 1; i <= segments; i++) {

                    // twist a little...
                    normals[i].applyMatrix4(mat.makeRotationAxis(tangents[i], theta * i));
                    binormals[i].crossVectors(tangents[i], normals[i]);
                }
            }

            return {
                    tangents,
                    normals,
                    binormals};
        }

        virtual ~Curve() = default;

        struct FrenetFrames {

            std::vector<T> tangents;
            std::vector<T> normals;
            std::vector<T> binormals;
        };

    private:
        std::vector<float> cacheArcLengths;
    };

    typedef Curve<Vector2> Curve2;
    typedef Curve<Vector3> Curve3;

}// namespace threepp

#endif//THREEPP_CURVE_HPP
