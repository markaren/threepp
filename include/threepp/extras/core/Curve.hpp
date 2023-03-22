// https://github.com/mrdoob/three.js/tree/r129/src/extras/core

#ifndef THREEPP_CURVE_HPP
#define THREEPP_CURVE_HPP

#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

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

        // Virtual base class method to overwrite and implement in subclasses
        //	- t [0 .. 1]
        virtual void getPoint(float t, T& target) = 0;

        virtual void getPointAt(float u, T& target);

        // Get sequence of points using getPoint( t )

        virtual std::vector<T> getPoints(unsigned int divisions = 5);

        // Get sequence of points using getPointAt( u )

        virtual std::vector<T> getSpacedPoints(unsigned int divisions = 5);

        // Get total curve arc length
        virtual float getLength();

        // Get list of cumulative segment lengths

        std::vector<float> getLengths();

        std::vector<float> getLengths(int divisions);

        virtual void updateArcLengths();

        // Given u ( 0 .. 1 ), get a t to find p. This gives you points which are equidistant

        float getUtoTmapping(float u, std::optional<float> distance = std::nullopt);

        // Returns a unit vector tangent at t
        // In case any sub curve does not implement its tangent derivation,
        // 2 points a small delta apart will be used to find its gradient
        // which seems to give a reasonable approximation

        virtual void getTangent(float t, T& tangent);

        void getTangentAt(float u, T& optionalTarget);

        virtual ~Curve() = default;


    protected:
        std::vector<float> cacheArcLengths;
    };


    typedef Curve<Vector2> Curve2;

    class Curve3: public Curve<Vector3> {

    public:
        struct FrenetFrames {

            std::vector<Vector3> tangents;
            std::vector<Vector3> normals;
            std::vector<Vector3> binormals;
        };

        FrenetFrames computeFrenetFrames(unsigned int segments, bool closed);
    };

    extern template class threepp::Curve<Vector2>;
    extern template class threepp::Curve<Vector3>;

}// namespace threepp

#endif//THREEPP_CURVE_HPP
