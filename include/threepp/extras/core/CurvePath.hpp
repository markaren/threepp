// https://github.com/mrdoob/three.js/blob/r129/src/extras/core/CurvePath.js

#ifndef THREEPP_CURVEPATH_HPP
#define THREEPP_CURVEPATH_HPP

#include "threepp/extras/core/Curve.hpp"

#include <memory>
#include <vector>

namespace threepp {

    template<class T>
    class CurvePath: public Curve<T> {

    public:
        bool autoClose = false;
        std::vector<std::shared_ptr<Curve<T>>> curves;

        void add(const std::shared_ptr<Curve<T>>& curve);

        void closePath();

        // To get accurate point with reference to
        // entire path distance at time t,
        // following has to be done:

        // 1. Length of each sub path have to be known
        // 2. Locate and identify type of curve
        // 3. Get t for the curve
        // 4. Return curve.getPointAt(t')

        void getPoint(float t, T& target) const override;

        // We cannot use the default THREE.Curve getPoint() with getLength() because in
        // THREE.Curve, getLength() depends on getPoint() but in THREE.CurvePath
        // getPoint() depends on getLength

        float getLength() const override;

        // cacheLengths must be recalculated.
        void updateArcLengths() override;

        // Compute lengths and cache them
        // We cannot overwrite getLengths() because UtoT mapping uses it.

        std::vector<float> getCurveLengths() const;

        std::vector<T> getSpacedPoints(unsigned int divisions = 40) const override;

        std::vector<T> getPoints(unsigned int divisions = 12) const override;

    private:
        mutable std::optional<std::vector<float>> cacheLengths;
    };


}// namespace threepp

#endif//THREEPP_CURVEPATH_HPP
