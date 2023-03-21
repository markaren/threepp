// https://github.com/mrdoob/three.js/blob/r129/src/extras/core/CurvePath.js

#ifndef THREEPP_CURVEPATH_HPP
#define THREEPP_CURVEPATH_HPP

#include "threepp/extras/core/Curve.hpp"
#include "threepp/extras/curves/EllipseCurve.hpp"
#include "threepp/extras/curves/LineCurve.hpp"
#include "threepp/extras/curves/LineCurve3.hpp"
#include "threepp/extras/curves/SplineCurve.hpp"

#include <memory>

namespace threepp {

    template<class T>
    class CurvePath: public Curve<T> {

    public:
        bool autoClose = false;
        std::vector<std::shared_ptr<Curve<T>>> curves;

        void add(const Curve<T>& curve) {

            curves.emplace_back(curve);
        }

        void closePath() {

            T startPoint;
            T endPoint;
            curves.front()->getPoint(0, startPoint);
            curves.back()->getPoint(1, endPoint);

            if (!startPoint.equals(endPoint)) {

                curves.emplace_back(std::make_shared<LineCurve>(endPoint, startPoint));
            }
        }

        // To get accurate point with reference to
        // entire path distance at time t,
        // following has to be done:

        // 1. Length of each sub path have to be known
        // 2. Locate and identify type of curve
        // 3. Get t for the curve
        // 4. Return curve.getPointAt(t')

        void getPoint(float t, T& target) override {

            auto d = t * this->getLength();
            auto curveLengths = this->getCurveLengths();
            int i = 0;

            // To think about boundaries points.

            while (i < curveLengths.size()) {

                if (curveLengths[i] >= d) {

                    auto diff = curveLengths[i] - d;
                    auto& curve = this->curves[i];

                    auto segmentLength = curve->getLength();
                    auto u = segmentLength == 0 ? 0 : 1 - diff / segmentLength;

                    curve->getPointAt(u, target);
                }

                ++i;
            }

            target.makeNan();

            // loop where sum != 0, sum > d , sum+1 <d
        }

        // We cannot use the default THREE.Curve getPoint() with getLength() because in
        // THREE.Curve, getLength() depends on getPoint() but in THREE.CurvePath
        // getPoint() depends on getLength

        float getLength() override {

            auto lens = this->getCurveLengths();
            return lens.back();
        }

        // cacheLengths must be recalculated.
        void updateArcLengths() override {

            this->needsUpdate = true;
            this->cacheLengths = std::nullopt;
            this->getCurveLengths();
        }


        // Compute lengths and cache them
        // We cannot overwrite getLengths() because UtoT mapping uses it.

        std::vector<float> getCurveLengths() {

            // We use cache values if curves and cache array are same length

            if (this->cacheLengths && this->cacheLengths->size() == this->curves.size()) {

                return *this->cacheLengths;
            }

            // Get length of sub-curve
            // Push sums into cached array

            std::vector<float> lengths;
            float sums = 0;

            for (unsigned i = 0, l = this->curves.size(); i < l; i++) {

                sums += this->curves[i]->getLength();
                lengths.emplace_back(sums);
            }

            this->cacheLengths = lengths;

            return lengths;
        }

        std::vector<T> getSpacedPoints(unsigned int divisions = 40) override {

            std::vector<T> points;

            for (unsigned i = 0; i <= divisions; i++) {

                auto& point = points.emplace_back();
                this->getPoint(static_cast<float>(i) / static_cast<float>(divisions), point);
            }

            if (this->autoClose) {

                points.emplace_back(points[0]);
            }

            return points;
        }

        std::vector<T> getPoints(unsigned int divisions = 12) override {

            std::vector<T> points;
            T last;

            for (unsigned i = 0; i < curves.size(); i++) {

                auto& curve = curves[i];

                auto ellipseCurve = std::dynamic_pointer_cast<EllipseCurve>(curve);
                auto lineCurve = std::dynamic_pointer_cast<LineCurve>(curve);
                auto lineCurve3 = std::dynamic_pointer_cast<LineCurve3>(curve);
                auto splineCurve = std::dynamic_pointer_cast<SplineCurve>(curve);


                auto resolution = (curve && ellipseCurve)                ? divisions * 2
                                  : (curve && (lineCurve || lineCurve3)) ? 1
                                  : (curve && splineCurve)               ? divisions * splineCurve->points.size()
                                                                         : divisions;

                auto pts = curve->getPoints(resolution);

                for (unsigned j = 0; j < pts.size(); j++) {

                    auto& point = pts[j];

                    if (!last.isNan() && last.equals(point)) continue;// ensures no consecutive points are duplicates

                    points.emplace_back(point);
                    last = point;
                }
            }

            if (this->autoClose && points.size() > 1 && !points[points.size() - 1].equals(points[0])) {

                points.emplace_back(points[0]);
            }

            return points;
        }

    private:
        std::optional<std::vector<float>> cacheLengths;
    };

}// namespace threepp

#endif//THREEPP_CURVEPATH_HPP
