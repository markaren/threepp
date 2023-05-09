
#include "threepp/extras/core/CurvePath.hpp"

#include "threepp/extras/curves/EllipseCurve.hpp"
#include "threepp/extras/curves/LineCurve.hpp"
#include "threepp/extras/curves/SplineCurve.hpp"

using namespace threepp;

template class threepp::CurvePath<Vector2>;
template class threepp::CurvePath<Vector3>;

template<class T>
void CurvePath<T>::add(const std::shared_ptr<Curve<T>>& curve) {

    curves.emplace_back(curve);
}

template<class T>
void CurvePath<T>::closePath() {

    T startPoint;
    T endPoint;
    curves.front()->getPoint(0, startPoint);
    curves.back()->getPoint(1, endPoint);

    if (!startPoint.equals(endPoint)) {

        auto curve = std::make_shared<LineCurveT<T>>(endPoint, startPoint);
        curves.emplace_back(curve);
    }
}

template<class T>
void CurvePath<T>::getPoint(float t, T& target) const {

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

template<class T>
float CurvePath<T>::getLength() const {

    auto lens = this->getCurveLengths();
    return lens.back();
}

template<class T>
void CurvePath<T>::updateArcLengths() {

    this->needsUpdate = true;
    this->cacheLengths = std::nullopt;
    this->getCurveLengths();
}

template<class T>
std::vector<float> CurvePath<T>::getCurveLengths() const {

    // We use cache values if curves and cache array are same length

    if (this->cacheLengths && this->cacheLengths->size() == this->curves.size()) {

        return *this->cacheLengths;
    }

    // Get length of sub-curve
    // Push sums into cached array

    std::vector<float> lengths(curves.size());
    float sums = 0;

    for (unsigned i = 0, l = this->curves.size(); i < l; i++) {

        sums += this->curves[i]->getLength();
        lengths[i] = sums;
    }

    this->cacheLengths = lengths;

    return lengths;
}

template<class T>
std::vector<T> CurvePath<T>::getSpacedPoints(unsigned int divisions) const {

    std::vector<T> points;
    points.reserve(divisions + 1);

    for (unsigned i = 0; i <= divisions; i++) {

        auto& point = points.emplace_back();
        this->getPoint(static_cast<float>(i) / static_cast<float>(divisions), point);
    }

    if (this->autoClose) {

        points.emplace_back(points[0]);
    }

    return points;
}

template<class T>
std::vector<T> CurvePath<T>::getPoints(unsigned int divisions) const {

    std::vector<T> points;
    std::optional<T> last;

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

            if (last && last.value().equals(point)) continue;// ensures no consecutive points are duplicates

            points.emplace_back(point);
            last = point;
        }
    }

    if (this->autoClose && points.size() > 1 && !points[points.size() - 1].equals(points[0])) {

        points.emplace_back(points[0]);
    }

    return points;
}
