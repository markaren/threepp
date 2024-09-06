
#include "threepp/extras/core/Curve.hpp"

#include "threepp/math/Matrix4.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;


template<class T>
void Curve<T>::getPointAt(float u, T& target) const {

    const auto t = this->getUtoTmapping(u);
    return this->getPoint(t, target);
}

template<class T>
std::vector<T> Curve<T>::getPoints(unsigned int divisions) const {

    std::vector<T> points(divisions + 1);

    for (unsigned d = 0; d <= divisions; d++) {
        T& point = points[d];
        this->getPoint(static_cast<float>(d) / static_cast<float>(divisions), point);
    }

    return points;
}

template<class T>
std::vector<T> Curve<T>::getSpacedPoints(unsigned int divisions) const {

    std::vector<T> points(divisions + 1);

    for (unsigned d = 0; d <= divisions; d++) {

        T& point = points[d];
        this->getPointAt(static_cast<float>(d) / static_cast<float>(divisions), point);
    }

    return points;
}

template<class T>
float Curve<T>::getLength() const {

    const auto lengths = this->getLengths();
    return lengths.back();
}

template<class T>
std::vector<float> Curve<T>::getLengths() const {

    return getLengths(this->arcLengthDivisions);
}

template<class T>
std::vector<float> Curve<T>::getLengths(int divisions) const {

    if (static_cast<int>(this->cacheArcLengths.size()) == divisions + 1 && !this->needsUpdate) {

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

template<class T>
void Curve<T>::updateArcLengths() {

    this->needsUpdate = true;
    this->getLengths();
}

template<class T>
float Curve<T>::getUtoTmapping(float u, std::optional<float> distance) const {

    const auto arcLengths = this->getLengths();

    const auto il = arcLengths.size();

    float targetArcLength;// The targeted u distance value to get

    if (distance) {

        targetArcLength = *distance;

    } else {

        targetArcLength = u * arcLengths[il - 1];
    }

    // binary search for the index with largest value smaller than target u distance

    float low = 0, high = il - 1, comparison;

    float i;
    while (low <= high) {

        i = std::floor(low + (high - low) / 2);

        comparison = arcLengths[i] - targetArcLength;

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

    if (arcLengths[i] == targetArcLength) {

        return static_cast<float>(i) / static_cast<float>(il - 1);
    }

    // we could get finer grain at lengths, or use simple interpolation between two points

    const float lengthBefore = arcLengths[i];
    const float lengthAfter = arcLengths[i + 1];

    const float segmentLength = lengthAfter - lengthBefore;

    // determine where we are between the 'before' and 'after' points

    const float segmentFraction = (targetArcLength - lengthBefore) / segmentLength;

    // add that fractional amount to t

    const float t = (i + segmentFraction) / static_cast<float>(il - 1);

    return t;
}

template<class T>
void Curve<T>::getTangent(float t, T& tangent) const {

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

template<class T>
void Curve<T>::getTangentAt(float u, T& optionalTarget) const {

    const float t = this->getUtoTmapping(u);
    this->getTangent(t, optionalTarget);
}

FrenetFrames FrenetFrames::compute(const Curve<Vector3>& curve, unsigned int segments, bool closed) {

    // see http://www.cs.indiana.edu/pub/techreports/TR425.pdf

    Vector3 normal;

    std::vector<Vector3> tangents;
    std::vector<Vector3> normals;
    std::vector<Vector3> binormals;

    Vector3 vec;
    Matrix4 mat;

    // compute the tangent vectors for each segment on the curve

    for (unsigned i = 0; i <= segments; i++) {

        const float u = static_cast<float>(i) / static_cast<float>(segments);

        auto& tangent = tangents.emplace_back();
        curve.getTangentAt(u, tangent);
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

    for (unsigned i = 1; i <= segments; i++) {

        normals.emplace_back(normals[i - 1].clone());

        binormals.emplace_back(binormals[i - 1].clone());

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
        theta /= static_cast<float>(segments);

        if (tangents[0].dot(vec.crossVectors(normals[0], normals[segments])) > 0) {

            theta = -theta;
        }

        for (unsigned i = 1; i <= segments; i++) {

            // twist a little...
            normals[i].applyMatrix4(mat.makeRotationAxis(tangents[i], theta * static_cast<float>(i)));
            binormals[i].crossVectors(tangents[i], normals[i]);
        }
    }

    return {
            tangents,
            normals,
            binormals};
}

template class threepp::Curve<Vector2>;
template class threepp::Curve<Vector3>;
