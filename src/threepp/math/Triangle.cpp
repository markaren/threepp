
#include "threepp/math/Triangle.hpp"

#include <cmath>

using namespace threepp;

namespace {

    Vector3 _v0{};
    Vector3 _v1{};
    Vector3 _v2{};
    Vector3 _v3{};

}

Triangle::Triangle(Vector3 a, Vector3 b, Vector3 c): a_(a), b_(b), c_(c) {}

const Vector3& Triangle::a() const {

    return a_;
}
const Vector3& Triangle::b() const {

    return b_;
}
const Vector3& Triangle::c() const {

    return c_;
}

void Triangle::getNormal(const Vector3& a, const Vector3& b, const Vector3& c, Vector3& target) {

    target.subVectors(c, b);
    _v0.subVectors(a, b);
    target.cross(_v0);

    const auto targetLengthSq = target.lengthSq();
    if (targetLengthSq > 0) {

        target.multiplyScalar(1 / std::sqrt(targetLengthSq));

    } else {

        target.set(0, 0, 0);
    }
}

void Triangle::getBarycoord(const Vector3& point, const Vector3& a, const Vector3& b, const Vector3& c, Vector3& target) {

    _v0.subVectors(c, a);
    _v1.subVectors(b, a);
    _v2.subVectors(point, a);

    const auto dot00 = _v0.dot(_v0);
    const auto dot01 = _v0.dot(_v1);
    const auto dot02 = _v0.dot(_v2);
    const auto dot11 = _v1.dot(_v1);
    const auto dot12 = _v1.dot(_v2);

    const float denom = (dot00 * dot11 - dot01 * dot01);

    // collinear or singular Triangle
    if (denom == 0) {

        // arbitrary location outside of Triangle?
        // not sure if this is the best idea, maybe should be returning undefined
        target.set(-2, -1, -1);

    } else {

        const auto invDenom = 1.0f / denom;
        const auto u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        const auto v = (dot00 * dot12 - dot01 * dot02) * invDenom;

        // barycentric coordinates must always sum to 1
        target.set(1 - u - v, v, u);
    }
}

bool Triangle::containsPoint(const Vector3& point, const Vector3& a, const Vector3& b, const Vector3& c) {

    getBarycoord(point, a, b, c, _v3);

    return (_v3.x >= 0) && (_v3.y >= 0) && ((_v3.x + _v3.y) <= 1);
}

void Triangle::getUV(const Vector3& point, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector2& uv1, const Vector2& uv2, const Vector2& uv3, Vector2& target) {

    getBarycoord(point, p1, p2, p3, _v3);

    target.set(0, 0);
    target.addScaledVector(uv1, _v3.x);
    target.addScaledVector(uv2, _v3.y);
    target.addScaledVector(uv3, _v3.z);
}

bool Triangle::isFrontFacing(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& direction) {

    _v0.subVectors(c, b);
    _v1.subVectors(a, b);

    // strictly front facing
    return _v0.cross(_v1).dot(direction) < 0;
}

Triangle& Triangle::set(const Vector3& a, const Vector3& b, const Vector3& c) {

    this->a_ = (a);
    this->b_ = (b);
    this->c_ = (c);

    return *this;
}

float Triangle::getArea() const {

    _v0.subVectors(this->c_, this->b_);
    _v1.subVectors(this->a_, this->b_);

    return _v0.cross(_v1).length() * 0.5f;
}

void Triangle::getMidpoint(Vector3& target) {

    target.addVectors(this->a_, this->b_).add(this->c_).addScalar(1.0f / 3);
}

void Triangle::getNormal(Vector3& target) {

    return Triangle::getNormal(this->a_, this->b_, this->c_, target);
}

void Triangle::getBarycoord(Vector3& point, Vector3& target) {

    return Triangle::getBarycoord(point, this->a_, this->b_, this->c_, target);
}

void Triangle::getUV(const Vector3& point, const Vector2& uv1, const Vector2& uv2, const Vector2& uv3, Vector2& target) {

    return Triangle::getUV(point, this->a_, this->b_, this->c_, uv1, uv2, uv3, target);
}

bool Triangle::containsPoint(const Vector3& point) {

    return Triangle::containsPoint(point, this->a_, this->b_, this->c_);
}

bool Triangle::isFrontFacing(const Vector3& direction) {

    return Triangle::isFrontFacing(this->a_, this->b_, this->c_, direction);
}

void Triangle::closestPointToPoint(const Vector3& p, Vector3& target) {

    const auto a = this->a_, b = this->b_, c = this->c_;
    float v, w;

    static Vector3 _vab{};
    static Vector3 _vac{};
    static Vector3 _vap{};
    static Vector3 _vbp{};
    static Vector3 _vcp{};
    static Vector3 _vbc{};


    // algorithm thanks to Real-Time Collision Detection by Christer Ericson,
    // published by Morgan Kaufmann Publishers, (c) 2005 Elsevier Inc.,
    // under the accompanying license; see chapter 5.1.5 for detailed explanation.
    // basically, we're distinguishing which of the voronoi regions of the Triangle
    // the point lies in with the minimum amount of redundant computation.

    _vab.subVectors(b, a);
    _vac.subVectors(c, a);
    _vap.subVectors(p, a);
    const float d1 = _vab.dot(_vap);
    const float d2 = _vac.dot(_vap);
    if (d1 <= 0 && d2 <= 0) {

        // vertex region of A; barycentric coords (1, 0, 0)
        target = (a);
        return;
    }

    _vbp.subVectors(p, b);
    const float d3 = _vab.dot(_vbp);
    const float d4 = _vac.dot(_vbp);
    if (d3 >= 0 && d4 <= d3) {

        // vertex region of B; barycentric coords (0, 1, 0)
        target = (b);
        return;
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {

        v = d1 / (d1 - d3);
        // edge region of AB; barycentric coords (1-v, v, 0)
        target = (a);
        target.addScaledVector(_vab, v);
        return;
    }

    _vcp.subVectors(p, c);
    const float d5 = _vab.dot(_vcp);
    const float d6 = _vac.dot(_vcp);
    if (d6 >= 0 && d5 <= d6) {

        // vertex region of C; barycentric coords (0, 0, 1)
        target = (c);
        return;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {

        w = d2 / (d2 - d6);
        // edge region of AC; barycentric coords (1-w, 0, w)
        target = (a);
        target.addScaledVector(_vac, w);
        return;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {

        _vbc.subVectors(c, b);
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        // edge region of BC; barycentric coords (0, 1-w, w)
        target = (b);
        target.addScaledVector(_vbc, w);// edge region of BC
        return;
    }

    // face region
    const float denom = 1.0f / (va + vb + vc);
    // u = va * denom
    v = vb * denom;
    w = vc * denom;

    target = (a);
    target.addScaledVector(_vab, v).addScaledVector(_vac, w);
}
const Vector3& Triangle::operator[](char c) const {
    switch (c) {
        case 'a': return a_;
        case 'b': return b_;
        case 'c': return c_;
        default: throw std::runtime_error("[Triangle] invalid key: " + std::to_string(c));
    }
}
