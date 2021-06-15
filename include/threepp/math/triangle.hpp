
#ifndef THREEPP_TRIANGLE_HPP
#define THREEPP_TRIANGLE_HPP

#include "vector3.hpp"
#include "vector2.hpp"

namespace threepp::math {

    class triangle {

    public:
        triangle() = default;

        triangle(vector3 a, vector3 b, vector3 c) : a_(a), b_(b), c_(c) {};

        static void getNormal(const vector3 &a, const vector3 &b, const vector3 &c, vector3 &target) {

            target.subVectors(c, b);
            _v0.subVectors(a, b);
            target.cross(_v0);

            const auto targetLengthSq = target.lengthSq();
            if (targetLengthSq > 0) {

                target.multiply(1 / std::sqrt(targetLengthSq));

            } else {

                target.set(0, 0, 0);

            }

        }

        // static/instance method to calculate barycentric coordinates
        // based on: http://www.blackpawn.com/texts/pointinpoly/default.html
        static void
        getBarycoord(const vector3 &point, const vector3 &a, const vector3 &b, const vector3 &c, vector3 &target) {

            _v0.subVectors(c, a);
            _v1.subVectors(b, a);
            _v2.subVectors(point, a);

            const auto dot00 = _v0.dot(_v0);
            const auto dot01 = _v0.dot(_v1);
            const auto dot02 = _v0.dot(_v2);
            const auto dot11 = _v1.dot(_v1);
            const auto dot12 = _v1.dot(_v2);

            const auto denom = (dot00 * dot11 - dot01 * dot01);

            // collinear or singular triangle
            if (denom == 0) {

                // arbitrary location outside of triangle?
                // not sure if this is the best idea, maybe should be returning undefined
                target.set(-2, -1, -1);

            } else {

                const auto invDenom = 1.0 / denom;
                const auto u = (dot11 * dot02 - dot01 * dot12) * invDenom;
                const auto v = (dot00 * dot12 - dot01 * dot02) * invDenom;

                // barycentric coordinates must always sum to 1
                target.set(1 - u - v, v, u);

            }

        }

        static bool containsPoint(const vector3 &point, const vector3 &a, const vector3 &b, const vector3 &c) {

            getBarycoord(point, a, b, c, _v3);

            return (_v3.x >= 0) && (_v3.y >= 0) && ((_v3.x + _v3.y) <= 1);

        }

        static void
        getUV(const vector3 &point, const vector3 &p1, const vector3 &p2, const vector3 &p3, const vector2 &uv1,
              const vector2 &uv2, const vector2 &uv3, vector2 &target) {

            getBarycoord(point, p1, p2, p3, _v3);

            target.set(0, 0);
            target.addScaledVector(uv1, _v3.x);
            target.addScaledVector(uv2, _v3.y);
            target.addScaledVector(uv3, _v3.z);

        }

        static bool isFrontFacing(const vector3 &a, const vector3 &b, const vector3 &c, const vector3 &direction) {

            _v0.subVectors(c, b);
            _v1.subVectors(a, b);

            // strictly front facing
            return (_v0.cross(_v1).dot(direction) < 0) ? true : false;

        }

        triangle &set(const vector3 &a, const vector3 &b, const vector3 &c) {

            this->a_ = (a);
            this->b_ = (b);
            this->c_ = (c);

            return *this;

        }

        template<class ArrayLike>
        triangle &setFromPointsAndIndices(const ArrayLike &points, unsigned int i0, unsigned int i1, unsigned int i2) {

            this->a_ = (points[i0]);
            this->b_ = (points[i1]);
            this->c_ = (points[i2]);

            return *this;

        }

        [[nodiscard]] double getArea() const {

            _v0.subVectors(this->c_, this->b_);
            _v1.subVectors(this->a_, this->b_);

            return _v0.cross(_v1).length() * 0.5;

        }

        void getMidpoint(vector3 &target) {

            target.addVectors(this->a_, this->b_).add(this->c_).multiply(1.0 / 3);

        }

        void getNormal(vector3 &target) {

            return triangle::getNormal(this->a_, this->b_, this->c_, target);

        }

        void getBarycoord(vector3 &point, vector3 &target) {

            return triangle::getBarycoord(point, this->a_, this->b_, this->c_, target);

        }

        void getUV(const vector3 &point, const vector2 &uv1, const vector2 &uv2, const vector2 &uv3, vector2 &target) {

            return triangle::getUV(point, this->a_, this->b_, this->c_, uv1, uv2, uv3, target);

        }

        bool containsPoint(const vector3 &point) {

            return triangle::containsPoint(point, this->a_, this->b_, this->c_);

        }

        bool isFrontFacing(const vector3 &direction) {

            return triangle::isFrontFacing(this->a_, this->b_, this->c_, direction);

        }

        void closestPointToPoint(const vector3 &p, vector3 &target) {

            const auto a = this->a_, b = this->b_, c = this->c_;
            double v, w;

            // algorithm thanks to Real-Time Collision Detection by Christer Ericson,
            // published by Morgan Kaufmann Publishers, (c) 2005 Elsevier Inc.,
            // under the accompanying license; see chapter 5.1.5 for detailed explanation.
            // basically, we're distinguishing which of the voronoi regions of the triangle
            // the point lies in with the minimum amount of redundant computation.

            _vab.subVectors(b, a);
            _vac.subVectors(c, a);
            _vap.subVectors(p, a);
            const auto d1 = _vab.dot(_vap);
            const auto d2 = _vac.dot(_vap);
            if (d1 <= 0 && d2 <= 0) {

                // vertex region of A; barycentric coords (1, 0, 0)
                target = (a);
                return;

            }

            _vbp.subVectors(p, b);
            const auto d3 = _vab.dot(_vbp);
            const auto d4 = _vac.dot(_vbp);
            if (d3 >= 0 && d4 <= d3) {

                // vertex region of B; barycentric coords (0, 1, 0)
                target = (b);
                return;
            }

            const auto vc = d1 * d4 - d3 * d2;
            if (vc <= 0 && d1 >= 0 && d3 <= 0) {

                v = d1 / (d1 - d3);
                // edge region of AB; barycentric coords (1-v, v, 0)
                target = (a);
                target.addScaledVector(_vab, v);
                return;

            }

            _vcp.subVectors(p, c);
            const auto d5 = _vab.dot(_vcp);
            const auto d6 = _vac.dot(_vcp);
            if (d6 >= 0 && d5 <= d6) {

                // vertex region of C; barycentric coords (0, 0, 1)
                target = (c);
                return;

            }

            const auto vb = d5 * d2 - d1 * d6;
            if (vb <= 0 && d2 >= 0 && d6 <= 0) {

                w = d2 / (d2 - d6);
                // edge region of AC; barycentric coords (1-w, 0, w)
                target = (a);
                target.addScaledVector(_vac, w);
                return;

            }

            const auto va = d3 * d6 - d5 * d4;
            if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {

                _vbc.subVectors(c, b);
                w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                // edge region of BC; barycentric coords (0, 1-w, w)
                target = (b);
                target.addScaledVector(_vbc, w); // edge region of BC
                return;

            }

            // face region
            const auto denom = 1.0 / (va + vb + vc);
            // u = va * denom
            v = vb * denom;
            w = vc * denom;

            target = (a);
            target.addScaledVector(_vab, v).addScaledVector(_vac, w);

        }


    private:
        vector3 a_ = vector3();
        vector3 b_ = vector3();
        vector3 c_ = vector3();

        static vector3 _v0;
        static vector3 _v1;
        static vector3 _v2;
        static vector3 _v3;

        static vector3 _vab;
        static vector3 _vac;
        static vector3 _vbc;
        static vector3 _vap;
        static vector3 _vbp;
        static vector3 _vcp;

    };

    vector3 triangle::_v0 = vector3();
    vector3 triangle::_v1 = vector3();
    vector3 triangle::_v2 = vector3();
    vector3 triangle::_v3 = vector3();

    vector3 triangle::_vab = vector3();
    vector3 triangle::_vac = vector3();
    vector3 triangle::_vbc = vector3();
    vector3 triangle::_vap = vector3();
    vector3 triangle::_vbp = vector3();
    vector3 triangle::_vcp = vector3();

}

#endif //THREEPP_TRIANGLE_HPP
