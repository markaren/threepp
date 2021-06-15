
#ifndef THREEPP_TRIANGLE_HPP
#define THREEPP_TRIANGLE_HPP

#include "vector2.hpp"
#include "vector3.hpp"

namespace threepp {

    class triangle {

    public:
        triangle() = default;

        triangle(vector3 a, vector3 b, vector3 c);;

        static void getNormal(const vector3 &a, const vector3 &b, const vector3 &c, vector3 &target);

        // static/instance method to calculate barycentric coordinates
        // based on: http://www.blackpawn.com/texts/pointinpoly/default.html
        static void
        getBarycoord(const vector3 &point, const vector3 &a, const vector3 &b, const vector3 &c, vector3 &target);

        static bool containsPoint(const vector3 &point, const vector3 &a, const vector3 &b, const vector3 &c);

        static void
        getUV(const vector3 &point, const vector3 &p1, const vector3 &p2, const vector3 &p3, const vector2 &uv1,
              const vector2 &uv2, const vector2 &uv3, vector2 &target);

        static bool isFrontFacing(const vector3 &a, const vector3 &b, const vector3 &c, const vector3 &direction);

        triangle &set(const vector3 &a, const vector3 &b, const vector3 &c);

        template<class ArrayLike>
        triangle &setFromPointsAndIndices(const ArrayLike &points, unsigned int i0, unsigned int i1, unsigned int i2) {

            this->a_ = (points[i0]);
            this->b_ = (points[i1]);
            this->c_ = (points[i2]);

            return *this;
        }

        [[nodiscard]] double getArea() const;

        void getMidpoint(vector3 &target);

        void getNormal(vector3 &target);

        void getBarycoord(vector3 &point, vector3 &target);

        void getUV(const vector3 &point, const vector2 &uv1, const vector2 &uv2, const vector2 &uv3, vector2 &target);

        bool containsPoint(const vector3 &point);

        bool isFrontFacing(const vector3 &direction);

        void closestPointToPoint(const vector3 &p, vector3 &target);

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


}// namespace threepp

#endif//THREEPP_TRIANGLE_HPP
