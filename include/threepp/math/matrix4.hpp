// https://github.com/mrdoob/three.js/blob/r129/src/math/Matrix4.js

#ifndef THREEPP_MATRIX4_HPP
#define THREEPP_MATRIX4_HPP


namespace threepp {

    class vector3;
    class euler;
    class quaternion;

    class matrix4 {

    public:
        matrix4() = default;

        double &operator[](unsigned int index);

        matrix4 &set(double n11, double n12, double n13, double n14, double n21, double n22, double n23, double n24, double n31, double n32, double n33, double n34, double n41, double n42, double n43, double n44);

        matrix4 &identity();

        matrix4 &copyPosition(const matrix4 &m);

        matrix4 &setFromMatrix3(const matrix4 &m);

        matrix4 &extractBasis(vector3 &xAxis, vector3 &yAxis, vector3 &zAxis);

        matrix4 &makeBasis(const vector3 &xAxis, const vector3 &yAxis, const vector3 &zAxis);

        matrix4 &extractRotation(const matrix4 &m);

        matrix4 &makeRotationFromEuler(const euler &e);

        matrix4 &makeRotationFromQuaternion(const quaternion &q);

        matrix4 &lookAt(const vector3 &eye, const vector3 &target, const vector3 &up);

        matrix4 &multiply(const matrix4 &m);

        matrix4 &premultiply(const matrix4 &m);

        matrix4 &multiplyMatrices(const matrix4 &a, const matrix4 &b);

        matrix4 &multiplyScalar(double s);

        [[nodiscard]] double determinant() const;

        matrix4 &transpose();

        matrix4 &setPosition(const vector3 &v);

        matrix4 &setPosition(double x, double y, double z);

        matrix4 &invert();

        matrix4 &scale(const vector3 &v);

        [[nodiscard]] double getMaxScaleOnAxis() const;

        matrix4 &makeTranslation(double x, double y, double z);

        matrix4 &makeRotationX(double theta);

        matrix4 &makeRotationY(double theta);

        matrix4 &makeRotationZ(double theta);

        matrix4 &makeRotationAxis(const vector3 &axis, double angle);

        matrix4 &makeScale(double x, double y, double z);

        matrix4 &makeShear(double xy, double xz, double yx, double yz, double zx, double zy);

        matrix4 &compose(const vector3 &position, const quaternion &quaternion, const vector3 &scale);

        matrix4 &decompose( vector3 &position, quaternion &quaternion, vector3 &scale );

        matrix4 &makePerspective( double left, double right, double top, double bottom, double near, double far );

        matrix4 &makeOrthographic( double left, double right, double top, double bottom, double near, double far );

        template<class ArrayLike>
        matrix4 &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            for (auto i = 0; i < 16; i++) {

                this->elements_[i] = array[i + offset];
            }

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) {

            auto te = this->elements_;

            array[offset] = te[0];
            array[offset + 1] = te[1];
            array[offset + 2] = te[2];
            array[offset + 3] = te[3];

            array[offset + 4] = te[4];
            array[offset + 5] = te[5];
            array[offset + 6] = te[6];
            array[offset + 7] = te[7];

            array[offset + 8] = te[8];
            array[offset + 9] = te[9];
            array[offset + 10] = te[10];
            array[offset + 11] = te[11];

            array[offset + 12] = te[12];
            array[offset + 13] = te[13];
            array[offset + 14] = te[14];
            array[offset + 15] = te[15];
        }

    private:
        double elements_[16] = {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1};

        friend class euler;
        friend class quaternion;

        friend class matrix3;

        friend class vector3;
        friend class vector4;

        static vector3 _v1;
        static matrix4 _m1;
        static vector3 _zero;
        static vector3 _one;
        static vector3 _x;
        static vector3 _y;
        static vector3 _z;
    };

}// namespace threepp

#endif//THREEPP_MATRIX4_HPP
