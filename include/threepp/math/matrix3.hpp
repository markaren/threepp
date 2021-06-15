// https://github.com/mrdoob/three.js/blob/r129/src/math/Matrix3.js

#ifndef THREEPP_MATRIX3_HPP
#define THREEPP_MATRIX3_HPP

namespace threepp {

    class vector3;
    class matrix4;

    class matrix3 {

    public:
        matrix3() = default;

        double &operator[](unsigned int index);

        matrix3 &set(double n11, double n12, double n13, double n21, double n22, double n23, double n31, double n32, double n33);

        matrix3 &identity();

        matrix3 &extractBasis(vector3 &xAxis, vector3 &yAxis, vector3 &zAxis);

        matrix3 &setFromMatrix4(const matrix4 &m);

        matrix3 &multiply(const matrix3 &m) {

            return this->multiplyMatrices(*this, m);
        }

        matrix3 &premultiply(const matrix3 &m) {

            return this->multiplyMatrices(m, *this);
        }

        matrix3 &multiplyMatrices(const matrix3 &a, const matrix3 &b);

        matrix3 &multiplyScalar(double s);

        [[nodiscard]] double determinant() const;

        matrix3 &invert();

        matrix3 &transpose();

        matrix3 &getNormalMatrix(const matrix4 &m);

        template<class ArrayLike>
        matrix3 &transposeIntoArray(ArrayLike &r) {

            const auto m = this->elements_;

            r[0] = m[0];
            r[1] = m[3];
            r[2] = m[6];
            r[3] = m[1];
            r[4] = m[4];
            r[5] = m[7];
            r[6] = m[2];
            r[7] = m[5];
            r[8] = m[8];

            return *this;
        }

        matrix3 &setUvTransform(double tx, double ty, double sx, double sy, double rotation, double cx, double cy);

        matrix3 &scale(double sx, double sy);

        matrix3 &rotate(double theta);

        matrix3 &translate(double tx, double ty);

        template<class ArrayLike>
        matrix3 &fromArray(ArrayLike &array, unsigned int offset = 0) {

            for (auto i = 0; i < 9; i++) {

                this->elements_[i] = array[i + offset];
            }

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            const auto te = this->elements_;

            array[offset] = te[0];
            array[offset + 1] = te[1];
            array[offset + 2] = te[2];

            array[offset + 3] = te[3];
            array[offset + 4] = te[4];
            array[offset + 5] = te[5];

            array[offset + 6] = te[6];
            array[offset + 7] = te[7];
            array[offset + 8] = te[8];

            return array;
        }

    private:
        double elements_[9] = {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1};

        friend class vector3;
        friend class euler;
    };


}// namespace threepp

#endif//THREEPP_MATRIX3_HPP
