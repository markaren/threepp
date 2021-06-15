
#include "threepp/math/vector4.hpp"

#include "threepp/math/matrix4.hpp"


using namespace threepp;

vector4::vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

vector4 &vector4::set(float x, float y, float z, float w) {

    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;

    return *this;
}

vector4 &vector4::setScalar(float value) {

    this->x = value;
    this->y = value;
    this->z = value;
    this->w = value;

    return *this;
}

vector4 &vector4::applyMatrix4(const matrix4 &m) {

    const auto x_ = this->x, y_ = this->y, z_ = this->z, w_ = this->w;
    auto e = m.elements_;

    this->x = e[0] * x_ + e[4] * y_ + e[8] * z_ + e[12] * w_;
    this->y = e[1] * x_ + e[5] * y_ + e[9] * z_ + e[13] * w_;
    this->z = e[2] * x_ + e[6] * y_ + e[10] * z_ + e[14] * w_;
    this->w = e[3] * x_ + e[7] * y_ + e[11] * z_ + e[15] * w_;

    return *this;
}
