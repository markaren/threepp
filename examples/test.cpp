
//#include <threepp/math/vector2.hpp>
#include <threepp/math/vector3.hpp>
//#include <threepp/math/vector4.hpp>
//#include <threepp/math/triangle.hpp>
//#include <threepp/math/quaternion.hpp>
//#include <threepp/math/euler.hpp>
#include <iostream>
#include <threepp/math/matrix4.hpp>
#include <vector>


using namespace threepp;

namespace {
    std::ostream &operator<<(std::ostream &os, const vector3 &v) {
        os << "vector3(x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ", z=" + std::to_string(v.z) +
                        ")";
        return os;
    }
}// namespace

int main() {

    vector3 v;
    v[1] = 2;

    std::cout << "x=" << v << std::endl;

    std::vector<double> a{1, 2, 3};

    v.fromArray(a);

    std::cout << v << std::endl;

    std::vector<double> arr(3);

    v.toArray(arr);

    std::cout << arr[0] << std::endl;

    matrix4 m4;
    m4.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    m4[0] = 3;
    std::cout << m4[0] << std::endl;

    auto copy = m4;
    copy[0] = -99;

    std::cout << m4[0] << ":" << copy[0];

    //    triangle t;
    //
    //    euler e;
}
