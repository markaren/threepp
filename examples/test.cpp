
#include <threepp/math/vector2.hpp>
#include <threepp/math/vector3.hpp>
#include <threepp/math/vector4.hpp>
#include <threepp/math/triangle.hpp>
#include <threepp/math/quaternion.hpp>
#include <threepp/math/euler.hpp>
#include <iostream>


using namespace threepp::math;

int main() {

    vector3 v;
    v.setComponent(1, 2).setX(1);

    std::vector<double> a{1, 2, 3};

    v.fromArray(a);

    std::cout << v << std::endl;

    std::vector<double> arr (3);

    v.toArray(arr);

    std::cout << arr[0] << std::endl;

    triangle t;

    euler e;


}
