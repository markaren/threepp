
//#include <threepp/math/Vector2.hpp>
#include <threepp/math/Vector3.hpp>
//#include <threepp/math/Vector4.hpp>
//#include <threepp/math/Triangle.hpp>
//#include <threepp/math/Quaternion.hpp>
//#include <threepp/math/Euler.hpp>
#include <iostream>
#include <thread>
#include <threepp/core/Clock.hpp>
#include <threepp/math/Box2.hpp>
#include <threepp/math/Matrix4.hpp>
#include <threepp/math/Color.hpp>
#include <threepp/lights/AmbientLight.hpp>

#include <threepp/core/Uniform.hpp>

#include <threepp/scenes/Fog.hpp>

#include <threepp/core/EventDispatcher.hpp>

#include <threepp/core/BufferGeometry.hpp>
#include <threepp/core/BufferAttribute.hpp>

#include "threepp/math/Sphere.hpp"

#include <threepp/geometries/BoxGeometry.hpp>
#include <vector>

using namespace threepp;

namespace {
    std::ostream &operator<<(std::ostream &os, const Vector3 &v) {
        os << "Vector3(x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ", z=" + std::to_string(v.z) +
                        ")";
        return os;
    }

}// namespace

int main() {

    Vector3 v;
    v[1] = 2;

    std::cout << "x=" << v << std::endl;

    std::vector<float> a{1, 2, 3};

    v.fromArray(a);

    std::cout << v << std::endl;

    std::vector<float> arr(3);

    v.toArray(arr);

    std::cout << arr[0] << std::endl;

    Matrix4 m4;
    m4.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    m4[0] = 3;
    std::cout << m4[0] << std::endl;

    auto copy = m4;
    copy[0] = -99;

    std::cout << m4[0] << ":" << copy[0] << std::endl;

    Box2 b2;
    std::cout << "empty box: " << (b2.isEmpty() ? "true" : "false") << std::endl;

    threepp::Clock c;
    c.start();
    std::this_thread::sleep_for(std::chrono::milliseconds (500));
    std::cout << "Elapsed=" << c.getElapsedTime() << std::endl;

    EventDispatcher evt;

    EventListener l = [](auto e){
      std::cout << "Event type:" << e.type << std::endl;
    };

    EventListener l1 = [](auto e){
      std::cout << "Event type:" << e.type << std::endl;
    };

    evt.addEventListener("per", &l);
    evt.addEventListener("truls", &l1);

    evt.dispatchEvent("per");
    evt.dispatchEvent("per");

    evt.removeEventListener("per", &l);

    evt.dispatchEvent("per");

    std::cout << "has evt:" << evt.hasEventListener("per", &l) << std::endl;

    Uniform uniform(m4);
    m4[0] = 98;
    auto m = std::any_cast<Matrix4>(uniform.value());
    std::cout << m4[0] << ":" << m[0] << std::endl;

    Fog f( Color::aliceblue);

    std::cout << AmbientLight(0xffffff).type << std::endl;

    std::vector<float> vec {1,2};
    std::vector<float> vec2 {-1, -1};

    BufferAttribute<float> b(vec, 2);
    b.copyArray(vec2);


    std::cout << b.getX(0) << std::endl;

    BoxGeometry box;
    auto& attr = box.getAttribute<float>("position");
    std::cout << attr.getX(0) << std::endl;
    attr.setX(0, 1);
    std::cout << attr.getX(0) << std::endl;
    std::cout << box.getAttribute<float>("position").getX(0) << std::endl;

    return 0;

}
