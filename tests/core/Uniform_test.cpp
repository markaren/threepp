

#include "threepp/threepp.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

using namespace threepp;


TEST_CASE("test_Uniform") {

    Uniform u(0.f);
    CHECK(0 == u.value<float>());
    u.value<float>() = 0.5f;
    CHECK(0.5f == Approx(u.value<float>()));

    Vector3 myVec(1.f, 1.f, 1.f);
    Uniform u1(myVec);
    Uniform u2 = u1;
    CHECK(u1.value<Vector3>() == myVec);
    CHECK(u2.value<Vector3>() == myVec);
    u1.value<Vector3>().y = -1;
    CHECK(u1.value<Vector3>().y == Approx(-1));
    CHECK(u2.value<Vector3>().y == Approx(1));

    std::vector<float> vector{1.f, 1.f, 1.f};
    Uniform uv(vector);
    Uniform uv2 = uv;
    CHECK(1.f == Approx(uv.value<std::vector<float>>()[1]));
    CHECK(1.f == Approx(uv2.value<std::vector<float>>()[1]));
    uv.value<std::vector<float>>()[1] = -1;
    CHECK(-1.f == Approx(uv.value<std::vector<float>>()[1]));
    CHECK(1.f == Approx(uv2.value<std::vector<float>>()[1]));

    Vector3 v(-1, -1, -1);
    std::unordered_map<std::string, Uniform> uniforms;
    uniforms["light"] = Uniform(v);
    uniforms["light"].value<Vector3>().x = 1;
    CHECK(v.clone().setX(1) == uniforms.at("light").value<Vector3>());

}
