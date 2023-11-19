
#include "threepp/core/Uniform.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace threepp;


TEST_CASE("test_Uniform") {

    Uniform u(0.f);
    CHECK(0 == u.value<float>());
    u.value<float>() = 0.5f;
    CHECK_THAT(0.5f, Catch::Matchers::WithinRel(u.value<float>()));

    Vector3 myVec(1.f, 1.f, 1.f);
    Uniform u1(myVec);
    Uniform u2 = u1;
    CHECK(u1.value<Vector3>() == myVec);
    CHECK(u2.value<Vector3>() == myVec);
    u1.value<Vector3>().y = -1;
    CHECK_THAT(u1.value<Vector3>().y, Catch::Matchers::WithinRel(-1.f));
    CHECK_THAT(u2.value<Vector3>().y, Catch::Matchers::WithinRel(1.f));

    std::vector<float> vector{1.f, 1.f, 1.f};
    Uniform uv(vector);
    Uniform uv2 = uv;
    CHECK_THAT(1.f, Catch::Matchers::WithinRel(uv.value<std::vector<float>>()[1]));
    CHECK_THAT(1.f, Catch::Matchers::WithinRel(uv2.value<std::vector<float>>()[1]));
    uv.value<std::vector<float>>()[1] = -1;
    CHECK_THAT(-1.f, Catch::Matchers::WithinRel(uv.value<std::vector<float>>()[1]));
    CHECK_THAT(1.f, Catch::Matchers::WithinRel(uv2.value<std::vector<float>>()[1]));

    Vector3 v(-1, -1, -1);
    std::unordered_map<std::string, Uniform> uniforms;
    uniforms["light"] = Uniform(v);
    uniforms["light"].value<Vector3>().x = 1;
    CHECK(v.clone().setX(1) == uniforms.at("light").value<Vector3>());
}
