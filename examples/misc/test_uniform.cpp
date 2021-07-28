
#include "threepp/core/Uniform.hpp"

using namespace threepp;

int main() {

    Uniform u;
    std::cout << "Color r=" << u.value<float>() << std::endl;
    u.value<float>() = 0.5f;
    std::cout << "Color r=" << u.value<float>() << std::endl;

    Vector3 myVec(1.f, 1.f, 1.f);
    Uniform u1(myVec);
    Uniform u2 = u1;
    std::cout << u1.value<Vector3>() << std::endl;
    u1.value<Vector3>().y = -1;
    std::cout << u1.value<Vector3>() << std::endl;
    std::cout << u2.value<Vector3>() << std::endl;

    std::vector<float> vector{1.f, 1.f, 1.f};
    Uniform uv(vector);
    Uniform uv2 = uv;
    std::cout << uv.value<std::vector<float>>()[1] << std::endl;
    uv.value<std::vector<float>>()[1] = -1;
    std::cout << uv.value<std::vector<float>>()[1] << std::endl;
    std::cout << uv2.value<std::vector<float>>()[1] << std::endl;

    std::unordered_map<std::string, Uniform> uniforms;
    uniforms["light"] = Uniform(Vector3(-1, -1, -1));
    uniforms["light"].value<Vector3>().x = 1;

    std::cout << uniforms["light"].value<Vector3>() << std::endl;

    return 0;
}
