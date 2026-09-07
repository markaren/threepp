
#ifndef THREEPP_TRANSFORMCONTROLSSTATE_HPP
#define THREEPP_TRANSFORMCONTROLSSTATE_HPP

#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"


using namespace threepp;

struct State {

    Vector3 eye;
    Vector3 worldPosition;
    Quaternion worldQuaternion;
    Quaternion cameraQuaternion;
    Vector3 cameraPosition;

    Vector3 worldPositionStart;
    Quaternion worldQuaternionStart;

    Vector3 rotationAxis;

    bool& enabled;
    std::string mode{"translate"};
    std::string space{"world"};
    std::optional<std::string> axis;
    float size{1.f};
    bool dragging{false};
    bool& showX;
    bool& showY;
    bool& showZ;

    std::optional<float> rotationSnap;
    std::optional<float> translationSnap;
    std::optional<float> scaleSnap;

    Camera* camera = nullptr;

    explicit State(bool& enabled, bool& showX, bool& showY, bool& showZ)
        : enabled(enabled), showX(showX), showY(showY), showZ(showZ) {}
};

#endif//THREEPP_TRANSFORMCONTROLSSTATE_HPP
