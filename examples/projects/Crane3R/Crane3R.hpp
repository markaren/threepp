
#ifndef THREEPP_CRANE3R_HPP
#define THREEPP_CRANE3R_HPP

#include "threepp/objects/Group.hpp"

#include <array>
#include <utility>

class Crane3R : public threepp::Group {

public:

    const std::array<float, 3> limMin{-90, -80, 40};
    const std::array<float, 3> limMax{90, 0, 140};

    [[nodiscard]] std::array<float, 3> getAngles(bool degrees = true) const;
    [[nodiscard]] std::array<float, 3> computeAngles(const threepp::Vector3&target) const;

    void setTargetAngles(const std::array<float, 3>& values, bool degrees = true);

    void update();

    static threepp::Vector3 calculateEndEffectorPosition(const std::array<float, 3>& values, bool degrees = true);

    static std::shared_ptr<Crane3R> create();


private:
    std::array<float, 3> targetangles{};
    std::array<threepp::Object3D *, 3> parts_{};
    std::array<std::pair<threepp::Object3D*, threepp::Object3D*>, 2> cylinders_{};

    explicit Crane3R(const std::shared_ptr<threepp::Group> &obj);
};


#endif//THREEPP_CRANE3R_HPP
