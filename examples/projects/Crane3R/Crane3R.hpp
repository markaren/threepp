
#ifndef THREEPP_CRANE3R_HPP
#define THREEPP_CRANE3R_HPP

#include "threepp/objects/Group.hpp"

#include <array>
#include <utility>

class Crane3R : public threepp::Group {

public:

    [[nodiscard]] std::vector<float> getValues(bool degrees = true) const;

    void setTargetValues(const std::vector<float>& values, bool degrees = true);

    void update();

    static std::shared_ptr<Crane3R> create();


private:
    std::array<float, 3> targetValues{};
    std::array<threepp::Object3D *, 3> parts_{};
    std::array<std::pair<threepp::Object3D*, threepp::Object3D*>, 2> cylinders_{};

    explicit Crane3R(const std::shared_ptr<threepp::Group> &obj);
};


#endif//THREEPP_CRANE3R_HPP
