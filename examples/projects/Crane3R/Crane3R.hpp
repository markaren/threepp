
#ifndef THREEPP_CRANE3R_HPP
#define THREEPP_CRANE3R_HPP

#include "threepp/objects/Group.hpp"

#include "Actuator.hpp"
#include "utility/Angle.hpp"
#include "utility/Regulator.hpp"

#include <array>
#include <utility>


class Crane3R: public threepp::Group {

public:
    bool controllerEnabled = false;

    [[nodiscard]] std::vector<Angle> getValues() const;

    void setTargetValues(const std::vector<Angle>& values);

    void update(float dt);

    static std::shared_ptr<Crane3R> create();

private:
    class Controller {
    public:
        enum ControlMode {
            DIRECT,
            POSITION
        };

        explicit Controller(const Crane3R& c);

        void update(float dt);

        void setGains(const std::vector<float>& values);

        void setTargetValues(const std::vector<Angle>& values);


    private:
        ControlMode mode_{DIRECT};

        std::array<PIDRegulator, 3> pids_{};
        std::array<float, 3> targetValues{};
        std::array<std::unique_ptr<Actuator>, 3> actuators_{};
    };

    std::unique_ptr<Controller> controller_;
    std::array<threepp::Object3D*, 3> parts_{};
    std::array<std::pair<threepp::Object3D*, threepp::Object3D*>, 2> cylinders_{};

    explicit Crane3R(const std::shared_ptr<threepp::Group>& obj);
};


#endif//THREEPP_CRANE3R_HPP
