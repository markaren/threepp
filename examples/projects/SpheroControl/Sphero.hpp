
#ifndef SPHEROSIM_SPHERO_HPP
#define SPHEROSIM_SPHERO_HPP

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"

#include "threepp/core/Raycaster.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"


class Sphero : public threepp::Object3D {

public:
    class Flags {

    public:
        // Define flag constants
        static const uint8_t None = 0x00;        // 00000001
        static const uint8_t DriveReverse = 0x01;// 00000001
        static const uint8_t Boost = 0x02;       // 00000010
        static const uint8_t FastTurnMode = 0x04;// 00000100

        explicit Flags(uint8_t flags = None) : flags(flags) {}

        // Set a flag
        void setFlag(uint8_t flag) {
            flags |= flag;
        }

        // Clear a flag
        void clearFlag(uint8_t flag) {
            flags &= ~flag;
        }

        void clearAll() {
            flags = None;
        }

        // Check if a flag is set
        [[nodiscard]] bool isFlagSet(uint8_t flag) const {
            return (flags & flag) != None;
        }

    private:
        uint8_t flags;
    };

    Sphero();

    void update(float dt);

    // Run left and right motors at a speed between 0 and 255. Set driving mode using flags.
    void driveRaw(uint8_t leftMode, uint8_t leftSpeed, uint8_t rightMode, uint8_t rightSpeed);

    // Drive towards a heading at a particular speed. Flags can be set to modify driving mode.
    void driveWithHeading(uint8_t speed, uint16_t heading, Flags flags);

    std::pair<float, float> getTofMeasurements();

    threepp::PerspectiveCamera &camera();

private:
    float maxSpeed_ = 4;
    float translationDelta_ = 0;
    float rotationDelta_ = 0;
    std::pair<float, float> tofReadings_;

    threepp::Raycaster raycaster_;
    std::unique_ptr<threepp::PerspectiveCamera> camera_;

    void updateTofMeasurements();

    float getTofMeasurement(const std::string &tofId);
};

#endif//SPHEROSIM_SPHERO_HPP
