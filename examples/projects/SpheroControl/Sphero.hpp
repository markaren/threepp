
#ifndef SPHEROSIM_SPHERO_HPP
#define SPHEROSIM_SPHERO_HPP

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/math/Color.hpp"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace threepp {
    class Group;
    class Line;
    class Mesh;
    class MeshBasicMaterial;
    class Texture;
}// namespace threepp

// A Sphero RVR at its real size (185 x 216 x 114 mm, metres throughout), driven
// through the two drive commands of the real robot. On top of the stock rover
// sit the usual lab additions: a camera and a time-of-flight sensor at each end.
//
// The robot stands on the ground at its origin and drives along its local +z.
// Left is local +x.
class Sphero: public threepp::Object3D {

public:
    class Flags {

    public:
        // Define flag constants
        static const uint8_t None = 0x00;        // 00000000
        static const uint8_t DriveReverse = 0x01;// 00000001
        static const uint8_t Boost = 0x02;       // 00000010
        static const uint8_t FastTurnMode = 0x04;// 00000100

        explicit Flags(uint8_t flags = None): flags(flags) {}

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

    // The disc the robot occupies on the floor, for whoever keeps it out of the walls.
    static constexpr float footprintRadius = 0.125f;

    // Top speed of a track at a motor speed of 255 [m/s].
    static constexpr float maxSpeed = 1.5f;

    // The time-of-flight sensors see this far [m]; beyond it they read infinity.
    static constexpr float tofRange = 2.f;

    Sphero();

    // Advances the motors, the pose and the sensors by dt seconds.
    void update(float dt);

    // Run the left and right motors at a speed between 0 and 255.
    // Mode 0x0 is off, 0x1 forward and 0x2 reverse.
    void driveRaw(uint8_t leftMode, uint8_t leftSpeed, uint8_t rightMode, uint8_t rightSpeed);

    // Drive towards a heading [deg, 0 to 359, clockwise seen from above] at a speed
    // between 0 and 255. Flags can be set to modify driving mode.
    void driveWithHeading(uint8_t speed, uint16_t heading, Flags flags);

    // Like the real robot, the motors stop when no drive command has arrived for this long [s].
    static constexpr float commandTimeout = 2.f;

    void setAllLeds(const threepp::Color& color);

    // Front and rear range [m], infinity when nothing is within tofRange.
    [[nodiscard]] std::pair<float, float> getTofMeasurements() const;

    // Heading [deg, 0 to 360, clockwise seen from above], as the IMU reports it.
    [[nodiscard]] float heading() const;

    // Forward speed [m/s], negative in reverse.
    [[nodiscard]] float speed() const;

    // Shows or hides the beams drawn for the time-of-flight sensors.
    // They are on layer 1, which the onboard camera does not see.
    void showTofBeams(bool show);

    threepp::PerspectiveCamera& camera();

    ~Sphero() override;

private:
    struct Track {
        float speed = 0;// [m/s]
        std::vector<std::shared_ptr<threepp::Texture>> treadMaps;// colour, normal, roughness: scrolled as one
        std::shared_ptr<threepp::Group> wheels[2];
    };

    struct Tof {
        std::shared_ptr<threepp::Object3D> mount;
        std::shared_ptr<threepp::Line> beam;
        std::shared_ptr<threepp::Mesh> dot;
        float reading;
    };

    enum class Command { None,
                         Raw,
                         Heading };

    Command command_ = Command::None;
    float commandAge_ = 0;
    float rawLeft_ = 0, rawRight_ = 0;// [m/s]
    float headingSpeed_ = 0;          // [m/s]
    float headingYaw_ = 0;            // [rad]
    float headingTurnRate_ = 0;       // [rad/s]

    Track left_, right_;
    Tof tofFront_, tofRear_;
    bool showTofBeams_ = true;

    std::shared_ptr<threepp::MeshBasicMaterial> ledMaterial_;

    threepp::Raycaster raycaster_;
    std::unique_ptr<threepp::PerspectiveCamera> camera_;

    void buildModel();

    void updateTof(Tof& tof);
};

#endif//SPHEROSIM_SPHERO_HPP
