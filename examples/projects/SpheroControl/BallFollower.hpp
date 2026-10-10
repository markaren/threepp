
#ifndef SPHEROSIM_BALLFOLLOWER_HPP
#define SPHEROSIM_BALLFOLLOWER_HPP

#include "Sphero.hpp"

#include <algorithm>
#include <cmath>

// Keeps the robot facing a magenta ball, a set distance from it, with nothing
// to go on but what the robot itself can sense: the camera frame gives the
// ball's bearing and (from its apparent size) its range, the IMU the heading,
// and the front time-of-flight sensor stops it short of anything in the way.
//
// The image step is a colour threshold and a bounding box, what
// cv::inRange and cv::boundingRect would do with OpenCV at hand.
class BallFollower {

public:
    struct Detection {
        bool found = false;
        float x = 0, y = 0;// centre [px], from the top-left corner of the image
        float diameter = 0;// [px]
        float bearing = 0; // [deg], positive to the right
        float range = 0;   // [m]
    };

    // ballDiameter and standoff (the range to hold) in metres.
    BallFollower(Sphero& sphero, float ballDiameter, float standoff = 0.4f)
        : sphero_(&sphero), ballDiameter_(ballDiameter), standoff_(standoff) {}

    // Looks for the ball in a camera frame and sends the next drive command.
    // The rows of `pixels` run bottom-up, as a frame read back from GL does.
    const Detection& update(const unsigned char* pixels, int width, int height, int channels) {

        detect(pixels, width, height, channels);

        if (detection_.found) {
            framesLost_ = 0;
            lastSide_ = detection_.bearing < 0 ? -1 : 1;

            const float error = detection_.range - standoff_;
            float speed = std::abs(error) < deadband ? 0 : std::min(std::abs(error) * speedGain, topSpeed);

            Sphero::Flags flags;
            if (error < 0) {
                flags.setFlag(Sphero::Flags::DriveReverse);
            } else if (sphero_->getTofMeasurements().first < tofStop) {
                speed = 0;
            }

            const float heading = std::fmod(sphero_->heading() + detection_.bearing + 360.f, 360.f);
            sphero_->driveWithHeading(static_cast<uint8_t>(speed), static_cast<uint16_t>(heading) % 360, flags);

        } else if (++framesLost_ > framesBeforeSearch) {

            // Turn on the spot, towards where the ball was last seen.
            if (lastSide_ > 0) {
                sphero_->driveRaw(0x1, searchSpeed, 0x2, searchSpeed);
            } else {
                sphero_->driveRaw(0x2, searchSpeed, 0x1, searchSpeed);
            }
        }

        return detection_;
    }

    [[nodiscard]] const Detection& detection() const {
        return detection_;
    }

private:
    static constexpr float deadband = 0.03f;  // [m]
    static constexpr float speedGain = 300.f; // motor speed per metre of range error
    static constexpr float topSpeed = 110.f;  // of 255
    static constexpr float tofStop = 0.12f;   // [m]
    static constexpr uint8_t searchSpeed = 25;// of 255
    static constexpr int framesBeforeSearch = 4;
    static constexpr int minPixels = 12;

    Sphero* sphero_;
    float ballDiameter_;
    float standoff_;
    Detection detection_;
    int framesLost_ = 0;
    int lastSide_ = 1;

    // Magenta: red and blue well above green.
    static bool isBall(const unsigned char* p) {
        return p[0] > 60 && p[2] > 50 && static_cast<float>(p[1]) < 0.6f * static_cast<float>(std::min(p[0], p[2]));
    }

    void detect(const unsigned char* pixels, int width, int height, int channels) {

        int count = 0;
        int minX = width, maxX = -1, minY = height, maxY = -1;
        for (int y = 0; y < height; ++y) {
            const unsigned char* row = pixels + static_cast<size_t>(y) * width * channels;
            for (int x = 0; x < width; ++x) {
                if (!isBall(row + x * channels)) continue;
                ++count;
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
            }
        }

        detection_.found = count >= minPixels;
        if (!detection_.found) return;

        // The longer side of the box still spans the ball when something hides part of it.
        detection_.diameter = static_cast<float>(std::max(maxX - minX, maxY - minY) + 1);
        detection_.x = static_cast<float>(minX + maxX + 1) / 2;
        detection_.y = static_cast<float>(height) - static_cast<float>(minY + maxY + 1) / 2;

        const float focalLength = static_cast<float>(height) / 2 / std::tan(sphero_->camera().fov * threepp::math::DEG2RAD / 2);// [px]
        detection_.bearing = std::atan((detection_.x - static_cast<float>(width) / 2) / focalLength) * threepp::math::RAD2DEG;
        detection_.range = ballDiameter_ * focalLength / detection_.diameter;
    }
};

#endif//SPHEROSIM_BALLFOLLOWER_HPP
