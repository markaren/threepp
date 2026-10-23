// Intel RealSense L515 capture without librealsense.
//
// The L515 enumerates as a USB composite device whose interfaces are all bound
// to inbox Windows drivers: depth/IR/confidence (MI_00) and RGB (MI_04) are
// UVC cameras under usbvideo, the hardware-monitor channel (MI_06) is WinUSB
// and the IMU (MI_07) is a HID sensor collection. This class reads the two UVC
// interfaces through Media Foundation, so it needs only system libraries.
//
// What the device delivers, and what this class does with it:
//   * Depth is Z16 in 0.25 mm units (depthUnit), 0 = no return. The sensor is
//     mounted rotated, so the UVC frames are portrait (480x640 for the
//     640x480 mode); they are turned to landscape here so that up and left
//     match the RGB frame.
//   * RGB is YUY2, converted to RGBA8 here.
//   * Intrinsics are NOMINAL, derived from the datasheet field of view
//     (depth 70 x 55 deg, RGB 70 x 43 deg). The per-unit calibration lives
//     behind the WinUSB hardware monitor and is not read yet, so a cloud
//     built from these is the right shape but not metrically exact, and depth
//     and RGB are not registered to each other.
//
// Each stream runs on its own capture thread; poll*() is non-blocking and
// hands out the newest frame once.
//
//     RealSenseL515 cam;                      // default: depth 640x480, RGB 1280x720
//     if (!cam.depthOpen()) { std::cerr << cam.lastError(); }
//     RealSenseL515::DepthFrame depth;
//     if (cam.pollDepth(depth)) {
//         auto p = depth.intrinsics.deproject(u, v, depth.metres(u, v));
//     }
//
// On platforms other than Windows the class compiles but never opens.

#ifndef THREEPP_SENSORS_REALSENSEL515_HPP
#define THREEPP_SENSORS_REALSENSEL515_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class RealSenseL515 {

    public:
        /// Pinhole model without distortion:
        /// u = fx * x / z + ppx, v = fy * y / z + ppy (x right, y down, z forward).
        struct Intrinsics {
            int width = 0;
            int height = 0;
            float fx = 0, fy = 0;
            float ppx = 0, ppy = 0;

            /// Camera-frame point (x right, y down, z forward) [m].
            [[nodiscard]] std::array<float, 3> deproject(float u, float v, float z) const {
                return {(u - ppx) / fx * z, (v - ppy) / fy * z, z};
            }

            /// Nominal intrinsics for an image of the given size and field of view [deg].
            static Intrinsics fromFov(int width, int height, float hfovDeg, float vfovDeg);
        };

        struct DepthFrame {
            int width = 0;
            int height = 0;
            std::vector<std::uint16_t> data;///< row-major, `width * height`
            float depthUnit = 0.00025f;     ///< metres per count
            Intrinsics intrinsics;
            std::uint64_t frameNumber = 0;///< counts frames this stream delivered
            double timestamp = 0;         ///< device sample time [s]

            [[nodiscard]] float metres(int u, int v) const {
                return static_cast<float>(data[static_cast<size_t>(v) * width + u]) * depthUnit;
            }
        };

        struct ColorFrame {
            int width = 0;
            int height = 0;
            std::vector<std::uint8_t> rgba;///< row-major RGBA8, `width * height * 4`
            Intrinsics intrinsics;
            std::uint64_t frameNumber = 0;
            double timestamp = 0;///< device sample time [s]
        };

        struct Config {
            bool depth = true;
            /// Landscape size after rotation: 320x240, 640x480 or 1024x768.
            int depthWidth = 640;
            int depthHeight = 480;

            bool color = true;
            /// 1920x1080 (15/30 fps), 1280x720 or 960x540 (15/30/60 fps).
            int colorWidth = 1280;
            int colorHeight = 720;
            int colorFps = 30;

            /// Which L515 to open when several are connected (enumeration order).
            int deviceIndex = 0;
        };

        RealSenseL515();
        explicit RealSenseL515(const Config& config);

        ~RealSenseL515();

        RealSenseL515(const RealSenseL515&) = delete;
        RealSenseL515& operator=(const RealSenseL515&) = delete;

        /// Number of L515 depth interfaces currently connected.
        static int deviceCount();

        [[nodiscard]] bool depthOpen() const;
        [[nodiscard]] bool colorOpen() const;

        /// Why a stream failed to open or stopped; empty while all is well.
        [[nodiscard]] std::string lastError() const;

        /// Copies the newest depth frame into `out` if one arrived since the
        /// previous call. Reuses `out`'s storage.
        bool pollDepth(DepthFrame& out);

        /// Copies the newest RGB frame into `out` if one arrived since the
        /// previous call. Reuses `out`'s storage.
        bool pollColor(ColorFrame& out);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_SENSORS_REALSENSEL515_HPP
