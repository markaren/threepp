// Side-by-side stereo for AR/XR glasses, with head tracking from VITURE's
// Luma Ultra.
//
// Glasses of this kind are a USB-C monitor. In their 3D mode the monitor is
// twice as wide (3840x1200 on the Luma Ultra), its left half is shown to the
// left eye and its right half to the right eye. This example renders the scene
// twice per frame, once into each half, from two cameras an eye separation
// apart (threepp::StereoCamera). The head's pose comes from the glasses, so the
// objects stay where they are in the room while the head moves.
//
// The displays are see-through and only add light: black is transparent. The
// scene has no background for that reason.
//
// Head tracking needs VITURE's XR Glasses SDK (viture.com/developer), which is
// not part of this repository. Configure with -DVITURE_SDK_DIR=<sdk root> to
// build it in. Without the SDK, or with no Luma Ultra connected, the left mouse
// button turns the head and W A S D move it. On Windows one process at a time
// can open the glasses, so VITURE's own SpaceWalker has to be closed. The SDK
// reaches them through libusb, which needs the WinUSB driver on the tracking
// device (USB 35CA:1104); Windows does not install one by itself.
//
// Observed with SDK 2.4.3 on Windows 11 (2026-11): a process killed while it
// has the glasses open leaves them in 3D mode, and the tracker then returns
// the identity pose to every later session until the glasses are replugged.
// Quit with Esc. The two sessions started right after plugging the glasses in
// lost their USB link; the one started a minute after did not.
//
// The window goes borderless fullscreen on a 3840x1200 or 3840x1080 monitor
// when there is one, and is an ordinary window otherwise. With the SDK the
// glasses are switched to 3D mode at start and back at exit; without it, switch
// them with their own button before starting.
//
//   R        recentre: the head's present place and heading become the origin
//   F        fullscreen on the monitor under the window, on/off
//   [ ]      eye separation, 1 mm steps
//   - =      convergence distance
//   , .      vertical field of view
//   Esc      quit
//
// Arguments:
//   --vulkan      render with the Vulkan renderer (OpenGL without it)
//   --log-pose    print the head's position and orientation once a second
//   --no-glasses  leave the glasses alone: mouse and keyboard move the head
//   --sdk-log     print everything VITURE's SDK logs, not only its errors
//
// On Vulkan the left eye is the frame's main render, confined to the left half
// by the scissor rectangle, and the right eye is an added view (addView) shown
// in the right half: both come from one scene build and one instant. An added
// view draws no lines, points or wireframes, so on Vulkan the ball is solid and
// the floor grid is built from thin strips of mesh.

#include "threepp/threepp.hpp"

#include "threepp/canvas/Monitor.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <thread>
#include <vector>

#ifdef THREEPP_EXAMPLE_WITH_VITURE
#include "viture_device_carina.h"
#include "viture_glasses_provider.h"
#include "viture_protocol_public.h"
#include "viture_result.h"

#ifdef _WIN32
// After the threepp headers: windows.h defines near, far and min/max macros.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <setupapi.h>
#endif
#endif

using namespace threepp;

namespace {

    // One eye's picture on the Luma Ultra is 1920x1200 and 52 degrees across
    // its diagonal (the published figure), which is 29 degrees vertically.
    constexpr float kFovVertical = 29.f;

    // The distance at which a point has the same pixel position in both
    // halves. It should equal the distance at which the glasses' optics make
    // the two displays coincide; VITURE's SDK documentation does not give that
    // distance, so this is a starting value for the - and = keys.
    constexpr float kConvergence = 4.f;

    constexpr float kEyeSeparation = 0.063f;

    // The scene's floor grid, below the origin. The origin is the head's place
    // at the last recentre.
    constexpr float kFloorBelowEyes = 1.2f;

    struct HeadPose {
        Vector3 position;
        Quaternion orientation;
    };

#ifdef THREEPP_EXAMPLE_WITH_VITURE

    // The USB product ids of the connected VITURE glasses (vendor id 0x35CA).
    // xr_device_provider_create returns a handle for a product that is not
    // plugged in, so the USB bus is what tells which one is.
    std::vector<int> connectedProducts() {
        std::vector<int> pids;
        auto add = [&](int pid) {
            if (xr_device_provider_is_product_id_valid(pid) && std::find(pids.begin(), pids.end(), pid) == pids.end()) {
                pids.push_back(pid);
            }
        };
#ifdef _WIN32
        const HDEVINFO devices = SetupDiGetClassDevsA(nullptr, "USB", nullptr, DIGCF_PRESENT | DIGCF_ALLCLASSES);
        if (devices == INVALID_HANDLE_VALUE) return pids;
        SP_DEVINFO_DATA device{};
        device.cbSize = sizeof(device);
        for (DWORD i = 0; SetupDiEnumDeviceInfo(devices, i, &device); ++i) {
            // A list of strings such as "USB\VID_35CA&PID_1101&REV_0100".
            char ids[4096] = {};
            if (!SetupDiGetDeviceRegistryPropertyA(devices, &device, SPDRP_HARDWAREID, nullptr,
                                                   reinterpret_cast<PBYTE>(ids), sizeof(ids) - 2, nullptr)) {
                continue;
            }
            for (const char* id = ids; *id; id += std::strlen(id) + 1) {
                const char* vid = std::strstr(id, "VID_35CA");
                const char* pid = std::strstr(id, "PID_");
                if (vid && pid) add(static_cast<int>(std::strtol(pid + 4, nullptr, 16)));
            }
        }
        SetupDiDestroyDeviceInfoList(devices);
#else
        // Without a look at the bus: the Luma Ultra's two product ids, from the
        // SDK's hardware table. The first one that starts is used.
        add(0x1101);
        add(0x1104);
#endif
        return pids;
    }

    // The Luma Ultra through VITURE's C SDK. The SDK calls it "Carina", after
    // its tracking engine, and gives its pose by polling, as
    // [px, py, pz, qw, qx, qy, qz] in metres with x right, y up and z backward:
    // threepp's own convention.
    class LumaUltra {

    public:
        // logLevel is the SDK's: 1 prints its errors, 3 everything.
        explicit LumaUltra(int logLevel) {
            xr_device_provider_set_log_level(logLevel);

            for (const int pid : connectedProducts()) {
                const XRDeviceProviderHandle handle = xr_device_provider_create(pid);
                if (!handle) continue;

                if (xr_device_provider_get_device_type(handle) != XR_DEVICE_TYPE_VITURE_CARINA) {
                    std::cout << "VITURE: USB product 0x" << std::hex << pid << std::dec
                              << " is connected; this example reads the Luma Ultra's pose only\n";
                    xr_device_provider_destroy(handle);
                    continue;
                }

                // Position and orientation (1) rather than orientation only
                // (0). The SDK takes this between create and initialize only.
                xr_device_provider_set_dof_type_carina(handle, 1);

                int rc = xr_device_provider_initialize(handle, nullptr, nullptr);
                if (rc == VITURE_GLASSES_SUCCESS) {
                    rc = xr_device_provider_start(handle);
                    if (rc != VITURE_GLASSES_SUCCESS) xr_device_provider_shutdown(handle);
                }
                if (rc != VITURE_GLASSES_SUCCESS) {
                    std::cerr << "VITURE: USB product 0x" << std::hex << pid << std::dec
                              << " did not start (error " << rc << ")\n";
#ifdef _WIN32
                    // The SDK reaches the Luma Ultra through libusb, which on
                    // Windows opens a device only through WinUSB (or libusbK).
                    std::cerr << "VITURE: check in Device Manager that \"VITURE Luma Ultra XR GLASSES\" has a driver,"
                                 " and that no other program has the glasses open\n";
#endif
                    xr_device_provider_destroy(handle);
                    continue;
                }

                handle_ = handle;
                return;
            }
        }

        LumaUltra(const LumaUltra&) = delete;
        LumaUltra& operator=(const LumaUltra&) = delete;

        ~LumaUltra() {
            if (!handle_) return;
            if (restoreMode_ > 0) {
                // The command has been seen to fail once and pass on the next
                // try ("Failed to send USB command", error -1).
                int rc = VITURE_GLASSES_ERROR_UNKNOWN;
                for (int attempt = 0; attempt < 5 && rc != VITURE_GLASSES_SUCCESS; ++attempt) {
                    if (attempt > 0) std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    rc = xr_device_provider_set_display_mode(handle_, restoreMode_);
                }
                if (rc != VITURE_GLASSES_SUCCESS) {
                    std::cerr << "VITURE: the glasses did not take their earlier display mode back (error " << rc
                              << "); they are still side by side" << std::endl;
                }
            }
            xr_device_provider_stop(handle_);
            xr_device_provider_shutdown(handle_);
            xr_device_provider_destroy(handle_);
        }

        [[nodiscard]] bool connected() const { return handle_ != nullptr; }

        // 3840x1200, at 90 Hz when the glasses accept it and 60 Hz otherwise.
        // The destructor puts back the mode the glasses were in. If that was
        // already a side-by-side mode (a run that ended without putting them
        // back leaves them so), it is the 2D mode of the same size and rate.
        bool enterSideBySide() {
            const int mode = xr_device_provider_get_display_mode(handle_);
            if (mode > 0) restoreMode_ = flatMode(mode);
            for (const int wanted : {VITURE_DISPLAY_MODE_3840_1200_90HZ, VITURE_DISPLAY_MODE_3840_1200_60HZ}) {
                if (mode == wanted) return true;
                if (xr_device_provider_set_display_mode(handle_, wanted) == VITURE_GLASSES_SUCCESS) return true;
            }
            return false;
        }

        // The pose predicted `aheadSeconds` from now. `stable` is false for a
        // short time after start and after a sudden movement. The SDK takes
        // the prediction time in seconds (its 2.4.0 release notes; the guide's
        // example passes nanoseconds).
        bool pose(float aheadSeconds, HeadPose& out, bool& stable) {
            float p[7];
            int status = 0;
            if (xr_device_provider_get_gl_pose_carina(handle_, p, aheadSeconds, &status) != VITURE_GLASSES_SUCCESS) {
                return false;
            }
            out.position.set(p[0], p[1], p[2]);
            out.orientation.set(p[4], p[5], p[6], p[3]);
            stable = status == 0;
            return true;
        }

        // Position and heading restart from the present pose. Pitch and roll
        // stay tied to gravity.
        void recentre() {
            float p[7];
            int status = 0;
            if (xr_device_provider_get_gl_pose_carina(handle_, p, 0.0, &status) == VITURE_GLASSES_SUCCESS) {
                xr_device_provider_reset_origin_carina(handle_, p);
            }
        }

    private:
        static int flatMode(int mode) {
            switch (mode) {
                case VITURE_DISPLAY_MODE_3840_1200_90HZ:
                    return VITURE_DISPLAY_MODE_1920_1200_90HZ;
                case VITURE_DISPLAY_MODE_3840_1200_60HZ:
                    return VITURE_DISPLAY_MODE_1920_1200_60HZ;
                case VITURE_DISPLAY_MODE_3840_1080_90HZ:
                    return VITURE_DISPLAY_MODE_1920_1080_90HZ;
                case VITURE_DISPLAY_MODE_3840_1080_60HZ:
                    return VITURE_DISPLAY_MODE_1920_1080_60HZ;
                default:
                    return mode;
            }
        }

        XRDeviceProviderHandle handle_ = nullptr;
        int restoreMode_ = 0;
    };

#endif

    // The head without glasses: the left mouse button turns it, W A S D move it.
    class ManualHead: public MouseListener, public KeyListener {

    public:
        void onMouseDown(int button, const Vector2& pos) override {
            if (button == 0) {
                dragging_ = true;
                last_ = pos;
            }
        }

        void onMouseUp(int button, const Vector2&) override {
            if (button == 0) dragging_ = false;
        }

        void onMouseMove(const Vector2& pos) override {
            if (!dragging_) return;
            constexpr float kRadiansPerPixel = 0.003f;
            yaw_ -= (pos.x - last_.x) * kRadiansPerPixel;
            pitch_ = std::clamp(pitch_ - (pos.y - last_.y) * kRadiansPerPixel, -1.4f, 1.4f);
            last_ = pos;
        }

        void onKeyPressed(KeyEvent evt) override { down_.insert(evt.key); }
        void onKeyReleased(KeyEvent evt) override { down_.erase(evt.key); }

        void recentre() {
            position_.set(0, 0, 0);
            yaw_ = pitch_ = 0;
        }

        HeadPose update(float dt) {
            constexpr float kMetresPerSecond = 1.f;
            const Quaternion turn = Quaternion().setFromAxisAngle({0, 1, 0}, yaw_);

            Vector3 step(static_cast<float>(down_.count(Key::D)) - static_cast<float>(down_.count(Key::A)),
                         0,
                         static_cast<float>(down_.count(Key::S)) - static_cast<float>(down_.count(Key::W)));
            position_.add(step.applyQuaternion(turn).multiplyScalar(kMetresPerSecond * dt));

            HeadPose pose;
            pose.position = position_;
            pose.orientation = turn;
            pose.orientation.multiply(Quaternion().setFromAxisAngle({1, 0, 0}, pitch_));
            return pose;
        }

    private:
        bool dragging_ = false;
        Vector2 last_;
        float yaw_ = 0, pitch_ = 0;
        Vector3 position_;
        std::set<Key> down_;
    };

    // The index of a monitor in a side-by-side mode. The glasses take a moment
    // to come back as a wider monitor after a mode switch, hence the wait.
    std::optional<int> findSideBySideMonitor(float waitSeconds) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<float>(waitSeconds);
        while (true) {
            const int count = monitor::monitorCount();
            for (int i = 0; i < count; ++i) {
                const auto size = monitor::monitorSize(i);
                if (size.width() == 3840 && (size.height() == 1200 || size.height() == 1080)) return i;
            }
            if (std::chrono::steady_clock::now() >= deadline) return std::nullopt;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // `lines` draws the ball as a wireframe and the floor grid as lines, which
    // only show in both eyes on OpenGL. Without it the ball is solid and the
    // grid is made of mesh.
    void addObjects(Scene& scene, const std::shared_ptr<Group>& spinning, bool lines) {

        scene.add(HemisphereLight::create(0xffffff, 0x445566, 1.2f));
        auto sun = DirectionalLight::create(0xffffff, 2.f);
        sun->position.set(1, 3, 2);
        scene.add(sun);

        auto knot = Mesh::create(
                TorusKnotGeometry::create(0.1f, 0.03f, 128, 16),
                MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::orange)));
        knot->position.set(0, 0, -1.5f);
        spinning->add(knot);

        auto cyan = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::cyan));
        cyan->wireframe = lines;
        auto ball = Mesh::create(IcosahedronGeometry::create(0.25f, 1), cyan);
        ball->position.set(-0.5f, 0.1f, -2.5f);
        spinning->add(ball);

        auto box = Mesh::create(
                BoxGeometry::create(0.12f, 0.12f, 0.12f),
                MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::limegreen)));
        box->position.set(0.25f, -0.08f, -1.1f);
        spinning->add(box);

        scene.add(spinning);

        // A ring of markers around the origin at eye height, 2 m away, to turn
        // towards.
        const auto marker = SphereGeometry::create(0.05f, 16, 12);
        constexpr int kMarkers = 12;
        for (int i = 0; i < kMarkers; ++i) {
            const float angle = math::TWO_PI * static_cast<float>(i) / kMarkers;
            auto m = Mesh::create(marker, MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(
                                                  Color().setHSL(static_cast<float>(i) / kMarkers, 0.9f, 0.55f))));
            m->position.set(2 * std::sin(angle), 0, -2 * std::cos(angle));
            scene.add(m);
        }

        // The floor grid: 6 m across, a line every half metre, the two centre
        // lines brighter.
        constexpr float kGridSize = 6;
        constexpr int kGridDivisions = 12;
        if (lines) {
            auto grid = GridHelper::create(static_cast<unsigned int>(kGridSize), kGridDivisions, 0x888888, 0x444444);
            grid->position.y = -kFloorBelowEyes;
            scene.add(grid);
        } else {
            // The same grid from 4 mm wide strips of mesh.
            const auto strip = BoxGeometry::create(kGridSize, 0.002f, 0.004f);
            const auto centre = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x888888)));
            const auto other = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x444444)));
            for (int i = 0; i <= kGridDivisions; ++i) {
                const float offset = (static_cast<float>(i) / kGridDivisions - 0.5f) * kGridSize;
                const auto& material = i == kGridDivisions / 2 ? centre : other;

                auto alongX = Mesh::create(strip, material);
                alongX->position.set(0, -kFloorBelowEyes, offset);
                scene.add(alongX);

                auto alongZ = Mesh::create(strip, material);
                alongZ->position.set(offset, -kFloorBelowEyes, 0);
                alongZ->rotation.y = math::PI / 2;
                scene.add(alongZ);
            }
        }
    }

}// namespace

int main(int argc, char** argv) {

    bool logPose = false;
    bool useVulkan = false;
    bool useGlasses = true;
    bool sdkLog = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--log-pose") == 0) logPose = true;
        if (std::strcmp(argv[i], "--vulkan") == 0) useVulkan = true;
        if (std::strcmp(argv[i], "--no-glasses") == 0) useGlasses = false;
        if (std::strcmp(argv[i], "--sdk-log") == 0) sdkLog = true;
    }
#ifndef THREEPP_WITH_VULKAN
    if (useVulkan) {
        std::cerr << "Built without the Vulkan renderer (THREEPP_WITH_VULKAN): using OpenGL\n";
        useVulkan = false;
    }
#endif

    HeadPose pose;
    ManualHead manual;

#ifdef THREEPP_EXAMPLE_WITH_VITURE
    std::optional<LumaUltra> glasses;
    if (useGlasses) {
        glasses.emplace(sdkLog ? 3 : 1);
        if (glasses->connected()) {
            std::cout << "VITURE: Luma Ultra found, head tracking on" << std::endl;
            if (!glasses->enterSideBySide()) std::cerr << "VITURE: the glasses refused the 3840x1200 mode" << std::endl;
        } else {
            std::cout << "VITURE: no Luma Ultra started, the mouse and W A S D move the head" << std::endl;
            glasses.reset();
        }
    }
    const bool tracked = glasses.has_value();
#else
    (void) useGlasses;
    (void) sdkLog;
    std::cout << "Built without VITURE's SDK (VITURE_SDK_DIR): the mouse and W A S D move the head" << std::endl;
    const bool tracked = false;
#endif

    const auto display = findSideBySideMonitor(tracked ? 5.f : 0.f);

    Canvas::Parameters parameters;
    parameters.title("Side-by-side stereo").antialiasing(4).vsync(true);
    if (display) {
        // On the glasses' monitor, so that fullscreen below covers that one.
        const auto corner = monitor::monitorPosition(*display);
        parameters.position(corner.first + 100, corner.second + 100).size(1600, 500);
    } else {
        parameters.size(1600, 500);
    }
    Canvas canvas(parameters);

    std::unique_ptr<Renderer> renderer;
#ifdef THREEPP_WITH_VULKAN
    VulkanRenderer* vulkan = nullptr;
    if (useVulkan) {
        auto vk = std::make_unique<VulkanRenderer>(canvas);
        vulkan = vk.get();
        renderer = std::move(vk);
    }
#endif
    if (!renderer) renderer = std::make_unique<GLRenderer>(canvas);
    renderer->setScissorTest(true);
    renderer->setClearColor(Color::black);

    if (display) canvas.setFullscreen(true);

    Scene scene;
    scene.background = Color(Color::black);
    const auto spinning = Group::create();
    addObjects(scene, spinning, !useVulkan);

    // The head. Its aspect is one eye's, half the window; StereoCamera derives
    // the two eyes from it.
    PerspectiveCamera head(kFovVertical, 1, 0.05f, 100);
    head.focus = kConvergence;

    StereoCamera stereo;
    stereo.eyeSep = kEyeSeparation;

    canvas.onWindowResize([&](WindowSize size) {
        renderer->setSize(size);
    });

    KeyAdapter keys(KeyAdapter::KEY_PRESSED | KeyAdapter::KEY_REPEAT, [&](KeyEvent evt) {
        switch (evt.key) {
            case Key::R:
#ifdef THREEPP_EXAMPLE_WITH_VITURE
                if (tracked) glasses->recentre();
#endif
                manual.recentre();
                return;
            case Key::F:
                canvas.setFullscreen(!canvas.isFullscreen());
                return;
            case Key::LEFT_BRACKET:
                stereo.eyeSep = std::max(0.f, stereo.eyeSep - 0.001f);
                break;
            case Key::RIGHT_BRACKET:
                stereo.eyeSep += 0.001f;
                break;
            case Key::MINUS:
                head.focus = std::max(0.25f, head.focus / 1.1f);
                break;
            case Key::EQUAL:
                head.focus *= 1.1f;
                break;
            case Key::COMMA:
                head.fov = std::max(5.f, head.fov - 0.5f);
                break;
            case Key::PERIOD:
                head.fov = std::min(120.f, head.fov + 0.5f);
                break;
            default:
                return;
        }
        std::cout << "eye separation " << stereo.eyeSep * 1000 << " mm, convergence " << head.focus
                  << " m, vertical field of view " << head.fov << " degrees\n";
    });
    canvas.addKeyListener(keys);
    if (!tracked) {
        canvas.addKeyListener(manual);
        canvas.addMouseListener(manual);
    }

    Clock clock;
    bool stable = true;
    float sinceLog = 0;

#ifdef THREEPP_WITH_VULKAN
    // The right eye's added view. It has a fixed size and is costly to make
    // (addView drains the device), so it is made once the window has kept one
    // size for kHeldFrames frames, and again after a later resize.
    constexpr int kHeldFrames = 10;
    std::uint32_t rightView = 0;
    int viewWidth = 0, viewHeight = 0;
    int wantedWidth = 0, wantedHeight = 0, held = 0;
#endif

    canvas.animate([&] {
        const float dt = clock.getDelta();

        for (const auto& child : spinning->children) {
            child->rotation.y += 0.5f * dt;
            child->rotation.x += 0.2f * dt;
        }

#ifdef THREEPP_EXAMPLE_WITH_VITURE
        if (tracked) {
            // One frame ahead: the picture rendered now is on the displays
            // about that much later.
            bool nowStable = stable;
            glasses->pose(std::max(dt, 1 / 120.f), pose, nowStable);
            if (nowStable != stable) {
                stable = nowStable;
                std::cout << "VITURE: tracking " << (stable ? "stable" : "unstable") << std::endl;
            }
        } else
#endif
        {
            pose = manual.update(dt);
        }

        sinceLog += dt;
        if (logPose && sinceLog >= 1) {
            sinceLog = 0;
            const auto& p = pose.position;
            const auto& q = pose.orientation;
            std::cout << "pose: position " << p.x << " " << p.y << " " << p.z
                      << "  quaternion (x y z w) " << q.x << " " << q.y << " " << q.z << " " << q.w
                      << (stable ? "" : "  unstable") << std::endl;
        }

        const auto size = canvas.size();
        const int leftWidth = size.width() / 2;
        const int rightWidth = size.width() - leftWidth;
        const int height = size.height();

        head.position.copy(pose.position);
        head.quaternion.copy(pose.orientation);
        head.aspect = static_cast<float>(leftWidth) / static_cast<float>(std::max(height, 1));
        head.updateProjectionMatrix();
        head.updateMatrixWorld();

        stereo.update(head);

        renderer->setViewport(0, 0, leftWidth, height);
        renderer->setScissor(0, 0, leftWidth, height);
        renderer->render(scene, stereo.cameraL);

#ifdef THREEPP_WITH_VULKAN
        if (vulkan) {
            if (rightWidth != wantedWidth || height != wantedHeight) {
                wantedWidth = rightWidth;
                wantedHeight = height;
                held = 0;
            } else if (held < kHeldFrames) {
                ++held;
            }
            const bool fits = rightView != 0 && viewWidth == rightWidth && viewHeight == height;
            if (!fits && held >= kHeldFrames) {
                if (rightView != 0) vulkan->removeView(rightView);
                // 0 until the renderer has rendered once; tried again next frame.
                rightView = vulkan->addView(stereo.cameraR, rightWidth, height);
                if (rightView != 0) {
                    viewWidth = rightWidth;
                    viewHeight = height;
                    vulkan->setViewDisplayRect(rightView, leftWidth, 0, rightWidth, height);
                }
            }
            return;
        }
#endif

        renderer->setViewport(leftWidth, 0, rightWidth, height);
        renderer->setScissor(leftWidth, 0, rightWidth, height);
        renderer->render(scene, stereo.cameraR);
    });

#ifdef THREEPP_EXAMPLE_WITH_VITURE
    // The glasses go back to their earlier display mode here, while the window
    // and the renderer still exist.
    glasses.reset();
#endif
}
