// ArduPilot SITL demo — threepp as the physics backend and visualizer.
//
// ArduCopter SITL (typically in WSL2, `sim_vehicle.py -v ArduCopter
// -f JSON:<this-machine>`) sends servo PWM to UDP :9002; we step a PhysX quad
// one substep per packet and reply with the vehicle state that drives SITL's
// own clock (lock-step). Full runbook: doc/ardupilot_sitl.md.
//
//   ardupilot_sitl [--port N]     interactive demo (default port 9002)
//   ardupilot_sitl --selftest     headless fake-SITL loopback check, exit 0/1
//   ardupilot_sitl --brownout     scripted landing loop, no SITL needed. The
//                                 rotor-wash dust is NOT this mode's feature —
//                                 it runs in every mode on Vulkan, driven off
//                                 the same quad state an ArduPilot flight has.
//   ardupilot_sitl --seq DIR [--start T] [--seqframes N] [--warm N]
//                                 headless deterministic brownout capture
//   ardupilot_sitl --rec DIR [--recevery N] [--recframes M]
//                                 film the LIVE SITL flight: every Nth drawn
//                                 frame to DIR/f%05d.png, starting when the
//                                 vehicle first lifts off. Excludes --seq.
//   ardupilot_sitl --proximity [--proximity-port N]
//                                 360-degree horizontal lidar fan -> MAVLink
//                                 OBSTACLE_DISTANCE at 10 Hz, plus the spoke
//                                 overlay. Combines with any mode above; the
//                                 scan runs whether or not SITL ever answers.
//                                 SITL side: --serial2=udpclient:<host>:14560
//   ardupilot_sitl --vulkan       pick the Vulkan backend without the prompt
//
// The render loop and the physics never race: each animate frame drains the
// socket under a time budget (SITL waits on our reply — that's the protocol —
// so rendering simply pauses the autopilot for a frame's worth of wall time),
// then draws the latest state.

// Raw UDP for the demo's own side channels — the selftest's fake SITL and the
// waypoint-marker feed. The library bridge keeps its socket behind a pimpl, so
// the platform headers are this file's own business.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX// windows.h min/max macros break PhysX and <algorithm>
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
// windef.h's 16-bit relics; threepp cameras have members named near/far.
#undef near
#undef far
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "renderer_factory.hpp"

#include "threepp/extras/uav/DownwashEffect.hpp"
#include "threepp/extras/uav/DroneVisual.hpp"
#include "threepp/extras/uav/FrameConv.hpp"
#include "threepp/extras/uav/MavlinkOut.hpp"
#include "threepp/extras/uav/ProximityScan.hpp"
#include "threepp/extras/uav/QuadSim.hpp"
#include "threepp/extras/uav/SitlBridge.hpp"

#include "threepp/helpers/LidarSensor.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/RockGeometry.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/extras/vegetation/GrassTiles.hpp"
#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#ifdef THREEPP_WITH_VULKAN
#include "threepp/objects/Ocean.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#endif
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/scenes/FogExp2.hpp"

#include <cstdlib>
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <string>

using namespace threepp;

namespace {

#ifdef _WIN32
    using socket_t = SOCKET;
#else
    using socket_t = int;
    void closesocket(socket_t s) { ::close(s); }
#endif

    // Winsock wants a one-time init before the raw side-channel sockets;
    // WSAStartup is refcounted, so calling it here as well as inside the
    // library bridge is harmless.
    void ensureSockets() {
#ifdef _WIN32
        static const int once = [] {
            WSADATA data;
            return WSAStartup(MAKEWORD(2, 2), &data);
        }();
        (void) once;
#endif
    }

    double nowSec() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }

    // ── Scripted brownout flight (closed form in sim time) ──────────────────
    // A 26 s cycle: climb, hover, DECELERATING descent into touchdown (the
    // final metres are slow, which is where ground effect makes the dust), a
    // motors-cut pause while the cloud drifts off, then a spool-up on the pad
    // that blasts what settled — and around again. Altitude target in metres;
    // < 0 means the throttle is CUT this instant.
    float scriptAlt(float c) {
        if (c < 4.f) return 12.f * math::smoothstep(0.f, 4.f, c);
        if (c < 8.f) return 12.f;
        if (c < 19.f) {
            const float u = (c - 8.f) / 11.f;
            return 12.f * (1.f - u) * (1.f - u);
        }
        return -1.f;// cut (the pad spool below 24 s is handled by the caller)
    }

    constexpr float kScriptPeriod = 26.f;

    // The noon_grass HDRI is a ground-level park photo: perfect LIGHT, wrong
    // HORIZON — houses and 100 m photographic trees ring the band just above
    // eye level, and the deferred renderer's one env image serves as both IBL
    // and sky, so they would loom behind the valley. Fade the band (elevation
    // −4°..+8°, feathered) toward each column's own sky sampled at +12°: the
    // sun, the upper sky and the lawn's green ground bounce survive; the
    // suburbs dissolve into horizon haze.
    std::shared_ptr<Texture> fadeHorizonBand(const std::shared_ptr<Texture>& src) {
        const Image& img = src->image();
        if (!img.isFloat() || img.channels() != 4) return src;
        const int W = static_cast<int>(img.width());
        const int H = static_cast<int>(img.height());
        std::vector<float> out = img.data<float>();// mutable copy
        // Row iy carries elevation e = ((iy + 0.5)/H − 0.5)·180 (row 0 = down).
        // The reference row sits at +30° — ABOVE the tallest photo trees (a
        // +12° reference sat IN them, and per-column blending smeared their
        // colours into vertical streaks) — and is azimuth-blurred wide, so the
        // band fades into smooth sky haze while keeping the broad bright-side
        // gradient toward the sun.
        const int refRow = std::clamp(
                static_cast<int>((30.f / 180.f + 0.5f) * H - 0.5f), 0, H - 1);
        std::vector<float> ref(static_cast<std::size_t>(W) * 3);
        {
            const int blur = std::max(1, W / 16);// ±blur box, wraps in azimuth
            for (int ix = 0; ix < W; ++ix) {
                float acc[3] = {};
                for (int k = -blur; k <= blur; ++k) {
                    const int x = ((ix + k) % W + W) % W;
                    const std::size_t r = (static_cast<std::size_t>(refRow) * W + x) * 4;
                    for (int c = 0; c < 3; ++c) acc[c] += out[r + c];
                }
                for (int c = 0; c < 3; ++c)
                    ref[static_cast<std::size_t>(ix) * 3 + c] = acc[c] / (2 * blur + 1);
            }
        }
        for (int iy = 0; iy < H; ++iy) {
            const float e = ((iy + 0.5f) / H - 0.5f) * 180.f;
            const float w = math::smoothstep(-7.f, -4.f, e) *
                            (1.f - math::smoothstep(24.f, 29.f, e));
            if (w <= 0.f) continue;
            for (int ix = 0; ix < W; ++ix) {
                const std::size_t o = (static_cast<std::size_t>(iy) * W + ix) * 4;
                for (int c = 0; c < 3; ++c) {
                    const float r = ref[static_cast<std::size_t>(ix) * 3 + c];
                    out[o + c] += (r - out[o + c]) * w;
                }
            }
        }
        Image outImg{ImageData(std::move(out)), static_cast<unsigned int>(W),
                     static_cast<unsigned int>(H)};
        auto tex = Texture::create(outImg);
        tex->name = src->name + "_horizonfade";
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->wrapS = src->wrapS;
        tex->wrapT = src->wrapT;
        tex->needsUpdate();
        return tex;
    }

    // --- selftest ----------------------------------------------------------

    // Pull "key":[a,b,c] out of the JSON reply. Enough parser for a selftest.
    bool jsonVec3(const std::string& json, const char* key, double out[3]) {
        const auto pos = json.find("\"" + std::string(key) + "\":[");
        if (pos == std::string::npos) return false;
        return std::sscanf(json.c_str() + pos + std::strlen(key) + 4, "%lf,%lf,%lf",
                           &out[0], &out[1], &out[2]) == 3;
    }

    bool jsonNumber(const std::string& json, const char* key, double& out) {
        const auto pos = json.find("\"" + std::string(key) + "\":");
        if (pos == std::string::npos) return false;
        return std::sscanf(json.c_str() + pos + std::strlen(key) + 3, "%lf", &out) == 1;
    }

    // In-process fake SITL: drives the real bridge + PhysX sim over loopback
    // UDP, asserting on the JSON replies. No ArduPilot, no window, CI-safe.
    int runSelftest() {
        constexpr std::uint16_t rate = 400;
        int failures = 0;
        const auto check = [&](bool ok, const char* what) {
            std::printf("[selftest] %-52s %s\n", what, ok ? "ok" : "FAIL");
            if (!ok) ++failures;
        };

        uav::SitlBridge bridge(0);// ephemeral port; never clashes with a live 9002
        check(bridge.valid(), "bridge binds an ephemeral UDP port");

        uav::DroneVisual visual;
        visual.root()->position.y = uav::DroneVisual::hullY / 2.f;
        uav::QuadSim quad(rate, *visual.root(), nullptr);// flat plane ground

        // Fake SITL's own socket.
        ensureSockets();
        socket_t tx = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
#ifdef _WIN32
        u_long nb = 1;
        ioctlsocket(tx, FIONBIO, &nb);
#else
        fcntl(tx, F_SETFL, fcntl(tx, F_GETFL, 0) | O_NONBLOCK);
#endif
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_port = htons(bridge.port());
        inet_pton(AF_INET, "127.0.0.1", &to.sin_addr);

        std::uint32_t frameCount = 0;
        std::string lastReply;

        // One lock-step exchange: send pwm, service the bridge, read the reply.
        const auto exchange = [&](std::uint16_t pwm0, std::uint16_t pwm1,
                                  std::uint16_t pwm2, std::uint16_t pwm3,
                                  int channels = 16) -> bool {
            uav::ServoInput in{};
            in.frameRate = rate;
            in.frameCount = ++frameCount;
            in.channels = channels;
            in.pwm[0] = pwm0;
            in.pwm[1] = pwm1;
            in.pwm[2] = pwm2;
            in.pwm[3] = pwm3;
            std::uint8_t buf[128];
            const int n = uav::SitlBridge::encode(in, buf);
            ::sendto(tx, reinterpret_cast<const char*>(buf), n, 0,
                     reinterpret_cast<const sockaddr*>(&to), sizeof to);

            const double deadline = nowSec() + 1.0;
            uav::ServoInput servo{};
            while (nowSec() < deadline) {
                const auto ev = bridge.poll(servo);
                if (ev == uav::SitlBridge::Event::None) continue;
                if (ev == uav::SitlBridge::Event::Reset) quad.reset();
                uav::FdmState state{};
                quad.step(servo, state);
                bridge.sendState(state);

                char rx[512];
                sockaddr_in from{};
#ifdef _WIN32
                int fl = sizeof from;
#else
                socklen_t fl = sizeof from;
#endif
                const double rxDeadline = nowSec() + 1.0;
                while (nowSec() < rxDeadline) {
                    const auto m = ::recvfrom(tx, rx, sizeof rx - 1, 0,
                                              reinterpret_cast<sockaddr*>(&from), &fl);
                    if (m > 0) {
                        rx[m] = '\0';
                        lastReply = rx;
                        return true;
                    }
                }
                return false;
            }
            return false;
        };

        const auto runSeconds = [&](double seconds, std::uint16_t p0, std::uint16_t p1,
                                    std::uint16_t p2, std::uint16_t p3) {
            const int frames = static_cast<int>(seconds * rate);
            for (int i = 0; i < frames; ++i) {
                if (!exchange(p0, p1, p2, p3)) return false;
            }
            return true;
        };

        // 1: idle on the ground.
        check(runSeconds(1.0, 1000, 1000, 1000, 1000), "1 s idle exchange completes");
        double pos[3] = {}, accel[3] = {}, ts = 0;
        check(jsonVec3(lastReply, "position", pos), "reply carries position");
        check(std::abs(pos[0]) < 0.01 && std::abs(pos[1]) < 0.01 && std::abs(pos[2]) < 0.02,
              "drone rests at origin");
        check(jsonVec3(lastReply, "accel_body", accel), "reply carries accel_body");
        check(std::abs(accel[2] + 9.81) < 0.15, "at-rest specific force ~ (0,0,-9.81)");
        check(jsonNumber(lastReply, "timestamp", ts) && std::abs(ts - 1.0) < 1e-6,
              "timestamp advances by exactly 1/frame_rate");

        // 2: full throttle climbs.
        check(runSeconds(2.0, 2000, 2000, 2000, 2000), "2 s full-throttle exchange completes");
        check(jsonVec3(lastReply, "position", pos), "climb reply carries position");
        check(-pos[2] > 5.0, "full throttle climbs above 5 m");
        double vel[3] = {};
        check(jsonVec3(lastReply, "velocity", vel) && -vel[2] > 0.5, "still climbing");

        // 3: differential thrust signs. Hover-ish base keeps it airborne;
        // left motors (M2 back-left pwm[1], M3 front-left pwm[2]) high => roll
        // right => positive FRD roll rate.
        double gyro[3] = {};
        check(exchange(1500, 1700, 1700, 1500) && runSeconds(0.1, 1500, 1700, 1700, 1500),
              "roll-input exchange completes");
        check(jsonVec3(lastReply, "gyro", gyro) && gyro[0] > 0.05, "left motors high => rolls right");

        quad.reset();
        frameCount += 1;// keep counter monotonic; reset() only re-homes physics
        // Front motors (M1 pwm[0], M3 pwm[2]) high => nose up => positive pitch rate.
        check(runSeconds(0.1, 1700, 1500, 1700, 1500), "pitch-input exchange completes");
        check(jsonVec3(lastReply, "gyro", gyro) && gyro[1] > 0.05, "front motors high => pitches up");

        quad.reset();
        // CCW pair (M1 pwm[0], M2 pwm[1]) high => positive yaw rate.
        check(runSeconds(0.2, 1800, 1800, 1400, 1400), "yaw-input exchange completes");
        check(jsonVec3(lastReply, "gyro", gyro) && gyro[2] > 0.05, "CCW motors high => yaws right");

        // 4: SITL restart detection: frame counter jumps backwards.
        frameCount = 0;
        uav::ServoInput servo{};
        {
            uav::ServoInput in{};
            in.frameRate = rate;
            in.frameCount = 1;
            in.channels = 16;
            for (auto& p : in.pwm) p = 1000;
            std::uint8_t buf[128];
            const int n = uav::SitlBridge::encode(in, buf);
            ::sendto(tx, reinterpret_cast<const char*>(buf), n, 0,
                     reinterpret_cast<const sockaddr*>(&to), sizeof to);
            const double deadline = nowSec() + 1.0;
            auto ev = uav::SitlBridge::Event::None;
            while (nowSec() < deadline &&
                   (ev = bridge.poll(servo)) == uav::SitlBridge::Event::None) {}
            check(ev == uav::SitlBridge::Event::Reset, "frame_count rollback detected as restart");
            quad.reset();
            uav::FdmState state{};
            quad.step(servo, state);
            bridge.sendState(state);
            check(std::abs(state.positionNed[2]) < 0.02, "reset re-homes the vehicle");
            frameCount = 1;
        }

        // 5: 32-channel packet accepted.
        check(exchange(1000, 1000, 1000, 1000, 32), "32-channel packet (magic 29569) accepted");

        closesocket(tx);
        std::printf("[selftest] %s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }

}// namespace

int main(int argc, char** argv) {

    std::uint16_t port = 9002;
    bool brownout = false;   // scripted landing loop, no SITL required
    std::string seqDir;      // deterministic frame capture (implies --brownout)
    int seqFrames = 600, seqWarm = 30;
    float seqStart = 12.f;   // sim seconds to fast-forward before capturing
    bool proximity = false;  // horizontal obstacle fan -> OBSTACLE_DISTANCE
    std::uint16_t proxPort = 14560;// where SITL's udpclient dials in
    bool forceVulkan = false;// skip the interactive backend prompt
    std::string recDir;      // live capture of a SITL flight (see below)
    int recEvery = 6, recFrames = 1400;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") return runSelftest();
        if (a == "--vulkan") forceVulkan = true;
        if (a == "--port" && i + 1 < argc) port = static_cast<std::uint16_t>(std::atoi(argv[++i]));
        if (a == "--proximity") proximity = true;
        if (a == "--proximity-port" && i + 1 < argc) {
            proxPort = static_cast<std::uint16_t>(std::atoi(argv[++i]));
            proximity = true;
        }
        if (a == "--brownout") brownout = true;
        if (a == "--seq" && i + 1 < argc) { seqDir = argv[++i]; brownout = true; }
        if (a == "--seqframes" && i + 1 < argc) seqFrames = std::atoi(argv[++i]);
        if (a == "--warm" && i + 1 < argc) seqWarm = std::atoi(argv[++i]);
        if (a == "--start" && i + 1 < argc) seqStart = static_cast<float>(std::atof(argv[++i]));
        if (a == "--rec" && i + 1 < argc) recDir = argv[++i];
        if (a == "--recevery" && i + 1 < argc) recEvery = std::max(1, std::atoi(argv[++i]));
        if (a == "--recframes" && i + 1 < argc) recFrames = std::max(1, std::atoi(argv[++i]));
    }
    // Two capture modes with two different flights behind them: --seq runs the
    // scripted brownout headless at a fixed dt, --rec films whatever ArduPilot
    // is flying live. Accepting both would silently give the caller the first
    // one, and --seq returns before the render loop the recorder lives in.
    if (!recDir.empty() && !seqDir.empty()) {
        std::fprintf(stderr, "--rec and --seq are different captures; pick one\n");
        return 1;
    }

    Canvas canvas("threepp - ArduPilot SITL",
                  {{"aa", 4},
                   {"size", std::pair<int, int>{1280, 720}},
                   {"headless", !seqDir.empty()},
                   {"vsync", seqDir.empty()}});
    // A capture cannot answer the factory's interactive renderer prompt, and
    // only the Vulkan backend renders the dust anyway. --vulkan is the same
    // escape for a scripted run that DOES have a window: piped stdin answers
    // the prompt with EOF, which the factory reads as "GL", and the traced
    // lidar the proximity fan wants exists only on Vulkan.
    auto renderer = (seqDir.empty() && !forceVulkan)
                            ? createRenderer(canvas)
                            : createRenderer(canvas, GraphicsAPI::Vulkan);
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    Scene scene;

    // Sky + sun — spot_slam's atmosphere. On Vulkan: an HDR equirect
    // environment drives image-based lighting and serves as the sky backdrop
    // (the same noon_grass HDRI spot_slam caches; THREEPP_HDRI overrides the
    // path), plus exponential-fog aerial perspective below. The Preetham dome
    // stays the GL fallback (its ShaderMaterial can't shade on the deferred
    // renderer), and a flat sky covers a Vulkan machine with no cached HDRI.
    const bool vulkan = canvas.graphicsApi() == GraphicsAPI::Vulkan;
    float hemiIntensity = 0.9f;
    bool hdrSky = false;
    if (!vulkan) {
        auto sky = Sky::create();
        sky->scale.setScalar(9000);
        auto& u = sky->materialAs<ShaderMaterial>()->uniforms;
        u.at("turbidity").value<float>() = 6;
        u.at("rayleigh").value<float>() = 1.2f;
        u.at("mieCoefficient").value<float>() = 0.006f;
        u.at("mieDirectionalG").value<float>() = 0.8f;
        u.at("sunPosition").value<Vector3>().setFromSphericalCoords(
                1.f, math::degToRad(90 - 28), math::degToRad(135));
        scene.add(sky);
    } else {
        std::string hdrPath;
        if (const char* p = std::getenv("THREEPP_HDRI")) {
            hdrPath = p;
        } else if (const char* home = std::getenv("USERPROFILE");
                   home || (home = std::getenv("HOME"))) {
            hdrPath = std::string(home) + "/.cache/threepp/hdri/noon_grass_2k.hdr";
        }
        RGBELoader rgbe;
        std::shared_ptr<Texture> env;
        if (!hdrPath.empty()) env = rgbe.load(hdrPath);
        if (env) {
            // The deferred renderer's one env image is BOTH the IBL and the
            // sky, so the photo's horizon band is faded out first (see
            // fadeHorizonBand) — sun, sky and ground bounce stay, the
            // photographic suburbs go.
            env = fadeHorizonBand(env);
            scene.environment = env;
            scene.background = env;
            hemiIntensity = 0.25f;// the HDR provides the ambient fill
            hdrSky = true;
        } else {
            std::fprintf(stderr, "[sky] no HDRI at %s — flat-sky fallback\n",
                         hdrPath.c_str());
            scene.background = Color(0.55f, 0.72f, 0.92f);
        }
    }
    // Volumetric exponential fog: on Vulkan this is THE single knob — froxel
    // sun shafts and aerial glow follow the fog medium (spot_slam's look).
    // Density scaled to this valley's 250 m vistas where spot_slam's 0.02
    // suited a 25 m robot world; 0.0055 read as overcast soup at 10 m — this
    // keeps the near action crisp and hazes the treeline (~78% at 100 m).
    scene.fog = FogExp2(0x8ab4d4, 0.0025f);

    auto sun = DirectionalLight::create(Color(0xfff8e0), 2.8f);
    Vector3 sunDir;// unit vector toward the sun; the light rig follows the drone
    auto sunTarget = Object3D::create();
    {
        const float phi = math::degToRad(90 - 28);// elevation 28 deg, matches the sky dome
        const float theta = math::degToRad(135);
        sunDir.setFromSphericalCoords(1.f, phi, theta);
        sun->position.copy(sunDir).multiplyScalar(300.f);
        sun->castShadow = true;
        // A tight frustum that tracks the drone (see the animate loop) buys far
        // more texel density than a big static box ever could.
        sun->shadow->mapSize.set(4096, 4096);
        sun->shadow->bias = -0.0004f;
        if (auto* cam = sun->shadow->camera->as<OrthographicCamera>()) {
            cam->left = cam->bottom = -25.f;
            cam->right = cam->top = 25.f;
            cam->farPlane = 700.f;
        }
        scene.add(sunTarget);
        sun->setTarget(*sunTarget);
        scene.add(sun);
    }
    // Sky/ground hemisphere fill (spot_slam's): a touch beside the HDR's own
    // ambient, the main fill on the fallbacks.
    scene.add(HemisphereLight::create(Color(0xd0e8ff), Color(0x3a4820), hemiIntensity));

#ifdef THREEPP_WITH_VULKAN
    if (auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get())) {
        // Forward scatter: the sun-ward glow and god-ray shafts through the
        // trees live on the phase anisotropy (spot_slam sets the same).
        vk->setFogAnisotropy(0.6f);
        // Volumetric cloud deck over the flat-blue backdrop (the HDRI serves
        // lighting only — see the environment note above).
        {
            VulkanRenderer::CloudSettings clouds;
            clouds.coverage = 0.45f;
            clouds.density = 1.0f;
            clouds.bottomY = 140.f;
            clouds.topY = 520.f;
            clouds.wind = Vector3(10.f, 0.f, 4.f);
            clouds.evolveSpeed = 1.0f;
            vk->setClouds(clouds);
        }
    }
#endif

    // Rolling-hills terrain, gentle at the center where the drone homes.
    // Hydraulic erosion carves drainage into the slopes (the ~1 s pass); all
    // home-height queries sample the eroded GEOMETRY, never heightAt(), which
    // knows nothing of erosion.
    terrain::TerrainParams tp;
    tp.worldSize = 1500.f;
    tp.resolution = 512;
    tp.amplitude = 65.f;
    tp.featureScale = 420.f;
    tp.octaves = 6;
    tp.falloff = terrain::Falloff::Radial;
    tp.erosion = terrain::ErosionType::Hydraulic;
    // The radial island puts HOME on the height field's top, and the default
    // snowline (0.5) is below it — so the whole landing area baked as SNOW and
    // the scene read as arctic (and tan rotor-wash dust vanished, white on
    // white). No snow anywhere: this is a summer meadow valley.
    tp.snowLine = 2.f;
    terrain::TerrainGenerator gen(1337);
    gen.buildField(tp);
    gen.erode(tp);

    auto terrainMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(Color::white).roughness(0.95f).metalness(0.f));
    auto terrainTex = DataTexture::create(ImageData{gen.bakeSplatColors(tp)},
                                          static_cast<unsigned int>(gen.dim()),
                                          static_cast<unsigned int>(gen.dim()));
    terrainTex->colorSpace = ColorSpace::sRGB;
    terrainTex->magFilter = Filter::Linear;
    terrainTex->minFilter = Filter::Linear;
    terrainMat->map = terrainTex;
    {
        // Cm-scale tiled detail (albedo breakup + normal relief + roughness),
        // consumed by the Vulkan deferred renderer; GL ignores these slots.
        const terrain::DetailMaps dm = terrain::makeDetailMaps({});
        terrainMat->detailMap = dm.albedo;
        terrainMat->detailNormalMap = dm.normalRough;
        terrainMat->detailRepeat = 0.5f;// one repeat per 2 m
        terrainMat->detailStrength = 0.5f;
        terrainMat->detailNormalScale = 1.2f;
        terrainMat->detailRoughStrength = 0.5f;
    }
    auto terrainGeo = gen.makeGeometry(tp);
    // Home ground height = the eroded mesh's center vertex (the grid is
    // (res+1)^2 with odd side, so an exact (0,0) vertex exists).
    float h0 = 0.f;
    {
        auto* pos = terrainGeo->getAttribute<float>("position");
        auto& a = pos->array();
        float best = std::numeric_limits<float>::max();
        for (int i = 0; i < pos->count(); ++i) {
            const float d = std::abs(a[i * 3 + 0]) + std::abs(a[i * 3 + 2]);
            if (d < best) {
                best = d;
                h0 = a[i * 3 + 1];
            }
        }
    }
    // Level a helipad apron around home: within 4 m the ground is exactly h0
    // (so the pad never pokes through a rising slope), blending back to the raw
    // terrain by 16 m. That is the ONLY place this demo sculpts the ground —
    // the physics trimesh is built from this same geometry, so contacts match
    // the picture.
    {
        auto* pos = terrainGeo->getAttribute<float>("position");
        auto& a = pos->array();
        for (int i = 0; i < pos->count(); ++i) {
            const float x = a[i * 3 + 0], z = a[i * 3 + 2];
            const float r = std::sqrt(x * x + z * z);
            if (r < 16.f) {
                const float t = math::smoothstep(4.f, 16.f, r);
                a[i * 3 + 1] = h0 + (a[i * 3 + 1] - h0) * t;
            }
        }
        pos->needsUpdate();
        terrainGeo->computeVertexNormals();
        terrainGeo->computeBoundingBox();
        terrainGeo->computeBoundingSphere();
    }

    // Pond site: FOUND in the eroded terrain, not carved into it. Hydraulic
    // erosion leaves real hollows behind, and spot_slam simply goes looking for
    // one — so this demo does too, asking the question the flat square sheet
    // actually poses: for a candidate centre and half-width, how high can the
    // water stand before it runs off the edge of its own mesh? That ceiling is
    // the lowest point of the square's rim (the spill point), which buries
    // every edge AND corner under shore by construction — the thing four rounds
    // of berm-carving were trying to force. Score is open-water AREA, and the
    // sizes are tried largest-first: the pond is as big as the valley will hold.
    float pondX = 0.f, pondZ = 0.f, pondR = 0.f, pondWaterRaw = 0.f;
    {
        const int dim = tp.resolution + 1;
        const float half = tp.worldSize * 0.5f;
        const float step = tp.worldSize / static_cast<float>(tp.resolution);
        const auto& a = terrainGeo->getAttribute<float>("position")->array();
        const auto hAt = [&](int ix, int iz) {
            return a[(static_cast<std::size_t>(iz) * dim + ix) * 3 + 1];
        };
        // Search an annulus: clear of the pad apron and the brownout site, close
        // enough that the flight and the capture tripod actually see it.
        constexpr float kNear = 45.f, kFar = 350.f;
        constexpr float kFreeboard = 0.25f;// wave crests + z-fight margin
        const int lo = static_cast<int>(std::lround((-kFar + half) / step));
        const int hi = static_cast<int>(std::lround((kFar + half) / step));
        float bestArea = 0.f;
        // Half-widths ~70 m down to ~12 m. Measured on this terrain (seed 1337,
        // 1500 m island, hydraulic erosion): NOTHING within 175 m of home holds
        // more than 0.6 m of water — fluvial erosion carves open drainage and
        // its droplets FILL hollows, so a young eroded massif has no ponds near
        // its crown. The real water is down the flank, and it is a proper tarn.
        for (int rad : {24, 20, 16, 13, 10, 8, 6, 5, 4}) {
            const float W = rad * step;
            const int stride = std::max(1, rad / 3);
            for (int cz = lo; cz <= hi; cz += stride) {
                for (int cx = lo; cx <= hi; cx += stride) {
                    if (cx - rad < 0 || cx + rad >= dim || cz - rad < 0 || cz + rad >= dim) continue;
                    const float x = cx * step - half, z = cz * step - half;
                    const float r = std::hypot(x, z);
                    if (r < kNear || r > kFar) continue;
                    // The spill point: water above this hangs off the rim.
                    float rim = std::numeric_limits<float>::max();
                    for (int k = -rad; k <= rad; ++k) {
                        rim = std::min(rim, std::min(std::min(hAt(cx + k, cz - rad),
                                                              hAt(cx + k, cz + rad)),
                                                     std::min(hAt(cx - rad, cz + k),
                                                              hAt(cx + rad, cz + k))));
                    }
                    const float wl = rim - kFreeboard;
                    float area = 0.f, vol = 0.f, deepest = 0.f;
                    for (int jz = -rad + 1; jz < rad; ++jz) {
                        for (int jx = -rad + 1; jx < rad; ++jx) {
                            const float d = wl - hAt(cx + jx, cz + jz);
                            if (d <= 0.f) continue;
                            area += step * step;
                            vol += d * step * step;
                            deepest = std::max(deepest, d);
                        }
                    }
                    // A pond, not a wet patch: deep somewhere, wide enough to
                    // read, and not a film spread over a whole hillside.
                    // A tarn, not a wet patch: deep somewhere, wide enough to
                    // read from the air, and not a film smeared over a hillside.
                    if (deepest < 1.2f || area < 400.f || vol < 0.35f * area) continue;
                    if (area > bestArea) {
                        bestArea = area;
                        pondX = x;
                        pondZ = z;
                        pondR = W;
                        pondWaterRaw = wl;
                    }
                }
            }
        }
        if (pondR > 0.f) {
            std::fprintf(stderr, "[pond] basin (%.0f, %.0f)  half-width %.1f m  "
                                 "water %.2f m  open water %.0f m2\n",
                         pondX, pondZ, pondR, pondWaterRaw - h0, bestArea);
        } else {
            std::fprintf(stderr, "[pond] no basin holds water in this terrain — dry valley\n");
        }
    }
    auto terrainMesh = Mesh::create(terrainGeo, terrainMat);
    terrainMesh->receiveShadow = true;
    // Home ground exactly at y=0 so NED altitude 0 = spawn ground level.
    terrainMesh->position.y = -h0;
    scene.add(terrainMesh);

    // Exact-grid height sampler over the FINAL (aproned) geometry; world
    // y = sample - h0 because the mesh is offset so home ground sits at 0.
    const auto groundY = [&, dim = tp.resolution + 1,
                          half = tp.worldSize * 0.5f,
                          step = tp.worldSize / static_cast<float>(tp.resolution)](float x, float z) {
        auto* pos = terrainGeo->getAttribute<float>("position");
        const int ix = std::clamp(static_cast<int>(std::lround((x + half) / step)), 0, dim - 1);
        const int iz = std::clamp(static_cast<int>(std::lround((z + half) / step)), 0, dim - 1);
        return pos->array()[(static_cast<std::size_t>(iz) * dim + ix) * 3 + 1] - h0;
    };

    // Open water: below the waterline and under the sheet. The shoreline is
    // whatever contour the terrain draws, so vegetation gates on THIS rather
    // than on a radius — reeds and willows crowd the inlets, nothing stands in
    // the pond (spot_slam's rule).
    const float pondWaterY = pondWaterRaw - h0;
    const auto inPond = [&](float x, float z) {
        if (pondR <= 0.f) return false;
        if (std::max(std::abs(x - pondX), std::abs(z - pondZ)) > pondR) return false;
        return groundY(x, z) < pondWaterY + 0.15f;
    };

    // Pond water. On Vulkan: spot_slam's REAL water — an FFT Ocean patch with
    // dm-scale ripples, traced reflections of the sky/forest, and shallow
    // freshwater shading veiled into a green-brown murk (the bottom is the
    // natural basin floor, genuinely visible through the surface). GL keeps the
    // flat disc: Ocean is a Vulkan-renderer object.
    if (pondR > 0.f) {
#ifdef THREEPP_WITH_VULKAN
        if (vulkan) {
            Ocean::Options oo;
            // The sheet spans exactly the square whose rim set the water level,
            // so every edge and corner is buried under shore.
            oo.size = 2.f * pondR;
            oo.resolution = 128;
            oo.windSpeed = 3.f;// a breeze: ripples, not a seascape
            oo.windTheta = 0.5f;
            oo.choppiness = 0.4f;
            oo.waveScale = 0.4f;
            oo.fftSize = 256;
            auto pond = Ocean::create(oo);
            pond->position.set(pondX, pondWaterY, pondZ);
            // spot_slam's green-brown murk, but stretched: its pond was 0.4 m
            // deep and needed a 1.2 m veil to stop reading as wet sand. This
            // basin is metres deep, so a longer attenuation keeps the shallows
            // legibly green-brown while the middle goes properly dark.
            if (auto* pm = pond->materialAs<MeshPhysicalMaterial>()) {
                pm->attenuationColor = Color(0x245238);
                pm->attenuationDistance = 2.5f;
                pm->thickness = 0.8f;
            }
            scene.add(pond);
        } else
#endif
        {
            auto waterMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                                 .color(Color(0.05f, 0.16f, 0.20f))
                                                                 .roughness(0.06f)
                                                                 .metalness(0.f));
            // Inscribed in the same square: the disc edge dies under the rim.
            auto water = Mesh::create(CircleGeometry::create(pondR * 0.98f, 48), waterMat);
            water->rotation.x = -math::PI / 2.f;
            water->position.set(pondX, pondWaterY, pondZ);
            scene.add(water);
        }
    }

    // Scattered stones: unique low-poly rocks, denser near home, none in the
    // pad apron or the pond.
    {
        auto rockMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                            .color(Color(0.44f, 0.42f, 0.40f))
                                                            .roughness(0.95f)
                                                            .metalness(0.f));
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        for (int i = 0; i < 60; ++i) {
            const float r = 22.f + std::pow(u01(rng), 1.6f) * 520.f;
            const float a = u01(rng) * 2.f * math::PI;
            const float x = std::cos(a) * r, z = std::sin(a) * r;
            if (inPond(x, z)) continue;
            const float s = 0.25f + std::pow(u01(rng), 2.f) * 1.5f;
            auto rock = Mesh::create(terrain::makeRockGeometry(1000 + i), rockMat);
            rock->scale.set(s * (0.8f + 0.4f * u01(rng)), s * (0.6f + 0.3f * u01(rng)),
                            s * (0.8f + 0.4f * u01(rng)));
            rock->position.set(x, groundY(x, z) - 0.08f * s, z);
            rock->rotation.y = u01(rng) * 2.f * math::PI;
            rock->castShadow = r < 130.f;// keep the shadow map honest
            rock->receiveShadow = true;
            scene.add(rock);
        }
    }

    // Meadow grass around home — merged GrassMesh tiles (Vulkan animates the
    // sway and culls/freezes far tiles; GL draws them as static grass).
    // spot_slam's grass, transplanted: blades grow in TUFTS (real grass
    // clumps; a uniform scatter reads as green static), SHORT (0.10-0.24 m
    // meadow blades, not the knee-high 0.3-0.66 the first pass had), and stay
    // on the valley floor — no tufts dotting the ridgelines.
    {
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        std::vector<vegetation::GrassBlade> blades;
        constexpr int kTuft = 6;       // blades per tuft
        constexpr float kSpread = 0.12f;// max in-tuft offset [m]
        constexpr int kBladeTarget = 15000;
        blades.reserve(kBladeTarget + kTuft);
        const Vector3 up(0, 1, 0);
        int attempts = 0;
        while (blades.size() < kBladeTarget && ++attempts < 60000) {
            const float r = 5.f + std::sqrt(u01(rng)) * 43.f;
            const float a = u01(rng) * 2.f * math::PI;
            const float x = std::cos(a) * r, z = std::sin(a) * r;
            if (inPond(x, z)) continue;
            // Valley floor only: low AND locally flat.
            const float y = groundY(x, z);
            if (y > 6.f) continue;
            const float slope = std::abs(groundY(x + 3.f, z) - groundY(x - 3.f, z)) +
                                std::abs(groundY(x, z + 3.f) - groundY(x, z - 3.f));
            if (slope > 1.8f) continue;
            // One height query per tuft: blades within a few cm of each other,
            // the terrain is locally flat (spot_slam's exact economy).
            for (int b = 0; b < kTuft; ++b) {
                vegetation::GrassBlade bl;
                bl.position.set(x + (u01(rng) - 0.5f) * 2.f * kSpread, y - 0.03f,
                                z + (u01(rng) - 0.5f) * 2.f * kSpread);
                const float s = 0.7f + u01(rng) * 0.6f;
                bl.scale.set(s, 0.10f + u01(rng) * 0.14f, s);
                bl.yaw.setFromAxisAngle(up, u01(rng) * 2.f * math::PI);
                blades.push_back(bl);
            }
        }
        auto grassMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                             .color(Color(0.30f, 0.40f, 0.16f))
                                                             .roughness(1.f)
                                                             .metalness(0.f));
        grassMat->vertexColors = true;
        grassMat->side = Side::Double;
        GrassMesh::Params mp;
        mp.windDir = Vector2(0.8f, 0.6f);
        mp.windStrength = 0.16f;
        mp.maxAnimDistance = 95.f;
        for (auto& tile : vegetation::buildGrassTiles(blades, 40.f, grassMat, mp)) {
            scene.add(tile);
        }
    }

    // Landing pad at home.
    auto pad = Mesh::create(CylinderGeometry::create(1.6f, 1.6f, 0.04f),
                            MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                                 .color(Color(0.75f, 0.16f, 0.12f))
                                                                 .roughness(0.8f)));
    pad->position.y = 0.02f;
    pad->receiveShadow = true;
    scene.add(pad);

    // Windsock beside the pad: yaw pivot -> droop pivot -> cone along +X.
    // Driven every frame from the sim's instantaneous (gusty) wind: hangs
    // limp in calm, flies horizontal by ~8 m/s, flutters with the gusts.
    Object3D* sockYaw = nullptr;
    Object3D* sockDroop = nullptr;
    {
        auto poleMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.75f, 0.76f, 0.78f)).roughness(0.4f));
        auto sockMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.95f, 0.45f, 0.05f)).roughness(0.7f));
        sockMat->side = Side::Double;

        // Scaled up ~1.8x over a realistic sock: from a chase camera framing a
        // 0.5 m quad, a to-scale windsock is a few pixels of orange and the
        // wind direction is unreadable on video.
        auto pole = Mesh::create(CylinderGeometry::create(0.05f, 0.07f, 6.f), poleMat);
        pole->position.set(3.2f, 3.f, -3.2f);
        pole->castShadow = true;
        scene.add(pole);

        auto yaw = Group::create();
        yaw->position.set(3.2f, 5.9f, -3.2f);
        auto droop = Group::create();
        // Truncated cone, open ended, axis +Y; rotate so it opens along +X.
        auto cone = Mesh::create(CylinderGeometry::create(0.16f, 0.40f, 2.f, 12, 1, true), sockMat);
        cone->rotation.z = math::PI / 2.f;// +Y -> +X (wide mouth at the pivot)
        cone->position.x = 1.f;
        cone->castShadow = true;
        droop->add(cone);
        yaw->add(droop);
        scene.add(yaw);
        sockYaw = yaw.get();
        sockDroop = droop.get();
    }

    // Air streaks: faint drifting dashes that advect with the wind inside a
    // bubble around the drone. Unlit transparent material so both renderers
    // blend them (see the rotor-blur note in DroneVisual).
    struct Streak {
        Mesh* mesh;
        Vector3 pos;
        float speedJitter;
    };
    // Sized for the camera, not for realism: on video a sparse scatter of
    // hairline dashes reads as compression noise, so the field is 3x denser,
    // the dashes are thicker and brighter, and the bubble is wider than the
    // chase framing so streaks enter from off-screen rather than popping in.
    constexpr int streakCount = 420;
    constexpr float streakBubble = 90.f;// half-extent in x/z; y uses 40
    std::vector<Streak> streaks;
    {
        auto streakGeo = BoxGeometry::create(1.f, 0.05f, 0.05f);
        auto streakMat = MeshBasicMaterial::create();
        streakMat->color = Color(0.9f, 0.93f, 1.f);
        streakMat->transparent = true;
        streakMat->opacity = 0.3f;
        streakMat->depthWrite = false;
        std::mt19937 rng(23);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        streaks.reserve(streakCount);
        for (int i = 0; i < streakCount; ++i) {
            auto m = Mesh::create(streakGeo, streakMat);
            m->visible = false;
            scene.add(m);
            streaks.push_back({m.get(),
                               Vector3((u01(rng) - 0.5f) * streakBubble * 2.f, u01(rng) * 80.f,
                                       (u01(rng) - 0.5f) * streakBubble * 2.f),
                               0.75f + 0.5f * u01(rng)});
        }
    }

    // Drone.
    uav::DroneVisual drone;
    drone.root()->position.y = 0.04f + uav::DroneVisual::hullY / 2.f;// on the pad
    scene.add(drone.root());

    PerspectiveCamera camera(60.f, canvas.aspect(), 0.1f, 10000.f);
    camera.position.set(2.2f, 1.3f, 2.9f);
    OrbitControls controls{camera, canvas};
    // Chase state: the drone position the camera was last reconciled against.
    // Seeded from the drone so the first frame contributes no displacement.
    Vector3 chaseAnchor{drone.root()->position};
    controls.target.copy(drone.root()->position);
    bool followDrone = true;

    // SITL link. The servo packet carries a frame_rate SITL would LIKE, but it
    // is dynamic (SITL ramps it while syncing its speedup) and the protocol
    // explicitly allows the physics backend to ignore it — lock-step means
    // SITL follows OUR timestamps regardless. So the PhysX world is built
    // exactly once at a FIXED rate, and every packet advances the sim by
    // exactly one substep. Rebuilding PhysxWorld per declared rate is not an
    // option anyway: PhysX foundation state does not survive rapid recreate.
    //
    // 1200 Hz, not ArduCopter's 400 Hz loop rate: the INS prearm check
    // demands gyro sample rate >= 1.8x loop rate ("Gyro 0 rate 400Hz < loop
    // rate*1.8 720Hz"), and 3x the loop rate is what SITL itself asks for.
    constexpr std::uint16_t simRateHz = 1200;
    uav::SitlBridge bridge(port);
    if (!bridge.valid()) {
        std::fprintf(stderr, "[sitl] failed to bind UDP port %u — is another instance running?\n", port);
        return 1;
    }
    auto quad = std::make_unique<uav::QuadSim>(simRateHz, *drone.root(), terrainMesh.get());
    quad->world().addStaticTrimesh(*pad);// land flush on the pad, not 4 cm inside it

    // ── Forest ──────────────────────────────────────────────────────────────
    // spot_slam's species mix (oak / spruce-heavy / birch), built the C++ way
    // (forest_demo's variant pooling: a handful of fully-realized prototypes,
    // every placement shares geometry + bark/leaf textures), scattered LARGER —
    // a treeline ringing the meadow out to ~240 m. Each trunk drops a static
    // box collider into the quad's world: the rangefinder raycast and any
    // future obstacle-avoidance sensor see the same forest the camera does.
    // Built after the quad so the colliders have a world to land in.
    {
        struct TreeVariant {
            std::shared_ptr<BufferGeometry> trunkGeo, leafGeo;
            std::shared_ptr<MeshStandardMaterial> barkMat, leafMat;
            float trunkRadius;
        };
        const auto makeLeafMat = [](const std::shared_ptr<Texture>& map) {
            auto m = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                          .color(Color::white)
                                                          .roughness(0.85f)
                                                          .metalness(0.f));
            m->map = map;
            m->alphaTest = 0.4f;// forest_demo's mip-safe cutout threshold
            m->side = Side::Double;
            m->vertexColors = true;
            m->translucency = 0.45f;// backlit canopy glow (Vulkan; no-op on GL)
            m->translucencyColor = Color(0.55f, 0.85f, 0.30f);
            return m;
        };

        std::vector<TreeVariant> variants;
        // Presets: 0=oak, 1=spruce (x2 — spot_slam's spruce-heavy mix), 2=birch.
        const int presets[] = {0, 1, 1, 2};
        std::mt19937 vrng(4242);
        for (int preset : presets) {
            for (int k = 0; k < 2; ++k) {
                const auto seed = static_cast<unsigned int>(vrng());
                vegetation::TreeParams tpar;
                vegetation::applyPreset(preset, tpar);
                tpar.seed = seed;
                vegetation::TreeGenerator tgen(seed);
                tgen.buildSkeleton(tpar);

                TreeVariant v;
                v.trunkGeo = tgen.makeTrunkGeometry(tpar);
                v.leafGeo = tgen.makeLeafGeometry(tpar);
                auto bark = vegetation::makeBarkTextures(192, seed, tpar.barkColor,
                                                         tpar.barkStyle);
                bark.first->repeat.set(3.f, 0.5f);
                bark.second->repeat.set(3.f, 0.5f);
                v.barkMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                                 .color(Color::white)
                                                                 .roughness(0.92f)
                                                                 .metalness(0.f));
                v.barkMat->map = bark.first;
                v.barkMat->normalMap = bark.second;
                v.leafMat = makeLeafMat(
                        tpar.leafStyle == vegetation::LeafStyle::Frond
                                ? vegetation::makeNeedleFrondTexture(224, seed, tpar.leafColor)
                                : vegetation::makeLeafClusterTexture(224, seed, tpar.leafColor,
                                                                     tpar.leafShape));
                v.trunkRadius = tpar.trunkRadius;
                variants.push_back(std::move(v));
            }
        }

        std::mt19937 rng(31);
        std::uniform_real_distribution<float> u01f(0.f, 1.f);
        int placed = 0;
        for (int i = 0; i < 3200 && placed < 240; ++i) {
            // Ring 22..240 m: a clear flight bowl around the pad, forest beyond.
            const float r = 22.f + std::pow(u01f(rng), 0.8f) * 218.f;
            const float a = u01f(rng) * 2.f * math::PI;
            const float x = std::cos(a) * r, z = std::sin(a) * r;
            if (inPond(x, z)) continue;
            const float y = groundY(x, z);
            // Valley and lower slopes only: a tree standing on a ridge crest
            // draws itself on the skyline and reads as a cardboard cutout.
            if (y > 12.f) continue;
            const float slope = std::abs(groundY(x + 3.f, z) - groundY(x - 3.f, z)) +
                                std::abs(groundY(x, z + 3.f) - groundY(x, z - 3.f));
            if (slope > 2.4f) continue;// no trees on steep ground either
            // Skyline intrusion is LOCAL PROMINENCE, not altitude. The absolute
            // gates above miss the case the complaint was actually about: a
            // tree on a spur crest at y = 8 still draws itself against the sky
            // from the valley floor, while one halfway up a long 30 m slope
            // never does — there is always more hill behind it. So ask the
            // terrain the local question: sample 12 m out in the four
            // cardinals, and refuse the site only if it stands more than 1.5 m
            // above the LOWEST neighbour AND above at least three of the four.
            // Both halves are needed — the drop alone also fires on a slope
            // shoulder (high on one side, low on the other), which is exactly
            // the ground trees should keep.
            {
                const float ring[4] = {groundY(x + 12.f, z), groundY(x - 12.f, z),
                                       groundY(x, z + 12.f), groundY(x, z - 12.f)};
                float lowest = ring[0];
                int above = 0;
                for (const float g : ring) {
                    lowest = std::min(lowest, g);
                    if (y > g) ++above;
                }
                if (y - lowest > 1.5f && above >= 3) continue;// a crest, not a slope
            }

            const auto& v = variants[rng() % variants.size()];
            const float s = 0.85f + u01f(rng) * 0.75f;
            // Mission planning against THIS forest needs the actual trunk
            // map, not bearings squinted from fan telemetry (three avoidance
            // test flights flew dead straight through guessed "treelines").
            if (std::getenv("THREEPP_FOREST_DUMP")) {
                std::fprintf(stderr, "[tree] %.1f %.1f s=%.2f y=%.1f\n", x, z, s, y);
            }
            auto trunk = Mesh::create(v.trunkGeo, v.barkMat);
            trunk->position.set(x, y - 0.06f, z);
            trunk->scale.setScalar(s);
            trunk->rotation.y = u01f(rng) * 2.f * math::PI;
            // The shadow frustum is a tight box tracking the drone, so only
            // trees the flight can actually reach need to cast.
            trunk->castShadow = r < 90.f;
            trunk->receiveShadow = true;
            auto leaves = Mesh::create(v.leafGeo, v.leafMat);
            leaves->castShadow = trunk->castShadow;
            trunk->add(leaves);
            scene.add(trunk);

            // The trunk stub the physics world sees (rangefinder, future
            // avoidance). Canopy stays collider-free: a downward beam through
            // leaves mostly gets ground, and the drone clipping a twig is not
            // this demo's physics story.
            {
                using namespace ::physx;
                const float hr = std::max(0.10f, v.trunkRadius * s * 1.2f);
                const float hh = 2.6f * s;
                quad->world().addStatic(
                        PxBoxGeometry(hr, hh, hr),
                        PxTransform(PxVec3(x, y + hh, z)));
            }
            ++placed;
        }
        std::fprintf(stderr, "[forest] placed %d trees (%zu variants)\n", placed,
                     variants.size());
    }
    uav::FdmState lastState{};
    uav::ServoInput lastServo{};
    std::uint32_t resets = 0;
    // HUD-driven. Brownout defaults: wind FROM the NW quadrant blows the cloud
    // toward the tripod camera at (7.5, _, 7.5), which is what lets the shot
    // end INSIDE the dust.
    float windSpeed = brownout ? 1.6f : 0.f;
    float windFromDeg = brownout ? 315.f : 270.f;
    float windGust = brownout ? 0.35f : 0.3f;

    // ── Rotor-wash dust (Vulkan only — DensityRepr has no GL path) ──────────
    // Driven from the SAME quad state in both modes: an ArduPilot-flown
    // landing browns out exactly like the scripted one.
    std::shared_ptr<uav::DownwashEffect> dust;
    if (vulkan) {
        dust = uav::DownwashEffect::create();
        scene.add(dust);
    }

    // The base HUD wind vector in threepp world space (blows FROM windFromDeg,
    // meteorological convention).
    const auto windBase = [&]() -> Vector3 {
        const float toRad = math::degToRad(windFromDeg + 180.f);
        return uav::frame::nedToTp(windSpeed * std::cos(toRad),
                                   windSpeed * std::sin(toRad), 0.0);
    };
    // In brownout mode the QUAD flies calm air (there is no position-hold
    // controller to fight a crosswind, and a drifting landing misses the pad)
    // while the DUST, the windsock and the streaks ride a styled breeze —
    // QuadSim's own gust model, mirrored. In SITL mode everything shares the
    // quad's one air, and ArduPilot does the fighting.
    const auto visualWindAt = [&](float t) -> Vector3 {
        Vector3 w = windBase();
        const float mag = 1.f + windGust * (0.45f * std::sin(0.9f * t) +
                                            0.30f * std::sin(2.3f * t + 1.7f) +
                                            0.20f * std::sin(5.1f * t + 0.4f));
        w.multiplyScalar(mag);
        return w;
    };

    // Scripted-flight throttle: altitude PD around the QuadSim hover point
    // (thrust curve at expo 0.65 hovers at ~0.407 throttle => pwm ~1407).
    std::uint32_t scriptFrame = 0;
    const auto scriptedPwm = [&](double simT) -> std::uint16_t {
        const float c = std::fmod(static_cast<float>(simT), kScriptPeriod);
        if (c >= 24.f) return 1150;// spool on the pad: dust, no liftoff
        const float altT = scriptAlt(c);
        if (altT < 0.f) return 1000;// cut
        const float altT2 = scriptAlt(std::min(c + 0.05f, 18.99f));
        const float vT = (altT2 - altT) / 0.05f;
        const float alt = static_cast<float>(-lastState.positionNed[2]);
        const float vUp = static_cast<float>(-lastState.velocityNed[2]);
        const float pwm = 1407.f + 150.f * (altT - alt) + 230.f * (vT - vUp);
        return static_cast<std::uint16_t>(std::clamp(pwm, 1090.f, 1850.f));
    };
    // The rangefinder flies THROUGH the dust: degrade rng_1 from the cached
    // optical depth before anyone consumes it. In SITL mode a NaN is omitted
    // from the JSON, so ArduPilot's rangefinder driver genuinely times out —
    // its landing logic and EKF experience the brownout, closed loop.
    std::uint32_t rngDrops = 0, rngEarly = 0;// per-telemetry-window counters
    // The CLEAN range, latched before the dust is allowed to eat it. The
    // proximity fan's ground hint is built from AGL, and reading it back off
    // lastState after degradation would hand it a NaN precisely during the
    // touchdown blast — i.e. lose ground segmentation in the one second of the
    // flight it exists for. What the autopilot sees stays degraded; this is
    // simulator ground truth and never goes on the wire.
    double trueAglM = std::numeric_limits<double>::quiet_NaN();
    const auto degradeRangefinder = [&](std::uint32_t tick) {
        trueAglM = lastState.rangefinderM;
        if (!dust) return;
        const double trueR = lastState.rangefinderM;
        lastState.rangefinderM = dust->degradedRange(trueR, tick);
        if (std::isnan(lastState.rangefinderM) && !std::isnan(trueR)) ++rngDrops;
        else if (lastState.rangefinderM < trueR * 0.9) ++rngEarly;
    };
    // ── Proximity fan -> MAVLink OBSTACLE_DISTANCE (--proximity) ────────────
    // A 72-sector horizontal fan, framed per MAV_FRAME_BODY_FRD and shipped at
    // 10 Hz. Off by default: absent --proximity nothing below constructs,
    // nothing is scanned and nothing is drawn.
    //
    // TWO SOURCES, and which one is live is a property of the backend:
    //
    //   Vulkan  a ray-traced LidarSensor, aimed at the renderer's own TLAS.
    //   GL      PhysX raycasts against the scene the quad flies in.
    //
    // The PhysX rays only ever see COLLIDERS, and this demo's canopy is
    // deliberately collider-free (a trunk stub is all the forest contributes) —
    // so the fan reported a clear lane straight through foliage the camera was
    // drawing at the time. A traced lidar sees what the renderer sees, leaves
    // included, which is also what a real scanner on this airframe would.
    //
    // The scan runs in BOTH step paths (scripted and SITL-driven), so the
    // overlay is inspectable — and capturable — with no autopilot on the wire.
    std::unique_ptr<uav::ProximityScan> prox;
    std::unique_ptr<uav::MavlinkOut> mavout;
    std::shared_ptr<BufferGeometry> proxGeom;
    // The traced source (Vulkan only) and its mount. The sensor hangs off the
    // SCENE, not off the drone visual, because its orientation is a software
    // gimbal: it rides the vehicle's position and yaw and ignores its attitude.
    // Parented under a quad that pitches 25 degrees to accelerate, the rings
    // would rake the dirt ahead and the horizon behind.
    std::unique_ptr<LidarSensor> proxLidar;
    std::vector<LidarReturn> proxCloud;
    // The traced backend needs a TLAS, which exists only after the first
    // render. --seq fast-forwards the whole flight before drawing anything, so
    // this is a real ordering hazard and not a theoretical one.
    bool renderedOnce = false;
    double nextProxScan = 0., nextProxBeat = 0., nextProxLog = 0.;
    if (proximity) {
        prox = std::make_unique<uav::ProximityScan>();
        mavout = std::make_unique<uav::MavlinkOut>(proxPort);
        if (!mavout->valid()) {
            std::fprintf(stderr, "[prox] could not bind UDP :%u — scanning, sending nothing\n",
                         proxPort);
        } else {
            std::fprintf(stderr,
                         "[prox] listening on UDP :%u (SITL: --serial2=udpclient:<host>:%u)\n",
                         mavout->port(), mavout->port());
        }

        // ONE geometry, updated in place forever after. Rebuilding a line
        // geometry per scan hands the renderer a recycled pointer at the old
        // key and the Vulkan line cache serves stale segments
        // (feedback_vulkan_line_geom_cache_pointer_recycle).
        proxGeom = BufferGeometry::create();
        const std::size_t floats = uav::ProximityScan::sectors * 2 * 3;
        proxGeom->setAttribute("position",
                               FloatBufferAttribute::create(std::vector<float>(floats), 3));
        proxGeom->setAttribute("color",
                               FloatBufferAttribute::create(std::vector<float>(floats), 3));
        proxGeom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
        proxGeom->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);
        // Unlit + transparent, so the Vulkan backend routes this to the overlay
        // pass instead of trying to shade it deferred
        // (feedback_vulkan_svg_ui_blend_overlay).
        auto fanMat = LineBasicMaterial::create(LineBasicMaterial::Params{}
                                                        .vertexColors(true)
                                                        .transparent(true)
                                                        .opacity(0.65f)
                                                        .depthWrite(false));
        auto fan = LineSegments::create(proxGeom, fanMat);
        // Spokes are written in WORLD space while the node stays at the origin,
        // so its bounds are permanently wrong — culling would blink the fan out.
        fan->frustumCulled = false;
        // No self-hit risk from the overlay: Line/LineSegments are collected
        // ahead of the mesh dispatch in the Vulkan scene walk and never reach
        // the TLAS at all, so the fan cannot range on itself.
        scene.add(fan);

        if (vulkan) {
            // Five rings across +-8 degrees at 2.5 degree azimuth = 720 beams:
            // enough vertical spread that a canopy edge or a trunk is caught by
            // SOME ring while the drone bobs, and cheap enough to trace at
            // 10 Hz. The slab filter downstream is what decides which of those
            // rings' returns are obstacles rather than ground.
            LidarModel model;
            model.elevationAngles = {-8.f, -4.f, 0.f, 4.f, 8.f};
            // 1 degree, not 2.5: at 2.5 a 0.4 m trunk subtends less than the
            // beam spacing beyond ~9 m and can hide BETWEEN azimuth samples —
            // measured as a 3.8 m pop-up that ended a BendyRuler flight in a
            // crash. Real scanning lidars are denser still (VLP-16: 0.2).
            model.azimuthResolution = 1.0f;
            // near = the blind SPHERE, and it is sized to swallow the airframe:
            // hull, booms and prop discs all sit inside 0.45 m of the mount, so
            // the vehicle cannot report itself as an obstacle at 0.2 m in every
            // sector at once. far matches what we declare on the wire.
            // faceSize is dead weight here (it sizes the GL cube-face targets,
            // and this sensor only ever traces) — keep it small.
            proxLidar = std::make_unique<LidarSensor>(model, 128u, 0.6f,
                                                      prox->params().maxRange);
            scene.addRef(*proxLidar);
        }
        std::fprintf(stderr, "[prox] source: %s\n",
                     proxLidar ? "lidar (traced)" : "physx rays");
    }

    // The GL source. Rays hit the world, never the vehicle: trunks, rocks and
    // terrain are STATIC and the drone body is dynamic, so the same eSTATIC
    // filter QuadSim::raycastAgl uses keeps a ray that starts inside the hull
    // from reporting 0 m in all 72 sectors. Horizontal, not down. Constructed
    // unconditionally and simply unused when the traced lidar is live.
    const uav::RayCaster proxCast = [&](const Vector3& o, const Vector3& d, float maxR) -> float {
        using namespace ::physx;
        PxRaycastBuffer hit;
        PxQueryFilterData fd;
        fd.flags = PxQueryFlag::eSTATIC;
        if (quad->world().scene().raycast(PxVec3(o.x, o.y, o.z), PxVec3(d.x, d.y, d.z), maxR,
                                          hit, PxHitFlag::eDEFAULT, fd) &&
            hit.hasBlock) {
            return hit.block.distance;
        }
        return -1.f;
    };

    // Paced on SIM time, not wall time: the capture path runs faster than real
    // time and the SITL path runs at whatever rate the autopilot manages, so a
    // wall clock would emit a different message stream in each and neither at
    // 10 Hz as declared.
    const auto serviceProximity = [&] {
        if (!prox) return;
        // Every tick, because SITL dials out to US — until one of its datagrams
        // lands there is no return address to send to.
        mavout->poll();

        const double t = lastState.timestampSec;
        // Sim time comes from SITL's frame counter, and a SITL REBOOT resets
        // it to zero. Gates stranded minutes in the future would mute the fan
        // until sim time caught back up — which ArduPilot reports as the
        // maximally misleading "PreArm: PRX1: No Data". A gate more than one
        // period ahead of now is only reachable by time running backwards.
        if (nextProxScan > t + 0.1) {
            nextProxScan = t;
            nextProxBeat = t;
            nextProxLog = t;
        }
        if (t >= nextProxBeat) {
            nextProxBeat = t + 1.0;// 1 Hz; a silent component is ignored
            mavout->sendHeartbeat();
        }
        if (t < nextProxScan) return;
        nextProxScan = t + 0.1;// 10 Hz

        // The nose: DroneVisual authors forward as -Z, and root() hangs
        // directly off the scene, so its local quaternion IS the world one.
        // ProximityScan flattens this to a yaw — the fan never tilts with the
        // vehicle, or a pitched-forward quad would scan into the dirt.
        Vector3 fwd(0.f, 0.f, -1.f);
        fwd.applyQuaternion(drone.root()->quaternion);

        if (proxLidar && renderedOnce) {
            const Vector3 mount(drone.root()->position.x,
                                drone.root()->position.y + 0.12f,
                                drone.root()->position.z);
            // Take delivery of whatever the GPU finished since last tick. A
            // refused or still-in-flight scan leaves proxCloud alone, so the
            // wire keeps its 10 Hz cadence carrying a slightly older picture —
            // which beats a gap: ArduPilot mutes avoidance on stale PRX data,
            // and a 100 ms hole is the wrong way to earn that.
            if (proxLidar->scanReady(*renderer)) proxLidar->scanCollect(*renderer, proxCloud);

            // Bin against the CURRENT pose, not the pose the cloud was fired
            // from: OBSTACLE_DISTANCE describes where the obstacles are
            // relative to the vehicle NOW, and one tick of parallax (< 0.5 m at
            // this demo's speeds) is smaller than the error of pretending the
            // vehicle is still back where it fired.
            // Ground segmentation: hand the fan the terrain height under the
            // vehicle, from the altimeter it already carries. Without it a
            // landing reads the floor as a wall in every sector once the
            // sensor drops inside a metre of it (the slab rides the sensor,
            // so the ground climbs into it) and ArduPilot's avoidance fights
            // the pad. NaN when nothing is in altimeter range — high cruise,
            // where the slab alone is already right.
            const float groundY = std::isnan(trueAglM)
                                          ? std::numeric_limits<float>::quiet_NaN()
                                          : drone.root()->position.y -
                                                    static_cast<float>(trueAglM);
            prox->beginFrame(mount, fwd, groundY);
            for (const auto& r : proxCloud) prox->feed(r.position);

            // Aim and fire the next one. Software gimbal: position + yaw from
            // the vehicle, attitude discarded, so the rings stay level and the
            // slab filter has a level frame to cut against. Local -Z is the
            // sensor's azimuth zero (LidarModel's convention), hence the sign.
            proxLidar->position.copy(mount);
            proxLidar->rotation.set(0.f, std::atan2(-fwd.x, -fwd.z), 0.f);
            // The scene's own matrix update runs at render time, i.e. after
            // this — force it, or the trace aims from last frame's pose.
            proxLidar->updateMatrixWorld(true);
            if (!proxLidar->scanPending()) proxLidar->scanBegin(*renderer, scene, proxCloud);
        } else if (!proxLidar) {
            prox->scan(proxCast, drone.root()->position, fwd);
        }
        const auto& dist = prox->distances();

        mavout->sendObstacleDistance(static_cast<std::uint64_t>(t * 1e6), dist,
                                     prox->minCm(), prox->maxCm(),
                                     static_cast<std::uint8_t>(prox->params().incrementDeg),
                                     0.f, uav::mavlink::frameBodyFrd);

        auto* posAttr = proxGeom->getAttribute<float>("position");
        auto* colAttr = proxGeom->getAttribute<float>("color");
        const Vector3& o = drone.root()->position;
        const std::uint16_t clearAt = prox->clearValue();
        const Color blockedColor(1.f, 0.42f, 0.10f);
        const Color clearColor(0.16f, 0.26f, 0.34f);
        int blocked = 0;
        float nearest = 0.f;
        std::size_t nearestK = 0;
        for (std::size_t k = 0; k < uav::ProximityScan::sectors; ++k) {
            const bool isHit = dist[k] < clearAt;
            // A clear sector still draws, but stubbed at 8 m: a full 50 m fan
            // of free space is a white disc that hides the obstacles.
            const float len = isHit ? static_cast<float>(dist[k]) * 0.01f : 8.f;
            const Vector3 dir = prox->sectorDir(k);
            const auto v = static_cast<int>(k) * 2;
            posAttr->setXYZ(v, o.x, o.y, o.z);
            posAttr->setXYZ(v + 1, o.x + dir.x * len, o.y, o.z + dir.z * len);
            const Color& c = isHit ? blockedColor : clearColor;
            colAttr->setXYZ(v, c.r, c.g, c.b);
            colAttr->setXYZ(v + 1, c.r, c.g, c.b);
            if (isHit && (blocked == 0 || len < nearest)) {
                nearest = len;
                nearestK = k;
            }
            blocked += isHit ? 1 : 0;
        }
        posAttr->needsUpdate();
        colAttr->needsUpdate();

        if (t >= nextProxLog) {
            nextProxLog = t + 1.0;
            char nearTxt[64];
            if (blocked > 0) {
                std::snprintf(nearTxt, sizeof nearTxt, "%.1f m @ %.0f deg", nearest,
                              prox->sectorBearingDeg(nearestK));
            } else {
                std::snprintf(nearTxt, sizeof nearTxt, "all clear");
            }
            std::fprintf(stderr,
                         "[prox] t=%.1f nearest %s, blocked %d/%zu, returns %zu, peer %s\n", t,
                         nearTxt, blocked, uav::ProximityScan::sectors,
                         proxLidar ? proxCloud.size() : std::size_t{0},
                         mavout->hasPeer() ? mavout->peer().c_str() : "no");
        }
    };

    // One scripted physics substep at the sim rate.
    const auto scriptedStep = [&] {
        uav::ServoInput in{};
        in.frameRate = simRateHz;
        in.frameCount = ++scriptFrame;
        in.channels = 16;
        const std::uint16_t pwm = scriptedPwm(lastState.timestampSec);
        for (int m = 0; m < 4; ++m) in.pwm[m] = pwm;
        quad->step(in, lastState);
        degradeRangefinder(in.frameCount);
        serviceProximity();
        lastServo = in;
    };
    // Feed the dust from whatever just flew (scripted or SITL).
    const auto updateDust = [&] {
        if (!dust) return;
        const float t = static_cast<float>(lastState.timestampSec);
        // STEADY wind + gustiness, never a gust-modulated sample: the effect
        // draws per-lobe gusts internally. Feeding it the shared gusty vector
        // advected every parcel in sync and the cloud surged on the gust
        // sines — the "wavelike" report.
        dust->setWind(windBase());
        dust->setGustiness(windGust);
        const float thrust = (quad->motorLevel(0) + quad->motorLevel(1) +
                              quad->motorLevel(2) + quad->motorLevel(3)) * 0.25f;
        // TRUE AGL, never the degraded rangefinder: the blast drops the
        // sensor to NaN, NaN read as "very high" switches entrainment off,
        // and the cloud starves itself mid-blast — dust physics consuming its
        // own corrupted measurement. The wire still carries the degraded
        // value; the ground does not move because the sensor got blinded.
        const float agl = std::isnan(trueAglM) ? 1e9f : static_cast<float>(trueAglM);
        dust->update(t, drone.root()->position, thrust, agl);
    };

    // Waypoint marker channel: the mission lives inside ArduPilot and the
    // physics protocol never mentions it, so fly_test.py posts its route here
    // as one plain-text datagram — "N,E,ALT;N,E,ALT;..." (home-relative
    // metres, NED) on UDP port+6 (9008 by default; SITL's own side ports sit
    // at 9003-9005). Empty datagram clears the markers.
    ensureSockets();
    socket_t markerSock = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(static_cast<std::uint16_t>(port + 6));
#ifdef _WIN32
        u_long nb = 1;
        ::bind(markerSock, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
        ioctlsocket(markerSock, FIONBIO, &nb);
#else
        ::bind(markerSock, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
        fcntl(markerSock, F_SETFL, fcntl(markerSock, F_GETFL, 0) | O_NONBLOCK);
#endif
    }
    auto markerGroup = Group::create();
    scene.add(markerGroup);
    const auto rebuildMarkers = [&](const std::string& text) {
        markerGroup->clear();
        // TRANSPARENT on purpose, and not for looks: unlit transparent routes
        // to the overlay pass, which never enters the TLAS — an opaque ring
        // is a real 3 m torus at flight altitude that the lidar dutifully
        // reports and the planner then avoids. The vehicle must never sense
        // its own UI. (Observed live: BendyRuler refusing its own waypoint.)
        auto ringMat = MeshBasicMaterial::create();
        ringMat->color = Color(1.f, 0.72f, 0.1f);
        ringMat->transparent = true;
        ringMat->opacity = 0.9f;
        ringMat->depthWrite = false;
        auto beamMat = MeshBasicMaterial::create();
        beamMat->color = Color(1.f, 0.72f, 0.1f);
        beamMat->transparent = true;
        beamMat->opacity = 0.22f;
        beamMat->depthWrite = false;
        auto pathMat = MeshBasicMaterial::create();
        pathMat->color = Color(0.35f, 0.75f, 1.f);
        pathMat->transparent = true;
        pathMat->opacity = 0.35f;
        pathMat->depthWrite = false;

        std::vector<Vector3> pts;
        double n, e, alt;
        const char* p = text.c_str();
        int consumed;
        while (std::sscanf(p, "%lf,%lf,%lf;%n", &n, &e, &alt, &consumed) == 3) {
            pts.push_back(uav::frame::nedToTp(n, e, -alt));
            p += consumed;
        }
        const Vector3 up(0, 1, 0);
        for (const auto& wp : pts) {
            auto ring = Mesh::create(TorusGeometry::create(3.f, 0.28f, 10, 28), ringMat);
            ring->rotation.x = math::PI / 2.f;// horizontal halo at the waypoint altitude
            ring->position.copy(wp);
            markerGroup->add(ring);
            const float ground = groundY(wp.x, wp.z);
            auto beam = Mesh::create(
                    CylinderGeometry::create(0.12f, 0.12f, wp.y - ground), beamMat);
            beam->position.set(wp.x, (wp.y + ground) * 0.5f, wp.z);
            markerGroup->add(beam);
        }
        for (std::size_t i = 1; i < pts.size(); ++i) {
            Vector3 dir = pts[i];
            dir.sub(pts[i - 1]);
            const float len = dir.length();
            if (len < 1e-3f) continue;
            dir.divideScalar(len);
            auto seg = Mesh::create(CylinderGeometry::create(0.15f, 0.15f, len, 6), pathMat);
            seg->position.copy(pts[i - 1]).add(pts[i]).multiplyScalar(0.5f);
            seg->quaternion.setFromUnitVectors(up, dir);
            markerGroup->add(seg);
        }
        std::fprintf(stderr, "[markers] %zu waypoints\n", pts.size());
    };

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    // ── Deterministic brownout capture ──────────────────────────────────────
    // Headless, fixed dt, closed-form flight, tripod camera, AE pinned
    // (feedback_vulkan_capture_confounds: AE would cancel the very darkening
    // the cloud causes). Two runs of the same command produce the same flight;
    // frames land as f%03d.png for the LOOK pass.
#ifdef THREEPP_WITH_VULKAN
    if (!seqDir.empty()) {
        auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get());
        if (!vk || !dust) {
            std::fprintf(stderr, "--seq needs the Vulkan backend\n");
            return 1;
        }
        // Auto-exposure stays ON, deliberately: this is a cinematic capture,
        // not an A/B metric (where feedback_vulkan_capture_confounds says pin
        // it), and a camera that meters — and gets fooled inside the cloud —
        // is what a real landing video does. The scripted flight drives the
        // same frames every run, so the metered exposure reproduces too.
        vk->setVolumetricFog(true);// aerial perspective + the sun's dust shafts
        // TAA, not DLSS: a vehicle descending THROUGH its own dust is the
        // exact emitter-fog silhouette case the upscalers still ghost on
        // (feedback_upscaler_emitter_fog_silhouettes) — under DLSS the drone
        // drags a dark smeared trail through the cloud. Fire shipped gated on
        // TAA for the same reason.
        vk->setDlss(false);
        vk->setFsr(false);
        namespace fs = std::filesystem;
        fs::create_directories(seqDir);
        constexpr float kDt = 1.f / 60.f;
        // Fast-forward the FLIGHT to --start without rendering, feeding the
        // dust at frame cadence so the cloud's latched history is the one a
        // full run would have.
        {
            int sub = 0;
            while (lastState.timestampSec < seqStart) {
                scriptedStep();
                if (++sub == 20) {
                    sub = 0;
                    updateDust();
                }
            }
        }
        // Tripod: downwind of the pad, chest height, so the cloud rolls over
        // the lens near touchdown. PERPENDICULAR to the sun azimuth, not
        // opposite it: on the anti-solar axis the drone's volumetric shadow
        // shaft through the dust points straight down the lens and reads as a
        // dark smudge glued to the vehicle; from the side it reads as what it
        // is — a slanted shadow beam in the cloud.
        camera.position.set(5.6f, 1.4f, 5.6f);
        // Scene-inspection override: THREEPP_SEQ_CAM="x,y,z[,tx,ty,tz]" moves
        // the tripod (and optionally aims it somewhere other than the drone).
        bool aimFixed = false;
        Vector3 aimAt;
        if (const char* c = std::getenv("THREEPP_SEQ_CAM")) {
            float v[6];
            const int got = std::sscanf(c, "%f,%f,%f,%f,%f,%f",
                                        &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]);
            if (got >= 3) camera.position.set(v[0], v[1], v[2]);
            if (got == 6) {
                aimAt.set(v[3], v[4], v[5]);
                aimFixed = true;
            }
        }
        const double t0 = lastState.timestampSec;
        for (int i = 0; i < seqWarm + seqFrames; ++i) {
            const double target = t0 + static_cast<double>(i + 1) * kDt;
            while (lastState.timestampSec < target - 1e-9) scriptedStep();

            float levels[4];
            for (int m = 0; m < 4; ++m) levels[m] = quad->motorLevel(m);
            drone.setMotors(levels, kDt);
            updateDust();

            // Windsock agrees with the dust's air (it may be in frame).
            {
                const Vector3 w = visualWindAt(static_cast<float>(lastState.timestampSec));
                const float speed = std::hypot(w.x, w.z);
                if (speed > 0.05f) sockYaw->rotation.y = std::atan2(-w.z, w.x);
                sockDroop->rotation.z = -(1.f - std::min(speed / 8.f, 1.f)) * 1.4f;
            }

            sunTarget->position.copy(drone.root()->position);
            sun->position.copy(sunDir).multiplyScalar(300.f).add(sunTarget->position);

            Vector3 aim = drone.root()->position;
            aim.y += 0.5f;
            if (aimFixed) aim = aimAt;
            camera.lookAt(aim);

            canvas.animateOnce([&] { renderer->render(scene, camera); });
            renderedOnce = true;// the TLAS exists now; the traced fan may fire
            if (i % 20 == 0) {
                const auto& dr = dust->field()->densityRepr();
                char rng[16];
                if (std::isnan(lastState.rangefinderM)) {
                    std::snprintf(rng, sizeof rng, "DROP");
                } else {
                    std::snprintf(rng, sizeof rng, "%.2f", lastState.rangefinderM);
                }
                std::fprintf(stderr,
                             "[dust] t=%.2f alt=%.2f str=%.2f air=%u tau=%.2f rng=%s "
                             "drops=%u early=%u gnd=%.1f%% "
                             "box c=(%.1f %.1f %.1f) h=(%.1f %.1f %.1f)\n",
                             lastState.timestampSec, -lastState.positionNed[2],
                             dust->dustiness(), dust->airborne(),
                             dust->opticalDepthBelow(), rng, rngDrops, rngEarly,
                             100.f * dust->groundDustAt(drone.root()->position.x,
                                                        drone.root()->position.z) /
                                     (dust->params().groundDustPerM2 *
                                      dust->params().gridCell * dust->params().gridCell),
                             dr.center.x, dr.center.y, dr.center.z,
                             dr.halfExtent.x, dr.halfExtent.y, dr.halfExtent.z);
            }
            if (i >= seqWarm) {
                char name[64];
                std::snprintf(name, sizeof name, "f%03d.png", i - seqWarm);
                vk->writeFramebuffer((fs::path(seqDir) / name).string());
            }
        }
        std::printf("[seq] %d frames -> %s (warm %d, AE pinned)\n", seqFrames,
                    seqDir.c_str(), seqWarm);
        return 0;
    }
#endif

    RendererSettings rendererSettings(*renderer);
    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        if (!bridge.connected() && !brownout) {
            ImGui::SetNextWindowPos({ImGui::GetIO().DisplaySize.x * 0.5f,
                                     ImGui::GetIO().DisplaySize.y * 0.14f},
                                    ImGuiCond_Always, {0.5f, 0.5f});
            ImGui::Begin("Waiting for ArduPilot SITL", nullptr,
                         ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::Text("Listening on UDP :%u ...", port);
            ImGui::Separator();
            ImGui::TextUnformatted("In WSL2:");
            ImGui::TextUnformatted("  HOST_IP=$(ip route show default | cut -d \" \" -f3)");
            ImGui::TextUnformatted("  sim_vehicle.py -v ArduCopter -f JSON:$HOST_IP --console");
            ImGui::Separator();
            ImGui::TextDisabled("If it never connects: allow inbound UDP %u in Windows Firewall.", port);
            ImGui::TextDisabled("Full runbook: doc/ardupilot_sitl.md");
            ImGui::End();
            return;
        }

        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
        ImGui::Begin("SITL telemetry", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        if (brownout) {
            ImGui::TextUnformatted("SCRIPTED LANDING LOOP (--brownout)");
        } else {
            ImGui::Text("peer         %s", bridge.peer().c_str());
            ImGui::Text("rate         %u Hz sim / %u Hz asked / %.0f Hz achieved",
                        simRateHz, lastServo.frameRate, bridge.achievedRateHz());
            ImGui::Text("frame        %u   (resets: %u)", bridge.lastFrameCount(), resets);
        }
        ImGui::Text("sim time     %.2f s", lastState.timestampSec);
        ImGui::Separator();
        if (std::isnan(lastState.rangefinderM)) {
            if (dust && dust->opticalDepthBelow() > 0.4f) {
                ImGui::TextColored({1.f, 0.55f, 0.25f, 1.f},
                                   "alt AGL      --  DUST DROPOUT (tau %.1f)",
                                   dust->opticalDepthBelow());
            } else {
                ImGui::TextUnformatted("alt AGL      --  (out of range)");
            }
        } else {
            ImGui::Text("alt AGL      %.2f m", lastState.rangefinderM);
        }
        ImGui::Text("alt home     %.2f m", -lastState.positionNed[2]);
        ImGui::Text("roll/pitch   %+.1f / %+.1f deg", lastState.attitudeRpy[0] * 57.2958,
                    lastState.attitudeRpy[1] * 57.2958);
        ImGui::Text("yaw          %.1f deg", lastState.attitudeRpy[2] * 57.2958);
        ImGui::Text("vel NED      %+.1f %+.1f %+.1f m/s", lastState.velocityNed[0],
                    lastState.velocityNed[1], lastState.velocityNed[2]);
        if (!std::isnan(lastState.airspeed)) {
            ImGui::Text("airspeed     %.1f m/s", lastState.airspeed);
        }
        if (dust) {
            ImGui::Text("dust         %.0f%%  (%u parcels airborne)",
                        dust->dustiness() * 100.f, dust->airborne());
            // The conserved reservoir: watch the ground under the vehicle
            // run dry over repeated landings.
            ImGui::Text("loose soil   %.0f%% left under the vehicle",
                        100.f * dust->groundDustAt(drone.root()->position.x,
                                                   drone.root()->position.z) /
                                (dust->params().groundDustPerM2 *
                                 dust->params().gridCell * dust->params().gridCell));
        }
        ImGui::Separator();
        ImGui::TextUnformatted("wind");
        ImGui::SliderFloat("m/s", &windSpeed, 0.f, 12.f, "%.1f");
        ImGui::SliderFloat("from deg", &windFromDeg, 0.f, 360.f, "%.0f");
        ImGui::SliderFloat("gust", &windGust, 0.f, 1.f, "%.2f");
        ImGui::Separator();
        const bool armed = lastServo.pwm[0] > 1050 || lastServo.pwm[1] > 1050 ||
                           lastServo.pwm[2] > 1050 || lastServo.pwm[3] > 1050;
        ImGui::Text("motors       %s", armed ? "ARMED" : "disarmed");
        for (int i = 0; i < 4; ++i) {
            char label[16];
            std::snprintf(label, sizeof label, "M%d %u", i + 1, lastServo.pwm[i]);
            ImGui::ProgressBar((lastServo.pwm[i] - 1000.f) / 1000.f, ImVec2(180, 0), label);
        }
        ImGui::Separator();
        // Unlock to leave the camera behind for a fly-past, then re-lock: the
        // rig resumes from wherever the drone is, it does not snap back.
        ImGui::Checkbox("camera follows drone", &followDrone);
        ImGui::Separator();
        rendererSettings.drawCollapsed();
        ImGui::End();
    });

    IOCapture cap;
    cap.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&cap);

    // ── Live capture of a SITL flight (--rec DIR) ───────────────────────────
    // --seq cannot film this one: it flies the SCRIPTED loop, headless, at a
    // fixed dt and returns before this render loop exists. A real mission is
    // whatever ArduPilot decides to do, so the recorder rides the render loop
    // and subsamples it — every --recevery-th drawn frame, ~10 fps against a
    // 60 Hz vsync, which is a video rather than a disk-throughput experiment.
    //
    // ARMED at startup, STARTED by the vehicle first leaving the ground: a
    // mission spends its opening minutes in pre-arm checks and GPS lock, and
    // the whole frame budget would go to a parked quad on the pad.
    int recTick = 0, recWritten = 0;
    bool recArmed = !recDir.empty(), recRunning = false;
    if (recArmed) std::filesystem::create_directories(recDir);

    Clock clock;
    canvas.animate([&] {
        const float dtRender = clock.getDelta();

        // Waypoint-marker channel (one datagram per mission upload; cheap poll).
        {
            char buf[2048];
            const auto n = ::recvfrom(markerSock, buf, sizeof buf - 1, 0, nullptr, nullptr);
            if (n >= 0) {
                buf[n] = '\0';
                rebuildMarkers(buf);
            }
        }

        // Wind blows FROM windFromDeg (compass, 0 = from north), toward the
        // opposite heading — the meteorological convention. In brownout mode
        // the quad keeps calm air (see visualWindAt's note).
        if (!brownout) {
            const float toRad = math::degToRad(windFromDeg + 180.f);
            quad->setWind(uav::frame::nedToTp(windSpeed * std::cos(toRad),
                                               windSpeed * std::sin(toRad), 0.0),
                          windGust);
        }

        if (brownout) {
            // Scripted landing loop: advance the sim by the render frame's own
            // time (clamped so a window drag doesn't fast-forward the flight).
            const double target =
                    lastState.timestampSec + std::min(static_cast<double>(dtRender), 0.05);
            while (lastState.timestampSec < target) scriptedStep();
        } else {
            // Service the SITL link under a time budget: each Frame is one
            // lock-step physics substep + reply. Lock-step means there is never
            // more than one packet in flight — the next one only arrives AFTER
            // our reply — so on an empty socket we spin briefly (the turnaround
            // is ~100 µs) instead of breaking, or the sim would be capped at
            // one physics step per vsync. A 1.5 ms silence means SITL is
            // idle/slow; then we stop burning the budget and go render.
            const double budgetEnd = nowSec() + 0.008;
            double lastActivity = nowSec();
            uav::ServoInput servo{};
            while (nowSec() < budgetEnd) {
                const auto ev = bridge.poll(servo);
                if (ev == uav::SitlBridge::Event::None) {
                    if (!bridge.connected() || nowSec() - lastActivity > 0.0015) break;
                    continue;
                }
                lastActivity = nowSec();
                if (ev == uav::SitlBridge::Event::Reset) {
                    quad->reset();
                    ++resets;
                }
                quad->step(servo, lastState);
                degradeRangefinder(servo.frameCount);
                serviceProximity();
                bridge.sendState(lastState);
                lastServo = servo;
            }
        }

        // Visuals: the physics binding already synced the drone mesh; spin the
        // rotors and keep the chase camera on target.
        float levels[4] = {};
        if (quad) {
            for (int i = 0; i < 4; ++i) levels[i] = quad->motorLevel(i);
        }
        drone.setMotors(levels, dtRender);
        updateDust();

        // Make the wind visible: windsock attitude + advecting air streaks,
        // both reading the sim's instantaneous gusty wind.
        {
            const Vector3 w = brownout
                                      ? visualWindAt(static_cast<float>(lastState.timestampSec))
                                      : quad->windNow();
            const float speed = std::hypot(w.x, w.z);

            if (speed > 0.05f) {
                sockYaw->rotation.y = std::atan2(-w.z, w.x);
            }
            const float fly = std::min(speed / 8.f, 1.f);
            const float flutter = windGust * 0.07f *
                                  std::sin(static_cast<float>(clock.elapsedTime) * 7.f);
            sockDroop->rotation.z = -(1.f - fly) * 1.4f + flutter * fly;

            const bool showStreaks = speed > 0.5f;
            const Vector3 center = drone.root()->position;
            const float yawStreak = std::atan2(-w.z, w.x);
            for (auto& s : streaks) {
                s.mesh->visible = showStreaks;
                if (!showStreaks) continue;
                s.pos.x += w.x * s.speedJitter * dtRender;
                s.pos.z += w.z * s.speedJitter * dtRender;
                // Wrap inside the bubble that follows the drone.
                const auto wrap = [](float v, float c, float half) {
                    while (v - c > half) v -= 2 * half;
                    while (v - c < -half) v += 2 * half;
                    return v;
                };
                s.pos.x = wrap(s.pos.x, center.x, streakBubble);
                s.pos.y = wrap(s.pos.y, std::max(center.y, 20.f), 40.f);
                s.pos.z = wrap(s.pos.z, center.z, streakBubble);
                s.mesh->position.copy(s.pos);
                s.mesh->rotation.y = yawStreak;
                s.mesh->scale.x = 1.3f + 0.9f * speed;
            }
        }

        // The sun rig follows the drone so the tight shadow frustum always
        // covers the action.
        sunTarget->position.copy(drone.root()->position);
        sun->position.copy(sunDir).multiplyScalar(300.f).add(sunTarget->position);

        // Chase: RIGID follow, not a leash. Each frame the drone's displacement
        // since the last frame is added to BOTH the orbit target and the camera
        // eye, which translates the whole rig without touching the vector
        // between them. That vector is exactly what OrbitControls encodes, so
        // every part of the user's framing survives the vehicle moving under it
        // -- orbit angle, dolly distance, AND any right-drag pan offset, since
        // panning displaces target and eye together and the shared translation
        // preserves that too.
        //
        // The old code instead slammed `target = dronePos` every frame, which
        // discarded pan outright and let the camera fall behind until a 14 m
        // leash yanked it; on video that reads as a stutter every time the
        // drone accelerates. Deltas also make a SITL restart free: the vehicle
        // teleports home and the camera rides along, keeping the shot.
        {
            const Vector3& dronePos = drone.root()->position;
            if (followDrone) {
                Vector3 delta = dronePos;
                delta.sub(chaseAnchor);
                camera.position.add(delta);
                controls.target.add(delta);
            }
            // Tracked even while unlocked, so re-enabling follow resumes from
            // where the drone is now instead of teleporting the camera by the
            // whole distance it covered meanwhile.
            chaseAnchor.copy(dronePos);
        }
        controls.update();

        renderer->render(scene, camera);
        renderedOnce = true;// the TLAS exists now; the traced fan may fire
        ui.render();

        if (recArmed) {
            // The HUD's "alt home" (NED down, negated) is the trigger, NOT
            // rng_1: rng_1 is the DEGRADED rangefinder and goes NaN inside the
            // takeoff dust — the exact instant the trigger is waiting for.
            const double alt = -lastState.positionNed[2];
            if (!recRunning && std::isfinite(alt) && alt > 1.0) {
                recRunning = true;
                std::fprintf(stderr, "[rec] airborne — recording every %d frames to %s (max %d)\n",
                             recEvery, recDir.c_str(), recFrames);
            }
            if (recRunning && recTick++ % recEvery == 0) {
                char name[64];
                std::snprintf(name, sizeof name, "f%05d.png", recWritten);
                try {
                    // The renderer's OWN readback + PNG writer, the same one
                    // --seq uses: it sizes the write from framebufferSize(),
                    // which is the only shape the pixel buffer ever has (the
                    // window's idea of its size is a request the platform may
                    // decline). A second readback here would re-earn that bug.
                    renderer->writeFramebuffer(std::filesystem::path(recDir) / name);
                    ++recWritten;
                } catch (const std::exception& e) {
                    // A capture that fails must not take the flight down with
                    // it — SITL is on the other end of a lock-step link.
                    std::fprintf(stderr, "[rec] stopped: %s\n", e.what());
                    recArmed = recRunning = false;
                }
                if (recWritten >= recFrames) {
                    recArmed = recRunning = false;
                    std::fprintf(stderr, "[rec] %d frames -> %s (cap reached)\n", recWritten,
                                 recDir.c_str());
                }
            }
        }
    });

    return 0;
}
