// ArduPilot SITL demo — threepp as the physics backend and visualizer.
//
// ArduCopter SITL (typically in WSL2, `sim_vehicle.py -v ArduCopter
// -f JSON:<this-machine>`) sends servo PWM to UDP :9002; we step a PhysX quad
// one substep per packet and reply with the vehicle state that drives SITL's
// own clock (lock-step). Full runbook: doc/ardupilot_sitl.md.
//
//   ardupilot_sitl [--port N]     interactive demo (default port 9002)
//   ardupilot_sitl --selftest     headless fake-SITL loopback check, exit 0/1
//
// The render loop and the physics never race: each animate frame drains the
// socket under a time budget (SITL waits on our reply — that's the protocol —
// so rendering simply pauses the autopilot for a frame's worth of wall time),
// then draws the latest state.

#include "DroneVisual.hpp"
#include "FrameConv.hpp"
#include "QuadSim.hpp"
#include "SitlBridge.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

using namespace threepp;

namespace {

    double nowSec() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
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

        sitl::SitlBridge bridge(0);// ephemeral port; never clashes with a live 9002
        check(bridge.valid(), "bridge binds an ephemeral UDP port");

        sitl::DroneVisual visual;
        visual.root()->position.y = sitl::DroneVisual::hullY / 2.f;
        sitl::QuadSim quad(rate, *visual.root(), nullptr);// flat plane ground

        // Fake SITL's own socket.
#ifdef _WIN32
        SOCKET tx = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        u_long nb = 1;
        ioctlsocket(tx, FIONBIO, &nb);
#else
        int tx = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
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
            sitl::ServoInput in{};
            in.frameRate = rate;
            in.frameCount = ++frameCount;
            in.channels = channels;
            in.pwm[0] = pwm0;
            in.pwm[1] = pwm1;
            in.pwm[2] = pwm2;
            in.pwm[3] = pwm3;
            std::uint8_t buf[128];
            const int n = sitl::SitlBridge::encode(in, buf);
            ::sendto(tx, reinterpret_cast<const char*>(buf), n, 0,
                     reinterpret_cast<const sockaddr*>(&to), sizeof to);

            const double deadline = nowSec() + 1.0;
            sitl::ServoInput servo{};
            while (nowSec() < deadline) {
                const auto ev = bridge.poll(servo);
                if (ev == sitl::SitlBridge::Event::None) continue;
                if (ev == sitl::SitlBridge::Event::Reset) quad.reset();
                sitl::FdmState state{};
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
        sitl::ServoInput servo{};
        {
            sitl::ServoInput in{};
            in.frameRate = rate;
            in.frameCount = 1;
            in.channels = 16;
            for (auto& p : in.pwm) p = 1000;
            std::uint8_t buf[128];
            const int n = sitl::SitlBridge::encode(in, buf);
            ::sendto(tx, reinterpret_cast<const char*>(buf), n, 0,
                     reinterpret_cast<const sockaddr*>(&to), sizeof to);
            const double deadline = nowSec() + 1.0;
            auto ev = sitl::SitlBridge::Event::None;
            while (nowSec() < deadline &&
                   (ev = bridge.poll(servo)) == sitl::SitlBridge::Event::None) {}
            check(ev == sitl::SitlBridge::Event::Reset, "frame_count rollback detected as restart");
            quad.reset();
            sitl::FdmState state{};
            quad.step(servo, state);
            bridge.sendState(state);
            check(std::abs(state.positionNed[2]) < 0.02, "reset re-homes the vehicle");
            frameCount = 1;
        }

        // 5: 32-channel packet accepted.
        check(exchange(1000, 1000, 1000, 1000, 32), "32-channel packet (magic 29569) accepted");

#ifdef _WIN32
        closesocket(tx);
#else
        ::close(tx);
#endif
        std::printf("[selftest] %s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }

}// namespace

int main(int argc, char** argv) {

    std::uint16_t port = 9002;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") return runSelftest();
        if (a == "--port" && i + 1 < argc) port = static_cast<std::uint16_t>(std::atoi(argv[++i]));
    }

    Canvas canvas("threepp - ArduPilot SITL", {{"aa", 4}});
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    Scene scene;

    // Sky + sun. The Preetham Sky dome is a GL ShaderMaterial the Vulkan
    // deferred renderer can't shade — there it would render as a black sphere
    // enclosing the camera (the whole frame goes black). On Vulkan use a flat
    // sky-color background instead (the mountains demo's fallback idiom); the
    // volumetric-cloud/env-map treatment is the future showpiece pass.
    const bool vulkan = canvas.graphicsApi() == GraphicsAPI::Vulkan;
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
        scene.background = Color(0.55f, 0.72f, 0.92f);
    }

    auto sun = DirectionalLight::create(Color(1.f, 0.96f, 0.90f), 2.2f);
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
    scene.add(AmbientLight::create(Color(0.55f, 0.65f, 0.8f), 0.35f));

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
    {
        // Level a helipad apron around home: within 4 m the ground is exactly
        // h0 (so the pad never pokes through a rising slope), blending back to
        // the raw terrain by 16 m. The physics trimesh is built from this same
        // geometry, so contacts match the picture.
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
    auto terrainMesh = Mesh::create(terrainGeo, terrainMat);
    terrainMesh->receiveShadow = true;
    // Home ground exactly at y=0 so NED altitude 0 = spawn ground level.
    terrainMesh->position.y = -h0;
    scene.add(terrainMesh);

    // Landing pad at home.
    auto pad = Mesh::create(CylinderGeometry::create(1.6f, 1.6f, 0.04f),
                            MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                                 .color(Color(0.75f, 0.16f, 0.12f))
                                                                 .roughness(0.8f)));
    pad->position.y = 0.02f;
    pad->receiveShadow = true;
    scene.add(pad);

    // Drone.
    sitl::DroneVisual drone;
    drone.root()->position.y = 0.04f + sitl::DroneVisual::hullY / 2.f;// on the pad
    scene.add(drone.root());

    PerspectiveCamera camera(60.f, canvas.aspect(), 0.1f, 10000.f);
    camera.position.set(2.2f, 1.3f, 2.9f);
    OrbitControls controls{camera, canvas};

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
    sitl::SitlBridge bridge(port);
    if (!bridge.valid()) {
        std::fprintf(stderr, "[sitl] failed to bind UDP port %u — is another instance running?\n", port);
        return 1;
    }
    auto quad = std::make_unique<sitl::QuadSim>(simRateHz, *drone.root(), terrainMesh.get());
    quad->world().addStaticTrimesh(*pad);// land flush on the pad, not 4 cm inside it
    sitl::FdmState lastState{};
    sitl::ServoInput lastServo{};
    std::uint32_t resets = 0;

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        if (!bridge.connected()) {
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
        ImGui::Text("peer         %s", bridge.peer().c_str());
        ImGui::Text("rate         %u Hz sim / %u Hz asked / %.0f Hz achieved",
                    simRateHz, lastServo.frameRate, bridge.achievedRateHz());
        ImGui::Text("frame        %u   (resets: %u)", bridge.lastFrameCount(), resets);
        ImGui::Text("sim time     %.2f s", lastState.timestampSec);
        ImGui::Separator();
        if (std::isnan(lastState.rangefinderM)) {
            ImGui::TextUnformatted("alt AGL      --  (out of range)");
        } else {
            ImGui::Text("alt AGL      %.2f m", lastState.rangefinderM);
        }
        ImGui::Text("alt home     %.2f m", -lastState.positionNed[2]);
        ImGui::Text("roll/pitch   %+.1f / %+.1f deg", lastState.attitudeRpy[0] * 57.2958,
                    lastState.attitudeRpy[1] * 57.2958);
        ImGui::Text("yaw          %.1f deg", lastState.attitudeRpy[2] * 57.2958);
        ImGui::Text("vel NED      %+.1f %+.1f %+.1f m/s", lastState.velocityNed[0],
                    lastState.velocityNed[1], lastState.velocityNed[2]);
        ImGui::Separator();
        const bool armed = lastServo.pwm[0] > 1050 || lastServo.pwm[1] > 1050 ||
                           lastServo.pwm[2] > 1050 || lastServo.pwm[3] > 1050;
        ImGui::Text("motors       %s", armed ? "ARMED" : "disarmed");
        for (int i = 0; i < 4; ++i) {
            char label[16];
            std::snprintf(label, sizeof label, "M%d %u", i + 1, lastServo.pwm[i]);
            ImGui::ProgressBar((lastServo.pwm[i] - 1000.f) / 1000.f, ImVec2(180, 0), label);
        }
        ImGui::End();
    });

    Clock clock;
    canvas.animate([&] {
        const float dtRender = clock.getDelta();

        // Service the SITL link under a time budget: each Frame is one
        // lock-step physics substep + reply. Lock-step means there is never
        // more than one packet in flight — the next one only arrives AFTER our
        // reply — so on an empty socket we spin briefly (the turnaround is
        // ~100 µs) instead of breaking, or the sim would be capped at one
        // physics step per vsync. A 1.5 ms silence means SITL is idle/slow;
        // then we stop burning the budget and go render.
        const double budgetEnd = nowSec() + 0.008;
        double lastActivity = nowSec();
        sitl::ServoInput servo{};
        while (nowSec() < budgetEnd) {
            const auto ev = bridge.poll(servo);
            if (ev == sitl::SitlBridge::Event::None) {
                if (!bridge.connected() || nowSec() - lastActivity > 0.0015) break;
                continue;
            }
            lastActivity = nowSec();
            if (ev == sitl::SitlBridge::Event::Reset) {
                quad->reset();
                ++resets;
            }
            quad->step(servo, lastState);
            bridge.sendState(lastState);
            lastServo = servo;
        }

        // Visuals: the physics binding already synced the drone mesh; spin the
        // rotors and keep the chase camera on target.
        float levels[4] = {};
        if (quad) {
            for (int i = 0; i < 4; ++i) levels[i] = quad->motorLevel(i);
        }
        drone.setMotors(levels, dtRender);

        // The sun rig follows the drone so the tight shadow frustum always
        // covers the action.
        sunTarget->position.copy(drone.root()->position);
        sun->position.copy(sunDir).multiplyScalar(300.f).add(sunTarget->position);

        // Chase: orbit stays user-controlled, but the camera is towed along
        // whenever the drone pulls further than a leash length away.
        controls.target.copy(drone.root()->position);
        {
            Vector3 toCam = camera.position;
            toCam.sub(controls.target);
            const float dist = toCam.length();
            constexpr float leash = 14.f;
            if (dist > leash) {
                toCam.multiplyScalar(leash / dist);
                camera.position.copy(controls.target).add(toCam);
            }
        }
        controls.update();

        renderer->render(scene, camera);
        ui.render();
    });

    return 0;
}
