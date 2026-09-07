// Native C++ port of the Spot deploy scene + controller, mirroring
// python/examples/spot/play_spot_steps.py (the plant, and the exact 96-d
// observation/action contract around it) so a trained policy drives the robot with
// no Python and no torch in the loop. The forward pass is SpotPolicy
// (SpotPolicy.hpp); everything here is the physics build and the contract.
//
// World is Z-up (gravity (0,0,-9.81)), matching the trainer and the Python deploy,
// so the URDF's own frame is the world's frame and no rotation is applied.
//
// THE PLANT COMES FROM THE URDF. threepp_data ships urdf/spot with <collision> and
// <inertial> on every link (see that directory's NOTICE), so the robot is
// loadArticulation() over that file rather than kinematics retyped here. Before,
// this header hardcoded the hip offsets, capsule sizes and link masses — a third
// copy of numbers that also live in spot_deploy.build_spot and in the URDF, kept in
// step by hand. One consequence worth naming: loadArticulation applies ONE set of
// PD gains to every joint, so the per-joint effort ceiling the hand-built version
// had (45 on the hips, 115 on the knees) is now 115 everywhere. It is a ceiling,
// not a target, and the policy does not saturate the hips.
//
// JOINT ORDER, the subtle part. The policy speaks "ISAAC" order: joints grouped by
// TYPE (all hip-x, then all hip-y, then all knee). A simulator's DOF order is its
// own business — the hand-built version added links per leg and needed a fixed
// permutation, while the URDF articulation happens to report them type-grouped —
// so the mapping is resolved BY NAME at load time and no permutation is baked in.

#ifndef THREEPP_EXAMPLES_SPOT_SCENE_HPP
#define THREEPP_EXAMPLES_SPOT_SCENE_HPP

#include "SpotPolicy.hpp"

#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/physx/UrdfArticulation.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace spot {

    // ── The trained contract (python/examples/spot) ─────────────────────────
    constexpr float ACTION_SCALE = 0.2f;
    constexpr float Z0 = 0.72f;        // build height: straight legs, feet just above ground
    constexpr float GAIT_PERIOD = 0.5f;// phase-clock period (s); = scratch_clock.GAIT_PERIOD
    constexpr float STIFFNESS = 90.f;  // = scratch_env.STIFF_GAINS; Isaac's default 60 sags this plant
    constexpr float DAMPING = 1.5f;
    constexpr float MAX_FORCE = 115.f;// the knee's effort; uniform here (see the header note)

    // The policy's joint order, by URDF joint name: type-grouped, four legs each.
    constexpr std::array<const char*, 12> ISAAC_JOINTS{
            "fl.hx", "fr.hx", "hl.hx", "hr.hx",
            "fl.hy", "fr.hy", "hl.hy", "hr.hy",
            "fl.kn", "fr.kn", "hl.kn", "hr.kn"};
    // Default pose in that order. hips: left +0.1 / right -0.1; thighs: front 0.9 /
    // hind 1.1; knees -1.5.
    constexpr std::array<float, 12> DEFAULT_Q{
            0.1f, -0.1f, 0.1f, -0.1f, 0.9f, 0.9f, 1.1f, 1.1f, -1.5f, -1.5f, -1.5f, -1.5f};

    // ── The 2-D height scan (spot_terrain_env.PROBE_DX / PROBE_DY) ──────────
    // A heading-relative grid: 9 forward offsets x 5 lateral, flattened FORWARD-MAJOR
    // (index = fi*5 + dj), sampled as terrain height MINUS the height under the base.
    // On flat ground every cell is 0, which is what the default height function gives;
    // the grid is spelled out anyway because it is part of the observation contract,
    // and a terrain scene only has to supply a height function.
    constexpr std::array<float, 9> PROBE_DX{-0.35f, -0.15f, 0.05f, 0.2f, 0.35f, 0.5f, 0.7f, 0.9f, 1.1f};
    constexpr std::array<float, 5> PROBE_DY{-0.30f, -0.15f, 0.0f, 0.15f, 0.30f};
    constexpr int N_SCAN = 45;// 9 * 5
    constexpr int OBS_DIM = 3 + 3 + 3 + 3 + 12 + 12 + 12 + 2 + 1 + N_SCAN;// = 96
    constexpr int ACT_DIM = 12;

    // Body->world rotation from a quaternion [x,y,z,w] (row-major 3x3), matching
    // spot_deploy.py:_quat_to_R exactly.
    inline void quatToR(float x, float y, float z, float w, float R[3][3]) {
        R[0][0] = 1 - 2 * (y * y + z * z);
        R[0][1] = 2 * (x * y - z * w);
        R[0][2] = 2 * (x * z + y * w);
        R[1][0] = 2 * (x * y + z * w);
        R[1][1] = 1 - 2 * (x * x + z * z);
        R[1][2] = 2 * (y * z - x * w);
        R[2][0] = 2 * (x * z - y * w);
        R[2][1] = 2 * (y * z + x * w);
        R[2][2] = 1 - 2 * (x * x + y * y);
    }

    struct SpotRobot {
        std::unique_ptr<threepp::Articulation> art;
        std::vector<std::shared_ptr<threepp::Mesh>> meshes;// add these to the scene
        std::vector<std::string> jointNames;               // the articulation's own DOF order
        std::array<int, 12> isaacToSim{};                  // policy index -> DOF index, resolved by name
    };

    // Build Spot from threepp_data's urdf/spot/spot_physics.urdf as a free-base PhysX
    // articulation standing at (baseX, baseY). Throws if the file is unreadable or does
    // not carry the 12 joints the policy drives.
    inline SpotRobot loadSpot(threepp::PhysxWorld& world, const std::filesystem::path& urdf,
                              float baseX = 0.f, float baseY = 0.f) {
        threepp::URDFArticulationOptions opts;
        opts.fixedBase = false;
        opts.basePosition = threepp::Vector3(baseX, baseY, Z0);
        opts.stiffness = STIFFNESS;
        opts.damping = DAMPING;
        opts.maxForce = MAX_FORCE;
        opts.solverPositionIterations = 12;
        opts.selfCollision = false;// the primitive colliders overlap at the joints
        opts.renderVisuals = true; // parent each link's <visual> under its collider

        auto result = threepp::loadArticulation(world, urdf, opts);
        if (!result.articulation) {
            throw std::runtime_error("loadSpot: could not build an articulation from " + urdf.string());
        }

        SpotRobot robot;
        robot.art = std::move(result.articulation);
        robot.meshes = std::move(result.meshes);
        robot.jointNames = std::move(result.jointNames);

        // Resolve the policy's joint order against whatever order this articulation
        // reports. By NAME, never by position: the DOF order is the loader's business
        // (fixed joints collapse, links are walked breadth-first) and a baked-in
        // permutation is how a policy ends up driving the wrong leg.
        for (int i = 0; i < 12; ++i) {
            const auto it = std::find(robot.jointNames.begin(), robot.jointNames.end(),
                                      std::string(ISAAC_JOINTS[i]));
            if (it == robot.jointNames.end()) {
                throw std::runtime_error("loadSpot: " + urdf.string() + " has no joint named '" +
                                         ISAAC_JOINTS[i] + "'");
            }
            robot.isaacToSim[i] = static_cast<int>(std::distance(robot.jointNames.begin(), it));
        }
        // Cast shadows from whatever actually renders. loadArticulation hides a
        // collider's own primitive once it has parented that link's <visual> under it,
        // so the visual is the caster there; a link with no visual keeps its primitive
        // and casts from that. Walking the subtree covers both without asking which
        // happened.
        for (auto& m : robot.meshes) {
            m->traverseType<threepp::Mesh>([](threepp::Mesh& o) {
                const auto mat = o.material();
                o.castShadow = !mat || mat->visible;
            });
        }
        return robot;
    }

    // Builds the 96-d observation, runs the policy, writes joint targets.
    // Mirrors play_spot_steps.v2_obs:
    //   [lin_b(3) | ang_b(3) | proj_g(3) | cmd(3) | qpos(12) | qvel(12) | last_action(12)
    //    | clock(2) | base_above(1) | scan(45)]
    // velocities and gravity rotated into the body frame; qpos/qvel/action in the
    // policy's order; clock = [sin, cos] of the gait phase.
    class SpotController {
    public:
        // `terrainHeight(x, y)` is the ground height under a world point — flat by
        // default. A terrain scene passes its own and the scan follows; nothing else
        // in the observation changes.
        SpotController(const SpotRobot& robot, const SpotPolicy& policy,
                       std::function<float(float, float)> terrainHeight = nullptr)
            : art_(*robot.art), map_(robot.isaacToSim), policy_(policy),
              height_(std::move(terrainHeight)) {
            last_.fill(0.f);
            if (policy_.inputDim() != OBS_DIM) {
                throw std::runtime_error("SpotController: policy expects " +
                                         std::to_string(policy_.inputDim()) + " observations, this contract is " +
                                         std::to_string(OBS_DIM) + " - export the 96-d terrain policy");
            }
        }

        [[nodiscard]] float heightAt(float x, float y) const { return height_ ? height_(x, y) : 0.f; }

        [[nodiscard]] std::array<float, OBS_DIM> obs(const std::array<float, 3>& cmd) const {
            const auto rs = art_.rootState();   // [px,py,pz, qx,qy,qz,qw]
            const auto rv = art_.rootVelocity();// [vx,vy,vz, wx,wy,wz]
            float R[3][3];
            quatToR(rs[3], rs[4], rs[5], rs[6], R);// body->world; R^T = world->body

            std::array<float, OBS_DIM> o{};
            int k = 0;
            for (int i = 0; i < 3; ++i) {// lin_b = R^T · v_lin
                float s = 0;
                for (int j = 0; j < 3; ++j) s += R[j][i] * rv[j];
                o[k++] = s;
            }
            for (int i = 0; i < 3; ++i) {// ang_b = R^T · v_ang
                float s = 0;
                for (int j = 0; j < 3; ++j) s += R[j][i] * rv[3 + j];
                o[k++] = s;
            }
            for (int i = 0; i < 3; ++i) o[k++] = -R[2][i];// proj_g = R^T · (0,0,-1)
            for (int i = 0; i < 3; ++i) o[k++] = cmd[i];

            const auto jp = art_.jointPositions();// the articulation's own DOF order
            const auto jv = art_.jointVelocities();
            for (int i = 0; i < 12; ++i) o[k++] = jp[map_[i]] - DEFAULT_Q[i];
            for (int i = 0; i < 12; ++i) o[k++] = jv[map_[i]];
            for (int i = 0; i < 12; ++i) o[k++] = last_[i];

            const float ang = 2.0f * threepp::math::PI * phi_;
            o[k++] = std::sin(ang);
            o[k++] = std::cos(ang);

            // Heading in the ground plane, from the body +x axis — the frame the scan
            // grid is expressed in (spot_terrain_env.scan_xy).
            float hx = R[0][0], hy = R[1][0];
            const float n = std::hypot(hx, hy);
            if (n > 1e-6f) {
                hx /= n;
                hy /= n;
            } else {
                hx = 1.f;
                hy = 0.f;
            }
            const float h0 = heightAt(rs[0], rs[1]);
            o[k++] = rs[2] - h0;// base height above the ground under it

            for (const float gx : PROBE_DX) {// forward-major, as the trainer flattens it
                for (const float gy : PROBE_DY) {
                    const float px = rs[0] + gx * hx - gy * hy;
                    const float py = rs[1] + gx * hy + gy * hx;
                    o[k++] = std::clamp(heightAt(px, py) - h0, -1.f, 1.f);
                }
            }
            return o;
        }

        // One control tick: obs -> policy -> drive targets -> step 0.02 s
        // (10 × 0.002 substeps, matching Isaac's decimation).
        void step(threepp::PhysxWorld& world, const std::array<float, 3>& cmd) {
            const auto o = obs(cmd);
            const std::vector<float> a = policy_.act(o.data(), o.size());// 12, policy order
            for (int i = 0; i < 12; ++i) last_[i] = a[i];
            std::vector<float> tgt(art_.numDof(), 0.f);
            for (int i = 0; i < 12; ++i) tgt[map_[i]] = DEFAULT_Q[i] + ACTION_SCALE * a[i];
            art_.setDriveTargets(tgt.data(), tgt.size());
            world.step(0.02f);
            // advance the phase clock AFTER the step so phi aligns with the NEXT obs
            phi_ = std::fmod(phi_ + 0.02f / GAIT_PERIOD, 1.0f);
        }

        // Hold the default stand pose for n ticks (settle on spawn).
        void hold(threepp::PhysxWorld& world, int n) {
            phi_ = 0.f;// the settle does NOT advance the clock (matches the training reset)
            std::vector<float> tgt(art_.numDof(), 0.f);
            for (int i = 0; i < 12; ++i) tgt[map_[i]] = DEFAULT_Q[i];
            for (int k = 0; k < n; ++k) {
                art_.setDriveTargets(tgt.data(), tgt.size());
                world.step(0.02f);
            }
        }

        [[nodiscard]] const std::array<float, 12>& lastAction() const { return last_; }

    private:
        threepp::Articulation& art_;
        std::array<int, 12> map_;// policy index -> DOF index
        const SpotPolicy& policy_;
        std::function<float(float, float)> height_;
        std::array<float, 12> last_{};
        float phi_ = 0.f;// gait phase in [0,1), advanced +DT/GAIT_PERIOD each step()
    };

}// namespace spot

#endif// THREEPP_EXAMPLES_SPOT_SCENE_HPP
