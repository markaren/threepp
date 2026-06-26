// Native C++ port of the Spot flat-ground deploy scene + controller, mirroring
// python/examples/spot/spot_deploy.py (build_spot + SpotController) so a trained
// Isaac Lab Spot policy drives the robot with no Python/torch in the loop. The
// policy forward pass is SpotPolicy (SpotPolicy.hpp); everything here is the
// physics build and the exact observation/action contract around it.
//
// World is Z-up (gravity (0,0,-9.81)), matching Isaac and the Python deploy.
// The robot is a reduced-coordinate PhysX articulation of a box torso + 4 legs
// (hip/upper/lower capsules), built from the SAME hardcoded kinematics and PD
// gains as the Python version. No URDF/assets are needed — the primitive
// colliders ARE the robot.
//
// Joint ordering, the subtle part: the policy ("ISAAC" order) groups joints by
// TYPE (all hip-x, then all hip-y, then all knee), while we add links per-leg
// ("ADD" order). The two index maps below convert between them; the observation
// is built in ISAAC order, the drive targets are written back in ADD order.

#ifndef THREEPP_EXAMPLES_SPOT_SCENE_HPP
#define THREEPP_EXAMPLES_SPOT_SCENE_HPP

#include "SpotPolicy.hpp"

#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/loaders/OBJLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace spot {

    // ── Isaac Spot velocity-task contract ──────────────────────────────────
    constexpr float ACTION_SCALE = 0.2f;
    constexpr float Z0 = 0.72f;// build height (straight-leg stance just above ground)

    // Default joint pose in ISAAC (type-grouped) order: [hx×4, hy×4, kn×4].
    // hips: left +0.1 / right -0.1; thighs: front 0.9 / hind 1.1; knees -1.5.
    constexpr std::array<float, 12> DEFAULT_Q{
            0.1f, -0.1f, 0.1f, -0.1f, 0.9f, 0.9f, 1.1f, 1.1f, -1.5f, -1.5f, -1.5f, -1.5f};
    // ISAAC index i -> its slot in ADD (per-leg) order, and the inverse.
    constexpr std::array<int, 12> ISAAC_TO_ADD{0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    constexpr std::array<int, 12> ADD_TO_ISAAC{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};

    // Body->world rotation from a quaternion [x,y,z,w] (row-major 3x3),
    // matching spot_deploy.py:_quat_to_R exactly.
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
        std::vector<std::shared_ptr<threepp::Mesh>> meshes;// torso, then per leg: hip, upper, lower
    };

    namespace detail {
        // A capsule link mesh oriented so its (Y-aligned) axis points along `dir`,
        // centred at `center`. Mirrors spot_deploy.py:_capsule; `volOut` is the
        // capsule volume used to back out a density from a target mass.
        inline std::shared_ptr<threepp::Mesh> capsule(float length, float radius,
                                                      const threepp::Vector3& center,
                                                      const threepp::Vector3& dir, int color, float& volOut) {
            using namespace threepp;
            auto mat = MeshStandardMaterial::create();
            mat->color = Color(color);
            auto m = Mesh::create(CapsuleGeometry::create(radius, length), mat);
            m->position.copy(center);
            Vector3 d;
            d.copy(dir).normalize();
            const Vector3 yax(0, 1, 0);
            Vector3 axis;
            axis.crossVectors(yax, d);
            const float s = axis.length();
            const float c = yax.dot(d);
            if (s < 1e-8f) {
                if (c < 0) m->rotation.z = math::PI;// antiparallel: flip
            } else {
                axis.normalize();
                m->quaternion.setFromAxisAngle(axis, std::atan2(s, c));
            }
            volOut = math::PI * radius * radius * length + 4.f / 3.f * math::PI * radius * radius * radius;
            return m;
        }

        // Parent a URDF visual mesh (link_models/<name>.obj, authored in the link
        // frame) under its bound collider so it tracks the physics, then hide the
        // collider's primitive. Mirrors spot_deploy.py:_attach_obj. Spot's visual
        // origins are identity, so the link frame is (linkPos, no rotation). Missing
        // OBJs are tolerated — the primitive collider just stays visible.
        inline void attachObj(threepp::Mesh& collider, const threepp::Vector3& linkPos,
                              const std::string& name, int color, const std::string& assetsDir) {
            using namespace threepp;
            std::shared_ptr<Group> grp;
            try {
                grp = OBJLoader().load(std::filesystem::path(assetsDir) / "link_models" / (name + ".obj"), false);
            } catch (...) {
                grp = nullptr;
            }
            if (!grp) return;

            // local = collider_world^-1 · link_frame_world
            Matrix4 cw, lf, invCw, loc;
            cw.compose(collider.position, collider.quaternion, Vector3(1, 1, 1));
            lf.compose(linkPos, Quaternion(), Vector3(1, 1, 1));
            invCw.copy(cw).invert();
            loc.multiplyMatrices(invCw, lf);
            Vector3 p, s;
            Quaternion q;
            loc.decompose(p, q, s);
            grp->position.copy(p);
            grp->quaternion.copy(q);

            grp->traverseType<Mesh>([&](Mesh& o) {
                auto mat = MeshStandardMaterial::create();
                mat->color = Color(color);
                mat->roughness = 0.5f;
                mat->metalness = 0.0f;
                o.setMaterial(mat);
                o.castShadow = true;
            });
            collider.add(grp);
            if (auto m = collider.material()) m->visible = false;// hide the primitive; the OBJ renders
            collider.castShadow = false;                        // only the OBJ casts (avoid a double shadow)
        }
    }// namespace detail

    // Build Spot as a free-base PhysX articulation at (baseX, baseY). Mirrors
    // spot_deploy.py:build_spot. When `assetsDir` is given (a folder with
    // link_models/*.obj, e.g. ~/.cache/threepp/spot), each link's URDF visual mesh
    // is parented under its collider so Spot renders as the real robot; otherwise
    // the primitive colliders are shown and cast shadows.
    inline SpotRobot buildSpot(threepp::PhysxWorld& world, float baseX = 0.f, float baseY = 0.f,
                               const std::string& assetsDir = "") {
        using namespace threepp;

        // kinematics + Isaac PD gains (stiffness, damping, effort) + link masses
        constexpr float HIP_X = 0.29785f, HIP_Y = 0.055f, HY_Y = 0.1108f;
        constexpr float MASS_BASE = 13.0f, MASS_HIP = 1.2f, MASS_ULEG = 2.0f, MASS_LLEG = 0.55f;
        const float ox = baseX, oy = baseY;

        SpotRobot robot;
        robot.art = std::make_unique<Articulation>(world, /*fixedBase*/ false,
                                                   /*solverPositionIters*/ 12, /*disableSelfCollision*/ true);
        Articulation& art = *robot.art;

        auto bmat = MeshStandardMaterial::create();
        bmat->color = Color(0xffc24d);
        auto bm = Mesh::create(BoxGeometry::create(0.70f, 0.18f, 0.19f), bmat);
        bm->position.set(ox, oy, Z0);
        const float baseDensity = MASS_BASE / (0.70f * 0.18f * 0.19f);
        ArticulationLink base = art.addLink(nullptr, *bm, baseDensity, {1, 0, 0}, {0, 0, 0},
                                            false, 0, 0, 0, 0, 1e6f, 0, "revolute", 0.f, nullptr);
        robot.meshes.push_back(bm);
        if (!assetsDir.empty()) detail::attachObj(*bm, Vector3(ox, oy, Z0), "base", 0xffc24d, assetsDir);

        // legs front/left signs (sx, sy); joints added per-leg => ADD order.
        struct Leg {
            const char* name;
            float sx, sy;
        };
        const std::array<Leg, 4> legs{Leg{"fl", +1, +1}, Leg{"fr", +1, -1}, Leg{"hl", -1, +1}, Leg{"hr", -1, -1}};
        for (const Leg& L : legs) {
            const float sx = L.sx, sy = L.sy;
            Vector3 Jhx(ox + sx * HIP_X, oy + sy * HIP_Y, Z0);
            Vector3 Jhy;
            Jhy.copy(Jhx).add(Vector3(0, sy * HY_Y, 0));
            Vector3 Jkn;
            Jkn.copy(Jhy).add(Vector3(0.025f, 0.f, -0.32f));
            Vector3 Jft;
            Jft.copy(Jkn).add(Vector3(0.f, 0.f, -0.34f));

            const float hxDef = sy > 0 ? 0.1f : -0.1f;
            const float hyDef = sx > 0 ? 0.9f : 1.1f;

            float hv = 0, uv = 0, lv = 0;
            Vector3 mid, dir;
            auto hm = detail::capsule(0.06f, 0.045f, mid.addVectors(Jhx, Jhy).multiplyScalar(0.5f),
                                      dir.subVectors(Jhy, Jhx), 0x303030, hv);
            ArticulationLink hip = art.addLink(&base, *hm, MASS_HIP / hv, {1, 0, 0}, {Jhx.x, Jhx.y, Jhx.z},
                                               true, -0.7854f, 0.7854f, 60.f, 1.5f, 45.f, hxDef, "revolute", 0.f, nullptr);

            auto um = detail::capsule(0.30f, 0.045f, mid.addVectors(Jhy, Jkn).multiplyScalar(0.5f),
                                      dir.subVectors(Jkn, Jhy), 0xffc24d, uv);
            ArticulationLink uleg = art.addLink(&hip, *um, MASS_ULEG / uv, {0, 1, 0}, {Jhy.x, Jhy.y, Jhy.z},
                                                true, -0.8988f, 2.295f, 60.f, 1.5f, 45.f, hyDef, "revolute", 0.f, nullptr);

            auto lm = detail::capsule(0.30f, 0.028f, mid.addVectors(Jkn, Jft).multiplyScalar(0.5f),
                                      dir.subVectors(Jft, Jkn), 0x303030, lv);
            art.addLink(&uleg, *lm, MASS_LLEG / lv, {0, 1, 0}, {Jkn.x, Jkn.y, Jkn.z},
                        true, -2.7929f, -0.247f, 60.f, 1.5f, 115.f, -1.5f, "revolute", 0.f, nullptr);

            robot.meshes.push_back(hm);
            robot.meshes.push_back(um);
            robot.meshes.push_back(lm);

            if (!assetsDir.empty()) {
                detail::attachObj(*hm, Jhx, std::string(L.name) + ".hip", 0x303030, assetsDir);
                detail::attachObj(*um, Jhy, std::string(L.name) + ".uleg", 0xffc24d, assetsDir);
                detail::attachObj(*lm, Jkn, std::string(L.name) + ".lleg", 0x303030, assetsDir);
            }
        }
        art.finalize();
        // Cast shadows from whatever is actually visible: a still-visible primitive
        // collider (no OBJ attached) casts; where attachObj hid the primitive, the
        // OBJ it parented is the caster instead. Robust to a missing assets folder.
        for (auto& m : robot.meshes)
            if (const auto mat = m->material(); mat && mat->visible) m->castShadow = true;
        return robot;
    }

    // Builds the 48-d Isaac observation, runs the policy, applies joint targets.
    // Mirrors spot_deploy.py:SpotController (deterministic act_mean for deploy).
    class SpotController {
    public:
        SpotController(threepp::Articulation& art, const SpotPolicy& policy)
            : art_(art), policy_(policy) { last_.fill(0.f); }

        // 48-d obs: [lin_b(3), ang_b(3), proj_g(3), cmd(3), qpos(12), qvel(12), last_action(12)],
        // velocities + gravity rotated into the body frame; qpos/qvel/action in ISAAC order.
        [[nodiscard]] std::array<float, 48> obs(const std::array<float, 3>& cmd) const {
            const auto rs = art_.rootState();    // [px,py,pz, qx,qy,qz,qw]
            const auto rv = art_.rootVelocity();// [vx,vy,vz, wx,wy,wz]
            float R[3][3];
            quatToR(rs[3], rs[4], rs[5], rs[6], R);// body->world; R^T = world->body

            std::array<float, 48> o{};
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

            const auto jp = art_.jointPositions();// ADD order
            const auto jv = art_.jointVelocities();
            for (int i = 0; i < 12; ++i) o[k++] = jp[ISAAC_TO_ADD[i]] - DEFAULT_Q[i];
            for (int i = 0; i < 12; ++i) o[k++] = jv[ISAAC_TO_ADD[i]];
            for (int i = 0; i < 12; ++i) o[k++] = last_[i];
            return o;
        }

        // One control tick: obs -> policy -> drive targets -> step 0.02 s
        // (10 × 0.002 substeps, matching Isaac's decimation).
        void step(threepp::PhysxWorld& world, const std::array<float, 3>& cmd) {
            const auto o = obs(cmd);
            const std::vector<float> a = policy_.act(o.data(), o.size());// 12, ISAAC order
            for (int i = 0; i < 12; ++i) last_[i] = a[i];
            float tgt[12];// drive targets in ADD order
            for (int i = 0; i < 12; ++i) {
                const int j = ADD_TO_ISAAC[i];
                tgt[i] = DEFAULT_Q[j] + ACTION_SCALE * a[j];
            }
            art_.setDriveTargets(tgt, 12);
            world.step(0.02f);
        }

        // Hold the default stand pose for n ticks (settle on spawn).
        void hold(threepp::PhysxWorld& world, int n) {
            float tgt[12];
            for (int i = 0; i < 12; ++i) tgt[i] = DEFAULT_Q[ADD_TO_ISAAC[i]];
            for (int k = 0; k < n; ++k) {
                art_.setDriveTargets(tgt, 12);
                world.step(0.02f);
            }
        }

        [[nodiscard]] const std::array<float, 12>& lastAction() const { return last_; }

    private:
        threepp::Articulation& art_;
        const SpotPolicy& policy_;
        std::array<float, 12> last_{};
    };

}// namespace spot

#endif// THREEPP_EXAMPLES_SPOT_SCENE_HPP
