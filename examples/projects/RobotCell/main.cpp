// KUKA iiwa robot cell — an interactive pick-and-place demo combining:
//
//   * URDFLoader: the KUKA LBR iiwa 14 R820 model, jointed via Robot.
//   * Numeric IK: damped-least-squares over a finite-difference Jacobian of
//     Robot::computeEndEffectorTransform, with a tool-down orientation
//     constraint, joint speed/range limits, and a null-space rest-posture
//     bias that keeps the redundant elbow DOF in a comfortable elbow-up
//     configuration instead of folding into the pedestal.
//   * Motion planning: the IK never tracks the raw goal. A Cartesian
//     setpoint moves gantry-style (ascend -> traverse at travel height ->
//     descend) at a capped speed, and keep-out volumes (floor, table slab,
//     base column) are enforced on the setpoint, so approaches are vertical
//     and the tool is never commanded through an obstacle.
//   * PhysX: crates are rigid bodies; the environment is static colliders; a
//     kinematic sphere rides the tool tip so the arm can nudge crates; the
//     suction gripper carries a crate by switching its actor kinematic and
//     driving it from the tool frame.
//   * DepthSensor: eye-in-hand depth camera on the flange. The point cloud is
//     not decoration — it is the ONLY source of crate positions. The robot
//     surveys the table from above, clusters the cloud into detections
//     (centroid + measured top height), and picks from those. The controller
//     never reads crate poses from the scene graph; its world knowledge is
//     perception (the cloud), proprioception (its own FK), and the suction
//     cup's vacuum-seal state. Turn the sensor off and the robot is blind.
//   * SVG UI: HUD built from runtime-generated SVG (panels, LEDs, joint-range
//     bars, buttons) + screen-space TextSprites, after examples/loaders/svg_ui.
//
// Interactions:
//   click  take manual control: jog the tool to hover just above the clicked
//          point (disengages AUTO, like grabbing the wheel). With the tool
//          over a crate, G seals the suction; jog elsewhere and G releases —
//          full manual pick-and-place.
//   AUTO   resume the perception loop: survey + clear the table
//   SPACE  spawn a crate on the infeed table
//
// Coordinate conventions: world is metres, Y-up. The URDF is Z-up, so the
// robot root is rotated -90 deg about X. The HUD ortho camera is (0,W,H,0),
// y-up; SVG widget groups are y-flipped (scale.y = -1), as in svg_ui.cpp.

#include "threepp/threepp.hpp"

#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/SpriteInteractor.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/helpers/DepthSensor.hpp"
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/TextSprite.hpp"

#ifdef ROBOT_CELL_WITH_VULKAN
#include "threepp/helpers/PathTracedLidarSensor.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <PxPhysicsAPI.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace threepp;
using namespace ::physx;

namespace {

    // ---- palette -----------------------------------------------------------
    constexpr int kPanel = 0x0e1b2a;
    constexpr int kPanelEdge = 0x1d3b57;
    constexpr int kAccent = 0x35c2ff;
    constexpr int kBtn = 0x16314a;
    constexpr int kBtnHover = 0x21496b;
    constexpr int kBtnActive = 0x35c2ff;
    constexpr int kGood = 0x47e07a;
    constexpr int kWarn = 0xffb347;
    constexpr int kDim = 0x254056;

    // ---- cell layout (metres) ----------------------------------------------
    constexpr float kPedestalH = 0.2f;
    constexpr float kTableTop = 0.30f;
    const Vector3 kTableCenter{0.55f, 0.28f, 0.f};
    const Vector3 kBinCenter{-0.42f, 0.f, 0.34f};
    constexpr float kCrate = 0.07f;     // crate edge length
    constexpr float kTipLen = 0.10f;    // flange -> suction tip along tool Z
    constexpr float kHoverH = 0.16f;    // hover height above a crate
    constexpr float kGrabRadius = 0.09f;// suction engage distance
    // survey pose: eye-in-hand sensor straight above the table centre, high
    // enough that the 70-deg FOV covers the whole crate zone
    const Vector3 kSurveyTip{kTableCenter.x, kTableTop + 0.45f, kTableCenter.z};
    constexpr float kTravelY = 0.52f;   // safe height for lateral traverses
    // Base-column keep-out: the top must sit ABOVE kTravelY, so table<->bin
    // traverses arc AROUND the pedestal instead of crossing directly over the
    // robot (tip directly above the base is near-singular — J4 saturates and
    // the IK crawls, stalling transports).
    constexpr float kBaseKeepOutR = 0.24f;  // base-column keep-out radius
    constexpr float kBaseKeepOutTop = 0.66f;// ...applies below this height

    // ========================================================================
    // SVG -> mesh helpers (same toolkit as examples/loaders/svg_ui.cpp)
    // ========================================================================

    std::shared_ptr<Group> buildSvg(const std::vector<SVGLoader::SVGData>& svgData) {
        auto group = Group::create();
        for (const auto& data : svgData) {
            const auto& fill = data.style.fill;
            if (fill && *fill != "none") {
                auto material = MeshBasicMaterial::create();
                material->color.copy(data.path.color);
                material->opacity = data.style.fillOpacity;
                material->transparent = true;
                material->depthTest = false;
                material->depthWrite = false;
                material->side = Side::Double;
                auto geometry = ShapeGeometry::create(SVGLoader::createShapes(data));
                auto mesh = Mesh::create(geometry, material);
                mesh->name = data.style.id;
                group->add(mesh);
            }
            const auto& stroke = data.style.stroke;
            if (stroke && *stroke != "none") {
                auto sMat = MeshBasicMaterial::create();
                sMat->color.setStyle(*stroke);
                sMat->opacity = data.style.strokeOpacity;
                sMat->transparent = true;
                sMat->depthTest = false;
                sMat->depthWrite = false;
                sMat->side = Side::Double;
                for (const auto& subPath : data.path.subPaths) {
                    auto sg = SVGLoader::pointsToStroke(subPath->getPoints(), data.style);
                    if (sg) group->add(Mesh::create(sg, sMat));
                }
            }
        }
        return group;
    }

    std::shared_ptr<Group> svgFromString(const std::string& svg) {
        SVGLoader loader;
        return buildSvg(loader.parse(svg));
    }

    std::string hex(int rgb) {
        std::ostringstream os;
        os << '#' << std::hex << std::setfill('0') << std::setw(6) << (rgb & 0xffffff);
        return os.str();
    }

    struct RectMesh {
        std::shared_ptr<Mesh> mesh;
        std::shared_ptr<MeshBasicMaterial> material;
    };

    RectMesh roundedRect(float w, float h, float r, int color, float opacity = 1.f) {
        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg">)"
            << R"(<rect x="0" y="0" width=")" << w << R"(" height=")" << h
            << R"(" rx=")" << r << R"(" ry=")" << r << R"(" fill="#ffffff"/></svg>)";
        SVGLoader loader;
        auto data = loader.parse(svg.str());
        auto mat = MeshBasicMaterial::create();
        mat->color = Color(color);
        mat->opacity = opacity;
        mat->transparent = true;
        mat->depthTest = false;
        mat->depthWrite = false;
        mat->side = Side::Double;
        auto geo = ShapeGeometry::create(SVGLoader::createShapes(data.front()));
        return {Mesh::create(geo, mat), mat};
    }

    // anchor a widget group in y-up pixel space; re-applied on resize
    struct Layout {
        std::vector<std::function<void(float, float)>> fns;
        void add(const std::shared_ptr<Object3D>& g, float ax, float ay, float ox, float oy, float z) {
            fns.emplace_back([=](float W, float H) {
                g->position.set(ax * W + ox, ay * H + oy, z);
            });
        }
        void addRaw(std::function<void(float, float)> fn) { fns.emplace_back(std::move(fn)); }
        void apply(float W, float H) {
            for (auto& f : fns) f(W, H);
        }
    };

    struct Readout {
        std::shared_ptr<TextSprite> sprite;
        std::string last;
        void set(const std::string& s) {
            if (s == last) return;
            last = s;
            sprite->setText(s);
        }
    };

    std::shared_ptr<TextSprite> makeText(const Font& font, const std::string& text, int color,
                                         float px, float ax, float ay, float ox, float oy,
                                         TextSprite::HorizontalAlignment h = TextSprite::HorizontalAlignment::Left,
                                         TextSprite::VerticalAlignment v = TextSprite::VerticalAlignment::Center) {
        auto t = TextSprite::create(font, px);
        t->setColor(Color(color));
        t->setText(text);
        t->setHorizontalAlignment(h);
        t->setVerticalAlignment(v);
        t->screenSpace = true;
        t->screenAnchor.set(ax, ay);
        t->position.set(ox, oy, 0.f);
        return t;
    }

    struct Button {
        std::shared_ptr<Group> group;
        std::shared_ptr<MeshBasicMaterial> bg;
        std::shared_ptr<Sprite> hit;
        std::shared_ptr<TextSprite> label;
        float ax, ay, ox, oy, w, h;
        bool active = false, hovered = false, pressed = false;
        std::function<void()> onClick;

        void refresh() const {
            int c = active ? kBtnActive : ((pressed || hovered) ? kBtnHover : kBtn);
            bg->color = Color(c);
            label->setColor(Color(active ? 0x06121d : 0xcfe8ff));
        }
        [[nodiscard]] bool contains(float mx, float myUp, float W, float H) const {
            const float px = ax * W + ox;
            const float py = ay * H + oy;
            return mx >= px && mx <= px + w && myUp >= py - h && myUp <= py;
        }
    };

    std::shared_ptr<Button> makeButton(Scene& ui, Layout& layout, const Font& font,
                                       const std::string& text, float w, float h,
                                       float ax, float ay, float ox, float oy,
                                       std::function<void()> onClick) {
        auto b = std::make_shared<Button>();
        b->ax = ax;
        b->ay = ay;
        b->ox = ox;
        b->oy = oy;
        b->w = w;
        b->h = h;
        b->onClick = std::move(onClick);

        auto rect = roundedRect(w, h, 8.f, kBtn);
        b->bg = rect.material;
        b->group = Group::create();
        b->group->add(rect.mesh);
        b->group->scale.y = -1.f;
        ui.add(b->group);
        layout.add(b->group, ax, ay, ox, oy, 0.4f);

        const float labelPx = h * 0.40f;
        b->label = makeText(font, text, 0xcfe8ff, labelPx, ax, ay, ox + w / 2.f, oy - h / 2.f,
                            TextSprite::HorizontalAlignment::Center, TextSprite::VerticalAlignment::Center);
        const float maxLabelW = w - 16.f;
        if (b->label->scale.x > maxLabelW) {
            b->label->setWorldScale(labelPx * maxLabelW / static_cast<float>(b->label->scale.x));
        }
        ui.add(b->label);

        auto hitMat = SpriteMaterial::create();
        hitMat->visible = false;
        b->hit = Sprite::create(hitMat);
        b->hit->screenSpace = true;
        b->hit->screenAnchor.set(ax, ay);
        b->hit->position.set(ox, oy, 0.f);
        b->hit->scale.set(w, h, 1.f);
        b->hit->center.set(0.f, 1.f);
        Button* raw = b.get();
        b->hit->onMouseDown = [raw](int) { raw->pressed = true; };
        b->hit->onMouseUp = [raw](int) {
            raw->pressed = false;
            if (raw->onClick) raw->onClick();
        };
        ui.add(b->hit);
        return b;
    }

    struct Led {
        std::shared_ptr<Group> group;
        std::shared_ptr<MeshBasicMaterial> mat;
        void set(int color) const { mat->color = Color(color); }
    };

    Led makeLed(float r) {
        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg"><circle cx=")" << r << R"(" cy=")" << r
            << R"(" r=")" << r << R"(" fill="#ffffff"/></svg>)";
        SVGLoader loader;
        auto data = loader.parse(svg.str());
        auto mat = MeshBasicMaterial::create();
        mat->color = Color(kDim);
        mat->transparent = true;
        mat->depthTest = false;
        mat->depthWrite = false;
        mat->side = Side::Double;
        auto mesh = Mesh::create(ShapeGeometry::create(SVGLoader::createShapes(data.front())), mat);
        auto g = Group::create();
        g->add(mesh);
        g->scale.y = -1.f;
        return {g, mat};
    }

    // horizontal joint-range bar: dark track + accent fill scaled in x
    struct JointBar {
        std::shared_ptr<Group> group;
        std::shared_ptr<Mesh> fill;
        float w;
        void set(float t) const {
            fill->scale.x = std::clamp(t, 0.f, 1.f) * w;
        }
    };

    JointBar makeJointBar(float w, float h) {
        JointBar bar;
        bar.w = w;
        bar.group = Group::create();
        bar.group->scale.y = -1.f;

        auto track = roundedRect(w, h, h / 2.f, kBtn, 0.9f);
        bar.group->add(track.mesh);

        // 1px-wide fill rect, x-scaled to the bar value (slight inset)
        auto fill = roundedRect(1.f, h - 2.f, 0.f, kAccent);
        fill.mesh->position.set(1.f, 1.f, 0.05f);
        bar.fill = fill.mesh;
        bar.fill->scale.x = 1.f;
        bar.group->add(bar.fill);
        return bar;
    }

    // ========================================================================
    // Damped-least-squares IK on Robot::computeEndEffectorTransform
    // ========================================================================

    Vector3 tipOf(const Matrix4& flange) {
        // tool tip = flange origin + tipLen along flange Z
        const auto& e = flange.elements;
        return {e[12] + e[8] * kTipLen, e[13] + e[9] * kTipLen, e[14] + e[10] * kTipLen};
    }

    Vector3 toolAxisOf(const Matrix4& flange) {
        const auto& e = flange.elements;
        return Vector3(e[8], e[9], e[10]).normalize();
    }

    // solve the 6x6 system A y = b in place (Gaussian elimination, partial pivot)
    bool solve6(std::array<float, 36>& A, std::array<float, 6>& b) {
        for (int c = 0; c < 6; ++c) {
            int piv = c;
            for (int r = c + 1; r < 6; ++r) {
                if (std::abs(A[r * 6 + c]) > std::abs(A[piv * 6 + c])) piv = r;
            }
            if (std::abs(A[piv * 6 + c]) < 1e-12f) return false;
            if (piv != c) {
                for (int k = 0; k < 6; ++k) std::swap(A[c * 6 + k], A[piv * 6 + k]);
                std::swap(b[c], b[piv]);
            }
            const float inv = 1.f / A[c * 6 + c];
            for (int r = c + 1; r < 6; ++r) {
                const float f = A[r * 6 + c] * inv;
                if (f == 0.f) continue;
                for (int k = c; k < 6; ++k) A[r * 6 + k] -= f * A[c * 6 + k];
                b[r] -= f * b[c];
            }
        }
        for (int c = 5; c >= 0; --c) {
            float s = b[c];
            for (int k = c + 1; k < 6; ++k) s -= A[c * 6 + k] * b[k];
            b[c] = s / A[c * 6 + c];
        }
        return true;
    }

    // One controller tick: move q toward placing the tool tip at `target` with
    // the tool axis pointing straight down. Joint speed is capped so the arm
    // glides rather than teleports. Returns the remaining tip position error.
    class IkController {

    public:
        IkController(Robot& robot, std::vector<float> restPose)
            : robot_(robot), n_(robot.numDOF()), ranges_(robot.getJointRanges(false)),
              qRest_(std::move(restPose)) {
            qRest_.resize(n_, 0.f);
        }

        float track(std::vector<float>& q, const Vector3& target, float dt) {

            const Vector3 down{0.f, -1.f, 0.f};
            const float maxDq = kJointSpeed * dt;
            const std::vector<float> q0 = q;

            float posErr = 0.f;
            for (int iter = 0; iter < kIters; ++iter) {

                const Matrix4 m = robot_.computeEndEffectorTransform(q);
                const Vector3 p = tipOf(m);
                const Vector3 a = toolAxisOf(m);

                // residual: position to target + tool-axis misalignment (a x down
                // is the small-angle rotation needed to swing a onto down)
                const Vector3 ePos = Vector3().subVectors(target, p);
                const Vector3 ori = Vector3().crossVectors(a, down);
                posErr = ePos.length();

                std::array<float, 6> e{ePos.x, ePos.y, ePos.z,
                                       -ori.x * kOriW, -ori.y * kOriW, -ori.z * kOriW};
                if (posErr < 1e-4f && ori.length() < 1e-3f) break;

                // finite-difference Jacobian, 6 x n
                std::vector<std::array<float, 6>> J(n_);
                std::vector<float> qh = q;
                for (size_t j = 0; j < n_; ++j) {
                    qh[j] = q[j] + kH;
                    const Matrix4 mj = robot_.computeEndEffectorTransform(qh);
                    qh[j] = q[j];
                    const Vector3 pj = tipOf(mj);
                    const Vector3 oj = Vector3().crossVectors(toolAxisOf(mj), down);
                    J[j] = {(pj.x - p.x) / kH, (pj.y - p.y) / kH, (pj.z - p.z) / kH,
                            (oj.x - ori.x) * kOriW / kH, (oj.y - ori.y) * kOriW / kH, (oj.z - ori.z) * kOriW / kH};
                }

                // task step: dq = J^T (J J^T + lambda^2 I)^-1 e
                std::array<float, 36> A{};
                for (int r = 0; r < 6; ++r) {
                    for (int c = 0; c < 6; ++c) {
                        float s = 0.f;
                        for (size_t j = 0; j < n_; ++j) s += J[j][r] * J[j][c];
                        A[r * 6 + c] = s + (r == c ? kLambda * kLambda : 0.f);
                    }
                }
                std::array<float, 36> A2 = A;// solve6 destroys its input
                std::array<float, 6> y = e;
                if (!solve6(A, y)) break;

                // null-space rest-posture bias: v = k (q_rest - q) projected
                // through (I - J^+ J), so the redundant elbow DOF drifts toward
                // the comfortable pose without disturbing the tip task.
                std::array<float, 6> Jv{};
                std::vector<float> v(n_);
                for (size_t j = 0; j < n_; ++j) {
                    v[j] = kNullGain * (qRest_[j] - q[j]);
                    for (int r = 0; r < 6; ++r) Jv[r] += J[j][r] * v[j];
                }
                const bool nullOk = solve6(A2, Jv);// Jv becomes (JJ^T+l^2)^-1 J v

                for (size_t j = 0; j < n_; ++j) {
                    float dq = 0.f;
                    for (int r = 0; r < 6; ++r) dq += J[j][r] * y[r];
                    if (nullOk) {
                        float jjv = 0.f;
                        for (int r = 0; r < 6; ++r) jjv += J[j][r] * Jv[r];
                        dq += v[j] - jjv;
                    }
                    q[j] = ranges_[j].clamp(q[j] + dq);
                }
            }

            // joint speed cap relative to the start of this tick
            for (size_t j = 0; j < n_; ++j) {
                q[j] = q0[j] + std::clamp(q[j] - q0[j], -maxDq, maxDq);
            }

            const Matrix4 m = robot_.computeEndEffectorTransform(q);
            return Vector3().subVectors(target, tipOf(m)).length();
        }

    private:
        static constexpr int kIters = 3;
        static constexpr float kH = 1e-3f;     // finite-difference step (rad)
        static constexpr float kLambda = 0.12f;// DLS damping
        static constexpr float kOriW = 0.30f;  // orientation error weight
        static constexpr float kJointSpeed = 1.8f;// rad/s
        static constexpr float kNullGain = 0.04f; // rest-posture pull per iteration

        Robot& robot_;
        size_t n_;
        std::vector<Robot::JointRange> ranges_;
        std::vector<float> qRest_;
    };

    // ========================================================================
    // Cartesian motion planner
    // ========================================================================
    // The IK never tracks the raw goal. Instead a setpoint moves toward it
    // gantry-style — ascend to travel height, traverse, descend vertically —
    // at a capped Cartesian speed, and keep-out volumes are enforced on the
    // setpoint each step. The IK then only ever chases a nearby, collision-
    // free point, which is what makes approaches vertical and predictable.
    struct MotionPlanner {

        Vector3 setpoint;

        void reset(const Vector3& p) { setpoint = p; }

        // push p out of the keep-out volumes; `goal` carves the descend
        // cylinder out of the table guard so picks can reach the crates
        static Vector3 guard(Vector3 p, const Vector3& goal) {
            // floor
            p.y = std::max(p.y, 0.03f);
            // base column (the robot itself + pedestal)
            const float r = std::hypot(p.x, p.z);
            if (r < kBaseKeepOutR && p.y < kBaseKeepOutTop) {
                const float s = kBaseKeepOutR / std::max(r, 1e-4f);
                p.x *= s;
                p.z *= s;
            }
            // table slab: stay clear above it unless descending onto the goal
            const bool overTable = std::abs(p.x - kTableCenter.x) < 0.29f &&
                                   std::abs(p.z - kTableCenter.z) < 0.39f;
            const float latGoal = std::hypot(p.x - goal.x, p.z - goal.z);
            if (overTable && latGoal > 0.06f) {
                p.y = std::max(p.y, kTableTop + 0.06f);
            }
            return p;
        }

        const Vector3& step(const Vector3& goal, float vmax, float dt) {
            // route: ascend in place, traverse at travel height, descend
            Vector3 wp = goal;
            const float lat = std::hypot(goal.x - setpoint.x, goal.z - setpoint.z);
            if (lat > 0.03f) {
                const float travelY = std::max(goal.y, kTravelY);
                if (setpoint.y < travelY - 0.02f) {
                    wp.set(setpoint.x, travelY, setpoint.z);
                } else {
                    wp.set(goal.x, travelY, goal.z);
                }
            }
            wp = guard(wp, goal);

            Vector3 d = Vector3().subVectors(wp, setpoint);
            const float len = d.length();
            const float maxStep = vmax * dt;
            if (len > maxStep) d.multiplyScalar(maxStep / len);
            setpoint.add(d);
            return setpoint;
        }
    };

    // ========================================================================
    // crates
    // ========================================================================

    struct Crate {
        std::shared_ptr<Mesh> mesh;
        PxRigidDynamic* actor = nullptr;
    };

    constexpr std::array<int, 4> kCrateColors{0xff8c42, 0x47e07a, 0xb087f5, 0xffd166};

    // ========================================================================
    // perception: point cloud -> crate detections
    // ========================================================================
    // The controller's only source of crate positions. Points above the table
    // surface inside the table footprint are binned into a 2 cm XZ grid and
    // flood-filled into clusters; each cluster of sufficient support becomes a
    // detection with an XZ centroid and a measured top height. Crates already
    // dropped in the bin fall outside the table footprint and disappear from
    // perception naturally — no bookkeeping flags.

    struct Detection {
        Vector3 center;// XZ centroid at measured top height
        int support = 0;// contributing cloud points
    };

    std::vector<Detection> detectCrates(const std::vector<Vector3>& cloud) {

        constexpr float cell = 0.02f;
        constexpr float yMin = kTableTop + 0.02f;// above table-surface returns + noise
        constexpr float yMax = kTableTop + 0.30f;// below the arm/tool
        constexpr int kGrid = 64;// covers +-0.64 m around the table centre

        struct CellAgg {
            float sx = 0, sz = 0, top = 0;
            int n = 0;
            int cluster = -1;
        };
        std::unordered_map<int, CellAgg> cells;
        const auto keyOf = [&](float x, float z) -> int {
            const int ix = static_cast<int>(std::floor((x - kTableCenter.x) / cell)) + kGrid;
            const int iz = static_cast<int>(std::floor((z - kTableCenter.z) / cell)) + kGrid;
            if (ix < 0 || iz < 0 || ix >= 2 * kGrid || iz >= 2 * kGrid) return -1;
            return ix * 2 * kGrid + iz;
        };

        for (const auto& p : cloud) {
            if (p.y < yMin || p.y > yMax) continue;
            if (std::abs(p.x - kTableCenter.x) > 0.27f || std::abs(p.z - kTableCenter.z) > 0.37f) continue;
            const int key = keyOf(p.x, p.z);
            if (key < 0) continue;
            auto& c = cells[key];
            c.sx += p.x;
            c.sz += p.z;
            c.top = std::max(c.top, p.y);
            c.n++;
        }

        // flood fill over occupied neighbouring cells (8-connected)
        std::vector<Detection> out;
        std::vector<int> stack;
        int nextCluster = 0;
        for (auto& [key, c] : cells) {
            if (c.cluster >= 0 || c.n < 2) continue;
            const int id = nextCluster++;
            float sx = 0, sz = 0, top = 0;
            int n = 0;
            stack.assign(1, key);
            c.cluster = id;
            while (!stack.empty()) {
                const int k = stack.back();
                stack.pop_back();
                auto& cc = cells.at(k);
                sx += cc.sx;
                sz += cc.sz;
                top = std::max(top, cc.top);
                n += cc.n;
                const int ix = k / (2 * kGrid), iz = k % (2 * kGrid);
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        const int nx = ix + dx, nz = iz + dz;
                        if (nx < 0 || nz < 0 || nx >= 2 * kGrid || nz >= 2 * kGrid) continue;
                        const auto it = cells.find(nx * 2 * kGrid + nz);
                        if (it == cells.end() || it->second.cluster >= 0 || it->second.n < 2) continue;
                        it->second.cluster = id;
                        stack.push_back(nx * 2 * kGrid + nz);
                    }
                }
            }
            if (n >= 30) {// reject speckle / partial edge returns
                out.push_back({{sx / static_cast<float>(n), top, sz / static_cast<float>(n)}, n});
            }
        }
        return out;
    }

    // --selftest: headless IK convergence check over representative cell
    // targets (no window / GL needed). Returns 0 when every target is reached
    // within tolerance, so it can run in CI or after controller changes.
    int runSelfTest(Robot& robot) {

        std::vector<float> q{0.f, 0.55f, 0.f, -1.5f, 0.f, 0.9f, 0.f};
        q.resize(robot.numDOF(), 0.f);
        IkController ik(robot, q);
        MotionPlanner planner;
        planner.reset(tipOf(robot.computeEndEffectorTransform(q)));

        // a full pick-place tour; the planner must route table<->bin legs at
        // travel height without dragging the tool through the table or base
        const std::vector<Vector3> targets{
                {kTableCenter.x - 0.07f, kTableTop + kCrate + kHoverH, kTableCenter.z - 0.2f},// hover, near table corner
                {kTableCenter.x - 0.07f, kTableTop + kCrate + 0.012f, kTableCenter.z - 0.2f}, // descend onto crate
                {kBinCenter.x, 0.46f, kBinCenter.z},                                          // transport to bin
                {kTableCenter.x + 0.07f, kTableTop + kCrate + 0.012f, kTableCenter.z + 0.2f}, // pick at far corner
                kSurveyTip,                                                                   // perception survey pose
        };

        constexpr float dt = 1.f / 60.f;
        constexpr float slack = 0.025f;// IK tracking lag behind the guarded setpoint
        int failures = 0;
        for (size_t t = 0; t < targets.size(); ++t) {
            const Vector3& goal = targets[t];
            float err = 1e9f;
            int ticks = 0;
            int guardHits = 0;
            for (; ticks < 8 * 60; ++ticks) {// up to 8 simulated seconds
                const Vector3& sp = planner.step(goal, 0.6f, dt);
                ik.track(q, sp, dt);
                const Vector3 tip = tipOf(robot.computeEndEffectorTransform(q));
                err = Vector3().subVectors(goal, tip).length();
                if (err < 0.005f) break;

                const bool overTable = std::abs(tip.x - kTableCenter.x) < 0.29f &&
                                       std::abs(tip.z - kTableCenter.z) < 0.39f;
                const float latGoal = std::hypot(tip.x - goal.x, tip.z - goal.z);
                if (overTable && latGoal > 0.09f && tip.y < kTableTop + 0.06f - slack) guardHits++;
                if (std::hypot(tip.x, tip.z) < kBaseKeepOutR - slack && tip.y < kBaseKeepOutTop - slack) guardHits++;
                if (tip.y < 0.03f - slack) guardHits++;
            }
            const Matrix4 m = robot.computeEndEffectorTransform(q);
            const float tilt = std::acos(std::clamp(-toolAxisOf(m).y, -1.f, 1.f)) * math::RAD2DEG;
            const bool ok = err < 0.005f && guardHits == 0;
            std::cout << "target " << t << ": err=" << err << " m, tool tilt=" << tilt
                      << " deg, ticks=" << ticks << ", guard hits=" << guardHits
                      << (ok ? "  OK" : "  FAIL") << std::endl;
            if (!ok) failures++;
        }
        return failures == 0 ? 0 : 1;
    }

    // --depthprobe [gl|wgpu|vulkan]: deterministic sensor check. Sensor 1 m
    // above a flat floor with one 0.4 m box offset in +x: floor points must
    // land at y=0, box-top points at y=0.4, on every backend. On Vulkan the
    // identical pinhole pattern is ray-traced instead of rasterised.
    int runDepthProbe(GraphicsAPI api) {

        Canvas canvas(Canvas::Parameters().title("depth probe").size(320, 240));
        auto renderer = createRenderer(canvas, api);

        Scene scene;
        scene.background = Color::black;
        auto floor = Mesh::create(BoxGeometry::create(4.f, 0.1f, 4.f), MeshBasicMaterial::create());
        floor->position.y = -0.05f;
        scene.add(floor);
        auto boxMesh = Mesh::create(BoxGeometry::create(0.4f, 0.4f, 0.4f), MeshBasicMaterial::create());
        boxMesh->position.set(0.5f, 0.2f, 0.f);
        scene.add(boxMesh);

        DepthSensor sensor(60.f, 64, 48, 0.05f, 3.f);
        sensor.rangeNoise = 0.f;
        sensor.position.set(0.f, 1.f, 0.f);
        sensor.rotation.x = -math::PI / 2;// look straight down
        scene.addRef(sensor);

#ifdef ROBOT_CELL_WITH_VULKAN
        auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get());
        PathTracedLidarSensor ptProbe(60.f, 64, 48, 3.f);
        std::vector<LidarReturn> returns;
        ptProbe.position.copy(sensor.position);
        ptProbe.rotation.x = -math::PI / 2;
        if (vk) scene.addRef(ptProbe);
#endif
        scene.updateMatrixWorld(true);

        PerspectiveCamera cam(60, canvas.aspect(), 0.1f, 10.f);
        cam.position.set(0, 2, 2);

        // demo-style HUD overlay: ortho scene + a screen-space TextSprite whose
        // text changes every frame, drawn as a second pass with clearDepth
        auto uiScene = Scene::create();
        auto uiCam = OrthographicCamera::create(0, 320, 240, 0, 0.1f, 100);
        uiCam->position.z = 10;
        FontLoader fontLoader;
        const Font font = fontLoader.defaultFont();
        auto label = TextSprite::create(font, 20);
        label->setText("0");
        label->screenSpace = true;
        label->screenAnchor.set(0.f, 1.f);
        label->position.set(10.f, -20.f, 0.f);
        uiScene->add(label);

        // Mimic the demo's frame structure exactly: scan + screen render per
        // animate tick. The box slides 0.1 m in z each frame; a correct scan
        // tracks it with at most one frame of latency, a stale readback lags
        // visibly, a broken one never sees it.
        std::vector<Vector3> cloud;
        int frame = 0;
        int failures = 0;
        canvas.animate([&] {
            frame++;
            // move the SENSOR, not the box: a correct pipeline renders and
            // unprojects with the same pose, so the box stays world-anchored.
            sensor.position.x = 0.15f * static_cast<float>(frame);
            sensor.position.y = 1.f + 0.05f * static_cast<float>(frame);

            // render BEFORE scanning: the ray-traced sensor walks the TLAS
            // the render just built (the raster sensor doesn't care)
            renderer->autoClear = true;
            renderer->render(scene, cam);
            renderer->autoClear = false;
            renderer->clearDepth();
            renderer->render(*uiScene, *uiCam);

#ifdef ROBOT_CELL_WITH_VULKAN
            if (vk) {
                ptProbe.position.copy(sensor.position);
                ptProbe.scan(*vk, returns);
                cloud.clear();
                for (const auto& r : returns) {
                    if (r.returnNo > 0) cloud.push_back(r.position);
                }
            } else
#endif
            {
                sensor.scan(*renderer, scene, cloud);
            }

            // mean position of all points more than 5 cm above the floor = box top
            int nBox = 0;
            Vector3 boxMean;
            for (const auto& p : cloud) {
                if (p.y > 0.05f) {
                    boxMean.add(p);
                    nBox++;
                }
            }
            if (nBox) boxMean.multiplyScalar(1.f / static_cast<float>(nBox));

            // box is fixed at x=0.5; the cloud's box estimate must stay there
            const float xErr = std::abs(boxMean.x - boxMesh->position.x);
            const bool ok = nBox > 0 && xErr < 0.06f;
            if (!ok) failures++;
            std::cout << "frame " << frame << ": sensorX=" << sensor.position.x
                      << " cloudBoxX=" << boxMean.x << " (want " << boxMesh->position.x
                      << ") cloudY=" << boxMean.y << " n=" << nBox
                      << (ok ? "  OK" : "  FAIL") << std::endl;

            label->setText(std::to_string(frame));

            if (frame >= 5) canvas.close();
        });
        return failures == 0 ? 0 : 1;
    }

}// namespace

int main(int argc, char** argv) {

    if (argc > 1 && std::string(argv[1]) == "--depthprobe") {
        const std::string backend = argc > 2 ? argv[2] : "gl";
        GraphicsAPI api = GraphicsAPI::OpenGL;
        if (backend == "wgpu") api = GraphicsAPI::WebGPU;
        if (backend == "vulkan") {
#ifdef ROBOT_CELL_WITH_VULKAN
            api = GraphicsAPI::Vulkan;
#else
            std::cerr << "built without Vulkan support" << std::endl;
            return 1;
#endif
        }
        return runDepthProbe(api);
    }

    // --colorprobe [gl|wgpu]: clear-color convention check. The framebuffer
    // bytes for a flat background must equal the user's sRGB color value on
    // BOTH the surface and render-target paths (the three.js convention all
    // backends follow). Catches double/missing encodes in either path.
    if (argc > 1 && std::string(argv[1]) == "--colorprobe") {
        const std::string backend = argc > 2 ? argv[2] : "gl";
        const GraphicsAPI api = backend == "wgpu" ? GraphicsAPI::WebGPU : GraphicsAPI::OpenGL;
        Canvas canvas(Canvas::Parameters().title("color probe").size(320, 240));
        auto renderer = createRenderer(canvas, api);

        Scene scene;
        scene.background = Color(0x10151c);// expect raw bytes 16,21,28
        PerspectiveCamera cam(60, canvas.aspect(), 0.1f, 10.f);

        auto rt = RenderTarget::create(320, 240, RenderTarget::Options{});
        int frame = 0;
        canvas.animate([&] {
            frame++;
            renderer->setRenderTarget(rt.get());
            renderer->render(scene, cam);
            renderer->copyTextureToImage(*rt->texture);
            renderer->setRenderTarget(nullptr);
            renderer->render(scene, cam);
            if (frame >= 3) {
                const auto& d = rt->texture->image().data();
                std::cout << "RT bytes:      " << int(d[0]) << "," << int(d[1]) << "," << int(d[2]) << std::endl;
                auto fb = renderer->readRGBPixels();
                if (fb.size() >= 3) {
                    const size_t mid = (fb.size() / 2 / 3) * 3;
                    std::cout << "surface bytes: " << int(fb[mid]) << "," << int(fb[mid + 1]) << "," << int(fb[mid + 2]) << std::endl;
                }
                canvas.close();
            }
        });
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "--selftest") {
        const auto urdfPath = std::filesystem::path(DATA_FOLDER) / "urdf" / "lbr_iiwa_14_r820.urdf";
        URDFLoader urdfLoader;
        auto robot = urdfLoader.load(urdfPath);
        robot->rotation.x = -math::PI / 2;
        robot->position.y = kPedestalH;
        robot->updateMatrix();
        return runSelfTest(*robot);
    }

    // --capture <sec>: dump one frame to robot_cell_capture.png after <sec>
    // seconds, then keep running. Verification aid for headless test runs.
    float captureAfter = 0.f;
    if (argc > 2 && std::string(argv[1]) == "--capture") {
        captureAfter = std::stof(argv[2]);
    }

    Canvas canvas(Canvas::Parameters().title("threepp - KUKA Robot Cell").size(1366, 820).antialiasing(4));
    auto renderer = createRenderer(canvas);

#ifdef ROBOT_CELL_WITH_VULKAN
    // On Vulkan the perception sensor is ray-traced through the renderer's
    // TLAS instead of rasterised into a render target.
    auto* vkRenderer = dynamic_cast<VulkanRenderer*>(renderer.get());
#endif

    const float dpi = monitor::contentScale().first;

    // ===== 3D world =========================================================
    auto scene = Scene::create();
    scene->background = Color(0x10151c);

    auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.05f, 50.f);
    camera->position.set(1.6f, 1.25f, 1.7f);

    OrbitControls controls(*camera, canvas);
    controls.enableDamping = true;
    controls.target.set(0.1f, 0.35f, 0.f);

    auto hemi = HemisphereLight::create(0xbfd8ff, 0x1a2030);
    hemi->intensity = 0.9f;
    scene->add(hemi);
    auto sun = DirectionalLight::create(0xffffff, 1.6f);
    sun->position.set(2.5f, 4.f, 1.5f);
    scene->add(sun);

    // ===== physics ==========================================================
    PhysxWorld world;

    // floor
    auto floorMesh = Mesh::create(
            BoxGeometry::create(6.f, 0.1f, 6.f),
            MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x272c34)).roughness(0.9f)));
    floorMesh->position.y = -0.05f;
    scene->add(floorMesh);
    world.addStatic(*floorMesh);

    auto grid = GridHelper::create(6, 30, Color(kPanelEdge), Color(0x1a2027));
    grid->position.y = 0.002f;
    scene->add(grid);

    // robot pedestal
    auto pedestal = Mesh::create(
            BoxGeometry::create(0.34f, kPedestalH, 0.34f),
            MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x343b46)).roughness(0.6f)));
    pedestal->position.y = kPedestalH / 2.f;
    scene->add(pedestal);
    world.addStatic(*pedestal);

    // infeed table
    auto table = Mesh::create(
            BoxGeometry::create(0.5f, 0.04f, 0.7f),
            MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x3b4252)).roughness(0.7f)));
    table->position.set(kTableCenter.x, kTableTop - 0.02f, kTableCenter.z);
    scene->add(table);
    world.addStatic(*table);
    {
        auto legGeom = BoxGeometry::create(0.03f, kTableTop - 0.04f, 0.03f);
        auto legMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x2b303b)));
        for (float sx : {-1.f, 1.f}) {
            for (float sz : {-1.f, 1.f}) {
                auto leg = Mesh::create(legGeom, legMat);
                leg->position.set(kTableCenter.x + sx * 0.22f, (kTableTop - 0.04f) / 2.f, kTableCenter.z + sz * 0.32f);
                scene->add(leg);
            }
        }
    }

    // outfeed bin: base + 4 walls, all static colliders
    {
        auto binMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x506072)).roughness(0.8f));
        auto addBinPart = [&](float w, float h, float d, float x, float y, float z) {
            auto part = Mesh::create(BoxGeometry::create(w, h, d), binMat);
            part->position.set(kBinCenter.x + x, y, kBinCenter.z + z);
            scene->add(part);
            world.addStatic(*part);
        };
        const float W = 0.30f, H = 0.14f, T = 0.015f;
        addBinPart(W, T, W, 0.f, T / 2.f, 0.f);
        addBinPart(W, H, T, 0.f, H / 2.f, -W / 2.f);
        addBinPart(W, H, T, 0.f, H / 2.f, W / 2.f);
        addBinPart(T, H, W, -W / 2.f, H / 2.f, 0.f);
        addBinPart(T, H, W, W / 2.f, H / 2.f, 0.f);
    }

    // ===== robot ============================================================
    const auto urdfPath = std::filesystem::path(DATA_FOLDER) / "urdf" / "lbr_iiwa_14_r820.urdf";
    URDFLoader urdfLoader;
    auto robot = urdfLoader.load(urdfPath);
    robot->rotation.x = -math::PI / 2;// URDF Z-up -> world Y-up
    robot->position.y = kPedestalH;
    robot->showColliders(false);
    robot->updateMatrix();// computeEndEffectorTransform premultiplies robot->matrix
    scene->add(robot);

    const size_t nDof = robot->numDOF();
    std::vector<float> q{0.f, 0.55f, 0.f, -1.5f, 0.f, 0.9f, 0.f};
    q.resize(nDof, 0.f);
    robot->setJointValues(q);

    IkController ik(*robot, q);
    MotionPlanner planner;

    // tool frame: posed every frame from forward kinematics (flange + tipLen)
    auto toolGroup = Group::create();
    scene->add(toolGroup);
    {
        auto toolMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x222831)).roughness(0.4f).metalness(0.6f));
        auto column = Mesh::create(CylinderGeometry::create(0.018f, 0.018f, kTipLen - 0.02f, 16), toolMat);
        column->rotation.x = math::PI / 2;// cylinder Y axis -> tool Z
        column->position.z = (kTipLen - 0.02f) / 2.f;
        toolGroup->add(column);

        auto cupMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(kAccent)).roughness(0.5f));
        auto cup = Mesh::create(CylinderGeometry::create(0.026f, 0.018f, 0.02f, 16), cupMat);
        cup->rotation.x = math::PI / 2;
        cup->position.z = kTipLen - 0.01f;
        toolGroup->add(cup);
    }

    // kinematic pusher so the arm can nudge crates it sweeps through
    PxRigidDynamic* pusher = world.addDynamic(PxSphereGeometry(0.03f), PxTransform(PxVec3(0, 0.8f, 0)), 100.f);
    pusher->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);

    // ===== eye-in-hand depth sensor =========================================
    // 70-deg FOV so the survey pose sees the whole crate zone in one scan.
    // GL/WGPU: raster DepthSensor (render-to-target + depth readback).
    // Vulkan: the SAME pinhole beam pattern, ray-traced through the
    // renderer's TLAS — perception downstream is identical.
    constexpr float kSensFov = 70.f;
    constexpr unsigned int kSensW = 160, kSensH = 120;
    constexpr float kSensFar = 1.6f;

    DepthSensor sensor(kSensFov, kSensW, kSensH, 0.04f, kSensFar);
    sensor.rangeNoise = 0.004f;
    sensor.position.set(0.065f, 0.f, 0.01f);
    sensor.rotation.x = math::PI;// camera looks -Z; flip to look along tool +Z (down)

    std::shared_ptr<CameraHelper> frustum;// raster only — the PT sensor has no Camera
#ifdef ROBOT_CELL_WITH_VULKAN
    PathTracedLidarSensor ptSensor(kSensFov, kSensW, kSensH, kSensFar);
    std::vector<LidarReturn> lidarReturns;
    if (vkRenderer) {
        ptSensor.position.copy(sensor.position);
        ptSensor.rotation.x = math::PI;
        toolGroup->addRef(ptSensor);
    } else
#endif
    {
        toolGroup->addRef(sensor);
        frustum = CameraHelper::create(sensor.getCamera());
        scene->add(frustum);
    }

    const size_t maxPoints = static_cast<size_t>(kSensW) * kSensH;
    auto pcGeom = BufferGeometry::create();
    pcGeom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(maxPoints * 3), 3));
    pcGeom->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(maxPoints * 3), 3));
    pcGeom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
    pcGeom->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);
    auto cloudPoints = Points::create(pcGeom, PointsMaterial::create(PointsMaterial::Params{}.size(0.006f).vertexColors(true)));
    cloudPoints->frustumCulled = false;
    scene->add(cloudPoints);

    bool sensorOn = true;
    std::vector<Vector3> cloud;
    std::vector<Detection> detections;// refreshed every scan; consumed at Survey

    // Detection markers: a ring hovering over each perceived crate, so what
    // the robot BELIEVES is visible next to what IS. Deliberately LINES, not
    // meshes: lines never enter the ray-tracing acceleration structure, so
    // the Vulkan sensor cannot see its own annotations by construction. The
    // raster sensors DO rasterise lines, so the group is still hidden during
    // raster scans.
    constexpr size_t kMaxMarkers = 10;
    auto markerGroup = Group::create();
    std::vector<std::shared_ptr<LineLoop>> markers;
    {
        auto ringMat = LineBasicMaterial::create();
        ringMat->color = Color(kGood);
        std::vector<float> pts;
        constexpr int kSeg = 48;
        for (int i = 0; i < kSeg; ++i) {
            const float a = 2.f * math::PI * static_cast<float>(i) / kSeg;
            pts.insert(pts.end(), {std::cos(a) * 0.052f, 0.f, std::sin(a) * 0.052f});
        }
        auto ringGeom = BufferGeometry::create();
        ringGeom->setAttribute("position", FloatBufferAttribute::create(pts, 3));
        for (size_t i = 0; i < kMaxMarkers; ++i) {
            auto m = LineLoop::create(ringGeom, ringMat);
            m->visible = false;
            markerGroup->add(m);
            markers.push_back(m);
        }
    }
    scene->add(markerGroup);

    // ===== crates ===========================================================
    std::vector<Crate> crates;
    int colorIdx = 0;

    auto spawnCrate = [&] {
        if (crates.size() >= 14) return;
        auto mat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(kCrateColors[colorIdx++ % kCrateColors.size()])).roughness(0.8f));
        auto mesh = Mesh::create(BoxGeometry::create(kCrate, kCrate, kCrate), mat);
        const float fx = static_cast<float>(std::rand() % 1000) / 1000.f;
        const float fz = static_cast<float>(std::rand() % 1000) / 1000.f;
        mesh->position.set(kTableCenter.x - 0.07f + fx * 0.14f,
                           kTableTop + 0.12f,
                           kTableCenter.z - 0.2f + fz * 0.4f);
        scene->add(mesh);
        auto* actor = world.add(*mesh, 300.f);
        crates.push_back({mesh, actor});
    };

    std::srand(7);
    for (int i = 0; i < 5; ++i) spawnCrate();

    // ===== gripper / job state ==============================================
    enum class Phase { Idle,
                       Survey,
                       Hover,
                       Descend,
                       Grip,
                       Lift,
                       Transport,
                       Release,
                       Retreat };

    Phase phase = Phase::Idle;
    bool autoMode = true;
    bool gripWanted = false;
    int pickedCount = 0;
    float phaseTime = 0.f;
    float surveySettle = 0.f;// time spent settled at the survey pose

    // The active job is a perceived detection (centroid + measured top), not
    // a crate index — the controller never touches crate ground truth.
    Detection job{};
    bool haveJob = false;
    std::optional<Vector3> jogRequest;// user click: take manual control, go here
    std::optional<Vector3> lastFailed;// skip this spot for one survey round

    PxRigidDynamic* carried = nullptr;
    PxTransform grabOffset(PxIdentity);

    Matrix4 flange = robot->computeEndEffectorTransform(q);
    Vector3 tip = tipOf(flange);
    Vector3 prevTip = tip;
    Vector3 tipVel;
    PxTransform tipPose(toPxVec3(tip));

    const Vector3 homeTip = tip;
    Vector3 manualTarget = homeTip;
    Vector3 tipTarget = homeTip;
    planner.reset(tip);

    auto releaseCarried = [&] {
        if (!carried) return;
        carried->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, false);
        carried->wakeUp();
        carried->setLinearVelocity(toPxVec3(tipVel * 0.5f));
        carried = nullptr;
    };

    // Suction actuator model: the cup seals against whatever crate face is
    // within reach of the tip. The proximity test below is the physical seal
    // forming, not the controller peeking at the world — the controller only
    // ever observes the resulting vacuum state (`carried != nullptr`).
    auto tryGrab = [&] {
        if (carried) return;
        for (auto& c : crates) {
            if (c.mesh->position.distanceTo(tip) < kGrabRadius) {
                carried = c.actor;
                carried->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
                grabOffset = tipPose.transformInv(carried->getGlobalPose());
                break;
            }
        }
    };

    // drive the kinematic pusher + carried crate from the latest tool pose
    world.onPreSubstep([&](float) {
        // pusher rides a bit up the tool column so the cup itself can touch
        const PxVec3 off = tipPose.q.rotate(PxVec3(0, 0, -0.05f));
        pusher->setKinematicTarget(PxTransform(tipPose.p + off, tipPose.q));
        if (carried) carried->setKinematicTarget(tipPose * grabOffset);
    });

    // choose a detection from the latest scan: nearest the robot base; a
    // just-failed spot is skipped for one round so the robot tries something
    // else first
    auto chooseDetection = [&]() -> bool {
        int best = -1;
        float bestD = 1e9f;
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            const auto& d = detections[i];
            if (lastFailed && detections.size() > 1 &&
                std::hypot(d.center.x - lastFailed->x, d.center.z - lastFailed->z) < 0.05f) {
                continue;
            }
            const float dist = std::hypot(d.center.x, d.center.z);
            if (dist < bestD) {
                bestD = dist;
                best = i;
            }
        }
        lastFailed.reset();
        if (best < 0) return false;
        job = detections[best];
        haveJob = true;
        return true;
    };

    auto resetCell = [&] {
        releaseCarried();
        gripWanted = false;
        phase = Phase::Idle;
        haveJob = false;
        jogRequest.reset();
        lastFailed.reset();
        pickedCount = 0;
        for (auto& c : crates) {
            world.unbind(*c.mesh);
            c.actor->release();
            scene->remove(*c.mesh);
        }
        crates.clear();
        manualTarget = homeTip;
        for (int i = 0; i < 5; ++i) spawnCrate();
    };

    // ===== HUD (SVG overlay) ================================================
    auto ui = Scene::create();
    auto size = canvas.size();
    auto uiCam = OrthographicCamera::create(0, size.width(), size.height(), 0, 0.1f, 100);
    uiCam->position.z = 10;

    Layout layout;
    FontLoader fontLoader;
    const Font font = fontLoader.defaultFont();

    // top status bar
    {
        std::ostringstream bar;
        bar << R"(<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width="1" height="44" fill=")"
            << hex(kPanel) << R"(" fill-opacity="0.85"/>)"
            << R"(<rect x="0" y="42" width="1" height="2" fill=")" << hex(kAccent) << R"("/></svg>)";
        auto g = svgFromString(bar.str());
        ui->add(g);
        layout.addRaw([g](float W, float H) {
            g->position.set(0.f, H, 0.1f);
            g->scale.set(W, -1.f, 1.f);
        });
    }
    ui->add(makeText(font, "THREEPP  //  KUKA ROBOT CELL", kAccent, 20 * dpi, 0.f, 1.f, 18.f, -22.f));
    Readout stateTxt{makeText(font, "", 0xcfe8ff, 18 * dpi, 0.5f, 1.f, 0.f, -22.f,
                              TextSprite::HorizontalAlignment::Center)};
    ui->add(stateTxt.sprite);
    Readout fpsTxt{makeText(font, "", 0x8fb6cf, 16 * dpi, 1.f, 1.f, -18.f, -22.f,
                            TextSprite::HorizontalAlignment::Right)};
    ui->add(fpsTxt.sprite);

    // left systems panel
    {
        auto panel = roundedRect(225, 205, 12, kPanel, 0.85f);
        auto g = Group::create();
        g->add(panel.mesh);
        g->scale.y = -1.f;
        ui->add(g);
        layout.add(g, 0.f, 1.f, 18.f, -64.f, 0.1f);
    }
    ui->add(makeText(font, "SYSTEMS", kAccent, 15 * dpi, 0.f, 1.f, 34.f, -86.f));

    struct StatusRow {
        Led led;
        const char* name;
    };
    std::vector<StatusRow> rows{
            {makeLed(8.f), "AUTO CYCLE"},
            {makeLed(8.f), "SUCTION"},
            {makeLed(8.f), "CARRYING"},
            {makeLed(8.f), "DEPTH SENSOR"}};
    for (size_t i = 0; i < rows.size(); ++i) {
        const float y = -112.f - static_cast<float>(i) * 24.f;
        ui->add(rows[i].led.group);
        layout.add(rows[i].led.group, 0.f, 1.f, 36.f, y + 8.f, 0.2f);
        ui->add(makeText(font, rows[i].name, 0xcfe8ff, 14 * dpi, 0.f, 1.f, 58.f, y));
    }
    Readout pickedTxt{makeText(font, "", 0xffffff, 15 * dpi, 0.f, 1.f, 36.f, -216.f)};
    ui->add(pickedTxt.sprite);
    Readout tipTxt{makeText(font, "", 0x8fb6cf, 13 * dpi, 0.f, 1.f, 36.f, -240.f)};
    ui->add(tipTxt.sprite);

    // right joints panel
    {
        auto panel = roundedRect(250, 64.f + 26.f * static_cast<float>(nDof), 12, kPanel, 0.85f);
        auto g = Group::create();
        g->add(panel.mesh);
        g->scale.y = -1.f;
        ui->add(g);
        layout.add(g, 1.f, 1.f, -268.f, -64.f, 0.1f);
    }
    ui->add(makeText(font, "JOINTS", kAccent, 15 * dpi, 1.f, 1.f, -252.f, -86.f));

    std::vector<JointBar> bars;
    for (size_t i = 0; i < nDof; ++i) {
        const float y = -112.f - static_cast<float>(i) * 26.f;
        ui->add(makeText(font, "J" + std::to_string(i + 1), 0xcfe8ff, 13 * dpi, 1.f, 1.f, -252.f, y));
        auto bar = makeJointBar(170.f, 10.f);
        ui->add(bar.group);
        layout.add(bar.group, 1.f, 1.f, -218.f, y + 5.f, 0.2f);
        bars.push_back(bar);
    }

    // buttons
    SpriteInteractor interactor(canvas, *ui);
    std::vector<std::shared_ptr<Button>> buttons;
    {
        const float bw = 118.f, bh = 40.f, gap = 10.f;
        float bx = 18.f;
        auto addBtn = [&](const std::string& label, std::function<void()> fn) {
            auto b = makeButton(*ui, layout, font, label, bw, bh, 0.f, 0.f, bx, bh + 18.f, std::move(fn));
            buttons.push_back(b);
            bx += bw + gap;
        };
        addBtn("AUTO", [&] {
            autoMode = !autoMode;
            if (autoMode) phaseTime = 99.f;// survey on the next tick, not in 3 s
        });
        addBtn("GRIP", [&] {
            if (phase != Phase::Idle) return;// the job sequencer owns the gripper
            gripWanted = !gripWanted;
        });
        addBtn("SENSOR", [&] { sensorOn = !sensorOn; });
        addBtn("FRUSTUM", [&] {
            if (frustum) frustum->visible = !frustum->visible;
        });
        addBtn("SPAWN", [&] { spawnCrate(); });
        addBtn("RESET", [&] { resetCell(); });
    }
    ui->add(makeText(font, "CLICK: JOG (TAKES MANUAL CONTROL)      G: GRIP / RELEASE      SPACE: SPAWN      AUTO: RESUME", 0x5f7a90,
                     12 * dpi, 0.5f, 0.f, 0.f, 84.f, TextSprite::HorizontalAlignment::Center));

    // ===== input ============================================================
    Vector2 cursor{-1e9f, -1e9f};
    MouseMoveListener moveL([&](const Vector2& p) { cursor = p; });
    canvas.addMouseListener(moveL);

    bool uiHover = false;
    Vector2 downPos;
    Raycaster raycaster;

    MouseDownListener downL([&](int button, const Vector2& pos) {
        if (button == 0) downPos = pos;
    });
    canvas.addMouseListener(downL);

    MouseUpListener upL([&](int button, const Vector2& pos) {
        if (button != 0 || uiHover) return;
        if (downPos.distanceTo(pos) > 6.f) return;// drag = orbit, not a click

        const auto s = canvas.size();
        Vector2 ndc{(pos.x / static_cast<float>(s.width())) * 2 - 1,
                    -(pos.y / static_cast<float>(s.height())) * 2 + 1};
        raycaster.setFromCamera(ndc, *camera);

        std::vector<Object3D*> targets;
        for (auto& c : crates) {
            targets.push_back(c.mesh.get());
        }
        targets.push_back(table.get());
        targets.push_back(floorMesh.get());

        const auto hits = raycaster.intersectObjects(targets);
        if (hits.empty()) return;
        const auto& hit = hits.front();

        // A click is the user taking the wheel: jog the tool to hover just
        // above the clicked surface point — low enough that the suction cup
        // can reach a crate top there (G then grips/releases). The controller
        // receives only the pointed-at position, never a crate identity.
        jogRequest = hit.point;
        jogRequest->y += 0.05f;
    });
    canvas.addMouseListener(upL);

    KeyAdapter keys(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (evt.key == Key::SPACE) spawnCrate();
        if (evt.key == Key::G && phase == Phase::Idle) gripWanted = !gripWanted;
    });
    canvas.addKeyListener(keys);

    // ===== resize ===========================================================
    auto relayout = [&](WindowSize s) {
        const float W = static_cast<float>(s.width());
        const float H = static_cast<float>(s.height());
        camera->aspect = s.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(s);
        uiCam->right = W;
        uiCam->top = H;
        uiCam->updateProjectionMatrix();
        layout.apply(W, H);
    };
    canvas.onWindowResize([&](WindowSize s) { relayout(s); });
    relayout(size);

    const char* phaseNames[]{"IDLE", "SURVEY", "MOVE TO TARGET", "DESCEND", "GRIP", "LIFT", "TO BIN", "RELEASE", "RETREAT"};

    // ===== main loop ========================================================
    Clock clock;
    std::vector<float> qCmd = q;
    float fpsTimer = 0.f;
    int frames = 0;
    bool renderedOnce = false;// the ray-traced sensor needs a TLAS first

    canvas.animate([&] {
        const float dt = std::min(clock.getDelta(), 0.05f);
        const auto s = canvas.size();
        const float W = static_cast<float>(s.width());
        const float H = static_cast<float>(s.height());
        phaseTime += dt;

        // --- job sequencer: a tiny pick-and-place controller ---------------
        const auto goTo = [&](Phase p) {
            phase = p;
            phaseTime = 0.f;
        };
        // Arrival must be tested AGAINST THE TARGET THE PHASE JUST ASSIGNED.
        // A precomputed flag silently carries the previous phase's target into
        // the first frame of the next one: Transport entered while standing on
        // the (just-reached) Lift target saw "arrived" instantly and fell
        // through to Release over the table — the crate dropped right after
        // pickup while still being counted as placed.
        const auto near = [&](float tol = 0.02f) { return tip.distanceTo(tipTarget) < tol; };
        const bool timedOut = phaseTime > 5.f;

        // A job that stalls mid-flight (unreachable pose, lost crate) drops
        // whatever is held where it is, remembers the spot so the next survey
        // tries a different detection first, and goes back to look again.
        const auto abortJob = [&] {
            gripWanted = false;
            if (haveJob) lastFailed = job.center;
            haveJob = false;
            goTo(autoMode && sensorOn ? Phase::Survey : Phase::Idle);
        };

        // Manual override: a world click disengages AUTO (taking the wheel)
        // and jogs the tool to the clicked point. Works in any phase, with or
        // without a crate on the cup — G then grips/releases.
        if (jogRequest) {
            manualTarget = *jogRequest;
            jogRequest.reset();
            autoMode = false;
            haveJob = false;
            if (!carried) gripWanted = false;
            goTo(Phase::Idle);
        }

        switch (phase) {
            case Phase::Idle:
                tipTarget = manualTarget;
                // sensorOn is the robot's eyes: without them there is nothing
                // to act on. AUTO patrols the table every few seconds in case
                // crates appeared.
                if (sensorOn && !carried && autoMode && phaseTime > 3.f) {
                    goTo(Phase::Survey);
                }
                break;
            case Phase::Survey:
                tipTarget = kSurveyTip;
                // decide only from a scan taken at rest: a survey done while
                // braking smears the cloud across the approach path
                surveySettle = near(0.03f) ? surveySettle + dt : 0.f;
                if (surveySettle > 0.35f) {
                    surveySettle = 0.f;
                    if (chooseDetection()) {
                        gripWanted = false;
                        goTo(Phase::Hover);
                    } else {
                        goTo(Phase::Idle);// table perceived empty
                    }
                } else if (timedOut) {
                    goTo(Phase::Idle);// survey pose unreachable
                }
                break;
            case Phase::Hover:
                tipTarget = Vector3(job.center.x, job.center.y + kHoverH, job.center.z);
                if (near()) goTo(Phase::Descend);
                else if (timedOut) abortJob();
                break;
            case Phase::Descend:
                // descend onto the PERCEIVED top (+ cup standoff); if the
                // crate moved since the survey, the grip simply misses and
                // the robot goes back to look again
                tipTarget = Vector3(job.center.x, job.center.y + 0.012f, job.center.z);
                if (near() || timedOut) goTo(Phase::Grip);
                break;
            case Phase::Grip:
                gripWanted = true;
                if (carried && phaseTime > 0.2f) {
                    // lift target anchored once, straight up from the pick
                    tipTarget = Vector3(tip.x, kTableTop + 0.35f, tip.z);
                    goTo(Phase::Lift);
                } else if (!carried && phaseTime > 1.f) {
                    abortJob();// no vacuum seal: re-perceive
                }
                break;
            case Phase::Lift:
                if (near()) goTo(Phase::Transport);
                else if (timedOut) abortJob();
                break;
            case Phase::Transport:
                tipTarget = Vector3(kBinCenter.x, 0.46f, kBinCenter.z);
                if (!carried) abortJob();// lost the crate en route
                else if (near()) goTo(Phase::Release);
                else if (timedOut) abortJob();
                break;
            case Phase::Release:
                if (phaseTime > 0.15f) {
                    // Count only verified deliveries: vacuum still sealed and
                    // the tip (own FK) actually above the bin.
                    const bool delivered = carried &&
                                           std::hypot(tip.x - kBinCenter.x, tip.z - kBinCenter.z) < 0.10f;
                    gripWanted = false;// release either way
                    if (delivered) {
                        pickedCount++;
                        haveJob = false;
                        goTo(Phase::Retreat);
                    } else {
                        abortJob();
                    }
                }
                break;
            case Phase::Retreat:
                tipTarget = Vector3(kBinCenter.x, 0.62f, kBinCenter.z);
                if (near() || timedOut) {
                    goTo(autoMode && sensorOn ? Phase::Survey : Phase::Idle);
                }
                break;
        }

        // --- gripper --------------------------------------------------------
        if (gripWanted && !carried) tryGrab();
        if (!gripWanted && carried) releaseCarried();

        // --- motion planning + IK + forward kinematics -----------------------
        // slow, careful Cartesian speed near contact; brisk while traversing
        const float vmax = (phase == Phase::Descend || phase == Phase::Grip) ? 0.25f : 0.6f;
        const Vector3& setpoint = planner.step(tipTarget, vmax, dt);
        ik.track(qCmd, setpoint, dt);
        robot->setJointValues(qCmd);

        flange = robot->computeEndEffectorTransform(qCmd);
        tip = tipOf(flange);
        tipVel = Vector3().subVectors(tip, prevTip).multiplyScalar(dt > 1e-5f ? 1.f / dt : 0.f);
        prevTip = tip;

        toolGroup->position.setFromMatrixPosition(flange);
        toolGroup->quaternion.setFromRotationMatrix(flange);
        tipPose = PxTransform(toPxVec3(tip), toPxQuat(toolGroup->quaternion));

        // --- physics ---------------------------------------------------------
        world.step(dt);

        // --- depth sensor + perception ---------------------------------------
        cloudPoints->visible = false;
        bool scanned = false;
        Vector3 sensorWorld;
        if (sensorOn) {
#ifdef ROBOT_CELL_WITH_VULKAN
            if (vkRenderer) {
                // Ray-traced path: the scan walks the TLAS from the previous
                // render (hence the first-frame guard). The point cloud, the
                // helper lines, and the line-based detection rings are not
                // triangles, so the rays cannot see any of them — no scan
                // hygiene required.
                if (renderedOnce) {
                    ptSensor.scan(*vkRenderer, lidarReturns);
                    cloud.clear();
                    for (const auto& r : lidarReturns) {
                        if (r.returnNo > 0) cloud.push_back(r.position);
                    }
                    ptSensor.getWorldPosition(sensorWorld);
                    scanned = true;
                }
            } else
#endif
            {
                const bool frustumWasVisible = frustum && frustum->visible;
                if (frustum) frustum->visible = false;
                markerGroup->visible = false;// perception must not see its own annotations
                sensor.scan(*renderer, *scene, cloud);
                if (frustum) frustum->visible = frustumWasVisible;
                markerGroup->visible = true;
                sensor.getWorldPosition(sensorWorld);
                scanned = true;
            }
        }

        if (scanned) {
            detections = detectCrates(cloud);

            auto* posAttr = pcGeom->getAttribute<float>("position");
            auto* colAttr = pcGeom->getAttribute<float>("color");
            Color c;
            int i = 0;
            for (const auto& p : cloud) {
                posAttr->setXYZ(i, p.x, p.y, p.z);
                c.setHSL(0.55f * (1.f - std::min(p.distanceTo(sensorWorld) / kSensFar, 1.f)), 1.f, 0.55f);
                colAttr->setXYZ(i, c.r, c.g, c.b);
                ++i;
            }
            pcGeom->setDrawRange(0, i);
            posAttr->needsUpdate();
            colAttr->needsUpdate();
            cloudPoints->visible = true;
        } else if (!sensorOn) {
            detections.clear();// eyes off, beliefs gone
        }

        // detection markers: one ring per perceived crate
        for (size_t i = 0; i < markers.size(); ++i) {
            const bool on = i < detections.size();
            markers[i]->visible = on;
            if (on) {
                markers[i]->position.set(detections[i].center.x,
                                         detections[i].center.y + 0.015f,
                                         detections[i].center.z);
            }
        }

        // --- HUD --------------------------------------------------------------
        const float myUp = H - cursor.y;
        uiHover = false;
        Button* top = nullptr;
        for (auto& b : buttons) {
            if (b->contains(cursor.x, myUp, W, H)) {
                uiHover = true;
                top = b.get();
            }
        }
        for (auto& b : buttons) b->hovered = (b.get() == top);
        controls.enabled = !uiHover;

        buttons[0]->active = autoMode;
        buttons[1]->active = gripWanted;
        buttons[2]->active = sensorOn;
        buttons[3]->active = frustum && frustum->visible;
        for (auto& b : buttons) b->refresh();

        rows[0].led.set(autoMode ? kGood : kDim);
        rows[1].led.set(gripWanted ? kWarn : kDim);
        rows[2].led.set(carried ? kGood : kDim);
        rows[3].led.set(sensorOn ? kGood : kDim);

        for (size_t i = 0; i < nDof; ++i) {
            const auto r = robot->getJointRange(i, false);
            bars[i].set((qCmd[i] - r.min) / (r.max - r.min));
        }

        stateTxt.set(!sensorOn && autoMode ? "BLIND (SENSOR OFF)"
                                           : phaseNames[static_cast<int>(phase)]);
        {
            std::ostringstream os;
            os << "PLACED " << pickedCount << "   DETECTED " << detections.size();
            pickedTxt.set(os.str());
        }
        {
            std::ostringstream os;
            os << "TIP  " << std::fixed << std::setprecision(2) << tip.x << "  " << tip.y << "  " << tip.z;
            tipTxt.set(os.str());
        }
        frames++;
        fpsTimer += dt;
        if (fpsTimer >= 0.5f) {
            std::ostringstream os;
            os << static_cast<int>(static_cast<float>(frames) / fpsTimer + 0.5f) << " FPS";
            fpsTxt.set(os.str());
            frames = 0;
            fpsTimer = 0.f;
        }

        controls.update();

        // ===== two-pass render =====
        renderer->autoClear = true;
        renderer->render(*scene, *camera);

        renderer->autoClear = false;
        renderer->clearDepth();
        renderer->render(*ui, *uiCam);
        renderedOnce = true;

        if (captureAfter > 0.f && clock.getElapsedTime() >= captureAfter) {
            captureAfter = 0.f;
            renderer->writeFramebuffer("robot_cell_capture.png");
        }
    });
}
