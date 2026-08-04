// Inverse kinematics for a Robot, by damped least squares.
//
// Each iteration finite-differences a 6 x n Jacobian of the tool pose with
// respect to the joint values, then takes a Gauss-Newton step damped by
// lambda^2 so the solve stays well behaved through singularities instead of
// flinging the arm. A redundant arm additionally gets a null-space pull toward
// a rest posture, which is what keeps a 7-DOF elbow in a comfortable
// configuration rather than wherever the pseudo-inverse happens to leave it.
//
// The Jacobian is differenced rather than derived because Robot exposes only
// forward kinematics, and one FK evaluation per DOF per iteration is cheap at
// arm scale. No SVD or eigensolver is required, so this has no external
// dependencies. Header-only.
//
// Only the joints on the robot's root-to-tool path are solved for — see
// Robot::chainDofs(). A gripper's finger joints keep their slots in the joint
// vector but are never touched, so closing the hand can never be mistaken for
// a way to reach further.

#ifndef THREEPP_KINEMATICS_INVERSEKINEMATICS_HPP
#define THREEPP_KINEMATICS_INVERSEKINEMATICS_HPP

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Robot.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace threepp {

    // How much of the tool pose the solve is required to reproduce.
    enum class IkTask {
        Position, // 3-DOF: reach the point, any orientation
        AxisAlign,// 5-DOF: reach the point AND aim the tool axis; spin about
                  //        that axis is left free, which is what a drill, a
                  //        suction cup or a symmetric two-finger grasp wants
        Pose      // 6-DOF: reproduce the full target transform
    };

    struct IkOptions {

        IkTask task = IkTask::Position;

        // Flange -> tool centre point. The solve drives the TCP, so a tool of
        // any length or mounting is described here rather than in the URDF.
        Matrix4 toolOffset;

        // AxisAlign only. toolAxis is in the TOOL frame (+Z is the URDF
        // convention for an approach direction); targetAxis is in WORLD space.
        Vector3 toolAxis{0.f, 0.f, 1.f};
        Vector3 targetAxis{0.f, -1.f, 0.f};

        int maxIterations = 100;
        float positionTolerance = 1e-4f;   // metres
        float orientationTolerance = 1e-3f;// radians

        // Largest correction any single iteration will attempt. A Gauss-Newton
        // step is a LOCAL statement: asking one iteration to close a metre of
        // error produces a step far outside the range where the linearisation
        // holds, and the usual result is an arm that overshoots, folds itself
        // into its own joint limits and stays pinned there — converging on
        // nothing while every joint sits hard against a stop. Clamping the
        // error keeps each step inside the trust region; the clamp stops
        // binding near the goal, so the endgame is still full-speed DLS.
        //
        // Travel per solve is therefore bounded by maxIterations * step. Zero
        // disables the clamp.
        float maxPositionStep = 0.05f;   // metres
        float maxOrientationStep = 0.35f;// radians

        // DLS damping. Must be > 0: it is what makes the normal matrix positive
        // definite, and therefore what lets the Cholesky solve succeed at a
        // singularity instead of failing and stalling the arm.
        float damping = 0.12f;

        // Weight on the orientation rows relative to the position rows. Below 1
        // the solver prefers to reach the point and only then straighten up,
        // which reads as natural motion; at 1 it fights position error to hold
        // the tool angle.
        float orientationWeight = 0.3f;

        // Finite-difference probe. Revolute joints are probed in radians and
        // prismatic ones in metres, so one step cannot serve both.
        float revoluteStep = 1e-3f;
        float prismaticStep = 1e-4f;

        // Null-space rest-posture pull, per iteration. Zero disables it. Only
        // does anything on a redundant arm, where the null space is non-empty.
        // restPose is indexed by global DOF, like every other joint vector.
        float restPoseGain = 0.f;
        std::vector<float> restPose;

        // Damping used for the null-space PROJECTION, as opposed to the task
        // step. It has to be much smaller: (I - J^+ J) built at the task's
        // damping leaks O(lambda^2) of the posture bias straight into the tool
        // pose, and because the bias does not vanish at the goal, that leak is
        // permanent — the arm hovers a millimetre off target and never reports
        // convergence. A light damping here keeps the projector honest while
        // still degrading gracefully at a singularity.
        float nullSpaceDamping = 1e-3f;

        // Per-call joint speed cap, in rad/s or m/s, applied against the joint
        // values as they arrived. Zero disables it. This is a controller
        // concern rather than a solver one, but it is cheap here and every
        // real-time caller wants it, so it is offered rather than imposed.
        float maxJointSpeed = 0.f;
    };

    struct IkResult {
        int iterations = 0;
        float positionError = 0.f;   // metres
        float orientationError = 0.f;// radians
        bool converged = false;
    };

    namespace detail {

        // Solve the symmetric positive-definite 6x6 system A x = b by Cholesky.
        // A = J J^T + lambda^2 I is PD for any lambda > 0, so this cannot fail
        // on a well-formed call; it returns false rather than producing noise if
        // a caller sets damping to zero and then hits a singularity.
        inline bool ikCholeskySolve6(const double A[6][6], const double b[6], double x[6]) {
            double L[6][6] = {{0}};
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double sum = A[i][j];
                    for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
                    if (i == j) {
                        if (sum <= 1e-20) return false;
                        L[i][i] = std::sqrt(sum);
                    } else {
                        L[i][j] = sum / L[j][j];
                    }
                }
            }
            double y[6];
            for (int i = 0; i < 6; ++i) {
                double sum = b[i];
                for (int k = 0; k < i; ++k) sum -= L[i][k] * y[k];
                y[i] = sum / L[i][i];
            }
            for (int i = 5; i >= 0; --i) {
                double sum = y[i];
                for (int k = i + 1; k < 6; ++k) sum -= L[k][i] * x[k];
                x[i] = sum / L[i][i];
            }
            return true;
        }

        // The rotation vector (axis * angle) of a quaternion, taking the shorter
        // of the two arcs so a small misalignment never reads as a nearly-full
        // turn the other way.
        inline Vector3 ikRotationVector(const Quaternion& q) {
            float x = q.x, y = q.y, z = q.z, w = q.w;
            if (w < 0.f) {
                x = -x;
                y = -y;
                z = -z;
                w = -w;
            }
            const float n = std::sqrt(x * x + y * y + z * z);
            if (n < 1e-8f) return {0.f, 0.f, 0.f};
            const float k = 2.f * std::atan2(n, w) / n;
            return {x * k, y * k, z * k};
        }

    }// namespace detail

    // Holds a NON-owning reference to its Robot; the Robot must outlive the
    // solver. Joint ranges and the solvable DOF set are cached at construction,
    // so a Robot that is re-parsed or re-articulated needs a fresh solver.
    class IkSolver {

    public:
        explicit IkSolver(const Robot& robot, IkOptions options = {})
            : robot_(&robot), options_(std::move(options)),
              dofs_(robot.chainDofs()), ranges_(robot.getJointRanges(false)) {

            const auto infos = robot.getArticulatedJointInfo();
            types_.reserve(dofs_.size());
            for (const size_t d : dofs_) {
                types_.push_back(infos.at(d).type);
            }
        }

        // The tool centre point for a given pose, in the robot's parent frame.
        // Call robot.updateMatrix() first if the robot itself has moved —
        // computeEndEffectorTransform composes against the robot's local matrix.
        [[nodiscard]] Matrix4 toolTransform(const std::vector<float>& q) const {
            Matrix4 m = robot_->computeEndEffectorTransform(q);
            return m.multiply(options_.toolOffset);
        }

        // Step `q` toward placing the tool at `target`. `q` is a full joint
        // vector indexed by global DOF; only the chain DOFs are modified.
        // `dt` is used solely by the speed cap and may be left at zero.
        IkResult solve(std::vector<float>& q, const Vector3& target, float dt = 0.f) const {
            return solve(q, target, nullptr, dt);
        }

        // Pose-task overload: reach the full target transform.
        IkResult solve(std::vector<float>& q, const Matrix4& target, float dt = 0.f) const {
            Vector3 p, s;
            Quaternion rot;
            target.decompose(p, rot, s);
            return solve(q, p, &rot, dt);
        }

        [[nodiscard]] const IkOptions& options() const { return options_; }
        IkOptions& options() { return options_; }

        // The DOF indices this solver is allowed to move.
        [[nodiscard]] const std::vector<size_t>& solvedDofs() const { return dofs_; }

    private:
        IkResult solve(std::vector<float>& q, const Vector3& targetPos, const Quaternion* targetRot, float dt) const {

            IkResult result;
            const size_t n = dofs_.size();
            if (n == 0) return result;

            const std::vector<float> q0 = q;
            const float w = options_.orientationWeight;
            const std::array<float, 6> rowWeight{1.f, 1.f, 1.f, w, w, w};

            std::array<float, 6> f{};
            std::vector<std::array<float, 6>> J(n);
            std::vector<float> qh = q;

            for (int iter = 0; iter < options_.maxIterations; ++iter) {

                residual(q, targetPos, targetRot, f);
                result.iterations = iter + 1;
                result.positionError = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
                result.orientationError = std::sqrt(f[3] * f[3] + f[4] * f[4] + f[5] * f[5]);

                if (result.positionError < options_.positionTolerance &&
                    result.orientationError < options_.orientationTolerance) {
                    result.converged = true;
                    break;
                }

                // Finite-difference the residual, one FK evaluation per DOF.
                for (size_t k = 0; k < n; ++k) {
                    const size_t d = dofs_[k];
                    const float h = types_[k] == Robot::JointType::Prismatic
                                            ? options_.prismaticStep
                                            : options_.revoluteStep;
                    qh[d] = q[d] + h;
                    std::array<float, 6> fh{};
                    residual(qh, targetPos, targetRot, fh);
                    qh[d] = q[d];
                    for (int r = 0; r < 6; ++r) {
                        J[k][r] = (fh[r] - f[r]) / h;
                    }
                }

                // The residual actually stepped along: the true one, shortened
                // to the trust region. Convergence is still judged on the true
                // error above, so clamping never fakes a converged result.
                std::array<float, 6> fs = f;
                if (options_.maxPositionStep > 0.f && result.positionError > options_.maxPositionStep) {
                    const float s = options_.maxPositionStep / result.positionError;
                    fs[0] *= s;
                    fs[1] *= s;
                    fs[2] *= s;
                }
                if (options_.maxOrientationStep > 0.f && result.orientationError > options_.maxOrientationStep) {
                    const float s = options_.maxOrientationStep / result.orientationError;
                    fs[3] *= s;
                    fs[4] *= s;
                    fs[5] *= s;
                }

                // G = J W J^T, shared by both solves below; the two differ only
                // in how much they are damped.
                double G[6][6];
                double b[6];
                for (int r = 0; r < 6; ++r) {
                    for (int c = 0; c < 6; ++c) {
                        double s = 0.0;
                        for (size_t k = 0; k < n; ++k) {
                            s += static_cast<double>(J[k][r] * rowWeight[r]) *
                                 static_cast<double>(J[k][c] * rowWeight[c]);
                        }
                        G[r][c] = s;
                    }
                    b[r] = -static_cast<double>(fs[r] * rowWeight[r]);
                }

                // The task step solves (G + lambda^2 I) y = -W f.
                double A[6][6];
                const double lam2 = static_cast<double>(options_.damping) * options_.damping;
                for (int r = 0; r < 6; ++r) {
                    for (int c = 0; c < 6; ++c) A[r][c] = G[r][c] + (r == c ? lam2 : 0.0);
                }

                double y[6];
                if (!detail::ikCholeskySolve6(A, b, y)) break;

                // Null-space rest-posture bias: v pulled toward the rest pose,
                // projected through (I - J^+ J) so it cannot disturb the tool.
                // The projector gets its own, far lighter damping — see
                // IkOptions::nullSpaceDamping for why sharing the task's would
                // stop the arm converging at all.
                std::vector<float> v;
                double z[6]{};
                bool haveNull = false;
                if (options_.restPoseGain != 0.f && !options_.restPose.empty()) {
                    v.resize(n);
                    double Jv[6]{};
                    for (size_t k = 0; k < n; ++k) {
                        const size_t d = dofs_[k];
                        const float rest = d < options_.restPose.size() ? options_.restPose[d] : 0.f;
                        v[k] = options_.restPoseGain * (rest - q[d]);
                        for (int r = 0; r < 6; ++r) {
                            Jv[r] += static_cast<double>(J[k][r] * rowWeight[r]) * v[k];
                        }
                    }

                    double An[6][6];
                    const double nlam2 = static_cast<double>(options_.nullSpaceDamping) * options_.nullSpaceDamping;
                    for (int r = 0; r < 6; ++r) {
                        for (int c = 0; c < 6; ++c) An[r][c] = G[r][c] + (r == c ? nlam2 : 0.0);
                    }

                    // A failure here means the projector is degenerate — at a
                    // singularity the safe answer is no posture bias at all,
                    // not an unprojected one that would fight the tool.
                    haveNull = detail::ikCholeskySolve6(An, Jv, z);
                }

                for (size_t k = 0; k < n; ++k) {
                    const size_t d = dofs_[k];
                    double dq = 0.0;
                    for (int r = 0; r < 6; ++r) {
                        dq += static_cast<double>(J[k][r] * rowWeight[r]) * y[r];
                    }
                    if (haveNull) {
                        double jz = 0.0;
                        for (int r = 0; r < 6; ++r) {
                            jz += static_cast<double>(J[k][r] * rowWeight[r]) * z[r];
                        }
                        dq += v[k] - jz;
                    }
                    q[d] = ranges_.at(d).clamp(q[d] + static_cast<float>(dq));
                    qh[d] = q[d];
                }
            }

            // Speed cap, applied against the pose this call started from.
            if (options_.maxJointSpeed > 0.f && dt > 0.f) {
                const float maxDq = options_.maxJointSpeed * dt;
                for (const size_t d : dofs_) {
                    q[d] = q0[d] + std::clamp(q[d] - q0[d], -maxDq, maxDq);
                }
            }

            // Report the error actually left on the table, after the cap.
            residual(q, targetPos, targetRot, f);
            result.positionError = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
            result.orientationError = std::sqrt(f[3] * f[3] + f[4] * f[4] + f[5] * f[5]);
            result.converged = result.positionError < options_.positionTolerance &&
                               result.orientationError < options_.orientationTolerance;
            return result;
        }

        // The task residual, unweighted, to be driven to zero. Position rows
        // first, then the orientation term the task calls for.
        void residual(const std::vector<float>& q, const Vector3& targetPos,
                      const Quaternion* targetRot, std::array<float, 6>& f) const {

            const Matrix4 tcp = toolTransform(q);
            Vector3 p, s;
            Quaternion rot;
            tcp.decompose(p, rot, s);

            f[0] = p.x - targetPos.x;
            f[1] = p.y - targetPos.y;
            f[2] = p.z - targetPos.z;
            f[3] = f[4] = f[5] = 0.f;

            if (options_.task == IkTask::AxisAlign) {

                // The full angle between the axes, along the axis of the
                // correction — NOT sin(angle) via the cross product alone.
                // The cross product vanishes for anti-parallel axes as well as
                // parallel ones, which makes "tool pointing exactly backwards"
                // a fixed point: a solve can converge, report zero error, and
                // leave the tool inverted. Taking the angle from atan2 makes
                // that configuration the maximum residual instead.
                Vector3 a = options_.toolAxis;
                a.applyQuaternion(rot).normalize();
                Vector3 t = options_.targetAxis;
                t.normalize();

                const Vector3 c = Vector3().crossVectors(a, t);
                const float sn = c.length();
                const float angle = std::atan2(sn, a.dot(t));

                Vector3 axis;
                if (sn > 1e-6f) {
                    axis.copy(c).divideScalar(sn);
                } else if (angle > 1.f) {
                    // Exactly anti-parallel: every perpendicular is an equally
                    // good way out, so take one and let the next iteration pick
                    // up the now well-conditioned gradient.
                    Vector3 perp{1.f, 0.f, 0.f};
                    if (std::abs(a.x) > 0.9f) perp.set(0.f, 1.f, 0.f);
                    axis.crossVectors(a, perp).normalize();
                }

                f[3] = -axis.x * angle;
                f[4] = -axis.y * angle;
                f[5] = -axis.z * angle;

            } else if (options_.task == IkTask::Pose && targetRot) {

                // The rotation carrying the target orientation onto the current
                // one; zero exactly when they agree.
                Quaternion err;
                err.copy(rot);
                Quaternion inv;
                inv.copy(*targetRot);
                inv.invert();
                err.multiply(inv);

                const Vector3 rv = detail::ikRotationVector(err);
                f[3] = rv.x;
                f[4] = rv.y;
                f[5] = rv.z;
            }
        }

        const Robot* robot_;
        IkOptions options_;
        std::vector<size_t> dofs_;
        std::vector<Robot::JointRange> ranges_;
        std::vector<Robot::JointType> types_;
    };

}// namespace threepp

#endif//THREEPP_KINEMATICS_INVERSEKINEMATICS_HPP
