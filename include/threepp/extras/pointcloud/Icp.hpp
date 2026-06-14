// Point-to-point ICP registration of a point cloud against a VoxelGrid.
//
// Linearized (small-angle) Gauss-Newton: each iteration builds a 6x6 normal
// system from point-to-point correspondences and solves it by Cholesky with
// Levenberg damping and Geman-McClure robust weighting. No SVD or eigensolver
// is required, so it has no external dependencies. Header-only.

#ifndef THREEPP_ICP_HPP
#define THREEPP_ICP_HPP

#include "threepp/extras/pointcloud/VoxelGrid.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace threepp {

    struct IcpOptions {
        int maxIterations = 20;
        float maxCorrespondenceDistance = 0.5f;// starting gate; decays each iteration
        float minCorrespondenceDistance = 0.2f;// floor for the decaying gate
        float robustSigma = 0.3f;              // Geman-McClure scale (metres)
    };

    struct IcpResult {
        int iterations = 0;
        int correspondences = 0;
        bool converged = false;
    };

    namespace detail {

        // Solve the symmetric positive-definite 6x6 system A x = b by Cholesky.
        // Returns false if A is not PD (degenerate / rank-deficient geometry).
        inline bool choleskySolve6(const double A[6][6], const double b[6], double x[6]) {
            double L[6][6] = {{0}};
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double sum = A[i][j];
                    for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
                    if (i == j) {
                        if (sum <= 1e-12) return false;
                        L[i][j] = std::sqrt(sum);
                    } else {
                        L[i][j] = sum / L[j][j];
                    }
                }
            }
            double y[6];
            for (int i = 0; i < 6; ++i) {
                double s = b[i];
                for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
                y[i] = s / L[i][i];
            }
            for (int i = 5; i >= 0; --i) {
                double s = y[i];
                for (int k = i + 1; k < 6; ++k) s -= L[k][i] * x[k];
                x[i] = s / L[i][i];
            }
            return true;
        }

        // Strip accumulated non-orthogonality left by the first-order updates.
        inline void reorthonormalize(Matrix4& t) {
            Vector3 p, s;
            Quaternion q;
            t.decompose(p, q, s);
            q.normalize();
            t.compose(p, q, Vector3(1, 1, 1));
        }

    }// namespace detail

    /**
     * Register a source point cloud against a VoxelGrid target using linearized
     * point-to-point ICP.
     *
     * `pose` maps the source points into the target's frame and must be seeded
     * with a reasonable initial guess (e.g. a motion-model prediction); it is
     * refined in place. Increments are applied on the left (premultiplied),
     * because the per-correspondence Jacobian is expressed about the global,
     * already-transformed source points.
     *
     * The target's voxelSize should be >= maxCorrespondenceDistance so the
     * grid's 27-cell nearest-neighbour search is exact.
     */
    inline IcpResult icpPointToPoint(const std::vector<Vector3>& source, const VoxelGrid& target,
                                     Matrix4& pose, const IcpOptions& opts = {}) {
        IcpResult result;
        if (source.empty() || target.empty()) return result;

        const double s2 = static_cast<double>(opts.robustSigma) * opts.robustSigma;

        for (int iter = 0; iter < opts.maxIterations; ++iter) {
            const float corr = std::max(opts.minCorrespondenceDistance,
                                        opts.maxCorrespondenceDistance * std::pow(0.9f, static_cast<float>(iter)));
            const float corr2 = corr * corr;

            double A[6][6] = {{0}};
            double g[6] = {0};
            int count = 0;

            Vector3 q, m;
            for (const auto& sp : source) {
                q = sp;
                q.applyMatrix4(pose);// source point in the target frame
                if (!target.nearest(q, corr, m)) continue;

                const double dx = m.x - q.x, dy = m.y - q.y, dz = m.z - q.z;
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 > corr2) continue;

                // Geman-McClure robust weight.
                double w = s2 / (s2 + d2);
                w *= w;

                // Residual model: m - q = (-[q]x) * omega + I * t.
                // Jacobian J (3x6), columns = [omega(3) | t(3)].
                const double qx = q.x, qy = q.y, qz = q.z;
                const double J[3][6] = {
                        {0, qz, -qy, 1, 0, 0},
                        {-qz, 0, qx, 0, 1, 0},
                        {qy, -qx, 0, 0, 0, 1}};
                const double dv[3] = {dx, dy, dz};
                for (int r = 0; r < 3; ++r) {
                    for (int a = 0; a < 6; ++a) {
                        g[a] += w * J[r][a] * dv[r];
                        for (int b = 0; b < 6; ++b) A[a][b] += w * J[r][a] * J[r][b];
                    }
                }
                ++count;
            }

            result.correspondences = count;
            if (count < 10) break;// too few correspondences — keep the current pose

            // Levenberg damping keeps A SPD and the solve well-conditioned.
            for (int i = 0; i < 6; ++i) A[i][i] += 1e-4 * A[i][i] + 1e-9;

            double x[6];
            if (!detail::choleskySolve6(A, g, x)) break;// degenerate — keep current pose

            const Vector3 omega(static_cast<float>(x[0]), static_cast<float>(x[1]), static_cast<float>(x[2]));
            const Vector3 trans(static_cast<float>(x[3]), static_cast<float>(x[4]), static_cast<float>(x[5]));
            const float ang = omega.length();
            if (ang > 0.5f) break;// implausible step — bad correspondences, reject

            Quaternion dq;// identity by default
            if (ang > 1e-9f) {
                dq.setFromAxisAngle(Vector3(omega.x / ang, omega.y / ang, omega.z / ang), ang);
            }
            Matrix4 inc;
            inc.compose(trans, dq, Vector3(1, 1, 1));
            pose.premultiply(inc);

            result.iterations = iter + 1;
            if (trans.length() < 1e-4f && ang < 1e-4f) {
                result.converged = true;
                break;
            }
        }

        detail::reorthonormalize(pose);
        return result;
    }

}// namespace threepp

#endif//THREEPP_ICP_HPP
