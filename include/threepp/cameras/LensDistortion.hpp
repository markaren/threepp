// LensDistortion — OpenCV-compatible lens models.
//
// A pinhole projection is what a renderer naturally produces and what no real
// lens actually does. To compare a synthetic frame against a real camera — or
// to train on synthetic frames and deploy on a real one — the sim has to bend
// light the same way the real lens does, using the same coefficients the real
// lens was calibrated with.
//
// The conventions here are OpenCV's exactly, so the k/p values out of
// `cv::calibrateCamera` (or a ROS `camera_info` / calibration YAML) drop
// straight in with no re-derivation:
//
//   BrownConrady  — cv::projectPoints / cv::undistortPoints, coefficients
//                   (k1, k2, p1, p2, k3). Ordinary rectilinear lenses.
//   Fisheye       — cv::fisheye, Kannala-Brandt equidistant, coefficients
//                   (k1, k2, k3, k4) over the incidence angle theta. Wide and
//                   ultra-wide lenses, where BrownConrady stops converging.
//
// All functions work in NORMALIZED image coordinates — the pinhole projection
// x = X/Z, y = Y/Z, BEFORE the intrinsics are applied — which is where the
// distortion physically acts and what keeps these independent of resolution.
//
// `lensDistort` is the forward map (ideal -> distorted): where a scene point
// actually lands on the sensor. `lensUndistort` is its inverse (distorted ->
// ideal): what a renderer needs, because rendering a distorted image means
// asking, for each output pixel, which part of the ideal pinhole image it sees.
// The inverse has no closed form in either model, so both iterate — the same
// fixed-point / Newton schemes OpenCV uses.
//
// This header is mirrored by
// `src/threepp/renderers/vulkan/shaders/lens_distortion.glsl`, which the
// display-path warp uses on the GPU while the sensor/AOV readback path uses
// this one on the CPU. The two must agree or synthetic labels stop lining up
// with synthetic pixels — VulkanCameraModel_test cross-checks them by reading
// an instance id out of a distorted AOV at a pixel picked in the distorted
// colour image.

#ifndef THREEPP_LENSDISTORTION_HPP
#define THREEPP_LENSDISTORTION_HPP

#include <cmath>

namespace threepp {

    enum class LensModel {
        None = 0,        // ideal pinhole — every function below is identity
        BrownConrady = 1,// rectilinear: radial k1..k3 + tangential p1/p2
        Fisheye = 2      // Kannala-Brandt equidistant: radial k1..k4 over theta
    };

    struct LensDistortion {

        LensModel model = LensModel::None;

        // Radial. BrownConrady uses k1..k3 (powers of r²); Fisheye uses k1..k4
        // (powers of theta²).
        float k1 = 0.f, k2 = 0.f, k3 = 0.f, k4 = 0.f;

        // Tangential (decentring). BrownConrady only; ignored by Fisheye,
        // which has no tangential term in OpenCV's model.
        float p1 = 0.f, p2 = 0.f;

        [[nodiscard]] bool active() const {
            return model != LensModel::None;
        }

        [[nodiscard]] bool operator==(const LensDistortion& o) const {
            return model == o.model && k1 == o.k1 && k2 == o.k2 && k3 == o.k3 &&
                   k4 == o.k4 && p1 == o.p1 && p2 == o.p2;
        }
    };

    // Iterations for the inverse maps.
    //
    // OpenCV's undistortPoints defaults to 5, which is NOT enough here. The
    // Brown-Conrady fixed point converges geometrically but slowly, and
    // pincushion is its worst case (the first estimate overshoots outward, so
    // it oscillates in). Measured worst-case residual over normalized radii up
    // to ~1.0, k1 = +0.22:
    //
    //   5 iters -> 5.5e-3    20 iters -> 1.4e-7
    //  10 iters -> 1.6e-4    30 iters -> 1.2e-10
    //
    // At a typical fx of ~300 px, 5.5e-3 normalized is 1.6 PIXELS of error —
    // enough to break the one thing this feature exists to provide, agreement
    // with a real calibration. 20 lands under a thousandth of a pixel, and the
    // cost is a few hundred flops in a once-per-pixel final pass, which is
    // nothing next to the frame that preceded it. Fisheye's Newton iteration
    // converges quadratically and is done in 3; it just shares the constant.
    inline constexpr int kLensInverseIterations = 20;

    // ── Forward: ideal (pinhole) normalized -> distorted normalized ──────────
    inline void lensDistort(const LensDistortion& d, float x, float y, float& xd, float& yd) {
        switch (d.model) {
            case LensModel::BrownConrady: {
                const float r2 = x * x + y * y;
                const float radial = 1.f + r2 * (d.k1 + r2 * (d.k2 + r2 * d.k3));
                xd = x * radial + 2.f * d.p1 * x * y + d.p2 * (r2 + 2.f * x * x);
                yd = y * radial + d.p1 * (r2 + 2.f * y * y) + 2.f * d.p2 * x * y;
                break;
            }
            case LensModel::Fisheye: {
                const float r = std::sqrt(x * x + y * y);
                if (r < 1e-8f) {
                    xd = x;
                    yd = y;
                    break;
                }
                const float th  = std::atan(r);// incidence angle
                const float t2  = th * th;
                const float thD = th * (1.f + t2 * (d.k1 + t2 * (d.k2 + t2 * (d.k3 + t2 * d.k4))));
                const float s   = thD / r;
                xd = x * s;
                yd = y * s;
                break;
            }
            case LensModel::None:
            default:
                xd = x;
                yd = y;
                break;
        }
    }

    // ── Inverse: distorted normalized -> ideal (pinhole) normalized ─────────
    inline void lensUndistort(const LensDistortion& d, float xd, float yd, float& x, float& y) {
        switch (d.model) {
            case LensModel::BrownConrady: {
                // OpenCV's undistortPoints fixed point: repeatedly subtract the
                // tangential term and divide out the radial one, re-evaluating
                // both at the improved estimate.
                x = xd;
                y = yd;
                for (int i = 0; i < kLensInverseIterations; ++i) {
                    const float r2 = x * x + y * y;
                    const float radial = 1.f + r2 * (d.k1 + r2 * (d.k2 + r2 * d.k3));
                    // A pathological coefficient set can fold the radial term
                    // through zero; refusing to divide keeps that a smear at
                    // the frame edge instead of a NaN through the whole image.
                    const float inv = std::abs(radial) > 1e-6f ? 1.f / radial : 1.f;
                    const float dx = 2.f * d.p1 * x * y + d.p2 * (r2 + 2.f * x * x);
                    const float dy = d.p1 * (r2 + 2.f * y * y) + 2.f * d.p2 * x * y;
                    x = (xd - dx) * inv;
                    y = (yd - dy) * inv;
                }
                break;
            }
            case LensModel::Fisheye: {
                // In the equidistant model the distorted radius IS the
                // distorted angle, so invert the odd polynomial in theta by
                // Newton, then go back to a pinhole radius with tan.
                const float rd = std::sqrt(xd * xd + yd * yd);
                if (rd < 1e-8f) {
                    x = xd;
                    y = yd;
                    break;
                }
                float th = rd;// theta_d is a good first guess for theta
                for (int i = 0; i < kLensInverseIterations; ++i) {
                    const float t2 = th * th;
                    const float poly = 1.f + t2 * (d.k1 + t2 * (d.k2 + t2 * (d.k3 + t2 * d.k4)));
                    const float f    = th * poly - rd;
                    // d/dtheta [theta·(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)]
                    const float df = 1.f + t2 * (3.f * d.k1 + t2 * (5.f * d.k2 + t2 * (7.f * d.k3 + t2 * 9.f * d.k4)));
                    if (std::abs(df) < 1e-6f) break;
                    th -= f / df;
                }
                // theta -> pi/2 is the horizon: an ideal pinhole image would
                // need infinite extent to hold it. Clamp just short so the
                // sample lands far outside the frame (and clamps to its edge)
                // rather than producing an infinity.
                constexpr float kMaxTheta = 1.55334f;// ~89 degrees
                th = th < -kMaxTheta ? -kMaxTheta : (th > kMaxTheta ? kMaxTheta : th);
                const float s = std::tan(th) / rd;
                x = xd * s;
                y = yd * s;
                break;
            }
            case LensModel::None:
            default:
                x = xd;
                y = yd;
                break;
        }
    }

}// namespace threepp

#endif//THREEPP_LENSDISTORTION_HPP
