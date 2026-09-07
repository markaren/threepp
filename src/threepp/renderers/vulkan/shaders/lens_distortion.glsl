// lens_distortion.glsl — GPU twin of include/threepp/cameras/LensDistortion.hpp.
//
// Keep the two IN LOCK-STEP. The display path warps on the GPU with this file;
// the sensor/AOV readback path warps on the CPU with the C++ header. If they
// disagree, a synthetic RGB frame and its synthetic depth/segmentation labels
// describe different lenses, which is the exact failure the whole feature
// exists to avoid. VulkanCameraModel_test cross-checks them.
//
// Conventions, models and coefficient meanings: see the C++ header. Everything
// here operates in NORMALIZED image coordinates (x = X/Z before intrinsics).

#ifndef LENS_DISTORTION_GLSL
#define LENS_DISTORTION_GLSL

// Must match threepp::LensModel.
#define LENS_MODEL_NONE          0
#define LENS_MODEL_BROWNCONRADY  1
#define LENS_MODEL_FISHEYE       2

// Must match threepp::kLensInverseIterations — the C++ header explains why
// this is 20 and not OpenCV's default 5 (5 leaves ~1.6 px of error under
// pincushion, which defeats the point of matching a real calibration).
#define LENS_INVERSE_ITERATIONS  20

// Coefficients as packed for the push constant: radial = (k1, k2, k3, k4),
// tangential = (p1, p2).

// Forward map: ideal (pinhole) -> distorted. Included for completeness and for
// any pass that needs to know where a scene point actually lands.
vec2 lensDistort(int model, vec4 radial, vec2 tangential, vec2 p) {
    if (model == LENS_MODEL_BROWNCONRADY) {
        float r2 = dot(p, p);
        float rad = 1.0 + r2 * (radial.x + r2 * (radial.y + r2 * radial.z));
        return vec2(
            p.x * rad + 2.0 * tangential.x * p.x * p.y + tangential.y * (r2 + 2.0 * p.x * p.x),
            p.y * rad + tangential.x * (r2 + 2.0 * p.y * p.y) + 2.0 * tangential.y * p.x * p.y);
    } else if (model == LENS_MODEL_FISHEYE) {
        float r = length(p);
        if (r < 1e-8) return p;
        float th = atan(r);
        float t2 = th * th;
        float thD = th * (1.0 + t2 * (radial.x + t2 * (radial.y + t2 * (radial.z + t2 * radial.w))));
        return p * (thD / r);
    }
    return p;
}

// Inverse map: distorted -> ideal (pinhole). This is the one the renderer
// needs — for each output (sensor) pixel, which part of the rendered pinhole
// image does it see.
vec2 lensUndistort(int model, vec4 radial, vec2 tangential, vec2 pd) {
    if (model == LENS_MODEL_BROWNCONRADY) {
        // OpenCV undistortPoints' fixed point.
        vec2 p = pd;
        for (int i = 0; i < LENS_INVERSE_ITERATIONS; ++i) {
            float r2  = dot(p, p);
            float rad = 1.0 + r2 * (radial.x + r2 * (radial.y + r2 * radial.z));
            float inv = abs(rad) > 1e-6 ? 1.0 / rad : 1.0;
            vec2  d   = vec2(
                2.0 * tangential.x * p.x * p.y + tangential.y * (r2 + 2.0 * p.x * p.x),
                tangential.x * (r2 + 2.0 * p.y * p.y) + 2.0 * tangential.y * p.x * p.y);
            p = (pd - d) * inv;
        }
        return p;
    } else if (model == LENS_MODEL_FISHEYE) {
        float rd = length(pd);
        if (rd < 1e-8) return pd;
        float th = rd;// theta_d is a good first guess for theta
        for (int i = 0; i < LENS_INVERSE_ITERATIONS; ++i) {
            float t2   = th * th;
            float poly = 1.0 + t2 * (radial.x + t2 * (radial.y + t2 * (radial.z + t2 * radial.w)));
            float f    = th * poly - rd;
            float df   = 1.0 + t2 * (3.0 * radial.x + t2 * (5.0 * radial.y +
                                     t2 * (7.0 * radial.z + t2 * 9.0 * radial.w)));
            if (abs(df) < 1e-6) break;
            th -= f / df;
        }
        // Clamp just short of the horizon (pi/2), where an ideal pinhole image
        // would need infinite extent — see the C++ header.
        const float kMaxTheta = 1.55334;// ~89 degrees
        th = clamp(th, -kMaxTheta, kMaxTheta);
        return pd * (tan(th) / rd);
    }
    return pd;
}

#endif// LENS_DISTORTION_GLSL
