// LensRig — how a fisheye frame is assembled from several pinhole views.
//
// A fisheye lens (LensModel::Fisheye) sees far more than a pinhole can hold:
// 139 degrees across a 16:9 frame, 180 and beyond for an all-sky lens. Warping
// ONE pinhole render into that field (the old setLensOverscan route) spends the
// render's pixels on a frustum four times too wide, so the image centre got a
// quarter of its linear resolution, and the pinhole could never reach 90
// degrees off-axis at all.
//
// The rig keeps the primary render exactly as the camera describes it — a
// pinhole with the fisheye's own centre focal length, so the middle of the
// fisheye frame is as sharp as a plain render of the same camera — and adds up
// to four side views for the directions the primary cannot see. Each side view
// looks along one of the camera's own axes (right, left, down, up: the four
// side faces of a cube around the camera), with an off-axis frustum cropped to
// exactly the directions it is responsible for and a resolution chosen so it
// never has fewer samples than the output pixels that read it (in every
// direction, from the local Jacobian of the output-to-view map).
//
// Who owns a direction:
//   * the primary, if the direction projects inside its frame (less a margin
//     of a couple of output pixels for the bilinear tap and TAA's edge);
//   * otherwise the side face whose side of the primary frustum it leaves
//     through, i.e. the same diagonal partition a cube map makes, but drawn
//     through the primary frustum's corners instead of through 45 degrees.
//     That keeps every face's region bounded for any aspect ratio.
//
// Everything here is in IMAGE coordinates: x right, y DOWN, z forward (the
// optical axis), which is what the normalized lens coordinates (top-left
// origin, OpenCV's convention) already use. The same functions run in three
// places — the face layout (here, at set-up), the per-pixel resample
// (sensor_image.comp mirrors them) and the AOV readback (VulkanRenderer.cpp) —
// so colour and labels agree on which view a pixel came from.

#ifndef THREEPP_VULKAN_LENS_RIG_HPP
#define THREEPP_VULKAN_LENS_RIG_HPP

#include "threepp/cameras/LensDistortion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace threepp::vulkan::lensrig {

    // The four side faces, indexed by the camera axis they look along.
    enum Face : int { Right = 0, Left = 1, Down = 2, Up = 3 };
    inline constexpr int kFaceCount = 4;

    // A face's camera basis in image coordinates: r (its image x), dn (its
    // image y, down) and f (its forward), right-handed like the primary's
    // (1,0,0), (0,1,0), (0,0,1) — every face is a proper rotation of it.
    struct Basis {
        float r[3], dn[3], f[3];
    };
    inline constexpr Basis kBasis[kFaceCount] = {
            {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}}, // Right
            {{0, 0, 1}, {0, 1, 0}, {-1, 0, 0}}, // Left
            {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}}, // Down
            {{1, 0, 0}, {0, 0, 1}, {0, -1, 0}}, // Up
    };

    inline float dot3(const float* a, const float* b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    // The deepest a fisheye ray may look: 170 degrees off the axis (a 340 degree
    // field). The side faces cover a hemisphere and a bit; this bounds theta so
    // a coefficient set whose polynomial folds back cannot point a ray down the
    // camera's own back axis, where no face looks.
    inline constexpr float kMaxTheta = 2.96706f;

    // Fisheye pixel -> unit ray direction (image coordinates). Same Newton
    // inversion of theta_d = theta (1 + k1 t^2 + ...) as lensUndistort, but it
    // returns a DIRECTION rather than a pinhole point, so it is not bound to
    // theta < 90 degrees.
    inline void fisheyeDirection(const LensDistortion& d, float xd, float yd, float out[3]) {
        const float rd = std::sqrt(xd * xd + yd * yd);
        if (rd < 1e-8f) {
            out[0] = 0.f;
            out[1] = 0.f;
            out[2] = 1.f;
            return;
        }
        float th = rd;
        for (int i = 0; i < kLensInverseIterations; ++i) {
            const float t2   = th * th;
            const float poly = 1.f + t2 * (d.k1 + t2 * (d.k2 + t2 * (d.k3 + t2 * d.k4)));
            const float fv   = th * poly - rd;
            if (std::abs(fv) <= 1e-7f * rd) break;// converged (3-4 steps)
            const float df = 1.f + t2 * (3.f * d.k1 + t2 * (5.f * d.k2 + t2 * (7.f * d.k3 + t2 * 9.f * d.k4)));
            if (std::abs(df) < 1e-6f) break;
            th -= fv / df;
        }
        th = std::clamp(th, 0.f, kMaxTheta);
        const float s = std::sin(th) / rd;
        out[0] = xd * s;
        out[1] = yd * s;
        out[2] = std::cos(th);
    }

    // One side face as rendered: its frustum in tan units of its own image
    // plane, u in [u0, u1] (its x), v in [v0, v1] (its y, down), and its size.
    struct FaceRect {
        float    u0 = 0.f, u1 = 0.f, v0 = 0.f, v1 = 0.f;
        uint32_t width = 0, height = 0;
    };

    // Everything a pixel's resample needs, for one frame.
    struct Frame {
        // Normalized intrinsics of the OUTPUT camera (fx/W, fy/H, cx/W, cy/H),
        // top-left origin: the primary pinhole and the fisheye share them.
        float normK[4] = {1.f, 1.f, 0.5f, 0.5f};
        // The primary owns a direction whose uv lies in [margin, 1 - margin].
        float marginU = 0.f, marginV = 0.f;
        uint32_t faceMask = 0u;// bit f set = face f exists and is rendered
        std::array<FaceRect, kFaceCount> faces{};
    };

    // Which view a direction is read from: -1 the primary, else a face, with
    // the sample position in that view's uv (top-left origin, [0, 1]).
    inline int owner(const Frame& fr, const float d[3], float& su, float& sv) {
        const float fx = fr.normK[0], fy = fr.normK[1], cx = fr.normK[2], cy = fr.normK[3];
        if (d[2] > 1e-6f) {
            su = d[0] / d[2] * fx + cx;
            sv = d[1] / d[2] * fy + cy;
            if (su >= fr.marginU && su <= 1.f - fr.marginU && sv >= fr.marginV && sv <= 1.f - fr.marginV) {
                return -1;
            }
        }
        // The side of the primary frustum this direction leaves through: the
        // diagonal through the frustum's corners (|x|/tanX vs |y|/tanY).
        const float tanX = d[0] > 0.f ? (1.f - cx) / fx : cx / fx;
        const float tanY = d[1] > 0.f ? (1.f - cy) / fy : cy / fy;
        const int   f    = std::abs(d[0]) * tanY >= std::abs(d[1]) * tanX ? (d[0] > 0.f ? Right : Left)
                                                                          : (d[1] > 0.f ? Down : Up);
        if ((fr.faceMask & (1u << f)) == 0u) {
            // No face there (not created yet, or a field the primary covers
            // without one): the primary's nearest edge, as a pinhole warp
            // with no overscan would give.
            if (d[2] <= 1e-6f) {
                su = d[0] > 0.f ? 1.f : (d[0] < 0.f ? 0.f : 0.5f);
                sv = d[1] > 0.f ? 1.f : (d[1] < 0.f ? 0.f : 0.5f);
            }
            su = std::clamp(su, 0.f, 1.f);
            sv = std::clamp(sv, 0.f, 1.f);
            return -1;
        }
        const Basis&    b  = kBasis[f];
        const FaceRect& r  = fr.faces[f];
        const float     dz = std::max(dot3(d, b.f), 1e-6f);
        const float     u  = dot3(d, b.r) / dz;
        const float     v  = dot3(d, b.dn) / dz;
        su = (u - r.u0) / (r.u1 - r.u0);
        sv = (v - r.v0) / (r.v1 - r.v0);
        return f;
    }

    namespace detail {
        // One face's running bounds and resolution demand while laying out.
        struct FaceAcc {
            float u0 = 1e30f, u1 = -1e30f, v0 = 1e30f, v1 = -1e30f;
            float density = 0.f;// face focal length (tan units -> px) needed
            bool  any     = false;
        };
    }// namespace detail

    // Lay the faces out for a fisheye lens on a `width` x `height` output.
    // Scans the output (every second pixel and the whole border), assigns each
    // sample an owner with all four faces allowed, and gives each face the
    // tightest frustum around its samples plus a pad, at the resolution the
    // most demanding of its samples needs: an output step of one pixel in any
    // direction must cross at least one face pixel (the smallest singular value
    // of the output-to-face Jacobian). `maxSide` caps a face's larger side.
    inline Frame layout(const LensDistortion& lens, const float normK[4], uint32_t width, uint32_t height,
                        uint32_t maxSide = 4096u) {
        Frame fr;
        for (int i = 0; i < 4; ++i) fr.normK[i] = normK[i];
        if (width == 0u || height == 0u) return fr;
        constexpr float kMarginPx = 3.f;// Catmull-Rom reaches two texels out
        fr.marginU  = kMarginPx / float(width);
        fr.marginV  = kMarginPx / float(height);
        fr.faceMask = (1u << kFaceCount) - 1u;// every face allowed while scanning

        std::array<detail::FaceAcc, kFaceCount> acc{};
        // Tan bound for a face's plane: 76 degrees off its axis. Anything beyond
        // belongs to a neighbouring face for every field up to kMaxTheta.
        constexpr float kTanCap = 4.f;

        auto dirAt = [&](float px, float py, float* out) {
            const float xd = ((px + 0.5f) / float(width) - normK[2]) / normK[0];
            const float yd = ((py + 0.5f) / float(height) - normK[3]) / normK[1];
            fisheyeDirection(lens, xd, yd, out);
        };
        auto faceUv = [&](int f, const float* d, float& u, float& v) {
            const Basis& b  = kBasis[f];
            const float  dz = dot3(d, b.f);
            if (dz <= 1e-4f) return false;
            u = dot3(d, b.r) / dz;
            v = dot3(d, b.dn) / dz;
            return true;
        };
        // Bounds of a face's region, from one sample.
        auto bound = [&](const float* d) {
            float su, sv;
            const int f = owner(fr, d, su, sv);
            if (f < 0) return -1;
            float u, v;
            if (!faceUv(f, d, u, v)) return -1;
            detail::FaceAcc& a = acc[f];
            a.any = true;
            a.u0  = std::min(a.u0, std::clamp(u, -kTanCap, kTanCap));
            a.u1  = std::max(a.u1, std::clamp(u, -kTanCap, kTanCap));
            a.v0  = std::min(a.v0, std::clamp(v, -kTanCap, kTanCap));
            a.v1  = std::max(a.v1, std::clamp(v, -kTanCap, kTanCap));
            return f;
        };

        // A grid every second pixel, with the last row and column on it: one
        // direction per sample, and its neighbours on the grid give the
        // output-to-face Jacobian (per output pixel: divided by the step).
        std::vector<uint32_t> gx, gy;
        for (uint32_t x = 0; x < width; x += 2u) gx.push_back(x);
        if (gx.back() != width - 1u) gx.push_back(width - 1u);
        for (uint32_t y = 0; y < height; y += 2u) gy.push_back(y);
        if (gy.back() != height - 1u) gy.push_back(height - 1u);
        const size_t nx = gx.size(), ny = gy.size();
        std::vector<float> dirs(nx * ny * 3u);
        for (size_t j = 0; j < ny; ++j)
            for (size_t i = 0; i < nx; ++i) dirAt(float(gx[i]), float(gy[j]), &dirs[(j * nx + i) * 3u]);
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                const float* d = &dirs[(j * nx + i) * 3u];
                const int    f = bound(d);
                if (f < 0 || nx < 2u || ny < 2u) continue;
                // Forward differences, backward on the last row / column.
                const size_t i2 = i + 1u < nx ? i + 1u : i - 1u;
                const size_t j2 = j + 1u < ny ? j + 1u : j - 1u;
                const float  sx = float(gx[i2]) - float(gx[i]);
                const float  sy = float(gy[j2]) - float(gy[j]);
                float u, v, ux, vx, uy, vy;
                if (!faceUv(f, d, u, v) || !faceUv(f, &dirs[(j * nx + i2) * 3u], ux, vx) ||
                    !faceUv(f, &dirs[(j2 * nx + i) * 3u], uy, vy)) {
                    continue;
                }
                const float a11 = (ux - u) / sx, a21 = (vx - v) / sx, a12 = (uy - u) / sy, a22 = (vy - v) / sy;
                // Smallest singular value of [[a11 a12] [a21 a22]].
                const float p    = a11 * a11 + a21 * a21;
                const float q    = a12 * a12 + a22 * a22;
                const float r    = a11 * a12 + a21 * a22;
                const float tr   = 0.5f * (p + q);
                const float det  = std::sqrt(std::max(0.f, 0.25f * (p - q) * (p - q) + r * r));
                const float smin = std::sqrt(std::max(tr - det, 0.f));
                if (smin > 1e-9f) acc[f].density = std::max(acc[f].density, 1.f / smin);
            }
        }
        // The whole border, for the bounds: a face's outermost extent is there.
        float d[3];
        for (uint32_t x = 0; x < width; ++x) {
            dirAt(float(x), 0.f, d);
            bound(d);
            dirAt(float(x), float(height - 1u), d);
            bound(d);
        }
        for (uint32_t y = 0; y < height; ++y) {
            dirAt(0.f, float(y), d);
            bound(d);
            dirAt(float(width - 1u), float(y), d);
            bound(d);
        }

        fr.faceMask = 0u;
        for (int f = 0; f < kFaceCount; ++f) {
            const detail::FaceAcc& a = acc[f];
            if (!a.any || a.density <= 0.f) continue;
            float density = a.density;
            // Pad by a few face pixels: the scan steps two output pixels, the
            // bilinear tap reaches one texel out, and TAA's border is its
            // weakest region.
            constexpr float kPadPx = 4.f;
            float spanU = (a.u1 - a.u0) + 2.f * kPadPx / density;
            float spanV = (a.v1 - a.v0) + 2.f * kPadPx / density;
            const float side = std::max(spanU, spanV) * density;
            if (side > float(maxSide)) {
                density *= float(maxSide) / side;
                spanU = (a.u1 - a.u0) + 2.f * kPadPx / density;
                spanV = (a.v1 - a.v0) + 2.f * kPadPx / density;
            }
            FaceRect& r = fr.faces[f];
            r.width     = std::max(8u, static_cast<uint32_t>(std::ceil(spanU * density)));
            r.height    = std::max(8u, static_cast<uint32_t>(std::ceil(spanV * density)));
            const float cu = 0.5f * (a.u0 + a.u1), cv = 0.5f * (a.v0 + a.v1);
            // The rect's tan extent follows its whole pixel count, so face
            // pixels are square in tan units (one focal length per face).
            r.u0 = cu - 0.5f * float(r.width) / density;
            r.u1 = cu + 0.5f * float(r.width) / density;
            r.v0 = cv - 0.5f * float(r.height) / density;
            r.v1 = cv + 0.5f * float(r.height) / density;
            fr.faceMask |= 1u << f;
        }
        return fr;
    }

}// namespace threepp::vulkan::lensrig

#endif//THREEPP_VULKAN_LENS_RIG_HPP
