// Internal (src-only) plumbing shared by the raster vision sensors
// (DepthSensor, LidarSensor): the linear-pass renderer guard, the
// depth-linearize post-process, and its RG decode. Both sides of the
// RG encode/decode contract live in THIS file — the shader packs
// R = floor(d*255)/255, G = fract(d*255); the reader reconstructs
// d = R + G/255 in bytes, i.e. px0/255 + px1/65025 (~16-bit precision
// over [0, far]). Previously each sensor carried its own copy of all
// three pieces, including the same hard-won autoClear fix twice.

#ifndef THREEPP_SENSOR_SCAN_UTIL_HPP
#define THREEPP_SENSOR_SCAN_UTIL_HPP

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <memory>

namespace threepp::sensorscan {

    // Sensors render *data* (linearized + packed depth, raw color) into render
    // targets and read it back. Color management (sRGB encode / tone mapping)
    // would corrupt those bytes — silently, on any backend that applies the
    // output encode to render targets. Force a linear, un-tonemapped pass for
    // the scan and restore the renderer's settings afterwards, so callers don't
    // need any backend-specific setup.
    struct DataPassGuard {
        Renderer& r;
        ColorSpace cs;
        ToneMapping tm;
        bool ac;
        explicit DataPassGuard(Renderer& renderer)
            : r(renderer), cs(renderer.outputColorSpace), tm(renderer.toneMapping), ac(renderer.autoClear) {
            r.outputColorSpace = LinearSRGBColorSpace;
            r.toneMapping = ToneMapping::None;
            // The scan re-renders its targets from scratch every call, so the
            // caller's autoClear must not leak in. HUD-overlay apps leave
            // autoClear=false between frames; without a depth clear the
            // post-process quad fails its own depth test (Less vs the equal
            // depth it wrote last scan) and the readback target silently
            // freezes at the previous image.
            r.autoClear = true;
        }
        ~DataPassGuard() {
            r.outputColorSpace = cs;
            r.toneMapping = tm;
            r.autoClear = ac;
        }
    };

    // Post-process material: linearize perspective depth, encode in RG.
    //
    // `near` must be the plane the projection ACTUALLY used, not the sensor's
    // blind-sphere radius — perspectiveDepthToViewZ inverts the projection,
    // and feeding it the wrong near would skew every depth read.
    inline std::shared_ptr<ShaderMaterial> makeDepthLinearizeMaterial(float near, float far) {
        auto m = ShaderMaterial::create();
        m->vertexShader = R"(
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    )";
        m->fragmentShader = R"(
        #include <packing>
        varying vec2 vUv;
        uniform sampler2D tDepth;
        uniform float cameraNear;
        uniform float cameraFar;
        void main() {
            float fragCoordZ = texture2D(tDepth, vUv).x;
            float viewZ = perspectiveDepthToViewZ(fragCoordZ, cameraNear, cameraFar);
            float d = clamp(-viewZ / cameraFar, 0.0, 1.0);
            float r = floor(d * 255.0) / 255.0;
            float g = fract(d * 255.0);
            gl_FragColor = vec4(r, g, 0, 1.0);
        }
    )";
        m->uniforms = {
                {"tDepth", Uniform()},
                {"cameraNear", Uniform(near)},
                {"cameraFar", Uniform(far)}};
        return m;
    }

    // Read side of the RG contract: normalized depth in [0,1] from the two
    // packed bytes of one pixel.
    inline float decodeDepthRG(const unsigned char* px) {
        return static_cast<float>(px[0]) * (1.f / 255.f) + static_cast<float>(px[1]) * (1.f / 65025.f);
    }

}// namespace threepp::sensorscan

#endif//THREEPP_SENSOR_SCAN_UTIL_HPP
