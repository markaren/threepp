
#include "threepp/helpers/LidarSensor.hpp"

#include "sensor_scan_util.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/textures/DepthTexture.hpp"

// The path-traced back-end is only *used* on a Vulkan build (there is no raster
// depth cube there), but its header is included unconditionally so the cached
// unique_ptr member has a complete type to destroy. The header itself is
// Vulkan-include-free; only the renderer header below is gated.
#include "threepp/helpers/PathTracedLidarSensor.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <cmath>

using namespace threepp;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

    // Map a direction vector (sensor-local space) to the cube face that it
    // primarily hits, and compute the NDC coordinates (u, v) in [-1, 1] of
    // that direction within the face camera's image.
    //
    // Face camera orientations (matching CubeCamera / threepp conventions):
    //   0 +X: forward=(1,0,0),  right=(0,0,-1), up=(0,-1,0)
    //   1 -X: forward=(-1,0,0), right=(0,0, 1), up=(0,-1,0)
    //   2 +Y: forward=(0,1,0),  right=(1,0, 0), up=(0, 0, 1)
    //   3 -Y: forward=(0,-1,0), right=(1,0, 0), up=(0, 0,-1)
    //   4 +Z: forward=(0,0,1),  right=(1,0, 0), up=(0,-1, 0)
    //   5 -Z: forward=(0,0,-1), right=(-1,0,0), up=(0,-1, 0)
    //
    // u = dot(d, right)   / dot(d, forward)
    // v = dot(d, up)      / dot(d, forward)
    void dirToFaceUV(float dx, float dy, float dz, int& face, float& u, float& v) {
        const float ax = std::abs(dx), ay = std::abs(dy), az = std::abs(dz);
        float num_u, num_v, denom;

        if (ax >= ay && ax >= az) {
            denom = ax;
            if (dx > 0.f) {
                face = 0;
                num_u = -dz;
                num_v = -dy;
            } else {
                face = 1;
                num_u = dz;
                num_v = -dy;
            }
        } else if (ay >= ax && ay >= az) {
            denom = ay;
            if (dy > 0.f) {
                face = 2;
                num_u = dx;
                num_v = dz;
            } else {
                face = 3;
                num_u = dx;
                num_v = -dz;
            }
        } else {
            denom = az;
            if (dz > 0.f) {
                face = 4;
                num_u = dx;
                num_v = -dy;
            } else {
                face = 5;
                num_u = -dx;
                num_v = -dy;
            }
        }

        const float inv = 1.f / denom;
        u = num_u * inv;
        v = num_v * inv;
    }

    // ── The near/far shell, and why a raster near plane is not it ────────────
    //
    // For a ranging sensor, `near` and `far` bound the RANGE it can report: a
    // blind sphere of radius `near`, out to a maximum range of `far`. That is
    // what LidarReturn::distance is measured in, what a real scanner's datasheet
    // quotes ("a VLP-16 cannot report inside ~0.4 m"), and what the ray-traced
    // backend enforces directly as the ray's [tMin, tMax] interval.
    //
    // A raster near/far plane is a different quantity: it clips on view-space Z,
    // the PERPENDICULAR depth along the face camera's axis. The two differ by the
    // pixel's slant ratio
    //
    //     k = |view ray| / |view z| = sqrt(1 + u^2 + v^2),   u, v = face NDC
    //
    // which is 1 dead ahead and sqrt(3) at the corner of a 90-degree square cube
    // face. Using the near plane AS the blind zone therefore made the blind zone
    // direction-dependent — the same object was clipped or returned depending on
    // which part of which cube face its beam landed on — and it did NOT agree
    // with the traced backend, which bounds the true range. A drone's rotor at
    // 1.35 m from a near=1.2 m sensor was clipped by GL (view z 0.95) and
    // returned by Vulkan (range 1.35). That is the parity bug this shell fixes.
    //
    // So: pull the raster near plane in by the worst-case k, then reject on the
    // exact range per pixel. Nothing inside the shell can be clipped, because
    // anything the pulled-in plane still clips has viewZ < near/kMaxSlant, hence
    // range = k*viewZ < near. The far plane needs no widening, by the same
    // argument in reverse: range >= viewZ always, so viewZ > far implies the
    // point is outside the shell anyway.
    //
    // Both bounds are INCLUSIVE — a surface exactly at near or at far is
    // reported — which is what traceRayEXT's [tMin, tMax] interval gives on the
    // Vulkan side, so the two backends agree on the boundary too.
    constexpr float kMaxSlant = 1.7320508f;// sqrt(3), at the corner of a 90-degree face

    using sensorscan::DataPassGuard;

}// namespace

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

void LidarSensor::init(float near, float far) {

    struct FaceDesc {
        Vector3 lookAt, up;
    };
    static const std::array<FaceDesc, kNumFaces> kFaces{{
            {{1, 0, 0}, {0, -1, 0}}, // +X
            {{-1, 0, 0}, {0, -1, 0}},// -X
            {{0, 1, 0}, {0, 0, 1}},  // +Y
            {{0, -1, 0}, {0, 0, -1}},// -Y
            {{0, 0, 1}, {0, -1, 0}}, // +Z
            {{0, 0, -1}, {0, -1, 0}},// -Z
    }};

    // See kMaxSlant above: the raster plane is pulled in so it cannot clip
    // anything inside the range shell; the shell itself is applied on the exact
    // per-pixel range during unprojection.
    const float rasterNear = std::max(near / kMaxSlant, 1e-4f);

    for (int i = 0; i < kNumFaces; ++i) {
        auto cam = PerspectiveCamera::create(90.f, 1.f, rasterNear, far);
        cam->up.copy(kFaces[i].up);
        cam->lookAt(kFaces[i].lookAt);
        add(cam);
        cameras_[i] = cam.get();
    }

    GLRenderTarget::Options sceneOpts;
    sceneOpts.format = Format::RGB;
    sceneOpts.minFilter = Filter::Nearest;
    sceneOpts.magFilter = Filter::Nearest;
    sceneOpts.generateMipmaps = false;
    sceneOpts.stencilBuffer = false;
    sceneOpts.depthBuffer = true;

    GLRenderTarget::Options readOpts;
    readOpts.format = Format::RG;
    readOpts.minFilter = Filter::Nearest;
    readOpts.magFilter = Filter::Nearest;
    readOpts.generateMipmaps = false;
    readOpts.depthBuffer = false;
    readOpts.stencilBuffer = false;

    for (int i = 0; i < kNumFaces; ++i) {
        sceneOpts.depthTexture = DepthTexture::create(Type::Float);
        sceneTargets_[i] = GLRenderTarget::create(faceSize_, faceSize_, sceneOpts);
        readbackTargets_[i] = GLRenderTarget::create(faceSize_, faceSize_, readOpts);
    }

    // Post-process: linearize + RG-encode (shared shader; the near plane must
    // be the one the projection actually used, not the blind-sphere radius —
    // see makeDepthLinearizeMaterial).
    postMaterial_ = sensorscan::makeDepthLinearizeMaterial(rasterNear, far_);

    postScene_.add(Mesh::create(PlaneGeometry::create(2, 2), postMaterial_));
}

void LidarSensor::buildBeamTable(const LidarModel& model) {
    const int numAzSteps = static_cast<int>(
            std::round((model.azimuthMax - model.azimuthMin) / model.azimuthResolution));

    beams_.clear();
    beams_.reserve(numAzSteps * static_cast<int>(model.elevationAngles.size()));

    const int fs = static_cast<int>(faceSize_);

    for (int ai = 0; ai < numAzSteps; ++ai) {
        const float azimuth = (model.azimuthMin + ai * model.azimuthResolution) * math::DEG2RAD;

        for (float elevDeg : model.elevationAngles) {
            const float elevation = elevDeg * math::DEG2RAD;
            const float cosElev = std::cos(elevation);

            // azimuth=0 → forward (-Z), increases CCW from above
            const float dx = cosElev * std::sin(azimuth);
            const float dy = std::sin(elevation);
            const float dz = -cosElev * std::cos(azimuth);

            int face;
            float u, v;
            dirToFaceUV(dx, dy, dz, face, u, v);

            const int px = std::clamp(static_cast<int>((u + 1.f) * 0.5f * static_cast<float>(fs)), 0, fs - 1);
            const int py = std::clamp(static_cast<int>((v + 1.f) * 0.5f * static_cast<float>(fs)), 0, fs - 1);

            beams_.push_back({static_cast<uint8_t>(face),
                              static_cast<uint16_t>(px),
                              static_cast<uint16_t>(py),
                              u, v});
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

LidarSensor::LidarSensor(unsigned int faceSize, float near, float far)
    // The sensor frame is this node's own world frame, so it attaches to itself.
    // Default noise mirrors the pre-port constant: 2 cm sigma, fixed seed.
    : VisionSensor(*this, RangeNoiseModel{/*stddev*/ 0.02f, /*stddevPerMetre*/ 0.f,
                                          /*bias*/ 0.f, /*seed*/ 0x2545F4914F6CDD1DULL}),
      faceSize_(faceSize), near_(near), far_(far), postCamera_(-1, 1, 1, -1, 0, 1) {

    init(near, far);

    // Dense-grid mode: precompute NDC coords for every pixel (tanHalfFov=1 for 90°)
    const auto fs = static_cast<float>(faceSize_);
    dir_.resize(faceSize_);
    for (unsigned i = 0; i < faceSize_; ++i)
        dir_[i] = (static_cast<float>(i) + 0.5f) / fs * 2.f - 1.f;
}

LidarSensor::LidarSensor(const LidarModel& model, unsigned int faceSize, float near, float far)
    // The sensor frame is this node's own world frame, so it attaches to itself.
    // Default noise mirrors the pre-port constant: 2 cm sigma, fixed seed.
    : VisionSensor(*this, RangeNoiseModel{/*stddev*/ 0.02f, /*stddevPerMetre*/ 0.f,
                                          /*bias*/ 0.f, /*seed*/ 0x2545F4914F6CDD1DULL}),
      faceSize_(faceSize), near_(near), far_(far), postCamera_(-1, 1, 1, -1, 0, 1),
      model_(model) {

    init(near, far);
    buildBeamTable(model);
}

LidarSensor::~LidarSensor() = default;

void LidarSensor::resetNoise() {
    VisionSensor::resetNoise();
    // On Vulkan the back-end draws the noise, so resetting only our own stream
    // would silently leave that path unreplayable.
    if (tracedBackend_) tracedBackend_->resetNoise();
}

#ifdef THREEPP_WITH_VULKAN
PathTracedLidarSensor& LidarSensor::tracedBackend() {
    if (!tracedBackend_) {
        if (model_) {
            tracedBackend_ = std::make_unique<PathTracedLidarSensor>(*model_, far_);
        } else {
            // Dense-grid mode: the cube resolves ~90° per faceSize pixels, so
            // match that angular resolution on the tracer's az × el grid.
            tracedBackend_ = std::make_unique<PathTracedLidarSensor>(faceSize_ * 4, faceSize_ * 2, far_);
        }
    }
    // This sensor owns the noise contract; the back-end just applies it, so
    // its model tracks ours (including a seed the caller re-rolled). Copying
    // an unchanged seed does not restart the stream — see VisionSensor.
    tracedBackend_->rangeNoise = rangeNoise;
    // The range shell, handed to the tracer as its [tMin, tMax] interval. This
    // is the SAME bound the raster path applies per pixel after unprojecting
    // (see kMaxSlant) — near is a blind sphere on both backends, not a plane.
    // Without it every beam returns the sensor's own housing from the inside.
    tracedBackend_->params.minRange = near_;
    return *tracedBackend_;
}
#endif

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

void LidarSensor::scan(Renderer& renderer, Scene& scene, std::vector<LidarReturn>& cloud) {
    // Fire and take delivery in one call. Both halves run unconditionally: the
    // raster path has already filled `cloud`, and its collect is what clears
    // the "one delivery owed" flag so the pair stays balanced.
    const bool immediate = scanBegin(renderer, scene, cloud);
    const bool delivered = scanCollect(renderer, cloud);
    // Synchronous contract: this call's cloud, or nothing.
    if (!immediate && !delivered) cloud.clear();
}

bool LidarSensor::scanBegin(Renderer& renderer, Scene& scene, std::vector<LidarReturn>& cloud) {
    beginScan();
    scanPending_ = true;

#ifdef THREEPP_WITH_VULKAN
    if (auto* vk = dynamic_cast<VulkanRenderer*>(&renderer)) {
        // No raster depth cube to read back here: trace the same beam pattern
        // through the renderer's TLAS instead. The TLAS, not the `scene` graph,
        // is what gets traced, so the scene must have been render()-ed at least
        // once before scanning. Range noise is applied by the back-end.
        if (!parent) updateMatrixWorld();

        auto& lidar = tracedBackend();
        Vector3 pos, scl;
        Quaternion quat;
        matrixWorld->decompose(pos, quat, scl);
        lidar.position = pos;
        lidar.quaternion = quat;
        lidar.scale = scl;

        lidar.scanBegin(*vk);
        // Refused (too many scans already in flight): nothing is owed, so the
        // caller keeps whatever cloud it had rather than being handed an empty
        // one, and tries again next frame.
        scanPending_ = lidar.scanFired();
        // `cloud` is deliberately NOT cleared. It still holds the last delivered
        // scan, and a viewer reading it between fire and delivery must see that
        // rather than an empty one — clearing here made the overlay blink and
        // made "the cloud is not empty" a coin flip on the frame you asked.
        return false;
    }
#endif

    // Raster: the six face renders and their readbacks ARE the scan, and they
    // block anyway. Do it here and let scanCollect hand the same cloud over.
    cloud.clear();
    DataPassGuard guard(renderer);
    renderFaces(renderer, scene);

    if (beams_.empty())
        unprojectDense(cloud);
    else
        unprojectBeams(cloud);
    return true;
}

bool LidarSensor::scanReady(const Renderer& renderer) const {

    if (!scanPending_) return false;
#ifdef THREEPP_WITH_VULKAN
    if (const auto* vk = dynamic_cast<const VulkanRenderer*>(&renderer)) {
        return tracedBackend_ && tracedBackend_->scanReady(*vk);
    }
#else
    (void) renderer;
#endif
    return true;// raster: filled by scanBegin
}

bool LidarSensor::scanCollect(Renderer& renderer, std::vector<LidarReturn>& cloud) {

    if (!scanPending_) return false;

#ifdef THREEPP_WITH_VULKAN
    if (auto* vk = dynamic_cast<VulkanRenderer*>(&renderer)) {
        if (!tracedBackend_ || !tracedBackend_->scanCollect(*vk, cloud)) {
            scanPending_ = false;
            return false;
        }
        scanPending_ = false;

        // The raster path emits only real surface hits inside the range shell.
        // Drop the tracer's sentinel returns (hitInstanceId < 0: -1 = miss,
        // -2 = fog/volume scatter — a raster scan never sees atmosphere) to
        // match. The shell itself is already enforced on this path (tMin/tMax
        // on the true range, then again after noise in applyNoise), so the
        // range test here is belt-and-braces on the same inclusive bound.
        std::erase_if(cloud, [this](const LidarReturn& r) {
            return r.hitInstanceId < 0 || r.distance < near_ || r.distance > far_;
        });
        return true;
    }
#else
    (void) renderer;
    (void) cloud;
#endif

    scanPending_ = false;
    return true;// raster: scanBegin already filled it
}

void LidarSensor::renderFaces(Renderer& renderer, Scene& scene) {
    if (!parent) updateMatrixWorld();

    // All twelve passes (six scene renders + six depth-linearize posts) are
    // submitted before the first readback. copyTextureToImage is a GPU->CPU sync
    // point that drains the pipeline, so reading each face right after drawing it
    // stalled the GPU six times per scan and left it idle while the CPU waited.
    // Batched, the driver has the whole cube to work through and the scan pays one
    // stall instead of six. (The readback reads the texture object, not the bound
    // framebuffer, but the target is still set per face so the call sees exactly
    // the state it did before.)
    for (int f = 0; f < kNumFaces; ++f) {
        renderer.setRenderTarget(sceneTargets_[f].get());
        renderer.render(scene, *cameras_[f]);

        postMaterial_->uniforms.at("tDepth").setValue(sceneTargets_[f]->depthTexture.get());
        renderer.setRenderTarget(readbackTargets_[f].get());
        renderer.render(postScene_, postCamera_);
    }

    for (int f = 0; f < kNumFaces; ++f) {
        renderer.setRenderTarget(readbackTargets_[f].get());
        renderer.copyTextureToImage(*readbackTargets_[f]->texture);
    }

    renderer.setRenderTarget(nullptr);
}

// ---------------------------------------------------------------------------
// Unprojection
// ---------------------------------------------------------------------------

void LidarSensor::unprojectDense(std::vector<LidarReturn>& points) {

    const bool addNoise = rangeNoise.active();

    for (int face = 0; face < kNumFaces; ++face) {
        points.reserve(points.size() + faceSize_ * faceSize_);

        const auto& pixels = readbackTargets_[face]->texture->image().data();
        const auto* px = pixels.data();

        const auto& me = cameras_[face]->matrixWorld->elements;
        const float m0 = me[0], m1 = me[1], m2 = me[2];
        const float m4 = me[4], m5 = me[5], m6 = me[6];
        const float m8 = me[8], m9 = me[9], m10 = me[10];
        const float m12 = me[12], m13 = me[13], m14 = me[14];

        for (unsigned y = 0; y < faceSize_; ++y) {
            const float yd = dir_[y];
            const float ry0 = m4 * yd, ry1 = m5 * yd, ry2 = m6 * yd;
            const float slantY = yd * yd + 1.f;

            for (unsigned x = 0; x < faceSize_; ++x, px += 2) {
                const float nd = sensorscan::decodeDepthRG(px);
                if (nd >= 0.9999f) continue;

                const float xd = dir_[x];
                // Slant range (Euclidean): depth is along the cube-face camera's
                // view axis; off-axis beams travel further per unit of view-z.
                const float k = std::sqrt(xd * xd + slantY);
                float depth = nd * far_;
                float slant = depth * k;

                // The range shell, applied to the true geometry — the same
                // interval the traced backend passes as tMin/tMax. Inclusive at
                // both ends (see kMaxSlant).
                if (slant < near_ || slant > far_) continue;

                if (addNoise) {
                    // Noise corrupts the RANGE, which is what the model is
                    // specified in (and what the traced backend perturbs); the
                    // view-space depth follows from it.
                    slant = applyRangeNoise(slant);
                    if (slant < near_ || slant > far_) continue;
                    depth = slant / k;
                }

                LidarReturn r;
                r.position.set(
                        (m0 * xd + ry0 - m8) * depth + m12,
                        (m1 * xd + ry1 - m9) * depth + m13,
                        (m2 * xd + ry2 - m10) * depth + m14);
                r.normal.set(0.f, 0.f, 0.f);
                r.distance      = slant;
                r.intensity     = 0.f;
                r.hitInstanceId = -1;
                r.returnNo      = 1;
                points.push_back(std::move(r));
            }
        }
    }
}

void LidarSensor::unprojectBeams(std::vector<LidarReturn>& points) {

    points.reserve(beams_.size());

    // Cache pixel data and matrix element pointers for all faces
    std::array<const unsigned char*, kNumFaces> facePixels{};
    std::array<const float*, kNumFaces> faceMat{};
    for (int f = 0; f < kNumFaces; ++f) {
        facePixels[f] = readbackTargets_[f]->texture->image().data().data();
        faceMat[f] = cameras_[f]->matrixWorld->elements.data();
    }

    const bool addNoise = rangeNoise.active();

    for (const auto& b : beams_) {
        const unsigned char* px = facePixels[b.face] + (static_cast<unsigned>(b.pixelY) * faceSize_ + b.pixelX) * 2;
        const float nd = sensorscan::decodeDepthRG(px);

        if (nd >= 0.9999f) continue;

        const float k = std::sqrt(b.u * b.u + b.v * b.v + 1.f);
        float depth = nd * far_;
        float slant = depth * k;

        // The range shell, applied to the true geometry — the same interval the
        // traced backend passes as tMin/tMax. Inclusive at both ends.
        if (slant < near_ || slant > far_) continue;

        if (addNoise) {
            slant = applyRangeNoise(slant);
            if (slant < near_ || slant > far_) continue;
            depth = slant / k;
        }

        // view-space point for this beam: (u*depth, v*depth, -depth)
        // transformed to world space via the face camera's world matrix
        const float* me = faceMat[b.face];

        LidarReturn r;
        r.position.set(
                (me[0] * b.u + me[4] * b.v - me[8]) * depth + me[12],
                (me[1] * b.u + me[5] * b.v - me[9]) * depth + me[13],
                (me[2] * b.u + me[6] * b.v - me[10]) * depth + me[14]);
        r.normal.set(0.f, 0.f, 0.f);
        r.distance      = slant;
        r.intensity     = 0.f;
        r.hitInstanceId = -1;
        r.returnNo      = 1;
        points.push_back(std::move(r));
    }
}
