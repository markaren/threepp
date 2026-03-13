
#include "threepp/helpers/LidarSensor.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/textures/DepthTexture.hpp"

#include <cmath>
#include <optional>
#include <random>

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
            if (dx > 0.f) { face = 0; num_u = -dz; num_v = -dy; }
            else           { face = 1; num_u =  dz; num_v = -dy; }
        } else if (ay >= ax && ay >= az) {
            denom = ay;
            if (dy > 0.f) { face = 2; num_u = dx;  num_v =  dz; }
            else           { face = 3; num_u = dx;  num_v = -dz; }
        } else {
            denom = az;
            if (dz > 0.f) { face = 4; num_u =  dx; num_v = -dy; }
            else           { face = 5; num_u = -dx; num_v = -dy; }
        }

        const float inv = 1.f / denom;
        u = num_u * inv;
        v = num_v * inv;
    }

}// namespace

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

void LidarSensor::init(float near, float far) {

    struct FaceDesc { Vector3 lookAt, up; };
    static const std::array<FaceDesc, kNumFaces> kFaces{{
        {{1,  0,  0}, {0, -1,  0}},  // +X
        {{-1, 0,  0}, {0, -1,  0}},  // -X
        {{0,  1,  0}, {0,  0,  1}},  // +Y
        {{0, -1,  0}, {0,  0, -1}},  // -Y
        {{0,  0,  1}, {0, -1,  0}},  // +Z
        {{0,  0, -1}, {0, -1,  0}},  // -Z
    }};

    for (int i = 0; i < kNumFaces; ++i) {
        auto cam = PerspectiveCamera::create(90.f, 1.f, near, far);
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

    // Post-process shader: linearize perspective depth, encode in RG for ~16-bit precision.
    postMaterial_ = ShaderMaterial::create();
    postMaterial_->vertexShader = R"(
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    )";
    postMaterial_->fragmentShader = R"(
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
    postMaterial_->uniforms = {
        {"tDepth", Uniform()},
        {"cameraNear", Uniform(near_)},
        {"cameraFar", Uniform(far_)}};

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
    : faceSize_(faceSize), near_(near), far_(far), postCamera_(-1, 1, 1, -1, 0, 1) {

    init(near, far);

    // Dense-grid mode: precompute NDC coords for every pixel (tanHalfFov=1 for 90°)
    const auto fs = static_cast<float>(faceSize_);
    dir_.resize(faceSize_);
    for (unsigned i = 0; i < faceSize_; ++i)
        dir_[i] = (static_cast<float>(i) + 0.5f) / fs * 2.f - 1.f;
}

LidarSensor::LidarSensor(const LidarModel& model, unsigned int faceSize, float near, float far)
    : faceSize_(faceSize), near_(near), far_(far), postCamera_(-1, 1, 1, -1, 0, 1) {

    init(near, far);
    buildBeamTable(model);
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

void LidarSensor::scan(GLRenderer& renderer, Scene& scene, std::vector<Vector3>& cloud) {
    cloud.clear();
    renderFaces(renderer, scene);

    if (beams_.empty())
        unprojectDense(cloud);
    else
        unprojectBeams(cloud);
}

void LidarSensor::renderFaces(GLRenderer& renderer, Scene& scene) {
    if (!parent) updateMatrixWorld();

    for (int f = 0; f < kNumFaces; ++f) {
        renderer.setRenderTarget(sceneTargets_[f].get());
        renderer.render(scene, *cameras_[f]);

        postMaterial_->uniforms.at("tDepth").setValue(sceneTargets_[f]->depthTexture.get());
        renderer.setRenderTarget(readbackTargets_[f].get());
        renderer.render(postScene_, postCamera_);

        renderer.copyTextureToImage(*readbackTargets_[f]->texture);
    }

    renderer.setRenderTarget(nullptr);
}

// ---------------------------------------------------------------------------
// Unprojection
// ---------------------------------------------------------------------------

void LidarSensor::unprojectDense(std::vector<Vector3>& points) const {
    static std::mt19937 rng_{std::random_device{}()};

    const bool addNoise = rangeNoise > 0.f;
    std::optional<std::normal_distribution<float>> noiseDist;
    if (addNoise) noiseDist = std::normal_distribution{0.f, rangeNoise};

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

            for (unsigned x = 0; x < faceSize_; ++x, px += 2) {
                const float nd = static_cast<float>(px[0]) * (1.f / 255.f) + static_cast<float>(px[1]) * (1.f / 65025.f);
                if (nd >= 0.9999f) continue;

                float depth = nd * far_;
                if (addNoise) {
                    depth += (*noiseDist)(rng_);
                    if (depth <= 0.f || depth > far_) continue;
                }

                const float xd = dir_[x];
                points.emplace_back(
                    (m0 * xd + ry0 - m8) * depth + m12,
                    (m1 * xd + ry1 - m9) * depth + m13,
                    (m2 * xd + ry2 - m10) * depth + m14);
            }
        }
    }
}

void LidarSensor::unprojectBeams(std::vector<Vector3>& points) const {
    static std::mt19937 rng_{std::random_device{}()};

    points.reserve(beams_.size());

    // Cache pixel data and matrix element pointers for all faces
    std::array<const unsigned char*, kNumFaces> facePixels{};
    std::array<const float*, kNumFaces> faceMat{};
    for (int f = 0; f < kNumFaces; ++f) {
        facePixels[f] = readbackTargets_[f]->texture->image().data().data();
        faceMat[f] = cameras_[f]->matrixWorld->elements.data();
    }

    const bool addNoise = rangeNoise > 0.f;
    std::optional<std::normal_distribution<float>> noiseDist;
    if (addNoise) noiseDist = std::normal_distribution{0.f, rangeNoise};

    for (const auto& b : beams_) {
        const unsigned char* px = facePixels[b.face] + (static_cast<unsigned>(b.pixelY) * faceSize_ + b.pixelX) * 2;
        const float nd = static_cast<float>(px[0]) * (1.f / 255.f) + static_cast<float>(px[1]) * (1.f / 65025.f);

        if (nd >= 0.9999f) continue;

        float depth = nd * far_;
        if (addNoise) {
            depth += (*noiseDist)(rng_);
            if (depth <= 0.f || depth > far_) continue;
        }

        // view-space point for this beam: (u*depth, v*depth, -depth)
        // transformed to world space via the face camera's world matrix
        const float* me = faceMat[b.face];
        points.emplace_back(
            (me[0] * b.u + me[4] * b.v - me[8])  * depth + me[12],
            (me[1] * b.u + me[5] * b.v - me[9])  * depth + me[13],
            (me[2] * b.u + me[6] * b.v - me[10]) * depth + me[14]);
    }
}
