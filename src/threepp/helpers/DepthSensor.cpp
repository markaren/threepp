
#include "threepp/helpers/DepthSensor.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/textures/DepthTexture.hpp"

#include <cmath>
#include <random>

using namespace threepp;

DepthSensor::DepthSensor(float fovY, unsigned int width, unsigned int height, float near, float far)
    : width_(width),
      height_(height),
      postCamera_(-1, 1, 1, -1, 0, 1),
      camera_(fovY, static_cast<float>(width) / static_cast<float>(height), near, far) {

    // Scene render target: renders geometry and captures depth
    GLRenderTarget::Options sceneOpts;
    sceneOpts.format = Format::RGB;
    sceneOpts.minFilter = Filter::Nearest;
    sceneOpts.magFilter = Filter::Nearest;
    sceneOpts.generateMipmaps = false;
    sceneOpts.stencilBuffer = false;
    sceneOpts.depthBuffer = true;
    sceneOpts.depthTexture = DepthTexture::create(Type::Float);
    sceneTarget_ = GLRenderTarget::create(width_, height_, sceneOpts);

    // Readback target: packed linear depth in RGBA8
    GLRenderTarget::Options readOpts;
    readOpts.format = Format::RG;
    readOpts.minFilter = Filter::Nearest;
    readOpts.magFilter = Filter::Nearest;
    readOpts.generateMipmaps = false;
    readOpts.depthBuffer = false;
    readOpts.stencilBuffer = false;
    readbackTarget_ = GLRenderTarget::create(width_, height_, readOpts);

    // Post-process: linearize perspective depth, encode in RG channels
    // R = floor(d * 255) / 255  (high byte)
    // G = fract(d * 255)        (low byte / 255 on CPU readback)
    // This gives ~16-bit precision over the [0, far] range.
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
            {"cameraNear", Uniform(camera_.nearPlane)},
            {"cameraFar", Uniform(camera_.farPlane)}};

    postScene_.add(Mesh::create(PlaneGeometry::create(2, 2), postMaterial_));

    DepthSensor::add(camera_);

    // Precompute per-column/row view-space ray direction factors.
    // view point = (xDir_[x] * depth, yDir_[y] * depth, -depth)
    const float tanHalfFovY = std::tan(math::degToRad(camera_.fov) * 0.5f);
    const float tanHalfFovX = tanHalfFovY * camera_.aspect;
    const auto fw = static_cast<float>(width_);
    const auto fh = static_cast<float>(height_);

    xDir_.resize(width_);
    for (unsigned x = 0; x < width_; ++x)
        xDir_[x] = ((static_cast<float>(x) + 0.5f) / fw * 2.f - 1.f) * tanHalfFovX;

    yDir_.resize(height_);
    for (unsigned y = 0; y < height_; ++y)
        yDir_[y] = ((static_cast<float>(y) + 0.5f) / fh * 2.f - 1.f) * tanHalfFovY;
}

void DepthSensor::scan(GLRenderer& renderer, Scene& scene, std::vector<Vector3>& cloud) {

    // Render scene from sensor viewpoint to capture depth
    renderer.setRenderTarget(sceneTarget_.get());
    renderer.render(scene, camera_);

    // Linearize depth into packed RGBA8
    postMaterial_->uniforms.at("tDepth").setValue(sceneTarget_->depthTexture.get());
    renderer.setRenderTarget(readbackTarget_.get());
    renderer.render(postScene_, postCamera_);

    renderer.copyTextureToImage(*readbackTarget_->texture);

    // Restore default render target
    renderer.setRenderTarget(nullptr);

    unprojectPoints(cloud);
}

void DepthSensor::unprojectPoints(std::vector<Vector3>& points) const {
    static std::mt19937 rng_{std::random_device{}()};

    points.clear();
    points.reserve(width_ * height_);

    const auto& pixels = readbackTarget_->texture->image().data();
    const auto* px = pixels.data();

    // Hoist matrix elements (column-major): transform view point (vx,vy,-depth,1)
    const auto& me = matrixWorld->elements;
    const float m0 = me[0], m1 = me[1], m2 = me[2];
    const float m4 = me[4], m5 = me[5], m6 = me[6];
    const float m8 = me[8], m9 = me[9], m10 = me[10];
    const float m12 = me[12], m13 = me[13], m14 = me[14];

    const float far = camera_.farPlane;
    const bool addNoise = rangeNoise > 0.f;
    std::normal_distribution noiseDist{0.f, rangeNoise};

    for (unsigned y = 0; y < height_; ++y) {
        // Hoist row-constant contributions from yDir_[y]
        const float yd = yDir_[y];
        const float ry0 = m4 * yd, ry1 = m5 * yd, ry2 = m6 * yd;

        for (unsigned x = 0; x < width_; ++x, px += 2) {
            // Unpack 16-bit normalised depth from RG channels
            const float normalizedDepth = static_cast<float>(px[0]) * (1.f / 255.f) + static_cast<float>(px[1]) * (1.f / 65025.f);

            if (normalizedDepth >= 0.9999f) continue;

            float depth = normalizedDepth * far;// positive distance along view axis
            if (addNoise) {
                depth += noiseDist(rng_);
                if (depth <= 0.f || depth > far) continue;
            }

            // Inline world-space transform: view point = (xDir*depth, yDir*depth, -depth)
            const float xd = xDir_[x];
            points.emplace_back(
                    (m0 * xd + ry0 - m8) * depth + m12,
                    (m1 * xd + ry1 - m9) * depth + m13,
                    (m2 * xd + ry2 - m10) * depth + m14);
        }
    }
}