
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
    : postCamera_(-1, 1, 1, -1, 0, 1),
      width_(width),
      height_(height),
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
    const float far = camera_.farPlane;
    const float tanHalfFovY = std::tan(math::degToRad(camera_.fov) * 0.5f);
    const float tanHalfFovX = tanHalfFovY * camera_.aspect;
    const auto fw = static_cast<float>(width_);
    const auto fh = static_cast<float>(height_);

    static std::mt19937 rng_{std::random_device{}()};

    points.clear();

    const auto& pixels = readbackTarget_->texture->image().data();

    for (unsigned y = 0; y < height(); ++y) {
        for (unsigned x = 0; x < width(); ++x) {
            const int idx = static_cast<int>((y * width() + x) * 2);

            // Unpack 16-bit normalised depth from RG channels
            const float r = static_cast<float>(pixels[idx + 0]) / 255.f;
            const float g = static_cast<float>(pixels[idx + 1]) / 255.f;
            const float normalizedDepth = r + g / 255.f;

            // Skip background (depth at or beyond the far plane)
            if (normalizedDepth >= 0.9999f) continue;

            float viewZ = -normalizedDepth * far;
            if (rangeNoise > 0.f) {
                viewZ += std::normal_distribution{0.f, rangeNoise}(rng_);
                if (viewZ >= 0.f || -viewZ > far) continue;
            }

            // NDC: OpenGL convention, y=0 in pixel buffer = bottom = ndcY=-1
            const float ndcX = (static_cast<float>(x) + 0.5f) / fw * 2.0f - 1.0f;
            const float ndcY = (static_cast<float>(y) + 0.5f) / fh * 2.0f - 1.0f;

            Vector3 p(ndcX * -viewZ * tanHalfFovX,
                      ndcY * -viewZ * tanHalfFovY,
                      viewZ);

            p.applyMatrix4(*matrixWorld);
            points.push_back(p);
        }
    }
}