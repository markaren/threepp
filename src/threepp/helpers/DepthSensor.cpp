
#include "threepp/helpers/DepthSensor.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/textures/DepthTexture.hpp"
#include "threepp/utils/ImageUtils.hpp"

// The path-traced back-end is only *used* on a Vulkan build (there is no raster
// depth pass there), but its header is included unconditionally so the cached
// unique_ptr member has a complete type to destroy. The header itself is
// Vulkan-include-free; only the renderer header below is gated.
#include "threepp/helpers/PathTracedLidarSensor.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <cmath>

using namespace threepp;

namespace {

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

#ifdef THREEPP_WITH_VULKAN
    // Vulkan backend: there is no raster depth buffer to read back, so reproduce
    // the same pinhole ray pattern with a PathTracedLidarSensor (camera mode —
    // identical fovY / width / height and the look-down-local-(-Z) convention)
    // and trace it through the renderer's path-tracing TLAS. The TLAS, not the
    // `scene` graph, is what gets traced, so the scene must have been render()-ed
    // at least once before scanning (the public scan() doc spells this out).
    //
    // `lidar` is the DepthSensor's cached back-end, not a fresh one per scan:
    // its RNG stream has to continue across scans, or every frame would be
    // perturbed by the identical noise pattern (a seeded sensor rebuilt each
    // frame replays its seed each frame — noise frozen into the geometry).
    void scanViaPathTracer(PathTracedLidarSensor& lidar, const Matrix4& world,
                           VulkanRenderer& vk,
                           std::vector<Vector3>& cloud, std::vector<Color>* colors) {
        Vector3 pos, scl;
        Quaternion quat;
        world.decompose(pos, quat, scl);
        lidar.position = pos;
        lidar.quaternion = quat;
        lidar.scale = scl;

        std::vector<LidarReturn> returns;
        lidar.scan(vk, returns);// range noise applied by the back-end

        cloud.clear();
        cloud.reserve(returns.size());
        if (colors) {
            colors->clear();
            colors->reserve(returns.size());
        }

        for (const auto& ret : returns) {
            // A depth camera measures geometry, not atmosphere. The path-traced
            // LIDAR back-end emits sentinel returns the per-LidarTypes.hpp contract
            // says callers must drop (hitInstanceId < 0): -1 = miss, -2 = volume
            // scatter (fog / haze / participating media). Without this filter, any
            // scene with fog set fills the cloud with mid-air fog-scatter points —
            // the raster (GL) DepthSensor never sees fog, so this also restores
            // GL/Vulkan parity.
            if (ret.hitInstanceId < 0) continue;
            cloud.push_back(ret.position);
            // The path tracer returns LIDAR intensity, not surface colour; expose
            // it as greyscale so scan_rgbd has a uniform (cloud, colors) shape.
            if (colors) colors->emplace_back(ret.intensity, ret.intensity, ret.intensity);
        }
    }
#endif

}// namespace

DepthSensor::DepthSensor(float fovY, unsigned int width, unsigned int height, float near, float far)
    // The sensor frame is this node's own world frame, so it attaches to itself.
    // Default noise mirrors the pre-port constant: 2 cm sigma, fixed seed.
    : VisionSensor(*this, RangeNoiseModel{/*stddev*/ 0.02f, /*stddevPerMetre*/ 0.f,
                                          /*bias*/ 0.f, /*seed*/ 0x94D049BB133111EBULL}),
      width_(width),
      height_(height),
      postCamera_(-1, 1, 1, -1, 0, 1),
      camera_(fovY, static_cast<float>(width) / static_cast<float>(height), near, far) {

    // Scene render target: renders geometry and captures depth
    RenderTarget::Options sceneOpts;
    sceneOpts.format = Format::RGB;
    sceneOpts.minFilter = Filter::Nearest;
    sceneOpts.magFilter = Filter::Nearest;
    sceneOpts.generateMipmaps = false;
    sceneOpts.stencilBuffer = false;
    sceneOpts.depthBuffer = true;
    sceneOpts.depthTexture = DepthTexture::create(Type::Float);
    sceneTarget_ = RenderTarget::create(width_, height_, sceneOpts);

    // Readback target: packed linear depth in RGBA8
    RenderTarget::Options readOpts;
    readOpts.format = Format::RG;
    readOpts.minFilter = Filter::Nearest;
    readOpts.magFilter = Filter::Nearest;
    readOpts.generateMipmaps = false;
    readOpts.depthBuffer = false;
    readOpts.stencilBuffer = false;
    readbackTarget_ = RenderTarget::create(width_, height_, readOpts);

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

    DepthSensor::addRef(camera_);

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

DepthSensor::~DepthSensor() = default;

void DepthSensor::resetNoise() {
    VisionSensor::resetNoise();
    // On Vulkan the back-end draws the noise, so resetting only our own stream
    // would silently leave that path unreplayable.
    if (tracedBackend_) tracedBackend_->resetNoise();
}

#ifdef THREEPP_WITH_VULKAN
PathTracedLidarSensor& DepthSensor::tracedBackend() {
    if (!tracedBackend_) {
        tracedBackend_ = std::make_unique<PathTracedLidarSensor>(
                camera_.fov, width_, height_, camera_.farPlane);
    }
    // The depth sensor owns the noise contract; the back-end just applies it,
    // so its model tracks ours (including a seed the caller re-rolled). Copying
    // an unchanged seed does not restart the stream — see VisionSensor.
    tracedBackend_->rangeNoise = rangeNoise;
    // The near plane doubles as the blind zone: without it every beam
    // returns the sensor's own housing mesh from the inside.
    tracedBackend_->params.minRange = camera_.nearPlane;
    return *tracedBackend_;
}
#endif

void DepthSensor::scan(Renderer& renderer, Scene& scene, std::vector<Vector3>& cloud) {

    beginScan();

#ifdef THREEPP_WITH_VULKAN
    if (auto* vk = dynamic_cast<VulkanRenderer*>(&renderer)) {
        scanViaPathTracer(tracedBackend(), *matrixWorld, *vk, cloud, nullptr);
        return;
    }
#endif

    DataPassGuard guard(renderer);

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

void DepthSensor::scan(Renderer& renderer, Scene& scene, std::vector<Vector3>& cloud, std::vector<Color>& colors) {

    beginScan();

#ifdef THREEPP_WITH_VULKAN
    if (auto* vk = dynamic_cast<VulkanRenderer*>(&renderer)) {
        scanViaPathTracer(tracedBackend(), *matrixWorld, *vk, cloud, &colors);
        return;
    }
#endif

    DataPassGuard guard(renderer);

    // Render scene from sensor viewpoint — color buffer is captured alongside depth
    renderer.setRenderTarget(sceneTarget_.get());
    renderer.render(scene, camera_);

    // Read back color from the scene color buffer
    renderer.copyTextureToImage(*sceneTarget_->texture);
    // The depth path goes through a post-pass that, on a flip-Y backend, already
    // compensates for the render-target origin. The color is read back directly,
    // so flip it to share the depth's row convention — keeping color and geometry
    // aligned identically on every backend. Inert on GL (bottom-left origin).
    if (renderer.renderTargetFlipY()) {
        flipImage(sceneTarget_->texture->image());
    }

    // Linearize depth into packed RG16
    postMaterial_->uniforms.at("tDepth").setValue(sceneTarget_->depthTexture.get());
    renderer.setRenderTarget(readbackTarget_.get());
    renderer.render(postScene_, postCamera_);

    renderer.copyTextureToImage(*readbackTarget_->texture);

    // Restore default render target
    renderer.setRenderTarget(nullptr);

    colors.clear();
    unprojectPoints(cloud, sceneTarget_->texture->image().data().data(), &colors);
}

void DepthSensor::unprojectPoints(std::vector<Vector3>& points,
                                   const unsigned char* colorPixels,
                                   std::vector<Color>* colors) {

    points.clear();
    points.reserve(width_ * height_);
    if (colors) colors->reserve(width_ * height_);

    const auto& pixels = readbackTarget_->texture->image().data();
    const auto* px = pixels.data();

    // Hoist matrix elements (column-major): transform view point (vx,vy,-depth,1)
    const auto& me = matrixWorld->elements;
    const float m0 = me[0], m1 = me[1], m2 = me[2];
    const float m4 = me[4], m5 = me[5], m6 = me[6];
    const float m8 = me[8], m9 = me[9], m10 = me[10];
    const float m12 = me[12], m13 = me[13], m14 = me[14];

    const float far = camera_.farPlane;
    const bool addNoise = rangeNoise.active();

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
                depth = applyRangeNoise(depth);
                if (depth <= 0.f || depth > far) continue;
            }

            // Inline world-space transform: view point = (xDir*depth, yDir*depth, -depth)
            const float xd = xDir_[x];
            points.emplace_back(
                    (m0 * xd + ry0 - m8) * depth + m12,
                    (m1 * xd + ry1 - m9) * depth + m13,
                    (m2 * xd + ry2 - m10) * depth + m14);

            if (colorPixels && colors) {
                const unsigned char* cp = colorPixels + (y * width_ + x) * 3;
                colors->emplace_back(cp[0] / 255.f, cp[1] / 255.f, cp[2] / 255.f);
            }
        }
    }
}
