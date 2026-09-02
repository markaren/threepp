
#include "threepp/objects/SplatCloud.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/core/Uniform.hpp"
#include "threepp/extras/DataUtils.hpp"
#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

using namespace threepp;

namespace {

    // One row of every splat texture is this wide; index -> (u, v) is integer
    // division.
    //
    // Both constants are the 16384 every desktop GL implementation of the last
    // decade reports, rather than a queried GL_MAX_TEXTURE_SIZE, and the reason
    // is ordering rather than laziness: the textures are built in the
    // constructor, which routinely runs before any GL context exists — a loader
    // thread, a headless test, or simply building the cloud before the Canvas.
    // There is nothing to ask at that point, and deferring the whole build to
    // first render would trade a real constraint for a worse one. The GL 3.3
    // spec floor is far lower, but nothing that reports it could hold a cloud
    // this size in any layout.
    //
    // 8192 x 16384 is 134M texels: 8.4M splats at SH degree 3 (16 texels each,
    // see buildTextures), 67M at degree 1, 134M at degree 0.
    constexpr int TEX_WIDTH = 8192;
    constexpr int MAX_TEX_HEIGHT = 16384;

    // Sort keys are 16-bit, so the counting sort is a fixed 65536-bucket pass.
    constexpr int SORT_BUCKETS = 65536;

    // The key range is a robust interval of the view depths, not their full
    // extent. A scan's stray splats live a thousand units outside a
    // twenty-unit subject: spreading 65536 buckets over THAT makes one bucket
    // 0.04 units wide against 0.017-unit content, so most of the cloud lands
    // in a handful of buckets and the sort quietly degrades to file order.
    //
    // Splats outside the interval do NOT collapse into a single end bucket.
    // Collapsing loses their ordering among themselves, and the stable sort
    // then composites them in file order — which repainted the Sanctuaire
    // scan's sky, a shell of huge overlapping translucent splats beyond p99,
    // in whatever pastels file order happened to blend. Each side instead
    // keeps a small band of buckets of its own, spread over [min, p1) and
    // (p99, max]: coarse, but monotone, and the sky stays the colour it was
    // scanned in.
    constexpr float SORT_CLAMP_LO = 0.01f;// p1 of the sampled view depths
    constexpr float SORT_CLAMP_HI = 0.99f;// p99

    // Buckets reserved for each tail. 2048 leaves the content interval 61440
    // of the 65536 — a 3% resolution tax — while a tail spanning a couple of
    // thousand units still resolves splats a unit apart.
    constexpr int SORT_TAIL_BUCKETS = 2048;

    // A little air on each end, so the p1/p99 splats themselves are not
    // sitting in the clamped end buckets.
    constexpr float SORT_CLAMP_MARGIN = 0.02f;

    // Exact percentiles over five million depths, every frame, are not free.
    // A fixed stride is: no RNG, no state, the same sample for the same cloud
    // every time, and 8192 depths estimate a 1st percentile far more tightly
    // than the sort needs.
    constexpr size_t SORT_SAMPLE_TARGET = 8192;

    int texHeightFor(size_t texels, const char* what, size_t splats) {

        const size_t rows = (texels + TEX_WIDTH - 1) / TEX_WIDTH;
        if (rows > static_cast<size_t>(MAX_TEX_HEIGHT)) {

            throw std::length_error(
                    "SplatCloud: " + std::to_string(splats) + " splats need a " + what +
                    " texture " + std::to_string(rows) + " rows tall, past the " +
                    std::to_string(MAX_TEX_HEIGHT) + "-row limit");
        }
        return static_cast<int>(rows);
    }

    // The unit quad, instanced `count` times, carrying the one per-instance
    // attribute the splat shader declares.
    //
    // splatIndex is allocated HERE, in the geometry the constructor hands to the
    // base class, rather than later with the rest of the GL-side memory in
    // ensureGlResources. GLRenderer decides what to upload for an object in
    // projectObject, while it is building the render list, which is a phase
    // EARLIER than the onBeforeRender that triggers the lazy build: an attribute
    // that does not exist yet is skipped there, and then setupVertexAttributes
    // binds the attribute the splat shader declares and looks up a GL buffer
    // nobody created ("invalid unordered_map key"). It crashed every splat import
    // in the editor, which draws through onBeforeRender, while the tests that
    // call update(camera) first — building the resources before the render list
    // is built — never saw it.
    //
    // The rule this encodes: an object may create UNIFORMS during onBeforeRender,
    // but never an ATTRIBUTE. The vertex layout has to be settled before the
    // renderer walks the scene. Cheap to honour — 4 bytes a splat against the 176
    // the data textures cost, which stay lazy.
    std::shared_ptr<InstancedBufferGeometry> splatQuad(size_t count) {

        // xy in [-1, 1]; the vertex shader scales it to the splat's 3-sigma
        // footprint in pixels. z is unused.
        auto geometry = InstancedBufferGeometry::create(count);
        geometry->setAttribute("position", FloatBufferAttribute::create(
                                                   {-1.f, -1.f, 0.f,
                                                    1.f, -1.f, 0.f,
                                                    1.f, 1.f, 0.f,
                                                    -1.f, 1.f, 0.f},
                                                   3));
        geometry->setIndex(std::vector<unsigned int>{0, 1, 2, 0, 2, 3});

        // File order until the first update() sorts. An EMPTY cloud allocates a
        // slot too: the shader binds splatIndex whether or not any instance
        // draws, and instanceCount drops to 0 afterwards so nothing is drawn.
        std::vector<float> order(std::max<size_t>(count, 1));
        for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<float>(i);
        geometry->setAttribute("splatIndex", InstancedBufferAttribute::create(std::move(order), 1));

        return geometry;
    }

}// namespace


// ---------------------------------------------------------------------------
// Shaders
//
// The SH constants below are written with the SAME decimal literals as
// threepp/splats/SplatSH.hpp. SplatSH_test asserts this text still contains
// every one of them, so the C++ evaluator and the GPU one cannot drift.
// ---------------------------------------------------------------------------

const std::string& SplatCloud::vertexShaderSource() {

    static const std::string source = R"(#version 330 core

// Standard threepp uniforms (GLRenderer sets these for a RawShaderMaterial too).
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 cameraPosition;

// Per-splat data, fetched by index. See SplatCloud.hpp for the packing.
uniform sampler2D splatMeanTex;// (mean.xyz, opacity)
uniform sampler2D splatCovTex; // (Sxx,Sxy,Sxz,Syy), (Syz,Szz,-,-)
uniform sampler2D splatShTex;  // one texel per SH coefficient, rgb used

uniform int splatTexWidth;
uniform int splatShCoeffs;// (degree+1)^2
uniform int splatShDegree;
uniform vec2 splatViewport;// framebuffer size in pixels
uniform float splatNear;   // camera near plane
uniform bool isOrthographic;
// Point rendering (SplatCloud::setPointMix): 0 = Gaussians, 1 = a disc of
// 3 * splatPointSigma pixels radius per splat. See pointSigmaPixels().
uniform float splatPointMix;
uniform float splatPointSigma;

in vec3 position;   // unit quad corner, xy in [-1, 1]
in float splatIndex;// the splat this draw slot renders (sorted)

out vec3 vColor;
out float vOpacity;
out vec3 vConic;
out vec2 vDelta;

const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2_0 = 1.0925484305920792;
const float SH_C2_1 = -1.0925484305920792;
const float SH_C2_2 = 0.31539156525252005;
const float SH_C2_3 = -1.0925484305920792;
const float SH_C2_4 = 0.5462742152960396;
const float SH_C3_0 = -0.5900435899266435;
const float SH_C3_1 = 2.890611442640554;
const float SH_C3_2 = -0.4570457994644658;
const float SH_C3_3 = 0.3731763325901154;
const float SH_C3_4 = -0.4570457994644658;
const float SH_C3_5 = 1.445305721320277;
const float SH_C3_6 = -0.5900435899266435;

// The low-pass the EWA splatting formulation adds to the projected covariance:
// without it a splat narrower than a pixel projects to a near-singular conic and
// either aliases or inverts. It costs a little brightness on small splats; that
// is a deliberate trade (no Mip-Splatting compensation here).
const float SCREEN_DILATION = 0.3;

ivec2 splatTexel(int i) {

    return ivec2(i % splatTexWidth, i / splatTexWidth);
}

vec3 shCoeff(int base, int c) {

    return texelFetch(splatShTex, splatTexel(base + c), 0).rgb;
}

// dir points FROM the camera TO the splat, in world space.
vec3 shColor(int splat, vec3 dir) {

    int base = splat * splatShCoeffs;

    vec3 c = SH_C0 * shCoeff(base, 0);

    if (splatShDegree >= 1) {

        float x = dir.x;
        float y = dir.y;
        float z = dir.z;

        c += -SH_C1 * y * shCoeff(base, 1) + SH_C1 * z * shCoeff(base, 2) - SH_C1 * x * shCoeff(base, 3);

        if (splatShDegree >= 2) {

            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;

            c += SH_C2_0 * xy * shCoeff(base, 4) + SH_C2_1 * yz * shCoeff(base, 5) + SH_C2_2 * (2.0 * zz - xx - yy) * shCoeff(base, 6) + SH_C2_3 * xz * shCoeff(base, 7) + SH_C2_4 * (xx - yy) * shCoeff(base, 8);

            if (splatShDegree >= 3) {

                c += SH_C3_0 * y * (3.0 * xx - yy) * shCoeff(base, 9) + SH_C3_1 * xy * z * shCoeff(base, 10) + SH_C3_2 * y * (4.0 * zz - xx - yy) * shCoeff(base, 11) + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shCoeff(base, 12) + SH_C3_4 * x * (4.0 * zz - xx - yy) * shCoeff(base, 13) + SH_C3_5 * z * (xx - yy) * shCoeff(base, 14) + SH_C3_6 * x * (xx - 3.0 * yy) * shCoeff(base, 15);
            }
        }
    }

    // The +0.5 belongs to the evaluation, not the data: the optimiser
    // initialises the DC term as (rgb - 0.5) / SH_C0. Kept in step with
    // splats::SH_COLOR_OFFSET.
    //
    // Deliberately NOT clamped here. max(NaN, 0.0) returns 0.0 on most
    // hardware, which would quietly turn a corrupt coefficient into a
    // plausible-looking colour; the fragment stage checks for non-finite
    // values first and clamps afterwards.
    return c + 0.5;
}

void discardVertex() {

    // Behind the near plane, or numerically hopeless. z/w > 1 clips the whole
    // primitive away; the varyings still need defined values.
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vColor = vec3(0.0);
    vOpacity = 0.0;
    vConic = vec3(1.0, 0.0, 1.0);
    vDelta = vec2(0.0);
}

void main() {

    // Floats are exact to 2^24, so the sorted index survives the trip through
    // the splatIndex attribute intact.
    int splat = int(splatIndex + 0.5);

    vec4 meanOpacity = texelFetch(splatMeanTex, splatTexel(splat), 0);
    vec3 meanLocal = meanOpacity.xyz;

    vec4 viewPos = modelViewMatrix * vec4(meanLocal, 1.0);
    if (viewPos.z > -splatNear) {

        discardVertex();
        return;
    }

    // 3D covariance, symmetric, precomputed on the CPU as R*S*S^T*R^T.
    int covBase = splat * 2;
    vec4 c0 = texelFetch(splatCovTex, splatTexel(covBase), 0);
    vec4 c1 = texelFetch(splatCovTex, splatTexel(covBase + 1), 0);
    mat3 sigma = mat3(
            c0.x, c0.y, c0.z,
            c0.y, c0.w, c1.x,
            c0.z, c1.x, c1.y);

    // Into view space. W is modelView's linear part, so a scaled or rotated
    // cloud transforms correctly.
    mat3 W = mat3(modelViewMatrix);
    mat3 sigmaView = W * sigma * transpose(W);

    // EWA: J is the local affine approximation of the perspective divide at the
    // splat's centre. Focal lengths in pixels come straight out of the
    // projection matrix -- fx = 0.5 * width * P[0][0].
    float focalX = 0.5 * splatViewport.x * projectionMatrix[0][0];
    float focalY = 0.5 * splatViewport.y * projectionMatrix[1][1];

    mat3 J;
    if (isOrthographic) {

        // No divide to approximate: a parallel projection is already affine, so
        // the Jacobian is the constant pixels-per-world-unit scale. (For an
        // orthographic camera P[0][0] is 2/(right-left), which makes focalX
        // come out as exactly that.) Using the perspective form here would
        // shrink distant splats under a projection that has no perspective.
        J = mat3(
                focalX, 0.0, 0.0,
                0.0, focalY, 0.0,
                0.0, 0.0, 0.0);

    } else {

        float zInv = 1.0 / -viewPos.z;// viewPos.z < 0 here
        float zInv2 = zInv * zInv;

        J = mat3(
                focalX * zInv, 0.0, 0.0,
                0.0, focalY * zInv, 0.0,
                focalX * viewPos.x * zInv2, focalY * viewPos.y * zInv2, 0.0);
    }

    mat3 cov2 = J * sigmaView * transpose(J);

    float a = cov2[0][0] + SCREEN_DILATION;
    float b = cov2[0][1];
    float c = cov2[1][1] + SCREEN_DILATION;

    // Point mix: pull the footprint toward an isotropic disc of the requested
    // pixel size. After the dilation, so mix 1 is exactly the disc whatever
    // the splat was; guarded, so mix 0 leaves a, b, c untouched. The same
    // lerp as splat_project.comp.
    float opacity = meanOpacity.w;
    if (splatPointMix > 0.0) {

        float v = splatPointSigma * splatPointSigma;
        a = mix(a, v, splatPointMix);
        b = mix(b, 0.0, splatPointMix);
        c = mix(c, v, splatPointMix);
        opacity = mix(opacity, 1.0, splatPointMix);
    }

    float det = a * c - b * b;
    if (!(det > 0.0)) {// also catches NaN

        discardVertex();
        return;
    }

    // Conic = inverse of the 2D covariance, which is what the fragment stage
    // evaluates the Gaussian with.
    vConic = vec3(c, -b, a) / det;

    // 3 sigma of the larger principal axis, in pixels.
    float mid = 0.5 * (a + c);
    float lambda = mid + sqrt(max(0.0, mid * mid - det));
    float radius = 3.0 * sqrt(lambda);
    if (!(radius > 0.0)) {

        discardVertex();
        return;
    }
    // A splat straddling the near plane can project to an arbitrarily large
    // footprint; clamping keeps one bad Gaussian from shading the whole screen.
    radius = min(radius, max(splatViewport.x, splatViewport.y));

    vec4 clip = projectionMatrix * viewPos;
    vec2 ndc = clip.xy / clip.w;

    vDelta = position.xy * radius;

    // Offset in NDC, then scaled back by w so the perspective divide undoes it.
    // clip.z and clip.w pass through untouched, so depth stays the splat centre's.
    vec2 ndcQuad = ndc + vDelta * 2.0 / splatViewport;
    gl_Position = vec4(ndcQuad * clip.w, clip.z, clip.w);

    vOpacity = opacity;

    vec3 meanWorld = (modelMatrix * vec4(meanLocal, 1.0)).xyz;
    vColor = shColor(splat, normalize(meanWorld - cameraPosition));
}
)";

    return source;
}

const std::string& SplatCloud::fragmentShaderSource() {

    static const std::string source = R"(#version 330 core

in vec3 vColor;
in float vOpacity;
in vec3 vConic;
in vec2 vDelta;

uniform bool splatDebugNonFinite;
uniform float splatPointMix;
uniform float splatPointSigma;

out vec4 fragColor;

const vec4 DEBUG_MAGENTA = vec4(1.0, 0.0, 1.0, 1.0);

void main() {

    // The Gaussian, evaluated through its inverse 2D covariance.
    float power = -0.5 * (vConic.x * vDelta.x * vDelta.x + 2.0 * vConic.y * vDelta.x * vDelta.y + vConic.z * vDelta.y * vDelta.y);

    // Negated comparison so NaN — which fails every comparison — lands here too.
    if (!(power <= 0.0)) {

        if (splatDebugNonFinite && isnan(power)) {

            fragColor = DEBUG_MAGENTA;
            return;
        }
        discard;
    }

    float falloff = exp(power);
    if (splatPointMix > 0.0) {

        // The disc: distance from the centre in sigma units is sqrt(-2 power),
        // the edge is at 3 sigma, and (3 - t) * sigma is a one-pixel ramp
        // ending there. The same expression as splat_raster.comp.
        float disc = clamp((3.0 - sqrt(max(0.0, -2.0 * power))) * splatPointSigma, 0.0, 1.0);
        falloff = mix(falloff, disc, splatPointMix);
    }

    float alpha = vOpacity * falloff;
    if (!(alpha >= 0.00392156862)) {// 1/255: below this the blend is a no-op

        discard;
    }
    alpha = min(0.99, alpha);

    // Checked before the clamp below, not after: max(NaN, 0.0) is 0.0 on most
    // hardware, so clamping first would hide a corrupt SH coefficient behind a
    // colour that looks fine.
    if (any(isnan(vColor)) || any(isinf(vColor))) {

        if (splatDebugNonFinite) {

            fragColor = DEBUG_MAGENTA;
            return;
        }
        discard;
    }

    // Display-referred already: splats are trained against sRGB-ish images, so
    // no tone mapping happens here. Straight (non-premultiplied) alpha, matching
    // Blending::Normal.
    fragColor = vec4(max(vColor, vec3(0.0)), alpha);
}
)";

    return source;
}


// ---------------------------------------------------------------------------
// Object
// ---------------------------------------------------------------------------

SplatCloud::SplatCloud(SplatData data)
    : Mesh(splatQuad(data.count()), RawShaderMaterial::create()),
      data_(std::move(data)) {

    std::string why;
    if (!data_.validate(&why)) {

        throw std::invalid_argument("SplatCloud: invalid SplatData (" + why + ")");
    }

    // The file's quaternions are arbitrary length; the covariance is only a
    // rotation-times-scale if they are unit.
    data_.normalizeRotations();

    // dynamic, not static: Material is a *virtual* base of ShaderMaterial, and
    // static_cast cannot walk down from a virtual base.
    splatMaterial_ = std::dynamic_pointer_cast<RawShaderMaterial>(material());
    splatMaterial_->vertexShader = vertexShaderSource();
    splatMaterial_->fragmentShader = fragmentShaderSource();

    // transparent = true is not cosmetic: GLState disables blending outright for
    // Blending::Normal on an opaque material, and every splat past the first
    // would overwrite instead of blending.
    splatMaterial_->transparent = true;
    splatMaterial_->depthTest = true;
    splatMaterial_->depthWrite = false;
    splatMaterial_->side = Side::Double;

    // The state-carrying uniforms get their defaults HERE, before any setter
    // can run — never in the lazy build below, which would overwrite whatever
    // setViewportSize / setDebugNonFinite recorded in the meantime.
    splatMaterial_->uniforms["splatViewport"] = Uniform(Vector2(1.f, 1.f));
    splatMaterial_->uniforms["splatNear"] = Uniform(0.1f);
    splatMaterial_->uniforms["splatDebugNonFinite"] = Uniform(false);
    splatMaterial_->uniforms["splatPointMix"] = Uniform(pointMix_);
    splatMaterial_->uniforms["splatPointSigma"] = Uniform(pointSigmaPixels());

    // The DATA TEXTURES are built LAZILY, on the first GL draw or update() —
    // see ensureGlResources. Only the GL path ever gets there, so on a Vulkan
    // backend a 6M-splat cloud never allocates the ~1 GB of CPU-side texture
    // images it would never sample, and neither does every copy of it the
    // editor's undo history retains. Measured on the editor's two-scan
    // import/delete/import sequence: peak RAM 5.5 GB eager, 4.3 GB lazy.

    // A splat's footprint reaches well past its centre, so the bounds are the
    // means dilated by 3 sigma. frustumCulled stays on: the sphere is honest.
    const auto box = data_.computeBounds(3.f);
    Sphere sphere;
    if (!box.isEmpty()) {

        box.getBoundingSphere(sphere);
        // Onto the GEOMETRY, which is where Frustum::intersectsObject and
        // Box3::expandByObject read an ordinary Mesh's bounds from — and they
        // COMPUTE them from the position attribute if absent, which here is the
        // unit quad: a 1x1 box at the origin. Anything that measures a scene by
        // walking it (frame the document, focus the selection) would aim at a
        // postage stamp instead of the scan. Safe to overwrite per cloud because
        // splatQuad() builds a fresh geometry for each one.
        geometry_->boundingSphere = sphere;
        geometry_->boundingBox = box;

    } else {

        geometry_->boundingSphere = Sphere(Vector3{}, 0.f);
    }

    depths_.resize(data_.count());
    keys_.resize(data_.count());
    histogram_.resize(SORT_BUCKETS);

    // Safety net for callers who forget update(): the renderer has already
    // uploaded splatIndex by the time this runs, so the sort lands one frame
    // late — but the viewport uniform is read at draw time, so that is current.
    onBeforeRender = RenderCallback(
            [this](void* renderer, Object3D*, Camera* camera, BufferGeometry*, Material*, std::optional<GeometryGroup>) {
                if (auto* base = static_cast<Renderer*>(renderer)) {

                    // The bound render target's viewport, which is what the
                    // splat footprint has to be measured in. Virtual on the
                    // renderer interface, so this links without any concrete
                    // backend — the no-GLFW build depends on that.
                    Vector4 viewport;
                    base->getCurrentViewport(viewport);
                    setViewportSize(static_cast<int>(viewport.z), static_cast<int>(viewport.w));
                }

                if (camera) update(*camera);
            });
}

std::shared_ptr<SplatCloud> SplatCloud::create(SplatData data) {

    return std::make_shared<SplatCloud>(std::move(data));
}

void SplatCloud::ensureGlResources() {

    if (glResourcesBuilt_) return;
    glResourcesBuilt_ = true;

    buildTextures();
}

void SplatCloud::buildTextures() {

    const size_t n = data_.count();
    const int coeffs = data_.coeffCount();

    // Allocate the texture first and fill its pixels in place, rather than
    // filling a staging vector and handing it over. DataTexture::create takes
    // its ImageData by const reference and the constructor takes it by value,
    // so passing a buffer in costs a full copy — at five million splats the SH
    // texture alone is 1.2 GB, and the copy put 2.5 GB in flight for no reason.
    auto allocate = [](int height) {
        auto texture = DataTexture::create<float>(4, TEX_WIDTH, static_cast<unsigned int>(height));
        texture->format = Format::RGBA;
        texture->type = Type::Float;
        return texture;
    };

    // --- means + opacity: one texel each -----------------------------------
    {
        meanTexture_ = allocate(std::max(1, texHeightFor(n, "mean", n)));
        auto& texels = meanTexture_->image().data<float>();

        for (size_t i = 0; i < n; ++i) {

            texels[i * 4 + 0] = data_.means[i].x;
            texels[i * 4 + 1] = data_.means[i].y;
            texels[i * 4 + 2] = data_.means[i].z;
            texels[i * 4 + 3] = data_.opacities[i];
        }
    }

    // --- 3D covariance: six floats, two texels each -------------------------
    {
        covTexture_ = allocate(std::max(1, texHeightFor(n * 2, "covariance", n)));
        auto& texels = covTexture_->image().data<float>();

        for (size_t i = 0; i < n; ++i) {

            float cov[6];
            data_.computeCovariance(i, cov);

            const size_t t = i * 2 * 4;
            texels[t + 0] = cov[0];// xx
            texels[t + 1] = cov[1];// xy
            texels[t + 2] = cov[2];// xz
            texels[t + 3] = cov[3];// yy
            texels[t + 4] = cov[4];// yz
            texels[t + 5] = cov[5];// zz
        }
    }

    // --- SH: one texel per coefficient, alpha unused ------------------------
    //
    // HALF, not float, and the only one of the three that is. SH coefficients
    // are small (|c| < ~4 after activation), smooth, and every one of them is
    // multiplied by a basis function and summed -- the error budget is a
    // fraction of a colour LSB. The mean and covariance stay fp32 because the
    // shader INVERTS the projected covariance: half's 11-bit significand on
    // Sxx..Szz costs real precision in the conic, and a near-singular conic is
    // how a splat renderer produces a screen-wide smear from nothing.
    //
    // It is also where all the memory is. At 5M splats and degree 3 the SH
    // texture is 16 texels per splat against 1 for the mean and 2 for the
    // covariance: 1220 MB of the 1450 MB total, halved to 610 MB.
    {
        shTexture_ = DataTexture::create<std::uint16_t>(
                4, TEX_WIDTH,
                static_cast<unsigned int>(std::max(1, texHeightFor(n * static_cast<size_t>(coeffs), "SH", n))));
        shTexture_->format = Format::RGBA;
        shTexture_->type = Type::HalfFloat;

        auto& texels = shTexture_->image().data<std::uint16_t>();

        for (size_t i = 0; i < n; ++i) {

            const float* c = data_.shAt(i);
            for (int k = 0; k < coeffs; ++k) {

                const size_t t = (i * static_cast<size_t>(coeffs) + k) * 4;
                texels[t + 0] = DataUtils::toHalfFloat(c[k * 3 + 0]);
                texels[t + 1] = DataUtils::toHalfFloat(c[k * 3 + 1]);
                texels[t + 2] = DataUtils::toHalfFloat(c[k * 3 + 2]);
            }
        }
    }

    auto& uniforms = splatMaterial_->uniforms;
    uniforms["splatMeanTex"] = Uniform(static_cast<Texture*>(meanTexture_.get()));
    uniforms["splatCovTex"] = Uniform(static_cast<Texture*>(covTexture_.get()));
    uniforms["splatShTex"] = Uniform(static_cast<Texture*>(shTexture_.get()));
    uniforms["splatTexWidth"] = Uniform(TEX_WIDTH);
    uniforms["splatShCoeffs"] = Uniform(coeffs);
    uniforms["splatShDegree"] = Uniform(data_.shDegree);
    // splatViewport / splatNear / splatDebugNonFinite are NOT set here, and
    // that is a hard rule: they carry STATE (setViewportSize, update,
    // setDebugNonFinite), this function runs LAZILY on first GL use, and a
    // default written here would stomp whatever a setter recorded in between —
    // which is exactly how the NaN-sentinel debug flag silently became false
    // and the sentinel test's magenta went to zero when the build went lazy.
    // Their defaults live in the constructor, before any setter can run.
}

std::size_t SplatCloud::cpuBytes() const {

    std::size_t bytes = sizeof(*this) + data_.byteSize();

    // The sorted index, which splatQuad allocates because the renderer needs the
    // attribute to exist before it walks the scene, and the counting-sort
    // scratch alongside it. 10 bytes a splat. No instanceMatrix line any more:
    // this used to derive from InstancedMesh and hold 64 bytes a splat of
    // identity matrices the splat shader never declared.
    if (const auto* index = splatIndexAttribute()) bytes += index->byteLength();

    bytes += depths_.capacity() * sizeof(float) +
             sample_.capacity() * sizeof(float) +
             keys_.capacity() * sizeof(std::uint16_t) +
             histogram_.capacity() * sizeof(std::uint32_t);

    // The data textures, on the other hand, appear only once a GL frame has drawn
    // this cloud (see ensureGlResources): 176 bytes a splat at SH degree 3 that a
    // cloud the Vulkan backend alone has drawn never allocates, which is the whole
    // point of lazy.
    for (const auto* texture : {meanTexture_.get(), covTexture_.get(), shTexture_.get()}) {
        if (texture) bytes += texture->image().byteSize();
    }

    bytes += lodTable_.levels.capacity() * sizeof(splats::LodLevel);
    for (const auto& level : lodTable_.levels) {
        bytes += level.chunks.capacity() * sizeof(splats::LodChunk);
    }
    bytes += submitRanges_.capacity() * sizeof(std::pair<std::uint32_t, std::uint32_t>);

    return bytes;
}

void SplatCloud::setViewportSize(int width, int height) {

    splatMaterial_->uniforms["splatViewport"].setValue(
            Vector2(static_cast<float>(std::max(1, width)), static_cast<float>(std::max(1, height))));
}

void SplatCloud::setDebugNonFinite(bool flag) {

    debugNonFinite_ = flag;
    splatMaterial_->uniforms["splatDebugNonFinite"].setValue(flag);
}

void SplatCloud::setPointMix(float mix) {

    pointMix_ = std::isfinite(mix) ? std::clamp(mix, 0.f, 1.f) : 0.f;
    splatMaterial_->uniforms["splatPointMix"].setValue(pointMix_);
}

void SplatCloud::setPointSize(float pixels) {

    pointSize_ = std::isfinite(pixels) ? std::max(1.f, pixels) : 2.f;
    splatMaterial_->uniforms["splatPointSigma"].setValue(pointSigmaPixels());
}

float SplatCloud::pointSigmaPixels() const {

    return (pointSize_ * 0.5f + 0.5f) / 3.f;
}

void SplatCloud::raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) {

    // The geometry's boundingSphere is the 3-sigma bound the constructor already
    // computed for frustum culling, in the cloud's own coordinates; the world
    // matrix takes it to where the ray is. A cloud with no splats has radius 0
    // and is not clickable, which is the honest answer for something that draws
    // nothing.
    if (!visible || data_.count() == 0) return;

    const auto& bounds = geometry_->boundingSphere;
    if (!bounds || bounds->radius <= 0.f) return;

    Sphere sphere;
    sphere.copy(*bounds);
    sphere.applyMatrix4(*matrixWorld);

    Vector3 point;
    raycaster.ray.intersectSphere(sphere, point);
    // The miss and the entirely-behind-the-ray cases both come back NaN.
    if (std::isnan(point.x)) return;

    const float distance = raycaster.ray.origin.distanceTo(point);
    if (distance < raycaster.nearPlane || distance > raycaster.farPlane) return;

    Intersection intersection{};
    intersection.distance = distance;
    intersection.point = point;
    intersection.object = this;
    intersects.push_back(intersection);
}

void SplatCloud::update(Camera& camera) {

    // update() is the GL path's documented per-frame entry, so it is the other
    // legitimate trigger for the lazy GL resources (onBeforeRender being the
    // first — a caller that renders without ever calling update() still draws).
    ensureGlResources();

    // Both matrices must be current before the sort reads them; a caller that
    // just moved the camera has not necessarily re-rendered yet.
    camera.updateWorldMatrix(true, false);
    this->updateWorldMatrix(true, false);

    splatMaterial_->uniforms["splatNear"].setValue(camera.nearPlane);

    sortByDepth(camera);
}

void SplatCloud::sortByDepth(Camera& camera) {

    const size_t n = data_.count();
    if (n == 0) return;

    Matrix4 modelView;
    modelView.multiplyMatrices(camera.matrixWorldInverse, *matrixWorld);
    const auto& e = modelView.elements;

    if (sorted_ && std::equal(e.begin(), e.end(), lastSortMatrix_.begin())) return;
    std::copy(e.begin(), e.end(), lastSortMatrix_.begin());

    // View-space z of each mean. Column-major elements, so row 2 is
    // (e[2], e[6], e[10], e[14]).
    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < n; ++i) {

        const auto& m = data_.means[i];
        const float z = e[2] * m.x + e[6] * m.y + e[10] * m.z + e[14];
        depths_[i] = z;
        minZ = std::min(minZ, z);
        maxZ = std::max(maxZ, z);
    }

    // GL view space looks down -z, so the farthest splat has the most negative
    // z: ascending z IS back-to-front. Quantise into 16 bits and counting-sort.
    //
    // The quantisation interval is p1..p99 of a fixed-stride sample of the
    // depths rather than min..max, so a handful of strays cannot rob the rest
    // of the cloud of its resolution. See the constants above.
    float lo = minZ;
    float hi = maxZ;

    {
        const size_t stride = std::max<size_t>(1, n / SORT_SAMPLE_TARGET);

        sample_.clear();
        sample_.reserve(n / stride + 1);
        // Non-finite depths are dropped rather than ranked: NaN breaks the
        // strict weak ordering nth_element is entitled to assume, and a single
        // corrupt mean would otherwise poison the interval for the whole
        // cloud. Such a splat still gets a key below (clamped, harmlessly) and
        // the shader still refuses to draw it.
        for (size_t i = 0; i < n; i += stride) {

            if (std::isfinite(depths_[i])) sample_.push_back(depths_[i]);
        }

        if (sample_.size() >= 3) {

            const auto last = sample_.size() - 1;
            const auto loRank = static_cast<size_t>(SORT_CLAMP_LO * static_cast<float>(last));
            const auto hiRank = static_cast<size_t>(SORT_CLAMP_HI * static_cast<float>(last));

            std::nth_element(sample_.begin(), sample_.begin() + static_cast<std::ptrdiff_t>(loRank), sample_.end());
            const float pLo = sample_[loRank];

            std::nth_element(sample_.begin() + static_cast<std::ptrdiff_t>(loRank) + 1,
                             sample_.begin() + static_cast<std::ptrdiff_t>(hiRank), sample_.end());
            const float pHi = sample_[hiRank];

            if (pHi > pLo) {

                const float mid = 0.5f * (pLo + pHi);
                const float half = 0.5f * (pHi - pLo) * (1.f + SORT_CLAMP_MARGIN);
                lo = mid - half;
                hi = mid + half;
            }
            // Otherwise the middle 98% of the cloud is at one depth (a flat
            // cloud seen face on, or a tiny one). min..max is then both the
            // honest interval and the one that still separates anything.
        }
    }

    // Three monotone segments: the content interval gets almost the whole
    // range, and each tail keeps its own small band (see SORT_TAIL_BUCKETS).
    // When a tail is empty — lo already at the cloud's edge, or the margin
    // pushed past it — its scale is zero and the segment degenerates to the
    // old clamp, harmlessly.
    constexpr int TAIL = SORT_TAIL_BUCKETS;
    constexpr int CONTENT = SORT_BUCKETS - 2 * TAIL;

    const float loSpan = lo - minZ;
    const float midSpan = hi - lo;
    const float hiSpan = maxZ - hi;

    const float loScale = loSpan > 0.f ? static_cast<float>(TAIL - 1) / loSpan : 0.f;
    const float midScale = midSpan > 0.f ? static_cast<float>(CONTENT - 1) / midSpan : 0.f;
    const float hiScale = hiSpan > 0.f ? static_cast<float>(TAIL - 1) / hiSpan : 0.f;

    std::fill(histogram_.begin(), histogram_.end(), 0u);

    for (size_t i = 0; i < n; ++i) {

        // std::clamp passes NaN straight through — NaN loses both of its
        // comparisons — and a float-to-integer conversion of NaN is undefined
        // behaviour, not a large number. Every branch is written so NaN falls
        // into a plain integer answer (the far end, bucket 0).
        const float z = depths_[i];
        std::uint16_t key;

        if (!(z >= lo)) {// below the interval, or NaN

            const float k = (z - minZ) * loScale;
            key = !(k > 0.f)                                ? 0
                : (k >= static_cast<float>(TAIL - 1))       ? static_cast<std::uint16_t>(TAIL - 1)
                                                            : static_cast<std::uint16_t>(k);
        } else if (z <= hi) {// the content interval

            const float k = (z - lo) * midScale;
            key = static_cast<std::uint16_t>(
                    TAIL + ((k >= static_cast<float>(CONTENT - 1)) ? CONTENT - 1
                                                                   : static_cast<int>(k)));
        } else {// beyond the interval

            const float k = (z - hi) * hiScale;
            key = static_cast<std::uint16_t>(
                    (SORT_BUCKETS - TAIL) +
                    (!(k > 0.f)                          ? 0
                     : (k >= static_cast<float>(TAIL - 1)) ? TAIL - 1
                                                           : static_cast<int>(k)));
        }

        keys_[i] = key;
        ++histogram_[keys_[i]];
    }

    std::uint32_t running = 0;
    for (int bucket = 0; bucket < SORT_BUCKETS; ++bucket) {

        const std::uint32_t hits = histogram_[bucket];
        histogram_[bucket] = running;
        running += hits;
    }

    auto* index = splatIndexAttribute();
    auto& slots = index->array();
    for (size_t i = 0; i < n; ++i) {

        slots[static_cast<size_t>(histogram_[keys_[i]]++)] = static_cast<float>(i);
    }

    index->needsUpdate();
    sorted_ = true;
}
