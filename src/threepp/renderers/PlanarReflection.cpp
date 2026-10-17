
#include "threepp/renderers/PlanarReflection.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Vector4.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/math/Sphere.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

namespace {

    // Objects whose bounding sphere lies wholly beyond `far` from `eye` are
    // set invisible for the pass: the far town and its lamps cost nothing in
    // the mirror, which is what the far limit is for (the projection's far
    // plane alone cannot do it: an oblique near plane skews it). An object
    // that straddles the limit, the terrain first of all, stays. World
    // matrices are taken as they are: a frame-old matrix on a moving boat
    // is well within the margin.
    void hideBeyond(Object3D& object, const Vector3& eye, float far, std::vector<Object3D*>& hidden) {

        if (!object.visible) return;

        if (const auto geometry = object.geometry()) {

            if (!geometry->boundingSphere) geometry->computeBoundingSphere();
            if (geometry->boundingSphere) {
                Sphere sphere = *geometry->boundingSphere;
                sphere.applyMatrix4(*object.matrixWorld);
                if (sphere.center.distanceTo(eye) - 1.1f * sphere.radius > far) {
                    object.visible = false;
                    hidden.push_back(&object);
                    return;// its children go with it
                }
            }
        }

        for (const auto& child : object.children) {
            hideBeyond(*child, eye, far, hidden);
        }
    }

}// namespace

struct PlanarReflection::Impl {

    Vector3 point{0, 0, 0};
    Vector3 normal{0, 1, 0};

    PerspectiveCamera virtualCamera;
    Matrix4 textureMatrix;
    std::unique_ptr<RenderTarget> target;
    std::vector<Object3D*> farHidden;// hidden by the far limit this pass

    // scratch
    Vector3 cameraWorldPosition, view, lookAtPosition, lookTarget, up;
    Matrix4 rotationMatrix;
    Plane reflectorPlane;
    Vector4 clipPlane, q;

    void ensureTarget(const Renderer& renderer, float scale) {

        const auto size = renderer.size();
        const auto w = std::max(1u, static_cast<unsigned int>(std::lround(static_cast<float>(size.width()) * scale)));
        const auto h = std::max(1u, static_cast<unsigned int>(std::lround(static_cast<float>(size.height()) * scale)));

        if (!target) {

            RenderTarget::Options parameters;
            parameters.minFilter = Filter::Linear;
            parameters.magFilter = Filter::Linear;
            parameters.wrapS = TextureWrapping::ClampToEdge;
            parameters.wrapT = TextureWrapping::ClampToEdge;
            parameters.format = Format::RGBA;
            // Scene-linear HDR, as Reflector and the transmission target: the
            // material mixes it into radiance before tone mapping, so a
            // highlight brighter than 1 has to arrive intact.
            parameters.type = Type::HalfFloat;
            parameters.generateMipmaps = false;

            target = RenderTarget::create(w, h, parameters);

        } else if (target->width != w || target->height != h) {

            target->setSize(w, h);
        }
    }

    // Reflector's camera construction: the camera's position, its look
    // target and its up vector mirrored about the plane.
    bool placeCamera(const Camera& camera, float farDistance) {

        cameraWorldPosition.setFromMatrixPosition(*camera.matrixWorld);

        view.subVectors(point, cameraWorldPosition);
        if (view.dot(normal) > 0) return false;// the camera is behind the plane

        view.reflect(normal).negate();
        view.add(point);

        rotationMatrix.extractRotation(*camera.matrixWorld);
        lookAtPosition.set(0, 0, -1);
        lookAtPosition.applyMatrix4(rotationMatrix);
        lookAtPosition.add(cameraWorldPosition);

        lookTarget.subVectors(point, lookAtPosition);
        lookTarget.reflect(normal).negate();
        lookTarget.add(point);

        virtualCamera.position.copy(view);
        up.set(0, 1, 0);
        up.applyMatrix4(rotationMatrix);
        up.reflect(normal);
        virtualCamera.up.copy(up);
        virtualCamera.lookAt(lookTarget);
        virtualCamera.updateMatrixWorld();

        if (const auto* perspective = camera.as<PerspectiveCamera>()) {

            // The camera's own frustum, closed at the far limit.
            virtualCamera.fov = perspective->fov;
            virtualCamera.aspect = perspective->aspect;
            virtualCamera.zoom = perspective->zoom;
            virtualCamera.filmGauge = perspective->filmGauge;
            virtualCamera.filmOffset = perspective->filmOffset;
            virtualCamera.nearPlane = perspective->nearPlane;
            virtualCamera.farPlane = farDistance > 0 ? std::min(perspective->farPlane, farDistance) : perspective->farPlane;
            virtualCamera.updateProjectionMatrix();

        } else {

            virtualCamera.nearPlane = camera.nearPlane;
            virtualCamera.farPlane = camera.farPlane;
            virtualCamera.projectionMatrix.copy(camera.projectionMatrix);
        }

        return true;
    }

    // World position -> texture coordinates; xy/w is what the material uses,
    // so the oblique rewrite of the z row below does not touch it.
    void updateTextureMatrix(bool flipY) {

        textureMatrix.set(0.5f, 0.f, 0.f, 0.5f,
                          0.f, 0.5f, 0.f, 0.5f,
                          0.f, 0.f, 0.5f, 0.5f,
                          0.f, 0.f, 0.f, 1.f);
        textureMatrix.multiply(virtualCamera.projectionMatrix);
        textureMatrix.multiply(virtualCamera.matrixWorldInverse);

        // GL targets have (0,0) at the bottom left: nothing to flip. Kept, as in
        // Reflector, for a backend whose targets are top-left-origin.
        if (flipY) {
            auto& e = textureMatrix.elements;
            e[1] = e[3] - e[1];
            e[5] = e[7] - e[5];
            e[9] = e[11] - e[9];
            e[13] = e[15] - e[13];
        }
    }

    // Lengyel's oblique near plane: the mirror plane becomes the near plane,
    // so nothing behind the mirror (what is really above the water, mirrored)
    // reaches the target. http://www.terathon.com/lengyel/Lengyel-Oblique.pdf
    void obliqueNearPlane(float clipBias) {

        reflectorPlane.setFromNormalAndCoplanarPoint(normal, point);
        reflectorPlane.applyMatrix4(virtualCamera.matrixWorldInverse);
        clipPlane.set(reflectorPlane.normal.x, reflectorPlane.normal.y, reflectorPlane.normal.z, reflectorPlane.constant);

        auto& projectionMatrix = virtualCamera.projectionMatrix;
        q.x = (static_cast<float>(math::sgn(clipPlane.x)) + projectionMatrix.elements[8]) / projectionMatrix.elements[0];
        q.y = (static_cast<float>(math::sgn(clipPlane.y)) + projectionMatrix.elements[9]) / projectionMatrix.elements[5];
        q.z = -1.f;
        q.w = (1.f + projectionMatrix.elements[10]) / projectionMatrix.elements[14];

        clipPlane.multiplyScalar(2.f / clipPlane.dot(q));

        projectionMatrix.elements[2] = clipPlane.x;
        projectionMatrix.elements[6] = clipPlane.y;
        projectionMatrix.elements[10] = clipPlane.z + 1.f - clipBias;
        projectionMatrix.elements[14] = clipPlane.w;
    }
};

PlanarReflection::PlanarReflection(const Options& options)
    : options(options), pimpl_(std::make_unique<Impl>()) {}

PlanarReflection::~PlanarReflection() = default;

void PlanarReflection::setPlane(const Vector3& point, const Vector3& normal) {

    pimpl_->point.copy(point);
    pimpl_->normal.copy(normal).normalize();
}

const Vector3& PlanarReflection::planePoint() const {

    return pimpl_->point;
}

const Vector3& PlanarReflection::planeNormal() const {

    return pimpl_->normal;
}

bool PlanarReflection::render(Renderer& renderer, Object3D& scene, Camera& camera) {

    auto& impl = *pimpl_;

    camera.updateMatrixWorld();
    if (!impl.placeCamera(camera, options.farDistance)) return false;

    impl.ensureTarget(renderer, options.resolutionScale);
    impl.updateTextureMatrix(renderer.renderTargetFlipY());
    impl.obliqueNearPlane(options.clipBias);

    // What the pass must not disturb: the target, the clear colour, the
    // shadow maps (already rendered for this frame, and a light's map does
    // not depend on the camera), the hidden objects, the background.
    auto* currentTarget = renderer.getRenderTarget();
    const bool shadowAutoUpdate = renderer.shadowMapAutoUpdate;
    Color clearColor;
    renderer.getClearColor(clearColor);
    const float clearAlpha = renderer.getClearAlpha();

    std::vector<std::pair<Object3D*, bool>> wasVisible;
    wasVisible.reserve(hidden.size());
    for (auto* object : hidden) {
        if (!object) continue;
        wasVisible.emplace_back(object, object->visible);
        object->visible = false;
    }
    if (options.farDistance > 0) {
        impl.farHidden.clear();
        hideBeyond(scene, impl.virtualCamera.position, options.farDistance, impl.farHidden);
    }

    auto* asScene = scene.as<Scene>();
    Background background;
    if (asScene && !options.background) {
        background = asScene->background;
        asScene->background = Background();
    }

    renderer.shadowMapAutoUpdate = false;
    // Transparent black: the material's mix falls back to the environment
    // wherever nothing was drawn.
    renderer.setClearColor(Color(0, 0, 0), 0.f);
    renderer.setRenderTarget(impl.target.get());
    renderer.setDepthMask(true);
    if (!renderer.autoClear) renderer.clear();

    renderer.render(scene, impl.virtualCamera);

    renderer.setRenderTarget(currentTarget);
    renderer.setClearColor(clearColor, clearAlpha);
    renderer.shadowMapAutoUpdate = shadowAutoUpdate;

    if (asScene && !options.background) {
        asScene->background = background;
    }
    for (auto* object : impl.farHidden) {
        object->visible = true;
    }
    for (auto& [object, visible] : wasVisible) {
        object->visible = visible;
    }

    return true;
}

size_t PlanarReflection::culledByFar() const {

    return pimpl_->farHidden.size();
}

std::shared_ptr<Texture> PlanarReflection::texture() const {

    return pimpl_->target ? pimpl_->target->texture : nullptr;
}

const Matrix4& PlanarReflection::textureMatrix() const {

    return pimpl_->textureMatrix;
}

void PlanarReflection::apply(MaterialWithPlanarReflection& material) const {

    material.planarReflection = texture();
    material.planarReflectionMatrix.copy(pimpl_->textureMatrix);
}
