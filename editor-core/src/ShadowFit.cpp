
#include "threepp/extras/editor/ShadowFit.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Sphere.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Smallest gap the near plane is allowed to leave. A light standing inside
    // the scene would otherwise ask for a negative near.
    constexpr float minNear = 0.05f;

    bool assign(float& slot, float value) {

        if (std::abs(slot - value) <= 1e-4f * std::max(1.f, std::abs(value))) return false;
        slot = value;
        return true;
    }

    // World-space AABB of one object's own geometry. Not Box3::setFromObject,
    // which walks the subtree and would re-visit every descendant once per
    // ancestor during a traverse.
    bool objectBounds(Object3D& object, Box3& out) {

        auto geometry = object.geometry();
        if (!geometry) return false;

        if (!geometry->boundingBox) geometry->computeBoundingBox();
        if (!geometry->boundingBox) return false;

        out.copy(*geometry->boundingBox).applyMatrix4(*object.matrixWorld);
        return !out.isEmpty();
    }

}// namespace


bool ShadowFit::fit(DirectionalLight& light, const Box3& bounds) {

    if (!light.shadow) return false;
    if (bounds.isEmpty()) return false;

    auto* camera = dynamic_cast<OrthographicCamera*>(light.shadow->camera.get());
    if (!camera) return false;

    // The bounding SPHERE, not the box: a sphere has the same silhouette from
    // every direction, so the extent does not change as the light swings around
    // and the shadow does not swim.
    Sphere sphere;
    bounds.getBoundingSphere(sphere);

    const float radius = std::max(sphere.radius, 1e-3f);

    // Where the shadow camera will stand and look, mirroring
    // LightShadow::updateMatrices: at the light, aimed at its target. Both
    // world matrices are read as they stand — fitAll updates the graph first,
    // and any other caller is expected to have done the same.
    Vector3 lightPosition;
    lightPosition.setFromMatrixPosition(*light.matrixWorld);

    Vector3 targetPosition;
    targetPosition.setFromMatrixPosition(*light.target().matrixWorld);

    Vector3 viewDirection = targetPosition;
    viewDirection.sub(lightPosition);
    if (viewDirection.lengthSq() < 1e-12f) return false;// light sits on its target
    viewDirection.normalize();

    Vector3 toCentre = sphere.center;
    toCentre.sub(lightPosition);

    // Split the offset to the centre into "along the view axis", which sets
    // near/far, and "across it", which the extents have to cover because the
    // ortho box is centred on the axis and the scene need not be.
    const float along = toCentre.dot(viewDirection);

    Vector3 across = viewDirection;
    across.multiplyScalar(along);
    across.subVectors(toCentre, across);

    const float extent = across.length() + radius;

    const float nearPlane = std::max(minNear, along - radius);
    const float farPlane = std::max(nearPlane + minNear, along + radius);

    bool changed = false;
    changed |= assign(camera->left, -extent);
    changed |= assign(camera->right, extent);
    changed |= assign(camera->top, extent);
    changed |= assign(camera->bottom, -extent);
    changed |= assign(camera->nearPlane, nearPlane);
    changed |= assign(camera->farPlane, farPlane);

    if (changed) camera->updateProjectionMatrix();

    return changed;
}

Box3 ShadowFit::shadowBounds(Object3D& scene) {

    scene.updateMatrixWorld();

    Box3 bounds;
    bounds.makeEmpty();

    // What casts or receives, not everything: a light's default target sits at
    // the origin and helpers have no business setting the shadow extent.
    Box3 one;
    scene.traverse([&](Object3D& object) {
        if (!object.visible) return;
        if (!object.castShadow && !object.receiveShadow) return;
        if (!objectBounds(object, one)) return;

        bounds.union_(one);
    });

    return bounds;
}

int ShadowFit::fitAll(Object3D& scene) {

    std::vector<DirectionalLight*> lights;
    scene.traverse([&](Object3D& object) {
        if (auto* light = object.as<DirectionalLight>()) {
            if (light->castShadow) lights.emplace_back(light);
        }
    });

    if (lights.empty()) return 0;

    const Box3 bounds = shadowBounds(scene);
    if (bounds.isEmpty()) return 0;

    int changed = 0;
    for (auto* light : lights) {
        if (fit(*light, bounds)) changed++;
    }

    return changed;
}
