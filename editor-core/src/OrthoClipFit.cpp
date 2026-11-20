
#include "threepp/extras/editor/OrthoClipFit.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/math/Box3.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace threepp;
using namespace threepp::editor;

float OrthoClipFit::farPlane(const OrthographicCamera& camera, const Box3& bounds, const Sheet* sheet) {

    Vector3 forward(0, 0, -1), right(1, 0, 0), up(0, 1, 0);
    forward.applyQuaternion(camera.quaternion);
    right.applyQuaternion(camera.quaternion);
    up.applyQuaternion(camera.quaternion);

    const auto depthOf = [&](const Vector3& world) {
        Vector3 offset;
        offset.subVectors(world, camera.position);
        return offset.dot(forward);
    };

    // The far side of the bounds, as a sphere, so the answer does not jump
    // while the view turns about them.
    float reach = 0.f;
    if (!bounds.isEmpty()) {
        reach = depthOf(bounds.getCenter()) + bounds.getSize().length() * 0.5f;
    }

    const float zoom = std::max(camera.zoom, 1e-4f);
    const float halfHeight = std::abs(camera.top - camera.bottom) * 0.5f / zoom;
    const float halfWidth = std::abs(camera.right - camera.left) * 0.5f / zoom;

    if (sheet && sheet->radius > 0.f) {
        // Depth is linear across the frustum, so the deepest point the view
        // sees of the sheet's plane is under one of its four corners. Edge on,
        // that is nowhere: the sheet's own rim is what bounds it then.
        const float rim = depthOf(sheet->centre) + sheet->radius;
        const float facing = forward.dot(sheet->normal);
        float seen = rim;
        if (std::abs(facing) > 1e-4f) {
            seen = -std::numeric_limits<float>::infinity();
            for (const float sx : {-1.f, 1.f}) {
                for (const float sy : {-1.f, 1.f}) {
                    Vector3 corner = camera.position;
                    corner.addScaledVector(right, sx * halfWidth).addScaledVector(up, sy * halfHeight);
                    Vector3 toPlane;
                    toPlane.subVectors(sheet->centre, corner);
                    seen = std::max(seen, toPlane.dot(sheet->normal) / facing);
                }
            }
        }
        reach = std::max(reach, std::min(seen, rim));
    }

    // Room behind for what is drawn around a subject at a size set by the view
    // rather than by the scene: a gizmo, a selection's outline.
    reach += std::max(halfHeight, 0.05f);

    const float depth = std::max(reach - camera.nearPlane, 1.f);
    return camera.nearPlane + std::exp2(std::ceil(std::log2(depth) * 2.f) * 0.5f);
}
