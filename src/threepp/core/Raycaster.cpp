
#include "threepp/core/Raycaster.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"

#include "threepp/utils/InstanceOf.hpp"

#include <algorithm>

using namespace threepp;

namespace {

    bool ascSort(const Intersection &a, const Intersection &b) {

        return a.distance < b.distance;
    }

    void intersectObject(Object3D *object, Raycaster &raycaster, std::vector<Intersection> &intersects, bool recursive) {

        if (object->layers.test(raycaster.layers)) {

            object->raycast(raycaster, intersects);
        }

        if (recursive) {

            auto &children = object->children;

            for (auto &child : children) {

                intersectObject(child, raycaster, intersects, true);
            }
        }
    }

}// namespace


void Raycaster::set(const Vector3 &origin, const Vector3 &direction) {

    // direction is assumed to be normalized (for accurate distance calculations)

    this->ray.set(origin, direction);
}

std::vector<Intersection> Raycaster::intersectObject(Object3D *object, bool recursive) {

    std::vector<Intersection> intersects;

    ::intersectObject(object, *this, intersects, recursive);

    std::stable_sort(intersects.begin(), intersects.end(), &ascSort);

    return intersects;
}

std::vector<Intersection> Raycaster::intersectObjects(std::vector<Object3D *> &objects, bool recursive) {

    std::vector<Intersection> intersects;

    for (auto &object : objects) {

        ::intersectObject(object, *this, intersects, recursive);
    }

    std::stable_sort(intersects.begin(), intersects.end(), &ascSort);

    return intersects;
}

void Raycaster::setFromCamera(const Vector2 &coords, const std::shared_ptr<Camera> &camera) {

    if (instanceof <PerspectiveCamera>(camera.get())) {

        this->ray.origin.setFromMatrixPosition(camera->matrixWorld);
        this->ray.direction.set(coords.x, coords.y, 0.5f).unproject(*camera).sub(this->ray.origin).normalize();

    } else if (instanceof <OrthographicCamera>(camera.get())) {

        this->ray.origin.set(coords.x, coords.y, (camera->near + camera->far) / (camera->near - camera->far)).unproject(*camera);// set origin in plane of camera
        this->ray.direction.set(0, 0, -1).transformDirection(camera->matrixWorld);

    } else {
        std::cerr << "THREE.Raycaster: Unsupported camera type" << std::endl;
    }
}
