// https://github.com/mrdoob/three.js/blob/r129/src/core/Raycaster.js

#ifndef THREEPP_RAYCASTER_HPP
#define THREEPP_RAYCASTER_HPP

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/Layers.hpp"
#include "threepp/core/Face3.hpp"
#include "threepp/math/Ray.hpp"

#include <limits>
#include <vector>

namespace threepp {

    class Object3D;

    struct Intersection {

        float distance;
        Vector3 point;
        Object3D* object;

        std::optional<int> faceIndex;
        std::optional<Vector2> uv;
        std::optional<Vector2> uv2;
        std::optional<Face3> face;
    };

    class Raycaster {

    public:

        float near;
        float far;

        Ray ray;
        Camera* camera;
        Layers layers{};

        explicit Raycaster(const Vector3 &origin = Vector3(), const Vector3 &direction = Vector3(), float near = 0, float far = std::numeric_limits<float>::infinity())
            : near(near), far(far), ray(origin, direction) {}

        void set(const Vector3 &origin, const Vector3 &direction);

        void setFromCamera(const Vector2 &coords, const std::shared_ptr<Camera> &camera);

        std::vector<Intersection> intersectObject(Object3D *object, bool recursive = false);

        std::vector<Intersection> intersectObjects(std::vector<std::shared_ptr<Object3D>> &objects, bool recursive = false);

    };

}// namespace threepp

#endif//THREEPP_RAYCASTER_HPP
