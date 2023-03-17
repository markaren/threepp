// https://github.com/mrdoob/three.js/blob/r129/src/core/Raycaster.js

#ifndef THREEPP_RAYCASTER_HPP
#define THREEPP_RAYCASTER_HPP

#include "threepp/core/Face3.hpp"
#include "threepp/core/Layers.hpp"
#include "threepp/math/Ray.hpp"
#include "threepp/math/Vector2.hpp"

#include <limits>
#include <memory>
#include <vector>

namespace threepp {

    class Camera;
    class Object3D;

    struct Intersection {

        float distance;
        Vector3 point;
        Object3D* object;

        std::optional<int> index;
        std::optional<int> faceIndex;
        std::optional<Vector2> uv;
        std::optional<Vector2> uv2;
        std::optional<Face3> face;
        std::optional<int> instanceId;
        std::optional<float> distanceToRay;
    };

    class Raycaster {

    public:
        float near;
        float far;

        Ray ray;
        Camera* camera;
        Layers layers;

        struct Params {
            float lineThreshold = 1;
            float pointsThreshold = 1;
        };
        Params params;

        explicit Raycaster(const Vector3& origin = Vector3(), const Vector3& direction = Vector3(), float near = 0, float far = std::numeric_limits<float>::infinity())
            : near(near), far(far), ray(origin, direction), camera(nullptr) {}

        void set(const Vector3& origin, const Vector3& direction);

        void setFromCamera(const Vector2& coords, Camera* camera);

        void setFromCamera(const Vector2& coords, const std::shared_ptr<Camera>& camera);

        std::vector<Intersection> intersectObject(Object3D* object, bool recursive = false);

        std::vector<Intersection> intersectObjects(std::vector<std::shared_ptr<Object3D>>& objects, bool recursive = false);
    };

}// namespace threepp

#endif//THREEPP_RAYCASTER_HPP
