
#ifndef THREEPP_CAMERAHELPER_HPP
#define THREEPP_CAMERAHELPER_HPP

#include "threepp/cameras/Camera.hpp"
#include "threepp/objects/LineSegments.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace threepp {

    class CameraHelper: public LineSegments {

    public:
        static std::shared_ptr<CameraHelper> create(const std::shared_ptr<Camera>& camera);

        void update();

    protected:
        Camera _camera;
        std::shared_ptr<Camera> camera;
        std::unordered_map<std::string, std::vector<size_t>> pointMap;

        explicit CameraHelper(const std::shared_ptr<Camera>& camera);

        void setPoint(const std::string& point, float x, float y, float z);
    };

}// namespace threepp

#endif//THREEPP_CAMERAHELPER_HPP
