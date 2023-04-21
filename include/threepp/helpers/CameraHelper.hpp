
#ifndef THREEPP_CAMERAHELPER_HPP
#define THREEPP_CAMERAHELPER_HPP

#include "threepp/objects/LineSegments.hpp"

#include <memory>

namespace threepp {

    class Camera;

    class CameraHelper: public LineSegments {

    public:
        static std::shared_ptr<CameraHelper> create(Camera& camera);

        void update();

        ~CameraHelper() override;

    private:
        struct Impl;
        std::shared_ptr<Impl> pimpl_;

        explicit CameraHelper(Camera& camera);
    };

}// namespace threepp

#endif//THREEPP_CAMERAHELPER_HPP
