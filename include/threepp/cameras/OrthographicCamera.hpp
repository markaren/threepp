// https://github.com/mrdoob/three.js/blob/r129/src/cameras/OrthographicCamera.js

#ifndef THREEPP_ORTHOGRAPHICCAMERA_HPP
#define THREEPP_ORTHOGRAPHICCAMERA_HPP

#include "threepp/cameras/Camera.hpp"

namespace threepp {

    class OrthographicCamera : public Camera {

    public:

        int left;
        int right;
        int top;
        int bottom;

        void setViewOffset(int fullWidth, int fullHeight, int x, int y, int width, int height);

        void clearViewOffset();

        void updateProjectionMatrix() override;

        static std::shared_ptr<OrthographicCamera> create(int left = -1, int right = 1, int top = 1, int bottom = -1, float near = 0.1f, float far = 2000) {

            return std::shared_ptr<OrthographicCamera>(new OrthographicCamera(left, right, top, bottom, near, far));
        }

    protected:
        OrthographicCamera(int left, int right, int top, int bottom, float near, float far);
    };

}// namespace threepp

#endif//THREEPP_ORTHOGRAPHICCAMERA_HPP
