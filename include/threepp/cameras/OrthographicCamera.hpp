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

        void setViewOffset(int fullWidth, int fullHeight, int x, int y, int width, int height) {

            if (this->view) {

                this->view = {
                        true,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1};
            }

            this->view->enabled = true;
            this->view->fullWidth = fullWidth;
            this->view->fullHeight = fullHeight;
            this->view->offsetX = x;
            this->view->offsetY = y;
            this->view->width = width;
            this->view->height = height;

            this->updateProjectionMatrix();
        }

        void clearViewOffset() {

            if (this->view) {

                this->view->enabled = false;
            }

            this->updateProjectionMatrix();
        }

        void updateProjectionMatrix() override {

            const auto dx = (this->right - this->left) / (2 * this->zoom);
            const auto dy = (this->top - this->bottom) / (2 * this->zoom);
            const auto cx = (this->right + this->left) / 2;
            const auto cy = (this->top + this->bottom) / 2;

            int left = cx - dx;
            int right = cx + dx;
            int top = cy + dy;
            int bottom = cy - dy;

            if (this->view && this->view->enabled) {

                const auto scaleW = (this->right - this->left) / this->view->fullWidth / this->zoom;
                const auto scaleH = (this->top - this->bottom) / this->view->fullHeight / this->zoom;

                left += scaleW * this->view->offsetX;
                right = left + scaleW * this->view->width;
                top -= scaleH * this->view->offsetY;
                bottom = top - scaleH * this->view->height;
            }

            this->projectionMatrix.makeOrthographic((float) left, (float) right, (float) top, (float) bottom, this->near, this->far);

            this->projectionMatrixInverse.copy(this->projectionMatrix).invert();
        }

        std::string type() const override {
            return "OrthographicCamera";
        }

        static std::shared_ptr<OrthographicCamera> create(int left = -1, int right = 1, int top = 1, int bottom = -1, float near = 0.1f, float far = 2000) {

            return std::shared_ptr<OrthographicCamera>(new OrthographicCamera(left, right, top, bottom, near, far));
        }

    protected:
        OrthographicCamera(int left, int right, int top, int bottom, float near, float far)
            : Camera(near, far), left(left), right(right), top(top), bottom(bottom) {

            updateProjectionMatrix();
        }
    };

}// namespace threepp

#endif//THREEPP_ORTHOGRAPHICCAMERA_HPP
