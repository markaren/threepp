
#include "threepp/cameras/OrthographicCamera.hpp"

using namespace threepp;

OrthographicCamera::OrthographicCamera(int left, int right, int top, int bottom, float near, float far)
    : Camera(near, far), left(left), right(right), top(top), bottom(bottom) {

    OrthographicCamera::updateProjectionMatrix();
}

void OrthographicCamera::setViewOffset(int fullWidth, int fullHeight, int x, int y, int width, int height) {

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

void OrthographicCamera::clearViewOffset() {

    if (this->view) {

        this->view->enabled = false;
    }

    this->updateProjectionMatrix();
}

void OrthographicCamera::updateProjectionMatrix() {

    const auto dx = static_cast<float>(this->right - this->left) / (2 * this->zoom);
    const auto dy = static_cast<float>(this->top - this->bottom) / (2 * this->zoom);
    const auto cx = static_cast<float>(this->right + this->left) / 2;
    const auto cy = static_cast<float>(this->top + this->bottom) / 2;

    float left = cx - dx;
    float right = cx + dx;
    float top = cy + dy;
    float bottom = cy - dy;

    if (this->view && this->view->enabled) {

        const auto scaleW = static_cast<float>(this->right - this->left) /static_cast<float>(this->view->fullWidth) / this->zoom;
        const auto scaleH = static_cast<float>(this->top - this->bottom) / static_cast<float>(this->view->fullHeight) / this->zoom;

        left += scaleW * static_cast<float>(this->view->offsetX);
        right = left + scaleW * static_cast<float>(this->view->width);
        top -= scaleH * static_cast<float>(this->view->offsetY);
        bottom = top - scaleH * static_cast<float>(this->view->height);
    }

    this->projectionMatrix.makeOrthographic((float) left, (float) right, (float) top, (float) bottom, this->near, this->far);

    this->projectionMatrixInverse.copy(this->projectionMatrix).invert();
}

std::shared_ptr<OrthographicCamera> OrthographicCamera::create(int left, int right, int top, int bottom, float near, float far) {

    return std::make_shared<OrthographicCamera>(left, right, top, bottom, near, far);
}
