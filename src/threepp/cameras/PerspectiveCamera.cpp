
#include "threepp/cameras/PerspectiveCamera.hpp"

#include "threepp/math/MathUtils.hpp"

#include <cmath>

using namespace threepp;


PerspectiveCamera::PerspectiveCamera(float fov, float aspect, float near, float far)
    : Camera(near, far), fov(fov), aspect(aspect) {

    PerspectiveCamera::updateProjectionMatrix();
}

void PerspectiveCamera::setFocalLength(float focalLength) {

    /** see {@link http://www.bobatkins.com/photography/technical/field_of_view->html} */
    const auto vExtentSlope = 0.5f * this->getFilmHeight() / focalLength;

    this->fov = (math::RAD2DEG * 2 * std::atan(vExtentSlope));
    this->updateProjectionMatrix();
}

float PerspectiveCamera::getFocalLength() const {

    const auto vExtentSlope = std::tan(math::DEG2RAD * 0.5f * this->fov);

    return 0.5f * this->getFilmHeight() / vExtentSlope;
}

float PerspectiveCamera::getEffectiveFOV() const {

    return math::RAD2DEG * 2.f * std::atan(std::tan(math::DEG2RAD * 0.5f * this->fov) / this->zoom);
}

float PerspectiveCamera::getFilmWidth() const {

    // film not completely covered in portrait format (aspect < 1)
    return this->filmGauge * std::min(this->aspect, 1.f);
}

float PerspectiveCamera::getFilmHeight() const {

    // film not completely covered in landscape format (aspect > 1)
    return this->filmGauge / std::max(this->aspect, 1.f);
}

void PerspectiveCamera::setViewOffset(int fullWidth, int fullHeight, int x, int y, int width, int height) {

    this->aspect = static_cast<float>(fullWidth) / static_cast<float>(fullHeight);

    if (!this->view) {

        this->view = CameraView{
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

void PerspectiveCamera::clearViewOffset() {

    if (this->view) {

        this->view->enabled = false;
    }

    this->updateProjectionMatrix();
}

void PerspectiveCamera::updateProjectionMatrix() {

    float top = near * std::tan(math::DEG2RAD * 0.5f * this->fov) / this->zoom;
    float height = 2.f * top;
    float width = this->aspect * height;
    float left = -0.5f * width;

    if (this->view && this->view->enabled) {

        const auto fullWidth = static_cast<float>(view->fullWidth),
                   fullHeight = static_cast<float>(view->fullHeight);

        left += static_cast<float>(view->offsetX) * width / fullWidth;
        top -= static_cast<float>(view->offsetY) * height / fullHeight;
        width *= static_cast<float>(view->width) / fullWidth;
        height *= static_cast<float>(view->height) / fullHeight;
    }

    const auto skew = this->filmOffset;
    if (skew != 0) {
        left += (near * skew / this->getFilmWidth());
    }

    this->projectionMatrix.makePerspective(left, (left + width), top, (top - height), near, far);

    this->projectionMatrixInverse.copy(this->projectionMatrix).invert();
}

std::shared_ptr<PerspectiveCamera> PerspectiveCamera::create(float fov, float aspect, float near, float far) {

    return std::make_shared<PerspectiveCamera>(fov, aspect, near, far);
}
