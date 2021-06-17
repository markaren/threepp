// https://github.com/mrdoob/three.js/blob/r129/src/cameras/PerspectiveCamera.js

#ifndef THREEPP_PERSPECTIVECAMERA_HPP
#define THREEPP_PERSPECTIVECAMERA_HPP

#include "threepp/cameras/Camera.hpp"

#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>

namespace threepp {

    class PerspectiveCamera : public Camera {

    public:
        std::string type = "PerspectiveCamera";

        float fov;
        float zoom = 1;

        float near;
        float far;
        float focus = 10;

        float aspect;

        float filmGauge = 35;// width of the film (default in millimeters)
        float filmOffset = 0;// horizontal film offset (same unit as gauge)


        /**
         * Sets the FOV by focal length in respect to the current .filmGauge.
         *
         * The default film gauge is 35, so that the focal length can be specified for
         * a 35mm (full frame) camera.
         *
         * Values for focal length and film gauge must have the same unit.
         */
        void setFocalLength(float focalLength) {

            /** see {@link http://www.bobatkins.com/photography/technical/field_of_view.html} */
            const auto vExtentSlope = 0.5f * this->getFilmHeight() / focalLength;

            this->fov = RAD2DEG * 2 * std::atan(vExtentSlope);
            this->updateProjectionMatrix();
        }

        /**
         * Calculates the focal length from the current .fov and .filmGauge.
         */
        [[nodiscard]] float getFocalLength() const {

            const auto vExtentSlope = std::tan(DEG2RAD * 0.5f * this->fov);

            return 0.5f * this->getFilmHeight() / vExtentSlope;
        }

        [[nodiscard]] float getEffectiveFOV() const {

            return RAD2DEG * 2 * std::atan(std::tan(DEG2RAD * 0.5f * this->fov) / this->zoom);
        }

        [[nodiscard]] float getFilmWidth() const {

            // film not completely covered in portrait format (aspect < 1)
            return this->filmGauge * std::min(this->aspect, 1.f);
        }

        [[nodiscard]] float getFilmHeight() const {

            // film not completely covered in landscape format (aspect > 1)
            return this->filmGauge / std::max(this->aspect, 1.f);
        }

        void updateProjectionMatrix() {

            const auto top = near * std::tan(DEG2RAD * 0.5f * this->fov) / this->zoom;
            const auto height = 2 * top;
            const auto width = this->aspect * height;
            auto left = -0.5f * width;
            //            const view = this.view;
            //
            //            if ( this.view !== null && this.view.enabled ) {
            //
            //                const fullWidth = view.fullWidth,
            //                        fullHeight = view.fullHeight;
            //
            //                left += view.offsetX * width / fullWidth;
            //                top -= view.offsetY * height / fullHeight;
            //                width *= view.width / fullWidth;
            //                height *= view.height / fullHeight;
            //
            //            }

            const auto skew = this->filmOffset;
            if (skew != 0) left += near * skew / this->getFilmWidth();

            this->projectionMatrix.makePerspective(left, left + width, top, top - height, near, far);

            this->projectionMatrixInverse.copy(this->projectionMatrix).invert();
        }

        PerspectiveCamera(const PerspectiveCamera&) = delete;

        static std::shared_ptr<PerspectiveCamera> create(float fov, float aspect = 1, float near = 0.1, float far = 2000) {
            return std::shared_ptr<PerspectiveCamera>(new PerspectiveCamera(fov, aspect, near, far));
        }

    protected:
        explicit PerspectiveCamera(float fov, float aspect, float near, float far) : fov(fov), aspect(aspect), near(near), far(far) {}

    };

}// namespace threepp

#endif//THREEPP_PERSPECTIVECAMERA_HPP
