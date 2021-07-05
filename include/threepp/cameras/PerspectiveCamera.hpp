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
        int fov;

        float focus = 10;

        int aspect;

        float filmGauge = 35;// width of the film (default in millimeters)
        float filmOffset = 0;// horizontal film offset (same unit as gauge)

        PerspectiveCamera(const PerspectiveCamera &) = delete;

        /**
         * Sets the FOV by focal length in respect to the current .filmGauge.
         *
         * The default film gauge is 35, so that the focal length can be specified for
         * a 35mm (full frame) camera.
         *
         * Values for focal length and film gauge must have the same unit.
         */
        void setFocalLength(float focalLength) {

            /** see {@link http://www.bobatkins.com/photography/technical/field_of_view->html} */
            const auto vExtentSlope = 0.5f * this->getFilmHeight() / focalLength;

            this->fov = (int) (math::RAD2DEG * 2 * std::atan(vExtentSlope));
            this->updateProjectionMatrix();
        }

        /**
         * Calculates the focal length from the current .fov and .filmGauge.
         */
        [[nodiscard]] float getFocalLength() const {

            const auto vExtentSlope = std::tan(math::DEG2RAD * 0.5f * (float) this->fov);

            return 0.5f * this->getFilmHeight() / vExtentSlope;
        }

        [[nodiscard]] float getEffectiveFOV() const {

            return math::RAD2DEG * 2.f * std::atan(std::tan(math::DEG2RAD * 0.5f * (float) this->fov) / (float) this->zoom);
        }

        [[nodiscard]] float getFilmWidth() const {

            // film not completely covered in portrait format (aspect < 1)
            return this->filmGauge * std::min((float) this->aspect, 1.f);
        }

        [[nodiscard]] float getFilmHeight() const {

            // film not completely covered in landscape format (aspect > 1)
            return this->filmGauge / std::max((float) this->aspect, 1.f);
        }

        /**
         * Sets an offset in a larger frustum. This is useful for multi-window or
         * multi-monitor/multi-machine setups.
         *
         * For example, if you have 3x2 monitors and each monitor is 1920x1080 and
         * the monitors are in grid like this
         *
         *   +---+---+---+
         *   | A | B | C |
         *   +---+---+---+
         *   | D | E | F |
         *   +---+---+---+
         *
         * then for each monitor you would call it like this
         *
         *   const w = 1920;
         *   const h = 1080;
         *   const fullWidth = w * 3;
         *   const fullHeight = h * 2;
         *
         *   --A--
         *   camera.setViewOffset( fullWidth, fullHeight, w * 0, h * 0, w, h );
         *   --B--
         *   camera.setViewOffset( fullWidth, fullHeight, w * 1, h * 0, w, h );
         *   --C--
         *   camera.setViewOffset( fullWidth, fullHeight, w * 2, h * 0, w, h );
         *   --D--
         *   camera.setViewOffset( fullWidth, fullHeight, w * 0, h * 1, w, h );
         *   --E--
         *   camera.setViewOffset( fullWidth, fullHeight, w * 1, h * 1, w, h );
         *   --F--
         *   camera.setViewOffset( fullWidth, fullHeight, w * 2, h * 1, w, h );
         *
         *   Note there is no reason monitors have to be the same size or in a grid.
         */
        void setViewOffset(int fullWidth, int fullHeight, int x, int y, int width, int height) {

            this->aspect = fullWidth / fullHeight;

            if (!this->view) {

                this->view = View{
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

            int top = (int) (near * std::tan(math::DEG2RAD * 0.5f * (float) this->fov) / (float) this->zoom);
            int height = (int) 2 * top;
            int width = this->aspect * height;
            int left = (int) -0.5 * width;

            if (this->view && this->view->enabled) {

                const auto fullWidth = view->fullWidth,
                           fullHeight = view->fullHeight;

                left += view->offsetX * width / fullWidth;
                top -= view->offsetY * height / fullHeight;
                width *= view->width / fullWidth;
                height *= view->height / fullHeight;
            }

            const auto skew = this->filmOffset;
            if (skew != 0) left += (int) (near * skew / this->getFilmWidth());

            this->projectionMatrix.makePerspective((float) left, (float) (left + width), (float) top, (float) (top - height), near, far);

            this->projectionMatrixInverse.copy(this->projectionMatrix).invert();
        }

        [[nodiscard]] std::string type() const override {
            return "PerspectiveCamera";
        }

        static std::shared_ptr<PerspectiveCamera> create(int fov, int aspect = 1, float near = 0.1, float far = 2000) {
            return std::shared_ptr<PerspectiveCamera>(new PerspectiveCamera(fov, aspect, near, far));
        }

    protected:
        explicit PerspectiveCamera(int fov, int aspect, float near, float far)
            : Camera(near, far), fov(fov), aspect(aspect) {}
    };

}// namespace threepp

#endif//THREEPP_PERSPECTIVECAMERA_HPP
