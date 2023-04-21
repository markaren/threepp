// https://github.com/mrdoob/three.js/blob/r129/src/cameras/PerspectiveCamera.js

#ifndef THREEPP_PERSPECTIVECAMERA_HPP
#define THREEPP_PERSPECTIVECAMERA_HPP

#include "threepp/cameras/Camera.hpp"

#include <memory>

namespace threepp {

    class PerspectiveCamera: public Camera {

    public:
        float fov;

        float focus = 10;

        float aspect;

        float filmGauge = 35;// width of the film (default in millimeters)
        float filmOffset = 0;// horizontal film offset (same unit as gauge)

        PerspectiveCamera(float fov, float aspect, float near, float far);
        PerspectiveCamera(const PerspectiveCamera&) = delete;

        /**
         * Sets the FOV by focal length in respect to the current .filmGauge.
         *
         * The default film gauge is 35, so that the focal length can be specified for
         * a 35mm (full frame) camera.
         *
         * Values for focal length and film gauge must have the same unit.
         */
        void setFocalLength(float focalLength);

        /**
         * Calculates the focal length from the current .fov and .filmGauge.
         */
        [[nodiscard]] float getFocalLength() const;

        [[nodiscard]] float getEffectiveFOV() const;

        [[nodiscard]] float getFilmWidth() const;

        [[nodiscard]] float getFilmHeight() const;

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
        void setViewOffset(int fullWidth, int fullHeight, int x, int y, int width, int height);

        void clearViewOffset();

        void updateProjectionMatrix() override;

        static std::shared_ptr<PerspectiveCamera> create(float fov = 60, float aspect = 1, float near = 0.1, float far = 2000);
    };

}// namespace threepp

#endif//THREEPP_PERSPECTIVECAMERA_HPP
