#ifndef THREEPP_LIDAR_HPP
#define THREEPP_LIDAR_HPP

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"


#include <vector>

namespace threepp {

    class ShaderMaterial;
    class GLRenderer;

    /**
     * Simulates a depth sensor using GPU depth rendering.
     *
     * Renders the scene from the sensor's viewpoint into a depth texture,
     * linearizes the result via a post-process shader, reads back the pixels,
     * and reprojects them into world-space 3D points.
     */
    class DepthSensor: public Object3D {

    public:
        // Gaussian range noise standard deviation in metres (0 = perfect sensor)
        float rangeNoise{0.02f};

        DepthSensor(float fovY, unsigned int width, unsigned int height,
                    float near = 0.1f, float far = 100.f);

        /**
         * Performs a scan and returns the hit points in world space.
         *
         * The Lidar object must be in the scene (or have its parent chain
         * updated) before calling this, so its world matrix is current.
         *
         * The renderer's active render target is restored to nullptr after the scan.
         */
        void scan(GLRenderer& renderer, Scene& scene, std::vector<Vector3>& cloud);

        [[nodiscard]] unsigned int width() const { return width_; }
        [[nodiscard]] unsigned int height() const { return height_; }
        [[nodiscard]] float fov() const { return camera_.fov; }
        [[nodiscard]] float near() const { return camera_.nearPlane; }
        [[nodiscard]] float far() const { return camera_.farPlane; }


    private:
        unsigned int width_;
        unsigned int height_;

        Scene postScene_;
        OrthographicCamera postCamera_;
        PerspectiveCamera camera_;
        std::unique_ptr<GLRenderTarget> sceneTarget_;
        std::unique_ptr<GLRenderTarget> readbackTarget_;
        std::shared_ptr<ShaderMaterial> postMaterial_;

        // Precomputed per-column and per-row view-space ray direction factors
        std::vector<float> xDir_;
        std::vector<float> yDir_;

        void unprojectPoints(std::vector<Vector3>& points) const;
    };

}// namespace threepp

#endif//THREEPP_LIDAR_HPP
