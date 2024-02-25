// https://github.com/mrdoob/three.js/blob/r129/src/cameras/CubeCamera.js

#ifndef THREEPP_CUBECAMERA_HPP
#define THREEPP_CUBECAMERA_HPP

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/renderers/GLRenderer.hpp"

namespace threepp {

    // Creates 6 cameras that render to a WebGLCubeRenderTarget.
    class CubeCamera: public Object3D {

    public:
        CubeCamera(float near, float far, GLRenderTarget& renderTarget)
            : renderTarget(&renderTarget) {

            const auto fov = 90, aspect = 1;

            const auto cameraPX = PerspectiveCamera::create(fov, aspect, near, far);
            cameraPX->layers = this->layers;
            cameraPX->up.set(0, -1, 0);
            cameraPX->lookAt(Vector3(1, 0, 0));
            this->add(cameraPX);

            const auto cameraNX = PerspectiveCamera::create(fov, aspect, near, far);
            cameraNX->layers = this->layers;
            cameraNX->up.set(0, -1, 0);
            cameraNX->lookAt(Vector3(-1, 0, 0));
            this->add(cameraNX);

            const auto cameraPY = PerspectiveCamera::create(fov, aspect, near, far);
            cameraPY->layers = this->layers;
            cameraPY->up.set(0, 0, 1);
            cameraPY->lookAt(Vector3(0, 1, 0));
            this->add(cameraPY);

            const auto cameraNY = PerspectiveCamera::create(fov, aspect, near, far);
            cameraNY->layers = this->layers;
            cameraNY->up.set(0, 0, -1);
            cameraNY->lookAt(Vector3(0, -1, 0));
            this->add(cameraNY);

            const auto cameraPZ = PerspectiveCamera::create(fov, aspect, near, far);
            cameraPZ->layers = this->layers;
            cameraPZ->up.set(0, -1, 0);
            cameraPZ->lookAt(Vector3(0, 0, 1));
            this->add(cameraPZ);

            const auto cameraNZ = PerspectiveCamera::create(fov, aspect, near, far);
            cameraNZ->layers = this->layers;
            cameraNZ->up.set(0, -1, 0);
            cameraNZ->lookAt(Vector3(0, 0, -1));
            this->add(cameraNZ);
        }

        [[nodiscard]] std::string type() const override {

            return "CubeCamera";
        }

        void update(GLRenderer& renderer, Object3D& scene) {

            if (!this->parent) this->updateMatrixWorld();

            auto cameraPX = this->children[0]->as<PerspectiveCamera>();
            auto cameraNX = this->children[1]->as<PerspectiveCamera>();
            auto cameraPY = this->children[2]->as<PerspectiveCamera>();
            auto cameraNY = this->children[3]->as<PerspectiveCamera>();
            auto cameraPZ = this->children[4]->as<PerspectiveCamera>();
            auto cameraNZ = this->children[5]->as<PerspectiveCamera>();

            const auto& currentRenderTarget = renderer.getRenderTarget();

            const auto generateMipmaps = renderTarget->texture->generateMipmaps;

            renderTarget->texture->generateMipmaps = false;

            renderer.setRenderTarget(renderTarget, 0);
            renderer.render(scene, *cameraPX);

            renderer.setRenderTarget(renderTarget, 1);
            renderer.render(scene, *cameraNX);

            renderer.setRenderTarget(renderTarget, 2);
            renderer.render(scene, *cameraPY);

            renderer.setRenderTarget(renderTarget, 3);
            renderer.render(scene, *cameraNY);

            renderer.setRenderTarget(renderTarget, 4);
            renderer.render(scene, *cameraPZ);

            renderTarget->texture->generateMipmaps = generateMipmaps;

            renderer.setRenderTarget(renderTarget, 5);
            renderer.render(scene, *cameraNZ);

            renderer.setRenderTarget(currentRenderTarget);
        }

        static std::shared_ptr<CubeCamera> create(float near, float far, GLRenderTarget& renderTarget) {

            return std::make_shared<CubeCamera>(near, far, renderTarget);
        }

    private:
        GLRenderTarget* renderTarget;
    };

}// namespace threepp

#endif//THREEPP_CUBECAMERA_HPP
