
#include <utility>

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Reflector.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;

namespace {

    Shader reflectorShader() {

        static Shader reflectorShader{

                UniformMap{
                        {"color", Uniform()},
                        {"tDiffuse", Uniform()},
                        {"textureMatrix", Uniform()}},

                R"(
                uniform mat4 textureMatrix;
                varying vec4 vUv;

                void main() {
                    vUv = textureMatrix * vec4( position, 1.0 );
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                uniform vec3 color;
                uniform sampler2D tDiffuse;
                varying vec4 vUv;

                float blendOverlay( float base, float blend ) {
                    return( base < 0.5 ? ( 2.0 * base * blend ) : ( 1.0 - 2.0 * ( 1.0 - base ) * ( 1.0 - blend ) ) );
                }

                vec3 blendOverlay( vec3 base, vec3 blend ) {
                    return vec3( blendOverlay( base.r, blend.r ), blendOverlay( base.g, blend.g ), blendOverlay( base.b, blend.b ) );
                }

                void main() {
                    vec4 base = texture2DProj( tDiffuse, vUv );
                    gl_FragColor = vec4( blendOverlay( base.rgb, color ), 1.0 );
                })"

        };

        return reflectorShader;
    }

}// namespace

struct Reflector::Impl {

    Impl(Reflector& reflector, Reflector::Options options)
        : reflector_(reflector), clipBias(options.clipBias.value_or(0.f)) {

        Color color{options.color.value_or(0x7f7f7f)};
        unsigned int textureWidth = (options.textureWidth) ? *options.textureWidth : 512;
        unsigned int textureHeight = (options.textureHeight) ? *options.textureHeight : 512;
        Shader shader = options.shader.value_or(reflectorShader());

        GLRenderTarget::Options parameters;
        parameters.minFilter = LinearFilter;
        parameters.magFilter = LinearFilter;
        parameters.format = RGBAFormat;

        renderTarget = GLRenderTarget::create(textureWidth, textureHeight, parameters);

        if (!math::isPowerOfTwo((int) textureWidth) || !math::isPowerOfTwo((int) textureHeight)) {

            renderTarget->texture->generateMipmaps = false;
        }

        auto material = ShaderMaterial::create();
        material->uniforms = std::make_shared<UniformMap>(shader.uniforms);
        material->fragmentShader = shader.fragmentShader;
        material->vertexShader = shader.vertexShader;

        (*material->uniforms)["tDiffuse"].setValue(renderTarget->texture.get());
        (*material->uniforms)["color"].setValue(color);
        (*material->uniforms)["textureMatrix"].setValue(&textureMatrix);

        reflector.onBeforeRender = RenderCallback([this, material](void* renderer, auto scene, auto camera, auto, auto, auto) {
            reflectorWorldPosition.setFromMatrixPosition(*reflector_.matrixWorld);
            cameraWorldPosition.setFromMatrixPosition(*camera->matrixWorld);
            rotationMatrix.extractRotation(*reflector_.matrixWorld);
            normal.set(0, 0, 1);
            normal.applyMatrix4(rotationMatrix);
            view.subVectors(reflectorWorldPosition, cameraWorldPosition);// Avoid rendering when reflector is facing away

            if (view.dot(normal) > 0) return;
            view.reflect(normal).negate();
            view.add(reflectorWorldPosition);
            rotationMatrix.extractRotation(*camera->matrixWorld);
            lookAtPosition.set(0, 0, -1);
            lookAtPosition.applyMatrix4(rotationMatrix);
            lookAtPosition.add(cameraWorldPosition);
            target.subVectors(reflectorWorldPosition, lookAtPosition);
            target.reflect(normal).negate();
            target.add(reflectorWorldPosition);
            virtualCamera->position.copy(view);
            virtualCamera->up.set(0, 1, 0);
            virtualCamera->up.applyMatrix4(rotationMatrix);
            virtualCamera->up.reflect(normal);
            virtualCamera->lookAt(target);
            virtualCamera->far = camera->far;// Used in WebGLBackground

            virtualCamera->updateMatrixWorld();
            virtualCamera->projectionMatrix.copy(camera->projectionMatrix);// Update the texture matrix

            textureMatrix.set(0.5f, 0.f, 0.f, 0.5f,
                              0.f, 0.5f, 0.f, 0.5f,
                              0.f, 0.f, 0.5f, 0.5f,
                              0.f, 0.f, 0.f, 1.f);
            textureMatrix.multiply(virtualCamera->projectionMatrix);
            textureMatrix.multiply(virtualCamera->matrixWorldInverse);
            textureMatrix.multiply(*reflector_.matrixWorld);// Now update projection matrix with new clip plane, implementing code from: http://www.terathon.com/code/oblique.html
            // Paper explaining this technique: http://www.terathon.com/lengyel/Lengyel-Oblique.pdf

            reflectorPlane.setFromNormalAndCoplanarPoint(normal, reflectorWorldPosition);
            reflectorPlane.applyMatrix4(virtualCamera->matrixWorldInverse);
            clipPlane.set(reflectorPlane.normal.x, reflectorPlane.normal.y, reflectorPlane.normal.z, reflectorPlane.constant);
            auto& projectionMatrix = virtualCamera->projectionMatrix;
            q.x = (static_cast<float>(math::sgn(clipPlane.x)) + projectionMatrix.elements[8]) / projectionMatrix.elements[0];
            q.y = (static_cast<float>(math::sgn(clipPlane.y)) + projectionMatrix.elements[9]) / projectionMatrix.elements[5];
            q.z = -1.f;
            q.w = (1.f + projectionMatrix.elements[10]) / projectionMatrix.elements[14];// Calculate the scaled plane vector

            clipPlane.multiplyScalar(2.f / clipPlane.dot(q));// Replacing the third row of the projection matrix

            projectionMatrix.elements[2] = clipPlane.x;
            projectionMatrix.elements[6] = clipPlane.y;
            projectionMatrix.elements[10] = clipPlane.z + 1.f - clipBias;
            projectionMatrix.elements[14] = clipPlane.w;// Render

            auto _renderer = static_cast<GLRenderer*>(renderer);

            renderTarget->texture->encoding = _renderer->outputEncoding;
            reflector_.visible = false;
            const auto currentRenderTarget = _renderer->getRenderTarget();
            const auto currentShadowAutoUpdate = _renderer->shadowMap().autoUpdate;

            _renderer->shadowMap().autoUpdate = false;// Avoid re-computing shadows

            _renderer->setRenderTarget(renderTarget);
            _renderer->state().depthBuffer.setMask(true);// make sure the depth buffer is writable so it can be properly cleared, see #18897

            if (!_renderer->autoClear) _renderer->clear();
            _renderer->render(scene, virtualCamera.get());
            _renderer->shadowMap().autoUpdate = currentShadowAutoUpdate;
            _renderer->setRenderTarget(currentRenderTarget);// Restore viewport

            reflector_.visible = true;
        });

        reflector.materials_[0] = material;
    }

    ~Impl() = default;

private:
    Reflector& reflector_;

    float clipBias;

    Plane reflectorPlane;
    Vector3 normal;
    Vector3 reflectorWorldPosition;
    Vector3 cameraWorldPosition;
    Matrix4 rotationMatrix;
    Vector3 lookAtPosition{0, 1, 0};
    Vector4 clipPlane;
    Vector3 view;
    Vector3 target;
    Vector4 q;
    Matrix4 textureMatrix;

    std::shared_ptr<PerspectiveCamera> virtualCamera = PerspectiveCamera::create();
    std::shared_ptr<GLRenderTarget> renderTarget;
};

Reflector::Reflector(const std::shared_ptr<BufferGeometry>& geometry, Reflector::Options options)
    : Mesh(geometry, nullptr), pimpl_(std::make_unique<Impl>(*this, std::move(options))) {}


std::string threepp::Reflector::type() const {

    return "Reflector";
}

std::shared_ptr<Reflector> Reflector::create(const std::shared_ptr<BufferGeometry>& geometry, Reflector::Options options) {

    return std::shared_ptr<Reflector>(new Reflector(geometry, std::move(options)));
}

Reflector::~Reflector() = default;
