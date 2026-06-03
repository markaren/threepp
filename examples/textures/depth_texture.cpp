
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/DepthTexture.hpp"
#include "threepp/threepp.hpp"

#include <threepp/geometries/TorusKnotGeometry.hpp>

#include <cmath>

using namespace threepp;

namespace {
    void setupScene(Scene& scene) {

        const auto geometry = TorusKnotGeometry::create(1, 0.3, 128, 64);
        const auto material = MeshBasicMaterial::create(MeshBasicMaterial::Params{}.color(Color::blue));

        const auto count = 50;
        const auto scale = 5;

        for (auto i = 0; i < count; i++) {

            const auto r = math::randFloat() * 2.0f * math::PI;
            const auto z = (math::randFloat() * 2.0f) - 1.0f;
            const auto zScale = std::sqrt(1.0f - z * z) * scale;

            const auto mesh = Mesh::create(geometry, material);
            mesh->position.set(
                    std::cos(r) * zScale,
                    std::sin(r) * zScale,
                    z * scale);
            mesh->rotation.set(math::randFloat(), math::randFloat(), math::randFloat());
            scene.add(mesh);
        }
    }
}// namespace

int main() {

    Canvas canvas("Depth texture");
    auto renderer = createRenderer(canvas);

    PerspectiveCamera camera(70, canvas.aspect(), 0.01f, 50.f);
    camera.position.set(0, 0, 4);

    Scene scene;
    setupScene(scene);

    OrbitControls controls(camera, canvas);
    controls.enableDamping = true;


    RenderTarget::Options options;
    options.format = Format::RGB;
    options.minFilter = Filter::Nearest;
    options.magFilter = Filter::Nearest;
    options.generateMipmaps = false;
    options.stencilBuffer = false;
    options.depthBuffer = true;

    options.depthTexture = DepthTexture::create();
    options.depthTexture->format = Format::Depth;
    options.depthTexture->type = Type::Float;


    RenderTarget target(canvas.size().width(), canvas.size().height(), options);


    auto postMaterial = ShaderMaterial::create();
    // flipUv compensates for WebGPU sampling render targets with a top-left UV
    // origin (OpenGL uses bottom-left). Driven by renderer->renderTargetFlipY().
    postMaterial->vertexShader = R"(
        varying vec2 vUv;
        uniform float flipUv;
        void main() {
            vUv = vec2(uv.x, mix(uv.y, 1.0 - uv.y, flipUv));
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    )";
    // The packing helpers are inlined (rather than #include <packing>) and the
    // depth is sampled at point of use — both required for the WebGPU GLSL→SPIR-V
    // path, which can't pass a combined sampler2D into a function.
    postMaterial->fragmentShader = R"(
        varying vec2 vUv;
        uniform sampler2D tDepth;
        uniform float cameraNear;
        uniform float cameraFar;

        float perspectiveDepthToViewZ(float invClipZ, float near, float far) {
            return (near * far) / ((far - near) * invClipZ - far);
        }
        float viewZToOrthographicDepth(float viewZ, float near, float far) {
            return (viewZ + near) / (near - far);
        }

        void main() {
            float fragCoordZ = texture2D(tDepth, vUv).x;
            float viewZ = perspectiveDepthToViewZ(fragCoordZ, cameraNear, cameraFar);
            float depth = viewZToOrthographicDepth(viewZ, cameraNear, cameraFar);

            gl_FragColor.rgb = 1.0 - vec3(depth);
            gl_FragColor.a = 1.0;
        }
    )";

    postMaterial->uniforms = {
            {"tDepth", Uniform()},
            {"cameraNear", Uniform(camera.nearPlane)},
            {"cameraFar", Uniform(camera.farPlane)},
            {"flipUv", Uniform(renderer->renderTargetFlipY() ? 1.0f : 0.0f)}};

    OrthographicCamera postCamera(-1, 1, 1, -1, 0, 1);
    const auto postPlane = PlaneGeometry::create(2, 2);
    const auto postQuad = Mesh::create(postPlane, postMaterial);
    Scene postScene;
    postScene.add(postQuad);


    canvas.onWindowResize([&](WindowSize size) {
        renderer->setSize(size);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    canvas.animate([&] {
        renderer->setRenderTarget(&target);
        renderer->render(scene, camera);

        postMaterial->uniforms.at("tDepth").setValue(static_cast<Texture*>(target.depthTexture.get()));

        renderer->setRenderTarget(nullptr);

        renderer->render(postScene, postCamera);

        controls.update();
    });
}
