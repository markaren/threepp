
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/textures/DepthTexture.hpp"
#include "threepp/threepp.hpp"

#include <threepp/geometries/TorusKnotGeometry.hpp>

#include <cmath>

using namespace threepp;

namespace {
    void setupScene(Scene& scene) {

        const auto geometry = TorusKnotGeometry::create(1, 0.3, 128, 64);
        const auto material = MeshBasicMaterial::create({{"color", Color::blue}});

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
    GLRenderer renderer(canvas.size());
    renderer.checkShaderErrors = true;

    PerspectiveCamera camera(70, canvas.aspect(), 0.01f, 50.f);
    camera.position.set(0, 0, 4);

    Scene scene;
    setupScene(scene);

    OrbitControls controls(camera, canvas);
    controls.enableDamping = true;


    GLRenderTarget::Options options;
    options.format = Format::RGB;
    options.minFilter = Filter::Nearest;
    options.magFilter = Filter::Nearest;
    options.generateMipmaps = false;
    options.stencilBuffer = false;
    options.depthBuffer = true;

    options.depthTexture = DepthTexture::create();
    options.depthTexture->format = Format::Depth;
    options.depthTexture->type = Type::Float;


    GLRenderTarget target(canvas.size().width(), canvas.size().height(), options);


    auto postMaterial = ShaderMaterial::create();
    postMaterial->vertexShader = R"(
        varying vec2 vUv;

		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
		}
    )";
    postMaterial->fragmentShader = R"(
            #include <packing>

			varying vec2 vUv;
			uniform sampler2D tDiffuse;
			uniform sampler2D tDepth;
			uniform float cameraNear;
			uniform float cameraFar;


			float readDepth( sampler2D depthSampler, vec2 coord ) {
				float fragCoordZ = texture2D( depthSampler, coord ).x;
				float viewZ = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
				return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );
			}

			void main() {
				//vec3 diffuse = texture2D( tDiffuse, vUv ).rgb;
				float depth = readDepth( tDepth, vUv );

				gl_FragColor.rgb = 1 - vec3( depth );
				gl_FragColor.a = 1.0;
			}
        )";

    postMaterial->uniforms = {
            {"tDiffuse", Uniform()},
            {"tDepth", Uniform()},
            {"cameraNear", Uniform(camera.nearPlane)},
            {"cameraFar", Uniform(camera.farPlane)}};

    OrthographicCamera postCamera(-1, 1, 1, -1, 0, 1);
    const auto postPlane = PlaneGeometry::create(2, 2);
    const auto postQuad = Mesh::create(postPlane, postMaterial);
    Scene postScene;
    postScene.add(postQuad);


    canvas.onWindowResize([&](WindowSize size) {
        renderer.setSize(size);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    canvas.animate([&] {
        renderer.setRenderTarget(&target);
        renderer.render(scene, camera);

        postMaterial->uniforms.at("tDiffuse").setValue(target.texture.get());
        postMaterial->uniforms.at("tDepth").setValue(target.depthTexture.get());

        renderer.setRenderTarget(nullptr);

        renderer.render(postScene, postCamera);

        controls.update();
    });
}