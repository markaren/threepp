// Post-processing on the GL backend: a scene rendered through an
// EffectComposer chain instead of straight to the screen.
//
// The chain here is RenderPass -> vignette+aberration -> desaturate. Both
// effects are ordinary ShaderPasses, which is the point: the shader strings are
// plain GLSL with the three.js spelling (varying / texture2D / gl_FragColor),
// and the composer's targets carry MSAA so turning post-processing on does not
// cost the anti-aliasing.
//
//   1 / 2   toggle the two effect passes
//   Space   cycle the desaturation amount

#include "threepp/input/KeyListener.hpp"
#include "threepp/postprocessing/postprocessing.hpp"
#include "threepp/threepp.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace threepp;

namespace {

    // Darkens toward the corners and splits the channels radially — cheap, and
    // it makes both the uniform plumbing and the pass ordering visible.
    Shader vignetteShader() {

        return Shader{
                UniformMap{
                        {"tDiffuse", Uniform()},
                        {"innerRadius", Uniform(0.5f)},
                        {"outerRadius", Uniform(1.f)},
                        {"darkness", Uniform(0.6f)},
                        {"aberration", Uniform(0.004f)}},

                R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                uniform sampler2D tDiffuse;
                uniform float innerRadius;
                uniform float outerRadius;
                uniform float darkness;
                uniform float aberration;

                varying vec2 vUv;

                void main() {
                    vec2 centred = vUv - 0.5;

                    // Sample each channel at a slightly different radius.
                    vec2 shift = centred * aberration;
                    vec4 color = vec4(
                        texture2D( tDiffuse, vUv + shift ).r,
                        texture2D( tDiffuse, vUv ).g,
                        texture2D( tDiffuse, vUv - shift ).b,
                        1.0 );

                    // 0 at the centre, 1 at a corner.
                    float d = length( centred ) * 1.41421356;
                    float vignette = 1.0 - darkness * smoothstep( innerRadius, outerRadius, d );

                    // Half a code value of noise. The vignette is a shallow ramp
                    // across a near-flat background, which 8-bit output bands into
                    // visible rings without it.
                    float dither = fract( sin( dot( vUv, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 ) - 0.5;

                    gl_FragColor = vec4( color.rgb * vignette + dither / 255.0, 1.0 );
                })"};
    }

    Shader desaturateShader() {

        return Shader{
                UniformMap{
                        {"tDiffuse", Uniform()},
                        {"amount", Uniform(0.5f)}},

                R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                uniform sampler2D tDiffuse;
                uniform float amount;

                varying vec2 vUv;

                void main() {
                    vec4 color = texture2D( tDiffuse, vUv );
                    float luma = dot( color.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
                    gl_FragColor = vec4( mix( color.rgb, vec3( luma ), amount ), color.a );
                })"};
    }

    std::shared_ptr<Mesh> createBox(const Vector3& pos, const Color& color) {

        auto material = MeshPhongMaterial::create();
        material->color = color;

        auto box = Mesh::create(BoxGeometry::create(), material);
        box->position.copy(pos);

        return box;
    }

}// namespace


int main(int argc, char** argv) {

    // Headless capture (dev): postprocessing --shot <name.png> [--frames N]
    // [--samples N]. Renders N frames and saves via writeFramebuffer, then
    // exits.
    std::string shotPath;
    int shotFrames = 30, shotFrame = 0;
    unsigned int samples = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--samples" && i + 1 < argc) samples = static_cast<unsigned int>(std::atoi(argv[++i]));
    }

    Canvas canvas("Post-processing", {{"aa", 0}});
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color(0x101018);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0, 2.5f, 6);
    camera->lookAt(Vector3{0, 0, 0});

    OrbitControls controls{*camera, canvas};

    auto light = DirectionalLight::create(0xffffff, 1.f);
    light->position.set(4, 6, 3);
    scene->add(light);
    scene->add(AmbientLight::create(0x606070));

    auto group = Group::create();
    group->add(createBox({-2, 0, 0}, Color::green));
    group->add(createBox({0, 0, 0}, Color::red));
    group->add(createBox({2, 0, 0}, Color::blue));
    scene->add(group);

    auto floorMaterial = MeshPhongMaterial::create();
    floorMaterial->color = Color::gray;
    auto floor = Mesh::create(PlaneGeometry::create(20, 20), floorMaterial);
    floor->rotateX(-math::PI / 2);
    floor->position.y = -1;
    scene->add(floor);

    // MSAA lives on the composer's internal targets, so the chain is fed an
    // anti-aliased image rather than resolving one at the end.
    EffectComposer::Options options;
    options.samples = samples;

    EffectComposer composer(renderer, options);

    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));

    auto vignettePass = std::make_shared<ShaderPass>(vignetteShader());
    composer.addPass(vignettePass);

    auto desaturatePass = std::make_shared<ShaderPass>(desaturateShader());
    composer.addPass(desaturatePass);

    float desaturation = 0.35f;

    canvas.onKeyPressed([&](KeyEvent evt) {
        if (evt.key == Key::NUM_1) {
            vignettePass->enabled = !vignettePass->enabled;
            std::cout << "vignette: " << (vignettePass->enabled ? "on" : "off") << "\n";
        } else if (evt.key == Key::NUM_2) {
            desaturatePass->enabled = !desaturatePass->enabled;
            std::cout << "desaturate: " << (desaturatePass->enabled ? "on" : "off") << "\n";
        } else if (evt.key == Key::SPACE) {
            desaturation = std::fmod(desaturation + 0.25f, 1.25f);
            std::cout << "desaturation: " << desaturation << "\n";
        }
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        // The composer owns its own targets; they do not follow the renderer.
        composer.setSize(size.width(), size.height());
    });

    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();

        group->rotation.y += 0.6f * dt;
        for (auto& child : group->children) {
            child->rotation.x += 1.1f * dt;
        }

        desaturatePass->uniforms().at("amount").setValue(desaturation);

        composer.render(dt);

        if (!shotPath.empty() && ++shotFrame >= shotFrames) {
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
