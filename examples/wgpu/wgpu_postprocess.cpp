// EffectComposer / ShaderPass demo — applies post-processing chains to a
// rendered scene and writes each result as a PNG next to the executable.

#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/EffectComposer.hpp"
#include "threepp/renderers/wgpu/ShaderPass.hpp"
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

namespace {

    constexpr int WIDTH = 512;
    constexpr int HEIGHT = 512;

    const std::string grayscaleWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    var uv = array<vec2f, 3>(vec2f(0, 1), vec2f(2, 1), vec2f(0, -1));
    var out: VertexOutput;
    out.position = vec4f(pos[idx], 0, 1);
    out.uv = uv[idx];
    return out;
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(inputTex, inputSampler, in.uv);
    let gray = dot(c.rgb, vec3f(0.299, 0.587, 0.114));
    return vec4f(gray, gray, gray, c.a);
}
)";

    const std::string invertWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    var uv = array<vec2f, 3>(vec2f(0, 1), vec2f(2, 1), vec2f(0, -1));
    var out: VertexOutput;
    out.position = vec4f(pos[idx], 0, 1);
    out.uv = uv[idx];
    return out;
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(inputTex, inputSampler, in.uv);
    return vec4f(1.0 - c.r, 1.0 - c.g, 1.0 - c.b, c.a);
}
)";

    // Chroma-shift: samples R/G/B from slightly offset UVs to produce a
    // chromatic-aberration look. Demonstrates a pass that uses neighbour
    // sampling instead of a pure per-pixel color transform.
    const std::string chromaShiftWGSL = R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    var uv = array<vec2f, 3>(vec2f(0, 1), vec2f(2, 1), vec2f(0, -1));
    var out: VertexOutput;
    out.position = vec4f(pos[idx], 0, 1);
    out.uv = uv[idx];
    return out;
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
    let off = vec2f(0.004, 0.0);
    let r = textureSample(inputTex, inputSampler, in.uv - off).r;
    let g = textureSample(inputTex, inputSampler, in.uv).g;
    let b = textureSample(inputTex, inputSampler, in.uv + off).b;
    return vec4f(r, g, b, 1.0);
}
)";

    std::shared_ptr<Scene> makeScene() {
        auto scene = Scene::create();

        scene->add(AmbientLight::create(Color(0x404040)));

        auto dir = DirectionalLight::create(Color(0xffffff), 1.0f);
        dir->position.set(3, 5, 4);
        scene->add(dir);

        auto boxMat = MeshStandardMaterial::create();
        boxMat->color = Color(0xff8844);
        boxMat->roughness = 0.4f;
        auto box = Mesh::create(BoxGeometry::create(1.2f, 1.2f, 1.2f), boxMat);
        box->position.x = -1.4f;
        box->rotation.y = 0.6f;
        scene->add(box);

        auto sphereMat = MeshStandardMaterial::create();
        sphereMat->color = Color(0x44aaff);
        sphereMat->metalness = 0.7f;
        sphereMat->roughness = 0.25f;
        auto sphere = Mesh::create(SphereGeometry::create(0.8f, 32, 16), sphereMat);
        sphere->position.x = 1.2f;
        scene->add(sphere);

        return scene;
    }

}// namespace

int main() {

    // Headless canvas — this example is one-shot and writes PNGs, no window.
    Canvas canvas(Canvas::Parameters().size(WIDTH, HEIGHT).headless(true));

    WgpuRenderer renderer(canvas);
    renderer.setClearColor(Color(0x202030));

    auto scene = makeScene();

    auto camera = PerspectiveCamera::create(60, 1.0f, 0.1f, 100);
    camera->position.set(0, 1.2f, 4);
    camera->lookAt(Vector3{0, 0, 0});

    auto target = RenderTarget::create(WIDTH, HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());

    // Each EffectComposer instance is a single scene + chain of passes.
    // `render()` runs the scene, then runs each pass in insertion order,
    // ping-ponging between internal textures. `readRGBPixels()` /
    // `writeFramebuffer()` return the final output.

    // No passes: composer just forwards the scene texture.
    {
        EffectComposer composer(renderer);
        composer.render(*scene, *camera);
        composer.writeFramebuffer("postprocess_00_baseline.png");
    }

    // Single pass: grayscale.
    {
        EffectComposer composer(renderer);
        composer.addPass(ShaderPass::create(grayscaleWGSL));
        composer.render(*scene, *camera);
        composer.writeFramebuffer("postprocess_01_grayscale.png");
    }

    // Single pass: color invert.
    {
        EffectComposer composer(renderer);
        composer.addPass(ShaderPass::create(invertWGSL));
        composer.render(*scene, *camera);
        composer.writeFramebuffer("postprocess_02_invert.png");
    }

    // Single pass: chroma-shift (neighbour-sampling effect).
    {
        EffectComposer composer(renderer);
        composer.addPass(ShaderPass::create(chromaShiftWGSL));
        composer.render(*scene, *camera);
        composer.writeFramebuffer("postprocess_03_chroma.png");
    }

    // Chained passes: chroma-shift, then grayscale. The second pass reads
    // the output of the first, so the final image is a grayscale version
    // of the chroma-shifted image.
    {
        EffectComposer composer(renderer);
        composer.addPass(ShaderPass::create(chromaShiftWGSL));
        composer.addPass(ShaderPass::create(grayscaleWGSL));
        composer.render(*scene, *camera);
        composer.writeFramebuffer("postprocess_04_chroma_then_gray.png");
    }

    std::cout << "wrote 5 post-processed PNGs (postprocess_*.png)" << std::endl;

    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    return 0;
}
