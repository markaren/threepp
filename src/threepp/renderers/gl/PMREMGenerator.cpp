
#include "threepp/renderers/gl/PMREMGenerator.hpp"

#include "GLPMREM.hpp"

#include "threepp/cameras/CubeCamera.hpp"
#include "threepp/renderers/GLCubeRenderTarget.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>

using namespace threepp;

struct PMREMGenerator::Impl {
    gl::GLPMREM pmrem;

    explicit Impl(GLRenderer& renderer): pmrem(renderer) {}
};

PMREMGenerator::PMREMGenerator(GLRenderer& renderer)
    : renderer_(renderer), impl_(std::make_unique<Impl>(renderer)) {}

PMREMGenerator::~PMREMGenerator() = default;

std::shared_ptr<Texture> PMREMGenerator::fromScene(Scene& scene, float sigma, float near, float far, const Options& options) {

    const int size = std::max(options.size, 16);

    // 1. The scene into a cube. HalfFloat and linear: the emissive panels of a
    //    RoomEnvironment are far above 1, and that range is the whole point.
    GLRenderTarget::Options cubeOptions;
    cubeOptions.type = Type::HalfFloat;
    cubeOptions.format = Format::RGBA;
    cubeOptions.encoding = ColorSpace::Linear;
    cubeOptions.magFilter = Filter::Linear;
    cubeOptions.minFilter = Filter::Linear;
    cubeOptions.generateMipmaps = false;
    GLCubeRenderTarget cubeTarget(size, cubeOptions);

    CubeCamera cubeCamera(near, far, cubeTarget);
    cubeCamera.position.copy(options.position);
    cubeCamera.updateMatrixWorld();

    auto* oldTarget = renderer_.getRenderTarget();
    const bool oldAutoClear = renderer_.autoClear;

    // As three.js _sceneToCubeUV: every face cleared (to the renderer's clear
    // colour when the scene has no background). No tone mapping to switch off:
    // the renderer never tone-maps into a render target (three.js r154+).
    renderer_.autoClear = true;

    cubeCamera.update(renderer_, scene);

    renderer_.autoClear = oldAutoClear;
    renderer_.setRenderTarget(oldTarget);

    // 2. Cube -> equirect, with the optional blur (a rendered cube: no flip).
    std::shared_ptr<RenderTarget> target = impl_->pmrem.cubeToEquirect(*cubeTarget.texture, size, sigma, false);
    target->texture->name = scene.name.empty() ? "PMREMGenerator.fromScene" : scene.name;

    // The texture keeps its render target (and so its GL storage) alive: an
    // aliasing pointer, so the caller holds a plain Texture.
    auto* texture = target->texture.get();
    return {std::move(target), texture};
}
