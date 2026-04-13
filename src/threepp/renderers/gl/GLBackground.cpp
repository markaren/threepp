
#include "threepp/renderers/gl/GLBackground.hpp"
#include "threepp/renderers/gl/GLCubeMaps.hpp"
#include "threepp/renderers/gl/GLObjects.hpp"
#include "threepp/renderers/gl/GLRenderLists.hpp"

#include "threepp/renderers/shaders/ShaderLib.hpp"

using namespace threepp;
using namespace threepp::gl;

GLBackground::GLBackground(GLRenderer& renderer, GLCubeMaps& cubemaps, GLState& state, GLObjects& objects, bool premultipliedAlpha)
    : renderer(renderer), cubemaps(cubemaps), state(state), objects(objects), premultipliedAlpha(premultipliedAlpha) {}

void GLBackground::render(GLRenderList& renderList, Object3D* scene) {

    bool forceClear = false;
    const bool isScene = scene->is<Scene>();

    std::optional<Background> background;

    if (isScene) {
        background = scene->as<Scene>()->background;
    }

    // Resolve equirectangular textures to cubemaps up front (mirrors three.js WebGLBackground).
    Texture* resolvedBackground = nullptr;
    if (background && background->isTexture()) {
        resolvedBackground = cubemaps.get(background->texture().get());
    }

    if (!background || (background && background->empty())) {

        setClear(clearColor, clearAlpha);

    } else if (background && background->isColor()) {

        setClear(background->color(), 1);
        forceClear = true;
    }

    if (renderer.autoClear || forceClear) {

        renderer.clear(renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil);
    }

    // resolvedBackground is either the original texture or the converted CubeTexture.
    // Only render the skybox if it resolves to a CubeTexture.
    if (auto* cubeBackground = dynamic_cast<CubeTexture*>(resolvedBackground)) {

        auto tex = background->texture();

        if (!boxMesh) {
            auto shaderMaterial = ShaderMaterial::create();
            shaderMaterial->name = "BackgroundCubeMaterial";
            shaderMaterial->uniforms = shaders::ShaderLib::instance().cube.uniforms;
            shaderMaterial->vertexShader = shaders::ShaderLib::instance().cube.vertexShader;
            shaderMaterial->fragmentShader = shaders::ShaderLib::instance().cube.fragmentShader;
            shaderMaterial->side = Side::Back;
            shaderMaterial->depthTest = false;
            shaderMaterial->depthWrite = false;
            shaderMaterial->fog = false;

            auto geometry = BoxGeometry::create(1, 1, 1);
            geometry->deleteAttribute("normal");
            geometry->deleteAttribute("uv");

            boxMesh = std::make_unique<Mesh>(geometry, shaderMaterial);

            boxMesh->onBeforeRender = [&](void*, Object3D*, Camera* camera, BufferGeometry*, Material*, std::optional<GeometryGroup>) {
                boxMesh->matrixWorld->copyPosition(*camera->matrixWorld);
            };

            objects.update(boxMesh.get());
        }

        auto shaderMaterial = boxMesh->material()->as<ShaderMaterial>();
        // envMap must be non-null for USE_ENVMAP to be compiled in; the uniform carries the actual ptr.
        shaderMaterial->envMap = tex;
        shaderMaterial->uniforms.at("envMap").setValue(cubeBackground);
        shaderMaterial->uniforms.at("flipEnvMap").setValue(cubeBackground->_needsFlipEnvMap);

        if (currentBackground != &background.value() || currentBackgroundVersion != tex->version() || currentTonemapping != renderer.toneMapping) {

            shaderMaterial->needsUpdate();

            currentBackground = &background.value();
            currentBackgroundVersion = tex->version();
            currentTonemapping = renderer.toneMapping;
        }

        renderList.unshift(boxMesh.get(), boxMesh->geometry().get(), boxMesh->material().get(), 0, 0, std::nullopt);
    }
}

void GLBackground::setClearColor(const Color& color, float alpha) {

    clearColor.copy(color);
    clearAlpha = alpha;
    setClear(clearColor, clearAlpha);
}

const Color& GLBackground::getClearColor() const {

    return clearColor;
}

float GLBackground::getClearAlpha() const {

    return clearAlpha;
}

void GLBackground::setClearAlpha(float alpha) {

    clearAlpha = alpha;
    setClear(clearColor, clearAlpha);
}

void GLBackground::setClear(const Color& color, float alpha) {

    state.colorBuffer.setClear(color.r, color.g, color.b, alpha, premultipliedAlpha);
}
