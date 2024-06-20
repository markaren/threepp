
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

    if (background && background->isTexture()) {

        cubemaps.get(background->texture().get());
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

    if (background && background->isTexture()) {

        auto tex = background->texture();
        if (auto cubeTexture = std::dynamic_pointer_cast<CubeTexture>(tex)) {

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
            shaderMaterial->envMap = background->texture();
            shaderMaterial->uniforms.at("envMap").setValue(background->texture().get());
            shaderMaterial->uniforms.at("flipEnvMap").setValue(cubeTexture->_needsFlipEnvMap);

            if (currentBackground != &background.value() || currentBackgroundVersion != tex->version() || currentTonemapping != renderer.toneMapping) {

                shaderMaterial->needsUpdate();

                currentBackground = &background.value();
                currentBackgroundVersion = tex->version();
                currentTonemapping = renderer.toneMapping;
            }

            renderList.unshift(boxMesh.get(), boxMesh->geometry().get(), boxMesh->material().get(), 0, 0, std::nullopt);
        }
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
