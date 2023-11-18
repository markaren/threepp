
#include "threepp/renderers/gl/GLBackground.hpp"
#include "threepp/renderers/gl/GLCubeMaps.hpp"
#include "threepp/renderers/gl/GLObjects.hpp"
#include "threepp/renderers/gl/GLState.hpp"
#include "threepp/renderers/gl/GLRenderLists.hpp"

#include "threepp/renderers/GLRenderer.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"

using namespace threepp;
using namespace threepp::gl;

GLBackground::GLBackground(GLRenderer& renderer, GLCubeMaps& cubemaps, GLState& state, GLObjects& objects, bool premultipliedAlpha)
    : renderer(renderer), cubemaps(cubemaps), state(state), objects(objects), premultipliedAlpha(premultipliedAlpha) {}

void GLBackground::render(GLRenderList& renderList, Object3D* scene) {

    bool forceClear = false;
    bool isScene = scene->is<Scene>();

    std::optional<Background> _background;

    if (isScene) {
        _background = scene->as<Scene>()->background;
    }

    if (_background && _background->isTexture()) {

        cubemaps.get(_background->texture());
    }

    if (!_background) {

        setClear(clearColor, clearAlpha);

    } else if (_background && _background->isColor()) {

        setClear(_background->color(), 1);
        forceClear = true;
    }

    if (renderer.autoClear || forceClear) {

        renderer.clear(renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil);
    }

    if (_background && _background->isTexture()) {

        auto tex = _background->texture();
        if (auto cubeTexture = dynamic_cast<CubeTexture*>(tex)) {

            if (!boxMesh) {
                auto shaderMaterial = ShaderMaterial::create();
                shaderMaterial->name = "BackgroundCubeMaterial";
                shaderMaterial->uniforms = std::make_shared<UniformMap>(shaders::ShaderLib::instance().cube.uniforms);
                shaderMaterial->vertexShader = shaders::ShaderLib::instance().cube.vertexShader;
                shaderMaterial->fragmentShader = shaders::ShaderLib::instance().cube.fragmentShader;
                shaderMaterial->side = Side::Back;
                shaderMaterial->depthTest = false;
                shaderMaterial->depthWrite = false;
                shaderMaterial->fog = false;

                auto geometry = BoxGeometry::create(1, 1, 1);
                geometry->deleteAttribute("normal");
                geometry->deleteAttribute("uv");

                boxMesh = Mesh::create(geometry, shaderMaterial);

                boxMesh->onBeforeRender = [&](void*, Object3D*, Camera* camera, BufferGeometry*, Material*, std::optional<GeometryGroup>) {
                    boxMesh->matrixWorld->copyPosition(*camera->matrixWorld);
                };
            }

            objects.update(boxMesh.get());
        }

        auto shaderMaterial = boxMesh->material()->as<ShaderMaterial>();
        shaderMaterial->uniforms->at("envMap").setValue(_background->texture());
        shaderMaterial->uniforms->at("flipEnvMap").setValue(true);
        shaderMaterial->needsUpdate();

        renderList.unshift(boxMesh.get(), boxMesh->geometry(), boxMesh->material(), 0, 0, std::nullopt);

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
