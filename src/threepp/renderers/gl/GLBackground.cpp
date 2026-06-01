
#include "threepp/renderers/gl/GLBackground.hpp"
#include "threepp/renderers/gl/GLCubeMaps.hpp"
#include "threepp/renderers/gl/GLObjects.hpp"
#include "threepp/renderers/gl/GLRenderLists.hpp"
#include "threepp/renderers/Renderer.hpp"

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
        // Wrap the resolved cube texture in a non-owning shared_ptr so we can assign it to
        // MaterialWithEnvMap::envMap. This matches three.js WebGLBackground: the material's
        // envMap points at the *resolved* cube texture, so WebGLPrograms/GLRenderer reads
        // CubeReflection mapping and compiles ENVMAP_TYPE_CUBE (samplerCube), which matches
        // the uniform bound below. Storage still lives in GLCubeMaps.
        auto resolvedShared = std::shared_ptr<Texture>(cubeBackground, [](Texture*) {});

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
        // Point envMap at the *resolved* cube texture so ProgramParameters reads
        // CubeReflection mapping (samplerCube path). The uniform carries the same ptr.
        shaderMaterial->envMap = resolvedShared;
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

    // Encode the clear color into the output color space before handing it to
    // glClearColor. The clear bypasses the fragment shader's output encode
    // (linearToOutputTexel), so without this an already color-managed (linear)
    // clear color would render too dark. Mirrors three.js WebGLBackground.setClear,
    // which does color.getRGB(_rgb, getUnlitUniformColorSpace(renderer)).
    // When ColorManagement is disabled this is a no-op (legacy raw behaviour).
    Color c;
    c.copy(color);
    ColorManagement::workingToColorSpace(c, renderer.outputColorSpace);

    state.colorBuffer.setClear(c.r, c.g, c.b, alpha, premultipliedAlpha);
}
