
#include "threepp/renderers/gl/GLBackground.hpp"
#include "threepp/renderers/gl/GLCubeMaps.hpp"
#include "threepp/renderers/gl/GLObjects.hpp"
#include "threepp/renderers/gl/GLRenderLists.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/Renderer.hpp"

#include "threepp/renderers/shaders/ShaderLib.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"

#include <algorithm>
#include <cmath>

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
                // Under perspective the unit box fills the view at any size: the
                // eye sits inside it and the divide expands whatever it hits. A
                // parallel projection has no divide, so the box would project at
                // its literal 1 unit — the environment rendered as a small box in
                // the middle of an orthographic viewport. three.js has the same
                // gap (its WebGLBackground never scales the box and has no
                // orthographic branch), so there is no upstream behaviour to
                // follow; match the Vulkan backend instead, where a parallel
                // camera has ONE view direction for every pixel (camRayDir in
                // camera_ray.glsl). Two parts: cover the frustum, and shade every
                // pixel along the camera's forward.
                auto* mat = boxMesh->materialAs<ShaderMaterial>();
                if (auto* ortho = camera->as<OrthographicCamera>()) {
                    const float halfExtent =
                            std::max(std::max(std::abs(ortho->left), std::abs(ortho->right)),
                                     std::max(std::abs(ortho->top), std::abs(ortho->bottom)));
                    // The box is axis-aligned in WORLD space while the camera may
                    // be turned any way, so size it off the shape that projects
                    // identically from every angle: a cube of side s contains an
                    // inscribed sphere of radius s/2, which always projects to a
                    // disc of that radius. The frustum's bounding circle is at
                    // most sqrt(2)*halfExtent, so s = 4*halfExtent clears it with
                    // margin regardless of orientation.
                    const float s = std::max(4.f * halfExtent, 1.f);
                    boxMesh->matrixWorld->makeScale(s, s, s);

                    Vector3 forward;
                    camera->getWorldDirection(forward);
                    mat->uniforms.at("orthoDirection")
                            .setValue(Vector4(forward.x, forward.y, forward.z, 1.f));
                } else {
                    boxMesh->matrixWorld->identity();
                    mat->uniforms.at("orthoDirection").setValue(Vector4(0.f, 0.f, -1.f, 0.f));
                }
                boxMesh->matrixWorld->copyPosition(*camera->matrixWorld);
            };

            objects.update(boxMesh.get());
        }

        auto shaderMaterial = boxMesh->materialAs<ShaderMaterial>();
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

void GLBackground::refreshClear() {

    setClear(clearColor, clearAlpha);
}

void GLBackground::setClear(const Color& color, float alpha) {

    // Encode the clear color into the output color space before handing it to
    // glClearColor. The clear bypasses the fragment shader's output encode
    // (linearToOutputTexel), so without this an already color-managed (linear)
    // clear color would render too dark. Mirrors three.js WebGLBackground.setClear,
    // which does color.getRGB(_rgb, getUnlitUniformColorSpace(renderer)).
    // When ColorManagement is disabled this is a no-op (legacy raw behaviour).
    //
    // Which space that is depends on what is bound, exactly as it does for the
    // shader encode (GLRenderer::currentOutputColorSpace): a render target is
    // an intermediate, normally linear, and only the screen wants the display
    // encode. Reading the renderer's output space unconditionally put the sRGB
    // encode into the clear of every offscreen pass — so an EffectComposer,
    // whose final draw encodes again, showed its background twice-encoded and
    // washed out while the geometry around it was right.
    const auto* target = renderer.getRenderTarget();
    const ColorSpace space = target ? target->texture->colorSpace : renderer.outputColorSpace;

    Color c;
    c.copy(color);
    ColorManagement::workingToColorSpace(c, space);

    state.colorBuffer.setClear(c.r, c.g, c.b, alpha, premultipliedAlpha);
}
