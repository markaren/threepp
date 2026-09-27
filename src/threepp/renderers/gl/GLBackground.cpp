
#include "threepp/renderers/gl/GLBackground.hpp"
#include "threepp/renderers/gl/GLCubeMaps.hpp"
#include "threepp/renderers/gl/GLObjects.hpp"
#include "threepp/renderers/gl/GLRenderLists.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/Renderer.hpp"

#include "threepp/renderers/shaders/ShaderLib.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;
using namespace threepp::gl;

GLBackground::GLBackground(GLRenderer& renderer, GLCubeMaps& cubemaps, GLState& state, GLObjects& objects, bool premultipliedAlpha)
    : renderer(renderer), cubemaps(cubemaps), state(state), objects(objects), premultipliedAlpha(premultipliedAlpha) {}

void GLBackground::render(GLRenderList& renderList, Object3D* scene) {

    bool forceClear = false;
    auto* asScene = scene->as<Scene>();

    std::optional<Background> background;

    if (asScene) {
        background = asScene->background;
    }

    // three.js r146/r147/r162 background knobs. The defaults leave every path
    // below exactly as it was.
    const float blurriness = asScene ? std::clamp(asScene->backgroundBlurriness, 0.f, 1.f) : 0.f;
    const float intensity = asScene ? asScene->backgroundIntensity : 1.f;

    // Resolve equirectangular textures to cubemaps up front (mirrors three.js WebGLBackground).
    // A blurred background resolves to the PMREM instead (r146: usePMREM =
    // backgroundBlurriness > 0) and is read at roughness = blurriness; that
    // holds for equirect and cube textures, not for a plain 2D one.
    Texture* resolvedBackground = nullptr;
    if (background && background->isTexture()) {
        auto* source = background->texture().get();
        const bool hasPMREM = source->mapping == Mapping::EquirectangularReflection ||
                              source->mapping == Mapping::EquirectangularRefraction ||
                              dynamic_cast<CubeTexture*>(source) != nullptr;
        resolvedBackground = (blurriness > 0.f && hasPMREM) ? cubemaps.getBackgroundPMREM(source)
                                                            : cubemaps.get(source);
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

    // resolvedBackground is the original texture, the converted CubeTexture, or
    // (blurred) the PMREM atlas. Cube and atlas go on the skybox; any other
    // texture is a plain 2D background drawn full-screen.
    auto* cubeBackground = dynamic_cast<CubeTexture*>(resolvedBackground);
    const bool isCubeUV = resolvedBackground && resolvedBackground->mapping == Mapping::CubeUVReflection;

    if (cubeBackground || isCubeUV) {

        auto tex = background->texture();
        // Wrap the resolved cube texture in a non-owning shared_ptr so we can assign it to
        // MaterialWithEnvMap::envMap. This matches three.js WebGLBackground: the material's
        // envMap points at the *resolved* cube texture, so WebGLPrograms/GLRenderer reads
        // CubeReflection mapping and compiles ENVMAP_TYPE_CUBE (samplerCube), which matches
        // the uniform bound below. Storage still lives in GLCubeMaps.
        auto resolvedShared = std::shared_ptr<Texture>(resolvedBackground, [](Texture*) {});

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
        shaderMaterial->uniforms.at("envMap").setValue(resolvedBackground);
        shaderMaterial->uniforms.at("flipEnvMap").setValue((cubeBackground && cubeBackground->_needsFlipEnvMap) ? -1.f : 1.f);
        shaderMaterial->uniforms.at("backgroundBlurriness").setValue(blurriness);
        shaderMaterial->uniforms.at("backgroundIntensity").setValue(intensity);

        // scene.backgroundRotation, applied exactly as scene.environmentRotation
        // is on materials (world -> env = transpose of the rotation, then the
        // flipEnvMap mirror), so a background and the reflections of the same
        // texture under the same rotation always agree. r186's WebGLBackground
        // builds this from the NEGATED Euler angles instead, which equals the
        // transpose for a rotation about one axis but not for a compound one.
        Matrix3 rotation;
        if (asScene) {
            const auto& r = asScene->backgroundRotation;
            if (static_cast<float>(r.x) != 0 || static_cast<float>(r.y) != 0 || static_cast<float>(r.z) != 0) {
                Matrix4 m;
                m.makeRotationFromEuler(r);
                rotation.setFromMatrix4(m).transpose();
            }
        }
        shaderMaterial->uniforms.at("envMapRotation").setValue(rotation);

        if (currentBackground != &background.value() || currentBackgroundVersion != tex->version() ||
            currentTonemapping != renderer.toneMapping || currentResolved != resolvedBackground) {

            shaderMaterial->needsUpdate();

            currentBackground = &background.value();
            currentBackgroundVersion = tex->version();
            currentTonemapping = renderer.toneMapping;
            currentResolved = resolvedBackground;
        }

        renderList.unshift(boxMesh.get(), boxMesh->geometry().get(), boxMesh->material().get(), 0, 0, std::nullopt);

    } else if (resolvedBackground) {

        // A plain 2D texture: r129/r186 WebGLBackground's planeMesh, a
        // full-screen quad the background vertex shader pins to the far plane.
        auto* tex = resolvedBackground;

        if (!planeMesh) {
            auto shaderMaterial = ShaderMaterial::create();
            shaderMaterial->name = "BackgroundMaterial";
            shaderMaterial->uniforms = shaders::ShaderLib::instance().background.uniforms;
            shaderMaterial->vertexShader = shaders::ShaderLib::instance().background.vertexShader;
            shaderMaterial->fragmentShader = shaders::ShaderLib::instance().background.fragmentShader;
            shaderMaterial->side = Side::Front;
            shaderMaterial->depthTest = false;
            shaderMaterial->depthWrite = false;
            shaderMaterial->fog = false;

            auto geometry = PlaneGeometry::create(2, 2);
            geometry->deleteAttribute("normal");

            planeMesh = std::make_unique<Mesh>(geometry, shaderMaterial);
            planeMesh->frustumCulled = false;

            objects.update(planeMesh.get());
        }

        auto shaderMaterial = planeMesh->materialAs<ShaderMaterial>();
        shaderMaterial->uniforms.at("t2D").setValue(tex);
        shaderMaterial->uniforms.at("backgroundIntensity").setValue(intensity);

        // ShaderMaterial carries no `map`, so the program cannot pick up the
        // texture's encoding the way a material's mapTexelToLinear does; the
        // shader decodes sRGB itself. r186 also leaves an sRGB (display-ready)
        // background out of tone mapping.
        const bool srgb = tex->colorSpace == ColorSpace::sRGB;
        shaderMaterial->toneMapped = !srgb;
        if (srgb) shaderMaterial->defines["BACKGROUND_SRGB"] = "";
        else shaderMaterial->defines.erase("BACKGROUND_SRGB");

        if (tex->matrixAutoUpdate) tex->updateMatrix();
        shaderMaterial->uniforms.at("uvTransform").value<Matrix3>().copy(tex->matrix);

        if (currentResolved != tex || currentBackgroundVersion != tex->version() ||
            currentTonemapping != renderer.toneMapping || currentSrgb != srgb) {

            shaderMaterial->needsUpdate();

            currentResolved = tex;
            currentBackgroundVersion = tex->version();
            currentTonemapping = renderer.toneMapping;
            currentSrgb = srgb;
        }

        renderList.unshift(planeMesh.get(), planeMesh->geometry().get(), planeMesh->material().get(), 0, 0, std::nullopt);
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
