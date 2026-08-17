
#include "threepp/renderers/gl/GLShadowMap.hpp"

#include "threepp/math/Frustum.hpp"

#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"

#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/MeshDistanceMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"

#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/PointLightShadow.hpp"

#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"

#include "threepp/renderers/gl/GLCapabilities.hpp"
#include "threepp/renderers/gl/GLObjects.hpp"
#include "threepp/renderers/gl/GLTextures.hpp"


#include <cmath>
#include <iostream>

using namespace threepp;
using namespace threepp::gl;

namespace {

    inline std::unordered_map<Side, Side> shadowSide{
            {Side::Front, Side::Back},
            {Side::Back, Side::Front},
            {Side::Double, Side::Double}};

    // Bind a shadow target and clear it to white.
    //
    // White is depth 1 - the far plane - so a texel no caster wrote reads as
    // lit. The white has to be set AFTER the bind, not once before the loop:
    // GLRenderer::setRenderTarget re-encodes the background's clear colour for
    // whatever is newly bound, so anything set beforehand is gone by the time
    // clear() runs. Setting it once up front cleared every shadow map to the
    // renderer's clear colour instead, and alpha carries almost all of
    // unpackRGBAToDepth - so the default alpha of 0 meant depth 0, the near
    // plane, and every receiver the light covered came back fully shadowed.
    void bindAndClearShadowTarget(GLRenderer& renderer, RenderTarget* target) {

        renderer.setRenderTarget(target);
        renderer.state().colorBuffer.setClear(1, 1, 1, 1);
        renderer.clear();
    }

}// namespace

struct GLShadowMap::Impl {

    GLShadowMap* scope;
    GLObjects& _objects;
    GLTextures& _textures;

    const Frustum* _frustum;

    Vector2 _shadowMapSize;
    Vector2 _viewportSize;

    Vector4 _viewport;

    std::vector<std::shared_ptr<MeshDepthMaterial>> _depthMaterials;
    std::vector<std::shared_ptr<MeshDistanceMaterial>> _distanceMaterials;

    std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Material>>> _materialCache;

    // Depth materials for alpha-cutout casters, one per SOURCE material.
    std::unordered_map<std::string, std::shared_ptr<MeshDepthMaterial>> _cutoutDepthMaterials;

    int _maxTextureSize;

    std::shared_ptr<Mesh> fullScreenMesh;

    Impl(GLShadowMap* scope, GLObjects& objects, GLTextures& textures)
        : scope(scope),
          _objects(objects),
          _textures(textures),
          _frustum(nullptr),
          _maxTextureSize(GLCapabilities::instance().maxTextureSize) {

        auto fullScreenTri = BufferGeometry::create();
        fullScreenTri->setAttribute("position", FloatBufferAttribute::create({-1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5}, 3));

        fullScreenMesh = Mesh::create(fullScreenTri, shadowMaterialVertical);
    }

    void VSMPass(GLRenderer& _renderer, LightShadow* shadow, Camera* camera) {

        // Belt and braces: the caller only reaches here for a non-point shadow
        // with type == VSM, and render() guarantees both targets exist for that
        // combination. Bail rather than dereference if that ever drifts again.
        if (!shadow->map || !shadow->mapPass) return;

        const auto& geometry = _objects.update(fullScreenMesh.get());

        // vertical pass

        shadowMaterialVertical->uniforms.at("shadow_pass").setValue(shadow->map->texture.get());
        shadowMaterialVertical->uniforms.at("resolution").value<Vector2>().copy(shadow->mapSize);
        shadowMaterialVertical->uniforms.at("radius").value<float>() = shadow->radius;
        bindAndClearShadowTarget(_renderer, shadow->mapPass.get());
        _renderer.renderBufferDirect(camera, nullptr, geometry, shadowMaterialVertical.get(), fullScreenMesh.get(), std::nullopt);

        // horizontal pass

        shadowMaterialHorizontal->uniforms.at("shadow_pass").setValue(shadow->mapPass->texture.get());
        shadowMaterialHorizontal->uniforms.at("resolution").value<Vector2>().copy(shadow->mapSize);
        shadowMaterialHorizontal->uniforms.at("radius").value<float>() = shadow->radius;
        bindAndClearShadowTarget(_renderer, shadow->map.get());
        _renderer.renderBufferDirect(camera, nullptr, geometry, shadowMaterialHorizontal.get(), fullScreenMesh.get(), std::nullopt);

        // The moments are final now, so build the mip chain the receiver will
        // pick its level from. Has to be here: the blur passes go through
        // renderBufferDirect, which is below the level of render() where the
        // renderer would otherwise do this for a bound target.
        _textures.updateRenderTargetMipmap(shadow->map.get());
    }

    MeshDepthMaterial* getDepthMaterialVariant(bool useMorphing) {
        unsigned index = useMorphing << 0;

        if (index >= _depthMaterials.size()) {

            auto material = MeshDepthMaterial::create();
            material->depthPacking = DepthPacking::RGBA;

            _depthMaterials.emplace_back(material);

            return material.get();
        }

        return _depthMaterials[index].get();
    }

    // Alpha-cutout casters (foliage cards, fences, grates) need their SILHOUETTE
    // in the shadow map, not their quad. The shared depth variants carry no
    // `map`/`alphaTest`, so a leaf card writes its full rectangle and a tree
    // casts one solid blob instead of dappled light — and, worse, the same solid
    // quads self-shadow the canopy into flat black. depth_frag.glsl already
    // includes <map_fragment>/<alphamap_fragment>/<alphatest_fragment>; the
    // properties simply have to be carried across.
    //
    // Cached per SOURCE material rather than set on the shared variant, because
    // the program cache keys on material version: re-pointing `map` on one
    // shared instance per object would need a version bump every draw (and so
    // re-derive program parameters for every shadow caster in the scene), or
    // else silently render with the previously-built program.
    MeshDepthMaterial* getCutoutDepthMaterial(Material* material) {
        if (material->alphaTest <= 0.f) return nullptr;

        auto* withMap = material->as<MaterialWithMap>();
        auto* withAlphaMap = material->as<MaterialWithAlphaMap>();
        std::shared_ptr<Texture> map = withMap ? withMap->map : nullptr;
        std::shared_ptr<Texture> alphaMap = withAlphaMap ? withAlphaMap->alphaMap : nullptr;
        if (!map && !alphaMap) return nullptr;

        auto& cached = _cutoutDepthMaterials[material->uuid()];
        if (!cached) {
            cached = MeshDepthMaterial::create();
            cached->depthPacking = DepthPacking::RGBA;
        }
        // Only touch (and re-version) the material when something really moved —
        // the source's map can be swapped at runtime, e.g. a regenerated tree.
        if (cached->map != map || cached->alphaMap != alphaMap ||
            cached->alphaTest != material->alphaTest) {
            cached->map = map;
            cached->alphaMap = alphaMap;
            cached->alphaTest = material->alphaTest;
            cached->needsUpdate();
        }
        return cached.get();
    }

    MeshDistanceMaterial* getDistanceMaterialVariant(bool useMorphing) {
        unsigned index = useMorphing << 0;

        if (index >= _distanceMaterials.size()) {

            auto material = MeshDistanceMaterial::create();

            _distanceMaterials.emplace_back(material);

            return material.get();
        }

        return _distanceMaterials[index].get();
    }

    Material* getDepthMaterial(GLRenderer& _renderer, Object3D* /*object*/, BufferGeometry* /*geometry*/, Material* material, Light* light, float shadowCameraNear, float shadowCameraFar) {

        Material* result;

        if (light->type() == "PointLight") {

            // MeshDistanceMaterial has no cutout path here — point-light shadows
            // from alpha-tested casters still write the full quad.
            result = getDistanceMaterialVariant(false);

        } else if (auto* cutout = getCutoutDepthMaterial(material)) {

            result = cutout;

        } else {

            result = getDepthMaterialVariant(false);
        }

        if (_renderer.localClippingEnabled && material->clipShadows && !material->clippingPlanes.empty()) {

            // in this case we need a unique material instance reflecting the
            // appropriate state

            auto keyA = result->uuid(), keyB = material->uuid();

            auto& materialsForVariant = _materialCache[keyA];

            auto& cachedMaterial = materialsForVariant[keyB];

            if (!cachedMaterial) {

                cachedMaterial = result->clone();
                materialsForVariant[keyB] = cachedMaterial;
            }

            result = cachedMaterial.get();
        }

        result->visible = material->visible;
        auto resultWithWireframe = result->as<MaterialWithWireframe>();
        auto materialWithWireframe = material->as<MaterialWithWireframe>();
        if (resultWithWireframe && materialWithWireframe) {
            resultWithWireframe->wireframe = materialWithWireframe->wireframe;
            resultWithWireframe->wireframeLinewidth = materialWithWireframe->wireframeLinewidth;
        }


        if (scope->type == ShadowMap::VSM) {

            result->side = (material->shadowSide) ? *material->shadowSide : material->side;

        } else {

            result->side = (material->shadowSide) ? *material->shadowSide : shadowSide[material->side];
        }

        result->clipShadows = material->clipShadows;
        result->clippingPlanes = material->clippingPlanes;
        result->clipIntersection = material->clipIntersection;

        auto resultWithLineWidth = result->as<MaterialWithLineWidth>();
        auto materialWithLineWidth = material->as<MaterialWithLineWidth>();
        if (resultWithLineWidth && materialWithLineWidth) {
            resultWithLineWidth->linewidth = materialWithLineWidth->linewidth;
        }

        if (light->type() == "PointLight") {
            if (auto distanceMaterial = material->as<MeshDistanceMaterial>()) {
                distanceMaterial->referencePosition.setFromMatrixPosition(*light->matrixWorld);
                distanceMaterial->nearDistance = shadowCameraNear;
                distanceMaterial->farDistance = shadowCameraFar;
            }
        }

        return result;
    }

    void renderObject(GLRenderer& _renderer, Object3D* object, Camera* camera, Camera* shadowCamera, Light* light) {

        if (!object->visible) return;

        bool visible = object->layers.test(camera->layers);

        if (visible && (object->is<Mesh>() || object->is<Line>() || object->is<Points>())) {

            if ((object->castShadow || (object->receiveShadow && scope->type == ShadowMap::VSM)) && (!object->frustumCulled || _frustum->intersectsObject(*object))) {

                object->modelViewMatrix.multiplyMatrices(shadowCamera->matrixWorldInverse, *object->matrixWorld);

                const auto geometry = _objects.update(object);
                const auto material = object->as<ObjectWithMaterials>()->materials();

                if (material.size() > 1) {

                    const auto& groups = geometry->groups;

                    for (const auto& group : groups) {

                        if (material.size() > group.materialIndex) {
                            const auto groupMaterial = material[group.materialIndex].get();

                            if (groupMaterial && groupMaterial->visible) {

                                const auto depthMaterial = getDepthMaterial(_renderer, object, geometry, groupMaterial, light, shadowCamera->nearPlane, shadowCamera->farPlane);

                                _renderer.renderBufferDirect(shadowCamera, nullptr, geometry, depthMaterial, object, group);
                            }
                        }
                    }

                } else if (material.front()->visible) {

                    const auto depthMaterial = getDepthMaterial(_renderer, object, geometry, material.front().get(), light, shadowCamera->nearPlane, shadowCamera->farPlane);

                    _renderer.renderBufferDirect(shadowCamera, nullptr, geometry, depthMaterial, object, std::nullopt);
                }
            }
        }

        for (auto& child : object->children) {

            renderObject(_renderer, child, camera, shadowCamera, light);
        }
    }

    void render(GLRenderer& _renderer, const std::vector<Light*>& lights, Object3D* scene, Camera* camera) {

        if (!scope->enabled) return;
        if (!scope->autoUpdate && !scope->needsUpdate) return;

        if (lights.empty()) return;

        auto currentRenderTarget = _renderer.getRenderTarget();
        auto activeCubeFace = _renderer.getActiveCubeFace();
        auto activeMipmapLevel = _renderer.getActiveMipmapLevel();

        auto& _state = _renderer.state();

        // Set GL state for depth map. The clear colour is NOT part of this
        // block - it belongs to each bind, see bindAndClearShadowTarget.
        _state.setBlending(Blending::None);
        _state.depthBuffer.setTest(true);
        _state.setScissorTest(false);

        // render depth map

        for (auto light : lights) {

            auto lightWithShadow = dynamic_cast<LightWithShadow*>(light);

            if (!lightWithShadow) {

                std::cerr << "THREE.GLShadowMap:'" << light->type() << "'has no shadow." << std::endl;
                continue;
            }

            auto shadow = lightWithShadow->shadow;

            if (!shadow->autoUpdate && !shadow->needsUpdate) continue;

            _shadowMapSize.copy(shadow->mapSize);

            auto shadowFrameExtents = shadow->getFrameExtents();

            _shadowMapSize.multiply(shadowFrameExtents);

            _viewportSize.copy(shadow->mapSize);

            if (_shadowMapSize.x > _maxTextureSize || _shadowMapSize.y > _maxTextureSize) {

                if (_shadowMapSize.x > _maxTextureSize) {

                    _viewportSize.x = std::floor(static_cast<float>(_maxTextureSize) / shadowFrameExtents.x);
                    _shadowMapSize.x = _viewportSize.x * shadowFrameExtents.x;
                    shadow->mapSize.x = _viewportSize.x;
                }

                if (_shadowMapSize.y > _maxTextureSize) {

                    _viewportSize.y = std::floor(static_cast<float>(_maxTextureSize) / shadowFrameExtents.y);
                    _shadowMapSize.y = _viewportSize.y * shadowFrameExtents.y;
                    shadow->mapSize.y = _viewportSize.y;
                }
            }

            // VSM blurs the depth map through a second target, so it needs
            // `mapPass` and must sample with Linear; every other type compares
            // depth directly and must NOT filter across texels. `mapPass` is
            // therefore also the marker for "these targets were built for VSM".
            const bool wantVsm = scope->type == ShadowMap::VSM && !std::dynamic_pointer_cast<PointLightShadow>(shadow);
            const bool haveVsm = shadow->mapPass != nullptr;

            if (shadow->map && wantVsm != haveVsm) {

                // The shadow type changed after the targets were allocated.
                // Previously the allocation was guarded on `!shadow->map` alone,
                // so switching TO VSM at runtime left mapPass null and VSMPass()
                // dereferenced it — an outright crash. Switching AWAY from VSM
                // silently kept the Linear filtering, softening every other type.
                // Drop both and rebuild for the current type.
                shadow->dispose();
                shadow->map.reset();
                shadow->mapPass.reset();
            }

            if (!shadow->map) {

                GLRenderTarget::Options pars{};
                // Mipmapped moments for VSM. This is the one thing VSM can do
                // that no depth-comparison filter can: a mean and a variance
                // average correctly, so a mip level *is* the right answer for a
                // pixel covering many texels, where an averaged depth would be
                // meaningless. Without it the moments are point-sampled and, on
                // a receiver at an angle to the light, neighbouring pixels land
                // on texels whose means straddle the surface — a stipple across
                // the whole frustum that looks like noise and is really
                // undersampling. The editor's default scene hits it: a 2048 map
                // over the 10-unit shadow camera against a ~640px view is 4:1.
                pars.minFilter = wantVsm ? Filter::LinearMipmapLinear : Filter::Nearest;
                pars.magFilter = wantVsm ? Filter::Linear : Filter::Nearest;
                pars.format = Format::RGBA;

                // VSM stores moments — a mean depth and a standard deviation —
                // and then asks for the variance, a difference of two nearly
                // equal numbers. Eight-bit channels cannot carry that: the
                // default shadow camera spans 0.5..500, so a scene a few units
                // from the light sits at a depth near 0.01 and uses a hundredth
                // of the range. The variance underflows to zero, Chebyshev's
                // inequality degenerates, and neighbouring texels disagree at
                // random — a moiré of fringes across every receiver.
                //
                // Float moments fix it at the source: precision no longer
                // bounds how finely two nearby depths can be told apart, at any
                // range the camera happens to have. Deliberately unlike
                // three.js, which packs the moments into RGBA8 and so works
                // only where the shadow camera was fitted to the scene by hand.
                // Nothing in the public API moves — the same ShadowMap::VSM
                // with the same LightShadow knobs.
                //
                // Full float, not half: at a depth of 0.01 a half's ulp is
                // ~8e-6 against the packed format's 1.5e-5, which measurably
                // does NOT clear the fringes. Costs 4x a packed map on both
                // targets, so VSM is the one type that pays for its map — fair,
                // since it is the one type that cannot work without it. The
                // caster pass still writes 24-bit packed depth exactly as
                // before; a float target stores that losslessly.
                //
                // Desktop GL 3.3 has RGBA32F both colour-renderable and
                // linearly filterable in core. WebGL2 needs EXT_color_buffer_float
                // to render to it and OES_texture_float_linear to filter it.
                if (wantVsm) pars.type = Type::Float;

                shadow->map = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), pars);
                shadow->map->texture->name = light->name + ".shadowMap";
                // Set on the texture rather than through Options, whose
                // generateMipmaps field the RenderTarget constructor never
                // reads. Only the map is mipmapped: mapPass is scratch that
                // only the horizontal blur samples, at level 0.
                shadow->map->texture->generateMipmaps = wantVsm;

                if (wantVsm) {

                    auto passPars = pars;
                    passPars.minFilter = Filter::Linear;
                    shadow->mapPass = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), passPars);
                    shadow->mapPass->texture->generateMipmaps = false;
                }

                shadow->camera->updateProjectionMatrix();
            }

            bindAndClearShadowTarget(_renderer, shadow->map.get());

            const auto viewportCount = shadow->getViewportCount();

            for (unsigned vp = 0; vp < viewportCount; vp++) {

                const auto& viewport = shadow->getViewport(vp);

                _viewport.set(
                        _viewportSize.x * viewport.x,
                        _viewportSize.y * viewport.y,
                        _viewportSize.x * viewport.z,
                        _viewportSize.y * viewport.w);

                _state.viewport(_viewport);

                if (auto pointLightShadow = std::dynamic_pointer_cast<PointLightShadow>(shadow)) {
                    pointLightShadow->updateMatrices(*light->as<PointLight>(), vp);
                } else {
                    shadow->updateMatrices(*light);
                }

                _frustum = &shadow->getFrustum();

                renderObject(_renderer, scene, camera, shadow->camera.get(), light);
            }

            // do blur pass for VSM

            if (!std::dynamic_pointer_cast<PointLightShadow>(shadow) && scope->type == ShadowMap::VSM) {

                VSMPass(_renderer, shadow.get(), camera);
            }

            shadow->needsUpdate = false;
        }

        scope->needsUpdate = false;

        _renderer.setRenderTarget(currentRenderTarget, activeCubeFace, activeMipmapLevel);
    }

private:
    static std::shared_ptr<ShaderMaterial> createShadowMaterialVertical() {

        auto shadowMaterialVertical = ShaderMaterial::create();
        shadowMaterialVertical->vertexShader = shaders::ShaderChunk::instance().get("vsm_vert");
        shadowMaterialVertical->fragmentShader = shaders::ShaderChunk::instance().get("vsm_frag");

        shadowMaterialVertical->defines["SAMPLE_RATE"] = std::to_string(2.f / 8.f);
        shadowMaterialVertical->defines["HALF_SAMPLE_RATE"] = std::to_string(1.f / 8.f);

        shadowMaterialVertical->uniforms = {
                {"shadow_pass", Uniform()},
                {"resolution", Uniform(Vector2())},
                {"radius", Uniform(4.f)}};

        return shadowMaterialVertical;
    }


    static std::shared_ptr<ShaderMaterial> createShadowMaterialHorizontal() {

        auto horizontal = createShadowMaterialVertical();
        horizontal->defines["HORIZONTAL_PASS "] = "1";

        return horizontal;
    }

    std::shared_ptr<ShaderMaterial> shadowMaterialVertical = createShadowMaterialVertical();
    std::shared_ptr<ShaderMaterial> shadowMaterialHorizontal = createShadowMaterialHorizontal();
};

GLShadowMap::GLShadowMap(GLObjects& objects, GLTextures& textures)
    : pimpl_(std::make_unique<Impl>(this, objects, textures)) {}


void GLShadowMap::render(GLRenderer& renderer, const std::vector<Light*>& lights, Object3D* scene, Camera* camera) {

    pimpl_->render(renderer, lights, scene, camera);
}

gl::GLShadowMap::~GLShadowMap() = default;
