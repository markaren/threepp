
#include "threepp/renderers/gl/GLShadowMap.hpp"

#include "threepp/math/Frustum.hpp"

#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"

#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"

#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/PointLightShadow.hpp"

#include "threepp/renderers/GLCubeRenderTarget.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"

#include "threepp/renderers/gl/GLCapabilities.hpp"
#include "threepp/renderers/gl/GLObjects.hpp"
#include "threepp/renderers/gl/GLTextures.hpp"

#include "threepp/textures/CubeDepthTexture.hpp"


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
    //
    // Since the r186 port only VSM still reads the packed colour (its caster
    // pass writes depth as RGBA for the blur); PCF and Basic read the depth
    // attachment, which clear() resets to 1 alongside.
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

    std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Material>>> _materialCache;

    // Depth materials for alpha-cutout casters, one per SOURCE material.
    std::unordered_map<std::string, std::shared_ptr<MeshDepthMaterial>> _cutoutDepthMaterials;

    int _maxTextureSize;

    std::shared_ptr<Mesh> fullScreenMesh;

    // Per-face frustum for point lights, which render into a cube target here
    // rather than through PointLightShadow's atlas viewports.
    Frustum _pointFrustum;

    // What a shadow's targets were built for. The receiver's sampler type
    // follows it: VSM samples moments from a colour target, Compare a depth
    // texture through a shadow sampler (hardware PCF), Plain raw depth.
    enum class MapKind { VSM, Compare, Plain };

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

    // Every light, point lights included, renders with the depth material: the
    // shadow is the hardware depth written into the target's depth texture
    // (native cube depth for point lights, as three.js r186), so the colour a
    // caster writes only matters for VSM, and point lights never use VSM here.
    // That also gives point-light shadows the alpha-cutout path, which the old
    // distance material never had.
    Material* getDepthMaterial(GLRenderer& _renderer, Material* material, bool vsmCaster) {

        Material* result;

        if (auto* cutout = getCutoutDepthMaterial(material)) {

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


        if (vsmCaster) {

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

        return result;
    }

    void renderObject(GLRenderer& _renderer, Object3D* object, Camera* camera, Camera* shadowCamera, bool vsmCaster) {

        if (!object->visible) return;

        bool visible = object->layers.test(camera->layers);

        if (visible && (object->is<Mesh>() || object->is<Line>() || object->is<Points>())) {

            if ((object->castShadow || (object->receiveShadow && vsmCaster)) && (!object->frustumCulled || _frustum->intersectsObject(*object))) {

                object->modelViewMatrix.multiplyMatrices(shadowCamera->matrixWorldInverse, *object->matrixWorld);

                const auto geometry = _objects.update(object);
                const auto material = object->as<ObjectWithMaterials>()->materials();

                if (material.size() > 1) {

                    const auto& groups = geometry->groups;

                    for (const auto& group : groups) {

                        if (material.size() > group.materialIndex) {
                            const auto groupMaterial = material[group.materialIndex].get();

                            if (groupMaterial && groupMaterial->visible) {

                                const auto depthMaterial = getDepthMaterial(_renderer, groupMaterial, vsmCaster);

                                _renderer.renderBufferDirect(shadowCamera, nullptr, geometry, depthMaterial, object, group);
                            }
                        }
                    }

                } else if (material.front()->visible) {

                    const auto depthMaterial = getDepthMaterial(_renderer, material.front().get(), vsmCaster);

                    _renderer.renderBufferDirect(shadowCamera, nullptr, geometry, depthMaterial, object, std::nullopt);
                }
            }
        }

        for (auto& child : object->children) {

            renderObject(_renderer, child, camera, shadowCamera, vsmCaster);
        }
    }

    // A point light's six faces, each rendered into its own face of the cube
    // target's native depth cube map. Face directions and ups are three.js
    // r186's (WebGLShadowMap _cubeDirections/_cubeUps), which lay the faces out
    // the way GL cube-map sampling expects, so the receiver can look the
    // shadow up with the raw light-to-fragment vector.
    void renderPointShadow(GLRenderer& _renderer, LightShadow& shadow, Light& light, Object3D* scene, Camera* camera) {

        static const Vector3 cubeDirections[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        static const Vector3 cubeUps[6] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};

        auto& shadowCamera = *shadow.camera;

        const auto* pointLight = light.as<PointLight>();
        const float far = (pointLight && pointLight->distance > 0) ? pointLight->distance : shadowCamera.farPlane;
        if (far != shadowCamera.farPlane) {
            shadowCamera.farPlane = far;
            shadowCamera.updateProjectionMatrix();
        }

        Vector3 lightPositionWorld;
        lightPositionWorld.setFromMatrixPosition(*light.matrixWorld);

        // The receiver's coordinate is the light-to-fragment vector.
        shadow.matrix.makeTranslation(-lightPositionWorld.x, -lightPositionWorld.y, -lightPositionWorld.z);

        Matrix4 projScreenMatrix;
        Vector3 lookTarget;

        for (int face = 0; face < 6; face++) {

            shadowCamera.position.copy(lightPositionWorld);
            lookTarget.copy(lightPositionWorld).add(cubeDirections[face]);
            shadowCamera.up.copy(cubeUps[face]);
            shadowCamera.lookAt(lookTarget);
            shadowCamera.updateMatrixWorld();

            projScreenMatrix.multiplyMatrices(shadowCamera.projectionMatrix, shadowCamera.matrixWorldInverse);
            _pointFrustum.setFromProjectionMatrix(projScreenMatrix);
            _frustum = &_pointFrustum;

            _renderer.setRenderTarget(shadow.map.get(), face);
            _renderer.state().colorBuffer.setClear(1, 1, 1, 1);
            _renderer.clear();

            renderObject(_renderer, scene, camera, &shadowCamera, false);
        }
    }

    void render(GLRenderer& _renderer, const std::vector<Light*>& lights, Object3D* scene, Camera* camera) {

        if (!scope->enabled) return;
        if (!scope->autoUpdate && !scope->needsUpdate) return;

        if (lights.empty()) return;

        if (scope->type == ShadowMap::PFCSoft) {

            // Removed in three.js r183; r186 warns and uses PCF, whose rotated
            // Vogel disk is softer than PCFSoft's bilinear 3x3 was anyway.
            static bool warned = false;
            if (!warned) {
                std::cerr << "THREE.GLShadowMap: PCFSoftShadowMap has been removed. Using PCFShadowMap instead." << std::endl;
                warned = true;
            }
            scope->type = ShadowMap::PFC;
        }

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

            const bool isPoint = std::dynamic_pointer_cast<PointLightShadow>(shadow) != nullptr;

            // Point lights have no VSM path (three.js r186 drops their shadow
            // under VSM with a warning); here they keep the PCF cube instead,
            // so switching a scene to VSM does not silently lose them.
            const MapKind wantKind = (scope->type == ShadowMap::VSM && !isPoint) ? MapKind::VSM
                                     : (scope->type == ShadowMap::Basic)        ? MapKind::Plain
                                                                                : MapKind::Compare;

            if (isPoint) {

                // One square cube face per side, rather than PointLightShadow's
                // 4x2 atlas of viewports.
                const float size = std::min(shadow->mapSize.x, static_cast<float>(_maxTextureSize));
                _shadowMapSize.set(size, size);
                _viewportSize.set(size, size);

            } else {

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
            }

            if (shadow->map) {

                // The shadow type (or the map size) changed after the targets
                // were allocated: drop them and rebuild for the current state.
                // The receiver's sampler type follows the kind, so a map built
                // for another kind cannot be reused, only replaced. (Switching
                // TO VSM once left mapPass null and crashed the blur; switching
                // away kept VSM's Linear moments.)
                MapKind haveKind = MapKind::Plain;
                if (shadow->mapPass) haveKind = MapKind::VSM;
                else if (shadow->map->depthTexture && shadow->map->depthTexture->compareFunction) haveKind = MapKind::Compare;

                const bool resized = shadow->map->width != static_cast<unsigned>(_shadowMapSize.x) ||
                                     shadow->map->height != static_cast<unsigned>(_shadowMapSize.y);

                if (haveKind != wantKind || resized) {
                    shadow->dispose();
                    shadow->map.reset();
                    shadow->mapPass.reset();
                }
            }

            if (!shadow->map && wantKind == MapKind::VSM) {

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
                    pars.minFilter = Filter::LinearMipmapLinear;
                    pars.magFilter = Filter::Linear;
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
                    pars.type = Type::Float;

                    shadow->map = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), pars);
                    shadow->map->texture->name = light->name + ".shadowMap";
                    // Set on the texture rather than through Options, whose
                    // generateMipmaps field the RenderTarget constructor never
                    // reads. Only the map is mipmapped: mapPass is scratch that
                    // only the horizontal blur samples, at level 0.
                    shadow->map->texture->generateMipmaps = true;

                    auto passPars = pars;
                    passPars.minFilter = Filter::Linear;
                    shadow->mapPass = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), passPars);
                    shadow->mapPass->texture->generateMipmaps = false;

                    shadow->camera->updateProjectionMatrix();

            } else if (!shadow->map) {

                // PCF and Basic: the shadow is the native depth buffer, kept as
                // a 24-bit DepthTexture (three.js r186: UnsignedIntType). PCF
                // sets a LessEqual compare and Linear filtering so each tap of
                // the receiver's sampler2DShadow is a hardware 2x2 PCF; Basic
                // reads raw depth, Nearest. The colour attachment is written by
                // the depth material but never sampled.
                GLRenderTarget::Options pars{};
                pars.minFilter = Filter::Nearest;
                pars.magFilter = Filter::Nearest;
                pars.format = Format::RGBA;

                const auto w = static_cast<int>(_shadowMapSize.x);
                const auto h = static_cast<int>(_shadowMapSize.y);

                std::shared_ptr<DepthTexture> depth;
                if (isPoint) {
                    shadow->map = std::make_unique<GLCubeRenderTarget>(w, pars);
                    depth = CubeDepthTexture::create(static_cast<unsigned>(w), Type::UnsignedInt);
                } else {
                    shadow->map = GLRenderTarget::create(w, h, pars);
                    depth = DepthTexture::create(Type::UnsignedInt);
                }
                shadow->map->texture->generateMipmaps = false;
                shadow->map->texture->name = light->name + ".shadowMap";

                depth->name = light->name + ".shadowMap";
                if (wantKind == MapKind::Compare) {
                    depth->compareFunction = DepthFunc::LessEqual;
                    depth->minFilter = Filter::Linear;
                    depth->magFilter = Filter::Linear;
                } else {
                    depth->compareFunction.reset();
                    depth->minFilter = Filter::Nearest;
                    depth->magFilter = Filter::Nearest;
                }
                shadow->map->depthTexture = depth;

                shadow->camera->updateProjectionMatrix();
            }

            const bool vsmCaster = wantKind == MapKind::VSM;

            if (isPoint) {

                renderPointShadow(_renderer, *shadow, *light, scene, camera);

            } else {

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

                    shadow->updateMatrices(*light);

                    _frustum = &shadow->getFrustum();

                    renderObject(_renderer, scene, camera, shadow->camera.get(), vsmCaster);
                }

                // do blur pass for VSM

                if (vsmCaster) {

                    VSMPass(_renderer, shadow.get(), camera);
                }
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
