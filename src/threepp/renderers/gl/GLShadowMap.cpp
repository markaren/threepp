
#include "threepp/renderers/gl/GLShadowMap.hpp"

#include "threepp/objects/Line.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    inline std::unordered_map<int, int> shadowSide{
            {0, BackSide},
            {1, FrontSide},
            {2, DoubleSide}};

    std::shared_ptr<ShaderMaterial> createShadowMaterialVertical() {

        auto shadowMaterialVertical = ShaderMaterial::create();
        shadowMaterialVertical->vertexShader = shaders::ShaderChunk::instance().get("vsm_vert");
        shadowMaterialVertical->fragmentShader = shaders::ShaderChunk::instance().get("vsm_frag");

        shadowMaterialVertical->defines["SAMPLE_RATE"] = std::to_string(2.f / 8.f);
        shadowMaterialVertical->defines["HALF_SAMPLE_RATE"] = std::to_string(1.f / 8.f);

        shadowMaterialVertical->uniforms = std::make_shared<UniformMap>(UniformMap{
                {"shadow_pass", Uniform()},
                {"resolution", Uniform(Vector2())},
                {"radius", Uniform(4.f)}});

        return shadowMaterialVertical;
    }


    std::shared_ptr<ShaderMaterial> createShadowMaterialHorizontal() {

        auto horizontal = createShadowMaterialVertical();
        horizontal->defines["HORIZONTAL_PASS "] = "1";

        return horizontal;
    }

    std::shared_ptr<ShaderMaterial> shadowMaterialVertical = createShadowMaterialVertical();
    std::shared_ptr<ShaderMaterial> shadowMaterialHorizontal = createShadowMaterialHorizontal();


}// namespace

GLShadowMap::GLShadowMap(GLObjects& _objects)
    : _objects(_objects),
      _maxTextureSize(GLCapabilities::instance().maxTextureSize) {


    auto fullScreenTri = BufferGeometry::create();
    fullScreenTri->setAttribute("position", FloatBufferAttribute::create(std::vector<float>{-1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5}, 3));

    fullScreenMesh = Mesh::create(fullScreenTri, shadowMaterialVertical);
}


void GLShadowMap::render(GLRenderer& _renderer, const std::vector<Light*>& lights, Scene* scene, Camera* camera) {

    if (!enabled) return;
    if (!autoUpdate && !needsUpdate) return;

    if (lights.empty()) return;

    auto currentRenderTarget = _renderer.getRenderTarget();
    auto activeCubeFace = _renderer.getActiveCubeFace();
    auto activeMipmapLevel = _renderer.getActiveMipmapLevel();

    auto& _state = _renderer.state;

    // Set GL state for depth map.
    _state.setBlending(NoBlending);
    _state.colorBuffer.setClear(1, 1, 1, 1);
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

                _viewportSize.x = std::floor((float) _maxTextureSize / shadowFrameExtents.x);
                _shadowMapSize.x = _viewportSize.x * shadowFrameExtents.x;
                shadow->mapSize.x = _viewportSize.x;
            }

            if (_shadowMapSize.y > _maxTextureSize) {

                _viewportSize.y = std::floor((float) _maxTextureSize / shadowFrameExtents.y);
                _shadowMapSize.y = _viewportSize.y * shadowFrameExtents.y;
                shadow->mapSize.y = _viewportSize.y;
            }
        }

        if (!shadow->map && !std::dynamic_pointer_cast<PointLightShadow>(shadow) && this->type == VSMShadowMap) {

            GLRenderTarget::Options pars{};
            pars.minFilter = LinearFilter;
            pars.magFilter = LinearFilter;
            pars.format = RGBAFormat;

            shadow->map = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), pars);
            shadow->map->texture->name = light->name + ".shadowMap";

            shadow->mapPass = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), pars);

            shadow->camera->updateProjectionMatrix();
        }

        if (!shadow->map) {

            GLRenderTarget::Options pars{};
            pars.minFilter = NearestFilter;
            pars.magFilter = NearestFilter;
            pars.format = RGBAFormat;

            shadow->map = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), pars);
            shadow->map->texture->name = light->name + ".shadowMap";

            shadow->camera->updateProjectionMatrix();
        }

        _renderer.setRenderTarget(shadow->map);
        _renderer.clear();

        auto viewportCount = shadow->getViewportCount();

        for (unsigned vp = 0; vp < viewportCount; vp++) {

            const auto& viewport = shadow->getViewport(vp);

            _viewport.set(
                    _viewportSize.x * viewport.x,
                    _viewportSize.y * viewport.y,
                    _viewportSize.x * viewport.z,
                    _viewportSize.y * viewport.w);

            _state.viewport(_viewport);

            if (std::dynamic_pointer_cast<PointLightShadow>(shadow)) {
                std::dynamic_pointer_cast<PointLightShadow>(shadow)->updateMatrices(light->as<PointLight>(), vp);
            } else {
                shadow->updateMatrices(light);
            }

            _frustum = shadow->getFrustum();

            renderObject(_renderer, scene, camera, shadow->camera.get(), light);
        }

        // do blur pass for VSM

        if (!std::dynamic_pointer_cast<PointLightShadow>(shadow) && this->type == VSMShadowMap) {

            VSMPass(_renderer, shadow.get(), camera);
        }

        shadow->needsUpdate = false;
    }

    needsUpdate = false;

    _renderer.setRenderTarget(currentRenderTarget, activeCubeFace, activeMipmapLevel);
}

void GLShadowMap::VSMPass(GLRenderer& _renderer, LightShadow* shadow, Camera* camera) {

    const auto& geometry = _objects.update(fullScreenMesh.get());

    // vertical pass

    shadowMaterialVertical->uniforms->operator[]("shadow_pass").setValue(shadow->map->texture);
    shadowMaterialVertical->uniforms->operator[]("resolution").value<Vector2>().copy(shadow->mapSize);
    shadowMaterialVertical->uniforms->operator[]("radius").value<float>() = shadow->radius;
    _renderer.setRenderTarget(shadow->mapPass);
    _renderer.clear();
    _renderer.renderBufferDirect(camera, nullptr, geometry, shadowMaterialVertical.get(), fullScreenMesh.get(), std::nullopt);

    // horizontal pass

    shadowMaterialHorizontal->uniforms->operator[]("shadow_pass").setValue(shadow->mapPass->texture);
    shadowMaterialHorizontal->uniforms->operator[]("resolution").value<Vector2>().copy(shadow->mapSize);
    shadowMaterialHorizontal->uniforms->operator[]("radius").value<float>() = shadow->radius;
    _renderer.setRenderTarget(shadow->map);
    _renderer.clear();
    _renderer.renderBufferDirect(camera, nullptr, geometry, shadowMaterialHorizontal.get(), fullScreenMesh.get(), std::nullopt);
}

MeshDepthMaterial* GLShadowMap::getDepthMaterialVariant(bool useMorphing) {

    int index = useMorphing << 0;

    if (index >= _depthMaterials.size()) {

        auto material = MeshDepthMaterial::create();
        material->depthPacking = RGBADepthPacking;

        _depthMaterials.emplace_back(material);

        return material.get();
    } else {

        return _depthMaterials[index].get();
    }
}

MeshDistanceMaterial* GLShadowMap::getDistanceMaterialVariant(bool useMorphing) {

    int index = useMorphing << 0;

    if (index >= _distanceMaterials.size()) {

        auto material = MeshDistanceMaterial::create();

        _distanceMaterials.emplace_back(material);

        return material.get();

    } else {

        return _distanceMaterials[index].get();
    }
}

Material* GLShadowMap::getDepthMaterial(GLRenderer& _renderer, Object3D* object, BufferGeometry* geometry, Material* material, Light* light, float shadowCameraNear, float shadowCameraFar) {

    Material* result = nullptr;

    if (light->is<PointLight>()) {

        result = getDistanceMaterialVariant(false);

    } else {

        result = getDepthMaterialVariant(false);
    }

    if (_renderer.localClippingEnabled && material->clipShadows && !material->clippingPlanes.empty()) {

        // in this case we need a unique material instance reflecting the
        // appropriate state

        auto keyA = result->uuid(), keyB = material->uuid();

        auto materialsForVariant = _materialCache[keyA];

        auto cachedMaterial = materialsForVariant[keyB];

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


    if (type == VSMShadowMap) {

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

    auto distanceMaterial = result->as<MeshDistanceMaterial>();
    if (light->is<PointLight>() && distanceMaterial) {

        distanceMaterial->referencePosition.setFromMatrixPosition(*light->matrixWorld);
        distanceMaterial->nearDistance = shadowCameraNear;
        distanceMaterial->farDistance = shadowCameraFar;
    }

    return result;
}

void GLShadowMap::renderObject(GLRenderer& _renderer, Object3D* object, Camera* camera, Camera* shadowCamera, Light* light) {

    if (!object->visible) return;

    bool visible = object->layers.test(camera->layers);

    if (visible && (object->is<Mesh>() || object->is<Line>() || object->is<Points>())) {

        if ((object->castShadow || (object->receiveShadow && type == VSMShadowMap)) && (!object->frustumCulled || _frustum.intersectsObject(*object))) {

            object->modelViewMatrix.multiplyMatrices(shadowCamera->matrixWorldInverse, *object->matrixWorld);

            const auto geometry = _objects.update(object);
            const auto material = object->materials();

            if (material.size() > 1) {

                const auto& groups = geometry->groups;

                for (const auto& group : groups) {

                    if (material.size() > group.materialIndex) {
                        const auto groupMaterial = material[group.materialIndex];

                        if (groupMaterial && groupMaterial->visible) {

                            const auto depthMaterial = getDepthMaterial(_renderer, object, geometry, groupMaterial, light, shadowCamera->near, shadowCamera->far);

                            _renderer.renderBufferDirect(shadowCamera, nullptr, geometry, depthMaterial, object, group);
                        }
                    }
                }

            } else if (material.front()->visible) {

                auto depthMaterial = getDepthMaterial(_renderer, object, geometry, material.front(), light, shadowCamera->near, shadowCamera->far);

                _renderer.renderBufferDirect(shadowCamera, nullptr, geometry, depthMaterial, object, std::nullopt);
            }
        }
    }

    for (auto& child : object->children) {

        renderObject(_renderer, child.get(), camera, shadowCamera, light);
    }
}
