
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


#include <cmath>
#include <iostream>

using namespace threepp;
using namespace threepp::gl;

namespace {

    inline std::unordered_map<Side, Side> shadowSide{
            {Side::Front, Side::Back},
            {Side::Back, Side::Front},
            {Side::Double, Side::Double}};


}// namespace

struct GLShadowMap::Impl {

    GLShadowMap* scope;
    GLObjects& _objects;

    const Frustum* _frustum;

    Vector2 _shadowMapSize;
    Vector2 _viewportSize;

    Vector4 _viewport;

    std::vector<std::shared_ptr<MeshDepthMaterial>> _depthMaterials;
    std::vector<std::shared_ptr<MeshDistanceMaterial>> _distanceMaterials;

    std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Material>>> _materialCache;

    int _maxTextureSize;

    std::shared_ptr<Mesh> fullScreenMesh;

    Impl(GLShadowMap* scope, GLObjects& objects)
        : scope(scope),
          _objects(objects),
          _frustum(nullptr),
          _maxTextureSize(GLCapabilities::instance().maxTextureSize) {

        auto fullScreenTri = BufferGeometry::create();
        fullScreenTri->setAttribute("position", FloatBufferAttribute::create({-1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5}, 3));

        fullScreenMesh = Mesh::create(fullScreenTri, shadowMaterialVertical);
    }

    void VSMPass(GLRenderer& _renderer, LightShadow* shadow, Camera* camera) {
        const auto& geometry = _objects.update(fullScreenMesh.get());

        // vertical pass

        shadowMaterialVertical->uniforms.at("shadow_pass").setValue(shadow->map->texture.get());
        shadowMaterialVertical->uniforms.at("resolution").value<Vector2>().copy(shadow->mapSize);
        shadowMaterialVertical->uniforms.at("radius").value<float>() = shadow->radius;
        _renderer.setRenderTarget(shadow->mapPass.get());
        _renderer.clear();
        _renderer.renderBufferDirect(camera, nullptr, geometry, shadowMaterialVertical.get(), fullScreenMesh.get(), std::nullopt);

        // horizontal pass

        shadowMaterialHorizontal->uniforms.at("shadow_pass").setValue(shadow->mapPass->texture.get());
        shadowMaterialHorizontal->uniforms.at("resolution").value<Vector2>().copy(shadow->mapSize);
        shadowMaterialHorizontal->uniforms.at("radius").value<float>() = shadow->radius;
        _renderer.setRenderTarget(shadow->map.get());
        _renderer.clear();
        _renderer.renderBufferDirect(camera, nullptr, geometry, shadowMaterialHorizontal.get(), fullScreenMesh.get(), std::nullopt);
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

            result = getDistanceMaterialVariant(false);

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

        // Set GL state for depth map.
        _state.setBlending(Blending::None);
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

            if (!shadow->map && !std::dynamic_pointer_cast<PointLightShadow>(shadow) && scope->type == ShadowMap::VSM) {

                GLRenderTarget::Options pars{};
                pars.minFilter = Filter::Linear;
                pars.magFilter = Filter::Linear;
                pars.format = Format::RGBA;

                shadow->map = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), pars);
                shadow->map->texture->name = light->name + ".shadowMap";

                shadow->mapPass = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), pars);

                shadow->camera->updateProjectionMatrix();
            }

            if (!shadow->map) {

                GLRenderTarget::Options pars{};
                pars.minFilter = Filter::Nearest;
                pars.magFilter = Filter::Nearest;
                pars.format = Format::RGBA;

                shadow->map = GLRenderTarget::create(static_cast<int>(_shadowMapSize.x), static_cast<int>(_shadowMapSize.y), pars);
                shadow->map->texture->name = light->name + ".shadowMap";

                shadow->camera->updateProjectionMatrix();
            }

            _renderer.setRenderTarget(shadow->map.get());
            _renderer.clear();

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

GLShadowMap::GLShadowMap(GLObjects& objects)
    : type(ShadowMap::PFC), pimpl_(std::make_unique<Impl>(this, objects)) {}


void GLShadowMap::render(GLRenderer& renderer, const std::vector<Light*>& lights, Object3D* scene, Camera* camera) {

    pimpl_->render(renderer, lights, scene, camera);
}

gl::GLShadowMap::~GLShadowMap() = default;
