
#include "threepp/renderers/gl/GLShadowMap.hpp"

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

    //    if (!enabled) return;
    //    if (!autoUpdate && !needsUpdate) return;
    //
    //    if (lights.empty()) return;
    //
    //    auto currentRenderTarget = _renderer.getRenderTarget();
    //    auto activeCubeFace = _renderer.getActiveCubeFace();
    //    auto activeMipmapLevel = _renderer.getActiveMipmapLevel();
    //
    //    auto _state = _renderer.state;
    //
    //    // Set GL state for depth map.
    //    _state.setBlending(NoBlending);
    //    _state.colorBuffer.setClear(1, 1, 1, 1);
    //    _state.depthBuffer.setTest(true);
    //    _state.setScissorTest(false);
    //
    //    // render depth map
    //
    //    for (auto &light : lights) {
    //
    //        auto lightWithShadow = dynamic_cast<LightWithShadow *>(light);
    //
    //        if (!lightWithShadow) {
    //
    //            std::cerr << "THREE.GLShadowMap:'" << light->type() << "'has no shadow." << std::endl;
    //            continue;
    //        }
    //
    //        auto shadow = lightWithShadow->getLightShadow();
    //
    //        if (shadow->autoUpdate == false && shadow->needsUpdate == false) continue;
    //
    //        _shadowMapSize.copy(shadow->mapSize);
    //
    //        auto shadowFrameExtents = shadow->getFrameExtents();
    //
    //        _shadowMapSize.multiply(shadowFrameExtents);
    //
    //        _viewportSize.copy(shadow->mapSize);
    //
    //        if (_shadowMapSize.x > _maxTextureSize || _shadowMapSize.y > _maxTextureSize) {
    //
    //            if (_shadowMapSize.x > _maxTextureSize) {
    //
    //                _viewportSize.x = std::floor((float) _maxTextureSize / shadowFrameExtents.x);
    //                _shadowMapSize.x = _viewportSize.x * shadowFrameExtents.x;
    //                shadow->mapSize.x = _viewportSize.x;
    //            }
    //
    //            if (_shadowMapSize.y > _maxTextureSize) {
    //
    //                _viewportSize.y = std::floor((float) _maxTextureSize / shadowFrameExtents.y);
    //                _shadowMapSize.y = _viewportSize.y * shadowFrameExtents.y;
    //                shadow->mapSize.y = _viewportSize.y;
    //            }
    //        }
    //
    //        if (!shadow->map && ! instanceof <PointLightShadow>(shadow) && this->type == VSMShadowMap) {
    //
    //            GLRenderTarget::Options pars{};
    //            pars.minFilter = LinearFilter;
    //            pars.magFilter = LinearFilter;
    //            pars.format = RGBAFormat;
    //
    //            shadow->map = GLRenderTarget::create(_shadowMapSize.x, _shadowMapSize.y, pars);
    //            shadow->map->texture.name = light->name + ".shadowMap";
    //
    //            shadow->mapPass = GLRenderTarget::create(_shadowMapSize.x, _shadowMapSize.y, pars);
    //
    //            shadow->camera->updateProjectionMatrix();
    //        }
    //
    //        if (!shadow->map) {
    //
    //            GLRenderTarget::Options pars{};
    //            pars.minFilter = NearestFilter;
    //            pars.magFilter = NearestFilter;
    //            pars.format = RGBAFormat;
    //
    //            shadow->map = GLRenderTarget::create(_shadowMapSize.x, _shadowMapSize.y, pars);
    //            shadow->map->texture.name = light->name + ".shadowMap";
    //
    //            shadow->camera->updateProjectionMatrix();
    //        }
    //
    //        _renderer.setRenderTarget(shadow->map);
    //        _renderer.clear();
    //
    //        auto viewportCount = shadow->getViewportCount();
    //
    //        for (int vp = 0; vp < viewportCount; vp++) {
    //
    //            auto &viewport = shadow->getViewport(vp);
    //
    //            _viewport.set(
    //                    _viewportSize.x * viewport.x,
    //                    _viewportSize.y * viewport.y,
    //                    _viewportSize.x * viewport.z,
    //                    _viewportSize.y * viewport.w);
    //
    //            _state.viewport(_viewport);
    //
    //            shadow->updateMatrices(*light, vp);
    //
    //            _frustum = shadow->getFrustum();
    //
    //            renderObject(scene, camera, shadow->camera, light, this->type);
    //        }
    //
    //        // do blur pass for VSM
    //
    //        if (! instanceof <PointLightShadow>(shadow) && this->type == VSMShadowMap) {
    //
    //            VSMPass(_renderer, shadow, camera);
    //        }
    //
    //        shadow->needsUpdate = false;
    //    }
    //
    //    needsUpdate = false;
    //
    //    _renderer.setRenderTarget(currentRenderTarget, activeCubeFace, activeMipmapLevel);
}

void GLShadowMap::VSMPass(GLRenderer& _renderer, LightShadow* shadow, Camera* camera) {

    //    const auto &geometry = _objects.update(fullScreenMesh);
    //
    //    // vertical pass
    //
    //    shadowMaterialVertical->uniforms->operator[]("shadow_pass").setValue(shadow->map->texture);
    //    shadowMaterialVertical->uniforms->operator[]("resolution").value<Vector2>().copy(shadow->mapSize);
    //    shadowMaterialVertical->uniforms->operator[]("radius").value<float>() = shadow->radius;
    //    _renderer.setRenderTarget(shadow->mapPass);
    //    _renderer.clear();
    //    _renderer.renderBufferDirect(camera, nullptr, geometry, shadowMaterialVertical, fullScreenMesh, std::nullopt);
    //
    //    // horizontal pass
    //
    //    shadowMaterialHorizontal->uniforms->operator[]("shadow_pass").setValue(shadow->mapPass->texture);
    //    shadowMaterialHorizontal->uniforms->operator[]("resolution").value<Vector2>().copy(shadow->mapSize);
    //    shadowMaterialHorizontal->uniforms->operator[]("radius").value<float>() = shadow->radius;
    //    _renderer.setRenderTarget(shadow->map);
    //    _renderer.clear();
    //    _renderer.renderBufferDirect(camera, nullptr, geometry, shadowMaterialHorizontal, fullScreenMesh, std::nullopt);
}
