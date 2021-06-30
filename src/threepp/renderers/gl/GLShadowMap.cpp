
#include "GLShadowMap.hpp"

#include "threepp/renderers/GLRenderer.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    inline std::unordered_map<int, int> shadowSide {
            {0, BackSide},
            {1, FrontSide},
            {2, DoubleSide}
    };

}

void GLShadowMap::render(GLRenderer &_renderer, const std::vector<Light *> &lights, Scene *scene, Camera *camera) {

    if ( enabled == false ) return;
    if ( autoUpdate == false && needsUpdate == false ) return;

    if ( lights.empty() ) return;

    auto currentRenderTarget = _renderer.getRenderTarget();
    auto activeCubeFace = _renderer.getActiveCubeFace();
    auto activeMipmapLevel = _renderer.getActiveMipmapLevel();

//    auto _state = _renderer.state;
//
//    // Set GL state for depth map.
//    _state.setBlending( NoBlending );
//    _state.buffers.color.setClear( 1, 1, 1, 1 );
//    _state.buffers.depth.setTest( true );
//    _state.setScissorTest( false );
//
//    // render depth map
//
//    for ( int i = 0, il = lights.size(); i < il; i ++ ) {
//
//        auto& light = lights[ i ];
//        const shadow = light.shadow;
//
//        if ( shadow === undefined ) {
//
//            console.warn( 'THREE.WebGLShadowMap:', light, 'has no shadow.' );
//            continue;
//
//        }
//
//        if ( shadow.autoUpdate == false && shadow.needsUpdate == false ) continue;
//
//        _shadowMapSize.copy( shadow.mapSize );
//
//        auto shadowFrameExtents = shadow.getFrameExtents();
//
//        _shadowMapSize.multiply( shadowFrameExtents );
//
//        _viewportSize.copy( shadow.mapSize );
//
//        if ( _shadowMapSize.x > _maxTextureSize || _shadowMapSize.y > _maxTextureSize ) {
//
//            if ( _shadowMapSize.x > _maxTextureSize ) {
//
//                _viewportSize.x = std::floor( _maxTextureSize / shadowFrameExtents.x );
//                _shadowMapSize.x = _viewportSize.x * shadowFrameExtents.x;
//                shadow.mapSize.x = _viewportSize.x;
//
//            }
//
//            if ( _shadowMapSize.y > _maxTextureSize ) {
//
//                _viewportSize.y = std::floor( _maxTextureSize / shadowFrameExtents.y );
//                _shadowMapSize.y = _viewportSize.y * shadowFrameExtents.y;
//                shadow.mapSize.y = _viewportSize.y;
//
//            }
//
//        }
//
//        if ( shadow.map === null && ! shadow.isPointLightShadow && this.type === VSMShadowMap ) {
//
//            std::unordered_map<std::string, int> pars {
//                    {"minFilter", LinearFilter},
//                    {"magFilter", LinearFilter},
//                    {"format", RGBAFormat}
//            };
//
//            shadow.map = GLRenderTarget( _shadowMapSize.x, _shadowMapSize.y, pars );
//            shadow.map.texture.name = light->name + ".shadowMap";
//
//            shadow.mapPass = GLRenderTarget( _shadowMapSize.x, _shadowMapSize.y, pars );
//
//            shadow.camera.updateProjectionMatrix();
//
//        }
//
//        if ( shadow.map == nullptr ) {
//
//            std::unordered_map<std::string, int> pars {
//                    {"minFilter", NearestFilter},
//                    {"magFilter", NearestFilter},
//                    {"format", RGBAFormat}
//            };
//
//            shadow.map = GLRenderTarget( _shadowMapSize.x, _shadowMapSize.y, pars );
//            shadow.map.texture.name = light->name + ".shadowMap";
//
//            shadow.camera.updateProjectionMatrix();
//
//        }
//
//        _renderer.setRenderTarget( shadow.map );
//        _renderer.clear();
//
//        auto viewportCount = shadow.getViewportCount();
//
//        for ( int vp = 0; vp < viewportCount; vp ++ ) {
//
//            auto viewport = shadow.getViewport( vp );
//
//            _viewport.set(
//                    _viewportSize.x * viewport.x,
//                    _viewportSize.y * viewport.y,
//                    _viewportSize.x * viewport.z,
//                    _viewportSize.y * viewport.w
//            );
//
//            _state.viewport( _viewport );
//
//            shadow.updateMatrices( light, vp );
//
//            _frustum = shadow.getFrustum();
//
//            renderObject( scene, camera, shadow.camera, light, this.type );
//
//        }
//
//        // do blur pass for VSM
//
//        if ( ! shadow.isPointLightShadow && this.type === VSMShadowMap ) {
//
//            VSMPass( shadow, camera );
//
//        }
//
//        shadow.needsUpdate = false;
//
//    }
//
//    needsUpdate = false;
//
//    _renderer.setRenderTarget( currentRenderTarget, activeCubeFace, activeMipmapLevel );

}
