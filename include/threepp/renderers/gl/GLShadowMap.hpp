// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLShadowMap.js

#ifndef THREEPP_GLSHADOWMAP_HPP
#define THREEPP_GLSHADOWMAP_HPP

#include "GLObjects.hpp"

#include "threepp/math/Frustum.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/scenes/Scene.hpp"

namespace threepp {

    class GLRenderer;

    namespace gl {

        struct GLShadowMap {

            bool enabled = false;

            bool autoUpdate = true;
            bool needsUpdate = false;

            int type = PCFShadowMap;

            explicit GLShadowMap(GLObjects& _objects);

            void render(GLRenderer& _renderer, const std::vector<Light*>& lights, Scene* scene, Camera* camera);

        private:
            GLObjects& _objects;

            Frustum _frustum;

            Vector2 _shadowMapSize;
            Vector2 _viewportSize;

            Vector4 _viewport;

            std::vector<std::shared_ptr<MeshDepthMaterial>> _depthMaterials;
            std::vector<std::shared_ptr<MeshDistanceMaterial>> _distanceMaterials;

            std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Material>>> _materialCache;

            int _maxTextureSize;

            std::shared_ptr<Mesh> fullScreenMesh;

            MeshDepthMaterial* getDepthMaterialVariant(bool useMorphing);

            MeshDistanceMaterial* getDistanceMaterialVariant(bool useMorphing);

            Material* getDepthMaterial(GLRenderer& _renderer, Object3D* object, BufferGeometry* geometry, Material* material, Light* light, float shadowCameraNear, float shadowCameraFar);

            void VSMPass(GLRenderer& _renderer, LightShadow* shadow, Camera* camera);

            void renderObject(GLRenderer& _renderer, Object3D* object, Camera* camera, Camera* shadowCamera, Light* light);
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLSHADOWMAP_HPP
